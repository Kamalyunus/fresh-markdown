"""events.store -- decision and outcome event log (design 5.10; the field-level contract is docs/engineering_handover.html).

Append-only JSONL with duplicate detection, durable writes, and replay. The
logger never silently discards a malformed event: invalid events are
quarantined with the validation failure attached and surfaced to monitoring.

Two invariants live HERE and nowhere else, because every caller once
re-derived them and two concurrent batches could both pass a caller's
check: ONE decision per hour (the hour key -- a second decision for a
priced hour is refused and counted, so a retried batch never lands two
prices on one feed row) and ONE outcome per decision (a second outcome
naming the same decision is refused on emit and skipped on load, so the
learner can never consume two outcomes for one hour -- the outcome-id
migration and every future double-outcome path close on this). The
store also keeps, per episode, the latest decision's stored forecast
(`episode_paths`), which a later request of the same episode is priced
on (engine.state.build_states slices it, design 5.10).
"""

import json
import os

from events.pairs import hour_key
# the contract -- the required fields and the value checks -- is
# events.contract; the names stay here for callers
from events.contract import (DECISION_OPTIONAL, DECISION_REQUIRED, OUTCOME_REQUIRED, ISO_DAY,      # noqa: F401
                             finite_number, _json_scalar, _is_iso_day,
                             _validate_decision, _validate_outcome)


def _quarantine_key(evt):
    """Identity of a quarantined event, or None if it carries no id. Both
    kinds share one quarantine file, so the kind is part of the key --
    otherwise an id collision silently swallows the second event. An
    unparseable line is keyed on its own text (`raw_line`)."""
    for kind in ("outcome", "decision"):
        ident = evt.get(f"{kind}_id")
        if ident is not None:
            return (kind, ident)
    if evt.get("raw_line") is not None:
        return ("raw", evt.get("stream"), evt["raw_line"])
    return None


def _decision_key(evt):
    """The hour key of a decision event, or None when it names no hour."""
    try:
        return hour_key(evt.get("sku_id"), evt.get("fc"), evt.get("date"),
                        evt.get("hour_of_day"))
    except (TypeError, ValueError):
        return None


def _episode_path(evt):
    """What a later request of this episode is priced on: the decision's
    hour and its stored forecast from that hour (design 5.10)."""
    path = evt.get("mu_ref_path")
    if not isinstance(path, list) or evt.get("episode_id") is None:
        return None
    try:
        out = {"date": str(evt["date"]), "hour_of_day": int(evt["hour_of_day"]),
               "hours_remaining": int(evt["hours_remaining"]),
               "mu_ref_path": [float(m) for m in path],
               # the features the forecast stood on (contract.DECISION_OPTIONAL):
               # None on a decision recorded before they were
               "features": None}
        if all(f in evt for f in DECISION_OPTIONAL):
            out["features"] = tuple(None if evt[f] is None else float(evt[f])
                                    for f in DECISION_OPTIONAL)
        return out
    except (KeyError, TypeError, ValueError):
        return None


class EventStore:
    def __init__(self, cfg, root=None, reset=False):
        self.cfg = cfg
        self.root = root or cfg["events"]["store_dir"]
        os.makedirs(self.root, exist_ok=True)
        self.paths = {k: os.path.join(self.root, f"{k}.jsonl")
                      for k in ("decisions", "outcomes", "quarantine")}
        if reset:
            self._reset()
        # duplicates seen by THIS store: on emit, and -- because a foreign
        # producer may write the JSONL directly -- while loading. Either way
        # the same id twice is what the duplicate gate exists to catch.
        self.duplicate_counts = {"decision": 0, "outcome": 0}
        # what the two invariants refused, on emit and on load (a foreign
        # writer): a second decision for a priced hour (counted; both stay
        # in the record, and ingest matches neither), a second outcome for
        # one decision (skipped on load), an outcome without is_stockout
        # (quarantined on emit, skipped on load). events.pairs.quality_counts
        # carries the last two into the monitor's safety block.
        self.completeness_counts = {"decisions_on_priced_hour": 0,
                                    "outcomes_per_decision_over_one": 0,
                                    "missing_stockout_field": 0}
        # quarantined by THIS run; load_quarantine returns the whole FILE
        # (every run ever). The shadow gate must read the per-run figure,
        # never cumulative state (docs/learnings.md).
        self.quarantined_this_run = 0
        # quarantine dedups too, by the same rule -- otherwise
        # quarantined_event_count grows on every re-run over the same store,
        # and the shadow gate reads it
        self._quarantined_ids = set()
        for i, parsed, raw in self._lines(self.paths["quarantine"]):
            if parsed is None:
                continue                  # a torn quarantine line: skipped
            # a foreign writer's record may carry anything under `event`
            evt = parsed.get("event")
            key = _quarantine_key(evt) if isinstance(evt, dict) else None
            if key is not None:
                self._quarantined_ids.add(key)
        self._terminate_last_line(self.paths["quarantine"])

        self._ids = {"decision": set(), "outcome": set()}
        self._hour_keys = set()          # every hour a stored decision priced
        self._decisions_with_outcome = set()
        self.episode_paths = {}          # episode_id -> the latest decision's path
        # (sku, fc) -> the latest decision on that shelf: what ops.price_hour
        # continues an episode from (the chain's window rule, live)
        self.latest_by_shelf = {}
        for kind, path in (("decision", self.paths["decisions"]),
                           ("outcome", self.paths["outcomes"])):
            torn = []
            for i, parsed, raw in self._lines(path):
                if parsed is None:
                    torn.append((i, raw))
                    continue
                ident = parsed.get(f"{kind}_id")
                if ident is None:
                    # a foreign line with no id: not an event the matcher
                    # can pair, and registering None counted every later
                    # id-less line as a duplicate of it
                    continue
                if ident in self._ids[kind]:
                    self.duplicate_counts[kind] += 1
                    continue
                self._ids[kind].add(ident)
                if kind == "decision":
                    self._register_decision(parsed)
                else:
                    self._register_outcome(parsed)
            for i, raw in torn:
                # a partial write (power loss mid-append) must not make the
                # store unconstructable: the line is quarantined with the
                # reason and the stream stays readable
                self._quarantine(
                    {"stream": f"{kind}s", "line_no": i, "raw_line": raw},
                    [f"unparseable JSONL line {i} in {kind}s.jsonl (torn "
                     "write?) -- skipped on load"])
            self._terminate_last_line(path)

    def _register_decision(self, evt):
        """The hour and the episode path of a decision entering the record
        (on load: a collision is counted, the line stays)."""
        key = _decision_key(evt)
        if key is not None:
            if key in self._hour_keys:
                self.completeness_counts["decisions_on_priced_hour"] += 1
            self._hour_keys.add(key)
            shelf, when = key[:2], (key[2], key[3])
            last = self.latest_by_shelf.get(shelf)
            if last is None or (last["date"], last["hour_of_day"]) <= when:
                self.latest_by_shelf[shelf] = {
                    "episode_id": evt.get("episode_id"), "date": key[2],
                    "hour_of_day": key[3],
                    "hours_remaining": evt.get("hours_remaining"),
                    "applied_discount": evt.get("applied_discount"),
                    "q_remaining": evt.get("q_remaining")}
        path = _episode_path(evt)
        if path is not None:
            # the hour the episode OPENED survives every later decision: a
            # restock extension is predicted on features as of that hour
            prev = self.episode_paths.get(evt["episode_id"])
            path["opened"] = (prev["opened"] if prev
                              else (path["date"], path["hour_of_day"]))
            self.episode_paths[evt["episode_id"]] = path

    def _register_outcome(self, evt):
        """On load: what `_load` will skip, counted once per store."""
        if "is_stockout" not in evt:
            self.completeness_counts["missing_stockout_field"] += 1
            return
        if evt.get("decision_id") in self._decisions_with_outcome:
            self.completeness_counts["outcomes_per_decision_over_one"] += 1
            return
        self._decisions_with_outcome.add(evt.get("decision_id"))

    @property
    def priced_hours(self):
        """The hour keys the store holds a decision for (read-only)."""
        return self._hour_keys

    def _reset(self):
        """Empty THIS store's own three streams, for a HARNESS whose run is a
        fresh evaluation: shadow re-run over the store an earlier run left
        re-prices the same shelf-hours, and since the decision id IS the
        shelf-hour (events.pairs.decision_id_of) every one of them collides
        with its own earlier copy. Before the ids were natural the re-run
        appended a second copy of everything instead, unnoticed -- the
        per-run counters (`quarantined_this_run`) were built to work around
        exactly that.

        Only the three files this class writes are removed, never the
        directory; and never the PRODUCTION store, which is append-only for
        the life of the pilot and is the one record `daily.update` learns
        from."""
        if os.path.abspath(self.root) == os.path.abspath(self.cfg["events"]["store_dir"]):
            raise ValueError(
                "refusing to reset the production event store "
                f"({self.root}): it is the append-only record the daily lane "
                "learns from. Only a harness store (shadow, a workspace copy) "
                "may be reset.")
        for path in self.paths.values():
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _lines(path):
        """(line_no, parsed or None, raw) per non-empty line."""
        if not os.path.exists(path):
            return
        # a torn last line can split a multi-byte character: decoded with a
        # replacement mark it is one more unparseable line to quarantine,
        # not a UnicodeDecodeError before the store exists
        with open(path, errors="replace") as f:
            for i, line in enumerate(f, start=1):
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("not a JSON object")
                except ValueError:
                    parsed = None
                yield i, parsed, raw

    @staticmethod
    def _terminate_last_line(path):
        """A torn last line has no newline; the next append would glue a good
        event onto it and lose both. Close the line first."""
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
                f.flush()
                os.fsync(f.fileno())

    def _append(self, path, evt):
        with open(path, "a") as f:
            f.write(json.dumps(evt, default=_json_scalar) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _quarantine(self, evt, problems):
        # idempotent like the good streams. An event with no usable id cannot
        # be deduped and is always appended -- better a double count on a
        # malformed event than a dropped record of one.
        key = _quarantine_key(evt)
        if key is not None:
            if key in self._quarantined_ids:
                return
            self._quarantined_ids.add(key)
        self.quarantined_this_run += 1
        self._append(self.paths["quarantine"], {"event": evt, "problems": problems})

    def emit_decision(self, evt):
        missing = [f for f in DECISION_REQUIRED if f not in evt]
        problems = ([f"missing fields: {missing}"] if missing
                    else _validate_decision(evt))
        key = None if problems else _decision_key(evt)
        if not problems and key is None:
            problems.append("sku_id, fc, date and hour_of_day must name one "
                            "hour of one item (the key ingest matches on); got "
                            f"{evt.get('sku_id')!r}, {evt.get('fc')!r}, "
                            f"{evt.get('date')!r}, {evt.get('hour_of_day')!r}")
        if problems:
            self._quarantine(evt, problems)
            return False
        # the hour gate runs FIRST and stays the reported signal. The
        # decision id is the shelf-hour (events.pairs.decision_id_of), so a
        # re-priced hour now trips both gates; read as a duplicate id it
        # would move the number the reports and the handover name
        if key in self._hour_keys:
            # the hour is already priced: a second decision would put two
            # prices on one feed row, and ingest would match neither
            self.completeness_counts["decisions_on_priced_hour"] += 1
            return False
        if evt["decision_id"] in self._ids["decision"]:
            # the backstop: an id already held whose hour this store does
            # not index (a foreign line replayed in, a hand-built event)
            self.duplicate_counts["decision"] += 1
            return False
        # append FIRST: an id registered before a failed write (disk full)
        # would refuse the retry as a duplicate of an event never written
        self._append(self.paths["decisions"], evt)
        self._ids["decision"].add(evt["decision_id"])
        self._register_decision(evt)
        return True

    def emit_outcome(self, evt):
        missing = [f for f in OUTCOME_REQUIRED if f not in evt]
        problems = ([f"missing fields: {missing}"] if missing
                    else _validate_outcome(evt))
        if problems:
            if "is_stockout" in missing:
                self.completeness_counts["missing_stockout_field"] += 1
            self._quarantine(evt, problems)
            return False
        if evt["outcome_id"] in self._ids["outcome"]:
            self.duplicate_counts["outcome"] += 1
            return False
        if evt["decision_id"] in self._decisions_with_outcome:
            # one outcome per decision: a second one (an id scheme that
            # moved, a re-ingest under another key) would be consumed by the
            # learner as a second hour of evidence
            self.completeness_counts["outcomes_per_decision_over_one"] += 1
            return False
        self._append(self.paths["outcomes"], evt)     # append first, as above
        self._ids["outcome"].add(evt["outcome_id"])
        self._decisions_with_outcome.add(evt["decision_id"])
        return True

    def _load(self, path, id_field=None, outcome=False):
        """The readable stream: torn lines were quarantined at construction
        and are skipped, as is a line carrying no id at all (not an event);
        a repeated id (a foreign producer's re-append) is COUNTED in
        duplicate_counts and loaded ONCE, first occurrence -- the learner
        must never consume the same outcome twice. An outcome stream also
        skips a second outcome for a decision already answered, and a line
        without `is_stockout` (counted at construction, `completeness_counts`)."""
        out, seen, answered = [], set(), set()
        for _, parsed, _ in self._lines(path):
            if parsed is None:
                continue
            ident = parsed.get(id_field) if id_field else None
            if id_field and ident is None:
                # a foreign line with no id: not an event -- the matcher and
                # the export index every event by its id and once died on it
                continue
            if ident is not None:
                if ident in seen:
                    continue
                seen.add(ident)
            if outcome:
                if "is_stockout" not in parsed:
                    continue
                if parsed.get("decision_id") in answered:
                    continue
                answered.add(parsed.get("decision_id"))
            out.append(parsed)
        return out

    def load_decisions(self):
        return self._load(self.paths["decisions"], "decision_id")

    def load_outcomes(self):
        return self._load(self.paths["outcomes"], "outcome_id", outcome=True)

    def load_quarantine(self):
        return self._load(self.paths["quarantine"])
