"""events.store -- decision, outcome and rejection event log (design 5.10; the field-level contract is docs/engineering_handover.html).

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

A third stream records the shelf-hours that reached us and were NOT
priced (`rejections`, contract.REJECTION_REQUIRED). A rejection is not a
decision: it holds no price, never enters `priced_hours`, the pairing or
the evidence. It exists so that a refused hour is distinguishable from an
hour engineering never sent -- `last_seen_by_shelf` spans decisions AND
rejections, so the live episode-id rule can step from the hour before even
when we declined to price it, and a gap in that index means a genuinely
missing hour.
"""

import fcntl
import json
import os
from contextlib import contextmanager

from events.pairs import hour_key
# the contract -- the required fields and the value checks -- is
# events.contract; the names stay here for callers
from events.contract import (DECISION_OPTIONAL, DECISION_REQUIRED, REJECTION_REQUIRED, _json_scalar, _validate_decision, _validate_rejection)


def _quarantine_key(evt):
    """Identity of a quarantined event, or None if it carries no id. Both
    kinds share one quarantine file, so the kind is part of the key --
    otherwise an id collision silently swallows the second event. An
    unparseable line is keyed on its own text (`raw_line`)."""
    for kind in ("outcome", "decision", "rejection"):
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


def _upsert_shelf(index, key, evt, **extra):
    """(sku, fc) -> the latest hour's record in `index`: the later hour
    wins (a tie keeps the later write). The one upsert both shelf indexes
    share; `extra` is what distinguishes them."""
    shelf, when = key[:2], (key[2], key[3])
    held = index.get(shelf)
    if held is None or (held["date"], held["hour_of_day"]) <= when:
        index[shelf] = {"episode_id": evt.get("episode_id"), "date": key[2],
                        "hour_of_day": key[3],
                        "hours_remaining": evt.get("hours_remaining"),
                        "q_remaining": evt.get("q_remaining"), **extra}


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


# the three event streams, in the order they are consumed: decisions
# first, so a rejection beside a priced hour never wins the seen index
# THIS FOLDER: two streams. Outcomes are built by the owner's lane, in the
# repository, from the feed; this store holds what the hour priced and what
# it refused, and nothing else reads or writes it here.
STREAMS = ("decision", "rejection")


class EventStore:
    def __init__(self, cfg, root=None):
        self.cfg = cfg
        self.root = root or cfg["events"]["store_dir"]
        os.makedirs(self.root, exist_ok=True)
        self.paths = {k: os.path.join(self.root, f"{k}.jsonl")
                      for k in ("decisions", "outcomes", "rejections", "quarantine")}
        # duplicates seen by THIS store: on emit, and -- because a foreign
        # producer may write the JSONL directly -- while loading. Either way
        # the same id twice is what the duplicate gate exists to catch.
        self.duplicate_counts = {"decision": 0, "outcome": 0, "rejection": 0}
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
        with self._locked():
            for i, parsed, raw in self._lines(self.paths["quarantine"]):
                if parsed is None:
                    continue              # a torn quarantine line: skipped
                # a foreign writer's record may carry anything under `event`
                evt = parsed.get("event")
                key = _quarantine_key(evt) if isinstance(evt, dict) else None
                if key is not None:
                    self._quarantined_ids.add(key)
            self._terminate_last_line(self.paths["quarantine"])

        self._ids = {"decision": set(), "outcome": set(), "rejection": set()}
        self._hour_keys = set()          # every hour a stored decision priced
        self._decisions_with_outcome = set()
        self.episode_paths = {}          # episode_id -> the latest decision's path
        # (sku, fc) -> the latest decision on that shelf: the anchor
        # ops.price_hour falls back to, and the episode it continues
        self.latest_by_shelf = {}
        # (sku, fc) -> the latest hour SEEN on that shelf, priced or refused:
        # what the live episode-id rule steps from, so a hold through a
        # rejection is not read as a gap
        self.last_seen_by_shelf = {}
        # how far into each stream THIS store has read: every emit re-reads
        # the tail past it under the lock before it checks the invariants,
        # so a line another process appended since construction is seen
        self._offsets = {k: 0 for k in STREAMS}
        self._line_counts = {k: 0 for k in STREAMS}
        # why the last emit was refused (a caller's response line names it)
        self.last_refusal = None
        with self._locked():
            for kind in STREAMS:
                self._consume(kind)

    @contextmanager
    def _locked(self):
        """The one lock every writer and every loader takes: an exclusive
        flock on `<root>/.lock`. Two hourly runs that overlap (a retry
        launched while the first still runs) each hold an in-memory index
        built at construction; without this, both saw the hour unpriced,
        both appended, and -- the decision id being the shelf-hour -- the
        second line was silently dropped on the next load while
        engineering may have applied the second price. Under the lock the
        second run re-reads the first's line and refuses the hour, loudly
        (`decisions_on_priced_hour`)."""
        with open(os.path.join(self.root, ".lock"), "w") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)

    def _consume(self, kind):
        """Register every complete line of `kind`'s stream past the offset
        this store has consumed, and move the offset. Called under the
        lock: at construction (from zero) and before every emit (the tail
        another process may have appended since). A torn last line -- a
        write that died mid-append -- is quarantined with the reason,
        closed with a newline so the next append cannot glue onto it, and
        consumed."""
        path = self.paths[kind + "s"]
        if not os.path.exists(path):
            return
        torn, partial = [], False
        with open(path, "rb") as f:
            f.seek(self._offsets[kind])
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    partial = True
                    self._line_counts[kind] += 1
                    torn.append((self._line_counts[kind],
                                 line.decode(errors="replace")))
                    break
                self._line_counts[kind] += 1
                # a torn line can split a multi-byte character: decoded with
                # a replacement mark it is one more unparseable line to
                # quarantine, never a UnicodeDecodeError
                raw = line.decode(errors="replace").rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("not a JSON object")
                except ValueError:
                    torn.append((self._line_counts[kind], raw))
                    continue
                self._register_line(kind, parsed)
            self._offsets[kind] = f.tell()
        if partial:
            self._terminate_last_line(path)
            self._offsets[kind] = os.path.getsize(path)
        for i, raw in torn:
            # a partial write (power loss mid-append) must not make the
            # store unconstructable: the line is quarantined with the
            # reason and the stream stays readable
            self._quarantine(
                {"stream": f"{kind}s", "line_no": i, "raw_line": raw},
                [f"unparseable JSONL line {i} in {kind}s.jsonl (torn "
                 "write?) -- skipped on load"])

    def _register_line(self, kind, parsed):
        """One parsed line of a stream entering the indexes (on load or on
        a tail refresh): a line with no id is not an event; a repeated id
        is counted and not registered twice."""
        ident = parsed.get(f"{kind}_id")
        if ident is None:
            # a foreign line with no id: not an event the matcher can
            # pair, and registering None counted every later id-less line
            # as a duplicate of it
            return
        if ident in self._ids[kind]:
            self.duplicate_counts[kind] += 1
            return
        self._ids[kind].add(ident)
        if kind == "decision":
            self._register_decision(parsed)
        elif kind == "rejection":
            # decisions are consumed first: an hour that was ultimately
            # priced stays priced in the index, whatever a foreign writer
            # left in this stream beside it
            if _decision_key(parsed) not in self._hour_keys:
                self._register_seen(parsed, priced=False)

    def _register_seen(self, evt, priced):
        """`last_seen_by_shelf`: the latest hour this shelf reached us at,
        whether it was priced or refused. The live episode-id rule steps
        from it, so an hour we declined to price is no longer a gap -- and a
        gap that remains is an hour engineering did not send."""
        key = _decision_key(evt)
        if key is not None:
            _upsert_shelf(self.last_seen_by_shelf, key, evt, priced=priced)
        return key

    def _register_decision(self, evt):
        """The hour and the episode path of a decision entering the record
        (on load: a collision is counted, the line stays)."""
        key = self._register_seen(evt, priced=True)
        if key is not None:
            if key in self._hour_keys:
                self.completeness_counts["decisions_on_priced_hour"] += 1
            self._hour_keys.add(key)
            _upsert_shelf(self.latest_by_shelf, key, evt,
                          applied_discount=evt.get("applied_discount"))
        path = _episode_path(evt)
        if path is not None:
            # the hour the episode OPENED survives every later decision: a
            # restock extension is predicted on features as of that hour
            prev = self.episode_paths.get(evt["episode_id"])
            path["opened"] = (prev["opened"] if prev
                              else (path["date"], path["hour_of_day"]))
            self.episode_paths[evt["episode_id"]] = path

    @property
    def priced_hours(self):
        """The hour keys the store holds a decision for (read-only)."""
        return self._hour_keys

        # (the lock file stays: another process may be holding it)

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

    def _write(self, kind, evt):
        """Append one event to its stream and move this store's offset past
        it, so the next tail refresh does not read our own line back as a
        duplicate. Under the lock, so the file's end IS our line."""
        path = self.paths[kind + "s"]
        self._append(path, evt)
        self._offsets[kind] = os.path.getsize(path)
        self._line_counts[kind] += 1

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

    def _refresh(self):
        for kind in STREAMS:
            self._consume(kind)

    def emit_decision(self, evt):
        with self._locked():
            self._refresh()
            return self._emit_decision(evt)

    def emit_rejection(self, evt):
        with self._locked():
            self._refresh()
            return self._emit_rejection(evt)

    def _emit_decision(self, evt):
        self.last_refusal = None
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
            self.last_refusal = "quarantined: " + "; ".join(problems)
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
            self.last_refusal = "already_priced: another run committed this hour first"
            return False
        if evt["decision_id"] in self._ids["decision"]:
            # the backstop: an id already held whose hour this store does
            # not index (a foreign line replayed in, a hand-built event)
            self.duplicate_counts["decision"] += 1
            self.last_refusal = "duplicate decision id"
            return False
        # append FIRST: an id registered before a failed write (disk full)
        # would refuse the retry as a duplicate of an event never written
        self._write("decision", evt)
        self._ids["decision"].add(evt["decision_id"])
        self._register_decision(evt)
        return True

    def _emit_rejection(self, evt):
        """Record a shelf-hour that reached us and was NOT priced. It never
        enters `priced_hours`, so the hour stays free to price if the data
        that refused it is corrected; it only says the shelf was SEEN. An
        hour this store already holds a DECISION for is not recorded at
        all -- that hour was priced, and a rejection beside it would make
        `last_seen` ambiguous."""
        missing = [f for f in REJECTION_REQUIRED if f not in evt]
        problems = ([f"missing fields: {missing}"] if missing
                    else _validate_rejection(evt))
        if problems:
            self._quarantine(evt, problems)
            return False
        if evt["rejection_id"] in self._ids["rejection"]:
            self.duplicate_counts["rejection"] += 1
            return False
        if _decision_key(evt) in self._hour_keys:
            return False
        self._write("rejection", evt)                 # append first, as above
        self._ids["rejection"].add(evt["rejection_id"])
        self._register_seen(evt, priced=False)
        return True
