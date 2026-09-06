"""events.store -- decision and outcome event log (design 5.10; the field-level contract is docs/event_contract.html).

Append-only JSONL with duplicate detection, durable writes, and replay. The
logger never silently discards a malformed event: invalid events are
quarantined with the validation failure attached and surfaced to monitoring.
"""

import datetime
import json
import os
import re

import numpy as np

# the one finiteness test, shared with the state validation that prices
from engine.decide import finite_number

DECISION_REQUIRED = [
    "decision_id", "episode_id", "is_entry", "sku_id", "fc", "category",
    "subcategory", "date", "hour_of_day", "hours_remaining", "q_remaining",
    "original_price", "cost", "d_max", "feasible_tier_count",
    # actions allowed at THIS decision -- what explorability is judged on, and
    # distinct from the grid size. Required, because a decision that did not
    # explore is unauditable without it.
    "action_set_size",
    "optimal_price", "optimal_discount", "expected_il", "expected_denominator",
    "applied_price", "applied_discount", "is_exploration", "exploration_cost",
    "affordable_set_size", "tau_current", "delta_min",
    "epsilon_posterior_mean", "epsilon_posterior_std",
    "reference_discount", "reference_mu", "mu_ref_path", "anchor_discount",
    "dispersion_r",
    "baseline_model_version", "posterior_version", "config_version",
    # the digest of the config in force: what maps a priced hour to one
    # audit snapshot
    "config_digest",
    "timestamp",
]

# the decision's trading day: ingest matches feed rows on it and every
# per-day series (tau walk, guardrail, spend) is keyed on it, so it must be
# exactly one calendar date in one spelling
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _json_scalar(v):
    """numpy scalars serialise as their native values; anything else raises --
    silent stringification corrupts the log the reproduction check replays."""
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    raise TypeError(f"event field of type {type(v).__name__} is not "
                    "JSON-serialisable; emit native types")


OUTCOME_REQUIRED = [
    "outcome_id", "decision_id", "units_sold", "starting_inventory",
    "ending_inventory", "applied_price", "is_stockout", "execution_status",
    "finalized_at",
]


def _iso_day(v):
    """Exactly `YYYY-MM-DD`, and a real calendar date."""
    if not isinstance(v, str) or not ISO_DAY.match(v):
        return False
    try:
        datetime.date.fromisoformat(v)
    except ValueError:
        return False
    return True


def _validate_decision(evt):
    problems = []
    if not _iso_day(evt.get("date")):
        problems.append("date must be an ISO 'YYYY-MM-DD' string (the trading "
                        f"day ingest and the daily series key on); got "
                        f"{evt.get('date')!r}")
    return problems


def _validate_outcome(evt):
    problems = []
    for f in ("units_sold", "starting_inventory", "ending_inventory"):
        v = evt.get(f)
        # np.integer counts (pandas producers must not quarantine in bulk);
        # bool does NOT -- True is not a quantity of 1
        if (isinstance(v, bool) or not isinstance(v, (int, np.integer))
                or v < 0):
            problems.append(f"{f} must be a non-negative integer")
    if not problems:
        reconciles = (evt["ending_inventory"]
                      == evt["starting_inventory"] - evt["units_sold"])
        if not reconciles and not evt.get("adjustment_reason"):
            # three breaks are legitimate and MUST be named by the producer
            # (restock, final-row write-off, shrink) -- an integration that
            # omits any of them quarantines real outcomes in bulk
            problems.append("ending_inventory does not reconcile and no "
                            "adjustment_reason documented (expected "
                            "'intraday_restock', 'episode_close_write_off' "
                            "or 'unexplained_shortfall')")
    if not finite_number(evt.get("applied_price")):
        problems.append("applied_price must be a finite number")
    return problems


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


class EventStore:
    def __init__(self, cfg, root=None):
        self.cfg = cfg
        self.root = root or cfg["events"]["store_dir"]
        os.makedirs(self.root, exist_ok=True)
        self.paths = {k: os.path.join(self.root, f"{k}.jsonl")
                      for k in ("decisions", "outcomes", "quarantine")}
        # duplicates seen by THIS store: on emit, and -- because a foreign
        # producer may write the JSONL directly -- while loading. Either way
        # the same id twice is what the duplicate gate exists to catch.
        self.duplicate_counts = {"decision": 0, "outcome": 0}
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
                self._ids[kind].add(ident)
            for i, raw in torn:
                # a partial write (power loss mid-append) must not make the
                # store unconstructable: the line is quarantined with the
                # reason and the stream stays readable
                self._quarantine(
                    {"stream": f"{kind}s", "line_no": i, "raw_line": raw},
                    [f"unparseable JSONL line {i} in {kind}s.jsonl (torn "
                     "write?) -- skipped on load"])
            self._terminate_last_line(path)

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
        if problems:
            self._quarantine(evt, problems)
            return False
        if evt["decision_id"] in self._ids["decision"]:
            self.duplicate_counts["decision"] += 1
            return False
        # append FIRST: an id registered before a failed write (disk full)
        # would refuse the retry as a duplicate of an event never written
        self._append(self.paths["decisions"], evt)
        self._ids["decision"].add(evt["decision_id"])
        return True

    def emit_outcome(self, evt):
        missing = [f for f in OUTCOME_REQUIRED if f not in evt]
        problems = ([f"missing fields: {missing}"] if missing
                    else _validate_outcome(evt))
        if problems:
            self._quarantine(evt, problems)
            return False
        if evt["outcome_id"] in self._ids["outcome"]:
            self.duplicate_counts["outcome"] += 1
            return False
        self._append(self.paths["outcomes"], evt)     # append first, as above
        self._ids["outcome"].add(evt["outcome_id"])
        return True

    def _load(self, path, id_field=None):
        """The readable stream: torn lines were quarantined at construction
        and are skipped; a repeated id (a foreign producer's re-append) is
        COUNTED in duplicate_counts and loaded ONCE, first occurrence -- the
        learner must never consume the same outcome twice."""
        out, seen = [], set()
        for _, parsed, _ in self._lines(path):
            if parsed is None:
                continue
            ident = parsed.get(id_field) if id_field else None
            if ident is not None:
                if ident in seen:
                    continue
                seen.add(ident)
            out.append(parsed)
        return out

    def load_decisions(self):
        return self._load(self.paths["decisions"], "decision_id")

    def load_outcomes(self):
        return self._load(self.paths["outcomes"], "outcome_id")

    def load_quarantine(self):
        return self._load(self.paths["quarantine"])
