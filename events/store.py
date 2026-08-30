"""events.store -- decision and outcome event log (design 5.10; the field-level contract is docs/event_contract.html).

Append-only JSONL with duplicate detection, durable writes, and replay. The
logger never silently discards a malformed event: invalid events are
quarantined with the validation failure attached and surfaced to monitoring.
"""

import json

import numpy as np
import os

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
    "affordable_set_size", "tau_current",
    "epsilon_posterior_mean", "epsilon_posterior_std",
    "reference_discount", "reference_mu", "mu_ref_path", "anchor_discount",
    "dispersion_r",
    "baseline_model_version", "posterior_version", "config_version",
    "timestamp",
]

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
    price = evt.get("applied_price")
    if not isinstance(price, (int, float)) or price != price:
        problems.append("applied_price must be finite")
    return problems


def _quarantine_key(evt):
    """Identity of a quarantined event, or None if it carries no id. Both
    kinds share one quarantine file, so the kind is part of the key --
    otherwise an id collision silently swallows the second event."""
    for kind in ("outcome", "decision"):
        ident = evt.get(f"{kind}_id")
        if ident is not None:
            return (kind, ident)
    return None


class EventStore:
    def __init__(self, cfg, root=None):
        self.root = root or cfg["events"]["store_dir"]
        os.makedirs(self.root, exist_ok=True)
        self.paths = {k: os.path.join(self.root, f"{k}.jsonl")
                      for k in ("decisions", "outcomes", "quarantine")}
        self.duplicate_counts = {"decision": 0, "outcome": 0}
        # quarantined by THIS run; load_quarantine returns the whole FILE
        # (every run ever). The shadow gate must read the per-run figure,
        # never cumulative state (docs/learnings.md).
        self.quarantined_this_run = 0
        self._ids = {"decision": set(), "outcome": set()}
        for kind, path in (("decision", self.paths["decisions"]),
                           ("outcome", self.paths["outcomes"])):
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        self._ids[kind].add(json.loads(line)[f"{kind}_id"])
        # quarantine dedups too, by the same rule -- otherwise
        # quarantined_event_count grows on every re-run over the same store,
        # and the shadow gate reads it
        self._quarantined_ids = set()
        if os.path.exists(self.paths["quarantine"]):
            with open(self.paths["quarantine"]) as f:
                for line in f:
                    key = _quarantine_key(json.loads(line).get("event", {}))
                    if key is not None:
                        self._quarantined_ids.add(key)

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
        if missing:
            self._quarantine(evt, [f"missing fields: {missing}"])
            return False
        if evt["decision_id"] in self._ids["decision"]:
            self.duplicate_counts["decision"] += 1
            return False
        self._ids["decision"].add(evt["decision_id"])
        self._append(self.paths["decisions"], evt)
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
        self._ids["outcome"].add(evt["outcome_id"])
        self._append(self.paths["outcomes"], evt)
        return True

    def _load(self, path):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f]

    def load_decisions(self):
        return self._load(self.paths["decisions"])

    def load_outcomes(self):
        return self._load(self.paths["outcomes"])

    def load_quarantine(self):
        return self._load(self.paths["quarantine"])
