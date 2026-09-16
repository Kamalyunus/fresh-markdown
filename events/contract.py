"""events.contract -- what a decision and an outcome event must carry.

The field lists (`DECISION_REQUIRED`, `OUTCOME_REQUIRED`) and the value
checks the store runs before an event lands (`docs/engineering_handover.html`
is the human-readable page; `tests/test_event_contract_doc.py` keeps the
two in step). The store (events.store) enforces these; nothing else
re-derives them.
"""

import datetime
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


# recorded on every decision the batch caller emits since the feature
# table (design 5.10), OPTIONAL so a decision from before then still loads:
# the two demand-rate features the forecast stood on, null when unknown
DECISION_OPTIONAL = ["sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate"]

OUTCOME_REQUIRED = [
    "outcome_id", "decision_id", "units_sold", "starting_inventory",
    "ending_inventory", "applied_price", "is_stockout", "execution_status",
    "finalized_at",
]


def _is_iso_day(v):
    """Exactly `YYYY-MM-DD`, and a real calendar date -- a strict check,
    not the coercion events.pairs.iso_day performs on a feed value."""
    if not isinstance(v, str) or not ISO_DAY.match(v):
        return False
    try:
        datetime.date.fromisoformat(v)
    except ValueError:
        return False
    return True


def _validate_decision(evt):
    problems = []
    if not _is_iso_day(evt.get("date")):
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
