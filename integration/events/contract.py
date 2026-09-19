"""events.contract -- what a decision, an outcome and a rejection must carry.

The field lists (`DECISION_REQUIRED`, `OUTCOME_REQUIRED`,
`REJECTION_REQUIRED`) and the value checks the store runs before an event
lands (`docs/engineering_handover.html` is the human-readable page;
`tests/test_event_contract_doc.py` keeps the two in step). The store
(events.store) enforces these; nothing else re-derives them.
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

# A shelf-hour that reached us and was NOT priced, with the reason the
# response carried. It is not a decision (no price, so it never enters
# `priced_hours`, the pairing, or the evidence) and not an outcome; it is
# the record that the shelf WAS SEEN at that hour. Without it a refused
# hour is indistinguishable from an hour engineering never sent, and the
# episode-id rule -- which steps from what the store last saw on a shelf --
# had to answer "unknown" for every shelf held through a rejection.
# `episode_id` is required but may be null: a row refused FOR having no id
# is exactly the case worth recording.
REJECTION_REQUIRED = [
    "rejection_id", "episode_id", "sku_id", "fc", "date", "hour_of_day",
    "hours_remaining", "q_remaining", "reason", "timestamp",
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


def rejection_event(row, reason, timestamp=None):
    """The rejection event for a refused shelf-hour: `row` is the request or
    the snapshot row it was built from (canonical names), `reason` the string
    the response carries. The ONE builder -- both refusal paths (the hourly
    script's own, before a request exists, and the batch's) go through it, so
    the record is one shape. Returns None when the row names no shelf-hour,
    which is the one refusal nothing can be recorded for."""
    from events.pairs import hour_key, rejection_id_of      # local: pairs is id-only
    try:
        key = hour_key(row.get("sku_id"), row.get("fc"), row.get("date"),
                       row.get("hour_of_day"))
    except (AttributeError, TypeError, ValueError):
        return None
    return {
        "event": "rejection",
        "rejection_id": rejection_id_of(key),
        "episode_id": row.get("episode_id"),
        "sku_id": key[0], "fc": key[1], "date": key[2], "hour_of_day": key[3],
        # what the live episode-id rule steps from next hour
        "hours_remaining": row.get("hours_remaining"),
        "q_remaining": row.get("q_remaining", row.get("q")),
        "reason": reason,
        "timestamp": timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _validate_rejection(evt):
    problems = []
    if not _is_iso_day(evt.get("date")):
        problems.append("date must be an ISO 'YYYY-MM-DD' string; got "
                        f"{evt.get('date')!r}")
    if not isinstance(evt.get("reason"), str) or not evt["reason"].strip():
        problems.append("reason must be the non-empty string the response "
                        f"carried; got {evt.get('reason')!r}")
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
