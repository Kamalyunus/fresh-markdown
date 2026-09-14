"""The ONE pairing of outcomes to decisions, the ONE day key, and the ONE
event-quality count.

Four modules each rebuilt `{decision_id: d}` and walked the outcomes with
their own eligibility rule; the learning path excluded failed pushes and
the assurance path did not, so assurance graded r against prices that were
never charged. Every per-day series shares `decision_day` -- the trading
date the decision priced, never the UTC wall clock of its outcome.
"""

import numpy as np
import pandas as pd

from common import episodes

# applied vs recommended price: float noise on a currency amount, not a knob
PRICE_MATCH_TOLERANCE = 1e-6

# what a pushed outcome must report for its (decision, outcome) pair to
# teach anything: the price we chose was the price on the shelf
LEARNABLE_STATUSES = (None, "ok", "success")


def decision_day(d):
    """The TRADING date a decision priced."""
    return str(d.get("date") or pd.Timestamp(d["timestamp"]).date())


def _day(value):
    """One spelling of a trading day, whatever the producer's dtype: a
    parquet datetime column reads `2026-08-19 00:00:00` under str()."""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _ident(v):
    """One spelling of an identifier column: pandas reads an integer column
    as float once it holds a NaN, so the feed's 7.0 must key the decision's
    "7". A NaN or an unparseable value raises -- the caller counts the row."""
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v) or v != int(v):
            raise ValueError(f"not an identifier: {v!r}")
        return str(int(v))
    return str(v)


def hour_key(sku, fc, date, hour):
    """The (sku, fc, day, hour) a feed row, a decision and a price request
    meet on -- the ONE key, spelt one way. Raises on a value that names no
    hour or no item; the caller decides whether that costs one row or one
    decision, never the batch."""
    h = float(hour)
    if not np.isfinite(h) or h != int(h):
        raise ValueError(f"not an hour: {hour!r}")
    return (_ident(sku), _ident(fc), _day(date), int(h))


def outcome_id_of(key):
    """The outcome id for the hour keyed `key`: "feed-<sku>|<fc>|<date>T<hh>",
    computable by anyone holding the feed row -- engineering can name the
    outcome an hour will produce before it exists -- and the same on every
    re-ingest, so a re-run dedups instead of double-counting."""
    sku, fc, day, hour = key
    return f"feed-{sku}|{fc}|{day}T{hour:02d}"


def is_learnable(o):
    return o.get("execution_status") in LEARNABLE_STATUSES


def has_stock(o):
    """Stock on hand when the hour opened. An hour that opened empty sold
    nothing whatever demand was: its censored term is log P(D >= 0) = 0,
    no information -- and the update's `logsf(q - 1)` read it as
    P(D >= 1), evidence it never was."""
    return o.get("starting_inventory", 0) >= 1


def is_restocked(o):
    """Stock arrived mid-hour: the hour has no single q to empty, so neither
    the censoring rule nor the dispersion check can read it."""
    return o.get("adjustment_reason") == episodes.RESTOCK


def price_matches(d, o):
    return abs(o["applied_price"] - d["applied_price"]) <= PRICE_MATCH_TOLERANCE


def match_pairs(decisions, outcomes, learnable=False):
    """[(decision, outcome)] for outcomes that name a known decision, in
    outcome order. `learnable=True` keeps only outcomes whose push
    succeeded -- the learning and assurance population; business and
    guardrail series read every matched pair (a failed push still sold
    units and still scrapped)."""
    dec = {d["decision_id"]: d for d in decisions}
    out = []
    for o in outcomes:
        d = dec.get(o.get("decision_id"))
        if d is None or (learnable and not is_learnable(o)):
            continue
        out.append((d, o))
    return out


def learnable_with_stock(decisions, outcomes, pairs=None):
    """The pairs a model can be graded or taught on: the push succeeded
    (`is_learnable`) and the shelf held stock when the hour opened
    (`has_stock`). daily.assurance grades r and rho on exactly this set;
    daily.update learns from it (minus restocked hours, `is_restocked`,
    which it counts). `pairs` is match_pairs(decisions, outcomes) if the
    caller already built it."""
    if pairs is None:
        pairs = match_pairs(decisions, outcomes)
    return [(d, o) for d, o in pairs if is_learnable(o) and has_stock(o)]


def quality_counts(decisions, outcomes, cfg, duplicate_counts=None, pairs=None):
    """The event-quality counts the update gate and the monitor's stop
    condition both compare, over the trailing
    `monitoring.stop_conditions.event_quality_window_days` TRADING days
    (decision_day) ending on the latest priced day. All-time rates re-fired
    a resumed stop until history diluted one incident; a window lets a
    fixed integration clear the gate.

    Windowed: the compared pairs and their price mismatches, the unmatched
    outcomes (dated by their own `finalized_at`, the only day they carry;
    an undated one counts) and the denominator `outcomes_in_window`.
    ALL-TIME, on purpose: `duplicate_counts` is the store's count of ids
    seen twice on emit or on load -- a duplicate line has no trading day
    the store can key, and a foreign producer's re-appended line is
    re-counted on every load until it is removed from the JSONL. The
    asymmetry is deliberate: a duplicate is a broken producer, not an
    incident that ages out. `pairs` as in learnable_with_stock."""
    window = int(cfg["monitoring"]["stop_conditions"]["event_quality_window_days"])
    if pairs is None:
        pairs = match_pairs(decisions, outcomes)
    known = {d["decision_id"] for d in decisions}
    days = sorted({decision_day(d) for d in decisions})
    through = days[-1] if days else None
    start = ((pd.Timestamp(through) - pd.Timedelta(days=window - 1))
             .strftime("%Y-%m-%d") if through else None)

    def in_window(day):
        return start is None or day is None or day >= start

    compared = mismatches = reported = 0
    for d, o in pairs:
        if not in_window(decision_day(d)):
            continue
        # a push engineering REPORTED as failed is not a silent mismatch:
        # the gate catches the failures the failures table missed
        # (docs/event_contract.html); the reported ones are counted apart
        # and NOT compared, so the rate is over the pushes actually judged
        if not is_learnable(o):
            reported += 1
            continue
        compared += 1
        mismatches += not price_matches(d, o)
    unmatched = sum(
        1 for o in outcomes if o.get("decision_id") not in known
        and in_window(str(o.get("finalized_at") or "")[:10] or None))
    dup = duplicate_counts or {}
    return {
        "event_quality_window_days": window,
        "event_quality_window_start": start,
        "event_quality_through": through,
        "outcomes_in_window": compared + reported + unmatched,
        "unmatched_outcome_count": unmatched,
        "compared_pair_count": compared,
        "price_mismatch_count": mismatches,
        "push_failures_reported": reported,
        # all-time, from the store (see above)
        "duplicate_decision_count": int(dup.get("decision", 0)),
        "duplicate_outcome_count": int(dup.get("outcome", 0)),
    }


def quality_rates(counts):
    """The two event-quality rates from a `quality_counts` block (or any
    mapping carrying its count keys -- the monitor's safety block does).
    UNROUNDED: the gate and the stop condition compare these; rounding
    belongs to whoever prints them."""
    return {
        "duplicate_or_unmatched_rate": (
            counts["duplicate_decision_count"] + counts["duplicate_outcome_count"]
            + counts["unmatched_outcome_count"])
            / max(counts["outcomes_in_window"], 1),
        "price_mismatch_rate": (counts["price_mismatch_count"]
                                / max(counts["compared_pair_count"], 1)),
    }
