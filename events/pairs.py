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


def iso_day(value):
    """One spelling of a trading day, whatever the producer's dtype: a
    parquet datetime column reads `2026-08-19 00:00:00` under str().
    Raises on a value that names no day (None, NaT, text that is not a
    date)."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"not a day: {value!r}")
    return stamp.strftime("%Y-%m-%d")


def ident(v):
    """One spelling of an identifier: pandas reads an integer column as
    float once it holds a NaN, so the feed's 7.0 must key the decision's
    "7" -- and a JSONL request's "7" must key an int history's 7. A NaN or
    an unparseable value raises -- the caller counts the row. The ONE
    spelling: the hour key, the price request and the feature service's
    history all read ids through it."""
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v) or v != int(v):
            raise ValueError(f"not an identifier: {v!r}")
        return str(int(v))
    if v is None or isinstance(v, (bool, np.bool_)):
        raise ValueError(f"not an identifier: {v!r}")
    return str(v)


def ident_series(s):
    """`ident` over a whole column, vectorised: the history's id columns
    are read once per batch (16M rows through a Python call per row was
    the morning). A NaN reads as None (a row that names no item can never
    be a feature for any request); a fractional value raises like `ident`."""
    s = pd.Series(s)
    if pd.api.types.is_bool_dtype(s):
        raise ValueError("not an identifier column: bool")
    if pd.api.types.is_numeric_dtype(s):
        x = s.astype(float)
        finite = np.isfinite(x)
        if not (x[finite] == np.floor(x[finite])).all():
            raise ValueError("not an identifier column: fractional values")
        out = pd.Series(None, index=s.index, dtype=object)
        out[finite] = x[finite].astype("int64").astype(str)
        return out
    out = pd.Series(None, index=s.index, dtype=object)
    present = s.notna()
    out[present] = s[present].astype(str)
    return out


def hour_key(sku, fc, date, hour):
    """The (sku, fc, day, hour) a feed row, a decision and a price request
    meet on -- the ONE key, spelt one way. Raises on a value that names no
    hour or no item; the caller decides whether that costs one row or one
    decision, never the batch."""
    h = float(hour)
    if not np.isfinite(h) or h != int(h):
        raise ValueError(f"not an hour: {hour!r}")
    return (ident(sku), ident(fc), iso_day(date), int(h))


def colliding_keys(keys):
    """The keys claimed MORE THAN ONCE in `keys` -- the one rule for "two
    states for one hour": two requests in a batch, two decisions in the
    store, two feed rows for one hour. None of the claimants is preferable,
    so every caller matches none of them and counts all of them."""
    seen, out = set(), set()
    for k in keys:
        if k in seen:
            out.add(k)
        seen.add(k)
    return out


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


def finalized_days(decisions, outcomes, pairs=None):
    """ONE pass over the matched pairs, keyed on the TRADING day the decision
    priced (decision_day, never the UTC clock of `finalized_at`). Every
    stored outcome is final -- `finalized_at` is in
    events.contract.OUTCOME_REQUIRED -- so a matched pair is a priced day.
    Returns (priced_days ascending, {day: realised exploration spend} over
    the forced decisions whose push EXECUTED -- a failed push
    (is_learnable) left the old price on the shelf, so its expected
    sacrifice was never spent) -- the day key and the spend the tau
    controller (daily.update) and the monitor's stop condition both read,
    so the correction and its backstop cannot drift apart. `pairs` is
    match_pairs(decisions, outcomes) if the caller already built it."""
    days, spend = set(), {}
    for d, o in (match_pairs(decisions, outcomes) if pairs is None else pairs):
        day = decision_day(d)
        days.add(day)
        if d.get("is_exploration") and is_learnable(o):
            spend[day] = spend.get(day, 0.0) + float(d["exploration_cost"])
    return sorted(days), spend


def suspended_days(decisions):
    """The trading days on which NO decision had a budget in force: every
    decision priced that day carries `tau_current` None (engine.decide
    records None while the store holds a suspension). The controller holds
    tau on such a day (engine.budget.budget_held) -- nothing was drawn, so
    its zero spend is no reading. A day with any budgeted decision is
    graded."""
    budgeted, seen = set(), set()
    for d in decisions:
        day = decision_day(d)
        seen.add(day)
        if d.get("tau_current") is not None:
            budgeted.add(day)
    return sorted(seen - budgeted)


def quality_counts(decisions, outcomes, cfg, duplicate_counts=None, pairs=None,
                   completeness_counts=None):
    """The event-quality counts the update gate and the monitor's stop
    condition both compare, over the trailing
    `monitoring.stop_conditions.event_quality_window_days` TRADING days
    (decision_day) ending on the latest priced day. All-time rates re-fired
    a resumed stop until history diluted one incident; a window lets a
    fixed integration clear the gate. TRADING days, on purpose: the tau
    controller's IL base (engine.budget.trailing_daily_il) and the
    guardrail series (common.guardrail.deterioration_series) window over
    CALENDAR days -- the three are distinct and never merged.

    Windowed: the compared pairs and their price mismatches, the unmatched
    outcomes (dated by their own `finalized_at`, the only day they carry;
    an undated one counts) and the denominator `outcomes_in_window`.
    ALL-TIME, on purpose: `duplicate_counts` is the store's count of ids
    seen twice on emit or on load -- a duplicate line has no trading day
    the store can key, and a foreign producer's re-appended line is
    re-counted on every load until it is removed from the JSONL. The
    asymmetry is deliberate: a duplicate is a broken producer, not an
    incident that ages out. `pairs` as in learnable_with_stock.

    The DECISION side of completeness, windowed the same way:
    `decisions_colliding_on_hour` -- decisions sharing one hour key
    (colliding_keys; ingest matches none of them, so every one is a gap);
    `decisions_without_outcome` -- decisions on an ANSWERED day (a day
    the feed has produced at least one outcome for, through
    `decisions_answered_through`) that no outcome names, over
    `decisions_on_answered_days`. Days after the last answered one are
    pending, not gaps -- today's decisions are ingested tomorrow morning.
    `completeness_counts` is the store's all-time record of what it
    refused: `outcomes_per_decision_over_one` (a second outcome for one
    decision) and `missing_stockout_field`; a caller without a store
    (update's gate) leaves them 0."""
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
    answered = set()
    for d, o in pairs:
        answered.add(decision_day(d))
        if not in_window(decision_day(d)):
            continue
        # a push engineering REPORTED as failed is not a silent mismatch:
        # the gate catches the failures the failures table missed
        # (docs/engineering_handover.html); the reported ones are counted apart
        # and NOT compared, so the rate is over the pushes actually judged
        if not is_learnable(o):
            reported += 1
            continue
        compared += 1
        mismatches += not price_matches(d, o)
    unmatched = sum(
        1 for o in outcomes if o.get("decision_id") not in known
        and in_window(str(o.get("finalized_at") or "")[:10] or None))

    # the decision side: keyed once, in the window; an unkeyable decision
    # (a foreign line naming no hour) can neither collide nor be answered
    answered_through = max(answered) if answered else None
    keyed = []
    for d in decisions:
        day = decision_day(d)
        if not in_window(day):
            continue
        try:
            keyed.append((d, day, hour_key(d.get("sku_id"), d.get("fc"), day,
                                           d.get("hour_of_day"))))
        except (TypeError, ValueError):
            continue
    collided = colliding_keys(k for _, _, k in keyed)
    colliding = sum(1 for _, _, k in keyed if k in collided)
    with_outcome = {d["decision_id"] for d, _ in pairs}
    on_answered = [d for d, day, _ in keyed
                   if answered_through is not None and day <= answered_through]
    without = sum(1 for d in on_answered if d["decision_id"] not in with_outcome)
    dup = duplicate_counts or {}
    comp = completeness_counts or {}
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
        # the decision side of completeness (windowed; see above)
        "decisions_colliding_on_hour": colliding,
        "decisions_answered_through": answered_through,
        "decisions_on_answered_days": len(on_answered),
        "decisions_without_outcome": without,
        # all-time, from the store: refused on emit or skipped on load
        "outcomes_per_decision_over_one": int(comp.get("outcomes_per_decision_over_one", 0)),
        "missing_stockout_field": int(comp.get("missing_stockout_field", 0)),
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
