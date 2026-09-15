"""How a guardrail metric is compared against its baseline. ONE definition:
derive_thresholds measures the noise floor and daily.monitor evaluates the
trigger, and both MUST compute the same quantity. Two bases: `relative`
(t/c - 1) for strictly positive rates (scrap); `absolute_pp` (t - c) when a
metric can cross zero -- margin_rate does, so its relative floor exceeded the
series' own level and was structurally blocked (measured; docs/learnings.md).
"""

import numpy as np
import pandas as pd

RELATIVE = "relative"
ABSOLUTE_PP = "absolute_pp"
BASES = (RELATIVE, ABSOLUTE_PP)


def smooth(series, days):
    """Average `days` days before comparing. Must be applied on BOTH sides
    (floor and trigger) or the guardrail is inert."""
    s = series.dropna()
    return (s.rolling(days, min_periods=days).mean().dropna() if days > 1
            else s)


def deviation(treatment, control, worse_when_higher, basis):
    """Deterioration of `treatment` against `control`, positive = WORSE.
    Inputs are already smoothed and index-aligned; the sign convention is the
    caller's (scrap worse when higher, margin when lower). A relative
    deviation from a ZERO control is undefined, not infinite: it comes back
    NaN (no reading), so neither the floor nor the trigger sees +-inf."""
    if basis == ABSOLUTE_PP:
        return (treatment - control) if worse_when_higher else (control - treatment)
    if basis == RELATIVE:
        ratio = (treatment / control).replace([np.inf, -np.inf], np.nan)
        return (ratio - 1) if worse_when_higher else (1 - ratio)
    raise ValueError(f"unknown deterioration basis {basis!r}, expected one of {BASES}")


def smoothed_calendar(series, days):
    """`smooth` over each contiguous run of close days, laid on the full
    daily calendar (DatetimeIndex, NaN where no day closed). Rolled over
    ROWS, the first days after a gap (data.exclusion_window pre-launch, a
    day with no close live) are averaged with, and graded against, days
    from before it. On the calendar a gap is NaN on both sides, so a
    trailing window that spans it reads NaN -- no reading, never a seam."""
    s = pd.Series(series.to_numpy(), index=pd.to_datetime(series.index)).dropna()
    if s.empty:
        return s
    run = (s.index.to_series().diff() != pd.Timedelta(days=1)).cumsum()
    out = pd.concat([smooth(g, days) for _, g in s.groupby(run.to_numpy())])
    return out.asfreq("D") if len(out) else out


def deterioration_series(series, smooth_days, window, worse_when_higher, basis):
    """THE deterioration series both the floor (derive_thresholds) and the
    trigger (daily.monitor) read, positive = WORSE: `series` (a daily rate
    keyed by close day) smoothed over `smooth_days` on the calendar, against
    its trailing `window`-calendar-day mean shifted by the same smoothing so
    the two windows never overlap. FULL windows only -- min_periods below
    `window` manufactures deviations that are an estimator artifact. Days
    with no reading are dropped; the index is the "YYYY-MM-DD" close day.
    A CALENDAR window laid over close days: distinct from the trading-day
    window of events.pairs.quality_counts and the calendar IL base of
    engine.budget.trailing_daily_il -- three windows, kept apart."""
    s = smoothed_calendar(series, smooth_days)
    if s.empty:
        return pd.Series(dtype=float)
    trailing = s.rolling(window, min_periods=window).mean().shift(smooth_days)
    dev = deviation(s, trailing, worse_when_higher, basis).dropna()
    dev.index = dev.index.strftime("%Y-%m-%d")
    return dev


def evaluate_guardrail(block, threshold, persistence_days):
    """Fires only after `persistence_days` CONSECUTIVE CALENDAR days over
    threshold, ending on the latest day in the series. Persistence is
    load-bearing, not decoration: it buys sensitivity for thresholds sitting
    just above the measured noise floor (design 5.12). A calendar day with
    no reading breaks the streak -- an unobserved day is not a day over.
    The ONE streak rule: the monitor's stop conditions and shadow's
    controller trace both read it."""
    base = {"fired": False, "threshold": threshold,
            "persistence_days": persistence_days,
            "basis": block.get("basis"), "latest": block.get("latest")}
    if threshold is None:
        return {**base, "status": "BLOCKED -- threshold is null (SET BY OWNER)"}
    by_day = block.get("by_day") or {}
    if not by_day:
        # the series is empty until window + 2 x smoothing - 1 consecutive
        # close days (first_reading_close_days, carried by the block): a
        # short series legitimately has nothing to compare yet, and the
        # note says when it will
        first = block.get("first_reading_after_close_days")
        return {**base, "consecutive_days_over": 0,
                "status": "no comparable days yet" + (
                    f" (first reading after {first} consecutive close days)"
                    if first is not None else "")}
    streak, prev = 0, None
    for day in sorted(by_day, reverse=True):
        stamp = pd.Timestamp(day)
        if prev is not None and (prev - stamp).days != 1:
            break                                  # a missing calendar day
        if not by_day[day] > threshold:
            break
        streak += 1
        prev = stamp
    fired = streak >= persistence_days
    return {
        **base,
        "fired": fired,
        "consecutive_days_over": streak,
        "status": (f"FIRED -- over {threshold} for {streak} consecutive days"
                   if fired else
                   f"{streak}/{persistence_days} consecutive days over threshold"),
    }


def first_reading_close_days(smooth_days, window):
    """Consecutive close days `deterioration_series` needs before its FIRST
    reading, from the series' own arithmetic: `smooth_days` close days fill
    the first smoothed value, the trailing mean needs `window` smoothed
    values (window + smooth_days - 1 close days), and it is shifted by
    `smooth_days` more so the two windows never overlap. The one home the
    monitor's short-series note and the simulator's readiness read; a
    `window + smoothing` count graded a correct machine as a silent stop."""
    return int(window) + 2 * int(smooth_days) - 1


def stop_ready_close_days(smooth_days, window, persistence_days):
    """Consecutive close days before the trigger CAN fire at all: the first
    reading, then `persistence_days - 1` more readings for the streak."""
    return first_reading_close_days(smooth_days, window) + int(persistence_days) - 1


def change_visible_close_days(smooth_days, persistence_days):
    """Close days from the first FULLY changed close day (inclusive) until a
    level change fills `persistence_days` consecutive smoothed readings:
    the smoother carries the old level for `smooth_days - 1` more days,
    then the streak needs `persistence_days` readings."""
    return int(smooth_days) + int(persistence_days) - 1


BASIS = {"scrap": RELATIVE,        # strictly positive rate
         "margin": ABSOLUTE_PP}    # can cross zero (relative floor blocked)


def basis_for(metric_key):
    return BASIS.get(metric_key, RELATIVE)


def units_of(basis):
    """Human-readable units, for the report to say what a number is."""
    return ("percentage points of the rate (t - c)" if basis == ABSOLUTE_PP
            else "relative deviation (t/c - 1); 0.15 means 15%")


def verdict_is_blocking(verdict):
    """Design 12's three blocking floor verdicts, in ONE place so the checker
    and the paster cannot disagree: TOO TIGHT (fires on ordinary days and
    silently suspends exploration), BLOCKED (no threshold on this basis is
    both safe and useful), LIKELY INERT (a guardrail that cannot fire is
    absent, not conservative). `ops.status` refuses the chain and
    `ops.tune` refuses the paste on this same test.
    """
    v = str(verdict or "").upper()
    return v.startswith("TOO") or "BLOCKED" in v or "INERT" in v


def verdict_is_insufficient(verdict):
    """derive_thresholds' "insufficient history" verdicts: not blocking, and
    NOT a pass -- a floor nobody could measure is a floor nobody checked.
    status reads them as WARN and tune names them, on this one test."""
    return str(verdict or "").lower().startswith("insufficient")


def floor_is_unusable(floor, basis):
    """A floor a threshold cannot be set above -- the guardrail is BLOCKED.
    On the RELATIVE basis a floor >= 1.0 means ordinary daily swing exceeds
    the series' own level: the wrong basis for the metric, not a tuning
    problem. Absolute-pp floors have no such bound."""
    return bool(floor is not None and basis == RELATIVE and floor >= 1.0)
