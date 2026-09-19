"""engine.budget -- the exploration budget and the tau controller (design 5.8).

tau is a CURRENCY amount, compared against Q(p_star) - Q(p), and the one
controller: the forced RATE is whatever the budget affords. The budget is
`budget_share_of_il` x the trailing realised daily IL x a posterior-width
scale, and tau walks toward it one clipped step per closed day
(`walk_tau`) -- production (daily.update) and shadow's trace both call
the one walk. `budget_held` says when a day's budget is no signal at all
(a suspension, no IL base, a base shorter than its window); the controller
holds and the overspend stop takes no reading on such a day. The draw
itself -- which tiers, uniform over the affordable set -- is
engine.explore.
"""

import pandas as pd


def trailing_daily_il(il_by_day, day, cfg):
    """Mean REALISED daily IL over the trailing `budget_il_window_days`
    calendar days ending yesterday -- trailing and realised, never same-day
    or a forecast (design 5.8). Zero-IL calendar days count as zero; no
    history at all means a zero budget, the conservative side to start on.
    CALENDAR days, keyed by close day: distinct from the trading-day window
    of events.pairs.quality_counts and the calendar-laid trailing mean of
    common.guardrail.deterioration_series -- three windows, three
    calendars, on purpose.
    """
    window = int(cfg["exploration"]["budget_il_window_days"])
    d0 = pd.Timestamp(str(day))
    days = [(d0 - pd.Timedelta(days=k)).strftime("%Y-%m-%d")
            for k in range(1, window + 1)]
    if not il_by_day or not any(k in il_by_day for k in days):
        return 0.0
    # zero-IL calendar days count as zero over the SAME span
    # budget_base_ready judges -- back to the earliest close known, capped
    # at the window; a no-close day at the window's leading edge once fell
    # out of the denominator and inflated the budget by window/span
    return float(sum(il_by_day.get(k, 0.0) for k in days)
                 / max(_base_span(il_by_day, d0, window), 1))


def _base_span(il_by_day, d0, window):
    """Days the IL base covers before `d0`, capped at the window."""
    earliest = pd.Timestamp(min(str(k) for k in il_by_day))
    return min(window, (d0 - earliest).days)


def budget_base_ready(il_by_day, day, cfg):
    """Does the trailing IL base reach back a whole `budget_il_window_days`
    before `day`? A pilot's first mornings hold a handful of early closers,
    not the shop's IL: a budget priced from them is tiny, and on the
    owner's rehearsal the overspend stop fired on day three and suspended
    exploration for the rest of the run. Until the base spans its window
    the budget is an absence of signal (like a zero one): the controller
    holds tau and the stop takes no reading. The ONE rule both read, and
    the span trailing_daily_il divides by."""
    if not il_by_day:
        return False
    window = int(cfg["exploration"]["budget_il_window_days"])
    return _base_span(il_by_day, pd.Timestamp(str(day)), window) >= window


SUSPENDED = "exploration suspended"


def budget_held(il_by_day, day, budget, cfg, suspended=False):
    """Why a day's budget is no signal, or None: exploration was SUSPENDED
    that day (nothing was drawn, so its zero spend says nothing about tau
    -- graded as under-spend it ratcheted tau up by the clip every
    suspended morning, and the resume then overspent at once), no trailing
    IL at all (a zero budget), or a base shorter than its window. The
    controller (walk_tau) holds tau on such a day and the monitor's
    overspend series takes no reading -- one composite, read by both."""
    if suspended:
        return SUSPENDED
    if budget <= 0:
        return "no trailing IL"
    if not budget_base_ready(il_by_day, day, cfg):
        return "IL base shorter than budget_il_window_days"
    return None


def budget_scale(posterior_std, cfg):
    """The budget's posterior-width factor: 1 at the reference std, shrinking
    with the widest routed std, never below budget_scale_floor."""
    ec = cfg["exploration"]
    return min(max(posterior_std / ec["budget_scale_ref_std"],
                   ec["budget_scale_floor"]), 1.0)


def budget_today(trailing_il, posterior_std, cfg):
    """budget_share_of_il x trailing daily IL x budget_scale (design 5.8)."""
    return cfg["exploration"]["budget_share_of_il"] * budget_scale(posterior_std, cfg) * trailing_il


def walk_tau(tau, days, spend_for, il_by_day, widest_std, cfg,
             suspended_days=()):
    """The controller walk, day by day, in one place: production (every
    closed day since the last calibration -- a weekly batch is seven
    steps, never one) and shadow's trace (expected spend at the tau in
    force) both call it. `spend_for(day, tau)` returns the day's realised
    or expected exploration spend. A ZERO budget (no trailing IL yet), a
    base shorter than its window (budget_base_ready) and a day in
    `suspended_days` (exploration suspended: nothing drawn, so the zero
    spend is no reading) are an absence of signal, not an overspend: tau
    holds that day. Returns (tau_end, rows); a row's `clipped` says the
    step sat on a clip bound, `held` why it did not move."""
    rows = []
    suspended_days = {str(d) for d in suspended_days}
    for day in days:
        budget = budget_today(trailing_daily_il(il_by_day, day, cfg),
                              widest_std, cfg)
        spend = float(spend_for(day, tau))
        held = budget_held(il_by_day, day, budget, cfg,
                           suspended=str(day) in suspended_days)
        after, clipped = (tau, False) if held else tau_next(tau, budget, spend, cfg)
        rows.append({"day": str(day), "tau": round(float(tau), 2),
                     "spend": round(spend, 1), "budget": round(budget, 1),
                     "tau_after": round(float(after), 2), "clipped": clipped,
                     # why the step did not move, or None: the overspend
                     # stop and shadow's trace take no reading on a held day
                     "held": held})
        tau = after
    return tau, rows


def tau_next(tau, budget, realised_cost, cfg):
    """tau * clip(budget/spend, *tau_adjust_clip) -- asymmetric on purpose:
    cutting is the safety direction, raising is never urgent (config).
    Returns (tau_after, clipped): whether the clip, not the ratio, set the
    step -- read from the ratio here, never inferred from rounded taus."""
    ec = cfg["exploration"]
    lo, hi = ec["tau_adjust_clip"]
    ratio = budget / max(realised_cost, ec["tau_spend_guard"])
    factor = min(max(ratio, lo), hi)
    return tau * factor, factor != ratio
