"""evaluate.level -- the level readings the two offline harnesses share.

The backtest (design 5.14) and shadow (5.13) price the SAME kind of frame
and grade the SAME question about the level factors -- what does the
weekly re-fit buy over the frozen anchor? -- so the pieces both read live
here, once: `predict_frame` (the frame either harness prices on),
`refit_scale` (the frozen-vs-refit rescale of a row's mu, through the
one applier, `BaselineModel.level_factors`), `weekly_refit_schedule`
(the factors re-fit per shadow week, in the harness rather than the
artifact -- rule 16) and `_coverage_preserved` (a side reading must not
disturb the gate's coverage counters). A harness importing its sibling
harness for these was the wrong direction; neither imports the other now.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd

from common import episodes
from fit.fit_dispersion import lookup_r_vec
from fit.prepare_data import population
from fit.calibrate import _solve_level_factors

# the per-hour columns extend_to_window regenerates on its synthetic tail;
# everything else is episode-constant and carried
_HOURLY = ("episode_id", "date", "hour_of_day", "hours_remaining",
           "starting_inventory", "ending_inventory", "units_sold")


def predict_frame(d, cfg, model, r_lookup):
    """The frame both harnesses price on: extended to the full window BEFORE
    predicting (an early sell-out must not shorten the DP horizon), in
    episode/hour order, with `r` (the dispersion lookup down its fallback
    chain, `fit.fit_dispersion.lookup_r_vec`) and `mu_ref_hat` (the
    CALIBRATED mu_ref, through the one applier). One home -- shadow's
    `_prepare_items` and the replay's `_attach_predictions` each carried a
    copy of this."""
    carry = [c for c in d.columns if c not in _HOURLY]
    d = episodes.extend_to_window(d, carry, cfg["data"]["max_window_hours"]).copy()
    d["r"] = lookup_r_vec(r_lookup, d.subcategory, d.category)
    d["mu_ref_hat"] = model.predict_mu_ref(d)
    return d


@contextmanager
def _coverage_preserved(model):
    """A side reading (the weekly-refit mechanism) must not disturb the
    calibration coverage counters or the freeze the gate pass set."""
    saved = (model._cal_rows_scheduled, model._cal_rows_fallback,
             model._cal_rows_frozen, model._cal_rows_static,
             set(model._cal_fallback_weeks))
    frozen_from = model._freeze_from
    try:
        yield
    finally:
        (model._cal_rows_scheduled, model._cal_rows_fallback,
         model._cal_rows_frozen, model._cal_rows_static,
         model._cal_fallback_weeks) = saved
        model.freeze_calibration_from(frozen_from)


def refit_scale(model, rows, schedule=None):
    """re-fit factor / frozen factor, per row of `rows` (`date` and the
    model's calibration grain): a factor swap is an exact rescale of
    mu_ref, so a row priced on the frozen anchor is re-read under the
    weekly schedule by this scale rather than predicted a second time.
    Both factors come from production's own applier
    (`BaselineModel.level_factors`, the one selection): the anchor factor
    the row was priced with (the freeze in force) and the factor the
    schedule gives it un-frozen -- the model's own schedule (the
    backtest's mechanism reading), or `schedule`, a table fitted in the
    harness (shadow's weekly re-fit; an unfitted week keeps the anchor, as
    the schedule does). The coverage counters and the freeze are left as
    they were."""
    with _coverage_preserved(model):
        frozen = model.level_factors(rows)
        own = model.calibration_schedule
        if schedule is not None:
            model.calibration_schedule = schedule
        model.freeze_calibration_from(None)
        try:
            refit = model.level_factors(rows)
        finally:
            model.calibration_schedule = own
    return refit / np.where(frozen > 0, frozen, 1.0)


def weekly_refit_schedule(d_full, cfg, model, r_lookup, start, end):
    """Re-fit the level factors per shadow week, as production's cron would.
    Fit HERE, not in the artifact, so the pre-launch bundle stays clean of
    hold-out rows (rule 16); at week k it reads only weeks < k.
    Returns ({week_start: {cell: factor}}, coverage)."""
    bm = cfg["baseline_model"]
    weeks_back = bm["calibration_fit_trailing_weeks"]
    scope = population(d_full, cfg).copy()
    dates = pd.to_datetime(scope.date)
    wk = dates.dt.to_period("W")
    lo_w = pd.Timestamp(start).to_period("W").start_time
    hi_w = pd.Timestamp(end).to_period("W").start_time
    # the episode's date key, computed ONCE for every week's cut
    opened = episodes.opening_dates(scope)

    out, coverage = {}, []
    for w in sorted(wk.unique()):
        w0 = w.start_time
        if w0 < lo_w or w0 > hi_w:
            continue
        # STRICTLY BEFORE this week: no look-ahead inside the replay; the
        # same whole-episode cut the artifact schedule uses
        window, weeks_seen = episodes.trailing_weeks_window(
            scope, w0, weeks_back, opened=opened)
        fitted = _solve_level_factors(
            window.copy(), model, bm["calibration_shrinkage_units"],
            bm["calibration_min_anchor_rows"], cfg["pricing"]["tier_step"],
            cfg["pricing"]["negbin_max_k"], r_lookup) if len(window) else None
        if fitted is None:                 # too thin: that week keeps the anchor
            coverage.append({"week": str(w0.date()), "fitted": False})
            continue
        out[str(w0.date())] = fitted[0]
        coverage.append({"week": str(w0.date()), "fitted": True,
                         "fit_rows": int(len(window)),
                         "weeks_in_window": weeks_seen,
                         "partial": weeks_seen < weeks_back})
    return out, coverage
