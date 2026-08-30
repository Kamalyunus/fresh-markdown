"""bootstrap.train_baseline -- fit and freeze the reference-demand model (design 5.4).

LightGBM/Tweedie on units_sold, predicting ONLY at the reference discount:
price features are overwritten to d_ref at inference, so the price gradient is
never queried. Frozen at launch; version recorded on every decision. Also fits
the level-calibration factors -- ALWAYS fitted and applied (owner, 2026-08-25).
Run: python3 -m bootstrap.train_baseline --input data/prepared.parquet [--fit-calibration]
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import load_config, reference_discount
from common.provenance import stamp
from bootstrap.prepare_data import population, pre_launch, split_frames
from pricing.demand import expected_min_demand_inventory_vec

# Feature order is authoritative in feature_schema.json; this list seeds the
# first fit. `total_discount` is the SINGLE price feature overwritten to d_ref
# at inference. Deliberately absent: hours_remaining, lag sales, inventory /
# stockout -- they leak price response or belong to the DP state (design.md).
FEATURES = ["category", "subcategory", "fc", "hour_of_day", "dow",
            "day_of_month", "original_price",
            "sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate",
            "total_discount"]
CATEGORICAL = ["category", "subcategory", "fc"]
PRICE_FEATURES = ["total_discount"]


def add_derived(d):
    d = d.copy()
    dates = pd.to_datetime(d.date)
    d["dow"] = dates.dt.dayofweek
    d["day_of_month"] = dates.dt.day
    return d


class BaselineModel:
    """Frozen mu_ref predictor. Loads model + schema + calibration artifacts."""

    # class-level defaults: artifacts written before the schedule carried a
    # stop, and instances built via __new__, still answer coverage queries
    calibration_stops_at = None
    _freeze_from = None

    def __init__(self, cfg):
        bm = cfg["baseline_model"]
        self.cfg = cfg
        self.booster = lgb.Booster(model_file=bm["model_path"])
        with open(bm["feature_schema_path"]) as f:
            self.schema = json.load(f)
        self.calibration, self.calibration_grain = {}, "category"
        # POINT-IN-TIME factors, when the artifact carries a schedule: the
        # week-keyed map is applied by ROW DATE, so a row is never priced by
        # a factor fitted on its own week or later. `calibration` remains the
        # fallback -- rows before the first fitted week, and artifacts written
        # before the schedule existed.
        self.calibration_schedule = None
        # the week the schedule deliberately stops at (the gate window's
        # start), if it does. Rows past a DELIBERATE stop are the graded
        # window holding a frozen anchor by design, not production drifting
        # onto stale factors -- coverage must not call the two the same thing.
        self.calibration_stops_at = None
        if os.path.exists(bm["calibration_factor_path"]):
            with open(bm["calibration_factor_path"]) as f:
                cal = json.load(f)
            self.calibration = cal.get("factors", cal.get("factor_by_category", {}))
            self.calibration_grain = cal.get("grain", "category")
            sched = cal.get("schedule")
            if sched and sched.get("by_week"):
                self.calibration_schedule = sched["by_week"]
                self.calibration_stops_at = sched.get("gate_freezes_at")
        self._reset_calibration_counters()
        self.version = self.schema["model_version"]

    def _reset_calibration_counters(self):
        """Coverage counters: which priced rows got their OWN week's factors."""
        self._cal_rows_scheduled = 0
        self._cal_rows_fallback = 0
        self._cal_rows_frozen = 0
        self._cal_rows_static = 0
        self._cal_fallback_weeks = set()

    def freeze_calibration_from(self, date):
        """Price rows on or after `date` with the frozen anchor, ignoring the
        weekly schedule. `None` restores the schedule.

        The two readings a harness needs from ONE artifact. Left alone, the
        schedule re-fits weekly and the model mirrors production, which is
        what a forward-time replay (shadow, the DP walk) should see: at week k
        only weeks < k were ever read, so there is no look-ahead. But the
        LAUNCH GATE asks a different question -- does the artifact as frozen
        reproduce hold-out sales -- and a factor re-fit inside the hold-out has
        read the rows it is graded on. Freezing at the gate start answers that
        one without a second artifact, and without a second copy of the
        factor-selection logic to drift out of sync.
        """
        self._freeze_from = pd.Timestamp(date) if date is not None else None
        return self

    def _factor_vector(self, d):
        """Per-row level factor. With a schedule, each row takes the factors
        in force for ITS week; without one, the single frozen set. Rows at or
        after a `freeze_calibration_from` point take the anchor regardless."""
        keys = d[self.calibration_grain].astype(str)
        if self.calibration_schedule is None:
            self._cal_rows_static += len(d)
            return keys.map(lambda k: self.calibration.get(k, 1.0)).to_numpy()
        dates = pd.to_datetime(d["date"])
        weeks = dates.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
        frozen = (dates >= self._freeze_from).to_numpy() \
            if getattr(self, "_freeze_from", None) is not None \
            else np.zeros(len(d), bool)
        out = np.ones(len(d))
        for i, (wk, key) in enumerate(zip(weeks.to_numpy(), keys.to_numpy())):
            # an unfitted week (too thin, or before the first trailing window
            # closed) holds at the frozen fallback, never at a later week's
            # factors -- borrowing forward is the leak this exists to prevent
            if frozen[i]:
                # DELIBERATE, not a gap: counted apart so a gate run cannot
                # read as production drifting onto stale factors
                self._cal_rows_frozen += 1
                out[i] = self.calibration.get(key, 1.0)
                continue
            table = self.calibration_schedule.get(wk)
            if table is None:
                self._cal_rows_fallback += 1
                self._cal_fallback_weeks.add(str(wk))
                out[i] = self.calibration.get(key, 1.0)
            else:
                self._cal_rows_scheduled += 1
                out[i] = table.get(key, 1.0)
        return out

    def calibration_coverage(self):
        """How many priced rows got a POINT-IN-TIME factor and how many fell
        back to the frozen set.

        Three ways a row can end up on the anchor instead, and they must never
        share a verdict. DELIBERATE: the run called
        `freeze_calibration_from` -- the launch gate, pricing its hold-out off
        a frozen artifact on purpose. BEFORE THE START: the first trailing
        window had not closed yet. PAST THE END: production drifting back onto
        stale factors the moment it runs beyond the last week
        `--fit-calibration` saw, which is the one the weekly re-fit exists to
        prevent and the only one that is a problem.
        """
        if self.calibration_schedule is None:
            return {"mode": "static", "rows": self._cal_rows_static,
                    "note": "no schedule in the artifact: one frozen factor "
                            "set applied to every row"}
        priced = (self._cal_rows_scheduled + self._cal_rows_fallback
                  + self._cal_rows_frozen)
        weeks = sorted(self.calibration_schedule)
        past_end = sorted(w for w in self._cal_fallback_weeks
                          if weeks and w > weeks[-1])
        share = self._cal_rows_fallback / max(priced, 1)
        return {
            "mode": "point_in_time",
            "schedule_covers": [weeks[0], weeks[-1]] if weeks else None,
            "rows_priced": priced,
            "rows_on_schedule": self._cal_rows_scheduled,
            "rows_on_fallback": self._cal_rows_fallback,
            "rows_frozen_at_anchor": self._cal_rows_frozen,
            "frozen_from": (str(self._freeze_from.date())
                            if self._freeze_from is not None else None),
            "fallback_share": round(self._cal_rows_fallback / priced, 4)
                if priced else None,
            "fallback_weeks": sorted(self._cal_fallback_weeks),
            "weeks_after_schedule_end": past_end,
            "gate_freezes_at": self.calibration_stops_at,
            "verdict": (
                "STALE FACTORS IN USE -- {} rows ({:.1%}) are in weeks PAST "
                "the end of the schedule and fell back to the frozen set. "
                "Re-run `train_baseline --fit-calibration`: the schedule "
                "stops at the last week it was fitted on.".format(
                    self._cal_rows_fallback, share)
                if past_end else
                "OK -- {} rows frozen at the anchor from {} on purpose (the "
                "launch gate prices its hold-out off the frozen artifact); "
                "every other priced row took its own week's factors".format(
                    self._cal_rows_frozen, self._freeze_from.date())
                if self._freeze_from is not None else
                "OK -- every priced row took its own week's factors"
                if not self._cal_rows_fallback else
                "OK -- {} rows ({:.1%}) fell back before the schedule opens; "
                "the first trailing window had not closed yet".format(
                    self._cal_rows_fallback, share)),
        }

    def _matrix(self, d):
        missing = [f for f in self.schema["features"]
                   if f not in d.columns and f not in ("dow", "day_of_month")]
        if missing:
            raise KeyError(
                f"frame is missing feature columns {missing} -- re-run "
                "bootstrap.prepare_data (the feature set includes "
                "point-in-time rate features it computes)")
        X = pd.DataFrame(index=d.index)
        for feat in self.schema["features"]:
            if feat in self.schema["categorical"]:
                cats = self.schema["category_levels"][feat]
                X[feat] = pd.Categorical(d[feat].astype(str), categories=cats).codes
            else:
                X[feat] = pd.to_numeric(d[feat])
        return X

    def predict_mu_ref(self, d):
        """mu_ref(context): price features overwritten to d_ref before predict."""
        d = add_derived(d)
        for feat in self.schema["price_features"]:
            d[feat] = d["d_ref"] if "d_ref" in d.columns else d["category"].map(
                lambda c: reference_discount(self.cfg, c))
        mu = self.booster.predict(self._matrix(d))
        mu = np.clip(mu, self.cfg["pricing"]["demand_floor"], None)
        if self.cfg["baseline_model"]["apply_level_calibration"]:
            mu = mu * self._factor_vector(d)
        return mu


def train(d, cfg):
    bm = cfg["baseline_model"]
    splits = split_frames(d, cfg)
    # train_population "integrity" keeps DP-ineligible episodes -- FEATURES
    # carries neither cost nor hours_remaining, so the model cannot see why
    train_d = add_derived(population(splits["train"], cfg))

    levels = {c: sorted(train_d[c].astype(str).unique().tolist()) for c in CATEGORICAL}
    X = pd.DataFrame(index=train_d.index)
    for feat in FEATURES:
        if feat in CATEGORICAL:
            X[feat] = pd.Categorical(train_d[feat].astype(str),
                                     categories=levels[feat]).codes
        else:
            X[feat] = pd.to_numeric(train_d[feat])

    booster = lgb.train(
        {
            "objective": bm["objective"],
            "tweedie_variance_power": bm["tweedie_variance_power"],
            "learning_rate": bm["learning_rate"],
            "num_leaves": bm["num_leaves"],
            "min_data_in_leaf": bm["min_data_in_leaf"],
            "verbosity": -1,
        },
        lgb.Dataset(X, label=train_d["units_sold"],
                    categorical_feature=[FEATURES.index(c) for c in CATEGORICAL]),
        num_boost_round=bm["num_boost_round"],
    )

    os.makedirs(os.path.dirname(bm["model_path"]) or ".", exist_ok=True)
    booster.save_model(bm["model_path"])
    schema = {
        "model_version": f"baseline-{pd.Timestamp.now('UTC'):%Y%m%d%H%M%S}",
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "price_features": PRICE_FEATURES,
        "category_levels": levels,
        "objective": bm["objective"],
        "train_rows": int(len(train_d)),
        "frozen": bool(bm["freeze_at_launch"]),
    }
    with open(bm["feature_schema_path"], "w") as f:
        json.dump(schema, f, indent=2)
    return schema


def _solve_level_factors(calib, cfg, model, grain, k_shrink, min_anchor,
                         tier_step, max_k, r_lookup):
    """Factors for ONE fit window. The single definition of the solve, shared
    by the static windows and by every week of the rolling schedule -- two
    copies would drift and only the rolling one is exercised weekly.

    Returns (factors, detail, global_factor), or None when the window holds
    too few anchor rows to fit on: the caller holds those weeks at 1.0 rather
    than fitting noise, and says which.
    """
    from bootstrap.fit_dispersion import lookup_r   # local: avoids a cycle

    saved = cfg["baseline_model"]["apply_level_calibration"]
    cfg["baseline_model"]["apply_level_calibration"] = False
    calib["mu_ref_hat"] = model.predict_mu_ref(calib)
    cfg["baseline_model"]["apply_level_calibration"] = saved

    if r_lookup is not None:
        calib["r_val"] = [lookup_r(r_lookup, s, c)
                          for s, c in zip(calib.subcategory, calib.category)]

    def solve_factor(anchor):
        sold = float(anchor["units_sold"].sum())
        mu = anchor["mu_ref_hat"].to_numpy()
        if r_lookup is None:
            pred = float(mu.sum())
            return (sold / pred if pred > 0 else 1.0), pred
        r = anchor["r_val"].to_numpy()
        q = anchor["starting_inventory"].to_numpy()

        def predicted(f):
            return float(expected_min_demand_inventory_vec(
                f * mu, r, q, max_k).sum())

        base = predicted(1.0)
        if base <= 0 or sold <= 0:
            return 1.0, base
        lo, hi = 0.1, 10.0
        if predicted(lo) > sold:
            return lo, base
        if predicted(hi) < sold:
            return hi, base
        for _ in range(40):                    # monotone in f -> bisection
            mid = (lo + hi) / 2
            if predicted(mid) < sold:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2, base

    def shrink(cell, parent, evidence):
        if evidence <= 0 or cell <= 0:
            return parent
        w = evidence / (evidence + k_shrink)
        return float(np.exp(w * np.log(cell) + (1 - w) * np.log(parent)))

    anchor_all = calib[(calib.total_discount - calib.d_ref).abs()
                       <= tier_step / 2]
    if len(anchor_all) < min_anchor or anchor_all["mu_ref_hat"].sum() <= 0:
        return None

    f_global, _ = solve_factor(anchor_all)

    def fit_level(groups, parent_of):
        out, det = {}, {}
        for key, g in groups:
            raw, pred = solve_factor(g)
            evidence = float(g["units_sold"].sum())
            parent = parent_of(key, g)
            f = shrink(raw, parent, evidence)
            out[str(key)] = round(float(f), 4)
            det[str(key)] = {
                "anchor_rows": int(len(g)),
                "anchor_sold": int(evidence),
                "anchor_predicted_at_f1": round(float(pred), 1),
                "raw_factor": round(float(raw), 4),
                "parent_factor": round(float(parent), 4),
                "shrinkage_weight_on_self": round(
                    float(evidence / (evidence + k_shrink)), 3),
            }
        return out, det

    cat_factors, cat_detail = fit_level(
        anchor_all.groupby("category"), lambda k, g: f_global)
    if grain == "subcategory":
        factors, detail = fit_level(
            anchor_all.groupby("subcategory"),
            lambda k, g: cat_factors.get(str(g["category"].iloc[0]), f_global))
    elif grain == "category":
        factors, detail = cat_factors, cat_detail
    else:
        raise ValueError(f"unknown calibration_grain: {grain}")
    return factors, detail, f_global


def fit_level_calibration(d, cfg):
    """Per-category multiplicative level factor, fit on ANCHOR ROWS ONLY
    (elasticity ~1 there, so slope error cannot leak in; thin cells stay 1.0).
    Fit window must stay DISJOINT from calibration_gate_window, or the fit
    grades itself. Solved on the censored basis E[min(D,q)] -- the gate's
    quantity; f scales mu before censoring, so solved for, not divided out."""
    from bootstrap.fit_dispersion import lookup_r   # local: avoids a cycle

    model = BaselineModel(cfg)
    splits = split_frames(d, cfg)
    fit_window = cfg["baseline_model"]["calibration_fit_window"]
    splits = {k: population(v, cfg) for k, v in splits.items()}
    # where the graded window opens -- the gate window's own start, not a
    # hardcoded calib_start: test_start when the gate reads test, calib_start
    # when it reads calib+test. NOTHING the calibration reads may sit at or
    # after this instant, or the level factor grades itself.
    split = cfg["data"]["split"]
    gate_start = pd.Timestamp(
        split["test_start"]
        if cfg["baseline_model"]["calibration_gate_window"] == "test"
        else split["calib_start"])
    if fit_window == "calib":
        calib = splits["calib"].copy()
    elif fit_window == "train+calib":
        calib = pd.concat([splits["train"], splits["calib"]])
    elif fit_window == "all":
        # "all" means all PRE-LAUNCH data, never the hold-out
        calib = population(pre_launch(d, cfg), cfg).copy()
    elif fit_window in ("trailing", "rolling_trailing"):
        # rolling_trailing fits a WEEKLY SCHEDULE below; this static set is
        # its fallback, for weeks before the first trailing window closes
        # last N weeks ENDING WHERE THE GATE WINDOW BEGINS: recent, and
        # disjoint from what the gate evaluates
        weeks = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
        lo = gate_start - pd.Timedelta(weeks=weeks)
        dates = pd.to_datetime(d.date)
        calib = d[(dates >= lo) & (dates < gate_start)].copy()
    else:
        raise ValueError(f"unknown calibration_fit_window: {fit_window}")
    if not len(calib):
        raise RuntimeError("calibration fit window contains no rows")

    # a trailing window inside the training period understates the needed
    # correction (the model fits there by construction) -- surfaced below
    fit_dates = pd.to_datetime(calib.date)
    train_end = pd.Timestamp(cfg["data"]["split"]["train_end"])
    in_sample_share = float((fit_dates <= train_end).mean())
    tier_step = cfg["pricing"]["tier_step"]

    # predict without any existing calibration applied
    saved = cfg["baseline_model"]["apply_level_calibration"]
    cfg["baseline_model"]["apply_level_calibration"] = False
    calib["mu_ref_hat"] = model.predict_mu_ref(calib)
    cfg["baseline_model"]["apply_level_calibration"] = saved

    # censoring basis: solve against E[min(D, q)], the gate's quantity -- a
    # factor fit on raw mu reads low and cannot move a censored gate
    max_k = cfg["pricing"]["negbin_max_k"]
    r_path = cfg["dispersion"]["r_lookup_path"]
    censored_basis = os.path.exists(r_path)
    if censored_basis:
        with open(r_path) as f:
            r_lookup = json.load(f)
        calib["r_val"] = [lookup_r(r_lookup, s, c)
                          for s, c in zip(calib.subcategory, calib.category)]

    min_anchor = cfg["baseline_model"]["calibration_min_anchor_rows"]
    grain = cfg["baseline_model"]["calibration_grain"]
    k_shrink = cfg["baseline_model"]["calibration_shrinkage_units"]
    r_lookup_or_none = r_lookup if censored_basis else None

    fitted = _solve_level_factors(calib, cfg, model, grain, k_shrink,
                                  min_anchor, tier_step, max_k,
                                  r_lookup_or_none)
    if fitted is None:
        anchors = len(calib[(calib.total_discount - calib.d_ref).abs()
                            <= tier_step / 2])
        raise RuntimeError(
            f"fit window has only {anchors} anchor rows "
            f"(need {min_anchor}) -- widen calibration_fit_window")
    factors, detail, f_global = fitted
    cat_factors = factors if grain == "category" else None

    # POINT-IN-TIME schedule: factors the way production would actually hold
    # them -- re-fit every week on the trailing window ENDING STRICTLY BEFORE
    # that week, so no row is ever priced by a factor fitted on its own week
    # or later. Both harnesses read it by row date, so the hold-out drift
    # ratio grades the MECHANISM rather than one frozen snapshot.
    schedule = None
    if fit_window == "rolling_trailing":
        weeks_back = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
        # Runs through EVERY week, because production re-fits every week and
        # a forward-time replay (shadow, the DP walk) should see exactly that:
        # at week k the factors were fit on weeks < k, so there is no
        # look-ahead to leak. What must NOT read its own graded rows is the
        # launch gate, and that is handled where the question is asked --
        # `freeze_calibration_from(gate_start)` prices the graded window off
        # the anchor instead. Bounding the artifact here would answer the gate
        # correctly and make shadow silently stop mirroring production.
        scope = population(pre_launch(d, cfg), cfg).copy()
        by_week, coverage = {}, []
        wk = pd.to_datetime(scope.date).dt.to_period("W")
        for w in sorted(wk.unique()):
            lo = w.start_time - pd.Timedelta(weeks=weeks_back)
            window = scope[(wk.dt.start_time >= lo)
                           & (wk.dt.start_time < w.start_time)]
            if not len(window):
                continue
            # PARTIAL WINDOW: the first weeks of an extract have less history
            # behind them than the trailing length asks for. They still fit
            # (a big extract clears min_anchor on one week), but a factor fit
            # on 1 of 4 intended weeks is noisier than its label claims, so
            # it is counted rather than passed off as a full-window fit.
            weeks_seen = int(wk[(wk.dt.start_time >= lo)
                                & (wk.dt.start_time < w.start_time)].nunique())
            fitted = _solve_level_factors(
                window.copy(), cfg, model, grain, k_shrink, min_anchor,
                tier_step, max_k, r_lookup if censored_basis else None)
            if fitted is None:                # too thin: hold 1.0, say so
                coverage.append({"week": str(w.start_time.date()),
                                 "fitted": False})
                continue
            by_week[str(w.start_time.date())] = fitted[0]
            coverage.append({"week": str(w.start_time.date()), "fitted": True,
                             "fit_rows": int(len(window)),
                             "weeks_in_window": weeks_seen,
                             "partial": weeks_seen < weeks_back})
        schedule = {
            "mode": "rolling_trailing",
            "trailing_weeks": weeks_back,
            # where the LAUNCH GATE freezes. The schedule itself runs past
            # this (production re-fits weekly, and a forward replay should see
            # that), but a factor fit inside the graded window has read the
            # rows it is graded on -- so the gate calls
            # freeze_calibration_from(this) and prices the hold-out off
            # `factors`, the anchor fit on the trailing window ending here.
            "gate_freezes_at": str(gate_start.date()),
            "anchor_fit_window": [str((gate_start - pd.Timedelta(
                weeks=weeks_back)).date()), str(gate_start.date())],
            "week_key": "ISO week start (Monday) the factors APPLY to; they "
                        "are fit on the trailing window ENDING STRICTLY "
                        "BEFORE it, so no row sees a factor fitted on its own "
                        "week or later",
            "weeks_fitted": sum(1 for c in coverage if c["fitted"]),
            "weeks_unfitted_held_at_1": [c["week"] for c in coverage
                                         if not c["fitted"]],
            # fitted on LESS history than trailing_weeks asks for -- the start
            # of the extract, where there is not yet a full window behind each
            # week. Noisier than the label; the gate window is normally far
            # from these, but say so rather than let "4w" mean 1w silently.
            "weeks_on_partial_window": [
                {"week": c["week"], "weeks_in_window": c["weeks_in_window"]}
                for c in coverage if c.get("partial")],
            "by_week": by_week,
        }

    path = cfg["baseline_model"]["calibration_factor_path"]
    with open(path, "w") as f:
        # factors clustered on 1.0 = model already level-correct; wide
        # dispersion = systematic per-cell bias, a training signal
        fv = np.array(list(factors.values()), dtype=float)
        payload = {"grain": grain,
                   "factor_summary": {
                       "p10": round(float(np.percentile(fv, 10)), 4),
                       "p50": round(float(np.percentile(fv, 50)), 4),
                       "p90": round(float(np.percentile(fv, 90)), 4),
                       "share_within_5pct_of_1": round(
                           float((np.abs(fv - 1.0) <= 0.05).mean()), 4),
                       "note": "clustered on 1.0 -> model is level-correct "
                               "and the factors are a near no-op; wide -> "
                               "systematic per-cell bias worth fixing in "
                               "training, not only in the multiplier",
                   },
                   "factors": factors,
                   # present only in rolling_trailing: the point-in-time
                   # schedule both harnesses read by row date. `factors`
                   # stays as the fallback for rows before the first fitted
                   # week and for any reader that predates the schedule.
                   "schedule": schedule,
                   "detail": detail,
                   "global_factor": round(float(f_global), 4),
                   "category_factors": cat_factors,
                   "shrinkage_units": k_shrink,
                   "fit_window": fit_window,
                   "fit_basis": "censored E[min(D,q)]" if censored_basis
                       else "raw mu (r_lookup missing)",
                   "fit_window_dates": [str(fit_dates.min().date()),
                                        str(fit_dates.max().date())],
                   "fit_rows": int(len(calib)),
                   "fit_in_sample_share": round(in_sample_share, 4),
                   "split": cfg["data"]["split"],
                   "basis": "anchor rows only; categories below "
                            "calibration_min_anchor_rows left at 1.0"}
        json.dump(stamp(payload, cfg, model.version,
                        "bootstrap.train_baseline --fit-calibration"),
                  f, indent=2)
    return factors


def check_calibration_convergence(d, cfg):
    """Has the calibration <-> dispersion fixed point settled?

    The chain is circular across runs: the factor solve consumes `r` (the
    censored basis), and `r`, `rho` and the prior are all fitted against
    CALIBRATED mu_ref. The pipeline breaks the cycle by iteration -- fit
    calibration, then re-fit prior and dispersion against it -- and that is
    only sound if one more turn of the loop would reproduce the factors.
    Nothing asserted it, so every downstream number silently depended on how
    many iterations happened to run.

    This re-solves the factors with the prior and r_lookup NOW on disk and
    compares per cell (and per schedule week) in log space against the
    artifact. It is a DRY RUN: the prior artifact is restored afterwards with
    a `convergence` block attached, so the chain on disk stays exactly the
    one prior and dispersion were fitted against -- committing the re-solve
    while they lag it would create the very inconsistency being tested for.
    NOT CONVERGED means: run --fit-calibration, estimate_prior and
    fit_dispersion once more, then check again.
    """
    path = cfg["baseline_model"]["calibration_factor_path"]
    with open(path) as f:
        old = json.load(f)
    tol = cfg["baseline_model"]["calibration_convergence_tol_log"]

    try:
        fit_level_calibration(d, cfg)          # iteration k+1, on disk briefly
        with open(path) as f:
            new = json.load(f)
    finally:
        with open(path, "w") as f:             # dry run: restore iteration k
            json.dump(old, f, indent=2)

    def compare(a, b, scope):
        worst, missing = (0.0, None), []
        for key in sorted(set(a) | set(b)):
            fa, fb = a.get(key), b.get(key)
            if not fa or not fb:
                missing.append(f"{scope}:{key}")
                continue
            dlog = abs(float(np.log(fb / fa)))
            if dlog > worst[0]:
                worst = (dlog, f"{scope}:{key}")
        return worst, missing

    worst, missing = compare(old.get("factors", {}),
                             new.get("factors", {}), "anchor")
    old_bw = (old.get("schedule") or {}).get("by_week", {})
    new_bw = (new.get("schedule") or {}).get("by_week", {})
    for wk in sorted(set(old_bw) | set(new_bw)):
        w, m = compare(old_bw.get(wk) or {}, new_bw.get(wk) or {}, wk)
        if w[0] > worst[0]:
            worst = w
        missing += m

    # WHAT IT WAS CHECKED AGAINST. A convergence verdict is only about the
    # artifacts in force when it ran: re-fit the prior or the dispersion and
    # the loop has turned again, so the stored verdict describes a chain that
    # no longer exists. Digests let `status` say so instead of showing a green
    # line for a check nobody re-ran (the same discipline as report vintages).
    from common.provenance import file_digest
    checked_against = {}
    for name, path_key in (("prior", ("posterior", "prior", "path")),
                           ("r_lookup", ("dispersion", "r_lookup_path")),
                           ("rho", ("dispersion", "rho_path"))):
        node = cfg
        for k in path_key:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, str) and os.path.exists(node):
            checked_against[name] = file_digest(node)

    # HOW MANY ANCHOR ROWS THE WORST CELL HAD. A cell whose factor is
    # shrinkage-dominated can post a large dlog on thin evidence, which reads
    # identically to an unsettled loop. Reported so the two can be told apart.
    worst_rows = None
    if worst[1]:
        cell = worst[1].split(":", 1)[1]
        worst_rows = ((old.get("detail") or {}).get(cell) or {}).get("anchor_rows")

    # THE TRAJECTORY, not just this turn. A contracting loop (2.3 -> 0.4 ->
    # 0.06 -> 0.006) is healthy and simply needs another turn; one that stalls
    # or oscillates is a different problem, and a single number cannot tell
    # them apart. Prior runs' readings are carried forward here.
    prior = ((old.get("convergence") or {}).get("history") or [])
    history = (prior + [round(worst[0], 6)])[-6:]

    converged = not missing and worst[0] <= tol
    block = {
        "tol_log": tol,
        "history": history,
        "worst_cell_anchor_rows": worst_rows,
        "checked_against": checked_against,
        "max_abs_dlog": round(worst[0], 6),
        "worst_cell": worst[1],
        "cells_appeared_or_gone": missing,
        "converged": converged,
        "method": "re-solved with the prior and r_lookup now on disk; "
                  "artifact restored (dry run)",
        "verdict": (
            "CONVERGED -- one more iteration reproduces the factors within "
            "tolerance; the calibration <-> dispersion loop has settled"
            if converged else
            "NOT CONVERGED -- the factors move {:.1%} (> {:.1%}) under the "
            "current prior/r{}. Run --fit-calibration, estimate_prior and "
            "fit_dispersion once more, then re-check.{}".format(
                worst[0], tol,
                f", worst cell on {worst_rows:,} anchor rows"
                if worst_rows else "",
                (" Trajectory " + " -> ".join(f"{h:.4f}" for h in history)
                 + (" is contracting: keep going, several turns is normal."
                    if len(history) > 1 and history[-1] < history[-2] else
                    " is NOT contracting -- investigate before iterating again."))
                if len(history) > 1 else
                " Several turns from a bare chain is normal.")),
    }
    old["convergence"] = block
    with open(path, "w") as f:
        json.dump(old, f, indent=2)
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fit-calibration", action="store_true",
                    help="fit the section 9.3 per-category level-calibration "
                         "factors from the already-trained baseline and the "
                         "section 9.5 prior, instead of training")
    ap.add_argument("--check-convergence", action="store_true",
                    help="dry-run re-solve of the calibration factors with "
                         "the prior/r now on disk; verifies the calibration "
                         "<-> dispersion fixed point has settled")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)

    if args.check_convergence:
        block = check_calibration_convergence(d, cfg)
        print(f"max |dlog f| = {block['max_abs_dlog']:.4f} "
              f"(tol {block['tol_log']}) at {block['worst_cell']}"
              + (f" ({block['worst_cell_anchor_rows']:,} anchor rows)"
                 if block.get("worst_cell_anchor_rows") else ""))
        if len(block.get("history") or []) > 1:
            print("trajectory   : "
                  + " -> ".join(f"{h:.4f}" for h in block["history"]))
        if block["cells_appeared_or_gone"]:
            print(f"cells appeared/disappeared: "
                  f"{block['cells_appeared_or_gone']}")
        print(block["verdict"])
        return

    if args.fit_calibration:
        factors = fit_level_calibration(d, cfg)
        with open(cfg["baseline_model"]["calibration_factor_path"]) as f:
            art = json.load(f)
        detail = art["detail"]
        print(f"grain: {art['grain']}  ({len(factors)} cells, global factor "
              f"{art['global_factor']:.4f})")
        print(f"fit window: {art['fit_window']} "
              f"{art['fit_window_dates'][0]}..{art['fit_window_dates'][1]} "
              f"({art['fit_rows']:,} rows, basis {art['fit_basis']})")
        if art["fit_in_sample_share"] > 0.5:
            print(f"WARNING: {art['fit_in_sample_share']:.0%} of the fit "
                  "window is inside the training period -- the model fits "
                  "there by construction, so the factor will understate what "
                  "the launch-adjacent weeks need. Move the split so the "
                  "trailing window sits after train_end.")
        widest = sorted(factors.items(), key=lambda kv: -abs(kv[1] - 1.0))[:12]
        for key, factor in widest:
            info = detail[key]
            print(f"  {key:26s} {factor:.4f}  (raw {info['raw_factor']:.4f} "
                  f"-> parent {info['parent_factor']:.4f}, self-weight "
                  f"{info['shrinkage_weight_on_self']:.2f}, "
                  f"{info['anchor_rows']:,} rows)")
        if len(factors) > len(widest):
            print(f"  ... {len(factors) - len(widest)} more cells nearer 1.0")
        below = [k for k, v in factors.items() if v < 1.0]
        if below:
            print(f"{len(below)}/{len(factors)} cells below 1.0 (model "
                  "over-predicts there) -- investigate (AGENTS rule 5); "
                  "factors are always applied")
        print(f"wrote {cfg['baseline_model']['calibration_factor_path']}")
        print("factors are applied automatically (apply_level_calibration "
              "is always true); re-run the backtest WITHOUT retraining to "
              "see the level diagnostic move (design 9.2)")
        return

    schema = train(d, cfg)
    print(f"trained {schema['model_version']} on {schema['train_rows']:,} rows")
    print(f"wrote {cfg['baseline_model']['model_path']} and "
          f"{cfg['baseline_model']['feature_schema_path']}")


if __name__ == "__main__":
    main()
