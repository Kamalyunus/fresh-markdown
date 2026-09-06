"""fit.train_baseline -- fit and freeze the reference-demand model (design 5.4).

LightGBM/Tweedie on units_sold, predicting only at the reference discount
(price features overwritten to d_ref at inference). Also fits the
level-calibration factors, which are always applied.
Run: python3 -m fit.train_baseline --input data/prepared.parquet [--fit-calibration]
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import load_config, reference_discount
from common.io import read_json, write_json
from common.provenance import stamp
from common import episodes
from fit.prepare_data import population, pre_launch, split_frames
from engine.demand import expected_min_demand_inventory_vec

# Feature order is authoritative in feature_schema.json. `total_discount` is
# the single price feature, overwritten to d_ref at inference. Deliberately
# absent: hours_remaining, lag sales, inventory (they leak price response or
# belong to the DP state -- AGENTS rule 12).
FEATURES = ["category", "subcategory", "fc", "hour_of_day", "dow",
            "day_of_month", "original_price",
            "sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate",
            "total_discount"]
CATEGORICAL = ["category", "subcategory", "fc"]
PRICE_FEATURES = ["total_discount"]

GRAIN = "subcategory"        # level-factor grain (settled; category is parent)


def add_derived(d):
    d = d.copy()
    dates = pd.to_datetime(d.date)
    d["dow"] = dates.dt.dayofweek
    d["day_of_month"] = dates.dt.day
    return d


def encode_features(d, features, categorical, levels):
    """The ONE feature matrix: categoricals as codes over `levels` (unseen ->
    -1), everything else numeric. Shared by training and inference so the two
    cannot encode a column differently."""
    X = pd.DataFrame(index=d.index)
    for feat in features:
        if feat in categorical:
            values = d[feat].astype(str)
            # an unseen level is masked to NaN first: same code (-1) as
            # before, without the Categorical deprecation for out-of-category
            # values that pandas will turn into an error
            known = values.where(values.isin(levels[feat]))
            X[feat] = pd.Categorical(known, categories=levels[feat]).codes
        else:
            X[feat] = pd.to_numeric(d[feat])
    return X


class BaselineModel:
    """Frozen mu_ref predictor. Loads model + schema + calibration artifacts."""

    # class-level so an applier built without __init__ (the tests' __new__
    # path) prices unfrozen with no gate; the instance sets both
    calibration_stops_at = None
    _freeze_from = None

    def __init__(self, cfg):
        bm = cfg["baseline_model"]
        self.cfg = cfg
        self.booster = lgb.Booster(model_file=bm["model_path"])
        with open(bm["feature_schema_path"]) as f:
            self.schema = json.load(f)
        self.calibration, self.calibration_grain = {}, GRAIN
        # week-keyed factors applied by ROW DATE (no row priced by its own
        # week's fit); `calibration` is the frozen-anchor fallback
        self.calibration_schedule = None
        self.calibration_stops_at = None
        if os.path.exists(bm["calibration_factor_path"]):
            with open(bm["calibration_factor_path"]) as f:
                cal = json.load(f)
            self.calibration = cal.get("factors", {})
            self.calibration_grain = cal.get("grain", GRAIN)
            sched = cal.get("schedule")
            if sched and sched.get("by_week"):
                self.calibration_schedule = sched["by_week"]
                self.calibration_stops_at = sched.get("gate_freezes_at")
        self._reset_calibration_counters()
        self.version = self.schema["model_version"]

    def _reset_calibration_counters(self):
        self._cal_rows_scheduled = 0
        self._cal_rows_fallback = 0
        self._cal_rows_frozen = 0
        self._cal_rows_static = 0
        self._cal_fallback_weeks = set()

    def freeze_calibration_from(self, date):
        """Price rows on/after `date` with the frozen anchor instead of the
        weekly schedule (None restores it). The schedule mirrors production
        (weekly re-fit, no look-ahead); the LAUNCH GATE must instead grade the
        artifact as frozen, because a factor re-fit inside the hold-out has
        read the rows it is graded on."""
        self._freeze_from = pd.Timestamp(date) if date is not None else None
        return self

    def level_factors(self, d):
        """Per-row level factor, the one `predict_mu_ref` applies: each row
        takes the factors in force for ITS week; unfitted weeks fall back to
        the frozen anchor, never forward. Public because a caller may need
        the factor itself (a rescale between two freezes is exact, so the
        backtest's weekly-refit reading never predicts twice)."""
        keys = d[self.calibration_grain].astype(str)
        anchor = keys.map(lambda key: self.calibration.get(key, 1.0)).to_numpy()
        if self.calibration_schedule is None:
            self._cal_rows_static += len(d)
            return anchor
        dates = pd.to_datetime(d["date"])
        weeks = episodes.week_key(dates).to_numpy()
        frozen = ((dates >= self._freeze_from).to_numpy()
                  if self._freeze_from is not None else np.zeros(len(d), bool))
        out = anchor.copy()                  # frozen rows keep the anchor
        self._cal_rows_frozen += int(frozen.sum())
        # one pass per distinct week, not per row: rows of a week share a table
        for wk in np.unique(weeks[~frozen]):
            rows = (weeks == wk) & ~frozen
            table = self.calibration_schedule.get(wk)
            if table is None:                # unfitted week: the anchor, above
                self._cal_rows_fallback += int(rows.sum())
                self._cal_fallback_weeks.add(str(wk))
                continue
            self._cal_rows_scheduled += int(rows.sum())
            out[rows] = keys[rows].map(lambda key: table.get(key, 1.0)).to_numpy()
        return out

    # the pre-rename name: tests/conftest.py's applier still calls it
    _factor_vector = level_factors

    def calibration_coverage(self):
        """Which priced rows got a point-in-time factor. Three anchor cases,
        never conflated: DELIBERATE (freeze_calibration_from -- the gate),
        BEFORE THE START (first trailing window not yet closed), and PAST THE
        END (stale factors in production -- the only problem case)."""
        if self.calibration_schedule is None:
            return {"mode": "static", "rows": self._cal_rows_static,
                    "note": "no schedule in the artifact: one frozen factor "
                            "set applied to every row",
                    "verdict": "OK -- static: one frozen factor set applied "
                               "to every priced row (no schedule to fall "
                               "behind)"}
        priced = (self._cal_rows_scheduled + self._cal_rows_fallback
                  + self._cal_rows_frozen)
        weeks = sorted(self.calibration_schedule)
        past_end = sorted(w for w in self._cal_fallback_weeks
                          if weeks and w > weeks[-1])
        share = self._cal_rows_fallback / max(priced, 1)
        if past_end:
            verdict = ("STALE FACTORS IN USE -- {} rows ({:.1%}) are in weeks "
                       "PAST the end of the schedule and fell back to the "
                       "frozen set. Re-run `train_baseline --fit-calibration`."
                       .format(self._cal_rows_fallback, share))
        elif self._freeze_from is not None:
            verdict = ("OK -- {} rows frozen at the anchor from {} on purpose "
                       "(the launch gate); every other priced row took its own "
                       "week's factors".format(self._cal_rows_frozen,
                                               self._freeze_from.date()))
        elif not self._cal_rows_fallback:
            verdict = "OK -- every priced row took its own week's factors"
        else:
            verdict = ("OK -- {} rows ({:.1%}) fell back before the schedule "
                       "opens".format(self._cal_rows_fallback, share))
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
            "verdict": verdict,
        }

    def _matrix(self, d):
        missing = [f for f in self.schema["features"]
                   if f not in d.columns and f not in ("dow", "day_of_month")]
        if missing:
            raise KeyError(
                f"frame is missing feature columns {missing} -- re-run "
                "fit.prepare_data")
        return encode_features(d, self.schema["features"],
                               self.schema["categorical"],
                               self.schema["category_levels"])

    def predict_mu_ref(self, d, raw=False):
        """mu_ref(context); price features overwritten to d_ref. `raw=True`
        skips the level factors (used only while fitting them)."""
        d = add_derived(d)
        for feat in self.schema["price_features"]:
            d[feat] = d["d_ref"] if "d_ref" in d.columns else d["category"].map(
                lambda c: reference_discount(self.cfg, c))
        mu = self.booster.predict(self._matrix(d))
        mu = np.clip(mu, self.cfg["pricing"]["demand_floor"], None)
        if not raw:
            mu = mu * self.level_factors(d)
        return mu


def train(d, cfg):
    bm = cfg["baseline_model"]
    splits = split_frames(d, cfg)
    train_d = add_derived(population(splits["train"], cfg))

    levels = {c: sorted(train_d[c].astype(str).unique().tolist()) for c in CATEGORICAL}
    X = encode_features(train_d, FEATURES, CATEGORICAL, levels)

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
        "frozen": True,
    }
    with open(bm["feature_schema_path"], "w") as f:
        json.dump(schema, f, indent=2)
    return schema


def attach_fit_basis(frame, model, r_lookup):
    """The two per-row inputs a level solve reads, attached in place: the
    RAW mu_ref (`mu_ref_hat`) and, on the censored basis, each row's r
    (`r_val`). Neither depends on the window, so a schedule attaches them to
    its whole scope once and slices, instead of predicting per week."""
    from fit.fit_dispersion import lookup_r_vec   # local: avoids a cycle
    frame["mu_ref_hat"] = model.predict_mu_ref(frame, raw=True)
    if r_lookup is not None:
        frame["r_val"] = lookup_r_vec(r_lookup, frame.subcategory,
                                      frame.category)
    return frame


def schedule_reaches(schedule):
    """The last week the factor schedule COVERS: a week it fitted, or one it
    judged too thin and deliberately holds at the frozen anchor
    (`weeks_unfitted_held_at_1`; `level_factors` applies the anchor there).
    None when there is no schedule. The ONE reading `daily.update`'s
    calibration_current gate and `ops.advance`'s re-fit trigger share --
    reading `by_week` alone made a thin week look like a missed cron: the
    gate refused every --apply and advance re-fit every morning."""
    if not schedule:
        return None
    weeks = list(schedule.get("by_week") or {}) + list(
        schedule.get("weeks_unfitted_held_at_1") or [])
    return max(weeks) if weeks else None


def pinned_cells(detail):
    """{cell: bracket end} for every cell of a detail table whose own solve
    pinned (rule 3)."""
    return {k: v["at_bound"] for k, v in detail.items() if v.get("at_bound")}


def _solve_level_factors(calib, model, k_shrink, min_anchor,
                         tier_step, max_k, r_lookup, predicted=False):
    """Factors for one fit window (shared by the anchor fit and every schedule
    week). Returns (factors, detail, global_factor, global_at_bound,
    detail_category), or None when the window holds too few anchor rows --
    the caller holds those weeks at 1.0. Cells ABOVE that floor are shrunk
    toward their parent (category, then global) by `k_shrink` pseudo-units.
    A bound is not a solve, at any level (rule 3): a cell whose bisection
    ran off the bracket carries `at_bound` in its detail; a subcategory
    whose PARENT category pinned carries `parent_at_bound` (it is shrunk
    toward a bracket end, however thin it is); `global_at_bound` names the
    end the GLOBAL solve pinned to (None when it converged). `predicted`
    says `attach_fit_basis` already ran on `calib` (a schedule attaches once
    to its scope); otherwise it runs here, on `calib` in place."""
    bm = model.cfg["baseline_model"]
    f_lo, f_hi = (float(x) for x in bm["calibration_factor_search_bounds"])
    halvings = int(bm["calibration_factor_bisection_steps"])

    if not predicted:
        attach_fit_basis(calib, model, r_lookup)

    def solve_factor(anchor):
        """(factor, predicted at f=1, at_bound): solved against the censored
        basis E[min(D,q)] -- the gate's quantity. `at_bound` names the bracket
        end the solve was pinned to (None when the bisection converged)."""
        sold = float(anchor["units_sold"].sum())
        mu = anchor["mu_ref_hat"].to_numpy()
        if r_lookup is None:
            pred = float(mu.sum())
            return (sold / pred if pred > 0 else 1.0), pred, None
        r = anchor["r_val"].to_numpy()
        q = anchor["starting_inventory"].to_numpy()

        def predicted(f):
            return float(expected_min_demand_inventory_vec(
                f * mu, r, q, max_k).sum())

        base = predicted(1.0)
        if base <= 0 or sold <= 0:
            return 1.0, base, None
        lo, hi = f_lo, f_hi
        if predicted(lo) > sold:
            return lo, base, "lower"
        if predicted(hi) < sold:
            return hi, base, "upper"
        for _ in range(halvings):        # monotone in f
            mid = (lo + hi) / 2
            if predicted(mid) < sold:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2, base, None

    def shrink(cell, parent, evidence):
        if evidence <= 0 or cell <= 0:
            return parent
        w = evidence / (evidence + k_shrink)
        return float(np.exp(w * np.log(cell) + (1 - w) * np.log(parent)))

    anchor_all = calib[episodes.is_anchor_row(calib, tier_step)]
    if len(anchor_all) < min_anchor or anchor_all["mu_ref_hat"].sum() <= 0:
        return None

    f_global, _, global_at_bound = solve_factor(anchor_all)

    def fit_level(groups, parent_of):
        """`parent_of(key, g)` -> (parent factor, the bracket end the
        parent's OWN solve pinned to, or None)."""
        out, det = {}, {}
        for key, g in groups:
            raw_f, pred, at_bound = solve_factor(g)
            evidence = float(g["units_sold"].sum())
            parent, parent_at_bound = parent_of(key, g)
            f = shrink(raw_f, parent, evidence)
            out[str(key)] = round(float(f), 4)
            det[str(key)] = {
                "anchor_rows": int(len(g)),
                "anchor_sold": int(evidence),
                "anchor_predicted_at_f1": round(float(pred), 1),
                "raw_factor": round(float(raw_f), 4),
                "parent_factor": round(float(parent), 4),
                "shrinkage_weight_on_self": round(
                    float(evidence / (evidence + k_shrink)), 3),
            }
            if at_bound:
                # the literal bracket end, not a solve: the sales this cell
                # wants sit outside [f_lo, f_hi] x its prediction
                det[str(key)]["at_bound"] = at_bound
                det[str(key)]["at_bound_note"] = (
                    f"raw_factor is the {at_bound} end of "
                    f"calibration_factor_search_bounds {[f_lo, f_hi]}, not a "
                    "solved value -- the bisection bracket does not contain "
                    "the sold total. Investigate the cell before trusting it.")
            if parent_at_bound:
                # the parent this cell is shrunk toward is itself a bracket
                # end: the thinner the cell, the more of its factor is bound
                det[str(key)]["parent_at_bound"] = parent_at_bound
        return out, det

    cat_factors, cat_detail = fit_level(
        anchor_all.groupby("category"),
        lambda k, g: (f_global, global_at_bound))

    def parent_of_sub(key, g):
        cat = str(g["category"].iloc[0])
        if cat not in cat_factors:
            return f_global, global_at_bound
        return cat_factors[cat], cat_detail[cat].get("at_bound")

    factors, detail = fit_level(anchor_all.groupby("subcategory"),
                                parent_of_sub)
    return factors, detail, f_global, global_at_bound, cat_detail


def fit_level_calibration(d, cfg):
    """Multiplicative level factors on ANCHOR ROWS only (elasticity ~1 there,
    so slope error cannot leak in). Each cell is shrunk toward its parent
    (category, then global) by `calibration_shrinkage_units` pseudo-units --
    a thin cell follows its parent, it is not held at 1.0; only a WINDOW
    with fewer than `calibration_min_anchor_rows` anchor rows is unfitted.
    The anchor set is the trailing W weeks ending at the gate window's start
    -- disjoint from what the gate grades -- plus a weekly point-in-time
    schedule fit on the trailing window ending strictly before each week."""

    model = BaselineModel(cfg)
    split = cfg["data"]["split"]
    gate_start = pd.Timestamp(split["test_start"])   # gate window = test
    weeks_back = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
    lo = gate_start - pd.Timedelta(weeks=weeks_back)
    # SAME population and SAME cut as the weekly schedule below (rules 14/15):
    # the frozen fallback and the by-week factors must be solved on the same
    # rows, since check_calibration_convergence compares them cell by cell
    calib = episodes.window_slice(population(d, cfg),
                                  lo.strftime("%Y-%m-%d"),
                                  (gate_start - pd.Timedelta(days=1))
                                  .strftime("%Y-%m-%d")).copy()
    if not len(calib):
        raise RuntimeError("calibration fit window contains no rows")

    fit_dates = pd.to_datetime(calib.date)
    train_end = pd.Timestamp(split["train_end"])
    in_sample_share = float((fit_dates <= train_end).mean())
    tier_step = cfg["pricing"]["tier_step"]
    max_k = cfg["pricing"]["negbin_max_k"]

    r_lookup = read_json(cfg["dispersion"]["r_lookup_path"])
    censored_basis = r_lookup is not None

    min_anchor = cfg["baseline_model"]["calibration_min_anchor_rows"]
    k_shrink = cfg["baseline_model"]["calibration_shrinkage_units"]

    fitted = _solve_level_factors(calib, model, k_shrink, min_anchor,
                                  tier_step, max_k, r_lookup)
    if fitted is None:
        anchors = int(episodes.is_anchor_row(calib, tier_step).sum())
        raise RuntimeError(
            f"fit window has only {anchors} anchor rows (need {min_anchor})"
            " -- widen calibration_fit_trailing_weeks")
    factors, detail, f_global, global_at_bound, detail_category = fitted

    # POINT-IN-TIME schedule. Pre-launch: over every pre-launch week (a
    # forward replay must see exactly what production re-fits weekly; the
    # gate freezes via freeze_calibration_from, not by bounding the artifact).
    # After `data.launch_date` is set: THIS is the weekly cron, so the
    # schedule runs through the latest data and one week past it -- the
    # week being priced, whose own rows are not complete yet -- while every
    # sealed fit keeps its pre-launch scope. Moving split.test_end instead
    # would rescope every other fit.
    launched = bool(cfg["data"].get("launch_date"))
    scope = population(d if launched else pre_launch(d, cfg), cfg).copy()
    weeks = sorted(episodes.week_key(scope.date).unique())
    if launched and weeks:
        weeks.append((episodes.week_start(weeks[-1]) + pd.Timedelta(days=7))
                     .strftime("%Y-%m-%d"))
    by_week, coverage, pinned_by_week = {}, [], {}
    opened = episodes.opening_dates(scope)     # once, not once per week
    attach_fit_basis(scope, model, r_lookup)   # once: the windows are slices
    for w in weeks:
        window, weeks_seen = episodes.trailing_weeks_window(
            scope, w, weeks_back, opened=opened)
        if not len(window):
            continue
        f = _solve_level_factors(window, model, k_shrink, min_anchor,
                                 tier_step, max_k, r_lookup, predicted=True)
        if f is None:                       # too thin: hold 1.0, say so
            coverage.append({"week": w, "fitted": False})
            continue
        by_week[w] = f[0]
        # the week's pinned cells, by level -- the per-cell detail itself is
        # the anchor fit's; a week keeps only what rule 3 needs
        pinned = {**{f"category:{k}": v for k, v in pinned_cells(f[4]).items()},
                  **{f"subcategory:{k}": v
                     for k, v in pinned_cells(f[1]).items()}}
        if pinned:
            pinned_by_week[w] = pinned
        coverage.append({"week": w, "fitted": True,
                         "fit_rows": int(len(window)),
                         "weeks_in_window": weeks_seen,
                         "partial": weeks_seen < weeks_back,
                         "global_at_bound": f[3]})
    schedule = {
        "mode": "rolling_trailing",
        "scope": (f"production -- launch_date {cfg['data']['launch_date']}; "
                  "through the latest data plus the week being priced"
                  if launched else
                  f"pre-launch -- through split.test_end {split['test_end']}"),
        "trailing_weeks": weeks_back,
        "gate_freezes_at": str(gate_start.date()),
        # the frozen anchor's window, both ends INCLUSIVE (the day before
        # the gate opens is its last day)
        "anchor_fit_window": [str(lo.date()),
                              str((gate_start - pd.Timedelta(days=1)).date())],
        "week_key": "ISO week start the factors APPLY to; fit on the "
                    "trailing window ending strictly before it",
        "weeks_fitted": sum(1 for c in coverage if c["fitted"]),
        "weeks_unfitted_held_at_1": [c["week"] for c in coverage
                                     if not c["fitted"]],
        # fit on less history than trailing_weeks asks for (extract start)
        "weeks_on_partial_window": [
            {"week": c["week"], "weeks_in_window": c["weeks_in_window"]}
            for c in coverage if c.get("partial")],
        # weeks whose GLOBAL solve pinned at a bracket end: every cell's
        # parent that week is a bound, not a solve (rule 3)
        "weeks_global_at_bound": {c["week"]: c["global_at_bound"]
                                  for c in coverage
                                  if c.get("global_at_bound")},
        # {week: {"level:cell": bracket end}} for the weeks with a pinned
        # cell -- the schedule's rule-3 flags, since by_week keeps factors only
        "pinned_by_week": pinned_by_week,
        "by_week": by_week,
    }

    # every pinned solve in the artifact, anchor and schedule, in one list
    # (ops.status reads this field alone): {scope, level, cell, at_bound}
    pins = ([{"scope": "anchor", "level": "category", "cell": k, "at_bound": v}
             for k, v in sorted(pinned_cells(detail_category).items())]
            + [{"scope": "anchor", "level": "subcategory", "cell": k,
                "at_bound": v} for k, v in sorted(pinned_cells(detail).items())]
            + [{"scope": w, "level": lc.split(":", 1)[0],
                "cell": lc.split(":", 1)[1], "at_bound": v}
               for w, cells in sorted(pinned_by_week.items())
               for lc, v in sorted(cells.items())])

    fv = np.array(list(factors.values()), dtype=float)
    payload = {"grain": GRAIN,
               "factor_summary": {
                   "p10": round(float(np.percentile(fv, 10)), 4),
                   "p50": round(float(np.percentile(fv, 50)), 4),
                   "p90": round(float(np.percentile(fv, 90)), 4),
                   "share_within_5pct_of_1": round(
                       float((np.abs(fv - 1.0) <= 0.05).mean()), 4),
                   "note": "clustered on 1.0 -> model is level-correct; "
                           "wide -> systematic per-cell bias worth fixing "
                           "in training",
               },
               "factors": factors,
               "schedule": schedule,
               "detail": detail,
               # the parent level's own solves: a subcategory shrinks toward
               # its category's factor, so a category pinned here is a bound
               # under every thin subcategory it holds (parent_at_bound)
               "detail_category": detail_category,
               "pinned_cells": pins,
               "global_factor": round(float(f_global), 4),
               # None, or the bracket end ("lower"/"upper") the global solve
               # pinned to: then the parent every category shrinks toward is
               # a bound, not a solve (rule 3)
               "global_factor_at_bound": global_at_bound,
               "shrinkage_units": k_shrink,
               # the frozen anchor is ONE trailing window ending the day
               # before the gate opens; the rolling re-fit is `schedule`
               "fit_window": "trailing_before_gate",
               "fit_basis": "censored E[min(D,q)]" if censored_basis
                   else "raw mu (r_lookup missing)",
               "fit_window_dates": [str(fit_dates.min().date()),
                                    str(fit_dates.max().date())],
               "fit_rows": int(len(calib)),
               "fit_in_sample_share": round(in_sample_share, 4),
               "split": split,
               "basis": ("anchor rows only; every cell shrunk toward its "
                         "parent (category, then global) by "
                         "calibration_shrinkage_units; a window below "
                         "calibration_min_anchor_rows is unfitted (held at "
                         "the frozen anchor); a cell whose bisection pinned "
                         "at calibration_factor_search_bounds carries "
                         "at_bound in detail, one whose parent pinned "
                         "parent_at_bound; pinned_cells lists every pin, "
                         "anchor and schedule")}
    path = cfg["baseline_model"]["calibration_factor_path"]
    # The weekly PRODUCTION re-fit (launch_date set) retrains nothing: the
    # prior, r and rho it was checked against are the ones on disk, so the
    # convergence verdict still holds and is carried forward -- written
    # without it, ops.tune read "never checked" and BLOCKED the daily lane
    # after every cron. The bootstrap path is unchanged: its verdict comes
    # from the --check-convergence that follows each fit.
    previous = read_json(path) or {}
    if launched and previous.get("convergence"):
        conv = dict(previous["convergence"])
        conv["carried_from"] = (
            conv.get("carried_from")
            or (previous.get("provenance") or {}).get("created_at"))
        conv["carried_note"] = (
            "carried forward from the previous artifact by the weekly "
            "production re-fit, which moves no prior/r/rho; status still "
            "reads checked_against against the artifacts on disk")
        payload["convergence"] = conv
    write_json(path, stamp(payload, cfg, model.version,
                           "fit.train_baseline --fit-calibration"))
    return factors


def check_calibration_convergence(d, cfg, commit=False):
    """Has the calibration <-> dispersion fixed point settled?

    The chain is circular: the factor solve consumes `r`, while `r`, `rho`
    and the prior are fitted against CALIBRATED mu_ref. This re-solves the
    factors with the prior/r now on disk and compares per cell (and per
    schedule week) in log space. Default is a DRY RUN (artifact restored);
    `commit` keeps the re-solve, which is bit-for-bit the next turn's
    --fit-calibration -- only sound inside the loop.
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
        if not commit:
            with open(path, "w") as f:         # dry run: restore iteration k
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

    # digests of what the verdict was checked against, so `status` can flag
    # a verdict whose chain has since moved -- read through the one artifact
    # walk in common.provenance
    from common.provenance import collect
    checked_against = {row["artifact"]: row["sha256"]
                       for row in collect(cfg)
                       if row["artifact"] in ("prior", "r_lookup", "rho")
                       and row["present"]}

    # anchor rows behind the worst cell IN THE FROZEN ANCHOR FIT: a thin,
    # shrinkage-dominated cell reads identically to an unsettled loop unless
    # the row count is shown. The per-week detail is not kept, so a worst
    # cell in a schedule week is sized by its anchor-fit count, and labelled
    # as such
    worst_rows = None
    if worst[1]:
        cell = worst[1].split(":", 1)[1]
        worst_rows = ((old.get("detail") or {}).get(cell) or {}).get("anchor_rows")

    # trajectory (carried across runs): contracting vs stalled/oscillating
    prior = ((old.get("convergence") or {}).get("history") or [])
    history = (prior + [round(worst[0], 6)])[-6:]

    converged = not missing and worst[0] <= tol
    block = {
        "tol_log": tol,
        "history": history,
        "worst_cell_anchor_rows": worst_rows,
        "worst_cell_anchor_rows_basis": (
            "the cell's anchor rows in the FROZEN ANCHOR fit, whatever "
            "scope the worst cell is in (per-week cell counts are not kept)"),
        "checked_against": checked_against,
        "max_abs_dlog": round(worst[0], 6),
        "worst_cell": worst[1],
        "cells_appeared_or_gone": missing,
        "converged": converged,
        "method": "re-solved with the prior and r_lookup now on disk; "
                  + ("re-solve KEPT (--commit-convergence: this is the next "
                     "turn's --fit-calibration)" if commit else
                     "artifact restored (dry run)"),
        "verdict": (
            "CONVERGED -- one more iteration reproduces the factors within "
            "tolerance"
            if converged else
            "NOT CONVERGED -- the factors move {:.1%} (> {:.1%}) under the "
            "current prior/r{}. Run --fit-calibration, estimate_prior and "
            "fit_dispersion once more, then re-check.{}".format(
                worst[0], tol,
                f", worst cell on {worst_rows:,} anchor rows in the frozen "
                "anchor fit" if worst_rows else "",
                (" Trajectory " + " -> ".join(f"{h:.4f}" for h in history)
                 + (" is contracting: keep going, several turns is normal."
                    if len(history) > 1 and history[-1] < history[-2] else
                    " is NOT contracting -- investigate before iterating again."))
                if len(history) > 1 else
                " Several turns from a bare chain is normal.")),
    }
    keep = new if commit else old
    keep["convergence"] = block
    with open(path, "w") as f:
        json.dump(keep, f, indent=2)
    return block


def _describe_calibration(art, factors, widest_n=12):
    """The --fit-calibration console summary of a calibration artifact."""
    detail = art["detail"]
    lines = [f"grain: {art['grain']}  ({len(factors)} cells, global factor "
             f"{art['global_factor']:.4f}"
             + (f" AT {art['global_factor_at_bound'].upper()} BOUND -- not "
                "a solve" if art.get("global_factor_at_bound") else "")
             + ")",
             f"fit window: {art['fit_window']} "
             f"{art['fit_window_dates'][0]}..{art['fit_window_dates'][1]} "
             f"({art['fit_rows']:,} rows, basis {art['fit_basis']})"]
    if art["fit_in_sample_share"] > 0.5:
        lines.append(f"WARNING: {art['fit_in_sample_share']:.0%} of the fit "
                     "window is inside the training period -- the factor will "
                     "understate what launch-adjacent weeks need.")
    widest = sorted(factors.items(), key=lambda kv: -abs(kv[1] - 1.0))[:widest_n]
    for key, factor in widest:
        info = detail[key]
        lines.append(f"  {key:26s} {factor:.4f}  (raw {info['raw_factor']:.4f} "
                     f"-> parent {info['parent_factor']:.4f}, self-weight "
                     f"{info['shrinkage_weight_on_self']:.2f}, "
                     f"{info['anchor_rows']:,} rows"
                     + (f", AT {info['at_bound'].upper()} BOUND"
                        if info.get("at_bound") else "") + ")")
    if len(factors) > len(widest):
        lines.append(f"  ... {len(factors) - len(widest)} more cells nearer 1.0")
    pinned = sorted(pinned_cells(detail))
    if pinned:
        lines.append(f"{len(pinned)} cell(s) pinned at "
                     "calibration_factor_search_bounds -- a bound is not a "
                     "solve: " + ", ".join(pinned))
    parents = sorted(pinned_cells(art.get("detail_category") or {}))
    if parents:
        under = sorted(k for k, v in detail.items() if v.get("parent_at_bound"))
        lines.append(f"{len(parents)} CATEGORY solve(s) pinned -- the parent "
                     f"{len(under)} subcategory cell(s) shrink toward is a "
                     "bound: " + ", ".join(parents))
    weekly = (art.get("schedule") or {}).get("pinned_by_week") or {}
    if weekly:
        lines.append(f"{len(weekly)} schedule week(s) with a pinned cell "
                     "(schedule.pinned_by_week)")
    below = [k for k, v in factors.items() if v < 1.0]
    if below:
        lines.append(f"{len(below)}/{len(factors)} cells below 1.0 (model "
                     "over-predicts there) -- investigate (AGENTS rule 5)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fit-calibration", action="store_true",
                    help="fit the level-calibration factors (no retrain)")
    ap.add_argument("--check-convergence", action="store_true",
                    help="dry-run re-solve; has the f<->r loop settled?")
    ap.add_argument("--commit-convergence", action="store_true",
                    help="with --check-convergence, KEEP the re-solve "
                         "(loop use only: saves a full re-solve per turn)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)

    if args.check_convergence:
        block = check_calibration_convergence(
            d, cfg, commit=args.commit_convergence)
        print(f"max |dlog f| = {block['max_abs_dlog']:.4f} "
              f"(tol {block['tol_log']}) "
              + (f"at {block['worst_cell']}"
                 + (f" ({block['worst_cell_anchor_rows']:,} anchor rows)"
                    if block.get("worst_cell_anchor_rows") else "")
                 if block.get("worst_cell") is not None else
                 "-- the re-solve reproduces the artifact on disk exactly"))
        if len(block.get("history") or []) > 1:
            print("trajectory   : "
                  + " -> ".join(f"{h:.4f}" for h in block["history"]))
        if block["cells_appeared_or_gone"]:
            print(f"cells appeared/disappeared: "
                  f"{block['cells_appeared_or_gone']}")
        if block.get("worst_cell_anchor_rows"):
            print("  (row count = the cell's anchor rows in the frozen "
                  "anchor fit)")
        print(block["verdict"])
        return

    if args.fit_calibration:
        factors = fit_level_calibration(d, cfg)
        path = cfg["baseline_model"]["calibration_factor_path"]
        print(_describe_calibration(read_json(path), factors))
        print(f"wrote {path}")
        return

    schema = train(d, cfg)
    print(f"trained {schema['model_version']} on {schema['train_rows']:,} rows")
    print(f"wrote {cfg['baseline_model']['model_path']} and "
          f"{cfg['baseline_model']['feature_schema_path']}")


if __name__ == "__main__":
    main()
