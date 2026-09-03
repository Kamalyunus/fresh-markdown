"""bootstrap.train_baseline -- fit and freeze the reference-demand model (design 5.4).

LightGBM/Tweedie on units_sold, predicting only at the reference discount
(price features overwritten to d_ref at inference). Also fits the
level-calibration factors, which are always applied.
Run: python3 -m bootstrap.train_baseline --input data/prepared.parquet [--fit-calibration]
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
from bootstrap.prepare_data import population, pre_launch, split_frames
from pricing.demand import expected_min_demand_inventory_vec

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


class BaselineModel:
    """Frozen mu_ref predictor. Loads model + schema + calibration artifacts."""

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

    def _factor_vector(self, d):
        """Per-row level factor: each row takes the factors in force for ITS
        week; unfitted weeks fall back to the frozen anchor, never forward."""
        keys = d[self.calibration_grain].astype(str)
        if self.calibration_schedule is None:
            self._cal_rows_static += len(d)
            return keys.map(lambda k: self.calibration.get(k, 1.0)).to_numpy()
        dates = pd.to_datetime(d["date"])
        weeks = episodes.week_key(dates)
        frozen = (dates >= self._freeze_from).to_numpy() \
            if getattr(self, "_freeze_from", None) is not None \
            else np.zeros(len(d), bool)
        out = np.ones(len(d))
        for i, (wk, key) in enumerate(zip(weeks.to_numpy(), keys.to_numpy())):
            if frozen[i]:
                self._cal_rows_frozen += 1
                out[i] = self.calibration.get(key, 1.0)
            elif (table := self.calibration_schedule.get(wk)) is None:
                self._cal_rows_fallback += 1
                self._cal_fallback_weeks.add(str(wk))
                out[i] = self.calibration.get(key, 1.0)
            else:
                self._cal_rows_scheduled += 1
                out[i] = table.get(key, 1.0)
        return out

    def calibration_coverage(self):
        """Which priced rows got a point-in-time factor. Three anchor cases,
        never conflated: DELIBERATE (freeze_calibration_from -- the gate),
        BEFORE THE START (first trailing window not yet closed), and PAST THE
        END (stale factors in production -- the only problem case)."""
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
                "Re-run `train_baseline --fit-calibration`.".format(
                    self._cal_rows_fallback, share)
                if past_end else
                "OK -- {} rows frozen at the anchor from {} on purpose (the "
                "launch gate); every other priced row took its own week's "
                "factors".format(self._cal_rows_frozen,
                                 self._freeze_from.date())
                if self._freeze_from is not None else
                "OK -- every priced row took its own week's factors"
                if not self._cal_rows_fallback else
                "OK -- {} rows ({:.1%}) fell back before the schedule "
                "opens".format(self._cal_rows_fallback, share)),
        }

    def _matrix(self, d):
        missing = [f for f in self.schema["features"]
                   if f not in d.columns and f not in ("dow", "day_of_month")]
        if missing:
            raise KeyError(
                f"frame is missing feature columns {missing} -- re-run "
                "bootstrap.prepare_data")
        X = pd.DataFrame(index=d.index)
        for feat in self.schema["features"]:
            if feat in self.schema["categorical"]:
                cats = self.schema["category_levels"][feat]
                X[feat] = pd.Categorical(d[feat].astype(str), categories=cats).codes
            else:
                X[feat] = pd.to_numeric(d[feat])
        return X

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
            mu = mu * self._factor_vector(d)
        return mu


def train(d, cfg):
    bm = cfg["baseline_model"]
    splits = split_frames(d, cfg)
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
        "frozen": True,
    }
    with open(bm["feature_schema_path"], "w") as f:
        json.dump(schema, f, indent=2)
    return schema


def _solve_level_factors(calib, model, k_shrink, min_anchor,
                         tier_step, max_k, r_lookup):
    """Factors for one fit window (shared by the anchor fit and every schedule
    week). Returns (factors, detail, global_factor), or None when the window
    holds too few anchor rows -- the caller holds those weeks at 1.0."""
    from bootstrap.fit_dispersion import lookup_r   # local: avoids a cycle

    calib["mu_ref_hat"] = model.predict_mu_ref(calib, raw=True)
    if r_lookup is not None:
        calib["r_val"] = [lookup_r(r_lookup, s, c)
                          for s, c in zip(calib.subcategory, calib.category)]

    def solve_factor(anchor):
        # solved against the censored basis E[min(D,q)] -- the gate's quantity
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
        for _ in range(20):        # monotone in f; 20 halvings ~ 1e-5
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

    anchor_all = calib[episodes.is_anchor_row(calib, tier_step)]
    if len(anchor_all) < min_anchor or anchor_all["mu_ref_hat"].sum() <= 0:
        return None

    f_global, _ = solve_factor(anchor_all)

    def fit_level(groups, parent_of):
        out, det = {}, {}
        for key, g in groups:
            raw_f, pred = solve_factor(g)
            evidence = float(g["units_sold"].sum())
            parent = parent_of(key, g)
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
        return out, det

    cat_factors, _ = fit_level(
        anchor_all.groupby("category"), lambda k, g: f_global)
    factors, detail = fit_level(
        anchor_all.groupby("subcategory"),
        lambda k, g: cat_factors.get(str(g["category"].iloc[0]), f_global))
    return factors, detail, f_global


def fit_level_calibration(d, cfg):
    """Multiplicative level factors on ANCHOR ROWS only (elasticity ~1 there,
    so slope error cannot leak in; thin cells stay 1.0). The anchor set is the
    trailing W weeks ending at the gate window's start -- disjoint from what
    the gate grades -- plus a weekly point-in-time schedule fit on the
    trailing window ending strictly before each week."""

    model = BaselineModel(cfg)
    split = cfg["data"]["split"]
    gate_start = pd.Timestamp(split["test_start"])   # gate window = test
    weeks_back = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
    lo = gate_start - pd.Timedelta(weeks=weeks_back)
    # SAME population and SAME cut as the weekly schedule below, which uses
    # population(pre_launch(d, cfg)) and window_slice. A row-level date cut on
    # the unfiltered frame put ineligible rows (final-hour restocks, unknown
    # outcomes) into the anchor fit and truncated windows at the midnight seam
    # (rules 14/15), so the frozen fallback and the by-week factors were
    # solved on different rows -- and check_calibration_convergence then
    # compares them cell by cell.
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

    r_path = cfg["dispersion"]["r_lookup_path"]
    censored_basis = os.path.exists(r_path)
    r_lookup = read_json(r_path)

    min_anchor = cfg["baseline_model"]["calibration_min_anchor_rows"]
    k_shrink = cfg["baseline_model"]["calibration_shrinkage_units"]

    fitted = _solve_level_factors(calib, model, k_shrink, min_anchor,
                                  tier_step, max_k, r_lookup)
    if fitted is None:
        anchors = int(episodes.is_anchor_row(calib, tier_step).sum())
        raise RuntimeError(
            f"fit window has only {anchors} anchor rows (need {min_anchor})"
            " -- widen calibration_fit_trailing_weeks")
    factors, detail, f_global = fitted

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
    by_week, coverage = {}, []
    for w in weeks:
        window, weeks_seen = episodes.trailing_weeks_window(
            scope, w, weeks_back)
        if not len(window):
            continue
        f = _solve_level_factors(window.copy(), model, k_shrink,
                                 min_anchor, tier_step, max_k, r_lookup)
        if f is None:                       # too thin: hold 1.0, say so
            coverage.append({"week": w, "fitted": False})
            continue
        by_week[w] = f[0]
        coverage.append({"week": w, "fitted": True,
                         "fit_rows": int(len(window)),
                         "weeks_in_window": weeks_seen,
                         "partial": weeks_seen < weeks_back})
    schedule = {
        "mode": "rolling_trailing",
        "scope": (f"production -- launch_date {cfg['data']['launch_date']}; "
                  "through the latest data plus the week being priced"
                  if launched else
                  f"pre-launch -- through split.test_end {split['test_end']}"),
        "trailing_weeks": weeks_back,
        "gate_freezes_at": str(gate_start.date()),
        "anchor_fit_window": [str(lo.date()), str(gate_start.date())],
        "week_key": "ISO week start the factors APPLY to; fit on the "
                    "trailing window ending strictly before it",
        "weeks_fitted": sum(1 for c in coverage if c["fitted"]),
        "weeks_unfitted_held_at_1": [c["week"] for c in coverage
                                     if not c["fitted"]],
        # fit on less history than trailing_weeks asks for (extract start)
        "weeks_on_partial_window": [
            {"week": c["week"], "weeks_in_window": c["weeks_in_window"]}
            for c in coverage if c.get("partial")],
        "by_week": by_week,
    }

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
               "global_factor": round(float(f_global), 4),
               "shrinkage_units": k_shrink,
               "fit_window": "rolling_trailing",
               "fit_basis": "censored E[min(D,q)]" if censored_basis
                   else "raw mu (r_lookup missing)",
               "fit_window_dates": [str(fit_dates.min().date()),
                                    str(fit_dates.max().date())],
               "fit_rows": int(len(calib)),
               "fit_in_sample_share": round(in_sample_share, 4),
               "split": split,
               "basis": "anchor rows only; cells below "
                        "calibration_min_anchor_rows left at 1.0"}
    write_json(cfg["baseline_model"]["calibration_factor_path"],
               stamp(payload, cfg, model.version,
                     "bootstrap.train_baseline --fit-calibration"))
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
    # a verdict whose chain has since moved
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

    # anchor rows behind the worst cell: a thin, shrinkage-dominated cell
    # reads identically to an unsettled loop unless the row count is shown
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
        "checked_against": checked_against,
        "max_abs_dlog": round(worst[0], 6),
        "worst_cell": worst[1],
        "cells_appeared_or_gone": missing,
        "converged": converged,
        "method": "re-solved with the prior and r_lookup now on disk; "
                  "artifact restored (dry run)",
        "verdict": (
            "CONVERGED -- one more iteration reproduces the factors within "
            "tolerance"
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
    keep = new if commit else old
    keep["convergence"] = block
    with open(path, "w") as f:
        json.dump(keep, f, indent=2)
    return block


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
                  "window is inside the training period -- the factor will "
                  "understate what launch-adjacent weeks need.")
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
                  "over-predicts there) -- investigate (AGENTS rule 5)")
        print(f"wrote {cfg['baseline_model']['calibration_factor_path']}")
        return

    schema = train(d, cfg)
    print(f"trained {schema['model_version']} on {schema['train_rows']:,} rows")
    print(f"wrote {cfg['baseline_model']['model_path']} and "
          f"{cfg['baseline_model']['feature_schema_path']}")


if __name__ == "__main__":
    main()
