"""bootstrap.train_baseline -- fit and freeze the reference-demand model.

Design section 5.4. LightGBM, Tweedie objective, target units_sold. The model
predicts demand ONLY at the category reference discount: at inference the price
features are overwritten to d_ref, so the model's price gradient is never
queried and the legacy confound stays out of the decision path.

Inventory, cost, and stockout indicators are never features -- inventory
belongs in the DP state and the censoring logic.

Frozen at launch; no retraining during the MVP window. The model version is
recorded on every decision event.

Also fits the per-category multiplicative level-calibration factor on the
calibration window (the section 9.3 remedy for LEVEL bias). Whether it is
applied is decided by `baseline_model.apply_level_calibration` after the
level/slope diagnostic -- fitting it is unconditional, applying it is not.

Usage:
    python3 -m bootstrap.train_baseline --input data/prepared.parquet
    python3 -m bootstrap.train_baseline --input data/prepared.parquet \
        --fit-calibration        # fit level factors from an existing baseline
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

# Feature order is authoritative in feature_schema.json; this list only seeds
# the first fit. `total_discount` is the SINGLE price feature and is the one
# overwritten to d_ref at inference -- one overwrite point is auditable,
# several are a standing leak risk.
#
# Deliberately absent (see docs/design.md for the full argument):
#   hours_remaining        planner state, not customer-visible demand context
#   last-hour lag sales    mediators of the episode's own price path -- they
#                          absorb price response and corrupt the learned
#                          elasticity, and at median inventory ~2 they are
#                          mostly a censoring indicator
#   inventory / stockout   belong to the DP state and the censoring logic
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

    def __init__(self, cfg):
        bm = cfg["baseline_model"]
        self.cfg = cfg
        self.booster = lgb.Booster(model_file=bm["model_path"])
        with open(bm["feature_schema_path"]) as f:
            self.schema = json.load(f)
        self.calibration, self.calibration_grain = {}, "category"
        if os.path.exists(bm["calibration_factor_path"]):
            with open(bm["calibration_factor_path"]) as f:
                cal = json.load(f)
            self.calibration = cal.get("factors", cal.get("factor_by_category", {}))
            self.calibration_grain = cal.get("grain", "category")
        self.version = self.schema["model_version"]

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
            key = d[self.calibration_grain]      # grain is the artifact's own
            factor = key.map(lambda k: self.calibration.get(str(k), 1.0)).to_numpy()
            mu = mu * factor
        return mu


def train(d, cfg):
    bm = cfg["baseline_model"]
    splits = split_frames(d, cfg)
    # baseline_model.train_population: "integrity" keeps the episodes the
    # DP cannot price -- FEATURES carries neither cost nor hours_remaining,
    # so the model cannot see what makes them ineligible.
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


def fit_level_calibration(d, cfg):
    """Per-category multiplicative level factor, fit at the reference anchor.

    Fit ONLY at the anchor, where the elasticity multiplier is ~1, so the
    factor captures LEVEL error in mu_ref alone. A category without enough
    anchor rows is left uncorrected (factor 1.0) rather than fit through an
    elasticity-dependent basis: any basis that scales predictions by the
    prior elasticity lets slope error leak into the level factor.

    The fit window is configurable (calibration_fit_window) and must be
    DISJOINT from calibration_gate_window, or the fit grades itself. The
    factor is solved on the CENSORED basis -- sales cannot exceed inventory,
    so predictions are compared as E[min(D, q)], the same quantity the gate
    measures. Fitting against raw mu instead reads systematically low (raw mu
    >= censored expectation) and produces a factor that cannot move the gate;
    and because the factor scales mu BEFORE censoring, the censored total
    moves by less than the factor, so f is solved for rather than divided out.

    A factor below 1 means the model OVER-predicts at the anchor on the fit
    window -- worth a manual look before applying.
    """
    from bootstrap.fit_dispersion import lookup_r   # local: avoids a cycle

    model = BaselineModel(cfg)
    splits = split_frames(d, cfg)
    fit_window = cfg["baseline_model"]["calibration_fit_window"]
    splits = {k: population(v, cfg) for k, v in splits.items()}
    if fit_window == "calib":
        calib = splits["calib"].copy()
    elif fit_window == "train+calib":
        calib = pd.concat([splits["train"], splits["calib"]])
    elif fit_window == "all":
        # "all" means all PRE-LAUNCH data, never the hold-out. Without the
        # bound this one branch quietly fits the level factors on the window
        # reserved for grading them.
        calib = population(pre_launch(d, cfg), cfg).copy()
    elif fit_window == "trailing":
        # the last N weeks ENDING WHERE THE GATE WINDOW BEGINS: recent enough
        # to track a moving level, and disjoint from what the gate evaluates
        # so a fit can never grade itself
        weeks = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
        # the gate window's own start -- test_start when the gate reads test,
        # calib_start when it reads calib+test. Hardcoding calib_start pushed
        # the trailing window entirely inside the training period whenever the
        # gate was on test.
        split = cfg["data"]["split"]
        gate_start = pd.Timestamp(
            split["test_start"]
            if cfg["baseline_model"]["calibration_gate_window"] == "test"
            else split["calib_start"])
        lo = gate_start - pd.Timedelta(weeks=weeks)
        dates = pd.to_datetime(d.date)
        calib = d[(dates >= lo) & (dates < gate_start)].copy()
    else:
        raise ValueError(f"unknown calibration_fit_window: {fit_window}")
    if not len(calib):
        raise RuntimeError("calibration fit window contains no rows")

    # a trailing window can land inside the training period, where the model
    # fits by construction and the factor understates the correction the
    # launch-adjacent weeks need -- surface it rather than hide it
    fit_dates = pd.to_datetime(calib.date)
    train_end = pd.Timestamp(cfg["data"]["split"]["train_end"])
    in_sample_share = float((fit_dates <= train_end).mean())
    tier_step = cfg["pricing"]["tier_step"]

    # predict without any existing calibration applied
    saved = cfg["baseline_model"]["apply_level_calibration"]
    cfg["baseline_model"]["apply_level_calibration"] = False
    calib["mu_ref_hat"] = model.predict_mu_ref(calib)
    cfg["baseline_model"]["apply_level_calibration"] = saved

    # censoring basis: sales are capped at inventory, so the factor must be
    # solved against E[min(D, q)] -- the same quantity the gate measures.
    # Fitting against raw mu makes the factor read systematically low and it
    # can never move a gate read on censored predictions. Scaling mu by f
    # also moves the censored total by LESS than f, so f is solved for, not
    # divided out.
    max_k = cfg["pricing"]["negbin_max_k"]
    r_path = cfg["dispersion"]["r_lookup_path"]
    censored_basis = os.path.exists(r_path)
    if censored_basis:
        with open(r_path) as f:
            r_lookup = json.load(f)
        calib["r_val"] = [lookup_r(r_lookup, s, c)
                          for s, c in zip(calib.subcategory, calib.category)]

    def solve_factor(anchor):
        sold = float(anchor["units_sold"].sum())
        mu = anchor["mu_ref_hat"].to_numpy()
        if not censored_basis:
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

    min_anchor = cfg["baseline_model"]["calibration_min_anchor_rows"]
    grain = cfg["baseline_model"]["calibration_grain"]
    k_shrink = cfg["baseline_model"]["calibration_shrinkage_units"]

    anchor_all = calib[(calib.total_discount - calib.d_ref).abs() <= tier_step / 2]
    if len(anchor_all) < min_anchor or anchor_all["mu_ref_hat"].sum() <= 0:
        raise RuntimeError(
            f"fit window has only {len(anchor_all)} anchor rows "
            f"(need {min_anchor}) -- widen calibration_fit_window")

    def shrink(cell, parent, evidence):
        """Geometric shrinkage toward the parent, weighted by evidence (anchor
        units sold). Factors are multiplicative, so the pull is in log space.
        Zero evidence lands exactly on the parent -- no threshold cliff."""
        if evidence <= 0 or cell <= 0:
            return parent
        w = evidence / (evidence + k_shrink)
        return float(np.exp(w * np.log(cell) + (1 - w) * np.log(parent)))

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

    path = cfg["baseline_model"]["calibration_factor_path"]
    with open(path, "w") as f:
        # is calibration needed at all? factors clustered on 1.0 mean the
        # frozen model is already level-correct and the remedy should stay
        # off; wide dispersion is evidence of systematic per-cell model bias
        # -- a training signal, not something to paper over indefinitely
        fv = np.array(list(factors.values()), dtype=float)
        payload = {"grain": grain,
                   "factor_summary": {
                       "p10": round(float(np.percentile(fv, 10)), 4),
                       "p50": round(float(np.percentile(fv, 50)), 4),
                       "p90": round(float(np.percentile(fv, 90)), 4),
                       "share_within_5pct_of_1": round(
                           float((np.abs(fv - 1.0) <= 0.05).mean()), 4),
                       "note": "clustered on 1.0 -> model is level-correct, "
                               "leave apply_level_calibration false; wide -> "
                               "systematic per-cell bias worth fixing in "
                               "training, not only in the multiplier",
                   },
                   "factors": factors,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fit-calibration", action="store_true",
                    help="fit the section 9.3 per-category level-calibration "
                         "factors from the already-trained baseline and the "
                         "section 9.5 prior, instead of training")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)

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
                  "over-predicts there) -- investigate before applying "
                  "(AGENTS rule 5)")
        print(f"wrote {cfg['baseline_model']['calibration_factor_path']}")
        print("next: set baseline_model.apply_level_calibration: true in "
              "config.yaml, re-run backtest WITHOUT retraining the baseline, "
              "and record the fidelity ratio before and after (design 9.2)")
        return

    schema = train(d, cfg)
    print(f"trained {schema['model_version']} on {schema['train_rows']:,} rows")
    print(f"wrote {cfg['baseline_model']['model_path']} and "
          f"{cfg['baseline_model']['feature_schema_path']}")


if __name__ == "__main__":
    main()
