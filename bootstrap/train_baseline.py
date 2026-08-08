"""bootstrap.train_baseline -- fit and freeze the reference-demand model.

PRD section 9.3. LightGBM, Tweedie objective, target units_sold. The model
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
from bootstrap.prepare_data import split_frames

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
        self.calibration = {}
        if os.path.exists(bm["calibration_factor_path"]):
            with open(bm["calibration_factor_path"]) as f:
                self.calibration = json.load(f)["factor_by_category"]
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
            factor = d["category"].map(
                lambda c: self.calibration.get(str(c), 1.0)).to_numpy()
            mu = mu * factor
        return mu


def train(d, cfg):
    bm = cfg["baseline_model"]
    splits = split_frames(d, cfg)
    train_d = add_derived(splits["train"])

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
    """Per-category multiplicative factor on the calibration window.

    Fit ONLY at the reference anchor, where the elasticity multiplier is ~1,
    so the factor captures LEVEL error in mu_ref alone. A category without
    enough anchor rows is left uncorrected (factor 1.0) rather than fit
    through an elasticity-dependent basis: any basis that scales predictions
    by the prior elasticity lets slope error leak into the level factor,
    which is exactly the contamination PRD section 9.3 forbids.

    Factors are >= 1 wherever the baseline under-predicts at the anchor;
    a factor below 1 means the model OVER-predicts there and is worth a
    manual look before applying.
    """
    model = BaselineModel(cfg)
    calib = split_frames(d, cfg)["calib"].copy()
    if not len(calib):
        raise RuntimeError("calibration window contains no rows")
    tier_step = cfg["pricing"]["tier_step"]

    # predict without any existing calibration applied
    saved = cfg["baseline_model"]["apply_level_calibration"]
    cfg["baseline_model"]["apply_level_calibration"] = False
    calib["mu_ref_hat"] = model.predict_mu_ref(calib)
    cfg["baseline_model"]["apply_level_calibration"] = saved

    min_anchor = cfg["baseline_model"]["calibration_min_anchor_rows"]
    factors, detail = {}, {}
    for cat, g in calib.groupby("category"):
        anchor = g[(g.total_discount - g.d_ref).abs() <= tier_step / 2]
        pred = float(anchor["mu_ref_hat"].sum())
        fitted = len(anchor) >= min_anchor and pred > 0
        factor = float(anchor["units_sold"].sum() / pred) if fitted else 1.0
        factors[str(cat)] = round(factor, 4)
        detail[str(cat)] = {
            "basis": "anchor" if fitted else "uncorrected",
            "anchor_rows": int(len(anchor)),
            "calib_rows": int(len(g)),
            "anchor_sold": int(anchor["units_sold"].sum()),
            "anchor_predicted": round(pred, 1),
        }

    path = cfg["baseline_model"]["calibration_factor_path"]
    with open(path, "w") as f:
        json.dump({"factor_by_category": factors,
                   "detail_by_category": detail,
                   "fit_window": cfg["data"]["split"],
                   "basis": "anchor rows only; categories below "
                            "calibration_min_anchor_rows left at 1.0"},
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
            detail = json.load(f)["detail_by_category"]
        for cat, factor in sorted(factors.items()):
            info = detail[cat]
            print(f"  {cat:24s} {factor:.4f}  "
                  f"({info['basis']}, {info['anchor_rows']:,} anchor rows)")
        uncorrected = [c for c, v in detail.items() if v["basis"] == "uncorrected"]
        if uncorrected:
            print(f"left uncorrected (below "
                  f"{cfg['baseline_model']['calibration_min_anchor_rows']} "
                  f"anchor rows): {', '.join(sorted(uncorrected))}")
        print(f"wrote {cfg['baseline_model']['calibration_factor_path']}")
        print("next: set baseline_model.apply_level_calibration: true in "
              "config.yaml, re-run backtest WITHOUT retraining the baseline, "
              "and record the fidelity ratio before and after (PRD 9.3)")
        return

    schema = train(d, cfg)
    print(f"trained {schema['model_version']} on {schema['train_rows']:,} rows")
    print(f"wrote {cfg['baseline_model']['model_path']} and "
          f"{cfg['baseline_model']['feature_schema_path']}")


if __name__ == "__main__":
    main()
