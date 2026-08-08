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
# the first fit. `total_discount` is the single price feature and is the one
# overwritten to d_ref at inference.
FEATURES = ["category", "subcategory", "fc", "hour_of_day", "dow",
            "hours_remaining", "total_discount"]
CATEGORICAL = ["category", "subcategory", "fc"]
PRICE_FEATURES = ["total_discount"]


def add_derived(d):
    d = d.copy()
    d["dow"] = pd.to_datetime(d.date).dt.dayofweek
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


def fit_level_calibration(d, cfg, prior_mean_by_category):
    """Per-category multiplicative factor on the calibration window.

    Fit at the reference anchor where the elasticity multiplier is ~1, so the
    factor captures LEVEL error in mu_ref alone; fall back to all calibration
    rows (elasticity-adjusted) when a category has no anchor rows.
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

    eps = calib["category"].map(
        lambda c: prior_mean_by_category.get(str(c), np.nan)).astype(float)
    ratio_term = ((1 - calib.total_discount) / (1 - calib.d_ref)) ** eps
    calib["mu_actual_hat"] = calib["mu_ref_hat"] * ratio_term.fillna(1.0)

    min_anchor = cfg["baseline_model"]["calibration_min_anchor_rows"]
    factors = {}
    for cat, g in calib.groupby("category"):
        anchor = g[(g.total_discount - g.d_ref).abs() <= tier_step / 2]
        use_anchor = len(anchor) >= min_anchor and anchor["mu_ref_hat"].sum() > 0
        basis = anchor if use_anchor else g
        pred = basis["mu_ref_hat"].sum() if use_anchor else basis["mu_actual_hat"].sum()
        if pred > 0:
            factors[str(cat)] = round(float(basis["units_sold"].sum() / pred), 4)

    path = cfg["baseline_model"]["calibration_factor_path"]
    with open(path, "w") as f:
        json.dump({"factor_by_category": factors,
                   "fit_window": cfg["data"]["split"],
                   "basis": "anchor rows where available, else all calib rows"},
                  f, indent=2)
    return factors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)
    schema = train(d, cfg)
    print(f"trained {schema['model_version']} on {schema['train_rows']:,} rows")
    print(f"wrote {cfg['baseline_model']['model_path']} and "
          f"{cfg['baseline_model']['feature_schema_path']}")


if __name__ == "__main__":
    main()
