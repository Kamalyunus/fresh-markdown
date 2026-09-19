"""fit.train_baseline -- fit and freeze the reference-demand model (design 5.4).

LightGBM/Tweedie on units_sold, predicting only at the reference discount
(price features overwritten to d_ref at inference). The model, its feature
encoding and the level-factor APPLIER live in `fit.model` -- the half a
priced hour imports; this command is the training run. The level-factor
FIT is `fit.calibrate`; its two flags stay on this command.
Run: python3 -m fit.train_baseline --input data/prepared.parquet [--fit-calibration]
"""

import argparse
import json
import os

import lightgbm as lgb
import pandas as pd

from common.config import load_config
from common.io import read_json
from fit.model import CATEGORICAL, FEATURES, PRICE_FEATURES, add_derived, encode_features
from fit.prepare_data import scope


def train(d, cfg):
    bm = cfg["baseline_model"]
    train_d = add_derived(scope(d, cfg, "train"))

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

    if args.check_convergence or args.fit_calibration:
        # the fit lives in fit.calibrate, which reads this module: bound
        # here, at the one entry point, rather than at import
        from fit import calibrate

    if args.check_convergence:
        block = calibrate.check_calibration_convergence(
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
        factors = calibrate.fit_level_calibration(d, cfg)
        path = cfg["baseline_model"]["calibration_factor_path"]
        print(calibrate.describe_calibration(read_json(path), factors))
        print(f"wrote {path}")
        return

    schema = train(d, cfg)
    print(f"trained {schema['model_version']} on {schema['train_rows']:,} rows")
    print(f"wrote {cfg['baseline_model']['model_path']} and "
          f"{cfg['baseline_model']['feature_schema_path']}")


if __name__ == "__main__":
    main()
