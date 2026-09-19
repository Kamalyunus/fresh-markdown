"""fit.train_baseline -- fit and freeze the reference-demand model (design 5.4).

LightGBM/Tweedie on units_sold, predicting only at the reference discount
(price features overwritten to d_ref at inference), and the level-factor
APPLIER (`BaselineModel.level_factors`), which is always applied. The
level-factor FIT is `fit.calibrate`; its two flags stay on this command.
Run: python3 -m fit.train_baseline --input data/prepared.parquet [--fit-calibration]
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import load_config, reference_discount
from common.io import read_json
from common import windows
from fit.prepare_data import scope

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


def weeks_held_at_anchor(schedule):
    """The weeks a schedule judged too thin to fit and holds at the frozen
    anchor -- `weeks_unfitted_held_at_anchor`, or the name an artifact
    sealed before the rename wrote it under (`weeks_unfitted_held_at_1`)."""
    return list((schedule or {}).get("weeks_unfitted_held_at_anchor")
                or (schedule or {}).get("weeks_unfitted_held_at_1") or [])


def schedule_reaches(schedule):
    """The last week the factor schedule COVERS: a week it fitted, or one it
    judged too thin and deliberately holds at the frozen anchor
    (`weeks_unfitted_held_at_anchor`; `level_factors` applies the anchor
    there). None when there is no schedule. The ONE reading
    `daily.update`'s calibration_current gate and `ops.advance`'s re-fit
    trigger share -- reading `by_week` alone made a thin week look like a
    missed cron: the gate refused every --apply and advance re-fit every
    morning."""
    if not schedule:
        return None
    weeks = list(schedule.get("by_week") or {}) + weeks_held_at_anchor(schedule)
    return max(weeks) if weeks else None


class BaselineModel:
    """Frozen mu_ref predictor. Loads model + schema + calibration artifacts."""

    # class-level so an applier built without __init__ (the tests' __new__
    # path, the harness appliers) prices unfrozen with no gate and no
    # parent tables; the instance sets them all
    calibration_stops_at = None
    calibration_reaches = None
    calibration_category = None
    calibration_schedule_category = None
    _freeze_from = None

    def __init__(self, cfg):
        bm = cfg["baseline_model"]
        self.cfg = cfg
        self.booster = lgb.Booster(model_file=bm["model_path"])
        with open(bm["feature_schema_path"]) as f:
            self.schema = json.load(f)
        self.calibration, self.calibration_grain = {}, GRAIN
        # week-keyed factors applied by the week the row's EPISODE OPENED
        # (no row priced by its own week's fit); `calibration` is the
        # frozen-anchor fallback. The `_category` tables are the parent
        # level, for a cell the window never saw (level_factors waterfalls
        # subcategory -> category -> 1.0)
        self.calibration_category = {}
        self.calibration_schedule = None
        self.calibration_schedule_category = {}
        self.calibration_stops_at = None
        if os.path.exists(bm["calibration_factor_path"]):
            with open(bm["calibration_factor_path"]) as f:
                cal = json.load(f)
            self.calibration = cal.get("factors", {})
            self.calibration_category = cal.get("factors_category") or {}
            self.calibration_grain = cal.get("grain", GRAIN)
            sched = cal.get("schedule")
            if sched and sched.get("by_week"):
                self.calibration_schedule = sched["by_week"]
                self.calibration_schedule_category = sched.get("by_week_category") or {}
                self.calibration_stops_at = sched.get("gate_freezes_at")
                # the week the schedule COVERS (fitted or held), for coverage
                self.calibration_reaches = schedule_reaches(sched)
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

    @staticmethod
    def level_lookup(table, parent_table, keys, parents):
        """Factors for `keys` from `table`, a key the table never fitted
        taking its PARENT's factor (`parents`, the category column) and
        only a cell no level of the table saw reading 1.0 -- the waterfall
        the fit itself applies (`solve_level_factors` emits every cell of
        the window's population at its parent), carried to cells outside
        the window entirely: a new subcategory prices at its category's
        level, never at raw mu while its category solved to something else."""
        out = keys.map(table).astype(float)
        if parents is not None and parent_table:
            out = out.fillna(parents.map(parent_table).astype(float))
        return out.fillna(1.0).to_numpy()

    def level_factors(self, d):
        """Per-row level factor, the one `predict_mu_ref` applies: each row
        takes the factors in force for the week its EPISODE OPENED (a frame
        without `episode_id` -- the live path's forecast rows -- reads the
        row's own date); unfitted weeks fall back to the frozen anchor,
        never forward. Applied by opening week because the fit windows are
        cut by opening week: a row-week read put the Monday rows of a
        Sunday-opened episode inside week w's fit window AND under week
        w's table. Public because a caller may need the factor itself (a
        rescale between two freezes is exact, so the backtest's
        weekly-refit reading never predicts twice)."""
        keys = d[self.calibration_grain].astype(str)
        parents = (d["category"].astype(str)
                   if self.calibration_grain != "category" and "category" in d
                   else None)
        anchor = self.level_lookup(self.calibration, self.calibration_category,
                              keys, parents)
        if self.calibration_schedule is None:
            self._cal_rows_static += len(d)
            return anchor
        dates = pd.to_datetime(d["date"])
        opened = (pd.to_datetime(windows.opening_dates(d))
                  if "episode_id" in d else dates)
        weeks = windows.week_key(opened).to_numpy()
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
            out[rows] = self.level_lookup(
                table, (self.calibration_schedule_category or {}).get(wk),
                keys[rows], None if parents is None else parents[rows])
        return out

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
        # past the end = past the last week the schedule COVERS, held
        # weeks included (schedule_reaches, the gate's reading); a trailing
        # held week once read here as STALE while the gate called it covered
        end = self.calibration_reaches or max(self.calibration_schedule)
        past_end = sorted(w for w in self._cal_fallback_weeks if w > end)
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
            "schedule_covers": [min(self.calibration_schedule), end],
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


# the level-factor fit moved to fit.calibrate; the names stay for callers.
# Resolved on first use rather than imported above: calibrate reads this
# module (the model, the applier), so an eager import here would be a cycle.
_MOVED_TO_CALIBRATE = ("attach_fit_basis", "pinned_cells",
                       "solve_level_factors", "_solve_level_factors",
                       "category_factors", "keys_held_at_parent",
                       "fit_level_calibration", "check_calibration_convergence",
                       "describe_calibration", "_describe_calibration")


def __getattr__(name):
    if name in _MOVED_TO_CALIBRATE:
        from fit import calibrate
        return getattr(calibrate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
