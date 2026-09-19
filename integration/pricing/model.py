"""The frozen demand model, applied: the schema's feature encoding, the
LightGBM booster asked for demand at the reference discount, the level
factors on top (the week's table by the row's date, the frozen anchor where
a week was never fitted, the category's factor where a subcategory was
never seen, 1.0 where nothing was)."""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from pricing.config import reference_discount


def week_key(dates):
    return (pd.to_datetime(dates).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d"))


def _encode(d, features, categorical, levels):
    X = pd.DataFrame(index=d.index)
    for feat in features:
        if feat in categorical:
            values = d[feat].astype(str)
            known = values.where(values.isin(levels[feat]))
            X[feat] = pd.Categorical(known, categories=levels[feat]).codes
        else:
            X[feat] = pd.to_numeric(d[feat])
    return X


def _lookup(table, parent_table, keys, parents):
    out = keys.map(table).astype(float)
    if parents is not None and parent_table:
        out = out.fillna(parents.map(parent_table).astype(float))
    return out.fillna(1.0).to_numpy()


class DemandModel:
    def __init__(self, cfg):
        bm = cfg["baseline_model"]
        self.cfg = cfg
        self.booster = lgb.Booster(model_file=bm["model_path"])
        with open(bm["feature_schema_path"]) as f:
            self.schema = json.load(f)
        self.version = self.schema["model_version"]
        self.factors, self.factors_category, self.grain = {}, {}, "subcategory"
        self.schedule, self.schedule_category = None, {}
        if os.path.exists(bm["calibration_factor_path"]):
            with open(bm["calibration_factor_path"]) as f:
                cal = json.load(f)
            self.factors = cal.get("factors", {})
            self.factors_category = cal.get("factors_category") or {}
            self.grain = cal.get("grain", "subcategory")
            sched = cal.get("schedule")
            if sched and sched.get("by_week"):
                self.schedule = sched["by_week"]
                self.schedule_category = sched.get("by_week_category") or {}

    def level_factors(self, d):
        """Per row: the week's table for the row's date, else the anchor."""
        keys = d[self.grain].astype(str)
        parents = (d["category"].astype(str)
                   if self.grain != "category" and "category" in d else None)
        out = _lookup(self.factors, self.factors_category, keys, parents)
        if self.schedule is None:
            return out
        weeks = week_key(pd.to_datetime(d["date"])).to_numpy()
        for wk in np.unique(weeks):
            table = self.schedule.get(wk)
            if table is None:
                continue                              # unfitted week: the anchor
            rows = weeks == wk
            out[rows] = _lookup(table, (self.schedule_category or {}).get(wk),
                                keys[rows], None if parents is None else parents[rows])
        return out

    def predict_mu_ref(self, d):
        """mu_ref per row of `d` (category, subcategory, fc, date, hour_of_day,
        original_price, the two demand-rate features), at the reference discount."""
        d = d.copy()
        dates = pd.to_datetime(d.date)
        d["dow"] = dates.dt.dayofweek
        d["day_of_month"] = dates.dt.day
        for feat in self.schema["price_features"]:
            d[feat] = d["d_ref"] if "d_ref" in d.columns else d["category"].map(
                lambda c: reference_discount(self.cfg, c))
        missing = [f for f in self.schema["features"]
                   if f not in d.columns and f not in ("dow", "day_of_month")]
        if missing:
            raise KeyError(f"frame is missing feature columns {missing}")
        X = _encode(d, self.schema["features"], self.schema["categorical"],
                    self.schema["category_levels"])
        mu = np.clip(self.booster.predict(X), self.cfg["pricing"]["demand_floor"], None)
        return mu * self.level_factors(d)
