"""A validated request becomes the engine's state: the request's fields in
one spelling, the dispersion r from the lookup, and the frozen model's
forecast over the remaining hours -- a fresh one for an entry request (its
two demand-rate features read off the day's table), the entry's stored
forecast sliced for a later hour of a known episode, extended by prediction
only when the window grew."""
import json
import math

import numpy as np
import pandas as pd

from pricing.decide import count_failures
from pricing.keys import hours_between, ident, iso_day

REQUEST_FIELDS = ("episode_id", "sku_id", "fc", "category", "subcategory",
                  "date", "hour_of_day", "hours_remaining", "q",
                  "original_price", "cost", "current_discount")
POOLED_FC = "*"                 # the SKU-pooled row of the feature table
_NUMBER = (int, float, np.integer, np.floating)


class Posterior:
    """The learning state, read once per batch: the cell a category prices
    at and the suspension flag."""

    def __init__(self, path):
        with open(path) as f:
            self.state = json.load(f)

    def cell(self, category):
        name = self.state["cell_of"].get(str(category), "GLOBAL")
        return self.state["cells"][name]

    @property
    def suspended(self):
        return self.state.get("exploration_suspended")


def lookup_r(r_lookup, subcategory, category):
    keys = {"subcategory": str(subcategory), "category": str(category)}
    for level in r_lookup["fallback_order"]:
        if level == "global":
            return r_lookup["global"]
        r = r_lookup[level].get(keys[level])
        if r is not None:
            return r
    return r_lookup["global"]


def _null(v):
    return v is None or (isinstance(v, (float, np.floating)) and not math.isfinite(v))


def validate_request(r, cfg):
    """What a request must carry before a state can be built from it."""
    problems = [f"missing {f}" for f in REQUEST_FIELDS if f not in r]
    if problems:
        return problems
    try:
        iso_day(r["date"])
    except (TypeError, ValueError):
        problems.append(f"date {r['date']!r} names no day")
    problems += count_failures(r, cfg)
    for f in ("episode_id", "sku_id", "fc"):
        try:
            ident(r[f])
        except (TypeError, ValueError):
            problems.append(f"{f} is null" if _null(r[f]) else
                            f"{f} is not an identifier: {r[f]!r}")
    for f in ("category", "subcategory"):
        if _null(r[f]):
            problems.append(f"{f} is null")
    for f in ("original_price", "cost"):
        v = r[f]
        if v is None:
            problems.append(f"{f} is null")
        elif isinstance(v, (bool, np.bool_)) or not isinstance(v, _NUMBER):
            problems.append(f"{f} is not a number: {v!r}")
    return problems


def canonical_request(r):
    anchor = r["current_discount"]
    if _null(anchor):
        anchor = None
    return {
        "episode_id": ident(r["episode_id"]), "sku_id": ident(r["sku_id"]),
        "fc": ident(r["fc"]), "category": str(r["category"]),
        "subcategory": str(r["subcategory"]), "date": iso_day(r["date"]),
        "hour_of_day": int(r["hour_of_day"]),
        "hours_remaining": int(r["hours_remaining"]), "q": int(r["q"]),
        "original_price": float(r["original_price"]), "cost": float(r["cost"]),
        "current_discount": anchor,
    }


def hour_grid(day, opening_hour, n_hours):
    base = pd.Timestamp(day) + pd.Timedelta(hours=opening_hour)
    return [((base + pd.Timedelta(hours=k)).strftime("%Y-%m-%d"),
             int((base + pd.Timedelta(hours=k)).hour)) for k in range(n_hours)]


def table_as_of(table):
    if table is None or not len(table) or "as_of" not in table:
        return None
    return str(table["as_of"].iloc[0])


def features_from_table(table, openings):
    """{episode_id: (rate_30d, prior_episode_rate)} for `openings` (dicts
    with episode_id, sku_id, fc): the (sku, fc) row, else the SKU's pooled
    row, else (NaN, NaN) -- the model's own "unknown"."""
    idx = {(str(r.sku_id), str(r.fc)): (float(r.sku_ref_sales_rate_30d),
                                        float(r.prior_episode_ref_sales_rate))
           for r in table.itertuples()}
    out = {}
    for o in openings:
        sku, fc = ident(o["sku_id"]), ident(o["fc"])
        f = idx.get((sku, fc))
        if f is None:
            f = idx.get((sku, POOLED_FC), (float("nan"), float("nan")))
        out[o["episode_id"]] = f
    return out


def features_unknown(feats):
    return all(isinstance(v, float) and math.isnan(v) for v in feats)


def mu_ref_paths(model, openings):
    """`mu_ref_path` for many openings in ONE prediction, aligned with them;
    each opening carries `template`, `grid` and `features`."""
    if not openings:
        return []
    rows = []
    for i, o in enumerate(openings):
        tpl, (rate30, prior_rate) = o["template"], o["features"]
        for date, hour in o["grid"]:
            rows.append((i, date, hour, tpl["category"], tpl["subcategory"],
                         tpl["fc"], tpl["original_price"], rate30, prior_rate))
    frame = pd.DataFrame(rows, columns=[
        "_i", "date", "hour_of_day", "category", "subcategory", "fc",
        "original_price", "sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate"])
    frame["total_discount"] = np.nan
    mu = model.predict_mu_ref(frame)
    out = [[] for _ in openings]
    for i, m in zip(frame["_i"].to_numpy(), mu):
        out[i].append(float(m))
    return out


def _template(r):
    return {"category": r["category"], "subcategory": r["subcategory"],
            "fc": r["fc"], "original_price": float(r["original_price"])}


def _feature_pair(stored):
    return tuple(float("nan") if v is None else float(v) for v in stored)


def build_states(requests, cfg, model, r_lookup, episode_paths, table):
    """The state for each VALIDATED request, aligned with it, and the
    counts: forecasts with no history behind them, later hours whose
    episode the store does not know, restock extensions predicted without
    the entry's recorded features."""
    empty = {"requests_with_unknown_features": 0,
             "non_entry_requests_without_stored_path": 0,
             "restock_extensions_without_stored_features": 0}
    if not requests:
        return [], empty
    requests = [canonical_request(r) for r in requests]
    paths = episode_paths or {}
    fresh, tails, sliced = {}, {}, {}
    without_stored = 0
    for i, r in enumerate(requests):
        stored = paths.get(r["episode_id"]) if r["current_discount"] is not None else None
        if stored is None:
            if r["current_discount"] is not None:
                without_stored += 1
            fresh[i] = (r["date"], r["hour_of_day"])
            continue
        k = hours_between(stored["date"], stored["hour_of_day"], r["date"], r["hour_of_day"])
        if k < 0:
            without_stored += 1
            fresh[i] = (r["date"], r["hour_of_day"])
            continue
        path = list(stored["mu_ref_path"][k:k + r["hours_remaining"]])
        if len(path) < r["hours_remaining"]:
            tails[i] = path
        else:
            sliced[i] = path

    stubs, feats, without_features = {}, {}, 0
    for i in fresh:
        r = requests[i]
        stubs.setdefault(r["episode_id"], {"episode_id": r["episode_id"],
                                           "sku_id": r["sku_id"], "fc": r["fc"]})
    for i in tails:
        r = requests[i]
        stored = paths[r["episode_id"]].get("features")
        if stored is not None:
            feats[r["episode_id"]] = _feature_pair(stored)
            continue
        without_features += 1
        stubs.setdefault(r["episode_id"], {"episode_id": r["episode_id"],
                                           "sku_id": r["sku_id"], "fc": r["fc"]})
    if stubs:
        if table is None:
            raise ValueError("a batch prices against the day's feature table")
        feats.update(features_from_table(table, list(stubs.values())))
    unknown, openings, owners = 0, [], []
    for i in fresh:
        r = requests[i]
        f = feats[r["episode_id"]]
        unknown += features_unknown(f)
        openings.append({"template": _template(r),
                         "grid": hour_grid(r["date"], r["hour_of_day"], r["hours_remaining"]),
                         "features": f})
        owners.append(i)
    for i, path in tails.items():
        r = requests[i]
        grid = hour_grid(r["date"], r["hour_of_day"], r["hours_remaining"])[len(path):]
        openings.append({"template": _template(r), "grid": grid,
                         "features": feats[r["episode_id"]]})
        owners.append(i)
    predicted = dict(zip(owners, mu_ref_paths(model, openings)))
    for i, path in tails.items():
        sliced[i] = path + predicted[i]
    for i in fresh:
        sliced[i] = predicted[i]

    def features_of(i, r):
        if i in fresh or i in tails:
            return feats[r["episode_id"]]
        stored = paths[r["episode_id"]].get("features")
        return None if stored is None else _feature_pair(stored)

    states = [{**{f: r[f] for f in REQUEST_FIELDS},
               "r": float(lookup_r(r_lookup, r["subcategory"], r["category"])),
               "mu_ref_path": list(sliced[i]),
               "features": None if features_of(i, r) is None else tuple(features_of(i, r))}
              for i, r in enumerate(requests)]
    return states, {"requests_with_unknown_features": int(unknown),
                    "non_entry_requests_without_stored_path": int(without_stored),
                    "restock_extensions_without_stored_features": int(without_features)}
