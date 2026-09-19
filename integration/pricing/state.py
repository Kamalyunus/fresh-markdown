"""A snapshot row becomes the engine's state, from the row alone.

The service is stateless: nothing is looked up from an earlier hour. The
row's `episode_id` is the opening tag of its episode, `<sku>|<fc>|<day>T<hh>`,
so the row is an ENTRY exactly when the tag names this shelf-hour, and a
later hour of the episode otherwise. An entry has no anchor (the entry
action set applies); a later hour is anchored on the price in force the
row carries, which is the price this service applied last hour, piped
back by the producers. The two demand-rate features are read from the
feature table OF THE EPISODE'S OPENING DAY, so every hour of an episode is
forecast on what the entry was forecast on, and the forecast is re-made
every hour over the remaining horizon."""
import json
import math
import re

import numpy as np
import pandas as pd

from pricing.decide import count_failures
from pricing.keys import hour_key, hours_between, ident, iso_day, nan_pair, shelf_hour_tag

REQUEST_FIELDS = ("episode_id", "sku_id", "fc", "category", "subcategory",
                  "date", "hour_of_day", "hours_remaining", "q",
                  "original_price", "cost", "current_discount")
POOLED_FC = "*"                 # the SKU-pooled row of the feature table
_NUMBER = (int, float, np.integer, np.floating)
_TAG = re.compile(r"^(?P<sku>[^|]+)\|(?P<fc>[^|]+)\|(?P<day>\d{4}-\d{2}-\d{2})T(?P<hh>\d{2})$")


class Posterior:
    """The learning state, read once per batch: the cell a category prices at."""

    def __init__(self, path):
        with open(path) as f:
            self.state = json.load(f)

    def cell(self, category):
        name = self.state["cell_of"].get(str(category), "GLOBAL")
        return self.state["cells"][name]



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


# ------------------------------------------------------------ the episode

def opening_of(episode_id):
    """(sku, fc, day, hour) the id names as the episode's first hour, or
    None when the id is not an opening tag."""
    if _null(episode_id):
        return None
    m = _TAG.match(str(episode_id).strip())
    if not m:
        return None
    try:
        return hour_key(m["sku"], m["fc"], m["day"], int(m["hh"]))
    except (TypeError, ValueError):
        return None


def episode_position(episode_id, key):
    """Where the shelf-hour `key` stands in its episode: ("entry", opening),
    ("later", opening), or (reason, None) when the id cannot place it --
    not an opening tag, another shelf's, or an opening after this hour."""
    opening = opening_of(episode_id)
    if opening is None:
        return "episode_id is not an opening tag (<sku>|<fc>|<day>T<hh>)", None
    if opening[:2] != key[:2]:
        return f"episode_id names another shelf: {shelf_hour_tag(opening)}", None
    elapsed = hours_between(opening[2], opening[3], key[2], key[3])
    if elapsed < 0:
        return f"episode_id opens after this hour: {shelf_hour_tag(opening)}", None
    return ("entry" if elapsed == 0 else "later"), opening


# ------------------------------------------------------------ the request

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
        "opening_day": r["opening_day"],
    }


# ----------------------------------------------------------- the forecast

def hour_grid(day, opening_hour, n_hours):
    base = pd.Timestamp(day) + pd.Timedelta(hours=opening_hour)
    return [((base + pd.Timedelta(hours=k)).strftime("%Y-%m-%d"),
             int((base + pd.Timedelta(hours=k)).hour)) for k in range(n_hours)]


def feature_index(table):
    """{(sku, fc): (rate_30d, prior_episode_rate)} of a day's table."""
    return {(str(r.sku_id), str(r.fc)): (float(r.sku_ref_sales_rate_30d),
                                         float(r.prior_episode_ref_sales_rate))
            for r in table.itertuples()}


def features_of(index, sku, fc):
    """The shelf's pair: its (sku, fc) row, else the SKU's pooled row, else
    (NaN, NaN) -- the model's own "unknown"."""
    f = index.get((ident(sku), ident(fc)))
    if f is None:
        f = index.get((ident(sku), POOLED_FC), (float("nan"), float("nan")))
    return f


def mu_ref_paths(model, openings):
    """`mu_ref_path` for many requests in ONE prediction, aligned with them;
    each carries `template`, `grid` and `features`."""
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


def build_states(requests, cfg, model, r_lookup, tables):
    """The state for each VALIDATED request, aligned with it, and the
    count of forecasts made on unknown features. `tables` maps each
    request's `opening_day` to the feature index (feature_index) of the
    table it reads."""
    if not requests:
        return [], {"requests_with_unknown_features": 0}
    requests = [canonical_request(r) for r in requests]
    unknown, openings = 0, []
    for r in requests:
        f = features_of(tables[r["opening_day"]], r["sku_id"], r["fc"])
        unknown += nan_pair(f)
        openings.append({"template": _template(r),
                         "grid": hour_grid(r["date"], r["hour_of_day"], r["hours_remaining"]),
                         "features": f})
    paths = mu_ref_paths(model, openings)
    states = [{**{f: r[f] for f in REQUEST_FIELDS},
               "r": float(lookup_r(r_lookup, r["subcategory"], r["category"])),
               "mu_ref_path": list(path), "features": tuple(o["features"])}
              for r, o, path in zip(requests, openings, paths)]
    return states, {"requests_with_unknown_features": int(unknown)}
