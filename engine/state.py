"""engine.state -- the price request becomes the engine's state (design 5.10).

The 12-field request (docs/event_contract.html section 03) is not what
`engine.decide` prices: the engine also needs the frozen model's
`mu_ref_path` over the remaining hours -- whose two demand-rate features
are computed point-in-time from the trailing feed by the one home,
`fit.prepare_data.add_ref_rate_features` -- and the dispersion `r` from the
lookup. This module is the one place a request turns into a state.
`ops.price_batch` prices a batch of requests through it; the pilot
simulator opens its episodes through the same functions, so a rehearsal
and production cannot compute a feature two ways.
"""

import math

import numpy as np
import pandas as pd

from common.config import reference_discount
from fit.fit_dispersion import lookup_r
from fit.prepare_data import add_ref_rate_features

# the request, in the contract's names and order (section 03)
REQUEST_FIELDS = ("episode_id", "sku_id", "fc", "category", "subcategory",
                  "date", "hour_of_day", "hours_remaining", "q",
                  "original_price", "cost", "current_discount")

# the feature service's history, in the prepared vocabulary: what the two
# demand-rate features read, and nothing more
HISTORY_COLS = ("episode_id", "sku_id", "fc", "category", "date", "hour_of_day",
                "starting_inventory", "units_sold", "total_discount")


def _count(v):
    return (isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))) \
        or (isinstance(v, float) and math.isfinite(v) and v == int(v))


def validate_request(r):
    """What a request must carry before a state can be BUILT from it: every
    field present, a day and an hour the grid can be laid on, counts that
    are counts. Prices, cost and the anchor are judged by `engine.decide`
    (economics_failures, validate_state) -- nothing is checked twice.
    Returns the problems, [] for a request that can become a state."""
    problems = [f"missing {f}" for f in REQUEST_FIELDS if f not in r]
    if problems:
        return problems
    try:
        day = pd.Timestamp(r["date"])
        if pd.isna(day):
            raise ValueError
    except (TypeError, ValueError):
        problems.append(f"date {r['date']!r} names no day")
    if not (_count(r["hour_of_day"]) and 0 <= int(r["hour_of_day"]) <= 23):
        problems.append("hour_of_day must be an integer in 0..23")
    if not (_count(r["hours_remaining"]) and int(r["hours_remaining"]) >= 1):
        problems.append("hours_remaining must be an integer >= 1")
    if not (_count(r["q"]) and int(r["q"]) >= 0):
        problems.append("q must be a non-negative integer")
    for f in ("episode_id", "sku_id", "fc", "category", "subcategory"):
        if r[f] is None or (isinstance(r[f], float) and not math.isfinite(r[f])):
            problems.append(f"{f} is null")
    return problems


def hour_grid(day, opening_hour, n_hours):
    """(date "YYYY-MM-DD", hour) for every hour of a window opening on
    `day` at `opening_hour` -- midnight is an ordinary hour."""
    base = pd.Timestamp(day) + pd.Timedelta(hours=opening_hour)
    return [((base + pd.Timedelta(hours=k)).strftime("%Y-%m-%d"),
             int((base + pd.Timedelta(hours=k)).hour)) for k in range(n_hours)]


def ref_rate_features(history, openings, cfg):
    """The two demand-rate features for episodes OPENING today, computed
    point-in-time by the one home (fit.prepare_data.add_ref_rate_features)
    over the trailing history -- the prepared extract plus every hour fed
    since. `openings` rows carry episode_id, sku_id, fc, category, date,
    hour_of_day, starting_inventory; they enter as the day's first hour
    with no sales (not anchor rows), so they read yesterday and before.
    Returns {episode_id: (sku_ref_sales_rate_30d,
    prior_episode_ref_sales_rate)} with NaN where history is empty -- the
    model's own encoding of "unknown"."""
    cols = list(HISTORY_COLS)
    stub = openings.assign(units_sold=0, total_discount=np.nan)[cols]
    frame = pd.concat([history[cols], stub], ignore_index=True)
    # one lookup per category, mapped over the column (a per-row lambda
    # was a config read per history row, every morning)
    d_ref = {c: reference_discount(cfg, c) for c in frame.category.unique()}
    frame["d_ref"] = frame.category.map(d_ref)
    feats = add_ref_rate_features(frame, cfg)
    mine = feats[feats.episode_id.isin(set(stub.episode_id))]
    return {r.episode_id: (float(r.sku_ref_sales_rate_30d),
                           float(r.prior_episode_ref_sales_rate))
            for r in mine.itertuples()}


def mu_ref_paths(model, openings):
    """`mu_ref_path` for many openings in ONE prediction: a frame of every
    (opening, hour) row, predicted once, split back. Per-episode
    prediction spent 30 ms of pandas per episode -- at 5,000 a day that
    was the run. Returns a list aligned with `openings`; each opening
    carries `template` (category, subcategory, fc, original_price),
    `grid`, `features`."""
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


def build_states(requests, history, cfg, model, r_lookup):
    """The engine's state for each VALIDATED request, aligned with it: the
    request's fields, `r` down the lookup's fallback chain, and
    `mu_ref_path` over `hours_remaining` consecutive hours from the
    request's own (date, hour), predicted once for the whole batch on
    features computed point-in-time over `history` (HISTORY_COLS)."""
    if not requests:
        return []
    stub = pd.DataFrame([{
        "episode_id": r["episode_id"], "sku_id": r["sku_id"], "fc": r["fc"],
        "category": r["category"], "date": str(pd.Timestamp(r["date"]).date()),
        "hour_of_day": int(r["hour_of_day"]), "starting_inventory": int(r["q"])}
        for r in requests]).drop_duplicates("episode_id")
    feats = ref_rate_features(history, stub, cfg)
    openings = []
    for r in requests:
        day = str(pd.Timestamp(r["date"]).date())
        openings.append({
            "template": {"category": r["category"], "subcategory": r["subcategory"],
                         "fc": r["fc"], "original_price": float(r["original_price"])},
            "grid": hour_grid(day, int(r["hour_of_day"]), int(r["hours_remaining"])),
            "features": feats[r["episode_id"]]})
    paths = mu_ref_paths(model, openings)
    states = []
    for r, path in zip(requests, paths):
        anchor = r["current_discount"]
        if isinstance(anchor, float) and math.isnan(anchor):
            anchor = None                      # a null read from a table
        states.append({
            "episode_id": r["episode_id"], "sku_id": r["sku_id"], "fc": r["fc"],
            "category": r["category"], "subcategory": r["subcategory"],
            "date": str(pd.Timestamp(r["date"]).date()),
            "hour_of_day": int(r["hour_of_day"]),
            "hours_remaining": int(r["hours_remaining"]), "q": int(r["q"]),
            "original_price": r["original_price"], "cost": r["cost"],
            "r": float(lookup_r(r_lookup, r["subcategory"], r["category"])),
            "mu_ref_path": path, "current_discount": anchor,
        })
    return states


class FrozenCells:
    """The posterior as a worker sees it: the cells resolved in the parent
    for this batch (the cell map, including the fallback to the global
    cell, applied exactly once by the real store) and the suspension in
    force. A batch is priced against one snapshot -- `PosteriorStore`
    is reloaded once per batch, never per decision."""

    def __init__(self, by_category, suspended=None):
        self._by_category, self._suspended = by_category, suspended

    def get(self, category):
        return self._by_category[str(category)]

    def exploration_suspended(self):
        return self._suspended


class BufferStore:
    """Buffers decision events instead of writing them: workers must not
    touch the event store. The parent commits every event through the real
    store, in request order."""

    def __init__(self):
        self.decisions = []

    def emit_decision(self, event):
        self.decisions.append(event)
        return True
