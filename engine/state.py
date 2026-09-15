"""engine.state -- the price request becomes the engine's state (design 5.10).

The 12-field request (docs/event_contract.html section 03) is not what
`engine.decide` prices: the engine also needs the frozen model's
`mu_ref_path` over the remaining hours -- whose two demand-rate features
are computed point-in-time from the trailing feed by the one home,
`fit.prepare_data.add_ref_rate_features` -- and the dispersion `r` from the
lookup. This module is the one place a request turns into a state, and
the one worker body that prices one (`price_one`). `ops.price_batch`
prices a batch of requests through it.

A request is read in ONE spelling (`canonical_request`: ids as the hour
key spells them, the day as `YYYY-MM-DD`, counts as ints), so a parquet
timestamp, an id read back as 7.0 and a JSONL "7" all name the same hour
and the same history rows. An episode's forecast is made ONCE, at its
entry decision: a later request of the same episode is priced on that
stored path, sliced to the hour (`episode_paths`, from the event store),
extended by prediction only when the window grew (a restock) -- so
serving equals the entry forecast, assurance can re-solve it, and a
mid-episode request costs no history pass.
"""

import math

import numpy as np
import pandas as pd

from common.config import reference_discount
from common.parallel import keyed_rng
from common.provenance import config_fingerprint
from engine.decide import StateRejected, count_failures, decide
from events.pairs import ident, ident_series, iso_day
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

_NUMBER = (int, float, np.integer, np.floating)


def _null(v):
    return v is None or (isinstance(v, (float, np.floating)) and not math.isfinite(v))


def validate_request(r, cfg):
    """What a request must carry before a state can be BUILT from it: every
    field present, a day and an hour the grid can be laid on, counts that
    are counts (engine.decide.count_failures -- the same three checks and
    the horizon bound the state is judged on), ids that name an item, a
    price and a cost that are numbers at all. Their sign, finiteness and
    the anchor are judged by `engine.decide` (economics_failures,
    validate_state) -- nothing is checked twice. Returns the problems, []
    for a request that can become a state."""
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
    """A VALIDATED request in the one spelling every consumer keys on:
    ids through `events.pairs.ident` (the hour key's spelling, so a
    request meets its history rows and its stored decisions whatever the
    producer's dtype), the day as `YYYY-MM-DD`, counts as ints, the
    price and cost as floats (their value is the engine's to judge), a
    null anchor as None. Idempotent."""
    anchor = r["current_discount"]
    if _null(anchor):
        anchor = None                      # a null read from a table
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
    """(date "YYYY-MM-DD", hour) for every hour of a window opening on
    `day` at `opening_hour` -- midnight is an ordinary hour."""
    base = pd.Timestamp(day) + pd.Timedelta(hours=opening_hour)
    return [((base + pd.Timedelta(hours=k)).strftime("%Y-%m-%d"),
             int((base + pd.Timedelta(hours=k)).hour)) for k in range(n_hours)]


def hours_between(day_a, hour_a, day_b, hour_b):
    """Whole hours from (day_a, hour_a) to (day_b, hour_b); negative when
    b is earlier."""
    a = pd.Timestamp(day_a) + pd.Timedelta(hours=int(hour_a))
    b = pd.Timestamp(day_b) + pd.Timedelta(hours=int(hour_b))
    return int(round((b - a).total_seconds() / 3600.0))


def ref_rate_features(history, openings, cfg):
    """The two demand-rate features for episodes OPENING today, computed
    point-in-time by the one home (fit.prepare_data.add_ref_rate_features)
    over the trailing history -- the prepared extract plus every hour fed
    since. `openings` rows carry episode_id, sku_id, fc, category, date,
    hour_of_day, starting_inventory; they enter as the day's first hour
    with no sales (not anchor rows), so they read yesterday and before.
    Ids on BOTH sides are read in the hour key's spelling
    (events.pairs.ident_series): an int history against a text request
    once merged nothing and priced every request on "unknown".
    Returns {episode_id: (sku_ref_sales_rate_30d,
    prior_episode_ref_sales_rate)} with NaN where history is empty -- the
    model's own encoding of "unknown"."""
    cols = list(HISTORY_COLS)
    stub = openings.assign(units_sold=0, total_discount=np.nan)[cols]
    frame = pd.concat([history[cols], stub], ignore_index=True)
    for col in ("sku_id", "fc", "episode_id"):
        frame[col] = ident_series(frame[col])
    frame = frame[frame.sku_id.notna() & frame.fc.notna()]
    # one lookup per category, mapped over the column (a per-row lambda
    # was a config read per history row, every morning)
    d_ref = {c: reference_discount(cfg, c) for c in frame.category.unique()}
    frame["d_ref"] = frame.category.map(d_ref)
    feats = add_ref_rate_features(frame, cfg)
    mine = feats[feats.episode_id.isin(set(ident_series(stub.episode_id)))]
    return {r.episode_id: (float(r.sku_ref_sales_rate_30d),
                           float(r.prior_episode_ref_sales_rate))
            for r in mine.itertuples()}


def features_unknown(feats):
    """True when the model will read this opening as "unknown": neither
    demand-rate feature found any history."""
    return all(isinstance(v, float) and math.isnan(v) for v in feats)


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


def _template(r):
    return {"category": r["category"], "subcategory": r["subcategory"],
            "fc": r["fc"], "original_price": float(r["original_price"])}


def assemble_state(request, r, mu_ref_path):
    """THE state `engine.decide` prices, in one spelling: the request's 12
    fields (REQUEST_FIELDS, in the contract's order), the dispersion `r`
    and the frozen model's `mu_ref_path` from this hour on. Lane B's
    batch (`build_states`), the simulator's pilot hour and shadow's
    re-anchored hour each spelt this dict for themselves."""
    return {**{f: request[f] for f in REQUEST_FIELDS},
            "r": float(r), "mu_ref_path": list(mu_ref_path)}


def build_states(requests, history, cfg, model, r_lookup, episode_paths=None):
    """The engine's state for each VALIDATED request, aligned with it: the
    request's fields (canonical_request), `r` down the lookup's fallback
    chain, and `mu_ref_path` over `hours_remaining` consecutive hours from
    the request's own (date, hour).

    An ENTRY request (null anchor) is a new forecast: features computed
    point-in-time over `history` (HISTORY_COLS), predicted once for the
    whole batch. A LATER request of an episode is priced on the path the
    store holds for it (`episode_paths`, events.store.EventStore: the
    latest decision's date, hour, hours_remaining, mu_ref_path and the
    hour the episode `opened`), sliced to this hour; when the request's
    horizon runs past the stored path (a restock extended the window) the
    missing hours are predicted on features as of the episode's OPENING,
    never today's, and appended. A later request whose episode the store
    does not know (a listing already on clearance when the pilot began, a
    request earlier than the stored decision) falls back to a fresh
    forecast and is counted.

    Returns (states, notes): `notes` carries
    `requests_with_unknown_features` (a fresh forecast with no history
    behind it -- the model prices it as "unknown") and
    `non_entry_requests_without_stored_path`."""
    if not requests:
        return [], {"requests_with_unknown_features": 0,
                    "non_entry_requests_without_stored_path": 0}
    requests = [canonical_request(r) for r in requests]
    paths = episode_paths or {}
    fresh, tails, sliced = {}, {}, {}      # index -> what to predict / the path
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
            opened = stored.get("opened") or (stored["date"], stored["hour_of_day"])
            tails[i] = (path, tuple(opened))
        else:
            sliced[i] = path

    # ONE history pass for every opening that needs a prediction: an entry
    # request as of its own hour, a restock extension as of the episode's
    # opening (the features the entry forecast stood on)
    stub_rows = {}
    for i, (day, hour) in fresh.items():
        r = requests[i]
        stub_rows.setdefault(r["episode_id"], {
            "episode_id": r["episode_id"], "sku_id": r["sku_id"], "fc": r["fc"],
            "category": r["category"], "date": day, "hour_of_day": hour,
            "starting_inventory": r["q"]})
    for i, (_, opened) in tails.items():
        r = requests[i]
        stub_rows.setdefault(r["episode_id"], {
            "episode_id": r["episode_id"], "sku_id": r["sku_id"], "fc": r["fc"],
            "category": r["category"], "date": opened[0], "hour_of_day": opened[1],
            "starting_inventory": r["q"]})
    feats = (ref_rate_features(history, pd.DataFrame(list(stub_rows.values())), cfg)
             if stub_rows else {})
    unknown = 0
    openings, owners = [], []
    for i in fresh:
        r = requests[i]
        f = feats[r["episode_id"]]
        unknown += features_unknown(f)
        openings.append({"template": _template(r),
                         "grid": hour_grid(r["date"], r["hour_of_day"], r["hours_remaining"]),
                         "features": f})
        owners.append(i)
    for i, (path, _) in tails.items():
        r = requests[i]
        grid = hour_grid(r["date"], r["hour_of_day"], r["hours_remaining"])[len(path):]
        openings.append({"template": _template(r), "grid": grid,
                         "features": feats[r["episode_id"]]})
        owners.append(i)
    predicted = dict(zip(owners, mu_ref_paths(model, openings)))
    for i, (path, _) in tails.items():
        sliced[i] = path + predicted[i]
    for i in fresh:
        sliced[i] = predicted[i]

    states = [assemble_state(r, lookup_r(r_lookup, r["subcategory"], r["category"]),
                             sliced[i]) for i, r in enumerate(requests)]
    return states, {"requests_with_unknown_features": int(unknown),
                    "non_entry_requests_without_stored_path": int(without_stored)}


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


def batch_context(cfg, posterior, model, categories, seed, **fixed):
    """The read-only context every `price_one` of one batch reads: the
    posterior snapshot for `categories` (`cells`, the cell map applied once
    by the real store; `suspended`), the `tau` in force, `cfg`, `seed`,
    the model version and the config `digest` (computed ONCE here: the
    fingerprint costs about as much as a solve). `fixed` replaces a key a
    caller settles for itself -- shadow prices at its own tau, is never
    suspended, and adds the grain its worker re-reads the drift by -- so
    the shape stays one shape. Lane B's batch, the simulator's tick and
    shadow's run each spelt this dict for themselves."""
    ctx = {"cfg": cfg, "cells": {str(c): posterior.get(c) for c in categories},
           "suspended": posterior.exploration_suspended(),
           "tau": posterior.tau(cfg), "seed": int(seed),
           "model_version": model.version,
           "digest": config_fingerprint(cfg)["digest"]}
    ctx.update(fixed)
    return ctx


def price_one(item, ctx, rng=None, spread_sink=None):
    """ONE decision in a worker, pure: `item` is (state, key) -- the key
    is what seeds the draw (the hour key for a batch, the episode and its
    hour for the simulator; common.parallel.keyed_rng), so the answer does
    not depend on which worker prices it or in what order. `ctx` is
    `batch_context`'s: the batch's posterior snapshot (`cells`,
    `suspended`), `tau`, `cfg`, `seed`, `model_version` and the config
    `digest`. Returns {"evt", "rejected"}; the parent commits. The one
    worker body the batch caller, the simulator and shadow share: shadow
    hands in `rng` (one stream per episode, drawn across its hours) and a
    `spread_sink` for the Q-spreads its ledger folds."""
    state, key = item
    store = BufferStore()
    try:
        evt = decide(state, FrozenCells(ctx["cells"], ctx["suspended"]), store,
                     ctx["cfg"], keyed_rng(ctx["seed"], *key) if rng is None else rng,
                     ctx["tau"], ctx["model_version"], spread_sink=spread_sink,
                     config_digest=ctx["digest"])
    except StateRejected as e:
        return {"evt": None, "rejected": str(e)}
    return {"evt": evt, "rejected": None}
