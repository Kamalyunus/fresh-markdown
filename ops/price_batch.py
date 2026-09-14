"""ops.price_batch -- price one batch of requests through engine.decide.

Lane B's reference caller (design 5.10, RUNBOOK "Lane B"): the 12-field
requests of one hour in, a decision per request out -- the price to apply,
or a rejection with its reason, never a best-effort price. Everything the
engine needs beyond the request is resolved here, by the one home for
each: the state (engine.state.build_states -- the frozen model's
`mu_ref_path`, its two demand-rate features computed point-in-time over
the trailing history, `r` from the lookup), the posterior and tau (ONE
`PosteriorStore` read per batch, so a suspension or an --apply landed by
another process is seen before the first decision), the config digest
(computed once per batch), the event store (every priced decision is
committed before its price is returned -- an hour that is not in the
record is not priced).

Row-scoped, never batch-scoped: a request that cannot become a state, a
state the engine rejects, a duplicate of another request's hour, or an
hour the store already holds a decision for, costs that request and is
returned as `rejected` with the reason; the rest of the batch prices.

Run: python3 -m ops.price_batch --requests <hour.jsonl|.parquet|.csv>
        --history <hourly FLC parquet, the table ingest reads>
        --out decisions.jsonl [--report report.json] [--workers N] [--seed S]
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import read_json, write_json
from common.parallel import map_episodes
from common.provenance import config_fingerprint
from engine.decide import StateRejected, decide
from engine.posterior import PosteriorStore
from engine.state import (HISTORY_COLS, REQUEST_FIELDS, BufferStore, FrozenCells,
                          build_states, validate_request)
from events.pairs import hour_key
from events.store import EventStore
from fit import prepare_data
from fit.train_baseline import BaselineModel

# what a caller gets back per request: the hour, the id the outcome will
# name, the price to put on the shelf -- or why there is none
RESPONSE_FIELDS = ("episode_id", "sku_id", "fc", "date", "hour_of_day",
                   "decision_id", "applied_discount", "applied_price",
                   "is_exploration", "rejected")


# ----------------------------------------------------------------- inputs

def read_requests(path):
    """Requests as a list of dicts, from JSONL (one object per line; a
    line that is not one is a request with no fields, rejected below and
    never a batch-wide raise), parquet or CSV (one row each, in the
    contract's column names; a null cell reads as None)."""
    if path.endswith(".parquet"):
        frame = pd.read_parquet(path)
    elif path.endswith(".csv"):
        frame = pd.read_csv(path)
    else:
        rows = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = {}
                rows.append(obj if isinstance(obj, dict) else {})
        return rows
    frame = frame.astype(object).where(frame.notna(), None)
    return frame.to_dict("records")


def load_history(path, cfg):
    """The feature service's history in HISTORY_COLS: a prepared parquet
    is read as is; the hourly FLC table in the source schema goes through
    the one chain (prepare_data.load_and_filter), so a feature is computed
    on the rows the bootstrap would have computed it on."""
    cols = set(pd.read_parquet(path).columns) if path.endswith(".parquet") else set()
    if {"episode_id", "total_discount", "starting_inventory"} <= cols:
        hist = pd.read_parquet(path, columns=list(HISTORY_COLS))
    else:
        hist, _ = prepare_data.load_and_filter(path, cfg)
        hist = hist[list(HISTORY_COLS)]
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    return hist.reset_index(drop=True)


def plan(requests, priced_keys=()):
    """Which requests become states, and which are refused before the
    engine sees them: a request missing what a state needs
    (validate_request), two requests for one hour (two states for one
    hour -- neither is preferable, the rule ingest applies to the feed),
    and an hour the store already holds a decision for (a retry of a
    priced hour would put two decisions on one feed row). Returns
    (to_price [(index, request, key)], rejected {index: reason})."""
    rejected, keyed = {}, []
    for i, r in enumerate(requests):
        problems = validate_request(r)
        if problems:
            rejected[i] = "; ".join(problems)
            continue
        try:
            k = hour_key(r["sku_id"], r["fc"], r["date"], r["hour_of_day"])
        except (TypeError, ValueError) as exc:
            rejected[i] = f"unkeyable request: {exc}"
            continue
        keyed.append((i, r, k))
    seen = {}
    for i, r, k in keyed:
        seen[k] = seen.get(k, 0) + 1
    to_price = []
    for i, r, k in keyed:
        if seen[k] > 1:
            rejected[i] = "duplicate_request: two requests for one hour"
        elif k in priced_keys:
            rejected[i] = "already_priced: the store holds a decision for this hour"
        else:
            to_price.append((i, r, k))
    return to_price, rejected


# ----------------------------------------------------------------- worker

def _decision_rng(seed, key):
    """One generator per (seed, hour key): the draw does not depend on
    which worker prices the request or in what order."""
    h = hashlib.blake2b("|".join(map(str, key)).encode(), digest_size=8).digest()
    return np.random.default_rng([int(seed), int.from_bytes(h, "big")])


def _price_one(item, ctx):
    """One decision in a worker: pure -- the state, the batch's posterior
    snapshot and tau, a generator seeded from the hour. Returns the event
    or the rejection; the parent commits."""
    state, key = item
    store = BufferStore()
    try:
        evt = decide(state, FrozenCells(ctx["cells"], ctx["suspended"]), store,
                     ctx["cfg"], _decision_rng(ctx["seed"], key), ctx["tau"],
                     ctx["model_version"], config_digest=ctx["digest"])
    except StateRejected as e:
        return {"evt": None, "rejected": str(e)}
    return {"evt": evt, "rejected": None}


# ------------------------------------------------------------------ batch

def run(cfg, requests, history, workers=None, seed=0, store=None, model=None,
        posterior=None, r_lookup=None):
    """Price `requests` (dicts in REQUEST_FIELDS) against `history`
    (HISTORY_COLS). Returns (rows, events, report): one response row per
    request in request order (RESPONSE_FIELDS), the committed decision
    events, and the batch's counts. `store`, `model`, `posterior`,
    `r_lookup` are built from `cfg` unless a long-lived caller holds them;
    the posterior is RELOADED here regardless -- one read per batch."""
    store = store or EventStore(cfg)
    model = model or BaselineModel(cfg)
    posterior = (posterior or PosteriorStore(cfg)).reload()
    r_lookup = r_lookup or read_json(cfg["dispersion"]["r_lookup_path"])
    if r_lookup is None:
        raise FileNotFoundError(cfg["dispersion"]["r_lookup_path"])

    priced = set()
    for d in store.load_decisions():
        try:
            priced.add(hour_key(d["sku_id"], d["fc"], d["date"], d["hour_of_day"]))
        except (KeyError, TypeError, ValueError):
            continue
    to_price, rejected = plan(requests, priced)

    states = build_states([r for _, r, _ in to_price], history, cfg, model, r_lookup)
    cats = sorted({r["category"] for _, r, _ in to_price})
    ctx = {"cfg": cfg, "cells": {c: posterior.get(c) for c in cats},
           "suspended": posterior.exploration_suspended(),
           "tau": posterior.tau(cfg), "seed": int(seed),
           "model_version": model.schema["model_version"],
           "digest": config_fingerprint(cfg)["digest"]}
    results = map_episodes(_price_one, [(s, k) for s, (_, _, k) in zip(states, to_price)],
                           ctx, workers=workers)

    events, quarantined, engine_rejected = [], 0, {}
    priced_rows = {}
    for (i, r, k), res in zip(to_price, results):
        if res["evt"] is None:
            rejected[i] = res["rejected"]
            engine_rejected[res["rejected"]] = engine_rejected.get(res["rejected"], 0) + 1
            continue
        evt = res["evt"]
        if not store.emit_decision(evt):
            # the store refused the event (quarantine.jsonl says why): an
            # hour not in the record is not priced
            rejected[i] = "quarantined: the store refused the decision event"
            quarantined += 1
            continue
        events.append(evt)
        priced_rows[i] = evt

    rows = []
    for i, r in enumerate(requests):
        base = {f: r.get(f) for f in ("episode_id", "sku_id", "fc", "date", "hour_of_day")}
        if i in priced_rows:
            evt = priced_rows[i]
            rows.append({**base, "decision_id": evt["decision_id"],
                         "applied_discount": evt["applied_discount"],
                         "applied_price": evt["applied_price"],
                         "is_exploration": evt["is_exploration"], "rejected": None})
        else:
            rows.append({**base, "decision_id": None, "applied_discount": None,
                         "applied_price": None, "is_exploration": None,
                         "rejected": rejected[i]})
    report = {
        "requests": len(requests),
        "decisions": len(events),
        "rejected": len(requests) - len(events),
        "rejected_before_the_engine": sum(
            1 for i in rejected if i not in {j for j, _, _ in to_price}),
        "rejected_by_the_engine": engine_rejected,
        "quarantined": quarantined,
        "explored": sum(1 for e in events if e["is_exploration"]),
        "tau_in_force": None if ctx["suspended"] else ctx["tau"],
        "exploration_suspended": ctx["suspended"],
        "model_version": ctx["model_version"],
        "config_digest": ctx["digest"],
        "posterior_versions": {c: int(v["version"]) for c, v in ctx["cells"].items()},
        "history_rows": int(len(history)),
        "history_dates": ([str(history.date.min()), str(history.date.max())]
                          if len(history) else None),
    }
    return rows, events, report


def write_rows(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps({k: row.get(k) for k in RESPONSE_FIELDS}) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ops.price_batch")
    ap.add_argument("--requests", required=True,
                    help="one hour's price requests: JSONL, parquet or CSV in "
                         f"the contract's names {list(REQUEST_FIELDS)}")
    ap.add_argument("--history", required=True,
                    help="the hourly FLC table the outcomes are ingested from "
                         "(source schema), or a prepared parquet; the trailing "
                         "ref_rate_window_days the features read")
    ap.add_argument("--out", required=True, help="response JSONL, one row per request")
    ap.add_argument("--report", default=None, help="the batch's counts, JSON")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes pricing the batch (0 = every core but one; "
                         "default serial); the same answer either way")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, strict=True)
    requests = read_requests(args.requests)
    history = load_history(args.history, cfg)
    rows, events, report = run(cfg, requests, history, workers=args.workers,
                               seed=args.seed)
    write_rows(rows, args.out)
    if args.report:
        write_json(args.report, report)
    print(f"requests {report['requests']:,}: {report['decisions']:,} priced "
          f"({report['explored']:,} explored), {report['rejected']:,} rejected"
          + (f" [{report['quarantined']} quarantined]" if report["quarantined"] else "")
          + (" -- exploration SUSPENDED" if report["exploration_suspended"] else ""))
    for why, n in sorted(report["rejected_by_the_engine"].items()):
        print(f"  {n:,}  {why}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
