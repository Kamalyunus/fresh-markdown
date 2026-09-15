"""ops.price_batch -- price one batch of requests through engine.decide.

Lane B's reference caller (design 5.10, RUNBOOK "Lane B"): the 12-field
requests of one hour in, a decision per request out -- the price to apply,
or a rejection with its reason, never a best-effort price. Everything the
engine needs beyond the request is resolved here, by the one home for
each: the state (engine.state.build_states -- the frozen model's
`mu_ref_path`: a fresh forecast for an entry request on the two
demand-rate features computed point-in-time over the trailing history, the
stored forecast sliced for a later hour of a known episode; `r` from the
lookup), the posterior and tau (ONE `PosteriorStore` read per batch, so a
suspension or an --apply landed by another process is seen before the
first decision), the config digest (computed once per batch), the event
store (every priced decision is committed before its price is returned --
an hour that is not in the record is not priced; the store itself refuses
a second decision for a priced hour).

Row-scoped, never batch-scoped: a request that cannot become a state, a
state the engine rejects, a duplicate of another request's hour, or an
hour the store already holds a decision for, costs that request and is
returned as `rejected` with the reason; the rest of the batch prices.
Every request is read in one spelling (engine.state.canonical_request),
so a parquet timestamp or an id read back as 7.0 prices and writes.

Run: python3 -m ops.price_batch --requests <hour.jsonl|.parquet|.csv>
        --history <hourly FLC parquet, the table ingest reads>
        --out decisions.jsonl [--report report.json] [--workers N] [--seed S]
"""

import argparse
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

from common.config import load_config
from common.io import read_rows, write_json, write_jsonl
from common.parallel import map_episodes
from engine.state import (HISTORY_COLS, REQUEST_FIELDS, batch_context, build_states,
                          canonical_request, price_one, validate_request)
from events.pairs import colliding_keys, hour_key, ident_series
from events.store import EventStore
from fit import prepare_data
from fit.artifacts import load_bundle

# what a caller gets back per request: the hour, the id the outcome will
# name, the price to put on the shelf -- or why there is none
RESPONSE_FIELDS = ("episode_id", "sku_id", "fc", "date", "hour_of_day",
                   "decision_id", "applied_discount", "applied_price",
                   "is_exploration", "rejected")


# ----------------------------------------------------------------- inputs

def read_requests(path):
    """Requests as a list of dicts (common.io.read_rows): JSONL, parquet or
    CSV in the contract's column names; a line that is not an object is a
    request with no fields, rejected below and never a batch-wide raise;
    a null cell reads as None."""
    return read_rows(path)


def load_history(path, cfg):
    """The feature service's history in HISTORY_COLS: a prepared parquet
    is read as is; the hourly FLC table in the source schema goes through
    the one chain (prepare_data.load_and_filter), so a feature is computed
    on the rows the bootstrap would have computed it on. Ids come back in
    the hour key's spelling (events.pairs.ident_series), the day as
    `YYYY-MM-DD`."""
    cols = set(pq.read_schema(path).names) if path.endswith(".parquet") else set()
    if {"episode_id", "total_discount", "starting_inventory"} <= cols:
        hist = pd.read_parquet(path, columns=list(HISTORY_COLS))
    else:
        hist, _ = prepare_data.load_and_filter(path, cfg)
        hist = hist[list(HISTORY_COLS)]
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    for col in ("sku_id", "fc"):
        hist[col] = ident_series(hist[col])
    return hist.reset_index(drop=True)


def plan(requests, priced_keys, cfg):
    """Which requests become states, and which are refused before the
    engine sees them: a request missing what a state needs
    (validate_request), two requests for one hour (two states for one
    hour -- neither is preferable, the rule ingest applies to the feed:
    events.pairs.colliding_keys), and an hour the store already holds a
    decision for (`priced_keys`, the store's own index; a retry of a
    priced hour would put two decisions on one feed row). Returns
    (to_price [(index, canonical request, key)], rejected {index: reason})
    -- the requests handed on are in the one spelling
    (engine.state.canonical_request)."""
    rejected, keyed = {}, []
    for i, r in enumerate(requests):
        problems = validate_request(r, cfg)
        if problems:
            rejected[i] = "; ".join(problems)
            continue
        c = canonical_request(r)
        keyed.append((i, c, hour_key(c["sku_id"], c["fc"], c["date"], c["hour_of_day"])))
    twice = colliding_keys(k for _, _, k in keyed)
    to_price = []
    for i, r, k in keyed:
        if k in twice:
            rejected[i] = "duplicate_request: two requests for one hour"
        elif k in priced_keys:
            rejected[i] = "already_priced: the store holds a decision for this hour"
        else:
            to_price.append((i, r, k))
    return to_price, rejected


# ------------------------------------------------------------------ batch

def run(cfg, requests, history, workers=None, seed=0, store=None, model=None,
        posterior=None, r_lookup=None):
    """Price `requests` (dicts in REQUEST_FIELDS) against `history`
    (HISTORY_COLS). Returns (rows, events, report): one response row per
    request in request order (RESPONSE_FIELDS), the committed decision
    events, and the batch's counts. `store`, `model`, `posterior`,
    `r_lookup` are built from `cfg` unless a long-lived caller holds them
    (fit.artifacts.load_bundle: a missing lookup is a FileNotFoundError
    naming its path); the posterior is RELOADED here regardless -- one
    read per batch."""
    bundle = load_bundle(cfg)
    store = store or EventStore(cfg)
    model = model or bundle.model
    posterior = (posterior or bundle.posterior).reload()
    r_lookup = r_lookup or bundle.r_lookup

    to_price, rejected = plan(requests, store.priced_hours, cfg)
    canon = {i: r for i, r, _ in to_price}

    # the history the batch's features read: this batch's SKUs only (the
    # trailing table is every SKU ever fed; the features are per SKU),
    # matched in the one id spelling whatever dtype the caller's table has
    skus = {r["sku_id"] for r in canon.values()}
    mine = history[ident_series(history.sku_id).isin(skus)] if len(history) else history
    states, notes = build_states(list(canon.values()), mine, cfg, model, r_lookup,
                                 episode_paths=store.episode_paths)
    # the one context every worker reads (engine.state.batch_context): the
    # cells in category order, so the report's posterior_versions read so
    cats = sorted({r["category"] for r in canon.values()})
    ctx = batch_context(cfg, posterior, model, cats, seed)
    results = map_episodes(price_one, [(s, k) for s, (_, _, k) in zip(states, to_price)],
                           ctx, workers=workers)

    priced, engine_rejected, quarantined = {}, Counter(), 0
    for (i, r, k), res in zip(to_price, results):
        if res["evt"] is None:
            rejected[i] = res["rejected"]
            engine_rejected[res["rejected"]] += 1
            continue
        if not store.emit_decision(res["evt"]):
            # the store refused the event (quarantine.jsonl says why, or
            # another batch priced the hour first): an hour not in the
            # record is not priced
            rejected[i] = "quarantined: the store refused the decision event"
            quarantined += 1
            continue
        priced[i] = res["evt"]              # request order: to_price is

    rows = []
    for i, r in enumerate(requests):
        base = {f: canon.get(i, r).get(f)
                for f in ("episode_id", "sku_id", "fc", "date", "hour_of_day")}
        evt = priced.get(i)
        rows.append({**base, "decision_id": evt["decision_id"],
                     "applied_discount": evt["applied_discount"],
                     "applied_price": evt["applied_price"],
                     "is_exploration": evt["is_exploration"], "rejected": None}
                    if evt else
                    {**base, "decision_id": None, "applied_discount": None,
                     "applied_price": None, "is_exploration": None,
                     "rejected": rejected[i]})
    events = list(priced.values())
    report = {
        "requests": len(requests),
        "decisions": len(events),
        "rejected": len(requests) - len(events),
        "rejected_before_the_engine": len(requests) - len(to_price),
        "rejected_by_the_engine": dict(engine_rejected),
        "quarantined": quarantined,
        "explored": sum(1 for e in events if e["is_exploration"]),
        # a fresh forecast with no history behind it prices as "unknown";
        # a batch where every request reads so is a history that does not
        # meet its requests (an id spelling, a table cut too short)
        **notes,
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
    """The response JSONL, RESPONSE_FIELDS per line (common.io.write_jsonl:
    a timestamp or a NaN in a rejected row's echo writes, never raises
    after the batch committed)."""
    write_jsonl(path, rows, fields=RESPONSE_FIELDS)


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
          + (f" [{report['requests_with_unknown_features']} priced on no history]"
             if report["requests_with_unknown_features"] else "")
          + (" -- exploration SUSPENDED" if report["exploration_suspended"] else ""))
    for why, n in sorted(report["rejected_by_the_engine"].items()):
        print(f"  {n:,}  {why}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
