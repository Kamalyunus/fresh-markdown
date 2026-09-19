"""The hourly command: the shelf at the top of the hour in, a price per
shelf out. Stateless: every row is priced from the row, the artifacts and
the feature table of its episode's opening day; nothing is looked up from
an earlier hour, and the log this hour appends to is never read here.

The snapshot is the twelve request fields (handover Appendix C), as sent.
The row's `episode_id` is the opening tag of its episode (`<sku>|<fc>|<day>T<hh>`,
the shelf-hour the episode began), so the row is an ENTRY when the tag names
this shelf-hour and a later hour of the episode otherwise. `current_discount`
is the price in force, a fraction: null on an entry row, and on a later row
the discount this service applied last hour, piped back by the producers --
the anchor a later hour may only step deeper from. `hours_remaining` counts
this hour.

Run: python3 price_hour.py --snapshot <rows> [--features features/]
        --out <hour>.csv [--report <hour>.json] [--hour YYYY-MM-DDTHH]
        [--workers 0] [--dry-run]
"""
import argparse
import json
import os
from collections import Counter

import pandas as pd

from pricing.config import config_digest, load_config
from pricing.decide import StateRejected, decide
from pricing.feed import RESPONSE_COLS, read_snapshot, write_frame, write_json
from pricing.keys import as_number as _num, hour_key, iso_day
from pricing.log import EventLog, rejection_event
from pricing.model import DemandModel
from pricing.pool import pmap
from pricing.state import (Posterior, build_states, canonical_request, episode_position,
                           feature_index, validate_request)

CONTRACT = ("Stateless. episode_id is the opening tag <sku>|<fc>|<day>T<hh>: an entry when "
            "it names this shelf-hour, a later hour of the episode otherwise. current_discount "
            "is the price in force, a fraction: null on an entry, the discount applied last "
            "hour on a later row (the anchor). hours_remaining counts this hour. The features "
            "are the opening day's table. Nothing is looked up from an earlier hour; the log "
            "is append-only and never read here.")


# ----------------------------------------------------------- the snapshot

def split_hours(rows, hour=None):
    """(the batch hour, its rows, the count of other hours' rows that rode along)."""
    keyed = [r for r in rows if r["key"] is not None]
    if hour:
        day, hh = hour.split("T")
        when = (iso_day(day), int(hh))
    elif keyed:
        when = max((r["key"][2], r["key"][3]) for r in keyed)
    else:
        when = None
    openings = [r for r in rows if r["key"] is None or (r["key"][2], r["key"][3]) == when]
    return when, openings, len(rows) - len(openings)


def _whole(v):
    """A count as an int when it is one (a CSV reads 4 as 4.0); else as it came."""
    n = _num(v)
    return int(n) if n is not None and n == int(n) else v


def build_requests(openings):
    """The engine's requests for the rows that can be priced and the
    refusals, aligned with `openings`, plus the counts."""
    requests, refused = [], []
    counts = {"shelves_empty": 0, "shelves_unkeyable": 0, "entries": 0, "later_hours": 0,
              "episode_ids_that_place_no_hour": 0,
              "later_hours_without_the_price_in_force": 0,
              "entries_with_a_price_in_force": 0}

    def refuse(count, why):
        counts[count] += 1
        requests.append(None)
        refused.append(why)

    for r in openings:
        if r["key"] is None:
            refuse("shelves_unkeyable", "row names no shelf-hour")
            continue
        q = _num(r.get("q"))
        if q is not None and q <= 0:
            refuse("shelves_empty", "empty shelf: nothing to price")
            continue
        position, opening = episode_position(r.get("episode_id"), r["key"])
        if opening is None:
            refuse("episode_ids_that_place_no_hour", position)
            continue
        anchor = _num(r.get("current_discount"))
        if position == "entry":
            counts["entries"] += 1
            if anchor is not None:
                counts["entries_with_a_price_in_force"] += 1     # ignored: an entry has no anchor
            anchor = None
        else:
            if anchor is None:
                refuse("later_hours_without_the_price_in_force",
                       "later hour of the episode without the price in force: the producers "
                       "pipe the discount applied last hour")
                continue
            counts["later_hours"] += 1
        sku, fc, day, hour = r["key"]
        requests.append({
            "episode_id": str(r["episode_id"]).strip(), "sku_id": sku, "fc": fc,
            "category": r.get("category"), "subcategory": r.get("subcategory"),
            "date": day, "hour_of_day": hour,
            "hours_remaining": _whole(r.get("hours_remaining")), "q": _whole(r.get("q")),
            "original_price": r.get("original_price"), "cost": r.get("cost"),
            "current_discount": anchor, "opening_day": opening[2]})
        refused.append(None)
    return requests, refused, counts


# ------------------------------------------------------ the feature tables

def feature_tables(days, features, batch_day, cfg):
    """{day: feature index} for the opening days the requests need, and
    the notes. `features` is one file (used for every day) or a directory
    of `<day>.parquet` (default `features.table_dir`): a day whose table is
    not there reads the batch day's and is listed under `fallback`; a day
    with neither is listed under `missing` and its requests are refused."""
    tables, read, fallback, missing = {}, {}, [], []
    if features and os.path.isfile(features):
        idx = feature_index(pd.read_parquet(features))
        return {d: idx for d in days}, {"feature_tables": {d: features for d in days},
                                        "fallback": [], "missing": []}
    root = features or cfg["features"]["table_dir"]
    for day in sorted(days):
        path = os.path.join(root, f"{day}.parquet")
        if not os.path.exists(path):
            fallback.append(day)
            path = os.path.join(root, f"{batch_day}.parquet")
            if not os.path.exists(path):
                missing.append(day)
                continue
        if path not in read:
            read[path] = feature_index(pd.read_parquet(path))
        tables[day] = read[path]
    paths = {d: os.path.join(root, f"{d if d not in fallback else batch_day}.parquet")
             for d in tables}
    return tables, {"feature_tables": paths, "fallback": fallback, "missing": missing}


# --------------------------------------------------------------- the batch

def plan(requests, cfg):
    """Which requests become states: a request missing what a state needs
    and two requests for one hour are refused with the reason. Returns
    (to_price [(index, canonical, key)], {index: reason})."""
    rejected, keyed = {}, []
    for i, r in enumerate(requests):
        problems = validate_request(r, cfg)
        if problems:
            rejected[i] = "; ".join(problems)
            continue
        c = canonical_request(r)
        keyed.append((i, c, hour_key(c["sku_id"], c["fc"], c["date"], c["hour_of_day"])))
    seen, twice = set(), set()
    for _, _, k in keyed:
        if k in seen:
            twice.add(k)
        seen.add(k)
    to_price = []
    for i, r, k in keyed:
        if k in twice:
            rejected[i] = "duplicate_request: two requests for one hour"
        else:
            to_price.append((i, r, k))
    return to_price, rejected


def price_one(state, ctx):
    """One decision in a worker, pure; the parent records."""
    try:
        evt = decide(state, ctx["cells"][state["category"]], ctx["cfg"],
                     ctx["model_version"], ctx["digest"])
    except StateRejected as e:
        return {"evt": None, "rejected": str(e)}
    return {"evt": evt, "rejected": None}


def price_batch(cfg, requests, features, batch_day, model, posterior, r_lookup, workers=None):
    """Price `requests`; returns (rows, decisions, rejections, report): one
    row per request in request order, the events to log, the counts."""
    to_price, rejected = plan(requests, cfg)
    tables, notes = feature_tables({r["opening_day"] for _, r, _ in to_price},
                                   features, batch_day, cfg)
    for i, r, _ in to_price:
        if r["opening_day"] in notes["missing"]:
            rejected[i] = (f"no feature table for the episode's opening day {r['opening_day']} "
                           f"nor for {batch_day}: run build_features.py")
    to_price = [t for t in to_price if t[0] not in rejected]
    canon = {i: r for i, r, _ in to_price}
    states, state_notes = build_states(list(canon.values()), cfg, model, r_lookup, tables)
    cats = sorted({r["category"] for r in canon.values()})
    ctx = {"cfg": cfg, "cells": {c: posterior.cell(c) for c in cats},
           "model_version": model.version, "digest": config_digest(cfg)}
    results = pmap(price_one, states, ctx, workers=workers)

    priced, engine_rejected = {}, Counter()
    for (i, r, k), res in zip(to_price, results):
        if res["evt"] is None:
            rejected[i] = res["rejected"]
            engine_rejected[res["rejected"]] += 1
        else:
            priced[i] = res["evt"]
    rejections, recorded = [], set()
    for i, reason in sorted(rejected.items()):
        evt = rejection_event(canon.get(i, requests[i]), reason)
        if evt is None or evt["rejection_id"] in recorded:
            continue
        recorded.add(evt["rejection_id"])
        rejections.append(evt)
    rows = []
    for i, r in enumerate(requests):
        base = {f: canon.get(i, r).get(f) for f in ("episode_id", "sku_id", "fc", "date", "hour_of_day")}
        evt = priced.get(i)
        rows.append({**base, **{f: (evt[f] if evt else None) for f in
                                ("decision_id", "applied_discount", "applied_price", "is_exploration")},
                     "rejected": None if evt else rejected[i]})
    decisions = list(priced.values())
    report = {
        "requests": len(requests), "decisions": len(decisions),
        "rejected": len(requests) - len(decisions),
        "rejected_before_the_engine": len(requests) - len(to_price),
        "rejected_by_the_engine": dict(engine_rejected),
        **state_notes,
        "feature_tables": notes["feature_tables"],
        "opening_days_on_the_batch_day_table": notes["fallback"],
        "opening_days_without_a_table": notes["missing"],
        "exploration_mode": "exploit", "model_version": model.version,
        "config_digest": ctx["digest"],
        "posterior_versions": {c: int(v["version"]) for c, v in ctx["cells"].items()},
    }
    return rows, decisions, rejections, report


# ---------------------------------------------------------------- the hour

def run(cfg, snapshot_rows, features=None, hour=None, workers=None, dry_run=False):
    """One hour: the requests from the snapshot, priced, the response in the
    feed's units; the decisions and rejections appended to the log unless
    `dry_run`. Returns (response rows, decisions, report)."""
    model = DemandModel(cfg)
    posterior = Posterior(cfg["posterior"]["path"])
    with open(cfg["dispersion"]["r_lookup_path"]) as f:
        r_lookup = json.load(f)
    log = EventLog(cfg, enabled=not dry_run)
    when, openings, other_hours = split_hours(snapshot_rows, hour)
    requests, refused, counts = build_requests(openings)
    sent = [r for r in requests if r is not None]
    rows, decisions, rejections, rep = price_batch(
        cfg, sent, features, when[0] if when else None, model, posterior, r_lookup, workers)
    recorded = {e["rejection_id"] for e in rejections}
    answered = iter(rows)
    response = []
    for r, req, why in zip(openings, requests, refused):
        if req is None:
            evt = rejection_event({**r, "hours_remaining": _whole(r.get("hours_remaining")),
                                   "q_remaining": _whole(r.get("q"))}, why)
            if evt is not None and evt["rejection_id"] not in recorded:
                recorded.add(evt["rejection_id"])
                rejections.append(evt)
            response.append(_response(r.get("sku_id"), r.get("fc"), r.get("date"),
                                      r.get("hour_of_day"), r.get("episode_id"), None, why))
            continue
        a = next(answered)
        response.append(_response(req["sku_id"], req["fc"], req["date"], req["hour_of_day"],
                                  req["episode_id"], a, a["rejected"]))
    logged = {"decisions_logged": log.append("decisions", decisions),
              "rejections_logged": log.append("rejections", rejections)}
    report = {"hour": f"{when[0]}T{when[1]:02d}" if when else None,
              "shelves": len(openings), "other_hours_rows_ignored": other_hours,
              **counts, "dry_run": bool(dry_run), "contract": CONTRACT, **rep, **logged}
    return response, decisions, report


def _response(skuseq, fc, date, hour, episode_id, answer, rejected):
    disc = answer["applied_discount"] if answer else None
    return {"skuseq": skuseq, "fc": fc, "date": date, "hour": hour,
            "episode_id": episode_id,
            "decision_id": answer["decision_id"] if answer else None,
            "apply_discount_pct": None if disc is None else round(float(disc) * 100.0, 4),
            "apply_price": answer["applied_price"] if answer else None,
            "is_exploration": answer["is_exploration"] if answer else None,
            "rejected": rejected}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="price_hour.py")
    ap.add_argument("--snapshot", required=True,
                    help="the shelf at the top of the hour: the twelve request fields "
                         "(parquet, CSV or JSONL); other hours' rows are ignored")
    ap.add_argument("--features", default=None,
                    help="the feature tables: a directory of <day>.parquet (default "
                         "features/), or one file used for every request")
    ap.add_argument("--out", required=True, help="the response: .csv, .parquet or .jsonl")
    ap.add_argument("--report", default=None, help="the hour's counts, JSON")
    ap.add_argument("--hour", default=None, help="YYYY-MM-DDTHH; default the latest hour in the snapshot")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=None, help="0 = every core but one")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the response and the report; append nothing to the log")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    response, decisions, report = run(cfg, read_snapshot(args.snapshot), features=args.features,
                                      hour=args.hour, workers=args.workers, dry_run=args.dry_run)
    write_frame(pd.DataFrame(response, columns=list(RESPONSE_COLS)), args.out)
    report["response"] = args.out
    if args.report:
        write_json(args.report, report)
    print(f"{report['hour']}: {report['shelves']} shelves -> {report['decisions']} priced "
          f"({report['entries']} entries, {report['later_hours']} later hours), "
          f"{report['rejected']} rejected, {report['shelves_empty']} empty"
          + " -- EXPLOIT ONLY" + (" -- DRY RUN, nothing logged" if args.dry_run else ""))
    for why, n in sorted(report["rejected_by_the_engine"].items()):
        print(f"  {n:,}  {why}")
    print(f"-> {args.out}")
    return 0
