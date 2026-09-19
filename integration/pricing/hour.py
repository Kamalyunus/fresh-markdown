"""The hourly command: the shelf at the top of the hour in, a price per
shelf out.

The producers' snapshot rows carry `episode_id`, which they assign. A new id
on a shelf is an entry decision (no anchor); the id the store last priced
on that shelf continues the episode (anchor = the price in force). The
rule for the id is evaluated only to COUNT the ids that disagree with it,
never to override one. Every refused row is recorded (a rejection: seen,
not priced), so a shelf held through a refusal still gives the rule an
hour to step from, and a gap in the record is an hour that was not sent.

Run: python3 price_hour.py --snapshot <rows> --features features/<day>.parquet
        --out <hour>.csv [--report <hour>.json] [--hour YYYY-MM-DDTHH]
        [--workers 0] [--dry-run]
"""
import argparse
import json
import os
import shutil
import tempfile
from collections import Counter

import pandas as pd

from pricing.config import config_digest, load_config
from pricing.decide import StateRejected, decide
from pricing.feed import (RESPONSE_COLS, read_snapshot, write_frame, write_json)
from pricing.keys import as_number as _num, hour_key, hours_between, iso_day, planning_horizon
from pricing.model import DemandModel
from pricing.pool import pmap
from pricing.rule import continues as rule_continues
from pricing.state import (Posterior, build_states, canonical_request, table_as_of,
                           validate_request)
from pricing.store import EventStore, rejection_event

LIVE_RULE = ("The producer's episode_id is read as given. Checked against the chain's "
             "rule (ops.assign_episode_ids.RULE): a shelf continues last hour's "
             "episode when that hour was one earlier, did not close the shelf, and "
             "the counter stepped down by one or up/flat with stock arrived. With "
             "last hour's closed row the check is the producer's own script; from "
             "the store alone only a counter RESET is decisive (the store never sees "
             "a close or a restock), so the rest is counted as not decidable. A "
             "disagreement is counted and sampled, never overridden; a shelf with no "
             "hour to step from is counted as a gap, never as a disagreement.")


# ----------------------------------------------------------- the snapshot

def split_hours(rows, hour=None):
    """(the batch hour, its opening rows, last hour's closed rows by shelf)."""
    keyed = [r for r in rows if r["key"] is not None]
    if hour:
        day, hh = hour.split("T")
        when = (iso_day(day), int(hh))
    elif keyed:
        when = max((r["key"][2], r["key"][3]) for r in keyed)
    else:
        when = None
    openings = [r for r in rows if r["key"] is None or (r["key"][2], r["key"][3]) == when]
    closed = {}
    for r in keyed:
        if when and hours_between(r["key"][2], r["key"][3], when[0], when[1]) == 1:
            closed[r["key"][:2]] = r
    return when, openings, closed


def rule_says_continues(seen, row, closed_row):
    """(verdict, basis): the rule with last hour's closed row is decisive
    ("closed_row"); from the store alone only a counter reset is ("reset");
    a one-hour step it cannot judge is "indeterminate"; no hour to step
    from is a "gap"."""
    if closed_row is not None:
        prev = {"date": closed_row["key"][2], "hour": closed_row["key"][3],
                "ending_inventory": closed_row.get("ending_inventory"),
                "inventory": closed_row.get("starting_inventory"),
                "units_sold": closed_row.get("units_sold"),
                "flc_window": closed_row.get("hours_remaining"), "episode_id": "x"}
        now = {"date": row["key"][2], "hour": row["key"][3],
               "flc_window": row.get("hours_remaining")}
        return rule_continues(prev, now), "closed_row"
    if seen is None:
        return None, "gap"
    last_h, now_c = _num(seen.get("hours_remaining")), _num(row.get("hours_remaining"))
    if last_h is None or now_c is None:
        return None, "gap"
    if hours_between(seen["date"], seen["hour_of_day"], row["key"][2], row["key"][3]) != 1:
        return None, "gap"
    if float(planning_horizon(now_c)) - last_h < -1:
        return False, "reset"
    return None, "indeterminate"


def _producers_id_last_hour(closed_row, seen_entry):
    if closed_row is not None:
        eid = closed_row.get("episode_id")
        if eid is not None and not (isinstance(eid, float) and eid != eid) and str(eid).strip():
            return eid
        if seen_entry is not None and (seen_entry["date"], seen_entry["hour_of_day"]) == \
                (closed_row["key"][2], closed_row["key"][3]):
            return seen_entry.get("episode_id")
        return None
    return seen_entry.get("episode_id") if seen_entry is not None else None


def _finite(v):
    return _num(v) is not None


def build_requests(openings, closed, latest, seen):
    """The engine's requests for the openings that can be priced and the
    refusals, aligned with `openings`, plus the counts."""
    requests, refused = [], []
    counts = {"shelves_empty": 0, "shelves_unkeyable": 0, "shelves_without_episode_id": 0,
              "episodes_new": 0, "episodes_continued": 0,
              "episode_ids_disagreeing_with_the_rule": 0,
              "episode_ids_the_rule_could_not_check": 0,
              "episode_ids_not_decidable_from_the_store": 0,
              "rejections_recorded_before_the_request": 0, "disagreements_sample": []}

    def refuse(count, why):
        counts[count] += 1
        requests.append(None)
        refused.append(why)

    for r in openings:
        if r["key"] is None:
            refuse("shelves_unkeyable", "row names no shelf-hour")
            continue
        q = _num(r.get("starting_inventory"))
        if q is not None and q <= 0:
            refuse("shelves_empty", "empty shelf: nothing to price")
            continue
        eid = r.get("episode_id")
        if eid is None or (isinstance(eid, float) and eid != eid) or str(eid).strip() == "":
            refuse("shelves_without_episode_id",
                   "episode_id missing: the producer assigns it (ops.assign_episode_ids)")
            continue
        eid = str(eid)
        sku, fc, day, hour = r["key"]
        last = latest.get((sku, fc))
        continued = last is not None and str(last.get("episode_id")) == eid
        if continued:
            counts["episodes_continued"] += 1
            anchor = r["total_discount"] if r["total_discount"] is not None else last["applied_discount"]
        else:
            counts["episodes_new"] += 1
            anchor = None
        closed_row = closed.get((sku, fc))
        seen_entry = seen.get((sku, fc))
        verdict, basis = rule_says_continues(seen_entry, r, closed_row)
        prev_id = _producers_id_last_hour(closed_row, seen_entry)
        if basis == "gap" or prev_id is None:
            counts["episode_ids_the_rule_could_not_check"] += 1
        elif basis == "indeterminate":
            counts["episode_ids_not_decidable_from_the_store"] += 1
        elif verdict != (str(prev_id) == eid):
            counts["episode_ids_disagreeing_with_the_rule"] += 1
            if len(counts["disagreements_sample"]) < 10:
                counts["disagreements_sample"].append({
                    "skuseq": sku, "fc": fc, "episode_id": eid,
                    "producer": "continued" if str(prev_id) == eid else "new",
                    "rule": "continued" if verdict else "new", "basis": basis})
        requests.append({
            "episode_id": eid, "sku_id": sku, "fc": fc,
            "category": r.get("category"), "subcategory": r.get("subcategory"),
            "date": day, "hour_of_day": hour,
            "hours_remaining": (planning_horizon(r["hours_remaining"])
                                if _finite(r.get("hours_remaining")) else None),
            "q": r.get("starting_inventory"),
            "original_price": r.get("original_price"), "cost": r.get("cost"),
            "current_discount": anchor})
        refused.append(None)
    return requests, refused, counts


# --------------------------------------------------------------- the batch

def plan(requests, priced_keys, cfg):
    """Which requests become states: a request missing what a state needs,
    two requests for one hour, and an hour already priced are refused with
    the reason. Returns (to_price [(index, canonical, key)], {index: reason})."""
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
        elif k in priced_keys:
            rejected[i] = "already_priced: the store holds a decision for this hour"
        else:
            to_price.append((i, r, k))
    return to_price, rejected


def price_one(state, ctx):
    """One decision in a worker, pure; the parent commits."""
    try:
        evt = decide(state, ctx["cells"][state["category"]], ctx["cfg"],
                     ctx["model_version"], ctx["digest"])
    except StateRejected as e:
        return {"evt": None, "rejected": str(e)}
    return {"evt": evt, "rejected": None}


def price_batch(cfg, requests, store, model, posterior, r_lookup, table, workers=None):
    """Price `requests`; returns (rows, events, report): one row per request
    in request order, the committed events, the batch's counts."""
    to_price, rejected = plan(requests, store.priced_hours, cfg)
    canon = {i: r for i, r, _ in to_price}
    states, notes = build_states(list(canon.values()), cfg, model, r_lookup,
                                 store.episode_paths, table)
    as_of = table_as_of(table)
    stale = sum(1 for r in canon.values()
                if r["current_discount"] is None and as_of is not None and as_of < r["date"])
    cats = sorted({r["category"] for r in canon.values()})
    ctx = {"cfg": cfg, "cells": {c: posterior.cell(c) for c in cats},
           "model_version": model.version, "digest": config_digest(cfg)}
    results = pmap(price_one, states, ctx, workers=workers)

    priced, engine_rejected, quarantined = {}, Counter(), 0
    for (i, r, k), res in zip(to_price, results):
        if res["evt"] is None:
            rejected[i] = res["rejected"]
            engine_rejected[res["rejected"]] += 1
            continue
        if not store.emit_decision(res["evt"]):
            rejected[i] = store.last_refusal or "the store refused the decision event"
            quarantined += 1
            continue
        priced[i] = res["evt"]
    rejections, recorded = 0, set()
    for i, reason in sorted(rejected.items()):
        evt = rejection_event(canon.get(i, requests[i]), reason)
        if evt is None or evt["rejection_id"] in recorded:
            continue
        recorded.add(evt["rejection_id"])
        if store.emit_rejection(evt):
            rejections += 1
    rows = []
    for i, r in enumerate(requests):
        base = {f: canon.get(i, r).get(f) for f in ("episode_id", "sku_id", "fc", "date", "hour_of_day")}
        evt = priced.get(i)
        rows.append({**base, **{f: (evt[f] if evt else None) for f in
                                ("decision_id", "applied_discount", "applied_price", "is_exploration")},
                     "rejected": None if evt else rejected[i]})
    events = list(priced.values())
    report = {
        "requests": len(requests), "decisions": len(events),
        "rejected": len(requests) - len(events),
        "rejected_before_the_engine": len(requests) - len(to_price),
        "rejected_by_the_engine": dict(engine_rejected),
        "rejections_recorded": rejections, "quarantined": quarantined,
        "explored": 0, **notes,
        "features_as_of": as_of, "entry_requests_on_stale_features": int(stale),
        "tau_in_force": None, "exploration_suspended": posterior.suspended,
        "exploration_mode": "exploit", "model_version": model.version,
        "config_digest": ctx["digest"],
        "posterior_versions": {c: int(v["version"]) for c, v in ctx["cells"].items()},
        "history_rows": 0, "history_dates": None,
    }
    return rows, events, report


# ---------------------------------------------------------------- the hour

def run(cfg, snapshot_rows, table, hour=None, workers=None, dry_run=False, store=None):
    """One hour: the requests from the snapshot, priced, the response in the
    feed's units. `dry_run` prices against a scratch copy of the store and
    commits nothing. Returns (response rows, events, report)."""
    if dry_run:
        scratch = tempfile.mkdtemp(prefix="price_hour_dry_")
        try:
            src = cfg["events"]["store_dir"]
            if os.path.isdir(src):
                shutil.rmtree(scratch)
                shutil.copytree(src, scratch)
            return _run(cfg, snapshot_rows, table, hour, workers, EventStore(cfg, root=scratch), True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    return _run(cfg, snapshot_rows, table, hour, workers, store or EventStore(cfg), False)


def _run(cfg, snapshot_rows, table, hour, workers, store, dry_run):
    model = DemandModel(cfg)
    posterior = Posterior(cfg["posterior"]["path"])
    with open(cfg["dispersion"]["r_lookup_path"]) as f:
        r_lookup = json.load(f)
    when, openings, closed = split_hours(snapshot_rows, hour)
    requests, refused, counts = build_requests(openings, closed, store.latest_by_shelf,
                                               store.last_seen_by_shelf)
    sent = [r for r in requests if r is not None]
    rows, events, rep = price_batch(cfg, sent, store, model, posterior, r_lookup, table, workers)
    answered = iter(rows)
    response = []
    for r, req, why in zip(openings, requests, refused):
        if req is None:
            evt = rejection_event(
                {**r, "hours_remaining": (planning_horizon(r["hours_remaining"])
                                          if _finite(r.get("hours_remaining")) else None),
                 "q_remaining": _num(r.get("starting_inventory"))}, why)
            if evt is not None and store.emit_rejection(evt):
                counts["rejections_recorded_before_the_request"] += 1
            response.append(_response(r.get("sku_id"), r.get("fc"), r.get("date"),
                                      r.get("hour_of_day"), None, None, why))
            continue
        a = next(answered)
        response.append(_response(req["sku_id"], req["fc"], req["date"], req["hour_of_day"],
                                  req["episode_id"], a, a["rejected"]))
    report = {"hour": f"{when[0]}T{when[1]:02d}" if when else None,
              "shelves": len(openings), "closed_rows_seen": len(closed),
              **counts, "dry_run": bool(dry_run), "live_rule": LIVE_RULE, **rep}
    return response, events, report


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
                    help="the shelf at the top of the hour, in the feed's schema "
                         "(parquet, CSV or JSONL); the closed hour's rows may ride along")
    ap.add_argument("--features", required=True, help="today's feature table")
    ap.add_argument("--out", required=True, help="the response: .csv, .parquet or .jsonl")
    ap.add_argument("--report", default=None, help="the hour's counts, JSON")
    ap.add_argument("--hour", default=None, help="YYYY-MM-DDTHH; default the latest hour in the snapshot")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=None, help="0 = every core but one")
    ap.add_argument("--dry-run", action="store_true",
                    help="price against a scratch copy of the store; nothing is committed")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, strict=True)
    response, events, report = run(cfg, read_snapshot(args.snapshot), pd.read_parquet(args.features),
                                   hour=args.hour, workers=args.workers, dry_run=args.dry_run)
    write_frame(pd.DataFrame(response, columns=list(RESPONSE_COLS)), args.out)
    report["response"] = args.out
    if args.report:
        write_json(args.report, report)
    print(f"{report['hour']}: {report['shelves']} shelves -> {report['decisions']} priced "
          f"({report['episodes_new']} new episodes, {report['episodes_continued']} continued), "
          f"{report['rejected']} rejected, {report['shelves_empty']} empty"
          + (f", {report['episode_ids_disagreeing_with_the_rule']} ids disagree with the rule"
             if report["episode_ids_disagreeing_with_the_rule"] else "")
          + " -- EXPLOIT ONLY" + (" -- DRY RUN, nothing committed" if args.dry_run else ""))
    for why, n in sorted(report["rejected_by_the_engine"].items()):
        print(f"  {n:,}  {why}")
    print(f"-> {args.out}")
    return 0
