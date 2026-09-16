"""ops.price_hour -- one clock hour of Lane B, from the feed's own rows.

The data producers hand over the shelf as it stands at the top of the
hour, in the hourly feed's schema (fit.prepare_data.SOURCE_TO_CANONICAL:
date, hour, skuseq, fc, inventory, discount in PERCENT, normal_asp,
cogs_wo_vat, flc_window, category, subcategory; units_sold,
ending_inventory and final_price null or absent for the hour that has
not closed) plus `episode_id`, which THEY assign -- it names their
listing, and the rule for it (ops.assign_episode_ids, the chain's
common.windows.EPISODE_RULE for one hour) is theirs to run or port. This
module reads the id as given: a new id on a shelf is an entry decision
(anchor null), the id the store last priced on that shelf continues the
episode (anchor = the price in force, the snapshot's `discount`). It
builds the engine's requests, prices them through ops.price_batch, and
writes the response in the feed's units: one row per shelf with the
discount to apply as a percent and the price it makes, or the reason it
was not priced.

The rule is still evaluated here -- against the rows of the hour before,
when they ride along, else the store's latest decision -- but only to
COUNT the ids that disagree with it (`episode_ids_disagreeing_with_the_rule`,
with a sample), never to override the producer's id.

Run: python3 -m ops.price_hour --snapshot <top-of-hour rows> \\
        --features features/<today>.parquet --out <hour>.csv --report <hour>.json \\
        [--workers 0] [--hour YYYY-MM-DDTHH] [--dry-run]
"""

import argparse
import math
import os
import shutil
import tempfile

import pandas as pd

from common.config import load_config
from common.io import read_rows, write_json, write_jsonl
from common.windows import planning_horizon
from ops.assign_episode_ids import continues as rule_continues
from engine.state import hours_between, load_history
from events.pairs import ident, iso_day
from events.store import EventStore
from fit.prepare_data import SOURCE_TO_CANONICAL
from ops import price_batch

LIVE_RULE = ("The producer's episode_id is read as given. Checked against the chain's "
             "rule (ops.assign_episode_ids.RULE): a shelf continues last hour's "
             "episode when that hour was one earlier, did not close the shelf, and "
             "the counter stepped down by one or up/flat with stock arrived. A "
             "disagreement is counted and sampled, never overridden.")

RESPONSE_COLS = ("skuseq", "fc", "date", "hour", "episode_id", "decision_id",
                 "apply_discount_pct", "apply_price", "is_exploration", "rejected")

# what a request needs from a snapshot row, in the canonical names
_NEEDED = ("episode_id", "sku_id", "fc", "date", "hour_of_day", "starting_inventory",
           "hours_remaining", "original_price", "cost", "category", "subcategory")


def read_snapshot(path):
    """The snapshot file's rows (parquet, CSV or JSONL in the feed's
    names) through `snapshot_rows`."""
    return snapshot_rows(read_rows(path, rename=SOURCE_TO_CANONICAL))


def snapshot_rows(records):
    """Snapshot records (dicts in the feed's names, or already renamed) in
    the canonical names (the discount a fraction, the day `YYYY-MM-DD`),
    each keyed by (sku, fc, date, hour) where the record names one; an
    unkeyable record is kept with `key` None so it costs itself a
    response line, never the batch."""
    rows = []
    for r in records:
        d = {SOURCE_TO_CANONICAL.get(k, k): v for k, v in dict(r).items()}
        disc = d.get("total_discount")
        d["total_discount"] = (None if disc is None or not _finite(disc)
                               else float(disc) / 100.0)
        try:
            d["key"] = (ident(d["sku_id"]), ident(d["fc"]), iso_day(d["date"]),
                        int(float(d["hour_of_day"])))
        except (KeyError, TypeError, ValueError):
            d["key"] = None
        rows.append(d)
    return rows


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _num(v):
    return float(v) if _finite(v) else None


def split_hours(rows, hour=None):
    """(the batch hour as (date, hour), its opening rows, the closed rows
    of the hour before keyed by shelf). The batch hour is `hour`
    ("YYYY-MM-DDTHH") or the latest hour the snapshot holds."""
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


def rule_says_continues(last, row, closed_row):
    """What the chain's rule would say for this row: with the closed row
    (last hour's feed row) it is ops.assign_episode_ids.RULE exactly;
    without it, the store's latest decision stands in for last hour and
    stock arrived is read as this opening above its remaining stock."""
    if closed_row is not None:
        prev = {"date": closed_row["key"][2], "hour": closed_row["key"][3],
                "ending_inventory": closed_row.get("ending_inventory"),
                "inventory": closed_row.get("starting_inventory"),
                "units_sold": closed_row.get("units_sold"),
                "flc_window": closed_row.get("hours_remaining"), "episode_id": "x"}
        now = {"date": row["key"][2], "hour": row["key"][3],
               "flc_window": row.get("hours_remaining")}
        return rule_continues(prev, now)
    if last is None or last.get("hours_remaining") is None:
        return False
    if hours_between(last["date"], last["hour_of_day"], row["key"][2], row["key"][3]) != 1:
        return False
    if not _finite(row.get("hours_remaining")):
        return False
    step = float(planning_horizon(row["hours_remaining"])) - float(last["hours_remaining"])
    if step == -1:
        return True
    if step < -1:
        return False
    q_last = _num(last.get("q_remaining"))
    return q_last is not None and float(row["starting_inventory"]) > q_last


def build_requests(openings, closed, latest):
    """The engine's 12-field requests for the openings that can be priced,
    aligned with `openings` (None where a row is not sent), and the
    counts: shelves empty, unkeyable, without an episode id, episodes new
    and continued (by the producer's id), and the ids that disagree with
    the rule (with a sample)."""
    requests = []
    counts = {"shelves_empty": 0, "shelves_unkeyable": 0, "shelves_without_episode_id": 0,
              "episodes_new": 0, "episodes_continued": 0,
              "episode_ids_disagreeing_with_the_rule": 0, "disagreements_sample": []}
    for r in openings:
        if r["key"] is None:
            counts["shelves_unkeyable"] += 1
            requests.append(None)
            continue
        q = _num(r.get("starting_inventory"))
        if q is not None and q <= 0:
            counts["shelves_empty"] += 1
            requests.append(None)
            continue
        eid = r.get("episode_id")
        if eid is None or (isinstance(eid, float) and eid != eid) or str(eid).strip() == "":
            counts["shelves_without_episode_id"] += 1
            requests.append(None)
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
        rule = rule_says_continues(last, r, closed.get((sku, fc)))
        if last is not None and rule != continued:
            counts["episode_ids_disagreeing_with_the_rule"] += 1
            if len(counts["disagreements_sample"]) < 10:
                counts["disagreements_sample"].append({
                    "skuseq": sku, "fc": fc, "episode_id": eid,
                    "producer": "continued" if continued else "new",
                    "rule": "continued" if rule else "new"})
        requests.append({
            "episode_id": eid, "sku_id": sku, "fc": fc,
            "category": r.get("category"), "subcategory": r.get("subcategory"),
            "date": day, "hour_of_day": hour,
            "hours_remaining": (planning_horizon(r["hours_remaining"])
                                if _finite(r.get("hours_remaining")) else r.get("hours_remaining")),
            "q": r.get("starting_inventory"),
            "original_price": r.get("original_price"), "cost": r.get("cost"),
            "current_discount": anchor})
    return requests, counts


def run(cfg, snapshot_rows, hour=None, features=None, history=None, workers=None,
        seed=0, store=None, dry_run=False, model=None, posterior=None, r_lookup=None):
    """One hour: the requests from the snapshot, priced (ops.price_batch.run),
    the response rows in the feed's units. `dry_run` prices against a
    scratch copy of the store and commits nothing. Returns (response
    rows, events, report)."""
    if dry_run:
        scratch = tempfile.mkdtemp(prefix="price_hour_dry_")
        src = cfg["events"]["store_dir"]
        if os.path.isdir(src):
            shutil.rmtree(scratch)
            shutil.copytree(src, scratch)
        store = EventStore(cfg, root=scratch)
    store = store or EventStore(cfg)
    when, openings, closed = split_hours(snapshot_rows, hour)
    requests, counts = build_requests(openings, closed, store.latest_by_shelf)
    sent = [r for r in requests if r is not None]
    rows, events, rep = price_batch.run(cfg, sent, history, workers=workers, seed=seed,
                                        store=store, features=features, model=model,
                                        posterior=posterior, r_lookup=r_lookup)
    answered = iter(rows)
    response = []
    for r, req in zip(openings, requests):
        if req is None:
            why = ("row names no shelf-hour" if r["key"] is None
                   else "empty shelf: nothing to price" if (_num(r.get("starting_inventory")) or 0) <= 0
                   else "episode_id missing: the producer assigns it (ops.assign_episode_ids)")
            response.append({"skuseq": r.get("sku_id"), "fc": r.get("fc"),
                             "date": r.get("date"), "hour": r.get("hour_of_day"),
                             "episode_id": None, "decision_id": None,
                             "apply_discount_pct": None, "apply_price": None,
                             "is_exploration": None, "rejected": why})
            continue
        a = next(answered)
        disc = a["applied_discount"]
        response.append({
            "skuseq": req["sku_id"], "fc": req["fc"], "date": req["date"],
            "hour": req["hour_of_day"], "episode_id": req["episode_id"],
            "decision_id": a["decision_id"],
            "apply_discount_pct": None if disc is None else round(float(disc) * 100.0, 4),
            "apply_price": a["applied_price"], "is_exploration": a["is_exploration"],
            "rejected": a["rejected"]})
    report = {"hour": f"{when[0]}T{when[1]:02d}" if when else None,
              "shelves": len(openings), "closed_rows_seen": len(closed),
              **counts, "dry_run": bool(dry_run), "live_rule": LIVE_RULE, **rep}
    return response, events, report


def write_response(rows, path):
    """The response in the feed's units, by extension: .csv, .parquet or
    .jsonl (RESPONSE_COLS)."""
    if path.endswith(".csv"):
        pd.DataFrame(rows, columns=list(RESPONSE_COLS)).to_csv(path, index=False)
    elif path.endswith(".parquet"):
        pd.DataFrame(rows, columns=list(RESPONSE_COLS)).to_parquet(path, index=False)
    else:
        write_jsonl(path, rows, fields=RESPONSE_COLS)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ops.price_hour")
    ap.add_argument("--snapshot", required=True,
                    help="the shelf at the top of the hour, in the feed's schema "
                         "(parquet, CSV or JSONL); the closed hour's rows may ride along")
    ap.add_argument("--features", default=None, help="today's feature table")
    ap.add_argument("--history", default=None, help="instead of --features (see ops.price_batch)")
    ap.add_argument("--out", required=True, help="the response: .csv, .parquet or .jsonl")
    ap.add_argument("--report", default=None, help="the hour's counts, JSON")
    ap.add_argument("--hour", default=None, help="YYYY-MM-DDTHH; default the latest hour in the snapshot")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="price against a scratch copy of the store; nothing is committed")
    args = ap.parse_args(argv)
    if bool(args.features) == bool(args.history):
        ap.error("pass exactly one of --features (the day's table) or --history")
    cfg = load_config(args.config, strict=True)
    features = pd.read_parquet(args.features) if args.features else None
    history = load_history(args.history, cfg) if args.history else None
    response, events, report = run(cfg, read_snapshot(args.snapshot), hour=args.hour,
                                   features=features, history=history,
                                   workers=args.workers, seed=args.seed, dry_run=args.dry_run)
    write_response(response, args.out)
    report["response"] = args.out
    if args.report:
        write_json(args.report, report)
    print(f"{report['hour']}: {report['shelves']} shelves -> {report['decisions']} priced "
          f"({report['episodes_new']} new episodes, {report['episodes_continued']} continued), "
          f"{report['rejected']} rejected, {report['shelves_empty']} empty"
          + (f", {report['episode_ids_disagreeing_with_the_rule']} ids disagree with the rule"
             if report["episode_ids_disagreeing_with_the_rule"] else "")
          + (" -- DRY RUN, nothing committed" if args.dry_run else ""))
    for why, n in sorted(report["rejected_by_the_engine"].items()):
        print(f"  {n:,}  {why}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
