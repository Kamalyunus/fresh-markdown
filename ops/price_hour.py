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
when they ride along, else the last hour SEEN on that shelf -- but only to
COUNT the ids that disagree with it (`episode_ids_disagreeing_with_the_rule`,
with a sample), never to override the producer's id. Every refused row is
RECORDED (events.store rejections: seen, not priced), so a shelf held
through a rejection still gives the rule an hour to step from. From the
store alone only a counter RESET is decisive; the rest is counted as
`episode_ids_not_decidable_from_the_store` (it vanishes when the closed
rows ride along). A shelf with no hour to step from at all is
`episode_ids_the_rule_could_not_check` -- and since our own refusals are
in the record, that is an hour engineering did not send.

Run: python3 -m ops.price_hour --snapshot <top-of-hour rows> \\
        --features features/<today>.parquet --out <hour>.csv --report <hour>.json \\
        [--workers 0] [--hour YYYY-MM-DDTHH] [--dry-run]
"""

import argparse
import os
import shutil
import tempfile

import pandas as pd

from common.config import load_config
from common.io import read_rows, write_frame, write_json
from common.windows import hours_between, planning_horizon
from ops.assign_episode_ids import continues as rule_continues
from engine.state import load_history
from events.contract import rejection_event
from events.pairs import as_number as _num, hour_key, iso_day
from events.store import EventStore
from fit.prepare_data import SOURCE_TO_CANONICAL
from ops import price_batch

LIVE_RULE = ("The producer's episode_id is read as given. Checked against the chain's "
             "rule (ops.assign_episode_ids.RULE): a shelf continues last hour's "
             "episode when that hour was one earlier, did not close the shelf, and "
             "the counter stepped down by one or up/flat with stock arrived. With "
             "last hour's closed row the check is the producer's own script; from "
             "the store alone only a counter RESET is decisive (the store never sees "
             "a close or a restock), so the rest is counted as not decidable. A "
             "disagreement is counted and sampled, never overridden; a shelf with no "
             "hour to step from is counted as a gap, never as a disagreement.")

RESPONSE_COLS = ("skuseq", "fc", "date", "hour", "episode_id", "decision_id",
                 "apply_discount_pct", "apply_price", "is_exploration", "rejected")

def read_snapshot(path):
    """The snapshot file's rows (parquet, CSV or JSONL in the feed's
    names) through `snapshot_rows`."""
    return snapshot_rows(read_rows(path))


def snapshot_rows(records):
    """Snapshot records (dicts in the feed's names, or already renamed) in
    the canonical names (the discount a fraction, the day `YYYY-MM-DD`),
    each keyed by (sku, fc, date, hour) where the record names one; an
    unkeyable record is kept with `key` None so it costs itself a
    response line, never the batch."""
    rows = []
    for r in records:
        d = {SOURCE_TO_CANONICAL.get(k, k): v for k, v in dict(r).items()}
        disc = _num(d.get("total_discount"))
        d["total_discount"] = None if disc is None else disc / 100.0
        try:
            d["key"] = hour_key(d["sku_id"], d["fc"], d["date"], d["hour_of_day"])
        except (KeyError, TypeError, ValueError):
            d["key"] = None
        rows.append(d)
    return rows


def _finite(v):
    return _num(v) is not None


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


def rule_says_continues(seen, row, closed_row):
    """What the chain's rule says for this row, as (verdict, basis):
    verdict True/False/None, basis one of "closed_row", "reset", "gap",
    "indeterminate".

    With the closed row (last hour's feed row) it is
    ops.assign_episode_ids.RULE exactly -- the same inputs the producer's
    own script had, so the verdict is decisive ("closed_row").

    Without it the last hour SEEN on the shelf (a decision or a recorded
    rejection, EventStore.last_seen_by_shelf) stands in, and it can only
    say one thing for certain: the counter RESET (stepped down by two or
    more), which is a new window whatever else happened ("reset", False).
    A step of minus one is only consistent with continuing -- the store
    never sees a sell-out close; a flat or upward step needs the ending
    stock the store never holds. Those are "indeterminate": neither a
    disagreement nor a gap, and they vanish when the closed rows ride
    along. No seen hour exactly one hour back, or an unreadable counter on
    either side, is a "gap": nothing to step from -- and since our own
    refusals are recorded, a gap is an hour engineering did not send."""
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
        return None, "gap"                       # not last hour: nothing to step from
    step = float(planning_horizon(now_c)) - last_h
    if step < -1:
        return False, "reset"
    return None, "indeterminate"


def _producers_id_last_hour(closed_row, seen_entry):
    """The producer's own episode id for the hour the rule stepped from:
    the closed row's when it rides along and carries one; else the id we
    recorded for that same hour (a decision or a rejection carries the
    producer's id as it was sent), so a producer who omits ids on the
    closed rows still gets the check; else None (nothing to compare)."""
    if closed_row is not None:
        eid = closed_row.get("episode_id")
        if eid is not None and not (isinstance(eid, float) and eid != eid) and str(eid).strip():
            return eid
        if seen_entry is not None and (seen_entry["date"], seen_entry["hour_of_day"]) == \
                (closed_row["key"][2], closed_row["key"][3]):
            return seen_entry.get("episode_id")
        return None
    return seen_entry.get("episode_id") if seen_entry is not None else None


def build_requests(openings, closed, latest, seen):
    """The engine's 12-field requests for the openings that can be priced,
    and the refusals, both aligned with `openings` (a row is in exactly one
    of them), plus the counts: shelves empty, unkeyable, without an episode
    id, episodes new and continued (by the producer's id against the last
    DECISION -- what the anchor and the stored path come from), the ids
    that disagree with the rule (with a sample), the ids the rule could not
    check (a gap: no hour to step from) and the ids not decidable from the
    store alone (they vanish when the closed rows ride along). `latest` is
    EventStore.latest_by_shelf, `seen` its last_seen_by_shelf."""
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
        # the rule is checked against the record it STEPS FROM -- last
        # hour's closed row when it rides along, else the last hour seen on
        # the shelf, priced or refused -- and so is the producer's own
        # answer: did their id at this hour keep that record's id?
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
            # an unreadable counter goes as None: validate_request refuses
            # it with the reason, and the rejection record carries None
            # rather than a value the rule would choke on next hour
            "hours_remaining": (planning_horizon(r["hours_remaining"])
                                if _finite(r.get("hours_remaining")) else None),
            "q": r.get("starting_inventory"),
            "original_price": r.get("original_price"), "cost": r.get("cost"),
            "current_discount": anchor})
        refused.append(None)
    return requests, refused, counts


def run(cfg, snapshot_rows, hour=None, features=None, history=None, workers=None,
        seed=0, store=None, dry_run=False, model=None, posterior=None, r_lookup=None):
    """One hour: the requests from the snapshot, priced (ops.price_batch.run),
    the response rows in the feed's units. `dry_run` prices against a
    scratch copy of the store and commits nothing. Returns (response
    rows, events, report)."""
    if dry_run:
        # a scratch COPY of the store, removed when the run is over: nothing
        # a dry run writes can reach the real record, and nothing it copies
        # outlives it
        scratch = tempfile.mkdtemp(prefix="price_hour_dry_")
        try:
            src = cfg["events"]["store_dir"]
            if os.path.isdir(src):
                shutil.rmtree(scratch)
                shutil.copytree(src, scratch)
            return _run(cfg, snapshot_rows, hour, features, history, workers, seed,
                        EventStore(cfg, root=scratch), True, model, posterior, r_lookup)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    return _run(cfg, snapshot_rows, hour, features, history, workers, seed,
                store or EventStore(cfg), False, model, posterior, r_lookup)


def _run(cfg, snapshot_rows, hour, features, history, workers, seed, store, dry_run,
         model, posterior, r_lookup):
    when, openings, closed = split_hours(snapshot_rows, hour)
    requests, refused, counts = build_requests(openings, closed, store.latest_by_shelf,
                                               store.last_seen_by_shelf)
    sent = [r for r in requests if r is not None]
    rows, events, rep = price_batch.run(cfg, sent, history, workers=workers, seed=seed,
                                        store=store, features=features, model=model,
                                        posterior=posterior, r_lookup=r_lookup)
    answered = iter(rows)
    response = []
    for r, req, why in zip(openings, requests, refused):
        if req is None:
            # recorded as SEEN, not priced: the shelf reached us at this
            # hour, so next hour's rule steps from it instead of reading a
            # gap (a row naming no shelf-hour records nothing -- there is no
            # shelf to record it against)
            refused = rejection_event(
                {**r, "hours_remaining": (planning_horizon(r["hours_remaining"])
                                          if _finite(r.get("hours_remaining")) else None),
                 "q_remaining": _num(r.get("starting_inventory"))}, why)
            if refused is not None and store.emit_rejection(refused):
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
    """One response row (RESPONSE_COLS) in the feed's units: the shelf-hour
    echoed, the producer's id, and the price to apply as a percent and as
    a price -- or the reason there is none."""
    disc = answer["applied_discount"] if answer else None
    return {"skuseq": skuseq, "fc": fc, "date": date, "hour": hour,
            "episode_id": episode_id,
            "decision_id": answer["decision_id"] if answer else None,
            "apply_discount_pct": None if disc is None else round(float(disc) * 100.0, 4),
            "apply_price": answer["applied_price"] if answer else None,
            "is_exploration": answer["is_exploration"] if answer else None,
            "rejected": rejected}


def write_response(rows, path):
    """The response in the feed's units, by extension: .csv, .parquet or
    .jsonl (RESPONSE_COLS, common.io.write_frame)."""
    write_frame(pd.DataFrame(rows, columns=list(RESPONSE_COLS)), path)


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
          + (" -- EXPLOIT ONLY (exploration.mode)" if report.get("exploration_mode") == "exploit" else "")
          + (" -- DRY RUN, nothing committed" if args.dry_run else ""))
    for why, n in sorted(report["rejected_by_the_engine"].items()):
        print(f"  {n:,}  {why}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
