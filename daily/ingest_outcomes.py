"""daily.ingest_outcomes -- construct outcome events from the hourly feed.

Engineering applies the price; outcomes are built HERE from the same hourly
FLC feed the bootstrap ingests, matched to decisions by (sku_id, fc, date,
hour_of_day). Every outcome field is derived from the feed row:
`adjustment_reason` and `is_stockout` through the common.episodes rules,
`applied_price` as the OFFERED price (original_price x (1 - discount)),
`outcome_id` as the hour's key -- "feed-<sku>|<fc>|<date>T<hh>"
(events.pairs.outcome_id_of: engineering can name it from the feed row;
re-runs dedup), `finalized_at` as the hour's close in UTC. The one fact
only engineering knows -- did the price push succeed -- arrives as an
optional failures input (parquet/CSV/JSONL, one row per failed push:
sku_id, fc, date, hour_of_day, reason). Two decisions for one hour match
neither (two prices claimed one feed row: a retried batch), counted as
`decisions_colliding_on_hour`. Runs as a DAILY batch over the previous
day's rows.

Run: python3 -m daily.ingest_outcomes --feed <hourly parquet> [--failures f.jsonl]
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from common.config import load_config
from common.episodes import adjustment_reason, is_censored_hour
from common.io import read_rows
from fit.prepare_data import SOURCE_TO_CANONICAL
from events.pairs import colliding_keys, decision_day, hour_key, outcome_id_of
from events.store import EventStore


def load_failures(path):
    """{key: reason} from the failures input -- a parquet/CSV table or
    JSONL (common.io.read_rows), in the contract's names or the feed's
    (skuseq, hour) -- or {} when no file is given. One unkeyable row (a
    NaN id, a blank line, a line that is not an object) costs that row,
    never the batch: it is counted in `push_failures_unkeyable` on the
    returned dict's `.unkeyable`."""
    out = _Failures()
    if not path:
        return out
    for r in read_rows(path, rename=SOURCE_TO_CANONICAL):
        try:
            k = hour_key(r["sku_id"], r["fc"], r["date"], r["hour_of_day"])
        except (KeyError, TypeError, ValueError):
            out.unkeyable += 1
            continue
        out[k] = r.get("reason") or "unspecified"
    return out


class _Failures(dict):
    """{key: reason} plus the rows that named no hour."""
    unkeyable = 0


def build_outcomes(decisions, feed, failures=None):
    """One outcome per decision, from the feed row for its hour.

    Returns (outcomes, report). A decision with no feed row yields no outcome
    -- that is the completeness gap the shadow gate measures, so it is
    counted, never invented. Only decisions whose trading day lies inside
    THIS feed's date range can be missing from it: a decision already
    ingested yesterday, or one not yet due, is outside the feed and is
    reported separately, never as a gap.
    """
    failures = failures or {}
    feed = feed.rename(columns=SOURCE_TO_CANONICAL)
    if len(feed):
        # one spelling of the day, whatever dtype the feed carries
        # one unparseable date is one unkeyable row (NaT -> NaN -> counted
        # below), never a batch-wide raise
        feed = feed.assign(date=pd.to_datetime(feed["date"], errors="coerce")
                           .dt.strftime("%Y-%m-%d"))
    rows, unusable = {}, []
    keyed_rows = []
    for i, r in enumerate(feed.itertuples()):
        # a row that names no hour or no item (NaN hour_of_day) is one
        # unusable row, never a fatal int(nan) before any decision is matched
        try:
            k = hour_key(r.sku_id, r.fc, r.date, r.hour_of_day)
        except (TypeError, ValueError) as exc:
            unusable.append({"decision_id": None, "feed_row": i,
                             "reason": f"unkeyable feed row: {type(exc).__name__}: {exc}"})
            continue
        keyed_rows.append((k, r))
    # two states for one hour: match neither (the one rule, colliding_keys)
    dup_hours = colliding_keys(k for k, _ in keyed_rows)
    for k, r in keyed_rows:
        rows[k] = None if k in dup_hours else r
    # the keyable days only: a NaN date is an unkeyable row, counted above
    feed_days = sorted(d for d in set(feed["date"]) if isinstance(d, str)) \
        if len(feed) else []
    feed_range = (feed_days[0], feed_days[-1]) if feed_days else None

    outcomes, unmatched, reasons = [], [], Counter()
    outside, failed_keys = 0, set()
    # key every decision the feed could answer for, once: two decisions on
    # one hour (a retried price batch) match neither -- neither is the
    # price the shelf held, and pairing both would count one hour twice
    keyed = []
    for dec in decisions:
        day = decision_day(dec)
        if feed_range is None:
            # an EMPTY feed is a whole day of gaps, not a day outside it:
            # every decision is missing its outcome (contract 07)
            unmatched.append(dec["decision_id"])
            continue
        if not (feed_range[0] <= day <= feed_range[1]):
            outside += 1         # not this feed's business: no gap, no match
            continue
        # one unusable row costs its own decision, never the day's batch: it
        # is counted below (a foreign line missing a key field included),
        # and a zero/absent base price is refused rather than priced as a
        # full-list discount
        try:
            k = hour_key(dec.get("sku_id"), dec.get("fc"), day, dec.get("hour_of_day"))
        except (TypeError, ValueError) as exc:
            unusable.append({"decision_id": dec["decision_id"],
                             "reason": f"unkeyable decision: {type(exc).__name__}: {exc}"})
            continue
        keyed.append((dec, k))
    claimed_twice = colliding_keys(k for _, k in keyed)
    colliding = []
    for dec, k in keyed:
        if k in claimed_twice:
            colliding.append(dec["decision_id"])
            continue
        r = rows.get(k)
        if r is None:
            unmatched.append(dec["decision_id"])
            continue
        try:
            start = int(round(float(r.starting_inventory)))
            sold = int(round(float(r.units_sold)))     # rounded, like the stock
            end = int(round(float(r.ending_inventory)))
            base = float(r.original_price)
            disc = float(r.total_discount)
            if not (np.isfinite(base) and np.isfinite(disc)) or base <= 0:
                raise ValueError(f"unusable price (original_price={base!r})")
            # the feed's discount is PERCENT: 0.30 here would be 0.3%, and
            # 100+ prices the hour at or below zero
            if not 0 <= disc < 100:
                raise ValueError(f"discount out of [0, 100) percent "
                                 f"(total_discount={disc!r})")
        except (TypeError, ValueError) as exc:
            unusable.append({"decision_id": dec["decision_id"],
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        # OFFERED price; feed discount is PERCENT (prepare_data converts the
        # same column the same way)
        offered = base * (1 - disc / 100.0)
        out = {
            "event": "outcome",
            "outcome_id": outcome_id_of(k),
            "decision_id": dec["decision_id"],
            "units_sold": sold,
            "starting_inventory": start,
            "ending_inventory": end,
            "applied_price": offered,
            "is_stockout": bool(is_censored_hour(start, sold, end)),
            "execution_status": "ok",
            "finalized_at": (pd.Timestamp(f"{r.date} {int(r.hour_of_day)}:00",
                                          tz="UTC")
                             + pd.Timedelta(hours=1)).isoformat(),
        }
        why = adjustment_reason(start, sold, end)
        if why:
            out["adjustment_reason"] = why
            reasons[why] += 1
        if k in failures:
            out["execution_status"] = "failed"
            out["execution_failure_reason"] = failures[k]
            failed_keys.add(k)
        outcomes.append(out)

    report = {
        "decisions": len(decisions),
        "feed_date_range": list(feed_range) if feed_range else None,
        # decisions whose trading day lies OUTSIDE this feed: already
        # ingested, or not yet due -- neither is a completeness gap
        "decisions_outside_feed_range": outside,
        "outcomes_built": len(outcomes),
        "decisions_without_feed_row": len(unmatched),
        "unmatched_decision_ids": unmatched[:20],
        # decisions that claimed one hour between them: none is matched,
        # and completeness falls by all of them -- a retried price batch
        # is an integration miss, not silence
        "decisions_colliding_on_hour": len(colliding),
        "colliding_hours": len(claimed_twice),
        "colliding_decision_ids": colliding[:20],
        "feed_duplicate_hours": len(dup_hours),
        # counted and named, never silently dropped: one unusable row costs
        # its own decision, not the day
        "unusable_feed_rows": len(unusable),
        "unusable_examples": unusable[:20],
        "adjustment_reasons": dict(reasons),
        "push_failures_applied": len(failed_keys),
        # a reported failure naming no outcome built here: a key spelt
        # differently, an hour with no decision or no feed row. Counted --
        # a failure that lands nowhere is an integration miss, not silence
        "push_failures_unmatched": len(set(failures) - failed_keys),
        # failures rows that named no hour (a NaN id, a blank line)
        "push_failures_unkeyable": int(getattr(failures, "unkeyable", 0)),
        "feed_empty": feed_range is None,
    }
    return outcomes, report


def emit_all(store, outcomes, report):
    """Emit the built outcomes through the store and record what it
    refused: duplicates, a second outcome for a decision the store already
    holds one for (`outcomes_per_decision_over_one`), the quarantined, and
    the contract's `missing_stockout_field` count (an outcome without the
    field never lands -- the store refuses it; every outcome built here
    carries it, so the count reads what a foreign line or an older stream
    lacked)."""
    before = dict(store.completeness_counts)
    emitted = sum(store.emit_outcome(o) for o in outcomes)
    after = store.completeness_counts
    report["emitted"] = int(emitted)
    report["outcomes_per_decision_over_one"] = (
        after["outcomes_per_decision_over_one"] - before["outcomes_per_decision_over_one"])
    report["quarantined"] = store.quarantined_this_run
    report["duplicates_skipped"] = (len(outcomes) - int(emitted) - report["quarantined"]
                                    - report["outcomes_per_decision_over_one"])
    # all-time, the store's: what the contract's "exactly 0" gate reads
    report["missing_stockout_field"] = after["missing_stockout_field"]
    return report


def main():
    ap = argparse.ArgumentParser(prog="daily.ingest_outcomes")
    ap.add_argument("--feed", required=True,
                    help="hourly FLC parquet (raw source schema)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--failures", default=None,
                    help="failed price pushes -- parquet/CSV table or JSONL: "
                         "sku_id, fc, date, hour_of_day, reason")
    ap.add_argument("--events-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = EventStore(cfg, root=args.events_dir)
    decisions = store.load_decisions()
    if not decisions:
        raise SystemExit("no decisions in the event store -- nothing to match")
    outcomes, report = build_outcomes(decisions, pd.read_parquet(args.feed),
                                      load_failures(args.failures))
    emit_all(store, outcomes, report)

    print(f"decisions          : {report['decisions']:,} "
          f"({report['decisions_outside_feed_range']:,} outside the feed's "
          f"date range {report['feed_date_range']})")
    print(f"outcomes built     : {report['outcomes_built']:,} "
          f"(emitted {report['emitted']:,}, "
          f"duplicates {report['duplicates_skipped']:,}, "
          f"quarantined {report['quarantined']:,})")
    if report["outcomes_per_decision_over_one"]:
        print(f"second outcomes    : {report['outcomes_per_decision_over_one']:,} "
              "refused -- the store already holds an outcome for the decision "
              "(an id scheme that moved? the first one stands)")
    if report["missing_stockout_field"]:
        print(f"missing is_stockout: {report['missing_stockout_field']:,} outcome "
              "line(s) in the store lack the field (contract 07: exactly 0) -- "
              "refused on emit, skipped on load")
    if report["unusable_feed_rows"]:
        print(f"unusable feed rows : {report['unusable_feed_rows']:,} "
              "(non-numeric inventory, a zero/absent base price, or a row "
              "naming no hour or item -- counted into the completeness gap, "
              "batch NOT aborted)")
        for row in report["unusable_examples"][:5]:
            who = row["decision_id"] or f"feed row {row.get('feed_row')}"
            print(f"  {who}: {row['reason']}")
    print(f"no feed row        : {report['decisions_without_feed_row']:,}"
          + (" -- this is the completeness gap the gate measures"
             if report["decisions_without_feed_row"] else ""))
    if report["feed_duplicate_hours"]:
        print(f"feed duplicate hrs : {report['feed_duplicate_hours']:,} "
              "(matched to no decision -- two states for one hour)")
    if report["decisions_colliding_on_hour"]:
        print(f"colliding decisions: {report['decisions_colliding_on_hour']:,} "
              f"on {report['colliding_hours']:,} hour(s) -- two prices claimed "
              "one feed row (a retried batch?); none matched")
    for why, n in sorted(report["adjustment_reasons"].items()):
        print(f"  {why:28s} {n:,}")
    if report["push_failures_applied"]:
        print(f"push failures      : {report['push_failures_applied']:,} "
              "marked ineligible for learning")
    if report["push_failures_unmatched"]:
        print(f"push failures      : {report['push_failures_unmatched']:,} "
              "reported for hours that built no outcome (unmatched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
