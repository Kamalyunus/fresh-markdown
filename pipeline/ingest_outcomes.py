"""pipeline.ingest_outcomes -- construct outcome events from the hourly feed.

The minimal integration: engineering calls the price API and applies the
price; outcomes are built HERE from the same hourly FLC feed the bootstrap
ingests, matched to decisions by (sku_id, fc, date, hour_of_day). Everything
the old producer contract asked for is derived:

  adjustment_reason   common.episodes.adjustment_reason on (start, sold, end)
  is_stockout         sold >= starting
  applied_price       original_price x (1 - discount) -- the OFFERED price,
                      never the realised-price column (zeroed on no-sale rows)
  outcome_id          "feed-<decision_id>" (idempotent re-runs dedup)
  finalized_at        the hour's close, UTC

The one fact only engineering knows -- did the price push succeed -- arrives
as an optional failures file (JSONL, one row per failed push:
{"sku_id", "fc", "date", "hour_of_day", "reason"}). Absent = ok.

Run: python3 -m pipeline.ingest_outcomes --feed <hourly parquet> [--failures f.jsonl]
"""

import argparse
import json

import pandas as pd

from common.config import load_config
from common.episodes import adjustment_reason
from bootstrap.prepare_data import SOURCE_TO_CANONICAL
from events.store import EventStore


def _key(sku, fc, date, hour):
    return (str(sku), str(fc), str(date), int(hour))


def load_failures(path):
    """{key: reason} from the failures JSONL, or {} when no file is given."""
    if not path:
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[_key(r["sku_id"], r["fc"], r["date"], r["hour_of_day"])] = \
                r.get("reason", "unspecified")
    return out


def build_outcomes(decisions, feed, failures=None, now=None):
    """One outcome per decision, from the feed row for its hour.

    Returns (outcomes, report). A decision with no feed row yields no outcome
    -- that is the completeness gap the shadow gate measures, so it is
    counted, never invented.
    """
    failures = failures or {}
    feed = feed.rename(columns=SOURCE_TO_CANONICAL)
    rows = {}
    dup_feed = 0
    for r in feed.itertuples():
        k = _key(r.sku_id, r.fc, r.date, r.hour_of_day)
        if k in rows:
            dup_feed += 1        # two states for one hour: match neither
            rows[k] = None
        else:
            rows[k] = r

    outcomes, unmatched, reasons = [], [], {}
    for dec in decisions:
        k = _key(dec["sku_id"], dec["fc"], dec["date"], dec["hour_of_day"])
        r = rows.get(k)
        if r is None:
            unmatched.append(dec["decision_id"])
            continue
        start = int(round(r.starting_inventory))
        sold = int(r.units_sold)
        end = int(round(r.ending_inventory))
        # OFFERED price; feed discount is PERCENT (prepare_data converts the
        # same column the same way)
        offered = float(r.original_price) * (1 - float(r.total_discount) / 100.0)
        out = {
            "event": "outcome",
            "outcome_id": f"feed-{dec['decision_id']}",
            "decision_id": dec["decision_id"],
            "units_sold": sold,
            "starting_inventory": start,
            "ending_inventory": end,
            "applied_price": offered,
            "is_stockout": sold >= start,
            "execution_status": "ok",
            "finalized_at": (pd.Timestamp(f"{r.date} {int(r.hour_of_day)}:00",
                                          tz="UTC")
                             + pd.Timedelta(hours=1)).isoformat(),
        }
        why = adjustment_reason(start, sold, end)
        if why:
            out["adjustment_reason"] = why
            reasons[why] = reasons.get(why, 0) + 1
        if k in failures:
            out["execution_status"] = "failed"
            out["execution_failure_reason"] = failures[k]
        outcomes.append(out)

    report = {
        "decisions": len(decisions),
        "outcomes_built": len(outcomes),
        "decisions_without_feed_row": len(unmatched),
        "unmatched_decision_ids": unmatched[:20],
        "feed_duplicate_hours": dup_feed,
        "adjustment_reasons": reasons,
        "push_failures_applied": sum(1 for o in outcomes
                                     if o["execution_status"] == "failed"),
    }
    return outcomes, report


def main():
    ap = argparse.ArgumentParser(prog="pipeline.ingest_outcomes")
    ap.add_argument("--feed", required=True,
                    help="hourly FLC parquet (raw source schema)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--failures", default=None,
                    help="JSONL of failed price pushes: sku_id, fc, date, "
                         "hour_of_day, reason")
    ap.add_argument("--events-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = EventStore(cfg, root=args.events_dir)
    decisions = store.load_decisions()
    if not decisions:
        raise SystemExit("no decisions in the event store -- nothing to match")
    outcomes, report = build_outcomes(decisions, pd.read_parquet(args.feed),
                                      load_failures(args.failures))
    emitted = sum(store.emit_outcome(o) for o in outcomes)
    report["emitted"] = int(emitted)
    report["duplicates_skipped"] = len(outcomes) - int(emitted) \
        - store.quarantined_this_run
    report["quarantined"] = store.quarantined_this_run

    print(f"decisions          : {report['decisions']:,}")
    print(f"outcomes built     : {report['outcomes_built']:,} "
          f"(emitted {report['emitted']:,}, "
          f"duplicates {report['duplicates_skipped']:,}, "
          f"quarantined {report['quarantined']:,})")
    print(f"no feed row        : {report['decisions_without_feed_row']:,}"
          + (" -- this is the completeness gap the gate measures"
             if report["decisions_without_feed_row"] else ""))
    if report["feed_duplicate_hours"]:
        print(f"feed duplicate hrs : {report['feed_duplicate_hours']:,} "
              "(matched to no decision -- two states for one hour)")
    for why, n in sorted(report["adjustment_reasons"].items()):
        print(f"  {why:28s} {n:,}")
    if report["push_failures_applied"]:
        print(f"push failures      : {report['push_failures_applied']:,} "
              "marked ineligible for learning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
