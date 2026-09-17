"""daily.export_events -- dump the event log to tables for the warehouse.

A one-way, DERIVED export: the JSONL streams stay the audit record and the
only thing learning or assurance reads; these tables exist so engineering
can load decisions (and outcomes) into the warehouse with plain SQL. One
row per event; list fields (`mu_ref_path`) are JSON-encoded strings so the
tables load into any warehouse. Idempotent -- re-running overwrites with
the same content for the same store.

Every table that names a shelf-hour also carries it in THE FEED'S OWN
spelling and types (FEED_COLUMNS: `skuseq`, `fc`, `date`, `hour`), so
appending the night's decisions to the hourly FLC feed is a join on four
columns with no rename and no cast. The event's own `sku_id` (the chain's
one text spelling) and `hour_of_day` stay beside them untouched -- the
export is a faithful dump first and a convenience second. `date` is the
one column the two share a name for, and it is written as a real date
rather than the event's text, which loses nothing: the contract validates
an ISO day on the way in. A SKU id that is not an integer cannot be spelt
the feed's way, so `skuseq` is null on that row and the count is reported
rather than the column quietly changing type between days.

Run: python3 -m daily.export_events [--out-dir exports] [--since YYYY-MM-DD]
"""

import json
import os

import pandas as pd

from common.cli import make_parser
from common.config import load_config
from common.paths import EXPORTS
from events.pairs import decision_day
from events.store import EventStore


def _frame(events):
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    for col in df.columns:                    # warehouse-safe: no list cells
        if df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(json.dumps)
    return df.reset_index(drop=True)


# the feed's own names for the shelf-hour, added beside the event's own
# fields on every table that names one (the outcome names its decision, not
# a shelf, and gets none). `fc` and `date` are already the feed's names;
# only their TYPES move -- the day becomes a real date.
FEED_COLUMNS = ("skuseq", "fc", "date", "hour")


def add_feed_columns(df):
    """`df` with the shelf-hour in the feed's spelling and types, so the
    table joins to the hourly FLC feed without a rename or a cast. Returns
    (frame, ids that could not be spelt as the feed's integer skuseq).
    The event's own `sku_id` and `hour_of_day` are untouched; `date`,
    whose name the two already share, becomes the feed's date type."""
    if df.empty or not {"sku_id", "hour_of_day", "date"} <= set(df.columns):
        return df, 0
    out = df.copy()
    # the chain normalises every id to one text spelling (events.pairs.ident)
    # because a producer may send any dtype; the feed's own column is an
    # integer, so it is recovered where the id IS one and null where it is
    # not -- a column whose type changed between days would be worse than a
    # null a load can see
    skuseq = pd.to_numeric(out.sku_id, errors="coerce")
    unspellable = int((skuseq.isna() | (skuseq % 1 != 0)).sum())
    out["skuseq"] = skuseq.where(skuseq % 1 == 0).astype("Int64")
    out["hour"] = pd.to_numeric(out.hour_of_day, errors="coerce").astype("Int64")
    # the feed's `date` is a date, not text: cut the day out of the event's
    # ISO string (the contract validates it on the way in)
    out["date"] = pd.to_datetime(out.date, errors="coerce").dt.date
    return out, unspellable


def since_filter(decisions, outcomes, since):
    """Both streams cut on the ONE day key: the trading day the decision
    priced (events.pairs.decision_day). An outcome inherits its decision's
    day -- its own `finalized_at` is a UTC clock that rolls to D+1 for hour
    23, so cutting on it split a trading day across two exports. An outcome
    naming no known decision has no trading day and falls back to the date
    of its `finalized_at`, the only day it carries. An orphan carrying
    neither has no day to cut on: skipped and counted, never a KeyError
    that stops the export. Returns (decisions, outcomes, undated_skipped)."""
    since = str(since)
    day_of = {d["decision_id"]: decision_day(d) for d in decisions}

    def outcome_day(o):
        return (day_of.get(o.get("decision_id"))
                or str(o.get("finalized_at") or "")[:10] or None)

    days = [(o, outcome_day(o)) for o in outcomes]
    return ([d for d in decisions if day_of[d["decision_id"]] >= since],
            [o for o, day in days if day is not None and day >= since],
            sum(1 for _, day in days if day is None))


def export(store, out_dir, since=None):
    """Returns ({table: (path, rows)}, undated outcomes skipped by --since,
    ids that could not be spelt as the feed's integer skuseq)."""
    os.makedirs(out_dir, exist_ok=True)
    decisions, outcomes = store.load_decisions(), store.load_outcomes()
    undated = 0
    if since:
        decisions, outcomes, undated = since_filter(decisions, outcomes, since)
    # the shelf-hours seen and NOT priced, with the reason the response
    # carried: cut on their own date, which they carry directly
    rejections = [r for r in store.load_rejections()
                  if not since or str(r.get("date", "")) >= since]
    written, unspellable = {}, 0
    for name, events in (("decisions", decisions), ("outcomes", outcomes),
                         ("rejections", rejections)):
        df, bad = add_feed_columns(_frame(events))
        unspellable += bad
        path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        written[name] = (path, len(df))
    return written, undated, unspellable


def main():
    ap = make_parser(prog="daily.export_events")
    ap.add_argument("--events-dir", default=None)
    ap.add_argument("--out-dir", default=EXPORTS)
    ap.add_argument("--since", default=None,
                    help="keep events whose TRADING day is on/after this date "
                         "(decisions by their pricing date, outcomes by their "
                         "decision's), so both tables hold whole trading days")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = EventStore(cfg, root=args.events_dir)
    written, undated, unspellable = export(store, args.out_dir, args.since)
    for name, (path, n) in written.items():
        print(f"{name:9s}: {n:,} rows -> {path}")
    if undated:
        print(f"skipped  : {undated:,} outcome(s) naming no known decision and "
              "carrying no finalized_at -- no trading day to cut on")
    if unspellable:
        print(f"skuseq   : {unspellable:,} row(s) whose SKU id is not an integer -- "
              "null in the feed-spelled column; join those on sku_id")
    print(f"the shelf-hour is also in the feed's own spelling ({', '.join(FEED_COLUMNS)}): "
          "join to the hourly feed with no rename and no cast")
    print("derived export -- the JSONL event streams remain the audit record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
