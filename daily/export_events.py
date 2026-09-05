"""daily.export_events -- dump the event log to tables for the warehouse.

A one-way, DERIVED export: the JSONL streams stay the audit record and the
only thing learning or assurance reads; these tables exist so engineering
can load decisions (and outcomes) into the warehouse with plain SQL. One
row per event; list fields (`mu_ref_path`) are JSON-encoded strings so the
tables load into any warehouse. Idempotent -- re-running overwrites with
the same content for the same store.

Run: python3 -m daily.export_events [--out-dir exports] [--since YYYY-MM-DD]
"""

import argparse
import json
import os

import pandas as pd

from common.config import load_config
from events.store import EventStore


def _frame(events, since=None, date_field="date"):
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    if since and date_field in df.columns:
        df = df[df[date_field].astype(str) >= str(since)]
    for col in df.columns:                    # warehouse-safe: no list cells
        if df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(json.dumps)
    return df.reset_index(drop=True)


def export(store, out_dir, since=None):
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for name, events, date_field in (
            ("decisions", store.load_decisions(), "date"),
            ("outcomes", store.load_outcomes(), "finalized_at")):
        df = _frame(events, since, date_field)
        path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        written[name] = (path, len(df))
    return written


def main():
    ap = argparse.ArgumentParser(prog="daily.export_events")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--events-dir", default=None)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--since", default=None,
                    help="keep events on/after this date (decisions by their "
                         "pricing date, outcomes by finalized_at)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = EventStore(cfg, root=args.events_dir)
    for name, (path, n) in export(store, args.out_dir, args.since).items():
        print(f"{name:9s}: {n:,} rows -> {path}")
    print("derived export -- the JSONL event streams remain the audit record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
