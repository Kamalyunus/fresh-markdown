"""The feed's schema and files: the source names against the engine's, the
row readers for parquet, CSV and JSONL, the top-of-hour snapshot in either
of its two spellings, the response."""
import json
import os

import numpy as np
import pandas as pd

from pricing.keys import as_number, hour_key

SOURCE_TO_CANONICAL = {
    "hour": "hour_of_day",
    "skuseq": "sku_id",
    "inventory": "starting_inventory",
    "discount": "total_discount",
    "normal_asp": "original_price",
    "final_price": "applied_price",
    "cogs_wo_vat": "cost",
    "flc_window": "hours_remaining",
}

RESPONSE_COLS = ("skuseq", "fc", "date", "hour", "episode_id", "decision_id",
                 "apply_discount_pct", "apply_price", "is_exploration", "rejected")


def read_rows(path):
    """Rows as dicts from JSONL (a line that is not an object is `{}`),
    parquet or CSV (a null cell reads as None)."""
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


def write_frame(df, path):
    """By extension: .parquet, .csv, else JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_json(path, orient="records", lines=True)


# the request's spelling (handover Appendix C): the twelve fields as the
# engine reads them -- the counter this hour INCLUDED, the discount a
# FRACTION, null on an entry -- against the engine's internal names
REQUEST_TO_CANONICAL = {"q": "starting_inventory", "current_discount": "total_discount"}
FEED_MARKERS = {"skuseq", "inventory", "flc_window", "normal_asp", "cogs_wo_vat"}
REQUEST_MARKERS = {"sku_id", "q", "current_discount", "original_price", "hour_of_day"}


def snapshot_schema(records):
    """Which spelling a snapshot is in: "feed" (the hourly table's columns) or
    "request" (Appendix C's twelve fields). One file, one spelling: a file
    carrying both is refused, never guessed row by row."""
    cols = set()
    for r in records:
        cols |= set(r)
    feed, request = bool(cols & FEED_MARKERS), bool(cols & REQUEST_MARKERS)
    if feed and request:
        raise SystemExit("the snapshot mixes the hourly table's columns "
                         f"({sorted(cols & FEED_MARKERS)}) with the request's "
                         f"({sorted(cols & REQUEST_MARKERS)}): send one spelling per file")
    return "request" if request else "feed"


def snapshot_rows(records, schema=None):
    """Snapshot records in the engine's names, whichever spelling they came
    in -- the discount a fraction, the counter the hours still to come after
    this one -- each keyed by its shelf-hour; an unkeyable record keeps `key`
    None so it costs itself a response line, never the batch."""
    schema = schema or snapshot_schema(records)
    rows = []
    for r in records:
        if schema == "request":
            d = {REQUEST_TO_CANONICAL.get(k, k): v for k, v in dict(r).items()}
            d["total_discount"] = as_number(d.get("total_discount"))       # a fraction already
            hours = as_number(d.get("hours_remaining"))                     # this hour included
            d["hours_remaining"] = None if hours is None else hours - 1
        else:
            d = {SOURCE_TO_CANONICAL.get(k, k): v for k, v in dict(r).items()}
            disc = as_number(d.get("total_discount"))
            d["total_discount"] = None if disc is None else disc / 100.0    # a percent
        try:
            d["key"] = hour_key(d["sku_id"], d["fc"], d["date"], d["hour_of_day"])
        except (KeyError, TypeError, ValueError):
            d["key"] = None
        rows.append(d)
    return rows


def read_snapshot(path):
    """(rows in the engine's names, the spelling the file came in)."""
    records = read_rows(path)
    schema = snapshot_schema(records)
    return snapshot_rows(records, schema), schema


def json_safe(value):
    """NaN and numpy scalars are not JSON: null and native values."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_safe(value.item())
    return value


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2, default=str)
