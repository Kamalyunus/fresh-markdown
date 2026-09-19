"""Files: the row readers for parquet, CSV and JSONL, the top-of-hour
snapshot (the twelve request fields, as sent), the response, JSON out."""
import json
import os

import numpy as np
import pandas as pd

from pricing.keys import hour_key

RESPONSE_COLS = ("sku_id", "fc", "date", "hour_of_day", "episode_id", "decision_id",
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


def read_snapshot(path):
    """The snapshot rows (the twelve request fields, handover Appendix C),
    each keyed by its shelf-hour; an unkeyable row keeps `key` None so it
    costs itself a response line, never the batch."""
    rows = []
    for r in read_rows(path):
        d = dict(r)
        try:
            d["key"] = hour_key(d["sku_id"], d["fc"], d["date"], d["hour_of_day"])
        except (KeyError, TypeError, ValueError):
            d["key"] = None
        rows.append(d)
    return rows


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
