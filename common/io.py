"""common.io -- the one way a JSON artifact, report or row file is read and written."""

import json
import os

import numpy as np
import pandas as pd


def read_json(path, default=None):
    """`default` when `path` is falsy or absent; otherwise the parsed file."""
    if not path or not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def json_safe(value):
    """NaN/Inf are not JSON: json.dump emits bare `NaN`, which most parsers
    reject. cogs_at_risk returns NaN for a whole stage when any episode has
    a null cost, and `cost_missing` is a FLAG, so the NaN survives."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    # np.bool_ is not a bool subclass: left to `default=str` it became the
    # STRING "False", which reads back as truthy
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_safe(value.item())
    return value


def write_json(path, payload, **dump_kw):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(payload), f,
                  **{"indent": 2, "default": str, **dump_kw})


def write_jsonl(path, rows, fields=None):
    """One JSON object per line, through `json_safe` (a NaN cell from a
    table reads as null, a numpy scalar as its value, a timestamp as its
    text) -- the ONE row writer. `fields`, when given, is the exact key
    set and order every line carries (a missing key writes null)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            out = {k: row.get(k) for k in fields} if fields else row
            f.write(json.dumps(json_safe(out), default=str) + "\n")


def read_rows(path, rename=None):
    """Rows as a list of dicts from JSONL (one object per line; a line that
    is not one is a row with NO fields, `{}`, so the caller refuses or
    counts that row and never raises for the batch), parquet or CSV (one
    row each; a null cell reads as None). `rename` maps column names on
    the way in (the feed's spellings to the contract's) -- the ONE
    three-way loader, shared by the price requests and the failures
    table."""
    rename = rename or {}
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
                rows.append({rename.get(k, k): v for k, v in obj.items()}
                            if isinstance(obj, dict) else {})
        return rows
    frame = frame.rename(columns=rename)
    frame = frame.astype(object).where(frame.notna(), None)
    return frame.to_dict("records")
