"""common.io -- the one way a JSON artifact or report is read and written."""

import json
import os

import numpy as np


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
