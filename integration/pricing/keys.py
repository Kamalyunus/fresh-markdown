"""The one spelling of every key: an id, a day, an hour, the shelf-hour tag
and the two event ids over it, the clock step and the planning horizon."""
import math

import numpy as np
import pandas as pd


def iso_day(value):
    """A trading day as `YYYY-MM-DD`, whatever the producer's dtype; raises
    on a value that names no day."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"not a day: {value!r}")
    return stamp.strftime("%Y-%m-%d")


def ident(v):
    """An identifier as text: a float column's 7.0 and a JSONL "7" both key
    the same item; NaN, None, a bool or a fraction names nothing (raises)."""
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v) or v != int(v):
            raise ValueError(f"not an identifier: {v!r}")
        return str(int(v))
    if v is None or isinstance(v, (bool, np.bool_)):
        raise ValueError(f"not an identifier: {v!r}")
    return str(v)


def ident_series(s):
    """`ident` over a column: NaN reads as None, a fraction raises."""
    s = pd.Series(s)
    if pd.api.types.is_bool_dtype(s):
        raise ValueError("not an identifier column: bool")
    if pd.api.types.is_numeric_dtype(s):
        x = s.astype(float)
        finite = np.isfinite(x)
        if not (x[finite] == np.floor(x[finite])).all():
            raise ValueError("not an identifier column: fractional values")
        out = pd.Series(None, index=s.index, dtype=object)
        out[finite] = x[finite].astype("int64").astype(str)
        return out
    out = pd.Series(None, index=s.index, dtype=object)
    present = s.notna()
    out[present] = s[present].astype(str)
    return out


def hour_int(hour):
    """An hour of the day as an int; a fraction or a non-finite value raises."""
    h = float(hour)
    if not np.isfinite(h) or h != int(h):
        raise ValueError(f"not an hour: {hour!r}")
    return int(h)


def hour_key(sku, fc, date, hour):
    """(sku, fc, day, hour): the key a feed row, a decision and a request meet on."""
    return (ident(sku), ident(fc), iso_day(date), hour_int(hour))


def shelf_hour_tag(key):
    sku, fc, day, hour = key
    return f"{sku}|{fc}|{day}T{hour:02d}"


def decision_id_of(key):
    return "dec-" + shelf_hour_tag(key)


def rejection_id_of(key):
    return "rej-" + shelf_hour_tag(key)


def as_number(v):
    """`v` as a finite float, or None (the lenient reading of a feed cell)."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def hours_between(day_a, hour_a, day_b, hour_b):
    """Whole hours from (day_a, hour_a) to (day_b, hour_b); negative when b is earlier."""
    a = pd.Timestamp(day_a) + pd.Timedelta(hours=int(hour_a))
    b = pd.Timestamp(day_b) + pd.Timedelta(hours=int(hour_b))
    return int(round((b - a).total_seconds() / 3600.0))


def planning_horizon(counter):
    """The counter is the hours still to come after this one: the horizon
    is this hour plus the counter."""
    return int(counter) + 1
