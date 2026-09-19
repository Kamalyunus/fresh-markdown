"""The ONE pairing of outcomes to decisions, the ONE day key, and the ONE
event-quality count.

Four modules each rebuilt `{decision_id: d}` and walked the outcomes with
their own eligibility rule; the learning path excluded failed pushes and
the assurance path did not, so assurance graded r against prices that were
never charged. Every per-day series shares `decision_day` -- the trading
date the decision priced, never the UTC wall clock of its outcome.
"""

import math

import numpy as np
import pandas as pd


# applied vs recommended price: float noise on a currency amount, not a knob
PRICE_MATCH_TOLERANCE = 1e-6

# what a pushed outcome must report for its (decision, outcome) pair to
# teach anything: the price we chose was the price on the shelf
LEARNABLE_STATUSES = (None, "ok", "success")


def iso_day(value):
    """One spelling of a trading day, whatever the producer's dtype: a
    parquet datetime column reads `2026-08-19 00:00:00` under str().
    Raises on a value that names no day (None, NaT, text that is not a
    date)."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"not a day: {value!r}")
    return stamp.strftime("%Y-%m-%d")


def ident(v):
    """One spelling of an identifier: pandas reads an integer column as
    float once it holds a NaN, so the feed's 7.0 must key the decision's
    "7" -- and a JSONL request's "7" must key an int history's 7. A NaN or
    an unparseable value raises -- the caller counts the row. The ONE
    spelling: the hour key, the price request and the feature service's
    history all read ids through it."""
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v) or v != int(v):
            raise ValueError(f"not an identifier: {v!r}")
        return str(int(v))
    if v is None or isinstance(v, (bool, np.bool_)):
        raise ValueError(f"not an identifier: {v!r}")
    return str(v)


def ident_series(s):
    """`ident` over a whole column, vectorised: the history's id columns
    are read once per batch (16M rows through a Python call per row was
    the morning). A NaN reads as None (a row that names no item can never
    be a feature for any request); a fractional value raises like `ident`."""
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
    """An hour of the day as the one int the key spells, or ValueError: a
    fractional or non-finite value names no hour (17.5 is not 17 -- every
    hand-rolled `int(float(h))` truncated it to one)."""
    h = float(hour)
    if not np.isfinite(h) or h != int(h):
        raise ValueError(f"not an hour: {hour!r}")
    return int(h)


def hour_key(sku, fc, date, hour):
    """The (sku, fc, day, hour) a feed row, a decision and a price request
    meet on -- the ONE key, spelt one way. Raises on a value that names no
    hour or no item; the caller decides whether that costs one row or one
    decision, never the batch."""
    return (ident(sku), ident(fc), iso_day(date), hour_int(hour))


def shelf_hour_tag(key):
    """`<sku>|<fc>|<date>T<hh>` for a hour key: the one spelling of a
    shelf-hour that every event id and the producers' new episode id are
    a prefix over (common.windows.assign_episode_ids spells the same tag
    vectorised for the history)."""
    sku, fc, day, hour = key
    return f"{sku}|{fc}|{day}T{hour:02d}"


def as_number(v):
    """`v` as a finite float, or None: None, NaN, inf, a non-numeric string
    or a value that will not parse all read as "no number". The one
    lenient reading the hourly scripts use on the feed's cells (a JSONL
    snapshot may carry "3"); engine.decide.finite_number is the STRICT
    test a contract value must pass and stays separate on purpose."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def colliding_keys(keys):
    """The keys claimed MORE THAN ONCE in `keys` -- the one rule for "two
    states for one hour": two requests in a batch, two decisions in the
    store, two feed rows for one hour. None of the claimants is preferable,
    so every caller matches none of them and counts all of them."""
    seen, out = set(), set()
    for k in keys:
        if k in seen:
            out.add(k)
        seen.add(k)
    return out


def decision_id_of(key):
    """The decision id for the hour keyed `key`: "dec-<sku>|<fc>|<date>T<hh>",
    the outcome id's twin over the ONE key (hour_key), so a decision and the
    outcome its hour produces are two prefixes of one string and engineering
    can name both before either exists. Natural, not a surrogate: a decision
    happens once per shelf-hour, so the id says the store's invariant out
    loud and a re-priced hour collides with ITSELF instead of landing a
    second price on one feed row.

    The episode is deliberately NOT in it. The episode id is the producers'
    and can be relabelled (a corrected relist, a drifted port); an audit
    record's identity may not move when an upstream label does."""
    return "dec-" + shelf_hour_tag(key)


def rejection_id_of(key):
    """The rejection id for the hour keyed `key`: "rej-<sku>|<fc>|<date>T<hh>",
    the third prefix over the one key. One record per shelf-hour refused, so
    re-running a refused hour collides with its own earlier copy."""
    return "rej-" + shelf_hour_tag(key)
