"""ops.assign_episode_ids -- the episode-id rule as a script the data
producers run (or port).

The producers own the episode id: it names their listing, and the whole
pipeline, engine included, will be theirs to run. This module is the
reference implementation of the rule the offline chain applies
(common.windows.EPISODE_RULE), stated for one clock hour with nothing but
this hour's rows and the previous hour's rows (already carrying their
ids): no engine, no store, no model, no state beyond last hour's file --
it imports pandas and this repo's `common/` and `events.pairs` only, so
it runs where the producers run. tests/test_assign_episode_ids.py holds
the parity check against the chain over a whole synthetic day.

    A shelf's row continues last hour's episode on that shelf when ALL hold:
    last hour's row is exactly one hour earlier; it did not close the shelf
    (ending_inventory not the write-off zero); and the counter stepped down
    by one, or stepped up or flat while stock arrived (ending > inventory -
    units_sold last hour). Otherwise the row opens a new episode named
    skuseq|fc|<date>T<hour>.

Run: python3 -m ops.assign_episode_ids --hour <this hour's rows> --previous <last hour's rows, with ids> --out <rows with ids>
A first hour (no --previous) opens every shelf. Input and output are in
the feed's own schema; the only column added is `episode_id`.
"""

import argparse

import pandas as pd

from common.io import read_rows, write_frame
from common.windows import hours_between
from events.pairs import as_number, hour_int, hour_key, iso_day, shelf_hour_tag

RULE = ("continue last hour's episode on the shelf when last hour's row is one "
        "hour earlier, did not end at the write-off zero, and the counter "
        "stepped down by one or stepped up/flat while stock arrived "
        "(ending > inventory - units_sold); otherwise open "
        "skuseq|fc|<date>T<hour>")


def continues(prev, row):
    """RULE, for one shelf: `prev` is last hour's row (feed names, with
    `episode_id`), `row` this hour's; either may be None."""
    if prev is None or row is None:
        return False
    try:
        one = hours_between(iso_day(prev["date"]), hour_int(prev["hour"]),
                            iso_day(row["date"]), hour_int(row["hour"])) == 1
    except (KeyError, TypeError, ValueError):
        return False
    if not one:
        return False
    ending, start, sold = (as_number(prev.get("ending_inventory")), as_number(prev.get("inventory")),
                           as_number(prev.get("units_sold")))
    if ending is not None and ending == 0:
        return False                                       # the write-off zero closed it
    c_prev, c_now = as_number(prev.get("flc_window")), as_number(row.get("flc_window"))
    if c_prev is None or c_now is None:
        return False
    # the raw counter step, exactly as common.windows.window_signals diffs it
    step = c_now - c_prev
    if step == -1:
        return True
    if step < -1:
        return False
    if None in (ending, start, sold):
        return False
    return ending > start - sold                           # stock arrived: the same window


def new_id(row):
    """The id a new window opens with: the shelf-hour's tag, spelt as every
    event id spells it (events.pairs.shelf_hour_tag over hour_key)."""
    return shelf_hour_tag(hour_key(row["skuseq"], row["fc"], row["date"], row["hour"]))


def assign(rows, previous=None):
    """`rows` (this hour, feed names) with `episode_id` set by RULE against
    `previous` (last hour's rows, with ids). Rows that name no shelf-hour
    get None. Returns (rows, counts)."""
    last = {}
    for p in previous or []:
        try:
            last[hour_key(p["skuseq"], p["fc"], p["date"], p["hour"])[:2]] = p
        except (KeyError, TypeError, ValueError):
            continue
    out, counts = [], {"continued": 0, "new": 0, "unkeyable": 0}
    for r in rows:
        r = dict(r)
        try:
            key = hour_key(r["skuseq"], r["fc"], r["date"], r["hour"])[:2]
            eid = new_id(r)
        except (KeyError, TypeError, ValueError):
            r["episode_id"] = None
            counts["unkeyable"] += 1
            out.append(r)
            continue
        prev = last.get(key)
        if continues(prev, r) and prev.get("episode_id"):
            r["episode_id"] = prev["episode_id"]
            counts["continued"] += 1
        else:
            r["episode_id"] = eid
            counts["new"] += 1
        out.append(r)
    return out, counts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ops.assign_episode_ids", description=RULE)
    ap.add_argument("--hour", required=True, help="this hour's rows (feed schema)")
    ap.add_argument("--previous", default=None, help="last hour's rows, with episode_id")
    ap.add_argument("--out", required=True, help="this hour's rows with episode_id")
    args = ap.parse_args(argv)
    # common.io.read_rows: a null cell is None, never a NaN that reads as a
    # truthy id one row later
    rows = read_rows(args.hour)
    prev = read_rows(args.previous) if args.previous else None
    out, counts = assign(rows, prev)
    write_frame(pd.DataFrame(out), args.out)
    print(f"{len(out)} rows: {counts['continued']} continued, {counts['new']} new"
          + (f", {counts['unkeyable']} without a shelf-hour" if counts["unkeyable"] else "")
          + f" -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
