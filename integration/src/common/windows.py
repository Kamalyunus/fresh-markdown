"""common.windows -- where an episode starts and ends, and which rows a fit
may read: the home of the window boundary rule (EPISODE_RULE, the ids, the
defective-window drops and the counter recovery -- design 5.2, 12a) and of
every episode-scoped cut (window_slice, the trailing fit windows, the week
keys, the opening dates, the planning horizon at a row).

Pure pandas over the canonical frame; reads no config and nothing else in
`common`. `common.episodes` (the inventory accounting) reads it, never the
reverse."""

import numpy as np
import pandas as pd


# ------------------------------------------------------- calendar cuts

def planning_horizon(counter):
    """Hours the solver plans over at a row whose window counter reads
    `counter`: the counter is the hours STILL TO COME after this one, so
    the horizon is this hour plus the counter. The ONE home of that `+ 1`
    (shadow, the replay, the simulator's templates and the request's
    `hours_remaining` all read it; four spellings once disagreed by an
    hour on a restock-extended window). `window_counter` is its inverse."""
    return int(counter) + 1


def week_key(dates):
    """`week_start` per row, as the "YYYY-MM-DD" key the factor schedules use."""
    return (pd.to_datetime(dates).dt.to_period("W").dt.start_time
            .dt.strftime("%Y-%m-%d"))


def last_rows(d, order=("date", "hour_of_day")):
    """Final row of each episode, in window order."""
    return d.sort_values(list(order)).groupby("episode_id").tail(1)


def hours_between(day_a, hour_a, day_b, hour_b):
    """Whole hours from (day_a, hour_a) to (day_b, hour_b); negative when
    b is earlier. The one clock-step reading the live rule, the hourly
    script and the producers' script share -- here, not in the engine, so
    the producers' script needs no engine."""
    a = pd.Timestamp(day_a) + pd.Timedelta(hours=int(hour_a))
    b = pd.Timestamp(day_b) + pd.Timedelta(hours=int(hour_b))
    return int(round((b - a).total_seconds() / 3600.0))


def opening_dates(d):
    """The date each row's episode OPENED on, as "YYYY-MM-DD" per row -- the
    key every episode-scoped cut assigns by. A caller slicing one frame many
    times (a weekly schedule, the prior's folds) computes it once and passes
    it to `window_slice` / `trailing_weeks_window` as `opened`."""
    return d.groupby("episode_id")["date"].transform("min").astype(str)


# ------------------------------------------------- the window boundary rule

# Persisted with the split manifest; production must derive identical boundaries.
EPISODE_RULE = (
    "episode_id = sku_id|fc|<first hour of the window>: a maximal run of "
    "consecutive hourly rows for one SKU x FC. A row opens a new window when "
    "the clock did not advance exactly one hour, when the previous hour "
    "CLOSED (ending_inventory == 0, the write-off sentinel -- whatever the "
    "counter does next), or when hours_remaining did not decrement by one -- "
    "EXCEPT an upward (or flat) step on the hour after stock arrived "
    "(ending > starting - sold on the previous row): a restock extends the "
    "window and the counter moves from the next hour, and every restock "
    "re-tests on its own, so a window restocked many times is one episode. "
    "NOT keyed by calendar date -- windows cross midnight, and a date key "
    "would split one episode in two at the seam.")


def window_signals(df):
    """The per-row signals every window reading is built from, per SKU x FC
    in window order: the clock step `dt_h` (hours since the previous row),
    the counter step `hr_diff`, what the PREVIOUS hour did --
    `prev_closed` (ending_inventory == 0: the listing closed, contract C2)
    and `prev_restock` (ending > starting - sold: stock arrived, C9) -- and
    `counter_ok`, the counter clause of EPISODE_RULE: the step is -1, or
    upward/flat on the hour after a restock. NaN on a group's first row for
    the two steps (and so False for `counter_ok`), False for the flags."""
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    grp = [df.sku_id, df.fc]
    prev = {c: df[c].groupby(grp).shift() for c in
            ("starting_inventory", "units_sold", "ending_inventory")}
    hr_diff = df.hours_remaining.groupby(grp).diff()
    prev_restock = (prev["ending_inventory"]
                    > prev["starting_inventory"] - prev["units_sold"])
    return {
        "dt_h": ts.groupby(grp).diff().dt.total_seconds() / 3600.0,
        "hr_diff": hr_diff,
        "prev_closed": prev["ending_inventory"].eq(0),
        "prev_restock": prev_restock,
        "counter_ok": hr_diff.eq(-1.0) | (hr_diff.gt(-1.0) & prev_restock),
    }


def window_starts(df):
    """True where a row opens a window (EPISODE_RULE) -- the ONE boundary
    reading: assign_episode_ids keys the ids on it and the defective-window
    drops (`defective_windows`) read the same signals. Either clock or
    counter alone would merge back-to-back windows or stitch across a feed
    gap; a null counter (a NaN step) opens a window here on purpose -- the
    phantom id is what null_counter_windows then drops with its whole run."""
    s = window_signals(df)
    return s["dt_h"].ne(1.0) | s["prev_closed"] | ~s["counter_ok"]


def assign_episode_ids(df):
    """Episode ids as `sku_id|fc|<first hour of the window>` over
    `window_starts`. Crossing midnight is an ordinary one-hour step, which
    is the point."""
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    start_ts = ts.where(window_starts(df)).groupby([df.sku_id, df.fc]).ffill()
    return (df.sku_id.astype(str) + "|" + df.fc.astype(str) + "|"
            + start_ts.dt.strftime("%Y-%m-%dT%H"))


def counter_step_detail(df):
    """The up/flat counter steps at a one-hour clock step, by what the
    previous hour did -- the measurement behind EPISODE_RULE's restock
    clause: `restock_continued` (stock arrived last hour: the same
    window), `closed_new_window` (last hour closed: a relist),
    `reset_new_window` (neither: back-to-back windows). And, whatever the
    counter did, `closed_then_resumed`: the hour after a close opened
    with NOTHING on the shelf -- the feed carried on past a sell-out
    (contract C3's open question), a window `opens_empty` then flags."""
    s = window_signals(df)
    one_hour = s["dt_h"].eq(1.0)
    up = one_hour & s["hr_diff"].gt(-1.0)
    return {"up_steps_at_one_hour": int(up.sum()),
            "restock_continued": int((up & s["prev_restock"] & ~s["prev_closed"]).sum()),
            "closed_new_window": int((up & s["prev_closed"]).sum()),
            "reset_new_window": int((up & ~s["prev_closed"] & ~s["prev_restock"]).sum()),
            "closed_then_resumed": int((one_hour & s["prev_closed"]
                                        & df.starting_inventory.eq(0)).sum())}


def gap_split_windows(df):
    """Ids of EVERY fragment of a source window a missing hour split in two --
    neither fragment is a real episode, and the second's first row would enter
    the entry-only elasticity fit with the wrong opening state. Detected from
    the counter falling in step with the clock (a new window resets upward)."""
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    grp = [df.sku_id, df.fc]
    dt_h = ts.groupby(grp).diff().dt.total_seconds() / 3600.0
    hr_drop = -df.hours_remaining.groupby(grp).diff()
    # the clock skipped hours and the counter ran down by exactly as many
    gap = (dt_h > 1) & (dt_h == hr_drop)
    if not gap.any():
        return np.array([], dtype=object), {}

    # walk fragments into windows: a row opens a NEW window only if it starts
    # a new episode for some reason OTHER than the gap
    starts = df.episode_id.ne(df.episode_id.groupby(grp).shift())
    window = df.episode_id.where(starts & ~gap).groupby(grp).ffill()
    per_window = df.groupby(window).episode_id.nunique()
    broken = per_window.index[per_window > 1]
    ids = df.loc[window.isin(broken), "episode_id"].unique()
    detail = {
        "windows_split_by_a_feed_gap": int(len(broken)),
        "fragments_dropped": int(len(ids)),
        "missing_hours": int((dt_h[gap] - 1).sum()),
        "note": ("Every fragment of a gap-split window is dropped: the "
                 "second opens mid-window and its first row would read as an "
                 "ENTRY row in the elasticity fit."),
    }
    return ids, detail


# The columns an episode id is built from. A null in any of them has no
# episode to belong to: `assign_episode_ids` groups on sku_id x fc (a NaN key
# falls out of every groupby) and stamps the window start from date x hour,
# so such rows collapsed into one NaN "episode" that later stages then read
# as a window. INTEGRITY, so it DROPS (rule 14) -- counted, never silent.
EPISODE_KEY = ("sku_id", "fc", "date", "hour_of_day")

# The integer quantities the inventory chain is read from. A null in one is
# an integrity defect of the window it sits in (a null counter likewise):
# dropped whole and counted, never an int cast that fails before the
# first waterfall row is written.
QUANTITY_COLS = ("starting_inventory", "units_sold", "ending_inventory")


def null_key_rows(df):
    """Mask of rows with a null in any EPISODE_KEY column, and a per-column
    count for the waterfall detail. Such a row can be placed in no window
    at all; row-scoped by construction."""
    nulls = df[list(EPISODE_KEY)].isna()
    return nulls.any(axis=1), {c: int(nulls[c].sum()) for c in EPISODE_KEY}


def source_windows(df):
    """The source WINDOW each row sits in, as `window_starts` reads it
    (EPISODE_RULE, one home), with exactly two tolerances a defective row
    needs: a DUPLICATE hour stays in its window (the clock did not advance
    -- the copies are one hour), and a step whose counter is UNREADABLE
    (NaN beside a null) is taken on the clock alone -- one hour on, or a
    feed gap the counter ran down by exactly. A flat or upward step
    between two readable counters is a new window unless the previous
    hour restocked, and a closed previous hour opens one whatever the
    counter does, exactly as the ids do; the run drop and the ids once
    disagreed here and the drop swallowed a neighbouring window. Returns
    (window ids, mask of rows that open a window across an unreadable
    gap -- a fragment nothing later can tell from a real entry)."""
    sig = window_signals(df)
    dt_h, hr_diff = sig["dt_h"], sig["hr_diff"]
    hr_drop = -hr_diff
    gap_ok = (dt_h > 1.0) & (dt_h == hr_drop)
    same = ((dt_h.eq(1.0) & (sig["counter_ok"] | hr_diff.isna()))
            | dt_h.eq(0.0) | gap_ok) & ~sig["prev_closed"]
    window = (~same.fillna(False) | dt_h.isna()).cumsum()
    unread = (dt_h > 1.0) & hr_drop.isna()
    return window, unread


def defective_windows(df, bad):
    """Mask of every row of the source window holding a row `bad` marks,
    and a detail: `windows` dropped and `gap_fragments_kept` -- windows
    that opened across a gap the null made unreadable, whose far side
    survives (bounded to a null adjacent to a gap). Rule 15 at the
    row-defect stages: a row-scoped drop left a fragment opening
    mid-window (when the defect was the window's first hour, no later
    stage could tell it from a real entry), so the whole window goes."""
    bad = pd.Series(np.asarray(bad, dtype=bool), index=df.index)
    if not bad.any():
        return pd.Series(False, index=df.index), {"windows": 0,
                                                   "gap_fragments_kept": 0}
    window, unread = source_windows(df)
    hit = set(window[bad])
    mask = window.isin(hit)
    grp = [df.sku_id, df.fc]
    kept = unread & mask.groupby(grp).shift(fill_value=False) & ~mask
    return mask, {"windows": len(hit), "gap_fragments_kept": int(kept.sum())}


def recover_negative_windows(d, cap):
    """`hours_remaining` rewritten as a synthetic countdown `(cap-1) -
    position` on episodes that ENTER negative and fit inside `cap` hours
    (manufacturing SKUs). Returns (frame, mask of rewritten rows). The ONE
    step that mutates the counter the ids derive from: it must run AFTER the
    re-segmentation check, and the ids are never re-derived from it."""
    entry = d.groupby("episode_id")["hours_remaining"].transform("first")
    length = d.groupby("episode_id")["hours_remaining"].transform("size")
    recoverable = (entry < 0) & (length <= cap)
    if recoverable.any():
        d = d.copy()
        position = d.groupby("episode_id").cumcount()
        d.loc[recoverable, "hours_remaining"] = (cap - 1) - position[recoverable]
    return d, recoverable
