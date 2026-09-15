"""common.windows: the window boundary rule (ids, the restock and close
clauses, the defective-window drops, the counter recovery) and the
episode-scoped cuts at the midnight seam."""

import numpy as np
import pandas as pd
import pytest

from common import windows
from common.windows import (assign_episode_ids, counter_step_detail,
                            gap_split_windows, null_counter_windows,
                            recover_negative_windows, window_signals,
                            window_starts)
from conftest import _frame, _shelf, _window, episode_frame


# ------------------------------------------------------ episode-scoped cuts

def test_window_slice_takes_whole_episodes_or_none():
    out = windows.window_slice(_frame(), "2026-08-04", "2026-08-21")
    assert set(out.episode_id) == {"inside"}
    assert len(out) == 4          # all four of its rows, none of the other's


def test_row_level_slicing_is_what_this_prevents():
    d = _frame()
    naive = d[d.date.astype(str).ge("2026-08-04")]
    # the naive cut keeps 4 orphan hours of an episode that opened the day
    # before -- a "short episode" that never existed
    assert (naive.episode_id == "crosses").sum() == 4
    assert "crosses" not in set(
        windows.window_slice(d, "2026-08-04", None).episode_id)


def test_window_slice_assigns_every_episode_to_exactly_one_slice():
    d = _frame()
    a = windows.window_slice(d, None, "2026-08-03")
    b = windows.window_slice(d, "2026-08-04", None)
    assert set(a.episode_id) | set(b.episode_id) == {"crosses", "inside"}
    assert not set(a.episode_id) & set(b.episode_id)
    assert len(a) + len(b) == len(d)


def test_window_slice_is_a_noop_without_bounds():
    d = _frame()
    assert windows.window_slice(d) is d


def test_a_precomputed_opening_date_gives_the_same_cut():
    """A weekly schedule sliced the whole scope once per week, regrouping it
    every time; the opening date is computed once and passed through
    `opened`, and the cut -- and the weeks-seen count -- are unchanged."""
    d = _frame()
    opened = windows.opening_dates(d)
    assert list(opened.unique()) == ["2026-08-03", "2026-08-04"]
    for start, end in (("2026-08-04", None), (None, "2026-08-03"),
                       ("2026-08-04", "2026-08-21")):
        pd.testing.assert_frame_equal(
            windows.window_slice(d, start, end),
            windows.window_slice(d, start, end, opened=opened))
    plain = windows.trailing_weeks_window(d, "2026-08-10", 1)
    fed = windows.trailing_weeks_window(d, "2026-08-10", 1, opened=opened)
    pd.testing.assert_frame_equal(plain[0], fed[0])
    assert plain[1] == fed[1] == 1


def test_the_trailing_fit_window_keeps_episodes_whole_at_the_week_seam():
    """Both schedule loops cut the trailing window by ROW week, so an
    episode opening Sunday and closing Monday lost its Monday rows -- and
    the artifact schedule and shadow's re-fit solved on different rows."""
    d = pd.DataFrame({
        "episode_id": ["a", "a", "b", "b"],
        "date": ["2026-08-09", "2026-08-10",       # Sun -> Mon (week seam)
                 "2026-08-10", "2026-08-11"],      # opens in the fit week
        "hour_of_day": [23, 0, 9, 10],
    })
    window, weeks = windows.trailing_weeks_window(d, "2026-08-10", 1)
    assert sorted(window.episode_id.unique()) == ["a"]
    assert len(window) == 2 and weeks == 1
    empty, none = windows.trailing_weeks_window(d, "2026-08-03", 1)
    assert len(empty) == 0 and none == 0


def test_anchor_rows_are_one_mask():
    """Five sites spelled `(total_discount - d_ref).abs() <= tier_step / 2`
    by hand; the fit, the fidelity ratio and the gate share must agree on
    what an anchor row is."""
    d = pd.DataFrame({"total_discount": [0.30, 0.32, 0.33, 0.28],
                      "d_ref": [0.30] * 4})
    assert windows.is_anchor_row(d, 0.05).tolist() == [True, True, False, True]


# ------------------------------------------------- episode ids and windows

def test_episode_spans_midnight_as_one_window():
    """FLC windows commonly run past midnight -- 36 hours is common. A
    date-keyed episode would split one economic window into three, resetting
    the monotonicity anchor and charging carried inventory to scrap twice."""
    long_window = _window(1, "FC1", "2026-03-01 10:00", 36)
    d = long_window.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    ids = assign_episode_ids(d)
    assert ids.nunique() == 1, "a 36-hour window must be ONE episode"
    assert d.date.nunique() == 2, "and it must genuinely cross midnight"
    assert ids.iloc[0] == "1|FC1|2026-03-01T10"
    assert len(d) == 36 and d.hours_remaining.iloc[-1] == 0

    # a window long enough to cross twice is still one episode
    three = _window(1, "FC1", "2026-03-01 20:00", 36)
    three = three.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert assign_episode_ids(three).nunique() == 1
    assert three.date.nunique() == 3


def test_back_to_back_windows_and_gaps_still_split():
    # two windows abutting with no time gap: only the counter reset separates
    # them, so time-contiguity alone would wrongly merge these
    a = _window(1, "FC1", "2026-03-01 10:00", 6)
    b = _window(1, "FC1", "2026-03-01 16:00", 6)
    d = pd.concat([a, b]).sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert assign_episode_ids(d).nunique() == 2

    # a missing hour inside a window splits it, so an episode's row count
    # always equals its clock -- validate_state rejects any mismatch
    g = _window(1, "FC1", "2026-03-01 10:00", 6).drop(index=3)
    assert assign_episode_ids(g).nunique() == 2

    # different sku x fc never merge
    two = pd.concat([_window(1, "FC1", "2026-03-01 10:00", 4),
                     _window(2, "FC1", "2026-03-01 10:00", 4)])
    two = two.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert assign_episode_ids(two).nunique() == 2


def test_a_new_window_is_not_mistaken_for_a_gap():
    """The counter is what tells them apart, and it must."""
    def frame(rows):
        d = episode_frame(rows, columns=["hour_of_day", "hours_remaining"],
                          date="2026-03-01", sku_id="S", fc="F",
                          starting_inventory=5, units_sold=0, ending_inventory=5)
        d["episode_id"] = assign_episode_ids(d)
        return d

    # one window, hour 13 missing: clock +2, counter -2 -> a GAP
    ids, detail = gap_split_windows(
        frame([(10, 5), (11, 4), (12, 3), (14, 1)]))
    assert detail["windows_split_by_a_feed_gap"] == 1
    assert len(ids) == 2, "both fragments must be named"
    assert detail["missing_hours"] == 1

    # two back-to-back windows, one idle hour between: the counter RESETS
    ids, detail = gap_split_windows(
        frame([(10, 3), (11, 2), (12, 1), (14, 9), (15, 8)]))
    assert len(ids) == 0, "a new window was deleted as if it were a gap"

    # and two windows with no idle hour at all
    ids, detail = gap_split_windows(
        frame([(10, 3), (11, 2), (12, 1), (13, 9), (14, 8)]))
    assert len(ids) == 0


def test_recovery_cannot_merge_a_negative_episode_into_its_neighbour():
    """`negative_window_recovered` rewrites the field the ids are derived from."""
    shelf = dict(starting_inventory=5, units_sold=0, ending_inventory=5)
    rows = ([dict(sku_id=1, fc="X", date="2026-03-01", hour_of_day=h,
                  hours_remaining=hr, **shelf) for h, hr in
             [(10, -5.0), (11, -6.0), (12, -7.0)]]                  # enters negative
            + [dict(sku_id=1, fc="X", date="2026-03-01", hour_of_day=h,
                    hours_remaining=hr, **shelf) for h, hr in
               [(13, 20.0), (14, 19.0)]])                           # a REAL next window
    raw = pd.DataFrame(rows)
    raw["episode_id"] = assign_episode_ids(raw)
    assert raw.episode_id.nunique() == 2, "the two windows are distinct at source"

    # recovery as the chain applies it -- the chain's own function
    d, rec = recover_negative_windows(raw, 24)
    assert rec.sum() == 3 and list(d.hours_remaining[:3]) == [23.0, 22.0, 21.0]

    # re-deriving ids from the REWRITTEN counter is what used to happen, and
    # it silently fuses the two windows
    assert assign_episode_ids(d).nunique() == 1, (
        "the collision this ordering exists to avoid no longer reproduces -- "
        "if recovery changed, re-check whether the ordering is still needed")
    # ...but the ids the pipeline carries are untouched, which is the fix
    assert d.episode_id.nunique() == 2


# ------------------------------------------ the null-counter window drop

def test_a_null_counter_drops_its_whole_window_not_a_fragment():
    """Rule 15 at the null-counter stage: the run the null sits in goes
    whole, and a back-to-back neighbour with its own clock break stays."""
    df = pd.DataFrame({
        "starting_inventory": [5] * 6, "units_sold": [0] * 6, "ending_inventory": [5] * 6,
        "sku_id": [7] * 6, "fc": ["F"] * 6,
        "date": ["2026-08-01"] * 6,
        "hour_of_day": [9, 10, 11, 14, 15, 16],       # a break between 11 and 14
        "hours_remaining": [2.0, np.nan, 0.0, 2.0, 1.0, 0.0]})
    mask, detail = null_counter_windows(df)
    assert list(mask) == [True, True, True, False, False, False]
    assert detail == {"windows": 1, "gap_fragments_kept": 0}
    clean, detail = null_counter_windows(df.assign(hours_remaining=[2.0, 1, 0, 2, 1, 0]))
    assert not clean.any() and detail["windows"] == 0

    # back-to-back windows with NO clock gap: the counter resets upward
    # between two non-null rows, so the neighbour is its own window and
    # survives (the contract: two windows back to back are two episodes)
    b2b = pd.DataFrame({
        "starting_inventory": [5] * 6, "units_sold": [0] * 6, "ending_inventory": [5] * 6,
        "sku_id": [7] * 6, "fc": ["F"] * 6, "date": ["2026-08-01"] * 6,
        "hour_of_day": [9, 10, 11, 12, 13, 14],
        "hours_remaining": [2.0, np.nan, 0.0, 2.0, 1.0, 0.0]})
    mask, detail = null_counter_windows(b2b)
    assert list(mask) == [True, True, True, False, False, False]
    assert detail["windows"] == 1
    # two nulls in two chained windows count as two windows
    mask, detail = null_counter_windows(
        b2b.assign(hours_remaining=[2.0, np.nan, 0.0, 2.0, np.nan, 0.0]))
    assert mask.all() and detail["windows"] == 2

    # a feed gap the counter ran down across is ONE window: the far
    # fragment goes with it (gap_split_windows could no longer see the gap
    # once the near side was dropped)
    gap = pd.DataFrame({
        "starting_inventory": [5] * 6, "units_sold": [0] * 6, "ending_inventory": [5] * 6,
        "sku_id": [7] * 6, "fc": ["F"] * 6, "date": ["2026-08-01"] * 6,
        "hour_of_day": [9, 10, 11, 14, 15, 16],
        "hours_remaining": [7.0, np.nan, 5.0, 2.0, 1.0, 0.0]})
    mask, detail = null_counter_windows(gap)
    assert mask.all() and detail == {"windows": 1, "gap_fragments_kept": 0}
    # a gap with the null right beside it cannot be read: the far side
    # survives as a fragment, and the detail says so
    beside = gap.assign(hours_remaining=[7.0, 6.0, np.nan, 2.0, 1.0, 0.0])
    mask, detail = null_counter_windows(beside)
    assert list(mask) == [True, True, True, False, False, False]
    assert detail["gap_fragments_kept"] == 1


# ------------------------------------------- restock-extended windows (rule)

def test_a_restock_extended_window_is_one_episode():
    """Engineering: stock arriving mid-window extends it, and the counter
    steps UP from the NEXT hour. Hour 12 opens with 3, sells 1 and ends
    with 6 (4 arrived); hour 13 opens with 6 and the counter jumps 1 -> 4.
    Same listing, one id -- and every restock re-tests on its own."""
    d = _shelf(hours=[10, 11, 12, 13, 14], counters=[4, 3, 2, 4, 3],
               start=[5, 4, 3, 6, 5], sold=[1, 1, 1, 1, 1], end=[4, 3, 6, 5, 4])
    assert assign_episode_ids(d).nunique() == 1
    assert counter_step_detail(d) == {"up_steps_at_one_hour": 1, "restock_continued": 1,
                                      "closed_new_window": 0, "reset_new_window": 0,
                                      "closed_then_resumed": 0}
    # the same up-step with NO stock arriving the hour before is a reset:
    # two back-to-back windows, two ids
    reset = d.copy()
    reset.loc[reset.hour_of_day == 12, "ending_inventory"] = 2   # 3 - 1, reconciles
    reset.loc[reset.hour_of_day >= 13, ["starting_inventory", "ending_inventory"]] -= 4
    assert assign_episode_ids(reset).nunique() == 2
    assert counter_step_detail(reset)["reset_new_window"] == 1
    # a flat step after a restock continues too; three restocks, one id
    three = _shelf(hours=list(range(10, 17)), counters=[6, 5, 4, 5, 4, 4, 3],
                   start=[5, 4, 3, 6, 5, 8, 7], sold=[1] * 7,
                   end=[4, 3, 6, 5, 8, 7, 6])
    assert assign_episode_ids(three).nunique() == 1


def test_a_write_off_zero_closes_the_window_whatever_the_counter_does():
    """Engineering: `ending_inventory == 0` is the close, even on an hour
    that also restocked; the next row opens a new id -- whether the
    counter resets or, mid-window, keeps counting down (that zero is a
    write-off leftover, not shrink)."""
    # closed at 12 (sold 5 of 5), relisted at 13 with a fresh counter
    relist = _shelf(hours=[10, 11, 12, 13, 14], counters=[2, 1, 0, 6, 5],
                    start=[7, 6, 5, 9, 8], sold=[1, 1, 5, 1, 1], end=[6, 5, 0, 8, 7])
    assert assign_episode_ids(relist).nunique() == 2
    assert counter_step_detail(relist)["closed_new_window"] == 1
    # the zero hour sold MORE than it opened with (a restock by C9) and
    # still ended at zero: closed, not continued
    oversold = relist.copy()
    oversold.loc[oversold.hour_of_day == 12, "units_sold"] = 8
    assert assign_episode_ids(oversold).nunique() == 2
    # a mid-window zero with the counter still ticking -1: two ids
    mid = _shelf(hours=[10, 11, 12, 13, 14], counters=[4, 3, 2, 1, 0],
                 start=[5, 4, 3, 2, 1], sold=[1, 1, 3, 1, 1], end=[4, 3, 0, 1, 0])
    assert assign_episode_ids(mid).nunique() == 2


def test_the_null_counter_run_reads_the_same_boundaries():
    """The null-counter drop reads the window the id rule would: a null
    after a restock-extended step is inside ONE window (the whole run
    drops), and a null after a close belongs to the relist alone."""
    ext = _shelf(hours=[10, 11, 12, 13, 14], counters=[4.0, 3.0, 2.0, np.nan, 3.0],
                 start=[5, 4, 3, 6, 5], sold=[1] * 5, end=[4, 3, 6, 5, 4])
    mask, detail = null_counter_windows(ext)
    assert mask.all() and detail["windows"] == 1
    closed = _shelf(hours=[10, 11, 12, 13, 14], counters=[2.0, 1.0, 0.0, np.nan, 5.0],
                    start=[7, 6, 5, 9, 8], sold=[1, 1, 5, 1, 1], end=[6, 5, 0, 8, 7])
    mask, detail = null_counter_windows(closed)
    assert list(mask) == [False, False, False, True, True] and detail["windows"] == 1


def test_the_null_run_drop_reads_the_ids_boundaries_on_flat_and_minus_two_steps():
    """Counters [3, NaN, 1, 1, 0]: the ids split at the flat step (no
    restock, EPISODE_RULE), so the null's window is the first three rows
    and the two after it are their own window -- the run drop once
    swallowed all five. `window_signals.counter_ok` is the one clause."""
    d = _shelf(hours=[10, 11, 12, 13, 14], counters=[3.0, np.nan, 1.0, 1.0, 0.0],
               start=[5] * 5, sold=[0] * 5, end=[5] * 5)
    assert list(window_signals(d)["counter_ok"]) == [False, False, False, False, True]
    assert list(window_starts(d)) == [True, True, True, True, False]
    assert assign_episode_ids(d).nunique() == 4
    mask, detail = null_counter_windows(d)
    assert list(mask) == [True, True, True, False, False] and detail["windows"] == 1
    # a -2 step between two readable counters is a new window too
    skip = _shelf(hours=[10, 11, 12, 13, 14], counters=[4.0, np.nan, 2.0, 0.0, 6.0],
                  start=[5] * 5, sold=[0] * 5, end=[5] * 5)
    mask, _ = null_counter_windows(skip)
    assert list(mask) == [True, True, True, False, False]
    assert list(window_starts(skip)) == [True, True, True, True, True]


def test_the_boundary_functions_stay_reachable_by_their_old_names():
    """Every caller of the rule imported it from `fit.prepare_data` (and the
    cuts from `common.episodes`); the names still resolve there, to the one
    function -- a second copy would be the drift the home exists to stop."""
    from common import episodes
    from fit import prepare_data
    assert prepare_data.window_starts is window_starts
    assert prepare_data.assign_episode_ids is assign_episode_ids
    assert prepare_data.EPISODE_RULE == windows.EPISODE_RULE
    assert episodes.window_slice is windows.window_slice
    assert episodes.is_anchor_row is windows.is_anchor_row
    with pytest.raises(AttributeError):
        windows.episode_flow                       # the accounting is episodes'
