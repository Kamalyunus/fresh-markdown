"""common.episodes: episode-scoped cuts at the midnight seam, the window
extension, closure and scrap keyed to the source's sentinel, and the
adjustment reasons."""

import numpy as np
import pandas as pd
import pytest

from common import episodes
from conftest import _frame, episode_frame


def test_window_slice_takes_whole_episodes_or_none():
    out = episodes.window_slice(_frame(), "2026-08-04", "2026-08-21")
    assert set(out.episode_id) == {"inside"}
    assert len(out) == 4          # all four of its rows, none of the other's


def test_row_level_slicing_is_what_this_prevents():
    d = _frame()
    naive = d[d.date.astype(str).ge("2026-08-04")]
    # the naive cut keeps 4 orphan hours of an episode that opened the day
    # before -- a "short episode" that never existed
    assert (naive.episode_id == "crosses").sum() == 4
    assert "crosses" not in set(
        episodes.window_slice(d, "2026-08-04", None).episode_id)


def test_window_slice_assigns_every_episode_to_exactly_one_slice():
    d = _frame()
    a = episodes.window_slice(d, None, "2026-08-03")
    b = episodes.window_slice(d, "2026-08-04", None)
    assert set(a.episode_id) | set(b.episode_id) == {"crosses", "inside"}
    assert not set(a.episode_id) & set(b.episode_id)
    assert len(a) + len(b) == len(d)


def test_window_slice_is_a_noop_without_bounds():
    d = _frame()
    assert episodes.window_slice(d) is d


def test_implausible_window_is_refused_not_expanded():
    """flc_window carries very large values from upstream data issues. The
    window drives episode identification, the DP horizon AND the synthetic
    tail, so a bad value must be dropped upstream -- and if one ever reaches
    the extension it must raise, not generate an unbounded frame."""
    from common import episodes

    d = pd.DataFrame({
        "episode_id": ["e"], "date": [pd.Timestamp("2026-03-01").date()],
        "hour_of_day": [10], "hours_remaining": [9000],
        "starting_inventory": [3], "ending_inventory": [3],
        "units_sold": [0], "category": ["MEAT"],
    })
    with pytest.raises(ValueError, match="exceeds max_window_hours"):
        episodes.extend_to_window(d, ["category"], max_tail_hours=48)

    ok = d.assign(hours_remaining=[5])
    assert len(episodes.extend_to_window(ok, ["category"], 48)) == 6


def test_window_extension_removes_the_lookahead_horizon():
    """Rows stop at zero inventory, so the DP's horizon must come from the
    window, not from how many rows happen to exist -- a short row count is
    short BECAUSE the item sold out, which is future information."""
    from common import episodes

    # an 8-hour window that sold out after 3 hours
    d = pd.DataFrame({
        "episode_id": ["e"] * 3,
        "date": [pd.Timestamp("2026-03-01").date()] * 3,
        "hour_of_day": [10, 11, 12],
        "hours_remaining": [7, 6, 5],
        "starting_inventory": [3, 2, 1],
        "ending_inventory": [2, 1, 0],
        "units_sold": [1, 1, 1],
        "category": ["MEAT"] * 3,
    })
    assert len(d) == 3 and d.hours_remaining.iloc[0] + 1 == 8

    e = episodes.extend_to_window(d, ["category"])
    assert len(e) == 8, "the DP must see the whole window, not the 3 rows"
    assert e.hours_remaining.iloc[-1] == 0
    assert e.is_observed.sum() == 3 and (~e.is_observed).sum() == 5

    # rows remaining now equals the window at every row -- the invariant
    # validate_state enforces on the live path
    rows_left = np.arange(len(e), 0, -1)
    assert (rows_left == e.hours_remaining.to_numpy() + 1).all()

    # synthetic rows carry features but no sales, so observed-world
    # economics and fidelity are untouched by the extension
    assert e[~e.is_observed].units_sold.eq(0).all()
    assert e.category.notna().all()

    # a window that ran to the end is left exactly as it was
    done = d.assign(hours_remaining=[2, 1, 0])
    assert len(episodes.extend_to_window(done, ["category"])) == 3


def _last_row_frame(rows):
    """One row per episode: (episode_id, hr, start, sold, ending_inventory)."""
    return episode_frame(rows, columns=[
        "episode_id", "hours_remaining", "starting_inventory", "units_sold",
        "ending_inventory"], date=pd.Timestamp("2026-03-01").date(),
        hour_of_day=9)


def test_scrap_is_keyed_to_the_closure_sentinel_not_the_nominal_counter():
    """The counter is nominal and usually still positive when a listing ends,
    so `hours_remaining <= 0` classified ~99% of real leftover as unknown. What
    marks closure is the source's own sentinel: ending_inventory zeroed on the
    final row. Its ABSENCE is the only thing that makes an outcome unknown."""
    from common import episodes

    d = _last_row_frame([
        # counter at zero, stock left, sentinel present -- scrap. RARE in real
        # data: the counter reaches zero on ~0.1% of episodes.
        ("counter-zero", 0, 7, 0, 0),
        # counter STILL POSITIVE, stock left, sentinel present. The common
        # case, and it must count as scrap rather than unknown.
        ("early-leftover", 28, 9, 4, 0),
        # sold out: a genuine zero, and unambiguous whatever the sentinel says
        ("sold-out", 4, 5, 5, 0),
        # stock left and NO sentinel -- still open, or the feed cut it
        ("still-open", 6, 9, 4, 5),
    ])

    kind = episodes.classify(d)
    assert kind["counter-zero"] == episodes.COMPLETED
    assert kind["early-leftover"] == episodes.COMPLETED     # the fix
    assert kind["sold-out"] == episodes.SOLD_OUT_EARLY
    assert kind["still-open"] == episodes.NOT_CLOSED

    scrap = episodes.scrap_units(d)
    assert scrap["counter-zero"] == 7
    assert scrap["early-leftover"] == 5      # 9 - 4, NOT dropped as unknown
    assert scrap["sold-out"] == 0
    assert pd.isna(scrap["still-open"])      # unknown, NOT zero and NOT 5

    # the regression this test exists for: under the counter-keyed rule only
    # `counter-zero` scrapped, so 5 of the 12 knowable units vanished
    assert scrap.sum() == 12


def test_a_feed_with_no_closure_sentinel_reads_unclosed_and_says_so():
    """Closure is `ending_inventory == 0` on the last row and NOTHING else."""
    from common import episodes

    honest = _last_row_frame([("a", 3, 9, 4, 5), ("b", 2, 6, 6, 0)])
    assert not episodes.write_off_convention(honest)     # <- read this first
    kind = episodes.classify(honest)
    assert kind["a"] == episodes.NOT_CLOSED
    # "b" sold out AND its ending is genuinely 0, so it closed on its own
    # evidence -- the sentinel's absence elsewhere does not touch it
    assert kind["b"] == episodes.SOLD_OUT_EARLY
    assert pd.isna(episodes.scrap_units(honest)["a"])

    mixed = _last_row_frame([("a", 3, 9, 4, 5), ("w", 1, 8, 2, 0)])
    assert episodes.write_off_convention(mixed)
    assert episodes.classify(mixed)["a"] == episodes.NOT_CLOSED
    assert pd.isna(episodes.scrap_units(mixed)["a"])


def test_write_off_outcome_is_documented_not_quarantined():
    """The source zeroes ending_inventory at the window close (~49.5% of
    episodes). Unnamed, every one of those final-hour outcomes quarantines
    and event completeness collapses -- the shadow gate fails for what looks
    like a pipeline defect."""
    from events.store import _validate_outcome

    base = {"outcome_id": "o", "decision_id": "d", "units_sold": 3,
            "starting_inventory": 4, "ending_inventory": 0,
            "applied_price": 5000.0, "is_stockout": False,
            "execution_status": "ok", "finalized_at": "2026-03-01T20:00:00Z"}

    # 4 in, 3 sold -> 1 left, reported as 0: does not reconcile
    assert _validate_outcome(base), "must not pass undocumented"

    assert not _validate_outcome({**base,
                                  "adjustment_reason": "episode_close_write_off"})
    assert not _validate_outcome({**base, "ending_inventory": 5,
                                  "adjustment_reason": "intraday_restock"})

    # a clean reconciliation needs no reason at all
    assert not _validate_outcome({**base, "ending_inventory": 1})


def test_adjustment_reason_names_every_legitimate_break():
    """Anything legitimate but unnamed quarantines, and a quarantined outcome
    never lands -- so a naming gap shows up as failed event completeness, not
    as a labelling bug."""
    from common.episodes import adjustment_reason as why

    # a reported ZERO with stock remaining is the source's write-off, wherever
    # it falls. Position must NOT matter: the source zeroes at its own episode
    # boundary, which sits mid-episode once we merge a window across midnight.
    assert why(4, 3, 0) == "episode_close_write_off"
    assert why(9, 4, 0) == "episode_close_write_off"
    # clean sellout reconciles on its own, no reason needed
    assert why(3, 3, 0) is None
    # stock added
    assert why(5, 1, 8) == "intraday_restock"
    # ordinary hour that reconciles
    assert why(5, 1, 4) is None
    # PARTIAL shortfall -- above zero but below the leftover -- is SHRINK, and
    # it is NAMED. It returned None on purpose until it was measured, so that
    # unexplained loss would quarantine and stay visible. That was the last
    # place the live path called shrink an anomaly while the offline chain
    # called it an ordinary event: counted gross, booked into scrap, gating
    # nothing. A quarantined outcome never lands, so event completeness fell by
    # the feed's whole shrink rate and the shadow gate failed for something no
    # integration work could fix -- it was measuring the SOURCE. At ~2.8% of
    # decision hours the harness read 0.9718 against a 0.99 threshold.
    assert why(5, 1, 2) == "unexplained_shortfall"
    # ORDER MATTERS at the boundary: a zero ending is the CLOSE, not a shrink.
    # Asking the shortfall first would swallow every write-off there is.
    assert why(5, 1, 0) == "episode_close_write_off"


def test_a_censored_entry_row_is_a_one_hour_episode():
    """Which is why dropping them is cheap -- and why the cost is a selection
    bias, not a coverage one."""
    from common import episodes

    d = pd.DataFrame({
        "episode_id": ["a", "a", "b"],
        "date": ["2026-03-01"] * 3,
        "hour_of_day": [10, 11, 10],
        "starting_inventory": [5, 3, 4],
        "units_sold": [2, 3, 4],       # a closes by sell-out; b is one hour
        "ending_inventory": [3, 0, 0],
    })
    cen = episodes.censored_hours(d)
    entry_idx = d.sort_values(["episode_id", "hour_of_day"]).groupby(
        "episode_id").head(1).index

    # episode a: censored on its LAST row (index 1), not its entry row (0)
    assert cen[1] and not cen[0]
    # episode b is one hour, so entry IS last -- the only censored entry row
    assert cen[2]
    assert list(pd.Series(cen)[entry_idx]) == [False, True]
