"""common.episodes: the window extension, closure and scrap keyed to the
source's sentinel, the adjustment reasons, and the COGS at risk per
episode (the episode-scoped cuts are test_windows)."""

import numpy as np
import pandas as pd
import pytest

from common import episodes
from conftest import episode_frame


def test_the_hour_discrepancy_has_one_home():
    """`(start - end) - sold` was spelled in three functions; shrink, the
    live adjustment and the flow's arrivals all read `hour_discrepancy`."""
    import inspect
    disc = episodes.hour_discrepancy([10, 8, 5], [1, 0, 3], [8, 8, 0])
    assert list(disc) == [1, 0, 2]          # shrink, clean, shrink
    assert list(episodes.hour_discrepancy([5], [1], [8])) == [-4]   # arrival
    for fn in (episodes.shrink_by_hour, episodes.episode_flow,
               episodes.hour_adjustment):
        src = inspect.getsource(fn)
        assert "hour_discrepancy(" in src, fn.__name__
        assert "- np.asarray(units_sold)" not in src, fn.__name__


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
    # <- read this first: the sentinel is read off the flow frame
    assert not episodes.write_off_convention(episodes.episode_flow(honest))
    kind = episodes.classify(honest)
    assert kind["a"] == episodes.NOT_CLOSED
    # "b" sold out AND its ending is genuinely 0, so it closed on its own
    # evidence -- the sentinel's absence elsewhere does not touch it
    assert kind["b"] == episodes.SOLD_OUT_EARLY
    assert pd.isna(episodes.scrap_units(honest)["a"])

    mixed = _last_row_frame([("a", 3, 9, 4, 5), ("w", 1, 8, 2, 0)])
    assert episodes.write_off_convention(episodes.episode_flow(mixed))
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


def test_identity_violations_are_the_rows_the_flow_already_marks():
    """`flow_identity_violations` re-derived `opening + arrived != sold +
    scrap`, which `episode_flow` had already decided as `accounting_closes`;
    it now reads that column, and the flow carries no leftover `net`."""
    rows = [("A", 10, 3, 10, 4, 6), ("A", 11, 2, 6, 3, 3), ("A", 12, 1, 3, 0, 0),
            ("B", 10, 9, 10, 1, 9), ("B", 11, 8, 9, 1, 8)]
    d = episode_frame(rows, columns=["episode_id", "hour_of_day", "hours_remaining",
                                     "starting_inventory", "units_sold",
                                     "ending_inventory"], date="2026-03-01")
    flow = episodes.episode_flow(d)
    assert "net" not in flow.columns
    assert flow.accounting_closes.all()
    assert episodes.flow_identity_violations(d).empty
    broken = flow.copy()
    broken.loc["B", "accounting_closes"] = False
    assert list(episodes.flow_identity_violations(d, flow=broken).index) == ["B"]


# ------------------------------------------------------- COGS at risk

def test_cogs_at_risk_counts_supply_not_opening_stock():
    """A window that opens with 3 and takes 10 mid-flight has 13 units of
    cost at risk; counting 3 understates every restocked episode."""
    from common.episodes import cogs_at_risk

    # one episode: opens with 3, 10 arrive in hour 2, sells 9, loses 1
    d = pd.DataFrame({
        "episode_id": ["e"] * 3,
        # `hour_adjustment` establishes window order from these, so the
        # arrival term needs them -- every real caller has them, since
        # `assign_episode_ids` needs them first
        "date": ["2026-03-01"] * 3, "hour_of_day": [10, 11, 12],
        "cost": [100.0] * 3,
        "starting_inventory": [3, 13, 4],
        "units_sold": [0, 9, 3],
        "ending_inventory": [13, 4, 0],
    })
    # 3 opening + 10 arrived = 13 units x 100
    assert cogs_at_risk(d) == pytest.approx(1300.0)

    # no arrivals -> unchanged from the old opening-stock reading
    flat = pd.DataFrame({
        "episode_id": ["f"] * 2, "cost": [50.0] * 2,
        "date": ["2026-03-01"] * 2, "hour_of_day": [10, 11],
        "starting_inventory": [8, 5], "units_sold": [3, 5],
        "ending_inventory": [5, 0],
    })
    assert cogs_at_risk(flat) == pytest.approx(400.0)


def test_the_per_episode_cogs_table_reproduces_every_stage_and_flag_reading():
    """cogs_at_risk re-ran the arrival pass on every waterfall row and every
    flag mask (~24 passes per run). Every stage after the ids are fixed drops
    WHOLE episodes, so one per-episode table summed over the episodes left
    gives the same number -- NaN included, never skipped."""
    from common.episodes import cogs_at_risk, episode_cogs

    d = pd.DataFrame({
        "episode_id": ["e"] * 3 + ["f"] * 2 + ["n"] * 2,
        "date": ["2026-03-01"] * 7, "hour_of_day": [10, 11, 12, 10, 11, 10, 11],
        "cost": [100.0] * 3 + [50.0] * 2 + [np.nan] * 2,
        "starting_inventory": [3, 13, 4, 8, 5, 2, 1],
        "units_sold": [0, 9, 3, 3, 5, 1, 1],
        "ending_inventory": [13, 4, 0, 5, 0, 1, 0],
    })
    table = episode_cogs(d)
    assert table["e"] == pytest.approx(1300.0) and table["f"] == pytest.approx(400.0)
    assert np.isnan(table["n"])
    for keep in (["e"], ["f"], ["e", "f"], ["e", "n"], ["e", "f", "n"], []):
        sub = d[d.episode_id.isin(keep)]
        direct, tabled = cogs_at_risk(sub), cogs_at_risk(sub, table)
        assert (np.isnan(direct) and np.isnan(tabled)) or direct == tabled, keep
    # the chain's callers reach it by the name they imported
    from fit import prepare_data
    assert prepare_data.episode_cogs is episode_cogs
