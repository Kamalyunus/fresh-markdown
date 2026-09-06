"""fit.prepare_data: episode ids and windows, the two populations and the
flow identity that decides them (and which consumer is entitled to which),
the rate features, the waterfall's own basis, and the pre-launch slice."""

import inspect

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fit.prepare_data import (DP_INELIGIBLE, load_and_filter, population,
                                    tag_dp_eligibility)
from conftest import episode_frame


def _frame(**over):
    """One clean two-hour episode; keyword overrides break it one way."""
    base = dict(episode_id="e", cost=4000.0, original_price=10_000.0,
                total_discount=0.25, date="2026-01-01", hour_of_day=[10, 11],
                starting_inventory=[12, 9], units_sold=[3, 2],
                ending_inventory=[9, 0], hours_remaining=[10.0, 9.0])
    d = episode_frame(**{**base, **over})
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    d["d_max"] = 1.0 - d.cost / d.original_price
    return d


# ------------------------------------------------------ the five conditions

@pytest.mark.parametrize("name,over", [
    ("cost_missing", {"cost": 0.0}),
    ("non_priceable", {"cost": 12_000.0}),
    ("negative_window", {"hours_remaining": [-242.0, -243.0]}),
    ("window_too_long", {"hours_remaining": [500.0, 499.0]}),
    # the last row sells more than it opened with, so the close is ambiguous
    ("final_hour_restock", {"starting_inventory": [12, 9],
                            "units_sold": [3, 12], "ending_inventory": [9, 0]}),
])
def test_each_condition_flags_and_names_itself(name, over, cfg):
    d, detail = tag_dp_eligibility(_frame(**over), cfg)
    assert not d.dp_eligible.any(), f"{name} was not caught"
    assert (d.dp_ineligible_reason == name).all()
    assert detail[name]["episodes"] == 1
    assert detail["episodes_dp_ineligible"] == 1


def test_a_clean_episode_is_eligible(cfg):
    d, detail = tag_dp_eligibility(_frame(), cfg)
    assert d.dp_eligible.all()
    assert d.dp_ineligible_reason.isna().all()
    assert detail["episodes_dp_eligible"] == 1


def test_the_flag_is_episode_scoped_not_row_scoped(cfg):
    """One bad hour flags the whole window: the monotonicity anchor carries
    that hour's price into every later one."""
    d = _frame()
    d.loc[1, "cost"] = 0.0
    tagged, _ = tag_dp_eligibility(d, cfg)
    assert not tagged.dp_eligible.any(), "only the offending row was flagged"


def test_below_cost_is_reported_but_does_not_gate(cfg):
    """A below-cost price is one the LEGACY policy set, and the agent is
    already constrained never to set one -- so it is a property of the
    history, not a defect in it."""
    d = _frame()
    d.loc[1, "total_discount"] = 0.95
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    tagged, detail = tag_dp_eligibility(d, cfg)

    assert tagged.dp_eligible.all(), "below-cost must not gate eligibility"
    assert tagged.below_cost_hours.all()
    assert tagged.dp_ineligible_reason.isna().all()
    assert detail["below_cost_hours"]["episodes"] == 1
    assert detail["below_cost_hours"]["still_dp_eligible"] == 1
    assert "below_cost" not in [n for n, _ in DP_INELIGIBLE]


def test_an_unfinished_episode_is_kept_but_gated_out_of_everything(cfg):
    """This reverses an earlier decision in this file, deliberately."""
    from fit.prepare_data import population
    unclosed = _frame(episode_id="u", ending_inventory=[9, 7])
    d = pd.concat([_frame(), unclosed], ignore_index=True)
    tagged, detail = tag_dp_eligibility(d, cfg)

    assert (tagged.edge_truncated == (tagged.episode_id == "u")).all()
    assert (tagged.outcome_known == (tagged.episode_id == "e")).all()

    # gated out of BOTH working populations...
    assert (tagged.dp_ineligible_reason.isna()
            == (tagged.episode_id == "e")).all()
    assert set(population(tagged, cfg, "eligible").episode_id) == {"e"}
    assert set(population(tagged, cfg, "dp_eligible").episode_id) == {"e"}
    # ...and still visible where the residue is measured
    assert set(population(tagged, cfg, "integrity").episode_id) == {"e", "u"}

    # the edge-vs-feed split survives the gating -- it is what says whether
    # the count is the extract boundary or something to chase
    edge = detail["edge_truncated"]
    assert edge["episodes_unclosed"] == 1
    assert edge["episodes_edge_truncated"] == 1
    assert edge["share_of_unclosed_explained_by_edge"] == 1.0
    assert "edge_truncated" not in [n for n, _ in DP_INELIGIBLE]


def test_edge_truncation_survives_a_counter_of_millions_of_hours(cfg):
    """The source emits counters in the MILLIONS, and this ran before the flag
    that gates them."""
    absurd = _frame(episode_id="huge", ending_inventory=[9, 7],
                    hours_remaining=[9_000_000.0, 8_999_999.0])
    d = pd.concat([_frame(), absurd], ignore_index=True)

    tagged, detail = tag_dp_eligibility(d, cfg)     # must not raise

    # it is unclosed and its window plainly outruns the extract, so it is edge
    assert tagged.loc[tagged.episode_id == "huge", "edge_truncated"].all()
    assert detail["edge_truncated"]["episodes_edge_truncated"] == 1
    # and the counter is nonsense, which is a DIFFERENT flag's job
    assert (tagged.loc[tagged.episode_id == "huge",
                       "dp_ineligible_reason"] == "window_too_long").all()


def test_a_closed_episode_is_never_flagged_edge_truncated(cfg):
    """The flag has to mean 'the extract stopped', or the residue it is meant
    to isolate -- unclosed for a reason a longer extract will NOT fix -- has
    nothing left to be measured against."""
    tagged, detail = tag_dp_eligibility(_frame(), cfg)
    assert not tagged.edge_truncated.any()
    assert detail["edge_truncated"]["episodes_unclosed"] == 0
    assert detail["edge_truncated"]["share_of_unclosed_explained_by_edge"] == 0.0


def test_nothing_is_dropped(cfg):
    for over in ({"cost": 0.0}, {"cost": 12_000.0}, {"total_discount": 0.9},
                 {"hours_remaining": [500.0, 499.0]},
                 {"hours_remaining": [-242.0, -243.0]},
                 {"starting_inventory": [12, 20], "ending_inventory": [9, 0]},
                 {"ending_inventory": [9, 7]}):
        before = _frame(**over)
        after, _ = tag_dp_eligibility(before, cfg)
        assert len(after) == len(before)


def test_reasons_are_first_match_so_the_column_reads_as_a_cause(cfg):
    """A zero cost is ALSO non-priceable by the `cost >= price` test... no,
    it is not -- but it IS the more fundamental fact whenever both fire. The
    label has to be the cause, not whichever test ran last."""
    d, _ = tag_dp_eligibility(_frame(cost=0.0), cfg)
    assert (d.dp_ineligible_reason == "cost_missing").all()
    assert [n for n, _ in DP_INELIGIBLE][0] == "cost_missing"


def test_every_condition_carries_a_stated_reason():
    for name, why in DP_INELIGIBLE:
        assert len(why) > 40, f"{name} has no explanation"


def test_recovery_cannot_merge_a_negative_episode_into_its_neighbour():
    """`negative_window_recovered` rewrites the field the ids are derived from."""
    from fit.prepare_data import assign_episode_ids

    rows = ([dict(sku_id=1, fc="X", date="2026-03-01", hour_of_day=h,
                  hours_remaining=hr) for h, hr in
             [(10, -5.0), (11, -6.0), (12, -7.0)]]                  # enters negative
            + [dict(sku_id=1, fc="X", date="2026-03-01", hour_of_day=h,
                    hours_remaining=hr) for h, hr in
               [(13, 20.0), (14, 19.0)]])                           # a REAL next window
    raw = pd.DataFrame(rows)
    raw["episode_id"] = assign_episode_ids(raw)
    assert raw.episode_id.nunique() == 2, "the two windows are distinct at source"

    # recovery as the chain applies it -- the chain's own function
    from fit.prepare_data import recover_negative_windows
    d, rec = recover_negative_windows(raw, 24)
    assert rec.sum() == 3 and list(d.hours_remaining[:3]) == [23.0, 22.0, 21.0]

    # re-deriving ids from the REWRITTEN counter is what used to happen, and
    # it silently fuses the two windows
    assert assign_episode_ids(d).nunique() == 1, (
        "the collision this ordering exists to avoid no longer reproduces -- "
        "if recovery changed, re-check whether the ordering is still needed")
    # ...but the ids the pipeline carries are untouched, which is the fix
    assert d.episode_id.nunique() == 2


def test_recovery_runs_after_the_resegmentation_check(cfg, synth_flc):
    """Order, asserted on the waterfall itself rather than on a comment."""
    _, wf = load_and_filter(synth_flc, cfg)
    steps = [t[0] for t in wf]
    assert steps.index("contiguous_episodes_built") < \
        steps.index("negative_window_recovered"), (
        "negative_window_recovered mutates hours_remaining, so it must run "
        "AFTER the re-segmentation invariant is checked against the source "
        "counter -- otherwise the check grades an invented counter")


def test_every_eligible_episode_is_closed_but_not_the_reverse():
    """Closure is NECESSARY for eligibility and not SUFFICIENT."""
    from common import episodes as E

    d = _closed_and_unclosed()
    flow = E.episode_flow(d)
    assert not (flow.eligible & ~flow.closed).any(), \
        "an episode is eligible without having closed"

    # and the containment is STRICT: closure alone does not confer eligibility
    dirty = _frame(episode_id="restocked-close").assign(
        starting_inventory=[9, 3], units_sold=[6, 7], ending_inventory=[3, 0])
    f2 = E.episode_flow(dirty)
    assert f2.loc["restocked-close", "closed"]
    assert not f2.loc["restocked-close", "final_hour_clean"]
    assert not f2.loc["restocked-close", "eligible"]


# ------------------------------------------------------- who reads what

def test_population_resolves_the_config_default(cfg):
    d, _ = tag_dp_eligibility(pd.concat([_frame(), _frame(cost=0.0)
                                         .assign(episode_id="bad")]), cfg)
    assert len(population(d, cfg, "integrity")) == len(d)
    # three nested populations, widest first
    assert (len(population(d, cfg, "dp_eligible"))
            <= len(population(d, cfg, "eligible"))
            <= len(population(d, cfg, "integrity")))
    assert len(population(d, cfg, "dp_eligible")) < len(d)
    with pytest.raises(ValueError):
        population(d, cfg, "whatever")


def test_the_artifact_fits_read_the_config_and_the_dp_side_does_not():
    """The choice exists for the frozen artifacts. For the DP it is a
    precondition -- an ineligible episode has no feasible tier at all.
    And each fit stays inside its own split: bounded already -- asserted so
    they stay that way."""
    from fit import train_baseline, fit_dispersion, prior_density
    from evaluate import backtest as bt
    from evaluate import shadow

    train = inspect.getsource(train_baseline.train)
    disp = inspect.getsource(fit_dispersion.fit_dispersion)
    curves = inspect.getsource(prior_density.build_curves)
    for name, src in (("train", train), ("fit_dispersion", disp),
                      ("build_curves", curves)):
        assert "population(" in src, name

    for fn in (bt.main, shadow.run_shadow):
        src = inspect.getsource(fn)
        assert 'population(d, cfg, "dp_eligible")' in src, fn.__qualname__

    assert 'splits["train"]' in train
    assert 'split_frames(d, cfg)["calib"]' in disp
    assert 'split_frames(d, cfg)[window]' in curves, (
        "the prior must take its rows from a named split, so the fit window "
        "and the held-out window cannot silently be the same one")
    assert '"train"' in inspect.getsource(prior_density.estimate), \
        "the prior fit must be built on the TRAIN window"
    hold = inspect.getsource(prior_density.holdout_comparison)
    assert 'window = "calib"' in hold, \
        "the held-out comparison must score the calib window"
    assert 'build_curves(d, cfg, model, grid, "train")' not in hold, \
        "the held-out comparison must not score the window the prior was fitted on"


def test_the_cost_floor_is_not_a_population_choice():
    """`dp_eligible` selects rows. It must never be the thing that keeps a
    below-cost price out of the action set -- that is structural."""
    from engine import dp
    tiers, d_max = dp.feasible_tiers(1000.0, 400.0, 0.025)
    assert all(1000.0 * (1 - t) >= 400.0 - 1e-9 for t in tiers)
    assert "dp_eligible" not in inspect.getsource(dp.feasible_tiers)


def test_stock_that_genuinely_vanishes_does_not_net_away():
    """The netting must not launder a real loss into a clean episode."""
    from common.episodes import episode_flow
    d = pd.DataFrame({
        "episode_id": ["V"] * 3, "date": ["2026-03-01"] * 3,
        "hour_of_day": [10, 11, 12],
        "starting_inventory": [10, 8, 8], "units_sold": [1, 0, 3],
        "ending_inventory": [8, 8, 0]})
    flow = episode_flow(d)
    assert flow.loc["V", "vanished"] == 1
    assert flow.loc["V", "arrived"] == 0
    assert flow.loc["V", "supply"] == 10


def test_a_restocked_episode_does_reach_the_backtest(cfg):
    """It used to be excluded, and that was wrong."""
    from fit.prepare_data import population
    restocked = _frame(episode_id="R", starting_inventory=[12, 20],
                       ending_inventory=[20, 0])
    d = pd.concat([_frame(), restocked], ignore_index=True)
    tagged, detail = tag_dp_eligibility(d, cfg)

    assert tagged.dp_eligible.all(), "a restock must not gate the DP"
    assert "R" in set(population(tagged, cfg, "dp_eligible").episode_id)
    assert "R" in set(population(tagged, cfg, "eligible").episode_id)
    # reported, though -- 11 units arrived and the DP had to be told
    assert detail["restocked"]["episodes"] == 1
    assert detail["restocked"]["units"] == 11
    assert detail["restocked"]["still_dp_eligible"] == 1
    assert "restocked" not in [n for n, _ in DP_INELIGIBLE]


def test_the_flow_identity_is_what_decides_a_trustworthy_episode(cfg):
    """supply = sold + remaining, computed two ways that must agree."""
    from common.episodes import episode_flow
    cases = {
        # sold everything: clearance 1, nothing left to scrap
        "cleared": [(10, 5, 2, 3), (11, 3, 3, 0)],
        # last hour RESTOCKED: opened 27, sold 30, ended holding 26
        "last_restock": [(10, 27, 0, 27), (11, 27, 0, 27), (12, 27, 30, 26)],
    }
    frames = []
    for k, rows in cases.items():
        f = pd.DataFrame(rows, columns=["hour_of_day", "starting_inventory",
                                        "units_sold", "ending_inventory"])
        f["episode_id"] = k
        f["date"] = "2026-03-01"
        frames.append(f)
    flow = episode_flow(pd.concat(frames, ignore_index=True))

    assert flow.accounting_closes.all()
    assert (flow.supply == flow.sold + flow.scrap).all()
    assert (flow.clearance <= 1.0).all(), "clearance above 1 is never valid"
    # scrap cannot happen on a fully cleared episode
    assert (flow[flow.clearance >= 1.0].scrap == 0).all()
    # and the last-hour leftover is read off `ending`, not `starting - sold`,
    # which would have given 0 and flagged a clean episode as an anomaly
    assert flow.loc["last_restock", "leftover"] == 26
    assert flow.loc["last_restock", "supply"] == 56


def test_shrink_is_counted_as_scrap_and_gates_nothing(cfg):
    """Fuzzy episodes are no good for IL, scrap, clearance, backtest or
    shadow -- and the business still needs to see them, so they are flagged
    rather than deleted."""
    from common.episodes import scrap_units
    from fit.prepare_data import population
    # 1 unit vanishes: opened 10, sold 4, last row leaves 5, supply says 6
    rows = [(10, 10, 1, 8), (11, 8, 0, 8), (12, 8, 3, 0)]
    v = pd.DataFrame(rows, columns=["hour_of_day", "starting_inventory",
                                    "units_sold", "ending_inventory"])
    v["episode_id"] = "V"
    v["date"] = "2026-01-01"
    for col, val in (("cost", 4000.0), ("original_price", 10_000.0),
                     ("total_discount", 0.25), ("hours_remaining", 3.0)):
        v[col] = val
    v["hours_remaining"] = [3.0, 2.0, 1.0]
    v["offered_price"] = v.original_price * (1 - v.total_discount)
    v["d_max"] = 1.0 - v.cost / v.original_price

    # the unit IS counted: scrap is the last hour's leftover plus the shrink,
    # so the economics close rather than leaving a hole
    assert scrap_units(v)["V"] == 6.0        # 5 left on the shelf + 1 shrunk

    tagged, detail = tag_dp_eligibility(v, cfg)
    assert tagged.units_shrink.eq(1).all()
    # shrink gates NOTHING now: the replay applies the same per-hour
    # adjustment, so the DP sees the unit go missing exactly as production
    # would -- at the next hour, in a smaller q
    assert tagged.dp_eligible.all()
    assert detail["shrink"]["units"] == 1
    # sales and the censoring call are both sound, so it still trains the
    # frozen artifacts -- only the DP is shut out, its transition assuming
    # stock leaves solely by sale
    assert tagged.episode_eligible.all()
    assert "V" in set(population(tagged, cfg, "dp_eligible").episode_id)
    assert "V" in set(population(tagged, cfg, "eligible").episode_id)
    # and the business still gets told where the missing units are
    assert detail["unreconciled_anomalies"]["units_unaccounted"] == 1
    assert detail["unreconciled_anomalies"]["by_month"]["2026-01"]["episodes"] == 1


def test_units_restocked_is_on_the_prepared_frame(cfg):
    """The owner asked for it by name: how much arrived, per episode."""
    d = pd.concat([_frame(),
                   _frame(episode_id="R", starting_inventory=[12, 20],
                          ending_inventory=[20, 0])], ignore_index=True)
    tagged, _ = tag_dp_eligibility(d, cfg)
    got = tagged.groupby("episode_id").units_restocked.first()
    assert got["e"] == 0
    assert got["R"] == 11, "12 opened, 3 sold, ended 20 -> 11 arrived"
    assert tagged.groupby("episode_id").episode_supply.first()["R"] == 23


def test_shrink_and_restock_in_one_episode_are_both_counted(cfg):
    """The owner's real episode, and the correction that produced this test."""
    rows = [(0, 24, 12, 0, 12), (1, 23, 12, 0, 12), (2, 22, 12, 0, 12),
            (3, 21, 12, 0, 12), (4, 20, 12, 0, 12), (5, 19, 12, 0, 12),
            (6, 18, 12, 0, 12), (7, 17, 12, 0, 12), (8, 16, 12, 1, 11),
            (9, 15, 11, 0, 11), (10, 14, 11, 0, 11), (11, 13, 11, 0, 10),
            (12, 12, 10, 1, 10), (13, 11, 10, 0, 10), (14, 10, 10, 0, 10),
            (15, 9, 10, 1, 9), (16, 8, 9, 1, 8), (17, 7, 8, 0, 7),
            (18, 6, 7, 1, 7), (19, 5, 7, 0, 7), (20, 4, 7, 4, 3),
            (21, 3, 3, 2, 0)]
    d = pd.DataFrame(rows, columns=["hour_of_day", "hours_remaining",
                                    "starting_inventory", "units_sold",
                                    "ending_inventory"])
    d["episode_id"] = "W"
    d["date"] = "2026-03-01"
    d["category"] = "MEAT"
    d["cost"] = 4000.0
    d["original_price"] = 10_000.0
    d["total_discount"] = 0.25
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    d["d_max"] = 1.0 - d.cost / d.original_price

    from common.episodes import classify, scrap_units, episode_flow
    flow = episode_flow(d)
    assert flow.loc["W", "arrived"] == 2, "the restock hours were netted away"
    assert flow.loc["W", "vanished"] == 2, "the shrink hours were netted away"
    assert flow.loc["W", "supply"] == 14      # 12 opened + 2 arrived
    assert flow.loc["W", "clearance"] == pytest.approx(11 / 14)
    # supply = sold + scrap, with scrap = leftover + shrink
    assert flow.loc["W", "leftover"] == 1
    assert flow.loc["W", "scrap"] == 3          # 1 left on the shelf + 2 shrunk
    assert 14 == 11 + 3

    tagged, detail = tag_dp_eligibility(d, cfg)
    t = tagged.iloc[0]
    assert int(t.units_restocked) == 2
    assert int(t.units_shrink) == 2
    assert int(t.episode_supply) == 14
    assert int(t.episode_scrap) == 3

    # BOTH are counted, and neither gates
    assert detail["restocked"]["episodes"] == 1
    assert detail["restocked"]["units"] == 2
    assert detail["shrink"]["episodes"] == 1
    assert detail["shrink"]["units"] == 2
    assert detail["shrink_and_restock_together"]["episodes"] == 1
    assert detail["shrink_and_restock_together"]["units_arrived"] == 2
    assert detail["shrink_and_restock_together"]["units_shrunk"] == 2

    # and it is fully usable: artifacts AND backtest/shadow
    assert bool(t.dp_eligible)
    assert "W" in set(population(tagged, cfg, "dp_eligible").episode_id)
    assert "W" in set(population(tagged, cfg, "eligible").episode_id)
    # the listing closed and every unit is accounted for: 1 left on the shelf
    # and 2 shrunk, both scrap
    assert classify(d).iloc[0] == "completed"
    assert scrap_units(d).iloc[0] == 3.0
    # and the business gets told where to look
    assert detail["unreconciled_anomalies"]["units_unaccounted"] == 2


# --------------------------------------------- clearance excludes unfinished

def _closed_and_unclosed():
    """Two episodes with the write-off convention in force, so `classify_last`
    can actually read closure: A finished, B cut off mid-window."""
    rows = [("A", 10, 3, 10, 4, 6), ("A", 11, 2, 6, 3, 3), ("A", 12, 1, 3, 0, 0),
            ("B", 10, 9, 10, 1, 9), ("B", 11, 8, 9, 1, 8)]
    return episode_frame(rows, columns=[
        "episode_id", "hour_of_day", "hours_remaining",
        "starting_inventory", "units_sold", "ending_inventory"],
        date="2026-03-01", category="MEAT", cost=4000.0,
        original_price=10_000.0, total_discount=0.25)


def test_an_unfinished_episode_has_no_clearance_to_report(cfg):
    """"Sold so far" is not clearance, and averaging it in biases ONE way."""
    from common import episodes as E
    d = _closed_and_unclosed()

    assert E.classify(d)["B"] == E.NOT_CLOSED
    # B satisfies the OTHER two conditions -- the identity holds and its final
    # hour is clean -- so for a long time it passed `eligible`, and every
    # consumer had to remember to exclude it separately. Two of them forgot.
    # Closure is the third condition now, checked in one place.
    flow = E.episode_flow(d)
    assert flow.loc["B", "accounting_closes"] and flow.loc["B", "final_hour_clean"]
    assert not flow.loc["B", "closed"]
    assert not flow.loc["B", "eligible"]

    # the property, asserted on episode_flow itself now that the descriptive
    # layer that used to report it is gone: clearance is only meaningful on a
    # CLOSED episode, and never exceeds 1 by construction
    closed = flow[flow.closed]
    assert float(closed.clearance.mean()) == pytest.approx(0.70), \
        "an unfinished episode is being averaged into clearance again"
    assert int((~flow.closed).sum()) == 1
    assert float(flow.clearance.max()) <= 1.0


def test_population_refuses_a_frame_without_its_eligibility_flag(cfg):
    """The one home for the filter must fail LOUDLY: returning the whole
    frame let a stale prepared.parquet fit every artifact on the integrity
    population while each report labelled it eligible."""
    import pandas as pd
    import pytest

    from fit.prepare_data import population

    bare = pd.DataFrame({"episode_id": ["e1", "e2"], "units_sold": [1, 2]})
    assert len(population(bare, cfg, "integrity")) == 2      # integrity is all
    for which in ("eligible", "dp_eligible"):
        with pytest.raises(ValueError, match="re-run fit.prepare_data"):
            population(bare, cfg, which)

    flagged = bare.assign(episode_eligible=[True, False],
                          dp_eligible=[True, False])
    assert len(population(flagged, cfg)) == 1
    assert len(population(flagged, cfg, "dp_eligible")) == 1


# ------------------------------------------------- episode ids and windows

def _window(sku, fc, start, hours, base_hr=None):
    """One selling window as hourly rows, counting hours_remaining down."""
    hr = hours - 1 if base_hr is None else base_hr
    ts = pd.date_range(start, periods=hours, freq="h")
    return episode_frame(sku_id=sku, fc=fc, date=ts.normalize(),
                         hour_of_day=ts.hour,
                         hours_remaining=[hr - i for i in range(hours)])


def test_episode_spans_midnight_as_one_window():
    """FLC windows commonly run past midnight -- 36 hours is common. A
    date-keyed episode would split one economic window into three, resetting
    the monotonicity anchor and charging carried inventory to scrap twice."""
    from fit.prepare_data import assign_episode_ids

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
    from fit.prepare_data import assign_episode_ids

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


def test_split_assigns_straddling_episode_by_start_date():
    """A window that starts in train and ends in calib belongs wholly to
    train -- otherwise the boundary runs through the middle of an episode."""
    from fit.prepare_data import split_frames

    cfg = {"data": {"split": {
        "train_start": "2026-03-01", "train_end": "2026-03-02",
        "calib_start": "2026-03-03", "calib_end": "2026-03-04",
        "test_start": "2026-03-05", "test_end": "2026-03-06"}}}
    d = _window(1, "FC1", "2026-03-02 10:00", 36)     # crosses into 03-03/04
    d["episode_id"] = "1|FC1|2026-03-02T10"
    frames = split_frames(d, cfg)
    assert len(frames["train"]) == len(d)
    assert len(frames["calib"]) == 0 and len(frames["test"]) == 0


def test_a_new_window_is_not_mistaken_for_a_gap():
    """The counter is what tells them apart, and it must."""
    from fit.prepare_data import gap_split_windows, assign_episode_ids

    def frame(rows):
        d = episode_frame(rows, columns=["hour_of_day", "hours_remaining"],
                          date="2026-03-01", sku_id="S", fc="F")
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


# ------------------------------------------ the production worked example

def _observed_episode():
    """The worked example from the production extract: a MEAT episode that
    closes with stock left, while ending_inventory reports zero."""
    return episode_frame(
        episode_id="m", date=pd.Timestamp("2026-03-01").date(),
        hour_of_day=list(range(11, 21)), hours_remaining=list(range(9, -1, -1)),
        starting_inventory=[8, 8, 8, 8, 8, 5, 5, 5, 5, 4],
        units_sold=[0, 0, 0, 0, 3, 0, 0, 0, 1, 3],
        ending_inventory=[8, 8, 8, 8, 5, 5, 5, 5, 4, 0])   # zeroed at the close


def test_true_leftover_on_the_production_worked_example():
    from common import episodes

    d = _observed_episode()
    last = episodes.last_rows(d)
    assert int(last.ending_inventory.iloc[0]) == 0     # what the source says

    # 4 units enter the final hour, 3 sell -> 1 is written off, not zero
    left = episodes.leftover_units(last.starting_inventory, last.units_sold)
    assert float(left.iloc[0]) == 1.0
    assert int(d.starting_inventory.iloc[0]) == 8 and int(d.units_sold.sum()) == 7

    # the final row carries the closure sentinel (ending_inventory zeroed with
    # a unit still on hand), so the listing ended and that unit IS scrap
    assert episodes.write_off_convention(episodes.episode_flow(d))
    kind = episodes.classify(d)
    assert kind.iloc[0] == episodes.COMPLETED
    assert float(episodes.scrap_units(d).iloc[0]) == 1.0

    # and the counter is at zero here only because the fixture makes it so;
    # on production it is still positive on ~99.9% of final rows, which is
    # why scrap must not be keyed to it
    assert int(episodes.last_rows(d).hours_remaining.iloc[0]) == 0


def test_a_restock_is_detected_from_the_source_convention():
    """The detector, and the fact that its output only ever sets a flag."""
    from common.episodes import episode_flow

    clean = _observed_episode()
    assert episode_flow(clean).arrived.eq(0).all()

    # 4 units arrive during hour 16, which opened with 5 and sold none. The
    # source reports the FINAL count, so ending goes to 9 and hour 17 opens
    # with 9 -- the chain stays continuous, which is what makes this a
    # restock rather than a break.
    restocked = clean.copy()
    h16 = restocked.hour_of_day == 16
    restocked.loc[h16, "ending_inventory"] += 4
    restocked.loc[restocked.hour_of_day > 16, "starting_inventory"] += 4
    restocked.loc[restocked.hour_of_day.between(17, 19), "ending_inventory"] += 4
    assert episode_flow(restocked).loc["m", "arrived"] == 4

    # the plainest restock of all: an hour selling MORE than it opened with.
    # `sold >= starting` is not an impossible quantity, it is stock arriving.
    oversell = clean.copy()
    h15 = oversell.hour_of_day == 15
    oversell.loc[h15, "units_sold"] = 12          # opened with 8
    oversell.loc[h15, "ending_inventory"] = 5     # so 9 arrived
    assert episode_flow(oversell).loc["m", "arrived"] == 9

    # selling stock down is never a restock, however steep the drop
    steep = clean.copy()
    steep.loc[steep.hour_of_day == 15, "units_sold"] = 8
    steep.loc[steep.hour_of_day == 15, "ending_inventory"] = 0
    steep.loc[steep.hour_of_day > 15, "starting_inventory"] = 0
    steep.loc[steep.hour_of_day > 15, "ending_inventory"] = 0
    steep.loc[steep.hour_of_day > 15, "units_sold"] = 0
    assert episode_flow(steep).arrived.eq(0).all()


# ------------------------------------------------ the waterfall's own basis

def test_the_raw_waterfall_row_counts_rows_episodes_and_cogs_on_one_frame(cfg, synth_flc):
    """`raw` used to count rows pre-dedup but episodes and COGS post-dedup,
    so the first row of the artifact was two frames wearing one label."""
    path = synth_flc
    _, wf = load_and_filter(path, cfg)
    raw, dedup = wf[0], wf[2]          # wf[1] is the null-key integrity drop
    assert raw[0] == "raw" and dedup[0] == "duplicate_hour_rows_dropped"
    assert raw[1] == len(pd.read_parquet(path)), "raw rows = the file's rows"
    # duplicated hours collide into extra ids, so the raw frame has AT LEAST
    # as many episodes as the deduplicated one -- never exactly the same by
    # construction, which is what counting them on the dedup frame gave
    assert raw[2] >= dedup[2] and raw[3] >= dedup[3] > 0
    assert raw[1] >= dedup[1]


def test_population_gate_rows_record_no_removed_episodes(cfg, synth_flc):
    """`eligible` and `dp_eligible` drop nothing, so the examples collector
    must not list their flagged episodes as removals -- they are still in
    the frame."""
    examples = {}
    _, wf = load_and_filter(synth_flc,
                            cfg, examples=examples)
    labels = [t[0] for t in wf]
    assert labels[-2:] == ["eligible", "dp_eligible"]
    assert "eligible" not in examples and "dp_eligible" not in examples
    assert "raw" not in examples and "duplicate_hour_rows_dropped" not in examples
    # gate rows carry their detail dict like any other detailed stage
    assert isinstance(wf[-1][4], dict) and isinstance(wf[-2][4], dict)


# ------------------------------------------ the rate features and the index

def _hourly_rows(days=12, skus=(1, 2)):
    rows = []
    for sku in skus:
        for day in range(days):
            date = str((pd.Timestamp("2026-03-01")
                        + pd.Timedelta(days=day)).date())
            for h in (10, 11, 12):
                rows.append(dict(sku_id=sku, fc="F", date=date, hour_of_day=h,
                                 total_discount=0.25 if h < 12 else 0.40,
                                 d_ref=0.25, starting_inventory=5,
                                 units_sold=(sku + day + h) % 3,
                                 episode_id=f"{sku}|F|{day}"))
    return pd.DataFrame(rows)


def test_ref_rate_features_do_not_depend_on_the_frames_index_labels(cfg):
    """The anchor mask was built on the caller's index and reused after a
    merge reset it, so on a frame whose index had gaps (every filtered frame)
    the mask aligned by LABEL and prior_episode_ref_sales_rate was built on
    the wrong rows. Same rows, any labels -> the same features."""
    from fit.prepare_data import add_ref_rate_features

    d = _hourly_rows()
    gappy = d.copy()
    gappy.index = gappy.index * 3 + 7               # sparse, offset labels
    cols = ["sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate"]
    contiguous = add_ref_rate_features(d, cfg)[cols].reset_index(drop=True)
    shuffled = add_ref_rate_features(gappy, cfg)[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(contiguous, shuffled)
    # and the feature is real: the second episode of a SKU sees the first's
    # anchor rate, the first sees nothing
    out = add_ref_rate_features(gappy, cfg)
    first = out[out.episode_id == "1|F|0"].prior_episode_ref_sales_rate
    second = out[out.episode_id == "1|F|1"].prior_episode_ref_sales_rate
    assert first.isna().all()
    assert second.notna().all()
    anchor_first = out[(out.episode_id == "1|F|0") & (out.hour_of_day < 12)]
    assert second.iloc[0] == pytest.approx(anchor_first.units_sold.mean())


def test_pre_launch_stops_at_the_gate_window(cfg):
    from fit.prepare_data import pre_launch
    end = cfg["data"]["split"]["test_end"]
    d = pd.DataFrame({
        "episode_id": ["before", "before", "straddles", "straddles",
                       "holdout", "after_holdout"],
        "date": [end, end, end, cfg["data"]["holdout"]["start"],
                 cfg["data"]["holdout"]["start"], "2026-09-01"],
        "hour_of_day": [22, 23, 23, 0, 9, 9],
    })
    kept = set(pre_launch(d, cfg).episode_id)
    assert kept == {"before", "straddles"}, \
        "an episode that OPENED before the gate window closed belongs to " \
        "pre-launch whole; one that opened after does not belong at all"


def test_cogs_at_risk_counts_supply_not_opening_stock():
    """A window that opens with 3 and takes 10 mid-flight has 13 units of
    cost at risk; counting 3 understates every restocked episode."""
    from fit.prepare_data import cogs_at_risk

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
    from fit.prepare_data import cogs_at_risk, episode_cogs

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


def test_a_precomputed_flow_and_cogs_table_change_nothing_in_the_flags(cfg):
    """load_and_filter hands tag_dp_eligibility the flow it already built at
    episode_universe (and the COGS table); the result must be the one a bare
    call computes for itself."""
    from common.episodes import episode_flow
    from fit.prepare_data import episode_cogs

    d = pd.concat([_frame(),
                   _frame(episode_id="R", starting_inventory=[12, 20],
                          ending_inventory=[20, 0]),
                   _frame(episode_id="u", ending_inventory=[9, 7]),
                   _frame(episode_id="bad", cost=0.0)], ignore_index=True)
    bare, bare_detail = tag_dp_eligibility(d, cfg)
    fed, fed_detail = tag_dp_eligibility(d, cfg, flow=episode_flow(d),
                                         per_episode_cogs=episode_cogs(d))
    pd.testing.assert_frame_equal(bare, fed)
    assert bare_detail == fed_detail
    # and the sentinel diagnostic the design names is on the manifest
    assert fed_detail["edge_truncated"]["write_off_convention_in_force"] is True
    none = _frame(episode_id="open", ending_inventory=[9, 7])
    _, no_sentinel = tag_dp_eligibility(none, cfg)
    assert no_sentinel["edge_truncated"]["write_off_convention_in_force"] is False


def test_the_universe_stage_reports_where_censoring_was_decided(cfg, synth_flc):
    """`censoring_off_last_row` was a function nothing in the pipeline
    called; the manifest now carries it (design 12a: censoring is decided
    on the LAST row only, and a row that empties the shelf mid-episode says
    the feed carried on past zero)."""
    _, wf = load_and_filter(synth_flc, cfg)
    universe = next(t for t in wf if t[0] == "episode_universe")
    cen = universe[4]["censoring"]
    assert {"rows_shelf_emptied_mid_episode",
            "rows_with_zero_starting_inventory"} <= set(cen)
    assert cen["rows_shelf_emptied_mid_episode"] == 0


def test_the_ref_rate_window_is_the_thirty_prior_days(cfg):
    """Design 5.4 says [t-30, t-1]. The right-closed rolling window is
    [t-29, t], and subtracting the day back out left 29 prior days; the
    left-closed window is the design's, so day t-30 counts and day t-31
    does not."""
    from fit.prepare_data import add_ref_rate_features

    days = pd.date_range("2026-03-01", periods=40, freq="D")
    rows = [dict(sku_id=1, fc="F", date=str(day.date()), hour_of_day=10,
                 total_discount=0.25, d_ref=0.25, starting_inventory=5,
                 units_sold=(1 if i == 0 else 0), episode_id=f"1|F|{i}")
            for i, day in enumerate(days)]
    out = add_ref_rate_features(pd.DataFrame(rows), cfg)
    rate = out.set_index("episode_id").sku_ref_sales_rate_30d
    W = cfg["baseline_model"]["ref_rate_window_days"]
    # day W still sees day 0 (W prior days); day W+1 no longer does
    assert rate[f"1|F|{W}"] == pytest.approx(1.0 / W)
    assert rate[f"1|F|{W + 1}"] == 0.0
    # and the day itself is excluded: day 0's own sale is not in its rate
    assert np.isnan(rate["1|F|0"])
    assert rate["1|F|1"] == pytest.approx(1.0)


def test_a_row_with_no_episode_key_is_dropped_and_counted(cfg, synth_flc, tmp_path):
    """A null sku_id or fc fell out of every groupby in assign_episode_ids,
    so such rows collapsed into one NaN "episode" that later stages read as
    a window. INTEGRITY (rule 14): a DROP with its own waterfall row."""
    from fit.prepare_data import EPISODE_KEY, null_key_rows

    raw = pd.read_parquet(synth_flc)
    dirty = raw.copy()
    dirty.loc[dirty.index[:3], "skuseq"] = None
    dirty.loc[dirty.index[5:7], "fc"] = None
    dirty.loc[dirty.index[5], "hour"] = None
    # a null window counter mid-episode: NaN != -1 opened a NEW episode on
    # that row, a one-row window that closed on its own zero and read
    # DP-eligible (the owner's extract; the pilot simulator's templates).
    # The WHOLE clock-contiguous run drops, not the row (rule 15)
    dirty.loc[dirty.index[40], "flc_window"] = None
    run = _clock_run_of(raw, raw.index[40])
    assert len(run) > 1, "pick a row inside a multi-hour window"
    path = tmp_path / "dirty.parquet"
    dirty.to_parquet(path, index=False)

    d, wf = load_and_filter(str(path), cfg)
    labels = [t[0] for t in wf]
    assert labels[:3] == ["raw", "null_key_rows_dropped",
                          "duplicate_hour_rows_dropped"]
    stage = wf[1]
    dropped = 5 + len(run)
    assert stage[1] == len(raw) - dropped and stage[4]["rows_dropped"] == dropped
    assert stage[4]["nulls_by_column"] == {"sku_id": 3, "fc": 2, "date": 0,
                                           "hour_of_day": 1}
    assert stage[4]["null_counter_rows"] == len(run)
    assert stage[4]["null_counter_windows_dropped"] == 1
    assert not d.episode_id.isna().any()
    assert not d.episode_id.astype(str).str.contains("nan").any()
    assert not d.hours_remaining.isna().any()
    # none of the run's other hours survives as a fragment
    keys = set(zip(run.skuseq, run.fc, run.date.astype(str), run.hour))
    assert not any((r.sku_id, r.fc, str(r.date), r.hour_of_day) in keys
                   for r in d.itertuples())
    assert EPISODE_KEY == ("sku_id", "fc", "date", "hour_of_day")
    # an extract with NO dirt loses nothing to this stage (the shared
    # fixture carries the generator's dirt, null counters included)
    from tools import make_dummy_flc as gen
    start, days = gen.span_covering_splits(cfg)
    spotless, _ = gen.generate(40, days, "randomized", 3, dirty_frac=0.0, start=start)
    spotless_path = tmp_path / "spotless.parquet"
    pq.write_table(pa.Table.from_pandas(spotless, schema=gen.SCHEMA,
                                        preserve_index=False), str(spotless_path))
    _, clean = load_and_filter(str(spotless_path), cfg)
    assert clean[1][1] == clean[0][1] and clean[1][4]["rows_dropped"] == 0
    mask, by_col = null_key_rows(pd.DataFrame({"sku_id": [1, None], "fc": ["F", "F"],
                                               "date": ["d", "d"], "hour_of_day": [1, 1]}))
    assert list(mask) == [False, True] and by_col["sku_id"] == 1


def _clock_run_of(raw, idx):
    """The rows of `raw` in the same sku x fc clock-contiguous run as `idx`."""
    r = raw.loc[idx]
    g = raw[(raw.skuseq == r.skuseq) & (raw.fc == r.fc)].copy()
    ts = pd.to_datetime(g.date.astype(str)) + pd.to_timedelta(g.hour, unit="h")
    g = g.assign(_ts=ts).sort_values("_ts")
    breaks = g._ts.diff().dt.total_seconds().div(3600).ne(1.0).cumsum()
    return g[breaks == breaks[idx]].drop(columns="_ts")


def test_a_null_counter_drops_its_whole_window_not_a_fragment():
    """Rule 15 at the null-counter stage: the run the null sits in goes
    whole, and a back-to-back neighbour with its own clock break stays."""
    from fit.prepare_data import null_counter_windows

    df = pd.DataFrame({
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
