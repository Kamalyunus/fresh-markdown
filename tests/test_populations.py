"""Two populations, and which consumer is entitled to which."""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

from bootstrap.prepare_data import (DP_INELIGIBLE, load_and_filter, population,
                                    tag_dp_eligibility)
from conftest import ROOT, episode_frame


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
    from bootstrap.prepare_data import population
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
    from bootstrap.prepare_data import assign_episode_ids

    rows = ([dict(sku_id=1, fc="X", date="2026-03-01", hour_of_day=h,
                  hours_remaining=hr) for h, hr in
             [(10, -5.0), (11, -6.0), (12, -7.0)]]                  # enters negative
            + [dict(sku_id=1, fc="X", date="2026-03-01", hour_of_day=h,
                    hours_remaining=hr) for h, hr in
               [(13, 20.0), (14, 19.0)]])                           # a REAL next window
    raw = pd.DataFrame(rows)
    raw["episode_id"] = assign_episode_ids(raw)
    assert raw.episode_id.nunique() == 2, "the two windows are distinct at source"

    # apply recovery exactly as the chain does
    cap, d = 24, raw.copy()
    entry = d.groupby("episode_id")["hours_remaining"].transform("first")
    length = d.groupby("episode_id")["hours_remaining"].transform("size")
    rec = (entry < 0) & (length <= cap)
    d.loc[rec, "hours_remaining"] = (cap - 1) - d.groupby("episode_id").cumcount()[rec]

    # re-deriving ids from the REWRITTEN counter is what used to happen, and
    # it silently fuses the two windows
    assert assign_episode_ids(d).nunique() == 1, (
        "the collision this ordering exists to avoid no longer reproduces -- "
        "if recovery changed, re-check whether the ordering is still needed")
    # ...but the ids the pipeline carries are untouched, which is the fix
    assert d.episode_id.nunique() == 2


def test_recovery_runs_after_the_resegmentation_check(cfg):
    """Order, asserted on the waterfall itself rather than on a comment."""
    _, wf = load_and_filter(os.path.join(ROOT, "data", "flc_synth.parquet"), cfg)
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
    from bootstrap import train_baseline, fit_dispersion, prior_density
    from backtest import __main__ as bt
    from pipeline import shadow

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
    from pricing import dp
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
    from bootstrap.prepare_data import population
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
    from bootstrap.prepare_data import population
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


def test_the_learning_yield_reports_the_terms_behind_it():
    """A disappointing yield has two causes with OPPOSITE remedies -- too few
    forced decisions, or forced prices sitting too close to the reference --
    and the per-episode aggregate cannot tell them apart. So the report
    carries the decomposition, and the identity that makes it checkable:"""
    import inspect
    from pipeline import shadow

    src = inspect.getsource(shadow.run_shadow)
    for field in ("forced_decisions", "information_per_forced_decision",
                  "mean_abs_log_price_ratio_forced",
                  "mean_discount_gap_from_reference_forced_pp",
                  "mean_mu_on_forced_hours"):
        assert field in src, f"{field} missing from the learning yield"

    # the components are accumulated on the SAME branch as the information,
    # or they would describe a different set of hours than they explain
    one = inspect.getsource(shadow._shadow_one)
    body = one[one.index('if evt["is_exploration"]'):]
    for comp in ('out["abs_log_ratio"] +=', 'out["forced_mu"] +=',
                 'out["forced_discount_gap"] +='):
        assert comp in body, f"{comp} is not accumulated on forced hours"
    # and the gap is measured against the REFERENCE, the same baseline the
    # log ratio uses -- not against the optimum or the anchor
    gap = body[body.index('out["forced_discount_gap"] +='):]
    assert 'reference_discount' in gap[:220], \
        "the discount gap must be measured against the reference discount"


def test_step_sensitivity_prices_the_cap_on_real_episodes(cfg):
    """`learning.max_mean_step` is justified by measurement, not judgment:
    the sweep re-solves the DP arm at eps +- step and reports what moves.
    Far below the deepening bar a step must change NOTHING -- that measured
    insensitivity is what makes a wrong-direction update cheap (design
    5.11). The block must also be structurally sound: shares in [0, 1],
    finite IL on both sides, crossers a subset of the sample."""
    from backtest.replay import _episode_frame, step_sensitivity

    def episode(eid, eps):
        g = pd.DataFrame({
            "episode_id": [eid] * 4,
            "date": ["2026-05-01"] * 4, "hour_of_day": [9, 10, 11, 12],
            "total_discount": [0.25, 0.25, 0.30, 0.30],
            "original_price": [10_000.0] * 4, "cost": [4000.0] * 4,
            "d_ref": [0.25] * 4, "starting_inventory": [6, 5, 4, 3],
            "units_sold": [1, 1, 1, 1], "mu_ref_hat": [1.5] * 4,
            "r": [3.0] * 4, "eps": [eps] * 4, "is_observed": [True] * 4,
            "sku_id": [7] * 4, "fc": ["FC1"] * 4, "category": ["FRUIT"] * 4,
        })
        g["ending_inventory"] = g.starting_inventory - g.units_sold
        return _episode_frame(g)

    # cost ratio 0.4, d_ref 0.25 -> deepening bar (1-d)/(gamma-d) ~ 5, so
    # |eps| = 1.0 sits far below it and a 0.15 step is deep inside the
    # insensitive region
    frames = [episode(f"e{i}", -1.0) for i in range(4)]
    out = step_sensitivity(frames, cfg, sample=4)

    assert out["episodes_swept"] == 4
    assert out["step"] == cfg["learning"]["max_mean_step"]
    for label in ("deeper_belief", "shallower_belief"):
        b = out[label]
        assert 0.0 <= b["share_prices_changed"] <= 1.0
        assert np.isfinite(b["il_base"]) and np.isfinite(b["il_shifted"])
        assert b["crossers"] <= out["episodes_swept"]
        assert b["crossers_prices_changed"] <= max(b["crossers"], 0)
    # the load-bearing claim: far below the bar, a bounded step is free
    assert out["deeper_belief"]["share_prices_changed"] == 0.0
    assert out["deeper_belief"]["il_delta"] == pytest.approx(0.0, abs=1e-6)


def test_simulated_arms_absorb_only_the_shrink_their_shelf_held(cfg):
    """A negative adjustment can only take what the SIMULATED shelf still
    holds -- units an arm already sold cannot also shrink. Charging the full
    observed shrink anyway drove the supply residual negative by exactly the
    clipped amount (the workbook's 'episode and hourly sheets disagree')."""
    from backtest.replay import _episode_frame, _replay_one

    g = pd.DataFrame({
        "episode_id": ["e"] * 3,
        "date": ["2026-05-01"] * 3, "hour_of_day": [9, 10, 11],
        "total_discount": [0.30] * 3,
        "original_price": [10_000.0] * 3, "cost": [4000.0] * 3,
        "d_ref": [0.25] * 3,
        # observed world sells nothing, then 2 units shrink mid-window
        "starting_inventory": [3, 3, 1], "units_sold": [0, 0, 0],
        "ending_inventory": [3, 1, 1],
        "mu_ref_hat": [2.5] * 3, "r": [3.0] * 3, "eps": [-1.5] * 3,
        "is_observed": [True] * 3, "sku_id": [7] * 3, "fc": ["FC1"] * 3,
        "category": ["FRUIT"] * 3,
    })
    e = _episode_frame(g)
    assert e["shrink"] == 2
    row, _ = _replay_one(e, cfg)
    # the identity holds for ALL THREE arms, exactly
    for arm in ("actual", "legacy_model", "dp"):
        assert row[f"{arm}_supply_residual"] == pytest.approx(0.0, abs=1e-9)
    # the sim arms sold most of the stock before the shrink hour, so they
    # absorb strictly less than the observed 2 units -- and are not charged
    # scrap for units they sold
    for arm in ("legacy_model", "dp"):
        assert 0.0 <= row[f"{arm}_shrink_applied"] < 2.0
        assert row[f"{arm}_scrap_units"] == pytest.approx(
            row[f"{arm}_leftover_units"] + row[f"{arm}_shrink_applied"])


def test_population_refuses_a_frame_without_its_eligibility_flag(cfg):
    """The one home for the filter must fail LOUDLY: returning the whole
    frame let a stale prepared.parquet fit every artifact on the integrity
    population while each report labelled it eligible."""
    import pandas as pd
    import pytest

    from bootstrap.prepare_data import population

    bare = pd.DataFrame({"episode_id": ["e1", "e2"], "units_sold": [1, 2]})
    assert len(population(bare, cfg, "integrity")) == 2      # integrity is all
    for which in ("eligible", "dp_eligible"):
        with pytest.raises(ValueError, match="re-run bootstrap.prepare_data"):
            population(bare, cfg, which)

    flagged = bare.assign(episode_eligible=[True, False],
                          dp_eligible=[True, False])
    assert len(population(flagged, cfg)) == 1
    assert len(population(flagged, cfg, "dp_eligible")) == 1


def test_the_prior_entry_row_is_the_first_HOUR_not_the_lowest_clock_time(cfg):
    """Rule 7: the prior identifies elasticity on ENTRY rows only. Sorting by
    hour_of_day alone picks the 00:00 row of an episode that opened at 22:00
    the night before -- a within-episode, post-price-path row, which is the
    confound the rule exists to exclude. Production windows cross midnight
    routinely (design 12a); the fixture has none, so nothing caught it."""
    import pandas as pd

    from bootstrap.prior_density import scored_rows

    ep = pd.DataFrame([
        # opens 22:00 at the anchor, deepens after midnight
        {"date": "2026-07-01", "hour_of_day": 22, "total_discount": 0.10},
        {"date": "2026-07-01", "hour_of_day": 23, "total_discount": 0.10},
        {"date": "2026-07-02", "hour_of_day": 0, "total_discount": 0.25},
        {"date": "2026-07-02", "hour_of_day": 3, "total_discount": 0.25},
    ]).assign(episode_id="EP-MIDNIGHT", starting_inventory=5, units_sold=1,
              ending_inventory=4, category="VEG", subcategory="LEAFY",
              original_price=1e4, cost=4e3, fc="F1", d_ref=0.30)

    row = scored_rows(ep)
    assert len(row) == 1
    assert (row.date.iloc[0], int(row.hour_of_day.iloc[0])) == ("2026-07-01", 22)
    assert row.total_discount.iloc[0] == 0.10      # the entry price, not 0.25


def test_the_priors_deflation_can_actually_engage(cfg):
    """scored_rows returns ONE row per episode, so an episode-grouped ICC was
    empty by construction: rho 0, deff exactly 1.0 for every category, and
    design 5.6's deflation could never do anything. Clustered on the unit that
    recurs -- SKU x FC -- it engages when correlation is present."""
    import numpy as np
    import pandas as pd

    from bootstrap.prior_density import deflation_deff

    class _Model:                     # mu_ref is the deflation's baseline only
        @staticmethod
        def predict_mu_ref(rows):
            return np.full(len(rows), 5.0)

    rng = np.random.default_rng(0)
    n_units, per_unit = 200, cfg["assurance"]["rho_min_hours_per_episode"] + 1
    rows = []
    for u in range(n_units):
        shared = rng.normal(0, 2.0)            # a persistent per-unit level
        for _ in range(per_unit):
            rows.append({"sku_id": f"S{u}", "fc": "F1",
                         "units_sold": 5.0 + shared + rng.normal(0, 0.5)})
    frame = pd.DataFrame(rows).assign(episode_id=lambda f: range(len(f)))

    deff, rho, m = deflation_deff(frame, _Model(), cfg)
    assert m == pytest.approx(per_unit)
    assert rho > 0.5                    # the shared level dominates
    assert deff > 1.0                   # and the deflation is real

    # one row per unit: no clustering to measure, deff falls back to 1.0
    solo = frame.drop_duplicates("sku_id").copy()
    assert deflation_deff(solo, _Model(), cfg)[0] == pytest.approx(1.0)


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
    from bootstrap.prepare_data import assign_episode_ids

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
    from bootstrap.prepare_data import assign_episode_ids

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
    from bootstrap.prepare_data import split_frames

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
    from bootstrap.prepare_data import gap_split_windows, assign_episode_ids

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
    assert episodes.write_off_convention(last)
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

def test_the_raw_waterfall_row_counts_rows_episodes_and_cogs_on_one_frame(cfg):
    """`raw` used to count rows pre-dedup but episodes and COGS post-dedup,
    so the first row of the artifact was two frames wearing one label."""
    path = os.path.join(ROOT, "data", "flc_synth.parquet")
    _, wf = load_and_filter(path, cfg)
    raw, dedup = wf[0], wf[1]
    assert raw[0] == "raw" and dedup[0] == "duplicate_hour_rows_dropped"
    assert raw[1] == len(pd.read_parquet(path)), "raw rows = the file's rows"
    # duplicated hours collide into extra ids, so the raw frame has AT LEAST
    # as many episodes as the deduplicated one -- never exactly the same by
    # construction, which is what counting them on the dedup frame gave
    assert raw[2] >= dedup[2] and raw[3] >= dedup[3] > 0
    assert raw[1] >= dedup[1]


def test_population_gate_rows_record_no_removed_episodes(cfg):
    """`eligible` and `dp_eligible` drop nothing, so the examples collector
    must not list their flagged episodes as removals -- they are still in
    the frame."""
    examples = {}
    _, wf = load_and_filter(os.path.join(ROOT, "data", "flc_synth.parquet"),
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
    from bootstrap.prepare_data import add_ref_rate_features

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


def test_within_episode_moves_are_counted_on_the_arms_own_path(cfg):
    """`pct_dp_deepened` compares episode MEANS against legacy and says
    nothing about whether the agent moves after entry. `intra_episode_moves`
    counts steps on the DP arm's own path -- a fresh solve every hour, so a
    high-cost shelf (low deepening bar) steps where a mid-cost one holds."""
    import numpy as np
    from backtest.replay import intra_episode_steps, intra_episode_moves, _episode_frame, _replay_one

    # a step is a deepening between consecutive priced hours; empty-shelf
    # hours (None) are skipped, and a flat path has none
    assert intra_episode_steps((0.10, 0.10, 0.15, None, 0.15, 0.20)) == 2
    assert intra_episode_steps((0.25,) * 6) == 0

    def episode(eid, cost):
        g = pd.DataFrame({
            "episode_id": [eid] * 6, "date": ["2026-05-01"] * 6,
            "hour_of_day": [9, 10, 11, 12, 13, 14],
            "total_discount": [0.25] * 6, "original_price": [10_000.0] * 6,
            "cost": [cost] * 6, "d_ref": [0.25] * 6,
            "starting_inventory": [12, 12, 12, 12, 12, 12],
            "units_sold": [0] * 6, "mu_ref_hat": [0.4] * 6,       # slow shelf
            "r": [3.0] * 6, "eps": [-2.0] * 6, "is_observed": [True] * 6,
            "sku_id": [7] * 6, "fc": ["FC1"] * 6, "category": ["FRUIT"] * 6,
        })
        g["ending_inventory"] = g.starting_inventory - g.units_sold
        return _episode_frame(g)

    rows = [_replay_one(episode(f"e{i}", cost), cfg)[0]
            for i, cost in enumerate((3000.0, 5000.0, 7000.0, 7500.0))]
    ep = pd.DataFrame(rows)
    assert {"dp_steps", "legacy_model_steps"} <= set(ep.columns)
    out = intra_episode_moves(ep, cfg)
    bands = out["by_cost_ratio_band"]
    assert set(bands) == {"cost_ratio<0.4", "0.4<=cost_ratio<0.6", "cost_ratio>=0.6"}
    # the summary is arithmetic over the rows it was given
    assert out["overall"]["episodes"] == 4
    assert out["overall"]["mean_steps_per_episode"] == pytest.approx(ep.dp_steps.mean(), abs=1e-3)
    assert out["overall"]["share_episodes_with_a_step"] == pytest.approx((ep.dp_steps > 0).mean(), abs=1e-4)
    # the flat legacy schedule never steps; the deepening bar is read per band
    assert out["overall"]["legacy_share_episodes_with_a_step"] == 0.0
    assert bands["cost_ratio>=0.6"]["share_episodes_eps_above_threshold"] == 1.0
    assert bands["0.4<=cost_ratio<0.6"]["share_episodes_eps_above_threshold"] == 0.0
    # and the high-cost shelves, above the bar, step at least as often as the
    # mid-cost ones below it (a fresh solve every hour, not a pinned price)
    assert bands["cost_ratio>=0.6"]["mean_steps_per_episode"] >= \
        bands["0.4<=cost_ratio<0.6"]["mean_steps_per_episode"]
