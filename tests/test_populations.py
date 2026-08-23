"""Two populations, and which consumer is entitled to which.

The distinction this file defends: an INTEGRITY defect is a row that cannot
be believed; everything else is a row that is perfectly believable and merely
hard to PRICE. Conflating them cost most of the COGS in the extract -- every
frozen artifact was fit on the priceable subset, including the elasticity
prior, for which below-cost hours are the widest price variation the data
contains.

And it made gate 1 undecidable. `share_non_explorable` counts episodes whose
cost floor leaves too few tiers to explore against, and the chain deleted
exactly those before `m1` looked, so the gate read 0.0 and could not fail.

What gates `dp_eligible` is now only what the SOLVER cannot do -- no feasible
tier (`cost_missing`, `non_priceable`) and no trustable horizon
(`negative_window`, `window_too_long`) -- plus `final_hour_restock`, which
makes the CLOSE ambiguous and so gates `eligible` too.

Nothing about the inventory PATH gates any more. A restock or a shrink used to,
on the grounds that the DP's transition assumes one pool draining
monotonically. That confused the solve with the replay: within one solve the DP
does assume it, and must, because production cannot see a future delivery
either -- but the replay re-solves hourly against the stock on hand and applies
the episode's own per-hour adjustment, so the DP learns of an arrival at the
next hour exactly as it does live.

`below_cost_hours`, `edge_truncated`, `restocked` and `shrink` are all reported
and gate nothing.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from bootstrap.prepare_data import (DP_INELIGIBLE, population,
                                    tag_dp_eligibility)
from common.config import load_config


@pytest.fixture(scope="module")
def cfg():
    import os
    return load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))


def _frame(**over):
    """One clean two-hour episode; keyword overrides break it one way.

    Per-hour lists (`hours_remaining`, `starting_inventory`, `units_sold`,
    `ending_inventory`) are laid down as given; scalars are broadcast. The
    inventory chain is closed by the write-off sentinel on the final row --
    `ending_inventory == 0` while stock remained -- so `classify_last` reads
    the episode as COMPLETED and it is not swept up by `edge_truncated`.
    """
    per_hour = ("hours_remaining", "starting_inventory", "units_sold",
                "ending_inventory")
    base = dict(episode_id="e", cost=4000.0, original_price=10_000.0,
                total_discount=0.25, date="2026-01-01", hour_of_day=[10, 11],
                starting_inventory=[12, 9], units_sold=[3, 2],
                ending_inventory=[9, 0], hours_remaining=[10.0, 9.0])
    base.update(over)
    cols = {k: base.pop(k) for k in per_hour + ("hour_of_day",) if k in base}
    d = pd.DataFrame({k: [v, v] for k, v in base.items()})
    for k, v in cols.items():
        d[k] = v
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
    history, not a defect in it.

    The backtest's DP arm is self-anchored and never sees the legacy price.
    Shadow uses it as the anchor and therefore refuses every hour from the
    crossing onward -- which is the cost floor working, and the hours BEFORE
    the crossing are good decisions the old chain deleted with the episode.
    """
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
    """This reverses an earlier decision in this file, deliberately.

    The old argument: an unclosed episode's OBSERVED hours are ordinary
    priced demand, `scrap_units` already returns NaN, and these are the
    LARGEST episodes in the extract -- so gating buys nothing and costs the
    demand fit its best-observed windows.

    True, and not enough. TWO consumers met one and silently mis-weighted it:
    the clearance panel averaged in "sold so far", and the backtest graded a
    truncated actual arm against two full-horizon simulated ones. A category
    that needs special-casing at every consumer belongs excluded at the
    source, not defended at each one.

    And the count should be tiny. Only the EXTRACT's last hours can leave an
    episode unfinished -- split boundaries cannot, because `window_slice`
    assigns episodes whole by opening date -- so on a 175-day extract of ~36h
    windows this is under 1% of episodes if the feed is healthy. Production
    measured 3.38%, which is the feed problem `edge_truncated` splits out,
    not a population worth protecting.

    KEPT in `integrity`, so m11 can still count the residue.
    """
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


# ------------------------------------------------------- who reads what

def test_population_resolves_the_config_default(cfg):
    d, _ = tag_dp_eligibility(pd.concat([_frame(), _frame(cost=0.0)
                                         .assign(episode_id="bad")]), cfg)
    assert cfg["baseline_model"]["train_population"] == "eligible"
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
    precondition -- an ineligible episode has no feasible tier at all."""
    from bootstrap import train_baseline, fit_dispersion, estimate_prior
    from backtest import __main__ as bt
    from pipeline import shadow

    for fn in (train_baseline.train, fit_dispersion.fit_dispersion,
               estimate_prior.estimate_prior):
        assert "population(" in inspect.getsource(fn), fn.__name__

    for fn in (bt.main, shadow.run_shadow):
        src = inspect.getsource(fn)
        assert 'population(d, cfg, "dp_eligible")' in src, fn.__qualname__


def test_gate_one_reads_the_integrity_population():
    """The whole reason for the change: measured on dp_eligible this gate is
    0.0 by construction and can never fire."""
    from bootstrap import measure
    doc = inspect.getdoc(measure.m1_cost_ratio)
    assert "INTEGRITY population" in doc
    src = inspect.getsource(measure.gates)
    assert "measured_on" in src


def test_il_is_reported_on_both_bases():
    """One is what the business loses; the other is what the MVP can address.
    Quoting either alone was how a sub-population figure became the headline."""
    from bootstrap import measure
    src = inspect.getsource(measure.m6_il_pct)
    assert '"integrity"' in src and '"dp_eligible"' in src
    assert "population_note" in src


def test_the_cost_floor_is_not_a_population_choice():
    """`dp_eligible` selects rows. It must never be the thing that keeps a
    below-cost price out of the action set -- that is structural."""
    from pricing import dp
    tiers, d_max = dp.feasible_tiers(1000.0, 400.0, 0.025)
    assert all(1000.0 * (1 - t) >= 400.0 - 1e-9 for t in tiers)
    assert "dp_eligible" not in inspect.getsource(dp.feasible_tiers)


def test_an_unknown_outcome_reaches_the_harnesses_and_is_not_charged_scrap(cfg):
    """`edge_truncated` staying in `dp_eligible` sends unclosed episodes into
    the backtest and shadow for the first time. Both already had the branch;
    nothing had exercised it, because the filter chain deleted the subject.

    What must hold is that the observed-world baseline does NOT book scrap it
    never saw. Charging an unfinished episode's leftover to scrap inflates the
    number the policy is compared against, which flatters the policy.
    """
    from backtest.replay import _episode_frame, _replay_one

    g = pd.DataFrame({
        "episode_id": ["e"] * 3,
        "date": ["2026-05-01"] * 3, "hour_of_day": [9, 10, 11],
        "total_discount": [0.25, 0.25, 0.30],
        "original_price": [10_000.0] * 3, "cost": [4000.0] * 3,
        "d_ref": [0.25] * 3, "starting_inventory": [10, 8, 6],
        "units_sold": [2, 2, 2], "mu_ref_hat": [2.0] * 3,
        "r": [3.0] * 3, "eps": [-2.0] * 3,
    })
    g["ending_inventory"] = g.starting_inventory - g.units_sold
    known = _episode_frame(g)
    unknown = _episode_frame(g, unfinished=frozenset({"e"}))
    assert known["outcome_known"] and not unknown["outcome_known"]
    assert known["end_inv"] == unknown["end_inv"] == 4

    row_k, _ = _replay_one(known, cfg)
    row_u, _ = _replay_one(unknown, cfg)
    assert row_k["actual_scrap_cost"] == pytest.approx(4 * 4000.0)
    assert row_u["actual_scrap_cost"] == 0.0
    # the DISCOUNT half is real either way -- those hours happened
    assert row_u["actual_discount_cost"] == pytest.approx(
        row_k["actual_discount_cost"])
    # and the model arms are untouched: they simulate the full window, so an
    # unknown ENDING says nothing about what the policy would have done
    for col in ("legacy_model_il", "dp_il"):
        assert row_u[col] == pytest.approx(row_k[col])


# ------------------------------------- supply, and who is allowed to see it

def test_supply_not_opening_stock_is_the_clearance_denominator():
    """A restocked episode has more to sell than it opened with.

    Against opening stock the ratio is nonsense in a specific and embarrassing
    way: opened with 3, took 10 mid-flight, sold 9, scrapped 4 -- and the EDA
    panel reported 300% cleared, counted it in `share_fully_cleared`, and hid
    the overflow behind a histogram clipped at 1.0.
    """
    from common.episodes import episode_flow
    d = pd.DataFrame({
        "episode_id": ["R"] * 3, "date": ["2026-03-01"] * 3,
        "hour_of_day": [10, 11, 12],
        "starting_inventory": [3, 12, 8], "units_sold": [1, 4, 4],
        "ending_inventory": [12, 8, 0]})
    flow = episode_flow(d)
    assert flow.loc["R", "opening"] == 3
    assert flow.loc["R", "arrived"] == 10
    assert flow.loc["R", "supply"] == 13
    assert flow.loc["R", "clearance"] == pytest.approx(9 / 13)
    assert flow.loc["R", "clearance"] <= 1.0


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
    """It used to be excluded, and that was wrong.

    The exclusion rested on the DP's transition assuming one pool draining
    monotonically -- which confuses the SOLVE with the REPLAY. Within a single
    solve the DP does assume that, and it should: production cannot see a
    future delivery either. But the replay re-solves every hour against the
    stock on hand and applies the episode's own per-hour adjustment, so the DP
    learns about an arrival at the next hour, exactly as it does live because
    `ending[t]` IS `starting[t+1]`.

    Excluding them also selected against precisely the fast-moving SKUs that
    get replenished, which is the bias this whole chain has been unwinding.
    """
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
    """supply = sold + remaining, computed two ways that must agree.

    `remaining` from the flow is `supply - sold`. `remaining_from_last_row` is
    read off the final hour -- `ending_inventory`, or `starting - sold` where
    the write-off zeroed it. Two arithmetic paths over different fields, so
    agreement is evidence and no tolerance has to be invented.

    The consequences the owner asked for fall straight out of it: clearance
    can never exceed 1, and a fully cleared episode cannot carry scrap.
    """
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
    """The owner's real episode, and the correction that produced this test.

    Hours 12 and 18 each take a unit in; hours 11 and 17 each lose one. An
    earlier version NETTED those to zero on the theory that each pair was one
    sale bucketed an hour late -- an inference dressed as arithmetic. It read
    a window with 2 units restocked and 2 units shrunk as having neither,
    which let a restocked episode into the DP-side population and priced its
    clearance against a supply short by the 2 that arrived.

    Gross, both of them, and the episode is flagged on both counts.
    """
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
    d = pd.DataFrame(rows, columns=[
        "episode_id", "hour_of_day", "hours_remaining",
        "starting_inventory", "units_sold", "ending_inventory"])
    d["date"] = "2026-03-01"
    d["category"] = "MEAT"
    d["cost"] = 4000.0
    d["original_price"] = 10_000.0
    d["total_discount"] = 0.25
    return d


def test_an_unfinished_episode_has_no_clearance_to_report(cfg):
    """"Sold so far" is not clearance, and averaging it in biases ONE way.

    A window that has not ended has by definition sold less than it will, and
    unclosed episodes are the LARGEST in the extract -- 3.38% of the count
    holding 78.6% of at-risk leftover units on production. So the drag is far
    bigger than the share. Here one unclosed episode pulled the mean from 0.70
    to 0.45 before it was excluded.
    """
    from common import episodes as E
    from tools.eda import p_clearance
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

    out = p_clearance(d, cfg)
    assert out["mean_clearance"] == pytest.approx(0.70), \
        "an unfinished episode is being averaged into clearance again"
    assert out["supply"]["episodes_excluded_unclosed"] == 1
    assert out["supply"]["max_clearance"] <= 1.0


def test_the_backtest_grades_on_known_outcomes_only(cfg):
    """The same bias, and worse, because the two arms are asymmetric.

    For an unfinished episode the ACTUAL arm carries only the observed sales
    -- synthetic extension rows carry `units_sold = 0` -- and no scrap, since
    `outcome_known` zeroes it. Both SIMULATED arms run the full extended
    horizon and book scrap at the end. So the actual arm is truncated while
    the arms it is graded against are not, and the DP looks better by exactly
    the tail the extract did not cover.
    """
    import inspect
    from backtest import replay
    src = inspect.getsource(replay.policy_replay)
    assert "ep = ep_all[ep_all.outcome_known]" in src, \
        "the backtest is aggregating over unfinished episodes again"
    assert "episodes_excluded_unclosed" in src, \
        "the exclusion must be counted, not silent"
