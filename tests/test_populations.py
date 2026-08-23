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

Five conditions gate `dp_eligible`, and each one names something the SOLVER
cannot do -- no feasible tier (`cost_missing`, `non_priceable`), no trustable
horizon (`negative_window`, `window_too_long`), no single draining pool
(`restocked`). None of them names anything the demand model can see: FEATURES
carries neither `cost` nor `hours_remaining` nor anything about the inventory
chain.

Two more conditions are flagged and gate NOTHING. `below_cost_hours` is a
price the legacy policy set and the agent is constrained never to repeat.
`edge_truncated` is a missing OUTCOME rather than a limit on the DP, and every
consumer of an outcome already excludes it by the closure sentinel.
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
    # 11 units arrive during hour 1: it opened with 12, sold 3, and the
    # source reports the FINAL count of 20. Hour 2 then opens with 20, so the
    # chain stays continuous -- that is what makes it a restock, not a break.
    ("restocked", {"starting_inventory": [12, 20], "ending_inventory": [20, 0]}),
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


def test_an_unclosed_ending_is_reported_but_does_not_gate(cfg):
    """`edge_truncated` marks a missing OUTCOME, not a limit on the DP.

    The extract stopping mid-window says nothing about the hours it did
    capture: they are ordinary, fully-priced demand observations, and these
    are the LARGEST episodes in the data (~25 units of opening stock against
    ~3 for the population). What is missing is the ending, and every consumer
    of an ending already handles that on its own -- `scrap_units` returns NaN,
    `backtest.replay` zeroes scrap under `outcome_known`, `pipeline.shadow`
    charges scrap only on COMPLETED. Gating on it would buy nothing and cost
    the demand fit its best-observed windows.
    """
    # TWO episodes, because closure is read from the source's own sentinel and
    # a frame with no sentinel anywhere proves nothing -- `classify_last`
    # rightly falls back to treating every episode as closed rather than
    # sweeping all the scrap into UNKNOWN. `e` carries the sentinel; `u` ends
    # holding honest stock with hours still on its counter, so it is unclosed
    # AND at the extract's edge.
    unclosed = _frame(episode_id="u", ending_inventory=[9, 7])
    d = pd.concat([_frame(), unclosed], ignore_index=True)
    tagged, detail = tag_dp_eligibility(d, cfg)

    assert (tagged.edge_truncated == (tagged.episode_id == "u")).all()
    assert tagged.dp_eligible.all(), "an unclosed ending must not gate the DP"
    assert tagged.dp_ineligible_reason.isna().all()
    assert "edge_truncated" not in [n for n, _ in DP_INELIGIBLE]

    edge = detail["edge_truncated"]
    assert edge["episodes_unclosed"] == 1
    assert edge["episodes_edge_truncated"] == 1
    assert edge["episodes_unclosed_not_edge"] == 0
    assert edge["share_of_unclosed_explained_by_edge"] == 1.0
    assert edge["still_dp_eligible"] == 1


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
    assert cfg["baseline_model"]["train_population"] == "integrity"
    assert len(population(d, cfg)) == len(d)                      # the default
    assert len(population(d, cfg, "integrity")) == len(d)
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


def test_a_sale_bucketed_an_hour_late_is_not_an_arrival():
    """The owner's episode. +1 at one hour and -1 at the next cancel, so
    supply is the opening stock and clearance is 11/12 -- not 11/14, which is
    what counting gross arrivals would give."""
    from common.episodes import episode_flow
    rows = [(0, 12, 0, 12), (1, 12, 0, 12), (2, 12, 0, 12), (3, 12, 0, 12),
            (4, 12, 0, 12), (5, 12, 0, 12), (6, 12, 0, 12), (7, 12, 0, 12),
            (8, 12, 1, 11), (9, 11, 0, 11), (10, 11, 0, 11), (11, 11, 0, 10),
            (12, 10, 1, 10), (13, 10, 0, 10), (14, 10, 0, 10), (15, 10, 1, 9),
            (16, 9, 1, 8), (17, 8, 0, 7), (18, 7, 1, 7), (19, 7, 0, 7),
            (20, 7, 4, 3), (21, 3, 2, 0)]
    d = pd.DataFrame(rows, columns=["hour_of_day", "starting_inventory",
                                    "units_sold", "ending_inventory"])
    d["episode_id"] = "S"
    d["date"] = "2026-03-01"
    flow = episode_flow(d)
    assert flow.loc["S", "arrived"] == 0, \
        "a one-hour bucket lag was counted as stock arriving"
    assert flow.loc["S", "vanished"] == 0
    assert flow.loc["S", "supply"] == 12
    assert flow.loc["S", "clearance"] == pytest.approx(11 / 12)


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


def test_a_restocked_episode_never_reaches_the_backtest(cfg):
    """Asserted on BEHAVIOUR, not on a source grep.

    A restocked episode is fine for the frozen artifacts -- the demand model
    cannot see the inventory chain -- and wrong for anything that compares
    policies or books IL, because the DP has no restock in its state
    transition and clearance against opening stock goes above 1. `dp_eligible`
    is what keeps them apart, and the other tests here check the flag; this
    one checks that the consumer actually honours it.
    """
    from bootstrap.prepare_data import population
    restocked = _frame(episode_id="R", starting_inventory=[12, 20],
                       ending_inventory=[20, 0])
    d = pd.concat([_frame(), restocked], ignore_index=True)
    tagged, _ = tag_dp_eligibility(d, cfg)

    assert set(tagged.loc[~tagged.dp_eligible, "episode_id"]) == {"R"}
    eligible = population(tagged, cfg, "dp_eligible")
    assert "R" not in set(eligible.episode_id), \
        "a restocked episode reached the DP-side population -- clearance " \
        "against opening stock would read above 1 and IL would book scrap " \
        "on an inventory pool the solver never modelled"
    # ...and it is still there for the artifact fits
    assert "R" in set(population(tagged, cfg, "integrity").episode_id)
