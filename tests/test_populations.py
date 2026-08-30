"""Two populations, and which consumer is entitled to which."""

import inspect

import numpy as np
import pandas as pd
import pytest

import pathlib

from bootstrap.prepare_data import (DP_INELIGIBLE, load_and_filter, population,
                                    tag_dp_eligibility)
from common.config import load_config

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    import os
    return load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))


def _frame(**over):
    """One clean two-hour episode; keyword overrides break it one way."""
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
    _, wf = load_and_filter(str(ROOT / "data" / "flc_synth.parquet"), cfg)
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
    precondition -- an ineligible episode has no feasible tier at all."""
    from bootstrap import train_baseline, fit_dispersion, estimate_prior
    from backtest import __main__ as bt
    from pipeline import shadow

    from bootstrap import prior_density
    for fn in (train_baseline.train, fit_dispersion.fit_dispersion,
               prior_density.build_curves):
        assert "population(" in inspect.getsource(fn), fn.__name__

    for fn in (bt.main, shadow.run_shadow):
        src = inspect.getsource(fn)
        assert 'population(d, cfg, "dp_eligible")' in src, fn.__qualname__


def test_il_is_reported_on_both_bases():
    """One is what the business loses; the other is what the MVP can address.
    Quoting either alone was how a sub-population figure became the headline."""
    from common import metrics
    src = inspect.getsource(metrics.il_pct)
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
    nothing had exercised it, because the filter chain deleted the subject."""
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
    """A restocked episode has more to sell than it opened with."""
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


def test_the_backtest_grades_on_known_outcomes_only(cfg):
    """The same bias, and worse, because the two arms are asymmetric."""
    import inspect
    from backtest import replay
    src = inspect.getsource(replay.policy_replay)
    assert "ep = ep_all[ep_all.outcome_known]" in src, \
        "the backtest is aggregating over unfinished episodes again"
    assert "episodes_excluded_unclosed" in src, \
        "the exclusion must be counted, not silent"


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
    info_at = body.index('out["raw_information"] +=')
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
            "r": [3.0] * 4, "eps": [eps] * 4,
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
