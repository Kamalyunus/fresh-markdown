"""A mid-episode restock in PRODUCTION, which is a different path from the one
`bootstrap.prepare_data` guards."""

import numpy as np
import pytest

from common.config import load_config
from events.store import EventStore
from pipeline import monitor as mon
from common.episodes import adjustment_reason
from pipeline.update import grid_update

P0, COST, PRICE = 10000.0, 4000.0, 7000.0


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _decision(i, hours_remaining, q):
    return {
        "event": "decision", "decision_id": f"D{i}", "episode_id": "EP-R",
        "is_entry": i == 0, "sku_id": "S1", "fc": "FC-04",
        "category": "vegetables", "subcategory": "leafy_greens",
        "date": "2026-08-19", "hour_of_day": 16 + i,
        "hours_remaining": hours_remaining,
        "q_remaining": q, "original_price": P0, "cost": COST,
        "d_max": 0.6, "feasible_tier_count": 25, "action_set_size": 5,
        "optimal_price": PRICE, "optimal_discount": 1 - PRICE / P0,
        "expected_il": 1000.0, "expected_denominator": 5000.0,
        "applied_price": PRICE, "applied_discount": 1 - PRICE / P0,
        "is_exploration": False, "exploration_cost": 0.0,
        "affordable_set_size": 0, "tau_current": 3202.33,
        "epsilon_posterior_mean": -1.0, "epsilon_posterior_std": 0.6,
        "reference_discount": 0.3, "reference_mu": 0.8,
        "mu_ref_path": [0.8] * hours_remaining,
        "anchor_discount": None if i == 0 else 0.3, "dispersion_r": 0.919,
        "baseline_model_version": "b", "posterior_version": 0,
        "config_version": "1.0.0",
        "timestamp": f"2026-08-19T{16 + i:02d}:00:00+00:00",
    }


def _outcome(i, sold, start, end):
    evt = {
        "event": "outcome", "outcome_id": f"O{i}", "decision_id": f"D{i}",
        "units_sold": sold, "starting_inventory": start,
        "ending_inventory": end, "applied_price": PRICE,
        "is_stockout": sold >= start, "execution_status": "ok",
        "finalized_at": f"2026-08-19T{17 + i:02d}:00:00+00:00",
    }
    # the producer's job, and the one thing that must not be skipped
    reason = adjustment_reason(start, sold, end)
    if reason:
        evt["adjustment_reason"] = reason
    return evt


def test_a_restocked_episode_lands_and_its_il_is_right(cfg, tmp_path):
    """Three hours: 3 units, one sells, FIVE ARRIVE mid-episode, then the
    window closes with four written off."""
    store = EventStore(cfg, root=str(tmp_path / "events"))
    #        i, hours, q,  sold, start, end
    plan = [(0, 3, 3, 1, 3, 2),
            (1, 2, 2, 1, 2, 6),     # +5 arrive during the hour
            (2, 1, 6, 2, 6, 0)]     # window closes, 4 written off
    for i, hours, q, sold, start, end in plan:
        assert store.emit_decision(_decision(i, hours, q))
        assert store.emit_outcome(_outcome(i, sold, start, end)), \
            f"hour {i} did not land"

    assert store.load_quarantine() == []
    reasons = [o.get("adjustment_reason") for o in store.load_outcomes()]
    assert reasons == [None, "intraday_restock", "episode_close_write_off"]

    il = mon.business_metrics(store.load_decisions(), store.load_outcomes(),
                              cfg)["il_pct_aggregate"]
    # scrap is read off the LAST row's starting inventory, which already
    # carries the restock -- which is exactly why the monitor survives one
    assert il["il_absolute"] == pytest.approx(28000.0)
    assert il["il_pct_denominator"] == pytest.approx(40000.0)
    assert il["il_pct"] == pytest.approx(0.70)


def test_a_shrink_hour_is_named_and_lands_rather_than_quarantining(cfg, tmp_path):
    """Shrink is the THIRD legitimate break, and it used to be the odd one out."""
    store = EventStore(cfg, root=str(tmp_path / "events"))
    #        i, hours, q, sold, start, end
    plan = [(0, 3, 6, 1, 6, 5),     # ordinary: 6 - 1 = 5
            (1, 2, 5, 1, 5, 3),     # SHRINK: one unit gone, unsold, not written off
            (2, 1, 3, 1, 3, 0)]     # close, 2 written off
    for i, hours, q, sold, start, end in plan:
        assert store.emit_decision(_decision(i, hours, q))
        assert store.emit_outcome(_outcome(i, sold, start, end)), \
            f"hour {i} did not land -- a shrink hour is quarantining again"

    assert store.load_quarantine() == []
    assert store.quarantined_this_run == 0
    reasons = [o.get("adjustment_reason") for o in store.load_outcomes()]
    assert reasons == [None, "unexplained_shortfall", "episode_close_write_off"]

    # every decision got an outcome: that is what event_completeness measures,
    # and shrink must not subtract from it
    assert len(store.load_outcomes()) == len(store.load_decisions())


def test_the_quarantine_count_is_a_property_of_the_run_not_the_file(cfg, tmp_path):
    """`quarantined_event_count` was read off the whole quarantine FILE."""
    root = str(tmp_path / "events")
    bad = {"outcome_id": "o1", "decision_id": "d1", "units_sold": -2,
           "starting_inventory": 5, "ending_inventory": 3,
           "applied_price": 100.0, "is_stockout": False,
           "execution_status": "ok", "finalized_at": "2026-08-19T17:00:00+00:00"}

    for run in range(3):
        store = EventStore(cfg, root=root)
        assert store.quarantined_this_run == 0, "a fresh store starts at zero"
        # a NEW id each run, exactly as uuid4 gives the real path
        assert not store.emit_outcome(dict(bad, outcome_id=f"o{run}",
                                           decision_id=f"d{run}"))
        assert store.quarantined_this_run == 1, \
            "the run count is picking up earlier runs again"
        assert len(store.load_quarantine()) == run + 1   # the FILE still grows


def test_stock_arriving_mid_hour_is_an_exact_count_not_a_lower_bound(cfg):
    """The degradation this file used to pin is now fixed, on purpose."""
    cell = {"mean": -1.0, "std": 0.6}
    dec = _decision(0, 1, 1)
    dec["applied_discount"] = 0.45          # away from the 0.30 reference, so
    ratio = (1 - 0.45) / (1 - 0.30)         # the batch carries real information

    # sold 3 having opened the hour with 1, ending with 3 on the shelf: 5
    # arrived mid-hour and nothing ran out
    oversell = [(dec, _outcome(0, 3, 1, 3), ratio)]
    # the identical observation on a shelf deep enough to be exact anyway
    exact = [(dec, _outcome(0, 3, 9, 6), ratio)]
    # and a genuine stockout: opened with 3, sold 3, ended empty, no restock
    stockout = [(dec, _outcome(0, 3, 3, 0), ratio)]

    m_over, s_over, info_over, rep_over = grid_update(oversell * 40, cell, cfg)
    m_exact, s_exact, info_exact, _ = grid_update(exact * 40, cell, cfg)
    m_stock, _, info_stock, rep_stock = grid_update(stockout * 40, cell, cfg)

    for value in (m_over, s_over, info_over):
        assert np.isfinite(value)
    assert s_over > 0

    # THE FIX: the over-sell is now treated exactly like the deep shelf,
    # because in both the hour ended holding stock and demand was observed
    assert m_over == pytest.approx(m_exact, abs=1e-9), (
        "an hour that ended holding stock is being treated as censored again "
        "-- `sold >= starting` is back somewhere")
    assert rep_over["stockout_share"] == 0.0

    # ...and a real stockout still is censored, so the flag has not simply
    # stopped distinguishing anything
    assert rep_stock["stockout_share"] == 1.0
    assert m_stock != pytest.approx(m_exact, abs=1e-6)

    # information does not depend on the censoring flag -- it is computed from
    # mu at the prior mean and the log price ratio -- so it must NOT move
    assert info_over == pytest.approx(info_exact)
    assert info_stock == pytest.approx(info_exact)


def test_information_is_in_nb_units_not_poisson(cfg):
    """The information counter must be the NB Fisher information
    mu * L^2 * r/(r+mu), not the Poisson mu * L^2. The Poisson form
    overstated evidence by ~1.6-1.9x at production mu and r -- on top of
    what deff corrects -- so `information_increment` fired earlier than its
    face value. Design 5.11."""
    from pipeline.update import deff

    cell = {"mean": -1.0, "std": 0.6}
    dec = _decision(0, 1, 9)
    dec["applied_discount"] = 0.45
    ratio = (1 - 0.45) / (1 - 0.30)
    pairs = [(dec, _outcome(0, 1, 9, 8), ratio)] * 40

    _, _, eff_info, _ = grid_update(pairs, cell, cfg)

    mu = max(dec["reference_mu"] * ratio ** cell["mean"],
             cfg["pricing"]["demand_floor"])
    L2 = np.log(ratio) ** 2
    r = dec["dispersion_r"]
    nb = 40 * mu * L2 * (r / (r + mu)) / deff(cfg)
    poisson = 40 * mu * L2 / deff(cfg)
    assert eff_info == pytest.approx(nb, rel=1e-9)
    assert eff_info < poisson  # the damping is real, not a no-op


def test_the_predictive_check_grades_the_belief_out_of_sample(cfg):
    """Tomorrow's real outcomes grade today's belief (design 5.11): every
    batch is scored against the PRE-update posterior, bracketed by oracle
    and uniform. A posterior sitting near what the data says must beat a
    flat belief; a confident posterior far from it must score BELOW uniform
    and be named -- that is the only test of max_std_shrink/min_std real
    data can run, and it runs rolling-forward in production."""
    dec = _decision(0, 1, 5)
    dec["applied_discount"] = 0.45
    ratio = (1 - 0.45) / (1 - 0.30)
    pairs = [(dec, _outcome(0, 1, 5, 4), ratio)] * 60

    _, _, _, d_good = grid_update(pairs, {"mean": -1.0, "std": 0.6}, cfg)
    _, _, _, d_bad = grid_update(pairs, {"mean": -3.9, "std": 0.06}, cfg)

    g, b = d_good["predictive_check"], d_bad["predictive_check"]
    # brackets are ordered: oracle is the hindsight best, so nothing beats it
    for c in (g, b):
        assert c["oracle_log_pred_per_row"] >= c["posterior_log_pred_per_row"]
        assert c["oracle_log_pred_per_row"] >= c["uniform_log_pred_per_row"]
        assert c["information_available_per_row"] >= 0
    # a reasonable belief predicts the batch better than no opinion
    assert g["posterior_minus_uniform"] > 0 and not g["worse_than_a_flat_prior"]
    # a confident belief far from the data is worse than knowing nothing,
    # and the report says so in those words
    assert b["posterior_minus_uniform"] < 0 and b["worse_than_a_flat_prior"]
