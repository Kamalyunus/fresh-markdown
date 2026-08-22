"""A mid-episode restock in PRODUCTION, which is a different path from the one
`bootstrap.prepare_data` guards.

Offline, `restocked_episodes_dropped` removes these episodes: the DP rests on a
single inventory pool draining monotonically, so a window that gains stock
half-way through is not evidence about a system that cannot model it. That rule
is tested in `test_end_to_end.test_restocked_episodes_are_dropped`.

Live, a restock simply happens, and dropping it is not an option. The claim
this file exists to hold up is that production ABSORBS it -- the agent is a
policy re-solved every hour rather than a plan, and the monitor never assumes a
single pool -- so nothing quarantines and IL stays correct. That claim was made
to a reviewer; it needs a test behind it rather than an argument.

The second test pins the one place a restock genuinely degrades something, so
that a future change to it is deliberate rather than accidental.
"""

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
        "hour_of_day": 16 + i, "hours_remaining": hours_remaining,
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
    window closes with four written off.

    Inventory does not reconcile twice over -- once upward at the restock and
    once downward at the close -- and both are legitimate, so both must be
    named and both must land. IL is then checked against the arithmetic done
    by hand, because "it produced a number" is not the same as "it produced
    the right number":

        discount given away  4 sold x (10000 - 7000) = 12,000
        scrap                4 left x 4,000          = 16,000
        IL                                             28,000
        denominator          4 sold x 10,000         = 40,000
    """
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


def test_stock_arriving_mid_hour_is_learnt_from_as_a_lower_bound(cfg):
    """The one case a restock degrades, pinned rather than left latent.

    `grid_update` decides censoring with `units_sold >= starting_inventory`.
    That is right for an ordinary stockout, and wrong for an hour that sold
    more than it opened with because stock arrived during it: demand was
    observed EXACTLY, and the likelihood uses "at least starting_inventory"
    instead.

    The direction is safe -- it discards information rather than biasing
    epsilon -- so it is recorded here rather than fixed. What this test
    guarantees is that the batch still converges, and that the two treatments
    are genuinely different, so anyone who later makes the over-sell an exact
    count will see this test change and know it was on purpose.
    """
    cell = {"mean": -1.0, "std": 0.6}
    dec = _decision(0, 1, 1)
    dec["applied_discount"] = 0.45          # away from the 0.30 reference, so
    ratio = (1 - 0.45) / (1 - 0.30)         # the batch carries real information

    # sold 3 having opened the hour with 1: stock arrived mid-hour
    oversell = [(dec, _outcome(0, 3, 1, 3), ratio)]
    # the identical observation on a shelf deep enough to make it an exact count
    exact = [(dec, _outcome(0, 3, 9, 6), ratio)]

    m_over, s_over, info_over, _ = grid_update(oversell * 40, cell, cfg)
    m_exact, s_exact, info_exact, _ = grid_update(exact * 40, cell, cfg)

    for value in (m_over, s_over, info_over):
        assert np.isfinite(value)
    assert s_over > 0
    # censored on one side, exact on the other -> different posteriors. If this
    # ever comes out equal, the censoring flag stopped distinguishing them.
    assert m_over != pytest.approx(m_exact, abs=1e-6), (
        "the over-sell row is no longer being treated as censored")
    # information does not depend on the censoring flag -- it is computed from
    # mu at the prior mean and the log price ratio -- so it must NOT move
    assert info_over == pytest.approx(info_exact)
