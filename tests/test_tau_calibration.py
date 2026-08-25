"""tau moves on spend, at the operator gate (design 5.8).

`explore.tau_next` existed, was unit-tested, and had no caller: tau was pasted
from config at launch and never moved again. The proportional controller the
design specifies was therefore absent, and the only response to overspending
was the monitor's stop condition -- which does not spend less, it stops
exploring altogether. These tests hold the controller in place and, more
importantly, hold it on the SAME basis the stop condition uses, so the two
cannot drift into disagreeing about what "over budget" means.
"""

import numpy as np
import pytest

from common.config import load_config
from events.store import EventStore
from pipeline import monitor as mon
from pipeline import update as upd
from pricing.posterior import PosteriorStore

P0, COST = 10000.0, 4000.0


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    if c["exploration"]["tau_initial"] is None:      # null until a gate passes
        c["exploration"]["tau_initial"] = 447.78     # the production paste
    return c


def _decision(i, discount, exploration_cost, date="2026-08-19"):
    return {
        "event": "decision", "decision_id": f"D{i}", "episode_id": f"EP{i}",
        "is_entry": True, "sku_id": f"S{i}", "fc": "FC-04",
        "category": "vegetables", "subcategory": "leafy_greens",
        "hour_of_day": 17, "hours_remaining": 1, "q_remaining": 2,
        "original_price": P0, "cost": COST, "d_max": 0.6,
        "feasible_tier_count": 25, "action_set_size": 5,
        "optimal_price": P0 * 0.85, "optimal_discount": 0.15,
        "expected_il": 1000.0, "expected_denominator": 5000.0,
        "applied_price": P0 * (1 - discount), "applied_discount": discount,
        "is_exploration": exploration_cost > 0,
        "exploration_cost": exploration_cost,
        "affordable_set_size": 3 if exploration_cost > 0 else 0,
        "tau_current": 447.78,
        "epsilon_posterior_mean": -1.0, "epsilon_posterior_std": 0.6,
        "reference_discount": 0.3, "reference_mu": 0.8, "mu_ref_path": [0.8],
        "anchor_discount": None, "dispersion_r": 0.919,
        "baseline_model_version": "b", "posterior_version": 0,
        "config_version": "1.0.0", "timestamp": f"{date}T17:00:00+00:00",
    }


def _outcome(i, sold, date="2026-08-19"):
    """A closed episode: ending_inventory 0 with stock left is the write-off
    sentinel, which is what makes it count toward IL rather than reading as
    still running."""
    return {
        "event": "outcome", "outcome_id": f"O{i}", "decision_id": f"D{i}",
        "units_sold": sold, "starting_inventory": 2, "ending_inventory": 0,
        "applied_price": P0 * 0.7, "is_stockout": sold >= 2,
        "execution_status": "ok", "adjustment_reason": "episode_close_write_off",
        "finalized_at": f"{date}T18:00:00+00:00",
    }


def _store(cfg, tmp_path, n, cost_each, date="2026-08-19"):
    store = EventStore(cfg, root=str(tmp_path / "events"))
    for i in range(n):
        store.emit_decision(_decision(i, 0.30, cost_each, date))
        store.emit_outcome(_outcome(i, 1, date))
    return store


def _posterior(cfg, tmp_path):
    return PosteriorStore.initialise(
        cfg, {"vegetables": {"mean": -1.0, "std": 0.6}}, {"vegetables": 500},
        path=str(tmp_path / "posterior.json"))


# ------------------------------------------------------------------ the loop
def test_underspending_raises_tau_and_overspending_lowers_it(cfg, tmp_path):
    """The controller's whole job, in both directions, from one fixture."""
    lean = _store(cfg, tmp_path / "lean", 20, cost_each=0.5)
    rich = _store(cfg, tmp_path / "rich", 20, cost_each=5000.0)
    p_lean = _posterior(cfg, tmp_path / "lean")
    p_rich = _posterior(cfg, tmp_path / "rich")

    under = upd.tau_calibration(lean.load_decisions(), lean.load_outcomes(),
                                p_lean, cfg)
    over = upd.tau_calibration(rich.load_decisions(), rich.load_outcomes(),
                               p_rich, cfg)

    assert under["tau_after"] > under["tau_before"], under
    assert over["tau_after"] < over["tau_before"], over
    # never more than the configured multiple in one step, either way
    lo, hi = cfg["exploration"]["tau_adjust_clip"]
    # tau_before and tau_after are both reported to 2dp, so the clip bound
    # computed from one rounded value against the other can miss by up to a
    # cent either side. 1e-6 was tight enough only by luck.
    for block in (under, over):
        assert (block["tau_before"] * lo - 0.01
                <= block["tau_after"]
                <= block["tau_before"] * hi + 0.01)


def test_tau_is_calibrated_on_the_same_numbers_the_stop_condition_uses(cfg, tmp_path):
    """If these two ever diverge, tau could be told to grow by the controller
    while the monitor suspends exploration for overspending -- the system
    arguing with itself. Same budget, same realised cost, by construction."""
    store = _store(cfg, tmp_path, 20, cost_each=800.0)
    posterior = _posterior(cfg, tmp_path)
    decisions, outcomes = store.load_decisions(), store.load_outcomes()

    block = upd.tau_calibration(decisions, outcomes, posterior, cfg)
    learning = mon.learning_metrics(decisions, posterior, cfg)
    business = mon.business_metrics(decisions, outcomes, cfg)

    assert block["realised_exploration_cost"] == \
        learning["realised_exploration_cost"]
    assert block["markdown_il"] == \
        pytest.approx(business["il_pct_aggregate"]["il_absolute"])


# -------------------------------------------------------------- exactly once
def test_a_second_run_on_the_same_day_does_not_move_tau_again(cfg, tmp_path):
    """Without the date stamp, two --apply runs would apply the same ratio
    twice and move tau by its square."""
    store = _store(cfg, tmp_path, 20, cost_each=0.5)
    posterior = _posterior(cfg, tmp_path)
    decisions, outcomes = store.load_decisions(), store.load_outcomes()

    first = upd.tau_calibration(decisions, outcomes, posterior, cfg)
    assert first["commit"]
    posterior.commit_tau(first["tau_after"], first["through_date"])

    second = upd.tau_calibration(decisions, outcomes, posterior, cfg)
    assert not second["commit"]
    assert "already calibrated through" in second["skipped"]
    assert second["tau_after"] == first["tau_after"]
    # ...and a new day releases it again
    store.emit_decision(_decision(99, 0.30, 0.5, date="2026-08-20"))
    store.emit_outcome(_outcome(99, 1, date="2026-08-20"))
    third = upd.tau_calibration(store.load_decisions(), store.load_outcomes(),
                                posterior, cfg)
    assert third["commit"] and third["through_date"] == "2026-08-20"


# ------------------------------------------------------------------- wiring
def test_apply_persists_tau_and_monitor_only_does_not(cfg, tmp_path):
    """The operator gate governs tau exactly as it governs the posterior."""
    for mode, expect_moved in (("dry", False), ("apply", True)):
        root = tmp_path / mode
        store = _store(cfg, root, 20, cost_each=0.5)
        posterior = _posterior(cfg, root)
        before = posterior.tau(cfg)

        report = upd.run(cfg, apply=(mode == "apply"),
                         events_root=str(root / "events"),
                         posterior_path=str(root / "posterior.json"))
        assert report["tau_calibration"]["commit"]

        after = PosteriorStore(cfg, path=str(root / "posterior.json")).tau(cfg)
        assert (after != before) is expect_moved, mode


def test_tau_moves_even_when_no_cell_reaches_the_information_threshold(cfg, tmp_path):
    """Spend and evidence are different currencies. A day that explored and
    learned nothing still cost money, and that is what tau prices."""
    store = _store(cfg, tmp_path, 20, cost_each=0.5)
    posterior = _posterior(cfg, tmp_path)
    report = upd.run(cfg, apply=True, events_root=str(tmp_path / "events"),
                     posterior_path=str(tmp_path / "posterior.json"))

    assert not any(c["update_triggered"] for c in report["cells"].values()), \
        "fixture must not clear the information threshold, or it proves nothing"
    reloaded = PosteriorStore(cfg, path=str(tmp_path / "posterior.json"))
    assert reloaded.tau(cfg) != cfg["exploration"]["tau_initial"]
    # the posterior itself must NOT have moved
    assert reloaded.state["cells"]["vegetables"]["version"] == 0


def test_the_posterior_carries_no_information_since_update_counter(cfg, tmp_path):
    """The original spec carried a running information counter and it was carried in
    the artifact for a while, always reset and never incremented.

    Adding it back would be a bug, not a schema restoration: the trigger reads
    the UNCONSUMED BATCH, and nothing consumes a sub-threshold one, so a
    counter incremented while the same outcomes are re-read next run double
    counts them. A field that is permanently zero also reads as "no evidence
    has accrued", which is the opposite of what a growing batch means.
    """
    posterior = _posterior(cfg, tmp_path)
    for rec in posterior.state["cells"].values():
        assert "information_since_update" not in rec

    posterior.commit_update("vegetables", -1.1, 0.55, 20, 15.0, ["X1"],
                            applied=True)
    rec = PosteriorStore(cfg, path=str(tmp_path / "posterior.json")) \
        .state["cells"]["vegetables"]
    assert "information_since_update" not in rec
    # the total that DID survive, and is the one to read
    assert rec["accumulated_information"] == pytest.approx(15.0)


def test_a_null_tau_initial_is_reported_not_crashed(cfg, tmp_path):
    """Before a gate-passing backtest there is nothing in force to calibrate."""
    blank = dict(cfg, exploration=dict(cfg["exploration"], tau_initial=None))
    store = _store(cfg, tmp_path, 4, cost_each=1.0)
    posterior = _posterior(cfg, tmp_path)
    block = upd.tau_calibration(store.load_decisions(), store.load_outcomes(),
                                posterior, blank)
    assert not block["commit"]
    assert block["tau_before"] is None
    assert "null" in block["skipped"]


def test_a_window_with_no_exploration_holds_tau_still(cfg, tmp_path):
    """The bug this exists for is positive feedback, not a rounding error.

    With no forced decisions the realised cost is 0. `tau_next` floors the
    denominator at `tau_spend_guard` (1 won), so `budget / 1` is enormous and
    clips to the 2x ceiling -- tau DOUBLES. And the window where exploration
    is absent is exactly the one where the stop condition suspended it for
    OVERSPENDING, so tau would grow every day it was switched off and return
    further over budget than it went away: 448 -> 896 -> 1792 -> 3584.

    An absence of spend is an absence of signal. Hold tau still.
    """
    store = EventStore(cfg, root=str(tmp_path / "events"))
    for i in range(20):                       # exploitation only: cost 0, no draws
        d = _decision(i, 0.30, exploration_cost=0.0)
        assert not d["is_exploration"]
        store.emit_decision(d)
        store.emit_outcome(_outcome(i, 1))
    posterior = _posterior(cfg, tmp_path)

    before = posterior.tau(cfg)
    block = upd.tau_calibration(store.load_decisions(), store.load_outcomes(),
                                posterior, cfg)
    assert not block["commit"], "tau must not move on a window with no exploration"
    assert "no exploration" in block["skipped"]
    assert block["tau_after"] == before

    # and the raw controller still does the dangerous thing, so the guard above
    # is what stands between the two -- not luck
    from pricing import explore
    doubled = explore.tau_next(before, 805_478.0, 0.0, cfg)
    assert doubled == pytest.approx(before * cfg["exploration"]["tau_adjust_clip"][1])
