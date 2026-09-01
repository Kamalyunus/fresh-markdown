"""tau moves on spend, at the operator gate (design 5.8)."""

import numpy as np
import pandas as pd
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
    # These tests are about TAU, not about level calibration. Point the factor
    # artifact at a path that does not exist so the calibration-currency gate
    # passes trivially: otherwise they read whichever schedule happens to be
    # in artifacts/ and start failing the week after it was last fitted.
    c["baseline_model"] = dict(c["baseline_model"],
                               calibration_factor_path="artifacts/__absent__.json")
    return c


def _decision(i, discount, exploration_cost, date="2026-08-19"):
    return {
        "event": "decision", "decision_id": f"D{i}", "episode_id": f"EP{i}",
        "is_entry": True, "sku_id": f"S{i}", "fc": "FC-04",
        "category": "vegetables", "subcategory": "leafy_greens",
        "date": date, "hour_of_day": 17, "hours_remaining": 1, "q_remaining": 2,
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


def _store(cfg, tmp_path, n, cost_each, date="2026-08-19", history_days=3,
           history_cost=0.0):
    """`history_days` of prior closed episodes before `date`, because the
    budget is a share of TRAILING realised IL -- a single-day store has no
    trailing base and the controller correctly holds tau still."""
    store = EventStore(cfg, root=str(tmp_path / "events"))
    i = 0
    for back in range(history_days, 0, -1):
        day = str(pd.Timestamp(date) - pd.Timedelta(days=back))[:10]
        for _ in range(n):
            store.emit_decision(_decision(i, 0.30, history_cost, day))
            store.emit_outcome(_outcome(i, 1, day))
            i += 1
    for _ in range(n):
        store.emit_decision(_decision(i, 0.30, cost_each, date))
        store.emit_outcome(_outcome(i, 1, date))
        i += 1
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
    learning = mon.learning_metrics(decisions, posterior, cfg, outcomes)
    business = mon.business_metrics(decisions, outcomes, cfg)

    day = block["through_date"]
    # SAME DAY on both sides: the controller prices the day just closed, and
    # the stop condition backstops that same day. All-time totals on both
    # sides diluted each day's correction by 1/N and blinded the backstop
    # with it.
    assert block["realised_exploration_cost"] == pytest.approx(
        learning["exploration_cost_by_day"][day])
    assert block["markdown_il"] == pytest.approx(
        upd.explore.trailing_daily_il(business["il_by_close_day"], day, cfg))
    # the day being priced closed episodes of its own, yet the budget base
    # is strictly the days BEFORE it (design 5.8: trailing, never same-day)
    il_by_day = business["il_by_close_day"]
    assert il_by_day[day] > 0
    assert block["markdown_il"] == pytest.approx(
        upd.explore.trailing_daily_il(
            {k: v for k, v in il_by_day.items() if k != day}, day, cfg))


def test_a_single_overspending_day_is_not_diluted_by_history(cfg, tmp_path):
    """All-time spend vs all-time IL made the correction weaker the longer
    the system ran: the ratio tends to 1.0, so a 10x day moved tau by 0.76x
    instead of the 0.5x clip, and the stop condition never fired."""
    lo = cfg["exploration"]["tau_adjust_clip"][0]
    short = _store(cfg, tmp_path / "short", 20, cost_each=5000.0,
                   history_days=3, history_cost=1.0)
    long = _store(cfg, tmp_path / "long", 20, cost_each=5000.0,
                  history_days=25, history_cost=1.0)
    a = upd.tau_calibration(short.load_decisions(), short.load_outcomes(),
                            _posterior(cfg, tmp_path / "short"), cfg)
    b = upd.tau_calibration(long.load_decisions(), long.load_outcomes(),
                            _posterior(cfg, tmp_path / "long"), cfg)
    for block in (a, b):
        assert block["tau_after"] == pytest.approx(
            block["tau_before"] * lo, abs=0.01), block
    assert a["realised_exploration_cost"] == b["realised_exploration_cost"]


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
    the artifact for a while, always reset and never incremented."""
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
    """The bug this exists for is positive feedback, not a rounding error."""
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


def test_an_unrouted_global_cell_cannot_pin_the_budget(cfg, tmp_path):
    """GLOBAL is always created -- cell_name falls back to it for an unknown
    category -- but when every category clears the episode floor nothing
    routes there, so it never receives an outcome and never narrows. Taking
    the max over ALL cells then pinned widest_std at the launch std forever:
    the budget could not scale down as the learning cells converged."""
    floor = cfg["posterior"]["min_episodes_per_week_for_cell"]
    store = PosteriorStore.initialise(
        cfg, {"vegetables": {"mean": -1.0, "std": 0.6},
              "fruit": {"mean": -1.2, "std": 0.5}},
        {"vegetables": floor, "fruit": floor},          # both own their cells
        path=str(tmp_path / "posterior.json"))

    cells, cell_of = store.state["cells"], store.state["cell_of"]
    assert "GLOBAL" in cells and "GLOBAL" not in cell_of.values()
    assert store.widest_std() == pytest.approx(0.6)

    # both routed cells converge to the floor; GLOBAL never moves
    for name in ("vegetables", "fruit"):
        cells[name]["std"] = cfg["posterior"]["min_std"]
    assert cells["GLOBAL"]["std"] == pytest.approx(0.6)
    assert store.widest_std() == pytest.approx(cfg["posterior"]["min_std"])

    # and when a category DOES route to GLOBAL it counts again
    routed = PosteriorStore.initialise(
        cfg, {"vegetables": {"mean": -1.0, "std": 0.6},
              "fruit": {"mean": -1.2, "std": 0.5}},
        {"vegetables": floor, "fruit": 0},
        path=str(tmp_path / "routed.json"))
    assert "GLOBAL" in routed.state["cell_of"].values()
    assert routed.widest_std() == pytest.approx(0.6)
