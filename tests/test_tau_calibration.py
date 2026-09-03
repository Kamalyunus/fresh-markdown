"""tau moves on spend, at the operator gate (design 5.8)."""

import numpy as np
import pandas as pd
import pytest

from conftest import P0, decision_event, outcome_event
from events.store import EventStore
from pipeline import monitor as mon
from pipeline import update as upd
from pricing.posterior import PosteriorStore


@pytest.fixture
def cfg(cfg):
    c = cfg
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
    """One entry decision per episode, exploring iff it cost something."""
    return decision_event(
        decision_id=f"D{i}", episode_id=f"EP{i}", sku_id=f"S{i}", date=date,
        applied_price=P0 * (1 - discount), applied_discount=discount,
        is_exploration=exploration_cost > 0, exploration_cost=exploration_cost,
        affordable_set_size=3 if exploration_cost > 0 else 0,
        timestamp=f"{date}T17:00:00+00:00")


def _outcome(i, sold, date="2026-08-19"):
    """A closed episode: ending_inventory 0 with stock left is the write-off
    sentinel, which is what makes it count toward IL rather than reading as
    still running."""
    return outcome_event(
        outcome_id=f"O{i}", decision_id=f"D{i}", units_sold=sold,
        is_stockout=sold >= 2, adjustment_reason="episode_close_write_off",
        finalized_at=f"{date}T18:00:00+00:00")


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


def _posterior(cfg, tmp_path, calibrated_through="2026-08-18"):
    """The controller walks EVERY closed day since its last calibration, so
    a single-day fixture says tau was calibrated through the day before:
    the trailing history days are budget base, not days to grade."""
    p = PosteriorStore.initialise(
        cfg, {"vegetables": {"mean": -1.0, "std": 0.6}}, {"vegetables": 500},
        path=str(tmp_path / "posterior.json"))
    if calibrated_through:
        p.commit_tau(p.tau(cfg), calibrated_through)
    return p


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
    posterior = _posterior(cfg, tmp_path, calibrated_through=None)   # nothing stored
    block = upd.tau_calibration(store.load_decisions(), store.load_outcomes(),
                                posterior, blank)
    assert not block["commit"]
    assert block["tau_before"] is None
    assert "null" in block["skipped"]


def test_zero_spend_across_priced_hours_raises_tau_by_the_clip(cfg, tmp_path):
    """Design 5.8: priced hours with NOTHING affordable is under-spend, and
    raising tau on it is the only way a tau cut below the smallest spread
    ever recovers. Production used to hold still here while shadow's trace
    walked the rule, so the two disagreed on the same data."""
    store = _store(cfg, tmp_path, 20, cost_each=0.0)      # exploitation only
    assert not any(d["is_exploration"] for d in store.load_decisions())
    posterior = _posterior(cfg, tmp_path)

    before = posterior.tau(cfg)
    block = upd.tau_calibration(store.load_decisions(), store.load_outcomes(),
                                posterior, cfg)
    assert block["commit"], block
    assert block["realised_exploration_cost"] == 0.0 and block["budget"] > 0
    assert block["tau_after"] == pytest.approx(
        before * cfg["exploration"]["tau_adjust_clip"][1], abs=0.01)


def test_an_hour_23_decision_is_graded_on_its_trading_day(cfg, tmp_path):
    """An hour-23 outcome finalizes at D+1T00:00Z. Keying the controller on
    that put it one day ahead of the IL side: it graded ONE hour of spend
    against a full day's budget, ratcheted tau up 25% a day on that basis,
    and `tau_calibrated_through` then guaranteed the other 23 hours were
    never priced at all."""
    store = _store(cfg, tmp_path, 20, cost_each=50.0)     # day-19 spend: 1000
    late = _decision(999, 0.30, 50.0)                     # same day, hour 23
    late["hour_of_day"], late["timestamp"] = 23, "2026-08-19T23:00:00+00:00"
    store.emit_decision(late)
    o = _outcome(999, 1)
    o["finalized_at"] = "2026-08-20T00:00:00+00:00"       # closes on D+1 UTC
    store.emit_outcome(o)
    decisions, outcomes = store.load_decisions(), store.load_outcomes()

    assert upd.latest_priced_day(decisions, outcomes) == "2026-08-19"
    spend = upd.daily_exploration_spend(decisions, outcomes)
    assert spend == {"2026-08-19": pytest.approx(21 * 50.0)}, spend

    block = upd.tau_calibration(decisions, outcomes, _posterior(cfg, tmp_path), cfg)
    assert block["through_date"] == "2026-08-19"
    assert block["realised_exploration_cost"] == pytest.approx(21 * 50.0)


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


def test_a_cell_that_took_outcomes_still_counts_for_the_budget(cfg, tmp_path):
    """Routing can move after a re-initialise; a cell that learned
    something keeps a std the budget must size for."""
    floor = cfg["posterior"]["min_episodes_per_week_for_cell"]
    store = PosteriorStore.initialise(
        cfg, {"vegetables": {"mean": -1.0, "std": 0.6},
              "fruit": {"mean": -1.2, "std": 0.9}},
        {"vegetables": floor, "fruit": floor},
        path=str(tmp_path / "posterior.json"))
    cells, cell_of = store.state["cells"], dict(store.state["cell_of"])
    cells["GLOBAL"]["std"] = 0.1                   # keep GLOBAL out of the max
    cell_of["fruit"] = "GLOBAL"                    # fruit no longer routes to itself
    assert PosteriorStore.widest_active_std(cells, cell_of) == pytest.approx(0.6)
    cells["fruit"]["n_obs"] = 12                   # ...but it has evidence
    assert PosteriorStore.widest_active_std(cells, cell_of) == pytest.approx(0.9)


def test_a_weekly_batch_is_walked_day_by_day_never_graded_as_one_day(cfg, tmp_path):
    """With a weekly --apply (learning.update_cadence_days) seven closed days
    arrive at once. Grading only the latest one skipped six corrections;
    the walk takes one clipped step per day, the same walk shadow's trace
    runs (explore.walk_tau)."""
    from pricing import explore
    store = _store(cfg, tmp_path, 20, cost_each=5000.0, history_days=3,
                   history_cost=5000.0)                # four overspending days
    posterior = _posterior(cfg, tmp_path, calibrated_through=None)
    block = upd.tau_calibration(store.load_decisions(), store.load_outcomes(),
                                posterior, cfg)
    assert block["commit"] and block["through_date"] == "2026-08-19"
    # day one has no trailing IL (zero budget: held), then three halvings
    lo = cfg["exploration"]["tau_adjust_clip"][0]
    walked = [r for r in block["by_day"] if r["budget"] > 0]
    assert block["days_walked"] == 4 and len(walked) == 3, block["by_day"]
    assert block["tau_after"] == pytest.approx(block["tau_before"] * lo ** 3, rel=1e-3)
    # and the walk is the shared one, step for step
    tau_end, rows = explore.walk_tau(
        block["tau_before"], [r["day"] for r in block["by_day"]],
        lambda day, t: {r["day"]: r["spend"] for r in block["by_day"]}[day],
        {}, posterior.widest_std(), cfg)
    assert rows[0]["tau_after"] == rows[0]["tau"]        # zero budget holds


def test_calibrate_tau_commits_tau_without_touching_the_cells(cfg, tmp_path):
    """tau moves on spend, not evidence: the daily --calibrate-tau commits
    the walk while the posterior cells wait for the operator's --apply."""
    store = _store(cfg, tmp_path, 20, cost_each=5000.0)
    posterior = _posterior(cfg, tmp_path)
    cells_before = {k: dict(v) for k, v in posterior.state["cells"].items()}
    rep = upd.run(cfg, apply=False, calibrate_tau=True,
                  events_root=str(tmp_path / "events"),
                  posterior_path=str(tmp_path / "posterior.json"))
    assert rep["tau_committed"] and not rep["applied"]
    again = PosteriorStore(cfg, path=str(tmp_path / "posterior.json"))
    assert again.tau_calibrated_through() == "2026-08-19"
    assert again.tau(cfg) < posterior.tau(cfg) or again.tau(cfg) == rep["tau_calibration"]["tau_after"]
    for k, v in again.state["cells"].items():
        assert (v["mean"], v["std"], v["n_obs"]) == \
            (cells_before[k]["mean"], cells_before[k]["std"], cells_before[k]["n_obs"])
