"""tau moves on spend, at the operator gate (design 5.8)."""

import numpy as np
import pandas as pd
import pytest

from conftest import P0, decision_event, outcome_event
from events.store import EventStore
from daily import monitor as mon
from daily import update as upd
from engine.posterior import PosteriorStore


@pytest.fixture
def cfg(cfg):
    # These tests are about TAU, not about level calibration. Point the factor
    # artifact at a path that does not exist so the calibration-currency gate
    # passes trivially: otherwise they read whichever schedule happens to be
    # in artifacts/ and start failing the week after it was last fitted.
    cfg["baseline_model"] = dict(cfg["baseline_model"],
                                 calibration_factor_path="artifacts/__absent__.json")
    return cfg


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
    business = mon.business_metrics(decisions, outcomes)

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
        assert block["clipped"] is True                # the walk reports the bound
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
        _store(cfg, root, 20, cost_each=0.5)
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
    _store(cfg, tmp_path, 20, cost_each=0.5)
    _posterior(cfg, tmp_path)
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

    days, spend = upd.finalized_days(decisions, outcomes)
    assert days[-1] == "2026-08-19"
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
    from engine import explore
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


def test_a_posterior_calibrated_past_the_store_is_reported_not_an_index_error(cfg, tmp_path):
    """A store restored from an older copy (or the wrong directory) sits
    BEHIND the posterior's `tau_calibrated_through`: there is no day to walk,
    and the walk used to index an empty row list."""
    store = _store(cfg, tmp_path, 4, cost_each=1.0)
    posterior = _posterior(cfg, tmp_path, calibrated_through="2026-08-25")
    block = upd.tau_calibration(store.load_decisions(), store.load_outcomes(),
                                posterior, cfg)
    assert not block["commit"]
    assert "ahead of the store" in block["skipped"]
    assert block["through_date"] == "2026-08-19"
    assert block["tau_after"] == block["tau_before"]


def _learning_store(cfg, tmp_path, n=40, cost_each=40.0, date="2026-08-19"):
    """Three closed history days (the IL base), then `n` forced decisions on
    `date` far from the reference on a shelf deep enough to be uncensored --
    a batch that clears the information threshold once
    `learning.information_increment` is set low."""
    store = _store(cfg, tmp_path, n, cost_each=0.0, history_days=3)
    base = len(store.load_decisions())
    for i in range(base, base + n):
        store.emit_decision(_decision(i, 0.45, cost_each, date))
        store.emit_outcome(outcome_event(
            outcome_id=f"O{i}", decision_id=f"D{i}", units_sold=1,
            starting_inventory=9, ending_inventory=8, applied_price=P0 * 0.55,
            finalized_at=f"{date}T18:00:00+00:00"))
    return store


def test_dry_run_and_apply_agree_on_tau_because_the_budget_is_sized_before_the_commit(cfg, tmp_path):
    """The budget scales with the widest routed std. `--apply` commits the
    cells first, so the walk used to price past days on the POST-update std
    and disagreed with the dry run on the same store."""
    # ref std 1.0 puts a 0.6 -> 0.45 std move inside the budget's linear
    # range (neither at the floor nor capped at 1), so a mis-ordered walk
    # would price a different budget; the share is pinned because the
    # clip-bound assertion below is sized against it, not the owner's paste
    learn = dict(cfg, learning=dict(cfg["learning"], information_increment=1e-6),
                 exploration=dict(cfg["exploration"], budget_scale_ref_std=1.0,
                                  budget_share_of_il=0.01))
    reports = {}
    for mode in ("dry", "apply"):
        root = tmp_path / mode
        _learning_store(learn, root)
        _posterior(learn, root)
        reports[mode] = upd.run(learn, apply=(mode == "apply"),
                                events_root=str(root / "events"),
                                posterior_path=str(root / "posterior.json"))
    applied = reports["apply"]
    cell = applied["cells"]["vegetables"]
    assert cell["update_triggered"] and applied["applied"], applied
    assert cell["proposed_std"] < cell["std_before"], "the fixture must move the std"
    tc = applied["tau_calibration"]
    assert tc["commit"] and tc["tau_after"] != tc["tau_before"], tc
    # the walk was sized on the std the days were priced under...
    assert tc["widest_posterior_std"] == pytest.approx(cell["std_before"])
    # ...so both modes report the same tau, budget for budget
    assert tc == reports["dry"]["tau_calibration"]
    # and the fixture would have told the two apart: the post-update std
    # prices a different budget on the same days
    lo, hi = learn["exploration"]["tau_adjust_clip"]
    ratio = tc["tau_after"] / tc["tau_before"]
    assert lo + 0.01 < ratio < hi - 0.01, "the walk must not sit on a clip bound"
    # ...and the walk itself says so: the flag is read off each step's
    # ratio, never inferred from the rounded taus
    assert tc["clipped"] is False and not any(r["clipped"] for r in tc["by_day"])


def test_the_bounded_step_holds_in_the_report_unrounded(cfg, tmp_path):
    """`max_mean_step` is a safety bound checked on the REPORT: the proposed
    mean must be exactly what bounded_step returns from the report's own
    raw moments, never a re-rounded copy that can read past the cap."""
    from engine.posterior import bounded_step
    learn = dict(cfg, learning=dict(cfg["learning"], information_increment=1e-6))
    _learning_store(learn, tmp_path)
    _posterior(learn, tmp_path)
    rep = upd.run(learn, events_root=str(tmp_path / "events"),
                  posterior_path=str(tmp_path / "posterior.json"))
    c = rep["cells"]["vegetables"]
    mean, std, _ = bounded_step(c["mean_before"], c["std_before"],
                                c["raw_mean"], c["raw_std"], learn)
    assert c["proposed_mean"] == mean and c["proposed_std"] == std
    assert abs(c["proposed_mean"] - c["mean_before"]) <= learn["learning"]["max_mean_step"]
    # one figure for the batch's information: the same number under two
    # names read as two quantities
    assert "information_pending" not in c


def test_the_bounded_step_never_leaves_the_epsilon_range(cfg):
    """The sign constraint (epsilon_max) and the grid floor hold on the
    stored mean structurally, not only because the grid moments happen to
    fall inside them: a step from a belief AT a bound stops at the bound."""
    from engine.posterior import bounded_step
    pc, step = cfg["posterior"], cfg["learning"]["max_mean_step"]
    m, _, clipped = bounded_step(pc["epsilon_max"], 0.5,
                                 pc["epsilon_max"] + step / 2, 0.5, cfg)
    assert m == pc["epsilon_max"] and clipped
    m, _, _ = bounded_step(pc["epsilon_min"], 0.5, pc["epsilon_min"] - step / 2, 0.5, cfg)
    assert m == pc["epsilon_min"]
    # inside the range the step rule is untouched, in both directions
    for raw in (-1.0 - step / 2, -1.0 + step / 2):
        m, _, _ = bounded_step(-1.0, 0.5, raw, 0.5, cfg)
        assert m == pytest.approx(raw)


def test_a_long_lived_store_sees_another_writer_only_after_reload(cfg, tmp_path):
    """The hourly pricing service holds ONE PosteriorStore; the monitor writes
    a suspension into the same file from another process. The service reads
    the file once per decision batch (reload), never per decision -- so the
    handle is stale by design until it does, and current once it has."""
    service = _posterior(cfg, tmp_path)
    monitor = PosteriorStore(cfg, path=str(tmp_path / "posterior.json"))

    monitor.suspend_exploration(["price_mismatch"], "2026-08-19")
    assert service.exploration_suspended() is None      # not read yet
    assert service.reload() is service
    assert service.exploration_suspended()["since"] == "2026-08-19"

    # every cached view goes with the old state: the processed-id set too
    assert not service.is_processed("X1")
    monitor.commit_update("vegetables", -1.1, 0.55, 1, 1.0, ["X1"], applied=True)
    assert not service.is_processed("X1")
    assert service.reload().is_processed("X1")
    assert service.state["cells"]["vegetables"]["version"] == 1


# ------------------------------------------------------- exploration suspension
def _price_once(cfg, posterior, store, tau, seed=0):
    from engine.decide import decide
    # no delta_min floor: these tests are about the suspension, and the
    # owner's per-category bias map would decide which tiers are drawable
    cfg = dict(cfg, exploration=dict(cfg["exploration"], delta_min_log_bias=None))
    state = {"episode_id": "EP-S", "sku_id": "S", "fc": "F", "category": "vegetables",
             "subcategory": "leafy", "date": "2026-08-20", "hour_of_day": 12,
             "hours_remaining": 3, "q": 4, "original_price": P0, "cost": 4000.0,
             "r": 0.919, "mu_ref_path": [0.8, 0.8, 0.8], "current_discount": None}
    return decide(state, posterior, store, cfg, np.random.default_rng(seed),
                  tau, "b")


def test_a_fired_stop_suspends_exploration_until_a_human_resumes_it(cfg, tmp_path):
    """Design 5.12: fire -> suspended -> decide never explores (exploitation
    continues, tau_current None) -> --resume-exploration -> explores again.
    Nothing resumes it automatically."""
    store = _store(cfg, tmp_path, 4, cost_each=1.0)
    posterior = _posterior(cfg, tmp_path)
    tau = 1e9                                        # everything affordable
    assert posterior.exploration_suspended() is None
    assert _price_once(cfg, posterior, store, tau)["is_exploration"]

    stop = {"fired": {"price_mismatch": True, "duplicate_or_unmatched": False,
                      "scrap_deterioration_pct": "BLOCKED"},
            "suspend_exploration": True}
    rec = mon.apply_stop_conditions(stop, posterior, "2026-08-19")
    assert rec == posterior.exploration_suspended()
    assert rec["reasons"] == ["price_mismatch"] and rec["since"] == "2026-08-19"
    # persisted in the same file, visible to a fresh reader
    assert PosteriorStore(cfg, path=str(tmp_path / "posterior.json")) \
        .exploration_suspended()["since"] == "2026-08-19"

    for seed in range(5):
        evt = _price_once(cfg, posterior, store, tau, seed)
        assert not evt["is_exploration"]
        assert evt["tau_current"] is None             # the budget was not in force
        assert evt["applied_discount"] == evt["optimal_discount"]   # exploitation
    assert store.quarantined_this_run == 0            # None is a legal value

    # a stop that keeps firing keeps the FIRST since and adds the reason
    again = mon.apply_stop_conditions(
        {"fired": {"price_mismatch": True, "exploration_cost_vs_budget": True},
         "suspend_exploration": True}, posterior, "2026-08-21")
    assert again["since"] == "2026-08-19"
    assert again["reasons"] == ["exploration_cost_vs_budget", "price_mismatch"]
    # a stop that no longer fires does NOT resume
    calm = mon.apply_stop_conditions(
        {"fired": {"price_mismatch": False}, "suspend_exploration": False},
        posterior, "2026-08-22")
    assert calm is not None
    assert not _price_once(cfg, posterior, store, tau, 7)["is_exploration"]

    # the human gate clears it and reports what it cleared
    rep = upd.run(cfg, resume_exploration=True,
                  events_root=str(tmp_path / "events"),
                  posterior_path=str(tmp_path / "posterior.json"))
    assert rep["exploration_resumed"]["since"] == "2026-08-19"
    assert rep["exploration_suspended"] is None
    reloaded = PosteriorStore(cfg, path=str(tmp_path / "posterior.json"))
    assert reloaded.exploration_suspended() is None
    evt = _price_once(cfg, reloaded, store, tau)
    assert evt["is_exploration"] and evt["tau_current"] == tau
    # resuming twice is a no-op, not an error
    assert upd.run(cfg, resume_exploration=True,
                   events_root=str(tmp_path / "events"),
                   posterior_path=str(tmp_path / "posterior.json"))[
        "exploration_resumed"] is None


def test_the_flat_std_alert_reads_only_cells_that_can_learn(cfg, tmp_path):
    """An unrouted GLOBAL takes no outcome and never updates; listing it is
    a permanent false alarm. Same routed-cells notion as the budget."""
    floor = cfg["posterior"]["min_episodes_per_week_for_cell"]
    posterior = PosteriorStore.initialise(
        cfg, {"vegetables": {"mean": -1.0, "std": 0.6}}, {"vegetables": floor},
        path=str(tmp_path / "posterior.json"))
    stale = (pd.Timestamp.now("UTC") - pd.Timedelta(days=400)).isoformat()
    for rec in posterior.state["cells"].values():
        rec["updated_at"] = stale
    alert = mon.learning_metrics([], posterior, cfg)["posterior_std_flat_alert"]
    assert alert == ["vegetables"]
    assert PosteriorStore.active_cells(posterior.state["cells"],
                                       posterior.state["cell_of"]) == ["vegetables"]
    # ...until GLOBAL has evidence of its own
    posterior.state["cells"]["GLOBAL"]["n_obs"] = 3
    assert mon.learning_metrics([], posterior, cfg)["posterior_std_flat_alert"] == \
        ["GLOBAL", "vegetables"]


def test_calibrate_tau_commits_tau_without_touching_the_cells(cfg, tmp_path):
    """tau moves on spend, not evidence: the daily --calibrate-tau commits
    the walk while the posterior cells wait for the operator's --apply."""
    _store(cfg, tmp_path, 20, cost_each=5000.0)
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


# ------------------------------------------------------------ launch belief
def test_the_launch_belief_is_the_prior_pushed_by_k_stds_and_clipped(cfg):
    """posterior.cold_start_shift_std: launch = prior mean - k*std, clipped to
    the epsilon range; std untouched (evidence weighs the same); k = 0 is the
    prior as measured. Owner posture, 2026-09-05."""
    from engine.posterior import launch_belief
    c = dict(cfg, posterior=dict(cfg["posterior"], cold_start_shift_std=0.5,
                                 epsilon_min=-5.0, epsilon_max=-0.05))
    assert launch_belief(-2.0, 0.5, c) == pytest.approx(-2.25)
    assert launch_belief(-2.0, 0.2, c) == pytest.approx(-2.10)   # tight prior, small push
    assert launch_belief(-4.9, 1.0, c) == -5.0                   # clipped at the bound
    c0 = dict(c, posterior=dict(c["posterior"], cold_start_shift_std=0))
    assert launch_belief(-2.0, 0.5, c0) == -2.0
    # initialise writes the launch belief and keeps the prior mean for audit
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = PosteriorStore.initialise(
            c, {"vegetables": {"mean": -2.0, "std": 0.5}}, {"vegetables": 500},
            path=os.path.join(d, "posterior.json"))
        rec = p.state["cells"]["vegetables"]
        assert rec["mean"] == pytest.approx(-2.25) and rec["prior_mean"] == -2.0
        assert rec["std"] == 0.5 and p.state["cold_start_shift_std"] == 0.5
        # stale only while unlearned AND the launch belief moved
        prior = {"vegetables": {"mean": -2.0, "std": 0.5}}
        assert not p.launch_stale(prior, {"vegetables": 500})
        p.cfg = dict(c, posterior=dict(c["posterior"], cold_start_shift_std=1.0))
        assert p.launch_stale(prior, {"vegetables": 500})
        # production state of any kind ends it: a consumed outcome, a walked
        # tau, or a standing suspension (a re-init would silently lift it)
        for key, value in (("processed_outcome_ids", ["O1"]), ("tau", 12.0),
                           ("exploration_suspended", {"reasons": ["x"], "since": "d"})):
            p.state = {k: v for k, v in p.state.items()
                       if k not in ("processed_outcome_ids", "tau", "exploration_suspended")}
            p.state["processed_outcome_ids"] = []
            assert p.launch_stale(prior, {"vegetables": 500})
            p.state[key] = value
            assert not p.launch_stale(prior, {"vegetables": 500})


# ------------------------------------------------ windowed event quality
def _incident_store(cfg, tmp_path, days_back, incident_back, per_day=5):
    """`days_back` + 1 clean trading days ending 2026-08-19, except the day
    `incident_back` days before it, whose every outcome sold at a price
    we did not set (the price-mismatch incident)."""
    store = EventStore(cfg, root=str(tmp_path / "events"))
    i = 0
    for back in range(days_back, -1, -1):
        day = str(pd.Timestamp("2026-08-19") - pd.Timedelta(days=back))[:10]
        for _ in range(per_day):
            store.emit_decision(_decision(i, 0.30, 0.5, day))
            o = _outcome(i, 1, day)
            if back == incident_back:
                o["applied_price"] = P0 * 0.5                # not our price
            store.emit_outcome(o)
            i += 1
    return store


def _with_window(cfg, days):
    sc = dict(cfg["monitoring"]["stop_conditions"], event_quality_window_days=days)
    return dict(cfg, monitoring=dict(cfg["monitoring"], stop_conditions=sc))


def test_a_resumed_pilot_is_not_re_suspended_by_an_incident_outside_the_window(cfg, tmp_path):
    """All-time rates re-fired the price-mismatch stop the morning after a
    human resumed exploration, until enough clean history diluted the one
    incident -- and update refused --calibrate-tau / --apply for as long.
    The rates are over the trailing event_quality_window_days of TRADING
    days: a fixed integration clears both once the incident ages out."""
    window = 7
    narrow = _with_window(cfg, window)
    store = _incident_store(narrow, tmp_path, days_back=window + 2,
                            incident_back=window)          # one day outside
    posterior = _posterior(narrow, tmp_path)
    posterior.suspend_exploration(["price_mismatch"], "2026-08-12")
    assert posterior.resume_exploration()["reasons"] == ["price_mismatch"]

    report = mon.build_report(store, posterior, narrow)
    assert report["stop_conditions"]["fired"]["price_mismatch"] is False
    assert report["exploration_suspended"] is None          # NOT re-suspended
    safety = report["safety"]
    assert safety["event_quality_window_days"] == window
    assert safety["event_quality_window_start"] == "2026-08-13"
    assert safety["compared_pair_count"] == 5 * window
    assert safety["price_mismatch_count"] == 0
    upd_report = upd.run(narrow, calibrate_tau=True,
                         events_root=str(tmp_path / "events"),
                         posterior_path=str(tmp_path / "posterior.json"))
    assert "refused" not in upd_report and upd_report["tau_committed"]
    gate = upd_report["event_quality_gates"]["price_mismatch_rate"]
    assert gate["pass"] and gate["value"] == 0.0 and gate["window_days"] == window

    # the same store under a window that still holds the incident: the
    # stop fires, update refuses, and both read ONE rate
    wide = _with_window(cfg, window + 3)
    fired = mon.build_report(store, posterior, wide)
    assert fired["stop_conditions"]["fired"]["price_mismatch"] is True
    assert fired["exploration_suspended"]["reasons"] == ["price_mismatch"]
    refused = upd.run(wide, calibrate_tau=True,
                      events_root=str(tmp_path / "events"),
                      posterior_path=str(tmp_path / "posterior.json"))
    assert "price_mismatch_rate" in refused["refused"]
    from events.pairs import quality_rates
    batch = upd.collect_batch(store, posterior, wide)
    assert quality_rates(batch["event_quality"]) == quality_rates(fired["safety"])
    assert refused["event_quality_gates"]["price_mismatch_rate"]["value"] == \
        fired["safety"]["applied_vs_recommended_price_mismatch"] == \
        round(5 / (5 * (window + 3)), 4)
    # duplicates stay all-time on both sides (the store counts them on load)
    assert fired["safety"]["duplicate_outcome_count"] == \
        batch["event_quality"]["duplicate_outcome_count"] == 0


# --------------------------------------------------- the learnable batch
def test_the_learner_refuses_hours_that_opened_empty_or_restocked(cfg, tmp_path):
    """An hour that opened with nothing on the shelf sold nothing whatever
    demand was -- its censored term is log P(D >= 0) = 0, and the update
    read it as P(D >= 1). A restocked hour has no single q to empty. Both
    are what assurance already excluded; the learner now reads the same
    rule (events.pairs.learnable_with_stock) and counts what it refused."""
    from events.pairs import learnable_with_stock, match_pairs
    store = _store(cfg, tmp_path, 3, cost_each=0.0, history_days=2)   # IL base
    base = len(store.load_decisions())
    shelves = [(9, 1, 8, None),                       # learnable
               (0, 0, 0, None),                       # opened empty
               (2, 1, 6, "intraday_restock"),         # stock arrived
               (4, 1, 3, None)]                       # learnable
    for j, (start, sold, end, why) in enumerate(shelves):
        i = base + j
        store.emit_decision(_decision(i, 0.45, 40.0))
        o = outcome_event(outcome_id=f"O{i}", decision_id=f"D{i}", units_sold=sold,
                          starting_inventory=start, ending_inventory=end,
                          applied_price=P0 * 0.55, finalized_at="2026-08-19T18:00:00+00:00")
        if why:
            o["adjustment_reason"] = why
        assert store.emit_outcome(o)
    posterior = _posterior(cfg, tmp_path)
    batch = upd.collect_batch(store, posterior, cfg)
    learned = [o["outcome_id"] for _, o, _ in batch["per_cell"]["vegetables"]]
    assert learned == [f"O{base}", f"O{base + 3}"]
    assert (batch["excluded_no_stock"], batch["excluded_restock"]) == (1, 1)
    # the SAME population assurance grades r on, minus the restock it
    # leaves out of the dispersion check
    decisions, outcomes = store.load_decisions(), store.load_outcomes()
    with_stock = {o["outcome_id"] for _, o in learnable_with_stock(decisions, outcomes)}
    assert f"O{base + 1}" not in with_stock and f"O{base + 2}" in with_stock
    assert len(with_stock) == len(match_pairs(decisions, outcomes)) - 1
    # and the report carries the counts
    rep = upd.run(cfg, events_root=str(tmp_path / "events"),
                  posterior_path=str(tmp_path / "posterior.json"))
    assert rep["batch"] == {"excluded_no_stock": 1, "excluded_restock": 1}
    assert rep["cells"]["vegetables"]["forced_outcomes"] == 2


def test_calibration_currency_is_judged_on_the_latest_trading_day(cfg, tmp_path):
    """The schedule check read the UTC wall clock: a store whose latest
    priced day sits inside the last fitted week read as stale the moment
    the clock rolled into the next week. It reads the latest TRADING day
    (events.pairs.decision_day); the clock only while nothing is priced."""
    import json
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"schedule": {"by_week": {"2026-08-17": {"FRUIT": 1.0}}}}))
    c = dict(cfg, baseline_model=dict(cfg["baseline_model"],
                                      calibration_factor_path=str(cal)))
    _store(c, tmp_path, 4, cost_each=1.0)             # latest priced day 08-19
    _posterior(c, tmp_path)
    rep = upd.run(c, events_root=str(tmp_path / "events"),
                  posterior_path=str(tmp_path / "posterior.json"))
    gate = rep["event_quality_gates"]["calibration_schedule_current"]
    assert gate["pass"] and gate["value"] == "2026-08-17"
    # the wall clock is long past that week, and would have refused
    assert not upd.calibration_current(c)["pass"]


def test_a_reported_push_failure_is_not_a_price_mismatch(cfg):
    """The mismatch gate catches the silent failures the failures table
    missed (event contract): a push engineering REPORTED as failed shows
    the old price on the shelf by definition and is counted apart. The
    pilot simulator, with a fifth of pushes failing and every one reported,
    had the gate refuse every --apply."""
    from events.pairs import quality_counts, quality_rates

    decisions = [{"decision_id": f"D{i}", "date": "2026-08-19", "applied_price": 100.0}
                 for i in range(10)]
    outcomes = [{"decision_id": f"D{i}", "applied_price": 100.0,
                 "execution_status": "ok"} for i in range(8)]
    outcomes += [{"decision_id": "D8", "applied_price": 90.0,
                  "execution_status": "failed"},           # reported: not a mismatch
                 {"decision_id": "D9", "applied_price": 90.0,
                  "execution_status": "ok"}]               # silent: a mismatch
    counts = quality_counts(decisions, outcomes, cfg)
    assert counts["compared_pair_count"] == 10
    assert counts["push_failures_reported"] == 1
    assert counts["price_mismatch_count"] == 1
    assert quality_rates(counts)["price_mismatch_rate"] == 0.1
