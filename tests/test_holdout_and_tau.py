"""The hold-out slice, the shared tau derivation, and the controller trace."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from common import episodes
from common.config import load_config as _load_config
from conftest import ROOT, episode_frame
from pricing.explore import SpreadLedger, tau_next


def load_config():
    """By path, not by CWD: the assertions below are about the config this
    repo SHIPS, whatever the end-to-end tests chdir'd into."""
    return _load_config(os.path.join(ROOT, "config.yaml"))


# ---------------------------------------------------------------- window_slice

def _frame():
    """Two episodes: one opens 08-03 22:00 and runs past midnight into 08-04,
    one opens 08-04 09:00. Only the second belongs to a 08-04 hold-out."""
    rows = ([("crosses", "2026-08-03", h) for h in range(22, 24)]
            + [("crosses", "2026-08-04", h) for h in range(0, 4)]
            + [("inside", "2026-08-04", h) for h in range(9, 13)])
    return episode_frame(rows, columns=["episode_id", "date", "hour_of_day"])


def test_window_slice_takes_whole_episodes_or_none():
    out = episodes.window_slice(_frame(), "2026-08-04", "2026-08-21")
    assert set(out.episode_id) == {"inside"}
    assert len(out) == 4          # all four of its rows, none of the other's


def test_row_level_slicing_is_what_this_prevents():
    d = _frame()
    naive = d[d.date.astype(str).ge("2026-08-04")]
    # the naive cut keeps 4 orphan hours of an episode that opened the day
    # before -- a "short episode" that never existed
    assert (naive.episode_id == "crosses").sum() == 4
    assert "crosses" not in set(
        episodes.window_slice(d, "2026-08-04", None).episode_id)


def test_window_slice_assigns_every_episode_to_exactly_one_slice():
    d = _frame()
    a = episodes.window_slice(d, None, "2026-08-03")
    b = episodes.window_slice(d, "2026-08-04", None)
    assert set(a.episode_id) | set(b.episode_id) == {"crosses", "inside"}
    assert not set(a.episode_id) & set(b.episode_id)
    assert len(a) + len(b) == len(d)


def test_window_slice_is_a_noop_without_bounds():
    d = _frame()
    assert episodes.window_slice(d) is d


# ------------------------------------------------------------------- ledger

def test_solve_tau_lands_just_under_budget():
    rng = np.random.default_rng(0)
    led = SpreadLedger()
    for i in range(400):
        led.add(f"2026-08-{1 + i % 4:02d}", rng.lognormal(6, 1, rng.integers(2, 12)))
    budget = 4000.0
    tau = led.solve_tau(budget, n_days=4)
    spend = led.implied_daily_spend(tau, 4)
    assert spend <= budget
    assert spend > 0.95 * budget          # under, but not by much


def test_spend_is_monotone_in_tau():
    rng = np.random.default_rng(1)
    led = SpreadLedger()
    for i in range(200):
        led.add("d", rng.lognormal(5, 1, 6))
    seq = [led.implied_daily_spend(t, 1) for t in range(0, 2000, 100)]
    assert seq == sorted(seq)
    assert led.implied_daily_spend(0.0, 1) == 0.0


def test_spend_by_day_sums_to_the_total():
    rng = np.random.default_rng(2)
    led = SpreadLedger()
    for i in range(120):
        led.add(f"d{i % 3}", rng.lognormal(5, 1, 4))
    tau = 500.0
    by_day = led.spend_by_day(tau)
    assert len(by_day) == 3
    assert by_day.sum() == pytest.approx(led.implied_daily_spend(tau, 1))


def test_chunked_flush_does_not_lose_costs():
    rng = np.random.default_rng(3)
    a, b = SpreadLedger(), SpreadLedger()
    b._FLUSH = 64                       # force many chunks
    for _ in range(200):
        c = rng.lognormal(5, 1, 5)
        a.add("d", c)
        b.add("d", c)
    assert a.decisions == b.decisions == 200
    assert a.implied_daily_spend(1e12, 1) == pytest.approx(
        b.implied_daily_spend(1e12, 1))


def test_empty_ledger_is_not_a_crash():
    led = SpreadLedger()
    assert led.decisions == 0
    assert led.solve_tau(100.0) is None
    assert led.distribution() == {}
    assert led.quantile_of(10.0) is None


def test_entry_only_collection_understates_the_funded_tau():
    """The bug this class exists to prevent, reproduced."""
    rng = np.random.default_rng(4)
    entry_only, all_hours = SpreadLedger(), SpreadLedger()
    for ep in range(300):
        for hour in range(8):
            costs = rng.lognormal(6, 0.8, 6)
            all_hours.add("d", costs)
            if hour == 0:
                entry_only.add("d", costs)

    budget = 20000.0
    tau_entry = entry_only.solve_tau(budget, n_days=1)
    tau_all = all_hours.solve_tau(budget, n_days=1)
    assert tau_entry > tau_all
    # launching at the entry-only tau overspends on the real decision count
    assert all_hours.implied_daily_spend(tau_entry, 1) > budget


def test_replay_collects_every_decision_hour(cfg):
    from backtest.replay import policy_replay
    d = pd.concat([_replay_episode(eid) for eid in ("a", "b")])
    _, _, ledger = policy_replay(d, cfg)
    # entry-only collection would give exactly one spread per episode
    assert ledger.decisions > d.episode_id.nunique()


def _replay_episode(eid, ending_last=0, hours_remaining_last=0):
    """One closed episode in the vocabulary `_attach_predictions` emits."""
    return episode_frame(
        episode_id=eid, date="2026-05-01", hour_of_day=[9, 10, 11],
        hours_remaining=[2, 1, hours_remaining_last],
        total_discount=[0.25, 0.25, 0.30], original_price=10_000.0,
        cost=4000.0, d_ref=0.25, starting_inventory=[10, 8, 6], units_sold=2,
        ending_inventory=[8, 6, ending_last], mu_ref_hat=2.0, r=3.0, eps=-2.0,
        is_observed=True, sku_id=7, fc="FC1", category="FRUIT",
        subcategory="BERRY")


def test_the_replay_refuses_an_unclosed_episode_rather_than_aggregate_it(cfg):
    """dp_eligible implies CLOSED (prepare_data's outcome_unknown gate), so
    the replay carries no unknown-outcome branch: an unfinished episode would
    truncate the actual arm against two full-horizon simulated arms, and it
    is refused by name instead of silently excluded."""
    from backtest.replay import policy_replay
    d = pd.concat([_replay_episode("a"), _replay_episode("open", ending_last=4)])
    with pytest.raises(ValueError, match="never closed"):
        policy_replay(d, cfg)


# ------------------------------------------------------- controller trace

def test_controller_cannot_correct_before_it_has_seen_a_day(cfg=None):
    """tau_next only ever reads the day just closed, so day 1 is spent at the
    launch tau whatever that is -- which is why the trace exists."""
    cfg = cfg or load_config()
    tau0 = 10_000.0
    budget, spend = 1_000.0, 8_700.0      # the 8.7x the shadow report measured
    assert tau_next(tau0, budget, spend, cfg) == tau0 * 0.5   # clip floor
    # three halvings to get under a 2.0x stop, if spend fell proportionally
    tau, over = tau0, spend / budget
    days_over = 0
    for _ in range(5):
        if over > cfg["monitoring"]["stop_conditions"]["exploration_cost_vs_budget"]:
            days_over += 1
        tau = tau_next(tau, budget, budget * over, cfg)
        over = over / 2
    assert days_over >= 3


# --------------------------------------------- shadow defaults to holdout

def test_shadow_runs_on_the_holdout_unless_told_otherwise():
    """The default matters more than the flag."""
    import inspect
    from pipeline import shadow

    # run_shadow assumes the hold-out, so a programmatic caller gets the same
    # default the CLI does (the CLI's own default is exercised end to end)
    assert inspect.signature(shadow.run_shadow).parameters[
        "window_basis"].default == shadow.HOLDOUT_BASIS


def test_missing_holdout_config_is_an_error_not_a_silent_full_run(
        cfg, tmp_path, monkeypatch):
    """The dangerous failure is running on everything and not saying so."""
    from pipeline import shadow

    del cfg["data"]["holdout"]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    _frame().to_parquet(tmp_path / "d.parquet")
    monkeypatch.setattr(sys, "argv", [
        "shadow", "--input", str(tmp_path / "d.parquet"),
        "--config", str(tmp_path / "config.yaml")])
    with pytest.raises(SystemExit, match="--all"):   # names the alternative
        shadow.main()


def test_the_trace_says_when_a_sample_is_too_thin_to_read_daily(cfg):
    """Sampling degrades exactly one figure, and it has to name itself."""
    from pipeline.shadow import _controller_trace

    led = SpreadLedger()
    led.add("2026-08-04", [50.0, 80.0])
    trace = _controller_trace(led, {"2026-08-03": 1e6}, tau0=100.0,
                              widest_std=1.0, cfg=cfg, window_days=1,
                              sampled_episodes=10, population_episodes=100)
    assert trace["episodes_per_day_sampled"] == 10.0
    assert trace["episodes_per_day_population"] == 100.0
    # ONE denominator for both rates -- the window's calendar span -- so the
    # two are comparable (they once divided by days walked vs calendar days)
    trace = _controller_trace(led, {"2026-08-03": 1e6}, tau0=100.0,
                              widest_std=1.0, cfg=cfg, window_days=4,
                              sampled_episodes=10, population_episodes=100)
    assert trace["window_days"] == 4
    assert trace["episodes_per_day_sampled"] == 2.5
    assert trace["episodes_per_day_population"] == 25.0
    # and it must point at the figure that DOES survive sampling, or the
    # caveat leaves the reader with nothing to quote
    assert "spend_over_budget" in trace["note"]


# ------------------------------------------------- pre-launch containment

def test_pre_launch_stops_at_the_gate_window(cfg):
    from bootstrap.prepare_data import pre_launch
    end = cfg["data"]["split"]["test_end"]
    d = pd.DataFrame({
        "episode_id": ["before", "before", "straddles", "straddles",
                       "holdout", "after_holdout"],
        "date": [end, end, end, cfg["data"]["holdout"]["start"],
                 cfg["data"]["holdout"]["start"], "2026-09-01"],
        "hour_of_day": [22, 23, 23, 0, 9, 9],
    })
    kept = set(pre_launch(d, cfg).episode_id)
    assert kept == {"before", "straddles"}, \
        "an episode that OPENED before the gate window closed belongs to " \
        "pre-launch whole; one that opened after does not belong at all"


def test_config_ships_tau_initial_null(cfg):
    # it is void until re-derived: the scoping fix changed what the backtest
    # produces, so any value carried over from before is wrong
    assert cfg["exploration"]["tau_initial"] is None
    # and shadow can re-derive it: the floor is set, so the derivation runs
    assert cfg["exploration"]["tau0_derivation_min_decisions"] > 0


# ------------------------------------------------------------------- config

def test_holdout_window_is_after_the_test_window(cfg):
    h, s = cfg["data"]["holdout"], cfg["data"]["split"]
    assert h["start"] > s["test_end"], \
        "a hold-out that overlaps the gate window is not a hold-out"
    assert h["end"] > h["start"]


def test_holdout_is_disjoint_from_every_fitting_window(cfg):
    h, s = cfg["data"]["holdout"], cfg["data"]["split"]
    for lo, hi in [(s["train_start"], s["train_end"]),
                   (s["calib_start"], s["calib_end"]),
                   (s["test_start"], s["test_end"])]:
        assert not (h["start"] <= hi and lo <= h["end"])


def test_the_budget_base_is_the_trailing_realised_il(cfg):
    """The IL base for a day's budget is the mean of REALISED daily IL over
    the trailing budget_il_window_days, ending YESTERDAY -- never the same
    day's own IL."""
    from pricing.explore import trailing_daily_il

    assert cfg["exploration"]["budget_il_window_days"] == 7

    il = {f"2026-08-{d:02d}": 700.0 for d in range(1, 8)}   # 7 flat days
    # day 8's base is the mean of days 1-7; day 8's own IL must not enter
    il["2026-08-08"] = 99_999.0
    assert trailing_daily_il(il, "2026-08-08", cfg) == pytest.approx(700.0)

    # a zero-IL calendar day inside the window counts as ZERO, not skipped:
    # the budget is a share of the actual run-rate
    il2 = {"2026-08-01": 700.0, "2026-08-07": 700.0}
    assert trailing_daily_il(il2, "2026-08-08", cfg) == pytest.approx(200.0)

    # no history at all -> no budget. The conservative side to start on.
    assert trailing_daily_il({}, "2026-08-08", cfg) == 0.0


def test_the_first_shadow_day_carries_the_pre_window_trailing_base(cfg):
    """Day one of the hold-out must not read budget 0: production's launch
    day holds the trailing legacy IL of the days before it. The history is
    computed from the full frame BEFORE the window slice, keyed by close day
    -- discount plus scrap, closed episodes only."""
    import pandas as pd
    from pipeline.shadow import pre_window_il_history

    def episode(eid, day, sold, start, end_last, dp=True):
        return pd.DataFrame({
            "episode_id": [eid] * 2, "date": [day] * 2, "hour_of_day": [9, 10],
            "total_discount": [0.30] * 2, "original_price": [10_000.0] * 2,
            "offered_price": [7_000.0] * 2,
            "cost": [4000.0] * 2, "starting_inventory": [start, start - sold],
            "units_sold": [sold, 0], "ending_inventory": [start - sold, end_last],
            "dp_eligible": [dp] * 2,
        })

    d = pd.concat([
        # closes 08-01 with 2 left (write-off): IL = 3000*1... discount 0.3*10000*1 + scrap 2*4000
        episode("pre", "2026-08-01", 1, 3, 0),
        # unclosed pre-window episode: contributes NOTHING
        episode("open", "2026-08-02", 1, 3, 2),
        # closes INSIDE the window: not part of the seed
        episode("in", "2026-08-05", 1, 2, 0),
    ])
    h = pre_window_il_history(d, cfg, "2026-08-04")
    assert set(h) == {"2026-08-01"}
    assert h["2026-08-01"] == pytest.approx(0.30 * 10_000 * 1 + 2 * 4000)
    # outside the trailing window -> empty; no `before` -> empty
    assert pre_window_il_history(d, cfg, "2026-08-20") == {}
    assert pre_window_il_history(d, cfg, None) == {}

    # THE SAME POPULATION as everything the seed is scaled against (the
    # derivation's sample fraction, run_shadow's seed_scale are dp_eligible
    # counts): a closed pre-window episode the DP could not have priced
    # carries IL, and must not enter the seed
    with_ineligible = pd.concat([d, episode("cost_missing", "2026-08-02", 2, 5, 0,
                                            dp=False)])
    assert pre_window_il_history(with_ineligible, cfg, "2026-08-04") == h
    assert pre_window_il_history(with_ineligible.assign(dp_eligible=True), cfg,
                                 "2026-08-04") != h, "the extra episode carries IL"


def test_a_sold_out_early_episode_still_counts_its_shrink_as_scrap(cfg):
    """Scrap = leftover + shrink, one definition. A sold-out-early close has
    leftover 0 but can still have lost units mid-window; gating scrap on
    COMPLETED zeroed that shrink on the budget base alone."""
    from pipeline.shadow import pre_window_il_history

    # 3 units: hour 9 sells 1 and ONE VANISHES (ending 1, not 2); hour 10
    # sells the last unit -> net leftover 0, closed, SOLD_OUT_EARLY
    d = pd.DataFrame({
        "episode_id": ["e"] * 2, "date": ["2026-08-01"] * 2,
        "hour_of_day": [9, 10], "total_discount": [0.30] * 2, "offered_price": [7_000.0] * 2,
        "original_price": [10_000.0] * 2, "cost": [4000.0] * 2,
        "starting_inventory": [3, 1], "units_sold": [1, 1],
        "ending_inventory": [1, 0], "dp_eligible": [True] * 2,
    })
    from common import episodes
    assert episodes.classify(d).iloc[0] == episodes.SOLD_OUT_EARLY
    h = pre_window_il_history(d, cfg, "2026-08-04")
    assert h["2026-08-01"] == pytest.approx(0.30 * 10_000 * 2 + 1 * 4000)


def test_the_controller_holds_tau_on_a_zero_budget(cfg):
    """A zero budget from an EMPTY trailing history is an absence of signal,
    not an overspend: the controller must hold tau, not halve it. The moment
    history exists, calibration resumes."""
    from pipeline.shadow import _controller_trace
    from pricing.explore import SpreadLedger

    led = SpreadLedger()
    led.add("2026-08-04", [50.0, 80.0])          # spend exists on day 1
    led.add("2026-08-05", [50.0, 80.0])
    # no IL history at all for day 1; day 2 sees day 1's realised IL
    il_by_day = {"2026-08-04": 1_000_000.0}
    trace = _controller_trace(led, il_by_day, tau0=100.0, widest_std=1.0,
                              cfg=cfg)
    d1, d2 = trace["by_day"][0], trace["by_day"][1]
    assert d1["budget"] == 0.0 and d1["over_budget"] is None
    assert d2["tau"] == pytest.approx(100.0), \
        "tau moved on a day whose budget was 0 -- empty history is not signal"
    assert d2["budget"] > 0


# ------------------------------------------------- point-in-time calibration

def test_calibration_factors_never_see_their_own_week_or_later(cfg):
    """The whole point of the rolling schedule: week W's factors are fit on
    the trailing window ENDING STRICTLY BEFORE W."""
    import json
    import os

    import pandas as pd

    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        pytest.skip("no calibration artifact on disk")
    with open(path) as f:
        cal = json.load(f)
    sched = cal.get("schedule")
    if not sched:
        pytest.skip("calibration is not on the rolling schedule")

    weeks = sorted(sched["by_week"])
    assert weeks, "a rolling schedule with no fitted weeks is not a schedule"
    n = int(sched["trailing_weeks"])
    for w in weeks:
        start = pd.Timestamp(w)
        # the window is [start - n weeks, start): its LAST instant is strictly
        # before the week the factors are applied to
        window_end = start - pd.Timedelta(seconds=1)
        assert window_end < start
        assert (start - (start - pd.Timedelta(weeks=n))).days == 7 * n
    # (the applier's own-week selection and frozen fallback are exercised
    # in test_a_level_shift_does_not_leak_into_its_own_weeks_factor)


def test_both_harnesses_get_point_in_time_factors_without_their_own_code():
    """backtest and shadow must not each re-implement factor selection: they
    call predict_mu_ref on frames carrying `date`, so the schedule reaches
    them through the one applier. A second copy would drift."""
    import inspect
    from backtest import replay
    from pipeline import shadow

    # one prediction path: replay.predict_frame calls the applier, shadow
    # predicts through predict_frame (shadow's own copy of extend/lookup/
    # predict is gone)
    assert "predict_mu_ref(" in inspect.getsource(replay.predict_frame)
    assert "predict_frame(" in inspect.getsource(shadow._prepare_items)
    for mod, name in ((replay, "backtest.replay"), (shadow, "pipeline.shadow")):
        src = inspect.getsource(mod)
        # `by_week` is deliberately NOT banned: fidelity has its own weekly
        # series. What must not appear is the calibration schedule's own names
        for banned in ("calibration_schedule", "_factor_vector",
                       "calibration_factor_path"):
            assert banned not in src, (
                f"{name} reaches into the calibration schedule itself -- "
                "factor selection belongs to BaselineModel alone")


def test_a_level_shift_does_not_leak_into_its_own_weeks_factor(tmp_path, cfg):
    """The functional half of the point-in-time guarantee."""
    import json

    import pandas as pd

    from bootstrap.train_baseline import BaselineModel

    quiet, jumped, fallback = 1.00, 1.80, 1.33
    artifact = {
        "grain": "category",
        "factors": {"FRUIT": fallback},           # frozen set, distinct value
        "schedule": {
            "mode": "rolling_trailing", "trailing_weeks": 4,
            "by_week": {
                "2026-07-06": {"FRUIT": quiet},   # before the jump
                "2026-07-13": {"FRUIT": quiet},   # THE JUMP WEEK: cannot see it
                "2026-07-20": {"FRUIT": jumped},  # first week that can
            },
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact))

    cfg = dict(cfg, baseline_model=dict(cfg["baseline_model"],
                                        calibration_factor_path=str(path)))
    model = BaselineModel.__new__(BaselineModel)      # applier only, no booster
    model.cfg = cfg
    model.calibration = artifact["factors"]
    model.calibration_grain = "category"
    model.calibration_schedule = artifact["schedule"]["by_week"]
    model._reset_calibration_counters()

    rows = pd.DataFrame({
        "category": ["FRUIT"] * 4,
        "date": ["2026-07-15",   # DURING the jump week
                 "2026-07-22",   # the week after, which may see it
                 "2026-07-08",   # the week before
                 "2026-06-29"],  # no fitted week -> the frozen fallback
    })
    f = model._factor_vector(rows)

    assert f[0] == pytest.approx(quiet), \
        "the jump leaked into the factor applied during its own week"
    assert f[1] == pytest.approx(jumped), \
        "the week after the jump never picked it up"
    assert f[2] == pytest.approx(quiet)
    assert f[3] == pytest.approx(fallback), \
        "an unfitted week must fall back to the frozen set, never borrow " \
        "a later week's factors"


def test_running_past_the_calibration_schedule_is_reported_not_silent():
    """The schedule only covers weeks it was fitted on. A row past its end
    takes the frozen fallback and nothing errors -- so production drifts back
    onto stale factors the moment the weekly re-fit is missed, invisibly.
    That is the failure the point-in-time change exists to remove, so it has
    to be counted and named."""
    import pandas as pd

    from bootstrap.train_baseline import BaselineModel

    model = BaselineModel.__new__(BaselineModel)
    model.cfg = load_config()
    model.calibration = {"FRUIT": 1.33}
    model.calibration_grain = "category"
    model.calibration_schedule = {"2026-07-06": {"FRUIT": 1.10}}
    model._reset_calibration_counters()

    rows = pd.DataFrame({"category": ["FRUIT"] * 3,
                         "date": ["2026-07-08",     # inside the schedule
                                  "2026-07-20",     # PAST its end
                                  "2026-07-27"]})   # past its end
    f = model._factor_vector(rows)
    assert f[0] == pytest.approx(1.10)
    assert f[1] == pytest.approx(1.33) and f[2] == pytest.approx(1.33)

    cov = model.calibration_coverage()
    assert cov["rows_on_schedule"] == 1
    assert cov["rows_on_fallback"] == 2
    assert cov["weeks_after_schedule_end"] == ["2026-07-20", "2026-07-27"]
    assert cov["verdict"].startswith("STALE FACTORS IN USE")
    # the verdict must name the remedy, not merely report the count
    assert "--fit-calibration" in cov["verdict"]

    # a run entirely inside the schedule says so, and says nothing alarming
    model._cal_rows_fallback, model._cal_fallback_weeks = 0, set()
    assert model.calibration_coverage()["verdict"].startswith("OK")


def test_apply_refuses_when_the_weekly_refit_was_missed(tmp_path, cfg):
    """Detection after the fact is not enough. Learning from prices set on
    stale factors banks evidence about a model that is not the one running,
    so a schedule that no longer reaches today is a HARD gate on --apply --
    the same standing as a duplicate-event breach."""
    import json

    from pipeline.update import calibration_current

    path = tmp_path / "calibration.json"
    cfg = dict(cfg, baseline_model=dict(cfg["baseline_model"],
                                        calibration_factor_path=str(path)))
    path.write_text(json.dumps({
        "grain": "category", "factors": {"FRUIT": 1.2},
        "schedule": {"mode": "rolling_trailing", "trailing_weeks": 4,
                     "by_week": {"2026-07-06": {"FRUIT": 1.1},
                                 "2026-07-13": {"FRUIT": 1.1}}}}))

    # inside the schedule -> passes
    assert calibration_current(cfg, today="2026-07-15")["pass"]
    # the week AFTER the last fitted one -> refuses, and names the remedy
    late = calibration_current(cfg, today="2026-07-21")
    assert not late["pass"]
    assert "--fit-calibration" in late["note"]

    # static calibration is a legitimate configuration: nothing to outrun
    path.write_text(json.dumps({"grain": "category", "factors": {"FRUIT": 1.2}}))
    assert calibration_current(cfg, today="2027-01-01")["pass"]

    # and no artifact at all means factors are 1.0 -- also not a failure
    path.unlink()
    assert calibration_current(cfg, today="2027-01-01")["pass"]


def test_a_partial_trailing_window_is_counted_not_passed_off_as_full(cfg):
    """`trailing_weeks: 4` does not mean every week HAS four behind it."""
    import json
    import os

    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        pytest.skip("no calibration artifact on disk")
    sched = (json.load(open(path)).get("schedule") or {})
    if not sched.get("by_week"):
        pytest.skip("calibration is not on the rolling schedule")

    assert "weeks_on_partial_window" in sched, \
        "a short trailing window must be counted, not silently labelled full"
    for row in sched["weeks_on_partial_window"]:
        assert row["weeks_in_window"] < sched["trailing_weeks"]
        assert row["week"] in sched["by_week"], \
            "a partial week is still fitted -- it is flagged, not dropped"


def test_the_gate_freezes_calibration_even_though_the_schedule_runs_past_it(cfg):
    """Two questions, one artifact, and they must not be confused."""
    import json
    import os

    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        pytest.skip("no calibration artifact on disk")
    sched = (json.load(open(path)).get("schedule") or {})
    if not sched.get("by_week"):
        pytest.skip("calibration is not on the rolling schedule")

    gate_start = pd.Timestamp(cfg["data"]["split"]["test_start"])
    assert sched.get("gate_freezes_at") == str(gate_start.date()), (
        "the artifact must record where the gate freezes, or coverage cannot "
        "tell a deliberate freeze from production running onto stale factors")

    # (that the fidelity gate does the freezing is exercised in
    # test_fidelity_grades_the_frozen_artifact_and_reports_the_refit_beside_it)

    # frozen rows take the anchor, scheduled rows take their own week
    from bootstrap.train_baseline import BaselineModel
    m = BaselineModel.__new__(BaselineModel)
    m.calibration_grain = "category"
    m.calibration = {"A": 2.0}
    m.calibration_schedule = {"2026-07-27": {"A": 5.0},
                              "2026-08-03": {"A": 9.0}}
    m._reset_calibration_counters()
    frame = pd.DataFrame({"date": ["2026-07-28", "2026-08-03"],
                          "category": ["A", "A"]})

    m.freeze_calibration_from(None)
    assert list(m._factor_vector(frame)) == [5.0, 9.0], \
        "unfrozen, every row takes its own week -- production's mechanism"

    m._reset_calibration_counters()
    m.freeze_calibration_from(gate_start)
    assert list(m._factor_vector(frame)) == [2.0, 2.0], \
        "frozen at the gate, every graded row takes the anchor"
    assert m._cal_rows_frozen == 2 and m._cal_rows_fallback == 0, \
        "a deliberate freeze must not be counted as a fallback gap"


def test_convergence_check_flags_drift_and_never_commits_the_resolve(
        tmp_path, monkeypatch):
    """The f <-> r cycle is broken by iteration, and this is the assertion
    that the iteration settled. Dry run is load-bearing: committing the
    re-solve while prior/dispersion lag it would CREATE the inconsistency
    being tested for."""
    import copy

    import bootstrap.train_baseline as tb

    cfg = copy.deepcopy(load_config())
    path = str(tmp_path / "cal.json")
    cfg["baseline_model"]["calibration_factor_path"] = path
    cfg["baseline_model"]["calibration_convergence_tol_log"] = 0.02
    old = {"factors": {"A": 1.0, "B": 1.2},
           "schedule": {"by_week": {"2026-07-06": {"A": 1.05}}}}
    json.dump(old, open(path, "w"))

    def fake(art):
        def _fit(d, c):
            json.dump(art, open(path, "w"))
        return _fit

    # a factor moved 1.2 -> 1.3 under the current r/prior -> NOT CONVERGED,
    # named, and the disk artifact still holds iteration k
    monkeypatch.setattr(tb, "fit_level_calibration", fake(
        {"factors": {"A": 1.0, "B": 1.3},
         "schedule": {"by_week": {"2026-07-06": {"A": 1.05}}}}))
    block = tb.check_calibration_convergence(None, cfg)
    assert not block["converged"]
    assert block["worst_cell"] == "anchor:B"
    assert abs(block["max_abs_dlog"] - abs(np.log(1.3 / 1.2))) < 1e-5
    disk = json.load(open(path))
    assert disk["factors"] == old["factors"], "dry run must restore"
    assert disk["convergence"]["converged"] is False

    # identical re-solve -> converged; a schedule week's cell moving is
    # caught the same way as the anchor's
    monkeypatch.setattr(tb, "fit_level_calibration", fake(old))
    assert tb.check_calibration_convergence(None, cfg)["converged"]
    monkeypatch.setattr(tb, "fit_level_calibration", fake(
        {"factors": {"A": 1.0, "B": 1.2},
         "schedule": {"by_week": {"2026-07-06": {"A": 1.30}}}}))
    block = tb.check_calibration_convergence(None, cfg)
    assert not block["converged"] and block["worst_cell"] == "2026-07-06:A"

    # a cell appearing or vanishing is never averaged away
    monkeypatch.setattr(tb, "fit_level_calibration", fake(
        {"factors": {"A": 1.0}, "schedule": {"by_week": {}}}))
    block = tb.check_calibration_convergence(None, cfg)
    assert not block["converged"]
    assert "anchor:B" in block["cells_appeared_or_gone"]


def test_rho_is_fit_on_the_calib_window_not_the_full_frame():
    """rho once read the whole input frame: ~83% training rows, where the
    model fits its own residuals and between-episode variance reads small
    (fixture: in-train rho 0.081 vs out-of-sample 0.230) -- plus, on a longer
    extract, rows past test_end (hard rule 16). An understated rho understates
    deff, and deff deflates every posterior update. calib is the one window
    both out-of-train and pre-gate, and r already lives there."""
    path = load_config()["dispersion"]["rho_path"]
    if not os.path.exists(path):
        pytest.skip("no rho artifact on disk")
    art = json.load(open(path))
    if "fit_window" in art:                 # artifact written post-change
        assert art["fit_window"] == "calib"


def test_convergence_carries_its_trajectory_and_the_worst_cell_s_evidence():
    """A single reading cannot tell a contracting loop from a stuck one, and
    an unweighted max cannot tell an unsettled chain from one thin cell."""
    import copy

    cfg = copy.deepcopy(load_config())
    cfg["baseline_model"]["calibration_convergence_tol_log"] = 0.02
    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        pytest.skip("no calibration artifact on disk")
    block = (json.load(open(path)).get("convergence") or {})
    if not block:
        pytest.skip("convergence never run")

    assert isinstance(block.get("history"), list) and block["history"]
    assert len(block["history"]) <= 6, "the history is bounded"
    assert block["history"][-1] == block["max_abs_dlog"], \
        "the last reading must be this run's"
    # the worst cell's evidence is reported, so a shrinkage-dominated cell is
    # distinguishable from a genuinely unsettled loop
    assert "worst_cell_anchor_rows" in block


def test_the_prior_fast_path_drops_only_what_cannot_move_the_fixed_point():
    """`fold_spread` only widens the std FLOOR -- while the loop compares
    FACTORS, which follow `r`, which is fitted at the prior MEAN. Skipping it
    is most of the prior's in-loop cost, now that `design_comparison` (which
    fed nothing at all) is gone rather than merely skipped."""
    import inspect

    from bootstrap import prior_density

    assert "fast" in inspect.signature(prior_density.estimate).parameters
    assert "design_comparison" not in inspect.getsource(prior_density), \
        "the alternative-design block was deleted, not made conditional"


def test_the_trailing_fit_window_keeps_episodes_whole_at_the_week_seam():
    """Both schedule loops cut the trailing window by ROW week, so an
    episode opening Sunday and closing Monday lost its Monday rows -- and
    the artifact schedule and shadow's re-fit solved on different rows."""
    d = pd.DataFrame({
        "episode_id": ["a", "a", "b", "b"],
        "date": ["2026-08-09", "2026-08-10",       # Sun -> Mon (week seam)
                 "2026-08-10", "2026-08-11"],      # opens in the fit week
        "hour_of_day": [23, 0, 9, 10],
    })
    window, weeks = episodes.trailing_weeks_window(d, "2026-08-10", 1)
    assert sorted(window.episode_id.unique()) == ["a"]
    assert len(window) == 2 and weeks == 1
    empty, none = episodes.trailing_weeks_window(d, "2026-08-03", 1)
    assert len(empty) == 0 and none == 0


def test_anchor_rows_are_one_mask():
    """Five sites spelled `(total_discount - d_ref).abs() <= tier_step / 2`
    by hand; the fit, the fidelity ratio and the gate share must agree on
    what an anchor row is."""
    d = pd.DataFrame({"total_discount": [0.30, 0.32, 0.33, 0.28],
                      "d_ref": [0.30] * 4})
    assert episodes.is_anchor_row(d, 0.05).tolist() == [True, True, False, True]


def test_budget_sweep_trades_count_for_depth_from_one_ledger():
    """The forced RATE is the budget's (tau); the delta_min multiple sets
    which tiers are drawn. From one ledger, a smaller share forces less at
    the same depth and a deeper multiple forces less but deeper -- so the
    owner reads the table instead of re-running shadow per candidate. A
    multiple below the one in force cannot be recovered and says so."""
    rng = np.random.default_rng(5)
    led = SpreadLedger()
    n_dec = 600
    for i in range(n_dec):
        moves = np.sort(rng.uniform(0.05, 0.6, 6))        # distance from ref
        costs = 40.0 * moves ** 2 * rng.lognormal(6, 0.3)  # deeper = dearer
        keep = moves >= 0.10                                # the floor in force
        led.add(f"d{i % 5}", costs[keep], moves[keep], delta_min=0.10)
    sw = led.sweep(daily_budget=20000.0, n_days=5, n_decisions=n_dec,
                   share_in_force=0.01, multiple_in_force=1.0,
                   shares=[0.0025, 0.005, 0.01], multiples=[0.5, 1.0, 2.0])
    rows = {(r["budget_share_of_il"], r["delta_min_bias_multiple"]): r
            for r in sw["rows"]}
    ref = rows[(0.01, 1.0)]
    assert ref["in_force"] and ref["information_rel"] == 1.0
    assert ref["implied_daily_spend"] <= ref["daily_budget"] == 20000.0
    # less budget: lower rate, same depth (within the draw's noise)
    half, quarter = rows[(0.005, 1.0)], rows[(0.0025, 1.0)]
    assert quarter["forced_rate"] < half["forced_rate"] < ref["forced_rate"]
    assert half["daily_budget"] == 10000.0
    assert abs(half["mean_log_move_forced"] - ref["mean_log_move_forced"]) < 0.05
    assert quarter["information_rel"] < half["information_rel"] < 1.0
    # deeper floor at the same budget: fewer, deeper forced moves
    deep = rows[(0.01, 2.0)]
    assert deep["forced_rate"] < ref["forced_rate"]
    assert deep["mean_log_move_forced"] > ref["mean_log_move_forced"] + 0.05
    # a shallower floor than recorded is not a number
    assert "forced_rate" not in rows[(0.01, 0.5)] and "never recorded" in rows[(0.01, 0.5)]["note"]
    # with no floor in force the multiples are inert, and the table says so
    flat = SpreadLedger()
    for i in range(100):
        flat.add("d", rng.lognormal(5, 1, 4))
    sw0 = flat.sweep(1000.0, 1, 100, 0.01, 1.0, [0.01], [1.0, 2.0])
    assert sw0["delta_min_in_force"] is False and "reads the same" in sw0["note"]
    a, b = (r for r in sw0["rows"])
    assert a["forced_rate"] == b["forced_rate"]


# ================================================= the harnesses, end to end
#
# A BaselineModel applier over a constant base rate (no booster), so shadow
# and the backtest run their real code -- decide(), the DP, the ledger, the
# event store, the level-factor applier -- on a frame small enough to reason
# about. Every figure below is arithmetic on the fixture, nothing more.

import datetime as dt

from bootstrap.train_baseline import BaselineModel
from pricing import explore as explore_mod
from pricing.posterior import PosteriorStore

WINDOW_START, WINDOW_END = "2026-08-10", "2026-08-28"     # config's hold-out


class _Applier(BaselineModel):
    """BaselineModel's factor applier over a constant mu_ref -- the real
    schedule/freeze/coverage code, no LightGBM."""

    def __init__(self, cfg, base_mu=2.0, anchor=None, schedule=None):
        self.cfg = cfg
        self.calibration = dict(anchor or {"FRUIT": 1.0})
        self.calibration_grain = "category"
        self.calibration_schedule = schedule
        self.calibration_stops_at = None
        self.version = "applier-only"
        self.base_mu = base_mu
        self._reset_calibration_counters()

    def predict_mu_ref(self, d, raw=False):
        mu = np.full(len(d), float(self.base_mu))
        return mu if raw else mu * self._factor_vector(d)


def _hours(eid, day, n, q0=6, sold=1, disc=0.30, tail=0, hour0=9, dp=True,
           sku=7, category="FRUIT"):
    """One episode in the prepared-frame vocabulary: `n` observed hours
    opening `day` at `hour0`, closed by the write-off sentinel on its last
    row; `tail` > 0 leaves window hours uncovered (extend_to_window adds
    them). `dp=False` marks it outside the dp_eligible population."""
    start = [q0 - sold * i for i in range(n)]
    end = [q - sold for q in start]
    end[-1] = 0
    return pd.DataFrame({
        "episode_id": [eid] * n, "date": [dt.date.fromisoformat(day)] * n,
        "hour_of_day": [hour0 + i for i in range(n)],
        "hours_remaining": [n - 1 - i + tail for i in range(n)],
        "sku_id": [sku] * n, "fc": ["FC1"] * n, "category": [category] * n,
        "subcategory": ["BERRY"] * n,
        "starting_inventory": start, "ending_inventory": end,
        "units_sold": [sold] * n, "total_discount": [disc] * n,
        "original_price": [10_000.0] * n,
        "offered_price": [10_000.0 * (1 - disc)] * n, "cost": [4000.0] * n,
        "d_ref": [0.30] * n, "dp_eligible": [dp] * n, "episode_eligible": [dp] * n,
    })


def _shadow_frame():
    """Pre-window week 08-03..08-09 and the hold-out from 08-10. Both spans
    carry a date gap and an episode whose window runs past its last observed
    hour (extend_to_window adds a day), plus a dp-INELIGIBLE episode with IL."""
    return pd.concat([
        _hours("p1", "2026-08-04", 4), _hours("p2", "2026-08-06", 4, q0=8),
        _hours("p3", "2026-08-08", 2, hour0=22, tail=3),      # -> 08-09 00:00..02:00
        _hours("x1", "2026-08-05", 4, dp=False),              # IL, not dp_eligible
        _hours("w1", "2026-08-10", 4), _hours("w2", "2026-08-11", 4, q0=8),
        _hours("w3", "2026-08-13", 2, hour0=22, tail=3),      # -> 08-14
        _hours("x2", "2026-08-12", 4, dp=False),
    ], ignore_index=True)


def _harness_cfg(cfg, tmp_path):
    """The shipped config pointed at throwaway artifacts."""
    r_path = tmp_path / "r_lookup.json"
    r_path.write_text(json.dumps({"fallback_order": ["subcategory", "category",
                                                     "global"],
                                  "subcategory": {}, "category": {},
                                  "global": 1.0}))
    cfg = dict(cfg)
    cfg["dispersion"] = dict(cfg["dispersion"], r_lookup_path=str(r_path))
    cfg["posterior"] = dict(cfg["posterior"], path=str(tmp_path / "posterior.json"))
    cfg["events"] = dict(cfg["events"], shadow_store_dir=str(tmp_path / "shadow_events"))
    cfg["exploration"] = dict(cfg["exploration"], tau0_derivation_min_decisions=1)
    PosteriorStore.initialise(cfg, {"FRUIT": {"mean": -1.2, "std": 0.5}},
                              {"FRUIT": 1000}, path=cfg["posterior"]["path"])
    return cfg


def _run_shadow(cfg, frame, model, monkeypatch, refit=None, **kw):
    """pipeline.shadow.main's wiring, on `frame`, with the applier."""
    from pipeline import shadow
    monkeypatch.setattr(shadow, "BaselineModel", lambda c: model)
    monkeypatch.setattr(shadow, "weekly_refit_schedule",
                        refit or (lambda *a, **k: ({}, [])))
    history = shadow.pre_window_il_history(frame, cfg, WINDOW_START)
    window = episodes.window_slice(frame, WINDOW_START, WINDOW_END)
    kw.setdefault("max_episodes", 0)
    return shadow.run_shadow(window, cfg, prior_il_by_day=history,
                             pre_window_frame=frame, window_start=WINDOW_START,
                             **kw)


def test_every_per_day_figure_divides_by_the_unsampled_unextended_span(
        cfg, tmp_path, monkeypatch):
    """n_days once came from three places: the extended frame (a synthetic
    tail adds a day), the sampled frame (a sample shrinks the span), the
    calendar. It is the window population's span, computed before either."""
    cfg = _harness_cfg(cfg, tmp_path)
    frame = _shadow_frame()
    report = _run_shadow(cfg, frame, _Applier(cfg), monkeypatch)
    b, td = report["exploration_budget_would_be"], report["tau_initial_derivation"]
    # window: 08-10..08-13 opened (4 days); w3's tail reaches 08-14 (5)
    assert b["days"] == 4 == b["tau_controller_trace"]["window_days"]
    assert report["window"]["date_max"] == "2026-08-14", \
        "the extended frame does reach a fifth day -- days must not read it"
    # pre-window: 08-04..08-08 opened (5 days); p3's tail reaches 08-09
    assert not td["fallback"] and td["days"] == 5
    assert b["implied_daily_spend"] == pytest.approx(
        report["exploration_would_be"]["would_be_cost_total"] / 4, abs=0.1)

    # a sample of one episode shrinks neither span
    sampled = _run_shadow(cfg, frame, _Applier(cfg), monkeypatch, max_episodes=1)
    assert sampled["window"]["sampled"] and sampled["window"]["episodes"] == 1
    assert sampled["exploration_budget_would_be"]["days"] == 4
    assert sampled["tau_initial_derivation"]["days"] == 5


def test_the_aggregate_budget_is_the_mean_over_the_windows_decision_days(
        cfg, tmp_path, monkeypatch):
    """The gate's daily_budget once averaged budget_today over EVERY il_by_day
    key -- the seven pre-window seed days included, the first of which has no
    trailing history and reads as a zero budget. It is the mean over the days
    the window decided on, the same days the controller trace walks."""
    from pipeline.shadow import _mean_daily_budget

    seed = {f"2026-08-0{d}": 700.0 for d in range(3, 10)}     # 7 seed days
    decision_days = ["2026-08-10", "2026-08-11"]
    il = dict(seed, **{"2026-08-10": 900.0})
    want = np.mean([explore_mod.budget_today(
        explore_mod.trailing_daily_il(il, day, cfg), 1.0, cfg)
        for day in decision_days])
    assert _mean_daily_budget(decision_days, il, 1.0, cfg) == pytest.approx(want)
    # the bug, reproduced: the seed days drag the mean down (08-03 budgets 0)
    assert _mean_daily_budget(sorted(il), il, 1.0, cfg) < want

    # and the report agrees with its own trace, day for day
    cfg = _harness_cfg(cfg, tmp_path)
    report = _run_shadow(cfg, _shadow_frame(), _Applier(cfg), monkeypatch)
    b = report["exploration_budget_would_be"]
    tr = b["tau_controller_trace"]
    assert tr["days_truncated"] == 0
    assert b["daily_budget"] == pytest.approx(
        np.mean([r["budget"] for r in tr["by_day"]]), abs=0.1)
    assert "decision days" in b["budget_basis"]


def test_the_pre_window_seed_is_scaled_to_the_sample(cfg, tmp_path, monkeypatch):
    """The seed is measured on the whole dp_eligible frame and everything it
    is compared against is sample-scale: day one's budget must read the seed
    times the sample fraction, or a 1-of-3 sample budgets 3x too much."""
    from pipeline.shadow import pre_window_il_history
    cfg = _harness_cfg(cfg, tmp_path)
    frame = _shadow_frame()
    seed = pre_window_il_history(frame, cfg, WINDOW_START)
    assert seed and all(v > 0 for v in seed.values())
    report = _run_shadow(cfg, frame, _Applier(cfg), monkeypatch, max_episodes=1)
    b = report["exploration_budget_would_be"]
    assert b["trailing_basis_seed_scale"] == pytest.approx(1 / 3)
    assert b["trailing_basis_seeded_days"] == len(seed)
    day1 = b["tau_controller_trace"]["by_day"][0]
    scaled = {k: v / 3 for k, v in seed.items()}
    std = PosteriorStore(cfg).widest_std()
    assert day1["budget"] == pytest.approx(explore_mod.budget_today(
        explore_mod.trailing_daily_il(scaled, day1["day"], cfg), std, cfg), abs=0.1)
    # a full run is unaffected: scale is exactly 1
    full = _run_shadow(cfg, frame, _Applier(cfg), monkeypatch)
    assert full["exploration_budget_would_be"]["trailing_basis_seed_scale"] == 1.0


def test_shadow_grades_every_window_row_on_the_frozen_anchor(
        cfg, tmp_path, monkeypatch):
    """frozen_anchor means the anchor. On the hold-out every row is past the
    schedule and takes it anyway; on a window that overlaps the schedule
    (--all, an explicit range) the rows would carry their own week's factors
    and the weekly_refit rescale (anchor -> re-fit) would be wrong for them.
    Shadow freezes the model at the window start, so both readings sit on
    the anchor basis, and calibration_coverage reads the freeze as the
    gate's -- not as STALE FACTORS IN USE, which the hold-out produced by
    construction."""
    cfg = _harness_cfg(cfg, tmp_path)
    frame = _shadow_frame()
    week = "2026-08-10"
    holdout_like = _Applier(cfg, schedule={"2026-08-03": {"FRUIT": 1.0}})
    overlapping = _Applier(cfg, schedule={"2026-08-03": {"FRUIT": 1.0},
                                          week: {"FRUIT": 2.0}})

    def refit(*a, **k):
        return ({week: {"FRUIT": 2.0}},
                [{"week": week, "fitted": True, "weeks_in_window": 1,
                  "partial": False}])

    a = _run_shadow(cfg, frame, holdout_like, monkeypatch, refit=refit)
    b = _run_shadow(cfg, frame, overlapping, monkeypatch, refit=refit)
    ra, rb = a["calibration_regimes"], b["calibration_regimes"]
    assert ra["frozen_anchor"] == rb["frozen_anchor"] > 0, \
        "the week's schedule factor leaked into the frozen-anchor reading"
    # the re-fit doubles mu on the rescaled rows: the censored ratio falls
    assert ra["weekly_refit"] == rb["weekly_refit"] < ra["frozen_anchor"]
    assert ra["rows_rescaled"] > 0 and ra["spread"] < 0
    for report in (a, b):
        cov = report["artifact_versions"]["calibration_coverage"]
        assert cov["verdict"].startswith("OK"), cov["verdict"]
        assert cov["frozen_from"] == WINDOW_START
        assert cov["rows_frozen_at_anchor"] > 0
        assert cov["weeks_after_schedule_end"] == []


def test_the_parent_commits_every_event_through_the_real_store(
        cfg, tmp_path, monkeypatch):
    """Workers buffer; the store the gate measures sees every decision and
    outcome exactly once, serial or parallel."""
    from events.store import EventStore
    cfg = _harness_cfg(cfg, tmp_path)
    for workers, root in ((None, "serial"), (2, "parallel")):
        report = _run_shadow(cfg, _shadow_frame(), _Applier(cfg), monkeypatch,
                             events_root=str(tmp_path / root), workers=workers)
        store = EventStore(cfg, root=str(tmp_path / root))
        assert len(store.load_decisions()) == report["decision_count"] > 0
        assert len(store.load_outcomes()) == report["outcome_count"]
        assert report["shadow_gate"]["event_completeness"]["value"] == 1.0
        assert all(o["execution_status"] == "shadow_not_applied"
                   for o in store.load_outcomes())


def test_the_tau_cross_check_uses_one_day_count_on_both_sides(cfg):
    """derive_tau_initial averaged IL over the days with episodes while the
    ledger divided spend by the CALENDAR span -- which, on the pre-launch
    frame, crosses the exclusion gap. Both sides now count the days that
    traded, so the bisection lands the spend under the budget it was given."""
    from backtest.replay import derive_tau_initial
    rng = np.random.default_rng(6)
    led = SpreadLedger()
    days = ["2026-07-01", "2026-07-02", "2026-07-10"]         # an 8-day gap
    for i in range(300):
        led.add(days[i % 3], rng.lognormal(6, 1, 6))
    ep = pd.DataFrame({"date": [days[i % 3] for i in range(30)],
                       "actual_il": 5_000.0})
    out = derive_tau_initial(led, ep, cfg, launch_std=1.0)
    assert out["days"] == 3
    budget = explore_mod.budget_today(ep.actual_il.sum() / 3, 1.0, cfg)
    assert out["daily_budget"] == pytest.approx(budget, abs=0.1)
    # the bisection lands just under the budget it was given -- on the SAME
    # day count (tau is reported to 2dp and spend steps at every cost, so
    # the re-computation is close, not exact)
    assert 0.9 * out["daily_budget"] < out["implied_daily_spend"] <= out["daily_budget"]
    tau = out["tau_initial"]
    s3 = led.implied_daily_spend(tau, 3)
    assert out["implied_daily_spend"] == pytest.approx(s3, rel=0.05)
    # the calendar span (10 days) would have read the same spend at 3/10 of
    # its size: a tau that overspends the trading day 3.3x, reported as within
    assert led.implied_daily_spend(tau, 10) == pytest.approx(s3 * 3 / 10)


def test_predict_frame_is_the_one_extend_lookup_predict_path(cfg, tmp_path):
    """Shadow and the replay each carried a copy of extend -> r -> mu_ref."""
    from backtest.replay import predict_frame
    cfg = _harness_cfg(cfg, tmp_path)
    r_lookup = json.load(open(cfg["dispersion"]["r_lookup_path"]))
    d = predict_frame(_hours("e", "2026-08-10", 2, hour0=22, tail=3), cfg,
                      _Applier(cfg, base_mu=1.7), r_lookup)
    assert len(d) == 5 and d.is_observed.tolist() == [True, True, False, False, False]
    assert (d.r == 1.0).all() and (d.mu_ref_hat == 1.7).all()
    assert d.hours_remaining.tolist() == [4, 3, 2, 1, 0]


def test_weekly_refit_fits_each_week_on_data_strictly_before_it(
        cfg, tmp_path, monkeypatch):
    """The re-fit at week k may read only weeks < k -- no look-ahead inside
    the replay -- through the one factor solve."""
    import bootstrap.train_baseline as tb
    from pipeline import shadow
    cfg = _harness_cfg(cfg, tmp_path)
    cfg["baseline_model"] = dict(cfg["baseline_model"],
                                 calibration_fit_trailing_weeks=1)
    seen = []

    def solve(window, model, *args):
        seen.append(window.copy())
        return {"FRUIT": 1.1}, {}, 1.1

    monkeypatch.setattr(tb, "_solve_level_factors", solve)
    frame = pd.concat([_hours("a", "2026-08-05", 3), _hours("b", "2026-08-12", 3),
                       _hours("c", "2026-08-19", 3)])
    table, cov = shadow.weekly_refit_schedule(
        frame, cfg, _Applier(cfg), {}, "2026-08-10", "2026-08-20")
    assert sorted(table) == ["2026-08-10", "2026-08-17"]
    assert len(seen) == 2
    for week, window in zip(sorted(table), seen):
        assert window.date.astype(str).max() < week
    assert all(c["fitted"] for c in cov)


def _fidelity_frame(cfg):
    """Anchor rows across the calib and test windows; one test-window row set
    a week the schedule can hold a factor for."""
    return pd.concat([
        _hours("c1", "2026-07-01", 4), _hours("c2", "2026-07-08", 4, q0=8),
        _hours("t1", "2026-07-28", 4), _hours("t2", "2026-08-04", 4, q0=8),
    ], ignore_index=True)


def test_fidelity_grades_the_frozen_artifact_and_reports_the_refit_beside_it(
        cfg, tmp_path):
    """The gate freezes calibration at the test window's start (a factor fit
    inside the graded window has read the rows it grades); the weekly-refit
    reading sits beside it and must not disturb the gate's coverage."""
    from backtest.replay import fidelity
    cfg = _harness_cfg(cfg, tmp_path)
    r_lookup = json.load(open(cfg["dispersion"]["r_lookup_path"]))
    prior = {"per_category": {"FRUIT": {"mean": -1.2, "std": 0.5}}}
    frame = _fidelity_frame(cfg)
    anchor_only = _Applier(cfg, schedule={"2026-07-27": {"FRUIT": 1.0},
                                          "2026-08-03": {"FRUIT": 1.0}})
    doubled = _Applier(cfg, schedule={"2026-07-27": {"FRUIT": 2.0},
                                      "2026-08-03": {"FRUIT": 2.0}})
    a, _ = fidelity(frame, cfg, anchor_only, prior, r_lookup)
    b, _ = fidelity(frame, cfg, doubled, prior, r_lookup)
    assert a["calibration_frozen_at"] == b["calibration_frozen_at"] == "2026-07-27"
    assert a["gate_window"] == "test"
    assert a["calibration_gate_value"] == b["calibration_gate_value"], \
        "the test-week schedule factor reached the frozen gate value"
    # the mechanism reading DOES see the schedule: doubled mu, lower ratio
    assert b["weekly_refit"]["level_bias_at_anchor"] < a["weekly_refit"]["level_bias_at_anchor"]
    assert b["weekly_refit"]["vs_frozen"] < 0
    # and the refit pass left the gate's coverage counters alone
    cov = doubled.calibration_coverage()
    assert cov["verdict"].startswith("OK") and cov["frozen_from"] == "2026-07-27"
    assert cov["rows_frozen_at_anchor"] == 8 and cov["rows_on_schedule"] == 0
    assert doubled._freeze_from == pd.Timestamp("2026-07-27")


def test_the_backtest_slices_to_pre_launch_before_anything_reads_the_frame(
        cfg, tmp_path, monkeypatch):
    """Rule 16. Two paths once reached the hold-out and neither announced
    itself; fidelity() must receive the dp_eligible, pre-launch frame."""
    import yaml
    from backtest import __main__ as bt

    test_end = cfg["data"]["split"]["test_end"]
    frame = pd.concat([_hours("before", test_end, 4),
                       _hours("after", cfg["data"]["holdout"]["start"], 4),
                       _hours("ineligible", "2026-07-01", 4, dp=False)])
    frame.to_parquet(tmp_path / "prepared.parquet")
    (tmp_path / "prior.json").write_text(json.dumps(
        {"source": "profile_density",
         "per_category": {"FRUIT": {"mean": -1.2, "std": 0.5}}}))
    cfg = _harness_cfg(cfg, tmp_path)
    cfg["posterior"]["prior"] = dict(cfg["posterior"]["prior"],
                                     path=str(tmp_path / "prior.json"))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    got = {}

    def fake_fidelity(d, cfg, model, prior, r_lookup):
        got["frame"] = d
        return {"fidelity_episode_sold_ratio": 1.0, "calibration_gate_metric": "m",
                "calibration_gate_value": 1.0, "calibration_gate_band": [0.9, 1.1],
                "calibration_gate": "PASS"}, d

    monkeypatch.setattr(bt, "BaselineModel", lambda c: _Applier(c))
    monkeypatch.setattr(bt, "fidelity", fake_fidelity)
    monkeypatch.setattr(bt, "policy_replay", lambda *a, **k: (
        {"actual_il": 1.0, "actual_il_pct": 0.1, "legacy_model_il": 1.0,
         "dp_il": 1.0, "pct_dp_deepened": 0.0,
         "policy_gap_like_for_like": {"dp_il_reduction_pct_of_legacy": None}},
        pd.DataFrame(), SpreadLedger()))
    monkeypatch.setattr(bt, "derive_tau_initial", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [
        "backtest", "--input", str(tmp_path / "prepared.parquet"),
        "--config", str(tmp_path / "config.yaml"),
        "--out", str(tmp_path / "bt.json")])
    bt.main()

    assert set(got["frame"].episode_id) == {"before"}, \
        "fidelity saw a hold-out or dp-ineligible episode"
    out = json.load(open(tmp_path / "bt.json"))
    assert out["population"]["episodes_excluded_after_test_end"] == 1
    assert out["population"]["episodes_excluded_dp_ineligible"] == 1
    assert out["population"]["sees_up_to"] == test_end
    # coverage is reported where the numbers are read (static mode here: the
    # applier carries no schedule)
    assert out["artifact_versions"]["calibration_coverage"]["mode"] == "static"


def test_apply_is_refused_when_the_calibration_schedule_is_stale(cfg, tmp_path):
    """The gate must bite inside update.run, beside the event-quality gates:
    a schedule that no longer reaches today refuses --apply."""
    from pipeline.update import run as update_run
    cfg = _harness_cfg(cfg, tmp_path)
    cal = tmp_path / "calibration.json"
    cfg["baseline_model"] = dict(cfg["baseline_model"],
                                 calibration_factor_path=str(cal))
    cal.write_text(json.dumps({
        "grain": "category", "factors": {"FRUIT": 1.2},
        "schedule": {"mode": "rolling_trailing", "trailing_weeks": 4,
                     "by_week": {"2026-07-06": {"FRUIT": 1.1}}}}))
    late = update_run(cfg, apply=True, events_root=str(tmp_path / "ev"),
                      posterior_path=cfg["posterior"]["path"], today="2026-07-21")
    assert not late["event_quality_gates"]["calibration_schedule_current"]["pass"]
    assert "calibration_schedule_current" in late["refused"]
    assert not late["applied"]
    current = update_run(cfg, apply=True, events_root=str(tmp_path / "ev"),
                         posterior_path=cfg["posterior"]["path"], today="2026-07-08")
    assert current["event_quality_gates"]["calibration_schedule_current"]["pass"]
    assert "refused" not in current


# ------------------------------------------------ the bootstrap loop driver

def _scripted_settle(monkeypatch, dlogs, converge_at=None, max_turns=20):
    """Run bootstrap.run.settle with the subprocess steps recorded and the
    convergence verdicts scripted: turn k reads dlogs[k-1]; `converge_at`
    names the turn whose verdict says settled."""
    import bootstrap.run as br
    calls, state = [], {"turn": 0}

    def step(label, args, fatal=True, quiet=False):
        calls.append(list(args))
        if "--check-convergence" in args and "--commit-convergence" in args:
            state["turn"] += 1
        return 0

    def convergence(cfg):
        t = state["turn"]
        return {"converged": t == converge_at,
                "max_abs_dlog": dlogs[min(t, len(dlogs)) - 1]}

    monkeypatch.setattr(br, "step", step)
    monkeypatch.setattr(br, "convergence", convergence)
    ok, turns, _ = br.settle({}, max_turns)
    return ok, turns, calls


def test_the_loop_runs_3b_once_and_finishes_with_a_full_prior(monkeypatch):
    """Turn k's --check-convergence solves the factors turn k+1 would; the
    loop commits that solve instead of recomputing it, runs the prior --fast
    inside the loop, and re-runs it FULL (then re-checks) once settled --
    a production-shaped 9-turn trajectory with a plateau must survive."""
    dlogs = [2.29, .9, .9, .4, .35, .2, .09, .02, .005]
    ok, turns, calls = _scripted_settle(monkeypatch, dlogs, converge_at=9)
    assert ok and turns == 9
    fits = [c for c in calls if "--fit-calibration" in c]
    assert len(fits) == 1, "3b belongs to the first turn only"
    priors = [c for c in calls if c[0] == "bootstrap.estimate_prior"]
    assert len(priors) == 10 and all("--fast" in c for c in priors[:9]) \
        and "--fast" not in priors[-1], "in-loop priors fast, the artifact's full"
    checks = [c for c in calls if "--check-convergence" in c]
    assert all("--commit-convergence" in c for c in checks[:9])
    assert "--commit-convergence" not in checks[-1], \
        "the confirm after the full prior is a DRY re-check"
    assert calls.index(priors[-1]) < calls.index(checks[-1])


def test_the_loop_stalls_on_three_turns_without_a_new_best(monkeypatch):
    """The STALL test reads THIS run's turns: stuck and oscillating both stop
    at turn 5; the cap is a runaway guard, not the budget."""
    for dlogs in ([.31, .30, .305, .30, .31, .30], [.5, .2, .5, .2, .5, .2]):
        ok, turns, _ = _scripted_settle(monkeypatch, dlogs)
        assert (ok, turns) == (False, 5)
    # a plain 20-turn cap never fires on a settling chain first
    ok, turns, _ = _scripted_settle(monkeypatch, [1 / (t + 1) for t in range(20)],
                                    converge_at=12)
    assert ok and turns == 12


def test_the_loop_cap_is_sized_for_production():
    """The owner measures 8-9 turns; the fixture 3-4. A cap under 20 would
    cut a real settle short (rule 19: size for the extract that matters)."""
    import subprocess
    r = subprocess.run([sys.executable, "-m", "bootstrap.run", "--help"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import re
    m = re.search(r"--max-turns.*?default (\d+)", r.stdout, re.S)
    assert m and int(m.group(1)) >= 20
