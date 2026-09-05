"""evaluate.shadow: the hold-out default, the budget base and its pre-window
seed, the controller trace, the calibration regimes, the weekly re-fit, and
the harness end to end through the real event store."""

import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from common import episodes
from conftest import _Applier, _frame, _harness_cfg, _hours, load_config
from engine import explore as explore_mod
from engine.explore import SpreadLedger
from engine.posterior import PosteriorStore

WINDOW_START, WINDOW_END = "2026-08-10", "2026-08-28"     # config's hold-out


def test_shadow_runs_on_the_holdout_unless_told_otherwise():
    """The default matters more than the flag."""
    import inspect
    from evaluate import shadow

    # run_shadow assumes the hold-out, so a programmatic caller gets the same
    # default the CLI does (the CLI's own default is exercised end to end)
    assert inspect.signature(shadow.run_shadow).parameters[
        "window_basis"].default == shadow.HOLDOUT_BASIS


def test_missing_holdout_config_is_an_error_not_a_silent_full_run(
        cfg, tmp_path, monkeypatch):
    """The dangerous failure is running on everything and not saying so."""
    from evaluate import shadow

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
    from evaluate.shadow import _controller_trace

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


def test_the_budget_base_is_the_trailing_realised_il(cfg):
    """The IL base for a day's budget is the mean of REALISED daily IL over
    the trailing budget_il_window_days, ending YESTERDAY -- never the same
    day's own IL."""
    from engine.explore import trailing_daily_il

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
    from evaluate.shadow import pre_window_il_history

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
    from evaluate.shadow import pre_window_il_history

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


def test_shadow_sample_size_defaults_to_config():
    """--max-episodes unset must read the config sample, not fall back to
    'every episode' -- the whole point is that a full sweep is too slow."""
    import inspect
    from evaluate import shadow

    assert inspect.signature(shadow.run_shadow).parameters[
        "max_episodes"].default is None
    cfg = load_config()
    assert cfg["monitoring"]["shadow_gate"]["sample_episodes"] > 0


def test_the_learning_yield_reports_the_terms_behind_it():
    """A disappointing yield has two causes with OPPOSITE remedies -- too few
    forced decisions, or forced prices sitting too close to the reference --
    and the per-episode aggregate cannot tell them apart. So the report
    carries the decomposition, and the identity that makes it checkable:"""
    import inspect
    from evaluate import shadow

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
    for comp in ('out["abs_log_ratio"] +=', 'out["forced_mu"] +=',
                 'out["forced_discount_gap"] +='):
        assert comp in body, f"{comp} is not accumulated on forced hours"
    # and the gap is measured against the REFERENCE, the same baseline the
    # log ratio uses -- not against the optimum or the anchor
    gap = body[body.index('out["forced_discount_gap"] +='):]
    assert 'reference_discount' in gap[:220], \
        "the discount gap must be measured against the reference discount"


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


def _run_shadow(cfg, frame, model, monkeypatch, refit=None, **kw):
    """evaluate.shadow.main's wiring, on `frame`, with the applier."""
    from evaluate import shadow
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
    from evaluate.shadow import _mean_daily_budget

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
    from evaluate.shadow import pre_window_il_history
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


def test_weekly_refit_fits_each_week_on_data_strictly_before_it(
        cfg, tmp_path, monkeypatch):
    """The re-fit at week k may read only weeks < k -- no look-ahead inside
    the replay -- through the one factor solve."""
    from fit import train_baseline as tb
    from evaluate import shadow
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
