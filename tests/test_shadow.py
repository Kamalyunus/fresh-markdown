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


def test_the_trace_streak_counts_consecutive_calendar_days(cfg):
    """The stop condition's persistence is the monitor's rule
    (daily.monitor.evaluate_guardrail): consecutive CALENDAR days over the
    multiple. The trace once incremented on consecutive ledger ENTRIES, so
    two over-budget days a day apart read as a two-day streak and fired."""
    from evaluate.shadow import _controller_trace

    sc = cfg["monitoring"]["stop_conditions"]
    sc["persistence_days"] = 2                    # set here, not read shipped

    def trace(days):
        led = SpreadLedger()
        for day in days:
            led.add(day, [10.0, 20.0])
        # a thin trailing IL base spanning a whole window: every day's
        # expected spend is over budget
        base = {f"2026-08-{d:02d}": 100.0 for d in range(2, 10)}
        return _controller_trace(led, base, tau0=1000.0,
                                 widest_std=1.0, cfg=cfg, window_days=3)

    gapped = trace(["2026-08-10", "2026-08-12"])
    rows = gapped["by_day"]
    assert all(r["over_budget"] > sc["exploration_cost_vs_budget"] for r in rows)
    assert [r["days_over"] for r in rows] == [1, 1]
    assert not any(r["stop_condition_fires"] for r in rows)
    assert gapped["days_stop_condition_fires"] == 0

    contiguous = trace(["2026-08-10", "2026-08-11"])
    assert [r["days_over"] for r in contiguous["by_day"]] == [1, 2]
    assert contiguous["by_day"][-1]["stop_condition_fires"]
    assert contiguous["days_stop_condition_fires"] == 1


def test_deeper_and_shallower_hours_are_counted_with_their_sign(
        cfg, tmp_path, monkeypatch):
    """`share_hours_recommending_deeper_than_legacy_price` was the unsigned
    differing share under a second name. Deeper and shallower are tallied
    apart, and the differing share is their sum."""
    from evaluate import shadow

    recommended = iter([0.40, 0.20, 0.30])        # deeper, shallower, same

    def fake_decide(state, posterior, store, cfg, rng, tau, model_version,
                    spread_sink=None):
        d = next(recommended)
        evt = {"decision_id": f"d-{state['hour_of_day']}",
               "applied_discount": d,
               "applied_price": state["original_price"] * (1 - d),
               "cost": state["cost"], "is_exploration": False,
               "affordable_set_size": 1, "reference_discount": 0.30,
               "epsilon_posterior_mean": -1.0, "solver_latency_s": 0.0}
        store.emit_decision(evt)
        return evt

    monkeypatch.setattr(shadow, "decide", fake_decide)
    g = _hours("e", "2026-08-10", 3, disc=0.30)
    g["r"], g["mu_ref_hat"], g["is_observed"] = 1.0, 2.0, True
    ep = dict({c: g[c].to_numpy() for c in shadow.EP_COLS}, episode_id="e")
    ctx = {"cfg": cfg, "tau": None, "model_version": "x", "seed": 0,
           "cal_grain": "category", "cells": {"FRUIT": None}}
    out = shadow._shadow_one(ep, ctx)
    assert (out["deeper"], out["shallower"]) == (1, 1)

    # and the report's three shares are one tally read three ways
    monkeypatch.undo()
    cfg = _harness_cfg(cfg, tmp_path)
    rv = _run_shadow(cfg, _shadow_frame(), _Applier(cfg),
                     monkeypatch)["recommendation_vs_legacy"]
    assert rv["share_hours_differing"] == pytest.approx(
        rv["share_hours_recommending_deeper_than_legacy_price"]
        + rv["share_hours_recommending_shallower_than_legacy_price"], abs=1e-4)


def test_the_budget_base_is_the_trailing_realised_il(cfg):
    """The IL base for a day's budget is the mean of REALISED daily IL over
    the trailing budget_il_window_days, ending YESTERDAY -- never the same
    day's own IL."""
    from engine.explore import trailing_daily_il

    window = int(cfg["exploration"]["budget_il_window_days"])
    days = [str((pd.Timestamp("2026-08-01") + pd.Timedelta(days=i)).date())
            for i in range(window + 1)]
    today, trailing = days[-1], days[:-1]

    il = {day: 700.0 for day in trailing}                  # `window` flat days
    # today's base is the mean of the trailing days; its own IL must not enter
    il[today] = 99_999.0
    assert trailing_daily_il(il, today, cfg) == pytest.approx(700.0)

    # a zero-IL calendar day inside the window counts as ZERO, not skipped:
    # the budget is a share of the actual run-rate
    il2 = {trailing[0]: 700.0, trailing[-1]: 700.0}
    assert trailing_daily_il(il2, today, cfg) == pytest.approx(1400.0 / window)

    # no history at all -> no budget. The conservative side to start on.
    assert trailing_daily_il({}, today, cfg) == 0.0


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


def test_the_learning_yield_terms_are_accumulated_on_the_forced_hours(
        cfg, tmp_path, monkeypatch):
    """A disappointing yield has two causes with OPPOSITE remedies -- too few
    forced decisions, or forced prices sitting too close to the reference --
    and the per-episode aggregate cannot tell them apart. The terms behind
    it are tallied on the FORCED hours only, the gap against the REFERENCE
    (the baseline the log ratio uses), and the report carries them."""
    from evaluate import shadow

    ref, forced_d, r_ep = 0.30, 0.40, 2.0

    def fake_decide(state, posterior, store, cfg, rng, tau, model_version,
                    spread_sink=None):
        forced = state["hour_of_day"] == 10               # one forced hour of three
        d = forced_d if forced else ref
        evt = {"decision_id": f"d-{state['hour_of_day']}",
               "applied_discount": d,
               "applied_price": state["original_price"] * (1 - d),
               "cost": state["cost"], "is_exploration": forced,
               "exploration_cost": 12.5 if forced else 0.0,
               "affordable_set_size": 1, "reference_discount": ref,
               "epsilon_posterior_mean": -1.0, "dispersion_r": r_ep,
               "solver_latency_s": 0.0}
        store.emit_decision(evt)
        return evt

    monkeypatch.setattr(shadow, "decide", fake_decide)
    g = _hours("e", "2026-08-10", 3, disc=ref)
    g["r"], g["mu_ref_hat"], g["is_observed"] = 1.0, 2.0, True
    ep = dict({c: g[c].to_numpy() for c in shadow.EP_COLS}, episode_id="e")
    ctx = {"cfg": cfg, "tau": None, "model_version": "x", "seed": 0,
           "cal_grain": "category", "cells": {"FRUIT": None}}
    out = shadow._shadow_one(ep, ctx)

    lr = np.log((1 - forced_d) / (1 - ref))
    mu_rec = 2.0 * np.exp(-1.0 * lr)
    assert out["n_forced"] == 1 and out["would_be_cost"] == 12.5
    assert out["abs_log_ratio"] == pytest.approx(abs(lr))
    assert out["forced_mu"] == pytest.approx(mu_rec)
    assert out["forced_discount_gap"] == pytest.approx(forced_d - ref)
    assert out["raw_information"] == pytest.approx(
        mu_rec * lr ** 2 * r_ep / (r_ep + mu_rec))

    # and the report carries the decomposition beside the aggregate
    monkeypatch.undo()
    cfg = _harness_cfg(cfg, tmp_path)
    ly = _run_shadow(cfg, _shadow_frame(), _Applier(cfg),
                     monkeypatch)["learning_yield_would_be"]
    assert {"forced_decisions", "information_per_forced_decision",
            "mean_abs_log_price_ratio_forced",
            "mean_discount_gap_from_reference_forced_pp",
            "mean_mu_on_forced_hours"} <= set(ly)


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
    kw.setdefault("pre_window_frame", frame)
    return shadow.run_shadow(window, cfg, prior_il_by_day=history,
                             window_start=WINDOW_START, **kw)


def test_every_per_day_figure_divides_by_the_unsampled_unextended_span(
        cfg, tmp_path, monkeypatch):
    """n_days once came from three places: the extended frame (a synthetic
    tail adds a day), the sampled frame (a sample shrinks the span), the
    calendar. It is the window population's span, computed before either --
    and the report's `window` is that frame's too, so a reader (ops.tune)
    dividing by its dates lands on the same span."""
    from common.io import read_json
    from evaluate.backtest import predict_frame
    cfg = _harness_cfg(cfg, tmp_path)
    frame = _shadow_frame()
    report = _run_shadow(cfg, frame, _Applier(cfg), monkeypatch)
    b, td = report["exploration_budget_would_be"], report["tau_initial_derivation"]
    w = report["window"]
    # window: 08-10..08-13 opened (4 days); w3's tail reaches 08-14 (5)
    assert b["days"] == 4 == b["tau_controller_trace"]["window_days"] == w["days"]
    assert (w["date_min"], w["date_max"]) == ("2026-08-10", "2026-08-13")
    # the extended frame really does reach a fifth day; nothing reads it
    extended = predict_frame(episodes.window_slice(frame, WINDOW_START, WINDOW_END),
                             cfg, _Applier(cfg),
                             read_json(cfg["dispersion"]["r_lookup_path"]))
    assert str(extended.date.max()) == "2026-08-14"
    # pre-window: 08-04..08-08 opened (5 days); p3's tail reaches 08-09
    assert not td["fallback"] and td["days"] == 5
    assert b["implied_daily_spend"] == pytest.approx(
        report["exploration_would_be"]["would_be_cost_total"] / 4, abs=0.1)

    # a sample of one episode shrinks neither span
    sampled = _run_shadow(cfg, frame, _Applier(cfg), monkeypatch, max_episodes=1)
    assert sampled["window"]["sampled"] and sampled["window"]["episodes"] == 1
    assert sampled["exploration_budget_would_be"]["days"] == 4
    assert sampled["window"]["days"] == 4
    assert sampled["tau_initial_derivation"]["days"] == 5


def _crossing(eid, day, q0=6):
    """An episode opening `day` at 22:00 whose OBSERVED rows run into the
    next day (00:00, 01:00)."""
    first = _hours(eid, day, 2, hour0=22, q0=q0, tail=2)
    first["ending_inventory"] = first.starting_inventory - first.units_sold
    second = _hours(eid, str(pd.Timestamp(day).date() + pd.Timedelta(days=1)),
                    2, hour0=0, q0=int(first.ending_inventory.iloc[-1]))
    return pd.concat([first, second], ignore_index=True)


def test_n_days_is_the_span_of_opening_days_not_of_row_dates(
        cfg, tmp_path, monkeypatch):
    """A W-day OPENING window whose last episode opened at 22:00 has rows
    dated the next day. Counted over row dates, n_days read W+1 and the
    derived tau overspent by (W+1)/W -- on both the window and the
    pre-window week. The span is episodes.calendar_days over
    episodes.opening_dates, and the report's window is that span too."""
    cfg = _harness_cfg(cfg, tmp_path)
    frame = pd.concat([_shadow_frame(),
                       _crossing("p4", "2026-08-08"),      # pre-window, -> 08-09
                       _crossing("w4", "2026-08-13")],     # window, -> 08-14
                      ignore_index=True)
    window = episodes.window_slice(frame, WINDOW_START, WINDOW_END)
    assert str(window.date.max()) == "2026-08-14", "the fixture must cross"
    report = _run_shadow(cfg, frame, _Applier(cfg), monkeypatch)
    b, td, w = (report["exploration_budget_would_be"],
                report["tau_initial_derivation"], report["window"])
    assert b["days"] == w["days"] == b["tau_controller_trace"]["window_days"] == 4
    assert (w["date_min"], w["date_max"]) == ("2026-08-10", "2026-08-13")
    assert not td["fallback"] and td["days"] == 5
    assert b["implied_daily_spend"] == pytest.approx(
        report["exploration_would_be"]["would_be_cost_total"] / 4, abs=0.1)


def test_the_report_carries_the_digest_of_the_posterior_it_priced_with(
        cfg, tmp_path, monkeypatch):
    """ops.advance re-runs shadow when the posterior FILE moved (a re-init,
    a rho paste after a retrain); the report has to say which one it read."""
    from common.provenance import file_digest
    cfg = _harness_cfg(cfg, tmp_path)
    report = _run_shadow(cfg, _shadow_frame(), _Applier(cfg), monkeypatch)
    av = report["artifact_versions"]
    assert av["posterior_digest"] == file_digest(cfg["posterior"]["path"])
    assert av["posterior_versions"]


def test_the_paste_fallback_is_checked_against_the_runs_own_out_path(
        cfg, tmp_path, monkeypatch):
    """When the tau in force falls back to the config paste, the paste is
    checked against the derivation at the path THIS run writes to (--out),
    not a hard-coded reports/shadow.json."""
    from common.config import ConfigError
    from common.io import write_json
    cfg = _harness_cfg(cfg, tmp_path)
    cfg["exploration"] = dict(cfg["exploration"], tau_initial=5.0)
    frame = _shadow_frame()
    out = str(tmp_path / "elsewhere" / "shadow.json")

    # no pre-window frame -> the paste is the tau in force; a derivation at
    # `out` that disagrees with it is a stale paste and blocks the run
    write_json(out, {"tau_initial_derivation": {"tau_initial": 100.0}})
    with pytest.raises(ConfigError, match="stale"):
        _run_shadow(cfg, frame, _Applier(cfg), monkeypatch,
                    pre_window_frame=None, shadow_path=out)
    write_json(out, {"tau_initial_derivation": {"tau_initial": 5.0}})
    report = _run_shadow(cfg, frame, _Applier(cfg), monkeypatch,
                         pre_window_frame=None, shadow_path=out)
    b = report["exploration_budget_would_be"]
    assert b["tau"] == 5.0 and b["tau_source"].startswith("config paste")


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
    # every drift row is in the one re-fit week, and the factor reaches it
    # through the applier (BaselineModel.level_factors), not a lookup copy
    assert ra["rows_rescaled"] == a["decision_count"] > 0 and ra["spread"] < 0
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
