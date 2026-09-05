"""Tests for evaluate.backtest: the calibration window sweep, the policy
replay and its spread ledger, the tau derivation, fidelity against the frozen
artifact, and the pre-launch slice.

The sweep ranks the calibration fit window, and ops.tune pastes the
winner -- so the comparison it runs has to be a fair one.
"""
import copy
import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import _Applier, _harness_cfg, _hours, episode_frame
from engine import explore as explore_mod
from engine.explore import SpreadLedger
from evaluate.backtest import calibration_window_sweep


def _anchor_rows(weeks, categories=("VEG", "FRUIT"), seed=7):
    """Anchor rows (total_discount == d_ref) the sweep can group by week."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-01-05", periods=7 * weeks, freq="D")
    return pd.DataFrame([
        {"date": str(d.date()), "category": c,
         "total_discount": 0.30, "d_ref": 0.30,
         "units_sold": float(rng.integers(40, 60)), "predicted_units": 50.0}
        for d in days for c in categories])


def test_every_sweep_row_is_scored_on_the_same_weeks(cfg):
    """Per-window burn-in judged an 8w window on 11 weeks and a 2w window on
    17 DIFFERENT weeks, so the ranking read which weeks, not which window.
    One common eval set, and `uncalibrated` ranked with the rest."""
    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [1, 2, 4]
    out = calibration_window_sweep(_anchor_rows(12), cfg)

    scored = {k: v["eval_weeks"] for k, v in out.items()
              if isinstance(v, dict) and "eval_weeks" in v}
    assert len(set(scored.values())) == 1, scored
    assert {"uncalibrated", "trailing_1w", "trailing_2w",
            "trailing_4w"} <= set(scored)
    assert out["eval_weeks_common_from"]
    # uncalibrated is ranked, but the PASTE target stays a real window: W=0
    # is not a config value
    assert out["recommended_fit_window"].startswith("trailing_")
    assert isinstance(out["uncalibrated_beats_all_windows"], bool)


def test_no_factors_winning_is_flagged_not_hidden(cfg):
    """On flat data the factors only add estimation noise, so uncalibrated
    wins -- and the sweep must say so instead of silently ranking the
    least-bad window."""
    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [1, 2]
    out = calibration_window_sweep(_anchor_rows(10), cfg)

    unc, best = out["uncalibrated"], out[out["recommended_fit_window"]]
    beats = ((-unc["share_weeks_in_band"], unc["mean_abs_log_error"])
             < (-best["share_weeks_in_band"], best["mean_abs_log_error"]))
    assert out["uncalibrated_beats_all_windows"] is beats
    assert beats, "flat anchor data: factors cannot beat no factors"
    assert "NO-FACTORS WINS" in out["verdict"]


def test_the_sweep_refuses_rather_than_score_a_stub_eval_set(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [8]
    out = calibration_window_sweep(_anchor_rows(3), cfg)
    assert isinstance(out, str) and out.startswith("NOT RUN")


def test_the_sweep_says_when_it_cannot_tell(cfg):
    """The ranking compares aggregates over ~10 weeks and then turns on a
    lexicographic tie-break, so ONE week of share_weeks_in_band can decide
    which window 'wins'. The paired test asks the question that matters --
    same week, did the factors move the ratio closer to 1 -- and says so when
    the answer is undecidable."""
    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [1, 2]
    out = calibration_window_sweep(_anchor_rows(10), cfg)

    for key in ("trailing_1w", "trailing_2w"):
        pv = out[key]["paired_vs_uncalibrated"]
        assert pv["weeks_paired"] == out[key]["eval_weeks"]
        assert 0 <= pv["weeks_calibration_helped"] <= pv["weeks_paired"]
        assert 0.0 <= pv["sign_test_p"] <= 1.0

    # flat anchor data: nothing to correct, so no window can separate
    assert out["calibration_earns_its_keep"].startswith("UNDECIDED")
    assert "tie-break, not a measurement" in out["calibration_earns_its_keep"]


def test_a_window_that_genuinely_helps_is_called_out(cfg):
    """A persistent per-category level offset is exactly what the factors
    exist to remove; the paired test must find it."""
    import numpy as np
    import pandas as pd

    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [2]
    rng = np.random.default_rng(3)
    days = pd.date_range("2026-01-05", periods=7 * 14, freq="D")
    rows = []
    for d in days:
        for c, bias in (("VEG", 1.6), ("FRUIT", 0.6)):   # stable, large offset
            rows.append({"date": str(d.date()), "category": c,
                         "total_discount": 0.30, "d_ref": 0.30,
                         "units_sold": 50.0 * bias + rng.normal(0, 1.0),
                         "predicted_units": 50.0})
    out = calibration_window_sweep(pd.DataFrame(rows), cfg)

    pv = out["trailing_2w"]["paired_vs_uncalibrated"]
    assert pv["verdict"] == "calibration helps", pv
    assert pv["median_abs_log_delta"] < 0            # error moved toward zero
    assert out["calibration_earns_its_keep"].startswith("YES")


def test_replay_collects_every_decision_hour(cfg):
    from evaluate.backtest import policy_replay
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
    from evaluate.backtest import policy_replay
    d = pd.concat([_replay_episode("a"), _replay_episode("open", ending_last=4)])
    with pytest.raises(ValueError, match="never closed"):
        policy_replay(d, cfg)


def test_the_tau_cross_check_uses_one_day_count_on_both_sides(cfg):
    """derive_tau_initial averaged IL over the days with episodes while the
    ledger divided spend by the CALENDAR span -- which, on the pre-launch
    frame, crosses the exclusion gap. Both sides now count the days that
    traded, so the bisection lands the spend under the budget it was given."""
    from evaluate.backtest import derive_tau_initial
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
    from evaluate.backtest import predict_frame
    cfg = _harness_cfg(cfg, tmp_path)
    r_lookup = json.load(open(cfg["dispersion"]["r_lookup_path"]))
    d = predict_frame(_hours("e", "2026-08-10", 2, hour0=22, tail=3), cfg,
                      _Applier(cfg, base_mu=1.7), r_lookup)
    assert len(d) == 5 and d.is_observed.tolist() == [True, True, False, False, False]
    assert (d.r == 1.0).all() and (d.mu_ref_hat == 1.7).all()
    assert d.hours_remaining.tolist() == [4, 3, 2, 1, 0]


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
    from evaluate.backtest import fidelity
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
    from evaluate import backtest as bt

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
         "intra_episode_moves": {"overall": {
             "share_episodes_with_a_step": 0.0, "mean_steps_per_episode": 0.0,
             "legacy_share_episodes_with_a_step": 0.0}, "by_cost_ratio_band": {}},
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


def test_step_sensitivity_prices_the_cap_on_real_episodes(cfg):
    """`learning.max_mean_step` is justified by measurement, not judgment:
    the sweep re-solves the DP arm at eps +- step and reports what moves.
    Far below the deepening bar a step must change NOTHING -- that measured
    insensitivity is what makes a wrong-direction update cheap (design
    5.11). The block must also be structurally sound: shares in [0, 1],
    finite IL on both sides, crossers a subset of the sample."""
    from evaluate.backtest import _episode_frame, step_sensitivity

    def episode(eid, eps):
        g = pd.DataFrame({
            "episode_id": [eid] * 4,
            "date": ["2026-05-01"] * 4, "hour_of_day": [9, 10, 11, 12],
            "total_discount": [0.25, 0.25, 0.30, 0.30],
            "original_price": [10_000.0] * 4, "cost": [4000.0] * 4,
            "d_ref": [0.25] * 4, "starting_inventory": [6, 5, 4, 3],
            "units_sold": [1, 1, 1, 1], "mu_ref_hat": [1.5] * 4,
            "r": [3.0] * 4, "eps": [eps] * 4, "is_observed": [True] * 4,
            "sku_id": [7] * 4, "fc": ["FC1"] * 4, "category": ["FRUIT"] * 4,
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
    from evaluate.backtest import _episode_frame, _replay_one

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
        "is_observed": [True] * 3, "sku_id": [7] * 3, "fc": ["FC1"] * 3,
        "category": ["FRUIT"] * 3,
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


def test_within_episode_moves_are_counted_on_the_arms_own_path(cfg):
    """`pct_dp_deepened` compares episode MEANS against legacy and says
    nothing about whether the agent moves after entry. `intra_episode_moves`
    counts steps on the DP arm's own path -- a fresh solve every hour, so a
    high-cost shelf (low deepening bar) steps where a mid-cost one holds."""
    import numpy as np
    from evaluate.backtest import intra_episode_steps, intra_episode_moves, _episode_frame, _replay_one

    # a step is a deepening between consecutive priced hours; empty-shelf
    # hours (None) are skipped, and a flat path has none
    assert intra_episode_steps((0.10, 0.10, 0.15, None, 0.15, 0.20)) == 2
    assert intra_episode_steps((0.25,) * 6) == 0

    def episode(eid, cost):
        g = pd.DataFrame({
            "episode_id": [eid] * 6, "date": ["2026-05-01"] * 6,
            "hour_of_day": [9, 10, 11, 12, 13, 14],
            "total_discount": [0.25] * 6, "original_price": [10_000.0] * 6,
            "cost": [cost] * 6, "d_ref": [0.25] * 6,
            "starting_inventory": [12, 12, 12, 12, 12, 12],
            "units_sold": [0] * 6, "mu_ref_hat": [0.4] * 6,       # slow shelf
            "r": [3.0] * 6, "eps": [-2.0] * 6, "is_observed": [True] * 6,
            "sku_id": [7] * 6, "fc": ["FC1"] * 6, "category": ["FRUIT"] * 6,
        })
        g["ending_inventory"] = g.starting_inventory - g.units_sold
        return _episode_frame(g)

    rows = [_replay_one(episode(f"e{i}", cost), cfg)[0]
            for i, cost in enumerate((3000.0, 5000.0, 7000.0, 7500.0))]
    ep = pd.DataFrame(rows)
    assert {"dp_steps", "legacy_model_steps"} <= set(ep.columns)
    out = intra_episode_moves(ep, cfg)
    bands = out["by_cost_ratio_band"]
    assert set(bands) == {"cost_ratio<0.4", "0.4<=cost_ratio<0.6", "cost_ratio>=0.6"}
    # the summary is arithmetic over the rows it was given
    assert out["overall"]["episodes"] == 4
    assert out["overall"]["mean_steps_per_episode"] == pytest.approx(ep.dp_steps.mean(), abs=1e-3)
    assert out["overall"]["share_episodes_with_a_step"] == pytest.approx((ep.dp_steps > 0).mean(), abs=1e-4)
    # the flat legacy schedule never steps; the deepening bar is read per band
    assert out["overall"]["legacy_share_episodes_with_a_step"] == 0.0
    assert bands["cost_ratio>=0.6"]["share_episodes_eps_above_threshold"] == 1.0
    assert bands["0.4<=cost_ratio<0.6"]["share_episodes_eps_above_threshold"] == 0.0
    # and the high-cost shelves, above the bar, step at least as often as the
    # mid-cost ones below it (a fresh solve every hour, not a pinned price)
    assert bands["cost_ratio>=0.6"]["mean_steps_per_episode"] >= \
        bands["0.4<=cost_ratio<0.6"]["mean_steps_per_episode"]


def test_the_replay_prices_at_the_launch_belief_and_transitions_at_the_prior(cfg):
    """The DP arm is the policy that will run (posterior.launch_belief); the
    world is the prior's best guess. So a steeper launch belief changes what
    the DP does, never what the legacy arm sells."""
    from evaluate.backtest import _episode_frame, _replay_one

    def frame(eps, eps_belief):
        g = pd.DataFrame({
            "episode_id": ["e"] * 6, "date": ["2026-05-01"] * 6,
            "hour_of_day": [9, 10, 11, 12, 13, 14],
            "total_discount": [0.25] * 6, "original_price": [10_000.0] * 6,
            "cost": [7000.0] * 6, "d_ref": [0.25] * 6,
            "starting_inventory": [12] * 6, "units_sold": [0] * 6,
            "mu_ref_hat": [0.4] * 6, "r": [3.0] * 6,
            "eps": [eps] * 6, "eps_belief": [eps_belief] * 6,
            "is_observed": [True] * 6, "sku_id": [7] * 6, "fc": ["FC1"] * 6,
            "category": ["FRUIT"] * 6})
        g["ending_inventory"] = g.starting_inventory - g.units_sold
        return _episode_frame(g)

    plain = _replay_one(frame(-2.0, -2.0), cfg)[0]
    steep = _replay_one(frame(-2.0, -3.5), cfg)[0]
    assert steep["eps"] == -2.0 and steep["eps_belief"] == -3.5
    # the legacy arm lives in the same world either way
    assert steep["legacy_model_il"] == pytest.approx(plain["legacy_model_il"])
    # the DP arm believes discounts move more, so it cuts at least as deep
    assert steep["dp_mean_discount"] >= plain["dp_mean_discount"] - 1e-9
    assert steep["dp_steps"] >= plain["dp_steps"]
