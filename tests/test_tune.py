"""Tests for pipeline.tune.

The tuner exists to end a copy-paste loop between a human and two agents, so
its failure mode is not a crash: it is confidently recommending a number read
off a stale report, or silently writing a SET BY OWNER value. Both are tested
here, because both are silent in production.
"""
import json
import os

import pytest
import yaml

from common.config import load_config
from pipeline import tune

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def cfg():
    return load_config(os.path.join(ROOT, "config.yaml"))


def _reports(root, **over):
    base = {
        "backtest.json": {
            "artifact_versions": {"baseline_model_version": "m1"},
            "fidelity": {"calibration_window_sweep": {
                "recommended_fit_window": "trailing_2w"}},
            "policy_deltas": {"step_sensitivity": {
                "deeper_belief": {"share_prices_changed": 0.02,
                                  "il_delta_pct": -0.0004}}},
        },
        "shadow.json": {
            "artifact_versions": {"baseline_model_version": "m1"},
            "tau_initial_derivation": {"tau_initial": 1234.5},
            "calibration_regimes": {"frozen_anchor": 1.0002,
                                    "weekly_refit": 0.9762},
            "learning_yield_would_be": {"episodes_per_bounded_update": 741.0,
                                        "calendar_floor_days_per_0.15_of_mean": 1},
            "window": {"date_min": "2026-08-10", "date_max": "2026-08-28",
                       "episodes": 111400},
        },
        "phase0.json": {"config_values_measured": {
            "ab_test.il_pct_ratio_se_clustered": 0.076918}},
        "thresholds.json": {
            "information_increment_recommendation": {
                "recommended": 0.341, "verdict": "measured"},
            "bounded_step_recommendation": {
                "consistent_max_mean_step": 0.485,
                "verdict": "MEAN RAIL BINDS FIRST"},
            "guardrail_threshold_recommendation": {},
            "ab_duration": {"target_mde_rel": 0.075, "by_duration": {}},
        },
    }
    base.update(over)
    for name, payload in base.items():
        (root / name).write_text(json.dumps(payload))
    return str(root)


def _cfg_with(cfg, tmp_path, cal=None, rho=None):
    cal_path = tmp_path / "calibration.json"
    cal_path.write_text(json.dumps(cal if cal is not None else {
        "provenance": {"bundle": "m1"},
        "convergence": {"converged": True, "max_abs_dlog": 0.001,
                        "tol_log": 0.02}}))
    rho_path = tmp_path / "rho.json"
    rho_path.write_text(json.dumps(rho or {"rho": 0.2436,
                                           "mean_forced_hours_per_episode": 5.909}))
    return dict(
        cfg,
        baseline_model=dict(cfg["baseline_model"],
                            calibration_factor_path=str(cal_path)),
        dispersion=dict(cfg["dispersion"], rho_path=str(rho_path)))


def test_a_missing_report_blocks_tuning_rather_than_tuning_on_nothing(cfg, tmp_path):
    c = _cfg_with(cfg, tmp_path)
    rep = tune.collect(c, str(tmp_path / "empty"))
    assert rep["blocked"] and not rep["to_paste"]


def test_reports_from_two_different_models_block(cfg, tmp_path):
    reports = tmp_path / "r"
    reports.mkdir()
    _reports(reports)
    (reports / "shadow.json").write_text(json.dumps(
        {"artifact_versions": {"baseline_model_version": "OTHER"}}))
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports))
    assert rep["blocked"]
    assert any("rule 1" in f["evidence"] for f in rep["findings"])


def test_an_unconverged_loop_blocks(cfg, tmp_path):
    reports = tmp_path / "r"
    reports.mkdir()
    _reports(reports)
    c = _cfg_with(cfg, tmp_path, cal={
        "provenance": {"bundle": "m1"},
        "convergence": {"converged": False, "max_abs_dlog": 0.09,
                        "tol_log": 0.02}})
    rep = tune.collect(c, str(reports))
    assert rep["blocked"]
    assert any("NOT CONVERGED" in f["evidence"] for f in rep["findings"])


def test_measured_values_are_recommended_and_owner_values_are_not_written(
        cfg, tmp_path):
    reports = tmp_path / "r"
    reports.mkdir()
    _reports(reports)
    c = _cfg_with(cfg, tmp_path)
    rep = tune.collect(c, str(reports))
    assert not rep["blocked"], [f for f in rep["findings"] if f["class"] == "BLOCK"]

    pasted = {f["key"] for f in rep["to_paste"]}
    assert "exploration.tau_initial" in pasted        # measured, appliable
    # every OWNER finding stays out of the paste set, whatever its status
    owner_keys = {f["key"] for f in rep["findings"] if f["class"] == "OWNER"}
    assert owner_keys and not (owner_keys & pasted), \
        "a SET BY OWNER value must never be auto-applied (AGENTS rule)"

    work = tmp_path / "config.yaml"
    work.write_text(open(os.path.join(ROOT, "config.yaml")).read())
    res = tune.apply(c, rep, str(work), out_dir=str(tmp_path / "out"))
    written = yaml.safe_load(work.read_text())
    assert written["exploration"]["tau_initial"] == 1234.5
    assert written["learning"]["max_mean_step"] == cfg["learning"]["max_mean_step"], \
        "the owner rail must be untouched"
    assert os.path.exists(res["backup"])
    log = json.load(open(res["log"]))["runs"][-1]
    assert log["applied"] and log["pending_owner_decisions"]
    assert all(a["source"] for a in log["applied"]), \
        "every applied value must name the report field it came from"


def test_apply_keeps_the_comment_that_carries_the_reasoning(cfg, tmp_path):
    work = tmp_path / "config.yaml"
    original = open(os.path.join(ROOT, "config.yaml")).read()
    work.write_text(original)
    out = tune.set_scalar(original, ("exploration", "tau_initial"), 999.0)
    line = [ln for ln in out.splitlines() if ln.startswith("  tau_initial:")][0]
    assert "999.0" in line and "#" in line, \
        "the comment IS the reasoning -- a round-trip that drops it is a loss"
    assert len(out.splitlines()) == len(original.splitlines())


def test_an_ambiguous_anchor_refuses_rather_than_guessing():
    text = "  rho: 1\nother:\n  rho: 2\n"
    with pytest.raises(RuntimeError, match="refusing to guess"):
        tune.set_scalar(text, ("dispersion", "rho"), 3)


def test_the_calendar_vs_evidence_bottleneck_is_named(cfg, tmp_path):
    """The reading that inverted the tuning advice: at ~5,900 episodes/day a
    741-episode update is 0.13 days of evidence against a 1-day gate, so
    chasing information buys nothing."""
    reports = tmp_path / "r"
    reports.mkdir()
    _reports(reports)
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports))
    line = [f for f in rep["findings"] if f["key"] == "learning bottleneck"][0]
    assert line["current"] == "CALENDAR"
    assert "max_mean_step" in line["evidence"]


def test_the_calibration_cadence_reading_prefers_whichever_is_nearer_one(
        cfg, tmp_path):
    reports = tmp_path / "r"
    reports.mkdir()
    _reports(reports)
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports))
    line = [f for f in rep["findings"] if f["key"] == "calibration cadence"][0]
    assert line["status"] == "OK"            # frozen 1.0002 beats weekly 0.9762
    assert "frozen anchor" in line["evidence"]
