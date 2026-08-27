"""Tests for pipeline.status.

The whole point of a status view is that a green line means something. So the
invariant worth testing hardest is the one that would quietly destroy that: a
check that did not run must never render as a check that passed.
"""
import json

import pytest

from common.config import load_config
from pipeline import status


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _write(root, name, payload):
    (root / f"{name}.json").write_text(json.dumps(payload))


def _verdicts(report):
    return {r["check"]: r["verdict"] for r in report["checks"]}


def test_missing_reports_read_as_not_run_never_as_pass(cfg, tmp_path):
    v = _verdicts(status.collect(cfg, str(tmp_path)))
    for check in ("calibration level", "shadow gate", "stop conditions", "assurance"):
        assert v[check] == status.NONE
    assert status.PASS not in {v[c] for c in ("calibration level", "shadow gate")}


def test_a_report_that_exists_but_predates_a_block_is_not_a_pass(cfg, tmp_path):
    _write(tmp_path, "thresholds", {"ab_duration": {}})     # older schema
    v = _verdicts(status.collect(cfg, str(tmp_path)))
    assert v["guardrail floors"] == status.NONE


def test_calibration_gate_reads_the_band_from_config(cfg, tmp_path):
    lo, hi = cfg["baseline_model"]["calibration_gate_band"]
    _write(tmp_path, "backtest", {"fidelity": {
        "calibration_gate_metric": "level_bias_at_anchor",
        "calibration_gate_value": (lo + hi) / 2, "gate_window": "test"}})
    assert _verdicts(status.collect(cfg, str(tmp_path)))["calibration level"] == status.PASS

    _write(tmp_path, "backtest", {"fidelity": {
        "calibration_gate_value": hi + 0.5, "gate_window": "test"}})
    # out of band is a DIAGNOSTIC warning, never a launch-blocking FAIL:
    # calibration is always applied
    assert _verdicts(status.collect(cfg, str(tmp_path)))["calibration level"] == status.WARN


def test_shadow_verdict_carries_a_trailing_note(cfg, tmp_path):
    """The report writes 'PASS -- proceed to ...', not a bare verdict."""
    _write(tmp_path, "shadow", {"shadow_gate": {
        "verdict": "PASS -- proceed to exploit-only pilot (section 19)",
        "event_completeness": {"value": 1.0, "pass": True},
        "matched_decision_rate": {"value": 1.0, "pass": True},
        "cost_floor_violations": {"value": 0, "pass": True}}})
    row = next(r for r in status.collect(cfg, str(tmp_path))["checks"]
               if r["check"] == "shadow gate")
    assert row["verdict"] == status.PASS
    assert "completeness 1.0" in row["detail"]      # the value, not the dict


def test_assurance_thin_window_is_a_warning_not_a_pass(cfg, tmp_path):
    _write(tmp_path, "assurance", {n: {"verdict": "INSUFFICIENT"} for n in
                                   ("reproduction", "dispersion",
                                    "correlation", "exploration")})
    assert _verdicts(status.collect(cfg, str(tmp_path)))["assurance"] == status.WARN

    _write(tmp_path, "assurance", {"reproduction": {"verdict": "FAIL"},
                                   "dispersion": {"verdict": "PASS"},
                                   "correlation": {"verdict": "PASS"},
                                   "exploration": {"verdict": "PASS"}})
    assert _verdicts(status.collect(cfg, str(tmp_path)))["assurance"] == status.FAIL


def test_stop_conditions_that_cannot_fire_are_flagged(cfg, tmp_path):
    """An owner threshold left null means the guardrail is not watching."""
    _write(tmp_path, "monitor", {"stop_conditions": {
        "scrap": {"fired": False, "blocked": True},
        "margin": {"fired": False}}})
    assert _verdicts(status.collect(cfg, str(tmp_path)))["stop conditions"] == status.WARN

    _write(tmp_path, "monitor", {"stop_conditions": {"scrap": {"fired": True}}})
    assert _verdicts(status.collect(cfg, str(tmp_path)))["stop conditions"] == status.FAIL


def test_a_stale_report_vintage_fails_rather_than_grading_a_ghost_model(cfg):
    """After a retrain, yesterday's backtest/shadow reports describe a model
    that is no longer on disk -- every row they feed would still read green
    without this check (hard rule 1: cross-version comparisons are void)."""
    state = {"bundle": "bundle-NEW"}
    old = {"artifact_versions": {
        "baseline_model_version": "bundle-OLD",
        "config_version": cfg["meta"]["config_version"]}}
    row = status._vintages(cfg, state, old, None)
    assert row["verdict"] == status.FAIL and "bundle-OLD" in row["detail"]

    fresh = {"artifact_versions": {
        "baseline_model_version": "bundle-NEW",
        "config_version": cfg["meta"]["config_version"]}}
    assert status._vintages(cfg, state, fresh, fresh)["verdict"] == status.PASS


def test_a_report_from_an_edited_config_warns(cfg):
    state = {"bundle": "b"}
    rep = {"artifact_versions": {"baseline_model_version": "b",
                                 "config_version": "0.0.0-old"}}
    row = status._vintages(cfg, state, None, rep)
    assert row["verdict"] == status.WARN and "0.0.0-old" in row["detail"]


def test_vintages_without_a_bundle_read_not_run_never_pass(cfg):
    row = status._vintages(cfg, {"bundle": None}, {}, {})
    assert row["verdict"] == status.NONE


def test_the_ab_power_se_paste_is_mirrored_against_phase0(cfg):
    """The MEASURED A/B power SE is pasted from phase 0 and consumed nowhere
    at runtime, so a stale paste is silent forever without this mirror."""
    c = dict(cfg, ab_test=dict(cfg["ab_test"],
                               il_pct_ratio_se_clustered=0.000875))
    stale = {"config_values_measured":
             {"ab_test.il_pct_ratio_se_clustered": 0.002915}}
    row = status._mirrors(c, stale)
    assert row["verdict"] == status.FAIL
    assert "0.002915" in row["detail"] and "0.000875" in row["detail"]
    # a matching paste adds no drift line, and a missing phase0 report is
    # not drift (measure has not run yet). Assert on THIS mirror's line, not
    # the row verdict -- the artifact mirrors may drift independently in a
    # local checkout
    fresh = {"config_values_measured":
             {"ab_test.il_pct_ratio_se_clustered": 0.000875}}
    assert "il_pct_ratio_se_clustered" not in status._mirrors(c, fresh)["detail"]
    assert "il_pct_ratio_se_clustered" not in status._mirrors(c, None)["detail"]


def test_overall_verdict_and_render(cfg, tmp_path):
    report = status.collect(cfg, str(tmp_path))
    assert report["verdict"] in (status.PASS, status.FAIL)
    assert report["not_run"]
    text = status.render(report)
    assert "not run" in text and len(text.splitlines()) >= len(report["checks"])


def test_convergence_row_warns_until_the_fixed_point_is_asserted(cfg, tmp_path):
    """The calibration <-> dispersion loop is resolved by iteration; this row
    is the difference between 'assumed settled' and 'asserted settled'."""
    p = tmp_path / "cal.json"
    c = dict(cfg, baseline_model=dict(cfg["baseline_model"],
                                      calibration_factor_path=str(p)))
    # no artifact / never checked -> not run, never a silent pass
    assert status._calibration_convergence(c)["verdict"] == status.NONE
    p.write_text(json.dumps({"factors": {"A": 1.1}}))
    row = status._calibration_convergence(c)
    assert row["verdict"] == status.NONE and "never checked" in row["detail"]
    # checked and settled -> PASS; moved past tolerance -> WARN (not FAIL:
    # a chain-health reading, not a launch gate)
    p.write_text(json.dumps({"factors": {"A": 1.1}, "convergence": {
        "converged": True, "max_abs_dlog": 0.001, "tol_log": 0.02}}))
    assert status._calibration_convergence(c)["verdict"] == status.PASS
    p.write_text(json.dumps({"factors": {"A": 1.1}, "convergence": {
        "converged": False, "max_abs_dlog": 0.09, "tol_log": 0.02}}))
    row = status._calibration_convergence(c)
    assert row["verdict"] == status.WARN and "one more iteration" in row["where"]
