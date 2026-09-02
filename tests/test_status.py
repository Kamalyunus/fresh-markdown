"""Tests for pipeline.status."""
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


def test_stop_conditions_are_read_in_the_shape_the_monitor_writes(cfg, tmp_path):
    """monitor.stop_conditions is {fired, guardrails, suspend_exploration};
    a per-condition flag lives under `fired`, and a null owner threshold
    arrives there as a BLOCKED status string. Reading the TOP level instead
    found three container keys, reported "3 evaluated", and left both the
    WARN and FAIL branches unreachable."""
    def report(fired):
        return {"stop_conditions": {
            "fired": fired,
            "guardrails": {k: {"fired": v is True} for k, v in fired.items()},
            "suspend_exploration": any(v is True for v in fired.values())}}

    _write(tmp_path, "monitor", report({
        "scrap_deterioration_pct": "BLOCKED -- threshold is null (SET BY OWNER)",
        "margin_deterioration_pct": False}))
    row = [r for r in status.collect(cfg, str(tmp_path))["checks"]
           if r["check"] == "stop conditions"][0]
    assert row["verdict"] == status.WARN
    assert "2 evaluated" in row["detail"]          # conditions, not containers
    assert "1 cannot fire" in row["detail"]

    _write(tmp_path, "monitor", report({"scrap_deterioration_pct": True,
                                        "margin_deterioration_pct": False}))
    row = [r for r in status.collect(cfg, str(tmp_path))["checks"]
           if r["check"] == "stop conditions"][0]
    assert row["verdict"] == status.FAIL
    assert "scrap_deterioration_pct" in row["detail"]   # WHICH one fired


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


def test_convergence_goes_stale_when_the_chain_it_checked_moves(cfg, tmp_path):
    """A convergence verdict is only about the artifacts in force when it ran.
    Re-fit the prior or the dispersion and the loop has turned again, so a
    green line would describe a chain that no longer exists -- the same
    failure `report vintages` prevents for reports. Production re-fits
    calibration weekly against a FROZEN r and rho, which does not turn the
    loop; a retrain of either does, and this is what notices."""
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"per_category": {}}))
    cal = tmp_path / "cal.json"
    c = dict(cfg,
             baseline_model=dict(cfg["baseline_model"],
                                 calibration_factor_path=str(cal)),
             posterior=dict(cfg["posterior"],
                            prior=dict(cfg["posterior"]["prior"],
                                       path=str(prior))),
             dispersion=dict(cfg["dispersion"],
                             r_lookup_path=str(tmp_path / "absent_r.json"),
                             rho_path=str(tmp_path / "absent_rho.json")))
    from common.provenance import file_digest

    cal.write_text(json.dumps({"factors": {"A": 1.1}, "convergence": {
        "converged": True, "max_abs_dlog": 0.001, "tol_log": 0.02,
        "checked_against": {"prior": file_digest(str(prior))}}}))
    assert status._calibration_convergence(c)["verdict"] == status.PASS

    prior.write_text(json.dumps({"per_category": {}, "refitted": True}))
    row = status._calibration_convergence(c)
    assert row["verdict"] == status.WARN
    assert "prior" in row["detail"] and "moved" in row["detail"]

    # an artifact written before digests existed cannot be checked for
    # staleness -- say so rather than implying the check still holds
    cal.write_text(json.dumps({"factors": {"A": 1.1}, "convergence": {
        "converged": True, "max_abs_dlog": 0.001, "tol_log": 0.02}}))
    row = status._calibration_convergence(c)
    assert row["verdict"] == status.PASS
    assert "staleness unverifiable" in row["detail"]


def test_every_blocking_floor_verdict_fails_not_just_TOO_TIGHT(cfg, tmp_path):
    """Design 12 names three blocking verdicts. status matched only "TOO",
    so BLOCKED and LIKELY INERT read PASS -- and tune pasted the BLOCKED
    floor's binding_floor while the chain stayed green."""
    for verdict in ("TOO TIGHT -- below the measured floor",
                    "BLOCKED -- the binding trailing floor is 1.4 on the "
                    "RELATIVE basis",
                    "CLEARS THE FLOOR BUT LIKELY INERT"):
        _write(tmp_path, "thresholds", {"guardrail_threshold_recommendation": {
            "scrap": {"verdict": verdict}}})
        assert _verdicts(status.collect(cfg, str(tmp_path)))[
            "guardrail floors"] == status.FAIL, verdict

    _write(tmp_path, "thresholds", {"guardrail_threshold_recommendation": {
        "scrap": {"verdict": "clears the floor"}}})
    assert _verdicts(status.collect(cfg, str(tmp_path)))[
        "guardrail floors"] == status.PASS


def test_a_measured_value_that_disagrees_with_its_report_fails(cfg, tmp_path):
    """`artifact mirrors` catches values pasted from a frozen artifact.
    Values pasted from a REPORT had no equivalent: a number from another run
    -- or from the repo's synthetic fixture, which ships in config.yaml --
    survived every check until somebody happened to run tune."""
    import copy as _copy

    from pipeline import tune

    _write(tmp_path, "backtest", {
        "artifact_versions": {"baseline_model_version": "m1"},
        "fidelity": {"calibration_window_sweep": {
            "recommended_fit_window": "trailing_1w",
            "trailing_1w": {"mean_abs_log_error": 0.001,
                            "share_weeks_in_band": 0.99},
            "trailing_2w": {"mean_abs_log_error": 0.02,
                            "share_weeks_in_band": 0.80}}}})

    def verdict(c):
        return [r for r in status.collect(c, str(tmp_path))["checks"]
                if r["check"] == "config mirrors reports"][0]

    # a BLOCK upstream suppresses it -- never a silent pass
    assert verdict(cfg)["verdict"] in (status.NONE, status.PASS, status.FAIL)

    drifted = _copy.deepcopy(cfg)
    drifted["baseline_model"]["calibration_fit_trailing_weeks"] = 2
    drifted["data"]["split"] = dict(drifted["data"]["split"],
                                    calib_start="2026-06-01",
                                    calib_end="2026-07-26")
    row = verdict(drifted)
    if row["verdict"] != status.NONE:            # not blocked upstream
        assert row["verdict"] == status.FAIL
        assert "calibration_fit_trailing_weeks" in row["detail"]

    # every key the check guards is one nobody CHOOSES -- owner preferences
    # (max_mean_step, max_std_shrink, the MDE) must never appear here or the
    # row would be permanently red on a legitimate disagreement
    owner = {("learning", "max_mean_step"), ("learning", "max_std_shrink"),
             ("ab_test", "min_detectable_effect_pct")}
    assert not (tune.MEASURED_KEYS & owner)
