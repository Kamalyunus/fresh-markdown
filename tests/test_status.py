"""Tests for ops.status."""
import json

import pytest

from conftest import _cfg_with, _reports, _write
from ops import status
import copy as _copy
import json as _json
from ops import tune
from common.provenance import config_fingerprint, file_digest
from engine.explore import tau_provenance_error


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
    row = status._vintages(cfg, state, {"backtest": old})
    assert row["verdict"] == status.FAIL and "bundle-OLD" in row["detail"]

    fresh = {"artifact_versions": {"baseline_model_version": "bundle-NEW"},
             "config": config_fingerprint(cfg, "backtest")}
    assert status._vintages(cfg, state, {"backtest": fresh, "shadow": fresh})["verdict"] == status.PASS


def test_a_report_without_a_fingerprint_warns_whatever_its_version_string(cfg):
    """meta.config_version is a human label nobody has to bump; a report
    that carries only it says nothing about the config it graded."""
    state = {"bundle": "b"}
    for version in ("0.0.0-old", cfg["meta"]["config_version"]):
        rep = {"artifact_versions": {"baseline_model_version": "b",
                                     "config_version": version}}
        row = status._vintages(cfg, state, {"shadow": rep})
        assert row["verdict"] == status.WARN and "no config fingerprint" in row["detail"]


def test_vintages_without_a_bundle_read_not_run_never_pass(cfg):
    row = status._vintages(cfg, {"bundle": None}, {})
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


    _reports(tmp_path)                           # a complete, block-free set
    # artifacts that MATCH config, so only the drift under test shows up
    cfg = _cfg_with(cfg, tmp_path,
                    rho={"rho": cfg["dispersion"]["rho"]})

    def row(cfg_in):
        return [r for r in status.collect(cfg_in, str(tmp_path))["checks"]
                if r["check"] == "config mirrors reports"][0]

    drifted = _copy.deepcopy(cfg)
    drifted["learning"]["information_increment"] = 999.0
    r = row(drifted)
    assert r["verdict"] == status.FAIL
    assert "learning.information_increment" in r["detail"]
    assert "999.0" in r["detail"]

    # a recommendation tune DOWNGRADED to OWNER is not a stale paste: the
    # remedy is the split, not this key, so it warns rather than holding the
    # row red on a decision the owner has already taken
    infeasible = _copy.deepcopy(cfg)
    # align every OTHER measured value with the fixture reports, so the only
    # thing left disagreeing is the one downgraded to OWNER
    infeasible["learning"]["information_increment"] = 0.341
    infeasible["exploration"]["tau_initial"] = 1234.5
    infeasible["exploration"]["delta_min_log_bias"] = None
    infeasible["baseline_model"]["calibration_gate_band"] = [0.997, 1.003]
    infeasible["baseline_model"]["calibration_fit_trailing_weeks"] = 1
    infeasible["data"]["split"] = dict(infeasible["data"]["split"],
                                       calib_start="2026-06-29",
                                       calib_end="2026-07-26")
    bt = json.loads((tmp_path / "backtest.json").read_text())
    bt["fidelity"]["calibration_window_sweep"].update(
        recommended_fit_window="trailing_8w",
        trailing_8w={"mean_abs_log_error": 0.0001, "share_weeks_in_band": 1.0})
    (tmp_path / "backtest.json").write_text(json.dumps(bt))
    r = row(infeasible)
    assert r["verdict"] == status.WARN
    assert "not a paste" in r["detail"]

    # every key the check guards is one nobody CHOOSES -- owner preferences
    # (max_mean_step, max_std_shrink) must never appear here or the row
    # would be permanently red on a legitimate disagreement
    owner = {("learning", "max_mean_step"), ("learning", "max_std_shrink")}
    assert not (tune.MEASURED_KEYS & owner)


def test_an_invariant_a_block_names_is_not_reported_as_green(cfg, tmp_path):
    """Setting W without the calib >= 2W the split can support raises a
    tune BLOCK. Reporting that as 'not evaluated' left the owner reading
    'all gates green (1 not run)' immediately after breaking an invariant."""

    _write(tmp_path, "backtest", {
        "artifact_versions": {"baseline_model_version": "m1"},
        "fidelity": {"calibration_window_sweep": {
            "recommended_fit_window": "trailing_4w",
            "trailing_4w": {"mean_abs_log_error": 0.01,
                            "share_weeks_in_band": 0.9}}}})

    def row(c):
        return [r for r in status.collect(c, str(tmp_path))["checks"]
                if r["check"] == "config mirrors reports"][0]

    broken = _copy.deepcopy(cfg)
    broken["baseline_model"]["calibration_fit_trailing_weeks"] = 4
    broken["data"]["split"] = dict(broken["data"]["split"],
                                   calib_start="2026-06-29",
                                   calib_end="2026-07-26")      # 4 weeks
    r = row(broken)
    assert r["verdict"] == status.FAIL
    assert "data.split" in r["detail"] and "calib >= 8w" in r["detail"]

    # a genuinely absent report is still "not run", never a pass and never a fail
    empty = tmp_path / "none"
    empty.mkdir()
    assert [x for x in status.collect(cfg, str(empty))["checks"]
            if x["check"] == "config mirrors reports"][0]["verdict"] == status.NONE


def test_an_unverified_key_does_not_hide_a_stale_paste(cfg, tmp_path):
    """An older thresholds schema (no increment block) made the row 'not run'
    BEFORE the drift check ran, so a drifted W read as unevaluated and
    advance walked past it into the daily lane."""
    _reports(tmp_path, thresholds={"bounded_step_recommendation": {
        "consistent_max_mean_step": 0.485, "verdict": "MEAN RAIL BINDS FIRST"}})
    drifted = _copy.deepcopy(_cfg_with(cfg, tmp_path, rho={"rho": cfg["dispersion"]["rho"]}))
    drifted["baseline_model"]["calibration_gate_band"] = [0.5, 1.5]
    row = [r for r in status.collect(drifted, str(tmp_path))["checks"]
           if r["check"] == "config mirrors reports"][0]
    assert row["verdict"] == status.FAIL and "calibration_gate_band" in row["detail"]


def test_report_vintages_names_the_config_values_that_moved(cfg, tmp_path):
    """A report is evidence about the config it ran under. After a paste the
    old check still read PASS because meta.config_version had not been
    bumped; the fingerprint makes the paste visible and names it."""


    bundle = "m1"
    state = {"bundle": bundle, "problems": [], "missing": [],
             "sealed_bundle": bundle, "verdict": "PASS"}
    same = {"artifact_versions": {"baseline_model_version": bundle},
            "config": config_fingerprint(cfg, "shadow")}
    row = status._vintages(cfg, state, {"shadow": same})
    assert row["verdict"] == status.PASS
    assert "shadow=shadow" in row["detail"]           # phase is reported

    # a MEASURED paste that writes back what shadow itself derived is not a
    # reason to re-grade shadow: this WARN re-ran shadow after every tau
    # paste and chased the fixed point for a day
    pasted = _copy.deepcopy(cfg)
    pasted["exploration"]["tau_initial"] = 999.0
    row = status._vintages(pasted, state, {"shadow": same})
    assert row["verdict"] == status.PASS and "none it reads" in row["detail"]
    # a key a report READS is
    edited = _copy.deepcopy(cfg)
    edited["pricing"]["tier_step"] = 0.05
    row = status._vintages(edited, state, {"shadow": same})
    assert row["verdict"] == status.WARN
    assert "pricing.tier_step" in row["detail"] and "0.05" in row["detail"]

    # status and advance route by the SAME table (tune.stale_keys): the
    # delta_min + stop-threshold pastes re-grade shadow and thresholds, not
    # the backtest -- and reports that share a vintage share ONE sentence
    # instead of repeating the whole list per report
    bt = {"artifact_versions": {"baseline_model_version": bundle},
          "config": config_fingerprint(cfg, "thresholds")}
    pasted = _copy.deepcopy(cfg)
    pasted["exploration"]["delta_min_log_bias"] = dict(
        cfg["exploration"]["delta_min_log_bias"], _default=0.15)
    pasted["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = 0.43
    row = status._vintages(pasted, state, {"backtest": bt, "thresholds": bt, "shadow": same})
    assert row["verdict"] == status.WARN
    assert row["detail"].count("delta_min_log_bias") == 1
    assert row["detail"].count("scrap_deterioration_pct") == 1
    assert "backtest" not in row["detail"].split("since then")[0].replace("backtest=", "")
    assert "thresholds ran under" in row["detail"] and "shadow ran under" in row["detail"]
    row = status._vintages(pasted, state, {"backtest": bt})
    assert row["verdict"] == status.PASS and "none it reads" in row["detail"]
    # a production report reads the live config every run: no move re-grades
    # it, and it is never labelled as if a paste could
    mon = {"artifact_versions": {"baseline_model_version": bundle},
           "config": config_fingerprint(cfg, "production")}
    row = status._vintages(edited, state, {"monitor": mon})
    assert row["verdict"] == status.PASS and "monitor=production" in row["detail"]
    assert "none it reads" not in row["detail"]

    # a model mismatch still outranks a config move, and is FAIL
    other = {"artifact_versions": {"baseline_model_version": "m0"},
             "config": config_fingerprint(cfg, "thresholds")}
    row = status._vintages(cfg, state, {"backtest": other, "shadow": same})
    assert row["verdict"] == status.FAIL and "m0" in row["detail"]

    # a report with no config fingerprint says nothing about the config it
    # graded, whatever its version string: WARN, re-run -- reading it as
    # current let a shadow from an older code version stand as the launch
    # record and advance never re-ran it
    for version in ("0.0.1", cfg["meta"]["config_version"]):
        legacy = {"artifact_versions": {"baseline_model_version": bundle,
                                        "config_version": version}}
        row = status._vintages(cfg, state, {"shadow": legacy})
        assert row["verdict"] == status.WARN and "no config fingerprint" in row["detail"]


def test_a_null_tau_derivation_is_reported_not_crashed(cfg):
    """A backtest that ran before any gate passed writes
    `tau_initial_derivation: null`; `.get` on None crashed the whole status
    page, which is the one surface meant to say what is missing."""
    live = dict(cfg, exploration=dict(cfg["exploration"], tau_initial=None))
    row = status._tau(live, {"tau_initial_derivation": None}, None)
    assert row["verdict"] == status.FAIL
    assert "no derivation" in row["detail"]


def test_an_unmeasured_guardrail_floor_warns_never_passes(cfg, tmp_path):
    """"insufficient history" is not a blocking verdict, but a floor nobody
    could measure is a floor nobody checked. It read PASS."""
    _write(tmp_path, "thresholds", {"guardrail_threshold_recommendation": {
        "scrap": {"verdict": "insufficient history on either basis"}}})
    assert _verdicts(status.collect(cfg, str(tmp_path)))[
        "guardrail floors"] == status.WARN


# -------------------------------------------------------- tau provenance

def _cfg_with_tau(cfg, tau):
    return dict(cfg, exploration=dict(cfg["exploration"], tau_initial=tau))


def _derivation(tau, scoped=True):
    block = {"tau_initial": tau}
    if scoped:
        block["spread_decisions"] = 12345
    return {"tau_initial_derivation": block}


def test_a_matching_paste_from_a_current_backtest_is_clean(cfg):
    assert tau_provenance_error(_cfg_with_tau(cfg, 410.74),
                                _derivation(410.74)) is None


def test_a_null_tau_is_not_this_check_s_business(cfg):
    # the null case is _require_shadow_config's, and it is louder
    assert tau_provenance_error(_cfg_with_tau(cfg, None), None) is None


def test_a_paste_with_no_derivation_on_disk_is_refused(cfg):
    assert "no backtest derivation" in tau_provenance_error(
        _cfg_with_tau(cfg, 410.74), None)


def test_a_derivation_predating_the_scoping_fix_is_refused(cfg):
    err = tau_provenance_error(_cfg_with_tau(cfg, 410.74),
                               _derivation(410.74, scoped=False))
    assert "ENTRY decisions only" in err


def test_a_paste_that_no_longer_matches_its_source_is_refused(cfg):
    err = tau_provenance_error(_cfg_with_tau(cfg, 500.0), _derivation(410.74))
    assert "500.0" in err and "410.74" in err


def test_shadow_refuses_to_start_on_a_stale_tau(cfg, tmp_path):
    from common.config import ConfigError
    from evaluate import shadow
    cfg = _cfg_with_tau(cfg, 410.74)
    path = tmp_path / "backtest.json"
    none = str(tmp_path / "missing.json")
    path.write_text(json.dumps(_derivation(410.74, scoped=False)))
    with pytest.raises(ConfigError, match="stale tau"):
        shadow._require_shadow_config(cfg, backtest_path=str(path),
                                      shadow_path=none)
    path.write_text(json.dumps(_derivation(410.74)))
    shadow._require_shadow_config(cfg, backtest_path=str(path),
                                  shadow_path=none)   # now fine


def test_a_shadow_derivation_is_the_trusted_paste_source(cfg):
    """The anchored-path derivation outranks the backtest's exploit-only one:
    a paste matching shadow is clean even when the backtest disagrees, and a
    paste matching only the backtest is refused once shadow has derived."""
    shadow = {"tau_initial_derivation": {"tau_initial": 257.48,
                                         "fallback": False}}
    assert tau_provenance_error(_cfg_with_tau(cfg, 257.48),
                                _derivation(410.74), shadow) is None
    err = tau_provenance_error(_cfg_with_tau(cfg, 410.74),
                               _derivation(410.74), shadow)
    assert "257.48" in err and "shadow" in err


def test_a_fallback_shadow_block_defers_to_the_backtest_checks(cfg):
    # a shadow run that itself fell back to the paste is not a paste source
    shadow = {"tau_initial_derivation": {"tau_initial": None, "fallback": True}}
    assert tau_provenance_error(_cfg_with_tau(cfg, 410.74),
                                _derivation(410.74), shadow) is None


def test_a_null_prior_block_reads_not_run_rather_than_crashing(cfg):
    """The same block is guarded with `or {}` two checks earlier; here it
    was read bare, and a config without a prior path took status down."""
    live = dict(cfg, posterior=dict(cfg["posterior"], prior=None))
    row = status._prior(live)
    assert row["verdict"] == status.NONE


def test_a_null_launch_date_is_a_launch_blocker(cfg):
    """The weekly re-fit cannot reach the week being priced until
    data.launch_date is set, so a null one blocks the pilot like a null
    tau."""
    live = dict(cfg, data=dict(cfg["data"], launch_date=None))
    row = status._launch_blockers(live)
    assert row["verdict"] == status.FAIL and "data.launch_date" in row["detail"]
    live["data"]["launch_date"] = "2026-09-01"
    assert "data.launch_date" not in status._launch_blockers(live)["detail"]


def test_a_missing_artifact_is_not_a_matching_mirror(cfg, tmp_path):
    c = dict(cfg, dispersion=dict(cfg["dispersion"], rho_path=str(tmp_path / "none.json")))
    assert status._mirrors(c)["verdict"] == status.NONE


def test_a_measured_key_no_report_measures_is_unverified_not_green(cfg, tmp_path, reports_dir):
    """An older thresholds schema with no information_increment block used
    to read PASS: no finding, no drift, 'every MEASURED value matches'."""
    th = _json.loads((reports_dir / "thresholds.json").read_text())
    th.pop("information_increment_recommendation", None)
    (reports_dir / "thresholds.json").write_text(_json.dumps(th))
    # every OTHER measured value aligned with the fixture reports: a drift
    # elsewhere is FAIL and outranks "unverified" (never hidden behind it)
    c = _cfg_with(cfg, tmp_path, rho={"rho": cfg["dispersion"]["rho"]})
    c["baseline_model"] = dict(c["baseline_model"], calibration_gate_band=[0.997, 1.003],
                               calibration_fit_trailing_weeks=1)
    c["exploration"] = dict(c["exploration"], tau_initial=1234.5, delta_min_log_bias=None)
    row = status._config_vs_reports(c, str(reports_dir))
    assert row["verdict"] == status.NONE, row
    assert "information_increment" in row["detail"]


def test_a_suspended_exploration_reads_warn_with_the_resume_command(cfg, tmp_path):
    _write(tmp_path, "monitor", {
        "stop_conditions": {"fired": {"exploration_cost_vs_budget": False},
                            "guardrails": {}, "suspend_exploration": False},
        "exploration_suspended": {"since": "2026-09-02",
                                  "reasons": ["exploration_cost_vs_budget"]}})
    row = [r for r in status.collect(cfg, str(tmp_path))["checks"]
           if r["check"] == "stop conditions"][0]
    assert row["verdict"] == status.WARN
    assert "SUSPENDED" in row["detail"] and "--resume-exploration" in row["where"]
