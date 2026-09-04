"""Tests for pipeline.tune."""
import json
import os

import pytest
import yaml

from conftest import ROOT, _cfg_with, _reports
from pipeline import tune


def test_a_missing_report_blocks_tuning_rather_than_tuning_on_nothing(cfg, tmp_path):
    c = _cfg_with(cfg, tmp_path)
    rep = tune.collect(c, str(tmp_path / "empty"))
    assert rep["blocked"] and not rep["to_paste"]


def test_reports_from_two_different_models_block(cfg, tmp_path, reports_dir):
    (reports_dir / "shadow.json").write_text(json.dumps(
        {"artifact_versions": {"baseline_model_version": "OTHER"}}))
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    assert rep["blocked"]
    assert any("rule 1" in f["evidence"] for f in rep["findings"])


def test_an_unconverged_loop_blocks(cfg, tmp_path, reports_dir):
    c = _cfg_with(cfg, tmp_path, cal={
        "provenance": {"bundle": "m1"},
        "convergence": {"converged": False, "max_abs_dlog": 0.09,
                        "tol_log": 0.02}})
    rep = tune.collect(c, str(reports_dir))
    assert rep["blocked"]
    assert any("NOT CONVERGED" in f["evidence"] for f in rep["findings"])


def test_a_measurable_value_is_pasted_not_left_to_a_human(cfg, tmp_path, reports_dir):
    """A value the data can decide should not wait on a decision (owner,
    2026-08-30). The guardrail stops are 3-sigma of the control arm's own
    noise -- a measurement, not a preference -- so they paste."""
    c = _cfg_with(cfg, tmp_path)
    # nothing set yet: this test is about what the tool decides, not about
    # what the shipped config happens to carry today
    c["monitoring"]["stop_conditions"].update(
        scrap_deterioration_pct=None, margin_deterioration_pct=None)
    c["baseline_model"]["calibration_fit_trailing_weeks"] = 2
    rep = tune.collect(c, str(reports_dir))
    pasted = {f["key"] for f in rep["to_paste"]}
    assert "monitoring.stop_conditions.scrap_deterioration_pct" in pasted
    assert "baseline_model.calibration_fit_trailing_weeks" in pasted


def test_the_rail_paste_is_gated_on_the_price_consequence(cfg, tmp_path, reports_dir):
    """`consistent_max_mean_step` is measured, but raising the rail re-prices
    real episodes -- so it pastes only when step_sensitivity agrees, and
    returns to the owner when the re-price is large."""
    # the default deeper arm: 2% of prices -> inside the gate
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    rail = [f for f in rep["findings"] if f["key"] == "learning.max_mean_step"][0]
    assert rail["class"] == "PASTE" and "inside the auto-apply gate" in rail["evidence"]

    bt = json.loads((reports_dir / "backtest.json").read_text())
    bt["policy_deltas"]["step_sensitivity"]["deeper_belief"][
        "share_prices_changed"] = 0.40     # a real price event
    (reports_dir / "backtest.json").write_text(json.dumps(bt))
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    rail = [f for f in rep["findings"] if f["key"] == "learning.max_mean_step"][0]
    assert rail["class"] == "OWNER" and "EXCEEDS the auto-apply gate" in rail["evidence"]
    assert rail["key"] not in {f["key"] for f in rep["to_paste"]}


def test_a_tolerance_stays_with_the_owner(cfg, tmp_path, reports_dir):
    """The one value the data genuinely cannot decide: it says what effect is
    DETECTABLE, never what size of effect is worth detecting."""
    _reports(reports_dir, **{"thresholds.json": {
        "information_increment_recommendation": {"recommended": 0.341},
        "bounded_step_recommendation": {},
        "guardrail_threshold_recommendation": {},
        "ab_duration": {"target_mde_rel": 0.075, "by_duration": {
            "4w": {"detectable_mde_rel": 0.241, "meets_target": "False"}}}}})
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    mde = [f for f in rep["findings"]
           if f["key"] == "ab_test.min_detectable_effect_pct"][0]
    assert mde["class"] == "OWNER"
    assert "NOT measurable" in mde["evidence"]
    assert mde["key"] not in {f["key"] for f in rep["to_paste"]}
    # the FRONTIER, not the target: echoing back the --mde flag would be
    # recommending the question as its own answer
    assert mde["recommended"] == 0.241, "must report what IS detectable"
    assert "4w -> 0.241" in mde["evidence"]
    assert "NO duration reaches" in mde["evidence"]


def test_owner_values_are_never_written(cfg, tmp_path, reports_dir):
    c = _cfg_with(cfg, tmp_path)
    rep = tune.collect(c, str(reports_dir))
    assert not rep["blocked"], [f for f in rep["findings"] if f["class"] == "BLOCK"]

    pasted = {f["key"] for f in rep["to_paste"]}
    assert "exploration.tau_initial" in pasted        # measured, appliable
    # every OWNER finding stays out of the paste set, whatever its status
    owner_keys = {f["key"] for f in rep["findings"] if f["class"] == "OWNER"}
    assert owner_keys and not (owner_keys & pasted), \
        "a SET BY OWNER value must never be auto-applied (AGENTS rule)"

    work = tmp_path / "config.yaml"
    work.write_text(open(os.path.join(ROOT, "config.yaml")).read())
    res = tune.apply(rep, str(work), out_dir=str(tmp_path / "out"))
    written = yaml.safe_load(work.read_text())
    assert written["exploration"]["tau_initial"] == 1234.5
    assert (written["ab_test"]["min_detectable_effect_pct"]
            == cfg["ab_test"]["min_detectable_effect_pct"]), \
        "a tolerance the data cannot decide must be left untouched"
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


def test_the_calendar_vs_evidence_bottleneck_is_named(cfg, tmp_path, reports_dir):
    """The reading that inverted the tuning advice: at ~5,900 episodes/day a
    741-episode update is 0.13 days of evidence against a 1-day gate, so
    chasing information buys nothing."""
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    line = [f for f in rep["findings"] if f["key"] == "learning bottleneck"][0]
    assert line["current"] == "CALENDAR"
    assert "max_mean_step" in line["evidence"]


def test_the_calibration_cadence_reading_prefers_whichever_is_nearer_one(
        cfg, tmp_path, reports_dir):
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    line = [f for f in rep["findings"] if f["key"] == "calibration cadence"][0]
    assert line["status"] == "OK"            # frozen 1.0002 beats weekly 0.9762
    assert "frozen anchor" in line["evidence"]


def test_the_level_band_may_tighten_but_never_widen(cfg, tmp_path, reports_dir):
    """The band is sized from measured week-to-week anchor volatility, but a
    band WIDER than the current one is a decision about tolerated level error,
    not a reading (owner, 2026-08-30). The clamp is in RATIO space on purpose:
    exp(0.10) is 1.1052, so clamping the log half-width would widen the upper
    edge past the ceiling it is meant to enforce."""
    c = _cfg_with(cfg, tmp_path)
    cap = c["tuning"]["calibration_band_max_half_width"]

    # a quiet extract tightens
    rep = tune.collect(c, str(reports_dir))
    band = [f for f in rep["findings"]
            if f["key"] == "baseline_model.calibration_gate_band"][0]
    lo, hi = band["recommended"]
    assert lo > 1 - cap and hi < 1 + cap, "quiet weeks should tighten the band"

    # a volatile one is clamped, and never wider than the ceiling on EITHER side
    bt = json.loads((reports_dir / "backtest.json").read_text())
    chosen = f"trailing_{c['baseline_model']['calibration_fit_trailing_weeks']}w"
    bt["fidelity"]["calibration_window_sweep"][chosen][
        "mean_abs_log_error"] = 0.5
    (reports_dir / "backtest.json").write_text(json.dumps(bt))
    rep = tune.collect(c, str(reports_dir))
    band = [f for f in rep["findings"]
            if f["key"] == "baseline_model.calibration_gate_band"][0]
    lo, hi = band["recommended"]
    assert (lo, hi) == (round(1 - cap, 4), round(1 + cap, 4))
    assert hi <= 1 + cap, "clamping in log space would have produced 1.1052"
    assert "may only tighten" in band["evidence"]


def test_max_std_shrink_is_suggested_with_its_alternative_never_written(
        cfg, tmp_path, reports_dir):
    """Both rails resolve the same mismatch; WHICH one moves is a safety
    posture. The tool supplies both numbers and takes neither decision."""
    _reports(reports_dir, **{"thresholds.json": {
        "information_increment_recommendation": {"recommended": 0.341},
        "bounded_step_recommendation": {"median_launch_std": 1.1088,
                                        "consistent_max_mean_step": 0.485},
        "guardrail_threshold_recommendation": {},
        "ab_duration": {"target_mde_rel": 0.075, "by_duration": {}}}})
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    f = [x for x in rep["findings"] if x["key"] == "learning.max_std_shrink"][0]
    assert f["class"] == "OWNER"
    # 1 - sqrt(1 - 0.15/1.1088) = 0.0701: the shrink that makes the CURRENT
    # mean step consistent, i.e. the other way to settle the same mismatch
    assert abs(f["recommended"] - 0.0701) < 1e-3
    assert "SAFETY POSTURE" in f["evidence"] and "0.485" in f["evidence"]
    assert f["key"] not in {x["key"] for x in rep["to_paste"]}


def test_apply_names_the_minimum_rerun_and_never_asks_for_a_retrain(
        cfg, tmp_path, reports_dir):
    """The bug this prevents put an agent in a loop."""
    c = _cfg_with(cfg, tmp_path)
    work = tmp_path / "config.yaml"
    work.write_text(open(os.path.join(ROOT, "config.yaml")).read())

    rep = tune.collect(c, str(reports_dir))
    res = tune.apply(rep, str(work), out_dir=str(tmp_path / "out"))

    # runtime-only values require nothing; the message must not send anyone
    # back to the bootstrap
    assert res["rerun"] in ("none", "calibration")
    # only the "retrain" class may send anyone to the full script, and only
    # data.split reaches it -- which is SET BY OWNER, so --apply never writes
    # one. The calibration text mentions retraining solely to forbid it.
    assert "run_bootstrap" not in tune.RERUN_STEPS[res["rerun"]]
    assert res["rerun"] != "retrain"

    # the fit window is the ONE paste that turns the loop -- and even it does
    # not retrain the baseline
    assert tune.RERUN[("baseline_model", "calibration_fit_trailing_weeks")] \
        == "calibration"
    steps = tune.RERUN_STEPS["calibration"]
    assert "bootstrap.run --check-only" in steps, \
        "the loop is driven by the module, not hand-iterated"
    assert "WITHOUT retraining" in steps

    # nothing that only production reads may claim to need a re-fit
    for key in (("learning", "information_increment"),
                ("learning", "max_mean_step"),
                ("exploration", "tau_initial"),
                ("dispersion", "rho"),
                ("monitoring", "stop_conditions", "scrap_deterioration_pct")):
        assert tune.RERUN.get(key, "none") == "none", key

    # and the decision log records which re-run the run required
    log = json.load(open(res["log"]))["runs"][-1]
    assert log["rerun_required"] == res["rerun"]


def test_the_fit_window_holds_on_a_near_tie_instead_of_oscillating(cfg, tmp_path):
    """W is the only paste that turns the calibration loop, and re-settling
    the loop re-scores the sweep -- a strict argmin flips between near-tied
    windows and loops an agent on apply -> check-only forever. Near-tie:
    HOLD. Material win: switch."""
    import pipeline.tune as tune

    def w_finding(sweep):
        backtest = {"fidelity": {"calibration_window_sweep": sweep}}
        finds = tune._readings(cfg, backtest, {})
        return [f for f in finds
                if f["key"] == "baseline_model.calibration_fit_trailing_weeks"][0]

    # W is set HERE: with cur == the candidate window there is no tie to hold
    cfg = {**cfg, "baseline_model": {**cfg["baseline_model"],
                                     "calibration_fit_trailing_weeks": 2}}
    cur = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
    near_tie = {
        "recommended_fit_window": "trailing_1w",
        "trailing_1w": {"mean_abs_log_error": 0.0195,
                        "share_weeks_in_band": 0.87},
        f"trailing_{cur}w": {"mean_abs_log_error": 0.0200,
                             "share_weeks_in_band": 0.85},
    }
    f = w_finding(near_tie)
    assert f["status"] == "OK" and f["recommended"] == cur
    assert "HELD" in f["evidence"]

    material = {
        "recommended_fit_window": "trailing_1w",
        "trailing_1w": {"mean_abs_log_error": 0.010,
                        "share_weeks_in_band": 0.95},
        f"trailing_{cur}w": {"mean_abs_log_error": 0.0200,
                             "share_weeks_in_band": 0.85},
    }
    f = w_finding(material)
    assert f["status"] == "ACT" and f["recommended"] == 1


def test_no_factors_winning_is_reported_and_is_never_a_paste(cfg):
    """`uncalibrated` beating every window says the level factors are adding
    noise. That is an owner reading, not a W: W=0 is not a config value, so
    the paste stays on the best CALIBRATED window."""
    import pipeline.tune as tune

    cur = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
    sweep = {
        "uncalibrated": {"mean_abs_log_error": 0.0803,
                         "share_weeks_in_band": 0.7222},
        f"trailing_{cur}w": {"mean_abs_log_error": 0.0967,
                             "share_weeks_in_band": 0.7059},
        "recommended_fit_window": f"trailing_{cur}w",
        "uncalibrated_beats_all_windows": True,
    }
    finds = tune._readings(cfg, {"fidelity": {"calibration_window_sweep": sweep}}, {})
    by = {f["key"]: f for f in finds}

    keep = by["level calibration earns its keep"]
    assert (keep["class"], keep["status"]) == ("INFO", "ACT")
    assert "0.0803" in keep["evidence"] and "0.0967" in keep["evidence"]

    # the W paste is untouched -- it never points at "no calibration"
    w = by["baseline_model.calibration_fit_trailing_weeks"]
    assert w["recommended"] == cur and w["status"] == "OK"

    del sweep["uncalibrated_beats_all_windows"]
    assert "level calibration earns its keep" not in {
        f["key"] for f in
        tune._readings(cfg, {"fidelity": {"calibration_window_sweep": sweep}}, {})}


def test_tau_uses_the_same_staleness_rule_the_status_gate_enforces(cfg):
    """tune's own 1% tolerance was 270x looser than the gate's absolute 0.01
    won: a 0.51 drift reported OK, --apply wrote nothing, and status still
    FAILed. tune ran clean against a red status and the only escape was a
    hand edit of config.yaml."""
    import copy as _copy

    from pricing.explore import tau_provenance_error

    cfg = _copy.deepcopy(cfg)
    cfg["exploration"]["tau_initial"] = 269.99
    shadow = {"tau_initial_derivation": {"tau_initial": 270.5,
                                         "spread_decisions": 1000}}

    def tau_finding(c, sh):
        return [f for f in tune._measured(c, sh, None, {}, None)
                if f["key"] == "exploration.tau_initial"][0]

    # 0.19% drift: inside the provenance tolerance, so neither complains.
    # tau_initial seeds day one only and tau_next absorbs it (posterior.py).
    assert tau_finding(cfg, shadow)["status"] == "OK"
    assert tau_provenance_error(cfg, None, shadow) is None

    # a wrong-run paste is the case the gate exists for, and both see it
    cfg["exploration"]["tau_initial"] = 34.0
    f = tau_finding(cfg, shadow)
    assert f["status"] == "ACT" and f["recommended"] == 270.5
    assert tau_provenance_error(cfg, None, shadow) is not None

    cfg["exploration"]["tau_initial"] = 270.5
    assert tau_finding(cfg, shadow)["status"] == "OK"
    assert tau_provenance_error(cfg, None, shadow) is None

    # a null paste is still ACT -- the gate stays quiet on null by design
    cfg["exploration"]["tau_initial"] = None
    assert tau_finding(cfg, shadow)["status"] == "ACT"


def test_an_unusable_guardrail_floor_is_never_pasted(cfg):
    """binding_floor is set BEFORE the unusability check, so a BLOCKED block
    still carries a number. Pasting it writes a threshold the report itself
    calls unusable while status stays green."""
    blocked = {"guardrail_threshold_recommendation": {"scrap": {
        "config_key": "monitoring.stop_conditions.scrap_deterioration_pct",
        "binding_floor": 1.4, "binding_basis": "trailing",
        "verdict": "BLOCKED -- the binding trailing floor is 1.4 on the "
                   "RELATIVE basis"}}}
    f = [x for x in tune._derived(cfg, {}, blocked)
         if x["key"].endswith("scrap_deterioration_pct")][0]
    assert f["class"] == "OWNER" and f["recommended"] is None
    assert "BLOCKED" in f["evidence"] and "NOT pasted" in f["evidence"]

    usable = {"guardrail_threshold_recommendation": {"scrap": {
        "config_key": "monitoring.stop_conditions.scrap_deterioration_pct",
        "binding_floor": 0.18, "binding_basis": "control_arm",
        "verdict": "clears the floor"}}}
    f = [x for x in tune._derived(cfg, {}, usable)
         if x["key"].endswith("scrap_deterioration_pct")][0]
    assert f["class"] == "PASTE" and f["recommended"] == 0.18


def test_a_not_run_sweep_is_a_finding_not_a_traceback(cfg, tmp_path, reports_dir):
    """replay writes the sweep as a STRING on its NOT RUN path; `.get` on
    that took tune down instead of reporting the missing measurement."""
    bt = json.loads((reports_dir / "backtest.json").read_text())
    bt["fidelity"]["calibration_window_sweep"] = "NOT RUN: calib < 2W"
    (reports_dir / "backtest.json").write_text(json.dumps(bt))
    assert tune._sweep_of(bt) == {}
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))  # no traceback
    assert not any(f["key"] == "baseline_model.calibration_fit_trailing_weeks"
                   and f["class"] == tune.PASTE for f in rep["findings"])


def test_a_not_run_measurement_is_an_act_never_silence(cfg, tmp_path, reports_dir):
    """derive_thresholds writes {verdict: "NOT RUN -- ..."} with no value
    when it cannot measure I*. tune emitted nothing, status read "every
    MEASURED value matches", and the fixture's paste stayed in force
    unverified. It must surface as ACT, and --apply must refuse to paste
    a value that does not exist."""
    th = json.loads((reports_dir / "thresholds.json").read_text())
    th["information_increment_recommendation"] = {
        "verdict": "NOT RUN -- no per-category prior stds"}
    th["bounded_step_recommendation"] = {
        "verdict": "NOT RUN -- degenerate prior widths"}
    (reports_dir / "thresholds.json").write_text(json.dumps(th))
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    by_key = {f["key"]: f for f in rep["findings"]}
    inc = by_key["learning.information_increment"]
    assert inc["class"] == tune.PASTE and inc["status"] == tune.ACT
    assert inc["recommended"] is None and "NOT RUN" in inc["evidence"]
    rail = by_key["learning.max_mean_step"]
    assert rail["class"] == tune.OWNER and rail["status"] == tune.ACT

    # and the paster refuses it instead of writing "None" into config.yaml
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(open(os.path.join(ROOT, "config.yaml")).read())
    log = tune.apply(rep, config_path=str(cfg_path),
                     out_dir=str(tmp_path / "art"))
    text = cfg_path.read_text()
    assert "information_increment: None" not in text
    assert any(f["key"] == "learning.information_increment"
               for f in log["failed"])


def test_an_unmeasured_floor_is_named_not_skipped(cfg, tmp_path, reports_dir):
    th = json.loads((reports_dir / "thresholds.json").read_text())
    th["guardrail_threshold_recommendation"]["scrap_rate"] = {
        "config_key": "monitoring.stop_conditions.scrap_deterioration_pct",
        "verdict": "insufficient history on either basis"}
    (reports_dir / "thresholds.json").write_text(json.dumps(th))
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    hits = [f for f in rep["findings"]
            if f["key"] == "monitoring.stop_conditions.scrap_deterioration_pct"]
    assert hits and hits[0]["class"] == tune.INFO
    assert "insufficient" in hits[0]["evidence"]
    assert not any(f["key"] == hits[0]["key"] for f in rep["to_paste"])


def test_a_refit_that_raised_in_shadow_is_an_open_question(cfg, tmp_path, reports_dir):
    """shadow swallowed the weekly re-fit's exception into coverage, wrote
    weekly_refit null, and tune then read the cadence as settled."""
    sh = json.loads((reports_dir / "shadow.json").read_text())
    sh["calibration_regimes"] = {"frozen_anchor": 1.0002, "weekly_refit": None,
                                 "refit_error": "KeyError: 'd_ref'"}
    (reports_dir / "shadow.json").write_text(json.dumps(sh))
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    cad = [f for f in rep["findings"] if f["key"] == "calibration cadence"]
    assert cad and cad[0]["status"] == tune.ACT
    assert "KeyError" in cad[0]["evidence"]


def test_the_bottleneck_reads_the_population_rate_not_the_sample(
        cfg, tmp_path, reports_dir):
    """A --max-episodes shadow sample understates episodes/day and flips
    the reading to EVIDENCE when the calendar is the real limit."""
    sh = json.loads((reports_dir / "shadow.json").read_text())
    sh["window"] = {"date_min": "2026-08-10", "date_max": "2026-08-28",
                    "episodes": 2000, "population_episodes": 111400}
    (reports_dir / "shadow.json").write_text(json.dumps(sh))
    rep = tune.collect(_cfg_with(cfg, tmp_path), str(reports_dir))
    bn = [f for f in rep["findings"] if f["key"] == "learning bottleneck"][0]
    assert bn["current"] == "CALENDAR", bn["evidence"]
    assert "5,863 episodes/day" in bn["evidence"]      # 111400 / 19


def test_the_delta_min_bias_scale_is_the_largest_of_three_readings(cfg, tmp_path):
    """Each reading understates alone: the week-aggregate MAE averages
    noise away, the by_category ratios are one window, the gate band is a
    tolerance. tune takes the largest and pastes it; the fixture ships null."""
    root = tmp_path / "r"
    root.mkdir()
    _reports(root)
    bt = json.loads((root / "backtest.json").read_text())
    bt["fidelity"]["by_category"] = {"MEAT": 0.97, "FRUIT": 1.29, "VEG": 1.09}
    (root / "backtest.json").write_text(json.dumps(bt))
    c = _cfg_with(cfg, tmp_path)
    rep = tune.collect(c, str(root))
    f = next(x for x in rep["findings"] if x["key"] == "exploration.delta_min_log_bias")
    import numpy as np
    logs = np.log([0.97, 1.29, 1.09])
    rms = float(np.sqrt((logs ** 2).mean()))
    half = float(np.log(c["baseline_model"]["calibration_gate_band"][1]))
    chosen = f"trailing_{c['baseline_model']['calibration_fit_trailing_weeks']}w"
    mae = bt["fidelity"]["calibration_window_sweep"][chosen]["mean_abs_log_error"]
    floor = max(mae, half)
    # PER CATEGORY, each floored by the catalogue-wide noise and tolerance;
    # _default (the old scalar) for a category the backtest never saw
    assert f["recommended"] == {
        "MEAT": round(max(abs(np.log(0.97)), floor), 4),
        "FRUIT": round(max(abs(np.log(1.29)), floor), 4),
        "VEG": round(max(abs(np.log(1.09)), floor), 4),
        "_default": round(max(floor, rms), 4)}
    assert f["recommended"]["FRUIT"] > f["recommended"]["MEAT"]
    assert f["class"] == tune.PASTE and f["status"] == tune.ACT     # null in config

    # the mapping pastes as ONE line the anchor owns, and reads back equal
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(open(os.path.join(ROOT, "config.yaml")).read())
    tune.apply(rep, config_path=str(cfg_path), out_dir=str(tmp_path / "art"))
    from common.config import load_config as _lc
    pasted = _lc(str(cfg_path))["exploration"]["delta_min_log_bias"]
    assert pasted == f["recommended"]
    again = next(x for x in tune.collect(
        dict(c, exploration=dict(c["exploration"], delta_min_log_bias=pasted)),
        str(root))["findings"] if x["key"] == "exploration.delta_min_log_bias")
    assert again["status"] == tune.OK
