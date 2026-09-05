"""pipeline.advance -- the order of operations as code. plan() is pure over
probe()'s state, so every branch is exercised without a workspace."""

import copy

import pytest

from pipeline import advance
from pipeline import tune


def _state(**over):
    st = {
        "raw": True, "prepared": True, "model": True, "bundle": "b1",
        "retrain": False, "stale": {},
        "have": {"backtest", "thresholds", "shadow"},
        "tune": {"findings": [], "blocked": False, "to_paste": [],
                 "owner_decisions": []},
        "posterior": True,
        "shadow_gate": "PASS -- proceed to exploit-only pilot (section 19)",
        "nulls": [], "launched": True,
        "schedule_scope": "production -- launch_date 2026-09-01",
        "schedule_end": "2026-09-07", "expected_schedule_end": "2026-09-07",
        "this_week": "2026-08-31",
        "events": True, "feed": None, "cadence": 7,
        "extract_range": ("2026-03-01", "2026-08-28"),
        "status": {"failing": [], "checks": []},
    }
    st.update(over)
    return st


def _kinds(steps):
    return [(s["kind"], s.get("phase"), s.get("label") or s.get("why")) for s in steps]


def test_no_extract_pulls_it_per_the_config_dates():
    """The extract is not a human step: the config's split and hold-out
    dates size the pull, and download_flc fails loudly without REDSHIFT_*."""
    steps = advance.plan(_state(raw=False, prepared=False))
    assert steps[0]["kind"] == "run" and steps[0]["phase"] == "data"
    assert steps[0]["args"] == ["bootstrap.download_flc", "--start-date",
                                "2026-03-01", "--end-date", "2026-08-28"]
    assert steps[0]["reevaluate"]


def test_the_bootstrap_runs_only_when_the_model_is_absent_or_asked_for():
    """Rule 1: a retrain is a NEW bundle. The driver never does it by
    accident -- with a model and a bundle on disk it goes to --check-only."""
    assert _kinds(advance.plan(_state(model=False)))[0] == \
        ("run", "bootstrap", "bootstrap")
    assert _kinds(advance.plan(_state(retrain=True)))[0] == \
        ("run", "bootstrap", "bootstrap")
    assert all(s["args"][0] != "bootstrap.run" or "--check-only" in s["args"]
               for s in advance.plan(_state()) if s["kind"] == "run")


def test_a_stale_report_is_regraded_before_anything_is_pasted():
    steps = advance.plan(_state(stale={"backtest": "calibration: baseline_model.calibration_fit_trailing_weeks"},
                                tune={"findings": [], "blocked": False,
                                      "to_paste": [{"key": "dispersion.rho"}],
                                      "owner_decisions": []}))
    assert steps[0]["args"] == ["bootstrap.run", "--check-only"]
    assert steps[0]["reevaluate"]


def test_pastes_are_applied_then_the_chain_is_reprobed():
    st = _state(tune={"findings": [], "blocked": False,
                      "to_paste": [{"key": "dispersion.rho"}],
                      "owner_decisions": []})
    steps = advance.plan(st)
    assert steps[0]["kind"] == "paste" and steps[0]["keys"] == ["dispersion.rho"]
    assert steps[0]["reevaluate"]


def test_a_tune_block_stops_the_driver_but_missing_reports_do_not():
    block = {"key": "calibration converged", "class": tune.BLOCK,
             "status": tune.ACT, "current": "NO", "recommended": "YES"}
    steps = advance.plan(_state(tune={"findings": [block], "blocked": True,
                                      "to_paste": [], "owner_decisions": []}))
    assert steps[0]["kind"] == "stop" and steps[0]["phase"] == "tune"
    missing = dict(block, key="reports present")
    steps = advance.plan(_state(posterior=False,
                                tune={"findings": [missing], "blocked": True,
                                      "to_paste": [], "owner_decisions": []}))
    assert steps[0]["args"] == ["bootstrap.init_posterior"]


def test_shadow_runs_on_the_holdout_with_every_episode_and_then_gates():
    steps = advance.plan(_state(have={"backtest", "thresholds"}))
    assert steps[0]["args"][:2] == ["pipeline.shadow", "--input"]
    assert steps[0]["args"][-2:] == ["--max-episodes", "0"]
    assert "--all" not in steps[0]["args"]                # hold-out by default
    steps = advance.plan(_state(shadow_gate="FAIL -- completeness 0.97"))
    assert steps[0]["kind"] == "stop" and steps[0]["phase"] == "shadow"


def test_owner_values_stop_the_driver_with_their_evidence():
    finding = {"key": "monitoring.stop_conditions.scrap_deterioration_pct",
               "class": tune.PASTE, "status": tune.ACT, "current": None,
               "recommended": 0.27, "evidence": "3-sigma control_arm floor",
               "source": "thresholds.guardrail_threshold_recommendation.scrap"}
    steps = advance.plan(_state(
        nulls=["monitoring.stop_conditions.scrap_deterioration_pct",
               "data.launch_date"],
        tune={"findings": [finding], "blocked": False, "to_paste": [],
              "owner_decisions": [finding]}))
    assert steps[0]["kind"] == "stop" and steps[0]["phase"] == "owner"
    assert "3-sigma" in steps[0]["detail"][0]
    # launch_date is its own, later stop -- not an owner value to derive
    assert not any("launch_date" in d for d in steps[0]["detail"])


def test_launch_day_refits_the_schedule_once_and_stops_on_a_stale_extract():
    steps = advance.plan(_state(launched=False))
    assert steps[0]["phase"] == "launch" and "launch_date" in steps[0]["why"]
    steps = advance.plan(_state(schedule_scope="pre-launch -- through 2026-08-09"))
    assert [s["args"][0] for s in steps] == ["bootstrap.train_baseline",
                                             "bootstrap.seal"]
    steps = advance.plan(_state(schedule_end="2026-08-17",
                                expected_schedule_end="2026-08-17"))
    assert steps[0]["kind"] == "stop" and "stale" in steps[0]["why"]


def test_the_daily_lane_ends_at_the_operator_gate_never_past_it():
    steps = advance.plan(_state(feed="feed.parquet"))
    mods = [s["args"][0] for s in steps if s["kind"] == "run"]
    assert mods == ["pipeline.ingest_outcomes", "pipeline.update",
                    "pipeline.monitor", "pipeline.assurance",
                    "pipeline.export_events", "pipeline.status"]
    assert all("--apply" not in s["args"] for s in steps if s["kind"] == "run")
    assert "--calibrate-tau" in steps[1]["args"]          # tau is daily, no operator
    assert "every 7 days" in steps[-1]["detail"][0]
    assert steps[-1]["kind"] == "stop" and "--apply" in steps[-1]["detail"][0]


def test_a_red_status_stops_before_the_daily_lane():
    steps = advance.plan(_state(status={"failing": ["exploration tau"], "checks": []}))
    assert steps[0]["kind"] == "stop" and "exploration tau" in steps[0]["detail"]


def test_render_marks_the_current_phase():
    out = advance.render_plan(advance.plan(_state(posterior=False)))
    assert "[posterior]" in out.splitlines()[0]


def test_the_readiness_report_is_assembled_from_the_journal_and_the_decision_log(
        cfg, tmp_path, monkeypatch):
    """What ran per phase, every value the process changed and why, the
    config in force, status, and what is waited on -- from files, never
    from memory."""
    import json
    journal = tmp_path / "journal.json"
    journal.write_text(json.dumps({"runs": [
        {"at": "2026-09-01T02:00:00+00:00", "phase": "bootstrap", "stop": None,
         "ran": [{"label": "bootstrap",
                  "command": "python3 -m bootstrap.run --input data/flc_raw.parquet"}]},
        {"at": "2026-09-01T03:00:00+00:00", "phase": "tune", "stop": None,
         "ran": [{"label": "tune --apply", "pasted": ["dispersion.rho"], "skipped": []}]},
        {"at": "2026-09-01T04:00:00+00:00", "phase": "launch", "ran": [],
         "stop": {"phase": "launch", "why": "data.launch_date is null",
                  "detail": ["set it on launch day"]}},
    ]}))
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"runs": [
        {"at": "2026-09-01T03:00:00+00:00", "applied": [
            {"key": "dispersion.rho", "current": 0.1161, "recommended": 0.2436,
             "evidence": "config mirrors the frozen artifact",
             "source": "artifacts/rho.json rho"}]}]}))
    text = advance.report(cfg, str(tmp_path / "no_reports"),
                          journal=str(journal), decisions=str(decisions))
    assert "### bootstrap" in text and "bootstrap.run --input" in text
    assert "### tune" in text and "pasted `dispersion.rho`" in text
    assert "| `dispersion.rho` | 0.1161 → 0.2436 |" in text
    assert "## Config in force" in text and "SET BY OWNER" in text
    assert "## Status" in text
    assert "**[launch] data.launch_date is null**" in text
    assert "set it on launch day" in text


def test_a_measured_paste_does_not_stale_the_report_that_derived_it(cfg):
    """Every digest change used to stale every report: paste tau -> shadow
    stale -> re-run shadow (hours) -> slightly different tau -> paste ->
    ... for a day on the owner's extract. Staleness is judged on the keys a
    report actually READS (tune.rerun_for)."""
    import copy
    from common.provenance import config_fingerprint
    reps = {n: {"artifact_versions": {"baseline_model_version": "b"},
                "config": config_fingerprint(cfg, n)}
            for n in ("backtest", "thresholds", "shadow")}
    assert advance.stale_reports(cfg, "b", reps) == {}
    # the pastes the process makes: none of them re-grades anything
    c = copy.deepcopy(cfg)
    c["exploration"]["tau_initial"] = 999.0
    c["dispersion"]["rho"] = 0.5
    c["learning"]["information_increment"] = 0.9
    c["baseline_model"]["calibration_gate_band"] = [0.95, 1.05]
    assert advance.stale_reports(c, "b", reps) == {}
    # keys a report reads DO, and only the reports that read them
    c = copy.deepcopy(cfg)
    c["exploration"]["delta_min_log_bias"] = 0.2
    assert set(advance.stale_reports(c, "b", reps)) == {"shadow"}
    c = copy.deepcopy(cfg)
    c["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = 0.3
    assert set(advance.stale_reports(c, "b", reps)) == {"thresholds"}
    c = copy.deepcopy(cfg)
    c["baseline_model"]["calibration_fit_trailing_weeks"] = 4
    assert set(advance.stale_reports(c, "b", reps)) == {"backtest", "thresholds", "shadow"}
    c = copy.deepcopy(cfg)
    c["pricing"]["tier_step"] = 0.05                      # an unclassified edit
    assert set(advance.stale_reports(c, "b", reps)) == {"backtest", "thresholds", "shadow"}
    # and a bundle mismatch always
    reps["backtest"]["artifact_versions"]["baseline_model_version"] = "old"
    assert "backtest" in advance.stale_reports(cfg, "b", reps)


def test_only_the_invalidated_report_is_re_run():
    steps = advance.plan(_state(stale={"thresholds": "thresholds: monitoring.stop_conditions.scrap_deterioration_pct"}))
    assert steps[0]["args"][0] == "bootstrap.derive_thresholds"
    steps = advance.plan(_state(stale={"shadow": "shadow: exploration.delta_min_log_bias"}))
    assert steps[0]["args"][0] == "pipeline.shadow"
