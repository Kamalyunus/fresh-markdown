"""pipeline.advance -- the order of operations as code. plan() is pure over
probe()'s state, so every branch is exercised without a workspace."""

import copy

import pytest

from pipeline import advance
from pipeline import tune


def _state(**over):
    st = {
        "raw": True, "prepared": True, "model": True, "bundle": "b1",
        "retrain": False, "stale": [],
        "have": {"backtest", "thresholds", "shadow"},
        "tune": {"findings": [], "blocked": False, "to_paste": [],
                 "owner_decisions": []},
        "posterior": True,
        "shadow_gate": "PASS -- proceed to exploit-only pilot (section 19)",
        "nulls": [], "launched": True,
        "schedule_scope": "production -- launch_date 2026-09-01",
        "schedule_end": "2026-09-07", "expected_schedule_end": "2026-09-07",
        "this_week": "2026-08-31",
        "events": True, "feed": None,
        "status": {"failing": [], "checks": []},
    }
    st.update(over)
    return st


def _kinds(steps):
    return [(s["kind"], s.get("phase"), s.get("label") or s.get("why")) for s in steps]


def test_no_extract_stops_at_the_credentials_step():
    steps = advance.plan(_state(raw=False, prepared=False))
    assert steps[0]["kind"] == "stop" and steps[0]["phase"] == "data"


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
    steps = advance.plan(_state(stale=["backtest"],
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
    assert steps[-1]["kind"] == "stop" and "--apply" in steps[-1]["detail"][0]


def test_a_red_status_stops_before_the_daily_lane():
    steps = advance.plan(_state(status={"failing": ["exploration tau"], "checks": []}))
    assert steps[0]["kind"] == "stop" and "exploration tau" in steps[0]["detail"]


def test_render_marks_the_current_phase():
    out = advance.render_plan(advance.plan(_state(posterior=False)))
    assert "[posterior]" in out.splitlines()[0]
