"""ops.advance -- the order of operations as code. plan() is pure over
probe()'s state, so every branch is exercised without a workspace."""

from ops import advance
from ops import tune
import copy
import json
import sys
from common.provenance import config_fingerprint


def _state(**over):
    st = {
        "raw": True, "prepared": True, "model": True, "bundle": "b1",
        "retrain": False, "stale": {},
        "have": {"backtest", "thresholds", "shadow"},
        "tune": {"findings": [], "blocked": False, "to_paste": [],
                 "owner_decisions": []},
        "posterior": True, "environment_drift": [],
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
    assert steps[0]["args"] == ["fit.download_flc", "--start-date",
                                "2026-03-01", "--end-date", "2026-08-28"]
    assert steps[0]["reevaluate"]


def test_the_bootstrap_runs_only_when_the_model_is_absent_or_asked_for():
    """Rule 1: a retrain is a NEW bundle. The driver never does it by
    accident -- with a model and a bundle on disk it goes to --check-only."""
    assert _kinds(advance.plan(_state(model=False)))[0] == \
        ("run", "bootstrap", "bootstrap")
    assert _kinds(advance.plan(_state(retrain=True)))[0] == \
        ("run", "bootstrap", "bootstrap")
    assert all(s["args"][0] != "ops.bootstrap_loop" or "--check-only" in s["args"]
               for s in advance.plan(_state()) if s["kind"] == "run")


def test_a_stale_report_is_regraded_before_anything_is_pasted():
    steps = advance.plan(_state(stale={"backtest": "calibration: baseline_model.calibration_fit_trailing_weeks"},
                                tune={"findings": [], "blocked": False,
                                      "to_paste": [{"key": "dispersion.rho"}],
                                      "owner_decisions": []}))
    assert steps[0]["args"] == ["ops.bootstrap_loop", "--check-only"]
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
    assert steps[0]["args"] == ["ops.init_posterior"]


def test_shadow_runs_on_the_holdout_with_every_episode_and_then_gates():
    steps = advance.plan(_state(have={"backtest", "thresholds"}))
    assert steps[0]["args"][:2] == ["evaluate.shadow", "--input"]
    assert "--max-episodes" in steps[0]["args"] and "0" in steps[0]["args"]
    assert steps[0]["args"][-2:] == ["--workers", "0"]   # parallel, byte-identical
    assert "--all" not in steps[0]["args"]                # hold-out by default
    steps = advance.plan(_state(shadow_gate="FAIL -- completeness 0.97"))
    assert steps[0]["kind"] == "stop" and steps[0]["phase"] == "shadow"


def test_owner_values_stop_the_driver_with_their_evidence():
    finding = {"key": "monitoring.stop_conditions.scrap_deterioration_pct",
               "class": tune.PASTE, "status": tune.ACT, "current": None,
               "recommended": 0.27, "evidence": "3-sigma trailing floor",
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
    assert [s["args"][0] for s in steps] == ["fit.train_baseline",
                                             "ops.seal"]
    steps = advance.plan(_state(schedule_end="2026-08-17",
                                expected_schedule_end="2026-08-17"))
    assert steps[0]["kind"] == "stop" and "stale" in steps[0]["why"]


def test_the_daily_lane_ends_at_the_operator_gate_never_past_it():
    steps = advance.plan(_state(feed="feed.parquet"))
    mods = [s["args"][0] for s in steps if s["kind"] == "run"]
    assert mods == ["daily.ingest_outcomes", "daily.update",
                    "daily.monitor", "daily.assurance",
                    "daily.export_events", "ops.status"]
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
    journal = tmp_path / "journal.json"
    journal.write_text(json.dumps({"runs": [
        {"at": "2026-09-01T02:00:00+00:00", "phase": "bootstrap", "stop": None,
         "ran": [{"label": "bootstrap",
                  "command": "python3 -m ops.bootstrap_loop --input data/flc_raw.parquet"}]},
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
    assert "### bootstrap" in text and "ops.bootstrap_loop --input" in text
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
    reps = {n: {"artifact_versions": {"baseline_model_version": "b"},
                "config": config_fingerprint(cfg, n)}
            for n in ("backtest", "thresholds", "shadow")}
    assert advance.stale_reports(cfg, "b", reps) == {}
    # the pastes the process makes: none of them re-grades anything
    c = copy.deepcopy(cfg)
    c["exploration"]["tau_initial"] = 999.0
    c["dispersion"]["rho"] = 0.5
    c["baseline_model"]["calibration_gate_band"] = [0.95, 1.05]
    assert advance.stale_reports(c, "b", reps) == {}
    # the increment is graded by thresholds (configured vs I*): cheap re-derive
    c = copy.deepcopy(cfg)
    c["learning"]["information_increment"] = 0.9
    assert set(advance.stale_reports(c, "b", reps)) == {"thresholds"}
    # a category re-round of the per-category floor is shadow's, not the loop's
    c = copy.deepcopy(cfg)
    c["exploration"]["delta_min_log_bias"] = {"MEAT": 0.2, "_default": 0.1}
    reps2 = {n: {"artifact_versions": {"baseline_model_version": "b"},
                 "config": config_fingerprint(c, n)}
             for n in ("backtest", "thresholds", "shadow")}
    c2 = copy.deepcopy(c)
    c2["exploration"]["delta_min_log_bias"]["MEAT"] = 0.21
    assert set(advance.stale_reports(c2, "b", reps2)) == {"shadow"}
    # keys a report reads DO, and only the reports that read them
    c = copy.deepcopy(cfg)
    c["exploration"]["delta_min_log_bias"] = 0.2
    assert set(advance.stale_reports(c, "b", reps)) == {"shadow"}
    c = copy.deepcopy(cfg)
    c["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = 0.3
    assert set(advance.stale_reports(c, "b", reps)) == {"thresholds"}
    # pasted TOGETHER (one tune --apply) each still re-runs its own report:
    # the strongest class used to swallow the weaker one and thresholds was
    # never re-derived after the stop-threshold paste
    c = copy.deepcopy(cfg)
    c["exploration"]["delta_min_log_bias"] = 0.2
    c["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = 0.3
    c["monitoring"]["stop_conditions"]["margin_deterioration_pct"] = 0.1
    st = advance.stale_reports(c, "b", reps)
    assert set(st) == {"shadow", "thresholds"}
    assert st["thresholds"].startswith("thresholds:") and "delta_min" not in st["thresholds"]
    assert st["shadow"].startswith("shadow:") and "scrap" not in st["shadow"]
    c = copy.deepcopy(cfg)
    c["baseline_model"]["calibration_fit_trailing_weeks"] = 4
    assert set(advance.stale_reports(c, "b", reps)) == {"backtest", "thresholds", "shadow"}
    c = copy.deepcopy(cfg)
    c["pricing"]["tier_step"] = 0.05                      # an unclassified edit
    assert set(advance.stale_reports(c, "b", reps)) == {"backtest", "thresholds", "shadow"}
    # and a bundle mismatch always
    reps["backtest"]["artifact_versions"]["baseline_model_version"] = "old"
    assert "backtest" in advance.stale_reports(cfg, "b", reps)
    # a report with no fingerprint is stale: it cannot say what it graded
    reps["shadow"].pop("config")
    assert "shadow" in advance.stale_reports(cfg, "b", reps)
    assert "fingerprint" in advance.stale_reports(cfg, "b", reps)["shadow"]


def test_only_the_invalidated_report_is_re_run():
    steps = advance.plan(_state(stale={"thresholds": "thresholds: monitoring.stop_conditions.scrap_deterioration_pct"}))
    assert steps[0]["args"][0] == "evaluate.derive_thresholds"
    steps = advance.plan(_state(stale={"shadow": "shadow: exploration.delta_min_log_bias"}))
    assert steps[0]["args"][0] == "evaluate.shadow"
    # a key only the backtest reads re-runs the backtest, not the loop
    steps = advance.plan(_state(stale={"backtest": "backtest: posterior.cold_start_shift_std"}))
    assert steps[0]["args"][:2] == ["evaluate.backtest", "--input"]
    steps = advance.plan(_state(stale={"backtest": "calibration: pricing.tier_step"}))
    assert steps[0]["args"] == ["ops.bootstrap_loop", "--check-only"]
    # a training input moved: rule 1, a retrain is deliberate -- STOP
    steps = advance.plan(_state(stale={"backtest": "retrain: data.split.train_end",
                                       "thresholds": "retrain: data.split.train_end",
                                       "shadow": "retrain: data.split.train_end"}))
    assert steps[0]["kind"] == "stop" and "--retrain" in " ".join(steps[0]["detail"])
    # and the deliberate retrain seals under its own reason
    steps = advance.plan(_state(retrain=True))
    assert steps[0]["args"][-2:] == ["--seal-reason", "retrain"]


def test_a_moved_environment_is_resealed_once_nothing_is_left_to_paste(monkeypatch, tmp_path):
    """The seal covers config, code and libraries. After a paste round the
    config has moved: advance re-seals under `config` (a deploy under
    `deploy`) so every environment the bundle ran in has a snapshot, and
    the cheap seal never counts toward the loop guard."""
    drift = ["config moved since sealing: exploration.tau_initial"]
    steps = advance.plan(_state(environment_drift=drift))
    assert steps[0]["args"] == ["ops.seal", "--reason", "config"] and steps[0]["reevaluate"]
    steps = advance.plan(_state(environment_drift=["code moved since sealing: a -> b"]))
    assert steps[0]["args"] == ["ops.seal", "--reason", "deploy"]
    # a paste comes first: the seal records the settled config, not a draft
    st = _state(environment_drift=drift,
                tune={"findings": [], "blocked": False, "owner_decisions": [],
                      "to_paste": [{"key": "dispersion.rho"}]})
    assert advance.plan(st)[0]["kind"] == "paste"
    # three seals in one invocation are a normal chain (bootstrap, tau paste,
    # thresholds paste), never "looping"
    import sys
    calls = {"n": 0}

    def fake_execute(steps, *a):
        calls["n"] += 1
        return (calls["n"] < 4, False)
    monkeypatch.setattr(advance, "load_config", lambda p: {})
    monkeypatch.setattr(advance, "probe", lambda *a, **k: {})
    monkeypatch.setattr(advance, "plan", lambda st: [advance._run("re-seal", ["ops.seal", "--reason", "config"], phase="tune", reevaluate=True)])
    monkeypatch.setattr(advance, "render_plan", lambda s: "")
    monkeypatch.setattr(advance, "execute", fake_execute)
    monkeypatch.setattr(advance, "_write_readiness", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["advance", "--reports", str(tmp_path)])
    assert advance.main() == 0 and calls["n"] == 4


def test_a_measured_value_the_report_could_not_derive_is_not_an_owner_stop():
    """tau_initial null after shadow ran is shadow's derivation failing (a
    thin week), never a SET BY OWNER decision; the stop names the report."""
    steps = advance.plan(_state(nulls=["exploration.tau_initial"]))
    assert steps[0]["kind"] == "stop" and steps[0]["phase"] == "tune"
    assert "shadow.json" in " ".join(steps[0]["detail"])
    steps = advance.plan(_state(nulls=["learning.max_std_shrink"]))
    assert steps[0]["phase"] == "owner"


def test_the_round_budget_counts_work_not_stops(monkeypatch, tmp_path):
    """A legitimate STOP on the ninth plan was raised as "did not settle"
    before the stop was journaled or the readiness report written."""
    calls = {"n": 0}
    stop = [advance._stop("owner", "SET BY OWNER values are null", ["x"])]
    monkeypatch.setattr(advance, "load_config", lambda p: {})
    monkeypatch.setattr(advance, "probe", lambda *a, **k: {})
    monkeypatch.setattr(advance, "plan", lambda st: stop)
    monkeypatch.setattr(advance, "render_plan", lambda s: "")
    monkeypatch.setattr(advance, "execute", lambda *a: (False, False))
    monkeypatch.setattr(advance, "_write_readiness", lambda *a: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(advance, "MAX_TUNE_ROUNDS", 0)   # budget exhausted at once
    monkeypatch.setattr(sys, "argv", ["advance", "--reports", str(tmp_path)])
    assert advance.main() == 0 and calls["n"] == 1


def test_a_failed_step_is_a_journaled_stop_with_a_readiness_report(monkeypatch, tmp_path):
    """shadow refused (a thin pre-window week, no tau to derive) and advance
    died on the subprocess's SystemExit: no journal entry, no readiness
    report, the previous stop still on disk as if current. A failed step is
    a STOP like any other -- journaled, reported, and exit 1."""
    def boom(label, args, fatal=True):
        raise SystemExit(f"\n{label} FAILED (exit 1) -- stopping here")
    monkeypatch.setattr(advance, "step", boom)
    steps = [advance._run("shadow (hold-out)", ["evaluate.shadow"], phase="shadow",
                          reevaluate=True)]
    journal = tmp_path / "journal.json"
    again, failed = advance.execute(steps, "config.yaml", str(tmp_path), journal=str(journal))
    assert (again, failed) == (False, True)
    entry = json.loads(journal.read_text())["runs"][-1]
    assert entry["ran"] == [] and entry["stop"]["phase"] == "shadow"
    assert "FAILED" in entry["stop"]["why"] and "exit 1" in entry["stop"]["detail"][0]
    # and main writes the report and exits non-zero on it
    calls = {"n": 0}
    monkeypatch.setattr(advance, "load_config", lambda p: {})
    monkeypatch.setattr(advance, "probe", lambda *a, **k: {})
    monkeypatch.setattr(advance, "plan", lambda st: steps)
    monkeypatch.setattr(advance, "render_plan", lambda s: "")
    monkeypatch.setattr(advance, "execute", lambda *a: (False, True))
    monkeypatch.setattr(advance, "_write_readiness", lambda *a: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(sys, "argv", ["advance", "--reports", str(tmp_path)])
    assert advance.main() == 1 and calls["n"] == 1


def test_the_tau_paste_is_journaled_under_the_shadow_phase():
    st = _state(tune={"findings": [], "blocked": False, "owner_decisions": [],
                      "to_paste": [{"key": "exploration.tau_initial"}]})
    assert advance.plan(st)[0]["phase"] == "shadow"
    st = _state(tune={"findings": [], "blocked": False, "owner_decisions": [],
                      "to_paste": [{"key": "exploration.tau_initial"},
                                   {"key": "dispersion.rho"}]})
    assert advance.plan(st)[0]["phase"] == "tune"


def test_an_unlearned_posterior_is_reinitialised_when_the_launch_belief_moves():
    """posterior.cold_start_shift_std changed after init: the file is state,
    not a report, so no vintage check catches it. While it holds no outcome
    a --force re-init is safe and due; once it has learned, never."""
    pre = dict(launched=False, nulls=["data.launch_date"])
    steps = advance.plan(_state(posterior_stale=True, **pre))
    assert steps[0]["args"] == ["ops.init_posterior", "--force"]
    assert steps[0]["phase"] == "posterior"
    assert advance.plan(_state(posterior_stale=False, **pre))[0].get("args", [""])[0] \
        != "ops.init_posterior"
    # after launch the file is production state whatever launch_stale says
    # (tau walks, a suspension may stand): never re-initialised by the process
    assert all(s.get("args", [""])[0] != "ops.init_posterior"
               for s in advance.plan(_state(posterior_stale=True)))


def test_a_shadow_graded_on_a_retrained_bundle_is_rerun_before_tune_can_block():
    """After `--retrain` the backtest names the new bundle and shadow.json the
    old one; tune's 'reports agree on one model' invariant is a BLOCK, and a
    BLOCK stops the chain -- so the stale shadow must be re-run before tune
    is consulted, or the only exit is deleting the report by hand."""
    st = _state(stale={"shadow": "ran against bundle b0"},
                tune={"findings": [{"key": "reports agree on one model",
                                    "class": tune.BLOCK, "current": {},
                                    "recommended": "one version"}],
                      "blocked": True, "to_paste": [], "owner_decisions": []})
    steps = advance.plan(st)
    assert steps[0]["kind"] == "run" and steps[0]["args"][0] == "evaluate.shadow"
    # a CONFIG-stale shadow still waits for its place after the posterior step
    st = _state(stale={"shadow": "shadow: exploration.delta_min_log_bias"},
                posterior=False)
    assert advance.plan(st)[0]["args"] == ["ops.init_posterior"]
