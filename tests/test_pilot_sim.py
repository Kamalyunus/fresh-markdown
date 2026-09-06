"""The pilot simulator's world and grading -- the parts that need no
artifacts. The end-to-end walk lives in test_end_to_end.py."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from conftest import _hours
from evaluate import pilot_sim, pilot_world
from evaluate.pilot_world import World, episode_templates, hour_grid, parse_faults


def test_faults_parse_to_their_shape_and_reject_the_unknown():
    f = parse_faults(["missing:0.02", "demand_shock:5:0.6", "discount_rounding"])
    assert f == {"missing": 0.02, "demand_shock": (5, 0.6), "discount_rounding": True}
    with pytest.raises(ValueError):
        parse_faults(["mismatch:1.5"])
    with pytest.raises(ValueError):
        parse_faults(["typo:0.1"])
    assert set(pilot_world.FAULTS) >= set(f)


def test_the_hour_grid_crosses_midnight_like_a_window():
    grid = hour_grid("2026-08-29", 22, 4)
    assert grid == [("2026-08-29", 22), ("2026-08-29", 23),
                    ("2026-08-30", 0), ("2026-08-30", 1)]


def test_templates_keep_the_real_context_and_hold_the_last_legacy_price(cfg):
    """A template is a real DP-eligible episode: its window length is the
    first row's counter plus one, and a sell-out before the window's end
    holds the last discount over the unobserved tail."""
    d = pd.concat([_hours("A", "2026-08-12", 3, q0=3, tail=2, hour0=20),
                   _hours("B", "2026-08-01", 4),           # before the hold-out
                   _hours("C", "2026-08-15", 2, dp=False)])
    cfg["data"]["holdout"] = {"start": "2026-08-10", "end": "2026-08-28"}
    temps = episode_templates(d, cfg)
    assert [t["template_id"] for t in temps] == ["A"]
    t = temps[0]
    assert t["n_hours"] == 5 and t["opening_hour"] == 20 and t["q0"] == 3
    assert t["legacy_path"] == [0.30] * 5
    assert hour_grid("2026-09-01", t["opening_hour"], t["n_hours"])[-1] == ("2026-09-02", 0)
    with pytest.raises(ValueError):
        episode_templates(d, cfg, opened_from="2027-01-01")
    # a null counter is refused, never skipped: the chain owns that drop
    d.loc[d.episode_id == "A", "hours_remaining"] = np.nan
    with pytest.raises(ValueError, match="null_key_rows_dropped"):
        episode_templates(d, cfg)


def _world(cfg, eps=-1.2, faults=None, r=2.0):
    w = World.__new__(World)
    w.cfg, w.epsilon_true = cfg, {"FRUIT": eps}
    w.r_scale, w.drift, w.faults = 1.0, 1.0, dict(faults or {})
    w.episode_shock_sd = 0.0
    w.r_lookup = {"fallback_order": ["subcategory", "category", "global"],
                  "subcategory": {}, "category": {}, "global": r}
    w.rng = np.random.default_rng(0)
    return w


TPL = {"category": "FRUIT", "subcategory": "BERRY", "sku_id": 7, "fc": "FC1",
       "original_price": 10_000.0, "cost": 4_000.0}


def test_demand_answers_the_shelf_price_with_the_assumed_elasticity(cfg):
    """Deeper than the reference sells more, by ratio^epsilon exactly in
    the mean; the shock multiplies from its day on; draws are NB at r."""
    w = _world(cfg, eps=-1.5, faults={"demand_shock": (3, 0.5)})
    _, mu_ref = w.demand(TPL, 2.0, 0.30, day_index=0)          # d_ref is 0.30
    _, mu_deep = w.demand(TPL, 2.0, 0.40, day_index=0)
    assert mu_ref == pytest.approx(2.0)
    assert mu_deep == pytest.approx(2.0 * (0.6 / 0.7) ** -1.5)
    _, shocked = w.demand(TPL, 2.0, 0.30, day_index=3)
    assert shocked == pytest.approx(1.0)
    # the level the agent's re-fit should track: drift and the shock
    # fault, never the per-episode noise
    assert w.level_multiplier(2) == 1.0 and w.level_multiplier(3) == 0.5
    w.drift = 0.9
    assert w.level_multiplier(3) == pytest.approx(0.9 ** 3 * 0.5)
    assert w.demand(TPL, 2.0, 0.30, 3, episode_shock=2.0)[1] == pytest.approx(
        2.0 * 2.0 * w.level_multiplier(3))
    w.drift = 1.0
    draws = [w.demand(TPL, 2.0, 0.30, 0)[0] for _ in range(4000)]
    assert 1.8 < np.mean(draws) < 2.2
    assert np.var(draws) > np.mean(draws)                       # over-dispersed
    # the episode shock multiplies every hour of an episode, mean one
    assert w.episode_shock() == 1.0
    w.episode_shock_sd = 0.5
    shocks = [w.episode_shock() for _ in range(4000)]
    assert 0.9 < np.mean(shocks) < 1.1 and np.std(shocks) > 0.3
    assert w.demand(TPL, 2.0, 0.30, 0, episode_shock=2.0)[1] == pytest.approx(4.0)


def test_the_feed_row_is_the_source_convention(cfg):
    """Percent discount, the write-off sentinel already applied by the
    caller, no price on a no-sale hour, the nominal counter one below the
    state's hours_remaining -- and the rounding fault rounds the percent."""
    w = _world(cfg)
    row = w.feed_row(TPL, "2026-08-29", 21, q=5, shelf_discount=0.125, sold=0,
                     ending=5, hours_remaining=3)
    assert row["discount"] == 12.5 and row["final_price"] == 0.0
    assert row["flc_window"] == 2.0 and row["inventory"] == 5.0
    assert str(row["date"]) == "2026-08-29"
    w.faults["discount_rounding"] = True
    assert w.feed_row(TPL, "2026-08-29", 21, 5, 0.125, 1, 4, 3)["discount"] == 12.0
    frame = World.feed_frame([row])
    assert list(frame.columns) == [f.name for f in pilot_world.FEED_SCHEMA]


def test_the_sim_config_moves_only_the_state_paths(cfg):
    c = pilot_sim.sim_config(cfg, "sim/x", "2026-09-01")
    assert c["data"]["launch_date"] == "2026-09-01"
    for key in (("posterior", "path"), ("events", "store_dir"),
                ("baseline_model", "calibration_factor_path"),
                ("artifacts", "bundle_path"), ("artifacts", "history_dir"),
                ("data", "split_manifest_path")):
        node = c
        for k in key:
            node = node[k]
        assert node.startswith("sim/x")
    # the frozen artifacts and every tunable are production's
    for key in ("model_path", "feature_schema_path"):
        assert c["baseline_model"][key] == cfg["baseline_model"][key]
    assert c["dispersion"] == cfg["dispersion"]
    assert c["exploration"] == cfg["exploration"]
    assert cfg["data"]["launch_date"] is None                 # untouched


# the simulator's own grading knobs (pilot_sim.yaml `grading`), explicit
GRADING = {"spend_over_budget_band": [0.5, 1.25], "tau_week_days": 7,
           "starve_days": 3, "lane_hour": 6, "feature_history_margin_days": 14}


def _pin(cfg):
    """Every config value the grader reads, set explicitly: the tests
    grade against these, never against whatever the shipped file says."""
    cfg["learning"]["update_cadence_days"] = 1
    cfg["baseline_model"]["calibration_gate_band"] = [0.90, 1.10]
    cfg["monitoring"]["guardrail_noise_window_days"] = 28
    cfg["monitoring"]["shadow_gate"]["min_event_completeness"] = 0.99
    sc = cfg["monitoring"]["stop_conditions"]
    sc["deterioration_smoothing_days"] = {"scrap": 7, "margin": 7}
    sc["persistence_days"] = 2
    sc["duplicate_or_unmatched_rate"] = 0.01
    sc["price_mismatch_rate"] = 0.01
    sc["scrap_deterioration_pct"] = 0.3
    sc["margin_deterioration_pct"] = 0.06
    return cfg


def _grade(rep, cfg):
    return {x["name"]: x for x in pilot_sim.grade(rep, _pin(cfg), GRADING)}


def _verdicts(rep, cfg):
    return {n: x["verdict"] for n, x in _grade(rep, cfg).items()}


def _walk(day, spend=90.0, budget=100.0, held=None):
    return {"day": day, "spend": spend, "budget": budget, "tau": 300.0,
            "tau_after": 300.0, "clipped": False, "held": held}


def _report(days=8, **over):
    """A clean run's record, small: every expectation should PASS."""
    def day(k):
        date = (pd.Timestamp("2026-08-31") + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        return {"day": k, "lane_c": None, "date": date,
                "calibration_current": {"pass": True, "held_at_anchor": False},
                "ingest": {"decisions": 50, "decisions_outside_feed_range": 10,
                           "outcomes_built": 40, "decisions_without_feed_row": 0},
                "gates": {"duplicate_or_unmatched_rate": True,
                          "price_mismatch_rate": True,
                          "calibration_schedule_current": True},
                # the morning's walk: yesterday, the one day closed since
                "tau": {"committed": True, "walked": [_walk(
                    (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))]},
                "stops": {"duplicate_or_unmatched": False, "price_mismatch": False,
                          "exploration_cost_vs_budget": False,
                          "scrap_deterioration_pct": False,
                          "margin_deterioration_pct": False},
                "assurance": {"reproduction": "PASS", "dispersion": "PASS",
                              "correlation": "INSUFFICIENT", "exploration": "PASS"},
                "assurance_detail": {"rho_live": 0.4},
                "guardrails": {"scrap_deterioration_pct": {"latest": 0.01, "threshold": 0.3}},
                "learning": {"forced_decision_count": 30 * k,
                             "affordable_set_empty_rate": 0.2},
                "posterior": {"GLOBAL": {"mean": -1.2, "std": 1.5}},
                "suspended": None,
                "apply": {"applied": True, "refused": None,
                          "calibration_schedule_current": True}}
    rep = {
        "world": {"faults": {}, "r_scale": 1.0},
        "engine": {"decisions": 400, "rejected": {}, "rejected_total": 0,
                   "quarantined": 0, "pilot_hours": 400,
                   "violations": {"price_rose_within_episode": 0, "below_cost": 0},
                   "tau_at_launch": 300.0, "tau_now": 280.0},
        "learning": {"GLOBAL": {"version": 2, "epsilon_true": -1.2,
                                "abs_error_at_launch": 0.8,
                                "abs_error_now": 0.3, "launch_std": 2.0, "std": 1.5}},
        "lane_c": [{"schedule_end": "2026-09-07"}],
        "level_tracking": {"2026-08-31": {"hours": 300, "mean_log_ratio": 0.02,
                                          "p10_p90": [-0.1, 0.15],
                                          "mean_forced_log_move": -0.15,
                                          "implied_elasticity_bias": 0.133}},
        "days": [day(k) for k in range(1, days + 1)],
    }
    rep.update(over)
    return rep


def test_a_clean_run_passes_every_expectation(cfg):
    verdicts = _verdicts(_report(), cfg)
    assert set(verdicts) == {n for n, _ in pilot_sim.EXPECTATIONS}
    assert all(v == "PASS" for v in verdicts.values()), verdicts


def test_a_fault_turns_its_expectation_around(cfg):
    """With `mismatch` injected above the gate's threshold the mismatch
    gate MUST fail and the stop MUST fire; a run where they stayed silent
    is the failure. Graded per gate: the unmatched/duplicate gate is not
    expected to move under a mismatch."""
    rep = _report()
    rep["world"]["faults"] = {"mismatch": 0.05}
    silent = _grade(rep, cfg)
    assert silent["event_quality_gates"]["verdict"] == "FAIL"
    assert silent["event_quality_gates"]["observed"]["fault_expects_a_failure"] == {
        "duplicate_or_unmatched_rate": False, "price_mismatch_rate": True}
    assert silent["stops_only_on_faults"]["verdict"] == "FAIL"
    assert silent["stops_only_on_faults"]["observed"]["expected_but_silent"] == ["price_mismatch"]
    # a mismatch rate UNDER the gate's threshold expects nothing to fail
    rep["world"]["faults"] = {"mismatch": 0.005}
    assert _verdicts(rep, cfg)["event_quality_gates"] == "PASS"
    rep["world"]["faults"] = {"mismatch": 0.05}

    loud = copy.deepcopy(rep)
    loud["days"][3]["gates"]["price_mismatch_rate"] = False
    loud["days"][3]["stops"]["price_mismatch"] = True
    loud["days"][3]["apply"] = {"applied": False, "refused": "hard gate(s) failed",
                                "calibration_schedule_current": True}
    fired = _grade(loud, cfg)
    assert fired["event_quality_gates"]["verdict"] == "PASS"
    assert fired["stops_only_on_faults"]["verdict"] == "PASS"
    assert fired["apply_ran_on_cadence"]["verdict"] == "PASS"     # refusal explained
    # the OTHER gate failing under a mismatch is a defect, not the fault
    loud["days"][4]["gates"]["duplicate_or_unmatched_rate"] = False
    assert _verdicts(loud, cfg)["event_quality_gates"] == "FAIL"

    # a duplicated hour matches neither row and a missing one has none:
    # both are completeness gaps, expected once their summed rate exceeds
    # what the floor admits; neither reaches the unmatched/duplicate gate
    gap = _report()
    gap["world"]["faults"] = {"duplicate": 0.01, "missing": 0.005}
    g = _grade(gap, cfg)
    assert g["outcome_completeness"]["verdict"] == "FAIL"
    assert g["outcome_completeness"]["observed"]["fault_expects_a_gap"]
    assert g["event_quality_gates"]["verdict"] == "PASS"
    for d in gap["days"]:
        d["ingest"]["decisions_without_feed_row"] = 2
    assert _verdicts(gap, cfg)["outcome_completeness"] == "PASS"

    # exploration silently off -- no forced decision for starve_days with a
    # budget in force -- is a finding, suspension is not
    off = copy.deepcopy(_report())
    for d in off["days"][2:6]:
        d["learning"]["forced_decision_count"] = off["days"][1]["learning"]["forced_decision_count"]
    assert _verdicts(off, cfg)["exploration_never_starves"] == "FAIL"
    for d in off["days"][2:6]:
        d["suspended"] = {"reasons": ["price_mismatch"]}
    assert _verdicts(off, cfg)["exploration_never_starves"] == "PASS"

    # a shock the scrap series saw but that stays under the owner's floor is
    # the world's reach, reported and not graded
    seen = copy.deepcopy(_report(days=40))
    seen["world"]["faults"] = {"demand_shock": (20, 0.5)}
    seen["days"][-1]["guardrails"]["scrap_deterioration_pct"]["latest"] = 0.17
    v = _grade(seen, cfg)["stops_only_on_faults"]
    assert v["verdict"] == "NOT MEASURED" and v["observed"]["shock_seen_but_under_the_floor"]
    seen["days"][-1]["guardrails"]["scrap_deterioration_pct"]["latest"] = 0.0
    assert _verdicts(seen, cfg)["stops_only_on_faults"] == "FAIL"

    # and a stop with no fault behind it is the defect the sim exists to find
    stray = copy.deepcopy(_report())
    stray["days"][2]["stops"]["scrap_deterioration_pct"] = True
    assert _verdicts(stray, cfg)["stops_only_on_faults"] == "FAIL"


def test_the_engine_accounts_for_every_pilot_hour(cfg):
    """An hour priced but refused by the store (quarantined) is not a
    decision; the pilot's hours must equal decisions + rejections +
    quarantined, and a healthy run quarantines nothing."""
    rep = _report()
    rep["engine"].update(quarantined=3, decisions=397)
    assert _verdicts(rep, cfg)["hourly_engine"] == "FAIL"
    rep["engine"].update(quarantined=0, decisions=399)          # an hour unaccounted
    assert _verdicts(rep, cfg)["hourly_engine"] == "FAIL"
    rep["engine"].update(decisions=400)
    assert _verdicts(rep, cfg)["hourly_engine"] == "PASS"


def test_learning_is_graded_only_once_a_cell_updated(cfg):
    rep = _report()
    rep["learning"]["GLOBAL"].update(version=0, abs_error_now=0.9, std=2.0)
    v = _verdicts(rep, cfg)
    assert v["learning_moves_toward_truth"] == "NOT MEASURED"
    assert v["posterior_narrows"] == "NOT MEASURED"
    rep["learning"]["GLOBAL"].update(version=1)
    v = _verdicts(rep, cfg)
    assert v["learning_moves_toward_truth"] == "FAIL"
    assert v["posterior_narrows"] == "FAIL"
    # a cell with no simulated member category has no truth: not graded
    rep["learning"]["GLOBAL"].update(epsilon_true=None, abs_error_at_launch=None,
                                     abs_error_now=None)
    assert _verdicts(rep, cfg)["learning_moves_toward_truth"] == "NOT MEASURED"


def test_tau_is_graded_on_a_full_week_of_walking(cfg):
    """Over the last tau_week_days NON-HELD walked rows, against the sim's
    own spend-over-budget band; a morning's held rows are listed, never
    graded, and a day is walked once however many mornings carry it."""
    short = _verdicts(_report(days=5), cfg)
    assert short["tau_walks_on_spend"] == "NOT MEASURED"
    far = _report()
    for d in far["days"]:
        for w in d["tau"]["walked"]:
            w["spend"] = 10.0
    x = _grade(far, cfg)["tau_walks_on_spend"]
    assert x["verdict"] == "FAIL" and x["observed"]["band"] == [0.5, 1.25]
    # a wider sim band passes the same walk: the band is the sim's knob
    wide = {x["name"]: x["verdict"] for x in pilot_sim.grade(
        far, _pin(cfg), dict(GRADING, spend_over_budget_band=[0.05, 2.0]))}
    assert wide["tau_walks_on_spend"] == "PASS"
    # held mornings do not count toward the week, and only distinct days do
    held = _report(days=10)
    for d in held["days"][:4]:
        for w in d["tau"]["walked"]:
            w["held"] = "IL base shorter than budget_il_window_days"
    x = _grade(held, cfg)["tau_walks_on_spend"]
    assert x["verdict"] == "NOT MEASURED" and len(x["observed"]["days_held"]) == 4
    held["days"][-1]["tau"]["walked"] += [copy.deepcopy(w) for d in held["days"][4:]
                                          for w in d["tau"]["walked"]]
    x = _grade(held, cfg)["tau_walks_on_spend"]
    assert x["verdict"] == "NOT MEASURED" and x["observed"]["days_walked"] == 6


def test_the_schedule_reaches_a_week_it_deliberately_held():
    """A week the re-fit judged too thin is held at the frozen anchor by
    the applier; the --apply gate and advance's re-fit trigger read it as
    covered, not as a missed cron (the pilot simulator found the gate
    refusing every --apply of such a week)."""
    from fit.train_baseline import schedule_reaches

    assert schedule_reaches({}) is None
    assert schedule_reaches({"by_week": {"2026-08-17": {}, "2026-08-24": {}},
                             "weeks_unfitted_held_at_1": ["2026-08-31"]}) == "2026-08-31"
    assert schedule_reaches({"by_week": {"2026-08-24": {}},
                             "weeks_unfitted_held_at_1": ["2026-03-02"]}) == "2026-08-24"


def test_the_apply_gate_passes_a_held_week_and_says_so(tmp_path, cfg):
    from daily.update import calibration_current

    path = tmp_path / "calibration.json"
    cfg = dict(cfg, baseline_model=dict(cfg["baseline_model"],
                                        calibration_factor_path=str(path)))
    path.write_text(json.dumps({
        "grain": "category", "factors": {"FRUIT": 1.2},
        "schedule": {"by_week": {"2026-08-24": {"FRUIT": 1.1}},
                     "weeks_unfitted_held_at_1": ["2026-08-31"]}}))
    held = calibration_current(cfg, today="2026-09-02")
    assert held["pass"] and held["held_at_anchor"]
    assert "anchor" in held["note"]
    missed = calibration_current(cfg, today="2026-09-08")
    assert not missed["pass"] and not missed["held_at_anchor"]


def test_the_sim_config_is_the_home_of_the_settings_and_flags_override(tmp_path):
    """pilot_sim.yaml carries every setting; a missing key is an error, not
    a default hidden in code; a flag overrides its key for one run; the
    faults list is replaced, never merged."""
    import os
    from conftest import ROOT

    st = pilot_sim.load_sim_config(os.path.join(ROOT, "pilot_sim.yaml"))
    for k in ("days", "epsilon_true", "r_scale", "sim_dir", "config", "faults"):
        assert k in st
    assert st["config"] == "config.yaml" and st["faults"] == []
    over = pilot_sim.load_sim_config(
        os.path.join(ROOT, "pilot_sim.yaml"),
        {"days": 3, "faults": ["mismatch:0.1"], "epsilon_true_map": '{"MEAT": -0.9}',
         "seed": None})
    assert over["days"] == 3 and over["faults"] == ["mismatch:0.1"]
    assert over["epsilon_true_map"] == {"MEAT": -0.9} and over["seed"] == st["seed"]
    broken = tmp_path / "sim.yaml"
    broken.write_text("run: {days: 2}\nworld: {}\npaths: {}\n")
    with pytest.raises(ValueError, match="lacks"):
        pilot_sim.load_sim_config(str(broken))


def test_a_level_error_of_the_agent_is_named_not_read_as_learning(cfg):
    """The learner has no level term: a weekly re-fit that misses the
    world's level by more than the gate band is read as elasticity, so the
    report names the week rather than letting the learning row take the
    blame alone."""
    rep = _report()
    rep["level_tracking"]["2026-09-07"] = {"hours": 280, "mean_log_ratio": 0.31,
                                          "p10_p90": [0.1, 0.5],
                                          "mean_forced_log_move": -0.15,
                                          "implied_elasticity_bias": 2.07}
    x = _grade(rep, cfg)["agent_level_tracks_world"]
    assert x["verdict"] == "FAIL" and "2026-09-07" in x["observed"]["weeks_off"]
    assert "2026-08-31" not in x["observed"]["weeks_off"]
    # a level error inside the band still fails when the elasticity bias it
    # implies exceeds what the posterior admits (small forced moves)
    rep["level_tracking"] = {"2026-09-07": {"hours": 280, "mean_log_ratio": 0.08,
                                            "p10_p90": [0, 0.2],
                                            "mean_forced_log_move": -0.04,
                                            "implied_elasticity_bias": 2.0}}
    for d in rep["days"]:
        d["posterior"] = {"GLOBAL": {"std": 0.4}}
    x = _grade(rep, cfg)["agent_level_tracks_world"]
    assert x["verdict"] == "FAIL" and x["observed"]["weeks_off"]["2026-09-07"]["posterior_std"] == 0.4
    rep["level_tracking"] = {}
    assert _verdicts(rep, cfg)["agent_level_tracks_world"] == "NOT MEASURED"


def test_a_decision_draws_from_its_own_episode_and_hour():
    """Serial and parallel runs must price identically: the generator comes
    from the ids, never from a shared stream or the worker."""
    a = pilot_sim._decision_rng(0, "sim|pilot|7|F|2026-09-01T10", 3)
    b = pilot_sim._decision_rng(0, "sim|pilot|7|F|2026-09-01T10", 3)
    c = pilot_sim._decision_rng(0, "sim|pilot|7|F|2026-09-01T10", 4)
    d = pilot_sim._decision_rng(1, "sim|pilot|7|F|2026-09-01T10", 3)
    assert a.random() == b.random()
    assert a.random() != c.random() and a.random() != d.random()


def test_the_worker_prices_against_the_ticks_snapshot_and_reports_a_rejection(cfg):
    from engine.posterior import launch_belief
    state = {"episode_id": "e", "sku_id": 7, "fc": "F", "category": "FRUIT",
             "subcategory": "BERRY", "date": "2026-09-01", "hour_of_day": 10,
             "hours_remaining": 3, "q": 4, "original_price": 10_000.0,
             "cost": 4_000.0, "r": 2.0, "mu_ref_path": [1.0, 1.0, 1.0],
             "current_discount": None}
    ctx = {"cfg": cfg, "tau": 1e9,
           "cells": {"FRUIT": {"mean": -1.2, "std": 0.5, "version": 0}},
           "suspended": None, "model_version": "m", "digest": "d", "seed": 0}
    res = pilot_sim._price_one((state, ("e", 0)), ctx)
    assert res["rejected"] is None and res["evt"]["config_digest"] == "d"
    assert res["evt"]["tau_current"] == 1e9
    # a suspension in force prices with no budget, as production does
    held = pilot_sim._price_one((state, ("e", 0)), dict(ctx, suspended={"since": "x", "reasons": ["y"]}))
    assert held["evt"]["tau_current"] is None and not held["evt"]["is_exploration"]
    bad = pilot_sim._price_one((dict(state, q=-1), ("e", 0)), ctx)
    assert bad["evt"] is None and bad["rejected"]
    assert launch_belief(-1.2, 0.5, cfg) < 0


# ------------------------------------------------- the pairing, in the small

def _template(tid, sku, n_hours=2, opening_hour=10, q0=4):
    return {"template_id": tid, "sku_id": sku, "fc": "F", "category": "FRUIT",
            "subcategory": "BERRY", "original_price": 10_000.0, "cost": 4_000.0,
            "opening_hour": opening_hour, "n_hours": n_hours, "q0": q0,
            "legacy_path": [0.30] * n_hours}


def _sim(cfg, templates, monkeypatch, per_day=1, shocks=None):
    """A PilotSim with only what `_sample_day`, `_open_due` and `_sell`
    touch, over a stub world: no artifacts, no store, unit demand."""
    from evaluate.pilot_sim import HIST_COLS, PilotSim

    shocks = iter(shocks or [])
    w = World.__new__(World)
    w.cfg, w.templates, w.epsilon_true = cfg, templates, {"FRUIT": -1.2}
    w.episode_shock = lambda: next(shocks, 1.0)
    w.mu_ref_paths = lambda openings, model=None: [[1.0] * len(o["grid"]) for o in openings]
    w.demand = lambda tpl, mu, shelf, k, shock=1.0: (0, mu)
    w.level_multiplier = lambda k: 1.0
    w.feed_row = lambda *a, **kw: {}
    w.draw_fault = lambda name: False
    monkeypatch.setattr(pilot_sim, "ref_rate_features",
                        lambda hist, stub, cfg: {e: (np.nan, np.nan) for e in stub.episode_id})
    s = PilotSim.__new__(PilotSim)
    s.cfg, s.world, s.rng, s.per_day, s.model = cfg, w, np.random.default_rng(0), per_day, None
    s.history, s.history_days, s.sim_frames = pd.DataFrame(columns=HIST_COLS), 44, {}
    s.open, s.pending, s.busy = [], {}, set()
    s.twins_due, s._twin_of, s.shock_by_template = {}, {}, {}
    s.opened_by_day, s.feed_by_day = {}, {}
    s._truth_rows, s._hist_rows = [], []
    return s


def _close(s, ep, k):
    while ep in s.open:
        s._sell(ep, k, 0.30)


def test_a_fresh_picks_twin_runs_the_day_after_it_closes_under_the_other_arm(cfg, monkeypatch):
    s = _sim(cfg, [_template("A", 1), _template("B", 2)], monkeypatch,
             shocks=[1.5, 0.7, 2.0, 2.0])
    s._sample_day(0, "2026-09-01")
    day0 = s.pending["2026-09-01"]
    arms0 = {o["template"]["template_id"]: o["arm"] for o in day0}
    shocks0 = {o["template"]["template_id"]: o["shock"] for o in day0}
    assert sorted(arms0.values()) == ["legacy", "pilot"]
    assert all(o["twin"] == ("legacy" if o["arm"] == "pilot" else "pilot") for o in day0)
    s._open_due(0, "2026-09-01", 10)
    assert "2026-09-01" not in s.pending and len(s.open) == 2      # emptied, deleted
    assert s.busy == {(1, "F"), (2, "F")}
    for ep in list(s.open):
        _close(s, ep, 0)
    assert not s.open and not s.busy
    assert sorted(a for a, _ in s.twins_due["2026-09-02"]) == ["legacy", "pilot"]
    s._sample_day(1, "2026-09-02")
    day1 = s.pending["2026-09-02"]
    arms1 = {o["template"]["template_id"]: o["arm"] for o in day1}
    assert arms1 == {t: ("legacy" if a == "pilot" else "pilot") for t, a in arms0.items()}
    assert all(o["twin"] is None for o in day1)                    # a twin has none
    # the twin sees the same world: the episode's shock is shared
    assert {o["template"]["template_id"]: o["shock"] for o in day1} == shocks0
    assert sorted(shocks0.values()) == [0.7, 1.5]


def test_a_busy_sku_fc_postpones_the_twin_and_it_runs_later(cfg, monkeypatch):
    a = _template("A", 1)
    s = _sim(cfg, [a], monkeypatch)
    s.twins_due["2026-09-02"] = [("legacy", a)]
    s.busy.add((1, "F"))                     # another episode of that sku x fc is open
    s._sample_day(1, "2026-09-02")
    assert s.pending["2026-09-02"] == [] and s.opened_by_day["2026-09-02"] == 0
    assert s.twins_due == {"2026-09-03": [("legacy", a)]}
    s.busy.discard((1, "F"))
    s._sample_day(2, "2026-09-03")
    (o,) = s.pending["2026-09-03"]
    assert o["arm"] == "legacy" and o["template"]["template_id"] == "A" and o["twin"] is None
    assert not s.twins_due
    # a reserved twin is not re-picked fresh while it waits, and a fresh
    # pick never opens a busy sku x fc
    s = _sim(cfg, [a, _template("B", 2)], monkeypatch, per_day=2)
    s.twins_due["2026-09-02"] = [("legacy", a)]
    s.busy.add((1, "F"))
    s._sample_day(1, "2026-09-02")
    assert [o["template"]["template_id"] for o in s.pending["2026-09-02"]] == ["B"]


def _truth(eid, tid, arm, date, sold=(1, 1), q0=4, mu_world=2.0, mu_agent=None):
    """One closed episode's hours in TRUTH_COLS: `sold` per hour, the
    write-off sentinel on the last."""
    rows, q = [], q0
    for i, u in enumerate(sold):
        ending = 0 if i == len(sold) - 1 else q - u
        rows.append((eid, tid, arm, date, 10 + i, q, u, ending, 10_000.0, 7_000.0,
                     4_000.0, "FRUIT", "F", 1, True, 0.30, mu_world, mu_world,
                     mu_agent if arm == "pilot" else None))
        q -= u
    return rows


def test_the_paired_economics_cover_only_templates_settled_under_both_arms(cfg):
    from evaluate.pilot_sim import TRUTH_COLS, PilotSim

    s = PilotSim.__new__(PilotSim)
    rows = (_truth("p|A", "A", "pilot", "2026-09-01") + _truth("l|A", "A", "legacy", "2026-09-02")
            + _truth("p|B", "B", "pilot", "2026-09-01", sold=(2, 2))
            + _truth("l|C", "C", "legacy", "2026-09-01", sold=(0, 0)))
    econ = s.economics(pd.DataFrame(rows, columns=TRUTH_COLS))
    assert econ["pilot"]["episodes"] == 2 and econ["legacy"]["episodes"] == 2
    paired = econ["paired"]
    assert paired["templates"] == 1 and paired["unpaired_templates"] == 2
    assert paired["pilot"]["episodes"] == 1 and paired["legacy"]["episodes"] == 1
    assert paired["pilot"]["hours"] == 2
    # like for like: the same template sold the same, so the arms agree
    assert paired["pilot"]["il_absolute"] == paired["legacy"]["il_absolute"]
    assert paired["pilot"]["scrap_units"] == 2 and econ["pilot"]["scrap_units"] == 2
    # per arm the legacy figure carries C's scrap (4) the paired block does not
    assert econ["legacy"]["scrap_units"] == 6 and paired["legacy"]["scrap_units"] == 2
    # the mean discount is over settled episodes' hours: an open episode's
    # hours are not in it
    rows += _truth("p|D", "D", "pilot", "2026-09-03", sold=(1,))[:1]
    open_rows = pd.DataFrame(rows, columns=TRUTH_COLS)
    open_rows.loc[open_rows.episode_id == "p|D", ["ending_inventory", "shelf_discount"]] = [3, 0.60]
    econ = s.economics(open_rows)
    assert econ["pilot"]["excluded"]["episodes_excluded_not_closed"] == 1
    assert econ["pilot"]["mean_discount"] == 0.30 and econ["pilot"]["hours"] == 4


def test_the_implied_elasticity_bias_has_the_sign_of_the_level_error(cfg):
    """A level error e read against SIGNED forced moves of mean L (negative:
    deeper) is an elasticity error of -e / L: a positive level error over
    deeper moves biases the belief toward zero (positive)."""
    from evaluate.pilot_sim import TRUTH_COLS, PilotSim

    s = PilotSim.__new__(PilotSim)
    e = 0.1
    rows = (_truth("p|A", "A", "pilot", "2026-09-01", mu_agent=2.0 * np.exp(e))
            + _truth("l|A", "A", "legacy", "2026-09-02"))
    forced = [{"date": "2026-09-01", "reference_discount": 0.30, "applied_discount": d,
               "is_exploration": True} for d in (0.40, 0.35)]
    exploit = [{"date": "2026-09-01", "reference_discount": 0.30, "applied_discount": 0.20,
                "is_exploration": False}]
    out = s.level_tracking(forced + exploit, pd.DataFrame(rows, columns=TRUTH_COLS))
    assert list(out) == ["2026-08-31"]
    wk = out["2026-08-31"]
    L = np.mean([np.log(0.6 / 0.7), np.log(0.65 / 0.7)])
    assert wk["hours"] == 2 and wk["mean_log_ratio"] == pytest.approx(e, abs=1e-4)
    assert wk["mean_forced_log_move"] == pytest.approx(L, abs=1e-4) and L < 0
    assert wk["implied_elasticity_bias"] == pytest.approx(-e / L, abs=1e-3)
    assert wk["implied_elasticity_bias"] > 0
    # a level error the other way biases the belief away from zero
    rows = _truth("p|A", "A", "pilot", "2026-09-01", mu_agent=2.0 * np.exp(-e))
    out = s.level_tracking(forced, pd.DataFrame(rows, columns=TRUTH_COLS))
    assert out["2026-08-31"]["implied_elasticity_bias"] == pytest.approx(e / L, abs=1e-3)
    assert out["2026-08-31"]["implied_elasticity_bias"] < 0
    # no forced decision that week: the level is reported, the bias is not
    out = s.level_tracking(exploit, pd.DataFrame(rows, columns=TRUTH_COLS))
    assert out["2026-08-31"]["implied_elasticity_bias"] is None
