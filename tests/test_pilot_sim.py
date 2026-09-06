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


def _report(days=8, **over):
    """A clean run's record, small: every expectation should PASS."""
    def day(k):
        return {"day": k, "date": f"2026-09-{k:02d}", "lane_c": None,
                "calibration_current": {"pass": True, "held_at_anchor": False},
                "ingest": {"decisions": 50, "decisions_outside_feed_range": 10,
                           "outcomes_built": 40, "decisions_without_feed_row": 0},
                "gates": {"duplicate_or_unmatched_rate": True,
                          "price_mismatch_rate": True,
                          "calibration_schedule_current": True},
                "tau": {"committed": True,
                        "last_day": {"spend": 90.0, "budget": 100.0}},
                "stops": {"duplicate_or_unmatched": False, "price_mismatch": False,
                          "exploration_cost_vs_budget": False,
                          "scrap_deterioration_pct": False,
                          "margin_deterioration_pct": False},
                "assurance": {"reproduction": "PASS", "dispersion": "PASS",
                              "correlation": "INSUFFICIENT", "exploration": "PASS"},
                "assurance_detail": {"rho_live": 0.4},
                "learning": {"forced_decision_count": 30 * k,
                             "affordable_set_empty_rate": 0.2},
                "suspended": None,
                "apply": {"applied": True, "refused": None,
                          "calibration_schedule_current": True}}
    rep = {
        "world": {"faults": {}, "r_scale": 1.0},
        "engine": {"decisions": 400, "rejected": {}, "rejected_total": 0,
                   "violations": {"price_rose_within_episode": 0, "below_cost": 0},
                   "tau_at_launch": 300.0, "tau_now": 280.0},
        "learning": {"GLOBAL": {"version": 2, "abs_error_at_launch": 0.8,
                                "abs_error_now": 0.3, "launch_std": 2.0, "std": 1.5}},
        "lane_c": [{"schedule_end": "2026-09-07"}],
        "days": [day(k) for k in range(1, days + 1)],
    }
    rep.update(over)
    return rep


def test_a_clean_run_passes_every_expectation(cfg):
    cfg["learning"]["update_cadence_days"] = 1
    verdicts = {x["name"]: x["verdict"] for x in pilot_sim.grade(_report(), cfg)}
    assert set(verdicts) == {n for n, _ in pilot_sim.EXPECTATIONS}
    assert all(v == "PASS" for v in verdicts.values()), verdicts


def test_a_fault_turns_its_expectation_around(cfg):
    """With `mismatch` injected the mismatch gate MUST fail and the stop
    MUST fire; a run where they stayed silent is the failure."""
    cfg["learning"]["update_cadence_days"] = 1
    rep = _report()
    rep["world"]["faults"] = {"mismatch": 0.05}
    silent = {x["name"]: x for x in pilot_sim.grade(rep, cfg)}
    assert silent["event_quality_gates"]["verdict"] == "FAIL"
    assert silent["stops_only_on_faults"]["verdict"] == "FAIL"
    assert "price_mismatch" in silent["stops_only_on_faults"]["observed"]["expected_but_silent"]

    loud = copy.deepcopy(rep)
    loud["days"][3]["gates"]["price_mismatch_rate"] = False
    loud["days"][3]["stops"]["price_mismatch"] = True
    loud["days"][3]["apply"] = {"applied": False, "refused": "hard gate(s) failed",
                                "calibration_schedule_current": True}
    fired = {x["name"]: x for x in pilot_sim.grade(loud, cfg)}
    assert fired["event_quality_gates"]["verdict"] == "PASS"
    assert fired["stops_only_on_faults"]["verdict"] == "PASS"
    assert fired["apply_ran_on_cadence"]["verdict"] == "PASS"     # refusal explained

    # exploration silently off -- no forced decision for three days with a
    # budget in force -- is a finding, suspension is not
    off = copy.deepcopy(_report())
    for d in off["days"][2:6]:
        d["learning"]["forced_decision_count"] = off["days"][1]["learning"]["forced_decision_count"]
    assert {x["name"]: x["verdict"] for x in pilot_sim.grade(off, cfg)}[
        "exploration_never_starves"] == "FAIL"
    for d in off["days"][2:6]:
        d["suspended"] = {"reasons": ["price_mismatch"]}
    assert {x["name"]: x["verdict"] for x in pilot_sim.grade(off, cfg)}[
        "exploration_never_starves"] == "PASS"

    # and a stop with no fault behind it is the defect the sim exists to find
    stray = copy.deepcopy(_report())
    stray["days"][2]["stops"]["scrap_deterioration_pct"] = True
    assert {x["name"]: x["verdict"] for x in pilot_sim.grade(stray, cfg)}[
        "stops_only_on_faults"] == "FAIL"


def test_learning_is_graded_only_once_a_cell_updated(cfg):
    cfg["learning"]["update_cadence_days"] = 1
    rep = _report()
    rep["learning"]["GLOBAL"].update(version=0, abs_error_now=0.9, std=2.0)
    v = {x["name"]: x["verdict"] for x in pilot_sim.grade(rep, cfg)}
    assert v["learning_moves_toward_truth"] == "NOT MEASURED"
    assert v["posterior_narrows"] == "NOT MEASURED"
    rep["learning"]["GLOBAL"].update(version=1)
    v = {x["name"]: x["verdict"] for x in pilot_sim.grade(rep, cfg)}
    assert v["learning_moves_toward_truth"] == "FAIL"
    assert v["posterior_narrows"] == "FAIL"


def test_tau_is_graded_on_a_full_week_of_walking(cfg):
    cfg["learning"]["update_cadence_days"] = 1
    short = {x["name"]: x["verdict"] for x in pilot_sim.grade(_report(days=5), cfg)}
    assert short["tau_walks_on_spend"] == "NOT MEASURED"
    far = _report()
    for d in far["days"]:
        d["tau"]["last_day"] = {"spend": 10.0, "budget": 100.0}
    assert {x["name"]: x["verdict"] for x in pilot_sim.grade(far, cfg)}[
        "tau_walks_on_spend"] == "FAIL"


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
