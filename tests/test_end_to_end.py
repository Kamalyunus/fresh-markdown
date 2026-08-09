"""End-to-end smoke over the synthetic generator: bootstrap chain, decision
loop, learning update. Exercises the PRD section 1a order on a small
randomized-policy dataset."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("mvp")
    with open(os.path.join(REPO, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    # shrink the expensive knobs for test speed; semantics unchanged
    cfg["baseline_model"]["num_boost_round"] = 60
    cfg["posterior"]["prior"]["search_grid_size"] = 41
    (ws / "config.yaml").write_text(yaml.safe_dump(cfg))

    env = {**os.environ, "PYTHONPATH": REPO}

    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=ws, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    run(os.path.join(REPO, "tools", "make_dummy_flc.py"),
        "--skus", "120", "--days", "160", "--policy", "randomized",
        "--seed", "3", "--out", "data/flc.parquet")
    run("-m", "bootstrap.prepare_data", "--input", "data/flc.parquet",
        "--out", "data/prepared.parquet")
    run("-m", "bootstrap.measure", "--input", "data/flc.parquet",
        "--out", "reports/phase0.json")
    run("-m", "bootstrap.train_baseline", "--input", "data/prepared.parquet")
    run("-m", "bootstrap.fit_dispersion", "--input", "data/prepared.parquet")
    run("-m", "bootstrap.estimate_prior", "--input", "data/prepared.parquet")
    run("-m", "backtest", "--input", "data/prepared.parquet",
        "--out", "reports/backtest.json", "--policy-episodes", "150")
    return ws


def _chdir(ws):
    os.chdir(ws)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def test_filter_chain_waterfall(workspace):
    _chdir(workspace)
    with open("artifacts/split_manifest.json") as f:
        manifest = json.load(f)
    wf = manifest["data_quality_waterfall"]
    rows = [s["rows"] for s in wf if s["step"] != "contiguous_episodes_built"]
    assert rows == sorted(rows, reverse=True)
    d = pd.read_parquet("data/prepared.parquet")
    assert d.category.notna().all()
    assert (d.units_sold <= d.starting_inventory).all()
    assert d.total_discount.between(0, 1).all()      # percent -> fraction once
    assert (d.original_price > 0).all()


def test_phase0_report_complete(workspace):
    _chdir(workspace)
    with open("reports/phase0.json") as f:
        res = json.load(f)
    for key in ["m1_cost_ratio", "m2_same_hour_variation",
                "m3_intra_episode_correlation", "m4_demand_density",
                "m5_censoring", "m6_il_pct", "m7_learning_rate",
                "m8_entry_hour", "reassessment_gates"]:
        assert key in res
    assert res["m5_censoring"]["episodes_reaching_zero_inventory"] < 1.0
    m3 = res["m3_intra_episode_correlation"]
    assert m3["implied_deff"] >= 1.0


def test_prior_artifact_within_bounds(workspace):
    _chdir(workspace)
    with open("artifacts/prior.json") as f:
        prior = json.load(f)
    assert prior["source"] in ("bracket", "fallback")
    lo, hi = prior["search_bounds"]
    for cat, v in prior["per_category"].items():
        assert lo <= v["epsilon_naive"] <= hi
        assert lo <= v["epsilon_controlled"] <= hi
        assert v["std"] > 0 and v["mean"] < 0


def test_backtest_blocks_reported_separately(workspace):
    _chdir(workspace)
    with open("reports/backtest.json") as f:
        bt = json.load(f)
    assert "fidelity" in bt and "policy_deltas" in bt
    assert bt["fidelity"]["fidelity_episode_sold_ratio"] > 0
    pol = bt["policy_deltas"]
    # absolute IL reported alongside every IL% figure (PRD 3.6 / 17.4)
    assert "actual_il" in pol and "actual_il_pct" in pol
    assert "dp_il" in pol and "dp_il_pct" in pol
    # policy comparison must be apples-to-apples: both arms under the model
    assert "legacy_model_il" in pol
    gap = pol["policy_gap_like_for_like"]
    assert abs(gap["dp_minus_legacy_il"]
               - (pol["dp_il"] - pol["legacy_model_il"])) < 1.0
    # the DP optimises expected IL under this same model; like-for-like it
    # should not lose materially (small slack: tier grid, entry band, and
    # the deterministic transition differ slightly from the DP's objective)
    if pol["legacy_model_il"] > 0:
        assert gap["dp_minus_legacy_il"] <= 0.05 * pol["legacy_model_il"]
    tau = bt["tau_initial_derivation"]
    assert tau is None or tau["tau_initial"] >= 0


def test_decision_loop_and_exactly_once_update(workspace):
    _chdir(workspace)
    from common.config import load_config
    from bootstrap.train_baseline import BaselineModel
    from backtest.replay import _attach_predictions
    from pricing.posterior import PosteriorStore
    from events.store import EventStore, DECISION_REQUIRED
    from inference.decide import decide
    from pipeline.update import run as update_run

    cfg = load_config("config.yaml")
    with open(cfg["posterior"]["prior"]["path"]) as f:
        prior = json.load(f)
    with open(cfg["dispersion"]["r_lookup_path"]) as f:
        r_lookup = json.load(f)
    model = BaselineModel(cfg)
    d = _attach_predictions(pd.read_parquet("data/prepared.parquet"),
                            cfg, model, prior, r_lookup)

    store = PosteriorStore.initialise(cfg, prior["per_category"],
                                      prior["episodes_per_week"])
    events = EventStore(cfg)
    rng = np.random.default_rng(5)
    tau = 5000.0

    n = 0
    for eid, g in list(d.groupby("episode_id"))[:80]:
        g = g.sort_values("hour_of_day")
        q = int(g.starting_inventory.iloc[0])
        if q <= 0:
            continue
        anchor = None
        mu_path = list(g.mu_ref_hat.to_numpy())
        for t in range(len(g)):
            if q <= 0:
                break
            row = g.iloc[t]
            evt = decide({
                "episode_id": eid, "sku_id": int(row.sku_id), "fc": row.fc,
                "category": row.category, "subcategory": row.subcategory,
                "hour_of_day": int(row.hour_of_day),
                "hours_remaining": len(g) - t, "q": q,
                "original_price": float(row.original_price),
                "cost": float(row.cost), "r": float(row.r),
                "mu_ref_path": mu_path[t:], "current_discount": anchor,
            }, store, events, cfg, rng, tau, model.version)
            n += 1

            assert all(f in evt for f in DECISION_REQUIRED)
            # safety invariants: cost floor and monotonicity on every path
            assert evt["applied_price"] >= evt["cost"] - 1e-6
            if anchor is not None:
                assert evt["applied_discount"] >= anchor - 1e-9
            anchor = evt["applied_discount"]

            mu = mu_path[t] * ((1 - anchor)
                               / (1 - evt["reference_discount"])) ** -1.3
            sold = min(int(rng.poisson(mu)), q)
            assert events.emit_outcome({
                "outcome_id": f"o-{evt['decision_id']}",
                "decision_id": evt["decision_id"], "units_sold": sold,
                "starting_inventory": q, "ending_inventory": q - sold,
                "applied_price": evt["applied_price"],
                "is_stockout": sold >= q, "execution_status": "ok",
                "finalized_at": pd.Timestamp.now("UTC").isoformat(),
            })
            q -= sold
    assert n > 50

    report = update_run(cfg, apply=True)
    assert all(g["pass"] for g in report["event_quality_gates"].values())
    assert report["applied"]
    for cell, c in report["cells"].items():
        assert abs(c["proposed_mean"] - c["mean_before"]) \
            <= cfg["learning"]["max_mean_step"] + 1e-9
        assert c["proposed_std"] >= cfg["posterior"]["min_std"]

    # exactly-once: a second apply consumes nothing (PRD 13.5)
    report2 = update_run(cfg, apply=True)
    assert not report2["cells"]


def test_fit_calibration_cli(workspace):
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": REPO}
    r = subprocess.run(
        [sys.executable, "-m", "bootstrap.train_baseline",
         "--input", "data/prepared.parquet", "--fit-calibration"],
        cwd=workspace, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open("artifacts/calibration.json") as f:
        calib = json.load(f)
    factors = calib["factor_by_category"]
    assert factors and all(v > 0 for v in factors.values())
    # the trained model must still be loadable with factors applied
    from common.config import load_config
    from bootstrap.train_baseline import BaselineModel
    cfg = load_config("config.yaml")
    cfg["baseline_model"]["apply_level_calibration"] = True
    d = pd.read_parquet("data/prepared.parquet").head(50)
    mu = BaselineModel(cfg).predict_mu_ref(d)
    assert (mu >= cfg["pricing"]["demand_floor"]).all()


def test_duplicate_and_malformed_events_quarantined(workspace):
    _chdir(workspace)
    from common.config import load_config
    from events.store import EventStore

    cfg = load_config("config.yaml")
    events = EventStore(cfg)
    good = {
        "outcome_id": "dup-1", "decision_id": "nonexistent", "units_sold": 1,
        "starting_inventory": 2, "ending_inventory": 1, "applied_price": 900.0,
        "is_stockout": False, "execution_status": "ok",
        "finalized_at": "2026-08-08T00:00:00+00:00",
    }
    assert events.emit_outcome(dict(good))
    assert not events.emit_outcome(dict(good))          # duplicate detected
    assert events.duplicate_counts["outcome"] == 1

    bad = dict(good, outcome_id="bad-1", units_sold=-2)
    before = len(events.load_quarantine())
    assert not events.emit_outcome(bad)                 # quarantined, not dropped
    assert len(events.load_quarantine()) == before + 1


def test_shadow_phase_harness(workspace):
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": REPO}

    # shadow needs the section 9.3 decision and a tau; supply them in config
    with open("config.yaml") as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw["baseline_model"]["apply_level_calibration"] = False
    cfg_raw["exploration"]["tau_initial"] = 500.0
    with open("config.yaml", "w") as f:
        f.write(yaml.safe_dump(cfg_raw))

    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=workspace, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    run("-m", "bootstrap.init_posterior", "--force")
    run("-m", "pipeline.shadow", "--input", "data/prepared.parquet",
        "--out", "reports/shadow.json", "--max-episodes", "60")

    with open("reports/shadow.json") as f:
        report = json.load(f)
    gate = report["shadow_gate"]
    # decisions logged, no prices applied: completeness 1:1, zero cost-floor
    assert gate["cost_floor_violations"]["value"] == 0
    assert gate["event_completeness"]["value"] == 1.0
    assert gate["matched_decision_rate"]["pass"]
    assert report["decision_count"] > 50

    # shadow outcomes are NOT learning evidence: update must consume nothing
    from common.config import load_config
    from pipeline.update import run as update_run
    cfg = load_config("config.yaml")
    report2 = update_run(cfg, apply=False,
                         events_root=cfg["events"]["shadow_store_dir"])
    assert not report2["cells"]


def test_derive_thresholds_cli(workspace):
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": REPO}
    r = subprocess.run(
        [sys.executable, "-m", "bootstrap.derive_thresholds",
         "--input", "data/prepared.parquet", "--mde", "0.075",
         "--out", "reports/thresholds.json"],
        cwd=workspace, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open("reports/thresholds.json") as f:
        report = json.load(f)
    ab = report["ab_duration"]
    assert ab["by_duration"], "no candidate duration produced an SE"
    for row in ab["by_duration"].values():
        assert row["se_pooled"] > 0
        # difference SE must exceed the pooled SE (arm split loses precision)
        assert row["se_arm_difference"] > row["se_pooled"]
    assert "scrap_rate" in report["guardrail_noise"]
    assert "margin_rate" in report["guardrail_noise"]


def test_state_rejected_not_priced(workspace):
    _chdir(workspace)
    from common.config import load_config
    from inference.decide import decide, StateRejected

    cfg = load_config("config.yaml")
    with pytest.raises(StateRejected):
        decide({
            "episode_id": "x", "sku_id": 1, "fc": "F", "category": "MEAT",
            "subcategory": "PORK", "hour_of_day": 12, "hours_remaining": 2,
            "q": 3, "original_price": -5.0, "cost": 10.0, "r": 1.0,
            "mu_ref_path": [1.0, 1.0], "current_discount": None,
        }, None, None, cfg, np.random.default_rng(0), 100.0, "v")
