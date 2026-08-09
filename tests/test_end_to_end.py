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
    # the fixture's dataset is a fraction of production size; scale the
    # anchor-row guard with it rather than disabling it
    cfg["baseline_model"]["calibration_min_anchor_rows"] = 20
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


def test_prepared_data_is_priceable_and_self_consistent(workspace):
    """Postconditions of the filter chain: anything that survives must be
    something the system can actually price and measure."""
    _chdir(workspace)
    d = pd.read_parquet("data/prepared.parquet")
    cfg = yaml.safe_load(open("config.yaml"))

    assert d.total_discount.between(0, 1).all()      # percent -> fraction once
    assert (d.starting_inventory >= 0).all()
    assert (d.units_sold >= 0).all()
    assert (d.cost >= 0).all()
    assert (d.units_sold <= d.starting_inventory).all()
    assert (d.original_price > 0).all()
    # every surviving episode has at least one feasible discount tier
    assert (d.cost < d.original_price).all() and (d.d_max > 0).all()
    # and no surviving HOUR is priced under cost. The filter must test the
    # offered price: applied_price is 0 on zero-sale rows (~78% of them), so
    # a filter reading it is blind on exactly those, and the survivors reach
    # the planner as an anchor no feasible tier can match.
    assert (d.total_discount <= d.d_max + 1e-9).all()
    assert (d.offered_price >= d.cost - 1e-9).all()
    assert d.category.notna().all() and d.subcategory.notna().all()
    assert (d.hours_remaining >= 0).all()
    assert (d.hours_remaining <= cfg["data"]["max_window_hours"]).all()

    # the exclusion window is removed whole-episode, so no survivor may have
    # ANY hour inside it
    excl = cfg["data"]["exclusion_window"]
    ds = d.date.astype(str)
    assert not (ds.ge(excl["start"]) & ds.le(excl["end"])).any()


def test_every_episode_has_a_monotone_window_counter(workspace):
    """The episode rule's own postcondition: inside an episode, flc_window
    (hours_remaining) steps down exactly one per row, and the window is at
    least as long as the rows we hold. A violation means two runs collided
    into one id -- which is what duplicate (sku, fc, date, hour) rows did."""
    _chdir(workspace)
    d = pd.read_parquet("data/prepared.parquet")
    g = d.sort_values(["date", "hour_of_day"]).groupby("episode_id")
    for eid, s in g.hours_remaining:
        steps = set(np.diff(s.to_numpy()))
        assert steps <= {-1.0}, f"{eid} has window-counter steps {steps}"
    first, n = g.hours_remaining.first(), g.size()
    assert (first >= n - 1).all()
    assert not d.duplicated(
        subset=["sku_id", "fc", "date", "hour_of_day"]).any()


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

    # a SUB-THRESHOLD batch must be banked, not burned. Under the old
    # semantics the outcomes were marked processed even when no revision
    # committed, so the evidence was destroyed and the posterior later stepped
    # on an information count it no longer had the observations to justify.
    cells = report["cells"]
    if not any(c["update_triggered"] for c in cells.values()):
        before = {k: (v["forced_outcomes"], v["effective_information"])
                  for k, v in cells.items()}
        again = update_run(cfg, apply=True)["cells"]
        assert {k: (v["forced_outcomes"], v["effective_information"])
                for k, v in again.items()} == before
        for c in again.values():
            assert c["batch_oldest_outcome_age_days"] is not None

    # exactly-once, on a batch that DOES commit: lower the bar so the same
    # evidence triggers, then confirm a second apply consumes nothing
    cfg["learning"]["information_increment"] = 1e-9
    triggered = update_run(cfg, apply=True)
    assert triggered["applied"]
    assert all(c["update_triggered"] for c in triggered["cells"].values())
    assert not update_run(cfg, apply=True)["cells"]

    # the section 15.4 guardrails must be computed from these events, not
    # merely declared in config: with both thresholds null they report BLOCKED,
    # and once set they evaluate a real deterioration series
    from pipeline.monitor import (guardrail_series, stop_conditions,
                                  business_metrics, learning_metrics,
                                  safety_metrics)
    decisions, outcomes = events.load_decisions(), events.load_outcomes()
    guard = guardrail_series(decisions, outcomes, cfg)
    assert guard["days_observed"] >= 1
    assert set(guard["scrap_deterioration"]) >= {"basis", "by_day", "latest"}

    args = (safety_metrics(events, decisions, outcomes),
            learning_metrics(decisions, store, cfg),
            business_metrics(decisions, outcomes, cfg), guard)
    blocked = stop_conditions(*args, cfg)
    assert "BLOCKED" in blocked["fired"]["scrap_deterioration_pct"]
    assert "BLOCKED" in blocked["fired"]["margin_deterioration_pct"]

    cfg["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = 0.20
    cfg["monitoring"]["stop_conditions"]["margin_deterioration_pct"] = 0.15
    live = stop_conditions(*args, cfg)
    for key in ("scrap_deterioration_pct", "margin_deterioration_pct"):
        assert live["fired"][key] in (True, False)     # evaluated, not skipped
        assert live["guardrails"][key]["threshold"] is not None


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
    factors = calib["factors"]
    assert factors and all(v > 0 for v in factors.values())
    assert calib["grain"] in ("subcategory", "category")
    # every cell must sit between its own raw fit and its parent: shrinkage
    # pulls toward the parent, it never extrapolates past either
    for key, info in calib["detail"].items():
        lo, hi = sorted([info["raw_factor"], info["parent_factor"]])
        assert lo - 1e-6 <= factors[key] <= hi + 1e-6, key
    # a thin cell must sit nearer its parent than a data-rich one does
    weights = [i["shrinkage_weight_on_self"] for i in calib["detail"].values()]
    assert all(0.0 <= w <= 1.0 for w in weights)
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

    # a sampled gate must say so, and must not let a zero violation COUNT
    # read as a proof over the whole window
    w = report["window"]
    assert w["sampled"] and w["episodes"] == 60
    assert w["population_episodes"] > w["episodes"]
    assert "sampling_caveat" in gate
    assert gate["verdict"].startswith("PASS")   # caveat is not a gate row

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


def _window(sku, fc, start, hours, base_hr=None):
    """One selling window as hourly rows, counting hours_remaining down."""
    hr = hours - 1 if base_hr is None else base_hr
    ts = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame({
        "sku_id": sku, "fc": fc,
        "date": ts.normalize(), "hour_of_day": ts.hour,
        "hours_remaining": [hr - i for i in range(hours)],
    })


def test_episode_spans_midnight_as_one_window():
    """FLC windows commonly run past midnight -- 36 hours is common. A
    date-keyed episode would split one economic window into three, resetting
    the monotonicity anchor and charging carried inventory to scrap twice."""
    from bootstrap.prepare_data import assign_episode_ids

    long_window = _window(1, "FC1", "2026-03-01 10:00", 36)
    d = long_window.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    ids = assign_episode_ids(d)
    assert ids.nunique() == 1, "a 36-hour window must be ONE episode"
    assert d.date.nunique() == 2, "and it must genuinely cross midnight"
    assert ids.iloc[0] == "1|FC1|2026-03-01T10"
    assert len(d) == 36 and d.hours_remaining.iloc[-1] == 0

    # a window long enough to cross twice is still one episode
    three = _window(1, "FC1", "2026-03-01 20:00", 36)
    three = three.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert assign_episode_ids(three).nunique() == 1
    assert three.date.nunique() == 3


def test_back_to_back_windows_and_gaps_still_split():
    from bootstrap.prepare_data import assign_episode_ids

    # two windows abutting with no time gap: only the counter reset separates
    # them, so time-contiguity alone would wrongly merge these
    a = _window(1, "FC1", "2026-03-01 10:00", 6)
    b = _window(1, "FC1", "2026-03-01 16:00", 6)
    d = pd.concat([a, b]).sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert assign_episode_ids(d).nunique() == 2

    # a missing hour inside a window splits it, so an episode's row count
    # always equals its clock -- validate_state rejects any mismatch
    g = _window(1, "FC1", "2026-03-01 10:00", 6).drop(index=3)
    assert assign_episode_ids(g).nunique() == 2

    # different sku x fc never merge
    two = pd.concat([_window(1, "FC1", "2026-03-01 10:00", 4),
                     _window(2, "FC1", "2026-03-01 10:00", 4)])
    two = two.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert assign_episode_ids(two).nunique() == 2


def test_split_assigns_straddling_episode_by_start_date():
    """A window that starts in train and ends in calib belongs wholly to
    train -- otherwise the boundary runs through the middle of an episode."""
    from bootstrap.prepare_data import split_frames

    cfg = {"data": {"split": {
        "train_start": "2026-03-01", "train_end": "2026-03-02",
        "calib_start": "2026-03-03", "calib_end": "2026-03-04",
        "test_start": "2026-03-05", "test_end": "2026-03-06"}}}
    d = _window(1, "FC1", "2026-03-02 10:00", 36)     # crosses into 03-03/04
    d["episode_id"] = "1|FC1|2026-03-02T10"
    frames = split_frames(d, cfg)
    assert len(frames["train"]) == len(d)
    assert len(frames["calib"]) == 0 and len(frames["test"]) == 0


def _observed_episode():
    """The worked example from the production extract: a MEAT episode that
    closes with stock left, while ending_inventory reports zero."""
    hours = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    inv = [8, 8, 8, 8, 8, 5, 5, 5, 5, 4]
    sold = [0, 0, 0, 0, 3, 0, 0, 0, 1, 3]
    end = [8, 8, 8, 8, 5, 5, 5, 5, 4, 0]      # zeroed at the close
    return pd.DataFrame({
        "episode_id": ["m"] * 10,
        "date": [pd.Timestamp("2026-03-01").date()] * 10,
        "hour_of_day": hours,
        "hours_remaining": list(range(9, -1, -1)),
        "starting_inventory": inv, "units_sold": sold, "ending_inventory": end,
    })


def test_true_leftover_on_the_production_worked_example():
    from common import episodes

    d = _observed_episode()
    last = episodes.last_rows(d)
    assert int(last.ending_inventory.iloc[0]) == 0     # what the source says

    # 4 units enter the final hour, 3 sell -> 1 is written off, not zero
    left = episodes.leftover_units(last.starting_inventory, last.units_sold)
    assert float(left.iloc[0]) == 1.0
    assert int(d.starting_inventory.iloc[0]) == 8 and int(d.units_sold.sum()) == 7

    # the window ran out, so that 1 unit IS scrap
    kind = episodes.classify(last.hours_remaining, last.starting_inventory,
                             last.units_sold)
    assert kind.iloc[0] == episodes.COMPLETED
    assert float(episodes.scrap_units(last.hours_remaining,
                                      last.starting_inventory,
                                      last.units_sold).iloc[0]) == 1.0


def test_restocked_episodes_are_dropped():
    from bootstrap.prepare_data import restocked_episodes

    clean = _observed_episode()
    assert len(restocked_episodes(clean)) == 0

    # hour 17 opens with 9 units after hour 16 left 5 behind
    restocked = clean.copy()
    restocked.loc[restocked.hour_of_day >= 17, "starting_inventory"] += 4
    assert list(restocked_episodes(restocked)) == ["m"]

    # selling stock down is never a restock, however steep the drop
    steep = clean.copy()
    steep.loc[steep.hour_of_day == 15, "units_sold"] = 8
    steep.loc[steep.hour_of_day > 15, "starting_inventory"] = 0
    assert len(restocked_episodes(steep)) == 0
