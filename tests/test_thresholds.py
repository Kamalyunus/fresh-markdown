"""derive_thresholds: one verdict per metric, and its tunables in config."""

import copy
import inspect
import json

import pandas as pd

from bootstrap import derive_thresholds as dt
from conftest import episode_frame


def _trailing(three_sigma=0.2):
    return {"scrap_rate": {"three_sigma": three_sigma,
                           "three_sigma_robust": three_sigma,
                           "outlier_dominated": False},
            "margin_rate": {"three_sigma": three_sigma,
                            "three_sigma_robust": three_sigma,
                            "outlier_dominated": False}}


def test_the_inert_multiple_is_read_from_config(cfg):
    cfg = copy.deepcopy(cfg)
    floor = 0.2
    cfg["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = floor * 5
    rec = dt.recommend_thresholds(_trailing(floor), cfg)["scrap_rate"]
    assert "LIKELY INERT" in rec["verdict"]                 # shipped multiple 3
    assert "guardrail_inert_floor_multiple" in rec["verdict"]

    cfg["tuning"]["guardrail_inert_floor_multiple"] = 10
    rec = dt.recommend_thresholds(_trailing(floor), cfg)["scrap_rate"]
    assert rec["verdict"].startswith("OK")
    # the keys tune and status read are still there, and constant by design
    assert rec["binding_floor"] == rec["trailing_floor"] == floor
    assert rec["binding_basis"] == "trailing" and rec["binding_label"] == "3-sigma"
    assert "3 * floor" not in inspect.getsource(dt.recommend_thresholds)


def test_the_noise_block_measures_and_points_at_the_one_verdict(cfg):
    """guardrail_noise carried a second `verdict()` that re-graded the
    threshold with its own wording; recommend_thresholds is the grader."""
    src = inspect.getsource(dt.guardrail_noise)
    assert "def verdict" not in src and "TOO TIGHT" not in src
    assert "from common import episodes" not in inspect.getsource(dt)

    cfg = copy.deepcopy(cfg)
    # three closed one-hour episodes on three days: far too short a series
    d = episode_frame(
        [("e1", "2026-03-01"), ("e2", "2026-03-02"), ("e3", "2026-03-03")],
        columns=["episode_id", "date"], hour_of_day=10, starting_inventory=5,
        units_sold=3, ending_inventory=0, original_price=1000.0,
        offered_price=800.0, cost=500.0, hours_remaining=0)
    out = dt.guardrail_noise(d, cfg)
    for metric in ("scrap_rate", "margin_rate"):
        block = out[metric]
        assert "verdict" not in block
        assert block["verdict_in"] == f"guardrail_threshold_recommendation.{metric}"
        assert block["config_key"].startswith("monitoring.stop_conditions.")
        need = (cfg["monitoring"]["guardrail_noise_window_days"]
                + cfg["monitoring"]["guardrail_noise_min_extra_days"])
        assert block["note"] == f"needs at least {need} days"
    # the extra-days margin is config, not `window + 7`
    cfg["monitoring"]["guardrail_noise_min_extra_days"] = 100
    assert "at least 128 days" in dt.guardrail_noise(d, cfg)["scrap_rate"]["note"]
    assert "window + 7" not in inspect.getsource(dt.guardrail_noise)


def test_outlier_dominance_uses_the_configured_sigma_ratio():
    rel = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 2.0])
    loose = dt._sigma_summary(rel, outlier_ratio=2.0)
    tight = dt._sigma_summary(rel, outlier_ratio=1e6)
    assert loose["outlier_dominated"] and not tight["outlier_dominated"]
    assert loose["three_sigma"] == tight["three_sigma"]
    assert "2 * sigma_robust" not in inspect.getsource(dt._sigma_summary)


def test_the_consistent_band_is_read_from_config(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps({"per_category": {
        "A": {"mean": -1.0, "std": 0.6}}}))
    cfg["posterior"]["prior"]["path"] = str(prior_path)
    # shrink 0.25 -> pull 0.4375; mean step 0.15 clips at 0.15/(0.4375*0.6)
    cfg["learning"]["max_std_shrink"] = 0.25
    cfg["learning"]["max_mean_step"] = 0.15
    clips_at = 0.15 / (0.4375 * 0.6)
    assert clips_at < 0.7
    assert dt.bounded_step(cfg)["verdict"].startswith("MEAN RAIL BINDS FIRST")

    cfg["tuning"]["bounded_step_consistent_band"] = [0.0, 10.0]
    out = dt.bounded_step(cfg)
    assert out["verdict"].startswith("CONSISTENT")
    assert out["consistent_band_std"] == [0.0, 10.0]
    src = inspect.getsource(dt.bounded_step)
    assert "0.7 <=" not in src and "<= 1.4" not in src
