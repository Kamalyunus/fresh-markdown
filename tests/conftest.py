"""Shared fixtures and builders for the test suite."""
import json
import os
import pathlib

import pandas as pd
import pytest

from common.config import load_config

# By path, not by CWD: the end-to-end tests chdir into a temp workspace, and
# a bare load_config() there would read whichever config ran last.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

P0, COST = 10000.0, 4000.0


@pytest.fixture
def cfg():
    """The config this repo SHIPS, freshly loaded for every test so a test
    that mutates it in place cannot leak into the next one."""
    return load_config(os.path.join(ROOT, "config.yaml"))


# ------------------------------------------------------------------ events

def decision_event(**over):
    """A decision event carrying every contract field; keywords override."""
    evt = {
        "event": "decision", "decision_id": "D0", "episode_id": "EP0",
        "is_entry": True, "sku_id": "S0", "fc": "FC-04",
        "category": "vegetables", "subcategory": "leafy_greens",
        "date": "2026-08-19", "hour_of_day": 17, "hours_remaining": 1,
        "q_remaining": 2, "original_price": P0, "cost": COST, "d_max": 0.6,
        "feasible_tier_count": 25, "action_set_size": 5,
        "optimal_price": P0 * 0.85, "optimal_discount": 0.15,
        "expected_il": 1000.0, "expected_denominator": 5000.0,
        "applied_price": P0 * 0.7, "applied_discount": 0.3,
        "is_exploration": False, "exploration_cost": 0.0,
        "affordable_set_size": 0, "tau_current": 447.78, "delta_min": 0.0,
        "epsilon_posterior_mean": -1.0, "epsilon_posterior_std": 0.6,
        "reference_discount": 0.3, "reference_mu": 0.8, "mu_ref_path": [0.8],
        "anchor_discount": None, "dispersion_r": 0.919,
        "baseline_model_version": "b", "posterior_version": 0,
        "config_version": "1.0.0", "timestamp": "2026-08-19T17:00:00+00:00",
    }
    evt.update(over)
    return evt


def outcome_event(**over):
    """An outcome event that reconciles on its own; keywords override."""
    evt = {
        "event": "outcome", "outcome_id": "O0", "decision_id": "D0",
        "units_sold": 1, "starting_inventory": 2, "ending_inventory": 0,
        "applied_price": P0 * 0.7, "is_stockout": False,
        "execution_status": "ok", "finalized_at": "2026-08-19T18:00:00+00:00",
    }
    evt.update(over)
    return evt


# ----------------------------------------------------------------- reports

def _write(root, name, payload):
    """Write `payload` as <root>/<name>.json."""
    pathlib.Path(root, f"{name}.json").write_text(json.dumps(payload))


def _reports(root, **over):
    """A complete, block-free set of the three reports pipeline.tune and
    pipeline.status read; keyword overrides replace a whole file."""
    base = {
        "backtest.json": {
            "artifact_versions": {"baseline_model_version": "m1"},
            "fidelity": {"calibration_window_sweep": {
                "recommended_fit_window": "trailing_1w",
                # every candidate present, so the band finding resolves
                # whatever W config happens to ship
                "trailing_1w": {"mean_abs_log_error": 0.001,
                                "share_weeks_in_band": 0.99},
                "trailing_2w": {"mean_abs_log_error": 0.02,
                                "share_weeks_in_band": 0.80},
                "trailing_4w": {"mean_abs_log_error": 0.02,
                                "share_weeks_in_band": 0.80},
                "trailing_8w": {"mean_abs_log_error": 0.02,
                                "share_weeks_in_band": 0.80}}},
            "policy_deltas": {"step_sensitivity": {
                "deeper_belief": {"share_prices_changed": 0.02,
                                  "il_delta_pct": -0.0004}}},
        },
        "shadow.json": {
            "artifact_versions": {"baseline_model_version": "m1"},
            "tau_initial_derivation": {"tau_initial": 1234.5},
            "calibration_regimes": {"frozen_anchor": 1.0002,
                                    "weekly_refit": 0.9762},
            "learning_yield_would_be": {"episodes_per_bounded_update": 741.0,
                                        "calendar_floor_days_per_step": 1},
            "window": {"date_min": "2026-08-10", "date_max": "2026-08-28",
                       "episodes": 111400},
        },
        "thresholds.json": {
            "information_increment_recommendation": {
                "recommended": 0.341, "verdict": "measured"},
            "bounded_step_recommendation": {
                "consistent_max_mean_step": 0.485,
                "verdict": "MEAN RAIL BINDS FIRST"},
            "guardrail_threshold_recommendation": {
                "scrap_rate": {
                    "config_key": "monitoring.stop_conditions.scrap_deterioration_pct",
                    "binding_floor": 0.2656, "binding_label": "3-sigma",
                    "binding_basis": "trailing", "verdict": "null"}},
        },
    }
    base.update(over)
    for name, payload in base.items():
        (root / name).write_text(json.dumps(payload))
    return str(root)


def _cfg_with(cfg, tmp_path, cal=None, rho=None):
    """`cfg` pointed at a calibration and a rho artifact under tmp_path."""
    cal_path = tmp_path / "calibration.json"
    cal_path.write_text(json.dumps(cal if cal is not None else {
        "provenance": {"bundle": "m1"},
        "convergence": {"converged": True, "max_abs_dlog": 0.001,
                        "tol_log": 0.02}}))
    rho_path = tmp_path / "rho.json"
    rho_path.write_text(json.dumps(rho or {"rho": 0.2436,
                                           "mean_forced_hours_per_episode": 5.909}))
    return dict(
        cfg,
        baseline_model=dict(cfg["baseline_model"],
                            calibration_factor_path=str(cal_path)),
        dispersion=dict(cfg["dispersion"], rho_path=str(rho_path)))


@pytest.fixture
def reports_dir(tmp_path):
    """tmp_path/"r" holding the default report set from `_reports`."""
    root = tmp_path / "r"
    root.mkdir()
    _reports(root)
    return root


# ---------------------------------------------------------------- episodes

def _per_row(v):
    return hasattr(v, "__len__") and not isinstance(v, (str, bytes))


def episode_frame(rows=None, columns=None, **cols):
    """Hourly episode rows as a DataFrame. `rows` is a list of tuples paired
    with `columns`, a list of dicts, or None. Every keyword in `cols` is then
    added as a column: a sequence is taken per row, a scalar is broadcast."""
    if rows is None:
        n = len(next(v for v in cols.values() if _per_row(v)))
        d = pd.DataFrame(index=range(n))
    else:
        d = pd.DataFrame(rows, columns=columns)
    for k, v in cols.items():
        d[k] = v
    return d
