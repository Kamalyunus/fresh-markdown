"""Shared fixtures and builders for the test suite."""
import datetime as dt
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from common.config import load_config as _load_config
from engine.posterior import PosteriorStore
from fit.train_baseline import BaselineModel

# By path, not by CWD: the end-to-end tests chdir into a temp workspace, and
# a bare load_config() there would read whichever config ran last.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

P0, COST = 10000.0, 4000.0


def load_config():
    """The config this repo SHIPS, by path."""
    return _load_config(os.path.join(ROOT, "config.yaml"))


CFG = load_config()


@pytest.fixture
def cfg():
    """The config this repo SHIPS, freshly loaded for every test so a test
    that mutates it in place cannot leak into the next one."""
    return load_config()


@pytest.fixture(scope="session")
def synth_flc(tmp_path_factory):
    """Path to a synthetic FLC extract covering the config's splits: the
    repo's data/flc_synth.parquet when one has been generated, else a small
    one generated once per session -- data/ is gitignored, so a fresh clone
    has none and `pytest` must not fail on that."""
    path = os.path.join(ROOT, "data", "flc_synth.parquet")
    if os.path.exists(path):
        return path
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tools import make_dummy_flc as gen
    start, days = gen.span_covering_splits(CFG)
    df, _ = gen.generate(120, days, "randomized", 3, 0.004, 0.02, start=start)
    out = str(tmp_path_factory.mktemp("synth") / "flc_synth.parquet")
    pq.write_table(pa.Table.from_pandas(df, schema=gen.SCHEMA, preserve_index=False), out)
    return out


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
    """A complete, block-free set of the three reports ops.tune and
    ops.status read; keyword overrides replace a whole file."""
    base = {
        "backtest": {
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
        "shadow": {
            "artifact_versions": {"baseline_model_version": "m1"},
            "tau_initial_derivation": {"tau_initial": 1234.5},
            "calibration_regimes": {"frozen_anchor": 1.0002,
                                    "weekly_refit": 0.9762},
            "learning_yield_would_be": {"episodes_per_bounded_update": 741.0,
                                        "calendar_floor_days_per_step": 1},
            "window": {"date_min": "2026-08-10", "date_max": "2026-08-28",
                       "episodes": 111400},
        },
        "thresholds": {
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
        _write(root, name, payload)
    return str(root)


def _cfg_with(cfg, tmp_path, cal=None, rho=None):
    """`cfg` pointed at a calibration and a rho artifact under tmp_path."""
    cal_path = tmp_path / "calibration.json"
    cal_path.write_text(json.dumps(cal if cal is not None else {
        "provenance": {"bundle": "m1"},
        "convergence": {"converged": True, "max_abs_dlog": 0.001,
                        "tol_log": 0.02}}))
    rho_path = tmp_path / "rho.json"
    rho_path.write_text(json.dumps(rho or {"rho": 0.2436}))
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


def _frame():
    """Two episodes: one opens 08-03 22:00 and runs past midnight into 08-04,
    one opens 08-04 09:00. Only the second belongs to a 08-04 hold-out."""
    rows = ([("crosses", "2026-08-03", h) for h in range(22, 24)]
            + [("crosses", "2026-08-04", h) for h in range(0, 4)]
            + [("inside", "2026-08-04", h) for h in range(9, 13)])
    return episode_frame(rows, columns=["episode_id", "date", "hour_of_day"])


# ---------------------------------------------- the harness applier and frames
#
# A BaselineModel applier over a constant base rate (no booster), so shadow
# and the backtest run their real code -- decide(), the DP, the ledger, the
# event store, the level-factor applier -- on a frame small enough to reason
# about.

class _Applier(BaselineModel):
    """BaselineModel's factor applier over a constant mu_ref -- the real
    schedule/freeze/coverage code, no LightGBM."""

    def __init__(self, cfg, base_mu=2.0, anchor=None, schedule=None):
        self.cfg = cfg
        self.calibration = dict(anchor or {"FRUIT": 1.0})
        self.calibration_grain = "category"
        self.calibration_schedule = schedule
        self.calibration_stops_at = None
        self.version = "applier-only"
        self.base_mu = base_mu
        self._reset_calibration_counters()

    def predict_mu_ref(self, d, raw=False):
        mu = np.full(len(d), float(self.base_mu))
        return mu if raw else mu * self._factor_vector(d)


def _hours(eid, day, n, q0=6, sold=1, disc=0.30, tail=0, hour0=9, dp=True,
           sku=7, category="FRUIT"):
    """One episode in the prepared-frame vocabulary: `n` observed hours
    opening `day` at `hour0`, closed by the write-off sentinel on its last
    row; `tail` > 0 leaves window hours uncovered (extend_to_window adds
    them). `dp=False` marks it outside the dp_eligible population."""
    start = [q0 - sold * i for i in range(n)]
    end = [q - sold for q in start]
    end[-1] = 0
    return pd.DataFrame({
        "episode_id": [eid] * n, "date": [dt.date.fromisoformat(day)] * n,
        "hour_of_day": [hour0 + i for i in range(n)],
        "hours_remaining": [n - 1 - i + tail for i in range(n)],
        "sku_id": [sku] * n, "fc": ["FC1"] * n, "category": [category] * n,
        "subcategory": ["BERRY"] * n,
        "starting_inventory": start, "ending_inventory": end,
        "units_sold": [sold] * n, "total_discount": [disc] * n,
        "original_price": [10_000.0] * n,
        "offered_price": [10_000.0 * (1 - disc)] * n, "cost": [4000.0] * n,
        "d_ref": [0.30] * n, "dp_eligible": [dp] * n, "episode_eligible": [dp] * n,
    })


def _harness_cfg(cfg, tmp_path):
    """The shipped config pointed at throwaway artifacts."""
    r_path = tmp_path / "r_lookup.json"
    r_path.write_text(json.dumps({"fallback_order": ["subcategory", "category",
                                                     "global"],
                                  "subcategory": {}, "category": {},
                                  "global": 1.0}))
    cfg = dict(cfg)
    cfg["dispersion"] = dict(cfg["dispersion"], r_lookup_path=str(r_path))
    cfg["posterior"] = dict(cfg["posterior"], path=str(tmp_path / "posterior.json"))
    cfg["events"] = dict(cfg["events"], shadow_store_dir=str(tmp_path / "shadow_events"))
    cfg["exploration"] = dict(cfg["exploration"], tau0_derivation_min_decisions=1)
    PosteriorStore.initialise(cfg, {"FRUIT": {"mean": -1.2, "std": 0.5}},
                              {"FRUIT": 1000}, path=cfg["posterior"]["path"])
    return cfg
