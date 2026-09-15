"""Shared fixtures and builders for the test suite."""
import copy
import datetime as dt
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from common.config import config_get, load_config as _load_config
from common.provenance import stamp
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
        "config_version": "1.0.0", "config_digest": "0123456789abcdef",
        "timestamp": "2026-08-19T17:00:00+00:00",
    }
    evt.update(over)
    return evt


def outcome_event(**over):
    """An outcome event that reconciles on its own; keywords override."""
    evt = {
        "event": "outcome", "outcome_id": "O0", "decision_id": "D0",
        "units_sold": 1, "starting_inventory": 2, "ending_inventory": 1,
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


@pytest.fixture
def reports_dir(tmp_path):
    """tmp_path/"r" holding the default report set from `_reports`."""
    root = tmp_path / "r"
    root.mkdir()
    _reports(root)
    return root


# ------------------------------------------- a config pointed at tmp artifacts
#
# ONE builder redirects every artifact the process writes; the others put a
# file or a knob on top of it. The file NAMES are the shipped ones, which the
# seal and the audit trail read back.

ARTIFACT_PATHS = (
    (("data", "split_manifest_path"), "split_manifest.json"),
    (("baseline_model", "model_path"), "baseline_model.txt"),
    (("baseline_model", "feature_schema_path"), "feature_schema.json"),
    (("baseline_model", "calibration_factor_path"), "calibration.json"),
    (("dispersion", "r_lookup_path"), "r_lookup.json"),
    (("dispersion", "rho_path"), "rho.json"),
    (("posterior", "prior", "path"), "prior.json"),
    (("posterior", "path"), "posterior.json"),
    (("artifacts", "bundle_path"), "bundle.json"),
)


def scratch_paths(cfg, tmp_path):
    """A deep copy of `cfg` with every artifact path under `tmp_path` and
    nothing written there yet."""
    cfg = copy.deepcopy(cfg)
    for key, name in ARTIFACT_PATHS:
        node = cfg
        for k in key[:-1]:
            node = node[k]
        node[key[-1]] = str(tmp_path / name)
    return cfg


def scratch_config(cfg, tmp_path):
    """`scratch_paths` (no r_lookup: raw basis) with a thin anchor floor and
    W=1, so the anchor window is the week before the gate -- the builder
    behind `scratch_cfg`, which the calibration tests fit artifacts on."""
    cfg = scratch_paths(cfg, tmp_path)
    cfg["baseline_model"]["calibration_min_anchor_rows"] = 10
    cfg["baseline_model"]["calibration_fit_trailing_weeks"] = 1
    return cfg


@pytest.fixture
def scratch_cfg(cfg, tmp_path):
    return scratch_config(cfg, tmp_path)


def _cfg_with(cfg, tmp_path, cal=None, rho=None):
    """`scratch_paths` with a calibration and a rho artifact written."""
    cfg = scratch_paths(cfg, tmp_path)
    pathlib.Path(cfg["baseline_model"]["calibration_factor_path"]).write_text(
        json.dumps(cal if cal is not None else {
            "provenance": {"bundle": "m1"},
            "convergence": {"converged": True, "max_abs_dlog": 0.001,
                            "tol_log": 0.02}}))
    pathlib.Path(cfg["dispersion"]["rho_path"]).write_text(
        json.dumps(rho or {"rho": 0.2436}))
    return cfg


def _harness_cfg(cfg, tmp_path):
    """`scratch_paths` with an empty r_lookup, an initialised posterior, a
    throwaway shadow store and the exploration keys the harness frame needs."""
    cfg = scratch_paths(cfg, tmp_path)
    pathlib.Path(cfg["dispersion"]["r_lookup_path"]).write_text(
        json.dumps({"fallback_order": ["subcategory", "category", "global"],
                    "subcategory": {}, "category": {}, "global": 1.0}))
    cfg["events"] = dict(cfg["events"], shadow_store_dir=str(tmp_path / "shadow_events"))
    cfg["exploration"] = dict(cfg["exploration"], tau0_derivation_min_decisions=1,
                              delta_min_log_bias=None,   # no floor: the shipped map is the owner's
                              # the harness frame's seed closes on six days
                              # before the window; the base must span the
                              # window or day one is held (budget_held)
                              budget_il_window_days=6)
    PosteriorStore.initialise(cfg, {"FRUIT": {"mean": -1.2, "std": 0.5}},
                              {"FRUIT": 1000}, path=cfg["posterior"]["path"])
    return cfg


# ------------------------------------------------------- the artifact bundle

BUNDLE = "baseline-20260101000000"


def artifact_at(cfg, key, payload, bundle=BUNDLE, stamped=True):
    """Write `payload` at the artifact path config names under `key`."""
    path = pathlib.Path(config_get(cfg, key))
    if stamped:
        stamp(payload, cfg, bundle, "test")
    _write(path.parent, path.stem, payload)


def full_bundle(cfg, bundle=BUNDLE):
    """One coherent set: model, its schema, and everything fitted against it."""
    with open(config_get(cfg, ("baseline_model", "model_path")), "w") as f:
        f.write("tree { }")                      # a model file, not JSON
    artifact_at(cfg, ("data", "split_manifest_path"), {"split": {}}, bundle=None)
    artifact_at(cfg, ("baseline_model", "feature_schema_path"),
                {"model_version": bundle}, stamped=False)   # names its model the old way
    artifact_at(cfg, ("dispersion", "r_lookup_path"), {"global": 0.9}, bundle)
    artifact_at(cfg, ("dispersion", "rho_path"), {"rho": 0.31}, bundle)
    artifact_at(cfg, ("posterior", "prior", "path"), {"source": "fallback"}, bundle)


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


def _window(sku, fc, start, hours, base_hr=None):
    """One selling window as hourly rows, counting hours_remaining down,
    on an open shelf that reconciles every hour (no close, no restock: the
    boundary rule reads the inventory too)."""
    hr = hours - 1 if base_hr is None else base_hr
    ts = pd.date_range(start, periods=hours, freq="h")
    return episode_frame(sku_id=sku, fc=fc, date=ts.normalize(),
                         hour_of_day=ts.hour,
                         hours_remaining=[hr - i for i in range(hours)],
                         starting_inventory=5, units_sold=0, ending_inventory=5)


def _shelf(hours, counters, start, sold, end, day="2026-03-01", sku="S", fc="F"):
    """Hourly rows of one SKU x FC with the inventory the boundary rule reads."""
    return episode_frame(hour_of_day=hours, hours_remaining=counters,
                         starting_inventory=start, units_sold=sold,
                         ending_inventory=end, date=day, sku_id=sku, fc=fc)


# ------------------------------------------------ rows in the SOURCE schema

def source_row(**over):
    """One row in the extract's own schema (`make_dummy_flc.SCHEMA` names:
    discount in PERCENT, `final_price` a realised price); keywords override."""
    row = {"date": dt.date(2026, 3, 2), "hour": 10, "skuseq": 1, "fc": "F1",
           "inventory": 10.0, "discount": 25.0, "units_sold": 1,
           "normal_asp": 10_000.0, "final_price": 7_500.0,
           "cogs_wo_vat": 4000.0, "ending_inventory": 9.0, "flc_window": 5.0,
           "category": "MEAT", "subcategory": "PORK"}
    row.update(over)
    return row


def source_window(sku, start_hour, n, day="2026-03-02", fc="F1", inv0=10,
                  discount=25.0, price=10_000.0, category="MEAT",
                  subcategory="PORK"):
    """One clean source window as `source_row`s: `n` hours from
    `start_hour`, selling one an hour, the write-off sentinel on its last
    row; discount in PERCENT, as the source emits it."""
    rows, inv = [], inv0
    for i in range(n):
        end = inv - 1 if i < n - 1 else 0
        rows.append(source_row(
            date=dt.date.fromisoformat(day), hour=start_hour + i, skuseq=sku,
            fc=fc, inventory=float(inv), discount=discount, units_sold=1,
            normal_asp=price, final_price=price * (1 - discount / 100),
            cogs_wo_vat=4000.0, ending_inventory=float(end),
            flc_window=float(n - 1 - i), category=category,
            subcategory=subcategory))
        inv = end
    return rows


def write_extract(tmp_path, rows, name="raw.parquet"):
    """`rows` (source_row dicts) as a parquet extract under the source
    schema; returns its path."""
    from tools.make_dummy_flc import SCHEMA
    df = pd.DataFrame(rows)[[f.name for f in SCHEMA]]
    path = tmp_path / name
    pq.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False), str(path))
    return str(path)


# ---------------------------------------------- the harness applier and frames
#
# A BaselineModel applier over a constant base rate (no booster), so shadow
# and the backtest run their real code -- decide(), the DP, the ledger, the
# event store, the level-factor applier -- on a frame small enough to reason
# about. The same stub stands in for the model in every fit test: a level
# solve reads its RAW mu, a residual frame its calibrated one.

class _Applier(BaselineModel):
    """BaselineModel's factor applier over a constant mu_ref -- the real
    schedule/freeze/coverage code, no LightGBM. `calls` counts predictions,
    for a test that pins how often a fit predicts."""

    def __init__(self, cfg, base_mu=2.0, anchor=None, schedule=None):
        self.cfg = cfg
        self.calibration = dict(anchor or {"FRUIT": 1.0})
        self.calibration_grain = "category"
        self.calibration_schedule = schedule
        self.calibration_stops_at = None
        self.version = "applier-only"
        self.base_mu = base_mu
        self.calls = 0
        self._reset_calibration_counters()

    def predict_mu_ref(self, d, raw=False):
        self.calls += 1
        mu = np.full(len(d), float(self.base_mu))
        return mu if raw else mu * self.level_factors(d)


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


def _prepared(cells, days):
    """A prepared-frame lookalike: one 4-hour anchor episode per cell per
    day over `days`, every row eligible. `cells`: {sub: (cat, sold)}."""
    rows = []
    for day in days:
        for sub, (cat, sold) in cells.items():
            for h in range(10, 14):
                rows.append(dict(
                    episode_id=f"{sub}|{day}|{h}", date=day, hour_of_day=h,
                    sku_id=sub, fc="F", category=cat, subcategory=sub,
                    total_discount=0.25, d_ref=0.25, starting_inventory=100,
                    units_sold=sold, ending_inventory=100 - sold,
                    episode_eligible=True, dp_eligible=True))
    return pd.DataFrame(rows)


def _calib_frame(cfg, groups):
    """`groups`: {subcategory: (category, rows)}; every row inside the calib
    window, stocked, eligible, at the anchor."""
    s = cfg["data"]["split"]
    rng = np.random.default_rng(0)
    rows = []
    for sub, (cat, n) in groups.items():
        for i in range(n):
            rows.append(dict(
                episode_id=f"{sub}-{i // 4}", date=s["calib_start"],
                hour_of_day=10 + i % 4, sku_id=sub, fc="F", category=cat,
                subcategory=sub, starting_inventory=10,
                units_sold=int(rng.negative_binomial(1.0, 1.0 / 3.0)),
                total_discount=0.25, d_ref=0.25,
                episode_eligible=True, dp_eligible=True))
    d = pd.DataFrame(rows)
    d["ending_inventory"] = (d.starting_inventory - d.units_sold).clip(lower=0)
    return d
