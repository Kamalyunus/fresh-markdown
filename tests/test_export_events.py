"""daily.export_events: warehouse-safe tables, read through the store."""

import json

import pandas as pd


def test_export_events_writes_warehouse_safe_tables(tmp_path):
    """Derived tables for the warehouse: one row per event, list fields
    JSON-encoded, idempotent, --since filters. The JSONL stays authoritative
    -- the export reads through the store, never bypasses it."""
    from common.config import load_config
    from events.store import EventStore
    from daily.export_events import export

    cfg = load_config()
    store = EventStore(cfg, root=str(tmp_path / "events"))
    for i, day in ((1, "2026-08-18"), (2, "2026-08-19")):
        store.emit_decision({f: 1 for f in ()} | {
            "decision_id": f"D{i}", "episode_id": "EP", "is_entry": True,
            "sku_id": "7", "fc": "F1", "category": "VEG",
            "subcategory": "LEAFY", "date": day, "hour_of_day": 17,
            "hours_remaining": 2, "q_remaining": 3, "original_price": 1e4,
            "cost": 4e3, "d_max": 0.6, "feasible_tier_count": 25,
            "action_set_size": 5, "optimal_price": 8500.0,
            "optimal_discount": 0.15, "expected_il": 1.0,
            "expected_denominator": 2.0, "applied_price": 8500.0,
            "applied_discount": 0.15, "is_exploration": False,
            "exploration_cost": 0.0, "affordable_set_size": 0,
            "tau_current": 1.0, "delta_min": 0.0, "epsilon_posterior_mean": -1.0,
            "epsilon_posterior_std": 0.6, "reference_discount": 0.3,
            "reference_mu": 0.8, "mu_ref_path": [0.8, 0.7],
            "anchor_discount": None, "dispersion_r": 0.9,
            "baseline_model_version": "b", "posterior_version": 0,
            "config_version": "1.0.0", "timestamp": f"{day}T17:00:00+00:00"})
    written = export(store, str(tmp_path / "exports"))
    path, n = written["decisions"]
    assert n == 2
    df = pd.read_parquet(path)
    # list field arrives JSON-encoded, so any warehouse loads it
    assert json.loads(df.mu_ref_path.iloc[0]) == [0.8, 0.7]
    # idempotent overwrite, and --since filters by pricing date
    assert export(store, str(tmp_path / "exports"))["decisions"][1] == 2
    assert export(store, str(tmp_path / "exports"),
                  since="2026-08-19")["decisions"][1] == 1
