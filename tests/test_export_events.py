"""daily.export_events: warehouse-safe tables, read through the store."""

import json

import pandas as pd


def test_export_events_writes_warehouse_safe_tables(tmp_path):
    """Derived tables for the warehouse: one row per event, list fields
    JSON-encoded, idempotent, --since filters. The JSONL stays authoritative
    -- the export reads through the store, never bypasses it."""
    from conftest import load_config
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
            "config_version": "1.0.0", "config_digest": "0123456789abcdef",
            "timestamp": f"{day}T17:00:00+00:00"})
    written, _ = export(store, str(tmp_path / "exports"))
    path, n = written["decisions"]
    assert n == 2
    df = pd.read_parquet(path)
    # list field arrives JSON-encoded, so any warehouse loads it
    assert json.loads(df.mu_ref_path.iloc[0]) == [0.8, 0.7]
    # idempotent overwrite, and --since filters by pricing date
    assert export(store, str(tmp_path / "exports"))[0]["decisions"][1] == 2
    assert export(store, str(tmp_path / "exports"),
                  since="2026-08-19")[0]["decisions"][1] == 1


def test_since_cuts_both_tables_on_the_trading_day(tmp_path):
    """An hour-23 outcome finalizes at D+1T00:00Z. Cut on `finalized_at`,
    `--since D+1` shipped that outcome without its decision and `--since D`
    shipped the decision without... the outcome landed in the next load: a
    trading day split across two exports. Both tables now cut on the
    decision's trading day (events.pairs.decision_day)."""
    from conftest import decision_event, outcome_event, load_config
    from events.store import EventStore
    from daily.export_events import export

    store = EventStore(load_config(), root=str(tmp_path / "events"))

    def outcome(**over):                      # reconciling (2 - 1 = 1): admitted
        return outcome_event(ending_inventory=1, **over)

    assert store.emit_decision(decision_event(
        decision_id="D-late", date="2026-08-18", hour_of_day=23,
        timestamp="2026-08-18T23:00:00+00:00"))
    assert store.emit_outcome(outcome(outcome_id="O-late", decision_id="D-late",
                                      finalized_at="2026-08-19T00:00:00+00:00"))
    assert store.emit_decision(decision_event(decision_id="D-next", date="2026-08-19"))
    assert store.emit_outcome(outcome(outcome_id="O-next", decision_id="D-next"))
    # an outcome naming no known decision has no trading day: it keeps the
    # only day it carries, its finalized_at
    assert store.emit_outcome(outcome(outcome_id="O-orphan", decision_id="D-gone",
                                      finalized_at="2026-08-18T12:00:00+00:00"))

    def rows(since):
        out, _ = export(store, str(tmp_path / "exports"), since=since)
        return {name: sorted(pd.read_parquet(path)[f"{name[:-1]}_id"])
                for name, (path, _) in out.items()}

    assert rows("2026-08-19") == {"decisions": ["D-next"], "outcomes": ["O-next"]}
    assert rows("2026-08-18") == {"decisions": ["D-late", "D-next"],
                                  "outcomes": ["O-late", "O-next", "O-orphan"]}
    assert rows(None)["outcomes"] == ["O-late", "O-next", "O-orphan"]


def test_an_orphan_outcome_with_no_finalized_at_is_skipped_and_counted(tmp_path):
    """An outcome naming no known decision falls back to its `finalized_at`
    day; one carrying neither raised KeyError and stopped the whole
    export. It has no day to cut on: skipped under --since, counted, and
    still exported in full without a cut."""
    from conftest import decision_event, outcome_event, load_config
    from events.store import EventStore
    from daily.export_events import export, since_filter

    store = EventStore(load_config(), root=str(tmp_path / "events"))
    assert store.emit_decision(decision_event(decision_id="D1", date="2026-08-19"))
    assert store.emit_outcome(outcome_event(outcome_id="O1", decision_id="D1",
                                            ending_inventory=1))
    # the store requires finalized_at, so the undated orphan is a foreign
    # line; since_filter is what meets it
    decisions, outcomes = store.load_decisions(), store.load_outcomes()
    orphan = {"outcome_id": "O-undated", "decision_id": "D-gone", "units_sold": 0,
              "starting_inventory": 1, "ending_inventory": 1, "applied_price": 1.0}
    kept_d, kept_o, undated = since_filter(decisions, outcomes + [orphan], "2026-08-01")
    assert [o["outcome_id"] for o in kept_o] == ["O1"] and undated == 1
    written, skipped = export(store, str(tmp_path / "exports"), since="2026-08-01")
    assert written["outcomes"][1] == 1 and skipped == 0
    written, skipped = export(store, str(tmp_path / "exports"))
    assert written["outcomes"][1] == 1 and skipped == 0
