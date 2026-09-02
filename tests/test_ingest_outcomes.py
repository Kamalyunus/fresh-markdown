"""Feed-based outcome construction: the minimal integration surface.

Engineering applies prices and (optionally) reports failed pushes; every
outcome field the old producer contract required is derived here from the
hourly feed and must agree with the offline chain's own definitions.
"""

import json

import pandas as pd
import pytest

from pipeline.ingest_outcomes import build_outcomes, load_failures


def _dec(i, sku="7", fc="F1", date="2026-08-19", hour=17):
    return {"decision_id": f"D{i}", "sku_id": sku, "fc": fc,
            "date": date, "hour_of_day": hour}


def _feed(rows):
    return pd.DataFrame([{
        "skuseq": r.get("sku", "7"), "fc": r.get("fc", "F1"),
        "date": r.get("date", "2026-08-19"), "hour": r.get("hour", 17),
        "inventory": r["start"], "units_sold": r["sold"],
        "ending_inventory": r["end"],
        "discount": r.get("disc", 25.0),            # PERCENT, like the source
        "normal_asp": r.get("price", 10000.0),
        "final_price": 0.0,                          # zeroed on no-sale rows
        "cogs_wo_vat": 4000.0, "flc_window": 3,
        "category": "VEG", "subcategory": "LEAFY",
    } for r in rows])


def test_outcomes_are_built_from_the_feed_not_from_a_producer():
    outs, rep = build_outcomes(
        [_dec(1)], _feed([{"start": 3, "sold": 1, "end": 2}]))
    assert rep["outcomes_built"] == 1 and not rep["decisions_without_feed_row"]
    o = outs[0]
    assert o["decision_id"] == "D1" and o["outcome_id"] == "feed-D1"
    assert (o["units_sold"], o["starting_inventory"],
            o["ending_inventory"]) == (1, 3, 2)
    # OFFERED price from the discount column, never the zeroed final_price
    assert o["applied_price"] == pytest.approx(10000.0 * 0.75)
    assert o["is_stockout"] is False
    assert o["execution_status"] == "ok"
    assert "adjustment_reason" not in o          # reconciling hour: omitted
    assert o["finalized_at"].startswith("2026-08-19T18:00")   # hour close


def test_adjustment_reasons_are_derived_not_asked_for():
    outs, rep = build_outcomes(
        [_dec(1), _dec(2, hour=18), _dec(3, hour=19), _dec(4, hour=20)],
        _feed([
            {"start": 3, "sold": 1, "end": 5},              # restock
            {"hour": 18, "start": 4, "sold": 1, "end": 0},  # write-off
            {"hour": 19, "start": 5, "sold": 1, "end": 2},  # shrink
            {"hour": 20, "start": 2, "sold": 2, "end": 0},  # clean sell-out
        ]))
    by = {o["decision_id"]: o for o in outs}
    assert by["D1"]["adjustment_reason"] == "intraday_restock"
    assert by["D2"]["adjustment_reason"] == "episode_close_write_off"
    assert by["D3"]["adjustment_reason"] == "unexplained_shortfall"
    assert "adjustment_reason" not in by["D4"]
    assert by["D4"]["is_stockout"] is True
    # a restocked hour that sold past its opening count is NOT a stockout:
    # the shelf never sat empty and demand was observed exactly (the naive
    # sold >= starting read it as censored -- the trap update.py names)
    assert by["D1"]["is_stockout"] is False
    assert rep["adjustment_reasons"] == {"intraday_restock": 1,
                                         "episode_close_write_off": 1,
                                         "unexplained_shortfall": 1}


def test_a_decision_with_no_feed_row_is_counted_never_invented():
    outs, rep = build_outcomes(
        [_dec(1), _dec(2, hour=23)],
        _feed([{"start": 3, "sold": 1, "end": 2}]))
    assert len(outs) == 1
    assert rep["decisions_without_feed_row"] == 1
    assert rep["unmatched_decision_ids"] == ["D2"]


def test_duplicate_feed_hours_match_nothing():
    """Two states for one hour is unresolvable -- same rule as prepare_data's
    duplicate_hour_rows_dropped, so the decision lands in the unmatched
    count instead of picking a copy at random."""
    outs, rep = build_outcomes(
        [_dec(1)],
        _feed([{"start": 3, "sold": 1, "end": 2},
               {"start": 9, "sold": 0, "end": 9}]))
    assert not outs
    assert rep["feed_duplicate_hours"] == 1
    assert rep["decisions_without_feed_row"] == 1


def test_push_failures_mark_the_outcome_ineligible(tmp_path):
    p = tmp_path / "failures.jsonl"
    p.write_text(json.dumps({"sku_id": "7", "fc": "F1", "date": "2026-08-19",
                             "hour_of_day": 17,
                             "reason": "price_push_timeout"}) + "\n")
    outs, rep = build_outcomes(
        [_dec(1), _dec(2, hour=18)],
        _feed([{"start": 3, "sold": 1, "end": 2},
               {"hour": 18, "start": 2, "sold": 0, "end": 2}]),
        failures=load_failures(str(p)))
    by = {o["decision_id"]: o for o in outs}
    assert by["D1"]["execution_status"] == "failed"
    assert by["D1"]["execution_failure_reason"] == "price_push_timeout"
    assert by["D2"]["execution_status"] == "ok"
    assert rep["push_failures_applied"] == 1


def test_built_outcomes_pass_the_store_and_rerun_is_a_dedup(tmp_path):
    """The derived events must clear the store's own validation (including
    the reconciliation rule), and re-ingesting the same feed is a no-op."""
    from common.config import load_config
    from events.store import EventStore

    cfg = load_config()
    store = EventStore(cfg, root=str(tmp_path / "events"))
    outs, _ = build_outcomes(
        [_dec(1), _dec(2, hour=18)],
        _feed([{"start": 3, "sold": 1, "end": 5},               # restock
               {"hour": 18, "start": 4, "sold": 1, "end": 0}]))  # write-off
    assert sum(store.emit_outcome(dict(o)) for o in outs) == 2
    assert store.quarantined_this_run == 0
    assert sum(store.emit_outcome(dict(o)) for o in outs) == 0   # dedup
    assert store.duplicate_counts["outcome"] == 2


def test_monitor_scrap_series_counts_shrink_like_the_floors_do():
    """The noise floors are measured on episodes.scrap_units (leftover +
    shrink); a leftover-only live trigger runs looser than its floor by the
    shrink rate, and a live shrink surge could never fire it."""
    from common.config import load_config
    from pipeline.monitor import guardrail_series

    cfg = load_config()
    hours = [  # one episode: a mid-window shrink of 2, closes holding 1
        # (hr, start, sold, ending)
        (3, 6, 1, 3),      # 6 - 1 = 5 expected, 3 reported -> shrink 2
        (2, 3, 1, 2),
        (1, 2, 1, 0),      # write-off row: leftover 1, NOT shrink
    ]
    decisions, outcomes = [], []
    for hr, start, sold, end in hours:
        decisions.append({
            "decision_id": f"g{hr}", "episode_id": "EP-G", "sku_id": "s",
            "fc": "f", "timestamp": "2026-08-19T10:00:00+00:00",
            "hours_remaining": hr, "cost": 100.0})
        outcomes.append({
            "decision_id": f"g{hr}", "starting_inventory": start,
            "units_sold": sold, "ending_inventory": end,
            "applied_price": 500.0})
    g = guardrail_series(decisions, outcomes, cfg)
    # scrap = leftover 1 + shrink 2 = 3, over opening 6
    assert g["daily_scrap_rate"]["2026-08-19"] == pytest.approx(3 / 6)


def test_failures_can_arrive_as_a_table(tmp_path):
    """Engineering keeps outputs in tables; parquet and CSV load like JSONL."""
    rows = pd.DataFrame([{"sku_id": "7", "fc": "F1", "date": "2026-08-19",
                          "hour_of_day": 17, "reason": "push_timeout"}])
    for name, writer in (("f.parquet", rows.to_parquet), ("f.csv", rows.to_csv)):
        p = tmp_path / name
        writer(p, index=False)
        assert load_failures(str(p)) == {
            ("7", "F1", "2026-08-19", 17): "push_timeout"}


def test_export_events_writes_warehouse_safe_tables(tmp_path):
    """Derived tables for the warehouse: one row per event, list fields
    JSON-encoded, idempotent, --since filters. The JSONL stays authoritative
    -- the export reads through the store, never bypasses it."""
    from common.config import load_config
    from events.store import EventStore
    from pipeline.export_events import export

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
            "tau_current": 1.0, "epsilon_posterior_mean": -1.0,
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


def test_business_metrics_counts_shrink_like_the_guardrail_and_il_pct_do():
    """Scrap = leftover + shrink has ONE definition. business_metrics read
    leftover only, so IL, waste_units, sell-through and the by-arm IL% -- the
    A/B's primary metric and production's exploration-budget base -- were all
    a different IL from the one the noise floors and common.metrics.il_pct
    are measured on."""
    from common.config import load_config
    from pipeline.monitor import business_metrics, guardrail_series

    cfg = load_config()
    hours = [(3, 6, 1, 3),      # 6 - 1 = 5 expected, 3 reported -> shrink 2
             (2, 3, 1, 2),
             (1, 2, 1, 0)]      # write-off row: leftover 1, NOT shrink
    decisions, outcomes = [], []
    for hr, start, sold, end in hours:
        decisions.append({
            "decision_id": f"b{hr}", "episode_id": "EP-B", "sku_id": "s",
            "fc": "f", "category": "VEG", "subcategory": "LEAFY",
            "timestamp": "2026-08-19T10:00:00+00:00", "hours_remaining": hr,
            "original_price": 1000.0, "cost": 100.0})
        outcomes.append({
            "decision_id": f"b{hr}", "starting_inventory": start,
            "units_sold": sold, "ending_inventory": end,
            "applied_price": 800.0})

    b = business_metrics(decisions, outcomes, cfg)
    g = guardrail_series(decisions, outcomes, cfg)
    # leftover 1 + shrink 2 = 3 units of scrap, on both sides
    assert b["waste_units"] == 3
    assert g["daily_scrap_rate"]["2026-08-19"] == pytest.approx(3 / 6)
    # IL charges cost x scrap, not cost x leftover
    assert b["il_pct_aggregate"]["il_absolute"] == pytest.approx(
        3 * 200.0 + 3 * 100.0)
    assert b["sell_through"] == pytest.approx(3 / 6)


def test_one_unusable_feed_row_costs_its_decision_not_the_days_batch():
    """int(nan) raised and aborted the whole daily ingest before the store --
    whose design is quarantine-with-a-reason -- ever saw a row, so the day
    read as a 100% completeness gap instead of one named row."""
    import numpy as np

    feed = _feed([{"start": 3, "sold": 1, "end": 2},
                  {"hour": 18, "start": 2, "sold": 0, "end": 2},
                  {"hour": 19, "start": 4, "sold": 1, "end": 3}])
    feed.loc[1, "ending_inventory"] = np.nan
    outs, rep = build_outcomes(
        [_dec(1), _dec(2, hour=18), _dec(3, hour=19)], feed)

    assert [o["decision_id"] for o in outs] == ["D1", "D3"]
    assert rep["unusable_feed_rows"] == 1
    assert rep["unusable_examples"][0]["decision_id"] == "D2"


def test_a_zero_base_price_is_refused_not_priced_at_full_discount():
    """Offline, prepare_data ffills then drops original_price == 0. Live it
    produced applied_price 0.0, which the store accepts and monitor charges
    as (original_price - 0) x sold -- the whole list price booked as IL."""
    feed = _feed([{"start": 3, "sold": 1, "end": 2, "price": 0.0},
                  {"hour": 18, "start": 2, "sold": 1, "end": 1}])
    outs, rep = build_outcomes([_dec(1), _dec(2, hour=18)], feed)

    assert [o["decision_id"] for o in outs] == ["D2"]
    assert rep["unusable_feed_rows"] == 1
    assert "original_price" in rep["unusable_examples"][0]["reason"]


def test_the_guardrail_is_not_inert_before_the_ab(tmp_path):
    """arm() hash-labels every priced SKU x FC, so before the A/B both labels
    exist and BOTH are system-priced: an arm comparison is
    treatment-vs-treatment and a catalogue-wide scrap doubling cancels to
    exactly zero. The guardrail was structurally unable to fire for the whole
    pilot."""
    import copy

    from common.config import load_config
    from pipeline.monitor import guardrail_series

    cfg = load_config()
    # the trailing basis needs guardrail_noise_window_days of history before
    # it produces a comparison at all, plus the smoothing shift
    span = cfg["monitoring"]["guardrail_noise_window_days"] + 20
    decisions, outcomes, i = [], [], 0
    for day in range(span):
        # scrap doubles for the last stretch, across the WHOLE catalogue
        sold, end = (2, 0) if day < span - 10 else (1, 0)
        for unit in range(40):
            i += 1
            decisions.append({
                "decision_id": f"g{i}", "episode_id": f"E{i}",
                "sku_id": f"S{unit}", "fc": "F1", "category": "VEG",
                "subcategory": "LEAFY", "hours_remaining": 1, "cost": 100.0,
                "original_price": 1000.0,
                "timestamp": (pd.Timestamp("2026-06-01")
                              + pd.Timedelta(days=day)).isoformat() + "+00:00",
                "date": str((pd.Timestamp("2026-06-01")
                             + pd.Timedelta(days=day)).date())})
            outcomes.append({
                "decision_id": f"g{i}", "starting_inventory": 4,
                "units_sold": sold, "ending_inventory": end,
                "applied_price": 800.0})

    pre = guardrail_series(decisions, outcomes, cfg)
    assert pre["scrap_deterioration"]["basis"].startswith("trailing_")
    assert pre["scrap_deterioration"]["latest"] > 0, pre["scrap_deterioration"]

    live = copy.deepcopy(cfg)
    live["ab_test"] = dict(live["ab_test"], active=True)
    during = guardrail_series(decisions, outcomes, live)
    assert during["scrap_deterioration"]["basis"] == "control_arm"
    # and with both arms system-priced the deviation is exactly zero --
    # which is precisely why this basis must not be the pre-A/B default
    assert during["scrap_deterioration"]["latest"] == 0.0


def test_a_discount_outside_percent_range_is_refused():
    """The feed discount is PERCENT. A fraction (0.30) would be 0.3% and
    price the hour at full list; 100+ prices it at or below zero."""
    feed = _feed([{"start": 3, "sold": 1, "end": 2, "disc": 100.0},
                  {"hour": 18, "start": 2, "sold": 1, "end": 1, "disc": -5.0},
                  {"hour": 19, "start": 1, "sold": 0, "end": 1, "disc": 99.0}])
    outs, rep = build_outcomes(
        [_dec(1), _dec(2, hour=18), _dec(3, hour=19)], feed)
    assert [o["decision_id"] for o in outs] == ["D3"]
    assert rep["unusable_feed_rows"] == 2
    assert all("total_discount" in x["reason"] for x in rep["unusable_examples"])


def test_the_guardrail_series_is_keyed_on_the_trading_day():
    """An hour-23 decision on day D carries a UTC timestamp that may read as
    D+1. Bucketing on the wall clock split the daily scrap series at
    midnight UTC while IL and the budget were keyed on the trading day."""
    from common.config import load_config
    from pipeline.monitor import guardrail_series

    cfg = load_config()
    decisions = [{
        "decision_id": "late", "episode_id": "EP-L", "sku_id": "s", "fc": "f",
        "date": "2026-08-19", "hour_of_day": 23,
        "timestamp": "2026-08-20T02:00:00+00:00",      # UTC is already D+1
        "hours_remaining": 1, "cost": 100.0}]
    outcomes = [{"decision_id": "late", "starting_inventory": 4,
                 "units_sold": 1, "ending_inventory": 0,
                 "applied_price": 500.0}]
    g = guardrail_series(decisions, outcomes, cfg)
    assert list(g["daily_scrap_rate"]) == ["2026-08-19"]


def test_price_mismatch_is_a_rate_over_compared_pairs():
    """Dividing by every outcome let unmatched outcomes dilute the rate
    below the stop threshold."""
    from pipeline.monitor import safety_metrics

    class _Store:
        duplicate_counts = {"decision": 0, "outcome": 0}
        def load_quarantine(self):
            return []

    decisions = [{"decision_id": "D1", "applied_price": 100.0,
                  "expected_denominator": 1.0, "original_price": 100.0}]
    outcomes = [{"decision_id": "D1", "applied_price": 90.0, "units_sold": 1,
                 "is_stockout": False}]
    outcomes += [{"decision_id": f"orphan{i}", "applied_price": 1.0,
                  "units_sold": 0, "is_stockout": False} for i in range(9)]
    s = safety_metrics(_Store(), decisions, outcomes)
    assert s["applied_vs_recommended_price_mismatch"] == 1.0
    assert s["unmatched_outcome_count"] == 9
