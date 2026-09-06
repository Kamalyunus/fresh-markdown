"""Feed-based outcome construction: the minimal integration surface.

Engineering applies prices and (optionally) reports failed pushes; every
outcome field the old producer contract required is derived here from the
hourly feed and must agree with the offline chain's own definitions.
"""

import json

import pandas as pd
import pytest

from daily.ingest_outcomes import build_outcomes, load_failures


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


def test_the_completeness_gap_is_measured_inside_the_feeds_date_range_only():
    """From day two the store holds yesterday's (already ingested) decisions
    and, late in the day, tomorrow's not-yet-due ones. Neither is missing
    from TODAY's feed; counting them read as a daily completeness gap."""
    decisions = [_dec(1, date="2026-08-18"),            # ingested yesterday
                 _dec(2, date="2026-08-19"),            # today, in the feed
                 _dec(3, date="2026-08-19", hour=23),   # today, feed row absent
                 _dec(4, date="2026-08-20")]            # not yet due
    outs, rep = build_outcomes(
        decisions, _feed([{"date": "2026-08-19", "start": 3, "sold": 1, "end": 2}]))
    assert [o["decision_id"] for o in outs] == ["D2"]
    assert rep["feed_date_range"] == ["2026-08-19", "2026-08-19"]
    assert rep["decisions_without_feed_row"] == 1
    assert rep["unmatched_decision_ids"] == ["D3"]
    assert rep["decisions_outside_feed_range"] == 2

    # a feed whose date column is datetime, not text, keys the same days
    feed = _feed([{"date": "2026-08-19", "start": 3, "sold": 1, "end": 2}])
    feed["date"] = pd.to_datetime(feed["date"])
    outs, rep = build_outcomes(decisions, feed)
    assert [o["decision_id"] for o in outs] == ["D2"]
    assert rep["decisions_without_feed_row"] == 1


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


def test_failures_can_arrive_as_a_table(tmp_path):
    """Engineering keeps outputs in tables; parquet and CSV load like JSONL."""
    rows = pd.DataFrame([{"sku_id": "7", "fc": "F1", "date": "2026-08-19",
                          "hour_of_day": 17, "reason": "push_timeout"}])
    for name, writer in (("f.parquet", rows.to_parquet), ("f.csv", rows.to_csv)):
        p = tmp_path / name
        writer(p, index=False)
        assert load_failures(str(p)) == {
            ("7", "F1", "2026-08-19", 17): "push_timeout"}
    # a datetime `date` column (what a warehouse export carries) keys the
    # same day -- str() of it is '2026-08-19 00:00:00' and matched nothing,
    # so every failed push was learned from as an applied price
    p = tmp_path / "dt.parquet"
    rows.assign(date=pd.to_datetime(rows["date"])).to_parquet(p, index=False)
    assert load_failures(str(p)) == {("7", "F1", "2026-08-19", 17): "push_timeout"}
    outs, rep = build_outcomes([_dec(1)], _feed([{"start": 3, "sold": 1, "end": 2}]),
                               failures=load_failures(str(p)))
    assert outs[0]["execution_status"] == "failed" and rep["push_failures_applied"] == 1


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
