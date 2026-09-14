"""ops.price_batch -- the batch surface Lane B calls: row-scoped refusals,
the one hour key, the outcome id engineering can name from the feed."""

import json

import pandas as pd
import pytest

from engine.state import REQUEST_FIELDS, validate_request
from events.pairs import hour_key, outcome_id_of
from ops.price_batch import RESPONSE_FIELDS, plan, read_requests, write_rows


def _req(**over):
    r = {"episode_id": "E1", "sku_id": 7, "fc": "F1", "category": "VEG",
         "subcategory": "LEAFY", "date": "2026-08-19", "hour_of_day": 17,
         "hours_remaining": 4, "q": 3, "original_price": 10000.0, "cost": 4000.0,
         "current_discount": None}
    r.update(over)
    return r


def test_a_request_needs_every_contract_field_and_a_layable_grid():
    assert validate_request(_req()) == []
    assert validate_request({k: v for k, v in _req().items() if k != "q"}) == ["missing q"]
    assert any("hour_of_day" in p for p in validate_request(_req(hour_of_day=24)))
    assert any("hours_remaining" in p for p in validate_request(_req(hours_remaining=0)))
    assert any("q" in p for p in validate_request(_req(q=-1)))
    assert any("date" in p for p in validate_request(_req(date="not a day")))
    assert any("sku_id" in p for p in validate_request(_req(sku_id=None)))
    # prices, cost and the anchor are the engine's to judge (nothing twice)
    assert validate_request(_req(original_price=0.0, cost=-1.0)) == []
    assert tuple(_req()) == REQUEST_FIELDS


def test_plan_refuses_row_by_row_never_the_batch():
    reqs = [_req(),                                          # prices
            _req(episode_id="E2", hour_of_day=99),          # cannot become a state
            _req(episode_id="E3", sku_id=8),                 # prices
            _req(episode_id="E4", sku_id=8),                 # E3's hour again
            _req(episode_id="E5", sku_id=9)]                 # already in the store
    priced = {hour_key(9, "F1", "2026-08-19", 17)}
    to_price, rejected = plan(reqs, priced)
    assert [i for i, _, _ in to_price] == [0]
    assert "hour_of_day" in rejected[1]
    assert rejected[2].startswith("duplicate_request") and rejected[3].startswith("duplicate_request")
    assert rejected[4].startswith("already_priced")
    # the key is spelt one way whatever the caller's dtype (7.0 is "7")
    assert plan([_req(sku_id=7.0)], {hour_key("7", "F1", "2026-08-19", 17.0)})[1] == {
        0: "already_priced: the store holds a decision for this hour"}


def test_requests_read_from_jsonl_or_a_table_and_a_bad_line_costs_itself(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(_req()) + "\nnot json\n\n" + json.dumps(_req(episode_id="E2")) + "\n")
    rows = read_requests(str(p))
    assert [r.get("episode_id") for r in rows] == ["E1", None, "E2"]
    assert validate_request(rows[1])                       # rejected, batch intact
    t = tmp_path / "r.parquet"
    pd.DataFrame([_req(), _req(episode_id="E2", current_discount=0.1)]).to_parquet(t)
    rows = read_requests(str(t))
    assert rows[0]["current_discount"] is None and rows[1]["current_discount"] == 0.1
    assert validate_request(rows[0]) == []


def test_the_outcome_id_is_the_hours_key_computable_from_the_feed_row():
    k = hour_key(7.0, "F1", pd.Timestamp("2026-08-19"), 17)
    assert k == ("7", "F1", "2026-08-19", 17)
    assert outcome_id_of(k) == "feed-7|F1|2026-08-19T17"
    assert outcome_id_of(hour_key("7", "F1", "2026-08-19", 5)) == "feed-7|F1|2026-08-19T05"
    with pytest.raises(ValueError):
        hour_key(7.5, "F1", "2026-08-19", 17)
    with pytest.raises(ValueError):
        hour_key(7, "F1", "2026-08-19", float("nan"))


def test_response_rows_carry_the_contract_fields_only(tmp_path):
    rows = [{"episode_id": "E1", "sku_id": 7, "fc": "F1", "date": "2026-08-19",
             "hour_of_day": 17, "decision_id": "D1", "applied_discount": 0.1,
             "applied_price": 9000.0, "is_exploration": False, "rejected": None,
             "extra": "dropped"}]
    p = tmp_path / "out.jsonl"
    write_rows(rows, str(p))
    got = json.loads(p.read_text().strip())
    assert tuple(got) == RESPONSE_FIELDS and "extra" not in got
