"""ops.price_batch -- the batch surface Lane B calls: row-scoped refusals,
the one hour key, one spelling of every request, the outcome id
engineering can name from the feed, and a later hour priced on the entry
forecast."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import load_config
from engine.posterior import PosteriorStore
from engine.state import (HISTORY_COLS, REQUEST_FIELDS, build_states,
                          canonical_request, validate_request)
from events.pairs import hour_key, outcome_id_of
from events.store import EventStore
from ops.price_batch import RESPONSE_FIELDS, plan, read_requests, run, write_rows

CFG = load_config()
R_LOOKUP = {"fallback_order": ["subcategory", "category", "global"],
            "subcategory": {}, "category": {}, "global": 0.9}


def _req(**over):
    r = {"episode_id": "E1", "sku_id": 7, "fc": "F1", "category": "VEG",
         "subcategory": "LEAFY", "date": "2026-08-19", "hour_of_day": 17,
         "hours_remaining": 4, "q": 3, "original_price": 10000.0, "cost": 4000.0,
         "current_discount": None}
    r.update(over)
    return r


class _Model:
    """A frozen model that answers a constant and remembers every frame it
    was asked to predict (so a test can say "no history pass happened")."""
    schema = {"model_version": "stub"}
    version = schema["model_version"]          # as BaselineModel spells it

    def __init__(self, mu=0.8):
        self.mu, self.calls = mu, []

    def predict_mu_ref(self, frame):
        self.calls.append(frame.copy())
        return np.full(len(frame), self.mu)


def _history(skus=(7,), days=range(1, 19), sku_dtype=None):
    """Anchor-priced hours for `skus` on August `days`: one episode a day
    selling one unit at the reference discount, so both rate features
    resolve to 1.0 for an opening on the 19th."""
    rows = [{"episode_id": f"{s}|F1|2026-08-{d:02d}T10", "sku_id": s, "fc": "F1",
             "category": "VEG", "date": f"2026-08-{d:02d}", "hour_of_day": 10,
             "starting_inventory": 3, "units_sold": 1, "total_discount": 0.30}
            for s in skus for d in days]
    h = pd.DataFrame(rows, columns=list(HISTORY_COLS))
    if sku_dtype:
        h["sku_id"] = h["sku_id"].astype(sku_dtype)
    return h


def _world(tmp_path, cfg=None):
    cfg = cfg or load_config()
    posterior = PosteriorStore.initialise(
        cfg, {"VEG": {"mean": -1.0, "std": 0.6}}, {"VEG": 10 ** 6},
        path=str(tmp_path / "posterior.json"))
    store = EventStore(cfg, root=str(tmp_path / "events"))
    return cfg, store, posterior


# --------------------------------------------------------------- validation
def test_a_request_needs_every_contract_field_and_a_layable_grid():
    v = lambda r: validate_request(r, CFG)                        # noqa: E731
    assert v(_req()) == []
    assert v({k: x for k, x in _req().items() if k != "q"}) == ["missing q"]
    assert any("hour_of_day" in p for p in v(_req(hour_of_day=24)))
    assert any("hours_remaining" in p for p in v(_req(hours_remaining=0)))
    assert any("q" in p for p in v(_req(q=-1)))
    assert any("date" in p for p in v(_req(date="not a day")))
    assert any("sku_id" in p for p in v(_req(sku_id=None)))
    # the counts are judged by the one home the state is judged by
    # (engine.decide.count_failures): the horizon bound included
    cap = CFG["data"]["max_window_hours"]
    assert any("max_window_hours" in p for p in v(_req(hours_remaining=cap + 1)))
    assert v(_req(hours_remaining=cap)) == []
    # a null or non-numeric price or cost cannot become a state; its VALUE
    # (zero, negative, above list) is the engine's to judge -- nothing twice
    assert any("original_price is null" in p for p in v(_req(original_price=None)))
    assert any("cost is null" in p for p in v(_req(cost=None)))
    assert any("not a number" in p for p in v(_req(cost="4000")))
    assert v(_req(cost=float("nan"))) == []                # finiteness: the engine's
    assert v(_req(original_price=0.0, cost=-1.0)) == []
    assert tuple(_req()) == REQUEST_FIELDS


def test_a_request_is_read_in_one_spelling_whatever_the_producers_dtypes():
    """A parquet timestamp, an id read back as 7.0, an integer category:
    one spelling (the hour key's), so the request meets its history rows,
    its stored decisions and the store's ISO day."""
    c = canonical_request(_req(date=pd.Timestamp("2026-08-19"), sku_id=7.0, fc=np.int64(4),
                               category=12, hour_of_day=17.0, hours_remaining=4.0,
                               q=np.int64(3), current_discount=float("nan")))
    assert (c["date"], c["sku_id"], c["fc"], c["category"]) == ("2026-08-19", "7", "4", "12")
    assert (c["hour_of_day"], c["hours_remaining"], c["q"]) == (17, 4, 3)
    assert c["current_discount"] is None
    assert canonical_request(c) == c                       # idempotent
    assert tuple(c) == REQUEST_FIELDS


# --------------------------------------------------------------------- plan
def test_plan_refuses_row_by_row_never_the_batch():
    reqs = [_req(),                                          # prices
            _req(episode_id="E2", hour_of_day=99),          # cannot become a state
            _req(episode_id="E3", sku_id=8),                 # prices
            _req(episode_id="E4", sku_id=8),                 # E3's hour again
            _req(episode_id="E5", sku_id=9)]                 # already in the store
    priced = {hour_key(9, "F1", "2026-08-19", 17)}
    to_price, rejected = plan(reqs, priced, CFG)
    assert [i for i, _, _ in to_price] == [0]
    assert to_price[0][1]["sku_id"] == "7"                   # handed on canonical
    assert "hour_of_day" in rejected[1]
    assert rejected[2].startswith("duplicate_request") and rejected[3].startswith("duplicate_request")
    assert rejected[4].startswith("already_priced")
    # the key is spelt one way whatever the caller's dtype (7.0 is "7")
    assert plan([_req(sku_id=7.0)], {hour_key("7", "F1", "2026-08-19", 17.0)}, CFG)[1] == {
        0: "already_priced: the store holds a decision for this hour"}


def test_requests_read_from_jsonl_or_a_table_and_a_bad_line_costs_itself(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(_req()) + "\nnot json\n\n" + json.dumps(_req(episode_id="E2")) + "\n")
    rows = read_requests(str(p))
    assert [r.get("episode_id") for r in rows] == ["E1", None, "E2"]
    assert validate_request(rows[1], CFG)                  # rejected, batch intact
    t = tmp_path / "r.parquet"
    pd.DataFrame([_req(), _req(episode_id="E2", current_discount=0.1)]).to_parquet(t)
    rows = read_requests(str(t))
    assert rows[0]["current_discount"] is None and rows[1]["current_discount"] == 0.1
    assert validate_request(rows[0], CFG) == []


def test_the_outcome_id_is_the_hours_key_computable_from_the_feed_row():
    k = hour_key(7.0, "F1", pd.Timestamp("2026-08-19"), 17)
    assert k == ("7", "F1", "2026-08-19", 17)
    assert outcome_id_of(k) == "feed-7|F1|2026-08-19T17"
    assert outcome_id_of(hour_key("7", "F1", "2026-08-19", 5)) == "feed-7|F1|2026-08-19T05"
    with pytest.raises(ValueError):
        hour_key(7.5, "F1", "2026-08-19", 17)
    with pytest.raises(ValueError):
        hour_key(7, "F1", "2026-08-19", float("nan"))
    with pytest.raises(ValueError):
        hour_key(None, "F1", "2026-08-19", 17)


def test_response_rows_carry_the_contract_fields_only_and_always_write(tmp_path):
    rows = [{"episode_id": "E1", "sku_id": 7, "fc": "F1", "date": "2026-08-19",
             "hour_of_day": 17, "decision_id": "D1", "applied_discount": 0.1,
             "applied_price": 9000.0, "is_exploration": False, "rejected": None,
             "extra": "dropped"},
            # a rejected row echoes what it was sent: a timestamp and a NaN
            # from a table must write, never raise after the batch committed
            {"episode_id": "E2", "sku_id": np.int64(8), "fc": "F1",
             "date": pd.Timestamp("2026-08-19"), "hour_of_day": float("nan"),
             "rejected": "hour_of_day must be an integer in 0..23"}]
    p = tmp_path / "out.jsonl"
    write_rows(rows, str(p))
    got = [json.loads(line) for line in p.read_text().splitlines()]
    assert tuple(got[0]) == RESPONSE_FIELDS and "extra" not in got[0]
    assert got[1]["hour_of_day"] is None and got[1]["decision_id"] is None
    assert got[1]["sku_id"] == 8 and "2026-08-19" in got[1]["date"]


# ---------------------------------------------------------------- the batch
def test_a_parquet_request_table_with_a_datetime_date_prices_commits_and_writes(tmp_path):
    """A parquet date column prices, commits, and then the response could
    not be written (a Timestamp is not JSON): the hour was in the record
    and no price reached the shelf, and the retry was `already_priced`.
    The request is canonicalised once; the store sees an ISO day."""
    cfg, store, posterior = _world(tmp_path)
    table = tmp_path / "r.parquet"
    pd.DataFrame([_req(), _req(episode_id="E2", sku_id=8)]).assign(
        date=pd.to_datetime("2026-08-19")).to_parquet(table)
    requests = read_requests(str(table))
    assert isinstance(requests[0]["date"], pd.Timestamp)
    rows, events, rep = run(cfg, requests, _history(skus=(7, 8)), store=store,
                            model=_Model(), posterior=posterior, r_lookup=R_LOOKUP)
    assert rep["decisions"] == 2 and rep["rejected"] == 0, rep
    assert all(e["date"] == "2026-08-19" and e["sku_id"] in ("7", "8") for e in events)
    assert [d["decision_id"] for d in store.load_decisions()] == [e["decision_id"] for e in events]
    out = tmp_path / "out.jsonl"
    write_rows(rows, str(out))
    got = [json.loads(line) for line in out.read_text().splitlines()]
    assert [g["date"] for g in got] == ["2026-08-19", "2026-08-19"]
    assert all(g["decision_id"] and g["rejected"] is None for g in got)
    # the same hour sent again is refused by the store's own index
    rows, events, again = run(cfg, requests, _history(skus=(7, 8)), store=store,
                              model=_Model(), posterior=posterior, r_lookup=R_LOOKUP)
    assert not events and all(r["rejected"].startswith("already_priced") for r in rows)


def test_an_id_dtype_mismatch_does_not_price_on_unknown_features(tmp_path):
    """A JSONL "7" against an int history merged nothing: both rate
    features NaN for every request, the model pricing on "unknown", and
    nothing counting it. Ids are read in the hour key's spelling on both
    sides, and a request priced on no history is counted."""
    cfg, store, posterior = _world(tmp_path)
    model = _Model()
    hist = _history(skus=(7,), sku_dtype="int64")
    rows, events, rep = run(cfg, [_req(sku_id="7")], hist, store=store, model=model,
                            posterior=posterior, r_lookup=R_LOOKUP)
    assert rep["decisions"] == 1 and rep["requests_with_unknown_features"] == 0
    frame = model.calls[0]
    assert frame.sku_ref_sales_rate_30d.notna().all()
    assert frame.sku_ref_sales_rate_30d.iloc[0] == pytest.approx(1.0)
    # a SKU the history has never seen IS unknown, and says so
    rows, events, rep = run(cfg, [_req(episode_id="E9", sku_id=99)], hist, store=store,
                            model=_Model(), posterior=posterior, r_lookup=R_LOOKUP)
    assert rep["decisions"] == 1 and rep["requests_with_unknown_features"] == 1


def test_a_null_price_and_an_integer_category_are_row_scoped(tmp_path):
    """float(None) in build_states and a KeyError inside the worker (cells
    keyed raw, looked up by str) each took the whole batch down. A null
    cost is one rejected row; an integer category prices."""
    cfg, store, posterior = _world(tmp_path)
    reqs = [_req(cost=None),                                   # rejected: null
            _req(episode_id="E2", sku_id=8, category=12),      # prices on GLOBAL
            _req(episode_id="E3", sku_id=9, original_price=None)]
    rows, events, rep = run(cfg, reqs, _history(skus=(7, 8, 9)), store=store,
                            model=_Model(), posterior=posterior, r_lookup=R_LOOKUP)
    assert rep["decisions"] == 1 and rep["rejected_before_the_engine"] == 2
    assert "cost is null" in rows[0]["rejected"]
    assert "original_price is null" in rows[2]["rejected"]
    assert events[0]["category"] == "12" and rows[1]["decision_id"] == events[0]["decision_id"]


def test_a_later_hour_of_a_known_episode_is_priced_on_the_entry_forecast_sliced():
    """A mid-episode request recomputed its features as of the request
    day, not the episode's opening, and the stub's episode_id never met
    history's derived ids. The forecast is made once, at entry: a later
    hour is the stored path sliced to the hour -- no history pass, no
    prediction -- and equals what the entry priced on, so assurance can
    re-solve it. A restock that grew the window is extended by
    prediction on features as of the OPENING, appended to the slice."""
    model = _Model(mu=0.5)
    stored = {"E1": {"date": "2026-08-19", "hour_of_day": 17, "hours_remaining": 4,
                     "mu_ref_path": [0.9, 0.7, 0.5, 0.3],
                     "opened": ("2026-08-19", 17)}}
    later = _req(hour_of_day=18, hours_remaining=3, q=2, current_discount=0.3)
    states, notes = build_states([later], _history(), CFG, model, R_LOOKUP,
                                 episode_paths=stored)
    assert states[0]["mu_ref_path"] == [0.7, 0.5, 0.3]
    assert model.calls == [] and notes["non_entry_requests_without_stored_path"] == 0
    # the window grew by two hours (a restock): the slice, then two
    # predicted hours on the opening's features
    restock = _req(hour_of_day=19, hours_remaining=4, q=5, current_discount=0.3)
    states, _ = build_states([restock], _history(), CFG, model, R_LOOKUP,
                             episode_paths=stored)
    assert states[0]["mu_ref_path"] == [0.5, 0.3, 0.5, 0.5]
    assert len(model.calls) == 1
    predicted = model.calls[0]
    assert list(zip(predicted.date, predicted.hour_of_day)) == [("2026-08-19", 21), ("2026-08-19", 22)]
    # a later request the store does not know falls back to a fresh
    # forecast, counted -- a listing already on clearance at launch
    model = _Model(mu=0.5)
    states, notes = build_states([later], _history(), CFG, model, R_LOOKUP, episode_paths={})
    assert states[0]["mu_ref_path"] == [0.5, 0.5, 0.5]
    assert notes["non_entry_requests_without_stored_path"] == 1
    # and an entry request is always a fresh forecast
    states, notes = build_states([_req()], _history(), CFG, model, R_LOOKUP,
                                 episode_paths=stored)
    assert states[0]["mu_ref_path"] == [0.5] * 4 and notes["requests_with_unknown_features"] == 0


def test_the_batch_prices_a_later_hour_from_the_store_it_committed_to(tmp_path):
    """Through run(): the entry decision's stored path is what the next
    hour of the same episode is priced on (events.store.episode_paths)."""
    cfg, store, posterior = _world(tmp_path)
    model = _Model()
    rows, events, _ = run(cfg, [_req()], _history(), store=store, model=model,
                          posterior=posterior, r_lookup=R_LOOKUP)
    entry = events[0]
    calls_after_entry = len(model.calls)
    later = _req(hour_of_day=18, hours_remaining=3, q=2,
                 current_discount=entry["applied_discount"])
    rows, events, rep = run(cfg, [later], _history(), store=store, model=model,
                            posterior=posterior, r_lookup=R_LOOKUP)
    assert rep["decisions"] == 1 and rep["non_entry_requests_without_stored_path"] == 0
    assert events[0]["mu_ref_path"] == entry["mu_ref_path"][1:]
    assert events[0]["is_entry"] is False
    assert len(model.calls) == calls_after_entry             # no history pass
