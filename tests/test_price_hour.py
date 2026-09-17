"""ops.price_hour -- one clock hour from the feed's own rows: the
producer's episode id read as given (a new id is an entry, the id the
store last priced on the shelf continues it), the anchor from the price in
force, the chain's rule evaluated only to COUNT disagreements
(LIVE_RULE), the response in the feed's units, and a dry run that commits
nothing."""
import json
import os

import pandas as pd
import pytest

from conftest import _Applier, load_config
from engine.posterior import PosteriorStore
from engine.state import ref_rate_table
from events.store import EventStore
from ops import assign_episode_ids, price_hour

R_LOOKUP = {"fallback_order": ["subcategory", "category", "global"],
            "subcategory": {}, "category": {}, "global": 0.9}


def _snap(**over):
    """One shelf at the top of the hour, in the feed's own names (the
    discount a PERCENT) with the producer's `episode_id` -- by default the
    id a first hour gets, sku|fc|<date>T<hour>; keywords override."""
    r = {"date": "2026-08-19", "hour": 18, "skuseq": 7, "fc": "F1", "inventory": 2.0,
         "discount": 15.0, "units_sold": None, "normal_asp": 10000.0, "final_price": None,
         "cogs_wo_vat": 4000.0, "ending_inventory": None, "flc_window": 3.0,
         "category": "VEG", "subcategory": "LEAFY"}
    r.update(over)
    if "episode_id" not in over:
        r["episode_id"] = (assign_episode_ids.new_id(r) if r["skuseq"] is not None else None)
    return r


def _history(skus=(7,), days=range(1, 19)):
    return pd.DataFrame([
        {"episode_id": f"{s}|F1|2026-08-{d:02d}T10", "sku_id": s, "fc": "F1",
         "category": "VEG", "date": f"2026-08-{d:02d}", "hour_of_day": 10,
         "starting_inventory": 3, "units_sold": 1, "total_discount": 0.30}
        for s in skus for d in days])


def _world(tmp_path):
    cfg = load_config()
    cfg["events"]["store_dir"] = str(tmp_path / "events")
    posterior = PosteriorStore.initialise(
        cfg, {"VEG": {"mean": -1.0, "std": 0.6}}, {"VEG": 10 ** 6},
        path=str(tmp_path / "posterior.json"))
    store = EventStore(cfg)
    model = _Applier(cfg, base_mu=0.8, anchor={"VEG": 1.0})
    model.calibration_grain = "category"
    table = ref_rate_table(_history(skus=(7, 8)), "2026-08-19", cfg)
    return cfg, store, posterior, model, table


def _price(cfg, rows, store, posterior, model, table, **kw):
    return price_hour.run(cfg, rows, features=table, store=store, model=model,
                          posterior=posterior, r_lookup=R_LOOKUP, **kw)


def _write(tmp_path, rows, name="snap.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_an_entry_hour_opens_the_producers_episode_and_answers_in_percent(tmp_path):
    """A shelf whose id the store has never priced is an entry (anchor
    null); an empty shelf and a row without an id each cost one response
    line and never reach the engine."""
    cfg, store, posterior, model, table = _world(tmp_path)
    rows = price_hour.read_snapshot(_write(tmp_path, [
        _snap(), _snap(skuseq=9, inventory=0.0), _snap(skuseq=10, episode_id="")]))
    response, events, rep = _price(cfg, rows, store, posterior, model, table)
    assert rep["hour"] == "2026-08-19T18" and rep["shelves"] == 3
    assert rep["episodes_new"] == 1 and rep["shelves_empty"] == 1 and rep["decisions"] == 1
    assert rep["shelves_without_episode_id"] == 1
    assert rep["episode_ids_disagreeing_with_the_rule"] == 0
    priced, empty, unnamed = response
    assert priced["episode_id"] == "7|F1|2026-08-19T18"          # the producer's id, as given
    assert events[0]["episode_id"] == "7|F1|2026-08-19T18"
    assert events[0]["is_entry"] and events[0]["anchor_discount"] is None
    assert priced["apply_discount_pct"] == pytest.approx(events[0]["applied_discount"] * 100)
    assert priced["apply_price"] == events[0]["applied_price"] and priced["rejected"] is None
    assert empty["rejected"].startswith("empty shelf") and empty["decision_id"] is None
    assert unnamed["rejected"].startswith("episode_id missing") and unnamed["decision_id"] is None
    assert list(response[0]) == list(price_hour.RESPONSE_COLS)


def test_the_next_hour_continues_the_episode_on_the_price_in_force(tmp_path):
    """The producer sends the same id one hour later: the same episode,
    anchored on the discount the snapshot says is on the shelf (a failed
    push keeps the old price, and the engine must know that). The rule,
    read against the store's latest decision, agrees."""
    cfg, store, posterior, model, table = _world(tmp_path)
    first = price_hour.read_snapshot(_write(tmp_path, [_snap(hour=17, inventory=3.0, flc_window=3.0)]))
    _, ev1, _ = _price(cfg, first, store, posterior, model, table)
    assert ev1[0]["hours_remaining"] == 4                        # this hour plus the counter
    applied = ev1[0]["applied_discount"]
    shelf_pct = round(applied * 100, 4)
    nxt = price_hour.read_snapshot(_write(tmp_path, [
        _snap(hour=18, inventory=2.0, flc_window=2.0, discount=shelf_pct,
              episode_id=ev1[0]["episode_id"])], "n.csv"))
    response, ev2, rep = _price(cfg, nxt, store, posterior, model, table)
    assert rep["episodes_continued"] == 1 and rep["episodes_new"] == 0
    assert rep["episode_ids_disagreeing_with_the_rule"] == 0
    assert response[0]["episode_id"] == ev1[0]["episode_id"]
    assert not ev2[0]["is_entry"] and ev2[0]["anchor_discount"] == pytest.approx(applied)
    assert rep["non_entry_requests_without_stored_path"] == 0     # the stored path, sliced


def test_the_producers_ids_are_read_as_given_and_only_checked_against_the_rule(tmp_path):
    """The producer runs the reference rule (ops.assign_episode_ids) on
    the hour that just closed: a write-off zero ends the episode however
    the counter moves; a counter that steps up with stock arrived
    continues it; a counter that steps up with nothing arrived is a new
    listing. Those ids reach the engine unchanged and agree with the
    rule. An id that contradicts the rule is priced as the producer says
    and COUNTED, never overridden."""
    cfg, store, posterior, model, table = _world(tmp_path)
    openings, _ = assign_episode_ids.assign(
        [_snap(skuseq=s, hour=17, inventory=3.0, flc_window=3.0) for s in (7, 8, 9)])
    _price(cfg, price_hour.read_snapshot(_write(tmp_path, openings)), store, posterior, model, table)
    by_sku = {r["skuseq"]: r for r in openings}
    closed = [
        # 7 sold out: the write-off zero closes it although the counter continues
        dict(by_sku[7], units_sold=3, ending_inventory=0.0),
        # 8 restocked: ending above opening - sold, counter steps UP next hour
        dict(by_sku[8], units_sold=1, ending_inventory=6.0),
        # 9 neither: counter steps up next hour with nothing arrived
        dict(by_sku[9], units_sold=1, ending_inventory=2.0)]
    now, counts = assign_episode_ids.assign(
        [_snap(skuseq=7, hour=18, inventory=5.0, flc_window=2.0),
         _snap(skuseq=8, hour=18, inventory=6.0, flc_window=5.0),
         _snap(skuseq=9, hour=18, inventory=2.0, flc_window=5.0)], closed)
    assert counts == {"continued": 1, "new": 2, "unkeyable": 0}
    rows = price_hour.read_snapshot(_write(tmp_path, closed + now, "n.csv"))
    response, _, rep = _price(cfg, rows, store, posterior, model, table)
    assert rep["closed_rows_seen"] == 3 and rep["shelves"] == 3
    assert rep["episodes_new"] == 2 and rep["episodes_continued"] == 1
    assert rep["episode_ids_disagreeing_with_the_rule"] == 0
    by = {r["skuseq"]: r for r in response}
    assert by["7"]["episode_id"] == "7|F1|2026-08-19T18"         # closed -> new
    assert by["8"]["episode_id"] == "8|F1|2026-08-19T17"         # restock -> continued
    assert by["9"]["episode_id"] == "9|F1|2026-08-19T18"         # reset -> new
    # without the closed rows: 8's opening above its last remaining stock reads as a restock
    latest = store.latest_by_shelf[("8", "F1")]
    assert price_hour.rule_says_continues(latest, price_hour.read_snapshot(
        _write(tmp_path, [_snap(skuseq=8, hour=19, inventory=9.0, flc_window=4.0)], "o.csv"))[0], None)
    # the producer keeps 9 on its first id although the rule says a new listing:
    # priced as continued, counted as a disagreement, never overridden
    contra = price_hour.read_snapshot(_write(tmp_path, [
        _snap(skuseq=9, hour=19, inventory=2.0, flc_window=4.0, episode_id="9|F1|2026-08-19T18")], "c.csv"))
    response, ev, rep = _price(cfg, contra, store, posterior, model, table)
    assert response[0]["episode_id"] == "9|F1|2026-08-19T18" and not ev[0]["is_entry"]
    assert rep["episodes_continued"] == 1 and rep["episode_ids_disagreeing_with_the_rule"] == 0
    contra2 = price_hour.read_snapshot(_write(tmp_path, [
        _snap(skuseq=9, hour=20, inventory=2.0, flc_window=6.0, episode_id="9|F1|2026-08-19T18")], "d.csv"))
    response, ev, rep = _price(cfg, contra2, store, posterior, model, table)
    assert response[0]["episode_id"] == "9|F1|2026-08-19T18" and not ev[0]["is_entry"]
    assert rep["episode_ids_disagreeing_with_the_rule"] == 1
    assert rep["disagreements_sample"] == [{"skuseq": "9", "fc": "F1", "episode_id": "9|F1|2026-08-19T18",
                                            "producer": "continued", "rule": "new"}]


def test_a_refused_hour_is_recorded_so_the_shelf_is_still_checkable_next_hour(tmp_path):
    """A refused row is not priced, but it IS seen. Recording it
    (events.store rejections) keeps the shelf's chain unbroken, so the rule
    steps from the refused hour instead of reporting a gap -- and what stays
    unknown is then an hour engineering never sent."""
    cfg, store, posterior, model, table = _world(tmp_path)
    first = price_hour.read_snapshot(_write(tmp_path, [_snap(hour=17, inventory=3.0, flc_window=4.0)]))
    _, ev1, _ = _price(cfg, first, store, posterior, model, table)
    eid = ev1[0]["episode_id"]
    # hour 18: the cost is impossible, so the engine refuses the row
    bad = price_hour.read_snapshot(_write(tmp_path, [
        _snap(hour=18, inventory=2.0, flc_window=3.0, cogs_wo_vat=99999.0,
              episode_id=eid)], "b.csv"))
    response, _, rep = _price(cfg, bad, store, posterior, model, table)
    assert response[0]["rejected"] and response[0]["decision_id"] is None
    assert rep["rejections_recorded"] == 1
    seen = store.last_seen_by_shelf[("7", "F1")]
    assert (seen["date"], seen["hour_of_day"], seen["priced"]) == ("2026-08-19", 18, False)
    assert store.latest_by_shelf[("7", "F1")]["hour_of_day"] == 17     # still the decision
    assert ("7", "F1", "2026-08-19", 18) not in store.priced_hours     # not priced
    # hour 19 on the producer's same id: the rule steps from the refused
    # hour, agrees, and nothing is reported as unknown
    nxt = price_hour.read_snapshot(_write(tmp_path, [
        _snap(hour=19, inventory=2.0, flc_window=2.0, episode_id=eid)], "n.csv"))
    _, _, rep = _price(cfg, nxt, store, posterior, model, table)
    assert rep["episodes_continued"] == 1
    assert rep["episode_ids_the_rule_could_not_check"] == 0
    assert rep["episode_ids_disagreeing_with_the_rule"] == 0
    # the rejection is in the record, with the reason the response carried
    rej = store.load_rejections()
    assert len(rej) == 1 and rej[0]["rejection_id"] == "rej-7|F1|2026-08-19T18"
    assert rej[0]["episode_id"] == eid and rej[0]["reason"] == response[0]["rejected"]


def test_a_gap_is_unknown_to_the_rule_never_a_disagreement(tmp_path):
    """An hour we refused to price stores no decision, so next hour the
    store's latest decision is two hours back and the rule has nothing to
    step from. That is unknown, not "a new window": counting it would
    report every shelf held through a rejection as a disagreement. The
    closed rows, when they ride along, answer it properly."""
    cfg, store, posterior, model, table = _world(tmp_path)
    first = price_hour.read_snapshot(_write(tmp_path, [_snap(hour=17, inventory=3.0, flc_window=4.0)]))
    _, ev1, _ = _price(cfg, first, store, posterior, model, table)
    eid = ev1[0]["episode_id"]
    # hour 18 is never priced (rejected upstream, or a missed cron hour);
    # hour 19 arrives on the producer's same id, the shelf having been held
    gap = price_hour.read_snapshot(_write(tmp_path, [
        _snap(hour=19, inventory=2.0, flc_window=2.0, episode_id=eid)], "g.csv"))
    _, _, rep = _price(cfg, gap, store, posterior, model, table)
    assert rep["episodes_continued"] == 1 and rep["episodes_new"] == 0
    assert rep["episode_ids_disagreeing_with_the_rule"] == 0
    assert rep["episode_ids_the_rule_could_not_check"] == 1     # counted, not hidden
    latest = store.latest_by_shelf[("7", "F1")]
    assert price_hour.rule_says_continues(latest, gap[0], None) is None
    # with last hour's closed row the rule is decisive again
    closed = _snap(hour=19, inventory=2.0, flc_window=2.0, units_sold=1, ending_inventory=1.0)
    now = price_hour.snapshot_rows([_snap(hour=20, inventory=1.0, flc_window=1.0, episode_id=eid)])[0]
    assert price_hour.rule_says_continues(None, now, price_hour.snapshot_rows([closed])[0]) is True


def test_a_dry_run_prices_and_commits_nothing(tmp_path):
    cfg, store, posterior, model, table = _world(tmp_path)
    rows = price_hour.read_snapshot(_write(tmp_path, [_snap()]))
    response, events, rep = _price(cfg, rows, None, posterior, model, table, dry_run=True)
    assert rep["dry_run"] and rep["decisions"] == 1 and response[0]["decision_id"]
    assert not EventStore(cfg).load_decisions()                   # the real store is untouched
    response2, _, rep2 = _price(cfg, rows, None, posterior, model, table)
    assert rep2["decisions"] == 1                                 # the hour is still free to price


def test_the_cli_writes_the_response_and_the_report(tmp_path, monkeypatch):
    """A row that names no shelf-hour costs itself one response line."""
    cfg, store, posterior, model, table = _world(tmp_path)
    monkeypatch.setattr(price_hour, "EventStore", lambda c, root=None: store)
    monkeypatch.setattr(price_hour.price_batch, "load_bundle",
                        lambda c: type("B", (), {"model": model, "posterior": posterior,
                                                 "r_lookup": R_LOOKUP, "prior": None})())
    monkeypatch.setattr(price_hour, "load_config", lambda path, strict=False: cfg)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("{}")
    table_path = tmp_path / "features.parquet"
    table.to_parquet(table_path, index=False)
    snap = _write(tmp_path, [_snap(), _snap(skuseq=None, hour=18)])
    out, rep = tmp_path / "18.csv", tmp_path / "18.json"
    rc = price_hour.main(["--snapshot", snap, "--features", str(table_path), "--out", str(out),
                          "--report", str(rep), "--config", str(cfg_path)])
    assert rc == 0
    got = pd.read_csv(out)
    assert list(got.columns) == list(price_hour.RESPONSE_COLS) and len(got) == 2
    assert got.rejected.iloc[1] == "row names no shelf-hour"
    report = json.load(open(rep))
    assert report["shelves_unkeyable"] == 1 and report["decisions"] == 1
    assert report["live_rule"] == price_hour.LIVE_RULE and os.path.exists(report["response"])
