"""ops.check_inputs -- engineering's three files, checked against what the
chain needs, with the count and the fix on every line."""
import pandas as pd

from conftest import source_row, source_window, write_extract
from ops import check_inputs


def _verdicts(rows):
    return {r["check"]: r["verdict"] for r in rows}


def test_a_fraction_discount_and_a_bad_counter_fail_a_snapshot(cfg, tmp_path):
    good = [source_row(hour=18, skuseq=s, inventory=3.0, flc_window=4.0, units_sold=None,
                       ending_inventory=None, final_price=None,
                       episode_id=f"{s}|F1|2026-03-02T18") for s in (1, 2)]
    snap = tmp_path / "snap.csv"
    pd.DataFrame(good).to_csv(snap, index=False)
    v = _verdicts(check_inputs.check_snapshot(str(snap), cfg).rows)
    assert all(x == "PASS" for x in v.values()), v
    bad = [dict(r, discount=0.25) for r in good] + [source_row(hour=18, skuseq=3, discount=0.3,
                                                                flc_window=9000.0, episode_id=None)]
    pd.DataFrame(bad).to_csv(snap, index=False)
    v = _verdicts(check_inputs.check_snapshot(str(snap), cfg).rows)
    assert v["discount is a PERCENT (25.0, not 0.25)"] == "FAIL"
    cap = cfg["data"]["max_window_hours"]
    assert v[f"flc_window means hours still to come: 0..{cap - 1} on stocked rows"] == "FAIL"
    assert v["episode_id present and never null (the producer's)"] == "FAIL"
    # the column absent altogether is the same failure, with every row counted
    pd.DataFrame(good).drop(columns=["episode_id"]).to_csv(snap, index=False)
    v = _verdicts(check_inputs.check_snapshot(str(snap), cfg).rows)
    assert v["episode_id present and never null (the producer's)"] == "FAIL"


def _idle(rows, i):
    """Hour `i` of a source window sells nothing, the chain stays whole."""
    rows = [dict(r) for r in rows]
    rows[i]["units_sold"] = 0
    for j in range(i, len(rows) - 1):
        rows[j]["ending_inventory"] = rows[j]["inventory"] - rows[j]["units_sold"]
        rows[j + 1]["inventory"] = rows[j]["ending_inventory"]
    return rows


def test_a_feed_runs_through_the_chain_and_its_waterfall_is_read(cfg, tmp_path):
    rows = _idle(source_window(1, 10, 5, day="2026-03-02"), 1) \
        + source_window(2, 12, 4, day="2026-03-02")
    feed = write_extract(tmp_path, rows, name="feed.parquet")
    out = check_inputs.check_feed(str(feed), cfg).rows
    v = _verdicts(out)
    assert v["the preparation chain runs on the feed"] == "PASS"
    assert v["zero-sale hours are present"] in ("PASS", "FAIL")     # measured, never skipped
    assert v["windows closed by the write-off zero"] == "PASS"
    assert not any(r["verdict"] == "FAIL" for r in out), out
    # a feed with no idle hour at all is the completeness trap, named
    busy = source_window(1, 10, 5, day="2026-03-02")
    feed2 = write_extract(tmp_path, busy, name="busy.parquet")
    assert _verdicts(check_inputs.check_feed(str(feed2), cfg).rows)["zero-sale hours are present"] == "FAIL"


def _feed_with_ids(tmp_path, rows, ids, name):
    """The feed's own rows plus the producers' `episode_id` column (the
    source schema has no such field, so it is appended here as the
    producers would append it)."""
    path = tmp_path / name
    pd.DataFrame([dict(r, episode_id=e) for r, e in zip(rows, ids)]).to_parquet(path, index=False)
    return str(path)


def test_the_feeds_episode_ids_are_read_against_the_rule_over_the_whole_day(cfg, tmp_path):
    """Asked of the producers so a window is one group-by for everyone: the
    check is on the window BOUNDARIES, never the id's spelling, because the
    scheme is theirs. Absent, it is a WARN and the feed still passes."""
    rows = _idle(source_window(1, 10, 5, day="2026-03-02"), 1)
    check = "the producers' ids group the same windows EPISODE_RULE derives"
    absent = _verdicts(check_inputs.check_feed(str(write_extract(tmp_path, rows)), cfg).rows)
    assert absent["episode_id carried in the feed"] == "WARN" and check not in absent
    # one window, one id -- any spelling of it, since the scheme is theirs
    ok = _feed_with_ids(tmp_path, rows, ["THEIR-WINDOW-7"] * len(rows), "ids_ok.parquet")
    v = _verdicts(check_inputs.check_feed(ok, cfg).rows)
    assert v[check] == "PASS" and v["episode_id never null (the feed's copy of the producers' id)"] == "PASS"
    # the same rows split in two mid-window: their id opens where the rule does not
    split = ["A", "A", "B", "B", "B"][:len(rows)]
    v = _verdicts(check_inputs.check_feed(_feed_with_ids(tmp_path, rows, split, "ids_bad.parquet"), cfg).rows)
    assert v[check] == "FAIL"
    nulls = [None] * len(rows)
    v = _verdicts(check_inputs.check_feed(_feed_with_ids(tmp_path, rows, nulls, "ids_null.parquet"), cfg).rows)
    assert v["episode_id never null (the feed's copy of the producers' id)"] == "FAIL"


def test_a_failures_table_must_name_shelf_hours(cfg, tmp_path):
    f = tmp_path / "fail.csv"
    pd.DataFrame([{"skuseq": 1, "fc": "F1", "date": "2026-03-02", "hour": 10, "reason": "timeout"},
                  {"skuseq": None, "fc": "F1", "date": "2026-03-02", "hour": 11, "reason": "x"}]
                 ).to_csv(f, index=False)
    v = _verdicts(check_inputs.check_failures(str(f), cfg).rows)
    assert v["every row names one shelf-hour"] == "FAIL"
    text = check_inputs.render(check_inputs.check_failures(str(f), cfg).rows)
    assert "1 FAIL" in text and "-> a row with a null id" in text


def test_a_null_feed_id_fails_its_own_gate_and_is_not_a_window_boundary(cfg, tmp_path):
    """A null id read through astype(str) became the token "nan" and
    opened a window of its own in the boundary compare, so one missing id
    reported two disagreements on top of the null it already was."""
    rows = _idle(source_window(1, 10, 5, day="2026-03-02"), 1)
    ids = ["W"] * len(rows)
    ids[2] = None
    v = _verdicts(check_inputs.check_feed(_feed_with_ids(tmp_path, rows, ids, "null_mid.parquet"), cfg).rows)
    assert v["episode_id never null (the feed's copy of the producers' id)"] == "FAIL"
    assert v["the producers' ids group the same windows EPISODE_RULE derives"] == "PASS"


def test_the_response_is_checked_on_its_shape_and_against_its_snapshot(cfg, tmp_path):
    """What engineering runs on an hour that looks wrong: the response's
    own shape (priced xor rejected, the id spelt from its row, a percent
    on the tier grid), and against the snapshot it answered (one row per
    shelf of the priced hour, the price the percent makes, a price
    shallower than the one in force a WARN -- an entry, or a defect)."""
    snap = pd.DataFrame([
        {"episode_id": "a", "date": "2026-08-29", "hour": 11, "skuseq": 1, "fc": "F1",
         "inventory": 5.0, "discount": 10.0, "normal_asp": 1000.0, "cogs_wo_vat": 400.0,
         "flc_window": 4.0, "category": "MEAT", "subcategory": "BEEF"},
        {"episode_id": "b", "date": "2026-08-29", "hour": 11, "skuseq": 2, "fc": "F1",
         "inventory": 0.0, "discount": 20.0, "normal_asp": 500.0, "cogs_wo_vat": 100.0,
         "flc_window": 4.0, "category": "MEAT", "subcategory": "BEEF"},
    ])
    good = pd.DataFrame([
        {"skuseq": 1, "fc": "F1", "date": "2026-08-29", "hour": 11, "episode_id": "a",
         "decision_id": "dec-1|F1|2026-08-29T11", "apply_discount_pct": 15.0,
         "apply_price": 850.0, "is_exploration": False, "rejected": None},
        {"skuseq": 2, "fc": "F1", "date": "2026-08-29", "hour": 11, "episode_id": "b",
         "decision_id": None, "apply_discount_pct": None, "apply_price": None,
         "is_exploration": None, "rejected": "empty shelf: nothing to price"},
    ])
    s = tmp_path / "snap.csv"
    snap.to_csv(s, index=False)
    r = tmp_path / "good.csv"
    good.to_csv(r, index=False)
    rows = check_inputs.check_response(str(r), cfg, snapshot=str(s)).rows
    assert all(x["verdict"] == "PASS" for x in rows), [x for x in rows if x["verdict"] != "PASS"]
    assert any(x["check"].startswith("rejected rows: empty shelf") and x["count"] == 1 for x in rows)

    bad = good.copy()
    bad.loc[0, "apply_discount_pct"] = 0.15             # a fraction, off the grid too
    bad.loc[0, "decision_id"] = "dec-9|F1|2026-08-29T11"
    bad.loc[1, "decision_id"] = "dec-2|F1|2026-08-29T11"   # rejected AND priced
    bad.to_csv(r, index=False)
    by = {x["check"]: x for x in check_inputs.check_response(str(r), cfg, snapshot=str(s)).rows}
    assert by["every row is priced or rejected, never both or neither"]["verdict"] == "FAIL"
    assert by["decision_id is dec-<skuseq>|<fc>|<date>T<hh> of its own row"]["verdict"] == "FAIL"
    assert by["apply_discount_pct sits on the 2.5-point tier grid"]["verdict"] == "FAIL"
    assert by["apply_price = normal_asp x (1 - apply_discount_pct / 100)"]["verdict"] == "FAIL"

    shallow = good.copy()
    shallow.loc[0, "apply_discount_pct"] = 5.0            # shallower than the 10 in force
    shallow.loc[0, "apply_price"] = 950.0
    shallow.to_csv(r, index=False)
    by = {x["check"]: x for x in check_inputs.check_response(str(r), cfg, snapshot=str(s)).rows}
    assert by["no price shallower than the one in force (the snapshot's discount)"]["verdict"] == "WARN"
    assert check_inputs.main(["--response", str(r), "--snapshot", str(s),
                              "--config", "config.yaml"]) == 0      # a WARN is not a FAIL
