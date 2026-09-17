"""ops.assign_episode_ids -- the episode-id rule as the data producers run
it: one hour's rows against last hour's, nothing else."""
import pandas as pd

from ops import assign_episode_ids as aid


def _row(hour, **over):
    r = {"date": "2026-08-19", "hour": hour, "skuseq": 7, "fc": "F1", "inventory": 3.0,
         "discount": 15.0, "units_sold": 1, "normal_asp": 10000.0, "final_price": 8500.0,
         "cogs_wo_vat": 4000.0, "ending_inventory": 2.0, "flc_window": 3.0,
         "category": "VEG", "subcategory": "LEAFY", "episode_id": "7|F1|2026-08-19T17"}
    r.update(over)
    return r


def test_the_rule_one_case_at_a_time():
    prev = _row(17, flc_window=3.0)
    assert aid.continues(prev, _row(18, flc_window=2.0))                 # counter stepped down
    assert not aid.continues(prev, _row(19, flc_window=1.0))             # two hours apart
    assert not aid.continues(prev, _row(18, flc_window=0.0))             # stepped down by two
    assert not aid.continues(_row(17, ending_inventory=0.0), _row(18, flc_window=2.0))  # write-off
    # counter up or flat: stock arrived (ending > inventory - sold) continues, else new
    assert aid.continues(_row(17, units_sold=1, ending_inventory=6.0), _row(18, flc_window=5.0))
    assert aid.continues(_row(17, units_sold=1, ending_inventory=6.0), _row(18, flc_window=3.0))
    assert not aid.continues(_row(17, units_sold=1, ending_inventory=2.0), _row(18, flc_window=5.0))
    assert not aid.continues(None, _row(18))                             # a first hour
    assert not aid.continues(_row(17, flc_window=None), _row(18))        # nothing to step from
    assert aid.new_id(_row(18)) == "7|F1|2026-08-19T18"


def test_assign_carries_the_id_or_opens_a_new_one_and_counts():
    first, counts = aid.assign([_row(17, skuseq=s, episode_id=None) for s in (7, 8, 9)])
    assert counts == {"continued": 0, "new": 3, "unkeyable": 0}
    assert [r["episode_id"] for r in first] == [f"{s}|F1|2026-08-19T17" for s in (7, 8, 9)]
    by = {r["skuseq"]: r for r in first}
    closed = [dict(by[7], units_sold=3, ending_inventory=0.0),
              dict(by[8], units_sold=1, ending_inventory=6.0),
              dict(by[9], units_sold=1, ending_inventory=2.0)]
    now, counts = aid.assign([_row(18, skuseq=7, flc_window=2.0), _row(18, skuseq=8, flc_window=5.0),
                              _row(18, skuseq=9, flc_window=5.0), _row(18, skuseq=None)], closed)
    assert counts == {"continued": 1, "new": 2, "unkeyable": 1}
    assert [r["episode_id"] for r in now] == ["7|F1|2026-08-19T18", "8|F1|2026-08-19T17",
                                             "9|F1|2026-08-19T18", None]


def test_the_cli_adds_only_the_episode_id_column(tmp_path):
    prev, this = tmp_path / "17.csv", tmp_path / "18.csv"
    out17, out18 = tmp_path / "17_ids.csv", tmp_path / "18_ids.csv"
    cols = [c for c in _row(17) if c != "episode_id"]
    pd.DataFrame([_row(17)])[cols].to_csv(prev, index=False)
    pd.DataFrame([_row(18, flc_window=2.0)])[cols].to_csv(this, index=False)
    assert aid.main(["--hour", str(prev), "--out", str(out17)]) == 0
    assert aid.main(["--hour", str(this), "--previous", str(out17), "--out", str(out18)]) == 0
    got = pd.read_csv(out18)
    assert list(got.columns) == cols + ["episode_id"]
    assert got.episode_id.iloc[0] == "7|F1|2026-08-19T17"


def test_the_script_and_the_chain_draw_the_same_window_boundaries_over_a_whole_day(synth_flc):
    """The one parity check: hour by hour over a synthetic day, the
    producers' script (this hour's rows against last hour's) must open a
    window exactly where the chain's vector rule (common.windows,
    over the whole frame) opens one. The close test is `ending == 0`, the
    write-off sentinel, on both sides; the counter step is the raw
    difference on both sides. A gap here is a disagreement the checker
    would report that the producers could never avoid."""
    from common import windows
    from fit.prepare_data import SOURCE_TO_CANONICAL
    raw = pd.read_parquet(synth_flc)
    raw["date"] = raw["date"].astype(str)
    skus = sorted(raw.skuseq.unique())[:80]
    day = raw[raw.skuseq.isin(skus)].sort_values(["skuseq", "fc", "date", "hour"])
    day = day[day.date.isin(sorted(day.date.unique())[:10])].reset_index(drop=True)
    assert len(day) > 100 and windows.window_starts(
        day.rename(columns=SOURCE_TO_CANONICAL)).sum() > 10     # enough windows to disagree on
    # the chain, over the frame at once
    canon = day.rename(columns=SOURCE_TO_CANONICAL)
    chain_opens = windows.window_starts(canon).to_numpy()
    # the script, one clock hour at a time, each hour against the last
    hours = sorted(set(zip(day.date, day.hour)))
    prev, ids = [], {}
    for d, h in hours:
        rows = day[(day.date == d) & (day.hour == h)].to_dict("records")
        out, _ = aid.assign(rows, prev)
        for r, o in zip(rows, out):
            ids[(r["skuseq"], r["fc"], d, h)] = o["episode_id"]
        prev = out
    script_ids = [ids[(r.skuseq, r.fc, r.date, r.hour)] for r in day.itertuples()]
    script_opens = pd.Series(script_ids).ne(
        pd.Series(script_ids).groupby([day.skuseq, day.fc]).shift()).to_numpy()
    assert script_opens.tolist() == chain_opens.tolist()


def test_only_the_write_off_zero_closes_a_shelf():
    """`ending == 0` is the contract's sentinel and the chain's test; a
    negative ending is a defect the chain carries, not a close."""
    assert aid.continues(_row(17, ending_inventory=-1.0), _row(18, flc_window=2.0))
    assert not aid.continues(_row(17, ending_inventory=0.0), _row(18, flc_window=2.0))
