"""daily.features -- the morning's feature table: the rolling feed history
it keeps (seeded from the extract, fed a day at a time, deduplicated,
trimmed), the table it writes for the day, and the table's equality with
the per-batch computation it replaces."""
import os

import pandas as pd
import pytest

from conftest import source_window, write_extract
from daily import features
from engine.state import (HISTORY_COLS, POOLED_FC, load_history, ref_rate_features,
                          ref_rate_table)


def _cfg(cfg, tmp_path):
    import copy
    c = copy.deepcopy(cfg)
    c["features"] = {"table_dir": str(tmp_path / "features"),
                     "history_path": str(tmp_path / "data" / "feed_history.parquet"),
                     "history_days": 10}
    return c


def _days(sku, start_day, n_days, fc="F1", hour=10, n=4):
    rows = []
    for k in range(n_days):
        day = str((pd.Timestamp(start_day) + pd.Timedelta(days=k)).date())
        rows += source_window(sku, hour, n, day=day, fc=fc, discount=25.0)
    return rows


def test_the_rolling_history_is_seeded_fed_deduplicated_and_trimmed(cfg, tmp_path, monkeypatch):
    """The first morning seeds from the raw extract; each morning's feed
    joins it; a day fed twice keeps one row per hour; days beyond
    features.history_days leave."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    c = _cfg(cfg, tmp_path)
    write_extract(tmp_path / "data", _days(1, "2026-03-01", 12), name="flc_raw.parquet")
    feed = write_extract(tmp_path, _days(1, "2026-03-13", 1), name="feed.parquet")
    raw = features.rolling_history(c, str(feed))
    days = sorted(pd.to_datetime(raw.date).dt.date.astype(str).unique())
    assert days[-1] == "2026-03-13" and len(days) == 10      # trimmed to history_days
    assert os.path.exists(c["features"]["history_path"])
    again = features.rolling_history(c, str(feed))            # the same day fed twice
    assert len(again) == len(raw)
    assert not again.duplicated(subset=list(features.SOURCE_HOUR_KEY)).any()


def test_the_days_table_equals_the_per_batch_computation(cfg, tmp_path, monkeypatch):
    """One row per (sku, fc) plus the SKU's pooled row, each the number
    ref_rate_features gives an opening of that sku on that day."""
    monkeypatch.chdir(tmp_path)
    c = _cfg(cfg, tmp_path)
    rows = _days(1, "2026-03-01", 8) + _days(2, "2026-03-01", 8, fc="F2") \
        + _days(2, "2026-03-03", 5, fc="F1")
    feed = write_extract(tmp_path, rows, name="feed.parquet")
    rep = features.build(c, feed_path=str(feed), as_of="2026-03-09")
    table = pd.read_parquet(rep["out"])
    assert rep["as_of"] == "2026-03-09" and rep["pooled_rows"] == 2
    # with no --as-of the day is the FEED's plus one, never the host clock:
    # a cron across a midnight in another zone read "today" one day off
    assert features.build(c, feed_path=str(feed))["as_of"] == "2026-03-09"
    assert set(zip(table.sku_id, table.fc)) == {("1", "F1"), ("2", "F1"), ("2", "F2"),
                                                ("1", POOLED_FC), ("2", POOLED_FC)}
    hist = load_history(c["features"]["history_path"], c)
    assert list(hist.columns) == list(HISTORY_COLS)
    stub = pd.DataFrame([
        {"episode_id": "o1", "sku_id": "1", "fc": "F1", "category": "MEAT",
         "date": "2026-03-09", "hour_of_day": 10, "starting_inventory": 5},
        {"episode_id": "o2", "sku_id": "1", "fc": "F9", "category": "MEAT",    # unseen fc
         "date": "2026-03-09", "hour_of_day": 10, "starting_inventory": 5}])
    want = ref_rate_features(hist, stub, c)
    row = table[(table.sku_id == "1") & (table.fc == "F1")].iloc[0]
    assert row.sku_ref_sales_rate_30d == pytest.approx(want["o1"][0])
    assert row.prior_episode_ref_sales_rate == pytest.approx(want["o1"][1])
    pooled = table[(table.sku_id == "1") & (table.fc == POOLED_FC)].iloc[0]
    assert pooled.sku_ref_sales_rate_30d == pytest.approx(want["o2"][0])
    assert pd.isna(pooled.prior_episode_ref_sales_rate) and pd.isna(want["o2"][1])
    # and a table straight from the frame agrees with the one on disk
    direct = ref_rate_table(hist, "2026-03-09", c)
    pd.testing.assert_frame_equal(direct.reset_index(drop=True), table.reset_index(drop=True),
                                  check_dtype=False)
