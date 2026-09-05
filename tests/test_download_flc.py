"""The extract query is a contract with step 1."""

import re

import pyarrow as pa
import pytest

from bootstrap import download_flc
from bootstrap.prepare_data import SOURCE_TO_CANONICAL
from tools.make_dummy_flc import SCHEMA


def selected_columns(sql):
    """Output column names of the SELECT list, alias where aliased."""
    body = sql.split("SELECT", 1)[1].split("FROM", 1)[0]
    names = []
    for item in body.split(","):
        item = " ".join(item.split())
        if not item:
            continue
        m = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)$", item)
        names.append(m.group(1) if m else item)
    return names


QUERY = download_flc.build_query("2026-03-01", "2026-08-03")


def test_query_emits_exactly_the_declared_columns():
    assert selected_columns(QUERY) == list(download_flc.REQUIRED_COLUMNS)


def test_declared_columns_cover_the_step_1_rename():
    assert set(SOURCE_TO_CANONICAL) <= set(download_flc.REQUIRED_COLUMNS)


def test_declared_columns_match_the_source_schema():
    # make_dummy_flc emits the source schema; the extract must be the same
    # frame, or the synthetic end-to-end tests are exercising a shape
    # production never produces.
    assert set(f.name for f in SCHEMA) == set(download_flc.REQUIRED_COLUMNS)


def test_the_generator_emits_both_source_inventory_conventions():
    """A fixture missing one of these does not FAIL. It passes, quietly."""
    from tools.make_dummy_flc import generate

    df, _ = generate(n_skus=40, n_days=60, policy="randomized", seed=11,
                     dirty_frac=0.0)
    net = df.inventory - df.units_sold

    assert ((df.ending_inventory == 0) & (net > 0)).sum() > 0, \
        "no write-off sentinel: every episode would read NOT_CLOSED"
    assert ((df.ending_inventory > 0) & (df.ending_inventory < net)).sum() > 0, \
        "no shrink: the leftover-PLUS-shrink half of scrap is untested"

    # and shrink must stay INTERIOR -- a zero ending is the close, not a
    # shrink, and `hour_status` would rightly call it a write-off
    assert (df.ending_inventory >= 0).all()

    # turning it off is a supported mode, and the only way to isolate the
    # leftover half of the identity
    off, _ = generate(n_skus=40, n_days=60, policy="randomized", seed=11,
                      dirty_frac=0.0, shrink_rate=0.0)
    net_off = off.inventory - off.units_sold
    assert ((off.ending_inventory > 0) & (off.ending_inventory < net_off)).sum() == 0


def test_the_negative_window_dirt_lands_on_WHOLE_windows(): # noqa: N802
    """A negative counter is a property of a window, not of one row."""
    from tools.make_dummy_flc import generate

    df, _ = generate(n_skus=60, n_days=60, policy="randomized", seed=5,
                     dirty_frac=0.02)
    neg = df[df.flc_window < 0]
    assert len(neg) > 0, "no negative-window dirt was injected at all"

    for key, g in neg.groupby(["skuseq", "fc", "date"]):
        g = g.sort_values("hour")
        # EVERY row of the window is negative -- it entered that way
        assert (g.flc_window < 0).all(), f"{key} is negative only part-way in"
        # and the countdown is intact, so segmentation keeps the window whole
        steps = set(g.flc_window.diff().dropna().astype(int))
        assert steps <= {-1}, f"{key} does not decrement by one: {steps}"

    # the payoff: unclosed episodes are no longer an artifact of the injection
    from bootstrap.prepare_data import load_and_filter
    from common import episodes as E
    from common.config import load_config
    import pyarrow as pa, pyarrow.parquet as pq, tempfile, os

    tmp = os.path.join(tempfile.mkdtemp(), "f.parquet")
    pq.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False), tmp)
    d, _ = load_and_filter(tmp, load_config())
    flow = E.episode_flow(d)
    assert (~flow.closed).mean() < 0.02, (
        "unclosed episodes are being manufactured by the dirt injection again: "
        f"{(~flow.closed).mean():.1%}")


def test_exclusion_window_comes_from_config():
    from common.config import load_config

    excl = load_config()["data"]["exclusion_window"]
    sql = download_flc.build_query("2026-03-01", "2026-08-03",
                                   excl["start"], excl["end"])
    assert f"NOT (date BETWEEN '{excl['start']}' AND '{excl['end']}')" in sql
    assert "NOT (date BETWEEN" not in QUERY  # omitted when not asked for


def test_dates_are_validated_before_interpolation():
    # The only defence against a value reaching SQL as text: it has to parse
    # as an ISO date first.
    with pytest.raises(SystemExit):
        download_flc.build_query("2026-03-01'; DROP TABLE x --", "2026-08-03")
    with pytest.raises(SystemExit):
        download_flc.build_query("2026-03-01", "not-a-date")


def test_no_credentials_in_the_source():
    src = open(download_flc.__file__).read()
    assert "REDSHIFT_PASSWORD" in src          # read from the environment
    assert not re.search(r"password\s*=\s*['\"][^'\"]", src)
    assert not re.search(r"\.redshift\.amazonaws\.com", src)


def test_summarise_reports_the_null_shares_that_drop_episodes():
    import pandas as pd

    df = pd.DataFrame({
        "date": ["2026-03-01"] * 2, "hour": [0, 1], "skuseq": [1, 1],
        "fc": ["A", "A"], "inventory": [5.0, 4.0], "units_sold": [1, 0],
        "ending_inventory": [4.0, 0.0], "discount": [0.0, 10.0],
        "normal_asp": [1000.0, 1000.0], "final_price": [1000.0, 0.0],
        "cogs_wo_vat": [600.0, None], "flc_window": [2.0, 1.0],
        "category": ["FRUIT", "FRUIT"], "subcategory": ["APPLE", "APPLE"],
    })
    out = download_flc.summarise(df)
    assert "rows        2" in out
    assert "null cogs_wo_vat 50.00%" in out


def test_schema_is_pyarrow_typed():
    # guards the import above against a refactor that turns SCHEMA into a list
    assert isinstance(SCHEMA, pa.Schema)


# ------------------------------------------------- the pull covers the config

def test_a_pull_that_stops_short_of_the_configured_windows_is_refused():
    """`--days 120` from yesterday is the manual default, and it silently
    left the calib window (or the hold-out) empty: fit_dispersion then died
    about 'no rows' three steps later. The requested range is checked against
    train_start .. holdout.end (or test_end), the range pipeline.advance
    passes, and the exit is non-zero with the dates named."""
    from datetime import date
    from common.config import load_config

    cfg = load_config()
    need_start, need_end = download_flc.required_range(cfg)
    assert need_start == date.fromisoformat(cfg["data"]["split"]["train_start"])
    assert need_end == date.fromisoformat(cfg["data"]["holdout"]["end"])

    # exactly the config's range, and a superset of it, both cover
    assert download_flc.coverage_gap(need_start, need_end, cfg) is None
    assert download_flc.coverage_gap(need_start.replace(day=1),
                                     need_end.replace(year=need_end.year + 1),
                                     cfg) is None

    # short at either end is refused, and the message names both ranges
    from datetime import timedelta
    late = download_flc.coverage_gap(need_start + timedelta(days=1), need_end, cfg)
    early = download_flc.coverage_gap(need_start, need_end - timedelta(days=1), cfg)
    for gap in (late, early):
        assert gap and str(need_start) in gap and str(need_end) in gap
        assert "train_start" in gap and "holdout" in gap

    # without a hold-out the requirement falls back to split.test_end
    no_holdout = dict(cfg, data={**cfg["data"], "holdout": None})
    assert download_flc.required_range(no_holdout)[1] == \
        date.fromisoformat(cfg["data"]["split"]["test_end"])


def test_main_exits_non_zero_when_the_range_does_not_cover(monkeypatch, tmp_path):
    """The check is wired into the CLI: the extract is still written (it is
    data), but the exit code says the chain cannot run on it."""
    import sys
    import pandas as pd
    from common.config import load_config

    cfg = load_config()
    train_start = cfg["data"]["split"]["train_start"]

    class _Conn:
        def close(self):
            pass

    frame = pd.DataFrame({c: [1] for c in download_flc.REQUIRED_COLUMNS})
    frame["date"] = train_start
    monkeypatch.setattr(download_flc, "get_conn", lambda env_file=None: _Conn())
    monkeypatch.setattr(pd, "read_sql", lambda q, conn: frame)
    out = str(tmp_path / "flc_raw.parquet")

    # covering range: exits cleanly
    end = cfg["data"]["holdout"]["end"]
    monkeypatch.setattr(sys, "argv", ["download_flc", "--start-date", train_start,
                                      "--end-date", end, "--out", out])
    download_flc.main()
    assert (tmp_path / "flc_raw.parquet").exists()

    # ten days: refused, after the file is written
    monkeypatch.setattr(sys, "argv", ["download_flc", "--start-date", train_start,
                                      "--days", "10", "--out", out,
                                      "--end-date", train_start])
    with pytest.raises(SystemExit) as exc:
        download_flc.main()
    assert "EXTRACT TOO SHORT" in str(exc.value)
    assert end in str(exc.value)
