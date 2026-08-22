"""The extract query is a contract with step 1.

`bootstrap.prepare_data` renames source columns exactly once, at load. If the
query stops emitting one of them -- a dropped alias, a renamed source column
-- nothing notices until `read_parquet(...).rename(...)` produces a frame with
a missing column, minutes and one network round-trip into the run. These tests
compare the query's own SELECT list against the two places that define the
source schema, so the break shows up here instead.
"""

import re

import pyarrow as pa
import pytest

from bootstrap import download_flc
from bootstrap.prepare_data import SOURCE_TO_PRD
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
    assert set(SOURCE_TO_PRD) <= set(download_flc.REQUIRED_COLUMNS)


def test_declared_columns_match_the_source_schema():
    # make_dummy_flc emits the source schema; the extract must be the same
    # frame, or the synthetic end-to-end tests are exercising a shape
    # production never produces.
    assert set(f.name for f in SCHEMA) == set(download_flc.REQUIRED_COLUMNS)


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
