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


def test_the_generator_emits_both_source_inventory_conventions():
    """A fixture missing one of these does not FAIL. It passes, quietly.

    Two conventions carry the whole of episode-level DQ, and each is read by
    code that has no other trigger:

      write-off sentinel   `ending == 0` while stock remained. The ONLY test
                           for closure.
      shrink               `0 < ending < starting - sold`. The reason scrap is
                           leftover PLUS shrink rather than just leftover.

    `data/flc_synth.parquet` was once generated before the write-off block
    existed and carried ZERO sentinel rows. Nothing failed -- `classify_last`
    had a fallback that read the absence as "every episode closed" -- so the
    closure path went unexercised for months while the suite stayed green.
    The fallback is gone, but the fixture could still drift the other way, and
    shrink has no fallback at all to make its absence visible.

    So the generator is asserted to produce both, on its own output, here.
    """
    import numpy as np
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
    """A negative counter is a property of a window, not of one row.

    It used to be written onto random rows, and `assign_episode_ids`
    differences the counter hour to hour -- so one bad value read as a window
    BOUNDARY and shredded a clean window into fragments (3, 2, [-1], 0). The
    fragments that did not carry the closing row came out `not_closed`, which
    manufactured 62 of the fixture's 65 unclosed episodes and drove
    `share_of_unclosed_explained_by_edge` to 0: the diagnostic meant to say
    "the extract boundary explains these" was answering an injection artifact.

    The real pattern is an episode ENTERING already negative and staying an
    episode, so the counter must still decrement by exactly one per hour.
    """
    import pandas as pd
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
