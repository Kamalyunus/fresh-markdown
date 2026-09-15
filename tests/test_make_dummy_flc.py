"""tools.make_dummy_flc: the fixture generator covers the configured splits."""

import pandas as pd

from conftest import CFG


def test_the_fixture_generator_covers_the_configured_splits():
    """`ops.bootstrap_loop --input <fixture>` must run end to end from a clean
    checkout. It could not: the generator started at a hardcoded date for 90
    days, the exclusion window removed the tail, and the data ended in April
    while config's calib window began in July -- so `fit_dispersion` died with
    "calibration window contains no rows" and the prior's held-out comparison
    came back empty, both silently about the cause."""
    import datetime as dt
    from tools.make_dummy_flc import span_covering_splits

    split = CFG["data"]["split"]
    start, days = span_covering_splits(CFG)
    assert start == dt.date.fromisoformat(str(split["train_start"]))
    assert start + dt.timedelta(days=days - 1) >= \
        dt.date.fromisoformat(str(split["test_end"])), \
        "the generated span must reach test_end, or the gate window is empty"

    # and it must still run standalone, with no config to read
    fallback_start, fallback_days = span_covering_splits({})
    assert fallback_days > 0 and fallback_start.year == 2026


def test_the_generator_injects_a_null_counter_outside_negative_windows():
    """The null-counter dirt exercises prepare_data's whole-run drop; it
    lands on single rows, never inside a negative window (that dirt's
    countdown must stay intact) and never on the clean fixture."""
    from tools.make_dummy_flc import generate

    df, _ = generate(n_skus=60, n_days=60, policy="randomized", seed=5, dirty_frac=0.02)
    nulls = df[df.flc_window.isna()]
    assert len(nulls) > 0
    neg_windows = set(map(tuple, df[df.flc_window < 0][["skuseq", "fc", "date"]]
                          .drop_duplicates().itertuples(index=False)))
    assert not any((r.skuseq, r.fc, r.date) in neg_windows for r in nulls.itertuples())
    clean, _ = generate(n_skus=20, n_days=20, policy="randomized", seed=5, dirty_frac=0.0)
    assert not clean.flc_window.isna().any()


def test_the_generator_extends_some_windows_by_a_restock():
    """EPISODE_RULE's restock clause needs the source pattern in the
    fixture: stock arrives on one hour (ending > starting - sold), the
    counter steps UP on the NEXT hour, the chain stays continuous."""
    from tools.make_dummy_flc import generate, restock_extended_windows

    df, _ = generate(n_skus=60, n_days=60, policy="randomized", seed=5, dirty_frac=0.0)
    assert restock_extended_windows(df) > 0
    # inside one window (two windows of one SKU x FC x day can abut or
    # overlap in the raw fixture; the chain drops overlaps, a close opens
    # the next), an up-step follows a restocked hour and nothing else, and
    # the shelf carries the arrival forward (continuity)
    g = df[~df.duplicated(["skuseq", "fc", "date", "hour"], keep=False)]
    g = g.sort_values(["skuseq", "fc", "date", "hour"])
    prev = g.groupby(["skuseq", "fc", "date"]).shift()
    inside = (g.hour - prev.hour == 1) & (prev.ending_inventory != 0)
    up = inside & ((g.flc_window - prev.flc_window) > -1)
    assert up.any()
    restocked = prev.ending_inventory > prev.inventory - prev.units_sold
    assert restocked[up].all()
    assert (g.inventory[up] == prev.ending_inventory[up]).all()
    assert not generate(n_skus=20, n_days=20, policy="randomized", seed=5,
                        dirty_frac=0.0, restock_extend_rate=0.0)[0].pipe(
        restock_extended_windows)


def test_the_generator_crosses_midnight_and_leaves_the_final_counter_positive():
    """The fixture never crossed midnight and 86% of its final rows carried
    counter 0 -- the opposite of production, where windows straddle the
    seam routinely and the counter is still positive on essentially every
    final row. Every 12a seam path (opening-date cuts, the entry-row sort,
    the week schedule) ran on data where the seam could not occur."""
    from tools.make_dummy_flc import (cross_midnight_windows, generate,
                                      final_counter_positive_share, window_opens)

    df, _ = generate(n_skus=60, n_days=60, policy="randomized", seed=5, dirty_frac=0.0)
    assert cross_midnight_windows(df) > 0
    # a crossing window is ONE window: hour 23 -> 0 with the date rolling,
    # the chain continuous across the seam
    g = df.sort_values(["skuseq", "fc", "date", "hour"])
    prev = g.groupby(["skuseq", "fc"]).shift()
    seam = (g.hour == 0) & (prev.hour == 23) & \
        ((pd.to_datetime(g.date) - pd.to_datetime(prev.date)).dt.days == 1)
    assert seam.any()
    assert (g.inventory[seam] == prev.ending_inventory[seam]).all()
    assert (g.flc_window[seam] == prev.flc_window[seam] - 1).all()
    assert not window_opens(df)[seam.reindex(df.index)].any()
    # most windows close with hours left on the counter, some run it down
    share = final_counter_positive_share(df)
    assert 0.3 < share < 0.95
    # both are knobs: off, the fixture is the old one
    flat, _ = generate(n_skus=60, n_days=60, policy="randomized", seed=5,
                       dirty_frac=0.0, cross_midnight_rate=0.0, early_close_rate=0.0)
    assert cross_midnight_windows(flat) == 0
    assert final_counter_positive_share(flat) < share
    # and the cross-midnight windows still pass the id rule as one episode
    from common.windows import assign_episode_ids
    from fit.prepare_data import SOURCE_TO_CANONICAL
    d = df.rename(columns=SOURCE_TO_CANONICAL).sort_values(["sku_id", "fc", "date", "hour_of_day"])
    ids = assign_episode_ids(d)
    assert d.groupby(ids).date.nunique().gt(1).sum() == cross_midnight_windows(df)
    # the tool reads its windows through the chain's rule and spells no
    # clock or counter step of its own: its window reading once omitted the
    # counter clause, and the printed counts and the chain's ids disagreed
    import inspect
    from tools import make_dummy_flc as gen
    src = inspect.getsource(gen)
    assert "assign_episode_ids(" in src and "window_starts(" in src
    assert ".dt.total_seconds()" not in src and "prev.flc_window" not in src


def test_an_extension_hour_arrives_more_than_it_shrinks():
    """An extension hour whose shrink equalled the arrival reconciled
    exactly, so the counter's up-step next hour read as a reset (a new
    window) and the restock clause went unexercised on that window."""
    from tools.make_dummy_flc import generate, restock_extended_windows

    df, _ = generate(n_skus=200, n_days=40, policy="randomized", seed=11,
                     dirty_frac=0.0, shrink_rate=0.5, restock_extend_rate=1.0)
    # two windows of one sku x fc can overlap in the raw fixture (the chain
    # drops the collision); read the rest
    g = df[~df.duplicated(["skuseq", "fc", "date", "hour"], keep=False)]
    g = g.sort_values(["skuseq", "fc", "date", "hour"])
    prev = g.groupby(["skuseq", "fc"]).shift()
    ts = pd.to_datetime(g.date) + pd.to_timedelta(g.hour, unit="h")
    one_hour = ts.groupby([g.skuseq, g.fc]).diff().dt.total_seconds().eq(3600)
    up = one_hour & ((g.flc_window - prev.flc_window) > -1) & (prev.ending_inventory != 0)
    assert up.any()
    # every up-step follows a row where stock ARRIVED (net of any shrink)
    assert (prev.ending_inventory[up] > prev.inventory[up] - prev.units_sold[up]).all()
    assert restock_extended_windows(df) > 0
