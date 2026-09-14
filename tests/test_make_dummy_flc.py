"""tools.make_dummy_flc: the fixture generator covers the configured splits."""

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
