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
