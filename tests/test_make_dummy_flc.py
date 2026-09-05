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
