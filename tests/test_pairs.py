"""events.pairs: the one hour key, the ids over it, and the normalisers
every hourly script reads a cell through."""
import pandas as pd
import pytest

from events.pairs import (as_number, decision_id_of, hour_int, hour_key, outcome_id_of,
                          rejection_id_of, shelf_hour_tag)


def test_every_event_id_is_the_hours_key_computable_from_the_feed_row():
    """Two prefixes over one key, so engineering can name either before it
    exists and a pair differs only in its prefix. The EPISODE is in neither:
    it is the producers' and can be relabelled, and an audit record's
    identity may not move when an upstream label does."""
    k = hour_key(7.0, "F1", pd.Timestamp("2026-08-19"), 17)
    assert k == ("7", "F1", "2026-08-19", 17)
    assert outcome_id_of(k) == "feed-7|F1|2026-08-19T17"
    assert decision_id_of(k) == "dec-7|F1|2026-08-19T17"
    assert outcome_id_of(k).split("-", 1)[1] == decision_id_of(k).split("-", 1)[1]
    assert decision_id_of(hour_key("7", "F1", "2026-08-19", 5)) == "dec-7|F1|2026-08-19T05"
    assert outcome_id_of(hour_key("7", "F1", "2026-08-19", 5)) == "feed-7|F1|2026-08-19T05"
    assert rejection_id_of(k) == "rej-" + shelf_hour_tag(k) == "rej-7|F1|2026-08-19T17"
    with pytest.raises(ValueError):
        hour_key(7.5, "F1", "2026-08-19", 17)
    with pytest.raises(ValueError):
        hour_key(7, "F1", "2026-08-19", float("nan"))
    with pytest.raises(ValueError):
        hour_key(None, "F1", "2026-08-19", 17)



def test_an_hour_is_an_integer_and_a_cell_is_a_number_or_nothing():
    """`hour_int` is what every hand-rolled `int(float(h))` truncated past:
    17.5 names no hour. `as_number` is the one lenient reading of a feed
    cell -- a numeric string parses, a bool does not, NaN and inf are
    nothing."""
    assert hour_int(17) == hour_int(17.0) == hour_int("17") == 17
    for bad in (17.5, float("nan"), float("inf"), "x"):
        with pytest.raises(ValueError):
            hour_int(bad)
    assert as_number("3") == 3.0 and as_number(2) == 2.0 and as_number(2.5) == 2.5
    for nothing in (None, float("nan"), float("inf"), "abc", "", True, [1]):
        assert as_number(nothing) is None
