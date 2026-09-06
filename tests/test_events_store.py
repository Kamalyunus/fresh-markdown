"""events.store: what the append-only log admits, refuses, and survives."""

import json

import numpy as np
import pytest

from conftest import decision_event, outcome_event
from events.store import EventStore
from daily import update as upd


def _store(cfg, tmp_path):
    return EventStore(cfg, root=str(tmp_path / "events"))


def _outcome(**over):
    """A RECONCILING outcome (2 - 1 = 1), so only the field under test can
    quarantine it."""
    return outcome_event(**{"ending_inventory": 1, **over})


# ------------------------------------------------------------------ duplicates
def test_duplicate_ids_written_by_a_foreign_producer_are_counted_on_load(cfg, tmp_path):
    """`duplicate_counts` only moved on emit, so a producer writing the JSONL
    directly could duplicate every id and the duplicate gate (update's and
    monitor's) could never fire."""
    root = tmp_path / "events"
    root.mkdir()
    d = decision_event()
    with open(root / "decisions.jsonl", "w") as f:
        for _ in range(3):
            f.write(json.dumps(d) + "\n")
    o = _outcome()
    with open(root / "outcomes.jsonl", "w") as f:
        f.write(json.dumps(o) + "\n")
        f.write(json.dumps(o) + "\n")

    store = EventStore(cfg, root=str(root))
    assert store.duplicate_counts == {"decision": 2, "outcome": 1}
    # counted, and loaded ONCE: a duplicate line must never reach the
    # learner as a second outcome or the matcher as a second decision
    assert len(store.load_decisions()) == 1 and len(store.load_outcomes()) == 1
    # and an emit of the same id counts on top
    assert not store.emit_outcome(dict(o))
    assert store.duplicate_counts["outcome"] == 2

    # the gate reads it: 4 duplicates over 2 outcome rows is far past 1%
    from engine.posterior import PosteriorStore
    posterior = PosteriorStore.initialise(
        cfg, {"vegetables": {"mean": -1.0, "std": 0.6}}, {"vegetables": 10**6},
        path=str(tmp_path / "posterior.json"))
    _, gates, _, _ = upd.collect_batch(store, posterior, cfg)
    assert gates["duplicate_or_unmatched_rate"]["value"] > 0
    assert not gates["duplicate_or_unmatched_rate"]["pass"]


# ------------------------------------------------------------------- the date
@pytest.mark.parametrize("bad", [
    "2026-8-19", "20260819", "2026-08-19T17:00:00", "2026-13-01", "2026-02-30",
    20260819, None, "yesterday",
])
def test_a_decision_whose_date_is_not_an_iso_day_is_quarantined(cfg, tmp_path, bad):
    """`date` is the key ingest matches feed rows on and every daily series
    (tau walk, guardrail, spend) buckets by; one other spelling silently
    splits a day in two."""
    store = _store(cfg, tmp_path)
    assert not store.emit_decision(decision_event(date=bad))
    assert store.quarantined_this_run == 1
    q = store.load_quarantine()
    assert len(q) == 1 and "date" in q[0]["problems"][0]
    assert store.load_decisions() == []


def test_an_iso_day_is_accepted(cfg, tmp_path):
    store = _store(cfg, tmp_path)
    assert store.emit_decision(decision_event(date="2026-08-19"))
    assert store.quarantined_this_run == 0


# ---------------------------------------------------------------- applied_price
@pytest.mark.parametrize("price, ok", [
    (7000.0, True), (7000, True), (np.float64(7000.0), True),
    (np.int64(7000), True), (np.float32(7000.0), True),
    (float("nan"), False), (float("inf"), False), (-float("inf"), False),
    (np.float64("nan"), False), (True, False), (np.bool_(True), False),
    ("7000", False), (None, False),
])
def test_applied_price_must_be_a_finite_number(cfg, tmp_path, price, ok):
    """inf passed the old `price != price` test, bool passed as an int, and
    numpy floats did not count as numbers -- pandas producers quarantined
    in bulk."""
    store = _store(cfg, tmp_path)
    assert store.emit_outcome(_outcome(applied_price=price)) is ok, price
    if not ok:
        assert "applied_price" in store.load_quarantine()[0]["problems"][0]


# --------------------------------------------------------------- torn last line
def test_a_torn_last_line_is_quarantined_and_the_store_stays_readable(cfg, tmp_path):
    """A power loss mid-append leaves half a JSON object as the last line.
    That used to make the store unconstructable (json.loads on load); now the
    line is quarantined with the reason, the next append lands on a fresh
    line, and the good events before it still load."""
    root = tmp_path / "events"
    root.mkdir()
    good = decision_event(decision_id="D-good")
    torn = json.dumps(decision_event(decision_id="D-torn"))[:40]
    with open(root / "decisions.jsonl", "w") as f:
        f.write(json.dumps(good) + "\n")
        f.write(torn)                                  # no newline: torn

    store = EventStore(cfg, root=str(root))
    assert [d["decision_id"] for d in store.load_decisions()] == ["D-good"]
    assert store.quarantined_this_run == 1
    q = store.load_quarantine()
    assert len(q) == 1
    assert "unparseable" in q[0]["problems"][0] and "line 2" in q[0]["problems"][0]
    assert q[0]["event"]["raw_line"] == torn

    # the next event lands on its own line and reads back
    assert store.emit_decision(decision_event(decision_id="D-next"))
    assert [d["decision_id"] for d in store.load_decisions()] == ["D-good", "D-next"]

    # a fresh store over the same files does not quarantine the same line twice
    again = EventStore(cfg, root=str(root))
    assert again.quarantined_this_run == 0
    assert len(again.load_quarantine()) == 1
    assert [d["decision_id"] for d in again.load_decisions()] == ["D-good", "D-next"]


def test_a_torn_quarantine_line_does_not_take_the_store_down(cfg, tmp_path):
    root = tmp_path / "events"
    root.mkdir()
    (root / "quarantine.jsonl").write_text('{"event": {"outcome_id": "x"}, "prob')
    store = EventStore(cfg, root=str(root))
    assert store.load_quarantine() == []
    assert store.emit_outcome(_outcome())
    # the torn line was closed, so the next quarantined record is its own
    # line and reads back -- not glued onto the fragment and lost with it
    assert not store.emit_outcome(_outcome(outcome_id="O-bad", ending_inventory=99))
    q = store.load_quarantine()
    assert [r["event"].get("outcome_id") for r in q] == ["O-bad"]
