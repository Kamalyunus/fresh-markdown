"""Parallelism must change speed and nothing else.

Two properties carry that, and both were violated by the code this replaces:

  1. **Order-independence.** The serial loop drew every exploration from one
     shared generator, so the draw an episode got depended on how many
     episodes preceded it. Split that across processes and the run stops
     reproducing. Each episode now seeds from its own id.
  2. **Workers compute, the parent commits.** The shadow gate MEASURES the
     event store -- completeness, matched rate, dedup, quarantine. Per-worker
     stores merged afterwards would mean the gate no longer tests the path
     production runs.
"""

import inspect
import time

import numpy as np
import pytest

from common.parallel import map_episodes, resolve_workers


def _square(x, ctx):
    return x * x + ctx["offset"]


def test_serial_and_parallel_agree_and_keep_order():
    items = list(range(97))          # prime, so chunks divide unevenly
    ctx = {"offset": 3}
    serial = map_episodes(_square, items, ctx, workers=None)
    assert serial == [x * x + 3 for x in items]
    for n in (2, 4):
        assert map_episodes(_square, items, ctx, workers=n) == serial


def _slow_early(x, ctx):
    """Deliberately inverted cost: item 0 is the slowest, so workers finish
    roughly in reverse. If results were collected as they completed, the
    output would come back reversed."""
    time.sleep((ctx["n"] - x) * 0.004)
    return x


def test_results_come_back_in_submission_order_not_completion_order():
    """A reduction that depended on completion order would differ from the
    serial run silently, and differently on every machine."""
    items = list(range(24))
    out = map_episodes(_slow_early, items, {"n": len(items)}, workers=4,
                       chunks_per_worker=6)
    assert out == items


def test_a_single_item_never_pays_for_a_pool():
    assert map_episodes(_square, [7], {"offset": 0}, workers=8) == [49]
    assert map_episodes(_square, [], {"offset": 0}, workers=8) == []


def test_resolve_workers():
    import os
    assert resolve_workers(None) == 1          # serial unless asked
    assert resolve_workers(1) == 1
    assert resolve_workers(3) == 3
    assert resolve_workers(0) == max((os.cpu_count() or 2) - 1, 1)


# ------------------------------------------------------- per-episode seeding

def test_the_episode_generator_is_reproducible_and_order_free():
    from pipeline.shadow import _episode_seed
    a = _episode_seed(0, "sku|fc|2026-08-04T09").integers(0, 10_000, 5)
    b = _episode_seed(0, "sku|fc|2026-08-04T09").integers(0, 10_000, 5)
    assert np.array_equal(a, b), "same episode, same seed -> same draws"

    other = _episode_seed(0, "sku|fc|2026-08-05T09").integers(0, 10_000, 5)
    assert not np.array_equal(a, other), "different episodes must not collide"

    seeded = _episode_seed(1, "sku|fc|2026-08-04T09").integers(0, 10_000, 5)
    assert not np.array_equal(a, seeded), "--seed must still move the draws"


def test_the_shared_generator_is_gone_from_the_episode_loop():
    """The bug this replaces: one generator for the whole loop, so an
    episode's draw depended on how many episodes ran before it."""
    from pipeline import shadow
    src = inspect.getsource(shadow._shadow_one)
    assert "_episode_seed(" in src
    assert "default_rng(seed)" not in src


# ------------------------------------------------ workers never write events

def test_workers_buffer_events_and_the_parent_commits_them():
    from pipeline import shadow

    buf = shadow._BufferStore()
    assert buf.emit_decision({"a": 1}) is True
    assert buf.decisions == [{"a": 1}]
    assert not hasattr(buf, "emit_outcome"), \
        "a worker must not be able to commit an outcome -- the gate measures " \
        "the real store's dedup and quarantine"

    src = inspect.getsource(shadow.run_shadow)
    commit = src[src.index("for decision, outcome in out[\"events\"]"):]
    assert "store.emit_decision(decision)" in commit
    assert "store.emit_outcome(outcome)" in commit


def test_the_episode_function_touches_no_shared_state():
    from pipeline import shadow
    src = inspect.getsource(shadow._shadow_one)
    for forbidden in ("ledger.", "last_rows.", "n_dec", "il_discount"):
        assert forbidden not in src, f"_shadow_one still reaches {forbidden}"


def test_the_replay_episode_function_touches_no_shared_state():
    from backtest import replay
    src = inspect.getsource(replay._replay_one)
    assert "ledger" not in src
    assert "rows.append" not in src
    assert "return row, spreads" in src


def test_the_frozen_posterior_is_read_only():
    from pipeline.shadow import _FrozenCells
    cells = _FrozenCells({"MEAT": {"mean": -1.0, "std": 0.4}})
    assert cells.get("MEAT")["mean"] == -1.0
    assert not hasattr(cells, "commit_update")
    with pytest.raises(KeyError):
        cells.get("NOT_A_CATEGORY")     # loud, not a silent default
