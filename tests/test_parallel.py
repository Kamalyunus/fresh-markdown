"""Parallelism must change speed and nothing else."""

import inspect
import time

import numpy as np
import pytest

from common.parallel import EpisodePool, map_episodes, resolve_workers


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


def test_a_held_pool_maps_many_batches_like_the_one_shot_map():
    """The simulator holds one pool across its hours (one executor per hour
    would fork hundreds of times a run): every batch through it agrees
    with map_episodes, in submission order; a batch under `serial_below`
    items, or a pool not entered, prices in-process with the same answer."""
    ctx = {"offset": 3}
    batches = [list(range(n)) for n in (1, 5, 97)]
    with EpisodePool(3, serial_below=4) as pool:
        assert pool.workers == 3
        for items in batches:
            assert pool.map(_square, items, ctx) == map_episodes(_square, items, ctx)
    unentered = EpisodePool(3)
    assert unentered.map(_square, batches[-1], ctx) == map_episodes(_square, batches[-1], ctx)


def test_resolve_workers():
    import os
    assert resolve_workers(None) == 1          # serial unless asked
    assert resolve_workers(1) == 1
    assert resolve_workers(3) == 3
    assert resolve_workers(0) == max((os.cpu_count() or 2) - 1, 1)


# ------------------------------------------------------- per-episode seeding

def test_keyed_rng_is_the_one_per_decision_seeding():
    """One generator per (seed, key), from the key alone: the same hour
    draws the same whichever worker prices it; another hour, another
    episode or another --seed draws differently. The batch caller's hour
    key and the simulator's (episode, hour) both spell a key."""
    from common.parallel import keyed_rng
    key = ("7", "F1", "2026-08-19", 17)
    a = keyed_rng(0, *key).integers(0, 10_000, 5)
    assert np.array_equal(a, keyed_rng(0, *key).integers(0, 10_000, 5))
    assert not np.array_equal(a, keyed_rng(0, "7", "F1", "2026-08-19", 18).integers(0, 10_000, 5))
    assert not np.array_equal(a, keyed_rng(1, *key).integers(0, 10_000, 5))
    ep = keyed_rng(0, "sku|fc|2026-08-04T09", 3).integers(0, 10_000, 5)
    assert not np.array_equal(ep, keyed_rng(0, "sku|fc|2026-08-04T09", 4).integers(0, 10_000, 5))


def test_the_episode_generator_is_reproducible_and_order_free():
    from common.parallel import keyed_rng as _episode_seed
    a = _episode_seed(0, "sku|fc|2026-08-04T09").integers(0, 10_000, 5)
    b = _episode_seed(0, "sku|fc|2026-08-04T09").integers(0, 10_000, 5)
    assert np.array_equal(a, b), "same episode, same seed -> same draws"

    other = _episode_seed(0, "sku|fc|2026-08-05T09").integers(0, 10_000, 5)
    assert not np.array_equal(a, other), "different episodes must not collide"

    seeded = _episode_seed(1, "sku|fc|2026-08-04T09").integers(0, 10_000, 5)
    assert not np.array_equal(a, seeded), "--seed must still move the draws"


# ------------------------------------------------ workers never write events

def test_workers_buffer_events_and_the_parent_commits_them():
    from evaluate import shadow
    from engine.state import BufferStore

    buf = BufferStore()
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
    from evaluate import shadow
    src = inspect.getsource(shadow._shadow_one)
    for forbidden in ("ledger.", "last_rows.", "n_dec", "il_discount"):
        assert forbidden not in src, f"_shadow_one still reaches {forbidden}"


def test_the_replay_episode_function_touches_no_shared_state():
    from evaluate import backtest as replay
    src = inspect.getsource(replay._replay_one)
    assert "ledger" not in src
    assert "rows.append" not in src
    assert "return row, spreads" in src


def test_the_frozen_posterior_is_read_only():
    from engine.state import FrozenCells
    cells = FrozenCells({"MEAT": {"mean": -1.0, "std": 0.4}})
    assert cells.get("MEAT")["mean"] == -1.0
    assert not hasattr(cells, "commit_update")
    with pytest.raises(KeyError):
        cells.get("NOT_A_CATEGORY")     # loud, not a silent default
