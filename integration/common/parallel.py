"""common.parallel -- run a per-episode function across CPUs, or not at all.

Episodes are independent in both offline harnesses (tau is fixed for a whole
shadow run; the replay is deterministic), so the whole window parallelises.
Two rules: results ALWAYS return in submission order (order must not change
the answer), and workers compute while the parent commits (shared state never
crosses the boundary). workers=None/1 runs in-process, serial path unchanged.
"""

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def keyed_rng(seed, *key):
    """ONE generator per (seed, key): the draw depends on what is priced
    (an episode id, an hour key, an episode and its hour), never on which
    worker prices it or in what order -- the rule that makes serial and
    parallel runs agree. `key` parts are joined as text and hashed
    (blake2b), so any tuple of ids spells one stream; the seed moves every
    stream. The one home for the per-decision seeding the batch caller,
    shadow and the simulator each spelt for themselves."""
    h = hashlib.blake2b("|".join(map(str, key)).encode(), digest_size=8).digest()
    return np.random.default_rng([int(seed), int.from_bytes(h, "big")])


def resolve_workers(workers):
    """`None` -> serial. 0 -> every core but one. N -> N."""
    if workers is None:
        return 1
    if workers == 0:
        return max((os.cpu_count() or 2) - 1, 1)
    return max(int(workers), 1)


def _run_chunk(args):
    fn, items, cfg = args
    return [fn(item, cfg) for item in items]


class EpisodePool:
    """The process pool a run holds across many batches (the simulator maps
    one batch per hour: one executor per hour would fork 500 times a run).
    `map` is `[fn(item, cfg) for item in items]`, chunked across the
    workers (per-task IPC would dwarf a millisecond DP solve; several
    chunks per worker since episode lengths vary by an order of magnitude),
    results in submission order; serial in-process until the pool is
    entered, for a single worker, or for a batch under `serial_below`
    items (a batch too small to be worth the IPC). `fn` and every item
    must be picklable -- the constraint that forces purity."""

    def __init__(self, workers=None, chunks_per_worker=4, serial_below=2):
        self.workers = resolve_workers(workers)
        self.chunks_per_worker = int(chunks_per_worker)
        self.serial_below = int(serial_below)
        self._pool = None

    def __enter__(self):
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self

    def __exit__(self, *exc):
        self.shutdown()

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def map(self, fn, items, cfg):
        n = self.workers
        if self._pool is None or n <= 1 or len(items) < self.serial_below:
            return [fn(item, cfg) for item in items]
        size = max(len(items) // (n * self.chunks_per_worker), 1)
        batches = [items[i:i + size] for i in range(0, len(items), size)]
        out = []
        # map, not as_completed: results must come back in submission order
        for got in self._pool.map(_run_chunk, [(fn, b, cfg) for b in batches]):
            out.extend(got)
        return out


def map_episodes(fn, items, cfg, workers=None, chunks_per_worker=4):
    """`[fn(item, cfg) for item in items]`, optionally across processes:
    one EpisodePool for one batch (the offline harnesses map their whole
    window once)."""
    n = resolve_workers(workers)
    if n <= 1 or len(items) < 2:
        return [fn(item, cfg) for item in items]
    with EpisodePool(n, chunks_per_worker) as pool:
        return pool.map(fn, items, cfg)
