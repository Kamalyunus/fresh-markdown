"""common.parallel -- run a per-episode function across CPUs, or not at all.

Episodes are independent in both offline harnesses (tau is fixed for a whole
shadow run; the replay is deterministic), so the whole window parallelises.
Two rules: results ALWAYS return in submission order (order must not change
the answer), and workers compute while the parent commits (shared state never
crosses the boundary). workers=None/1 runs in-process, serial path unchanged.
"""

import os
from concurrent.futures import ProcessPoolExecutor


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


def map_episodes(fn, items, cfg, workers=None, chunks_per_worker=4):
    """`[fn(item, cfg) for item in items]`, optionally across processes.
    Chunked (per-task IPC would dwarf a millisecond DP solve), several chunks
    per worker since episode lengths vary by an order of magnitude. `fn` and
    every item must be picklable -- the constraint that forces purity."""
    n = resolve_workers(workers)
    if n <= 1 or len(items) < 2:
        return [fn(item, cfg) for item in items]

    size = max(len(items) // (n * chunks_per_worker), 1)
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    out = []
    with ProcessPoolExecutor(max_workers=n) as pool:
        # map, not as_completed: results must come back in submission order
        for got in pool.map(_run_chunk, [(fn, b, cfg) for b in batches]):
            out.extend(got)
    return out
