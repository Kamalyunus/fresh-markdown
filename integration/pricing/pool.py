"""The worker pool for an hour's batch: `[fn(item, ctx) for item in items]`,
chunked across processes, results in submission order. Every `fn` is pure."""
import os
from concurrent.futures import ProcessPoolExecutor


def resolve_workers(workers):
    """None -> serial; 0 -> every core but one; N -> N."""
    if workers is None:
        return 1
    if workers == 0:
        return max((os.cpu_count() or 2) - 1, 1)
    return max(int(workers), 1)


def _chunk(args):
    fn, items, ctx = args
    return [fn(item, ctx) for item in items]


def pmap(fn, items, ctx, workers=None, chunks_per_worker=4):
    n = resolve_workers(workers)
    if n <= 1 or len(items) < 2:
        return [fn(item, ctx) for item in items]
    size = max(len(items) // (n * chunks_per_worker), 1)
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    out = []
    with ProcessPoolExecutor(max_workers=n) as pool:
        for got in pool.map(_chunk, [(fn, b, ctx) for b in batches]):
            out.extend(got)
    return out
