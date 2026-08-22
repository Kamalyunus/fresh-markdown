"""common.parallel -- run a per-episode function across CPUs, or not at all.

Episodes are independent in both offline harnesses. Nothing inside either
loop reads another episode, and nothing reads another DAY either: `tau` is
fixed for the whole of a shadow run (the day-by-day controller walk is
post-processing over aggregates), and the replay is deterministic. So the
unit of work is one episode, and the whole window parallelises -- not just
within a day.

Two rules this module exists to enforce:

  1. **Order must not change the answer.** Results come back in submission
     order, always, whatever order the workers finish in. A reduction that
     depended on completion order would make a parallel run irreproducible
     and -- worse -- would make it differ from the serial one silently.
  2. **Workers compute, the parent commits.** Anything with shared state
     (an event store, a random draw sequence) stays in the parent. A worker
     returns data; it never writes.

`workers=0` or 1 runs in-process with no pool at all, which keeps the serial
path exactly what it was and makes "is the parallel result identical?" a
question you can answer by running both.
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

    Chunked rather than one task per episode: an episode is milliseconds of
    work and per-task IPC would cost more than the DP solve it replaces.
    Several chunks per worker keeps a slow chunk from stranding a core at the
    end -- episode lengths vary by an order of magnitude, so equal-sized
    chunks are not equal-time chunks.

    `fn` and every item must be picklable, which is the real constraint: it
    is what forces the per-episode function to be pure. A closure over the
    frame, or a worker that writes to the event store, will not survive the
    boundary -- by design.
    """
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
