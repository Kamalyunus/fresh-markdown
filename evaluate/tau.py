"""evaluate.tau -- what the two harnesses share around the tau cross-check.

The backtest's `derive_tau_initial` (design 5.14, the exploit-only
cross-check) and shadow's `derive_tau0` (5.13, the launch paste) fill the
same `engine.explore.SpreadLedger` from a mapped episode function, on an
episode sample drawn the same way, and report the solve in a block whose
shared keys `ops.config_keys.tau_provenance_error` reads. Those three
pieces live here once: `sample_ids`, `fill_ledger`, `tau_derivation_block`.
"""

from common.parallel import map_episodes


def sample_ids(ids, n, rng):
    """`n` of `ids` without replacement, from `rng` -- the ONE draw every
    sampled harness report stands on (shadow's window and pre-window
    samples, the replay's policy sample, the step sweep's episodes). `ids`
    may be an int: a sample of positions."""
    return rng.choice(ids, n, replace=False)


def fill_ledger(fn, items, ctx, workers, ledger, spreads_of):
    """`map_episodes(fn, items, ctx, workers)`, folding every decision's
    Q-spreads into `ledger` as each episode's result arrives -- results in
    submission order, the ledger built in it -- and yielding the result
    for the caller's own fold. `spreads_of(out)` lists them as
    `ledger.add` arguments (shadow's carry the moves and delta_min; the
    replay's are (day, costs))."""
    for out in map_episodes(fn, items, ctx, workers):
        for spread in spreads_of(out):
            ledger.add(*spread)
        yield out


def tau_derivation_block(ledger, budget, n_days, **extra):
    """The block both derivations report the solve in: `tau_initial` (the
    ledger's bisection at `budget` per day over `n_days`, 2 dp), the
    caller's `extra` keys in their order, then `implied_daily_spend` at
    that tau (1 dp). Returns (block, tau) -- the unrounded tau for the
    caller's own readings -- or (None, None) when the ledger has nothing
    to bisect on (no spread, no budget)."""
    tau = ledger.solve_tau(budget, n_days=n_days)
    if tau is None:
        return None, None
    return {"tau_initial": round(float(tau), 2), **extra,
            "implied_daily_spend": round(ledger.implied_daily_spend(tau, n_days), 1)}, tau
