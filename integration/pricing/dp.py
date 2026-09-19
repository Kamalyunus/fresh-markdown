"""The monotone DP over the feasible tier set.

State (anchor tier, inventory, hours remaining); action a feasible tier at
or above the anchor discount and at or above cost; the chosen price is the
next anchor. Reward is absolute inventory loss; the terminal value books
the leftover at cost. The value function is built over the FULL grid; only
the action set allowed NOW is restricted (the coarse entry arms at entry,
the tiers at or deeper than the anchor after)."""
import time
from dataclasses import dataclass

import numpy as np

from pricing.demand import mu_at, nb_pmf_table

TIER_EPS = 1e-9     # two discounts closer than this are the same tier


def feasible_tiers(original_price, cost, tier_step):
    """{k * tier_step : 0 <= k * tier_step <= d_max}, ascending; 100% excluded."""
    d_max = 1.0 - cost / original_price
    if d_max < 0:
        return [], d_max
    n = int(np.floor(d_max / tier_step + TIER_EPS))
    tiers = [round(k * tier_step, 6) for k in range(n + 1)]
    return [d for d in tiers if d < 1.0], d_max


def entry_action_set(tiers, d_ref, d_max, pcfg):
    """Tier indices allowed at ENTRY: `pricing.entry_offsets` from d_ref,
    snapped to the grid and filtered by the cost floor; the deepest feasible
    tier alone when the floor forbids every arm."""
    step = pcfg["tier_step"]
    allowed = []
    for offset in pcfg["entry_offsets"]:
        target = d_ref + offset
        if target < -TIER_EPS or target > d_max + TIER_EPS:
            continue
        j = min(range(len(tiers)), key=lambda i: abs(tiers[i] - target))
        if abs(tiers[j] - target) <= step / 2 + TIER_EPS and j not in allowed:
            allowed.append(j)
    if not allowed:
        allowed = [len(tiers) - 1]
    return sorted(allowed)


@dataclass
class DPResult:
    tiers: list
    q_by_tier: dict
    d_ref: float
    solver_latency_s: float
    tail_mass_max: float

    @property
    def optimal_index(self):
        return max(self.q_by_tier, key=self.q_by_tier.get)


def solve(original_price, cost, q0, mu_ref_path, d_ref, epsilon, r, cfg,
          anchor_discount=None, entry=False):
    """Q over the actions allowed NOW, from the current decision onward.
    `mu_ref_path` index 0 is the hour being priced."""
    t0 = time.monotonic()
    pcfg = cfg["pricing"]
    tiers, d_max = feasible_tiers(original_price, cost, pcfg["tier_step"])
    if not tiers or q0 <= 0 or not len(mu_ref_path):
        raise ValueError("empty feasible set or degenerate state")
    horizon, n_tiers = len(mu_ref_path), len(tiers)
    max_k = max(int(pcfg["negbin_max_k"]), int(q0))

    mu = np.array([[mu_at(m, d, d_ref, epsilon, pcfg["demand_floor"])
                    for d in tiers] for m in mu_ref_path])
    pmf, tail = nb_pmf_table(mu, r, max_k)
    tail_max = float(tail.max())
    reward_per_unit = np.array([-(original_price - original_price * (1 - d)) for d in tiers])

    V = np.zeros((horizon + 1, n_tiers, q0 + 1))
    V[horizon, :, :] = -cost * np.arange(q0 + 1)[None, :]
    k = np.arange(max_k + 1)
    q_grid = np.arange(q0 + 1)
    sold = np.minimum(k[None, :], q_grid[:, None])
    left = q_grid[:, None] - sold
    Q_now = None
    for t in range(horizon - 1, -1, -1):
        Q = np.sum(pmf[t][:, None, :]
                   * (sold[None] * reward_per_unit[:, None, None] + V[t + 1][:, left]), axis=2)
        V[t] = np.maximum.accumulate(Q[::-1], axis=0)[::-1]
        if t == 0:
            Q_now = Q

    if entry:
        allowed = entry_action_set(tiers, d_ref, d_max, pcfg)
    else:
        if anchor_discount is None:
            raise ValueError("hourly decision requires anchor_discount")
        allowed = [j for j, d in enumerate(tiers) if d >= anchor_discount - TIER_EPS]
        if not allowed:
            raise ValueError("no feasible tier at or below the current anchor price")
    q_by_tier = {j: float(Q_now[j, q0]) for j in allowed}
    return DPResult(tiers, q_by_tier, d_ref, time.monotonic() - t0, tail_max)
