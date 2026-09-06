"""engine.dp -- monotone DP over the feasible tier set (design section 5.7).

State (price_anchor, inventory_q, hours_remaining_t). Action: a feasible tier
at or below the anchor price (discount at or above the anchor discount) and at
or above cost. The chosen price becomes the next anchor.

    Q(anchor, q, h, p) = sum_k P(D = k | r, mu(p)) * [ min(k, q) * (-(P0 - p))
                                                       + V(p, q - min(k, q), h - 1) ]
    V(anchor, q, h)    = max over feasible p <= anchor of Q(anchor, q, h, p)
    V(anchor, q, 0)    = -cost * q

The reward is absolute IL per design 2.2: no ratio transform, no outer loop.
mu(p) uses the posterior MEAN epsilon. State scale is small (tiers ~2-20,
horizon under twelve hours, inventory under thirty), so evaluation is
exhaustive.
"""

import time
from dataclasses import dataclass

import numpy as np

from engine.demand import mu_at, nb_pmf_table

# float noise on the discount grid: tiers are round(k * step, 6), so two
# discounts closer than this ARE the same tier. Every "is this on / at /
# below a tier" comparison in the repo uses this one epsilon.
TIER_EPS = 1e-9


def feasible_tiers(original_price, cost, tier_step):
    """{k * tier_step : 0 <= k * tier_step <= d_max}, ascending discounts.

    A 100% discount is EXCLUDED even when a zero cost makes d_max = 1.0:
    this layer owns "which prices are legal" and must not rely on upstream
    filters, and mu(d) is undefined at d = 1. `d_max` is still returned as
    the true cost floor so the decision event records the economics.
    """
    d_max = 1.0 - cost / original_price
    if d_max < 0:
        return [], d_max
    n = int(np.floor(d_max / tier_step + TIER_EPS))
    tiers = [round(k * tier_step, 6) for k in range(n + 1)]
    return [d for d in tiers if d < 1.0], d_max


def deepening_threshold_epsilon(original_price, cost, d):
    """|epsilon| above which deepening below `d` reduces IL:
    (1-d)/(gamma-d), gamma = cost/price (design 5.7 for the derivation).

    inf when gamma <= d. OPTIMISTIC: censoring at small inventory pushes the
    true switch point above this value.
    """
    gamma = cost / original_price
    return float("inf") if gamma - d <= TIER_EPS else (1.0 - d) / (gamma - d)


def entry_action_set(tiers, d_ref, d_max, pcfg):
    """Tier indices allowed at ENTRY: `pricing.entry_offsets` relative to
    d_ref, snapped to the grid and filtered by the cost floor (rationale for
    the coarse one-sided arm set: design 5.7 / config comment).

    If the floor forbids every requested arm, the deepest feasible tier is
    the only action -- correctly non-explorable, never a silent fallback to
    the full grid.
    """
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
    tiers: list                 # discounts, ascending
    q_by_tier: dict             # {tier_index: Q value} for the actions allowed NOW
    d_ref: float                # the reference discount the forecast is quoted at
    solver_latency_s: float
    tail_mass_max: float

    @property
    def optimal_index(self):
        return max(self.q_by_tier, key=self.q_by_tier.get)


def solve(original_price, cost, q0, mu_ref_path, d_ref, epsilon, r, cfg,
          anchor_discount=None, entry=False):
    """Solve the episode DP from the current decision onward.

    mu_ref_path: mu_ref for each remaining hour, index 0 = the hour being
    priced. Returns Q over the actions allowed NOW: the coarse entry arms
    (see entry_action_set) for an entry decision, else feasible tiers at or
    above the current anchor discount.

    Note the value function is built over the FULL tier grid either way --
    only the action set allowed at this decision is restricted. An episode
    that enters on a coarse arm still deepens on the 2.5pp grid, and the DP
    accounts for that when valuing the entry choice.
    """
    t0 = time.monotonic()
    pcfg = cfg["pricing"]
    tiers, d_max = feasible_tiers(original_price, cost, pcfg["tier_step"])
    if not tiers or q0 <= 0 or not len(mu_ref_path):
        raise ValueError("empty feasible set or degenerate state")

    horizon = len(mu_ref_path)
    n_tiers = len(tiers)
    max_k = pcfg["negbin_max_k"]

    # pmf[t][j] over sold counts for pricing tier j at stage t
    mu = np.array([[mu_at(m, d, d_ref, epsilon, pcfg["demand_floor"])
                    for d in tiers] for m in mu_ref_path])
    pmf, tail = nb_pmf_table(mu, r, max_k)
    tail_max = float(tail.max())

    reward_per_unit = np.array([-(original_price - original_price * (1 - d))
                                for d in tiers])          # -(P0 - p) = -P0 * d

    # V[t][j][q] = value at stage t with anchor tier j and inventory q
    V = np.zeros((horizon + 1, n_tiers, q0 + 1))
    V[horizon, :, :] = -cost * np.arange(q0 + 1)[None, :]

    # sold[q, k] = min(k, q) and the inventory left after it, for every
    # (inventory, demand) pair -- the gather is the same at every stage
    k = np.arange(max_k + 1)
    q_grid = np.arange(q0 + 1)
    sold = np.minimum(k[None, :], q_grid[:, None])
    left = q_grid[:, None] - sold
    Q_now = None
    for t in range(horizon - 1, -1, -1):
        # Q[j, q] = sum_k pmf[t, j, k] * (sold[q, k] * reward[j] + V[t+1, j, left[q, k]])
        # over every action tier j and inventory q at once
        Q = np.sum(pmf[t][:, None, :]
                   * (sold[None] * reward_per_unit[:, None, None]
                      + V[t + 1][:, left]), axis=2)
        # V under anchor a = max over actions j >= a (deeper or equal discount)
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
