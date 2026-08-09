"""pricing.dp -- monotone DP over the feasible tier set (PRD section 11).

State (price_anchor, inventory_q, hours_remaining_t). Action: a feasible tier
at or below the anchor price (discount at or above the anchor discount) and at
or above cost. The chosen price becomes the next anchor.

    Q(anchor, q, h, p) = sum_k P(D = k | r, mu(p)) * [ min(k, q) * (-(P0 - p))
                                                       + V(p, q - min(k, q), h - 1) ]
    V(anchor, q, h)    = max over feasible p <= anchor of Q(anchor, q, h, p)
    V(anchor, q, 0)    = -cost * q

The reward is absolute IL per section 3.1: no ratio transform, no outer loop.
mu(p) uses the posterior MEAN epsilon. State scale is small (tiers ~2-20,
horizon under twelve hours, inventory under thirty), so evaluation is
exhaustive.
"""

import time

import numpy as np

from pricing.demand import mu_at, nb_pmf_vector


def feasible_tiers(original_price, cost, tier_step):
    """{k * tier_step : 0 <= k * tier_step <= d_max}, ascending discounts."""
    d_max = 1.0 - cost / original_price
    if d_max < 0:
        return [], d_max
    n = int(np.floor(d_max / tier_step + 1e-9))
    return [round(k * tier_step, 6) for k in range(n + 1)], d_max


def deepening_threshold_epsilon(original_price, cost, d):
    """|epsilon| above which deepening the discount below `d` reduces IL.

    The hourly action set already contains EVERY tier deeper than the anchor
    -- 2.5pp, 5pp, or a jump straight to the cost floor. Whether the DP uses
    them is an economics question, not an action-set question, and it has a
    closed form. Ignoring inventory censoring, one hour of IL is

        IL(d) = P0*d*mu(d) + c*(q - mu(d)),   mu(d) = mu_ref*((1-d)/(1-d_ref))^eps

    so with a = |epsilon|, dmu/dd = mu*a/(1-d) and

        dIL/dd = mu * [ P0 + (P0*d - c) * a/(1-d) ]

    which is negative -- deepening pays -- only when a > (1-d)/(gamma-d),
    gamma = cost/price. The first term is the cost of discounting the units
    that would have sold anyway; it dominates until demand responds hard
    enough to outrun it.

    Returns inf when gamma <= d (price already at or under cost: no deepening
    can pay). The bound is OPTIMISTIC: censoring at small inventory blunts the
    demand gain, so the true switch point sits above this value -- with median
    starting inventory ~2, materially so.
    """
    gamma = cost / original_price
    return float("inf") if gamma - d <= 1e-9 else (1.0 - d) / (gamma - d)


def entry_action_set(tiers, d_ref, d_max, pcfg):
    """Tier indices allowed for the ENTRY decision.

    Entry is a genuinely different decision from the hourly one and gets its
    own, coarser action set: `pricing.entry_offsets` in percentage points of
    discount RELATIVE TO the category reference, e.g. [-0.15, -0.10, -0.05, 0]
    means "enter up to 15pp shallower than reference, or at reference".

    Two reasons the entry grid is coarse and one-sided rather than the hourly
    2.5pp grid:

    - Monotonicity makes entry irreversible in one direction. Price may only
      hold or deepen afterwards, so the entry choice sets the ceiling on
      every later price in the episode. Entering deeper than reference spends
      margin in hour one AND forfeits the room to deepen later; entering
      shallower is the lever that saves IL, and the whole episode's path
      hangs off it.
    - Coarse arms concentrate exploration evidence. Entry is where
      identification lives (section 3.2 -- the confound-free variation is
      entry-hour variation across episodes), so four well-separated arms
      produce a far sharper elasticity read than sixteen 2.5pp arms whose
      demand differences are inside the noise.

    Offsets are snapped to the tier grid and filtered by the cost floor. If
    the floor forbids every requested arm, the deepest feasible tier is the
    only action -- a single-action decision, correctly non-explorable, rather
    than a silent fallback to the full grid.
    """
    step = pcfg["tier_step"]
    allowed = []
    for offset in pcfg["entry_offsets"]:
        target = d_ref + offset
        if target < -1e-9 or target > d_max + 1e-9:
            continue
        j = min(range(len(tiers)), key=lambda i: abs(tiers[i] - target))
        if abs(tiers[j] - target) <= step / 2 + 1e-9 and j not in allowed:
            allowed.append(j)
    if not allowed:
        allowed = [len(tiers) - 1]
    return sorted(allowed)


class DPResult:
    def __init__(self, tiers, q_by_tier, v_star, solver_latency_s, tail_mass_max):
        self.tiers = tiers                    # discounts, ascending
        self.q_by_tier = q_by_tier            # {tier_index: Q value} for allowed actions
        self.v_star = v_star
        self.solver_latency_s = solver_latency_s
        self.tail_mass_max = tail_mass_max

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
    pmf = np.empty((horizon, n_tiers, max_k + 1))
    tail_max = 0.0
    for t in range(horizon):
        for j, d in enumerate(tiers):
            mu = mu_at(mu_ref_path[t], d, d_ref, epsilon, pcfg["demand_floor"])
            pmf[t, j], tail = nb_pmf_vector(mu, r, max_k)
            tail_max = max(tail_max, tail)

    reward_per_unit = np.array([-(original_price - original_price * (1 - d))
                                for d in tiers])          # -(P0 - p) = -P0 * d

    # V[t][j][q] = value at stage t with anchor tier j and inventory q
    V = np.zeros((horizon + 1, n_tiers, q0 + 1))
    V[horizon, :, :] = -cost * np.arange(q0 + 1)[None, :]

    k = np.arange(max_k + 1)
    Q_now = None
    for t in range(horizon - 1, -1, -1):
        Q = np.full((n_tiers, q0 + 1), -np.inf)
        for j in range(n_tiers):                      # action tier j
            for q in range(q0 + 1):
                sold = np.minimum(k, q)
                Q[j, q] = np.sum(pmf[t, j] * (sold * reward_per_unit[j]
                                              + V[t + 1, j, q - sold]))
        # V under anchor a = max over actions j >= a (deeper or equal discount)
        running = np.maximum.accumulate(Q[::-1], axis=0)[::-1]
        V[t] = running
        if t == 0:
            Q_now = Q

    if entry:
        allowed = entry_action_set(tiers, d_ref, d_max, pcfg)
    else:
        if anchor_discount is None:
            raise ValueError("hourly decision requires anchor_discount")
        allowed = [j for j, d in enumerate(tiers) if d >= anchor_discount - 1e-9]
        if not allowed:
            raise ValueError("no feasible tier at or below the current anchor price")

    q_by_tier = {j: float(Q_now[j, q0]) for j in allowed}
    v_star = max(q_by_tier.values())
    return DPResult(tiers, q_by_tier, v_star, time.monotonic() - t0, tail_max)
