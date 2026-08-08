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
    priced. Returns Q over the actions allowed NOW: entry arms within
    d_ref +/- entry_window for an entry decision, else feasible tiers at or
    above the current anchor discount.
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
        lo = max(0.0, d_ref - pcfg["entry_window"])
        hi = min(d_max, d_ref + pcfg["entry_window"])
        allowed = [j for j, d in enumerate(tiers) if lo - 1e-9 <= d <= hi + 1e-9]
        if not allowed:                     # clip to [0, d_max] can empty the band
            allowed = list(range(n_tiers))
    else:
        if anchor_discount is None:
            raise ValueError("hourly decision requires anchor_discount")
        allowed = [j for j, d in enumerate(tiers) if d >= anchor_discount - 1e-9]
        if not allowed:
            raise ValueError("no feasible tier at or below the current anchor price")

    q_by_tier = {j: float(Q_now[j, q0]) for j in allowed}
    v_star = max(q_by_tier.values())
    return DPResult(tiers, q_by_tier, v_star, time.monotonic() - t0, tail_max)
