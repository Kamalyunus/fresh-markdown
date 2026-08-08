"""pricing.demand -- elasticity-scaled demand and the NB demand distribution.

    mu(d) = mu_ref * ((1 - d) / (1 - d_ref)) ^ epsilon,  floored at demand_floor

D ~ NegBin(r, mu) with Var[D] = mu + mu^2 / r. The pmf is truncated at
negbin_max_k with the tail mass assigned to the final bucket (PRD section 11.3);
tail-mass diagnostics are returned so callers can emit them.
"""

import numpy as np
from scipy.stats import nbinom


def mu_at(mu_ref, d, d_ref, epsilon, demand_floor):
    ratio = (1.0 - d) / (1.0 - d_ref)
    return float(max(mu_ref * ratio ** epsilon, demand_floor))


def nb_pmf_vector(mu, r, max_k):
    """P(D = k) for k = 0..max_k, tail mass folded into the last bucket.

    scipy parameterisation: nbinom(n=r, p=r/(r+mu)) has mean mu.
    Returns (pmf, tail_mass).
    """
    p = r / (r + mu)
    k = np.arange(max_k + 1)
    pmf = nbinom.pmf(k, r, p)
    tail = float(max(0.0, 1.0 - pmf.sum()))
    pmf[-1] += tail
    return pmf, tail


def nb_logpmf(k, mu, r):
    return float(nbinom.logpmf(k, r, r / (r + mu)))


def nb_logsf_at_least(k, mu, r):
    """log P(D >= k) = log survival_function(k - 1)."""
    return float(nbinom.logsf(k - 1, r, r / (r + mu)))


def expected_min_demand_inventory(mu, r, q, max_k):
    """E[min(D, q)] under the truncated NB -- deterministic replay transition."""
    if q <= 0:
        return 0.0
    pmf, _ = nb_pmf_vector(mu, r, max_k)
    k = np.arange(len(pmf))
    return float(np.sum(pmf * np.minimum(k, q)))
