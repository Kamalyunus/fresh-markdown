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


def expected_min_demand_inventory_vec(mu, r, q, max_k, chunk=100000):
    """Vectorised E[min(D, q)].

    This is the CENSORED expectation -- what can actually be observed as
    sales, since an hour cannot sell more than its inventory. Any comparison
    of predictions against realised sales (fidelity, the calibration gate,
    the level factors) must use this, never raw mu: E[min(D,q)] <= E[D], so
    mixing the two makes the model look better on one basis than the other
    and a factor fit on raw mu can never move a gate read on censored
    predictions.
    """
    mu = np.asarray(mu, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    out = np.empty(len(mu))
    k = np.arange(max_k + 1)
    for start in range(0, len(mu), chunk):
        sl = slice(start, min(start + chunk, len(mu)))
        p = (r[sl] / (r[sl] + mu[sl]))[:, None]
        pmf = nbinom.pmf(k[None, :], r[sl][:, None], p)
        pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0.0, None)
        out[sl] = np.sum(pmf * np.minimum(k[None, :], q[sl][:, None]), axis=1)
    return out
