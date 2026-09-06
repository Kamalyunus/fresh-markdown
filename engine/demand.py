"""engine.demand -- elasticity-scaled demand and the NB demand distribution.

    mu(d) = mu_ref * ((1 - d) / (1 - d_ref)) ^ epsilon,  floored at demand_floor

D ~ NegBin(r, mu) with Var[D] = mu + mu^2 / r. The pmf is truncated at
negbin_max_k with the tail mass assigned to the final bucket (design 5.7);
tail-mass diagnostics are returned so callers can emit them.
"""

import numpy as np
from scipy.stats import nbinom


def mu_at(mu_ref, d, d_ref, epsilon, demand_floor):
    ratio = (1.0 - d) / (1.0 - d_ref)
    return float(max(mu_ref * ratio ** epsilon, demand_floor))


def nb_pmf_table(mu, r, max_k):
    """P(D = k) for k = 0..max_k at every mu in `mu` (any shape), tail mass
    folded into the last bucket -- one scipy call for a whole DP's worth of
    (stage, tier) cells. `r` is one dispersion for the whole table (the DP)
    or one per row of `mu`, shape (n,) against an (n,) `mu` (the censored
    expectation over prepared rows). Returns (pmf of shape
    mu.shape + (max_k + 1,), tail of shape mu.shape). The ONE NB pmf.

    scipy parameterisation: nbinom(n=r, p=r/(r+mu)) has mean mu.
    """
    mu = np.asarray(mu, dtype=float)
    r = np.asarray(r, dtype=float)
    p = r / (r + mu)
    k = np.arange(max_k + 1)
    pmf = nbinom.pmf(k, r[..., None], p[..., None])
    tail = np.maximum(0.0, 1.0 - pmf.sum(axis=-1))
    pmf[..., -1] += tail
    return pmf, tail


def nb_pmf_vector(mu, r, max_k):
    """The one-mu case of nb_pmf_table: (pmf over k = 0..max_k, tail_mass)."""
    pmf, tail = nb_pmf_table(mu, r, max_k)
    return pmf, float(tail)


def expected_min_demand_inventory(mu, r, q, max_k):
    """E[min(D, q)] under the truncated NB -- deterministic replay transition."""
    if q <= 0:
        return 0.0
    pmf, _ = nb_pmf_vector(mu, r, max_k)
    k = np.arange(len(pmf))
    return float(np.sum(pmf * np.minimum(k, q)))


def expected_min_demand_inventory_vec(mu, r, q, max_k, chunk=100000):
    """Vectorised E[min(D, q)] -- the CENSORED expectation, what can actually
    be observed as sales. Every comparison of predictions against realised
    sales (fidelity, calibration gate, level factors) must use this, never
    raw mu: E[min(D,q)] <= E[D], so mixing the two bases misleads."""
    mu = np.asarray(mu, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    out = np.empty(len(mu))
    k = np.arange(max_k + 1)
    for start in range(0, len(mu), chunk):
        sl = slice(start, min(start + chunk, len(mu)))
        pmf, _ = nb_pmf_table(mu[sl], r[sl], max_k)       # per-row r
        out[sl] = np.sum(pmf * np.minimum(k[None, :], q[sl][:, None]), axis=1)
    return out
