"""Elasticity-scaled demand and its negative-binomial distribution.

    mu(d) = mu_ref * ((1 - d) / (1 - d_ref)) ^ epsilon,  floored at demand_floor

D ~ NegBin(r, mu). The pmf table runs to `max_k` with the tail mass folded
into the last bucket; the censored expectation E[min(D, q)] is exact at
every q (the table where q fits inside it, the closed form beyond)."""
import numpy as np
from scipy.stats import nbinom


def mu_at(mu_ref, d, d_ref, epsilon, demand_floor):
    ratio = (1.0 - d) / (1.0 - d_ref)
    return float(max(mu_ref * ratio ** epsilon, demand_floor))


def nb_pmf_table(mu, r, max_k):
    """P(D = k) for k = 0..max_k at every mu (any shape), tail folded into
    the last bucket. scipy: nbinom(n=r, p=r/(r+mu)) has mean mu."""
    mu = np.asarray(mu, dtype=float)
    r = np.asarray(r, dtype=float)
    p = r / (r + mu)
    k = np.arange(max_k + 1)
    pmf = nbinom.pmf(k, r[..., None], p[..., None])
    tail = np.maximum(0.0, 1.0 - pmf.sum(axis=-1))
    pmf[..., -1] += tail
    return pmf, tail


def censored_mean_closed_form(mu, r, q):
    """E[min(D, q)] in closed form: mu * P(D' <= q-2 | r+1) + q * P(D >= q)."""
    mu = np.asarray(mu, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    p = r / (r + mu)
    return mu * nbinom.cdf(q - 2, r + 1, p) + q * nbinom.sf(q - 1, r, p)


def expected_min_demand_inventory(mu, r, q, max_k):
    """E[min(D, q)]: what can be observed as sales, exact at every q."""
    if q <= 0:
        return 0.0
    mu_a, r_a, q_a = np.asarray([mu], float), np.asarray([r], float), np.asarray([q], float)
    if q > max_k:
        return float(censored_mean_closed_form(mu_a, r_a, q_a)[0])
    pmf, _ = nb_pmf_table(mu_a, r_a, max_k)
    k = np.arange(max_k + 1)
    return float(np.sum(pmf * np.minimum(k[None, :], q_a[:, None]), axis=1)[0])
