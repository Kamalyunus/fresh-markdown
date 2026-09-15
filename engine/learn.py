"""engine.learn -- the censored NB posterior update on the grid (design 5.11).

The learning maths and nothing else: the censored likelihood over the
epsilon grid, the log prior, the moments, the sequential predictive check
and the Fisher information (deflated by the batch's own deff). No store,
no gate, no CLI -- `daily.update` collects the batch, calls `grid_update`
per cell and bounds the step (engine.posterior.bounded_step) before the
operator-gated commit.
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import nbinom

from common import episodes
from common.clustering import deff_from_episodes


def grid_update(pairs, cell_record, cfg):
    """Evaluate the censored likelihood on the grid, add the log prior,
    normalise with log-sum-exp, and take moments (design 5.11)."""
    pc = cfg["posterior"]
    grid = np.linspace(pc["epsilon_min"], pc["epsilon_max"], pc["grid_size"])

    k = np.array([o["units_sold"] for _, o, _ in pairs])
    inv = np.array([o["starting_inventory"] for _, o, _ in pairs])
    mu0 = np.array([d["reference_mu"] for d, _, _ in pairs])
    r = np.array([d["dispersion_r"] for d, _, _ in pairs])
    log_ratio = np.log([ratio for _, _, ratio in pairs])
    # censoring is the shared rule (the shelf EMPTIED), never `sold >= q`,
    # which reads a restocked hour as censored
    end = np.array([o["ending_inventory"] for _, o, _ in pairs])
    censored = episodes.is_censored_hour(inv, k, end)
    lgamma_const = gammaln(k + r) - gammaln(r) - gammaln(k + 1)

    # the censored term only where the shelf emptied: logsf is the costly
    # call, and on an uncensored row it was computed and thrown away
    inv_c, r_c = np.maximum(inv[censored], 1) - 1, r[censored]
    loglik = np.empty(len(grid))
    for i, eps in enumerate(grid):
        mu = np.clip(mu0 * np.exp(eps * log_ratio),
                     cfg["pricing"]["demand_floor"], None)
        p = r / (r + mu)
        ll = lgamma_const + r * np.log(p) + k * np.log1p(-p)
        if len(inv_c):
            ll[censored] = nbinom.logsf(inv_c, r_c, p[censored])
        loglik[i] = ll.sum()

    log_prior = -0.5 * ((grid - cell_record["mean"]) / cell_record["std"]) ** 2
    log_post = loglik + log_prior
    log_post -= log_post.max()
    w = np.exp(log_post)
    w /= w.sum()
    raw_mean = float(np.sum(w * grid))
    raw_std = float(np.sqrt(np.sum(w * (grid - raw_mean) ** 2)))

    # sequential predictive check (design 5.11): the batch's log marginal
    # predictive under the PRE-update posterior, bracketed by oracle and
    # uniform -- read differences, never absolutes
    n = len(pairs)
    log_w_prior = log_prior - logsumexp(log_prior)
    pred_posterior = float((logsumexp(loglik + log_w_prior)) / n)
    pred_uniform = float((logsumexp(loglik) - np.log(len(grid))) / n)
    pred_oracle = float(loglik.max() / n)
    predictive_check = {
        "posterior_log_pred_per_row": round(pred_posterior, 5),
        "uniform_log_pred_per_row": round(pred_uniform, 5),
        "oracle_log_pred_per_row": round(pred_oracle, 5),
        "information_available_per_row": round(pred_oracle - pred_uniform, 5),
        "posterior_minus_uniform": round(pred_posterior - pred_uniform, 5),
        "worse_than_a_flat_prior": bool(pred_posterior < pred_uniform),
        "note": ("batch scored against the PRE-update posterior -- an "
                 "out-of-sample grade of the current belief. Read "
                 "information_available_per_row first; a gap that is a large "
                 "share of a tiny number is still tiny. worse_than_a_flat_"
                 "prior persisting across batches means the posterior "
                 "tightened faster than the evidence justified."),
    }

    # NB Fisher information at the pre-update mean: mu * L^2 * r/(r+mu),
    # never the Poisson mu * L^2 (design 5.11) -- and on a CENSORED row
    # the information of the event actually observed (D >= q), which is
    # strictly less; crediting the uncensored figure overstated evidence
    # exactly where sell-outs dominate the batch
    mu_at_mean = np.clip(mu0 * np.exp(cell_record["mean"] * log_ratio),
                         cfg["pricing"]["demand_floor"], None)
    information = float(np.sum(row_information(
        mu_at_mean, r, log_ratio, inv, censored)))
    # deff at THIS batch's clustering: how many forced outcomes each
    # episode actually contributed, not a frozen paste
    batch_deff = deff_from_episodes(
        cfg["dispersion"]["rho"], [d["episode_id"] for d, _, _ in pairs])
    effective_information = information / batch_deff
    return raw_mean, raw_std, effective_information, {
        "zero_sales_share": round(float((k == 0).mean()), 4),
        "stockout_share": round(float(censored.mean()), 4),
        "exploration_cost": round(float(sum(
            d["exploration_cost"] for d, _, _ in pairs)), 2),
        "deff_applied": round(batch_deff, 3),
        "predictive_check": predictive_check,
    }


def row_information(mu, r, log_ratio, inv, censored):
    """Fisher information about epsilon per row, at `mu` (design 5.11).

    An uncensored count carries mu * L^2 * r/(r+mu). A censored row was
    observed only as the event D >= q (the shelf emptied), a Bernoulli
    with S = P(D >= q): its information is (dS/deps)^2 / (S (1 - S)), with
    dS/dmu = r/(r+mu) * (P(D' >= q-1) - P(D >= q)) for D' ~ NB(r+1, mu)
    (k P(D=k) = mu P(D'=k-1)) and dmu/deps = mu L. A certain event (S at
    0 or 1) teaches nothing and reads 0."""
    mu = np.asarray(mu, dtype=float)
    r = np.asarray(r, dtype=float)
    L = np.asarray(log_ratio, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    exact = mu * L ** 2 * r / (r + mu)
    if not censored.any():
        return exact
    q = np.maximum(np.asarray(inv, dtype=float), 1.0)
    p = r / (r + mu)
    s = nbinom.sf(q - 1, r, p)                         # P(D >= q)
    ds_dmu = r / (r + mu) * (nbinom.sf(q - 2, r + 1, p) - s)
    var = s * (1.0 - s)
    event = np.divide((ds_dmu * mu * L) ** 2, var,
                      out=np.zeros_like(exact), where=var > 0)
    return np.where(censored, event, exact)
