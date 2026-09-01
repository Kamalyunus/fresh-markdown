"""bootstrap.prior_density -- the elasticity prior AS the profile likelihood.

Censored Poisson profile (QMLE consistent for the mean, so no dispersion
parameter and no epsilon <-> r cycle), read whole-curve as a density; a 50/50
naive/controlled mixture generalises the old midpoint bracket. No fallback
constant: uninformative categories shrink to a pooled density measured on the
right-signed categories. Spec: docs/design.md 5.6; history: docs/learnings.md.
"""

import copy

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from common.config import design_effect, intraclass_correlation
from common import episodes
from bootstrap.prepare_data import population, split_frames


def scored_rows(frame, cfg):
    """ENTRY ROWS -- one per episode, its first hour, before any
    within-episode selection has happened (rule 7; alternatives are gone,
    see learnings.md)."""
    f = frame.copy()
    f["censored"] = episodes.censored_hours(f)
    f = f[f.starting_inventory >= 1]
    # censored entry rows kept: the Poisson likelihood handles censoring
    # without a dispersion parameter, so there is nothing to drop them for
    # DATE first: sorting on hour alone picks the 00:00 row of an episode
    # that opened at 22:00 the night before -- a within-episode, post-price-
    # path row, exactly the confound rule 7 exists to exclude. Production
    # windows routinely cross midnight (design 12a); episodes.last_rows
    # already orders by ("date", "hour_of_day") for the same reason.
    return f.sort_values(["episode_id", "date", "hour_of_day"]) \
            .groupby("episode_id").head(1).copy()


def hour_multipliers(mu, cell, k, censored, min_rows=1):
    """Multiplicative time-cell fixed effects profiled out by first-moment
    matching on uncensored rows -- family-agnostic (Poisson and NB alike).
    Thin cells (under `min_rows`) fall back to 1.0 -- a small cell absorbs the
    price response itself; returns (multipliers, fallback share)."""
    frame = pd.DataFrame({"cell": cell, "k": k, "mu": mu, "cen": censored})
    unc = frame[~frame.cen]
    if not len(unc):
        return np.ones(len(frame)), 0.0
    g = unc.groupby("cell")
    m = (g.k.sum() / g.mu.sum()).clip(0.05, 20.0)
    if min_rows > 1:
        m = m[g.size() >= min_rows]
    mult = frame.cell.map(m).fillna(1.0).to_numpy()
    thin = float(np.mean(frame.cell.map(m).isna().to_numpy()))
    return mult, thin


def time_cell(g):
    """DATE x HOUR -- the cell the controlled arm profiles out (the pooled
    hour_of_day alternative is gone; see learnings.md)."""
    return (g.date.astype(str) + "|" + g.hour_of_day.astype(str)).to_numpy()


def loglik(eps, base, log_ratio, k, censored, floor, controlled, cell, const,
           min_cell=1):
    """Censored Poisson log-likelihood at one elasticity -- QMLE consistent
    for the mean, so no dispersion parameter (no r) enters this step."""
    mu = np.clip(base * np.exp(eps * log_ratio), floor, None)
    if controlled:
        mult, _ = hour_multipliers(mu, cell, k, censored, min_cell)
        mu = np.clip(mu * mult, floor, None)
    exact = k * np.log(mu) - mu - const
    tail = poisson.logsf(np.maximum(k, 1) - 1, mu)
    return float(np.sum(np.where(censored, tail, exact)))


def curve(g, mu_ref, grid, controlled, cfg):
    """The full log-likelihood over the grid -- the shape, not just an
    argmax."""
    pc = cfg["posterior"]["prior"]
    k = g.units_sold.to_numpy()
    censored = g.censored.to_numpy()
    cell = time_cell(g)
    min_cell = int(pc.get("min_rows_per_time_cell", 1))
    log_ratio = np.log((1 - g.total_discount.to_numpy())
                       / (1 - g.d_ref.to_numpy()))
    floor = cfg["pricing"]["demand_floor"]
    const = gammaln(k + 1)
    return np.array([loglik(e, mu_ref, log_ratio, k, censored, floor,
                            controlled, cell, const, min_cell)
                     for e in grid])


def deflation_deff(rows, model, cfg):
    """Design effect for the prior's likelihood -- eps-FREE, keeping the
    epsilon step out of the dispersion chain (`fit_dispersion` owns the
    fitted rho). Returns (deff, rho, mean rows per cluster).

    Clustered on SKU x FC, NOT on episode. These are ENTRY rows (rule 7), so
    there is exactly one per episode and a within-episode ICC is 1.0 by
    construction -- the old episode grouping made `sizes >= 3` empty, rho 0
    and deff exactly 1.0 for every category, so design 5.6's deflation could
    never engage and the pooled shrinkage ran on undeflated spans. The
    correlation that does exist between entry rows is the same unit recurring
    across days: the same cluster the A/B randomises on, and the one that
    does not average away.
    """
    resid = rows.units_sold.to_numpy() - model.predict_mu_ref(rows)
    unit = (rows.sku_id.astype(str) + "|" + rows.fc.astype(str)).to_numpy()
    f = pd.DataFrame({"unit": unit, "resid": resid})
    sizes = f.groupby("unit").resid.size()
    min_rows = cfg["assurance"]["rho_min_hours_per_episode"]
    sub = f[f.unit.isin(sizes[sizes >= min_rows].index)]
    rho = intraclass_correlation(sub.resid, sub.unit,
                                 cfg["dispersion"]["rho_clip_max"])
    m = float(sizes.mean()) if len(sizes) else 1.0
    return max(1.0, design_effect(rho, m)), rho, m


def density(ll, deff):
    """Log-likelihood -> normalised density on the grid. deff deflates the
    log-likelihood BEFORE densification -- correlated hours are not
    independent rows, and skipping it tightens intervals by ~sqrt(deff)."""
    z = np.asarray(ll, dtype=float) / deff
    w = np.exp(z - z.max())
    return w / w.sum()


def moments(grid, w):
    m = float((grid * w).sum())
    return m, float(np.sqrt(((grid - m) ** 2 * w).sum()))


def mixture(a, b, wa=0.5):
    return wa * np.asarray(a) + (1 - wa) * np.asarray(b)


def marginal_score(ll_hold, w):
    """log integral p(y_holdout | eps) pi(eps) deps on the grid -- the
    comparison metric: predictive fit on unseen data, marginalised over
    epsilon, assuming only that `w` is a density on this grid."""
    ll = np.asarray(ll_hold, dtype=float)
    logw = np.log(np.maximum(np.asarray(w, dtype=float), 1e-300))
    return float(logsumexp(ll + logw) - logsumexp(logw))


def build_curves(d, cfg, model, grid, window):
    """Per-category (naive, controlled) curves and the deff used, on one split
    window. Shared by the fit and by the held-out scoring so the two cannot
    diverge in how they build rows."""
    pc = cfg["posterior"]["prior"]
    frame = population(split_frames(d, cfg)[window], cfg)
    rows = scored_rows(frame, cfg)
    out = {}
    for cat, g in rows.groupby("category"):
        deff, rho, m = deflation_deff(g, model, cfg)
        mu_ref = model.predict_mu_ref(g)
        lr = np.log((1 - g.total_discount.to_numpy())
                    / (1 - g.d_ref.to_numpy()))
        # de-meaned on the SAME cell the controlled arm profiles out, or the
        # reported share would describe a control that is not being applied
        cells = time_cell(g)
        within = lr - pd.Series(lr).groupby(cells).transform("mean").to_numpy()
        v = float(np.var(lr))
        sizes = pd.Series(cells).value_counts()
        out[str(cat)] = {
            "naive": curve(g, mu_ref, grid, False, cfg),
            "controlled": curve(g, mu_ref, grid, True, cfg),
            "deff": deff, "rho_eps_free": rho, "mean_rows_per_episode": m,
            "rows": int(len(g)), "episodes": int(g.episode_id.nunique()),
            "censored_share": float(g.censored.mean()),
            "log_ratio_sd": float(np.std(lr)),
            "distinct_discounts": int(g.total_discount.nunique()),
            "identifying_variation_share": (float(np.var(within) / v)
                                            if v > 0 else 0.0),
            "time_cells": int(len(sizes)),
            "median_rows_per_time_cell": float(sizes.median()),
        }
    return out


def unconstrained_argmax(d, cfg, model, lo, hi, n):
    """Each arm's unconstrained peak, searched PAST the policy bounds: a
    wrong-signed likelihood must be caught, not clipped and reported as
    measured."""
    wide = np.linspace(lo, max(1.0, hi), n)
    out = {}
    for cat, c in build_curves(d, cfg, model, wide, "train").items():
        out[cat] = {
            "naive": float(wide[int(np.argmax(c["naive"]))]),
            "controlled": float(wide[int(np.argmax(c["controlled"]))]),
        }
    return out


def fold_spread(d, cfg, model, grid, folds=3):
    """How far the estimate moves between disjoint slices of the train window
    -- MODEL uncertainty (confounding, imperfect mu_ref, non-stationarity)
    that within-sample curvature cannot see. Measured, not configured; feeds
    the std floor in `estimate`. Rationale: design 5.6."""
    frames = split_frames(d, cfg)
    train = population(frames["train"], cfg)
    if train.empty:
        return {}
    dates = np.array_split(np.sort(train.date.astype(str).unique()), folds)
    out = {}
    per_cat = {}
    for chunk in dates:
        if not len(chunk):
            continue
        sl = train[train.date.astype(str).isin(set(chunk))]
        rows = scored_rows(sl, cfg)
        for cat, g in rows.groupby("category"):
            if len(g) < 50:
                continue
            mu_ref = model.predict_mu_ref(g)
            n = grid[int(np.argmax(curve(g, mu_ref, grid, False, cfg)))]
            c = grid[int(np.argmax(curve(g, mu_ref, grid, True, cfg)))]
            per_cat.setdefault(str(cat), []).append((n + c) / 2.0)
    for cat, vals in per_cat.items():
        if len(vals) >= 2:
            out[cat] = {"folds": len(vals),
                        "estimates": [round(float(v), 4) for v in vals],
                        "spread": round(float(np.std(vals, ddof=1)), 4)}
    return out



def estimate(d, cfg, model, fast=False):
    """The prior, as a density per category. No constant anywhere in it.
    `fast` skips fold_spread (it only widens the std FLOOR, so it cannot move
    the calibration fixed point); the loop's final turn must run FULL."""
    pc = cfg["posterior"]["prior"]
    lo, hi = cfg["posterior"]["epsilon_min"], cfg["posterior"]["epsilon_max"]
    grid = np.linspace(lo, hi, pc["search_grid_size"])
    step = float(grid[1] - grid[0])
    sat = float(pc["own_information_saturation"])

    fit = build_curves(d, cfg, model, grid, "train")
    unconstrained = unconstrained_argmax(d, cfg, model, lo, hi,
                                         pc["search_grid_size"])
    folds = ({} if fast else
             fold_spread(d, cfg, model, grid,
                         int(pc.get("stability_folds", 3))))
    if not fit:
        raise SystemExit("no rows to profile epsilon on in the train window")

    # sign decided BEFORE the pool is built: the pool excludes wrong-signed
    # categories, or the fallback inherits the confound they were rejected for
    signs = {}
    for cat in fit:
        u = unconstrained.get(cat, {})
        peak = max(u.get("naive", lo), u.get("controlled", lo))
        signs[cat] = (peak >= -step, peak)

    # pooled by SUMMING log-likelihoods across usable categories: one measured
    # likelihood, not an average of summaries -- the anti-fallback-constant
    usable = [c for cat, c in fit.items()
              if not signs[cat][0] and c["log_ratio_sd"] > 1e-9]
    if usable:
        pooled_deff = float(np.mean([c["deff"] for c in usable]))
        pooled = mixture(
            density(np.sum([c["naive"] for c in usable], axis=0), pooled_deff),
            density(np.sum([c["controlled"] for c in usable], axis=0),
                    pooled_deff))
        pooled_basis = (f"{len(usable)} of {len(fit)} categories -- those with "
                        "a right-signed likelihood and some price variation")
    else:
        # nothing usable to pool: the uniform on the support is the measured
        # answer -- inventing a fallback constant is what this method removes
        pooled_deff = 1.0
        pooled = np.full(len(grid), 1.0 / len(grid))
        pooled_basis = (
            "UNIFORM ON THE SUPPORT -- no category had both a right-signed "
            "likelihood and price variation, so there is no measured pool. "
            "This extract does not identify elasticity; only exogenous price "
            "variation will.")
    pooled_mean, pooled_std = moments(grid, pooled)

    per_category, densities = {}, {}
    for cat, c in fit.items():
        # 50/50 arm mixture = the bracket generalised: sharp arms reproduce
        # midpoint +- half-gap, flat ones widen honestly (design 5.6)
        own = mixture(density(c["naive"], c["deff"]),
                      density(c["controlled"], c["deff"]))
        own_mean, own_std = moments(grid, own)

        # own-information weight: deflated likelihood span over the support,
        # saturating at `sat` (2.0 default -- the chi-square 95% cutoff)
        span = float(max(np.ptp(c["naive"]) / c["deff"],
                         np.ptp(c["controlled"]) / c["deff"]))
        w_own = float(min(1.0, span / sat)) if sat > 0 else 1.0

        # wrong sign rejects (the only reject): the peak was searched PAST the
        # bounds; the own density is discarded and the pooled one taken
        u = unconstrained.get(cat, {})
        wrong_sign, peak = signs[cat]

        w = pooled if wrong_sign else mixture(own, pooled, w_own)
        mean, std = moments(grid, w)

        # std is the WIDEST of three measured floors (density width, grid
        # step, fold_spread): curvature alone is sampling precision and
        # collapses to zero at production scale (design 5.6)
        fold = folds.get(cat, {})
        floor_reasons = {"density": std, "grid_resolution": step}
        if fold.get("spread") is not None:
            floor_reasons["fold_spread"] = float(fold["spread"])
        binding = max(floor_reasons, key=floor_reasons.get)
        std = float(floor_reasons[binding])
        densities[cat] = w
        per_category[cat] = {
            "mean": round(mean, 4), "std": round(std, 4),
            # which floor bound the std (semantics: design 5.6)
            "std_basis": binding,
            "std_candidates": {k: round(float(v), 4)
                               for k, v in floor_reasons.items()},
            "wrong_sign": bool(wrong_sign),
            "unconstrained_argmax": {k: round(v, 4) for k, v in u.items()},
            "own_mean": round(own_mean, 4), "own_std": round(own_std, 4),
            "own_information_weight": round(w_own, 4),
            "likelihood_span": round(span, 3),
            "epsilon_naive": round(float(grid[int(np.argmax(c["naive"]))]), 4),
            "epsilon_controlled": round(
                float(grid[int(np.argmax(c["controlled"]))]), 4),
            "rows": c["rows"], "episodes": c["episodes"],
            "deff": round(c["deff"], 3),
            "rho_eps_free": round(c["rho_eps_free"], 4),
            "censored_share": round(c["censored_share"], 4),
            "log_ratio_sd": round(c["log_ratio_sd"], 6),
            "distinct_discounts": c["distinct_discounts"],
            "identifying_variation_share": round(
                c["identifying_variation_share"], 4),
            # the full curve, so a suspicious prior can be inspected
            "density": [round(float(x), 8) for x in w],
        }
        if fold:
            per_category[cat]["stability"] = fold
        if wrong_sign:
            per_category[cat]["wrong_sign_note"] = (
                f"unconstrained peak at {peak:+.3f} -- demand rising with "
                "price. Own density discarded for the pooled one; usual cause "
                "is the legacy ramp, and only exogenous price variation fixes "
                "it.")
        if c["log_ratio_sd"] < 1e-9:
            per_category[cat]["no_price_variation"] = (
                "every scored row sits at one discount: epsilon is ABSENT "
                "from this likelihood, and the category takes the pooled "
                "prior. Only price variation can change that.")
    return grid, per_category, densities, {
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "pooled_deff": round(pooled_deff, 3),
        "pooled_basis": pooled_basis,
        "pooled_categories": sorted(
            c for c in fit if not signs[c][0] and fit[c]["log_ratio_sd"] > 1e-9),
        "pooled_density": [round(float(x), 8) for x in pooled],
    }


def holdout_comparison(d, cfg, model, grid, candidates):
    """Score every candidate prior on held-out data: log marginal predictive
    likelihood per row, deff-deflated -- narrow-and-wrong loses badly,
    too-wide loses mildly. `oracle` (hindsight-best eps, unreachable) and
    `uniform` (flat, the floor) bracket the result. Rationale: design 5.6."""
    window = "calib"                       # the out-of-train scoring window
    hold = build_curves(d, cfg, model, grid, window)
    if not hold:
        return {"window": window, "note": "held-out window has no scorable rows"}

    uniform = np.full(len(grid), 1.0 / len(grid))
    per_cat, totals = {}, {name: 0.0 for name in candidates}
    totals["uniform"], totals["oracle"] = 0.0, 0.0
    rows_total = 0

    for cat, c in hold.items():
        # both arms carry the held-out evidence; sum them for one score, the
        # same 50/50 weighting the prior itself is built with
        ll = (c["naive"] + c["controlled"]) / 2.0 / c["deff"]
        n = max(c["rows"], 1)
        rows_total += c["rows"]
        entry = {"rows": c["rows"], "episodes": c["episodes"],
                 "deff": round(c["deff"], 3),
                 "log_ratio_sd": round(c["log_ratio_sd"], 6)}
        for name, dens in candidates.items():
            w = dens.get(cat)
            if w is None:
                continue
            s = marginal_score(ll, w)
            entry[name] = round(s / n, 6)
            totals[name] += s
        entry["uniform"] = round(marginal_score(ll, uniform) / n, 6)
        entry["oracle"] = round(float(np.max(ll)) / n, 6)
        totals["uniform"] += marginal_score(ll, uniform)
        totals["oracle"] += float(np.max(ll))
        per_cat[cat] = entry

    names = [k for k in candidates if any(k in v for v in per_cat.values())]
    ranked = sorted(names, key=lambda k: totals[k], reverse=True)
    out = {
        "window": window,
        "rows_scored": rows_total,
        "metric": ("log marginal predictive likelihood per held-out row, "
                   "deff-deflated: log integral p(y|eps) pi(eps) deps. Higher "
                   "is better. `oracle` picks one epsilon with hindsight and "
                   "is unreachable; `uniform` is a flat prior over the support "
                   "and is the bar anything must clear to have added value."),
        "total_per_row": {k: round(totals[k] / max(rows_total, 1), 6)
                          for k in list(names) + ["uniform", "oracle"]},
        "ranking": ranked,
        "per_category": per_cat,
    }
    # oracle minus uniform = the ENTIRE value of knowing epsilon on this
    # window; any method gap is a share of it, so read this first
    available = (totals["oracle"] - totals["uniform"]) / max(rows_total, 1)
    out["information_available_per_row"] = round(available, 6)
    out["information_available_note"] = (
        "oracle minus uniform: the whole value of knowing epsilon on this "
        "window, in nats per row. Every method's score lies between those two. "
        "Read this FIRST -- a method gap that is a large share of a tiny "
        "number is still a tiny number.")
    # a sub-uniform candidate is worse than knowing nothing, and would let a
    # pairwise share exceed 100% -- flag it in its own words
    below_floor = sorted(k for k in names if totals[k] < totals["uniform"])
    out["worse_than_a_flat_prior"] = below_floor
    if below_floor:
        out["worse_than_a_flat_prior_note"] = (
            f"{', '.join(below_floor)} scored BELOW a flat prior over the "
            "support: on this held-out window that prior is worse than knowing "
            "nothing about epsilon. A confident wrong answer costs more than "
            "an honest wide one, and this is what that looks like in nats.")

    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        gain = (totals[a] - totals[b]) / max(rows_total, 1)
        # measured against the floor, not against the loser, so a sub-uniform
        # rival cannot manufacture a share above 100%
        headroom = (totals["oracle"] - max(totals[b], totals["uniform"])) \
            / max(rows_total, 1)
        share = gain / headroom if headroom > 0 else float("nan")
        out["method_gap_per_row"] = round(gain, 6)
        out["method_gap_share_of_available"] = (
            None if headroom <= 0 else round(share, 3))
        out["verdict"] = (
            f"{a} beats {b} by {gain:.6f} nats per held-out row "
            f"({totals[a] - totals[b]:.1f} total over {rows_total:,} rows). "
            + (f"{b} is itself below a flat prior, so this is not a close "
               f"contest between two reasonable answers -- it is one answer "
               f"and one that costs more than knowing nothing. "
               if b in below_floor else
               f"That is {share:.0%} of the {headroom:.6f} still on the table "
               f"above the loser. ")
            + ("The information available on this window (oracle minus "
               "uniform) is under 0.01 nats/row, so read any ranking here as "
               "weak evidence and decide on which prior is more honest about "
               "what it does not know. Compare their WIDTHS."
               if available < 0.01 else "Large enough to act on."))
    if ranked and totals[ranked[0]] < totals["uniform"]:
        out["warning"] = (
            "NO CANDIDATE BEATS A FLAT PRIOR. On this held-out window the data "
            "prefers knowing nothing about epsilon to anything either method "
            "concluded. That is a finding about the extract, not about the "
            "estimators: it says the train window carries no transferable "
            "information about price response.")
    return out
