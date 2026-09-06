"""fit.prior_density -- the elasticity prior AS the profile likelihood.

Censored Poisson profile (QMLE consistent for the mean, so no dispersion
parameter and no epsilon <-> r cycle), read whole-curve as a density; a 50/50
naive/controlled mixture generalises the old midpoint bracket. No fallback
constant: uninformative categories shrink to a pooled density measured on the
right-signed categories. Spec: docs/design.md 5.6; history: docs/learnings.md.
"""


import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from common.config import design_effect, intraclass_correlation
from common import episodes
from fit.prepare_data import population, split_frames


def scored_rows(frame):
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
    Thin cells (under `min_rows` uncensored rows) fall back to 1.0 -- a small
    cell absorbs the price response itself; returns (multipliers, fallback
    share). `cell` is integer codes (or labels, factorised here): a caller
    profiling a whole grid factorises once and this runs on bincount."""
    codes = np.asarray(cell)
    if codes.dtype.kind not in "iu":
        codes = pd.factorize(codes)[0]
    n_cells = int(codes.max()) + 1 if len(codes) else 0
    unc = ~np.asarray(censored, dtype=bool)
    if not unc.any():
        return np.ones(len(codes)), 0.0
    size = np.bincount(codes[unc], minlength=n_cells)
    sum_k = np.bincount(codes[unc], weights=np.asarray(k, dtype=float)[unc],
                        minlength=n_cells)
    sum_mu = np.bincount(codes[unc], weights=np.asarray(mu, dtype=float)[unc],
                         minlength=n_cells)
    fitted = size >= max(int(min_rows), 1)
    m = np.ones(n_cells)
    m[fitted] = np.clip(sum_k[fitted] / sum_mu[fitted], 0.05, 20.0)
    return m[codes], float(np.mean(~fitted[codes]))


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
    cell = pd.factorize(time_cell(g))[0]      # once, not per grid point
    min_cell = int(pc.get("min_rows_per_time_cell", 1))
    log_ratio = np.log((1 - g.total_discount.to_numpy())
                       / (1 - g.d_ref.to_numpy()))
    floor = cfg["pricing"]["demand_floor"]
    const = gammaln(k + 1)
    return np.array([loglik(e, mu_ref, log_ratio, k, censored, floor,
                            controlled, cell, const, min_cell)
                     for e in grid])


def deflation_deff(rows, mu_ref, cfg):
    """Design effect for the prior's likelihood -- eps-FREE, keeping the
    epsilon step out of the dispersion chain (`fit_dispersion` owns the
    fitted rho). Returns (deff, rho, mean rows per cluster).

    Clustered on SKU x FC, NOT on episode: these are ENTRY rows (rule 7), one
    per episode, so a within-episode ICC is 1.0 by construction. The
    correlation that does exist between entry rows is the same unit recurring
    across days -- the unit the pilot's outcomes recur on.

    ONE unit set for both moments: rho and the mean cluster size are both
    taken over the units with at least `rho_min_hours_per_episode` rows
    (singletons carry no within-unit correlation and would only shrink `m`
    against a rho they did not enter). No such unit: deff is 1.0.
    """
    resid = rows.units_sold.to_numpy() - np.asarray(mu_ref, dtype=float)
    unit = (rows.sku_id.astype(str) + "|" + rows.fc.astype(str)).to_numpy()
    f = pd.DataFrame({"unit": unit, "resid": resid})
    sizes = f.groupby("unit").resid.size()
    kept = sizes[sizes >= cfg["assurance"]["rho_min_hours_per_episode"]]
    sub = f[f.unit.isin(kept.index)]
    rho = intraclass_correlation(sub.resid, sub.unit,
                                 cfg["dispersion"]["rho_clip_max"])
    m = float(kept.mean()) if len(kept) else 1.0
    return design_effect(rho, m), rho, m     # design_effect floors at 1.0


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
    frame = population(split_frames(d, cfg)[window], cfg)
    rows = scored_rows(frame)
    out = {}
    for cat, g in rows.groupby("category"):
        mu_ref = model.predict_mu_ref(g)
        deff, rho, m = deflation_deff(g, mu_ref, cfg)
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


def extend_below(grid, margin):
    """`grid` with points appended below its first value, at its own step,
    down to (at least) `margin` below it."""
    step = float(grid[1] - grid[0])
    below = grid[0] - step * np.arange(1, int(np.ceil(margin / step)) + 1)
    return np.concatenate([below[::-1], grid])


def extend_above(grid, margin):
    """`grid` with points appended above its last value, at its own step,
    up to (at least) `margin` above it."""
    step = float(grid[1] - grid[0])
    above = grid[-1] + step * np.arange(1, int(np.ceil(margin / step)) + 1)
    return np.concatenate([grid, above])


def search_grid(grid, cfg):
    """The lattice the unconstrained peak is searched on: the FIT grid
    itself, extended at ITS step past both policy bounds -- to at least +1.0
    above (a positive optimum must be visible; not a tunable) and
    `unconstrained_search_below` past epsilon_min below. Returns (wide, sl):
    `wide[sl]` is exactly `grid`, so one set of curves over `wide` serves
    both the fit and the boundary search. Rationale: design 5.6."""
    hi = float(grid[-1])
    margin = float(cfg["posterior"]["prior"]["unconstrained_search_below"])
    above = extend_above(grid, max(1.0, hi) - hi)
    wide = extend_below(above, margin)
    start = len(wide) - len(above)           # points prepended below lo
    return wide, slice(start, start + len(grid))


# Float noise on a FLAT curve (no price variation) can put its argmax on an
# edge point by ~1e-12; an edge "wins" only by more than this RELATIVE margin.
# A tolerance on floating-point noise, not a tunable.
_FLAT_TOL_REL = 1e-9


def unconstrained_peaks(curves, wide, fit_start):
    """Each arm's unconstrained peak, searched PAST both policy bounds: a
    wrong-signed likelihood must be caught, not clipped and reported as
    measured, and one running off the lower bound is a boundary solution
    (rule 3), not an estimate. `wide[fit_start]` is epsilon_min; a peak at
    or below epsilon_min + one grid step is "at the bound"."""
    interior = np.arange(len(wide)) > fit_start + 1
    out = {}
    for cat, c in curves.items():
        peaks, pinned = {}, []
        for arm in ("naive", "controlled"):
            ll = np.asarray(c[arm])
            i = int(np.argmax(ll))
            peaks[arm] = float(wide[i])
            # pinned = the edge STRICTLY beats every interior point. A flat
            # likelihood (no price variation) argmaxes at the first grid
            # point too, and that is absence of information, not a run-off.
            best_in = float(ll[interior].max())
            if not interior[i] and \
                    ll[i] > best_in + _FLAT_TOL_REL * max(1.0, abs(best_in)):
                pinned.append(arm)
        out[cat] = {**peaks, "lower_pinned": pinned}
    return out


def fold_spread(d, cfg, model, grid, folds=3):
    """How far the estimate moves between disjoint slices of the train window
    -- MODEL uncertainty (confounding, imperfect mu_ref, non-stationarity)
    that within-sample curvature cannot see. Measured, not configured; feeds
    the std floor in `estimate`. Folds are cut by EPISODE (opening date,
    rule 15): a row-level cut put a midnight-straddling episode in two folds
    and scored its 00:00 row as an entry row. Rationale: design 5.6."""
    frames = split_frames(d, cfg)
    train = population(frames["train"], cfg)
    if train.empty:
        return {}
    opened = episodes.opening_dates(train)
    dates = np.array_split(np.sort(opened.unique()), folds)
    out = {}
    per_cat = {}
    for chunk in dates:
        if not len(chunk):
            continue
        sl = episodes.window_slice(train, chunk[0], chunk[-1], opened=opened)
        rows = scored_rows(sl)
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

    # ONE pass over the wide lattice: the fit range is sliced out of it, the
    # rest is the unconstrained search past both bounds
    wide, fit_sl = search_grid(grid, cfg)
    curves = build_curves(d, cfg, model, wide, "train")
    fit = {cat: {**c, "naive": c["naive"][fit_sl],
                 "controlled": c["controlled"][fit_sl]}
           for cat, c in curves.items()}
    peaks = unconstrained_peaks(curves, wide, fit_sl.start)
    folds = ({} if fast else
             fold_spread(d, cfg, model, grid,
                         int(pc.get("stability_folds", 3))))
    if not fit:
        raise SystemExit("no rows to profile epsilon on in the train window")

    # sign and boundary decided BEFORE the pool is built: the pool excludes
    # wrong-signed and lower-boundary categories, or the fallback inherits
    # the confound (or the run-off) they were rejected for
    rejected = {}
    for cat, c in fit.items():
        u = peaks[cat]
        pinned = sorted(u["lower_pinned"])
        rejected[cat] = {
            # wrong sign: the unconstrained peak at or above zero -- within
            # one grid step of it, since the lattice need not hold zero
            "wrong_sign": max(u["naive"], u["controlled"]) >= -step,
            # an arm whose likelihood is maximised at or below epsilon_min +
            # one grid step ran off the support: a boundary solution, not
            # an estimate (rule 3). epsilon_min MAY be widened when this
            # fires; epsilon_max never (design 5.6).
            "boundary": "lower" if pinned else None,
            "pinned_arms": pinned,
            "no_price_variation": c["log_ratio_sd"] < 1e-9,
        }

    def takes_pool(cat):
        r = rejected[cat]
        return r["wrong_sign"] or r["boundary"] is not None

    def in_pool(cat):
        return not takes_pool(cat) and not rejected[cat]["no_price_variation"]

    # pooled by SUMMING log-likelihoods across usable categories: one measured
    # likelihood, not an average of summaries -- the anti-fallback-constant
    usable = [c for cat, c in fit.items() if in_pool(cat)]
    if usable:
        pooled_deff = float(np.mean([c["deff"] for c in usable]))
        pooled = mixture(
            density(np.sum([c["naive"] for c in usable], axis=0), pooled_deff),
            density(np.sum([c["controlled"] for c in usable], axis=0),
                    pooled_deff))
        pooled_basis = (f"{len(usable)} of {len(fit)} categories -- those with "
                        "a right-signed, interior likelihood and some price "
                        "variation")
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

        # wrong sign and a lower-boundary peak both reject: the peak was
        # searched PAST the bounds; the own density is discarded for the pool
        u, r = peaks[cat], rejected[cat]
        w = pooled if takes_pool(cat) else mixture(own, pooled, w_own)
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
            "wrong_sign": bool(r["wrong_sign"]),
            # None, or "lower": an arm's unconstrained peak sits at or below
            # epsilon_min + step (the sign bound above is `wrong_sign`)
            "boundary": r["boundary"],
            "unconstrained_argmax": {k: round(u[k], 4)
                                     for k in ("naive", "controlled")},
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
        if r["wrong_sign"]:
            per_category[cat]["wrong_sign_note"] = (
                f"unconstrained peak at {max(u['naive'], u['controlled']):+.3f}"
                " -- demand rising with price. Own density discarded for the "
                "pooled one; usual cause is the legacy ramp, and only "
                "exogenous price variation fixes it.")
        if r["boundary"] is not None:
            per_category[cat]["boundary_note"] = (
                f"{' and '.join(r['pinned_arms'])} arm peak at or below "
                f"epsilon_min {lo:+.2f} (+ one grid step): the likelihood ran "
                "off the lower end of the support, so this is a boundary "
                "solution, not an estimate (rule 3). Own density discarded "
                "for the pooled one and the category left OUT of the pool. "
                "Owner: consider widening posterior.epsilon_min -- the lower "
                "bound may be widened when a fit pins there; epsilon_max never.")
        if r["no_price_variation"]:
            per_category[cat]["no_price_variation"] = (
                "every scored row sits at one discount: epsilon is ABSENT "
                "from this likelihood, and the category takes the pooled "
                "prior. Only price variation can change that.")
    return grid, per_category, densities, {
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "pooled_deff": round(pooled_deff, 3),
        "pooled_basis": pooled_basis,
        "pooled_categories": sorted(c for c in fit if in_pool(c)),
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
