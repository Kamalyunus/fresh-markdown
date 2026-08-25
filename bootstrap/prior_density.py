"""bootstrap.prior_density -- the elasticity prior AS the profile likelihood.

PRD section 9.5, method `profile_density`. Replaces two things at once, and
they are separable questions that were previously entangled:

  HOW THE CURVE IS COMPUTED.  A censored POISSON profile, not a censored NB.
        The Poisson quasi-MLE is consistent for the MEAN parameters even when
        the truth is negative binomial -- dispersion moves the standard errors,
        not the point estimate (Gourieroux, Monfort & Trognon 1984). Epsilon
        lives entirely in the mean, so `r` drops out of this step BY THEOREM
        rather than by measuring a sensitivity and calling it small. That is
        what removes the epsilon <-> r cycle: `fit_dispersion` still needs an
        elasticity, but this no longer needs a dispersion.

  WHAT THE CURVE BECOMES.  The whole curve, read as a density, instead of its
        argmax. The bracket kept two points and threw the shape away, so a
        sharp peak and a horizontal line produced the same kind of answer: a
        number with four decimals and a std derived from the gap between two
        point estimates, which knows nothing about curvature. Measured on the
        fixture, four of five categories had a likelihood span under 2.4 across
        the ENTIRE support and two were exactly constant -- and still reported
        -4.000 +- 0.400, because argmax of a constant array returns index 0.

THE FALLBACK CONSTANT IS GONE, and that is the point of the change. There is no
`fallback_mean`, no `fallback_std`, no `std_floor` in this path. A category the
data says nothing about does not get someone's guess; it gets, in order:

  1. its own density, which for a flat likelihood IS the uniform on the
     support -- mean -2.025, std 1.140 on [-4, -0.05], reached automatically
     rather than configured, and
  2. shrinkage toward the POOLED density fitted across every category, which is
     also measured. `own_information_weight` says how much of each was used.

The only external input left is `search_bounds`, and that is a policy statement
about the elasticity range the DP supports, not a guess about demand.

EXACTLY REPRODUCES THE BRACKET IN THE SHARP LIMIT. A 50/50 mixture of two point
masses at a and b has mean (a+b)/2 and std |a-b|/2 -- which is the bracket's
own `prior_mean` and `prior_std` formula. So this is a strict generalisation:
identical where the old procedure was justified, honestly wider where it was
not, and it needs no floor because the density's own width is the floor.

ROWS: every stocked hour, not just entry rows. The entry hour is BEFORE the
legacy ramp starts discounting, so entry rows sit at the opening discount --
which is the reference -- and carry almost no price variation. Measured on the
fixture, `log_ratio` sd went 0.00267 -> 0.02352 for MEAT and 0.0 -> 0.02448 for
SIDE DISH by scoring every hour. The extra rows are not free: they repeat the
same episode, so the log-likelihood is deflated by the design effect before any
interval is read from it.

WHY THE DEFLATION deff IS eps-FREE. It is computed from `units_sold - mu_ref`
residuals, with no elasticity applied, which is what keeps this acyclic --
using the working elasticity would put rho back inside the epsilon step.
`fit_dispersion` still re-fits rho properly at the estimated elasticity
afterwards, and that value is the one the posterior update uses; this one only
sets how much a repeated hour counts while reading the curve.

Usage is via `bootstrap.estimate_prior`, which dispatches on
`posterior.prior.method`.
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson, nbinom

from common.config import design_effect
from common import episodes
from bootstrap.prepare_data import population, split_frames


def scored_rows(frame, cfg, which):
    """The rows a profile is built on. `all_stocked_hours` is the default and
    the reason is measured -- see the module docstring."""
    f = frame.copy()
    f["censored"] = episodes.censored_hours(f)
    f = f[f.starting_inventory >= 1]
    if which == "entry":
        f = f.sort_values(["episode_id", "hour_of_day"]) \
             .groupby("episode_id").head(1)
        # a censored entry row is a one-hour episode; kept here because the
        # Poisson likelihood handles censoring without a dispersion parameter,
        # so there is nothing to drop them FOR any more
    return f.copy()


def hour_multipliers(mu, hour, k, censored):
    """Multiplicative hour fixed effects, profiled out by moment matching on
    uncensored rows. Family-agnostic: it matches first moments, so the same
    construction serves the Poisson and the NB profile."""
    frame = pd.DataFrame({"hour": hour, "k": k, "mu": mu, "cen": censored})
    unc = frame[~frame.cen]
    m = (unc.groupby("hour").k.sum()
         / unc.groupby("hour").mu.sum()).clip(0.05, 20.0)
    return frame.hour.map(m).fillna(1.0).to_numpy()


def loglik(eps, base, log_ratio, k, censored, floor, controlled, hour, const,
           r=None):
    """Censored log-likelihood at one elasticity.

    `r=None` gives the censored POISSON -- no dispersion parameter, which is
    the whole point. Pass an `r` array for the censored NB, kept so the two
    families can be compared on identical rows.
    """
    mu = np.clip(base * np.exp(eps * log_ratio), floor, None)
    if controlled:
        mu = mu * hour_multipliers(mu, hour, k, censored)
        mu = np.clip(mu, floor, None)
    if r is None:
        exact = k * np.log(mu) - mu - const
        tail = poisson.logsf(np.maximum(k, 1) - 1, mu)
    else:
        p = r / (r + mu)
        exact = const + r * np.log(p) + k * np.log1p(-p)
        tail = nbinom.logsf(np.maximum(k, 1) - 1, r, p)
    return float(np.sum(np.where(censored, tail, exact)))


def curve(g, mu_ref, grid, controlled, cfg, r=None):
    """The full log-likelihood over the grid. Everything the old `_estimate`
    computed and then discarded except one index."""
    k = g.units_sold.to_numpy()
    censored = g.censored.to_numpy()
    hour = g.hour_of_day.to_numpy()
    log_ratio = np.log((1 - g.total_discount.to_numpy())
                       / (1 - g.d_ref.to_numpy()))
    floor = cfg["pricing"]["demand_floor"]
    const = (gammaln(k + 1) if r is None
             else gammaln(k + r) - gammaln(r) - gammaln(k + 1))
    return np.array([loglik(e, mu_ref, log_ratio, k, censored, floor,
                            controlled, hour, const, r) for e in grid])


def deflation_deff(rows, model, cfg):
    """How much a repeated hour counts, computed WITHOUT an elasticity.

    rho here is the within-episode correlation of `units_sold - mu_ref`
    residuals. No price response is applied, which is what keeps the epsilon
    step free of the dispersion chain: `fit_dispersion`'s rho is fitted at the
    estimated elasticity and is the authoritative one for the posterior update,
    but reading it here would reintroduce the cycle this method exists to cut.

    Returns (deff, rho, mean rows per episode).
    """
    resid = rows.units_sold.to_numpy() - model.predict_mu_ref(rows)
    f = pd.DataFrame({"episode_id": rows.episode_id.to_numpy(), "resid": resid})
    sizes = f.groupby("episode_id").resid.size()
    sub = f[f.episode_id.isin(sizes[sizes >= 3].index)]
    total = float(sub.resid.var(ddof=1)) if len(sub) > 1 else 0.0
    between = (float(sub.groupby("episode_id").resid.mean().var(ddof=1))
               if sub.episode_id.nunique() > 1 else 0.0)
    rho = float(np.clip(between / total, 0.0, 0.95)) if total > 0 else 0.0
    m = float(sizes.mean()) if len(sizes) else 1.0
    return max(1.0, design_effect(rho, m)), rho, m


def density(ll, deff):
    """log-likelihood -> a normalised density on the grid.

    Deflated by deff first: without it a curve built from six correlated hours
    per episode claims six times the information it has, and every interval
    read off it is too tight by roughly sqrt(deff).
    """
    z = np.asarray(ll, dtype=float) / deff
    w = np.exp(z - z.max())
    return w / w.sum()


def moments(grid, w):
    m = float((grid * w).sum())
    return m, float(np.sqrt(((grid - m) ** 2 * w).sum()))


def mixture(a, b, wa=0.5):
    return wa * np.asarray(a) + (1 - wa) * np.asarray(b)


def normal_on_grid(grid, mean, std):
    """A (mean, std) pair rendered as a density on the same grid, so the old
    bracket prior and the new one are scored by identical arithmetic. Without
    this the comparison would be between a density and a summary of one."""
    z = -0.5 * ((np.asarray(grid) - mean) / max(std, 1e-6)) ** 2
    w = np.exp(z - z.max())
    return w / w.sum()


def marginal_score(ll_hold, w):
    """log integral p(y_holdout | eps) pi(eps) deps, on the grid.

    THE COMPARISON METRIC, and it is the right one: it asks how well a prior
    predicts data it never saw, marginalising over the elasticity rather than
    committing to a point. A prior that is narrow and wrong is punished hard, a
    prior that is too wide is punished mildly, and one centred where the
    held-out data likes epsilon with a width that matches its own uncertainty
    wins. Nothing about the shape of the prior is assumed -- only that it is a
    density on this grid.
    """
    ll = np.asarray(ll_hold, dtype=float)
    logw = np.log(np.maximum(np.asarray(w, dtype=float), 1e-300))
    return float(logsumexp(ll + logw) - logsumexp(logw))


def build_curves(d, cfg, model, grid, window, r_by_cat=None, r_pooled=None):
    """Per-category (naive, controlled) curves and the deff used, on one split
    window. Shared by the fit and by the held-out scoring so the two cannot
    diverge in how they build rows."""
    pc = cfg["posterior"]["prior"]
    frame = population(split_frames(d, cfg)[window], cfg)
    rows = scored_rows(frame, cfg, pc.get("rows", "all_stocked_hours"))
    out = {}
    for cat, g in rows.groupby("category"):
        r = None
        if r_by_cat is not None:
            r = np.full(len(g), float(r_by_cat.get(str(cat), r_pooled)))
        deff, rho, m = deflation_deff(g, model, cfg)
        mu_ref = model.predict_mu_ref(g)
        lr = np.log((1 - g.total_discount.to_numpy())
                    / (1 - g.d_ref.to_numpy()))
        within = lr - pd.Series(lr).groupby(
            g.hour_of_day.to_numpy()).transform("mean").to_numpy()
        v = float(np.var(lr))
        out[str(cat)] = {
            "naive": curve(g, mu_ref, grid, False, cfg, r),
            "controlled": curve(g, mu_ref, grid, True, cfg, r),
            "deff": deff, "rho_eps_free": rho, "mean_rows_per_episode": m,
            "rows": int(len(g)), "episodes": int(g.episode_id.nunique()),
            "censored_share": float(g.censored.mean()),
            "log_ratio_sd": float(np.std(lr)),
            "distinct_discounts": int(g.total_discount.nunique()),
            "identifying_variation_share": (float(np.var(within) / v)
                                            if v > 0 else 0.0),
        }
    return out


def unconstrained_argmax(d, cfg, model, lo, hi, n):
    """Where the likelihood's peak REALLY is, searched past the policy bounds.

    `search_bounds` is a statement about the elasticities the DP supports, not
    a belief about demand, so a peak outside it is clipped to the nearest bound
    and reported as if measured. That is how a category whose likelihood
    prefers POSITIVE elasticity -- demand rising with price, which is
    nonsensical -- comes back as a confident -0.05.

    It is not a rare pathology on this data. Measured on the fixture, three of
    five categories had their unconstrained optimum above zero, and the legacy
    ramp is the reason: deep discounts land on stock that is not selling, so
    conditional on a price-neutral `mu_ref` the deeper price reads as LOWER
    demand. The old bracket method rejected exactly this as `wrong_sign` -- its
    single unconditional reject -- and the density method dropped the check
    when it dropped the reject path. This puts it back.
    """
    wide = np.linspace(lo, max(1.0, hi), n)
    out = {}
    for cat, c in build_curves(d, cfg, model, wide, "train").items():
        out[cat] = {
            "naive": float(wide[int(np.argmax(c["naive"]))]),
            "controlled": float(wide[int(np.argmax(c["controlled"]))]),
        }
    return out


def fold_spread(d, cfg, model, grid, folds=3):
    """How far the estimate moves between disjoint slices of the train window.

    THE WIDTH A 125,000-ROW LIKELIHOOD CANNOT GIVE YOU. Curvature measures
    SAMPLING precision, and at production scale that is enormous -- a span of
    9,402 log-likelihood units across the grid collapses `exp(ll)` onto a
    single grid point and returns a prior std of ZERO. A prior of zero width is
    not a confident belief, it is a broken one: `bounded_step` can never move
    it, and the posterior is frozen before the first outcome arrives.

    What is actually uncertain here is not the sample, it is the MODEL. The
    history is confounded, `mu_ref` is imperfect, and demand is not stationary
    -- none of which the within-sample curvature can see, and all of which the
    bracket procedure existed to express. Re-fitting on disjoint time slices
    measures it directly: if the estimate moves 0.4 between the first half of
    the window and the second, then 0.4 is what is known, whatever the
    curvature claims.

    Measured, not configured, which is the whole point -- this is the width the
    data supports, not one somebody chose.
    """
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
        rows = scored_rows(sl, cfg, cfg["posterior"]["prior"].get(
            "rows", "all_stocked_hours"))
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


def estimate(d, cfg, model):
    """The prior, as a density per category. No constant anywhere in it."""
    pc = cfg["posterior"]["prior"]
    lo, hi = pc["search_bounds"]
    grid = np.linspace(lo, hi, pc["search_grid_size"])
    step = float(grid[1] - grid[0])
    sat = float(pc["own_information_saturation"])

    fit = build_curves(d, cfg, model, grid, "train")
    unconstrained = unconstrained_argmax(d, cfg, model, lo, hi,
                                         pc["search_grid_size"])
    folds = fold_spread(d, cfg, model, grid,
                        int(pc.get("stability_folds", 3)))
    if not fit:
        raise SystemExit("no rows to profile epsilon on in the train window")

    # THE SIGN IS DECIDED BEFORE THE POOL IS BUILT, because a pool that
    # includes the categories it is meant to rescue inherits exactly the
    # confound they were rejected for. Summing a backwards likelihood into the
    # fallback makes the fallback backwards too, quietly, and it still reads as
    # a measurement.
    signs = {}
    for cat in fit:
        u = unconstrained.get(cat, {})
        peak = max(u.get("naive", lo), u.get("controlled", lo))
        signs[cat] = (peak >= -step, peak)

    # POOLED ACROSS CATEGORIES, by summing log-likelihoods -- categories are
    # independent samples of the same question, so this is one likelihood over
    # all of them, not an average of summaries. It is what a category with no
    # usable variation of its own borrows, and it is MEASURED, which is the
    # whole difference from a fallback constant.
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
        # NOTHING LEFT TO POOL. Every category is either backwards or flat, so
        # there is no measured fallback to fall back TO, and inventing one
        # would be the constant this method exists to remove. The uniform on
        # the support is what the data supports: it says, correctly, that this
        # extract does not identify elasticity at all.
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
        # THE 50/50 ARM MIXTURE IS THE BRACKET, GENERALISED. The old procedure
        # took the midpoint of two argmaxes and half their gap; a 50/50 mixture
        # of two POINT MASSES at a and b has exactly mean (a+b)/2 and std
        # |a-b|/2. So where both arms are sharp this reproduces the old answer,
        # and where they are not it widens instead of reporting false
        # precision. The directional argument -- naive too elastic, controlled
        # toward zero, truth between -- is unchanged and still what justifies
        # weighting them equally.
        own = mixture(density(c["naive"], c["deff"]),
                      density(c["controlled"], c["deff"]))
        own_mean, own_std = moments(grid, own)

        # HOW MUCH OWN INFORMATION THERE IS, in the units the cutoff is in:
        # log-likelihood units across the whole support. `sat` is the
        # saturation point, 2.0 by default, which is the chi-square 95% cutoff
        # -- a category that discriminates at least as much as its own
        # significance threshold stands on its own data.
        span = float(max(np.ptp(c["naive"]) / c["deff"],
                         np.ptp(c["controlled"]) / c["deff"]))
        w_own = float(min(1.0, span / sat)) if sat > 0 else 1.0

        # WRONG SIGN REJECTS, and it is the only thing that does. The peak is
        # searched PAST the policy bounds, so a likelihood that prefers
        # positive elasticity -- demand rising with price -- is caught instead
        # of being clipped to the nearest bound and reported as measured. Its
        # own density is not used at all: the number is not weak, it is
        # nonsensical, and no part of a curve peaked in the wrong direction
        # belongs in a prior. The category takes the POOLED density, which is
        # still measured, and is listed at the top of the artifact.
        u = unconstrained.get(cat, {})
        wrong_sign, peak = signs[cat]

        w = pooled if wrong_sign else mixture(own, pooled, w_own)
        mean, std = moments(grid, w)

        # THE WIDTH A SHARP LIKELIHOOD CANNOT GIVE. Curvature is SAMPLING
        # precision; at 125,000 rows a span of 9,402 log-likelihood units puts
        # every grid point but one at exp(-59) and returns a std of ZERO. A
        # zero-width prior is not confidence, it is a frozen posterior --
        # `bounded_step` cannot move it and no outcome ever will.
        #
        # What is uncertain is the MODEL, not the sample: confounded history, an
        # imperfect mu_ref, non-stationary demand. `fold_spread` measures that
        # by re-fitting on disjoint slices of the train window, and the grid
        # step bounds it from below because nothing finer than one grid cell was
        # ever resolved. Both are MEASURED -- neither is a floor someone chose,
        # which is what the fallback constant was.
        fold = folds.get(cat, {})
        floor_reasons = {"density": std, "grid_resolution": step}
        if fold.get("spread") is not None:
            floor_reasons["fold_spread"] = float(fold["spread"])
        binding = max(floor_reasons, key=floor_reasons.get)
        std = float(floor_reasons[binding])
        densities[cat] = w
        per_category[cat] = {
            "mean": round(mean, 4), "std": round(std, 4),
            # WHICH FLOOR IS DOING THE WORK. `density` means the curve's own
            # width is the widest of the three and nothing was imposed;
            # `fold_spread` means the estimate is unstable across the window
            # and that instability is the honest width; `grid_resolution` means
            # the likelihood is sharper than the grid can express and the prior
            # is as tight as this machinery can defend.
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
            # a density is a curve, and a reader checking a suspicious prior
            # should be able to see the curve rather than infer it from two
            # moments
            "density": [round(float(x), 8) for x in w],
        }
        if fold:
            per_category[cat]["stability"] = fold
        if wrong_sign:
            per_category[cat]["wrong_sign_note"] = (
                f"the likelihood's unconstrained peak is at {peak:+.3f} -- at "
                "or above zero, which says demand RISES with price. That is "
                "nonsensical rather than weak, so this category's own density "
                "is discarded and it takes the pooled one. The usual cause is "
                "the legacy ramp: deep discounts land on stock that is not "
                "selling, so conditional on a price-neutral mu_ref the deeper "
                "price reads as lower demand. Only exogenous price variation "
                "fixes it -- see pricing.explore.")
        if c["log_ratio_sd"] < 1e-9:
            per_category[cat]["no_price_variation"] = (
                "every scored row sits at the same discount, so epsilon is "
                "ABSENT from this category's likelihood -- not weakly "
                "identified, absent. Its own density is the uniform on the "
                "support by construction, and the prior it gets is the pooled "
                "one. Nothing about this category's own elasticity has been "
                "measured, and no estimator can change that: only price "
                "variation can.")
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
    """Score every candidate prior on data it never saw.

    THE ONLY QUESTION THAT SETTLES "IS THIS BETTER". Neither prior can be
    judged from the inside -- "is -1.61 +- 1.09 better than -1.00 +- 0.60" has
    no answer by inspection, and that argument ran for a long time. This one
    does: fit on train, then ask how well each prior predicts the HELD-OUT
    window, marginalising over epsilon rather than committing to a point.

        score = log integral p(y_hold | eps) pi(eps) deps

    A prior that is narrow and wrong loses badly, one that is too wide loses a
    little, and one centred where the held-out data likes epsilon with a width
    matching its own uncertainty wins. Reported per row so the number is
    comparable across categories of different size.

    TWO REFERENCE POINTS bracket the result and stop it being read as better
    than it is:

      oracle    the single best epsilon chosen WITH hindsight on the held-out
                data. Unreachable, and the gap to it is what remains on the
                table.
      uniform   a flat prior over the whole support. Anything that cannot beat
                this has added nothing at all.

    The held-out curves are deflated by the held-out window's own deff, for the
    same reason the fit is: repeated hours from one episode are not
    independent observations, and scoring as if they were would reward a prior
    for matching correlated noise.
    """
    window = cfg["posterior"]["prior"]["holdout_window"]
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
    # HOW MUCH IS ON THE TABLE AT ALL, read before who won. The oracle picks
    # the best single epsilon with hindsight and the uniform knows nothing, so
    # the gap between them is the ENTIRE value of knowing this extract's
    # elasticity on this window. Any method comparison is a fight over a share
    # of that gap, and if the gap is tiny the fight does not matter however it
    # comes out -- which is the thing an argument about prior methods will not
    # tell you from the inside.
    available = (totals["oracle"] - totals["uniform"]) / max(rows_total, 1)
    out["information_available_per_row"] = round(available, 6)
    out["information_available_note"] = (
        "oracle minus uniform: the whole value of knowing epsilon on this "
        "window, in nats per row. Every method's score lies between those two. "
        "Read this FIRST -- a method gap that is a large share of a tiny "
        "number is still a tiny number.")
    # WHICH CANDIDATES CLEAR THE FLOOR AT ALL. A prior that scores BELOW the
    # uniform is not merely weaker than its rival -- it is worse than knowing
    # nothing, which is a stronger statement than any pairwise gap and has to
    # be said in those words. It also breaks the share arithmetic: measured
    # against a range that runs from uniform to oracle, a gap involving a
    # sub-uniform candidate can exceed 100% and read as decisive when what it
    # actually shows is one candidate falling off the bottom.
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
