"""bootstrap.estimate_prior -- elasticity prior via the bracket procedure.

PRD section 9.5. History cannot point-identify elasticity; it can set-identify
it. Two estimates per category, both via the censored likelihood of section
13.2 with frozen mu_ref as baseline:

    epsilon_naive       no hour control -- absorbs the evening demand lift
                        into price -> biased too elastic (too negative)
    epsilon_controlled  hour fixed effects -- removes most surviving price
                        variation with the confound -> biased toward zero

    prior_mean = midpoint of [epsilon_controlled, epsilon_naive]
    prior_std  = max(|epsilon_naive - epsilon_controlled| / 2, std_floor)

Both estimates search the FULL posterior support [epsilon_min, epsilon_max];
a bound tighter than epsilon_min is a defect (the phase-0 run pinned five
categories at a -1.5 bound).

ACCEPTANCE IS PER CATEGORY, AND ONLY THE SIGN REJECTS UNCONDITIONALLY.
PRD 9.5 was amended for this (owner, 2026-08-24); the previous rule was
all-or-nothing with boundary and orientation both fatal, and it was throwing
away measured information in favour of a constant. On the production extract
BAKERY & PASTRY passed every check on 21,484 rows and was still overwritten
with the fallback because two OTHER categories failed.

The principle: a measured bracket beats an assumption. `fallback_std: 0.60`
is a config constant -- nothing measured produced it -- so "the bracket is
wider, therefore worse" compares a measurement against a number someone
chose. Judged by the PRD's own criterion, "not confidently wrong", the
fallback was the worse of the two for every category on that extract: it puts
the measured deepening bar (median 2.429) at 2.38 sigma, against 0.08-0.75
sigma under the brackets.

    wrong sign   ALWAYS rejects. A non-negative endpoint says demand rises
                 with price -- nonsensical rather than weak, and no midpoint
                 of it means anything.
    boundary     reported, does NOT reject. The ordering survives; only the
                 lower end is truncated at the search bound. Still a bracket,
                 just a conservative one -- the midpoint understates |eps| and
                 the std understates the width.
    inverted     FLAGGED, not rejected (owner, 2026-08-24). The midpoint is
                 used because it is the best reading the data supports and the
                 alternative is a constant. The caveat is real and the flag
                 carries it: the bracket's argument is DIRECTIONAL, and the
                 midpoint is set-identified only because
                 naive <= eps_true <= controlled. Reverse the ordering and it
                 is the centre of two measurements rather than a bracket on
                 the truth -- weaker evidence than an upright category's,
                 though the artifact reports both the same way. Listed at the
                 top of the artifact as `inverted_categories`.

READ `identifying_variation_share` BEFORE ACTING ON AN INVERSION. The
controlled fit profiles out hour effects, so it is identified by WITHIN-HOUR
price variation and nothing else. Near zero, it is noise that can land
anywhere -- including past the naive estimate -- and the inversion says
nothing about the category; pool hours or coarsen the control. Ample, and the
control IS identified, so the inversion is a real result: the evening-lift
confound story does not hold for that category.

All three are configurable (`reject_wrong_sign`, `reject_boundary_solutions`,
`reject_orientation_violations`).

Both estimates use ENTRY-HOUR rows only. Identifying variation is same-hour
cross-episode, never adjacent-hour within-episode (section 9.5): under the
legacy ramp a row at a deep discount exists precisely because earlier hours
did not sell, so within-episode rows carry a survivorship confound that
biases the estimate toward zero. Entry rows carry the cross-episode
variation (different start hours, cost-floor truncation) without it.

Acceptance rests on orientation and sufficiency, not correctness. On any
failed check the category (or the whole prior) falls back per config, and the
rejection is recorded -- rejection is an acceptable outcome.

Usage:
    python3 -m bootstrap.estimate_prior --input data/prepared.parquet
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom

from common.config import load_config
from common import episodes
from common.provenance import stamp
from bootstrap.prepare_data import population, split_frames
from bootstrap.train_baseline import BaselineModel
from bootstrap import prior_density


def _censored_loglik(eps, mu_ref, log_ratio, k, r, censored, lgamma_const,
                     hour_mult=None):
    """Sum of censored NB log-likelihood at elasticity eps.

    lgamma_const = gammaln(k + r) - gammaln(r) - gammaln(k + 1), precomputed
    (it does not depend on eps).
    """
    mu = mu_ref * np.exp(eps * log_ratio)
    if hour_mult is not None:
        mu = mu * hour_mult
    p = r / (r + mu)
    exact = lgamma_const + r * np.log(p) + k * np.log1p(-p)
    ll = np.where(censored,
                  nbinom.logsf(np.maximum(k, 1) - 1, r, p),
                  exact)
    return float(np.sum(ll))


def _profile_hour_multipliers(eps, g, mu_ref, log_ratio):
    """Multiplicative hour fixed effects, profiled out at each eps by moment
    matching on uncensored rows."""
    mu = mu_ref * np.exp(eps * log_ratio)
    frame = pd.DataFrame({"hour": g.hour_of_day.to_numpy(),
                          "k": g.units_sold.to_numpy(), "mu": mu,
                          "cen": g.censored.to_numpy()})
    unc = frame[~frame.cen]
    m = (unc.groupby("hour").k.sum() / unc.groupby("hour").mu.sum()).clip(0.05, 20.0)
    return frame.hour.map(m).fillna(1.0).to_numpy()


def _estimate(g, mu_ref, grid, controlled):
    k = g.units_sold.to_numpy()
    r = g.r.to_numpy()
    censored = g.censored.to_numpy()
    log_ratio = np.log((1 - g.total_discount.to_numpy()) / (1 - g.d_ref.to_numpy()))
    lgamma_const = gammaln(k + r) - gammaln(r) - gammaln(k + 1)

    lls = []
    for eps in grid:
        hm = _profile_hour_multipliers(eps, g, mu_ref, log_ratio) if controlled else None
        lls.append(_censored_loglik(eps, mu_ref, log_ratio, k, r, censored,
                                    lgamma_const, hm))
    return float(grid[int(np.argmax(lls))])


def _reference_r(rows, cfg, model, pc):
    """Dispersion for the bracket's likelihood, PER CATEGORY.

    Returns `(by_category, pooled, basis)`. Each category's bracket is scored
    at its OWN r; the pooled fit is the fallback for a category with too few
    entry rows to fit one.

    Fitted on THE ROWS THE BRACKET SCORES -- entry rows, censored ones already
    dropped -- at the fallback elasticity, using the same `fit_dispersion.fit_r`
    the frozen artifact uses. A seed, not the artifact: `fit_dispersion` runs
    after this module and fits per subcategory at each category's real prior
    mean.

    WHY PER CATEGORY AND NOT ONE POOLED SCALAR (owner, 2026-08-24). Dispersion
    is not a property of the extract, it is a property of the category, and the
    spread is far wider than the pooled fit's own imprecision. Measured on the
    fixture's entry rows, at the pooled value 8.04:

        SEAFOOD      1.60    (Pearson 1.685 -- genuinely overdispersed)
        VEGETABLE    5.39
        FRUIT       14.12
        MEAT        28.71
        SIDE DISH   49.9998  (pinned at the search bound)

    That is 5x below to 6x above pooled, where the sensitivity band this
    approximation was justified over was +-2x. Scoring SEAFOOD's bracket at 8.04
    is not a small error in a nuisance parameter, it is the wrong likelihood.

    THE WINDOW MATTERS MORE THAN THE ROW TYPE, which is why these are fitted
    here rather than borrowed from `r_lookup.json`: that artifact describes the
    CALIBRATION window and this likelihood sums over TRAIN entry rows. It also
    does not exist yet in the current step order.

    A FIT AT THE SEARCH BOUND IS NOT AN ESTIMATE -- SIDE DISH above is
    under-dispersed relative to Poisson, so the MLE runs to the ceiling. Those
    are clamped at `dispersion.clamp_percentile` of the converged set, the same
    treatment and the same config key `fit_dispersion` uses, so the two modules
    cannot drift in how they handle a boundary.

    Explicit config wins if set -- for reproducing an old run, or pinning the
    bootstrap against a known value -- and then applies to every category.
    """
    if pc.get("reference_r") is not None:
        return {}, float(pc["reference_r"]), "config reference_r (pinned)"

    from bootstrap.fit_dispersion import fit_r

    eps0 = float(pc["fallback_mean"])
    bounds = cfg["dispersion"]["r_search_bounds"]
    min_rows = int(pc["reference_r_min_rows"])

    def fit(g):
        mu_ref = model.predict_mu_ref(g)
        ratio = (1 - g.total_discount.to_numpy()) / (1 - g.d_ref.to_numpy())
        mu = np.clip(mu_ref * ratio ** eps0, cfg["pricing"]["demand_floor"], None)
        return fit_r(g.units_sold.to_numpy(), mu,
                     g.censored.to_numpy(), bounds)

    pooled, ok = fit(rows)
    if not ok:
        # the pooled fit failing at all means something is wrong with the
        # window, and a silent midpoint would hide it
        raise SystemExit(
            "could not fit a reference dispersion on the bracket's own rows. "
            "Set posterior.prior.reference_r explicitly, or check the train "
            "window has usable entry rows.")

    by_cat, thin = {}, {}
    for cat, g in rows.groupby("category"):
        if len(g) < min_rows:
            thin[str(cat)] = int(len(g))
            continue
        r, cat_ok = fit(g)
        if cat_ok:
            by_cat[str(cat)] = r
        else:
            thin[str(cat)] = int(len(g))

    # same clamp as fit_dispersion: hold down boundary solutions, preserve low
    # values (a low r is real overdispersion and the conservative direction)
    cap = float(np.percentile(list(by_cat.values()) + [pooled],
                             cfg["dispersion"]["clamp_percentile"] * 100))
    by_cat = {k: min(v, cap) for k, v in by_cat.items()}
    pooled = min(pooled, cap)

    basis = (f"fitted per category on {len(rows)} entry rows at fallback "
             f"elasticity {eps0}, clamped at {cap:.4f}; pooled "
             f"{pooled:.4f} where a category has fewer than {min_rows} rows")
    if thin:
        basis += f" (pooled for {', '.join(sorted(thin))})"
    return by_cat, float(pooled), basis


def _episodes_per_week(d, cfg):
    """Volume per category on the train window, which decides cell structure in
    `pricing.posterior.initialise`. Shared by both methods -- it is a property
    of the population, not of how epsilon was estimated."""
    train = population(split_frames(d, cfg)["train"], cfg)
    weeks = max(train.date.astype(str).map(
        lambda x: pd.Timestamp(x).to_period("W")).nunique(), 1)
    per = train.groupby("category")["episode_id"].nunique() / weeks
    return {str(k): round(float(v), 1) for k, v in per.items()}


def estimate_prior_density(d, cfg):
    """PRD 9.5 method `profile_density` -- the prior IS the profile likelihood.

    See `bootstrap.prior_density` for why. Two changes from the bracket, and
    they are separable: the curve is a censored POISSON profile, so `r` leaves
    the epsilon step by theorem rather than by a measured sensitivity; and the
    whole curve becomes the prior instead of its argmax, so there is no
    fallback constant, no std floor, and a category the data says nothing about
    degrades to the uniform on the support rather than to someone's guess.

    The held-out comparison runs in the same pass and scores BOTH methods, so
    the artifact carries the evidence for its own method rather than asserting
    it. Nothing here decides the prior-acceptance gate; that is still a human
    reading this file.
    """
    pc = cfg["posterior"]["prior"]
    model = BaselineModel(cfg)
    grid, per_category, densities, pooled = prior_density.estimate(d, cfg, model)

    # THE COMPARISON IS AGAINST THE METHOD BEING REPLACED, not against nothing.
    # Both priors are rendered as densities on the same grid and scored by the
    # same arithmetic -- the bracket's (mean, std) as a normal, this method's
    # as itself -- so the difference is the method and not the accounting.
    candidates = {"profile_density": densities}
    bracket_note = None
    try:
        bracket = estimate_prior_bracket(d, cfg)
        candidates["bracket"] = {
            c: prior_density.normal_on_grid(grid, v["mean"], v["std"])
            for c, v in bracket["per_category"].items()}
    except Exception as exc:                       # noqa: BLE001
        # a comparison that cannot run must not take the fit down with it
        bracket_note = f"bracket comparison unavailable: {exc}"

    comparison = prior_density.holdout_comparison(
        d, cfg, model, grid, candidates)
    if bracket_note:
        # its own key: `note` may already carry why the window was unscorable,
        # and one message must not silently replace the other
        comparison["bracket_note"] = bracket_note

    return {
        "source": "profile_density",
        "method": "profile_density",
        "requested_source": pc["source"],
        "identifying_rows": pc.get("rows", "all_stocked_hours"),
        "search_bounds": list(pc["search_bounds"]),
        "grid_step": float(grid[1] - grid[0]),
        "uniform_limit": {
            "mean": round(float(np.mean(pc["search_bounds"])), 4),
            "std": round(float((pc["search_bounds"][1] - pc["search_bounds"][0])
                               / np.sqrt(12)), 4),
            "note": ("what a category with a FLAT likelihood gets, by "
                     "construction rather than by configuration. Any "
                     "per-category mean and std at these values means the data "
                     "said nothing and the support bounds are the whole "
                     "answer."),
        },
        "pooled": pooled,
        "no_fallback_note": (
            "There is no fallback_mean, fallback_std or std_floor in this "
            "method. A category with no information of its own takes the "
            "POOLED density, which is fitted across every category on this "
            "extract -- measured, not chosen. `own_information_weight` says "
            "how much of each category's prior is its own data."),
        "per_category": per_category,
        # `pricing.posterior.initialise` reads this to decide which categories
        # get their own cell, so it is required of any method, not a bracket
        # detail. Measured on the same train window the prior is fitted on.
        "episodes_per_week": _episodes_per_week(d, cfg),
        # SURFACED AT THE TOP. A category whose likelihood peaks at positive
        # elasticity is not weakly measured, it is measured backwards, and the
        # reader must not have to find that per category.
        "wrong_sign_categories": sorted(
            c for c, v in per_category.items() if v.get("wrong_sign")),
        "no_price_variation_categories": sorted(
            c for c, v in per_category.items() if "no_price_variation" in v),
        "holdout_comparison": comparison,
        "acceptance": {
            "passed": True,
            "failures": [],
            "note": ("this method has no reject path: a category that fails to "
                     "identify epsilon widens instead of being replaced, which "
                     "is what removes the constant. Read "
                     "`holdout_comparison` and `own_information_weight` "
                     "instead of an accept/reject flag."),
        },
    }


def estimate_prior(d, cfg, seed=0):
    """Dispatch on `posterior.prior.method`. `bracket` is kept so an older run
    reproduces exactly; `profile_density` is the default."""
    if cfg["posterior"]["prior"].get("method", "bracket") == "profile_density":
        return estimate_prior_density(d, cfg)
    return estimate_prior_bracket(d, cfg, seed=seed)


def estimate_prior_bracket(d, cfg, seed=0):
    pc = cfg["posterior"]["prior"]
    lo, hi = pc["search_bounds"]
    grid = np.linspace(lo, hi, pc["search_grid_size"])
    step = grid[1] - grid[0]
    rng = np.random.default_rng(seed)

    model = BaselineModel(cfg)
    # DISPERSION FOR THE BRACKET, and why it does not come from r_lookup.
    #
    # `fit_dispersion` forms its residuals at a working elasticity, and this
    # module produces that elasticity -- a genuine circular dependency. It used
    # to be broken by running dispersion FIRST at the fallback constant, which
    # was harmless only while the fallback WAS the prior. Once the bracket is
    # accepted per category (-1.6 to -2.3 measured), fitting r and rho at -1.0
    # measures residual correlation against a demand curve nothing uses -- and
    # that direction is strong: -1.0 -> -1.5 moved rho 0.3103 -> 0.4236 and
    # deff 3.347 -> 4.204, 26% of the learning rate.
    #
    # The dependency is broken on THIS side instead, because it is much weaker
    # here. With censored entry rows dropped above, `r` only reaches the
    # bracket through the weighting term `r/(r+mu)` -- and the reference is
    # fitted PER CATEGORY, so that term is not being fed a pooled average that
    # fits no category. `fit_dispersion` runs AFTER, on the real per-category
    # elasticities.
    #
    # the widest price spread in the extract is the below-cost hours, and
    # the bracket is the quantity most starved of variation -- so this is
    # the fit that gains most from train_population "integrity"
    train = population(split_frames(d, cfg)["train"], cfg).copy()
    train["censored"] = episodes.censored_hours(train)

    # THE REFERENCE IS MEASURED FROM THIS DATA, per category, not pinned to a
    # constant and not pooled.
    #
    # A pinned number was the first attempt and it was worse in every way that
    # matters: it goes stale against a retrain, it has to be re-pasted by hand,
    # and it says nothing about the extract in front of it. A pooled fit was the
    # second, and it averaged over a 5x-below to 6x-above spread -- see
    # `_reference_r` for the measurement. Both cost the same bounded scalar
    # minimisation as doing it properly.
    #
    # This does NOT reintroduce the cycle, because the reference is COMPUTED
    # here rather than read from `r_lookup.json`. It uses the fallback
    # elasticity, which is exactly the approximation the old ordering made --
    # except it now lands on the quantity that cares far less, and fitting
    # DISPERSION at the wrong elasticity cost 26% of the learning rate.
    #
    # Same estimator `fit_dispersion` uses -- imported, not reimplemented, so
    # the two cannot drift, and the same boundary clamp for the same reason.
    # `r` is assigned after `entry` is built -- the reference is fitted on
    # those rows, so it cannot be known yet.

    weeks = max(train.date.astype(str).map(
        lambda x: pd.Timestamp(x).to_period("W")).nunique(), 1)
    episodes_per_week = (train.groupby("category")["episode_id"].nunique() / weeks)

    # entry rows only -- same-hour cross-episode identifying variation (9.5);
    # zero-stock rows carry no demand information for the censored likelihood
    entry = (train.sort_values(["episode_id", "hour_of_day"])
             .groupby("episode_id").head(1))
    entry = entry[entry.starting_inventory >= 1]

    # A CENSORED ENTRY ROW IS A ONE-HOUR EPISODE, and it is dropped.
    #
    # Censoring is decided at an episode's LAST row, so an entry row can only
    # be censored when entry IS the last row -- the window sold out within its
    # opening hour. Fixture: 33 of 1,731 entry rows (1.91%).
    #
    # Dropping them is what lets the bracket run BEFORE `fit_dispersion`. The
    # censored branch of the likelihood is `nbinom.logsf(k-1, r, p)`, and how
    # much probability mass sits above k is entirely a dispersion question --
    # it is the channel through which `r` really bites. With no censored rows
    # left, `r` acts only through the weighting term `r/(r+mu)`, and a
    # per-category reference is then good enough: the circular dependency is
    # gone. Pooled was NOT good enough -- see `_reference_r`.
    #
    # THE COST IS A SELECTION BIAS, and it is small only because the count is:
    # these are the fastest-selling episodes, so removing them truncates the
    # top of the demand distribution and pulls |epsilon| TOWARD ZERO -- the
    # same direction the controlled arm is already biased. `entry_rows_censored`
    # is reported per category so the trade stays visible; if it climbs above a
    # couple of percent, this is no longer cheap and the ordering has to be
    # reconsidered rather than the number ignored.
    censored_entry = entry.censored.to_numpy()
    entry_censored_by_cat = (entry[censored_entry].groupby("category").size()
                             .to_dict())
    entry_total_by_cat = entry.groupby("category").size().to_dict()
    dropped_censored = int(censored_entry.sum())
    dropped_share = float(censored_entry.mean()) if len(censored_entry) else 0.0
    entry = entry[~censored_entry].copy()

    # NOW the reference can be fitted, on exactly these rows, per category --
    # dispersion is a property of the category, not of the extract, and the
    # spread across them (1.6 to the search bound on the fixture) dwarfs the
    # +-2x band this approximation was ever justified over. A category too thin
    # to fit its own takes the pooled value.
    r_by_cat, ref_r, r_basis = _reference_r(entry, cfg, model, pc)
    entry["r"] = (entry.category.astype(str).map(r_by_cat)
                  .fillna(float(ref_r)).astype(float))

    per_category, failures = {}, []
    for cat, g in entry.groupby("category"):
        if len(g) > pc["max_rows_per_category"]:
            keep = rng.choice(len(g), pc["max_rows_per_category"], replace=False)
            g = g.iloc[np.sort(keep)]
        mu_ref = model.predict_mu_ref(g)

        e_naive = _estimate(g, mu_ref, grid, controlled=False)
        e_ctrl = _estimate(g, mu_ref, grid, controlled=True)

        boundary = any(abs(e - b) <= step + 1e-12
                       for e in (e_naive, e_ctrl) for b in (lo, hi))
        # THE THREE CHECKS ARE SEPARATE, and only the first is unconditional.
        #
        #   wrong sign    a non-negative endpoint says demand rises with price.
        #                 Not a weak estimate -- a nonsensical one, and no
        #                 midpoint of it means anything. Always rejects.
        #   boundary      an endpoint pinned to a search bound. The estimate is
        #                 "at least this elastic", which is still information
        #                 about the SIGN and magnitude even though the point is
        #                 not identified.
        #   inverted      `naive > controlled`. Controlling for the hour
        #                 confound is expected to weaken the estimate; when it
        #                 strengthens it, our story about which estimator
        #                 bounds which is wrong -- but both endpoints are still
        #                 measured, and the interval between them is still the
        #                 interval the data supports.
        wrong_sign = not (e_naive < 0 and e_ctrl < 0)
        inverted = e_naive > e_ctrl

        # WHAT THE CONTROLLED ESTIMATE HAD TO WORK WITH. It profiles out hour
        # effects, so it is identified by WITHIN-HOUR price variation and
        # nothing else. Where a category has almost none, that estimate is
        # noise and can land anywhere -- including past the naive one, which
        # is what an inverted bracket usually is. Reported so the two cases
        # can be told apart: an unidentified control (fixable -- pool hours,
        # coarsen the control) versus a confound story that is genuinely wrong
        # for this category (a real finding).
        lr = np.log((1 - g.total_discount.to_numpy())
                    / (1 - g.d_ref.to_numpy()))
        v_tot = float(np.var(lr))
        within = lr - pd.Series(lr).groupby(
            g.hour_of_day.to_numpy()).transform("mean").to_numpy()
        identifying = float(np.var(within) / v_tot) if v_tot > 0 else 0.0

        mean = (e_naive + e_ctrl) / 2
        std = max(abs(e_naive - e_ctrl) / 2, pc["std_floor"])

        why = []
        if wrong_sign:
            why.append("wrong sign")
        if boundary and pc["reject_boundary_solutions"]:
            why.append("boundary solution")
        if inverted and pc["reject_orientation_violations"]:
            why.append("orientation violated")
        accepted = not why
        if not accepted:
            failures.append(f"{cat}: " + " + ".join(why))

        per_category[str(cat)] = {
            "epsilon_naive": e_naive, "epsilon_controlled": e_ctrl,
            "mean": round(mean, 4), "std": round(std, 4),
            # WHAT THE DATA SAID, kept even when the fallback overwrites
            # mean/std below. Without these the cost of a rejection is
            # invisible: the numbers the bracket produced are gone and the
            # owner cannot see what was given up without recomputing by hand.
            "bracket_mean": round(mean, 4), "bracket_std": round(std, 4),
            "boundary": boundary, "inverted": inverted,
            "wrong_sign": wrong_sign,
            # share of price variation surviving hour de-meaning: everything
            # the CONTROLLED estimate is identified by. Near zero means that
            # estimate is not identified, whatever number it returned.
            "identifying_variation_share": round(identifying, 4),
            # one-hour episodes, dropped so the bracket needs no fitted r.
            # Small here; if it climbs the selection bias stops being cheap.
            "entry_rows_censored_dropped": int(
                entry_censored_by_cat.get(cat, 0)),
            # THE DISPERSION THIS BRACKET WAS SCORED AT, and whether it is the
            # category's own or the pooled stand-in. Per row rather than once
            # at the top because that is the level it now varies at, and a
            # reader checking a suspicious bracket should not have to
            # cross-reference which r produced it.
            "reference_r": round(float(g.r.iloc[0]), 6),
            "reference_r_scope": ("category" if str(cat) in r_by_cat
                                  else "pooled"),
            "accepted": accepted, "rows": int(len(g)),
        }
        if not accepted:
            per_category[str(cat)]["rejected_for"] = why
        if inverted:
            per_category[str(cat)]["inversion_note"] = (
                "epsilon_naive > epsilon_controlled: the hour control "
                "STRENGTHENED the estimate when the bracket argument requires "
                "it to weaken it, so neither endpoint bounds the truth in the "
                "assumed direction. The midpoint is USED -- it is the best "
                "reading the data supports and the alternative is a constant "
                "-- but it is the centre of two measurements rather than a "
                "set-identified bracket, so treat it as weaker evidence than "
                "an upright category's. "
                f"Within-hour price variation is {identifying:.1%} of total -- "
                + ("below 10%, so the controlled fit is barely identified and "
                   "this is most likely noise rather than a finding: pool "
                   "hours or coarsen the control and re-run."
                   if identifying < 0.10 else
                   "ample, so the control IS identified and the inversion is "
                   "a real result about this category -- the evening-lift "
                   "confound story does not hold here. Worth investigating "
                   "before dismissing."))

    stds = {v["std"] for v in per_category.values() if v["accepted"]}
    constant_std = len(per_category) > 1 and len(stds) == 1 \
        and next(iter(stds)) != pc["std_floor"]

    per_cat_scope = pc["acceptance_scope"] == "per_category"
    if constant_std:
        # A std identical across every category asserts uniform confidence the
        # data does not support. Frame-wide by nature, so it fails the whole
        # prior under either scope.
        failures.append("prior_std constant across categories -- asserts "
                        "uniform confidence the data does not support")

    # WHICH CATEGORIES FALL BACK. Under `per_category` a category that passed
    # its own checks keeps its own bracket, because the checks are
    # pre-registered and applied per cell -- using the cells that pass is not
    # selecting on the outcome. Under `all_or_nothing` one failure replaces
    # every category, which is what PRD 9.5 originally specified.
    #
    # The all-or-nothing rule was costing real information. On the production
    # extract BAKERY & PASTRY passed every check on 21,484 rows and was still
    # overwritten with -1.0 +- 0.6 because two OTHER categories failed. Its own
    # bracket was -1.6125 +- 1.0875. Judged by the PRD's own criterion -- "not
    # confidently wrong" -- the fallback was the worse of the two: it puts the
    # measured deepening bar (median 2.429) 2.38 sigma away, against 0.75 sigma
    # under the bracket. 0.6 is a config constant; 1.0875 was measured.
    if constant_std or (failures and not per_cat_scope):
        fell_back = list(per_category)          # frame-wide failure
    else:
        fell_back = [c for c, v in per_category.items() if not v["accepted"]]
    for cat in fell_back:
        per_category[cat]["mean"] = pc["fallback_mean"]
        per_category[cat]["std"] = pc["fallback_std"]
        per_category[cat]["using"] = "fallback"
    for cat in per_category:
        per_category[cat].setdefault("using", "bracket")

    all_accepted = not failures
    n_bracket = sum(1 for v in per_category.values() if v["using"] == "bracket")
    source = ("bracket" if n_bracket == len(per_category) and per_category
              else "fallback" if n_bracket == 0 else "mixed")

    return {
        "source": source,
        "requested_source": pc["source"],
        "identifying_rows": ("entry-hour only (same-hour cross-episode, PRD "
                             "9.5), censored entry rows excluded"),
        # THE DISPERSION THE BRACKET USED, and where it came from. Recorded
        # because it is the assumption that lets this step run BEFORE
        # fit_dispersion -- the reader should be able to see it, not infer it.
        #
        # `reference_r` is the POOLED value, used only for categories too thin
        # to fit their own. What actually scored each bracket is
        # `reference_r_by_category` (and `per_category[c].reference_r`). The
        # two differ by a lot -- 1.6 to the search bound against pooled 8.04 on
        # the fixture -- so reading the pooled number as "the r the bracket
        # used" would be wrong for every category that fitted its own.
        "reference_r": round(float(ref_r), 6),
        "reference_r_by_category": {k: round(float(v), 6)
                                    for k, v in sorted(r_by_cat.items())},
        "reference_r_basis": r_basis,
        # THE COST OF THAT ORDERING. A censored entry row is a one-hour
        # episode -- the fastest sellers -- so dropping them pulls |epsilon|
        # toward zero. Cheap only while the count is small: above a couple of
        # percent the trade is no longer worth it and the ordering should be
        # revisited rather than this number ignored.
        "entry_rows_censored_dropped": dropped_censored,
        "entry_rows_censored_share": round(dropped_share, 4),
        "search_bounds": [lo, hi],
        "grid_step": float(step),
        "per_category": per_category,
        # SURFACED AT THE TOP, not left for someone to notice per category. An
        # inverted bracket still supplies its midpoint (owner, 2026-08-24), so
        # nothing downstream refuses it -- which is exactly why the count has
        # to be somewhere a reader cannot miss.
        "inverted_categories": sorted(c for c, v in per_category.items()
                                      if v["inverted"]),
        "inverted_note": (
            "These categories have epsilon_naive > epsilon_controlled, so the "
            "bracket's DIRECTIONAL argument -- naive biased too elastic, "
            "controlled biased toward zero, truth between them -- does not "
            "hold. Their midpoint is the best available reading of the data "
            "and is used, but it is not set-identified the way the others "
            "are. Read identifying_variation_share: near zero means the "
            "controlled fit was not identified and the inversion is noise "
            "(pool hours or coarsen the control); ample means the confound "
            "story genuinely fails for that category."),
        "episodes_per_week": {str(k): round(float(v), 1)
                              for k, v in episodes_per_week.items()},
        "acceptance": {"passed": all_accepted, "failures": failures,
                       "note": "rejection is an acceptable outcome; the loop "
                               "needs the prior not confidently wrong, not good"},
    }


def _print_density(prior):
    u = prior["uniform_limit"]
    print(f"prior method: profile_density   rows: {prior['identifying_rows']}")
    print(f"  flat-likelihood limit: {u['mean']:+.3f} +- {u['std']:.3f}  "
          f"(a category AT these values learned nothing)")
    print(f"  pooled across categories: {prior['pooled']['pooled_mean']:+.3f} "
          f"+- {prior['pooled']['pooled_std']:.3f}")
    print(f"  pooled basis: {prior['pooled'].get('pooled_basis', '')}\n")
    print(f"  {'category':12s} {'mean':>7s} {'std':>6s} {'std from':>9s} "
          f"{'peak':>7s} {'own':>5s} {'span':>9s} {'deff':>5s}  note")
    for cat, v in prior["per_category"].items():
        note = ("WRONG SIGN -- pooled" if v.get("wrong_sign")
                else "NO PRICE VARIATION -- pooled" if "no_price_variation" in v
                else "own data" if v["own_information_weight"] >= 0.999
                else f"{v['own_information_weight']:.0%} own, rest pooled")
        u = v.get("unconstrained_argmax") or {}
        peak = max(u.values()) if u else float("nan")
        print(f"  {cat:12s} {v['mean']:>+7.3f} {v['std']:>6.3f} "
              f"{v.get('std_basis', '?'):>9s} {peak:>+7.3f} "
              f"{v['own_information_weight']:>5.2f} {v['likelihood_span']:>9.1f} "
              f"{v['deff']:>5.2f}  {note}")
    if prior.get("wrong_sign_categories"):
        print(f"\n  !! {len(prior['wrong_sign_categories'])} category(s) with a "
              f"WRONG-SIGN likelihood -- its unconstrained peak is at or above "
              f"zero, i.e. demand rising with price: "
              + ", ".join(prior["wrong_sign_categories"])
              + ". Their own densities are discarded and they take the pooled "
                "one. `peak` above is where the likelihood really wanted to "
                "sit before search_bounds clipped it.")
    c = prior.get("holdout_comparison", {})
    if c.get("total_per_row"):
        print(f"\n  held out on '{c['window']}' ({c['rows_scored']:,} rows), "
              f"log marginal predictive per row -- higher is better:")
        for k, v in sorted(c["total_per_row"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:18s} {v:>10.6f}")
        print(f"  information available on this window (oracle - uniform): "
              f"{c.get('information_available_per_row')} nats/row")
        if c.get("worse_than_a_flat_prior"):
            print(f"  !! {c['worse_than_a_flat_prior_note']}")
        if c.get("verdict"):
            print(f"  -> {c['verdict']}")
        if c.get("warning"):
            print(f"  !! {c['warning']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)
    prior = estimate_prior(d, cfg, seed=args.seed)

    stamp(prior, cfg, BaselineModel(cfg).version, "bootstrap.estimate_prior")
    path = cfg["posterior"]["prior"]["path"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(prior, f, indent=2)

    if prior.get("method") == "profile_density":
        _print_density(prior)
        print(f"wrote {path}")
        return

    print(f"prior source: {prior['source']}"
          + ("" if prior["acceptance"]["passed"]
             else f"  (bracket rejected: {prior['acceptance']['failures']})"))
    for cat, v in prior["per_category"].items():
        print(f"  {cat:12s} naive {v['epsilon_naive']:+.3f}  "
              f"controlled {v['epsilon_controlled']:+.3f}  "
              f"-> mean {v['mean']:+.3f} std {v['std']:.3f}"
              + ("  [BOUNDARY]" if v["boundary"] else "")
              + (f"  [INVERTED ident={v['identifying_variation_share']:.2f}]"
                 if v["inverted"] else ""))
    if prior.get("inverted_categories"):
        print(f"  !! {len(prior['inverted_categories'])} inverted bracket(s): "
              + ", ".join(prior["inverted_categories"])
              + " -- midpoint used anyway (owner, 2026-08-24); the bracket's "
                "directional argument does NOT hold for these. Read "
                "identifying_variation_share before trusting them.")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
