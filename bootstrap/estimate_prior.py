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
from bootstrap.fit_dispersion import lookup_r


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


def estimate_prior(d, cfg, seed=0):
    pc = cfg["posterior"]["prior"]
    lo, hi = pc["search_bounds"]
    grid = np.linspace(lo, hi, pc["search_grid_size"])
    step = grid[1] - grid[0]
    rng = np.random.default_rng(seed)

    model = BaselineModel(cfg)
    with open(cfg["dispersion"]["r_lookup_path"]) as f:
        r_lookup = json.load(f)

    # the widest price spread in the extract is the below-cost hours, and
    # the bracket is the quantity most starved of variation -- so this is
    # the fit that gains most from train_population "integrity"
    train = population(split_frames(d, cfg)["train"], cfg).copy()
    train["censored"] = episodes.censored_hours(train)
    train["r"] = [lookup_r(r_lookup, s, c)
                  for s, c in zip(train.subcategory, train.category)]

    weeks = max(train.date.astype(str).map(
        lambda x: pd.Timestamp(x).to_period("W")).nunique(), 1)
    episodes_per_week = (train.groupby("category")["episode_id"].nunique() / weeks)

    # entry rows only -- same-hour cross-episode identifying variation (9.5);
    # zero-stock rows carry no demand information for the censored likelihood
    entry = (train.sort_values(["episode_id", "hour_of_day"])
             .groupby("episode_id").head(1))
    entry = entry[entry.starting_inventory >= 1]

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
        "identifying_rows": "entry-hour only (same-hour cross-episode, PRD 9.5)",
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
