"""bootstrap.fit_dispersion -- fit and freeze r_lookup and rho (PRD section 9.4).

    D ~ NegBin(r, mu),  Var[D] = mu + mu^2 / r

r is fit by subcategory on calibration-period residuals via censored maximum
likelihood, with fallback per config (subcategory -> category -> global), low
values preserved, and high converged values clamped at clamp_percentile.

rho is a single global scalar, re-fit here against FITTED mu residuals -- this
value supersedes the phase-0 proxy (which used category-hour means). It
measures correlation structure, not price response, so the policy confound
does not affect it.

RUNS AFTER `bootstrap.estimate_prior`, and scales mu_ref to the actual price
using each category's OWN prior mean. Fitting at a constant working
elasticity measures correlation against a demand curve nothing uses, and the
direction is strong: moving the working elasticity -1.0 -> -1.5 moved rho
0.3103 -> 0.4236 and deff 3.347 -> 4.204, 26% of the learning rate. There is
no cycle in the other direction: the prior is a censored POISSON profile with
no dispersion parameter, so it needs nothing from this module.

Run standalone with no prior artifact, this falls back to the -1.0 constant
and says so in `working_elasticity_basis`. History of the orderings tried:
docs/learnings.md.

Usage:
    python3 -m bootstrap.fit_dispersion --input data/prepared.parquet
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

from common.config import load_config, design_effect
from common import episodes
from common.provenance import stamp
from bootstrap.prepare_data import population, split_frames
from bootstrap.train_baseline import BaselineModel


def _censored_nll(r, k, mu, censored):
    p = r / (r + mu)
    ll = np.where(censored,
                  nbinom.logsf(np.maximum(k, 1) - 1, r, p),
                  nbinom.logpmf(k, r, p))
    return -float(np.sum(ll))


def pearson_dispersion(k, mu, floor=1e-9):
    """mean((k - mu)^2 / mu). One under Poisson, above one when overdispersed,
    BELOW ONE when the data is steadier than Poisson -- which no negative
    binomial can represent, since Var = mu + mu^2/r >= mu for every finite r.

    Used to tell a genuinely tight group apart from a thin one whose MLE
    wandered to the search ceiling. The two produce the same r and want
    opposite treatment from the clamp.
    """
    mu = np.maximum(np.asarray(mu, dtype=float), floor)
    return float(np.mean((np.asarray(k, dtype=float) - mu) ** 2 / mu))


def fit_r(k, mu, censored, bounds):
    res = minimize_scalar(
        lambda lr: _censored_nll(np.exp(lr), k, mu, censored),
        bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x)), bool(res.success)


def _working_elasticity(cfg):
    """(per-category means, fallback) from the prior in force.

    Returns the fallback alone when no prior artifact exists yet -- a first
    bootstrap, or the old ordering -- so this module still runs standalone.
    Both are reported in `r_lookup.json`, because `working_elasticity` is what
    someone reads to check that r and rho were fitted against the demand curve
    the system actually uses.
    """
    pc = cfg["posterior"]["prior"]
    # the standalone fallback: a working constant for a first run with no
    # prior artifact on disk. NOT a prior -- the prior has no constant in it.
    fallback = -1.0
    path = pc["path"]
    if not os.path.exists(path):
        return {}, fallback
    with open(path) as f:
        prior = json.load(f)
    return ({str(c): float(v["mean"])
             for c, v in prior.get("per_category", {}).items()}, fallback)


def fit_dispersion(d, cfg):
    dc = cfg["dispersion"]
    calib = population(split_frames(d, cfg)["calib"], cfg).copy()
    # rows with no stock carry no demand information and would be mis-scored
    # as "demand >= 1" by the censored likelihood
    calib = calib[calib.starting_inventory >= 1]
    if not len(calib):
        raise RuntimeError("calibration window contains no rows")

    model = BaselineModel(cfg)
    # THE WORKING ELASTICITY, per category, from the prior actually in force
    # (see module docstring: residuals formed at a constant measure the wrong
    # curve, and the cost is ~26% of the learning rate)
    eps_by_cat, eps0 = _working_elasticity(cfg)
    mu_ref = model.predict_mu_ref(calib)
    ratio = (1 - calib.total_discount.to_numpy()) / (1 - calib.d_ref.to_numpy())
    # per row, from its own category's prior; `eps0` only where none exists
    eps_calib = calib.category.astype(str).map(eps_by_cat).fillna(eps0).to_numpy()
    calib["mu_hat"] = np.clip(mu_ref * ratio ** eps_calib,
                              cfg["pricing"]["demand_floor"], None)
    calib["censored"] = episodes.censored_hours(calib)

    bounds = dc["r_search_bounds"]
    min_rows = dc["min_rows_per_group"]

    def fit_group(g):
        return fit_r(g.units_sold.to_numpy(), g.mu_hat.to_numpy(),
                     g.censored.to_numpy(), bounds)

    by_sub, by_cat, under = {}, {}, {}
    for level, store in (("subcategory", by_sub), ("category", by_cat)):
        for key, g in calib.groupby(level):
            if len(g) < min_rows:
                continue
            r, ok = fit_group(g)
            if not ok:
                continue
            store[str(key)] = r
            p = pearson_dispersion(g.units_sold.to_numpy(), g.mu_hat.to_numpy())
            if p < 1.0:
                under[f"{level}:{key}"] = round(p, 4)
    r_global, _ = fit_group(calib)
    pearson_global = pearson_dispersion(calib.units_sold.to_numpy(),
                                        calib.mu_hat.to_numpy())

    # CLAMP HIGH CONVERGED VALUES, PRESERVE LOW ONES -- AND EXEMPT THE GROUPS
    # THAT ARE GENUINELY UNDER-DISPERSED (owner, 2026-08-24).
    #
    # The clamp exists to hold down a SPURIOUSLY high r: a thin group whose MLE
    # wanders to the search ceiling because it has too few rows to pin down.
    # But an r at the ceiling has a second, entirely different cause -- the data
    # really is Poisson or tighter, and an NB cannot represent that at all,
    # since Var = mu + mu^2/r >= mu for every finite r. The MLE runs to the
    # ceiling because the ceiling is the closest the family can get.
    #
    # Clamping THOSE groups is wrong in a specific and harmful direction: it
    # makes the model claim MORE variance than the data has, for the groups
    # that have least. Everything reading `r` inherits it -- the censored
    # demand expectation the DP maximises, the posterior's likelihood, the
    # exploration cost of a tier -- all of them then over-weight tail outcomes
    # for the steadiest sellers in the extract. Measured on the fixture's entry
    # rows, SIDE DISH came back at Pearson 0.597 with its r at the 50.0 bound
    # and was being clamped to the 90th percentile of OTHER groups' values.
    #
    # Pearson dispersion is what tells the two cases apart, and it is cheap:
    # mean((k - mu)^2 / mu), which is 1 under Poisson. Below 1 the group is
    # under-dispersed and its high r is a fact rather than an artifact, so it
    # is exempt and REPORTED, because "this category is steadier than Poisson"
    # is a finding about the business worth someone's attention.
    converged = list(by_sub.values()) + list(by_cat.values()) + [r_global]
    cap = float(np.percentile(converged, dc["clamp_percentile"] * 100))

    def clamped(level, store):
        return {k: (v if f"{level}:{k}" in under else min(v, cap))
                for k, v in store.items()}

    by_sub = clamped("subcategory", by_sub)
    by_cat = clamped("category", by_cat)
    if pearson_global >= 1.0:
        r_global = min(r_global, cap)

    r_lookup = {"subcategory": by_sub, "category": by_cat,
                "global": r_global, "clamp_at": cap,
                # WHERE THE NB FAMILY DOES NOT FIT. Var = mu + mu^2/r is never
                # below mu, so a group under Pearson 1.0 cannot be represented
                # at any r and its fit sits at the search ceiling by necessity.
                # Exempt from the clamp for that reason, and listed so the
                # misfit is visible rather than absorbed into a percentile.
                "under_dispersed_groups": dict(sorted(under.items())),
                "pearson_global": round(float(pearson_global), 4),
                "under_dispersion_note": (
                    "Pearson dispersion mean((k-mu)^2/mu) below 1.0. These "
                    "groups are steadier than Poisson, which the negative "
                    "binomial cannot express at any r, so their fit sits at "
                    "the search ceiling and is EXEMPT from clamp_percentile -- "
                    "clamping it down would make the model claim more variance "
                    "than the data has, for the steadiest cells in the "
                    "extract. If this list is long, the NB is the wrong family "
                    "for this extract and not just for these cells."),
                "fallback_order": dc["fallback_order"],
                # what the residuals were ACTUALLY formed at. Per category
                # now, from the prior in force -- `working_elasticity` alone
                # was the constant -1.0 and said nothing about whether r and
                # rho matched the demand curve in use.
                "working_elasticity": eps0,
                "working_elasticity_basis": (
                    "per-category prior means" if eps_by_cat
                    else "constant -1.0 (no prior artifact yet)"),
                "working_elasticity_by_category": {
                    k: round(v, 4) for k, v in sorted(eps_by_cat.items())}}

    # rho against fitted residuals -- the authoritative value (PRD section 8 m3)
    full = d.copy()
    mu_ref_full = model.predict_mu_ref(full)
    ratio_full = (1 - full.total_discount.to_numpy()) / (1 - full.d_ref.to_numpy())
    full["resid"] = full.units_sold - np.clip(
        mu_ref_full * ratio_full ** full.category.astype(str).map(
            eps_by_cat).fillna(eps0).to_numpy(),
        cfg["pricing"]["demand_floor"], None)

    sizes = full.groupby("episode_id")["resid"].size()
    sub_d = full[full.episode_id.isin(sizes[sizes >= 3].index)]
    between = sub_d.groupby("episode_id")["resid"].mean().var(ddof=1)
    total = sub_d["resid"].var(ddof=1)
    rho = float(np.clip(between / total, 0.0, 0.95)) if total > 0 else 0.0

    hours = full.groupby("episode_id").size()
    changed = full.groupby("episode_id")["total_discount"].nunique() > 1
    hours_forced = hours[changed[changed].index]
    h_forced = float(hours_forced.mean()) if len(hours_forced) else float(hours.mean())

    rho_out = {"rho": round(rho, 4),
               "rho_method": "variance_decomposition_on_fitted_mu_residuals",
               "mean_forced_hours_per_episode": round(h_forced, 3),
               "implied_deff": round(design_effect(rho, h_forced), 3)}
    return r_lookup, rho_out


def lookup_r(r_lookup, subcategory, category):
    for level in r_lookup["fallback_order"]:
        if level == "subcategory" and str(subcategory) in r_lookup["subcategory"]:
            return r_lookup["subcategory"][str(subcategory)]
        if level == "category" and str(category) in r_lookup["category"]:
            return r_lookup["category"][str(category)]
        if level == "global":
            return r_lookup["global"]
    return r_lookup["global"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)
    r_lookup, rho_out = fit_dispersion(d, cfg)

    dc = cfg["dispersion"]
    os.makedirs(os.path.dirname(dc["r_lookup_path"]) or ".", exist_ok=True)
    bundle = BaselineModel(cfg).version
    stamp(r_lookup, cfg, bundle, "bootstrap.fit_dispersion")
    stamp(rho_out, cfg, bundle, "bootstrap.fit_dispersion")
    with open(dc["r_lookup_path"], "w") as f:
        json.dump(r_lookup, f, indent=2)
    with open(dc["rho_path"], "w") as f:
        json.dump(rho_out, f, indent=2)

    print(f"r by subcategory : {len(r_lookup['subcategory'])} groups, "
          f"global r = {r_lookup['global']:.3f}, clamp at {r_lookup['clamp_at']:.3f}")
    print(f"rho              : {rho_out['rho']}  "
          f"(forced hours {rho_out['mean_forced_hours_per_episode']}, "
          f"deff {rho_out['implied_deff']})")
    print("paste into config.yaml: dispersion.rho, "
          "dispersion.mean_forced_hours_per_episode")


if __name__ == "__main__":
    main()
