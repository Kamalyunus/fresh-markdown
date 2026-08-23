"""bootstrap.fit_dispersion -- fit and freeze r_lookup and rho (PRD section 9.4).

    D ~ NegBin(r, mu),  Var[D] = mu + mu^2 / r

r is fit by subcategory on calibration-period residuals via censored maximum
likelihood, with fallback per config (subcategory -> category -> global), low
values preserved, and high converged values clamped at clamp_percentile.

rho is a single global scalar, re-fit here against FITTED mu residuals -- this
value supersedes the phase-0 proxy (which used category-hour means). It
measures correlation structure, not price response, so the policy confound
does not affect it.

The working mean for the censored likelihood uses the fallback prior
elasticity to scale mu_ref to the actual price; r is insensitive to modest
error in that scaling, and the elasticity prior proper does not exist until
bootstrap.estimate_prior runs (which needs r).

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


def fit_r(k, mu, censored, bounds):
    res = minimize_scalar(
        lambda lr: _censored_nll(np.exp(lr), k, mu, censored),
        bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x)), bool(res.success)


def fit_dispersion(d, cfg):
    dc = cfg["dispersion"]
    calib = population(split_frames(d, cfg)["calib"], cfg).copy()
    # rows with no stock carry no demand information and would be mis-scored
    # as "demand >= 1" by the censored likelihood
    calib = calib[calib.starting_inventory >= 1]
    if not len(calib):
        raise RuntimeError("calibration window contains no rows")

    model = BaselineModel(cfg)
    eps0 = cfg["posterior"]["prior"]["fallback_mean"]
    mu_ref = model.predict_mu_ref(calib)
    ratio = (1 - calib.total_discount.to_numpy()) / (1 - calib.d_ref.to_numpy())
    calib["mu_hat"] = np.clip(mu_ref * ratio ** eps0,
                              cfg["pricing"]["demand_floor"], None)
    calib["censored"] = episodes.censored_hours(
        calib.starting_inventory, calib.units_sold, calib.ending_inventory)

    bounds = dc["r_search_bounds"]
    min_rows = dc["min_rows_per_group"]

    def fit_group(g):
        return fit_r(g.units_sold.to_numpy(), g.mu_hat.to_numpy(),
                     g.censored.to_numpy(), bounds)

    by_sub, by_cat = {}, {}
    for sub, g in calib.groupby("subcategory"):
        if len(g) >= min_rows:
            r, ok = fit_group(g)
            if ok:
                by_sub[str(sub)] = r
    for cat, g in calib.groupby("category"):
        if len(g) >= min_rows:
            r, ok = fit_group(g)
            if ok:
                by_cat[str(cat)] = r
    r_global, _ = fit_group(calib)

    # clamp high converged values; preserve low ones
    converged = list(by_sub.values()) + list(by_cat.values()) + [r_global]
    cap = float(np.percentile(converged, dc["clamp_percentile"] * 100))
    by_sub = {k: min(v, cap) for k, v in by_sub.items()}
    by_cat = {k: min(v, cap) for k, v in by_cat.items()}
    r_global = min(r_global, cap)

    r_lookup = {"subcategory": by_sub, "category": by_cat,
                "global": r_global, "clamp_at": cap,
                "fallback_order": dc["fallback_order"],
                "working_elasticity": eps0}

    # rho against fitted residuals -- the authoritative value (PRD section 8 m3)
    full = d.copy()
    mu_ref_full = model.predict_mu_ref(full)
    ratio_full = (1 - full.total_discount.to_numpy()) / (1 - full.d_ref.to_numpy())
    full["resid"] = full.units_sold - np.clip(
        mu_ref_full * ratio_full ** eps0, cfg["pricing"]["demand_floor"], None)

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
