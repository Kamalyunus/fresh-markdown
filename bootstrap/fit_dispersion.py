"""bootstrap.fit_dispersion -- fit and freeze r_lookup and rho (design 5.5).

r per subcategory via censored MLE on calibration residuals (config fallback
chain; low values preserved, high converged values clamped). rho is one global
scalar on FITTED mu residuals -- supersedes the phase-0 proxy. RUNS AFTER
estimate_prior: residuals use each category's own prior-mean elasticity
(standalone falls back to -1.0 and says so). History: docs/learnings.md.
Run: python3 -m bootstrap.fit_dispersion --input data/prepared.parquet
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
    """mean((k - mu)^2 / mu): 1 under Poisson, BELOW 1 when steadier than
    Poisson -- which no NB can represent (Var = mu + mu^2/r >= mu). Tells a
    genuinely tight group from a thin one whose MLE hit the search ceiling;
    the two produce the same r and want opposite treatment from the clamp."""
    mu = np.maximum(np.asarray(mu, dtype=float), floor)
    return float(np.mean((np.asarray(k, dtype=float) - mu) ** 2 / mu))


def fit_r(k, mu, censored, bounds):
    res = minimize_scalar(
        lambda lr: _censored_nll(np.exp(lr), k, mu, censored),
        bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x)), bool(res.success)


def _working_elasticity(cfg):
    """(per-category means, fallback) from the prior in force. Returns the
    fallback alone when no prior artifact exists yet, so the module still
    runs standalone; both are reported in r_lookup.json."""
    pc = cfg["posterior"]["prior"]
    # standalone working constant -- NOT a prior; the prior has no constant
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
    # working elasticity per category from the prior in force -- residuals
    # formed at a constant measure the wrong demand curve
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

    # Clamp high CONVERGED r (a thin group's MLE at the ceiling), preserve low
    # -- but EXEMPT groups with Pearson < 1: genuinely under-dispersed data no
    # NB can represent, whose high r is a fact, not an artifact (owner,
    # 2026-08-24; docs/learnings.md).
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
                # where the NB family does not fit (Pearson < 1): exempt from
                # the clamp, and listed so the misfit stays visible
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
                # what the residuals were ACTUALLY formed at -- per category,
                # from the prior in force
                "working_elasticity": eps0,
                "working_elasticity_basis": (
                    "per-category prior means" if eps_by_cat
                    else "constant -1.0 (no prior artifact yet)"),
                "working_elasticity_by_category": {
                    k: round(v, 4) for k, v in sorted(eps_by_cat.items())}}

    # rho against fitted residuals -- the authoritative value (phase 0's m3 is a proxy)
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
