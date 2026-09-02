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

from common.config import (intraclass_correlation,
                           load_config)
from common import episodes
from common.io import write_json
from common.provenance import stamp
from bootstrap.prepare_data import population, pre_launch, split_frames
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
                "fallback_order": ["subcategory", "category", "global"],
                # what the residuals were ACTUALLY formed at -- per category,
                # from the prior in force
                "working_elasticity": eps0,
                "working_elasticity_basis": (
                    "per-category prior means" if eps_by_cat
                    else "constant -1.0 (no prior artifact yet)"),
                "working_elasticity_by_category": {
                    k: round(v, 4) for k, v in sorted(eps_by_cat.items())}}

    # rho on the CALIB window, same as r: in-train rows understate it (the
    # model fits its own residuals), an understated rho understates deff, and
    # deff deflates every posterior update. `m` is NOT frozen alongside it:
    # production measures forced hours per episode per batch, because that
    # number moves with the exploration rate by construction.
    calib["resid"] = calib.units_sold - calib.mu_hat

    sizes = calib.groupby("episode_id")["resid"].size()
    min_hours = cfg["assurance"]["rho_min_hours_per_episode"]
    sub_d = calib[calib.episode_id.isin(sizes[sizes >= min_hours].index)]
    rho = intraclass_correlation(sub_d["resid"], sub_d["episode_id"],
                                 dc["rho_clip_max"])

    dates = pd.to_datetime(calib.date)
    rho_out = {"rho": round(rho, 4),
               "rho_method": "variance_decomposition_on_fitted_mu_residuals",
               "fit_window": "calib",
               "fit_window_dates": [str(dates.min().date()),
                                    str(dates.max().date())],
               "fit_rows": int(len(calib)),
               "fit_episodes": int(calib.episode_id.nunique())}
    return r_lookup, rho_out


def drift_by_window(d, cfg, freq="W"):
    """Do r and rho actually move, or are they stable enough to freeze?
    Refits both on rolling windows with the SAME estimators as the frozen
    fit. A MEASUREMENT, not a re-fit schedule: weekly re-fits would add noise
    to the learning rate's denominators and reintroduce the eps<->r cycle.
    Use: decide the retrain cadence; baseline the assurance alerts."""
    dc = cfg["dispersion"]
    model = BaselineModel(cfg)
    eps_by_cat, eps0 = _working_elasticity(cfg)
    bounds = dc["r_search_bounds"]

    # rule 16: drift_by_window sets the retrain cadence and baselines the
    # rho drift alert, both pre-launch readings -- the hold-out is shadow's
    full = population(pre_launch(d, cfg), cfg).copy()
    full = full[full.starting_inventory >= 1]
    if not len(full):
        return {"verdict": "NOT RUN -- no rows"}
    mu_ref = model.predict_mu_ref(full)
    ratio = (1 - full.total_discount.to_numpy()) / (1 - full.d_ref.to_numpy())
    eps_row = full.category.astype(str).map(eps_by_cat).fillna(eps0).to_numpy()
    full["mu_hat"] = np.clip(mu_ref * ratio ** eps_row,
                             cfg["pricing"]["demand_floor"], None)
    full["censored"] = episodes.censored_hours(full)
    full["resid"] = full.units_sold - full.mu_hat
    full["_win"] = pd.to_datetime(full.date).dt.to_period(freq)

    by_window, thin = {}, []
    for win, g in full.groupby("_win"):
        label = str(win.start_time.date())
        if len(g) < dc["min_rows_per_group"]:
            thin.append(label)
            continue
        r, ok = fit_r(g.units_sold.to_numpy(), g.mu_hat.to_numpy(),
                      g.censored.to_numpy(), bounds)
        pear = pearson_dispersion(g.units_sold.to_numpy(), g.mu_hat.to_numpy())
        # An r near the search bound is the estimator FAILING, not a large r:
        # under-dispersed data (Pearson < 1) cannot be expressed by any NB
        # (Var = mu + mu^2/r >= mu), so the MLE runs to the ceiling. Folding
        # those into a spread reads a failed fit as drift -- the exact
        # mistake this block exists to prevent someone making.
        at_bound = bool(ok and (r >= bounds[1] * 0.99 or r <= bounds[0] * 1.01))
        usable = bool(ok and pear >= 1.0 and not at_bound)
        # the ONE rho estimator (ANOVA ICC); this block still ran the biased
        # var(means)/var(all) after the frozen fit moved off it, so the drift
        # baseline and the frozen value disagreed by ~(1-rho)/m
        sizes = g.groupby("episode_id")["resid"].size()
        sub = g[g.episode_id.isin(
            sizes[sizes >= cfg["assurance"]["rho_min_hours_per_episode"]].index)]
        rho_w = (intraclass_correlation(sub["resid"], sub["episode_id"],
                                        dc["rho_clip_max"])
                 if sub.episode_id.nunique() > 1 else None)
        by_window[label] = {
            "rows": int(len(g)),
            "r": round(r, 4) if ok else None,
            "rho": round(rho_w, 4) if rho_w is not None else None,
            "pearson": round(pear, 3),
            "nb_expressible": bool(pear >= 1.0),
            "r_at_search_bound": at_bound,
            "r_usable": usable,
        }

    # r stats over the windows where an NB fit MEANS anything
    rs = [v["r"] for v in by_window.values() if v["r_usable"]]
    unusable = [w for w, v in by_window.items() if not v["r_usable"]]
    rhos = [v["rho"] for v in by_window.values() if v["rho"] is not None]
    pears = [v["pearson"] for v in by_window.values()]
    alert = cfg["assurance"].get("rho_drift_alert")

    def spread(xs):
        return (round(max(xs) - min(xs), 4), round(float(np.median(xs)), 4)) \
            if xs else (None, None)

    r_spread, r_med = spread(rs)
    rho_spread, rho_med = spread(rhos)
    unusable_share = len(unusable) / max(len(by_window), 1)
    return {
        "freq": freq,
        "windows_fitted": len(by_window),
        "windows_too_thin": thin,
        "by_window": by_window,
        "r_median": r_med, "r_spread": r_spread,
        "r_windows_usable": len(rs),
        "r_windows_unusable": unusable,
        "r_unusable_share": round(unusable_share, 3),
        "rho_median": rho_med, "rho_spread": rho_spread,
        "rho_drift_alert": alert,
        "rho_spread_vs_alert": round(rho_spread / alert, 2)
            if rho_spread and alert else None,
        "pearson_median": round(float(np.median(pears)), 3) if pears else None,
        "pearson_range": [round(min(pears), 3), round(max(pears), 3)]
            if pears else None,
        "verdict": (
            "NOT ENOUGH WINDOWS -- widen freq or the extract"
            if len(by_window) < 3 else
            f"r IS NOT FITTABLE AT THIS CADENCE -- {len(unusable)} of "
            f"{len(by_window)} windows are under-dispersed (Pearson < 1, which "
            "no NB expresses) or pinned at a search bound, so their r is a "
            "failed fit rather than a value. Re-fitting r this often would "
            "bank those as if they were measurements; read pearson_range "
            "before reading r at all"
            if unusable_share > 0.34 else
            f"rho varies by {rho_spread} across {freq} windows against a "
            f"{alert} live alert ({rho_spread / alert:.1f}x): the alert would "
            "fire on ORDINARY variation, so treat it as an alarm to retune "
            "rather than a drift detector"
            if rho_spread and alert and rho_spread > alert else
            f"rho varies by {rho_spread} across {freq} windows, inside the "
            f"{alert} live alert -- freezing it is defensible and the alert "
            "discriminates real drift"),
        "note": ("A MEASUREMENT, not a re-fit schedule: both are second "
                 "moments from weak signal, and re-fitting r during learning "
                 "reintroduces the eps <-> r cycle design 5.6 removed. Use it "
                 "to set the RETRAIN cadence and to sanity-check the live "
                 "assurance alerts. READ PEARSON FIRST: a window below 1.0 is "
                 "steadier than Poisson, so its r is the MLE running to a "
                 "bound rather than a dispersion estimate, and a SHIFT in "
                 "pearson across windows is a regime change -- which is a "
                 "retrain question, not a re-fit-more-often one."),
    }


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
    # is freezing these defensible on THIS extract? Measured, not assumed --
    # and it sets the retrain cadence rather than a weekly re-fit (see
    # drift_by_window on why weekly is the wrong answer)
    rho_out["drift_by_window"] = drift_by_window(d, cfg)
    bundle = BaselineModel(cfg).version
    stamp(r_lookup, cfg, bundle, "bootstrap.fit_dispersion")
    stamp(rho_out, cfg, bundle, "bootstrap.fit_dispersion")
    write_json(dc["r_lookup_path"], r_lookup)
    write_json(dc["rho_path"], rho_out)

    print(f"r by subcategory : {len(r_lookup['subcategory'])} groups, "
          f"global r = {r_lookup['global']:.3f}, clamp at {r_lookup['clamp_at']:.3f}")
    print(f"rho              : {rho_out['rho']}  (m is measured per batch "
          "in production -- common.config.deff_from_episodes)")
    dr = rho_out["drift_by_window"]
    if dr.get("windows_fitted"):
        print(f"drift ({dr['freq']})       : r {dr['r_median']} +-{dr['r_spread']} | "
              f"rho {dr['rho_median']} +-{dr['rho_spread']} over "
              f"{dr['windows_fitted']} windows")
        print(f"                   {dr['verdict']}")
    print("paste into config.yaml: dispersion.rho, "
          "-- m is measured per batch, never pasted")


if __name__ == "__main__":
    main()
