"""bootstrap.estimate_prior -- the elasticity prior (design 5.6).

The prior IS the profile likelihood read as a density (estimator:
bootstrap.prior_density); this runner builds the artifact, attaches per-
category volume, scores a held-out window, writes artifacts/prior.json.
No fallback constant, no reject path: an unidentified or wrong-signed
category takes the pooled density. Superseded designs: docs/learnings.md.
Run: python3 -m bootstrap.estimate_prior --input data/prepared.parquet
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config
from common.provenance import stamp
from bootstrap.prepare_data import population, split_frames
from bootstrap.train_baseline import BaselineModel
from bootstrap import prior_density


def _episodes_per_week(d, cfg):
    """Volume per category on the train window, which decides cell structure in
    `pricing.posterior.initialise`. A property of the population, not of how
    epsilon was estimated."""
    train = population(split_frames(d, cfg)["train"], cfg)
    weeks = max(train.date.astype(str).map(
        lambda x: pd.Timestamp(x).to_period("W")).nunique(), 1)
    per = train.groupby("category")["episode_id"].nunique() / weeks
    return {str(k): round(float(v), 1) for k, v in per.items()}


def estimate_prior(d, cfg, seed=0, fast=False):
    """The prior as a density per category, with its own evidence attached."""
    pc = cfg["posterior"]["prior"]
    model = BaselineModel(cfg)
    grid, per_category, densities, pooled = prior_density.estimate(
        d, cfg, model, fast=fast)

    comparison = prior_density.holdout_comparison(
        d, cfg, model, grid, {"profile_density": densities})

    return {
        "source": "profile_density",
        "method": "profile_density",
        "identifying_rows": pc.get("rows", "entry"),
        "hour_control": pc.get("hour_control", "date_hour"),
        "search_bounds": list(pc["search_bounds"]),
        "grid_step": float(grid[1] - grid[0]),
        "uniform_limit": {
            "mean": round(float(np.mean(pc["search_bounds"])), 4),
            "std": round(float((pc["search_bounds"][1] - pc["search_bounds"][0])
                               / np.sqrt(12)), 4),
            "note": ("what a category with a FLAT likelihood gets, by "
                     "construction rather than by configuration. A "
                     "per-category mean and std at these values means the "
                     "data said nothing and the support bounds are the whole "
                     "answer."),
        },
        "pooled": pooled,
        "no_fallback_note": (
            "There is no fallback_mean, fallback_std or std_floor in this "
            "method. A category with no information of its own takes the "
            "POOLED density, which is fitted across the right-signed "
            "categories of this extract -- measured, not chosen. "
            "`own_information_weight` says how much of each category's prior "
            "is its own data."),
        "per_category": per_category,
        "episodes_per_week": _episodes_per_week(d, cfg),
        # surfaced at the top: a positive-elasticity peak is measured
        # backwards, not weakly, and must not hide per category
        "wrong_sign_categories": sorted(
            c for c, v in per_category.items() if v.get("wrong_sign")),
        "no_price_variation_categories": sorted(
            c for c, v in per_category.items() if "no_price_variation" in v),
        "holdout_comparison": comparison,
        "acceptance": {
            "passed": True,
            "failures": [],
            "note": ("this method has no reject path: a category that fails "
                     "to identify epsilon widens instead of being replaced, "
                     "which is what removes the constant. Read "
                     "`holdout_comparison`, `wrong_sign_categories` and "
                     "`own_information_weight` instead of an accept/reject "
                     "flag."),
        },
    }


def _print(prior):
    u = prior["uniform_limit"]
    print(f"prior method: profile_density   rows: {prior['identifying_rows']}"
          f"   hour control: {prior['hour_control']}")
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
        u2 = v.get("unconstrained_argmax") or {}
        peak = max(u2.values()) if u2 else float("nan")
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
    rc = (prior.get("pooled") or {}).get("design_comparison") or {}
    if isinstance(rc, str):                # skipped under --fast
        print(f"\n  design: {rc}")
        rc = {}
    if rc:
        print(f"\n  design (in use: {rc.get('in_use')}) -- fewer wrong-signed "
              f"first, then span:")
        print(f"    {'rows + control':34s} {'wrong':>7s} {'span':>8s} "
              f"{'ident':>7s} {'rows/cell':>10s}")
        for key, v in rc.items():
            if not isinstance(v, dict):
                continue
            if "error" in v:
                print(f"    {key:34s} unavailable: {v['error']}")
                continue
            mark = "  <- in use" if key == rc.get("in_use") else ""
            print(f"    {key:34s} {v['wrong_signed_count']:>3}/"
                  f"{v['categories']:<3} {v['median_span']:>8.2f} "
                  f"{v['median_identifying_variation_share']:>7.3f} "
                  f"{v['median_rows_per_time_cell']:>10.1f}{mark}")
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
    ap.add_argument("--fast", action="store_true",
                    help="skip the diagnostics that cannot move the "
                         "calibration<->dispersion fixed point (~70% of the "
                         "cost). For LOOP TURNS only -- the final run must be "
                         "full, since init_posterior reads the std this "
                         "widens.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)
    prior = estimate_prior(d, cfg, seed=args.seed, fast=args.fast)

    stamp(prior, cfg, BaselineModel(cfg).version, "bootstrap.estimate_prior")
    path = cfg["posterior"]["prior"]["path"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(prior, f, indent=2)

    _print(prior)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
