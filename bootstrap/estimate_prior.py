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
categories at a -1.5 bound). Boundary solutions are rejected outright.

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
from common.provenance import stamp
from bootstrap.prepare_data import split_frames
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

    train = split_frames(d, cfg)["train"].copy()
    train["censored"] = train.units_sold >= train.starting_inventory
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
        oriented = e_naive <= e_ctrl < 0

        mean = (e_naive + e_ctrl) / 2
        std = max(abs(e_naive - e_ctrl) / 2, pc["std_floor"])
        accepted = oriented and not (boundary and pc["reject_boundary_solutions"])
        if not accepted:
            failures.append(f"{cat}: " + ("boundary solution" if boundary
                                          else "orientation violated"))
        per_category[str(cat)] = {
            "epsilon_naive": e_naive, "epsilon_controlled": e_ctrl,
            "mean": round(mean, 4), "std": round(std, 4),
            "boundary": boundary, "accepted": accepted, "rows": int(len(g)),
        }

    stds = {v["std"] for v in per_category.values() if v["accepted"]}
    constant_std = len(per_category) > 1 and len(stds) == 1 \
        and next(iter(stds)) != pc["std_floor"]
    if constant_std:
        failures.append("prior_std constant across categories -- asserts "
                        "uniform confidence the data does not support")

    all_accepted = not failures
    source = "bracket" if all_accepted else "fallback"
    if not all_accepted:
        for cat in per_category:
            per_category[cat]["mean"] = pc["fallback_mean"]
            per_category[cat]["std"] = pc["fallback_std"]

    return {
        "source": source,
        "requested_source": pc["source"],
        "identifying_rows": "entry-hour only (same-hour cross-episode, PRD 9.5)",
        "search_bounds": [lo, hi],
        "grid_step": float(step),
        "per_category": per_category,
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
              + ("  [BOUNDARY]" if v["boundary"] else ""))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
