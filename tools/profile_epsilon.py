"""tools.profile_epsilon -- how much does the data actually say about epsilon?

THIS IS A DIAGNOSTIC, NOT A FIT. It writes no artifact anything reads, and it
changes no number in the pipeline. It answers one question that every argument
about the elasticity prior has so far assumed rather than checked:

    over the search grid, how much does the log-likelihood MOVE?

`bootstrap.estimate_prior` reports the argmax of exactly this curve and nothing
about its shape, so a sharp peak and a dead-flat line come back looking
identical -- both as a number with four decimals. If the curve is flat, epsilon
is not identified, and no estimator, ordering or dispersion treatment will
rescue it; the honest response is a wide prior and exogenous price variation,
not a better algorithm. If it is sharp, the bracket's width is overstating the
uncertainty and the fallback is costing real information.

WHAT IS PLOTTED. Per category, the same censored NB log-likelihood
`estimate_prior` maximises, over the same grid, for both arms -- naive (no hour
control) and controlled (hour effects profiled out at each epsilon by moment
matching). Curves are shown as `ll(eps) - max ll`, so every panel is on one
scale and the only thing to read is the shape.

HOW TO READ IT.

  span            max(ll) - min(ll) across the whole grid, in log-likelihood
                  units. This is the total discrimination available: how much
                  better the best epsilon in [-4, -0.05] explains the data than
                  the worst. Under ~2 there is essentially nothing to choose
                  between the ends of the support.
  support_95      {eps : ll(eps) >= max - 1.92}, the profile-likelihood
                  interval. Width is the honest uncertainty in epsilon FROM
                  THIS LIKELIHOOD -- compare it against the bracket's own std,
                  which is derived from the gap between two point estimates and
                  knows nothing about curvature.
  at_bound        the argmax sits on a search bound, so the curve is monotone
                  and the point estimate is "at least this elastic", not a
                  maximum.
  open_interval   the 1.92 interval runs off the end of the grid: epsilon is
                  bounded on one side only, or not at all.
  se_curvature    1/sqrt(-d2ll/deps2) at the peak, by finite difference. A
                  quadratic approximation, so it disagrees with support_95
                  exactly when the curve is asymmetric -- which is itself worth
                  seeing.

THE INTERVALS ARE OPTIMISTIC, and by a knowable factor. They assume rows are
independent. Entry rows are one per episode, so within-episode correlation is
not the issue, but sku and day clustering is unmodelled -- the same structure
`rho` and `deff` exist to handle elsewhere. Multiply the width by roughly
sqrt(deff) for a defensible figure. If the interval is already wide, that
correction only makes the conclusion stronger.

Usage:
    python3 -m tools.profile_epsilon --input data/prepared.parquet
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.special import gammaln

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

from common.config import load_config
from common import episodes
from bootstrap.prepare_data import population, split_frames
from bootstrap.train_baseline import BaselineModel
from bootstrap import estimate_prior as ep

INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#c1440e"
GOOD = "#2f6b4f"

# 0.5 * chi2(1, 0.95). The standard profile-likelihood cutoff.
CUTOFF = 1.920729


def entry_rows(d, cfg):
    """The rows the bracket scores, rebuilt exactly as `estimate_prior` does.

    Duplicated deliberately rather than refactored out of `estimate_prior`: a
    diagnostic that shares mutable setup with the thing it audits can be made
    to agree with it by changing the shared part. This function is allowed to
    drift only if the fit does, and the test pins them together.
    """
    train = population(split_frames(d, cfg)["train"], cfg).copy()
    train["censored"] = episodes.censored_hours(train)
    entry = (train.sort_values(["episode_id", "hour_of_day"])
             .groupby("episode_id").head(1))
    entry = entry[entry.starting_inventory >= 1]
    return entry[~entry.censored.to_numpy()].copy()


def curve(g, mu_ref, grid, controlled):
    """log-likelihood at every epsilon on the grid. The full curve, where
    `estimate_prior._estimate` keeps only the argmax."""
    k = g.units_sold.to_numpy()
    r = g.r.to_numpy()
    censored = g.censored.to_numpy()
    log_ratio = np.log((1 - g.total_discount.to_numpy())
                       / (1 - g.d_ref.to_numpy()))
    lgamma_const = gammaln(k + r) - gammaln(r) - gammaln(k + 1)
    out = []
    for eps in grid:
        hm = (ep._profile_hour_multipliers(eps, g, mu_ref, log_ratio)
              if controlled else None)
        out.append(ep._censored_loglik(eps, mu_ref, log_ratio, k, r, censored,
                                       lgamma_const, hm))
    return np.asarray(out, dtype=float)


def read_curve(grid, ll):
    """Turn one curve into the numbers that decide whether epsilon is
    identified. Shape, not location -- the argmax is already reported by the
    fit and is the least informative thing here."""
    step = float(grid[1] - grid[0])
    i = int(np.argmax(ll))
    peak = float(ll[i])
    inside = ll >= peak - CUTOFF
    lo_i, hi_i = int(np.argmax(inside)), int(len(inside) - 1 - np.argmax(inside[::-1]))

    # curvature by central difference at the peak; undefined on a boundary
    se = None
    if 0 < i < len(ll) - 1:
        d2 = (ll[i + 1] - 2 * ll[i] + ll[i - 1]) / step ** 2
        if d2 < 0:
            se = float(1.0 / np.sqrt(-d2))

    return {
        "epsilon_hat": round(float(grid[i]), 4),
        "span": round(peak - float(np.min(ll)), 3),
        "support_95": [round(float(grid[lo_i]), 4), round(float(grid[hi_i]), 4)],
        "support_95_width": round(float(grid[hi_i] - grid[lo_i]), 4),
        "share_of_grid_within_cutoff": round(float(inside.mean()), 4),
        "at_bound": bool(i == 0 or i == len(ll) - 1),
        "open_interval": bool(inside[0] or inside[-1]),
        "se_curvature": None if se is None else round(se, 4),
        "ll_at_peak": round(peak, 2),
    }


def price_variation(g):
    """What epsilon has to work with, BEFORE any likelihood is evaluated.

    Epsilon enters only as `exp(eps * log_ratio)`, so if `log_ratio` is
    constant across the rows, epsilon is not weakly identified -- it is absent
    from the likelihood, and every value on the grid gives the identical
    number. The fit still reports an argmax, because argmax of a constant array
    returns the first index, which is the lower search bound. That is how a
    category with no price variation comes back as a confident -4.0.

    Reported here so the CAUSE sits beside the symptom: a flat profile is a
    fact about the curve, and this is the reason for it.
    """
    lr = np.log((1 - g.total_discount.to_numpy())
                / (1 - g.d_ref.to_numpy()))
    return {
        "distinct_discounts": int(g.total_discount.nunique()),
        "log_ratio_sd": round(float(np.std(lr)), 6),
        "log_ratio_range": [round(float(lr.min()), 4), round(float(lr.max()), 4)],
        "share_at_reference": round(float(np.mean(np.abs(lr) < 1e-12)), 4),
        "zero_sale_share": round(float((g.units_sold == 0).mean()), 4),
    }


def verdict(naive, ctrl, pv):
    """One sentence a reader can act on. Deliberately blunt: the point of the
    plot is to end an argument, and a hedged caption restarts it."""
    flat = max(naive["span"], ctrl["span"])
    open_both = naive["open_interval"] and ctrl["open_interval"]
    if pv["log_ratio_sd"] < 1e-9:
        return ("NO PRICE VARIATION AT ALL. Every scored row sits at the same "
                "discount ({} distinct value(s)), so `log_ratio` is constant "
                "and epsilon does not enter the likelihood -- the curve is not "
                "flat-ish, it is CONSTANT. Any epsilon on the grid fits "
                "identically, and the reported argmax is the first grid point, "
                "which is the lower bound. Nothing about this category's "
                "elasticity has been measured."
                .format(pv["distinct_discounts"]))
    if flat < 2.0:
        return ("NOT IDENTIFIED. The whole grid is within {:.1f} log-likelihood "
                "units, so the data does not prefer any elasticity in [-4, "
                "-0.05] over any other. No estimator fixes this -- only "
                "exogenous price variation does.".format(flat))
    if open_both:
        return ("BOUNDED ON ONE SIDE AT BEST. Both arms' 95% support runs off "
                "the end of the grid, so the data rules out part of the "
                "support and leaves the rest open. A prior is doing the work.")
    if max(naive["support_95_width"], ctrl["support_95_width"]) > 1.0:
        return ("WEAKLY IDENTIFIED. There is a peak, but the 95% support spans "
                "more than a full unit of elasticity -- wider than the gap "
                "between a bracket and the fallback, which is what the "
                "argument has been about.")
    return ("IDENTIFIED. The support is tighter than 1.0 in elasticity. A "
            "fallback constant is throwing away real information here.")


def build(d, cfg):
    pc = cfg["posterior"]["prior"]
    lo, hi = pc["search_bounds"]
    grid = np.linspace(lo, hi, pc["search_grid_size"])
    model = BaselineModel(cfg)

    entry = entry_rows(d, cfg)
    r_by_cat, pooled, r_basis = ep._reference_r(entry, cfg, model, pc)
    entry["r"] = (entry.category.astype(str).map(r_by_cat)
                  .fillna(float(pooled)).astype(float))

    out = {"grid": [float(x) for x in grid],
           "reference_r_basis": r_basis,
           "rows": int(len(entry)),
           "per_category": {}}
    for cat, g in entry.groupby("category"):
        mu_ref = model.predict_mu_ref(g)
        ll_n = curve(g, mu_ref, grid, controlled=False)
        ll_c = curve(g, mu_ref, grid, controlled=True)
        n, c = read_curve(grid, ll_n), read_curve(grid, ll_c)
        pv = price_variation(g)
        out["per_category"][str(cat)] = {
            "rows": int(len(g)),
            "reference_r": round(float(g.r.iloc[0]), 4),
            "price_variation": pv,
            "naive": n, "controlled": c,
            "bracket_mean": round((n["epsilon_hat"] + c["epsilon_hat"]) / 2, 4),
            "verdict": verdict(n, c, pv),
            # the curves themselves, so the picture can be redrawn without
            # refitting and so a reader can check the shape by eye in the JSON
            "ll_naive": [round(float(x), 4) for x in ll_n],
            "ll_controlled": [round(float(x), 4) for x in ll_c],
        }
    return out


def plot(res, path):
    cats = sorted(res["per_category"])
    grid = np.asarray(res["grid"])
    ncol = 2 if len(cats) > 1 else 1
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.4 * nrow),
                             squeeze=False)
    # ONE Y-SCALE FOR EVERY PANEL. The comparison the plot exists to support is
    # "which categories are flat and which are not", and per-panel autoscaling
    # destroys exactly that -- it draws a 0.01-unit wiggle and a 7-unit slope
    # at the same apparent steepness.
    worst = max(max(p["naive"]["span"], p["controlled"]["span"])
                for p in res["per_category"].values())
    ylim = -min(12.0, max(3.0, worst * 1.05))
    for ax, cat in zip(axes.ravel(), cats):
        p = res["per_category"][cat]
        # naive dashed: where both curves are constant they sit on top of each
        # other, and a solid line hides the other entirely
        for key, colour, style, label in (
                ("ll_naive", MUTED, "--", "naive"),
                ("ll_controlled", ACCENT, "-", "controlled")):
            ll = np.asarray(p[key])
            ax.plot(grid, ll - ll.max(), color=colour, lw=1.6, ls=style,
                    label=label)
        # the 95% cutoff. Everything above this line is elasticity the data
        # cannot distinguish from the best one.
        ax.axhline(-CUTOFF, color=GOOD, lw=1, ls="--")
        ax.text(grid[0], -CUTOFF, " 95% support", color=GOOD, fontsize=7.5,
                va="bottom")
        ax.set_ylim(ylim, 0.6)
        ax.set_title(f"{cat}   n={p['rows']:,}   r={p['reference_r']}",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel("epsilon", fontsize=8.5, color=MUTED)
        ax.set_ylabel("log-lik  -  max", fontsize=8.5, color=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes.ravel()[len(cats):]:
        ax.set_visible(False)
    fig.suptitle("Profile likelihood in epsilon -- a flat curve means the data "
                 "does not identify it", x=0.01, ha="left", fontsize=12,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/prepared.parquet")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="reports/epsilon_profile.json")
    ap.add_argument("--chart", default="reports/charts/epsilon_profile.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    res = build(pd.read_parquet(args.input), cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    plot(res, args.chart)

    print(f"entry rows: {res['rows']:,}   reference r: {res['reference_r_basis']}\n")
    hdr = (f"{'category':14s} {'arm':11s} {'eps':>7s} {'span':>7s} "
           f"{'95% support':>18s} {'width':>7s} {'SE':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for cat in sorted(res["per_category"]):
        p = res["per_category"][cat]
        pv = p["price_variation"]
        # the cause, printed BEFORE the curves it explains
        print(f"{cat:14s} n={p['rows']:,}  r={p['reference_r']}  "
              f"discounts={pv['distinct_discounts']}  "
              f"log_ratio sd={pv['log_ratio_sd']:.5f}  "
              f"at reference={pv['share_at_reference']:.1%}  "
              f"zero-sale={pv['zero_sale_share']:.1%}")
        for arm in ("naive", "controlled"):
            a = p[arm]
            se = "--" if a["se_curvature"] is None else f"{a['se_curvature']:.3f}"
            flags = ("  AT BOUND" if a["at_bound"] else "") + \
                    ("  OPEN" if a["open_interval"] else "")
            print(f"{'':14s} {arm:11s} "
                  f"{a['epsilon_hat']:>7.3f} {a['span']:>7.2f} "
                  f"{str(a['support_95']):>18s} {a['support_95_width']:>7.3f} "
                  f"{se:>7s}{flags}")
        print(f"{'':14s} -> {p['verdict']}\n")
    print(f"wrote {args.out} and {args.chart}")


if __name__ == "__main__":
    main()
