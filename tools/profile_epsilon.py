"""tools.profile_epsilon -- draw the profile likelihood the prior is built on.

A DIAGNOSTIC, NOT A FIT: it writes nothing the pipeline reads. It renders the
same censored-Poisson curves `bootstrap.prior_density` turns into the prior --
per category, both arms, both row sets -- so a human can SEE whether epsilon
is identified instead of inferring it from summary numbers. A flat curve means
the data does not identify epsilon; no estimator fixes that, only exogenous
price variation does (pricing.explore).

Reads the current config for `hour_control` and `min_rows_per_time_cell`, and
shows both row sets side by side regardless of which is configured, because
the comparison is the point of the picture.

Usage:
    python3 -m tools.profile_epsilon --input data/prepared.parquet
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

from common.config import load_config
from bootstrap.train_baseline import BaselineModel
from bootstrap import prior_density as pdn

INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#c1440e"
GOOD = "#2f6b4f"

# 0.5 * chi2(1, 0.95): the standard profile-likelihood cutoff
CUTOFF = 1.920729

SETS = ("entry", "all_stocked_hours")


def build(d, cfg):
    pc = cfg["posterior"]["prior"]
    lo, hi = pc["search_bounds"]
    # extended PAST zero so a wrong-signed peak is visible instead of clipped
    grid = np.linspace(lo, max(1.0, hi), pc["search_grid_size"])
    model = BaselineModel(cfg)

    out = {"grid": [float(x) for x in grid],
           "hour_control": pc.get("hour_control", "date_hour"),
           "per_category": {}}
    for which in SETS:
        alt = copy.deepcopy(cfg)
        alt["posterior"]["prior"]["rows"] = which
        for cat, c in pdn.build_curves(d, alt, model, grid, "train").items():
            entry = out["per_category"].setdefault(cat, {})
            peak = max(float(grid[int(np.argmax(c["naive"]))]),
                       float(grid[int(np.argmax(c["controlled"]))]))
            entry[which] = {
                "rows": c["rows"], "deff": round(c["deff"], 3),
                "log_ratio_sd": round(c["log_ratio_sd"], 6),
                "identifying_variation_share": round(
                    c["identifying_variation_share"], 4),
                "median_rows_per_time_cell": c["median_rows_per_time_cell"],
                "span": round(float(max(np.ptp(c["naive"]),
                                        np.ptp(c["controlled"]))
                                    / c["deff"]), 2),
                "unconstrained_peak": round(peak, 4),
                "wrong_sign": bool(peak >= -float(grid[1] - grid[0])),
                "ll_naive": [round(float(x) / c["deff"], 4)
                             for x in c["naive"]],
                "ll_controlled": [round(float(x) / c["deff"], 4)
                                  for x in c["controlled"]],
            }
    return out


def plot(res, path):
    cats = sorted(res["per_category"])
    grid = np.asarray(res["grid"])
    fig, axes = plt.subplots(len(cats), len(SETS),
                             figsize=(5.8 * len(SETS), 2.9 * len(cats)),
                             squeeze=False)
    # ONE y-scale for every panel: the comparison is "which of these is flat",
    # and per-panel autoscaling draws a 0.01-unit wiggle and a 30-unit slope
    # at the same apparent steepness
    worst = max(p[s]["span"] for p in res["per_category"].values()
                for s in SETS if s in p)
    ylim = -min(20.0, max(3.0, worst * 1.05))
    for row, cat in enumerate(cats):
        for col, s in enumerate(SETS):
            ax = axes[row][col]
            p = res["per_category"][cat].get(s)
            if p is None:
                ax.set_visible(False)
                continue
            for key, colour, style, label in (
                    ("ll_naive", MUTED, "--", "naive"),
                    ("ll_controlled", ACCENT, "-", "controlled")):
                ll = np.asarray(p[key])
                ax.plot(grid, ll - ll.max(), color=colour, lw=1.6, ls=style,
                        label=label)
            ax.axhline(-CUTOFF, color=GOOD, lw=1, ls="--")
            ax.axvline(0.0, color=INK, lw=0.8, ls=":")
            ax.set_ylim(ylim, 0.6)
            flag = "  WRONG SIGN" if p["wrong_sign"] else ""
            ax.set_title(f"{cat} -- {s}   n={p['rows']:,}  deff={p['deff']}  "
                         f"sd(lr)={p['log_ratio_sd']:.4f}{flag}",
                         loc="left", fontsize=9,
                         color=ACCENT if p["wrong_sign"] else INK)
            ax.set_xlabel("epsilon", fontsize=8, color=MUTED)
            if col == 0:
                ax.set_ylabel("log-lik / deff  -  max", fontsize=8,
                              color=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Profile likelihood in epsilon -- flat means unidentified; "
                 "a peak right of the dotted zero line is WRONG-SIGNED",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
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

    print(f"hour control: {res['hour_control']}\n")
    hdr = (f"{'category':12s} {'set':18s} {'n':>6s} {'sd(lr)':>8s} "
           f"{'span':>8s} {'peak':>8s} {'rows/cell':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for cat in sorted(res["per_category"]):
        for s in SETS:
            p = res["per_category"][cat].get(s)
            if not p:
                continue
            flag = "  WRONG SIGN" if p["wrong_sign"] else ""
            print(f"{cat if s == SETS[0] else '':12s} {s:18s} "
                  f"{p['rows']:>6,} {p['log_ratio_sd']:>8.5f} "
                  f"{p['span']:>8.2f} {p['unconstrained_peak']:>+8.3f} "
                  f"{p['median_rows_per_time_cell']:>10.1f}{flag}")
        print()
    print(f"wrote {args.out} and {args.chart}")


if __name__ == "__main__":
    main()
