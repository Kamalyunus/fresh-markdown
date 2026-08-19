"""tools.make_charts -- one diagnostic chart per component, from the reports.

Every chart is generated from a report artifact, never hand-drawn, so a chart
that disagrees with the pipeline cannot exist: re-run this after a bootstrap
and the pictures move with the numbers. Missing reports are skipped with a
note rather than failing, so the tool is useful at any stage of the pipeline.

Each chart answers ONE question about ONE component, and the question is the
title. Charts are deliberately plain -- no gridline decoration, no dual axes,
no colour carrying information a label could carry -- because these go into a
design document where a misread costs more than a dull picture.

Exactly the seven charts `docs/design.md` embeds are generated. Five others
existed and nothing referenced them; a picture nobody looks at still has to be
kept true. **The numbering has gaps (02-06, 08, 09) and must keep them** --
design.md embeds these by filename, so renumbering would silently blank the
images it shows.

Usage:
    python3 -m tools.make_charts [--reports reports] [--out reports/charts]
"""

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#c1440e"          # the one thing to look at
GOOD = "#2f6b4f"
FILL = "#d9d3c7"


def _style(ax, title, subtitle=None, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, color=INK,
                 pad=34 if subtitle else 8)
    if subtitle:
        # wrap by hand: matplotlib will not, and a subtitle running off the
        # canvas is worse than two lines
        words, lines, cur = subtitle.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 108:
                lines.append(cur); cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        ax.text(0, 1.015, "\n".join(lines), transform=ax.transAxes,
                fontsize=8.5, color=MUTED, va="bottom", linespacing=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)


def _save(fig, out, name, written):
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    written.append(name)


def chart_episode_endings(phase0, out, written):
    """common.episodes -- how episodes end, and which ending owns the scrap."""
    e = phase0.get("m11_episode_endings")
    if not isinstance(e, dict):
        return
    shares = e["shares"]
    labels = ["sold out\n(nothing left)", "ended with stock\n(scrap)",
              "not closed\n(still running)"]
    vals = [shares["sold_out_early"], shares["completed"],
            shares["not_closed"]]
    colors = [GOOD, ACCENT, MUTED]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.1%}",
                va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.invert_yaxis()
    _style(ax, "Two endings, and one state that is not an ending",
           "Scrap is the leftover on the last row: zero if it sold out, the "
           "leftover if it ended holding stock. `not_closed` is an unfinished "
           "episode -- UNKNOWN, not zero -- and is empty on a closed extract.")
    _save(fig, out, "02_episode_endings.png", written)


def chart_gate(fid, out, written):
    """Calibration gate -- the frozen model's only production responsibility."""
    by_win = fid.get("by_window") or {}
    if not by_win:
        return
    lo, hi = fid["calibration_gate_band"]
    names = [k for k in ("train", "calib", "test", "all") if k in by_win]
    vals = [by_win[k]["sold_ratio"] for k in names]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhspan(lo, hi, color=GOOD, alpha=0.10)
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.axhline(lo, color=GOOD, lw=1); ax.axhline(hi, color=GOOD, lw=1)
    cols = [ACCENT if n == fid.get("gate_window") else INK for n in names]
    ax.bar(names, vals, color=cols, width=0.5)
    for n, v, c in zip(names, vals, cols):
        ax.text(n, v + 0.012, f"{v:.4f}", ha="center", fontsize=9, color=c)
    gv, verdict = fid.get("calibration_gate_value"), fid.get("calibration_gate", "")
    _style(ax, "Calibration gate: realised sales ÷ predicted, by window",
           f"Gate metric {fid.get('calibration_gate_metric')} = {gv} on "
           f"`{fid.get('gate_window')}` (highlighted) — {verdict.split('--')[0].strip()}. "
           f"Band [{lo}, {hi}] is ~2σ of measured weekly volatility.",
           ylabel="sold / predicted")
    _save(fig, out, "03_calibration_gate.png", written)


def chart_weekly_fidelity(fid, out, written):
    """Why the gate band is 2σ and not ±5%: the weekly series itself."""
    weeks = fid.get("by_week") or {}
    if len(weeks) < 4:
        return
    keys = sorted(weeks)
    vals = [weeks[k] if isinstance(weeks[k], (int, float))
            else weeks[k].get("sold_ratio") for k in keys]
    vals = [v for v in vals if v is not None]
    lo, hi = fid["calibration_gate_band"]

    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.axhspan(lo, hi, color=GOOD, alpha=0.10)
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.plot(range(len(vals)), vals, color=INK, lw=1.4, marker="o", ms=3)
    ax.set_xticks(range(0, len(keys), max(1, len(keys) // 8)))
    ax.set_xticklabels([keys[i][:10] for i in
                        range(0, len(keys), max(1, len(keys) // 8))],
                       fontsize=7, rotation=45, ha="right")
    _style(ax, "Weekly sold-ratio — the volatility the gate band must absorb",
           f"{len(vals)} weeks, spread {min(vals):.2f}–{max(vals):.2f}. "
           "A ±5% band would be about 1σ of this: a coin flip on noise.",
           ylabel="sold / predicted")
    _save(fig, out, "04_weekly_fidelity.png", written)


def chart_policy(pol, out, written):
    """backtest -- where the DP's IL advantage comes from, and what it costs."""
    keys = ("legacy_model", "dp")
    if not all(f"{k}_il" in pol for k in keys):
        return
    disc = [pol[f"{k}_discount_cost"] for k in keys]
    scrap = [pol[f"{k}_scrap_cost"] for k in keys]
    labels = ["legacy\n(under model)", "DP\n(under model)"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.3),
                                  gridspec_kw={"width_ratios": [2, 1]})
    ax.bar(labels, disc, color=ACCENT, width=0.5, label="discount cost")
    ax.bar(labels, scrap, bottom=disc, color=MUTED, width=0.5,
           label="scrap cost")
    for i, (d, s) in enumerate(zip(disc, scrap)):
        ax.text(i, d + s, f"  {(d + s) / 1e6:,.1f}M", ha="center",
                va="bottom", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    gap = pol.get("policy_gap_like_for_like", {})
    _style(ax, "Inventory Loss, like-for-like",
           f"DP reduces IL by {gap.get('dp_il_reduction_pct_of_legacy', 0):.1%} "
           "— by discounting less, not by clearing more.",
           ylabel="currency")

    cl = [pol.get("legacy_model_clearance"), pol.get("dp_clearance")]
    ax2.bar(labels, cl, color=[MUTED, ACCENT], width=0.5)
    for i, v in enumerate(cl):
        ax2.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9, color=INK)
    ax2.set_ylim(0, max(cl) * 1.25)
    _style(ax2, "…and what it costs",
           f"clearance {gap.get('clearance_delta', 0):+.2%}")
    _save(fig, out, "05_policy_il_and_clearance.png", written)


def chart_deepening(pol, out, written):
    """pricing.dp -- why the planner holds price at the launch prior."""
    ied = pol.get("intra_episode_deepening")
    if not isinstance(ied, dict):
        return
    thr = ied.get("median_threshold_abs_eps")
    use = ied.get("median_abs_eps_in_use")
    if thr is None or use is None:
        return

    # the closed form, so the reader sees WHY the bar sits where it does
    d = np.linspace(0.0, 0.5, 200)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for gamma, ls in ((0.60, ":"), (0.66, "-"), (0.70, "--")):
        bar = np.where(gamma - d > 0.01, (1 - d) / np.maximum(gamma - d, 1e-9),
                       np.nan)
        ax.plot(d, bar, color=INK, ls=ls, lw=1.3,
                label=f"cost ratio {gamma:.2f}")
    ax.axhline(use, color=ACCENT, lw=1.6)
    ax.text(0.005, use + 0.08, f"launch prior |ε| = {use}", color=ACCENT,
            fontsize=9)
    ax.axhline(thr, color=GOOD, lw=1.2, ls="-.")
    ax.text(0.005, thr + 0.08, f"measured median bar = {thr}", color=GOOD,
            fontsize=9)
    ax.set_ylim(0, max(4.0, thr * 1.6))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "Deepening only pays above |ε| = (1−d) / (γ−d)",
           f"{ied.get('share_episodes_eps_above_threshold', 0):.0%} of episodes "
           "clear the bar, so day one is enter-shallow-and-hold. Widening the "
           "action set cannot change this; only the posterior can.",
           xlabel="current discount d", ylabel="|ε| required")
    _save(fig, out, "06_deepening_threshold.png", written)


def chart_ab_duration(th, out, written):
    """derive_thresholds -- how long the A/B must run."""
    ab = th.get("ab_duration") or {}
    rows = ab.get("by_duration") or {}
    if not rows:
        return
    labels = list(rows)
    mde = [rows[k]["detectable_mde_rel"] for k in labels]
    blocks = [rows[k]["blocks_measured"] for k in labels]
    target = ab.get("target_mde_rel")

    fig, ax = plt.subplots(figsize=(9, 4.3))
    cols = [INK if b >= 4 else MUTED for b in blocks]
    ax.bar(labels, mde, color=cols, width=0.55)
    for i, (m, b) in enumerate(zip(mde, blocks)):
        ax.text(i, m + 0.001, f"{m:.2%}\n{b} blk", ha="center", fontsize=8,
                color=INK if b >= 4 else MUTED)
    if target:
        ax.axhline(target, color=ACCENT, lw=1.5)
        ax.text(len(labels) - 0.4, target, f"  target {target:.1%}",
                color=ACCENT, fontsize=9, va="bottom", ha="right")
    ax.set_ylim(0, max(mde) * 1.35)
    _style(ax, "Detectable effect by A/B duration",
           "Measured on real T-week blocks, not √T-scaled. Grey bars rest on "
           "fewer than 4 blocks and are mostly noise — read the dark ones.",
           ylabel="smallest detectable relative effect on IL%")
    _save(fig, out, "08_ab_duration.png", written)


def chart_guardrail_noise(th, cfg, out, written):
    """derive_thresholds -- why scrap needs smoothing and margin does not."""
    gn = th.get("guardrail_noise") or {}
    if "scrap_rate" not in gn or "three_sigma" not in gn["scrap_rate"]:
        return
    sm = cfg["monitoring"]["stop_conditions"]["deterioration_smoothing_days"]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    names, raw, rob, cols = [], [], [], []
    for key, label in (("scrap_rate", "scrap"), ("margin_rate", "margin")):
        b = gn.get(key, {})
        if "three_sigma" not in b:
            continue
        names.append(f"{label}\n({sm[label]}-day average)")
        raw.append(b["three_sigma"])
        rob.append(b.get("three_sigma_robust", b["three_sigma"]))
        cols.append(ACCENT if b.get("outlier_dominated") else GOOD)
    x = np.arange(len(names))
    ax.bar(x - 0.18, raw, width=0.34, color=MUTED, label="3σ raw")
    ax.bar(x + 0.18, rob, width=0.34, color=cols, label="3σ robust (MAD)")
    for i, (r, o) in enumerate(zip(raw, rob)):
        ax.text(i - 0.18, r, f" {r:.2f}", ha="center", va="bottom", fontsize=8,
                color=MUTED)
        ax.text(i + 0.18, o, f" {o:.2f}", ha="center", va="bottom", fontsize=8,
                color=INK)
    ax.axhline(1.0, color=INK, lw=1, ls=":")
    ax.text(len(names) - 0.5, 1.02, "swings by its own level", fontsize=8,
            color=INK, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Guardrail noise floors — a threshold must sit above these",
           "Red = outlier-dominated, so read the robust bar. A floor above "
           "1.0 means no threshold on that basis is both safe and useful.",
           ylabel="3σ relative deviation")
    _save(fig, out, "09_guardrail_noise.png", written)


def main():
    ap = argparse.ArgumentParser(prog="tools.make_charts")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="reports/charts")
    args = ap.parse_args()

    from common.config import load_config
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    def load(path):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    bt = load(f"{args.reports}/backtest.json")
    th = load(f"{args.reports}/thresholds.json")
    p0 = load(f"{args.reports}/phase0.json")

    written, skipped = [], []
    jobs = [
        ("episode endings", p0, lambda: chart_episode_endings(p0, args.out, written)),
        ("calibration gate", bt, lambda: chart_gate(bt["fidelity"], args.out, written)),
        ("weekly fidelity", bt, lambda: chart_weekly_fidelity(bt["fidelity"], args.out, written)),
        ("policy", bt, lambda: chart_policy(bt["policy_deltas"], args.out, written)),
        ("deepening", bt, lambda: chart_deepening(bt["policy_deltas"], args.out, written)),
        ("A/B duration", th, lambda: chart_ab_duration(th, args.out, written)),
        ("guardrail noise", th, lambda: chart_guardrail_noise(th, cfg, args.out, written)),
    ]
    for name, source, fn in jobs:
        if source is None:
            skipped.append(f"{name} (no report)")
            continue
        before = len(written)
        # a chart that reads a field an OLDER report does not carry raised
        # straight out of the loop and took the other six down with it, which
        # contradicts the skip-don't-fail contract above. Name the missing
        # field and carry on -- one stale report must not cost every picture.
        try:
            fn()
        except KeyError as e:
            skipped.append(f"{name} (report lacks {e})")
            continue
        if len(written) == before:
            skipped.append(f"{name} (report lacks the fields)")

    for n in written:
        print(f"  wrote {args.out}/{n}")
    for s in skipped:
        print(f"  skipped {s}")
    print(f"{len(written)} charts in {args.out}")


if __name__ == "__main__":
    main()
