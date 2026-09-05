"""evaluate.derive_thresholds -- evidence for the SET BY OWNER config keys.

Guardrail noise floors on the trailing-mean basis the monitor compares
against (a threshold below 3-sigma false-fires and silently suspends
exploration, design 15.4), and the two learning-rail consistency checks.
Run: python3 -m evaluate.derive_thresholds --input data/prepared.parquet
"""

import argparse
import json

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import write_json
from common import guardrail
from common import metrics
from fit.prepare_data import pre_launch
from common.provenance import config_fingerprint


# ------------------------------------------------------- guardrail floors

# MAD -> sigma for a normal distribution (1 / Phi^-1(3/4)); a constant of the
# estimator, not a tunable
MAD_TO_SIGMA = 1.4826


def _sigma_summary(rel, outlier_ratio):
    """3-sigma and robust 3-sigma of a deviation series; shared by both bases
    so the two floors cannot drift apart. `outlier_dominated` marks a raw
    sigma above `outlier_ratio` x the robust one -- inflated by
    low-denominator days; set thresholds from the robust figure there."""
    sigma = float(rel.std(ddof=1))
    mad = float(np.median(np.abs(rel - np.median(rel))))
    sigma_robust = MAD_TO_SIGMA * mad
    return {
        "daily_rel_dev_sigma": round(sigma, 4),
        "three_sigma": round(3 * sigma, 4),
        "daily_rel_dev_sigma_robust": round(sigma_robust, 4),
        "three_sigma_robust": round(3 * sigma_robust, 4),
        "outlier_dominated": bool(sigma_robust > 0
                                  and sigma > outlier_ratio * sigma_robust),
    }


def _floor_of(block):
    """The floor a threshold must clear: robust where the raw sigma is
    outlier-dominated, otherwise the raw one. Returns (floor, label)."""
    if "three_sigma" not in block:
        return None, None
    if block.get("outlier_dominated"):
        return block["three_sigma_robust"], "robust 3-sigma"
    return block["three_sigma"], "3-sigma"


_smooth = guardrail.smooth   # one definition, shared with daily.monitor


def recommend_thresholds(trailing, cfg):
    """Per metric: the trailing-mean floor (the basis the monitor compares
    against) and whether the configured threshold clears it -- above it, but
    not so far above that the guardrail can never fire."""
    sc = cfg["monitoring"]["stop_conditions"]
    inert_multiple = float(cfg["tuning"]["guardrail_inert_floor_multiple"])
    out = {}
    for metric, key in (("scrap_rate", "scrap_deterioration_pct"),
                        ("margin_rate", "margin_deterioration_pct")):
        floor, label = _floor_of(trailing.get(metric, {}))
        threshold = sc[key]
        metric_key = "scrap" if metric == "scrap_rate" else "margin"
        dev_basis = guardrail.basis_for(metric_key)
        rec = {"config_key": f"monitoring.stop_conditions.{key}",
               "deterioration_basis": dev_basis,
               "units": guardrail.units_of(dev_basis),
               "current_threshold": threshold,
               "trailing_floor": floor}
        if floor is None:
            rec["verdict"] = "insufficient history"
            out[metric] = rec
            continue
        # `binding_*` are what ops.tune and ops.status read; the
        # trailing basis is the only one left, so binding_floor IS
        # trailing_floor and binding_basis is always "trailing"
        rec.update(binding_floor=floor, binding_basis="trailing",
                   binding_label=label)
        # a floor no threshold can clear is a BLOCKED guardrail, not a large
        # number -- said here too, the block an owner reads to pick a value
        if guardrail.floor_is_unusable(floor, dev_basis):
            rec["verdict"] = (
                f"BLOCKED -- the {label} floor is {floor} on the "
                f"RELATIVE basis, i.e. ordinary daily swing exceeds the "
                f"series' own level. No threshold can clear it without also "
                f"clearing the failure this guardrail exists to catch. This is "
                f"the wrong basis for this metric, not a tuning problem: set "
                f"common.guardrail.BASIS[{metric_key!r}] to 'absolute_pp' "
                f"and re-derive.")
            out[metric] = rec
            continue
        if threshold is None:
            rec["verdict"] = (f"null -- owner should set it at or above the "
                              f"{label} trailing floor {floor}")
        elif threshold < floor:
            rec["verdict"] = (f"TOO TIGHT -- {threshold} is below the {label} "
                              f"trailing floor {floor}; it will false-fire "
                              "and silently suspend exploration")
        elif floor > 0 and threshold > inert_multiple * floor:
            # clearing the floor is necessary, not sufficient: a threshold
            # far above it cannot fire either
            rec["verdict"] = (
                f"CLEARS THE FLOOR BUT LIKELY INERT -- {threshold} is "
                f"{round(threshold / floor, 1)}x the {label} trailing floor "
                f"{floor} (tuning.guardrail_inert_floor_multiple "
                f"{inert_multiple:g}), and the {sc['persistence_days']}-day "
                "persistence rule sits on top. A guardrail this loose will "
                "not fire; consider a different metric or an absolute floor "
                "instead")
        else:
            rec["verdict"] = f"OK -- above the {label} trailing floor {floor}"
        out[metric] = rec
    out["note"] = ("The floor is the trailing-mean basis: the monitor compares "
                   "every day against the trailing window of the same "
                   "system-priced episodes, so this is the noise the threshold "
                   "must clear.")
    return out


def guardrail_noise(d, cfg):
    """3-sigma daily noise of scrap rate and realised margin rate, as relative
    deviation from a trailing-window mean. Measurement only: the verdict per
    metric lives in `recommend_thresholds` (guardrail_threshold_recommendation
    in the report), the one place that grades a threshold."""
    mon = cfg["monitoring"]
    window = mon["guardrail_noise_window_days"]
    min_days = window + int(mon["guardrail_noise_min_extra_days"])
    outlier_ratio = float(mon["guardrail_outlier_sigma_ratio"])
    day = metrics.daily_rates(metrics.settled(metrics.episode_economics(d))[0])

    def noise(series, smooth=1, basis=guardrail.RELATIVE):
        # average `smooth` days BEFORE comparing; the trailing baseline is
        # shifted by the same amount so the two windows never overlap
        s = _smooth(series, smooth)
        if len(s) < min_days:
            return {"days": int(len(s)),
                    "note": f"needs at least {min_days} days"}
        # FULL trailing window only: min_periods below `window` manufactures
        # huge deviations that are an estimator artifact, not the series
        trailing = s.rolling(window, min_periods=window).mean().shift(smooth)
        # worse_when_higher=True for BOTH metrics: a noise floor is two-sided
        # (size of swing, not direction); the trigger applies the sign
        rel_dev = guardrail.deviation(s, trailing, True, basis).dropna()
        # plain std of a ratio series is dominated by low-denominator days --
        # set thresholds from the MAD figure when the two disagree. Units
        # depend on the basis; the report says which (RELATIVE: 0.1336 = 13.36%)
        out = {
            "days": int(len(s)),
            "days_scored": int(len(rel_dev)),
            "smoothing_days": smooth,
            "deterioration_basis": basis,
            "units": guardrail.units_of(basis),
            "mean_level": round(float(s.mean()), 4),
            # the fact that decides whether RELATIVE is even defined for this
            # metric: a series that changes sign has no meaningful ratio to
            # its own mean
            "days_at_or_below_zero": int((s <= 0).sum()),
            **_sigma_summary(rel_dev, outlier_ratio),
            "p95_abs_rel_dev": round(float(np.percentile(np.abs(rel_dev), 95)), 4),
            "worst_observed_rel_dev": round(float(rel_dev.abs().max()), 4),
        }
        floor, label = _floor_of(out)
        if guardrail.floor_is_unusable(floor, basis):
            out["unusable_floor"] = (
                f"BLOCKED: the {label} floor is {floor} on the RELATIVE basis "
                f"-- at or above 1.0, so ordinary daily swing exceeds the "
                f"series' own level ({out['mean_level']}) and no threshold can "
                f"sit above it without also clearing the failure the guardrail "
                f"exists to catch. "
                + (f"The series is at or below zero on "
                   f"{out['days_at_or_below_zero']} of {out['days']} days, so "
                   "a ratio to its mean is undefined in practice: switch this "
                   "metric to the absolute_pp basis (common.guardrail.BASIS)."
                   if out["days_at_or_below_zero"] else
                   "Consider the absolute_pp basis (common.guardrail.BASIS), "
                   "or more smoothing if the series is strictly positive."))
        if out["outlier_dominated"]:
            sigma = out["daily_rel_dev_sigma"]
            sigma_robust = out["daily_rel_dev_sigma_robust"]
            out["note"] = (
                f"raw 3-sigma ({out['three_sigma']}) is "
                f"{round(sigma / sigma_robust, 1)}x the robust estimate "
                f"({out['three_sigma_robust']}) -- a few low-denominator days "
                "dominate it. Set the threshold against three_sigma_robust "
                "and investigate the outlier days before trusting either.")
        return out

    sc = cfg["monitoring"]["stop_conditions"]
    sm = sc["deterioration_smoothing_days"]
    scrap = noise(day.scrap_rate, sm["scrap"],
                  guardrail.basis_for("scrap"))
    margin = noise(day.margin_rate, sm["margin"],
                   guardrail.basis_for("margin"))

    return {
        "basis": ("daily ratio-of-sums series over all episodes, smoothed over "
                  "deterioration_smoothing_days; relative deviation vs "
                  f"trailing {window}-day mean -- the basis the monitor "
                  "compares against"),
        "scrap_rate": {**scrap,
                       "config_key": "monitoring.stop_conditions.scrap_deterioration_pct",
                       "verdict_in": "guardrail_threshold_recommendation.scrap_rate"},
        "margin_rate": {**margin,
                        "config_key": "monitoring.stop_conditions.margin_deterioration_pct",
                        "verdict_in": "guardrail_threshold_recommendation.margin_rate"},
        "units": ("all sigma figures are RELATIVE deviations, not percentage "
                  "points: 0.1336 = 13.36%, 9.1386 = 914%. Read the magnitude "
                  "before quoting one -- a floor above 1.0 means the series "
                  "swings by more than its own level and no useful threshold "
                  "sits above it."),
        "note": ("A false fire only suspends exploration (section 15.4), but a "
                 "threshold that fires constantly kills the learning loop. "
                 "Set thresholds at or above the floor the verdict names; add "
                 "a persistence rule rather than lowering below it."),
    }


def information_increment(cfg):
    """information_increment from the posterior arithmetic: precision adds,
    so I* = (1/s0^2) * [1/(1-max_std_shrink)^2 - 1] saturates the shrink cap.
    A CEILING, not a target (excess evidence is clipped away); derived at the
    LAUNCH stds and re-derived after any prior change."""
    s = cfg["learning"]["max_std_shrink"]
    k = 1.0 / (1.0 - s) ** 2 - 1.0
    with open(cfg["posterior"]["prior"]["path"]) as f:
        prior = json.load(f)
    per = prior.get("per_category", {})
    stds = {c: float(v["std"]) for c, v in per.items() if v.get("std")}
    if not stds:
        return {"verdict": "NOT RUN -- no per-category prior stds"}

    need = {c: k / v ** 2 for c, v in stds.items()}
    widest = min(need.values())          # widest std -> cheapest to move
    narrowest = max(need.values())
    configured = float(cfg["learning"]["information_increment"])
    # the std at which the CONFIGURED value exactly saturates the cap: what
    # that number was implicitly sized for
    implied_std = float(np.sqrt(k / configured)) if configured > 0 else None
    median_need = float(np.median(list(need.values())))
    return {
        "max_std_shrink": s,
        "k_shrink_to_cap": round(k, 4),
        "prior_std_by_category": {c: round(v, 4) for c, v in stds.items()},
        "information_to_saturate_cap_by_category": {
            c: round(v, 3) for c, v in sorted(need.items(), key=lambda kv: kv[1])},
        "recommended": round(median_need, 3),
        "recommended_basis": ("median across cells of I* at the LAUNCH prior "
                              "std -- the ceiling above which evidence is "
                              "clipped away rather than used"),
        "range_across_cells": [round(widest, 3), round(narrowest, 3)],
        "configured": configured,
        "configured_implied_std": round(implied_std, 4) if implied_std else None,
        "wastes_at_launch": round(configured / median_need, 1)
            if median_need > 0 else None,
        "verdict": (
            "OK -- at or below the launch ceiling" if configured <= median_need
            else f"TOO LARGE -- {configured / median_need:.1f}x the launch "
                 f"ceiling; every update gathers that multiple of the evidence "
                 f"one capped step can use, and bounded_step discards the rest"),
        "note": ("Information here is the NB Fisher information "
                 "mu*L^2*r/(r+mu) that `daily.update` accumulates, AFTER "
                 "deff deflation -- the same quantity the trigger compares. "
                 "Re-derive after any change to the prior or to "
                 "max_std_shrink. As the posterior narrows the ceiling rises "
                 "as 1/std^2, so a value set for launch becomes conservative "
                 "later: that direction is safe, the other is not."),
    }


def bounded_step(cfg):
    """The two rails on one update, and whether they trip at the SAME
    surprise. A cap-sized update moves the mean by
    [1 - (1-max_std_shrink)^2] x |batch - mean| (Normal approx), so a
    max_mean_step far below that clips on ordinary batches while the std rail
    never binds. Neither is derivable (risk appetite); their CONSISTENCY is.
    Cross-check the price consequence in backtest.step_sensitivity."""
    lc = cfg["learning"]
    shrink, step = lc["max_std_shrink"], lc["max_mean_step"]
    band_lo, band_hi = (float(x) for x in
                        cfg["tuning"]["bounded_step_consistent_band"])
    pull_frac = 1.0 - (1.0 - shrink) ** 2
    min_std = cfg["posterior"]["min_std"]
    with open(cfg["posterior"]["prior"]["path"]) as f:
        prior = json.load(f)
    stds = {c: float(v["std"]) for c, v in prior.get("per_category", {}).items()
            if v.get("std")}
    if not stds:
        return {"verdict": "NOT RUN -- no per-category prior stds"}

    med = float(np.median(list(stds.values())))
    # what a cap-sized update does to the mean, per prior std of surprise
    move_per_std = pull_frac * med
    # ...and therefore the surprise at which the MEAN rail starts clipping
    clips_at_std = step / move_per_std if move_per_std > 0 else None
    # convergence the shrink cap allows, at one human-gated update per day
    updates_to_floor = {
        c: round(float(np.log(min_std / v) / np.log(1.0 - shrink)), 1)
        for c, v in sorted(stds.items())}

    consistent = pull_frac * med
    return {
        "max_std_shrink": shrink,
        "max_mean_step": step,
        "median_launch_std": round(med, 4),
        "mean_move_fraction_of_pull_at_cap": round(pull_frac, 4),
        "mean_move_at_cap_per_prior_std": round(move_per_std, 4),
        "mean_rail_clips_above_pull_of_std": round(clips_at_std, 3)
            if clips_at_std else None,
        "updates_to_min_std_by_category": updates_to_floor,
        "days_to_min_std_median": round(float(np.median(
            list(updates_to_floor.values()))), 1),
        "consistent_max_mean_step": round(consistent, 3),
        "consistent_basis": ("the mean move a CAP-SIZED update makes on a "
                             "one-prior-std surprise -- set here, both rails "
                             "trip at the same surprise instead of one "
                             "clipping every batch"),
        "consistent_band_std": [band_lo, band_hi],
        "verdict": (
            f"CONSISTENT -- both rails trip near a "
            f"{clips_at_std:.2f}-std surprise (inside "
            f"tuning.bounded_step_consistent_band [{band_lo:g}, {band_hi:g}])"
            if clips_at_std is not None and band_lo <= clips_at_std <= band_hi else
            f"MEAN RAIL BINDS FIRST -- max_mean_step {step} clips at a "
            f"{clips_at_std:.2f}-std surprise while max_std_shrink needs a "
            f"full cap-sized update, so the mean cap does the work and "
            f"`bound_clipped` fires routinely. OWNER DECISION: raise "
            f"max_mean_step toward {consistent:.2f} (check the price "
            f"consequence in backtest.step_sensitivity FIRST), or lower "
            f"max_std_shrink so the two agree"
            if clips_at_std is not None and clips_at_std < band_lo else
            f"STD RAIL BINDS FIRST -- max_mean_step {step} only clips beyond a "
            f"{clips_at_std:.2f}-std surprise, so convergence is limited by "
            f"max_std_shrink alone" if clips_at_std is not None else
            "NOT RUN -- degenerate prior widths"),
        "note": ("Both rails widen in effect as the posterior narrows: "
                 "mean_move_at_cap scales with the CURRENT std, so a launch "
                 "derivation is the tight case and the rails loosen from "
                 "there. max_std_shrink is the one to set first -- "
                 "information_increment is derived from it, and changing it "
                 "moves that too."),
    }


def main():
    ap = argparse.ArgumentParser(prog="evaluate.derive_thresholds")
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="reports/thresholds.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # RULE 16: the hold-out is read once, by evaluate.shadow. These floors
    # are PASTED into config by ops.tune, so
    # measuring them on the full extract tunes config on the window that
    # exists to grade it. backtest cuts the same way (evaluate/backtest.py).
    d = pre_launch(pd.read_parquet(args.input), cfg)

    trailing = guardrail_noise(d, cfg)
    report = {
        "config": config_fingerprint(cfg, "backtest"),
        "guardrail_noise": trailing,
        "guardrail_threshold_recommendation": recommend_thresholds(trailing, cfg),
        "information_increment_recommendation": information_increment(cfg),
        "bounded_step_recommendation": bounded_step(cfg),
    }

    write_json(args.out, report)

    gn = report["guardrail_noise"]
    for key in ("scrap_rate", "margin_rate"):
        block = gn[key]
        sigma3 = block.get("three_sigma", "n/a")
        line = f"{key:12s}: trailing 3-sigma {sigma3}"
        if block.get("outlier_dominated"):
            line += (f" (OUTLIER-DOMINATED; robust "
                     f"{block['three_sigma_robust']})")
        print(line)
    # the sign-off line: both floors side by side and which one binds
    for key, rec in report["guardrail_threshold_recommendation"].items():
        if not isinstance(rec, dict):
            continue
        print(f"{key:12s}: trailing floor {rec.get('trailing_floor')} "
              f"-> {rec['verdict']}")
    ii = report["information_increment_recommendation"]
    if isinstance(ii, dict) and "recommended" in ii:
        print(f"{'info increment':12s}: configured {ii['configured']} | launch "
              f"ceiling {ii['recommended']} "
              f"(range {ii['range_across_cells']}) -> {ii['verdict']}")
    bs = report["bounded_step_recommendation"]
    if isinstance(bs, dict) and "consistent_max_mean_step" in bs:
        print(f"{'bounded step':12s}: mean {bs['max_mean_step']} / shrink "
              f"{bs['max_std_shrink']} | mean rail clips above a "
              f"{bs['mean_rail_clips_above_pull_of_std']}-std surprise | "
              f"{bs['days_to_min_std_median']} updates to min_std")
        print(f"{'':12s}  -> {bs['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
