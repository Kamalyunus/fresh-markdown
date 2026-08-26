"""bootstrap.derive_thresholds -- evidence for the SET BY OWNER config keys.

Derives (1) A/B duration vs MDE: the clustered SE of the IL% ratio estimator,
measured EMPIRICALLY on candidate-duration blocks (never sqrt(T)-scaled), with
MDE_abs(T) = (z_{1-a/2} + z_pow) x 2 x SE_pooled(T) at 50/50; and (2) guardrail
noise floors: a threshold below 3-sigma false-fires and silently suspends
exploration (design 15.4).
Run: python3 -m bootstrap.derive_thresholds --input data/prepared.parquet [--mde 0.075]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from common.ab import arm
from common.config import load_config
from common import episodes
from common import guardrail
from bootstrap.measure import m6_il_pct


# ------------------------------------------------------------- A/B duration

def empirical_se_by_duration(d, cfg):
    """Median clustered SE of the IL% ratio estimator over non-overlapping
    T-week blocks of history, per candidate duration T."""
    ab = cfg["ab_test"]
    dates = pd.to_datetime(d.date)
    d = d.assign(_date=dates)
    start, end = dates.min(), dates.max()

    out = {}
    for weeks in ab["candidate_durations_weeks"]:
        span = pd.Timedelta(weeks=weeks)
        ses, aggs, blocks = [], [], 0
        t = start
        while t + span <= end + pd.Timedelta(days=1):
            block = d[(d._date >= t) & (d._date < t + span)]
            t = t + span
            if block.episode_id.nunique() < ab["min_episodes_per_block"]:
                continue
            m6 = m6_il_pct(block)
            if m6["il_pct_ratio_se_clustered"] is None:
                continue
            blocks += 1
            ses.append(m6["il_pct_ratio_se_clustered"])
            aggs.append(m6["il_pct_aggregate"])
        if ses:
            out[weeks] = {"blocks": blocks,
                          "se_pooled_median": float(np.median(ses)),
                          "il_pct_median": float(np.median(aggs))}
    return out


def duration_table(se_by_T, cfg, mde_rel):
    ab = cfg["ab_test"]
    z = float(norm.ppf(1 - ab["alpha"] / 2) + norm.ppf(ab["power"]))
    # pooled all-units SE -> between-arm difference SE:
    # SE_diff = SE_pooled / sqrt(a(1-a))  (= 2 x SE_pooled at 50/50)
    arm_factor = 1.0 / np.sqrt(ab["allocation"] * (1 - ab["allocation"]))

    rows, recommended = {}, None
    for weeks in sorted(se_by_T):
        e = se_by_T[weeks]
        se_diff = arm_factor * e["se_pooled_median"]
        mde_abs = z * se_diff
        mde_rel_t = mde_abs / e["il_pct_median"] if e["il_pct_median"] else None
        meets = (mde_rel_t is not None and mde_rel is not None
                 and mde_rel_t <= mde_rel)
        rows[f"{weeks}w"] = {
            "blocks_measured": e["blocks"],
            "se_pooled": round(e["se_pooled_median"], 6),
            "se_arm_difference": round(se_diff, 6),
            "detectable_mde_abs": round(mde_abs, 5),
            "detectable_mde_rel": round(mde_rel_t, 4) if mde_rel_t else None,
            "meets_target": meets,
        }
        if meets and recommended is None:
            recommended = weeks
    return {
        "z_factor": round(z, 4),
        "power": ab["power"], "alpha": ab["alpha"],
        "target_mde_rel": mde_rel,
        "by_duration": rows,
        "recommended_duration_weeks": recommended,
        "recommended_duration_days": recommended * 7 if recommended else None,
        "note": ("SE measured on actual T-week blocks, not sqrt-scaled. "
                 "If no duration meets the target, either the MDE or the MVP "
                 "window must change (design 5.3 reassessment, measurement 6)."),
    }


# --------------------------------------------------------- guardrail noise

def _daily_series(d):
    ep = d.sort_values(["date", "hour_of_day"]).groupby("episode_id").agg(
        date=("date", "first"),
        start_inv=("starting_inventory", "first"),
        sold=("units_sold", "sum"))
    # ending with stock on hand IS disposal; only extract-boundary episodes
    # are unknown -- the floor is measured on neither invented numbers nor a
    # sliver of the population
    ep["end_inv"] = episodes.scrap_units(d)
    ep = ep[ep.end_inv.notna()]
    rev = ((d.offered_price * d.units_sold).groupby(d.episode_id).sum()
           .rename("revenue"))
    margin = (((d.offered_price - d.cost) * d.units_sold)
              .groupby(d.episode_id).sum().rename("margin"))
    ep = ep.join(rev).join(margin)

    day = ep.groupby("date").agg(start_inv=("start_inv", "sum"),
                                 end_inv=("end_inv", "sum"),
                                 revenue=("revenue", "sum"),
                                 margin=("margin", "sum")).sort_index()
    day["scrap_rate"] = day.end_inv / day.start_inv
    day["margin_rate"] = day.margin / day.revenue.replace(0, np.nan)
    return day


def _sigma_summary(rel):
    """3-sigma and robust 3-sigma of a deviation series; shared by both bases
    so the two floors cannot drift apart. `outlier_dominated` marks a raw
    sigma inflated by low-denominator days -- set thresholds from the robust
    figure there."""
    sigma = float(rel.std(ddof=1))
    mad = float(np.median(np.abs(rel - np.median(rel))))
    sigma_robust = 1.4826 * mad
    return {
        "daily_rel_dev_sigma": round(sigma, 4),
        "three_sigma": round(3 * sigma, 4),
        "daily_rel_dev_sigma_robust": round(sigma_robust, 4),
        "three_sigma_robust": round(3 * sigma_robust, 4),
        "outlier_dominated": bool(sigma_robust > 0 and sigma > 2 * sigma_robust),
    }


def _floor_of(block):
    """The floor a threshold must clear: robust where the raw sigma is
    outlier-dominated, otherwise the raw one. Returns (floor, label)."""
    if "three_sigma" not in block:
        return None, None
    if block.get("outlier_dominated"):
        return block["three_sigma_robust"], "robust 3-sigma"
    return block["three_sigma"], "3-sigma"


_smooth = guardrail.smooth   # one definition, shared with pipeline.monitor


def control_arm_noise(d, cfg):
    """Same-day treatment-vs-control noise -- the basis the monitor uses once
    the A/B is live (cancels the common day effect the trailing basis keeps).
    Both arms are smoothed over deterioration_smoothing_days BEFORE
    differencing, exactly as pipeline.monitor.deterioration does -- an
    unsmoothed floor overstates by up to ~sqrt(smooth) and leaves the
    guardrail inert. Arm assignment is the monitor's own common.ab.arm."""

    ep = d.sort_values(["date", "hour_of_day"]).groupby("episode_id").agg(
        date=("date", "first"), sku_id=("sku_id", "first"), fc=("fc", "first"),
        start_inv=("starting_inventory", "first"))
    ep["scrap"] = episodes.scrap_units(d)
    ep = ep[ep.scrap.notna()]
    rev = ((d.offered_price * d.units_sold).groupby(d.episode_id).sum()
           .rename("revenue"))
    mar = (((d.offered_price - d.cost) * d.units_sold)
           .groupby(d.episode_id).sum().rename("margin"))
    ep = ep.join(rev).join(mar)
    alloc = cfg["ab_test"]["allocation"]
    ep["arm"] = [arm(s, f, alloc) for s, f in zip(ep.sku_id, ep.fc)]

    def daily(g):
        day = g.groupby("date").agg(start_inv=("start_inv", "sum"),
                                    scrap=("scrap", "sum"),
                                    revenue=("revenue", "sum"),
                                    margin=("margin", "sum")).sort_index()
        return pd.DataFrame({"scrap_rate": day.scrap / day.start_inv,
                             "margin_rate": day.margin
                             / day.revenue.replace(0, np.nan)})

    arms = {a: daily(g) for a, g in ep.groupby("arm")}
    if set(arms) < {"treatment", "control"}:
        return {"note": "one arm empty -- cannot measure a same-day basis"}

    sm = cfg["monitoring"]["stop_conditions"]["deterioration_smoothing_days"]
    out = {"basis": ("same-day treatment vs control, each arm smoothed over "
                     "deterioration_smoothing_days before differencing, "
                     "arm hash as in monitor"),
           "allocation": alloc}
    for metric, worse_high, key in (("scrap_rate", True, "scrap"),
                                    ("margin_rate", False, "margin")):
        smooth = sm[key]
        basis = guardrail.basis_for(cfg, key)
        # smooth each arm FIRST, then intersect and difference -- monitor's
        # order; anything else measures a floor the live comparison never sees
        t = _smooth(arms["treatment"][metric], smooth)
        c = _smooth(arms["control"][metric], smooth)
        common = t.index.intersection(c.index)
        t, c = t.loc[common], c.loc[common]
        rel = guardrail.deviation(t, c, worse_high, basis)
        rel = rel.replace([np.inf, -np.inf], np.nan).dropna()
        if len(rel) < 8:
            out[metric] = {"days": int(len(rel)), "smoothing_days": smooth,
                           "deterioration_basis": basis,
                           "note": f"too few paired days after {smooth}-day "
                                   "smoothing"}
            continue
        out[metric] = {
            "days": int(len(rel)),
            "smoothing_days": smooth,
            "deterioration_basis": basis,
            "units": guardrail.units_of(basis),
            **_sigma_summary(rel),
            "median_gap": round(float(np.median(rel)), 4),
        }
    out["note"] = ("Set the A/B-phase threshold against THIS floor. The "
                   "trailing-mean floor in guardrail_noise applies only "
                   "before an A/B is running, where no control arm exists. "
                   "One config value serves both phases, so it must clear the "
                   "LARGER of the two -- see guardrail_threshold_recommendation.")
    return out


def recommend_thresholds(trailing, control_arm, cfg):
    """Per metric: the floor from each basis, which one binds, and whether the
    configured threshold clears it. One config value is graded against the
    trailing basis before the A/B and the control-arm basis during it, so it
    must sit above BOTH."""
    sc = cfg["monitoring"]["stop_conditions"]
    out = {}
    for metric, key in (("scrap_rate", "scrap_deterioration_pct"),
                        ("margin_rate", "margin_deterioration_pct")):
        t_floor, t_label = _floor_of(trailing.get(metric, {}))
        c_floor, c_label = _floor_of((control_arm or {}).get(metric, {})
                                     if isinstance(control_arm, dict) else {})
        known = [(f, lab, basis) for f, lab, basis in
                 ((t_floor, t_label, "trailing"),
                  (c_floor, c_label, "control_arm")) if f is not None]
        threshold = sc[key]
        metric_key = "scrap" if metric == "scrap_rate" else "margin"
        dev_basis = guardrail.basis_for(cfg, metric_key)
        rec = {"config_key": f"monitoring.stop_conditions.{key}",
               "deterioration_basis": dev_basis,
               "units": guardrail.units_of(dev_basis),
               "current_threshold": threshold,
               "trailing_floor": t_floor,
               "control_arm_floor": c_floor}
        if not known:
            rec["verdict"] = "insufficient history on either basis"
            out[metric] = rec
            continue
        binding, label, basis = max(known, key=lambda x: x[0])
        rec.update(binding_floor=binding, binding_basis=basis,
                   binding_label=label)
        # a floor no threshold can clear is a BLOCKED guardrail, not a large
        # number -- said here too, the block an owner reads to pick a value
        if guardrail.floor_is_unusable(binding, dev_basis):
            rec["verdict"] = (
                f"BLOCKED -- the binding {label} floor is {binding} on the "
                f"RELATIVE basis, i.e. ordinary daily swing exceeds the "
                f"series' own level. No threshold can clear it without also "
                f"clearing the failure this guardrail exists to catch. This is "
                f"the wrong basis for this metric, not a tuning problem: set "
                f"monitoring.stop_conditions.deterioration_basis.{metric_key} "
                f"to 'absolute_pp' and re-derive.")
            out[metric] = rec
            continue
        if c_floor is None:
            rec["caveat"] = ("control-arm floor not measurable yet; re-derive "
                             "before the A/B starts -- the binding floor can "
                             "change basis once both arms carry data")
        if threshold is None:
            rec["verdict"] = (f"null -- owner should set it at or above the "
                              f"{label} {basis} floor {binding}")
        elif threshold < binding:
            rec["verdict"] = (f"TOO TIGHT -- {threshold} is below the {label} "
                              f"{basis} floor {binding}; it will false-fire "
                              "and silently suspend exploration")
        elif binding > 0 and threshold > 3 * binding:
            # clearing the floor is necessary, not sufficient: a threshold
            # far above it cannot fire either
            rec["verdict"] = (
                f"CLEARS THE FLOOR BUT LIKELY INERT -- {threshold} is "
                f"{round(threshold / binding, 1)}x the {label} {basis} floor "
                f"{binding}, and the {sc['persistence_days']}-day persistence "
                "rule sits on top. A guardrail this loose will not fire; "
                "consider a different metric or an absolute floor instead")
        else:
            rec["verdict"] = (f"OK -- above the {label} {basis} floor {binding}")
        out[metric] = rec
    out["note"] = ("The binding floor is the LARGER of the two bases because "
                   "one config value is graded against the trailing mean "
                   "before the A/B and against the control arm during it.")
    return out


def guardrail_noise(d, cfg):
    """3-sigma daily noise of scrap rate and realised margin rate, as relative
    deviation from a trailing-window mean."""
    window = cfg["monitoring"]["guardrail_noise_window_days"]
    day = _daily_series(d)

    def noise(series, smooth=1, basis=guardrail.RELATIVE):
        # average `smooth` days BEFORE comparing; the trailing baseline is
        # shifted by the same amount so the two windows never overlap
        s = _smooth(series, smooth)
        if len(s) < window + 7:
            return {"days": int(len(s)),
                    "note": f"needs at least {window + 7} days"}
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
            # metric: a series that changes sign has no meaningful ratio to its
            # own mean, and this is what made the margin floor read 65.4497
            "days_at_or_below_zero": int((s <= 0).sum()),
            **_sigma_summary(rel_dev),
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
                   "metric to deterioration_basis 'absolute_pp'."
                   if out["days_at_or_below_zero"] else
                   "Consider deterioration_basis 'absolute_pp', or more "
                   "smoothing if the series is strictly positive."))
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
                  guardrail.basis_for(cfg, "scrap"))
    margin = noise(day.margin_rate, sm["margin"],
                   guardrail.basis_for(cfg, "margin"))

    def verdict(block, key):
        """Grades against the TRAILING floor only (pre-A/B basis). Necessary,
        not sufficient -- the sign-off number lives in
        guardrail_threshold_recommendation."""
        threshold = sc[key]
        floor, basis = _floor_of(block)
        if floor is None:
            return "insufficient history to validate"
        if threshold is None:
            return (f"{key} is null -- owner should set it at or above the "
                    f"trailing-basis {basis} floor {floor}, then check it "
                    "against the control-arm floor too")
        if threshold >= floor:
            return (f"clears the trailing-basis {basis} floor {floor} -- see "
                    "guardrail_threshold_recommendation for the binding one")
        return (f"TOO TIGHT -- {threshold} is below the trailing-basis {basis} "
                f"floor {floor}; it will false-fire and silently suspend "
                "exploration")

    return {
        "basis": ("daily ratio-of-sums series over all episodes, smoothed over "
                  "deterioration_smoothing_days; relative deviation vs "
                  f"trailing {window}-day mean. Applies BEFORE the A/B; once "
                  "both arms carry data the monitor switches to the control-arm "
                  "basis and so must the threshold"),
        "scrap_rate": {**scrap,
                       "config_key": "monitoring.stop_conditions.scrap_deterioration_pct",
                       "verdict": verdict(scrap, "scrap_deterioration_pct")},
        "margin_rate": {**margin,
                        "config_key": "monitoring.stop_conditions.margin_deterioration_pct",
                        "verdict": verdict(margin, "margin_deterioration_pct")},
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
    """`learning.information_increment` from the posterior's own arithmetic,
    not judgment.

    Fisher information adds to PRECISION: 1/s1^2 = 1/s0^2 + I. So the
    information that shrinks the std by exactly `max_std_shrink` is

        I* = (1/s0^2) * [1/(1-max_std_shrink)^2 - 1]

    I* is a CEILING, not a target. `bounded_step` clips the update at the cap
    and the excess is discarded -- outcomes are marked processed either way --
    so an increment above I* waits longer to gather evidence it then throws
    away. Below I* the update is simply a smaller step, which is safe.

    I* moves with s0 (as 1/s0^2), so no single constant is right for the whole
    pilot: it is smallest at launch, when the prior is widest and cheapest to
    move, and grows as the posterior narrows. Derive it for the LAUNCH stds --
    the phase the pilot exists to get through -- and re-derive after a prior
    change. `wastes_at_launch` is what the CONFIGURED value throws away on a
    launch-width cell, in multiples of what that step could use.
    """
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
                 "mu*L^2*r/(r+mu) that `pipeline.update` accumulates, AFTER "
                 "deff deflation -- the same quantity the trigger compares. "
                 "Re-derive after any change to the prior or to "
                 "max_std_shrink. As the posterior narrows the ceiling rises "
                 "as 1/std^2, so a value set for launch becomes conservative "
                 "later: that direction is safe, the other is not."),
    }


def bounded_step(cfg):
    """`max_mean_step` and `max_std_shrink` -- the two rails on one update.

    They are NOT independent. `max_std_shrink` is the primary rate limiter:
    it fixes how fast a cell may converge, and `information_increment` is
    derived from it (see `information_increment`). `max_mean_step` then rails
    the mean, and the coherent question is whether the two bind at the SAME
    level of surprise.

    A cap-sized update -- one carrying exactly the information that saturates
    `max_std_shrink` -- moves the mean toward the batch's own estimate by

        [1 - (1-max_std_shrink)^2] x |batch estimate - current mean|

    (precision-weighted average, Normal approximation to the grid update).
    So for a batch pulling one prior std away, the mean moves
    `0.4375 x std` at a 25% shrink cap. If `max_mean_step` is far below that,
    the mean rail clips on ordinary batches while the std rail almost never
    binds -- the RUNBOOK's "most updates clip -> mis-sized" condition, by
    construction rather than by accident.

    Neither is fully derivable: they encode risk appetite. What IS derivable
    is their consistency, the surprise level each one trips at, and the
    convergence they imply. The policy consequence of a mean step is measured
    separately, by `backtest.step_sensitivity`, which re-solves the DP arm at
    eps +- max_mean_step on real episodes -- cross-check there before moving it.
    """
    lc = cfg["learning"]
    shrink, step = lc["max_std_shrink"], lc["max_mean_step"]
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
        "verdict": (
            f"CONSISTENT -- both rails trip near a "
            f"{clips_at_std:.2f}-std surprise"
            if clips_at_std is not None and 0.7 <= clips_at_std <= 1.4 else
            f"MEAN RAIL BINDS FIRST -- max_mean_step {step} clips at a "
            f"{clips_at_std:.2f}-std surprise while max_std_shrink needs a "
            f"full cap-sized update, so the mean cap does the work and "
            f"`bound_clipped` fires routinely. OWNER DECISION: raise "
            f"max_mean_step toward {consistent:.2f} (check the price "
            f"consequence in backtest.step_sensitivity FIRST), or lower "
            f"max_std_shrink so the two agree"
            if clips_at_std is not None and clips_at_std < 0.7 else
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
    ap = argparse.ArgumentParser(prog="bootstrap.derive_thresholds")
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mde", type=float, default=None,
                    help="target relative MDE on IL% (falls back to "
                         "ab_test.min_detectable_effect_pct)")
    ap.add_argument("--out", default="reports/thresholds.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mde = args.mde if args.mde is not None \
        else cfg["ab_test"]["min_detectable_effect_pct"]
    d = pd.read_parquet(args.input)

    se_by_T = empirical_se_by_duration(d, cfg)
    trailing = guardrail_noise(d, cfg)
    control = control_arm_noise(d, cfg)
    report = {
        "ab_duration": duration_table(se_by_T, cfg, mde),
        "guardrail_noise": trailing,
        "guardrail_noise_control_arm_basis": control,
        "guardrail_threshold_recommendation": recommend_thresholds(
            trailing, control, cfg),
        "information_increment_recommendation": information_increment(cfg),
        "bounded_step_recommendation": bounded_step(cfg),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    ab = report["ab_duration"]
    print(f"target MDE (relative)  : {mde if mde is not None else 'not set'}")
    for label, row in ab["by_duration"].items():
        print(f"  {label:>4s}: detectable {row['detectable_mde_rel']}"
              f" rel ({row['blocks_measured']} blocks)"
              + ("  <-- meets target" if row["meets_target"] else ""))
    if ab["recommended_duration_weeks"]:
        print(f"recommended duration   : {ab['recommended_duration_weeks']} weeks "
              f"({ab['recommended_duration_days']} days) "
              "-> paste into ab_test.duration_days")
    elif mde is not None:
        print("NO candidate duration meets the target MDE -- "
              "loosen the MDE or extend the window (design 5.3)")
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
        print(f"{key:12s}: trailing {rec.get('trailing_floor')} | "
              f"control-arm {rec.get('control_arm_floor')} | "
              f"binding {rec.get('binding_floor')} "
              f"({rec.get('binding_basis')}) -> {rec['verdict']}")
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
