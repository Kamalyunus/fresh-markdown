"""bootstrap.derive_thresholds -- data-driven values for the owner decisions.

Three SET BY OWNER config keys block strict start-up. This tool does not set
them -- it derives the evidence the owner should set them from:

1. A/B duration vs MDE (ab_test.min_detectable_effect_pct, duration_days).
   The clustered SE of the IL% ratio estimator is measured EMPIRICALLY on
   candidate-duration windows sliced from filtered history -- not scaled by
   sqrt(T), which is optimistic under SKU x FC clustering. For each candidate
   duration T: MDE_abs(T) = (z_{1-alpha/2} + z_power) x 2 x SE_pooled(T)
   (the factor 2 converts the pooled all-units SE to the between-arm
   difference SE at 50/50 allocation).

2. Guardrail noise floors (monitoring.stop_conditions.scrap_deterioration_pct
   and margin_deterioration_pct). The daily scrap-rate and realised-margin
   series are computed from history, expressed as relative deviation from a
   trailing baseline, and summarised as 3-sigma noise levels. A stop-condition
   threshold below the 3-sigma level will false-fire on noise and silently
   suspend exploration; a sound threshold sits at or above it.

Usage:
    python3 -m bootstrap.derive_thresholds --input data/prepared.parquet \
        [--mde 0.075] [--out reports/thresholds.json]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from common.config import load_config
from common import episodes
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
                 "window must change (PRD section 8.1, measurement 6)."),
    }


# --------------------------------------------------------- guardrail noise

def _daily_series(d):
    ep = d.sort_values(["date", "hour_of_day"]).groupby("episode_id").agg(
        date=("date", "first"),
        start_inv=("starting_inventory", "first"),
        end_start_inv=("starting_inventory", "last"),
        end_sold=("units_sold", "last"),
        end_hours_remaining=("hours_remaining", "last"),
        sold=("units_sold", "sum"))
    # scrap only where the window ran out; a truncated episode's leftover is
    # unknown, and the noise floor must not be measured on an invented number
    ep["end_inv"] = episodes.scrap_units(
        ep.end_hours_remaining, ep.end_start_inv, ep.end_sold)
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


def control_arm_noise(d, cfg):
    """Same-day treatment-vs-control noise -- the basis the monitor actually
    uses once the A/B is live.

    guardrail_noise measures each day against a TRAILING MEAN, which carries
    the full day-to-day swing of the series. `pipeline.monitor` compares the
    two arms on the SAME day whenever both are populated, which cancels the
    common day effect entirely. On a series as volatile as daily scrap the two
    floors are not comparable, and setting an A/B threshold from the trailing
    figure grades against the wrong yardstick.

    Arm assignment uses the same stable SKU x FC hash as the monitor, so the
    split measured here is the split that will run.
    """
    from pipeline.monitor import _arm

    ep = d.sort_values(["date", "hour_of_day"]).groupby("episode_id").agg(
        date=("date", "first"), sku_id=("sku_id", "first"), fc=("fc", "first"),
        start_inv=("starting_inventory", "first"),
        end_start_inv=("starting_inventory", "last"),
        end_sold=("units_sold", "last"),
        end_hours_remaining=("hours_remaining", "last"))
    ep["scrap"] = episodes.scrap_units(ep.end_hours_remaining,
                                       ep.end_start_inv, ep.end_sold)
    ep = ep[ep.scrap.notna()]
    rev = ((d.offered_price * d.units_sold).groupby(d.episode_id).sum()
           .rename("revenue"))
    mar = (((d.offered_price - d.cost) * d.units_sold)
           .groupby(d.episode_id).sum().rename("margin"))
    ep = ep.join(rev).join(mar)
    alloc = cfg["ab_test"]["allocation"]
    ep["arm"] = [_arm(s, f, alloc) for s, f in zip(ep.sku_id, ep.fc)]

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

    out = {"basis": "same-day treatment vs control, arm hash as in monitor",
           "allocation": alloc}
    for metric, worse_high in (("scrap_rate", True), ("margin_rate", False)):
        t, c = arms["treatment"][metric], arms["control"][metric]
        common = t.index.intersection(c.index)
        t, c = t.loc[common], c.loc[common]
        rel = ((t / c - 1) if worse_high else (1 - t / c))
        rel = rel.replace([np.inf, -np.inf], np.nan).dropna()
        if len(rel) < 8:
            out[metric] = {"days": int(len(rel)), "note": "too few paired days"}
            continue
        mad = float(np.median(np.abs(rel - np.median(rel))))
        out[metric] = {
            "days": int(len(rel)),
            "three_sigma": round(3 * float(rel.std(ddof=1)), 4),
            "three_sigma_robust": round(3 * 1.4826 * mad, 4),
            "median_gap": round(float(np.median(rel)), 4),
        }
    out["note"] = ("Set the A/B-phase threshold against THIS floor. The "
                   "trailing-mean floor in guardrail_noise applies only "
                   "before an A/B is running, where no control arm exists.")
    return out


def guardrail_noise(d, cfg):
    """3-sigma daily noise of scrap rate and realised margin rate, as relative
    deviation from a trailing-window mean."""
    window = cfg["monitoring"]["guardrail_noise_window_days"]
    day = _daily_series(d)

    def noise(series, smooth=1):
        s = series.dropna()
        # average `smooth` days BEFORE comparing. On a low-base series the
        # daily relative swing can exceed the failure it must detect; sigma
        # falls ~1/sqrt(smooth), and the trailing baseline is shifted by the
        # same amount so the two windows never overlap.
        if smooth > 1:
            s = s.rolling(smooth, min_periods=smooth).mean().dropna()
        if len(s) < window + 7:
            return {"days": int(len(s)),
                    "note": f"needs at least {window + 7} days"}
        # a FULL trailing window only: with min_periods below `window` the
        # warm-up days divide by a mean built from a handful of observations,
        # which manufactures enormous relative deviations that are an artifact
        # of the estimator rather than a property of the series
        trailing = s.rolling(window, min_periods=window).mean().shift(smooth)
        rel_dev = (s / trailing - 1).dropna()
        sigma = float(rel_dev.std(ddof=1))
        # a plain std over a ratio series is dominated by any day whose
        # denominator is small -- a single low-volume day can move it by an
        # order of magnitude. The MAD-based estimate is the one to set a
        # threshold against when the two disagree.
        mad = float(np.median(np.abs(rel_dev - np.median(rel_dev))))
        sigma_robust = 1.4826 * mad
        out = {
            "days": int(len(s)),
            "days_scored": int(len(rel_dev)),
            "smoothing_days": smooth,
            "mean_level": round(float(s.mean()), 4),
            "daily_rel_dev_sigma": round(sigma, 4),
            "three_sigma": round(3 * sigma, 4),
            "daily_rel_dev_sigma_robust": round(sigma_robust, 4),
            "three_sigma_robust": round(3 * sigma_robust, 4),
            "p95_abs_rel_dev": round(float(np.percentile(np.abs(rel_dev), 95)), 4),
            "worst_observed_rel_dev": round(float(rel_dev.abs().max()), 4),
        }
        # units: these are RELATIVE deviations, so 0.1336 means 13.36% and
        # 9.1386 means 914%. A value above 1.0 is not a percentage point --
        # it is a series whose daily swing exceeds its own level.
        out["outlier_dominated"] = bool(sigma_robust > 0
                                        and sigma > 2 * sigma_robust)
        if out["outlier_dominated"]:
            out["note"] = (
                f"raw 3-sigma ({out['three_sigma']}) is "
                f"{round(sigma / sigma_robust, 1)}x the robust estimate "
                f"({out['three_sigma_robust']}) -- a few low-denominator days "
                "dominate it. Set the threshold against three_sigma_robust "
                "and investigate the outlier days before trusting either.")
        return out

    sc = cfg["monitoring"]["stop_conditions"]
    sm = sc["deterioration_smoothing_days"]
    scrap = noise(day.scrap_rate, sm["scrap"])
    margin = noise(day.margin_rate, sm["margin"])

    def verdict(block, key):
        threshold = sc[key]
        if "three_sigma" not in block:
            return "insufficient history to validate"
        # against the robust floor where the raw one is outlier-dominated,
        # otherwise they are the same number
        floor = (block["three_sigma_robust"] if block.get("outlier_dominated")
                 else block["three_sigma"])
        basis = "robust 3-sigma" if block.get("outlier_dominated") else "3-sigma"
        if threshold is None:
            return (f"{key} is null -- owner should set it at or above the "
                    f"{basis} floor {floor}")
        if threshold >= floor:
            return f"OK -- threshold above the {basis} daily noise floor {floor}"
        return (f"TOO TIGHT -- {threshold} is below the {basis} noise floor "
                f"{floor}; it will false-fire and silently suspend exploration")

    return {
        "basis": ("daily ratio-of-sums series over all episodes; relative "
                  f"deviation vs trailing {window}-day mean"),
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
    report = {
        "ab_duration": duration_table(se_by_T, cfg, mde),
        "guardrail_noise": guardrail_noise(d, cfg),
        "guardrail_noise_control_arm_basis": control_arm_noise(d, cfg),
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
              "loosen the MDE or extend the window (PRD 8.1)")
    gn = report["guardrail_noise"]
    for key in ("scrap_rate", "margin_rate"):
        block = gn[key]
        sigma3 = block.get("three_sigma", "n/a")
        line = f"{key:12s}: 3-sigma daily noise {sigma3}"
        if block.get("outlier_dominated"):
            line += (f" (OUTLIER-DOMINATED; robust "
                     f"{block['three_sigma_robust']})")
        print(f"{line} -> {block['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
