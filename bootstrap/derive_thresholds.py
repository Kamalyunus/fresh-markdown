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
        sold=("units_sold", "sum"))
    # An episode that ended with stock on hand disposed of it. Only the ones
    # sitting at the extract boundary have an unknown outcome, and the noise
    # floor must not be measured on an invented number -- but it must not be
    # measured on a sliver of the population either, which is what keying
    # scrap to the nominal counter did.
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
    """3-sigma and robust 3-sigma of a relative-deviation series.

    Shared by both bases so the two floors are computed identically and cannot
    drift apart. `outlier_dominated` marks a raw sigma inflated by a handful of
    low-denominator days; where it is set, the robust figure is the one to set
    a threshold against.
    """
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


def _smooth(series, days):
    """Average `days` days before comparing -- the same transform
    pipeline.monitor.deterioration applies. On a low-base series the daily
    relative swing can exceed the failure the stop condition must detect."""
    s = series.dropna()
    return (s.rolling(days, min_periods=days).mean().dropna() if days > 1 else s)


def control_arm_noise(d, cfg):
    """Same-day treatment-vs-control noise -- the basis the monitor actually
    uses once the A/B is live.

    guardrail_noise measures each day against a TRAILING MEAN, which carries
    the full day-to-day swing of the series. `pipeline.monitor` compares the
    two arms on the SAME day whenever both are populated, which cancels the
    common day effect entirely. On a series as volatile as daily scrap the two
    floors are not comparable, and setting an A/B threshold from the trailing
    figure grades against the wrong yardstick.

    Both arms are SMOOTHED over deterioration_smoothing_days BEFORE they are
    differenced, in that order, because that is exactly what
    pipeline.monitor.deterioration does on this basis. Measuring the floor on
    an unsmoothed daily difference and then evaluating the threshold against a
    7-day-smoothed one overstates the floor by up to ~sqrt(7): a scrap
    threshold set that way sits several times above its true operating noise
    and, with the persistence rule on top, cannot fire at all. Smoothing must
    be applied on BOTH sides or the guardrail is inert.

    Arm assignment uses the same stable SKU x FC hash as the monitor, so the
    split measured here is the split that will run.
    """
    from pipeline.monitor import _arm

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

    sm = cfg["monitoring"]["stop_conditions"]["deterioration_smoothing_days"]
    out = {"basis": ("same-day treatment vs control, each arm smoothed over "
                     "deterioration_smoothing_days before differencing, "
                     "arm hash as in monitor"),
           "allocation": alloc}
    for metric, worse_high, key in (("scrap_rate", True, "scrap"),
                                    ("margin_rate", False, "margin")):
        smooth = sm[key]
        # smooth each arm FIRST, then intersect and difference -- the order
        # pipeline.monitor.deterioration uses. Reversing it, or skipping the
        # smoothing, measures a floor the live comparison never sees.
        t = _smooth(arms["treatment"][metric], smooth)
        c = _smooth(arms["control"][metric], smooth)
        common = t.index.intersection(c.index)
        t, c = t.loc[common], c.loc[common]
        rel = ((t / c - 1) if worse_high else (1 - t / c))
        rel = rel.replace([np.inf, -np.inf], np.nan).dropna()
        if len(rel) < 8:
            out[metric] = {"days": int(len(rel)), "smoothing_days": smooth,
                           "note": f"too few paired days after {smooth}-day "
                                   "smoothing"}
            continue
        out[metric] = {
            "days": int(len(rel)),
            "smoothing_days": smooth,
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
    configured threshold clears it.

    A single config value is evaluated against the trailing basis before the
    A/B and the control-arm basis during it, so it must sit above BOTH. Grading
    it against only one is how a threshold ends up either false-firing in the
    pilot or sitting inert through the A/B.
    """
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
        rec = {"config_key": f"monitoring.stop_conditions.{key}",
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
            # clearing the floor is necessary, not sufficient: a threshold far
            # above it cannot fire either, and the persistence rule raises the
            # bar further. Say so rather than printing OK.
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

    def noise(series, smooth=1):
        # average `smooth` days BEFORE comparing. On a low-base series the
        # daily relative swing can exceed the failure it must detect; sigma
        # falls ~1/sqrt(smooth), and the trailing baseline is shifted by the
        # same amount so the two windows never overlap.
        s = _smooth(series, smooth)
        if len(s) < window + 7:
            return {"days": int(len(s)),
                    "note": f"needs at least {window + 7} days"}
        # a FULL trailing window only: with min_periods below `window` the
        # warm-up days divide by a mean built from a handful of observations,
        # which manufactures enormous relative deviations that are an artifact
        # of the estimator rather than a property of the series
        trailing = s.rolling(window, min_periods=window).mean().shift(smooth)
        rel_dev = (s / trailing - 1).dropna()
        # a plain std over a ratio series is dominated by any day whose
        # denominator is small -- a single low-volume day can move it by an
        # order of magnitude. The MAD-based estimate is the one to set a
        # threshold against when the two disagree.
        #
        # units: these are RELATIVE deviations, so 0.1336 means 13.36% and
        # 9.1386 means 914%. A value above 1.0 is not a percentage point --
        # it is a series whose daily swing exceeds its own level.
        out = {
            "days": int(len(s)),
            "days_scored": int(len(rel_dev)),
            "smoothing_days": smooth,
            "mean_level": round(float(s.mean()), 4),
            **_sigma_summary(rel_dev),
            "p95_abs_rel_dev": round(float(np.percentile(np.abs(rel_dev), 95)), 4),
            "worst_observed_rel_dev": round(float(rel_dev.abs().max()), 4),
        }
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
    scrap = noise(day.scrap_rate, sm["scrap"])
    margin = noise(day.margin_rate, sm["margin"])

    def verdict(block, key):
        """Grades against the TRAILING floor only -- the basis that applies
        before an A/B is running. Clearing it is necessary, not sufficient:
        the same config value is graded against the control arm once the A/B
        starts, so the sign-off number is the one in
        guardrail_threshold_recommendation, not this line."""
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
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
