"""bootstrap.measure -- phase 0 historical measurement suite.

Implements section 8 of the Perishable Markdown MVP PRD (Appendix A). Produces
every value marked MEASURED in config.yaml, plus the three reassessment gates
in section 8.1.

Run against real filtered FLC data. Outputs a JSON report and a human-readable
summary. No causal assumption about price response is made anywhere here.

Usage:
    python3 -m bootstrap.measure --input data/flc_filtered.parquet \
        --out reports/phase0.json

Measurements 1-8 run standalone. Measurement 9 (controller replay) requires the
fitted baseline model and runs through the backtest module once
bootstrap.train_baseline exists. Measurement 10 runs here when a
`predicted_units` column is supplied (backtest attaches it), and is skipped
otherwise.

The two corrections from review of the first run (PRD section 8) are in place:
deff uses mean FORCED hours, and episodes_reaching_zero_inventory checks the
last hour only.
"""

import argparse
import json

import numpy as np
import pandas as pd

from common.config import load_config
from bootstrap.prepare_data import load_and_filter
from common import episodes

PCTS = [10, 25, 50, 75, 90]


# ------------------------------------------------------------------ measurement

def m1_cost_ratio(d, cfg):
    """Cost ratio and feasible ceiling. Decides exploration viability."""
    tier_step = cfg["pricing"]["tier_step"]
    min_tiers = cfg["exploration"]["min_feasible_tiers"]
    ep = d.groupby("episode_id").agg(
        category=("category", "first"),
        cost=("cost", "first"),
        original_price=("original_price", "first"),
        d_max=("d_max", "first"),
    )
    ep["cost_ratio"] = ep.cost / ep.original_price
    ep["n_feasible"] = np.floor(ep.d_max.clip(lower=0) / tier_step).astype(int) + 1

    def block(g):
        return {
            "episodes": int(len(g)),
            "cost_ratio_pct": {f"p{p}": round(float(np.percentile(g.cost_ratio, p)), 4)
                               for p in PCTS},
            "d_max_pct": {f"p{p}": round(float(np.percentile(g.d_max, p)), 4)
                          for p in PCTS},
            "share_d_max_below_0.15": round(float((g.d_max < 0.15).mean()), 4),
            "share_non_explorable": round(
                float((g.n_feasible < min_tiers).mean()), 4),
        }

    return {
        "overall": block(ep),
        "by_category": {k: block(g) for k, g in ep.groupby("category")},
    }


def m2_same_hour_variation(d):
    """Same-hour identifying variation. Decides whether the bracket is estimable."""
    g = (d.groupby(["category", "hour_of_day"])
           .agg(discount_std=("total_discount", "std"),
                n_levels=("total_discount", "nunique"),
                n_episodes=("episode_id", "nunique"),
                n_rows=("total_discount", "size"))
           .reset_index())
    g["discount_std_pp"] = g.discount_std * 100

    weak = g.discount_std_pp < 0.5
    return {
        "cells": int(len(g)),
        "discount_std_pp": {f"p{p}": round(float(np.nanpercentile(g.discount_std_pp, p)), 3)
                            for p in PCTS},
        "n_levels": {f"p{p}": int(np.nanpercentile(g.n_levels, p)) for p in PCTS},
        "share_cells_std_below_0.5pp": round(float(weak.mean()), 4),
        "by_category": {
            k: {"median_std_pp": round(float(np.nanmedian(sub.discount_std_pp)), 3),
                "median_n_levels": int(np.nanmedian(sub.n_levels)),
                "median_episodes_per_cell": int(np.nanmedian(sub.n_episodes))}
            for k, sub in g.groupby("category")},
    }


def m3_intra_episode_correlation(d):
    """rho and mean hours per episode. Sets deff.

    Baseline here is a within-(category, hour) mean, which is a proxy for the
    fitted model. Re-run against fitted mu_ref residuals once the baseline
    model exists (bootstrap.fit_dispersion does) -- that value is authoritative.
    """
    d = d.copy()
    d["base"] = d.groupby(["category", "hour_of_day"])["units_sold"].transform("mean")
    d["resid"] = d.units_sold - d.base

    # rho = mean within-episode pairwise correlation, via variance decomposition
    g = d.groupby("episode_id")["resid"]
    n = g.size()
    keep = n[n >= 3].index
    sub = d[d.episode_id.isin(keep)]
    between = sub.groupby("episode_id")["resid"].mean().var(ddof=1)
    total = sub["resid"].var(ddof=1)
    rho = float(np.clip(between / total, 0.0, 0.95)) if total > 0 else 0.0

    hours = d.groupby("episode_id").size()
    changed = d.groupby("episode_id")["total_discount"].nunique() > 1
    hours_changed = hours[changed[changed].index]

    h_forced = float(hours_changed.mean()) if len(hours_changed) else float(hours.mean())
    return {
        "rho": round(rho, 4),
        "rho_method": "variance_decomposition_on_category_hour_residuals",
        "mean_hours_per_episode": round(float(hours.mean()), 3),
        "mean_forced_hours_per_episode": round(h_forced, 3),
        # deff uses FORCED hours -- these are the correlated observations that
        # actually enter the likelihood. Using all-episode hours understates it.
        "implied_deff": round(1 + (h_forced - 1) * rho, 3),
        "note": "re-run against fitted mu_ref residuals; that value is authoritative",
    }


def m4_demand_density(d):
    def block(g):
        return {
            "rows": int(len(g)),
            "zero_sale_rate": round(float((g.units_sold == 0).mean()), 4),
            "mean_units_sold": round(float(g.units_sold.mean()), 4),
            "units_sold_p50": float(np.percentile(g.units_sold, 50)),
            "units_sold_p90": float(np.percentile(g.units_sold, 90)),
        }
    ep = d.groupby("episode_id").agg(hours=("hour_of_day", "size"),
                                     start_inv=("starting_inventory", "first"))
    return {
        "overall": block(d),
        "by_category": {k: block(g) for k, g in d.groupby("category")},
        "episode_hours_pct": {f"p{p}": float(np.percentile(ep.hours, p)) for p in PCTS},
        "starting_inventory_pct": {f"p{p}": float(np.percentile(ep.start_inv, p))
                                   for p in PCTS},
    }


def m5_censoring(d):
    d = d.copy()
    d["censored"] = d.units_sold >= d.starting_inventory
    # end-of-episode inventory only. Checking .any() across all hours returns a
    # spurious 1.0 because inventory dips to zero transiently before restock.
    last = d.sort_values("hour_of_day").groupby("episode_id").tail(1)
    ep_zero = (last.ending_inventory <= 0)
    return {
        "overall_censored_hour_rate": round(float(d.censored.mean()), 4),
        "episodes_reaching_zero_inventory": round(float(ep_zero.mean()), 4),
        "by_category": {k: round(float(g.censored.mean()), 4)
                        for k, g in d.groupby("category")},
    }


def m6_il_pct(d):
    """IL% under legacy policy. Sets A/B power. PRD sections 3.2, 3.5, 3.6.

    Denominator is original_price x units_sold -- ENDOGENOUS. Zero-sale
    episodes have an undefined per-episode ratio and are handled only by the
    ratio-of-sums aggregation, never by averaging per-episode ratios.
    """
    d = d.copy()
    d["discount_cost"] = (d.original_price - d.offered_price) * d.units_sold

    ep = d.sort_values(["date", "hour_of_day"]).groupby("episode_id").agg(
        category=("category", "first"),
        fc=("fc", "first"),
        sku_id=("sku_id", "first"),
        original_price=("original_price", "first"),
        start_inv=("starting_inventory", "first"),
        end_inv=("ending_inventory", "last"),
        end_hours_remaining=("hours_remaining", "last"),
        cost=("cost", "first"),
        discount_cost=("discount_cost", "sum"),
        units_sold=("units_sold", "sum"),
    )
    # scrap only where the window actually ran out. An episode that sold out
    # early scrapped nothing; a truncated one has no recorded window end, so
    # its leftover units are unknown -- charging them to scrap overstates IL.
    ep["scrap_units"] = episodes.scrap_units(ep.end_hours_remaining, ep.end_inv)
    dropped = int(ep.scrap_units.isna().sum())
    ep = ep[ep.scrap_units.notna()]
    ep["il"] = ep.discount_cost + ep.cost * ep.scrap_units
    ep["denom"] = ep.original_price * ep.units_sold      # ENDOGENOUS denominator

    zero_denom_share = float((ep.denom <= 0).mean())
    excluded = {"episodes_excluded_unknown_scrap": dropped,
                "excluded_share": round(dropped / max(dropped + len(ep), 1), 4)}

    def ratio_of_sums(g):
        den = g.denom.sum()
        return round(float(g.il.sum() / den), 6) if den > 0 else None

    # A/B unit is SKU x FC; ratio estimator, never a mean of unit ratios
    unit = ep.groupby(["sku_id", "fc"]).agg(il=("il", "sum"), denom=("denom", "sum"))
    unit_nonzero = unit[unit.denom > 0].copy()
    unit_nonzero["il_pct"] = unit_nonzero.il / unit_nonzero.denom

    # Linearised (delta-method) variance of the ratio estimator, clustered on unit.
    # Var(R) ~ (1/ (n * Dbar^2)) * Var(il_i - R * denom_i)
    R = float(unit.il.sum() / unit.denom.sum()) if unit.denom.sum() > 0 else np.nan
    n = len(unit)
    dbar = float(unit.denom.mean())
    resid = unit.il - R * unit.denom
    var_ratio = (float(resid.var(ddof=1)) / (n * dbar ** 2)) if n > 1 and dbar > 0 else np.nan

    return {
        "denominator_definition": "original_price * units_sold (endogenous)",
        "il_pct_aggregate": ratio_of_sums(ep),
        "il_pct_denominator_total": float(ep.denom.sum()),
        "il_absolute_total": float(ep.il.sum()),
        "share_episodes_zero_denominator": round(zero_denom_share, 4),
        "scrap_basis": excluded,
        "by_category": {k: {"il_pct": ratio_of_sums(g),
                            "denominator": float(g.denom.sum()),
                            "il_absolute": float(g.il.sum())}
                        for k, g in ep.groupby("category")},
        "by_fc": {k: {"il_pct": ratio_of_sums(g), "denominator": float(g.denom.sum())}
                  for k, g in ep.groupby("fc")},
        "sku_fc_units": int(n),
        "sku_fc_units_zero_denominator": int((unit.denom <= 0).sum()),
        "il_pct_ratio_estimator": round(R, 6) if R == R else None,
        "il_pct_ratio_se_clustered": round(float(np.sqrt(var_ratio)), 6)
            if var_ratio == var_ratio else None,
        "il_pct_dispersion_across_units": round(float(unit_nonzero.il_pct.std(ddof=1)), 6)
            if len(unit_nonzero) > 1 else None,
        "note": "A/B power uses il_pct_ratio_se_clustered, not the per-unit dispersion. "
                "Per-episode ratios are undefined for zero-sale episodes and are never used.",
    }


def m7_learning_rate(d):
    d = d.copy()
    d["week"] = pd.to_datetime(d.date).dt.to_period("W").astype(str)
    ep = d.groupby(["category", "week"])["episode_id"].nunique().reset_index()
    return {
        "categories": int(d.category.nunique()),
        "episodes_per_category_per_week": {
            k: {"median": int(np.median(g.episode_id)),
                "min": int(g.episode_id.min()),
                "max": int(g.episode_id.max())}
            for k, g in ep.groupby("category")},
        "weeks_observed": int(d.week.nunique()),
    }


def m8_entry_hour(d):
    start = d.groupby("episode_id").agg(category=("category", "first"),
                                        start_hour=("hour_of_day", "min"))
    return {
        "overall": {f"p{p}": float(np.percentile(start.start_hour, p)) for p in PCTS},
        "std_by_category": {k: round(float(g.start_hour.std(ddof=1)), 3)
                            for k, g in start.groupby("category")},
        "distinct_start_hours": int(start.start_hour.nunique()),
    }


def m11_episode_endings(d):
    """How episodes end, and how much scrap that leaves genuinely unknown.

    Rows stop either when the window runs out or when inventory hits zero,
    whichever comes first -- so a last row with hours_remaining > 0 is usually
    a SELL-OUT, not missing data. The two must not be pooled: one scrapped
    nothing by construction, the other has an unrecorded window end whose
    leftover units are unknown. Only completed windows contribute scrap.
    """
    if not d.episode_id.nunique():
        return "NOT RUN -- no episodes"
    out = episodes.ending_summary(d)
    last = episodes.last_rows(d)
    trunc = last[episodes.classify(last.hours_remaining,
                                   last.ending_inventory) == episodes.TRUNCATED]
    out["median_hours_unrecorded_when_truncated"] = float(
        trunc.hours_remaining.median()) if len(trunc) else 0.0
    return out


def m10_fidelity_decomposition(d, cfg, pred_col="predicted_units"):
    """Measurement 10 -- separates LEVEL bias from SLOPE bias in the baseline.

    Requires a fitted baseline: `d` must carry a column of predicted units at
    the ACTUAL historical price (i.e. mu_ref scaled by the prior elasticity).
    Run this after bootstrap.train_baseline; it is skipped otherwise.

    Level bias  -> ratio at the reference anchor, where elasticity scaling ~1
    Slope bias  -> how the ratio changes as discount moves away from d_ref
    """
    if pred_col not in d.columns:
        return "NOT RUN -- requires fitted baseline predictions"

    tier_step = cfg["pricing"]["tier_step"]
    d = d.copy()
    d["gap"] = d.total_discount - d.d_ref

    def ratio(g):
        pred = g[pred_col].sum()
        return round(float(g.units_sold.sum() / pred), 4) if pred > 0 else None

    at_anchor = d[d.gap.abs() <= tier_step / 2]

    # ratio by distance from the anchor, in tier-width bins
    bins = np.arange(-0.20, 0.225, 0.05)
    d["gap_bin"] = pd.cut(d.gap, bins)
    by_gap = {str(k): ratio(g) for k, g in d.groupby("gap_bin", observed=True)}

    return {
        "overall_sold_ratio": ratio(d),
        "level_bias_at_anchor": ratio(at_anchor),
        "rows_at_anchor": int(len(at_anchor)),
        "slope_ratio_by_discount_gap": by_gap,
        "by_category": {k: ratio(g) for k, g in d.groupby("category")},
        "interpretation": (
            "level_bias_at_anchor well below 1 with a flat slope -> mu_ref level "
            "error, multiplicative recalibration permitted. Ratio near 1 at the "
            "anchor degrading with gap -> epsilon understated; do NOT recalibrate "
            "the level, widen the search bound and re-estimate."),
    }


# ---------------------------------------------------------------------- reassess

def gates(res):
    m1 = res["m1_cost_ratio"]["overall"]
    m2 = res["m2_same_hour_variation"]
    m6 = res["m6_il_pct"]
    return {
        "gate_1_exploration_viable": {
            "share_non_explorable": m1["share_non_explorable"],
            "verdict": "REVIEW — narrow MVP scope to explorable subset"
                       if m1["share_non_explorable"] > 0.30 else "PASS",
        },
        "gate_2_bracket_estimable": {
            "share_cells_std_below_0.5pp": m2["share_cells_std_below_0.5pp"],
            "verdict": "REVIEW — set posterior.prior.source=fallback"
                       if m2["share_cells_std_below_0.5pp"] > 0.50 else "PASS",
        },
        "gate_3_ab_powerable": {
            "il_pct_ratio_se_clustered": m6["il_pct_ratio_se_clustered"],
            "sku_fc_units": m6["sku_fc_units"],
            "share_episodes_zero_denominator": m6["share_episodes_zero_denominator"],
            "verdict": "supply min_detectable_effect_pct to derive duration",
        },
    }


def config_values(res):
    """The MEASURED values to paste into config.yaml."""
    return {
        "dispersion.rho": res["m3_intra_episode_correlation"]["rho"],
        "dispersion.mean_forced_hours_per_episode":
            res["m3_intra_episode_correlation"]["mean_forced_hours_per_episode"],
        "dispersion.implied_deff": res["m3_intra_episode_correlation"]["implied_deff"],
        "ab_test.il_pct_ratio_se_clustered": res["m6_il_pct"]["il_pct_ratio_se_clustered"],
        "exploration.tau_initial": None,  # measurement 9, needs fitted baseline
        "posterior.prior.per_category": None,  # section 9.5, needs fitted baseline
    }


def run_all(d, cfg):
    res = {
        "m1_cost_ratio": m1_cost_ratio(d, cfg),
        "m2_same_hour_variation": m2_same_hour_variation(d),
        "m3_intra_episode_correlation": m3_intra_episode_correlation(d),
        "m4_demand_density": m4_demand_density(d),
        "m5_censoring": m5_censoring(d),
        "m6_il_pct": m6_il_pct(d),
        "m7_learning_rate": m7_learning_rate(d),
        "m8_entry_hour": m8_entry_hour(d),
        "m11_episode_endings": m11_episode_endings(d),
        "m9_controller_replay": "NOT RUN -- requires fitted baseline model (backtest)",
        "m10_fidelity_decomposition": m10_fidelity_decomposition(d, cfg),
    }
    res["reassessment_gates"] = gates(res)
    res["config_values_measured"] = config_values(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="reports/phase0.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, waterfall = load_and_filter(args.input, cfg)

    res = {"data_quality_waterfall": [
        {"step": s, "rows": r, "episodes": e} for s, r, e in waterfall]}
    res.update(run_all(d, cfg))

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)

    # ---- readable summary ----
    m1 = res["m1_cost_ratio"]["overall"]
    m2 = res["m2_same_hour_variation"]
    m3 = res["m3_intra_episode_correlation"]
    m6 = res["m6_il_pct"]
    print(f"rows after filter        : {len(d):,}")
    print(f"episodes                 : {d.episode_id.nunique():,}")
    print(f"categories               : {d.category.nunique()}")
    print()
    print(f"[m1] cost ratio p50      : {m1['cost_ratio_pct']['p50']}")
    print(f"[m1] d_max p50           : {m1['d_max_pct']['p50']}")
    print(f"[m1] share non-explorable: {m1['share_non_explorable']:.1%}")
    print(f"[m2] same-hour std p50   : {m2['discount_std_pp']['p50']} pp")
    print(f"[m2] cells std < 0.5pp   : {m2['share_cells_std_below_0.5pp']:.1%}")
    print(f"[m3] rho                 : {m3['rho']}")
    print(f"[m3] mean hours/episode  : {m3['mean_hours_per_episode']}")
    print(f"[m3] implied deff        : {m3['implied_deff']}")
    print(f"[m6] IL% aggregate       : {m6['il_pct_aggregate']:.4%}")
    print(f"[m6] IL% clustered SE    : {m6['il_pct_ratio_se_clustered']}")
    print(f"[m6] zero-denom episodes : {m6['share_episodes_zero_denominator']:.1%}")
    print(f"[m6] SKUxFC units        : {m6['sku_fc_units']:,}")
    print()
    for k, v in res["reassessment_gates"].items():
        print(f"{k}: {v['verdict']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
