"""
Phase 0 — historical measurement suite.

Implements section 8 of the Perishable Markdown MVP PRD. Produces every value
marked MEASURED in config.yaml, plus the three reassessment gates in section 8.1.

Run against real filtered FLC data. Outputs a JSON report and a human-readable
summary. No causal assumption about price response is made anywhere here.

Usage:
    python3 phase0_measure.py --input data/flc_filtered.parquet \
        --out reports/phase0.json

Measurements 1-8 run standalone. Measurement 9 (controller replay) requires the
fitted baseline model and is run separately once bootstrap.train_baseline exists;
it is stubbed here with the inputs it will need.
"""

import argparse
import json
import numpy as np
import pandas as pd

# Mirrors config.yaml -- keep in sync, do not diverge.
EXCLUSION_START = "2026-04-25"
EXCLUSION_END = "2026-06-03"
TIER_STEP = 0.025
MIN_FEASIBLE_TIERS = 2
REFERENCE_DISCOUNT = {"MEAT": 0.25, "SIDE DISH": 0.25, "_default": 0.30}

PCTS = [10, 25, 50, 75, 90]


# ---------------------------------------------------------------- load / filter

def load_and_filter(path):
    """PRD section 9.1 mapping + 9.2 filter chain. Returns (df, waterfall)."""
    df = pd.read_parquet(path)
    df = df.rename(columns={
        "hour": "hour_of_day",
        "skuseq": "sku_id",
        "inventory": "starting_inventory",
        "discount": "total_discount",
        "normal_asp": "original_price",
        "final_price": "applied_price",
        "cogs_wo_vat": "cost",
        "flc_window": "hours_remaining",
    })

    # discount is PERCENT in source -> fraction, exactly once
    df["total_discount"] = df["total_discount"] / 100.0

    df["episode_id"] = (df.sku_id.astype(str) + "|" + df.fc.astype(str)
                        + "|" + df.date.astype(str))

    wf = [("raw", len(df), df.episode_id.nunique())]

    def step(d, label):
        wf.append((label, len(d), d.episode_id.nunique()))
        return d

    d = df[df.date.astype(str).lt(EXCLUSION_START)
           | df.date.astype(str).gt(EXCLUSION_END)]
    d = step(d, "exclusion_window_removed")

    d = d[d.category.notna() & d.subcategory.notna()]
    d = step(d, "null_category_dropped")

    d["original_price"] = (d.groupby("episode_id")["original_price"]
                           .transform(lambda s: s.replace(0, np.nan).ffill().bfill()))
    d = d[d.original_price.notna() & (d.original_price > 0)]
    d = step(d, "zero_base_price_dropped")

    bad = d.groupby("episode_id")["hours_remaining"].min().lt(0)
    d = d[~d.episode_id.isin(bad[bad].index)]
    d = step(d, "negative_window_dropped")

    below = (d.applied_price > 0) & (d.applied_price < d.cost)
    bad = d.loc[below, "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "below_cost_dropped")

    bad = d.loc[d.units_sold > d.starting_inventory, "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "units_gt_inventory_dropped")

    d = d.copy()
    d["d_ref"] = d.category.map(REFERENCE_DISCOUNT).fillna(REFERENCE_DISCOUNT["_default"])
    d["d_max"] = 1.0 - d.cost / d.original_price
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    return d.reset_index(drop=True), wf


# ------------------------------------------------------------------ measurement

def m1_cost_ratio(d):
    """Cost ratio and feasible ceiling. Decides exploration viability."""
    ep = d.groupby("episode_id").agg(
        category=("category", "first"),
        cost=("cost", "first"),
        original_price=("original_price", "first"),
        d_max=("d_max", "first"),
    )
    ep["cost_ratio"] = ep.cost / ep.original_price
    ep["n_feasible"] = np.floor(ep.d_max.clip(lower=0) / TIER_STEP).astype(int) + 1

    def block(g):
        return {
            "episodes": int(len(g)),
            "cost_ratio_pct": {f"p{p}": round(float(np.percentile(g.cost_ratio, p)), 4)
                               for p in PCTS},
            "d_max_pct": {f"p{p}": round(float(np.percentile(g.d_max, p)), 4)
                          for p in PCTS},
            "share_d_max_below_0.15": round(float((g.d_max < 0.15).mean()), 4),
            "share_non_explorable": round(
                float((g.n_feasible < MIN_FEASIBLE_TIERS).mean()), 4),
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
    model exists -- that value is authoritative.
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

    return {
        "rho": round(rho, 4),
        "rho_method": "variance_decomposition_on_category_hour_residuals",
        "mean_hours_per_episode": round(float(hours.mean()), 3),
        "mean_hours_per_episode_with_price_change": round(float(hours_changed.mean()), 3)
            if len(hours_changed) else None,
        "implied_deff_at_mean_hours": round(1 + (float(hours.mean()) - 1) * rho, 3),
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
    ep_zero = d.groupby("episode_id").apply(
        lambda g: bool((g.ending_inventory <= 0).any()), include_groups=False)
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

    ep = d.sort_values("hour_of_day").groupby("episode_id").agg(
        category=("category", "first"),
        fc=("fc", "first"),
        sku_id=("sku_id", "first"),
        original_price=("original_price", "first"),
        start_inv=("starting_inventory", "first"),
        end_inv=("ending_inventory", "last"),
        cost=("cost", "first"),
        discount_cost=("discount_cost", "sum"),
        units_sold=("units_sold", "sum"),
    )
    ep["il"] = ep.discount_cost + ep.cost * ep.end_inv.clip(lower=0)
    ep["denom"] = ep.original_price * ep.units_sold      # ENDOGENOUS denominator

    zero_denom_share = float((ep.denom <= 0).mean())

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
            res["m3_intra_episode_correlation"]["mean_hours_per_episode_with_price_change"],
        "ab_test.il_pct_ratio_se_clustered": res["m6_il_pct"]["il_pct_ratio_se_clustered"],
        "exploration.tau_initial": None,  # measurement 9, needs fitted baseline
        "posterior.prior.per_category": None,  # section 9.5, needs fitted baseline
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="reports/phase0.json")
    args = ap.parse_args()

    d, waterfall = load_and_filter(args.input)

    res = {
        "data_quality_waterfall": [
            {"step": s, "rows": r, "episodes": e} for s, r, e in waterfall],
        "m1_cost_ratio": m1_cost_ratio(d),
        "m2_same_hour_variation": m2_same_hour_variation(d),
        "m3_intra_episode_correlation": m3_intra_episode_correlation(d),
        "m4_demand_density": m4_demand_density(d),
        "m5_censoring": m5_censoring(d),
        "m6_il_pct": m6_il_pct(d),
        "m7_learning_rate": m7_learning_rate(d),
        "m8_entry_hour": m8_entry_hour(d),
        "m9_controller_replay": "NOT RUN — requires fitted baseline model",
    }
    res["reassessment_gates"] = gates(res)
    res["config_values_measured"] = config_values(res)

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
    print(f"[m3] implied deff        : {m3['implied_deff_at_mean_hours']}")
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
