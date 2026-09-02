"""common.metrics -- the two shared measurements the pipeline consumes.

Both were phase-0 measurements (m6, m10) and outlived it: phase 0 answered
"is this problem tractable at all", a launch question asked once, and was
removed with the rest of the descriptive layer. These two are different --
they are read on every run by code that decides something:

  il_pct                  -> bootstrap.derive_thresholds (A/B power)
  fidelity_decomposition  -> backtest.replay (the level gate value)

They live here rather than in either caller because both callers need them
and a second copy would drift.
"""

import numpy as np
import pandas as pd

from common import episodes

def il_pct(d):
    """IL% under legacy policy. Sets A/B power (design 2.2-2.3). Denominator
    is original_price x units_sold -- ENDOGENOUS. Zero-sale episodes are
    handled only by ratio-of-sums aggregation, never by averaging
    per-episode ratios (undefined there)."""
    d = d.copy()
    d["discount_cost"] = (d.original_price - d.offered_price) * d.units_sold

    ep = d.sort_values(["date", "hour_of_day"]).groupby("episode_id").agg(
        category=("category", "first"),
        fc=("fc", "first"),
        sku_id=("sku_id", "first"),
        original_price=("original_price", "first"),
        start_inv=("starting_inventory", "first"),
        end_start_inv=("starting_inventory", "last"),
        end_sold=("units_sold", "last"),
        end_hours_remaining=("hours_remaining", "last"),
        cost=("cost", "first"),
        discount_cost=("discount_cost", "sum"),
        units_sold=("units_sold", "sum"),
        **({"dp_eligible": ("dp_eligible", "first")} if "dp_eligible" in d else {}),
    )
    # a missing cost makes scrap read zero, deflating IL -- excluded, counted
    cost_missing = int((~(ep.cost > 0)).sum())   # NaN counts as missing
    ep = ep[ep.cost > 0]
    # the listing ending IS the disposal, whatever the nominal counter says;
    # only episodes at the extract boundary have a genuinely unknown outcome
    ep["scrap_units"] = episodes.scrap_units(d)
    dropped = int(ep.scrap_units.isna().sum())
    ep = ep[ep.scrap_units.notna()]
    ep["il"] = ep.discount_cost + ep.cost * ep.scrap_units
    ep["denom"] = ep.original_price * ep.units_sold      # ENDOGENOUS denominator

    zero_denom_share = float((ep.denom <= 0).mean())
    excluded = {"episodes_excluded_not_closed": dropped,
                "episodes_excluded_cost_missing": cost_missing,
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

    # TWO baselines: "integrity" is what the business loses; "dp_eligible" is
    # what the MVP can address and the only valid policy/A-B comparison
    dp = ep[ep.dp_eligible] if "dp_eligible" in ep else ep
    by_population = {
        "integrity": {"episodes": int(len(ep)), "il_pct": ratio_of_sums(ep),
                      "il_absolute": round(float(ep.il.sum()), 1)},
        "dp_eligible": {"episodes": int(len(dp)), "il_pct": ratio_of_sums(dp),
                        "il_absolute": round(float(dp.il.sum()), 1)},
    }

    return {
        "denominator_definition": "original_price * units_sold (endogenous)",
        "il_pct_aggregate": ratio_of_sums(ep),
        "il_pct_denominator_total": float(ep.denom.sum()),
        "il_absolute_total": float(ep.il.sum()),
        "by_population": by_population,
        "population_note": (
            "il_pct_aggregate is the INTEGRITY population -- what the business "
            "loses. by_population.dp_eligible is what this MVP can address, and "
            "is the figure the A/B and any policy comparison must use. Quoting "
            "the wrong one overstates or understates the addressable loss."),
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



def fidelity_decomposition(d, cfg, pred_col="predicted_units"):
    """Measurement 10 -- separates LEVEL bias (sold ratio at the reference
    anchor, where elasticity scaling ~1) from SLOPE bias (how the ratio moves
    with distance from d_ref). Requires predicted units at the ACTUAL
    historical price; skipped when the column is absent."""
    if pred_col not in d.columns:
        return "NOT RUN -- requires fitted baseline predictions"

    tier_step = cfg["pricing"]["tier_step"]
    d = d.copy()
    d["gap"] = d.total_discount - d.d_ref

    def ratio(g):
        pred = g[pred_col].sum()
        return round(float(g.units_sold.sum() / pred), 4) if pred > 0 else None

    at_anchor = d[episodes.is_anchor_row(d, tier_step)]

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

