"""common.metrics -- the shared measurements the pipeline consumes.

  episode_economics / settled / daily_rates -> every IL, scrap and margin
      figure (guardrail floors, live guardrail, business metrics, shadow's
      budget base)
  fidelity_decomposition  -> backtest.replay (the level gate value)

They live here rather than in any caller because several need them and a
second copy would drift.
"""

import numpy as np
import pandas as pd

from common import episodes

ECON_CARRY = ("category", "fc", "sku_id", "dp_eligible")


def episode_economics(d):
    """THE episode-grain frame every IL, scrap and margin consumer reads:
    the guardrail noise floors, the live guardrail series and the business
    metrics.

    `d` is hourly in the prepared-frame vocabulary: episode_id, date,
    hour_of_day, starting_inventory, units_sold, ending_inventory,
    original_price, offered_price, cost (+ any of ECON_CARRY). Returns one
    row per episode: `date` (opened), `close_day`, `opening`, `units_sold`,
    `scrap` (episodes.scrap_units -- NaN where the episode is not settled),
    `revenue`, `margin`, `discount_cost`, `il`, `denom`.
    """
    d = d.sort_values(["date", "hour_of_day"])
    carry = {c: (c, "first") for c in ECON_CARRY if c in d.columns}
    disc = (d.original_price - d.offered_price) * d.units_sold
    ep = d.assign(_disc=disc, _rev=d.offered_price * d.units_sold,
                  _mar=(d.offered_price - d.cost) * d.units_sold).groupby(
        "episode_id").agg(
        date=("date", "first"), close_day=("date", "last"),
        original_price=("original_price", "first"), cost=("cost", "first"),
        opening=("starting_inventory", "first"), units_sold=("units_sold", "sum"),
        discount_cost=("_disc", "sum"), revenue=("_rev", "sum"),
        margin=("_mar", "sum"), **carry)
    ep["scrap"] = episodes.scrap_units(d)
    ep["il"] = ep.discount_cost + ep.cost * ep.scrap
    ep["denom"] = ep.original_price * ep.units_sold      # ENDOGENOUS denominator
    return ep


def settled(ep):
    """The rows a figure may be built on, and why the rest were not: a
    missing cost makes scrap read zero (deflating IL), and an unsettled
    episode's scrap is unknown -- excluded and COUNTED, never zeroed."""
    cost_missing = int((~(ep.cost > 0)).sum())          # NaN counts as missing
    ep = ep[ep.cost > 0]
    not_closed = int(ep.scrap.isna().sum())
    ep = ep[ep.scrap.notna()]
    return ep, {"episodes_excluded_not_closed": not_closed,
                "episodes_excluded_cost_missing": cost_missing,
                "excluded_share": round(not_closed / max(not_closed + len(ep), 1), 4)}


def daily_rates(ep):
    """Scrap rate and realised-margin rate by CLOSE day, ratio of sums --
    the series the noise floors are measured on and the live guardrail
    triggers on, from one function so the two cannot drift.

    Keyed on `close_day`, never the opening day: an episode's scrap is known
    when it closes, so a close-day bucket is complete once its episodes are
    settled. An opening-day bucket over settled episodes is not -- on the
    newest days only the episodes that closed EARLY (sold out, low scrap)
    have settled, which reads as an improvement exactly where the persistence
    rule evaluates."""
    day = ep.groupby("close_day").agg(opening=("opening", "sum"), scrap=("scrap", "sum"),
                                      revenue=("revenue", "sum"),
                                      margin=("margin", "sum")).sort_index()
    day["scrap_rate"] = day.scrap / day.opening
    day["margin_rate"] = day.margin / day.revenue.replace(0, np.nan)
    return day



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

    # ratio by distance from the anchor: bins two tiers wide, spanning eight
    # tiers either side of it (the shipped tier_step gives -0.20..0.20 by 0.05)
    width, half_span = 2 * tier_step, 8 * tier_step
    bins = np.arange(-half_span, half_span + width / 2, width)
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
