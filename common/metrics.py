"""common.metrics -- the shared measurements the pipeline consumes.

  episode_economics / settled / daily_rates -> every IL, scrap and margin
      figure (guardrail floors, live guardrail, business metrics, shadow's
      budget base)
  summary -> the IL / scrap / sell-through / margin block over a settled
      frame (the monitor's business metrics, the simulator's arm
      economics, shadow's markdown IL)

They live here rather than in any caller because several need them and a
second copy would drift. `fidelity_decomposition` (the level gate value)
is the backtest's own and lives there.
"""

import numpy as np

from common import episodes

ECON_CARRY = ("category", "fc", "sku_id", "dp_eligible")


def episode_economics(d):
    """THE episode-grain frame every IL, scrap and margin consumer reads:
    the guardrail noise floors, the live guardrail series and the business
    metrics.

    `d` is hourly in the prepared-frame vocabulary: episode_id, date,
    hour_of_day, starting_inventory, units_sold, ending_inventory,
    original_price, offered_price, cost (+ any of ECON_CARRY). Returns one
    row per episode: `date` (opened), `close_day`, `opening`, `supply`
    (opening + arrived, episodes.episode_flow -- design 12a), `units_sold`,
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
    # what the episode actually had to sell: a restocked window's scrap is a
    # share of everything that arrived, not of the opening count alone
    ep["supply"] = episodes.episode_flow(d).supply.reindex(ep.index)
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
    day = ep.groupby("close_day").agg(opening=("opening", "sum"), supply=("supply", "sum"),
                                      scrap=("scrap", "sum"),
                                      revenue=("revenue", "sum"),
                                      margin=("margin", "sum")).sort_index()
    # scrap over SUPPLY (opening + arrived, design 12a): over opening alone a
    # restocked day reads a scrap rate above what it could have scrapped
    day["scrap_rate"] = day.scrap / day.supply
    day["margin_rate"] = day.margin / day.revenue.replace(0, np.nan)
    return day


def summary(ep, hours=None, rounding=None):
    """The IL, scrap, sell-through and margin block over a SETTLED episode
    frame (`settled(episode_economics(d))`), every figure a ratio of sums
    with its denominator and the absolute IL alongside (design 2.3):
    `episodes`, `il_absolute`, `il_pct`, `il_pct_denominator`,
    `il_discount`, `il_scrap` (the two terms of `il`), `scrap_units`,
    `scrap_rate` (scrap over SUPPLY), `sell_through` (sold over sold +
    scrap), `margin`; with `hours` (the hourly frame the episodes came
    from) also `hours` and `mean_discount` over the settled episodes' rows.
    Unrounded unless `rounding` maps a key to its decimals -- each reader
    keeps the precision it reports at, so the monitor, the simulator and
    shadow read one block without one moving another's figures."""
    den = float(ep.denom.sum())
    il = float(ep.il.sum())
    units = float(ep.units_sold.sum() + ep.scrap.sum())
    out = {
        "episodes": int(len(ep)),
        "il_absolute": il,
        "il_pct": float(ep.il.sum() / den) if den > 0 else None,
        "il_pct_denominator": den,
        "il_discount": float(ep.discount_cost.sum()),
        "il_scrap": float((ep.cost * ep.scrap).sum()),
        "scrap_units": int(ep.scrap.sum()),
        "scrap_rate": float(ep.scrap.sum() / ep.supply.sum())
        if ep.supply.sum() > 0 else None,
        "sell_through": float(ep.units_sold.sum() / units) if units > 0 else None,
        "margin": float(ep.margin.sum()),
    }
    if hours is not None:
        mine = hours[hours.episode_id.isin(ep.index)]
        out["hours"] = int(len(mine))
        out["mean_discount"] = float(mine.shelf_discount.mean()) if len(mine) else None
    for key, digits in (rounding or {}).items():
        if out.get(key) is not None:
            out[key] = round(out[key], digits)
    return out


# moved to evaluate.backtest (its one reader); the name stays for callers
from evaluate.backtest import fidelity_decomposition                     # noqa: E402,F401
