"""bootstrap.prepare_data -- schema mapping, filter chain, episode construction.

Implements PRD sections 9.1 and 9.2. The source-to-PRD column mapping is applied
here once and nowhere else. The three load-bearing properties of section 9.1:

  1. `discount` is PERCENT in source (25.0 = 25%); converted to a fraction
     exactly once, at load.
  2. `final_price` is a realised transaction price and is 0 on zero-sale rows.
     Offered price is always original_price * (1 - d); final_price is never
     used to reconstruct it.
  3. There is no episode_id in the source. It is built from episode_key as
     contiguous selling hours; the construction rule is persisted in the split
     manifest so production and evaluation derive identical boundaries.

Usage:
    python3 -m bootstrap.prepare_data --input data/flc_filtered.parquet \
        --out data/prepared.parquet --manifest artifacts/split_manifest.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config, reference_discount
from common import episodes
from common.provenance import stamp

SOURCE_TO_PRD = {
    "hour": "hour_of_day",
    "skuseq": "sku_id",
    "inventory": "starting_inventory",
    "discount": "total_discount",
    "normal_asp": "original_price",
    "final_price": "applied_price",
    "cogs_wo_vat": "cost",
    "flc_window": "hours_remaining",
}

# Persisted with the split manifest; production must derive identical boundaries.
EPISODE_RULE = (
    "episode_id = sku_id|fc|<first hour of the window>, where a window is a "
    "maximal run of consecutive hourly rows for one sku x fc over which the "
    "source hours_remaining counter decrements by exactly one per elapsed "
    "hour. The window is NOT keyed by calendar date: FLC windows commonly run "
    "past midnight (36-hour windows are common), and a date key would split "
    "one economic episode into two, resetting the monotonicity anchor and "
    "charging the carried-over inventory to scrap at the seam.")


def assign_episode_ids(df):
    """Maximal runs of consecutive hours with a consistent window countdown.

    Two signals must agree for a row to continue the previous episode: the
    timestamp advances exactly one hour, and `hours_remaining` -- the source's
    own view of the window -- ticks down exactly one. Either alone is too
    weak. Time alone would merge two back-to-back windows; the counter alone
    would stitch across a gap in the data, leaving an episode whose row count
    disagrees with its clock (and `validate_state` rejects exactly that).

    Crossing midnight is a one-hour step like any other, which is the point.
    """
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    grp = [df.sku_id, df.fc]
    dt_h = ts.groupby(grp).diff().dt.total_seconds() / 3600.0
    hr_diff = df.hours_remaining.groupby(grp).diff()
    starts = (dt_h.ne(1.0) | hr_diff.ne(-1.0)).fillna(True)
    start_ts = ts.where(starts).groupby(grp).ffill()
    return (df.sku_id.astype(str) + "|" + df.fc.astype(str) + "|"
            + start_ts.dt.strftime("%Y-%m-%dT%H"))


def restocked_episodes(d):
    """Episode ids with at least one hour that took a restock.

    Read WITHIN the hour, from the source's own convention: `ending_inventory`
    is the final quantity after anything that arrived, so `ending > starting -
    sold` IS the restock, stated by the source rather than inferred by us.

    This used to be read ACROSS hours -- next hour's opening against this
    hour's computed leftover -- which was the same quantity at one remove,
    since `ending` becomes the next `starting`. Two things were wrong with it.
    It could not see a restock in an episode's FINAL hour, there being no next
    hour, so those went undetected and their leftover was booked as scrap. And
    it inherited a `clip(lower=0)` that erased the plainest restock of all:
    an hour selling MORE than it opened with has a negative net, and clipping
    it to zero makes any ending look ordinary.

    Read on the NET arrival over the episode, not on the hour. The hour-level
    test fires on bucket skew: a sale recorded an hour after the stock left
    shows a shortfall at t and a restock of the same size at t+1, and nothing
    arrived. Gating on that pushed a perfectly reconciled window out of
    `dp_eligible` while its own `units_restocked` read 0 -- the flag and the
    quantity contradicting each other. `episode_flow` nets first.

    FLAGS, never filters (`dp_ineligible_reason = "restocked"`). What a
    restock breaks is the DP's state transition, which assumes one pool
    draining monotonically. The hours themselves are honest demand
    observations -- each censored against its own opening stock.
    """
    flow = episodes.episode_flow(d)
    return flow.index[flow.arrived > 0].to_numpy()


def cogs_at_risk(d):
    """Money on the shelf: unit cost x opening stock, ONCE per episode.

    A row count says how much data a filter removed. This says how much
    exposure -- which is the quantity the system exists to protect, since IL
    is discount given away plus scrap at cost. The two can diverge sharply: a
    stage can drop 1% of rows and 15% of the money, or the reverse, and only
    the second number tells you whether the surviving population still
    represents the business.

    Counted at the episode's OPENING row, not summed over hours: inventory
    persists from hour to hour, so a per-row sum would multiply the same
    stock by the length of the window. Relies on the frame being in window
    order, which `load_and_filter` guarantees -- it sorts once up front and
    again at re-segmentation, and every drop between them preserves order.

    Before `negative_quantities_dropped` this includes impossible values
    (negative stock, non-positive cost), so the first few stages carry
    figures that are not real exposure. That is deliberate: clipping them
    would hide the size of the bad data, and the drop that removes them is
    exactly where the waterfall should show it.
    """
    if not len(d):
        return 0.0
    opening = ~d.episode_id.duplicated()
    return float((d.cost.to_numpy()[opening.to_numpy()]
                  * d.starting_inventory.to_numpy()[opening.to_numpy()]).sum())


def edge_truncated_episodes(d):
    """Split the unclosed episodes into the two reasons they are unclosed.

    An episode with no closure sentinel on its final row -- `ending_inventory`
    still positive while stock remained -- has an unknown outcome. There are
    two very different reasons for that, and only one of them is a defect:

      EDGE      the nominal window still had hours to run when the extract was
                cut. Unavoidable, not a defect, and a longer extract is the
                only thing that closes it.
      NOT EDGE  the window ended INSIDE the data and no sentinel appeared
                anyway -- a gap in the hourly feed splitting one window into
                fragments, or a subset whose feed never writes off. A
                data-quality problem, not fixed by a longer extract.

    NEITHER is removed. Both stay `not_closed`, so m11's not_closed share and
    `scrap_units_unknown_not_closed` keep measuring the whole unknown, and the
    two reasons are told apart by `share_of_unclosed_explained_by_edge` rather
    than by deleting one of them. Removing the edge cases used to be the
    default and it cost more than it bought: they are the LARGEST episodes in
    the extract (~25 units of opening stock against ~3 for the population),
    so the demand fit was giving up its best-observed windows to protect a
    scrap figure that was already protected -- `common.episodes.scrap_units`
    returns NaN for an unclosed episode, `backtest.replay` zeroes its scrap
    under `outcome_known`, and `pipeline.shadow` charges scrap only on
    COMPLETED. Their observed hours are ordinary, fully-priced demand
    observations; only their ENDING is missing.

    Returns (edge_episode_ids, detail). The caller flags, never filters.

    Edge is tested exactly rather than by a tolerance: the final row's
    timestamp plus its remaining window against the extract's last hour. An
    episode still being observed AT that last hour counts as edge whatever the
    nominal counter says -- the counter reaches zero on ~0.5% of final rows,
    so it cannot be trusted to mark a window's end.
    """
    last = episodes.last_rows(d)
    ids = last.episode_id.to_numpy()
    kind = episodes.classify_last(last)
    unknown = kind.index[kind == episodes.NOT_CLOSED]

    ts = pd.Series(
        (pd.to_datetime(last.date) + pd.to_timedelta(last.hour_of_day, unit="h")
         ).to_numpy(), index=ids)
    extract_end = ts.max()
    hr = pd.Series(last.hours_remaining.to_numpy(), index=ids).clip(lower=0)
    left = pd.Series(
        episodes.leftover_units(last.starting_inventory, last.units_sold).to_numpy(),
        index=ids)
    edge = ((ts + pd.to_timedelta(hr, unit="h") > extract_end)
            | (ts >= extract_end))

    at_edge = unknown[edge.loc[unknown].to_numpy()]
    not_edge = unknown[~edge.loc[unknown].to_numpy()]
    n_unknown = len(unknown)
    detail = {
        "episodes_unclosed": int(n_unknown),
        "episodes_edge_truncated": int(len(at_edge)),
        "leftover_units_edge_truncated":
            int(left.loc[at_edge].sum()) if len(at_edge) else 0,
        # unknown for a reason the extract boundary does NOT explain. This is
        # the number to watch: it is a feed problem, and a longer extract will
        # not move it.
        "episodes_unclosed_not_edge": int(len(not_edge)),
        "leftover_units_unclosed_not_edge":
            int(left.loc[not_edge].sum()) if len(not_edge) else 0,
        "share_of_unclosed_explained_by_edge":
            round(float(len(at_edge) / n_unknown), 4) if n_unknown else 0.0,
        "extract_last_hour": str(extract_end),
        "note": ("FLAGGED, NOT DROPPED -- `edge_truncated` is a column, and "
                 "these episodes stay dp_eligible. Their observed hours are "
                 "good demand data and they are the largest episodes in the "
                 "extract; only their ENDING is unknown, and every scrap and "
                 "IL consumer already excludes an unclosed ending on its own "
                 "(scrap_units returns NaN, replay zeroes scrap under "
                 "outcome_known, shadow charges scrap only on COMPLETED). "
                 "Read share_of_unclosed_explained_by_edge: near 1.0 and the "
                 "unknown-scrap problem is purely the extract boundary; well "
                 "below and there is a feed gap or a subset that never writes "
                 "off, spread across the whole period. "
                 "m11.not_closed_by_month shows which."),
    }
    return at_edge, detail


def load_and_filter(path, cfg=None, probe=None):
    """Section 9.1 mapping + section 9.2 filter chain. Returns (df, waterfall).

    Every stage that DROPS is an integrity or scope rule: the row is
    impossible (negative stock), unusable by anything (null category, zero
    base price), unreconcilable (chain break), ambiguous (duplicate hour), or
    outside the period the study covers (exclusion window). Nothing is dropped
    for being hard to PRICE. Those conditions -- missing cost, an untrusted
    window counter, a mid-window restock, an unclosed ending -- are flagged by
    `tag_dp_eligibility` and stay in the frame, because the demand model can
    see none of them and the frozen artifacts want the data.

    Filter order is deterministic and auditable; the waterfall records row,
    episode and COGS-at-risk counts after every step.

    `probe`, if given, is called as `probe(label, before, after)` at every
    DROP stage, where `before` is the frame as it entered that stage. It is a
    hook for `tools.filter_forensics`, which decomposes a stage's casualties
    into named causes -- a waterfall says how much a filter took, never what
    it took or whether it was right to. Both frames are shallow copies, so a
    probe that writes cannot move the population. It does not fire for
    `contiguous_episodes_built` or `dp_eligible`, which drop nothing.
    """
    cfg = cfg or load_config()
    excl = cfg["data"]["exclusion_window"]

    df = pd.read_parquet(path).rename(columns=SOURCE_TO_PRD)

    # discount is PERCENT in source -> fraction, exactly once
    df["total_discount"] = df["total_discount"] / 100.0
    df["starting_inventory"] = df["starting_inventory"].round().astype("int64")
    df["ending_inventory"] = df["ending_inventory"].round().astype("int64")

    # One sku x fc cannot have two states in the same hour, and there is no
    # principled way to choose between them -- keep neither. Left in, they
    # also break episode identification: two runs starting at the same instant
    # collide into one id, and the "episode" that results has a
    # non-monotonic window counter.
    df = df.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    dup = df.duplicated(subset=["sku_id", "fc", "date", "hour_of_day"],
                        keep=False)
    df = df[~dup]
    df["episode_id"] = assign_episode_ids(df)

    wf = [("raw", len(df) + int(dup.sum()), df.episode_id.nunique(),
           cogs_at_risk(df)),
          ("duplicate_hour_rows_dropped", len(df), df.episode_id.nunique(),
           cogs_at_risk(df))]

    # the frame handed to the PREVIOUS step is the frame this one filtered,
    # so `before` needs no extra bookkeeping at the call sites
    entering = [df]

    def step(d, label):
        wf.append((label, len(d), d.episode_id.nunique(), cogs_at_risk(d)))
        if probe is not None:
            # shallow copies: copy-on-write shares the data blocks, so this
            # costs nothing on a 10M-row frame, but a probe that writes gets
            # its own block and the chain carries on unperturbed. A diagnostic
            # hook that can move the population is worse than no hook.
            probe(label, entering[0].copy(deep=False), d.copy(deep=False))
        entering[0] = d
        return d

    # Episode-scoped, not row-scoped: a window running past midnight can
    # straddle the boundary, and removing only its inside hours would leave a
    # half-episode that re-segmentation turns into a spurious short window.
    ds = df.date.astype(str)
    inside = ds.ge(excl["start"]) & ds.le(excl["end"])
    d = df[~df.episode_id.isin(df.loc[inside, "episode_id"].unique())]
    d = step(d, "exclusion_window_removed")

    # A discount outside [0, 1] means the percent -> fraction conversion has
    # been applied twice, or not at all. Silent and catastrophic: it inverts
    # every price. Cheap to assert, so assert it.
    bad = d.loc[~d.total_discount.between(0, 1), "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "discount_out_of_range_dropped")

    # Negative stock or negative sales are IMPOSSIBLE, so they go. A missing
    # cost (`cost <= 0`) used to go with them and no longer does: it is an
    # ECONOMIC condition, not an integrity one. The demand model cannot see
    # cost -- it is not in FEATURES -- so such an episode is an ordinary
    # demand observation to every frozen artifact. It is fatal only to IL
    # (scrap is `cost x leftover`) and to the action set (`d_max = 1.0` reads
    # as maximally priceable, which put a 100% discount in front of
    # mu(d) = mu_ref * ((1-d)/(1-d_ref))^eps, i.e. 0 ** negative). Both of
    # those consumers now filter on `dp_eligible`, and `pricing.dp` refuses a
    # non-positive price independently, since that layer owns which prices
    # are legal and must protect a production caller no filter here sees.
    # `ending_inventory` is included since it became load-bearing: restock,
    # censoring and chain continuity are all read off it, and a negative count
    # of physical stock would be classified rather than rejected -- with a
    # net more negative still, it reads as a restock.
    neg = ((d.starting_inventory < 0) | (d.units_sold < 0)
           | (d.ending_inventory < 0))
    d = d[~d.episode_id.isin(d.loc[neg, "episode_id"].unique())]
    d = step(d, "negative_quantities_dropped")

    d = d[d.category.notna() & d.subcategory.notna()]
    d = step(d, "null_category_dropped")

    d = d.copy()
    d["original_price"] = (d.groupby("episode_id")["original_price"]
                           .transform(lambda s: s.replace(0, np.nan).ffill().bfill()))
    d = d[d.original_price.notna() & (d.original_price > 0)]
    d = step(d, "zero_base_price_dropped")

    # "Manufacturing" SKUs enter with a window counter that is ALREADY
    # negative -- a large negative constant rather than a countdown from the
    # window length. Dropping them outright is not neutral: they are
    # concentrated in a handful of categories, so it selects on category and
    # biases every per-category figure downstream.
    #
    # They are recoverable because they behave like a standard short window:
    # an episode entering negative resolves -- sells out or is written off --
    # inside `data.manufacturing_window_hours`. That claim is CHECKED here
    # rather than trusted: any episode entering negative with MORE observed
    # hours than the cap is not this pattern, is not recovered, and falls
    # through to the drop below with its count reported.
    #
    # Recovery rewrites `hours_remaining` as a synthetic countdown from the
    # cap (23, 22, ... at 24). It has to be a countdown rather than a clamp
    # because the counter is load-bearing three ways: episode identification
    # differences it, the DP takes its horizon from it, and
    # `extend_to_window` generates the synthetic tail from it. A clamp would
    # leave a flat counter that re-segmentation would then split every hour.
    cap = int(cfg["data"]["manufacturing_window_hours"])
    entry = d.groupby("episode_id")["hours_remaining"].transform("first")
    length = d.groupby("episode_id")["hours_remaining"].transform("size")
    recoverable = (entry < 0) & (length <= cap)
    n_recovered_ep = int(d.loc[recoverable, "episode_id"].nunique())
    if n_recovered_ep:
        d = d.copy()
        position = d.groupby("episode_id").cumcount()
        d.loc[recoverable, "hours_remaining"] = (cap - 1) - position[recoverable]
    recovery = {"episodes_recovered": n_recovered_ep,
                "rows_recovered": int(recoverable.sum()),
                "window_hours_assumed": cap,
                "episodes_entering_negative_but_longer_than_cap":
                    int(d.loc[(entry < 0) & (length > cap), "episode_id"].nunique())}
    d = step(d, "negative_window_recovered")
    wf[-1] = wf[-1] + (recovery,)

    # An episode whose counter is STILL negative after recovery used to be
    # dropped here. It is now flagged `negative_window` and gates dp_eligible
    # instead -- same argument as `window_too_long`, which it sits beside in
    # DP_INELIGIBLE. Safe to carry through re-segmentation: what survives
    # recovery decrements by exactly one per hour (a flat negative constant is
    # a one-row episode, which recovery already took), so the contiguity test
    # reads it the same way it reads any other window.

    # `units_gt_inventory_dropped` USED TO RUN HERE and it was a mistake. An
    # hour selling more than it opened with is not an impossible quantity, it
    # is a RESTOCK -- stock arrived during the hour, and `ending_inventory`
    # carries the final count. The source says so, and
    # `common.episodes.adjustment_reason` has always agreed: with a negative
    # net, any ending at all exceeds it, so the production reconciler names
    # the row `intraday_restock`. The stage ran FIRST and deleted the episode
    # before the reconciler was ever asked, taking 18.1pp of the extract's
    # COGS with it. These episodes are now flagged `restocked` like any other.

    # ONE hard integrity rule survives at the hour level, and it is the only
    # one with no legitimate exception: `ending` already carries any restock
    # forward, so it MUST be the next hour's `starting`. A violation is the
    # feed contradicting itself about a single instant, which no business
    # event explains, and it makes the whole inventory chain unreadable.
    #
    # Nothing tested this before the source convention was understood --
    # restock was INFERRED from the next hour's opening, which quietly assumed
    # the continuity it should have been checking.
    d = d.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    status = episodes.hour_status(d.starting_inventory, d.units_sold,
                                  d.ending_inventory)
    discontinuous = episodes.continuity_breaks(d)

    # An hour-level SHORTFALL is deliberately NOT a drop any more. On the
    # production feed a sale is routinely bucketed one hour off the inventory
    # snapshot: the stock leaves at hour t and the sale is recorded at t+1, so
    # t reads as a shortfall of one unit and t+1 as a restock of one, and they
    # cancel exactly. Dropping on the hour deleted those episodes -- and since
    # a sale is likelier to straddle a bucket boundary the more the SKU sells,
    # it deleted the fastest-selling windows first. That is the 4.5x size
    # selection in the waterfall.
    #
    # What matters is whether the EPISODE reconciles, and that is tested below
    # in `tag_dp_eligibility` against the flow identity
    # `supply = sold + remaining`. Skew nets to zero there; a genuine loss
    # does not.
    detail = {
        "episodes_dropped": int(d.loc[discontinuous, "episode_id"].nunique()),
        "rows_chain_discontinuous": int(discontinuous.sum()),
        "rows_restock": int((status == episodes.RESTOCK).sum()),
        "rows_write_off": int((status == episodes.WRITE_OFF).sum()),
        "rows_hour_shortfall_NOT_dropped": int(
            (status == episodes.SHORTFALL).sum()),
        "note": ("Drops ONE thing: a feed that disagrees with itself about "
                 "one instant (ending != next starting). An hour-level "
                 "shortfall is left alone -- it is usually a sale bucketed an "
                 "hour late, which cancels against the restock it creates in "
                 "the next hour. The episode-level flow identity in "
                 "dp_eligible.unreconciled is what catches genuine loss. "
                 "A restock is not a break either: ending is the final "
                 "quantity after it, so units_sold > starting_inventory is "
                 "the restock signal, not an impossible value."),
    }
    d = d[~d.episode_id.isin(d.loc[discontinuous, "episode_id"].unique())]
    d = step(d, "chain_break_dropped")
    wf[-1] = wf[-1] + (detail,)

    # re-segment: the filters above drop rows, which can punch a hole in a
    # window that was contiguous in the raw extract
    d = d.sort_values(["sku_id", "fc", "date", "hour_of_day"]).copy()
    d["episode_id"] = assign_episode_ids(d)
    wf.append(("contiguous_episodes_built", len(d), d.episode_id.nunique(),
               cogs_at_risk(d)))

    # Intraday restocks and extract-edge truncation used to be two more drops
    # here. Both are now flags set in `tag_dp_eligibility` below -- the
    # restock gates dp_eligible (it breaks the DP's state transition and
    # nothing else), the truncation does not gate anything (only the episode's
    # ENDING is missing, and every scrap consumer already handles that).
    # Both tests must run AFTER re-segmentation, which is why they live at the
    # end of the chain rather than beside the drops.

    d["d_ref"] = d.category.map(lambda c: reference_discount(cfg, c))
    d["d_max"] = 1.0 - d.cost / d.original_price
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    d, economic = tag_dp_eligibility(d, cfg)
    wf.append(("dp_eligible", int(d.dp_eligible.sum()),
               int(d.loc[d.dp_eligible, "episode_id"].nunique()),
               cogs_at_risk(d[d.dp_eligible]), economic))
    d = add_ref_rate_features(d, cfg)
    # Guarantee window order on the way out. Several consumers take .first()
    # / .last() / .iloc[-1] per episode without re-sorting, and the feature
    # merges above can reorder rows -- an episode read out of order picks the
    # wrong opening inventory and the wrong final hour.
    d = d.sort_values(["episode_id", "date", "hour_of_day"])
    return d.reset_index(drop=True), wf


def add_ref_rate_features(d, cfg):
    """Point-in-time, price-standardised demand-rate features.

    Both are built ONLY from anchor hours -- stocked hours priced within
    ref_rate_anchor_band of the category reference discount -- so they measure
    "how fast does this SKU sell at reference conditions" regardless of which
    policy produced the price. Both are lagged strictly before the episode's
    date: an episode never sees its own day. Censored hours are included,
    capped, matching the censoring the training target itself carries.

      sku_ref_sales_rate_30d      trailing [t-W, t-1] anchor-hour rate at
                                  SKU x FC, falling back to SKU pooled across
                                  FCs, else NaN (LightGBM-native missing)
      prior_episode_ref_sales_rate  anchor-hour rate of the most recent
                                  previous episode of the same SKU x FC;
                                  NaN if that episode had no anchor hours

    Within-episode lag features (last-hour sales) are deliberately absent:
    they are mediators of the episode's own price path and would corrupt the
    learned elasticity (see docs/design.md).
    """
    band = cfg["baseline_model"]["ref_rate_anchor_band"]
    window = cfg["baseline_model"]["ref_rate_window_days"]

    d = d.copy()
    anchor = ((d.total_discount - d.d_ref).abs() <= band + 1e-9) \
        & (d.starting_inventory >= 1)
    day = (pd.DataFrame({
        "sku_id": d.sku_id, "fc": d.fc, "date": pd.to_datetime(d.date),
        "a_sold": d.units_sold.where(anchor, 0),
        "a_hours": anchor.astype(int)})
        .groupby(["sku_id", "fc", "date"], as_index=False).sum())

    def trailing_rate(frame, keys):
        """Trailing [t-W, t-1] anchor rate per key group; rolling includes the
        current day, so the day's own totals are subtracted back out."""
        g = frame.sort_values(keys + ["date"]).set_index("date")
        grouped = g.groupby(keys)
        sold = grouped.a_sold.rolling(f"{window}D").sum() - g.a_sold.to_numpy()
        hours = grouped.a_hours.rolling(f"{window}D").sum() - g.a_hours.to_numpy()
        rate = (sold / hours.replace(0, np.nan)).rename("rate")
        return rate.reset_index()

    # SKU x FC grain, with a SKU-pooled fallback for sparse combinations;
    # the fallback is aggregated to SKU-day first so no same-day cross-FC
    # sales can enter its trailing window
    day = day.merge(trailing_rate(day, ["sku_id", "fc"])
                    .rename(columns={"rate": "rate_sku_fc"}),
                    on=["sku_id", "fc", "date"], how="left")
    sku_day = (day.groupby(["sku_id", "date"], as_index=False)
               [["a_sold", "a_hours"]].sum())
    day = day.merge(trailing_rate(sku_day, ["sku_id"])
                    .rename(columns={"rate": "rate_sku"}),
                    on=["sku_id", "date"], how="left")
    day["sku_ref_sales_rate_30d"] = day.rate_sku_fc.fillna(day.rate_sku)

    day["date"] = day.date.astype(str)
    feats = day[["sku_id", "fc", "date", "sku_ref_sales_rate_30d"]]

    # Features are read as of the episode's FIRST date, not each row's own
    # date. A window running past midnight would otherwise let its second-day
    # rows read a trailing window ending the previous day -- which contains
    # that same episode's first-day sales. The episode would be predicting
    # itself.
    d["_date_str"] = (d.groupby("episode_id")["date"].transform("min")
                      .astype(str))
    d = (d.merge(feats.rename(columns={"date": "_date_str"}),
                 on=["sku_id", "fc", "_date_str"], how="left"))

    # prior_episode_ref_sales_rate at true EPISODE grain. Shifting the daily
    # series would hand a multi-day episode its own earlier day as its
    # "previous episode".
    ep = (pd.DataFrame({
        "episode_id": d.episode_id, "sku_id": d.sku_id, "fc": d.fc,
        "start": d._date_str,
        "a_sold": d.units_sold.where(anchor, 0),
        "a_hours": anchor.astype(int)})
        .groupby(["episode_id", "sku_id", "fc", "start"], as_index=False).sum()
        .sort_values(["sku_id", "fc", "start", "episode_id"]))
    ep["rate"] = ep.a_sold / ep.a_hours.replace(0, np.nan)
    ep["prior_episode_ref_sales_rate"] = (
        ep.rate.groupby([ep.sku_id, ep.fc]).shift(1))
    d = (d.merge(ep[["episode_id", "prior_episode_ref_sales_rate"]],
                 on="episode_id", how="left")
         .drop(columns=["_date_str"]))
    return d


def pre_launch(d, cfg):
    """Everything the PRE-LAUNCH artifacts are allowed to see: episodes whose
    window opened on or before `split.test_end`.

    The three fits are already bounded -- the baseline to `train`, dispersion
    to `calib`, the prior to `train` -- but two things were not, and both
    reached past the gate window:

      * `calibration_fit_window: "all"` resolves to the whole frame, so one
        config edit fits the level factors on the hold-out.
      * `policy_replay` and `derive_tau_initial` run on the whole frame, so
        `tau_initial` -- a MEASURED launch value -- was being derived partly
        on hold-out episodes.

    Neither would announce itself. The hold-out is worth exactly one honest
    reading (see `data.holdout`), and a value fitted on it has spent that
    reading without anyone deciding to.

    Episode-scoped, so a window opening before the boundary is kept whole.
    """
    return episodes.window_slice(d, None, cfg["data"]["split"]["test_end"])



# The five conditions that make an episode unpriceable. Each is a MODELLING
# limit, not an integrity defect: the demand model's FEATURES carry neither
# `cost` nor `hours_remaining` nor anything about the inventory chain, so it
# cannot see any of them, and an episode failing one is an ordinary demand
# observation to every frozen artifact. They are fatal to exactly two things
# -- the DP's state space, and any figure with scrap in it.
#
# Reasons are ordered by how fundamental they are, and an episode is labelled
# with the FIRST it trips, so the reason column reads as a cause rather than
# as whichever test happened to run last.
DP_INELIGIBLE = (
    ("cost_missing",
     "cost <= 0 -- a MISSING cost, not a free good. d_max reads 1.0, so the "
     "DP would discount to the tier cap believing scrap is free, and IL "
     "(scrap = cost x leftover) reads zero"),
    ("non_priceable",
     "cost >= original_price, so d_max <= 0 and feasible_tiers is EMPTY -- "
     "there is no price for the DP to choose"),
    ("negative_window",
     "hours_remaining still < 0 after negative_window_recovered -- an episode "
     "entering negative that runs LONGER than data.manufacturing_window_hours, "
     "so it is not the recoverable source pattern and its true window length "
     "is unknown. The DP takes its horizon from the counter and "
     "extend_to_window generates the synthetic tail from it; neither can read "
     "a negative one. Same class as window_too_long"),
    ("window_too_long",
     "hours_remaining above data.max_window_hours. extend_to_window RAISES "
     "above the cap, so this is a crash rather than a refusal. Only backtest "
     "and shadow extend; the artifact fits never read the counter"),
    ("unreconciled",
     "the episode's flow identity fails: supply (opening + net arrivals) does "
     "not equal units sold plus the stock left on the last row. Stock moved "
     "that no sale, restock or write-off accounts for, so the episode has no "
     "trustworthy clearance -- it would read above 1, or show scrap on a "
     "window that sold everything it had -- and therefore no trustworthy "
     "scrap and no trustworthy IL. Kept in the population and FLAGGED so the "
     "business can find out what moved; `units_unreconciled` is how many"),
    ("restocked",
     "an hour opens with more stock than the previous hour left behind. The "
     "DP's state transition is V(p, q - min(k,q), h-1) over ONE inventory "
     "pool draining monotonically, so a mid-window replenishment leaves the "
     "episode's horizon, scrap and IL undefined. The demand observations "
     "themselves are fine -- censoring is per hour against that hour's own "
     "opening stock -- which is why this flags rather than drops. Detected on "
     "the inventory CHAIN, never on ending_inventory, which the source zeroes "
     "at the window close; run after re-segmentation, since across a data gap "
     "the jump would read as a restock"),
)

# Reported, NOT gating. A below-cost hour is a price the LEGACY policy set,
# and the agent is already constrained never to set one -- so this is a
# property of the history, not a defect in it, and neither harness needs the
# episode removed:
#
#   backtest  the DP arm is self-anchored (`anchor = d_t`, its own previous
#             choice), so it never sees the legacy price at all.
#   shadow    the legacy price IS the anchor, so from the hour it crosses
#             below cost the action set is empty and `validate_state` refuses
#             every remaining hour. That refusal is CORRECT behaviour on real
#             data, counted in `rejected_reasons` -- and the hours before the
#             crossing are perfectly good decisions the old chain threw away
#             with the whole episode.
BELOW_COST_HOURS = (
    "some hour's OFFERED price is under cost. Tested on "
    "original_price x (1 - d), never applied_price, which the source zeroes "
    "on the ~78% of rows that sold nothing")


def tag_dp_eligibility(d, cfg):
    """Flag the episodes the DP cannot price. Flag, never drop.

    Dropping them was costing more than it saved. It removed most of the COGS
    in the extract from every frozen artifact, including the elasticity prior
    -- which is starved of price variation and for which below-cost hours are
    the WIDEST spread in the data. And it answered a gate with the filter that
    removes the gate's subject: `reassessment_gates.max_share_non_explorable`
    exists to say "too many episodes have a cost floor too tight to explore
    against", and `share_non_explorable` measured 0.0 because those episodes
    were already gone.

    Two further stages that used to be drops are computed here for the same
    reason: `negative_window` and `restocked` break the DP's horizon and its
    state transition respectively, and break nothing at all in the demand fit.
    A third, `edge_truncated`, is flagged and gates NOTHING -- see
    `edge_truncated_episodes`.

    Episode-scoped throughout: one bad hour makes the whole window
    unpriceable, because the monotonicity anchor carries that price into every
    later hour and the DP plans the window as a unit. Runs after
    re-segmentation, so the ids it groups on are final and the chain tests
    cannot mistake a data gap for a restock.
    """
    cap = cfg["data"]["max_window_hours"]
    # the episode's supply accounting, computed once and attached to the frame
    # -- `units_restocked` is the quantity the DP would have to model and the
    # business will want to see, and `episode_supply` is the only correct
    # denominator for clearance now that a window can gain stock
    flow = episodes.episode_flow(d)
    for col, src in (("units_restocked", "arrived"),
                     ("units_unreconciled", "vanished"),
                     ("episode_supply", "supply"),
                     ("episode_clearance", "clearance")):
        d = d.copy()
        d[col] = d.episode_id.map(flow[src]).astype(
            float if src == "clearance" else "int64")
    d["flow_reconciled"] = d.episode_id.map(flow.reconciled).astype(bool)

    tests = {
        "cost_missing": d.cost <= 0,
        "non_priceable": d.cost >= d.original_price,
        "negative_window": d.hours_remaining < 0,
        "window_too_long": d.hours_remaining > cap,
        "unreconciled": ~d.flow_reconciled,
        "restocked": d.episode_id.isin(restocked_episodes(d)),
    }
    reason = pd.Series(pd.NA, index=d.index, dtype="object")
    detail = {}
    for name, _ in DP_INELIGIBLE:
        hit = d.episode_id.isin(d.loc[tests[name], "episode_id"].unique())
        detail[name] = {
            "episodes": int(d.loc[hit, "episode_id"].nunique()),
            "rows": int(hit.sum()),
            "cogs_at_risk": round(cogs_at_risk(d[hit]), 1),
        }
        reason = reason.mask(hit & reason.isna(), name)
    d = d.copy()
    d["dp_ineligible_reason"] = reason
    d["dp_eligible"] = reason.isna()
    # informational: kept IN dp_eligible, because the DP refusing a below-cost
    # anchor is the constraint working, not a reason to delete the episode
    below = d.episode_id.isin(
        d.loc[d.offered_price < d.cost - 1e-9, "episode_id"].unique())
    d["below_cost_hours"] = below
    detail["below_cost_hours"] = {
        "episodes": int(d.loc[below, "episode_id"].nunique()),
        "rows": int(below.sum()),
        "cogs_at_risk": round(cogs_at_risk(d[below]), 1),
        "still_dp_eligible": int(
            d.loc[below & d.dp_eligible, "episode_id"].nunique()),
        "why": BELOW_COST_HOURS,
    }
    # also reported-only, and for a different reason: an unclosed ending is
    # not a limit on the DP, it is a missing OUTCOME, and every consumer of an
    # outcome already handles it (scrap_units -> NaN, replay's outcome_known,
    # shadow's COMPLETED-only scrap).
    at_edge, edge_detail = edge_truncated_episodes(d)
    edge = d.episode_id.isin(at_edge)
    d["edge_truncated"] = edge
    edge_detail["rows"] = int(edge.sum())
    edge_detail["cogs_at_risk"] = round(cogs_at_risk(d[edge]), 1)
    edge_detail["still_dp_eligible"] = int(
        d.loc[edge & d.dp_eligible, "episode_id"].nunique())
    detail["edge_truncated"] = edge_detail

    # WHAT THE NETTING ABSORBED. An episode can reconcile at the episode level
    # while its hours wobble, and the wobble is almost always a sale bucketed
    # an hour off the inventory snapshot. Almost always is not always: netting
    # could in principle cancel a real arrival against a real loss hours
    # apart. Rather than invent an adjacency window, the gross movement is
    # reported against the net so the case is visible.
    #
    # Read `median_gross_per_episode`: ones and twos are bucket skew. If it
    # climbs, something other than skew is being cancelled out.
    netted = flow[(flow.gross_in + flow.gross_out > 0)
                  & (flow.arrived == 0) & (flow.vanished == 0)]
    detail["hour_wobble_netted_to_zero"] = {
        "episodes": int(len(netted)),
        "share_of_episodes": round(float(len(netted)) / max(len(flow), 1), 6),
        "gross_units_in": int(netted.gross_in.sum()),
        "gross_units_out": int(netted.gross_out.sum()),
        "median_gross_per_episode": float(
            (netted.gross_in + netted.gross_out).median()) if len(netted) else 0.0,
        "max_gross_per_episode": int(
            (netted.gross_in + netted.gross_out).max()) if len(netted) else 0,
        "note": ("Episodes whose hours do not individually reconcile but "
                 "whose EPISODE does. The sale left inventory in one hour and "
                 "was recorded in the next, so a shortfall and a restock of "
                 "the same size cancel. These are fully trusted -- they keep "
                 "their scrap, clearance and IL, and they are NOT flagged "
                 "restocked, because nothing arrived. Small ones and twos are "
                 "skew; a large gross netting to zero is worth a look."),
    }

    # WHERE THE ANOMALIES SIT, so the business can go and find out what moved.
    # A count on its own says "the feed is imperfect" and stops there. Broken
    # out by category and month it says whether this is one incident, one
    # corner of the catalogue, or a standing property of the feed -- which are
    # three different investigations.
    bad = ~d.flow_reconciled
    if bad.any():
        ep = d[bad].groupby("episode_id").agg(
            month=("date", lambda s: str(pd.to_datetime(s.iloc[0]))[:7]),
            units=("units_unreconciled", "first"),
            # `category` is always present on the real chain; guarded so a
            # caller tagging a bare frame is not forced to invent one
            **({"category": ("category", "first")} if "category" in d else {}))

        def _by(col):
            if col not in ep:
                return {}
            g = ep.groupby(col).agg(episodes=("units", "size"),
                                    units=("units", "sum"))
            g = g.sort_values("units", ascending=False).head(10) \
                if col == "category" else g
            return {str(k): {"episodes": int(r.episodes), "units": int(r.units)}
                    for k, r in g.iterrows()}
        detail["unreconciled_anomalies"] = {
            "episodes": int(len(ep)),
            "units_unaccounted": int(ep.units.sum()),
            "cogs_at_risk": round(cogs_at_risk(d[bad]), 1),
            "median_units_per_episode": float(ep.units.median()),
            "by_category": _by("category"),
            "by_month": _by("month"),
            "note": ("supply != sold + remaining. Stock moved that no sale, "
                     "restock or write-off accounts for. These episodes are "
                     "KEPT and flagged, out of dp_eligible and out of every "
                     "scrap/IL/clearance figure (scrap_units returns NaN for "
                     "them). Concentrated in one month reads as an incident; "
                     "spread evenly reads as a standing feed property; "
                     "concentrated in a few categories names the subset. "
                     "This is the list to hand back to the business."),
        }

    detail["episodes_dp_eligible"] = int(
        d.loc[d.dp_eligible, "episode_id"].nunique())
    detail["episodes_dp_ineligible"] = int(
        d.loc[~d.dp_eligible, "episode_id"].nunique())
    detail["note"] = (
        "FLAGGED, NOT DROPPED. Every row above is still in the population. "
        "below_cost_hours and edge_truncated are REPORTED ONLY and stay "
        "dp_eligible: the backtest's DP arm is self-anchored so it never sees "
        "a below-cost legacy price, in shadow the refusal from the crossing "
        "hour onward is the cost floor working (counted in rejected_reasons), "
        "and an unclosed ending is already excluded from every scrap and IL "
        "figure by the closure sentinel rather than by a filter. "
        "The three artifact fits read baseline_model.train_population "
        "('integrity' = all of it, 'dp_eligible' = this subset); the DP, the "
        "calibration gate and the A/B always read dp_eligible. Reasons are "
        "first-match in the order " + ", ".join(n for n, _ in DP_INELIGIBLE)
        + ". Counts overlap, so they do not sum to episodes_dp_ineligible.")
    return d, detail


def population(d, cfg, which=None):
    """The rows a consumer is entitled to. ONE definition, so a consumer
    states which population it wants rather than re-deriving it.

    `which` is 'integrity' (everything that survived the filter chain) or
    'dp_eligible'. Passing None reads baseline_model.train_population, which
    is what the artifact fits do; the DP-side callers pass 'dp_eligible'
    explicitly because for them it is not a choice.
    """
    which = which or cfg["baseline_model"]["train_population"]
    if which == "integrity":
        return d
    if which == "dp_eligible":
        return d[d.dp_eligible] if "dp_eligible" in d else d
    raise ValueError(f"unknown population {which!r}")


def split_frames(d, cfg):
    """Date splits for baseline fitting only (config data.split).

    An episode is assigned WHOLLY to the split its window started in. Slicing
    by row date would put the later hours of a cross-midnight window in a
    different split from the entry decision that set its price path -- the
    train/calib boundary would run through the middle of an episode.
    """
    s = cfg["data"]["split"]
    slice_ = episodes.window_slice
    return {
        "train": slice_(d, s["train_start"], s["train_end"]),
        "calib": slice_(d, s["calib_start"], s["calib_end"]),
        "test": slice_(d, s["test_start"], s["test_end"]),
    }


def waterfall_rows(waterfall):
    """The waterfall as JSON rows. One definition, because two consumers write
    it -- the split manifest and phase 0 -- and a stage that reports a detail
    block would otherwise appear in one and crash the other.

    Stages are (label, rows, episodes, cogs_at_risk). A stage may carry a
    fifth element, a dict merged into its row: the flc_window recovery reports
    what it recovered, which a count cannot show because it changes no counts.

    Each row also gets what the stage COST, in money and as a share of the raw
    exposure. Rows and episodes were never enough on their own: a filter can
    take 1% of the rows and 15% of the money, and only the second figure says
    whether the surviving population still looks like the business.

    `cogs_dropped` goes NEGATIVE at `contiguous_episodes_built`, which is
    correct and not a bug -- re-segmentation splits windows, so one opening
    row becomes two and the same stock is counted twice. It is the only stage
    that adds rather than removes, in episodes and in money alike.
    """
    raw = waterfall[0][3] if waterfall else 0.0
    out, prev = [], None
    for t in waterfall:
        label, rows, eps, cogs = t[:4]
        row = {"step": label, "rows": rows, "episodes": eps,
               "cogs_at_risk": round(cogs, 1),
               "cogs_pct_of_raw": round(cogs / raw, 6) if raw else None}
        if prev is not None:
            row["cogs_dropped"] = round(prev - cogs, 1)
            row["cogs_dropped_pct_of_raw"] = (
                round((prev - cogs) / raw, 6) if raw else None)
        row.update(t[4] if len(t) > 4 else {})
        out.append(row)
        prev = cogs
    return out


def write_manifest(path, cfg, waterfall):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stamp({
            "episode_rule": EPISODE_RULE,
            "split": cfg["data"]["split"],
            "holdout": cfg["data"].get("holdout"),
            "exclusion_window": cfg["data"]["exclusion_window"],
            "config_version": cfg["meta"]["config_version"],
            "data_quality_waterfall": waterfall_rows(waterfall),
        }, cfg, None, "bootstrap.prepare_data"), f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/prepared.parquet")
    ap.add_argument("--manifest", default="artifacts/split_manifest.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, wf = load_and_filter(args.input, cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    d.to_parquet(args.out, index=False)
    write_manifest(args.manifest, cfg, wf)

    # Money beside counts on every line, because they disagree and the
    # disagreement is the point: a stage taking 1% of rows and 15% of the
    # exposure has changed what the surviving population represents.
    fixed = ("step", "rows", "episodes", "cogs_at_risk", "cogs_pct_of_raw",
             "cogs_dropped", "cogs_dropped_pct_of_raw")
    rows = waterfall_rows(wf)
    for row in rows:
        pct = row.get("cogs_dropped_pct_of_raw")
        print(f"{row['step']:34s} rows {row['rows']:>10,}  "
              f"episodes {row['episodes']:>9,}  "
              f"cogs {row['cogs_at_risk']:>16,.0f}"
              + (f"  dropped {pct:>7.2%}" if pct is not None else ""))
        for k, v in row.items():
            if k not in fixed:
                print(f"{'':34s}   {k}: {v}")
    if rows:
        kept = rows[-1]["cogs_pct_of_raw"]
        print(f"\nCOGS at risk surviving: {rows[-1]['cogs_at_risk']:,.0f} of "
              f"{rows[0]['cogs_at_risk']:,.0f} raw"
              + (f" ({kept:.2%})" if kept is not None else ""))
    print(f"wrote {args.out} and {args.manifest}")


if __name__ == "__main__":
    main()
