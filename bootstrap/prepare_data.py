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
from common.episodes import adjustment_reason
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
    """Episode ids where the next hour opens with more stock than this hour
    left behind.

    Tested on the inventory CHAIN, never on `ending_inventory`: the source
    zeroes that field at the window close, so an equality test against it
    would flag every episode's last hour. Callers must run this only on
    contiguous episodes -- across a data gap the jump reads as a restock.
    """
    leftover = (d.starting_inventory - d.units_sold).clip(lower=0)
    nxt = d.groupby("episode_id")["starting_inventory"].shift(-1)
    restocked = nxt.notna() & (nxt > leftover)
    return d.loc[restocked, "episode_id"].unique()


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


def _drop_edge_truncated(d):
    """Drop ONLY the episodes the extract cut off mid-window. Keep the rest.

    An episode with no closure sentinel on its final row -- `ending_inventory`
    still positive while stock remained -- has an unknown outcome. There are
    two very different reasons for that, and dropping both would delete the
    evidence for telling them apart:

      EDGE      the nominal window still had hours to run when the extract was
                cut. Unavoidable, not a defect, and nothing to learn from
                keeping it: a longer extract is the only thing that closes it.
                DROPPED here.
      NOT EDGE  the window ended INSIDE the data and no sentinel appeared
                anyway -- a gap in the hourly feed splitting one window into
                fragments, or a subset whose feed never writes off. That is a
                data-quality problem, it is not fixed by a longer extract, and
                it must stay VISIBLE. KEPT, still classified `not_closed`, so
                m11's not_closed share and scrap_units_unknown_not_closed
                measure exactly this residue and nothing else.

    So after this stage `not_closed` means "unclosed for a reason the extract
    boundary does not explain", which is the number worth watching month to
    month. Dropping all of them instead would have driven it to zero by
    construction and hidden a systemic feed problem behind a clean population.

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

    dropped = unknown[edge.loc[unknown].to_numpy()]
    kept = unknown[~edge.loc[unknown].to_numpy()]
    n_unknown = len(unknown)
    detail = {
        "episodes_dropped": int(len(dropped)),
        # the LARGEST episodes, so this runs well above their episode share --
        # it is the training signal the demand fit gives up
        "rows_dropped": int(d.episode_id.isin(dropped).sum()),
        "leftover_units_dropped": int(left.loc[dropped].sum()) if len(dropped) else 0,
        # what remains unknown for a reason the extract boundary does NOT
        # explain. This is the number to watch: it is a feed problem, and a
        # longer extract will not move it.
        "unclosed_kept_not_edge": int(len(kept)),
        "leftover_units_kept_unknown": int(left.loc[kept].sum()) if len(kept) else 0,
        "share_of_unclosed_explained_by_edge":
            round(float(len(dropped) / n_unknown), 4) if n_unknown else 0.0,
        "extract_last_hour": str(extract_end),
        "note": ("Only extract-edge truncation is dropped. Episodes unclosed "
                 "for any OTHER reason are kept and stay `not_closed`, so "
                 "m11 measures the residue a longer extract cannot fix. Read "
                 "share_of_unclosed_explained_by_edge: near 1.0 and the "
                 "unknown-scrap problem is purely the extract boundary; well "
                 "below and there is a feed gap or a subset that never writes "
                 "off, spread across the whole period. "
                 "m11.not_closed_by_month shows which."),
    }
    return d[~d.episode_id.isin(dropped)], detail


def load_and_filter(path, cfg=None):
    """Section 9.1 mapping + section 9.2 filter chain. Returns (df, waterfall).

    Filter order is deterministic and auditable; the waterfall records row and
    episode counts after every step. Episodes with an intraday restock are
    dropped whole: mid-window replenishment breaks the one-inventory-pool
    assumption the DP's state transition rests on, and the demand the extra
    units meet is not the demand the episode's price path was chosen for.
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

    def step(d, label):
        wf.append((label, len(d), d.episode_id.nunique(), cogs_at_risk(d)))
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

    # Cost is tested `<= 0`, not `< 0`. A zero cost is a MISSING cost -- nobody
    # gives perishable stock away -- and it is the one impossible value that
    # looks plausible, so nothing downstream catches it: `non_priceable`
    # tests `cost >= original_price`, and zero reads as MAXIMALLY priceable
    # (d_max = 1.0). It did damage in two directions. It put a 100% discount
    # in the action set, where mu(d) = mu_ref * ((1-d)/(1-d_ref))^eps is
    # 0 ** negative; `pricing.dp.feasible_tiers` now refuses that tier
    # independently, since that layer owns which prices are legal and must
    # protect a production caller this filter never sees. And scrap is
    # `cost x leftover`, so these episodes contributed discount cost and NO
    # SCRAP -- quietly deflating every IL figure measured over them, which was
    # the expensive half.
    neg = (d.starting_inventory < 0) | (d.units_sold < 0) | (d.cost <= 0)
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

    bad = d.groupby("episode_id")["hours_remaining"].min().lt(0)
    d = d[~d.episode_id.isin(bad[bad].index)]
    d = step(d, "negative_window_dropped")

    # flc_window carries occasional very large values from upstream data
    # issues. An episode claiming an implausible window is not trusted: its
    # counter drives episode identification, the DP horizon, and the window
    # extension, so a bad value propagates everywhere rather than staying
    # local. Dropped whole rather than clamped -- clamping would invent a
    # window end the data never recorded.
    cap = cfg["data"]["max_window_hours"]
    bad = d.loc[d.hours_remaining > cap, "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "window_too_long_dropped")

    # Test the OFFERED price, not applied_price. The source sets
    # applied_price to 0 on zero-sale rows -- ~78% of rows -- so a filter
    # reading it is blind on exactly those, and deep-discount zero-sale hours
    # priced under cost survive. They then reach the planner as an anchor no
    # feasible tier can match, and every remaining hour of the episode is
    # rejected at decision time instead of being excluded here.
    offered = d.original_price * (1 - d.total_discount)
    below = offered < d.cost - 1e-9
    bad = d.loc[below, "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "below_cost_dropped")

    # cost at or above the full price leaves d_max <= 0: no discount tier is
    # feasible, so the episode cannot be priced by this system at all. It was
    # being caught only incidentally, whenever such a row also happened to
    # carry a discount; a zero-discount row at cost == price slipped through
    # and polluted the cost-ratio and IL measurements.
    bad = d.loc[d.cost >= d.original_price, "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "non_priceable_dropped")

    bad = d.loc[d.units_sold > d.starting_inventory, "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "units_gt_inventory_dropped")

    # Every hour must either reconcile -- ending == starting - sold -- or name
    # why it does not. `common.episodes.adjustment_reason` is the same
    # function `pipeline.shadow` uses to build outcomes and the same rule
    # `events.store` enforces in production, so the analysis population cannot
    # contain a break that production would quarantine.
    #
    # Recognised BY THE ZERO, never by position. The source writes stock off
    # at ITS OWN window boundary, and once a window is merged across midnight
    # that row sits in the MIDDLE of ours -- a "only the last hour may break
    # the chain" test drops those episodes in bulk. A partial shortfall
    # (0 < ending < leftover) matches no convention: that is unexplained
    # inventory loss, and it is what this stage exists to remove.
    reconciles = (d.starting_inventory - d.units_sold) == d.ending_inventory
    documented = [adjustment_reason(s_, u, e) is not None for s_, u, e
                  in zip(d.starting_inventory, d.units_sold, d.ending_inventory)]
    broken = ~(reconciles | np.array(documented))
    d = d[~d.episode_id.isin(d.loc[broken, "episode_id"].unique())]
    d = step(d, "chain_break_dropped")

    # re-segment: the filters above drop rows, which can punch a hole in a
    # window that was contiguous in the raw extract
    d = d.sort_values(["sku_id", "fc", "date", "hour_of_day"]).copy()
    d["episode_id"] = assign_episode_ids(d)
    wf.append(("contiguous_episodes_built", len(d), d.episode_id.nunique(),
               cogs_at_risk(d)))

    # Intraday restock: the next hour opens with more stock than this hour
    # left behind. Detected on the inventory CHAIN, never on
    # ending_inventory, which is written off to zero at the window close.
    # Run only after re-segmentation -- across a data gap the chain test
    # would read the jump as a restock.
    d = d[~d.episode_id.isin(restocked_episodes(d))]
    d = step(d, "restocked_episodes_dropped")

    # Episodes the extract cut off mid-window. Their outcome is unknown and
    # unknowable from this data -- nothing to learn from keeping them, and a
    # longer extract is the only thing that closes them.
    #
    # Deliberately NOT every episode with an unknown outcome. The others are
    # unclosed for reasons the extract boundary does not explain -- a gap in
    # the hourly feed, or a subset whose feed never writes off -- and those
    # are a data-quality problem that must stay visible. Dropping them too
    # would drive m11's not_closed share to zero by construction and hide a
    # systemic issue behind a clean population.
    #
    # THE COST, since it is real: the dropped episodes' observed hours are
    # good training data, and they are the LARGEST episodes. `rows_dropped`
    # is how much mu_ref and r give up. Set
    # `data.drop_edge_truncated_episodes: false` to keep them and go back to
    # excluding all unclosed episodes from IL only.
    if cfg["data"].get("drop_edge_truncated_episodes", True):
        d, unclosed = _drop_edge_truncated(d)
        d = step(d, "edge_truncated_episodes_dropped")
        wf[-1] = wf[-1] + (unclosed,)

    d["d_ref"] = d.category.map(lambda c: reference_discount(cfg, c))
    d["d_max"] = 1.0 - d.cost / d.original_price
    d["offered_price"] = d.original_price * (1 - d.total_discount)
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
