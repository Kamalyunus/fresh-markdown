"""bootstrap.prepare_data -- schema mapping, filter chain, episode construction.

Implements docs/design.md section 5.2; column mapping applied here once and
nowhere else. Load-bearing (section 9.1): `discount` is PERCENT in source,
converted to a fraction exactly once at load; `final_price` is a realised
price (0 on zero-sale rows), never used to reconstruct the offered price;
episode_id is built here, the rule persisted in the split manifest."""

import argparse
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config, reference_discount
from common import episodes
from common.provenance import stamp

SOURCE_TO_CANONICAL = {
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
    """Episode ids as maximal runs where the timestamp advances one hour AND
    `hours_remaining` ticks down one -- either signal alone would merge
    back-to-back windows or stitch across feed gaps. Crossing midnight is an
    ordinary one-hour step, which is the point."""
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    grp = [df.sku_id, df.fc]
    dt_h = ts.groupby(grp).diff().dt.total_seconds() / 3600.0
    hr_diff = df.hours_remaining.groupby(grp).diff()
    starts = (dt_h.ne(1.0) | hr_diff.ne(-1.0)).fillna(True)
    start_ts = ts.where(starts).groupby(grp).ffill()
    return (df.sku_id.astype(str) + "|" + df.fc.astype(str) + "|"
            + start_ts.dt.strftime("%Y-%m-%dT%H"))


def gap_split_windows(df):
    """Ids of EVERY fragment of a source window a missing hour split in two --
    neither fragment is a real episode, and the second's first row would enter
    the entry-only elasticity fit with the wrong opening state. Detected from
    the counter falling in step with the clock (a new window resets upward)."""
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    grp = [df.sku_id, df.fc]
    dt_h = ts.groupby(grp).diff().dt.total_seconds() / 3600.0
    hr_drop = -df.hours_remaining.groupby(grp).diff()
    # the clock skipped hours and the counter ran down by exactly as many
    gap = (dt_h > 1) & (dt_h == hr_drop)
    if not gap.any():
        return np.array([], dtype=object), {}

    # walk fragments into windows: a row opens a NEW window only if it starts
    # a new episode for some reason OTHER than the gap
    starts = df.episode_id.ne(df.episode_id.groupby(grp).shift())
    window = df.episode_id.where(starts & ~gap).groupby(grp).ffill()
    per_window = df.groupby(window).episode_id.nunique()
    broken = per_window.index[per_window > 1]
    ids = df.loc[window.isin(broken), "episode_id"].unique()
    detail = {
        "windows_split_by_a_feed_gap": int(len(broken)),
        "fragments_dropped": int(len(ids)),
        "missing_hours": int((dt_h[gap] - 1).sum()),
        "note": ("A hole in the hourly feed splits one source window into "
                 "fragments. Every fragment is dropped, not just the "
                 "unclosed one: the second opens mid-window with the wrong "
                 "starting stock and a counter part-way through, and its "
                 "first row would be read as an ENTRY row by the elasticity "
                 "fit. Detected from the counter falling in step with the "
                 "clock, which a genuinely new window never does."),
    }
    return ids, detail


def cogs_at_risk(d):
    """Exposure: unit cost x SUPPLY (opening stock + GROSS arrivals; shrink
    not subtracted), once at each episode's opening row. Requires window
    order, which `load_and_filter` guarantees. Pre-`episode_universe` stages
    deliberately carry impossible values / an unverified arrival term (5.2)."""
    if not len(d):
        return 0.0
    opening = (~d.episode_id.duplicated()).to_numpy()
    # gross arrivals per episode (ending[t] IS starting[t+1]); reset_index +
    # sort_index realigns hour_adjustment's (date, hour)-sorted values to
    # frame order before grouping, whatever labels the caller's frame carries
    dd = d.reset_index(drop=True)
    arrivals = (episodes.hour_adjustment(dd).sort_index().clip(lower=0)
                .groupby(dd.episode_id).sum())
    opening_ids = pd.Series(d.episode_id.to_numpy()[opening])
    supply = (d.starting_inventory.to_numpy()[opening]
              + opening_ids.map(arrivals).fillna(0.0).to_numpy())
    return float((d.cost.to_numpy()[opening] * supply).sum())


def edge_truncated_episodes(d):
    """Split unclosed episodes into EDGE (window still running when the
    extract was cut -- not a defect) vs NOT EDGE (ended inside the data with
    no sentinel -- a feed problem). Neither is removed: returns
    (edge_episode_ids, detail) and the caller flags, never filters."""
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
    # "Still running at the extract's last hour?" -- compared NUMERICALLY in
    # hours: `ts + to_timedelta(hr)` overflows timedelta64[ns] on production's
    # million-hour counters and wraps silently (learnings.md); the equivalent
    # `hr > extract_end - ts` is bounded by the extract's own span.
    hours_to_end = (extract_end - ts).dt.total_seconds() / 3600.0
    edge = (hr > hours_to_end) | (hours_to_end <= 0)

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


def load_and_filter(path, cfg=None, examples=None, examples_per_step=3):
    """Section 9.1 mapping + section 9.2 filter chain. Returns (df, waterfall).
    Every DROP is an integrity or scope rule; pricing conditions are FLAGS set
    by `tag_dp_eligibility` and stay in the frame. Pass a dict as `examples`
    to collect up to `examples_per_step` removed episode ids per step."""
    cfg = cfg or load_config()
    excl = cfg["data"]["exclusion_window"]

    df = pd.read_parquet(path).rename(columns=SOURCE_TO_CANONICAL)

    # discount is PERCENT in source -> fraction, exactly once
    df["total_discount"] = df["total_discount"] / 100.0
    df["starting_inventory"] = df["starting_inventory"].round().astype("int64")
    df["ending_inventory"] = df["ending_inventory"].round().astype("int64")

    # Two states for one sku x fc x hour is unresolvable -- keep neither;
    # left in, they also collide two runs into one episode id.
    df = df.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    dup = df.duplicated(subset=["sku_id", "fc", "date", "hour_of_day"],
                        keep=False)
    df = df[~dup]
    df["episode_id"] = assign_episode_ids(df)

    wf = [("raw", len(df) + int(dup.sum()), df.episode_id.nunique(),
           cogs_at_risk(df)),
          ("duplicate_hour_rows_dropped", len(df), df.episode_id.nunique(),
           cogs_at_risk(df))]

    prev_ids = {"ids": set(df.episode_id.unique())}

    def step(d, label):
        wf.append((label, len(d), d.episode_id.nunique(), cogs_at_risk(d)))
        if examples is not None:
            now = set(d.episode_id.unique())
            gone = prev_ids["ids"] - now
            if gone:
                examples[label] = sorted(gone)[:examples_per_step]
            prev_ids["ids"] = now
        return d

    # Feed-gap fragments are not episodes. Runs FIRST: everything downstream
    # assumes an episode_id is a whole window.
    gap_ids, gap_detail = gap_split_windows(df)
    d = df[~df.episode_id.isin(gap_ids)]
    d = step(d, "gap_split_windows_dropped")
    if gap_detail:
        wf[-1] = wf[-1] + (gap_detail,)
    df = d

    # Episode-scoped: a cross-midnight window straddling the boundary must go
    # whole, or re-segmentation turns the remnant into a spurious short window.
    ds = df.date.astype(str)
    inside = ds.ge(excl["start"]) & ds.le(excl["end"])
    d = df[~df.episode_id.isin(df.loc[inside, "episode_id"].unique())]
    d = step(d, "exclusion_window_removed")

    # Outside [0, 1] means the percent -> fraction conversion ran twice or
    # never -- it silently inverts every price, so check it here.
    bad = d.loc[~d.total_discount.between(0, 1), "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "discount_out_of_range_dropped")

    # Impossible quantities drop (incl. negative ending_inventory, which would
    # otherwise be classified as a restock); a missing cost (`cost <= 0`) is
    # an ECONOMIC condition, flagged by tag_dp_eligibility (design.md 5.2).
    neg = ((d.starting_inventory < 0) | (d.units_sold < 0)
           | (d.ending_inventory < 0))
    d = d[~d.episode_id.isin(d.loc[neg, "episode_id"].unique())]
    d = step(d, "negative_quantities_dropped")

    d_before_universe = d
    # THE EPISODE UNIVERSE (design.md 5.2): only CONTINUITY drops; the
    # accounting identity and a clean final close are flags. Hour-level
    # restock/shrink are real events settled at episode level (learnings.md).
    d = d.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    discontinuous = episodes.continuity_breaks(d)
    d = d[~d.episode_id.isin(d.loc[discontinuous, "episode_id"].unique())]
    d = step(d, "episode_universe")

    flow0 = episodes.episode_flow(d)
    status0 = episodes.hour_status(d.starting_inventory, d.units_sold,
                                   d.ending_inventory)
    wf[-1] = wf[-1] + ({
        "rule": ("continuity AND identity AND a clean final hour -- the three "
                 "things that make an episode's inventory readable"),
        "rows_chain_discontinuous_dropped": int(discontinuous.sum()),
        "episodes_dropped": int(len(set(
            d_before_universe.episode_id) - set(d.episode_id))),
        "identity_violations": int(len(episodes.flow_identity_violations(d))),
        "episodes_final_hour_not_clean": int((~flow0.final_hour_clean).sum()),
        "episodes_with_restock": int((flow0.arrived > 0).sum()),
        "episodes_with_shrink": int((flow0.vanished > 0).sum()),
        "rows_restock": int((status0 == episodes.RESTOCK).sum()),
        "rows_write_off": int((status0 == episodes.WRITE_OFF).sum()),
        "rows_hour_shortfall_NOT_dropped": int(
            (status0 == episodes.SHORTFALL).sum()),
        "note": ("Only continuity DROPS here. The identity is provable once "
                 "it holds, so a violation means the supply arithmetic is "
                 "broken rather than the feed; a dirty final hour FLAGS "
                 "final_hour_restock and gates the eligible population. An "
                 "hour-level restock or shrink is neither -- both are real "
                 "events, counted gross and settled at the episode level."),
    },)


    # EPISODE-scoped like every drop after id assignment -- a row-scoped hole
    # mid-window would invalidate the episode universe checked above.
    bad = d.loc[d.category.isna() | d.subcategory.isna(), "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "null_category_dropped")

    d = d.copy()
    d["original_price"] = (d.groupby("episode_id")["original_price"]
                           .transform(lambda s: s.replace(0, np.nan).ffill().bfill()))
    bad = d.loc[d.original_price.isna() | (d.original_price <= 0),
                "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = step(d, "zero_base_price_dropped")

    # `units_gt_inventory_dropped` used to run here and was a mistake: an hour
    # selling more than it opened with is a RESTOCK, now flagged `restocked`
    # (learnings.md on the 18.1pp of COGS the drop cost).

    # RE-SEGMENTATION, which must be a no-op -- checked, not assumed, because
    # a stale continuity check would be silent.
    # MUST RUN BEFORE `negative_window_recovered`: recovery mutates
    # `hours_remaining`, the field the ids derive from, and its synthetic
    # countdown can merge an episode with a real neighbour (learnings.md).
    d = d.sort_values(["sku_id", "fc", "date", "hour_of_day"]).copy()
    resegmented = assign_episode_ids(d)
    moved = int((resegmented != d.episode_id).sum())
    if moved:
        raise AssertionError(
            f"re-segmentation moved {moved} rows to a different episode, so "
            "the ids every flag above is keyed to are stale -- episode_universe "
            "already ran its continuity check against them. TWO causes, and "
            "the second is the one that actually happened:\n"
            "  (a) a filter after the ids are assigned is dropping ROWS rather "
            "than whole episodes, punching a hole in a window. Check that every "
            "drop is `isin(episode_id)`-scoped.\n"
            "  (b) something between id assignment and here MUTATED "
            "`hours_remaining`, which is the field the ids are derived from. "
            "`negative_window_recovered` does exactly that, and its synthetic "
            "countdown can line up with a genuine neighbouring window and MERGE "
            "them -- no filter is misbehaving at all. That is why recovery runs "
            "after this check; do not move it back above.")
    wf.append(("contiguous_episodes_built", len(d), d.episode_id.nunique(),
               cogs_at_risk(d)))

    # "Manufacturing" SKUs enter with an already-negative counter; dropping
    # them selects on category. Recovered as a synthetic COUNTDOWN (not a
    # clamp -- ids, DP horizon and extend_to_window all read the counter) iff
    # the episode fits inside `data.manufacturing_window_hours` -- checked,
    # not trusted.
    # RUNS AFTER the re-segmentation check: this is the one step that mutates
    # `hours_remaining`, the field the ids are derived from (design.md 5.2).
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

    # A counter still negative after recovery gates dp_eligible as
    # `negative_window` (see DP_INELIGIBLE). Restock and edge-truncation flags
    # are set in `tag_dp_eligibility`; both tests must run AFTER
    # re-segmentation, hence the end of the chain.

    d["d_ref"] = d.category.map(lambda c: reference_discount(cfg, c))
    d["d_max"] = 1.0 - d.cost / d.original_price
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    d, economic = tag_dp_eligibility(d, cfg)

    # THE TWO POPULATION GATES GET A ROW EACH: they drop NOTHING, and they
    # are exactly where the consumers diverge (design.md 5.2).
    elig = d.episode_eligible
    wf.append(("eligible", int(elig.sum()),
               int(d.loc[elig, "episode_id"].nunique()),
               cogs_at_risk(d[elig]), eligible_detail(d)))
    # The nesting (integrity > eligible > dp_eligible) follows from
    # continuity, not the gate list -- asserted, since a break reads as a
    # report error rather than a broken invariant.
    leaked = int(d.loc[d.dp_eligible & ~elig, "episode_id"].nunique())
    assert not leaked, (
        f"{leaked} episodes are dp_eligible but not eligible. The waterfall "
        "is read as strictly nested (integrity > eligible > dp_eligible) and "
        "this breaks that. `accounting_closes` is the only `eligible` "
        "condition with no DP gate of its own, so suspect it first: it is an "
        "identity given continuity, so a failure means episode_universe let "
        "a discontinuous episode through.")
    wf.append(("dp_eligible", int(d.dp_eligible.sum()),
               int(d.loc[d.dp_eligible, "episode_id"].nunique()),
               cogs_at_risk(d[d.dp_eligible]), economic))
    d = add_ref_rate_features(d, cfg)
    # Guarantee window order on the way out: consumers take .first()/.last()
    # per episode without re-sorting, and the merges above can reorder rows.
    d = d.sort_values(["episode_id", "date", "hour_of_day"])
    return d.reset_index(drop=True), wf


def add_ref_rate_features(d, cfg):
    """Point-in-time demand-rate features (sku_ref_sales_rate_30d,
    prior_episode_ref_sales_rate), built ONLY from anchor hours and lagged
    strictly before the episode's first date -- an episode never sees its own
    day. Within-episode lags are deliberately absent (docs/design.md)."""
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

    # Read as of the episode's FIRST date: row-dated reads would let a
    # cross-midnight window's later rows see its own first-day sales.
    d["_date_str"] = (d.groupby("episode_id")["date"].transform("min")
                      .astype(str))
    d = (d.merge(feats.rename(columns={"date": "_date_str"}),
                 on=["sku_id", "fc", "_date_str"], how="left"))

    # prior_episode_ref_sales_rate at true EPISODE grain: a daily shift would
    # hand a multi-day episode its own earlier day as "previous episode".
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
    """Everything the PRE-LAUNCH artifacts may see: episodes whose window
    opened on or before `split.test_end`; episode-scoped, so a window opening
    before the boundary is kept whole. Bounds `calibration_fit_window: "all"`
    and the tau_initial derivation away from the hold-out (see data.holdout)."""
    return episodes.window_slice(d, None, cfg["data"]["split"]["test_end"])



# WATERFALL_STEPS: the single wording of every step, exported verbatim to the
# workbook for readers who have never seen the code. DP_INELIGIBLE below is
# the same idea for the gates -- modelling limits invisible to the demand
# model's FEATURES, ordered most-fundamental-first (episodes are labelled with
# the FIRST reason tripped, so the column reads as a cause).
WATERFALL_STEPS = (
    ("raw", "--",
     "the starting count, before anything is removed. Every percentage in the "
     "report is a share of this row's COGS at risk"),
    ("duplicate_hour_rows_dropped", "rows, BOTH copies",
     "two different states recorded for one sku x fc x hour. There is no "
     "principled way to choose between them, and keeping both collides two "
     "runs into a single episode id"),
    ("gap_split_windows_dropped", "episode, EVERY fragment",
     "a hole in the hourly feed splits one source window into two, and neither "
     "half is an episode: the first ends with no write-off sentinel so its "
     "scrap is unknown, and the second opens MID-window with the wrong "
     "starting stock and a first row that reads as an entry row. Detected from "
     "the window counter, which across a gap keeps falling in step with the "
     "clock instead of resetting upward"),
    ("exclusion_window_removed", "episode",
     "any episode with ANY hour inside the known demand-issue period. This is "
     "a SCOPE rule, not an integrity one -- the rows are fine, the period is "
     "not representative of normal trading"),
    ("discount_out_of_range_dropped", "episode",
     "a discount outside [0, 1], which means the percent-to-fraction "
     "conversion was applied twice or not at all"),
    ("negative_quantities_dropped", "episode",
     "impossible quantities: negative inventory or negative sales. NOT "
     "`cost <= 0`, which is a flag rather than a drop -- see cost_missing"),
    ("episode_universe", "episode",
     "the three conditions that make an episode's inventory readable, checked "
     "once and before any filter with an opinion about price, category or "
     "cost. Only CONTINUITY drops (ending[t] must equal starting[t+1]); the "
     "accounting identity and a clean final hour are flagged instead. Note "
     "`units_sold > starting_inventory` is a RESTOCK, not an impossible "
     "quantity"),
    ("null_category_dropped", "episode",
     "no category or subcategory, so there is no reference discount and no "
     "dispersion cell to put the episode in. Episode-scoped on purpose: "
     "dropping single ROWS punched holes mid-window and manufactured chain "
     "breaks the feed never had"),
    ("zero_base_price_dropped", "episode",
     "`original_price` still null or zero after filling forward and backward "
     "within the episode, so there is no price to discount from"),
    ("contiguous_episodes_built", "--",
     "not a filter. Re-segmentation, and now a no-op that RAISES if it ever "
     "stops being one, since every drop after the ids are assigned is "
     "episode-scoped and so nothing can punch a hole in a window"),
    ("negative_window_recovered", "episode",
     "not a drop. A counter that enters ALREADY negative is a known source "
     "pattern for manufactured goods, not a defect; where the episode runs no "
     "longer than the manufacturing window its counter is rewritten as a "
     "synthetic countdown. Changes no counts, so the row reports what it "
     "recovered. This is the last hard-drop row, so its counts ARE the "
     "`integrity` population"),
    ("eligible", "GATE -- flags, drops nothing",
     "the accounting identity holds, the final hour is clean, and the episode "
     "CLOSED. What a frozen artifact needs: a censored likelihood is only "
     "correct if we know which hours ran out. Read by the demand model and the "
     "artifact fits, and by every scrap / IL / clearance figure -- "
     "`scrap_units` returns NaN outside this set"),
    ("dp_eligible", "GATE -- flags, drops nothing",
     "everything `eligible` requires PLUS what the solver needs: a feasible "
     "price tier, a horizon it can read, and one inventory pool. Read by the "
     "DP, the backtest, shadow mode, the calibration gate and the A/B. The six "
     "reasons are listed separately below"),
)

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
    ("outcome_unknown",
     "the episode never closed inside this data: its last row carries no "
     "write-off sentinel, so it is still holding stock and nothing about how "
     "it ENDED is observed. Only the extract's final hours can do this "
     "legitimately -- split boundaries cannot, since window_slice assigns "
     "episodes whole by opening date -- so on a 175-day extract of ~36h "
     "windows this should be under 1% of episodes. Gates `eligible` as well "
     "as `dp_eligible`: an unfinished episode is not a complete observation "
     "of anything, and every consumer that met one silently mis-weighted it "
     "(clearance read sold-so-far, and the backtest graded a truncated actual "
     "arm against two full-horizon simulated ones). KEPT in the integrity "
     "population so m11 can still count the residue"),
    ("final_hour_restock",
     "the LAST row sold more than it opened with, so stock arrived during the "
     "episode's final hour. If the source also zeroed `ending` to write the "
     "remainder off, how much arrived and how much was scrapped are two "
     "unknowns with one equation -- the close is a guess, so the episode's "
     "scrap is NaN and it is out of every IL and clearance figure. Gates "
     "`episode_eligible` too, because a censored/not-censored call cannot be "
     "made on an ambiguous final hour. Two of the six gates do that -- this "
     "one and `outcome_unknown` -- and they are exactly conditions 2 and 3 of "
     "`eligible`; the other four are solver requirements the demand model "
     "cannot see"),
)

# Reported, NOT gating: a below-cost hour is a legacy-policy price. The
# backtest's DP arm is self-anchored and never sees it; shadow's refusal from
# the crossing hour on is the cost floor working (design.md 5.2).
BELOW_COST_HOURS = (
    "some hour's OFFERED price is under cost. Tested on "
    "original_price x (1 - d), never applied_price, which the source zeroes "
    "on the ~78% of rows that sold nothing")


# WHO READS EACH ROW'S POPULATION. Hard drops leave the frame for everyone;
# the last two rows are flags, and they are where the consumers split.
HARD_DROP_USED_BY = (
    "everything downstream -- the rows are gone, no consumer can see them")

GATE_USED_BY = {
    "eligible": (
        "the DEMAND MODEL and the frozen artifacts (baseline mu_ref, the "
        "elasticity prior, r and rho) when baseline_model.train_population is "
        "'eligible', plus every scrap / IL / clearance figure -- `scrap_units` "
        "returns NaN outside this set, so an ineligible episode cannot enter "
        "an IL number even by accident"),
    "dp_eligible": (
        "the DP SOLVER, the backtest, shadow mode, the calibration gate and "
        "the A/B. These callers pass 'dp_eligible' explicitly because for them "
        "it is not a configurable choice -- an episode here is one the solver "
        "can actually price"),
}


def eligible_detail(d):
    """What the `eligible` gate costs. Deliberately thin: per-reason blocks
    belong to `dp_eligible`; repeating them here would invite adding the two
    rows' exclusions together, and they are nested, not disjoint."""
    not_elig = ~d.episode_eligible
    return {
        "gated_by": ["accounting_closes", "final_hour_clean", "closed"],
        "episodes_excluded": int(d.loc[not_elig, "episode_id"].nunique()),
        "rows_excluded": int(not_elig.sum()),
        "note": (
            "FLAGGED, NOT DROPPED -- every row is still in the frame and the "
            "integrity population still holds them. Three conditions, all in "
            "common.episodes.episode_flow: the accounting identity holds, the "
            "final hour is clean, and the episode CLOSED. That is what a "
            "FROZEN ARTIFACT needs -- a censored likelihood is only correct if "
            "we know which hours ran out. Two of the three are also DP gates "
            "(`outcome_unknown` = not closed, `final_hour_restock` = unclean "
            "final hour), so this row and the next OVERLAP by construction: "
            "the populations are nested, integrity > eligible > dp_eligible, "
            "and their exclusions must never be added together."),
    }


def tag_dp_eligibility(d, cfg):
    """Flag (never drop) the episodes the DP cannot price; episode-scoped,
    since the DP plans the window as a unit (docs/learnings.md on what
    dropping cost). Must run AFTER re-segmentation so the ids it groups on
    are final and the chain tests cannot mistake a data gap for a restock."""
    cap = cfg["data"]["max_window_hours"]
    # supply accounting attached once; `episode_supply` is the only correct
    # clearance denominator now that a window can gain stock
    flow = episodes.episode_flow(d)
    for col, src in (("units_restocked", "arrived"),
                     ("units_shrink", "vanished"),
                     ("episode_supply", "supply"),
                     ("episode_scrap", "scrap"),
                     ("episode_clearance", "clearance")):
        d = d.copy()
        d[col] = d.episode_id.map(flow[src]).astype(
            float if src == "clearance" else "int64")
    # ELIGIBLE: the frozen-artifact gate; `dp_eligible` is a strict subset
    # with solver requirements on top.
    d["final_hour_clean"] = d.episode_id.map(flow.final_hour_clean).astype(bool)
    # flow.eligible carries all three conditions (reconciles, clean final
    # hour, CLOSED); outcome_known stays its own column -- the DP gate
    # reports on it by name.
    d["outcome_known"] = d.episode_id.map(flow.closed).astype(bool)
    d["episode_eligible"] = d.episode_id.map(flow.eligible).astype(bool)

    tests = {
        # ~(cost > 0), not cost <= 0: NaN <= 0 is False, so a NULL cost
        # sailed through and handed the DP a NaN d_max
        "cost_missing": ~(d.cost > 0),
        "non_priceable": d.cost >= d.original_price,
        "negative_window": d.hours_remaining < 0,
        "window_too_long": d.hours_remaining > cap,
        "outcome_unknown": ~d.outcome_known,
        "final_hour_restock": ~d.final_hour_clean,
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
    # Moved inventory is reported and gates nothing: the replay re-solves
    # hourly against actual stock, learning of arrivals at the next hour
    # exactly as production does.
    for name, col in (("restocked", "units_restocked"),
                      ("shrink", "units_shrink")):
        hit = d[col] > 0
        detail[name] = {
            "episodes": int(d.loc[hit, "episode_id"].nunique()),
            "rows": int(hit.sum()),
            "cogs_at_risk": round(cogs_at_risk(d[hit]), 1),
            "units": int(d.loc[~d.episode_id.duplicated() & hit, col].sum()),
            "still_dp_eligible": int(
                d.loc[hit & d.dp_eligible, "episode_id"].nunique()),
        }

    # SHRINK OR SKEW: shrink is ~independent of sales rate, feed-timing skew
    # grows with units_sold. Realign hour_adjustment's (date, hour)-sorted
    # values to frame order before pairing with units_sold.
    adj = (episodes.hour_adjustment(d.reset_index(drop=True))
           .sort_index().to_numpy())
    lost, sold = -adj[adj < 0], d.units_sold.to_numpy()
    corr = None
    if len(lost) > 2:
        s_sold = sold[adj < 0]
        if lost.std() > 0 and s_sold.std() > 0:
            c = np.corrcoef(lost, s_sold)[0, 1]
            # None, never NaN: json.dump writes a bare NaN literal that no
            # strict parser reads back
            corr = round(float(c), 4) if np.isfinite(c) else None
    detail["shrink"]["diagnosis"] = {
        "units_p50": float(np.median(lost)) if len(lost) else 0.0,
        "units_p90": float(np.percentile(lost, 90)) if len(lost) else 0.0,
        "share_of_1_unit": round(float((lost == 1).mean()), 4) if len(lost) else 0.0,
        "shrink_vs_sold_corr": corr,
        "mean_sold_on_shrink_rows": round(
            float(sold[adj < 0].mean()), 3) if len(lost) else 0.0,
        "mean_sold_on_clean_rows": round(float(sold[adj == 0].mean()), 3)
            if (adj == 0).any() else 0.0,
        "reading": ("corr near 0 with small absolute losses reads as real "
                    "SHRINK -- damage, transfer, sampling. corr well above 0, "
                    "or shrink rows selling far more than clean ones, reads as "
                    "TIMING SKEW between the sales feed and the stock "
                    "snapshot: a join to fix upstream, not shrink to chase."),
    }

    # also reported-only: an unclosed ending is a missing OUTCOME, and every
    # outcome consumer already handles it (NaN scrap, outcome_known, COMPLETED).
    at_edge, edge_detail = edge_truncated_episodes(d)
    edge = d.episode_id.isin(at_edge)
    d["edge_truncated"] = edge
    edge_detail["rows"] = int(edge.sum())
    edge_detail["cogs_at_risk"] = round(cogs_at_risk(d[edge]), 1)
    edge_detail["still_dp_eligible"] = int(
        d.loc[edge & d.dp_eligible, "episode_id"].nunique())
    detail["edge_truncated"] = edge_detail

    # THE EPISODE IDENTITY, checked not assumed: provable given continuity,
    # so a violation is a bug in episode_flow that would move every IL figure.
    violations = episodes.flow_identity_violations(d)
    detail["flow_identity"] = {
        "rule": "opening + restocked == sold + shrink + leftover_at_last_hour",
        "episodes_checked": int(d.episode_id.nunique()),
        "violations": int(len(violations)),
        "holds": not len(violations),
        "note": ("Every unit is accounted for by exactly one of three fates: "
                 "sold, shrunk, or still on the shelf at the last hour. "
                 "Guaranteed by chain continuity, so a violation means the "
                 "supply arithmetic is broken, not the source."),
    }

    # Shrink/restock pairs are NOT netted (an earlier version did, wrongly);
    # both counted in full, adjacency reported for the business.
    paired = flow[(flow.arrived > 0) & (flow.vanished > 0)]
    detail["shrink_and_restock_together"] = {
        "episodes": int(len(paired)),
        "share_of_episodes": round(float(len(paired)) / max(len(flow), 1), 6),
        "units_arrived": int(paired.arrived.sum()),
        "units_shrunk": int(paired.vanished.sum()),
        "episodes_equal_in_and_out": int((paired.arrived == paired.vanished).sum()),
        "note": ("Episodes with BOTH an arrival and a loss. Where the two are "
                 "equal and the hours adjacent, a sale recorded an hour after "
                 "the stock moved would look identical -- but that is a guess, "
                 "and the counts here are not netted on the strength of it. "
                 "Both figures stand: the episode is flagged restocked, the "
                 "shrink settles into scrap, and it stays dp_eligible."),
    }

    # Anomaly location by category and month: an incident, a corner of the
    # catalogue, or a standing feed property are three different chases.
    bad = d.units_shrink > 0
    if bad.any():
        ep = d[bad].groupby("episode_id").agg(
            month=("date", lambda s: str(pd.to_datetime(s.iloc[0]))[:7]),
            units=("units_shrink", "first"),
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
            "note": ("Hours where stock left the shelf that no sale or "
                     "write-off accounts for. These episodes are KEPT and "
                     "stay dp_eligible: the shrink settles into scrap at the "
                     "episode level, so the identity still closes. "
                     "Concentrated in one month reads as an incident; "
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
        "The three artifact fits read baseline_model.train_population, which "
        "selects one of the THREE NESTED populations -- 'integrity' (every row "
        "that survived the hard drops), 'eligible' (the previous waterfall "
        "row), or 'dp_eligible' (this row); the DP, the backtest, shadow, the "
        "calibration gate and the A/B always read dp_eligible. Reasons are "
        "first-match in the order " + ", ".join(n for n, _ in DP_INELIGIBLE)
        + ". Counts overlap, so they do not sum to episodes_dp_ineligible.")
    return d, detail


def population(d, cfg, which=None):
    """ONE definition of the three NESTED populations: integrity (survived
    the chain), eligible (identity holds + clean final hour + CLOSED -- what
    a frozen artifact needs), dp_eligible (eligible + solver requirements).
    None reads baseline_model.train_population; DP callers pass explicitly."""
    which = which or cfg["baseline_model"]["train_population"]
    if which == "integrity":
        return d
    if which == "eligible":
        return d[d.episode_eligible] if "episode_eligible" in d else d
    if which == "dp_eligible":
        return d[d.dp_eligible] if "dp_eligible" in d else d
    raise ValueError(f"unknown population {which!r}")


def split_frames(d, cfg):
    """Date splits for baseline fitting only (config data.split). An episode
    is assigned WHOLLY to the split its window started in, so a boundary never
    runs through the middle of a cross-midnight episode."""
    s = cfg["data"]["split"]
    slice_ = episodes.window_slice
    return {
        "train": slice_(d, s["train_start"], s["train_end"]),
        "calib": slice_(d, s["calib_start"], s["calib_end"]),
        "test": slice_(d, s["test_start"], s["test_end"]),
    }


def waterfall_rows(waterfall, cfg=None):
    """The waterfall as JSON rows -- one definition for both writers (split
    manifest and phase 0). Stages are (label, rows, episodes, cogs_at_risk)
    plus an optional detail dict merged into the row; rows carry kind/used_by
    and COGS cost. `cogs_dropped` is correctly NEGATIVE at re-segmentation."""
    train_pop = (cfg or {}).get("baseline_model", {}).get("train_population")
    raw = waterfall[0][3] if waterfall else 0.0
    out, prev = [], None
    for t in waterfall:
        label, rows, eps, cogs = t[:4]
        gate = label in GATE_USED_BY
        used_by = GATE_USED_BY[label] if gate else HARD_DROP_USED_BY
        if label == "eligible" and train_pop is not None:
            used_by += (
                f". baseline_model.train_population is {train_pop!r}, so the "
                + ("artifact fits DO read this population"
                   if train_pop == "eligible" else
                   f"artifact fits read {train_pop!r} instead"))
        row = {"step": label, "rows": rows, "episodes": eps,
               "cogs_at_risk": round(cogs, 1),
               "cogs_pct_of_raw": round(cogs / raw, 6) if raw else None,
               "kind": "population_gate" if gate else "hard_drop",
               "used_by": used_by}
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
            "data_quality_waterfall": waterfall_rows(waterfall, cfg),
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
             "cogs_dropped", "cogs_dropped_pct_of_raw", "kind", "used_by")
    rows = waterfall_rows(wf, cfg)
    # The two GATE rows are marked in the margin. Without it the last rows read
    # as more drops, and the reader cannot see that the population splits there
    # rather than shrinking.
    for row in rows:
        pct = row.get("cogs_dropped_pct_of_raw")
        gate = row["kind"] == "population_gate"
        print(f"{'GATE ' if gate else '     '}{row['step']:29s} "
              f"rows {row['rows']:>10,}  "
              f"episodes {row['episodes']:>9,}  "
              f"cogs {row['cogs_at_risk']:>16,.0f}"
              + (f"  dropped {pct:>7.2%}" if pct is not None else ""))
        if gate:
            print(f"{'':34s}   read by: {row['used_by']}")
        for k, v in row.items():
            if k not in fixed:
                print(f"{'':34s}   {k}: {v}")
    if rows:
        kept = rows[-1]["cogs_pct_of_raw"]
        print(f"\nCOGS at risk surviving: {rows[-1]['cogs_at_risk']:,.0f} of "
              f"{rows[0]['cogs_at_risk']:,.0f} raw"
              + (f" ({kept:.2%})" if kept is not None else ""))
        # WHICH NUMBER TO QUOTE FOR WHICH CLAIM. The unmarked rows shrink one
        # population; the two GATE rows split it three ways, and the three are
        # NESTED -- their exclusions must never be added together.
        by_step = {r["step"]: r for r in rows}
        hard = [r for r in rows if r["kind"] == "hard_drop"]
        integrity = hard[-1] if hard else None
        print("\nThree populations come out of this, NESTED -- never add their "
              "exclusions together:")
        for name, r, what in (
                ("integrity", integrity,
                 "everything that survived the hard drops above"),
                ("eligible", by_step.get("eligible"),
                 "demand model, prior, dispersion; every scrap / IL / "
                 "clearance figure"),
                ("dp_eligible", by_step.get("dp_eligible"),
                 "DP solver, backtest, shadow, calibration gate, A/B")):
            if r:
                print(f"  {name:14s} episodes {r['episodes']:>9,}  "
                      f"cogs {r['cogs_pct_of_raw']:>7.2%} of raw   {what}")
    print(f"wrote {args.out} and {args.manifest}")


if __name__ == "__main__":
    main()
