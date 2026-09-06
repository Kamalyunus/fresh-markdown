"""fit.prepare_data -- schema mapping, filter chain, episode construction.

Implements docs/design.md section 5.2 (the filter chain; episodes per 12a);
column mapping applied here once and nowhere else. Load-bearing (source
schema, section 6): `discount` is PERCENT in source,
converted to a fraction exactly once at load; `final_price` is a realised
price (0 on zero-sale rows), never used to reconstruct the offered price;
episode_id is built here, the rule persisted in the split manifest."""

import argparse
import os

import numpy as np
import pandas as pd

from common.config import load_config, reference_discount
from common import episodes
from common.io import write_json
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
    "episode_id = sku_id|fc|<first hour of the window>: a maximal run of "
    "consecutive hourly rows over which hours_remaining decrements by one "
    "per elapsed hour. NOT keyed by calendar date -- windows cross midnight, "
    "and a date key would split one episode in two at the seam.")


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
        "note": ("Every fragment of a gap-split window is dropped: the "
                 "second opens mid-window and its first row would read as an "
                 "ENTRY row in the elasticity fit."),
    }
    return ids, detail


def episode_cogs(d):
    """Exposure PER EPISODE (indexed by episode_id): unit cost x SUPPLY
    (opening stock + GROSS arrivals; shrink not subtracted), read at the
    opening row. NaN where the opening row's cost is null. Requires window
    order, which `load_and_filter` guarantees. Pre-`episode_universe` stages
    deliberately carry impossible values / an unverified arrival term (5.2).
    Per episode so the waterfall and the flag details reuse ONE table."""
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
    return pd.Series(d.cost.to_numpy()[opening] * supply,
                     index=opening_ids.to_numpy(), dtype=float)


def cogs_at_risk(d, per_episode=None):
    """`episode_cogs` summed over the episodes in `d` -- NaN, never skipped,
    when any is NaN. `per_episode` is a precomputed `episode_cogs` table
    covering every episode in `d` (the chain drops whole episodes, so one
    table serves every stage after the ids are fixed)."""
    if not len(d):
        return 0.0
    if per_episode is None:
        per_episode = episode_cogs(d)
    return float(per_episode.reindex(pd.unique(d.episode_id)).to_numpy().sum())


def edge_truncated_episodes(d, flow):
    """Split unclosed episodes into EDGE (window still running when the
    extract was cut -- not a defect) vs NOT EDGE (ended inside the data with
    no sentinel -- a feed problem). Neither is removed: returns
    (edge_episode_ids, detail) and the caller flags, never filters. `flow`
    is `episode_flow(d)`, which the one caller already holds."""
    last = episodes.last_rows(d)
    ids = last.episode_id.to_numpy()
    unknown = flow.index[~flow.closed]    # NOT_CLOSED is exactly ~closed

    ts = pd.Series(
        (pd.to_datetime(last.date) + pd.to_timedelta(last.hour_of_day, unit="h")
         ).to_numpy(), index=ids)
    extract_end = ts.max()
    hr = pd.Series(last.hours_remaining.to_numpy(), index=ids).clip(lower=0)
    left = pd.Series(
        episodes.leftover_units(last.starting_inventory, last.units_sold).to_numpy(),
        index=ids)
    # compared numerically in hours: ts + to_timedelta(hr) overflows
    # timedelta64[ns] on million-hour counters and wraps silently
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
        "note": ("FLAGGED, NOT DROPPED; stays dp_eligible. Only the ENDING "
                 "is unknown, and every scrap/IL consumer already excludes an "
                 "unclosed ending. share_of_unclosed_explained_by_edge near "
                 "1.0 = purely the extract boundary; well below = a feed "
                 "problem."),
    }
    return at_edge, detail


# The columns an episode id is built from. A null in any of them has no
# episode to belong to: `assign_episode_ids` groups on sku_id x fc (a NaN key
# falls out of every groupby) and stamps the window start from date x hour,
# so such rows collapsed into one NaN "episode" that later stages then read
# as a window. INTEGRITY, so it DROPS (rule 14) -- counted, never silent.
EPISODE_KEY = ("sku_id", "fc", "date", "hour_of_day")


def null_key_rows(df):
    """Mask of rows with a null in any EPISODE_KEY column, and a per-column
    count for the waterfall detail. Such a row can be placed in no window
    at all; row-scoped by construction."""
    nulls = df[list(EPISODE_KEY)].isna()
    return nulls.any(axis=1), {c: int(nulls[c].sum()) for c in EPISODE_KEY}


def null_counter_windows(df):
    """Mask of every row of the source WINDOW that holds a null
    `hours_remaining`, and the number of such windows. The counter is the
    field the ids derive from: a null one made assign_episode_ids open a
    NEW episode on that row (NaN != -1), so one bad hour became a one-row
    "episode" that closed on its own zero and read DP-eligible -- found on
    the owner's extract by the pilot simulator, whose templates then
    carried a NaN window length. The row itself can be placed (its key is
    whole), so the drop is the WHOLE window it sits in (rule 15): a
    row-scoped drop left a fragment opening mid-window that no later stage
    could tell from a real entry.

    The window is read with assign_episode_ids' and gap_split_windows'
    own signals, since the null row breaks neither cleanly: rows of one
    sku x fc stay in one window while the clock advances an hour (a
    duplicate hour, too), or skips hours the counter ran down by exactly
    (a feed gap -- the fragment beyond it must go too); a counter that
    resets upward between two non-null rows opens a NEW window, so a
    back-to-back neighbour is kept. A gap whose counter is unreadable on
    either side is a break: the far side survives as a fragment nothing
    later can see -- bounded to a null adjacent to a gap, and counted in
    `null_counter_gap_fragments_kept`."""
    if not df.hours_remaining.isna().any():
        return pd.Series(False, index=df.index), {"windows": 0,
                                                   "gap_fragments_kept": 0}
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    grp = [df.sku_id, df.fc]
    dt_h = ts.groupby(grp).diff().dt.total_seconds() / 3600.0
    hr = df.hours_remaining
    hr_drop = -hr.groupby(grp).diff()                 # NaN beside a null
    same_window = ((dt_h <= 1.0) | ((dt_h > 1.0) & (dt_h == hr_drop))) \
        & ~(hr_drop < 0)                              # an upward reset opens one
    window = (~same_window.fillna(False) | dt_h.isna()).cumsum()
    bad = set(window[hr.isna()])
    mask = window.isin(bad)
    # a gap this window could not be read across (a null on either side)
    unread = (dt_h > 1.0) & hr_drop.isna() & mask.groupby(grp).shift().fillna(False)
    return mask, {"windows": len(bad), "gap_fragments_kept": int(unread.sum())}


def recover_negative_windows(d, cap):
    """`hours_remaining` rewritten as a synthetic countdown `(cap-1) -
    position` on episodes that ENTER negative and fit inside `cap` hours
    (manufacturing SKUs). Returns (frame, mask of rewritten rows). The ONE
    step that mutates the counter the ids derive from: it must run AFTER the
    re-segmentation check, and the ids are never re-derived from it."""
    entry = d.groupby("episode_id")["hours_remaining"].transform("first")
    length = d.groupby("episode_id")["hours_remaining"].transform("size")
    recoverable = (entry < 0) & (length <= cap)
    if recoverable.any():
        d = d.copy()
        position = d.groupby("episode_id").cumcount()
        d.loc[recoverable, "hours_remaining"] = (cap - 1) - position[recoverable]
    return d, recoverable


def load_and_filter(path, cfg=None, examples=None, examples_per_step=3):
    """Source mapping (design 6) + the filter chain (design 5.2, episodes per
    12a). Returns (df, waterfall).
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

    df = df.sort_values(list(EPISODE_KEY))
    # the `raw` row counts rows, episodes and COGS on ONE basis: the frame as
    # read, duplicates included (their ids collide -- that is the defect)
    # and null counters included (each opens up to two phantom ids, gone
    # with their window at the next stage)
    df["episode_id"] = assign_episode_ids(df)
    wf = []
    prev_ids = {"ids": set()}
    # per-episode COGS, computed ONCE the ids are final: every later stage
    # drops whole episodes (asserted at contiguous_episodes_built), so each
    # stage's cogs_at_risk is a sum over this table, not a fresh pass
    per_ep = {"cogs": None}

    def step(d, label, detail=None, gate=False):
        """Append one waterfall row for `d`. `gate` rows (the population
        gates) drop nothing, so they neither record removed-episode examples
        nor advance the set the next drop is measured against."""
        row = (label, len(d), d.episode_id.nunique(),
               cogs_at_risk(d, per_ep["cogs"]))
        wf.append(row if detail is None else row + (detail,))
        if examples is not None and not gate:
            now = set(d.episode_id.unique())
            gone = prev_ids["ids"] - now
            if gone:
                examples[label] = sorted(gone)[:examples_per_step]
            prev_ids["ids"] = now
        return d

    step(df, "raw", gate=True)
    # no episode key, no episode: dropped before the duplicate test, which
    # would otherwise read every null-keyed hour as a duplicate of the rest
    null_key, null_by_col = null_key_rows(df)
    df = df[~null_key]
    null_counter, null_detail = null_counter_windows(df)
    df = df[~null_counter]
    step(df, "null_key_rows_dropped", {
        "rows_dropped": int(null_key.sum() + null_counter.sum()),
        "nulls_by_column": null_by_col,
        "null_counter_rows": int(null_counter.sum()),
        "null_counter_windows_dropped": int(null_detail["windows"]),
        "null_counter_gap_fragments_kept": int(null_detail["gap_fragments_kept"]),
        "note": ("a row with no sku_id, fc, date or hour belongs to no "
                 "episode; kept, it collapsed into one NaN episode id "
                 "(INTEGRITY: drop, rule 14; row-scoped by construction -- "
                 "there is no episode to scope it to). A null window "
                 "counter drops the whole clock-contiguous run it sits in: "
                 "kept, it opened a one-row episode that read DP-eligible; "
                 "dropped alone, it left a fragment opening mid-window.")})
    # Two states for one sku x fc x hour is unresolvable -- keep neither;
    # left in, they also collide two runs into one episode id.
    dup = df.duplicated(subset=list(EPISODE_KEY), keep=False)
    df = df[~dup]
    df["episode_id"] = assign_episode_ids(df)
    per_ep["cogs"] = episode_cogs(df)
    prev_ids["ids"] = set(df.episode_id.unique())
    step(df, "duplicate_hour_rows_dropped", gate=True)

    # Feed-gap fragments are not episodes. Runs FIRST: everything downstream
    # assumes an episode_id is a whole window.
    gap_ids, gap_detail = gap_split_windows(df)
    d = df[~df.episode_id.isin(gap_ids)]
    df = step(d, "gap_split_windows_dropped", gap_detail or None)

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

    flow0 = episodes.episode_flow(d)
    status0 = episodes.hour_status(d.starting_inventory, d.units_sold,
                                   d.ending_inventory)
    d = step(d, "episode_universe", {
        "rule": "continuity AND identity AND a clean final hour",
        "rows_chain_discontinuous_dropped": int(discontinuous.sum()),
        "episodes_dropped": int(len(set(
            d_before_universe.episode_id) - set(d.episode_id))),
        "identity_violations": int(len(
            episodes.flow_identity_violations(d, flow=flow0))),
        "episodes_final_hour_not_clean": int((~flow0.final_hour_clean).sum()),
        "episodes_with_restock": int((flow0.arrived > 0).sum()),
        "episodes_with_shrink": int((flow0.vanished > 0).sum()),
        "rows_restock": int((status0 == episodes.RESTOCK).sum()),
        "rows_write_off": int((status0 == episodes.WRITE_OFF).sum()),
        "rows_hour_shortfall_NOT_dropped": int(
            (status0 == episodes.SHORTFALL).sum()),
        # censoring is decided on the LAST row only (design 12a): a row that
        # emptied the shelf mid-episode means the feed carried on past zero
        "censoring": episodes.censoring_off_last_row(d),
        "note": ("Only continuity DROPS here; the identity and a dirty "
                 "final hour FLAG. Hour-level restock/shrink are real events, "
                 "counted gross, settled at episode level."),
    })

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

    # RE-SEGMENTATION, which must be a no-op -- checked, not assumed. Runs
    # BEFORE `negative_window_recovered`, the one step that mutates
    # `hours_remaining`, the field the ids derive from.
    d = d.sort_values(["sku_id", "fc", "date", "hour_of_day"]).copy()
    resegmented = assign_episode_ids(d)
    moved = int((resegmented != d.episode_id).sum())
    if moved:
        raise AssertionError(
            f"re-segmentation moved {moved} rows -- the ids the flags are "
            "keyed to are stale. Either (a) a filter is dropping ROWS rather "
            "than whole episodes, or (b) something mutated hours_remaining "
            "before this check (negative_window_recovered must stay AFTER "
            "it).")
    d = step(d, "contiguous_episodes_built")

    # manufacturing SKUs enter with a negative counter; recovered as a
    # synthetic countdown iff the episode fits inside the cap. Runs AFTER the
    # re-segmentation check -- this is the one step that mutates the counter.
    cap = int(cfg["data"]["manufacturing_window_hours"])
    entry = d.groupby("episode_id")["hours_remaining"].transform("first")
    d, recoverable = recover_negative_windows(d, cap)
    d = step(d, "negative_window_recovered", {
        "episodes_recovered": int(d.loc[recoverable, "episode_id"].nunique()),
        "rows_recovered": int(recoverable.sum()),
        "window_hours_assumed": cap,
        "episodes_entering_negative_but_longer_than_cap":
            int(d.loc[(entry < 0) & ~recoverable, "episode_id"].nunique())})

    # A counter still negative after recovery gates dp_eligible as
    # `negative_window` (see DP_INELIGIBLE). Restock and edge-truncation flags
    # are set in `tag_dp_eligibility`; both tests must run AFTER
    # re-segmentation, hence the end of the chain.

    d["d_ref"] = d.category.map(lambda c: reference_discount(cfg, c))
    d["d_max"] = 1.0 - d.cost / d.original_price
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    # the flow of every episode still here is unchanged since the universe
    # stage (only whole episodes left; the counter and price fills do not
    # enter it), so the table is re-used rather than re-derived
    d, economic = tag_dp_eligibility(
        d, cfg, flow=flow0.loc[pd.unique(d.episode_id)],
        per_episode_cogs=per_ep["cogs"])

    # THE TWO POPULATION GATES GET A ROW EACH: they drop NOTHING, and they
    # are exactly where the consumers diverge (design.md 5.2).
    elig = d.episode_eligible
    step(d[elig], "eligible", eligible_detail(d), gate=True)
    # The nesting (integrity > eligible > dp_eligible) follows from
    # continuity, not the gate list -- asserted.
    leaked = int(d.loc[d.dp_eligible & ~elig, "episode_id"].nunique())
    assert not leaked, (
        f"{leaked} episodes are dp_eligible but not eligible; the waterfall "
        "is read as strictly nested (integrity > eligible > dp_eligible). "
        "`accounting_closes` is the only `eligible` condition with no DP gate "
        "of its own -- suspect episode_universe letting a discontinuous "
        "episode through.")
    step(d[d.dp_eligible], "dp_eligible", economic, gate=True)
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

    def anchor_mask(frame):
        # its OWN band (ref_rate_anchor_band), wider than
        # episodes.is_anchor_row's half-tier on purpose: a demand-rate
        # feature wants more hours
        return ((frame.total_discount - frame.d_ref).abs() <= band + 1e-9) \
            & (frame.starting_inventory >= 1)

    d = d.copy()
    anchor = anchor_mask(d)
    day = (pd.DataFrame({
        "sku_id": d.sku_id, "fc": d.fc, "date": pd.to_datetime(d.date),
        "a_sold": d.units_sold.where(anchor, 0),
        "a_hours": anchor.astype(int)})
        .groupby(["sku_id", "fc", "date"], as_index=False).sum())

    def trailing_rate(frame, keys):
        """Trailing [t-W, t-1] anchor rate per key group (design 5.4): the
        W prior days, the day itself excluded. `closed="left"` is exactly
        that window; the default right-closed one is [t-W+1, t], and
        subtracting the day back out left W-1 prior days."""
        g = frame.sort_values(keys + ["date"]).set_index("date")
        grouped = g.groupby(keys)
        sold = grouped.a_sold.rolling(f"{window}D", closed="left").sum()
        hours = grouped.a_hours.rolling(f"{window}D", closed="left").sum()
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
    # the merge reset the index: a mask built on the caller's labels would
    # align by label, not position, and corrupt the feature on a gappy frame
    anchor = anchor_mask(d)

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
    opened on or before split.test_end (episode-scoped, kept whole)."""
    return episodes.window_slice(d, None, cfg["data"]["split"]["test_end"])




DP_INELIGIBLE = (
    ("cost_missing",
     "cost <= 0 -- a MISSING cost, not a free good: d_max would read 1.0 and "
     "scrap would read free"),
    ("non_priceable",
     "cost >= original_price: d_max <= 0, feasible_tiers is empty"),
    ("negative_window",
     "hours_remaining still < 0 after recovery -- true window length unknown; "
     "the DP horizon and extend_to_window read the counter"),
    ("window_too_long",
     "hours_remaining above data.max_window_hours (extend_to_window raises)"),
    ("outcome_unknown",
     "the episode never closed inside this data (no write-off sentinel). "
     "Gates `eligible` too: an unfinished episode is not a complete "
     "observation of anything. Kept in the integrity population"),
    ("final_hour_restock",
     "the LAST row sold more than it opened with: arrival and write-off are "
     "two unknowns with one equation, so scrap is NaN. Gates `eligible` too "
     "-- the censoring call cannot be made on an ambiguous final hour"),
)

# Reported, NOT gating (design 5.2).
BELOW_COST_HOURS = (
    "some hour's OFFERED price is under cost -- tested on "
    "original_price x (1 - d), never applied_price (zeroed on no-sale rows)")




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
            "FLAGGED, NOT DROPPED. Three conditions (episode_flow): identity "
            "holds, clean final hour, CLOSED -- what a frozen artifact needs. "
            "Populations are NESTED (integrity > eligible > dp_eligible); "
            "never add the rows' exclusions together."),
    }


def tag_dp_eligibility(d, cfg, flow=None, per_episode_cogs=None):
    """Flag (never drop) the episodes the DP cannot price; episode-scoped,
    since the DP plans the window as a unit (docs/learnings.md on what
    dropping cost). Must run AFTER re-segmentation so the ids it groups on
    are final and the chain tests cannot mistake a data gap for a restock.
    `flow` / `per_episode_cogs` are `episode_flow(d)` / `episode_cogs(d)`
    when the caller already holds them."""
    cap = cfg["data"]["max_window_hours"]
    # supply accounting attached once; `episode_supply` is the only correct
    # clearance denominator now that a window can gain stock
    if flow is None:
        flow = episodes.episode_flow(d)
    ep_cogs = (episode_cogs(d) if per_episode_cogs is None
               else per_episode_cogs)
    d = d.copy()
    for col, src in (("units_restocked", "arrived"),
                     ("units_shrink", "vanished"),
                     ("episode_supply", "supply"),
                     ("episode_scrap", "scrap"),
                     ("episode_clearance", "clearance")):
        d[col] = d.episode_id.map(flow[src]).astype(
            float if src == "clearance" else "int64")

    def cogs_of(hit):
        """COGS at risk behind an episode-scoped mask, off the one table."""
        return round(cogs_at_risk(d[hit], ep_cogs), 1)

    def hit_detail(hit, **extra):
        """Episodes, rows and COGS at risk behind one flag mask."""
        return {"episodes": int(d.loc[hit, "episode_id"].nunique()),
                "rows": int(hit.sum()), "cogs_at_risk": cogs_of(hit), **extra}
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
        # must be caught by the negation
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
        detail[name] = hit_detail(hit)
        reason = reason.mask(hit & reason.isna(), name)
    d["dp_ineligible_reason"] = reason
    d["dp_eligible"] = reason.isna()

    def still_eligible(hit):
        return int(d.loc[hit & d.dp_eligible, "episode_id"].nunique())

    # informational: kept IN dp_eligible, because the DP refusing a below-cost
    # anchor is the constraint working, not a reason to delete the episode
    below = d.episode_id.isin(
        d.loc[d.offered_price < d.cost - 1e-9, "episode_id"].unique())
    d["below_cost_hours"] = below
    detail["below_cost_hours"] = hit_detail(
        below, still_dp_eligible=still_eligible(below), why=BELOW_COST_HOURS)
    # Moved inventory is reported and gates nothing: the replay re-solves
    # hourly against actual stock, learning of arrivals at the next hour
    # exactly as production does.
    for name, col in (("restocked", "units_restocked"),
                      ("shrink", "units_shrink")):
        hit = d[col] > 0
        detail[name] = hit_detail(
            hit, units=int(d.loc[~d.episode_id.duplicated() & hit, col].sum()),
            still_dp_eligible=still_eligible(hit))

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
        "reading": ("corr near 0 + small losses = real SHRINK; corr well "
                    "above 0 = TIMING SKEW between feeds -- a join to fix "
                    "upstream, not shrink to chase."),
    }

    # also reported-only: an unclosed ending is a missing OUTCOME, and every
    # outcome consumer already handles it (NaN scrap, outcome_known, COMPLETED).
    at_edge, edge_detail = edge_truncated_episodes(d, flow=flow)
    edge = d.episode_id.isin(at_edge)
    d["edge_truncated"] = edge
    edge_detail.update(rows=int(edge.sum()), cogs_at_risk=cogs_of(edge),
                       still_dp_eligible=still_eligible(edge),
                       # False on a feed that never zeroes a close: every
                       # unclosed episode above is then the feed, not the
                       # extract boundary (design 12a)
                       write_off_convention_in_force=(
                           episodes.write_off_convention(flow)))
    detail["edge_truncated"] = edge_detail

    # THE EPISODE IDENTITY, checked not assumed: provable given continuity,
    # so a violation is a bug in episode_flow that would move every IL figure.
    violations = episodes.flow_identity_violations(d, flow=flow)
    detail["flow_identity"] = {
        "rule": "opening + restocked == sold + shrink + leftover_at_last_hour",
        "episodes_checked": int(d.episode_id.nunique()),
        "violations": int(len(violations)),
        "holds": not len(violations),
        "note": ("Guaranteed by chain continuity: a violation means the "
                 "supply arithmetic is broken, not the source."),
    }

    # Shrink/restock pairs are NOT netted; both counted in full, adjacency
    # reported for the business.
    paired = flow[(flow.arrived > 0) & (flow.vanished > 0)]
    detail["shrink_and_restock_together"] = {
        "episodes": int(len(paired)),
        "share_of_episodes": round(float(len(paired)) / max(len(flow), 1), 6),
        "units_arrived": int(paired.arrived.sum()),
        "units_shrunk": int(paired.vanished.sum()),
        "episodes_equal_in_and_out": int((paired.arrived == paired.vanished).sum()),
        "note": ("BOTH an arrival and a loss; never netted -- both counted "
                 "in full, the episode stays dp_eligible."),
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
            "cogs_at_risk": cogs_of(bad),
            "median_units_per_episode": float(ep.units.median()),
            "by_category": _by("category"),
            "by_month": _by("month"),
            "note": ("Stock left the shelf that no sale or write-off "
                     "accounts for; KEPT, shrink settles into scrap. "
                     "Concentrated in a month = incident; spread evenly = "
                     "standing feed property; few categories = the subset."),
        }

    detail["episodes_dp_eligible"] = int(
        d.loc[d.dp_eligible, "episode_id"].nunique())
    detail["episodes_dp_ineligible"] = int(
        d.loc[~d.dp_eligible, "episode_id"].nunique())
    detail["note"] = (
        "FLAGGED, NOT DROPPED. below_cost_hours and edge_truncated are "
        "reported only and stay dp_eligible. Artifact fits read 'eligible'; "
        "the DP, backtest, shadow and calibration gate read "
        "'dp_eligible'. Reasons are first-match in the order "
        + ", ".join(n for n, _ in DP_INELIGIBLE)
        + "; counts overlap, so they do not sum to episodes_dp_ineligible.")
    return d, detail


def population(d, cfg, which=None):
    """ONE definition of the three NESTED populations: integrity (survived
    the chain), eligible (identity holds + clean final hour + CLOSED -- what
    a frozen artifact needs), dp_eligible (eligible + solver requirements).
    None means "eligible" (the artifact-fit population); DP callers pass
    "dp_eligible" explicitly. `cfg` is not read: every caller passes it
    positionally, so the signature is kept."""
    which = which or "eligible"
    if which == "integrity":
        return d
    if which not in ("eligible", "dp_eligible"):
        raise ValueError(f"unknown population {which!r}")
    flag = "episode_eligible" if which == "eligible" else "dp_eligible"
    # REFUSE rather than fall back to the whole frame
    if flag not in d:
        raise ValueError(
            f"population({which!r}) needs the {flag!r} column and this frame "
            "has none -- re-run fit.prepare_data; a frame without the "
            "eligibility flags is the integrity population, not this one")
    return d[d[flag]]


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




def write_manifest(path, cfg, waterfall=None):
    """The waterfall is the artifact, not the console: every stage's rows,
    episodes, COGS at risk and detail dict are persisted here."""
    stages = [{"stage": s[0], "rows": int(s[1]), "episodes": int(s[2]),
               "cogs_at_risk": s[3] if len(s) > 3 else None,
               "detail": s[4] if len(s) > 4 else None}
              for s in (waterfall or [])]
    write_json(path, stamp({
        "episode_rule": EPISODE_RULE,
        "split": cfg["data"]["split"],
        "holdout": cfg["data"].get("holdout"),
        "exclusion_window": cfg["data"]["exclusion_window"],
        "config_version": cfg["meta"]["config_version"],
        "waterfall": stages,
    }, cfg, None, "fit.prepare_data"))


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

    for stage in wf:                       # some carry a 5th detail dict
        label, rows, eps = stage[0], stage[1], stage[2]
        print(f"  {label:32s} rows {rows:>10,}  episodes {eps:>9,}")
    print(f"wrote {args.out} and {args.manifest}")


if __name__ == "__main__":
    main()
