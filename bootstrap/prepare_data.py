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
EPISODE_RULE = ("episode_id = sku_id|fc|date, split into contiguous runs of "
                "selling hours; a gap of more than one hour starts a new "
                "segment, suffixed |s<n> for n > 0")


def load_and_filter(path, cfg=None):
    """Section 9.1 mapping + section 9.2 filter chain. Returns (df, waterfall).

    Filter order is deterministic and auditable; the waterfall records row and
    episode counts after every step. Intraday restocks are preserved (nothing
    here drops rows on inventory increase).
    """
    cfg = cfg or load_config()
    excl = cfg["data"]["exclusion_window"]

    df = pd.read_parquet(path).rename(columns=SOURCE_TO_PRD)

    # discount is PERCENT in source -> fraction, exactly once
    df["total_discount"] = df["total_discount"] / 100.0
    df["starting_inventory"] = df["starting_inventory"].round().astype("int64")
    df["ending_inventory"] = df["ending_inventory"].round().astype("int64")

    df["episode_id"] = (df.sku_id.astype(str) + "|" + df.fc.astype(str)
                        + "|" + df.date.astype(str))

    wf = [("raw", len(df), df.episode_id.nunique())]

    def step(d, label):
        wf.append((label, len(d), d.episode_id.nunique()))
        return d

    d = df[df.date.astype(str).lt(excl["start"]) | df.date.astype(str).gt(excl["end"])]
    d = step(d, "exclusion_window_removed")

    d = d[d.category.notna() & d.subcategory.notna()]
    d = step(d, "null_category_dropped")

    d = d.copy()
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

    # contiguous-hours episode construction (section 9.1 property 3)
    d = d.sort_values(["sku_id", "fc", "date", "hour_of_day"]).copy()
    gap = d.groupby("episode_id")["hour_of_day"].diff().fillna(1).gt(1)
    seg = gap.groupby(d["episode_id"]).cumsum().astype(int)
    d.loc[seg > 0, "episode_id"] = (
        d.loc[seg > 0, "episode_id"] + "|s" + seg[seg > 0].astype(str))
    wf.append(("contiguous_episodes_built", len(d), d.episode_id.nunique()))

    d["d_ref"] = d.category.map(lambda c: reference_discount(cfg, c))
    d["d_max"] = 1.0 - d.cost / d.original_price
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    d = add_ref_rate_features(d, cfg)
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

    day = day.sort_values(["sku_id", "fc", "date"])
    ep_rate = day.a_sold / day.a_hours.replace(0, np.nan)
    day["prior_episode_ref_sales_rate"] = (
        ep_rate.groupby([day.sku_id, day.fc]).shift(1))

    day["date"] = day.date.astype(str)
    feats = day[["sku_id", "fc", "date",
                 "sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate"]]
    d["_date_str"] = d.date.astype(str)
    d = (d.merge(feats.rename(columns={"date": "_date_str"}),
                 on=["sku_id", "fc", "_date_str"], how="left")
         .drop(columns=["_date_str"]))
    return d


def split_frames(d, cfg):
    """Date splits for baseline fitting only (config data.split)."""
    s = cfg["data"]["split"]
    ds = d.date.astype(str)
    return {
        "train": d[ds.ge(s["train_start"]) & ds.le(s["train_end"])],
        "calib": d[ds.ge(s["calib_start"]) & ds.le(s["calib_end"])],
        "test": d[ds.ge(s["test_start"]) & ds.le(s["test_end"])],
    }


def write_manifest(path, cfg, waterfall):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "episode_rule": EPISODE_RULE,
            "split": cfg["data"]["split"],
            "exclusion_window": cfg["data"]["exclusion_window"],
            "config_version": cfg["meta"]["config_version"],
            "data_quality_waterfall": [
                {"step": s, "rows": r, "episodes": e} for s, r, e in waterfall],
        }, f, indent=2)


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

    for s, r, e in wf:
        print(f"{s:32s} rows {r:>10,}  episodes {e:>9,}")
    print(f"wrote {args.out} and {args.manifest}")


if __name__ == "__main__":
    main()
