"""The morning command: the day's extract in, the day's feature table out.

The extract (`download_flc.py`: the trailing days of the hourly table,
the producers' `episode_id` on every row) is read as it is -- the ids are
theirs, no rule re-derives them -- with row-level hygiene only: a row
without its key, its id or its counts, a re-fed hour's earlier copy, a
discount outside 0..100, a negative count, a null category, a base price
that is not positive. The two demand-rate features every episode opening
TODAY reads (the trailing 30-day reference sales rate and the previous
episode's) are written once per SKU x FC, plus one pooled row per SKU.
Both read strictly before the opening day, so the table's row IS the
hour's number.

Run: python3 build_features.py [--extract data/flc.parquet] [--as-of YYYY-MM-DD]
"""
import argparse
import os

import numpy as np
import pandas as pd

from pricing.config import load_config, reference_discount
from pricing.feed import SOURCE_TO_CANONICAL, write_json
from pricing.keys import ident_series, iso_day

EXTRACT_PATH = os.path.join("data", "flc.parquet")     # where download_flc.py leaves the day's pull
HOUR_KEY = ("sku_id", "fc", "date", "hour_of_day")
QUANTITY_COLS = ("starting_inventory", "units_sold", "ending_inventory")
HISTORY_COLS = ("episode_id", "sku_id", "fc", "category", "date", "hour_of_day",
                "starting_inventory", "units_sold", "total_discount")
POOLED_FC = "*"
FEATURE_COLS = ("sku_id", "fc", "sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate", "as_of")

# The two features' definition, FIXED with the model: the trailing window
# (the "30d" in the feature's name) and the band around the reference
# discount an hour must sit in to count as an anchor hour. They equal the
# repository's baseline_model.ref_rate_window_days and ref_rate_anchor_band,
# which the model was trained on; the owner's check refuses a folder whose
# constants differ from the repository's config.
REF_RATE_WINDOW_DAYS = 30
REF_RATE_ANCHOR_BAND = 0.025


# ------------------------------------------------------------ the extract

def prepare(path):
    """The extract as history: the feed's names, the discount a fraction,
    the ids as given, and only rows that carry what the features read."""
    df = pd.read_parquet(path).rename(columns=SOURCE_TO_CANONICAL)
    if "episode_id" not in df.columns:
        raise SystemExit(f"{path} carries no episode_id column: the producers' id must be "
                         "in the extract (download_flc.py selects it)")
    df["total_discount"] = pd.to_numeric(df["total_discount"], errors="coerce") / 100.0
    for col in QUANTITY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")
    for col in ("sku_id", "fc", "episode_id"):
        df[col] = ident_series(df[col])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    keep = (df[list(HOUR_KEY)].notna().all(axis=1) & df.episode_id.notna()
            & df[list(QUANTITY_COLS)].notna().all(axis=1)
            & (df[list(QUANTITY_COLS)] >= 0).all(axis=1)
            & df.total_discount.between(0, 1)
            & df.category.notna() & (df.original_price > 0))
    df = df[keep].sort_values(list(HOUR_KEY))
    df = df.drop_duplicates(subset=list(HOUR_KEY), keep="last")     # a re-fed hour: the last row wins
    hist = df[list(HISTORY_COLS)].copy()
    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    hist["hour_of_day"] = hist["hour_of_day"].astype(int)
    for col in QUANTITY_COLS[:2]:
        hist[col] = hist[col].astype("int64")
    return hist.reset_index(drop=True)


# ------------------------------------------------------- the two features

def add_ref_rate_features(d, cfg):
    """Point-in-time demand-rate features, from anchor hours only and
    lagged strictly before the episode's first date."""
    band, window = REF_RATE_ANCHOR_BAND, REF_RATE_WINDOW_DAYS

    def anchor_mask(frame):
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
        g = frame.sort_values(keys + ["date"]).set_index("date")
        grouped = g.groupby(keys)
        sold = grouped.a_sold.rolling(f"{window}D", closed="left").sum()
        hours = grouped.a_hours.rolling(f"{window}D", closed="left").sum()
        return (sold / hours.replace(0, np.nan)).rename("rate").reset_index()

    day = day.merge(trailing_rate(day, ["sku_id", "fc"]).rename(columns={"rate": "rate_sku_fc"}),
                    on=["sku_id", "fc", "date"], how="left")
    sku_day = day.groupby(["sku_id", "date"], as_index=False)[["a_sold", "a_hours"]].sum()
    day = day.merge(trailing_rate(sku_day, ["sku_id"]).rename(columns={"rate": "rate_sku"}),
                    on=["sku_id", "date"], how="left")
    day["sku_ref_sales_rate_30d"] = day.rate_sku_fc.fillna(day.rate_sku)
    day["date"] = day.date.astype(str)
    feats = day[["sku_id", "fc", "date", "sku_ref_sales_rate_30d"]]
    d["_date_str"] = d.groupby("episode_id")["date"].transform("min").astype(str)
    d = d.merge(feats.rename(columns={"date": "_date_str"}), on=["sku_id", "fc", "_date_str"], how="left")
    anchor = anchor_mask(d)
    ep = (pd.DataFrame({
        "episode_id": d.episode_id, "sku_id": d.sku_id, "fc": d.fc, "start": d._date_str,
        "a_sold": d.units_sold.where(anchor, 0), "a_hours": anchor.astype(int)})
        .groupby(["episode_id", "sku_id", "fc", "start"], as_index=False).sum()
        .sort_values(["sku_id", "fc", "start", "episode_id"]))
    ep["rate"] = ep.a_sold / ep.a_hours.replace(0, np.nan)
    ep["prior_episode_ref_sales_rate"] = ep.rate.groupby([ep.sku_id, ep.fc]).shift(1)
    return (d.merge(ep[["episode_id", "prior_episode_ref_sales_rate"]], on="episode_id", how="left")
            .drop(columns=["_date_str"]))


def ref_rate_features(history, openings, cfg):
    """{episode_id: (rate_30d, prior_rate)} for `openings` (a frame of
    synthetic first hours with no sales) over the trailing history."""
    cols = list(HISTORY_COLS)
    stub = openings.assign(units_sold=0, total_discount=np.nan)[cols]
    frame = pd.concat([history[cols], stub], ignore_index=True)
    for col in ("sku_id", "fc", "episode_id"):
        frame[col] = ident_series(frame[col])
    frame = frame[frame.sku_id.notna() & frame.fc.notna()]
    d_ref = {c: reference_discount(cfg, c) for c in frame.category.unique()}
    frame["d_ref"] = frame.category.map(d_ref)
    feats = add_ref_rate_features(frame, cfg)
    mine = feats[feats.episode_id.isin(set(ident_series(stub.episode_id)))]
    return {r.episode_id: (float(r.sku_ref_sales_rate_30d), float(r.prior_episode_ref_sales_rate))
            for r in mine.itertuples()}


def _unknown(f):
    return all(isinstance(v, float) and np.isnan(v) for v in f)


def ref_rate_table(history, as_of, cfg):
    """The day's table: the two features an episode opening on `as_of`
    reads, for every (sku, fc) the history holds, plus one pooled row per
    SKU; a (sku, fc) that resolves to nothing is left out."""
    hist = history.copy()
    for col in ("sku_id", "fc"):
        hist[col] = ident_series(hist[col])
    hist = hist[hist.sku_id.notna() & hist.fc.notna()]
    as_of = iso_day(as_of)
    if not len(hist):
        return pd.DataFrame(columns=list(FEATURE_COLS))
    last = hist.sort_values("date").drop_duplicates(["sku_id", "fc"], keep="last")
    last_sku = hist.sort_values("date").drop_duplicates(["sku_id"], keep="last")
    openings = [{"episode_id": f"feat|{r.sku_id}|{r.fc}", "sku_id": r.sku_id, "fc": r.fc,
                 "category": r.category, "date": as_of, "hour_of_day": 0,
                 "starting_inventory": 1} for r in last.itertuples()]
    openings += [{"episode_id": f"feat|{r.sku_id}|{POOLED_FC}", "sku_id": r.sku_id,
                  "fc": POOLED_FC, "category": r.category, "date": as_of,
                  "hour_of_day": 0, "starting_inventory": 1} for r in last_sku.itertuples()]
    feats = ref_rate_features(hist, pd.DataFrame(openings), cfg)
    rows = [{"sku_id": o["sku_id"], "fc": o["fc"],
             "sku_ref_sales_rate_30d": feats[o["episode_id"]][0],
             "prior_episode_ref_sales_rate": feats[o["episode_id"]][1], "as_of": as_of}
            for o in openings if not _unknown(feats[o["episode_id"]])]
    return pd.DataFrame(rows, columns=list(FEATURE_COLS))


# ------------------------------------------------------------- the morning

def build(cfg, extract_path=EXTRACT_PATH, as_of=None, out=None):
    """The table for `as_of` (default: the day after the extract's last
    day) from the extract at `extract_path`; returns the report."""
    hist = prepare(extract_path)
    if as_of is None:
        if not len(hist):
            raise SystemExit(f"{extract_path} holds no usable row and no --as-of was given")
        as_of = str((pd.Timestamp(hist.date.max()) + pd.Timedelta(days=1)).date())
    as_of = iso_day(as_of)
    table = ref_rate_table(hist, as_of, cfg)
    out = out or os.path.join(cfg["features"]["table_dir"], f"{as_of}.parquet")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    table.to_parquet(out, index=False)
    pooled = int((table["fc"] == POOLED_FC).sum()) if len(table) else 0
    return {"as_of": as_of, "out": out, "rows": int(len(table)),
            "sku_fc_rows": int(len(table)) - pooled, "pooled_rows": pooled,
            "extract": extract_path, "history_rows": int(len(hist)),
            "history_from": str(hist.date.min()) if len(hist) else None,
            "history_through": str(hist.date.max()) if len(hist) else None}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="build_features.py")
    ap.add_argument("--extract", default=EXTRACT_PATH,
                    help=f"the day's extract from download_flc.py (default {EXTRACT_PATH})")
    ap.add_argument("--as-of", default=None,
                    help="the day the table is for (default: the extract's last day plus one)")
    ap.add_argument("--out", default=None, help="default features/<as_of>.parquet")
    ap.add_argument("--report", default=None, help="the morning's counts, JSON")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    rep = build(load_config(args.config), extract_path=args.extract, as_of=args.as_of, out=args.out)
    if args.report:
        write_json(args.report, rep)
    print(f"feature table {rep['as_of']}: {rep['sku_fc_rows']:,} sku x fc rows + "
          f"{rep['pooled_rows']:,} pooled, from {rep['history_rows']:,} extract rows "
          f"({rep['history_from']}..{rep['history_through']}) -> {rep['out']}")
    return 0
