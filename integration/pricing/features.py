"""The morning command: yesterday's feed in, the day's feature table out.

The rolling feed history (the extract on the first morning, then the last
`features.history_days` days of feed) is prepared the way the training
extract was -- the window rule names the episodes, and every window a
defect sits in is dropped whole -- and the two demand-rate features every
episode opening TODAY reads (the trailing 30-day reference sales rate and
the previous episode's) are written once per SKU x FC, plus one pooled row
per SKU. Both read strictly before the opening day, so the table's row IS
the hour's number.

Run: python3 build_features.py --feed feed/<yesterday>.parquet [--as-of YYYY-MM-DD]
"""
import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd

from pricing.config import load_config, reference_discount
from pricing.feed import SOURCE_TO_CANONICAL, write_json
from pricing.keys import ident_series, iso_day

RAW = "data/flc_raw.parquet"
SOURCE_HOUR_KEY = ("skuseq", "fc", "date", "hour")
EPISODE_KEY = ("sku_id", "fc", "date", "hour_of_day")
QUANTITY_COLS = ("starting_inventory", "units_sold", "ending_inventory")
HISTORY_COLS = ("episode_id", "sku_id", "fc", "category", "date", "hour_of_day",
                "starting_inventory", "units_sold", "total_discount")
POOLED_FC = "*"
FEATURE_COLS = ("sku_id", "fc", "sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate", "as_of")


# ------------------------------------------------------- the rolling feed

def rolling_history(cfg, feed_path=None):
    """The rolling feed after `feed_path` joins it: seeded from the extract
    when no rolling file exists, deduplicated on the source hour key (the
    last row of a re-fed hour wins), trimmed to `history_days`, written back."""
    fc = cfg["features"]
    path, days = fc["history_path"], int(fc["history_days"])
    frames = []
    if os.path.exists(path):
        frames.append(pd.read_parquet(path))
    elif os.path.exists(RAW):
        frames.append(pd.read_parquet(RAW))
    if feed_path:
        frames.append(pd.read_parquet(feed_path))
    if not frames:
        raise SystemExit(f"no rolling feed at {path}, no extract at {RAW} and no --feed")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=list(SOURCE_HOUR_KEY), keep="last")
    day = pd.to_datetime(raw["date"])
    keep = day >= (day.max() - pd.Timedelta(days=days - 1))
    raw = raw[keep].sort_values(list(SOURCE_HOUR_KEY)).reset_index(drop=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw.to_parquet(path, index=False)
    return raw


# -------------------------------------------------- the window rule, applied

def _signals(df):
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    grp = [df.sku_id, df.fc]
    prev = {c: df[c].groupby(grp).shift() for c in QUANTITY_COLS}
    hr_diff = df.hours_remaining.groupby(grp).diff()
    prev_restock = prev["ending_inventory"] > prev["starting_inventory"] - prev["units_sold"]
    return {"dt_h": ts.groupby(grp).diff().dt.total_seconds() / 3600.0,
            "hr_diff": hr_diff,
            "prev_closed": prev["ending_inventory"].eq(0),
            "prev_restock": prev_restock,
            "counter_ok": hr_diff.eq(-1.0) | (hr_diff.gt(-1.0) & prev_restock)}


def _episode_ids(df):
    """`sku|fc|<first hour of the window>`: a row opens a window when the
    clock did not step one hour, the previous hour closed the shelf, or the
    counter did not step down by one (unless stock arrived)."""
    s = _signals(df)
    starts = s["dt_h"].ne(1.0) | s["prev_closed"] | ~s["counter_ok"]
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    start_ts = ts.where(starts).groupby([df.sku_id, df.fc]).ffill()
    return df.sku_id.astype(str) + "|" + df.fc.astype(str) + "|" + start_ts.dt.strftime("%Y-%m-%dT%H")


def _defective_windows(df, bad):
    """Every row of the source window a `bad` row sits in (a duplicate hour
    stays in its window; an unreadable counter step is taken on the clock)."""
    bad = pd.Series(np.asarray(bad, dtype=bool), index=df.index)
    if not bad.any():
        return pd.Series(False, index=df.index)
    s = _signals(df)
    dt_h, hr_diff = s["dt_h"], s["hr_diff"]
    gap_ok = (dt_h > 1.0) & (dt_h == -hr_diff)
    same = ((dt_h.eq(1.0) & (s["counter_ok"] | hr_diff.isna()))
            | dt_h.eq(0.0) | gap_ok) & ~s["prev_closed"]
    window = (~same.fillna(False) | dt_h.isna()).cumsum()
    return window.isin(set(window[bad]))


def _gap_split_ids(df):
    """Every fragment of a window a missing hour split in two."""
    ts = pd.to_datetime(df.date) + pd.to_timedelta(df.hour_of_day, unit="h")
    grp = [df.sku_id, df.fc]
    dt_h = ts.groupby(grp).diff().dt.total_seconds() / 3600.0
    hr_drop = -df.hours_remaining.groupby(grp).diff()
    gap = (dt_h > 1) & (dt_h == hr_drop)
    if not gap.any():
        return np.array([], dtype=object)
    starts = df.episode_id.ne(df.episode_id.groupby(grp).shift())
    window = df.episode_id.where(starts & ~gap).groupby(grp).ffill()
    per_window = df.groupby(window).episode_id.nunique()
    broken = per_window.index[per_window > 1]
    return df.loc[window.isin(broken), "episode_id"].unique()


def _continuity_breaks(d):
    nxt = d.groupby("episode_id")["starting_inventory"].shift(-1)
    return (nxt.notna() & (nxt != d.ending_inventory)).to_numpy()


def prepare(path, cfg):
    """The rolling feed prepared as the training extract was: every drop is
    a window or an episode, whole. Returns the history in HISTORY_COLS."""
    excl = cfg["data"].get("exclusion_window") or {}
    df = pd.read_parquet(path).rename(columns=SOURCE_TO_CANONICAL)
    df["total_discount"] = df["total_discount"] / 100.0
    for col in QUANTITY_COLS:
        df[col] = pd.to_numeric(df[col]).round()
    df = df.sort_values(list(EPISODE_KEY))
    df["episode_id"] = _episode_ids(df)
    readable = ~df[list(QUANTITY_COLS)].isna().any(axis=1)
    df = df[~df[list(EPISODE_KEY)].isna().any(axis=1)]           # no key, no window
    null_win = _defective_windows(df, df.hours_remaining.isna() | ~readable[df.index])
    df = df[~null_win]
    for col in QUANTITY_COLS:
        df[col] = df[col].astype("int64")
    dup = df.duplicated(subset=list(EPISODE_KEY), keep=False)
    df = df[~_defective_windows(df, dup)]
    df["episode_id"] = _episode_ids(df)
    d = df[~df.episode_id.isin(_gap_split_ids(df))]
    if excl.get("start"):
        ds = d.date.astype(str)
        inside = ds.ge(excl["start"]) & ds.le(excl["end"])
        d = d[~d.episode_id.isin(d.loc[inside, "episode_id"].unique())]
    bad = d.loc[~d.total_discount.between(0, 1), "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    neg = (d.starting_inventory < 0) | (d.units_sold < 0) | (d.ending_inventory < 0)
    d = d[~d.episode_id.isin(d.loc[neg, "episode_id"].unique())]
    d = d.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    discontinuous = _continuity_breaks(d)
    d = d[~d.episode_id.isin(d.loc[discontinuous, "episode_id"].unique())]
    bad = d.loc[d.category.isna() | d.subcategory.isna(), "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)].copy()
    d["original_price"] = (d.groupby("episode_id")["original_price"]
                           .transform(lambda s: s.replace(0, np.nan).ffill().bfill()))
    bad = d.loc[d.original_price.isna() | (d.original_price <= 0), "episode_id"].unique()
    d = d[~d.episode_id.isin(bad)]
    d = d.sort_values(["episode_id", "date", "hour_of_day"])
    hist = d[list(HISTORY_COLS)].copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    for col in ("sku_id", "fc"):
        hist[col] = ident_series(hist[col])
    return hist.reset_index(drop=True)


# ------------------------------------------------------- the two features

def add_ref_rate_features(d, cfg):
    """Point-in-time demand-rate features, from anchor hours only and
    lagged strictly before the episode's first date."""
    band = cfg["baseline_model"]["ref_rate_anchor_band"]
    window = cfg["baseline_model"]["ref_rate_window_days"]

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

def build(cfg, feed_path=None, as_of=None, out=None):
    fc = cfg["features"]
    raw = rolling_history(cfg, feed_path)
    if as_of is None:
        as_of = (str((pd.Timestamp(raw.date.max()) + pd.Timedelta(days=1)).date())
                 if feed_path and len(raw) else dt.datetime.now(dt.timezone.utc).date())
    as_of = iso_day(as_of)
    hist = prepare(fc["history_path"], cfg)
    table = ref_rate_table(hist, as_of, cfg)
    out = out or os.path.join(fc["table_dir"], f"{as_of}.parquet")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    table.to_parquet(out, index=False)
    pooled = int((table["fc"] == POOLED_FC).sum()) if len(table) else 0
    return {"as_of": as_of, "out": out, "rows": int(len(table)),
            "sku_fc_rows": int(len(table)) - pooled, "pooled_rows": pooled,
            "history_rows": int(len(raw)),
            "history_from": str(pd.to_datetime(raw["date"]).min().date()) if len(raw) else None,
            "history_through": str(pd.to_datetime(raw["date"]).max().date()) if len(raw) else None,
            "prepared_rows": int(len(hist)), "feed": feed_path}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="build_features.py")
    ap.add_argument("--feed", default=None,
                    help="yesterday's hourly feed parquet; omitted, the rolling history stands")
    ap.add_argument("--as-of", default=None,
                    help="the day the table is for (default: the feed's day plus one)")
    ap.add_argument("--out", default=None, help="default features/<as_of>.parquet")
    ap.add_argument("--report", default=None, help="the morning's counts, JSON")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    rep = build(load_config(args.config), feed_path=args.feed, as_of=args.as_of, out=args.out)
    if args.report:
        write_json(args.report, rep)
    print(f"feature table {rep['as_of']}: {rep['sku_fc_rows']:,} sku x fc rows + "
          f"{rep['pooled_rows']:,} pooled, from {rep['history_rows']:,} feed rows "
          f"({rep['history_from']}..{rep['history_through']}) -> {rep['out']}")
    return 0
