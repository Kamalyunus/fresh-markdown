"""daily.features -- the day's feature table for Lane B's batches (design 5.10).

Every morning, after the feed lands: yesterday's hourly rows join the
rolling feed history (`features.history_path`, the source schema, the last
`features.history_days` days), the one chain prepares it, and the two
demand-rate features every episode opening TODAY reads -- the trailing
reference sales rate and the prior episode's -- are written once per
SKU x FC (plus the SKU-pooled fallback row) to
`features/<today>.parquet` (engine.state.ref_rate_table). A batch then
joins on SKU and FC instead of reading and re-rolling the trailing table
every hour: both features read strictly before the opening date, so the
table's row IS the batch's number.

Run: python3 -m daily.features --feed <yesterday's hourly parquet> [--as-of YYYY-MM-DD]
(ops.advance --feed runs it after ingest)
"""

import datetime as dt
import os

import pandas as pd

from common.cli import make_parser
from common.config import load_config
from common.io import write_json
from common.paths import RAW
from engine.state import load_history, ref_rate_table
from events.pairs import iso_day

# the source schema's hour key (fit.prepare_data.SOURCE_TO_CANONICAL maps
# these to the prepared names): a day fed twice keeps its last row
SOURCE_HOUR_KEY = ("skuseq", "fc", "date", "hour")


def rolling_history(cfg, feed_path=None, today=None):
    """The rolling feed history after `feed_path`'s rows join it: seeded
    from the raw extract (common.paths.RAW) when no rolling file exists
    yet, deduplicated on the source hour key (the last row of a re-fed
    hour wins), trimmed to `features.history_days` days ending at the
    latest row, written back to `features.history_path`. Returns the
    frame written."""
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


def build(cfg, feed_path=None, as_of=None, out=None):
    """One morning: the rolling history updated with `feed_path`, prepared
    through the one chain (engine.state.load_history), the table for
    episodes opening on `as_of` (today) written to `out`
    (features/<as_of>.parquet). Returns the morning's report."""
    fc = cfg["features"]
    raw = rolling_history(cfg, feed_path)
    if as_of is None:
        # the table is for the day AFTER the feed being ingested -- the
        # feed's own trading day plus one, never the host's clock: a cron
        # running after midnight in one zone and before it in another read
        # "today" one day off and wrote a table the batches counted stale
        as_of = (str((pd.Timestamp(raw.date.max()) + pd.Timedelta(days=1)).date())
                 if feed_path and len(raw)
                 else dt.datetime.now(dt.timezone.utc).date())
    as_of = iso_day(as_of)
    hist = load_history(fc["history_path"], cfg)
    table = ref_rate_table(hist, as_of, cfg)
    out = out or os.path.join(fc["table_dir"], f"{as_of}.parquet")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    table.to_parquet(out, index=False)
    pooled = int((table["fc"] == "*").sum()) if len(table) else 0
    return {
        "as_of": as_of, "out": out,
        "rows": int(len(table)), "sku_fc_rows": int(len(table)) - pooled,
        "pooled_rows": pooled,
        "history_rows": int(len(raw)),
        "history_from": str(pd.to_datetime(raw["date"]).min().date()) if len(raw) else None,
        "history_through": str(pd.to_datetime(raw["date"]).max().date()) if len(raw) else None,
        "prepared_rows": int(len(hist)),
        "feed": feed_path,
    }


def main(argv=None):
    ap = make_parser("daily.features")
    ap.add_argument("--out", default=None,
                    help="the table's path; default features/<as_of>.parquet")
    ap.add_argument("--feed", default=None,
                    help="yesterday's hourly feed parquet (source schema); "
                         "omitted, the rolling history stands as it is")
    ap.add_argument("--as-of", default=None,
                    help="the day the table is for (default today): the "
                         "features every episode opening that day reads")
    ap.add_argument("--report", default=None, help="the morning's counts, JSON")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rep = build(cfg, feed_path=args.feed, as_of=args.as_of, out=args.out)
    if args.report:
        write_json(args.report, rep)
    print(f"feature table {rep['as_of']}: {rep['sku_fc_rows']:,} sku x fc rows + "
          f"{rep['pooled_rows']:,} pooled, from {rep['history_rows']:,} feed rows "
          f"({rep['history_from']}..{rep['history_through']}) -> {rep['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
