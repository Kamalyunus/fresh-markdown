"""The day's extract: the trailing days of the hourly FLC table, pulled
from the warehouse into one parquet the morning table is built from.

One SELECT, the source columns aliased to the feed's names (the same
names the snapshot uses), the producers' `episode_id` carried through.
REDSHIFT_* credentials come from the environment or `~/.env` only, never
from the config or this file; the driver and dotenv are imported when the
pull runs, so the query stays importable without them."""
import os
from datetime import date, timedelta

import pandas as pd

from pricing.features import EXTRACT_PATH, REF_RATE_WINDOW_DAYS

SOURCE_TABLE = "sb_scm.fresh_flc_detail"
# the window the two features read plus a margin, so the prior episode of
# a shelf opening today is in the pull
DEFAULT_DAYS = REF_RATE_WINDOW_DAYS + 15
CREDENTIALS = ("REDSHIFT_HOST", "REDSHIFT_DATABASE", "REDSHIFT_USERNAME", "REDSHIFT_PASSWORD")


def _iso(value, label):
    """A plain ISO date, or a refusal: nothing else reaches the SQL."""
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError:
        raise SystemExit(f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}")


def build_query(start_date, end_date):
    """The extract over [start, end], both days included."""
    start_date, end_date = _iso(start_date, "start"), _iso(end_date, "end")
    return f"""
    SELECT
        date,
        hour,
        sku                AS skuseq,
        fc,
        episode_id,
        starting_inventory AS inventory,
        units_sold,
        ending_inventory,
        discount_pct       AS discount,
        base_price         AS normal_asp,
        final_price,
        cost               AS cogs_wo_vat,
        flc_window,
        UPPER(depth2)      AS category,
        UPPER(kan5)        AS subcategory
    FROM {SOURCE_TABLE}
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY skuseq, fc, date, hour
    """


def get_conn(env_file=None):
    """A connection from REDSHIFT_* in the environment (or `env_file`,
    default ~/.env); names the missing variables rather than the host."""
    from dotenv import load_dotenv
    import psycopg2

    load_dotenv(env_file or os.path.expanduser("~/.env"))
    missing = [k for k in CREDENTIALS if not os.environ.get(k)]
    if missing:
        raise RuntimeError("missing Redshift credentials in the environment: "
                           + ", ".join(missing) + f" (looked in {env_file or '~/.env'})")
    return psycopg2.connect(
        host=os.environ["REDSHIFT_HOST"], port=int(os.environ.get("REDSHIFT_PORT", 5439)),
        dbname=os.environ["REDSHIFT_DATABASE"], user=os.environ["REDSHIFT_USERNAME"],
        password=os.environ["REDSHIFT_PASSWORD"], connect_timeout=30)


def download(days=DEFAULT_DAYS, end_date=None, out=EXTRACT_PATH, env_file=None, conn=None):
    """Pull the trailing `days` through `end_date` (default yesterday) to
    `out`; returns the report."""
    end = date.fromisoformat(_iso(end_date, "--end-date")) if end_date \
        else date.today() - timedelta(days=1)
    start = end - timedelta(days=int(days) - 1)
    conn = conn or get_conn(env_file)
    try:
        frame = pd.read_sql(build_query(start, end), conn)
    finally:
        conn.close()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame.to_parquet(out, index=False)
    return {"out": out, "rows": int(len(frame)), "start": str(start), "end": str(end),
            "skus": int(frame.skuseq.nunique()) if len(frame) else 0,
            "days": int(frame.date.nunique()) if len(frame) else 0}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="download_flc.py")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"trailing days to pull (default {DEFAULT_DAYS})")
    ap.add_argument("--end-date", default=None, help="ISO date; default yesterday")
    ap.add_argument("--out", default=EXTRACT_PATH, help=f"default {EXTRACT_PATH}")
    ap.add_argument("--env-file", default=None, help="dotenv file with REDSHIFT_*; default ~/.env")
    args = ap.parse_args(argv)
    rep = download(days=args.days, end_date=args.end_date, out=args.out, env_file=args.env_file)
    print(f"extract {rep['start']}..{rep['end']}: {rep['rows']:,} rows, {rep['skus']:,} skus, "
          f"{rep['days']} days -> {rep['out']}")
    return 0
