"""bootstrap.download_flc -- pull the raw hourly FLC extract from Redshift.

Step 0 of the pipeline: everything else in `AGENTS.md` starts from the parquet
this writes. One SELECT against `sb_scm.fresh_flc_detail`, aliased into the
column names `bootstrap.prepare_data` expects, saved to
`data/flc_raw.parquet`.

The alias list is a contract, not a convenience. `prepare_data.SOURCE_TO_PRD`
renames these columns and nothing renames them a second time, so a column
missing or misspelled here fails at load in step 1, several minutes and one
network round-trip later. `tests/test_download_flc.py` asserts the query
covers every column steps 1 and 2 read.

Credentials are read from the environment, never from this file and never
from config.yaml:

    REDSHIFT_HOST  REDSHIFT_PORT  REDSHIFT_DATABASE
    REDSHIFT_USERNAME  REDSHIFT_PASSWORD

`--env-file` (default `~/.env`) is loaded with python-dotenv first, so the
usual setup is a `.env` outside the repo. `.env` is gitignored; no hostname,
credential or connection string belongs in version control.

The known-bad demand period is excluded in SQL from
`data.exclusion_window` in config.yaml -- the same window
`prepare_data` removes, read from the same place rather than restated here.
The SQL filter is row-scoped and so severs an episode that straddles either
edge; step 1's removal is episode-scoped and is the one that counts. The
SQL copy exists only to avoid pulling 40 days of rows that get dropped.

Usage:
    python3 -m bootstrap.download_flc                       # last 120 days
    python3 -m bootstrap.download_flc --days 180
    python3 -m bootstrap.download_flc --start-date 2026-03-01 \
        --end-date 2026-08-03 --out data/flc_raw.parquet
"""

import argparse
import os
from datetime import date, timedelta

import pandas as pd

from common.config import load_config

DEFAULT_DAYS = 120
DATA_DIR = "data"
DEFAULT_OUT_PARQUET = os.path.join(DATA_DIR, "flc_raw.parquet")
SOURCE_TABLE = "sb_scm.fresh_flc_detail"

# Every column steps 1 and 2 read, under the names they read them by:
# the keys of prepare_data.SOURCE_TO_PRD plus the ones used unrenamed.
# Kept here as data so the test can compare the two lists directly.
REQUIRED_COLUMNS = (
    "date", "hour", "skuseq", "fc", "inventory", "units_sold",
    "ending_inventory", "discount", "normal_asp", "final_price",
    "cogs_wo_vat", "flc_window", "category", "subcategory",
)


def get_conn(env_file=None):
    """Connect using REDSHIFT_* from the environment.

    psycopg2 and python-dotenv are imported here rather than at module scope
    so `build_query` stays importable -- and testable -- on a machine with no
    database driver installed.
    """
    from dotenv import load_dotenv
    import psycopg2

    load_dotenv(env_file or os.path.expanduser("~/.env"))

    missing = [k for k in ("REDSHIFT_HOST", "REDSHIFT_DATABASE",
                           "REDSHIFT_USERNAME", "REDSHIFT_PASSWORD")
               if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "missing Redshift credentials in the environment: "
            + ", ".join(missing)
            + f" (looked in {env_file or '~/.env'}; see the module docstring)")

    return psycopg2.connect(
        host=os.environ["REDSHIFT_HOST"],
        port=int(os.environ.get("REDSHIFT_PORT", 5439)),
        dbname=os.environ["REDSHIFT_DATABASE"],
        user=os.environ["REDSHIFT_USERNAME"],
        password=os.environ["REDSHIFT_PASSWORD"],
        connect_timeout=30,
    )


def _iso(value, label):
    """Reject anything that is not a plain ISO date before it reaches SQL."""
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError:
        raise SystemExit(f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}")


def build_query(start_date, end_date, exclude_start=None, exclude_end=None):
    """The extract, with source columns aliased to their step-1 names.

    Dates are validated as ISO dates and interpolated -- Redshift does not
    accept a bound parameter everywhere a literal is legal, and a value that
    has survived `date.fromisoformat` cannot carry SQL.
    """
    start_date = _iso(start_date, "--start-date")
    end_date = _iso(end_date, "--end-date")

    exclusion = ""
    if exclude_start and exclude_end:
        exclusion = (
            f"\n      AND NOT (date BETWEEN '{_iso(exclude_start, 'exclusion start')}'"
            f" AND '{_iso(exclude_end, 'exclusion end')}')")

    return f"""
    SELECT
        date,
        hour,
        sku                AS skuseq,
        fc,
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
    WHERE date BETWEEN '{start_date}' AND '{end_date}'{exclusion}
    ORDER BY skuseq, fc, date, hour
    """


def summarise(df):
    """What to look at before spending an hour on the rest of the pipeline."""
    lines = [
        f"rows        {len(df):,}",
        f"skus        {df.skuseq.nunique():,}",
        f"fcs         {df.fc.nunique():,}",
        f"dates       {df.date.min()} -> {df.date.max()} "
        f"({df.date.nunique():,} distinct)",
        f"units_sold  mean {df.units_sold.mean():.3f}, "
        f"nonzero {(df.units_sold > 0).mean():.1%} of rows",
    ]
    # A null here is not a nuisance -- prepare_data drops the whole episode
    # for a null category or a zero base price, so a high share means the
    # population is about to shrink for a reason that has nothing to do with
    # the economics.
    for col in ("normal_asp", "cogs_wo_vat", "category", "subcategory"):
        lines.append(f"null {col:<11} {df[col].isna().mean():.2%}")
    return "\n".join("  " + line for line in lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"days back from today (default {DEFAULT_DAYS}); "
                         "ignored when --start-date is given")
    ap.add_argument("--start-date", help="ISO date; overrides --days")
    ap.add_argument("--end-date", help="ISO date; defaults to yesterday")
    ap.add_argument("--out", default=DEFAULT_OUT_PARQUET)
    ap.add_argument("--env-file", help="dotenv file with REDSHIFT_* "
                                       "(default ~/.env)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-exclude", action="store_true",
                    help="do not apply data.exclusion_window in SQL; step 1 "
                         "removes it either way, episode-scoped")
    args = ap.parse_args()

    end = date.fromisoformat(args.end_date) if args.end_date \
        else date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start_date) if args.start_date \
        else end - timedelta(days=args.days - 1)
    if start > end:
        raise SystemExit(f"empty range: {start} > {end}")

    excl = {} if args.no_exclude \
        else load_config(args.config)["data"]["exclusion_window"]

    query = build_query(start, end, excl.get("start"), excl.get("end"))
    print(f"== {SOURCE_TABLE}: {start} -> {end}"
          + (f", excluding {excl['start']} -> {excl['end']}" if excl else "")
          + " ==")

    conn = get_conn(args.env_file)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    if df.empty:
        raise SystemExit("no rows returned -- check the date range and table")

    print(summarise(df))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit("extract is missing columns step 1 requires: "
                         + ", ".join(missing))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}\nnext: scripts/run_bootstrap.sh {args.out}")


if __name__ == "__main__":
    main()
