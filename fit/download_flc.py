"""fit.download_flc -- pull the raw hourly FLC extract from Redshift (step 0).

One SELECT against sb_scm.fresh_flc_detail, aliased to the names
fit.prepare_data expects (the alias list is a contract --
tests/test_download_flc.py). REDSHIFT_* credentials come from the environment
/ --env-file only, never config or this file. The SQL exclusion only saves
transfer and skips the window's INTERIOR (both edge days stay): an episode
straddling an edge always has a row on the edge day, and step 1 removes it
whole from that row -- cut in SQL, its remnant would enter the prior's
entry-only fit mid-window. Step 1's episode-scoped removal is the one that counts.
Run: python3 -m fit.download_flc [--days N | --start-date A --end-date B]
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

# Every column steps 1 and 2 read, under the names they read them by; kept as
# data so the test can compare the two lists directly.
REQUIRED_COLUMNS = (
    "date", "hour", "skuseq", "fc", "inventory", "units_sold",
    "ending_inventory", "discount", "normal_asp", "final_price",
    "cogs_wo_vat", "flc_window", "category", "subcategory",
)


def get_conn(env_file=None):
    """Connect using REDSHIFT_* from the environment. psycopg2/dotenv are
    imported here so build_query stays importable with no driver installed."""
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
    """The extract, source columns aliased to their step-1 names. Dates are
    validated as ISO and interpolated -- a value that survived
    date.fromisoformat cannot carry SQL."""
    start_date = _iso(start_date, "--start-date")
    end_date = _iso(end_date, "--end-date")

    exclusion = ""
    if exclude_start and exclude_end:
        # the interior only: both edge days are pulled so step 1 sees every
        # episode that touches the window and drops it whole
        lo = date.fromisoformat(_iso(exclude_start, "exclusion start")) + timedelta(days=1)
        hi = date.fromisoformat(_iso(exclude_end, "exclusion end")) - timedelta(days=1)
        if lo <= hi:
            exclusion = f"\n      AND NOT (date BETWEEN '{lo}' AND '{hi}')"

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


def required_range(cfg):
    """The dates the config's windows need the extract to cover: from
    data.split.train_start through the hold-out's end (or split.test_end when
    no hold-out is configured) -- the range ops.advance passes."""
    data = cfg["data"]
    start = date.fromisoformat(str(data["split"]["train_start"]))
    end = date.fromisoformat(str((data.get("holdout") or {}).get("end")
                                 or data["split"]["test_end"]))
    return start, end


def coverage_gap(start, end, cfg):
    """Why a pull over [start, end] cannot feed the configured windows, or
    None when it covers them. The manual `--days` default is what this
    guards: a range that stops short leaves a split empty and fit_dispersion
    (or the prior's held-out scoring) fails later, about something else."""
    need_start, need_end = required_range(cfg)
    if start <= need_start and end >= need_end:
        return None
    return (f"the pull {start} -> {end} does not cover the range the config's "
            f"windows need, {need_start} -> {need_end} (data.split.train_start "
            "through data.holdout.end or split.test_end). Pass --start-date / "
            "--end-date over that range, or run `python3 -m ops.advance`, "
            "which sizes the pull from the config.")


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
    # prepare_data drops whole episodes on null category / zero base price --
    # a high share here means the population is about to shrink
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
                    help="pull the exclusion window too (its interior is "
                         "skipped in SQL by default); step 1 removes it "
                         "either way, episode-scoped")
    args = ap.parse_args()

    end = date.fromisoformat(args.end_date) if args.end_date \
        else date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start_date) if args.start_date \
        else end - timedelta(days=args.days - 1)
    if start > end:
        raise SystemExit(f"empty range: {start} > {end}")

    cfg = load_config(args.config)
    excl = {} if args.no_exclude else cfg["data"]["exclusion_window"]

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
    print(f"\nwrote {args.out}")

    # the extract is on disk either way (it is still data); the exit code
    # says whether the chain can run on it
    gap = coverage_gap(start, end, cfg)
    if gap:
        raise SystemExit(f"EXTRACT TOO SHORT -- {gap}")
    print(f"next: python3 -m ops.bootstrap_loop --input {args.out}")


if __name__ == "__main__":
    main()
