"""ops.check_inputs -- are engineering's three files what the chain needs?

Run before launch on samples of the three tables Lane B delivers, and
again whenever a producer changes: the top-of-hour SNAPSHOT
(ops.price_hour reads it), the daily hourly FEED (daily.ingest_outcomes
builds the outcomes from it; daily.features the feature table), and the
FAILED PUSHES table. Every check prints PASS, WARN or FAIL with the count
behind it and what to fix; the feed is also run through the one
preparation chain (fit.prepare_data.load_and_filter) so its waterfall --
what every stage dropped -- is read here, before a morning depends on it.
Exit 1 on any FAIL. The contract's checklist, as a script result.

Run: python3 -m ops.check_inputs [--snapshot <file>] [--feed <file>] [--failures <file>] [--report <json>]
"""

import argparse

import pandas as pd

from common.config import load_config
from common.io import read_rows, write_json
from common.windows import counter_step_detail, planning_horizon, window_signals
from daily.ingest_outcomes import load_failures
from fit import prepare_data
from fit.prepare_data import SOURCE_TO_CANONICAL

# the hourly feed's columns, in its own names (fit.prepare_data.SOURCE_TO_CANONICAL
# maps the renamed ones); a snapshot may leave the three closing columns null
FEED_COLUMNS = ("date", "hour", "skuseq", "fc", "inventory", "discount", "units_sold",
                "normal_asp", "final_price", "cogs_wo_vat", "ending_inventory",
                "flc_window", "category", "subcategory")
CLOSING_COLUMNS = ("units_sold", "final_price", "ending_inventory")
FAILURE_COLUMNS = (("skuseq", "sku_id"), ("fc",), ("date",), ("hour", "hour_of_day"), ("reason",))


def _read(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.DataFrame(read_rows(path))


class Checks:
    def __init__(self, name):
        self.name, self.rows = name, []

    def add(self, check, verdict, count, fix=""):
        self.rows.append({"file": self.name, "check": check, "verdict": verdict,
                          "count": count, "fix": fix if verdict != "PASS" else ""})

    def gate(self, check, bad, fix, warn=False):
        """FAIL (or WARN) when `bad` > 0, PASS otherwise."""
        self.add(check, ("WARN" if warn else "FAIL") if bad > 0 else "PASS", int(bad), fix)


def _common(df, c, cfg, snapshot):
    """The checks a snapshot and a feed share."""
    present = set(df.columns)
    need = [col for col in FEED_COLUMNS if col not in present
            and not (snapshot and col in CLOSING_COLUMNS)]
    c.gate("columns present (the feed's own names)", len(need),
           f"missing {need}: the file must carry the feed's columns")
    if need:
        return False
    c.gate("skuseq and fc never null", int(df.skuseq.isna().sum() + df.fc.isna().sum()),
           "a row without an id can be placed on no shelf")
    day = pd.to_datetime(df.date, errors="coerce")
    c.gate("date is a calendar day", int(day.isna().sum()), "a date that parses to no day")
    hour = pd.to_numeric(df.hour, errors="coerce")
    c.gate("hour in 0..23", int((~hour.between(0, 23)).sum()), "the clock hour of the row")
    disc = pd.to_numeric(df.discount, errors="coerce")
    frac = disc.dropna()
    c.add("discount is a PERCENT (25.0, not 0.25)",
          "FAIL" if len(frac) and frac.abs().max() <= 1.0 and (frac != 0).any() else "PASS",
          int((frac.abs() <= 1.0).sum()),
          "every value is at most 1: the column is a fraction; the chain divides by 100 once")
    cap = int(cfg["data"]["max_window_hours"])
    win = pd.to_numeric(df.flc_window, errors="coerce")
    stocked = pd.to_numeric(df.inventory, errors="coerce") > 0
    c.gate(f"flc_window means hours still to come: 0..{cap - 1} on stocked rows",
           int((stocked & ~win.between(0, planning_horizon(cap) - 2)).sum()),
           "a counter above data.max_window_hours is not hours (minutes? a sentinel?): "
           "the engine rejects the row; fix the source, do not clip")
    price = pd.to_numeric(df.normal_asp, errors="coerce")
    cost = pd.to_numeric(df.cogs_wo_vat, errors="coerce")
    c.gate("normal_asp > 0 and 0 <= cogs_wo_vat <= normal_asp",
           int(((price <= 0) | (cost < 0) | (cost > price)).sum()),
           "a row the engine refuses (no legal discount): a stale or missing cost", warn=True)
    dup = df.duplicated(subset=["skuseq", "fc", "date", "hour"]).sum()
    c.gate("one row per shelf-hour", int(dup),
           "two rows for one (skuseq, fc, date, hour): the outcome matches neither")
    return True


def check_snapshot(path, cfg):
    c = Checks("snapshot")
    df = _read(path)
    if not _common(df, c, cfg, snapshot=True):
        return c
    hours = sorted(set(zip(pd.to_datetime(df.date, errors="coerce").dt.strftime("%Y-%m-%d"),
                           pd.to_numeric(df.hour, errors="coerce"))))
    c.add("one or two consecutive hours (the opening, and the hour just closed)",
          "PASS" if len(hours) <= 2 else "WARN", len(hours),
          "a snapshot is the shelf now, plus at most the hour that just closed")
    c.gate("episode_id present and never null (the producer's)",
           int(df["episode_id"].isna().sum()) if "episode_id" in df.columns else len(df),
           "the producer assigns the episode id (ops.assign_episode_ids is the rule to run or port); "
           "a row without one is not priced")
    closed = [col for col in CLOSING_COLUMNS if col in df.columns]
    c.add("the closed hour's rows carry sales and ending inventory",
          "PASS" if len(closed) == len(CLOSING_COLUMNS) else "WARN", len(closed),
          "without them a sell-out close and a restock are read from the counter and "
          "the opening stock alone (ops.price_hour LIVE_RULE)")
    return c


def check_feed(path, cfg):
    c = Checks("feed")
    df = _read(path)
    if not _common(df, c, cfg, snapshot=False):
        return c
    sold = pd.to_numeric(df.units_sold, errors="coerce")
    c.gate("zero-sale hours are present", 0 if (sold == 0).any() else 1,
           "no row sold nothing: the feed drops idle hours, and every priced hour "
           "must land as an outcome")
    c.gate("units_sold, ending_inventory never null", int(sold.isna().sum()
           + pd.to_numeric(df.ending_inventory, errors="coerce").isna().sum()),
           "a closed hour without its sales or ending stock")
    # the one chain, on their rows: what every stage would drop
    try:
        prepared, waterfall = prepare_data.load_and_filter(path, cfg)
    except Exception as e:                                   # noqa: BLE001 -- reported, not raised
        c.add("the preparation chain runs on the feed", "FAIL", 0, f"{type(e).__name__}: {e}")
        return c
    c.add("the preparation chain runs on the feed", "PASS", int(len(prepared)))
    stages = [(s[0], int(s[1]), int(s[2])) for s in waterfall]
    prev = None
    for name, rows, eps in stages:
        if prev is not None and rows < prev:
            c.add(f"waterfall: {name}", "WARN", prev - rows,
                  "rows this stage dropped; read the stage's rule (design 5.2) -- a "
                  "large drop is a producer defect, not a filter working")
        prev = rows
    c.add("waterfall: rows in -> out", "PASS", int(len(prepared)),
          f"{stages[0][1] if stages else 0} rows in, {len(prepared)} out")
    canon = df.rename(columns=SOURCE_TO_CANONICAL).copy()
    canon["total_discount"] = pd.to_numeric(canon.total_discount, errors="coerce") / 100.0
    canon = canon.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    try:
        s = window_signals(canon)
        detail = counter_step_detail(canon)
        c.add("windows closed by the write-off zero", "PASS", int(s["prev_closed"].sum()),
              "hours whose previous hour ended at zero (the closure sentinel)")
        c.gate("counter never steps up without a restock or a close",
               detail.get("reset_new_window", 0),
               "the counter rose while the shelf neither closed nor restocked: two "
               "listings stitched under one counter, or a missing hour", warn=True)
        c.add("restocks that extended a window", "PASS", detail.get("restock_continued", 0))
        c.add("a close followed by a resumed listing", "PASS", detail.get("closed_then_resumed", 0))
    except Exception as e:                                   # noqa: BLE001
        c.add("counter signals", "WARN", 0, f"not read: {type(e).__name__}: {e}")
    return c


def check_failures(path, cfg):
    c = Checks("failures")
    df = _read(path)
    missing = [alts[0] for alts in FAILURE_COLUMNS if not any(a in df.columns for a in alts)]
    c.gate("columns: skuseq, fc, date, hour, reason", len(missing), f"missing {missing}")
    if missing:
        return c
    f = load_failures(path)
    c.gate("every row names one shelf-hour", int(getattr(f, "unkeyable", 0)),
           "a row with a null id, a bad date or a bad hour matches no decision")
    c.add("failed hours read", "PASS", len(f))
    return c


def run(cfg, snapshot=None, feed=None, failures=None):
    out = []
    if snapshot:
        out += check_snapshot(snapshot, cfg).rows
    if feed:
        out += check_feed(feed, cfg).rows
    if failures:
        out += check_failures(failures, cfg).rows
    return out


def render(rows):
    lines = [f"{'file':<10} {'verdict':<6} {'count':>8}  check"]
    for r in rows:
        lines.append(f"{r['file']:<10} {r['verdict']:<6} {r['count']:>8}  {r['check']}")
        if r["fix"]:
            lines.append(f"{'':<27}-> {r['fix']}")
    fails = sum(1 for r in rows if r["verdict"] == "FAIL")
    warns = sum(1 for r in rows if r["verdict"] == "WARN")
    lines.append(f"{fails} FAIL, {warns} WARN, {len(rows) - fails - warns} PASS")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ops.check_inputs")
    ap.add_argument("--snapshot", default=None, help="a top-of-hour snapshot (ops.price_hour's input)")
    ap.add_argument("--feed", default=None, help="one day's hourly feed (the daily lane's input)")
    ap.add_argument("--failures", default=None, help="a failed-pushes table")
    ap.add_argument("--report", default=None, help="the checks, JSON")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    if not (args.snapshot or args.feed or args.failures):
        ap.error("pass at least one of --snapshot, --feed, --failures")
    cfg = load_config(args.config)
    rows = run(cfg, args.snapshot, args.feed, args.failures)
    print(render(rows))
    if args.report:
        write_json(args.report, {"checks": rows})
    return 1 if any(r["verdict"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
