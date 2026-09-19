"""ops.check_inputs -- are engineering's three files what the chain needs,
and is the hour's response what the chain promised?

Run before launch on samples of the three tables Lane B delivers, and
again whenever a producer changes: the top-of-hour SNAPSHOT
(ops.price_hour reads it), the daily hourly FEED (daily.ingest_outcomes
builds the outcomes from it; daily.features the feature table), and the
FAILED PUSHES table. `--response` checks the other direction: the hour's
RESPONSE (ops.price_hour --out) on its own shape, and against the
snapshot it answered when that rides along -- one row per shelf-hour of
the priced hour, the id spelt from its own row, the price the percent
makes, no price shallower than the one in force except on an entry.
Every check prints PASS, WARN or FAIL with the count behind it and what
to fix; the feed is also run through the one preparation chain
(fit.prepare_data.load_and_filter) so its waterfall -- what every stage
dropped -- is read here, before a morning depends on it. Exit 1 on any
FAIL. The contract's checklist, as a script result.

Run: python3 -m ops.check_inputs [--snapshot <file>] [--feed <file>] [--failures <file>]
                                 [--response <file>] [--report <json>]
"""

import argparse

import pandas as pd

from common.config import load_config
from common.io import read_rows, write_json
from common.windows import (counter_step_detail, planning_horizon, window_signals,
                            window_starts)
from daily.failures import load_failures
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
    """The file as a frame -- dtypes matter to the checks below, so parquet
    and CSV are read natively and JSONL through common.io.read_rows."""
    if path.endswith((".parquet", ".csv")):
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
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
        _check_feed_episode_ids(canon, c)
    except Exception as e:                                   # noqa: BLE001
        c.add("counter signals", "WARN", 0, f"not read: {type(e).__name__}: {e}")
    return c


def _check_feed_episode_ids(canon, c):
    """The producers' ids in the nightly feed, read against EPISODE_RULE over
    the WHOLE day -- the strong form of the hourly disagreement count. The
    comparison is on the window BOUNDARIES, never the spelling: their id
    scheme is theirs, and two ids agree when they group the same rows."""
    if "episode_id" not in canon.columns:
        c.add("episode_id carried in the feed", "WARN", len(canon),
              "the nightly feed does not carry the episode id the snapshot does; "
              "carrying it makes a window a plain group-by for everyone and lets "
              "this check read a whole day at once instead of an hour at a time")
        return
    ids = canon.episode_id
    c.gate("episode_id never null (the feed's copy of the producers' id)",
           int(ids.isna().sum()),
           "a feed row with no episode id: the id step did not run for that hour")
    # the boundary compare runs on the rows that CARRY an id: a null is
    # its own FAIL above, not the token "nan" grouped as a window, and it
    # must not break the row after it either -- the id is carried across
    # it (ffill within the shelf) so the compare reads the last id seen
    shelf = [canon.sku_id, canon.fc]
    theirs = ids.where(ids.notna()).astype(object).groupby(shelf).ffill()
    opens_theirs = theirs.ne(theirs.groupby(shelf).shift())
    has_id = ids.notna().to_numpy()
    c.gate("the producers' ids group the same windows EPISODE_RULE derives",
           int(((opens_theirs != window_starts(canon)) & has_id).sum()),
           "rows where their id opens a window and the rule does not, or the "
           "reverse: the live ids and the history's derived ids would disagree. "
           "Run ops.assign_episode_ids over the same day and diff")


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


def _keys(df):
    """Each row's hour key (events.pairs.hour_key) or None where the row
    names no shelf-hour, in row order."""
    from events.pairs import hour_key
    out = []
    for r in df.itertuples(index=False):
        try:
            out.append(hour_key(r.skuseq, r.fc, r.date, r.hour))
        except (KeyError, TypeError, ValueError):
            out.append(None)
    return out


def check_response(path, cfg, snapshot=None):
    """The hour's response (ops.price_hour --out, RESPONSE_COLS) on its own
    shape and, with the snapshot it answered, against that snapshot's rows
    of the priced hour (its latest; a closed hour riding along is not
    answered)."""
    from events.pairs import decision_id_of
    from ops.price_hour import RESPONSE_COLS
    c = Checks("response")
    df = _read(path)
    missing = [col for col in RESPONSE_COLS if col not in df.columns]
    c.gate("columns: " + ", ".join(RESPONSE_COLS), len(missing), f"missing {missing}")
    if missing:
        return c
    keys = _keys(df)
    c.gate("every row names one shelf-hour", sum(k is None for k in keys),
           "a null skuseq, fc, date or hour matches no snapshot row")
    priced = df["decision_id"].notna()
    rejected = df["rejected"].notna() & (df["rejected"].astype(str).str.strip() != "")
    c.gate("every row is priced or rejected, never both or neither",
           int((priced == rejected).sum()),
           "a priced row carries decision_id and an empty `rejected`; a rejected row the reason and no id")
    p = df[priced & ~rejected]
    bad_id = sum(1 for k, r in zip([k for k, ok in zip(keys, (priced & ~rejected)) if ok],
                                   p.itertuples(index=False))
                 if k is None or str(r.decision_id) != decision_id_of(k))
    c.gate("decision_id is dec-<skuseq>|<fc>|<date>T<hh> of its own row", bad_id,
           "the id is the shelf-hour's, never a surrogate; a mismatch is an altered or re-keyed row")
    pct = pd.to_numeric(p["apply_discount_pct"], errors="coerce")
    price = pd.to_numeric(p["apply_price"], errors="coerce")
    c.gate("a priced row carries apply_discount_pct and apply_price",
           int(pct.isna().sum() + price.isna().sum()), "a price without its percent, or the reverse")
    c.gate("apply_discount_pct is a PERCENT in [0, 100)",
           int(((pct < 0) | (pct >= 100)).sum()),
           "15 means fifteen percent; a fraction (0.15) would be applied as 0.15%")
    step = float(cfg["pricing"]["tier_step"]) * 100.0
    off = ((pct / step) - (pct / step).round()).abs() > 1e-6
    c.gate(f"apply_discount_pct sits on the {step:g}-point tier grid", int(off.fillna(False).sum()),
           "the engine prices on the grid; an off-grid percent is an altered response")
    if rejected.any():
        by = df.loc[rejected, "rejected"].astype(str).value_counts()
        c.add("rejected rows: " + ", ".join(f"{k} x{v}" for k, v in by.items()),
              "PASS", int(rejected.sum()))
    else:
        c.add("rejected rows", "PASS", 0)
    if not snapshot:
        return c

    s = _read(snapshot)
    if s.empty or not {"date", "hour", "skuseq", "fc", "normal_asp", "discount"} <= set(s.columns):
        c.add("the snapshot rides along", "WARN", 0, "no rows or no feed columns to compare against")
        return c
    hours = list(zip(pd.to_datetime(s["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
                     pd.to_numeric(s["hour"], errors="coerce")))
    latest = max(h for h in hours if h[0] is not None and pd.notna(h[1]))
    opening = s[[h == latest for h in hours]]
    skeys = _keys(opening)
    resp_keys = {k for k in keys if k is not None}
    snap_keys = {k for k in skeys if k is not None}
    c.gate("one response row per snapshot row of the priced hour",
           len(resp_keys ^ snap_keys),
           "a shelf in one file and not the other; the response answers the snapshot's latest hour")
    asp = {k: float(v) for k, v in zip(skeys, opening["normal_asp"]) if k is not None and pd.notna(v)}
    in_force = {k: float(v) for k, v in zip(skeys, opening["discount"]) if k is not None and pd.notna(v)}
    pkeys = [k for k, ok in zip(keys, (priced & ~rejected)) if ok]
    wrong = sum(1 for k, d, pr in zip(pkeys, pct, price)
                if k in asp and pd.notna(d) and pd.notna(pr)
                and abs(pr - asp[k] * (1.0 - d / 100.0)) > 1e-6 * max(asp[k], 1.0))
    c.gate("apply_price = normal_asp x (1 - apply_discount_pct / 100)", wrong,
           "the price and the percent must make each other; apply the percent, the price is its check")
    shallower = sum(1 for k, d in zip(pkeys, pct)
                    if k in in_force and pd.notna(d) and d + 1e-9 < in_force[k])
    c.gate("no price shallower than the one in force (the snapshot's discount)", shallower,
           "expected only on an ENTRY (a new episode_id on the shelf); a continuing episode never "
           "rises on shoppers -- a count here on continuing shelves is a defect to send us", warn=True)
    return c


def run(cfg, snapshot=None, feed=None, failures=None, response=None):
    out = []
    if snapshot:
        out += check_snapshot(snapshot, cfg).rows
    if feed:
        out += check_feed(feed, cfg).rows
    if failures:
        out += check_failures(failures, cfg).rows
    if response:
        out += check_response(response, cfg, snapshot=snapshot).rows
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
    ap.add_argument("--response", default=None,
                    help="an hour's response (ops.price_hour --out); checked against "
                         "--snapshot when both are given")
    ap.add_argument("--report", default=None, help="the checks, JSON")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    if not (args.snapshot or args.feed or args.failures or args.response):
        ap.error("pass at least one of --snapshot, --feed, --failures, --response")
    cfg = load_config(args.config)
    rows = run(cfg, args.snapshot, args.feed, args.failures, args.response)
    print(render(rows))
    if args.report:
        write_json(args.report, {"checks": rows})
    return 1 if any(r["verdict"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
