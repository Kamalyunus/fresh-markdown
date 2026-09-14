"""tools.e2e_cycle -- one full integration cycle, end to end, in a workspace.

What engineering's Lane B does in production, run once against a
simulated shop so the whole loop can be seen before a line of their code
exists: price requests in the contract's 12 fields go in per hour
(`ops.price_batch`), decisions come back with the price to apply and the
`decision_id`, the shop applies them and sells, the hourly feed lands in
the source schema, `daily.ingest_outcomes` builds the outcomes and names
them from the feed row (`feed-<sku>|<fc>|<date>T<hh>`), and
`daily.export_events` writes the paired tables engineering reads back.
Every write goes under `--dir` (a copy of the production state, sealed:
evaluate.pilot_sim.build_workspace); production is never touched.

The shop is evaluate.pilot_world's: demand at the frozen model's level
with an ASSUMED elasticity, NB noise at the agent's own r. Episodes are
DP-eligible hold-out templates re-dated onto the day after the extract
ends, all opening at `--opening-hour` so one batch is one clock hour.
Every number it prints is the simulated shop's (AGENTS rule 19).

Run: python3 -m tools.e2e_cycle [--episodes N] [--hours H] [--dir sim/e2e]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common.config import load_config
from common.io import read_json, write_json
from daily import export_events
from daily.ingest_outcomes import build_outcomes
from engine.posterior import PosteriorStore
from engine.state import HISTORY_COLS, hour_grid
from events.pairs import hour_key, match_pairs, outcome_id_of, price_matches
from events.store import EventStore
from evaluate.pilot_sim import build_workspace
from evaluate.pilot_world import FEED_SCHEMA, World
from fit.train_baseline import BaselineModel
from ops import price_batch

PAIR_COLS = ("decision_id", "outcome_id", "sku_id", "fc", "date", "hour_of_day",
             "applied_discount", "applied_price", "offered_price", "price_matches",
             "units_sold", "starting_inventory", "ending_inventory",
             "adjustment_reason", "is_exploration")


def _request(ep):
    tpl, t = ep["template"], ep["t"]
    date, hour = ep["grid"][t]
    return {"episode_id": ep["episode_id"], "sku_id": tpl["sku_id"], "fc": tpl["fc"],
            "category": tpl["category"], "subcategory": tpl["subcategory"],
            "date": date, "hour_of_day": hour,
            "hours_remaining": tpl["n_hours"] - t, "q": int(ep["q"]),
            "original_price": tpl["original_price"], "cost": tpl["cost"],
            "current_discount": ep["anchor"]}


def _write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run(cfg, prepared, out_dir, episodes=20, hours=3, opening_hour=10, seed=0,
        workers=None, epsilon_true=-1.2):
    """The cycle. Returns the report (also written to <out_dir>/e2e_report.json)."""
    prepared = prepared.copy()
    prepared["date"] = prepared.date.astype(str)
    launch = (pd.Timestamp(prepared.date.max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    c, config_path = build_workspace(cfg, out_dir, launch)
    world = World(c, prepared, epsilon_true, seed=seed)
    rng = np.random.default_rng([int(seed), 7])

    # one episode per SKU x FC, so no hour holds two states for one shelf
    chosen, keys = [], set()
    for i in rng.permutation(len(world.templates)):
        t = world.templates[i]
        if (t["sku_id"], t["fc"]) in keys:
            continue
        keys.add((t["sku_id"], t["fc"]))
        chosen.append(t)
        if len(chosen) == episodes:
            break
    open_eps = [{"episode_id": f"e2e|{t['sku_id']}|{t['fc']}|{launch}T{opening_hour:02d}",
                 "template": t, "grid": hour_grid(launch, opening_hour, t["n_hours"]),
                 "t": 0, "q": t["q0"], "anchor": None} for t in chosen]

    history = prepared[list(HISTORY_COLS)]
    store, model = EventStore(c), BaselineModel(c)
    posterior = PosteriorStore(c)
    r_lookup = read_json(c["dispersion"]["r_lookup_path"])

    batches, feed_rows, dropped = [], [], {}
    for h in range(hours):
        live = [ep for ep in open_eps if ep["t"] < ep["template"]["n_hours"] and ep["q"] > 0]
        if not live:
            break
        requests = [_request(ep) for ep in live]
        date, hour = live[0]["grid"][live[0]["t"]]
        tag = f"{date}T{hour:02d}"
        req_path = os.path.join(out_dir, "requests", f"{tag}.jsonl")
        _write_jsonl(requests, req_path)
        rows, events, rep = price_batch.run(c, requests, history, workers=workers,
                                            seed=seed, store=store, model=model,
                                            posterior=posterior, r_lookup=r_lookup)
        dec_path = os.path.join(out_dir, "decisions", f"{tag}.jsonl")
        price_batch.write_rows(rows, dec_path)
        batches.append({"hour": tag, "requests_path": req_path,
                        "decisions_path": dec_path, **rep})

        # engineering applies the price; the shop sells; the feed row lands
        by_ep = {r["episode_id"]: r for r in rows}
        by_id = {e["decision_id"]: e for e in events}
        for ep in live:
            row, tpl, t = by_ep[ep["episode_id"]], ep["template"], ep["t"]
            if row["rejected"]:
                dropped[ep["episode_id"]] = row["rejected"]
                ep["q"] = 0                              # the fallback: not priced
                continue
            shelf = float(row["applied_discount"])
            evt = by_id[row["decision_id"]]
            # the world's level IS the frozen model's prediction (drift 1.0)
            draw, _ = world.demand(tpl, evt["mu_ref_path"][0], shelf, 0)
            q = int(ep["q"])
            sold = min(draw, q)
            left = q - sold
            close = (t == tpl["n_hours"] - 1) or left == 0
            ending = 0 if close else left                  # write-off sentinel
            feed_rows.append(world.feed_row(tpl, date, hour, q, shelf, sold, ending,
                                            hours_remaining=tpl["n_hours"] - t))
            ep["anchor"], ep["q"], ep["t"] = shelf, left, t + 1

    feed = World.feed_frame(feed_rows)
    feed_path = os.path.join(out_dir, "feed", f"{launch}.parquet")
    pq.write_table(pa.Table.from_pandas(feed, schema=FEED_SCHEMA, preserve_index=False),
                   feed_path)

    # the daily lane's outcome side, on the feed the shop wrote
    decisions = store.load_decisions()
    outcomes, ingest = build_outcomes(decisions, pd.read_parquet(feed_path))
    ingest["emitted"] = int(sum(store.emit_outcome(o) for o in outcomes))
    written, _ = export_events.export(store, os.path.join(out_dir, "exports"))

    pairs = match_pairs(decisions, store.load_outcomes())
    table = []
    for d, o in pairs:
        table.append({
            "decision_id": d["decision_id"], "outcome_id": o["outcome_id"],
            "sku_id": d["sku_id"], "fc": d["fc"], "date": d["date"],
            "hour_of_day": d["hour_of_day"],
            "applied_discount": d["applied_discount"], "applied_price": d["applied_price"],
            "offered_price": o["applied_price"], "price_matches": price_matches(d, o),
            "units_sold": o["units_sold"], "starting_inventory": o["starting_inventory"],
            "ending_inventory": o["ending_inventory"],
            "adjustment_reason": o.get("adjustment_reason"),
            "is_exploration": d["is_exploration"]})
    formula_holds = all(
        o["outcome_id"] == outcome_id_of(hour_key(d["sku_id"], d["fc"], d["date"],
                                                  d["hour_of_day"]))
        for d, o in pairs)
    report = {
        "workspace": out_dir, "config_path": config_path, "launch_date": launch,
        "episodes_opened": len(open_eps), "hours": hours, "opening_hour": opening_hour,
        "epsilon_true": epsilon_true, "seed": seed,
        "batches": batches,
        "decisions": len(decisions),
        "episodes_dropped_on_rejection": dropped,
        "feed_path": feed_path, "feed_rows": int(len(feed)),
        "ingest": ingest,
        "exports": {k: {"path": p, "rows": n} for k, (p, n) in written.items()},
        "pairs": len(pairs),
        "price_mismatches": sum(1 for r in table if not r["price_matches"]),
        "outcome_ids_follow_the_formula": formula_holds,
        "paired_sample": table[:10],
    }
    write_json(os.path.join(out_dir, "e2e_report.json"), report)
    return report


def _print(rep):
    print(f"workspace {rep['workspace']}  launch {rep['launch_date']}  "
          f"episodes {rep['episodes_opened']}  hours {len(rep['batches'])}")
    for b in rep["batches"]:
        print(f"  {b['hour']}: {b['requests']} requests -> {b['decisions']} priced "
              f"({b['explored']} explored), {b['rejected']} rejected   {b['decisions_path']}")
    ing = rep["ingest"]
    print(f"feed {rep['feed_rows']} rows -> {rep['feed_path']}")
    print(f"ingest: {ing['outcomes_built']} outcomes built, {ing['emitted']} emitted, "
          f"{ing['decisions_without_feed_row']} without a feed row, "
          f"{ing['decisions_colliding_on_hour']} colliding")
    print("exports: " + ", ".join(f"{k} {v['rows']} rows" for k, v in rep["exports"].items()))
    print(f"pairs {rep['pairs']}  price mismatches {rep['price_mismatches']}  "
          f"outcome ids follow the formula: {rep['outcome_ids_follow_the_formula']}")
    if rep["paired_sample"]:
        cols = ("decision_id", "outcome_id", "applied_discount", "units_sold",
                "adjustment_reason", "is_exploration")
        print(pd.DataFrame(rep["paired_sample"])[list(cols)].to_string(index=False))
    print(f"-> {os.path.join(rep['workspace'], 'e2e_report.json')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.e2e_cycle")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default="data/prepared.parquet",
                    help="the prepared extract: templates and the feature history")
    ap.add_argument("--dir", default=os.path.join("sim", "e2e"),
                    help="the workspace; wiped and rebuilt each run")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--hours", type=int, default=3, help="hourly batches to price")
    ap.add_argument("--opening-hour", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--epsilon-true", type=float, default=-1.2)
    args = ap.parse_args(argv)
    # not strict: `data.launch_date` is the workspace's to set (the day after
    # the extract ends); build_workspace refuses every other null
    cfg = load_config(args.config)
    rep = run(cfg, pd.read_parquet(args.input), args.dir, episodes=args.episodes,
              hours=args.hours, opening_hour=args.opening_hour, seed=args.seed,
              workers=args.workers, epsilon_true=args.epsilon_true)
    _print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
