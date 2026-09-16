"""tools.e2e_cycle -- one full integration cycle, end to end, in a workspace.

What engineering's Lane B does in production, run once against a
simulated shop so the whole loop can be seen before a line of their code
exists: the shelf at the top of the hour goes in, in the feed's own
schema with the episode ids the shop assigns as the producer would
(`ops.assign_episode_ids` is the rule; `ops.price_hour` reads the ids as
given and builds the requests), decisions come back with the price to apply and the
`decision_id`, the shop applies them and sells, the hourly feed lands in
the source schema, `daily.ingest_outcomes` builds the outcomes and names
them from the feed row (`feed-<sku>|<fc>|<date>T<hh>`), and
`daily.export_events` writes the paired tables engineering reads back.
Every write goes under `--dir` (a copy of the production state, sealed:
evaluate.pilot_shop.build_workspace); production is never touched.

The shop is the pilot simulator's (`evaluate.pilot_shop.PilotSim`, driven
hour by hour with `LaneBPricer`, which goes through `ops.price_batch`
instead of the engine directly): demand from evaluate.pilot_world at the
frozen model's level with an ASSUMED elasticity, NB noise at the agent's
own r, the feed row the source would write, a rejected state held at the
defined fallback. Episodes are DP-eligible hold-out templates re-dated
onto the day after the extract ends, all opening at `--opening-hour` so
one batch is one clock hour. Every number it prints is the simulated
shop's (AGENTS rule 19).

Run: python3 -m tools.e2e_cycle [--episodes N] [--hours H] [--dir sim/e2e]
"""

import argparse
import os

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import write_json
from engine.state import hour_grid
from evaluate.pilot_shop import LaneBPricer, PilotSim, build_workspace, ingest_and_pair
from evaluate.pilot_sim import load_sim_config
from evaluate.pilot_world import World

# the simulator's own settings file, beside config.yaml at the repo root:
# the shop's driving knobs (the feature history margin, the lane hour)
# live there and nowhere in code
SIM_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "pilot_sim.yaml")


def run(cfg, prepared, out_dir, episodes=20, hours=3, opening_hour=10, seed=0,
        workers=None, epsilon_true=-1.2, sim_config=SIM_CONFIG_PATH):
    """The cycle. Returns the report (also written to <out_dir>/e2e_report.json)."""
    prepared = prepared.copy()
    prepared["date"] = prepared.date.astype(str)
    launch = (pd.Timestamp(prepared.date.max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    c, config_path = build_workspace(cfg, out_dir, launch)
    # the world's level is the frozen model's own prediction (drift 1.0),
    # built on the production config as the simulator builds it
    world = World(cfg, prepared, epsilon_true, seed=seed)
    pricer = LaneBPricer(out_dir, workers=workers, seed=seed)
    sim = PilotSim(c, world, out_dir, config_path, days=1, episodes_per_day=2 * episodes,
                   sim_settings=load_sim_config(sim_config), seed=seed,
                   prepared=prepared, workers=workers, pricer=pricer)

    # one episode per SKU x FC, so no hour holds two states for one shelf;
    # every one opens at the same hour, pilot arm only, no twin
    rng = np.random.default_rng([int(seed), 7])
    chosen, keys = [], set()
    for i in rng.permutation(len(world.templates)):
        t = world.templates[i]
        if (t["sku_id"], t["fc"]) in keys:
            continue
        keys.add((t["sku_id"], t["fc"]))
        chosen.append(dict(t, opening_hour=opening_hour))
        if len(chosen) == episodes:
            break
    sim.open_templates(0, launch, [("pilot", t, None) for t in chosen])
    sim.run_hours(0, launch, hour_grid(launch, opening_hour, hours))

    feed_path = os.path.join(out_dir, "feed", f"{launch}.parquet")
    feed = sim.write_feed(sorted(sim.feed_by_day), feed_path)
    # the daily lane's outcome side, on the feed the shop wrote
    paired = ingest_and_pair(sim.store, feed_path, out_dir)
    report = {
        "workspace": out_dir, "config_path": config_path, "launch_date": launch,
        "episodes_opened": len(chosen), "hours": hours, "opening_hour": opening_hour,
        "epsilon_true": epsilon_true, "seed": seed,
        "batches": pricer.batches,
        "features": pricer.tables,
        "decisions": paired["decisions"],
        # a rejected state is held at the shop's fallback and priced again
        # next hour, as the simulator does: counted by reason, never dropped
        "rejected": sim.rejected,
        "feed_path": feed_path, "feed_rows": int(len(feed)),
        **{k: paired[k] for k in ("ingest", "exports", "pairs", "price_mismatches",
                                  "outcome_ids_follow_the_formula", "paired_sample")},
    }
    write_json(os.path.join(out_dir, "e2e_report.json"), report)
    return report


def _print(rep):
    print(f"workspace {rep['workspace']}  launch {rep['launch_date']}  "
          f"episodes {rep['episodes_opened']}  hours {len(rep['batches'])}")
    for b in rep["batches"]:
        print(f"  {b['hour']}: {b['shelves']} shelves -> {b['decisions']} priced "
              f"({b['episodes_new']} new, {b['episodes_continued']} continued, "
              f"{b['explored']} explored), {b['rejected']} rejected   {b['decisions_path']}")
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
    ap.add_argument("--sim-config", default=SIM_CONFIG_PATH,
                    help="the simulator's settings (the shop's driving knobs)")
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
              workers=args.workers, epsilon_true=args.epsilon_true,
              sim_config=args.sim_config)
    _print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
