"""evaluate.pilot_sim -- walk the system past launch_date against a demand world.

The weakest evidence before launch is what happens AFTER it: the hourly
engine, the outcome ingester, the tau walk, the monitor's stops, the
assurance checks, the weekly re-fit and the operator's --apply have each
been tested alone and rehearsed by shadow on history, never run together
for weeks on a shop that answers back. This simulator plays engineering
and the shop. Every day it opens episodes, prices every hour through the
REAL engine.decide against a REAL posterior and event store (copies under
sim/), applies the price to a shelf (with the faults asked for), writes the
hourly feed row the source would, and every morning runs the daily lane's
own functions -- ingest, tau walk, monitor, assurance, export, status,
--apply on the cadence -- plus Lane C's weekly re-fit and re-seal, in a
workspace that never touches a production artifact. Demand comes from
evaluate.pilot_world: the frozen model's level, an ASSUMED elasticity, NB
noise. Each fresh pick's twin runs under the other arm the day after it
closes (on a small pool a template is re-picked), so the economics read
like-for-like: per arm, and paired over the templates settled under both.

The report grades a fixed list of expectations (`EXPECTATIONS`) and reads
the posterior against the truth it was learning. Every number is about the
WORLD it simulated (rule 19): a PASS says the machinery does what it claims
on a shop with that elasticity, never that the shop has it.

This module is the run: its settings (pilot_sim.yaml at the repo root --
the world, the run, the faults, the grading, the paths -- apart from
config.yaml on purpose, which the sim rehearses unchanged; every flag
overrides its key for one run) and the console. The shop is
`evaluate.pilot_shop` (PilotSim), the grading `evaluate.pilot_grade`.
Run: python3 -m evaluate.pilot_sim [--days 21] [--epsilon-true -1.2]
        [--fault mismatch:0.03 --fault demand_shock:30:0.5 ...]
"""

import argparse
import json
import os

import pandas as pd
import yaml

from common.config import load_config
from common.io import write_json
from evaluate.pilot_world import FAULTS, World, parse_faults
# the shop and the grading moved to evaluate.pilot_shop / evaluate.pilot_grade;
# the names stay for callers
from evaluate.pilot_grade import (EVENT_GATE_FAULTS, EXPECTATIONS, GRADING_KEYS,  # noqa: F401
                                  expected_gate_failures, grade)
from evaluate.pilot_shop import (HIST_COLS, TRUTH_COLS, PilotSim, build_workspace,  # noqa: F401
                                 price_one, ref_rate_features, sim_config)


# --------------------------------------------------------------------- CLI

def _print(rep, out_path):
    w, e = rep["world"], rep["engine"]
    print(f"world: epsilon_true {w['epsilon_true']}  r_scale {w['r_scale']}  "
          f"episode shock sd {w['episode_shock_sd']}  drift/day {w['level_drift_per_day']}  "
          f"faults {w['faults'] or 'none'}")
    print(f"{w['days']} days from {w['launch_date']}, {w['episodes_per_day']} episodes/day "
          f"asked ({w['episodes_opened_per_day_mean']} opened, both arms) from "
          f"{w['templates']} templates, {w['workers']} worker(s)")
    print(f"engine: {e['decisions']:,} decisions, {e['forced']:,} forced "
          f"({e['forced_share']}), {e['rejected_total']} rejected, "
          f"{e['quarantined']} quarantined; tau {e['tau_at_launch']} -> {e['tau_now']}")
    for c, r in rep["learning"].items():
        eps = (f"{r['epsilon_true']:+.3f}" if r["epsilon_true"] is not None
               else "n/a (no simulated member)")
        print(f"  [{c}] eps_true {eps}  mean {r['launch_mean']:+.3f} -> "
              f"{r['mean']:+.3f}  std {r['launch_std']:.3f} -> {r['std']:.3f}  "
              f"(v{r['version']}, {r['n_obs']} outcomes)")
    econ = rep["economics"]
    arms = [(arm, econ[arm]) for arm in ("pilot", "legacy") if arm in econ]
    paired = econ.get("paired") or {}
    arms += [(f"{arm} (paired)", paired[arm]) for arm in ("pilot", "legacy") if arm in paired]
    for arm, x in arms:
        print(f"  {arm:16s} IL {x['il_absolute']:>12,.0f}  IL% {x['il_pct']}  "
              f"scrap_rate {x['scrap_rate']}  sell-through {x['sell_through']}  "
              f"mean discount {x['mean_discount']}  ({x['episodes']} episodes)")
    if paired:
        print(f"  paired over {paired.get('templates')} templates settled under both "
              f"arms ({paired.get('unpaired_templates')} unpaired)")
    for x in rep["expectations"]:
        print(f"  {x['verdict']:<12} {x['name']}")
    print(f"wrote {out_path}")


SIM_CONFIG = "pilot_sim.yaml"

# the sim config's sections and keys, with the CLI flag that overrides each
SIM_KEYS = {
    "run": ("days", "launch_date", "episodes_per_day", "seed", "templates_from",
            "workers"),
    "world": ("epsilon_true", "epsilon_true_map", "r_scale", "episode_shock_sd",
              "level_drift_per_day"),
    # the simulator's own grading and driving knobs (no flag: they shape
    # how a run is read, not what it rehearses) -- the grader's own list
    "grading": GRADING_KEYS,
    "paths": ("config", "input", "raw", "sim_dir", "out"),
}


def load_sim_config(path=SIM_CONFIG, overrides=None):
    """pilot_sim.yaml, flattened to one dict of settings, with every
    non-None entry of `overrides` (the CLI flags) replacing its key. The
    file must carry every key: a missing one is a typo, never a default
    hidden in code."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    out = {}
    for section, keys in SIM_KEYS.items():
        block = raw.get(section) or {}
        missing = [k for k in keys if k not in block]
        if missing:
            raise ValueError(f"{path}: `{section}` lacks {missing}")
        out.update({k: block[k] for k in keys})
    out["faults"] = list(raw.get("faults") or [])
    known = set(out) | {"faults"}
    for k, v in (overrides or {}).items():
        if k in known and v is not None and (k != "faults" or v):
            out[k] = v
    if out["epsilon_true_map"] is not None and not isinstance(out["epsilon_true_map"], dict):
        out["epsilon_true_map"] = json.loads(out["epsilon_true_map"])
    out["_source"] = path
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evaluate.pilot_sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-config", default=SIM_CONFIG,
                    help="the simulator's settings; every flag below overrides "
                         "its key there for one run")
    ap.add_argument("--config", default=None, help="the production config rehearsed")
    ap.add_argument("--input", default=None, help="the prepared extract")
    ap.add_argument("--raw", default=None, help="the raw extract Lane C extends")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--launch-date", default=None)
    ap.add_argument("--episodes-per-day", type=int, default=None,
                    help="the day's total, split across the two arms")
    ap.add_argument("--epsilon-true", type=float, default=None)
    ap.add_argument("--epsilon-true-map", default=None, help="JSON {category: epsilon}")
    ap.add_argument("--r-scale", type=float, default=None)
    ap.add_argument("--episode-shock-sd", type=float, default=None)
    ap.add_argument("--level-drift", type=float, default=None, dest="level_drift_per_day")
    ap.add_argument("--fault", action="append", default=[], dest="faults",
                    help="name[:arg], repeatable; one of " + ", ".join(sorted(FAULTS))
                         + " (replaces the file's list)")
    ap.add_argument("--templates-from", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes pricing the hour's batch; 0 = every core but "
                         "one, 1 = serial (same answer either way)")
    ap.add_argument("--sim-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    settings = load_sim_config(args.sim_config, vars(args))
    return run_from_settings(settings)


def run_from_settings(st):
    cfg = load_config(st["config"])
    prepared = pd.read_parquet(st["input"])
    last = prepared.date.astype(str).max()
    launch = st["launch_date"] or (pd.Timestamp(last)
                                   + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if str(launch) <= last:
        # the feature service's history is the extract: a launch inside it
        # would read real rows dated after launch as the trailing history
        raise SystemExit(f"launch_date {launch} is on or before the extract's "
                         f"last date {last}; simulate from the day after it")
    eps = st["epsilon_true_map"] if st["epsilon_true_map"] else st["epsilon_true"]
    world = World(cfg, prepared, eps, seed=st["seed"], opened_from=st["templates_from"],
                  r_scale=st["r_scale"], level_drift_per_day=st["level_drift_per_day"],
                  faults=parse_faults(st["faults"]),
                  episode_shock_sd=st["episode_shock_sd"])
    cfg_sim, config_path = build_workspace(cfg, st["sim_dir"], launch)
    # the settings this run used, beside the sim config, for the record
    with open(os.path.join(st["sim_dir"], "pilot_sim.yaml"), "w") as f:
        yaml.safe_dump({k: v for k, v in st.items() if not k.startswith("_")}, f,
                       sort_keys=False)
    sim = PilotSim(cfg_sim, world, st["sim_dir"], config_path, int(st["days"]),
                   st["episodes_per_day"], st, seed=st["seed"], raw_path=st["raw"],
                   prepared=prepared, workers=st["workers"])
    rep = sim.run()
    rep["sim_config"] = {k: v for k, v in st.items() if not k.startswith("_")}
    rep["sim_config"]["source"] = st["_source"]
    write_json(st["out"], rep)
    _print(rep, st["out"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
