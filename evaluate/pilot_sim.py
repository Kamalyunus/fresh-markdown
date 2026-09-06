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

Settings live in pilot_sim.yaml at the repo root (the world, the run, the
faults, the grading, the paths) -- apart from config.yaml on purpose, which
the sim rehearses unchanged; every flag overrides its key for one run.
Run: python3 -m evaluate.pilot_sim [--days 21] [--epsilon-true -1.2]
        [--fault mismatch:0.03 --fault demand_shock:30:0.5 ...]
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from common import episodes, metrics, provenance
from common.config import load_config, reference_discount
from common.io import read_json, write_json
from daily import assurance, export_events, monitor
from daily import ingest_outcomes as ingest
from daily import update
from engine import dp as dp_mod
from engine.decide import StateRejected, decide
from engine.posterior import PosteriorStore
from events.store import EventStore
from evaluate.pilot_world import (FEED_SCHEMA, FAULTS, World, hour_grid, parse_faults,
                                  ref_rate_features)
from evaluate.shadow import _BufferStore
from common.parallel import resolve_workers
from fit import prepare_data
from fit.train_baseline import BaselineModel, fit_level_calibration, schedule_reaches
from ops import seal as seal_mod
from ops import status

# what a healthy run shows, each graded in `grade()`; the fault that turns
# an expectation around is named so a fault run reads PASS when it fires
EXPECTATIONS = (
    ("hourly_engine", "every hour with stock is priced and stored; no state rejected, no event quarantined"),
    ("price_monotone_within_episode", "no applied price rises within an episode"),
    ("never_below_cost", "no applied price under cost"),
    ("outcome_completeness", "outcomes land for >= shadow_gate.min_event_completeness of decisions (faults: missing, duplicate -- a duplicated hour matches neither row)"),
    ("event_quality_gates", "each event-quality gate passes every day unless a fault's rate exceeds its threshold (price_mismatch_rate: mismatch, discount_rounding; duplicate_or_unmatched_rate: no sim fault reaches it)"),
    ("learning_moves_toward_truth", "every cell that updated ends closer to epsilon_true than it launched"),
    ("posterior_narrows", "every cell that updated ends with a smaller std"),
    ("tau_walks_on_spend", "tau moved and the last week's spend sits within grading.spend_over_budget_band of its budget"),
    ("stops_only_on_faults", "no stop condition fires without the fault that causes it; with it, the stop fires"),
    ("exploration_never_starves", "no grading.starve_days consecutive days with a budget in force and nothing forced (an empty affordable set is exploration off without a stop)"),
    ("agent_level_tracks_world", "every week's mean log(agent mu_ref / world mu_ref) over the pilot's hours sits inside the calibration gate band AND the elasticity bias it implies (level error / mean forced move) inside the posterior's std (the learner has no level term and reads a level error as elasticity)"),
    ("assurance_holds", "reproduction and exploration never FAIL; dispersion never FAILs with the world's marginal untouched (correlation is reported: the world's rho is a knob)"),
    ("lane_c_keeps_the_schedule_current", "the weekly re-fit reaches every week priced, so --apply is never refused on calibration_schedule_current"),
    ("apply_ran_on_cadence", "--apply ran every learning.update_cadence_days and was refused only under a fault"),
)

# the ops.status rows that matter for a RUNNING pilot; the others (the
# shadow gate, report vintages, the tune mirrors) grade the launch, which
# the sim workspace is past
STATUS_ROWS = ("launch blockers", "artifact bundle", "artifact mirrors",
               "stop conditions", "assurance")

# the event-quality gates by name and the sim faults whose RATE reaches
# each: a mismatch lands on the compared pair; the rounding fault moves
# every tier off the grid. `missing` and `duplicate` drop the hour's
# OUTCOME (ingest matches neither row of a duplicated hour), which is a
# completeness gap -- the unmatched/duplicate gate counts outcomes without
# a decision and duplicate ids, which no sim fault produces
EVENT_GATE_FAULTS = {"duplicate_or_unmatched_rate": (),
                     "price_mismatch_rate": ("mismatch",)}
COMPLETENESS_FAULTS = ("missing", "duplicate")


# --------------------------------------------------------------- workspace

def sim_config(cfg, sim_dir, launch_date):
    """The production config with launch_date set and every path that
    holds STATE moved under sim_dir; the frozen artifacts stay production's
    (read-only from here)."""
    c = copy.deepcopy(cfg)
    c["data"]["launch_date"] = str(launch_date)
    c["data"]["split_manifest_path"] = os.path.join(sim_dir, "split_manifest.json")
    c["artifacts"]["bundle_path"] = os.path.join(sim_dir, "bundle.json")
    c["artifacts"]["history_dir"] = os.path.join(sim_dir, "history")
    c["baseline_model"]["calibration_factor_path"] = os.path.join(sim_dir, "calibration.json")
    c["posterior"]["path"] = os.path.join(sim_dir, "posterior.json")
    c["events"]["store_dir"] = os.path.join(sim_dir, "events_store")
    c["events"]["shadow_store_dir"] = os.path.join(sim_dir, "events_store_shadow")
    return c


def build_workspace(cfg, sim_dir, launch_date):
    """A fresh sim_dir holding copies of the production state the daily
    lane mutates (posterior, calibration, split manifest), a sealed bundle
    over them, and the sim config. Returns (cfg_sim, config_path)."""
    if os.path.isdir(sim_dir):
        shutil.rmtree(sim_dir)
    os.makedirs(sim_dir)
    for sub in ("feed", "reports", "exports"):
        os.makedirs(os.path.join(sim_dir, sub))
    c = sim_config(cfg, sim_dir, launch_date)
    nulls = [n for n in status.runtime_nulls(c)]
    if nulls:
        raise SystemExit(f"cannot simulate a launch with null values: {nulls}")
    for src, dst in ((cfg["baseline_model"]["calibration_factor_path"],
                      c["baseline_model"]["calibration_factor_path"]),
                     (cfg["data"]["split_manifest_path"], c["data"]["split_manifest_path"]),
                     (cfg["posterior"]["path"], c["posterior"]["path"])):
        if os.path.exists(src):
            shutil.copyfile(src, dst)
    if not os.path.exists(c["posterior"]["path"]):
        prior = read_json(cfg["posterior"]["prior"]["path"])
        if not prior:
            raise SystemExit("no posterior and no prior to initialise one from")
        PosteriorStore.initialise(c, prior["per_category"],
                                  prior["episodes_per_week"])
    config_path = os.path.join(sim_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)
    _seal(c, config_path, "config")
    return c, config_path


def _seal(cfg, config_path, reason):
    payload = seal_mod.seal(cfg)
    write_json(cfg["artifacts"]["bundle_path"], payload)
    provenance.archive(cfg, payload, config_path=config_path, reason=reason)
    return payload["bundle"]


def _write_feed(rows, path):
    df = World.feed_frame(rows)
    pq.write_table(pa.Table.from_pandas(df, schema=FEED_SCHEMA,
                                        preserve_index=False), path)
    return df


# ------------------------------------------------------------ the worker

# the hourly truth, columnar (a dict per hour was the memory at 5k a day):
# buffered as tuples through the day, a DataFrame per day from then on
TRUTH_COLS = ("episode_id", "template_id", "arm", "date", "hour_of_day",
              "starting_inventory", "units_sold", "ending_inventory",
              "original_price", "offered_price", "cost", "category", "fc",
              "sku_id", "dp_eligible", "shelf_discount", "mu_true",
              "mu_ref_world", "mu_ref_agent")
HIST_COLS = ("episode_id", "sku_id", "fc", "category", "date", "hour_of_day",
             "starting_inventory", "units_sold", "total_discount")


class _SimCells:
    """The posterior as the worker sees it: the cells resolved in the
    parent for this tick, and the suspension in force (unlike shadow's
    rehearsal, a simulated pilot IS suspended when the monitor says so)."""

    def __init__(self, by_category, suspended):
        self._by_category, self._suspended = by_category, suspended

    def get(self, category):
        return self._by_category[str(category)]

    def exploration_suspended(self):
        return self._suspended


def _decision_rng(seed, episode_id, t):
    """One generator per (episode, hour), from the ids alone: the draw does
    not depend on which worker prices it or in what order."""
    h = hashlib.blake2b(str(episode_id).encode(), digest_size=8).digest()
    return np.random.default_rng([int(seed), int.from_bytes(h, "big"), int(t)])


def _price_one(item, ctx):
    """One decision in a worker: pure -- the state, the tick's posterior
    snapshot and tau, a generator seeded from the episode and the hour, so
    serial and parallel runs price identically. Returns the event or the
    rejection; the parent commits the event and runs the shop."""
    state, (episode_id, t) = item
    rng = _decision_rng(ctx["seed"], episode_id, t)
    store = _BufferStore()
    try:
        evt = decide(state, _SimCells(ctx["cells"], ctx["suspended"]), store,
                     ctx["cfg"], rng, ctx["tau"], ctx["model_version"],
                     config_digest=ctx["digest"])
    except StateRejected as e:
        return {"evt": None, "rejected": str(e)}
    return {"evt": evt, "rejected": None}


def _price_chunk(args):
    items, ctx = args
    return [_price_one(it, ctx) for it in items]


# ---------------------------------------------------------------- simulator

class PilotSim:
    def __init__(self, cfg, world, sim_dir, config_path, days, episodes_per_day,
                 sim_settings, seed=0, raw_path=None, prepared=None, workers=None):
        self.cfg, self.world, self.sim_dir = cfg, world, sim_dir
        self.config_path = config_path
        # the simulator's own knobs (pilot_sim.yaml `grading`), never the
        # system's: the lane's hour, the history margin, the grading bands
        self.grading = {k: sim_settings[k] for k in SIM_KEYS["grading"]}
        self.lane_hour = int(self.grading["lane_hour"])
        self.raw_path = raw_path
        launch = pd.Timestamp(cfg["data"]["launch_date"])
        self.dates = [(launch + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
                      for k in range(days)]
        # `episodes_per_day` is the day's total, split across the two arms
        self.per_day = max(int(episodes_per_day) // 2, 1)
        self.workers = resolve_workers(workers)
        self.pool = None
        self.seed = int(seed)
        self.rng = np.random.default_rng([int(seed), 1])       # engineering's draws
        self.opened_by_day = {}
        self.model = BaselineModel(cfg)
        self.posterior = PosteriorStore(cfg)
        self.store = EventStore(cfg)
        self.digest = provenance.config_fingerprint(cfg)["digest"]
        self.tier_step = cfg["pricing"]["tier_step"]
        # the feature service's history: the prepared extract plus every
        # simulated hour, in the prepared vocabulary. The features read a
        # trailing window, so a prepared row older than launch minus that
        # window (plus the margin) can never be read: sliced ONCE here
        self.history_days = (int(cfg["baseline_model"]["ref_rate_window_days"])
                             + int(self.grading["feature_history_margin_days"]))
        hist = prepared[list(HIST_COLS)].copy()
        hist["date"] = hist.date.astype(str)
        since = (launch - pd.Timedelta(days=self.history_days)).strftime("%Y-%m-%d")
        self.history = hist[hist.date >= since].reset_index(drop=True)
        # the day's hours as tuples, a DataFrame per day once it closes
        # (the run's whole truth as flat tuples was the memory at 5k a day)
        self._truth_rows, self._hist_rows = [], []
        self.truth_frames, self.sim_frames = {}, {}       # date -> DataFrame
        self.open, self.pending, self.busy = [], {}, set()
        self.twins_due = {}                  # date -> [(arm, template)]
        self._twin_of = {}                   # (arm, template_id, date) -> twin arm
        self.shock_by_template = {}          # the per-episode shock, shared by twins
        # the feed rows not yet written to sim_dir/feed (the lane writes a
        # day and drops it; Lane C reads the written days back from disk)
        self.feed_by_day, self.failures_by_day = {}, {}
        self.feed_written = []               # dates whose parquet is on disk
        self.days = []
        self.rejected = {}
        self.quarantined = 0
        self.launch_cells = copy.deepcopy(self.posterior.state["cells"])
        self.launch_tau = self.posterior.tau(cfg)
        self.violations = {"price_rose_within_episode": 0, "below_cost": 0}
        self.lane_c_runs = []

    # ------------------------------------------------------------- days

    def run(self):
        try:
            if self.workers > 1:
                self.pool = ProcessPoolExecutor(max_workers=self.workers)
            for k, date in enumerate(self.dates):
                self._sample_day(k, date)
                for hour in range(24):
                    if hour == self.lane_hour and k > 0:
                        self.days.append(self.daily_lane(k))
                    self._open_due(k, date, hour)
                    self._tick(k, date, hour)
                self._close_day(date)
                print(f"  day {k + 1}/{len(self.dates)} {date}: "
                      f"{self.opened_by_day.get(date, 0)} episodes opened, "
                      f"{len(self.open)} open", flush=True)
        finally:
            if self.pool is not None:
                self.pool.shutdown()
        return self.report()

    def _close_day(self, date):
        """The day's hours, buffered as tuples, become its frames."""
        self.truth_frames[date] = pd.DataFrame(self._truth_rows, columns=TRUTH_COLS)
        self.sim_frames[date] = pd.DataFrame(self._hist_rows, columns=HIST_COLS)
        self._truth_rows, self._hist_rows = [], []

    def truth(self):
        """Every simulated hour so far, one frame (concatenated once)."""
        frames = list(self.truth_frames.values())
        if self._truth_rows:
            frames.append(pd.DataFrame(self._truth_rows, columns=TRUTH_COLS))
        return (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=TRUTH_COLS))

    def _tick(self, k, date, hour):
        """One hour: every open pilot episode due now is priced in one batch
        (across the workers) against one posterior snapshot and one tau --
        the batch Lane B reloads the store for -- then the shop sells."""
        self.posterior.reload()                    # once per batch, as Lane B must
        due = [ep for ep in self.open if ep["grid"][ep["t"]] == (date, hour)]
        pilot = [ep for ep in due if ep["arm"] == "pilot"]
        if pilot:
            cats = {ep["template"]["category"] for ep in pilot}
            ctx = {"cfg": self.cfg, "tau": self.posterior.tau(self.cfg),
                   "cells": {c: self.posterior.get(c) for c in cats},
                   "suspended": self.posterior.exploration_suspended(),
                   "model_version": self.model.version, "digest": self.digest,
                   "seed": self.seed}
            items = [(self._pilot_state(ep), (ep["episode_id"], ep["t"])) for ep in pilot]
            for ep, res in zip(pilot, self._map(items, ctx)):
                self._pilot_hour(ep, k, res)
        for ep in due:
            if ep["arm"] == "legacy":
                self._legacy_hour(ep, k)

    def _map(self, items, ctx):
        """`[_price_one(it, ctx) for it in items]`, chunked across the pool
        held for the run (one executor per hour would fork 500 times a
        run); results in submission order."""
        if self.pool is None or len(items) < 2 * self.workers:
            return [_price_one(it, ctx) for it in items]
        size = max(len(items) // (self.workers * 4), 1)
        batches = [items[i:i + size] for i in range(0, len(items), size)]
        out = []
        for got in self.pool.map(_price_chunk, [(b, ctx) for b in batches]):
            out.extend(got)
        return out

    def _sample_day(self, k, date):
        """Each arm opens `per_day` episodes a day: the twins due (a fresh
        pick's twin runs under the OTHER arm the day after it closes --
        never while its sku x fc is still open, or the feed would hold two
        states for one hour; a busy twin waits a day) plus fresh templates
        to fill up. Every fresh pick gets a twin; on a small pool a template
        is re-picked, so the paired economics are computed over the
        templates settled under both arms, never assumed."""
        pilot, legacy = [], []
        for arm, t in self.twins_due.pop(date, []):
            if (t["sku_id"], t["fc"]) in self.busy:          # still open: tomorrow
                self._schedule_twin(arm, t, date)
                continue
            (pilot if arm == "pilot" else legacy).append(t)
            self.busy.add((t["sku_id"], t["fc"]))
        reserved = {(t["sku_id"], t["fc"]) for due in self.twins_due.values()
                    for _, t in due}
        pool = [t for t in self.world.templates
                if (t["sku_id"], t["fc"]) not in self.busy | reserved]
        need = {"pilot": max(self.per_day - len(pilot), 0),
                "legacy": max(self.per_day - len(legacy), 0)}
        for i in self.rng.permutation(len(pool)):
            if not need["pilot"] and not need["legacy"]:
                break
            t = pool[i]
            key = (t["sku_id"], t["fc"])
            if key in self.busy:
                continue
            arm = "pilot" if need["pilot"] >= need["legacy"] else "legacy"
            need[arm] -= 1
            (pilot if arm == "pilot" else legacy).append(t)
            self.busy.add(key)
            self._twin_of[(arm, t["template_id"], date)] = \
                "legacy" if arm == "pilot" else "pilot"
        openings = []
        for arm, temps in (("pilot", pilot), ("legacy", legacy)):
            for t in temps:
                eid = f"sim|{arm}|{t['sku_id']}|{t['fc']}|{date}T{t['opening_hour']:02d}"
                # the episode's shock is drawn at its first opening and
                # shared by its twin: the pair sees the same world
                if t["template_id"] not in self.shock_by_template:
                    self.shock_by_template[t["template_id"]] = self.world.episode_shock()
                openings.append({"arm": arm, "episode_id": eid, "template": t,
                                 "grid": hour_grid(date, t["opening_hour"], t["n_hours"]),
                                 "t": 0, "q": t["q0"], "anchor": None, "day": k,
                                 "shock": self.shock_by_template[t["template_id"]],
                                 "twin": self._twin_of.pop((arm, t["template_id"], date), None)})
        # the two demand-rate features, point-in-time, by the one home
        stub = pd.DataFrame([{
            "episode_id": o["episode_id"], "sku_id": o["template"]["sku_id"],
            "fc": o["template"]["fc"], "category": o["template"]["category"],
            "date": date, "hour_of_day": o["template"]["opening_hour"],
            "starting_inventory": o["q"]} for o in openings],
            columns=list(HIST_COLS[:7]))
        feats = ref_rate_features(self._feature_history(date), stub, self.cfg)
        for o in openings:
            o["features"] = feats[o["episode_id"]]
        # one prediction per model for the whole day's openings
        for o, path in zip(openings, self.world.mu_ref_paths(openings)):
            o["mu_world"] = path
        pilot_open = [o for o in openings if o["arm"] == "pilot"]
        for o, path in zip(pilot_open, self.world.mu_ref_paths(pilot_open, model=self.model)):
            o["mu_agent"] = path
        self.pending[date] = openings
        self.opened_by_day[date] = len(openings)

    def _feature_history(self, date):
        """The feature service's history as of `date`: the prepared rows and
        the simulated days inside the trailing window the features read --
        only those frames are concatenated."""
        since = (pd.Timestamp(date) - pd.Timedelta(days=self.history_days)
                 ).strftime("%Y-%m-%d")
        frames = [self.history[self.history.date >= since]]
        frames += [f for d, f in self.sim_frames.items() if d >= since and len(f)]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def _schedule_twin(self, arm, template, after_date):
        day = (pd.Timestamp(after_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        self.twins_due.setdefault(day, []).append((arm, template))

    def _open_due(self, k, date, hour):
        for o in list(self.pending.get(date, [])):
            if o["template"]["opening_hour"] == hour:
                self.pending[date].remove(o)
                self.open.append(o)
        if date in self.pending and not self.pending[date]:
            del self.pending[date]

    # ------------------------------------------------------------ hours

    def _pilot_state(self, ep):
        t, tpl = ep["t"], ep["template"]
        date, hour = ep["grid"][t]
        return {
            "episode_id": ep["episode_id"], "sku_id": tpl["sku_id"], "fc": tpl["fc"],
            "category": tpl["category"], "subcategory": tpl["subcategory"],
            "date": date, "hour_of_day": hour, "hours_remaining": tpl["n_hours"] - t,
            "q": int(ep["q"]), "original_price": tpl["original_price"],
            "cost": tpl["cost"], "r": self.world.r_of(tpl) / self.world.r_scale,
            "mu_ref_path": list(ep["mu_agent"][t:]),
            "current_discount": ep["anchor"],
        }

    def _pilot_hour(self, ep, k, res):
        """Engineering's side of one priced hour, after the worker's
        decision: commit the event, apply the price (or the fault), sell."""
        t, tpl = ep["t"], ep["template"]
        date, hour = ep["grid"][t]
        d_ref = reference_discount(self.cfg, tpl["category"])
        applied = None
        if res["evt"] is not None:
            evt = res["evt"]
            # the store validates on emit: a refused event is quarantined,
            # and an hour priced but never stored is graded (hourly_engine)
            if not self.store.emit_decision(evt):
                self.quarantined += 1
            applied = float(evt["applied_discount"])
            if ep["anchor"] is not None and applied < ep["anchor"] - dp_mod.TIER_EPS:
                self.violations["price_rose_within_episode"] += 1
            if evt["applied_price"] < tpl["cost"] - 1e-6:
                self.violations["below_cost"] += 1
        else:
            self.rejected[res["rejected"]] = self.rejected.get(res["rejected"], 0) + 1
        # engineering applies it -- or fails to. The defined fallback holds
        # the shelf; at entry the shelf opens at the legacy anchor, on the
        # tier grid at or above cost (a bare d_max off the grid would leave
        # the next decision no feasible tier at or below its anchor)
        if ep["anchor"] is not None:
            hold = ep["anchor"]
        else:
            tiers, _ = dp_mod.feasible_tiers(tpl["original_price"], tpl["cost"],
                                             self.tier_step)
            hold = max([d for d in tiers if d <= d_ref + dp_mod.TIER_EPS] or tiers[:1])
        if applied is None:
            shelf = hold                                   # the defined fallback
        elif self.world.draw_fault("push_fail"):
            shelf = hold
            self.failures_by_day.setdefault(date, []).append({
                "sku_id": tpl["sku_id"], "fc": tpl["fc"], "date": date,
                "hour_of_day": hour, "reason": "simulated push failure"})
        elif self.world.draw_fault("mismatch"):
            d_max = 1 - tpl["cost"] / tpl["original_price"]
            shelf = applied + self.tier_step if applied + self.tier_step <= d_max \
                else max(applied - self.tier_step, 0.0)
        else:
            shelf = applied
        self._sell(ep, k, shelf)

    def _legacy_hour(self, ep, k):
        tpl = ep["template"]
        d_max = 1 - tpl["cost"] / tpl["original_price"]
        self._sell(ep, k, min(tpl["legacy_path"][ep["t"]], d_max))

    def _sell(self, ep, k, shelf):
        """The shop: demand at the shelf price, the feed row, the truth row."""
        t, tpl = ep["t"], ep["template"]
        date, hour = ep["grid"][t]
        q = int(ep["q"])
        draw, mu = self.world.demand(tpl, ep["mu_world"][t], shelf, k, ep["shock"])
        sold = min(draw, q)
        left = q - sold
        close = (t == tpl["n_hours"] - 1) or left == 0
        ending = 0 if close else left                      # write-off sentinel
        row = self.world.feed_row(tpl, date, hour, q, shelf, sold, ending,
                                  hours_remaining=tpl["n_hours"] - t)
        if not self.world.draw_fault("missing"):
            self.feed_by_day.setdefault(date, []).append(row)
            if self.world.draw_fault("duplicate"):
                self.feed_by_day[date].append(dict(row))
        # TRUTH_COLS order; the two levels at the reference -- the world's
        # (the frozen prediction times the day's drift and shock fault,
        # World.level_multiplier: what the re-fit should track) and the
        # agent's (its own re-fit factors) -- are the level error the
        # elasticity learner, which has no level term, absorbs
        self._truth_rows.append((
            ep["episode_id"], tpl["template_id"], ep["arm"], date, hour, q, sold,
            ending, tpl["original_price"], tpl["original_price"] * (1 - shelf),
            tpl["cost"], tpl["category"], tpl["fc"], tpl["sku_id"], True, shelf, mu,
            ep["mu_world"][t] * self.world.level_multiplier(k),
            ep["mu_agent"][t] if ep["arm"] == "pilot" else None))
        self._hist_rows.append((                     # HIST_COLS order
            ep["episode_id"], tpl["sku_id"], tpl["fc"], tpl["category"], date, hour,
            q, sold, shelf))
        ep["q"], ep["anchor"], ep["t"] = left, shelf, t + 1
        if close:
            self.open.remove(ep)
            self.busy.discard((tpl["sku_id"], tpl["fc"]))
            if ep.get("twin"):
                self._schedule_twin(ep["twin"], tpl, date)

    # ------------------------------------------------------- daily lane

    def daily_lane(self, k):
        """The morning of day k: yesterday's feed through the lane's own
        functions, in the order ops.advance --feed runs them."""
        cfg, today, yesterday = self.cfg, self.dates[k], self.dates[k - 1]
        lane = {"day": k, "date": yesterday, "lane_c": None}
        # Lane C on ops.advance's own rule: the schedule must reach one week
        # past the latest data's week (episodes.week_after of the max date
        # Lane C would prepare -- the feed's, today's early hours included);
        # the first morning always re-fits, since the sealed schedule is
        # pre-launch
        cal = read_json(cfg["baseline_model"]["calibration_factor_path"]) or {}
        reaches = schedule_reaches(cal.get("schedule") or {}) or ""
        latest = max(self.feed_written + list(self.feed_by_day), default=yesterday)
        expected = episodes.week_after(latest)
        if reaches < expected or not self.lane_c_runs:
            lane["lane_c"] = self.lane_c(k)
        lane["calibration_current"] = update.calibration_current(cfg, today)

        # yesterday's feed goes to disk and out of memory: from here Lane C
        # reads the day back from its parquet
        feed_path = os.path.join(self.sim_dir, "feed", f"{yesterday}.parquet")
        feed = _write_feed(self.feed_by_day.pop(yesterday, []), feed_path)
        self.feed_written.append(yesterday)
        failures, failed = None, self.failures_by_day.pop(yesterday, None)
        if failed:
            failures = os.path.join(self.sim_dir, "feed", f"{yesterday}-failures.jsonl")
            with open(failures, "w") as f:
                for r in failed:
                    f.write(json.dumps(r) + "\n")
        # ONE store for the morning: ingest emits through it, the monitor,
        # assurance and the export read it (update.run builds its own)
        store = EventStore(cfg)
        outcomes, rep = ingest.build_outcomes(store.load_decisions(), feed,
                                              ingest.load_failures(failures))
        emitted = sum(store.emit_outcome(o) for o in outcomes)
        lane["ingest"] = {k_: rep[k_] for k_ in (
            "decisions", "decisions_outside_feed_range", "outcomes_built",
            "decisions_without_feed_row", "feed_duplicate_hours",
            "unusable_feed_rows", "push_failures_applied",
            "push_failures_unmatched")}
        lane["ingest"].update(emitted=int(emitted),
                              quarantined=store.quarantined_this_run,
                              feed_rows=int(len(feed)))

        walk = update.run(cfg, calibrate_tau=True)
        tc = walk["tau_calibration"]
        lane["gates"] = {n: g["pass"] for n, g in walk["event_quality_gates"].items()}
        # every row the controller walked this morning (engine.explore
        # .walk_tau: day, spend, budget, tau, tau_after, clipped, held)
        lane["tau"] = {"before": tc["tau_before"], "after": tc["tau_after"],
                       "committed": walk.get("tau_committed", False),
                       "skipped": tc.get("skipped"),
                       "walked": list(tc.get("by_day") or [])}

        mon = monitor.build_report(store, PosteriorStore(cfg), cfg)
        write_json(os.path.join(self.sim_dir, "reports", "monitor.json"), mon)
        lane["stops"] = {n: v for n, v in mon["stop_conditions"]["fired"].items()}
        lane["guardrails"] = {n: {"latest": g.get("latest"),
                                  "threshold": g.get("threshold"),
                                  "consecutive_days_over": g.get("consecutive_days_over")}
                              for n, g in mon["stop_conditions"]["guardrails"].items()}
        lane["suspended"] = mon["exploration_suspended"]
        lane["business"] = {n: mon["business"].get(n) for n in
                            ("il_pct_aggregate", "sell_through", "waste_units")}
        lane["learning"] = {n: mon["learning"].get(n) for n in
                            ("forced_decision_count", "affordable_set_empty_rate",
                             "realised_exploration_cost", "tau_current",
                             "posterior_std_flat_alert")}

        ass = assurance.run(store.load_decisions(), store.load_outcomes(), cfg)
        write_json(os.path.join(self.sim_dir, "reports", "assurance.json"), ass)
        lane["assurance"] = {n: ass[n]["verdict"] for n in
                             ("reproduction", "dispersion", "correlation", "exploration")}
        lane["assurance_detail"] = {
            "rho_live": ass["correlation"].get("rho_live"),
            "deff_live": ass["correlation"].get("deff_live"),
            "uniformity_p": ass["exploration"].get("p_value"),
            "uniformity_max_bin_deviation": ass["exploration"].get("max_bin_deviation")}

        export_events.export(store, os.path.join(self.sim_dir, "exports"), since=yesterday)
        # status runs whole, as the lane does; recorded are the rows that
        # matter for a RUNNING pilot (the pre-launch rows -- the shadow
        # gate, report vintages -- read the sim workspace as stale)
        st = status.collect(cfg, os.path.join(self.sim_dir, "reports"))
        lane["status_failing"] = [
            {"check": r["check"], "verdict": r["verdict"], "detail": r["detail"]}
            for r in st["checks"] if r["check"] in STATUS_ROWS and r["verdict"] != "PASS"]

        cadence = int(cfg["learning"]["update_cadence_days"])
        if k % cadence == 0:
            app = update.run(cfg, apply=True)
            lane["apply"] = {
                "applied": app["applied"], "refused": app.get("refused"),
                "calibration_schedule_current": app["event_quality_gates"]
                ["calibration_schedule_current"]["pass"],
                "excluded": app["batch"],
                "cells": {c: {n: r[n] for n in (
                    "forced_outcomes", "effective_information", "update_triggered",
                    "mean_before", "proposed_mean", "std_before", "proposed_std",
                    "bound_clipped")} for c, r in app["cells"].items()}}
        self.posterior.reload()
        lane["posterior"] = {c: {"mean": r["mean"], "std": r["std"], "n_obs": r["n_obs"],
                                 "version": r["version"]}
                             for c, r in self.posterior.state["cells"].items()}
        lane["tau_in_force"] = self.posterior.tau(cfg)
        return lane

    def lane_c(self, k):
        """The weekly cron: the extract refreshed with every simulated hour
        so far (as the source would report it, faults included), prepared,
        the level factors re-fit to the week being priced, the bundle
        re-sealed, the agent's model re-read -- and every open pilot
        episode re-priced on it from its next hour, as production would."""
        cfg = self.cfg
        raw_sim = os.path.join(self.sim_dir, "raw_sim.parquet")
        self._write_raw_sim(raw_sim)
        d, wf = prepare_data.load_and_filter(raw_sim, cfg)
        prepare_data.write_manifest(cfg["data"]["split_manifest_path"], cfg, wf)
        fit_level_calibration(d, cfg)
        cal = read_json(cfg["baseline_model"]["calibration_factor_path"])
        sched = cal["schedule"]
        bundle = _seal(cfg, self.config_path, "weekly-refit")
        self.model = BaselineModel(cfg)
        self._repredict_open_pilot()
        run = {"day": k, "date": self.dates[k], "bundle": bundle,
               "schedule_end": schedule_reaches(sched),
               "last_fitted_week": max(sched["by_week"]) if sched["by_week"] else None,
               "weeks_fitted": sched["weeks_fitted"],
               "weeks_unfitted_held_at_1": sched["weeks_unfitted_held_at_1"],
               "scope": sched["scope"], "prepared_rows": int(len(d))}
        self.lane_c_runs.append(run)
        return run

    def _write_raw_sim(self, path):
        """The raw extract plus every feed row so far, in the source schema,
        streamed: the raw file's row groups are copied batch by batch, the
        written feed days appended from their parquets, the days not yet
        written from memory -- the raw extract never enters pandas here."""
        names = [f.name for f in FEED_SCHEMA]
        with pq.ParquetWriter(path, FEED_SCHEMA) as writer:
            for src in [self.raw_path] + [
                    os.path.join(self.sim_dir, "feed", f"{d}.parquet")
                    for d in sorted(self.feed_written)]:
                for batch in pq.ParquetFile(src).iter_batches(columns=names):
                    writer.write_table(pa.Table.from_batches([batch])
                                       .select(names).cast(FEED_SCHEMA))
            rows = [r for d in sorted(self.feed_by_day) for r in self.feed_by_day[d]]
            if rows:
                writer.write_table(pa.Table.from_pandas(
                    World.feed_frame(rows), schema=FEED_SCHEMA, preserve_index=False))

    def _repredict_open_pilot(self):
        """After a re-fit the hours still to come price on the new factors:
        `mu_agent` from each open pilot episode's next hour on (and every
        hour of the day's pilot openings not yet open) is predicted again
        with the model now in force -- the grid from t onward."""
        eps = [ep for ep in self.open if ep["arm"] == "pilot"]
        eps += [o for opened in self.pending.values() for o in opened
                if o["arm"] == "pilot"]
        stubs = [{"template": ep["template"], "grid": ep["grid"][ep["t"]:],
                  "features": ep["features"]} for ep in eps]
        for ep, path in zip(eps, self.world.mu_ref_paths(stubs, model=self.model)):
            ep["mu_agent"] = list(ep["mu_agent"][:ep["t"]]) + list(path)

    # ----------------------------------------------------------- report

    def economics(self, truth=None):
        """Both arms through the one episode frame (metrics.episode_economics
        over metrics.settled): per arm over everything settled, and PAIRED
        over the templates settled under both arms (a twin postponed past
        the run's end, or a template re-picked, leaves the arms' template
        sets unequal -- `unpaired_templates` counts them)."""
        df = self.truth() if truth is None else truth
        settled = {}
        for arm, g in df.groupby("arm"):
            ep, excluded = metrics.settled(metrics.episode_economics(g))
            settled[arm] = (g, ep, excluded)
        out = {arm: _arm_economics(g, ep, excluded)
               for arm, (g, ep, excluded) in settled.items()}
        template_of = df.drop_duplicates("episode_id").set_index("episode_id").template_id
        by_arm = {arm: set(template_of.reindex(ep.index)) for arm, (_, ep, _) in settled.items()}
        both = set.intersection(*by_arm.values()) if len(by_arm) == 2 else set()
        paired = {"templates": len(both),
                  "unpaired_templates": len(set.union(*by_arm.values()) - both)
                  if by_arm else 0}
        for arm, (g, ep, _) in settled.items():
            mine = ep[template_of.reindex(ep.index).isin(both).to_numpy()]
            paired[arm] = _arm_economics(g, mine, {})
        out["paired"] = paired
        return out

    def learning(self):
        cells = self.posterior.state["cells"]
        cell_of = self.posterior.state["cell_of"]
        truth = self.world.epsilon_true
        out = {}
        for c, rec in cells.items():
            members = [cat for cat, cell in cell_of.items() if cell == c] or list(truth)
            known = [truth[m] for m in members if m in truth]
            # a cell none of whose categories the world simulates has no
            # truth to grade against: reported, never averaged over nothing
            eps = float(np.mean(known)) if known else None
            launch = self.launch_cells[c]
            out[c] = {"members": members,
                      "epsilon_true": round(eps, 4) if eps is not None else None,
                      "launch_mean": launch["mean"], "launch_std": launch["std"],
                      "mean": rec["mean"], "std": rec["std"], "n_obs": rec["n_obs"],
                      "version": rec["version"],
                      "abs_error_at_launch": (round(abs(launch["mean"] - eps), 4)
                                              if eps is not None else None),
                      "abs_error_now": (round(abs(rec["mean"] - eps), 4)
                                        if eps is not None else None),
                      "accumulated_information": round(rec["accumulated_information"], 3)}
        return out

    def level_tracking(self, decisions, truth=None):
        """Per ISO week, the mean log ratio of the agent's mu_ref to the
        world's over the pilot's priced hours: 0 when the weekly re-fit
        reproduces the world's level, off by the re-fit's error otherwise.
        The elasticity learner reads every outcome against the agent's
        mu_ref and carries no level term, so a level error this size is
        read as elasticity -- the diagnostic that tells a learning FAIL
        from a re-fit artefact. `decisions` is the store's list, loaded
        once by the caller."""
        df = self.truth() if truth is None else truth
        df = df[df.arm == "pilot"]
        if df.empty:
            return {}
        df = df.assign(log_ratio=np.log(df.mu_ref_agent.astype(float)
                                        / df.mu_ref_world.astype(float)),
                       week=episodes.week_key(df.date))
        # the lever the learner identifies elasticity with: the forced
        # moves' SIGNED log price ratio, log((1 - applied) / (1 - reference))
        # -- negative for a deeper move. A level error of e read against
        # moves of mean L is an elasticity error of about -e / L (e > 0
        # and L < 0 bias the belief toward zero); small moves make the
        # learner hypersensitive to the level
        forced = pd.DataFrame([(d["date"], d["reference_discount"], d["applied_discount"])
                               for d in decisions if d["is_exploration"]],
                              columns=["date", "reference_discount", "applied_discount"])
        if len(forced):
            forced["move"] = np.log((1.0 - forced.applied_discount.astype(float))
                                    / (1.0 - forced.reference_discount.astype(float)))
            moves = forced.groupby(episodes.week_key(forced.date)).move.mean()
        else:
            moves = pd.Series(dtype=float)
        out = {}
        for wk, g in df.groupby("week"):
            e = float(g.log_ratio.mean())
            move = float(moves.get(wk, np.nan))
            out[wk] = {"hours": int(len(g)),
                       "mean_log_ratio": round(e, 4),
                       "p10_p90": [round(float(g.log_ratio.quantile(q)), 4)
                                   for q in (0.1, 0.9)],
                       "mean_forced_log_move": round(move, 4) if np.isfinite(move) else None,
                       "implied_elasticity_bias": (round(-e / move, 3)
                                                   if np.isfinite(move) and move != 0
                                                   else None)}
        return out

    def report(self):
        decisions = self.store.load_decisions()             # once, for every reader
        truth = self.truth()
        n_dec = len(decisions)
        forced = sum(1 for d in decisions if d["is_exploration"])
        rep = {
            "world": {
                "epsilon_true": self.world.epsilon_true, "r_scale": self.world.r_scale,
                "level_drift_per_day": self.world.drift, "faults": self.world.faults,
                "episode_shock_sd": self.world.episode_shock_sd,
                "templates": len(self.world.templates),
                "launch_date": self.cfg["data"]["launch_date"],
                "days": len(self.dates), "episodes_per_day": 2 * self.per_day,
                # what the template pool and the open sku x fc keys allowed
                "episodes_opened_per_day_mean": round(float(np.mean(
                    list(self.opened_by_day.values()))), 1) if self.opened_by_day else 0,
                "workers": self.workers,
                "seed_note": "every figure is the simulated world's, not the shop's (rule 19)",
            },
            "config": provenance.config_fingerprint(self.cfg, "pilot_sim"),
            "engine": {
                "decisions": n_dec, "forced": forced,
                "forced_share": round(forced / n_dec, 4) if n_dec else None,
                "rejected": self.rejected,
                "rejected_total": int(sum(self.rejected.values())),
                "quarantined": int(self.quarantined),
                "violations": self.violations,
                "pilot_hours": int((truth.arm == "pilot").sum()),
                "tau_at_launch": self.launch_tau, "tau_now": self.posterior.tau(self.cfg),
            },
            "learning": self.learning(),
            "economics": self.economics(truth),
            "lane_c": self.lane_c_runs,
            "level_tracking": self.level_tracking(decisions, truth),
            "days": self.days,
        }
        rep["expectations"] = grade(rep, self.cfg, self.grading)
        return rep


def _arm_economics(hours, ep, excluded):
    """One arm's figures from its hourly frame and its SETTLED episode
    frame; the mean discount is over the settled episodes' hours only."""
    den = float(ep.denom.sum())
    units = float(ep.units_sold.sum() + ep.scrap.sum())
    mine = hours[hours.episode_id.isin(ep.index)]
    return {
        "episodes": int(len(ep)), "hours": int(len(mine)),
        "il_absolute": round(float(ep.il.sum()), 1),
        "il_pct": round(float(ep.il.sum() / den), 6) if den > 0 else None,
        "il_pct_denominator": round(den, 1),
        "scrap_units": int(ep.scrap.sum()),
        "scrap_rate": round(float(ep.scrap.sum() / ep.supply.sum()), 4)
        if ep.supply.sum() > 0 else None,
        "sell_through": round(float(ep.units_sold.sum() / units), 4) if units else None,
        "margin": round(float(ep.margin.sum()), 1),
        "mean_discount": round(float(mine.shelf_discount.mean()), 4) if len(mine) else None,
        "excluded": excluded,
    }


# ------------------------------------------------------------------ grading

def _verdict(ok, measured=True):
    if not measured:
        return "NOT MEASURED"
    return "PASS" if ok else "FAIL"


def _fault_rate(faults, *names):
    """The summed rate of the named faults in force (a rate-less fault --
    `discount_rounding`, `demand_shock` -- contributes nothing here)."""
    return sum(float(faults[n]) for n in names
               if n in faults and isinstance(faults[n], (int, float))
               and not isinstance(faults[n], bool))


def expected_gate_failures(faults, cfg):
    """{gate name: should it fail}: an event-quality gate is expected to
    fail only when the rate of the faults that reach it exceeds ITS
    threshold (`EVENT_GATE_FAULTS`); `discount_rounding` always moves the
    price-mismatch gate (every tier off the grid)."""
    sc = cfg["monitoring"]["stop_conditions"]
    out = {name: _fault_rate(faults, *fs) > sc[name]
           for name, fs in EVENT_GATE_FAULTS.items()}
    out["price_mismatch_rate"] |= bool(faults.get("discount_rounding"))
    return out


def grade(rep, cfg, sim_settings):
    """EXPECTATIONS against the run. A fault that is present turns its
    expectation around: the gate/stop it targets must fire. `cfg` is the
    system's config (its thresholds are what the lane compared against);
    `sim_settings` carries the simulator's own grading knobs
    (pilot_sim.yaml `grading`)."""
    faults = rep["world"]["faults"]
    days = rep["days"]
    eng = rep["engine"]
    gr = sim_settings
    out = []

    def add(name, ok, observed, measured=True):
        out.append({"name": name, "expected": dict(EXPECTATIONS)[name],
                    "verdict": _verdict(ok, measured), "observed": observed})

    # every pilot hour is a decision, a rejection or a quarantined event
    quarantined = int(eng.get("quarantined", 0))
    accounted = eng["decisions"] + eng["rejected_total"] + quarantined
    add("hourly_engine", eng["decisions"] > 0 and eng["rejected_total"] == 0
        and quarantined == 0 and eng["pilot_hours"] == accounted,
        {"decisions": eng["decisions"], "rejected": eng["rejected"],
         "quarantined": quarantined, "pilot_hours": eng["pilot_hours"]})
    add("price_monotone_within_episode", eng["violations"]["price_rose_within_episode"] == 0,
        eng["violations"])
    add("never_below_cost", eng["violations"]["below_cost"] == 0, eng["violations"])

    ing = [d["ingest"] for d in days]
    due = sum(i["decisions"] - i["decisions_outside_feed_range"] for i in ing)
    built = sum(i["outcomes_built"] for i in ing)
    gaps = sum(i["decisions_without_feed_row"] for i in ing)
    completeness = built / (built + gaps) if built + gaps else None
    floor = cfg["monitoring"]["shadow_gate"]["min_event_completeness"]
    # a missing row and a duplicated hour (ingest matches neither state)
    # both cost the decision its outcome: a gap is expected once their
    # summed rate exceeds what the floor admits
    gap_rate = _fault_rate(faults, *COMPLETENESS_FAULTS)
    expect_gap = gap_rate > 1 - floor
    ok = completeness is not None and ((completeness >= floor) != expect_gap)
    add("outcome_completeness", ok,
        {"completeness": round(completeness, 4) if completeness is not None else None,
         "floor": floor, "fault_expects_a_gap": expect_gap,
         "fault_gap_rate": round(gap_rate, 4), "decisions_due": due},
        measured=completeness is not None)

    # per gate, by name: the calibration gate is graded by
    # lane_c_keeps_the_schedule_current, not here
    expect_by_gate = expected_gate_failures(faults, cfg)
    failed_by_gate = {name: [d["date"] for d in days if d["gates"].get(name) is False]
                      for name in expect_by_gate}
    gates_off = [name for name, exp in expect_by_gate.items()
                 if bool(failed_by_gate[name]) != exp]
    expect_fail = any(expect_by_gate.values())
    add("event_quality_gates", not gates_off,
        {"days_a_gate_failed": failed_by_gate, "fault_expects_a_failure": expect_by_gate,
         "gates_off_expectation": gates_off},
        measured=bool(days))

    learned = {c: r for c, r in rep["learning"].items()
               if r["version"] > 0 and r["epsilon_true"] is not None}
    add("learning_moves_toward_truth",
        all(r["abs_error_now"] < r["abs_error_at_launch"] for r in learned.values()),
        {c: {"launch": r["abs_error_at_launch"], "now": r["abs_error_now"]}
         for c, r in rep["learning"].items()}, measured=bool(learned))
    add("posterior_narrows", all(r["std"] < r["launch_std"] for r in learned.values()),
        {c: {"launch_std": r["launch_std"], "std": r["std"]}
         for c, r in rep["learning"].items()}, measured=bool(learned))

    # the level error is graded by what it does to the learner: the
    # elasticity bias it implies (level error / mean forced move) must stay
    # inside the posterior's own uncertainty at that week's end, or the
    # re-fit is steering the belief. The gate band bounds the level itself
    band = cfg["baseline_model"]["calibration_gate_band"]
    tol = max(abs(np.log(band[0])), abs(np.log(band[1])))
    level = rep.get("level_tracking") or {}
    std_by_week = {}
    if days:
        weeks = episodes.week_key(pd.Series([d["date"] for d in days]))
        for d, wk in zip(days, weeks):
            std_by_week[wk] = max(r["std"] for r in d["posterior"].values())
    off = {}
    for wk, v in level.items():
        bias, std = v.get("implied_elasticity_bias"), std_by_week.get(wk)
        if abs(v["mean_log_ratio"]) > tol or (
                bias is not None and std is not None and abs(bias) > std):
            off[wk] = {"mean_log_ratio": v["mean_log_ratio"],
                       "implied_elasticity_bias": bias, "posterior_std": std}
    add("agent_level_tracks_world", not off,
        {"weeks_off": off, "band_tolerance_log": round(float(tol), 4),
         "by_week": {wk: {"mean_log_ratio": v["mean_log_ratio"],
                          "implied_elasticity_bias": v.get("implied_elasticity_bias")}
                     for wk, v in level.items()}},
        measured=bool(level))

    # the controller moves tau by at most the clip per day, so a launch tau
    # far from this world's budget needs a week or more to arrive: graded
    # on the last `tau_week_days` days the controller actually MOVED on
    # (held days -- no base yet -- are not walks), once there are that
    # many, against the sim's own spend-over-budget band (the clip bounds
    # the daily step, not the ratio)
    walked = {}
    for d in days:
        if d["tau"].get("committed"):
            for w in d["tau"].get("walked") or []:
                walked[w["day"]] = w                  # a day walked once
    walked = [walked[day] for day in sorted(walked)]
    live = [w for w in walked if not w.get("held")]
    week_days = int(gr["tau_week_days"])
    lo, hi = gr["spend_over_budget_band"]
    week = live[-week_days:]
    ratios = [w["spend"] / w["budget"] for w in week if w["budget"] > 0]
    ratio = float(np.mean(ratios)) if ratios else None
    moved = eng["tau_now"] != eng["tau_at_launch"]
    held = [w["day"] for w in walked if w.get("held")]
    suspended = [d["date"] for d in days if d.get("suspended")]
    add("tau_walks_on_spend", moved and ratio is not None and lo <= ratio <= hi,
        {"tau_at_launch": eng["tau_at_launch"], "tau_now": eng["tau_now"],
         "last_week_spend_over_budget": round(ratio, 3) if ratio is not None else None,
         "band": [lo, hi], "week_days": week_days, "days_walked": len(live),
         "days_held": held, "days_exploration_suspended": suspended},
        measured=len(live) >= week_days)

    fired = {}
    for d in days:
        for name, v in d["stops"].items():
            if v is True:
                fired.setdefault(name, []).append(d["date"])
    # the stop behind each event gate fires on the gate's own rate: the
    # same per-gate expectation
    stop_of_gate = {"duplicate_or_unmatched_rate": "duplicate_or_unmatched",
                    "price_mismatch_rate": "price_mismatch"}
    causes = {stop_of_gate[g]: EVENT_GATE_FAULTS[g] + (("discount_rounding",)
                                                       if g == "price_mismatch_rate" else ())
              for g in stop_of_gate}
    causes.update({"scrap_deterioration_pct": ("demand_shock",),
                   "margin_deterioration_pct": ("demand_shock",)})
    unexpected = [n for n in fired if not any(f in faults for f in causes.get(n, ()))]
    # the guardrail compares against its own trailing window, smoothed,
    # over persistence_days: a shock can only be seen once the run is
    # that long past it (the monitor's series starts at launch)
    mc, sc = cfg["monitoring"], cfg["monitoring"]["stop_conditions"]
    shock_day = faults["demand_shock"][0] if "demand_shock" in faults else None
    settle = max(sc["deterioration_smoothing_days"].values()) + sc["persistence_days"]
    guardrail_ready = (shock_day is not None
                       and len(days) >= mc["guardrail_noise_window_days"] + settle
                       and len(days) >= shock_day + settle + 1)
    expected_missing = [stop_of_gate[g] for g, exp in expect_by_gate.items()
                        if exp and stop_of_gate[g] not in fired]
    if "demand_shock" in faults and guardrail_ready \
            and "scrap_deterioration_pct" not in fired:
        expected_missing.append("scrap_deterioration_pct")
    # a shock the series SAW (the scrap deviation moved worse) that still
    # sits under the owner's floor is this world's reach, not a silent
    # stop: reported with the reading, not graded
    scrap = (days[-1].get("guardrails") or {}).get("scrap_deterioration_pct") or {} if days else {}
    under_floor = ("scrap_deterioration_pct" in expected_missing
                   and scrap.get("latest") is not None and scrap.get("threshold") is not None
                   and 0 < scrap["latest"] < scrap["threshold"])
    if under_floor:
        expected_missing.remove("scrap_deterioration_pct")
    add("stops_only_on_faults", not unexpected and not expected_missing,
        {"fired": fired, "unexpected": unexpected, "expected_but_silent": expected_missing,
         "guardrail_window_reached": guardrail_ready if shock_day is not None else None,
         "scrap_deviation_latest": scrap.get("latest"), "scrap_floor": scrap.get("threshold"),
         "shock_seen_but_under_the_floor": under_floor},
        measured=bool(days) and (shock_day is None or guardrail_ready or bool(unexpected))
        and not under_floor)

    # forced per day from the monitor's cumulative count; a day with a tau
    # in force (not suspended) and no forced decision is exploration off
    starve_days = int(gr["starve_days"])
    starved, streak, worst = [], 0, 0
    prev = 0
    for d in days:
        forced_today = d["learning"]["forced_decision_count"] - prev
        prev = d["learning"]["forced_decision_count"]
        streak = streak + 1 if forced_today == 0 and not d["suspended"] else 0
        worst = max(worst, streak)
        if streak >= starve_days:
            starved.append(d["date"])
    add("exploration_never_starves", not starved,
        {"days_starved": starved, "longest_streak": worst, "starve_days": starve_days,
         "affordable_set_empty_rate_latest": days[-1]["learning"]
         ["affordable_set_empty_rate"] if days else None}, measured=bool(days))

    verdicts = {n: sorted({d["assurance"][n] for d in days}) for n in
                ("reproduction", "dispersion", "correlation", "exploration")} if days else {}
    # correlation grades the world's within-episode shock against the
    # frozen rho -- a knob of the world (`episode_shock_sd`), not a claim
    # about the machinery: reported with the live reading, never graded
    # a world whose hours share a shock is over-dispersed against an r
    # fitted without one: dispersion is then a reading of the knob too
    # ... and the check bins by PREDICTED mu, so while a gate fault keeps
    # --apply refused the belief never converges and a wrong elasticity
    # reads as a shape problem: dispersion is excused while any event
    # gate is expected to fail
    marginal_moved = (rep["world"]["r_scale"] != 1.0
                      or rep["world"].get("episode_shock_sd", 0) > 0
                      or expect_fail)
    bad = [n for n, vs in verdicts.items() if "FAIL" in vs and n != "correlation"
           and not (n == "dispersion" and marginal_moved)]
    rho = [d["assurance_detail"]["rho_live"] for d in days
           if d["assurance_detail"].get("rho_live") is not None]
    add("assurance_holds", not bad,
        {"verdicts": verdicts, "failing": bad,
         "rho_live_latest": rho[-1] if rho else None,
         "rho_frozen": cfg["dispersion"]["rho"]}, measured=bool(days))

    applies = [d["apply"] for d in days if "apply" in d]
    stale = [d["date"] for d in days if not d["calibration_current"]["pass"]]
    held = [d["date"] for d in days if d["calibration_current"].get("held_at_anchor")]
    add("lane_c_keeps_the_schedule_current", not stale,
        {"re_fits": len(rep["lane_c"]), "mornings_schedule_stale": stale,
         "mornings_priced_on_the_held_anchor": held,
         "schedule_reaches": [r["schedule_end"] for r in rep["lane_c"]]},
        measured=bool(days))

    cadence = int(cfg["learning"]["update_cadence_days"])
    expected_applies = sum(1 for d in days if d["day"] % cadence == 0)
    refused = [d["date"] for d in days if d.get("apply", {}).get("refused")]
    fault_refusal = expect_fail
    add("apply_ran_on_cadence", len(applies) == expected_applies
        and (not refused or fault_refusal),
        {"applies": len(applies), "expected": expected_applies, "refused_on": refused,
         "fault_explains_refusal": fault_refusal}, measured=bool(days))
    return out


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
    # how a run is read, not what it rehearses)
    "grading": ("spend_over_budget_band", "tau_week_days", "starve_days",
                "lane_hour", "feature_history_margin_days"),
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
