"""evaluate.pilot_shop -- the simulated shop and engineering's side of the loop.

The home of `PilotSim`: the workspace a rehearsal runs in (copies of the
production state the daily lane mutates, sealed; `build_workspace`), the
shop's clock (`run_hours`: the openings due, one priced batch per hour,
the sale, the day's close), the shelf with its faults, the feed row the
source would write (`write_feed`), the twins, the daily lane and Lane C's
weekly re-fit -- and the two ways an hour's pilot states get their
decisions: the rehearsal's (`engine.decide` in the workers, through the
one worker body `engine.state.price_one`) and Lane B's (`LaneBPricer`:
the contract's 12-field requests through `ops.price_batch`, with the
request and decision files engineering would see). The outcome side of
that cycle -- ingest, emit, export, the pairing and the outcome-id check
-- is `ingest_and_pair`, on the same `ingest_feed` the lane's morning
runs. `evaluate.pilot_sim` drives a multi-day run from `pilot_sim.yaml`
and grades it (`evaluate.pilot_grade`); `tools.e2e_cycle` drives one day
through Lane B.
"""

import copy
import json
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from common import episodes, provenance
from common.config import reference_discount
from common.io import read_json, write_json
from common.parallel import EpisodePool, resolve_workers
from daily import assurance, export_events, monitor
from daily import ingest_outcomes as ingest
from daily import update
from engine import dp as dp_mod
from engine.posterior import PosteriorStore
from engine.state import (HISTORY_COLS as HIST_COLS, assemble_state,
                          ref_rate_table,
                          batch_context, hour_grid, price_one, ref_rate_features)
from events.pairs import hour_key, match_pairs, outcome_id_of, price_matches
from events.store import EventStore
from evaluate.pilot_grade import GRADING_KEYS, economics, grade, learning, level_tracking
from evaluate.pilot_world import FEED_SCHEMA, World
from fit import prepare_data
from fit.artifacts import load_bundle
from fit.train_baseline import fit_level_calibration, schedule_reaches
from ops import price_hour
from ops import seal as seal_mod
from ops import status

# the ops.status rows that matter for a RUNNING pilot; the others (the
# shadow gate, report vintages, the tune mirrors) grade the launch, which
# the sim workspace is past
STATUS_ROWS = ("launch blockers", "artifact bundle", "artifact mirrors",
               "stop conditions", "assurance")


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


# ---------------------------------------------------------------- the feed

# the hourly truth, columnar (a dict per hour was the memory at 5k a day):
# buffered as tuples through the day, a DataFrame per day from then on
TRUTH_COLS = ("episode_id", "template_id", "arm", "date", "hour_of_day",
              "starting_inventory", "units_sold", "ending_inventory",
              "original_price", "offered_price", "cost", "category", "fc",
              "sku_id", "dp_eligible", "shelf_discount", "mu_true",
              "mu_ref_world", "mu_ref_agent")


def ingest_feed(store, feed, failures=None):
    """The outcome side of one feed, through the lane's own functions:
    every decision the store holds against `feed` (the hourly frame in the
    source schema) and the reported push failures (a path, or None),
    every outcome built emitted through `store`. Returns (decisions, the
    ingester's report with `emitted` added) -- the same step the
    simulator's morning and the integration cycle run."""
    decisions = store.load_decisions()
    outcomes, rep = ingest.build_outcomes(decisions, feed, ingest.load_failures(failures))
    rep["emitted"] = int(sum(store.emit_outcome(o) for o in outcomes))
    return decisions, rep


# ---------------------------------------------------------------- simulator

class PilotSim:
    """The shop and engineering's side of the loop. `pricer` is how an
    hour's pilot states get their decisions: None runs engine.decide in
    the workers (`_price_pilot`, the rehearsal); tools.e2e_cycle passes a
    `LaneBPricer` that goes through ops.price_batch, so the integration
    cycle and the simulator share one shop -- the shelf, the demand draw,
    the feed row, the twins -- and differ only in who prices. A pricer
    with a `bind` method is handed the shop at construction."""

    def __init__(self, cfg, world, sim_dir, config_path, days, episodes_per_day,
                 sim_settings, seed=0, raw_path=None, prepared=None, workers=None,
                 pricer=None):
        self.cfg, self.world, self.sim_dir = cfg, world, sim_dir
        self.config_path = config_path
        self.pricer = pricer or self._price_pilot
        # the simulator's own knobs (pilot_sim.yaml `grading`), never the
        # system's: the lane's hour, the history margin, the grading bands
        self.grading = {k: sim_settings[k] for k in GRADING_KEYS}
        self.lane_hour = int(self.grading["lane_hour"])
        self.raw_path = raw_path
        launch = pd.Timestamp(cfg["data"]["launch_date"])
        self.dates = [(launch + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
                      for k in range(days)]
        # `episodes_per_day` is the day's total, split across the two arms
        self.per_day = max(int(episodes_per_day) // 2, 1)
        # the pool held for the run (common.parallel.EpisodePool, entered
        # by `run`); an hour's batch under two items per worker prices
        # in-process, the same answer either way
        self.workers = resolve_workers(workers)
        self.pool = EpisodePool(self.workers, serial_below=2 * self.workers)
        self.seed = int(seed)
        self.rng = np.random.default_rng([int(seed), 1])       # engineering's draws
        self.opened_by_day = {}
        bundle = load_bundle(cfg)
        self.model = bundle.model
        self.posterior = bundle.posterior
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
        self.launch_tau = self.posterior.tau()
        self.violations = {"price_rose_within_episode": 0, "below_cost": 0}
        self.lane_c_runs = []
        if hasattr(self.pricer, "bind"):
            self.pricer.bind(self)

    # ------------------------------------------------------------- days

    def run(self):
        with self.pool:
            for k, date in enumerate(self.dates):
                self._sample_day(k, date)
                self.run_hours(k, date, [(date, hour) for hour in range(24)])
                print(f"  day {k + 1}/{len(self.dates)} {date}: "
                      f"{self.opened_by_day.get(date, 0)} episodes opened, "
                      f"{len(self.open)} open", flush=True)
        return self.report()

    def run_hours(self, k, date, hours):
        """The shop's clock over `hours` -- (date, hour) pairs, day k's --
        then day `date` closes: at the lane hour (from the second day) the
        morning lane runs on yesterday's feed; every hour the openings due
        open and the batch is priced and sold (`_tick`)."""
        for day, hour in hours:
            if hour == self.lane_hour and k > 0:
                self.days.append(self.daily_lane(k))
            self._open_due(k, day, hour)
            self._tick(k, day, hour)
        self._close_day(date)

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
            for ep, res in zip(pilot, self.pricer(pilot, k, date, hour)):
                self._pilot_hour(ep, k, res)
        for ep in due:
            if ep["arm"] == "legacy":
                self._legacy_hour(ep, k)

    def _price_pilot(self, pilot, k, date, hour):
        """The default pricer: every due pilot state through engine.decide
        in the workers (engine.state.price_one, the one worker body)
        against one posterior snapshot and one tau
        (engine.state.batch_context, the one context; the digest computed
        once for the run). A pricer returns one `{"evt", "rejected"}` per
        episode in order -- `committed` True when it already wrote the
        event through the store itself (ops.price_batch does), so the
        shop does not emit it twice. Unlike shadow's rehearsal a simulated
        pilot IS suspended when the monitor says so: the context carries
        the record."""
        cats = {ep["template"]["category"] for ep in pilot}
        ctx = batch_context(self.cfg, self.posterior, self.model, cats, self.seed,
                            digest=self.digest)
        items = [(self._pilot_state(ep), (ep["episode_id"], ep["t"])) for ep in pilot]
        return self.pool.map(price_one, items, ctx)

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
        self.open_templates(k, date, [
            (arm, t, self._twin_of.pop((arm, t["template_id"], date), None))
            for arm, temps in (("pilot", pilot), ("legacy", legacy)) for t in temps])

    def open_templates(self, k, date, picks):
        """Open `picks` -- (arm, template, twin arm or None) -- on `date`:
        the ids and grids, the episode's shock (drawn at its first opening
        and shared by its twin, so the pair sees the same world), the two
        demand-rate features by the one home and one prediction per model
        for the day's openings. `_sample_day` picks for the rehearsal;
        tools.e2e_cycle hands the shop its own picks."""
        openings = []
        for arm, t, twin in picks:
            eid = f"sim|{arm}|{t['sku_id']}|{t['fc']}|{date}T{t['opening_hour']:02d}"
            if t["template_id"] not in self.shock_by_template:
                self.shock_by_template[t["template_id"]] = self.world.episode_shock()
            openings.append({"arm": arm, "episode_id": eid, "template": t,
                             "grid": hour_grid(date, t["opening_hour"], t["n_hours"]),
                             "t": 0, "q": t["q0"], "anchor": None, "day": k,
                             "shock": self.shock_by_template[t["template_id"]],
                             "twin": twin})
            self.busy.add((t["sku_id"], t["fc"]))
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
        """The engine's state for an open pilot episode's next hour, in the
        one spelling (engine.state.assemble_state): the world's r at the
        agent's scale, the agent's own forecast from this hour on."""
        t, tpl = ep["t"], ep["template"]
        date, hour = ep["grid"][t]
        return assemble_state({
            "episode_id": ep["episode_id"], "sku_id": tpl["sku_id"], "fc": tpl["fc"],
            "category": tpl["category"], "subcategory": tpl["subcategory"],
            "date": date, "hour_of_day": hour, "hours_remaining": tpl["n_hours"] - t,
            "q": int(ep["q"]), "original_price": tpl["original_price"],
            "cost": tpl["cost"], "current_discount": ep["anchor"],
        }, self.world.r_of(tpl) / self.world.r_scale, ep["mu_agent"][t:])

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
            # and an hour priced but never stored is graded (hourly_engine);
            # a pricer that committed through the store itself says so
            if not res.get("committed") and not self.store.emit_decision(evt):
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

    # ------------------------------------------------------------- feed

    def write_feed(self, dates, path):
        """The buffered feed rows of `dates`, in the source parquet's exact
        schema, to `path`; those days leave memory (the lane reads a day
        back from its parquet from here on). Returns the frame written."""
        rows = [r for d in dates for r in self.feed_by_day.pop(d, [])]
        df = World.feed_frame(rows)
        pq.write_table(pa.Table.from_pandas(df, schema=FEED_SCHEMA,
                                            preserve_index=False), path)
        return df

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
        feed = self.write_feed([yesterday], feed_path)
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
        _, rep = ingest_feed(store, feed, failures)
        # every COUNT the ingester reports (its example lists dropped), so a
        # gap it learns to name -- colliding decisions, an unusable row --
        # reaches the grader without a list here to extend; plus the
        # store's side: outcomes accepted and outcomes quarantined
        lane["ingest"] = {k_: v for k_, v in rep.items() if not isinstance(v, (list, dict))}
        lane["ingest"].update(emitted=rep["emitted"],
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
        # the routing, so a reader of the cells takes the widest std over
        # the cells a category reaches (an unrouted GLOBAL never narrows)
        lane["cell_of"] = dict(self.posterior.state["cell_of"])
        lane["tau_in_force"] = self.posterior.tau()
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
        self.model = load_bundle(cfg).model
        self._repredict_open_pilot()
        run = {"day": k, "date": self.dates[k], "bundle": bundle,
               "schedule_end": schedule_reaches(sched),
               "last_fitted_week": max(sched["by_week"]) if sched["by_week"] else None,
               "weeks_fitted": sched["weeks_fitted"],
               "weeks_unfitted_held_at_anchor": sched["weeks_unfitted_held_at_anchor"],
               "scope": sched["scope"], "prepared_rows": int(len(d)),
               "anchor_rows_by_arm": self._anchor_rows_by_arm(schedule_reaches(sched))}
        self.lane_c_runs.append(run)
        return run

    def _anchor_rows_by_arm(self, week):
        """The anchor rows (episodes.is_anchor_row, the one mask the level
        fit reads) in the trailing window the factors for `week` are fit
        on, per arm, over every simulated hour. Production's re-fit sees
        the whole feed -- every shelf in the FC, system-priced or not --
        and the legacy arm stands in for the rest of the shop; a pilot
        covering the whole FC has only the pilot column, so
        `pilot_alone_below_min` says whether that pilot's own rows would
        clear calibration_min_anchor_rows or hold the week at the anchor."""
        bm = self.cfg["baseline_model"]
        truth = self.truth()
        if truth.empty or not week:
            return None
        d_ref = {c: reference_discount(self.cfg, c) for c in truth.category.unique()}
        d = truth.assign(total_discount=truth.shelf_discount.astype(float),
                         d_ref=truth.category.map(d_ref))
        window, _ = episodes.trailing_weeks_window(
            d, week, bm["calibration_fit_trailing_weeks"])
        anchor = window[episodes.is_anchor_row(window, self.tier_step)]
        by_arm = {arm: int(n) for arm, n in anchor.groupby("arm").size().items()}
        pilot = by_arm.get("pilot", 0)
        return {"week": week, "pilot": pilot, "legacy": by_arm.get("legacy", 0),
                "min_anchor_rows": bm["calibration_min_anchor_rows"],
                "pilot_alone_below_min": pilot < bm["calibration_min_anchor_rows"]}

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
        """Both arms, per arm and paired (evaluate.pilot_grade.economics)
        over every simulated hour so far, or `truth`."""
        return economics(self.truth() if truth is None else truth)

    def learning(self):
        """The posterior's cells against the world's elasticity
        (evaluate.pilot_grade.learning)."""
        return learning(self.posterior.state["cells"], self.posterior.state["cell_of"],
                        self.launch_cells, self.world.epsilon_true)

    def level_tracking(self, decisions, truth=None):
        """The agent's level against the world's, per week
        (evaluate.pilot_grade.level_tracking); `decisions` is the store's
        list, loaded once by the caller."""
        return level_tracking(decisions, self.truth() if truth is None else truth)

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
                "tau_at_launch": self.launch_tau, "tau_now": self.posterior.tau(),
            },
            "learning": self.learning(),
            "economics": self.economics(truth),
            "lane_c": self.lane_c_runs,
            "level_tracking": self.level_tracking(decisions, truth),
            "days": self.days,
        }
        rep["expectations"] = grade(rep, self.cfg, self.grading)
        return rep


# ------------------------------------------------------------------ Lane B

class LaneBPricer:
    """The integration cycle's pricer: the hour's shelves as the top-of-hour
    SNAPSHOT in the feed's own schema (what engineering hands over, the
    shop assigning the episode ids as the producer does), plus the hour
    that just closed from the shop's feed rows, through ops.price_hour --
    which reads the ids as given, derives the anchors, builds the
    requests, prices (ops.price_batch) and commits every decision
    itself -- with the snapshot and response files engineering would see
    under `out_dir`. Bound to the shop by PilotSim (`bind`); every hour's
    counts are kept in `batches`."""

    def __init__(self, out_dir, workers=None, seed=0):
        self.out_dir, self.workers, self.seed = out_dir, workers, seed
        self.batches = []
        self.sim = None
        self.table_date, self.table = None, None    # the day's feature table
        self.tables = {}                            # date -> path written

    def bind(self, sim):
        self.sim = sim
        self.r_lookup = load_bundle(sim.cfg).r_lookup

    def features_for(self, date):
        """The day's feature table, built once per day the way the morning
        lane builds it (engine.state.ref_rate_table over the trailing
        history) and written where engineering would find it."""
        if self.table_date != date:
            self.table = ref_rate_table(self.sim._feature_history(date), date, self.sim.cfg)
            path = os.path.join(self.out_dir, "features", f"{date}.parquet")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.table.to_parquet(path, index=False)
            self.table_date, self.tables[date] = date, path
        return self.table

    def snapshot(self, pilot, date, hour):
        """The shelf at the top of the hour in the feed's schema (sales and
        ending stock unknown yet), then the rows of the hour that just
        closed as the shop's feed carries them."""
        rows = []
        for ep, s in zip(pilot, map(self.sim._pilot_state, pilot)):
            anchor = s["current_discount"]
            # the shop is the producer: it assigns the episode id
            rows.append({"episode_id": ep["episode_id"],
                         "date": date, "hour": hour, "skuseq": s["sku_id"], "fc": s["fc"],
                         "inventory": float(s["q"]),
                         "discount": None if anchor is None else float(anchor) * 100.0,
                         "units_sold": None, "normal_asp": float(s["original_price"]),
                         "final_price": None, "cogs_wo_vat": float(s["cost"]),
                         "ending_inventory": None,
                         "flc_window": float(episodes.window_counter(s["hours_remaining"])),
                         "category": s["category"], "subcategory": s["subcategory"]})
        prev = pd.Timestamp(date) + pd.Timedelta(hours=int(hour) - 1)
        prev_day = prev.strftime("%Y-%m-%d")
        rows += [dict(r, date=prev_day) for r in self.sim.feed_by_day.get(prev_day, [])
                 if int(r["hour"]) == prev.hour]
        return rows

    def __call__(self, pilot, k, date, hour):
        sim = self.sim
        tag = f"{date}T{hour:02d}"
        snapshot = self.snapshot(pilot, date, hour)
        snap_path = os.path.join(self.out_dir, "snapshots", f"{tag}.csv")
        os.makedirs(os.path.dirname(snap_path), exist_ok=True)
        pd.DataFrame(snapshot).to_csv(snap_path, index=False)
        response, events, rep = price_hour.run(
            sim.cfg, price_hour.snapshot_rows(snapshot), hour=tag,
            features=self.features_for(date), workers=self.workers, seed=self.seed,
            store=sim.store, model=sim.model, posterior=sim.posterior, r_lookup=self.r_lookup)
        dec_path = os.path.join(self.out_dir, "decisions", f"{tag}.csv")
        os.makedirs(os.path.dirname(dec_path), exist_ok=True)
        price_hour.write_response(response, dec_path)
        self.batches.append({"hour": tag, "snapshot_path": snap_path,
                             "decisions_path": dec_path, **rep})
        by_id = {e["decision_id"]: e for e in events}
        # a rejected shelf holds its price (the shop's defined fallback), a
        # priced one is already in the store: the shop must not emit it twice
        return [{"evt": None, "rejected": r["rejected"]} if r["rejected"] else
                {"evt": by_id[r["decision_id"]], "rejected": None, "committed": True}
                for r in response[:len(pilot)]]


PAIR_COLS = ("decision_id", "outcome_id", "sku_id", "fc", "date", "hour_of_day",
             "applied_discount", "applied_price", "offered_price", "price_matches",
             "units_sold", "starting_inventory", "ending_inventory",
             "adjustment_reason", "is_exploration")


def ingest_and_pair(store, feed_path, out_dir):
    """The daily lane's outcome side on the feed the shop wrote, as
    engineering reads it back: the outcomes built and emitted
    (`ingest_feed`, on the parquet), the paired tables exported under
    `out_dir`, every decision paired with its outcome (events.pairs
    .match_pairs) and the check that each outcome id is the one the feed
    row names (`outcome_id_of(hour_key(...))`). Returns the cycle's
    counts: `decisions`, `ingest`, `exports`, `pairs`, `price_mismatches`,
    `outcome_ids_follow_the_formula`, `paired_sample` (PAIR_COLS)."""
    decisions, rep = ingest_feed(store, pd.read_parquet(feed_path))
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
    return {
        "decisions": len(decisions),
        "ingest": rep,
        "exports": {k: {"path": p, "rows": n} for k, (p, n) in written.items()},
        "pairs": len(pairs),
        "price_mismatches": sum(1 for r in table if not r["price_matches"]),
        "outcome_ids_follow_the_formula": formula_holds,
        "paired_sample": table[:10],
    }
