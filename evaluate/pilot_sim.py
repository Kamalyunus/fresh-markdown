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
noise. Each template runs once under the pilot and once under the legacy
ramp (on consecutive days), so the economics read like-for-like.

The report grades a fixed list of expectations (`EXPECTATIONS`) and reads
the posterior against the truth it was learning. Every number is about the
WORLD it simulated (rule 19): a PASS says the machinery does what it claims
on a shop with that elasticity, never that the shop has it.

Run: python3 -m evaluate.pilot_sim --days 21 [--epsilon-true -1.2]
        [--fault mismatch:0.03 --fault demand_shock:10:0.6 ...]
"""

import argparse
import copy
import json
import os
import shutil

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
from fit import prepare_data
from fit.train_baseline import BaselineModel, fit_level_calibration, schedule_reaches
from ops import seal as seal_mod
from ops import status

SIM_DIR = "sim/pilot"
OUT = "reports/pilot_sim.json"
LANE_HOUR = 6                      # the daily cron's hour, after yesterday closed

# what a healthy run shows, each graded in `grade()`; the fault that turns
# an expectation around is named so a fault run reads PASS when it fires
EXPECTATIONS = (
    ("hourly_engine", "every hour with stock is priced; no state rejected"),
    ("price_monotone_within_episode", "no applied price rises within an episode"),
    ("never_below_cost", "no applied price under cost"),
    ("outcome_completeness", "outcomes land for >= shadow_gate.min_event_completeness of decisions (fault: missing)"),
    ("event_quality_gates", "duplicate/unmatched and price-mismatch gates pass every day (faults: duplicate, mismatch, discount_rounding)"),
    ("learning_moves_toward_truth", "every cell that updated ends closer to epsilon_true than it launched"),
    ("posterior_narrows", "every cell that updated ends with a smaller std"),
    ("tau_walks_on_spend", "tau moved and the last week's spend sits within the clip band of its budget"),
    ("stops_only_on_faults", "no stop condition fires without the fault that causes it; with it, the stop fires"),
    ("exploration_never_starves", "no three consecutive days with a budget in force and nothing forced (an empty affordable set is exploration off without a stop)"),
    ("assurance_holds", "reproduction and exploration never FAIL; dispersion never FAILs with the world's marginal untouched (correlation is reported: the world's rho is a knob)"),
    ("lane_c_keeps_the_schedule_current", "the weekly re-fit reaches every week priced, so --apply is never refused on calibration_schedule_current"),
    ("apply_ran_on_cadence", "--apply ran every learning.update_cadence_days and was refused only under a fault"),
)


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


# ---------------------------------------------------------------- simulator

class PilotSim:
    def __init__(self, cfg, world, sim_dir, config_path, days, episodes_per_day,
                 seed=0, lane_hour=LANE_HOUR, raw_path=None, prepared=None):
        self.cfg, self.world, self.sim_dir = cfg, world, sim_dir
        self.config_path, self.lane_hour = config_path, lane_hour
        self.raw_path = raw_path
        self.dates = [(pd.Timestamp(cfg["data"]["launch_date"]) + pd.Timedelta(days=k))
                      .strftime("%Y-%m-%d") for k in range(days)]
        self.per_day = int(episodes_per_day)
        self.rng = np.random.default_rng([int(seed), 1])       # engineering's draws
        self.agent_rng = np.random.default_rng([int(seed), 2])  # the agent's draws
        self.model = BaselineModel(cfg)
        self.posterior = PosteriorStore(cfg)
        self.store = EventStore(cfg)
        self.digest = provenance.config_fingerprint(cfg)["digest"]
        self.tier_step = cfg["pricing"]["tier_step"]
        # the feature service's history: the prepared extract plus every
        # simulated hour, in the prepared vocabulary
        hist = prepared.copy()
        hist["date"] = hist.date.astype(str)
        self.history = hist[["episode_id", "sku_id", "fc", "category", "date",
                             "hour_of_day", "starting_inventory", "units_sold",
                             "total_discount"]]
        self.sim_history = []
        self.open, self.pending, self.busy = [], {}, set()
        self.previous_sets = None
        self.feed_by_day, self.failures_by_day = {}, {}
        self.truth = []
        self.days = []
        self.rejected = {}
        self.launch_cells = copy.deepcopy(self.posterior.state["cells"])
        self.launch_tau = self.posterior.tau(cfg)
        self.violations = {"price_rose_within_episode": 0, "below_cost": 0}
        self.lane_c_runs = []

    # ------------------------------------------------------------- days

    def run(self):
        for k, date in enumerate(self.dates):
            self._sample_day(k, date)
            for hour in range(24):
                if hour == self.lane_hour and k > 0:
                    self.days.append(self.daily_lane(k))
                self._open_due(k, date, hour)
                self.posterior.reload()                # once per batch, as Lane B must
                tau = self.posterior.tau(self.cfg)
                for ep in list(self.open):
                    if ep["grid"][ep["t"]] != (date, hour):
                        continue
                    if ep["arm"] == "pilot":
                        self._pilot_hour(ep, k, tau)
                    else:
                        self._legacy_hour(ep, k)
        return self.report()

    def _sample_day(self, k, date):
        """2E templates a day, split into the two arms; the split swaps the
        next day so every template runs once under each policy."""
        if self.previous_sets is not None and k % 2 == 1:
            pilot, legacy = self.previous_sets[1], self.previous_sets[0]
        else:
            pool = [t for t in self.world.templates
                    if (t["sku_id"], t["fc"]) not in self.busy]
            keys, picked = set(), []
            for i in self.rng.permutation(len(pool)):
                t = pool[i]
                key = (t["sku_id"], t["fc"])
                if key in keys:
                    continue
                keys.add(key)
                picked.append(t)
                if len(picked) == 2 * self.per_day:
                    break
            pilot, legacy = picked[:self.per_day], picked[self.per_day:]
            self.previous_sets = (pilot, legacy)
        openings = []
        for arm, temps in (("pilot", pilot), ("legacy", legacy)):
            for t in temps:
                eid = f"sim|{arm}|{t['sku_id']}|{t['fc']}|{date}T{t['opening_hour']:02d}"
                openings.append({"arm": arm, "episode_id": eid, "template": t,
                                 "grid": hour_grid(date, t["opening_hour"], t["n_hours"]),
                                 "t": 0, "q": t["q0"], "anchor": None, "day": k,
                                 "shock": self.world.episode_shock()})
        # the two demand-rate features, point-in-time, by the one home
        stub = pd.DataFrame([{
            "episode_id": o["episode_id"], "sku_id": o["template"]["sku_id"],
            "fc": o["template"]["fc"], "category": o["template"]["category"],
            "date": date, "hour_of_day": o["template"]["opening_hour"],
            "starting_inventory": o["q"]} for o in openings])
        hist = self.history
        if self.sim_history:
            hist = pd.concat([hist, pd.DataFrame(self.sim_history)], ignore_index=True)
        since = (pd.Timestamp(date) - pd.Timedelta(
            days=self.cfg["baseline_model"]["ref_rate_window_days"] + 14)
        ).strftime("%Y-%m-%d")
        feats = ref_rate_features(hist[hist.date >= since], stub, self.cfg)
        for o in openings:
            o["features"] = feats[o["episode_id"]]
            o["mu_world"] = self.world.mu_ref_path(o["template"], o["grid"], o["features"])
            if o["arm"] == "pilot":
                o["mu_agent"] = self.world.mu_ref_path(o["template"], o["grid"],
                                                       o["features"], model=self.model)
        self.pending[date] = openings

    def _open_due(self, k, date, hour):
        for o in list(self.pending.get(date, [])):
            if o["template"]["opening_hour"] == hour:
                self.pending[date].remove(o)
                self.open.append(o)
                self.busy.add((o["template"]["sku_id"], o["template"]["fc"]))

    # ------------------------------------------------------------ hours

    def _pilot_hour(self, ep, k, tau):
        t, tpl = ep["t"], ep["template"]
        date, hour = ep["grid"][t]
        n = tpl["n_hours"]
        d_ref = reference_discount(self.cfg, tpl["category"])
        state = {
            "episode_id": ep["episode_id"], "sku_id": tpl["sku_id"], "fc": tpl["fc"],
            "category": tpl["category"], "subcategory": tpl["subcategory"],
            "date": date, "hour_of_day": hour, "hours_remaining": n - t,
            "q": int(ep["q"]), "original_price": tpl["original_price"],
            "cost": tpl["cost"], "r": self.world.r_of(tpl) / self.world.r_scale,
            "mu_ref_path": list(ep["mu_agent"][t:]),
            "current_discount": ep["anchor"],
        }
        applied = None
        try:
            evt = decide(state, self.posterior, self.store, self.cfg, self.agent_rng,
                         tau, self.model.version, config_digest=self.digest)
            applied = float(evt["applied_discount"])
            if ep["anchor"] is not None and applied < ep["anchor"] - dp_mod.TIER_EPS:
                self.violations["price_rose_within_episode"] += 1
            if evt["applied_price"] < tpl["cost"] - 1e-6:
                self.violations["below_cost"] += 1
        except StateRejected as e:
            self.rejected[str(e)] = self.rejected.get(str(e), 0) + 1
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
        self.truth.append({
            "episode_id": ep["episode_id"], "arm": ep["arm"], "date": date,
            "hour_of_day": hour, "starting_inventory": q, "units_sold": sold,
            "ending_inventory": ending, "original_price": tpl["original_price"],
            "offered_price": tpl["original_price"] * (1 - shelf), "cost": tpl["cost"],
            "category": tpl["category"], "fc": tpl["fc"], "sku_id": tpl["sku_id"],
            "dp_eligible": True, "shelf_discount": shelf, "mu_true": mu,
        })
        self.sim_history.append({
            "episode_id": ep["episode_id"], "sku_id": tpl["sku_id"], "fc": tpl["fc"],
            "category": tpl["category"], "date": date, "hour_of_day": hour,
            "starting_inventory": q, "units_sold": sold, "total_discount": shelf})
        ep["q"], ep["anchor"], ep["t"] = left, shelf, t + 1
        if close:
            self.open.remove(ep)
            self.busy.discard((tpl["sku_id"], tpl["fc"]))

    # ------------------------------------------------------- daily lane

    def daily_lane(self, k):
        """The morning of day k: yesterday's feed through the lane's own
        functions, in the order ops.advance --feed runs them."""
        cfg, today, yesterday = self.cfg, self.dates[k], self.dates[k - 1]
        lane = {"day": k, "date": yesterday, "lane_c": None}
        # Lane C on ops.advance's own rule: the schedule must reach one week
        # past the latest data's week (the week being priced); the first
        # morning always re-fits, since the sealed schedule is pre-launch
        cal = read_json(cfg["baseline_model"]["calibration_factor_path"]) or {}
        reaches = schedule_reaches(cal.get("schedule") or {}) or ""
        expected = (episodes.week_start(yesterday)
                    + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        if reaches < expected or not self.lane_c_runs:
            lane["lane_c"] = self.lane_c(k)
        lane["calibration_current"] = update.calibration_current(cfg, today)

        feed_path = os.path.join(self.sim_dir, "feed", f"{yesterday}.parquet")
        feed = _write_feed(self.feed_by_day.get(yesterday, []), feed_path)
        failures = None
        if self.failures_by_day.get(yesterday):
            failures = os.path.join(self.sim_dir, "feed", f"{yesterday}-failures.jsonl")
            with open(failures, "w") as f:
                for r in self.failures_by_day[yesterday]:
                    f.write(json.dumps(r) + "\n")
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
        lane["tau"] = {"before": tc["tau_before"], "after": tc["tau_after"],
                       "committed": walk.get("tau_committed", False),
                       "skipped": tc.get("skipped"),
                       "last_day": (tc["by_day"][-1] if tc.get("by_day") else None)}

        mon = monitor.build_report(EventStore(cfg), PosteriorStore(cfg), cfg)
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

        store = EventStore(cfg)
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
        st = status.collect(cfg, os.path.join(self.sim_dir, "reports"))
        lane["status_failing"] = st["failing"]

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
        re-sealed, the agent's model re-read."""
        cfg = self.cfg
        rows = [r for rows in self.feed_by_day.values() for r in rows]
        raw = pd.read_parquet(self.raw_path)[[f.name for f in FEED_SCHEMA]]
        full = pd.concat([raw, World.feed_frame(rows)], ignore_index=True) if rows else raw
        raw_sim = os.path.join(self.sim_dir, "raw_sim.parquet")
        pq.write_table(pa.Table.from_pandas(full, schema=FEED_SCHEMA,
                                            preserve_index=False), raw_sim)
        d, wf = prepare_data.load_and_filter(raw_sim, cfg)
        d.to_parquet(os.path.join(self.sim_dir, "prepared_sim.parquet"), index=False)
        prepare_data.write_manifest(cfg["data"]["split_manifest_path"], cfg, wf)
        fit_level_calibration(d, cfg)
        cal = read_json(cfg["baseline_model"]["calibration_factor_path"])
        sched = cal["schedule"]
        bundle = _seal(cfg, self.config_path, "weekly-refit")
        self.model = BaselineModel(cfg)
        run = {"day": k, "date": self.dates[k], "bundle": bundle,
               "schedule_end": schedule_reaches(sched),
               "last_fitted_week": max(sched["by_week"]) if sched["by_week"] else None,
               "weeks_fitted": sched["weeks_fitted"],
               "weeks_unfitted_held_at_1": sched["weeks_unfitted_held_at_1"],
               "scope": sched["scope"], "prepared_rows": int(len(d))}
        self.lane_c_runs.append(run)
        return run

    # ----------------------------------------------------------- report

    def economics(self):
        """Both arms through the one episode frame (metrics.episode_economics
        over metrics.settled), like-for-like: same templates, same world."""
        df = pd.DataFrame(self.truth)
        out = {}
        for arm, g in df.groupby("arm"):
            ep, excluded = metrics.settled(metrics.episode_economics(g))
            den = float(ep.denom.sum())
            units = float(ep.units_sold.sum() + ep.scrap.sum())
            out[arm] = {
                "episodes": int(len(ep)), "hours": int(len(g)),
                "il_absolute": round(float(ep.il.sum()), 1),
                "il_pct": round(float(ep.il.sum() / den), 6) if den > 0 else None,
                "il_pct_denominator": round(den, 1),
                "scrap_units": int(ep.scrap.sum()),
                "scrap_rate": round(float(ep.scrap.sum() / ep.supply.sum()), 4)
                if ep.supply.sum() > 0 else None,
                "sell_through": round(float(ep.units_sold.sum() / units), 4) if units else None,
                "margin": round(float(ep.margin.sum()), 1),
                "mean_discount": round(float(g.shelf_discount.mean()), 4),
                "excluded": excluded,
            }
        return out

    def learning(self):
        cells = self.posterior.state["cells"]
        cell_of = self.posterior.state["cell_of"]
        truth = self.world.epsilon_true
        out = {}
        for c, rec in cells.items():
            members = [cat for cat, cell in cell_of.items() if cell == c] or list(truth)
            eps = float(np.mean([truth[m] for m in members if m in truth]))
            launch = self.launch_cells[c]
            out[c] = {"members": members, "epsilon_true": round(eps, 4),
                      "launch_mean": launch["mean"], "launch_std": launch["std"],
                      "mean": rec["mean"], "std": rec["std"], "n_obs": rec["n_obs"],
                      "version": rec["version"],
                      "abs_error_at_launch": round(abs(launch["mean"] - eps), 4),
                      "abs_error_now": round(abs(rec["mean"] - eps), 4),
                      "accumulated_information": round(rec["accumulated_information"], 3)}
        return out

    def report(self):
        decisions = self.store.load_decisions()
        n_dec = len(decisions)
        forced = sum(1 for d in decisions if d["is_exploration"])
        rep = {
            "world": {
                "epsilon_true": self.world.epsilon_true, "r_scale": self.world.r_scale,
                "level_drift_per_day": self.world.drift, "faults": self.world.faults,
                "episode_shock_sd": self.world.episode_shock_sd,
                "templates": len(self.world.templates),
                "launch_date": self.cfg["data"]["launch_date"],
                "days": len(self.dates), "episodes_per_day_per_arm": self.per_day,
                "seed_note": "every figure is the simulated world's, not the shop's (rule 19)",
            },
            "config": provenance.config_fingerprint(self.cfg, "pilot_sim"),
            "engine": {
                "decisions": n_dec, "forced": forced,
                "forced_share": round(forced / n_dec, 4) if n_dec else None,
                "rejected": self.rejected,
                "rejected_total": int(sum(self.rejected.values())),
                "violations": self.violations,
                "pilot_hours": int(sum(1 for r in self.truth if r["arm"] == "pilot")),
                "tau_at_launch": self.launch_tau, "tau_now": self.posterior.tau(self.cfg),
            },
            "learning": self.learning(),
            "economics": self.economics(),
            "lane_c": self.lane_c_runs,
            "days": self.days,
        }
        rep["expectations"] = grade(rep, self.cfg)
        return rep


# ------------------------------------------------------------------ grading

def _verdict(ok, measured=True):
    if not measured:
        return "NOT MEASURED"
    return "PASS" if ok else "FAIL"


def grade(rep, cfg):
    """EXPECTATIONS against the run. A fault that is present turns its
    expectation around: the gate/stop it targets must fire."""
    faults = rep["world"]["faults"]
    days = rep["days"]
    eng = rep["engine"]
    out = []

    def add(name, ok, observed, measured=True):
        out.append({"name": name, "expected": dict(EXPECTATIONS)[name],
                    "verdict": _verdict(ok, measured), "observed": observed})

    add("hourly_engine", eng["decisions"] > 0 and eng["rejected_total"] == 0,
        {"decisions": eng["decisions"], "rejected": eng["rejected"]})
    add("price_monotone_within_episode", eng["violations"]["price_rose_within_episode"] == 0,
        eng["violations"])
    add("never_below_cost", eng["violations"]["below_cost"] == 0, eng["violations"])

    ing = [d["ingest"] for d in days]
    due = sum(i["decisions"] - i["decisions_outside_feed_range"] for i in ing)
    built = sum(i["outcomes_built"] for i in ing)
    gaps = sum(i["decisions_without_feed_row"] for i in ing)
    completeness = built / (built + gaps) if built + gaps else None
    floor = cfg["monitoring"]["shadow_gate"]["min_event_completeness"]
    expect_gap = "missing" in faults
    ok = completeness is not None and ((completeness >= floor) != expect_gap)
    add("outcome_completeness", ok,
        {"completeness": round(completeness, 4) if completeness is not None else None,
         "floor": floor, "fault_expects_a_gap": expect_gap, "decisions_due": due},
        measured=completeness is not None)

    gate_fail_days = [d["date"] for d in days if not all(d["gates"].values())]
    expect_fail = any(f in faults for f in ("duplicate", "mismatch", "discount_rounding"))
    add("event_quality_gates", bool(gate_fail_days) == expect_fail,
        {"days_a_gate_failed": gate_fail_days, "fault_expects_a_failure": expect_fail},
        measured=bool(days))

    learned = {c: r for c, r in rep["learning"].items() if r["version"] > 0}
    add("learning_moves_toward_truth",
        all(r["abs_error_now"] < r["abs_error_at_launch"] for r in learned.values()),
        {c: {"launch": r["abs_error_at_launch"], "now": r["abs_error_now"]}
         for c, r in rep["learning"].items()}, measured=bool(learned))
    add("posterior_narrows", all(r["std"] < r["launch_std"] for r in learned.values()),
        {c: {"launch_std": r["launch_std"], "std": r["std"]}
         for c, r in rep["learning"].items()}, measured=bool(learned))

    # the controller moves tau by at most the clip per day, so a launch tau
    # far from this world's budget needs a week or more to arrive: graded
    # on the last seven walked days, and only once there are seven
    walked = [d["tau"]["last_day"] for d in days
              if d["tau"].get("last_day") and d["tau"].get("committed")]
    lo, hi = cfg["exploration"]["tau_adjust_clip"]
    week = walked[-7:]
    ratios = [w["spend"] / w["budget"] for w in week if w["budget"] > 0]
    ratio = float(np.mean(ratios)) if ratios else None
    moved = eng["tau_now"] != eng["tau_at_launch"]
    add("tau_walks_on_spend", moved and ratio is not None and lo <= ratio <= hi,
        {"tau_at_launch": eng["tau_at_launch"], "tau_now": eng["tau_now"],
         "last_week_spend_over_budget": round(ratio, 3) if ratio is not None else None,
         "band": [lo, hi], "days_walked": len(walked)}, measured=len(walked) >= 7)

    fired = {}
    for d in days:
        for name, v in d["stops"].items():
            if v is True:
                fired.setdefault(name, []).append(d["date"])
    causes = {"duplicate_or_unmatched": ("duplicate", "missing"),
              "price_mismatch": ("mismatch", "discount_rounding"),
              "scrap_deterioration_pct": ("demand_shock",),
              "margin_deterioration_pct": ("demand_shock",)}
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
    expected_missing = [n for n, fs in causes.items()
                        if any(f in faults for f in fs) and n not in fired
                        and (n in ("duplicate_or_unmatched", "price_mismatch")
                             or (n == "scrap_deterioration_pct" and guardrail_ready))]
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
    starved, streak, worst = [], 0, 0
    prev = 0
    for d in days:
        forced_today = d["learning"]["forced_decision_count"] - prev
        prev = d["learning"]["forced_decision_count"]
        streak = streak + 1 if forced_today == 0 and not d["suspended"] else 0
        worst = max(worst, streak)
        if streak >= 3:
            starved.append(d["date"])
    add("exploration_never_starves", not starved,
        {"days_starved": starved, "longest_streak": worst,
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
    # reads as a shape problem: dispersion is excused under a gate fault
    marginal_moved = (rep["world"]["r_scale"] != 1.0
                      or rep["world"].get("episode_shock_sd", 0) > 0
                      or expect_fail or "missing" in faults)
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
    fault_refusal = expect_fail or "missing" in faults
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
    print(f"{w['days']} days from {w['launch_date']}, {w['episodes_per_day_per_arm']} "
          f"episodes/day/arm from {w['templates']} templates")
    print(f"engine: {e['decisions']:,} decisions, {e['forced']:,} forced "
          f"({e['forced_share']}), {e['rejected_total']} rejected; "
          f"tau {e['tau_at_launch']} -> {e['tau_now']}")
    for c, r in rep["learning"].items():
        print(f"  [{c}] eps_true {r['epsilon_true']:+.3f}  mean {r['launch_mean']:+.3f} -> "
              f"{r['mean']:+.3f}  std {r['launch_std']:.3f} -> {r['std']:.3f}  "
              f"(v{r['version']}, {r['n_obs']} outcomes)")
    for arm, x in rep["economics"].items():
        print(f"  {arm:7s} IL {x['il_absolute']:>12,.0f}  IL% {x['il_pct']}  "
              f"scrap_rate {x['scrap_rate']}  sell-through {x['sell_through']}  "
              f"mean discount {x['mean_discount']}")
    for x in rep["expectations"]:
        print(f"  {x['verdict']:<12} {x['name']}")
    print(f"wrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evaluate.pilot_sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default="data/prepared.parquet",
                    help="the prepared extract: episode templates and the "
                         "feature service's history")
    ap.add_argument("--raw", default="data/flc_raw.parquet",
                    help="the raw extract Lane C's weekly re-fit extends")
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--launch-date", default=None,
                    help="default: the day after the extract's last day")
    ap.add_argument("--episodes-per-day", type=int, default=40,
                    help="per arm; each template runs once under each")
    ap.add_argument("--epsilon-true", type=float, default=-1.2,
                    help="the world's elasticity, every category")
    ap.add_argument("--epsilon-true-map", default=None,
                    help='JSON {category: epsilon}, overrides --epsilon-true')
    ap.add_argument("--r-scale", type=float, default=1.0)
    ap.add_argument("--episode-shock-sd", type=float, default=0.0,
                    help="log-sd of a demand shock every hour of an episode "
                         "shares (0: independent hours, rho ~ 0; above 0 the "
                         "world is over-dispersed against the frozen r, which "
                         "was fitted without it)")
    ap.add_argument("--level-drift", type=float, default=1.0,
                    help="world demand multiplier per day (0.99: 1%%/day decay)")
    ap.add_argument("--fault", action="append", default=[],
                    help="name[:arg]; one of " + ", ".join(sorted(FAULTS)))
    ap.add_argument("--templates-from", default=None,
                    help="opening date the templates are sampled from "
                         "(default: the hold-out start)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-dir", default=SIM_DIR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    prepared = pd.read_parquet(args.input)
    launch = args.launch_date or (pd.Timestamp(prepared.date.astype(str).max())
                                  + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    eps = json.loads(args.epsilon_true_map) if args.epsilon_true_map else args.epsilon_true
    world = World(cfg, prepared, eps, seed=args.seed, opened_from=args.templates_from,
                  r_scale=args.r_scale, level_drift_per_day=args.level_drift,
                  faults=parse_faults(args.fault),
                  episode_shock_sd=args.episode_shock_sd)
    cfg_sim, config_path = build_workspace(cfg, args.sim_dir, launch)
    sim = PilotSim(cfg_sim, world, args.sim_dir, config_path, args.days,
                   args.episodes_per_day, seed=args.seed, raw_path=args.raw,
                   prepared=prepared)
    rep = sim.run()
    write_json(args.out, rep)
    _print(rep, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
