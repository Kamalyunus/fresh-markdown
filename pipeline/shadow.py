"""Shadow-mode harness (docs/design.md section 5.13).

Runs the full production decision path against observed data; NO prices are
applied. Outcomes are stamped execution_status="shadow_not_applied" and are
ineligible for pipeline.update. Runs on `data.holdout` BY DEFAULT -- frozen
artifacts are fit up to split.test_end, so `--all` is partly in-sample and
the report says so. Exit gate: event completeness, matched decision rate,
and ZERO cost-floor violations."""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config, deff, ConfigError
from common import episodes
from common.parallel import map_episodes
from common.episodes import adjustment_reason
from bootstrap.train_baseline import BaselineModel
from bootstrap.fit_dispersion import lookup_r
from events.store import EventStore
from pricing import explore
from pricing.demand import expected_min_demand_inventory_vec
from inference.decide import decide, StateRejected
from pricing.posterior import PosteriorStore

SHADOW_STATUS = "shadow_not_applied"

# Default window: the hold-out is the only span no frozen artifact was fit
# on, so honesty is the default rather than a flag someone must remember.
HOLDOUT_BASIS = "holdout"




def _require_shadow_config(cfg, backtest_path="reports/backtest.json",
                           shadow_path="reports/shadow.json", why=None):
    """Fallback path only: when the run cannot derive its own launch tau
    (`why` says why), the config paste must exist and match its source."""
    prefix = f"tau derivation unavailable ({why}); " if why else ""
    if cfg["exploration"]["tau_initial"] is None:
        raise ConfigError(
            "shadow phase blocked: " + prefix + "exploration.tau_initial is "
            "null. Give the run a pre-window week (the default hold-out run "
            "does) or paste a derived tau.")

    # Non-null is not enough: tau_initial is hand-pasted and decides day-one
    # spend, before the controller has any spend to correct from.
    def _read(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None
    stale = explore.tau_provenance_error(cfg, _read(backtest_path),
                                         _read(shadow_path))
    if stale:
        raise ConfigError("shadow phase blocked by a stale tau: "
                          + prefix + stale)



def pre_window_il_history(d, cfg, before):
    """Realised legacy IL by close day for episodes that CLOSED before the
    window -- the trailing budget base production holds at launch. Without it
    the window's first day reads budget 0 (empty history), which is an
    ABSENCE OF SIGNAL, not an overspend. Only the budget_il_window_days
    before `before` matter."""
    if before is None or d.empty:
        return {}
    start = pd.Timestamp(str(before))
    window = int(cfg["exploration"]["budget_il_window_days"])
    lo = (start - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
    hi = start.strftime("%Y-%m-%d")
    g = d.sort_values(["episode_id", "date", "hour_of_day"])
    last = g.groupby("episode_id").tail(1)
    close = last.date.astype(str)
    last = last[(close >= lo) & (close < hi)]
    if last.empty:
        return {}
    rows = g[g.episode_id.isin(set(last.episode_id))]
    disc = ((rows.original_price * rows.total_discount * rows.units_sold)
            .groupby(rows.episode_id).sum())
    kind = episodes.classify_last(last)
    leftover = episodes.leftover_units(last.starting_inventory, last.units_sold)
    scrap = (last.cost.to_numpy() * leftover.to_numpy()
             * (kind == episodes.COMPLETED).to_numpy())
    closed = (kind != episodes.NOT_CLOSED).to_numpy()
    out = {}
    for eid, day, ok, sc in zip(last.episode_id.to_numpy(),
                                last.date.astype(str).to_numpy(), closed, scrap):
        if ok:
            out[day] = out.get(day, 0.0) + float(disc.get(eid, 0.0)) + float(sc)
    return out


# episodes are independent (tau is fixed; the controller walk is
# post-processing), so the unit of work is one episode and all parallelise
EP_COLS = ("hour_of_day", "sku_id", "fc", "category", "subcategory",
           "starting_inventory", "ending_inventory", "units_sold",
           "total_discount", "original_price", "cost", "r_val",
           "mu_ref_hat", "date", "is_observed")


def weekly_refit_schedule(d_full, cfg, model, r_lookup, start, end):
    """Re-fit the level factors for each week of the shadow window, the way
    production's weekly cron would.

    THIS IS WHAT THE ARTIFACT CANNOT CARRY. The calibration schedule is built
    over pre-launch data and stops at `split.test_end`, so every hold-out row
    falls back to the frozen anchor -- shadow therefore measures "launch and
    never re-calibrate". Production does re-calibrate weekly, and the honest
    comparison needs both. Fitting it HERE rather than in the artifact keeps
    the pre-launch bundle clean of hold-out rows (hard rule 16): shadow is a
    forward replay, so at week k it may fit on weeks < k, which is exactly
    what the cron has available on that Monday.

    Returns {week_start: {cell: factor}} for the weeks the window spans.
    """
    from bootstrap.train_baseline import _solve_level_factors
    from bootstrap.prepare_data import population

    bm = cfg["baseline_model"]
    weeks_back = bm["calibration_fit_trailing_weeks"]
    grain = model.calibration_grain
    scope = population(d_full, cfg).copy()
    dates = pd.to_datetime(scope.date)
    wk = dates.dt.to_period("W")
    lo_w = pd.Timestamp(start).to_period("W").start_time
    hi_w = pd.Timestamp(end).to_period("W").start_time

    out, coverage = {}, []
    for w in sorted(wk.unique()):
        w0 = w.start_time
        if w0 < lo_w or w0 > hi_w:
            continue
        lo = w0 - pd.Timedelta(weeks=weeks_back)
        # STRICTLY BEFORE this week: no look-ahead inside the replay
        window = scope[(wk.dt.start_time >= lo) & (wk.dt.start_time < w0)]
        weeks_seen = int(wk[(wk.dt.start_time >= lo)
                            & (wk.dt.start_time < w0)].nunique())
        fitted = _solve_level_factors(
            window.copy(), cfg, model, grain, bm["calibration_shrinkage_units"],
            bm["calibration_min_anchor_rows"], cfg["pricing"]["tier_step"],
            cfg["pricing"]["negbin_max_k"], r_lookup) if len(window) else None
        if fitted is None:                 # too thin: that week keeps the anchor
            coverage.append({"week": str(w0.date()), "fitted": False})
            continue
        out[str(w0.date())] = fitted[0]
        coverage.append({"week": str(w0.date()), "fitted": True,
                         "fit_rows": int(len(window)),
                         "weeks_in_window": weeks_seen,
                         "partial": weeks_seen < weeks_back})
    return out, coverage


def _prepare_items(d, cfg, model, r_lookup):
    """Extend to the full window BEFORE predicting (early sell-out must not
    shorten the DP horizon), then pack per-episode arrays for _shadow_one."""
    carry = [c for c in d.columns if c not in
             ("episode_id", "date", "hour_of_day", "hours_remaining",
              "starting_inventory", "ending_inventory", "units_sold")]
    d = episodes.extend_to_window(d, carry, cfg["data"]["max_window_hours"])
    d = d.sort_values(["episode_id", "date", "hour_of_day"]).copy()
    d["mu_ref_hat"] = model.predict_mu_ref(d)
    d["r_val"] = [lookup_r(r_lookup, s, c)
                  for s, c in zip(d.subcategory, d.category)]
    groups = list(d.groupby("episode_id", sort=False))
    items = [dict({c: g[c].to_numpy() for c in EP_COLS}, episode_id=eid)
             for eid, g in groups]
    return d, groups, items


def derive_tau0(d_full, cfg, start, model, posterior, r_lookup, il_history,
                seed=0, max_episodes=None, workers=None):
    """Launch tau for THIS run: the run's own anchored decision path over the
    trailing budget_il_window_days before the window -- the same span the
    day-one budget base comes from. Replaces the config paste, which carried
    two staleness modes: an old backtest's number, and the exploit-vs-anchored
    path mismatch (different affordable sets, so the same tau buys a
    different spend). Returns the report block; tau_initial is None when the
    week is too thin to bisect on, and the caller falls back to the paste."""
    from bootstrap.prepare_data import population
    window = int(cfg["exploration"]["budget_il_window_days"])
    floor = int(cfg["exploration"]["tau0_derivation_min_decisions"])
    start_ts = pd.Timestamp(str(start))
    lo = (start_ts - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
    hi = (start_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    block = {
        "basis": (f"anchored decision path over episodes opened in the "
                  f"trailing {window} days before the window -- the same "
                  "span as the day-one budget base"),
        "week": [lo, hi],
        "tau_initial": None,
        "decisions": 0,
        "min_decisions": floor,
        "trailing_il_seeded_days": len(il_history),
        "fallback": True,
    }
    pre = episodes.window_slice(d_full, lo, hi)
    if not pre.empty:
        pre = population(pre, cfg, "dp_eligible")
    if pre.empty:
        block["note"] = "no DP-eligible episodes opened in the pre-window week"
        return block

    # day-one budget: production's own quantity on the seeded trailing base
    cells = posterior.state["cells"]
    widest_std = max(rec["std"] for rec in cells.values())
    budget = explore.budget_today(
        explore.trailing_daily_il(il_history, start_ts.strftime("%Y-%m-%d"),
                                  cfg), widest_std, cfg)
    block["day_one_budget"] = round(budget, 1)
    if budget <= 0:
        block["note"] = ("no realised IL in the trailing window -- a zero "
                         "budget cannot price a tau")
        return block

    pre_ids = pre.episode_id.unique()
    n_pop = len(pre_ids)
    # decoupled from the window's sample draw, same reproducibility contract
    rng = np.random.default_rng([int(seed), 1])
    if max_episodes and n_pop > max_episodes:
        keep = rng.choice(pre_ids, max_episodes, replace=False)
        pre = pre[pre.episode_id.isin(keep)]
    n_days = max((pd.Timestamp(pre.date.max())
                  - pd.Timestamp(pre.date.min())).days + 1, 1)
    _, groups, items = _prepare_items(pre, cfg, model, r_lookup)
    # tau None: nothing explores, and the ledger does not care -- spreads are
    # recorded before the draw, independent of the tau in force
    ctx = {"cfg": cfg, "tau": None, "model_version": model.version,
           "seed": seed, "cal_grain": model.calibration_grain,
           "cells": {str(c): posterior.get(c) for c in pre.category.unique()}}
    ledger = explore.SpreadLedger()
    for out in map_episodes(_shadow_one, items, ctx, workers):
        for day, costs in out["spreads"]:
            ledger.add(day, costs)

    block.update(decisions=ledger.decisions, episodes=len(groups),
                 episodes_population=int(n_pop),
                 days_with_decisions=len(ledger.days), days=n_days)
    if ledger.decisions < floor:
        block["note"] = (f"{ledger.decisions} decisions in the pre-window "
                         f"week, below the {floor} floor -- too thin to "
                         "bisect a launch tau on")
        return block
    # a sample carries only its fraction of the population's spend, so the
    # bisection targets the budget scaled by the same fraction
    frac = len(groups) / n_pop
    tau0 = ledger.solve_tau(budget * frac, n_days=n_days)
    if not tau0:
        block["note"] = "bisection found no positive tau at this budget"
        return block
    block.update(
        tau_initial=round(float(tau0), 2), fallback=False,
        sample_fraction=round(frac, 4),
        budget_target=round(budget * frac, 1),
        implied_daily_spend=round(ledger.implied_daily_spend(tau0, n_days), 1),
        q_spread_distribution=ledger.distribution(),
        note=("derived on this run's own anchored path over the week the "
              "day-one budget reads, so the backtest's exploit-vs-anchored "
              "mismatch does not apply and the day-one controller trace is "
              "an out-of-sample test of it. For the pilot, paste this into "
              "exploration.tau_initial; tau_provenance_error accepts this "
              "block as the source."))
    return block


def _controller_trace(ledger, il_by_day, tau0, widest_std, cfg, window_days=None,
                      sampled_episodes=None, population_episodes=None,
                      max_days=60):
    """Day-by-day tau-controller walk: does the pilot survive its first week?
    The controller reads only the day just closed, so a too-generous launch
    tau cannot be corrected before day one's spend is on the books. Spend per
    day is EXPECTED spend at the tau in force (a counterfactual tau path)."""
    stop_at = cfg["monitoring"]["stop_conditions"]["exploration_cost_vs_budget"]
    days = ledger.days
    order = sorted(range(len(days)), key=lambda i: days[i])
    tau, rows, first_within, suspend_days = float(tau0), [], None, 0
    for rank, i in enumerate(order[:max_days]):
        day = days[i]
        spend = float(ledger.spend_by_day(tau)[i])
        # production's budget for the day: a share of TRAILING realised IL,
        # never the same day's own (unknown until its episodes close)
        budget = explore.budget_today(
            explore.trailing_daily_il(il_by_day, day, cfg), widest_std, cfg)
        over = (spend / budget) if budget > 0 else None
        fired = bool(over is not None and over > stop_at)
        suspend_days += int(fired)
        if over is not None and over <= 1.0 and first_within is None:
            first_within = rank + 1
        rows.append({"day": day, "tau": round(tau, 2),
                     "spend": round(spend, 1), "budget": round(budget, 1),
                     "over_budget": round(over, 2) if over is not None else None,
                     "stop_condition_fires": fired})
        # the controller runs at the operator gate, on the day just closed.
        # A ZERO budget (no trailing IL history yet) is an absence of signal,
        # not an overspend -- calibrating on it would halve tau for nothing
        if budget > 0:
            tau = explore.tau_next(tau, budget, spend, cfg)
    return {
        "tau_start": round(float(tau0), 2),
        "tau_end": round(tau, 2),
        "by_day": rows,
        # three distinct day counts -- calendar span, days with decisions,
        # days walked -- none interchangeable, especially on a sample
        "window_days": int(window_days) if window_days else len(ledger.days),
        "days_with_decisions": len(ledger.days),
        "days_simulated": len(rows),
        "days_truncated": max(len(ledger.days) - len(rows), 0),
        "days_stop_condition_fires": suspend_days,
        "first_day_within_budget": first_within,
        "clip": cfg["exploration"]["tau_adjust_clip"],
        # the ONE figure a sample degrades: this series divides the sample
        # across the window's days (everything else reads rates or is invariant)
        "episodes_per_day_sampled": round(
            sampled_episodes / max(len(rows), 1), 1) if sampled_episodes else None,
        "episodes_per_day_population": round(
            population_episodes / max(window_days or len(rows), 1), 1)
            if population_episodes else None,
        "verdict": (
            "no days simulated" if not rows else
            f"exploration suspends on day 1 and stays suspended for "
            f"{suspend_days} of {len(rows)} days -- the controller cannot "
            "correct a tau it has not yet seen spend from"
            if rows[0]["stop_condition_fires"] else
            f"survives launch; {suspend_days} of {len(rows)} days would fire "
            "the stop condition" if suspend_days else
            "survives launch; the stop condition never fires"),
        "note": ("Expected spend at the tau in force each day, so this is the "
                 "path a pilot launched at tau_start would have walked. Run "
                 "it again with tau_initial set to tau_recommended to confirm "
                 "the launch value clears day 1."
                 + (" ON A SAMPLE the day-to-day movement mixes real "
                    "volatility with sampling noise, and the controller will "
                    "look jumpier than it is; the pooled "
                    "exploration_budget_would_be.spend_over_budget is "
                    "sample-invariant and is the figure to quote. Raise "
                    "--max-episodes if reading this series closely."
                    if sampled_episodes and population_episodes
                    and sampled_episodes < population_episodes else "")
                 + (f" TRUNCATED: {len(ledger.days) - len(rows)} later days "
                    f"not walked (cap {max_days})."
                    if len(rows) < len(ledger.days) else "")),
    }


class _BufferStore:
    """Buffers decision events instead of writing them: workers must not
    touch the event store the gate measures. The parent commits every event
    through the real store, in episode order."""

    def __init__(self):
        self.decisions = []

    def emit_decision(self, event):
        self.decisions.append(event)
        return True


class _FrozenCells:
    """Read-only posterior cells, pre-resolved in the parent so the cell map
    (including the fallback to the global cell) is applied exactly once, by
    the real store. Nothing updates the posterior during a shadow run."""

    def __init__(self, by_category):
        self._by_category = by_category

    def get(self, category):
        return self._by_category[str(category)]


def _episode_seed(seed, episode_id):
    """A generator per episode, seeded from its id: draws are reproducible
    and independent of visit order, so serial and parallel runs agree."""
    h = hashlib.blake2b(str(episode_id).encode(), digest_size=8).digest()
    return np.random.default_rng([int(seed), int.from_bytes(h, "big")])


def _shadow_one(ep, ctx):
    """Price one episode's hours. Pure: no store, no shared generator.
    Returns everything the parent folds in, including events to commit.
    `ep` carries rows as arrays, not a DataFrame (per-hour .iloc was the
    second-largest cost after the DP)."""
    cfg, tau = ctx["cfg"], ctx["tau"]
    posterior = _FrozenCells(ctx["cells"])
    store = _BufferStore()
    rng = _episode_seed(ctx["seed"], ep["episode_id"])
    n = len(ep["hour_of_day"])

    out = {
        "events": [], "rejected": {}, "spreads": [],
        "cost_floor_violations": 0, "n_forced": 0, "empty_affordable": 0,
        "would_be_cost": 0.0, "raw_information": 0.0,
        # the two factors behind raw_information, kept apart: how FAR a forced
        # price moved from the reference, and how much demand was there to
        # measure it on. Information is quadratic in the first.
        "abs_log_ratio": 0.0, "forced_mu": 0.0, "forced_discount_gap": 0.0,
        "rec_disc": 0.0, "leg_disc": 0.0, "differs": 0,
        "ep_discount_cost": 0.0, "latencies": [],
        # `date` and `cell` ride along so the drift ratio can be recomputed
        # under a WEEKLY RE-FIT calibration as well as the frozen anchor --
        # a level factor is a per-(cell, week) multiplier, so swapping it is
        # an exact rescale of mu, no re-prediction needed
        "drift": {"mu": [], "r": [], "q": [], "sold": [], "date": [],
                  "cell": []},
        "last_row": None,
    }
    anchor, last_obs = None, None

    for t in range(n):
        if not ep["is_observed"][t]:      # window tail: no outcome to record
            continue
        q = int(ep["starting_inventory"][t])
        if q <= 0:                        # restock gap: no decision this hour
            anchor = float(ep["total_discount"][t])
            continue

        row_day = str(ep["date"][t])
        legacy_d = float(ep["total_discount"][t])
        sold = int(ep["units_sold"][t])
        ending = int(ep["ending_inventory"][t])

        # What the business did, recorded UNconditioned on decision success
        # (gating on it under-counted IL and mispointed the scrap classifier);
        # accumulated on the episode and attributed to its close day.
        out["ep_discount_cost"] = out.get("ep_discount_cost", 0.0) + \
            float(ep["original_price"][t]) * legacy_d * sold
        last_obs = (q, sold, float(ep["cost"][t]), ending, row_day)

        state = {
            "episode_id": ep["episode_id"], "sku_id": int(ep["sku_id"][t]),
            "fc": ep["fc"][t], "category": ep["category"][t],
            "subcategory": ep["subcategory"][t],
            "hour_of_day": int(ep["hour_of_day"][t]),
            "hours_remaining": n - t, "q": q,
            "original_price": float(ep["original_price"][t]),
            "cost": float(ep["cost"][t]), "r": float(ep["r_val"][t]),
            "mu_ref_path": list(ep["mu_ref_hat"][t:]),
            "current_discount": anchor,
        }
        spreads_here = []
        try:
            evt = decide(state, posterior, store, cfg, rng, tau,
                         ctx["model_version"],
                         spread_sink=spreads_here.append)
        except StateRejected as e:
            out["rejected"][str(e)] = out["rejected"].get(str(e), 0) + 1
            anchor = float(ep["total_discount"][t])
            continue

        for costs in spreads_here:
            out["spreads"].append((row_day, costs))
        out["latencies"].append(evt["solver_latency_s"])
        if evt["applied_price"] < evt["cost"] - 1e-6:
            out["cost_floor_violations"] += 1
        if evt["is_exploration"]:
            out["n_forced"] += 1
            out["would_be_cost"] += evt["exploration_cost"]
            # would-be learning yield: MUST match pipeline.update's NB Fisher
            # information mu * L^2 * r/(r+mu), ratio vs the REFERENCE discount
            lr = np.log((1 - evt["applied_discount"])
                        / (1 - evt["reference_discount"]))
            mu_rec = max(ep["mu_ref_hat"][t] * np.exp(
                evt["epsilon_posterior_mean"] * lr), cfg["pricing"]["demand_floor"])
            r_ep = evt["dispersion_r"]
            out["raw_information"] += mu_rec * lr ** 2 * r_ep / (r_ep + mu_rec)
            out["abs_log_ratio"] += abs(lr)
            out["forced_mu"] += mu_rec
            out["forced_discount_gap"] += abs(evt["applied_discount"]
                                              - evt["reference_discount"])
        if evt["affordable_set_size"] == 0:
            out["empty_affordable"] += 1
        out["rec_disc"] += evt["applied_discount"]
        out["leg_disc"] += legacy_d
        if abs(evt["applied_discount"] - legacy_d) > 1e-9:
            out["differs"] += 1

        # outcome = what actually happened under the LEGACY price
        outcome = {
            "event": "outcome",
            "outcome_id": f"shadow-{evt['decision_id']}",
            "decision_id": evt["decision_id"],
            "units_sold": sold, "starting_inventory": q,
            "ending_inventory": ending,
            "applied_price": float(ep["original_price"][t] * (1 - legacy_d)),
            "is_stockout": sold >= q,
            "execution_status": SHADOW_STATUS,
        }
        # unreconciled inventory without a documented reason is quarantined,
        # so the legitimate breaks (restock, write-off, shrink) must be named
        reason = adjustment_reason(q, sold, ending)
        if reason:
            outcome["adjustment_reason"] = reason
        out["events"].append((store.decisions[-1], outcome))

        # drift check at the legacy price (the price the outcome saw)
        eps = evt["epsilon_posterior_mean"]
        ratio = (1 - legacy_d) / (1 - evt["reference_discount"])
        out["drift"]["mu"].append(max(ep["mu_ref_hat"][t] * ratio ** eps,
                                      cfg["pricing"]["demand_floor"]))
        out["drift"]["r"].append(float(ep["r_val"][t]))
        out["drift"]["q"].append(q)
        out["drift"]["sold"].append(sold)
        out["drift"]["date"].append(str(ep["date"][t]))
        out["drift"]["cell"].append(str(ep[ctx["cal_grain"]][t]))

        anchor = legacy_d                 # reality's price is the next anchor

    # scrap is end-of-episode: keep the final row and classify after the
    # loop, all episodes together in one frame
    if last_obs is not None:
        start, sold_last, unit_cost, ending_last, close_day = last_obs
        out["last_row"] = {"episode_id": ep["episode_id"],
                           "discount_cost": out.get("ep_discount_cost", 0.0),
                           "starting_inventory": start,
                           "units_sold": sold_last, "cost": unit_cost,
                           "ending_inventory": ending_last,
                           "close_day": close_day}
    return out


def run_shadow(d, cfg, events_root=None, seed=0, max_episodes=None,
               prior_il_by_day=None, pre_window_frame=None, window_start=None,
               window_basis=HOLDOUT_BASIS, workers=None):
    # precondition, inside run_shadow so a programmatic caller cannot skip it
    from bootstrap.prepare_data import population
    d = population(d, cfg, "dp_eligible")
    if d.empty:
        raise RuntimeError("no DP-eligible episodes in this window")
    model = BaselineModel(cfg)
    posterior = PosteriorStore(cfg)
    with open(cfg["dispersion"]["r_lookup_path"]) as f:
        r_lookup = json.load(f)
    store = EventStore(cfg, root=events_root or cfg["events"]["shadow_store_dir"])
    rng = np.random.default_rng(seed)

    if max_episodes is None:
        max_episodes = cfg["monitoring"]["shadow_gate"]["sample_episodes"]

    # tau in force: derived on the trailing pre-window week when one exists;
    # the config paste (with full provenance checks) only as fallback
    tau_deriv = None
    if pre_window_frame is not None and window_start is not None:
        tau_deriv = derive_tau0(pre_window_frame, cfg, window_start, model,
                                posterior, r_lookup,
                                dict(prior_il_by_day or {}), seed=seed,
                                max_episodes=max_episodes, workers=workers)
    if tau_deriv is not None and tau_deriv["tau_initial"] is not None:
        tau = float(tau_deriv["tau_initial"])
        tau_source = "derived from the trailing pre-window week"
    else:
        why = (tau_deriv.get("note") if tau_deriv
               else "no pre-window frame or window start given")
        _require_shadow_config(cfg, why=why)
        tau = float(cfg["exploration"]["tau_initial"])
        tau_source = "config paste (exploration.tau_initial)"

    # SAMPLE FIRST: the gate reads rates a uniform episode sample estimates,
    # and sampling after the predict step costs a full run for sample evidence
    population = d.episode_id.unique()
    sampled = bool(max_episodes) and len(population) > max_episodes
    if sampled:
        keep = rng.choice(population, max_episodes, replace=False)
        d = d[d.episode_id.isin(keep)]

    d, groups, items = _prepare_items(d, cfg, model, r_lookup)

    rejected = {}
    n_dec = n_out = cost_floor_violations = differs = 0
    rec_disc = leg_disc = would_be_cost = 0.0
    n_forced = empty_affordable = 0
    raw_information = 0.0
    abs_log_ratio = forced_mu = forced_discount_gap = 0.0
    # markdown IL on the SAME episodes/window as the spend, so budget and
    # spend share a population; accumulated as scalars, not rows
    il_discount = 0.0
    # Q-spreads for every decision on THIS path, so tau is re-derived on the
    # population that will actually run (not the replay's entry-only one)
    ledger = explore.SpreadLedger()
    # one FINAL row per episode (source ending_inventory, not simulated) so
    # scrap is classified by common.episodes.classify_last, not a copy of it
    last_rows = []
    latencies = []
    drift = {"mu": [], "r": [], "q": [], "sold": [], "date": [],
             "cell": []}

    ctx = {"cfg": cfg, "tau": tau, "model_version": model.version,
           "seed": seed, "cal_grain": model.calibration_grain,
           "cells": {str(c): posterior.get(c) for c in d.category.unique()}}

    for out in map_episodes(_shadow_one, items, ctx, workers):
        for reason, k in out["rejected"].items():
            rejected[reason] = rejected.get(reason, 0) + k
        cost_floor_violations += out["cost_floor_violations"]
        n_forced += out["n_forced"]
        empty_affordable += out["empty_affordable"]
        would_be_cost += out["would_be_cost"]
        raw_information += out["raw_information"]
        abs_log_ratio += out["abs_log_ratio"]
        forced_mu += out["forced_mu"]
        forced_discount_gap += out["forced_discount_gap"]
        rec_disc += out["rec_disc"]
        leg_disc += out["leg_disc"]
        differs += out["differs"]
        latencies.extend(out["latencies"])
        for key in drift:
            drift[key].extend(out["drift"][key])
        il_discount += out["ep_discount_cost"]
        for day, costs in out["spreads"]:
            ledger.add(day, costs)
        if out["last_row"] is not None:
            last_rows.append(out["last_row"])
        # the parent commits through the real store, in episode order, so
        # dedup and quarantine (which the gate measures) run where they ran
        for decision, outcome in out["events"]:
            store.emit_decision(decision)
            n_dec += 1
            outcome["finalized_at"] = pd.Timestamp.now("UTC").isoformat()
            if store.emit_outcome(outcome):
                n_out += 1

    if n_dec == 0:
        raise RuntimeError("no decisions produced -- empty input or all states rejected")

    # censored basis: sales cannot exceed inventory, so the drift ratio
    # compares realised sales against E[min(D, q)] -- never raw mu
    mu_arr = np.array(drift["mu"])
    r_arr = np.array(drift["r"])
    q_arr = np.array(drift["q"], dtype=float)
    max_k = cfg["pricing"]["negbin_max_k"]
    sold_total = float(np.sum(drift["sold"]))

    def _ratio(mu):
        pred = expected_min_demand_inventory_vec(mu, r_arr, q_arr, max_k)
        return (float(sold_total / pred.sum()) if pred.sum() > 0 else None)

    drift_ratio = _ratio(mu_arr)

    # THE SECOND READING: the same rows re-priced under a WEEKLY RE-FIT, the
    # way production's cron would hold the factors. The artifact's schedule
    # stops at test_end, so `drift_ratio` above is "launch and never
    # re-calibrate"; this one is "launch and re-fit weekly", and the spread
    # between them is what weekly re-calibration is worth on this window.
    # A factor is a per-(cell, week) multiplier, so swapping it is an exact
    # rescale of mu -- no re-prediction, and eps/r/q are untouched.
    refit_ratio, refit_cov, refit_applied = None, [], 0
    refit = {}
    if pre_window_frame is not None and drift["date"]:
        try:
            refit, refit_cov = weekly_refit_schedule(
                pre_window_frame, cfg, model, r_lookup,
                min(drift["date"]), max(drift["date"]))
        except Exception as exc:                          # noqa: BLE001
            refit, refit_cov = {}, [{"error": str(exc)}]
    if refit:
        anchor_f = model.calibration
        weeks = pd.to_datetime(pd.Series(drift["date"])) \
                  .dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
        scale = np.ones(len(mu_arr))
        for i, (wkey, cell) in enumerate(zip(weeks.to_numpy(),
                                             drift["cell"])):
            table = refit.get(wkey)
            if table is None:
                continue                       # unfitted week keeps the anchor
            base = float(anchor_f.get(cell, 1.0)) or 1.0
            scale[i] = float(table.get(cell, base)) / base
            refit_applied += 1
        refit_ratio = _ratio(mu_arr * scale)

    # weeks-to-convergence input: evidence bought -> bounded posterior steps;
    # the step cap and daily human gate keep a calendar floor regardless
    eff_information = raw_information / deff(cfg)
    inc = cfg["learning"]["information_increment"]
    n_ep = len(groups)
    # Would-be spend vs budget on SHADOW'S OWN basis: the backtest bisection
    # solves on the exploit-only path, but shadow's anchored path has
    # different affordable sets, so the same tau buys different exploration.
    # Both sides here cover the same episodes over the same days.
    n_days = max((pd.Timestamp(d.date.max()) - pd.Timestamp(d.date.min())).days + 1, 1)
    last = pd.DataFrame(last_rows)
    kind = episodes.classify_last(last)
    leftover = episodes.leftover_units(last.starting_inventory, last.units_sold)
    completed = (kind == episodes.COMPLETED).to_numpy()
    scrap_per_ep = (last.cost.to_numpy() * leftover.to_numpy()) * completed
    il_scrap = float(scrap_per_ep.sum())
    il_unknown_scrap = int((kind == episodes.NOT_CLOSED).sum())
    markdown_il = il_discount + il_scrap
    # a day's realised IL = IL (discount AND scrap) of episodes that CLOSED
    # that day; unclosed episodes contribute nothing until close, so the
    # trailing budget base is knowable at the start of each day
    closed = (kind != episodes.NOT_CLOSED).to_numpy()
    ep_il = last.discount_cost.to_numpy() + scrap_per_ep
    # seeded with pre-window closed-episode IL, so the window's first days
    # carry the trailing base production would actually hold at launch.
    # SCALE IT TO THE SAMPLE: the seed is measured on the FULL frame (it has
    # to be -- those episodes are outside the window and never sampled),
    # while the window's own IL and the SPEND it is compared against are
    # measured on the sampled slice. Left unscaled it inflates the first
    # budget_il_window_days of budget by 1/sample_fraction and drives
    # spend_over_budget toward zero on any sampled run.
    seed_scale = len(groups) / max(len(population), 1)
    il_by_day = {day: amount * seed_scale
                 for day, amount in (prior_il_by_day or {}).items()}
    for day, amount, ok in zip(last.close_day.to_numpy(), ep_il, closed):
        if ok:
            il_by_day[day] = il_by_day.get(day, 0.0) + float(amount)

    # production's budget_today, not a simplified one: it scales the share
    # down as the posterior narrows (constant here, but the same quantity
    # the stop condition is evaluated against)
    cells = posterior.state["cells"]
    widest_std = max(rec["std"] for rec in cells.values())
    # aggregate gate grades mean spend against the MEAN daily budget on the
    # same trailing basis as the controller trace, so the two cannot disagree
    daily_budgets = [explore.budget_today(
        explore.trailing_daily_il(il_by_day, day, cfg), widest_std, cfg)
        for day in sorted(il_by_day)]
    daily_budget = (float(np.mean(daily_budgets)) if daily_budgets
                    else explore.budget_today(markdown_il / max(n_days, 1),
                                              widest_std, cfg))
    implied_daily_spend = would_be_cost / n_days
    over = (implied_daily_spend / daily_budget) if daily_budget > 0 else None
    stop_at = cfg["monitoring"]["stop_conditions"]["exploration_cost_vs_budget"]

    # re-derive tau on THIS path: same bisection as the replay, but on the
    # decisions that actually happen (the replay solved on entry only)
    tau_rec = ledger.solve_tau(daily_budget, n_days=n_days)
    budget_check = {
        "basis": "shadow's own anchored decision path, same episodes and days "
                 "on both sides",
        "days": int(n_days),
        "implied_daily_spend": round(implied_daily_spend, 1),
        "daily_budget": round(daily_budget, 1),
        "trailing_basis_seeded_days": len(prior_il_by_day or {}),
        # the seed is population-scale and everything it is compared against
        # is sample-scale; this is the factor that reconciles them
        "trailing_basis_seed_scale": round(seed_scale, 6),
        "budget_basis": (f"mean of per-day budgets on the trailing "
                         f"{cfg['exploration']['budget_il_window_days']}-day "
                         "realised-IL base (explore.trailing_daily_il) -- the "
                         "budget production would apply, not a whole-window "
                         "average"),
        "spend_over_budget": round(over, 2) if over is not None else None,
        "stop_condition_multiple": stop_at,
        "markdown_il_total": round(markdown_il, 1),
        "markdown_il_discount": round(il_discount, 1),
        "markdown_il_scrap": round(il_scrap, 1),
        "episodes_unknown_scrap_excluded": il_unknown_scrap,
        "budget_share_of_il": cfg["exploration"]["budget_share_of_il"],
        "budget_scale_applied": round(min(max(
            widest_std / cfg["exploration"]["budget_scale_ref_std"],
            cfg["exploration"]["budget_scale_floor"]), 1.0), 4),
        "tau": tau,
        "tau_source": tau_source,
        "tau_recommended": round(tau_rec, 2) if tau_rec else None,
        "tau_recommended_ratio": round(tau_rec / tau, 4)
            if tau_rec and tau else None,
        # checkable derivation: must sit just under daily_budget
        "tau_recommended_implied_spend": round(
            ledger.implied_daily_spend(tau_rec, n_days), 1) if tau_rec else None,
        "spread_decisions": ledger.decisions,
        "spread_decisions_per_episode": round(ledger.decisions / n_ep, 2)
            if n_ep else None,
        "q_spread_distribution": ledger.distribution(),
        "verdict": (
            "NO IL -- cannot project a budget" if over is None else
            f"WOULD SUSPEND -- {over:.2f}x budget, above the {stop_at}x stop "
            "condition; re-derive tau on this basis before the pilot"
            if over > stop_at else
            f"OVER BUDGET -- {over:.2f}x; the tau controller shrinks tau at the "
            "operator gate, capped at halving per day" if over > 1 else
            f"within budget -- {over:.2f}x"),
        "note": (("tau was derived at launch on this run's own trailing "
                  "pre-window week (tau_initial_derivation), so day one of "
                  "tau_controller_trace is an out-of-sample test of it. "
                  "tau_recommended is the same bisection pooled over the "
                  "WHOLE window -- a cross-check, not a correction; for the "
                  "pilot, paste the derivation's tau_initial the way rho is "
                  "pasted, after reading the trace.")
                 if tau_deriv is not None and not tau_deriv["fallback"] else
                 ("tau came from the config paste: the pre-window week was "
                  "missing or too thin to derive on (see "
                  "tau_initial_derivation). The paste's backtest source runs "
                  "the EXPLOIT-ONLY path, whose affordable sets differ from "
                  "this anchored one, so read tau_recommended and "
                  "tau_controller_trace before trusting the launch value.")),
    }
    budget_check["tau_controller_trace"] = _controller_trace(
        ledger, il_by_day, tau, widest_std, cfg, window_days=n_days,
        sampled_episodes=n_ep, population_episodes=len(population))

    per_episode = eff_information / n_ep if n_ep else 0.0
    step = cfg["learning"]["max_mean_step"]
    learning_yield = {
        "effective_information_total": round(eff_information, 2),
        "effective_information_per_episode": round(per_episode, 5),
        "deff_applied": round(deff(cfg), 3),
        "bounded_updates_supported": round(eff_information / inc, 2),
        "episodes_per_bounded_update": round(inc / per_episode, 1)
            if per_episode > 0 else None,
        "max_mean_step": step,
        "calendar_floor_days_per_0.15_of_mean": 1,
        # WHY the yield is what it is. A low per-episode figure has two very
        # different causes with opposite remedies -- too few forced decisions
        # (raise tau) or forced prices sitting too close to the reference
        # (the affordable set only reaches the nearest tiers) -- and the
        # aggregate cannot tell them apart. Information is QUADRATIC in the
        # log price ratio, so the mean move is the term to read first.
        "forced_decisions": n_forced,
        "information_per_forced_decision": round(
            raw_information / n_forced, 6) if n_forced else None,
        "mean_abs_log_price_ratio_forced": round(
            abs_log_ratio / n_forced, 4) if n_forced else None,
        "mean_discount_gap_from_reference_forced_pp": round(
            100 * forced_discount_gap / n_forced, 2) if n_forced else None,
        "mean_mu_on_forced_hours": round(
            forced_mu / n_forced, 3) if n_forced else None,
        "note": ("Would-be: no price was applied, so this is the evidence the "
                 "recommendations WOULD have bought. Each bounded update moves "
                 f"the posterior mean at most {step} and at most one commits "
                 "per day (human gate), so shifting the mean by X takes at "
                 f"least ceil(X/{step}) calendar days however much evidence "
                 "arrives. Divide episodes_per_bounded_update by the pilot's "
                 "daily episode count for the evidence-side estimate; the "
                 "binding constraint is whichever is larger. To read a "
                 "DISAPPOINTING yield, go to the three terms behind it: "
                 "information per forced decision is mu * L^2 * r/(r+mu) with "
                 "L the log price ratio vs the REFERENCE discount, so it is "
                 "quadratic in how far exploration moved the price and only "
                 "linear in how much demand was there. A small "
                 "mean_discount_gap_from_reference_forced_pp is the usual "
                 "culprit and tau is its lever: tau buys the CHEAPEST "
                 "alternatives first, and those are the tiers nearest the "
                 "optimum -- the quadratically least informative ones. Compare "
                 "the tau in force against q_spread_distribution: funding only "
                 "sub-p25 spreads means exploring often and learning little."),
    }

    completeness = n_out / n_dec
    matched = n_out / n_dec        # 1:1 by construction; gaps = quarantined/dupes
    sg = cfg["monitoring"]["shadow_gate"]
    gate = {
        "event_completeness": {
            "value": round(completeness, 4),
            "threshold": sg["min_event_completeness"],
            "pass": completeness >= sg["min_event_completeness"]},
        "matched_decision_rate": {
            "value": round(matched, 4),
            "threshold": sg["min_matched_decision_rate"],
            "pass": matched >= sg["min_matched_decision_rate"]},
        "cost_floor_violations": {
            "value": cost_floor_violations,
            "threshold": 0,
            "pass": cost_floor_violations == 0},
    }
    if sampled:
        # a zero COUNT is only zero over what was sampled; say so rather
        # than letting "0 violations" read as a proof over the window
        gate["sampling_caveat"] = (
            f"gate measured on {len(groups):,} of {len(population):,} episodes "
            f"({len(groups) / max(len(population), 1):.1%}, seed {seed}). The "
            "rates are sample estimates -- at this size the standard error on "
            "a rate near 0.99 is "
            f"{(0.99 * 0.01 / max(len(groups), 1)) ** 0.5:.4f}, against the "
            "0.01 the gate discriminates on. The zero cost-floor violation "
            "count is zero OVER THE SAMPLE, not a proof over the window. "
            "Cost-floor safety is structural (the action set cannot express a "
            "below-cost price) and separately unit-tested -- this gate "
            "confirms it end-to-end, it does not establish it.")
    if window_basis != HOLDOUT_BASIS:
        # in-sample rows flatter the drift ratio, tau and learning yield;
        # the plumbing checks survive
        gate["in_sample_caveat"] = (
            f"run on '{window_basis}', NOT the hold-out. Every artifact was "
            "fit on data up to split.test_end, so any of that window included "
            "here is in-sample: the drift ratio, tau_recommended and the "
            "learning yield are flattered by it. The completeness, matched-rate "
            "and cost-floor checks are unaffected -- they test plumbing, not "
            "fit. Re-run without --all for the launch record.")
    gate["verdict"] = ("PASS -- proceed to exploit-only pilot (section 19)"
                       if all(g["pass"] for g in gate.values()
                              if isinstance(g, dict))
                       else "FAIL -- do not apply prices")

    return {
        "artifact_versions": {
            "baseline_model_version": model.version,
            "posterior_versions": {c: r["version"]
                                   for c, r in posterior.state["cells"].items()},
            "config_version": cfg["meta"]["config_version"],
            # did every priced row get its OWN week's level factors, or did
            # some fall back to the frozen set? Silent by construction
            "calibration_coverage": model.calibration_coverage(),
        },
        "window": {"date_min": str(d.date.min()), "date_max": str(d.date.max()),
                   "episodes": len(groups),
                   "population_episodes": int(len(population)),
                   "basis": window_basis,
                   "out_of_sample": window_basis == HOLDOUT_BASIS,
                   "sampled": sampled,
                   "sample_seed": seed if sampled else None},
        "decision_count": n_dec,
        "outcome_count": n_out,
        "state_rejected_count": int(sum(rejected.values())),
        "rejected_reasons": rejected,
        "duplicate_counts": store.duplicate_counts,
        # THIS RUN, not the whole quarantine file -- see EventStore.__init__.
        "quarantined_event_count": store.quarantined_this_run,
        "shadow_gate": gate,
        "exploration_would_be": {
            "forced_rate": round(n_forced / n_dec, 4),
            "would_be_cost_total": round(would_be_cost, 1),
            "affordable_set_empty_rate": round(empty_affordable / n_dec, 4),
            "tau": tau,
            "note": "no price was applied; costs are the expected IL the "
                    "recommendations would have spent",
        },
        "exploration_budget_would_be": budget_check,
        # the launch tau's own derivation (or why it fell back to the paste);
        # tau_provenance_error accepts this block as a paste source
        "tau_initial_derivation": tau_deriv,
        "learning_yield_would_be": learning_yield,
        "recommendation_vs_legacy": {
            "mean_recommended_discount": round(rec_disc / n_dec, 4),
            "mean_legacy_discount": round(leg_disc / n_dec, 4),
            "share_hours_differing": round(differs / n_dec, 4),
        },
        "realised_vs_predicted_sold_ratio_at_legacy_price": round(drift_ratio, 4)
            if drift_ratio else None,
        # BOTH calibration regimes on the same rows (owner, 2026-08-28).
        # frozen = the shipped artifact, whose weekly schedule stops at
        # test_end so every hold-out row holds the anchor: "launch and never
        # re-calibrate". weekly_refit = factors re-fit each week on the
        # trailing window ending strictly before it, the way production's
        # cron holds them. The SPREAD is what weekly re-calibration is worth
        # over this window; read it before deciding the production cadence.
        "calibration_regimes": {
            "frozen_anchor": round(drift_ratio, 4) if drift_ratio else None,
            "weekly_refit": round(refit_ratio, 4) if refit_ratio else None,
            "spread": (round(refit_ratio - drift_ratio, 4)
                       if (refit_ratio and drift_ratio) else None),
            "rows_rescaled": refit_applied,
            "weeks_refit": sum(1 for c in refit_cov if c.get("fitted")),
            "weeks_on_partial_window": [
                c["week"] for c in refit_cov if c.get("partial")],
            "weeks_unfitted_held_at_anchor": [
                c["week"] for c in refit_cov if c.get("fitted") is False],
            "note": ("Both ratios are realised/E[min(D,q)] on the SAME rows "
                     "at the legacy price -- only the level factor differs. "
                     "The schedule is fit here, not in the artifact, so the "
                     "pre-launch bundle stays clean of hold-out rows (rule "
                     "16); at week k it reads only weeks < k, which is what "
                     "the cron has on that Monday. Both near 1.0 = the level "
                     "held. Only frozen off = the anchor went stale and "
                     "weekly re-fitting earns its keep. Both off = the level "
                     "moved faster than a weekly cadence can track."),
        },
        "solver_latency_p95_s": round(float(np.percentile(latencies, 95)), 4),
        "note": ("Shadow outcomes carry execution_status="
                 f"'{SHADOW_STATUS}' and are ineligible for pipeline.update: "
                 "the recommended price was never in force. The drift ratio is "
                 "the production continuation of the section 9.3 gate."),
    }


def main():
    ap = argparse.ArgumentParser(prog="pipeline.shadow")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="reports/shadow.json")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date-start", default=None,
                    help="keep episodes whose WINDOW OPENED on or after this; "
                         "overrides the hold-out default")
    ap.add_argument("--date-end", default=None,
                    help="keep episodes whose WINDOW OPENED on or before this")
    ap.add_argument("--holdout", action="store_true",
                    help="run on data.holdout (THE DEFAULT -- accepted for "
                         "explicitness, changes nothing)")
    ap.add_argument("--all", action="store_true",
                    help="run on the whole extract instead. Partly IN-SAMPLE: "
                         "the drift ratio, tau_recommended and the learning "
                         "yield are flattered by rows the artifacts were fit "
                         "on. The report says so. Not for the launch record.")
    ap.add_argument("--events-dir", default=None)
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="episode sample size; 0 = all episodes. Default: "
                         "monitoring.shadow_gate.sample_episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for the episode loop. 0 = every core but "
                         "one. Each episode draws from its OWN generator "
                         "seeded by episode id, so the result is identical "
                         "serial or parallel, and independent of order.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)
    # Hold-out by default. Anything else is a deliberate, labelled exception.
    start, end = args.date_start, args.date_end
    if start or end:
        basis = f"explicit range {start or 'start'} -> {end or 'end'}"
        print(f"== {basis} ==")
    elif args.all:
        basis = "full extract"
        print("== full extract -- PARTLY IN-SAMPLE, not the launch record ==")
    else:
        h = cfg["data"].get("holdout")
        if not h:
            raise SystemExit(
                "no data.holdout in config.yaml. Shadow runs on the hold-out "
                "by default because every artifact is fit up to test_end; "
                "add the window, or pass --all and read the in-sample caveat.")
        basis = HOLDOUT_BASIS
        start, end = h["start"], h["end"]
        print(f"== holdout window {start} -> {end} "
              "(no artifact was fit on it) ==")
    # trailing IL history from BEFORE the window, computed on the full frame
    # before it is sliced away -- production's day-one budget base
    history = pre_window_il_history(d, cfg, start)
    full = d
    # episode-scoped date cut, never row-scoped: window_slice keeps a
    # cross-midnight episode whole instead of leaving an orphan tail
    d = episodes.window_slice(d, start, end)
    if d.empty:
        raise SystemExit(f"no episodes opened in [{start}, {end}]")

    report = run_shadow(d, cfg, events_root=args.events_dir,
                        seed=args.seed, max_episodes=args.max_episodes,
                        window_basis=basis, workers=args.workers,
                        prior_il_by_day=history,
                        pre_window_frame=full, window_start=start)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    g = report["shadow_gate"]
    w = report["window"]
    print(f"window             : {w['basis']} · {w['date_min']} -> "
          f"{w['date_max']}"
          + ("" if w["out_of_sample"] else "  [PARTLY IN-SAMPLE]"))
    print(f"episodes           : {w['episodes']:,} of "
          f"{w['population_episodes']:,}"
          + (f" (sample, seed {w['sample_seed']})" if w["sampled"] else ""))
    print(f"decisions          : {report['decision_count']:,} "
          f"({report['state_rejected_count']} states rejected)")
    print(f"event completeness : {g['event_completeness']['value']:.4f} "
          f"-> {'PASS' if g['event_completeness']['pass'] else 'FAIL'}")
    print(f"matched rate       : {g['matched_decision_rate']['value']:.4f} "
          f"-> {'PASS' if g['matched_decision_rate']['pass'] else 'FAIL'}")
    print(f"cost-floor viol.   : {g['cost_floor_violations']['value']} "
          f"-> {'PASS' if g['cost_floor_violations']['pass'] else 'FAIL'}")
    rv = report["recommendation_vs_legacy"]
    print(f"mean discount      : recommended {rv['mean_recommended_discount']:.3f} "
          f"vs legacy {rv['mean_legacy_discount']:.3f} "
          f"(differs {rv['share_hours_differing']:.1%} of hours)")
    print(f"drift ratio        : "
          f"{report['realised_vs_predicted_sold_ratio_at_legacy_price']}")
    cr = report.get("calibration_regimes") or {}
    if cr.get("weekly_refit") is not None:
        print(f"calibration      : frozen {cr['frozen_anchor']} | "
              f"weekly re-fit {cr['weekly_refit']} | spread {cr['spread']} "
              f"({cr['weeks_refit']} weeks re-fit)")
    elif cr:
        print("calibration      : frozen anchor only -- no week could be "
              "re-fit on this window")
    ly = report["learning_yield_would_be"]
    print(f"would-be learning  : {ly['bounded_updates_supported']} bounded "
          f"updates from this window "
          f"({ly['episodes_per_bounded_update']} episodes per update); "
          f"calendar floor is 1 update/day")
    if ly["forced_decisions"]:
        print(f"  evidence per hour: {ly['forced_decisions']:,} forced "
              f"decisions x {ly['information_per_forced_decision']} info each"
              f" · mean move {ly['mean_discount_gap_from_reference_forced_pp']}"
              f"pp from reference (info is QUADRATIC in this)")
    td = report.get("tau_initial_derivation")
    if td and td.get("tau_initial") is not None:
        print(f"tau launch         : {td['tau_initial']:,.2f} derived on the "
              f"pre-window week [{td['week'][0]} .. {td['week'][1]}] "
              f"({td['decisions']:,} decisions)")
    elif td:
        print(f"tau launch         : config paste in force -- {td['note']}")
    bc = report["exploration_budget_would_be"]
    print(f"exploration budget : spend {bc['implied_daily_spend']:,.0f}/day vs "
          f"budget {bc['daily_budget']:,.0f}/day over {bc['days']} days")
    print(f"                     {bc['verdict']}")
    if bc["tau_recommended"]:
        print(f"tau                : in force {bc['tau']:,.2f} -> recommended "
              f"{bc['tau_recommended']:,.2f} "
              f"({bc['tau_recommended_ratio']:.2f}x) on "
              f"{bc['spread_decisions']:,} decisions "
              f"({bc['spread_decisions_per_episode']}/episode)")
    print(f"tau controller     : {bc['tau_controller_trace']['verdict']}")
    print(g["verdict"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
