"""Shadow-mode harness (docs/design.md section 5.13).

Runs the full production decision path against observed data; NO prices are
applied. Outcomes are stamped execution_status="shadow_not_applied" and are
ineligible for daily.update. Runs on `data.holdout` BY DEFAULT -- frozen
artifacts are fit up to split.test_end, so `--all` is partly in-sample and
the report says so. Exit gate: event completeness, matched decision rate,
and ZERO cost-floor violations."""

import argparse
import hashlib

import numpy as np
import pandas as pd

from common.config import load_config, deff_from_episodes, ConfigError
from common import episodes
from common import metrics
from common.io import read_json, write_json
from common.parallel import map_episodes
from common.provenance import config_fingerprint
from common.episodes import adjustment_reason
from evaluate.backtest import predict_frame
from fit.prepare_data import population
from fit.train_baseline import BaselineModel
from events.store import EventStore
from engine import dp as dp_mod
from engine import explore
from engine.demand import expected_min_demand_inventory_vec
from engine.decide import decide, StateRejected
from engine.posterior import PosteriorStore

SHADOW_STATUS = "shadow_not_applied"

# Default window: the hold-out is the only span no frozen artifact was fit
# on, so honesty is the default rather than a flag someone must remember.
HOLDOUT_BASIS = "holdout"


def _require_shadow_config(cfg, backtest_path="reports/backtest.json",
                           shadow_path="reports/shadow.json", why=None):
    """Fallback only: the config-paste tau must exist and match its source."""
    prefix = f"tau derivation unavailable ({why}); " if why else ""
    if cfg["exploration"]["tau_initial"] is None:
        raise ConfigError(
            "shadow phase blocked: " + prefix + "exploration.tau_initial is "
            "null. Give the run a pre-window week (the default hold-out run "
            "does) or paste a derived tau.")

    # Non-null is not enough: tau_initial is hand-pasted and decides day-one
    # spend, before the controller has any spend to correct from.
    stale = explore.tau_provenance_error(cfg, read_json(backtest_path),
                                         read_json(shadow_path))
    if stale:
        raise ConfigError("shadow phase blocked by a stale tau: "
                          + prefix + stale)


def pre_window_il_history(d, cfg, before):
    """Realised legacy IL by close day for DP-ELIGIBLE episodes that CLOSED
    in the budget_il_window_days before the window -- the day-one budget
    base. Same population as everything it is scaled against (derive_tau0's
    sample fraction, run_shadow's seed_scale)."""
    if before is None or d.empty:
        return {}
    d = population(d, cfg, "dp_eligible")
    if d.empty:
        return {}
    start = pd.Timestamp(str(before))
    window = int(cfg["exploration"]["budget_il_window_days"])
    lo = (start - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
    hi = start.strftime("%Y-%m-%d")
    econ, _ = metrics.settled(metrics.episode_economics(d))
    close = econ.close_day.astype(str)
    econ = econ[(close >= lo) & (close < hi)]
    return {str(k): float(v) for k, v in econ.groupby("close_day").il.sum().items()}


# episodes are independent (tau is fixed; the controller walk is
# post-processing), so the unit of work is one episode and all parallelise
EP_COLS = ("hour_of_day", "sku_id", "fc", "category", "subcategory",
           "starting_inventory", "ending_inventory", "units_sold",
           "total_discount", "original_price", "cost", "r",
           "mu_ref_hat", "date", "is_observed")

# per-episode scalars _shadow_one returns and the parent sums -- one list,
# so a new term cannot be produced without being folded in
SCALARS = ("cost_floor_violations", "n_forced", "empty_affordable",
           "would_be_cost", "raw_information",
           # info is quadratic in the log price move, linear in demand
           "abs_log_ratio", "forced_mu", "forced_discount_gap",
           # SIGNED: hours the recommendation is deeper / shallower than
           # the legacy price in force (a differing hour is one or the other)
           "rec_disc", "leg_disc", "deeper", "shallower")


def weekly_refit_schedule(d_full, cfg, model, r_lookup, start, end):
    """Re-fit the level factors per shadow week, as production's cron would.
    Fit HERE, not in the artifact, so the pre-launch bundle stays clean of
    hold-out rows (rule 16); at week k it reads only weeks < k.
    Returns ({week_start: {cell: factor}}, coverage)."""
    from fit.train_baseline import _solve_level_factors

    bm = cfg["baseline_model"]
    weeks_back = bm["calibration_fit_trailing_weeks"]
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
        # STRICTLY BEFORE this week: no look-ahead inside the replay; the
        # same whole-episode cut the artifact schedule uses
        window, weeks_seen = episodes.trailing_weeks_window(scope, w0, weeks_back)
        fitted = _solve_level_factors(
            window.copy(), model, bm["calibration_shrinkage_units"],
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
    """Pack per-episode arrays for _shadow_one over `predict_frame` (the one
    extend/lookup/predict path, shared with the backtest)."""
    d = predict_frame(d, cfg, model, r_lookup)
    groups = list(d.groupby("episode_id", sort=False))
    items = [dict({c: g[c].to_numpy() for c in EP_COLS}, episode_id=eid)
             for eid, g in groups]
    return d, groups, items


def _ctx(cfg, tau, model, posterior, seed, categories):
    """The read-only context every episode worker gets."""
    return {"cfg": cfg, "tau": tau, "model_version": model.version,
            "seed": seed, "cal_grain": model.calibration_grain,
            "cells": {str(c): posterior.get(c) for c in categories}}


def _fill_ledger(items, ctx, workers, ledger):
    """Run _shadow_one over `items`, adding every decision's Q-spreads to
    `ledger`; yields each episode's result for the caller to fold."""
    for out in map_episodes(_shadow_one, items, ctx, workers):
        for day, costs, moves, dmin in out["spreads"]:
            ledger.add(day, costs, moves, dmin)
        yield out


def _mean_daily_budget(days, il_by_day, widest_std, cfg):
    """Mean of production's per-day budget over `days` -- the window's
    DECISION days, never the pre-window seed days (the first of those has
    no trailing history and would read as a zero budget)."""
    return float(np.mean([explore.budget_today(
        explore.trailing_daily_il(il_by_day, day, cfg), widest_std, cfg)
        for day in days]))


def derive_tau0(d_full, cfg, start, model, posterior, r_lookup, il_history,
                seed=0, max_episodes=None, workers=None):
    """Launch tau derived on THIS run's anchored path over the trailing
    pre-window week (same span as the day-one budget base). tau_initial is
    None when the week is too thin; the caller falls back to the paste."""
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
    widest_std = posterior.widest_std()
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
    # the span the spend is divided by: the POPULATION's, unextended (a
    # sample can shrink it; extend_to_window's synthetic tail can add a day)
    n_days = episodes.calendar_days(pre.date)
    # decoupled from the window's sample draw, same reproducibility contract
    rng = np.random.default_rng([int(seed), 1])
    if max_episodes and n_pop > max_episodes:
        keep = rng.choice(pre_ids, max_episodes, replace=False)
        pre = pre[pre.episode_id.isin(keep)]
    _, groups, items = _prepare_items(pre, cfg, model, r_lookup)
    # tau None: nothing explores, and the ledger does not care -- spreads are
    # recorded before the draw, independent of the tau in force
    ctx = _ctx(cfg, None, model, posterior, seed, pre.category.unique())
    ledger = explore.SpreadLedger()
    for _ in _fill_ledger(items, ctx, workers, ledger):
        pass

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
        note="design 5.13 -- paste into exploration.tau_initial")
    return block


def _controller_trace(ledger, il_by_day, tau0, widest_std, cfg, window_days=None,
                      sampled_episodes=None, population_episodes=None):
    """Day-by-day tau-controller walk: does the pilot survive its first
    week? Spend per day is EXPECTED spend at the tau in force."""
    sc = cfg["monitoring"]["stop_conditions"]
    stop_at, persist = sc["exploration_cost_vs_budget"], int(sc["persistence_days"])
    max_days = int(cfg["tuning"]["controller_trace_max_days"])
    days = ledger.days
    order = sorted(range(len(days)), key=lambda i: days[i])[:max_days]
    index = {days[i]: i for i in order}
    # the SAME walk production runs (explore.walk_tau); spend here is
    # EXPECTED spend at the tau in force
    tau, walked = explore.walk_tau(
        float(tau0), [days[i] for i in order],
        lambda day, t: ledger.spend_by_day(t)[index[day]],
        il_by_day, widest_std, cfg)
    rows, first_within, suspend_days, streak, prev = [], None, 0, 0, None
    for rank, r in enumerate(walked):
        over = (r["spend"] / r["budget"]) if r["budget"] > 0 else None
        # the monitor's rule (daily.monitor.evaluate_guardrail): over the
        # multiple on persistence_days CONSECUTIVE CALENDAR days -- a
        # calendar day with no decision breaks the streak, as does a
        # zero-budget day (no reading, not an overspend)
        day = pd.Timestamp(r["day"])
        if not (over is not None and over > stop_at):
            streak = 0
        elif prev is not None and (day - prev).days == 1:
            streak += 1
        else:
            streak = 1
        prev = day
        fired = streak >= persist
        suspend_days += int(fired)
        if over is not None and over <= 1.0 and first_within is None:
            first_within = rank + 1
        rows.append({"day": r["day"], "tau": r["tau"], "spend": r["spend"],
                     "budget": r["budget"],
                     "over_budget": round(over, 2) if over is not None else None,
                     "days_over": streak,
                     "stop_condition_fires": fired})
    # three distinct day counts -- calendar span, days with decisions, days
    # walked -- none interchangeable, especially on a sample; the per-day
    # episode rates both divide by the calendar span
    span = int(window_days) if window_days else len(ledger.days)
    return {
        "tau_start": round(float(tau0), 2),
        "tau_end": round(tau, 2),
        "by_day": rows,
        "window_days": span,
        "days_with_decisions": len(ledger.days),
        "days_simulated": len(rows),
        "days_truncated": max(len(ledger.days) - len(rows), 0),
        "days_stop_condition_fires": suspend_days,
        "first_day_within_budget": first_within,
        # the ONE figure a sample degrades: this series divides the sample
        # across the window's days (everything else reads rates or is invariant)
        "episodes_per_day_sampled": round(
            sampled_episodes / max(span, 1), 1) if sampled_episodes else None,
        "episodes_per_day_population": round(
            population_episodes / max(span, 1), 1)
            if population_episodes else None,
        "verdict": (
            "no days simulated" if not rows else
            f"exploration suspends on day {persist} and the stop condition "
            f"holds on {suspend_days} of {len(rows)} days -- the controller "
            "cannot correct a tau it has not yet seen spend from"
            if len(rows) >= persist and rows[persist - 1]["stop_condition_fires"] else
            f"survives launch; {suspend_days} of {len(rows)} days would fire "
            "the stop condition" if suspend_days else
            "survives launch; the stop condition never fires"),
        "note": ("design 5.13 -- expected spend at the tau in force; on a "
                 "sample quote the pooled spend_over_budget, not by_day"),
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

    def exploration_suspended(self):
        # a rehearsal never suspends: the report is what the tau in force
        # WOULD buy, which a production suspension record must not zero
        return None


def _episode_seed(seed, episode_id):
    """A generator per episode, seeded from its id: draws are reproducible
    and independent of visit order, so serial and parallel runs agree."""
    h = hashlib.blake2b(str(episode_id).encode(), digest_size=8).digest()
    return np.random.default_rng([int(seed), int.from_bytes(h, "big")])


def _shadow_one(ep, ctx):
    """Price one episode's hours. Pure (no store, no shared RNG); returns
    everything the parent folds in. `ep` carries arrays, not a DataFrame."""
    cfg, tau = ctx["cfg"], ctx["tau"]
    posterior = _FrozenCells(ctx["cells"])
    store = _BufferStore()
    rng = _episode_seed(ctx["seed"], ep["episode_id"])
    n = len(ep["hour_of_day"])

    out = {k: 0 for k in SCALARS}
    out.update({
        "events": [], "rejected": {}, "spreads": [], "latencies": [],
        "episode_id": ep["episode_id"],
        # date/cell ride along so the drift ratio can be re-read under a
        # weekly re-fit (a factor swap is an exact rescale of mu)
        "drift": {"mu": [], "r": [], "q": [], "sold": [], "date": [],
                  "cell": []},
        # every OBSERVED hour in the prepared-frame vocabulary, so the
        # budget base is metrics.episode_economics -- the same IL the
        # guardrail floors and il_pct measure, not a third approximation
        "hours": [],
    })
    anchor = None

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

        # legacy IL is unconditioned on decision success: recorded here,
        # before decide() can reject the state
        out["hours"].append({
            "episode_id": ep["episode_id"], "date": row_day,
            "hour_of_day": int(ep["hour_of_day"][t]),
            "starting_inventory": q, "units_sold": sold,
            "ending_inventory": ending,
            "original_price": float(ep["original_price"][t]),
            "offered_price": float(ep["original_price"][t]) * (1 - legacy_d),
            "cost": float(ep["cost"][t])})

        state = {
            "episode_id": ep["episode_id"], "sku_id": int(ep["sku_id"][t]),
            "fc": ep["fc"][t], "category": ep["category"][t],
            "subcategory": ep["subcategory"][t],
            "date": row_day,
            "hour_of_day": int(ep["hour_of_day"][t]),
            "hours_remaining": n - t, "q": q,
            "original_price": float(ep["original_price"][t]),
            "cost": float(ep["cost"][t]), "r": float(ep["r"][t]),
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

        for costs, moves, dmin in spreads_here:
            out["spreads"].append((row_day, costs, moves, dmin))
        out["latencies"].append(evt["solver_latency_s"])
        if evt["applied_price"] < evt["cost"] - 1e-6:
            out["cost_floor_violations"] += 1
        if evt["is_exploration"]:
            out["n_forced"] += 1
            out["would_be_cost"] += evt["exploration_cost"]
            # must match daily.update's NB Fisher info mu*L^2*r/(r+mu)
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
        gap = evt["applied_discount"] - legacy_d
        if gap > dp_mod.TIER_EPS:
            out["deeper"] += 1
        elif gap < -dp_mod.TIER_EPS:
            out["shallower"] += 1

        # outcome = what actually happened under the LEGACY price
        outcome = {
            "event": "outcome",
            "outcome_id": f"shadow-{evt['decision_id']}",
            "decision_id": evt["decision_id"],
            "units_sold": sold, "starting_inventory": q,
            "ending_inventory": ending,
            "applied_price": float(ep["original_price"][t] * (1 - legacy_d)),
            "is_stockout": bool(episodes.is_censored_hour(q, sold, ending)),
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
        out["drift"]["r"].append(float(ep["r"][t]))
        out["drift"]["q"].append(q)
        out["drift"]["sold"].append(sold)
        out["drift"]["date"].append(str(ep["date"][t]))
        out["drift"]["cell"].append(str(ep[ctx["cal_grain"]][t]))

        anchor = legacy_d                 # reality's price is the next anchor

    return out


def run_shadow(d, cfg, events_root=None, seed=0, max_episodes=None,
               prior_il_by_day=None, pre_window_frame=None, window_start=None,
               window_basis=HOLDOUT_BASIS, workers=None,
               shadow_path="reports/shadow.json"):
    """`shadow_path` is where THIS run's report will be written -- the
    derivation a config-paste tau is checked against when the run falls
    back to the paste."""
    # precondition, inside run_shadow so a programmatic caller cannot skip it
    d = population(d, cfg, "dp_eligible")
    if d.empty:
        raise RuntimeError("no DP-eligible episodes in this window")
    model = BaselineModel(cfg)
    posterior = PosteriorStore(cfg)
    r_lookup = read_json(cfg["dispersion"]["r_lookup_path"])
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
        _require_shadow_config(cfg, shadow_path=shadow_path, why=why)
        tau = float(cfg["exploration"]["tau_initial"])
        tau_source = "config paste (exploration.tau_initial)"

    # FROZEN ANCHOR for every window row: the "launch and never re-calibrate"
    # regime the drift ratio grades, stated rather than assumed. On the
    # hold-out every row is past the schedule anyway; on a window that
    # overlaps the schedule (--all, an explicit range) the rows would
    # otherwise carry their own week's factors and the weekly_refit rescale
    # (anchor -> re-fit) would be wrong for them. Deliberate, so
    # calibration_coverage reads it as the gate's freeze, not stale factors.
    model.freeze_calibration_from(
        window_start if window_start is not None else d.date.min())

    # SAMPLE FIRST: the gate reads rates a uniform episode sample estimates,
    # and sampling after the predict step costs a full run for sample evidence
    population_ids = d.episode_id.unique()
    n_population = len(population_ids)
    # the span every "per day" figure divides by: the WINDOW's, on the
    # unsampled, unextended frame (a sample can shrink it; extend_to_window's
    # synthetic tail can add a day). Both sides -- spend and budget -- cover
    # the same episodes over the same days. The report's window is this
    # frame's too, so a reader dividing by its dates gets the same span.
    n_days = episodes.calendar_days(d.date)
    date_min, date_max = str(d.date.min()), str(d.date.max())
    sampled = bool(max_episodes) and n_population > max_episodes
    if sampled:
        keep = rng.choice(population_ids, max_episodes, replace=False)
        d = d[d.episode_id.isin(keep)]

    d, groups, items = _prepare_items(d, cfg, model, r_lookup)

    rejected = {}
    n_dec = n_out = 0
    tot = {k: 0 for k in SCALARS}
    # one entry per FORCED hour, so deff is measured at the clustering this
    # run actually produced rather than a frozen calib-window paste
    forced_episode_ids = []
    # Q-spreads for every decision on THIS path, so tau is re-derived on the
    # population that will actually run (not the replay's entry-only one)
    ledger = explore.SpreadLedger()
    # every observed hour, so markdown IL is measured on the SAME episodes
    # and window as the spend (metrics.episode_economics, the one home)
    hours = []
    latencies = []
    drift = {"mu": [], "r": [], "q": [], "sold": [], "date": [],
             "cell": []}

    ctx = _ctx(cfg, tau, model, posterior, seed, d.category.unique())
    for out in _fill_ledger(items, ctx, workers, ledger):
        for reason, k in out["rejected"].items():
            rejected[reason] = rejected.get(reason, 0) + k
        for k in SCALARS:
            tot[k] += out[k]
        forced_episode_ids.extend([out["episode_id"]] * out["n_forced"])
        latencies.extend(out["latencies"])
        for key in drift:
            drift[key].extend(out["drift"][key])
        hours.extend(out["hours"])
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
    n_forced, would_be_cost = tot["n_forced"], tot["would_be_cost"]

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

    # second reading: same rows under a WEEKLY RE-FIT (frozen vs weekly is
    # what re-calibration is worth); a factor swap is an exact mu rescale,
    # and every row carries the ANCHOR factor (frozen above), so the scale
    # is re-fit / anchor for every row
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
        weeks = episodes.week_key(pd.Series(drift["date"]))
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
    shadow_deff = deff_from_episodes(cfg["dispersion"]["rho"],
                                     forced_episode_ids)
    eff_information = tot["raw_information"] / shadow_deff
    inc = cfg["learning"]["information_increment"]
    n_ep = len(groups)
    # Would-be spend vs budget on SHADOW'S OWN basis: the backtest bisection
    # solves on the exploit-only path, but shadow's anchored path has
    # different affordable sets, so the same tau buys different exploration.
    # SCRAP = leftover + shrink (episodes.scrap_units); dp_eligible episodes
    # are closed with a known cost, so `settled` excludes nothing here -- it
    # is called because it is the one home, not because it filters
    econ, _ = metrics.settled(metrics.episode_economics(pd.DataFrame(hours)))
    il_discount = float(econ.discount_cost.sum())
    il_scrap = float((econ.cost * econ.scrap).sum())
    markdown_il = il_discount + il_scrap
    # pre-window IL seed, SCALED TO THE SAMPLE (it is measured on the full
    # dp_eligible frame; unscaled it inflates the first days' budgets by
    # 1/fraction)
    seed_scale = len(groups) / max(n_population, 1)
    il_by_day = {day: amount * seed_scale
                 for day, amount in (prior_il_by_day or {}).items()}
    for day, amount in econ.groupby("close_day").il.sum().items():
        il_by_day[str(day)] = il_by_day.get(str(day), 0.0) + float(amount)

    # production's budget_today, not a simplified one: it scales the share
    # down as the posterior narrows (constant here, but the same quantity
    # the stop condition is evaluated against)
    widest_std = posterior.widest_std()
    # aggregate gate grades mean spend against the MEAN daily budget over the
    # window's DECISION days on the same trailing basis as the controller
    # trace, so the two cannot disagree
    daily_budget = (_mean_daily_budget(ledger.days, il_by_day, widest_std, cfg)
                    if ledger.days else
                    explore.budget_today(markdown_il / max(n_days, 1),
                                         widest_std, cfg))
    implied_daily_spend = would_be_cost / n_days
    over = (implied_daily_spend / daily_budget) if daily_budget > 0 else None
    stop_at = cfg["monitoring"]["stop_conditions"]["exploration_cost_vs_budget"]
    ec = cfg["exploration"]
    share, mult = float(ec["budget_share_of_il"]), float(ec["delta_min_bias_multiple"])

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
        "budget_basis": (f"mean over the window's decision days of the "
                         f"per-day budget on the trailing "
                         f"{ec['budget_il_window_days']}-day realised-IL base "
                         "(explore.trailing_daily_il) -- the budget production "
                         "would apply, not a whole-window average"),
        "spend_over_budget": round(over, 2) if over is not None else None,
        "stop_condition_multiple": stop_at,
        "markdown_il_total": round(markdown_il, 1),
        "markdown_il_discount": round(il_discount, 1),
        "markdown_il_scrap": round(il_scrap, 1),
        "budget_share_of_il": share,
        "budget_scale_applied": round(explore.budget_scale(widest_std, cfg), 4),
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
        "note": ("design 5.13 -- tau_recommended is the same bisection pooled "
                 "over the window: a cross-check on the tau in force, not a "
                 "correction"),
    }
    budget_check["tau_controller_trace"] = _controller_trace(
        ledger, il_by_day, tau, widest_std, cfg, window_days=n_days,
        sampled_episodes=n_ep, population_episodes=n_population)
    # what a smaller budget or a deeper floor would buy, from this ledger
    budget_sweep = ledger.sweep(
        daily_budget, n_days, n_dec, share, mult,
        shares=sorted({round(share * f, 6) for f in (0.25, 0.5, 0.75, 1.0, 1.5)}),
        multiples=sorted({mult, round(mult * 1.5, 4), round(mult * 2, 4)}))

    per_episode = eff_information / n_ep if n_ep else 0.0
    step = cfg["learning"]["max_mean_step"]
    cadence = int(cfg["learning"]["update_cadence_days"])
    per_day_pop = n_population / max(n_days, 1)
    learning_yield = {
        "effective_information_total": round(eff_information, 2),
        "effective_information_per_episode": round(per_episode, 5),
        "deff_applied": round(shadow_deff, 3),
        "bounded_updates_supported": round(eff_information / inc, 2),
        "episodes_per_bounded_update": round(inc / per_episode, 1)
            if per_episode > 0 else None,
        "max_mean_step": step,
        # one bounded update per learning.update_cadence_days: the calendar
        # floor on learning (each step moves the mean at most max_mean_step),
        # and how much evidence each period brings
        "update_cadence_days": cadence,
        "calendar_floor_days_per_step": cadence,
        "bounded_updates_worth_per_period": round(
            per_episode * per_day_pop * cadence / inc, 2)
            if per_episode > 0 and per_day_pop else None,
        # low yield has two causes with opposite remedies: few forced
        # decisions (raise tau) vs small price moves (info is QUADRATIC in
        # the log ratio) -- the terms below tell them apart
        "forced_decisions": n_forced,
        "information_per_forced_decision": round(
            tot["raw_information"] / n_forced, 6) if n_forced else None,
        "mean_abs_log_price_ratio_forced": round(
            tot["abs_log_ratio"] / n_forced, 4) if n_forced else None,
        "mean_discount_gap_from_reference_forced_pp": round(
            100 * tot["forced_discount_gap"] / n_forced, 2) if n_forced else None,
        "mean_mu_on_forced_hours": round(
            tot["forced_mu"] / n_forced, 3) if n_forced else None,
        "note": ("design 5.13 -- would-be evidence; info per decision is "
                 "mu*L^2*r/(r+mu), quadratic in the log price move"),
    }

    # outcomes accepted per decision emitted; the gap is quarantine + dupes.
    # (A separate "matched rate" gate was the same expression under a
    # second name and threshold.)
    completeness = n_out / n_dec
    sg = cfg["monitoring"]["shadow_gate"]
    gate = {
        "event_completeness": {
            "value": round(completeness, 4),
            "threshold": sg["min_event_completeness"],
            "pass": completeness >= sg["min_event_completeness"]},
        "cost_floor_violations": {
            "value": tot["cost_floor_violations"],
            "threshold": 0,
            "pass": tot["cost_floor_violations"] == 0},
    }
    if sampled:
        # a zero COUNT is only zero over what was sampled; say so rather
        # than letting "0 violations" read as a proof over the window
        gate["sampling_caveat"] = (
            f"gate measured on {len(groups):,} of {n_population:,} episodes "
            f"(seed {seed}): rates are sample estimates, and the zero "
            "cost-floor count is zero OVER THE SAMPLE, not a proof over the "
            "window (cost-floor safety is structural and unit-tested).")
    if window_basis != HOLDOUT_BASIS:
        # in-sample rows flatter the drift ratio, tau and learning yield;
        # the plumbing checks survive
        gate["in_sample_caveat"] = (
            f"run on '{window_basis}', NOT the hold-out: the drift ratio, "
            "tau_recommended and the learning yield are flattered by "
            "in-sample rows. The completeness, matched-rate and cost-floor "
            "checks test plumbing, not fit, and are unaffected. Re-run "
            "without --all for the launch record.")
    gate["verdict"] = ("PASS -- proceed to exploit-only pilot (design 9.4, 10)"
                       if all(g["pass"] for g in gate.values()
                              if isinstance(g, dict))
                       else "FAIL -- do not apply prices")

    return {
        "config": config_fingerprint(cfg, "shadow"),
        "artifact_versions": {
            "baseline_model_version": model.version,
            "posterior_versions": {c: r["version"]
                                   for c, r in posterior.state["cells"].items()},
            "config_version": cfg["meta"]["config_version"],
            # every window row is frozen at the anchor ON PURPOSE (above), so
            # this reads OK; STALE here means a pre-window row ran past the
            # schedule's end
            "calibration_coverage": model.calibration_coverage(),
        },
        "window": {"date_min": date_min, "date_max": date_max,
                   # the ONE n_days (episodes.calendar_days) every per-day
                   # figure in this report divides by
                   "days": int(n_days),
                   "episodes": len(groups),
                   "population_episodes": int(n_population),
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
            "affordable_set_empty_rate": round(tot["empty_affordable"] / n_dec, 4),
            "note": "no price was applied; costs are the expected IL the "
                    "recommendations would have spent",
        },
        "exploration_budget_would_be": budget_check,
        "exploration_budget_sweep": budget_sweep,
        # the launch tau's own derivation (or why it fell back to the paste);
        # tau_provenance_error accepts this block as a paste source
        "tau_initial_derivation": tau_deriv,
        "learning_yield_would_be": learning_yield,
        "recommendation_vs_legacy": {
            "mean_recommended_discount": round(tot["rec_disc"] / n_dec, 4),
            "mean_legacy_discount": round(tot["leg_disc"] / n_dec, 4),
            "share_hours_differing": round(
                (tot["deeper"] + tot["shallower"]) / n_dec, 4),
            # every hour re-anchors on LEGACY's price, so "deeper" is the
            # share of hours the agent would cut below the price in force
            # and "shallower" the share it would hold above it; the agent's
            # own within-episode steps are the backtest's
            # intra_episode_moves (shadow never walks its own path)
            "share_hours_recommending_deeper_than_legacy_price": round(
                tot["deeper"] / n_dec, 4),
            "share_hours_recommending_shallower_than_legacy_price": round(
                tot["shallower"] / n_dec, 4),
        },
        "realised_vs_predicted_sold_ratio_at_legacy_price": round(drift_ratio, 4)
            if drift_ratio else None,
        # BOTH calibration regimes on the same rows: frozen = "launch and
        # never re-calibrate"; weekly_refit = production's cron. The SPREAD
        # decides the production cadence.
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
            # a re-fit that raised is a MISSING reading, and the cadence
            # question then cannot be answered -- say so where tune reads
            "refit_error": next((c["error"] for c in refit_cov
                                 if "error" in c), None),
            "basis": ("every window row priced on the frozen anchor factors; "
                      "weekly_refit rescales the same rows to each week's "
                      "re-fit"),
            "note": "design 9.2 -- same rows, legacy price, censored basis",
        },
        "solver_latency_p95_s": round(float(np.percentile(latencies, 95)), 4),
        "note": (f"Shadow outcomes carry execution_status='{SHADOW_STATUS}' "
                 "and are ineligible for daily.update; the drift ratio is "
                 "the production continuation of the design 9.2 calibration "
                 "diagnostic."),
    }


def _summary(report):
    """The console summary as (label, text) rows; None text drops the row."""
    g, w = report["shadow_gate"], report["window"]
    rv = report["recommendation_vs_legacy"]
    cr = report.get("calibration_regimes") or {}
    ly = report["learning_yield_would_be"]
    td = report.get("tau_initial_derivation")
    bc = report["exploration_budget_would_be"]

    def verdict(row):
        return "PASS" if row["pass"] else "FAIL"

    return [
        ("window", f"{w['basis']} · {w['date_min']} -> {w['date_max']}"
                   + ("" if w["out_of_sample"] else "  [PARTLY IN-SAMPLE]")),
        ("episodes", f"{w['episodes']:,} of {w['population_episodes']:,}"
                     + (f" (sample, seed {w['sample_seed']})"
                        if w["sampled"] else "")),
        ("decisions", f"{report['decision_count']:,} "
                      f"({report['state_rejected_count']} states rejected)"),
        ("event completeness", f"{g['event_completeness']['value']:.4f} "
                               f"-> {verdict(g['event_completeness'])}"),
        ("cost-floor viol.", f"{g['cost_floor_violations']['value']} "
                             f"-> {verdict(g['cost_floor_violations'])}"),
        ("mean discount", f"recommended {rv['mean_recommended_discount']:.3f} "
                          f"vs legacy {rv['mean_legacy_discount']:.3f} "
                          f"(differs {rv['share_hours_differing']:.1%} of hours)"),
        ("drift ratio",
         f"{report['realised_vs_predicted_sold_ratio_at_legacy_price']}"),
        ("calibration",
         f"frozen {cr['frozen_anchor']} | weekly re-fit {cr['weekly_refit']} "
         f"| spread {cr['spread']} ({cr['weeks_refit']} weeks re-fit)"
         if cr.get("weekly_refit") is not None else
         "frozen anchor only -- no week could be re-fit on this window"
         if cr else None),
        ("would-be learning",
         f"{ly['bounded_updates_supported']} bounded updates from this window "
         f"({ly['episodes_per_bounded_update']} episodes per update); "
         f"calendar floor is 1 update per {ly['update_cadence_days']} day(s)"),
        ("  evidence per hour",
         f"{ly['forced_decisions']:,} forced decisions x "
         f"{ly['information_per_forced_decision']} info each · mean move "
         f"{ly['mean_discount_gap_from_reference_forced_pp']}pp from "
         "reference (info is QUADRATIC in this)"
         if ly["forced_decisions"] else None),
        ("tau launch",
         f"{td['tau_initial']:,.2f} derived on the pre-window week "
         f"[{td['week'][0]} .. {td['week'][1]}] ({td['decisions']:,} decisions)"
         if td and td.get("tau_initial") is not None else
         f"config paste in force -- {td['note']}" if td else None),
        ("exploration budget",
         f"spend {bc['implied_daily_spend']:,.0f}/day vs budget "
         f"{bc['daily_budget']:,.0f}/day over {bc['days']} days"),
        ("", bc["verdict"]),
        ("tau",
         f"in force {bc['tau']:,.2f} -> recommended {bc['tau_recommended']:,.2f} "
         f"({bc['tau_recommended_ratio']:.2f}x) on {bc['spread_decisions']:,} "
         f"decisions ({bc['spread_decisions_per_episode']}/episode)"
         if bc["tau_recommended"] else None),
        ("tau controller", bc["tau_controller_trace"]["verdict"]),
    ]


def _print_summary(report, out_path):
    for label, text in _summary(report):
        if text is not None:
            print(f"{label:<19}: {text}" if label else f"{'':<19}  {text}")
    sw = report.get("exploration_budget_sweep") or {}
    rows = [r for r in sw.get("rows", []) if "forced_rate" in r]
    if rows:
        print("budget sweep       : share  x_dmin   forced   spend/day   move   info")
        for r in rows:
            print(f"                     {r['budget_share_of_il']:<6g} "
                  f"{r['delta_min_bias_multiple']:<7g} "
                  f"{r['forced_rate']:>6.1%}  {r['implied_daily_spend']:>10,.0f}   "
                  f"{r['mean_log_move_forced'] or 0:.3f}  {r.get('information_rel', 1):.2f}"
                  + ("  <- in force" if r["in_force"] else ""))
    print(report["shadow_gate"]["verdict"])
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(prog="evaluate.shadow")
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
                        pre_window_frame=full, window_start=start,
                        shadow_path=args.out)

    write_json(args.out, report)
    _print_summary(report, args.out)


if __name__ == "__main__":
    main()
