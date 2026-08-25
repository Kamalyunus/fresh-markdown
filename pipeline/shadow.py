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




def _require_shadow_config(cfg, backtest_path="reports/backtest.json"):
    missing = []
    if cfg["exploration"]["tau_initial"] is None:
        missing.append("exploration.tau_initial (from a PASSING backtest)")
    if missing:
        raise ConfigError("shadow phase blocked by null config: " + "; ".join(missing))

    # Non-null is not enough: tau_initial is hand-pasted and decides day-one
    # spend, before the controller has any spend to correct from.
    report = None
    if os.path.exists(backtest_path):
        with open(backtest_path) as f:
            report = json.load(f)
    stale = explore.tau_provenance_error(cfg, report)
    if stale:
        raise ConfigError("shadow phase blocked by a stale tau: " + stale)


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
        # the controller runs at the operator gate, on the day just closed
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
        "rec_disc": 0.0, "leg_disc": 0.0, "differs": 0,
        "ep_discount_cost": 0.0, "latencies": [],
        "drift": {"mu": [], "r": [], "q": [], "sold": []},
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
               window_basis=HOLDOUT_BASIS, workers=None):
    _require_shadow_config(cfg)
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
    tau = float(cfg["exploration"]["tau_initial"])

    if max_episodes is None:
        max_episodes = cfg["monitoring"]["shadow_gate"]["sample_episodes"]

    # SAMPLE FIRST: the gate reads rates a uniform episode sample estimates,
    # and sampling after the predict step costs a full run for sample evidence
    population = d.episode_id.unique()
    sampled = bool(max_episodes) and len(population) > max_episodes
    if sampled:
        keep = rng.choice(population, max_episodes, replace=False)
        d = d[d.episode_id.isin(keep)]

    # extend to the full window BEFORE predicting (early sell-out must not
    # shorten the DP horizon); sort must include date for cross-midnight runs
    carry = [c for c in d.columns if c not in
             ("episode_id", "date", "hour_of_day", "hours_remaining",
              "starting_inventory", "ending_inventory", "units_sold")]
    d = episodes.extend_to_window(d, carry, cfg["data"]["max_window_hours"])
    d = d.sort_values(["episode_id", "date", "hour_of_day"]).copy()
    d["mu_ref_hat"] = model.predict_mu_ref(d)
    d["r_val"] = [lookup_r(r_lookup, s, c)
                  for s, c in zip(d.subcategory, d.category)]

    groups = list(d.groupby("episode_id", sort=False))

    rejected = {}
    n_dec = n_out = cost_floor_violations = differs = 0
    rec_disc = leg_disc = would_be_cost = 0.0
    n_forced = empty_affordable = 0
    raw_information = 0.0
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
    drift = {"mu": [], "r": [], "q": [], "sold": []}

    # episodes are independent (tau is fixed; the controller walk is
    # post-processing), so the unit of work is one episode and all parallelise
    EP_COLS = ("hour_of_day", "sku_id", "fc", "category", "subcategory",
               "starting_inventory", "ending_inventory", "units_sold",
               "total_discount", "original_price", "cost", "r_val",
               "mu_ref_hat", "date", "is_observed")
    items = [dict({c: g[c].to_numpy() for c in EP_COLS}, episode_id=eid)
             for eid, g in groups]
    ctx = {"cfg": cfg, "tau": tau, "model_version": model.version,
           "seed": seed,
           "cells": {str(c): posterior.get(c) for c in d.category.unique()}}

    for out in map_episodes(_shadow_one, items, ctx, workers):
        for reason, k in out["rejected"].items():
            rejected[reason] = rejected.get(reason, 0) + k
        cost_floor_violations += out["cost_floor_violations"]
        n_forced += out["n_forced"]
        empty_affordable += out["empty_affordable"]
        would_be_cost += out["would_be_cost"]
        raw_information += out["raw_information"]
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
    predicted = expected_min_demand_inventory_vec(
        np.array(drift["mu"]), np.array(drift["r"]),
        np.array(drift["q"], dtype=float), cfg["pricing"]["negbin_max_k"])
    drift_ratio = (float(np.sum(drift["sold"]) / predicted.sum())
                   if predicted.sum() > 0 else None)

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
    il_by_day = {}
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
        "note": ("tau_initial came from the replay's EXPLOIT-ONLY path and, "
                 "before the SpreadLedger fix, from its ENTRY decisions only "
                 "-- roughly one exploration per episode against a system "
                 "that explores every hour, which is most of any large "
                 "multiple here. tau_recommended is the same bisection run on "
                 "this path over these decisions; paste it into "
                 "exploration.tau_initial the way rho and "
                 "mean_forced_hours_per_episode are pasted, after reading "
                 "tau_controller_trace."),
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
        "note": ("Would-be: no price was applied, so this is the evidence the "
                 "recommendations WOULD have bought. Each bounded update moves "
                 f"the posterior mean at most {step} and at most one commits "
                 "per day (human gate), so shifting the mean by X takes at "
                 f"least ceil(X/{step}) calendar days however much evidence "
                 "arrives. Divide episodes_per_bounded_update by the pilot's "
                 "daily episode count for the evidence-side estimate; the "
                 "binding constraint is whichever is larger."),
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
        "learning_yield_would_be": learning_yield,
        "recommendation_vs_legacy": {
            "mean_recommended_discount": round(rec_disc / n_dec, 4),
            "mean_legacy_discount": round(leg_disc / n_dec, 4),
            "share_hours_differing": round(differs / n_dec, 4),
        },
        "realised_vs_predicted_sold_ratio_at_legacy_price": round(drift_ratio, 4)
            if drift_ratio else None,
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
    # episode-scoped date cut, never row-scoped: window_slice keeps a
    # cross-midnight episode whole instead of leaving an orphan tail
    d = episodes.window_slice(d, start, end)
    if d.empty:
        raise SystemExit(f"no episodes opened in [{start}, {end}]")

    report = run_shadow(d, cfg, events_root=args.events_dir,
                        seed=args.seed, max_episodes=args.max_episodes,
                        window_basis=basis, workers=args.workers)

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
    ly = report["learning_yield_would_be"]
    print(f"would-be learning  : {ly['bounded_updates_supported']} bounded "
          f"updates from this window "
          f"({ly['episodes_per_bounded_update']} episodes per update); "
          f"calendar floor is 1 update/day")
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
