"""pipeline.update -- censored NB posterior update, operator-gated.

Daily batch (design 5.11): `--apply` is the operator gate. Exploration
outcomes only; censored NB likelihood; deff-deflated Fisher information;
bounded step; exactly-once commit. Posterior moves on INFORMATION, tau on
SPEND; both persist to artifacts/posterior.json. Refuses to apply while any
hard event-quality gate fails.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import nbinom

from common.config import load_config, deff_from_episodes
from common import episodes
from events.store import EventStore
from pricing import explore
from pricing.posterior import PosteriorStore, bounded_step


def collect_batch(store, posterior, cfg):
    """Match outcomes to decisions, compute event-quality gates, and return
    eligible (decision, outcome) pairs per cell -- plus the loaded event
    lists, so the caller does not re-parse the whole JSONL log a second time
    for tau calibration."""
    decision_list = store.load_decisions()
    decisions = {d["decision_id"]: d for d in decision_list}
    outcomes = store.load_outcomes()

    unmatched = [o for o in outcomes if o["decision_id"] not in decisions]
    matched = [o for o in outcomes if o["decision_id"] in decisions]
    n_events = max(len(outcomes), 1)

    mismatch = sum(
        1 for o in matched
        if abs(o["applied_price"] - decisions[o["decision_id"]]["applied_price"])
        > 1e-6) / max(len(matched), 1)
    dup_or_unmatched = (store.duplicate_counts["decision"]
                        + store.duplicate_counts["outcome"]
                        + len(unmatched)) / n_events

    sc = cfg["monitoring"]["stop_conditions"]
    gates = {
        "duplicate_or_unmatched_rate": {
            "value": round(dup_or_unmatched, 4),
            "threshold": sc["duplicate_or_unmatched_rate"],
            "pass": dup_or_unmatched <= sc["duplicate_or_unmatched_rate"]},
        "price_mismatch_rate": {
            "value": round(mismatch, 4),
            "threshold": sc["price_mismatch_rate"],
            "pass": mismatch <= sc["price_mismatch_rate"]},
    }

    per_cell = {}
    for o in matched:
        if posterior.is_processed(o["outcome_id"]):
            continue
        dec = decisions[o["decision_id"]]
        if not dec["is_exploration"]:      # MVP: exploration outcomes only
            continue
        ratio = (1 - dec["applied_discount"]) / (1 - dec["reference_discount"])
        ok = (dec["reference_mu"] > 0 and dec["dispersion_r"] > 0
              and ratio > 0 and all(math.isfinite(x) for x in
                                    (dec["reference_mu"], dec["dispersion_r"], ratio)))
        if not ok or o.get("execution_status") not in (None, "ok", "success"):
            continue
        cell = posterior.cell_name(dec["category"])
        per_cell.setdefault(cell, []).append((dec, o, ratio))
    return per_cell, gates, decision_list, outcomes


def grid_update(pairs, cell_record, cfg):
    """Evaluate the censored likelihood on the grid, add the log prior,
    normalise with log-sum-exp, and take moments (13.2)."""
    pc = cfg["posterior"]
    grid = np.linspace(pc["epsilon_min"], pc["epsilon_max"], pc["grid_size"])

    k = np.array([o["units_sold"] for _, o, _ in pairs])
    inv = np.array([o["starting_inventory"] for _, o, _ in pairs])
    mu0 = np.array([d["reference_mu"] for d, _, _ in pairs])
    r = np.array([d["dispersion_r"] for d, _, _ in pairs])
    log_ratio = np.log([ratio for _, _, ratio in pairs])
    # `k >= inv` was wrong in one direction and it was the expensive one: it
    # marked every restock hour censored, and a restock hour is where the
    # stock was. `ending_inventory` is required on every outcome, so the same
    # rule the offline fits use applies live -- see episodes.censored_hours.
    end = np.array([o["ending_inventory"] for _, o, _ in pairs])
    censored = episodes.is_censored_hour(inv, k, end)
    lgamma_const = gammaln(k + r) - gammaln(r) - gammaln(k + 1)

    loglik = np.empty(len(grid))
    for i, eps in enumerate(grid):
        mu = np.clip(mu0 * np.exp(eps * log_ratio),
                     cfg["pricing"]["demand_floor"], None)
        p = r / (r + mu)
        exact = lgamma_const + r * np.log(p) + k * np.log1p(-p)
        ll = np.where(censored, nbinom.logsf(np.maximum(inv, 1) - 1, r, p), exact)
        loglik[i] = ll.sum()

    log_prior = -0.5 * ((grid - cell_record["mean"]) / cell_record["std"]) ** 2
    log_post = loglik + log_prior
    log_post -= log_post.max()
    w = np.exp(log_post)
    w /= w.sum()
    raw_mean = float(np.sum(w * grid))
    raw_std = float(np.sqrt(np.sum(w * (grid - raw_mean) ** 2)))

    # SEQUENTIAL PREDICTIVE CHECK: the batch arrived after the current
    # posterior was set, so its log marginal predictive under the PRE-update
    # posterior is an out-of-sample grade of the belief, bracketed by oracle
    # and uniform (design 5.11). Correlated hours inflate all three scores
    # alike -- read differences, never absolutes.
    n = len(pairs)
    log_w_prior = log_prior - logsumexp(log_prior)
    pred_posterior = float((logsumexp(loglik + log_w_prior)) / n)
    pred_uniform = float((logsumexp(loglik) - np.log(len(grid))) / n)
    pred_oracle = float(loglik.max() / n)
    predictive_check = {
        "posterior_log_pred_per_row": round(pred_posterior, 5),
        "uniform_log_pred_per_row": round(pred_uniform, 5),
        "oracle_log_pred_per_row": round(pred_oracle, 5),
        "information_available_per_row": round(pred_oracle - pred_uniform, 5),
        "posterior_minus_uniform": round(pred_posterior - pred_uniform, 5),
        "worse_than_a_flat_prior": bool(pred_posterior < pred_uniform),
        "note": ("batch scored against the PRE-update posterior -- an "
                 "out-of-sample grade of the current belief. Read "
                 "information_available_per_row first; a gap that is a large "
                 "share of a tiny number is still tiny. worse_than_a_flat_"
                 "prior persisting across batches means the posterior "
                 "tightened faster than the evidence justified."),
    }

    # NB Fisher information at the pre-update mean: mu * L^2 * r/(r+mu),
    # never the Poisson mu * L^2 (overstates ~1.6-1.9x; learnings.md)
    mu_at_mean = np.clip(mu0 * np.exp(cell_record["mean"] * log_ratio),
                         cfg["pricing"]["demand_floor"], None)
    information = float(np.sum(
        mu_at_mean * log_ratio ** 2 * r / (r + mu_at_mean)))
    # deff at THIS batch's clustering: how many forced outcomes each
    # episode actually contributed, not a frozen paste
    batch_deff = deff_from_episodes(
        cfg["dispersion"]["rho"], [d["episode_id"] for d, _, _ in pairs])
    effective_information = information / batch_deff
    return raw_mean, raw_std, effective_information, {
        "zero_sales_share": round(float((k == 0).mean()), 4),
        "stockout_share": round(float(censored.mean()), 4),
        "exploration_cost": round(float(sum(
            d["exploration_cost"] for d, _, _ in pairs)), 2),
        "deff_applied": round(batch_deff, 3),
        "predictive_check": predictive_check,
    }


def daily_exploration_spend(decisions, outcomes):
    """Realised exploration cost keyed by the day the outcome finalized.

    The ONE definition of "what exploration cost that day", shared by the tau
    controller and the monitor's stop condition -- attributed on the same day
    key `through` is derived from, so the correction and its backstop cannot
    drift apart.
    """
    dec = {d["decision_id"]: d for d in decisions}
    by_day = {}
    for o in outcomes:
        d = dec.get(o["decision_id"])
        if not d or not d.get("is_exploration") or not o.get("finalized_at"):
            continue
        day = str(o["finalized_at"])[:10]
        by_day[day] = by_day.get(day, 0.0) + float(d["exploration_cost"])
    return by_day


def tau_calibration(decisions, outcomes, posterior, cfg):
    """Move tau toward the budget from realised spend (design 5.8) -- on
    the SAME two numbers the monitor's stop condition compares, so the
    correction and the backstop cannot disagree. Always returns a block;
    `commit` False means nothing to calibrate from."""
    from pipeline.monitor import business_metrics      # sibling; no cycle

    tau_now = posterior.tau(cfg)
    block = {"tau_before": tau_now, "tau_after": tau_now, "commit": False}

    if tau_now is None:
        block["skipped"] = ("exploration.tau_initial is null -- nothing in "
                            "force to calibrate")
        return block

    dates = [o["finalized_at"][:10] for o in outcomes if o.get("finalized_at")]
    if not dates:
        block["skipped"] = "no finalized outcomes"
        return block
    through = max(dates)
    if posterior.tau_calibrated_through() == through:
        block["skipped"] = f"already calibrated through {through}"
        block["through_date"] = through
        return block

    # THE DAY JUST CLOSED, on both sides -- the controller design 5.8
    # specifies and the one shadow's trace walks. Summing every forced
    # decision ever against all-time IL diluted each day's correction by 1/N
    # (a 10x overspend on day 27 moved tau by 0.76x instead of the 0.5x clip)
    # and took the exploration_cost_vs_budget stop condition blind with it,
    # because both compared the same two cumulative totals.
    spend_by_day = daily_exploration_spend(decisions, outcomes)
    realised = float(spend_by_day.get(through, 0.0))
    business = business_metrics(decisions, outcomes, cfg)
    il_by_day = business.get("il_by_close_day") or {}
    cells = posterior.state["cells"]
    if not il_by_day or not cells:
        block["skipped"] = ("no closed-episode IL to project a budget from"
                            if not il_by_day else "no posterior cells")
        return block
    if not realised:
        # zero spend is an ABSENCE OF SIGNAL, not underspend: calibrating on
        # it clips tau upward every day exploration is suspended. Hold still.
        block["skipped"] = ("no exploration on the day just closed -- nothing "
                            "to calibrate from")
        block["through_date"] = through
        return block

    # the widest ROUTED cell's std, matching the monitor: the budget is sized
    # for the cell that still has the most to learn. GLOBAL, when nothing
    # routes to it, never narrows and would pin this at the launch std.
    widest_std = posterior.widest_std()
    # a share of TRAILING realised IL, never the same day's own
    trailing_il = explore.trailing_daily_il(il_by_day, through, cfg)
    budget = explore.budget_today(trailing_il, widest_std, cfg)
    block.update({
        "through_date": through,
        "realised_exploration_cost": round(realised, 1),
        "markdown_il": round(float(trailing_il), 1),
        "markdown_il_basis": (
            f"mean realised IL/day over the trailing "
            f"{cfg['exploration']['budget_il_window_days']} days to {through}"),
        "widest_posterior_std": widest_std,
        "budget": round(budget, 1),
        "tau_after": round(explore.tau_next(tau_now, budget, realised, cfg), 2),
        "commit": True,
    })
    block["clipped"] = block["tau_after"] in (
        round(tau_now * cfg["exploration"]["tau_adjust_clip"][0], 2),
        round(tau_now * cfg["exploration"]["tau_adjust_clip"][1], 2))
    return block


def calibration_current(cfg, today=None):
    """Does the level-calibration schedule cover the week being priced?
    A week past the schedule takes the FROZEN fallback silently, so this is
    the hard gate against learning from stale factors. Static calibration
    passes; `today` is injectable for tests."""
    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        return {"value": "none", "threshold": "schedule covers today",
                "pass": True, "note": "no calibration artifact; factors are 1.0"}
    with open(path) as f:
        sched = (json.load(f).get("schedule") or {}).get("by_week") or {}
    if not sched:
        return {"value": "static", "threshold": "schedule covers today",
                "pass": True,
                "note": "artifact carries no schedule -- one frozen factor "
                        "set, nothing to keep current"}
    now = pd.Timestamp(today) if today is not None else pd.Timestamp.now("UTC")
    week = pd.Timestamp(now).tz_localize(None) if now.tzinfo else now
    week = week.to_period("W").start_time.strftime("%Y-%m-%d")
    last = max(sched)
    ok = week <= last
    return {
        "value": week,
        "threshold": f"<= {last} (last fitted week)",
        "pass": bool(ok),
        "note": ("schedule covers this week" if ok else
                 f"THE WEEKLY RE-FIT WAS MISSED: pricing is in week {week} but "
                 f"the schedule stops at {last}, so production is running on "
                 "the frozen fallback factors. Run `bootstrap.train_baseline "
                 "--fit-calibration` and `bootstrap.seal`, then re-run "
                 "(RUNBOOK Lane C)."),
    }


def run(cfg, apply=False, events_root=None, posterior_path=None, today=None):
    store = EventStore(cfg, root=events_root)
    posterior = PosteriorStore(cfg, path=posterior_path)
    per_cell, gates, decision_list, outcome_list = collect_batch(
        store, posterior, cfg)
    # a HARD gate beside the event-quality ones: learning from prices set on
    # stale factors banks evidence about a model that is not the one running
    gates["calibration_schedule_current"] = calibration_current(cfg, today)

    hard_fail = [name for name, g in gates.items() if not g["pass"]]
    report = {"event_quality_gates": gates, "cells": {}, "applied": False}

    for cell, pairs in sorted(per_cell.items()):
        rec = posterior.state["cells"][cell]
        raw_mean, raw_std, eff_info, diag = grid_update(pairs, rec, cfg)
        new_mean, new_std, clipped = bounded_step(
            rec["mean"], rec["std"], raw_mean, raw_std, cfg)
        # `pairs` is every eligible outcome not yet consumed by a revision, so
        # it already spans however many days it took to accumulate -- eff_info
        # is the whole batch's information, not one day's. No running counter:
        # adding to one while re-reading the same outcomes would double count.
        trigger = eff_info >= cfg["learning"]["information_increment"]
        oldest = min((o.get("finalized_at") for _, o, _ in pairs
                      if o.get("finalized_at")), default=None)
        age_days = None
        if oldest is not None:
            # tolerate a tz-naive finalized_at: the contract says UTC with
            # offset, but a producer omitting it must age the batch, not
            # crash the whole daily run before any gate can report
            ts = pd.Timestamp(oldest)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            age_days = round((pd.Timestamp.now("UTC")
                              - ts).total_seconds() / 86400, 2)

        report["cells"][cell] = {
            "forced_outcomes": len(pairs),
            "effective_information": round(eff_info, 3),
            "information_pending": round(eff_info, 3),
            "information_required": cfg["learning"]["information_increment"],
            "update_triggered": trigger,
            # a batch that keeps growing without triggering is the learning
            # loop stalling; this surfaces it before the 21-day flat alert
            "batch_oldest_outcome_age_days": age_days,
            # UNROUNDED, all four: `max_mean_step` is a SAFETY bound checked
            # downstream, and rounding one side of it makes the report
            # disagree with itself by up to 5e-5. Rounding for humans belongs
            # in the printed line below, which already formats to 3dp.
            "mean_before": rec["mean"], "std_before": rec["std"],
            "raw_mean": raw_mean, "raw_std": raw_std,
            "proposed_mean": new_mean, "proposed_std": new_std,
            "bound_clipped": clipped,
            **diag,
        }

        if apply and not hard_fail:
            outcome_ids = [o["outcome_id"] for _, o, _ in pairs]
            posterior.commit_update(cell, new_mean, new_std, len(pairs),
                                    eff_info, outcome_ids, applied=trigger)

    # tau moves on SPEND, not on evidence, so it is calibrated whether or not
    # any cell crossed the information threshold -- a day that explored and
    # learned nothing still cost money, and that is exactly what tau prices.
    report["tau_calibration"] = tau_calibration(
        decision_list, outcome_list, posterior, cfg)

    if apply:
        if hard_fail:
            report["refused"] = (f"hard gate(s) failed: {hard_fail}; "
                                 "no update applied")
        else:
            report["applied"] = True
            tc = report["tau_calibration"]
            if tc["commit"]:
                posterior.commit_tau(tc["tau_after"], tc["through_date"])
    return report


def main():
    ap = argparse.ArgumentParser(prog="pipeline.update")
    ap.add_argument("--apply", action="store_true",
                    help="apply bounded posterior updates (operator gate)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = run(cfg, apply=args.apply)

    for name, g in report["event_quality_gates"].items():
        print(f"gate {name}: {g['value']} vs {g['threshold']} "
              f"-> {'PASS' if g['pass'] else 'FAIL'}")
    for cell, c in report["cells"].items():
        print(f"[{cell}] forced={c['forced_outcomes']} "
              f"info+={c['effective_information']} "
              f"(pending {c['information_pending']}, "
              f"trigger={c['update_triggered']}) "
              f"mean {c['mean_before']:+.3f}->{c['proposed_mean']:+.3f} "
              f"std {c['std_before']:.3f}->{c['proposed_std']:.3f}"
              + ("  [CLIPPED -> operator review]" if c["bound_clipped"] else ""))

    tc = report["tau_calibration"]
    if tc.get("skipped"):
        print(f"tau: {tc['tau_before']} unchanged -- {tc['skipped']}")
    else:
        print(f"tau: {tc['tau_before']} -> {tc['tau_after']}  "
              f"(spent {tc['realised_exploration_cost']} of {tc['budget']} "
              f"through {tc['through_date']})"
              + ("  [CLIP BOUND]" if tc.get("clipped") else ""))

    if "refused" in report:
        print("REFUSED:", report["refused"])
    elif report["applied"]:
        print("applied bounded posterior updates")
    else:
        print("monitor only -- rerun with --apply to commit")


if __name__ == "__main__":
    main()
