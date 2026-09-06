"""daily.update -- censored NB posterior update, operator-gated.

Daily batch (design 5.11): `--apply` is the operator gate. Exploration
outcomes only; censored NB likelihood; deff-deflated Fisher information;
bounded step; exactly-once commit. Posterior moves on INFORMATION, tau on
SPEND; both persist to artifacts/posterior.json. Refuses to apply while any
hard event-quality gate fails.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import nbinom

from common.config import load_config, deff_from_episodes
from common import episodes
from common.io import read_json
from events.store import EventStore
from events.pairs import (match_pairs, decision_day, is_learnable, has_stock,
                          is_restocked, quality_counts, quality_rates)
from engine import explore
from engine.posterior import PosteriorStore, bounded_step
from fit.train_baseline import schedule_reaches


def collect_batch(store, posterior, cfg):
    """Match outcomes to decisions, compute the event-quality gates, and
    return the batch as a dict: `per_cell` -- the (decision, outcome, ratio)
    triples a cell may learn from; `gates`; `decisions`, `outcomes` and
    `pairs` (every matched pair) so tau calibration does not re-parse the
    log or re-pair it; `event_quality`, the windowed counts the gates read
    (events.pairs.quality_counts -- the monitor's stop condition reads the
    same); and the pairs the learner refused: `excluded_no_stock` (the hour
    opened empty -- no information, and a wrong censored term) and
    `excluded_restock` (no single q to empty), the same rule assurance
    grades on (events.pairs.learnable_with_stock)."""
    decision_list = store.load_decisions()
    outcomes = store.load_outcomes()
    pairs = match_pairs(decision_list, outcomes)

    quality = quality_counts(decision_list, outcomes, cfg,
                             store.duplicate_counts, pairs)
    rates = quality_rates(quality)
    sc = cfg["monitoring"]["stop_conditions"]
    gates = {name: {"value": round(rates[name], 4), "threshold": sc[name],
                    "pass": rates[name] <= sc[name],
                    "window_days": quality["event_quality_window_days"]}
             for name in ("duplicate_or_unmatched_rate", "price_mismatch_rate")}

    per_cell = {}
    excluded = {"excluded_no_stock": 0, "excluded_restock": 0}
    for dec, o in pairs:
        if posterior.is_processed(o["outcome_id"]):
            continue
        if not dec["is_exploration"]:      # MVP: exploration outcomes only
            continue
        ratio = (1 - dec["applied_discount"]) / (1 - dec["reference_discount"])
        ok = (dec["reference_mu"] > 0 and dec["dispersion_r"] > 0
              and ratio > 0 and all(math.isfinite(x) for x in
                                    (dec["reference_mu"], dec["dispersion_r"], ratio)))
        if not ok or not is_learnable(o):
            continue
        if not has_stock(o):
            excluded["excluded_no_stock"] += 1
            continue
        if is_restocked(o):
            excluded["excluded_restock"] += 1
            continue
        cell = posterior.cell_name(dec["category"])
        per_cell.setdefault(cell, []).append((dec, o, ratio))
    return {"per_cell": per_cell, "gates": gates, "decisions": decision_list,
            "outcomes": outcomes, "pairs": pairs, "event_quality": quality,
            **excluded}


def grid_update(pairs, cell_record, cfg):
    """Evaluate the censored likelihood on the grid, add the log prior,
    normalise with log-sum-exp, and take moments (design 5.11)."""
    pc = cfg["posterior"]
    grid = np.linspace(pc["epsilon_min"], pc["epsilon_max"], pc["grid_size"])

    k = np.array([o["units_sold"] for _, o, _ in pairs])
    inv = np.array([o["starting_inventory"] for _, o, _ in pairs])
    mu0 = np.array([d["reference_mu"] for d, _, _ in pairs])
    r = np.array([d["dispersion_r"] for d, _, _ in pairs])
    log_ratio = np.log([ratio for _, _, ratio in pairs])
    # censoring is the shared rule (the shelf EMPTIED), never `sold >= q`,
    # which reads a restocked hour as censored
    end = np.array([o["ending_inventory"] for _, o, _ in pairs])
    censored = episodes.is_censored_hour(inv, k, end)
    lgamma_const = gammaln(k + r) - gammaln(r) - gammaln(k + 1)

    # the censored term only where the shelf emptied: logsf is the costly
    # call, and on an uncensored row it was computed and thrown away
    inv_c, r_c = np.maximum(inv[censored], 1) - 1, r[censored]
    loglik = np.empty(len(grid))
    for i, eps in enumerate(grid):
        mu = np.clip(mu0 * np.exp(eps * log_ratio),
                     cfg["pricing"]["demand_floor"], None)
        p = r / (r + mu)
        ll = lgamma_const + r * np.log(p) + k * np.log1p(-p)
        if len(inv_c):
            ll[censored] = nbinom.logsf(inv_c, r_c, p[censored])
        loglik[i] = ll.sum()

    log_prior = -0.5 * ((grid - cell_record["mean"]) / cell_record["std"]) ** 2
    log_post = loglik + log_prior
    log_post -= log_post.max()
    w = np.exp(log_post)
    w /= w.sum()
    raw_mean = float(np.sum(w * grid))
    raw_std = float(np.sqrt(np.sum(w * (grid - raw_mean) ** 2)))

    # sequential predictive check (design 5.11): the batch's log marginal
    # predictive under the PRE-update posterior, bracketed by oracle and
    # uniform -- read differences, never absolutes
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
    # never the Poisson mu * L^2 (design 5.11)
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


def finalized_days(decisions, outcomes, pairs=None):
    """ONE pass over the matched pairs, keyed on the TRADING day the decision
    priced (events.pairs.decision_day, never the UTC clock of
    `finalized_at`). Every stored outcome is final -- `finalized_at` is in
    events.store.OUTCOME_REQUIRED -- so a matched pair is a priced day.
    Returns (priced_days ascending, {day: realised exploration spend} over
    the forced decisions) -- the day key and the spend the tau controller
    and the monitor's stop condition both read, so the correction and its
    backstop cannot drift apart. `pairs` is match_pairs(decisions, outcomes)
    if the caller already built it."""
    days, spend = set(), {}
    for d, o in (match_pairs(decisions, outcomes) if pairs is None else pairs):
        day = decision_day(d)
        days.add(day)
        if d.get("is_exploration"):
            spend[day] = spend.get(day, 0.0) + float(d["exploration_cost"])
    return sorted(days), spend


def tau_calibration(decisions, outcomes, posterior, cfg, widest_std=None,
                    pairs=None):
    """Move tau toward the budget from realised spend (design 5.8) -- on
    the SAME two numbers the monitor's stop condition compares, so the
    correction and the backstop cannot disagree. Always returns a block;
    `commit` False means nothing to calibrate from.

    `widest_std` is the posterior std the budget is sized for; it defaults
    to the store's current widest routed std. `run` passes the value read
    BEFORE any cell is committed, so a dry run and `--apply` price the same
    days from the same posterior. `pairs` is match_pairs(decisions,
    outcomes) if the caller already built it (collect_batch did)."""
    from daily.monitor import business_metrics, settled_episodes  # sibling; no cycle

    tau_now = posterior.tau(cfg)
    block = {"tau_before": tau_now, "tau_after": tau_now, "commit": False}

    if tau_now is None:
        block["skipped"] = ("exploration.tau_initial is null -- nothing in "
                            "force to calibrate")
        return block

    if pairs is None:
        pairs = match_pairs(decisions, outcomes)      # once, for both reads
    priced_days, spend_by_day = finalized_days(decisions, outcomes, pairs)
    if not priced_days:
        block["skipped"] = "no finalized outcomes"
        return block
    through = priced_days[-1]
    done = posterior.tau_calibrated_through()
    if done == through:
        block["skipped"] = f"already calibrated through {through}"
        block["through_date"] = through
        return block

    # EVERY closed day since the last calibration, in order, one step each
    # -- design 5.8 is a daily walk, and a weekly batch is seven steps, not
    # one graded day and six skipped. Each day's spend is graded against
    # the budget priced from the days before it. Zero realised spend on a
    # priced day is NOT skipped: nothing was affordable, which is exactly
    # the under-spend the rule raises tau on, and the only way a tau cut
    # below the smallest spread ever recovers.
    days = [d for d in priced_days if done is None or d > str(done)]
    if not days:
        # the posterior says tau is calibrated PAST the store's latest priced
        # day: the store is behind (restored from an older copy, or pointed
        # at the wrong directory). Nothing to walk -- report, never index
        block["skipped"] = (f"posterior is calibrated through {done}, ahead "
                            f"of the store's latest priced day {through}; "
                            "no day to walk")
        block["through_date"] = through
        return block
    business = business_metrics(decisions, outcomes,
                                settled_episodes(decisions, outcomes, pairs))
    il_by_day = business.get("il_by_close_day") or {}
    cells = posterior.state["cells"]
    if not il_by_day or not cells:
        block["skipped"] = ("no closed-episode IL to project a budget from"
                            if not il_by_day else "no posterior cells")
        return block

    # the widest ROUTED cell's std, matching the monitor: the budget is sized
    # for the cell that still has the most to learn
    if widest_std is None:
        widest_std = posterior.widest_std()
    tau_end, rows = explore.walk_tau(
        tau_now, days, lambda day, _tau: spend_by_day.get(day, 0.0),
        il_by_day, widest_std, cfg)
    last = rows[-1]
    block.update({
        "through_date": through,
        "days_walked": len(rows),
        "by_day": rows,
        # the last day walked, for the printed line
        "realised_exploration_cost": last["spend"],
        "markdown_il": round(float(explore.trailing_daily_il(
            il_by_day, through, cfg)), 1),
        "markdown_il_basis": (
            f"mean realised IL/day over the trailing "
            f"{cfg['exploration']['budget_il_window_days']} days to {through}"),
        "widest_posterior_std": widest_std,
        "budget": last["budget"],
        "tau_after": round(float(tau_end), 2),
        "commit": True,
        # the walk says which steps sat on a clip bound; never inferred
        # from the rounded taus
        "clipped": any(r["clipped"] for r in rows),
        # days the controller did not act on (no base, or one shorter than
        # its window -- explore.budget_held); the printed line names them
        "held_days": [r["day"] for r in rows if r.get("held")],
    })
    return block


def calibration_current(cfg, today=None):
    """Does the level-calibration schedule cover the week being priced?
    A week past the schedule takes the FROZEN fallback silently, so this is
    the hard gate against learning from stale factors. Static calibration
    passes. `today` is the day being priced -- `run` passes the store's
    latest TRADING day (events.pairs.decision_day), never the UTC wall
    clock, which rolls into the next week hours before the shop does; the
    clock is the fallback only while no decision has been priced."""
    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        return {"value": "none", "threshold": "schedule covers today",
                "pass": True, "note": "no calibration artifact; factors are 1.0"}
    sched = read_json(path).get("schedule") or {}
    last = schedule_reaches(sched)
    if last is None:
        return {"value": "static", "threshold": "schedule covers today",
                "pass": True,
                "note": "artifact carries no schedule -- one frozen factor "
                        "set, nothing to keep current"}
    now = pd.Timestamp(today) if today is not None else pd.Timestamp.now("UTC")
    week = pd.Timestamp(now).tz_localize(None) if now.tzinfo else now
    week = episodes.week_start(week).strftime("%Y-%m-%d")
    ok = week <= last
    held = week in (sched.get("weeks_unfitted_held_at_1") or [])
    return {
        "value": week,
        "threshold": f"<= {last} (last week the schedule covers)",
        "pass": bool(ok),
        # a covered-but-thin week prices on the frozen anchor BY DECISION
        # of the re-fit (below calibration_min_anchor_rows): not a missed
        # cron, so not a refusal -- but said, since the factors are stale
        "held_at_anchor": held,
        "note": (("schedule covers this week" if not held else
                  f"the re-fit ran but held week {week} at the frozen anchor "
                  "(too few anchor rows in its trailing window); production "
                  "is pricing on the anchor factors this week") if ok else
                 f"THE WEEKLY RE-FIT WAS MISSED: pricing is in week {week} but "
                 f"the schedule stops at {last}, so production is running on "
                 "the frozen fallback factors. Run `fit.train_baseline "
                 "--fit-calibration` and `ops.seal`, then re-run "
                 "(RUNBOOK Lane C)."),
    }


def run(cfg, apply=False, events_root=None, posterior_path=None, today=None,
        calibrate_tau=False, resume_exploration=False):
    store = EventStore(cfg, root=events_root)
    posterior = PosteriorStore(cfg, path=posterior_path)
    batch = collect_batch(store, posterior, cfg)
    gates = batch["gates"]
    # a HARD gate beside the event-quality ones: learning from prices set on
    # stale factors banks evidence about a model that is not the one running.
    # Judged on the latest TRADING day priced, not the wall clock
    if today is None:
        today = max((decision_day(d) for d in batch["decisions"]), default=None)
    gates["calibration_schedule_current"] = calibration_current(cfg, today)

    hard_fail = [name for name, g in gates.items() if not g["pass"]]
    report = {"event_quality_gates": gates, "cells": {}, "applied": False,
              "batch": {k: batch[k] for k in ("excluded_no_stock",
                                              "excluded_restock")},
              "exploration_suspended": posterior.exploration_suspended()}

    # tau moves on SPEND, not on evidence, so it is calibrated whether or not
    # any cell crosses the information threshold -- a day that explored and
    # learned nothing still cost money, and that is exactly what tau prices.
    # Computed BEFORE the commit loop: the budget is sized on the posterior
    # the days were priced under, so a dry run and --apply agree on tau_after.
    report["tau_calibration"] = tau_calibration(
        batch["decisions"], batch["outcomes"], posterior, cfg,
        widest_std=posterior.widest_std(), pairs=batch["pairs"])

    for cell, pairs in sorted(batch["per_cell"].items()):
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

    # tau is committed by --apply AND by --calibrate-tau: it moves on spend,
    # not evidence, so it needs no operator and must not wait for the
    # learning cadence (weekly --apply with a daily tau)
    if (apply or calibrate_tau) and hard_fail:
        report["refused"] = (f"hard gate(s) failed: {hard_fail}; "
                             "no update applied")
    elif apply or calibrate_tau:
        tc = report["tau_calibration"]
        if tc["commit"]:
            posterior.commit_tau(tc["tau_after"], tc["through_date"])
        report["tau_committed"] = bool(tc["commit"])
        report["applied"] = bool(apply)

    # the HUMAN gate on exploration (design 5.12): a fired stop condition
    # suspends forced exploration and nothing resumes it automatically --
    # the operator reads the monitor, fixes the cause, and clears it here
    if resume_exploration:
        report["exploration_resumed"] = posterior.resume_exploration()
        report["exploration_suspended"] = None
    return report


def main():
    ap = argparse.ArgumentParser(prog="daily.update")
    ap.add_argument("--apply", action="store_true",
                    help="apply bounded posterior updates (operator gate, "
                         "every learning.update_cadence_days)")
    ap.add_argument("--calibrate-tau", action="store_true",
                    help="commit the tau walk only (daily, no operator): "
                         "spend, not evidence")
    ap.add_argument("--resume-exploration", action="store_true",
                    help="clear the exploration suspension a stop condition "
                         "set (operator gate: nothing resumes it "
                         "automatically)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = run(cfg, apply=args.apply, calibrate_tau=args.calibrate_tau,
                 resume_exploration=args.resume_exploration)

    for name, g in report["event_quality_gates"].items():
        print(f"gate {name}: {g['value']} vs {g['threshold']} "
              f"-> {'PASS' if g['pass'] else 'FAIL'}"
              + (f"  (trailing {g['window_days']} trading days)"
                 if "window_days" in g else ""))
    b = report["batch"]
    if b["excluded_no_stock"] or b["excluded_restock"]:
        print(f"batch excluded: {b['excluded_no_stock']} hour(s) that opened "
              f"empty, {b['excluded_restock']} restocked hour(s)")
    for cell, c in report["cells"].items():
        print(f"[{cell}] forced={c['forced_outcomes']} "
              f"info+={c['effective_information']} "
              f"(required {c['information_required']}, "
              f"trigger={c['update_triggered']}) "
              f"mean {c['mean_before']:+.3f}->{c['proposed_mean']:+.3f} "
              f"std {c['std_before']:.3f}->{c['proposed_std']:.3f}"
              + ("  [CLIPPED -> operator review]" if c["bound_clipped"] else ""))

    tc = report["tau_calibration"]
    if tc.get("skipped"):
        print(f"tau: {tc['tau_before']} unchanged -- {tc['skipped']}")
    else:
        last = tc["by_day"][-1]
        print(f"tau: {tc['tau_before']} -> {tc['tau_after']}  "
              f"({tc['days_walked']} day(s) walked through {tc['through_date']}; "
              + (f"last day HELD -- {last['held']}" if last.get("held") else
                 f"last day spent {tc['realised_exploration_cost']} of "
                 f"{tc['budget']}") + ")"
              + ("  [CLIP BOUND]" if tc.get("clipped") else "")
              + (f"  [{len(tc['held_days'])} day(s) held]" if tc["held_days"] else ""))

    if "refused" in report:
        print("REFUSED:", report["refused"])
    elif report["applied"]:
        print("applied bounded posterior updates")
    elif report.get("tau_committed"):
        print("tau committed; posterior cells untouched -- --apply is the "
              "operator gate")
    else:
        print("monitor only -- rerun with --apply to commit")

    if "exploration_resumed" in report:
        cleared = report["exploration_resumed"]
        print("exploration resumed -- cleared: "
              + (f"suspended since {cleared['since']} for "
                 f"{', '.join(cleared['reasons'])}" if cleared
                 else "nothing (exploration was not suspended)"))
    elif report["exploration_suspended"]:
        s = report["exploration_suspended"]
        print(f"EXPLORATION SUSPENDED since {s['since']} "
              f"({', '.join(s['reasons'])}); exploitation continues. Clear "
              "with --resume-exploration once the cause is fixed.")


if __name__ == "__main__":
    main()
