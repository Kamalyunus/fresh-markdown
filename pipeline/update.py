"""pipeline.update -- censored NB posterior update, operator-gated.

PRD sections 13 and 14. Runs as a daily batch:

    python3 -m pipeline.update             # monitor only
    python3 -m pipeline.update --apply     # apply bounded posterior updates

Eligibility (13.1): only finalized outcomes whose decision has
is_exploration = true. Exploitation outcomes are never used -- prices chosen
by the DP depend on the current posterior, so learning from them feeds the
model's own beliefs back into itself. reference_mu comes from the decision
event, never recomputed.

The likelihood (13.2) retains zero sales through the exact P(D = 0) term and
treats stockout hours as censored (P(D >= inventory)), never as exact counts.
Accumulated information is deflated by deff (13.3). Each update is bounded
(13.4) and updates are exactly-once (13.5): the posterior store commits the
revision and the consumed outcome IDs in a single atomic write.

The command refuses to apply while any hard event-quality gate fails (14.1).
"""

import argparse
import math

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom

from common.config import load_config, deff
from events.store import EventStore
from pricing.posterior import PosteriorStore, bounded_step


def collect_batch(store, posterior, cfg):
    """Match outcomes to decisions, compute event-quality gates, and return
    eligible (decision, outcome) pairs per cell."""
    decisions = {d["decision_id"]: d for d in store.load_decisions()}
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
        if not dec["is_exploration"] and not cfg["learning"]["use_exploitation_outcomes"]:
            continue
        ratio = (1 - dec["applied_discount"]) / (1 - dec["reference_discount"])
        ok = (dec["reference_mu"] > 0 and dec["dispersion_r"] > 0
              and ratio > 0 and all(math.isfinite(x) for x in
                                    (dec["reference_mu"], dec["dispersion_r"], ratio)))
        if not ok or o.get("execution_status") not in (None, "ok", "success"):
            continue
        cell = posterior.cell_name(dec["category"])
        per_cell.setdefault(cell, []).append((dec, o, ratio))
    return per_cell, gates


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
    censored = k >= inv
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

    # information at the pre-update posterior mean (13.3)
    mu_at_mean = np.clip(mu0 * np.exp(cell_record["mean"] * log_ratio),
                         cfg["pricing"]["demand_floor"], None)
    information = float(np.sum(mu_at_mean * log_ratio ** 2))
    effective_information = information / deff(cfg)
    return raw_mean, raw_std, effective_information, {
        "zero_sales_share": round(float((k == 0).mean()), 4),
        "stockout_share": round(float(censored.mean()), 4),
        "exploration_cost": round(float(sum(
            d["exploration_cost"] for d, _, _ in pairs)), 2),
    }


def run(cfg, apply=False, events_root=None, posterior_path=None):
    store = EventStore(cfg, root=events_root)
    posterior = PosteriorStore(cfg, path=posterior_path)
    per_cell, gates = collect_batch(store, posterior, cfg)

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
            age_days = round((pd.Timestamp.now("UTC")
                              - pd.Timestamp(oldest)).total_seconds() / 86400, 2)

        report["cells"][cell] = {
            "forced_outcomes": len(pairs),
            "effective_information": round(eff_info, 3),
            "information_pending": round(eff_info, 3),
            "information_required": cfg["learning"]["information_increment"],
            "update_triggered": trigger,
            # a batch that keeps growing without triggering is the learning
            # loop stalling; this surfaces it before the 21-day flat alert
            "batch_oldest_outcome_age_days": age_days,
            "mean_before": rec["mean"], "std_before": rec["std"],
            "raw_mean": round(raw_mean, 4), "raw_std": round(raw_std, 4),
            "proposed_mean": round(new_mean, 4), "proposed_std": round(new_std, 4),
            "bound_clipped": clipped,
            **diag,
        }

        if apply and not hard_fail:
            outcome_ids = [o["outcome_id"] for _, o, _ in pairs]
            posterior.commit_update(cell, new_mean, new_std, len(pairs),
                                    eff_info, outcome_ids, applied=trigger)

    if apply:
        if hard_fail:
            report["refused"] = (f"hard event-quality gate(s) failed: {hard_fail}; "
                                 "no update applied")
        else:
            report["applied"] = True
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
    if "refused" in report:
        print("REFUSED:", report["refused"])
    elif report["applied"]:
        print("applied bounded posterior updates")
    else:
        print("monitor only -- rerun with --apply to commit")


if __name__ == "__main__":
    main()
