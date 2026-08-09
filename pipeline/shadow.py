"""pipeline.shadow -- section 19 phase-1 harness: decisions logged, NO prices
applied.

Runs the full production decision path (validate -> DP -> explore -> decision
event) against observed FLC data while the legacy policy keeps pricing.
Outcomes are built from what actually happened under legacy prices and are
stamped execution_status="shadow_not_applied", which makes them ineligible for
pipeline.update (the recommended price was never in force, so they are not
evidence about it).

Exit gate (section 19): event completeness above min_event_completeness,
matched decision rate above min_matched_decision_rate, and ZERO cost-floor
violations, before any price is applied.

State construction: inventory and the monotonicity anchor come from reality --
the anchor entering hour t is the legacy discount applied at t-1 (entry has no
anchor). The recommendation answers "what would we have done from the real
state", not "what would our price path have been".

Usage:
    python3 -m pipeline.shadow --input data/prepared.parquet \
        --out reports/shadow.json [--date-start D --date-end D] \
        [--max-episodes N] [--seed N]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config, deff, ConfigError
from common import episodes
from bootstrap.train_baseline import BaselineModel
from bootstrap.fit_dispersion import lookup_r
from events.store import EventStore
from pricing.demand import expected_min_demand_inventory_vec
from inference.decide import decide, StateRejected
from pricing.posterior import PosteriorStore

SHADOW_STATUS = "shadow_not_applied"


def _require_shadow_config(cfg):
    missing = []
    if cfg["baseline_model"]["apply_level_calibration"] is None:
        missing.append("baseline_model.apply_level_calibration (section 9.3 decision)")
    if cfg["exploration"]["tau_initial"] is None:
        missing.append("exploration.tau_initial (from a PASSING backtest)")
    if missing:
        raise ConfigError("shadow phase blocked by null config: " + "; ".join(missing))


def run_shadow(d, cfg, events_root=None, seed=0, max_episodes=None):
    _require_shadow_config(cfg)
    model = BaselineModel(cfg)
    posterior = PosteriorStore(cfg)
    with open(cfg["dispersion"]["r_lookup_path"]) as f:
        r_lookup = json.load(f)
    store = EventStore(cfg, root=events_root or cfg["events"]["shadow_store_dir"])
    rng = np.random.default_rng(seed)
    tau = float(cfg["exploration"]["tau_initial"])

    if max_episodes is None:
        max_episodes = cfg["monitoring"]["shadow_gate"]["sample_episodes"]

    # SAMPLE FIRST. The gate reads rates (completeness, matched, drift) plus a
    # violation count, and a uniform episode sample estimates all of them --
    # but only if the sample is drawn before the work, not after. Predicting
    # mu_ref over every row and then discarding most of them costs the full
    # run for a sample's worth of evidence.
    population = d.episode_id.unique()
    sampled = bool(max_episodes) and len(population) > max_episodes
    if sampled:
        keep = rng.choice(population, max_episodes, replace=False)
        d = d[d.episode_id.isin(keep)]

    # extend each episode to its full window BEFORE predicting: rows stop at
    # zero inventory, so an episode that sold out early would hand the DP a
    # horizon shortened by its own realised outcome. Sorting must include the
    # date -- hour_of_day alone scrambles a window that runs past midnight.
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
    latencies = []
    drift = {"mu": [], "r": [], "q": [], "sold": []}

    for eid, g in groups:
        mu_path = list(g.mu_ref_hat.to_numpy())
        anchor = None
        for t in range(len(g)):
            row = g.iloc[t]
            if not row.is_observed:      # window tail: no outcome to record
                continue
            q = int(row.starting_inventory)
            if q <= 0:                      # restock gap: no decision this hour
                anchor = float(row.total_discount)
                continue
            state = {
                "episode_id": eid, "sku_id": int(row.sku_id), "fc": row.fc,
                "category": row.category, "subcategory": row.subcategory,
                "hour_of_day": int(row.hour_of_day),
                "hours_remaining": len(g) - t, "q": q,
                "original_price": float(row.original_price),
                "cost": float(row.cost), "r": float(row.r_val),
                "mu_ref_path": mu_path[t:], "current_discount": anchor,
            }
            try:
                evt = decide(state, posterior, store, cfg, rng, tau, model.version)
            except StateRejected as e:
                rejected[str(e)] = rejected.get(str(e), 0) + 1
                anchor = float(row.total_discount)
                continue

            n_dec += 1
            latencies.append(evt["solver_latency_s"])
            if evt["applied_price"] < evt["cost"] - 1e-6:
                cost_floor_violations += 1
            if evt["is_exploration"]:
                n_forced += 1
                would_be_cost += evt["exploration_cost"]
                # would-be learning yield. pipeline.update accumulates
                # mu * (log price ratio)^2 with the ratio taken against the
                # REFERENCE discount, not against the DP optimum -- so what
                # drives information is how far the applied price sits from
                # the anchor, not how large the perturbation was.
                lr = np.log((1 - evt["applied_discount"])
                            / (1 - evt["reference_discount"]))
                mu_rec = max(mu_path[t] * np.exp(evt["epsilon_posterior_mean"] * lr),
                             cfg["pricing"]["demand_floor"])
                raw_information += mu_rec * lr ** 2
            if evt["affordable_set_size"] == 0:
                empty_affordable += 1
            legacy_d = float(row.total_discount)
            rec_disc += evt["applied_discount"]
            leg_disc += legacy_d
            if abs(evt["applied_discount"] - legacy_d) > 1e-9:
                differs += 1

            # outcome = what actually happened under the LEGACY price
            sold = int(row.units_sold)
            ending = int(row.ending_inventory)
            outcome = {
                "event": "outcome",
                "outcome_id": f"shadow-{evt['decision_id']}",
                "decision_id": evt["decision_id"],
                "units_sold": sold, "starting_inventory": q,
                "ending_inventory": ending,
                "applied_price": float(row.original_price * (1 - legacy_d)),
                "is_stockout": sold >= q,
                "execution_status": SHADOW_STATUS,
                "finalized_at": pd.Timestamp.now("UTC").isoformat(),
            }
            # The store quarantines any outcome whose inventory does not
            # reconcile and carries no documented reason, so both legitimate
            # breaks must be named -- and only those two.
            #   ending > leftover : stock was added mid-window
            #   ending < leftover at the window close: the source wrote the
            #     remainder off, which is ~49.5% of episodes. Unnamed, every
            #     one of them would quarantine and sink event completeness.
            #   ending < leftover mid-window: unexplained shrinkage. Left
            #     undocumented ON PURPOSE so it quarantines.
            leftover = max(q - sold, 0)
            if ending > leftover:
                outcome["adjustment_reason"] = "intraday_restock"
            elif ending < leftover and int(row.hours_remaining) <= 0:
                outcome["adjustment_reason"] = "window_close_write_off"
            if store.emit_outcome(outcome):
                n_out += 1

            # drift check at the legacy price (the price the outcome saw)
            eps = evt["epsilon_posterior_mean"]
            ratio = (1 - legacy_d) / (1 - evt["reference_discount"])
            drift["mu"].append(max(mu_path[t] * ratio ** eps,
                                   cfg["pricing"]["demand_floor"]))
            drift["r"].append(float(row.r_val))
            drift["q"].append(q)
            drift["sold"].append(sold)

            anchor = legacy_d                 # reality's price is the next anchor

    if n_dec == 0:
        raise RuntimeError("no decisions produced -- empty input or all states rejected")

    # censored basis: sales cannot exceed inventory, so the drift ratio
    # compares realised sales against E[min(D, q)] -- never raw mu
    predicted = expected_min_demand_inventory_vec(
        np.array(drift["mu"]), np.array(drift["r"]),
        np.array(drift["q"], dtype=float), cfg["pricing"]["negbin_max_k"])
    drift_ratio = (float(np.sum(drift["sold"]) / predicted.sum())
                   if predicted.sum() > 0 else None)

    # weeks-to-convergence input (section 19 / risk 1): how much evidence the
    # recommendations would have bought, and therefore how many BOUNDED
    # posterior steps it supports. The step cap and the daily human gate put a
    # calendar floor under this that no amount of evidence removes.
    eff_information = raw_information / deff(cfg)
    inc = cfg["learning"]["information_increment"]
    n_ep = len(groups)
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
        # the rate gates are estimates and a sample supports them; a zero
        # COUNT is only zero over what was sampled. Say so in the artifact
        # rather than letting "0 violations" read as a proof over the window.
        gate["sampling_caveat"] = (
            f"gate measured on {len(groups):,} of {len(population):,} episodes "
            f"(seed {seed}). The rates are sample estimates; the zero "
            "cost-floor violation count is zero OVER THE SAMPLE, not a proof "
            "over the window. Cost-floor safety is structural (the action set "
            "cannot express a below-cost price) and separately unit-tested -- "
            "this gate confirms it end-to-end, it does not establish it.")
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
                   "sampled": sampled,
                   "sample_seed": seed if sampled else None},
        "decision_count": n_dec,
        "outcome_count": n_out,
        "state_rejected_count": int(sum(rejected.values())),
        "rejected_reasons": rejected,
        "duplicate_counts": store.duplicate_counts,
        "quarantined_event_count": len(store.load_quarantine()),
        "shadow_gate": gate,
        "exploration_would_be": {
            "forced_rate": round(n_forced / n_dec, 4),
            "would_be_cost_total": round(would_be_cost, 1),
            "affordable_set_empty_rate": round(empty_affordable / n_dec, 4),
            "tau": tau,
            "note": "no price was applied; costs are the expected IL the "
                    "recommendations would have spent",
        },
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
    ap.add_argument("--date-start", default=None)
    ap.add_argument("--date-end", default=None)
    ap.add_argument("--events-dir", default=None)
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="episode sample size; 0 = all episodes. Default: "
                         "monitoring.shadow_gate.sample_episodes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)
    ds = d.date.astype(str)
    if args.date_start:
        d = d[ds.ge(args.date_start)]
        ds = d.date.astype(str)
    if args.date_end:
        d = d[ds.le(args.date_end)]

    report = run_shadow(d, cfg, events_root=args.events_dir,
                        seed=args.seed, max_episodes=args.max_episodes)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    g = report["shadow_gate"]
    w = report["window"]
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
    print(g["verdict"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
