"""inference -- validate state, decide, emit decision event.

Design section 5.10. Validation rejects the state rather than returning
an unsafe price. The decision path is: feasible tiers -> DP over Q(p) ->
exploit or explore -> price -> decision event.
"""

import math
import uuid

import numpy as np
import pandas as pd

from common.config import reference_discount
from pricing import dp as dp_mod
from pricing import explore
from pricing.demand import mu_at, expected_min_demand_inventory


class StateRejected(ValueError):
    """Raised instead of returning an unsafe price (section 11.4)."""


def validate_state(s, tiers, anchor_discount, mu_ref_path):
    failures = []
    if not (s["original_price"] > 0):
        failures.append("original_price must be positive")
    if not (0 <= s["cost"] <= s["original_price"]):
        failures.append("cost must be within [0, original_price]")
    if not (isinstance(s["q"], (int, np.integer)) and s["q"] >= 0):
        failures.append("q must be a non-negative integer")
    if not (isinstance(s["hours_remaining"], (int, np.integer))
            and s["hours_remaining"] >= 1):
        failures.append("hours_remaining must be an integer >= 1")
    if anchor_discount is not None and not math.isfinite(anchor_discount):
        failures.append("p_current must be finite")
    if not (s["r"] > 0):
        failures.append("r must be positive")
    if not tiers:
        failures.append("feasible set is empty")
    if not all(math.isfinite(m) and m > 0 for m in mu_ref_path):
        failures.append("demand predictions must be finite and positive")
    # the DP plans over len(mu_ref_path) and applies terminal scrap value at
    # the end of it, while the event records hours_remaining. If the caller
    # disagrees with itself the system silently optimises the wrong horizon --
    # exactly what a window truncated at a date boundary looks like. Reject.
    if len(mu_ref_path) != s["hours_remaining"]:
        failures.append(
            f"mu_ref_path has {len(mu_ref_path)} hours but hours_remaining is "
            f"{s['hours_remaining']}: the planning horizon and the recorded "
            "horizon must be the same window")
    if anchor_discount is not None and tiers \
            and not any(d >= anchor_discount - 1e-9 for d in tiers):
        failures.append("no feasible tier at or below the current anchor price")
    return failures


def decide(state, posterior_store, event_store, cfg, rng, tau_current,
           baseline_version, spread_sink=None):
    """Price one decision interval and emit the 16.1 decision event.
    `state` carries the episode context (mu_ref_path index 0 = now;
    current_discount None at entry). `spread_sink` receives the
    Q(p_star)-Q(p) costs out of band, before the draw, so the record is
    tau-independent."""
    s = state
    d_ref = reference_discount(cfg, s["category"])
    entry = s["current_discount"] is None
    anchor = None if entry else float(s["current_discount"])

    tiers, d_max = dp_mod.feasible_tiers(
        s["original_price"], s["cost"], cfg["pricing"]["tier_step"])
    failures = validate_state(s, tiers, anchor, s["mu_ref_path"])
    if failures:
        raise StateRejected("; ".join(failures))

    cell = posterior_store.get(s["category"])
    eps = cell["mean"]

    # the contract is REJECT, never crash: validate_state accepts q == 0
    # ("non-negative integer"), but the DP cannot price an empty shelf and
    # raises a bare ValueError -- which would escape the caller's
    # StateRejected handler and take the pricing loop down for a state that
    # is merely unpriceable (inventory hit zero between snapshot and call)
    try:
        result = dp_mod.solve(
            s["original_price"], s["cost"], int(s["q"]), s["mu_ref_path"],
            d_ref, eps, s["r"], cfg, anchor_discount=anchor, entry=entry)
    except ValueError as e:
        raise StateRejected(str(e))

    # explorability is a property of the actions allowed AT THIS DECISION, not
    # of the full grid: late in an episode the monotonicity anchor can leave
    # one action while the grid still has twenty, and at entry the coarse arm
    # set is the action set. Judging on len(tiers) overstated both.
    explorable = len(result.q_by_tier) >= cfg["exploration"]["min_feasible_tiers"]
    if spread_sink is not None and explorable:
        # reuse affordable_set's cost definition rather than restating it --
        # tau is calibrated against exactly the quantity it is compared to
        _, spread_costs = explore.affordable_set(result, 0.0)
        spread_sink([c for j, c in spread_costs.items()
                     if j != result.optimal_index])
    choice = explore.select(result, tau_current, rng, explorable=explorable)

    d_opt = result.tiers[result.optimal_index]
    d_applied = result.tiers[choice["chosen_index"]]
    mu_now = mu_at(s["mu_ref_path"][0], d_applied, d_ref, eps,
                   cfg["pricing"]["demand_floor"])
    expected_sold_now = expected_min_demand_inventory(
        mu_now, s["r"], int(s["q"]), cfg["pricing"]["negbin_max_k"])

    event = {
        "event": "decision",
        "decision_id": str(uuid.uuid4()),
        "episode_id": s["episode_id"],
        "is_entry": entry,
        "sku_id": s["sku_id"], "fc": s["fc"],
        "category": s["category"], "subcategory": s["subcategory"],
        # the pricing hour's calendar date: with hour_of_day it is the key
        # pipeline.ingest_outcomes matches feed rows to decisions on
        "date": str(s["date"]),
        "hour_of_day": int(s["hour_of_day"]),
        "hours_remaining": int(s["hours_remaining"]),
        "q_remaining": int(s["q"]),
        "original_price": float(s["original_price"]),
        "cost": float(s["cost"]),
        "d_max": float(d_max),
        "feasible_tier_count": len(tiers),
        # actions allowed at THIS decision -- the coarse entry arms, or the
        # tiers at/deeper than the anchor. Distinct from the grid size, and
        # the quantity explorability is judged on
        "action_set_size": len(result.q_by_tier),
        "optimal_price": float(s["original_price"] * (1 - d_opt)),
        "optimal_discount": float(d_opt),
        # absolute IL under the chosen policy -- the objective (section 3.1)
        "expected_il": float(-result.q_by_tier[choice["chosen_index"]]),
        # diagnostic only: enables predicted-vs-realised IL% tracking
        "expected_denominator": float(s["original_price"] * expected_sold_now),
        "applied_price": float(s["original_price"] * (1 - d_applied)),
        "applied_discount": float(d_applied),
        "is_exploration": choice["is_exploration"],
        "exploration_cost": choice["exploration_cost"],
        "affordable_set_size": choice["affordable_set_size"],
        "tau_current": tau_current,
        "epsilon_posterior_mean": float(cell["mean"]),
        "epsilon_posterior_std": float(cell["std"]),
        "reference_discount": float(d_ref),
        "reference_mu": float(s["mu_ref_path"][0]),
        # the FULL path and anchor: without them the event cannot be
        # re-solved, and pipeline.assurance exists to re-solve it
        "mu_ref_path": [float(m) for m in s["mu_ref_path"]],
        "anchor_discount": None if entry else float(anchor),
        "dispersion_r": float(s["r"]),
        "baseline_model_version": baseline_version,
        "posterior_version": int(cell["version"]),
        "config_version": cfg["meta"]["config_version"],
        "solver_latency_s": result.solver_latency_s,
        "nb_tail_mass_max": result.tail_mass_max,
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
    }
    event_store.emit_decision(event)
    return event
