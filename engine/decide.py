"""engine.decide -- validate state, decide, emit decision event.

Design section 5.10. Validation rejects the state rather than returning
an unsafe price. The decision path is: feasible tiers -> DP over Q(p) ->
exploit or explore -> price -> decision event.
"""

import math
import uuid

import numpy as np
import pandas as pd

from common.config import reference_discount
from engine import dp as dp_mod
from engine import explore
from engine.demand import mu_at, expected_min_demand_inventory


class StateRejected(ValueError):
    """Raised instead of returning an unsafe price (design 5.10)."""


def finite_number(v):
    """A real, finite number. bool is excluded (True is not a price of 1);
    numpy numerics count -- pandas producers must not quarantine in bulk.
    The ONE finiteness test: events.store reads it for applied_price."""
    return (isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_)) and math.isfinite(v))


def economics_failures(original_price, cost):
    """Why no tier grid can be built from this price and cost -- [] when one
    can. Judged BEFORE the grid is asked for (it would divide by a zero
    price or floor a NaN), and only here: validate_state takes everything
    else, so nothing is checked twice."""
    failures = []
    if not (finite_number(original_price) and original_price > 0):
        failures.append("original_price must be a finite positive number")
    if not (finite_number(cost) and cost >= 0):
        failures.append("cost must be a finite non-negative number")
    elif not failures and cost > original_price:
        failures.append("cost must not exceed original_price")
    return failures


def validate_state(s, tiers, anchor_discount, mu_ref_path):
    """Every check but the economics (economics_failures runs first, because
    `tiers` is built from its verdict)."""
    failures = []
    if not (isinstance(s["q"], (int, np.integer)) and s["q"] >= 0):
        failures.append("q must be a non-negative integer")
    if not (isinstance(s["hours_remaining"], (int, np.integer))
            and s["hours_remaining"] >= 1):
        failures.append("hours_remaining must be an integer >= 1")
    if anchor_discount is not None and not math.isfinite(anchor_discount):
        failures.append("p_current must be finite")
    # r parameterises every pmf the DP sums: None is a TypeError in the
    # solver and inf a NaN Q, neither of which is a rejection
    if not (finite_number(s["r"]) and s["r"] > 0):
        failures.append("r must be a finite positive number")
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
            and not any(d >= anchor_discount - dp_mod.TIER_EPS for d in tiers):
        failures.append("no feasible tier at or below the current anchor price")
    return failures


def decide(state, posterior_store, event_store, cfg, rng, tau_current,
           baseline_version, spread_sink=None):
    """Price one decision interval and emit the decision event (design 5.10;
    field contract in docs/event_contract.html). `state` carries the episode
    context (mu_ref_path index 0 = now; current_discount None at entry).
    `spread_sink` receives (costs, log moves, delta_min) over the admissible
    tiers out of band, before the draw, so the record is tau-independent.

    While the posterior store carries an exploration suspension (design
    5.12, set by daily.monitor) the decision is priced with NO budget:
    exploitation continues, nothing is drawn, and `tau_current` is recorded
    as None so the event says why it did not explore. The store is read as
    loaded: a long-lived caller reloads it per decision batch
    (PosteriorStore.reload), never per decision."""
    s = state
    d_ref = reference_discount(cfg, s["category"])
    entry = s["current_discount"] is None
    anchor = None if entry else float(s["current_discount"])

    # the contract is REJECT, never crash: a bad price or cost has no tier
    # grid, so judge the economics first and only then build one
    failures = economics_failures(s["original_price"], s["cost"])
    tiers, d_max = ([], float("nan")) if failures else dp_mod.feasible_tiers(
        s["original_price"], s["cost"], cfg["pricing"]["tier_step"])
    failures += validate_state(s, tiers, anchor, s["mu_ref_path"])
    if failures:
        raise StateRejected("; ".join(failures))

    cell = posterior_store.get(s["category"])
    eps = cell["mean"]

    # same contract for the solver: a state it cannot price (an empty shelf
    # between snapshot and call) is rejected, never a bare ValueError
    try:
        result = dp_mod.solve(
            s["original_price"], s["cost"], int(s["q"]), s["mu_ref_path"],
            d_ref, eps, s["r"], cfg, anchor_discount=anchor, entry=entry)
    except ValueError as e:
        raise StateRejected(str(e))

    # explorability is judged on the actions allowed AT THIS DECISION
    # (result.q_by_tier), never on the size of the full grid
    explorable = len(result.q_by_tier) >= cfg["exploration"]["min_feasible_tiers"]
    # the smallest informative move for THIS cell: tiers closer to p* than
    # this are neither drawn nor priced into tau (explore.admissible)
    dmin = explore.delta_min(cfg, eps, s["category"])
    # one cost table for the ledger and the draw -- the same set, priced once
    costs = explore.admissible_costs(result, dmin)
    if spread_sink is not None and explorable:
        spread_sink((*explore.spread_table(result, costs=costs), dmin))
    suspended = posterior_store.exploration_suspended()
    tau_in_force = None if suspended else tau_current
    choice = explore.select(result, tau_in_force, rng, explorable=explorable,
                            costs=costs)

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
        # daily.ingest_outcomes matches feed rows to decisions on
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
        # absolute IL under the chosen policy -- the objective (design 2.2)
        "expected_il": float(-result.q_by_tier[choice["chosen_index"]]),
        # diagnostic only: enables predicted-vs-realised IL% tracking
        "expected_denominator": float(s["original_price"] * expected_sold_now),
        "applied_price": float(s["original_price"] * (1 - d_applied)),
        "applied_discount": float(d_applied),
        "is_exploration": choice["is_exploration"],
        "exploration_cost": choice["exploration_cost"],
        "affordable_set_size": choice["affordable_set_size"],
        # None while exploration is suspended (design 5.12): the budget was
        # not in force for this decision, whatever tau the store holds
        "tau_current": tau_in_force,
        "delta_min": float(dmin),
        "epsilon_posterior_mean": float(cell["mean"]),
        "epsilon_posterior_std": float(cell["std"]),
        "reference_discount": float(d_ref),
        "reference_mu": float(s["mu_ref_path"][0]),
        # the FULL path and anchor: without them the event cannot be
        # re-solved, and daily.assurance exists to re-solve it
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
