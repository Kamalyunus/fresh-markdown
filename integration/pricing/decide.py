"""A state is judged, solved and recorded. Validation REJECTS a state rather
than pricing it best-effort: a bad price or cost has no tier grid, a
horizon that disagrees with the forecast is the wrong window. Exploit
only: the optimal tier is the price; nothing is drawn."""
import math

import numpy as np
import pandas as pd

from pricing import dp as dp_mod
from pricing.config import reference_discount
from pricing.demand import expected_min_demand_inventory, mu_at
from pricing.keys import decision_id_of, hour_key


class StateRejected(ValueError):
    """Raised instead of returning an unsafe price."""


def finite_number(v):
    return (isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_)) and math.isfinite(v))


def is_count(v):
    """An integer, or a finite float that IS an integer (a parquet integer
    column with a null in it reads back as float); never a bool."""
    if isinstance(v, (bool, np.bool_)):
        return False
    return isinstance(v, (int, np.integer)) or (finite_number(v) and float(v) == int(v))


def economics_failures(original_price, cost):
    failures = []
    if not (finite_number(original_price) and original_price > 0):
        failures.append("original_price must be a finite positive number")
    if not (finite_number(cost) and cost >= 0):
        failures.append("cost must be a finite non-negative number")
    elif not failures and cost > original_price:
        failures.append("cost must not exceed original_price")
    return failures


def count_failures(rec, cfg):
    """The three counts a request and a state share, and the horizon bound."""
    failures = []
    q, hours, hour = rec.get("q"), rec.get("hours_remaining"), rec.get("hour_of_day")
    if not (is_count(q) and q >= 0):
        failures.append("q must be a non-negative integer")
    cap = int(cfg["data"]["max_window_hours"])
    if not (is_count(hours) and hours >= 1):
        failures.append("hours_remaining must be an integer >= 1")
    elif hours > cap:
        failures.append(f"hours_remaining must not exceed data.max_window_hours "
                        f"({cap}); got {int(hours)}")
    if not (is_count(hour) and 0 <= hour <= 23):
        failures.append("hour_of_day must be an integer in 0..23")
    return failures


def validate_state(s, tiers, anchor_discount, mu_ref_path, cfg):
    failures = count_failures(s, cfg)
    if anchor_discount is not None and not finite_number(anchor_discount):
        failures.append("current_discount must be a finite number")
    if not (finite_number(s["r"]) and s["r"] > 0):
        failures.append("r must be a finite positive number")
    if not tiers:
        failures.append("feasible set is empty")
    path = list(mu_ref_path) if isinstance(mu_ref_path, (list, tuple, np.ndarray)) else None
    if path is None or not all(finite_number(m) and m > 0 for m in path):
        failures.append("demand predictions must be finite and positive")
    if path is not None and len(path) != s["hours_remaining"]:
        failures.append(
            f"mu_ref_path has {len(path)} hours but hours_remaining is "
            f"{s['hours_remaining']}: the planning horizon and the recorded "
            "horizon must be the same window")
    if anchor_discount is not None and finite_number(anchor_discount) and tiers \
            and not any(d >= anchor_discount - dp_mod.TIER_EPS for d in tiers):
        failures.append("no feasible tier at or below the current anchor price")
    return failures


def _feature_or_none(features, k):
    if features is None:
        return None
    v = features[k]
    return None if v is None or not finite_number(v) else float(v)


def decide(s, cell, cfg, model_version, config_digest):
    """The decision event for state `s` priced at the cell's posterior mean,
    or StateRejected with every reason."""
    d_ref = reference_discount(cfg, s["category"])
    entry = s["current_discount"] is None
    anchor = s["current_discount"]
    failures = economics_failures(s["original_price"], s["cost"])
    tiers, d_max = ([], float("nan")) if failures else dp_mod.feasible_tiers(
        s["original_price"], s["cost"], cfg["pricing"]["tier_step"])
    failures += validate_state(s, tiers, anchor, s["mu_ref_path"], cfg)
    if failures:
        raise StateRejected("; ".join(failures))
    if not entry:
        anchor = float(anchor)
    eps = cell["mean"]
    try:
        result = dp_mod.solve(s["original_price"], s["cost"], int(s["q"]), s["mu_ref_path"],
                              d_ref, eps, s["r"], cfg, anchor_discount=anchor, entry=entry)
    except ValueError as e:
        raise StateRejected(str(e))
    star = result.optimal_index
    d_opt = result.tiers[star]
    mu_now = mu_at(s["mu_ref_path"][0], d_opt, d_ref, eps, cfg["pricing"]["demand_floor"])
    expected_sold_now = expected_min_demand_inventory(
        mu_now, s["r"], int(s["q"]), cfg["pricing"]["negbin_max_k"])
    return {
        "event": "decision",
        "decision_id": decision_id_of(hour_key(s["sku_id"], s["fc"], s["date"], s["hour_of_day"])),
        "episode_id": s["episode_id"],
        "is_entry": entry,
        "sku_id": s["sku_id"], "fc": s["fc"],
        "category": s["category"], "subcategory": s["subcategory"],
        "date": str(s["date"]),
        "hour_of_day": int(s["hour_of_day"]),
        "hours_remaining": int(s["hours_remaining"]),
        "q_remaining": int(s["q"]),
        "original_price": float(s["original_price"]),
        "cost": float(s["cost"]),
        "d_max": float(d_max),
        "feasible_tier_count": len(tiers),
        "action_set_size": len(result.q_by_tier),
        "optimal_price": float(s["original_price"] * (1 - d_opt)),
        "optimal_discount": float(d_opt),
        "expected_il": float(-result.q_by_tier[star]),
        "expected_denominator": float(s["original_price"] * expected_sold_now),
        "applied_price": float(s["original_price"] * (1 - d_opt)),
        "applied_discount": float(d_opt),
        "is_exploration": False,
        "exploration_cost": 0.0,
        "affordable_set_size": 0,
        "tau_current": None,
        "delta_min": 0.0,           # the floor under a draw: no draw here, no floor
        "epsilon_posterior_mean": float(cell["mean"]),
        "epsilon_posterior_std": float(cell["std"]),
        "reference_discount": float(d_ref),
        "reference_mu": float(s["mu_ref_path"][0]),
        "mu_ref_path": [float(m) for m in s["mu_ref_path"]],
        "sku_ref_sales_rate_30d": _feature_or_none(s.get("features"), 0),
        "prior_episode_ref_sales_rate": _feature_or_none(s.get("features"), 1),
        "anchor_discount": anchor,
        "dispersion_r": float(s["r"]),
        "baseline_model_version": model_version,
        "posterior_version": int(cell["version"]),
        "config_version": cfg["meta"]["config_version"],
        "config_digest": str(config_digest),
        "solver_latency_s": result.solver_latency_s,
        "nb_tail_mass_max": result.tail_mass_max,
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
    }
