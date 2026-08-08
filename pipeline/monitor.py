"""pipeline.monitor -- learning, business, and safety series (PRD section 15).

Reads the event store and the posterior and emits one JSON snapshot of the
three metric families, plus stop-condition evaluation (15.4). All IL% figures
are ratios of sums reported WITH their denominators, and absolute IL is
reported alongside every IL% (section 3.6).

Usage:
    python3 -m pipeline.monitor --out reports/monitor.json
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config, deff
from events.store import EventStore
from pricing.posterior import PosteriorStore
from pricing import explore


def _arm(sku_id, fc, allocation):
    """Stable-hash A/B assignment at the SKU x FC unit (section 18)."""
    h = hashlib.md5(f"{sku_id}|{fc}".encode()).digest()
    return "treatment" if int.from_bytes(h[:4], "big") / 2 ** 32 < allocation \
        else "control"


def business_metrics(decisions, outcomes, cfg):
    if not outcomes:
        return {"note": "no finalized outcomes yet"}
    dec = {d["decision_id"]: d for d in decisions}
    rows = []
    for o in outcomes:
        d = dec.get(o["decision_id"])
        if not d:
            continue
        rows.append({
            "episode_id": d["episode_id"], "category": d["category"],
            "fc": d["fc"], "sku_id": d["sku_id"],
            "hours_remaining": d["hours_remaining"],
            "original_price": d["original_price"], "cost": d["cost"],
            "units_sold": o["units_sold"],
            "ending_inventory": o["ending_inventory"],
            "discount_cost": (d["original_price"] - o["applied_price"])
                             * o["units_sold"],
            "arm": _arm(d["sku_id"], d["fc"], cfg["ab_test"]["allocation"]),
        })
    df = pd.DataFrame(rows)

    ep = df.sort_values("hours_remaining", ascending=False).groupby("episode_id").agg(
        category=("category", "first"), fc=("fc", "first"), arm=("arm", "first"),
        original_price=("original_price", "first"), cost=("cost", "first"),
        discount_cost=("discount_cost", "sum"), units_sold=("units_sold", "sum"),
        end_inv=("ending_inventory", "last"))
    ep["il"] = ep.discount_cost + ep.cost * ep.end_inv.clip(lower=0)
    ep["denom"] = ep.original_price * ep.units_sold

    def cut(g):
        den = float(g.denom.sum())
        return {"il_pct": round(float(g.il.sum() / den), 6) if den > 0 else None,
                "il_pct_denominator": den,
                "il_absolute": round(float(g.il.sum()), 1)}

    return {
        "il_pct_aggregate": cut(ep),
        "il_pct_by_category": {k: cut(g) for k, g in ep.groupby("category")},
        "il_pct_by_fc": {k: cut(g) for k, g in ep.groupby("fc")},
        "il_pct_by_arm": {k: cut(g) for k, g in ep.groupby("arm")},
        "sell_through": round(float(ep.units_sold.sum()
                                    / (ep.units_sold.sum() + ep.end_inv.sum())), 4)
            if (ep.units_sold.sum() + ep.end_inv.sum()) > 0 else None,
        "waste_units": int(ep.end_inv.clip(lower=0).sum()),
    }


def learning_metrics(decisions, posterior, cfg):
    cells = posterior.state["cells"]
    forced = [d for d in decisions if d["is_exploration"]]
    realised_cost = sum(d["exploration_cost"] for d in forced)
    empty_rate = (np.mean([d["affordable_set_size"] == 0 for d in decisions])
                  if decisions else None)
    return {
        "posterior_by_cell": {
            c: {"mean": r["mean"], "std": r["std"], "n_obs": r["n_obs"],
                "accumulated_information": round(r["accumulated_information"], 2),
                "version": r["version"], "updated_at": r["updated_at"]}
            for c, r in cells.items()},
        "forced_decision_count": len(forced),
        "affordable_set_empty_rate": round(float(empty_rate), 4)
            if empty_rate is not None else None,
        "mean_forced_log_price_ratio": round(float(np.mean(
            [abs(np.log((1 - d["applied_discount"])
                        / (1 - d["reference_discount"]))) for d in forced])), 4)
            if forced else None,
        "realised_exploration_cost": round(realised_cost, 1),
        "tau_current": decisions[-1]["tau_current"] if decisions else None,
        "deff_applied": round(deff(cfg), 3),
        # std only moves when an update commits, so "std flat for N days" is
        # exactly "no committed update in N days" (section 15.2 alert)
        "posterior_std_flat_alert": sorted(
            c for c, r in cells.items()
            if (pd.Timestamp.now("UTC")
                - pd.Timestamp(r["updated_at"])).days
            >= cfg["monitoring"]["alert_posterior_std_flat_days"]),
        "realised_vs_predicted_sold_ratio": None,  # filled by safety_metrics
    }


def safety_metrics(store, decisions, outcomes):
    matched = {o["decision_id"] for o in outcomes}
    mismatches = 0
    dec = {d["decision_id"]: d for d in decisions}
    expected_denom, realised_denom = 0.0, 0.0
    for o in outcomes:
        d = dec.get(o["decision_id"])
        if not d:
            continue
        if abs(o["applied_price"] - d["applied_price"]) > 1e-6:
            mismatches += 1
        expected_denom += d["expected_denominator"]
        realised_denom += d["original_price"] * o["units_sold"]
    n_out = max(len(outcomes), 1)
    return {
        "decision_count": len(decisions),
        "finalized_outcome_count": len(outcomes),
        "matched_decision_count": sum(1 for d in decisions
                                      if d["decision_id"] in matched),
        "unmatched_outcome_count": sum(1 for o in outcomes
                                       if o["decision_id"] not in dec),
        "duplicate_decision_count": store.duplicate_counts["decision"],
        "duplicate_outcome_count": store.duplicate_counts["outcome"],
        "applied_vs_recommended_price_mismatch": round(mismatches / n_out, 4),
        "zero_sales_rate": round(float(np.mean(
            [o["units_sold"] == 0 for o in outcomes])), 4) if outcomes else None,
        "stockout_rate": round(float(np.mean(
            [bool(o["is_stockout"]) for o in outcomes])), 4) if outcomes else None,
        "missing_stockout_field_rate": round(float(np.mean(
            ["is_stockout" not in o for o in outcomes])), 4) if outcomes else None,
        "quarantined_event_count": len(store.load_quarantine()),
        "solver_latency_p95_s": round(float(np.percentile(
            [d.get("solver_latency_s", 0.0) for d in decisions], 95)), 4)
            if decisions else None,
        "realised_vs_predicted_sold_ratio": round(
            realised_denom / expected_denom, 4) if expected_denom > 0 else None,
    }


def stop_conditions(safety, learning, business, cfg):
    """Section 15.4. Suspension stops forced exploration only; exploitation
    pricing continues. Owner-null thresholds cannot fire and are reported as
    blocked."""
    sc = cfg["monitoring"]["stop_conditions"]
    n = max(safety["finalized_outcome_count"], 1)
    dup_unmatched = (safety["duplicate_decision_count"]
                     + safety["duplicate_outcome_count"]
                     + safety["unmatched_outcome_count"]) / n
    fired = {}
    fired["duplicate_or_unmatched"] = dup_unmatched > sc["duplicate_or_unmatched_rate"]
    fired["price_mismatch"] = (safety["applied_vs_recommended_price_mismatch"]
                               > sc["price_mismatch_rate"])
    fired["missing_stockout_field"] = (safety["missing_stockout_field_rate"] or 0) > 0

    # realised exploration cost vs budget over the event window; the budget
    # uses realised markdown IL as the projection and the widest cell std
    il_abs = (business.get("il_pct_aggregate") or {}).get("il_absolute")
    cells = learning["posterior_by_cell"]
    if il_abs and cells:
        widest_std = max(rec["std"] for rec in cells.values())
        budget = explore.budget_today(il_abs, widest_std, cfg)
        fired["exploration_cost_vs_budget"] = (
            learning["realised_exploration_cost"]
            > sc["exploration_cost_vs_budget"] * budget) if budget > 0 else False
    else:
        fired["exploration_cost_vs_budget"] = False

    for key in ("scrap_deterioration_pct", "margin_deterioration_pct"):
        if sc[key] is None:
            fired[key] = f"BLOCKED -- {key} is null (SET BY OWNER)"
    return {"fired": fired,
            "suspend_exploration": any(v is True for v in fired.values())}


def main():
    ap = argparse.ArgumentParser(prog="pipeline.monitor")
    ap.add_argument("--out", default="reports/monitor.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = EventStore(cfg)
    posterior = PosteriorStore(cfg)
    decisions = store.load_decisions()
    outcomes = store.load_outcomes()

    learning = learning_metrics(decisions, posterior, cfg)
    safety = safety_metrics(store, decisions, outcomes)
    learning["realised_vs_predicted_sold_ratio"] = \
        safety["realised_vs_predicted_sold_ratio"]

    business = business_metrics(decisions, outcomes, cfg)
    report = {
        "business": business,
        "learning": learning,
        "safety": safety,
    }
    report["stop_conditions"] = stop_conditions(safety, learning, business, cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report["stop_conditions"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
