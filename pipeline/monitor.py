"""pipeline.monitor -- learning, business, and safety series (design 5.12).

Reads the event store and posterior; emits one JSON snapshot of the three
metric families plus stop-condition evaluation (15.4). IL% is a ratio of sums
reported WITH its denominator, absolute IL alongside every IL% (3.6).
Run: python3 -m pipeline.monitor --out reports/monitor.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from common.ab import arm
from common.config import load_config, deff
from events.store import EventStore
from pricing.posterior import PosteriorStore
from pipeline import assurance as assurance_mod
from pricing import explore
from common import episodes
# aliased: stop_conditions() takes a parameter named `guardrail`
from common import guardrail as guard


def _still_running(ep):
    """Episodes with stock on hand and no closure sentinel -- still open.
    Their leftover is stock on the shelf, not scrap; excluding them keeps the
    series on the population the noise floors were measured on."""
    kind = episodes.classify_last(ep)
    return set(kind.index[kind == episodes.NOT_CLOSED])


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
            "timestamp": d.get("timestamp"),
            "hours_remaining": d["hours_remaining"],
            "original_price": d["original_price"], "cost": d["cost"],
            "units_sold": o["units_sold"],
            "starting_inventory": o["starting_inventory"],
            "ending_inventory": o["ending_inventory"],
            "discount_cost": (d["original_price"] - o["applied_price"])
                             * o["units_sold"],
            "arm": arm(d["sku_id"], d["fc"], cfg["ab_test"]["allocation"]),
        })
    if not rows:
        return {"note": "no outcome matches a decision -- nothing to measure"}
    df = pd.DataFrame(rows)

    df = df.sort_values("hours_remaining", ascending=False)
    # SCRAP = leftover + shrink, the one definition (episodes.scrap_units, and
    # what common.metrics.il_pct and the noise floors are measured on). IL,
    # waste_units, sell-through and the by-arm IL% here read leftover only,
    # so the A/B's primary metric was a different IL from the one it was
    # powered on and production's exploration budget base was a third.
    df["_shrink"] = episodes.shrink_by_hour(
        df.starting_inventory, df.units_sold, df.ending_inventory,
        ~df.duplicated("episode_id", keep="last"))
    ep = df.groupby("episode_id").agg(
        category=("category", "first"), fc=("fc", "first"), arm=("arm", "first"),
        original_price=("original_price", "first"), cost=("cost", "first"),
        discount_cost=("discount_cost", "sum"), units_sold=("units_sold", "sum"),
        starting_inventory=("starting_inventory", "last"),
        end_sold=("units_sold", "last"), shrink=("_shrink", "sum"),
        ending_inventory=("ending_inventory", "last"))
    # leftover, never the reported ending_inventory (written off to zero at
    # window close -- reading it directly zeroes IL's scrap term)
    ep["end_inv"] = (episodes.leftover_units(ep.starting_inventory, ep.end_sold)
                     + ep.shrink)
    # still-open episodes excluded: same population as bootstrap.measure and
    # derive_thresholds, so floor and trigger measure the same thing
    running = _still_running(ep.assign(units_sold=ep.end_sold))
    ep = ep[~ep.index.isin(running)]
    ep["il"] = ep.discount_cost + ep.cost * ep.end_inv
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
        # visible: a rising count means early reporting, not falling waste
        "episodes_excluded_still_running": len(running),
    }


def guardrail_series(decisions, outcomes, cfg):
    """Daily scrap and realised-margin rates plus the 15.4 deterioration
    series. Definitions match bootstrap.derive_thresholds._daily_series
    exactly -- the noise floors are measured on them. Basis: control arm when
    both arms carry a day's data, else the trailing-window mean."""
    dec = {d["decision_id"]: d for d in decisions}
    rows = []
    for o in outcomes:
        d = dec.get(o["decision_id"])
        if not d or "timestamp" not in d:
            continue
        rows.append({
            "date": pd.Timestamp(d["timestamp"]).date(),
            "timestamp": d["timestamp"],
            "episode_id": d["episode_id"],
            "arm": arm(d["sku_id"], d["fc"], cfg["ab_test"]["allocation"]),
            "hours_remaining": d["hours_remaining"],
            "cost": d["cost"],
            "start_inv": o["starting_inventory"],
            "ending_inventory": o["ending_inventory"],
            "sold": o["units_sold"],
            "revenue": o["applied_price"] * o["units_sold"],
            "margin": (o["applied_price"] - d["cost"]) * o["units_sold"],
        })
    if not rows:
        return {"note": "no finalized outcomes with timestamps yet"}
    df = pd.DataFrame(rows)

    # episode grain first: scrap is an end-of-episode quantity; summing hourly
    # ending_inventory would count the same unsold unit every hour
    df = df.sort_values("hours_remaining", ascending=False)
    # SHRINK counts into scrap (scrap = leftover + vanished), same basis as
    # episodes.scrap_units, which the noise floors are measured on -- a
    # leftover-only trigger runs looser than its floor by the shrink rate.
    # A write-off row (last hour, zeroed ending, stock remaining) is the
    # leftover, not shrink; restock rows clip to zero.
    df["_shrink"] = episodes.shrink_by_hour(
        df.start_inv, df.sold, df.ending_inventory,
        ~df.duplicated("episode_id", keep="last"))
    ep = df.groupby(
        "episode_id").agg(date=("date", "first"), arm=("arm", "first"),
                          start_inv=("start_inv", "first"),
                          starting_inventory=("start_inv", "last"),
                          units_sold=("sold", "last"),
                          ending_inventory=("ending_inventory", "last"),
                          shrink=("_shrink", "sum"),
                          revenue=("revenue", "sum"),
                          margin=("margin", "sum"))
    # leftover = max(0, inventory - sold) on the last hour, never the reported
    # ending_inventory (source writes it off to zero at window close)
    ep["end_inv"] = episodes.leftover_units(
        ep.starting_inventory, ep.units_sold) + ep.shrink
    # same population rule as business_metrics and the threshold derivation:
    # no closure sentinel = still open -- leftover, not scrap
    ep = ep[~ep.index.isin(_still_running(ep))]

    def daily(frame):
        day = frame.groupby("date").agg(
            start_inv=("start_inv", "sum"), end_inv=("end_inv", "sum"),
            revenue=("revenue", "sum"), margin=("margin", "sum")).sort_index()
        return pd.DataFrame({
            "scrap_rate": day.end_inv.clip(lower=0) / day.start_inv,
            "margin_rate": day.margin / day.revenue.replace(0, np.nan),
        })

    overall = daily(ep)
    by_arm = {arm: daily(g) for arm, g in ep.groupby("arm")}
    window = cfg["monitoring"]["guardrail_noise_window_days"]

    smoothing = cfg["monitoring"]["stop_conditions"]["deterioration_smoothing_days"]

    def deterioration(metric, worse_when_higher, smooth, dev_basis):
        """Deterioration against baseline, positive = worse. Both series are
        averaged over `smooth` days first, then compared via the shared
        common.guardrail.deviation -- floor and trigger MUST use the same
        smoothing AND basis, so the comparison lives in one function."""
        treat = by_arm.get("treatment")
        ctrl = by_arm.get("control")
        if treat is not None and ctrl is not None:
            t = guard.smooth(treat[metric], smooth)
            c = guard.smooth(ctrl[metric], smooth)
            common = t.index.intersection(c.index)
            t, c, basis = t.loc[common], c.loc[common], "control_arm"
        else:
            t = guard.smooth(overall[metric], smooth)
            c = t.rolling(window, min_periods=window).mean().shift(smooth)
            basis = f"trailing_{window}d_mean"
        dev = guard.deviation(t, c, worse_when_higher, dev_basis)
        return dev.replace([np.inf, -np.inf], np.nan).dropna(), basis

    out = {"days_observed": int(len(overall)),
           "daily_scrap_rate": {str(k): round(float(v), 6)
                                for k, v in overall.scrap_rate.items()},
           "daily_margin_rate": {str(k): round(float(v), 6)
                                 for k, v in overall.margin_rate.dropna().items()}}
    for metric, worse_high, key in (("scrap_rate", True, "scrap"),
                                    ("margin_rate", False, "margin")):
        dev_basis = guard.basis_for(cfg, key)
        dev, basis = deterioration(metric, worse_high, smoothing[key], dev_basis)
        out[f"{key}_deterioration"] = {
            "basis": basis,
            "deterioration_basis": dev_basis,
            # a reader cannot tell 0.15 relative from 0.15 pp by looking
            "units": guard.units_of(dev_basis),
            "smoothing_days": smoothing[key],
            "by_day": {str(k): round(float(v), 4) for k, v in dev.items()},
            "latest": round(float(dev.iloc[-1]), 4) if len(dev) else None,
        }
    return out


def evaluate_guardrail(block, threshold, persistence_days):
    """Fires only after `persistence_days` CONSECUTIVE days over threshold.
    Persistence is load-bearing, not decoration: it buys sensitivity for
    thresholds sitting just above the measured noise floor (design 15.4)."""
    base = {"fired": False, "threshold": threshold,
            "persistence_days": persistence_days,
            "basis": block.get("basis"), "latest": block.get("latest")}
    if threshold is None:
        return {**base, "status": "BLOCKED -- threshold is null (SET BY OWNER)"}
    by_day = block.get("by_day") or {}
    if not by_day:
        # smoothing consumes the first `smoothing_days - 1` days, so a short
        # window legitimately has nothing to compare yet
        return {**base, "consecutive_days_over": 0,
                "status": "no comparable days yet"}
    days = sorted(by_day)
    streak = 0
    for day in reversed(days):
        if by_day[day] > threshold:
            streak += 1
        else:
            break
    return {
        "fired": streak >= persistence_days,
        "consecutive_days_over": streak,
        "persistence_days": persistence_days,
        "threshold": threshold,
        "latest": block.get("latest"),
        "basis": block.get("basis"),
        "status": (f"FIRED -- over {threshold} for {streak} consecutive days"
                   if streak >= persistence_days else
                   f"{streak}/{persistence_days} consecutive days over threshold"),
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


def stop_conditions(safety, learning, business, guardrail, cfg):
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

    # the two owner thresholds, evaluated against the daily deterioration
    # series with the persistence rule the design commits to
    guardrails = {}
    for key, block_key in (("scrap_deterioration_pct", "scrap_deterioration"),
                           ("margin_deterioration_pct", "margin_deterioration")):
        block = (guardrail or {}).get(block_key) or {}
        result = evaluate_guardrail(block, sc[key], sc["persistence_days"])
        guardrails[key] = result
        fired[key] = result["fired"] if sc[key] is not None else result["status"]

    return {"fired": fired,
            "guardrails": guardrails,
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
    guardrail = guardrail_series(decisions, outcomes, cfg)
    # assurance asks whether the frozen artifacts still describe the world;
    # it informs the operator gate, it does not suspend pricing
    assurance = assurance_mod.run(decisions, outcomes, cfg)
    report = {
        "business": business,
        "guardrails": guardrail,
        "learning": learning,
        "safety": safety,
        "assurance": assurance,
    }
    report["stop_conditions"] = stop_conditions(
        safety, learning, business, guardrail, cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report["stop_conditions"], indent=2))
    print("assurance: " + ("PASS" if assurance["verdict"] == "PASS"
                           else "FAIL -> " + ", ".join(assurance["failing"])))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
