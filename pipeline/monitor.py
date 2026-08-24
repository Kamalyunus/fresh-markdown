"""pipeline.monitor -- learning, business, and safety series (PRD section 15).

Reads the event store and the posterior and emits one JSON snapshot of the
three metric families, plus stop-condition evaluation (15.4). All IL% figures
are ratios of sums reported WITH their denominators, and absolute IL is
reported alongside every IL% (section 3.6).

Usage:
    python3 -m pipeline.monitor --out reports/monitor.json
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
# aliased: `stop_conditions` below takes a parameter called `guardrail`, and a
# module shadowed by an argument is a trap worth not setting
from common import guardrail as guard


def _still_running(ep):
    """Episodes with stock on hand and no closure sentinel -- still open.

    This is the ONLY reason common.episodes has a third state: offline every
    episode has finished, so scrap is just "leftover on the last row". Live,
    an in-flight episode's latest row is not a final row and its leftover is
    stock on the shelf, not scrap in the bin -- booking it would count it today
    and count something different tomorrow. Same function in both worlds, which
    is also what keeps this series on the population the noise floors were
    measured on.
    """
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
    df = pd.DataFrame(rows)

    ep = df.sort_values("hours_remaining", ascending=False).groupby("episode_id").agg(
        category=("category", "first"), fc=("fc", "first"), arm=("arm", "first"),
        original_price=("original_price", "first"), cost=("cost", "first"),
        discount_cost=("discount_cost", "sum"), units_sold=("units_sold", "sum"),
        starting_inventory=("starting_inventory", "last"),
        end_sold=("units_sold", "last"),
        ending_inventory=("ending_inventory", "last"))
    # leftover, not the reported ending_inventory (written off to zero at the
    # window close). Reading the field directly zeroes IL's scrap term.
    ep["end_inv"] = episodes.leftover_units(ep.starting_inventory, ep.end_sold)
    # ...but an episode with no closure sentinel has not ended, so its leftover
    # is not yet scrap. Excluding it keeps this metric on the same population
    # bootstrap.measure and derive_thresholds measure on; a population mismatch
    # between floor and trigger is the same defect class as a smoothing one.
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
        # visible, because a rising count means episodes are being reported
        # before they finish rather than that waste fell
        "episodes_excluded_still_running": len(running),
    }


def guardrail_series(decisions, outcomes, cfg):
    """Daily scrap and realised-margin rates, and the relative deterioration
    the section 15.4 stop conditions are written against.

    Metric definitions match bootstrap.derive_thresholds._daily_series exactly
    -- the noise floors the owner sets thresholds from are measured on those
    definitions, so a monitor computing anything else would be grading against
    the wrong yardstick.

    Comparison basis is the control arm when both arms carry a day's data
    (the A/B design), and the trailing guardrail_noise_window_days mean of the
    same series otherwise (before the A/B, and for any day one arm is empty).
    """
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

    # episode grain first: scrap is an end-of-episode quantity, so summing
    # hourly ending_inventory would count the same unsold unit every hour
    ep = df.sort_values("hours_remaining", ascending=False).groupby(
        "episode_id").agg(date=("date", "first"), arm=("arm", "first"),
                          start_inv=("start_inv", "first"),
                          starting_inventory=("start_inv", "last"),
                          units_sold=("sold", "last"),
                          ending_inventory=("ending_inventory", "last"),
                          revenue=("revenue", "sum"),
                          margin=("margin", "sum"))
    # scrap is max(0, inventory - sold) on the last hour, never the reported
    # ending_inventory: the source writes the remainder off to zero when the
    # window closes, so reading it directly reports zero scrap for every
    # episode and silently deletes the scrap term from IL.
    ep["end_inv"] = episodes.leftover_units(ep.starting_inventory, ep.units_sold)
    # same population rule as business_metrics and the threshold derivation:
    # an episode with no closure sentinel is still open -- leftover, not scrap
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
        """Deterioration against baseline, positive = worse.

        Both series are averaged over `smooth` days before comparing, and the
        comparison itself comes from `common.guardrail.deviation`, exactly as
        bootstrap.derive_thresholds measures the noise floor. A low-base series
        like daily scrap swings by more than its own level day to day;
        smoothing is what makes the floor smaller than the failure the
        condition exists to catch. Floor and trigger MUST use the same
        smoothing AND the same basis, or the threshold is set against a
        yardstick nothing computes -- which is why the comparison lives in one
        shared function rather than in two that resemble each other.
        """
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
            # say what the numbers ARE. A reader cannot tell 0.15 relative from
            # 0.15 percentage points by looking, and the two differ by an order
            # of magnitude for margin.
            "units": guard.units_of(dev_basis),
            "smoothing_days": smoothing[key],
            "by_day": {str(k): round(float(v), 4) for k, v in dev.items()},
            "latest": round(float(dev.iloc[-1]), 4) if len(dev) else None,
        }
    return out


def evaluate_guardrail(block, threshold, persistence_days):
    """A stop condition fires only after `persistence_days` CONSECUTIVE days
    over threshold.

    Persistence is how the design buys sensitivity without dipping below the
    measured noise floor: a single day above a 3-sigma floor is expected
    roughly once a year per guardrail, two in a row essentially never. It is
    load-bearing, not decoration -- the scrap threshold sits ~6% above its
    floor, so without this rule it would be a coin flip on the tail.
    """
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
    # Section 15 answers "is the business ok". Assurance answers the prior
    # question nothing else asks: are the frozen artifacts still a description
    # of the world we are pricing in. Same cadence, same report, separate
    # verdict -- it informs the operator gate, it does not suspend pricing.
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
