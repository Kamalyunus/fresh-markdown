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
from common.config import load_config, deff_from_episodes
from events.store import EventStore
from events.pairs import match_pairs, decision_day, price_matches
from pricing.posterior import PosteriorStore
from pipeline import assurance as assurance_mod
from pipeline import update as update_mod
from pricing import explore
from common import episodes
from common import metrics
from common.provenance import config_fingerprint
# aliased: stop_conditions() takes a parameter named `guardrail`
from common import guardrail as guard


def event_frame(decisions, outcomes, cfg):
    """Matched (decision, outcome) pairs as HOURLY rows in the prepared-frame
    vocabulary, so metrics.episode_economics is the one episode-grain
    definition on live events too -- floor and trigger measure one thing."""
    alloc = cfg["ab_test"]["allocation"]
    return pd.DataFrame([{
        "episode_id": d["episode_id"], "date": decision_day(d),
        "hour_of_day": d["hour_of_day"],
        "category": d.get("category"), "fc": d["fc"], "sku_id": d["sku_id"],
        "arm": arm(d["sku_id"], d["fc"], alloc),
        "original_price": d["original_price"], "offered_price": o["applied_price"],
        "cost": d["cost"], "starting_inventory": o["starting_inventory"],
        "units_sold": o["units_sold"], "ending_inventory": o["ending_inventory"],
    } for d, o in match_pairs(decisions, outcomes)])


def business_metrics(decisions, outcomes, cfg):
    if not outcomes:
        return {"note": "no finalized outcomes yet"}
    df = event_frame(decisions, outcomes, cfg)
    if df.empty:
        return {"note": "no outcome matches a decision -- nothing to measure"}
    ep, excluded = metrics.settled(metrics.episode_economics(df))

    def cut(g):
        den = float(g.denom.sum())
        return {"il_pct": round(float(g.il.sum() / den), 6) if den > 0 else None,
                "il_pct_denominator": den,
                "il_absolute": round(float(g.il.sum()), 1)}

    units = float(ep.units_sold.sum() + ep.scrap.sum())
    return {
        "il_pct_aggregate": cut(ep),
        "il_pct_by_category": {k: cut(g) for k, g in ep.groupby("category")},
        "il_pct_by_fc": {k: cut(g) for k, g in ep.groupby("fc")},
        "il_pct_by_arm": {k: cut(g) for k, g in ep.groupby("arm")},
        "sell_through": round(float(ep.units_sold.sum() / units), 4)
            if units > 0 else None,
        "waste_units": int(ep.scrap.sum()),
        # realised IL by CLOSE DAY: the trailing base the tau controller
        # prices a day's budget from (explore.trailing_daily_il). Settled
        # episodes only -- an open one contributes nothing until it closes,
        # which is what makes the base knowable at the start of each day.
        "il_by_close_day": {str(k): round(float(v), 2) for k, v
                            in ep.groupby("close_day").il.sum().items()},
        # visible: a rising count means early reporting, not falling waste
        "scrap_basis": excluded,
    }


def guardrail_series(decisions, outcomes, cfg):
    """Daily scrap and realised-margin rates plus the 15.4 deterioration
    series, on metrics.daily_rates -- the series the noise floors are
    measured on. Basis: the trailing window mean, unless `ab_test.active`
    says an A/B is running AND both arms carry data. Before the A/B every
    priced unit is system-priced and merely hash-LABELLED into arms, so an
    arm comparison there is treatment-vs-treatment: a catalogue-wide
    deterioration cancels exactly and the guardrail cannot fire (design 12)."""
    df = event_frame(decisions, outcomes, cfg)
    if df.empty:
        return {"note": "no finalized outcomes yet"}
    ep, _ = metrics.settled(metrics.episode_economics(df))
    overall = metrics.daily_rates(ep)
    by_arm = {a: metrics.daily_rates(g) for a, g in ep.groupby("arm")}
    window = cfg["monitoring"]["guardrail_noise_window_days"]

    smoothing = cfg["monitoring"]["stop_conditions"]["deterioration_smoothing_days"]
    ab_active = bool(cfg["ab_test"].get("active"))

    def deterioration(metric, worse_when_higher, smooth, dev_basis):
        """Deterioration against baseline, positive = worse. Both series are
        averaged over `smooth` days first, then compared via the shared
        common.guardrail.deviation -- floor and trigger MUST use the same
        smoothing AND basis, so the comparison lives in one function."""
        treat = by_arm.get("treatment")
        ctrl = by_arm.get("control")
        # The control-arm basis needs a control arm that is NOT system-priced.
        # `arm()` labels every priced SKU x FC by hash, so before the A/B both
        # labels are present and both are treated: the comparison is
        # treatment-vs-treatment and a catalogue-wide scrap doubling is a
        # deviation of exactly ZERO. The guardrail was structurally inert for
        # the whole pilot. `ab_test.active` is the only thing that can say
        # which regime we are in -- it cannot be inferred from the labels.
        note = None
        if ab_active and treat is not None and ctrl is not None:
            t = guard.smooth(treat[metric], smooth)
            c = guard.smooth(ctrl[metric], smooth)
            common = t.index.intersection(c.index)
            t, c, basis = t.loc[common], c.loc[common], "control_arm"
        else:
            t = guard.smooth(overall[metric], smooth)
            c = t.rolling(window, min_periods=window).mean().shift(smooth)
            basis = f"trailing_{window}d_mean"
            if ab_active and ctrl is None:
                # control units are legacy-priced, so they emit no decisions
                # and never reach the event store -- say so rather than let a
                # silent fallback read as an arm comparison
                note = ("ab_test.active is true but the event store holds no "
                        "control-arm rows: control units are not system-priced, "
                        "so their outcomes never enter it. Comparing against "
                        "the trailing mean instead -- a genuine arm comparison "
                        "needs control outcomes from the feed.")
        dev = guard.deviation(t, c, worse_when_higher, dev_basis)
        return dev.replace([np.inf, -np.inf], np.nan).dropna(), basis, note

    out = {"days_observed": int(len(overall)),
           "daily_scrap_rate": {str(k): round(float(v), 6)
                                for k, v in overall.scrap_rate.items()},
           "daily_margin_rate": {str(k): round(float(v), 6)
                                 for k, v in overall.margin_rate.dropna().items()}}
    for metric, worse_high, key in (("scrap_rate", True, "scrap"),
                                    ("margin_rate", False, "margin")):
        dev_basis = guard.basis_for(key)
        dev, basis, note = deterioration(metric, worse_high, smoothing[key],
                                         dev_basis)
        out[f"{key}_deterioration"] = {
            "basis": basis,
            **({"basis_note": note} if note else {}),
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


def learning_metrics(decisions, posterior, cfg, outcomes=()):
    cells = posterior.state["cells"]
    forced = [d for d in decisions if d["is_exploration"]]
    realised_cost = sum(d["exploration_cost"] for d in forced)
    empty_rate = (np.mean([d["affordable_set_size"] == 0 for d in decisions])
                  if decisions else None)
    return {
        # routing, so the budget's widest-std is taken over cells a category
        # actually reaches (an unrouted GLOBAL never narrows)
        "cell_of": dict(posterior.state["cell_of"]),
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
        # per-day, from update.daily_exploration_spend -- the ONE definition
        # the tau controller and the stop condition both price a day with
        "exploration_cost_by_day": {
            k: round(v, 1) for k, v in
            update_mod.daily_exploration_spend(decisions, outcomes).items()},
        "latest_priced_day": update_mod.latest_priced_day(decisions, outcomes),
        "tau_current": decisions[-1]["tau_current"] if decisions else None,
        "deff_applied": round(deff_from_episodes(
            cfg["dispersion"]["rho"],
            [d["episode_id"] for d in forced]), 3),
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
    dec = {d["decision_id"]: d for d in decisions}
    pairs = match_pairs(decisions, outcomes)
    compared = len(pairs)
    mismatches = sum(1 for d, o in pairs if not price_matches(d, o))
    expected_denom, realised_denom = 0.0, 0.0
    for d, o in pairs:
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
        # a rate over the outcomes that HAVE a decision to compare against;
        # unmatched outcomes are counted separately and diluted this. The
        # counts are what the stop condition compares on -- the rounded
        # rate is for reading, and comparing it disagreed with update's
        # unrounded gate at the boundary
        "price_mismatch_count": mismatches,
        "compared_pair_count": compared,
        "applied_vs_recommended_price_mismatch": round(
            mismatches / max(compared, 1), 4),
        "zero_sales_rate": round(float(np.mean(
            [o["units_sold"] == 0 for o in outcomes])), 4) if outcomes else None,
        "stockout_rate": round(float(np.mean(
            [bool(o["is_stockout"]) for o in outcomes])), 4) if outcomes else None,
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
    fired["price_mismatch"] = (
        safety["price_mismatch_count"] / max(safety["compared_pair_count"], 1)
        > sc["price_mismatch_rate"])

    # realised exploration cost vs budget over the event window; the budget
    # uses TRAILING realised IL as the projection and the widest cell std --
    # the same two per-day numbers pipeline.update's tau controller moves on
    # (design 5.8). Comparing all-time spend against all-time IL made both
    # blind together: the ratio tends to 1.0 as history grows, so a day at
    # 10x budget could not fire it.
    il_by_day = business.get("il_by_close_day") or {}
    cells = learning["posterior_by_cell"]
    # the SAME day the controller prices (update.latest_priced_day) -- the
    # last day with a closed episode is a different day whenever the latest
    # day's episodes are still open
    day = learning.get("latest_priced_day")
    if il_by_day and cells and day:
        widest_std = PosteriorStore.widest_active_std(
            cells, learning.get("cell_of") or {})
        budget = explore.budget_today(
            explore.trailing_daily_il(il_by_day, day, cfg), widest_std, cfg)
        spend = float((learning.get("exploration_cost_by_day") or {}).get(day, 0.0))
        fired["exploration_cost_vs_budget"] = (
            spend > sc["exploration_cost_vs_budget"] * budget) if budget > 0 else False
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

    learning = learning_metrics(decisions, posterior, cfg, outcomes)
    safety = safety_metrics(store, decisions, outcomes)
    learning["realised_vs_predicted_sold_ratio"] = \
        safety["realised_vs_predicted_sold_ratio"]

    business = business_metrics(decisions, outcomes, cfg)
    guardrail = guardrail_series(decisions, outcomes, cfg)
    # assurance asks whether the frozen artifacts still describe the world;
    # it informs the operator gate, it does not suspend pricing
    assurance = assurance_mod.run(decisions, outcomes, cfg)
    report = {
        "config": config_fingerprint(cfg, "production"),
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
