"""backtest -- offline replay through the production decision path.

PRD section 17. Three jobs, none of which is deciding whether to launch:

  1. Baseline fidelity -- the section 9.3 calibration gate (most important).
  2. tau_initial derivation -- the Q(p_star) - Q(p) distribution of section 12.3.
  3. Sanity-checking the DP and feasible-tier logic before any price is applied.

Replay output is NEVER evidence that the policy works; the A/B is that
evidence. Replay is deterministic (no stochastic demand sampling): transitions
use E[min(D, q)] under the truncated NB.

Fidelity and policy blocks are reported separately and never summed. IL% here
uses the section 3.2 denominator (original_price x units_sold, currency).

Usage:
    python3 -m backtest --input data/prepared.parquet --out reports/backtest.json
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from common.config import load_config
from bootstrap.train_baseline import BaselineModel
from bootstrap.fit_dispersion import lookup_r
from bootstrap.measure import m10_fidelity_decomposition
from pricing import dp as dp_mod
from pricing.demand import mu_at, expected_min_demand_inventory


def _attach_predictions(d, cfg, model, prior, r_lookup):
    """Predicted units at ACTUAL historical prices: mu_ref scaled by the prior
    elasticity, expectation censored at starting inventory."""
    d = d.copy()
    d["r"] = [lookup_r(r_lookup, s, c) for s, c in zip(d.subcategory, d.category)]
    d["eps"] = d.category.map(
        lambda c: prior["per_category"][str(c)]["mean"]).astype(float)
    mu_ref = model.predict_mu_ref(d)
    ratio = (1 - d.total_discount.to_numpy()) / (1 - d.d_ref.to_numpy())
    mu = np.clip(mu_ref * ratio ** d.eps.to_numpy(),
                 cfg["pricing"]["demand_floor"], None)
    d["mu_ref_hat"] = mu_ref
    d["mu_hat"] = mu

    max_k = cfg["pricing"]["negbin_max_k"]
    k = np.arange(max_k + 1)
    pred = np.empty(len(d))
    q = d.starting_inventory.to_numpy()
    r = d.r.to_numpy()
    for start in range(0, len(d), 100000):
        sl = slice(start, min(start + 100000, len(d)))
        p = (r[sl] / (r[sl] + mu[sl]))[:, None]
        pmf = nbinom.pmf(k[None, :], r[sl][:, None], p)
        pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0.0, None)
        pred[sl] = np.sum(pmf * np.minimum(k[None, :], q[sl][:, None]), axis=1)
    d["predicted_units"] = pred
    return d


def _fidelity_metrics(d, cfg):
    err = d.predicted_units - d.units_sold
    nz = d[d.units_sold > 0]
    return {
        "fidelity_episode_sold_ratio": round(
            float(d.units_sold.sum() / d.predicted_units.sum()), 4),
        "fidelity_hourly_mae": round(float(err.abs().mean()), 4),
        "fidelity_hourly_rmse": round(float(np.sqrt((err ** 2).mean())), 4),
        "fidelity_hourly_bias": round(float(err.mean()), 4),
        "fidelity_nz_mae": round(float((nz.predicted_units - nz.units_sold)
                                       .abs().mean()), 4),
        "fidelity_nz_bias": round(float((nz.predicted_units - nz.units_sold)
                                        .mean()), 4),
        "fidelity_zero_acc": round(float(((d.predicted_units < 0.5)
                                          == (d.units_sold == 0)).mean()), 4),
        "fidelity_pct_nonzero": round(float((d.units_sold > 0).mean()), 4),
        "by_category": {
            k: round(float(g.units_sold.sum() / g.predicted_units.sum()), 4)
            for k, g in d.groupby("category") if g.predicted_units.sum() > 0},
    }


def fidelity(d, cfg, model, prior, r_lookup):
    """Section 17.3 fidelity block: how well the model reproduces what actually
    happened, at actual historical prices.

    The calibration GATE is read on the calibration + test windows: that is
    the launch-adjacent regime, and it is the window the section 9.3 level
    factors are fit on -- correction and evaluation must share a regime. The
    all-history ratio is reported as a diagnostic; it is dominated by
    in-sample training rows and, when the demand level drifts between the
    training period and launch, it cannot be fixed by any static level factor.
    """
    from bootstrap.prepare_data import split_frames

    d = _attach_predictions(d, cfg, model, prior, r_lookup)
    splits = split_frames(d, cfg)
    gate_d = pd.concat([splits["calib"], splits["test"]])
    gate_window = "calib+test"
    if not len(gate_d) or gate_d.predicted_units.sum() <= 0:
        gate_d, gate_window = d, "all (calib/test windows empty)"

    block = _fidelity_metrics(gate_d, cfg)
    sold_ratio = block["fidelity_episode_sold_ratio"]
    block["gate_window"] = gate_window
    block["by_window"] = {
        name: {"rows": int(len(g)),
               "sold_ratio": round(float(g.units_sold.sum()
                                         / g.predicted_units.sum()), 4)
               if g.predicted_units.sum() > 0 else None}
        for name, g in [("train", splits["train"]), ("calib", splits["calib"]),
                        ("test", splits["test"]), ("all", d)]}
    block["measurement_10"] = m10_fidelity_decomposition(gate_d, cfg)
    band = cfg["baseline_model"]["calibration_gate_band"]
    block["calibration_gate_band"] = band
    block["calibration_gate"] = ("PASS" if band[0] <= sold_ratio <= band[1]
                                 else "FAIL -- blocking (PRD section 9.3)")
    return block, d


def _episode_frame(g):
    g = g.sort_values("hour_of_day")
    return {
        "original_price": float(g.original_price.iloc[0]),
        "cost": float(g.cost.iloc[0]),
        "d_ref": float(g.d_ref.iloc[0]),
        "q0": int(g.starting_inventory.iloc[0]),
        "hours": len(g),
        "date": str(g.date.iloc[0]),
        "actual_discounts": g.total_discount.to_numpy(),
        "actual_sold": g.units_sold.to_numpy(),
        "end_inv": int(g.ending_inventory.iloc[-1]),
        "mu_ref_path": g.mu_ref_hat.to_numpy(),
        "r": float(g.r.iloc[0]),
        "eps": float(g.eps.iloc[0]),
    }


def policy_replay(d_pred, cfg, max_episodes=2000, seed=0):
    """Section 17.3 policy block: what the DP would have done differently, plus
    the q-spread distribution for tau_initial. Deterministic expected-value
    transitions; replays the same pricing.dp code path production uses."""
    rng = np.random.default_rng(seed)
    pcfg = cfg["pricing"]
    max_k = pcfg["negbin_max_k"]

    eps_ids = d_pred.episode_id.unique()
    if len(eps_ids) > max_episodes:
        eps_ids = rng.choice(eps_ids, max_episodes, replace=False)
    sub = d_pred[d_pred.episode_id.isin(eps_ids)]

    rows, spreads = [], []
    for eid, g in sub.groupby("episode_id"):
        e = _episode_frame(g)
        if e["q0"] <= 0 or e["hours"] < 1:
            continue
        p0, cost = e["original_price"], e["cost"]
        tiers, _ = dp_mod.feasible_tiers(p0, cost, pcfg["tier_step"])
        if not tiers:
            continue

        # ---- actual path economics
        a_sold = e["actual_sold"]
        a_disc_cost = float(np.sum(p0 * e["actual_discounts"] * a_sold))
        a_scrap = cost * max(e["end_inv"], 0)
        a_denom = p0 * float(a_sold.sum())

        # ---- DP path, deterministic expected transitions
        q = float(e["q0"])
        anchor = None
        dp_disc_cost, dp_sold_total, dp_disc_weighted = 0.0, 0.0, 0.0
        for t in range(e["hours"]):
            q_int = int(round(q))
            if q_int <= 0:
                break
            try:
                res = dp_mod.solve(p0, cost, q_int, list(e["mu_ref_path"][t:]),
                                   e["d_ref"], e["eps"], e["r"], cfg,
                                   anchor_discount=anchor, entry=(t == 0))
            except ValueError:
                break
            star = res.optimal_index
            costs = [res.q_by_tier[star] - res.q_by_tier[j]
                     for j in res.q_by_tier if j != star]
            if t == 0 and costs:
                spreads.append({"date": e["date"], "costs": costs})
            d_t = res.tiers[star]
            anchor = d_t
            mu = mu_at(e["mu_ref_path"][t], d_t, e["d_ref"], e["eps"],
                       pcfg["demand_floor"])
            sold = min(expected_min_demand_inventory(mu, e["r"], q_int, max_k), q)
            dp_disc_cost += p0 * d_t * sold
            dp_disc_weighted += d_t * sold
            dp_sold_total += sold
            q -= sold
        dp_scrap = cost * max(q, 0.0)

        rows.append({
            "actual_il": a_disc_cost + a_scrap,
            "actual_discount_cost": a_disc_cost, "actual_scrap_cost": a_scrap,
            "actual_denom": a_denom,
            "actual_cleared": float(a_sold.sum()) / e["q0"],
            "actual_mean_discount": float(np.average(
                e["actual_discounts"],
                weights=a_sold if a_sold.sum() else None)),
            "dp_il": dp_disc_cost + dp_scrap,
            "dp_discount_cost": dp_disc_cost, "dp_scrap_cost": dp_scrap,
            "dp_denom": p0 * dp_sold_total,
            "dp_cleared": dp_sold_total / e["q0"],
            "dp_mean_discount": (dp_disc_weighted / dp_sold_total
                                 if dp_sold_total else 0.0),
            "date": e["date"],
        })

    ep = pd.DataFrame(rows)
    if not len(ep):
        raise RuntimeError("no episodes replayed")

    def money(col):
        return round(float(ep[col].sum()), 1)

    block = {
        "episodes_replayed": int(len(ep)),
        "actual_il": money("actual_il"),
        "actual_discount_cost": money("actual_discount_cost"),
        "actual_scrap_cost": money("actual_scrap_cost"),
        "actual_il_pct": round(float(ep.actual_il.sum() / ep.actual_denom.sum()), 4)
            if ep.actual_denom.sum() > 0 else None,
        "actual_clearance": round(float(ep.actual_cleared.mean()), 4),
        "actual_mean_discount": round(float(ep.actual_mean_discount.mean()), 4),
        "dp_il": money("dp_il"),
        "dp_discount_cost": money("dp_discount_cost"),
        "dp_scrap_cost": money("dp_scrap_cost"),
        "dp_il_pct": round(float(ep.dp_il.sum() / ep.dp_denom.sum()), 4)
            if ep.dp_denom.sum() > 0 else None,
        "dp_clearance": round(float(ep.dp_cleared.mean()), 4),
        "dp_mean_discount": round(float(ep.dp_mean_discount.mean()), 4),
        "pct_dp_deepened": round(float(
            (ep.dp_mean_discount > ep.actual_mean_discount).mean()), 4),
        "note": ("Replay output is never evidence that the policy works; the "
                 "A/B is that evidence (PRD 17.1). If pct_dp_deepened is 0 with "
                 "clearance falling, check fidelity_episode_sold_ratio first "
                 "(PRD 17.5)."),
    }
    all_costs = np.array([c for s in spreads for c in s["costs"]])
    block["q_spread_distribution"] = {
        f"p{p}": round(float(np.percentile(all_costs, p)), 2)
        for p in [10, 25, 50, 75, 90, 95, 99]} if len(all_costs) else {}
    return block, ep, spreads


def derive_tau_initial(spreads, ep, cfg):
    """Section 12.3: tau_initial is the currency quantile of the observed
    Q(p_star) - Q(p) distribution whose implied daily exploration spend matches
    budget_share_of_il of daily markdown IL. Never a rate."""
    if not spreads:
        return None
    daily_il = pd.DataFrame(ep).groupby("date")["actual_il"].sum()
    budget_per_day = float(cfg["exploration"]["budget_share_of_il"]
                           * daily_il.mean())

    by_day = {}
    for s in spreads:
        by_day.setdefault(s["date"], []).append(np.asarray(s["costs"]))

    def implied_daily_spend(tau):
        # uniform selection over the affordable set -> expected cost per
        # forced decision is the mean affordable cost
        total = 0.0
        for day, sets in by_day.items():
            for costs in sets:
                aff = costs[costs <= tau]
                if len(aff):
                    total += float(aff.mean())
        return total / max(len(by_day), 1)

    all_costs = np.sort(np.concatenate([np.asarray(s["costs"]) for s in spreads]))
    lo, hi = 0.0, float(all_costs[-1])
    for _ in range(60):
        mid = (lo + hi) / 2
        if implied_daily_spend(mid) < budget_per_day:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2
    quantile = float(np.searchsorted(all_costs, tau) / len(all_costs))
    return {"tau_initial": round(tau, 2),
            "unit": "currency (expected IL given up, per section 12.3)",
            "implied_daily_spend": round(implied_daily_spend(tau), 1),
            "daily_budget": round(budget_per_day, 1),
            "cost_distribution_quantile": round(quantile, 4)}
