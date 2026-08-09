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
from bootstrap.prepare_data import split_frames
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


def level_mix_decomposition(d, cfg):
    """Is the weekly anchor-level movement demand drift, or SKU mix?

    The aggregate anchor ratio is a volume-weighted average of per-SKU
    ratios, and which SKUs enter the markdown cohort each week is driven by
    overstock and expiry -- not stable. So the aggregate can move purely
    because the composition moved. Each SKU x FC ratio is estimated ONCE over
    all history (stable), then every week is recomputed holding those ratios
    fixed and letting only that week's composition vary:

      raw_t          = sum(sold_t) / sum(pred_t)
      mix_expected_t = sum(pred_t * ratio_sku) / sum(pred_t)

    mix_expected tracking raw  -> composition explains the movement
    raw climbing above it      -> genuine within-SKU level drift

    Also reports whether per-SKU calibration is even feasible: a factor fit
    on a handful of anchor rows is noise, and applying it forward projects
    that noise onto prices.
    """
    tier_step = cfg["pricing"]["tier_step"]
    min_unit_rows = cfg["baseline_model"]["mix_decomposition_min_unit_rows"]

    a = d[(d.total_discount - d.d_ref).abs() <= tier_step / 2].copy()
    if not len(a) or a.predicted_units.sum() <= 0:
        return "NOT RUN -- no anchor rows"
    a["week"] = pd.to_datetime(a.date).dt.to_period("W").astype(str)
    a["unit"] = a.sku_id.astype(str) + "|" + a.fc.astype(str)
    overall = float(a.units_sold.sum() / a.predicted_units.sum())

    u = a.groupby("unit").agg(sold=("units_sold", "sum"),
                              pred=("predicted_units", "sum"),
                              rows=("units_sold", "size"),
                              category=("category", "first"))
    c = a.groupby("category").agg(sold=("units_sold", "sum"),
                                  pred=("predicted_units", "sum"))
    cat_ratio = c.sold / c.pred.replace(0, np.nan)
    enough = (u.rows >= min_unit_rows) & (u.pred > 0)
    u["ratio"] = np.where(enough, u.sold / u.pred.replace(0, np.nan),
                          u.category.map(cat_ratio).fillna(overall))
    ratio_map = u.ratio

    series = {}
    for w, g in a.groupby("week"):
        pred = float(g.predicted_units.sum())
        if pred <= 0:
            continue
        mix_expected = float(
            (g.predicted_units * g.unit.map(ratio_map).fillna(overall)).sum() / pred)
        series[w] = {"raw": round(float(g.units_sold.sum() / pred), 4),
                     "mix_expected": round(mix_expected, 4)}

    weeks = sorted(series)
    summary = {}
    if len(weeks) >= 2:
        raw_chg = series[weeks[-1]]["raw"] - series[weeks[0]]["raw"]
        mix_chg = (series[weeks[-1]]["mix_expected"]
                   - series[weeks[0]]["mix_expected"])
        summary = {
            "raw_change_first_to_last": round(raw_chg, 4),
            "mix_change_first_to_last": round(mix_chg, 4),
            "mix_explained_share": round(mix_chg / raw_chg, 4)
                if abs(raw_chg) > 1e-9 else None,
        }

    trusted = u[enough]
    return {
        "by_week": series,
        "summary": summary,
        "sku_level_feasibility": {
            "sku_fc_units": int(len(u)),
            "share_units_ge_10_anchor_rows": round(float((u.rows >= 10).mean()), 4),
            "share_units_ge_30_anchor_rows": round(float((u.rows >= 30).mean()), 4),
            "share_units_ge_100_anchor_rows": round(float((u.rows >= 100).mean()), 4),
            "anchor_rows_per_unit_p50": int(np.percentile(u.rows, 50)),
            "trusted_unit_ratio_p10_p90": [
                round(float(np.percentile(trusted.ratio, 10)), 3),
                round(float(np.percentile(trusted.ratio, 90)), 3)]
                if len(trusted) > 10 else None,
        },
        "note": ("mix_explained_share near 1 means the level movement is "
                 "composition, not demand -- the remedy is a mix-robust gate "
                 "metric, not a finer calibration grain. Per-SKU factors are "
                 "only viable where units carry enough anchor rows to fit "
                 "them on signal rather than noise."),
    }


def calibration_window_sweep(d, cfg):
    """Rolling-origin test of the calibration MECHANISM, as production runs it.

    For each candidate trailing window W: fit per-category level factors on
    weeks [t-W, t-1], apply them to week t, and measure the anchor ratio that
    results -- repeated over every eligible week. This answers "how long
    should the fit window be" with data rather than assertion, and it is the
    honest test when the level moves: a longer window averages more history
    and is therefore MORE stale, not more accurate. `uncalibrated` is the
    same series with no factor applied, as the comparison baseline.
    """
    band = cfg["baseline_model"]["calibration_gate_band"]
    tier_step = cfg["pricing"]["tier_step"]
    windows = cfg["baseline_model"]["calibration_window_sweep_weeks"]

    a = d[(d.total_discount - d.d_ref).abs() <= tier_step / 2].copy()
    if not len(a):
        return "NOT RUN -- no anchor rows"
    a["week"] = pd.to_datetime(a.date).dt.to_period("W")
    cw = (a.groupby(["category", "week"], observed=True)
          .agg(sold=("units_sold", "sum"), pred=("predicted_units", "sum"))
          .reset_index())
    weeks = sorted(cw.week.unique())

    def summarise(ratios):
        arr = np.array(ratios)
        return {
            "eval_weeks": int(len(arr)),
            "median_anchor_ratio": round(float(np.median(arr)), 4),
            "share_weeks_in_band": round(
                float(((arr >= band[0]) & (arr <= band[1])).mean()), 4),
            "mean_abs_log_error": round(float(np.mean(np.abs(np.log(arr)))), 4),
        }

    def ratios_for(window):
        out = []
        for i, t in enumerate(weeks):
            if i < max(window, 1):
                continue
            cur = cw[cw.week == t]
            if window == 0:
                factor = None
            else:
                fit = (cw[cw.week.isin(weeks[i - window:i])]
                       .groupby("category", observed=True)[["sold", "pred"]].sum())
                factor = fit.sold / fit.pred.replace(0, np.nan)
            f = (np.ones(len(cur)) if factor is None
                 else cur.category.map(factor).fillna(1.0).to_numpy())
            adj = float((cur.pred.to_numpy() * f).sum())
            if adj > 0:
                out.append(float(cur.sold.sum() / adj))
        return out

    result = {}
    base = ratios_for(0)
    if base:
        result["uncalibrated"] = summarise(base)
    for w in windows:
        r = ratios_for(w)
        if r:
            result[f"trailing_{w}w"] = summarise(r)

    candidates = {k: v for k, v in result.items() if k != "uncalibrated"}
    if candidates:
        best = min(candidates,
                   key=lambda k: (-candidates[k]["share_weeks_in_band"],
                                  candidates[k]["mean_abs_log_error"]))
        result["recommended_fit_window"] = best
        result["note"] = (
            "Rolling-origin: factors fit on the trailing window, applied to "
            "the NEXT week. If shorter windows score better, the demand level "
            "is trending and recency beats sample size; if longer windows win, "
            "the variation is noise and averaging helps.")
    return result


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
    d = _attach_predictions(d, cfg, model, prior, r_lookup)
    splits = split_frames(d, cfg)
    gate_window = cfg["baseline_model"]["calibration_gate_window"]
    gate_d = (splits["test"] if gate_window == "test"
              else pd.concat([splits["calib"], splits["test"]]))
    if not len(gate_d) or gate_d.predicted_units.sum() <= 0:
        gate_d, gate_window = d, "all (configured gate window empty)"

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
    # weekly sold ratios over ALL history: if these swing wider than the gate
    # band, the gate is measuring week-scale demand volatility, not model
    # quality -- a gate-window/band decision, not a retraining problem
    week = pd.to_datetime(d.date).dt.to_period("W").astype(str)
    block["by_week"] = {
        str(w): round(float(g.units_sold.sum() / g.predicted_units.sum()), 4)
        for w, g in d.groupby(week) if g.predicted_units.sum() > 0}
    block["measurement_10"] = m10_fidelity_decomposition(gate_d, cfg)

    # trend triage: an anchor-level climb driven by NEW-assortment SKUs
    # (no rate-feature history -> NaN -> model predicts low while they sell)
    # is an assortment effect, not a macro demand trend
    if "sku_ref_sales_rate_30d" in gate_d.columns:
        tier_step = cfg["pricing"]["tier_step"]
        anchor_rows = gate_d[
            (gate_d.total_discount - gate_d.d_ref).abs() <= tier_step / 2]
        has_hist = anchor_rows.sku_ref_sales_rate_30d.notna()

        def _ratio(g):
            pred = g.predicted_units.sum()
            return round(float(g.units_sold.sum() / pred), 4) if pred > 0 else None

        block["anchor_ratio_by_rate_history"] = {
            "with_history": _ratio(anchor_rows[has_hist]),
            "no_history": _ratio(anchor_rows[~has_hist]),
            "share_rows_no_history": round(float((~has_hist).mean()), 4)
                if len(anchor_rows) else None,
            "note": "no_history far above with_history means new-assortment "
                    "SKUs are driving the level gap, not a macro trend",
        }

    # is the weekly level movement demand drift, or SKU composition?
    block["level_mix_decomposition"] = level_mix_decomposition(d, cfg)
    # how long should the level factor's fit window be? measured, not assumed
    block["calibration_window_sweep"] = calibration_window_sweep(d, cfg)

    # gate metric: the pooled ratio judges the model at actual prices (and so
    # embeds the elasticity prior); level_at_anchor judges only the frozen
    # artifact's own responsibility -- the level at reference price
    metric = cfg["baseline_model"]["calibration_gate_metric"]
    m10 = block["measurement_10"]
    anchor_val = (m10.get("level_bias_at_anchor")
                  if isinstance(m10, dict) else None)
    if metric == "level_at_anchor" and anchor_val:
        gate_value, gate_metric = anchor_val, "level_bias_at_anchor"
    else:
        gate_value, gate_metric = sold_ratio, "pooled_ratio"
        if metric == "level_at_anchor":
            gate_metric += " (no anchor rows -- fell back from level_at_anchor)"
    band = cfg["baseline_model"]["calibration_gate_band"]
    block["calibration_gate_band"] = band
    block["calibration_gate_metric"] = gate_metric
    block["calibration_gate_value"] = gate_value
    block["calibration_gate"] = ("PASS" if band[0] <= gate_value <= band[1]
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

        # ---- actual path economics (observed world, legacy prices)
        a_sold = e["actual_sold"]
        a_disc_cost = float(np.sum(p0 * e["actual_discounts"] * a_sold))
        a_scrap = cost * max(e["end_inv"], 0)
        a_denom = p0 * float(a_sold.sum())

        # ---- LEGACY path under the MODEL's demand: same generator as the DP
        # simulation below, so the two policies are compared apples-to-apples
        # and model bias hits both arms identically
        q = float(e["q0"])
        lg_disc_cost = lg_sold_total = lg_disc_weighted = 0.0
        for t in range(e["hours"]):
            q_int = int(round(q))
            if q_int <= 0:
                break
            d_t = float(e["actual_discounts"][t])
            mu = mu_at(e["mu_ref_path"][t], d_t, e["d_ref"], e["eps"],
                       pcfg["demand_floor"])
            sold = min(expected_min_demand_inventory(mu, e["r"], q_int, max_k), q)
            lg_disc_cost += p0 * d_t * sold
            lg_disc_weighted += d_t * sold
            lg_sold_total += sold
            q -= sold
        lg_scrap = cost * max(q, 0.0)

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
            "legacy_model_il": lg_disc_cost + lg_scrap,
            "legacy_model_discount_cost": lg_disc_cost,
            "legacy_model_scrap_cost": lg_scrap,
            "legacy_model_denom": p0 * lg_sold_total,
            "legacy_model_cleared": lg_sold_total / e["q0"],
            "legacy_model_mean_discount": (lg_disc_weighted / lg_sold_total
                                           if lg_sold_total else 0.0),
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

    lg_il, dp_il = float(ep.legacy_model_il.sum()), float(ep.dp_il.sum())
    block = {
        "episodes_replayed": int(len(ep)),
        "actual_il": money("actual_il"),
        "actual_discount_cost": money("actual_discount_cost"),
        "actual_scrap_cost": money("actual_scrap_cost"),
        "actual_il_pct": round(float(ep.actual_il.sum() / ep.actual_denom.sum()), 4)
            if ep.actual_denom.sum() > 0 else None,
        "actual_clearance": round(float(ep.actual_cleared.mean()), 4),
        "actual_mean_discount": round(float(ep.actual_mean_discount.mean()), 4),
        "legacy_model_il": money("legacy_model_il"),
        "legacy_model_discount_cost": money("legacy_model_discount_cost"),
        "legacy_model_scrap_cost": money("legacy_model_scrap_cost"),
        "legacy_model_il_pct": round(
            lg_il / float(ep.legacy_model_denom.sum()), 4)
            if ep.legacy_model_denom.sum() > 0 else None,
        "legacy_model_clearance": round(float(ep.legacy_model_cleared.mean()), 4),
        "legacy_model_mean_discount": round(
            float(ep.legacy_model_mean_discount.mean()), 4),
        "dp_il": money("dp_il"),
        "dp_discount_cost": money("dp_discount_cost"),
        "dp_scrap_cost": money("dp_scrap_cost"),
        "dp_il_pct": round(dp_il / float(ep.dp_denom.sum()), 4)
            if ep.dp_denom.sum() > 0 else None,
        "dp_clearance": round(float(ep.dp_cleared.mean()), 4),
        "dp_mean_discount": round(float(ep.dp_mean_discount.mean()), 4),
        "pct_dp_deepened": round(float(
            (ep.dp_mean_discount > ep.actual_mean_discount).mean()), 4),
        # apples-to-apples: both policies simulated under the SAME demand
        # model, so model bias hits both arms identically and cancels in
        # the comparison. actual_* vs model figures are fidelity, not policy.
        "policy_gap_like_for_like": {
            "dp_minus_legacy_il": round(dp_il - lg_il, 1),
            "dp_il_reduction_pct_of_legacy": round(
                (lg_il - dp_il) / lg_il, 4) if lg_il > 0 else None,
            "clearance_delta": round(float(
                ep.dp_cleared.mean() - ep.legacy_model_cleared.mean()), 4),
            "basis": "legacy prices vs DP prices, demand generated by the "
                     "same frozen model + prior for both arms",
        },
        "note": ("Policy comparison is legacy-under-model vs DP-under-model "
                 "(same demand generator both arms). actual_* figures are the "
                 "observed world and belong to fidelity, not policy. Replay "
                 "output is never evidence the policy works; the A/B is that "
                 "evidence (PRD 17.1)."),
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
