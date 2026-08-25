"""backtest -- offline replay through the production decision path (design.md 5.14).

Jobs: baseline fidelity (9.3 gate), tau_initial derivation (12.3), DP sanity.
Replay output is never evidence the policy works; the A/B is that evidence.
Deterministic: transitions use E[min(D, q)] under the truncated NB. Fidelity
and policy blocks are never summed; IL% uses the section 3.2 denominator.
"""


import numpy as np
import pandas as pd

from bootstrap.prepare_data import split_frames
from bootstrap.fit_dispersion import lookup_r
from bootstrap.measure import m10_fidelity_decomposition
from common import episodes
from common.parallel import map_episodes
from pricing import dp as dp_mod
from pricing import explore
from pricing.demand import (mu_at, expected_min_demand_inventory,
                            expected_min_demand_inventory_vec)


def _attach_predictions(d, cfg, model, prior, r_lookup):
    """Predicted units at ACTUAL historical prices: mu_ref scaled by the prior
    elasticity, censored at starting inventory. The frame is first extended to
    the full window so an early sell-out cannot shorten the DP's horizon."""
    carry = [c for c in d.columns if c not in
             ("episode_id", "date", "hour_of_day", "hours_remaining",
              "starting_inventory", "ending_inventory", "units_sold")]
    d = episodes.extend_to_window(d, carry, cfg["data"]["max_window_hours"])
    d["r"] = [lookup_r(r_lookup, s, c) for s, c in zip(d.subcategory, d.category)]
    d["eps"] = d.category.map(
        lambda c: prior["per_category"][str(c)]["mean"]).astype(float)
    mu_ref = model.predict_mu_ref(d)
    ratio = (1 - d.total_discount.to_numpy()) / (1 - d.d_ref.to_numpy())
    mu = np.clip(mu_ref * ratio ** d.eps.to_numpy(),
                 cfg["pricing"]["demand_floor"], None)
    d["mu_ref_hat"] = mu_ref
    d["mu_hat"] = mu

    d["predicted_units"] = expected_min_demand_inventory_vec(
        mu, d.r.to_numpy(), d.starting_inventory.to_numpy(),
        cfg["pricing"]["negbin_max_k"])
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
    """Splits weekly anchor-level movement into demand drift vs SKU mix: each
    SKU x FC ratio is fit ONCE over all history, then every week is recomputed
    holding ratios fixed so only composition varies (mix_expected tracks raw ->
    composition; raw climbs above it -> drift). Also reports per-SKU feasibility."""
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
    """Rolling-origin sweep of the calibration fit window: per-category factors
    fit on weeks [t-W, t-1], applied to week t, over every eligible week. When
    the level trends, longer windows are MORE stale, not more accurate.
    `uncalibrated` is the no-factor comparison baseline."""
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
    """Section 17.3 fidelity block: how well the model reproduces observed
    sales at actual historical prices. The gate reads the calib+test windows
    -- the launch-adjacent regime the 9.3 level factors are fit on; the
    all-history ratio is diagnostic only (dominated by in-sample rows)."""
    # predict over the FULL window (the DP plans over it); every ratio below
    # sees observed rows only -- synthetic rows would read as under-prediction
    d_full = _attach_predictions(d, cfg, model, prior, r_lookup)
    d = d_full[d_full.is_observed]
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
    # weekly ratios swinging wider than the gate band = week-scale demand
    # volatility, a gate-window/band decision rather than a retraining problem
    week = pd.to_datetime(d.date).dt.to_period("W").astype(str)
    block["by_week"] = {
        str(w): round(float(g.units_sold.sum() / g.predicted_units.sum()), 4)
        for w, g in d.groupby(week) if g.predicted_units.sum() > 0}
    block["measurement_10"] = m10_fidelity_decomposition(gate_d, cfg)

    # trend triage: an anchor-level climb driven by new-assortment SKUs (no
    # rate history -> low prediction) is assortment, not a macro demand trend
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

    # sizes the censoring gap: the gate basis is censored E[min(D,q)], and a
    # comparison against raw mu (E[D]) would flatter the model by this much
    raw_mu_total = float(gate_d.mu_hat.sum())
    censored_total = float(gate_d.predicted_units.sum())
    sold_total = float(gate_d.units_sold.sum())
    block["prediction_basis"] = {
        "sold_over_censored_prediction": round(sold_total / censored_total, 4)
            if censored_total > 0 else None,
        "sold_over_raw_mu": round(sold_total / raw_mu_total, 4)
            if raw_mu_total > 0 else None,
        "censoring_shrinkage": round(censored_total / raw_mu_total, 4)
            if raw_mu_total > 0 else None,
        "censored_hour_rate": round(
            float((gate_d.units_sold >= gate_d.starting_inventory).mean()), 4),
        "note": ("censored is the gate basis and the one every fit must use; "
                 "the raw-mu figure is shown only to size the gap between "
                 "them -- a factor fit on raw mu is wrong by this much"),
    }

    # is the weekly level movement demand drift, or SKU composition?
    block["level_mix_decomposition"] = level_mix_decomposition(d, cfg)
    # how long should the level factor's fit window be? measured, not assumed
    block["calibration_window_sweep"] = calibration_window_sweep(d, cfg)

    # gate metric: pooled ratio judges the model at actual prices (embeds the
    # prior); level_at_anchor judges only the artifact's level at reference price
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
    # DIAGNOSTIC, not a gate (owner, 2026-08-25): out-of-band = drift reading
    # to investigate; fidelity.by_week distinguishes wobble from trend
    block["calibration_gate"] = ("PASS" if band[0] <= gate_value <= band[1]
                                 else "OUT OF BAND -- level diagnostic; "
                                      "investigate drift (design 9.2)")
    return block, d_full


def _episode_frame(g, unfinished=frozenset()):
    g = g.sort_values(["date", "hour_of_day"])
    obs = g.is_observed.to_numpy() if "is_observed" in g else np.ones(len(g), bool)
    # uncovered hours hold the last observed discount (legacy ramps to a cap
    # and holds); both arms must run the same horizon to stay like-for-like
    disc = pd.Series(g.total_discount.to_numpy()).where(
        pd.Series(obs)).ffill().to_numpy()
    obs_rows = g[obs] if obs.any() else g
    # adjustment computed on OBSERVED rows only: the write-off exemption keys
    # on the LAST row, and the extended frame's last row is a synthetic tail
    adj_obs = (episodes.hour_adjustment(obs_rows).to_numpy()
               if obs.any() else np.zeros(0))
    adjustment = np.zeros(len(g))
    adjustment[obs] = adj_obs
    return {
        "n_observed": int(obs.sum()),
        # the listing ending IS the disposal; only an unclosed episode lacks
        # an outcome (keying on hours_remaining <= 0 mischarged scrap)
        "outcome_known": bool(g.episode_id.iloc[0] not in unfinished),
        "original_price": float(g.original_price.iloc[0]),
        "cost": float(g.cost.iloc[0]),
        "d_ref": float(g.d_ref.iloc[0]),
        "q0": int(g.starting_inventory.iloc[0]),
        "hours": len(g),
        "date": str(g.date.iloc[0]),
        "actual_discounts": disc,
        # synthetic rows carry units_sold = 0 -- extension leaves economics alone
        "actual_sold": g.units_sold.to_numpy(),
        # written-off zero on the last row -- take the true leftover
        "end_inv": max(int(obs_rows.starting_inventory.iloc[-1])
                       - int(obs_rows.units_sold.iloc[-1]), 0),
        # EXOGENOUS per-hour inventory change (+arrival, -shrink); both arms
        # apply it and learn of it next hour, as production does. Extension rows: 0.
        "adjustment": adjustment,
        # what the episode actually had to sell, and what it actually lost
        "supply": int(g.starting_inventory.iloc[0])
                  + int(np.clip(adj_obs, 0, None).sum()),
        "shrink": int(-np.clip(adj_obs, None, 0).sum()),
        "mu_ref_path": g.mu_ref_hat.to_numpy(),
        "r": float(g.r.iloc[0]),
        "eps": float(g.eps.iloc[0]),
        # LABELS for traced replays only -- no economics reads them; tolerant
        # of missing columns so a diagnostic label can never break the replay
        "episode_id": str(g.episode_id.iloc[0]) if "episode_id" in g else "",
        "sku_id": (int(g.sku_id.iloc[0]) if "sku_id" in g else None),
        "fc": str(g.fc.iloc[0]) if "fc" in g else "",
        "category": str(g.category.iloc[0]) if "category" in g else "",
        "hour_dates": ([str(x) for x in g.date] if "date" in g
                       else [""] * len(g)),
        "hours_of_day": ([int(x) for x in g.hour_of_day]
                         if "hour_of_day" in g else list(range(len(g)))),
        "is_observed": obs.astype(bool),
        "starting_inventory": g.starting_inventory.to_numpy(),
    }


def _replay_one(e, cfg):
    """One episode's replay: actual path, legacy-under-model arm, DP arm.
    Pure -- reads only `e` and `cfg` -- so it runs in-process or in a worker.
    Returns (row, [(date, q_spread_costs), ...]) or None."""
    pcfg = cfg["pricing"]
    max_k = pcfg["negbin_max_k"]
    p0, cost = e["original_price"], e["cost"]
    tiers, _ = dp_mod.feasible_tiers(p0, cost, pcfg["tier_step"])
    if not tiers:
        return None
    spreads = []

    # PER-HOUR TRACE, off by default, read only by tools.export_backtest; it
    # lives here because design 5.14 forbids a parallel implementation
    tr = {} if e.get("trace") else None

    def rec(t, **kw):
        if tr is not None:
            tr.setdefault(t, {}).update(kw)

    # ---- actual path economics (observed world, legacy prices)
    a_sold = e["actual_sold"]
    a_disc_cost = float(np.sum(p0 * e["actual_discounts"] * a_sold))
    # scrap = leftover at close + shrink (units paid for, no revenue); an
    # unfinished episode charges none, or the baseline would be overstated
    a_scrap = (cost * (max(e["end_inv"], 0) + e["shrink"])
               if e["outcome_known"] else 0.0)
    a_denom = p0 * float(a_sold.sum())

    # ---- LEGACY path under the MODEL's demand: same generator as the DP arm,
    # so model bias hits both identically (like-for-like)
    q = float(e["q0"])
    adj = e["adjustment"]
    lg_disc_cost = lg_sold_total = lg_disc_weighted = 0.0
    for t in range(e["hours"]):
        q_int = int(round(q))
        # an empty shelf is not the end if stock is still to arrive
        if q_int > 0:
            d_t = float(e["actual_discounts"][t])
            mu = mu_at(e["mu_ref_path"][t], d_t, e["d_ref"], e["eps"],
                       pcfg["demand_floor"])
            sold = min(expected_min_demand_inventory(mu, e["r"], q_int, max_k), q)
            lg_disc_cost += p0 * d_t * sold
            lg_disc_weighted += d_t * sold
            lg_sold_total += sold
            q -= sold
            rec(t, legacy_q=q_int, legacy_discount=d_t,
                legacy_price=p0 * (1 - d_t), legacy_mu=float(mu),
                legacy_units=float(sold))
        else:
            rec(t, legacy_q=0, legacy_discount=None, legacy_price=None,
                legacy_mu=None, legacy_units=0.0)
        q = max(q + adj[t], 0.0)
        if q <= 0 and not adj[t + 1:].any():
            break
    # captured before the DP loop reuses `q`
    lg_q_final = q
    lg_scrap = cost * (max(q, 0.0) + e["shrink"])

    # ---- DP path, deterministic expected transitions
    q = float(e["q0"])
    anchor = None
    dp_disc_cost, dp_sold_total, dp_disc_weighted = 0.0, 0.0, 0.0
    for t in range(e["hours"]):
        q_int = int(round(q))
        # empty shelf ends the episode only if nothing more is coming; the DP
        # never anticipates a delivery -- it learns next hour, as production does
        if q_int <= 0:
            rec(t, dp_q=0, dp_discount=None, dp_price=None, dp_mu=None,
                dp_units=0.0)
            q = max(q + adj[t], 0.0)
            if q <= 0 and not adj[t + 1:].any():
                break
            continue
        try:
            res = dp_mod.solve(p0, cost, q_int, list(e["mu_ref_path"][t:]),
                               e["d_ref"], e["eps"], e["r"], cfg,
                               anchor_discount=anchor, entry=(t == 0))
        except ValueError:
            break
        star = res.optimal_index
        # spreads collected at EVERY decision hour, not just entry: entry-only
        # tau underfunded ~8x, invisible here and surfaced only in shadow
        spreads.append((e["date"], [res.q_by_tier[star] - res.q_by_tier[j]
                                    for j in res.q_by_tier if j != star]))
        d_t = res.tiers[star]
        anchor = d_t
        mu = mu_at(e["mu_ref_path"][t], d_t, e["d_ref"], e["eps"],
                   pcfg["demand_floor"])
        sold = min(expected_min_demand_inventory(mu, e["r"], q_int, max_k), q)
        dp_disc_cost += p0 * d_t * sold
        dp_disc_weighted += d_t * sold
        dp_sold_total += sold
        rec(t, dp_q=q_int, dp_discount=d_t, dp_price=p0 * (1 - d_t),
            dp_mu=float(mu), dp_units=float(sold),
            dp_is_entry=bool(t == 0),
            dp_feasible_tiers=len(res.tiers))
        q = max(q - sold + adj[t], 0.0)
    dp_scrap = cost * (max(q, 0.0) + e["shrink"])

    # units emitted as their own fields, never derived downstream from
    # scrap_cost/cost; scrap = leftover + shrink, and exogenous shrink is
    # identical across arms so only the leftover differs
    a_left = max(e["end_inv"], 0) if e["outcome_known"] else 0
    lg_left, dp_left = max(lg_q_final, 0.0), max(q, 0.0)
    row = {
        "outcome_known": e["outcome_known"],
        "actual_sold_units": float(a_sold.sum()),
        "actual_leftover_units": float(a_left),
        "actual_scrap_units": float(a_left + (e["shrink"]
                                              if e["outcome_known"] else 0)),
        "legacy_model_sold_units": float(lg_sold_total),
        "legacy_model_leftover_units": float(lg_left),
        "legacy_model_scrap_units": float(lg_left + e["shrink"]),
        "dp_sold_units": float(dp_sold_total),
        "dp_leftover_units": float(dp_left),
        "dp_scrap_units": float(dp_left + e["shrink"]),
        # per-arm identity: supply = sold + leftover + shrink -- holds by
        # construction, so a nonzero residual is a real defect, not rounding
        "actual_supply_residual": float(
            e["supply"] - a_sold.sum() - a_left
            - (e["shrink"] if e["outcome_known"] else 0)),
        "legacy_model_supply_residual": float(
            e["supply"] - lg_sold_total - lg_left - e["shrink"]),
        "dp_supply_residual": float(
            e["supply"] - dp_sold_total - dp_left - e["shrink"]),
        "actual_il": a_disc_cost + a_scrap,
        "actual_discount_cost": a_disc_cost, "actual_scrap_cost": a_scrap,
        "actual_denom": a_denom,
        "actual_cleared": float(a_sold.sum()) / max(e["supply"], 1),
        "actual_mean_discount": float(np.average(
            e["actual_discounts"],
            weights=a_sold if a_sold.sum() else None)),
        "legacy_model_il": lg_disc_cost + lg_scrap,
        "legacy_model_discount_cost": lg_disc_cost,
        "legacy_model_scrap_cost": lg_scrap,
        "legacy_model_denom": p0 * lg_sold_total,
        "legacy_model_cleared": lg_sold_total / max(e["supply"], 1),
        "legacy_model_mean_discount": (lg_disc_weighted / lg_sold_total
                                       if lg_sold_total else 0.0),
        "dp_il": dp_disc_cost + dp_scrap,
        "dp_discount_cost": dp_disc_cost, "dp_scrap_cost": dp_scrap,
        "dp_denom": p0 * dp_sold_total,
        "dp_cleared": dp_sold_total / max(e["supply"], 1),
        "dp_mean_discount": (dp_disc_weighted / dp_sold_total
                             if dp_sold_total else 0.0),
        "date": e["date"],
        "eps": e["eps"],
        "deepening_threshold": dp_mod.deepening_threshold_epsilon(
            e["original_price"], e["cost"], e["d_ref"]),
        "episode_id": e["episode_id"], "sku_id": e["sku_id"],
        "fc": e["fc"], "category": e["category"],
        "q0": e["q0"], "supply": e["supply"], "shrink": e["shrink"],
        "end_inv": e["end_inv"], "hours": e["hours"],
        "n_observed": e["n_observed"], "r": e["r"],
        "original_price": p0, "cost": cost, "d_ref": e["d_ref"],
    }
    if tr is not None:
        # one row per hour, all three arms side by side; legacy and DP share
        # the same demand model, so the comparison stays like-for-like
        hours = []
        for t in range(e["hours"]):
            rt = tr.get(t, {})
            a_d = float(e["actual_discounts"][t])
            hours.append({
                "episode_id": e["episode_id"], "sku_id": e["sku_id"],
                "fc": e["fc"], "category": e["category"],
                "date": e["hour_dates"][t], "hour_of_day": e["hours_of_day"][t],
                "t": t, "is_observed": bool(e["is_observed"][t]),
                "original_price": p0, "cost": cost, "d_ref": e["d_ref"],
                "mu_ref": float(e["mu_ref_path"][t]),
                "hour_adjustment": float(e["adjustment"][t]),
                # observed world
                "actual_q": int(e["starting_inventory"][t]),
                "actual_discount": a_d,
                "actual_price": p0 * (1 - a_d),
                "actual_units": float(e["actual_sold"][t]),
                # legacy prices under the MODEL's demand
                "legacy_q": rt.get("legacy_q"),
                "legacy_discount": rt.get("legacy_discount"),
                "legacy_price": rt.get("legacy_price"),
                "legacy_mu": rt.get("legacy_mu"),
                "legacy_units": rt.get("legacy_units"),
                # the DP's own choice, same demand model
                "dp_q": rt.get("dp_q"),
                "dp_discount": rt.get("dp_discount"),
                "dp_price": rt.get("dp_price"),
                "dp_mu": rt.get("dp_mu"),
                "dp_units": rt.get("dp_units"),
                "dp_is_entry": rt.get("dp_is_entry", False),
                "dp_feasible_tiers": rt.get("dp_feasible_tiers"),
            })
            h = hours[-1]
            h["dp_minus_actual_discount"] = (
                None if h["dp_discount"] is None
                else h["dp_discount"] - h["actual_discount"])
        row["hours_trace"] = hours
    return row, spreads


def _dp_arm(e, cfg, eps_belief, eps_world=None):
    """The DP arm alone (trace/spreads/other arms stripped), for
    `step_sensitivity`: the solver prices at `eps_belief` while demand
    transitions at `eps_world` (default: the same), so a shifted belief is
    charged only for the prices it changes. Returns (il, mean_discount, path)."""
    if eps_world is None:
        eps_world = eps_belief
    pcfg = cfg["pricing"]
    max_k = pcfg["negbin_max_k"]
    p0, cost = e["original_price"], e["cost"]
    q = float(e["q0"])
    adj = e["adjustment"]
    anchor = None
    disc_cost = sold_total = disc_weighted = 0.0
    path = []
    for t in range(e["hours"]):
        q_int = int(round(q))
        if q_int <= 0:
            path.append(None)
            q = max(q + adj[t], 0.0)
            if q <= 0 and not adj[t + 1:].any():
                break
            continue
        try:
            res = dp_mod.solve(p0, cost, q_int, list(e["mu_ref_path"][t:]),
                               e["d_ref"], eps_belief, e["r"], cfg,
                               anchor_discount=anchor, entry=(t == 0))
        except ValueError:
            break
        d_t = res.tiers[res.optimal_index]
        anchor = d_t
        mu = mu_at(e["mu_ref_path"][t], d_t, e["d_ref"], eps_world,
                   pcfg["demand_floor"])
        sold = min(expected_min_demand_inventory(mu, e["r"], q_int, max_k), q)
        disc_cost += p0 * d_t * sold
        disc_weighted += d_t * sold
        sold_total += sold
        path.append(d_t)
        q = max(q - sold + adj[t], 0.0)
    il = disc_cost + cost * (max(q, 0.0) + e["shrink"])
    return il, (disc_weighted / sold_total if sold_total else 0.0), tuple(path)


def step_sensitivity(frames, cfg, seed=0, sample=300):
    """Prices `learning.max_mean_step` on real episodes: re-solve the DP arm
    at eps +- step; report how many episodes change price, and the IL cost.
    The step shifts the BELIEF while the world stays at base eps. `crossers`
    (deepening bar within one step of |eps|) are where the cap is load-bearing."""
    step = float(cfg["learning"]["max_mean_step"])
    lo = cfg["posterior"]["epsilon_min"]
    hi = cfg["posterior"]["epsilon_max"]
    rng = np.random.default_rng(seed)
    take = min(sample, len(frames))
    picked = [frames[i] for i in rng.choice(len(frames), take, replace=False)]

    shifts = {"deeper_belief": -step, "shallower_belief": +step}
    out = {"step": step, "episodes_swept": take,
           "note": ("DP arm re-solved at eps +- learning.max_mean_step on a "
                    "sample of the replayed episodes, same demand model. "
                    "share_prices_changed near 0 below the deepening bar is "
                    "the measured insensitivity that makes a wrong step "
                    "cheap; crossers (bar within one step of |eps|) are "
                    "where the cap is actually load-bearing.")}
    base = {id(e): _dp_arm(e, cfg, e["eps"]) for e in picked}
    for label, shift in shifts.items():
        changed = il_base = il_shift = 0.0
        cross_n = cross_changed = 0
        disc_delta = 0.0
        for e in picked:
            b_il, b_disc, b_path = base[id(e)]
            eps_s = float(np.clip(e["eps"] + shift, lo, hi))
            # belief shifts, the world stays at base eps -- the delta is the
            # cost of the changed prices alone
            s_il, s_disc, s_path = _dp_arm(e, cfg, eps_s, eps_world=e["eps"])
            moved = s_path != b_path
            changed += moved
            il_base += b_il
            il_shift += s_il
            disc_delta += s_disc - b_disc
            bar = dp_mod.deepening_threshold_epsilon(
                e["original_price"], e["cost"], e["d_ref"])
            # crosser: bar lies between |eps| and |eps_shifted| (direction-aware)
            lo_abs = min(abs(e["eps"]), abs(eps_s))
            hi_abs = max(abs(e["eps"]), abs(eps_s))
            if np.isfinite(bar) and lo_abs < bar <= hi_abs:
                cross_n += 1
                cross_changed += moved
        out[label] = {
            "share_prices_changed": round(changed / take, 4),
            "il_base": round(il_base, 1),
            "il_shifted": round(il_shift, 1),
            "il_delta": round(il_shift - il_base, 1),
            "il_delta_pct": (round((il_shift - il_base) / il_base, 4)
                             if il_base > 0 else None),
            "mean_discount_delta": round(disc_delta / take, 4),
            "crossers": cross_n,
            "crossers_prices_changed": cross_changed,
        }
    return out


def policy_replay(d_pred, cfg, max_episodes=2000, seed=0, workers=None,
                  trace=False):
    """Section 17.3 policy block plus the q-spread distribution for
    tau_initial; replays the same pricing.dp path production uses, and every
    aggregate is over outcome_known episodes only. `trace=True` (for
    tools.export_backtest) adds a per-hour frame, changing the return arity to 4."""
    rng = np.random.default_rng(seed)
    pcfg = cfg["pricing"]
    max_k = pcfg["negbin_max_k"]

    # classify on OBSERVED rows only -- extension rows have no closure sentinel
    obs = (d_pred[d_pred.is_observed] if "is_observed" in d_pred
           else d_pred)
    kind = episodes.classify(obs)
    unfinished = frozenset(kind.index[kind == episodes.NOT_CLOSED])

    eps_ids = d_pred.episode_id.unique()
    if len(eps_ids) > max_episodes:
        eps_ids = rng.choice(eps_ids, max_episodes, replace=False)
    sub = d_pred[d_pred.episode_id.isin(eps_ids)]

    frames = []
    for _, g in sub.groupby("episode_id"):
        e = _episode_frame(g, unfinished)
        if e["q0"] > 0 and e["hours"] >= 1:
            e["trace"] = trace
            frames.append(e)

    results = map_episodes(_replay_one, frames, cfg, workers)

    rows, ledger = [], explore.SpreadLedger()
    for out in results:
        if out is None:
            continue
        row, spreads = out
        rows.append(row)
        for day, costs in spreads:
            ledger.add(day, costs)

    ep_all = pd.DataFrame(rows)
    if not len(ep_all):
        raise RuntimeError("no episodes replayed")

    # pulled out before aggregation: a lists column would break every groupby
    hourly = None
    if trace:
        hourly = pd.DataFrame(
            [h for r in rows for h in r.get("hours_trace", [])])
        ep_all = ep_all.drop(columns=["hours_trace"], errors="ignore")

    # aggregates are over outcome_known episodes only (exclusions counted): an
    # unclosed episode truncates the actual arm while the simulated arms run
    # the full horizon, flattering the DP by exactly the missing tail
    ep = ep_all[ep_all.outcome_known]
    if not len(ep):
        raise RuntimeError(
            "no episode in this sample has a known outcome -- every aggregate "
            "would compare a truncated actual arm against two full-horizon "
            "simulated ones")

    def money(col):
        return round(float(ep[col].sum()), 1)

    lg_il, dp_il = float(ep.legacy_model_il.sum()), float(ep.dp_il.sum())
    block = {
        "episodes_replayed": int(len(ep)),
        "episodes_excluded_unclosed": int(len(ep_all) - len(ep)),
        "share_outcome_known": round(float(ep_all.outcome_known.mean()), 4),
        "basis_note": ("every figure below is over episodes with a KNOWN "
                       "outcome. An unfinished one gives the actual arm only "
                       "the hours the extract covered while both simulated "
                       "arms run the full window, which flatters the DP by "
                       "exactly the missing tail."),
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
        # deepening is economics, not action-set: it reduces IL only when
        # |eps| clears (1-d)/(cost/price - d), so compare the prior to that bar
        "intra_episode_deepening": {
            "median_threshold_abs_eps": round(float(
                ep.deepening_threshold.replace(np.inf, np.nan).median()), 3),
            "median_abs_eps_in_use": round(float(ep.eps.abs().median()), 3),
            "share_episodes_eps_above_threshold": round(float(
                (ep.eps.abs() > ep.deepening_threshold).mean()), 4),
            "note": ("share near 0 means the DP is structurally an "
                     "enter-and-hold policy at the current elasticity: the "
                     "deeper tiers are available every hour and decline to "
                     "pay for themselves. Only a posterior moving past the "
                     "threshold changes that -- widening the action set "
                     "cannot. Threshold ignores censoring and is therefore "
                     "optimistic."),
        },
        # like-for-like: both policies under the SAME demand model, so bias
        # cancels; actual_* vs model figures are fidelity, not policy
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
                 "evidence (design 5.14)."),
    }
    block["q_spread_distribution"] = ledger.distribution()
    block["step_sensitivity"] = step_sensitivity(frames, cfg, seed=seed)
    if trace:
        return block, ep, ledger, hourly
    return block, ep, ledger


def derive_tau_initial(ledger, ep, cfg):
    """Section 12.3: tau_initial is the currency amount (never a rate) whose
    implied daily exploration spend matches budget_share_of_il of daily
    markdown IL. Reports 1.00x by construction -- evidence a tau EXISTS, not
    that it is right; `pipeline.shadow` on an unseen window grades it."""
    if not ledger.decisions:
        return None
    # the launch constant solves against the window's MEAN daily IL;
    # production budgets on the trailing basis and tau_next walks tau with it
    daily_il = pd.DataFrame(ep).groupby("date")["actual_il"].sum()
    budget_per_day = float(cfg["exploration"]["budget_share_of_il"]
                           * daily_il.mean())
    n_days = len(ledger.days)
    tau = ledger.solve_tau(budget_per_day, n_days=n_days)
    if tau is None:
        return None
    return {"tau_initial": round(tau, 2),
            "unit": "currency (expected IL given up, per section 12.3)",
            "implied_daily_spend": round(
                ledger.implied_daily_spend(tau, n_days), 1),
            "daily_budget": round(budget_per_day, 1),
            "budget_basis": ("window mean daily IL for this launch constant; "
                             "production budgets on the trailing "
                             "budget_il_window_days mean"),
            "cost_distribution_quantile": round(ledger.quantile_of(tau), 4),
            "spread_decisions": ledger.decisions,
            "basis": ("every decision hour on the exploit-only replay path. "
                      "Entry-only collection understated the funded decision "
                      "count ~8x; see pricing.explore.SpreadLedger."),
            "validate_on": ("pipeline.shadow --holdout reports "
                            "tau_recommended and tau_controller_trace on a "
                            "window no artifact was fit on.")}
