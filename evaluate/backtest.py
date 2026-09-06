"""backtest -- offline replay through the production decision path (design.md 5.14).

Jobs: the calibration diagnostic (9.2), the tau_initial cross-check (5.8,
5.13), DP sanity. Replay output is never evidence the policy works; the
pilot's own outcomes are. Deterministic: transitions use E[min(D, q)] under
the truncated NB. Fidelity and policy blocks are never summed; IL% uses the
section 2.3 denominator.
"""

from contextlib import contextmanager

import argparse
from common.config import load_config
from common.io import read_json, write_json
from common.provenance import config_fingerprint
from fit.train_baseline import BaselineModel
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from fit.prepare_data import population, pre_launch, split_frames
from fit.fit_dispersion import lookup_r
from common.metrics import fidelity_decomposition
from common import episodes
from common.parallel import map_episodes
from engine import dp as dp_mod
from engine import explore
from engine.posterior import launch_belief
from engine.demand import (mu_at, expected_min_demand_inventory,
                            expected_min_demand_inventory_vec)


# the per-hour columns extend_to_window regenerates on its synthetic tail;
# everything else is episode-constant and carried
_HOURLY = ("episode_id", "date", "hour_of_day", "hours_remaining",
           "starting_inventory", "ending_inventory", "units_sold")


def predict_frame(d, cfg, model, r_lookup):
    """The frame both harnesses price on: extended to the full window BEFORE
    predicting (an early sell-out must not shorten the DP horizon), in
    episode/hour order, with `r` (dispersion lookup) and `mu_ref_hat`.
    One home -- shadow's `_prepare_items` and the replay's
    `_attach_predictions` each carried a copy of this."""
    carry = [c for c in d.columns if c not in _HOURLY]
    d = episodes.extend_to_window(d, carry, cfg["data"]["max_window_hours"]).copy()
    d["r"] = [lookup_r(r_lookup, s, c) for s, c in zip(d.subcategory, d.category)]
    d["mu_ref_hat"] = model.predict_mu_ref(d)
    return d


def _attach_predictions(d, cfg, model, prior, r_lookup):
    """Predicted units at ACTUAL historical prices: mu_ref scaled by the prior
    elasticity, censored at starting inventory, over `predict_frame`."""
    d = predict_frame(d, cfg, model, r_lookup)
    # the WORLD transitions at the prior mean; the DP arm PRICES at the launch
    # belief (posterior.launch_belief), so the replay grades the policy that
    # will actually run against the prior's best guess of the world
    per = prior["per_category"]
    d["eps"] = d.category.map(lambda c: per[str(c)]["mean"]).astype(float)
    d["eps_belief"] = d.category.map(
        lambda c: launch_belief(per[str(c)]["mean"], per[str(c)]["std"], cfg)
    ).astype(float)
    d["predicted_units"] = _predicted_at_actual_prices(
        d, cfg, d.mu_ref_hat.to_numpy())
    return d


def _predicted_at_actual_prices(d, cfg, mu_ref):
    """E[min(D, q)] at the row's ACTUAL price from `mu_ref` (the level in
    force) and the prior-mean elasticity `d.eps` -- the censored basis every
    prediction-vs-sales comparison uses (rule 5a)."""
    ratio = (1 - d.total_discount.to_numpy()) / (1 - d.d_ref.to_numpy())
    mu = np.clip(mu_ref * ratio ** d.eps.to_numpy(),
                 cfg["pricing"]["demand_floor"], None)
    return expected_min_demand_inventory_vec(
        mu, d.r.to_numpy(), d.starting_inventory.to_numpy(),
        cfg["pricing"]["negbin_max_k"])


def _fidelity_metrics(d):
    err = d.predicted_units - d.units_sold
    return {
        "fidelity_episode_sold_ratio": round(
            float(d.units_sold.sum() / d.predicted_units.sum()), 4),
        "fidelity_hourly_mae": round(float(err.abs().mean()), 4),
        "fidelity_hourly_bias": round(float(err.mean()), 4),
        "by_category": {
            k: round(float(g.units_sold.sum() / g.predicted_units.sum()), 4)
            for k, g in d.groupby("category") if g.predicted_units.sum() > 0},
    }


def calibration_window_sweep(d, cfg):
    """Rolling-origin sweep of the calibration fit window: per-category factors
    fit on weeks [t-W, t-1], applied to week t. When the level trends, longer
    windows are MORE stale, not more accurate.

    Every row -- `uncalibrated` included -- is scored on the SAME evaluation
    weeks (the longest window's burn-in). Per-window burn-in would judge a
    long window on a later, smaller sample than a short one, and the ranking
    would then read WHICH WEEKS rather than which window.

    `uncalibrated` is ranked with the rest. When no-factors wins, the level
    factors are adding estimation noise instead of removing bias -- a finding
    for the owner, not a window to paste (W=0 is not a config value), so
    `recommended_fit_window` stays the best CALIBRATED window and the reading
    is carried separately.
    """
    band = cfg["baseline_model"]["calibration_gate_band"]
    tier_step = cfg["pricing"]["tier_step"]
    windows = cfg["baseline_model"]["calibration_window_sweep_weeks"]

    a = d[episodes.is_anchor_row(d, tier_step)].copy()
    if not len(a):
        return "NOT RUN -- no anchor rows"
    a["week"] = episodes.week_key(a.date)
    cw = (a.groupby(["category", "week"], observed=True)
          .agg(sold=("units_sold", "sum"), pred=("predicted_units", "sum"))
          .reset_index())
    weeks = sorted(cw.week.unique())
    start = max(list(windows) + [1])       # common burn-in: one eval set
    if start >= len(weeks):
        return (f"NOT RUN -- {len(weeks)} anchor weeks cannot score a common "
                f"eval set behind the longest window ({start}w)")

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
        """{week: anchor ratio} -- KEYED, so windows can be compared week by
        week rather than only in aggregate."""
        out = {}
        for i, t in enumerate(weeks):
            if i < start:                  # the SAME weeks for every row
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
                out[str(t)] = float(cur.sold.sum() / adj)
        return out

    def paired_vs(window_ratios, base_ratios):
        """Week-paired sign test: did the factors move the anchor ratio
        closer to 1 on the SAME weeks (design 9.2)?"""
        common = sorted(set(window_ratios) & set(base_ratios))
        if len(common) < 2:
            return None
        w = np.abs(np.log([window_ratios[k] for k in common]))
        b = np.abs(np.log([base_ratios[k] for k in common]))
        delta = w - b                          # negative = calibration helped
        better = int((delta < 0).sum())
        decided = int((delta != 0).sum())
        p = (float(binomtest(better, decided, 0.5).pvalue)
             if decided else 1.0)
        return {
            "weeks_paired": len(common),
            "weeks_calibration_helped": better,
            "median_abs_log_delta": round(float(np.median(delta)), 5),
            "sign_test_p": round(p, 4),
            "verdict": ("calibration helps" if p < 0.05 and better * 2 > decided
                        else "calibration hurts" if p < 0.05
                        else "NOT DISTINGUISHABLE at this many weeks"),
        }

    result = {}
    base = ratios_for(0)
    if base:
        result["uncalibrated"] = summarise(list(base.values()))
    for w in windows:
        r = ratios_for(w)
        if r:
            row = summarise(list(r.values()))
            if base:
                row["paired_vs_uncalibrated"] = paired_vs(r, base)
            result[f"trailing_{w}w"] = row

    def rank(k):
        return (-result[k]["share_weeks_in_band"],
                result[k]["mean_abs_log_error"])

    candidates = [k for k in result if k.startswith("trailing_")]
    if candidates:
        best = min(candidates, key=rank)
        result["recommended_fit_window"] = best
        result["eval_weeks_common_from"] = str(weeks[start])
        beats = "uncalibrated" in result and rank("uncalibrated") < rank(best)
        result["uncalibrated_beats_all_windows"] = bool(beats)

        # THE RANKING IS NOT THE EVIDENCE. It compares a handful of aggregate
        # numbers over ~10 weeks and then turns on a lexicographic tie-break,
        # so a ONE-WEEK difference in share_weeks_in_band can decide it. The
        # paired test asks the question that actually matters -- on the same
        # week, did the factors move the anchor ratio closer to 1 -- and says
        # when the answer is "cannot tell at this many weeks".
        paired = {k: (result[k].get("paired_vs_uncalibrated") or {})
                  for k in candidates}
        helped = [k for k, v in paired.items()
                  if v.get("verdict") == "calibration helps"]
        hurt = [k for k, v in paired.items()
                if v.get("verdict") == "calibration hurts"]
        result["calibration_earns_its_keep"] = (
            f"YES for {', '.join(sorted(helped))}" if helped else
            f"NO -- calibration measurably HURTS on {', '.join(sorted(hurt))}"
            if hurt else
            "UNDECIDED -- no window separates from `uncalibrated` week by week. "
            "The ranking still names one, but on this many weeks that choice "
            "is a tie-break, not a measurement. Read "
            "paired_vs_uncalibrated.sign_test_p before acting on it.")
        result["note"] = ("design 9.2 -- rolling-origin sweep, every row "
                          "scored on the same eval weeks")
        if beats:
            result["verdict"] = (
                "NO-FACTORS WINS -- `uncalibrated` beats every fit window on "
                "the sweep's own metrics, so level calibration is adding "
                "estimation noise rather than removing bias. "
                "recommended_fit_window remains the best CALIBRATED window "
                "(W=0 is not a config value); the reading to act on is "
                "whether level calibration earns its keep at all.")
    return result


@contextmanager
def _coverage_preserved(model):
    """A side reading (the weekly-refit mechanism) must not disturb the
    calibration coverage counters or the freeze the gate pass set."""
    saved = (model._cal_rows_scheduled, model._cal_rows_fallback,
             model._cal_rows_frozen, model._cal_rows_static,
             set(model._cal_fallback_weeks))
    frozen_from = model._freeze_from
    try:
        yield
    finally:
        (model._cal_rows_scheduled, model._cal_rows_fallback,
         model._cal_rows_frozen, model._cal_rows_static,
         model._cal_fallback_weeks) = saved
        model.freeze_calibration_from(frozen_from)


def fidelity(d, cfg, model, prior, r_lookup):
    """Design 5.14 fidelity block: how well the model reproduces observed
    sales at actual historical prices. The gate reads the test window -- the
    launch-adjacent regime the 9.2 level factors are fit on; the all-history
    ratio is diagnostic only (dominated by in-sample rows)."""
    # THE GATE GRADES A FROZEN ARTIFACT: a factor fit inside the graded
    # window has read the rows it is graded on, so fidelity freezes at the
    # gate start; the weekly-refit mechanism reading sits beside it.
    split = cfg["data"]["split"]
    gate_start = pd.Timestamp(split["test_start"])
    # predict over the FULL window (the DP plans over it); every ratio below
    # sees observed rows only -- synthetic rows would read as under-prediction
    model.freeze_calibration_from(gate_start)
    d_full = _attach_predictions(d, cfg, model, prior, r_lookup)
    d = d_full[d_full.is_observed]
    splits = split_frames(d, cfg)
    gate_d, gate_window = splits["test"], "test"
    if not len(gate_d) or gate_d.predicted_units.sum() <= 0:
        gate_d, gate_window = d, "all (configured gate window empty)"

    block = _fidelity_metrics(gate_d)
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
    block["measurement_10"] = fidelity_decomposition(gate_d, cfg)

    # trend triage: an anchor-level climb driven by new-assortment SKUs (no
    # rate history -> low prediction) is assortment, not a macro demand trend
    if "sku_ref_sales_rate_30d" in gate_d.columns:
        tier_step = cfg["pricing"]["tier_step"]
        anchor_rows = gate_d[
            episodes.is_anchor_row(gate_d, tier_step)]
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

    # how long should the level factor's fit window be? measured, not assumed
    block["calibration_window_sweep"] = calibration_window_sweep(d, cfg)

    # gate metric: level_at_anchor judges only the artifact's LEVEL at the
    # reference price (pooled_ratio embeds the prior; fallback only)
    m10 = block["measurement_10"]
    anchor_val = (m10.get("level_bias_at_anchor")
                  if isinstance(m10, dict) else None)
    if anchor_val:
        gate_value, gate_metric = anchor_val, "level_bias_at_anchor"
    else:
        gate_value = sold_ratio
        gate_metric = "pooled_ratio (no anchor rows -- fell back)"
    band = cfg["baseline_model"]["calibration_gate_band"]
    block["calibration_gate_band"] = band
    block["calibration_gate_metric"] = gate_metric
    block["calibration_gate_value"] = gate_value
    # DIAGNOSTIC, not a gate (owner, 2026-08-25): out-of-band = drift reading
    # to investigate; fidelity.by_week distinguishes wobble from trend
    block["calibration_gate"] = ("PASS" if band[0] <= gate_value <= band[1]
                                 else "OUT OF BAND -- level diagnostic; "
                                      "investigate drift (design 9.2)")
    block["calibration_frozen_at"] = str(gate_start.date())

    # mechanism reading: the same gate rows under the weekly schedule. NOT
    # the gate; the spread to it is what weekly re-fitting is worth. A
    # factor swap is an exact rescale of mu_ref, so the rows are rescaled
    # by re-fit / frozen factor (what shadow's calibration_regimes does)
    # rather than predicted a second time
    refit_m10 = {}
    if len(gate_d):
        with _coverage_preserved(model):
            frozen_factor = model.level_factors(gate_d)
            model.freeze_calibration_from(None)
            scale = model.level_factors(gate_d) / frozen_factor
        refit_gate = gate_d.assign(predicted_units=_predicted_at_actual_prices(
            gate_d, cfg, gate_d.mu_ref_hat.to_numpy() * scale))
        refit_m10 = fidelity_decomposition(refit_gate, cfg)
    refit_anchor = refit_m10.get("level_bias_at_anchor")
    block["weekly_refit"] = {
        "level_bias_at_anchor": refit_anchor,
        "overall_sold_ratio": refit_m10.get("overall_sold_ratio"),
        "in_band": (None if refit_anchor is None
                    else bool(band[0] <= refit_anchor <= band[1])),
        "vs_frozen": (None if (refit_anchor is None or not anchor_val)
                      else round(refit_anchor - anchor_val, 4)),
        "note": "the gate window under the weekly schedule -- production's "
                "mechanism, not a gate (its factors read the graded rows). "
                "The gap to calibration_gate_value is what weekly re-fitting "
                "buys.",
    }
    return block, d_full


def _episode_frame(g):
    """One episode of `_attach_predictions` output as arrays for the replay.
    The frame is the dp_eligible population, so every episode CLOSED (a
    prepare_data gate: outcome_unknown -> dp_ineligible) and its scrap is
    known; policy_replay refuses anything else."""
    g = g.sort_values(["date", "hour_of_day"])
    obs = g.is_observed.to_numpy()
    # uncovered hours hold the last observed discount (legacy ramps to a cap
    # and holds); both arms must run the same horizon to stay like-for-like
    disc = pd.Series(g.total_discount.to_numpy()).where(
        pd.Series(obs)).ffill().to_numpy()
    obs_rows = g[obs]
    # adjustment computed on OBSERVED rows only: the write-off exemption keys
    # on the LAST row, and the extended frame's last row is a synthetic tail
    adj_obs = episodes.hour_adjustment(obs_rows).to_numpy()
    adjustment = np.zeros(len(g))
    adjustment[obs] = adj_obs
    return {
        "n_observed": int(obs.sum()),
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
        # the belief the DP arm prices at (posterior.launch_belief); no
        # fallback to the world's eps -- a frame without it is not
        # _attach_predictions output
        "eps_belief": float(g["eps_belief"].iloc[0]),
        "episode_id": str(g.episode_id.iloc[0]),
        "sku_id": int(g.sku_id.iloc[0]),
        "fc": str(g.fc.iloc[0]),
        "category": str(g.category.iloc[0]),
    }


def _simulate_arm(e, cfg, price_at, eps_world):
    """Forward-simulate one arm under the model's demand with deterministic
    E[min(D, q)] transitions -- the ONE loop behind the legacy-under-model
    arm, the DP arm and step_sensitivity's re-solved DP arm (three copies
    of the clip/shrink/adjustment bookkeeping used to drift apart here).
    `price_at(t, q_int, anchor)` returns the hour's discount, or None to
    end the arm. Returns disc_cost, sold, left, shrink (APPLIED), scrap,
    mean_discount, path."""
    pcfg = cfg["pricing"]
    max_k = pcfg["negbin_max_k"]
    p0, cost = e["original_price"], e["cost"]
    q, adj, anchor = float(e["q0"]), e["adjustment"], None
    disc_cost = sold_total = disc_weighted = clip = 0.0
    path = []
    for t in range(e["hours"]):
        q_int = int(round(q))
        # an empty shelf ends the arm only if nothing more is coming; the DP
        # never anticipates a delivery -- it learns next hour, as production does
        if q_int <= 0:
            path.append(None)
            clip += max(0.0, -(q + adj[t]))
            q = max(q + adj[t], 0.0)
            if q <= 0 and not adj[t + 1:].any():
                break
            continue
        d_t = price_at(t, q_int, anchor)
        if d_t is None:
            break
        anchor = d_t
        mu = mu_at(e["mu_ref_path"][t], d_t, e["d_ref"], eps_world,
                   pcfg["demand_floor"])
        sold = min(expected_min_demand_inventory(mu, e["r"], q_int, max_k), q)
        disc_cost += p0 * d_t * sold
        disc_weighted += d_t * sold
        sold_total += sold
        path.append(d_t)
        # a negative adjustment can only take what the SIMULATED shelf still
        # holds -- units this arm already sold cannot also shrink
        clip += max(0.0, -(q - sold + adj[t]))
        q = max(q - sold + adj[t], 0.0)
    shrink = max(e["shrink"] - clip, 0.0)
    left = max(q, 0.0)
    return {"disc_cost": disc_cost, "sold": sold_total, "left": left,
            "shrink": shrink, "scrap": cost * (left + shrink),
            "mean_discount": disc_weighted / sold_total if sold_total else 0.0,
            "path": tuple(path)}


def _dp_price(e, cfg, eps_belief, spread_sink=None):
    """`price_at` for a DP arm: solve at `eps_belief`, hand the Q-spread
    costs to `spread_sink` at EVERY decision hour (entry-only tau
    underfunded ~8x) -- on engine.decide's definition and its
    explorability gate, so the backtest ledger and the shadow ledger are
    built over the same decision population."""
    p0, cost = e["original_price"], e["cost"]
    min_tiers = cfg["exploration"]["min_feasible_tiers"]
    dmin = explore.delta_min(cfg, eps_belief, e["category"])

    def price_at(t, q_int, anchor):
        try:
            res = dp_mod.solve(p0, cost, q_int, list(e["mu_ref_path"][t:]),
                               e["d_ref"], eps_belief, e["r"], cfg,
                               anchor_discount=anchor, entry=(t == 0))
        except ValueError:
            return None
        if spread_sink is not None and len(res.q_by_tier) >= min_tiers:
            spread_sink((e["date"], explore.spread_costs(res, dmin)))
        return res.tiers[res.optimal_index]
    return price_at


def _replay_one(e, cfg):
    """One episode's replay: actual path, legacy-under-model arm, DP arm.
    Pure. Returns (row, [(date, q_spread_costs), ...], dp_arm) or None;
    `dp_arm` is (il, mean_discount, path), the base step_sensitivity shifts
    the belief from rather than re-solving it."""
    pcfg = cfg["pricing"]
    p0, cost = e["original_price"], e["cost"]
    tiers, _ = dp_mod.feasible_tiers(p0, cost, pcfg["tier_step"])
    if not tiers:
        return None
    spreads = []

    # ---- actual path economics (observed world, legacy prices); scrap =
    # leftover at close + shrink (units paid for, no revenue)
    a_sold = e["actual_sold"]
    a_disc_cost = float(np.sum(p0 * e["actual_discounts"] * a_sold))
    a_left = max(e["end_inv"], 0)
    a_scrap = cost * (a_left + e["shrink"])
    supply = max(e["supply"], 1)

    # LEGACY path under the MODEL's demand and the DP path: the same
    # generator, so model bias hits both identically (like-for-like)
    arms = {
        "legacy_model": _simulate_arm(
            e, cfg, lambda t, q_int, anchor: float(e["actual_discounts"][t]),
            e["eps"]),
        "dp": _simulate_arm(e, cfg, _dp_price(e, cfg, e["eps_belief"], spreads.append),
                            e["eps"]),
    }

    # units emitted as their own fields, never derived downstream from
    # scrap_cost/cost; scrap = leftover + APPLIED shrink. Shrink is exogenous
    # but each simulated arm absorbs only what its own shelf still held --
    # units it already sold cannot also shrink. Per-arm identity: supply =
    # sold + leftover + shrink holds by construction, so a nonzero residual
    # is a real defect, not rounding
    row = {
        "actual_sold_units": float(a_sold.sum()),
        "actual_leftover_units": float(a_left),
        "actual_scrap_units": float(a_left + e["shrink"]),
        "actual_supply_residual": float(
            e["supply"] - a_sold.sum() - a_left - e["shrink"]),
        "actual_il": a_disc_cost + a_scrap,
        "actual_discount_cost": a_disc_cost, "actual_scrap_cost": a_scrap,
        "actual_denom": p0 * float(a_sold.sum()),
        "actual_cleared": float(a_sold.sum()) / supply,
        "actual_mean_discount": float(np.average(
            e["actual_discounts"],
            weights=a_sold if a_sold.sum() else None)),
    }
    for name, arm in arms.items():
        row.update({
            f"{name}_steps": intra_episode_steps(arm["path"]),
            f"{name}_sold_units": float(arm["sold"]),
            f"{name}_leftover_units": float(arm["left"]),
            f"{name}_scrap_units": float(arm["left"] + arm["shrink"]),
            f"{name}_shrink_applied": float(arm["shrink"]),
            f"{name}_supply_residual": float(
                e["supply"] - arm["sold"] - arm["left"] - arm["shrink"]),
            f"{name}_il": arm["disc_cost"] + arm["scrap"],
            f"{name}_discount_cost": arm["disc_cost"],
            f"{name}_scrap_cost": arm["scrap"],
            f"{name}_denom": p0 * arm["sold"],
            f"{name}_cleared": arm["sold"] / supply,
            f"{name}_mean_discount": arm["mean_discount"],
        })
    row.update({
        "date": e["date"],
        "eps": e["eps"],
        "eps_belief": e["eps_belief"],
        "deepening_threshold": dp_mod.deepening_threshold_epsilon(
            e["original_price"], e["cost"], e["d_ref"]),
        "episode_id": e["episode_id"], "sku_id": e["sku_id"],
        "fc": e["fc"], "category": e["category"],
        "q0": e["q0"], "supply": e["supply"], "shrink": e["shrink"],
        "end_inv": e["end_inv"], "hours": e["hours"],
        "n_observed": e["n_observed"], "r": e["r"],
        "original_price": p0, "cost": cost, "d_ref": e["d_ref"],
    })
    dp = arms["dp"]
    return row, spreads, (dp["disc_cost"] + dp["scrap"], dp["mean_discount"],
                          dp["path"])


def intra_episode_steps(path):
    """How many times an arm's price moved AFTER entry: hours where the
    discount in force deepened against the previous hour (monotone, so a
    move is always a deepening; None hours -- empty shelf -- are skipped)."""
    seen = [d for d in path if d is not None]
    return int(sum(1 for a, b in zip(seen, seen[1:]) if b > a + dp_mod.TIER_EPS))


def intra_episode_moves(ep, cfg):
    """Does the agent move after entry, and where? Overall and by cost-ratio
    band (the deepening bar (1-d)/(gamma-d) falls as cost rises, so the
    bands are where a difference should show). `pct_dp_deepened` compares
    episode MEANS against legacy and says nothing about this."""
    edges = list(cfg["tuning"]["cost_ratio_bands"])
    gamma = ep.cost / ep.original_price
    labels = ([f"cost_ratio<{edges[0]}"]
              + [f"{a}<=cost_ratio<{b}" for a, b in zip(edges, edges[1:])]
              + [f"cost_ratio>={edges[-1]}"])
    band = pd.cut(gamma, [-np.inf] + edges + [np.inf], right=False, labels=labels)

    def summary(g):
        return {"episodes": int(len(g)),
                "share_episodes_with_a_step": round(float((g.dp_steps > 0).mean()), 4),
                "mean_steps_per_episode": round(float(g.dp_steps.mean()), 3),
                "legacy_share_episodes_with_a_step": round(
                    float((g.legacy_model_steps > 0).mean()), 4),
                "share_episodes_eps_above_threshold": round(
                    float((g.eps_belief.abs() > g.deepening_threshold).mean()), 4)}

    return {"overall": summary(ep),
            "by_cost_ratio_band": {str(k): summary(g)
                                   for k, g in ep.groupby(band, observed=True)},
            "note": ("a step is an hour whose discount deepened against the "
                     "previous hour on the arm's OWN path; every hour is a "
                     "fresh solve, so zero steps means holding won every "
                     "hour, not that the price was pinned (design 5.7)")}


def step_sensitivity(replayed, cfg, seed=0):
    """Prices `learning.max_mean_step` on real episodes: re-solve the DP arm
    at eps +- step; report how many episodes change price, and the IL cost.
    The step shifts the BELIEF while the world stays at base eps. `crossers`
    (deepening bar within one step of |eps|) are where the cap is load-bearing.
    `replayed` is [(episode_frame, dp_arm), ...] from `_replay_one`: the
    base arm is the replay's own, never solved a second time."""
    step = float(cfg["learning"]["max_mean_step"])
    lo = cfg["posterior"]["epsilon_min"]
    hi = cfg["posterior"]["epsilon_max"]
    rng = np.random.default_rng(seed)
    take = min(int(cfg["tuning"]["step_sensitivity_episodes"]), len(replayed))
    picked = [replayed[i] for i in rng.choice(len(replayed), take, replace=False)]

    shifts = {"deeper_belief": -step, "shallower_belief": +step}
    out = {"step": step, "episodes_swept": take,
           "note": ("DP arm re-solved at the launch belief +- max_mean_step, "
                    "world held at the prior mean; crossers are where the cap "
                    "is load-bearing.")}
    for label, shift in shifts.items():
        changed = il_base = il_shift = 0.0
        cross_n = cross_changed = 0
        disc_delta = 0.0
        for e, (b_il, b_disc, b_path) in picked:
            eps_s = float(np.clip(e["eps_belief"] + shift, lo, hi))
            # belief shifts, the world stays at base eps -- the delta is the
            # cost of the changed prices alone
            arm = _simulate_arm(e, cfg, _dp_price(e, cfg, eps_s), e["eps"])
            s_il, s_disc, s_path = (arm["disc_cost"] + arm["scrap"],
                                    arm["mean_discount"], arm["path"])
            moved = s_path != b_path
            changed += moved
            il_base += b_il
            il_shift += s_il
            disc_delta += s_disc - b_disc
            bar = dp_mod.deepening_threshold_epsilon(
                e["original_price"], e["cost"], e["d_ref"])
            # crosser: bar lies between |eps| and |eps_shifted| (direction-aware)
            lo_abs = min(abs(e["eps_belief"]), abs(eps_s))
            hi_abs = max(abs(e["eps_belief"]), abs(eps_s))
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


def policy_replay(d_pred, cfg, max_episodes=2000, seed=0, workers=None):
    """Design 5.14 policy block plus the q-spread distribution for the tau
    cross-check; replays the same engine.dp path production uses. Takes
    `_attach_predictions` output over the dp_eligible population."""
    rng = np.random.default_rng(seed)

    # dp_eligible means CLOSED (prepare_data's outcome_unknown gate). An
    # unclosed episode would truncate the actual arm while the simulated arms
    # run the full horizon, flattering the DP by exactly the missing tail --
    # refused loudly rather than aggregated. Classified on OBSERVED rows only:
    # extension rows have no closure sentinel.
    kind = episodes.classify(d_pred[d_pred.is_observed])
    unclosed = int((kind == episodes.NOT_CLOSED).sum())
    if unclosed:
        raise ValueError(
            f"policy_replay takes the dp_eligible population, but {unclosed} "
            "episode(s) never closed -- apply "
            "fit.prepare_data.population(d, cfg, 'dp_eligible') first")

    eps_ids = d_pred.episode_id.unique()
    if len(eps_ids) > max_episodes:
        eps_ids = rng.choice(eps_ids, max_episodes, replace=False)
    sub = d_pred[d_pred.episode_id.isin(eps_ids)]

    frames = []
    for _, g in sub.groupby("episode_id"):
        e = _episode_frame(g)
        if e["q0"] > 0 and e["hours"] >= 1:
            frames.append(e)

    # results return in submission order, so each pairs with its frame
    results = map_episodes(_replay_one, frames, cfg, workers)

    rows, ledger, replayed = [], explore.SpreadLedger(), []
    for e, out in zip(frames, results):
        if out is None:
            continue
        row, spreads, dp_arm = out
        rows.append(row)
        replayed.append((e, dp_arm))
        for day, costs in spreads:
            ledger.add(day, costs)

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
        # deepening is economics, not action-set: it reduces IL only when
        # |eps| clears (1-d)/(cost/price - d), so compare the prior to that bar
        "intra_episode_deepening": {
            "median_threshold_abs_eps": round(float(
                ep.deepening_threshold.replace(np.inf, np.nan).median()), 3),
            "median_abs_eps_in_use": round(float(ep.eps_belief.abs().median()), 3),
            "median_abs_eps_prior": round(float(ep.eps.abs().median()), 3),
            "cold_start_shift_std": float(cfg["posterior"].get("cold_start_shift_std") or 0.0),
            "share_episodes_eps_above_threshold": round(float(
                (ep.eps_belief.abs() > ep.deepening_threshold).mean()), 4),
            "note": ("share near 0 = enter-and-hold at the current "
                     "elasticity; only the posterior moving past the bar "
                     "changes that, widening the action set cannot."),
        },
        # the direct measurement of enter-and-hold: steps on the DP arm's own
        # path, overall and where cost makes the bar reachable
        "intra_episode_moves": intra_episode_moves(ep, cfg),
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
        "note": ("legacy-under-model vs DP-under-model, same demand "
                 "generator both arms; actual_* is fidelity, not policy. "
                 "Replay output is never evidence the policy works (5.14)."),
    }
    block["q_spread_distribution"] = ledger.distribution()
    block["step_sensitivity"] = step_sensitivity(replayed, cfg, seed=seed)
    return block, ep, ledger


def derive_tau_initial(ledger, ep, cfg, launch_std):
    """The tau whose implied daily exploration spend matches
    budget_share_of_il of daily markdown IL (design 5.8), solved on the
    exploit-only replay path -- a CROSS-CHECK; the launch paste is shadow's
    anchored-path derivation (5.13). Reports 1.00x by construction: evidence
    a tau EXISTS, not that it is right (AGENTS rule 17)."""
    if not ledger.decisions:
        return None
    # ONE day count on both sides, and the same one shadow divides by
    # (episodes.calendar_days over the replayed sample's dates): the budget
    # is IL per calendar day and the spend is divided by the same days, so
    # the count cancels in tau; `days`, implied_daily_spend and daily_budget
    # are on the calendar basis. On the pre-launch frame that span crosses
    # data.exclusion_window, which lowers both per-day figures equally.
    ep = pd.DataFrame(ep)
    n_days = int(episodes.calendar_days(ep.date))
    # production's own budget rule at the launch posterior width
    budget_per_day = float(explore.budget_today(
        ep.actual_il.sum() / n_days, launch_std, cfg))
    tau = ledger.solve_tau(budget_per_day, n_days=n_days)
    if tau is None:
        return None
    return {"tau_initial": round(tau, 2),
            "unit": "currency: expected IL given up (design 5.8)",
            # solved on policy_replay's SAMPLE (--policy-episodes), not the
            # window: the daily IL and spend are both sample-scaled, so the
            # ratio holds but the currency amount is the sample's
            "episodes_in_sample": int(len(ep)),
            "days": n_days,
            "implied_daily_spend": round(
                ledger.implied_daily_spend(tau, n_days), 1),
            "daily_budget": round(budget_per_day, 1),
            "budget_scale_std": round(float(launch_std), 4),
            "cost_distribution_quantile": round(ledger.quantile_of(tau), 4),
            "spread_decisions": ledger.decisions,
            "note": ("design 5.14 -- exploit-only path, every decision hour; "
                     "the launch value is shadow's tau_initial_derivation "
                     "(5.13)")}


def main():
    ap = argparse.ArgumentParser(prog="evaluate.backtest")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="reports/backtest.json")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--policy-episodes", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for the episode replay. 0 = every core "
                         "but one. Episodes are independent and the replay is "
                         "deterministic, so this changes speed and nothing "
                         "else -- the report is identical either way.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)
    # PRE-LAUNCH artifact: must see nothing past the gate window, or
    # tau_initial gets fitted on the window reserved for grading it
    before = d.episode_id.nunique()
    d = pre_launch(d, cfg)
    excluded = before - d.episode_id.nunique()
    # dp_eligible is a precondition, not a population choice, and must
    # precede fidelity() -- that is where extend_to_window happens
    on_dp = d.episode_id.nunique()
    d = population(d, cfg, "dp_eligible")
    dp_excluded = on_dp - d.episode_id.nunique()
    if d.empty:
        raise SystemExit(
            f"no episodes opened on or before split.test_end "
            f"({cfg['data']['split']['test_end']})")
    model = BaselineModel(cfg)
    prior = read_json(cfg["posterior"]["prior"]["path"])
    r_lookup = read_json(cfg["dispersion"]["r_lookup_path"])

    fid, d_pred = fidelity(d, cfg, model, prior, r_lookup)
    pol, ep, ledger = policy_replay(d_pred, cfg,
                                    max_episodes=args.policy_episodes,
                                    seed=args.seed, workers=args.workers)
    # the widest launch cell: GLOBAL's std is the widest member's, so the
    # max over categories is the day-one widest_std whatever the routing
    tau = derive_tau_initial(
        ledger, ep, cfg,
        max(v["std"] for v in prior["per_category"].values()))

    out = {
        "population": {
            "episodes": int(d.episode_id.nunique()),
            "episodes_excluded_after_test_end": int(excluded),
            "episodes_excluded_dp_ineligible": int(dp_excluded),
            "sees_up_to": cfg["data"]["split"]["test_end"],
            "note": ("The backtest is pre-launch and sees nothing past the "
                     "gate window, so by_week and by_window['all'] stop at "
                     "test_end. The hold-out is read once, by "
                     "`evaluate.shadow --holdout`."),
        },
        "config": config_fingerprint(cfg, "backtest"),
        "artifact_versions": {
            "baseline_model_version": model.version,
            "train_population": "eligible",
            "prior_source": prior["source"],
            "config_version": cfg["meta"]["config_version"],
            # did every priced row get its OWN week's level factors, or did
            # some fall back to the frozen set? Silent by construction
            "calibration_coverage": model.calibration_coverage(),
        },
        # two blocks, reported separately and never summed (design 5.14)
        "fidelity": fid,
        "policy_deltas": pol,
        "tau_initial_derivation": tau,
    }
    write_json(args.out, out)

    print(f"fidelity_episode_sold_ratio : {fid['fidelity_episode_sold_ratio']}")
    print(f"gate ({fid['calibration_gate_metric']}) : "
          f"{fid['calibration_gate_value']} vs {fid['calibration_gate_band']}"
          f"  -> {fid['calibration_gate']}")
    gap = pol["policy_gap_like_for_like"]
    print(f"observed world  : legacy IL {pol['actual_il']:,.0f} "
          f"(IL% {pol['actual_il_pct']})")
    print(f"model world     : legacy IL {pol['legacy_model_il']:,.0f} "
          f"vs DP IL {pol['dp_il']:,.0f}  -> DP reduces IL by "
          f"{gap['dp_il_reduction_pct_of_legacy']:.1%} (like-for-like)"
          if gap["dp_il_reduction_pct_of_legacy"] is not None else
          "model world     : like-for-like gap unavailable")
    print(f"pct_dp_deepened             : {pol['pct_dp_deepened']:.1%}  "
          "(episode MEAN deeper than legacy's)")
    mv = pol["intra_episode_moves"]
    print(f"moves after entry (DP arm)  : {mv['overall']['share_episodes_with_a_step']:.1%} "
          f"of episodes step at least once, {mv['overall']['mean_steps_per_episode']:.2f} "
          f"steps/episode (legacy {mv['overall']['legacy_share_episodes_with_a_step']:.1%})")
    for band, b in mv["by_cost_ratio_band"].items():
        print(f"    {band:<22s} {b['share_episodes_with_a_step']:>6.1%} step, "
              f"{b['share_episodes_eps_above_threshold']:>6.1%} of episodes above the bar "
              f"({b['episodes']:,} episodes)")
    if tau:
        print(f"tau_initial (currency)      : {tau['tau_initial']}  "
              f"(q{tau['cost_distribution_quantile']:.2f} of Q-spread; "
              f"daily spend {tau['implied_daily_spend']:,.0f} "
              f"vs budget {tau['daily_budget']:,.0f})")
        print("advisory cross-check on the exploit-only path -- NOT the paste; "
              "exploration.tau_initial is pasted from "
              "shadow.tau_initial_derivation (ops.tune)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
