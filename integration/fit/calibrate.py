"""fit.calibrate -- the level-factor fit (design 9.2): the ONE level
estimator, the fit basis it reads, the weekly point-in-time schedule, the
convergence check of the factor <-> r loop and the console summary.

The applier (`BaselineModel.level_factors`) and the CLI stay on
`fit.train_baseline`: `python3 -m fit.train_baseline --fit-calibration` and
`--check-convergence` call in here, and the `written_by` stamp keeps that
command's name. Reads the model and `r`; nothing here is imported by
`fit.train_baseline` at module level, so the two do not form a cycle.
"""

import json

import numpy as np
import pandas as pd

from common.io import read_json, write_json
from common.provenance import stamp
from common import windows
from fit.prepare_data import population, scope
from fit.fit_dispersion import lookup_r_vec
# the schedule readers stay beside the applier that reads them (the one-home
# anchor is `fit.train_baseline.schedule_reaches`); named here too so a
# caller of the fit finds them where the schedule is written
from fit.train_baseline import (BaselineModel, GRAIN, schedule_reaches,   # noqa: F401
                                weeks_held_at_anchor)
from engine.demand import expected_min_demand_inventory_vec


def attach_fit_basis(frame, model, r_lookup, raw=True):
    """The two per-row inputs a prediction-vs-sales comparison reads,
    attached in place: `mu_ref_hat` -- the RAW mu_ref (`raw=True`, what a
    level solve reads: the factors are what it is solving for) or the
    CALIBRATED one (`raw=False`, what the harnesses price on) -- and, on
    the censored basis, each row's r (`r_val`, `lookup_r_vec` down the
    fallback chain) when `r_lookup` is given. Neither depends on the
    window, so a schedule attaches them to its whole scope once and
    slices, instead of predicting per week."""
    frame["mu_ref_hat"] = (model.predict_mu_ref(frame, raw=True) if raw
                           else model.predict_mu_ref(frame))
    if r_lookup is not None:
        frame["r_val"] = lookup_r_vec(r_lookup, frame.subcategory,
                                      frame.category)
    return frame


def pinned_cells(detail):
    """{cell: bracket end} for every cell of a detail table whose own solve
    pinned (rule 3)."""
    return {k: v["at_bound"] for k, v in detail.items() if v.get("at_bound")}


def solve_level_factors(calib, model, k_shrink, min_anchor,
                        tier_step, max_k, r_lookup, predicted=False, cfg=None):
    """Factors for one fit window (shared by the anchor fit, every schedule
    week, shadow's weekly re-fit and the backtest's window sweep -- the ONE
    level estimator). Returns (factors, detail, global_factor, global_at_bound,
    detail_category), or None when the window holds too few anchor rows --
    the caller holds those weeks at the frozen anchor. Cells ABOVE that
    floor are shrunk toward their parent (category, then global) by
    `k_shrink` pseudo-units. EVERY cell of the window's population gets a
    factor: one with no anchor rows at all takes its parent's outright
    (`held_at_parent` in its detail, `factor` on every entry) -- left out,
    it priced at 1.0 while its category solved to something else. A bound
    is not a solve, at any level (rule 3): a cell whose bisection ran off
    the bracket carries `at_bound` in its detail; a subcategory whose
    PARENT category pinned carries `parent_at_bound` (it is shrunk toward
    a bracket end, however thin it is); `global_at_bound` names the end the
    GLOBAL solve pinned to (None when it converged). `predicted` says
    `attach_fit_basis` already ran on `calib` (a schedule attaches once to
    its scope); otherwise it runs here, on `calib` in place. A caller
    whose basis is already attached may pass `cfg` and no model."""
    bm = (cfg if model is None else model.cfg)["baseline_model"]
    f_lo, f_hi = (float(x) for x in bm["calibration_factor_search_bounds"])
    halvings = int(bm["calibration_factor_bisection_steps"])

    if not predicted:
        attach_fit_basis(calib, model, r_lookup)

    def solve_factor(anchor):
        """(factor, predicted at f=1, at_bound): solved against the censored
        basis E[min(D,q)] -- the gate's quantity. `at_bound` names the bracket
        end the solve was pinned to (None when the bisection converged)."""
        sold = float(anchor["units_sold"].sum())
        mu = anchor["mu_ref_hat"].to_numpy()
        if r_lookup is None:
            pred = float(mu.sum())
            return (sold / pred if pred > 0 else 1.0), pred, None
        r = anchor["r_val"].to_numpy()
        q = anchor["starting_inventory"].to_numpy()

        def predicted(f):
            return float(expected_min_demand_inventory_vec(
                f * mu, r, q, max_k).sum())

        base = predicted(1.0)
        if base <= 0 or sold <= 0:
            return 1.0, base, None
        lo, hi = f_lo, f_hi
        if predicted(lo) > sold:
            return lo, base, "lower"
        if predicted(hi) < sold:
            return hi, base, "upper"
        for _ in range(halvings):        # monotone in f
            mid = (lo + hi) / 2
            if predicted(mid) < sold:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2, base, None

    def shrink(cell, parent, evidence):
        if evidence <= 0 or cell <= 0:
            return parent
        w = evidence / (evidence + k_shrink)
        return float(np.exp(w * np.log(cell) + (1 - w) * np.log(parent)))

    anchor_all = calib[windows.is_anchor_row(calib, tier_step)]
    if len(anchor_all) < min_anchor or anchor_all["mu_ref_hat"].sum() <= 0:
        return None

    f_global, _, global_at_bound = solve_factor(anchor_all)

    def fit_level(groups, parent_of, population):
        """`parent_of(key, frame)` -> (parent factor, the bracket end the
        parent's OWN solve pinned to, or None); `population` is the
        window's whole frame, whose cells without an anchor row are held
        at their parent."""
        out, det = {}, {}
        for key, g in groups:
            raw_f, pred, at_bound = solve_factor(g)
            evidence = float(g["units_sold"].sum())
            parent, parent_at_bound = parent_of(key, g)
            f = shrink(raw_f, parent, evidence)
            out[str(key)] = round(float(f), 4)
            det[str(key)] = {
                "factor": out[str(key)],
                "anchor_rows": int(len(g)),
                "anchor_sold": int(evidence),
                "anchor_predicted_at_f1": round(float(pred), 1),
                "raw_factor": round(float(raw_f), 4),
                "parent_factor": round(float(parent), 4),
                "shrinkage_weight_on_self": round(
                    float(evidence / (evidence + k_shrink)), 3),
            }
            if at_bound:
                # the literal bracket end, not a solve: the sales this cell
                # wants sit outside [f_lo, f_hi] x its prediction
                det[str(key)]["at_bound"] = at_bound
                det[str(key)]["at_bound_note"] = (
                    f"raw_factor is the {at_bound} end of "
                    f"calibration_factor_search_bounds {[f_lo, f_hi]}, not a "
                    "solved value -- the bisection bracket does not contain "
                    "the sold total. Investigate the cell before trusting it.")
            if parent_at_bound:
                # the parent this cell is shrunk toward is itself a bracket
                # end: the thinner the cell, the more of its factor is bound
                det[str(key)]["parent_at_bound"] = parent_at_bound
        # a cell of the population with NO anchor row: nothing to solve or
        # shrink, so it is its parent's -- said so, never silently 1.0
        for key, g in population:
            if str(key) in out:
                continue
            parent, parent_at_bound = parent_of(key, g)
            out[str(key)] = round(float(parent), 4)
            det[str(key)] = {"factor": out[str(key)], "anchor_rows": 0,
                             "anchor_sold": 0, "held_at_parent": True,
                             "parent_factor": round(float(parent), 4),
                             "shrinkage_weight_on_self": 0.0}
            if parent_at_bound:
                det[str(key)]["parent_at_bound"] = parent_at_bound
        return out, det

    cat_factors, cat_detail = fit_level(
        anchor_all.groupby("category"),
        lambda k, g: (f_global, global_at_bound),
        calib.groupby("category"))

    def parent_of_sub(key, g):
        cat = str(g["category"].iloc[0])
        if cat not in cat_factors:
            return f_global, global_at_bound
        return cat_factors[cat], cat_detail[cat].get("at_bound")

    factors, detail = fit_level(anchor_all.groupby("subcategory"),
                                parent_of_sub, calib.groupby("subcategory"))
    return factors, detail, f_global, global_at_bound, cat_detail


# the name the harnesses and the older anchors call the estimator by
_solve_level_factors = solve_level_factors


def category_factors(detail_category):
    """{category: factor} off a level solve's category detail -- the parent
    table `BaselineModel.level_factors` waterfalls to for a subcategory
    the window never saw."""
    return {k: v["factor"] for k, v in detail_category.items() if "factor" in v}


def keys_held_at_parent(detail):
    """The cells a level solve held at their parent's factor for want of an
    anchor row in its window."""
    return sorted(k for k, v in detail.items() if v.get("held_at_parent"))


def fit_level_calibration(d, cfg):
    """Multiplicative level factors on ANCHOR ROWS only (elasticity ~1 there,
    so slope error cannot leak in). Each cell is shrunk toward its parent
    (category, then global) by `calibration_shrinkage_units` pseudo-units --
    a thin cell follows its parent, it is not held at 1.0; only a WINDOW
    with fewer than `calibration_min_anchor_rows` anchor rows is unfitted.
    The anchor set is the trailing W weeks ending at the gate window's start
    -- disjoint from what the gate grades -- plus a weekly point-in-time
    schedule fit on the trailing window ending strictly before each week."""

    model = BaselineModel(cfg)
    split = cfg["data"]["split"]
    gate_start = pd.Timestamp(split["test_start"])   # gate window = test
    weeks_back = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
    lo = gate_start - pd.Timedelta(weeks=weeks_back)
    # SAME population and SAME cut as the weekly schedule below (rules 14/15):
    # the frozen fallback and the by-week factors must be solved on the same
    # rows, since check_calibration_convergence compares them cell by cell
    calib = windows.window_slice(population(d, cfg),
                                 lo.strftime("%Y-%m-%d"),
                                 (gate_start - pd.Timedelta(days=1))
                                 .strftime("%Y-%m-%d")).copy()
    if not len(calib):
        raise RuntimeError("calibration fit window contains no rows")

    fit_dates = pd.to_datetime(calib.date)
    train_end = pd.Timestamp(split["train_end"])
    in_sample_share = float((fit_dates <= train_end).mean())
    tier_step = cfg["pricing"]["tier_step"]
    max_k = cfg["pricing"]["negbin_max_k"]

    r_lookup = read_json(cfg["dispersion"]["r_lookup_path"])
    censored_basis = r_lookup is not None

    min_anchor = cfg["baseline_model"]["calibration_min_anchor_rows"]
    k_shrink = cfg["baseline_model"]["calibration_shrinkage_units"]

    fitted = solve_level_factors(calib, model, k_shrink, min_anchor,
                                 tier_step, max_k, r_lookup)
    if fitted is None:
        anchors = int(windows.is_anchor_row(calib, tier_step).sum())
        raise RuntimeError(
            f"fit window has only {anchors} anchor rows (need {min_anchor})"
            " -- widen calibration_fit_trailing_weeks")
    factors, detail, f_global, global_at_bound, detail_category = fitted

    # POINT-IN-TIME schedule. Pre-launch: over every pre-launch week (a
    # forward replay must see exactly what production re-fits weekly; the
    # gate freezes via freeze_calibration_from, not by bounding the artifact).
    # After `data.launch_date` is set: THIS is the weekly cron, so the
    # schedule runs through the latest data and one week past it -- the
    # week being priced, whose own rows are not complete yet -- while every
    # sealed fit keeps its pre-launch scope. Moving split.test_end instead
    # would rescope every other fit.
    launched = bool(cfg["data"].get("launch_date"))
    frame = (population(d, cfg) if launched
             else scope(d, cfg, "pre_launch")).copy()
    weeks = sorted(windows.week_key(frame.date).unique())
    if launched and weeks:
        weeks.append(windows.week_after(weeks[-1]))
    by_week, by_week_category, coverage, pinned_by_week = {}, {}, [], {}
    opened = windows.opening_dates(frame)      # once, not once per week
    attach_fit_basis(frame, model, r_lookup)   # once: the windows are slices
    for w in weeks:
        window, weeks_seen = windows.trailing_weeks_window(
            frame, w, weeks_back, opened=opened)
        if not len(window):
            # an EMPTY trailing window (data start, a gap) holds the anchor
            # exactly as a thin one does, and must be recorded the same way
            # or schedule_reaches never reaches it: the gate then read it
            # as a missed cron and advance re-fit it every morning
            coverage.append({"week": w, "fitted": False})
            continue
        f = solve_level_factors(window, model, k_shrink, min_anchor,
                                tier_step, max_k, r_lookup, predicted=True)
        if f is None:                       # too thin: hold the anchor, say so
            coverage.append({"week": w, "fitted": False})
            continue
        by_week[w] = f[0]
        by_week_category[w] = category_factors(f[4])
        # the week's pinned cells, by level -- the per-cell detail itself is
        # the anchor fit's; a week keeps only what rule 3 needs
        pinned = {**{f"category:{k}": v for k, v in pinned_cells(f[4]).items()},
                  **{f"subcategory:{k}": v
                     for k, v in pinned_cells(f[1]).items()}}
        if pinned:
            pinned_by_week[w] = pinned
        coverage.append({"week": w, "fitted": True,
                         "fit_rows": int(len(window)),
                         "weeks_in_window": weeks_seen,
                         "partial": weeks_seen < weeks_back,
                         "global_at_bound": f[3],
                         "keys_held_at_parent": keys_held_at_parent(f[1])})
    schedule = {
        "mode": "rolling_trailing",
        "scope": (f"production -- launch_date {cfg['data']['launch_date']}; "
                  "through the latest data plus the week being priced"
                  if launched else
                  f"pre-launch -- through split.test_end {split['test_end']}"),
        "trailing_weeks": weeks_back,
        "gate_freezes_at": str(gate_start.date()),
        # the frozen anchor's window, both ends INCLUSIVE (the day before
        # the gate opens is its last day)
        "anchor_fit_window": [str(lo.date()),
                              str((gate_start - pd.Timedelta(days=1)).date())],
        "week_key": "ISO week start the factors APPLY to; fit on the "
                    "trailing window ending strictly before it",
        "weeks_fitted": sum(1 for c in coverage if c["fitted"]),
        # too thin (or empty) to fit: priced on the FROZEN ANCHOR
        # (level_factors), never on raw mu
        "weeks_unfitted_held_at_anchor": [c["week"] for c in coverage
                                          if not c["fitted"]],
        # per fitted week, the cells priced at their PARENT's factor for
        # want of an anchor row in the trailing window
        "keys_held_at_parent_by_week": {
            c["week"]: c["keys_held_at_parent"] for c in coverage
            if c.get("keys_held_at_parent")},
        # fit on less history than trailing_weeks asks for (extract start)
        "weeks_on_partial_window": [
            {"week": c["week"], "weeks_in_window": c["weeks_in_window"]}
            for c in coverage if c.get("partial")],
        # weeks whose GLOBAL solve pinned at a bracket end: every cell's
        # parent that week is a bound, not a solve (rule 3)
        "weeks_global_at_bound": {c["week"]: c["global_at_bound"]
                                  for c in coverage
                                  if c.get("global_at_bound")},
        # {week: {"level:cell": bracket end}} for the weeks with a pinned
        # cell -- the schedule's rule-3 flags, since by_week keeps factors only
        "pinned_by_week": pinned_by_week,
        "by_week": by_week,
        # the parent level per week: a subcategory absent from a week's
        # window prices at its category's factor (level_factors waterfall)
        "by_week_category": by_week_category,
    }

    # every pinned solve in the artifact, anchor and schedule, in one list
    # (ops.status reads this field alone): {scope, level, cell, at_bound}
    pins = ([{"scope": "anchor", "level": "category", "cell": k, "at_bound": v}
             for k, v in sorted(pinned_cells(detail_category).items())]
            + [{"scope": "anchor", "level": "subcategory", "cell": k,
                "at_bound": v} for k, v in sorted(pinned_cells(detail).items())]
            + [{"scope": w, "level": lc.split(":", 1)[0],
                "cell": lc.split(":", 1)[1], "at_bound": v}
               for w, cells in sorted(pinned_by_week.items())
               for lc, v in sorted(cells.items())])

    fv = np.array(list(factors.values()), dtype=float)
    payload = {"grain": GRAIN,
               "factor_summary": {
                   "p10": round(float(np.percentile(fv, 10)), 4),
                   "p50": round(float(np.percentile(fv, 50)), 4),
                   "p90": round(float(np.percentile(fv, 90)), 4),
                   "share_within_5pct_of_1": round(
                       float((np.abs(fv - 1.0) <= 0.05).mean()), 4),
                   "note": "clustered on 1.0 -> model is level-correct; "
                           "wide -> systematic per-cell bias worth fixing "
                           "in training",
               },
               "factors": factors,
               # the parent level, for a subcategory outside the window
               "factors_category": category_factors(detail_category),
               # anchor-window cells held at their parent (no anchor rows)
               "keys_held_at_parent": keys_held_at_parent(detail),
               "schedule": schedule,
               "detail": detail,
               # the parent level's own solves: a subcategory shrinks toward
               # its category's factor, so a category pinned here is a bound
               # under every thin subcategory it holds (parent_at_bound)
               "detail_category": detail_category,
               "pinned_cells": pins,
               "global_factor": round(float(f_global), 4),
               # None, or the bracket end ("lower"/"upper") the global solve
               # pinned to: then the parent every category shrinks toward is
               # a bound, not a solve (rule 3)
               "global_factor_at_bound": global_at_bound,
               "shrinkage_units": k_shrink,
               # the frozen anchor is ONE trailing window ending the day
               # before the gate opens; the rolling re-fit is `schedule`
               "fit_window": "trailing_before_gate",
               "fit_basis": "censored E[min(D,q)]" if censored_basis
                   else "raw mu (r_lookup missing)",
               "fit_window_dates": [str(fit_dates.min().date()),
                                    str(fit_dates.max().date())],
               "fit_rows": int(len(calib)),
               "fit_in_sample_share": round(in_sample_share, 4),
               "split": split,
               "basis": ("anchor rows only; every cell shrunk toward its "
                         "parent (category, then global) by "
                         "calibration_shrinkage_units; a cell of the window "
                         "with no anchor row takes its parent's factor "
                         "(keys_held_at_parent); a window below "
                         "calibration_min_anchor_rows is unfitted (held at "
                         "the frozen anchor); a cell whose bisection pinned "
                         "at calibration_factor_search_bounds carries "
                         "at_bound in detail, one whose parent pinned "
                         "parent_at_bound; pinned_cells lists every pin, "
                         "anchor and schedule")}
    path = cfg["baseline_model"]["calibration_factor_path"]
    # The weekly PRODUCTION re-fit (launch_date set) retrains nothing: the
    # prior, r and rho it was checked against are the ones on disk, so the
    # convergence verdict still holds and is carried forward -- written
    # without it, ops.tune read "never checked" and BLOCKED the daily lane
    # after every cron. The bootstrap path is unchanged: its verdict comes
    # from the --check-convergence that follows each fit.
    previous = read_json(path) or {}
    if launched and previous.get("convergence"):
        conv = dict(previous["convergence"])
        conv["carried_from"] = (
            conv.get("carried_from")
            or (previous.get("provenance") or {}).get("created_at"))
        conv["carried_note"] = (
            "carried forward from the previous artifact by the weekly "
            "production re-fit, which moves no prior/r/rho; status still "
            "reads checked_against against the artifacts on disk")
        payload["convergence"] = conv
    # stamped with the COMMAND that writes it, which stays on train_baseline
    write_json(path, stamp(payload, cfg, model.version,
                           "fit.train_baseline --fit-calibration"))
    return factors


def check_calibration_convergence(d, cfg, commit=False):
    """Has the calibration <-> dispersion fixed point settled?

    The chain is circular: the factor solve consumes `r`, while `r`, `rho`
    and the prior are fitted against CALIBRATED mu_ref. This re-solves the
    factors with the prior/r now on disk and compares per cell (and per
    schedule week) in log space. Default is a DRY RUN (artifact restored);
    `commit` keeps the re-solve, which is bit-for-bit the next turn's
    --fit-calibration -- only sound inside the loop.
    """
    path = cfg["baseline_model"]["calibration_factor_path"]
    with open(path) as f:
        old = json.load(f)
    tol = cfg["baseline_model"]["calibration_convergence_tol_log"]

    try:
        fit_level_calibration(d, cfg)          # iteration k+1, on disk briefly
        with open(path) as f:
            new = json.load(f)
    finally:
        if not commit:
            with open(path, "w") as f:         # dry run: restore iteration k
                json.dump(old, f, indent=2)

    def compare(a, b, scope):
        worst, missing = (0.0, None), []
        for key in sorted(set(a) | set(b)):
            fa, fb = a.get(key), b.get(key)
            if not fa or not fb:
                missing.append(f"{scope}:{key}")
                continue
            dlog = abs(float(np.log(fb / fa)))
            if dlog > worst[0]:
                worst = (dlog, f"{scope}:{key}")
        return worst, missing

    worst, missing = compare(old.get("factors", {}),
                             new.get("factors", {}), "anchor")
    old_bw = (old.get("schedule") or {}).get("by_week", {})
    new_bw = (new.get("schedule") or {}).get("by_week", {})
    for wk in sorted(set(old_bw) | set(new_bw)):
        w, m = compare(old_bw.get(wk) or {}, new_bw.get(wk) or {}, wk)
        if w[0] > worst[0]:
            worst = w
        missing += m

    # digests of what the verdict was checked against, so `status` can flag
    # a verdict whose chain has since moved -- read through the one artifact
    # walk in common.provenance
    from common.provenance import collect
    checked_against = {row["artifact"]: row["sha256"]
                       for row in collect(cfg)
                       if row["artifact"] in ("prior", "r_lookup", "rho")
                       and row["present"]}

    # anchor rows behind the worst cell IN THE FROZEN ANCHOR FIT: a thin,
    # shrinkage-dominated cell reads identically to an unsettled loop unless
    # the row count is shown. The per-week detail is not kept, so a worst
    # cell in a schedule week is sized by its anchor-fit count, and labelled
    # as such
    worst_rows = None
    if worst[1]:
        cell = worst[1].split(":", 1)[1]
        worst_rows = ((old.get("detail") or {}).get(cell) or {}).get("anchor_rows")

    # trajectory (carried across runs): contracting vs stalled/oscillating
    prior = ((old.get("convergence") or {}).get("history") or [])
    history = (prior + [round(worst[0], 6)])[-6:]

    converged = not missing and worst[0] <= tol
    block = {
        "tol_log": tol,
        "history": history,
        "worst_cell_anchor_rows": worst_rows,
        "worst_cell_anchor_rows_basis": (
            "the cell's anchor rows in the FROZEN ANCHOR fit, whatever "
            "scope the worst cell is in (per-week cell counts are not kept)"),
        "checked_against": checked_against,
        "max_abs_dlog": round(worst[0], 6),
        "worst_cell": worst[1],
        "cells_appeared_or_gone": missing,
        "converged": converged,
        "method": "re-solved with the prior and r_lookup now on disk; "
                  + ("re-solve KEPT (--commit-convergence: this is the next "
                     "turn's --fit-calibration)" if commit else
                     "artifact restored (dry run)"),
        "verdict": (
            "CONVERGED -- one more iteration reproduces the factors within "
            "tolerance"
            if converged else
            "NOT CONVERGED -- the factors move {:.1%} (> {:.1%}) under the "
            "current prior/r{}. Run --fit-calibration, estimate_prior and "
            "fit_dispersion once more, then re-check.{}".format(
                worst[0], tol,
                f", worst cell on {worst_rows:,} anchor rows in the frozen "
                "anchor fit" if worst_rows else "",
                (" Trajectory " + " -> ".join(f"{h:.4f}" for h in history)
                 + (" is contracting: keep going, several turns is normal."
                    if len(history) > 1 and history[-1] < history[-2] else
                    " is NOT contracting -- investigate before iterating again."))
                if len(history) > 1 else
                " Several turns from a bare chain is normal.")),
    }
    keep = new if commit else old
    keep["convergence"] = block
    with open(path, "w") as f:
        json.dump(keep, f, indent=2)
    return block


def describe_calibration(art, factors, widest_n=12):
    """The --fit-calibration console summary of a calibration artifact."""
    detail = art["detail"]
    lines = [f"grain: {art['grain']}  ({len(factors)} cells, global factor "
             f"{art['global_factor']:.4f}"
             + (f" AT {art['global_factor_at_bound'].upper()} BOUND -- not "
                "a solve" if art.get("global_factor_at_bound") else "")
             + ")",
             f"fit window: {art['fit_window']} "
             f"{art['fit_window_dates'][0]}..{art['fit_window_dates'][1]} "
             f"({art['fit_rows']:,} rows, basis {art['fit_basis']})"]
    if art["fit_in_sample_share"] > 0.5:
        lines.append(f"WARNING: {art['fit_in_sample_share']:.0%} of the fit "
                     "window is inside the training period -- the factor will "
                     "understate what launch-adjacent weeks need.")
    widest = sorted(factors.items(), key=lambda kv: -abs(kv[1] - 1.0))[:widest_n]
    for key, factor in widest:
        info = detail[key]
        if info.get("held_at_parent"):
            # no anchor row in the window: nothing was solved for this cell
            lines.append(f"  {key:26s} {factor:.4f}  (held at its parent's "
                         f"{info['parent_factor']:.4f} -- no anchor row)")
            continue
        lines.append(f"  {key:26s} {factor:.4f}  (raw {info['raw_factor']:.4f} "
                     f"-> parent {info['parent_factor']:.4f}, self-weight "
                     f"{info['shrinkage_weight_on_self']:.2f}, "
                     f"{info['anchor_rows']:,} rows"
                     + (f", AT {info['at_bound'].upper()} BOUND"
                        if info.get("at_bound") else "") + ")")
    if len(factors) > len(widest):
        lines.append(f"  ... {len(factors) - len(widest)} more cells nearer 1.0")
    pinned = sorted(pinned_cells(detail))
    if pinned:
        lines.append(f"{len(pinned)} cell(s) pinned at "
                     "calibration_factor_search_bounds -- a bound is not a "
                     "solve: " + ", ".join(pinned))
    parents = sorted(pinned_cells(art.get("detail_category") or {}))
    if parents:
        under = sorted(k for k, v in detail.items() if v.get("parent_at_bound"))
        lines.append(f"{len(parents)} CATEGORY solve(s) pinned -- the parent "
                     f"{len(under)} subcategory cell(s) shrink toward is a "
                     "bound: " + ", ".join(parents))
    weekly = (art.get("schedule") or {}).get("pinned_by_week") or {}
    if weekly:
        lines.append(f"{len(weekly)} schedule week(s) with a pinned cell "
                     "(schedule.pinned_by_week)")
    below = [k for k, v in factors.items() if v < 1.0]
    if below:
        lines.append(f"{len(below)}/{len(factors)} cells below 1.0 (model "
                     "over-predicts there) -- investigate (AGENTS rule 5)")
    return "\n".join(lines)


# the name the CLI called the summary by
_describe_calibration = describe_calibration
