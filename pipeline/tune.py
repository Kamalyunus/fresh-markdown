"""pipeline.tune -- read the reports, decide the config, record the reasoning.

    python3 -m pipeline.tune            # report only, changes nothing
    python3 -m pipeline.tune --apply    # paste the MEASURED values, back up
                                        # config.yaml, write the decision log

Each check names the report field it reads; the class decides who may act:

  PASTE  a MEASURED value the pipeline computed. Auto-applied by --apply.
  OWNER  a SET BY OWNER value. NEVER auto-applied (AGENTS rule: never invent
         one) -- reported with the evidence the owner needs, and nothing else.
  INFO   a reading with no config key: the bottleneck, the cadence question.
  BLOCK  an invariant that must hold before any of the above means anything --
         a stale report, an unsettled fixed point, a split that violates its
         own sizing rule.

BLOCKs are checked first and suppress the rest: tuning against a report that
graded a different model is worse than not tuning at all (hard rule 1).
"""

import argparse
import datetime as dt
import os
import re
import shutil

import numpy as np
import pandas as pd

from common.config import artifact_mirror_drift, load_config
from common.guardrail import verdict_is_blocking, verdict_is_insufficient
from common.io import read_json, write_json
from pricing.explore import tau_provenance_error

PASTE, OWNER, INFO, BLOCK = "PASTE", "OWNER", "INFO", "BLOCK"
OK, ACT = "OK", "ACT"

# Every config key --apply may touch, one row each.
#   anchor   -> the unique line anchor that carries its scalar. Targeted line
#               edits rather than a YAML round-trip: every value in config.yaml
#               carries the reasoning for it in a comment, and a round-trip
#               would drop them all.
#   measured -> the value is DERIVED from a report rather than chosen by the
#               owner. A number here that disagrees with its report is stale
#               or FOREIGN -- from another run, or from the repo's synthetic
#               fixture -- never a preference. pipeline.status refuses the
#               chain on any of them, whatever class the finding ended up in:
#               a W the split rule downgrades to OWNER is still a value nobody
#               chose.
#   rerun    -> what a paste invalidates: "none" = read at runtime or mirrors
#               the artifact; "calibration" = the loop turns (3b-5b, NO
#               retrain); "retrain" = only data.split, which is SET BY OWNER
#               and never auto-applied.
KEYS = {
    ("learning", "information_increment"):
        {"anchor": "  information_increment:", "measured": True, "rerun": "none"},
    ("learning", "max_mean_step"):
        {"anchor": "  max_mean_step:", "measured": False, "rerun": "none"},
    ("learning", "max_std_shrink"):
        {"anchor": "  max_std_shrink:", "measured": False, "rerun": "none"},
    ("exploration", "tau_initial"):
        {"anchor": "  tau_initial:", "measured": True, "rerun": "none"},
    ("exploration", "delta_min_log_bias"):
        {"anchor": "  delta_min_log_bias:", "measured": True, "rerun": "shadow"},
    ("dispersion", "rho"):
        {"anchor": "  rho:", "measured": True, "rerun": "none"},
    ("baseline_model", "calibration_fit_trailing_weeks"):
        {"anchor": "  calibration_fit_trailing_weeks:", "measured": True,
         "rerun": "calibration"},
    ("monitoring", "stop_conditions", "scrap_deterioration_pct"):
        {"anchor": "    scrap_deterioration_pct:", "measured": False,
         "rerun": "thresholds"},
    ("monitoring", "stop_conditions", "margin_deterioration_pct"):
        {"anchor": "    margin_deterioration_pct:", "measured": False,
         "rerun": "thresholds"},
    ("ab_test", "min_detectable_effect_pct"):
        {"anchor": "  min_detectable_effect_pct:", "measured": False,
         "rerun": "none"},
    ("baseline_model", "calibration_gate_band"):
        {"anchor": "  calibration_gate_band:", "measured": True, "rerun": "none"},
}
MEASURED_KEYS = {k for k, v in KEYS.items() if v["measured"]}
RERUN = {k: v["rerun"] for k, v in KEYS.items() if v["rerun"] != "none"}

# what a config change invalidates, weakest to strongest. A MEASURED paste
# writes back what a report measured -- it invalidates NOTHING unless the
# key is an INPUT to a report (W turns the loop; delta_min changes shadow's
# spreads and tau; a stop threshold changes derive_thresholds' verdict).
# Re-grading every report on every paste chased the fixed point for a day.
RERUN_ORDER = ["none", "thresholds", "shadow", "calibration", "retrain"]

# config keys no report reads: runtime-only, paths, the driver's own knobs
INERT_PREFIXES = ("meta.", "events.", "artifacts.", "tuning.", "assurance.",
                  "data.launch_date", "data.split_manifest_path",
                  "ab_test.active", "learning.update_cadence_days",
                  "monitoring.alert_posterior_std_flat_days",
                  "posterior.path", "posterior.prior.path",
                  "baseline_model.model_path", "baseline_model.feature_schema_path",
                  "baseline_model.calibration_factor_path",
                  "dispersion.r_lookup_path", "dispersion.rho_path")


def rerun_for(keys):
    """The strongest re-run a set of moved dotted config keys demands. A key
    tune owns says so itself; an inert key says none; anything else is an
    edit nobody classified, so every report is re-graded (calibration)."""
    need = "none"
    for key in keys:
        entry = KEYS.get(tuple(key.split(".")))
        if entry is not None:
            r = entry["rerun"]
        elif key.startswith(INERT_PREFIXES):
            r = "none"
        else:
            r = "calibration"
        if RERUN_ORDER.index(r) > RERUN_ORDER.index(need):
            need = r
    return need


RERUN_STEPS = {
    "none": ("nothing to re-run: every value written is read at runtime or "
             "mirrors an artifact that already holds it"),
    "thresholds": ("a stop threshold moved -- re-derive its verdict:\n"
                   "    python3 -m bootstrap.derive_thresholds --input "
                   "data/prepared.parquet"),
    "shadow": ("shadow's own inputs moved (the forced-move floor) -- re-run "
               "pipeline.shadow so its tau derivation and gate are graded "
               "under the value in force"),
    "calibration": (
        "the calibration loop turned -- settle it WITHOUT retraining:\n"
        "    python3 -m bootstrap.run --check-only\n"
        "  (iterates 3b -> 4 -> 5 -> 5b to CONVERGED and refreshes the "
        "reports), then re-run pipeline.shadow"),
    "retrain": ("the model's own training data changed -- a full "
                "`python3 -m bootstrap.run --input <raw>` is required; "
                "nothing from before is comparable (rule 1)"),
}

def _sweep_of(backtest):
    """The sweep block, or {} -- it is a STRING on its NOT RUN path, and
    `.get` on that took tune down with a traceback instead of a finding."""
    s = ((backtest or {}).get("fidelity") or {}).get("calibration_window_sweep")
    return s if isinstance(s, dict) else {}


def _get(cfg, path):
    node = cfg
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _finding(key, cls, status, current, recommended, evidence, source):
    return {"key": ".".join(key) if isinstance(key, tuple) else key,
            "class": cls, "status": status, "current": current,
            "recommended": recommended, "evidence": evidence, "source": source}


def _close(a, b, rel=1e-3):
    if a is None or b is None:
        return False
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b)
                and all(_close(a[k], b[k], rel) for k in b))
    try:
        return abs(float(a) - float(b)) <= rel * max(abs(float(b)), 1e-9)
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------- the checks

def _blocks(cfg, backtest, shadow, calibration):
    """Invariants that make every other reading meaningful. Checked first."""
    out = []

    # 1. one model behind every report (hard rule 1)
    versions = {}
    for name, rep in (("backtest", backtest), ("shadow", shadow)):
        v = ((rep or {}).get("artifact_versions") or {}).get(
            "baseline_model_version")
        if v:
            versions[name] = v
    bundle = (calibration or {}).get("provenance", {}).get("bundle")
    if len(set(versions.values())) > 1:
        out.append(_finding("reports agree on one model", BLOCK, ACT,
                            versions, "one version",
                            "backtest and shadow graded DIFFERENT models; no "
                            "comparison between them is valid (rule 1)",
                            "artifact_versions.baseline_model_version"))
    elif versions and bundle and set(versions.values()) != {bundle}:
        out.append(_finding("reports match the artifacts", BLOCK, ACT,
                            versions, bundle,
                            "the reports grade a model that is not the one on "
                            "disk -- re-run them before reading any number",
                            "artifact_versions vs calibration provenance"))

    # 2. the calibration <-> dispersion fixed point settled, and still holds
    conv = (calibration or {}).get("convergence")
    if not conv:
        out.append(_finding("calibration convergence", BLOCK, ACT,
                            "never checked", "CONVERGED",
                            "the f<->r loop is assumed, not asserted; every "
                            "artifact below depends on how many iterations "
                            "happened to run",
                            "train_baseline --check-convergence"))
    elif not conv.get("converged"):
        out.append(_finding("calibration convergence", BLOCK, ACT,
                            conv.get("max_abs_dlog"), f"<= {conv.get('tol_log')}",
                            "NOT CONVERGED: run 3b -> 4 -> 5 -> 5b again "
                            "before tuning anything",
                            "calibration.json convergence"))

    # 3. the split obeys its own sizing rule (design 6): calib >= 2W, or the
    #    anchor's first W weeks carry train-contaminated factors
    s = cfg["data"]["split"]
    w = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
    calib_weeks = ((pd.Timestamp(s["calib_end"]) - pd.Timestamp(s["calib_start"])
                    ).days + 1) / 7.0
    if calib_weeks < 2 * w:
        out.append(_finding("data.split", BLOCK, ACT,
                            f"calib {calib_weeks:.1f}w, W={w}", f"calib >= {2*w}w",
                            f"calib is {calib_weeks:.1f} weeks against W={w}: "
                            "the anchor's first W weeks carry factors fit "
                            "partly on train rows, where the model fits by "
                            "construction and biases them low (design 6)",
                            "config data.split vs calibration_fit_trailing_weeks"))

    # 4. no graded window straddles the exclusion gap
    ex = cfg["data"].get("exclusion_window") or {}
    if ex.get("start"):
        lo, hi = pd.Timestamp(ex["start"]), pd.Timestamp(ex["end"])
        for name, a, b in (("calib", s["calib_start"], s["calib_end"]),
                           ("test", s["test_start"], s["test_end"])):
            if pd.Timestamp(a) <= hi and pd.Timestamp(b) >= lo:
                out.append(_finding(f"data.split.{name}", BLOCK, ACT,
                                    f"{a}..{b}", "clear of the exclusion gap",
                                    f"{name} spans the exclusion window "
                                    f"{ex['start']}..{ex['end']}, so it is not "
                                    "the duration its label claims",
                                    "config data.exclusion_window"))
    return out


def _measured(cfg, shadow, rho_art, thresholds, backtest=None):
    """MEASURED values the pipeline computed and config must mirror."""
    out = []

    inc_rec = (thresholds or {}).get("information_increment_recommendation") or {}
    rec = inc_rec.get("recommended")
    cur = _get(cfg, ("learning", "information_increment"))
    if rec is not None:
        out.append(_finding(
            ("learning", "information_increment"), PASTE,
            OK if _close(cur, rec) else ACT, cur, rec,
            inc_rec.get("verdict") or
            f"I* at the launch prior std; configured {cur}",
            "thresholds.information_increment_recommendation.recommended"))
    elif str(inc_rec.get("verdict") or "").startswith("NOT RUN"):
        # the value in force was measured on SOME earlier run (or is the
        # fixture's); this run could not measure it. Silence here read as
        # "matches", and status stayed green on an unverified key.
        out.append(_finding(
            ("learning", "information_increment"), PASTE, ACT, cur, None,
            f"{inc_rec['verdict']} -- the configured value is unverified "
            "by this run; --apply cannot paste a value that was not measured",
            "thresholds.information_increment_recommendation.verdict"))

    for key, art_key in ((("dispersion", "rho"), "rho"),):
        got = (rho_art or {}).get(art_key)
        cur = _get(cfg, key)
        if got is not None:
            # the SAME rule pipeline.status and load_config(strict=True)
            # enforce (common.config.artifact_mirror_drift). tune's own 0.1%
            # relative check reported OK on a paste those two refuse, so
            # --apply wrote nothing and status stayed red.
            stale = any(art_key in d for d in artifact_mirror_drift(cfg))
            out.append(_finding(
                key, PASTE, ACT if (cur is None or stale) else OK, cur, got,
                "config mirrors the frozen artifact; a stale paste "
                "mis-weights every posterior step, silently",
                f"artifacts/rho.json {art_key}"))


    tau = ((shadow or {}).get("tau_initial_derivation") or {}).get("tau_initial")
    cur = _get(cfg, ("exploration", "tau_initial"))
    if tau is not None:
        # ONE definition of "is this paste current": the same gate
        # pipeline.status enforces. A looser tolerance here reports OK,
        # --apply writes nothing, and status still FAILs -- tune runs clean
        # against a red status and the only escape is a hand edit.
        stale = tau_provenance_error(cfg, backtest, shadow)
        out.append(_finding(
            ("exploration", "tau_initial"), PASTE,
            ACT if (cur is None or stale) else OK, cur, tau,
            stale or ("matches shadow's own derivation on the trailing "
                      "pre-window week -- the gate pipeline.status enforces"),
            "shadow.tau_initial_derivation.tau_initial"))
    # delta_min's bias scale: the LARGEST of three level-error readings,
    # each of which understates alone (design 5.8)
    fid = (backtest or {}).get("fidelity") or {}
    sweep = _sweep_of(backtest)
    chosen = f"trailing_{cfg['baseline_model']['calibration_fit_trailing_weeks']}w"
    mae = ((sweep.get(chosen) or {}).get("mean_abs_log_error")
           if isinstance(sweep.get(chosen), dict) else None)
    bc = {str(k).replace(" ", "_"): float(v) for k, v in
          (fid.get("by_category") or {}).items() if v and v > 0}
    logs = np.log(list(bc.values())) if bc else np.array([])
    rms = float(np.sqrt((logs ** 2).mean())) if len(logs) else None
    band = cfg["baseline_model"]["calibration_gate_band"]
    half = float(np.log(band[1])) if band else None
    # catalogue-wide floor: the aggregate noise (MAE at W) and the accepted
    # tolerance; no category's floor may sit below either
    floor = max(v for v in (mae, half) if v) if (mae or half) else None
    cur = _get(cfg, ("exploration", "delta_min_log_bias"))
    if floor is not None:
        # PER CATEGORY: its own surviving level error, floored; `_default`
        # (the old scalar: largest of floor and the rms) for a category the
        # backtest never saw
        rec = {c: round(max(abs(float(np.log(v))), floor), 4) for c, v in bc.items()}
        rec["_default"] = round(max(floor, rms or 0.0), 4)
        out.append(_finding(
            ("exploration", "delta_min_log_bias"), PASTE,
            OK if _close(cur, rec) else ACT, cur, rec,
            f"per category: max(|log by_category ratio|, floor) with floor="
            f"max(mean_abs_log_error@W={mae}, gate_half_width={half:.4f}); "
            f"_default=max(floor, by_category rms {rms:.4f})"
            if rms is not None else
            f"floor only (no by_category in the backtest): {floor:.4f}"
            " -- a forced move must beat this level error to say anything "
            "about eps (delta_min = multiple x bias / |eps| per cell)",
            f"backtest.fidelity.calibration_window_sweep.{chosen}, "
            "backtest.fidelity.by_category, config calibration_gate_band"))
    return out


def _derived(cfg, backtest, thresholds):
    """Measured values that auto-apply; a failing gate downgrades the
    finding to OWNER with the reason."""
    out = []

    # the rail, gated on its measured price consequence
    bs = (thresholds or {}).get("bounded_step_recommendation") or {}
    cur = _get(cfg, ("learning", "max_mean_step"))
    rec = bs.get("consistent_max_mean_step")
    if rec is None and str(bs.get("verdict") or "").startswith("NOT RUN"):
        out.append(_finding(
            ("learning", "max_mean_step"), OWNER, ACT, cur, None,
            f"{bs['verdict']} -- the rail in force is unverified by this run",
            "thresholds.bounded_step_recommendation.verdict"))
    if rec is not None:
        ss = (((backtest or {}).get("policy_deltas") or {})
              .get("step_sensitivity") or {})
        arms = [a for a in (ss.get("deeper_belief"), ss.get("shallower_belief"))
                if isinstance(a, dict)]
        share = max((a.get("share_prices_changed") or 0) for a in arms) if arms else None
        il = max((abs(a.get("il_delta_pct") or 0)) for a in arms) if arms else None
        g = cfg.get("tuning") or {}
        max_share = g.get("max_price_share_changed_for_auto_rail")
        max_il = g.get("max_il_delta_pct_for_auto_rail")
        cost = "; ".join(
            f"{k.split('_')[0]} {a.get('share_prices_changed', 0):.1%} of prices, "
            f"IL {a.get('il_delta_pct', 0):+.4%}"
            for k, a in (("deeper_belief", ss.get("deeper_belief") or {}),
                         ("shallower_belief", ss.get("shallower_belief") or {}))
            if a)
        gated = (share is None or il is None
                 or share > max_share or il > max_il)
        why = (f"price consequence at the current step: {cost}"
               if cost else "no step_sensitivity in the backtest report")
        out.append(_finding(
            ("learning", "max_mean_step"), OWNER if gated else PASTE,
            OK if _close(cur, rec, 1e-2) else ACT, cur, rec,
            bs.get("verdict", "").split(". OWNER")[0] + " · " + why
            + (f" -- EXCEEDS the auto-apply gate ({max_share:.0%} of prices, "
               f"{max_il:.1%} IL), so this rail change is a real price event "
               "and stays an owner decision" if gated and cost else
               "" if gated else " -- inside the auto-apply gate"),
            "thresholds.bounded_step_recommendation + backtest.policy_deltas."
            "step_sensitivity"))

    # guardrail stops: the floor IS the measurement -- 3-sigma of the control
    # arm's own noise. Setting them at it is what the report recommends.
    gr = (thresholds or {}).get("guardrail_threshold_recommendation") or {}
    for name, block in gr.items():
        if not isinstance(block, dict):
            continue
        path = tuple(block.get("config_key", "").split("."))
        if not path or not path[0]:
            continue
        cur = _get(cfg, path)
        floor = block.get("binding_floor")
        if floor is None:
            if verdict_is_insufficient(block.get("verdict")):
                out.append(_finding(
                    path, INFO, OK, cur, None,
                    f"{block.get('verdict')} -- no floor to check "
                    f"{'the configured stop' if cur is not None else 'a stop'} "
                    "against yet; status reads this as WARN",
                    f"thresholds.guardrail_threshold_recommendation.{name}.verdict"))
            continue
        # `binding_floor` is set BEFORE the unusability check, so a BLOCKED /
        # TOO TIGHT / LIKELY INERT block still carries a number. Pasting it
        # writes a threshold the report itself calls unusable. Same test the
        # status row uses -- one definition, in common.guardrail.
        if verdict_is_blocking(block.get("verdict")):
            out.append(_finding(
                path, OWNER, ACT, cur, None,
                f"{block.get('verdict')} -- NOT pasted. binding_floor "
                f"({floor}) exists but the block is unusable, so no value "
                "here is both safe and useful; fix the basis or the metric, "
                "not this key",
                f"thresholds.guardrail_threshold_recommendation.{name}.verdict"))
            continue
        ok = cur is not None and float(cur) >= float(floor)
        out.append(_finding(
            path, PASTE, OK if ok else ACT, cur, floor,
            f"{block.get('binding_label', '3-sigma')} "
            f"{block.get('binding_basis', 'control_arm')} noise floor -- a "
            "stop below it fires on noise. Raise it later if the pilot wants "
            "a looser trip, but never below this",
            f"thresholds.guardrail_threshold_recommendation.{name}"))

    # the level band: sized from measured anchor volatility, TIGHTEN ONLY
    g = cfg.get("tuning") or {}
    sweep = _sweep_of(backtest)
    chosen = f"trailing_{cfg['baseline_model']['calibration_fit_trailing_weeks']}w"
    mae = ((sweep.get(chosen) or {}).get("mean_abs_log_error")
           if isinstance(sweep.get(chosen), dict) else None)
    cap = g.get("calibration_band_max_half_width")
    if mae and cap:
        k = g.get("calibration_band_sigma_multiple", 3)
        half = k * float(mae)
        # clamp in RATIO space (exp(0.10)=1.1052 would breach a log clamp)
        cap = float(cap)
        band = [round(max(1 - cap, float(np.exp(-half))), 4),
                round(min(1 + cap, float(np.exp(half))), 4)]
        cur_band = list(cfg["baseline_model"]["calibration_gate_band"])
        clamped = band == [round(1 - cap, 4), round(1 + cap, 4)]
        out.append(_finding(
            ("baseline_model", "calibration_gate_band"), PASTE,
            OK if [round(x, 4) for x in cur_band] == band else ACT,
            cur_band, band,
            f"{k}x the rolling-origin mean_abs_log_error of {chosen} "
            f"({mae}) = half-width {k * float(mae):.4f} in log space"
            + (f", CLAMPED to the {cap} ceiling -- the band may only tighten"
               if clamped else
               " -- inside the ceiling, so the measured volatility sets it"),
            f"backtest.fidelity.calibration_window_sweep.{chosen}."
            "mean_abs_log_error"))
    return out


def _business(cfg, thresholds):
    """Tolerances data cannot decide; reported with evidence, never
    written by --apply."""
    out = []

    # which rail moves is a safety posture; the tool supplies both numbers
    bs = (thresholds or {}).get("bounded_step_recommendation") or {}
    std0 = bs.get("median_launch_std")
    step = _get(cfg, ("learning", "max_mean_step"))
    cur = _get(cfg, ("learning", "max_std_shrink"))
    if std0 and step and float(std0) > 0:
        frac = float(step) / float(std0)
        alt = (round(1.0 - (1.0 - frac) ** 0.5, 4) if frac < 1 else None)
        out.append(_finding(
            ("learning", "max_std_shrink"), OWNER,
            OK if _close(cur, alt, 5e-2) else ACT, cur, alt,
            (f"the rails disagree: max_mean_step {step} against a consistent "
             f"{bs.get('consistent_max_mean_step')}. Two ways to fix it, and "
             f"the choice is a SAFETY POSTURE, not a reading -- raise "
             f"max_mean_step to {bs.get('consistent_max_mean_step')} (prices "
             f"move faster) or lower max_std_shrink to {alt} (the system "
             f"becomes confident more slowly). Suggestion only; nothing is "
             f"written")
            if alt is not None else
            "max_mean_step exceeds the launch prior std, so no shrink value "
            "makes the rails agree -- the step itself is the thing to revisit",
            "thresholds.bounded_step_recommendation.median_launch_std"))
    ab = (thresholds or {}).get("ab_duration") or {}
    by = ab.get("by_duration") or {}
    cur = _get(cfg, ("ab_test", "min_detectable_effect_pct"))
    if by:
        # the achievable FRONTIER, not the --mde flag echoed back
        achievable = sorted(
            ((v.get("detectable_mde_rel"), k) for k, v in by.items()
             if v.get("detectable_mde_rel") is not None))
        best, best_dur = achievable[0] if achievable else (None, None)
        target = ab.get("target_mde_rel")
        passing = [k for k, v in by.items()
                   if str(v.get("meets_target")).lower() == "true"]
        frontier = ", ".join(f"{d} -> {m}" for m, d in achievable[:5])
        out.append(_finding(
            ("ab_test", "min_detectable_effect_pct"), OWNER,
            OK if cur is not None else ACT, cur, best,
            (f"detectable by duration: {frontier}. "
             + (f"The {target} target is met by {', '.join(sorted(passing))}."
                if passing else
                f"NO duration reaches the {target} target -- the best the "
                f"extract can resolve is {best} at {best_dur}.")
             + " NOT measurable: the data says what is DETECTABLE, never what "
               "size of effect is worth acting on. Set the smallest effect the "
               "business would act on; if that is below the frontier above, "
               "this A/B cannot resolve it at any duration and the honest "
               "options are a longer pilot, more traffic, or a different "
               "question"),
            "thresholds.ab_duration.by_duration"))
    return out


def _readings(cfg, backtest, shadow):
    """Findings with no config key: the ones that decide what to DO next."""
    out = []

    # calibration cadence -- the weekly cron is worth running only if it beats
    # the frozen anchor on the hold-out. Measured, not assumed.
    cr = (shadow or {}).get("calibration_regimes") or {}
    frozen, weekly = cr.get("frozen_anchor"), cr.get("weekly_refit")
    if cr.get("refit_error"):
        out.append(_finding(
            "calibration cadence", INFO, ACT, frozen, None,
            f"the weekly re-fit RAISED in shadow ({cr['refit_error']}) -- "
            "the cadence question is unanswered, not answered 'frozen'",
            "shadow.calibration_regimes.refit_error"))
    if frozen is not None and weekly is not None:
        better = "weekly re-fit" if abs(weekly - 1) < abs(frozen - 1) else "frozen anchor"
        out.append(_finding(
            "calibration cadence", INFO,
            ACT if better.startswith("weekly") else OK, frozen, weekly,
            f"frozen {frozen} vs weekly re-fit {weekly} on the hold-out -- "
            f"{better} is closer to 1.0. Run the weekly --fit-calibration cron "
            f"in production ONLY if the weekly reading wins here",
            "shadow.calibration_regimes"))

    # which constraint actually limits learning: evidence or the daily gate
    ly = (shadow or {}).get("learning_yield_would_be") or {}
    per_update = ly.get("episodes_per_bounded_update")
    win = (shadow or {}).get("window") or {}
    days = win.get("days")
    if days is None and win.get("date_min") and win.get("date_max"):
        days = (pd.Timestamp(win["date_max"])
                - pd.Timestamp(win["date_min"])).days + 1
    # the POPULATION's episodes/day: a --max-episodes sample understates
    # the daily evidence rate and flips the reading to EVIDENCE
    episodes = (win.get("population_episodes") or win.get("episodes")
                or (shadow or {}).get("episodes"))
    if per_update and days and episodes:
        per_day = episodes / days
        evidence_days = per_update / per_day if per_day else None
        floor = ly.get("calendar_floor_days_per_0.15_of_mean",
                       cfg["learning"]["update_cadence_days"])
        binds = "CALENDAR" if (evidence_days or 0) < floor else "EVIDENCE"
        out.append(_finding(
            "learning bottleneck", INFO, OK, binds, None,
            f"{per_update:,.0f} episodes per bounded update at "
            f"{per_day:,.0f} episodes/day = {evidence_days:.2f} days of "
            f"evidence per update, against a {floor}-day update cadence -> "
            f"{binds} BINDS. "
            + (f"Each {floor}-day period brings "
               f"{ly.get('bounded_updates_worth_per_period')} bounded "
               "updates' worth of evidence and one update can absorb one: "
               "the rest is discarded. Widening the forced discount gap buys "
               "nothing; raise max_std_shrink (then max_mean_step), or "
               "shorten learning.update_cadence_days."
               if binds == "CALENDAR" else
               "Evidence is the limit: wider forced arms (information is "
               "QUADRATIC in the log price ratio) before anything else."),
            "shadow.learning_yield_would_be"))

    # the fit window W the rolling-origin sweep prefers, bounded by calib >= 2W
    sweep = _sweep_of(backtest)
    rec = sweep.get("recommended_fit_window")
    if rec:
        want = int(re.sub(r"\D", "", rec) or 0)
        cur = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
        s = cfg["data"]["split"]
        calib_weeks = ((pd.Timestamp(s["calib_end"])
                        - pd.Timestamp(s["calib_start"])).days + 1) / 7.0
        feasible = want and calib_weeks >= 2 * want
        # HYSTERESIS. W is the only paste that turns the calibration loop,
        # and --check-only re-settling the loop moves the sweep's own scores
        # -- so a strict argmin flips between near-tied windows and sends an
        # agent into apply -> check-only forever. Switch only when the
        # candidate beats the CURRENT window materially on the sweep's own
        # metrics; a near-tie holds the value in force.
        held = False
        cur_e, want_e = sweep.get(f"trailing_{cur}w"), sweep.get(rec)
        if (want != cur and isinstance(cur_e, dict)
                and isinstance(want_e, dict)):
            better_mae = (want_e.get("mean_abs_log_error", 9e9)
                          < 0.9 * cur_e.get("mean_abs_log_error", 0))
            better_band = (want_e.get("share_weeks_in_band", 0)
                           >= cur_e.get("share_weeks_in_band", 1) + 0.05)
            held = not (better_mae or better_band)
        out.append(_finding(
            ("baseline_model", "calibration_fit_trailing_weeks"),
            PASTE if feasible else OWNER,
            OK if (want == cur or held) else ACT, cur,
            cur if held else want,
            (f"the sweep prefers {rec} but within noise of the current "
             f"{cur}w (mae {want_e.get('mean_abs_log_error')} vs "
             f"{cur_e.get('mean_abs_log_error')}, in-band "
             f"{want_e.get('share_weeks_in_band')} vs "
             f"{cur_e.get('share_weeks_in_band')}) -- HELD. A W change turns "
             "the calibration loop and re-scores the sweep, so near-ties "
             "oscillate; it switches only on a material win"
             if held else
             f"the rolling-origin sweep prefers {rec}"
             + (f"; calib is {calib_weeks:.1f}w so calib >= 2W holds"
                if feasible else
                f", but calib is {calib_weeks:.1f}w and the rule is calib >= "
                f"2W ({2*want}w needed) -- move the SPLIT, not this key")),
            "backtest.fidelity.calibration_window_sweep.recommended_fit_window"))

    # no-factors beating every window is not a W to paste -- it says the
    # factors are adding noise, which is an owner call on the design, not a
    # key. Reported whatever W is in force.
    if sweep.get("uncalibrated_beats_all_windows"):
        unc, cal = sweep.get("uncalibrated") or {}, sweep.get(rec) or {}
        out.append(_finding(
            "level calibration earns its keep", INFO, ACT,
            "factors applied", "compare against no factors",
            f"on the same eval weeks, uncalibrated scores mae "
            f"{unc.get('mean_abs_log_error')} / in-band "
            f"{unc.get('share_weeks_in_band')} against {rec}'s "
            f"{cal.get('mean_abs_log_error')} / "
            f"{cal.get('share_weeks_in_band')} -- the factors lose on both. "
            "A model already at the anchor level has no bias for them to "
            "remove, so they contribute estimation noise. Check "
            "fidelity.by_category before acting: factors that are near 1 "
            "everywhere are inert, factors that scatter are being fit on too "
            "little data",
            "backtest.fidelity.calibration_window_sweep.verdict"))
    return out


def collect(cfg, root="reports"):
    backtest = read_json(os.path.join(root, "backtest.json"))
    shadow = read_json(os.path.join(root, "shadow.json"))
    thresholds = read_json(os.path.join(root, "thresholds.json"))
    calibration = read_json(cfg["baseline_model"]["calibration_factor_path"])
    rho_art = read_json(cfg["dispersion"]["rho_path"])

    missing = [n for n, r in (("backtest", backtest), ("shadow", shadow),
                              ("thresholds", thresholds))
               if r is None]
    blocks = _blocks(cfg, backtest, shadow, calibration)
    if missing:
        blocks.insert(0, _finding(
            "reports present", BLOCK, ACT, f"missing: {', '.join(missing)}",
            "all three", "run the pipeline before tuning: a missing report is "
            "not a passing check", "reports/"))

    findings = list(blocks)
    if not blocks:
        findings += _measured(cfg, shadow, rho_art, thresholds, backtest)
        findings += _derived(cfg, backtest, thresholds)
        findings += _business(cfg, thresholds)
        findings += _readings(cfg, backtest, shadow)
    return {"findings": findings,
            "blocked": bool(blocks),
            "to_paste": [f for f in findings
                         if f["class"] == PASTE and f["status"] == ACT],
            "owner_decisions": [f for f in findings
                                if f["class"] == OWNER and f["status"] == ACT]}


# --------------------------------------------------------------- applying

def set_scalar(text, path, value):
    """Replace one scalar in config.yaml, keeping its comment and every
    other line byte-identical (a YAML round-trip drops comments)."""
    anchor = (KEYS.get(tuple(path)) or {}).get("anchor")
    if anchor is None:
        raise KeyError(f"no line anchor for {'.'.join(path)}")
    hits = [i for i, ln in enumerate(text.splitlines()) if ln.startswith(anchor)]
    if len(hits) != 1:
        raise RuntimeError(
            f"anchor {anchor!r} matched {len(hits)} lines -- refusing to guess")
    lines = text.splitlines(keepends=True)
    i = hits[0]
    line = lines[i]
    head, _, tail = line.partition(":")
    comment = ""
    if "#" in tail:
        comment = "  " + tail[tail.index("#"):].rstrip("\n")
    if isinstance(value, dict):
        # a per-category mapping pastes as a one-line YAML flow mapping, so
        # the anchor still owns exactly one line
        value = "{" + ", ".join(f"{k}: {v}" for k, v in value.items()) + "}"
    lines[i] = f"{head}: {value}{comment}\n"
    return "".join(lines)


def apply(report, config_path="config.yaml", out_dir="artifacts"):
    """Paste the MEASURED values, back up the config, write the decision
    log. OWNER values are never touched."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(out_dir, exist_ok=True)
    backup = os.path.join(out_dir, f"config_backup_{stamp}.yaml")
    shutil.copyfile(config_path, backup)

    with open(config_path) as f:
        text = f.read()
    applied, failed = [], []
    for f_ in report["to_paste"]:
        path = tuple(f_["key"].split("."))
        if f_["recommended"] is None:
            failed.append(dict(f_, error="the report carries no value to "
                                         "paste (NOT RUN) -- re-run it"))
            continue
        try:
            text = set_scalar(text, path, f_["recommended"])
            applied.append(f_)
        except (KeyError, RuntimeError) as exc:               # noqa: PERF203
            failed.append(dict(f_, error=str(exc)))
    if applied:
        with open(config_path, "w") as f:
            f.write(text)

    log_path = os.path.join(out_dir, "config_decisions.json")
    history = read_json(log_path) or {"runs": []}
    history["runs"].append({
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_backup": backup,
        "applied": [{k: v for k, v in f_.items() if k != "status"}
                    for f_ in applied],
        "not_applied": failed,
        "pending_owner_decisions": [
            {k: v for k, v in f_.items() if k != "status"}
            for f_ in report["owner_decisions"]],
        "readings": [f_ for f_ in report["findings"] if f_["class"] == INFO],
        "note": ("MEASURED values are pasted from the reports named in each "
                 "`source`. SET BY OWNER values are never written here -- they "
                 "are listed with the evidence the owner needs. Re-run the "
                 "bootstrap after applying: a changed increment or rail "
                 "changes what the next run measures."),
    })
    needed = rerun_for([f_["key"] for f_ in applied])
    history["runs"][-1]["rerun_required"] = needed
    write_json(log_path, history)
    return {"backup": backup, "log": log_path, "applied": applied,
            "failed": failed, "rerun": needed}


# --------------------------------------------------------------- rendering

def render(report):
    lines = []
    if report["blocked"]:
        lines.append("BLOCKED -- fix these before any tuning reading is valid:")
    for f_ in report["findings"]:
        mark = {BLOCK: "BLOCK", PASTE: "PASTE", OWNER: "OWNER",
                INFO: "READ "}[f_["class"]]
        flag = "  " if f_["status"] == OK else "->"
        lines.append(f"{flag} {mark}  {f_['key']}")
        if f_["class"] in (PASTE, OWNER) and f_["status"] == ACT:
            lines.append(f"        {f_['current']}  ->  {f_['recommended']}")
        lines.append(f"        {f_['evidence']}")
        lines.append(f"        source: {f_['source']}")
    if not report["blocked"]:
        n_p, n_o = len(report["to_paste"]), len(report["owner_decisions"])
        lines.append("")
        lines.append(f"{n_p} measured value(s) to paste (--apply writes them), "
                     f"{n_o} owner decision(s) outstanding.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--apply", action="store_true",
                    help="write the MEASURED pastes, back up config.yaml and "
                         "record every decision in artifacts/config_decisions.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = collect(cfg, args.reports)
    print(render(report))

    if args.apply:
        if report["blocked"]:
            raise SystemExit(
                "\nrefusing to apply: the blocking checks above must pass "
                "first. Tuning against a stale or unconverged chain writes "
                "numbers that describe a pipeline nobody ran.")
        res = apply(report, args.config)
        print(f"\nbacked up   {res['backup']}")
        print(f"decisions   {res['log']}")
        for f_ in res["applied"]:
            print(f"  pasted    {f_['key']} = {f_['recommended']}")
        for f_ in res["failed"]:
            print(f"  SKIPPED   {f_['key']}: {f_['error']}")
        if res["applied"]:
            # the MINIMUM sufficient re-run for what was written
            print(f"\n{RERUN_STEPS[res['rerun']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
