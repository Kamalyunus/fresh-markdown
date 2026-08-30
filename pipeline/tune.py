"""pipeline.tune -- read the reports, decide the config, record the reasoning.

    python3 -m pipeline.tune            # report only, changes nothing
    python3 -m pipeline.tune --apply    # paste the MEASURED values, back up
                                        # config.yaml, write the decision log

Every tuning decision this project has made was reached the same way: run the
bootstrap, read a named field in a named report, compare it against what
config.yaml holds, and either paste the measured value or record an owner
decision with the evidence beside it. That loop was being carried between a
human and two agents by copy-paste, which is where numbers get stale -- a run
was analysed at `information_increment: 12.0` after the measured 0.341 had
already been pasted, and nothing noticed.

So the loop lives here. Each check names the report field it reads, so a
disagreement is traceable to a file rather than to someone's memory, and the
class decides who may act:

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
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

from common.config import load_config

PASTE, OWNER, INFO, BLOCK = "PASTE", "OWNER", "INFO", "BLOCK"
OK, ACT = "OK", "ACT"

# WHAT A PASTE ACTUALLY INVALIDATES. Getting this wrong sent an agent into a
# loop: --apply said "re-run the bootstrap", the script RETRAINED THE BASELINE
# (step 3), every artifact moved, the fixed point reset, and the convergence
# check the agent was trying to close went red again -- on values that touch no
# artifact at all. A retrain also breaks hard rule 1 outright.
#
#   "none"        read at runtime, or a mirror of a value the artifact already
#                 holds. Nothing to re-fit; status re-reads config directly.
#   "calibration" changes what --fit-calibration solves, so the loop turns:
#                 3b -> 4 -> 5 -> 5b, then backtest and shadow to refresh the
#                 reports. The BASELINE IS NOT RETRAINED.
#   "retrain"     changes what the model is fitted on. Only data.split does
#                 this, and it is SET BY OWNER, so --apply never writes one.
RERUN = {
    ("baseline_model", "calibration_fit_trailing_weeks"): "calibration",
}

RERUN_STEPS = {
    "none": ("nothing to re-run: every value written is read at runtime or "
             "mirrors an artifact that already holds it"),
    "calibration": (
        "the calibration loop turned -- settle it WITHOUT retraining:\n"
        "    python3 -m bootstrap.run --check-only\n"
        "  (iterates 3b -> 4 -> 5 -> 5b to CONVERGED and refreshes the "
        "reports), then re-run pipeline.shadow"),
    "retrain": ("the model's own training data changed -- a full "
                "full `python3 -m bootstrap.run --input <raw>` is required "
                "(hard rule 1: nothing "
                "from before is comparable)"),
}

# config path -> the unique line anchor that carries its scalar. Targeted line
# edits rather than a YAML round-trip: every value in config.yaml carries the
# reasoning for it in a comment, and a round-trip would drop them all.
ANCHORS = {
    ("learning", "information_increment"): "  information_increment:",
    ("learning", "max_mean_step"): "  max_mean_step:",
    ("learning", "max_std_shrink"): "  max_std_shrink:",
    ("exploration", "tau_initial"): "  tau_initial:",
    ("dispersion", "rho"): "  rho:",
    ("dispersion", "mean_forced_hours_per_episode"):
        "  mean_forced_hours_per_episode:",
    ("baseline_model", "calibration_fit_trailing_weeks"):
        "  calibration_fit_trailing_weeks:",
    ("monitoring", "stop_conditions", "scrap_deterioration_pct"):
        "    scrap_deterioration_pct:",
    ("monitoring", "stop_conditions", "margin_deterioration_pct"):
        "    margin_deterioration_pct:",
    ("ab_test", "min_detectable_effect_pct"): "  min_detectable_effect_pct:",
}


def _read(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


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
    try:
        return abs(float(a) - float(b)) <= rel * max(abs(float(b)), 1e-9)
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------- the checks

def _blocks(cfg, backtest, shadow, calibration, root):
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


def _measured(cfg, shadow, rho_art, thresholds):
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

    for key, art_key in ((("dispersion", "rho"), "rho"),
                         (("dispersion", "mean_forced_hours_per_episode"),
                          "mean_forced_hours_per_episode")):
        got = (rho_art or {}).get(art_key)
        cur = _get(cfg, key)
        if got is not None:
            out.append(_finding(
                key, PASTE, OK if _close(cur, got) else ACT, cur, got,
                "config mirrors the frozen artifact; a stale paste "
                "mis-weights every posterior step, silently",
                f"artifacts/rho.json {art_key}"))


    tau = ((shadow or {}).get("tau_initial_derivation") or {}).get("tau_initial")
    cur = _get(cfg, ("exploration", "tau_initial"))
    if tau is not None:
        out.append(_finding(("exploration", "tau_initial"), PASTE,
                            OK if _close(cur, tau, 1e-2) else ACT, cur, tau,
                            "derived on shadow's own trailing pre-window week "
                            "-- preferred over a backtest bisection",
                            "shadow.tau_initial_derivation.tau_initial"))
    return out


def _derived(cfg, backtest, thresholds):
    """Values the pipeline MEASURES, which used to sit behind a human.

    A value the data can decide should not wait on a decision (owner,
    2026-08-30). Two of these carry a GATE -- a second measurement that must
    agree before the paste is safe -- and a failing gate downgrades the
    finding to OWNER with the reason, rather than hiding it.
    """
    out = []

    # the rail: `consistent_max_mean_step` is derived from the launch prior
    # std so both rails trip at the same surprise instead of one clipping
    # every batch. The threshold note says to read the PRICE consequence
    # first, so that read is the gate rather than a human's recollection.
    bs = (thresholds or {}).get("bounded_step_recommendation") or {}
    cur = _get(cfg, ("learning", "max_mean_step"))
    rec = bs.get("consistent_max_mean_step")
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
            continue
        ok = cur is not None and float(cur) >= float(floor)
        out.append(_finding(
            path, PASTE, OK if ok else ACT, cur, floor,
            f"{block.get('binding_label', '3-sigma')} "
            f"{block.get('binding_basis', 'control_arm')} noise floor -- a "
            "stop below it fires on noise. Raise it later if the pilot wants "
            "a looser trip, but never below this",
            f"thresholds.guardrail_threshold_recommendation.{name}"))

    # the level band: sized from the MEASURED week-to-week anchor volatility,
    # and allowed to TIGHTEN ONLY. A band wider than the current one is a
    # decision about tolerated level error, not a reading (owner, 2026-08-30).
    g = cfg.get("tuning") or {}
    sweep = (((backtest or {}).get("fidelity") or {})
             .get("calibration_window_sweep") or {})
    chosen = f"trailing_{cfg['baseline_model']['calibration_fit_trailing_weeks']}w"
    mae = ((sweep.get(chosen) or {}).get("mean_abs_log_error")
           if isinstance(sweep.get(chosen), dict) else None)
    cap = g.get("calibration_band_max_half_width")
    if mae and cap:
        k = g.get("calibration_band_sigma_multiple", 3)
        half = k * float(mae)
        # clamp in RATIO space, not log: exp(0.10) is 1.1052, so clamping the
        # log half-width would WIDEN the upper edge past the ceiling. Each
        # side is pulled in independently, so the result can never be wider
        # than [1-cap, 1+cap] on either side -- tighten only.
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
    """The values data cannot decide, because they are tolerances, not facts.

    Each one is reported with the number the data DOES supply, so the owner
    decides against evidence rather than against nothing -- but the tool never
    writes them, and `--apply` never touches them.
    """
    out = []

    # the two rails resolve the same mismatch, and WHICH one moves is a safety
    # posture: raising max_mean_step lets prices move faster, lowering
    # max_std_shrink makes the system slower to become confident. The tool
    # supplies both numbers and takes neither decision (owner, 2026-08-30).
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
        # the FRONTIER, not the target. Echoing back the --mde flag would be
        # recommending the question as its own answer; what the data supplies
        # is what each duration can resolve, and the owner picks a number they
        # would act on -- or learns the A/B cannot resolve one that small.
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
    episodes = win.get("episodes") or (shadow or {}).get("episodes")
    if per_update and days and episodes:
        per_day = episodes / days
        evidence_days = per_update / per_day if per_day else None
        floor = ly.get("calendar_floor_days_per_0.15_of_mean", 1)
        binds = "CALENDAR" if (evidence_days or 0) < floor else "EVIDENCE"
        out.append(_finding(
            "learning bottleneck", INFO, OK, binds, None,
            f"{per_update:,.0f} episodes per bounded update at "
            f"{per_day:,.0f} episodes/day = {evidence_days:.2f} days of "
            f"evidence per update, against a {floor}-day operator gate -> "
            f"{binds} BINDS. "
            + ("Widening the forced discount gap buys nothing while the gate "
               "is the limit; raise max_mean_step instead."
               if binds == "CALENDAR" else
               "Evidence is the limit: wider forced arms (information is "
               "QUADRATIC in the log price ratio) before anything else."),
            "shadow.learning_yield_would_be"))

    # the fit window W the rolling-origin sweep prefers, bounded by calib >= 2W
    sweep = (((backtest or {}).get("fidelity") or {})
             .get("calibration_window_sweep") or {})
    rec = sweep.get("recommended_fit_window")
    if rec:
        want = int(re.sub(r"\D", "", rec) or 0)
        cur = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
        s = cfg["data"]["split"]
        calib_weeks = ((pd.Timestamp(s["calib_end"])
                        - pd.Timestamp(s["calib_start"])).days + 1) / 7.0
        feasible = want and calib_weeks >= 2 * want
        # MEASURED by the sweep (fit on trailing W, score the NEXT week), so
        # it pastes -- but only when the split can hold it. calib >= 2W is the
        # dependency rule, and a W the split cannot support is a SPLIT
        # decision, which is the owner's.
        out.append(_finding(
            ("baseline_model", "calibration_fit_trailing_weeks"),
            PASTE if feasible else OWNER,
            OK if want == cur else ACT, cur, want,
            f"the rolling-origin sweep prefers {rec}"
            + (f"; calib is {calib_weeks:.1f}w so calib >= 2W holds"
               if feasible else
               f", but calib is {calib_weeks:.1f}w and the rule is calib >= 2W "
               f"({2*want}w needed) -- move the SPLIT, not this key"),
            "backtest.fidelity.calibration_window_sweep.recommended_fit_window"))
    return out


def collect(cfg, root="reports"):
    backtest = _read(os.path.join(root, "backtest.json"))
    shadow = _read(os.path.join(root, "shadow.json"))
    thresholds = _read(os.path.join(root, "thresholds.json"))
    calibration = _read(cfg["baseline_model"]["calibration_factor_path"])
    rho_art = _read(cfg["dispersion"]["rho_path"])

    missing = [n for n, r in (("backtest", backtest), ("shadow", shadow),
                              ("thresholds", thresholds))
               if r is None]
    blocks = _blocks(cfg, backtest, shadow, calibration, root)
    if missing:
        blocks.insert(0, _finding(
            "reports present", BLOCK, ACT, f"missing: {', '.join(missing)}",
            "all three", "run the pipeline before tuning: a missing report is "
            "not a passing check", "reports/"))

    findings = list(blocks)
    if not blocks:
        findings += _measured(cfg, shadow, rho_art, thresholds)
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
    """Replace one scalar in config.yaml, keeping its comment and every other
    line byte-identical. A YAML round-trip would drop the comments, and in
    this config the comment IS the reasoning."""
    anchor = ANCHORS.get(tuple(path))
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
    lines[i] = f"{head}: {value}{comment}\n"
    return "".join(lines)


def apply(cfg, report, config_path="config.yaml", out_dir="artifacts"):
    """Paste the MEASURED values, back up the config, write the decision log.

    OWNER values are never touched (AGENTS rule: never invent a SET BY OWNER
    value) -- they are recorded as pending decisions with their evidence.
    """
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(out_dir, exist_ok=True)
    backup = os.path.join(out_dir, f"config_backup_{stamp}.yaml")
    shutil.copyfile(config_path, backup)

    with open(config_path) as f:
        text = f.read()
    applied, failed = [], []
    for f_ in report["to_paste"]:
        path = tuple(f_["key"].split("."))
        try:
            text = set_scalar(text, path, f_["recommended"])
            applied.append(f_)
        except (KeyError, RuntimeError) as exc:               # noqa: PERF203
            failed.append(dict(f_, error=str(exc)))
    if applied:
        with open(config_path, "w") as f:
            f.write(text)

    log_path = os.path.join(out_dir, "config_decisions.json")
    history = _read(log_path) or {"runs": []}
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
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    order = ["none", "calibration", "retrain"]
    needed = max((RERUN.get(tuple(f_["key"].split(".")), "none")
                  for f_ in applied), key=order.index, default="none")
    history["runs"][-1]["rerun_required"] = needed
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
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
        res = apply(cfg, report, args.config)
        print(f"\nbacked up   {res['backup']}")
        print(f"decisions   {res['log']}")
        for f_ in res["applied"]:
            print(f"  pasted    {f_['key']} = {f_['recommended']}")
        for f_ in res["failed"]:
            print(f"  SKIPPED   {f_['key']}: {f_['error']}")
        if res["applied"]:
            # the MINIMUM sufficient re-run, computed from what was written.
            # Saying "re-run the bootstrap" for a runtime-only paste retrains
            # the baseline, resets the fixed point and breaks hard rule 1.
            print(f"\n{RERUN_STEPS[res['rerun']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
