"""pipeline.status -- the dozen numbers that decide something.

Prints only the checks that gate a decision, each with the figure behind it
and where to look when red. Nothing here computes: every line is read from a
report or artifact some other step wrote, and a missing report reads "not
run", never as a pass. Exit code 1 on any FAIL, so it can gate a script.
Run: python3 -m pipeline.status [--json]
"""

import argparse
import json
import os

from common.config import (OWN_DATA_WEIGHT, RUNTIME_REQUIRED,
                           artifact_mirror_drift, config_get, load_config)
from common import provenance
from common.guardrail import verdict_is_blocking, verdict_is_insufficient
from common.io import read_json
from pricing import explore

PASS, FAIL, WARN, NONE = "PASS", "FAIL", "WARN", "not run"


def _row(name, verdict, detail, where=""):
    return {"check": name, "verdict": verdict, "detail": detail, "where": where}


def _needs(report, name, missing, run_cmd="", key=None, predates=()):
    """The NONE row a report-backed check opens with, or None to go on."""
    if not report:
        return _row(name, NONE, f"no {missing}", run_cmd)
    if key and not report.get(key):
        # file exists but predates the block: stale, not never-ran
        return _row(name, NONE, *predates)
    return None


def _launch_blockers(cfg):
    """Config values strict mode refuses to start without."""
    missing = []
    for path in RUNTIME_REQUIRED:
        try:
            if config_get(cfg, path) is None:
                missing.append(".".join(path))
        except KeyError:
            missing.append(".".join(path))
    if missing:
        return _row("launch blockers", FAIL,
                    f"{len(missing)} config values still null: "
                    + ", ".join(missing), "config.yaml")
    return _row("launch blockers", PASS, "every required value set",
                "config.yaml")


def _bundle(state):
    """Do the frozen artifacts form one bundle, unedited since sealing?"""
    if state["verdict"] == "INSUFFICIENT":
        return _row("artifact bundle", NONE, "no stamped artifacts",
                    "run the bootstrap, then python3 -m bootstrap.seal")
    if state["problems"]:
        return _row("artifact bundle", FAIL, "; ".join(state["problems"]),
                    "python3 -m bootstrap.seal after re-running the bootstrap")
    detail = f"{state['bundle']}"
    if not state["sealed_bundle"]:
        detail += " · unsealed"
    if state["missing"]:
        detail += " · absent: " + ", ".join(state["missing"])
    return _row("artifact bundle", WARN if not state["sealed_bundle"] else PASS,
                detail,
                "python3 -m bootstrap.seal" if not state["sealed_bundle"] else "")


def _mirrors(cfg):
    """A stale paste mis-weights every posterior step, silently. The remedy
    is NOT automatically "re-paste": the check says the two disagree, not
    which is right -- read the bundle line first."""
    drift = artifact_mirror_drift(cfg)
    if drift:
        return _row("artifact mirrors", FAIL, "; ".join(drift),
                    "check the bundle line first, then align the stale side")
    return _row("artifact mirrors", PASS,
                "config matches the frozen artifacts")


def _config_vs_reports(cfg, root):
    """MEASURED values pasted from a REPORT, checked against that report.

    `artifact mirrors` covers the values pasted from a frozen artifact. These
    come from backtest/shadow/thresholds instead, and nothing refused a stale
    one: a number from someone else's run -- or from the repo's SYNTHETIC
    fixture, which ships in config.yaml -- survived every check until somebody
    happened to run `tune`. Values still null are launch blockers and are
    reported there, not twice here.
    """
    from pipeline import tune

    try:
        rep = tune.collect(cfg, root)
    except Exception as exc:                                   # noqa: BLE001
        return _row("config mirrors reports", NONE,
                    f"could not evaluate: {type(exc).__name__}: {exc}",
                    "python3 -m pipeline.tune")
    blocks = [f for f in rep["findings"] if f["class"] == tune.BLOCK]
    if blocks:
        # NAME them. "a BLOCK upstream" left an owner who had just broken an
        # invariant staring at an all-green screen with one quiet "not run".
        # A missing report is genuinely not-run; every other BLOCK is an
        # invariant violated, and status must not report that as green.
        missing_only = all(f["key"] == "reports present" for f in blocks)
        detail = "; ".join(f"{f['key']}: {f['current']} -- needs "
                           f"{f['recommended']}" for f in blocks)
        return _row("config mirrors reports",
                    NONE if missing_only else FAIL, detail,
                    "python3 -m pipeline.tune  (it prints the full reason)")
    measured = {".".join(k) for k in tune.MEASURED_KEYS}
    drift = [f for f in rep["findings"]
             if f["status"] == tune.ACT and f["key"] in measured
             and f["current"] is not None]
    if not drift:
        return _row("config mirrors reports", PASS,
                    "every MEASURED value matches the report that derived it")

    # A MEASURED value tune classes PASTE is simply stale: --apply fixes it,
    # so FAIL. One it downgraded to OWNER cannot be fixed at that key at all
    # -- W is downgraded when the split cannot support the window the sweep
    # wants, and the remedy is the SPLIT. Failing on that would hold the row
    # red forever on a decision the owner has already taken, which is the
    # "a guardrail that fires constantly kills the loop" failure mode.
    stale = [f for f in drift if f["class"] == tune.PASTE]
    decisions = [f for f in drift if f["class"] != tune.PASTE]
    if stale:
        return _row("config mirrors reports", FAIL,
                    "; ".join(f"{f['key']} is {f['current']} but the report "
                              + (f"says {f['recommended']}"
                                 if f['recommended'] is not None else
                                 f"did not measure it ({f['evidence']})")
                              for f in stale),
                    "python3 -m pipeline.tune --apply")
    return _row("config mirrors reports", WARN,
                "; ".join(f"{f['key']} is {f['current']} and the sweep prefers "
                          f"{f['recommended']}, but that is not a paste: "
                          f"{f['evidence'].split(' -- ')[-1][:80]}"
                          for f in decisions),
                "python3 -m pipeline.tune  (OWNER decision, not a stale value)")


def _calibration(cfg, backtest):
    # DIAGNOSTIC, not a gate: calibration is always applied (owner,
    # 2026-08-25). Out of band -> WARN, never FAIL.
    if row := _needs(backtest, "calibration level", "backtest report",
                     "python3 -m backtest"):
        return row
    fid = backtest.get("fidelity", {})
    metric = fid.get("calibration_gate_metric", "level_bias_at_anchor")
    value = fid.get("calibration_gate_value",
                    fid.get("measurement_10", {}).get("level_bias_at_anchor"))
    lo, hi = cfg["baseline_model"]["calibration_gate_band"]
    if value is None:
        return _row("calibration level", NONE, "diagnostic value absent")
    ok = lo <= value <= hi
    return _row("calibration level", PASS if ok else WARN,
                f"{metric} {value:.4f} in band [{lo}, {hi}]"
                f" · window {fid.get('gate_window', '?')}",
                "" if ok else "fidelity.by_week, by_window, measurement_10")


def _calibration_convergence(cfg):
    # the calibration <-> dispersion loop is resolved by iteration; this row
    # says whether anyone ASSERTED the fixed point settled. WARN, not FAIL:
    # like the level band it is a chain-health reading, not a launch gate.
    cal = read_json(cfg["baseline_model"]["calibration_factor_path"])
    if row := _needs(cal, "calibration convergence", "calibration artifact",
                     key="convergence", predates=(
                         "never checked -- the factor <-> r loop is assumed, "
                         "not asserted",
                         "python3 -m bootstrap.train_baseline --input "
                         "data/prepared.parquet --check-convergence")):
        return row
    block = cal["convergence"]
    # STALE BEATS CONVERGED: a verdict is only about the artifacts in force
    # when it ran; a moved prior/r/rho means the loop has turned again.
    from common.provenance import file_digest
    moved = []
    for name, path in (("prior", (cfg["posterior"]["prior"] or {}).get("path")),
                       ("r_lookup", cfg["dispersion"].get("r_lookup_path")),
                       ("rho", cfg["dispersion"].get("rho_path"))):
        was = (block.get("checked_against") or {}).get(name)
        if was and path and os.path.exists(path) and file_digest(path) != was:
            moved.append(name)
    if moved:
        return _row("calibration convergence", WARN,
                    f"checked against a chain that has since moved: "
                    f"{', '.join(moved)} re-fitted after the check",
                    "re-run --check-convergence; the loop turned again")

    ok = bool(block.get("converged"))
    return _row("calibration convergence", PASS if ok else WARN,
                f"max |dlog f| {block.get('max_abs_dlog')} vs tol "
                f"{block.get('tol_log')}"
                + ("" if block.get("checked_against") else
                   " · pre-digest check, staleness unverifiable"),
                "" if ok else "one more iteration: --fit-calibration, "
                              "estimate_prior, fit_dispersion, re-check")


def _prior(cfg):
    prior = read_json((cfg["posterior"].get("prior") or {}).get("path"))
    if row := _needs(prior, "elasticity prior", "prior artifact"):
        return row
    per = prior.get("per_category", {})
    own = sum(1 for v in per.values()
              if v.get("own_information_weight", 0) >= OWN_DATA_WEIGHT
              and not v.get("wrong_sign"))
    wrong = len(prior.get("wrong_sign_categories", []))
    # a pooled or uniform prior is the DESIGNED outcome, not a failure (9.3/9.5)
    return _row("elasticity prior", PASS,
                f"profile_density · {own}/{len(per)} categories on own data"
                + (f" · {wrong} wrong-signed (pooled)" if wrong else ""))


def _shadow(shadow):
    if row := _needs(shadow, "shadow gate", "shadow report",
                     "python3 -m pipeline.shadow"):
        return row
    g = shadow.get("shadow_gate", {})
    # verdict carries a trailing note; sub-checks are {value, threshold, pass}
    ok = str(g.get("verdict", "")).upper().startswith(PASS)

    def val(key):
        v = g.get(key)
        return v.get("value") if isinstance(v, dict) else v

    return _row("shadow gate", PASS if ok else FAIL,
                f"completeness {val('event_completeness')} "
                "· cost-floor violations "
                f"{val('cost_floor_violations')}",
                "" if ok else "shadow.rejected_reasons, quarantined_event_count")


def _vintages(cfg, state, reports):
    """Gate evidence is only evidence about the artifacts AND config it ran
    under (hard rule 1): after a retrain, yesterday's backtest and shadow
    grade a model no longer on disk; after a paste, they grade a config no
    longer in force. Model mismatch is FAIL. A config that moved is WARN and
    NAMES what moved -- every report now carries the fingerprint of the
    config it read (`config.digest` + `config.snapshot`, by phase), so this
    no longer depends on a human remembering to bump `meta.config_version`.
    """
    bundle = state["bundle"]
    if bundle is None:
        return _row("report vintages", NONE,
                    "no single artifact bundle to compare against",
                    "see the artifact bundle line")
    live = provenance.config_fingerprint(cfg, phase=None)["digest"]
    stale, moved, checked = [], [], []
    for name, rep in reports.items():
        if not rep:
            continue                # its own row already reads "not run"
        av = rep.get("artifact_versions") or {}
        if "baseline_model_version" in av and av["baseline_model_version"] != bundle:
            stale.append(f"{name} ran against bundle "
                         f"{av['baseline_model_version']}")
            continue
        fp = rep.get("config")
        if fp:
            if fp.get("digest") != live:
                from pipeline import tune                    # sibling; no cycle
                diff = provenance.config_diff(fp.get("snapshot") or {}, cfg)
                # a paste that only writes back what a report measured, or a
                # key no report reads, is not a reason to re-grade it
                live_moves = [d for d in diff
                              if tune.rerun_for([d.split(":")[0]]) != "none"]
                if live_moves:
                    moved.append(f"{name} ({fp.get('phase')}) ran under config "
                                 f"{fp.get('digest')}; since then: "
                                 + "; ".join(live_moves))
                else:
                    checked.append(f"{name}={fp.get('phase')} (inert pastes since)")
            else:
                checked.append(f"{name}={fp.get('phase')}")
        elif av:                    # a report from before fingerprints
            if av.get("config_version") != cfg["meta"]["config_version"]:
                moved.append(f"{name} ran under config_version "
                             f"{av.get('config_version')} (pre-fingerprint)")
            else:
                checked.append(f"{name}=unfingerprinted")
    if stale:
        return _row("report vintages", FAIL,
                    "; ".join(stale) + f" -- artifacts on disk are {bundle}",
                    "re-run it: every row it feeds grades a model that is "
                    "no longer deployed")
    if moved:
        return _row("report vintages", WARN, "; ".join(moved),
                    "re-run to re-grade under the config now in force")
    if not checked:
        return _row("report vintages", NONE, "no stamped reports yet")
    return _row("report vintages", PASS,
                f"bundle {bundle} · config {live} · " + ", ".join(checked))


def _tau(cfg, backtest, shadow=None):
    pasted = cfg["exploration"]["tau_initial"]
    sh = (shadow or {}).get("tau_initial_derivation") or {}
    derived = sh.get("tau_initial") if sh.get("tau_initial") is not None else \
        ((backtest or {}).get("tau_initial_derivation") or {}).get("tau_initial")
    src = "shadow" if sh.get("tau_initial") is not None else "backtest"
    if pasted is None:
        # shadow derives its own launch tau; the paste gates the PILOT
        return _row("exploration tau", FAIL,
                    f"config null; {src} derived {derived}"
                    if derived is not None else "config null and no derivation",
                    "paste from shadow's tau_initial_derivation (preferred) "
                    "or a gate-passing backtest")
    # a paste must still match its source; shadow's anchored-path derivation
    # outranks the backtest's exploit-only one
    stale = explore.tau_provenance_error(cfg, backtest, shadow)
    if stale:
        return _row("exploration tau", FAIL, stale.split(". ")[0],
                    "python3 -m pipeline.shadow, then re-paste tau_initial")
    return _row("exploration tau", PASS, f"{pasted} in force"
                + (f" · latest {src} derivation {derived}"
                   if derived is not None else ""))


def _guardrails(thresholds):
    if row := _needs(thresholds, "guardrail floors", "thresholds report",
                     "python3 -m bootstrap.derive_thresholds",
                     key="guardrail_threshold_recommendation", predates=(
                         "report predates the recommendation block",
                         "re-run python3 -m bootstrap.derive_thresholds")):
        return row
    rec = thresholds["guardrail_threshold_recommendation"]
    verdicts = {k: v.get("verdict") for k, v in rec.items()
                if isinstance(v, dict) and "verdict" in v}
    # All three of design 12's blocking verdicts, not just the first. TOO
    # TIGHT fires on ordinary days and silently suspends exploration; BLOCKED
    # means no threshold on that basis is both safe and useful; LIKELY INERT
    # means the guardrail cannot fire, and a guardrail that cannot fire is
    # absent, not conservative. Passing two of the three let --apply paste a
    # floor the report itself calls unusable while status stayed green.
    bad = [k for k, v in verdicts.items() if verdict_is_blocking(v)]
    # "insufficient history" is neither: nobody measured the floor, so
    # nothing was checked -- WARN, never PASS
    thin = [k for k, v in verdicts.items() if verdict_is_insufficient(v)]
    return _row("guardrail floors",
                FAIL if bad else WARN if thin else PASS,
                ", ".join(f"{k}={v}" for k, v in verdicts.items()) or "reported",
                "thresholds.guardrail_threshold_recommendation" if bad else
                "more closed-episode history, then re-run derive_thresholds"
                if thin else "")


def _stops(monitor):
    if row := _needs(monitor, "stop conditions", "monitor report",
                     "python3 -m pipeline.monitor"):
        return row
    # monitor.stop_conditions is {fired: {name: bool|status}, guardrails:
    # {...}, suspend_exploration: bool} -- read THAT shape. Iterating the
    # top level looking for v["fired"] found three container keys, counted
    # them as "3 evaluated", never saw a per-condition flag, and left the
    # owner-null WARN branch dead.
    sc = monitor.get("stop_conditions", {})
    flags = sc.get("fired") if isinstance(sc.get("fired"), dict) else {}
    fired = [k for k, v in flags.items() if v is True]
    blocked = [k for k, v in flags.items()
               if isinstance(v, str) and v.upper().startswith("BLOCKED")]
    if fired:
        return _row("stop conditions", FAIL, "fired: " + ", ".join(fired),
                    "monitor.safety, monitor.guardrails")
    detail = f"{len(flags)} evaluated, none fired"
    if blocked:
        detail += f" · {len(blocked)} cannot fire (owner threshold null)"
    return _row("stop conditions", WARN if blocked else PASS, detail,
                "config.yaml" if blocked else "")


def _assurance(rep):
    if row := _needs(rep, "assurance", "assurance report",
                     "python3 -m pipeline.assurance"):
        return row
    names = ("reproduction", "dispersion", "correlation", "exploration")
    v = {n: rep.get(n, {}).get("verdict", "?") for n in names}
    failing = [n for n, x in v.items() if x == FAIL]
    thin = [n for n, x in v.items() if x == "INSUFFICIENT"]
    detail = " · ".join(f"{n} {x.lower()}" for n, x in v.items())
    if failing:
        return _row("assurance", FAIL, detail, "reports/assurance.json")
    # thin is not passing: a check that saw almost nothing must not read as
    # one that looked and found nothing
    return _row("assurance", WARN if thin else PASS, detail)


def collect(cfg, root="reports"):
    """`root` is injectable so the tests can point at a fixture directory --
    a status view that cannot itself be tested is not worth trusting."""
    backtest = read_json(os.path.join(root, "backtest.json"))
    shadow = read_json(os.path.join(root, "shadow.json"))
    reports = {"backtest": backtest, "shadow": shadow,
               "thresholds": read_json(os.path.join(root, "thresholds.json")),
               "monitor": read_json(os.path.join(root, "monitor.json")),
               "assurance": read_json(os.path.join(root, "assurance.json"))}
    state = provenance.verify(cfg, provenance.load_seal(cfg))
    rows = [
        _launch_blockers(cfg),
        _bundle(state),
        _mirrors(cfg),
        _config_vs_reports(cfg, root),
        _vintages(cfg, state, reports),
        _calibration(cfg, backtest),
        _calibration_convergence(cfg),
        _prior(cfg),
        _tau(cfg, backtest, shadow),
        _shadow(shadow),
        _guardrails(reports["thresholds"]),
        _stops(reports["monitor"]),
        _assurance(reports["assurance"]),
    ]
    return {
        "checks": rows,
        "failing": [r["check"] for r in rows if r["verdict"] == FAIL],
        "not_run": [r["check"] for r in rows if r["verdict"] == NONE],
        "verdict": FAIL if any(r["verdict"] == FAIL for r in rows) else PASS,
    }


def render(report):
    width = max(len(r["check"]) for r in report["checks"])
    lines = []
    for r in report["checks"]:
        lines.append(f"  {r['verdict']:<8} {r['check']:<{width}}  {r['detail']}")
        if r["where"]:
            lines.append(f"  {'':<8} {'':<{width}}  → {r['where']}")
    tail = ("all gates green" if report["verdict"] == PASS
            else "FAILING: " + ", ".join(report["failing"]))
    if report["not_run"]:
        tail += f"   ({len(report['not_run'])} not run)"
    return "\n".join(lines) + "\n\n  " + tail


def main():
    ap = argparse.ArgumentParser(prog="pipeline.status", description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    report = collect(load_config(args.config), args.reports)
    print(json.dumps(report, indent=2) if args.json else render(report))
    raise SystemExit(1 if report["verdict"] == FAIL else 0)


if __name__ == "__main__":
    main()
