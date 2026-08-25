"""pipeline.status -- the dozen numbers that decide something.

The four reports this pipeline writes carry roughly two hundred fields between
them, and that is the right number to WRITE: when a gate goes red, the
diagnostics beside it are how the cause gets found, and this project's history
is a list of times that paid for itself.

It is the wrong number to READ. So this prints only the checks that gate a
decision, each with the figure behind it, and says where to look when one is
red. Everything else stays exactly where it is.

Nothing here computes: every line is read from a report or an artifact that
some other step already wrote. A missing report is reported as "not run", never
as a pass -- an unrun check and a passing check must never look the same.

Usage:
    python3 -m pipeline.status
    python3 -m pipeline.status --json        # same content, machine-readable

Exit code is 1 if anything is FAIL, so it can gate a script.
"""

import argparse
import json
import os

from common.config import (RUNTIME_REQUIRED, artifact_mirror_drift,
                           config_get, load_config)
from common import provenance
from pricing import explore

PASS, FAIL, WARN, NONE = "PASS", "FAIL", "WARN", "not run"


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _row(name, verdict, detail, where=""):
    return {"check": name, "verdict": verdict, "detail": detail, "where": where}


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


def _bundle(cfg):
    """Do the frozen artifacts form one bundle, unedited since sealing?"""
    state = provenance.verify(cfg, provenance.load_seal(cfg))
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
    """A stale paste mis-weights every posterior step, silently.

    The remedy is NOT automatically "re-paste": the check says the two
    disagree, not which is right. Read the bundle line first -- if the
    artifacts are from an older model than the one in force, pasting their
    numbers into config walks the system backwards.
    """
    drift = artifact_mirror_drift(cfg)
    if drift:
        return _row("artifact mirrors", FAIL, "; ".join(drift),
                    "check the bundle line first, then align the stale side")
    return _row("artifact mirrors", PASS, "config matches the frozen artifacts")


def _calibration(cfg, backtest):
    if not backtest:
        return _row("calibration gate", NONE, "no backtest report",
                    "python3 -m backtest")
    fid = backtest.get("fidelity", {})
    metric = fid.get("calibration_gate_metric", "level_bias_at_anchor")
    value = fid.get("calibration_gate_value",
                    fid.get("measurement_10", {}).get("level_bias_at_anchor"))
    lo, hi = cfg["baseline_model"]["calibration_gate_band"]
    if value is None:
        return _row("calibration gate", NONE, "gate value absent")
    ok = lo <= value <= hi
    return _row("calibration gate", PASS if ok else FAIL,
                f"{metric} {value:.4f} in band [{lo}, {hi}]"
                f" · window {fid.get('gate_window', '?')}",
                "" if ok else "fidelity.by_week, by_window, measurement_10")


def _prior(cfg):
    prior = _read(cfg["posterior"]["prior"]["path"])
    if not prior:
        return _row("elasticity prior", NONE, "no prior artifact")
    per = prior.get("per_category", {})
    own = sum(1 for v in per.values()
              if v.get("own_information_weight", 0) >= 0.999
              and not v.get("wrong_sign"))
    wrong = len(prior.get("wrong_sign_categories", []))
    # A pooled or uniform prior is the DESIGNED outcome, not a failure --
    # history cannot always identify elasticity, and saying so is the honest
    # answer (9.3/9.5).
    return _row("elasticity prior", PASS,
                f"profile_density · {own}/{len(per)} categories on own data"
                + (f" · {wrong} wrong-signed (pooled)" if wrong else ""))


def _shadow(shadow):
    if not shadow:
        return _row("shadow gate", NONE, "no shadow report",
                    "python3 -m pipeline.shadow")
    g = shadow.get("shadow_gate", {})
    # the report's verdict carries a trailing note ("PASS -- proceed to ..."),
    # and each sub-check is {value, threshold, pass} rather than a scalar
    ok = str(g.get("verdict", "")).upper().startswith(PASS)

    def val(key):
        v = g.get(key)
        return v.get("value") if isinstance(v, dict) else v

    return _row("shadow gate", PASS if ok else FAIL,
                f"completeness {val('event_completeness')} · matched "
                f"{val('matched_decision_rate')} · cost-floor violations "
                f"{val('cost_floor_violations')}",
                "" if ok else "shadow.rejected_reasons, quarantined_event_count")


def _tau(cfg, backtest):
    pasted = cfg["exploration"]["tau_initial"]
    derived = (backtest or {}).get("tau_initial_derivation", {}).get("tau_initial")
    if pasted is None:
        return _row("exploration tau", FAIL,
                    f"config null; backtest derived {derived}"
                    if derived else "config null and no derivation",
                    "paste from a GATE-PASSING backtest only")
    # a pasted value is not a good one: it has to still match its source, and
    # its source has to postdate the entry-only scoping fix
    stale = explore.tau_provenance_error(cfg, backtest)
    if stale:
        return _row("exploration tau", FAIL, stale.split(". ")[0],
                    "python3 -m backtest, then re-paste tau_initial")
    return _row("exploration tau", PASS, f"{pasted} in force"
                + (f" · latest derivation {derived}" if derived else ""))


def _walkthrough(root):
    """Do the figures printed on docs/system_walkthrough.html still hold?

    The page is prose with numbers typed in, so a re-run silently invalidates
    every measured figure on it. This is the one place that gets read after a
    run, so it is the one place that can catch it. A report from a DIFFERENT
    model version is not proof the page is wrong -- it cannot be compared at
    all (hard rule 1) -- so that is WARN. A disagreement within the same run
    is a real contradiction, and FAIL.
    """
    from tools.walkthrough import figures
    # figures.check resolves paths like "reports/backtest.json", so it
    # wants the directory ABOVE the reports root status was given
    base = os.path.dirname(os.path.abspath(root))
    rows = []
    for tab in figures.SOURCES:
        verdict, detail, problems = figures.check(tab, root=base)
        status = {"ok": PASS, "drift": FAIL, "stale": WARN,
                  "pending": NONE, "no report": NONE}[verdict]
        where = ("tools/walkthrough/figures.py, then rebuild"
                 if verdict in ("drift", "stale") else "")
        rows.append(_row(f"walkthrough · {tab}", status,
                         "; ".join(problems) if problems else detail, where))
    return rows


def _guardrails(thresholds):
    if not thresholds:
        return _row("guardrail floors", NONE, "no thresholds report",
                    "python3 -m bootstrap.derive_thresholds")
    rec = thresholds.get("guardrail_threshold_recommendation")
    if not rec:
        # the file exists but predates the block -- a stale report, which is a
        # different thing from a step that never ran, and reads differently
        return _row("guardrail floors", NONE,
                    "report predates the recommendation block",
                    "re-run python3 -m bootstrap.derive_thresholds")
    verdicts = {k: v.get("verdict") for k, v in rec.items()
                if isinstance(v, dict) and "verdict" in v}
    bad = [k for k, v in verdicts.items() if str(v).upper().startswith("TOO")]
    # TOO TIGHT is blocking, not advisory: it fires on ordinary days and
    # silently suspends exploration, which is the product.
    return _row("guardrail floors", FAIL if bad else PASS,
                ", ".join(f"{k}={v}" for k, v in verdicts.items()) or "reported",
                "thresholds.guardrail_threshold_recommendation" if bad else "")


def _stops(monitor):
    if not monitor:
        return _row("stop conditions", NONE, "no monitor report",
                    "python3 -m pipeline.monitor")
    sc = monitor.get("stop_conditions", {})
    fired = [k for k, v in sc.items()
             if (v.get("fired") if isinstance(v, dict) else v) is True]
    blocked = [k for k, v in sc.items()
               if isinstance(v, dict) and v.get("blocked")]
    if fired:
        return _row("stop conditions", FAIL, "fired: " + ", ".join(fired),
                    "monitor.safety, monitor.guardrails")
    detail = f"{len(sc)} evaluated, none fired"
    if blocked:
        detail += f" · {len(blocked)} cannot fire (owner threshold null)"
    return _row("stop conditions", WARN if blocked else PASS, detail,
                "config.yaml" if blocked else "")


def _assurance(rep):
    if not rep:
        return _row("assurance", NONE, "no assurance report",
                    "python3 -m pipeline.assurance")
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
    backtest = _read(os.path.join(root, "backtest.json"))
    rows = [
        _launch_blockers(cfg),
        _bundle(cfg),
        _mirrors(cfg),
        _calibration(cfg, backtest),
        _prior(cfg),
        _tau(cfg, backtest),
        _shadow(_read(os.path.join(root, "shadow.json"))),
        _guardrails(_read(os.path.join(root, "thresholds.json"))),
        _stops(_read(os.path.join(root, "monitor.json"))),
        _assurance(_read(os.path.join(root, "assurance.json"))),
        *_walkthrough(root),
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
