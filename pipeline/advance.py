"""pipeline.advance -- the one command that knows the order.

Reads the state from disk (the same reports, artifacts and config that
pipeline.status reads), works out which phase the chain is in, runs every
step that needs no human, and stops at the first decision that does --
printing exactly what to set and where the evidence is. Run it again after
acting. Idempotent: state is recomputed from disk every time, so a second
run never repeats a retrain.

    python3 -m pipeline.advance --plan            # show the phase table, touch nothing
    python3 -m pipeline.advance                   # run to the next human decision
    python3 -m pipeline.advance --feed <parquet>  # the daily lane, up to update --apply
    python3 -m pipeline.advance --retrain         # phase 1 again (rule 1: a NEW bundle)

Three properties: it never retrains unless the model is absent or --retrain
is given; a config edit re-runs what it invalidates (report vintages); it
never invents a value -- SET BY OWNER keys stop it. `pipeline.update --apply`
stays the human gate: the daily lane runs everything up to it.
"""

import argparse
import os
import sys

import pandas as pd

from bootstrap.run import PREPARED, step
from common import episodes, provenance
from common.config import RUNTIME_REQUIRED, config_get, load_config
from common.io import read_json
from pipeline import status, tune, update

RAW = "data/flc_raw.parquet"
PHASES = ("data", "bootstrap", "tune", "posterior", "shadow", "owner",
          "launch", "daily")
MAX_TUNE_ROUNDS = 4


# ------------------------------------------------------------------ probe

def stale_reports(cfg, bundle, reports):
    """Reports produced against a bundle or config no longer in force --
    the same test status.report_vintages applies, as a list to act on."""
    live = provenance.config_fingerprint(cfg, phase=None)["digest"]
    out = []
    for name, rep in reports.items():
        if not rep:
            continue
        av = rep.get("artifact_versions") or {}
        if bundle and av.get("baseline_model_version") not in (None, bundle):
            out.append(name)
        elif (rep.get("config") or {}).get("digest") not in (None, live):
            out.append(name)
    return out


def probe(cfg, root="reports", feed=None, retrain=False):
    """Everything plan() decides on, read once from disk."""
    reports = {n: read_json(os.path.join(root, f"{n}.json"))
               for n in ("backtest", "thresholds", "shadow", "monitor",
                         "assurance")}
    seal = provenance.verify(cfg, provenance.load_seal(cfg))
    cal = read_json(cfg["baseline_model"]["calibration_factor_path"]) or {}
    sched = (cal.get("schedule") or {})
    launched = bool(cfg["data"].get("launch_date"))
    events_dir = cfg["events"]["store_dir"]
    # the week the schedule must reach: one past the latest prepared week
    # (the week being priced). If the extract is stale no re-fit can reach
    # this week, and that is a data problem to stop on, not a loop to run.
    this_week = episodes.week_key(
        pd.Series([pd.Timestamp.now("UTC").tz_localize(None)]))[0]
    expected_end = None
    if os.path.exists(PREPARED):
        last = episodes.week_key(pd.read_parquet(PREPARED, columns=["date"]).date).max()
        expected_end = (episodes.week_start(last)
                        + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return {
        "raw": os.path.exists(RAW),
        "prepared": os.path.exists(PREPARED),
        "model": os.path.exists(cfg["baseline_model"]["model_path"]),
        "bundle": seal.get("bundle"),
        "retrain": retrain,
        "stale": stale_reports(cfg, seal.get("bundle"), reports),
        "have": {n for n, r in reports.items() if r},
        "tune": tune.collect(cfg, root),
        "posterior": os.path.exists(cfg["posterior"]["path"]),
        "shadow_gate": ((reports["shadow"] or {}).get("shadow_gate") or {}
                        ).get("verdict"),
        "nulls": [".".join(p) for p in RUNTIME_REQUIRED
                  if config_get(cfg, p) is None],
        "launched": launched,
        "schedule_scope": str(sched.get("scope") or ""),
        "schedule_end": max(sched["by_week"]) if sched.get("by_week") else None,
        "expected_schedule_end": expected_end,
        "this_week": this_week,
        "events": os.path.isdir(events_dir) and bool(os.listdir(events_dir)),
        "feed": feed,
        "cadence": cfg["learning"]["update_cadence_days"],
        "status": status.collect(cfg, root),
    }


# ------------------------------------------------------------------- plan

def _run(label, args, **kw):
    return {"kind": "run", "label": label, "args": args, **kw}


def _stop(phase, why, detail=()):
    return {"kind": "stop", "phase": phase, "why": why, "detail": list(detail)}


def plan(st):
    """The next steps, in order, ending in a stop -- pure over probe()'s
    state so every branch is testable without a workspace."""
    steps = []
    if not st["raw"] and not st["prepared"]:
        return [_stop("data", "no extract on disk",
                      ["python3 -m bootstrap.download_flc  (REDSHIFT_* from ~/.env)"])]

    # 1. bootstrap: the ONLY place a retrain happens
    if not st["model"] or not st["bundle"] or st["retrain"]:
        if not st["raw"]:
            return [_stop("bootstrap", "no raw extract to (re)train from",
                          [f"expected {RAW}"])]
        steps.append(_run("bootstrap", ["bootstrap.run", "--input", RAW],
                          phase="bootstrap", reevaluate=True))
        return steps

    # 2. reports that grade a bundle or config no longer in force
    if {"backtest", "thresholds"} & set(st["stale"]):
        steps.append(_run("re-grade under the config now in force",
                          ["bootstrap.run", "--check-only"],
                          phase="tune", reevaluate=True))
        return steps

    # 3. tune: paste what the reports measured, settle, repeat
    rep = st["tune"]
    blocks = [f for f in rep["findings"] if f["class"] == tune.BLOCK]
    if blocks and not all(f["key"] == "reports present" for f in blocks):
        return [_stop("tune", "tune is BLOCKED -- an invariant is violated",
                      [f"{f['key']}: {f['current']} -- needs {f['recommended']}"
                       for f in blocks])]
    if rep["to_paste"]:
        steps.append({"kind": "paste", "label": "tune --apply",
                      "phase": "tune", "reevaluate": True,
                      "keys": [f["key"] for f in rep["to_paste"]]})
        return steps

    # 4. posterior, once
    if not st["posterior"]:
        steps.append(_run("init posterior", ["bootstrap.init_posterior"],
                          phase="posterior", reevaluate=True))
        return steps

    # 5. shadow on the hold-out, the launch record
    if "shadow" not in st["have"] or "shadow" in st["stale"]:
        steps.append(_run("shadow (hold-out, every episode)",
                          ["pipeline.shadow", "--input", PREPARED,
                           "--out", "reports/shadow.json", "--max-episodes", "0"],
                          phase="shadow", reevaluate=True))
        return steps
    if st["shadow_gate"] and not str(st["shadow_gate"]).startswith("PASS"):
        return [_stop("shadow", f"shadow gate: {st['shadow_gate']}",
                      ["read reports/shadow.json -> shadow_gate, rejected_reasons"])]

    # 6. owner decisions: never invented
    owner = [n for n in st["nulls"] if n != "data.launch_date"]
    if owner:
        detail = []
        by_key = {f["key"]: f for f in rep["findings"]}
        for key in owner:
            f = by_key.get(key)
            detail.append(f"{key}: " + (f"{f['evidence']}  [{f['source']}]"
                                        if f else "null -- see reports/thresholds.json"))
        return [_stop("owner", "SET BY OWNER values are null", detail)]

    # 7. launch day
    if not st["launched"]:
        return [_stop("launch", "data.launch_date is null",
                      ["set it on launch day; the weekly re-fit then schedules "
                       "through the latest data (never move split.test_end)"])]
    behind = (not st["schedule_scope"].startswith("production")
              or (st["schedule_end"] or "") < (st["expected_schedule_end"] or ""))
    if behind:
        steps.append(_run("weekly level re-fit",
                          ["bootstrap.train_baseline", "--input", PREPARED,
                           "--fit-calibration"], phase="launch"))
        steps.append(_run("re-seal", ["bootstrap.seal"], phase="launch",
                          reevaluate=True))
        return steps
    if (st["schedule_end"] or "") < st["this_week"]:
        return [_stop("launch", "the extract is stale: no re-fit can reach the "
                      "week being priced",
                      [f"schedule ends {st['schedule_end']}, this week is "
                       f"{st['this_week']} -- refresh data/prepared.parquet "
                       "(download_flc + prepare_data), then run again"])]
    if st["status"]["failing"]:
        return [_stop("launch", "status is red",
                      [f"{c}" for c in st["status"]["failing"]])]

    # 8. daily lane, up to the human gate
    if st["feed"]:
        steps.append(_run("ingest outcomes",
                          ["pipeline.ingest_outcomes", "--feed", st["feed"]],
                          phase="daily"))
    if st["events"] or st["feed"]:
        steps += [_run("update: tau walk (no operator)",
                       ["pipeline.update", "--calibrate-tau"], phase="daily"),
                  _run("monitor", ["pipeline.monitor"], phase="daily"),
                  _run("assurance", ["pipeline.assurance"], phase="daily"),
                  _run("export events", ["pipeline.export_events"], phase="daily"),
                  _run("status", ["pipeline.status"], phase="daily", fatal=False)]
    steps.append(_stop("daily", "LAUNCHED -- the operator gate is yours",
                       [f"python3 -m pipeline.update --apply   every "
                        f"{st['cadence']} days (learning.update_cadence_days); "
                        "read the batch summary above first -- a second --apply "
                        "consuming nothing is correct"]))
    return steps


# ----------------------------------------------------------------- driver

def current_phase(steps):
    return steps[0].get("phase") if steps else "daily"


def render_plan(steps):
    phase = current_phase(steps)
    lines = ["phase   " + "  ".join(f"[{p}]" if p == phase else p for p in PHASES), ""]
    for s in steps:
        if s["kind"] == "run":
            lines.append(f"  run   {s['label']:<42} python3 -m " + " ".join(s["args"]))
        elif s["kind"] == "paste":
            lines.append(f"  paste {s['label']:<42} " + ", ".join(s["keys"]))
        else:
            lines.append(f"  STOP  [{s['phase']}] {s['why']}")
            lines += [f"          {d}" for d in s["detail"]]
    return "\n".join(lines)


def execute(steps, config_path, root="reports"):
    """Run the plan's steps; True when the caller should re-probe."""
    for s in steps:
        if s["kind"] == "run":
            step(s["label"], s["args"] + ["--config", config_path],
                 fatal=s.get("fatal", True))
        elif s["kind"] == "paste":
            res = tune.apply(tune.collect(load_config(config_path), root),
                             config_path)
            for f_ in res["applied"]:
                print(f"  pasted    {f_['key']} = {f_['recommended']}")
            for f_ in res["failed"]:
                print(f"  SKIPPED   {f_['key']}: {f_['error']}")
        else:
            print(f"\nSTOP [{s['phase']}] {s['why']}")
            for d in s["detail"]:
                print(f"  {d}")
            return False
        if s.get("reevaluate"):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(prog="pipeline.advance", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--feed", default=None,
                    help="yesterday's hourly feed parquet: runs the daily lane")
    ap.add_argument("--retrain", action="store_true",
                    help="run the bootstrap again even though a bundle exists "
                         "(rule 1: this is a NEW bundle; reports re-run after)")
    ap.add_argument("--plan", action="store_true",
                    help="print the phase table and the next steps; touch nothing")
    args = ap.parse_args()

    rounds = 0
    while True:
        cfg = load_config(args.config)
        st = probe(cfg, args.reports, feed=args.feed,
                   retrain=args.retrain and rounds == 0)
        steps = plan(st)
        print(render_plan(steps))
        if args.plan:
            return 0
        rounds += 1
        if rounds > 2 * MAX_TUNE_ROUNDS:
            raise SystemExit("advance did not settle -- a step keeps "
                             "invalidating another; read the plan above")
        if not execute(steps, args.config, args.reports):
            return 0


if __name__ == "__main__":
    sys.exit(main())
