"""ops.advance -- the one command that knows the order.

Reads the state from disk (the same reports, artifacts and config that
ops.status reads), works out which phase the chain is in, runs every
step that needs no human, and stops at the first decision that does --
printing exactly what to set and where the evidence is. Run it again after
acting. Idempotent: state is recomputed from disk every time, so a second
run never repeats a retrain.

    python3 -m ops.advance --plan            # show the phase table, touch nothing
    python3 -m ops.advance                   # run to the next human decision
    python3 -m ops.advance --feed <parquet>  # the daily lane, up to update --apply
    python3 -m ops.advance --retrain         # phase 1 again (rule 1: a NEW bundle)

Three properties: it never retrains unless the model is absent or --retrain
is given; a config edit re-runs what it invalidates (report vintages); it
never invents a value -- SET BY OWNER keys stop it. `daily.update --apply`
stays the human gate: the daily lane runs everything up to it.
"""

import argparse
import os
import sys

import pandas as pd

from ops.bootstrap_loop import PREPARED, step
from common import episodes, provenance
from common.config import load_config
from common.io import read_json, write_json
from ops import status, tune

RAW = "data/flc_raw.parquet"
JOURNAL = "artifacts/advance_journal.json"
DECISIONS = "artifacts/config_decisions.json"
READINESS = "launch_readiness.md"
PHASES = ("data", "bootstrap", "tune", "posterior", "shadow", "owner",
          "launch", "daily")
MAX_TUNE_ROUNDS = 4


# ------------------------------------------------------------------ probe

# a paste's phase in the journal: tau is shadow's value, everything else tune's
PASTE_PHASE = {"exploration.tau_initial": "shadow"}


def stale_reports(cfg, bundle, reports):
    """Reports produced against a bundle no longer on disk, or under a
    config whose MOVED KEYS they actually read (tune.stale_keys, the same
    routing status uses). A MEASURED paste that only writes back what a
    report measured invalidates nothing. Returns {name: why}."""
    out = {}
    for name, rep in reports.items():
        if not rep:
            continue
        av = rep.get("artifact_versions") or {}
        if bundle and av.get("baseline_model_version") not in (None, bundle):
            out[name] = f"ran against bundle {av['baseline_model_version']}"
            continue
        fp = rep.get("config") or {}
        if not fp.get("snapshot"):
            continue
        moved = [m.split(":")[0] for m in
                 provenance.config_diff(fp["snapshot"], cfg)]
        mine = tune.stale_keys(name, moved)
        if mine:
            out[name] = f"{tune.rerun_for(mine)}: " + ", ".join(mine)
    return out


def _posterior_stale(cfg):
    from engine.posterior import PosteriorStore
    prior = read_json(cfg["posterior"]["prior"]["path"])
    if not prior or not os.path.exists(cfg["posterior"]["path"]):
        return False
    return PosteriorStore(cfg).launch_stale(prior["per_category"],
                                            prior["episodes_per_week"])


def probe(cfg, root="reports", feed=None, retrain=False):
    """Everything plan() decides on, read once from disk."""
    reports = status.read_reports(root)
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
        "tune": tune.collect(cfg, root, reports=reports),
        "posterior": os.path.exists(cfg["posterior"]["path"]),
        # unlearned posterior whose cells differ from what init would write
        # now (the launch belief or the prior moved): re-init is safe and due
        "posterior_stale": _posterior_stale(cfg),
        "shadow_gate": ((reports["shadow"] or {}).get("shadow_gate") or {}
                        ).get("verdict"),
        "nulls": status.runtime_nulls(cfg),
        "launched": launched,
        "schedule_scope": str(sched.get("scope") or ""),
        "schedule_end": max(sched["by_week"]) if sched.get("by_week") else None,
        "expected_schedule_end": expected_end,
        "this_week": this_week,
        "events": os.path.isdir(events_dir) and bool(os.listdir(events_dir)),
        "feed": feed,
        "cadence": cfg["learning"]["update_cadence_days"],
        # the extract the config's windows need: train_start through the
        # hold-out's end (download_flc clips at yesterday)
        "extract_range": (cfg["data"]["split"]["train_start"],
                          (cfg["data"].get("holdout") or {}).get("end")
                          or cfg["data"]["split"]["test_end"]),
        "status": status.collect(cfg, root, reports=reports),
    }


# ------------------------------------------------------------------- plan

def _run(label, args, **kw):
    return {"kind": "run", "label": label, "args": args, **kw}


def _stop(phase, why, detail=()):
    return {"kind": "stop", "phase": phase, "why": why, "detail": list(detail)}


def _shadow_step(st):
    why = st["stale"].get("shadow")
    return _run("shadow (hold-out, every episode" + (f"; {why}" if why else "") + ")",
                ["evaluate.shadow", "--input", PREPARED,
                 "--out", "reports/shadow.json", "--max-episodes", "0",
                 "--workers", "0"],          # every core but one; byte-identical
                phase="shadow", reevaluate=True)


def plan(st):
    """The next steps, in order, ending in a stop -- pure over probe()'s
    state so every branch is testable without a workspace."""
    steps = []
    if not st["raw"] and not st["prepared"]:
        # per config: the split and hold-out dates size the pull. Credentials
        # come from ~/.env (REDSHIFT_*); download_flc fails loudly without them
        start, end = st["extract_range"]
        return [_run("download the extract (config split/holdout dates)",
                     ["fit.download_flc", "--start-date", str(start),
                      "--end-date", str(end)], phase="data", reevaluate=True)]

    # 1. bootstrap: the ONLY place a retrain happens
    if not st["model"] or not st["bundle"] or st["retrain"]:
        if not st["raw"]:
            return [_stop("bootstrap", "no raw extract to (re)train from",
                          [f"expected {RAW}"])]
        steps.append(_run("bootstrap", ["ops.bootstrap_loop", "--input", RAW],
                          phase="bootstrap", reevaluate=True))
        return steps

    # 2. reports that grade a bundle or config no longer in force -- only
    #    for the keys they read (stale_reports says which)
    if "backtest" in st["stale"]:
        steps.append(_run(f"re-grade ({st['stale']['backtest']})",
                          ["ops.bootstrap_loop", "--check-only"],
                          phase="tune", reevaluate=True))
        return steps
    if "thresholds" in st["stale"]:
        steps.append(_run(f"re-derive thresholds ({st['stale']['thresholds']})",
                          ["evaluate.derive_thresholds", "--input", PREPARED],
                          phase="tune", reevaluate=True))
        return steps
    # a shadow graded on a bundle no longer on disk (a retrain) is re-run
    # HERE: left until step 5, tune's "reports agree on one model" invariant
    # blocks on it first and the chain has no exit. A config-stale shadow
    # waits for step 5 -- the posterior it prices with may be about to move.
    if str(st["stale"].get("shadow", "")).startswith("ran against bundle"):
        return [_shadow_step(st)]

    # 3. tune: paste what the reports measured, settle, repeat
    rep = st["tune"]
    blocks = [f for f in rep["findings"] if f["class"] == tune.BLOCK]
    if blocks and not all(f["key"] == "reports present" for f in blocks):
        return [_stop("tune", "tune is BLOCKED -- an invariant is violated",
                      [f"{f['key']}: {f['current']} -- needs {f['recommended']}"
                       for f in blocks])]
    if rep["to_paste"]:
        keys = [f["key"] for f in rep["to_paste"]]
        phases = {PASTE_PHASE.get(k, "tune") for k in keys}
        steps.append({"kind": "paste", "label": "tune --apply",
                      "phase": phases.pop() if len(phases) == 1 else "tune",
                      "reevaluate": True, "keys": keys})
        return steps

    # 4. posterior, once -- re-initialised only BEFORE launch, while it holds
    #    no production state and the launch belief it was written with moved
    if not st["posterior"]:
        steps.append(_run("init posterior", ["ops.init_posterior"],
                          phase="posterior", reevaluate=True))
        return steps
    if st.get("posterior_stale") and not st["launched"]:
        steps.append(_run("re-init posterior (launch belief moved; no outcome "
                          "consumed yet)", ["ops.init_posterior", "--force"],
                          phase="posterior", reevaluate=True))
        return steps

    # 5. shadow on the hold-out, the launch record
    if "shadow" not in st["have"] or "shadow" in st["stale"]:
        return [_shadow_step(st)]
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
                          ["fit.train_baseline", "--input", PREPARED,
                           "--fit-calibration"], phase="launch"))
        steps.append(_run("re-seal", ["ops.seal", "--reason", "weekly-refit"], phase="launch",
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
                          ["daily.ingest_outcomes", "--feed", st["feed"]],
                          phase="daily"))
    if st["events"] or st["feed"]:
        steps += [_run("update: tau walk (no operator)",
                       ["daily.update", "--calibrate-tau"], phase="daily"),
                  _run("monitor", ["daily.monitor"], phase="daily"),
                  _run("assurance", ["daily.assurance"], phase="daily"),
                  _run("export events", ["daily.export_events"], phase="daily"),
                  _run("status", ["ops.status"], phase="daily", fatal=False)]
    steps.append(_stop("daily", "LAUNCHED -- the operator gate is yours",
                       [f"python3 -m daily.update --apply   every "
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


def execute(steps, config_path, root="reports", journal=JOURNAL):
    """Run the plan's steps, journal what ran; True when the caller should
    re-probe."""
    entry = {"at": pd.Timestamp.now("UTC").isoformat(),
             "phase": current_phase(steps), "ran": [], "stop": None}
    again = False
    for s in steps:
        if s["kind"] == "run":
            step(s["label"], s["args"] + ["--config", config_path],
                 fatal=s.get("fatal", True))
            entry["ran"].append({"label": s["label"],
                                 "command": "python3 -m " + " ".join(s["args"])})
        elif s["kind"] == "paste":
            res = tune.apply(tune.collect(load_config(config_path), root),
                             config_path)
            for f_ in res["applied"]:
                print(f"  pasted    {f_['key']} = {f_['recommended']}")
            for f_ in res["failed"]:
                print(f"  SKIPPED   {f_['key']}: {f_['error']}")
            entry["ran"].append({"label": "tune --apply",
                                 "pasted": [f_["key"] for f_ in res["applied"]],
                                 "skipped": [f_["key"] for f_ in res["failed"]]})
        else:
            print(f"\nSTOP [{s['phase']}] {s['why']}")
            for d in s["detail"]:
                print(f"  {d}")
            entry["stop"] = {"phase": s["phase"], "why": s["why"],
                             "detail": s["detail"]}
            break
        if s.get("reevaluate"):
            again = True
            break
    log = read_json(journal) or {"runs": []}
    log["runs"].append(entry)
    write_json(journal, log)
    return again


# ---------------------------------------------------------------- report

def report(cfg, root="reports", journal=JOURNAL, decisions=DECISIONS):
    """The launch-readiness report: what ran in each phase, every config
    value the process changed and why, the owner's decisions, the config in
    force, status, and what is still waited on. Assembled from the journal
    advance keeps, tune's decision log, the config and the reports -- never
    from memory."""
    runs = (read_json(journal) or {}).get("runs", [])
    pastes = (read_json(decisions) or {}).get("runs", [])
    st = status.collect(cfg, root)
    findings = tune.collect(cfg, root)["findings"]
    seal = provenance.verify(cfg, provenance.load_seal(cfg))
    fp = provenance.config_fingerprint(cfg, phase=None)
    now = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    last_stop = next((r["stop"] for r in reversed(runs) if r.get("stop")), None)

    lines = [f"# Launch readiness — {now}", "",
             f"bundle `{seal.get('bundle')}` · config `{cfg['meta']['config_version']}` "
             f"(digest `{fp['digest']}`) · status **{st['verdict']}**", ""]

    lines += ["## What ran, by phase", ""]
    by_phase = {}
    for r in runs:
        by_phase.setdefault(r["phase"], []).append(r)
    for phase in PHASES:
        rs = by_phase.get(phase)
        if not rs:
            continue
        lines.append(f"### {phase}")
        for r in rs:
            for item in r["ran"]:
                if "command" in item:
                    lines.append(f"- {r['at'][:16]}  `{item['command']}`")
                else:
                    lines.append(f"- {r['at'][:16]}  tune --apply pasted "
                                 + (", ".join(f"`{k}`" for k in item["pasted"]) or "nothing")
                                 + (f"; skipped {', '.join(item['skipped'])}"
                                    if item["skipped"] else ""))
            if r.get("stop"):
                lines.append(f"- {r['at'][:16]}  STOP: {r['stop']['why']}")
        lines.append("")

    lines += ["## Config values the process changed, and why", "",
              "| when | key | before → after | why | source |", "|---|---|---|---|---|"]
    for run in pastes:
        for f in run.get("applied", []):
            lines.append(f"| {run['at'][:16]} | `{f['key']}` | {f.get('current')} → "
                         f"{f.get('recommended')} | {str(f.get('evidence', '')).replace('|', '/')} "
                         f"| {f.get('source', '')} |")
    if len(lines) and lines[-1].startswith("|---"):
        lines.append("| — | — | no paste recorded yet | — | — |")
    lines.append("")

    lines += ["## Config in force (every MEASURED and SET BY OWNER value)", "",
              "| key | value | class | current? | source |", "|---|---|---|---|---|"]
    for f in findings:
        if f["class"] in (tune.PASTE, tune.OWNER):
            lines.append(f"| `{f['key']}` | {f.get('current')} | "
                         f"{'MEASURED' if f['class'] == tune.PASTE else 'SET BY OWNER'} | "
                         f"{f['status']} | {f.get('source', '')} |")
    nulls = status.runtime_nulls(cfg)
    lines += ["", "Still null: " + (", ".join(f"`{n}`" for n in nulls) or "none"), ""]

    lines += ["## Status", "", "| check | verdict | detail |", "|---|---|---|"]
    lines += [f"| {r['check']} | {r['verdict']} | {r['detail'].replace('|', '/')} |"
              for r in st["checks"]]
    lines.append("")

    lines += ["## Waiting on", ""]
    if last_stop:
        lines.append(f"**[{last_stop['phase']}] {last_stop['why']}**")
        lines += [f"- {d}" for d in last_stop["detail"]]
    else:
        lines.append("nothing recorded -- run `python3 -m ops.advance`")
    lines.append("")
    return "\n".join(lines)


def _write_readiness(config_path, root):
    cfg = load_config(config_path)
    text = report(cfg, root)
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, READINESS), "w").write(text)
    # the audit trail: the bundle's snapshot carries how it graded
    seal = provenance.load_seal(cfg) or {}
    provenance.archive_reports(cfg, root, seal.get("bundle"))
    return text


def main():
    ap = argparse.ArgumentParser(prog="ops.advance", description=__doc__,
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
    ap.add_argument("--report", action="store_true",
                    help="write and print reports/launch_readiness.md; touch "
                         "nothing else")
    args = ap.parse_args()

    if args.report:
        print(_write_readiness(args.config, args.reports))
        return 0

    rounds, seen = 0, {}
    while True:
        cfg = load_config(args.config)
        st = probe(cfg, args.reports, feed=args.feed,
                   retrain=args.retrain and rounds == 0)
        steps = plan(st)
        print(render_plan(steps))
        if args.plan:
            return 0
        rounds += 1
        # the same expensive step a third time in one invocation is a loop,
        # not progress: stop and name it rather than run shadow for a day
        for s in steps:
            if s["kind"] == "run":
                mod = s["args"][0]
                seen[mod] = seen.get(mod, 0) + 1
                if seen[mod] > 2:
                    raise SystemExit(
                        f"advance is looping: {mod} would run a third time in "
                        "this invocation. Something re-invalidates it after each "
                        "run -- read the plan above (the stale reason names the "
                        "moved keys) and reports/launch_readiness.md")
        # the round budget counts WORK, not plans: a legitimate stop on the
        # ninth round must still be journaled and reported
        if steps and steps[0]["kind"] != "stop" and rounds > 2 * MAX_TUNE_ROUNDS:
            raise SystemExit("advance did not settle -- a step keeps "
                             "invalidating another; read the plan above")
        if not execute(steps, args.config, args.reports):
            # every stop leaves the readiness report behind it
            _write_readiness(args.config, args.reports)
            print(f"\nreport      {os.path.join(args.reports, READINESS)}")
            return 0


if __name__ == "__main__":
    sys.exit(main())
