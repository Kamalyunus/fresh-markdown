"""ops.readiness -- the launch-readiness report `ops.advance` leaves behind.

What ran in each phase, every config value the process changed and why,
the owner's decisions, the config in force, status, and what is still
waited on -- assembled from the journal advance keeps, tune's decision
log, the config and the reports, never from memory. Every stop writes it
(`advance --report` regenerates it); the driver is ops.advance.
"""

import os

import pandas as pd

from common import provenance
from common.config import load_config
from common.io import read_json
from common.paths import DECISIONS, JOURNAL, READINESS
from ops import status, tune
from ops.run import PHASES


def report(cfg, root="reports", journal=JOURNAL, decisions=DECISIONS):
    """The launch-readiness report: what ran in each phase, every config
    value the process changed and why, the owner's decisions, the config in
    force, status, and what is still waited on. Assembled from the journal
    advance keeps, tune's decision log, the config and the reports -- never
    from memory."""
    runs = (read_json(journal) or {}).get("runs", [])
    pastes = (read_json(decisions) or {}).get("runs", [])
    reports, artifacts = status.read_reports(root), status.read_artifacts(cfg)
    st = status.collect(cfg, root, reports, artifacts)
    findings = tune.collect(cfg, root, reports, artifacts)["findings"]
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
    if lines[-1].startswith("|---"):
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


def write_readiness(config_path, root):
    """Write reports/launch_readiness.md and archive the reports beside
    the bundle's audit snapshot; returns the text."""
    cfg = load_config(config_path)
    text = report(cfg, root)
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, READINESS), "w").write(text)
    # the audit trail: the bundle's snapshot carries how it graded
    seal = provenance.load_seal(cfg) or {}
    provenance.archive_reports(cfg, root, seal.get("bundle"))
    return text
