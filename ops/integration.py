"""ops.integration -- keep the standalone pricing folder engineering runs
honest and current: the artifact sync, and the check that its copies
still match their sources.

`integration/` is what the pricing host needs and nothing else:

    integration/
      README.md            the handoff
      price_hour.py        every clock hour: the snapshot in, a price per shelf out
      build_features.py    every morning: the day's feature table from the feed
      check_inputs.py      their three tables, and the hour's response, checked
      config.yaml          the owner's readings (synced from the repository's)
      requirements.lock    the pinned libraries
      artifacts/  data/    SYNCED from the owner's chain (below); never in git
      snapshots/ feed/ failures/ features/ decisions/ reports/hours/ logs/ events_store/
      examples/  docs/     the four example tables, the handover page
      src/                 the hourly path, one file per repository module
      MANIFEST.json        every file: its source, verbatim or curated (and why)

The folder is maintained IN PLACE, never generated: the owner asked for a
minimal standalone folder without refactoring the repository, so `src/`
carries the hourly path's modules copied file by file, and the handful
that had to differ from their source -- a re-export dropped, the model
applier split from the trainer, one function lifted into a small module,
exploit-only fixed on -- are listed in MANIFEST.json as `curated` with
the reason. Every other file is `verbatim` and `check()` refuses a copy
that no longer equals its repository source, so a fix to the engine is
either ported to the folder or fails the suite. A curated file is ported
by hand when its source moves; the manifest is the list to walk.

`sync(cfg)` copies the artifacts the config names (the sealed bundle, the
posterior, the prior), the extract and the rolling feed history, and the
config itself into the folder. `ops.seal` calls it after every seal and
`ops.advance` at the end of every run, so a retrain, a paste round or a
posterior re-init reaches the folder without a hand step. It targets
`integration/` under the CURRENT working directory -- the repo root when
the owner runs the chain, a workspace's own folder in a rehearsal -- and
does nothing where no folder exists.

Run: python3 -m ops.integration --check | --sync [--out integration]
"""

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import shutil
import sys

from common.config import config_get, load_config
from common.paths import RAW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "integration"
SRC = "src"
PACKAGES = ("common", "engine", "events", "fit", "daily", "ops")
MANIFEST = "MANIFEST.json"
SYNCED = os.path.join("artifacts", "synced.json")
COMMANDS = ("price_hour.py", "build_features.py", "check_inputs.py")

# the working directories the cron lines name; the runtime state, ignored by git
RUNTIME_DIRS = ("snapshots", "feed", "failures", "features", "decisions",
                "reports/hours", "logs", "events_store", "artifacts", "data")

# what sync carries: the artifacts the config names, by config key
ARTIFACT_KEYS = (("artifacts", "bundle_path"),
                 ("data", "split_manifest_path"),
                 ("baseline_model", "model_path"),
                 ("baseline_model", "feature_schema_path"),
                 ("baseline_model", "calibration_factor_path"),
                 ("dispersion", "r_lookup_path"),
                 ("dispersion", "rho_path"),
                 ("posterior", "prior", "path"),
                 ("posterior", "path"))


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src, out, rel):
    dst = os.path.join(out, rel)
    os.makedirs(os.path.dirname(dst) or out, exist_ok=True)
    shutil.copyfile(src, dst)
    return rel


# ---------------------------------------------------------------- closure
# read against the folder's src/: every repo-local import a module makes
# must resolve inside src/, or the folder is not standalone

def _module_path(root, mod):
    p = os.path.join(root, *mod.split("."))
    if os.path.isfile(p + ".py"):
        return p + ".py"
    if os.path.isfile(os.path.join(p, "__init__.py")):
        return os.path.join(p, "__init__.py")
    return None


def local_imports(root, path):
    """The repo-local modules `path` imports, module level or nested."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in PACKAGES:
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom) and node.module \
                and node.module.split(".")[0] in PACKAGES:
            out.add(node.module)
            for a in node.names:                 # `from pkg import module`
                if _module_path(root, f"{node.module}.{a.name}"):
                    out.add(f"{node.module}.{a.name}")
    return out


def modules(src):
    """Every module under `src`, dotted."""
    out = []
    for dp, _, fs in os.walk(src):
        if "__pycache__" in dp:
            continue
        for f in fs:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dp, f), src)[:-3]
                out.append(rel.replace(os.sep, ".").removesuffix(".__init__"))
    return sorted(out)


def unresolved(src):
    """(module, import) pairs whose import is not inside src/: [] when the
    folder is standalone."""
    out = []
    for mod in modules(src):
        for dep in local_imports(src, _module_path(src, mod)):
            if _module_path(src, dep) is None:
                out.append((mod, dep))
    return out


# ------------------------------------------------------------------ check

def check(root=ROOT, out=None):
    """The folder against its manifest and the repository: `drift` names a
    verbatim file that no longer equals its source (or is gone), `unlisted`
    a file on disk the manifest does not carry, `missing` a manifested file
    not on disk, `dangling` an import that leaves src/. All four empty
    means the folder is current and standalone."""
    out = out or os.path.join(root, OUT)
    with open(os.path.join(out, MANIFEST)) as f:
        manifest = json.load(f)
    files = manifest["files"]
    drift, missing = [], []
    for rel, entry in files.items():
        here = os.path.join(out, rel)
        if not os.path.exists(here):
            missing.append(rel)
            continue
        if entry.get("verbatim"):
            src = os.path.join(root, entry["from"])
            if not os.path.exists(src) or _digest(src) != _digest(here):
                drift.append(rel)
    skip = {d.split("/")[0] for d in RUNTIME_DIRS}
    on_disk = set()
    for dp, dns, fs in os.walk(out):
        dns[:] = [d for d in dns if d != "__pycache__"
                  and not (dp == out and d in skip)]
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), out)
            if rel != MANIFEST:
                on_disk.add(rel)
    return {"drift": sorted(drift), "missing": sorted(missing),
            "unlisted": sorted(on_disk - set(files)),
            "dangling": unresolved(os.path.join(out, SRC)),
            "curated": {rel: e["curated"] for rel, e in files.items() if e.get("curated")}}


def current(root=ROOT, out=None):
    r = check(root, out)
    return not (r["drift"] or r["missing"] or r["unlisted"] or r["dangling"])


# ------------------------------------------------------------------- sync

def sync(cfg, out=OUT, config_path="config.yaml"):
    """Copy the artifacts `cfg` names, the extract and the rolling feed
    history, and the config into the folder at `out` (relative to the
    working directory). Returns {"copied", "absent", "bundle"} or None
    where no folder exists. An artifact not on disk yet is listed, never
    an error: the chain syncs after every seal, and the first seal
    predates the posterior."""
    if not os.path.exists(os.path.join(out, MANIFEST)):
        return None
    copied, absent = [], []
    for key in ARTIFACT_KEYS:
        src = config_get(cfg, key, default=None)
        if src and os.path.exists(src):
            copied.append(_copy(src, out, os.path.join("artifacts", os.path.basename(src))))
        else:
            absent.append(".".join(key))
    for src in (RAW, config_get(cfg, ("features", "history_path"), default=None)):
        if src and os.path.exists(src):
            copied.append(_copy(src, out, os.path.join("data", os.path.basename(src))))
    if os.path.exists(config_path):
        copied.append(_copy(config_path, out, "config.yaml"))
    bundle = None
    bp = config_get(cfg, ("artifacts", "bundle_path"), default=None)
    if bp and os.path.exists(bp):
        with open(bp) as f:
            bundle = (json.load(f) or {}).get("bundle")
    for d in RUNTIME_DIRS:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    record = {"at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "bundle": bundle, "copied": copied, "absent": absent,
              "sha256": {rel: _digest(os.path.join(out, rel)) for rel in copied}}
    with open(os.path.join(out, SYNCED), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    return record


def sync_and_print(cfg, out=OUT, config_path="config.yaml"):
    """sync, one line on stdout; the hook ops.seal and ops.advance call."""
    rec = sync(cfg, out, config_path)
    if rec is None:
        return None
    print(f"synced {len(rec['copied'])} file(s) -> {out}/ (bundle {rec['bundle']})"
          + (f"; not on disk yet: {', '.join(rec['absent'])}" if rec["absent"] else ""))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ops.integration")
    ap.add_argument("--out", default=None, help=f"default {OUT}/")
    ap.add_argument("--check", action="store_true",
                    help="verbatim copies against their sources, the manifest against "
                         "the disk, every import inside src/; exit 1 on any finding")
    ap.add_argument("--sync", action="store_true",
                    help="copy the artifacts, the data and the config into the folder")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    if args.check:
        r = check(ROOT, args.out)
        for what in ("drift", "missing", "unlisted", "dangling"):
            for item in r[what]:
                print(f"  {what:<9} {item}")
        ok = not (r["drift"] or r["missing"] or r["unlisted"] or r["dangling"])
        print(f"{OUT}/ is {'current' if ok else 'NOT current'}: "
              f"{len(r['curated'])} curated file(s), the rest verbatim")
        return 0 if ok else 1
    if args.sync:
        rec = sync_and_print(load_config(args.config), args.out or OUT, args.config)
        if rec is None:
            print(f"no folder at {args.out or OUT}/")
            return 1
        return 0
    ap.error("pass --check or --sync")


if __name__ == "__main__":
    sys.exit(main())
