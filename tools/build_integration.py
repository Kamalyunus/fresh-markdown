"""tools.build_integration -- the standalone folder engineering runs.

`integration/` is what the pricing host needs and nothing else: the exact
import closure of the four entry points below, copied verbatim with the
package layout kept (so `python3 -m ops.price_hour` runs there unchanged
and no import is rewritten), the config, the pinned requirements, the
example files, the handover page, and a README that is the handoff. The
learning lane (`ops.advance --feed`, `daily.update`, the monitor) is NOT
in it: this folder PRICES; the owner's daily lane in the repository reads
the event store it writes and re-writes the posterior it reads.

The folder is GENERATED. Never edit a file under integration/ by hand: a
change belongs in the module's one home and the next build carries it
over. `tests/test_integration_bundle.py` pins that the committed folder is
byte-identical to a fresh build, that every local import inside it
resolves inside it, and that the hourly job prices an hour with the
repository off the path.

Run: python3 -m tools.build_integration [--out integration] [--check]
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "integration"
PACKAGES = ("common", "engine", "events", "fit", "daily", "ops")

# what engineering runs, in the order the README gives it
ENTRIES = ("ops.price_hour",           # every clock hour: the snapshot in, a price per shelf out
           "daily.features",           # every morning: the day's feature table from the feed
           "ops.check_inputs",         # before launch: their three tables against the contract
           "ops.assign_episode_ids")   # the episode-id rule as a script they run or port

# copied as they are, beside the packages
FILES = ("config.yaml", "requirements.txt", "requirements.lock",
         "docs/engineering_handover.html")
DIRS = ("examples",)
README_SRC = os.path.join("tools", "integration_README.md")

# the working directories the handover's cron lines name; created empty
# (the repo ignores their contents) so the first run finds them
RUNTIME_DIRS = ("snapshots", "feed", "failures", "features", "decisions",
                "reports/hours", "logs", "events_store", "artifacts", "data")

MANIFEST = "MANIFEST.json"


# ---------------------------------------------------------------- closure

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


def closure(root, entries=ENTRIES):
    """Every repo-local module reachable from `entries`, sorted."""
    seen, todo = set(), list(entries)
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        path = _module_path(root, mod)
        if path is None:
            raise SystemExit(f"cannot resolve {mod} under {root}")
        seen.add(mod)
        todo.extend(local_imports(root, path) - seen)
    return sorted(seen)


def module_files(root, mods):
    """The files that carry `mods`: each module and every package
    __init__ above it, relative to `root`."""
    files = set()
    for mod in mods:
        files.add(os.path.relpath(_module_path(root, mod), root))
        parts = mod.split(".")
        for i in range(1, len(parts)):
            init = os.path.join(*parts[:i], "__init__.py")
            if os.path.isfile(os.path.join(root, init)):
                files.add(init)
    return sorted(files)


# ------------------------------------------------------------------ build

def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(root, rel, out):
    dst = os.path.join(out, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(os.path.join(root, rel), dst)
    return rel


def build(root=ROOT, out=None):
    """Write the folder at `out` (default <root>/integration). Returns the
    manifest. Refuses to replace a directory that is not a previous build
    (no MANIFEST.json in it) and not empty."""
    out = out or os.path.join(root, OUT)
    if os.path.isdir(out) and os.listdir(out) \
            and not os.path.exists(os.path.join(out, MANIFEST)):
        raise SystemExit(f"{out} exists and is not a previous build -- refusing to replace it")
    keep = {d.split("/")[0] for d in RUNTIME_DIRS}      # the host's state stays
    if os.path.isdir(out):
        for name in os.listdir(out):
            if name in keep:
                continue
            p = os.path.join(out, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(out, exist_ok=True)

    mods = closure(root)
    copied = [_copy(root, rel, out) for rel in module_files(root, mods)]
    for rel in FILES:
        copied.append(_copy(root, rel, out))
    for d in DIRS:
        for name in sorted(os.listdir(os.path.join(root, d))):
            copied.append(_copy(root, os.path.join(d, name), out))
    shutil.copyfile(os.path.join(root, README_SRC), os.path.join(out, "README.md"))
    copied.append("README.md")
    for d in RUNTIME_DIRS:
        os.makedirs(os.path.join(out, d), exist_ok=True)

    manifest = {
        "what": "the standalone pricing folder; GENERATED by tools.build_integration, never edited",
        "entries": list(ENTRIES),
        "modules": mods,
        "files": {rel: _digest(os.path.join(out, rel)) for rel in sorted(copied)},
    }
    with open(os.path.join(out, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return manifest


def stale(root=ROOT, out=None):
    """The files that differ between `out` and a fresh build: [] when the
    committed folder is current."""
    out = out or os.path.join(root, OUT)
    with tempfile.TemporaryDirectory() as tmp:
        want = build(root, os.path.join(tmp, "fresh"))["files"]
    have = {}
    if os.path.exists(os.path.join(out, MANIFEST)):
        with open(os.path.join(out, MANIFEST)) as f:
            have = json.load(f).get("files") or {}

    def on_disk(rel):
        p = os.path.join(out, rel)
        return _digest(p) if os.path.exists(p) else None

    # a file the fresh build has whose bytes differ (or is missing), a file
    # the folder's manifest lists that the build no longer carries
    return sorted(rel for rel in set(want) | set(have)
                  if want.get(rel) is None or on_disk(rel) != want[rel])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.build_integration")
    ap.add_argument("--out", default=None, help=f"default {OUT}/ under the repo root")
    ap.add_argument("--check", action="store_true",
                    help="compare the folder to a fresh build; exit 1 if it is stale")
    args = ap.parse_args(argv)
    if args.check:
        diff = stale(ROOT, args.out)
        if diff:
            print("integration/ is STALE -- rebuild with python3 -m tools.build_integration:")
            for rel in diff:
                print(f"  {rel}")
            return 1
        print("integration/ is current")
        return 0
    m = build(ROOT, args.out)
    print(f"{len(m['modules'])} modules, {len(m['files'])} files -> {args.out or OUT}/")
    for mod in m["modules"]:
        print(f"  {mod}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
