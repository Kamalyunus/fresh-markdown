"""integration/ -- the standalone pricing folder engineering runs: the
hourly command and what it needs, nothing else.

Maintained in place, never generated wholesale. Pinned here: every
`verbatim` copy still equals its repository source, the generated config
equals a fresh prune, and every file on disk is in the manifest (a fix to
the engine reaches the folder or fails here); every repo-local import
inside src/ resolves inside src/ and every module there is reached by
the one command (standalone AND minimal); the command runs from anywhere
with the repository off the path, exploit-only; and the sync carries the
five artifacts, the feature tables and the pruned config. The hour that
prices from the folder against real artifacts, synced by a real seal,
is in test_end_to_end.py where the trained workspace already exists."""
import json
import os
import shutil
import subprocess
import sys

import yaml

from conftest import ROOT, scratch_paths
from ops import integration as bi

FOLDER = os.path.join(ROOT, "integration")


def _isolated(cwd, *args):
    """Run `python3 *args` from `cwd` with the repository off the path."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_every_copy_matches_its_source_and_the_manifest_is_the_disk():
    """A repository module changed and not ported, the config pruned from a
    config that moved, a file added to the folder and not listed, a
    manifested file gone, a module nothing reaches: each is named."""
    r = bi.check(ROOT, FOLDER)
    assert r["drift"] == [], f"folder copies behind their source: {r['drift']}"
    assert r["missing"] == [] and r["unlisted"] == [], r
    assert r["unreached"] == [], f"modules the hourly command never imports: {r['unreached']}"
    assert set(r["curated"]) >= {"src/fit/model.py", "src/ops/price_batch.py",
                                 "src/engine/explore.py", "src/common/config.py"}
    assert bi.current(ROOT, FOLDER)


def test_the_closure_is_closed_and_is_the_hourly_command_alone():
    """Standalone: nothing under src/ imports a repository module the
    folder does not carry. Minimal: no trainer, fitter, harness, checker,
    feature builder or learning-lane module rides along. The id rule does,
    because the hourly job counts disagreements with it."""
    src = os.path.join(FOLDER, bi.SRC)
    assert bi.unresolved(src) == []
    mods = set(bi.modules(src))
    assert "ops.assign_episode_ids" in mods and "ops.price_hour" in mods
    for absent in ("fit.train_baseline", "fit.calibrate", "fit.fit_dispersion",
                   "engine.budget", "engine.spread_ledger", "engine.learn",
                   "daily.features", "daily.update", "daily.ingest_outcomes",
                   "ops.check_inputs", "common.history", "common.clustering",
                   "common.cli", "common.paths", "ops.advance", "ops.tune"):
        assert absent not in mods, absent


def test_the_pruned_config_carries_only_what_the_hour_reads(cfg):
    """The generated config is the repository's pruned to KEEP: the switch,
    the anchors, the artifact paths, the floor's inputs, the grid, the
    store -- and none of the training, harness or learning keys."""
    with open(os.path.join(FOLDER, "config.yaml")) as f:
        folder_cfg = yaml.safe_load(f)
    assert folder_cfg == bi.prune(cfg)
    assert set(folder_cfg) == {"meta", "data", "reference_discount", "baseline_model",
                               "dispersion", "posterior", "exploration", "pricing", "events"}
    assert set(folder_cfg["data"]) == {"launch_date", "max_window_hours"}
    assert "rho" not in folder_cfg["dispersion"] and "tau_initial" not in folder_cfg["exploration"]
    assert "prior" not in folder_cfg["posterior"]
    for absent in ("learning", "monitoring", "tuning", "features", "artifacts", "assurance"):
        assert absent not in folder_cfg


def test_the_command_runs_from_anywhere_with_the_repository_off_the_path(tmp_path):
    """Copied elsewhere and called from a third directory: the command
    chdirs into the folder, imports from the copy alone, and is exploit
    only by construction."""
    folder = str(tmp_path / "pricing")
    shutil.copytree(FOLDER, folder, ignore=shutil.ignore_patterns("__pycache__"))
    elsewhere = str(tmp_path / "elsewhere")
    os.makedirs(elsewhere)
    r = _isolated(os.path.join(folder, bi.SRC), "-c", (
        "import os, sys\n"
        "import ops.price_hour\n"
        "here = os.getcwd()\n"
        "bad = [m.__name__ for m in list(sys.modules.values())\n"
        "       if getattr(m, '__file__', None) and m.__name__.split('.')[0] in %r\n"
        "       and not os.path.abspath(m.__file__).startswith(here)]\n"
        "assert not bad, bad\n"
        "from ops.price_batch import EXPLOIT_ONLY\n"
        "assert EXPLOIT_ONLY is True\n"
        "from common.config import RUNTIME_REQUIRED\n"
        "assert RUNTIME_REQUIRED == [('data', 'launch_date')]\n") % (bi.PACKAGES,))
    assert r.returncode == 0, r.stdout + r.stderr
    r = _isolated(elsewhere, os.path.join(folder, "price_hour.py"), "--help")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--snapshot" in r.stdout and "--features" in r.stdout


def test_the_sync_carries_the_five_artifacts_the_tables_and_the_pruned_config(
        cfg, tmp_path, monkeypatch):
    """What the owner's chain calls after every seal and every advance run:
    the five artifacts the hour opens, every feature table, and the config
    pruned to what the hour reads; one not on disk yet is listed, never
    an error; no folder means no sync."""
    folder = str(tmp_path / "pricing")
    assert bi.sync(cfg, out=folder) is None
    shutil.copytree(FOLDER, folder, ignore=shutil.ignore_patterns("__pycache__"))
    c = scratch_paths(cfg, tmp_path)
    for key in (("dispersion", "r_lookup_path"), ("posterior", "path")):
        path = c
        for k in key:
            path = path[k]
        with open(path, "w") as f:
            json.dump({"key": ".".join(key)}, f)
    monkeypatch.chdir(tmp_path)
    os.makedirs("features")
    with open(os.path.join("features", "2026-09-01.parquet"), "wb") as f:
        f.write(b"not really parquet")
    rec = bi.sync(c, out=folder)
    assert set(rec["copied"]) == {"artifacts/r_lookup.json", "artifacts/posterior.json",
                                  "features/2026-09-01.parquet", "config.yaml"}
    assert set(rec["absent"]) == {"baseline_model.model_path",
                                  "baseline_model.feature_schema_path",
                                  "baseline_model.calibration_factor_path"}
    with open(os.path.join(folder, "artifacts", "synced.json")) as f:
        assert json.load(f)["copied"] == rec["copied"]
    with open(os.path.join(folder, "config.yaml")) as f:
        assert yaml.safe_load(f) == bi.prune(c)


def test_the_folder_carries_the_handoff():
    for rel in ("README.md", "MANIFEST.json", "config.yaml", "requirements.lock",
                "examples/README.md", "docs/engineering_handover.html", "price_hour.py"):
        assert os.path.exists(os.path.join(FOLDER, rel)), rel
    for gone in ("build_features.py", "check_inputs.py", "src/daily", "src/ops/check_inputs.py"):
        assert not os.path.exists(os.path.join(FOLDER, gone)), gone
    with open(os.path.join(FOLDER, "MANIFEST.json")) as f:
        m = json.load(f)
    assert m["commands"] == ["price_hour.py"]
