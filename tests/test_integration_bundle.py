"""integration/ -- the standalone pricing folder engineering runs.

Maintained in place, never generated. Pinned here: every `verbatim` copy
in it still equals its repository source and every file on disk is in
its manifest (a fix to the engine reaches the folder or fails here);
every repo-local import inside src/ resolves inside src/ (it is
standalone); its three commands run from anywhere with the repository
off the path, exploit-only; and the sync the owner's chain calls carries
the artifacts into it. The hour that prices from the folder against
real artifacts, synced by a real seal, is in test_end_to_end.py where
the trained workspace already exists."""
import json
import os
import shutil
import subprocess
import sys

from conftest import ROOT, scratch_paths
from ops import integration as bi

FOLDER = os.path.join(ROOT, "integration")


def _isolated(cwd, *args):
    """Run `python3 *args` from `cwd` with the repository off the path."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_every_verbatim_copy_equals_its_source_and_the_manifest_is_the_disk():
    """A repository module changed and not ported, a file added to the
    folder and not listed, a manifested file gone: each is named. The
    curated files are the short list the manifest explains."""
    r = bi.check(ROOT, FOLDER)
    assert r["drift"] == [], f"folder copies behind their source: {r['drift']}"
    assert r["missing"] == [] and r["unlisted"] == [], r
    assert set(r["curated"]) >= {"src/fit/model.py", "src/ops/price_batch.py",
                                 "src/engine/explore.py", "src/daily/failures.py"}
    assert bi.current(ROOT, FOLDER)


def test_every_local_import_inside_src_resolves_inside_src():
    """Standalone means the closure is closed: nothing under src/ imports a
    repository module the folder does not carry. The id rule rides along
    because the hourly job counts disagreements with it; the trainer, the
    fitters, the harnesses and the learning lane do not."""
    src = os.path.join(FOLDER, bi.SRC)
    assert bi.unresolved(src) == []
    mods = set(bi.modules(src))
    assert "ops.assign_episode_ids" in mods
    for absent in ("fit.train_baseline", "fit.calibrate", "fit.fit_dispersion",
                   "engine.budget", "engine.spread_ledger", "engine.learn",
                   "daily.update", "daily.ingest_outcomes", "daily.monitor",
                   "common.history", "common.clustering", "ops.advance", "ops.tune"):
        assert absent not in mods, absent


def test_the_commands_run_from_anywhere_with_the_repository_off_the_path(tmp_path):
    """Copied elsewhere and called from a third directory: each command
    chdirs into the folder (a relative argument is relative to it), imports
    from the copy alone, and the checker passes the bundled examples and
    the example response against its snapshot."""
    folder = str(tmp_path / "pricing")
    shutil.copytree(FOLDER, folder, ignore=shutil.ignore_patterns("__pycache__"))
    elsewhere = str(tmp_path / "elsewhere")
    os.makedirs(elsewhere)
    r = _isolated(os.path.join(folder, bi.SRC), "-c", (
        "import os, sys\n"
        "import ops.price_hour, daily.features, ops.check_inputs\n"
        "here = os.getcwd()\n"
        "bad = [m.__name__ for m in list(sys.modules.values())\n"
        "       if getattr(m, '__file__', None) and m.__name__.split('.')[0] in %r\n"
        "       and not os.path.abspath(m.__file__).startswith(here)]\n"
        "assert not bad, bad\n"
        "from ops.price_batch import EXPLOIT_ONLY\n"
        "assert EXPLOIT_ONLY is True\n") % (bi.PACKAGES,))
    assert r.returncode == 0, r.stdout + r.stderr
    for name in bi.COMMANDS:
        r = _isolated(elsewhere, os.path.join(folder, name), "--help")
        assert r.returncode == 0, name + "\n" + r.stdout + r.stderr
    r = _isolated(elsewhere, os.path.join(folder, "check_inputs.py"),
                  "--snapshot", "examples/snapshot_2026-08-29T11.csv",
                  "--feed", "examples/feed_2026-08-29.parquet",
                  "--failures", "examples/failures_2026-08-29.csv",
                  "--response", "examples/decisions_2026-08-29T11.csv",
                  "--report", "reports/hours/examples.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 FAIL" in r.stdout, r.stdout
    assert os.path.exists(os.path.join(folder, "reports", "hours", "examples.json"))


def test_the_sync_carries_the_artifacts_the_config_names(cfg, tmp_path, monkeypatch):
    """What the owner's chain calls after every seal and every advance run:
    the artifacts the config names, the extract and the feed history, and
    the config land in the folder; one not on disk yet is listed, never
    an error; no folder means no sync."""
    folder = str(tmp_path / "pricing")
    assert bi.sync(cfg, out=folder) is None
    shutil.copytree(FOLDER, folder, ignore=shutil.ignore_patterns("__pycache__"))
    c = scratch_paths(cfg, tmp_path)
    for key in (("dispersion", "rho_path"), ("posterior", "prior", "path")):
        path = c
        for k in key:
            path = path[k]
        with open(path, "w") as f:
            json.dump({"key": ".".join(key)}, f)
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    with open(os.path.join("data", "flc_raw.parquet"), "wb") as f:
        f.write(b"not really parquet")
    rec = bi.sync(c, out=folder, config_path=os.path.join(ROOT, "config.yaml"))
    assert set(rec["copied"]) == {"artifacts/rho.json", "artifacts/prior.json",
                                  "data/flc_raw.parquet", "config.yaml"}
    assert "posterior.path" in rec["absent"] and "baseline_model.model_path" in rec["absent"]
    assert rec["bundle"] is None                      # no bundle.json sealed yet
    with open(os.path.join(folder, "artifacts", "synced.json")) as f:
        assert json.load(f)["copied"] == rec["copied"]
    with open(os.path.join(folder, "artifacts", "rho.json")) as f:
        assert json.load(f) == {"key": "dispersion.rho_path"}


def test_the_folder_carries_the_handoff():
    for rel in ("README.md", "MANIFEST.json", "config.yaml", "requirements.lock",
                "examples/README.md", "docs/engineering_handover.html", *bi.COMMANDS):
        assert os.path.exists(os.path.join(FOLDER, rel)), rel
    with open(os.path.join(FOLDER, "MANIFEST.json")) as f:
        m = json.load(f)
    assert m["commands"] == list(bi.COMMANDS)
    assert all(e["from"] or e.get("curated") for e in m["files"].values())
