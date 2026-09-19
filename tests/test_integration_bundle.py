"""integration/ -- the standalone pricing folder engineering runs.

Pinned here: the committed folder IS a fresh build (no hand edit
survives), every repo-local import inside it resolves inside it (it is
standalone), its three commands run from anywhere with the repository
off the path, and the sync the owner's chain calls carries the artifacts
into it. The hour that prices from the folder against real artifacts,
synced by a real seal, is in test_end_to_end.py where the trained
workspace already exists."""
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


def test_the_committed_folder_is_a_fresh_build():
    """A file edited under integration/ by hand, or a module changed in its
    one home and not rebuilt, differs from a fresh build -- and is named."""
    assert bi.stale(ROOT, FOLDER) == [], \
        "integration/ is stale: python3 -m ops.integration"


def test_every_local_import_inside_the_folder_resolves_inside_it():
    """Standalone means the closure is closed: nothing under src/ imports
    a repo module the folder does not carry, and src/ carries nothing the
    three commands do not reach. The id rule rides along because the
    hourly job counts disagreements with it; it is not a command."""
    src = os.path.join(FOLDER, bi.SRC)
    mods = bi.closure(src)                          # resolved against src/
    for mod in mods:
        for dep in bi.local_imports(src, bi._module_path(src, mod)):
            assert bi._module_path(src, dep), f"{mod} imports {dep}, absent from src/"
    present = {os.path.relpath(os.path.join(dp, f), src)
               for dp, _, fs in os.walk(src) for f in fs
               if f.endswith(".py") and "__pycache__" not in dp}
    assert present == set(bi.module_files(src, mods))
    assert "ops.assign_episode_ids" in mods and "ops.assign_episode_ids" not in bi.ENTRIES
    assert not os.path.exists(os.path.join(FOLDER, "assign_episode_ids.py"))


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
        "assert not bad, bad\n") % (bi.PACKAGES,))
    assert r.returncode == 0, r.stdout + r.stderr
    for name, _, _ in bi.COMMANDS:
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
    an error; no built folder means no sync."""
    folder = str(tmp_path / "pricing")
    assert bi.sync(cfg, out=folder) is None
    bi.build(ROOT, folder)
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
    # a rebuild keeps what the sync put there
    bi.build(ROOT, folder)
    assert os.path.exists(os.path.join(folder, "artifacts", "rho.json"))


def test_the_folder_carries_the_handoff_and_refuses_to_replace_a_stranger(tmp_path):
    for rel in ("README.md", "MANIFEST.json", "config.yaml", "requirements.lock",
                "examples/README.md", "docs/engineering_handover.html",
                "price_hour.py", "build_features.py", "check_inputs.py"):
        assert os.path.exists(os.path.join(FOLDER, rel)), rel
    for d in bi.RUNTIME_DIRS:
        assert os.path.isdir(os.path.join(FOLDER, d)), d
    stranger = tmp_path / "mine"
    stranger.mkdir()
    (stranger / "notes.txt").write_text("keep me")
    try:
        bi.build(ROOT, str(stranger))
    except SystemExit as exc:
        assert "refusing" in str(exc)
    else:
        raise AssertionError("built over a directory that was not a previous build")
    assert (stranger / "notes.txt").read_text() == "keep me"
