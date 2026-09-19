"""integration/ -- the standalone pricing folder engineering runs.

Three properties, pinned: the committed folder IS a fresh build (no hand
edit survives), every repo-local import inside it resolves inside it (it
is standalone), and its scripts run from it with the repository off the
path. The hour that prices against real artifacts is in
test_end_to_end.py, where the trained workspace already exists."""
import os
import shutil
import subprocess
import sys

from conftest import ROOT
from tools import build_integration as bi

FOLDER = os.path.join(ROOT, "integration")


def _isolated(cwd, *args):
    """Run `python3 *args` from `cwd` with the repository off the path:
    no PYTHONPATH, and the interpreter's own cwd entry pointing at `cwd`."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_the_committed_folder_is_a_fresh_build():
    """A file edited under integration/ by hand, or a module changed in its
    one home and not rebuilt, differs from a fresh build -- and is named."""
    assert bi.stale(ROOT, FOLDER) == [], \
        "integration/ is stale: python3 -m tools.build_integration"


def test_every_local_import_inside_the_folder_resolves_inside_it():
    """Standalone means the closure is closed: nothing under integration/
    imports a repo module the folder does not carry, and the folder
    carries nothing the entry points do not reach."""
    mods = bi.closure(FOLDER)                     # resolved against the FOLDER
    for mod in mods:
        path = bi._module_path(FOLDER, mod)
        for dep in bi.local_imports(FOLDER, path):
            assert bi._module_path(FOLDER, dep), f"{mod} imports {dep}, absent from integration/"
    present = {os.path.relpath(os.path.join(dp, f), FOLDER)
               for dp, _, fs in os.walk(FOLDER) for f in fs
               if f.endswith(".py") and "__pycache__" not in dp}
    assert present == set(bi.module_files(FOLDER, mods))


def test_the_scripts_run_from_the_folder_with_the_repository_off_the_path(tmp_path):
    """Copied elsewhere, the four entry points import from the copy alone
    (every loaded repo module's file is under it), the checker passes the
    bundled examples, and the id rule assigns the example snapshot."""
    folder = str(tmp_path / "pricing")
    shutil.copytree(FOLDER, folder, ignore=shutil.ignore_patterns("__pycache__"))
    r = _isolated(folder, "-c", (
        "import os, sys\n"
        "import ops.price_hour, daily.features, ops.check_inputs, ops.assign_episode_ids\n"
        "here = os.getcwd()\n"
        "bad = [m.__name__ for m in list(sys.modules.values())\n"
        "       if getattr(m, '__file__', None)\n"
        "       and m.__name__.split('.')[0] in %r\n"
        "       and not os.path.abspath(m.__file__).startswith(here)]\n"
        "assert not bad, bad\n"
        "print(len([m for m in sys.modules if m.split('.')[0] in %r]))"
    ) % (bi.PACKAGES, bi.PACKAGES))
    assert r.returncode == 0, r.stdout + r.stderr
    r = _isolated(folder, "-m", "ops.check_inputs",
                  "--snapshot", "examples/snapshot_2026-08-29T11.csv",
                  "--feed", "examples/feed_2026-08-29.parquet",
                  "--failures", "examples/failures_2026-08-29.csv")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 FAIL" in r.stdout, r.stdout
    r = _isolated(folder, "-m", "ops.assign_episode_ids",
                  "--hour", "examples/snapshot_2026-08-29T11.csv",
                  "--out", "snapshots/with_ids.csv")
    assert r.returncode == 0, r.stdout + r.stderr
    assert os.path.exists(os.path.join(folder, "snapshots", "with_ids.csv"))


def test_the_folder_carries_the_handoff_and_refuses_to_replace_a_stranger(tmp_path):
    """The README, the manifest, the config, the pinned requirements, the
    examples and the handover page ride along; the working directories
    exist empty; and a directory that is not a previous build is never
    emptied by a build."""
    for rel in ("README.md", "MANIFEST.json", "config.yaml", "requirements.lock",
                "examples/README.md", "docs/engineering_handover.html"):
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
