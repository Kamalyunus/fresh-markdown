"""integration/ -- the standalone pricing folder engineering runs: the
hourly command, the morning table it joins on, and what the two need.

Maintained in place; its code (`pricing/`) is the folder's own, written
for it, not a copy of the repository's modules. Pinned here: every
verbatim copy (the handover page, the examples, the requirements) still
equals its repository source, the generated config equals a fresh prune,
and every file on disk is in the manifest; every import inside the
package resolves inside the package and every module there is reached
by one of the two commands (standalone AND minimal); the package carries
nothing of the learning lane; the commands run from anywhere with the
repository off the path, exploit-only; and the sync carries the five
artifacts, the extract seed and the pruned config. The proof that the
folder's code prices as the repository does -- the same hour through
both, every field of the decision equal -- is in test_end_to_end.py,
where the trained workspace already exists."""
import ast
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
    """A handover page or example changed and not recopied, the config
    pruned from a config that moved, a file added to the folder and not
    listed, a manifested file gone, a module nothing reaches: each is
    named. The package's files are the folder's own and listed as such."""
    r = bi.check(ROOT, FOLDER)
    assert r["drift"] == [], f"folder copies behind their source: {r['drift']}"
    assert r["missing"] == [] and r["unlisted"] == [], r
    assert r["dangling"] == [], f"imports that leave the package: {r['dangling']}"
    assert r["unreached"] == [], f"modules neither command imports: {r['unreached']}"
    assert set(r["own"]) >= {"pricing/hour.py", "pricing/features.py", "pricing/decide.py",
                             "pricing/dp.py", "pricing/log.py", "price_hour.py",
                             "build_features.py", "README.md"}
    assert bi.current(ROOT, FOLDER)


def test_the_package_is_closed_and_carries_nothing_of_the_learning_lane():
    """Standalone: nothing under pricing/ imports a repository package.
    Minimal: no module and no function of the posterior update, the
    exploration draw, the budget, the outcome side of the store, the
    pairing, the seal, the fitter, the checker or the episode-id rule
    rides along. Stateless: no store is read -- the log is append-only
    and has no reader in the package."""
    assert bi.unresolved(FOLDER) == []
    mods = set(bi.modules(FOLDER))
    assert {"pricing.hour", "pricing.features", "pricing.decide", "pricing.dp",
            "pricing.log", "pricing.state", "pricing.model"} <= mods
    assert not {m for m in mods if any(w in m for w in
                                       ("posterior", "explore", "budget", "learn", "outcome",
                                        "pairs", "seal", "train", "calibrate", "check", "store",
                                        "rule"))}
    with open(bi._module_path(FOLDER, "pricing.log")) as f:
        log_tree = ast.parse(f.read())
    public = {n.name for c in ast.walk(log_tree) if isinstance(c, ast.ClassDef) and c.name == "EventLog"
              for n in c.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    assert public == {"append"}, public
    names = set()
    for mod in mods:
        with open(bi._module_path(FOLDER, mod)) as f:
            names |= {n.name for n in ast.walk(ast.parse(f.read()))
                      if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert not names & {"commit_update", "initialise", "suspend_exploration", "commit_tau",
                        "emit_outcome", "load_decisions", "match_pairs", "quality_rates",
                        "draw", "spread_table", "budget_today", "train", "verify", "seal",
                        "split_frames", "continues", "latest_by_shelf", "episode_paths",
                        "priced_hours", "_consume", "_quarantine"}, names
    for repo_pkg in ("common", "engine", "events", "fit", "daily", "ops", "evaluate"):
        for mod in mods:
            with open(bi._module_path(FOLDER, mod)) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(a.name.split(".")[0] != repo_pkg for a in node.names), (mod, repo_pkg)
                elif isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] != repo_pkg, (mod, repo_pkg)


def test_the_pruned_config_carries_only_what_the_hour_reads(cfg):
    """The generated config is the repository's pruned to KEEP: the switch,
    the horizon cap, the anchors, the artifact paths, the morning table's
    three, the grid, the log -- and none of the training, harness,
    exploration or learning keys. KEEP and the code agree both ways."""
    with open(os.path.join(FOLDER, "config.yaml")) as f:
        folder_cfg = yaml.safe_load(f)
    assert folder_cfg == bi.prune(cfg)
    assert set(folder_cfg) == {"meta", "data", "reference_discount", "baseline_model",
                               "dispersion", "posterior", "pricing", "events", "features"}
    assert set(folder_cfg["data"]) == {"launch_date", "max_window_hours"}
    assert set(folder_cfg["posterior"]) == {"path"} and set(folder_cfg["dispersion"]) == {"r_lookup_path"}
    assert set(folder_cfg["baseline_model"]) == {"model_path", "feature_schema_path",
                                                 "calibration_factor_path",
                                                 "ref_rate_window_days", "ref_rate_anchor_band"}
    for absent in ("exploration", "learning", "monitoring", "tuning", "artifacts", "assurance"):
        assert absent not in folder_cfg
    # every two-level read in the package is a kept key, and every kept
    # key is named somewhere in the package (a key nothing reads is pruned)
    import re
    source = ""
    for mod in bi.modules(FOLDER):
        with open(bi._module_path(FOLDER, mod)) as f:
            source += f.read()
    for section, key in re.findall(r'cfg\["([a-z_]+)"\]\["([a-z_]+)"\]', source):
        assert (section, key) in bi.KEEP or (section,) in bi.KEEP, (section, key)
    for path in bi.KEEP:
        assert f'"{path[-1]}"' in source, f"{'.'.join(path)} is kept but nothing reads it"


def test_the_command_runs_from_anywhere_with_the_repository_off_the_path(tmp_path):
    """Copied elsewhere and called from a third directory: the command
    chdirs into the folder, imports from the copy alone, is exploit only
    by construction (nothing drawn: the applied price IS the optimal
    price, tau and the budget absent from the event), and its loader's
    one gate is the launch date."""
    folder = str(tmp_path / "pricing_folder")
    shutil.copytree(FOLDER, folder, ignore=shutil.ignore_patterns("__pycache__"))
    elsewhere = str(tmp_path / "elsewhere")
    os.makedirs(elsewhere)
    r = _isolated(folder, "-c", (
        "import os, sys\n"
        "import pricing.hour, pricing.features\n"
        "here = os.getcwd()\n"
        "bad = [m.__name__ for m in list(sys.modules.values())\n"
        "       if getattr(m, '__file__', None) and m.__name__.split('.')[0] in %r\n"
        "       and not os.path.abspath(m.__file__).startswith(here)]\n"
        "assert not bad, bad\n"
        "import inspect\n"
        "from pricing import decide\n"
        "src = inspect.getsource(decide.decide)\n"
        "assert '\"is_exploration\": False' in src and '\"tau_current\": None' in src, src\n"
        "from pricing.config import load_config, ConfigError\n"
        "import yaml\n"
        "with open('config.yaml') as f: cfg = yaml.safe_load(f)\n"
        "cfg['data']['launch_date'] = None\n"
        "with open('c.yaml', 'w') as f: yaml.safe_dump(cfg, f)\n"
        "try:\n"
        "    load_config('c.yaml', strict=True)\n"
        "except ConfigError as e:\n"
        "    assert 'data.launch_date' in str(e), e\n"
        "else:\n"
        "    raise AssertionError('a null launch date priced')\n") % (bi.PACKAGES,))
    assert r.returncode == 0, r.stdout + r.stderr
    r = _isolated(elsewhere, os.path.join(folder, "price_hour.py"), "--help")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--snapshot" in r.stdout and "--features" in r.stdout and "--dry-run" in r.stdout
    r = _isolated(elsewhere, os.path.join(folder, "build_features.py"), "--help")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--feed" in r.stdout and "--as-of" in r.stdout


def test_the_sync_carries_the_five_artifacts_the_seed_and_the_pruned_config(
        cfg, tmp_path, monkeypatch):
    """What the owner's chain calls after every seal and every advance run:
    the five artifacts the hour opens, the extract the first morning
    seeds from, and the config pruned to what the two commands read; one
    not on disk yet is listed, never an error; no folder means no sync."""
    folder = str(tmp_path / "pricing_folder")
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
    os.makedirs("data")
    with open(os.path.join("data", "flc_raw.parquet"), "wb") as f:
        f.write(b"not really parquet")
    rec = bi.sync(c, out=folder)
    assert set(rec["copied"]) == {"artifacts/r_lookup.json", "artifacts/posterior.json",
                                  "data/flc_raw.parquet", "config.yaml"}
    assert set(rec["absent"]) == {"baseline_model.model_path",
                                  "baseline_model.feature_schema_path",
                                  "baseline_model.calibration_factor_path"}
    with open(os.path.join(folder, "artifacts", "synced.json")) as f:
        assert json.load(f)["copied"] == rec["copied"]
    with open(os.path.join(folder, "config.yaml")) as f:
        assert yaml.safe_load(f) == bi.prune(c)


def test_the_folder_carries_the_handoff():
    for rel in ("README.md", "MANIFEST.json", "config.yaml", "requirements.lock",
                "examples/README.md", "docs/engineering_handover.html",
                "price_hour.py", "build_features.py", "pricing/__init__.py"):
        assert os.path.exists(os.path.join(FOLDER, rel)), rel
    for gone in ("src", "check_inputs.py", "assign_episode_ids.py"):
        assert not os.path.exists(os.path.join(FOLDER, gone)), gone
    with open(os.path.join(FOLDER, "MANIFEST.json")) as f:
        m = json.load(f)
    assert m["commands"] == ["price_hour.py", "build_features.py"] and m["code"] == bi.CODE
