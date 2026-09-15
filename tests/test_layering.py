"""The import graph is the package map (AGENTS repo conventions): the
lane and the engine never load a harness, and the engine never loads a
driver. A re-export is how a moved name keeps its old dotted path -- it
must not carry the new home's whole import tree with it."""
import subprocess
import sys

from conftest import ROOT


def _loaded_after(imports):
    code = ("import sys, " + ", ".join(imports) + "; "
            "print(' '.join(sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('evaluate', 'ops', 'tools', 'lightgbm'))))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                       text=True, env={"PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr
    return set(r.stdout.split())


def test_the_lanes_definitions_do_not_load_a_harness():
    """common.metrics once re-exported fidelity_decomposition from
    evaluate.backtest: every monitor start loaded the backtest harness and
    LightGBM, and a cycle waited one import away."""
    loaded = _loaded_after(["common.metrics", "events.frame", "daily.monitor"])
    assert not {m for m in loaded if m.startswith(("evaluate", "lightgbm"))}, loaded


def test_the_engine_does_not_load_a_driver():
    """engine.explore once re-exported the tau paste gate from
    ops.config_keys -- an engine -> ops edge for a name nothing reached
    through the engine."""
    loaded = _loaded_after(["engine.explore", "engine.decide", "engine.state"])
    assert not {m for m in loaded if m.startswith(("ops", "evaluate", "tools"))}, loaded
