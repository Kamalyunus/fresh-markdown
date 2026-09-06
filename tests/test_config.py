"""config.yaml as shipped: the hold-out sits after every fitting window,
tau_initial is void until re-derived, strict mode refuses a null measured
key, and a paste left over from a previous retrain is detected."""

import pytest
import yaml

from common.config import load_config


def test_holdout_window_is_after_the_test_window(cfg):
    h, s = cfg["data"]["holdout"], cfg["data"]["split"]
    assert h["start"] > s["test_end"], \
        "a hold-out that overlaps the gate window is not a hold-out"
    assert h["end"] > h["start"]


def test_holdout_is_disjoint_from_every_fitting_window(cfg):
    h, s = cfg["data"]["holdout"], cfg["data"]["split"]
    for lo, hi in [(s["train_start"], s["train_end"]),
                   (s["calib_start"], s["calib_end"]),
                   (s["test_start"], s["test_end"])]:
        assert not (h["start"] <= hi and lo <= h["end"])


def test_config_ships_tau_initial_null(cfg):
    # it is void until re-derived: the scoping fix changed what the backtest
    # produces, so any value carried over from before is wrong
    assert cfg["exploration"]["tau_initial"] is None
    # and shadow can re-derive it: the floor is set, so the derivation runs
    assert cfg["exploration"]["tau0_derivation_min_decisions"] > 0


def test_config_strict_refuses_null_measured(tmp_path):
    from common.config import ConfigError
    with pytest.raises(ConfigError, match="refusing to start"):
        load_config(strict=True)


def test_config_detects_stale_paste_from_frozen_artifact(tmp_path):
    """A config rho left over from a previous retrain silently mis-weights
    every posterior update, because rho and forced-hours set deff."""
    import json
    from common.config import artifact_mirror_drift

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    rho_path = tmp_path / "rho.json"
    cfg["dispersion"]["rho_path"] = str(rho_path)
    cfg["dispersion"]["rho"] = 0.3183

    # no artifact yet -- bootstrap has not run, which is not drift
    assert artifact_mirror_drift(cfg) == []

    rho_path.write_text(json.dumps({"rho": 0.3183}))
    assert artifact_mirror_drift(cfg) == []

    # retrain moved rho; config still holds the old paste
    rho_path.write_text(json.dumps({"rho": 0.2510}))
    drift = artifact_mirror_drift(cfg)
    assert len(drift) == 1 and "dispersion.rho" in drift[0]
    # a --check-only contraction step (~1e-3) is NOT drift: the tolerance is
    # the config's 1% of the frozen rho, above the step the loop takes while
    # settling -- and relative, so it means the same on 0.12 and 0.65
    rho_path.write_text(json.dumps({"rho": 0.3183 + 0.0012}))
    assert artifact_mirror_drift(cfg) == []
    assert artifact_mirror_drift(cfg, tol=5e-4)     # the old value re-pasted every settle
    rho_path.write_text(json.dumps({"rho": 0.3183 * 1.02}))      # 2%: drift
    assert artifact_mirror_drift(cfg)
    cfg["dispersion"]["rho"] = 0.65
    rho_path.write_text(json.dumps({"rho": 0.65 + 0.005}))       # <1% of 0.65
    assert artifact_mirror_drift(cfg) == []
