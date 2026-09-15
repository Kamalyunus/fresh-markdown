"""Tests for common.provenance and ops.seal (the audit trail: test_history)."""
import json
import pathlib

import pytest

from common import history, provenance
from common.config import config_get
from ops import seal as seal_mod
from conftest import BUNDLE, _write, artifact_at, full_bundle, scratch_paths


@pytest.fixture
def cfg(cfg, tmp_path):
    """A config whose artifact paths all point into a scratch directory."""
    return scratch_paths(cfg, tmp_path)


def test_stamp_records_what_the_artifact_was_fitted_against(cfg):
    payload = provenance.stamp({"rho": 0.31}, cfg, BUNDLE, "test")
    p = payload["provenance"]
    assert p["bundle"] == BUNDLE and p["written_by"] == "test"
    assert p["config_version"] == cfg["meta"]["config_version"]
    assert p["created_at"].endswith("+00:00")     # UTC, not local


def test_a_coherent_set_verifies(cfg):
    full_bundle(cfg)
    state = provenance.verify(cfg)
    assert state["verdict"] == "PASS", state["problems"]
    assert state["bundle"] == BUNDLE
    assert state["missing"] == ["calibration"]    # absent is not inconsistent


def test_mixed_vintages_are_caught(cfg):
    """The whole point: rho fitted against a different model than the prior."""
    full_bundle(cfg)
    artifact_at(cfg, ("dispersion", "rho_path"), {"rho": 0.42},
              bundle="baseline-20250101000000")
    state = provenance.verify(cfg)
    assert state["verdict"] == "FAIL"
    assert any("mixed bundle" in p for p in state["problems"])
    assert state["bundle"] is None                # no single answer to report


def test_an_unstamped_artifact_is_caught(cfg):
    full_bundle(cfg)
    artifact_at(cfg, ("dispersion", "rho_path"), {"rho": 0.31}, stamped=False)
    state = provenance.verify(cfg)
    assert state["verdict"] == "FAIL"
    assert any("no provenance: rho" in p for p in state["problems"])


def test_the_model_file_and_split_manifest_need_no_stamp(cfg):
    """One cannot carry JSON, the other precedes the model. Neither is a fault."""
    full_bundle(cfg)
    assert provenance.verify(cfg)["verdict"] == "PASS"


def test_sealing_then_editing_an_artifact_is_caught(cfg):
    """Provenance alone cannot see this: an editor leaves the stamp intact."""
    full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    assert sealed["bundle"] == BUNDLE
    assert provenance.verify(cfg, sealed)["verdict"] == "PASS"

    path = config_get(cfg, ("dispersion", "rho_path"))
    payload = json.load(open(path))
    payload["rho"] = 0.99                          # stamp untouched
    json.dump(payload, open(path, "w"))

    state = provenance.verify(cfg, sealed)
    assert state["verdict"] == "FAIL"
    assert any("changed since sealing: rho" in p for p in state["problems"])


def test_the_seal_covers_the_environment_not_only_the_artifacts(cfg, tmp_path):
    """A config edit or a library upgrade changes what an hour is priced
    with as surely as an edited artifact, and neither moved a sealed byte.
    The seal records both (and the posterior as it stands); verify reads a
    move as a problem, the same row, on purpose."""
    import copy
    full_bundle(cfg)
    cfg["posterior"]["path"] = str(tmp_path / "posterior.json")
    _write(tmp_path, "posterior", {"cells": {"GLOBAL": {"mean": -1.2, "prior_mean": -1.0,
                                                          "std": 0.4, "n_obs": 0, "version": 0}},
                                   "cell_of": {"MEAT": "GLOBAL"}, "processed_outcome_ids": [],
                                   "cold_start_shift_std": 0.5})
    sealed = seal_mod.seal(cfg)
    env = sealed["environment"]
    assert env["config_digest"] == provenance.config_fingerprint(cfg)["digest"]
    assert set(env["libraries"]) >= {"python", "numpy", "scipy", "pandas", "lightgbm"}
    lp = sealed["launch_posterior"]
    assert lp["cells"]["GLOBAL"]["launch_mean"] == -1.2 and lp["outcomes_consumed"] == 0
    assert sealed["config_snapshot"]["meta"] == cfg["meta"]
    assert provenance.verify(cfg, sealed)["verdict"] == "PASS"

    # config moved: named by key, and the row is FAIL until a deliberate re-seal
    edited = copy.deepcopy(cfg)
    edited["exploration"]["budget_share_of_il"] = 0.5
    state = provenance.verify(edited, sealed)
    assert state["verdict"] == "FAIL"
    assert any(p.startswith("config moved since sealing") and "exploration.budget_share_of_il" in p
               for p in state["problems"])

    # a library moved
    libs = dict(env["libraries"], numpy="0.0.1")
    sealed_lib = dict(sealed, environment=dict(env, libraries=libs))
    assert any("libraries moved since sealing: numpy 0.0.1 ->" in p
               for p in provenance.verify(cfg, sealed_lib)["problems"])

    # the posterior is recorded, never verified: learning moves it by design
    _write(tmp_path, "posterior", {"cells": {"GLOBAL": {"mean": -1.5, "std": 0.3, "n_obs": 9,
                                                          "version": 3}},
                                   "cell_of": {"MEAT": "GLOBAL"}, "processed_outcome_ids": ["x"]})
    assert provenance.verify(cfg, sealed)["verdict"] == "PASS"
    assert seal_mod.seal(cfg)["launch_posterior"]["cells"]["GLOBAL"]["launch_mean"] is None

    # a seal from before the environment record is ONE problem -- read as
    # "no drift" it would never be re-sealed with the record; no seal at all
    # is still silence (nothing to compare against yet)
    legacy = {k: v for k, v in sealed.items() if k not in ("environment", "config_snapshot")}
    state = provenance.verify(edited, legacy)
    assert state["verdict"] == "FAIL"
    assert state["problems"] == [
        "environment not sealed -- re-seal once to record config and libraries"]
    assert provenance.environment_drift(cfg, None) == []
    assert provenance.environment_drift(cfg, {}) == []

    # and the audit MANIFEST carries the record
    cfg["artifacts"]["history_dir"] = str(tmp_path / "history")
    conf = tmp_path / "config.yaml"; conf.write_text("meta: {config_version: t}\n")
    snap = history.archive(cfg, sealed, config_path=str(conf), reason="config")
    manifest = json.load(open(pathlib.Path(snap, "MANIFEST.json")))
    assert manifest["environment"] == env and manifest["launch_posterior"] == lp


def test_seal_refuses_an_inconsistent_set(cfg):
    """A sealed mixed bundle is worse than an unsealed one: it looks decided."""
    full_bundle(cfg)
    artifact_at(cfg, ("dispersion", "rho_path"), {"rho": 0.42}, bundle="other-model")
    with pytest.raises(SystemExit) as exc:
        seal_mod.seal(cfg)
    assert "mixed bundle" in str(exc.value)


def test_seal_refuses_when_there_is_nothing_stamped(cfg):
    with pytest.raises(SystemExit):
        seal_mod.seal(cfg)


def test_a_seal_naming_a_bundle_that_is_gone_is_caught(cfg):
    """Artifacts replaced wholesale by a newer run, seal never refreshed."""
    full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    full_bundle(cfg, bundle="baseline-20270101000000")
    state = provenance.verify(cfg, sealed)
    assert state["verdict"] == "FAIL"
    assert any("is not on disk" in p for p in state["problems"])


def test_config_fingerprint_moves_on_any_value_and_names_what_moved():
    """meta.config_version is a hand-bumped string that tune --apply never
    touches, so two reports run under different tau/rho/W both said '1.0.0'.
    The fingerprint is a digest of the whole config plus the snapshot, so
    status can say WHICH values moved since a report ran."""
    import copy
    from common.config import load_config
    from common.provenance import config_diff, config_fingerprint

    cfg = load_config()
    a = config_fingerprint(cfg, "backtest")
    assert a["phase"] == "backtest" and len(a["digest"]) == 16
    assert a["snapshot"]["pricing"]["tier_step"] == cfg["pricing"]["tier_step"]
    # deterministic
    assert config_fingerprint(cfg, "backtest")["digest"] == a["digest"]

    moved = copy.deepcopy(cfg)
    moved["exploration"]["tau_initial"] = 1234.5
    moved["dispersion"]["rho"] = 0.5
    b = config_fingerprint(moved, "shadow")
    assert b["digest"] != a["digest"]
    assert b["config_version"] == a["config_version"]     # the string did NOT move

    diff = config_diff(a["snapshot"], moved)
    assert any(d.startswith("exploration.tau_initial:") for d in diff)
    assert any(d.startswith("dispersion.rho:") for d in diff)
    assert len(diff) == 2


def test_a_sealed_artifact_that_vanished_or_appeared_is_caught(cfg):
    """verify() compared hashes only for files present, so `rm r_lookup.json`
    after sealing still read PASS; a fit made after sealing was invisible too."""
    import os
    full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    os.remove(config_get(cfg, ("dispersion", "r_lookup_path")))
    v = provenance.verify(cfg, sealed)
    assert v["verdict"] == "FAIL" and any("no longer on disk" in p for p in v["problems"])

    # the other direction: the calibration was ABSENT at sealing and is
    # fitted afterwards -- stamped with the right bundle, hashes all match,
    # and the seal still does not describe what is on disk
    full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    assert "calibration" in sealed["missing"]
    artifact_at(cfg, ("baseline_model", "calibration_factor_path"),
              {"factors": {"A": 1.0}})
    v = provenance.verify(cfg, sealed)
    assert v["verdict"] == "FAIL"
    assert any(p == "fitted after sealing (re-seal): calibration"
               for p in v["problems"])
    assert v["missing"] == []                    # present, just unsealed


def test_a_config_reseal_refuses_an_artifact_that_moved_since_the_previous_seal(cfg, tmp_path):
    """`advance` re-seals under `config`/`libraries` on its own whenever the
    environment drifted. seal() verified with NO prior seal, so a prior
    hand-edited at the same time was blessed as the record under reason
    `config` -- the automatic path erased exactly the red row rule 18 says
    only a deliberate re-seal may clear. A re-seal's reason says which
    artifacts may have moved: none for config/libraries, the calibration
    for weekly-refit; a fit reason or none seals the set as it stands."""
    from common.io import write_json
    full_bundle(cfg)
    artifact_at(cfg, ("baseline_model", "calibration_factor_path"), {"factors": {"A": 1.0}}, BUNDLE)
    write_json(cfg["artifacts"]["bundle_path"], seal_mod.seal(cfg, reason="bootstrap"))

    # the environment moved AND someone edited the prior (stamp intact)
    cfg["exploration"]["budget_share_of_il"] = 0.02
    artifact_at(cfg, ("posterior", "prior", "path"), {"source": "hand-edited"}, BUNDLE)
    assert provenance.verify(cfg)["verdict"] == "PASS"       # stamps alone see nothing
    for reason in ("config", "libraries"):
        with pytest.raises(SystemExit) as exc:
            seal_mod.seal(cfg, reason=reason)
        assert "prior" in str(exc.value) and reason in str(exc.value)
    with pytest.raises(SystemExit):
        seal_mod.seal(cfg, reason="weekly-refit")             # the prior is not the calibration
    # the reason that made the change seals it, deliberately
    assert seal_mod.seal(cfg, reason="check-only")["bundle"] == BUNDLE
    assert seal_mod.seal(cfg)["bundle"] == BUNDLE

    # a weekly re-fit moves the calibration alone: sealed under its reason
    write_json(cfg["artifacts"]["bundle_path"], seal_mod.seal(cfg, reason="check-only"))
    artifact_at(cfg, ("baseline_model", "calibration_factor_path"), {"factors": {"A": 1.1}}, BUNDLE)
    assert seal_mod.seal(cfg, reason="weekly-refit")["bundle"] == BUNDLE
    with pytest.raises(SystemExit) as exc:
        seal_mod.seal(cfg, reason="config")
    assert "calibration" in str(exc.value)
    # nothing moved: a config re-seal is what advance runs
    write_json(cfg["artifacts"]["bundle_path"], seal_mod.seal(cfg, reason="weekly-refit"))
    cfg["exploration"]["budget_share_of_il"] = 0.03
    assert seal_mod.seal(cfg, reason="config")["bundle"] == BUNDLE


def test_a_numpy_bool_is_written_as_a_json_bool(tmp_path):
    """`json_safe` let np.bool_ through to `default=str`, so an artifact flag
    computed with numpy comparison read back as the STRING "False" -- which
    is truthy. One home for JSON out (AGENTS): the fix is there."""
    import numpy as np

    from common.io import json_safe, read_json, write_json

    payload = {"flag": np.bool_(False), "nested": [np.bool_(True), np.int64(3),
                                                  np.float64("nan"), True]}
    safe = json_safe(payload)
    assert safe["flag"] is False and safe["nested"] == [True, 3, None, True]
    path = tmp_path / "a.json"
    write_json(str(path), payload)
    back = read_json(str(path))
    assert back["flag"] is False and back["nested"][0] is True
    assert '"False"' not in path.read_text()
