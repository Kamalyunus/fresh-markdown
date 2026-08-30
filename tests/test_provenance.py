"""Tests for common.provenance and bootstrap.seal."""
import copy
import json

import pytest

from common.config import load_config
from common import provenance
from bootstrap import seal as seal_mod


BUNDLE = "baseline-20260101000000"


@pytest.fixture
def cfg(tmp_path):
    """A config whose artifact paths all point into a scratch directory."""
    c = copy.deepcopy(load_config())
    c["artifacts"]["bundle_path"] = str(tmp_path / "bundle.json")
    c["data"]["split_manifest_path"] = str(tmp_path / "split_manifest.json")
    c["baseline_model"]["model_path"] = str(tmp_path / "baseline_model.txt")
    c["baseline_model"]["feature_schema_path"] = str(tmp_path / "feature_schema.json")
    c["baseline_model"]["calibration_factor_path"] = str(tmp_path / "calibration.json")
    c["dispersion"]["r_lookup_path"] = str(tmp_path / "r_lookup.json")
    c["dispersion"]["rho_path"] = str(tmp_path / "rho.json")
    c["posterior"]["prior"]["path"] = str(tmp_path / "prior.json")
    return c


def _write(cfg, key, payload, bundle=BUNDLE, stamped=True):
    path = provenance._path(cfg, key)
    if stamped:
        provenance.stamp(payload, cfg, bundle, "test")
    with open(path, "w") as f:
        json.dump(payload, f)


def _full_bundle(cfg, bundle=BUNDLE):
    """One coherent set: model, its schema, and everything fitted against it."""
    with open(provenance._path(cfg, ("baseline_model", "model_path")), "w") as f:
        f.write("tree { }")                      # a model file, not JSON
    _write(cfg, ("data", "split_manifest_path"), {"split": {}}, bundle=None)
    _write(cfg, ("baseline_model", "feature_schema_path"), {"model_version": bundle},
           stamped=False)                        # names its model the old way
    _write(cfg, ("dispersion", "r_lookup_path"), {"global": 0.9}, bundle)
    _write(cfg, ("dispersion", "rho_path"), {"rho": 0.31}, bundle)
    _write(cfg, ("posterior", "prior", "path"), {"source": "fallback"}, bundle)


def test_stamp_records_what_the_artifact_was_fitted_against(cfg):
    payload = provenance.stamp({"rho": 0.31}, cfg, BUNDLE, "test")
    p = payload["provenance"]
    assert p["bundle"] == BUNDLE and p["written_by"] == "test"
    assert p["config_version"] == cfg["meta"]["config_version"]
    assert p["created_at"].endswith("+00:00")     # UTC, not local


def test_a_coherent_set_verifies(cfg):
    _full_bundle(cfg)
    state = provenance.verify(cfg)
    assert state["verdict"] == "PASS", state["problems"]
    assert state["bundle"] == BUNDLE
    assert state["missing"] == ["calibration"]    # absent is not inconsistent


def test_mixed_vintages_are_caught(cfg):
    """The whole point: rho fitted against a different model than the prior."""
    _full_bundle(cfg)
    _write(cfg, ("dispersion", "rho_path"), {"rho": 0.42},
           bundle="baseline-20250101000000")
    state = provenance.verify(cfg)
    assert state["verdict"] == "FAIL"
    assert any("mixed bundle" in p for p in state["problems"])
    assert state["bundle"] is None                # no single answer to report


def test_an_unstamped_artifact_is_caught(cfg):
    _full_bundle(cfg)
    _write(cfg, ("dispersion", "rho_path"), {"rho": 0.31}, stamped=False)
    state = provenance.verify(cfg)
    assert state["verdict"] == "FAIL"
    assert any("no provenance: rho" in p for p in state["problems"])


def test_the_model_file_and_split_manifest_need_no_stamp(cfg):
    """One cannot carry JSON, the other precedes the model. Neither is a fault."""
    _full_bundle(cfg)
    assert provenance.verify(cfg)["verdict"] == "PASS"


def test_sealing_then_editing_an_artifact_is_caught(cfg):
    """Provenance alone cannot see this: an editor leaves the stamp intact."""
    _full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    assert sealed["bundle"] == BUNDLE
    assert provenance.verify(cfg, sealed)["verdict"] == "PASS"

    path = provenance._path(cfg, ("dispersion", "rho_path"))
    payload = json.load(open(path))
    payload["rho"] = 0.99                          # stamp untouched
    json.dump(payload, open(path, "w"))

    state = provenance.verify(cfg, sealed)
    assert state["verdict"] == "FAIL"
    assert any("changed since sealing: rho" in p for p in state["problems"])


def test_seal_refuses_an_inconsistent_set(cfg):
    """A sealed mixed bundle is worse than an unsealed one: it looks decided."""
    _full_bundle(cfg)
    _write(cfg, ("dispersion", "rho_path"), {"rho": 0.42}, bundle="other-model")
    with pytest.raises(SystemExit) as exc:
        seal_mod.seal(cfg)
    assert "mixed bundle" in str(exc.value)


def test_seal_refuses_when_there_is_nothing_stamped(cfg):
    with pytest.raises(SystemExit):
        seal_mod.seal(cfg)


def test_a_seal_naming_a_bundle_that_is_gone_is_caught(cfg):
    """Artifacts replaced wholesale by a newer run, seal never refreshed."""
    _full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    _full_bundle(cfg, bundle="baseline-20270101000000")
    state = provenance.verify(cfg, sealed)
    assert state["verdict"] == "FAIL"
    assert any("is not on disk" in p for p in state["problems"])
