"""Tests for common.provenance and ops.seal."""
import json
import pathlib

import pytest

from common import provenance
from common.config import config_get
from ops import seal as seal_mod
from conftest import _write


BUNDLE = "baseline-20260101000000"


@pytest.fixture
def cfg(cfg, tmp_path):
    """A config whose artifact paths all point into a scratch directory."""
    c = cfg
    c["artifacts"]["bundle_path"] = str(tmp_path / "bundle.json")
    c["data"]["split_manifest_path"] = str(tmp_path / "split_manifest.json")
    c["baseline_model"]["model_path"] = str(tmp_path / "baseline_model.txt")
    c["baseline_model"]["feature_schema_path"] = str(tmp_path / "feature_schema.json")
    c["baseline_model"]["calibration_factor_path"] = str(tmp_path / "calibration.json")
    c["dispersion"]["r_lookup_path"] = str(tmp_path / "r_lookup.json")
    c["dispersion"]["rho_path"] = str(tmp_path / "rho.json")
    c["posterior"]["prior"]["path"] = str(tmp_path / "prior.json")
    return c


def _artifact(cfg, key, payload, bundle=BUNDLE, stamped=True):
    """Write `payload` at the artifact path config names under `key`."""
    path = pathlib.Path(config_get(cfg, key))
    if stamped:
        provenance.stamp(payload, cfg, bundle, "test")
    _write(path.parent, path.stem, payload)


def _full_bundle(cfg, bundle=BUNDLE):
    """One coherent set: model, its schema, and everything fitted against it."""
    with open(config_get(cfg, ("baseline_model", "model_path")), "w") as f:
        f.write("tree { }")                      # a model file, not JSON
    _artifact(cfg, ("data", "split_manifest_path"), {"split": {}}, bundle=None)
    _artifact(cfg, ("baseline_model", "feature_schema_path"),
              {"model_version": bundle}, stamped=False)   # names its model the old way
    _artifact(cfg, ("dispersion", "r_lookup_path"), {"global": 0.9}, bundle)
    _artifact(cfg, ("dispersion", "rho_path"), {"rho": 0.31}, bundle)
    _artifact(cfg, ("posterior", "prior", "path"), {"source": "fallback"}, bundle)


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
    _artifact(cfg, ("dispersion", "rho_path"), {"rho": 0.42},
              bundle="baseline-20250101000000")
    state = provenance.verify(cfg)
    assert state["verdict"] == "FAIL"
    assert any("mixed bundle" in p for p in state["problems"])
    assert state["bundle"] is None                # no single answer to report


def test_an_unstamped_artifact_is_caught(cfg):
    _full_bundle(cfg)
    _artifact(cfg, ("dispersion", "rho_path"), {"rho": 0.31}, stamped=False)
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
    _full_bundle(cfg)
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
    snap = provenance.archive(cfg, sealed, config_path=str(conf), reason="config")
    manifest = json.load(open(pathlib.Path(snap, "MANIFEST.json")))
    assert manifest["environment"] == env and manifest["launch_posterior"] == lp


def test_seal_refuses_an_inconsistent_set(cfg):
    """A sealed mixed bundle is worse than an unsealed one: it looks decided."""
    _full_bundle(cfg)
    _artifact(cfg, ("dispersion", "rho_path"), {"rho": 0.42}, bundle="other-model")
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
    _full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    os.remove(config_get(cfg, ("dispersion", "r_lookup_path")))
    v = provenance.verify(cfg, sealed)
    assert v["verdict"] == "FAIL" and any("no longer on disk" in p for p in v["problems"])

    # the other direction: the calibration was ABSENT at sealing and is
    # fitted afterwards -- stamped with the right bundle, hashes all match,
    # and the seal still does not describe what is on disk
    _full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    assert "calibration" in sealed["missing"]
    _artifact(cfg, ("baseline_model", "calibration_factor_path"),
              {"factors": {"A": 1.0}})
    v = provenance.verify(cfg, sealed)
    assert v["verdict"] == "FAIL"
    assert any(p == "fitted after sealing (re-seal): calibration"
               for p in v["problems"])
    assert v["missing"] == []                    # present, just unsealed


def test_every_seal_leaves_an_audit_snapshot_and_stops_add_the_reports(cfg, tmp_path):
    """A retrain overwrites artifacts/ in place; the history folder is the
    audit trail: bundle files, config and posterior per seal, the reports
    per stop, never pruned by the process."""
    import os
    cfg["artifacts"]["history_dir"] = str(tmp_path / "history")
    cfg["posterior"]["path"] = str(tmp_path / "posterior.json")
    (tmp_path / "posterior.json").write_text("{}")
    conf = tmp_path / "config.yaml"; conf.write_text("meta: {config_version: t}\n")
    _full_bundle(cfg)
    sealed = seal_mod.seal(cfg)

    snap = provenance.archive(cfg, sealed, config_path=str(conf), reason="bootstrap")
    assert snap.startswith(str(tmp_path / "history" / BUNDLE))
    names = set(os.listdir(snap))
    assert {"MANIFEST.json", "rho.json", "r_lookup.json", "prior.json",
            "baseline_model.txt", "config.yaml", "posterior.json"} <= names
    manifest = json.load(open(os.path.join(snap, "MANIFEST.json")))
    assert manifest["bundle"] == BUNDLE and manifest["reason"] == "bootstrap"
    assert manifest["sha256"] == sealed["sha256"]
    # the copy is byte-identical to what was sealed
    assert provenance.file_digest(os.path.join(snap, "rho.json")) == sealed["sha256"]["rho"]

    # a second seal is a second folder, never an overwrite -- even inside
    # the same SECOND (the stamp carries the microseconds)
    later = dict(sealed, sealed_at="2030-01-01T00:00:00.000001+00:00")
    snap2 = provenance.archive(cfg, later, config_path=str(conf), reason="weekly-refit")
    same_second = dict(sealed, sealed_at="2030-01-01T00:00:00.000002+00:00")
    snap3 = provenance.archive(cfg, same_second, config_path=str(conf), reason="retrain")
    assert len({snap, snap2, snap3}) == 3
    assert provenance.latest_snapshot(cfg, BUNDLE) == snap3
    assert [r for _, _, r in provenance.history_index(cfg)] == \
        ["bootstrap", "weekly-refit", "retrain"]

    # a stop copies the reports as they stand into the LATEST snapshot
    reports = tmp_path / "reports"; reports.mkdir()
    (reports / "shadow.json").write_text("{}"); (reports / "launch_readiness.md").write_text("x")
    dst = provenance.archive_reports(cfg, str(reports), BUNDLE)
    assert dst == os.path.join(snap3, "reports")
    assert {"shadow.json", "launch_readiness.md"} <= set(os.listdir(dst))
    assert provenance.archive_reports(cfg, str(reports), "no-such-bundle") is None


def test_the_history_is_ordered_by_seal_time_not_by_folder_name(cfg, tmp_path):
    """history_index claimed "oldest first" and sorted by PATH, which sorts
    by bundle name first: a bundle whose name sorts earlier but was sealed
    later came out first, and status printed it as the latest snapshot."""
    cfg["artifacts"]["history_dir"] = str(tmp_path / "history")
    cfg["posterior"]["path"] = str(tmp_path / "posterior.json")
    (tmp_path / "posterior.json").write_text("{}")
    conf = tmp_path / "config.yaml"; conf.write_text("meta: {config_version: t}\n")
    # "zzz" sorts AFTER "aaa" by name but is sealed FIRST
    for bundle, when, reason in (("zzz-model", "2026-01-01T00:00:00+00:00", "first"),
                                 ("aaa-model", "2026-06-01T00:00:00+00:00", "second")):
        _full_bundle(cfg, bundle=bundle)
        sealed = dict(seal_mod.seal(cfg), sealed_at=when)
        provenance.archive(cfg, sealed, config_path=str(conf), reason=reason)
    assert [r for _, _, r in provenance.history_index(cfg)] == ["first", "second"]
    assert [b for b, _, _ in provenance.history_index(cfg)] == ["zzz-model", "aaa-model"]


def test_archive_refuses_a_copy_that_does_not_match_the_seal(cfg, tmp_path):
    """The audit trail is only evidence if the copy IS what was sealed: an
    artifact edited (or re-fitted) between seal and archive must raise, not
    become the record of that bundle."""
    cfg["artifacts"]["history_dir"] = str(tmp_path / "history")
    conf = tmp_path / "config.yaml"; conf.write_text("meta: {config_version: t}\n")
    _full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    path = config_get(cfg, ("dispersion", "rho_path"))
    payload = json.load(open(path))
    payload["rho"] = 0.99
    json.dump(payload, open(path, "w"))
    with pytest.raises(RuntimeError, match="rho on disk does not match its seal"):
        provenance.archive(cfg, sealed, config_path=str(conf), reason="bootstrap")
    # an artifact that appeared after sealing is refused too: nothing vouches for it
    _full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    _artifact(cfg, ("baseline_model", "calibration_factor_path"), {"factors": {}})
    with pytest.raises(RuntimeError, match="calibration on disk does not match its seal"):
        provenance.archive(cfg, sealed, config_path=str(conf))


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
