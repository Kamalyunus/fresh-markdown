"""common.history: the audit snapshot every seal leaves, the reports a stop
adds, and the index ordered by seal time."""

import json
import os
import pathlib

import pytest

from common import history, provenance
from common.config import config_get
from conftest import BUNDLE, artifact_at, full_bundle, scratch_paths
from ops import seal as seal_mod


@pytest.fixture
def cfg(cfg, tmp_path):
    """Every artifact, the posterior and the history under a scratch dir."""
    c = scratch_paths(cfg, tmp_path)
    c["artifacts"]["history_dir"] = str(tmp_path / "history")
    (tmp_path / "posterior.json").write_text("{}")
    (tmp_path / "config.yaml").write_text("meta: {config_version: t}\n")
    return c


def test_every_seal_leaves_an_audit_snapshot_and_stops_add_the_reports(cfg, tmp_path):
    """A retrain overwrites artifacts/ in place; the history folder is the
    audit trail: bundle files, config and posterior per seal, the reports
    per stop, never pruned by the process."""
    conf = tmp_path / "config.yaml"
    full_bundle(cfg)
    sealed = seal_mod.seal(cfg)

    snap = history.archive(cfg, sealed, config_path=str(conf), reason="bootstrap")
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
    snap2 = history.archive(cfg, later, config_path=str(conf), reason="weekly-refit")
    same_second = dict(sealed, sealed_at="2030-01-01T00:00:00.000002+00:00")
    snap3 = history.archive(cfg, same_second, config_path=str(conf), reason="retrain")
    assert len({snap, snap2, snap3}) == 3
    assert history.latest_snapshot(cfg, BUNDLE) == snap3
    assert [r for _, _, r in history.history_index(cfg)] == \
        ["bootstrap", "weekly-refit", "retrain"]

    # a stop copies the reports as they stand into the LATEST snapshot
    reports = tmp_path / "reports"; reports.mkdir()
    (reports / "shadow.json").write_text("{}"); (reports / "launch_readiness.md").write_text("x")
    dst = history.archive_reports(cfg, str(reports), BUNDLE)
    assert dst == os.path.join(snap3, "reports")
    assert {"shadow.json", "launch_readiness.md"} <= set(os.listdir(dst))
    assert history.archive_reports(cfg, str(reports), "no-such-bundle") is None
    # the callers that read the trail through provenance still reach it
    assert provenance.archive is history.archive
    assert provenance.history_index(cfg) == history.history_index(cfg)


def test_the_history_is_ordered_by_seal_time_not_by_folder_name(cfg, tmp_path):
    """history_index claimed "oldest first" and sorted by PATH, which sorts
    by bundle name first: a bundle whose name sorts earlier but was sealed
    later came out first, and status printed it as the latest snapshot."""
    conf = tmp_path / "config.yaml"
    # "zzz" sorts AFTER "aaa" by name but is sealed FIRST
    for bundle, when, reason in (("zzz-model", "2026-01-01T00:00:00+00:00", "first"),
                                 ("aaa-model", "2026-06-01T00:00:00+00:00", "second")):
        full_bundle(cfg, bundle=bundle)
        sealed = dict(seal_mod.seal(cfg), sealed_at=when)
        history.archive(cfg, sealed, config_path=str(conf), reason=reason)
    assert [r for _, _, r in history.history_index(cfg)] == ["first", "second"]
    assert [b for b, _, _ in history.history_index(cfg)] == ["zzz-model", "aaa-model"]


def test_archive_refuses_a_copy_that_does_not_match_the_seal(cfg, tmp_path):
    """The audit trail is only evidence if the copy IS what was sealed: an
    artifact edited (or re-fitted) between seal and archive must raise, not
    become the record of that bundle."""
    conf = tmp_path / "config.yaml"
    full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    path = config_get(cfg, ("dispersion", "rho_path"))
    payload = json.load(open(path))
    payload["rho"] = 0.99
    json.dump(payload, open(path, "w"))
    with pytest.raises(RuntimeError, match="rho on disk does not match its seal"):
        history.archive(cfg, sealed, config_path=str(conf), reason="bootstrap")
    # an artifact that appeared after sealing is refused too: nothing vouches for it
    full_bundle(cfg)
    sealed = seal_mod.seal(cfg)
    artifact_at(cfg, ("baseline_model", "calibration_factor_path"), {"factors": {}})
    with pytest.raises(RuntimeError, match="calibration on disk does not match its seal"):
        history.archive(cfg, sealed, config_path=str(conf))
    assert pathlib.Path(tmp_path / "history").is_dir()
