"""common.history -- the audit trail: every seal copied into
`artifacts/history/<bundle>/<sealed_at>/` with a MANIFEST, the reports a
stop adds beside it, and the index `status` prints (design 5.14a).

Reads the artifact walk and the digests from `common.provenance` (what was
fitted against what); provenance never reads this module.
"""

import glob
import json
import os
import shutil
from datetime import datetime

from common.io import read_json
from common.provenance import collect, config_fingerprint, file_digest


def _history_root(cfg):
    return cfg["artifacts"].get("history_dir") or os.path.join(
        os.path.dirname(cfg["artifacts"]["bundle_path"]) or ".", "history")


def _folder_stamp(sealed_at):
    """The seal instant as a folder name, to the MICROSECOND
    (`YYYYMMDDTHHMMSS.ffffff`): two seals inside one second are two
    snapshots, never a silent overwrite of the first."""
    return datetime.fromisoformat(str(sealed_at)).strftime("%Y%m%dT%H%M%S.%f")


def archive(cfg, sealed, config_path="config.yaml", reason=None):
    """Copy the sealed bundle -- every present artifact, the config in force,
    the posterior state -- into history/<bundle>/<sealed_at>/ with a
    MANIFEST. A retrain, a re-fit and a re-seal each leave their own
    snapshot, so what ran under which artifacts is answerable later without
    trusting anyone's memory. Never pruned by the process. Every copied
    artifact is hashed against the seal: a copy that does not match what was
    sealed (edited or re-fitted since) raises rather than becoming the
    record."""
    out = os.path.join(_history_root(cfg), sealed["bundle"],
                       _folder_stamp(sealed["sealed_at"]))
    os.makedirs(out, exist_ok=True)
    files = {}
    for row in collect(cfg):
        if row["present"]:
            dst = os.path.join(out, os.path.basename(row["path"]))
            shutil.copyfile(row["path"], dst)
            want = (sealed.get("sha256") or {}).get(row["artifact"])
            if file_digest(dst) != want:
                raise RuntimeError(
                    f"refusing to archive: {row['artifact']} on disk does "
                    f"not match its seal ({'not sealed' if want is None else 'digest moved'})"
                    " -- re-run ops.seal on the artifacts as they stand")
            files[row["artifact"]] = os.path.basename(dst)
    for label, path in (("config", config_path),
                        ("posterior", cfg["posterior"]["path"]),
                        ("bundle", cfg["artifacts"]["bundle_path"])):
        if path and os.path.exists(path):
            shutil.copyfile(path, os.path.join(out, os.path.basename(path)))
            files[label] = os.path.basename(path)
    manifest = {"bundle": sealed["bundle"], "sealed_at": sealed["sealed_at"],
                "reason": reason, "config_version": sealed.get("config_version"),
                "config_digest": config_fingerprint(cfg)["digest"],
                "environment": sealed.get("environment"),
                "launch_posterior": sealed.get("launch_posterior"),
                "sha256": sealed["sha256"], "files": files}
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return out


def _snapshots(cfg, bundle="*"):
    """[(folder, manifest)] for every snapshot of `bundle` (default: all),
    oldest first BY `sealed_at` -- path order sorts by bundle name first,
    and a bundle sealed later can sort earlier by name."""
    pattern = os.path.join(_history_root(cfg), str(bundle), "*", "MANIFEST.json")
    found = [(os.path.dirname(m), read_json(m) or {}) for m in glob.glob(pattern)]
    return sorted(found, key=lambda fm: str(fm[1].get("sealed_at") or ""))


def latest_snapshot(cfg, bundle):
    """The newest history folder for `bundle` (by `sealed_at`), or None."""
    snaps = _snapshots(cfg, bundle)
    return snaps[-1][0] if snaps else None


def archive_reports(cfg, reports_root, bundle):
    """Copy the reports as they stand into the bundle's latest snapshot
    (reports/ inside it), so the record of a bundle carries how it graded.
    Later stops overwrite: the snapshot holds the latest read of that
    bundle. Returns the folder, or None when the bundle has no snapshot."""
    snap = latest_snapshot(cfg, bundle) if bundle else None
    if snap is None:
        return None
    dst = os.path.join(snap, "reports")
    os.makedirs(dst, exist_ok=True)
    for path in glob.glob(os.path.join(reports_root, "*.json")) + \
            glob.glob(os.path.join(reports_root, "*.md")):
        shutil.copyfile(path, os.path.join(dst, os.path.basename(path)))
    return dst


def history_index(cfg):
    """[(bundle, sealed_at, reason)] for every snapshot, oldest first by
    `sealed_at`."""
    return [(p.get("bundle"), p.get("sealed_at"), p.get("reason"))
            for _, p in _snapshots(cfg)]
