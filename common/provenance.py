"""common.provenance -- the frozen artifacts, versioned as one bundle.

The artifacts are fitted in sequence and only meaningful TOGETHER: mix
vintages and nothing errors, the numbers just silently stop describing one
world. The bundle id IS the baseline model version -- every downstream
artifact is fitted AGAINST a model. ops.seal adds per-file hashes so a
hand-edited artifact is detectable too, which stamps alone cannot catch.
"""

import glob
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

from common.io import read_json

# Every frozen artifact, and the config key holding its path. Order is fitting
# order, which is also the order a mismatch propagates in.
ARTIFACTS = [
    ("split_manifest", ("data", "split_manifest_path")),
    ("baseline_model", ("baseline_model", "model_path")),
    ("feature_schema", ("baseline_model", "feature_schema_path")),
    ("calibration", ("baseline_model", "calibration_factor_path")),
    ("r_lookup", ("dispersion", "r_lookup_path")),
    ("rho", ("dispersion", "rho_path")),
    ("prior", ("posterior", "prior", "path")),
]


def _path(cfg, key):
    node = cfg
    for k in key:
        node = node[k]
    return node


def stamp(payload, cfg, bundle, tool):
    """Attach provenance to an artifact payload, in place, and return it.
    `bundle` is the baseline model version fitted against -- None only for
    artifacts that precede the model (the split manifest)."""
    payload["provenance"] = {
        "bundle": bundle,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_version": cfg["meta"]["config_version"],
        "written_by": tool,
    }
    return payload


def config_fingerprint(cfg, phase=None):
    """What a report RAN UNDER: the full config it read, a digest of it, and
    the phase the report belongs to (backtest / shadow / production).

    `meta.config_version` is a string a human is meant to bump when a
    tunable a report reads changes. Nobody does, and `tune --apply` pastes
    values without touching it, so the report-vintage check was blind to
    every paste -- a shadow run under tau 270 and one under tau 1,300 both
    said "1.0.0". The digest moves on any change; the snapshot says WHICH
    values were in force, so status can name what moved since.
    """
    canon = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return {
        "phase": phase,
        "digest": hashlib.sha256(canon.encode()).hexdigest()[:16],
        "config_version": cfg["meta"]["config_version"],
        "snapshot": json.loads(json.dumps(cfg, default=str)),
    }


def config_diff(snapshot, cfg, prefix=""):
    """Dotted keys whose value differs between a report's snapshot and the
    config now in force -- the list status prints when a digest has moved."""
    live = json.loads(json.dumps(cfg, default=str))
    out = []
    for key in sorted(set(snapshot) | set(live)):
        a, b = snapshot.get(key), live.get(key)
        path = f"{prefix}{key}"
        if isinstance(a, dict) and isinstance(b, dict):
            out.extend(config_diff(a, b, path + "."))
        elif a != b:
            out.append(f"{path}: {a!r} -> {b!r}")
    return out


def file_digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _bundle_of(path):
    """The bundle an artifact names, or None if it carries no stamp."""
    if not path.endswith(".json"):
        return None                       # the model itself is the bundle id
    try:
        payload = read_json(path, {})
    except (OSError, json.JSONDecodeError):
        return None
    prov = payload.get("provenance") or {}
    # feature_schema predates provenance stamping and names the model directly
    return prov.get("bundle") or payload.get("model_version")


def collect(cfg):
    """Every frozen artifact: present, its bundle, its hash."""
    out = []
    for name, key in ARTIFACTS:
        path = _path(cfg, key)
        row = {"artifact": name, "path": path, "present": os.path.exists(path)}
        if row["present"]:
            row["bundle"] = _bundle_of(path)
            row["sha256"] = file_digest(path)
        out.append(row)
    return out


def verify(cfg, sealed=None):
    """Do the artifacts on disk form ONE bundle, and match the seal? Two
    failures, deliberately not merged: a mixed bundle (process mistake) vs a
    hash moved since sealing (edited artifact) -- the remedy differs."""
    rows = collect(cfg)
    present = [r for r in rows if r["present"]]
    missing = [r["artifact"] for r in rows if not r["present"]]

    # legitimately unstamped: the split manifest precedes the model, and the
    # model file IS the bundle id (a LightGBM dump has nowhere to put a stamp)
    UNSTAMPABLE = ("split_manifest", "baseline_model")
    stamped = [r for r in present
               if r["artifact"] not in UNSTAMPABLE and r.get("bundle")]
    bundles = sorted({r["bundle"] for r in stamped})
    unstamped = [r["artifact"] for r in present
                 if r["artifact"] not in UNSTAMPABLE and not r.get("bundle")]

    changed, gone, added = [], [], []
    if sealed:
        by_name = {r["artifact"]: r for r in present}
        sealed_names = set(sealed.get("sha256") or {})
        for name, digest in (sealed.get("sha256") or {}).items():
            row = by_name.get(name)
            if row is None:
                gone.append(name)                   # sealed, now deleted
            elif row["sha256"] != digest:
                changed.append(name)
        added = [r["artifact"] for r in present
                 if r["artifact"] not in sealed_names]

    problems = []
    if len(bundles) > 1:
        problems.append("mixed bundle: " + ", ".join(
            f"{r['artifact']}={r['bundle']}" for r in stamped))
    if unstamped:
        problems.append("no provenance: " + ", ".join(unstamped))
    if changed:
        problems.append("changed since sealing: " + ", ".join(changed))
    if gone:
        problems.append("sealed but no longer on disk: " + ", ".join(gone))
    if added:
        problems.append("fitted after sealing (re-seal): " + ", ".join(added))
    if sealed and bundles and sealed.get("bundle") not in bundles:
        problems.append(f"sealed bundle {sealed.get('bundle')} "
                        f"is not on disk ({', '.join(bundles)})")

    return {
        "bundle": bundles[0] if len(bundles) == 1 else None,
        "artifacts": rows,
        "missing": missing,
        "problems": problems,
        "sealed_bundle": (sealed or {}).get("bundle"),
        "verdict": "PASS" if not problems and stamped else
                   "INSUFFICIENT" if not stamped else "FAIL",
    }


def load_seal(cfg):
    return read_json(_path(cfg, ("artifacts", "bundle_path")))


# ----------------------------------------------------------------- audit trail

def _history_root(cfg):
    return cfg["artifacts"].get("history_dir") or os.path.join(
        os.path.dirname(cfg["artifacts"]["bundle_path"]) or ".", "history")


def archive(cfg, sealed, config_path="config.yaml", reason=None):
    """Copy the sealed bundle -- every present artifact, the config in force,
    the posterior state -- into history/<bundle>/<sealed_at>/ with a
    MANIFEST. A retrain, a re-fit and a re-seal each leave their own
    snapshot, so what ran under which artifacts is answerable later without
    trusting anyone's memory. Never pruned by the process."""
    stamp = str(sealed["sealed_at"]).replace(":", "").replace("-", "")[:15]
    out = os.path.join(_history_root(cfg), sealed["bundle"], stamp)
    os.makedirs(out, exist_ok=True)
    files = {}
    for row in collect(cfg):
        if row["present"]:
            dst = os.path.join(out, os.path.basename(row["path"]))
            shutil.copyfile(row["path"], dst)
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
                "sha256": sealed["sha256"], "files": files}
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return out


def latest_snapshot(cfg, bundle):
    """The newest history folder for `bundle`, or None."""
    root = os.path.join(_history_root(cfg), str(bundle))
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    return dirs[-1] if dirs else None


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
    """[(bundle, sealed_at, reason)] for every snapshot, oldest first."""
    out = []
    for m in sorted(glob.glob(os.path.join(_history_root(cfg), "*", "*", "MANIFEST.json"))):
        payload = read_json(m) or {}
        out.append((payload.get("bundle"), payload.get("sealed_at"), payload.get("reason")))
    return out
