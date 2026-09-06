"""common.provenance -- the frozen artifacts, versioned as one bundle.

The artifacts are fitted in sequence and only meaningful TOGETHER: mix
vintages and nothing errors, the numbers just silently stop describing one
world. The bundle id IS the baseline model version -- every downstream
artifact is fitted AGAINST a model. ops.seal adds per-file hashes so a
hand-edited artifact is detectable too, which stamps alone cannot catch.
"""

import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
from datetime import datetime, timezone

from common.config import config_get
from common.io import read_json

# the libraries whose numerics reach a price: a version move changes
# predictions (LightGBM), the pmf and the DP's arithmetic (scipy, numpy) or
# the frame semantics every fit reads (pandas) with no artifact byte moving
LIBRARIES = ("numpy", "scipy", "pandas", "lightgbm", "pyarrow")

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


def library_versions():
    return {"python": platform.python_version(),
            **{lib: _version_of(lib) for lib in LIBRARIES}}


def _version_of(lib):
    try:
        return importlib.metadata.version(lib)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment(cfg):
    """Everything a priced hour depends on that is NOT a frozen artifact:
    the config (its digest) and the libraries. Sealed beside the artifacts,
    compared by verify(); the config digest is stamped on every event."""
    return {"config_digest": config_fingerprint(cfg)["digest"],
            "libraries": library_versions()}


def launch_posterior(cfg):
    """The posterior file as it stands at seal time: its digest, what each
    cell launched from and where it is now, and how much it has consumed.
    RECORDED, never verified -- the file is learning state and moves by
    design; the record is what a later cell is traced back to."""
    path = cfg["posterior"]["path"]
    state = read_json(path)
    if not state:
        return None
    return {"digest": file_digest(path),
            "cells": {c: {"launch_mean": r.get("mean") if not r.get("n_obs") else None,
                          "prior_mean": r.get("prior_mean"), "mean": r.get("mean"),
                          "std": r.get("std"), "version": r.get("version")}
                      for c, r in (state.get("cells") or {}).items()},
            "cell_of": state.get("cell_of"),
            "cold_start_shift_std": state.get("cold_start_shift_std"),
            "outcomes_consumed": len(state.get("processed_outcome_ids") or []),
            "tau": state.get("tau")}


def environment_drift(cfg, sealed):
    """What moved since the seal, outside the artifacts: config keys and
    library versions. [] when nothing did, or when there is no seal at all;
    a seal that predates the environment record is ONE problem, not silence
    -- read as "no drift" it would never be re-sealed with the record."""
    if not sealed:
        return []
    env = sealed.get("environment")
    if not env:
        return ["environment not sealed -- re-seal once to record config "
                "and libraries"]
    out = []
    if env.get("config_digest") != config_fingerprint(cfg)["digest"]:
        snap = (sealed.get("config_snapshot") or {})
        moved = [d.split(":")[0] for d in config_diff(snap, cfg)] if snap else []
        out.append("config moved since sealing"
                   + (": " + ", ".join(moved[:8]) + (" ..." if len(moved) > 8 else "")
                      if moved else ""))
    libs_then, libs_now = env.get("libraries") or {}, library_versions()
    moved = [f"{k} {libs_then[k]} -> {libs_now.get(k)}" for k in libs_then
             if libs_then[k] and libs_now.get(k) != libs_then[k]]
    if moved:
        out.append("libraries moved since sealing: " + ", ".join(moved))
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
        path = config_get(cfg, key)
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
    # the environment is part of the seal: a config edit or a library
    # upgrade changes what an hour is priced with as surely as an edited
    # artifact, and reads the same way -- re-seal, on purpose
    problems.extend(environment_drift(cfg, sealed))

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
    return read_json(config_get(cfg, ("artifacts", "bundle_path")))


# ----------------------------------------------------------------- audit trail

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
