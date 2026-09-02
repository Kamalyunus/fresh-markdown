"""common.provenance -- the frozen artifacts, versioned as one bundle.

The artifacts are fitted in sequence and only meaningful TOGETHER: mix
vintages and nothing errors, the numbers just silently stop describing one
world. The bundle id IS the baseline model version -- every downstream
artifact is fitted AGAINST a model. bootstrap.seal adds per-file hashes so a
hand-edited artifact is detectable too, which stamps alone cannot catch.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

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
        with open(path) as f:
            payload = json.load(f)
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

    changed = []
    if sealed:
        by_name = {r["artifact"]: r for r in present}
        for name, digest in (sealed.get("sha256") or {}).items():
            row = by_name.get(name)
            if row and row["sha256"] != digest:
                changed.append(name)

    problems = []
    if len(bundles) > 1:
        problems.append("mixed bundle: " + ", ".join(
            f"{r['artifact']}={r['bundle']}" for r in stamped))
    if unstamped:
        problems.append("no provenance: " + ", ".join(unstamped))
    if changed:
        problems.append("changed since sealing: " + ", ".join(changed))
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
    path = _path(cfg, ("artifacts", "bundle_path"))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
