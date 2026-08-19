"""common.provenance -- the frozen artifacts, versioned as one bundle.

Six files are fitted in sequence and then frozen together. They are only
meaningful TOGETHER: `rho` deflates evidence measured against a particular
model's residuals, the level factors correct that same model, and the prior was
estimated using that model's predictions and that `r_lookup`. Mix vintages and
nothing errors -- the numbers simply stop describing the same world, silently
and for the whole window.

Until now only `feature_schema.json` recorded which model it came from, so a
mismatch between config and an artifact told you the two disagreed but not
which one was stale. That is not a question anyone should have to answer from
memory.

**The bundle id is the baseline model version.** Not a fresh timestamp: every
downstream artifact is fitted AGAINST a model, so keying on the model answers
"which model was this fitted against" directly, and an artifact that names a
different model is by definition not part of this bundle.

    artifacts/                       bundle
      baseline_model.txt      ─┐
      feature_schema.json      │     baseline-20260811043259
      calibration.json         ├──   created_at per file
      r_lookup.json            │     config_version per file
      rho.json                 │
      prior.json              ─┘

`bootstrap.seal` writes `artifacts/bundle.json`: the agreed id plus a hash of
every file. After that, a hand-edited artifact is detectable too -- which the
provenance stamps alone cannot catch, since an editor would leave them intact.
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

    `bundle` is the baseline model version this artifact was fitted against --
    None only for artifacts that precede the model (the split manifest), where
    there is nothing to be fitted against yet.
    """
    payload["provenance"] = {
        "bundle": bundle,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_version": cfg["meta"]["config_version"],
        "written_by": tool,
    }
    return payload


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
    """Do the artifacts on disk form ONE bundle, and match the seal?

    Two separate failures, deliberately not merged: artifacts naming different
    models is a mixed bundle, while a hash that moved since sealing is an
    edited artifact. The first is a process mistake, the second is closer to
    tampering, and the remedy differs.
    """
    rows = collect(cfg)
    present = [r for r in rows if r["present"]]
    missing = [r["artifact"] for r in rows if not r["present"]]

    # Two artifacts legitimately carry no stamp: the split manifest precedes
    # the model, and the model file IS the bundle id (recorded in
    # feature_schema, since a LightGBM dump has nowhere to put one).
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
