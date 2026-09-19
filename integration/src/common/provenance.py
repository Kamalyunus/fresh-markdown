"""common.provenance -- the frozen artifacts, versioned as one bundle.

The artifacts are fitted in sequence and only meaningful TOGETHER: mix
vintages and nothing errors, the numbers just silently stop describing one
world. The bundle id IS the baseline model version -- every downstream
artifact is fitted AGAINST a model. ops.seal adds per-file hashes so a
hand-edited artifact is detectable too, which stamps alone cannot catch.
"""

import hashlib
import json


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
