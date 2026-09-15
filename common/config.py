"""Config loader and validation. config.yaml is the single tuning surface
(design 5.1); modules never carry their own numeric literals.

load_config() permits nulls (bootstrap produces the MEASURED values);
load_config(strict=True) refuses to start while any runtime-required
MEASURED / SET BY OWNER value is null.
"""

import json
import os

import yaml

# moved to common.clustering; the names stay for callers
from common.clustering import (intraclass_correlation, design_effect,   # noqa: F401
                               deff_from_episodes)


class ConfigError(RuntimeError):
    pass


# Keys that must be non-null before any price is applied in production.
RUNTIME_REQUIRED = [
    ("data", "launch_date"),
    ("dispersion", "rho"),
    ("exploration", "tau_initial"),
    ("monitoring", "stop_conditions", "scrap_deterioration_pct"),
    ("monitoring", "stop_conditions", "margin_deterioration_pct"),
]


# Values living in TWO places: a frozen artifact and a hand-paste in config.
# A stale paste silently mis-weights every posterior step (rho/forced-hours
# set deff); strict mode refuses to start on divergence.
ARTIFACT_MIRRORS = [
    (("dispersion", "rho_path"), "rho", ("dispersion", "rho")),
]


_MISSING = object()


def config_get(cfg, path, default=_MISSING):
    """The value at a dotted `path` (a tuple of keys). Without `default` a
    missing key raises, as a config read should; with one, a missing or
    non-mapping node returns it -- the one getter tune's findings and
    status's null check share."""
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            if default is _MISSING:
                raise KeyError(".".join(map(str, path)))
            return default
        node = node[key]
    return node


def artifact_mirror_drift(cfg, tol=None):
    """Config values that disagree with the artifact they were pasted from.

    Returns a list of human-readable divergences (empty when consistent).
    A missing artifact is not drift -- bootstrap has not run yet. The
    tolerance is `dispersion.rho_paste_tolerance_rel` of the frozen value
    (1%): each --check-only turn contracts rho by ~1e-3 while the loop is
    still converging, and a tolerance below that step made every settle a
    new paste. Relative, so it means the same thing on a fixture rho of
    0.12 and a production rho of 0.65. `tol`, when given, is absolute.
    """
    rel = float(cfg["dispersion"]["rho_paste_tolerance_rel"])   # config, no default
    drift = []
    for path_key, field, cfg_path in ARTIFACT_MIRRORS:
        path = config_get(cfg, path_key)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            artifact = json.load(f)
        if field not in artifact:
            continue
        pasted, frozen = config_get(cfg, cfg_path), artifact[field]
        allow = tol if tol is not None else rel * abs(float(frozen))
        if pasted is None or abs(float(pasted) - float(frozen)) > allow:
            drift.append(f"{'.'.join(cfg_path)}={pasted} but "
                         f"{path}:{field}={frozen}")
    return drift


def load_config(path="config.yaml", strict=False):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if strict:
        missing = [".".join(p) for p in RUNTIME_REQUIRED
                   if config_get(cfg, p) is None]
        if not os.path.exists(cfg["posterior"]["prior"]["path"]):
            missing.append("artifacts/prior.json (run fit.estimate_prior)")
        if missing:
            raise ConfigError(
                "refusing to start: null MEASURED / SET BY OWNER values: "
                + ", ".join(missing))
        drift = artifact_mirror_drift(cfg)
        if drift:
            raise ConfigError(
                "refusing to start: config disagrees with the frozen "
                "artifacts it was pasted from (" + "; ".join(drift)
                + "). Re-paste from the artifact -- these set deff, which "
                  "deflates every posterior update.")
    return cfg


def reference_discount(cfg, category):
    """Category anchor d_ref. Config keys use underscores ('SIDE_DISH');
    source data uses spaces ('SIDE DISH')."""
    table = cfg["reference_discount"]
    key = str(category).replace(" ", "_")
    return float(table.get(key, table["_default"]))


# a category's prior is "own data" when the pooled term carries nothing:
# estimate_prior labels it and status counts it on this one threshold
OWN_DATA_WEIGHT = 0.999
