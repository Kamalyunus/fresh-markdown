"""Config loader and validation. config.yaml is the single tuning surface
(design 5.1); modules never carry their own numeric literals.

load_config() permits nulls (bootstrap produces the MEASURED values);
load_config(strict=True) refuses to start while any runtime-required
MEASURED / SET BY OWNER value is null.
"""

import json
import os

import yaml


class ConfigError(RuntimeError):
    pass


# Keys that must be non-null before any price is applied in production.
RUNTIME_REQUIRED = [
    ("dispersion", "rho"),
    ("dispersion", "mean_forced_hours_per_episode"),
    ("exploration", "tau_initial"),
    ("monitoring", "stop_conditions", "scrap_deterioration_pct"),
    ("monitoring", "stop_conditions", "margin_deterioration_pct"),
    ("ab_test", "min_detectable_effect_pct"),
]


# Values living in TWO places: a frozen artifact and a hand-paste in config.
# A stale paste silently mis-weights every posterior step (rho/forced-hours
# set deff); strict mode refuses to start on divergence.
ARTIFACT_MIRRORS = [
    (("dispersion", "rho_path"), "rho", ("dispersion", "rho")),
    (("dispersion", "rho_path"), "mean_forced_hours_per_episode",
     ("dispersion", "mean_forced_hours_per_episode")),
]


def config_get(cfg, path):
    node = cfg
    for key in path:
        node = node[key]
    return node


def artifact_mirror_drift(cfg, tol=5e-4):
    """Config values that disagree with the artifact they were pasted from.

    Returns a list of human-readable divergences (empty when consistent).
    A missing artifact is not drift -- bootstrap has not run yet.
    """
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
        if pasted is None or abs(float(pasted) - float(frozen)) > tol:
            drift.append(f"{'.'.join(cfg_path)}={pasted} but "
                         f"{path}:{field}={frozen}")
    return drift


def load_config(path="config.yaml", strict=False):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    sb = cfg["posterior"]["prior"]["search_bounds"]
    if [sb[0], sb[1]] != [cfg["posterior"]["epsilon_min"], cfg["posterior"]["epsilon_max"]]:
        raise ConfigError(
            "posterior.prior.search_bounds must equal [epsilon_min, epsilon_max] "
            "(design 5.6: a bound tighter than epsilon_min is a defect)")

    if strict:
        missing = [".".join(p) for p in RUNTIME_REQUIRED
                   if config_get(cfg, p) is None]
        if not os.path.exists(cfg["posterior"]["prior"]["path"]):
            missing.append("artifacts/prior.json (run bootstrap.estimate_prior)")
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


def design_effect(rho, forced_hours):
    """Cluster design effect: 1 + (m - 1) * rho, floored at 1 (design 5.11).
    The single definition -- the floor keeps a negative rho from DIVIDING
    information instead of deflating it. Import it; do not retype it."""
    return max(1.0, 1.0 + (forced_hours - 1.0) * rho)


def deff(cfg):
    """Design effect from the frozen rho and forced-hours in config."""
    return design_effect(cfg["dispersion"]["rho"],
                         cfg["dispersion"]["mean_forced_hours_per_episode"])
