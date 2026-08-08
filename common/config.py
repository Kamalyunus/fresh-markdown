"""Config loader and validation.

config.yaml is the single tuning surface (PRD section 7). Everything tunable is
read from here; modules never carry their own numeric literals.

Two loading modes:

  load_config()                 bootstrap mode -- nulls permitted, because
                                bootstrap is what produces the MEASURED values.
  load_config(strict=True)      runtime mode -- refuses to start while any
                                runtime-required MEASURED / SET BY OWNER value
                                is null (PRD section 7).
"""

import os

import yaml


class ConfigError(RuntimeError):
    pass


# Keys that must be non-null before any price is applied in production.
RUNTIME_REQUIRED = [
    ("baseline_model", "apply_level_calibration"),
    ("dispersion", "rho"),
    ("dispersion", "mean_forced_hours_per_episode"),
    ("exploration", "tau_initial"),
    ("monitoring", "stop_conditions", "scrap_deterioration_pct"),
    ("monitoring", "stop_conditions", "margin_deterioration_pct"),
    ("ab_test", "min_detectable_effect_pct"),
]


def _get(cfg, path):
    node = cfg
    for key in path:
        node = node[key]
    return node


def load_config(path="config.yaml", strict=False):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    sb = cfg["posterior"]["prior"]["search_bounds"]
    if [sb[0], sb[1]] != [cfg["posterior"]["epsilon_min"], cfg["posterior"]["epsilon_max"]]:
        raise ConfigError(
            "posterior.prior.search_bounds must equal [epsilon_min, epsilon_max] "
            "(PRD section 9.5: a bound tighter than epsilon_min is a defect)")

    if strict:
        missing = [".".join(p) for p in RUNTIME_REQUIRED if _get(cfg, p) is None]
        if cfg["posterior"]["prior"]["source"] == "bracket" \
                and cfg["posterior"]["prior"]["per_category"] is None \
                and not os.path.exists(cfg["posterior"]["prior"]["path"]):
            missing.append("posterior.prior.per_category (or artifacts/prior.json)")
        if missing:
            raise ConfigError(
                "refusing to start: null MEASURED / SET BY OWNER values: "
                + ", ".join(missing))
    return cfg


def reference_discount(cfg, category):
    """Category anchor d_ref. Config keys use underscores ('SIDE_DISH');
    source data uses spaces ('SIDE DISH')."""
    table = cfg["reference_discount"]
    key = str(category).replace(" ", "_")
    return float(table.get(key, table["_default"]))


def deff(cfg):
    """Design effect from frozen rho and forced-hours (PRD section 13.3)."""
    rho = cfg["dispersion"]["rho"]
    h = cfg["dispersion"]["mean_forced_hours_per_episode"]
    return max(1.0, 1.0 + (h - 1.0) * rho)
