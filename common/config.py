"""Config loader and validation. config.yaml is the single tuning surface
(design 5.1); modules never carry their own numeric literals.

load_config() permits nulls (bootstrap produces the MEASURED values);
load_config(strict=True) refuses to start while any runtime-required
MEASURED / SET BY OWNER value is null.
"""

import json
import os

import numpy as np
import pandas as pd
import yaml


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


def config_get(cfg, path):
    node = cfg
    for key in path:
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
    rel = float(cfg["dispersion"].get("rho_paste_tolerance_rel", 0.01))
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


# a category's prior is "own data" when the pooled term carries nothing:
# estimate_prior labels it and status counts it on this one threshold
OWN_DATA_WEIGHT = 0.999


def intraclass_correlation(residuals, groups, clip_max=0.95):
    """One-way random-effects ICC -- the ONE home for rho.

    `var(group means) / var(all)` estimates `rho + (1 - rho)/m`, not rho:
    a group mean of m independent draws still varies by sigma^2/m, and that
    term is read as shared signal. On INDEPENDENT hours it returns 1/m
    (measured: 0.164 at m=6), so deff deflated every posterior step by ~1.8x
    of pure estimator artifact. The ANOVA form subtracts MSW, which is
    exactly that term, and recovers rho at every m.
    """
    s = pd.Series(np.asarray(residuals, dtype=float)).reset_index(drop=True)
    g = pd.Series(np.asarray(groups)).reset_index(drop=True)
    finite = np.isfinite(s.to_numpy())          # one NaN poisons every sum
    s, g = s[finite].reset_index(drop=True), g[finite].reset_index(drop=True)
    sizes = g.groupby(g).size().to_numpy()
    k, n = len(sizes), len(s)
    if k < 2 or n <= k:
        return 0.0
    means = s.groupby(g).mean()
    msb = float((sizes * (means.to_numpy() - s.mean()) ** 2).sum() / (k - 1))
    msw = float(((s - g.map(means)) ** 2).sum() / (n - k))
    # n0: the unbalanced-design effective group size (== m when balanced)
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    den = msb + (n0 - 1) * msw
    if den <= 0:
        return 0.0
    return float(np.clip((msb - msw) / den, 0.0, clip_max))


def design_effect(rho, forced_hours):
    """Cluster design effect: 1 + (m - 1) * rho, floored at 1 (design 5.11).
    The single definition -- the floor keeps a negative rho from DIVIDING
    information instead of deflating it. Import it; do not retype it."""
    return max(1.0, 1.0 + (forced_hours - 1.0) * rho)


def deff_from_episodes(rho, episode_ids):
    """deff at the clustering ACTUALLY present: `m` is the mean number of
    forced outcomes per episode in `episode_ids`.

    `m` was a frozen config paste measured on the calib window as the mean
    LENGTH of legacy episodes whose discount changed -- not forced hours at
    all, and fixed while the real quantity moves with the exploration rate by
    construction. Measuring it per batch closes that drift channel instead of
    alerting on it, and removes a paste, a mirror and a staleness failure.
    """
    ids = [e for e in episode_ids if e is not None]
    if not ids:
        return 1.0
    counts = pd.Series(ids).value_counts()
    return design_effect(rho, float(counts.mean()))
