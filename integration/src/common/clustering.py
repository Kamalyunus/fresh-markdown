"""common.clustering -- the cluster statistics behind deff: the ONE rho
estimator (`intraclass_correlation`), the design effect and its per-batch
reading (`deff_from_episodes`, design 5.5, 5.11).

Reads no config: every caller passes `dispersion.rho_clip_max` itself.
"""

import numpy as np
import pandas as pd


def intraclass_correlation(residuals, groups, clip_max=None):
    """One-way random-effects ICC -- the ONE home for rho.

    `var(group means) / var(all)` estimates `rho + (1 - rho)/m`, not rho:
    a group mean of m independent draws still varies by sigma^2/m, and that
    term is read as shared signal. On INDEPENDENT hours it returns 1/m, so
    deff deflated every posterior step by a factor that was pure estimator
    artifact (on the repo FIXTURE, 0.164 at m=6 -- a fixture reading, rule
    19). The ANOVA form subtracts MSW, which is exactly that term, and
    recovers rho at every m.

    `clip_max` is the ceiling on the fitted correlation -- every production
    caller passes `dispersion.rho_clip_max`; None clips at the estimator's
    own range (1.0), never at a tunable hidden here.
    """
    clip_max = 1.0 if clip_max is None else float(clip_max)
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
