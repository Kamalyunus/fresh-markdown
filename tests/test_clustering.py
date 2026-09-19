"""common.clustering: the one rho estimator and the design effect it feeds."""

import numpy as np
import pytest

from common.clustering import (deff_from_episodes, design_effect,
                               intraclass_correlation)


def test_the_icc_clips_at_its_own_range_unless_told_a_tunable(cfg):
    """`intraclass_correlation(clip_max=0.95)` duplicated
    dispersion.rho_clip_max: a config with the key deleted read the literal
    with no error while status called config the single source. The
    default clips at the estimator's own range, never at a tunable."""
    # a perfectly clustered residual: the estimator's own range is 1.0
    resid = np.repeat([1.0, -1.0, 3.0, -3.0], 5)
    groups = np.repeat(["a", "b", "c", "d"], 5)
    assert intraclass_correlation(resid, groups) == pytest.approx(1.0)
    assert intraclass_correlation(resid, groups, 0.95) == pytest.approx(0.95)
    assert intraclass_correlation(resid, groups, cfg["dispersion"]["rho_clip_max"]) \
        == pytest.approx(cfg["dispersion"]["rho_clip_max"])


def test_the_anova_form_recovers_rho_on_independent_hours():
    """`var(group means) / var(all)` read 1/m on independent draws; the ANOVA
    ICC reads (near) zero there and the shared level where there is one."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(400), 6)
    independent = rng.normal(0, 1.0, len(groups))
    assert intraclass_correlation(independent, groups) < 0.05
    shared = independent + np.repeat(rng.normal(0, 2.0, 400), 6)
    assert intraclass_correlation(shared, groups) > 0.7
    # fewer than two groups, or a NaN residual, is no correlation -- not a crash
    assert intraclass_correlation([1.0, 2.0], ["a", "a"]) == 0.0
    assert intraclass_correlation([1.0, np.nan, 2.0, 1.0], ["a", "a", "b", "b"]) == 0.0


def test_the_design_effect_floors_at_one_and_reads_m_per_batch():
    assert design_effect(-0.5, 6.0) == 1.0
    assert design_effect(0.5, 3.0) == pytest.approx(2.0)
    # m is the mean forced outcomes per episode in the batch at hand
    assert deff_from_episodes(0.5, []) == 1.0
    assert deff_from_episodes(0.5, ["e1", "e1", "e2", None]) == \
        pytest.approx(design_effect(0.5, 1.5))

