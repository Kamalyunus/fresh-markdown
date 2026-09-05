"""The prior's boundary handling: a peak pinned at epsilon_min is a boundary
solution (rule 3), not a right-signed estimate."""

import copy

import numpy as np

from bootstrap import prior_density as pdn


def _sharp_curves(peaks):
    """A fake `build_curves`: one sharp Gaussian log-likelihood per category,
    both arms alike, peaking at `peaks[cat]` wherever the grid reaches."""
    def build(d, cfg, model, grid, window):
        out = {}
        for cat, peak in peaks.items():
            ll = -50.0 * (np.asarray(grid) - peak) ** 2
            out[cat] = {"naive": ll, "controlled": ll.copy(), "deff": 1.0,
                        "rho_eps_free": 0.0, "mean_rows_per_episode": 1.0,
                        "rows": 100, "episodes": 100, "censored_share": 0.0,
                        "log_ratio_sd": 0.1, "distinct_discounts": 3,
                        "identifying_variation_share": 0.5, "time_cells": 5,
                        "median_rows_per_time_cell": 20.0}
        return out
    return build


def test_the_unconstrained_search_extends_below_epsilon_min(cfg):
    lo, hi = cfg["posterior"]["epsilon_min"], cfg["posterior"]["epsilon_max"]
    n = cfg["posterior"]["prior"]["search_grid_size"]
    margin = cfg["posterior"]["prior"]["unconstrained_search_below"]
    upper = np.linspace(lo, max(1.0, hi), n)
    wide = pdn.extend_below(upper, margin)
    # the upper part is untouched (the sign search is unchanged)...
    assert np.allclose(wide[-n:], upper)
    # ...and the grid now reaches at least `margin` below lo at the same step
    assert wide[0] <= lo - margin + 1e-9
    assert np.allclose(np.diff(wide), upper[1] - upper[0])


def test_a_peak_pinned_at_epsilon_min_is_a_boundary_solution_not_pooled_evidence(
        cfg, monkeypatch):
    """Before: the search widened only upward, so a likelihood running off
    the LOWER bound argmax'd at lo, read as right-signed, contributed its own
    density AND entered the pool. Now it is flagged `boundary: "lower"`,
    takes the pool like a rejected category, stays out of the pool, and the
    note tells the owner which bound may be widened."""
    lo = cfg["posterior"]["epsilon_min"]
    peaks = {"INTERIOR": -1.0, "RUNOFF": lo - 5.0, "WRONG": +0.5}
    monkeypatch.setattr(pdn, "build_curves", _sharp_curves(peaks))
    grid, per, dens, pooled = pdn.estimate(None, cfg, None, fast=True)

    interior, runoff, wrong = per["INTERIOR"], per["RUNOFF"], per["WRONG"]
    assert interior["boundary"] is None and not interior["wrong_sign"]
    assert runoff["boundary"] == "lower" and not runoff["wrong_sign"]
    assert wrong["wrong_sign"] and wrong["boundary"] is None
    # the pinned arms are named, and the argmax really sits at the grid's
    # lower edge (at or below lo + one step)
    step = grid[1] - grid[0]
    assert max(runoff["unconstrained_argmax"].values()) <= lo + step + 1e-9
    assert "epsilon_min" in runoff["boundary_note"]
    assert "epsilon_max never" in runoff["boundary_note"]
    assert runoff["boundary_note"].startswith("controlled and naive arm")

    # the pool is the interior category alone...
    assert pooled["pooled_categories"] == ["INTERIOR"]
    assert "interior" in pooled["pooled_basis"]
    # ...and both rejected categories take it wholesale
    assert np.allclose(dens["RUNOFF"], pooled["pooled_density"], atol=1e-7)
    assert np.allclose(dens["WRONG"], pooled["pooled_density"], atol=1e-7)
    assert runoff["mean"] == wrong["mean"] == pooled["pooled_mean"]
    # the interior one keeps its own (sharp, near -1.0) density
    assert abs(interior["mean"] - (-1.0)) < 0.05


def test_one_pinned_arm_is_enough_to_flag_the_boundary(cfg, monkeypatch):
    """Rule 3 is per fit: each arm is a fit, and the 50/50 own density piles
    half its mass on the edge when only one runs off."""
    lo = cfg["posterior"]["epsilon_min"]

    def build(d, c, model, grid, window):
        g = np.asarray(grid)
        return {"HALF": {"naive": -50.0 * (g + 1.0) ** 2,
                         "controlled": -50.0 * (g - (lo - 3.0)) ** 2,
                         "deff": 1.0, "rho_eps_free": 0.0,
                         "mean_rows_per_episode": 1.0, "rows": 50,
                         "episodes": 50, "censored_share": 0.0,
                         "log_ratio_sd": 0.1, "distinct_discounts": 2,
                         "identifying_variation_share": 0.5, "time_cells": 3,
                         "median_rows_per_time_cell": 10.0}}
    monkeypatch.setattr(pdn, "build_curves", build)
    grid, per, dens, pooled = pdn.estimate(None, cfg, None, fast=True)
    assert per["HALF"]["boundary"] == "lower"
    assert per["HALF"]["boundary_note"].startswith("controlled arm")
    # nothing usable to pool: the uniform is the measured answer
    assert pooled["pooled_categories"] == []
    assert pooled["pooled_basis"].startswith("UNIFORM")


def test_the_lower_margin_is_a_config_key_and_the_upper_bound_never_moves(cfg):
    """The one bound that MAY be widened is epsilon_min; the search past
    epsilon_max is fixed at +1.0 (a positive optimum must be visible) and
    is not a tunable."""
    import inspect
    assert "unconstrained_search_below" in cfg["posterior"]["prior"]
    src = inspect.getsource(pdn.unconstrained_argmax)
    assert 'cfg["posterior"]["prior"]["unconstrained_search_below"]' in src
    assert "max(1.0, hi)" in src

    # the design_effect floor lives in common.config, not re-applied here
    assert "max(1.0, design_effect" not in inspect.getsource(pdn.deflation_deff)


def test_estimate_prior_surfaces_the_boundary_categories(cfg, monkeypatch):
    from bootstrap import estimate_prior as ep

    lo = cfg["posterior"]["epsilon_min"]
    monkeypatch.setattr(pdn, "build_curves",
                        _sharp_curves({"A": -1.0, "B": lo - 5.0}))
    monkeypatch.setattr(ep, "BaselineModel", lambda c: None)
    monkeypatch.setattr(ep, "_episodes_per_week", lambda d, c: {})
    monkeypatch.setattr(pdn, "holdout_comparison",
                        lambda *a, **k: {"window": "calib"})
    prior = ep.estimate_prior(None, copy.deepcopy(cfg), fast=True)
    assert prior["lower_boundary_categories"] == ["B"]
    assert prior["wrong_sign_categories"] == []
    assert prior["pooled"]["pooled_categories"] == ["A"]


def test_a_flat_likelihood_is_not_a_boundary_solution(cfg, monkeypatch):
    """No price variation -> epsilon is absent from the likelihood and the
    curve is flat. np.argmax then returns the FIRST grid point, which is now
    below epsilon_min -- absence of information, not a run-off, so it must
    not be flagged (and must not tell the owner to widen epsilon_min)."""
    lo = cfg["posterior"]["epsilon_min"]

    def build(d, c, model, grid, window):
        g = np.asarray(grid)
        flat = np.full(len(g), -12.5)
        return {"FLAT": {"naive": flat, "controlled": flat.copy(), "deff": 1.0,
                         "rho_eps_free": 0.0, "mean_rows_per_episode": 1.0,
                         "rows": 40, "episodes": 40, "censored_share": 0.0,
                         "log_ratio_sd": 0.0, "distinct_discounts": 1,
                         "identifying_variation_share": 0.0, "time_cells": 2,
                         "median_rows_per_time_cell": 20.0},
                "REAL": {"naive": -50.0 * (g + 1.5) ** 2,
                         "controlled": -50.0 * (g + 1.5) ** 2, "deff": 1.0,
                         "rho_eps_free": 0.0, "mean_rows_per_episode": 1.0,
                         "rows": 40, "episodes": 40, "censored_share": 0.0,
                         "log_ratio_sd": 0.2, "distinct_discounts": 3,
                         "identifying_variation_share": 0.5, "time_cells": 2,
                         "median_rows_per_time_cell": 20.0}}
    monkeypatch.setattr(pdn, "build_curves", build)
    grid, per, dens, pooled = pdn.estimate(None, cfg, None, fast=True)
    flat = per["FLAT"]
    # the argmax does sit at the grid's lower edge, as before the change...
    assert max(flat["unconstrained_argmax"].values()) < lo
    # ...but it is not a boundary solution, and it is pooled for the reason
    # it always was: no price variation
    assert flat["boundary"] is None and "boundary_note" not in flat
    assert "no_price_variation" in flat and not flat["wrong_sign"]
    assert pooled["pooled_categories"] == ["REAL"]
    assert set(flat["unconstrained_argmax"]) == {"naive", "controlled"}
