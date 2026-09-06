"""fit.prior_density: boundary handling (a peak pinned at epsilon_min is a
boundary solution, rule 3, not a right-signed estimate), the entry row, the
hour control, the deflation, and the std floor."""

import copy

import numpy as np
import pytest

from fit import prior_density as pdn


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


def test_the_unconstrained_search_extends_the_fit_grid_at_its_own_step(cfg):
    """The wide lattice used to be a SEPARATE linspace(lo, 1.0, n), so its
    step differed from the fit grid's and "epsilon_min + one grid step" was
    measured in the wrong units. It is now the fit grid extended past both
    bounds at the fit step, and the fit range is a slice of it."""
    lo, hi = cfg["posterior"]["epsilon_min"], cfg["posterior"]["epsilon_max"]
    n = cfg["posterior"]["prior"]["search_grid_size"]
    margin = cfg["posterior"]["prior"]["unconstrained_search_below"]
    grid = np.linspace(lo, hi, n)
    wide, sl = pdn.search_grid(grid, cfg)
    # the fit grid is IN the lattice, bit for bit
    assert np.array_equal(wide[sl], grid)
    # one step everywhere, the fit grid's own
    assert np.allclose(np.diff(wide), grid[1] - grid[0])
    # past the sign bound to at least +1.0, and `margin` below epsilon_min
    assert wide[-1] >= 1.0 - 1e-9 and wide[0] <= lo - margin + 1e-9
    assert wide[sl.start] == lo


def test_estimate_profiles_the_train_window_once_over_the_wide_lattice(
        cfg, monkeypatch):
    """The fit grid and the boundary search used to be two `build_curves`
    passes over the train window; the second is a slice of the first."""
    calls = []
    inner = _sharp_curves({"A": -1.0})

    def build(d, c, model, grid, window):
        calls.append((window, np.asarray(grid)))
        return inner(d, c, model, grid, window)
    monkeypatch.setattr(pdn, "build_curves", build)
    grid, per, dens, pooled = pdn.estimate(None, cfg, None, fast=True)
    assert len(calls) == 1 and calls[0][0] == "train"
    wide = calls[0][1]
    assert wide[0] < grid[0] and wide[-1] > grid[-1]
    assert abs(per["A"]["mean"] - (-1.0)) < 0.05


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
    src = inspect.getsource(pdn.search_grid)
    assert 'cfg["posterior"]["prior"]["unconstrained_search_below"]' in src
    assert "max(1.0, hi)" in src

    # the design_effect floor lives in common.config, not re-applied here
    assert "max(1.0, design_effect" not in inspect.getsource(pdn.deflation_deff)


def test_wrong_sign_is_a_peak_within_one_grid_step_of_zero_or_above(
        cfg, monkeypatch):
    """Design 5.6: wrong-signed = unconstrained peak at or above zero. The
    lattice need not hold zero, so "at zero" is within one FIT step of it;
    a peak two steps below zero is a (weak) right-signed estimate."""
    lo, hi = cfg["posterior"]["epsilon_min"], cfg["posterior"]["epsilon_max"]
    grid = np.linspace(lo, hi, cfg["posterior"]["prior"]["search_grid_size"])
    step = grid[1] - grid[0]
    wide, sl = pdn.search_grid(grid, cfg)
    just_below_zero = wide[wide < 0][-1]          # the lattice point under 0
    assert -step < just_below_zero < 0
    two_below = wide[wide < -2 * step][-1]
    monkeypatch.setattr(pdn, "build_curves", _sharp_curves(
        {"NEAR": just_below_zero, "WEAK": two_below, "POS": 0.4}))
    _, per, _, pooled = pdn.estimate(None, cfg, None, fast=True)
    assert per["NEAR"]["wrong_sign"] and per["POS"]["wrong_sign"]
    assert not per["WEAK"]["wrong_sign"]
    assert pooled["pooled_categories"] == ["WEAK"]


def test_estimate_prior_surfaces_the_boundary_categories(cfg, monkeypatch):
    from fit import estimate_prior as ep

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


def test_float_noise_on_a_flat_curve_is_not_a_boundary_solution(cfg, monkeypatch):
    """A flat likelihood carries ~1e-12 of arithmetic noise; if that noise
    happens to peak on the edge point it "strictly beats" the interior by
    nothing at all, and must not read as a run-off."""
    def build(d, c, model, grid, window):
        g = np.asarray(grid)
        flat = np.full(len(g), -3000.0)
        flat[0] += 1e-12                       # the edge "wins" by noise
        return {"FLAT": {"naive": flat, "controlled": flat.copy(), "deff": 1.0,
                         "rho_eps_free": 0.0, "mean_rows_per_episode": 1.0,
                         "rows": 40, "episodes": 40, "censored_share": 0.0,
                         "log_ratio_sd": 0.0, "distinct_discounts": 1,
                         "identifying_variation_share": 0.0, "time_cells": 2,
                         "median_rows_per_time_cell": 20.0}}
    monkeypatch.setattr(pdn, "build_curves", build)
    _, per, _, _ = pdn.estimate(None, cfg, None, fast=True)
    assert per["FLAT"]["boundary"] is None and "boundary_note" not in per["FLAT"]


def test_a_prior_std_can_never_be_zero(cfg, monkeypatch):
    """A zero-width prior is not confidence, it is a frozen posterior: a
    likelihood so sharp the density sits on one grid point still reports
    the grid step as its std, and says that is what bound it."""
    def build(d, c, model, grid, window):
        g = np.asarray(grid)
        # thousands of nats per grid step: the density is one grid point
        ll = -1e6 * (g + 1.0) ** 2
        return {"SHARP": {"naive": ll, "controlled": ll.copy(), "deff": 1.0,
                          "rho_eps_free": 0.0, "mean_rows_per_episode": 1.0,
                          "rows": 100, "episodes": 100, "censored_share": 0.0,
                          "log_ratio_sd": 0.3, "distinct_discounts": 5,
                          "identifying_variation_share": 0.5, "time_cells": 5,
                          "median_rows_per_time_cell": 20.0}}
    monkeypatch.setattr(pdn, "build_curves", build)
    grid, per, _, _ = pdn.estimate(None, cfg, None, fast=True)
    step = float(grid[1] - grid[0])
    sharp = per["SHARP"]
    assert sharp["std_candidates"]["density"] < step / 10, \
        "the collapse this guards against must actually happen"
    assert sharp["std"] == pytest.approx(round(step, 4))
    assert sharp["std_basis"] == "grid_resolution"


def test_the_bincount_multipliers_match_the_groupby_reference():
    """hour_multipliers ran a pandas groupby per grid point; it is now one
    factorisation and three bincounts -- the same numbers, incl. thin cells,
    cells with no uncensored row, and censored rows excluded from the fit."""
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 500
    cells = rng.choice([f"d{i}|h{j}" for i in range(4) for j in range(6)], n)
    mu = rng.uniform(0.2, 6.0, n)
    k = rng.poisson(mu * 1.3)
    cen = rng.random(n) < 0.15
    cen[cells == "d0|h0"] = True               # a cell with no uncensored row

    def reference(mu, cell, k, censored, min_rows):
        frame = pd.DataFrame({"cell": cell, "k": k, "mu": mu, "cen": censored})
        unc = frame[~frame.cen]
        g = unc.groupby("cell")
        m = (g.k.sum() / g.mu.sum()).clip(0.05, 20.0)
        if min_rows > 1:
            m = m[g.size() >= min_rows]
        return (frame.cell.map(m).fillna(1.0).to_numpy(),
                float(np.mean(frame.cell.map(m).isna().to_numpy())))

    for min_rows in (1, 5, 30):
        want, want_thin = reference(mu, cells, k, cen, min_rows)
        got, thin = pdn.hour_multipliers(mu, cells, k, cen, min_rows)
        assert np.allclose(got, want) and thin == pytest.approx(want_thin)
        # integer codes (what `curve` passes) give the same answer
        got2, thin2 = pdn.hour_multipliers(mu, pd.factorize(cells)[0], k, cen,
                                           min_rows)
        assert np.array_equal(got2, got) and thin2 == thin


def test_fold_spread_cuts_the_train_window_by_episode_not_by_row(
        cfg, monkeypatch):
    """Rule 15: a fold boundary at midnight used to put a cross-midnight
    episode in BOTH folds, and its 00:00 row -- a post-price-path row --
    became the second fold's entry row (the rule 7 confound)."""
    import pandas as pd

    days = [f"2026-04-0{i}" for i in range(1, 7)]      # 3 folds x 2 days
    rows = []
    for di, day in enumerate(days):
        for e in range(60):                            # > 50 per fold
            rows.append({"episode_id": f"{day}-{e}", "date": day,
                         "hour_of_day": 10, "total_discount": 0.10})
    # opens on day 2 at 22:00, runs past midnight into day 3 (fold 2's
    # first day), deepening its discount after the seam
    rows += [{"episode_id": "STRADDLE", "date": days[1], "hour_of_day": 22,
              "total_discount": 0.10},
             {"episode_id": "STRADDLE", "date": days[1], "hour_of_day": 23,
              "total_discount": 0.10},
             {"episode_id": "STRADDLE", "date": days[2], "hour_of_day": 0,
              "total_discount": 0.40},
             {"episode_id": "STRADDLE", "date": days[2], "hour_of_day": 1,
              "total_discount": 0.40}]
    train = pd.DataFrame(rows).assign(
        starting_inventory=5, units_sold=1, ending_inventory=4,
        category="VEG", sku_id="S", fc="F", d_ref=0.30)

    seen = []

    def fake_curve(g, mu_ref, grid, controlled, c):
        if not controlled:
            seen.append(g[["episode_id", "date", "hour_of_day",
                           "total_discount"]].copy())
        return -((np.asarray(grid) + 1.0) ** 2)

    class _M:
        @staticmethod
        def predict_mu_ref(g):
            return np.full(len(g), 2.0)

    monkeypatch.setattr(pdn, "split_frames", lambda d, c: {"train": d})
    monkeypatch.setattr(pdn, "population", lambda f, c: f)
    monkeypatch.setattr(pdn, "curve", fake_curve)
    grid = np.linspace(-5.0, -0.05, 20)
    out = pdn.fold_spread(train, cfg, _M(), grid, folds=3)

    assert out["VEG"]["folds"] == 3
    assert len(seen) == 3
    # every episode is scored in exactly ONE fold...
    ids = [set(g.episode_id) for g in seen]
    assert sum("STRADDLE" in s for s in ids) == 1
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2])
    # ...the straddler in the fold its OPENING date belongs to, on its entry
    # row (22:00 at the entry price), never on its 00:00 row
    fold = next(g for g in seen if "STRADDLE" in set(g.episode_id))
    row = fold[fold.episode_id == "STRADDLE"]
    assert len(row) == 1
    assert (row.date.iloc[0], int(row.hour_of_day.iloc[0])) == (days[1], 22)
    assert row.total_discount.iloc[0] == 0.10
    assert days[0] in set(fold.date)                   # fold 1, not fold 2


def test_the_hour_control_is_keyed_on_the_day_not_just_the_clock():
    """Design 5.6 says "same-hour CROSS-EPISODE", and a control pooled across
    dates is only an approximation to it: it removes the average evening lift
    and leaves a Tuesday storm or a rival's promotion in the residual, still
    correlated with how far the legacy ramp has run."""
    import inspect

    import numpy as np
    import pandas as pd

    from fit import prior_density as pdn

    assert list(inspect.signature(pdn.time_cell).parameters) == ["g"], \
        "the control is not selectable any more -- date_hour won"

    g = pd.DataFrame({
        "date": ["2026-03-01"] * 3 + ["2026-03-02"] * 3,
        "hour_of_day": [10, 11, 10, 10, 11, 10],
    })
    cells = list(pdn.time_cell(g))
    assert len(set(cells)) == 4, cells
    # the same clock hour on two different days must NOT share a cell -- that
    # is the entire difference between the two controls
    assert cells[0] != cells[3]

    # A CELL FITTED FROM TOO FEW ROWS absorbs the price response it is meant to
    # control for, and biases |eps| toward zero. Thin cells fall back to 1.0
    # rather than fitting a multiplier from three observations.
    mu = np.full(6, 2.0)
    k = np.array([3, 3, 3, 1, 1, 1])
    cen = np.zeros(6, bool)
    mult, thin = pdn.hour_multipliers(mu, np.array(cells), k, cen, min_rows=5)
    assert np.allclose(mult, 1.0), "every cell here has 1-2 rows; none qualify"
    assert thin == pytest.approx(1.0)
    # with no minimum they are all fitted, which is the behaviour the guard
    # exists to prevent at small cell sizes
    fitted, _ = pdn.hour_multipliers(mu, np.array(cells), k, cen, min_rows=1)
    assert not np.allclose(fitted, 1.0)


def test_the_prior_entry_row_is_the_first_HOUR_not_the_lowest_clock_time(cfg):
    """Rule 7: the prior identifies elasticity on ENTRY rows only. Sorting by
    hour_of_day alone picks the 00:00 row of an episode that opened at 22:00
    the night before -- a within-episode, post-price-path row, which is the
    confound the rule exists to exclude. Production windows cross midnight
    routinely (design 12a); the fixture has none, so nothing caught it."""
    import pandas as pd

    from fit.prior_density import scored_rows

    ep = pd.DataFrame([
        # opens 22:00 at the anchor, deepens after midnight
        {"date": "2026-07-01", "hour_of_day": 22, "total_discount": 0.10},
        {"date": "2026-07-01", "hour_of_day": 23, "total_discount": 0.10},
        {"date": "2026-07-02", "hour_of_day": 0, "total_discount": 0.25},
        {"date": "2026-07-02", "hour_of_day": 3, "total_discount": 0.25},
    ]).assign(episode_id="EP-MIDNIGHT", starting_inventory=5, units_sold=1,
              ending_inventory=4, category="VEG", subcategory="LEAFY",
              original_price=1e4, cost=4e3, fc="F1", d_ref=0.30)

    row = scored_rows(ep)
    assert len(row) == 1
    assert (row.date.iloc[0], int(row.hour_of_day.iloc[0])) == ("2026-07-01", 22)
    assert row.total_discount.iloc[0] == 0.10      # the entry price, not 0.25


def test_the_priors_deflation_can_actually_engage(cfg):
    """scored_rows returns ONE row per episode, so an episode-grouped ICC was
    empty by construction: rho 0, deff exactly 1.0 for every category, and
    design 5.6's deflation could never do anything. Clustered on the unit that
    recurs -- SKU x FC -- it engages when correlation is present."""
    import numpy as np
    import pandas as pd

    from fit.prior_density import deflation_deff

    rng = np.random.default_rng(0)
    n_units, per_unit = 200, cfg["assurance"]["rho_min_hours_per_episode"] + 1
    rows = []
    for u in range(n_units):
        shared = rng.normal(0, 2.0)            # a persistent per-unit level
        for _ in range(per_unit):
            rows.append({"sku_id": f"S{u}", "fc": "F1",
                         "units_sold": 5.0 + shared + rng.normal(0, 0.5)})
    frame = pd.DataFrame(rows).assign(episode_id=lambda f: range(len(f)))
    mu_ref = np.full(len(frame), 5.0)     # mu_ref is the deflation's baseline only

    deff, rho, m = deflation_deff(frame, mu_ref, cfg)
    assert m == pytest.approx(per_unit)
    assert rho > 0.5                    # the shared level dominates
    assert deff > 1.0                   # and the deflation is real

    # one row per unit: no clustering to measure, deff falls back to 1.0
    solo = frame.drop_duplicates("sku_id").copy()
    assert deflation_deff(solo, np.full(len(solo), 5.0), cfg)[0] == \
        pytest.approx(1.0)


def test_the_deflation_takes_rho_and_cluster_size_over_the_same_units(cfg):
    """rho was measured on units with >= rho_min_hours_per_episode rows while
    `m` averaged EVERY unit, singletons included -- two bases in one deff.
    Both now read the same unit set."""
    import numpy as np
    import pandas as pd

    from fit.prior_density import deflation_deff

    lo = cfg["assurance"]["rho_min_hours_per_episode"]
    rows = []
    for u in range(50):                        # units at the minimum
        rows += [{"sku_id": f"K{u}", "fc": "F", "units_sold": 5.0 + u % 3}] * lo
    for u in range(300):                       # singletons: no correlation
        rows.append({"sku_id": f"S{u}", "fc": "F", "units_sold": 5.0})
    frame = pd.DataFrame(rows).assign(episode_id=lambda f: range(len(f)))
    deff, rho, m = deflation_deff(frame, np.full(len(frame), 5.0), cfg)
    assert m == pytest.approx(lo), "m is the mean over the units rho was measured on"
    assert rho > 0 and deff == pytest.approx(1.0 + (m - 1.0) * rho)
