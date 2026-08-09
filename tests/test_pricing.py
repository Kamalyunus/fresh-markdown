"""Unit tests for the decision core: tiers, DP, exploration, bounded step."""

import numpy as np
import pytest
import yaml

from common.config import load_config
from pricing import dp as dp_mod
from pricing import explore
from pricing.demand import mu_at, nb_pmf_vector
from pricing.posterior import bounded_step

CFG = load_config()


class FixedRng:
    def integers(self, lo, hi):
        return lo


def test_feasible_tiers_respect_cost_floor():
    tiers, d_max = dp_mod.feasible_tiers(10000, 6500, CFG["pricing"]["tier_step"])
    assert d_max == pytest.approx(0.35)
    assert tiers[0] == 0.0
    assert max(tiers) <= d_max + 1e-9        # every price at or above cost
    assert np.allclose(np.diff(tiers), CFG["pricing"]["tier_step"])


def test_nb_pmf_tail_mass_folded():
    pmf, tail = nb_pmf_vector(mu=3.0, r=1.2, max_k=CFG["pricing"]["negbin_max_k"])
    assert pmf.sum() == pytest.approx(1.0)
    assert tail >= 0


def test_dp_terminal_value_and_entry_band():
    mu_path = [1.5, 1.2, 1.0]
    res = dp_mod.solve(10000, 6000, q0=5, mu_ref_path=mu_path, d_ref=0.30,
                       epsilon=-1.2, r=1.0, cfg=CFG, entry=True)
    lo = 0.30 - CFG["pricing"]["entry_window"]
    hi = 0.30 + CFG["pricing"]["entry_window"]
    for j in res.q_by_tier:
        assert lo - 1e-9 <= res.tiers[j] <= hi + 1e-9
    # value can never be worse than scrapping everything immediately
    assert res.v_star >= -6000 * 5


def test_dp_hourly_monotonicity():
    res = dp_mod.solve(10000, 6000, q0=4, mu_ref_path=[1.0, 1.0], d_ref=0.30,
                       epsilon=-1.2, r=1.0, cfg=CFG, anchor_discount=0.30)
    # price non-increasing within episode: only tiers at/deeper than the anchor
    assert all(res.tiers[j] >= 0.30 - 1e-9 for j in res.q_by_tier)


def test_explore_tau_zero_never_explores():
    res = dp_mod.solve(10000, 6000, q0=4, mu_ref_path=[1.0, 1.0], d_ref=0.30,
                       epsilon=-1.2, r=1.0, cfg=CFG, entry=True)
    choice = explore.select(res, tau=0.0, rng=FixedRng())
    assert not choice["is_exploration"]
    assert choice["chosen_index"] == res.optimal_index
    assert choice["exploration_cost"] == 0.0


def test_explore_affordable_draw_and_cost():
    res = dp_mod.solve(10000, 6000, q0=4, mu_ref_path=[1.0, 1.0], d_ref=0.30,
                       epsilon=-1.2, r=1.0, cfg=CFG, entry=True)
    big_tau = 1e12
    choice = explore.select(res, tau=big_tau, rng=FixedRng())
    if len(res.q_by_tier) > 1:
        assert choice["is_exploration"]
        assert choice["chosen_index"] != res.optimal_index
        expect = res.q_by_tier[res.optimal_index] - res.q_by_tier[choice["chosen_index"]]
        assert choice["exploration_cost"] == pytest.approx(expect)


def test_explore_non_explorable_excluded():
    res = dp_mod.solve(10000, 6000, q0=4, mu_ref_path=[1.0], d_ref=0.30,
                       epsilon=-1.2, r=1.0, cfg=CFG, entry=True)
    choice = explore.select(res, tau=1e12, rng=FixedRng(), explorable=False)
    assert not choice["is_exploration"]
    assert choice["affordable_set_size"] == 0


def test_budget_scales_down_as_posterior_narrows():
    wide = explore.budget_today(1e6, CFG["exploration"]["budget_scale_ref_std"], CFG)
    narrow = explore.budget_today(1e6, 0.0, CFG)
    assert wide == pytest.approx(CFG["exploration"]["budget_share_of_il"] * 1e6)
    assert narrow == pytest.approx(wide * CFG["exploration"]["budget_scale_floor"])


def test_tau_next_clipped():
    lo, hi = CFG["exploration"]["tau_adjust_clip"]
    assert explore.tau_next(100.0, 1e9, 1.0, CFG) == pytest.approx(100.0 * hi)
    assert explore.tau_next(100.0, 0.0, 1e9, CFG) == pytest.approx(100.0 * lo)


def test_bounded_step_clips_mean_and_floors_std():
    step = CFG["learning"]["max_mean_step"]
    m, s, clipped = bounded_step(-1.0, 0.5, -2.0, 0.01, CFG)
    assert m == pytest.approx(-1.0 - step)
    assert s == pytest.approx(0.5 * (1 - CFG["learning"]["max_std_shrink"]))
    assert clipped


def test_mu_floor():
    assert mu_at(0.0, 0.5, 0.3, -1.5, CFG["pricing"]["demand_floor"]) \
        == CFG["pricing"]["demand_floor"]


def test_censored_expectation_below_raw_mu():
    """E[min(D,q)] <= E[D]: the basis mismatch that broke calibration."""
    from pricing.demand import expected_min_demand_inventory_vec as E
    mu = np.array([0.5, 1.0, 3.0])
    r = np.array([1.2, 1.2, 1.2])
    q = np.array([2.0, 2.0, 2.0])
    censored = E(mu, r, q, CFG["pricing"]["negbin_max_k"])
    assert (censored <= mu + 1e-9).all()
    assert censored[2] < mu[2] * 0.8        # heavy censoring at low inventory


def test_level_factor_must_be_solved_on_censored_basis():
    """A factor fit against raw mu reads low -- it can even land below 1 while
    the true correction is above 1, which is why calibration could not move
    the gate. Solving against E[min(D,q)] recovers the true scaling."""
    from pricing.demand import expected_min_demand_inventory_vec as E
    max_k = CFG["pricing"]["negbin_max_k"]
    rng = np.random.default_rng(0)
    n = 2000
    mu = rng.uniform(0.2, 3.0, n)
    r = np.full(n, 1.2)
    q = rng.integers(1, 4, n).astype(float)     # tiny inventories: heavy censoring
    true_f = 1.45
    sold = E(true_f * mu, r, q, max_k).sum()

    raw_fit = sold / mu.sum()
    assert raw_fit < 1.0 < true_f               # wrong side of 1

    lo, hi = 0.1, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if E(mid * mu, r, q, max_k).sum() < sold:
            lo = mid
        else:
            hi = mid
    assert abs((lo + hi) / 2 - true_f) < 0.01


def test_config_rejects_narrowed_search_bounds(tmp_path):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["posterior"]["prior"]["search_bounds"] = [-1.5, -0.05]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    from common.config import ConfigError
    with pytest.raises(ConfigError):
        load_config(str(p))


def test_config_strict_refuses_null_measured(tmp_path):
    from common.config import ConfigError
    with pytest.raises(ConfigError, match="refusing to start"):
        load_config(strict=True)


def test_shadow_sample_size_defaults_to_config():
    """--max-episodes unset must read the config sample, not fall back to
    'every episode' -- the whole point is that a full sweep is too slow."""
    import inspect
    from pipeline import shadow

    assert inspect.signature(shadow.run_shadow).parameters[
        "max_episodes"].default is None
    cfg = load_config()
    assert cfg["monitoring"]["shadow_gate"]["sample_episodes"] > 0


def test_config_detects_stale_paste_from_frozen_artifact(tmp_path):
    """A config rho left over from a previous retrain silently mis-weights
    every posterior update, because rho and forced-hours set deff."""
    import json
    from common.config import artifact_mirror_drift

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    rho_path = tmp_path / "rho.json"
    cfg["dispersion"]["rho_path"] = str(rho_path)
    cfg["dispersion"]["rho"] = 0.3183
    cfg["dispersion"]["mean_forced_hours_per_episode"] = 9.134

    # no artifact yet -- bootstrap has not run, which is not drift
    assert artifact_mirror_drift(cfg) == []

    rho_path.write_text(json.dumps(
        {"rho": 0.3183, "mean_forced_hours_per_episode": 9.134}))
    assert artifact_mirror_drift(cfg) == []

    # retrain moved rho; config still holds the old paste
    rho_path.write_text(json.dumps(
        {"rho": 0.2510, "mean_forced_hours_per_episode": 9.134}))
    drift = artifact_mirror_drift(cfg)
    assert len(drift) == 1 and "dispersion.rho" in drift[0]
