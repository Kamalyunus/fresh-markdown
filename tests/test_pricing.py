"""Unit tests for the decision core: tiers, the DP, censored demand, the
exploration draw and its floor, the bounded step."""

import numpy as np
import pytest

from conftest import CFG
from engine import dp as dp_mod
from engine import explore
from engine.demand import mu_at, nb_pmf_vector
from engine.posterior import bounded_step


class FixedRng:
    def integers(self, lo, hi):
        return lo


def test_feasible_tiers_respect_cost_floor():
    tiers, d_max = dp_mod.feasible_tiers(10000, 6500, CFG["pricing"]["tier_step"])
    assert d_max == pytest.approx(0.35)
    assert tiers[0] == 0.0
    assert max(tiers) <= d_max + 1e-9        # every price at or above cost
    assert np.allclose(np.diff(tiers), CFG["pricing"]["tier_step"])


def test_zero_cost_never_offers_a_zero_price():
    """A zero-cost row put d_max at 1.0, which put a 100% discount in the
    action set, which crashed the solver."""
    step = CFG["pricing"]["tier_step"]
    tiers, d_max = dp_mod.feasible_tiers(10000.0, 0.0, step)
    assert d_max == 1.0                      # the true cost floor, unchanged
    assert 1.0 not in tiers
    assert max(tiers) == pytest.approx(1.0 - step)
    assert all(10000.0 * (1 - d) > 0 for d in tiers)

    # the call that raised, end to end
    res = dp_mod.solve(10000.0, 0.0, 3, [0.8, 0.8, 0.62, 0.41], 0.30,
                       -1.0, 0.919, CFG, anchor_discount=None, entry=True)
    assert np.isfinite(res.q_by_tier[res.optimal_index])
    # and at the widest |epsilon| the search may return, where the divergence
    # at d -> 1 is steepest
    deep = dp_mod.solve(10000.0, 0.0, 3, [0.8] * 4, 0.30,
                        CFG["posterior"]["epsilon_min"], 0.919, CFG,
                        anchor_discount=0.0, entry=False)
    assert np.isfinite(deep.q_by_tier[deep.optimal_index])


def test_a_tier_step_that_does_not_divide_one_still_excludes_zero_price():
    """The exclusion is `price > 0`, not "drop the last tier"."""
    tiers, _ = dp_mod.feasible_tiers(10000.0, 0.0, 0.5)
    assert tiers == [0.0, 0.5]               # 1.0 dropped, 0.5 kept
    tiers, _ = dp_mod.feasible_tiers(10000.0, 0.0, 0.3)
    assert max(tiers) == pytest.approx(0.9)  # 0.9 < 1.0, nothing to drop


def test_nb_pmf_tail_mass_folded():
    pmf, tail = nb_pmf_vector(mu=3.0, r=1.2, max_k=CFG["pricing"]["negbin_max_k"])
    assert pmf.sum() == pytest.approx(1.0)
    assert tail >= 0


def test_dp_terminal_value_and_entry_arms():
    mu_path = [1.5, 1.2, 1.0]
    d_ref = 0.30
    res = dp_mod.solve(10000, 6000, q0=5, mu_ref_path=mu_path, d_ref=d_ref,
                       epsilon=-1.2, r=1.0, cfg=CFG, entry=True)
    offsets = CFG["pricing"]["entry_offsets"]
    chosen = sorted(round(res.tiers[j], 6) for j in res.q_by_tier)
    assert chosen == sorted(round(d_ref + o, 6) for o in offsets)
    # value can never be worse than scrapping everything immediately
    assert res.q_by_tier[res.optimal_index] >= -6000 * 5
    # the optimum IS the best Q -- there is no separate stored value to drift
    assert res.q_by_tier[res.optimal_index] == max(res.q_by_tier.values())


def test_entry_action_set_matches_offsets_and_cost_floor():
    step = CFG["pricing"]["tier_step"]
    offsets = CFG["pricing"]["entry_offsets"]
    d_ref = 0.30

    tiers, d_max = dp_mod.feasible_tiers(10000, 5000, step)      # d_max 0.50
    arms = [tiers[j] for j in dp_mod.entry_action_set(
        tiers, d_ref, d_max, CFG["pricing"])]
    assert arms == [0.15, 0.20, 0.25, 0.30, 0.35]

    # entry is BOUNDED on the deep side, not open-ended: the DP may open one
    # step past the reference, never a discount the hourly grid would take
    # many hours to reach
    assert max(arms) <= d_ref + max(offsets) + 1e-9
    assert max(offsets) <= 2 * step + 1e-9

    # every arm is a real tier and separated from its neighbours by at least
    # one hourly step -- adjacent arms would dilute the exploration draw
    assert all(a in tiers for a in arms)
    assert all(round(b - a, 6) >= step for a, b in zip(arms, arms[1:]))

    # a cost floor inside the requested band drops only the infeasible arms
    tiers, d_max = dp_mod.feasible_tiers(10000, 7800, step)      # d_max 0.22
    arms = [tiers[j] for j in dp_mod.entry_action_set(
        tiers, d_ref, d_max, CFG["pricing"])]
    assert arms == [0.15, 0.20] and max(arms) <= d_max + 1e-9

    # a floor below every requested arm leaves ONE action -- the deepest
    # feasible tier -- not a silent fallback to the whole grid
    tiers, d_max = dp_mod.feasible_tiers(10000, 9500, step)      # d_max 0.05
    arms = dp_mod.entry_action_set(tiers, d_ref, d_max, CFG["pricing"])
    assert arms == [len(tiers) - 1] and len(arms) < len(tiers)


def test_hourly_set_holds_every_deeper_tier_and_dp_uses_them_when_paid():
    """The hourly action set is not a single 2.5pp step -- it is every tier
    down to the cost floor. Whether the DP takes a deep one is economics, and
    the closed-form threshold must agree with what the solver actually does.
    """
    P0, cost, d_ref, anchor = 10000, 6000, 0.30, 0.15
    res = dp_mod.solve(P0, cost, q0=5, mu_ref_path=[1.0] * 3, d_ref=d_ref,
                       epsilon=-1.2, r=1.0, cfg=CFG, anchor_discount=anchor)
    arms = sorted(res.tiers[j] for j in res.q_by_tier)
    assert min(arms) == pytest.approx(anchor)          # may hold
    assert max(arms) == pytest.approx(0.40)            # may jump to the floor
    steps = [round(a - anchor, 3) for a in arms if a > anchor]
    assert 0.025 in steps and 0.05 in steps and 0.10 in steps

    thr = dp_mod.deepening_threshold_epsilon(P0, cost, anchor)
    assert thr == pytest.approx((1 - anchor) / (cost / P0 - anchor))

    def chosen(eps):
        r = dp_mod.solve(P0, cost, q0=5, mu_ref_path=[1.0] * 3, d_ref=d_ref,
                         epsilon=eps, r=1.0, cfg=CFG, anchor_discount=anchor)
        return r.tiers[r.optimal_index]

    # comfortably inside the hold region, and comfortably past it. Censoring
    # makes the true switch later than `thr`, so the bound is one-sided:
    # below it the DP must hold; above it, eventually deepens.
    assert chosen(-(thr * 0.5)) == pytest.approx(anchor)
    assert chosen(-(thr * 0.9)) == pytest.approx(anchor)
    assert chosen(-(thr * 2.0)) > anchor

    # price at or below cost: no deepening can ever pay
    assert dp_mod.deepening_threshold_epsilon(10000, 6000, 0.60) == float("inf")


def test_entry_choice_never_blocks_later_deepening():
    """The shallowest entry arm must still leave the hourly grid room to
    deepen, or entry would trade one decision away for the whole episode."""
    step = CFG["pricing"]["tier_step"]
    d_ref = 0.30
    tiers, d_max = dp_mod.feasible_tiers(10000, 6000, step)
    arms = dp_mod.entry_action_set(tiers, d_ref, d_max, CFG["pricing"])
    for j in arms:
        deeper = [d for d in tiers if d >= tiers[j] - 1e-9]
        assert len(deeper) >= CFG["exploration"]["min_feasible_tiers"]


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
    from engine.demand import expected_min_demand_inventory_vec as E
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
    from engine.demand import expected_min_demand_inventory_vec as E
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


def test_the_bounded_step_clamp_is_exact_at_and_past_the_boundary():
    """`max_mean_step` is a safety bound (the report-side check that it is
    written unrounded lives in test_update)."""
    cap = CFG["learning"]["max_mean_step"]
    for raw in (-1.0 - cap, -1.0 - cap * 3, -1.0 + cap * 3):
        m, _, _ = bounded_step(-1.0, 0.6, raw, 0.5, CFG)
        assert abs(m - (-1.0)) <= cap + 1e-12, (raw, m)
    # an awkward starting mean, the shape that exposed the reporting bug
    m, _, _ = bounded_step(-0.76098, 0.6, -3.0, 0.5, CFG)
    assert abs(m - (-0.76098)) <= cap + 1e-12


def test_the_tier_epsilon_has_one_home():
    """Seven modules carried their own 1e-9 for "is this the same tier"."""
    import inspect
    from engine import dp
    from engine import decide
    from evaluate import shadow
    assert dp.TIER_EPS == 1e-9
    for fn in (dp.feasible_tiers, dp.entry_action_set, dp.solve):
        assert "1e-9" not in inspect.getsource(fn), fn.__name__
    assert "TIER_EPS" in inspect.getsource(decide)
    assert "TIER_EPS" in inspect.getsource(shadow)


def _solved(eps=-1.0):
    return dp_mod.solve(10000.0, 4000.0, 6, [0.8] * 6, 0.30, eps, 0.9, CFG,
                        anchor_discount=None, entry=True)


def test_delta_min_removes_the_tiers_the_model_cannot_tell_apart():
    """The learner reads a forced outcome against mu_ref at the REFERENCE
    discount, so the informative distance is from d_ref, not from p*: a
    tier far from p* but at d_ref costs budget and teaches nothing. Below
    bias/|eps| the signal sits inside the model's own level error; such
    tiers are neither drawn nor budgeted."""
    res = _solved()
    assert res.d_ref == 0.30
    everything, costs = explore.affordable_set(res, 1e12)
    assert everything == explore.admissible(res, 0.0)
    dmin = 0.10
    kept, _ = explore.affordable_set(res, 1e12, dmin)
    assert kept and len(kept) < len(everything)
    for j in everything:
        move = explore.log_move(res.d_ref, res.tiers[j])
        assert (j in kept) == (move >= dmin - 1e-9), (j, move)
    # a tier AT the reference is never admissible under a floor, however far
    # p* sits from it
    at_ref = [j for j in everything if abs(res.tiers[j] - res.d_ref) < 1e-9]
    assert at_ref and not any(j in kept for j in at_ref)
    # the ledger prices tau against the SAME set the chooser draws from
    assert explore.spread_costs(res, dmin) == [costs[j] for j in kept]
    # and the draw never lands below the floor
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = explore.select(res, 1e12, rng, delta_min=dmin)
        assert c["is_exploration"] and c["chosen_index"] in kept


def test_delta_min_is_derived_never_a_second_knob():
    """tau stays the one controller: the floor is k x bias / |eps| from a
    MEASURED bias scale and the cell's own eps, and 0 while no scale is
    pasted (null = no floor)."""
    none = dict(CFG, exploration=dict(CFG["exploration"], delta_min_log_bias=None))
    assert explore.delta_min(none, -1.0) == 0.0
    live = dict(CFG, exploration=dict(CFG["exploration"], delta_min_log_bias=0.30,
                                      delta_min_bias_multiple=1.0))
    assert explore.delta_min(live, -1.5) == pytest.approx(0.20)
    assert explore.delta_min(live, -0.5) == pytest.approx(0.60)
    # |eps| is floored at the sign constraint, never at zero
    assert explore.delta_min(live, -0.001) == pytest.approx(
        0.30 / abs(CFG["posterior"]["epsilon_max"]))
    # PER CATEGORY: its own bias, `_default` for one the backtest never saw,
    # and the reference_discount key convention ('SIDE DISH' -> SIDE_DISH)
    per = dict(CFG, exploration=dict(CFG["exploration"], delta_min_bias_multiple=1.0,
                                     delta_min_log_bias={"MEAT": 0.12, "SIDE_DISH": 0.40,
                                                         "_default": 0.15}))
    assert explore.delta_min(per, -1.0, "MEAT") == pytest.approx(0.12)
    assert explore.delta_min(per, -1.0, "SIDE DISH") == pytest.approx(0.40)
    assert explore.delta_min(per, -1.0, "FRUIT") == pytest.approx(0.15)


def test_a_mapping_without_the_category_or_a_default_is_an_error_not_no_floor():
    """A per-category mapping that names neither the category nor `_default`
    is a broken paste; reading it as "no floor" would silently let the
    uninformative tiers back into the draw."""
    broken = dict(CFG, exploration=dict(CFG["exploration"],
                                        delta_min_log_bias={"MEAT": 0.12}))
    assert explore.delta_min(broken, -1.0, "MEAT") == pytest.approx(0.12)
    with pytest.raises(KeyError, match="_default"):
        explore.delta_min(broken, -1.0, "FRUIT")
    with pytest.raises(KeyError):
        explore.delta_min(broken, -1.0)               # no category, no default


def test_the_deepening_threshold_treats_a_tier_width_gap_as_zero():
    """gamma - d within the grid epsilon is the same tier: inf, not a
    division by float noise."""
    P0, step = 10000.0, CFG["pricing"]["tier_step"]
    d = 0.4
    assert dp_mod.deepening_threshold_epsilon(P0, P0 * d, d) == float("inf")
    assert dp_mod.deepening_threshold_epsilon(
        P0, P0 * (d + dp_mod.TIER_EPS / 2), d) == float("inf")
    finite = dp_mod.deepening_threshold_epsilon(P0, P0 * (d + step), d)
    assert np.isfinite(finite) and finite == pytest.approx((1 - d) / step)


def test_a_forced_move_never_raises_the_price_within_an_episode():
    """Monotonicity is STRUCTURAL: under an anchor the DP's action set holds
    only tiers at or deeper than it, admissible is a subset of that set,
    and the draw comes from admissible -- so no floor, budget or draw can
    move a price back up. A forced move is always a deeper discount."""
    anchor = 0.30
    res = dp_mod.solve(10000.0, 4000.0, 6, [0.8] * 6, 0.30, -1.0, 0.9, CFG,
                       anchor_discount=anchor, entry=False)
    assert all(res.tiers[j] >= anchor - 1e-9 for j in res.q_by_tier)
    rng = np.random.default_rng(1)
    for dmin in (0.0, 0.05, 0.15):
        for j in explore.admissible(res, dmin):
            assert res.tiers[j] >= anchor - 1e-9
        for _ in range(30):
            c = explore.select(res, 1e12, rng, delta_min=dmin)
            assert res.tiers[c["chosen_index"]] >= anchor - 1e-9
