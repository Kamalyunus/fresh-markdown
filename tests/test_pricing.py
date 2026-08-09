"""Unit tests for the decision core: tiers, DP, exploration, bounded step."""

import numpy as np
import pandas as pd
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


def test_dp_terminal_value_and_entry_arms():
    mu_path = [1.5, 1.2, 1.0]
    d_ref = 0.30
    res = dp_mod.solve(10000, 6000, q0=5, mu_ref_path=mu_path, d_ref=d_ref,
                       epsilon=-1.2, r=1.0, cfg=CFG, entry=True)
    offsets = CFG["pricing"]["entry_offsets"]
    chosen = sorted(round(res.tiers[j], 6) for j in res.q_by_tier)
    assert chosen == sorted(round(d_ref + o, 6) for o in offsets)
    # value can never be worse than scrapping everything immediately
    assert res.v_star >= -6000 * 5


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


def test_implausible_window_is_refused_not_expanded():
    """flc_window carries very large values from upstream data issues. The
    window drives episode identification, the DP horizon AND the synthetic
    tail, so a bad value must be dropped upstream -- and if one ever reaches
    the extension it must raise, not generate an unbounded frame."""
    from common import episodes

    d = pd.DataFrame({
        "episode_id": ["e"], "date": [pd.Timestamp("2026-03-01").date()],
        "hour_of_day": [10], "hours_remaining": [9000],
        "starting_inventory": [3], "ending_inventory": [3],
        "units_sold": [0], "category": ["MEAT"],
    })
    with pytest.raises(ValueError, match="exceeds max_window_hours"):
        episodes.extend_to_window(d, ["category"], max_tail_hours=48)

    ok = d.assign(hours_remaining=[5])
    assert len(episodes.extend_to_window(ok, ["category"], 48)) == 6


def test_window_extension_removes_the_lookahead_horizon():
    """Rows stop at zero inventory, so the DP's horizon must come from the
    window, not from how many rows happen to exist -- a short row count is
    short BECAUSE the item sold out, which is future information."""
    from common import episodes

    # an 8-hour window that sold out after 3 hours
    d = pd.DataFrame({
        "episode_id": ["e"] * 3,
        "date": [pd.Timestamp("2026-03-01").date()] * 3,
        "hour_of_day": [10, 11, 12],
        "hours_remaining": [7, 6, 5],
        "starting_inventory": [3, 2, 1],
        "ending_inventory": [2, 1, 0],
        "units_sold": [1, 1, 1],
        "category": ["MEAT"] * 3,
    })
    assert len(d) == 3 and d.hours_remaining.iloc[0] + 1 == 8

    e = episodes.extend_to_window(d, ["category"])
    assert len(e) == 8, "the DP must see the whole window, not the 3 rows"
    assert e.hours_remaining.iloc[-1] == 0
    assert e.is_observed.sum() == 3 and (~e.is_observed).sum() == 5

    # rows remaining now equals the window at every row -- the invariant
    # validate_state enforces on the live path
    rows_left = np.arange(len(e), 0, -1)
    assert (rows_left == e.hours_remaining.to_numpy() + 1).all()

    # synthetic rows carry features but no sales, so observed-world
    # economics and fidelity are untouched by the extension
    assert e[~e.is_observed].units_sold.eq(0).all()
    assert e.category.notna().all()

    # a window that ran to the end is left exactly as it was
    done = d.assign(hours_remaining=[2, 1, 0])
    assert len(episodes.extend_to_window(done, ["category"])) == 3


def test_scrap_counted_only_where_the_window_actually_ran_out():
    """An episode stops at the window end OR at zero inventory, whichever
    comes first. Only the first ending scrapped anything; the second scrapped
    nothing, and a truncated episode's leftover is simply unknown."""
    from common import episodes

    # the source writes the remainder off at the window close, so
    # ending_inventory is ZERO on every last row -- scrap must come from
    # max(0, starting_inventory - units_sold) instead
    hr = pd.Series([0, 0, 4, 3])            # window counter at the last row
    start = pd.Series([7, 3, 5, 9])         # inventory entering the last hour
    sold = pd.Series([0, 3, 5, 4])          # write-off / sellout / sellout / left
    kind = episodes.classify(hr, start, sold)
    assert list(kind) == [episodes.COMPLETED, episodes.COMPLETED,
                          episodes.SOLD_OUT_EARLY, episodes.TRUNCATED]

    scrap = episodes.scrap_units(hr, start, sold)
    assert scrap.iloc[0] == 7                # window ran out, 7 written off
    assert scrap.iloc[1] == 0                # ran out with nothing left
    assert scrap.iloc[2] == 0                # sold out early: nothing to scrap
    assert pd.isna(scrap.iloc[3])            # unknown, NOT zero and NOT 5

    # reading the zeroed ending_inventory instead would report NO scrap at all
    zeroed_ending = pd.Series([0, 0, 0, 0])
    assert zeroed_ending.sum() == 0 and scrap.sum() == 7


def test_state_rejected_when_planning_horizon_disagrees_with_recorded_one():
    """A window truncated at a date boundary looks exactly like this: the
    caller believes the episode runs longer than the path it supplied."""
    from inference.decide import validate_state

    base = {"original_price": 10000.0, "cost": 6000.0, "q": 3,
            "hours_remaining": 4, "r": 1.0}
    tiers = [0.0, 0.025, 0.05]

    assert validate_state(base, tiers, None, [1.0, 1.0, 1.0, 1.0]) == []

    failures = validate_state(base, tiers, None, [1.0, 1.0])
    assert any("planning horizon" in f for f in failures)


def test_guardrail_fires_only_after_persistence():
    """The owner thresholds must actually be evaluated -- and must not fire on
    a single day over, which is what the noise floor makes routine."""
    from pipeline.monitor import evaluate_guardrail

    block = {"basis": "control_arm", "latest": 0.30,
             "by_day": {"2026-09-01": 0.05, "2026-09-02": 0.30}}

    one_day = evaluate_guardrail(block, threshold=0.20, persistence_days=2)
    assert not one_day["fired"] and one_day["consecutive_days_over"] == 1

    block["by_day"]["2026-09-03"] = 0.25
    two_days = evaluate_guardrail(block, threshold=0.20, persistence_days=2)
    assert two_days["fired"] and two_days["consecutive_days_over"] == 2

    # a day back under the threshold breaks the streak
    block["by_day"]["2026-09-04"] = 0.01
    assert not evaluate_guardrail(block, 0.20, 2)["fired"]

    # a null threshold is blocked, never silently passing
    blocked = evaluate_guardrail(block, None, 2)
    assert not blocked["fired"] and "BLOCKED" in blocked["status"]


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


def test_write_off_outcome_is_documented_not_quarantined():
    """The source zeroes ending_inventory at the window close (~49.5% of
    episodes). Unnamed, every one of those final-hour outcomes quarantines
    and event completeness collapses -- the shadow gate fails for what looks
    like a pipeline defect."""
    from events.store import _validate_outcome

    base = {"outcome_id": "o", "decision_id": "d", "units_sold": 3,
            "starting_inventory": 4, "ending_inventory": 0,
            "applied_price": 5000.0, "is_stockout": False,
            "execution_status": "ok", "finalized_at": "2026-03-01T20:00:00Z"}

    # 4 in, 3 sold -> 1 left, reported as 0: does not reconcile
    assert _validate_outcome(base), "must not pass undocumented"

    assert not _validate_outcome({**base,
                                  "adjustment_reason": "episode_close_write_off"})
    assert not _validate_outcome({**base, "ending_inventory": 5,
                                  "adjustment_reason": "intraday_restock"})

    # a clean reconciliation needs no reason at all
    assert not _validate_outcome({**base, "ending_inventory": 1})


def test_adjustment_reason_names_every_legitimate_break():
    """Anything legitimate but unnamed quarantines, and a quarantined outcome
    never lands -- so a naming gap shows up as failed event completeness, not
    as a labelling bug."""
    from pipeline.shadow import adjustment_reason as why

    # window ran out with stock left: written off at episode close
    assert why(4, 3, 0, True) == "episode_close_write_off"
    # TRUNCATED -- closes with stock left and window time to spare. Same
    # write-off; keying this to hours_remaining == 0 left it quarantining.
    assert why(9, 4, 0, True) == "episode_close_write_off"
    # clean sellout reconciles on its own, no reason needed
    assert why(3, 3, 0, True) is None
    # stock added mid-episode
    assert why(5, 1, 8, False) == "intraday_restock"
    # ordinary mid-episode hour that reconciles
    assert why(5, 1, 4, False) is None
    # shortfall part-way through is unexplained loss: must stay unnamed so it
    # quarantines rather than being absorbed
    assert why(5, 1, 2, False) is None
