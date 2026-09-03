"""Unit tests for the decision core: tiers, DP, exploration, bounded step."""

import numpy as np
import pandas as pd
import pytest
import yaml

from common.config import load_config
from conftest import episode_frame
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
    assert np.isfinite(res.v_star)
    # and at the widest |epsilon| the search may return, where the divergence
    # at d -> 1 is steepest
    deep = dp_mod.solve(10000.0, 0.0, 3, [0.8] * 4, 0.30,
                        CFG["posterior"]["epsilon_min"], 0.919, CFG,
                        anchor_discount=0.0, entry=False)
    assert np.isfinite(deep.v_star)


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


def _last_row_frame(rows):
    """One row per episode: (episode_id, hr, start, sold, ending_inventory)."""
    return episode_frame(rows, columns=[
        "episode_id", "hours_remaining", "starting_inventory", "units_sold",
        "ending_inventory"], date=pd.Timestamp("2026-03-01").date(),
        hour_of_day=9)


def test_scrap_is_keyed_to_the_closure_sentinel_not_the_nominal_counter():
    """The counter is nominal and usually still positive when a listing ends,
    so `hours_remaining <= 0` classified ~99% of real leftover as unknown. What
    marks closure is the source's own sentinel: ending_inventory zeroed on the
    final row. Its ABSENCE is the only thing that makes an outcome unknown."""
    from common import episodes

    d = _last_row_frame([
        # counter at zero, stock left, sentinel present -- scrap. RARE in real
        # data: the counter reaches zero on ~0.1% of episodes.
        ("counter-zero", 0, 7, 0, 0),
        # counter STILL POSITIVE, stock left, sentinel present. The common
        # case, and it must count as scrap rather than unknown.
        ("early-leftover", 28, 9, 4, 0),
        # sold out: a genuine zero, and unambiguous whatever the sentinel says
        ("sold-out", 4, 5, 5, 0),
        # stock left and NO sentinel -- still open, or the feed cut it
        ("still-open", 6, 9, 4, 5),
    ])

    kind = episodes.classify(d)
    assert kind["counter-zero"] == episodes.COMPLETED
    assert kind["early-leftover"] == episodes.COMPLETED     # the fix
    assert kind["sold-out"] == episodes.SOLD_OUT_EARLY
    assert kind["still-open"] == episodes.NOT_CLOSED

    scrap = episodes.scrap_units(d)
    assert scrap["counter-zero"] == 7
    assert scrap["early-leftover"] == 5      # 9 - 4, NOT dropped as unknown
    assert scrap["sold-out"] == 0
    assert pd.isna(scrap["still-open"])      # unknown, NOT zero and NOT 5

    # the regression this test exists for: under the counter-keyed rule only
    # `counter-zero` scrapped, so 5 of the 12 knowable units vanished
    assert scrap.sum() == 12


def test_a_feed_with_no_closure_sentinel_reads_unclosed_and_says_so():
    """Closure is `ending_inventory == 0` on the last row and NOTHING else."""
    from common import episodes

    honest = _last_row_frame([("a", 3, 9, 4, 5), ("b", 2, 6, 6, 0)])
    assert not episodes.write_off_convention(honest)     # <- read this first
    kind = episodes.classify(honest)
    assert kind["a"] == episodes.NOT_CLOSED
    # "b" sold out AND its ending is genuinely 0, so it closed on its own
    # evidence -- the sentinel's absence elsewhere does not touch it
    assert kind["b"] == episodes.SOLD_OUT_EARLY
    assert pd.isna(episodes.scrap_units(honest)["a"])

    mixed = _last_row_frame([("a", 3, 9, 4, 5), ("w", 1, 8, 2, 0)])
    assert episodes.write_off_convention(mixed)
    assert episodes.classify(mixed)["a"] == episodes.NOT_CLOSED
    assert pd.isna(episodes.scrap_units(mixed)["a"])


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


def test_state_rejected_not_priced():
    from inference.decide import decide, StateRejected

    with pytest.raises(StateRejected):
        decide({
            "episode_id": "x", "sku_id": 1, "fc": "F", "category": "MEAT",
            "subcategory": "PORK", "date": "2026-08-01", "hour_of_day": 12,
            "hours_remaining": 2,
            "q": 3, "original_price": -5.0, "cost": 10.0, "r": 1.0,
            "mu_ref_path": [1.0, 1.0], "current_discount": None,
        }, None, None, CFG, np.random.default_rng(0), 100.0, "v")


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
    from common.episodes import adjustment_reason as why

    # a reported ZERO with stock remaining is the source's write-off, wherever
    # it falls. Position must NOT matter: the source zeroes at its own episode
    # boundary, which sits mid-episode once we merge a window across midnight.
    assert why(4, 3, 0) == "episode_close_write_off"
    assert why(9, 4, 0) == "episode_close_write_off"
    # clean sellout reconciles on its own, no reason needed
    assert why(3, 3, 0) is None
    # stock added
    assert why(5, 1, 8) == "intraday_restock"
    # ordinary hour that reconciles
    assert why(5, 1, 4) is None
    # PARTIAL shortfall -- above zero but below the leftover -- is SHRINK, and
    # it is NAMED. It returned None on purpose until it was measured, so that
    # unexplained loss would quarantine and stay visible. That was the last
    # place the live path called shrink an anomaly while the offline chain
    # called it an ordinary event: counted gross, booked into scrap, gating
    # nothing. A quarantined outcome never lands, so event completeness fell by
    # the feed's whole shrink rate and the shadow gate failed for something no
    # integration work could fix -- it was measuring the SOURCE. At ~2.8% of
    # decision hours the harness read 0.9718 against a 0.99 threshold.
    assert why(5, 1, 2) == "unexplained_shortfall"
    # ORDER MATTERS at the boundary: a zero ending is the CLOSE, not a shrink.
    # Asking the shortfall first would swallow every write-off there is.
    assert why(5, 1, 0) == "episode_close_write_off"


def test_noise_floor_and_monitor_use_the_same_smoothing():
    """The floor the owner sets a threshold from and the series the monitor
    triggers on must be averaged identically, or the threshold is graded
    against a yardstick nothing uses. (That both sides read the shared
    comparison and smoothing is asserted where the shared module is.)"""
    sm = CFG["monitoring"]["stop_conditions"]["deterioration_smoothing_days"]
    assert set(sm) == {"scrap", "margin"}
    # scrap is a low-base series and needs averaging; margin does not
    assert sm["scrap"] > 1 and sm["margin"] == 1


def _paired_arm_frame(days=70, skus=60, seed=0):
    """Episode-hour rows spanning both A/B arms, with a common day effect so
    the same-day comparison has something to cancel."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, day in enumerate(pd.date_range("2026-01-01", periods=days)):
        day_effect = 1.0 + 0.35 * np.sin(i / 3.0)      # shared by both arms
        for sku in range(skus):
            sold = int(np.clip(rng.poisson(9 * day_effect), 0, 20))
            rows.append(dict(episode_id=f"{day.date()}|{sku}", date=day.date(),
                             hour_of_day=10, sku_id=sku, fc="FC1",
                             starting_inventory=20, units_sold=sold,
                             ending_inventory=0, hours_remaining=0,
                             offered_price=1000.0, original_price=1000.0,
                             cost=600.0))
    return pd.DataFrame(rows)


def test_control_arm_floor_is_measured_on_the_smoothed_series():
    """Smoothing must actually be applied, not just mentioned. Measuring the
    same data at smoothing 1 must give a strictly wider floor -- if the two
    agree, the smoothing is being ignored and a threshold set from this floor
    sits several times above its true operating noise."""
    import copy
    from bootstrap import derive_thresholds as dt

    d = _paired_arm_frame()
    cfg_smoothed = copy.deepcopy(CFG)
    cfg_flat = copy.deepcopy(CFG)
    cfg_flat["monitoring"]["stop_conditions"]["deterioration_smoothing_days"] \
        ["scrap"] = 1

    smoothed = dt.control_arm_noise(d, cfg_smoothed)["scrap_rate"]
    flat = dt.control_arm_noise(d, cfg_flat)["scrap_rate"]

    assert smoothed["smoothing_days"] == \
        CFG["monitoring"]["stop_conditions"]["deterioration_smoothing_days"]["scrap"]
    assert flat["smoothing_days"] == 1
    assert smoothed["three_sigma"] < flat["three_sigma"]
    # smoothing consumes the leading window
    assert smoothed["days"] < flat["days"]


def test_threshold_recommendation_binds_on_the_larger_floor():
    """One config value is graded against the trailing mean before the A/B and
    the control arm during it, so it must clear both -- and a value far above
    the binding floor is called out rather than blessed."""
    import copy
    from bootstrap import derive_thresholds as dt

    d = _paired_arm_frame()
    cfg = copy.deepcopy(CFG)
    trailing = dt.guardrail_noise(d, cfg)
    control = dt.control_arm_noise(d, cfg)

    rec = dt.recommend_thresholds(trailing, control, cfg)["scrap_rate"]
    floors = [f for f in (rec["trailing_floor"], rec["control_arm_floor"])
              if f is not None]
    assert rec["binding_floor"] == max(floors)

    binding = rec["binding_floor"]
    sc = cfg["monitoring"]["stop_conditions"]

    sc["scrap_deterioration_pct"] = binding / 2
    assert "TOO TIGHT" in dt.recommend_thresholds(
        trailing, control, cfg)["scrap_rate"]["verdict"]

    sc["scrap_deterioration_pct"] = binding * 1.5
    assert dt.recommend_thresholds(
        trailing, control, cfg)["scrap_rate"]["verdict"].startswith("OK")

    # a guardrail that cannot fire is a failure mode of its own, not a pass
    sc["scrap_deterioration_pct"] = binding * 20
    assert "INERT" in dt.recommend_thresholds(
        trailing, control, cfg)["scrap_rate"]["verdict"]


def test_a_relative_floor_above_one_is_reported_as_blocked_not_as_a_number():
    """A floor at or above 1.0 means the series' ordinary daily swing exceeds
    its own level. No threshold can clear it without also clearing the failure
    the guardrail exists to catch -- so it is BLOCKED, and the report has to
    say the word."""
    import copy
    from bootstrap import derive_thresholds as dt
    from common import guardrail

    assert guardrail.floor_is_unusable(1.0, guardrail.RELATIVE)
    assert guardrail.floor_is_unusable(3.5853, guardrail.RELATIVE)
    assert not guardrail.floor_is_unusable(0.4156, guardrail.RELATIVE)
    # absolute-pp floors are in the metric's own units and have no such bound
    assert not guardrail.floor_is_unusable(3.5853, guardrail.ABSOLUTE_PP)

    cfg = copy.deepcopy(CFG)
    trailing = {"margin_rate": {"three_sigma": 3.5853,
                                "three_sigma_robust": 3.5853,
                                "outlier_dominated": True,
                                "mean_level": 0.0308, "days": 134,
                                "days_at_or_below_zero": 36}}
    # on the wrong (relative) basis the floor is unusable and must BLOCK...
    saved = guardrail.BASIS["margin"]
    try:
        guardrail.BASIS["margin"] = guardrail.RELATIVE
        v = dt.recommend_thresholds(trailing, {}, cfg)["margin_rate"]["verdict"]
    finally:
        guardrail.BASIS["margin"] = saved
    assert v.startswith("BLOCKED"), v
    assert "absolute_pp" in v, "the verdict must name the remedy, not just complain"

    # ...and on the shipped absolute_pp basis it is an ordinary, settable floor
    v2 = dt.recommend_thresholds(trailing, {}, cfg)["margin_rate"]["verdict"]
    assert not v2.startswith("BLOCKED"), v2


def test_the_information_increment_is_derived_from_the_posterior_arithmetic(
        tmp_path):
    """`information_increment` is the evidence a bounded update may USE, and
    that is fixed by the posterior's own algebra rather than judgment."""
    import copy
    import json
    import numpy as np

    from bootstrap import derive_thresholds as dt

    cfg = copy.deepcopy(CFG)
    prior_path = tmp_path / "prior.json"
    # two cells an octave apart in width: 4x the precision, 4x the cost
    prior_path.write_text(json.dumps({"per_category": {
        "WIDE": {"mean": -1.0, "std": 1.0},
        "NARROW": {"mean": -1.0, "std": 0.5}}}))
    cfg["posterior"]["prior"]["path"] = str(prior_path)
    cfg["learning"]["max_std_shrink"] = 0.25

    out = dt.information_increment(cfg)
    k = 1.0 / 0.75 ** 2 - 1.0
    need = out["information_to_saturate_cap_by_category"]
    assert need["WIDE"] == pytest.approx(k, abs=1e-3)
    assert need["NARROW"] == pytest.approx(k / 0.25, abs=1e-3)
    assert need["NARROW"] == pytest.approx(4 * need["WIDE"], rel=1e-3), \
        "halving the std must quadruple the information -- precision is 1/s^2"

    # the ceiling is checkable against the posterior step it claims to fund:
    # spending exactly I* on the wide cell lands on the shrink cap
    s0 = 1.0
    s1 = 1.0 / np.sqrt(1.0 / s0 ** 2 + need["WIDE"])
    # tolerance is what the report's 3dp rounding permits, nothing looser
    assert s1 == pytest.approx(s0 * (1 - cfg["learning"]["max_std_shrink"]),
                               rel=1e-3)

    # and an over-sized configured value is named, with the std it was
    # implicitly sized for -- the number that says what it was fitted to
    cfg["learning"]["information_increment"] = 12.0
    over = dt.information_increment(cfg)
    assert over["verdict"].startswith("TOO LARGE")
    assert over["configured_implied_std"] == pytest.approx(
        np.sqrt(k / 12.0), abs=1e-3)
    assert over["wastes_at_launch"] > 1

    cfg["learning"]["information_increment"] = over["recommended"]
    assert dt.information_increment(cfg)["verdict"].startswith("OK")


def test_the_two_bounded_step_rails_are_graded_against_each_other(tmp_path):
    """`max_mean_step` and `max_std_shrink` are one decision expressed twice.
    A cap-sized update moves the mean by [1-(1-shrink)^2] x pull, so if the
    mean rail sits far below that it clips every ordinary batch while the
    shrink rail never binds -- and `bound_clipped` stops meaning anything.
    The report has to say which rail binds first, and at what surprise."""
    import copy
    import json

    from bootstrap import derive_thresholds as dt

    cfg = copy.deepcopy(CFG)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(
        {"per_category": {"A": {"mean": -1.0, "std": 1.0}}}))
    cfg["posterior"]["prior"]["path"] = str(prior_path)
    cfg["learning"]["max_std_shrink"] = 0.25

    # a mean rail far under the cap-sized move: clips on ordinary batches
    cfg["learning"]["max_mean_step"] = 0.15
    tight = dt.bounded_step(cfg)
    assert tight["mean_move_fraction_of_pull_at_cap"] == pytest.approx(0.4375)
    assert tight["consistent_max_mean_step"] == pytest.approx(0.4375, abs=1e-3)
    assert tight["mean_rail_clips_above_pull_of_std"] == pytest.approx(
        0.15 / 0.4375, abs=1e-2)
    assert tight["verdict"].startswith("MEAN RAIL BINDS FIRST")
    # the verdict must name the owner's actual options, not just complain
    assert "step_sensitivity" in tight["verdict"]

    # set to the consistent value and both rails trip at a 1-std surprise
    cfg["learning"]["max_mean_step"] = tight["consistent_max_mean_step"]
    assert dt.bounded_step(cfg)["verdict"].startswith("CONSISTENT")

    # and the shrink cap alone fixes convergence: 25%/update from 1.0 to the
    # 0.05 floor is log(0.05)/log(0.75) updates, nothing to do with the mean
    assert dt.bounded_step(cfg)["updates_to_min_std_by_category"]["A"] == \
        pytest.approx(np.log(cfg["posterior"]["min_std"]) / np.log(0.75), abs=0.1)


def test_the_floor_and_the_trigger_compute_the_same_quantity():
    """`derive_thresholds` measures the floor and `pipeline.monitor` evaluates
    the trigger. If they compute different things the threshold is graded
    against a yardstick nothing uses, and nothing downstream would notice --
    so both import the comparison from `common.guardrail` rather than keeping
    two implementations that resemble each other."""
    import inspect
    from bootstrap import derive_thresholds as dt
    from pipeline import monitor

    for mod in (dt, monitor):
        src = inspect.getsource(mod)
        assert "guardrail.deviation(" in src or "guard.deviation(" in src, \
            f"{mod.__name__} is not using the shared comparison"
        assert "guardrail.smooth(" in src or "guard.smooth(" in src \
            or "_smooth = guardrail.smooth" in src, \
            f"{mod.__name__} is not using the shared smoothing"

    # the sign convention is the caller's, and it must be opposite per metric
    from common import guardrail
    import pandas as pd
    t, c = pd.Series([0.12]), pd.Series([0.10])
    for basis in (guardrail.RELATIVE, guardrail.ABSOLUTE_PP):
        # scrap: higher is worse -> positive
        assert guardrail.deviation(t, c, True, basis).iloc[0] > 0
        # margin: higher is BETTER -> negative
        assert guardrail.deviation(t, c, False, basis).iloc[0] < 0


def _bracket_verdict(naive, ctrl, cfg, lo=-4.0, hi=-0.05, step=0.025):
    """The acceptance decision from bootstrap.estimate_prior, in isolation."""
    pc = cfg["posterior"]["prior"]
    boundary = any(abs(e - b) <= step + 1e-12 for e in (naive, ctrl)
                   for b in (lo, hi))
    why = []
    if not (naive < 0 and ctrl < 0):
        why.append("wrong sign")
    if boundary and pc["reject_boundary_solutions"]:
        why.append("boundary solution")
    if naive > ctrl and pc["reject_orientation_violations"]:
        why.append("orientation violated")
    return why


def test_a_censored_entry_row_is_a_one_hour_episode():
    """Which is why dropping them is cheap -- and why the cost is a selection
    bias, not a coverage one."""
    from common import episodes

    d = pd.DataFrame({
        "episode_id": ["a", "a", "b"],
        "date": ["2026-03-01"] * 3,
        "hour_of_day": [10, 11, 10],
        "starting_inventory": [5, 3, 4],
        "units_sold": [2, 3, 4],       # a closes by sell-out; b is one hour
        "ending_inventory": [3, 0, 0],
    })
    cen = episodes.censored_hours(d)
    entry_idx = d.sort_values(["episode_id", "hour_of_day"]).groupby(
        "episode_id").head(1).index

    # episode a: censored on its LAST row (index 1), not its entry row (0)
    assert cen[1] and not cen[0]
    # episode b is one hour, so entry IS last -- the only censored entry row
    assert cen[2]
    assert list(pd.Series(cen)[entry_idx]) == [False, True]


def test_the_bounded_step_holds_in_the_REPORT_not_just_in_the_store(): # noqa: N802
    """`max_mean_step` is a safety bound, and something checks it on the
    REPORT rather than on the posterior file."""
    import inspect
    from pipeline import update as up

    src = inspect.getsource(up)
    for field in ("proposed_mean", "proposed_std", "raw_mean", "raw_std"):
        assert f'"{field}": round(' not in src, (
            f"{field} is rounded again -- it is compared against the unrounded "
            "mean_before, so the step can read over max_mean_step")

    # and the clamp itself is exact, at the boundary and past it
    cap = CFG["learning"]["max_mean_step"]
    for raw in (-1.0 - cap, -1.0 - cap * 3, -1.0 + cap * 3):
        m, _, _ = bounded_step(-1.0, 0.6, raw, 0.5, CFG)
        assert abs(m - (-1.0)) <= cap + 1e-12, (raw, m)
    # an awkward starting mean, the shape that exposed the reporting bug
    m, _, _ = bounded_step(-0.76098, 0.6, -3.0, 0.5, CFG)
    assert abs(m - (-0.76098)) <= cap + 1e-12


def test_an_under_dispersed_group_is_exempt_from_the_clamp():
    """The clamp must not make the steadiest cells claim variance they lack."""
    from bootstrap.fit_dispersion import pearson_dispersion

    rng = np.random.default_rng(4)
    mu = 6.0
    # binomial with the same mean is UNDER-dispersed: var = mu(1-p) < mu
    tight = rng.binomial(12, mu / 12, 4000)
    assert pearson_dispersion(tight, np.full(len(tight), mu)) < 1.0
    # negative binomial at the same mean is over-dispersed
    loose = rng.negative_binomial(1.5, 1.5 / (1.5 + mu), 4000)
    assert pearson_dispersion(loose, np.full(len(loose), mu)) > 1.0
    # Poisson sits at 1 either side of noise
    poi = rng.poisson(mu, 20000)
    assert 0.9 < pearson_dispersion(poi, np.full(len(poi), mu)) < 1.1

    import inspect
    from bootstrap import fit_dispersion as fd

    src = inspect.getsource(fd.fit_dispersion)
    assert "under" in src and "pearson_dispersion(" in src, \
        "the clamp must consult Pearson dispersion, not just the fitted r"
    assert "under_dispersed_groups" in src, \
        "an NB misfit must be REPORTED, not absorbed into a percentile"


def test_cogs_at_risk_counts_supply_not_opening_stock():
    """A window that opens with 3 and takes 10 mid-flight has 13 units of
    cost at risk; counting 3 understates every restocked episode."""
    from bootstrap.prepare_data import cogs_at_risk

    # one episode: opens with 3, 10 arrive in hour 2, sells 9, loses 1
    d = pd.DataFrame({
        "episode_id": ["e"] * 3,
        # `hour_adjustment` establishes window order from these, so the
        # arrival term needs them -- every real caller has them, since
        # `assign_episode_ids` needs them first
        "date": ["2026-03-01"] * 3, "hour_of_day": [10, 11, 12],
        "cost": [100.0] * 3,
        "starting_inventory": [3, 13, 4],
        "units_sold": [0, 9, 3],
        "ending_inventory": [13, 4, 0],
    })
    # 3 opening + 10 arrived = 13 units x 100
    assert cogs_at_risk(d) == pytest.approx(1300.0)

    # no arrivals -> unchanged from the old opening-stock reading
    flat = pd.DataFrame({
        "episode_id": ["f"] * 2, "cost": [50.0] * 2,
        "date": ["2026-03-01"] * 2, "hour_of_day": [10, 11],
        "starting_inventory": [8, 5], "units_sold": [3, 5],
        "ending_inventory": [5, 0],
    })
    assert cogs_at_risk(flat) == pytest.approx(400.0)


def test_the_fixture_generator_covers_the_configured_splits():
    """`bootstrap.run --input <fixture>` must run end to end from a clean
    checkout. It could not: the generator started at a hardcoded date for 90
    days, the exclusion window removed the tail, and the data ended in April
    while config's calib window began in July -- so `fit_dispersion` died with
    "calibration window contains no rows" and the prior's held-out comparison
    came back empty, both silently about the cause."""
    import datetime as dt
    from tools.make_dummy_flc import span_covering_splits

    split = CFG["data"]["split"]
    start, days = span_covering_splits(CFG)
    assert start == dt.date.fromisoformat(str(split["train_start"]))
    assert start + dt.timedelta(days=days - 1) >= \
        dt.date.fromisoformat(str(split["test_end"])), \
        "the generated span must reach test_end, or the gate window is empty"

    # and it must still run standalone, with no config to read
    fallback_start, fallback_days = span_covering_splits({})
    assert fallback_days > 0 and fallback_start.year == 2026


def test_a_prior_std_can_never_be_zero():
    """A zero-width prior is not confidence, it is a frozen posterior."""
    import numpy as np

    from bootstrap import prior_density as pdn

    grid = np.linspace(-4.0, -0.05, 159)
    step = grid[1] - grid[0]
    # a likelihood dropping 59 nats per grid step: exactly the production case
    ll = -59.5 * np.arange(len(grid))[::-1]
    w = pdn.density(ll, 1.0)
    _, raw_std = pdn.moments(grid, w)
    assert raw_std == pytest.approx(0.0, abs=1e-9), \
        "the collapse this guards against must actually happen"

    # nothing finer than one grid cell was ever resolved, so the reported std
    # can never be below the grid's own step
    assert max(raw_std, step) >= step


def test_a_likelihood_peaking_at_positive_elasticity_is_rejected():
    """`search_bounds` is a policy statement, not a belief about demand, so a
    peak outside it gets CLIPPED to the nearest bound and reported as measured.
    That is how a category whose likelihood prefers demand RISING with price
    came back as a confident -0.05 with four decimals."""
    import inspect

    from bootstrap import prior_density as pdn

    src = inspect.getsource(pdn.estimate)
    assert "wrong_sign" in src and "unconstrained" in src, \
        "the sign check must consult the UNCONSTRAINED peak, not the clipped one"
    # the search must actually go past the upper bound, or it cannot see a
    # positive optimum at all
    wide = inspect.getsource(pdn.unconstrained_argmax)
    assert "max(1.0, hi)" in wide, \
        "the unconstrained search must extend past the policy bound"

    # and a rejected category must not contaminate the pool it falls back to
    assert "signs[cat][0]" in src, \
        "the pool must exclude wrong-sign categories -- otherwise the fallback " \
        "inherits the confound the rejection exists to remove"


def test_the_hour_control_is_keyed_on_the_day_not_just_the_clock():
    """Design 5.6 says "same-hour CROSS-EPISODE", and a control pooled across
    dates is only an approximation to it: it removes the average evening lift
    and leaves a Tuesday storm or a rival's promotion in the residual, still
    correlated with how far the legacy ramp has run."""
    import inspect

    import numpy as np
    import pandas as pd

    from bootstrap import prior_density as pdn

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


def test_dispersion_drift_separates_a_failed_fit_from_a_moved_parameter():
    """The measurement that says whether freezing r and rho is defensible --
    and the trap it has to avoid."""
    import json
    import os

    from bootstrap import fit_dispersion as fd

    # the discriminator itself: below 1 is inexpressible, above is ordinary
    steady = np.full(400, 2.0)
    assert fd.pearson_dispersion(steady, np.full(400, 2.0)) < 1.0
    rng = np.random.default_rng(0)
    bursty = rng.negative_binomial(0.7, 0.7 / (0.7 + 2.0), 400)
    assert fd.pearson_dispersion(bursty, np.full(400, 2.0)) > 1.0

    path = "artifacts/rho.json"
    if not os.path.exists(path):
        pytest.skip("no rho artifact on disk")
    drift = json.load(open(path)).get("drift_by_window")
    if not drift or not drift.get("windows_fitted"):
        pytest.skip("no drift block in the artifact")

    for w, v in drift["by_window"].items():
        # every window declares whether its r means anything, and why
        assert set(("pearson", "nb_expressible", "r_at_search_bound",
                    "r_usable")) <= set(v), w
        if v["pearson"] < 1.0:
            assert not v["r_usable"], \
                f"{w}: Pearson < 1 but its r is being treated as a measurement"

    # the headline r spread must be over USABLE windows only
    usable = [v["r"] for v in drift["by_window"].values() if v["r_usable"]]
    if usable:
        assert drift["r_spread"] == pytest.approx(
            max(usable) - min(usable), abs=1e-3)
        assert drift["r_windows_usable"] == len(usable)

    # and when most windows are unfittable the verdict says THAT, not "drift"
    if drift["r_unusable_share"] > 0.34:
        assert "NOT FITTABLE AT THIS CADENCE" in drift["verdict"]


def test_the_tier_epsilon_has_one_home():
    """Seven modules carried their own 1e-9 for "is this the same tier"."""
    import inspect
    from pricing import dp
    from inference import decide
    from pipeline import shadow
    assert dp.TIER_EPS == 1e-9
    for fn in (dp.feasible_tiers, dp.entry_action_set, dp.solve):
        assert "1e-9" not in inspect.getsource(fn), fn.__name__
    assert "TIER_EPS" in inspect.getsource(decide)
    assert "TIER_EPS" in inspect.getsource(shadow)


def _solved(eps=-1.0):
    return dp_mod.solve(10000.0, 4000.0, 6, [0.8] * 6, 0.30, eps, 0.9, CFG,
                        anchor_discount=None, entry=True)


def test_delta_min_removes_the_tiers_the_model_cannot_tell_apart():
    """A forced move smaller than bias/|eps| in log price has its signal
    inside the model's own level error; it is neither drawn nor budgeted.
    Shadow spent ~22% of decisions there, all one tier step out."""
    res = _solved()
    star = res.optimal_index
    everything, costs = explore.affordable_set(res, 1e12)
    assert everything == explore.admissible(res, 0.0)
    dmin = 0.10
    kept, _ = explore.affordable_set(res, 1e12, dmin)
    assert kept and len(kept) < len(everything)
    for j in everything:
        move = explore.log_move(res.tiers[star], res.tiers[j])
        assert (j in kept) == (move >= dmin - 1e-9), (j, move)
    # the ledger prices tau against the SAME set the chooser draws from
    assert explore.spread_costs(res, dmin) == [costs[j] for j in kept]
    # and the draw never lands below the floor
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = explore.select(res, 1e12, rng, delta_min=dmin)
        assert c["is_exploration"] and c["chosen_index"] in kept


def test_delta_min_is_derived_never_a_second_knob():
    """tau stays the one controller: the floor is k x bias / |eps| from a
    MEASURED bias scale and the cell's own eps, and 0 until tune pastes
    the scale -- the fixture ships no number."""
    assert CFG["exploration"]["delta_min_log_bias"] is None
    assert explore.delta_min(CFG, -1.0) == 0.0
    live = dict(CFG, exploration=dict(CFG["exploration"], delta_min_log_bias=0.30,
                                      delta_min_bias_multiple=1.0))
    assert explore.delta_min(live, -1.5) == pytest.approx(0.20)
    assert explore.delta_min(live, -0.5) == pytest.approx(0.60)
    # |eps| is floored at the sign constraint, never at zero
    assert explore.delta_min(live, -0.001) == pytest.approx(
        0.30 / abs(CFG["posterior"]["epsilon_max"]))
