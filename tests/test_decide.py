"""engine.decide: what a state must look like to be priced, and what is
rejected by name rather than crashing the pricing loop; and the DP it
calls, held bit-identical to the scalar loop it replaced."""

import numpy as np
import pytest
from scipy.stats import nbinom

from conftest import CFG
from engine import dp as dp_mod
from engine.demand import mu_at


def test_state_rejected_when_planning_horizon_disagrees_with_recorded_one():
    """A window truncated at a date boundary looks exactly like this: the
    caller believes the episode runs longer than the path it supplied."""
    from engine.decide import validate_state

    base = {"original_price": 10000.0, "cost": 6000.0, "q": 3,
            "hours_remaining": 4, "r": 1.0}
    tiers = [0.0, 0.025, 0.05]

    assert validate_state(base, tiers, None, [1.0, 1.0, 1.0, 1.0]) == []

    failures = validate_state(base, tiers, None, [1.0, 1.0])
    assert any("planning horizon" in f for f in failures)


def _state(**over):
    s = {"episode_id": "x", "sku_id": 1, "fc": "F", "category": "MEAT",
         "subcategory": "PORK", "date": "2026-08-01", "hour_of_day": 12,
         "hours_remaining": 2, "q": 3, "original_price": 10000.0,
         "cost": 4000.0, "r": 1.0, "mu_ref_path": [1.0, 1.0],
         "current_discount": None}
    s.update(over)
    return s


def test_state_rejected_not_priced():
    from engine.decide import decide, StateRejected

    with pytest.raises(StateRejected):
        decide(_state(original_price=-5.0, cost=10.0), None, None, CFG,
               np.random.default_rng(0), 100.0, "v")


@pytest.mark.parametrize("bad", [
    {"original_price": 0.0},                 # divided by, before the fix
    {"original_price": float("nan")},        # floored to an int, before the fix
    {"original_price": float("inf")},
    {"cost": float("nan")},
    {"cost": float("inf")},
    {"original_price": np.float64(0.0)},
])
def test_every_bad_price_or_cost_is_a_state_rejection_never_a_crash(bad):
    """The contract is REJECT: the caller catches StateRejected and moves on.
    A zero or NaN price used to reach the tier grid first and escape as
    ZeroDivisionError / ValueError, taking the pricing loop down."""
    from engine.decide import decide, StateRejected

    with pytest.raises(StateRejected) as exc:
        decide(_state(**bad), None, None, CFG, np.random.default_rng(0),
               100.0, "v")
    field = next(iter(bad))
    assert field in str(exc.value)


@pytest.mark.parametrize("bad_r", [None, float("inf"), float("nan"), 0.0, -1.0,
                                   np.float64("nan"), True])
def test_a_bad_dispersion_is_a_state_rejection_never_a_nan_q(bad_r):
    """`r` parameterises every pmf the DP sums: None was a TypeError in the
    solver, inf a NaN Q priced as if it were a number, bool a dispersion
    of 1. Judged like every other numeric, by name."""
    from engine.decide import decide, StateRejected

    with pytest.raises(StateRejected) as exc:
        decide(_state(r=bad_r), None, None, CFG, np.random.default_rng(0),
               100.0, "v")
    assert "r must be" in str(exc.value)


def test_the_decision_event_names_the_config_it_was_priced_with(tmp_path):
    """config_version is a label nobody bumps; the digest maps the hour to
    exactly one audit snapshot (design 5.14a)."""
    from common import provenance
    from engine.decide import decide
    from engine.posterior import PosteriorStore
    from events.store import EventStore

    store = EventStore(CFG, root=str(tmp_path / "events"))
    posterior = PosteriorStore.initialise(
        CFG, {"MEAT": {"mean": -1.0, "std": 0.6}}, {"MEAT": 10**6},
        path=str(tmp_path / "posterior.json"))
    evt = decide(_state(), posterior, store, CFG, np.random.default_rng(0), 100.0, "v")
    assert evt["config_digest"] == provenance.config_fingerprint(CFG)["digest"]
    assert store.load_decisions()[0]["config_digest"] == evt["config_digest"]


def test_the_finiteness_test_has_one_home():
    """events.store quarantines a non-finite applied_price and engine.decide
    rejects a non-finite state by the SAME predicate -- a second copy is
    how the two drift (inf passed one and not the other, once)."""
    from engine import decide
    from events import store

    assert store.finite_number is decide.finite_number
    assert decide.finite_number(np.float64(1.0)) and decide.finite_number(3)
    for v in (True, np.bool_(True), "1", None, float("inf"), float("nan")):
        assert not decide.finite_number(v), v


def test_the_economics_are_judged_once_before_the_grid():
    """economics_failures is the precondition the tier grid is built on;
    validate_state takes everything else, so a bad price is named once."""
    from engine.decide import decide, economics_failures, StateRejected, validate_state

    assert economics_failures(10000.0, 4000.0) == []
    assert economics_failures(10000.0, 12000.0) == ["cost must not exceed original_price"]
    assert any("original_price" in f for f in economics_failures(0.0, 4000.0))
    assert any("cost" in f for f in economics_failures(10000.0, -1.0))
    # the rest of the state is validated on its own, tiers given
    assert validate_state(_state(), [0.0, 0.025], None, [1.0, 1.0]) == []
    with pytest.raises(StateRejected) as exc:
        decide(_state(cost=12000.0), None, None, CFG, np.random.default_rng(0),
               100.0, "v")
    assert str(exc.value).count("original_price") == 1


# ------------------------------------------------------------------ the DP
def _reference_solve(original_price, cost, q0, mu_ref_path, d_ref, epsilon, r,
                     cfg, anchor_discount=None, entry=False):
    """engine.dp.solve as it stood before the vectorised gather -- one scipy
    pmf per (stage, tier) and the scalar (t, j, q) loop -- kept here as the
    reference the production solver must match bit for bit."""
    pcfg = cfg["pricing"]
    tiers, d_max = dp_mod.feasible_tiers(original_price, cost, pcfg["tier_step"])
    horizon, n_tiers, max_k = len(mu_ref_path), len(tiers), pcfg["negbin_max_k"]
    k = np.arange(max_k + 1)
    pmf = np.empty((horizon, n_tiers, max_k + 1))
    tail_max = 0.0
    for t in range(horizon):
        for j, d in enumerate(tiers):
            mu = mu_at(mu_ref_path[t], d, d_ref, epsilon, pcfg["demand_floor"])
            row = nbinom.pmf(k, r, r / (r + mu))
            tail = float(max(0.0, 1.0 - row.sum()))
            row[-1] += tail
            pmf[t, j] = row
            tail_max = max(tail_max, tail)
    reward = np.array([-(original_price - original_price * (1 - d)) for d in tiers])
    V = np.zeros((horizon + 1, n_tiers, q0 + 1))
    V[horizon, :, :] = -cost * np.arange(q0 + 1)[None, :]
    for t in range(horizon - 1, -1, -1):
        Q = np.full((n_tiers, q0 + 1), -np.inf)
        for j in range(n_tiers):
            for q in range(q0 + 1):
                sold = np.minimum(k, q)
                Q[j, q] = np.sum(pmf[t, j] * (sold * reward[j] + V[t + 1, j, q - sold]))
        V[t] = np.maximum.accumulate(Q[::-1], axis=0)[::-1]
        if t == 0:
            Q_now = Q
    if entry:
        allowed = dp_mod.entry_action_set(tiers, d_ref, d_max, pcfg)
    else:
        allowed = [j for j, d in enumerate(tiers) if d >= anchor_discount - dp_mod.TIER_EPS]
    return {j: float(Q_now[j, q0]) for j in allowed}, tail_max


def test_the_vectorised_dp_is_bit_identical_to_the_scalar_loop():
    """The gather over (tier, inventory) is one numpy expression now; the
    values it produces must be the ones the (t, j, q) loop produced --
    exactly, not approximately, because assurance re-solves logged decisions
    and the cost floor / monotone constraint are read off these numbers."""
    rng = np.random.default_rng(11)
    step = CFG["pricing"]["tier_step"]
    checked = 0
    for _ in range(24):
        p0 = float(rng.uniform(1000, 30000))
        cost = float(rng.uniform(0.0, 0.9)) * p0
        q0 = int(rng.integers(1, 30))
        path = list(rng.uniform(0.05, 5.0, int(rng.integers(1, 13))))
        eps = float(rng.uniform(CFG["posterior"]["epsilon_min"],
                                CFG["posterior"]["epsilon_max"]))
        r = float(rng.uniform(0.2, 3.0))
        d_ref = float(rng.choice([0.1, 0.2, 0.3, 0.4]))
        tiers, _ = dp_mod.feasible_tiers(p0, cost, step)
        entry = bool(rng.integers(0, 2))
        anchor = None if entry else float(rng.choice(tiers))
        res = dp_mod.solve(p0, cost, q0, path, d_ref, eps, r, CFG,
                           anchor_discount=anchor, entry=entry)
        ref_q, ref_tail = _reference_solve(p0, cost, q0, path, d_ref, eps, r, CFG,
                                           anchor_discount=anchor, entry=entry)
        assert res.q_by_tier == ref_q                # dict equality: exact floats
        assert res.tail_mass_max == ref_tail
        checked += len(ref_q)
    assert checked > 100


def test_the_one_mu_pmf_is_the_table_it_delegates_to():
    """nb_pmf_vector (the censored expectation's pmf) and the DP's table are
    one NB parameterisation: same values, tail folded the same way."""
    from engine.demand import nb_pmf_table, nb_pmf_vector

    rng = np.random.default_rng(3)
    max_k = CFG["pricing"]["negbin_max_k"]
    k = np.arange(max_k + 1)
    mus, r = rng.uniform(0.01, 40.0, 50), 0.7
    table, tails = nb_pmf_table(mus, r, max_k)
    for i, mu in enumerate(mus):
        row = nbinom.pmf(k, r, r / (r + mu))              # the pre-table arithmetic
        tail = float(max(0.0, 1.0 - row.sum()))
        row[-1] += tail
        pmf_i, tail_i = nb_pmf_vector(mu, r, max_k)
        assert np.array_equal(pmf_i, row) and tail_i == tail
        assert np.array_equal(table[i], row) and tails[i] == tail


def test_a_finite_positive_state_still_prices(tmp_path):
    """The guard rejects only what the grid cannot take."""
    from events.store import EventStore
    from engine.decide import decide
    from engine.posterior import PosteriorStore

    posterior = PosteriorStore.initialise(
        CFG, {"MEAT": {"mean": -1.0, "std": 0.6}}, {"MEAT": 10**6},
        path=str(tmp_path / "posterior.json"))
    store = EventStore(CFG, root=str(tmp_path / "events"))
    evt = decide(_state(original_price=np.float64(10000.0), cost=np.float64(4000.0)),
                 posterior, store, CFG, np.random.default_rng(0), 100.0, "v")
    assert evt["applied_price"] >= evt["cost"]
