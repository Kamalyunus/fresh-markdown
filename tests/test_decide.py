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
            "hours_remaining": 4, "hour_of_day": 12, "r": 1.0}
    tiers = [0.0, 0.025, 0.05]

    assert validate_state(base, tiers, None, [1.0, 1.0, 1.0, 1.0], CFG) == []

    failures = validate_state(base, tiers, None, [1.0, 1.0], CFG)
    assert any("planning horizon" in f for f in failures)


def test_a_horizon_past_max_window_hours_is_rejected_not_priced():
    """The DP plans `hours_remaining` stages: a counter defect (a feed
    emitting minutes) allocated a value function over thousands of hours
    and priced a twenty-year episode. The live path refuses above the same
    `data.max_window_hours` prepare_data flags offline, and the request
    and the state read ONE set of count checks (count_failures)."""
    from engine.decide import StateRejected, count_failures, decide, validate_state

    cap = CFG["data"]["max_window_hours"]
    ok = {"q": 3, "hours_remaining": cap, "hour_of_day": 12}
    assert count_failures(ok, CFG) == []
    too_long = count_failures({**ok, "hours_remaining": cap + 1}, CFG)
    assert too_long and "max_window_hours" in too_long[0]
    # the bound reads the config, not a literal
    tighter = dict(CFG, data=dict(CFG["data"], max_window_hours=cap - 1))
    assert count_failures(ok, tighter)
    # an integral float from a table IS a count; 12.5 names no hour
    assert count_failures({"q": 3.0, "hours_remaining": 4.0, "hour_of_day": 12.0}, CFG) == []
    assert count_failures({"q": 3, "hours_remaining": 4, "hour_of_day": 12.5}, CFG)
    assert count_failures({}, CFG) and len(count_failures({}, CFG)) == 3
    # and the whole state path rejects, by name, never allocates
    path = [1.0] * (cap + 1)
    failures = validate_state(_state(hours_remaining=cap + 1, mu_ref_path=path),
                              [0.0, 0.025], None, path, CFG)
    assert any("max_window_hours" in f for f in failures)
    with pytest.raises(StateRejected, match="max_window_hours"):
        decide(_state(hours_remaining=cap + 1, mu_ref_path=path), None, None,
               CFG, np.random.default_rng(0), 100.0, "v")


def test_a_floor_mapping_that_names_neither_the_category_nor_a_default_rejects_the_row(tmp_path):
    """A broken per-category delta_min paste refused the WHOLE batch: the
    bare KeyError escaped every caller that catches StateRejected. It is a
    row rejection naming the config key -- loud, per row, never the batch."""
    from common.config import ConfigError
    from engine import explore
    from engine.decide import StateRejected, decide
    from events.store import EventStore

    broken = dict(CFG, exploration=dict(CFG["exploration"],
                                        delta_min_log_bias={"FRUIT": 0.12}))
    with pytest.raises(ConfigError, match="_default"):
        explore.delta_min(broken, -1.0, "MEAT")
    with pytest.raises(StateRejected, match="delta_min_log_bias"):
        decide(_state(), _posterior_for("MEAT", tmp_path / "a"), None, broken,
               np.random.default_rng(0), 100.0, "v")
    # a mapped category on the same config still prices
    store = EventStore(CFG, root=str(tmp_path / "events"))
    assert decide(_state(category="FRUIT"), _posterior_for("FRUIT", tmp_path / "b"),
                  store, broken, np.random.default_rng(0), 100.0, "v")["applied_price"] > 0


def _posterior_for(category, root):
    from engine.posterior import PosteriorStore
    return PosteriorStore.initialise(
        CFG, {category: {"mean": -1.0, "std": 0.6}}, {category: 10**6},
        path=str(root / "posterior.json"))


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


def test_the_decision_id_a_priced_hour_carries_is_the_shelf_hour(tmp_path):
    """Not a surrogate: one decision per shelf-hour is what the id SAYS, so a
    re-priced hour collides with its own earlier copy instead of putting a
    second price on one feed row, and engineering can name the decision for
    an hour before it exists. The EPISODE is deliberately not in it -- that
    id is the producers' and can be relabelled, and an audit record's
    identity may not move when an upstream label does."""
    from engine.decide import decide
    from engine.posterior import PosteriorStore
    from events.pairs import decision_id_of, hour_key, outcome_id_of
    from events.store import EventStore

    posterior = PosteriorStore.initialise(
        CFG, {"MEAT": {"mean": -1.0, "std": 0.6}}, {"MEAT": 10**6},
        path=str(tmp_path / "posterior.json"))

    def priced(root, **over):
        store = EventStore(CFG, root=str(tmp_path / root))
        return decide(_state(**over), posterior, store, CFG,
                      np.random.default_rng(0), 100.0, "v")

    key = hour_key(1, "F", "2026-08-01", 12)
    evt = priced("a")
    assert evt["decision_id"] == decision_id_of(key) == "dec-1|F|2026-08-01T12"
    assert evt["decision_id"][4:] == outcome_id_of(key)[5:]   # the pair, one key
    # relabel the episode: the same shelf-hour keeps the same decision id
    relabelled = priced("b", episode_id="RELISTED-9")
    assert relabelled["decision_id"] == evt["decision_id"]
    # a different hour of the same shelf is a different decision
    assert priced("c", hour_of_day=13)["decision_id"] == "dec-1|F|2026-08-01T13"


def test_the_finiteness_test_has_one_home():
    """events.store quarantines a non-finite applied_price and engine.decide
    rejects a non-finite state by the SAME predicate -- a second copy is
    how the two drift (inf passed one and not the other, once)."""
    from engine import decide
    from events import contract, store

    assert contract.finite_number is decide.finite_number
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
    assert validate_state(_state(), [0.0, 0.025], None, [1.0, 1.0], CFG) == []
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
    horizon, n_tiers = len(mu_ref_path), len(tiers)
    # the table runs to the shelf when the shelf is deeper than negbin_max_k
    # (dp.table_width): the same rule the vectorised solver reads
    max_k = dp_mod.table_width(pcfg["negbin_max_k"], q0)
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


def test_the_dp_sells_a_shelf_deeper_than_negbin_max_k_exactly():
    """With q0 above negbin_max_k the DP could sell at most max_k units a
    stage (sold = min(k, q) over a pmf folded at max_k), so Q undervalued
    clearing a large shelf. The table now runs to the shelf; the value is
    the one a table wide enough to hold every unit of demand gives."""
    max_k = CFG["pricing"]["negbin_max_k"]
    q0, mu = max_k * 2 + 10, float(max_k) * 1.6
    got = dp_mod.solve(10000.0, 4000.0, q0, [mu], 0.30, -1.0, 1.5, CFG,
                       anchor_discount=0.0, entry=False)
    wide = dict(CFG, pricing=dict(CFG["pricing"], negbin_max_k=q0 * 8))
    want = dp_mod.solve(10000.0, 4000.0, q0, [mu], 0.30, -1.0, 1.5, wide,
                        anchor_discount=0.0, entry=False)
    for j, v in want.q_by_tier.items():
        assert got.q_by_tier[j] == pytest.approx(v, rel=1e-9), j
    # and the folded table at max_k alone reads pessimistic: the fix is real
    from engine.demand import expected_min_demand_inventory, nb_pmf_vector
    pmf, _ = nb_pmf_vector(mu, 1.5, max_k)
    folded = float(np.sum(pmf * np.minimum(np.arange(max_k + 1), q0)))
    assert expected_min_demand_inventory(mu, 1.5, q0, max_k) > folded * 1.2
    # the table is never narrower than negbin_max_k
    assert dp_mod.table_width(max_k, 3) == max_k and dp_mod.table_width(max_k, q0) == q0


def test_the_censored_expectation_is_exact_past_negbin_max_k():
    """E[min(D, q)] folded the NB tail at negbin_max_k and read
    E[min(D, min(q, max_k))]: a shelf deeper than the table was capped,
    the sold/predicted ratio pushed above 1 and the level factor absorbed
    a truncation artefact. Every q matches a brute-force sum now."""
    from engine.demand import (censored_mean_closed_form,
                               expected_min_demand_inventory_vec)

    max_k = CFG["pricing"]["negbin_max_k"]
    mu, r = np.array([40.0, 2.0, 60.0]), np.array([1.5, 0.7, 3.0])
    q = np.array([100.0, 100.0, 30.0])              # every shelf past the table
    k = np.arange(20000)
    brute = np.array([np.sum(nbinom.pmf(k, ri, ri / (ri + mi)) * np.minimum(k, qi))
                      for mi, ri, qi in zip(mu, r, q)])
    got = expected_min_demand_inventory_vec(mu, r, q, max_k)
    assert np.allclose(got, brute, rtol=1e-10)
    assert np.allclose(censored_mean_closed_form(mu, r, q), brute, rtol=1e-10)
    # the closed form agrees with the folded table wherever the table is exact
    small_q = np.array([3.0, 12.0, max_k])
    assert np.allclose(censored_mean_closed_form(mu, r, small_q),
                       expected_min_demand_inventory_vec(mu, r, small_q, max_k), rtol=1e-10)
    # mixed shelves in one call: each row on its own basis
    mixed_q = np.array([3.0, 100.0, 30.0])
    mixed = expected_min_demand_inventory_vec(mu, r, mixed_q, max_k)
    assert mixed[0] == expected_min_demand_inventory_vec(mu[:1], r[:1], mixed_q[:1], max_k)[0]
    assert mixed[1] == pytest.approx(brute[1], rel=1e-10)


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


@pytest.mark.parametrize("bad, field", [
    ({"q": True}, "q"),                              # bool is not a count
    ({"q": np.bool_(True)}, "q"),
    ({"hours_remaining": True}, "hours_remaining"),
    ({"hour_of_day": 24}, "hour_of_day"),
    ({"hour_of_day": True}, "hour_of_day"),
    ({"hour_of_day": 12.5}, "hour_of_day"),          # a float naming no hour
    ({"hour_of_day": None}, "hour_of_day"),
    ({"hours_remaining": 0}, "hours_remaining"),
    # the field engineering sends is `current_discount` (contract section
    # 03): the rejection names it, never a field that exists nowhere
    ({"current_discount": "0.3"}, "current_discount"),  # cast before it was judged
    ({"current_discount": float("nan")}, "current_discount"),
    ({"current_discount": True}, "current_discount"),
    ({"mu_ref_path": [1.0, None]}, "demand predictions"),   # TypeError before
    ({"mu_ref_path": None}, "demand predictions"),
])
def test_every_malformed_count_hour_anchor_or_path_is_a_state_rejection(bad, field):
    """The remaining gaps in the REJECT contract: a bool passed as a count,
    an hour that names no hour, an anchor cast to float before it was
    validated (a string crashed the loop), a None inside the demand path
    (math.isfinite raised)."""
    from engine.decide import decide, StateRejected

    with pytest.raises(StateRejected) as exc:
        decide(_state(**bad), None, None, CFG, np.random.default_rng(0),
               100.0, "v")
    assert field in str(exc.value)
    assert "p_current" not in str(exc.value)


def test_the_caller_may_pass_the_config_digest_once_per_batch(tmp_path):
    """config_fingerprint snapshots the whole config per call -- about half
    the cost of a decision. A batch caller computes it once and passes it;
    the event carries exactly what was passed, and the default is unchanged."""
    from common import provenance
    from engine.decide import decide
    from engine.posterior import PosteriorStore
    from events.store import EventStore

    store = EventStore(CFG, root=str(tmp_path / "events"))
    posterior = PosteriorStore.initialise(
        CFG, {"MEAT": {"mean": -1.0, "std": 0.6}}, {"MEAT": 10**6},
        path=str(tmp_path / "posterior.json"))
    digest = provenance.config_fingerprint(CFG)["digest"]
    passed = decide(_state(), posterior, store, CFG, np.random.default_rng(0),
                    100.0, "v", config_digest=digest)
    assert passed["config_digest"] == digest
    # recorded, never verified: the caller owns the digest it passes
    marked = decide(_state(), posterior, store, CFG, np.random.default_rng(0),
                    100.0, "v", config_digest="batch-7")
    assert marked["config_digest"] == "batch-7"
    # a non-entry decision records its anchor as the float it was judged as
    later = decide(_state(current_discount=np.float64(0.1)), posterior, store, CFG,
                   np.random.default_rng(0), 100.0, "v", config_digest=digest)
    assert later["anchor_discount"] == 0.1 and type(later["anchor_discount"]) is float
    assert passed["anchor_discount"] is None


def test_the_censored_expectation_reads_the_one_pmf_table_bit_for_bit():
    """expected_min_demand_inventory_vec built its own per-row NB pmf inline;
    nb_pmf_table now takes one r per row and the vec reads it. Held to the
    inline arithmetic exactly, on a random probe (the level factors and the
    calibration gate are solved on these numbers)."""
    from engine.demand import expected_min_demand_inventory_vec, nb_pmf_table

    rng = np.random.default_rng(21)
    max_k = CFG["pricing"]["negbin_max_k"]
    n = 5000
    mu = rng.uniform(0.01, 40.0, n)
    r = rng.uniform(0.1, 5.0, n)
    q = rng.integers(0, 12, n).astype(float)
    k = np.arange(max_k + 1)
    # the inline arithmetic the vec carried before
    p = (r / (r + mu))[:, None]
    pmf = nbinom.pmf(k[None, :], r[:, None], p)
    pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0.0, None)
    want = np.sum(pmf * np.minimum(k[None, :], q[:, None]), axis=1)
    got = expected_min_demand_inventory_vec(mu, r, q, max_k, chunk=1234)
    assert np.array_equal(got, want)
    table, _ = nb_pmf_table(mu, r, max_k)
    assert np.array_equal(table, pmf)
    # and the scalar-r table (the DP's) is the row-wise one at a constant r
    scalar, _ = nb_pmf_table(mu, 0.7, max_k)
    rowwise, _ = nb_pmf_table(mu, np.full(n, 0.7), max_k)
    assert np.array_equal(scalar, rowwise)
