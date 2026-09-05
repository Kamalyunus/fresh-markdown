"""engine.decide: what a state must look like to be priced, and what is
rejected by name rather than crashing the pricing loop."""

import numpy as np
import pytest

from conftest import CFG


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
