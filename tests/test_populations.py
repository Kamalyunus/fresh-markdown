"""Two populations, and which consumer is entitled to which.

The distinction this file defends: an INTEGRITY defect is a row that cannot
be believed; an ECONOMIC condition is a row that is perfectly believable and
merely unpriceable. Conflating them cost most of the COGS in the extract --
every frozen artifact was fit on the priceable subset, including the
elasticity prior, for which below-cost hours are the widest price variation
the data contains.

And it made gate 1 undecidable. `share_non_explorable` counts episodes whose
cost floor leaves too few tiers to explore against, and the chain deleted
exactly those before `m1` looked, so the gate read 0.0 and could not fail.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from bootstrap.prepare_data import (DP_INELIGIBLE, population,
                                    tag_dp_eligibility)
from common.config import load_config


@pytest.fixture(scope="module")
def cfg():
    import os
    return load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))


def _frame(**over):
    """One clean two-hour episode; keyword overrides break it one way."""
    base = dict(episode_id="e", cost=4000.0, original_price=10_000.0,
                total_discount=0.25, starting_inventory=12,
                hours_remaining=[10.0, 9.0])
    base.update(over)
    hr = base.pop("hours_remaining")
    d = pd.DataFrame({k: [v, v] for k, v in base.items()})
    d["hours_remaining"] = hr
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    d["d_max"] = 1.0 - d.cost / d.original_price
    return d


# ------------------------------------------------------ the four conditions

@pytest.mark.parametrize("name,over", [
    ("cost_missing", {"cost": 0.0}),
    ("non_priceable", {"cost": 12_000.0}),
    ("below_cost", {"total_discount": 0.9}),
    ("window_too_long", {"hours_remaining": [500.0, 499.0]}),
])
def test_each_condition_flags_and_names_itself(name, over, cfg):
    d, detail = tag_dp_eligibility(_frame(**over), cfg)
    assert not d.dp_eligible.any(), f"{name} was not caught"
    assert (d.dp_ineligible_reason == name).all()
    assert detail[name]["episodes"] == 1
    assert detail["episodes_dp_ineligible"] == 1


def test_a_clean_episode_is_eligible(cfg):
    d, detail = tag_dp_eligibility(_frame(), cfg)
    assert d.dp_eligible.all()
    assert d.dp_ineligible_reason.isna().all()
    assert detail["episodes_dp_eligible"] == 1


def test_the_flag_is_episode_scoped_not_row_scoped(cfg):
    """One below-cost hour poisons the window: the monotonicity anchor
    carries that price into every later hour."""
    d = _frame()
    d.loc[1, "total_discount"] = 0.95
    d["offered_price"] = d.original_price * (1 - d.total_discount)
    tagged, _ = tag_dp_eligibility(d, cfg)
    assert not tagged.dp_eligible.any(), "only the offending row was flagged"


def test_nothing_is_dropped(cfg):
    for over in ({"cost": 0.0}, {"cost": 12_000.0}, {"total_discount": 0.9}):
        before = _frame(**over)
        after, _ = tag_dp_eligibility(before, cfg)
        assert len(after) == len(before)


def test_reasons_are_first_match_so_the_column_reads_as_a_cause(cfg):
    """A zero cost is ALSO below cost and ALSO non-priceable. The label has
    to be the fundamental one, not whichever test ran last."""
    d, _ = tag_dp_eligibility(_frame(cost=0.0, total_discount=0.9), cfg)
    assert (d.dp_ineligible_reason == "cost_missing").all()
    assert [n for n, _ in DP_INELIGIBLE][0] == "cost_missing"


def test_every_condition_carries_a_stated_reason():
    for name, why in DP_INELIGIBLE:
        assert len(why) > 40, f"{name} has no explanation"


# ------------------------------------------------------- who reads what

def test_population_resolves_the_config_default(cfg):
    d, _ = tag_dp_eligibility(pd.concat([_frame(), _frame(cost=0.0)
                                         .assign(episode_id="bad")]), cfg)
    assert cfg["baseline_model"]["train_population"] == "integrity"
    assert len(population(d, cfg)) == len(d)                      # the default
    assert len(population(d, cfg, "integrity")) == len(d)
    assert len(population(d, cfg, "dp_eligible")) < len(d)
    with pytest.raises(ValueError):
        population(d, cfg, "whatever")


def test_the_artifact_fits_read_the_config_and_the_dp_side_does_not():
    """The choice exists for the frozen artifacts. For the DP it is a
    precondition -- an ineligible episode has no feasible tier at all."""
    from bootstrap import train_baseline, fit_dispersion, estimate_prior
    from backtest import __main__ as bt
    from pipeline import shadow

    for fn in (train_baseline.train, fit_dispersion.fit_dispersion,
               estimate_prior.estimate_prior):
        assert "population(" in inspect.getsource(fn), fn.__name__

    for fn in (bt.main, shadow.run_shadow):
        src = inspect.getsource(fn)
        assert 'population(d, cfg, "dp_eligible")' in src, fn.__qualname__


def test_gate_one_reads_the_integrity_population():
    """The whole reason for the change: measured on dp_eligible this gate is
    0.0 by construction and can never fire."""
    from bootstrap import measure
    doc = inspect.getdoc(measure.m1_cost_ratio)
    assert "INTEGRITY population" in doc
    src = inspect.getsource(measure.gates)
    assert "measured_on" in src


def test_il_is_reported_on_both_bases():
    """One is what the business loses; the other is what the MVP can address.
    Quoting either alone was how a sub-population figure became the headline."""
    from bootstrap import measure
    src = inspect.getsource(measure.m6_il_pct)
    assert '"integrity"' in src and '"dp_eligible"' in src
    assert "population_note" in src


def test_the_cost_floor_is_not_a_population_choice():
    """`dp_eligible` selects rows. It must never be the thing that keeps a
    below-cost price out of the action set -- that is structural."""
    from pricing import dp
    tiers, d_max = dp.feasible_tiers(1000.0, 400.0, 0.025)
    assert all(1000.0 * (1 - t) >= 400.0 - 1e-9 for t in tiers)
    assert "dp_eligible" not in inspect.getsource(dp.feasible_tiers)
