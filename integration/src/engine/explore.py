"""engine.explore -- IL-budgeted exploration: the draw (design section 5.8).

The DP already computes Q(p) for every feasible price; exploration is a
constrained selection over those same values:

    p_star     = argmax_p Q(p)
    cost(p)    = Q(p_star) - Q(p)          expected IL loss, in currency
    admissible = { p != p_star : |log((1-d_p)/(1-d_ref))| >= delta_min }
    affordable = { p in admissible : cost(p) <= tau }

If affordable is non-empty, select UNIFORMLY AT RANDOM from it. Uniform
selection is not a detail -- it is the randomisation that makes the outcome
clean evidence; any state-dependent choice of forced price reintroduces the
endogeneity that makes legacy history unusable.

tau is a CURRENCY amount, compared against Q(p_star) - Q(p) in won, and the
one controller: the forced RATE is whatever the budget affords. There is no
exploration probability schedule or base rate; delta_min is a floor on the
MOVE from the reference (derived per cell in `delta_min`, never a second
knob), and the budget's std scaling has its own floor (`budget_scale`).

This module is the chooser. The budget and the controller that walks tau
are engine.budget; the Q-spread ledger the harnesses price tau against is
engine.spread_ledger; the tau paste's provenance gate is ops.config_keys.
Every name keeps resolving here for its callers.
"""

import math

from common.config import ConfigError

# float noise on a LOG price move (|log((1-d)/(1-d_ref))|). Same magnitude as
# engine.dp.TIER_EPS, but a log move is not a tier: the two comparisons are
# kept apart so a change to the grid epsilon cannot silently move the
# admissibility floor.
LOG_EPS = 1e-9


def delta_min(cfg, eps, category=None):
    """The smallest INFORMATIVE log price move FROM THE REFERENCE, per cell
    (design 5.8).

    The learner reads every forced outcome against mu_ref at the reference
    discount: L = log((1-d)/(1-d_ref)) and the signal is eps*L, against a
    level bias in mu_ref of scale `delta_min_log_bias` (the largest of the
    three fidelity readings tune derives it from). Below k*bias/|eps| the
    signal sits inside the model's own level error and the outcome teaches
    nothing about eps. Cost is measured from p*; information from d_ref --
    a tier far from p* but at d_ref costs money and teaches nothing, which
    is why the floor is on the reference distance. 0 while the bias scale
    is null (nothing pasted yet).

    The floor grows as |eps| shrinks: a belief stepping toward zero can
    inflate it past every feasible tier, and then nothing is forced, no
    outcome arrives to move the belief back, and tau climbs by the clip
    every zero-spend day. The only floor is |epsilon_max|; the monitor's
    affordable_set_empty_rate and the simulator's exploration_never_starves
    are where it shows (design 11.3).
    """
    ec = cfg["exploration"]
    bias = ec.get("delta_min_log_bias")
    if isinstance(bias, dict):
        # per category, `_default` for one the backtest never saw -- the
        # same key convention as reference_discount ('SIDE DISH' -> SIDE_DISH)
        key = str(category).replace(" ", "_") if category is not None else "_default"
        if key not in bias and "_default" not in bias:
            # a config defect, named as one: engine.decide turns it into a
            # per-row StateRejected so one unmapped category never takes a
            # whole batch down (design 5.10)
            raise ConfigError(
                f"exploration.delta_min_log_bias has no entry for {key!r} and "
                "no `_default`: a per-category floor mapping must name every "
                "priced category or carry `_default` (ops.tune writes "
                "both). A missing key is not 'no floor'.")
        bias = bias.get(key, bias.get("_default"))
    if not bias:
        return 0.0
    floor = abs(float(cfg["posterior"]["epsilon_max"]))       # the sign constraint
    return (float(ec["delta_min_bias_multiple"]) * float(bias)
            / max(abs(float(eps)), floor))


def log_move(d_from, d_to):
    """|log((1-d_to)/(1-d_from))|: the price move in log space."""
    return abs(math.log((1.0 - d_to) / (1.0 - d_from)))


def admissible(dp_result, delta_min=0.0):
    """Non-optimal tiers at least `delta_min` from the REFERENCE discount in
    log price -- the tiers a perturbation may land on before the budget is
    asked. ONE definition: the chooser, the spread ledger and the assurance
    check all read it, so tau is calibrated against exactly the set it is
    spent on."""
    star = dp_result.optimal_index
    d_ref = dp_result.d_ref
    return [j for j in dp_result.q_by_tier if j != star
            and (delta_min <= 0
                 or log_move(d_ref, dp_result.tiers[j]) >= delta_min - LOG_EPS)]


def admissible_costs(dp_result, delta_min=0.0):
    """{tier index: Q(p_star) - Q(p)} over the ADMISSIBLE tiers, in tier
    order -- the expected IL each forced action forgoes, in currency. The
    one table the chooser, the ledger and the assurance check all read;
    `costs=` on the readers below takes it precomputed so a decision prices
    its tiers once."""
    q = dp_result.q_by_tier
    q_star = q[dp_result.optimal_index]
    return {j: q_star - q[j] for j in admissible(dp_result, delta_min)}


def affordable_set(dp_result, tau, delta_min=0.0, costs=None):
    """Tier indices a perturbation may legally land on, and the admissible
    tiers' costs. Public: `daily.assurance` reconstructs this exact set to
    test that the draw was uniform."""
    if costs is None:
        costs = admissible_costs(dp_result, delta_min)
    return [j for j, c in costs.items() if c <= tau], costs


def spread_table(dp_result, delta_min=0.0, costs=None):
    """(costs, log moves from the reference) over the admissible tiers -- the
    Q-spreads the ledger prices tau against, so a tau solved here funds the
    draws production will make; the moves let the ledger re-judge
    admissibility at a deeper floor (SpreadLedger.sweep)."""
    if costs is None:
        costs = admissible_costs(dp_result, delta_min)
    return (list(costs.values()),
            [log_move(dp_result.d_ref, dp_result.tiers[j]) for j in costs])


def spread_costs(dp_result, delta_min=0.0):
    return list(admissible_costs(dp_result, delta_min).values())


def select(dp_result, tau, rng, explorable=True, delta_min=0.0, costs=None):
    """Returns a dict describing the chosen action.

    explorable=False marks a structurally non-explorable episode (fewer than
    min_feasible_tiers): it is priced by the DP as normal but excluded from
    the exploration budget and never logged as a blocked attempt.
    """
    star = dp_result.optimal_index
    choice = {
        "optimal_index": star,
        "chosen_index": star,
        "is_exploration": False,
        "exploration_cost": 0.0,
        "affordable_set_size": 0,
    }
    if not explorable or tau is None:
        return choice

    affordable, costs = affordable_set(dp_result, tau, delta_min, costs)
    choice["affordable_set_size"] = len(affordable)
    if affordable:
        j = affordable[int(rng.integers(0, len(affordable)))]
        choice.update(chosen_index=j, is_exploration=True,
                      exploration_cost=float(costs[j]))
    return choice


# moved to engine.spread_ledger and engine.budget; the names stay here for
# callers (`from engine import explore; explore.walk_tau`). The tau paste
# gate is ops.config_keys.tau_provenance_error: a driver's check, not the
# engine's
from engine.spread_ledger import SpreadLedger                                  # noqa: E402,F401
from engine.budget import (SUSPENDED, budget_base_ready, budget_held,          # noqa: E402,F401
                           budget_scale, budget_today, tau_next,
                           trailing_daily_il, walk_tau, _base_span)
