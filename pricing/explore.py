"""pricing.explore -- IL-budgeted exploration (PRD section 12).

The DP already computes Q(p) for every feasible price; exploration is a
constrained selection over those same values:

    p_star     = argmax_p Q(p)
    cost(p)    = Q(p_star) - Q(p)          expected IL loss, in currency
    affordable = { p : cost(p) <= tau, p != p_star }

If affordable is non-empty, select UNIFORMLY AT RANDOM from it. Uniform
selection is not a detail -- it is the randomisation that makes the outcome
clean evidence; any state-dependent choice of forced price reintroduces the
endogeneity that makes legacy history unusable.

tau is a CURRENCY amount, compared against Q(p_star) - Q(p) in won. There is
no exploration probability schedule, base rate, floor, ceiling, or cold-start
std.
"""


def select(dp_result, tau, rng, explorable=True):
    """Returns a dict describing the chosen action.

    explorable=False marks a structurally non-explorable episode (fewer than
    min_feasible_tiers): it is priced by the DP as normal but excluded from
    the exploration budget and never logged as a blocked attempt.
    """
    q = dp_result.q_by_tier
    star = dp_result.optimal_index
    costs = {j: q[star] - q[j] for j in q}

    choice = {
        "optimal_index": star,
        "chosen_index": star,
        "is_exploration": False,
        "exploration_cost": 0.0,
        "affordable_set_size": 0,
    }
    if not explorable or tau is None:
        return choice

    affordable = [j for j in q if j != star and costs[j] <= tau]
    choice["affordable_set_size"] = len(affordable)
    if affordable:
        j = affordable[int(rng.integers(0, len(affordable)))]
        choice.update(chosen_index=j, is_exploration=True,
                      exploration_cost=float(costs[j]))
    return choice


def budget_today(projected_markdown_il, posterior_std, cfg):
    """Section 12.3: a share of markdown IL, scaled down as the posterior
    narrows, never below budget_scale_floor of the full budget."""
    ec = cfg["exploration"]
    scale = min(max(posterior_std / ec["budget_scale_ref_std"],
                    ec["budget_scale_floor"]), 1.0)
    return ec["budget_share_of_il"] * scale * projected_markdown_il


def tau_next(tau, budget, realised_cost, cfg):
    """Daily multiplicative calibration of tau from realised spend."""
    ec = cfg["exploration"]
    lo, hi = ec["tau_adjust_clip"]
    ratio = budget / max(realised_cost, ec["tau_spend_guard"])
    return tau * min(max(ratio, lo), hi)
