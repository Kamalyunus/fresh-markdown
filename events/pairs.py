"""The ONE pairing of outcomes to decisions, and the ONE day key.

Four modules each rebuilt `{decision_id: d}` and walked the outcomes with
their own eligibility rule; the learning path excluded failed pushes and
the assurance path did not, so assurance graded r against prices that were
never charged. Every per-day series shares `decision_day` -- the trading
date the decision priced, never the UTC wall clock of its outcome.
"""

import pandas as pd

# applied vs recommended price: float noise on a currency amount, not a knob
PRICE_MATCH_TOLERANCE = 1e-6

# what a pushed outcome must report for its (decision, outcome) pair to
# teach anything: the price we chose was the price on the shelf
LEARNABLE_STATUSES = (None, "ok", "success")


def decision_day(d):
    """The TRADING date a decision priced."""
    return str(d.get("date") or pd.Timestamp(d["timestamp"]).date())


def is_learnable(o):
    return o.get("execution_status") in LEARNABLE_STATUSES


def price_matches(d, o):
    return abs(o["applied_price"] - d["applied_price"]) <= PRICE_MATCH_TOLERANCE


def match_pairs(decisions, outcomes, learnable=False):
    """[(decision, outcome)] for outcomes that name a known decision, in
    outcome order. `learnable=True` keeps only outcomes whose push
    succeeded -- the learning and assurance population; business and
    guardrail series read every matched pair (a failed push still sold
    units and still scrapped)."""
    dec = {d["decision_id"]: d for d in decisions}
    out = []
    for o in outcomes:
        d = dec.get(o.get("decision_id"))
        if d is None or (learnable and not is_learnable(o)):
            continue
        out.append((d, o))
    return out
