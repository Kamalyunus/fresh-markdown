"""The integration contract handed to engineering must match the code.

`docs/event_contract.html` is what an integrating team builds against: it lists
every field they send and every field they return. Nothing else in the repo
would notice if a field were added to `DECISION_REQUIRED` and never written
down -- the doc would simply be quietly wrong, and the first symptom would be a
partner integration failing validation on a field nobody told them about.

So the doc is checked against the schema in both directions: every required
field appears in it, and every field name it prints is one the system actually
knows. It is hand-authored, unlike the walkthrough, which is exactly why it
needs a guard the walkthrough does not.
"""

import os
import re

from events.store import DECISION_REQUIRED, OUTCOME_REQUIRED

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "event_contract.html")

# Names the doc prints that are deliberately not event fields: the two
# conditional outcome fields, and the request-state names, which differ from
# their logged counterparts (`q` is logged as q_remaining, `current_discount`
# as anchor_discount). Anything else appearing here is a typo or a rename the
# doc missed.
NOT_EVENT_FIELDS = {
    "adjustment_reason",           # conditional, only when inventory breaks
    "execution_failure_reason",    # conditional, only on a failed apply
    "q",                           # request state -> q_remaining
    "current_discount",            # request state -> anchor_discount
}


def documented_fields():
    with open(DOC) as f:
        html = f.read()
    names = set()
    # field names live in `<td class="f">`; a few rows carry several, separated
    # by a middot, where the fields share one description
    for cell in re.findall(r'<td class="f">(.*?)</td>', html, re.S):
        for part in re.sub(r"<[^>]+>", "", cell).split("·"):
            part = part.strip()
            if re.fullmatch(r"[a-z_0-9]+", part):
                names.add(part)
    return names


def test_every_required_decision_field_is_documented():
    missing = [f for f in DECISION_REQUIRED if f not in documented_fields()]
    assert not missing, (
        f"docs/event_contract.html does not mention {missing}. A required "
        "field absent from the contract is one an integration will not send.")


def test_every_required_outcome_field_is_documented():
    missing = [f for f in OUTCOME_REQUIRED if f not in documented_fields()]
    assert not missing, (
        f"docs/event_contract.html does not mention {missing}. The outcome "
        "event is the part engineering actually produces -- it must be whole.")


def test_the_doc_invents_no_fields():
    """A field the doc names but the system does not know is worse than a
    missing one: it gets built, sent, and silently ignored."""
    known = set(DECISION_REQUIRED) | set(OUTCOME_REQUIRED) | NOT_EVENT_FIELDS
    invented = sorted(documented_fields() - known)
    assert not invented, (
        f"docs/event_contract.html names {invented}, which no event carries. "
        "Either the schema was renamed and the doc was not, or it is a typo.")
