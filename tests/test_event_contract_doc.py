"""The integration contract handed to engineering must match the code."""

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
