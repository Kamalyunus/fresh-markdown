# Maintaining the documents

The doc surface is deliberately small: `docs/design.md` (the authoritative
spec), `docs/learnings.md` (superseded designs), `docs/event_contract.html`
(the integration contract), `AGENTS.md`, `README.md`, `RUNBOOK.md`,
`REVIEW_GUIDE.md`. The chart/walkthrough/EDA/metrics-index tooling that once
kept generated pages fresh was removed with the descriptive layer; if a page
like that is wanted again, build it from the reports, never by hand-typing
numbers.

## The two guarded docs

**`docs/design.md`** is cross-checked by `tests/test_docs_match_the_code.py`:
every waterfall stage, every `dp_eligible` gate, the reported-only flags, the
three populations and the episode identity must appear in it, and no retired
rule may read as live. Editing the filter chain means editing the doc in the
same change.

**`docs/event_contract.html`** is what an integrating team builds against:
the request fields, the ~36 logged decision fields, the outcome fields, the
three `adjustment_reason` values, quarantine rules and event-quality
thresholds. Hand-authored, so `tests/test_event_contract_doc.py` checks it
against `events/store.py` in both directions — adding a required event field
fails the suite until the contract is updated. Its thresholds are quoted
from `config.yaml` and are NOT guarded: re-read them when
`monitoring.stop_conditions` or `monitoring.shadow_gate` moves. The worked
episode in §05 is real solver output — regenerate it rather than
hand-patching numbers.

## Standing rule for code changes

Every code change updates its docs in the same commit: AGENTS.md's
one-home list and paste table, the RUNBOOK step it touches, the design.md
section, `docs/event_contract.html` for any event field, and a
learnings.md entry when a design was superseded. The two guarded docs
fail the suite when they lag; the rest lag silently, which is worse.

## Standing rules for numbers in documents

- **AGENTS.md is a router, not a reference** — a 400-line budget, enforced
  by test. New material goes to design.md (spec) or learnings.md (history),
  with at most a one-liner and a pointer.
- **Quote only from a gate-passing run**, and never invent a figure: if a
  report is missing, say which number could not be refreshed. Repo-local
  runs are on SYNTHETIC data (AGENTS rule 19) — never present a fixture
  number as a production finding.
