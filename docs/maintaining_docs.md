# Maintaining the documents — charts, walkthrough, EDA, metrics index, contract

Moved from `AGENTS.md` so the operating guide stays short. Everything here is
the machinery that keeps the published documents in step with the pipeline:
read it before editing any file under `docs/` or `tools/`. The authoritative
spec is `docs/design.md`; superseded approaches are in `docs/learnings.md`.

## Charts

```bash
python3 -m tools.make_charts        # -> reports/charts/*.png
```

The charts `docs/design.md` embeds, plus the un-embedded extras Appendix A
names (exploration threshold, learning-yield floor, shadow gate, profile
likelihoods) — every one generated
from a report artifact, never hand-drawn, so a chart that disagrees with the
pipeline cannot exist. **Re-run this after any bootstrap or the document shows
the previous run's pictures beside the current run's numbers.** Process
diagrams (architecture, episode construction, gate sequence) are Mermaid inside
the design doc and need no regeneration.

A missing report is skipped with a note rather than failing, and so is a report
that exists but predates a field a chart reads — the note names the field. One
stale report must not cost the other six pictures.

**The filenames have gaps (`02`–`06`, `08`, `09`) and must keep them.**
`design.md` embeds these by name, so renumbering blanks the images it shows.
Five further charts were generated for a while and nothing ever referenced
them; they are gone, and adding a chart means embedding it in the same change
or it will go the same way.

`reports/` is gitignored, so the PNGs are build output, not tracked files.

## Refreshing the numbers in the docs

`docs/system_walkthrough.html` is the deliverable — one tab per frozen
artifact, plus the hourly decision, the learning loop, the replay evidence and
the production assurance. It is built, not hand-edited:

```bash
python3 -m tools.walkthrough.build      # writes docs/system_walkthrough.html
```

**Tab prose lives in `tools/walkthrough/panels/<tab>.html` — one file per tab,
plain HTML.** Edit those, never the built output. `panels.py` is now only a
loader: it reads each file verbatim and expands the two fragments that carry
values which must not be typed twice —

```html
<x-filecard path=… holds=… state=… reader=… [moves="1"]></x-filecard>
<x-pmfbars></x-pmfbars>
```

— and `_source.html` is the original single-topic decision-core page, whose
sections the builder lifts verbatim for the Decision tab. The panels were
Python f-strings until every literal brace in them had to be doubled, which
broke the page twice; they are files now so that markup is just markup.

Figures on the artifact tabs are
quoted from `docs/design.md` (the `baseline-20260811043259` run) so the page
holds one vintage throughout; the decision tab is a self-contained solve whose
inputs are printed on it. It is published as a claude.ai artifact — deploy the
built file with the EXISTING artifact URL, so the same link updates rather than
a second page appearing.

**Every measured figure is registered in `tools/walkthrough/figures.py`**,
against the JSON path it was read from *and* the `baseline_model_version` of
the run it came from. That buys two different checks:

- `tests/test_walkthrough_figures.py` fails if a panel stops printing a
  registered literal, or if a same-version report disagrees with the page.
- `pipeline.status` carries a `walkthrough · <tab>` row. A report from a
  **different** model version is WARN, not FAIL — it cannot be compared at
  all (hard rule 1), so the honest verdict is "stale, unverifiable". A
  disagreement *within* one run is FAIL.

**After a re-run, refreshing the page is a two-part edit**: update the
numbers in the panel and bump `model_version` in `figures.py`, in the same
commit. That pairing is the whole mechanism — this is the failure the v3 deck
already has, where slides 2 and 42 still show 36.68% against a report that
says 38.68%.

### Replay, Shadow, A/B — three rungs, and they are not interchangeable

The Replay tab is the agent against **our model of the world**; the Shadow tab
is the same machine against **the world itself**. Shadow is the more realistic
of the two about the decision path and says strictly *less* about the policy:
no price was applied, so there is no counterfactual outcome and **no IL figure
exists in a shadow run at all**. Do not "replace replay with shadow" — that
deletes the only loss number in the document and puts nothing in its place.
Replay states the value question inside a believed world; shadow states that
the machine runs correctly against the real one and how far its advice
diverges; only the A/B answers whether the advice is better.

The shadow tab's figure slots are registered as `PENDING` and its
`model_version` is `None` until the hold-out run lands. Registering the slots
before the numbers exist is deliberate: it fixes how the result will be read
before anyone can see it, and the pre-registered τ decision rule is printed on
the tab.

### The population EDA

`docs/eda.html` describes the population every other number is measured on:
15 panels built by `python3 -m tools.eda` from the prepared parquet and
`config.yaml` alone — no artifacts, no model, no DP, so it runs in seconds and
is worth re-running on every new extract.

**It decides nothing.** No gate, no verdict, no MEASURED value.
`bootstrap.measure` owns those, and a second source for one of them is exactly
the drift `artifact_mirror_drift` exists to catch. A test asserts the report
contains no `verdict`, `pass`, `tau_initial` or `rho`.

What makes it more than a notebook: **every panel names the config keys it
should change your mind about**, and `tests/test_eda.py` parametrises over
every one of those keys and asserts it resolves — so a rename breaks the
claim instead of leaving it stale. `reports/eda.json` carries every number
including the chart series; `docs/eda.html` is a pure view over it and cannot
show a figure the report does not contain.

It is also a **walkthrough tab** ("Population", after Data). That tab is
authored prose with the figures read LIVE from `reports/eda.json` at build
time via two tags the panel loader expands:

```html
<x-eda-chips></x-eda-chips>
<x-eda-chart key="pareto"></x-eda-chart>
```

Charts come from `tools.eda_page.KINDS`, the same renderer `docs/eda.html`
uses — two pages, one definition, so the same series cannot be drawn two
different ways. Nothing on that tab is typed, so it cannot go stale the way
the Replay tab's figures can (which is why those needed
`tools/walkthrough/figures.py` and this does not).

`reports/` is gitignored, so a fresh clone has no report: the tags then render
a visible "not built yet" note naming the command. **Never make the
walkthrough build depend on a pipeline run** — an empty chart reads like a
finding of zero, and a build that fails without artifacts is a build nobody
can do.

The panels worth reading first on a fresh extract:

- **anchors** — anchor rows per subcategory in BOTH bands (`tier_step/2` for
  calibration, `ref_rate_anchor_band` for the velocity features). Calibration
  is fit entirely on the first and nothing else shows the count before the fit
  runs.
- **entry_arms** — how often each of the five entry offsets survives the cost
  floor. config asserts the deepest one vanishes above a ~0.65 cost ratio;
  this is the first thing that measures it.
- **cells** — one table saying whether the subcategory → category → global
  hierarchy has anything to work with, or falls through to global everywhere.
- **drift** — the weekly level series with the split boundaries marked. The
  panel that would have caught the calibration fortnight being the most
  anomalous stretch in five months.

### The metrics index

`docs/metrics.html` is the reference for "what is this number": **135 metrics
across 17 components**, each with its unit, the component that writes it, and
whether anything downstream is gated on it. Built, not hand-edited:

```bash
python3 -m tools.metrics_glossary       # writes docs/metrics.html
```

The catalogue lives in `tools/metrics_glossary.py` as data — short strings in
a table, not prose documents, which is why it stays in Python. The page
carries a live filter and a **gates-only** toggle, because the question is
almost always "which of these blocks something".

Read it as the tier-two companion to `pipeline.status`: status prints the ten
checks that gate a decision, this explains the ~700 fields behind them. It
does NOT list all 45 event fields — `docs/event_contract.html` does that
exhaustively under its own guard, and two exhaustive lists of one schema is
how they come to disagree. `tests/test_metrics_glossary.py` asserts that
non-duplication, and cross-checks the three things that drift silently: every
event field named must be real, every artifact path must be in
`provenance.ARTIFACTS`, and the Status board section must match
`pipeline.status`'s check names verbatim. It also pins the config figures the
index quotes (`rho`, forced hours, `deff`, `information_increment`), since a
re-run moves them and a stale number in a reference gets quoted in a meeting.

### The integration contract

`docs/event_contract.html` is what an integrating engineering team builds
against: the 11 fields they send to request a price, the 36 the service logs
per decision, and the 9 (+2 conditional) they return per outcome, with the
quarantine rules and the event-quality thresholds. Unlike the walkthrough it is
**hand-authored** — there is no builder — which is why it carries a guard the
walkthrough does not need.

`tests/test_event_contract_doc.py` checks it against `events/store.py` in both
directions: every name in `DECISION_REQUIRED` and `OUTCOME_REQUIRED` appears in
the doc, and every field name the doc prints is one the system knows. **Adding
a required event field now fails the suite until the contract is updated**,
which is the point — nothing else in the repo would notice a partner building
against a field list that had quietly moved. Request-state names that differ
from their logged counterparts (`q` → `q_remaining`, `current_discount` →
`anchor_discount`) and the two conditional outcome fields are allow-listed in
that test; extend the list deliberately, not to make a failure go away.

Its thresholds are quoted from `config.yaml` (§06 of the page) and are NOT
guarded — re-read them when `monitoring.stop_conditions` or
`monitoring.shadow_gate` moves.

The worked episode in §05 is real output: every payload was produced by running
`inference.decide` against the frozen artifacts and capturing what it emitted,
with the write-off outcome built by `pipeline.shadow.adjustment_reason`. If the
event schema or the solver changes, regenerate it rather than hand-patching the
numbers — the point of that section is that an integrator can trust the shapes.

### The deck is retired

`docs/perishable_markdown_deck_v3.pptx` (44 slides) and its build input
`tools/deck_source.pptx` (34 slides) are still in the repo, but the six modules
that built, diffed, patched and number-tagged them are gone — the walkthrough
replaced the deck as the thing that gets presented. Consequences worth knowing
before anyone quotes it:

- **The `.pptx` is a frozen document now, not build output.** There is no
  rebuild path and no `deck_diff` guard. Everything on it is as of
  `baseline-20260811043259`, and nothing re-derives it when the pipeline runs.
- **Two of its figures are known wrong.** Slides 2 and 42 give observed IL% as
  `36.68% / 36.7%`; `reports/backtest_calibrated.json` gives `0.3868` — i.e.
  **38.68%**, which is what the walkthrough carries. Do not quote the deck for
  that number.
- If a deck is wanted again, build it from the walkthrough rather than
  restoring the old modules: they encoded a slide ordering the walkthrough no
  longer follows. They are recoverable from git history all the same — last
  present at `10120c8`.

`docs/design.md` and the walkthrough quote ~25 measured quantities that go
stale on every re-run (the launch-freeze retrain moves most of them).
`tools.deck_numbers` used to list them in one block; it is gone with the rest,
so read them off the reports directly. Two rules survive it:

- **Only from a gate-passing backtest** — the same rule that governs pasting
  `tau_initial`. A number from a failing run must not reach a document.
- **Never invent one.** If a report is missing, leave the figure alone and say
  which one could not be refreshed. A plausible-looking wrong number in a
  leadership document is the worst possible output of this task.

