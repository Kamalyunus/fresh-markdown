# Review guide — what to review, how deeply, and why

The part of the repo that can touch a live price is ~2,100 lines. This
guide scopes a code review to risk, so the whole exercise is about two
sittings.

## Tier 1 — review line by line (~1,500 lines; prices real money, hourly)

| File | What to verify |
| --- | --- |
| `engine/dp.py` | The two safety properties are **structural**: `feasible_tiers` cannot express a price below cost or above the anchor (no post-hoc check to forget), and the terminal value books scrap once. The state space is small enough that the solve is exact — check the truncation diagnostic, not the math |
| `engine/demand.py` | `mu(d) = mu_ref × ((1−d)/(1−d_ref))^ε` with a floor; censored expectation `E[min(D, q)]`. This is the only demand math in the system — everything else calls it |
| `engine/explore.py` | Exploration is a **uniform** draw from the tau-affordable subset of `admissible` (tiers at least `delta_min` from the REFERENCE discount in log price — information is measured from the reference, cost from p*) — any weighting would un-randomise the evidence the learner consumes, and the ledger, the chooser and the assurance check must read the one `admissible`. Also: `walk_tau` — tau moves by `clip(budget/spend, 0.5, 1.25)` one step per closed day on the trailing 7-day close-day IL base, and it is the ONE walk (production and shadow's trace) |
| `engine/decide.py` | Validation **rejects** rather than returning a best-effort price; the decision event carries enough to re-solve itself (`mu_ref_path`, `anchor_discount`) |
| `events/store.py` | Append-only, durable writes, dedup on emit AND on load (a duplicated line is counted and loaded once; the id is registered after the append succeeds); malformed events **quarantine with the reason attached** rather than being dropped, and every stream's torn last line is closed on open; exactly three `adjustment_reason` values reconcile inventory |

The review question for this tier is single: *can any path emit an unsafe or
unauditable price?* The test files that pin these properties —
`test_pricing.py`, `test_end_to_end.py` (decision path), `test_restock.py` —
are worth reading as the specification.

## Tier 2 — review the state mutations (~750 lines; the only writes in production)

| File | What to verify |
| --- | --- |
| `daily/update.py` | Only exploration outcomes are eligible; the censored NB likelihood on the grid; information in NB units (`μL²·r/(r+μ)`) deflated by deff; the trigger evaluates the **unconsumed batch**, never a running counter; `predictive_check` scores the batch against the pre-update posterior; tau calibrates on spend, not evidence — `--calibrate-tau` commits it daily with no operator, one step per closed day since the last calibration (a missed day is graded, not skipped), keyed by the decision's TRADING day (`events.pairs.decision_day`, never the outcome's UTC finalize time), and zero spend on a priced day raises it — under-spend, not absence of signal |
| `events/pairs.py` | The one decision↔outcome pairing; `learnable=` is what keeps a failed push out of the evidence |
| `engine/posterior.py` | The bounded step (mean ≤ `max_mean_step`, std shrink ≤ `max_std_shrink`, floored, clipped to the epsilon range); revision + consumed outcome IDs commit in **one atomic write** — a crash between them cannot double-count; re-applying with nothing new is a verified no-op; `launch_stale` is False on ANY production state (outcome, τ, suspension), so the process can never re-initialise a live file |

The review question: *can evidence be spent twice, or a belief move more
than its bounds?* `test_update.py` (the tau walk and the bounded step) and
the exactly-once tests in `test_end_to_end.py` are the pinned answers.

## Tier 3 — skim; the gates and suite carry it (~9,600 lines, offline)

`fit/` (data preparation, model/prior/dispersion fits), `evaluate/`
(backtest, shadow, thresholds) and `ops/` (`advance.py` is the phase order
as code; its `plan()` is pure and unit-tested per stop) run before launch,
produce frozen artifacts, and sit behind human gate readings plus the test
suite. `daily/monitor.py`, `assurance.py`, `ingest_outcomes.py` and
`export_events.py` run in production but write reports, never prices. A defect here cannot touch a
shelf without first passing a gate whose inputs a human reads. Skim for
structure; audit only if a gate behaves surprisingly. The one file worth a
real read is `fit/prepare_data.py`'s waterfall — it defines the
population every other number is measured on, and its rules are
cross-checked against the docs by `test_docs_match_the_code.py`.

`common/` is shared definitions (episodes, config loading, guardrail
comparison, JSON I/O); read `common/episodes.py` if you touch anything that
counts inventory — it is the single source of closure/scrap/censoring truth
— and `common/metrics.py::episode_economics`, the one episode-grain frame
every IL, scrap and margin figure (floors, live guardrail,
business metrics, shadow's budget base) is built on. `events/pairs.py` is
the one decision↔outcome pairing.

## Out of review scope

`tools/` (the fixture generator and the leadership scenario deck, which only
calls the reviewed solver) and `docs/` pages. The test suite is the
reviewers' asset, not their burden: every non-obvious rule named above has a
test whose docstring states it in prose.

CI runs `python3 -m pytest tests/` (a couple of minutes) and `python3 -m ops.status`
on deploy (exit code 1 on any FAIL).
