# Agent operating guide — Perishable Markdown MVP

This file is for any coding agent (Claude Code, Devin, Cursor, …) working in
this repo. The authoritative specification is
`docs/perishable_markdown_mvp_prd.md` (PRD); section numbers below refer to it.
When this guide and the PRD disagree, the PRD wins.

## Setup and tests

```bash
pip install -r requirements.txt
python3 -m pytest tests/          # ~1 min; must pass before any push
```

All commands run from the repo root. Artifact and report paths in
`config.yaml` are relative to the repo root — running a module from another
working directory silently reads/writes the wrong artifacts.

## Pipeline: what runs in what order

```
step                                          writes                                  reads
0. bootstrap.download_flc                     data/flc_raw.parquet                    sb_scm.fresh_flc_detail
   (Redshift extract; REDSHIFT_* from ~/.env, never from config.yaml. Not
   part of run_bootstrap.sh -- it takes the parquet as its argument.)
1. bootstrap.prepare_data --input <raw>       data/prepared.parquet,                  raw FLC parquet
                                              artifacts/split_manifest.json
1b. tools.eda --input prepared               reports/eda.json, docs/eda.html         prepared + config
   (15 descriptive panels on the population. Decides nothing and produces no
   config value -- read it BEFORE the fits, it costs seconds)
2. bootstrap.measure --input <raw>            reports/phase0.json                     raw FLC parquet
3. bootstrap.train_baseline --input prepared  artifacts/baseline_model.txt,           prepared
                                              artifacts/feature_schema.json
4. bootstrap.estimate_prior --input prepared  artifacts/prior.json                    prepared + baseline
5. bootstrap.fit_dispersion --input prepared  artifacts/r_lookup.json,                prepared + baseline + PRIOR
                                              artifacts/rho.json
6. backtest --input prepared --out <json>     reports/backtest*.json                  prepared + baseline + prior + r_lookup
7. bootstrap.train_baseline --fit-calibration artifacts/calibration.json              prepared + baseline
   (only when the calibration gate fails on a level error)
7b. bootstrap.derive_thresholds --input prep. reports/thresholds.json                 prepared
   (evidence for the three SET BY OWNER values: empirical A/B duration vs
   MDE, and 3-sigma noise floors for the scrap/margin guardrails)
8. bootstrap.init_posterior                   artifacts/posterior.json                prior.json
   (once at launch; refuses overwrite without --force -- posterior is
   production learning state)
9. pipeline.shadow --input prepared           reports/shadow.json,                    prepared + all artifacts
                                              events_store_shadow/
10. tools.make_charts                         reports/charts/*.png                    every report written above
11. bootstrap.seal                            artifacts/bundle.json                   every frozen artifact
```

### Two populations, and who reads which

`prepare_data` emits ONE parquet carrying `dp_eligible` (bool) and
`dp_ineligible_reason` (`cost_missing` | `non_priceable` | `window_too_long`,
first match in that order). Nothing economic is dropped.

**A below-cost hour is NOT one of the three.** It is a price the legacy policy
set, and the agent is constrained never to set one, so it is a property of the
history rather than a defect in it. Neither harness needs the episode removed:
the backtest's DP arm is **self-anchored** (`anchor = d_t`, its own previous
choice) and never sees the legacy price, and in shadow the legacy price IS the
anchor, so from the crossing hour the action set is empty and `validate_state`
refuses every remaining hour — the cost floor working, counted in
`rejected_reasons`. The hours *before* the crossing are good decisions the old
chain deleted along with the episode. Those episodes stay `dp_eligible` and
carry `below_cost_hours` for the record.

| Consumer | Population |
| --- | --- |
| `mu_ref`, `r`, `rho`, elasticity prior, level factors | `baseline_model.train_population` — **default `integrity`** |
| `m1` / gate 1 | **integrity, always** — on `dp_eligible` it reads ~0 and cannot fail |
| `m6` IL% | **both**, in `by_population`: integrity is what the business loses, `dp_eligible` is what the MVP addresses |
| `backtest`, `pipeline.shadow`, `tau`, the calibration **gate**, the A/B | **`dp_eligible`, always** — not a choice; the DP has no feasible tier otherwise, and `extend_to_window` refuses a counter above the cap |

Use `bootstrap.prepare_data.population(d, cfg[, which])` — never re-derive the
filter. `dp_eligible` selects rows and **never relaxes a safety property**:
the cost floor is structural in `pricing.dp.feasible_tiers`, and
`validate_state` still rejects an unpriceable state at decision time.

The waterfall's last row, `dp_eligible`, drops nothing — it reports the
subset, per reason, with episodes, rows and COGS. Read it next to the
`cogs_dropped` column to see what the MVP addresses as a share of the
business.

Choosing between the two train populations is a **two-run comparison**, not a
field in one report: flip `train_population`, re-run, compare
`calibration_gate_value` (which is on `dp_eligible` either way). The backtest
records `artifact_versions.train_population` so the two reports cannot be
confused.

`scripts/run_bootstrap.sh <raw>` runs 1–6 in order, then 10 and 11, then prints
`pipeline.status` so a run ends with where it stands rather than with the last
step's log. **It retrains the baseline every time.** To iterate on one step, run
that step's module directly — do not re-run the whole script.

Two things it deliberately does not do. It does not decide the calibration or
prior gates — those are human reviews, and it stops at their evidence. And if
you then run `--fit-calibration` (step 7), **re-run `bootstrap.seal`**:
`calibration.json` is a seventh artifact that did not exist when the set was
sealed, so the seal taken during the script no longer describes it.

Shadow phase (§19 — after gates clear, before any price is applied):

```bash
python3 -m bootstrap.init_posterior
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
```

**It runs on `data.holdout` by default.** Every frozen artifact is fit on data
up to `split.test_end`, so a shadow run that includes that data grades the
pipeline on rows it already saw — the drift ratio, `tau_recommended` and the
learning yield all read better than they are. `--all` runs the whole extract
and the report carries `shadow_gate.in_sample_caveat` naming exactly which
numbers that flatters (and which it does not: completeness, matched rate and
cost-floor test plumbing, not fit). `window.basis` and `window.out_of_sample`
record which happened. No `data.holdout` in config is an error, not a silent
full run.

Shadow runs on a uniform sample of `monitoring.shadow_gate.sample_episodes`
episodes (**default 3,000** — roughly 3.5 minutes against ~47 for a full
18-day hold-out sweep; sample first, always, drawn before `mu_ref` prediction so the cost
scales with the sample, not the extract). 3,000 keeps the standard error on a
rate near 0.99 at 0.18pp against the 1.00pp the gate discriminates on — wide
margin, and fast enough that the gate actually gets re-run after a change.
Override with `--max-episodes N`, or `--max-episodes 0` for every episode —
worth doing once for the final pre-launch record, not for iteration. A sampled
report sets `window.sampled` and adds `shadow_gate.sampling_caveat`; quote the
caveat whenever you quote the zero violation count, and note it covers ~0.9%
of the extract at the default.

**Sampling degrades exactly one figure**, and it is worth knowing which.
The gate reads rates. `tau_recommended` and `spend_over_budget` equate two
quantities that both scale linearly with the sample, so they are
**sample-invariant** — the same at 3,000 as at 40,000. The exception is
`tau_controller_trace.by_day`: it divides the sample across the window's
days, so 3,000 episodes over an 18-day hold-out leaves ~167 behind each day's
budget and spend, and the controller looks jumpier than it is. The trace
reports `episodes_per_day_sampled` against `episodes_per_day_population` and
says so. Quote the pooled `spend_over_budget`; raise `--max-episodes` only if
you are reading the daily series closely.

`exploration_budget_would_be` answers the question the backtest cannot:
**is this `tau` affordable?** `backtest.tau_initial_derivation` reports
`implied_daily_spend` against `daily_budget` and they match BY CONSTRUCTION —
the bisection solves `tau` until they do — but it solves on the **exploit-only
replay path**. Shadow runs the **anchored** path, where each hour's action set
is constrained by the price already in force, so the affordable sets differ
and the same `tau` buys a different amount of exploration. Shadow now reports
both sides on its own basis, same episodes and same days, plus the ratio and
whether it clears the `exploration_cost_vs_budget` stop multiple. **Read it
before the pilot**: over 2× and exploration suspends on day one; between 1×
and 2× the `tau` controller walks it down, capped at halving per day.

Two numbers in that block do the deciding:

- **`tau_recommended`** — the same bisection, re-run on shadow's own
  decisions. Report only: `tau_initial` is MEASURED and goes through the
  paste gate like `rho`, and `artifact_mirror_drift` exists to catch the
  silent-rewrite case. Check `tau_recommended_implied_spend` sits just under
  `daily_budget`; it will never equal it, because spend **steps** as each
  cost crosses `tau` rather than sliding.
- **`tau_controller_trace`** — the day-by-day walk. A single multiple cannot
  say whether the pilot survives its own launch: `tau_next` only reads the
  day just closed, so day one is spent at whatever `tau` you launched with,
  and the stop condition is evaluated on that same day. Three day counts are
  reported and **none is interchangeable** — `window_days` (the calendar span
  the budget divides by), `days_with_decisions`, and `days_simulated`
  (capped at 60; `days_truncated` says what was dropped).

**`tau_initial` is null and any earlier paste is void.** The scoping fix below
changed what the backtest derives, so a value carried over from before is
wrong by roughly the decisions-per-episode factor. Paste only from
`reports/backtest.json` → `tau_initial_derivation.tau_initial`, and only from
a report whose block carries `spread_decisions` (older reports do not).
`pricing.explore.tau_provenance_error` enforces all three failure modes — no
derivation on disk, a derivation predating the fix, a paste that no longer
matches its source. Shadow refuses to start on any of them and
`pipeline.status` reports it as FAIL instead of passing a pasted number
because it is non-null.

**The entry-only scoping bug.** Until the `SpreadLedger` refactor,
`policy_replay` collected spreads at `t == 0` only, so `tau_initial` was
solved to fund roughly **one exploration per episode** against a system that
calls `explore.select` every hour. That is most of any large multiple shadow
reports, and it could not surface in the backtest — the bisection reports
1.00x whatever population it is given. `pricing.explore.SpreadLedger` is now
the single definition, used by both, and
`tests/test_holdout_and_tau.py` asserts the replay collects every hour.

### The hold-out window

`data.holdout` names a window **after** `test_end` that nothing is fit on and
no gate was decided on. Shadow uses it by default; `--holdout` is accepted for
explicitness and changes nothing:

```bash
python3 -m pipeline.shadow --input data/prepared.parquet --max-episodes 0
```

Every artifact stops at `test_end`, so standing there and running this window
forward is the only unrehearsed test the extract can give — the shadow gate,
the drift ratio and the `tau` derivation all report in-sample numbers or
1.00x on any window they were fitted against. **One shot**: tune a value on it
and re-run, and it is a second calibration set, not a hold-out.

Date cuts are **episode-scoped** (`common.episodes.window_slice`), never
row-scoped. Windows run past midnight, so `d[d.date >= start]` keeps the tail
of an episode that opened the day before — no entry decision, wrong opening
inventory, a countdown starting mid-window. Episodes are assigned by the date
their window opened, so nothing straddles the seam. `split_frames` uses the
same function.

Its budget uses `common.episodes.classify_last` for scrap, not a local copy.
An inline copy was written first and dropped **all** scrap on a feed with no
write-off sentinel — that function carries a fallback for exactly that case.
It understated the budget 10× and flipped the verdict from "within budget" to
"WOULD SUSPEND", which would have sent someone to re-derive `tau` against a
budget that was never real. `tests/test_end_to_end.test_shadow_phase_harness`
asserts scrap is present and exceeds the discount term.

Shadow needs `apply_level_calibration` and `tau_initial` non-null. Its exit
gate: event completeness and matched rate above `monitoring.shadow_gate`
thresholds and ZERO cost-floor violations. Shadow outcomes carry
`execution_status="shadow_not_applied"` and are structurally ineligible for
`pipeline.update` — the recommended price was never in force, so they are not
learning evidence. Watch `realised_vs_predicted_sold_ratio_at_legacy_price`
in the report: it is the production continuation of the calibration gate and
the first place frozen-baseline drift shows.

Production loop (after the shadow gate passes):

```bash
python3 -m pipeline.update             # monitor only, always safe
python3 -m pipeline.update --apply     # operator gate; refuses on failed event-quality gates
python3 -m pipeline.monitor            # section 15 families + assurance
python3 -m pipeline.assurance          # the same checks, standalone
python3 -m pipeline.status             # the dozen numbers that decide something
```

### What `--apply` moves, and on what evidence

`artifacts/posterior.json` is the only file production writes, and **two
different things move in it on two different kinds of evidence**:

| moves | on | when |
| --- | --- | --- |
| `mean`, `std`, `version`, `accumulated_information`, `n_obs` | INFORMATION | only when a cell's effective information crosses `learning.information_increment` |
| `processed_outcome_ids` | — | with the revision that consumed them, in the same atomic write |
| `tau`, `tau_calibrated_through` | SPEND (§12.3) | every run, whether or not any cell triggered |

**`tau` moves on spend, not on evidence.** A day that explored and learned
nothing still cost money, and that is exactly what `tau` prices — so it is
calibrated on every `--apply`, independently of the information threshold.
`tau` lives in `posterior.json`, not `config.yaml`: it is production learning
state in the same sense the posterior is, and a running system must not edit
its own hand-maintained source of truth. `PosteriorStore.tau(cfg)` falls back
to `exploration.tau_initial` until the first calibration, and **is what a
production caller should pass to `inference.decide`** — reading
`config.exploration.tau_initial` directly pins `tau` at its launch value
forever. (`pipeline.shadow` reads config on purpose: nothing has been spent
yet, so there is nothing to calibrate from.)

`tau_calibration` deliberately uses **the same two numbers `pipeline.monitor`
compares** for its `exploration_cost_vs_budget` stop condition — realised
exploration cost against `budget_today` on realised markdown IL, both over the
whole event window. One definition, so the proportional correction and the
suspension backstop cannot disagree about what "over budget" means: `tau`
starts shrinking well before the 2× stop condition fires, which is the ordering
that keeps exploration running rather than switching it off. The date stamp is
the exactly-once guard — two runs in a day would otherwise apply the same
ratio twice and move `tau` by its square.

Everything else in that file is static by design: `cell_of` (cell assignment
does not move during the MVP window) and `prior_source` (provenance).

**There is no `information_since_update` counter, and adding one back would be
a bug.** The trigger is evaluated on the UNCONSUMED BATCH, not on a running
total, because nothing consumes a sub-threshold batch — so incrementing a
counter while the same outcomes are re-read next run double counts them. PRD
§13.4 specified the counter and now records why it was replaced.
`accumulated_information` is the running total across committed revisions.

## The frozen artifacts are one bundle

Six artifacts are fitted in sequence and frozen together, and they are only
meaningful together: `rho` deflates evidence measured against one model's
residuals, the level factors correct that same model, the prior used that
model's predictions and that `r_lookup`. **Mix vintages and nothing errors** —
the numbers simply stop describing the same world, silently, for the whole
window.

Every artifact is therefore stamped with the model version it was fitted
against — `provenance.bundle` — and the bundle id IS the baseline model
version, because "which model was this fitted against" is the question that
matters. After a bootstrap run, seal it:

```bash
python3 -m bootstrap.seal        # writes artifacts/bundle.json: id + sha256 of each file
```

Sealing catches what stamps cannot: an artifact edited after the fact leaves
its stamp intact, but not its hash. `seal` refuses an inconsistent set — a
sealed mixed bundle is worse than an unsealed one, because it looks decided.

**When `artifact mirrors` fails, read the `artifact bundle` line first.** The
mirror check says config and an artifact disagree; it does not say which is
stale, and the answer is not always "re-paste from the artifact". If the
artifacts on disk are an older bundle than the model in force, pasting their
numbers into config walks the system backwards — a smaller `deff` over-counts
every future update. Establish the live bundle, then align the stale side.

**Start with `python3 -m pipeline.status`.** The four reports carry ~200 fields
between them, which is the right number to write and the wrong number to read.
`status` prints only the checks that gate a decision — launch blockers, artifact
mirrors, calibration gate, prior, tau, shadow gate, guardrail floors, stop
conditions, assurance — each with the figure behind it and, when red, which
diagnostic block to open next. It computes nothing; every line is read from a
report some other step wrote. Exit code 1 on any FAIL, so it can gate a script.

A check that did not run reports `not run`, never `PASS`: an unrun check and a
passing check must never look the same, and `tests/test_status.py` asserts it.
Everything below `status` is tier two — read it when a gate goes red, not
routinely. The gate decision tree further down maps each failure to the blocks
worth opening.

`pipeline.assurance` (design 5.15) tests the FROZEN ARTIFACTS against live data,
which is the thing the unit suite structurally cannot do — every production
failure this system has had was an assumption failure, not a logic bug. Four
checks, each aimed at something that would otherwise be silent:

- `reproduction` re-solves logged decisions from their own event payload. The
  DP is deterministic, so a mismatch means config, artifact, code or a library
  moved underneath it. **This is why `mu_ref_path` and `anchor_discount` are on
  the decision event** — without them a decision cannot be recomputed, and the
  check cannot tell a drifted artifact from a correct one. Do not remove event
  fields because "nothing reads them": the event is the audit surface, and
  `test_end_to_end` asserts every emitted decision re-solves to itself.
- `dispersion` compares realised zero-sale and stockout rates against
  `NB(mu, r)`. Both are exact under censoring; a variance comparison is not.
- `correlation` re-measures `rho` on live residuals **at the working elasticity**
  -- each category's PRIOR MEAN, via the shared
  `fit_dispersion._working_elasticity`, which is the same basis
  `bootstrap.fit_dispersion` used. Measuring at the posterior mean would show
  drift that is not there; so would measuring at the fallback constant now
  that dispersion is fitted after the prior, and `rho_drift_alert` (0.10) is
  tight enough to fire on a pure basis mismatch. One helper, so they cannot
  diverge.
- `exploration` reconstructs the affordable set and tests that the applied tier
  is a uniform draw from it, plus the invariant that a non-empty affordable set
  always produced an exploration.

Verdicts are `PASS` / `FAIL` / `INSUFFICIENT`, and thin windows report
`INSUFFICIENT` rather than `PASS`. Nothing here suspends pricing: it is read at
the operator gate, because "the world stopped matching the model" is a human
decision. Thresholds live in `config.yaml` under `assurance:`.

## Hard rules — violating these has already caused wrong conclusions

1. **Never retrain the baseline between two runs you intend to compare.**
   The model is frozen by design (§9.3). Any before/after fidelity comparison
   is void unless `artifact_versions.baseline_model_version` is identical in
   both reports. `--fit-calibration` does NOT retrain; plain
   `train_baseline` and `run_bootstrap.sh` DO.

1a. **Changing the ELASTICITY PRIOR invalidates `rho`, and the calibration
   factor with it.** `fit_dispersion` scales `mu_ref` to actual prices at a
   *working* elasticity, so a different prior gives a different `rho` and
   `deff`. Measured: −1.0 → −1.5 moved `rho` 0.3103 → 0.4236 and `deff`
   3.347 → 4.204, and the level-calibration factor from 1.4779 to 1.6222.
   Re-run the dispersion step onward and re-paste the mirrors, or strict
   start-up will refuse on `ARTIFACT_MIRRORS` drift — which is the check
   working, not a nuisance.

   **The working elasticity is now each category's PRIOR MEAN, not
   `fallback_mean`** (owner, 2026-08-24), which is why `estimate_prior` runs
   BEFORE `fit_dispersion`. Fitting at the constant was harmless only while
   the bracket was rejected for all 16 categories and the fallback *was* the
   prior; with brackets accepted per category it measured correlation against
   a demand curve nothing uses. The two steps are circular — the bracket's
   censored NB likelihood needs `r` — and the loop is cut on the prior's
   side. What makes `r → ε` weak is that the bracket drops its censored entry
   rows, so the dispersion-sensitive `logsf` term never fires; see PRD
   §9.4/§9.5.

   **THE DEFAULT PRIOR METHOD IS `profile_density`** (owner, 2026-08-24), and
   it removes the epsilon <-> r cycle rather than managing it. The curve is a
   censored POISSON profile -- the Poisson QMLE is consistent for the MEAN
   whatever the true dispersion is, and epsilon lives entirely in the mean, so
   `r` leaves this step by theorem. The whole curve then becomes the prior as a
   density instead of its argmax. **There is no `fallback_mean`,
   `fallback_std` or `std_floor` in this path**: a flat likelihood gives the
   uniform on the support by construction (-2.025 +- 1.140 on [-4, -0.05]) and
   then shrinks to the MEASURED pooled density. A 50/50 mixture of two point
   masses at a and b has mean (a+b)/2 and std |a-b|/2, so this reproduces the
   bracket exactly wherever both arms were sharp -- it is a generalisation, not
   a replacement. Rows are every stocked hour, deflated by an eps-free deff.
   `method: bracket` still reproduces an older run. See PRD 9.5a/9.5b.

   **EVERY RUN SCORES BOTH METHODS ON HELD-OUT DATA**, in
   `prior.json.holdout_comparison`: the log marginal predictive per row, with
   `oracle` (best epsilon with hindsight) and `uniform` (flat prior) as the
   ceiling and floor.

   **READ IT IN THIS ORDER, EVERY RUN.** These two fields decide whether any of
   the prior machinery earned its place on your extract, and neither can be
   guessed from the fixture.

   1. **`information_available_per_row`** (= oracle - uniform). The entire
      value of knowing this extract's elasticity on that window. **Under
      ~0.01 nats/row the window cannot settle the method question at all** --
      a method gap that is a large share of a tiny number is still tiny, so
      decide on which prior is more honest about what it does not know and
      compare their WIDTHS instead of their scores. The fixture measures
      0.0013, which is why nothing there is evidence for either method.
   2. **`worse_than_a_flat_prior`.** A prior listed here scored BELOW the
      uniform: on that window it is worse than knowing nothing, which is a
      stronger finding than any pairwise gap. On the full-span fixture
      `bracket` lands here and `profile_density` does not -- a confident wrong
      answer costing more than an honest wide one, in nats.
   3. `ranking` and `verdict` last, and only once 1 and 2 are read.

   **ROWS ARE ENTRY ROWS** (owner, 2026-08-25), one per episode, which is what
   PRD 9.5 specified from the start. Scoring every stocked hour buys price
   variation and the within-episode SURVIVORSHIP confound together: a row at a
   deep discount exists precisely because earlier hours did not sell, so
   conditional on a price-neutral `mu_ref` the deeper price reads as LOWER
   demand. `hour_of_day` is already a `mu_ref` feature and the controlled arm
   profiles hour effects out on top, so the evening-lift half is handled twice
   — and the sign STILL came out positive on 4 of 5 fixture categories against
   2 of 5 on entry rows, because selection on the unobserved demand shock is
   not something an hour control can reach. **`rows_comparison` reports the
   sign outcome for BOTH sets every run** — read it before changing
   `posterior.prior.rows`.

   **READ `wrong_sign_categories` AND `std_basis` FIRST.** Two failures the
   method could not see until production data hit it, both now caught:

   * A likelihood peaking OUTSIDE `search_bounds` was clipped to the nearest
     bound and reported as measured — a category came back as a confident
     `-0.05` on 125,749 rows. The peak is now searched past the bound; at or
     above zero (demand rising with price) the category's own density is
     discarded, it takes the pooled one, and it is listed. The cause is the
     ramp: deep discounts land on stock that is not selling. A rejected
     category is also excluded from the POOL, or the fallback inherits the
     confound; with none left the pool IS the uniform, and `pooled_basis`
     says so.
   * `std` could be **zero**. A span of 9,402 log-likelihood units across a
     159-point grid is 59 nats per step, so the density collapses onto one
     point. A zero-width prior is a frozen posterior — `bounded_step` cannot
     move it. The std is now the widest of three MEASURED quantities: the
     density's width, the grid resolution, and `fold_spread` (how far the
     estimate moves across disjoint slices of the train window).
     `std_basis` names the binding one. `fold_spread` binding means the
     estimate is unstable across the window and that instability IS the
     honest width.

   **THE DOCUMENTS REFRESH THEMSELVES FROM THE ARTIFACTS.** Numbers in the
   docs are anchored — `<!--f:rho.implied_deff|dec3-->3.347<!--/f-->` renders
   as `3.347` and nothing else — and `scripts/run_bootstrap.sh` calls
   `tools.refresh_figures --write --dataset "$INPUT"` as step 10b, so a
   completed run leaves the docs saying what its own artifacts say. Check
   without writing at any time:

       python3 -m tools.refresh_figures        # non-zero if anything is stale

   Two rules when adding a figure to a document.

   * **Anchor CURRENT-STATE measurements only.** A figure describing what a
     PAST decision cost — "deleting restocked episodes took 18.1pp of the
     extract's COGS" — is historical. Refreshing it against today's artifacts
     replaces a fact about a decision with an unrelated number and destroys
     the argument it supports. Leave those as prose with their own date.
   * **Never anchor a config value.** `dispersion.rho` and
     `mean_forced_hours_per_episode` are pasted by hand on purpose, and
     `pipeline.status` refuses to start when they drift from the artifacts.
     Auto-writing them would silence a gate that is meant to stop a human.

   `--write` REFUSES a dataset whose name says it is synthetic. Fixture
   numbers are plausible and silent — they read as measurements — and
   production figures in `design.md` were once overwritten with them, with
   nothing about the result looking wrong.

   **TUNE `own_information_saturation` AGAINST PRODUCTION, ONCE.** It is the
   log-likelihood span at which a category stops borrowing from the pooled
   density and stands on its own data; the shipped 2.0 is the chi-square 95%
   cutoff and is a principled default, not a measured one. Read
   `likelihood_span` across categories in `prior.json`: if nearly every
   category clears it, the pooling never fires and thin cells are trusting
   their own noise -- raise it. If nearly none does, every category is being
   dragged to the pool -- lower it. `own_information_weight` per category says
   which way it went.

   **`reference_r` (bracket method only) is DERIVED, not pinned, and fitted PER CATEGORY** — `r` on
   each category's own entry rows at the fallback elasticity, so it cannot go
   stale and there is no constant to re-paste. Computed, not read from
   `r_lookup.json`, so a fresh clone runs. **Do not borrow the artifact's
   global:** it is fitted on the CALIBRATION window, and this likelihood sums
   over TRAIN entry rows.

   **Pooled was the wrong unit.** Dispersion belongs to the category, and the
   spread swamps the ±2× band the approximation was justified over — fixture
   entry rows against a pooled 8.04: SEAFOOD 1.60, VEGETABLE 5.39, FRUIT 14.12,
   MEAT 28.71, SIDE DISH at the 50.0 search bound. Going per-category moved
   FRUIT's midpoint 0.212, half the `std_floor`; the other four fixture
   brackets are pinned at a search bound and cannot move, because the fixture
   builds `corr(discount, hour) ≈ 0.97` by design and is a weak testbed here.
   Boundary fits are clamped at `dispersion.clamp_percentile` — the same key
   §9.4 uses, so the two cannot drift. Read `reference_r_by_category` and the
   per-category `reference_r_scope` (`category` vs `pooled`), not the pooled
   `reference_r` alone.

   **Where it matters most.** The reference being right matters most exactly
   where `identifying_variation_share` is already low, so read the two
   together. Two figures previously in this repo were measured on bases that no
   longer apply: "~0.099 for ±2×" was anchored at the wrong `r`, and the 0.049
   / 0.592 pair that replaced it was measured against a pooled reference the
   step no longer uses.
   **AN `r` AT THE SEARCH CEILING HAS TWO CAUSES AND THEY WANT OPPOSITE
   TREATMENT.** A thin group whose MLE wandered there wants clamping at
   `dispersion.clamp_percentile`. A group that is genuinely steadier than
   Poisson also lands there — because no NB can represent it, `Var = mu +
   mu^2/r` being at least `mu` for every finite `r` — and clamping THAT one
   inflates the variance the model claims for the cell that has least, which
   the DP's censored demand expectation, the posterior likelihood and the
   exploration cost of a tier all inherit. Pearson dispersion
   `mean((k-mu)^2/mu)` tells them apart: below 1.0 the group is exempt from the
   clamp and listed in `r_lookup.under_dispersed_groups`. **If that list is
   long, the NB is the wrong family for the extract and not just for those
   cells** — read `pearson_global` beside it.

   `pipeline.assurance` shares `_working_elasticity` so its live `rho` check
   cannot drift onto a different basis.

   Also worth knowing before anyone proposes a "better" prior: **the policy
   is insensitive to the prior mean anywhere below the deepening bar
   (~2.43)**. The −1.5 re-run chose identical prices — mean discount 0.1285,
   0% deepened — while costing 26% of the learning rate. A larger \|ε\| guess
   buys no behaviour change and slows the loop that would find the real one.

2. **`posterior.epsilon_max` (−0.05) is a sign constraint, never a bound to
   widen** (§10.4). An estimate pinned at the UPPER bound means the estimator
   found no negative price response — an artifact of confounded data, not
   evidence that elasticity is near zero. Positive elasticity must remain
   unrepresentable. (Widening applies only to the LOWER bound, per the −1.5
   defect described in §9.5.)

3. **Do not infer `fallback_mean` from bracket estimates that hit a search
   bound.** A boundary solution is not an estimate (§9.5). If the bracket is
   rejected, use the configured fallback and let production exploration learn
   elasticity — that is the system's entire premise.

4. **The gate window is whatever `baseline_model.calibration_gate_window`
   says** — currently `test`; the run records it as `fidelity.gate_window`,
   so read that field rather than assuming. The rule that matters is that it
   must be **DISJOINT from `calibration_fit_window`** (currently
   `train+calib`), or the gate grades its own fit. The all-history ratio in
   `fidelity.by_window.all` is diagnostic only; when the demand level drifts
   between the training period and launch, no static factor can (or should)
   fix it.

5a. **Everything that compares predictions to realised sales uses the
   CENSORED expectation `E[min(D, inventory)]`** — fidelity, the gate, and
   the level factors. Raw `mu` is always ≥ the censored expectation, so
   mixing bases makes a factor read low: measured, a true correction of
   1.45 fits as 0.68 on raw mu — the wrong side of 1, which is why
   calibration used to leave the gate unmoved (or worse). The factor is
   *solved* on that basis, not divided out, because scaling mu before
   censoring moves the censored total by less than the factor.

5b. **The baseline's training label is censored on purpose, and the level
   factor is what pays for it.** `units_sold` stops at the shelf. Training on
   uncensored rows only was tried and rejected: it drops 14% of rows,
   concentrated in the later hours where the selling happens, so it selects on
   the outcome. A censored likelihood is the right fix but is not off-the-shelf
   under Tweedie and was not bought for the MVP. Measured cost of the trade:
   pre-calibration hourly bias −0.09 on uncensored hours vs −0.21 with censored
   hours included. Do NOT "fix" this by filtering the training set, and do not
   treat the level factor as optional cleanup — it is the second half of this
   decision (design 5.4).

5. **Level-calibration factors are fit on anchor rows only**, over the
   `calibration_fit_window` (default train+calib — measured weekly demand
   swings ±8%, so a factor fit on one fortnight inherits that fortnight's
   anomaly; the 07-13 calib week measured 1.06 against a five-month mean of
   1.30). The GATE stays on whatever `calibration_gate_window` names, and
   must not overlap the fit window (rule 4). A factor **below 1** on a long
   fit window means the model genuinely over-predicts at the anchor —
   investigate before applying; do not apply blindly.

6. **Only the level component may be corrected multiplicatively** (§9.3). A
   sold-ratio that degrades as `|discount − d_ref|` grows is slope error
   (prior elasticity), fixed by re-estimating the prior — never by scaling
   `mu_ref`.

7. **Elasticity identification uses entry-hour rows only** (§9.5:
   same-hour cross-episode variation, never adjacent-hour within-episode).
   Under the legacy ramp, deep-discount rows exist because earlier hours did
   not sell; fitting on all rows biases elasticity toward zero.

8. **`config.yaml` is the single source of every tunable.** No numeric
   literals for tunables in code (§6.1 configuration rule). Adding a tunable
   to code without adding it to config is a review failure.

9. **IL% is always a ratio of sums, reported with its denominator, with
   absolute IL alongside** (§3.5–3.6). Per-episode IL% is undefined for
   zero-sale episodes and must never be computed or averaged.

10. **`pipeline.update --apply` is the operator gate** (§14). It refuses when
    event-quality gates fail; do not work around a refusal. Updates are
    exactly-once — a second `--apply` consuming nothing is correct behaviour,
    not a bug.

11. **The discount column is percent in raw data and a fraction after
    `prepare_data`** — the conversion happens exactly once (§9.1). Never
    convert again downstream; never feed raw data to modules that expect
    `data/prepared.parquet` (only `bootstrap.measure` and
    `bootstrap.prepare_data` accept raw).

12. **The baseline's SKU rate features are computed in `prepare_data`**
    (`sku_ref_sales_rate_30d`, `prior_episode_ref_sales_rate`: anchor-hour
    only, point-in-time, SKU-pooled fallback). A prepared parquet from
    before this feature set will fail prediction with a clear error —
    re-run `prepare_data` before retraining. Never add within-episode lag
    sales, `hours_remaining`, or extra price features to the model: lags
    are mediators of the episode's own price path and corrupt the learned
    elasticity; hours-remaining is planner state; one overwritten price
    feature is the auditable maximum (see design doc 5.4).
13. **Never gate on a condition the constraint already handles.** A
    below-cost legacy hour needs no filter: the DP cannot express that price,
    so it refuses and says so. Dropping the episode instead deleted the good
    hours with the bad one, and deleted the widest price variation the
    extract has from the elasticity bracket. Ask what actually happens if the
    row stays — a loud refusal counted in a report is usually better than a
    silent removal upstream.
14. **An INTEGRITY defect and an ECONOMIC condition are not the same thing.**
    Integrity means the row cannot be believed — negative stock, a null
    category, sales above inventory, an unexplained chain break. Those are
    **dropped**, for everyone, frozen artifacts included. Economic means the
    row is believable and merely **unpriceable** — cost missing, cost above
    price, an hour below cost, a window over the cap. Those are **flagged**
    `dp_eligible=False` and kept.
    Conflating them cost most of the COGS in the extract: every frozen
    artifact was fit on the priceable subset, including the elasticity prior,
    for which below-cost hours are the widest price variation the data has.
    And it made gate 1 undecidable — `share_non_explorable` counts episodes
    whose cost floor leaves too few tiers, and the chain deleted exactly those
    before `m1` looked, so it read 0.0 and could not fail. Before adding a
    filter, ask which of the two it is. The test is simple: **can the demand
    model see it?** `FEATURES` carries neither `cost` nor `hours_remaining`,
    so all four economic conditions are invisible to it.
15. **Cut this data by episode, never by row.** Use
    `common.episodes.window_slice`, which assigns an episode by the date its
    window OPENED. `d[d.date >= start]` keeps the tail of a window that
    opened the evening before as its own short episode — no entry decision,
    wrong opening inventory, a countdown starting mid-window. This is the
    midnight-seam failure the whole episode definition exists to prevent,
    and it was live in `pipeline.shadow`'s `--date-start` until the hold-out
    work. `split_frames` and shadow both call the one function.
16. **Nothing pre-launch may see past `split.test_end`.** The three artifact
    fits are bounded by `split_frames`, but two paths were not and neither
    announced itself: `policy_replay` / `derive_tau_initial` ran on the whole
    frame, so `tau_initial` — a MEASURED launch value — was being fitted on
    the hold-out; and `calibration_fit_window: "all"` resolved to the whole
    frame. Both now go through `bootstrap.prepare_data.pre_launch`, and the
    backtest reports `population.episodes_excluded_after_test_end`.
    `fidelity.by_week` and `by_window["all"]` therefore stop at `test_end` —
    if a week past it appears there again, something upstream of the slice
    has changed. The hold-out is read once, by `pipeline.shadow --holdout`.
17. **A number a procedure solves for is not evidence about that number.**
    `backtest.derive_tau_initial` bisects until implied spend equals budget,
    so it reports 1.00× on any population — including one where the answer
    is eight times wrong. It hid the entry-only scoping bug for the whole
    life of the code. Grade a fitted quantity somewhere it was not fitted:
    that is what `data.holdout` and `pipeline.shadow --holdout` are for.
    The same reading applies to `budget_share_of_il` and every gate whose
    window overlaps its own fit window.

## Reading a backtest report

- `fidelity.fidelity_episode_sold_ratio` = actual ÷ predicted on the gate
  window. Above 1 → model under-predicts; below 1 → over-predicts. The gate
  verdict uses `calibration_gate_metric` (owner-set 2026-08-09:
  `level_at_anchor` — the model's only production job is the level at
  d_ref; the pooled ratio embeds the unidentifiable prior's slope) against
  `calibration_gate_band` ([0.90, 1.10], ~2σ of measured weekly
  volatility). The report's `calibration_gate_metric` /
  `calibration_gate_value` fields name what the verdict used; the pooled
  ratio stays reported as a diagnostic.
- `fidelity.by_window` — compare `train` vs `calib`/`test` sold ratios; a
  large gap means demand-level drift the frozen features don't capture.
  Config-only remedy to try first: move `data.split.train_start` later so the
  model learns the launch-adjacent regime, then retrain (a fresh baseline —
  restart any before/after comparison).
- `fidelity.measurement_10` — `level_bias_at_anchor` far from 1 with a flat
  slope → level error (calibration permitted). Near 1 at anchor but degrading
  with gap → slope error (re-estimate prior).
- `policy_deltas`: the policy verdict is `policy_gap_like_for_like` —
  legacy-under-model vs DP-under-model, same demand generator both arms, so
  model bias cancels. Never compare `actual_*` (observed world) against
  `dp_*` (model world) as a policy statement — that charges all model bias
  to the DP; `actual_*` vs model figures are fidelity only (§17.5). Even
  like-for-like, replay is internal consistency, not launch evidence.
- `tau_initial_derivation.tau_initial` is a currency amount (§12.3). Only
  paste it into config from a report whose fidelity gate PASSED.
- Replay output is never evidence the policy works (§17.1). The A/B is.

## Gate decision tree

```
backtest fidelity gate FAIL
├─ FIRST check fidelity.by_week, and distinguish WOBBLE from TREND:
│  · wobble (swings around a level wider than the band) = week-scale
│    demand volatility — no retrain or calibration can pass it; owner
│    decision: longer gate window, wider band, or gating on
│    level_bias_at_anchor (baseline_model.calibration_gate_metric — the
│    coherent choice when the anchor is in band but the pooled ratio is
│    dominated by the unidentifiable prior's slope).
│  · monotone trend (anchor ratio climbing week over week) = the demand
│    level is in motion and the gated model is STALE — do not tune bands
│    to pass it; the launch verdict belongs to the freeze-time retrain,
│    and in-window level re-fits (scheduled --fit-calibration on a
│    trailing window) track the level thereafter. Check
│    anchor_ratio_by_rate_history first: no_history ≫ with_history means
│    new-assortment SKUs, not a macro trend.
├─ by_window shows train ≉ calib/test  → regime drift: consider later
│  train_start, retrain, re-run (new comparison baseline)
├─ level_bias_at_anchor far from 1, flat slope
│  → train_baseline --fit-calibration (factors ≥ 1 expected)
│  → set apply_level_calibration: true, re-run backtest (NO retrain)
├─ anchor ≈ 1 but slope degrades with gap
│  → re-run estimate_prior; if bracket rejected, fallback prior stands
└─ still failing after both remedies → STOP (§8.1): the MVP does not
   proceed to a learning pilot; escalate to the PRD owner
```

## What blocks launch (strict config)

`common.config.load_config(strict=True)` refuses while any of these is null:
`baseline_model.apply_level_calibration` (decided by the §9.3 diagnostic),
`dispersion.rho`, `dispersion.mean_forced_hours_per_episode`,
`exploration.tau_initial` (from a PASSING backtest),
`monitoring.stop_conditions.scrap_deterioration_pct` and
`margin_deterioration_pct`, `ab_test.min_detectable_effect_pct` (owner
decisions — an agent must never invent these).

MEASURED values produced by the pipeline are pasted into `config.yaml` by
hand; SET BY OWNER values come from the PRD owner only.

**A guardrail threshold means nothing without its basis.**
`monitoring.stop_conditions.deterioration_basis` says, per metric, whether the
comparison is `relative` (`t/c − 1`) or `absolute_pp` (`t − c`), and 0.15
means two different things under the two. Scrap is `relative` — strictly
positive, mean level 0.1814, 3σ floor 0.4156. Margin is `absolute_pp` because
it **crosses zero**: 36 of 134 production days at or below it, which drove the
relative floor to a raw 3σ of 65.4497 and a robust 3.5853, i.e. a guardrail
that could not be set at all. Smoothing cannot fix a sign change; only the
basis can. `derive_thresholds` now stamps `BLOCKED` on any relative floor ≥
1.0 and names the remedy, and the comparison lives once in
`common.guardrail.deviation` so the floor and the live trigger cannot measure
different quantities.

**`dispersion.rho` and `dispersion.mean_forced_hours_per_episode` must be
re-pasted from `artifacts/rho.json` after every retrain.** They set `deff`,
which divides accumulated information in `pipeline.update`, so a paste left
over from a previous model version mis-weights every posterior step for the
whole window — silently, and in the direction of slower learning. Strict
start-up now refuses to run on divergence, but the check only fires in
strict mode: re-paste as part of the retrain, not when something breaks.
Take them from `artifacts/rho.json` (fitted against the model's own
residuals), never from phase 0's `m3_intra_episode_correlation`, which is a
category × hour proxy computed before any model exists and says so in its
own `note`.

For the two guardrail thresholds, `bootstrap.derive_thresholds` measures the
noise floor and stamps `TOO TIGHT` on anything set below it — that verdict is
blocking, not advisory.

**Read the floor on the basis the monitor actually compares against.** The
report carries two:

- `guardrail_noise` — each day vs a **trailing 28-day mean**. Applies only
  before an A/B is running. Measured on production: margin 3σ 0.1363 (robust
  0.1494, well behaved); scrap 3σ **4.7962 raw / 1.5282 robust**, flagged
  `outlier_dominated`. A floor above 1.0 means the series swings by more than
  its own level — **no scrap threshold on this basis is both safe and
  useful**, and quoting one is a mistake.
- `guardrail_noise_control_arm_basis` — **same-day treatment vs control**,
  using the identical arm hash as `pipeline.monitor`, and each arm smoothed
  over `deterioration_smoothing_days` **before** the two are differenced,
  exactly as the monitor does it. This is what the monitor uses once both
  arms are populated, and it cancels the day effect that dominates the scrap
  series.

Both bases are smoothed the same way the monitor smooths, and both must be
consulted: **one config value is graded against the trailing mean before the
A/B and against the control arm during it, so it has to clear the larger of
the two floors.** Read that off
`guardrail_threshold_recommendation`, which reports both floors, names the
binding one, and stamps the verdict. Do not sign a threshold off the
`guardrail_noise` line alone — it only speaks for the pre-A/B phase.

Two verdicts are blocking, not advisory:

- `TOO TIGHT` — below the binding floor. Fires on ordinary days and silently
  suspends exploration, which is the product. A margin threshold under ~0.136
  is in this band. Buy sensitivity back with `persistence_days`, never by
  going under the floor.
- `CLEARS THE FLOOR BUT LIKELY INERT` — more than 3× the binding floor. Such
  a threshold passes every check and still cannot fire, especially with the
  persistence rule on top. A guardrail that cannot fire is not a conservative
  setting, it is an absent one: change the metric or add an absolute floor
  instead of accepting the number.

## What counts as a usable episode

`bootstrap.prepare_data` runs a deterministic, auditable filter chain; the
waterfall in `artifacts/split_manifest.json` records rows, episodes and
**COGS at risk** (unit cost × **supply** — opening stock plus gross arrivals — counted once per episode) after
every step. **Almost every filter drops the WHOLE EPISODE, not the offending
row** — a hole punched mid-window re-segments into a spurious short episode,
which is worse than losing the episode.

**Only integrity and scope rules DROP.** A stage removes an episode when its
rows cannot be believed (impossible quantities, an unreconcilable chain, two
states for one hour), cannot be used by anything at all (no category, no base
price), or sit outside the period the study covers. Nothing is dropped for
being hard to PRICE. Those conditions are FLAGS on the surviving frame, and
the reason is one fact: `bootstrap.train_baseline.FEATURES` carries neither
`cost` nor `hours_remaining` nor anything about the inventory chain, so the
demand model cannot see any of them and such an episode is an ordinary
observation to every frozen artifact. Dropping them removed **>70% of the
extract's COGS** from every fit, including the elasticity prior — which is
starved of price variation and for which below-cost hours are the widest
spread in the data.

**`python3 -m tools.export_waterfall --input <raw>.parquet`** writes the whole
thing as a workbook: the stages with `kind` and `used_by`, then three WHOLE
episodes per removal reason drawn from the raw feed, then every rule in prose.
The example episode ids come from `load_and_filter` itself as it drops them --
an exporter re-deriving which episodes a filter removed would be a second copy
of the chain, disagreeing silently in the document meant to establish trust.

### The chain (13 waterfall rows)

Every row carries `kind` and `used_by`. **`kind: hard_drop`** — the rows leave
the frame, so every consumer downstream sees the same population and "who uses
this" is not a question. **`kind: population_gate`** — the last two rows, which
drop nothing at all; they are flags, and they are the only place the consumers
diverge. Read `used_by` before quoting any row as "the data we trained on".

| Step | Scope | Drops |
| --- | --- | --- |
| `raw` | — | the starting count, before any drop |
| `duplicate_hour_rows_dropped` | rows (both copies) | two states for one sku x fc x hour; no way to choose, and they collide two runs into one episode id |
| `gap_split_windows_dropped` | episode, **every fragment** | a hole in the hourly feed splits one source window into two episodes, and neither is one. The first ends with no closure sentinel — `not_closed`, scrap unknown, clearance a partial figure against a window that did not end. The second opens MID-WINDOW: wrong starting stock, counter part-way down, and its first row reads as an ENTRY row, which `estimate_prior` fits elasticity on. Detected from the counter, which across a gap falls in step with the clock; a genuinely new window resets it upward instead |
| `exclusion_window_removed` | episode | any episode with ANY hour in the known demand-issue window. SCOPE, not integrity: the rows are fine, the period is not |
| `discount_out_of_range_dropped` | episode | discount outside [0,1] — the percent->fraction conversion applied twice or not at all |
| `negative_quantities_dropped` | episode | impossible quantities: negative inventory or sales. **Not `cost <= 0`** — that is a flag now; see the note below |
| `null_category_dropped` | episode | missing category/subcategory (no reference discount, no dispersion cell). EPISODE-scoped: row scoping punched a hole mid-window, which re-segmentation then had to clean up — and it manufactured chain breaks, 20 of them on the fixture, that the feed never had |
| `zero_base_price_dropped` | episode | `original_price` still null/zero after ffill+bfill within the episode. EPISODE-scoped, same reason |
| `episode_universe` | episode | the three conditions that make an episode's inventory readable, evaluated once and BEFORE any filter with an opinion about price, category or cost. Only continuity DROPS; see below |
| `contiguous_episodes_built` | — | re-segmentation, and now a NO-OP that RAISES if it ever stops being one. It used to split windows that row-scoped drops had holed; every drop after the ids are assigned is episode-scoped, so nothing punches a hole. The invariant is load-bearing and invisible — `episode_universe` runs BEFORE this, so a future row-scoped filter would leave its continuity check and every id-keyed flag stale, silently. Hence an assertion rather than a bare recompute |
| `negative_window_recovered` | episode | **not a drop.** A counter entering ALREADY negative is a known source pattern, not a defect — see the note below the table. **Runs AFTER the re-segmentation check, deliberately** — it is the one step that MUTATES `hours_remaining`, the field the ids are derived from. The last `hard_drop` row, so its counts ARE the `integrity` population |
| `eligible` | — | **not a drop — a GATE.** `accounting_closes & final_hour_clean & closed`. The population the DEMAND MODEL and the frozen artifacts read (when `baseline_model.train_population` is `eligible`, which it is), and the population every scrap / IL / clearance figure reads unconditionally — `scrap_units` returns NaN outside it. It used to be missing from the waterfall entirely, so its cost was folded into `dp_eligible` as though the solver had caused it |
| `dp_eligible` | — | **not a drop — a GATE.** How much of the surviving population the DP can act on, with a per-reason breakdown in its detail block. Read by the DP solver, backtest, shadow, the calibration gate and the A/B, which pass it explicitly because for them it is not a setting |

### The source's inventory convention

**`ending_inventory` is the FINAL quantity on hand at the close of the hour,
AFTER anything that arrived during it.** It is not `starting − sold`; it is
what the source counted at the end. Every hour-level rule follows from that
one sentence, and `common.episodes.hour_status` is the only place it is
written down:

| | Meaning |
| --- | --- |
| `ending == starting − sold` | ordinary hour, nothing arrived |
| `ending > starting − sold` | **RESTOCK.** Holds whenever stock arrived — including an hour that sold MORE than it opened with, where `starting − sold` goes negative and any ending exceeds it |
| `ending == 0` and `net > 0` | the source wrote the remainder off: how a listing closes |
| `0 < ending < starting − sold` | stock left unsold and unwritten-off |

And ACROSS hours, with **no legitimate exception**: `starting[t+1] ==
ending[t]`. `ending` already carries the restock forward, so the chain is
continuous. This is the only hour-level rule that still DROPS — a violation is
the feed contradicting itself about one instant.

**`units_sold > starting_inventory` is not an impossible quantity.** It is the
restock signal in its plainest form, and `units_gt_inventory_dropped` deleted
18.1pp of the extract's COGS on the strength of calling it one — while
`adjustment_reason`, the same rule `events.store` enforces live, would have
named those rows `intraday_restock` if the stage had not run first.

### The episode is the unit of reconciliation

An hour-level shortfall is NOT a drop. An hour can genuinely take stock in
(`ending > starting − sold`) or genuinely lose it (`0 < ending < starting −
sold`), and both are events the source is reporting rather than defects.

**Both are counted GROSS, never netted against each other.** A window that
takes 2 units in and loses 2 to shrink has 2 of each, not zero of both. An
earlier version netted adjacent pairs away on the theory that each was one
sale bucketed an hour late — an inference dressed as arithmetic. It read a
restocked episode as ordinary, let it into the DP-side population, and priced
its clearance against a supply short by the units that arrived.

```
opening + restocked  ==  sold + scrap          where scrap = leftover + shrink
clearance            ==  sold / supply         cannot exceed 1
```

Every unit an episode ever had has exactly one fate: it sold, or it is scrap.
Shrink is scrap — those units were paid for and returned no revenue, and
keeping them out left the economics with a hole in the middle. There is no
third fate, so this is not a heuristic with a tolerance: it balances or the
arithmetic is broken. `common.episodes.flow_identity_violations` enforces it,
`dp_eligible.flow_identity` reports it every run, and `tests/test_end_to_end`
asserts it on every episode of the prepared frame. Chain continuity makes the
two sides provably equal, so a violation is a bug here rather than a defect in
the feed — worth checking anyway, since it caught one.

### The close: two independent facts

**DID IT CLOSE** is ONE condition — `ending_inventory == 0` on the last row.
The source zeroes that field when a listing ends, whatever remained, so the
zero IS the closure and its absence means the window was still running when
the extract stopped. Nothing else is consulted: not `hours_remaining` (nominal,
still positive on ~99.9% of final rows), not proximity to the extract's last
timestamp, and no longer a frame-wide fallback that declared every episode
closed when the sentinel was missing anywhere. That fallback was removed
because it failed in the one direction nobody can see — a feed that STOPS
emitting the sentinel reads as perfectly healthy — and it did exactly that: it
hid a synthetic fixture that had never modelled the convention at all, so the
closure path went unexercised for months. `ending_summary.
write_off_convention_in_force` is the flag to read before believing an
unclosed share.

**WHAT THE CLOSE WAS** is the sign of the leftover on the last row, read
**UNCLIPPED** — `starting − sold`, not `max(0, starting − sold)`:

| Last row | `close_outcome` | Meaning |
| --- | --- | --- |
| `sold < starting` | `scrap` | `starting − sold` is left, and that is scrap |
| `sold == starting` | `censored` | the shelf emptied. True demand is only known to be `>= sold` |
| `sold > starting` | `final_hour_restock` | stock ARRIVED during the close. With `ending` also zeroed, how much arrived and how much was scrapped are two unknowns with one equation |

Clipping folds the third case into the second, which is how a restocked close
spent months reading as a clean sell-out. The two axes are independent on
purpose: an episode can be censored and unclosed at once, and one three-way
label could not say so.

**Censoring is decided at the LAST ROW only.** It cannot happen anywhere else:
the source stops emitting rows once inventory reaches zero — which is why
`extend_to_window` exists — so an empty shelf ends the episode. Measured on
the extract: 259 of 259 rows with `starting == sold` are final rows, and no row
anywhere has `starting_inventory == 0`. `censoring_off_last_row` reports any
row that breaks it.

### Three nested populations

| | What it guarantees | Who reads it |
| --- | --- | --- |
| `integrity` | rows that can be believed | nothing, by default |
| **`eligible`** | **three conditions, stated once in `episode_flow`** — see below | **the frozen artifacts** (`baseline_model.train_population`, default) |
| `dp_eligible` | `eligible` plus what the SOLVER needs — a feasible tier, a readable horizon, one inventory pool | the DP, calibration gate, backtest, shadow, A/B |

`eligible` is **three conditions and no more**, all three evaluated in
`common.episodes.episode_flow` and exposed as one column:

| # | Condition | Column |
| --- | --- | --- |
| 1 | the episode reconciles — `opening + restocked == sold + scrap` | `accounting_closes` |
| 2 | the final hour is clean — `starting − sold >= 0` on the last row | `final_hour_clean` |
| 3 | the episode CLOSED — `ending_inventory == 0` on the last row | `closed` |

All three were live before this; the third was assembled independently at each
consumer — `prepare_data` ANDed `outcome_known` in, `eda` and `scrap_units`
each re-derived it — which is three chances to forget it and no single place
to read the definition. `scrap_units` returns NaN on exactly `~eligible` now,
rather than on its own two-way list that had drifted from this one.

The middle tier exists because of the censored likelihood, and that corrected
an earlier mistake in this file. The argument used to be that the demand model
cannot see the inventory chain — `FEATURES` carries neither `cost` nor
`hours_remaining` — so every surviving episode is an ordinary observation to
it. That is wrong in one place: the likelihood treats an hour as censored when
the shelf ran out, and **that call is read off the inventory**. An ambiguous
final hour means an untrustworthy censoring flag, and a wrong censoring flag
biases demand directly.

Everything `dp_eligible` additionally rejects IS invisible to the model, so
those episodes stay in `eligible`: a missing cost, an unreadable horizon, a
mid-window restock and mid-window shrink all leave the sales and the censoring
call intact.

The prepared frame carries `units_restocked`, `units_shrink`,
`episode_supply`, `episode_scrap`, `episode_clearance`, `final_hour_clean`,
`outcome_known` and `episode_eligible`.

### The flags (`tag_dp_eligibility`)

Five conditions gate `dp_eligible`. Each names something the SOLVER cannot do,
never something wrong with the data. An episode failing one stays in the frame
with `dp_eligible = False` and `dp_ineligible_reason` set to the FIRST it
trips, so the reason column reads as a cause.

| Flag | Why the DP cannot price it |
| --- | --- |
| `cost_missing` | `cost <= 0` — a MISSING cost, not a free good. `d_max` reads 1.0, so the DP would discount to the tier cap believing scrap is free, and IL reads zero |
| `non_priceable` | `cost >= original_price`, so `d_max <= 0` and `feasible_tiers` is EMPTY |
| `negative_window` | `hours_remaining` still `< 0` after recovery. The DP takes its horizon from the counter and `extend_to_window` builds the synthetic tail from it; neither can read a negative one |
| `window_too_long` | `hours_remaining` above `data.max_window_hours` (**120**) — flc_window carries very large values from upstream data issues. Raised from 48 by the owner: 48 was cutting legitimate multi-day windows, not only defects. `extend_to_window` RAISES above the cap, so this is a crash rather than a refusal |
| `outcome_unknown` | the episode never closed inside this data — no write-off sentinel on its last row. Gates `eligible` too: an unfinished episode is not a complete observation of anything, and two consumers silently mis-weighted one before this flag existed |
| `final_hour_restock` | the last row sold more than it opened with, so stock arrived during the close and the leftover is a guess — two unknowns, one equation. Gates `eligible` too |
| `unreconciled` | the episode's flow identity fails: `supply != sold + remaining`. Stock moved that no sale, restock or write-off accounts for, so clearance would read above 1 or scrap would appear on a window that sold everything. No trustworthy clearance, scrap or IL — `scrap_units` returns NaN and `unreconciled_anomalies` in the manifest says where they sit, by category and month, for the business to deep-dive |

Two more are flagged and gate **nothing**:

| Flag | Why it does not gate |
| --- | --- |
| `below_cost_hours` | any hour whose OFFERED price is under cost. That is a price the LEGACY policy set and the agent is constrained never to set, so it is a property of the history, not a defect in it. The backtest's DP arm is self-anchored (`anchor = d_t`) and never sees it; in shadow the legacy price IS the anchor, so from the crossing hour the action set is empty and `validate_state` refuses — correct behaviour, counted in `rejected_reasons`, and the hours BEFORE the crossing are good decisions the old chain deleted with the whole episode. Test `original_price × (1 − discount)`, NEVER `applied_price`: the source zeroes that on zero-sale rows (~78% of rows) |
| `edge_truncated` | of the unfinished episodes, the ones the extract boundary explains. Does not gate on its own — `outcome_unknown` already did — but it is the diagnostic that says whether the unfinished count is the boundary or a feed problem. Only the extract's last hours can leave an episode unfinished, since `window_slice` assigns episodes whole by opening date, so on a 175-day extract of ~36h windows the count should be under 1%. Production measured 3.38% |

Who reads which population is ONE decision, in `baseline_model.train_population`
(default **`integrity`**), resolved through `prepare_data.population(d, cfg,
which)`. The three artifact fits read the config; the DP, the calibration gate,
the backtest and shadow always pass `"dp_eligible"` explicitly, because for
them it is a precondition rather than a choice.

**"Episode `Σ sold` exceeds its opening stock" is not a defect at all.** It is
what a restock looks like. The invariant that does hold is against SUPPLY —
`sold <= opening + restocked` — and it follows from the identity rather than
from any filter, since scrap is non-negative.
`test_prepared_data_is_priceable_and_self_consistent` asserts it on the output.
The older form, against opening stock, was simply false the moment restocked
episodes stopped being excluded: 13 episodes tripped it, every one correctly.

**A counter that enters ALREADY negative is recovered, not dropped.** Some
SKUs arrive with `flc_window` set to a large negative constant rather than a
countdown. Dropping them is not neutral — they concentrate in a handful of
categories, so it selects on category and biases the prior, the per-subcategory
`r`, and every category-level IL figure. They behave like a standard short
window, so `negative_window_recovered` rewrites the counter as a synthetic
countdown from `data.manufacturing_window_hours`.

**The cap is a claim about the data, and the stage checks it rather than
trusting it.** An episode entering negative that runs LONGER than the cap is
not the pattern: it is not recovered, it is flagged `negative_window`, and its
count is reported as `episodes_entering_negative_but_longer_than_cap`. **If
that count is not near-zero on your extract, the cap is wrong — fix the cap, do
not widen the recovery.** Recovery writes a countdown, never a clamp: the
counter is load-bearing three ways (episode identification differences it, the
DP takes its horizon from it, `extend_to_window` generates the tail from it),
and a flat value would make re-segmentation split every hour.

**Recovery runs AFTER `contiguous_episodes_built`, and the order is not
cosmetic.** It is the only step in the chain that MUTATES `hours_remaining`,
and that is the field `assign_episode_ids` differences to find window
boundaries. Run it first and the invariant is graded against a counter the
pipeline just invented — and the synthetic countdown can line up with a real
neighbour: an episode entering negative, rewritten to 23, 22, 21, sitting one
hour before a genuine window that opens at 20 reads as one continuous run, and
the two MERGE into an episode with a fabricated boundary. That fired on the
production extract (**165 rows**) and the assertion reported it as "a filter is
dropping rows", which was the one explanation that was not true — every drop is
`isin(episode_id)`-scoped. With the check first, recovery only rewrites values
inside boundaries that are already settled. **Do not move it back above.**

**Only CONTINUITY drops, and it uses a rule production also enforces.**
`ending[t]` must be `starting[t+1]`, and separately every live outcome must
reconcile or name a reason via `adjustment_reason`, exactly as
`events.store._validate_outcome` requires. The write-off is recognised **by
the zero** — the source writes stock off at its own window close — but the
exemption applies to the LAST ROW only: mid-episode a zero ending with stock
still owed is shrink, not a close, and exempting it there loses those units.

Two hour-level tests used to live beside it and neither was right. An hour
selling more than it opened with is a RESTOCK, and deleting it took 18.1pp of
the extract's COGS while `adjustment_reason` — the production reconciler —
would have named those rows `intraday_restock` had it been asked. An hour
whose ending falls short is shrink, which settles into scrap at the episode
level; dropping the episode for it deleted the fastest-selling windows first,
since a sale is likelier to straddle an hour boundary the more the SKU sells.

**`cost_missing` tests `cost <= 0`, not `cost < 0`, and the `=` is
load-bearing.** It was `< 0` until a zero cost crashed the solver, and the
damage ran in two directions.

`non_priceable` tests `cost >= original_price`, so a zero cost gives
`d_max = 1.0` and reads as *maximally* priceable — nothing downstream catches
it. That put a 100% discount in the action set, and
`mu(d) = mu_ref · ((1−d)/(1−d_ref))^ε` at `d = 1` is `0 ** negative` — a
`ZeroDivisionError` out of `pricing.demand`. Quieter and worse: **scrap is
`cost × leftover`, so those episodes contributed discount cost and no scrap at
all**, deflating every IL figure measured over them. The crash was the symptom;
the deflation was the cost. Nobody gives perishable stock away: a zero cost is
a MISSING cost.

It surfaced from `pipeline.shadow --max-episodes 0`. The 3,000-episode default
had never drawn a zero-cost episode, so **the gate passed on a sample that hid
a crash** — quote the sampling caveat for more than the violation count.

The fix is in two layers on purpose. `pricing.dp.feasible_tiers` excludes any
tier whose price is not strictly positive, so the action set is safe whatever
reaches it — that layer owns "which prices are legal" and must not depend on a
filter upstream. The `cost_missing` flag then keeps episodes whose cost we do
not know out of every DP-side number. **Neither makes the other redundant**:
the flag cannot protect a production caller, and the tier rule cannot
un-deflate an IL baseline. It is a FLAG rather than a drop because the demand
model never sees `cost`: the episode is a perfectly good demand observation
and only its economics are unknown, which is why `m6_il_pct` reports
`by_population` and excludes `cost <= 0` from both bases.

Restocks are detected on the inventory CHAIN (`next starting_inventory >
max(0, this starting_inventory - units_sold)`), never by comparing against
`ending_inventory` — that field is zeroed at the window close, so an equality
test would flag every episode's last hour. In production a restock can still
happen after the fact; the outcome records it with `adjustment_reason`.

**Excluding restocked episodes is a DP-SIDE rule, not a data-quality one, and
production absorbs them — `tests/test_restock.py` is what holds that claim
up.** The question gets asked every time someone reads the filter table. The
agent is a policy re-solved each hour rather than a plan, so a restock is just
a larger `q` on the next call; the monotone price constraint does not bind the
wrong way, because more stock argues for a *deeper* discount and deeper is
always allowed; and `pipeline.monitor` reads scrap off the LAST row's
`starting_inventory`, which already carries the restock, so IL comes out right.
The test runs a three-hour episode that gains five units mid-window and asserts
IL to the won.

One thing genuinely degrades, and is pinned rather than fixed. `grid_update`
flags censoring with `units_sold >= starting_inventory`, which is wrong for an
hour that sold MORE than it opened with because stock arrived during it:
demand was observed exactly, and the likelihood uses "at least
`starting_inventory`" instead. That discards information rather than biasing
epsilon — the safe direction — so it is recorded, not repaired. Make it an
exact count if you like, but do it deliberately: the test will tell you.

**Any outcome whose inventory does not reconcile MUST name a reason or it is
quarantined** — and a quarantined outcome never lands, so event completeness
drops and the shadow gate fails. Exactly three reasons are legitimate:

- `intraday_restock` — `ending_inventory > max(0, starting - sold)`
- `episode_close_write_off` — `ending_inventory == 0` while stock remained.
  This is ~49.5% of episodes. An integration that omits it quarantines
  roughly half its outcomes and fails the gate for what looks like a pipeline
  defect.
- `unexplained_shortfall` — shrink: `0 < ending_inventory < starting - sold`.
  **This one returned None on purpose until it was measured.** The theory was
  that an unexplained loss should quarantine and stay visible; the effect was
  that the live path treated shrink as an anomaly while the offline chain
  treated it as an ordinary event — counted gross, booked into scrap, gating
  nothing. Every shrink hour then subtracted from `event_completeness`, so the
  gate (`min_event_completeness`, **0.99**) failed by the feed's shrink rate,
  for a reason no integration work could fix: it was measuring the SOURCE. At
  ~2.8% of decision hours the harness read 0.9718. Quarantine is for what the
  system cannot interpret; a shrink is interpreted, so it is named, and the
  units stay visible through `units_shrink` and `episode_scrap`.

**Recognise the write-off by the ZERO, never by position in the episode.**
Two earlier versions keyed it to `hours_remaining == 0` and then to "our last
observed hour"; both quarantined real outcomes in bulk. The source zeroes at
ITS OWN episode boundary, and once a window is merged across midnight that
row sits in the MIDDLE of ours. Position is our bookkeeping; the zero is the
source's fact. `pipeline.shadow.adjustment_reason` is the one implementation
— production integrations should call it rather than reimplement.

A PARTIAL shortfall — `0 < ending_inventory < leftover` — matches no
convention and is left undocumented on purpose, so it quarantines. That is
unexplained inventory loss and the quarantine file is the only place it is
visible; do not add a catch-all reason to drive the count to zero. Do not add a blanket reason to
make the count go to zero — the quarantine file is the only place that
failure is visible.

The window cap is load-bearing beyond data hygiene: `hours_remaining` drives
episode identification, the DP horizon, and the synthetic tail that
`extend_to_window` generates. An unbounded counter would generate an
unbounded frame, so the extension raises rather than hanging if a frame ever
reaches it above the cap. The bad value is dropped, never clamped — clamping
would invent a window end the data never recorded.

Postconditions are asserted by test, not assumed: discount in [0,1],
non-negative quantities, sales <= inventory, `d_max > 0`, category present,
no hour inside the exclusion window, `hours_remaining` within the cap, and a
monotone window counter inside every episode.

## Multi-day episodes

FLC windows commonly run past midnight; 36-hour windows are common. An
episode is therefore NOT keyed by date. It is a maximal run of consecutive
hourly rows for one sku x fc over which the source `hours_remaining` counter
ticks down exactly one per elapsed hour (`prepare_data.assign_episode_ids`).
Both signals are required: time alone merges back-to-back windows, the
counter alone stitches across missing rows.

Three things follow the episode, not the row date, and must stay that way:

1. **Split assignment** — an episode belongs wholly to the split its window
   STARTED in (`split_frames`), or the train/calib boundary runs through the
   middle of an episode.
2. **Velocity features** — read as of the episode's FIRST date. Per-row
   keying lets a window's second-day rows read a trailing window containing
   that same episode's first-day sales.
3. **`prior_episode_ref_sales_rate`** — computed at episode grain; a daily
   shift hands a multi-day episode its own earlier day.

Duplicate `(sku, fc, date, hour)` rows are dropped outright
(`duplicate_hour_rows_dropped` in the waterfall) — both copies, since there is
no way to pick. Left in, they collide two runs into one `episode_id` and the
window counter stops being monotone.

An episode ends at the window end OR at zero inventory, whichever comes
first, so its row count is NOT its window length. `m11_episode_endings` in
`reports/phase0.json` splits the three cases: `completed` (hours_remaining
hit 0 -- leftover inventory IS scrap), `sold_out_early` (no scrap by
construction), `truncated` (no recorded window end -- scrap UNKNOWN).

**`ending_inventory` IS ALWAYS ZERO ON AN EPISODE'S LAST ROW** -- the source
writes off the remainder when the window closes (~49.5% of episodes end this
way). Reading it as scrap reports ZERO SCRAP EVERYWHERE and silently deletes
the scrap term from IL; dropping those episodes as "broken chain" keeps only
guaranteed sellouts. Scrap is `max(0, starting_inventory - units_sold)` on the
last row -- `common.episodes.leftover_units` is the only definition, and
`scrap_units` wraps it, returning NaN for truncated episodes so a sum cannot
treat unknown as zero. Truncated episodes are excluded from scrap
and IL aggregates, with the excluded share reported.

**No unclosed episode is dropped, and the two reasons are told apart by a
flag.** On production, unclosed episodes were 3.38% of the count holding
**78.6% of all at-risk leftover units** (334,622 against 91,096) — 24.9 units
each against 3.05 — so what happens to them decides how much of the scrap
picture is real. Two causes:

| | Test | Flag |
| --- | --- | --- |
| **edge** | last row's timestamp + `hours_remaining` runs past the extract's last hour (or the last row IS that hour) | `edge_truncated = True` — unknowable here, only a longer extract closes it |
| **not edge** | the window ended inside the data and no sentinel appeared | `edge_truncated = False`, still `not_closed` — a feed problem no re-download fixes |

Both stay in the population and both stay `dp_eligible`. **Dropping the edge
group was the default until it was measured.** It cost the demand fit the
largest, slowest, most heavily stocked windows in the extract to protect a
scrap figure that was already protected three times over — `scrap_units`
returns NaN for an unclosed episode, `backtest.replay` zeroes its scrap under
`outcome_known`, and `pipeline.shadow` charges scrap only on `COMPLETED`.
Only the ENDING is missing; the observed hours are ordinary priced demand.

**Do not "clean up" either group.** Dropping the non-edge one drives
`m11.not_closed` and `scrap_units_unknown_not_closed` to zero BY CONSTRUCTION
and hides a systemic feed problem behind a tidy population — that mistake was
made once already and caught in review.

Where to read what:

- **`dp_eligible.edge_truncated` in the waterfall detail** —
  `share_of_unclosed_explained_by_edge` (near 1.0 → the whole problem was the
  extract cut), `episodes_unclosed_not_edge`, and the leftover units on each
  side. `still_dp_eligible` should equal `episodes_edge_truncated`: if it does
  not, something has started gating on a missing outcome.
- **`m11.not_closed_by_month` / `not_closed_by_category`** — whether the
  residue is one incident, a standing property of the feed, or one corner of
  the catalogue. `not_closed` counts BOTH kinds now, so read it beside
  `share_of_unclosed_explained_by_edge`: concentrated in the LAST month is the
  boundary, evenly loaded months mean no re-download will fix it.

The DP horizon comes from the WINDOW, not the row count. `backtest` and
`pipeline.shadow` call `common.episodes.extend_to_window` before predicting,
which appends the hours a sold-out episode never recorded (marked
`is_observed = False`). Without it the horizon is short precisely because the
item sold out -- lookahead bias on ~10% of decision rows, biased toward
over-discounting fast movers.

Anything measuring the model against reality -- fidelity, the calibration
gate, the likelihood, IL -- must filter to `is_observed`. A synthetic row has
no sales and reads as a pure under-prediction. Sort by
`["episode_id", "date", "hour_of_day"]`, never `hour_of_day` alone: a window
running past midnight comes out scrambled.
`validate_state` rejects any decision whose `mu_ref_path` length disagrees
with `hours_remaining`.

**Any figure measured before this change is void** — IL baseline, clearance,
rho/deff, guardrail noise floors, replay IL. Re-run the full bootstrap.

## Charts

```bash
python3 -m tools.make_charts        # -> reports/charts/*.png
```

Seven charts — exactly the ones `docs/design.md` embeds — every one generated
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

## Repo conventions

- Modules are run as `python3 -m package.module` from the repo root.
- `data/`, `reports/`, `artifacts/`, `events_store/` are gitignored run
  outputs — never commit them.
- **`--workers N` on `backtest` and `pipeline.shadow`** (`0` = every core but
  one) parallelises the episode loop via `common.parallel.map_episodes`.
  Measured on this repo: backtest 99s → 38s, shadow 95s → 36s, reports
  byte-identical. Two invariants make that safe and both are tested:
  results come back in **submission order** (never completion order), and
  **workers compute while the parent commits** — a worker gets a
  `_BufferStore` with no `emit_outcome` at all, so every event still goes
  through the real `EventStore` in the parent, where the dedup and quarantine
  the shadow gate MEASURES actually run.
- **Each episode draws from its own generator**, seeded from its episode id
  (`shadow._episode_seed`). The old shared generator made an episode's
  exploration draw depend on how many episodes preceded it, so a reordered or
  split run stopped reproducing. Order-independence is the property; parallel
  execution is what needed it. Note this **changed the numbers** from any run
  before it — same seed, different draws.
- **Every waterfall stage reports money, not just counts.** `cogs_at_risk` is
  unit cost × supply (opening stock plus gross arrivals), once per episode (never summed over hours —
  inventory persists, so a per-row sum multiplies the same stock by the window
  length). Each row carries `cogs_dropped` and `cogs_dropped_pct_of_raw`.
  Rows and money diverge and the divergence is the point: a stage taking a
  small share of rows and a large share of the exposure has changed what the
  surviving population represents. `cogs_dropped` goes NEGATIVE exactly once,
  at `contiguous_episodes_built`, because re-segmentation turns one opening
  row into two — the same stage where episode count rises.
- Credentials live in `~/.env` and reach the code as `REDSHIFT_*` environment
  variables. `.env` is gitignored. No hostname, credential or connection
  string goes in `config.yaml`, in a module, or in a commit — config.yaml is
  the source of every *tunable*, not of any secret.
- Synthetic validation: `tools/make_dummy_flc.py --policy randomized` makes
  elasticity recoverable (estimator should RECOVER it); `--policy legacy`
  reproduces the production confound (estimator should DETECT it).
- **Dirt is injected at the scope the defect really has.** Null category, null
  subcategory, zero base price and the multi-lot over-sell are ROW properties.
  A negative `flc_window` is a WINDOW property — an episode entering already
  negative — and is injected across whole windows. Writing it to random rows
  was wrong in a way that poisoned a diagnostic: `assign_episode_ids`
  differences the counter hour to hour, so one bad value read as a window
  BOUNDARY and shredded a clean window into fragments (3, 2, **−1**, 0). The
  fragments without the closing row came out `not_closed`, manufacturing 62 of
  the fixture's 65 unclosed episodes and driving
  `share_of_unclosed_explained_by_edge` to 0 — the number that is supposed to
  tell you whether the extract boundary explains the unclosed count was
  answering an injection artifact. Fixed: unclosed fell 3.9% → **0.2%**, and
  `negative_window_recovered` now recovers whole episodes as intended.
- **The generator models both source inventory conventions, and must keep
  doing so** — the write-off sentinel (`ending == 0` while stock remained) and
  shrink (`0 < ending < starting − sold`, rate `--shrink-rate`, default 0.02).
  It prints a count of each on every run and
  `test_the_generator_emits_both_source_inventory_conventions` asserts both
  are non-zero. This is not belt-and-braces: a fixture missing one does not
  fail, it PASSES — a stale `flc_synth.parquet` predating the write-off block
  carried zero sentinel rows and left the entire closure path unexercised for
  months while the suite stayed green. **Regenerate the fixture after any
  change to the conventions**, and read the two counts.
