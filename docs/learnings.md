# Learnings — what was tried, why it lost, what replaced it

The working code carries only the current design; superseded designs live
here so an agent does not re-propose them. One entry each: was → learned →
now. Dates are owner sign-off.

## Elasticity prior

- **Bracket method → profile density.** Two argmax estimates per category
  (naive/controlled), midpoint ± half-gap, and a constant fallback
  `−1.00 ± 0.60`. The argmax throws the curve's shape away (a flat
  likelihood reported a confident boundary estimate); the fallback constant
  was produced by nothing measured and once overwrote a measured bracket;
  on the first held-out comparison the bracket scored below a flat prior.
  Now: the whole deff-deflated profile likelihood is the prior as a density
  (`fit.prior_density`) — the 50/50 arm mixture reproduces the
  bracket in the sharp limit and degrades to the uniform where the data
  says nothing. No fallback constant, no std_floor; every run writes
  `holdout_comparison`.
- **Censored NB in the prior → censored Poisson (QMLE).** The NB likelihood
  needed `r`, and `fit_dispersion` needed an elasticity — a genuine ε ↔ r
  cycle, patched three ways in a week (the signal the design was wrong).
  The Poisson quasi-MLE is consistent for the mean whatever the true
  dispersion, so `r` leaves the ε step by theorem; prior runs first, then
  dispersion at the real per-category means.
- **Rows scored: entry → all stocked hours → entry again.** All hours
  bought price variation plus the survivorship confound (a deep-discount
  row exists because earlier hours did not sell; hour controls cannot reach
  selection on the unobserved demand shock) — wrong-signed categories went
  from 2/5 to 4/5. Entry-only is the rule (hard rule 7); the all-hours arm
  and the design_comparison sweep that scored the alternatives were
  removed once the comparison settled — a losing branch kept as a config
  key is a way for the confound to come back silently.
- **Hour control: pooled `hour_of_day` → same `date_hour` across sku×fc.**
  "Same-hour cross-episode" means the same hour *of the same day* — that
  absorbs weather, footfall, promotions. Cells under
  `min_rows_per_time_cell` fall back to 1.0 (a thin cell absorbs the price
  response itself). `date_hour` is the only control now; the pooled arm was
  removed with the 2×2.
- **Prior std: floor constant → zero-width bug → measured floors.**
  Removing the chosen `std_floor` produced `std: 0.0` on production (a
  delta-function density freezes the posterior). Curvature is *sampling*
  precision; the uncertain thing at production scale is the model. The std
  is now the widest of three measured floors — density width, grid
  resolution, `fold_spread` — with `std_basis` naming the binding one.
- **Wrong-sign handling: reject → accidentally lost → reject again.** The
  density method dropped the sign reject and production returned a
  confident −0.05. Restored: the unconstrained peak is searched past the
  bounds, a peak ≥ 0 discards the category's own density, and rejected
  categories are excluded from the pool they fall back to (or the fallback
  inherits the confound).
- **The −1.5 boundary defect.** An estimate pinned at the LOWER bound was
  read as "elasticity ≈ −1.5" when it meant the likelihood ran off the
  support. Asymmetric remedy: the lower bound may be widened when a fit
  pins there; the upper bound (`epsilon_max`) is a sign constraint, never
  widened.

- **Only the upper bound was searched past.** `unconstrained_argmax`
  widened the top of the grid to catch wrong signs and left the bottom at
  `epsilon_min`, so a likelihood monotone to −4 was read as "measured −4"
  and pooled. The search now extends below the bound too; a peak at or
  below it that strictly beats every interior point is a boundary, rejected
  to the pool and named in `lower_boundary_categories` (a flat curve whose
  argmax lands on the first grid point is not).
- **Deflation that could never engage.** The old episode grouping made
  `sizes >= 3` empty, so rho read 0 and deff exactly 1.0 for every
  category; the pooled shrinkage ran on undeflated spans until the grouping
  followed the recurring unit across days.

## Dispersion

- **Clamp everything high → under-dispersion exemption.** An `r` at the
  search ceiling has two causes wanting opposite treatment: a thin group's
  wandering MLE (clamp it) and a group genuinely steadier than Poisson —
  no NB can represent Pearson < 1 — where clamping claims variance the data
  does not have. Pearson dispersion separates them;
  `under_dispersed_groups` lists the exempt (a long list indicts the NB
  family).
- **Dispersion-first → prior-first.** Fitting r/ρ at a constant −1.0
  measured correlation against a curve nothing used; moving the working
  elasticity −1.0 → −1.5 moved ρ 0.31 → 0.42 and deff by 26% of the
  learning rate. `fit_dispersion` now reads the prior's means and records
  the basis.
- **`var(group means)/var(all)` → the ANOVA ICC.** The ratio estimates
  ρ + (1−ρ)/m, not ρ: on independent hours it returns 1/m (measured 0.164
  at m = 6), so deff deflated every posterior step by ~1.8× of pure
  estimator artefact. The frozen fit moved to the ICC first; two copies of
  the biased form survived in `drift_by_window` and the prior-density
  deflation, so the drift baseline and the frozen value disagreed by
  (1−ρ)/m until `common.config.intraclass_correlation` became the one home.

- **Pinned `r` fed the clamp.** An `r` at the search ceiling was stored
  unflagged and counted in the clamp percentile, so a thin extract with
  ≥ 10% of groups at the ceiling set `cap = bounds[1]` — no clamp at all.
  Pinned groups are flagged (`at_bound`) and excluded from the percentile.
- **The drift baseline ran the biased ICC.** `drift_by_window` kept the
  `var(means)/var(all)` form after the frozen fit moved to the ANOVA ICC,
  so baseline and frozen value disagreed by (1−ρ)/m with no drift at all.
- **One group fit, spelled twice.** The frozen fit and the drift
  measurement each fitted a group's `r`, read its Pearson dispersion and
  tested the bound in their own three lines, and the rho-on-thick-episodes
  block was a literal copy. `_fit_group` and `_episode_rho` are the one
  reading of each; the cluster statistics themselves (the ANOVA ICC, the
  design effect, `m` per batch) left the config loader for
  `common.clustering`, which reads no config.

## Population and data quality

- **Closure heuristics → the write-off sentinel.** Counter-based and
  tolerance-based "did it end?" rules collapsed to one source-native rule:
  `ending_inventory == 0` on the last row, full stop; the unclipped sign of
  `starting − sold` there distinguishes censored/scrap/restock. The
  sentinel once had a fallback (absent everywhere → treat all closed): it
  failed in the invisible direction — a fixture that never modelled the
  convention looked healthy for months. No fallback; a sentinel-free feed
  reads every episode unclosed, loudly.
- **Drops → flags** (restocked, below-cost, edge-truncated,
  negative-window). Each was once a hard drop; each cost the artifacts
  population they needed (restocked alone: 18.1pp of COGS) or answered a
  gate by deleting its subject. All are flags now; consumers exclude only
  what they specifically cannot use.
- **Hour-level quantity tests were wrong twice.** "units > inventory is
  impossible" deleted restocks; "ending falls short is dirty" deleted
  shrink, fastest sellers first. Continuity is the only hour-level rule
  that drops.
- **Gross, never netted.** An early `episode_flow` netted a shortfall
  against a same-size restock — inference dressed as arithmetic; it let a
  restocked episode price its clearance against the wrong supply.
- **Row-scoped drops manufacture chain breaks.** Row-scoped null-category /
  zero-price drops punched holes mid-window and re-segmentation split
  episodes into fragments. All post-id drops are episode-scoped;
  re-segmentation is a checked no-op (an assertion, because the invariant
  fails silently).
- **cogs_at_risk: opening stock → supply** (opening + gross arrivals) —
  opening stock understated every restocked episode's exposure.
- **Timedelta overflow.** "still running at the extract edge" compared
  `ts + to_timedelta(hours_remaining)`, which wraps silently on
  million-hour counters; compare in numeric hours bounded by the extract's
  own span.

- **Row-scoped drops at the row-defect stages.** The null-key and
  duplicate-hour drops were row-scoped; with the defect on a window's
  FIRST hour the rest re-id'd as a clean window opening one hour late,
  eligible, dp-eligible and scored as an entry row (the gap check only
  sees interior holes). A defective row whose key is whole drops its
  whole source window (`defective_windows`); a null quantity is the same
  drop instead of a cast error before the first waterfall row. And the
  null-run reading disagreed with the ids on a flat or −2 counter step —
  it swallowed the neighbouring window; `window_signals.counter_ok` is
  the one counter clause both read, the run drop keeping only the two
  tolerances a defective row needs.
- **An episode that opened empty.** The feed can resume rows after a
  sell-out; a fragment opening with `q0 = 0` was dp-eligible, the replay
  could not open it and shadow priced it. It is a flag (`opens_empty`),
  and `closed_then_resumed` keeps the producer's open question measurable.
- **The identity was recorded, not asserted.** The design said the stock
  invariant was asserted on the output; a broken `episode_flow` wrote
  `holds: False` into the manifest and the run succeeded. The chain
  asserts it once continuity holds; a bare frame only records it.
- **A fixture that never crossed midnight.** Every §12a seam path ran on
  data where the seam could not occur, and 86% of the fixture's final
  rows carried counter 0 while production's are positive on essentially
  all: the generator now opens a share of windows in the evening and
  closes most listings with hours left, and prints both counts.
- **A mask built before the merge.** `add_ref_rate_features` built its
  anchor mask on the incoming frame's labels, merged (which resets the
  index), then reused the mask — pandas aligned by label, and on the gappy
  frames `load_and_filter` always hands over, `prior_episode_ref_sales_rate`
  was wrong on about a third of rows. The baseline trained on it. The mask
  is recomputed after the merge; a test pins gappy = contiguous.
- **`units_gt_inventory_dropped`** once ran in the filter chain and deleted
  every restock (18.1pp of COGS); a NULL cost once sailed through and handed
  the DP a NaN `d_max`; `population()` refuses an unknown name rather than
  falling back; the waterfall's `raw` row once mixed a pre-dedup row count
  with post-dedup episodes and COGS.
- **The window rule in two packages.** The ids and the defective-window
  drops lived in `fit.prepare_data`, the episode-scoped cuts in
  `common.episodes`, and a reader asking "where does an episode start and
  which rows may a fit read" walked both. `common.windows` is the one home
  of both halves (the rule, the cuts, the week keys, the planning horizon);
  `episodes` keeps the inventory accounting and gained `episode_cogs`, the
  exposure read off the flow it already computes. The flag stage split the
  same way: `dp_flags` sets the columns, `flag_detail` writes the stage's
  dict in the manifest's order. And the ten spellings of "the eligible
  population of one split window" are `prepare_data.scope`.

## Calibration

- **The fixture generator's own window rule.** `tools.make_dummy_flc`
  counted its windows, seams and restock extensions on the raw schema with
  a rule of its own that omitted the counter clause, spelled the counter
  by hand and kept a copy of the reference discounts, so the tool's
  printed counts and the chain's ids were two readings of one fixture.
  It now renames to the chain's names and reads `common.windows`
  (`assign_episode_ids`, `window_signals`, `window_counter`) and
  `reference_discount(cfg, ...)`; the emitted bytes are unchanged and only
  the counts that were read off the second rule moved.
- **Blocking gate → always applied, level as a diagnostic** (owner,
  08-25). The feared mask cannot happen: factors are fit on anchor rows
  only, where the price term is 1, so slope error never enters; the band is
  a reported diagnostic (WARN), and the daily
  `realised_vs_predicted_sold_ratio` is the continuous guard.
- **Censored basis.** Factors were once fit against raw `mu` (always the
  larger number), so they read systematically low — a true 1.45 correction
  fit as 0.68, the wrong side of 1. Factors are solved by bisection against
  `E[min(D, q)]`, the gate's own quantity.
- **Fixture behaviour mistaken for a rule, twice.** "The loop settles in
  two turns" and a stall test that stopped on two flat readings were both
  sized on the small fixture; production settles in 8–9 turns. Caps and
  impatience thresholds are sized for production, and the stall test needs
  three turns with no new best.

- **Boundary factors returned as estimates.** `_solve_level_factors`
  returned the literal bracket bound silently when the bisection failed to
  bracket; cells now carry `at_bound`. The payload said thin cells were
  "left at 1.0" while the code shrank them toward the parent; and
  `convergence.method` said "dry run" under `--commit-convergence`.
- **A cell with no anchor rows priced at 1.0.** The level solve emitted
  only the cells that had anchor rows, so a subcategory absent from a
  window (thin, or new assortment) priced at raw mu while its category
  solved to 1.4 — one anchor row with no sales shrank to the parent, none
  was silently 1.0. Every cell of the window's population is emitted
  (`held_at_parent`), the parent tables ship in the artifact and the
  applier waterfalls subcategory → category → 1.0.
- **Fit by opening week, applied by row week.** The trailing windows cut
  whole episodes by opening week; the applier read the row's week, so the
  Monday rows of a Sunday-opened episode were both in week w's fit window
  and priced by week w's table. The applier keys on the episode's opening
  week (the live forecast rows, with no episode, read their own date).
- **A truncated censored expectation.** `E[min(D, q)]` folded the NB tail
  at `negbin_max_k`, so a shelf deeper than the table was read as
  `E[min(D, min(q, 25))]`: the sold/predicted ratio pushed above 1 and the
  factor absorbed a truncation artefact as under-prediction; the DP had
  the same cap and undervalued clearing a large shelf. The table runs to
  the shelf, the expectation has a closed form beyond it, and the tail
  mass is a diagnostic of demand beyond the shelf, not a truncation.
- **A row-level date cut on the unfiltered frame** once put ineligible rows
  into the level fit, so the gate and the fit solved on different rows.
- **A failing 5b iterated to `--max-turns`.** When `--check-convergence`
  itself failed, the artifact carried no `convergence` block, the stall test
  had nothing to compare and prior/dispersion re-ran twenty times; a failing
  5b now stops the loop with its own message.
- **The level-factor fit inside the model module.** `train_baseline` was
  a quarter model and three quarters calibration, and the fit reached its
  own `r` through a function-local import to dodge the cycle it created.
  `fit.calibrate` now holds the one estimator (`solve_level_factors`), the
  fit basis, the weekly schedule, the convergence check and the console
  summary; `train_baseline` keeps the model, the applier and the two CLI
  flags, which call in. The stamp still names the command. The fit basis
  is one function, `attach_fit_basis(frame, model, r_lookup, raw=)`: a
  level solve reads the raw mu, the harnesses the calibrated one, and both
  read `r` through the vectorised lookup -- the replay once looped a Python
  lookup per row beside it.

## Exploration budget and tau

- **Budget base: same-day / window-mean IL → trailing close-day IL.** An
  episode's IL is settled only at close, and today's tau must be computable
  at midnight from history alone. A day's realised IL is the whole-episode
  IL of episodes that CLOSED that day; the budget is a share of the
  trailing 7-day mean.
- **tau clip: symmetric [0.5, 2.0] → asymmetric [0.5, 1.25].** With a
  trailing base the budget barely moves; cutting is the safety direction
  (a measured 8.7× overspend needs three halvings to get inside the 2×
  stop), raising is never urgent.
- **Every tier is worth exploring → δ_min.** Shadow spent ~22% of
  decisions exploring, almost all one tier step (2.5pp) from the optimum:
  cheap, so the budget bought many of them, and worthless, because a move
  whose signal ε·L sits inside the model's own level error teaches nothing
  about ε. A second knob (an exploration probability) was rejected — τ
  stays the one controller — and the floor is DERIVED: the bias scale is
  measured by `tune` from the backtest, ε is the cell's posterior mean,
  and the ledger prices τ against admissible tiers so the budget still
  funds exactly the draws made. The first cut measured the floor from p*
  and changed nothing on the owner's shadow (forced rate and the mean gap
  from the reference identical): cost is measured from p*, but the learner
  reads every outcome against `mu_ref` at the REFERENCE discount, so the
  informative distance is from the reference. The floor is on that. The
  bias scale was then made per category: one catalogue scalar
  under-floored the categories with the worst surviving level error and
  over-floored the best, and `tune` already had every category's own
  reading in `by_category`.
- **Entry-only spread collection.** The replay collected Q-spreads at entry
  only, funding ~1 exploration per episode against a system that explores
  every hour — its own bisection reported 1.00× regardless (a number a
  procedure solves for is not evidence about that number). Spreads are
  collected at every decision hour; shadow derives the launch tau on its
  own anchored path.
- **Poisson information under an NB likelihood.** `daily.update`
  accumulated `μ·L²` while the likelihood is NB (information
  `μ·L²·r/(r+μ)`), overstating evidence ~1.6–1.9× on top of what deff
  corrects. Fixed by theorem on both paths.
- **Scrap through a local copy.** An inline scrap rule in the budget base
  dropped ALL scrap on a sentinel-free feed, understated the budget 10× and
  flipped the verdict to WOULD SUSPEND. Five hand-synced copies of the
  episode groupby (business metrics, live guardrail, the two noise floors,
  `il_pct`) were then found kept equal by comments; all of them, and
  shadow's budget base, now read `common.metrics.episode_economics` over
  `episodes.scrap_units`.
- **Day key from the outcome's finalize time → the decision's trading
  day.** An hour-23 decision finalizes at D+1T00:00Z; keying spend on that
  put the controller a day ahead of the IL side, graded ONE hour of spend
  against a full day's budget, ratcheted tau up 25%/day, and
  `tau_calibrated_through` then guaranteed the other 23 hours were never
  priced. `events.pairs.decision_day` keys both sides.
- **Zero spend held tau still → raises it.** "No exploration is an absence
  of signal" was wrong on a priced day: nothing was affordable, which is
  the under-spend design 5.8 raises tau on, and the only way a tau cut
  below the smallest spread ever recovers. Shadow's trace already walked
  that rule; production held still and the two disagreed on one log.
- **Every digest change staled every report.** `advance` treated any
  config change as invalidating every report, so pasting `tau_initial`
  (what shadow itself derived) staled shadow, shadow re-ran for hours,
  derived a tau a few percent different, which pasted, which staled
  shadow — a day on the owner's extract with no readiness report. Staleness
  is now judged on the keys a report READS (`tune.rerun_for`: W turns the
  loop, `delta_min` re-runs shadow, a stop threshold re-derives
  thresholds, unclassified edits re-grade everything, MEASURED write-backs
  nothing), and `advance` refuses to run the same step a third time in one
  invocation. Its first fix judged staleness by the STRONGEST class among
  the moved keys, and the classes do not nest: the delta_min paste (shadow)
  swallowed the stop-threshold paste (thresholds) made in the same `--apply`,
  so thresholds was never re-derived while `status` — which had its own,
  looser rule — still flagged it, and the backtest with it, twice over.
  One routing (`tune.stale_keys`, per key, union) now serves both readers.
  The second half of the same loop: the rho paste tolerance was
  5e-4 while each `--check-only` turn still contracts rho by ~1e-3, so every
  settle was a new paste; the tolerance is now the config's
  `rho_paste_tolerance_rel` (1% of the frozen rho; tau's is 5% because tau
  self-corrects daily and rho is frozen for the pilot). And shadow ran
  single-threaded for an hour per pass; `advance` and `ops.bootstrap_loop` now
  pass `--workers 0` (reports are byte-identical serial or parallel).
- **Weekly learning gate, tried and reverted.** A weekly `--apply` was
  considered to lighten the daily chore. It buys nothing: the trigger is
  per cell, so under a daily gate a fast category updates the day its
  batch reaches `information_increment` and a slow one simply waits; a
  weekly gate delays the fast ones and, with per-update rails, discards
  their surplus. What the exercise did fix stays: tau moves on spend and
  needs no operator (`--calibrate-tau`, committed daily), and it walks
  every closed day since its last calibration (`explore.walk_tau`, shared
  with shadow's trace) -- before that a missed day was skipped, not
  graded. `learning.update_cadence_days` stays as the knob, at 1.
- **Backtest tau on its own budget rule → production's.** The backtest
  solved against the bare `budget_share_of_il` share, collected Q-spreads
  without `engine.decide`'s explorability gate, and counted `n_days` as
  days-with-decisions while shadow used the calendar span — three ways for
  its cross-check to disagree with the value it checks. All three now
  share production's definitions.

- **The seed on the wrong population.** Shadow's pre-window IL seed ran
  `episode_economics` on the full frame while `frac` and `seed_scale` were
  dp_eligible counts, inflating the day-one budget and the derived tau by
  the ineligible episodes' IL. And `daily_budget` averaged over the seed
  days too, the first of which has no trailing history and a budget of
  exactly zero. Both now read the dp_eligible window's decision days.
- **Three `n_days`.** The window's span was computed on the extended frame
  (a next-day row with no decisions), the pre-window's after sampling (a
  sample shrinks the span), and the backtest's IL mean over days with
  episodes while its spend divided by a calendar span crossing the
  exclusion gap. One count each, on the unsampled, unextended frame.
- **The τ walk after the commit.** `--apply` committed the cells first, so
  the walk priced past days' budgets on the post-update std and disagreed
  with the dry run on the same store; the std is snapshotted before the
  commit loop.
- **Keying spend on `finalized_at`** once put the controller a day ahead:
  one hour of spend graded against a full day's budget ratcheted τ +25%
  a day and `tau_calibrated_through` skipped the other 23 hours. Spend is
  keyed by the decision's trading day.
- **Entry-only spread collection** funded ~1 exploration per episode
  against a system that explores every hour (~8× under); every decision
  hour is recorded, tau-independent, before the draw.
- **The budget pinned at the launch std** by an unrouted GLOBAL cell: max
  over all cells never moved, `budget_scale_floor` was unreachable and the
  flat-std alert listed GLOBAL forever. Both read the routed cells.
- **Poisson information overstated NB evidence** ~1.6–1.9×; `k >= inv`
  censoring marked every restock hour censored. The update reads
  `mu·L²·r/(r+mu)` and the shared censoring rule.
- **A censored row credited with a count's information.** A stocked-out
  row was observed only as the event `D ≥ q`, and the update credited it
  the full NB figure: a batch of sell-outs crossed the increment before
  the evidence justified it. A censored row carries the Bernoulli
  information of the event it observed, strictly less.
- **Suspended days walked as under-spend.** While a stop kept exploration
  suspended, every morning's walk multiplied τ by the clip on the day's
  zero spend; the resume overspent at once and re-fired the stop. A
  suspended day (every decision priced with no budget in force) is held.
- **A failed push spent its expected cost.** The spend side summed
  `exploration_cost` over every matched forced decision, reported failures
  included: an integration incident read as an overspend. Only executed
  pushes spend.
- **The walk graded days after the stamp.** `tau_calibrated_through` was
  the only record of what was walked, so a day whose outcomes arrived
  after a later day was walked sat behind the stamp forever. The commit
  keeps the walked days; a late day is walked next time and counted (its
  step is τ-independent, so it lands the walk where a timely one would).
- **Two writers, whole-state writes.** `--apply` and the monitor each
  wrote the state they had loaded; a suspension written between the
  other's load and write was gone. Every write re-reads the file under a
  lock and applies its change to what is on disk.
- **History re-priced at today's std.** The overspend series priced every
  past day's budget at today's widest std; once the posterior narrowed,
  history read over a budget it never had. Each day takes the std its own
  decisions were priced with.
- **A stale schedule refused the walk.** The factor-schedule gate exists
  so evidence is not banked on stale prices; it also refused
  `--calibrate-tau`, leaving overspend uncorrected while the cron was
  down. It refuses `--apply` alone.

## Evaluation

- **Judging estimators by their outputs → held-out comparison.** "Is
  −1.61 ± 1.09 better than −1.00 ± 0.60" has no answer by inspection. Every
  prior run scores candidates on unseen data, bracketed by `oracle` and
  `uniform`, leading with `information_available_per_row` — a method gap
  that is a large share of a tiny number is still tiny.
- **Round only for display.** A 4dp-rounded mean against an unrounded one
  made a reported step read over `max_mean_step`; artifacts carry unrounded
  values.
- **Per-run counters live on the store; files accumulate.** Reading
  `quarantined_event_count` from the cumulative file made serial and
  parallel runs "disagree".
- **Two gates on one expression.** Shadow's `matched_decision_rate` was
  `event_completeness` under a second name and threshold; a gate that
  cannot disagree with another is not a second check. Dropped.
- **Rounded for reading, compared for real.** The monitor rounded the
  price-mismatch rate to 4 dp before the stop condition compared it, while
  the operator gate compared unrounded — the same log, two answers at the
  boundary. Stops compare exact counts.
- **Source-pinned tests.** Dozens of `inspect.getsource` substring
  assertions failed on identical behaviour whenever a helper was
  extracted, and blocked the consolidation they were guarding against.
  Where a behaviour exists it is tested by calling the function; a source
  assertion is kept only for an architecture ban no behaviour can express.

- **The A/B module, removed (owner, 2026-09-05).** Hash-assigned arms,
  the empirical MDE-by-duration table, the control-arm guardrail basis and
  `ab_test.active` all existed for a randomised readout the pilot will not
  run: the system prices every episode engineering supplies, sampled across
  FCs and categories, and is read pre/post on the same units (design §11).
  The control-arm basis had also been structurally inert before any A/B
  (both hash-labelled halves were system-priced, so a catalogue-wide
  deterioration cancelled to exactly zero); the trailing-mean basis is the
  one the floors are measured on and the only one left. Exploration's
  evidence is unaffected: the forced moves are randomised within the pilot.

- **The strongest re-run class swallowed the weaker one**, and a
  per-category paste diffed one key per category that `rerun_for` did not
  recognise, so a floor re-round routed to `calibration` and tripped the
  loop guard; the round budget then counted plans instead of work and raised
  on a legitimate ninth-round stop before writing the journal. Routing is
  per key on the longest `KEYS` prefix (`READ_BY` for keys tune does not
  paste), the union over keys, and only run steps count toward the budget.
- **Stop conditions that suspended nothing.** The monitor wrote
  `suspend_exploration` and nothing read it; `decide` took τ from the
  caller. The monitor now writes the suspension into the posterior state,
  `decide` prices with no budget while it is set, and only
  `update --resume-exploration` lifts it.
- **Survivorship on the newest days.** Daily scrap/margin rates were keyed
  by OPENING day over settled episodes, so the days the persistence rule
  evaluates counted only the early sell-outs. Both floor and trigger key by
  close day.
- **Gates that could not fire.** `duplicate_counts` moved only on emit
  (update and monitor only load); the reproduction check counted an
  exception as a mismatch but not as checked, so a broken solver read
  INSUFFICIENT; `artifact mirrors` read PASS with no artifact on disk;
  `config mirrors reports` read PASS when no report produced a finding;
  `verify()` compared hashes only for files present, so a deleted sealed
  artifact passed. Each now reads the honest verdict.
- **The ingest gap from day two.** `decisions_without_feed_row` counted
  every stored decision absent from THIS feed — everything already ingested
  and everything not yet due. It counts inside the feed's date range only.
- **Assurance's own histories.** Judging correlation drift on rho alone
  was blind to the m channel (m was a frozen paste of legacy episode
  length); a p-value-only uniformity test tightened every day on an
  append-only store, so the same draw passed in week one and failed at
  volume. `int(nan)` once aborted a whole daily ingest before the store saw
  a row; a zero base price once booked the whole list price as IL.
- **Hand-written ledger notes.** The shadow and replay reports carried
  paragraphs of reading guidance (`tau_recommended` is a cross-check not a
  correction; the controller trace looks jumpier on a sample and
  `spend_over_budget` is the sample-invariant figure to quote; the paired
  calibration comparison removes between-week variance; rolling-origin
  windows read the trend). They live in §5.13–5.14 now; the reports point
  there.

- **Launch at the prior's best guess → launch half a std steeper (owner,
  2026-09-05).** Under the cold prior the DP was enter-and-hold on most
  shelves and the owner would not carry the clearance loss into the pilot.
  Rather than a new mechanism, the existing init step pushes each cell's
  launch mean `cold_start_shift_std` prior stds toward more elastic (0.5;
  std untouched, so evidence weighs the same and the bounded step walks it
  back), the backtest prices its DP arm at that belief against a prior-mean
  world so the launch record grades the policy that will run, and
  `epsilon_min` was widened −4 → −5 under rule 3. k = 1 was judged too
  aggressive.

- **Packages by phase → packages by responsibility (2026-09-05).**
  `pipeline/` held the hourly production lane, the pre-launch harness and
  the operator tooling side by side; `bootstrap/` held the fits and the
  driver that runs them; `backtest/` was the only package run as
  `python -m backtest`. Now `engine/` prices and learns, `fit/` builds the
  artifacts, `evaluate/` grades them, `daily/` is the production lane in
  run order, `ops/` drives and gates. One package per REVIEW_GUIDE tier
  and `advance` phase; the tests mirror the modules.

- **`config_version` label → the environment sealed (2026-09-06).** The
  seal hashed six artifacts and nothing else. A runtime-only key (budget
  share, δ_min multiple, a stop threshold) took effect on the next hour with
  no record beyond a `config_version` string nobody bumps; a LightGBM
  upgrade moved predictions with every artifact byte intact. The seal now
  records the config (digest + snapshot), the library versions and the
  posterior as it stands; `verify` reads a move as a problem on the bundle
  row; `advance` re-seals under `config` / `libraries`; every decision
  event carries `config_digest`. Out on purpose: the code (the owner's
  call — a deploy is the repository's own history, and reproduction
  catches a solver that moved), the extract (too large to hash per seal;
  the split manifest is its provenance) and the event store (the record).
- **The second review after the move (2026-09-06).** Re-reading every
  package with the new layout found defects the first review's structure
  had hidden, most of them "two homes for one fact":
  - the overspend stop read its streak from the days with *spend*, so a
    pilot resumed after a suspension was re-suspended by the next `--feed`
    (the suspended days had no reading and the over-budget days stayed
    "latest"); every priced day now reads, 0 where nothing was forced;
  - a duplicated event line was *counted* on load and still *loaded*
    twice; a failed-push table with a datetime `date` matched nothing and
    every failed push was learned from; the quarantine file's torn last
    line was never closed; the SQL exclusion cut episodes at the window's
    edges and fed their remnants to the prior as entry rows (the interior
    only is skipped now — step 1 drops the straddlers whole);
  - `advance --retrain` could not finish: the retrained backtest and the
    old shadow tripped tune's one-model BLOCK before the step that re-runs
    shadow; and a launch-belief re-init after launch would have erased the
    τ walk and a standing suspension (production state of any kind ends
    `launch_stale` now);
  - routing: unknown keys fell to `calibration`, so `data.split` and the
    LightGBM keys re-fit factors against the OLD model instead of stopping
    for a retrain; `tuning.` and `assurance.` were "inert" while three
    reports and two fits read them; a status row went "not run" before its
    drift check and hid a stale paste;
  - the prior's wide lattice and fit grid shared no points, so the "one
    grid step" boundary tolerance was dead; `fold_spread` and
    `drift_by_window` cut by row date (rule 15); the deff mixed rho over
    recurring units with `m` over all; the global level factor's
    `at_bound` was discarded; snapshot folders were keyed to the second and
    ordered by name; shadow's "deeper than legacy" share counted shallower
    hours; its stop streak ignored calendar gaps the monitor honours; the
    guardrail floor rolled across the exclusion gap on row order.
  Dead code removed: the level-mix decomposition and its config key, the
  prediction-basis rows, the second `__main__` guard, `information_pending`,
  the `_dp_arm` re-solve, the `latest_priced_day`/`daily_exploration_spend`
  wrappers. Hard-coded `sample=300` and `max_days=60` became `tuning.` keys.
- **The third review (2026-09-06), after the seal.** Mostly windows and
  bases that had drifted between two readers of one series:
  - the event-quality gates were all-time rates on an append-only store,
    so one incident kept a resumed pilot suspended forever; they now read
    the trailing `event_quality_window_days`, from one home
    (`events.pairs.quality_counts`), and the monitor names the window;
  - the guardrail floor and the trigger each smoothed the deterioration
    series their own way (the floor scored one reading on 41 days and
    called it a floor; a NaN floor read OK); one series now
    (`common.guardrail.deterioration_series`), and the floor needs
    `guardrail_noise_min_extra_days` SCORED readings or reads
    `insufficient history`;
  - the calibration-window sweep keyed rows by row date, bridged the
    exclusion gap with the last W fitted weeks and scored weeks the
    baseline was trained on; it now keys by opening week, windows by
    calendar and scores after `train_end` — the schedule production runs;
  - shadow's `n_days` spanned row dates, so a 22:00 opener bought an
    extra day of budget; `calendar_days(opening_dates)` in both harnesses;
  - `scrap_rate` divided by the opening count while the flow identity
    divides by supply; a restocked episode read scrap above 1;
  - zero-stock and restocked hours were learned from (no demand at any
    price; a restocked count is not a censored draw) —
    `learnable_with_stock`;
  - the uniformity check re-solved every forced decision ever logged
    (unbounded at volume) and `reproduction` re-solved them again;
    capped at `assurance.uniformity_sample`, one re-solve shared;
  - shadow's report carried no posterior digest, so a re-init after a
    retrain left a ghost shadow `advance` could not tell from a current
    one; a failed-push row matching no decision vanished; an orphan outcome
    was exported under a guessed date; `decide` accepted a non-integer
    hour and cast `current_discount` before validating it.
  Renamed for what they are: `deff_applied_all_time`,
  `updates_to_min_std_median`, `range_across_categories`.
- **The first weeks after launch had never been run (2026-09-06).** Every
  production piece was unit-tested and shadow rehearsed the decision path
  on history, but nothing ran the hourly engine, the ingester, the tau
  walk, the monitor, assurance, the weekly re-fit and `--apply` together
  on a shop that answers back. `evaluate.pilot_sim` (design §11.3) does,
  against a demand world built on the frozen model with an assumed
  elasticity, and its first two runs found what the tests could not:
  - a week the weekly re-fit judged too thin is held at the frozen anchor
    on purpose (`weeks_unfitted_held_at_1`), but the `--apply` gate and
    `advance`'s re-fit trigger read `by_week` alone, so the held week
    looked like a missed cron: every `--apply` of that week was refused
    and `advance` re-fit every morning. One reading now
    (`train_baseline.schedule_reaches`); the gate says `held_at_anchor`;
  - the uniformity check mapped the applied tier's rank to `(rank +
    0.5) / n`, so a two-tier set only ever landed on 0.25 and 0.75 and an
    honest uniform chooser FAILed once small sets dominated (p = 0 on
    578 draws, on the fixture run). Rank plus a jitter from the decision id is exactly U(0, 1)
    at every set size.
  - a push engineering REPORTED as failed was counted as a price
    mismatch, so a fifth of pushes failing — every one reported — refused
    every `--apply` and suspended exploration; the gate now counts the
    reported ones apart (`push_failures_reported`) and catches only the
    silent ones, as the contract always said.
  - on the owner's extract the simulator's templates carried a NaN window
    length: rows with a null `flc_window` reached the DP-eligible
    population as one-row episodes, because `assign_episode_ids` reads a
    NaN counter as a new window (NaN ≠ −1) and the negative-window flag
    reads NaN < 0 as False. The run holding a null counter now drops
    whole at `null_key_rows_dropped` (a row drop left a fragment opening
    mid-window); the templates refuse a null rather than skip it, and the
    fixture injects the dirt so the path stays exercised.
  - on the owner's rehearsal the overspend stop fired on day three and
    suspended exploration for the run: the budget is a share of trailing
    closed-episode IL, and two days after launch that base is a handful
    of early closers. A base shorter than `budget_il_window_days` is now
    an absence of signal like a zero budget (`explore.budget_base_ready`):
    the controller holds, the stop takes no reading. The same run showed
    the simulator overlapping a template's two arms on multi-day windows
    (two feed states for one hour, a completeness gap with no fault); the
    twin now opens the day after the first run closes.
  - the learner's sensitivity to the level: it reads every forced outcome
    against the agent's `mu_ref` with no level term, so a level error `e`
    identified from moves of mean log ratio `L` is an elasticity error of
    about `e / L`. On the fixture a thin week's re-fit moved the level by
    8% (inside the 10% gate band) against forced moves near −0.15 and
    the posterior walked from the truth it had reached (−1.2) to −2.2
    with a std of 0.39. The simulator's `agent_level_tracks_world` grades
    the implied bias against the posterior's std, not the band alone.
  Also learned: the sim's `hold the current price` fallback opened an
  episode at a bare `d_max` off the tier grid, and the next decision had
  no feasible tier at or below its anchor — the same trap Lane B's
  fallback must avoid (open at a tier). And a dynamic worth the owner's
  eye (design §11.3): a posterior mean stepping toward zero inflates
  `delta_min` until few action sets hold an admissible tier, exploration
  starves with no stop fired, and tau climbs by the clip every zero-spend
  day — `exploration_never_starves` grades it.

- **The fourth review (2026-09-06), after the simulator.** Five readers
  over the simulator, the controller/stop/assurance changes, the chain,
  the docs and the tests. What they found, structural throughout:
  - the budget's two readers disagreed on the denominator: the mean IL
    divided by the days since the first close INSIDE the window, readiness
    judged the earliest close anywhere, so a no-close day at the window's
    leading edge inflated the budget by window/span; one span now
    (`explore.budget_held`, the composite the controller, the stop, shadow's
    mean budget and its day-one derivation all read);
  - a reported push failure was counted in the pairs "compared", diluting
    the silent-mismatch rate by the failure rate; the rate is over the
    pushes judged, the window's denominator keeps the reported ones;
  - a week whose trailing window was EMPTY (data start, a gap) was in
    neither `by_week` nor the held list, so the gate read it as a missed
    cron by the other branch; `calibration_coverage` still read `by_week`
    alone and called a held trailing week STALE while the gate passed it;
  - the null-counter run was clock-only: a feed gap the counter ran down
    across left its far fragment, and a back-to-back window fell with its
    neighbour; the run now reads the ids' own signals and counts windows;
  - a refreshed extract after launch moved the split manifest alone and
    stopped `advance` on a red bundle row with no step to absorb it; the
    weekly re-fit + re-seal now does. `event_quality_window_days` routed to
    `calibration` and turned the loop on a paste; inert now;
  - an id-less line the store admitted crashed every consumer that indexes
    by id; an empty feed graded as a day with no gaps; one bad failures
    row aborted the batch; three copies of "one week past the latest data
    week" (`episodes.week_after` now);
  - the simulator graded the agent's level against the undrifted world (a
    tracking re-fit FAILED under drift), printed the implied elasticity
    bias with the wrong sign, held every feed row of the run in memory,
    and read a template's economics as paired when its twin never ran.

- **The integration surface (2026-09-14).** Lane B had a function
  (`engine.decide`) and a worked example inside the simulator, but no
  entry point that took the contract's payload; engineering asked for one
  they could call with a batch and get outcomes back from. `ops.price_batch`
  is that caller and `engine.state.build_states` the one request → state
  (lifted out of `pilot_world`, which now reads it — a second feature path
  was the risk); `tools.e2e_cycle` runs the whole loop in a workspace.
  Two things the loop exposed: `outcome_id = "feed-<decision_id>"` could
  not be named by engineering from their own table, and nothing refused
  two decisions on one hour — a retried batch paired both with the one
  feed row and the duplicate gate saw nothing (distinct ids). The id is
  now the hour's key (`events.pairs.outcome_id_of`), `price_batch` refuses
  an hour already priced, and ingest matches neither of two decisions that
  claim one hour (`decisions_colliding_on_hour`).

- **Restock-extended windows (2026-09-14).** The id rule read any upward
  counter step as a new window, and the contract carried it as a known
  limitation to retire when engineering's id landed. Engineering settled
  the two facts the derivation needed instead: the counter moves UP from
  the hour AFTER stock arrives (the restocked hour still counts down), and
  a zeroed `ending_inventory` is the close whatever the counter does next
  — even on an hour that restocked, and even mid-window (that zero is a
  write-off leftover, not shrink, which reverses the earlier shrink
  reading for those rows). The rule now has one home
  (`prepare_data.window_starts`) for the ids and the null-counter run;
  three copies of the boundary had drifted. Two things the change
  exposed: the replay and shadow planned every hour over the rows the
  episode turned out to have (`mu_ref_path[t:]`), so a merged episode's
  early hours saw an extension production could not — the horizon is the
  row's own counter now; and the synthetic fixture never emitted the
  pattern, so the clause had no end-to-end exercise until the generator
  learned to extend windows (a third printed count).

- **Readiness counted twice (2026-09-14).** The simulator judged when a
  scrap stop MUST have fired from `window + smoothing + persistence`, but
  the deterioration series it grades yields its first reading only after
  `window + 2·smoothing − 1` close days (the trailing mean is shifted by
  the smoothing so the two windows never overlap): a correct machine was
  graded a silent stop for the smoothing's worth of mornings. The count now lives beside the series
  (`common.guardrail.first_reading_close_days`, `stop_ready_close_days`)
  and both the monitor's short-series note and the grader read it. The
  same pass moved other second readings onto their homes: the sim's bias
  bound onto `widest_active_std` (every cell in the file once counted, so
  an unrouted GLOBAL excused any bias); shadow's persistence streak onto
  `monitor.evaluate_guardrail`; the `counter + 1` horizon onto
  `episodes.planning_horizon` (four spellings); the deck's `delta_min` and
  admissible set out of the browser into `engine.explore`; the replay's
  sample size out of a flag into `tuning.backtest_policy_episodes`; and
  the integration cycle's shop loop onto the simulator's (`PilotSim` takes
  a pricer; `e2e_cycle` is a driver) — a second shop had its own
  rejection sentinel. Two silent paths surfaced beside them: shadow
  dropped a zero-stock restock-gap hour before recording it (the episode
  never settled, its decisions stayed in the ledger) and crashed on a
  window whose every decision day was held; the sim's completeness never
  read the store's outcome-side quarantine and excused a refused `--apply`
  under any fault.

- **A request in the producer's dtypes → one spelling first.** The batch
  caller priced and COMMITTED a parquet request table, then died writing
  the response (a timestamp is not JSON), and the retry was refused as
  already priced; a JSONL `"7"` against an int history merged nothing and
  priced every request on "unknown" features, uncounted; a null cost and
  an integer category each took the batch down inside the worker. Now
  every request is canonicalised once (`engine.state.canonical_request`,
  ids through the hour key's `events.pairs.ident`), the history's ids are
  read the same way, the response goes through the one NaN-safe row
  writer, a null price is one rejected row, and a batch priced on no
  history says so (`requests_with_unknown_features`).
- **Mid-episode features as of the request day → the entry forecast,
  sliced.** A later hour of an episode recomputed its two demand-rate
  features as of ITS day (rule-12 skew) and its `episode_id` never met the
  history's derived ids anyway. The forecast is made once, at entry; the
  store keeps each episode's latest path (`episode_paths`) and a later
  request is that path sliced to the hour — extended only when a restock
  grew the window, on the opening's features — so serving equals what was
  forecast, assurance re-solves it, and no history pass runs.
- **Invariants in the callers → in the store.** "One decision per hour"
  lived in two callers (each a full parse of the log per batch) and two
  concurrent batches could both pass it; "one outcome per decision" lived
  nowhere, so the outcome-id migration handed the learner two outcomes
  per decision; the contract advertised a stockout-field gate nothing
  enforced. The store indexes hours and answered decisions, refuses the
  second of either and counts it, and skips on load what it would have
  refused; the counts ride into `quality_counts` beside the outcome side.
- **The daily lane behind the red check.** `advance` stopped on a red
  `status` before ingest, and the rows that go red after launch (a fired
  stop, assurance) are refreshed only by the lane it skipped: nothing
  ingested, the window never diluted, the monitor never re-read — a fired
  stop deadlocked the lane. The lane runs first; the red stop follows,
  naming the resume path.
- **A re-seal that verifies against nothing.** `advance` re-seals under
  `config` whenever the environment drifted; `seal()` verified with no
  prior seal, so an artifact hand-edited at the same time became the
  record under reason `config`. A re-seal's reason now says which
  artifacts may have moved (none for config/libraries, the calibration
  for weekly-refit) and refuses the rest.
- **The W sweep ranked an estimator nobody runs.** `calibration_window_sweep`
  fit each candidate window with a category-grain sold/pred ratio, while
  the factor production pastes W into is `_solve_level_factors`
  (subcategory grain, shrinkage, censored basis, a thinness floor). The
  ranking could prefer a window for a solver that never sees it. The
  sweep now calls the one solver on the raw basis fidelity already holds
  (the frame's mu divided by the factor in force, no second predict) and
  applies through the same waterfall.
- **assurance's own `m` → the learner's.** `correlation_drift` averaged
  every learnable hour of every moved episode as `m`, while the learner
  deflates by forced outcomes per episode (`deff_from_episodes`): a
  one-forced-hour world read as six and the alert tripped on a rho error
  the update never applied. One home for `m`, one population.
- **The tau paste gate lived in the engine.** `tau_provenance_error`
  reads two reports and a paste tolerance and neither prices nor learns;
  a reader hunting the paste checks looked in `ops` and found it beside
  the uniform draw. It is `ops.config_keys` now, with the key registry it
  belongs to; `engine.explore` keeps the name for its callers.
- **The persistence streak lived in the monitor.** `evaluate_guardrail`
  is the rule the floor, the trigger and shadow's trace all read, and
  shadow importing it from `daily.monitor` made the harness depend on
  the production lane. It sits with the deterioration series it grades,
  `common.guardrail`; the monitor re-exports it.
- **The learning maths in the daily lane.** `grid_update` and
  `row_information` are pure likelihood and information -- no store, no
  gate, no CLI -- and lived in `daily.update` beside the operator gate.
  `engine.learn` holds them; `engine` prices and learns, `daily` collects
  and commits.
- **update and monitor imported each other.** The monitor read
  `finalized_days` from update; update lazily imported the monitor's
  episode frame to price a budget. Both are functions over the event
  records: `events.pairs` (the priced and the suspended days) and
  `events.frame` (records to the settled episode frame, and the IL base
  by close day). Neither lane module needs the other now.
- **One `explore.py` for four concerns.** The draw, the Q-spread ledger,
  the budget controller and the paste gate shared a file named for the
  first; a reader hunting `walk_tau` looked for "budget". The draw stays
  in `engine.explore`; the budget and the walk are `engine.budget`, the
  ledger `engine.spread_ledger`; `explore` keeps every name.
- **The store carried the contract.** The required fields and value
  checks are the integration contract, read by the doc test and by
  engineering's page; they were the first third of `events.store`. They
  are `events.contract`; the store enforces them and keeps the names.
- **`tune.py` was two modules.** A registry of what each key is to the
  chain (read by `advance` and `status`, never by tune's own checks) and
  the findings, four functions carrying three or four unrelated readings
  each. The registry is `ops.config_keys`, with the one report-staleness
  judgement both drivers had re-implemented; the findings are one
  function each, grouped by the class they emit, in the report's order.
- **The frozen bundle, spelt seven times.** Model, posterior, `r`
  lookup and prior were loaded in seven places with two wordings for a
  missing file. `fit.artifacts.load_bundle` loads each on first use;
  `ops.status.read_artifacts` reads the JSON ones once per run and hands
  them down instead of each row re-reading them.
- **`advance.plan` was one ladder.** The run order was a single
  if/return chain; `AGENTS`' phase table was the only place it read as a
  list. It is now one function per phase in an ordered table, each
  answering "what runs next" or nothing; the step dicts did not move.
  The readiness report it writes is `ops.readiness`, the step runner and
  the phase names `ops.run`.
- **Path literals and parser boilerplate in every driver.** The
  drivers re-spelt the modules' own report defaults in their argv and
  every CLI declared `--config` by hand. `common.paths` names each file
  once (the argv strings are unchanged) and `common.cli` builds the
  shared flags; a driver that restates a default is a drift waiting to
  happen.
- **The audit trail inside provenance.** `common.provenance` answered "what
  was this fitted against" and, in its second half, copied bundles into
  history folders and indexed them. `common.history` holds the trail; it
  reads provenance's artifact walk and provenance never reads it back.
- **The integration cycle reached into the shop's privates.** Its pricer
  was a closure inside `tools.e2e_cycle.run`, capturing the simulator
  before it was bound and calling six private methods, and its hour loop,
  feed write and ingest were second copies of the simulator's. The shop
  now offers them (`evaluate.pilot_shop.PilotSim.run_hours`, `write_feed`,
  `LaneBPricer`, `ingest_and_pair` over the lane's own `ingest_feed`) and
  the cycle is a driver again.
- **The worker's context and the engine's state, spelt three times.**
  Lane B, the simulator and shadow each built the dict `price_one` reads
  and the twelve-field state, shadow with its own shape because its
  worker called `decide` directly. `engine.state.batch_context` and
  `assemble_state` are the one spelling of each; shadow's worker goes
  through `price_one` (handing in its per-episode stream and its spread
  sink), so a field added to the state or the context reaches every
  caller at once.
- **The economics block, three times.** The monitor's business metrics,
  the simulator's arm economics and shadow's markdown IL each summed the
  same settled frame with their own rounding. `common.metrics.summary` is
  the block, unrounded; each reader names the precision it reports at.
- **A harness importing its sibling.** Shadow imported the priced frame
  and the coverage guard from the backtest and carried its own copy of
  the frozen-vs-refit rescale. `evaluate.level` holds what both read
  about the level (`predict_frame`, `refit_scale`,
  `weekly_refit_schedule`); neither harness imports the other.
- **The tau derivation's bookkeeping, twice.** The ledger fold, the
  `rng.choice` sample and the reported block were spelt in the backtest
  and in shadow. `evaluate.tau` holds them (`fill_ledger`, `sample_ids`,
  `tau_derivation_block`); the sample is the same call in the same order,
  so every sampled report reads as before.
- **The simulator was one file.** The workspace, the shop, the lanes, the
  readings, the grading and the CLI shared `evaluate.pilot_sim`. The shop
  is `evaluate.pilot_shop`, the readings and expectations
  `evaluate.pilot_grade`, and `pilot_sim` is the run and its settings;
  the old names still resolve.
- **A second process pool.** The simulator kept its own executor and
  chunking beside `common.parallel.map_episodes` because it maps one
  batch per hour. `EpisodePool` is the held pool; `map_episodes` is the
  one-shot case of it.
- **The deck re-implemented the replay's shelf.** `scenario_deck.simulate`
  walked its own hour loop with the clip and restock bookkeeping the
  replay already had. Its paths now walk `evaluate.backtest._simulate_arm`
  priced by `_dp_price` (the restock is an adjustment, a fixed schedule
  the price callback); the deck keeps only its solve cache.
- **`fidelity_decomposition` in `common`.** One reader, the backtest,
  and the module said "several need them". It lives in
  `evaluate.backtest`; the old name is gone (nothing called it).
- **A re-export that carries its new home's imports.** Two of the moves
  above left the old name resolving by importing the new module at the
  bottom of the old one: `common.metrics` then loaded `evaluate.backtest`
  (and LightGBM) into every monitor start, and `engine.explore` loaded
  `ops.config_keys` for a gate nothing reached through the engine -- each
  one import away from a cycle. A re-export is for callers that exist; a
  name with none is deleted, and `tests/test_layering.py` pins the
  package map as an import graph: the lane's definitions load no harness,
  the engine loads no driver. Two smaller ones from the same pass: a key
  the caller settles (`batch_context(digest=...)`) is not computed and
  discarded, and a report whose digest is the live one is not diffed
  against the config.

- **The weekly refresh was an owner stop.** After launch the factor
  schedule must reach the week being priced, which needs an extract that
  reaches yesterday. `advance` stopped and told the owner to refresh it,
  so the one weekly step the pilot depends on sat outside the cron that
  runs everything else, on a person's calendar. The `--feed` run now
  pulls the extract through yesterday, prepares it, re-fits and re-seals
  when the schedule falls behind, and stops only when a current extract
  still cannot reach the week, which is a data problem. The cron host
  holds the Redshift credentials; that is the whole cost.

- **The history table in every hourly batch.** A batch read the whole
  trailing feed, sliced it to its SKUs and re-rolled the two demand-rate
  features, twenty-four times a day; handed the source schema, it ran the
  preparation chain each hour. Both features read strictly before the
  opening date, so one table per morning is exactly the batch's number.
  `daily.features` writes it after ingest from a rolling feed history the
  lane keeps itself; the batch joins on SKU × FC; the decision records
  the features it stood on so a later hour never recomputes them as of
  another day. The hourly path is now join, predict once, solve, commit.

- **The handoff asked engineering to speak our language.** Twelve fields,
  an episode id, an anchor discount: every one derivable from the hourly
  table they already produce, and the id rule was the one integration bug
  that kept recurring. `ops.price_hour` takes the shelf snapshot in the
  feed's own schema and answers in their units; `ops.check_inputs` turns
  the contract's checklist into a script result. The handover page
  became a process — phases, tasks, outputs, actions — with the contract
  as its appendices, and the rehearsal drives the same script the cron
  will.

- **The consumer must not define the producer's data** (owner). The first
  cut of `ops.price_hour` derived the episode id itself, against the
  store's latest decision — convenient, and wrong in ownership: the id
  names the producers' listing, they know whether it was extended or
  relisted, and the whole pipeline will one day be theirs to run. The
  id went back to them, with the rule shipped as a script
  (`ops.assign_episode_ids`: one hour's rows against the hour before,
  nothing else) to run or port; the hourly script reads the id as
  given, refuses a row without one, and evaluates the rule only to count
  disagreements. What we own is the check, never the assignment.

- **A surrogate id hid a defect the natural key exposed** (owner, 09-17).
  `decision_id` was a UUID while `outcome_id` had already become the
  shelf-hour. Making the decision id its twin (`dec-<sku>|<fc>|<date>T<hh>`)
  cost one line and turned the store's one-decision-per-hour invariant into
  something the id SAYS rather than something a side index enforces. It also
  broke the serial-versus-parallel report test on the first run, and the
  break was real: a shadow re-run had always appended a second copy of every
  decision and outcome into the shadow store, and random ids meant nothing
  ever collided to reveal it. The per-run counters (`quarantined_this_run`)
  were a workaround for that accumulation. A harness run now starts from an
  empty store (`EventStore(reset=True)`, which removes only its own three
  streams and refuses the production store outright). The episode stayed out
  of both ids on purpose: it is the producers' and can be relabelled, and an
  audit record's identity may not move when an upstream label does.

- **What we refuse belongs in the record too** (owner, 09-17). The live
  episode-id rule steps from the last hour the store saw on a shelf, and
  a refused hour stored nothing, so a shelf held through a rejection
  looked exactly like a shelf whose hour never arrived. The first fix was
  honest but weak: answer UNKNOWN for a gap rather than pretend the window
  restarted. The owner's point went further — we already send the rejected
  row back with its reason, so keep it. A third stream (`rejections`,
  `rej-<sku>|<fc>|<date>T<hh>`) now records every shelf-hour seen and not
  priced. It carries no price, never enters `priced_hours` (a corrected
  hour is still free to price), and never reaches the pairing or the
  evidence; `last_seen_by_shelf` spans decisions and rejections while
  `latest_by_shelf` stays decisions-only, since the anchor and the
  continued test must come from a real price. The check that was noise is
  now a signal: an unknown is an hour engineering did not send.

- **The export's friction was types, not names** (owner, 09-17). Asked to
  ship the warehouse tables in the feed's own spelling, the obvious read
  was a rename: `sku_id` to `skuseq`, `hour_of_day` to `hour`. Looking
  properly, the names were the smaller half. The chain normalises every id
  to ONE TEXT spelling (`events.pairs.ident`) precisely so a value read
  back as `7.0` still meets one stored as `7`, and it writes the day as
  text; the feed's columns are an integer and a date. A rename alone would
  have left a join that silently matched nothing. `add_feed_columns` now
  writes the shelf-hour in the feed's types beside the event's own fields,
  so the export stays a faithful dump and the join needs no cast. An id
  that is not an integer is null in `skuseq` and counted, because a column
  whose type changes between days is worse for a warehouse than a null a
  load can see.

- **A fresh-eyes review before launch found what the build had walked past**
  (owner, 09-17). Four reviewers with separate briefs — the hourly path,
  redundancy, structure and handoff, the daily lane — and eight defects
  confirmed. The one that mattered most was created by an improvement:
  with the decision id the shelf-hour, two overlapping hourly runs each
  built an in-memory index, both priced the hour, and the second line was
  dropped on the next load as a duplicate — silently, where UUIDs had
  made the same collision loud. The store now takes a lock on every emit
  and re-reads each stream's tail past what it has consumed before it
  checks (`_consume`), so the second run is refused with the reason
  named. Two crashes came from what the hourly script HANDED the store:
  an unreadable counter recorded as-is on a rejection took the next hour
  down on `float()`, and a null opening stock did the same on the restock
  test — both now read as unknown. The disagreement count compared the
  producer's answer against the last DECISION while the rule stepped from
  the last SEEN hour, so a listing that opened during a refused hour was
  reported as a contradiction; both are read from the record the rule
  stepped from. And the store-only rule claimed more than it could know:
  it never sees a close or a restock, so only a counter reset is decisive
  from there — the rest is counted as not decidable, which is also the
  nudge to send the closed rows. Two rules that were meant to be one
  disagreed at the edges (`<= 0` against `== 0`; truncated against raw
  counters), caught by a parity test over a synthetic day that should
  have existed from the start; and the producers' script imported one
  helper from the engine and so loaded LightGBM — `hours_between` moved
  to `common.windows`, where the rule lives.

- **What the build had spelt twice** (owner, 09-17, the review's second
  pass). The shelf-hour tag was formatted in five places and the hour
  parsed by hand in three, each `int(float(h))` truncating 17.5 to 17
  where the one key function raises; three by-extension file readers sat
  beside the shared one, and the producers' script's own let a NaN cell
  through as a truthy id; "is this a number" had four spellings. One
  home each now: `shelf_hour_tag` and `hour_int` beside `hour_key`,
  `as_number` for the lenient feed-cell reading (the strict
  `finite_number` stays the contract's test on purpose), `write_frame` in
  `common.io`, one shelf-index upsert in the store, one response-row
  builder per script. The tests had the same shape: the id formulas
  asserted in three files (now `test_pairs.py`), the closed/restock/reset
  trio built twice, two builders copied verbatim (now `conftest`'s
  `ref_rate_history`, `shelf_row`, `R_LOOKUP`), three docstrings still
  describing the design the reversal removed. What was NOT consolidated,
  deliberately: the store-only rule and the producers' rule take
  different inputs and answer differently, and the vector rule in
  `common.windows` and the scalar one in the script are two homes with a
  parity test between them rather than one home with two callers.

- **The layer between "clone" and "cron" did not exist** (owner, 09-17,
  the review's third pass). The code and the contract were specified to
  the field; how to run them on a host was implied. Task 0.7 of the
  handover page is that layer: the ship list (what must be on the host
  before the first hour, `reports/` included, since `status` reads it),
  the two cron lines with the working directory, the zone and `flock`,
  the time-zone and daylight-saving rule, the exit codes (the hourly
  script exits 0 when every shelf is refused -- alert on the report),
  the store's growth and the backup rule. With it: `requirements.lock`
  (the seal records library versions, so an unpinned install could fail
  status on day one), `.env.example`, `examples/` with one file of each
  table from the synthetic rehearsal, a CI workflow that runs the suite,
  and `.gitignore` entries for every directory the page tells
  engineering to write. One code change rode along: the feature table's
  day is the ingested feed's plus one, never the host clock -- the
  first fix had moved it from local to UTC, which is still a clock. Four
  doc contradictions closed: `config.yaml` is in git, the field lists
  live in `events/contract.py`, the CI claim is now true, and the
  failed-pushes columns are read in either spelling.

- **The trailing fit window is the owner's, not the sweep's** (owner,
  09-18). W was MEASURED: the backtest's rolling-origin sweep recommended
  it and `tune` pasted it. A pasted W is the "calibration" re-run class,
  the heaviest short of a retrain -- the loop turns, the backtest re-runs
  and re-scores the very sweep that chose W, shadow re-runs -- and a
  near-tie between two windows could cycle it; the near-tie hold was the
  first defence. The owner's reading: a smoothing window is a judgement,
  stable across retrains, and the sweep's evidence is what the judgement
  reads. W is SET BY OWNER now, the sweep's recommendation an owner
  decision `tune` reports with the evidence, and a change still turns the
  loop once, deliberately.

- **A paste waited for a report it did not read** (owner, 09-18). With
  W the owner's, shadow still ran twice on a clean chain, and the reason
  was `tune`'s own gate: no finding until all three reports existed. The
  exploration bias is the backtest's number and shadow's input; held
  until shadow had run, it was pasted after the first shadow and made it
  stale. The fix is a table, `READS`: each check names the report(s) it
  reads and is asked once they exist. A missing report now withholds
  only the findings that read it, listed as `waiting`; `blocked` is an
  invariant violated, nothing else. Bootstrap -> paste what the backtest
  and thresholds measured -> posterior -> shadow once -> paste tau (re-run
  class `none`) -> re-seal. The same shape as the fit-window lesson: a
  loop turned because a value sat in the wrong phase, not because the
  chain needed it.

- **The level error is not absorbed into ε, it is tilted** (owner,
  09-18). The case against the floor was a good one: ε is fitted, not
  physical; the model keeps improving and the posterior keeps moving; a
  demand curve closer to reality at the prices we use is what the DP
  wants. The arithmetic answered it. With only ε free to explain a forced
  outcome, the learner settles on `ε + β/Λ_f`, and the model's remaining
  error is `β·(Λ/Λ_f − 1)`: zero at the one forced price, the full level
  error at the reference regardless, and growing with slope `β/Λ_f`
  beyond. A short forced move makes the model exactly right where it
  rarely prices and multiples wrong at the deep markdowns it exists to
  get right; an under-forecast level becomes an overstated |ε|, the DP
  marks down too lightly and scraps; the benign error (level, which
  shifts urgency) becomes the harmful one (slope, which IS the tier
  comparison); and the walk-back is slow because cheap evidence still
  narrows the posterior, the rails bound the step, and every retrain
  moves β under a belief that cannot see it move. δ_min caps the tilt at
  `k·|ε|`; it does not remove it. The owner's reading: the IL budget is
  fixed, so take the larger gap and the lower forced rate — fewer forced
  decisions that each teach something over many that buy confidence in a
  tilted ε. `delta_min_bias_multiple` is that lever and shadow's
  `exploration_budget_sweep` prices it; the value is the owner's to set
  from that table, never a paste. Design 5.8 carries the derivation.

- **The pricing host gets a folder, not a repository** (owner, 09-19).
  Engineering's host needs the hourly job, the morning feature job, the
  checker and the id rule, and nothing else; handing over the repository
  handed over the training chain, the harnesses and a morning cron
  (`ops.advance --feed`) that drags the learning lane onto a host meant
  to price. `integration/` is that folder: GENERATED by `ops.integration`
  as three commands at the root (the hour, the morning table, the checks)
  over `src/`, the exact import closure of the three, package layout kept
  so no import is rewritten; the config, the pinned requirements, the
  examples, the handover page and a README that is the short handoff.
  The owner's chain fills it: `seal` and `advance` end by syncing the
  sealed artifacts, the posterior, the extract and the config into it, so
  a retrain reaches the host's folder with no hand step and the folder's
  `synced.json` says which bundle. The id is the producers' -- the first
  cut shipped the id rule as a command, which invited them to run ours
  instead of theirs; now nothing in the folder assigns one and the
  checker gained the other direction, `--response`, the hour's output
  against the snapshot it answered. Generated, never edited: a second copy of a
  module is the thing this repo forbids, and the only way to carry one is
  a test that refuses a copy differing from its source (the manifest's
  digests, byte for byte) and a build that refuses to empty a directory
  it did not write. The first standalone run found a defect the suite had
  walked past: `daily.features` raised on its own `--out`, because every
  test called `build()` and the command line -- the cron line, and what
  `advance --feed` runs -- had never been executed. Fixed with the test
  that runs the command.

- **The pricing folder is maintained in place; the repository is not
  refactored for it** (owner, 09-19). The first folder was the verbatim
  import closure, 36 modules, and the owner asked why a minimal folder
  was not minimal. The closure was fat for reasons in the repository:
  five re-export shims ("moved to X; the names stay here for callers")
  pulling the budget walk, the ledger, the audit trail, the clustering
  maths and the level-factor fit into a folder that prices; two modules
  there for one function; and the trainer being the applier's file. I
  removed all of it across sixty files, and the owner's answer was to
  confine the change to the folder and leave the codebase alone. So the
  refactor is reverted, and the folder is a hand-maintained copy: one
  file per repository module, most verbatim and pinned to their sources
  by `ops.integration --check` (a fix to the engine is ported or the
  suite fails), and a listed few curated with the reason -- the shims
  dropped in the copies, `fit/model.py` the applier without `train()`,
  `daily/failures.py` the one loader, `EXPLOIT_ONLY` fixed on in the
  batch caller. 26 modules, 6,459 lines. The trade is explicit: the
  curated files are ported by hand when their source moves, and the
  manifest is the list to walk. Exploit-only lives in the folder's copy,
  not in config.yaml, so the synced config needs no key and the
  rehearsal, shadow and the repository's lane are untouched.

- **The folder is the hourly command and what it needs** (owner, 09-19,
  the third cut). Two more things were still in it that the owner's
  chain already produces: the morning feature builder and the checker.
  "Why have scripts to generate artifacts in the folder when I can copy
  what the parent's scripts already generated?" -- the feature table is
  an artifact like the model, so the morning `advance --feed` run in the
  repository writes it and the sync carries it in; the checks run in the
  repository on the files engineering sends. And the config was the
  whole config. A tracer over one dry-run hour (every key path read,
  every file opened) gave the honest list: nine sections, five artifact
  files, no learning value at all. `ops.integration.KEEP` is that list,
  the sync writes the pruned config, and the folder's strict loader keeps
  the one gate that matters here, `launch_date`. The folder: one
  command, 21 modules, the five artifacts, the day's table. What the
  tracer taught: the hourly path never reads rho, tau, the stop
  thresholds or the prior -- those are the learning lane's, and a
  standalone hour has no business refusing to start over them.

- **Minimal is measured per function, not per file** (owner, 09-19, the
  fourth cut). Module-level pruning left files like `posterior.py` and
  `explore.py` in the folder, and the owner asked why an exploit-only
  hour needs the posterior's update or the exploration draw. It does
  not. A function-level trace of the two commands over a realistic flow
  (four consecutive hours: entries, continuations with the closed hour
  riding along, an empty shelf, a row without an id, an unkeyable row,
  a re-priced hour, the pool; and a first and a later morning) found
  1,300 of 6,000 lines in functions never called -- the posterior's
  update, init, tau walk and suspension writes; the outcome side of the
  store and the pairing; the seal and the verify; the training-time
  split; the batch command's readers. Cut by syntax tree, the orphaned
  imports and main blocks after them, and the run repeated: every
  decision, rejection, response row and feature value identical to the
  pre-trim reference (only solver wall-clock timings differ). Two
  lessons under it. The first tracer matched decorated functions on the
  wrong line (a decorator moves the code object's first line) and
  reported the store's lock, the cached loaders and every property as
  unused -- a trace is evidence only once its matching is checked
  against a call you know happened. And a fixture does not reach every
  live branch: the malformed-line quarantine, the deep-inventory closed
  form, the JSONL writer, the pool child and four functions live
  branches still referenced were kept or restored by reading the code,
  not the trace. The folder: two commands, 21 modules, 4,660 lines.

- **A minimal folder is written, not trimmed** (owner, 09-19, the fifth
  cut). Four rounds of pruning the repository's modules into the folder
  still left 4,660 lines across 21 files, because a copy keeps the shape
  of its source: the store's four streams where two are written, the
  batch caller's readers, a config loader that knows every SET BY OWNER
  key, a demand module that is the trainer's file. The owner's answer:
  "it by no means has to mimic the parent -- write from scratch if you
  need to." So `integration/pricing/` is a second implementation of the
  exploit-only hour and the morning table, fourteen modules, 1,992
  lines, each the size of what it does. What holds it to the first: the
  decision event is the contract (the learning lane in the repository
  reads the folder's store), so the rewrite had to produce the SAME
  events, rejection records, responses, hour reports and feature table
  as the repository for the same inputs -- not similar, equal. Two
  proofs. While writing: a black-box reference flow (four consecutive
  hours with an empty shelf, a row without an id, an unkeyable row, the
  pool, a re-priced hour, a dry run; a first and a later morning) run on
  the trimmed folder and on the rewrite, every stream compared, zero
  differences apart from wall-clock timings. In the suite, forever: the
  end-to-end test prices one hour through `ops.price_hour` and through
  the folder's command and compares every field of every decision and
  rejection, twice -- exploration on, where the drawn fields are the one
  allowed difference, and exploration suspended on a copy of the
  posterior, where only the clock and the config digest may differ. The
  lesson under it: two implementations of one rule are the thing this
  repository forbids, and the parity test is the only reason the folder
  may be one -- an engine fix is ported to `pricing/` by hand, and the
  test, not the manifest, says whether the port is complete. The
  first parity run was itself a finding: `expected_il` and
  `expected_denominator` are the APPLIED action's expectations, so on an
  explored decision they differ from the optimal's and belong with the
  drawn fields, which the handover's field table already says and the
  first draft of the comparison had not read.

- **The hourly service is stateless** (owner, 09-19, the sixth cut).
  "For the DP to put a price on a SKU × FC at an hour, it doesn't need
  the episode id; the service should be stateless." It is right about
  the DP: the solver takes the anchor, the stock, the horizon, the path
  and r, and the id reaches none of them. What the store answered was
  one question -- is this hour the episode's first? -- plus the forecast
  made at that first hour. Both come from the row instead. The
  producers' id is the opening tag (`<sku>|<fc>|<day>T<hh>`, the spelling
  the rule already used), so entry versus later hour is read off the id;
  the row's price in force is the anchor, null at entry and the applied
  price piped back on every later row (engineering's undertaking); the
  features are the OPENING day's table, kept on the host, so a later hour
  is forecast on what its first hour was -- the design's rule against a
  mid-episode recompute, met without the store. The forecast is re-made
  every hour over the remaining horizon and equals the entry's path
  sliced, bit for bit in the suite, except across a week boundary inside
  an episode, where the level factor is the request week's; taken as the
  fresher value. Gone: the store's three indexes and the tail re-read,
  the continuation branch, the path slicing and the restock tail, the id
  rule and its three counters, the missing-id refusal. The log stays,
  append-only and never read in the folder; the learning lane's reader
  dedups by id, and a re-sent hour appends the identical decision. Two
  producer faults gained a reason each: a later row without the price in
  force (refused, never priced as an entry -- that would let the price
  rise mid-episode) and an id that places no hour. The proof is the same
  parity test, extended by one hour: the applied prices piped back as
  the next snapshot's price in force, the repository continuing from its
  store and the folder from the row alone, every field equal. The lesson:
  state that exists to answer a question the request could carry is a
  dependency in disguise; the question was who owns the episode
  boundary, and the answer was already "the producers".
  The config followed: read off every `cfg[...]` in the package (the
  bundle test now holds KEEP and the code equal both ways), the
  exploration section and `posterior.epsilon_max` set only `delta_min`,
  the floor under a draw the folder never makes (the event records 0,
  the value the handover already documents for "before the scale is
  measured"); `data.exclusion_window` excluded a past demand-issue
  period the folder's 45-day rolling history never reaches, so the
  builder no longer applies it -- a FUTURE exclusion near launch is a
  training-time change and the folder's history would not follow it
  unless the step is put back; `manufacturing_window_hours` and
  `min_feasible_tiers` were read by nothing. Fourteen keys in nine
  sections became fifteen keys in eight.

## The lesson under all of it

Legacy history is confounded three ways (ramp ↔ hour, survivorship,
common day shocks), and every estimator change above is a way of being
honest about that rather than fixing it. The fix is exogenous price
variation: `engine.explore`'s uniform draw is the randomisation, tau its
budget. The prior only needs to be *not confidently wrong* until then.
