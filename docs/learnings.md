# Learnings — what was tried, why it lost, what replaced it

The working code carries only the current design; superseded designs live
here so an agent does not re-propose them. One entry each: was → learned →
now. Dates are owner sign-off, 2026-08.

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

## Calibration

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
- **A row-level date cut on the unfiltered frame** once put ineligible rows
  into the level fit, so the gate and the fit solved on different rows.
- **A failing 5b iterated to `--max-turns`.** When `--check-convergence`
  itself failed, the artifact carried no `convergence` block, the stall test
  had nothing to compare and prior/dispersion re-ran twenty times; a failing
  5b now stops the loop with its own message.

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
    578 draws). Rank plus a jitter from the decision id is exactly U(0, 1)
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
  Also learned: the sim's `hold the current price` fallback opened an
  episode at a bare `d_max` off the tier grid, and the next decision had
  no feasible tier at or below its anchor — the same trap Lane B's
  fallback must avoid (open at a tier). And a dynamic worth the owner's
  eye (design §11.3): a posterior mean stepping toward zero inflates
  `delta_min` until few action sets hold an admissible tier, exploration
  starves with no stop fired, and tau climbs by the clip every zero-spend
  day — `exploration_never_starves` grades it.

## The lesson under all of it

Legacy history is confounded three ways (ramp ↔ hour, survivorship,
common day shocks), and every estimator change above is a way of being
honest about that rather than fixing it. The fix is exogenous price
variation: `engine.explore`'s uniform draw is the randomisation, tau its
budget. The prior only needs to be *not confidently wrong* until then.
