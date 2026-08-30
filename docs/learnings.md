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
  (`bootstrap.prior_density`) — the 50/50 arm mixture reproduces the
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
- **Entry-only spread collection.** The replay collected Q-spreads at entry
  only, funding ~1 exploration per episode against a system that explores
  every hour — its own bisection reported 1.00× regardless (a number a
  procedure solves for is not evidence about that number). Spreads are
  collected at every decision hour; shadow derives the launch tau on its
  own anchored path.
- **Poisson information under an NB likelihood.** `pipeline.update`
  accumulated `μ·L²` while the likelihood is NB (information
  `μ·L²·r/(r+μ)`), overstating evidence ~1.6–1.9× on top of what deff
  corrects. Fixed by theorem on both paths.
- **Scrap through a local copy.** An inline scrap rule in the budget base
  dropped ALL scrap on a sentinel-free feed, understated the budget 10× and
  flipped the verdict to WOULD SUSPEND. Scrap goes through
  `common.episodes.classify_last`, nowhere else.

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

## The lesson under all of it

Legacy history is confounded three ways (ramp ↔ hour, survivorship,
common day shocks), and every estimator change above is a way of being
honest about that rather than fixing it. The fix is exogenous price
variation: `pricing.explore`'s uniform draw is the randomisation, tau its
budget. The prior only needs to be *not confidently wrong* until then.
