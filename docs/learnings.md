# Learnings — what was tried, why it was replaced, what replaced it

The working code carries only the current design. This file is where the
superseded designs live, so the codebase stays clean without the reasoning
being lost. Each entry: what we did, what broke or what we learned, what the
code does now. Newest decisions at the top of each section.

Dates are decision dates (owner sign-off), 2026-08.

---

## Elasticity prior

### The bracket method → the profile-density method
**Was:** the original PRD's §9.5 procedure. Two argmax estimates per category —
`epsilon_naive` (no hour control, biased too elastic) and `epsilon_controlled`
(hour fixed effects, biased toward zero) — with
`prior_mean = midpoint`, `prior_std = max(half-gap, std_floor)`, and a
constant fallback `−1.00 ± 0.60` whenever a category failed acceptance checks
(wrong sign / boundary / inverted ordering).

**Learned:**
- The argmax throws the curve's *shape* away, so a sharp peak and a dead-flat
  likelihood report the same kind of answer. Two fixture categories with a
  literally *constant* likelihood reported `−4.000 ± 0.400` — argmax of a
  constant array is index 0, the search bound.
- The fallback constant was produced by nothing measured. On the production
  extract it overwrote a measured bracket (BAKERY & PASTRY, −1.6125 ± 1.0875
  on 21,484 rows) because two *other* categories failed — the all-or-nothing
  scope. We first fixed the scope (per-category acceptance, inversions flagged
  not fatal), then removed the constant entirely.
- On the first real held-out comparison the bracket scored *below a flat
  prior* — worse than knowing nothing.

**Now:** `bootstrap.prior_density`. The whole deff-deflated profile
likelihood becomes the prior as a density; the 50/50 arm mixture reproduces
the bracket exactly in the sharp limit (two point masses at a, b have mean
(a+b)/2, std |a−b|/2), and degrades to the uniform on the support when the
data says nothing. No `fallback_mean`, no `fallback_std`, no `std_floor`.
Every run writes `holdout_comparison` (log marginal predictive vs `oracle`
and `uniform`) so the method carries its own evidence.

### Censored NB likelihood in the prior → censored Poisson (QMLE)
**Was:** the bracket scored a censored *negative binomial* likelihood, which
needs a dispersion `r` — and `fit_dispersion` needs an elasticity to form
residuals. A genuine ε ↔ r cycle, first broken by running dispersion at the
fallback constant, then by handing the bracket a "reference r".

**Learned:** the reference-r seed went through three designs in a week
(pinned 0.42 → derived pooled → derived per-category, each fixing the last),
which was the signal the design was wrong, not under-tuned. The pooled fit
averaged over a 5×-below to 6×-above per-category spread; the per-category
fit still fed the term where dispersion really bites (`nbinom.logsf` on
censored rows).

**Now:** the prior's curve is a censored **Poisson** profile. The Poisson
quasi-MLE is consistent for the mean parameters whatever the true dispersion
(Gourieroux–Monfort–Trognon 1984), and ε lives entirely in the mean — so `r`
leaves the ε step *by theorem*, not by a measured sensitivity. The cycle is
gone: prior first, then `fit_dispersion` at the real per-category means. As a
bonus, censored entry rows no longer need dropping (the drop selected out the
fastest sellers and pulled |ε| toward zero).

### Ordering: dispersion-first → prior-first
**Was:** `fit_dispersion` ran before `estimate_prior`, forming residuals at
the constant −1.0 whatever the prior said.

**Learned:** harmless only while the fallback WAS the prior. Once brackets
were accepted per category, fitting r and ρ at −1.0 measured correlation
against a demand curve nothing used — moving the working elasticity
−1.0 → −1.5 moved ρ 0.3103 → 0.4236 and deff 3.347 → 4.204, 26% of the
learning rate. The ε → r direction is strong; the r → ε direction became
zero once the prior went Poisson.

**Now:** `estimate_prior` runs first; `fit_dispersion` reads
`prior.json` per-category means and records
`working_elasticity_by_category` so the basis is auditable.
`pipeline.assurance` shares the same basis for its live ρ check.

### Rows scored: entry → all stocked hours → entry again
**Was (original):** entry rows only, as originally specified. **Tried:** every stocked
hour, deff-deflated, because entry rows carry almost no price variation (the
entry hour predates the ramp, so it sits at `d_ref`; two fixture categories
had literally constant likelihoods).

**Learned:** the extra hours bought the variation *and* the within-episode
survivorship confound: a row at a deep discount exists precisely because
earlier hours did not sell, so conditional on a price-neutral μ_ref a deeper
price reads as *lower* demand. Hour controls cannot reach it — it is
selection on the unobserved demand shock. Wrong-signed categories went from
2/5 (entry) to 4/5 (all hours) on the fixture. The failure was invisible
until the peak was searched *past* the search bound: a positive optimum was
being clipped to −0.05 and reported as measured.

**Now:** `rows: entry` (the original spec was right), with `design_comparison` in the
artifact scoring all four rows × hour-control combinations every run, so the
choice is re-evidenced on each extract rather than fixed forever.

### Hour control: pooled `hour_of_day` → same `date_hour` across sku×fc
**Was:** the controlled arm profiled out the clock hour pooled across dates —
removing the average evening lift and nothing else.

**Learned (owner's insight):** "same-hour cross-episode" means the same hour
*of the same day*, compared across sku×fc — that absorbs weather, footfall, a
rival's promotion, everything shared by the moment. On the all-hours set it
cut wrong-signed categories from 4/5 to 2/5 while keeping ~50× the
identifying power of entry rows. Caveat that ships with it: a time cell
fitted from too few rows absorbs the price response itself (incidental
parameters, biases |ε| toward zero), so cells under
`min_rows_per_time_cell` fall back to 1.0 and the median cell size is
reported.

**Now:** `hour_control: date_hour` with the thin-cell guard; the pooled
control remains available and the 2×2 is measured per run.

### Prior std: floor constant → zero-width bug → measured floors
**Was:** `std_floor: 0.40`, a chosen constant. **Then:** the density method
removed it and production promptly produced `std: 0.0` — a 9,402-unit
likelihood span across a 159-point grid is 59 nats per step; the density is a
delta function, and a zero-width prior freezes the posterior (`bounded_step`
can never move it).

**Learned:** curvature is *sampling* precision. At production scale the
uncertain thing is the model — confounded history, imperfect μ_ref,
non-stationarity — none of it visible in the within-sample curve.

**Now:** the reported std is the widest of three *measured* quantities: the
density's own width, the grid resolution, and `fold_spread` (movement of the
estimate across disjoint time slices of the train window). `std_basis` names
the binding one.

### Wrong-sign handling: reject → (accidentally lost) → reject again
The bracket's one unconditional reject was a non-negative endpoint. The
density method dropped it along with the reject path; production returned a
confident `−0.05`. Restored: the unconstrained peak is searched past the
bounds, a peak ≥ 0 discards the category's own density (it takes the pooled
one, and is listed in `wrong_sign_categories`), and rejected categories are
excluded from the pool they fall back to — otherwise the fallback inherits
the confound. With nothing left to pool, the pool is the uniform, which is
the honest statement that the extract does not identify elasticity.

---

## Dispersion (r, ρ, deff)

### Clamp: percentile for everything → under-dispersion exemption
**Was:** every fitted r above `clamp_percentile` of the converged set was
clamped down, "high values are spurious".

**Learned:** an r at the search ceiling has two causes wanting opposite
treatment. A thin group's wandering MLE wants clamping; a group genuinely
*steadier than Poisson* also lands there — NB variance ≥ mean at every finite
r, so the ceiling is the closest the family can get — and clamping it makes
the model claim variance the data does not have, for the steadiest cells.

**Now:** Pearson dispersion (`mean((k−μ)²/μ)`) separates them; groups below
1.0 are exempt and listed in `under_dispersed_groups`. A long list means the
NB is the wrong family for the extract, and the artifact says so.

---

## Population and data quality

### Closure: heuristics → the write-off sentinel
Several generations of "did this episode end?" logic (counter-based,
tolerance constants) collapsed to one source-native rule: the source zeroes
`ending_inventory` on the final row when it writes off — so closure is
`ending_inventory == 0` on the last row, full stop, and the unclipped sign of
`starting − sold` there distinguishes censored / scrap / restock. No
tolerance constant survives.

### Drops → flags (restocked, below-cost, edge-truncated, negative-window)
Each of these was once a hard drop; each deletion cost the frozen artifacts
population they needed (restocked alone: 18.1pp of the extract's COGS) or
answered a gate by deleting the gate's subject (`share_non_explorable`
measured 0.0 because non-explorable episodes were already gone). All are now
flags: kept for the demand model, excluded only where the specific consumer
cannot use them (`dp_eligible` for the solver; `eligible` for scrap/IL).
Restocks became tractable once `hour_adjustment` treated arrivals as
exogenous state the replay applies — the DP finds out at the next hour,
exactly as production does.

### One dp_eligible summary row → two population-gate rows with `used_by`
The waterfall reported only `dp_eligible`, so the eligibility gate's cost
read as the solver's fault and the population the demand model trains on
appeared nowhere. Now `eligible` and `dp_eligible` are both gate rows, every
row carries `kind` (hard_drop vs population_gate) and `used_by`, and the
three nested populations (integrity ⊃ eligible ⊃ dp_eligible) print at the
end of every run with the containment asserted.

### cogs_at_risk: opening stock → supply
Opening stock understated every restocked episode's exposure (a window
opening with 3 that takes 10 mid-flight has 13 units at risk). Now unit cost
× supply (opening + gross arrivals), matching the clearance denominator that
had already made the same correction.

---

## Exploration budget and tau

### Budget base: same-day / window-mean IL → trailing close-day IL
**Was:** the daily budget was 1% of a markdown-IL figure that mixed bases —
the replay derivation used a window mean, the shadow harness charged an
episode's discount cost to the day it was spent and its scrap to the day it
closed, and "today's budget" implicitly needed today's own IL, which is
unobservable until today's episodes close.

**Learned (owner's rule):** an episode's IL is a settled number only at its
close, and tau for today must be computable at midnight from history alone.
Attributing discount and scrap to different days split one episode's loss
across the calendar; funding exploration from the same day's own IL would
spend hardest on exactly the days already losing most.

**Now:** a day's realised IL is the whole-episode IL (discount AND scrap) of
episodes that CLOSED that day, attributed at close; the budget is 1% of the
mean of that series over the trailing `budget_il_window_days` (7) calendar
days ending yesterday — a moving window that adds yesterday and drops the
eighth day back. Nothing about today needs to be known when tau is set.
`derive_tau_initial` keeps a window-mean base (one launch constant) and
records `budget_basis`.

### tau clip: symmetric [0.5, 2.0] → asymmetric [0.5, 1.25]
**Was:** `tau_adjust_clip: [0.5, 2.0]` — tau could halve or double in a day,
sized for a noisy single-day budget basis.

**Learned:** with a 7-day trailing base the budget barely moves day to day,
so a 2× daily rise had nothing legitimate to do. The two directions are not
symmetric: cutting is the safety direction (the measured 8.7× overspend
incident needs three halvings to walk inside the 2× stop condition — a 0.75
floor would take ~8 days, all of them above the stop), while raising tau is
never urgent. **Now:** down stays 0.5, up is capped at 1.25.

### Stale numbers in docs → anchored figures refreshed by the run
Figures (`rho 0.3103`, `deff 3.347`, the IL table) outlived the runs that
produced them and were quoted for weeks as current. Now figures are anchored
in place (`<!--f:rho.rho|dec4-->…<!--/f-->`) and
`tools.refresh_figures --write --dataset "$INPUT"` runs as bootstrap step
10b. The tool refuses a dataset whose name says it is synthetic — the first
`--write` from the fixture silently replaced the production IL table with
generator numbers, and nothing about the result looked wrong. Historical
figures (what a past decision cost) are deliberately *not* anchored.

### Judging estimators by their outputs → held-out comparison
"Is −1.61 ± 1.09 better than −1.00 ± 0.60" has no answer by inspection; the
argument ran for days. Now every prior run scores its candidates on a window
none of them saw (log marginal predictive), bracketed by `oracle` and
`uniform` — and the report leads with `information_available_per_row`
(oracle − uniform), because a method gap that is a large share of a tiny
number is still tiny. A candidate below `uniform` is named as worse than
knowing nothing.

### The profile plot ends arguments the fit cannot
`tools.profile_epsilon` exists because the fit reports an argmax with four
decimals whether the likelihood is a spike or a horizontal line. Plotting
`ll(ε) − max` per category, both arms, one shared y-scale, was what exposed:
constant likelihoods behind confident boundary estimates, the entry-hour
price-variation famine, and the positive unconstrained peaks.

### The lesson under all of it
Legacy history is confounded three ways (ramp ↔ hour, survivorship within
episodes, common day shocks), and every estimator change above is a way of
being honest about that rather than fixing it. The fix is exogenous price
variation: `pricing.explore`'s uniform draw from the tau-affordable set is
the randomisation, tau is its budget, and the posterior learns from those
draws. The prior only needs to be *not confidently wrong* until then.

---

## Smaller lessons swept out of code comments (2026-08-25)

- **Gross, never netted.** An early `episode_flow` netted a shortfall against
  a same-size restock ("one sale bucketed an hour late"). Inference dressed as
  arithmetic: it read a window with 2 restocked and 2 shrunk as having
  neither, letting a restocked episode into the DP-side population with its
  clearance priced against the wrong supply. Arrivals and losses are counted
  gross ever since.
- **Row-scoped drops manufacture chain breaks.** `null_category` and
  `zero_base_price` were once row-scoped; they punched holes mid-window, so
  re-segmentation split episodes into fragments and episode counts *rose* at
  that stage. All post-id drops are episode-scoped now and re-segmentation is
  a checked no-op — an assertion, because the invariant fails silently.
- **Shrink was quarantined on the live path** while the offline chain counted
  it as an ordinary event. Every shrink hour then failed the event store's
  reconciliation, `event_completeness` fell by the feed's shrink rate, and the
  shadow gate failed for a reason no integration work could fix. Shrink is
  now named on the event, not quarantined, on both paths.
- **The closure sentinel once had a fallback** — absent the sentinel, treat
  every episode as closed. That kept a fixture that had never modelled the
  convention looking healthy for months, and would have done the same for a
  real feed that stopped emitting it. No fallback: a sentinel-free feed reads
  every episode unclosed, loudly.
- **Hour-level quantity tests were wrong twice.** "units > inventory is
  impossible" deleted restocks (18.1pp of the extract's COGS); "ending falls
  short is dirty" deleted shrink, fastest sellers first (sales straddle
  bucket boundaries more the faster a SKU sells). Continuity is the only
  hour-level rule that drops.
- **Round only for display.** `proposed_mean` was rounded to 4dp while
  `mean_before` was not, so the reported step could read 5e-5 over
  `max_mean_step` — a safety bound checked downstream. The stored posterior
  was always right; the report disagreed with itself. Artifacts carry
  unrounded values; the printed line formats.
- **`quarantined_event_count` was read from the cumulative file**, so a
  serial and a parallel run of the same input reported different counts and
  the difference was misread as a parallelism bug. Per-run counters live on
  the store; files accumulate.
