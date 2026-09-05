# Perishable Markdown MVP — System Design

**Status:** Implemented; validated on production FLC data after the section 12a data-definition corrections; calibration and shadow gates PASSED (model `baseline-20260809120225`)
**Standing:** The authoritative specification. Superseded designs and the incidents behind the rules live in `docs/learnings.md`.

---

## 1. Executive summary

The system prices perishable FLC inventory through its final selling window,
replacing a legacy policy that ramps discounts ~1pp/hour on the clock. It
minimises **Inventory Loss (IL)** — discount given away on units sold, plus
scrap cost on units unsold at expiry.

The central technical fact: **price elasticity cannot be estimated from our
own history.** The legacy ramp makes price collinear with hour-of-day, and a
second confound (within-episode survivorship, §3.2) biases every estimate
toward zero. The design uses history only for what it can support — baseline
demand, demand variance, correlation structure — and learns elasticity **in
production**, from deliberately randomized price perturbations whose total
cost is capped at 1% of markdown IL per day.

The safety engineering is structural (a below-cost or rising price is
unrepresentable, not merely checked for). The open question is the speed of
the learning loop within the MVP window — quantified in the risk register
(§13), with the shadow phase designed to answer it before any price is
applied.

---

## 2. Business context and objective

### 2.1 The problem

Each perishable SKU at each fulfilment centre gets one final selling window —
one **episode**: a run of contiguous selling hours for one SKU × FC, small
starting inventory (median ≈ 2 units). Windows routinely run past midnight
(36-hour windows are common), which is why an episode is keyed by the
source's own window counter rather than by calendar date (§12a). Each hour a
discount is chosen; whatever is unsold at close is scrapped at cost.

### 2.2 What the system optimises

```
IL = Σ over hours (original_price − applied_price) × units_sold   ← discount cost
   + cost × unsold_inventory_at_expiry                            ← scrap cost
```

Absolute IL is the currency amount the business is accountable for, and it is
additive across hours — a valid DP reward with no transformation.

### 2.3 What the business reads — and why the two can disagree

The headline metric is **IL% = IL / (original_price × units_sold)**. Its
denominator is **endogenous**: deeper markdowns sell more units and enlarge
it, so minimising IL and minimising IL% are different optimisations. Worked
example (price 10,000, cost 2,000, 10 units): policy A at 9,000 sells 4
(IL 16,000, IL% 40.0%); policy B at 7,000 sells 8 (IL 28,000, IL% 35.0%).
The planner chooses A; IL% prefers B. The pilot will be read on a metric the
planner does not optimise; the decision rule for "absolute IL improved, IL%
flat" is pre-committed (§11.2).

Reporting discipline that follows:

- Aggregation is **always a ratio of sums** (Σ IL / Σ denominator).
  Per-episode IL% is undefined for zero-sale episodes (~12% of episodes) and
  is never computed or averaged.
- Every IL% figure is reported **with its denominator**, and **absolute IL
  alongside** in every cut.

### 2.4 Learning rate is the product

Learning time scales with the number of cells and the volume of usable
evidence, so the MVP cuts anything that slows learning without protecting
money: subcategory-level learning, Thompson sampling, weekly model
retraining, stochastic replay rollouts.

---

## 3. The identification problem

**History predicts demand. It cannot identify price response.**

Almost everything the planner needs comes from history and transfers:
demand level, hour-of-day shape, seasonality, per-SKU velocity, the NB
dispersion `r`, the intra-episode correlation `rho`. Exactly one quantity
does not: the **price → demand slope**, because price was never a free
variable — every price was produced by the legacy rule, so price, hour, and
"this episode is already failing" are the same column of numbers.

Two operational consequences:

- **More history does not help.** The bias is a property of how the data was
  generated, not of sample size.
- **Randomisation is the only route.** One day of randomised prices
  identifies ε better than five months of history. Exploration is a budgeted
  P&L line because it is the sole supply of the one number history cannot
  provide.

### 3.1 The clock confound

Under the legacy ramp, discount deepens as the evening peak arrives; within
an episode price and hour correlate ≈ 0.8. Demand lift from the evening is
indistinguishable from demand lift from the discount.

### 3.2 The survivorship confound

Under the ramp, an observation at a deep discount exists *only because*
earlier hours failed to sell — deep-discount rows are adversely selected
toward low-demand episodes. Fitting on all rows drags every estimate toward
zero (observed: all sixteen categories pinned at the zero-side boundary).
The fix: identify only from **entry-hour variation across episodes**, never
from within-episode hours.

### 3.3 How the design responds

1. **The demand model never exposes a price gradient.** It predicts demand
   only at the category reference discount (price features overwritten at
   inference); price response enters through one learned scalar per
   category: `mu(d) = mu_ref × ((1−d)/(1−d_ref))^ε`.
2. **History contributes a density, not an estimate** (§5.6): the censored
   profile likelihood, deff-deflated, read as a density per category — sharp
   where the data identifies ε, degrading to the pooled or uniform density
   where it does not. A wide density is a designed outcome.
3. **Truth comes from production randomization** (§5.8): uniformly
   randomized, costed price perturbations whose outcomes are the only
   evidence the learner consumes.

### 3.4 Why the cold-start elasticity is not fitted to history

The best-fit shortcut was run, in its most defensible form (censored
likelihood, per category, entry rows). Its output is the argument against
it: a 30× swing on a single modelling choice, estimates pinned at search
bounds, direction inverting between model versions. Worse, a fitted ε makes
the offline replay look excellent by construction while the planner prices
with the confound: if the fit absorbed survivorship, the DP concludes
markdowns are futile and burns scrap. History legitimately contributes
sharp densities where supported and the wide pooled/uniform density where
not; the first weeks of the learning pilot are themselves the ε fit.

---

## 4. Architecture overview

```
FROZEN (fit offline, unchanged during the MVP window)
  baseline_model.txt      demand at the reference discount, by context
  feature_schema.json     feature order and categorical levels
  calibration.json        per-subcategory level factors (always fitted AND applied)
  r_lookup.json           negative-binomial dispersion by subcategory
  rho.json                intra-episode demand correlation (one scalar)
  prior.json              elasticity prior: a density per category

LEARNING (the only thing that updates in production)
  posterior.json          elasticity by cell {mean, std, n_obs, information,
                          version} + processed-outcome ledger (atomic)

DECISION PATH (per hourly decision interval)
  state ── validate (reject, never an unsafe price)
        ──▶ feasible discount tiers, constructed from the cost floor
        ──▶ exact DP: expected IL for every tier
        ──▶ exploit the argmax | explore the affordable, admissible set (§5.8)
        ──▶ price ──▶ decision event (~30 fields)
                          │
                 finalized outcome event
                          │
        censored NB update, correlation-deflated, bounded step,
        human-gated ──▶ posterior
```

Three properties:

- **Safety is structural.** The action set is built from the cost floor and
  the price-monotonicity anchor; a below-cost or rising price is
  unrepresentable, on every path including forced exploration.
- **Evidence is append-only and versioned.** Every decision event carries
  the exact prediction, posterior moments, and artifact versions it used;
  learning replays from events, never from recomputation.
- **Exactly one thing learns.** Everything else is frozen so posterior
  movement is attributable to learning, not drift.

---

## 5. Component deep-dive

### 5.1 Configuration — one tuning surface

Every threshold, window, rate, and bound lives in `config.yaml`; code
contains no numeric literals for anything tunable (a tunable without a
config key is a review failure). The strict-mode loader refuses to start
while any measured or owner-decided value is null. Every value is labelled:
`MEASURED` (produced by the pipeline), `SET` (design choice), or `SET BY
OWNER` (business decision — **an agent must never invent one**). The
runtime-required values `load_config(strict=True)` refuses on while null:
`dispersion.rho`, `exploration.tau_initial`,
`monitoring.stop_conditions.
scrap_deterioration_pct` and `margin_deterioration_pct`. Config is the source of every tunable
and of **no secret**: credentials live in `~/.env` as `REDSHIFT_*` — no
hostname, credential, or connection string in config, code, or a commit.

Settled design choices are code, not config keys: the artifact-fit
population (`eligible`), the calibration grain (subcategory), gate metric
(`level_bias_at_anchor`), gate window (`test`), fit window
(rolling trailing), the prior's identifying rows (entry) and time control
(date × hour), the dispersion fallback order, and the guardrail bases
(`common.guardrail.BASIS`). A losing branch kept as a config key is a way
for a settled decision to silently unsettle.

### 5.2 Data preparation — schema mapping and filter chain

Applies the source-to-canonical mapping exactly once, converts the discount
from percent to fraction exactly once, builds episodes as contiguous
windows, and runs a fixed filter chain with a row/episode waterfall after
every step. **Every filter drops the whole episode, never the row** — a hole
punched mid-window re-segments into a spurious short episode.

**Only integrity and scope rules drop.** Nothing is dropped for being hard
to *price* — those conditions are flags, because the demand model's features
carry neither cost nor the inventory chain, so such an episode is an
ordinary observation to every frozen artifact. Dropping them once removed
>70% of the extract's COGS from every fit. Transferable rule: **never gate
on a condition a constraint already handles** — a loud refusal counted in a
report beats a silent removal upstream.

| Step | Drops |
| --- | --- |
| `duplicate_hour_rows_dropped` | both copies of any repeated (SKU, FC, hour) — no principled way to choose, and they collide two runs into one episode id |
| `gap_split_windows_dropped` | **every fragment** of a source window a missing hour split in two — the second fragment opens mid-window and its first row would read as an ENTRY row in the elasticity fit. Detected from the counter falling in step with the clock |
| `exclusion_window_removed` | any episode with any hour in the known bad-data window (scope, not integrity) |
| `discount_out_of_range_dropped` | discount outside [0,1] — the percent→fraction conversion ran twice or never |
| `negative_quantities_dropped` | negative inventory or sales. **Not `cost <= 0`** — that is the `cost_missing` flag |
| `null_category_dropped` | no category/subcategory — no reference discount, no dispersion cell |
| `zero_base_price_dropped` | `original_price` still absent after fill-within-episode |
| `episode_universe` | the three conditions that make an episode's inventory readable: CONTINUITY (`ending[t] == starting[t+1]`, the only one that drops), the IDENTITY (`opening + restocked == sold + scrap`, a guard on the arithmetic), and a CLEAN CLOSE (`starting >= sold` on the last row, which flags `final_hour_restock`). Hour-level restock/shrink are real events, counted gross, settled at episode level |
| `contiguous_episodes_built` | *(not a filter)* re-segmentation — a NO-OP that RAISES if it stops being one |
| `negative_window_recovered` | *(not a filter)* "manufacturing" SKUs enter with an already-negative counter; episodes fitting inside `data.manufacturing_window_hours` (24) get a synthetic countdown `(cap−1) − position` (a countdown, never a clamp — the counter drives episode ids, the DP horizon, and `extend_to_window`). Longer ones carry the `negative_window` flag. **Runs after the re-segmentation check** because it is the only step that mutates `hours_remaining` |
| `eligible` | *(a GATE, drops nothing)* `accounting_closes & final_hour_clean & closed` — the population the frozen artifacts fit on, and the one every scrap/IL/clearance figure reads (`scrap_units` returns NaN outside it) |
| `dp_eligible` | *(a GATE, drops nothing)* what the DP can act on. Read by the DP, the backtest, shadow and the calibration gate |

Six conditions gate `dp_eligible`, each naming something the *solver*
cannot do; an episode is labelled with the first it trips:

| Flag | Why the DP cannot act on it |
| --- | --- |
| `cost_missing` | `cost <= 0` — a *missing* cost, not a free good. At zero cost `d_max` reads 1.0, a 100% discount enters the action set (`0 ** negative` crash), and scrap = cost × leftover reads zero, deflating IL. Two layers on purpose: `feasible_tiers` excludes non-positive prices (owns "which prices are legal"), and this flag keeps unknown-cost episodes out of every DP-side number |
| `non_priceable` | `cost >= original_price`, so `d_max <= 0` and `feasible_tiers` is EMPTY |
| `negative_window` | `hours_remaining` still `< 0` after recovery — the DP takes its horizon from the counter |
| `window_too_long` | above `data.max_window_hours` (120) — `extend_to_window` RAISES above the cap. Dropped, never clamped: clamping would invent a window end |
| `outcome_unknown` | the episode never closed inside this data. Gates `eligible` too: an unfinished episode is not a complete observation of anything |
| `final_hour_restock` | the last row sold more than it opened with, so the leftover is a guess. Gates `eligible` too — the censoring call cannot be made on an ambiguous final hour |

Four conditions are flagged and gate nothing:

| Flag | Why it does not gate |
| --- | --- |
| `below_cost_hours` | a LEGACY price. The backtest's DP arm is self-anchored and never sees it; in shadow the refusal from the crossing hour on is the cost floor working (counted in `rejected_reasons`). Test as `original_price × (1 − discount)`, never `applied_price` (zeroed on ~78% of rows). These episodes carry the widest price spread the extract has |
| `edge_truncated` | of the unfinished episodes, the ones the extract boundary explains. Expected under 1% on a long extract; well above means a feed problem |
| `restocked` | units arrived mid-window. The replay re-solves hourly and meets an arrival exactly as production does |
| `shrink` | units left unsold and unwritten-off. Counted into scrap, so `supply == sold + scrap` still closes; `unreconciled_anomalies` locates them by category and month |

The episode stock invariant is against SUPPLY: `sold <= opening +
restocked` (follows from the identity; asserted on the output). The
negative-window cap is a claim about the data and the stage checks it: if
`episodes_entering_negative_but_longer_than_cap` is not near-zero on an
extract, fix the cap, do not widen the recovery.

`eligible` — the middle population — is three conditions and no more, all
in `common.episodes.episode_flow`. It exists because of the censored
likelihood: the censoring call is read off the inventory, and an ambiguous
final hour means an untrustworthy censoring flag, which biases demand
directly. Everything `dp_eligible` additionally rejects is invisible to the
model, so those episodes stay in `eligible`. `integrity` — everything that
survived the chain — is read by `m1`/gate 1 only. The three populations are
NESTED (integrity > eligible > dp_eligible) and resolved through
`prepare_data.population(d, cfg[, which])` — always call it, never
re-derive the filter. Artifact fits read `eligible`; the DP, calibration
gate, backtest, shadow and tau pass `"dp_eligible"` explicitly.

Every stage reports **`cogs_at_risk`** — unit cost × supply (opening +
gross arrivals), once per episode — because rows are not the unit the
business cares about: a stage can take 1% of rows and 15% of the money. The
whole waterfall, COGS and per-stage detail dicts included, is **persisted to
`split_manifest.json`**: it used to be printed as three columns and dropped,
so `flow_identity.holds` turning False, the restock/edge diagnostics and the
shrink-vs-skew reading all landed on the floor while the run succeeded. A
NaN `cogs_at_risk` (any episode with a null cost) is written as `null` —
bare `NaN` is not JSON.

Why the paranoia: the three source-schema traps all fail silently. A missed
percent conversion produces discounts of 25.0 with no error; the realised-
price column is 0 on zero-sale rows (~78% — reconstructing offered price
from it drops the demand signal at shallow discounts); the source has no
episode ID, so the construction rule is persisted with the split manifest.
Only `bootstrap.prepare_data` accepts raw data.

### 5.3 Historical measurement — measure first, then build

Before any component was built, a measurement pass produced every value the
design needs but must not guess (cost-ratio distributions, same-hour price
variation, censoring shares, correlation). Two measurement outcomes were
designated in advance as design-changing (non-explorable catalogue → scope
shrinks; no identifying variation → prior falls back), decided by humans
against the measured numbers.

### 5.4 Baseline demand model — frozen, and blind to price

LightGBM/Tweedie predicts units sold per hour. Features: category,
subcategory, FC, hour of day, day of week, day of month, base price, two
point-in-time SKU demand rates, and a **single** price feature
**overwritten to the category reference discount at every inference call**.
Tweedie because hourly perishable demand is a zero-heavy count (~78% zeros)
with a positive tail. The overwrite is the load-bearing trick: the model
trains on confounded history but its price gradient is never queried; price
response enters exclusively through ε. **Frozen for the MVP window** so
posterior movement is attributable to learning, not drift; a
per-subcategory multiplicative level factor is always fitted and applied.
It cannot mask a slope error: factors are fit on anchor rows only, where
the price term is 1 by construction (§9.2).

**The SKU demand-rate features** supply per-SKU velocity, both
price-standardised (built only from *anchor hours* — stocked hours priced
within one tier of the reference) and point-in-time (lagged strictly before
the episode's first date):

- `sku_ref_sales_rate_30d` — trailing [t−30, t−1] anchor-hour rate at
  SKU × FC grain, with a SKU-pooled fallback (aggregated to SKU-day first,
  so no same-day cross-FC sales enter), NaN when even that is empty.
- `prior_episode_ref_sales_rate` — the anchor-hour rate of the same
  SKU × FC's most recent previous episode; NaN if it had no anchor hours.

**The training label is censored, and that is a priced trade.** The target
stops at the shelf. Dropping censored hours selects on the outcome (14% of
rows, concentrated where the selling happens); a censored Tweedie
likelihood is custom work the MVP did not buy. Chosen: keep the label and
correct the level. What makes it safe is the feature exclusion below: with
no inventory or stockout indicator the censoring bias arrives smooth, which
a multiplicative level factor can remove — **the level calibration is the
second half of this decision**, and why the factor must be solved on the
censored basis (§9.2). A proper censored objective is the first phase-2
item.

**Deliberately NOT features:** *hours remaining* (planner state, not demand
context); *within-episode lag sales* (post-treatment mediators of the
episode's own price path — part of the price effect routes through the
feature and no inference-time overwrite fixes it, because training already
attributed price response to the lag); *inventory, cost, stockout
indicators* (teach "low stock predicts low sales", the censoring artifact).
AGENTS rule 12.

### 5.5 Dispersion and correlation — frozen variance structure

Demand is negative-binomial: `Var = mu + mu²/r`. `r` is fitted per
subcategory by censored MLE on the **calib** window, with a fallback chain
(subcategory → category → global) and a clamp on implausibly-high converged
values. **An `r` at the search ceiling has two causes wanting opposite
treatment**: a thin group whose MLE wandered there wants the clamp; a group
genuinely steadier than Poisson also lands there (no NB can represent
Pearson < 1) and clamping it inflates the variance claimed for exactly the
cell with least. Pearson dispersion tells them apart; under-dispersed
groups are exempt and listed in `r_lookup.under_dispersed_groups` (a long
list indicts the NB family for the extract). An `r` within
`r_bound_tolerance_rel` of either search bound is stored but flagged in
`r_lookup.at_bound` (rule 3) and excluded from the clamp percentile, so a
thin extract with many pinned groups cannot talk the clamp up to the ceiling.

`rho` is one global scalar fitted against the model's own residuals **on
the calib window** (in-train rows understate it — the model fits its own
residuals — and an understated rho understates deff, which deflates every
posterior update). `dispersion.rho` must be re-pasted from
`artifacts/rho.json` after every retrain and after a prior change (the
working elasticity moves the residuals); a stale paste mis-weights every
posterior step silently, in the direction of slower learning.

rho is fitted with `common.config.intraclass_correlation`, the one-way ANOVA
ICC. `var(group means)/var(all)` — the form used before — estimates
`rho + (1−rho)/m`, because a group mean of m *independent* draws still varies
by σ²/m and that term reads as shared signal: on independent hours it returns
1/m (measured 0.164 at m=6, deff 1.82) and every posterior step was deflated
by that much pure estimator artifact.

**`m` is not pasted.** It is measured wherever deff is applied
(`common.config.deff_from_episodes`) as the forced outcomes per episode in
the batch at hand. The frozen key held the mean *length* of legacy episodes
whose discount changed — not forced hours at all — and the real quantity
moves with the exploration rate by construction, so freezing it guaranteed
drift. Measuring it closes that channel instead of alerting on it, and
`assurance.correlation` is then a check on the one thing still frozen: rho,
priced at today's clustering. Why NB:
hourly demand is far more variable than Poisson. Why `r` per subcategory
but `rho` global: dispersion differs by product type and the data supports
it there; a noisy per-category `rho` would inject noise into the
evidence-deflation factor. Why legacy data is legitimate here when banned
for elasticity: these are second moments, and the policy confound moves the
mean.

**What moves in production:** level factors weekly (§9.2) on anchor rows; ε
daily, human-gated (§5.11), on forced exploration outcomes only. `mu_ref`,
`r`, `rho` stay frozen — the two that move are identified by different
evidence and can be estimated without disturbing each other; the frozen
ones are what make attribution possible. `r`/`rho` are frozen because they
are second moments from weak signal, because re-fitting `r` inside the
learning phase reintroduces the ε ↔ r cycle §5.6 removed, and because
`fit_dispersion.drift_by_window` measures whether a shorter cadence is even
estimable (on the fixture, 14 of 17 weekly windows cannot fit `r` at all).
Consequence: `assurance.dispersion`/`assurance.correlation` are the sole
guard on `r`/`rho`; `drift_by_window` gives their thresholds a baseline
(`rho_spread_vs_alert` above 1 = the alert fires on ordinary variation).

### 5.6 Elasticity prior — the profile likelihood as a density

Per category, a censored **Poisson** log-likelihood is profiled over the ε
grid twice — *naive* (no time control) and *controlled* (same-`date_hour`
fixed effects across sku × fc, profiled out by moment matching) — on
**entry rows only** (§3.2). The deff-deflated curve becomes the prior as a
density: the 50/50 arm mixture, shrunk toward a pooled density built from
the right-signed categories. Poisson because the quasi-MLE is consistent
for the mean whatever the true dispersion — no `r` enters, so the prior
runs before the dispersion fit with nothing circular between them.

**No fallback constant.** A flat likelihood degrades to the uniform on the
support; a wrong-signed one (unconstrained peak at or above zero, searched
*past* the sign bounds) is discarded for the pooled density and named in
`wrong_sign_categories`; a **lower-pinned** one (the search extends
`unconstrained_search_below` past `epsilon_min` too, and a peak at or below
the bound that strictly beats every interior point means the likelihood ran
off the support — rule 3) is likewise replaced by the pool, excluded from
it, and named in `lower_boundary_categories` with a `boundary_note`
recommending that `epsilon_min` be widened (never `epsilon_max`); a flat
curve whose argmax merely lands on the first grid point is not pinned; the std is the widest of three measured floors
(density width, grid resolution, fold spread) and can never be zero — a
zero-width prior would freeze the posterior. `posterior.epsilon_max`
(−0.05) is a **sign constraint, never to be widened**: an estimate pinned
at the upper bound means the estimator found no negative price response —
a confound artifact, not evidence elasticity is near zero. Wrong-signed
categories are measured *backwards, not weakly*; only exogenous price
variation fixes them.

The specification (`bootstrap.prior_density`):

- **Estimator.** Per category and arm, the censored Poisson log-likelihood
  over the ε grid with frozen `μ_ref` as baseline; censored hours enter as
  `P(D ≥ q)`.
- **Density.** Each curve, deflated by an ε-free design effect **clustered on
  SKU × FC**, becomes `w(ε) ∝ exp(ll/deff)`. The cluster is the unit, not the
  episode: these are entry rows, one per episode, so a within-episode ICC is
  1.0 by construction and the deflation could never engage. What correlates
  between entry rows is the same unit recurring across days — the unit the
  pilot's outcomes recur on, and the one that does not average away. The category's own density is the 50/50 arm
  mixture, shrunk toward the pooled density (log-likelihoods summed across
  right-signed categories) with weight
  `own_information_weight = min(1, span/own_information_saturation)`.
  A category the data says nothing about gets the uniform — mean
  `(lo+hi)/2`, std `(hi−lo)/√12` — by construction. The search bounds are
  `[epsilon_min, epsilon_max]`, a policy statement about the range the DP
  supports; the lower bound was widened −4 → −5 (owner, 2026-09-05) under
  rule 3's asymmetric remedy, the upper never moves.
- **Thin time cells.** Controlled-arm cells with fewer than
  `min_rows_per_time_cell` uncensored rows get no multiplier — a cell
  fitted from a few observations absorbs the price response it is meant to
  control for and biases |ε| toward zero.
- **Read the artifact in this order:** `wrong_sign_categories` with
  `unconstrained_argmax`, then `holdout_comparison` —
  `log ∫ p(y_hold|ε)π(ε)dε` per held-out row on the calib window, bracketed
  by `oracle` and `uniform`, reading `information_available_per_row`
  (oracle − uniform) first; candidates below `uniform` are named in
  `worse_than_a_flat_prior`.
- **Tune `own_information_saturation` against production once** (shipped
  2.0, the chi-square 95% cutoff): nearly all categories clearing it means
  pooling never fires; nearly none means everything drags to the pool.
- **The acceptance gate is human** (§9.3): there is no reject path in the
  estimator — a category that fails to identify ε widens instead.

**Why honesty over sharpness:** with bounded steps (§5.11), a
confidently-wrong prior takes many update cycles to walk back; a weak
honest prior costs only patience. A pooled or uniform prior is the system
working, not failing.

### 5.7 The decision core — exact DP over a safe action set

Per episode, the feasible action set is every 2.5pp discount tier from zero
to `1 − cost/price` — the cost floor **is the boundary of the set**. Hourly
actions are restricted to tiers at or deeper than the current anchor
(price never rises within an episode — encoded in the transition, not
checked after the fact).

**Entry is a separate decision with a coarser action set**
(`pricing.entry_offsets`, `[−15, −10, −5, 0, +5] pp` relative to the
reference). Coarse arms concentrate the evidence: information about ε
scales as (log price ratio)², so fine arms near the optimum dilute the
uniform exploration draw. The deep side is bounded (one +5pp arm), because
under monotonicity a deep entry is irreversible. Offsets snap to the grid
and are filtered by the cost floor; if the floor forbids every arm, the
deepest feasible tier is the only action — correctly non-explorable, never
a fallback to the full grid. The value function is built over the full
grid, so the DP prices each entry arm knowing the episode deepens on 2.5pp
steps afterwards.

**The hourly action set is every tier deeper than the anchor** — hold, step
2.5pp, or jump straight to the cost floor. Whether the DP deepens is
economics, with a closed form: ignoring censoring, deepening reduces IL
only when

```
|ε|  >  (1 − d) / (γ − d)        γ = cost / price
```

On the corrected extract the measured median bar is |ε| = 2.429 (censoring
pushes the true switch higher). Whenever a cell's posterior mean sits below
the bar, the DP is structurally an **enter-and-hold** policy — not a
defect: if demand is that inelastic, holding *is* the IL-minimising action.
Three consequences stated out loud: the day-one IL advantage comes from the
entry choice and from not ramping; the clearance loss and scrap pressure
follow from this and are expected; **widening the action set cannot change
it** — only a posterior moving past the bar will, which is what exploration
is funded to find out (`intra_episode_deepening` in the backtest tracks the
gap). Enter-and-hold is a *result*, never a rule: every hour is a fresh
solve from the actual shelf with the price in force as the anchor, and
holding wins only while no deeper tier has a lower expected loss. The
backtest measures it directly as `intra_episode_moves` — the share of
episodes with at least one step on the DP arm's **own** path and the mean
steps per episode, overall and by cost-ratio band (`tuning.cost_ratio_bands`;
the bar falls as cost rises, so high-COGS shelves step on day one where
mid-cost ones hold), beside the legacy ramp's share and the share of
episodes above the bar. `pct_dp_deepened` is a different question — is the
DP's episode *mean* deeper than legacy's — and reads zero whenever the agent
is shallower on average, whatever it does within the episode. Shadow cannot
measure the agent's own steps: it re-anchors every hour on the legacy price
in force, so its `share_hours_recommending_deeper_than_legacy_price` is the
share of hours it would cut below the shelf's actual price.

Enter-and-hold does not starve the learner — the opposite. Information is
`mu · L² · r/(r+mu)` with `L` the log price ratio **against the
reference**: holding at `d_ref − 15pp` parks every hour at `L² ≈ 0.038`
where a 2.5pp wiggle at the reference yields ≈ 0.001. Shadow reports the
realised figure as `learning_yield_would_be`.

The planner solves the finite-horizon problem exactly:

```
Q(anchor, q, h, p) = Σ_k P(D=k | r, mu(p)) × [ min(k,q)·(−(P₀−p)) + V(p, q−min(k,q), h−1) ]
V(anchor, q, h)    = max over feasible p ≤ anchor of Q(anchor, q, h, p)
V(·, q, 0)         = −cost × q                       ← terminal scrap value
```

Exact DP because the state space is tiny (≤ ~20 tiers × ≤ 30 units × ≤ 12
hours) and exploration requires the full Q vector per tier. The demand
distribution is truncated at `negbin_max_k` with tail mass folded in and
emitted as a diagnostic.

### 5.8 Exploration — a P&L line item, uniformly randomized

The DP already prices every tier; exploration is a selection:

```
p*          = argmax Q(p)
cost(p)     = Q(p*) − Q(p)          ← expected IL sacrificed, in currency
affordable  = { p ≠ p* : cost(p) ≤ tau }
```

**δ_min — the smallest informative move.** The learner reads every forced
outcome against `mu_ref` at the reference discount, so the distance that
carries information is from the reference, not from p*: with
L = log((1−d)/(1−d_ref)) the signal is ε·L in log demand, against a level
bias in `mu_ref` of scale σ_b, **per category**: the category's own
surviving level error `|log by_category ratio|`, floored by two
catalogue-wide readings no category may sit below (the rolling-origin MAE
at the W in force, the gate half-width `log(1.1)`); `_default`, the larger
of that floor and the `by_category` rms, serves a category the backtest
never saw. One catalogue scalar under-floored the worst categories and
over-floored the best. Below
`δ_min = k·σ_b/|ε|` the move's signal sits inside the model's own level
error and the outcome teaches nothing about ε; shadow spent ~22% of
decisions there, all at one tier step. So the set is
`admissible = { p ≠ p* : |log((1−p)/(1−p_ref))| ≥ δ_min }` (cost is
measured from p*, information from p_ref — a tier far from p* but at the
reference costs budget and teaches nothing) and
`affordable = { p ∈ admissible : cost(p) ≤ tau }`; the ledger prices tau
against admissible tiers only, so tau still funds exactly the draws that
will be made. τ stays the one controller; `δ_min` is derived — σ_b is
MEASURED by `tune` (`exploration.delta_min_log_bias`), ε is the cell's
posterior mean at the decision, `k = delta_min_bias_multiple` (1). Fisher
information ∝ n·L² is conserved at a fixed budget; the case for large
moves is bias, not variance.

`admissible` is a subset of the DP's action set, which under an anchor
holds only tiers at or deeper than the price in force — so a forced move
is always a deeper discount, whatever δ_min or τ say; monotonicity is
structural, not a check.

If the affordable set is non-empty, the applied price is drawn **uniformly
at random** from it. Uniformity is the randomisation that makes outcomes
causal evidence; any state-dependent choice reintroduces the endogeneity
that poisoned the history. The spend is an explicit budget line — 1% of the
trailing 7-day mean of realised daily IL (a day's realised IL = the
whole-episode IL of episodes that *closed* that day, so the budget is a
share of a settled number) — scaled down (never below 25%) as the posterior
narrows. `tau` self-calibrates daily:
`tau_next = tau × clip(budget/spend, 0.5, 1.25)` — asymmetric because
halving is the safety direction.

**The daily loop:** at midnight, (1) yesterday's realised IL closes; (2)
today's budget = `budget_share_of_il` × trailing mean × the posterior-std
scale; (3) today's τ = `tau_next(yesterday's τ, budget, spend)`. **Both
sides are ONE DAY** — the spend of the day just closed against the budget
priced from the days before it. Comparing all-time spend against all-time IL
(as the controller once did) dilutes each day's correction by 1/N: the ratio
tends to 1 as history accumulates, so a day at ten times budget moves τ by
0.76× instead of the 0.5× clip, and because
`monitoring.exploration_cost_vs_budget` compares the same two numbers by
design, the backstop goes blind at exactly the same rate. `il_by_close_day`
(monitor) and `daily_exploration_spend` (update) are the one definition of
each side, both keyed by the decision's TRADING day
(`events.pairs.decision_day`) — never the outcome's UTC finalize time,
which for an hour-23 decision is D+1 and put the controller a day ahead of
the IL side. Zero realised spend on a priced day is under-spend, not
absence of signal: nothing was affordable, τ rises by the clip, and that is
the only way a τ cut below the smallest spread recovers. `pipeline.shadow`'s
controller trace walks the same arithmetic, so "would the pilot survive its
first week" grades the controller production actually runs — literally:
both call `explore.walk_tau`, one clipped step per closed day since the
last calibration (`tau_calibrated_through`), so a missed day is graded,
never skipped. τ moves on **spend**,
not evidence, so it needs no operator: `pipeline.update --calibrate-tau`
commits the walk daily, and `--apply` commits it too.
τ persists in the posterior artifact; `exploration.tau_initial` is only the
launch value, and a production caller reads `PosteriorStore.tau(cfg)` or τ
stays pinned at launch forever. Why budget-only rationing is sound:
information and IL cost of a perturbation both scale as
`mu × (log ratio)²`, so information per won is approximately constant —
there is no clever targeting to do. The launch value comes from shadow's
own derivation (§5.13); the backtest's exploit-only derivation is a
cross-check, not the source.

### 5.9 Posterior store — small, atomic, exactly-once

One record per learning cell — a Normal summary (mean, std), counters, a
version — plus the ledger of consumed outcome IDs, in one atomically-
written file. A Normal summary rather than a grid: the grid exists only
inside the update computation. **~10 category cells plus one pooled global
cell, not subcategory cells**, three reasons in order of force: (1) a finer
ε would change no price — the policy is insensitive to the mean anywhere
below the deepening bar; (2) evidence divides proportionally — split a
category three ways and each third takes three times the calendar; (3)
where history cannot identify ε per category it certainly cannot per
subcategory. Categories above `min_episodes_per_week_for_cell` earn their
own cell; assignment fixed at launch. Phase-2 fix, if within-category
variation shows up, is partial pooling, not a threshold ladder.

**The launch belief** (`pricing.posterior.launch_belief`): each cell
launches at the prior mean pushed `posterior.cold_start_shift_std` prior
stds toward more elastic and clipped to the ε range, std untouched. An
owner's risk posture (0.5 at 2026-09-05; 0 is the prior as measured): a
steeper day-one belief moves |ε| toward the deepening bar, buys clearance
and pays discount if the prior is right, and the push is largest exactly
where the prior is widest — where the data cannot object. The std is
unchanged so evidence weighs the same, and the bounded step walks the mean
back at up to `max_mean_step` per update if outcomes disagree. The cell
keeps `prior_mean` for audit. The same function sets the backtest's DP-arm
belief (§5.14), so the launch record grades the policy that will run, and
the deck's cold-start belief. The key is read at `init_posterior` only:
`advance` re-initialises the file while it holds no consumed outcome and its
cells differ from what init would write now (`launch_stale`); once an
outcome is consumed the learner owns the mean and the key is inert. The ledger
lives inside the posterior file because exactly-once learning requires
"revision applied" and "outcomes consumed" to commit together — one file
renames atomically, two cannot.

### 5.10 Inference and the event contract

The source-data conventions everything below stands on (the write-off
sentinel, shrink, the counter, percent discounts) were inferred from the
extract, not decreed: they are registered as claims in
`docs/event_contract.html` §01, owned by the data's producers, and a
corrected claim is a code change here, never work on their side.

Validation checks nine invariants and **rejects the state rather than
returning any price** — the worst failure is a confidently wrong price, not
"no answer". Outcomes are constructed by `pipeline.ingest_outcomes` from
the hourly FLC feed (matched by SKU, FC, date, hour; `adjustment_reason`,
`is_stockout` and the offered price derived), so the integration surface is
the price request, applying the price, and a failed-push report. Every
decision emits ~30 fields: the full pricing context, the
exact prediction, posterior moments, exploration flags and cost, and the
versions of model, posterior, config. Learning replays evidence from
events, never recomputation, and any decision must be reproducible from its
event alone. The store is append-only JSONL with duplicate detection;
malformed events are **quarantined with their validation failures
attached**, never silently dropped.

### 5.11 Learning update — censored, deflated, bounded, gated

A daily batch consumes **exploration outcomes only**, evaluates the
censored likelihood on the grid, adds the current posterior, normalises,
takes moments.

- **Censoring.** A stocked-out hour enters as `P(D ≥ inventory)`; treating
  it as exact systematically understates elasticity where it matters.
  Zero-sale hours are retained through `P(D = 0)`.
- **Exploitation outcomes are discarded.** Exploitation prices are chosen
  *by* the posterior; learning from them feeds beliefs back to themselves.
  Off-policy correction is phase 2.
- **Correlation deflation.** Hours within an episode share an inventory
  pool, a demand shock, and a monotone price path; accumulated information
  is divided by `deff = 1 + (forced_hours − 1) × rho`. Without it the
  system declares convergence several times too early. Strict start-up
  refuses when config disagrees with `artifacts/rho.json`.
- **Evidence is banked until spent.** The batch is every eligible outcome
  not yet consumed by a revision; outcomes are marked processed only when a
  revision commits. `batch_oldest_outcome_age_days` makes a batch that
  grows without firing visible.
- **Every batch grades the belief before updating it.** The outcomes
  arrived after the current posterior was set, so their log marginal
  predictive under the pre-update posterior is an out-of-sample score:
  `predictive_check` per cell, bracketed by `oracle` and `uniform`.
  `worse_than_a_flat_prior` persisting across batches is the signature of a
  posterior that tightened faster than the evidence justified. Read the
  differences, never the absolute values.
- **Bounded steps, human-gated.** An update applies when effective
  information crosses `learning.information_increment`; each step moves the
  mean at most `max_mean_step` and shrinks the std at most
  `max_std_shrink` (floored at `min_std`), clipped bounds flagged. **The
  increment is MEASURED**: Fisher information adds to precision, so
  `I* = (1/s₀²)·[1/(1−max_std_shrink)² − 1]` saturates the cap — a ceiling,
  not a target (excess is discarded). Derive it at the launch stds; it
  becomes conservative as the posterior narrows, the safe direction.
  **The two rails are one decision expressed twice**: a cap-sized update
  moves the mean by `[1 − (1−max_std_shrink)²] × |pull|`, so the rails
  should trip at the same surprise; `bounded_step_recommendation` reports
  which binds first, and `backtest.step_sensitivity` prices the step on
  real episodes (re-solving the DP at ε ± step). A human approves each
  update, one per `learning.update_cadence_days` (1: daily). Daily is
  deliberate — each cell triggers on its own batch, so a fast category
  updates the day it has the evidence while a slow one waits; a longer
  cadence only delays the fast ones and discards their surplus
  (`learning_yield_would_be.bounded_updates_worth_per_period`). A human
  approves each
  day's update; the apply refuses while event-quality gates fail
  (duplicates/unmatched > 1%, price mismatch > 1% — compared on exact
  counts, never the rounded rate the report prints) or when
  `calibration_current` finds the factor schedule no longer reaches the
  week being priced (a missed weekly re-fit silently reverts production to
  stale factors).
- **No `information_since_update` counter — re-adding one is a bug.** The
  trigger reads the UNCONSUMED BATCH; a counter double-counts outcomes that
  are re-read next run.

**What `--apply` moves:** posterior mean/std/version on INFORMATION (only
when a cell crosses the increment); `processed_outcome_ids` with the
revision, same atomic write; `tau` on SPEND, every run.
`bootstrap.init_posterior` refuses to overwrite without `--force`.

### 5.12 Monitoring — three families, three questions

Business (is IL improving — ratio-of-sums IL% with denominators, absolute
IL alongside, by category/FC); learning (is the posterior moving — per-
cell trajectories, forced counts, spend vs budget, affordable-set-empty
rate, current τ); safety (is the event pipeline healthy — match/duplicate/
mismatch rates, quarantine, latency). A posterior std flat for 21 days
alerts directly; `affordable_set_empty_rate` is the leading indicator of a
non-explorable catalogue; `realised_vs_predicted_sold_ratio` is the daily
continuation of the calibration diagnostic. Stop conditions (cost-floor
violation, event-quality breach, mismatch, realised spend > 2× the day's
budget, scrap/margin deterioration — the last three over `persistence_days`
consecutive priced days, because one day over is a thin-IL day or two
expensive draws and the τ controller halves τ on it the next morning) **suspend
exploration — exploitation pricing continues**: the monitor writes
`exploration_suspended` into the posterior state, `decide` selects with no
budget (no draw) while it is set, `status` reads WARN with the reason, and
only a human lifts it (`pipeline.update --resume-exploration`). Nothing
auto-resumes: a stop is a finding to investigate, not a transient. The
guardrail series is keyed by an episode's **close day**, so the newest
days count every episode that closed rather than only the early
sell-outs, and floor and trigger read the same, unbiased series.

### 5.13 Shadow harness — full rehearsal at zero pricing risk

Runs the complete production decision path against live data while legacy
keeps pricing: state from reality (the anchor entering hour *t* is the
legacy price from *t−1*), full decision events logged, outcomes stamped
ineligible for learning. Exit gate: event completeness > 99% (outcomes
accepted per decision emitted; the gap is quarantine plus duplicates) and
**zero** cost-floor violations.

**The hold-out run is the DEFAULT.** `data.holdout` names a window after
`test_end` that no artifact was fit on and no gate was decided on; shadow
runs it with no flag. Every other window grades something fitted to it.
`--all` sweeps the whole extract and stamps `shadow_gate.in_sample_caveat`
naming which numbers it flatters (drift ratio, `tau_recommended`, learning
yield) and which it does not (completeness, cost-floor — plumbing). A missing `data.holdout` is an error, never a silent full run.
The hold-out is **one-shot**: tune a value on it and it becomes a second
calibration set. Date cuts are episode-scoped
(`common.episodes.window_slice`, by the date a window opened).

**Sampling.** Shadow draws a uniform episode sample
(`monitoring.shadow_gate.sample_episodes`, default 3,000; `--max-episodes
0` = everything, for the final pre-launch record). A sampled report adds
`sampling_caveat` — quote it with the zero violation count (a crash once
passed on a sample). Sampling degrades exactly one figure: the
`tau_controller_trace.by_day` series; the pooled `spend_over_budget` is
sample-invariant. Each episode draws from its own RNG seeded by episode id,
so results are identical serial or parallel.

**Deriving `tau` where it will actually run.** Shadow derives its own
launch tau (`derive_tau0`): the bisection run on the run's own **anchored**
path over the trailing `budget_il_window_days` before the window — the
exact span the day-one budget base reads. This kills both staleness modes
of a config paste: an old backtest's number, and the exploit-vs-anchored
path mismatch (affordable sets measured ~1.66× apart). The pre-window week
is out-of-window for the run, so day one of the controller trace is an
out-of-sample test of the launch value. Too-thin weeks fall back to the
`exploration.tau_initial` paste, behind
`pricing.explore.tau_provenance_error` (refuses a paste with no source or
that no longer matches its derivation). Shadow also reports
`tau_recommended` (the bisection pooled over the whole window — a
cross-check) and `tau_controller_trace` (day-by-day walk: `tau_next` reads
only the day just closed, so a tau 8× too generous suspends exploration
before the controller can correct). The trace seeds its trailing-IL base
with pre-window closed-episode IL on the **dp_eligible population**, scaled
to the sample — the same population every figure it is scaled against
(`frac`, `seed_scale`) counts, or the day-one budget is inflated by the
ineligible episodes' IL. The budget base — seed and in-window — is
`common.metrics.episode_economics` over every observed hour: the same
scrap and IL the guardrail floors and the monitor read, never a local copy.
`daily_budget` is the mean of `budget_today` over the window's **decision
days** (`ledger.days`, the days the controller trace walks), never over the
seed days, whose first day has no trailing history and a budget of exactly
zero. Every per-day figure divides by one `n_days`: the calendar span of
the unsampled, unextended window frame (a sample can shrink the span; the
window extension adds a next-day row with no decisions). Shadow freezes the
calibration at the window start (`freeze_calibration_from`) before
predicting, so `calibration_regimes.frozen_anchor` is the anchor by
construction on any window, and `calibration_coverage` reads the deliberate
freeze as OK rather than STALE. `tau_initial_derivation.days` is the count
of trading days the ledger and the IL mean share.

**The forced rate is the budget's, and the sweep says what a change
buys.** The chooser explores whenever τ affords an admissible tier, so
`forced_rate = 1 − affordable_set_empty_rate` exactly and the rate is set by
τ, i.e. by `budget_share_of_il`; `delta_min_bias_multiple` sets *which*
tiers are drawn, not how many. Shadow's `exploration_budget_sweep` re-solves
its own spread ledger per (share, multiple) — shares at ¼, ½, ¾, 1 and 1.5×
the one in force, multiples at 1, 1.5 and 2× — and reports for each the τ
the budget bisects to, the forced rate, spend, mean log move and
`information_rel`, the sum over forced decisions of E[move²] relative to
the in-force pair (the NB Fisher information is quadratic in the move, so
this is the count-and-depth trade in one number; a proxy). Lower the share
to force less at the same depth; raise the multiple to force less but
deeper — the latter can *raise* total information at the same budget. The
ledger records only tiers admissible at the floor in force, so multiples
below it are reported as unrecoverable, and with `delta_min_log_bias` null
every multiple reads the same. The owner picks from the table; only the
chosen pair needs a shadow re-run.

`realised_vs_predicted_sold_ratio_at_legacy_price` is the production
continuation of the calibration diagnostic — the first place frozen-
baseline drift shows. `calibration_regimes` reports it under both the
frozen anchor and a weekly re-fit (§9.2).

### 5.14 Replay and threshold derivation — evaluation discipline

Offline replay has three jobs: the calibration diagnostic, tau derivation
cross-check, and planner sanity. Its policy comparison is **like-for-like
by construction**: both the legacy path and the DP path are simulated under
the same frozen demand model and prior, so model bias cancels. Comparing
observed reality against model-simulated outcomes charges all model bias to
one side; observed-vs-model differences belong to *fidelity*, never the
policy verdict. Even like-for-like, **replay output is never evidence the
policy works** — the shared model's price response is an unvalidated prior;
the pilot's own outcomes are the evidence (§11).

**Three rungs, not interchangeable.** Replay is the agent against our model
of the world; shadow is the same machine against the world itself; only
applied prices answer whether the advice is better. Shadow says strictly
*less* about the policy — no price was applied, so no IL figure exists in a
shadow run at all. Never "replace replay with shadow".

The replay's headline (production, corrected extract): the DP arm shows
~38% less IL than the legacy arm like-for-like, at ~1pp of clearance,
by opening far shallower and holding — which is what §5.7 predicts of an
enter-and-hold policy. Pre-committed consequences: the pilot reports
clearance and scrap alongside IL from day one, and a scrap-guardrail breach
driven by this mechanism is a business decision about the IL/clearance
trade, not a system fault.

### 5.14a The frozen artifacts are one bundle

The artifacts are fitted in sequence and only meaningful together: `rho`
deflates evidence against one model's residuals, the level factors correct
that model, the prior was estimated from its predictions and that
`r_lookup`. Mixing vintages raises no error — the numbers just stop
describing the same world. **The bundle id is the baseline model version.**
Each artifact carries a `provenance` block; `bootstrap.seal` writes
`artifacts/bundle.json` with a SHA-256 of every file and refuses an
inconsistent set. Re-run `seal` after `--fit-calibration`.

Stale *reports* are the other half: after a retrain, yesterday's
`backtest.json` still parses and silently grades a ghost model. Every
gate-feeding report stamps `artifact_versions` **and a `config`
fingerprint** — the phase it belongs to (`backtest` / `shadow` /
`production`), a digest of the whole config it read, and the full snapshot.
`status` compares them against disk: model mismatch is FAIL; a config key
**the report reads** that has moved since is WARN and names it
(`tune.stale_keys` routes moved keys to the reports they invalidate, and
`advance` re-runs by the same table: W turns the loop; `delta_min` re-runs
shadow; a stop threshold, `max_std_shrink` or `information_increment`
re-derives thresholds; `max_mean_step` re-derives thresholds AND re-runs
shadow; keys tune does not paste but one report reads are routed by prefix
(`tune.READ_BY`: the budget and controller knobs to shadow, the guardrail
window/smoothing/persistence to thresholds); runtime-only knobs are inert;
an unclassified edit re-grades everything; a MEASURED paste that writes
back what a report measured is inert. The classes do not nest, so a paste
of several keys re-runs the union, and a per-category mapping is matched on
the LONGEST `KEYS` prefix — a category re-round of `delta_min_log_bias`
diffs as `…delta_min_log_bias.MEAT` and is shadow's, not the loop's. The
backtest's exploration ledger reads `delta_min` too, but no pasted value
comes from it, so it is not re-graded; `calibration_gate_band` is an input
to the W sweep yet stays inert on purpose, or every band paste would reopen
the oscillation the hysteresis exists for). Treating every digest change as staleness made
`advance` re-run shadow after each tau paste and chase the fixed point for a
day; and the rho paste tolerance must sit above the ~1e-3 step a
`--check-only` turn takes while the loop contracts
(`dispersion.rho_paste_tolerance_rel`, 1% of the frozen ρ — tighter than
τ's 5% because ρ is frozen for the pilot and divides every unit of
evidence, while τ self-corrects daily), or every settle is a new paste. The
snapshot is also the answer to "what config was in force for each phase" —
read `reports/<name>.json → config.snapshot`. `meta.config_version` stays as
a human label only; nothing depends on anyone remembering to bump it.

**The operating instruction:** run `python3 -m pipeline.status` before
quoting from any report and before ending any session that touched
artifacts, config, or reports. Never quote or paste from a report a
freshness line calls stale. The re-run map:

| after changing | re-run | re-paste |
| --- | --- | --- |
| baseline (retrain) | `bootstrap.run` (whole loop) → `shadow` | `rho` |
| elasticity prior | `fit_dispersion` onward (§5.5) | `rho` |
| a config tunable a report reads | that report onward — `status` names the moved keys until you do | — |
| the extract | everything from `prepare_data` | — |

### 5.15 Production assurance — testing the assumptions, not the code

The unit suite checks logic against fixtures; what has actually broken the
system every time is an assumption about real data. `pipeline.assurance`
runs daily and tests the frozen artifacts against the live world:

| Check | Question | Why nothing else catches it |
| --- | --- | --- |
| `reproduction` | Do logged decisions re-solve to themselves? | The DP is deterministic; a mismatch means something moved underneath it |
| `dispersion` | Is live demand as lumpy as frozen `r` claims? | Every bounded update assumes it, and no business metric moves |
| `correlation` | Is **`deff`** still the frozen value? | It divides all accumulated evidence |
| `config mirrors reports` (status) | Does every DERIVED config value still match the report that derived it? | `artifact mirrors` covers only artifact pastes; a report-derived value could be stale, foreign, or the repo's shipped fixture number and nothing refused it |
| `exploration` | Is the applied price a uniform draw from the affordable set, reconstructed with the decision's own `delta_min`? | The causal claim rests on it entirely |

Two thresholds here are set on the quantity with the *consequence*, not the
one that is easiest to measure. **`correlation` judges `deff = 1 + (m−1)ρ`,
not `rho`**: deff drifts through both terms, and a rho-only verdict was blind
to the forced-hours channel — `m` moves whenever the exploration rate does,
rescaling every update while `rho` sits still (`rho_drift` remains as the
diagnostic saying which term moved). **`exploration` needs significance AND
effect size**: χ² power grows with `n` and the event store is append-only
with no window, so a p-value alone tightens every day the system runs — at
100 draws it takes a ~47% bin deviation to FAIL and at a million, ~0.5%. The
same draw distribution would pass in week one and fail at volume with nothing
about the draw having changed. `uniformity_max_bin_deviation` is scale-free
and carries the meaning; the p-value only stops noise being called bias.

Details that carry the value: the decision event carries `mu_ref_path` and
`anchor_discount` so it is *sufficient* to re-solve (never remove an event
field because "nothing reads it"); the dispersion check uses the two
statistics that survive censoring exactly (`P(sold=0) = P(D=0)`,
`P(sold≥q) = P(D≥q)`), binned by predicted `mu` so a shape problem indicts
`r` and a level problem does not; `rho` is re-measured on the basis it was
frozen on (the working elasticity via the shared
`fit_dispersion._working_elasticity`). None of these suspend pricing —
they are read at the operator gate. Thin windows report `INSUFFICIENT`,
never `PASS`, and the report's top-line verdict stays `INSUFFICIENT` until
every check has run.

## 6. Data foundation

**Source:** hourly FLC snapshots (date, hour, SKU, FC, inventory, discount,
units sold, base price, realised price, cost, remaining window, category,
subcategory). A known bad-demand window is excluded. Split boundaries and
the episode-construction rule are persisted in
`artifacts/split_manifest.json`.

**How to size the split.** The four windows are not symmetric; sizes follow
from the dependency graph. `test` and `holdout` are pure grading windows —
nothing may be fit on them. Every residual-based quantity (`r`, `rho`, the
level anchor, the prior's *selection*) must live in `calib`, because
residuals on `train` are in-sample. That makes `calib` the scarce resource.
Rules, in order:

1. **`calib ≥ 2 × calibration_fit_trailing_weeks`.** The first W calib
   weeks carry factors fit partly on train rows (biased low); only the back
   half is wholly out-of-train.
2. **W from the rolling-origin sweep** (`calibration_window_sweep`). If the
   sweep wants a W that violates rule 1, revisit the split, not the band.
   Every row — `uncalibrated` included — is scored on **one common set of
   evaluation weeks** (the longest candidate's burn-in). Per-window burn-in
   judged a long window on a later, smaller sample than a short one, so the
   ranking read *which weeks* rather than which window. `uncalibrated` is
   ranked with the windows: when no-factors wins, `verdict` says so and
   `tune` raises it as an INFO reading. **The ranking is not the evidence.**
   It compares aggregates over ~10 weeks and then turns on a lexicographic
   tie-break, so a one-week difference in `share_weeks_in_band` can decide
   which window "wins" while it loses on error. `paired_vs_uncalibrated`
   asks the question that matters — on the *same* week, did the factors move
   the anchor ratio closer to 1 — with a sign test over the paired weeks, and
   `calibration_earns_its_keep` reports UNDECIDED when nothing separates.
   Pairing removes the between-week variance that swamps the aggregate
   comparison; a level model already near 1 at the anchor has little bias to
   remove and the factors mostly contribute estimation noise. It is never a paste — W = 0 is not a
   config value — so `recommended_fit_window` stays the best *calibrated*
   window, and whether level calibration earns its keep at all is an owner
   call on the design.
3. **`test` = 2 weeks**: one week carries the full weekly swing; beyond
   two, the frozen anchor only gets staler.
4. **`holdout` ≈ 3 weeks**: enough for the tau-controller walk and several
   posterior updates; the most recent regime.
5. **`train` takes the remainder.** It may straddle the exclusion gap
   (episodes whole); `calib`/`test`/`holdout` must not, and all three sit
   contiguous at the extract's end.
6. **`test_start` on an ISO Monday** keeps the anchor window and the weekly
   schedule on one week grid.

After any split move: check rows-per-subcategory in the new calib vs
`dispersion.min_rows_per_group`, re-read the sweep, and get
`--check-convergence` green. A split change moves `train_end`, so it is a
full retrain — nothing from the old split is comparable (rule 1).

**Synthetic validation:** `tools.make_dummy_flc` reproduces the schema with
known ground-truth elasticity in two modes — `legacy` (the clock confound;
estimators must *detect* it) and `randomized` (identifiable; estimators
must *recover* it). The suite runs the full pipeline against it.

## 7. What running on production data changed

Five findings, each caught by a gate doing its job: the fidelity gate is
read on launch-adjacent windows with per-week ratios (regime structure
visible, not averaged); level factors are fit on anchor rows only (slope
error cannot contaminate them); entry-rows-only identification (§3.2);
per-SKU velocity features (better per-row accuracy, ~18% more information
per outcome via lower deff); and weekly demand volatility is a measured
fact — the gate band was set by the owner at ~2σ of measured weekly noise.
Full history: `docs/learnings.md`.

## 8. Measured results (production data, model `baseline-20260809120225`)

Historical record of the owner's production run after the §12a corrections;
re-measure before quoting any figure against a newer extract.

| Quantity | Value |
| --- | --- |
| Calibration diagnostic | `level_bias_at_anchor` 1.0389, inside the owner band [0.90, 1.10] |
| Shadow gate | PASS — 12,771 decisions, completeness 0.9974, zero cost-floor violations, drift ratio 1.0225, solver p95 102 ms |
| DP vs legacy (like-for-like) | −38.0% IL at −0.97pp clearance |
| Intra-episode deepening | 0% of episodes; median \|ε\| needed 2.429 *(bracket-era prior — re-measure)* |
| Guardrail 3σ noise, trailing basis | margin well behaved; scrap outlier-dominated and unusable on that basis (§12) |

## 9. Evaluation gates

### 9.1 Why gates instead of judgment

Certain results must block the build rather than parameterise it, and the
blocking conditions were fixed before the numbers were seen.

### 9.2 Calibration — always applied, with the level as a diagnostic

**Calibration is not a gate.** The level factors are always fitted and
always applied (fit on anchor rows only, so slope error cannot contaminate
them; with no artifact on disk every factor is 1.0). The anchor-level band
is a reported **diagnostic**: out of band means drift or staleness to
investigate, surfaced as WARN in `pipeline.status`, never a launch blocker.

The diagnostic judges the frozen model on its only production
responsibility: **the demand level at the reference discount** (inference
always overwrites price to the anchor; every other price comes from
`mu_ref × ratio^ε`). The metric is `level_bias_at_anchor` on the test
window, within `calibration_gate_band` — sized from measured weekly demand
volatility, and it may only ever **tighten** (clamped in RATIO space to
`tuning.calibration_band_max_half_width`; `exp(0.10) = 1.1052` would breach
a log clamp). A pooled-at-actual-prices gate was rejected: it structurally
measures the one quantity history cannot identify (the slope), and would
block launch forever on the prior rather than the model. The pooled ratio
and `slope_ratio_by_discount_gap` remain diagnostics; the continuous
production guard is the daily `realised_vs_predicted_sold_ratio`.

Level factors are fit on anchor rows over a window **disjoint from the gate
window**, and on the **censored basis**: sales cannot exceed inventory, so
predictions compare as `E[min(D, q)]` — the gate's quantity. A factor fit
on raw `mu` (always the larger number) reads systematically low, the wrong
side of 1. Because the factor scales `mu` *before* censoring, it is solved
by bisection, not divided out.

Temporal semantics: a monotone anchor-ratio *trend* across gate weeks means
the level is in motion and the gated model is stale — a staleness reading,
not a band-tuning problem (`anchor_ratio_by_rate_history` triages: new-
assortment SKUs with no rate history predicting low is assortment, not a
macro trend). The level factors are re-fit **weekly** on the trailing
window; the frozen model — and with it price response and attribution —
stays frozen; only the level multiplier tracks the world.

**Point-in-time factors.** The factors are re-fit every week on the
trailing `calibration_fit_trailing_weeks`, on the window *ending strictly
before* that week, applied **by row date** — no row is ever priced by a
factor fitted on its own week or later, and the harnesses grade the
mechanism production runs. The schedule lives in
`calibration.json → schedule.by_week`; `factors` is the fallback for weeks
before the first trailing window closes, and an unfitted week holds the
fallback rather than borrowing a later week's. Both harnesses read it
through `BaselineModel.predict_mu_ref` alone.
`schedule.weeks_on_partial_window` names weeks fitted on less history than
the label claims (extract start, post-exclusion-gap); they are flagged, not
dropped — holding them at the fallback would price them on a factor fitted
*later*, the leak this mechanism exists to prevent.

**The gate freezes; the schedule does not.** Two questions share one
artifact:

| | question | calibration |
| --- | --- | --- |
| launch gate | does the artifact **as frozen** reproduce hold-out sales? | anchor, frozen at `gate_start` |
| mechanism | does frozen model + **weekly re-fit** track demand? | weekly schedule |

The schedule runs through the whole extract (production re-fits weekly and
a forward replay at week *k* legitimately reads weeks < *k* — no
look-ahead). But a factor re-fit *inside* the graded window has read the
rows it is graded on, so `backtest.replay.fidelity` calls
`model.freeze_calibration_from(gate_start)` and prices the graded window
off the anchor, reporting the mechanism reading beside it as
`fidelity.weekly_refit`. The spread between them is what weekly re-fitting
buys. `schedule.gate_freezes_at` records the boundary, and
`calibration_coverage()` counts frozen rows apart from fallback rows — a
deliberate freeze and production running past its last fitted week must
never share a verdict.

**Shadow reports both calibration regimes.** The artifact's schedule stops
at `test_end`, so every hold-out row falls back to the frozen anchor —
"launch and never re-calibrate". `pipeline.shadow.weekly_refit_schedule`
re-fits the factors per shadow week (trailing window ending strictly before
each week, fitted in shadow rather than the artifact so the pre-launch
bundle stays clean of hold-out rows — rule 16) and `calibration_regimes`
carries `frozen_anchor`, `weekly_refit`, and their `spread` on the same
rows. Reading the pair: both near 1.0, the level held; only the frozen one
off, the anchor went stale and weekly re-fitting earns its keep; both off,
the level moved faster than a weekly cadence can track — a retrain, not a
cadence change.

**What a factor change does to the prior and dispersion.** Both are fitted
against `mu_ref`, so the chain re-runs (§5.5, §5.6). The dispersion side is
a formality (1–4% on the fixture) — and the intuition that a level
correction should cut `rho` is wrong: `rho` is *within-episode* residual
correlation, and a per-category weekly factor rescales an episode's
residuals uniformly, so it cannot remove the correlation between them.

**The loop is circular by construction, and convergence is asserted, not
assumed.** The factor solve consumes `r` (the censored basis), while `r`,
`rho` and the prior are all fitted against *calibrated* `mu_ref` — a cycle,
broken by iteration. It converges because the loop gain is small in both
directions (`r` enters `f` only through the censoring correction; `f`
shifts the level while `r`/`rho` measure second moments). Convergence is a
NUMERICAL property, in-sample by construction — "would this hold on unseen
data" is not a meaningful question about a fixed point, and **converged is
not correct**: out-of-sample validity is answered by held-out instruments
(`level_bias_at_anchor` on `test`, the rolling-origin sweep, shadow's
`calibration_regimes`), and convergence is the precondition that makes
those three mean anything.

In production the loop does not turn (only calibration and ε re-fit; the
model, `r`, `rho` stay frozen — a one-directional `f ← r`).
`pipeline.assurance` watches the decay on live outcomes. What re-turns the
loop is a retrain of the prior or dispersion; the `convergence` block
records digests of the artifacts it was checked against so `status`
reports the verdict as STALE once any of them moves.

**The loop is the slowest thing in the pipeline** (production: hours), so
`bootstrap.run` cuts the costs that buy nothing: `estimate_prior --fast`
skips `fold_spread` on loop turns (it only widens the std FLOOR, while the
loop compares factors, which follow the prior MEAN), and
`--commit-convergence` keeps the check's re-solve — turn *k*'s check
computes bit-for-bit what turn *k+1*'s `--fit-calibration` would, so 3b
runs on turn 1 only. The artifact gets a full prior once settled, and the
default `--check-convergence` stays a dry run.

`train_baseline --check-convergence` re-solves the factors with the
prior/`r` now on disk, compares per cell and per schedule week in log space
against `calibration_convergence_tol_log`, then restores the artifact (a
dry run — committing while prior and dispersion lag it would create the
inconsistency being tested for; `convergence.method` names which of the
two ran). A cell whose bisection pinned at `calibration_factor_search_bounds`
carries `detail[cell].at_bound` (rule 3); thin cells are shrunk toward the
parent by `calibration_shrinkage_units`, never "left at 1.0" — only a whole
window under `calibration_min_anchor_rows` is. A failing `--check-convergence`
stops `bootstrap.run`'s loop with its own message rather than iterating to
`--max-turns` with a stall test that can never fire. The verdict lands in
`calibration.json → convergence` (status row is WARN — chain health, not a
launch gate).

**Read the TRAJECTORY, not the turn count.** The owner measures **8–9
turns** from a bare chain on the production extract with nothing wrong; the
repo fixture settles in 3–4 because it is small — never size a cap or an
impatience threshold on the fixture. The block carries a `history`; a
contracting series simply needs another turn. `bootstrap.run` stops only
after three turns with no new best (a two-turn plateau inside a nine-turn
settle is ordinary). `worst_cell_anchor_rows` sizes the evidence behind the
worst cell: the max is unweighted, so a thin shrinkage-dominated cell reads
identically to an unsettled loop unless the row count is shown.

**The tuning loop is a program** (`pipeline.tune`): each check names the
report field it reads, and the CLASS decides who may act — **PASTE** (a
MEASURED value; `--apply` writes it), **OWNER** (never auto-applied;
reported with evidence), **READ** (no config key), **BLOCK** (an invariant
that must hold first: reports from one model, a settled loop,
`calib ≥ 2W`, no graded window across the exclusion gap — a BLOCK
suppresses everything and `--apply` refuses). *A value the data can decide
does not wait on a human*: the guardrail stops (3σ floors), the fit window
W (the sweep, subject to `calib ≥ 2W`, **with hysteresis** — W is the one
paste that turns the calibration loop, and re-settling re-scores the
sweep, so a strict argmin oscillates near-tied windows and loops an agent
on apply → check-only; it switches only on a material win), and
`max_mean_step` (the consistent-rails value) PASTE — `max_mean_step` behind a gate on its
measured price consequence (`tuning.max_price_share_changed_for_auto_rail`,
`max_il_delta_pct_for_auto_rail`), downgrading to OWNER with the reason
when exceeded.

**What remains OWNER** — *a number that encodes what you are willing to
lose, wait for, or risk*: `budget_share_of_il` (risk appetite — the forced
rate is its consequence; shadow's `exploration_budget_sweep` prices the
alternatives); `learning.max_std_shrink` (which rail moves is a safety
posture — `tune` computes both numbers, decides neither); and `data.split`
(§6 gives sizing rules; how much history still represents the business is a
market judgment).

**`--apply` names the MINIMUM sufficient re-run**: `none` for values read
at runtime or mirroring an artifact; `calibration` for
`calibration_fit_trailing_weeks` (the loop turns — `bootstrap.run
--check-only`, **no retrain**); `retrain` only for `data.split`, which is
OWNER, so `--apply` never writes one. It backs up the config and appends
what was written, its source field, and outstanding owner decisions to
`artifacts/config_decisions.json`. Edits are targeted line replacements —
the comment beside each value is the reasoning and a YAML round-trip drops
them. Iterate until `tune` reports no PASTE and no BLOCK.

**Triage when the level diagnostic is out of band:**

```
├─ FIRST fidelity.by_week — wobble vs trend:
│  · wobble wider than the band: week-scale volatility. OWNER decision —
│    longer gate window or wider band; no retrain can pass it
│  · monotone trend: the level is in motion, the model is STALE — check
│    anchor_ratio_by_rate_history first (assortment vs macro)
├─ by_window shows train ≉ calib/test → regime drift: later train_start,
│  then retrain (a fresh baseline — restart comparisons)
├─ level_bias_at_anchor far from 1, flat slope → re-fit the factors
│  (--fit-calibration), NO retrain — the factors are stale, not absent
├─ anchor ≈ 1 but slope degrades with gap → re-run estimate_prior; a
│  pooled/uniform prior is a valid outcome
└─ far out of band AFTER a re-fit → the model itself is stale: escalate
   (retrain decision)
```

### 9.3 Prior-acceptance gate (blocking) — is the prior honest?

A human reading of `prior.json`, not a flag in it: `wrong_sign_categories`
(peaks at positive ε — discarded for the pooled density), `std_basis` per
category (which measured floor set the width), and the `holdout_comparison`
against `oracle` and `uniform`. A pooled or uniform prior is a designed
outcome: history that cannot identify ε says so and hands the job to
exploration.

### 9.4 Shadow gate (blocking) — is the pipeline production-ready?

Event completeness > 99% and zero cost-floor violations, before any price
is applied. The verdict "proceed to
exploit-only pilot" is the phase-2 entry condition, not permission to apply
prices in phase 1.

## 10. Launch plan

| Phase | What happens | Exit gate |
| --- | --- | --- |
| 0. Measurement | historical measurement, config populated | gates reviewed |
| 0b. Calibration | level factors fitted + applied; prior estimation | level diagnostic reviewed (9.2); prior gate (9.3) |
| 1. Shadow | decisions logged, no prices applied | §9.4 |
| 2. Exploit-only pilot | small SKU set, exploration off | price mismatch < 1%, finalization SLA |
| 3. Learning pilot | exploration on at the configured budget | posterior std falling; spend within budget |
| 4. Pilot read | §11 | the pre-committed decision table on the pilot's own outcomes; no guardrail breach |
| 5. Scale | more episodes from engineering, same read | absolute IL and IL% both improve |

## 11. Pilot evaluation — on the episodes engineering supplies

### 11.1 Design

There is no A/B and no control arm. The pilot prices **every episode
engineering hands the system**; the only requirement on that set is that
it **spans FCs and categories** (several of each, so that no single site,
season or category carries the read and exploration can be tested at a
small scale across the catalogue). Evaluation is a **pre/post read on the
same units**: the pilot window against the trailing history of the same
SKU × FC units, on IL% as a ratio of sums with absolute IL and the
denominator alongside, in every cut (overall, by category, by FC). This
read cannot separate the policy from season or common day shocks — it is
weaker evidence than a randomised comparison, by design and on purpose
(owner decision, 2026-09-05); the risk register carries it as item 3. What
it *can* say is whether the system did what it was built to do (§11.2), and
the guardrails still stop it if it does not: sell-through, waste units,
realised margin, on the trailing-mean basis of the same system-priced
episodes. Exploration's own evidence is unaffected — the forced moves are
randomised within the pilot, so the elasticity update is clean regardless
of how the pilot's IL is read.

### 11.2 The pre-committed decision table

| Absolute IL | IL% | Action |
| --- | --- | --- |
| improves | improves | ship |
| improves | flat or worse | **escalate to the product owner** — the system did what it was built to do; acceptability is a business call |
| flat or worse | improves | do **not** ship on IL% alone — the denominator grew |
| flat or worse | flat or worse | do not ship |

The second row is the case this design makes most likely.

## 12. Open owner decisions — recommendations and tooling

`bootstrap.derive_thresholds` produces the evidence for the SET BY OWNER
thresholds.

**Guardrail stop thresholds.** The floor a threshold must clear is 3σ of
the series' own noise **on the basis the monitor compares against**: the
trailing mean of the same system-priced episodes, each day smoothed over
`deterioration_smoothing_days` before the comparison, the monitor's own
order (`guardrail_threshold_recommendation` reports the floor and the
verdict). The comparison itself lives once, in `common.guardrail.deviation`.

Bases are per metric in `common.guardrail.BASIS`: `scrap: relative`
(strictly positive), `margin: absolute_pp` — `margin_rate` **crosses
zero**, and a ratio to a sign-changing mean has no scale (measured: a
6,545% "floor"). **A relative floor at or above 1.0 is reported `BLOCKED`,
not as a number** (`floor_is_unusable`): ordinary daily swing exceeds the
series' own level and no threshold on that basis is both safe and useful.
`TOO TIGHT` (below the floor: false-fires and silently suspends
exploration), `insufficient history` (nobody measured the floor — `status`
reads WARN, never PASS, and `tune` names it rather than skipping it) and
`CLEARS THE FLOOR BUT LIKELY INERT` (> 3× the floor: a
guardrail that cannot fire is absent, not conservative) are both blocking
verdicts. **The persistence rule is load-bearing**:
`persistence_days` (2) means a condition fires only after consecutive days
over threshold; a single day past a 3σ floor is expected roughly annually,
two in a row essentially never. Calibration principle: a false fire is
cheap (suspends exploration only), but a threshold that fires constantly
kills the learning loop — tighten via persistence, never below the floor.

## 12a. Multi-day episodes

FLC windows commonly run past midnight. The episode key was
`sku_id | fc | date`, so one economic window became two or three episodes
and everything episode-terminal was wrong at each seam.

**An episode is a maximal run of consecutive hourly rows for one SKU × FC
over which the source's `hours_remaining` counter ticks down exactly one
per elapsed hour.** Both signals must agree: time-contiguity alone merges
back-to-back windows; the counter alone stitches across a data hole.
Crossing midnight is a one-hour step like any other. Known limitation,
confirmed by the producer (contract §01 C5): the counter can also step UP
mid-window when a restock extends the window, which this rule reads as a
new-window boundary — so a restock-extended window in the extract splits
in two. Accepted for now (owner): the derivation retires once
engineering's `episode_id` lands in the feed. Duplicate
`(sku, fc, date, hour)` rows collide two runs into one id, so both copies
drop (`duplicate_hour_rows_dropped`).

Three things moved with the key: **split assignment** (an episode belongs
wholly to the split its window started in), the **leakage guard on the
velocity features** (read as of the episode's *first* date — row-dated
reads let a second-day row see its own first-day sales), and
`prior_episode_ref_sales_rate` at true episode grain. What it fixed, in
order of consequence: the monotonicity anchor no longer resets mid-window
(price could rise across midnight); the DP terminal value fires once, at
the real window end; scrap/IL/clearance no longer count carried-over
inventory as scrapped at each seam; `rho`/`deff` are measured over whole
windows, so evidence is no longer over-counted.

### The last hour writes off, it does not report

**`ending_inventory` is always zero on an episode's last row** — the source
writes off whatever remains at close, breaking the inventory chain by
design. Two silent failure modes: reading the field as scrap reports zero
scrap for every episode (IL collapses to discount cost — what this repo did
until the quirk surfaced); treating the broken chain as a data error and
dropping those episodes discards essentially all genuine waste. True
leftover is `max(0, starting − sold)` on the last row —
`common.episodes.leftover_units` is the single definition; every IL and
scrap figure reads it through `episodes.scrap_units` and
`metrics.episode_economics`.

The convention, stated positively: **`ending_inventory` is the final
quantity on hand at the close of the hour, after anything that arrived
during it.** `common.episodes.hour_status` is the only place the hour rules
are written down:

| | Meaning |
| --- | --- |
| `ending == starting − sold` | ordinary hour |
| `ending > starting − sold` | **RESTOCK** — including an hour that sold more than it opened with |
| `ending == 0` and `net > 0` | the source wrote the remainder off: how a listing closes |
| `0 < ending < starting − sold` | shrink |

The flow identity: **`opening + restocked == sold + scrap`** (scrap =
leftover + shrink), with `clearance == sold / supply`,
`supply = opening + restocked` — so clearance cannot exceed 1. Enforced in
`common.episodes.flow_identity_violations`; continuity makes the two sides
provably equal, so a violation is a bug here, not a feed defect.

Two quantities are reported side by side everywhere: `*_mu` (the NB mean —
units wanted) and `*_units` (`E[min(D, q)]` — units sellable off this
shelf). Never equal for finite q: what is lost is `Σ_{k>q} (k−q)·P(k)` —
demand that overflowed — lost sales, not model shrinkage.

**Censoring is decided at the LAST ROW only**: the source stops emitting
rows once inventory reaches zero (which is why `extend_to_window` exists),
so an empty shelf ends the episode; `censoring_off_last_row` reports any
row that breaks it. A restock never binds the monotone-price constraint the
wrong way — more stock argues for a deeper discount, always allowed.

Event-store knock-ons: exactly three `adjustment_reason` values are
legitimate — `intraday_restock`, `episode_close_write_off` (recognised **by
the zero itself**, not by position: a merged cross-midnight window can put
the source's boundary zero mid-episode for us; the *offline* continuity
drop applies the write-off exemption to the last row only — mid-episode a
zero ending with stock owed is shrink), and `unexplained_shortfall`
(shrink — named, not quarantined: quarantine is for what the system cannot
interpret, and a quarantined outcome never lands, so leaving shrink unnamed
failed the shadow completeness gate at the feed's whole shrink rate).
`common.episodes.adjustment_reason` is the one implementation, and
`pipeline.ingest_outcomes` runs it when it builds outcomes from the hourly
feed — the classification is derived, never asked of an integration.

**Restocked episodes are kept and gate nothing**: `ending[t] ==
starting[t+1]`, so the arrival is carried forward and the solver meets a
larger `q` next hour, as it does live. Only a restock on the final hour
gates (`final_hour_restock`) — the close is then ambiguous.

### An episode is not as long as its window

Closure is asked FIRST and is one condition — `ending_inventory == 0` on
the last row. Then the leftover, read UNCLIPPED, says which ending:

| | Test on the last row | Scrap |
| --- | --- | --- |
| `sold_out_early` | closed, leftover 0 | none |
| `completed` | closed, leftover ≠ 0 | the leftover |
| `not_closed` | `ending != 0` | **unknown** — excluded, never counted as zero |

`hours_remaining` is not consulted: the counter is *nominal* and still
positive on essentially every final row. A counter-keyed rule classified
~13.4% of episodes (the ones ending with stock) as truncated and excluded
~99% of real leftover from every scrap statistic. Confirmed with the
business: when a listing ends with stock on hand, those units are disposed
of and counted as scrap, whatever the counter says.

Unclosed episodes are **flagged, not dropped** (`edge_truncated` splits the
extract-boundary cases from the residue): their observed hours are ordinary
priced demand, only the ending is missing, and every consumer of an ending
already excludes it on its own (`scrap_units` NaN, replay's
`outcome_known`, shadow via `metrics.settled`). They are also the largest,
slowest-clearing windows in the extract. Read
`share_of_unclosed_explained_by_edge`: near 1.0 and the unknown-scrap
problem is purely the extract cut; the residue is a feed gap or a subset
that never writes off (`not_closed_by_month` / `by_category` locate it).

There is deliberately **no frame-wide closure fallback** ("no sentinel
anywhere → treat everything closed"): it fails in the invisible direction —
a feed that stops emitting the sentinel reads as perfectly healthy. Missing
sentinel reads as unclosed, loudly; `write_off_convention_in_force` names
the cause. Live, an in-flight episode's most recent row carries honest
inventory — stock on the shelf, not in the bin — and `pipeline.monitor`
calls the same classifier.

**The backtest sees nothing past the gate window**: `pre_launch` slices to
episodes opened on or before `split.test_end` before anything reads the
frame (previously `policy_replay` and `derive_tau_initial` ran on the whole
frame, so `tau_initial` was partly fitted on the hold-out). The one
artifact that must outlive the gate is the level-factor schedule:
`data.launch_date` (owner, null until launch) switches `--fit-calibration`
from the pre-launch scope to the latest data plus the week being priced,
so the weekly cron reaches the week `calibration_current` checks. Every
other fit keeps `pre_launch`; moving `split.test_end` would rescope them
all and turn `--check-only` into a re-settle on hold-out rows.
The DP arm is **priced at the launch belief and transitions at the prior
mean** — the policy that will run, graded against the prior's best guess of
the world — while the legacy arm transitions at the prior mean too, so a
steeper launch belief changes what the DP does and never what legacy sells;
the like-for-like IL gap is then what the belief costs if the prior is
right. `step_sensitivity` shifts around the launch belief with the world
held at the prior mean, and `intra_episode_deepening` reports both medians
(`median_abs_eps_prior`, `median_abs_eps_in_use`).
`derive_tau_initial` solves production's own equation: the budget is
`explore.budget_today` at the widest launch prior std, Q-spreads are
collected under `inference.decide`'s explorability gate, and `n_days` is
the calendar span (`episodes.calendar_days`) — the same three definitions
shadow's derivation uses, so the cross-check can only disagree on the
path, never on the bookkeeping.

### The horizon comes from the window, not from the row count

Rows stop at zero inventory, so an episode's row count is short *because
it sold out* — handing the DP that count as its horizon feeds it the
outcome it is deciding, brings the terminal scrap penalty forward, and
pushes it to discount too hard. `common.episodes.extend_to_window` extends
every episode to its full window before `mu_ref` is predicted, added rows
marked `is_observed = False`. The extension is exact (every feature is
episode-constant or a function of the timestamp). Synthetic rows carry no
sales; fidelity, calibration and every gate ratio see observed rows only;
the legacy arm's price extends by holding its last observed discount (the
legacy ramp-to-cap behaviour), so both arms run the same horizon.

## 13. Risk register

| # | Risk | Mitigation |
| --- | --- | --- |
| 1 | **Learning throughput** — per-outcome information is small, the prior is wide where unidentified, monotonicity concentrates identification at entry | Shadow emits `learning_yield_would_be` so weeks-to-convergence is read before the pilot. Two floors bind separately: evidence (episodes per update) and calendar (one human-gated update per `learning.update_cadence_days`). Levers: budget share, coarser cells. 21-day flat-posterior alert |
| 2 | **Frozen-model drift** over the window | Final retrain at the launch freeze; daily drift ratio; weekly level re-fit (§9.2) |
| 3 | **No control arm** — the pilot's read is pre/post on the same units, confounded by season and day shocks | Engineering's sample spans FCs × categories so no single site or season carries the read; absolute IL and IL% are read together with their denominators; the trailing-mean guardrail basis is the same series the floors were measured on; scale is gated on the decision table, not on a p-value |
| 4 | **Metric divergence at readout** — planner optimises IL, business reads IL% | Both metrics + denominators everywhere; decision table pre-committed (§11.2) |
| 5 | **Single-elasticity misspecification** | Residuals logged by discount region; piecewise response in phase 2 |
| 6 | **Enter-and-hold at the launch prior** — deepening pays only when \|ε\| > (1−d)/(γ−d), measured median 2.429 | Track the threshold gap every run; pre-brief the pilot; exploration closes the gap |
| 7 | **Episode fragmentation from missing source hours** — a single absent hour splits one window in two, and the second fragment's first hour reads as an entry hour | Conservative today (ambiguous leftover excluded). Fix is to stitch where clock and counter agree — deferred (changes the analysis population) |
| 8 | **Model under-prediction from censored training labels** | First phase-2 priority: censored-count training |
| 9 | **Censoring flag discards information on restocked hours** (sold more than it opened with reads "at least starting") | Deliberately pinned: discards information rather than biasing ε — the safe direction; held by test |

## 14. Phase 2 (deferred until the loop demonstrably works)

Censored-count model training; episode stitching across missing hours;
subcategory learning cells with partial pooling; automated posterior
updates with criteria drafted from observed gate behaviour; episode-level
random effects replacing the deff deflation; off-policy correction to
recover exploitation outcomes.

---

## Appendix A — operational quick reference

```
step                                          writes
0. bootstrap.download_flc                     data/flc_raw.parquet   (Redshift; REDSHIFT_* from ~/.env)
1. bootstrap.prepare_data --input <raw>       data/prepared.parquet, artifacts/split_manifest.json
3. bootstrap.train_baseline --input prepared  artifacts/baseline_model.txt, feature_schema.json
3b. train_baseline --fit-calibration          artifacts/calibration.json    ┐
4. bootstrap.estimate_prior --input prepared  artifacts/prior.json          │ ONE TURN of the
5. bootstrap.fit_dispersion --input prepared  artifacts/r_lookup.json, rho.json │ f<->r loop
5b. train_baseline --check-convergence        (dry run: settled?)           ┘
6. backtest --input prepared                  reports/backtest.json
6b. bootstrap.derive_thresholds               reports/thresholds.json
8. bootstrap.init_posterior                   artifacts/posterior.json      (once; --force to overwrite)
9. pipeline.shadow --input prepared           reports/shadow.json           (holdout by default)
11. bootstrap.seal                            artifacts/bundle.json
```

```bash
# THE driver: probes the state on disk, runs every step below that needs no
# human, stops at the next decision (owner keys, launch_date, update --apply).
python3 -m pipeline.advance --plan       # touches nothing
python3 -m pipeline.advance              # to the next human decision
python3 -m pipeline.advance --feed <yesterday's parquet>   # the daily lane

# what it runs, for stepping through one at a time:
# the bootstrap: 1, 3, then 3b-5b iterated to CONVERGED, then 6, 6b, 11, status.
# RETRAINS THE MODEL (rule 1) — never re-run it to settle calibration.
python3 -m bootstrap.run --input data/flc_raw.parquet
python3 -m bootstrap.run --check-only     # settle after a config paste, NO retrain

# launch
python3 -m bootstrap.init_posterior
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
#   --max-episodes 0 sweeps everything (final pre-launch record);
#   --workers N (0 = every core but one); byte-identical serial or parallel

# tuning loop
python3 -m pipeline.tune              # what to change, on what evidence
python3 -m pipeline.tune --apply      # paste MEASURED values, log decisions

# daily production loop -- advance --feed runs it in this order
python3 -m pipeline.ingest_outcomes --feed <yesterday's parquet>
python3 -m pipeline.update --calibrate-tau   # tau walks every closed day, no operator
python3 -m pipeline.monitor
python3 -m pipeline.assurance
python3 -m pipeline.export_events
python3 -m pipeline.status             # exits 1 on any FAIL
python3 -m pipeline.update --apply     # human-gated bounded update (§5.11)
python3 -m pipeline.update --resume-exploration   # human: lift a stop-condition suspension

# leadership deck: twelve scenarios answered by dp.solve on this config
python3 -m tools.scenario_deck --workers 0   # reports/scenarios.html (~2 min)
```

Run outputs (`data/`, `reports/`, `artifacts/`, `events_store*/`) are never
committed.

## Appendix B — glossary

| Term | Meaning |
| --- | --- |
| Episode | One SKU × FC selling window: a maximal run of hourly rows over which `hours_remaining` decrements by one per elapsed hour. NOT keyed by date |
| IL / IL% | Inventory Loss in currency / IL over full-price value of units sold |
| `d_ref` | Category reference discount — the anchor at which the frozen model predicts |
| `d_max` | Feasible discount ceiling `1 − cost/price` |
| `mu_ref` | Frozen baseline demand prediction at `d_ref` |
| ε | Exponent mapping price ratio to demand; the only quantity learned in production |
| `r` | Frozen NB dispersion (`Var = mu + mu²/r`) |
| `rho` | Frozen intra-episode demand correlation |
| deff | Design effect deflating correlated within-episode evidence |
| `tau` | Currency threshold defining the affordable exploration set |
| `delta_min` | Smallest forced move from the REFERENCE discount, in log price, whose signal clears the model's level bias: `k·σ_b/|ε|` per cell (§5.8) |
| Cell | A learning unit: one high-volume category, or the pooled global cell |
| Anchor | The price currently in force; hourly actions may only deepen from it |
| Censored hour | Sales hit inventory — demand known only as a lower bound |
