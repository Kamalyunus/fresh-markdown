# Perishable Markdown MVP — System Design

**Status:** Implemented; validated on production FLC data after the section 12a data-definition corrections; calibration and shadow gates PASSED (model `baseline-20260809120225`)
**Audience:** Technical leadership review — this document is self-contained and requires no companion reading

---

## 1. Executive summary

This system prices perishable FLC (fresh-limited-clearance) inventory through
its final selling window, replacing a legacy policy that ramps discounts
deterministically ~1 percentage point per hour on the clock. It minimises
**Inventory Loss (IL)** — discount given away on units sold, plus scrap cost
on units unsold at expiry — running at **32.3% of full-price sales value**
on the replay sample of 2,000 episodes.

The central technical fact shaping the design: **price elasticity cannot be
estimated from our own history.** The legacy ramp makes price collinear with
hour-of-day, and our bootstrap analysis surfaced a second, subtler confound
(within-episode survivorship — section 3.2). Any system that claims to have
learned price response from this data has actually learned the clock. The
design therefore uses history only for what it can support — baseline demand,
demand variance, correlation structure — and learns elasticity **in
production**, from deliberately randomized price perturbations whose total
cost is capped at **1% of markdown IL** per day.

Current state: the pipeline has been exercised end-to-end on production data
through four bootstrap iterations, each of which caught and fixed a real
defect (section 7). The demand model now carries per-SKU velocity features,
and the calibration process surfaced a measured fact about the business:
weekly demand levels swing ±8% around a persistent baseline deficit, so the
fidelity gate band was reset by the owner to [0.90, 1.10] (~2σ of the
measured 3-week noise) with the daily drift ratio as the continuous guard.
The calibrated model clears that gate (`level_bias_at_anchor` **1.0389**),
and the shadow phase has run to completion on live data with no prices
applied — 12,771 decisions, event completeness 0.9974, zero cost-floor
violations. Under the same demand model the planner shows **38.0% less
Inventory Loss** than the legacy ramp for **0.97pp** of clearance. Two
stop-condition thresholds and the A/B minimum detectable effect (section 12)
remain open owner decisions; they are the only things between here and the
exploit-only pilot.

The honest risk summary: the *safety* engineering is strong (a below-cost or
rising price is structurally impossible, not merely checked for; every gate
that has fired so far fired correctly and caught a real problem). The open
question is the *speed* of the learning loop within the 13-week MVP window —
quantified in the risk register (section 13), with the shadow phase designed
to answer it before any price is applied.

---

## 2. Business context and objective

### 2.1 The problem

Each perishable SKU at each fulfilment centre gets one final selling window —
one **episode**: a run of contiguous selling hours for one SKU × FC, with
small starting inventory (median ≈ 2 units). Windows are **not** a single
trading day — they routinely run past midnight, and 36-hour windows are
common, which is why an episode is keyed by the source's own window counter
rather than by calendar date (section 12a). Each hour, a discount is chosen;
whatever is unsold when the window closes is scrapped at cost.
Price too shallow and scrap dominates; too deep and margin is given away on
units that would have sold anyway. The legacy policy ignores demand entirely:
enter at a fixed reference discount, deepen ~1pp/hour to a cap.

### 2.2 What the system optimises

The planner minimises expected **absolute Inventory Loss** per episode:

```
IL = Σ over hours (original_price − applied_price) × units_sold   ← discount cost
   + cost × unsold_inventory_at_expiry                            ← scrap cost
```

Absolute IL is the currency amount the business is accountable for, and it is
additive across hours — which makes it a valid dynamic-programming reward
with no transformation.

### 2.3 What the business reads — and why the two can disagree

The headline reporting metric is **IL% = IL / (original_price ×
units_sold)** — loss over the full-price value of what actually sold. Its
denominator is **endogenous**: deeper markdowns sell more units and enlarge
it. Minimising IL and minimising IL% are therefore *different optimisations*.
A worked example (original price 10,000, cost 2,000, 10 units, fixed
horizon):

| Policy | Price | Units sold | IL | Denominator | IL% |
| --- | --- | --- | --- | --- | --- |
| A | 9,000 | 4 | 16,000 | 40,000 | **40.0%** |
| B | 7,000 | 8 | 28,000 | 80,000 | **35.0%** |

The planner chooses A (16,000 of loss beats 28,000); IL% prefers B. The
divergence is widest on low-cost SKUs, where scrap savings are small and the
denominator effect dominates. This is a deliberate decision: a ratio
objective can be "improved" by discounting harder purely to grow its own
denominator, which is not behaviour a planner should learn. The consequence
is stated plainly rather than discovered later: **the A/B will be read on a
metric the planner does not optimise**, the likeliest outcome is "absolute IL
improved, IL% roughly flat", and the decision rule for that case is
pre-committed (section 11.2) rather than left to whoever reads the dashboard.

Reporting discipline that follows from the endogenous denominator:

- Aggregation is **always a ratio of sums** (Σ IL / Σ denominator).
  Per-episode IL% is undefined for zero-sale episodes — which are ~12% of
  episodes, not an edge case — and is never computed or averaged.
- Every IL% figure is reported **with its denominator**, so a sales-mix shift
  is distinguishable from a pricing-performance shift, and **absolute IL is
  reported alongside** in every cut (total, category, FC, A/B arm).

### 2.4 Learning rate is the product

The system's value is not its day-one policy — it is that the policy improves
from its own decisions. Learning time scales with the number of
independently-estimated cells and with the volume of usable evidence, so the
MVP deliberately cuts anything that slows learning without protecting money:
subcategory-level learning (100+ cells vs ~10), Thompson sampling (redundant
with budgeted exploration, and a known defect source in a predecessor),
weekly model retraining (confounds learning with drift), deployment
machinery for things that don't change, stochastic replay rollouts (the A/B
is the evidence; replay is a sanity check).

---

## 3. The identification problem

**History predicts demand. It cannot identify price response.** Those are
different jobs, and the distinction is the reason this system exists in the
shape it does.

Almost everything the planner needs comes from history and transfers cleanly:
demand level by SKU, category and FC; hour-of-day shape; day-of-week and
seasonality; per-SKU velocity; the negative-binomial dispersion `r`; the
intra-episode correlation `rho`. All of it is fit offline, frozen, and used in
production. Section 5 is largely an account of *what history can support*.

Exactly one quantity does not transfer: the **price → demand slope**. Not
because the data is small or dirty, but because **price was never a free
variable in it**. Every price in the history was produced by the legacy rule —
enter at the reference discount, ramp ~1pp/hour to a cap — so price is a
deterministic function of the clock, and a deep-discount row exists only
because earlier hours failed to sell. Price, hour, and "this episode is
already failing" are the same column of numbers.

This is the standard distinction between **prediction** and **causal
identification**. History is excellent at predicting what happens *under the
policy that generated it*. It is silent about what happens under a *different*
policy, unless the policy variable was varied independently of everything
else. We are proposing a different policy, so that is exactly the question it
cannot answer.

Two consequences follow, and both matter operationally:

- **More history does not help.** The bias is a property of how the data was
  generated, not of the sample size. Ten years of the same ramp is ten years
  of the same confound, so "collect more data first" is not a route to an
  answer.
- **Randomisation is the only route.** Prices chosen independently of state
  break the rule's grip on the price column. One day of randomised prices
  identifies ε better than five months of history, because it is the only data
  in which price was not set by a rule. That is why exploration is a budgeted
  P&L line rather than a nice-to-have: it is the sole supply of the one number
  the planner needs and history structurally cannot provide.

### 3.1 The clock confound

Under the legacy ramp, the discount deepens as the evening demand peak
arrives. Within an episode, price and hour move in lock-step (correlation
≈ 0.8). Demand lift from the evening is statistically indistinguishable from
demand lift from the discount. History can *bound* elasticity; it cannot
point-identify it.

### 3.2 The survivorship confound (found during bootstrap, not in the original analysis)

There is a second trap: under the ramp, an observation at a deep discount
exists *only because* earlier hours failed to sell — deep-discount rows are
adversely selected toward low-demand episodes. Fitting elasticity on all
rows drags every estimate toward zero ("discounts don't seem to help"),
which we observed directly: all sixteen categories initially pinned at the
zero-side boundary of the search grid. The fix is to identify only from
**entry-hour variation across episodes** (different episodes start at
different hours and are truncated differently by the cost floor), never from
adjacent hours within an episode. After that fix the boundary pinning
stopped — the estimates moved into the interior, evidence the mechanism and
not the data was the problem. They still do not survive the acceptance
checks: on the corrected extract **no category brackets cleanly** (section
9.3), so every cell launches on the wide fallback prior.

### 3.3 How the design responds

1. **The demand model never exposes a price gradient.** The baseline predicts
   demand *only at the category reference discount* — at inference its price
   features are overwritten to that anchor — so whatever price-hour artifact
   it absorbed in training is never queried. Price response enters through
   exactly one learned scalar per category:
   `mu(d) = mu_ref × ((1−d)/(1−d_ref))^ε`.
2. **History contributes a bracketed prior, not an estimate** (section 5.6):
   two deliberately-biased estimators with *opposite, known* bias directions
   bound the plausible range; the prior is the midpoint with a width-derived,
   floored standard deviation. If acceptance checks fail, the system falls
   back to a wide, honest prior — a designed outcome, not a failure.
3. **Truth comes from production randomization** (section 6.2): a small,
   costed, uniformly-randomized set of price perturbations whose outcomes
   are the only evidence the learner consumes.

### 3.4 Why the cold-start elasticity is not fitted to history

The obvious shortcut — pick the ε where `mu_ref × ratio^ε` best fits
history — deserves a direct answer, because it *was* run, in its most
defensible form: proper censored likelihood, per category, restricted to the
cleanest identifying slice (entry hours). That is exactly what the prior
procedure computes, and its output is the argument against trusting it. On
production data, DAIRY's best-fit ε was **−0.10 under one defensible
specification and −3.05 under another** — a 30× swing on a single modelling
choice; BAKERY's "estimate" sat pinned at the search bound (an optimiser
reporting its cage, not the customer); and the implied direction *inverted*
between model versions. A best-fit ε is unstable because the likelihood
optimum is set by whatever variation dominates the data — the clock,
survivorship, weekly demand shocks — not by price response.

The trap makes it worse than useless: a fitted ε would make the offline
replay look excellent, because it fits the residual by construction — and
then the planner prices with it. If the fit absorbed "deep discounts
coincide with items that don't sell" (survivorship), the DP concludes
markdowns are futile, holds price, and burns scrap: the exact under-clearing
failure the calibration gate exists to prevent, induced by the number that
made the gate green. Optimising replay optics with ε is how a pricing
system fails while its dashboard smiles.

What history legitimately contributes is therefore bounded: honest brackets
where the acceptance checks pass, and a neutral, wide, cheap-to-correct
cold start (−1.0 ± 0.6) where they do not. The wide std is not resignation
— it deliberately holds the exploration budget at full scale, so **the
first weeks of the learning pilot are themselves the ε fit**, run on
randomized data where the optimum finally means price response, at a cost
capped at 1% of markdown IL per day.

---

## 4. Architecture overview

```
FROZEN (fit offline, unchanged during the MVP window)
  baseline_model.txt      demand at the reference discount, by context
  feature_schema.json     feature order and categorical levels
  calibration.json        per-category level factors (fitted; NOT applied — gate passed without)
  r_lookup.json           negative-binomial dispersion by subcategory
  rho.json                intra-episode demand correlation (one scalar)
  prior.json              elasticity prior: bracket or recorded fallback

LEARNING (the only thing that updates in production)
  posterior.json          elasticity by cell {mean, std, n_obs, information,
                          version} + processed-outcome ledger (atomic)

DECISION PATH (per hourly decision interval)
  state ── validate (reject, never an unsafe price)
        ──▶ feasible discount tiers, constructed from the cost floor
        ──▶ exact DP: expected IL for every tier
        ──▶ exploit the argmax | explore the affordable set
        ──▶ price ──▶ decision event (~30 fields)
                          │
                 finalized outcome event
                          │
        censored NB update, correlation-deflated, bounded step,
        human-gated ──▶ posterior
```

Three properties worth stating up front:

- **Safety is structural.** The action set is built from the cost floor and
  the price-monotonicity anchor; a below-cost or rising price is
  *unrepresentable*, on every path including forced exploration.
- **Evidence is append-only and versioned.** Every decision event carries the
  exact demand prediction, posterior moments, and artifact versions it used;
  learning replays from events, never from recomputation.
- **Exactly one thing learns.** Everything else is frozen so that posterior
  movement is attributable to learning, not drift.

---

## 5. Component deep-dive — what each part does, and why

### 5.1 Configuration — one tuning surface

Every threshold, window, rate, and bound lives in a single `config.yaml`;
code contains no numeric literals for anything tunable, and adding a tunable
to code without adding it to config is defined as a review failure. **Why:**
pricing systems die by configuration drift — a constant tuned in one module,
a stale copy in another, and a post-incident review that cannot reconstruct
what was actually running. One file gives the operator a complete review
surface, and the strict-mode loader *refuses to start* while any measured or
owner-decided value is null — the system cannot silently run on a guessed
parameter. Every value is labelled by provenance: `MEASURED` (produced by
the bootstrap pipeline), `SET` (design choice), or `SET BY OWNER` (business
decision), so responsibility for every number is explicit.

### 5.2 Data preparation — schema mapping and filter chain

Applies the source-to-canonical column mapping exactly once, converts the
discount column from percent to fraction exactly once, builds episodes as
contiguous windows, and runs a fixed filter chain, emitting a row/episode
waterfall after every step. **Almost every filter drops the whole episode
rather than the offending row** — a hole punched mid-window re-segments into
a spurious short episode, which loses more than the episode does.

**Only integrity and scope rules drop.** A stage removes an episode when its
rows cannot be believed, cannot be used by anything at all, or fall outside
the period the study covers. Nothing is dropped for being hard to *price* —
those conditions are flags, because `FEATURES` carries neither `cost` nor
`hours_remaining` nor anything about the inventory chain, so the demand model
cannot see them and such an episode is an ordinary observation to every frozen
artifact. Dropping them removed **>70% of the extract's COGS** from every fit.

| Step | Drops |
| --- | --- |
| `duplicate_hour_rows_dropped` | both copies of any repeated (SKU, FC, hour) — no principled way to choose, and they collide two runs into one episode id |
| `gap_split_windows_dropped` | **every fragment** of a source window a missing hour split in two. Neither fragment is an episode: the first ends unclosed, the second opens mid-window with the wrong starting stock, a counter part-way down, and a first row that reads as an ENTRY row — which `estimate_prior` fits elasticity on. Detected from the counter, which across a gap falls in step with the clock; a genuinely new window resets it upward |
| `exclusion_window_removed` | any episode with *any* hour in the known bad-data window. Scope, not integrity: the rows are fine, the period is not |
| `discount_out_of_range_dropped` | discount outside [0,1] — the percent→fraction conversion applied twice or not at all |
| `negative_quantities_dropped` | impossible quantities: negative inventory or sales. **Not `cost <= 0`** — that is the `cost_missing` flag |
| `null_category_dropped` | no category/subcategory — no reference discount, no dispersion cell |
| `zero_base_price_dropped` | `original_price` still absent after fill-within-episode |
| `episode_universe` | the three conditions that make an episode's inventory readable, evaluated once and before any filter with an opinion about price, category or cost: CONTINUITY (`ending[t] == starting[t+1]`, the only one that drops), the IDENTITY (`opening + restocked == sold + scrap`, a guard on the arithmetic since continuity makes it provable), and a CLEAN CLOSE (`starting >= sold` on the last row, which flags `final_hour_restock`). An hour-level restock or shrink is neither: both are real events, counted gross and settled at the episode level |
| `contiguous_episodes_built` | *(not a filter)* re-segmentation, and a NO-OP that RAISES if it stops being one — every drop after the ids are assigned is episode-scoped, so nothing punches a hole in a window |
| `negative_window_recovered` | *(not a filter)* "manufacturing" SKUs enter with a counter that is **already negative** — a large negative constant, not a countdown from the window length. Dropping them is not neutral: they are concentrated in a few categories, so it selects on category and biases every per-category figure. Measured on production, 381,805 of 384,055 such episodes (**99.4%**) resolve inside `data.manufacturing_window_hours` (**24**), so those get a synthetic countdown `(cap−1) − position`. A countdown, never a clamp: the counter drives episode identification, the DP horizon, and `extend_to_window`, and a flat counter re-segments every hour. The 2,250 that run longer are **not** recovered and carry the `negative_window` flag instead, with their count reported. **Runs after re-segmentation**, because it is the only step that mutates `hours_remaining` — the field the ids are derived from — and its synthetic countdown can otherwise line up with a genuine neighbouring window and merge the two |
| `dp_eligible` | *(not a filter)* the terminal summary row: how much of the surviving population the DP can act on, with the per-flag breakdown in its detail block |

`tag_dp_eligibility` then flags, on the surviving frame. Five conditions gate
`dp_eligible`, each naming something the *solver* cannot do; an episode is
labelled with the first it trips.

| Flag | Why the DP cannot act on it |
| --- | --- |
| `cost_missing` | `cost <= 0` — a *missing* cost, not a free good. `d_max` reads 1.0, so the DP would discount to the tier cap believing scrap is free, and IL reads zero |
| `non_priceable` | `cost >= original_price`, so `d_max <= 0` and `feasible_tiers` is EMPTY |
| `negative_window` | `hours_remaining` still `< 0` after recovery — the DP takes its horizon from the counter and `extend_to_window` builds the synthetic tail from it |
| `window_too_long` | above `data.max_window_hours` (**120**) — `extend_to_window` RAISES above the cap, so this is a crash rather than a refusal |
| `outcome_unknown` | the episode never closed inside this data. Gates `eligible` as well: an unfinished episode is not a complete observation of anything, and two consumers silently mis-weighted one before this existed |
| `final_hour_restock` | the last row sold more than it opened with, so stock arrived during the close and the leftover is a guess. Gates `eligible` too |

Four conditions are flagged and gate nothing:

| Flag | Why it does not gate |
| --- | --- |
| `below_cost_hours` | a price the LEGACY policy set, which the agent is constrained never to set |
| `edge_truncated` | of the unfinished episodes, the ones the extract boundary explains — the diagnostic that says whether the count is the boundary or a feed problem |
| `restocked` | units arrived mid-window. Does NOT gate: the replay re-solves hourly and applies the episode's own per-hour adjustment, so the DP meets an arrival exactly as it does live |
| `shrink` | units left unsold and unwritten-off. Does NOT gate: they are counted into scrap, so `supply == sold + scrap` still closes |

`eligible` — the middle population — is **three conditions and no more**, all
evaluated in `common.episodes.episode_flow` and exposed as one column:
`accounting_closes` (the identity `opening + restocked == sold + scrap`
balances), `final_hour_clean` (`starting − sold >= 0` on the last row), and
`closed` (`ending_inventory == 0` on the last row). All three were live
before; closure used to be re-derived independently at each consumer, which is
one chance per consumer to forget it, and two did.

Which population a consumer reads is one decision, in
`baseline_model.train_population` (default `integrity`), resolved through
`prepare_data.population`. The three artifact fits read the config; the DP,
the calibration gate, the backtest and shadow always pass `"dp_eligible"`.

Every stage also reports **`cogs_at_risk`** — unit cost × opening stock,
counted once per episode — with `cogs_dropped` and `cogs_dropped_pct_of_raw`
beside it. Rows are not the unit the business cares about: IL is discount
given away plus scrap at cost, so what a filter costs is exposure, and the two
measures diverge. A stage can take 1% of rows and 15% of the money, and only
the second figure says whether the surviving population still represents the
business. `cogs_dropped` is negative at exactly one stage — see below.

`contiguous_episodes_built` runs between the row-level and episode-level
passes and is **not** a filter: episode count can *rise* there, because
earlier drops split windows that were contiguous in the raw extract. The
waterfall in `artifacts/split_manifest.json` records rows and episodes after
every step, in this order. **Why the paranoia:** the three source-schema traps all
fail *silently*. A missed percent conversion produces discounts of 25.0
instead of 0.25 with no error anywhere downstream. The realised-price column
is 0 on zero-sale rows — reconstructing offered price from it silently
drops every zero-sale hour, which is ~78% of rows and precisely the
population carrying the demand signal at shallow discounts. And the source
has no episode ID, so the construction rule is persisted alongside the data
splits — production and evaluation must derive identical episode boundaries
or every episode-level metric diverges unauditably. The waterfall exists
because a filter chain that cannot show what it dropped, in order, cannot
be reviewed.

### 5.3 Historical measurement — measure first, then build

Before any component was built, a measurement pass produced every value the
design needs but must not guess: cost-ratio distributions (is exploration
even feasible under the cost floor?), same-hour price variation (is a prior
estimable at all?), demand density and censoring shares, intra-episode
correlation, per-category weekly volumes, and the A/B variance. **Why:**
each of these, guessed wrong, produces a system that is confidently wrong in
a specific way — e.g. assuming independent hourly evidence declares learning
converged four times too early (section 6.3). Three measurement outcomes
were designated in advance as *design-changing* rather than parameterising
(non-explorable catalogue → scope shrinks; no identifying variation → prior
falls back; A/B variance too high → duration or effect size must change), and
those decisions were made by humans against the measured numbers, not
absorbed silently.

### 5.4 Baseline demand model — frozen, and blind to price

A gradient-boosted tree model (LightGBM) with a Tweedie objective predicts
units sold per hour. Features: category, subcategory, FC, hour of day, day
of week, day of month, base price, two point-in-time SKU demand rates
(below) — and a **single** price feature that is **overwritten to the
category reference discount at every inference call**. **Why Tweedie:**
hourly perishable demand is a zero-heavy count (~78% zeros) with a positive
tail; Tweedie's point mass at zero fits this shape where squared-error
under-predicts and Poisson under-disperses. **Why the price-feature
overwrite is the load-bearing trick:** the model trains on confounded
history and would happily learn the price-hour artifact — but its price
gradient is *never queried*. It answers only "what is demand at the
reference discount in this context"; price response enters exclusively
through the learned elasticity scalar. One price feature means one
overwrite point, which is auditable; the legacy model's four
(depth, entry, incremental, discount×hour) were four places for the
confound to leak. **Why frozen for the whole MVP window:** if the model
retrains while the posterior learns, posterior movement cannot be
attributed — it could be learning or drift. Freezing buys attribution and
costs drift risk, which is accepted, monitored daily (section 5.11), and
bounded by the window end date. Freezing is a phase, not a posture: once the
experiment has read out there is nothing left to attribute, and the baseline
returns to an ordinary retraining cadence with the level factors tracking
drift between retrains. A per-category multiplicative correction
factor is *fitted* unconditionally on the calibration weeks but *applied*
only behind an explicit config decision, because the diagnostic in section
9.2 shows the correction is right for level errors and actively harmful for
slope errors — fitting and applying are different decisions with different
owners.

**The SKU demand-rate features.** The model's largest historical error was
level bias, and its most plausible cause is that category/subcategory/FC
alone cannot express per-SKU velocity (a few SKUs move far faster than
their category mean). Two features supply it, both **price-standardised**
— built only from *anchor hours*, stocked hours priced within one tier of
the reference discount, so they measure "how fast does this SKU sell at
reference conditions" regardless of which policy produced the price — and
both **point-in-time**, lagged strictly before the episode's date:

- `sku_ref_sales_rate_30d` — trailing [t−30, t−1] anchor-hour sales rate at
  **SKU × FC** grain (the episode grain; store traffic differs by FC), with
  a SKU-pooled fallback for sparse combinations (aggregated to SKU-day
  first, so no same-day cross-FC sales enter the window) and native-missing
  (NaN) when even that is empty — "no data" is never conflated with
  "doesn't sell".
- `prior_episode_ref_sales_rate` — the anchor-hour rate of the same
  SKU × FC's most recent previous episode: the recency signal on top of the
  30-day level signal. NaN if that episode had no anchor hours — purity
  over coverage.

Censored hours are included capped, matching the censoring the training
target itself carries. The residual caveat is a slow cross-day feedback
loop (in production these rates are computed from sales under our own
pricing); the anchor restriction is what keeps it tame.

**The training label is censored, and that is a priced trade.**

The target is `units_sold`, which stops at the shelf: an hour that sold out
records what fitted, not what customers wanted. Three options, one chosen:

- *Drop the censored hours.* Rejected — it selects on the outcome. Sell-outs
  are **14% of rows** and they concentrate in the later hours of a window,
  which is exactly when the selling happens. Training on what is left means
  training on a population reweighted toward the quiet part of the window.
- *Fit a censored likelihood* (a sold-out hour entering as `D >= q`). This is
  the principled answer and is what every other estimator here already does —
  `fit_dispersion`, `estimate_prior`, and the posterior update all carry the
  survival term. It is **not off-the-shelf under a Tweedie objective**:
  custom likelihood work plus its own validation, which the MVP did not buy.
- *Keep the label and correct the level.* Chosen. The bias is real and
  measurable — on the pre-calibration model the hourly bias runs −0.09 across
  uncensored hours against −0.21 once censored hours are counted, so the
  under-prediction roughly doubles when the unobservable hours are the busy
  ones.

What makes that trade safe is the feature exclusion below: with no inventory
or stockout indicator, the model cannot learn *"the shelf was empty"* and so
the censoring bias arrives smooth rather than structured — which is precisely
the shape a single multiplicative level factor can remove. **Section 5.4's
level calibration is therefore not housekeeping; it is the second half of
this decision**, and it is why that factor must be solved on the censored
basis (§9.3) rather than divided out: a correction for censoring cannot be
found on a basis that ignores censoring.

Revisiting it is a phase-2 item: a proper censored objective removes the need
for the level factor to carry this particular load.

**What is deliberately NOT a feature — and why.**

- *Hours remaining* is planner state, not demand context: the customer sees
  the shelf and the price, not our countdown. The DP keeps the horizon; the
  demand model does not (it is nearly collinear with hour-of-day anyway).
- *Within-episode lag sales* (`last_1hr`, `last_3hr`) were the legacy
  model's momentum features, and their exclusion is a priced trade, not an
  oversight. They are **post-treatment mediators** of the episode's own
  price path: a deeper price at hour t raises sales at t, which raises the
  lag feature at t+1, which raises the prediction — so part of the price
  effect is routed *through* the feature and the learned elasticity is no
  longer the price response. No inference-time overwrite fixes this,
  because training already attributed price response to the lag. (At median
  inventory ~2 they are also mostly a censoring indicator.) The
  cross-episode rates above are different in kind — yesterday's sales
  cannot be caused by today's price, and their anchor-restricted
  construction removes the pricing-level contamination; no such
  construction exists for within-episode lags, because under a ramping
  policy mid-episode hours are never at reference. Genuine momentum
  ("customers are discovering this item now") has a principled vehicle in
  phase 2: observe within-episode sales, divide out the price effect using
  the current elasticity, and update a **price-free multiplicative
  episode-level demand factor** applied symmetrically at every candidate
  price — momentum without corrupting the counterfactual.
- *Inventory, cost, stockout indicators* belong to the planner state and
  the censoring logic; as features they teach "low stock predicts low
  sales", which is the censoring artifact, not demand.

### 5.5 Dispersion and correlation — frozen variance structure

Demand is modelled negative-binomial: `Var[D] = mu + mu²/r`. The dispersion
`r` is fitted per subcategory by censored maximum likelihood on the
calibration weeks, with a fallback chain (subcategory → category → global)
for thin groups and a clamp on implausibly-high converged values. `rho`, the
correlation of within-episode demand residuals, is one global scalar fitted
against the model's own residuals. **Why negative-binomial:** observed
hourly demand is far more variable than Poisson (bursty shoppers, basket
effects); a Poisson likelihood would make every learning update
overconfident. **Why `r` per subcategory but `rho` global:** dispersion
genuinely differs by product type and the data supports estimating it there;
correlation is estimated from much weaker signal, and a noisy per-category
`rho` would inject noise directly into the evidence-deflation factor that
gates learning. **Why legacy data is legitimate here** when it is banned for
elasticity: these are second-moment structures — variance and correlation
*around* the mean — and the policy confound moves the mean, not them.

### 5.6 Elasticity prior — a bracket, not an estimate

Two estimators per category, both by censored negative-binomial likelihood
over the full sign-constrained grid, both on **entry-hour rows only**:

```
epsilon_naive        no hour control    → absorbs the evening lift into price
                                        → biased TOO ELASTIC (too negative)
epsilon_controlled   hour effects profiled out
                                        → removes most price variation along
                                          with the confound
                                        → biased TOWARD ZERO

prior_mean = midpoint of the two        prior_std = max(half the width, 0.40)
```

**Why a bracket instead of one best estimate:** on this data any single
estimator has bias of unknown magnitude but *known direction*; two
estimators with opposite known directions bound the truth without pretending
to point-identify it. **Why entry rows only:** the survivorship confound of
section 3.2. **Why boundary solutions are rejected outright:** an optimiser
pinned at a search bound is reporting the bound, not the data — an earlier
run with a carelessly tight bound manufactured five fake "estimates" this
way, and the upper bound (−0.05) is a *sign constraint*, never to be
widened: positive elasticity is structurally unrepresentable, which replaces
all wrong-sign filtering downstream. **Why rejection is a designed outcome:**
with bounded update steps (section 6.3), a confidently-wrong prior takes at
least seven update cycles to walk back, across every cell at once; a weak
honest prior costs only patience. On production data, 14 of 16 categories
rejected honestly and use the fallback (−1.0 ± 0.6); that is the system
working, and the std floor is precisely the insurance that makes fallback
safe.

### 5.7 The decision core — exact DP over a safe action set

Per episode, the feasible action set is every 2.5pp discount tier from zero
to `1 − cost/price` — the cost floor *is the boundary of the set*, so a
below-cost price cannot be expressed. Hourly actions are further restricted
to tiers at or deeper than the current anchor (price never rises within an
episode — a customer-trust constraint encoded in the transition, not
checked after the fact).

**Entry is a separate decision with its own, coarser action set.** It has no
anchor to respect, and because monotonicity makes it irreversible in one
direction, the entry choice sets the ceiling on every later price in the
episode. Its arms are `pricing.entry_offsets` — discount offsets relative to
the category reference, currently `[−15, −10, −5, 0, +5] pp`: entry may open
up to 15pp shallower than reference, at it, or one 5pp step deeper. Two
reasons the grid differs from the hourly one:

- *Coarse arms concentrate the evidence.* Entry carries most of the
  identifying variation (section 3.2 — the confound-free signal is entry-hour
  variation across episodes). Information about ε scales as (log price
  ratio)², so an arm 2.5pp from the reference carries only ~24% of the
  information of one 5pp away while taking an equal share of the uniform
  exploration draw. Fine arms near the optimum dilute evidence rather than
  adding to it, which is why the entry spacing is a uniform 5pp while the
  hourly grid stays at 2.5pp.
- *The deep side is bounded, not symmetric.* The predecessor design allowed
  entry anywhere in a symmetric ±10pp band; under monotonicity that let an
  episode open 10pp deeper than reference and never recover, spending margin
  in hour one and forfeiting the room to deepen later. A single +5pp arm
  keeps the option where it is worth having — it is the escape valve for the
  clearance/scrap trade documented above, taken only when Q says so — without
  reopening the range that made deep entry a default. The cost floor removes
  it entirely for the default categories above a ~0.65 cost ratio.

Offsets are snapped to the tier grid and filtered by the cost floor; if the
floor forbids every requested arm, the deepest feasible tier becomes the only
action — a single-action decision, correctly non-explorable, rather than a
fallback to the full grid. The value function is built over the full grid
either way, so the DP prices each entry arm knowing the episode will deepen
on 2.5pp steps afterwards.

**The hourly action set is every tier deeper than the anchor** — not a single
2.5pp step. The DP may hold, step 2.5pp, or jump straight to the cost floor
in one hour, and it re-solves each hour from the true state, so stranded
inventory with few hours left is exactly the situation that should pull it
deeper.

Whether it *does* is economics, not action-set width, and the condition has a
closed form. Ignoring censoring, one hour of IL is
`P₀·d·mu(d) + c·(q − mu(d))`, so deepening reduces IL only when

```
|ε|  >  (1 − d) / (γ − d)        γ = cost / price
```

The first term of the derivative is the cost of discounting units that would
have sold anyway; it dominates until demand responds hard enough to outrun
it. On the corrected extract the **measured median bar is |ε| = 2.429**, and
censoring at a median starting inventory of 2 pushes the true switch point
higher still. **Against the launch prior of −1.0, the DP is therefore
structurally an enter-and-hold policy**: on the synthetic harness the median
threshold is 2.429 against |ε| = 1.0 in use, and the DP deepens
intra-episode in 0% of episodes — 0% of them clear the bar. This is not a defect — if demand really is that
inelastic, holding price *is* the IL-minimising action, and the replay's
−12.0% IL comes precisely from refusing to ramp. But it means three things
should be said out loud before the pilot:

- The day-one policy's entire IL advantage comes from **the entry choice and
  from not ramping**, not from dynamic intra-episode markdown.
- The clearance loss (−3.25pp) and any scrap-guardrail pressure follow
  directly from this, and are the expected behaviour rather than a surprise.
- **Widening the action set cannot change it.** Only a posterior that moves
  past the threshold will, which is exactly what exploration is funded to
  find out. `intra_episode_deepening` in the backtest report tracks the gap
  between the threshold and the elasticity in use, so the distance left to
  travel is visible every run.

![deepening threshold](../reports/charts/06_deepening_threshold.png)

One reassurance about the evidence side, since enter-and-hold sounds like it
should starve the learner. It does the opposite. `pipeline.update` accumulates
`mu × (log price ratio)²` with the ratio taken against the **reference**
discount, not against the DP's own optimum — so what generates information is
how far the applied price sits from the anchor, not how large the random
perturbation was. Holding at `d_ref − 15pp` parks every hour of the episode
at `(log ratio)² ≈ 0.038`, whereas a 2.5pp wiggle around a price sitting *on*
the reference would yield ≈ 0.001. The enter-and-hold regime is a
high-information regime, not a desert; on the launch prior, moving the mean
from 1.0 to 1.9 needs roughly 8,600 exploration outcomes. `shadow` reports
the realised figure as `learning_yield_would_be` before any price is applied.

```mermaid
flowchart LR
  subgraph FROZEN["FROZEN at launch"]
    M["baseline mu_ref<br/>(price-blind)"]
    R["dispersion r<br/>+ correlation rho"]
    P["elasticity prior<br/>(bracket or fallback)"]
  end
  subgraph HOURLY["HOURLY decision path"]
    V["validate state<br/>reject, never guess"] --> T["feasible tiers<br/>from cost floor"]
    T --> D["exact DP<br/>Q(p) per tier"]
    D --> X["exploit argmax<br/>or explore affordable"]
    X --> E["decision event"]
  end
  subgraph DAILY["DAILY learning loop"]
    O["outcome events"] --> L["censored NB likelihood<br/>exploration outcomes only"]
    L --> DF["divide by deff"]
    DF --> B["bounded step<br/>+ human gate"]
    B --> PO["posterior epsilon"]
  end
  M --> D
  R --> D
  P --> PO
  PO --> D
  E --> O
```

The planner solves the finite-horizon problem exactly:

```
Q(anchor, q, h, p) = Σ_k P(D=k | r, mu(p)) × [ min(k,q)·(−(P₀−p)) + V(p, q−min(k,q), h−1) ]
V(anchor, q, h)    = max over feasible p ≤ anchor of Q(anchor, q, h, p)
V(·, q, 0)         = −cost × q                       ← terminal scrap value
```

**Why exact DP rather than a heuristic:** the state space is tiny (≤ ~20
tiers × ≤ 30 units × ≤ 12 hours), so exhaustive evaluation costs
milliseconds — and exploration *requires* the full vector of Q-values per
tier, which an approximation would not provide. The demand distribution is
truncated at 25 units with tail mass folded into the last bucket and the
tail emitted as a diagnostic: bounded compute with a visible error term.

### 5.8 Exploration — a P&L line item, uniformly randomized

The DP already prices every tier, so exploration is a *selection*, not a
separate mechanism:

```
p*          = argmax Q(p)
cost(p)     = Q(p*) − Q(p)          ← expected IL sacrificed, in currency
affordable  = { p ≠ p* : cost(p) ≤ tau }
```

If the affordable set is non-empty, the applied price is drawn **uniformly
at random** from it and flagged as exploration. **Why uniform:** uniformity
over the affordable set is the randomisation that makes outcomes causal
evidence; any smarter state-dependent choice reintroduces the endogeneity
that poisoned the historical data. **Why a currency budget instead of an
exploration probability:** epsilon-greedy-style schedules spend an
un-costed, invisible amount; here the spend is an explicit budget line — 1%
of markdown IL per day, scaled down (never below 25%) as the posterior
narrows — and `tau` self-calibrates daily so realised spend tracks it. The
theory that makes budget-only rationing sound: information about ε and the
IL cost of a perturbation both scale as `mu × (log price ratio)²`, so
**information per won is approximately constant** — there is no clever
targeting to do, only a budget to respect. High-volume SKUs automatically
receive small perturbations because their loss curve is steeper. Measured
starting point on production data: `tau` = **₩447.78**, derived from the
gate-passing calibrated backtest as the Q-spread quantile whose implied daily
spend matches the budget.

### 5.9 Posterior store — small, atomic, exactly-once

One record per learning cell — a Normal summary (mean, std), observation and
information counters, and a version — plus the ledger of consumed outcome
IDs, in a single atomically-written file. **Why a Normal summary rather than
a stored distribution grid:** the grid exists only inside the update
computation; persisting moments keeps storage trivial and makes the bounded
step well-defined, and the pricing path reads two floats. **Why ~10 category
cells plus one pooled global cell, and not subcategory cells:** three
reasons, in order of force.

1. **A finer ε would change no price.** The policy is insensitive to the
   posterior mean anywhere below the deepening bar (~2.43) — measured, not
   asserted: a full re-run at a −1.5 prior produced identical prices to −1.0.
   Behaviour only changes when a cell's posterior crosses the bar, and
   splitting cells makes every cell slower to get there. At launch, finer
   grain is not neutral; it is counterproductive.
2. **Evidence divides proportionally.** Learning time scales with cell count.
   Split a category into three subcategories and each receives roughly a
   third of the forced outcomes, so each takes about three times the calendar
   to travel the same distance — against a throughput that is already risk 1.
3. **There is no finer-grained prior to preserve.** The bracket was rejected
   for all 16 categories, so every cell starts at the same −1.0 ± 0.6.
   Subcategory cells would begin identical and learn slower: dilution bought
   for nothing. And history cannot identify ε per category, so it certainly
   cannot per subcategory — the same confound with less data behind it.

Categories above 250 episodes/week earn their own cell, everything below
reads and feeds the global cell, assignment fixed at launch. The honest
counter-argument is that if elasticity genuinely varies *within* a category,
pooling biases every subcategory in it. That is real — but it is undetectable
until the cells have moved at all, which is why the fix belongs in phase 2
and why it should be **partial pooling rather than a threshold ladder**: a
hard "use the subcategory if it clears N episodes/week, otherwise the
category" rule discards the category signal the moment it is crossed and puts
a cliff at the boundary, where 251 episodes/week buys a noisy standalone
estimate and 249 buys the category's. Shrinking each subcategory toward its
category mean in proportion to its own precision uses all the data, has no
discontinuity, and collapses to the right answer at both extremes. **Why the
ledger lives inside the posterior file:** exactly-once learning requires
"revision applied" and "outcomes consumed" to commit together; two files
cannot be renamed atomically, one can. A crash between learning and marking
cannot double-count evidence, and re-running an apply that has nothing new
to consume is a verified no-op.

### 5.10 Inference and the event contract

Validation checks nine invariants (prices, costs, integer inventory and
horizon, finite predictions, non-empty feasible set, a feasible tier under
the anchor) and **rejects the state rather than returning any price** — a
pricing system's worst failure is not "no answer", it is a confidently
wrong answer applied to real inventory. Every decision emits an event of
~30 fields: the full pricing context, the exact demand prediction used, the
posterior moments, the exploration flags and cost, and the versions of
model, posterior, and config. Every decision interval gets exactly one
finalized outcome event. **Why so heavy:** learning replays evidence from
events, never from recomputation — a feature-pipeline change must not be
able to silently rewrite historical evidence — and any decision must be
reproducible from its event alone. The store is append-only JSONL with
durable writes and duplicate detection, and malformed events are
**quarantined with their validation failures attached**, never silently
dropped: an event logger that discards what it cannot parse hides exactly
the anomalies monitoring exists to surface.

### 5.11 Learning update — censored, deflated, bounded, gated

A daily batch consumes **exploration outcomes only**, evaluates the censored
likelihood on the elasticity grid, adds the current posterior as prior,
normalises, and takes moments. Mechanics and rationale:

- **Censoring.** A stocked-out hour that "sold 2 of 2" is evidence demand
  was *at least* 2, not exactly 2 — it enters through the survival
  probability `P(D ≥ inventory)`. Treating stockouts as exact counts
  systematically understates elasticity exactly where it matters (deep
  discounts, where stockouts concentrate). Zero-sale hours are retained
  through the exact `P(D = 0)` term — they carry the signal at shallow
  discounts.
- **Exploitation outcomes are discarded.** Exploitation prices are chosen
  *by* the posterior; learning from them is the model feeding its own
  beliefs back to itself. This wastes most outcomes — accepted for the MVP,
  with off-policy correction queued for phase 2.
- **Correlation deflation.** Hours within an episode share an inventory
  pool, a demand shock, and a monotone price path; summed as independent
  they overstate evidence and declare convergence early. Accumulated
  information is divided by `deff = 1 + (forced_hours − 1) × rho` — measured
  at `1 + 7.563 × 0.3103 ≈ 3.35`. Without this one line, the system would
  report convergence three-and-a-third times too early. Both inputs are
  fitted against the model's own residuals and frozen in
  `artifacts/rho.json`; strict start-up refuses to run when `config.yaml`
  disagrees with that artifact, because a paste left over from a previous
  retrain mis-weights every posterior step in the window.
- **Evidence is banked until it is spent.** A day's exploration rarely
  reaches the information increment on its own. The batch is therefore every
  eligible outcome *not yet consumed by a revision*, so it accumulates across
  days and the likelihood is always evaluated over the whole of it. Outcomes
  are marked processed only when a revision actually commits — never when the
  update declines to fire. (The earlier implementation marked them either
  way, which banked a scalar and discarded the observations: the posterior
  would later step on the strength of an information count it no longer had
  the data to justify, and on a pilot small enough that no single day cleared
  the threshold it would have destroyed every outcome while the mean never
  moved.) `pipeline.update` reports `batch_oldest_outcome_age_days` so a batch
  that keeps growing without firing is visible long before the 21-day
  flat-posterior alert.
- **Bounded steps, human-gated.** An update applies when accumulated
  effective information crosses a threshold; each step moves the mean at
  most 0.15 and shrinks the std at most 25% (floored), with any clipped
  bound flagged for review. A human approves each day's update, and the
  apply command *refuses* while event-quality gates fail (duplicate or
  unmatched events above 1%, applied-vs-recommended price mismatch above
  1%). **Why:** bounded steps make daily updating safe — no single batch can
  destroy the posterior; the human gate caps learning at one reviewed step
  per day until an evidence record justifies automating it, with the
  automation criteria deliberately drafted later from observed behaviour
  rather than guessed now.

### 5.12 Monitoring — three families, three questions

Business (is IL improving — ratio-of-sums IL% with denominators, absolute IL
alongside, by category/FC/arm, plus sell-through and waste); learning (is
the posterior moving and how fast — per-cell mean/std trajectories, forced
decision counts, realised exploration cost vs budget, affordable-set-empty
rate, current `tau`); safety (is the event pipeline healthy — counts,
match/duplicate/mismatch rates, quarantine size, finalization lag, solver
latency). **Why three:** a learning system whose dashboard shows only
business outcomes discovers a dead learning loop weeks late via a flat IL
curve; here a posterior std flat for 21 days alerts directly. The
`affordable_set_empty_rate` is the leading indicator of a structurally
non-explorable catalogue, and `realised_vs_predicted_sold_ratio` is the
daily continuation of the calibration gate — frozen-model drift shows there
before it shows in IL%. Stop conditions (cost-floor violation, event-quality
breaches, mismatch, execution failures, missing stockout fields, solver
timeout, exploration overspend >2× budget, scrap/margin deterioration beyond
owner thresholds) **suspend exploration only for the affected cohort —
exploitation pricing continues**, so guardrails can be tight without taking
pricing offline.

### 5.13 Shadow harness — full rehearsal at zero pricing risk

Runs the complete production decision path against live data while the
legacy policy keeps pricing: state from reality (actual inventory; the
monotonicity anchor entering hour *t* is the legacy price from *t−1*), full
decision events logged, outcomes built from what legacy actually sold and
stamped ineligible for learning — the recommended price was never in force,
so those outcomes are not evidence about it. Exit gate: event completeness
> 99%, matched decision rate > 99%, **zero** cost-floor violations. Beyond
the gate, shadow produces the three numbers later phases need: would-be
exploration spend (validates the budget calibration), recommended-vs-legacy
discount deltas (first look at how different the policy really is), and the
drift ratio above — which also answers the learning-throughput question of
section 13 *before* any price is applied.

**The hold-out run.** `data.holdout` names a window *after* `test_end` that
no artifact was fit on and no gate was decided on; `--holdout` runs it.
Standing at `test_end` and walking that window forward is the only
unrehearsed test the extract can give, because every other window grades
something that was fitted to it — the calibration gate, the drift ratio and
the `tau` derivation all report in-sample numbers or 1.00× on their own
population. It is a **one-shot** resource: tune a value on it and re-run, and
it is a second calibration set. Date cuts are episode-scoped
(`common.episodes.window_slice`, assigning by the date a window *opened*);
row-scoped slicing would keep the tail of an episode that opened the evening
before as its own short episode — no entry decision, wrong opening
inventory, a countdown starting mid-window.

**Re-deriving `tau` where it will actually run.** The replay's bisection
reports 1.00× *by construction* — it solves until it does — so it is
evidence that a `tau` exists at this budget, never that the launch value is
right. Shadow re-runs the same bisection on its own decisions
(`tau_recommended`) and walks the controller day by day
(`tau_controller_trace`). The trace exists because a single spend/budget
multiple cannot answer the question that matters: `tau_next` reads only the
day just closed, so day one is spent at whatever `tau` was launched with,
the stop condition is evaluated on that same day's spend, and a `tau` that
is 8× too generous suspends exploration before the controller has anything
to correct from. Both are **reported, never applied** — `tau_initial` is
MEASURED and goes through the paste gate, with
`pricing.explore.tau_provenance_error` refusing a paste that has no source,
predates the entry-only scoping fix, or no longer matches its derivation.

### 5.14 Replay and threshold derivation — evaluation discipline

Offline replay has exactly three jobs: the calibration gate, deriving the
initial exploration threshold, and sanity-checking the planner. Its policy
comparison is **like-for-like by construction**: both the legacy price path
and the DP price path are simulated under the *same* frozen demand model
and prior, so model bias hits both arms identically and cancels in the
comparison. Comparing observed reality (legacy) against model-simulated
outcomes (DP) — the naive framing — charges every ounce of model bias to
one side and can make a superior policy look catastrophic; observed-vs-model
differences belong to *fidelity*, never to the policy verdict. Even
like-for-like, **replay output is never evidence the policy works** — the
model whose world both arms share is the same model whose price response is
an unvalidated prior, so replay can only show internal consistency; the
controlled experiment is the only evidence of policy quality (an early run
made this concrete: a 9.4% simulated improvement on a model selling 24%
light).

**The replay's headline result comes with a caveat that must travel with
it.** The DP arm shows **38.0% less IL** than the legacy arm, and it gets
there by opening far shallower — mean discount 0.1285 against legacy's 0.2935
— and holding. The cost is **0.97pp of clearance** (77.58% → 76.61% under the
model, +9.5% scrap cost), so the trade is real but small: the DP gives up
about a twentieth of the unsold-unit base to save nearly two fifths of the
loss.

That is a materially better trade than the pre-correction run showed
(−12.0% IL at −3.25pp clearance), and the reason is instructive rather than
lucky. Before the section 12a fixes the planner was handed a horizon
shortened by each episode's own realised sellout, which pulled the terminal
scrap penalty forward and made it discount harder than warranted; and scrap
was read as zero, which mispriced the very trade-off it was optimising. Both
now corrected.

All three quantities the replay produces, on the calibrated run over 2,000
episodes, with IL split into its two components because the split *is* the
result:

| | Observed | Legacy under model | DP under model |
| --- | --- | --- | --- |
| Inventory Loss | ₩17.11M | ₩19.51M | **₩12.09M** |
| — discount given away | ₩13.96M | ₩13.27M | ₩5.26M |
| — scrap | ₩3.15M | ₩6.24M | ₩6.84M |
| IL% | 38.68% | 45.14% | 28.77% |
| Clearance | 93.28% | 77.58% | 76.61% |
| Mean discount | 0.3094 | 0.2935 | 0.1285 |

Like-for-like: **−₩7.42M, −38.02% of the legacy arm**, clearance −0.97pp.
Scrap rises ₩0.59M, so the entire gain comes out of a discount line that falls
₩8.01M — the DP buys the result by not over-discounting, which is precisely
what section 5.7 predicts of an enter-and-hold policy.

The observed column is here only so the middle column can be judged against
it, and that judgement is worth stating: the model expects legacy to lose
₩19.51M where it actually lost ₩17.11M (**14% pessimistic**) and clears 77.58%
where the world cleared 93.28%. That gap is the model bias the like-for-like
comparison cancels. It also shows why the naive framing is not merely wrong in
principle but wrong in magnitude — DP-under-model against observed reality
gives **−29.3%**, nine points from −38.0%, and nothing about the size or the
direction of that error is predictable in advance.

![IL and clearance, like-for-like](../reports/charts/05_policy_il_and_clearance.png)

Two consequences are pre-committed here rather than improvised in the pilot:
the exploit-only pilot reports clearance and scrap alongside IL from day one,
and a scrap-guardrail breach driven by *this* mechanism is a business
decision about the IL/clearance trade-off, not a system fault to be debugged.
Whether the trade survives contact with real demand is an A/B question;
replay cannot settle it, for the reason just given. Finally, a derivation tool anchors even the *business* thresholds
to measurement: it computes A/B power empirically on actual candidate-
duration blocks of history, and the daily noise floors of the scrap and
margin series — so the last three judgment calls in the system are made
against measured evidence (section 12).

---

### 5.14a The frozen artifacts are one bundle

Six artifacts are fitted in sequence and frozen together, and they are only
meaningful together: `rho` deflates evidence measured against one model's
residuals, the level factors correct that same model, and the prior was
estimated from that model's predictions and that `r_lookup`. Mixing vintages
raises no error — the numbers simply stop describing the same world, silently,
for the whole window. Section 9.2's insistence that only the level multiplier
tracks the world depends on that coherence holding.

**The bundle id is the baseline model version**, not a separate timestamp.
Every downstream artifact is fitted *against* a model, so keying on the model
answers the question that actually arises — "which model was this fitted
against" — and an artifact naming a different model is by definition not part
of the bundle. Each artifact carries a `provenance` block: bundle, creation
time, config version, and the tool that wrote it. Two carry none by design: the
split manifest precedes the model, and the model file is a LightGBM dump with
nowhere to put one, so it *is* the id (recorded in `feature_schema.json`).

`bootstrap.seal` then writes `artifacts/bundle.json` — the agreed id plus a
SHA-256 of every file. This catches what stamps cannot: an artifact edited after
the fact leaves its provenance intact but not its hash. Sealing refuses an
inconsistent set, because a sealed mixed bundle is worse than an unsealed one —
it looks decided.

The practical consequence is that mirror drift (§7) becomes answerable. Before,
a disagreement between `config.dispersion.rho` and `artifacts/rho.json` told you
the two differed but not which was stale, and the obvious remedy — re-paste from
the artifact — is wrong whenever the artifacts on disk are an older bundle than
the model in force: a smaller `deff` then over-counts every future update.
`pipeline.status` reports the bundle line above the mirror line for that reason.

### 5.15 Production assurance — testing the assumptions, not the code

The unit suite checks logic against fixtures. It cannot check the thing that has
actually broken this system every time: an assumption about real data. The
censoring basis, the horizon taken from a row count, scrap read as zero, a stale
`rho` paste — none were logic bugs, and a test that supplies its own inputs
could not have caught any of them. `pipeline.assurance` runs beside section 15
on the same daily cadence and tests the frozen artifacts against the live world.
Each check is built for a failure that would otherwise be **silent**.

| Check | Question | Why nothing else catches it |
| --- | --- | --- |
| `reproduction` | Do logged decisions re-solve to themselves? | The DP is deterministic, so a mismatch means something moved underneath it — config edit, artifact swap, bad deploy, library upgrade. One check, four causes |
| `dispersion` | Is live demand as lumpy as the frozen `r` claims? | Every bounded update assumes it. Demand burstier than `r` says makes each one overconfident, and no business metric moves |
| `correlation` | Is `rho` still the frozen 0.3103? | It divides all accumulated evidence through `deff`. Drift silently rescales every update; the loop looks healthy while being wrong about how much it knows |
| `exploration` | Is the applied price a uniform draw from the affordable set? | The causal claim rests on it entirely. A biased draw keeps prices legal and IL reported — it only stops the evidence being evidence |

Three details carry most of the value.

**Reproduction needs the event to be sufficient, not just complete.** Q at every
tier depends on the whole remaining forecast and the action set depends on the
anchor, so the decision event now carries `mu_ref_path` and `anchor_discount`
(section 16.1). Without them an event says what was decided but not enough to
recompute it, and "the price we logged no longer follows from the inputs we
logged" is the one failure that is never benign.

**The dispersion check uses the two statistics that survive censoring exactly.**
With at least one unit on the shelf, `P(sold = 0) = P(D = 0)` — selling nothing
is never censored — and `P(sold >= q) = P(D >= q)` — selling out *is* the tail.
Both compare against the negative binomial directly, with no correction and no
bias, which a variance comparison could not do. Binned by predicted demand,
because miscalibration flat in `mu` is a level problem and miscalibration that
grows with `mu` is a shape problem, and only the second indicts `r`.

**`rho` is re-measured on the basis it was frozen on** — residuals against raw
`mu` at the *working* elasticity (the prior fallback), never at the posterior
mean. Measuring at a moved posterior would make `rho` drift for a reason that has
nothing to do with the world, and the number would stop being comparable to the
one `deff` came from.

None of these suspend pricing. They report beside the section 15 families with
their own verdict and are read at the operator gate, because the right response
to "the world stopped matching the model" is a human decision. Thin windows
report `INSUFFICIENT` rather than `PASS`: a check that cannot see enough data
must not be mistaken for one that looked and found nothing.

## 6. Data foundation

**Source:** hourly FLC snapshots (date, hour, SKU, FC, inventory, discount,
units sold, base price, realised price, cost, remaining window, category,
subcategory). ~2.37M rows / ~18 usable weeks after filtering in the current
extract, across 16 categories and 5 FCs. A 40-day known-bad-demand window is
excluded. Date splits: a training period for model fitting, a two-week
calibration window (dispersion, level factors, fidelity), and a held-out
test week — split boundaries and the episode-construction rule persisted in
a manifest so every consumer derives identical data.

**Synthetic validation:** a generator reproduces the schema with known
ground-truth elasticity in two modes — `legacy` (reproduces the clock
confound; estimators must *detect* it) and `randomized` (elasticity
identifiable; estimators must *recover* it). The test suite (25 automated
tests) runs the full pipeline against it, asserting among other things: the
filter chain removes exactly the injected dirt, cost floor and monotonicity
hold on every emitted decision, posterior updates are exactly-once,
malformed events quarantine, and shadow outcomes cannot reach the learner.

---

## 7. What running on production data changed

Five findings from the real-data bootstrap iterations materially improved
the design — each caught by a gate or diagnostic doing its job:

1. **Demand-regime awareness in the gate.** Whole-history fidelity mixed the
   spring training regime with the launch-adjacent summer one and made the
   gate unreadable. **Change:** the gate is read on the calibration+test
   window, with per-window and per-week ratios reported so regime structure
   is visible instead of silently averaged.
2. **Slope error must not contaminate level correction.** An early
   correction basis scaled predictions by the prior elasticity, letting a
   wrong prior push level factors below 1. **Change:** level factors are
   fitted exclusively on anchor rows, where the elasticity multiplier is ~1
   by construction; under-fed categories stay uncorrected rather than
   contaminated.
3. **The survivorship confound** (section 3.2). **Change:** entry-rows-only
   identification; the boundary pinning stopped, though on the corrected
   extract no category clears the acceptance checks and all 16 fall back
   honestly.
4. **Per-SKU velocity features** (section 5.4). Adding them improved per-row
   accuracy (hourly MAE 0.405 → 0.373) and cut residual intra-episode
   correlation (deff 4.07 → 3.347, ~18% more information per exploration
   outcome) — while exposing that Tweedie's objective is not sum-calibrated,
   leaving a persistent aggregate level deficit for the calibration factor
   to absorb.
5. **Weekly demand volatility is a measured fact, not noise in the gate.**
   The weekly fidelity series showed the model under-predicting in *every*
   week of five months (range 1.06–1.54, mean ≈ 1.30) with ±8% weekly swing
   — and the two-week calibration window was the single most anomalous
   stretch in the series, which is why a factor fit on it moved the gate the
   wrong way. **Changes:** calibration factors fit on a long window
   (`calibration_fit_window: train+calib`) so no one week dominates, and the
   gate band reset by the owner to **[0.90, 1.10]** (~2σ of the measured
   3-week pooled noise; the original ±5% band was ~1σ — a coin flip on
   natural volatility).

## 8. Measured results (production data, model `baseline-20260809120225`)

Measured after the data-definition corrections in section 12a. Every
episode-terminal figure below differs from the pre-correction run, in the
direction those corrections predicted.

| Quantity | Value |
| --- | --- |
| Calibration gate | **PASS** — `level_bias_at_anchor` **1.0389**, inside the owner band [0.90, 1.10] |
| Sold ratio by window (calibrated) | train 1.0134 / calib 0.8705 / test 1.0193 / all 0.9940, over 2,071,682 rows |
| Post-calibration episode sold ratio | 1.0193 |
| Hourly MAE (gate window) | 0.4399 on the shipped calibrated artifact |
| Share of selling hours with a sale | 24.1% — three in four hours sell nothing |
| Shadow gate | **PASS** — 12,771 decisions, completeness 0.9974, matched 0.9974, cost-floor violations 0, drift ratio 1.0225, solver p95 102 ms |
| Actual IL% (replay sample, 2,000 episodes) | **32.27%** (IL ≈ ₩14.27M), clearance 93.3% |
| DP vs legacy (like-for-like, same demand model) | **−38.0% IL** at **−0.97pp clearance** (77.58% → 76.61% … see section 5.7) |
| DP vs legacy mean discount | 0.1285 vs 0.2935 — the DP opens far shallower and holds |
| Intra-episode deepening | 0% of episodes; median \|ε\| needed 2.429 against 1.0 in use |
| Correlation `rho` / forced hours / implied deff | 0.3103 / 8.563 / **3.347** (fitted-residual basis, `artifacts/rho.json`) |
| IL% clustered SE | **0.002915** (SKU × FC, 71,559 units) |
| A/B minimum detectable effect | **6.75% at 2 weeks** (9 blocks); the duration curve is flat — 6 weeks reaches only 5.74% |
| Elasticity prior | **fallback −1.0 ± 0.6 for all 16 categories — 0 brackets accepted** |
| Exploration `tau` | ₩447.78, pasted from the gate-passing calibrated backtest |
| Would-be learning yield (shadow) | 1.09 bounded updates from the window; 1,837 episodes per update |
| Guardrail 3σ noise, trailing-mean basis | realised margin **13.63%** (robust 14.94%, well behaved); scrap **480%** raw / **153%** robust — outlier-dominated and unusable, see section 12 |

Four of these changed the story rather than the digits, and each is picked up
where it belongs: the **elasticity bracket now fails everywhere** (section
9.3), the **DP's IL advantage tripled while its clearance cost nearly
vanished** (section 5.7), the **A/B became cheap** (section 12), and the
**scrap guardrail lost its yardstick** (section 12).

![calibration gate by window](../reports/charts/03_calibration_gate.png)

![weekly sold-ratio series](../reports/charts/04_weekly_fidelity.png)

## 9. Evaluation gates

### 9.1 Why gates instead of judgment

Certain results must *block the build* rather than parameterise it, and the
blocking conditions were fixed before the numbers were seen. Every gate that
has fired so far fired correctly.

### 9.2 Calibration gate (blocking) — is the demand model usable at all?

The gate judges the frozen model on its **only production responsibility:
the demand level at the reference discount.** Inference always overwrites
price to the reference anchor; every other price is produced by the
parametric layer (`mu_ref × ratio^ε`). The gate metric is therefore
`level_bias_at_anchor` on the launch-adjacent window, within **[0.90,
1.10]** — a band set by the owner at ~2σ of the *measured* weekly demand
volatility (finding 5, section 7). Both choices are owner decisions
(2026-08-09), made after the pooled-at-actual-prices ratio was shown to be
dominated by the elasticity prior's slope (anchor ≈ 1.0 over five months
while the pooled ratio sat at 1.17): a pooled gate was structurally
measuring the one quantity history cannot identify, and would have blocked
launch forever on the prior rather than the model. The pooled ratio and
`slope_ratio_by_discount_gap` remain reported as diagnostics; the slope
itself is validated where it can be — by posterior movement and the A/B —
and the continuous production guard is the daily
`realised_vs_predicted_sold_ratio`. Why the gate matters at all: every
economic quantity is denominated in the demand prediction, and an early
25%-light model priced to under-clear (replay clearance 91% → 50%, scrap
+70%). Level factors are fit on anchor rows over a fit window disjoint from
the gate window, and — critically — on the **censored basis**: sales cannot
exceed inventory, so predictions are compared as `E[min(D, q)]`, the same
quantity the gate measures. An earlier implementation fit against raw `mu`,
which is always the larger number, so factors read systematically low; on a
controlled check a true correction of 1.45 fit as **0.68** — the wrong side
of 1. That single basis mismatch explains why level calibration had appeared
useless on real data (factors below 1 on a model that under-predicts, a gate
that never moved, and one run that got worse). Because the factor scales
`mu` *before* censoring, the censored total moves by less than the factor,
so the factor is solved for by bisection rather than divided out.

Two temporal rules complete the gate's semantics. First, the verdict that
matters is the one on the **freeze-time model**: a monotone anchor-ratio
*trend* across the gate weeks (as opposed to wobble) means the demand level
is in motion and the gated model is stale — August 2026 measured the anchor
climbing 1.04 → 1.73 over four consecutive weeks against a July-trained
model, which is a staleness reading, not a launch verdict, and no band
should be tuned to pass it. (The report's `anchor_ratio_by_rate_history`
triages such a climb first: new-assortment SKUs with no rate history
predicting low is an assortment effect, not a macro trend.) Second, when
the level is measured to move within the MVP window, the level factors are
**re-fit on a schedule** (weekly, trailing window) and applied through the
versioned calibration artifact — the frozen model, and with it price
response and the attribution of posterior movement, stays frozen; only the
level multiplier tracks the world, which is exactly what a multiplier is
for. Status: final retrain + re-gate at the launch freeze, with scheduled
in-window recalibration adopted in response to the measured August trend.

### 9.3 Prior-acceptance gate (blocking) — is the bracket honest?

Orientation must hold (naive ≤ controlled < 0), neither endpoint may sit at
a search bound, and the width-derived std must not be a constant. Any
failure → fallback prior, recorded. Status: applied; **all 16 categories
fell back** on the corrected extract. An earlier run accepted MEAT; the
data-definition fixes in section 12a changed the identifying sample and it no
longer clears the checks. That is the gate doing its job — a bracket that
survives one definition of an episode and not another was never an estimate —
but it removes the single category that would have launched informed, and it
raises the weight the exploration budget carries.

### 9.4 Shadow gate (blocking) — is the pipeline production-ready?

Event completeness > 99%, matched decisions > 99%, zero cost-floor
violations, before any price is applied. Status: **run and passed** —
completeness 1.0000, matched decision rate 1.0000, zero cost-floor
violations, with a drift ratio of 1.0225 at the legacy price across 12,771 decisions (solver p95 102 ms). The verdict
line reads "proceed to exploit-only pilot"; that is the phase-2 entry
condition, not permission to apply prices in phase 1.

## 10. Launch plan

```mermaid
flowchart LR
  A["phase 0<br/>measure"] --> B{"reassessment<br/>gates"}
  B --> C["train + freeze<br/>artifacts"]
  C --> D{"calibration gate<br/>level at anchor"}
  D -- fail --> C2["fit level factors<br/>re-gate, no retrain"] --> D
  D -- pass --> E{"prior acceptance<br/>gate"}
  E --> F["shadow<br/>no prices applied"]
  F --> G{"shadow gate<br/>completeness, cost floor"}
  G -- pass --> H["exploit-only pilot"]
  H --> I["learning pilot<br/>exploration on"]
  I --> J["A/B"] --> K["scale"]
```


| Phase | What happens | Exit gate | Status |
| --- | --- | --- | --- |
| 0. Measurement | historical measurement, config populated | gates reviewed | **Done** |
| 0b. Calibration | fidelity diagnostic, prior estimation | section 9.2 + 9.3 | **Done** — `level_bias_at_anchor` 1.0389, inside [0.90, 1.10] |
| 1. Shadow | decisions logged, no prices applied | section 9.4 | **Done** — completeness 0.9974, matched 0.9974, 0 cost-floor violations |
| 2. Exploit-only pilot | small SKU set, exploration off | price mismatch <1%, finalization SLA | pending |
| 3. Learning pilot | exploration at half budget on pilot set | posterior std falling; spend within budget | pending |
| 4. A/B | full design below | powered duration; no guardrail breach | blocked on owner MDE (evidence says 7.5% at 2 weeks is comfortable) |
| 5. Scale | rollout | positive A/B on IL% | — |

## 11. A/B evaluation

### 11.1 Design

Randomisation unit: **SKU × FC by stable hash** — not episode, because
consecutive episodes of the same SKU × FC share inventory carryover and
would contaminate arms. Allocation 50/50. Primary metric: IL% as a ratio of
sums; analysis by the linearised (delta-method) ratio estimator with
standard errors clustered on the assignment unit, which handles zero-sale
units naturally. Absolute IL is reported alongside. The A/B measures whether
the *policy* beats legacy; it cannot estimate elasticity (both arms are
policies, not price randomisations) — elasticity learning happens inside the
treatment arm. Guardrails: sell-through, waste units, realised margin,
stockout rate; any breach halts the treatment arm. Duration is fixed before
launch and honoured — no early reads.

### 11.2 The pre-committed decision table

| Absolute IL | IL% | Action |
| --- | --- | --- |
| improves | improves | ship |
| improves | flat or worse | **escalate to the product owner** — the system did what it was built to do; acceptability is a business call |
| flat or worse | improves | do **not** ship on IL% alone — the denominator grew because more units sold at deeper discounts |
| flat or worse | flat or worse | do not ship |

The second row is the case this design makes most likely, and it must not be
resolved ad hoc by whoever reads the dashboard.

![A/B duration vs detectable effect](../reports/charts/08_ab_duration.png)

![guardrail noise floors](../reports/charts/09_guardrail_noise.png)

## 12. Open owner decisions — recommendations and tooling

Three thresholds are business decisions that block strict start-up. A
derivation tool (`bootstrap.derive_thresholds`) produces the evidence to set
them from; recommended values:

**Minimum detectable effect for the A/B — recommend 7.5% relative on IL% at
a 2-week duration.** The tool measures the SE **empirically on actual T-week
blocks of history** rather than scaling by √T (which is optimistic under
unit-level clustering). On the corrected extract:

| Duration | Detectable MDE (relative) | √T would predict | Blocks measured |
| --- | --- | --- | --- |
| **2 weeks** | **6.75%** | — | **9** |
| 3 weeks | 6.17% | 5.51% | 6 |
| 4 weeks | 6.31% | 4.77% | 4 |
| 5 weeks | 6.04% | 4.27% | 4 |
| 6 weeks | 5.74% | 3.90% | 3 |
| 8 weeks | 7.62% | 3.38% | 2 |
| 10 weeks | 5.89% | 3.02% | 2 |
| 12 weeks | 7.36% | 2.76% | 1 |

**The striking result is that the curve is flat.** Two weeks detects 6.75%;
six weeks — three times the calendar — reaches 5.74%, where √T promised 3.90%.
Duration buys almost nothing here. The variance is dominated by spread
*between* the 71,559 SKU × FC units, and those same units recur every week, so
extra weeks add correlated observations rather than new clusters. Past six
weeks the block count falls to 1–3 and the rows are noise: eight weeks
measures **worse** than two. Trust the 2–4 week rows only.

**Power is adequate, but not by the margin an earlier draft of this document
claimed.** That draft read the clustered SE as 0.000875 and called the
experiment comfortable. It measures **0.002915** once scrap is counted
properly — IL that carries a real scrap term is *noisier*, not more stable,
because scrap is large and lumpy per episode. The earlier low variance was an
artifact of a metric that was silently dropping ~99% of its own scrap.

Recommend committing to **7.5% at 2 weeks (14 days)**. It is met with 0.75pp
to spare, which is thin against the stated target — but the target is ours to
choose, and the number that matters is the comparison to the effect: the
replay measures a **38%** policy effect against a **6.75%** detection limit, a
**5.6× margin**. Waiting is not the lever; if more power were genuinely
needed, the answer would be more SKU × FC units in the pilot, not more weeks.

**Scrap-deterioration stop threshold — the daily basis has no usable
threshold; set it against the control arm instead.** The intent is unchanged:
a legitimately-functioning system spending 1% of IL on exploration cannot
move scrap far, while the failure mode this guards against — price-holding
miscalibration — breached the equivalent of a 20% threshold several times
over in replay, caught within a day. What the corrected extract shows is that
the *measurement basis* was wrong. The daily scrap series compared against a
trailing 28-day mean has a 3σ noise floor of **480% raw / 153% robust**, and
the tool flags it outlier-dominated. A floor above 1.0 means the series swings
by more than its own level: **no threshold on that basis is both safe and
useful**, and the 20% figure was never defensible against it.

The fix is not a looser number, it is the right comparison.
`pipeline.monitor` already compares **treatment against control on the same
day** whenever both arms are populated, which cancels the common day effect
that dominates this series; only before an A/B exists does it fall back to
the trailing mean. `derive_thresholds` now measures that basis too, as
`guardrail_noise_control_arm_basis`, so the owner sets the threshold against
the yardstick the monitor will actually apply. **Set
`scrap_deterioration_pct` from the control-arm floor on the production
extract, and treat the pre-A/B phases as guarded by the scrap *level* review
in the pilot readout rather than by an automatic daily trigger.**

Two corrections to how that floor is read, both of which change the number:

**The control-arm floor is measured on the smoothed series.** The monitor
averages each arm over `deterioration_smoothing_days` (7 for scrap) and only
then differences them. The first version of `control_arm_noise` differenced
raw daily arms, which measures a floor up to ~√7 wider than the comparison
the monitor performs — so a threshold set from it sits several times above
its true operating noise, and with the 2-day persistence rule on top it
cannot fire at all. Both arms are now smoothed before differencing, in that
order.

**One config value serves both phases, so it must clear the larger of the
two floors.** `guardrail_threshold_recommendation` reports the trailing floor,
the control-arm floor, which one binds, and the verdict. It also stamps
`CLEARS THE FLOOR BUT LIKELY INERT` on anything more than 3× the binding
floor — because clearing the floor is necessary, not sufficient, and a
guardrail that cannot fire is an absent one rather than a conservative one.
If the re-derived scrap number is still large enough to trip that verdict,
the honest response is a different instrument — a longer smoothing window, an
absolute scrap-unit floor, or monitored-and-escalated rather than
auto-stopped — not a number that technically passes.

**Margin-deterioration stop threshold — recommend 15% relative** vs control,
with a 2-day persistence rule. This series is well behaved: 3σ noise
**13.63%**, robust **14.94%**, *not* outlier-dominated. 15% clears the raw
floor by ~10% and sits just under the robust one, so the persistence rule is
what covers the gap rather than headroom. Anything at or below ~13.6% is
inside the noise band and would false-fire on ordinary days — silently
suspending exploration, which is the product. A tighter trigger than 15% must
be paired with a longer persistence window (3+ consecutive days), never
adopted alone.

**The persistence rule is load-bearing, not decorative.**
`monitoring.stop_conditions.persistence_days` (default 2) means a condition
fires only after that many *consecutive* days over threshold. A single day
past a 3σ floor is expected roughly once a year per guardrail; two in a row
essentially never. `pipeline.monitor` computes both series on exactly the
definitions the floors are measured on, and reports the consecutive-day
streak beside each verdict. Until this build the two thresholds had no
evaluation path at all — setting them would have changed nothing.

Calibration principle for both guardrails: a false fire is cheap (it
suspends exploration only; pricing continues), but a threshold that fires
constantly silently kills the learning loop — which is the product.
Thresholds sit at or above the measured 3σ noise **on the basis the monitor
compares against**; tighten via persistence rules, never by dipping below the
floor. `bootstrap.derive_thresholds` re-measures both bases on the current
extract and stamps `TOO TIGHT` on any threshold set beneath its own — run it
before committing the owner values, and treat that verdict as blocking.

## 12a. Multi-day episodes

FLC windows commonly run past midnight — **36-hour windows are common**. The
episode key was `sku_id | fc | date`, so one economic window became two or
three episodes and everything episode-terminal was wrong at each seam. This
is now fixed at the source rather than worked around downstream.

**An episode is a maximal run of consecutive hourly rows for one SKU × FC
over which the source's own `hours_remaining` counter ticks down exactly one
per elapsed hour.** Two signals must agree, because either alone is too weak:
time-contiguity alone would merge two back-to-back windows, and the counter
alone would stitch across a hole in the data, leaving an episode whose row
count disagrees with its clock. Crossing midnight is a one-hour step like any
other — which is the entire point. `episode_id` is now keyed by the window's
first hour, not by a calendar date.

```mermaid
flowchart TD
  S["rows for one SKU x FC,<br/>ordered by timestamp"] --> Q{"timestamp advances<br/>exactly 1 hour?"}
  Q -- no --> N["NEW episode"]
  Q -- yes --> C{"hours_remaining<br/>ticks down exactly 1?"}
  C -- no --> N
  C -- yes --> K["same episode<br/>(midnight is just another hour)"]
```

A precondition had to be enforced first. Two rows sharing a
`(sku, fc, date, hour)` timestamp make two runs start at the same instant,
which collides them into one `episode_id` and produces an "episode" whose
window counter is not monotone. One SKU × FC cannot have two states in the
same hour and there is no principled way to choose between them, so **both
copies are dropped**, recorded as `duplicate_hour_rows_dropped` in the
waterfall. On the synthetic harness this was 114 rows and the sole cause of
every counter anomaly; the resulting hole simply splits the window, which is
loud rather than silent. A test now asserts the postcondition directly: inside
every episode the counter steps down exactly one per row, and the window is
at least as long as the rows held.

Three things moved with it:

| Component | Change |
| --- | --- |
| **Split assignment** | An episode belongs *wholly* to the split its window started in. Slicing by row date put the later hours of a window in a different split from the entry decision that set its price path — the train/calib boundary ran through the middle of episodes |
| **Leakage guard on the velocity features** | Features are read as of the episode's **first** date. Keyed per row, a window's second-day rows read a trailing window ending the previous day — which contains that same episode's first-day sales. The episode was predicting itself |
| **`prior_episode_ref_sales_rate`** | Computed at true episode grain. The daily shift handed a multi-day episode its own earlier day as its "previous episode" |

What this fixes, in order of consequence: the **monotonicity anchor no longer
resets mid-window**, so price can no longer rise across midnight — previously
the continuation was treated as an entry, breaking the one guarantee the
design says is structural. The **DP terminal value** `−cost × q` now fires
once, at the real window end, instead of two or three times, so the planner
no longer over-discounts into a false deadline. **Scrap, IL, IL% and
clearance** no longer count carried-over inventory as scrapped — the IL
baseline and the guardrail noise floors were biased pessimistic by however
much inventory sat at each seam. And `rho` / `mean_forced_hours` are now
measured over whole windows, so `deff` rises and evidence is no longer
over-counted, which was the anti-conservative direction.

### The last hour writes off, it does not report

**`ending_inventory` is always zero on an episode's last row.** When the
window closes the source writes off whatever remains, so the last hour breaks
the inventory chain by design: `ending_inventory == 0` regardless of whether
it equals `starting_inventory − units_sold`. Verified upstream across 953K
episodes — never positive there. On the corrected extract **~13.5% of
episodes end holding stock** (the rest sell out); an earlier ~49.5% figure
came from a different extract and should not be quoted.

Two failure modes follow, and both are silent:

- **Reading the field as scrap reports zero scrap for every episode.** IL
  collapses to discount cost alone, the planner's reward loses the term that
  makes markdown worth doing, the guardrail noise floors are measured on a
  flat-zero series, and nothing anywhere looks broken. This is what the code
  in this repo did until the quirk surfaced.
- **Treating the broken chain as a data error and dropping those episodes**
  discards essentially all the genuine waste and keeps only guaranteed
  sellouts — precisely the wrong sample for estimating a scrap cost.

True leftover is therefore `max(0, starting_inventory − units_sold)` on the
last row, never `ending_inventory`. That formula is also correct wherever the
chain is honest, so it is the only one used: `common.episodes.leftover_units`
is the single definition, and `m6_il_pct`, `m5_censoring`, the replay's
observed-world scrap, `derive_thresholds`, the monitor's IL and guardrail
series, and the window extension all go through it.
`m11_episode_endings` reports
`last_row_ending_inventory_ever_positive` so the convention can be confirmed
on any new extract rather than assumed.

Three knock-ons for the event store, which quarantines any outcome whose
inventory does not reconcile without a documented reason. A **restock** is
inventory going *up* — the next hour opening with more than this one left
behind — not inequality in either direction, since the write-off makes many
hours fail an inequality test. The **write-off is recognised by the zero
itself**, not by position: the source zeroes at its own episode boundary, so
after a window is merged across midnight that row can sit mid-episode for us.
And a partial shortfall (above zero but below the leftover) is **shrink**, now
named `unexplained_shortfall` rather than left undocumented.

That third one is a correction. Leaving it unnamed so it would quarantine and
"stay visible" was the last place the live path treated shrink as an anomaly
while the offline chain treated it as an ordinary event — counted gross,
booked into scrap, gating nothing. A quarantined outcome never lands, so
`event_completeness` fell by the feed's whole shrink rate and the shadow gate
(`min_event_completeness`, 0.99) failed for something no integration work
could fix: it was measuring the source. Measured at ~2.8% of decision hours,
the harness read 0.9718. Quarantine is for what the system cannot interpret;
shrink is interpreted, and the units remain visible through `units_shrink`,
`episode_scrap` and the named reason on the event itself.

**Episodes with an intraday restock are no longer dropped, and no longer gate
anything.** They were dropped whole on the reasoning that mid-window
replenishment breaks the single-inventory-pool assumption the DP's state
transition rests on. It does not: `ending[t] == starting[t+1]`, so the arrival
is already carried forward and the solver simply meets a larger `q` on the
next hour — exactly as it does live, where the policy re-solves hourly. The
drop cost every frozen artifact 2.69pp of the extract's COGS to protect a
solver that reads `dp_eligible` anyway. Only a restock on the FINAL hour still
gates (`final_hour_restock`), and only because the close is then ambiguous.

![episode endings](../reports/charts/02_episode_endings.png)

### An episode is not as long as its window

**Two endings, and one state that is not an ending.** Closure is asked FIRST,
and it is one condition — `ending_inventory == 0` on the last row. Only then
does the leftover say which ending it was, read UNCLIPPED so that a close that
took a restock is not mistaken for a clean sell-out:

    closed   = ending_inventory == 0                     on the last row
    leftover = starting_inventory − units_sold           on the last row, UNCLIPPED

| | Test on the last row | Scrap |
| --- | --- | --- |
| `sold_out_early` | closed, leftover 0 | **none** — nothing left, by fact rather than assumption |
| `completed` | closed, leftover ≠ 0 | **the leftover** — those units were disposed of |
| `not_closed` | `ending_inventory != 0` | **unknown** — excluded, never counted as zero |

`hours_remaining` is not consulted. An earlier version tested the leftover
first and clipped it at zero, which made both a still-running window and a
restocked close read as `sold_out_early`.

**The counter is not the end-of-window signal, and keying scrap to it was a
serious error.** The first version of this classification tested
`hours_remaining == 0` for `completed`. On production data that fires on
**~0.1% of episodes**: `flc_window` is a *nominal* countdown and is still
positive on essentially every final row. Measured across ~356K episodes,
**13.4% end holding stock with the counter still positive** — and under the
counter-keyed rule every one of them was classified `truncated` and dropped.
The result was that **~99% of all real leftover was excluded from every
scrap-bearing statistic**, with only a few hundred episodes contributing
scrap at all.

Confirmed with the business: **when a listing ends with stock on hand, those
units are disposed of and counted as scrap**, whatever the nominal counter
says.

**The backtest sees nothing past the gate window.** `bootstrap.prepare_data.
pre_launch` slices the frame to episodes that opened on or before
`split.test_end` before anything reads it, so `fidelity.by_week` and
`by_window["all"]` stop there and `population.episodes_excluded_after_test_end`
records what was held back. This was not always true, and the two paths that
reached past it were both silent: `policy_replay` and `derive_tau_initial` ran
on the whole frame — so `tau_initial`, a MEASURED launch value, was partly
fitted on the hold-out — and `calibration_fit_window: "all"` resolved to the
whole frame, one config edit from fitting the level factors there.

**Offline, `not_closed` was empty — and then it wasn't.** On the earlier
extract, 356,228 of 356,228 final rows carried `ending_inventory = 0`; not one
reported honest inventory, so every episode had finished and the split was
purely on leftover. On the current 397,764-episode extract that is **no longer
true**: 24,540 final rows lack the sentinel and 13,444 episodes (3.38%) are
`not_closed`.

Two things about that population make it matter more than 3.38% suggests.
They hold **334,622 leftover units against the closed population's 91,096** —
78.6% of everything at risk — because they average 24.9 units each against
3.05, and a big slow-clearing window is exactly the kind still open when an
extract is cut. And they were half-in: excluded from scrap and IL, but
contributing hours to the demand and dispersion fits, so no two figures were
measured on the same rows.

Neither is removed. `prepare_data` flags them instead — `edge_truncated` on
the frame, split from the residue by an exact test — because the half-in
problem is fixed by *including* them consistently, not by deleting them. Their
observed hours are ordinary priced demand; only the ending is missing, and
every consumer of an ending already excludes it on its own (`scrap_units`
returns NaN, `backtest.replay` zeroes scrap under `outcome_known`,
`pipeline.shadow` charges scrap only on `COMPLETED`). Dropping them gave up
the largest, slowest, most heavily stocked windows in the extract to protect a
figure that was already protected.

The split still matters, because the two causes mean different things:

| | What it is | What to do about it |
| --- | --- | --- |
| **edge** | the window still had hours to run at the extract's last hour | nothing — unknowable from this data, and only a longer extract closes it |
| **not edge** | the window ended *inside* the data and no sentinel appeared | investigate — a feed gap or a subset that never writes off, which no re-download fixes |

`m11.not_closed` counts both, so read it beside
`share_of_unclosed_explained_by_edge` in the `dp_eligible` waterfall detail:
near 1.0 and the whole unknown-scrap problem was the extract cut.
`m11.not_closed_by_month` / `not_closed_by_category` then say whether the
residue is one incident, a standing property of the feed, or one corner of the
catalogue. Deleting the residue would have driven that number to zero *by
construction* and hidden a systemic feed problem behind a clean population.

**Worth checking against the flc_window recovery**: the sentinel-free rows
appeared in the same run family that began recovering the "manufacturing"
SKUs. If the kept residue concentrates in those categories, that is the cause.

**The third state exists for production, not for the extract.** Live, the
monitor reads events while episodes are still in flight, and an in-flight
episode's most recent row is *not* a final row — it carries an honest,
non-zero `ending_inventory`. Its leftover is stock **on the shelf, not in the
bin**. Booking it as scrap would count it today and count a different number
for the same episode tomorrow. The source's write-off sentinel — the zero it
writes when a listing closes — is what separates "finished" from "still
running", and `pipeline.monitor` calls the *same* classifier rather than a
rule of its own. (An intermediate version inferred closure from proximity to
the extract's last timestamp; the sentinel supersedes it and handles per-FC
feed lag for free.)

Closure is **one condition and nothing else**: `ending_inventory == 0` on the
last row. There used to be a frame-wide fallback — no sentinel anywhere in the
frame, so treat every episode as closed — on the reasoning that a feed
reporting honest inventory throughout would otherwise move all scrap into
UNKNOWN. It has been **removed**. It failed in the one direction nobody can
see: a feed that *stops* emitting the sentinel reads as perfectly healthy
under it, and that is exactly what happened to the synthetic fixture, which
had never modelled the convention at all and so left the closure path
unexercised for months while every test passed. Missing sentinel now reads as
unclosed everywhere — loudly — and `ending_summary` reports
`write_off_convention_in_force` and `final_rows_without_closure_sentinel` to
name the cause, so the number is not mistaken for a finding about the
business.

Closure and **outcome** are separate axes. Once closed, the sign of the
*unclipped* `starting − sold` on the last row says which close it was:
positive is scrap, zero is censored, negative means stock arrived during the
final hour (`close_outcome` → `scrap` / `censored` / `final_hour_restock`).
Clipping at zero folds the third into the second, so a restocked close reads
as a clean sell-out.

Two consequences worth being explicit about. First, the old rule's effect was
**anti-conservative in measurement and conservative in reporting**: the
observed-world IL baseline was missing nearly all of its scrap term, which
understated the target the policy has to beat. Second, and worse, the two
halves of the system disagreed — `bootstrap.measure` and `derive_thresholds`
excluded those episodes while `pipeline.monitor` counted every one of them.
The scrap noise floor was therefore measured on a few hundred episodes while
the live monitor triggers on tens of thousands, which is the most likely
explanation for the absurd 480%/153% daily floor and for the fact that no
sane scrap threshold could be found. **A population mismatch between the
floor and the trigger is the same class of defect as a smoothing mismatch,
and both are now closed:** the monitor calls the *same* classifier, so an
episode with no closure sentinel — one still running — is excluded there
exactly as it is in the threshold derivation, and the excluded count is
reported alongside the metric.

`m11_episode_endings` reports the split, plus
`share_last_row_counter_at_zero` and
`share_completed_with_counter_still_positive` — the two diagnostics that make
this failure visible on any new extract instead of requiring someone to
notice it.

`validate_state` rejects any decision whose `mu_ref_path` length disagrees
with `hours_remaining`, so a truncated planning horizon fails loudly instead
of silently optimising the wrong window.

### The horizon comes from the window, not from the row count

Because rows stop at zero inventory, an episode's row count is short *because
the item sold out* — so handing the DP that count as its horizon feeds it the
outcome it is supposed to be deciding. Measured on the harness: the row count
and `flc_window + 1` agree on 89.5% of decision rows and on 99.2% of rows
within completed windows, which is why the error hid; where they disagree
the horizon is **short** (9.7% of rows), concentrated exactly in the
fast-selling episodes, and a short horizon brings the terminal scrap penalty
forward and pushes the DP to discount harder than it should.

`common.episodes.extend_to_window` now extends every episode to its full
window before `mu_ref` is predicted, with the added rows marked
`is_observed = False`. The extension is exact rather than an approximation:
every baseline feature is either episode-constant (category, FC, price, and
the velocity features, which are keyed to the episode's *first* date) or a
function of the advancing timestamp. After it, rows-remaining equals
`flc_window + 1` on every row — the invariant `validate_state` enforces on
the live path.

Three details make the extension safe rather than merely convenient.
Synthetic rows carry no sales, so observed-world IL is untouched. **Fidelity,
calibration and every ratio in the gate see only the observed rows** — a
synthetic row would otherwise read as a pure under-prediction and corrupt the
one metric the frozen model is gated on. And the legacy arm's price is
extended by holding its last observed discount, which is the legacy policy's
own ramp-to-cap behaviour, so both arms still run the same horizon and the
comparison stays like-for-like.

**Every figure in section 8 is from the bootstrap re-run after these
corrections**, and they moved as predicted. The IL baseline fell to 32.3% and
moved with it; the clustered SE went the OTHER way once scrap was counted in
full — 0.000875 → **0.002915**, because a real scrap term is lumpy — so the
A/B is adequately powered rather than comfortable;
`deff` fell to 3.347 as whole windows replaced fragments; and the DP's
measured advantage tripled to 38.0% once the planner stopped being handed a
horizon shortened by each episode's own realised sellout. One moved against
expectation and is called out in section 9.3: the elasticity bracket, which
previously accepted MEAT, now fails for all 16 categories.

## 13. Risk register

| # | Risk | Evidence | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| 1 | **Learning throughput** — and the deepening bar in risk 6 sets how far the posterior must travel, not just how fast | Per-outcome information is small (demand ~0.5–1/hr × squared log-price-ratio ~0.01–0.04, ÷ deff 3.347); prior is wide fallback for **all 16** categories; monotonicity concentrates identification at entry | Shadow now emits `learning_yield_would_be` — effective information per episode, episodes per bounded update — so weeks-to-convergence is read off before the pilot, not guessed. Two floors bind separately: evidence (episodes needed) and calendar (the 0.15 step cap with one human-gated update per day means ≥6 days to move the mean 1.0 → 1.9 however much evidence arrives). Levers: raise the budget share, coarser cells. A 21-day flat-posterior alert catches a dead loop | Eng + owner |
| 2 | **Frozen-model drift over Sep–Dec** (seasonality incl. Chuseok; no trend features) | Drift already measured: 1.144 → 0.990 → 1.095 across windows; every economic quantity is denominated in the demand prediction | Final retrain immediately before the launch freeze (gate re-checked); daily drift ratio in shadow and production; pre-register a mid-window recalibration rule now so a drift response is not improvised | Eng |
| 3 | **A/B power** — adequate, and duration is not the lever | SE 0.002915 once scrap is counted in full; 6.75% detectable at 2 weeks against a measured 38% effect (5.6×). The duration curve is nearly flat — 6 weeks reaches only 5.74% where √T promised 3.90%, because variance is between-unit and the same units recur weekly | Empirical duration table from the derivation tool; owner commits to a feasible (effect, duration) pair before launch. If more power is ever needed the lever is more SKU × FC units, not more weeks | Owner |
| 4 | **Metric divergence at readout** — planner optimises IL, business reads IL% | Worked example in 2.3; likeliest A/B outcome is the escalation row | Both metrics + denominators in every cut; divergence flag monitored; decision table pre-committed | Owner |
| 5 | **Single-elasticity misspecification** — threshold-shaped price response averaged into one exponent | Discount-gap diagnostics are noisy/non-monotonic | Residuals logged by discount region so the failure is visible before it is modelled; piecewise response in phase 2 | Eng |
| 6 | **Enter-and-hold at the launch prior** — deepening pays only when \|ε\| > (1−d)/(γ−d), measured median **2.429**, against a prior of 1.0 | Backtest `intra_episode_deepening`: 0% of episodes clear the bar, the DP deepens in none of them; mean discount 0.1285 vs legacy 0.2935; clearance −0.97pp. **Measured sensitivity:** a full bootstrap re-run at a fallback prior of −1.5 produced IDENTICAL prices — mean discount 0.1285, 0% deepened — because 1.5 is still far below the bar. It also cost 26% of the learning rate (`deff` 3.347 → 4.204, since `fit_dispersion` measures residual correlation at the working elasticity) and 1pp of the IL gain (38.0% → 37.0%) | The policy is INSENSITIVE to the prior mean anywhere below ~2.43, so guessing a larger \|ε\| buys no behaviour change and slows the loop that would find the real value. Correct behaviour given the prior, not a bug — but pre-brief the pilot on lower clearance and higher scrap, and track the threshold gap every run. Exploration is the only thing that closes it | Eng + owner |
| 7 | **Multi-day episode fix invalidates the measured baseline** (section 12a) — 36-hour windows are common, so every episode-terminal figure was measured under a broken key | Monotonicity reset mid-window; DP terminal value fired 2-3x per window; carried inventory counted as scrap at each seam | Fixed at the source: episodes are now maximal runs with a consistent `hours_remaining` countdown, split assignment and the feature leakage guard follow the episode. **Full bootstrap must be re-run before any number is quoted** | Eng |
| 8 | **Episode fragmentation from missing source hours** — a single absent hour splits one economic episode into two | Worked example: a BABY FOOD episode runs 06:00–15:00, hour 16 is absent, and the feed resumes at 17:00 with `flc_window` stepping 33→31. The clock and the counter still AGREE (both step 2), but `assign_episode_ids` requires both to step exactly 1, so it starts a new episode. Measured: **2.61% of episodes (8,711) end with no closure sentinel**, holding **27,105 units of ambiguous scrap** against 111,694 counted; median 21 hours nominally unrecorded | Conservative today — ambiguous leftover is excluded rather than invented, and the later fragment usually carries the real outcome, so scrap TOTALS are close to right. What is distorted: episode counts are inflated, and the second fragment's first hour looks like an entry hour when it is mid-episode, which is dirt in exactly the rows section 9.5 identification depends on. Fix is to stitch where clock and counter agree (capped), with interior synthetic rows so `validate_state`'s horizon invariant still holds — deferred to after the launch decision because it changes the analysis population again | Eng |
| 9 | **Model under-prediction from censored training labels** | Anchor under-prediction with median starting inventory ~2 and ~12.6% stocked-out hours | First phase-2 priority: censored-count training | Eng |

## 14. Phase 2 (deferred until the loop demonstrably works)

Priority-ordered by what bootstrap revealed: censored-count model training
(risk 9); episode stitching across missing source hours (risk 8 — relax the
contiguity rule from "both step exactly one" to "clock and counter agree",
capped, with interior synthetic rows); per-category prior acceptance (currently any failure falls the
whole prior back, the conservative reading — and on the corrected extract
every category fails, so nothing is lost by it today); subcategory learning cells with leave-one-out
pooling (deferred, not dismissed — see 5.9: finer cells change no price below
the deepening bar and dilute evidence, so the gain only exists once cells have
moved); automated posterior updates with criteria drafted from observed
operator-gate behaviour; episode-level random effects replacing the deff
deflation; off-policy correction to recover the majority of outcomes
currently discarded as exploitation.

---

## Appendix A — operational quick reference

Charts are generated from the reports, never drawn by hand — re-run
`tools.make_charts` after a bootstrap and every picture in this document
moves with the numbers. A chart that disagrees with the pipeline cannot
exist. Beyond those embedded above, the run also produces the exploration
threshold against the Q-spread (07), the learning-yield calendar floor (10),
the shadow gate against its thresholds (11) and the elasticity bracket by
category (12), all under `reports/charts/`.

```bash
# bootstrap, in order (retrains the model — see AGENTS.md before iterating)
scripts/run_bootstrap.sh data/flc_filtered.parquet

# evidence for the three owner thresholds
python3 -m bootstrap.derive_thresholds --input data/prepared.parquet --mde 0.075

# launch
python3 -m bootstrap.init_posterior
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
#   samples monitoring.shadow_gate.sample_episodes episodes (default 3,000);
#   --max-episodes 0 sweeps everything, for the final pre-launch record

# daily production loop
python3 -m pipeline.update             # monitor only
python3 -m pipeline.update --apply     # human-gated bounded update
python3 -m pipeline.monitor

# regenerate every chart in this document from the current reports
python3 -m tools.make_charts
```

Artifacts are versioned and stamped on every decision event, so any decision
is reproducible from its event alone. Run outputs (`data/`, `reports/`,
`artifacts/`, `events_store*/`) are never committed.

## Appendix B — glossary

| Term | Meaning |
| --- | --- |
| Episode | One SKU × FC selling window: a maximal run of consecutive hourly rows over which the source `hours_remaining` counter decrements by exactly one per elapsed hour. **Deliberately NOT keyed by date** — FLC windows routinely run past midnight, and a date key splits one economic episode into two, resets the price anchor mid-window, and charges the carried-over stock to scrap at the seam |
| IL / IL% | Inventory Loss in currency / IL over full-price value of units sold |
| FLC window | The remaining hours a perishable item may be sold |
| `d_ref` | Category reference discount — the anchor at which the frozen model predicts |
| `d_max` | Feasible discount ceiling `1 − cost/price`; the cost floor in discount space |
| `mu_ref` | Frozen baseline demand prediction at `d_ref` |
| ε (elasticity) | Exponent mapping price ratio to demand; the only quantity learned in production |
| `r` | Frozen negative-binomial dispersion (`Var = mu + mu²/r`) |
| `rho` | Frozen intra-episode demand correlation |
| deff | Design effect deflating correlated within-episode evidence (3.347) |
| `tau` | Currency threshold defining the affordable exploration set |
| Cell | A learning unit: one high-volume category, or the pooled global cell |
| Anchor | The price currently in force; hourly actions may only deepen from it |
| Censored hour | An hour where sales hit inventory — demand is known only as a lower bound |
