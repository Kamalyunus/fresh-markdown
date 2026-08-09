# Perishable Markdown MVP — System Design

**Status:** Implemented; validated on production FLC data; calibration and shadow gates PASSED (2026-08-09)
**Audience:** Technical leadership review — this document is self-contained and requires no companion reading

---

## 1. Executive summary

This system prices perishable FLC (fresh-limited-clearance) inventory through
its final selling window, replacing a legacy policy that ramps discounts
deterministically ~1 percentage point per hour on the clock. It minimises
**Inventory Loss (IL)** — discount given away on units sold, plus scrap cost
on units unsold at expiry — currently running at **35.6% of full-price sales
value** across the markdown cohort (356,114 episodes).

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
The calibrated model clears that gate (`level_bias_at_anchor` 1.0395), and
the shadow phase has since run to completion on live data with no prices
applied — 66,484 decisions, perfect event completeness, zero cost-floor
violations. Two stop-condition thresholds and the A/B minimum detectable
effect (section 12) remain open owner decisions; they are the only things
between here and the exploit-only pilot.

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
one **episode**: a (SKU, FC, date) run of contiguous selling hours, typically
10:00–20:00, with small starting inventory (median ≈ 2 units). Each hour, a
discount is chosen; whatever is unsold at window end is scrapped at cost.
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
adjacent hours within an episode. After that fix, one high-volume category
(MEAT) produced a well-behaved interior estimate — evidence the mechanism,
not the data, was the problem.

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
contiguous selling hours, and runs a fixed seven-step filter chain —
(1) drop the known bad-data window, (2) drop null category/subcategory,
(3) recover zero base prices by fill-within-episode and drop episodes with
none, (4) drop episodes with negative remaining-window values, (5) drop
episodes ever priced below cost, (6) drop episodes with units sold exceeding
inventory, (7) preserve intraday restocks — emitting a row/episode waterfall
after every step. **Why the paranoia:** the three source-schema traps all
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
bounded by the window end date. A per-category multiplicative correction
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
it. At the measured cost ratio (~0.66) that bar is **|ε| ≈ 1.7–1.9**, and
censoring at a median starting inventory of 2 pushes the true switch point
higher still. **Against the launch prior of −1.0, the DP is therefore
structurally an enter-and-hold policy**: on the synthetic harness the median
threshold is 1.89 against |ε| = 1.0 in use, and the DP deepens intra-episode
in 0% of episodes. This is not a defect — if demand really is that
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
starting point on production data: `tau` = ₩203.09 (the 22.7th percentile of
the Q-value spread), implying ₩1,271/day of exploration spend against a
₩1,271/day budget on the replay sample.

### 5.9 Posterior store — small, atomic, exactly-once

One record per learning cell — a Normal summary (mean, std), observation and
information counters, and a version — plus the ledger of consumed outcome
IDs, in a single atomically-written file. **Why a Normal summary rather than
a stored distribution grid:** the grid exists only inside the update
computation; persisting moments keeps storage trivial and makes the bounded
step well-defined, and the pricing path reads two floats. **Why ~10 category
cells plus one pooled global cell:** learning time scales with cell count;
categories above 250 episodes/week earn their own cell, everything below
reads and feeds the global cell, assignment fixed at launch. **Why the
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
  at `1 + 8.134 × 0.3183 ≈ 3.59`. Without this one line, the system would
  report convergence three-and-a-half times too early. Both inputs are
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
it.** The DP arm shows 12.0% less IL than the legacy arm, but also **3.25pp
less clearance** — it holds price where the legacy ramp would have cut, sells
fewer units, and scraps more. That is a coherent trade (discount saved
exceeds scrap added, which is exactly what minimising absolute IL is supposed
to find), and it is the *same* trade the scrap-deterioration guardrail exists
to police. A 3.25pp clearance loss on a ~91% base is roughly a third more
unsold units in relative terms — comfortably enough to trip a 20% scrap
guardrail if it reproduces in production. Two consequences, both
pre-committed here rather than improvised in the pilot: the exploit-only
pilot reports clearance and scrap alongside IL from day one, and a scrap
guardrail breach driven by *this* mechanism is a business decision about the
IL/clearance trade-off, not a system fault to be debugged. Whether the
trade survives contact with real demand is an A/B question; replay cannot
settle it, for the reason just given. Finally, a derivation tool anchors even the *business* thresholds
to measurement: it computes A/B power empirically on actual candidate-
duration blocks of history, and the daily noise floors of the scrap and
margin series — so the last three judgment calls in the system are made
against measured evidence (section 12).

---

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
   identification; one category now brackets cleanly, the rest fall back
   honestly.
4. **Per-SKU velocity features** (section 5.4). Adding them improved per-row
   accuracy (hourly MAE 0.405 → 0.373) and cut residual intra-episode
   correlation (deff 4.07 → 3.589, ~13% more information per exploration
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

## 8. Measured results (production data, 2026-08-09, feature-set v2)

| Quantity | Value |
| --- | --- |
| Weekly sold-ratio series | every week > 1: range 1.06–1.54, mean ≈ 1.30, σ ≈ ±8% |
| Sold ratio by window (uncalibrated) | train 1.295 / calib 1.117 / test 1.312 |
| Calibration gate | **PASS** — `level_bias_at_anchor` 1.0395 inside the owner band [0.90, 1.10] (post-calibration fidelity 1.0055) |
| Shadow gate | **PASS** — 66,484 decisions, completeness 1.0000, matched 1.0000, cost-floor violations 0, drift ratio 0.9745, solver p95 24ms |
| DP vs legacy (like-for-like, same demand model) | **−12.0% IL**, at **−3.25pp clearance** — the IL win is partly bought with scrap (see below) |
| Guardrail 3σ daily noise floors | realised margin **13.36%**; scrap **914%** — outlier-dominated, see section 12 |
| Hourly MAE (gate window) | 0.4053 on the shipped calibrated artifact (0.373 uncalibrated — calibration trades per-hour error for aggregate level) |
| Actual IL% (full cohort, 356,114 episodes) | **35.58%** |
| Actual IL% (replay sample of 2,000 episodes) | 34.64% (IL ≈ ₩14.7M) |
| Correlation `rho` / forced hours / implied deff | 0.3183 / 9.134 / **3.589** (was 4.07; fitted-residual basis, `artifacts/rho.json`) |
| IL% clustered SE (full ~18-week window) | 0.002383 |
| Elasticity prior | fallback −1.0 ± 0.6 for 14/16 categories; MEAT interior bracket accepted |
| Exploration `tau` | ₩203.09, taken from the gate-passing report (earlier failing runs ranged ₩187–203; only a passing report may be pasted) |

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
failure → fallback prior, recorded. Status: applied; 14/16 fell back, by
design.

### 9.4 Shadow gate (blocking) — is the pipeline production-ready?

Event completeness > 99%, matched decisions > 99%, zero cost-floor
violations, before any price is applied. Status: **run and passed** —
completeness 1.0000, matched decision rate 1.0000, zero cost-floor
violations, with a drift ratio of 0.9745 at the legacy price across 66,484 decisions (solver p95 24ms). The verdict
line reads "proceed to exploit-only pilot"; that is the phase-2 entry
condition, not permission to apply prices in phase 1.

## 10. Launch plan

| Phase | What happens | Exit gate | Status |
| --- | --- | --- | --- |
| 0. Measurement | historical measurement, config populated | gates reviewed | **Done** |
| 0b. Calibration | fidelity diagnostic, prior estimation | section 9.2 + 9.3 | **Done** — `level_bias_at_anchor` 1.0395, inside [0.90, 1.10] |
| 1. Shadow | decisions logged, no prices applied | section 9.4 | **Done** — completeness 1.0000, matched 1.0000, 0 cost-floor violations |
| 2. Exploit-only pilot | small SKU set, exploration off | price mismatch <1%, finalization SLA | pending |
| 3. Learning pilot | exploration at half budget on pilot set | posterior std falling; spend within budget | pending |
| 4. A/B | full design below | powered duration; no guardrail breach | blocked on owner MDE |
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

## 12. Open owner decisions — recommendations and tooling

Three thresholds are business decisions that block strict start-up. A
derivation tool (`bootstrap.derive_thresholds`) produces the evidence to set
them from; recommended values:

**Minimum detectable effect for the A/B — recommend 7.5% relative on IL% at
a 2-week duration.** The measured full-window clustered SE (0.002383) implies
~1.3pp absolute detectable *only with all ~18 weeks of data*; a real A/B is
shorter and loses precision, so the tool measures the SE **empirically on
actual T-week blocks of history** rather than scaling by √T (which is
optimistic under unit-level clustering). The measured table:

| Duration | Detectable MDE (relative) | Blocks measured |
| --- | --- | --- |
| **2 weeks** | **5.54%** | 9 |
| 3 weeks | 8.82% | 6 |
| 4 weeks | 6.98% | 4 |
| 5 weeks | 7.20% | 4 |
| 6 weeks | 7.23% | 3 |
| 8 weeks | 5.63% | 2 |
| 10 weeks | 5.71% | 2 |
| 12 weeks | 6.46% | 1 |

Two things to read carefully. First, this **overturns the earlier arithmetic**
that suggested ~5 weeks for 7.5%: on real blocks, 7.5% is detectable in two
weeks, and 5% is within reach — the tool recommends 2 weeks. Second, the
table is **non-monotonic**, which is impossible for a real power curve:
longer windows cannot lose precision. The cause is visible in the right-hand
column — the block count collapses from 9 to 1, so the long-duration rows are
estimating an SE from one or two samples and are mostly noise. Trust the
2–4 week rows; treat everything from 6 weeks on as uninformative rather than
as evidence that a longer test is worse. The owner picks the (effect,
duration) pair from the reliable rows; the duration is then fixed.

**Scrap-deterioration stop threshold — recommend 20% relative** vs the
control arm (or trailing 28 days), as a ratio of sums. Scrap is ~7% of IL at
91% clearance; a legitimately-functioning system spending 1% of IL on
exploration cannot move scrap anywhere near 20%, while the failure mode this
guards against (price-holding miscalibration) breached the equivalent of
this threshold 5–10× over in replay — caught in a day. The measured 3σ daily
noise of the raw scrap series came back at **9.1386 — 914% relative**,
because a handful of low-volume days dominate the ratio. No usable threshold
sits above a floor larger than the level itself, so the tool now reports a
MAD-based robust floor alongside the raw one and flags the series as
outlier-dominated. The **robust floor is 18.9%**, and that is the number the
decision rests on: 20% clears it by roughly 6%, not the comfortable multiple
scrap first appeared to offer. That is acceptable *only* because of the
persistence rule — see below.

**Margin-deterioration stop threshold — recommend 15% relative** vs control,
with the same 2-day persistence rule. Markdown-cohort margins are thin (cost
ratio ~0.66, mean discount ~31%) and the daily realised-margin series is
correspondingly noisy: the measured 3σ daily noise is **13.58%**. Any
threshold at or below ~13.5% is *inside* the noise band and would false-fire
on ordinary days — silently suspending exploration, which is the product. 15%
clears the floor by ~10%; the persistence rule, not a tighter number, is what
buys sensitivity back. If the owner wants a tighter trigger than 15%, it must
be paired with a longer persistence window (3+ consecutive days), never
adopted alone.

**Both thresholds sit close to their floors, which makes the persistence rule
load-bearing rather than decorative.**
`monitoring.stop_conditions.persistence_days` (default 2) means a condition
fires only after that many *consecutive* days over threshold. A single day
past a 3σ floor is expected roughly once a year per guardrail; two in a row
essentially never. `pipeline.monitor` computes the daily scrap and
realised-margin series on exactly the definitions the noise floors were
measured on, compares treatment against the control arm (falling back to a
trailing 28-day mean before the A/B), and reports the consecutive-day streak
beside each verdict. Until this build the two thresholds had no evaluation
path at all — setting them would have changed nothing.

Calibration principle for both guardrails: a false fire is cheap (it
suspends exploration only; pricing continues), but a threshold that fires
constantly silently kills the learning loop — which is the product.
Thresholds sit at or above the measured 3σ daily noise; tighten via
persistence rules, never by dipping below the noise floor.
`bootstrap.derive_thresholds` re-measures both noise floors on the current
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
episodes — never positive there — and **~49.5% of episodes end by write-off**
rather than clean sellout.

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

One knock-on: a restock is detected as inventory going *up* — the next hour
opening with more stock than this one left behind — not as inequality in
either direction. The write-off makes the last hour of every episode fail an
inequality test, which would have flagged all of them as restocks.

**Episodes with an intraday restock are dropped whole.** Mid-window
replenishment breaks the single-inventory-pool assumption the DP's state
transition rests on, and the demand those extra units meet is not the demand
the episode's price path was chosen for. The check runs after re-segmentation,
because across a data gap the inventory jump would read as a restock.

### An episode is not as long as its window

Rows stop at the window end **or at zero inventory, whichever comes first**,
so an episode's row count is not its window length and a last row carrying
`hours_remaining > 0` is usually a **sell-out**, not missing data. Three
endings, and they are economically opposite:

| Ending | Test on the last row | Scrap | Measured share |
| --- | --- | --- | --- |
| `completed` | `hours_remaining == 0` | leftover inventory **is** scrapped | 85.4% |
| `sold_out_early` | `hours_remaining > 0`, inventory 0 | none, by construction | 10.9% |
| `truncated` | `hours_remaining > 0`, inventory left | **unknown** — no recorded window end | 3.7% |

Every scrap figure in the system used to take the last row's
`ending_inventory` regardless, which charges the truncated episodes' unsold
units to scrap on no evidence. `common.episodes` now classifies the ending
once and returns scrap as **NaN** for truncated episodes, so a sum cannot
silently treat unknown as zero; those episodes are excluded from scrap and IL
aggregates and the excluded share is reported. This corrects `m6_il_pct` (the
IL baseline), the replay's observed-world scrap, and `derive_thresholds` —
which was measuring the guardrail noise floors on the contaminated series.
`m11_episode_endings` reports the split and how much a naive scrap figure
would have overstated.

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

**Re-run the full bootstrap on production data before quoting any number in
this document.** Every episode-terminal figure here — the 35.58% IL baseline,
clearance, `rho`/`deff`, the guardrail noise floors, and the replay's IL
comparison — was measured under the date-keyed definition and is expected to
move.

## 13. Risk register

| # | Risk | Evidence | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| 1 | **Learning throughput** — and the deepening bar in risk 6 sets how far the posterior must travel, not just how fast | Per-outcome information is small (demand ~0.5–1/hr × squared log-price-ratio ~0.01–0.04, ÷ deff 3.589); prior is wide fallback for 14/16 categories; monotonicity concentrates identification at entry | Shadow now emits `learning_yield_would_be` — effective information per episode, episodes per bounded update — so weeks-to-convergence is read off before the pilot, not guessed. Two floors bind separately: evidence (episodes needed) and calendar (the 0.15 step cap with one human-gated update per day means ≥6 days to move the mean 1.0 → 1.9 however much evidence arrives). Levers: raise the budget share, coarser cells. A 21-day flat-posterior alert catches a dead loop | Eng + owner |
| 2 | **Frozen-model drift over Sep–Dec** (seasonality incl. Chuseok; no trend features) | Drift already measured: 1.144 → 0.990 → 1.095 across windows; every economic quantity is denominated in the demand prediction | Final retrain immediately before the launch freeze (gate re-checked); daily drift ratio in shadow and production; pre-register a mid-window recalibration rule now so a drift response is not improvised | Eng |
| 3 | **A/B power** — measured SE is 6× the original assumption | 0.002383 vs 0.000383; small effects may not fit the window | Empirical duration table from the derivation tool; owner commits to a feasible (effect, duration) pair before launch | Owner |
| 4 | **Metric divergence at readout** — planner optimises IL, business reads IL% | Worked example in 2.3; likeliest A/B outcome is the escalation row | Both metrics + denominators in every cut; divergence flag monitored; decision table pre-committed | Owner |
| 5 | **Single-elasticity misspecification** — threshold-shaped price response averaged into one exponent | Discount-gap diagnostics are noisy/non-monotonic | Residuals logged by discount region so the failure is visible before it is modelled; piecewise response in phase 2 | Eng |
| 6 | **Enter-and-hold at the launch prior** — deepening pays only when \|ε\| > (1−d)/(γ−d) ≈ 1.7–1.9, against a prior of 1.0 | Backtest `intra_episode_deepening`: median threshold 1.89 vs \|ε\| 1.0 in use; DP deepens in 0% of episodes; clearance −3.25pp | Correct behaviour given the prior, not a bug — but pre-brief the pilot on lower clearance and higher scrap, and track the threshold gap every run. Exploration is the only thing that closes it; a wider action set cannot | Eng + owner |
| 7 | **Multi-day episode fix invalidates the measured baseline** (section 12a) — 36-hour windows are common, so every episode-terminal figure was measured under a broken key | Monotonicity reset mid-window; DP terminal value fired 2-3x per window; carried inventory counted as scrap at each seam | Fixed at the source: episodes are now maximal runs with a consistent `hours_remaining` countdown, split assignment and the feature leakage guard follow the episode. **Full bootstrap must be re-run before any number is quoted** | Eng |
| 8 | **Model under-prediction from censored training labels** | Anchor under-prediction with median starting inventory ~2 and ~12.6% stocked-out hours | First phase-2 priority: censored-count training | Eng |

## 14. Phase 2 (deferred until the loop demonstrably works)

Priority-ordered by what bootstrap revealed: censored-count model training
(risk 6); per-category prior acceptance (one category already brackets
cleanly — currently any failure falls the whole prior back, the
conservative reading); subcategory learning cells with leave-one-out
pooling; automated posterior updates with criteria drafted from observed
operator-gate behaviour; episode-level random effects replacing the deff
deflation; off-policy correction to recover the majority of outcomes
currently discarded as exploitation.

---

## Appendix A — operational quick reference

```bash
# bootstrap, in order (retrains the model — see AGENTS.md before iterating)
scripts/run_bootstrap.sh data/flc_filtered.parquet

# evidence for the three owner thresholds
python3 -m bootstrap.derive_thresholds --input data/prepared.parquet --mde 0.075

# launch
python3 -m bootstrap.init_posterior
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
#   samples monitoring.shadow_gate.sample_episodes episodes (default 10k);
#   --max-episodes 0 sweeps everything, for the final pre-launch record

# daily production loop
python3 -m pipeline.update             # monitor only
python3 -m pipeline.update --apply     # human-gated bounded update
python3 -m pipeline.monitor
```

Artifacts are versioned and stamped on every decision event, so any decision
is reproducible from its event alone. Run outputs (`data/`, `reports/`,
`artifacts/`, `events_store*/`) are never committed.

## Appendix B — glossary

| Term | Meaning |
| --- | --- |
| Episode | One (SKU, FC, date) selling window of contiguous hours |
| IL / IL% | Inventory Loss in currency / IL over full-price value of units sold |
| FLC window | The remaining hours a perishable item may be sold |
| `d_ref` | Category reference discount — the anchor at which the frozen model predicts |
| `d_max` | Feasible discount ceiling `1 − cost/price`; the cost floor in discount space |
| `mu_ref` | Frozen baseline demand prediction at `d_ref` |
| ε (elasticity) | Exponent mapping price ratio to demand; the only quantity learned in production |
| `r` | Frozen negative-binomial dispersion (`Var = mu + mu²/r`) |
| `rho` | Frozen intra-episode demand correlation |
| deff | Design effect deflating correlated within-episode evidence (3.589) |
| `tau` | Currency threshold defining the affordable exploration set |
| Cell | A learning unit: one high-volume category, or the pooled global cell |
| Anchor | The price currently in force; hourly actions may only deepen from it |
| Censored hour | An hour where sales hit inventory — demand is known only as a lower bound |
