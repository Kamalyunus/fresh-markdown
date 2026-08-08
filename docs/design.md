# Perishable Markdown MVP — System Design

**Status:** Implemented; bootstrap validated on production FLC data; calibration gate PASSED (2026-08-08)
**Audience:** Technical leadership review
**Spec:** [`perishable_markdown_mvp_prd.md`](perishable_markdown_mvp_prd.md) is the authoritative requirements document; section references (§) point there
**Operations:** [`../AGENTS.md`](../AGENTS.md) — run order, hard rules, gate decision tree

---

## 1. Executive summary

This system prices perishable FLC (fresh-limited-clearance) inventory through
its final selling window, replacing a legacy policy that ramps discounts
deterministically ~1pp/hour on the clock. It minimises **Inventory Loss (IL)**
— discount given away on units sold plus scrap cost on units unsold at expiry
— currently running at **~34.6% of full-price sales value** in the markdown
cohort.

The central technical fact shaping the design: **price elasticity cannot be
estimated from our own history.** The legacy ramp makes price collinear with
hour-of-day, and bootstrap analysis surfaced a second, subtler confound
(within-episode survivorship — see §4.3 below). Any system that claims to have
learned price response from this data has actually learned the clock. The
design therefore uses history only for what it can support — baseline demand,
demand variance, correlation structure — and learns elasticity **in
production**, from deliberately randomized price perturbations whose total
cost is capped at **1% of markdown IL** per day.

Current state: all offline gates have cleared on production data. The
calibration gate passed at 1.0247 (band 0.95–1.05). The shadow-phase harness
is built and ready to run — decisions logged against live data, no prices
applied. Three business thresholds (§8) remain open and block launch; they
are owner decisions, and this document provides recommended values with a
data-derivation tool.

The honest risk summary for leadership: the *safety* engineering is strong
(cost floor and price monotonicity are structurally unviolable; every gate
that has fired so far fired correctly). The open question is the *speed* of
the learning loop within the 13-week MVP window — quantified in §9.1, with
the shadow phase designed to answer it before any price is applied.

---

## 2. Business objective and metric politics

### 2.1 What is optimised

The planner minimises expected **absolute IL** per episode (one SKU × FC ×
day selling window):

```
IL = Σ_hours (original_price − applied_price) × units_sold     (discount cost)
   + cost × unsold_inventory_at_expiry                          (scrap cost)
```

Absolute IL is additive across hours, which makes it a valid dynamic-
programming reward with no transformation, and it is the currency amount the
business is accountable for.

### 2.2 What is reported — and why they can disagree

The headline metric is **IL% = IL / (original_price × units_sold)**. Its
denominator is *endogenous*: deeper markdowns sell more units and enlarge it.
Minimising IL and minimising IL% are therefore different optimisations (§3.3
gives a worked example where they choose opposite prices). This is a
deliberate, documented trade: a ratio objective can be "improved" by
discounting low-cost SKUs harder purely to grow the denominator, which is not
a behaviour we want a planner to learn.

The consequence is stated plainly rather than discovered later: **the A/B is
read on a metric the planner does not optimise.** The most likely readout is
"absolute IL improved, IL% roughly flat" — and the §18 decision table routes
exactly that case to the PRD owner as a business call, not a technical
verdict. Every IL% figure in the system is a ratio of sums reported with its
denominator (per-episode ratios are undefined for zero-sale episodes, which
are ~12% of episodes, and are never computed), and absolute IL is reported
alongside in every cut.

### 2.3 Why learning rate is the product

The system's value is not the day-one policy — it is that the policy improves
from its own decisions. Every design choice below is justified by one of two
things: pricing safely, or learning faster. Features serving neither were cut
(§4.2 of the PRD): subcategory posteriors, Thompson sampling, weekly
retraining, immutable deployment bundles, stochastic replay.

---

## 3. Architecture

```
FROZEN (fit offline, unchanged during the MVP window)
  artifacts/baseline_model.txt      mu_ref: demand at the reference discount (§9.3)
  artifacts/feature_schema.json     feature order and categorical levels
  artifacts/calibration.json        per-category level factors (§9.3; NOT applied — gate passed without)
  artifacts/r_lookup.json           negative-binomial dispersion by subcategory (§9.4)
  artifacts/rho.json                intra-episode demand correlation (§9.4)
  artifacts/prior.json              elasticity prior: bracket or recorded fallback (§9.5)

LEARNING (the only thing that updates in production)
  artifacts/posterior.json          elasticity by cell {mean, std, n_obs, information,
                                    version} + processed-outcome ledger (§10, §13.5)

DECISION PATH (per hourly decision interval)
  state ──validate (§11.4: reject, never an unsafe price)
        ──▶ feasible tiers from the cost floor (§11.1)
        ──▶ DP over Q(price) for every tier (§11.3)
        ──▶ exploit argmax | explore affordable set (§12)
        ──▶ price ──▶ decision event (§16.1, ~30 fields)
                          │
                 finalized outcome event (§16.2)
                          │
        censored NB update, deff-deflated, bounded step,
        operator-gated ──▶ posterior (§13–14)
```

### 3.1 Module map

| Module | PRD | Responsibility |
| --- | --- | --- |
| `config.yaml` + `common/config.py` | §7 | Every tunable in one file; code contains no numeric literals for anything tunable. Strict mode refuses to start while any MEASURED or owner value is null. |
| `bootstrap/prepare_data.py` | §9.1–9.2 | Source mapping (discount percent→fraction exactly once), 7-step filter chain with an auditable row/episode waterfall, contiguous-hour episode construction persisted in the split manifest. |
| `bootstrap/measure.py` | §8 | Phase-0 measurement suite (m1–m8, m10) and reassessment gates. |
| `bootstrap/train_baseline.py` | §9.3 | Frozen LightGBM/Tweedie `mu_ref`; price features overwritten to `d_ref` at inference; `--fit-calibration` fits anchor-only level factors. |
| `bootstrap/fit_dispersion.py` | §9.4 | Censored-MLE NB `r` per subcategory (fallback chain, high-value clamp); global `rho` on fitted residuals. |
| `bootstrap/estimate_prior.py` | §9.5 | Elasticity bracket (naive vs hour-controlled) on entry rows, full search bound, acceptance checks, recorded fallback. |
| `bootstrap/init_posterior.py` | §10 | One-time cell initialisation; refuses to overwrite learning state without `--force`. |
| `bootstrap/derive_thresholds.py` | §8, §15.4, §18 | Empirical A/B duration vs MDE; 3σ noise floors for the scrap/margin guardrails (see §8). |
| `pricing/demand.py` | §9.3, §11.3 | `mu(d) = mu_ref·((1−d)/(1−d_ref))^ε`, floored; truncated NB pmf with tail-mass fold and diagnostics. |
| `pricing/dp.py` | §11 | Monotone DP over feasible tiers; absolute-IL reward; entry arms within `d_ref ± entry_window`. |
| `pricing/explore.py` | §12 | Affordable-set construction under currency `tau`; uniform draw; daily budget and `tau` calibration. |
| `pricing/posterior.py` | §10, §13.4–13.5 | Normal-summary cells; bounded step; atomic commit of revision + consumed outcome IDs. |
| `inference/decide.py` | §11.4, §16.1 | Nine-point state validation; full decision-event emission. |
| `events/store.py` | §16 | Append-only JSONL; duplicate detection; malformed events quarantined with reasons, never silently dropped. |
| `pipeline/update.py` | §13–14 | Censored NB grid update; deff deflation; operator gate — `--apply` refuses on failed event-quality gates. |
| `pipeline/monitor.py` | §15 | Business (ratio-of-sums IL% with denominators), learning, and safety series; stop-condition evaluation. |
| `pipeline/shadow.py` | §19 | Phase-1 harness: full decision path against live data, no prices applied; exit-gate report. |
| `backtest/` | §17 | Calibration-gate fidelity (calib+test window), policy deltas, `tau_initial` derivation. |

### 3.2 Event contracts and exactly-once learning

Every decision emits an event carrying the complete pricing context —
including the `reference_mu` the decision used, so learning never recomputes
a prediction that feature-pipeline drift could silently alter (§13.1). Every
decision interval gets exactly one finalized outcome event. The posterior
store persists its revision and the outcome IDs it consumed in a **single
atomic write** (tmp + rename), so a crash between "learn" and "mark
processed" cannot double-count evidence; a second `--apply` consuming nothing
is verified behaviour, not an accident. Malformed events are quarantined with
the validation failure attached and surfaced to monitoring.

---

## 4. The identification problem — and what bootstrap proved about it

### 4.1 The known confound

Under the legacy ramp, discount deepens as the evening demand peak arrives.
Price and hour-of-day move together (within-episode correlation ~0.8 in
synthetic reproduction), so demand lift attributable to the evening is
statistically indistinguishable from demand lift attributable to the
discount. History can bound elasticity; it cannot point-identify it.

### 4.2 How the design responds

- The **baseline model never exposes a price gradient**: `mu_ref` is always
  predicted with price features overwritten to the category reference
  discount `d_ref`, so the confound cannot leak into the decision path
  through the demand model. Elasticity enters only as one learned scalar per
  category: `mu(d) = mu_ref × ((1−d)/(1−d_ref))^ε`.
- History supplies a **bracket prior** (§9.5): two deliberately-biased
  estimates (no hour control → too elastic; hour fixed effects → too flat)
  whose known bias directions bracket the truth. Acceptance checks test
  orientation and sufficiency; **rejection falls back to a wide prior and is
  an acceptable, recorded outcome** — the loop needs the prior *not
  confidently wrong*, not good.
- Truth comes from **production randomization** (§5 below).

### 4.3 What running on real data changed (all encoded in AGENTS.md)

Three findings from the first production-data bootstrap materially improved
the design; each was caught by a gate doing its job:

1. **Demand-regime drift, not model bias.** All-history fidelity read 1.1196
   (model 11% light) and failed the gate — but per-window ratios showed
   train 1.144 / calib 0.990 / test 1.095: the demand *level* shifted between
   the spring training period and the launch-adjacent summer window. A
   July-fit correction applied to an all-history evaluation made things
   mechanically worse. **Fix:** the calibration gate is read on the
   calib+test window — the regime the level factors are fit on and the one
   launch will see. Result: **PASS at 1.0247, no level calibration needed**
   (`apply_level_calibration: false`).
2. **Slope error must not contaminate level correction.** An early
   calibration basis scaled predictions by the prior elasticity, letting a
   wrong prior push level factors below 1. **Fix:** level factors are fit
   exclusively on anchor rows (|discount − d_ref| ≤ half a tier), where the
   elasticity multiplier is ~1 by construction; under-fed categories stay
   uncorrected rather than contaminated.
3. **A second confound: within-episode survivorship.** Fitting the bracket on
   all rows pinned every category at the −0.05 sign bound — because under
   the ramp, a row at a deep discount exists *precisely because* earlier
   hours didn't sell. Deep-discount observations are adversely selected
   toward low-demand episodes, biasing elasticity toward zero. **Fix:** the
   bracket uses entry-hour rows only (§9.5's same-hour cross-episode rule).
   After the fix, MEAT produced an accepted interior bracket; 14/16
   categories still rejected honestly and use the fallback prior
   (mean −1.0, std 0.6).

Standing rules derived from the same runs: `epsilon_max = −0.05` is a sign
constraint and is never widened (positive elasticity is structurally
unrepresentable, §10.4); boundary-pinned estimates never inform the fallback
mean; fidelity before/after comparisons are valid only with an identical
frozen `baseline_model_version`.

---

## 5. The decision core

### 5.1 Feasible tiers and the DP (§11)

Per episode: `d_max = 1 − cost/original_price`, and the action set is every
`tier_step` (2.5pp) multiple in `[0, d_max]` — the cost floor is enforced by
*construction of the action set*, not by a check that could be skipped.
Hourly actions are restricted to tiers at or deeper than the current anchor
(price never rises within an episode); the entry decision is unanchored
within `d_ref ± 0.10`, and carries most of the identifying variation.

The DP state is (anchor, integer inventory, hours remaining):

```
Q(anchor, q, h, p) = Σ_k P(D=k | r, mu(p)) · [ min(k,q)·(−(P₀−p)) + V(p, q−min(k,q), h−1) ]
V(anchor, q, h)    = max over feasible p ≤ anchor of Q(anchor, q, h, p)
V(·, q, 0)         = −cost · q
```

Demand is negative-binomial with frozen dispersion `r` (Var = mu + mu²/r),
truncated at 25 units with tail mass folded into the last bucket and tail
diagnostics emitted. State scale is tiny (≤ ~20 tiers, ≤ 12 hours, ≤ ~30
units), so evaluation is exhaustive; measured solver latency p95 is
milliseconds.

### 5.2 Exploration as a P&L line item (§12)

The DP already prices every feasible tier, so exploration is a *selection*,
not a separate mechanism:

```
p*          = argmax Q(p)
cost(p)     = Q(p*) − Q(p)            expected IL sacrificed, in currency
affordable  = { p ≠ p* : cost(p) ≤ tau }
```

If the affordable set is non-empty, the price is drawn **uniformly at
random** from it. Uniformity is the point: any state-dependent choice of
forced price reintroduces the endogeneity that makes legacy history
unusable. There is no exploration probability schedule, floor, or ceiling
anywhere in the system.

`tau` is a currency threshold, self-calibrating daily so realised spend
tracks the budget: `budget = 1% of markdown IL × clip(posterior_std/0.40,
0.25, 1.0)` — spend shrinks to a quarter as the posterior converges, never to
zero, so drift stays detectable. `tau_initial` was derived from replay as the
quantile of the Q-spread whose implied daily spend matches the budget:
**202.8 KRW at the 27.5th percentile — implied spend 1,271/day vs budget
1,271/day**. The theory note that makes the budget-only control sound:
information about ε scales as `mu·(log price ratio)²` and so does the IL cost
of the perturbation, so information *per won* is roughly constant — no SKU
class is intrinsically cheap to explore, and cost is the right (and only)
rationing device.

### 5.3 Learning update (§13)

Daily batch over **exploration outcomes only** — exploitation prices depend
on the posterior, so learning from them feeds beliefs back into themselves.
The likelihood is censored: exact NB probability when demand was observed,
survival probability `P(D ≥ inventory)` when the hour stocked out — stockout
sales are never treated as exact demand, and zero-sale hours are retained
through the exact `P(D=0)` term (they carry the signal at shallow
discounts).

Evidence is deflated by a design effect before it can trigger an update:

```
deff = 1 + (mean_forced_hours_per_episode − 1) × rho  =  1 + 8.134 × 0.377 ≈ 4.07
```

Hours within an episode share an inventory pool and a demand shock; counting
them as independent declares convergence ~4× early. Updates apply when
accumulated effective information reaches `information_increment = 12`, and
each step is bounded (mean moves ≤ 0.15, std shrinks ≤ 25%, floored at 0.05)
with any clipped bound flagged for operator review. For the MVP, a human
approves each day's update (`pipeline.update --apply`), and the command
refuses when event-quality gates fail. Bounded steps are what make frequent
updating safe; the operator gate caps it at one step per day.

Cells are per-category for high-volume categories (≥ 250 episodes/week) with
everything else pooled into a global cell — roughly ten learning cells,
assigned once at launch. Fewer cells learn faster; subcategory resolution is
deliberately deferred.

---

## 6. Validation and current results

### 6.1 Measured values (production data, 2026-08-08)

| Quantity | Value | Source |
| --- | --- | --- |
| Fidelity gate (calib+test) | **1.0247 — PASS** [0.95, 1.05] | backtest |
| Sold ratio by window | train 1.144 / calib 0.990 / test 1.095 | backtest `by_window` |
| Actual IL% (replay sample) | 34.64% (IL ≈ ₩14.7M / 2,000 episodes) | backtest |
| `rho` / forced hours / deff | 0.3772 / 9.134 / ≈ 4.07 | fit_dispersion |
| `tau_initial` | ₩202.8 (q0.275 of Q-spread) | backtest |
| IL% clustered SE (full window) | 0.002383 | measure m6 |
| Prior | fallback −1.0 ± 0.6 (14/16 rejected; MEAT interior bracket accepted) | estimate_prior |

### 6.2 What the replay says — and what it cannot say

The DP replay under the current (weak, fallback) prior holds price relative
to legacy (clearance 50% vs 91%). Per §17.5 this pattern is expected when the
elasticity prior is conservative and **is not evidence about the policy** —
replay under-predicting response will always flatter price-holding. The A/B
is the only evidence of policy quality; replay's jobs are the calibration
gate, `tau_initial`, and DP sanity — all complete.

### 6.3 Test suite

23 automated tests: unit coverage of the decision core (tier construction,
DP monotonicity and terminal values, exploration selection and budget
arithmetic, bounded step) plus an end-to-end run over synthetic data with
known ground-truth elasticity — the generator reproduces both the legacy
confound (estimators must *detect* it) and a randomized policy (estimators
must *recover* ε). Invariants asserted include: cost floor and monotonicity
on every emitted decision, exactly-once posterior updates, quarantine of
malformed events, and shadow outcomes being ineligible for learning.

---

## 7. Launch plan and status

| Phase (§19) | Exit gate | Status |
| --- | --- | --- |
| 0. Measurement | §8 complete, gates reviewed, MEASURED config populated | **Done** |
| 0b. Calibration | Fidelity in band on gate window; prior re-estimated, no boundary acceptance | **Done** — 1.0247 PASS; fallback prior recorded |
| 1. Shadow | Completeness > 99%, match > 99%, **zero** cost-floor violations | **Harness ready** — `python3 -m pipeline.shadow` |
| 2. Exploit-only pilot | Price mismatch < 1%, finalization within SLA | pending |
| 3. Learning pilot | Posterior std falling in ≥ 1 category; spend within budget | pending |
| 4. A/B | Powered duration (§8 below), no guardrail breach | blocked on owner MDE |
| 5. Scale | Positive A/B on IL%; automation criteria drafted from observed behaviour | — |

The shadow harness replays each live episode's real state (actual inventory;
the monotonicity anchor entering hour *t* is the legacy price from *t−1*)
through the production decide path, logs full decision events, and builds
outcomes from what legacy actually sold. Shadow outcomes are stamped
`execution_status="shadow_not_applied"` and are structurally ineligible for
learning — the recommended price was never in force. Beyond the exit gate,
the report gives three things the later phases need: would-be exploration
spend against `tau` (validating the budget calibration), recommended-vs-
legacy discount deltas (the first look at how different the policy actually
is), and `realised_vs_predicted_sold_ratio` at the legacy price — the
production continuation of the calibration gate and the first place
frozen-baseline drift will show.

### 7.1 A/B design (§18)

Unit: SKU × FC by stable hash (episodes share inventory carryover and would
contaminate). Allocation 50/50. Primary metric: IL% as ratio of sums;
analysis via the linearised (delta-method) ratio estimator with clustering on
the assignment unit — which handles zero-denominator units naturally.
Absolute IL reported alongside; the four-cell decision table in §18 governs
disagreement, and the likely "absolute improves / IL% flat" cell escalates to
the PRD owner by design.

---

## 8. Open owner decisions — recommendations and derivation tooling

Three config values are business decisions that block strict start-up. They
should be set from data, and `bootstrap.derive_thresholds` produces the
evidence (`python3 -m bootstrap.derive_thresholds --input data/prepared.parquet
--mde 0.075`):

**`ab_test.min_detectable_effect_pct` — recommended 0.075 (7.5% relative on
IL%).** The measured full-window clustered SE (0.002383) implies a detectable
effect of ~1.3pp absolute (~3.9% relative) *only with all ~18 weeks of data*;
a real A/B runs shorter and loses precision. The tool measures the SE
**empirically on actual T-week blocks of history** (√-time scaling is
optimistic under SKU×FC clustering) and reports, per candidate duration, the
smallest detectable effect — the owner picks the (MDE, duration) pair that
fits the window. Prior arithmetic suggests ~5 weeks at 7.5%; 5% likely does
not fit the MVP window, and the PRD's earlier "0.7% MDE" claim (built on a
6×-smaller SE) is obsolete. Duration must be honoured once set — no early
reads.

**`monitoring.stop_conditions.scrap_deterioration_pct` — recommended 0.20
(20% relative vs control arm / trailing 28 days, ratio of sums).** Scrap is
~7% of IL at 91% clearance; a legitimately functioning system, spending 1% of
IL on exploration, cannot move scrap by anything near 20% — while the failure
mode this guards against (a price-holding miscalibration) moved clearance
91→50% in replay, a 5–10× breach caught in a day. The tool computes the
actual 3σ daily noise of the scrap-rate series and flags a threshold set
below it.

**`monitoring.stop_conditions.margin_deterioration_pct` — recommended 0.10
(10% relative vs control) with a 2-day persistence rule.** Markdown-cohort
margins are thin (cost ratio ~0.66, mean discount ~31%), so daily relative
margin is noisy on small cohorts; persistence, not a looser threshold, is the
correct noise control. The tool reports the measured 3σ level to validate.

Calibration principle for both guardrails: a false fire is cheap (it
suspends *exploration only* — exploitation pricing continues), but a
threshold that fires constantly silently kills the learning loop, which is
the product. Thresholds sit at or above 3σ of daily noise; tightening happens
via persistence rules, not by dipping below the noise floor.

---

## 9. Risk register

| # | Risk | Evidence | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| 1 | **Learning throughput** — posterior may converge too slowly for the 13-week window | Per-outcome information is small (mu ~0.5–1 × (log ratio)² ~0.01–0.04, ÷ deff 4.07); prior is wide fallback for 14/16 categories; monotonicity confines most identification to entry decisions | Quantify weeks-to-convergence from shadow's would-be exploration stats **before** phase 3; levers if short: concentrate budget on entry decisions, raise `budget_share_of_il`, coarser cells. `alert_posterior_std_flat_days: 21` catches a dead loop early | Eng + PRD owner |
| 2 | **Frozen-baseline drift over Sep–Dec** (seasonality incl. Chuseok; no trend features) | Live drift already measured: train 1.144 → calib 0.990 → test 1.095; every economic quantity is denominated in `mu_ref` | Final retrain immediately before freeze at `mvp_window_start` (gate re-checked on that model); daily `realised_vs_predicted_sold_ratio` in shadow and production; pre-register a mid-window recalibration rule now so a drift response is not improvised | Eng |
| 3 | **A/B power** — measured SE is 6× the phase-0 assumption | SE 0.002383 vs 0.000383; duration may not fit the window at small MDEs | `derive_thresholds` empirical duration table; owner picks a feasible (MDE, duration) pair before launch, not after | PRD owner |
| 4 | **Metric divergence at readout** — planner optimises IL, business reads IL% | §3.3 worked example; likeliest A/B outcome is the escalation row | Both metrics + denominators in every cut; `divergence_flag` monitored; §18 table pre-commits the decision process | PRD owner |
| 5 | **Scalar ε misspecification** — threshold-shaped price response averaged away | Known limitation (§21.1); slope bins in measurement 10 are noisy/non-monotonic | Residuals logged by discount region so the failure is visible before it is modelled; phase-2 piecewise response | Eng |
| 6 | **Baseline level bias from censored labels** | Anchor under-prediction with median starting inventory ~2 and ~12.6% censored hours | First phase-2 priority: censored-count training (§20) | Eng |

---

## 10. Phase 2 (deferred until the loop demonstrably works, §20)

Priority-ordered by what bootstrap revealed: **censored-count baseline
training** (risk 6); **per-category bracket acceptance** (MEAT shows partial
acceptance is achievable — currently any category failing falls the whole
prior back, the strict PRD reading); subcategory cells with leave-one-out
hierarchical pooling; automated posterior updates with criteria drafted from
observed operator-gate behaviour; episode-level random effects (Gauss-Hermite)
replacing deff deflation; off-policy correction to recover the ~large
majority of outcomes currently discarded as exploitation.

---

## Appendix A — operational quick reference

```bash
# bootstrap (PRD §1a order; retrains the baseline — see AGENTS.md before iterating)
scripts/run_bootstrap.sh data/flc_filtered.parquet

# owner-threshold evidence
python3 -m bootstrap.derive_thresholds --input data/prepared.parquet --mde 0.075

# launch
python3 -m bootstrap.init_posterior
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json

# daily production loop
python3 -m pipeline.update             # monitor only
python3 -m pipeline.update --apply     # operator-gated bounded update
python3 -m pipeline.monitor
```

Artifacts are versioned and stamped on every decision event
(`baseline_model_version`, `posterior_version`, `config_version`), so any
decision is reproducible from its event alone. Run outputs (`data/`,
`reports/`, `artifacts/`, `events_store*/`) are never committed.

## Appendix B — glossary

| Term | Meaning |
| --- | --- |
| Episode | One (SKU, FC, date) selling window of contiguous hours |
| IL / IL% | Inventory Loss (currency) / IL over full-price sales value of units sold |
| `d_ref` | Category reference discount; the anchor where the baseline predicts |
| `d_max` | Feasible ceiling `1 − cost/price`; the cost floor in discount space |
| `mu_ref` | Frozen baseline demand prediction at `d_ref` |
| ε | Elasticity scalar; the only quantity learned in production |
| `r`, `rho` | Frozen NB dispersion; frozen intra-episode correlation |
| deff | Design effect deflating correlated within-episode evidence (≈ 4.07) |
| `tau` | Currency threshold defining the affordable exploration set |
| Cell | A learning unit: one high-volume category, or the pooled global cell |
