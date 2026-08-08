# Perishable Markdown MVP — System Design

**Status:** Implemented; bootstrap validated on production FLC data (2026-08-08)
**Spec:** [`perishable_markdown_mvp_prd.md`](perishable_markdown_mvp_prd.md) — authoritative; section references below point there
**Operations:** [`../AGENTS.md`](../AGENTS.md) — run order, hard rules, gate decision tree

---

## 1. Problem and shape of the solution

Perishable FLC inventory must be marked down through a short selling window
(roughly 10:00–20:00) or scrapped at cost. The legacy policy ramps the
discount deterministically ~1pp/hour on the clock, which makes price
collinear with hour-of-day: **price elasticity cannot be point-identified
from history** (§1). Every naive "learn from the data" approach inherits that
confound, plus a second one this project surfaced during bootstrap: under the
ramp, rows at deep discounts exist *because* earlier hours didn't sell, so
within-episode variation carries survivorship bias toward zero elasticity.

The design therefore splits cleanly:

- **From history, only what history can support** (§2): baseline demand at a
  fixed reference discount, NB dispersion, intra-episode correlation, and a
  *bounded* elasticity prior. All frozen at launch.
- **In production, the one thing history cannot give**: elasticity, learned
  per category from randomized, IL-budgeted exploration inside the treatment
  arm.

Learning rate is the product. Every mechanism below is justified either by
pricing safely or by learning faster.

## 2. Architecture

```
FROZEN (fit offline, unchanged during the MVP window)
  artifacts/baseline_model.txt      mu_ref by context at d_ref     (§9.3)
  artifacts/feature_schema.json     feature order, categorical levels
  artifacts/calibration.json        per-category level factors     (§9.3, gated)
  artifacts/r_lookup.json           NB dispersion by subcategory   (§9.4)
  artifacts/rho.json                intra-episode correlation      (§9.4)
  artifacts/prior.json              elasticity bracket / fallback  (§9.5)

LEARNING (the only thing that updates in production)
  artifacts/posterior.json          epsilon by cell {mean, std, n_obs,
                                    information, version} + processed-outcome
                                    ledger (atomic, exactly-once)   (§10, §13.5)

DECISION PATH (per decision interval)
  state ──validate (§11.4)──▶ feasible tiers (§11.1)
        ──▶ DP over Q(p) (§11.3) ──▶ exploit | explore (§12)
        ──▶ price ──▶ decision event (§16.1)
                          │
                 finalized outcome event (§16.2)
                          │
        censored NB update, deff-deflated, bounded step,
        operator-gated ──▶ posterior (§13–14)
```

| Module | PRD | Responsibility |
| --- | --- | --- |
| `config.yaml` + `common/config.py` | §7 | Every tunable. Strict mode refuses to start on null MEASURED / owner values. |
| `bootstrap/prepare_data.py` | §9.1–9.2 | Source mapping (percent→fraction exactly once), filter chain with waterfall, contiguous-hour episode construction persisted in the split manifest. |
| `bootstrap/measure.py` | §8 | Phase-0 measurements m1–m8, m10; reassessment gates. |
| `bootstrap/train_baseline.py` | §9.3 | Frozen LightGBM/Tweedie `mu_ref`; price features overwritten to `d_ref` at inference; `--fit-calibration` fits anchor-only level factors. |
| `bootstrap/fit_dispersion.py` | §9.4 | Censored-MLE `r` per subcategory (fallback chain, clamp); `rho` on fitted residuals. |
| `bootstrap/estimate_prior.py` | §9.5 | Bracket (naive vs hour-controlled) on **entry rows only**, full search bound, acceptance checks, recorded fallback. |
| `bootstrap/init_posterior.py` | §10 | One-time cell initialisation from prior + weekly volumes; refuses overwrite without `--force`. |
| `pricing/demand.py` | §9.3, §11.3 | `mu(d) = mu_ref·((1−d)/(1−d_ref))^ε`, floored; truncated NB pmf with tail fold. |
| `pricing/dp.py` | §11 | Monotone DP over feasible tiers; absolute-IL reward; entry arms within `d_ref ± entry_window`. |
| `pricing/explore.py` | §12 | Affordable set under currency `tau`; uniform draw; budget and `tau` calibration. |
| `pricing/posterior.py` | §10, §13.4–13.5 | Normal-summary cells, bounded step, atomic commit of revision + consumed outcome IDs. |
| `inference/decide.py` | §11.4, §16.1 | Validation (reject, never an unsafe price); full decision event. |
| `events/store.py` | §16 | Append-only JSONL; duplicate detection; quarantine, never silent drops. |
| `pipeline/update.py` | §13–14 | Censored NB grid update; deff; operator gate (`--apply` refuses on failed event-quality gates). |
| `pipeline/monitor.py` | §15 | Business (ratio-of-sums IL% with denominators), learning, safety series; stop conditions. |
| `pipeline/shadow.py` | §19 | Phase-1 harness: full decision path, no prices applied; exit-gate report. |
| `backtest/` | §17 | Fidelity gate (calib+test window), policy deltas, `tau_initial` derivation. |

## 3. Key design decisions and why

**The planner optimises absolute IL; the business reads IL%** (§3). IL% has
an endogenous denominator (deeper markdowns sell more units and enlarge it),
so a ratio objective can be gamed by discounting low-cost SKUs. The DP reward
is currency; IL% is reported alongside, always as a ratio of sums with its
denominator, never per-episode. The §18 table governs what happens when the
two disagree — the likely case (absolute improves, IL% flat) escalates to the
PRD owner by design.

**The baseline never sees a price gradient.** `mu_ref` is predicted with
price features overwritten to `d_ref`, so the legacy confound cannot enter
the decision path through the demand model. Elasticity enters only through
the learned scalar.

**Exploration is a currency budget, not a probability** (§12). The DP already
prices every feasible tier; exploration selects uniformly among tiers whose
expected IL sacrifice `Q(p*) − Q(p)` is under `tau` (won, not a rate).
Uniformity is the randomisation that makes outcomes clean evidence; the
budget scales down as the posterior narrows but never to zero, so drift stays
detectable. Information per unit of IL is approximately constant (§12.1), so
no SKU class is intrinsically cheap to explore — cost is the only constraint.

**Only exploration outcomes teach** (§13.1). Exploitation prices depend on
the posterior; learning from them feeds beliefs back into themselves. The
shadow phase extends this: its outcomes are stamped
`execution_status="shadow_not_applied"` and are structurally ineligible for
updates, because the recommended price was never in force.

**Evidence is deflated by `deff = 1 + (forced_hours − 1)·rho`** (§13.3).
Hourly outcomes within an episode share an inventory pool and a demand shock;
counting them as independent declares convergence early by a factor of ~4
(measured `rho` 0.377, forced hours 9.13 → deff ≈ 4.07).

**Updates are bounded and exactly-once** (§13.4–13.5). Mean steps clip at
`max_mean_step`, std shrink floors at `max_std_shrink`, and the posterior
revision commits atomically with the outcome IDs it consumed (single
tmp+rename write). A second `--apply` consuming nothing is correct behaviour.
Bounded steps are what make daily updating safe; the operator gate (§14) caps
it at one human-approved step per day for the MVP.

**Safety is structural, not checked-in-passing** (§2.5, §11): the feasible
tier set is constructed from the cost floor, hourly actions are constrained
to the current anchor, and validation rejects the state rather than returning
any price. Cost floor and monotonicity therefore hold on every path,
including forced exploration.

## 4. Decisions forced by real-data bootstrap (2026-08-08)

The first runs against production FLC data changed three things — these are
now load-bearing and encoded in `AGENTS.md`:

1. **The calibration gate is read on the calib+test window, not all history.**
   All-history fidelity (1.1196) mixed the training regime with the
   launch-adjacent one; per-window ratios showed train 1.144 vs calib 0.990
   vs test 1.095 — demand-level drift, not a fixable model bias. Level
   factors are fit on the calibration window, so the gate must be read there
   too. Result: **gate PASS at 1.0247**, no level calibration required
   (`apply_level_calibration: false`).

2. **Level factors are fit on anchor rows only.** An earlier fallback basis
   scaled predictions by the prior elasticity, letting slope error
   contaminate the level factor (factors < 1 fit on July made the all-history
   ratio *worse*, 1.1196 → 1.36). Anchor-only fitting plus the
   `by_window` diagnostic separates level, slope, and drift cleanly.

3. **The elasticity bracket uses entry-hour rows only.** On all rows, every
   category pinned at the −0.05 sign bound — the within-episode survivorship
   artifact, not evidence about demand. On entry rows, MEAT produced an
   accepted interior bracket; 14/16 categories still rejected (boundary) and
   fall back to the wide prior (mean −1.0, std 0.6). Per §9.5, rejection is
   an acceptable outcome; the learning loop needs the prior *not confidently
   wrong*, not good.

Standing cautions from the same runs: `epsilon_max = −0.05` is a **sign
constraint**, never widened; boundary-pinned estimates never inform
`fallback_mean`; before/after fidelity comparisons are valid only with an
identical `baseline_model_version`; and the measured clustered SE
(0.002383, ~6× the phase-0 value) means A/B duration must be re-derived
before `min_detectable_effect_pct` is set.

## 5. Launch sequence and where we are

| Phase (§19) | Exit gate | Status |
| --- | --- | --- |
| 0. Measurement | §8 complete, gates reviewed | **Done** — measured values in config |
| 0b. Calibration | fidelity in [0.95, 1.05] on gate window; prior re-estimated with no boundary acceptance | **Done** — 1.0247 PASS; prior = fallback (recorded) |
| 1. Shadow | completeness > 99%, match > 99%, zero cost-floor violations | **Harness ready** — `python3 -m pipeline.shadow` |
| 2. Exploit-only pilot | price mismatch below threshold, finalization SLA | pending |
| 3. Learning pilot | posterior std falling, exploration within budget | pending |
| 4. A/B | powered duration, no guardrail breach | blocked on owner MDE |
| 5. Scale | positive A/B on IL% | — |

Still blocking strict start-up (owner decisions, never invented by an agent):
`scrap_deterioration_pct`, `margin_deterioration_pct`,
`min_detectable_effect_pct`.

### Shadow phase operation

```bash
python3 -m bootstrap.init_posterior            # once, from artifacts/prior.json
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
```

The harness replays each episode's real state through the production decide
path — the monotonicity anchor entering hour *t* is the legacy discount from
*t−1*, inventory is actual — logs full decision events, builds outcomes from
what legacy actually sold, and reports the three §19 gate checks plus:
would-be exploration spend against `tau`, recommended-vs-legacy discount
deltas, solver latency, and `realised_vs_predicted_sold_ratio` at the legacy
price — the production continuation of the §9.3 calibration gate and the
first place frozen-model drift will show. Watch that ratio daily: the
calib→test trend (0.99 → 1.10) suggests drift is live, and a final retrain
just before freezing at `mvp_window_start` may be warranted (with the gate
re-checked on the new model).

## 6. Deferred (phase 2, §20)

Priority-ordered from what bootstrap showed: censored-count baseline training
(the anchor under-prediction is consistent with inventory-capped labels;
median starting inventory ~2, 12.6% censored hours); per-category bracket
acceptance (MEAT shows partial acceptance is achievable — currently any
failure falls the whole prior back, the strict PRD reading);
subcategory-level cells with leave-one-out pooling; automated updates;
episode-level random effects replacing deff deflation; off-policy correction
to recover exploitation outcomes.
