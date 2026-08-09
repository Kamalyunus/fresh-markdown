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
1. bootstrap.prepare_data --input <raw>       data/prepared.parquet,                  raw FLC parquet
                                              artifacts/split_manifest.json
2. bootstrap.measure --input <raw>            reports/phase0.json                     raw FLC parquet
3. bootstrap.train_baseline --input prepared  artifacts/baseline_model.txt,           prepared
                                              artifacts/feature_schema.json
4. bootstrap.fit_dispersion --input prepared  artifacts/r_lookup.json,                prepared + baseline
                                              artifacts/rho.json
5. bootstrap.estimate_prior --input prepared  artifacts/prior.json                    prepared + baseline + r_lookup
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
```

`scripts/run_bootstrap.sh <raw>` runs 1–6 in order. **It retrains the baseline
every time.** To iterate on one step, run that step's module directly — do not
re-run the whole script.

Shadow phase (§19 — after gates clear, before any price is applied):

```bash
python3 -m bootstrap.init_posterior
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
```

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
python3 -m pipeline.monitor
```

## Hard rules — violating these has already caused wrong conclusions

1. **Never retrain the baseline between two runs you intend to compare.**
   The model is frozen by design (§9.3). Any before/after fidelity comparison
   is void unless `artifact_versions.baseline_model_version` is identical in
   both reports. `--fit-calibration` does NOT retrain; plain
   `train_baseline` and `run_bootstrap.sh` DO.

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

4. **The calibration gate is read on the calib+test window** (the
   `fidelity.gate_window` field in the backtest report), because the level
   factors are fit on the calibration window and correction/evaluation must
   share a demand regime. The all-history ratio in `fidelity.by_window.all`
   is diagnostic only; when demand level drifts between the training period
   and launch, no static factor can (or should) fix it.

5a. **Everything that compares predictions to realised sales uses the
   CENSORED expectation `E[min(D, inventory)]`** — fidelity, the gate, and
   the level factors. Raw `mu` is always ≥ the censored expectation, so
   mixing bases makes a factor read low: measured, a true correction of
   1.45 fits as 0.68 on raw mu — the wrong side of 1, which is why
   calibration used to leave the gate unmoved (or worse). The factor is
   *solved* on that basis, not divided out, because scaling mu before
   censoring moves the censored total by less than the factor.

5. **Level-calibration factors are fit on anchor rows only**, over the
   `calibration_fit_window` (default train+calib — measured weekly demand
   swings ±8%, so a factor fit on one fortnight inherits that fortnight's
   anomaly; the 07-13 calib week measured 1.06 against a five-month mean of
   1.30). The GATE stays on calib+test. A factor **below 1** on a long fit
   window means the model genuinely over-predicts at the anchor —
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

For the two guardrail thresholds, `bootstrap.derive_thresholds` measures the
3σ daily noise floor and stamps `TOO TIGHT` on anything set below it — that
verdict is blocking, not advisory. Measured on production data: scrap 0.0914,
realised margin 0.1336. A margin threshold under ~0.13 fires on ordinary days
and silently suspends exploration, which is the product. Buy sensitivity back
with a persistence rule, never by going under the floor.

## Repo conventions

- Modules are run as `python3 -m package.module` from the repo root.
- `data/`, `reports/`, `artifacts/`, `events_store/` are gitignored run
  outputs — never commit them.
- Synthetic validation: `tools/make_dummy_flc.py --policy randomized` makes
  elasticity recoverable (estimator should RECOVER it); `--policy legacy`
  reproduces the production confound (estimator should DETECT it).
