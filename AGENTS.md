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

Shadow runs on a uniform sample of `monitoring.shadow_gate.sample_episodes`
episodes (default 10,000, drawn before `mu_ref` prediction so the cost scales
with the sample). Override with `--max-episodes N`, or `--max-episodes 0` for
every episode — worth doing once for the final pre-launch record, not for
iteration. A sampled report sets `window.sampled` and adds
`shadow_gate.sampling_caveat`; quote the caveat whenever you quote the zero
violation count.

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

4. **The gate window is whatever `baseline_model.calibration_gate_window`
   says** — currently `test`; the run records it as `fidelity.gate_window`,
   so read that field rather than assuming. The rule that matters is that it
   must be **DISJOINT from `calibration_fit_window`** (currently
   `train+calib`), or the gate grades its own fit. The all-history ratio in
   `fidelity.by_window.all` is diagnostic only; when the demand level drifts
   between the training period and launch, no static factor can (or should)
   fix it.

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

**`dispersion.rho` and `dispersion.mean_forced_hours_per_episode` must be
re-pasted from `artifacts/rho.json` after every retrain.** They set `deff`,
which divides accumulated information in `pipeline.update`, so a paste left
over from a previous model version mis-weights every posterior step for the
whole window — silently, and in the direction of slower learning. Strict
start-up now refuses to run on divergence, but the check only fires in
strict mode: re-paste as part of the retrain, not when something breaks.
Take them from `artifacts/rho.json` (fitted against the model's own
residuals), never from phase 0's `m3_intra_episode_correlation`, which is a
category × hour proxy computed before any model exists and says so in its
own `note`.

For the two guardrail thresholds, `bootstrap.derive_thresholds` measures the
3σ daily noise floor and stamps `TOO TIGHT` on anything set below it — that
verdict is blocking, not advisory. Measured on production data: scrap 0.0914,
realised margin 0.1336. A margin threshold under ~0.13 fires on ordinary days
and silently suspends exploration, which is the product. Buy sensitivity back
with a persistence rule, never by going under the floor.

## What counts as a usable episode

`bootstrap.prepare_data` runs a deterministic, auditable filter chain; the
waterfall in `artifacts/split_manifest.json` records rows and episodes after
every step. **Almost every filter drops the WHOLE EPISODE, not the offending
row** — a hole punched mid-window re-segments into a spurious short episode,
which is worse than losing the episode.

| Step | Scope | Drops |
| --- | --- | --- |
| `duplicate_hour_rows_dropped` | rows (both copies) | two states for one sku x fc x hour; no way to choose, and they collide two runs into one episode id |
| `exclusion_window_removed` | episode | any episode with ANY hour in the known demand-issue window |
| `discount_out_of_range_dropped` | episode | discount outside [0,1] — the percent->fraction conversion applied twice or not at all |
| `negative_quantities_dropped` | episode | negative inventory, sales or cost |
| `null_category_dropped` | rows | missing category/subcategory (no reference discount, no dispersion cell) |
| `zero_base_price_dropped` | rows | `original_price` still null/zero after ffill+bfill within the episode |
| `negative_window_dropped` | episode | any `hours_remaining < 0` |
| `window_too_long_dropped` | episode | `hours_remaining` above `data.max_window_hours` (48) — flc_window carries very large values from upstream data issues |
| `below_cost_dropped` | episode | any hour whose OFFERED price is under cost — legacy already violated the floor, so the episode is not evidence about a system that cannot. Test `original_price × (1 − discount)`, NEVER `applied_price`: the source zeroes that on zero-sale rows (~78% of rows), so a filter reading it is blind on exactly those and below-cost hours survive to be rejected one-by-one at decision time |
| `non_priceable_dropped` | episode | `cost >= original_price`, i.e. `d_max <= 0`: no feasible tier exists |
| `units_gt_inventory_dropped` | episode | sales exceed the inventory on hand |
| `contiguous_episodes_built` | — | re-segmentation, not a filter: episode count can RISE here because earlier drops split windows |
| `restocked_episodes_dropped` | episode | an hour opens with more stock than the previous hour left — mid-window replenishment breaks the one-inventory-pool assumption the DP rests on. Runs AFTER re-segmentation; across a data gap the jump would read as a restock |

Restocks are detected on the inventory CHAIN (`next starting_inventory >
max(0, this starting_inventory - units_sold)`), never by comparing against
`ending_inventory` — that field is zeroed at the window close, so an equality
test would flag every episode's last hour. In production a restock can still
happen after the fact; the outcome records it with `adjustment_reason`.

**Any outcome whose inventory does not reconcile MUST name a reason or it is
quarantined** — and a quarantined outcome never lands, so event completeness
drops and the shadow gate fails. Exactly two reasons are legitimate:

- `intraday_restock` — `ending_inventory > max(0, starting - sold)`
- `episode_close_write_off` — `ending_inventory < leftover` on the episode's
  **final observed row**. This is ~49.5% of episodes. An integration that
  omits it quarantines roughly half its final-hour outcomes and fails the
  gate for what looks like a pipeline defect.

Key the write-off to the LAST OBSERVED HOUR, not to `hours_remaining == 0`.
The source writes off when the EPISODE closes, and a sold-out-early or
truncated episode closes before its window does — gating on the window
counter leaves those quarantining. `pipeline.shadow.adjustment_reason` is the
one implementation; production integrations should use the same rule.

`ending_inventory < leftover` PART-WAY THROUGH an episode is unexplained
inventory loss and is left undocumented on purpose, so it quarantines. Do not add a blanket reason to
make the count go to zero — the quarantine file is the only place that
failure is visible.

The window cap is load-bearing beyond data hygiene: `hours_remaining` drives
episode identification, the DP horizon, and the synthetic tail that
`extend_to_window` generates. An unbounded counter would generate an
unbounded frame, so the extension raises rather than hanging if a frame ever
reaches it above the cap. The bad value is dropped, never clamped — clamping
would invent a window end the data never recorded.

Postconditions are asserted by test, not assumed: discount in [0,1],
non-negative quantities, sales <= inventory, `d_max > 0`, category present,
no hour inside the exclusion window, `hours_remaining` within the cap, and a
monotone window counter inside every episode.

## Multi-day episodes

FLC windows commonly run past midnight; 36-hour windows are common. An
episode is therefore NOT keyed by date. It is a maximal run of consecutive
hourly rows for one sku x fc over which the source `hours_remaining` counter
ticks down exactly one per elapsed hour (`prepare_data.assign_episode_ids`).
Both signals are required: time alone merges back-to-back windows, the
counter alone stitches across missing rows.

Three things follow the episode, not the row date, and must stay that way:

1. **Split assignment** — an episode belongs wholly to the split its window
   STARTED in (`split_frames`), or the train/calib boundary runs through the
   middle of an episode.
2. **Velocity features** — read as of the episode's FIRST date. Per-row
   keying lets a window's second-day rows read a trailing window containing
   that same episode's first-day sales.
3. **`prior_episode_ref_sales_rate`** — computed at episode grain; a daily
   shift hands a multi-day episode its own earlier day.

Duplicate `(sku, fc, date, hour)` rows are dropped outright
(`duplicate_hour_rows_dropped` in the waterfall) — both copies, since there is
no way to pick. Left in, they collide two runs into one `episode_id` and the
window counter stops being monotone.

An episode ends at the window end OR at zero inventory, whichever comes
first, so its row count is NOT its window length. `m11_episode_endings` in
`reports/phase0.json` splits the three cases: `completed` (hours_remaining
hit 0 -- leftover inventory IS scrap), `sold_out_early` (no scrap by
construction), `truncated` (no recorded window end -- scrap UNKNOWN).

**`ending_inventory` IS ALWAYS ZERO ON AN EPISODE'S LAST ROW** -- the source
writes off the remainder when the window closes (~49.5% of episodes end this
way). Reading it as scrap reports ZERO SCRAP EVERYWHERE and silently deletes
the scrap term from IL; dropping those episodes as "broken chain" keeps only
guaranteed sellouts. Scrap is `max(0, starting_inventory - units_sold)` on the
last row -- `common.episodes.leftover_units` is the only definition, and
`scrap_units` wraps it, returning NaN for truncated episodes so a sum cannot
treat unknown as zero. Truncated episodes are excluded from scrap
and IL aggregates, with the excluded share reported.

The DP horizon comes from the WINDOW, not the row count. `backtest` and
`pipeline.shadow` call `common.episodes.extend_to_window` before predicting,
which appends the hours a sold-out episode never recorded (marked
`is_observed = False`). Without it the horizon is short precisely because the
item sold out -- lookahead bias on ~10% of decision rows, biased toward
over-discounting fast movers.

Anything measuring the model against reality -- fidelity, the calibration
gate, the likelihood, IL -- must filter to `is_observed`. A synthetic row has
no sales and reads as a pure under-prediction. Sort by
`["episode_id", "date", "hour_of_day"]`, never `hour_of_day` alone: a window
running past midnight comes out scrambled.
`validate_state` rejects any decision whose `mu_ref_path` length disagrees
with `hours_remaining`.

**Any figure measured before this change is void** — IL baseline, clearance,
rho/deff, guardrail noise floors, replay IL. Re-run the full bootstrap.

## Refreshing the numbers in the docs and deck

`docs/design.md` and `docs/perishable_markdown_tech_deck.pptx` quote ~25
measured quantities that go stale on every re-run (the launch-freeze retrain
changes most of them). Do not hunt through the JSON:

```bash
python3 -m tools.deck_numbers --backtest reports/<gate-passing>.json \
    [--shadow reports/shadow.json] [--phase0 reports/phase0.json] \
    [--thresholds reports/thresholds.json]
```

It prints each quantity tagged with the slide(s) and design-doc sections that
carry it. Missing reports print `--` rather than failing. Only ever quote a
**gate-passing** backtest — the same rule that governs pasting `tau_initial`.

## Repo conventions

- Modules are run as `python3 -m package.module` from the repo root.
- `data/`, `reports/`, `artifacts/`, `events_store/` are gitignored run
  outputs — never commit them.
- Synthetic validation: `tools/make_dummy_flc.py --policy randomized` makes
  elasticity recoverable (estimator should RECOVER it); `--policy legacy`
  reproduces the production confound (estimator should DETECT it).
