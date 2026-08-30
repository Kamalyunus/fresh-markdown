# Agent operating guide — Perishable Markdown MVP

This file is deliberately short: it holds only what an agent must obey
without looking anything up. The authoritative specification — including the
rationale and the measured incident behind every rule below — is
**`docs/design.md`** (§ numbers refer to it). Superseded approaches live in
`docs/learnings.md`. When this guide and the design doc disagree, the design
doc wins. Doc/chart tooling has its own guide: `docs/maintaining_docs.md`.

## Non-negotiables

Each one-liner is binding on its own; the cited section carries the full
statement and the incident that created the rule.

1. **Never retrain the baseline between two runs you intend to compare** —
   a comparison is valid only when `artifact_versions.baseline_model_version`
   matches in both reports. `--fit-calibration` does NOT retrain; plain
   `train_baseline` and `run_bootstrap.sh` DO. (§5.4)
1a. **Changing the elasticity prior invalidates `rho`, `deff` and the level
   factor** — re-run `fit_dispersion` onward and re-paste the mirrors. (§5.6)
2. **`posterior.epsilon_max` (−0.05) is a sign constraint, never a bound to
   widen** — positive elasticity must remain unrepresentable. (§5.6)
3. **A boundary solution is not an estimate** — a fit pinned at a search
   bound means the likelihood ran off the support. (§5.6)
4. **The calibration gate window must be DISJOINT from the fit window** —
   read `fidelity.gate_window`, never assume it. (§9.2)
5. **Level factors are fit on anchor rows only** over
   `calibration_fit_window`; a factor below 1 on a long window means genuine
   over-prediction — investigate before applying. (§9.2)
5a. **Every prediction-vs-sales comparison uses the CENSORED expectation
   `E[min(D, inventory)]`** — never raw `mu`; mixing bases flips factors to
   the wrong side of 1. (§5.4, §9.2)
5b. **The training label is censored on purpose** — do not "fix" it by
   filtering the training set; the level factor is the other half of that
   decision. (§5.4)
6. **Only the level component may be corrected multiplicatively** — a
   sold-ratio degrading with `|discount − d_ref|` is slope error: re-estimate
   the prior, never scale `mu_ref`. (§9.2)
7. **Elasticity identification uses entry-hour rows only** — same-hour
   cross-episode variation, never within-episode. (§5.6)
8. **`config.yaml` is the single source of every tunable** — no numeric
   literals for tunables in code; a tunable without a config key is a review
   failure. (§5.1)
9. **IL% is always a ratio of sums, with its denominator and absolute IL
   alongside** — per-episode IL% is undefined and must never be averaged.
   (§2.3)
10. **`pipeline.update --apply` is the operator gate** — never work around a
    refusal; a second `--apply` consuming nothing is correct, not a bug.
    (§5.11)
11. **The discount column is percent in raw data and a fraction after
    `prepare_data`** — converted exactly once; only `measure` and
    `prepare_data` accept raw. (§5.2)
12. **Never add within-episode lag sales, `hours_remaining`, or extra price
    features to the demand model** — lags are mediators of the episode's own
    price path; the SKU rate features are computed in `prepare_data`. (§5.4)
13. **Never gate on a condition a constraint already handles** — a loud
    refusal counted in a report beats a silent removal upstream. (§5.2)
14. **INTEGRITY defects DROP; ECONOMIC conditions FLAG (`dp_eligible=False`)
    and keep** — the test is "can the demand model see it?". Conflating the
    two once cost >70% of the extract's COGS. (§5.2)
15. **Cut this data by episode, never by row** — always
    `common.episodes.window_slice`; a row-level date cut manufactures
    orphan episodes at the midnight seam. (§12a)
16. **Nothing pre-launch may see past `split.test_end`** — the hold-out is
    read once, by `pipeline.shadow`, and never tuned on. (§5.13, §9)
17. **A number a procedure solves for is not evidence about that number** —
    a bisection reports 1.00× on any population; grade fitted quantities
    where they were not fitted. (§5.14)
18. **A metric is only current if its whole chain is current** — run
    `python3 -m pipeline.status` before quoting or pasting from any report
    and before ending a session that touched artifacts, config or reports;
    never use a report the `artifact bundle` / `artifact mirrors` /
    `report vintages` / `walkthrough` lines call stale. Re-run map: §5.14.
19. **The repo's data is SYNTHETIC; the owner's is real** — every number a
    local run prints is a fixture number and is evidence about the fixture
    only. Never state one as a finding about the owner's extract, and never
    advise on their data from one: **ask them for the number first.** What
    does carry over is structural — code paths, leak/ordering arguments,
    arithmetic (e.g. "a W-week hold-out self-calibrates (W−1)/W of the
    gate"). What does not is anything measured: gate values, ratios, `r`,
    `rho`, prior scores, week counts, verdicts. Label which one you are
    giving. (§9.2)

And the standing prohibitions:

- Never commit `data/`, `reports/`, `artifacts/`, `events_store*/`, or any
  secret. Redshift credentials come only from `~/.env` as `REDSHIFT_*` —
  no hostname or credential in config, code, or a commit.
- Never hand-edit `artifacts/posterior.json` (production learning state).
- Never re-derive logic that has one home: the population filter
  (`bootstrap.prepare_data.population`), the window cut
  (`common.episodes.window_slice`), outcome reconciliation
  (`common.episodes.adjustment_reason`), scrap
  (`common.episodes.leftover_units` / `classify_last`), guardrail deviation
  (`common.guardrail.deviation`), spread accounting
  (`pricing.explore.SpreadLedger`).
- Never invent a SET BY OWNER value; never drive a quarantine count to zero
  with a catch-all reason.
- Quote the sampling caveat with any sampled-run count — a zero over a
  sample is not a proof over the window.
- `python3 -m pytest tests/` must pass before any push (~3 min).

## Setup

```bash
pip install -r requirements.txt
python3 -m pytest tests/
```

All commands run from the repo root — paths in `config.yaml` are relative to
it, and running a module from elsewhere silently reads/writes the wrong
artifacts.

## Pipeline order

```
step                                          writes
0. bootstrap.download_flc                     data/flc_raw.parquet   (Redshift; REDSHIFT_* from ~/.env)
1. bootstrap.prepare_data --input <raw>       data/prepared.parquet, artifacts/split_manifest.json
3. bootstrap.train_baseline --input prepared  artifacts/baseline_model.txt, feature_schema.json
3b. bootstrap.train_baseline --fit-calibration artifacts/calibration.json    (BEFORE prior — see below)
4. bootstrap.estimate_prior --input prepared  artifacts/prior.json           (BEFORE dispersion — §5.6)
5. bootstrap.fit_dispersion --input prepared  artifacts/r_lookup.json, rho.json
5b. bootstrap.train_baseline --check-convergence  (dry run; asserts the f<->r loop settled)
6. backtest --input prepared                  reports/backtest.json
6b. bootstrap.derive_thresholds               reports/thresholds.json  (pipeline.tune reads it)
8. bootstrap.init_posterior                   artifacts/posterior.json       (once; --force to overwrite)
9. pipeline.shadow --input prepared           reports/shadow.json            (holdout by default)
11. bootstrap.seal                            artifacts/bundle.json
```

**The order above is a LOOP, not a line, and 3b is where it turns.** The
factor solve consumes `r`, while `r`, `rho` and the prior are all fitted
against *calibrated* `mu_ref` — so calibration comes **before** prior and
dispersion, and they are re-fitted against it. On a first run there is no
`r_lookup` yet and 3b silently uses the raw-mu basis; the second pass gets
the censored one. Step 5b asserts the fixed point settled: **NOT CONVERGED
means run 3b → 4 → 5 → 5b again**, which is normal after a retrain or a
split change. (§9.2)

`scripts/run_bootstrap.sh <raw>` runs 1–6b and 11 and prints `status`, in
exactly this order — it is the executable copy of the table above; if the two
disagree, the script is right. **It retrains the baseline every time**
(rule 1) — iterate on single modules, not the script. After a later
standalone `--fit-calibration`, **re-run `bootstrap.seal`**.

## Tuning loop — before production

`python3 -m pipeline.tune` reads the four reports and says what config should
be, with the report field behind every recommendation. Run it after any
bootstrap; it changes nothing without `--apply`.

```bash
scripts/run_bootstrap.sh <raw>          # 1–6, charts, seal, status
python3 -m pipeline.shadow --input data/prepared.parquet   # writes shadow.json
python3 -m pipeline.tune                # what to change, and on what evidence
python3 -m pipeline.tune --apply        # pastes MEASURED values only
scripts/run_bootstrap.sh <raw>          # re-measure under the new config
```

Four classes, and the class decides who may act. **PASTE** — the pipeline
measured it, so it does not wait on a human; `--apply` writes it. That
includes values that used to be SET BY OWNER but are decided by data: the
guardrail stop thresholds (3σ of the control arm's own noise), the fit window
W (the rolling-origin sweep, when `calib >= 2W` allows it), and
`max_mean_step` — the last **gated**, pasted only when `step_sensitivity`
says the re-price is small (`tuning.*_for_auto_rail`), and returned to the
owner when it is not. **OWNER** — what data cannot decide: a preference, not
a fact. Four qualify, and they are the whole list: `budget_share_of_il` (how
much margin learning is worth), `min_detectable_effect_pct` (what effect size
is worth detecting — the reports say what is DETECTABLE, and pasting that
would make the power check pass by construction), `max_std_shrink` (how fast
the system may become confident — `tune` supplies both ways to settle the
rail mismatch and takes neither), and `data.split` (how much history still
represents the business). Reported with the evidence, never auto-applied
(see the prohibitions). The test: **SET BY OWNER is for a number that encodes
what you are willing to lose, wait for, or risk — everything else is
measurable.** **READ** — a finding with no config key (which
constraint binds learning; whether the weekly calibration cron is worth
running). **BLOCK** —
an invariant that must hold first: reports from one model, a settled `f<->r`
loop, `calib >= 2W`, no graded window across the exclusion gap. A BLOCK
suppresses everything else and `--apply` refuses, because tuning against a
report that graded a different model is worse than not tuning (rule 1).

`--apply` copies `config.yaml` to `artifacts/config_backup_<stamp>.yaml` and
appends to `artifacts/config_decisions.json`: what was written, the report
field it came from, and every owner decision still outstanding. That file is
the record of WHY the config is what it is — read it before changing a value
by hand. **Iterate until `tune` reports no PASTE and no BLOCK**; a changed
increment or rail changes what the next run measures, so one pass is rarely
enough. Full rationale for each rule: §5.14, §6, §9.2.

Daily production loop (Lane C — full operator guidance in `RUNBOOK.md`):

```bash
python3 -m pipeline.update             # monitor only, always safe
python3 -m pipeline.update --apply     # OPERATOR GATE (rule 10)
python3 -m pipeline.monitor
python3 -m pipeline.assurance
python3 -m pipeline.status             # done = every line green
```

## Populations

One prepared parquet; `dp_eligible` + `dp_ineligible_reason` (`cost_missing`
| `non_priceable` | `negative_window` | `window_too_long` | `outcome_unknown`
| `final_hour_restock`) flag what the solver cannot price — nothing economic
is dropped (rule 14). Resolve via `prepare_data.population(d, cfg, which)`:

| Consumer | Population |
| --- | --- |
| baseline, `r`, `rho`, prior, level factors | `baseline_model.train_population` (default `eligible`) |
| `m1` / gate 1 | `integrity`, always |
| DP, backtest, shadow, calibration gate, A/B | `dp_eligible`, always (passed explicitly) |

The waterfall (13 rows, `artifacts/split_manifest.json`) records rows,
episodes and COGS after every stage; `kind: hard_drop` drops, the two
`population_gate` rows (`eligible`, `dp_eligible`) only flag.
`python3 -m tools.export_waterfall --input <raw>` writes the full workbook.
The chain, the inventory convention, the flow identity and the close rules
are specified in §5.2 and §12a.

## MEASURED pastes and launch blockers

`load_config(strict=True)` refuses while any RUNTIME_REQUIRED value is null.
Every paste has one source and one checker:

| Config value | Paste from | Checked by |
| --- | --- | --- |
| `dispersion.rho`, `mean_forced_hours_per_episode` | `artifacts/rho.json`, after EVERY retrain | `artifact mirrors` (strict start-up refuses drift) |
| `exploration.tau_initial` | `reports/shadow.json` → `tau_initial_derivation` (backtest = cross-check only) | `tau_provenance_error` — shadow refuses a stale paste |
| `ab_test.il_pct_ratio_se_clustered` | `reports/phase0.json` → `config_values_measured` | `artifact mirrors` |
| `scrap/margin_deterioration_pct`, `min_detectable_effect_pct` | OWNER, from `reports/thresholds.json` — `TOO TIGHT` and `LIKELY INERT` are blocking | `guardrail floors` |

## Where to look

| Working on | Read first |
| --- | --- |
| filter chain, waterfall, eligibility, restocks, scrap | §5.2, §12a; `bootstrap/prepare_data.py`, `common/episodes.py`; rules 13–15 |
| baseline, calibration, fidelity | §5.4, §9.2 (incl. the gate decision tree); rules 1, 4–6 |
| elasticity prior, dispersion | §5.5, §5.6; rules 1a, 2, 3, 7 |
| DP, pricing, exploration, tau, budget | §5.7, §5.8, §5.10 |
| backtest, replay, tau derivation | §5.14, §9; rule 17 |
| shadow phase | §5.13 (holdout default, sampling caveats, tau₀ derivation) |
| posterior, update, operator gate | §5.9, §5.11 |
| monitoring, guardrails, stop conditions, A/B | §5.12, §11, §12 |
| events, integration, quarantine | `docs/event_contract.html`; `events/store.py` |
| provenance, seal, freshness | §5.14; rule 18 |
| docs, charts, walkthrough, EDA, metrics index | `docs/maintaining_docs.md` |
| operator runbook, review tiers | `RUNBOOK.md`, `REVIEW_GUIDE.md` |
| why is it not done the other way? | `docs/learnings.md` |

## Repo conventions

- Run modules as `python3 -m package.module` from the repo root.
- `--workers N` (`0` = all cores but one) parallelises backtest and shadow;
  reports are byte-identical serial or parallel — results return in
  submission order and only the parent touches the event store.
- Each episode draws from its own RNG seeded by episode id — draws are
  order-independent by design.
- Synthetic fixtures: `tools/make_dummy_flc.py` (`--policy randomized` =
  recoverable elasticity, `--policy legacy` = the production confound). It
  must keep emitting both source inventory conventions — regenerate the
  fixture after any change to them, and read the two printed counts.
