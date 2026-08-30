# Perishable Markdown MVP

Implementation of the Perishable Markdown MVP — specified in
[`docs/design.md`](docs/design.md), the authoritative design doc:
the smallest markdown-pricing system that can run in production and improve
itself from its own decisions. Legacy history cannot point-identify price
elasticity (price is collinear with hour-of-day under the legacy ramp), so
history supplies baseline demand, dispersion, correlation structure, and a
bounded prior — elasticity itself is learned in production from IL-budgeted
randomized exploration.

## Layout

| Path | Design | Responsibility |
| --- | --- | --- |
| `config.yaml` | 5.1 | Every tunable parameter. Single source of truth; no numeric literals in code. |
| `common/config.py` | 5.1 | Loader; strict mode refuses to start on null MEASURED values. |
| `bootstrap/download_flc.py` | 5.2 | Redshift extract of the raw hourly FLC feed into `data/flc_raw.parquet`, aliased to the column names step 1 renames. Credentials from `REDSHIFT_*` in `~/.env`; the exclusion window from `config.yaml`. |
| `bootstrap/prepare_data.py` | 5.2 | Schema mapping, the integrity/scope filter chain plus the `dp_eligible`, `below_cost_hours` and `edge_truncated` flags (only rows that cannot be believed or fall outside the study period are dropped; anything merely hard to price is flagged, so the frozen artifacts keep the population the DP cannot act on), window-keyed episode construction (not date-keyed — 36-hour windows are common), 13-row waterfall with COGS at risk (cost × supply), split manifest. |
| `common/episodes.py` | 5.2, 12a | One definition of episode endings and true leftover: `ending_inventory` is written off to zero on an episode's last row, so scrap is `max(0, starting − sold)`. Also extends episodes to their full window so the DP horizon is not shortened by a realised sellout. |
| `bootstrap/run.py` | 5.14 | The pipeline with its loop driven to convergence: prepare, train ONCE, then iterate calibration → prior → dispersion → convergence until settled, then backtest, thresholds, seal, status. `--check-only` settles the loop against artifacts already on disk with no retrain. |
| `common/metrics.py` | 2.3, 9.3 | The two shared measurements the pipeline consumes: `il_pct` (A/B power, read by `derive_thresholds`) and `fidelity_decomposition` (the level gate value, read by `backtest.replay`). |
| `pipeline/tune.py` | 5.14 | Reads the reports and says what config should be, with the report field behind every recommendation. `--apply` pastes MEASURED values, backs up the config and records why in `artifacts/config_decisions.json`. |
| `bootstrap/train_baseline.py` | 5.4, 9.2 | Frozen LightGBM/Tweedie `mu_ref`; price features overwritten to `d_ref` at inference; level-calibration factor fit. |
| `bootstrap/fit_dispersion.py` | 5.5 | Frozen NB `r` by subcategory (censored MLE, fallback, clamp) and global `rho` vs fitted residuals. |
| `bootstrap/estimate_prior.py` | 5.6 | The elasticity prior as a profile-likelihood density (censored Poisson, naive + controlled arms on entry rows, pooled shrinkage, no fallback constant); writes its own held-out and design comparisons into `prior.json`. |
| `bootstrap/prior_density.py` | 5.6 | The estimator itself — curves, densities, wrong-sign and zero-width guards, `design_comparison`, `holdout_comparison`. Superseded designs: `docs/learnings.md`. |
| `bootstrap/init_posterior.py` | 5.9 | One-time posterior initialisation from the prior artifact; refuses overwrite without `--force`. |
| `bootstrap/derive_thresholds.py` | 5.3, 5.12, 11, 12 | Evidence for the owner decisions: empirical A/B duration vs MDE, 3σ guardrail noise floors. |
| `pricing/demand.py` | 5.4, 5.7 | `mu(d) = mu_ref × ((1−d)/(1−d_ref))^ε`, truncated NB pmf. |
| `pricing/dp.py` | 5.7 | Monotone DP over feasible tiers; absolute-IL reward; entry arms. |
| `pricing/explore.py` | 5.8 | Affordable-set uniform selection under currency `tau`; budget and `tau` calibration. |
| `pricing/posterior.py` | 5.9, 5.11 | Cell records, bounded step, atomic commit with processed outcome IDs (exactly-once). |
| `inference/decide.py` | 5.10 | State validation (reject, never an unsafe price), decision event emission. |
| `events/store.py` | 5.10 | JSONL event log: dedup, quarantine for malformed events, replay. |
| `pipeline/update.py` | 5.11 | Censored NB grid update, deff deflation, bounded step, operator gate (`--apply`). |
| `pipeline/monitor.py` | 5.12 | Business (IL% ratio-of-sums with denominators), learning, safety series; stop conditions. |
| `pipeline/shadow.py` | 5.13 | Phase-1 harness: full decision path against live data, no prices applied; exit-gate report. |
| `backtest/` | 5.14 | Fidelity gate, policy deltas, `Q(p*) − Q(p)` spread and `tau_initial` derivation. |
| `common/parallel.py` | — | Runs the per-episode work across CPUs (`--workers N`, `0` = all but one). Results in submission order, never completion order; workers compute and the parent commits, so the event path the shadow gate measures is unchanged. Reports are identical either way. |
| `tools/make_dummy_flc.py` | — | Synthetic FLC generator (legacy + randomized policies, known ground-truth elasticity). The generated span defaults to whatever covers `data.split` (`--start`/`--days` override), so `bootstrap.run --input <fixture>` runs end to end — it previously stopped at `fit_dispersion` with an empty calibration window. |
| `docs/stockout_gap.html` | 5.4, 12a | Business-facing explainer for the one backtest question that always gets asked: why `*_units` comes in under `*_mu`. Answer — it is lost sales, not model shrinkage. HAND-AUTHORED against the fitted dispersion (r = 0.70) and the current extract's median shelf of 2 units; regenerate the figures if a retrain moves `r` materially. |

## Running the bootstrap

```bash
pip install -r requirements.txt
python3 -m bootstrap.download_flc --days 120      # step 0: data/flc_raw.parquet
python3 -m bootstrap.run --input data/flc_raw.parquet
```

Step 0 needs `REDSHIFT_HOST`, `REDSHIFT_PORT`, `REDSHIFT_DATABASE`,
`REDSHIFT_USERNAME` and `REDSHIFT_PASSWORD` in the environment — put them in
`~/.env` (gitignored, and outside the repo). If you already have the extract,
skip it and pass that file instead; the pipeline takes the path as an argument
and does not care what the file is called.

This runs prepare → **eda** → measure → train_baseline → estimate_prior →
fit_dispersion → backtest (the prior runs first — dispersion is fitted at the
per-category prior means), then stops at the human gates. (The script
retrains the baseline every time — to iterate on one step, run that step's
module directly. Agents: read `AGENTS.md` before touching the pipeline.)

1. **Calibration gate (blocking, design 9.2)** — `reports/backtest.json` carries
   `calibration_gate_value` and `calibration_gate`, read on the window named
   by `baseline_model.calibration_gate_window` and echoed as
   `fidelity.gate_window` — it must stay disjoint from
   `calibration_fit_window` or the gate grades its own fit. Per-window ratios
   in `fidelity.by_window` expose demand-regime drift, and measurement 10
   gives the level/slope decomposition. Only the *level* component may be corrected
   multiplicatively; a slope deficit means the prior misstates elasticity and
   must be re-estimated, never papered over. To apply the level remedy:

   ```bash
   python3 -m bootstrap.train_baseline --input data/prepared.parquet --fit-calibration
   # set baseline_model.apply_level_calibration: true in config.yaml, then
   # re-run backtest WITHOUT retraining the baseline:
   python3 -m backtest --input data/prepared.parquet --out reports/backtest_calibrated.json
   ```

   and record the fidelity ratio before and after — the comparison is valid
   only if `baseline_model_version` matches across the two reports.
2. **Prior acceptance gate (blocking, human, design 9.3)** — there is no reject flag
   in the artifact; the gate is a reading of `artifacts/prior.json`:
   `design_comparison` (which rows × hour-control combination this extract
   supports), `wrong_sign_categories` (own density discarded for the pooled
   one), `std_basis` per category, and the `holdout_comparison` against
   `oracle` and `uniform`. A pooled or uniform prior is a designed outcome.
3. Paste MEASURED values into `config.yaml` (`rho`,
   `mean_forced_hours_per_episode`, `tau_initial`,
   `il_pct_ratio_se_clustered`); the owner sets the SET BY OWNER keys.
   `common.config.load_config(strict=True)` refuses to start until then.
   `tau_initial` is checked further: it must match its derivation —
   `reports/shadow.json` → `tau_initial_derivation.tau_initial` (shadow
   derives its own launch tau on the trailing pre-window week; the backtest
   block is accepted only while no shadow derivation exists). A stale paste
   is refused by shadow's fallback path, and `pipeline.status` reports it as
   FAIL rather than passing it for being non-null.

Then initialise the posterior and run the shadow phase (design 5.13 — decisions
logged, no prices applied):

```bash
python3 -m bootstrap.init_posterior
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
```

The shadow report carries the shadow exit gate (event completeness, matched
rate, zero cost-floor violations) plus would-be exploration spend,
recommended-vs-legacy discount deltas, and the frozen-baseline drift ratio.
Architecture and rationale live in [`docs/design.md`](docs/design.md).

It runs on the hold-out **by default** — the window after `test_end` that no
artifact was fit on and no gate was decided on. For the launch record, sweep
every episode in it:

```bash
python3 -m pipeline.shadow --input data/prepared.parquet --max-episodes 0
```

`--all` runs the whole extract instead; the report then carries an in-sample
caveat naming which numbers that flatters.

That is the only unrehearsed test the extract can give. It re-derives `tau`
on the path production runs (`tau_recommended`) and walks the `tau` controller
day by day (`tau_controller_trace`) so you can see whether the pilot survives
its own first day — the backtest's derivation reports 1.00x by construction
and cannot. One shot: tune anything on this window and it stops being a
hold-out. See `docs/design.md` §5.13.

Daily production loop after the shadow gate passes: `inference.decide` per
decision interval, then

```bash
python3 -m pipeline.update             # monitor only
python3 -m pipeline.update --apply     # apply bounded posterior updates (operator gate)
python3 -m pipeline.monitor
```

## Validating against synthetic data

```bash
python3 tools/make_dummy_flc.py --skus 300 --days 160 --policy randomized \
    --out data/flc_synth.parquet
python3 -m bootstrap.run --input data/flc_synth.parquet
python3 -m pytest tests/
```

`--policy legacy` reproduces the production confound (discount ramps with the
clock) to confirm estimators *detect* it; `--policy randomized` makes
elasticity identifiable to confirm estimators *recover* it.

**Read the two convention counts the generator prints.** `write-off rows` and
`shrink rows` must both be non-zero, or the fixture is not exercising the code
that reads them — and a fixture missing one does not fail, it passes quietly.
A stale `data/flc_synth.parquet` generated before the write-off block existed
carried zero sentinel rows and went unnoticed for months, because the closure
classifier's fallback treated the absence as "everything closed". Regenerate
the fixture after any change to the inventory conventions.

## Design invariants worth knowing

- The planner minimises **absolute IL**; the business reads **IL%** with an
  endogenous denominator. They can diverge by design (design 2.3); both are always
  reported together and per-episode IL% is never computed.
- Exploration is a **currency budget**, not a probability: `tau` is compared
  against `Q(p*) − Q(p)` in won, and the forced price is drawn uniformly from
  the affordable set — that uniformity is the randomisation that makes
  outcomes clean evidence.
- Only exploration outcomes update the posterior; information is deflated by
  `deff = 1 + (forced_hours − 1) × rho`; every step is bounded and
  exactly-once.
- Cost floor and monotonicity bind on every path, including exploration, by
  construction of the feasible tier set.
