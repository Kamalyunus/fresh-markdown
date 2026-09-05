# Perishable Markdown MVP

Implementation of the Perishable Markdown MVP — specified in
[`docs/design.md`](docs/design.md): the smallest markdown-pricing system
that can run in production and improve itself from its own decisions.
Legacy history cannot point-identify price elasticity (price is collinear
with hour-of-day under the legacy ramp), so history supplies baseline
demand, dispersion, correlation structure and a bounded prior — elasticity
itself is learned in production from IL-budgeted randomized exploration.

## Layout

| Path | Design | Responsibility |
| --- | --- | --- |
| `config.yaml` | 5.1 | Every tunable. Single source of truth; no numeric literals in code. |
| `common/config.py` | 5.1 | Loader; strict mode refuses to start on null MEASURED values. |
| `bootstrap/download_flc.py` | 5.2 | Redshift extract of the raw hourly FLC feed (`REDSHIFT_*` from `~/.env`). |
| `bootstrap/prepare_data.py` | 5.2 | Schema mapping, integrity/scope filter chain + eligibility flags, window-keyed episode construction, waterfall with COGS at risk, split manifest. |
| `common/episodes.py` | 5.2, 12a | One definition of episode endings, leftover, censoring, the flow identity, and window extension. |
| `pipeline/advance.py` | — | The order of operations as code: probes the state on disk, runs every step that needs no human, stops at the next decision (owner keys, launch date, `update --apply`). `--plan` touches nothing; every stop writes `reports/launch_readiness.md`, the handover report. |
| `bootstrap/run.py` | 5.14 | The pipeline with its loop driven to convergence: prepare, train ONCE, iterate calibration → prior → dispersion → convergence until settled, then backtest, thresholds, seal, status. `--check-only` settles with no retrain. |
| `bootstrap/train_baseline.py` | 5.4, 9.2 | Frozen LightGBM/Tweedie `mu_ref` (price overwritten to `d_ref` at inference); level-calibration factors; convergence check. |
| `bootstrap/estimate_prior.py` + `prior_density.py` | 5.6 | The elasticity prior as a profile-likelihood density (censored Poisson, entry rows, pooled shrinkage, no fallback constant) with its held-out comparison. |
| `bootstrap/fit_dispersion.py` | 5.5 | Frozen NB `r` by subcategory and global `rho` vs fitted residuals, on the calib window. |
| `bootstrap/derive_thresholds.py` | 12 | Evidence for the owner decisions: empirical A/B duration vs MDE, guardrail noise floors, the learning-rail consistency checks. |
| `bootstrap/init_posterior.py` | 5.9 | One-time posterior init from the prior; refuses overwrite without `--force`. |
| `pricing/` | 5.7–5.9 | `demand.py` (mu(d), censored expectation), `dp.py` (monotone DP, absolute-IL reward), `explore.py` (uniform draw from the admissible, tau-affordable set; `delta_min` floor on the move; budget and the `walk_tau` controller), `posterior.py` (bounded step, atomic exactly-once commit). |
| `inference/decide.py` | 5.10 | State validation (reject, never an unsafe price), decision event emission. |
| `events/store.py` | 5.10 | Append-only JSONL: dedup, quarantine with reasons, replay. |
| `pipeline/` | 5.11–5.15 | `update.py` (censored NB grid update, `--apply` operator gate, `--calibrate-tau` daily tau walk), `monitor.py`, `shadow.py` (phase-1 harness), `assurance.py` (frozen artifacts vs the live world), `status.py` (exit 1 on FAIL), `tune.py` (the config loop as code), `ingest_outcomes.py` (outcome events built from the hourly feed — the minimal integration), `export_events.py` (decision/outcome tables for the warehouse — derived, never the record). |
| `common/metrics.py` | 2.3 | `episode_economics` — the one episode-grain IL/scrap/margin frame behind `il_pct`, the noise floors, the live guardrail, the business metrics and shadow's budget base; `fidelity_decomposition`. |
| `common/io.py` | — | `read_json` / `write_json`: the one NaN-safe way an artifact or report is read and written. |
| `events/pairs.py` | 5.10 | The one decision↔outcome pairing (`match_pairs`, `learnable=` excludes failed pushes) and the trading-day key (`decision_day`). |
| `common/parallel.py` | — | `--workers N` for backtest/shadow; reports byte-identical serial or parallel. |
| `tools/make_dummy_flc.py` | 6 | Synthetic FLC generator (legacy + randomized policies, known ground-truth elasticity); span defaults to covering `data.split`. |
| `tools/scenario_deck.py` | 5.7–5.8 | Leadership deck: twelve interactive scenarios (heavy/light stock, hours left, high COGS, exploration cost, legacy ramp, demand shock, restock, dead stock, learning, refusals) answered by real `dp.solve` runs over a state grid → `reports/scenarios.html`. Demand is a slider, not a forecast; the A/B is the evidence. |

## Running the bootstrap

```bash
pip install -r requirements.txt
python3 -m bootstrap.download_flc --days 120      # step 0: data/flc_raw.parquet
python3 -m bootstrap.run --input data/flc_raw.parquet
python3 -m bootstrap.init_posterior
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
```

Step 0 needs `REDSHIFT_*` in `~/.env` (gitignored, outside the repo); with
an extract in hand, skip it and pass the file. `bootstrap.run` trains the
baseline once, then **iterates the calibration ↔ dispersion loop to
convergence** — never hand-run the steps or re-run the script to settle it
(AGENTS rule 1b). Then:

1. **Prior acceptance gate (human, design 9.3)** — read
   `artifacts/prior.json`: `wrong_sign_categories`, per-category
   `mean/std/std_basis`, and `holdout_comparison` against `oracle` and
   `uniform` (read `information_available_per_row` first). A pooled or
   uniform prior is a designed outcome.
2. **Tune from the reports:** `python3 -m pipeline.tune` says what config
   should be, with the report field behind every recommendation;
   `--apply` pastes the MEASURED values and logs why. The owner sets the
   SET BY OWNER keys; `load_config(strict=True)` refuses to start until
   nothing required is null. `tau_initial` must match its derivation
   (`shadow.json → tau_initial_derivation`) — a stale paste is refused.

Shadow runs the full decision path with no prices applied, **on the
hold-out by default** — the window after `test_end` no artifact was fit on.
`--max-episodes 0` sweeps everything for the launch record; `--all` runs
the whole extract and stamps an in-sample caveat. One shot: tune anything
on the hold-out and it stops being one.

Daily production loop after the shadow gate passes:

```bash
python3 -m pipeline.update             # monitor only
python3 -m pipeline.update --apply     # bounded posterior updates (operator gate)
python3 -m pipeline.monitor
python3 -m pipeline.assurance
python3 -m pipeline.status
```

## Validating against synthetic data

```bash
python3 -m tools.make_dummy_flc --skus 300 --policy randomized \
    --out data/flc_synth.parquet     # module form: script form cannot read
                                     # config and falls back to a 90-day span
python3 -m bootstrap.run --input data/flc_synth.parquet
python3 -m pytest tests/
```

`--policy legacy` reproduces the production confound (estimators must
*detect* it); `--policy randomized` makes elasticity identifiable
(estimators must *recover* it). **Read the two convention counts the
generator prints** — `write-off rows` and `shrink rows` must both be
non-zero, or the fixture is not exercising the code that reads them.
Everything a repo-local run prints is a FIXTURE number (AGENTS rule 19).

## Design invariants worth knowing

- The planner minimises **absolute IL**; the business reads **IL%** with an
  endogenous denominator. They can diverge by design (2.3); both are always
  reported together and per-episode IL% is never computed.
- Exploration is a **currency budget**, not a probability: `tau` is
  compared against `Q(p*) − Q(p)` in won, and the forced price is drawn
  uniformly from the affordable set — the uniformity is what makes outcomes
  clean evidence. Tiers closer to the REFERENCE discount than `delta_min` (the category's
  level-bias scale over its |ε|, both derived) are neither drawn nor budgeted: the learner
  reads outcomes against `mu_ref` at the reference, and inside that
  distance the signal sits within the model's own error.
- Only exploration outcomes update the posterior; information is deflated
  by `deff = 1 + (forced_hours − 1) × rho`; every step is bounded and
  exactly-once.
- Cost floor and monotonicity bind on every path, including exploration, by
  construction of the feasible tier set.
