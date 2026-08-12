# Perishable Markdown MVP

Implementation of the [Perishable Markdown MVP PRD](docs/perishable_markdown_mvp_prd.md):
the smallest markdown-pricing system that can run in production and improve
itself from its own decisions. Legacy history cannot point-identify price
elasticity (price is collinear with hour-of-day under the legacy ramp), so
history supplies baseline demand, dispersion, correlation structure, and a
bounded prior — elasticity itself is learned in production from IL-budgeted
randomized exploration.

## Layout

| Path | PRD | Responsibility |
| --- | --- | --- |
| `config.yaml` | §7 | Every tunable parameter. Single source of truth; no numeric literals in code. |
| `common/config.py` | §7 | Loader; strict mode refuses to start on null MEASURED values. |
| `bootstrap/prepare_data.py` | §9.1–9.2 | Schema mapping, 12-step filter chain, window-keyed episode construction (not date-keyed — 36-hour windows are common), waterfall, split manifest. |
| `common/episodes.py` | §9.2 | One definition of episode endings and true leftover: `ending_inventory` is written off to zero on an episode's last row, so scrap is `max(0, starting − sold)`. Also extends episodes to their full window so the DP horizon is not shortened by a realised sellout. |
| `bootstrap/measure.py` | §8, App. A | Phase-0 measurement suite (m1–m8, m10, m11 episode endings) and reassessment gates. |
| `bootstrap/train_baseline.py` | §9.3 | Frozen LightGBM/Tweedie `mu_ref`; price features overwritten to `d_ref` at inference; level-calibration factor fit. |
| `bootstrap/fit_dispersion.py` | §9.4 | Frozen NB `r` by subcategory (censored MLE, fallback, clamp) and global `rho` vs fitted residuals. |
| `bootstrap/estimate_prior.py` | §9.5 | Bracket procedure (naive vs hour-controlled) on entry rows over the full search bound, acceptance checks, fallback on rejection. |
| `bootstrap/init_posterior.py` | §10 | One-time posterior initialisation from the prior artifact; refuses overwrite without `--force`. |
| `bootstrap/derive_thresholds.py` | §8, §15.4, §18 | Evidence for the owner decisions: empirical A/B duration vs MDE, 3σ guardrail noise floors. |
| `pricing/demand.py` | §9.3, §11.3 | `mu(d) = mu_ref × ((1−d)/(1−d_ref))^ε`, truncated NB pmf. |
| `pricing/dp.py` | §11 | Monotone DP over feasible tiers; absolute-IL reward; entry arms. |
| `pricing/explore.py` | §12 | Affordable-set uniform selection under currency `tau`; budget and `tau` calibration. |
| `pricing/posterior.py` | §10, §13.4–13.5 | Cell records, bounded step, atomic commit with processed outcome IDs (exactly-once). |
| `inference/decide.py` | §11.4, §16.1 | State validation (reject, never an unsafe price), decision event emission. |
| `events/store.py` | §16 | JSONL event log: dedup, quarantine for malformed events, replay. |
| `pipeline/update.py` | §13–14 | Censored NB grid update, deff deflation, bounded step, operator gate (`--apply`). |
| `pipeline/monitor.py` | §15 | Business (IL% ratio-of-sums with denominators), learning, safety series; stop conditions. |
| `pipeline/shadow.py` | §19 | Phase-1 harness: full decision path against live data, no prices applied; exit-gate report. |
| `backtest/` | §17 | Fidelity gate, policy deltas, `Q(p*) − Q(p)` spread and `tau_initial` derivation. |
| `tools/make_dummy_flc.py` | — | Synthetic FLC generator (legacy + randomized policies, known ground-truth elasticity). |
| `tools/make_charts.py` | — | One diagnostic chart per component, generated from the report artifacts into `reports/charts/`. |
| `tools/deck_numbers.py` | — | Every figure the design doc and deck quote, in one pasteable block, tagged with the slide that carries it. |
| `tools/deck_text.py` | — | Lists and patches the deck's text runs. Refuses any replacement that does not match exactly once, so a stale number cannot survive a "successful" refresh. |
| `tools/deck_diff.py` | — | Regression check between two versions of the deck: what is new, what was dropped, and any reused slide whose text changed unintentionally. Exits non-zero on either. |
| `tools/deckkit.py` | — | Slide-package surgery — duplicate, reorder, and set text without a PowerPoint install. |
| `tools/build_v2.py` | — | Reproducible build of the v2 deck from the v1 slides plus seven new ones. |

## Running the bootstrap (PRD §1a order)

```bash
pip install -r requirements.txt
scripts/run_bootstrap.sh data/flc_filtered.parquet
```

This runs prepare → measure → train_baseline → fit_dispersion →
estimate_prior → backtest, then stops at the human gates. (The script
retrains the baseline every time — to iterate on one step, run that step's
module directly. Agents: read `AGENTS.md` before touching the pipeline.)

1. **Calibration gate (blocking, §9.3)** — `reports/backtest.json` carries
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
2. **Prior acceptance gate (blocking, §9.5)** — `artifacts/prior.json` records
   orientation/boundary/std checks. Rejection falls back per config and is an
   acceptable outcome.
3. Paste MEASURED values into `config.yaml` (`rho`,
   `mean_forced_hours_per_episode`, `tau_initial`,
   `il_pct_ratio_se_clustered`); the owner sets the SET BY OWNER keys.
   `common.config.load_config(strict=True)` refuses to start until then.

Then initialise the posterior and run the shadow phase (§19 — decisions
logged, no prices applied):

```bash
python3 -m bootstrap.init_posterior
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
```

The shadow report carries the §19 exit gate (event completeness, matched
rate, zero cost-floor violations) plus would-be exploration spend,
recommended-vs-legacy discount deltas, and the frozen-baseline drift ratio.
Architecture and rationale live in [`docs/design.md`](docs/design.md).

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
scripts/run_bootstrap.sh data/flc_synth.parquet
python3 -m pytest tests/
```

`--policy legacy` reproduces the production confound (discount ramps with the
clock) to confirm estimators *detect* it; `--policy randomized` makes
elasticity identifiable to confirm estimators *recover* it.

## Design invariants worth knowing

- The planner minimises **absolute IL**; the business reads **IL%** with an
  endogenous denominator. They can diverge by design (§3.3); both are always
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
