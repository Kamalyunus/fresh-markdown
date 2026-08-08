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
| `bootstrap/prepare_data.py` | §9.1–9.2 | Schema mapping, filter chain, contiguous-hour episode construction, waterfall, split manifest. |
| `bootstrap/measure.py` | §8, App. A | Phase-0 measurement suite (m1–m8, m10) and reassessment gates. |
| `bootstrap/train_baseline.py` | §9.3 | Frozen LightGBM/Tweedie `mu_ref`; price features overwritten to `d_ref` at inference; level-calibration factor fit. |
| `bootstrap/fit_dispersion.py` | §9.4 | Frozen NB `r` by subcategory (censored MLE, fallback, clamp) and global `rho` vs fitted residuals. |
| `bootstrap/estimate_prior.py` | §9.5 | Bracket procedure (naive vs hour-controlled) over the full search bound, acceptance checks, fallback on rejection. |
| `pricing/demand.py` | §9.3, §11.3 | `mu(d) = mu_ref × ((1−d)/(1−d_ref))^ε`, truncated NB pmf. |
| `pricing/dp.py` | §11 | Monotone DP over feasible tiers; absolute-IL reward; entry arms. |
| `pricing/explore.py` | §12 | Affordable-set uniform selection under currency `tau`; budget and `tau` calibration. |
| `pricing/posterior.py` | §10, §13.4–13.5 | Cell records, bounded step, atomic commit with processed outcome IDs (exactly-once). |
| `inference/decide.py` | §11.4, §16.1 | State validation (reject, never an unsafe price), decision event emission. |
| `events/store.py` | §16 | JSONL event log: dedup, quarantine for malformed events, replay. |
| `pipeline/update.py` | §13–14 | Censored NB grid update, deff deflation, bounded step, operator gate (`--apply`). |
| `pipeline/monitor.py` | §15 | Business (IL% ratio-of-sums with denominators), learning, safety series; stop conditions. |
| `backtest/` | §17 | Fidelity gate, policy deltas, `Q(p*) − Q(p)` spread and `tau_initial` derivation. |
| `tools/make_dummy_flc.py` | — | Synthetic FLC generator (legacy + randomized policies, known ground-truth elasticity). |

## Running the bootstrap (PRD §1a order)

```bash
pip install -r requirements.txt
scripts/run_bootstrap.sh data/flc_filtered.parquet
```

This runs prepare → measure → train_baseline → fit_dispersion →
estimate_prior → backtest, then stops at the human gates:

1. **Calibration gate (blocking, §9.3)** — `reports/backtest.json` carries
   `fidelity_episode_sold_ratio` and the measurement-10 level/slope
   decomposition. Only the *level* component may be corrected multiplicatively
   (`baseline_model.apply_level_calibration`); a slope deficit means the prior
   understates elasticity and must be re-estimated, never papered over.
2. **Prior acceptance gate (blocking, §9.5)** — `artifacts/prior.json` records
   orientation/boundary/std checks. Rejection falls back per config and is an
   acceptable outcome.
3. Paste MEASURED values into `config.yaml` (`rho`,
   `mean_forced_hours_per_episode`, `tau_initial`,
   `il_pct_ratio_se_clustered`); the owner sets the SET BY OWNER keys.
   `common.config.load_config(strict=True)` refuses to start until then.

Then initialise the posterior and enter shadow (§19):

```python
from common.config import load_config
from pricing.posterior import PosteriorStore
import json

cfg = load_config(strict=True)
prior = json.load(open(cfg["posterior"]["prior"]["path"]))
PosteriorStore.initialise(cfg, prior["per_category"], prior["episodes_per_week"])
```

Daily production loop: `inference.decide` per decision interval, then

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
