# Perishable Markdown MVP

Implementation of the Perishable Markdown MVP — specified in
[`docs/design.md`](docs/design.md): the smallest markdown-pricing system
that can run in production and improve itself from its own decisions.
Legacy history cannot point-identify price elasticity (price is collinear
with hour-of-day under the legacy ramp), so history supplies baseline
demand, dispersion, correlation structure and a bounded prior — elasticity
itself is learned in production from IL-budgeted randomized exploration.

## Layout

One package per responsibility; each maps to one REVIEW_GUIDE tier and one
`ops.advance` phase.

| Package | Design | What lives there |
| --- | --- | --- |
| `config.yaml` | 5.1 | Every tunable. Single source of truth; no numeric literals in code. |
| `pilot_sim.yaml` | 11.3 | The pilot simulator's world, run, faults and paths — nothing the system reads. |
| `engine/` | 5.7–5.10 | What prices a shelf and learns: `demand.py` (mu(d), censored expectation), `dp.py` (monotone DP, absolute-IL reward), `explore.py` (uniform draw from the admissible, tau-affordable set; `delta_min`; budget, `walk_tau`, `SpreadLedger.sweep`), `posterior.py` (launch belief, bounded step, atomic exactly-once commit, exploration suspension), `decide.py` (state validation — reject, never an unsafe price — and the decision event). |
| `events/` | 5.10 | `store.py` (append-only JSONL: dedup, quarantine with reasons, torn-line safe), `pairs.py` (the one decision↔outcome pairing and trading-day key). |
| `common/` | 5.1, 5.2, 2.3 | Shared definitions: `config.py` (loader, strict mode), `episodes.py` (endings, leftover, censoring, flow identity, window extension), `metrics.py` (`episode_economics`, `fidelity_decomposition`), `guardrail.py`, `provenance.py` (stamps, seal, config fingerprint), `io.py`, `parallel.py`. |
| `fit/` | 5.2–5.6 | The frozen artifacts: `download_flc.py` (Redshift extract, `REDSHIFT_*` from `~/.env`), `prepare_data.py` (filter chain, eligibility flags, episodes, waterfall, split manifest), `train_baseline.py` (LightGBM/Tweedie `mu_ref`, level calibration, convergence check), `estimate_prior.py` + `prior_density.py` (the elasticity prior as a profile-likelihood density), `fit_dispersion.py` (NB `r`, `rho`). |
| `evaluate/` | 5.13, 5.14, 11.3, 12 | Grades the artifacts before launch: `backtest.py` (like-for-like replay, fidelity, tau derivation, step sensitivity, within-episode moves), `shadow.py` (the full decision path on the hold-out, no prices applied), `pilot_sim.py` + `pilot_world.py` (the weeks after launch against a simulated shop: real engine, real daily lane, injected faults, graded expectations), `derive_thresholds.py` (guardrail floors, learning-rail checks). |
| `daily/` | 5.11, 5.12, 5.15 | The production lane, in run order: `ingest_outcomes.py` (outcome events from the hourly feed), `update.py` (censored NB grid update; `--calibrate-tau` daily; `--apply` and `--resume-exploration` are the human gates), `monitor.py` (business, learning, safety; stop conditions), `assurance.py` (frozen artifacts vs the live world), `export_events.py` (warehouse tables — derived, never the record). |
| `ops/` | 9, App. A | Drivers and gates: `advance.py` (the order of operations as code; `--plan`, `--feed`, `--report`), `bootstrap_loop.py` (train ONCE, iterate the calibration ↔ dispersion loop to convergence, backtest, thresholds, seal), `tune.py` (the config loop as code), `status.py` (the checks that gate a decision; exit 1 on FAIL), `init_posterior.py`, `seal.py` (every seal also writes an audit snapshot to `artifacts/history/<bundle>/<sealed_at>/`; every `advance` stop adds the reports). |
| `tools/` | 6, 5.7 | `make_dummy_flc.py` (synthetic FLC generator, legacy + randomized policies), `scenario_deck.py` (the leadership deck: twelve scenarios answered by `dp.solve` → `reports/scenarios.html`). |
| `tests/` | — | One file per module plus `test_end_to_end.py` and `test_docs_match_the_code.py`; shared builders in `conftest.py`. |

## Running the bootstrap

```bash
pip install -r requirements.txt
python3 -m fit.download_flc --start-date <train_start> --end-date <holdout end>   # step 0 (ops.advance sizes it from config)
python3 -m ops.bootstrap_loop --input data/flc_raw.parquet
python3 -m ops.init_posterior
python3 -m evaluate.shadow --input data/prepared.parquet --out reports/shadow.json
```

Step 0 needs `REDSHIFT_*` in `~/.env` (gitignored, outside the repo); with
an extract in hand, skip it and pass the file. `ops.bootstrap_loop` trains the
baseline once, then **iterates the calibration ↔ dispersion loop to
convergence** — never hand-run the steps or re-run the script to settle it
(AGENTS rule 1b). Then:

1. **Prior acceptance gate (human, design 9.3)** — read
   `artifacts/prior.json`: `wrong_sign_categories`, per-category
   `mean/std/std_basis`, and `holdout_comparison` against `oracle` and
   `uniform` (read `information_available_per_row` first). A pooled or
   uniform prior is a designed outcome.
2. **Tune from the reports:** `python3 -m ops.tune` says what config
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
python3 -m ops.advance --feed <yesterday's parquet>   # the whole lane, in order:
#   ingest_outcomes -> update --calibrate-tau (tau walks daily, no operator)
#   -> monitor -> assurance -> export_events -> status, stopping at
python3 -m daily.update --apply     # bounded posterior updates (the human gate)
python3 -m daily.update --resume-exploration   # after a stop condition, a human's call
```

## Validating against synthetic data

```bash
python3 -m tools.make_dummy_flc --skus 300 --policy randomized \
    --out data/flc_synth.parquet     # module form: script form cannot read
                                     # config and falls back to a 90-day span
python3 -m ops.bootstrap_loop --input data/flc_synth.parquet
python3 -m pytest tests/
```

`--policy legacy` reproduces the production confound (estimators must
*detect* it); `--policy randomized` makes elasticity identifiable
(estimators must *recover* it). **Read the two convention counts the
generator prints** — `write-off rows` and `shrink rows` must both be
non-zero, or the fixture is not exercising the code that reads them.
Everything a repo-local run prints is a FIXTURE number (AGENTS rule 19).

```bash
python3 -m evaluate.pilot_sim                           # the weeks AFTER launch, per pilot_sim.yaml
python3 -m evaluate.pilot_sim --days 10 --fault mismatch:0.05   # one run's overrides
python3 -m evaluate.pilot_sim --workers 1                       # serial; the same answer
```

The simulator (design §11.3) prices every hour through the real engine
and runs the real daily lane in a workspace under `sim/`, against a demand
world built on the frozen model with an assumed elasticity;
`reports/pilot_sim.json` grades what a healthy launch shows and whether
the gates fire under an injected fault. Its settings — the world, the
run, the faults, the paths — live in `pilot_sim.yaml` beside
`config.yaml`, which it rehearses unchanged.

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
