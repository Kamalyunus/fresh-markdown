# Perishable Markdown MVP

Implementation of the Perishable Markdown MVP — specified in
[`docs/design.md`](docs/design.md): the smallest markdown-pricing system
that can run in production and improve itself from its own decisions.
Legacy history cannot point-identify price elasticity (price is collinear
with hour-of-day under the legacy ramp), so history supplies baseline
demand, dispersion, correlation structure and a bounded prior — elasticity
itself is learned in production from IL-budgeted randomized exploration.
Engineering's page is [`docs/engineering_handover.html`](docs/engineering_handover.html):
the process phase by phase, then the integration contract as its appendices.

## Layout

One package per responsibility; each maps to one REVIEW_GUIDE tier and one
`ops.advance` phase.

| Package | Design | What lives there |
| --- | --- | --- |
| `config.yaml` | 5.1 | Every tunable. Single source of truth; no numeric literals in code. |
| `pilot_sim.yaml` | 11.3 | The pilot simulator's world, run, faults and paths — nothing the system reads. |
| `engine/` | 5.7–5.10 | What prices a shelf and learns: `demand.py` (mu(d), censored expectation), `dp.py` (monotone DP, absolute-IL reward), `explore.py` (uniform draw from the admissible, tau-affordable set; `delta_min`), `budget.py` (the IL budget and the tau controller, `walk_tau`), `spread_ledger.py` (the Q-spread ledger the harnesses price tau against, `SpreadLedger.sweep`), `learn.py` (the censored NB grid update and its Fisher information), `posterior.py` (launch belief, bounded step, atomic exactly-once commit, exploration suspension), `decide.py` (state validation — reject, never an unsafe price — and the decision event), `state.py` (the 12-field request → the engine's state: point-in-time features, `mu_ref_path`, `r`). |
| `events/` | 5.10 | `store.py` (append-only JSONL: decisions, outcomes and rejections — the shelf-hours seen and NOT priced, which keep a refused hour from reading later as one never sent — plus dedup, quarantine with reasons, torn-line safe, and a harness-only `reset` the production store refuses), `contract.py` (the required fields and value checks the store enforces, and the one rejection-event builder), `pairs.py` (the one decision↔outcome pairing, the three event ids over one hour key, trading-day key, priced/suspended days), `frame.py` (live events as the episode frame every IL, scrap and margin figure reads). |
| `common/` | 5.1, 5.2, 2.3 | Shared definitions: `config.py` (loader, strict mode), `windows.py` (where an episode starts and ends -- the id rule, the defective-window drops -- and every episode-scoped cut), `episodes.py` (endings, leftover, censoring, flow identity, COGS at risk, window extension), `clustering.py` (rho, deff), `metrics.py` (`episode_economics`; `summary`, the IL/scrap/sell-through block every reader rounds for itself), `guardrail.py` (the deterioration series and the persistence streak), `provenance.py` (stamps, seal, config fingerprint), `history.py` (the audit trail), `io.py` (the JSON, JSONL and table readers, and `write_frame`, the one by-extension writer of the files engineering reads back), `paths.py` (the files the drivers name outside config), `cli.py` (the shared `--config`/`--reports`/`--out` parser), `parallel.py`. |
| `fit/` | 5.2–5.6 | The frozen artifacts: `download_flc.py` (Redshift extract, `REDSHIFT_*` from `~/.env`), `prepare_data.py` (filter chain, eligibility flags, episodes, waterfall, split manifest), `train_baseline.py` (LightGBM/Tweedie `mu_ref`, the level-factor applier; its `--fit-calibration` / `--check-convergence` flags call `calibrate.py`: the one level estimator, the weekly schedule, the convergence check), `estimate_prior.py` + `prior_density.py` (the elasticity prior as a profile-likelihood density), `fit_dispersion.py` (NB `r`, `rho`), `artifacts.py` (`load_bundle`: the one way the frozen bundle is loaded for pricing or grading). |
| `evaluate/` | 5.13, 5.14, 11.3, 12 | Grades the artifacts before launch: `backtest.py` (like-for-like replay, fidelity, tau derivation, step sensitivity, within-episode moves), `shadow.py` (the full decision path on the hold-out, no prices applied), `level.py` (what both harnesses read about the level: the priced frame, the frozen-vs-refit rescale, the weekly re-fit) and `tau.py` (the ledger fold, the episode sample, the tau block), `pilot_shop.py` (the simulated shop and its two pricers) + `pilot_grade.py` (its readings and expectations) + `pilot_sim.py` (the run) + `pilot_world.py` (the demand world) — the weeks after launch against a simulated shop: real engine, real daily lane, injected faults, graded expectations — `derive_thresholds.py` (guardrail floors, learning-rail checks). |
| `daily/` | 5.11, 5.12, 5.15 | The production lane, in run order: `ingest_outcomes.py` (outcome events from the hourly feed), `features.py` (the day's feature table Lane B's batches join, from the rolling feed), `update.py` (collects the batch, gates it, walks tau — the grid update itself is `engine/learn.py`; `--calibrate-tau` daily; `--apply` and `--resume-exploration` are the human gates), `monitor.py` (business, learning, safety; stop conditions), `assurance.py` (frozen artifacts vs the live world), `export_events.py` (warehouse tables — derived, never the record; the shelf-hour also in the feed's own spelling and types, so a night's decisions append to the hourly feed with no rename and no cast). |
| `ops/` | 9, App. A | Drivers and gates: `advance.py` (the order of operations as code, one function per phase; `--plan`, `--feed`, `--report`), `readiness.py` (the launch-readiness report every stop writes), `run.py` (the step runner and the phase order), `bootstrap_loop.py` (train ONCE, iterate the calibration ↔ dispersion loop to convergence, backtest, thresholds, seal), `tune.py` (the config loop as code: one function per finding), `config_keys.py` (what each key is to the chain — who pastes it, who reads it, what a move re-runs; report staleness; the tau paste gate), `status.py` (the checks that gate a decision; exit 1 on FAIL), `init_posterior.py`, `integration.py` (the pricing folder's check — its verbatim copies against their sources, the config against a fresh prune, the manifest against the disk, every module of its package reached — and the sync `seal` and `advance` call: five artifacts, the extract seed, the pruned config), `seal.py` (every seal also writes an audit snapshot to `artifacts/history/<bundle>/<sealed_at>/`; every `advance` stop adds the reports), and Lane B's four: `price_hour.py` (the hourly script — the feed's own rows in, carrying the `episode_id` the producers assign, a price per shelf out), `price_batch.py` (the caller beneath it: one hour's requests in, a price per request out), `assign_episode_ids.py` (the id rule as a script the producers run or port), `check_inputs.py` (their three tables checked before launch). |
| `tools/` | 6, 5.7, 5.10 | `make_dummy_flc.py` (synthetic FLC generator, legacy + randomized policies), `scenario_deck.py` (the leadership deck: twelve scenarios answered by `dp.solve` → `reports/scenarios.html`), `e2e_cycle.py` (one whole integration cycle — requests, decisions, feed, outcomes, exports — in a workspace under `sim/e2e`: a thin driver of `evaluate.pilot_shop`'s shop through `LaneBPricer`). |
| `integration/` | 9, App. A | The standalone pricing folder for engineering's host: the hourly command, the morning table it joins on, and what the two need, maintained IN PLACE (the owner's call: minimal, and no refactor of the repository for it). `price_hour.py`, `download_flc.py` and `build_features.py` at its root chdir into the folder and run `pricing/`, the folder's OWN code — fourteen small modules written from scratch for the exploit-only hour, the morning pull and the morning table (the request, the state, the DP, the demand model applied at the reference discount, the append-only log, the feed schema, one SELECT over the trailing days of the hourly table with the producers' `episode_id` carried through, `REDSHIFT_*` from `~/.env`), not copies of the repository's modules; the map is `pricing/__init__.py`. The morning table is built from that day's extract alone, the producers' ids as given and row-level hygiene only — no window rule, no rolling history, no seed from the repository's extract. The hour is STATELESS: a row is priced from itself, the artifacts and the feature table of its episode's opening day — `episode_id` is the opening tag, so an entry is the row whose tag is its own shelf-hour, and a later hour is anchored on the price in force the row carries (the price applied last hour, piped back by the producers); no store is read, the log is append-only, and a re-sent hour gets the identical answer. It decides as the repository's stateful hour does: `test_end_to_end` runs an entry hour through `ops.price_hour` and the folder's command and refuses a difference in any field of the decision event or the rejection record (twice: exploration on, where the drawn fields are the one allowed difference, and suspended, where only the clock and the config digest may differ), then the next hour — the applied price piped back as the row's price in force — through both again, the repository continuing from its store and the folder from the row alone, every field equal; the faults a producer can send (a later row without the price in force, an id that is not a tag, an id opening after the hour) are refused with their reasons. So an engine fix is PORTED to `pricing/` by hand and the parity test says whether the port is complete. `ops.seal` and `ops.advance` call `integration.sync`, which copies the five artifacts the hour opens and the config pruned to the keys the commands read (`KEEP`, held equal to the code by the bundle test). `ops.integration --check` refuses a verbatim copy (the handover page, the examples, the requirements) behind its source, a config behind a fresh prune, a file the manifest does not list, an import that leaves the package or a module neither command reaches. Exploit-only by construction; the id is the producers'; the checks stay here on the files engineering sends. `tests/test_integration_bundle.py` pins all of it. |
| `tests/` | — | One file per behaviour (a module's tests often live under the behaviour's name, not the module's), plus `test_end_to_end.py` and `test_docs_match_the_code.py`; shared builders in `conftest.py`. |

## Running the bootstrap

```bash
pip install -r requirements.lock    # the verified versions; requirements.txt holds the floors
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
python3 -m ops.advance --feed <yesterday's parquet> [--failures <pushes>]   # the whole lane, in order:
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

```bash
python3 -m tools.e2e_cycle --episodes 20 --hours 3     # one integration cycle under sim/e2e
python3 -m ops.assign_episode_ids --hour rows.parquet --previous last_hour.parquet --out snapshots/<hour>.parquet   # the producers' id step (theirs to run or port)
python3 -m ops.check_inputs --snapshot s.parquet --feed f.parquet --failures x.csv   # their three tables, checked
python3 -m ops.price_hour --snapshot snapshots/<hour>.parquet --features features/<today>.parquet --out <hour>.csv
python3 -m ops.price_batch --requests hour.jsonl --features features/<today>.parquet --out decisions.jsonl   # the 12-field caller beneath it
```

The cycle is what engineering's Lane B does in production, run once
against the simulated shop — the pilot simulator's own, driven hour by
hour with a pricer that goes through `ops.price_hour` (the hourly script:
the shelf snapshot in the feed's schema in, with the episode ids the
producers assign — the shop assigns them here — a price per shelf out) instead of
the engine — then the feed the shop wrote, `daily.ingest_outcomes` naming
the outcomes from the feed row, the exported pair tables.

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
