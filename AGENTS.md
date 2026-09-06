# Agent operating guide — Perishable Markdown MVP

This file is deliberately short: it holds only what an agent must obey
without looking anything up. The authoritative specification — including the
rationale and the measured incident behind every rule below — is
**`docs/design.md`** (§ numbers refer to it). Superseded approaches live in
`docs/learnings.md`. When this guide and the design doc disagree, the design
doc wins. Doc maintenance: the last section of this file.

## Non-negotiables

Each one-liner is binding on its own; the cited section carries the full
statement and the incident that created the rule.

1. **Never retrain the baseline between two runs you intend to compare** —
   a comparison is valid only when `artifact_versions.baseline_model_version`
   matches in both reports. `--fit-calibration` does NOT retrain; plain
   `train_baseline` and `ops.bootstrap_loop` DO. (§5.4)
1a. **Changing the elasticity prior invalidates `rho`, `deff` and the level
   factor** — re-run `fit_dispersion` onward and re-paste `rho`. (§5.6)
1b. **Drive the chain with `python3 -m ops.advance`, never by hand-running
   the step list** — it calls `ops.bootstrap_loop` (which iterates steps 3b–5b to
   the fixed point, 8–9 turns on production data), re-runs after a paste
   only what read it (a W paste turns the loop with `--check-only`; the
   rest re-derive a report or nothing), and retrains only when the model is
   absent or `--retrain` is given. Re-running the step list to settle calibration retrains the
   baseline (rule 1). (§9.2, Appendix A)
2. **`posterior.epsilon_max` (−0.05) is a sign constraint, never a bound to
   widen** — positive elasticity must remain unrepresentable. (§5.6)
3. **A boundary solution is not an estimate** — a fit pinned at a search
   bound means the likelihood ran off the support: the prior searches past
   both bounds and rejects a lower-pinned category to the pool
   (`lower_boundary_categories`; widen `epsilon_min`, never `epsilon_max`);
   `r_lookup.at_bound`, `calibration.pinned_cells` (every level-factor
   pin, anchor and weekly, incl. a pinned PARENT category) flag the others;
   `status`'s `boundary solutions` row READS them all. (§5.5, §5.6, §9.3)
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
   failure. The pilot simulator's world and run live in `pilot_sim.yaml`,
   never here: nothing the system reads. (§5.1, §11.3)
9. **IL% is always a ratio of sums, with its denominator and absolute IL
   alongside** — per-episode IL% is undefined and must never be averaged.
   (§2.3)
10. **`daily.update --apply` is the operator gate** — never work around a
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
    read once, by `evaluate.shadow`, and never tuned on. (§5.13, §9)
17. **A number a procedure solves for is not evidence about that number** —
    a bisection reports 1.00× on any population; grade fitted quantities
    where they were not fitted. (§5.14)
18. **A metric is only current if its whole chain is current** — run
    `python3 -m ops.status` before quoting or pasting from any report
    and before ending a session that touched artifacts, config or reports;
    never use a report the `artifact bundle` / `artifact mirrors` /
    `report vintages` lines call stale. The seal covers the CONFIG and the
    LIBRARY versions too: a move in either is a red bundle row until a
    deliberate re-seal; every decision event carries `config_digest`. (§5.14a)
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

- Never commit `data/`, `reports/`, `artifacts/` (including
  `artifacts/history/`, the per-seal audit trail — compare bundles there,
  never by re-fitting), `events_store*/`, or any secret. Redshift credentials come only from `~/.env` as `REDSHIFT_*` —
  no hostname or credential in config, code, or a commit.
- Never hand-edit `artifacts/posterior.json` (production learning state).
- Never re-derive logic that has one home. The homes, and what each
  replaced (a second copy of any of these is a review failure):
  - population filter — `fit.prepare_data.population`; the null-counter
    run drop — `null_counter_windows` (the whole clock run, rule 15)
  - episode-scoped cuts — `common.episodes.window_slice`,
    `trailing_weeks_window` (both factor-fit schedules), `week_key`,
    `opening_dates` (the episode's date key: folds, drift windows, the
    schedule), `calendar_days` (the one `n_days`); the hour arithmetic —
    `hour_discrepancy`; COGS at risk — `prepare_data.episode_cogs`
  - outcome reconciliation — `common.episodes.adjustment_reason`
  - scrap, IL, margin at episode grain — `common.metrics.episode_economics`
    (+ `settled`, `daily_rates`; `scrap_rate` is scrap over SUPPLY,
    opening + restocked) over `common.episodes.scrap_units`; live events
    enter it through `daily.monitor.event_frame`. The floors, the live
    guardrail, the business metrics and shadow's budget base all read
    this one frame; the floor and the trigger read one deterioration
    series — `common.guardrail.deterioration_series`
  - decision↔outcome pairing and the trading day —
    `events.pairs.match_pairs` (`learnable=` excludes failed pushes,
    `learnable_with_stock` also the hours with nothing to sell),
    `decision_day`, `price_matches`; the event-quality gates, windowed
    over `event_quality_window_days` — `quality_counts`, `quality_rates`
  - anchor rows — `common.episodes.is_anchor_row`
  - guardrail deviation and verdicts — `common.guardrail.deviation`,
    `verdict_is_blocking`, `verdict_is_insufficient`
  - the tiers a forced move may land on — `engine.explore.admissible`
    (`affordable_set`, `spread_costs` and the assurance uniformity check
    all read it; `delta_min` is derived there, never a second knob); their
    costs — `admissible_costs`, priced once per decision
  - the finiteness test — `engine.decide.finite_number`; the NB pmf table
    — `engine.demand.nb_pmf_table`; the live episode frame —
    `daily.monitor.settled_episodes` (built once per monitor run); the
    week the factor schedule reaches (a held week counts) —
    `fit.train_baseline.schedule_reaches` (the `--apply` gate and
    `advance`'s re-fit trigger)
  - spread accounting — `engine.explore.SpreadLedger`; the tau controller
    walk — `engine.explore.walk_tau` (production and shadow's trace) and
    whether a day's budget is signal — `budget_held` (the controller's
    hold, the overspend stop's no-reading, shadow's mean budget, §5.8);
    the week after a data week — `common.episodes.week_after` (the
    schedule's appended week, advance's re-fit trigger, the simulator);
    the backtest's forward simulation — `evaluate.backtest._simulate_arm`
    (+ `_dp_price`)
  - rho — `common.config.intraclass_correlation`; `m` per batch —
    `deff_from_episodes`
  - JSON in/out — `common.io.read_json` / `write_json` (NaN-safe)
  - what an hour was priced with beyond the artifacts —
    `common.provenance.environment` / `environment_drift`; the config digest — `config_fingerprint`
  - the discount-grid epsilon — `engine.dp.TIER_EPS`; own-data prior
    weight — `common.config.OWN_DATA_WEIGHT`
  - pastable config keys — `ops.tune.KEYS` (anchor, measured, rerun);
    the status "not run" prologue — `ops.status._needs`
- Never invent a SET BY OWNER value; never drive a quarantine count to zero
  with a catch-all reason.
- **A code change ships with its doc change in the same commit** — this
  file's one-home list and paste table, the RUNBOOK step it touches, the
  design.md section, the event contract for any event field, and a
  learnings.md entry when a design was superseded. Docs that lag the code
  are how the next agent re-derives what already has a home.
- Quote the sampling caveat with any sampled-run count — a zero over a
  sample is not a proof over the window.
- `python3 -m pytest tests/` must pass before any push (a couple of minutes; the
  end-to-end module runs the bootstrap chain in subprocesses).

## Setup

```bash
pip install -r requirements.txt
python3 -m pytest tests/
```

All commands run from the repo root — paths in `config.yaml` are relative to
it, and running a module from elsewhere silently reads/writes the wrong
artifacts.

## Driving the chain — `ops.advance`, not the step list

```bash
python3 -m ops.advance --plan    # phase table + next steps, touches nothing
python3 -m ops.advance           # runs to the next HUMAN decision and stops
```

**Two readers, one command.** The OWNER tells their agent: read this file,
then run `ops.advance` until `reports/launch_readiness.md` says it is
waiting on `data.launch_date`; the agent pulls the extract per the config's
split and hold-out dates (`REDSHIFT_*` in `~/.env`), derives every MEASURED
value, and stops at each owner decision with the evidence. ENGINEERING
builds Lane B against `docs/event_contract.html` (the caller, applying the
price, reporting failed pushes) and runs the daily lane
(`advance --feed`) on a cron; `RUNBOOK.md` is their document.

It owns the order and recomputes state from disk every run. Phase by
phase — what runs, which config keys move, and who moves them:

| phase | runs | config keys | moved by |
| --- | --- | --- | --- |
| data | `fit.download_flc` over `split.train_start` → the hold-out's end, only when no extract is on disk | none | — |
| bootstrap | `ops.bootstrap_loop` (train ONCE, loop to the fixed point, backtest, thresholds, seal). A moved TRAINING input (`data.split`, `exclusion_window`, the LightGBM keys) is a STOP: a retrain is a new bundle and only `--retrain` runs one | none | — |
| tune | `tune --apply`, then `ops.bootstrap_loop --check-only` / `evaluate.backtest` / `derive_thresholds` / shadow as the moved keys demand (`tune.stale_keys`; `READ_BY` routes unpasted keys to the one report or fit that reads them), until nothing is left to paste. A MEASURED value a report ran and still could not derive is a STOP naming that report, never an owner decision; then `ops.seal --reason config` (or `libraries`) whenever the environment moved since the last seal | `rho`, `calibration_fit_trailing_weeks`, `calibration_gate_band`, `information_increment`, `delta_min_log_bias`, `scrap/margin_deterioration_pct` (the 3σ trailing floor), `max_mean_step` (inside its price-consequence gate) | the process, from the report that derives each |
| posterior | `init_posterior`, once — re-run with `--force` by the process only BEFORE launch, while the file holds no production state (no consumed outcome, no walked τ, no suspension) and its cells differ from what init would write now (the launch belief or the prior moved) | none | — |
| shadow | `evaluate.shadow` on the hold-out, every episode; then `tune --apply` | `tau_initial` | the process, from `shadow.tau_initial_derivation`. The forced rate is the budget's: to change it the owner reads `shadow.exploration_budget_sweep` (forced rate, spend, move, `information_rel` per `budget_share_of_il` × `delta_min_bias_multiple`), sets the pair, and shadow re-runs once |
| owner | STOP | `max_std_shrink`; `max_mean_step` when its re-price EXCEEDS the gate; a stop threshold only when its floor is `BLOCKED`, `TOO TIGHT`, `LIKELY INERT` or `insufficient history`; `posterior.cold_start_shift_std` never stops (it ships 0.5) but is yours — `tune` reports it with the backtest evidence | you, from `thresholds.json` (advance prints floor, verdict, source) |
| launch | STOP, then `--fit-calibration` + `seal` | `data.launch_date` | you, on launch day |
| daily | ingest, `update --calibrate-tau`, monitor, assurance, export, status; STOP at `update --apply` | none | you approve each update. A fired stop condition SUSPENDS exploration (the monitor writes it into the posterior state; `decide` stops drawing; exploitation continues) until a human runs `update --resume-exploration` |

Every stop writes `reports/launch_readiness.md` (`--report` regenerates
it), a failed step included — that stop is journaled with the step's own
exit message and `advance` exits 1: what ran per phase, every value the process changed with before,
after, why and source, the config in force, status, and what is still
waited on — the handover document, assembled from the journal and tune's
decision log, never from memory. It never retrains unless the model is
absent or `--retrain` is given; it re-grades a report only when its bundle
moved or a config key that report READS moved (`tune.stale_keys`, the one
routing `status`'s `report vintages` line also uses: W turns the loop,
`delta_min` re-runs shadow, a stop threshold re-derives thresholds, the
launch belief re-runs the backtest alone, a training input STOPS for
`--retrain`, judged per key — pasted together, both re-run; a MEASURED
paste that writes back what a report measured invalidates nothing; a
bundle-stale shadow is re-run before tune can BLOCK on it); it refuses to run the same step a third time in one
invocation; and it stops on every SET BY OWNER null with the evidence. Its daily lane walks tau
(`daily.update --calibrate-tau`, no operator) and stops at
`daily.update --apply`, which it never runs: learning is gated daily
and per cell — a fast category updates the day its batch has the evidence.
Read its stop before doing anything by hand; the sections below explain the steps it runs.

## Running the bootstrap — use `ops.bootstrap_loop`, not the step list

**Do not hand-run the steps in order. Run this:**

```bash
python3 -m fit.download_flc            # step 0, only if you need a fresh extract
python3 -m ops.bootstrap_loop --input data/flc_raw.parquet
python3 -m ops.init_posterior          # step 8, once
python3 -m evaluate.shadow --input data/prepared.parquet --out reports/shadow.json
```

`ops.bootstrap_loop` is the whole bootstrap: it runs 1 and 3, then **iterates
3b–5b until the fixed point CONVERGES**, then 6, 6b, 11 and `status`; it
exits non-zero if the loop never settles. Its closing `status` is ADVISORY
(`tau_initial` null, shadow not run): red rows are the next steps, not a
broken bundle; `status` gates the pilot (RUNBOOK), never the bootstrap.

**Why this is not optional.** Steps 3b–5 are one TURN of a fixed-point
iteration (the factor solve consumes `r`; `r`, `rho` and the prior are fitted
against *calibrated* `mu_ref`): one pass leaves artifacts that disagree and
a NOT CONVERGED check (owner: 8–9 turns on production; fixture 3–4 — rule 19).
Re-running the list restarts at 3 and RETRAINS THE BASELINE (rule 1).

After a **config paste**, settle without retraining:

```bash
python3 -m ops.bootstrap_loop --check-only   # settle on the artifacts on disk, NO retrain
```

`--max-turns` (default 20) is a runaway guard, not a budget — the STALL test
stops a loop three turns without a new best. Reading its output: turn one
has no `r_lookup` (raw-mu basis; early turns move a lot), and the loop's
`estimate_prior --fast` drops `fold_spread`, which only widens the std
FLOOR and cannot move the fixed point; the settled artifact gets a FULL
prior. (§9.2)

### The steps it runs — for debugging ONE step, not for driving the pipeline

```
step                                          writes
0. fit.download_flc                     data/flc_raw.parquet   (Redshift; REDSHIFT_* from ~/.env)
1. fit.prepare_data --input <raw>       data/prepared.parquet, artifacts/split_manifest.json
3. fit.train_baseline --input prepared  artifacts/baseline_model.txt, feature_schema.json
3b. fit.train_baseline --fit-calibration artifacts/calibration.json    ┐
4. fit.estimate_prior --input prepared  artifacts/prior.json           │ ONE TURN
5. fit.fit_dispersion --input prepared  artifacts/r_lookup.json, rho.json │ of a loop —
5b. fit.train_baseline --check-convergence  (dry run: did it settle?)  ┘ 8-9 of these
6. evaluate.backtest --input prepared         reports/backtest.json
6b. evaluate.derive_thresholds               reports/thresholds.json  (ops.tune reads it)
8. ops.init_posterior                   artifacts/posterior.json       (once; --force to overwrite)
9. evaluate.shadow --input prepared           reports/shadow.json            (holdout by default)
11. ops.seal                            artifacts/bundle.json
```

Daily production loop: `advance --feed` runs it (ingest → `update
--calibrate-tau` → monitor → assurance → export → status) and stops at
`daily.update --apply`, the operator gate (rule 10); `RUNBOOK.md` is the
operator's document, Appendix A the step list.

## Populations

One prepared parquet; `dp_eligible` + `dp_ineligible_reason` (`cost_missing`
| `non_priceable` | `negative_window` | `window_too_long` | `outcome_unknown`
| `final_hour_restock`) flag what the solver cannot price — nothing economic
is dropped (rule 14). Resolve via `prepare_data.population(d, cfg, which)`:

| Consumer | Population |
| --- | --- |
| baseline, `r`, `rho`, prior, level factors | `eligible`, always |
| `m1` / gate 1 | `integrity`, always |
| DP, backtest, shadow, calibration gate | `dp_eligible`, always (passed explicitly) |

The waterfall (14 rows, `artifacts/split_manifest.json`) records rows,
episodes and COGS after every stage; `kind: hard_drop` drops, the two
`population_gate` rows (`eligible`, `dp_eligible`) only flag. The chain,
the inventory convention, the flow identity and the close rules: §5.2, §12a.

## MEASURED pastes and launch blockers

`load_config(strict=True)` refuses while any RUNTIME_REQUIRED value is null.
**`config.yaml` ships the OWNER's production readings** (their `ops.advance` run of 2026-09-06: `rho` 0.6364, `information_increment` 0.237, `tau_initial` 348.93, the 16-category + `_default` `delta_min_log_bias` map, scrap/margin stops 0.3217/0.0614) **and their postures** (`max_std_shrink` 0.10 — conservative launch; `max_mean_step` 0.796, pasted inside its gate; both guardrail series smoothed over 7 days; `cold_start_shift_std` 0.5) as the defaults — the table in design §12 is the record; a local run on the
fixture re-derives fixture values — read them, never commit them
(`git checkout config.yaml` afterwards). Every paste has one source and one checker:

| Config value | Paste from | Checked by |
| --- | --- | --- |
| `dispersion.rho` | `artifacts/rho.json`, after EVERY retrain (`m` is measured per batch, never pasted) | `artifact mirrors` (strict start-up refuses drift beyond `rho_paste_tolerance_rel`, 1% of the frozen rho — above the ~1e-3 step a `--check-only` turn takes, so a settle is not a new paste; tighter than tau's 5% because rho is frozen for the pilot while tau self-corrects daily) |
| `calibration_fit_trailing_weeks`, `information_increment`, `calibration_gate_band` | the REPORT that derives each (`tune` names it) | `config mirrors reports` — these cannot be null, so the check is the only thing between a pulled repo and a number from another extract |
| `exploration.tau_initial` | `reports/shadow.json` → `tau_initial_derivation` (backtest = cross-check only) | `tau_provenance_error` — shadow refuses a stale paste |
| `exploration.delta_min_log_bias` | `tune` from `backtest.fidelity`, PER CATEGORY as a one-line mapping (own log ratio floored by MAE@W and the gate half-width; `_default` for unseen categories) — null = no floor | `config mirrors reports` |
| `scrap/margin_deterioration_pct` | `tune` pastes the 3σ trailing-mean floor from `reports/thresholds.json` (owner, 2026-08-30); OWNER only when the verdict is `TOO TIGHT`, `BLOCKED`, `LIKELY INERT` or `insufficient history` — all blocking, none pasted | `guardrail floors` |
| `posterior.cold_start_shift_std` | OWNER — launch belief = prior mean − k·std per cell (0.5); read by `init_posterior` and the backtest's DP arm; inert once the posterior has consumed an outcome | `tune` (OWNER reading with `intra_episode_deepening` medians and the like-for-like IL gap) |
| `data.launch_date` | OWNER — null until launch day; once set, `--fit-calibration` schedules through the latest data (the weekly cron) while every sealed fit keeps its pre-launch scope. Never move `split.test_end` for this | `launch blockers`; `calibration_schedule_current` on every `--apply` |

## Where to look

| Working on | Read first |
| --- | --- |
| filter chain, waterfall, eligibility, restocks, scrap | §5.2, §12a; `fit/prepare_data.py`, `common/episodes.py`; rules 13–15 |
| baseline, calibration, fidelity | §5.4, §9.2 (incl. the gate decision tree); rules 1, 4–6 |
| elasticity prior, dispersion | §5.5, §5.6; rules 1a, 2, 3, 7 |
| DP, pricing, exploration, tau, budget | §5.7, §5.8, §5.10 |
| backtest, replay, tau derivation | §5.14, §9; rule 17 |
| "does the agent move after entry?" | §5.7 `intra_episode_moves` (steps on the DP arm's own path, by cost band) — NOT `pct_dp_deepened`, which compares episode means with legacy; shadow re-anchors on legacy's price and cannot measure it |
| shadow phase | §5.13 (holdout default, sampling caveats, tau₀ derivation) |
| the weeks after launch, before launch (`evaluate.pilot_sim`: real engine + daily lane vs a simulated shop; its world, run and faults in `pilot_sim.yaml`, key table in §11.3) | §11.3 |
| posterior, update, operator gate | §5.9, §5.11 |
| monitoring, guardrails, stop conditions, the pilot read | §5.12, §11, §12 |
| events, integration, quarantine | `docs/event_contract.html`; `events/store.py` |
| provenance, seal, freshness, the audit trail (`artifacts/history/<bundle>/<sealed_at>/`) | §5.14a; rule 18 |
| operating the chain end to end, phase order | `ops/advance.py` (`--plan`); rule 1b |
| operator runbook, review tiers | `RUNBOOK.md`, `REVIEW_GUIDE.md` |
| why is it not done the other way? | `docs/learnings.md` |

## Repo conventions

- One package per responsibility: `engine/` prices and learns, `events/`
  records, `common/` defines, `fit/` builds the frozen artifacts,
  `evaluate/` grades them (backtest, shadow, thresholds), `daily/` is the
  production lane in run order, `ops/` drives and gates (`advance`,
  `bootstrap_loop`, `tune`, `status`, `init_posterior`, `seal`), `tools/`
  is out of review scope. A new module goes where its reader is.
- Run modules as `python3 -m package.module` from the repo root.
- `--workers N` (`0` = all cores but one) parallelises backtest, shadow and
  `pilot_sim` (each hour's batch); reports are byte-identical serial or
  parallel — results return in submission order and only the parent
  touches the event store. Each episode draws from its own RNG seeded by
  episode id (the simulator: episode and hour) — order-independent.
- Tests: shared builders live in `tests/conftest.py` (`cfg`,
  `decision_event`/`outcome_event`, `episode_frame`, `_reports`,
  `reports_dir`, `synth_flc` — the synthetic extract, generated once per
  session when `data/` is empty, so a fresh clone's `pytest` passes) —
  extend those, never add a per-file copy. Test a
  behaviour by calling the function; an `inspect.getsource` assertion is
  reserved for an architecture ban (no second copy of X, no shared state in
  a worker) that no behavioural test can express.
- Synthetic fixtures: `tools/make_dummy_flc.py` (`--policy randomized` =
  recoverable elasticity, `--policy legacy` = the production confound). It
  must keep emitting both source inventory conventions — regenerate the
  fixture after any change to them, and read the two printed counts.
- Leadership deck: `python3 -m tools.scenario_deck --workers 0` writes
  `reports/scenarios.html` — twelve scenarios answered by the real solver on
  this machine's config over a state grid (~2 min). Regenerate after any change
  to `engine/dp.py`, `engine/explore.py`, the tier/entry/δ_min config, or the
  posterior prior; rule 19 applies to every number on it. It states what the
  solver does under a chosen demand input, never what a SKU will sell.

## Maintaining the documents

The doc surface is small on purpose: `docs/design.md` (the spec),
`docs/learnings.md` (superseded designs), `docs/event_contract.html` (the
integration contract), this file, `README.md`, `RUNBOOK.md`,
`REVIEW_GUIDE.md`. Two are guarded by tests: `design.md` (every waterfall
stage, gate, flag and population must appear, no retired rule may read as
live — `test_docs_match_the_code.py`) and `event_contract.html` (checked
against `events/store.py` in both directions; its quoted thresholds are NOT
guarded — re-read them when `monitoring.*` moves; the worked episode is real
solver output, regenerate it rather than hand-patch numbers).

**A code change ships with its doc change in the same commit**: this
file's one-home list and paste table, the RUNBOOK step it touches, the
design section (a simulator setting: its `pilot_sim.yaml` comment and the
§11.3 key table, test-guarded), the event contract for any event field,
and a learnings entry when a design was superseded. This file is a router, not a
reference — a 400-line budget, enforced by test; new material goes to
design.md or learnings.md with a one-liner and a pointer here. Quote only
from a gate-passing run, never invent a figure, and never present a
fixture number as a production finding (rule 19).
