# Agent operating guide — Perishable Markdown MVP

This file is deliberately short: it holds only what an agent must obey
without looking anything up. The authoritative specification — including the
rationale and the measured incident behind every rule below — is
**`docs/design.md`** (§ numbers refer to it). Superseded approaches live in
`docs/learnings.md`. When this guide and the design doc disagree, the design
doc wins. Doc maintenance: `docs/maintaining_docs.md`.

## Non-negotiables

Each one-liner is binding on its own; the cited section carries the full
statement and the incident that created the rule.

1. **Never retrain the baseline between two runs you intend to compare** —
   a comparison is valid only when `artifact_versions.baseline_model_version`
   matches in both reports. `--fit-calibration` does NOT retrain; plain
   `train_baseline` and `bootstrap.run` DO. (§5.4)
1a. **Changing the elasticity prior invalidates `rho`, `deff` and the level
   factor** — re-run `fit_dispersion` onward and re-paste `rho`. (§5.6)
1b. **Drive the chain with `python3 -m pipeline.advance`, never by hand-running
   the step list** — it calls `bootstrap.run` (which iterates steps 3b–5b to
   the fixed point, 8–9 turns on production data), settles every paste with
   `--check-only`, and retrains only when the model is absent or `--retrain`
   is given. Re-running the step list to settle calibration retrains the
   baseline (rule 1). (§9.2, Appendix A)
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
    `report vintages` lines call stale. Re-run map: §5.14.
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
- Never re-derive logic that has one home. The homes, and what each
  replaced (a second copy of any of these is a review failure):
  - population filter — `bootstrap.prepare_data.population`
  - episode-scoped cuts — `common.episodes.window_slice`,
    `trailing_weeks_window` (both factor-fit schedules), `week_key`,
    `calendar_days` (the one `n_days`)
  - outcome reconciliation — `common.episodes.adjustment_reason`
  - scrap, IL, margin at episode grain — `common.metrics.episode_economics`
    (+ `settled`, `daily_rates`) over `common.episodes.scrap_units`;
    live events enter it through `pipeline.monitor.event_frame`. The
    floors, the live guardrail, `il_pct`, the business metrics and
    shadow's budget base all read this one frame
  - decision↔outcome pairing and the trading day —
    `events.pairs.match_pairs` (`learnable=` excludes failed pushes),
    `decision_day`, `price_matches`
  - anchor rows — `common.episodes.is_anchor_row`
  - guardrail deviation and verdicts — `common.guardrail.deviation`,
    `verdict_is_blocking`, `verdict_is_insufficient`
  - spread accounting — `pricing.explore.SpreadLedger`; the tau controller
    walk — `pricing.explore.walk_tau` (production and shadow's trace);
    the backtest's forward simulation — `backtest.replay._simulate_arm`
    (+ `_dp_price`)
  - rho — `common.config.intraclass_correlation`; `m` per batch —
    `deff_from_episodes`
  - JSON in/out — `common.io.read_json` / `write_json` (NaN-safe)
  - the discount-grid epsilon — `pricing.dp.TIER_EPS`; own-data prior
    weight — `common.config.OWN_DATA_WEIGHT`
  - pastable config keys — `pipeline.tune.KEYS` (anchor, measured, rerun);
    the status "not run" prologue — `pipeline.status._needs`
- Never invent a SET BY OWNER value; never drive a quarantine count to zero
  with a catch-all reason.
- Quote the sampling caveat with any sampled-run count — a zero over a
  sample is not a proof over the window.
- `python3 -m pytest tests/` must pass before any push (~5.5 min; the
  end-to-end module runs the bootstrap chain in subprocesses).

## Setup

```bash
pip install -r requirements.txt
python3 -m pytest tests/
```

All commands run from the repo root — paths in `config.yaml` are relative to
it, and running a module from elsewhere silently reads/writes the wrong
artifacts.

## Driving the chain — `pipeline.advance`, not the step list

```bash
python3 -m pipeline.advance --plan    # phase table + next steps, touches nothing
python3 -m pipeline.advance           # runs to the next HUMAN decision and stops
```

It owns the order (bootstrap → tune/settle → posterior → shadow → owner
keys → launch_date → weekly re-fit → daily lane) and recomputes state from
disk every run. It never retrains unless the model is absent or `--retrain`
is given, re-grades any report whose bundle or config moved, and stops on
every SET BY OWNER null with the evidence. Its daily lane walks tau
(`pipeline.update --calibrate-tau`, no operator) and stops at
`pipeline.update --apply`, which it never runs: learning is gated daily
and per cell — a fast category updates the day its batch has the evidence.
Read its stop before doing anything by hand; the sections below explain
the steps it runs.

## Running the bootstrap — use `bootstrap.run`, not the step list

**Do not hand-run the steps in order. Run this:**

```bash
python3 -m bootstrap.download_flc            # step 0, only if you need a fresh extract
python3 -m bootstrap.run --input data/flc_raw.parquet
python3 -m bootstrap.init_posterior          # step 8, once
python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
```

`bootstrap.run` is the whole bootstrap: it runs 1 and 3, then **iterates
3b–5b until the fixed point CONVERGES**, then 6, 6b, 11 and `status`. It
exits non-zero if the loop never settles. The `status` it ends on is
ADVISORY at that point — `tau_initial` is null and shadow has not run, so
red rows there are the next steps, not a broken bundle; `status` gates the
pilot (RUNBOOK), never the bootstrap.

**Why this is not optional.** Steps 3b–5 are one TURN of a fixed-point
iteration, not three steps in a line: the factor solve consumes `r`, while
`r`, `rho` and the prior are all fitted against *calibrated* `mu_ref`. Run
the list top-to-bottom once and you get one turn — artifacts that disagree
with each other, and a `--check-convergence` that says NOT CONVERGED. **The
owner measures 8–9 turns to settle on the production extract** (the repo
fixture takes 3–4 because it is small — rule 19). Nobody is going to hand-run
that correctly, and the obvious repair is the wrong one: re-running the step
list restarts at 3, which RETRAINS THE BASELINE, moves every artifact, resets
the fixed point and breaks rule 1. That loop is the one an agent cannot
escape by trying harder; `bootstrap.run` trains once, outside the loop, and
is the only supported way to reach a converged chain.

After a **config paste**, settle without retraining:

```bash
python3 -m bootstrap.run --check-only      # loop against the artifacts on
                                           # disk, NO retrain. What
                                           # pipeline.tune tells you to run.
```

`--max-turns` (default 20) is a runaway guard, not a budget — the STALL test
stops a loop that has gone three turns without a new best, long before the
cap. Raise it only if the trajectory printed at exit is still contracting.

Two details worth knowing when reading its output: on the first turn there is
no `r_lookup` yet, so 3b uses the raw-mu basis and the second turn gets the
censored one (early turns are expected to move a lot); and the loop runs
`estimate_prior --fast`, which drops `fold_spread` — that only widens the std
FLOOR while factors follow the prior MEAN, so it cannot move the fixed point.
The artifact still gets a FULL prior once the loop settles. (§9.2)

### The steps it runs — for debugging ONE step, not for driving the pipeline

```
step                                          writes
0. bootstrap.download_flc                     data/flc_raw.parquet   (Redshift; REDSHIFT_* from ~/.env)
1. bootstrap.prepare_data --input <raw>       data/prepared.parquet, artifacts/split_manifest.json
3. bootstrap.train_baseline --input prepared  artifacts/baseline_model.txt, feature_schema.json
3b. bootstrap.train_baseline --fit-calibration artifacts/calibration.json    ┐
4. bootstrap.estimate_prior --input prepared  artifacts/prior.json           │ ONE TURN
5. bootstrap.fit_dispersion --input prepared  artifacts/r_lookup.json, rho.json │ of a loop —
5b. bootstrap.train_baseline --check-convergence  (dry run: did it settle?)  ┘ 8-9 of these
6. backtest --input prepared                  reports/backtest.json
6b. bootstrap.derive_thresholds               reports/thresholds.json  (pipeline.tune reads it)
8. bootstrap.init_posterior                   artifacts/posterior.json       (once; --force to overwrite)
9. pipeline.shadow --input prepared           reports/shadow.json            (holdout by default)
11. bootstrap.seal                            artifacts/bundle.json
```

Daily production loop (Lane C — full operator guidance in `RUNBOOK.md`):

```bash
python3 -m pipeline.update --calibrate-tau   # daily tau walk, no operator
python3 -m pipeline.update --apply     # OPERATOR GATE (rule 10), daily
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
The chain, the inventory convention, the flow identity and the close rules
are specified in §5.2 and §12a.

## MEASURED pastes and launch blockers

`load_config(strict=True)` refuses while any RUNTIME_REQUIRED value is null.
Every paste has one source and one checker:

| Config value | Paste from | Checked by |
| --- | --- | --- |
| `dispersion.rho` | `artifacts/rho.json`, after EVERY retrain (`m` is measured per batch, never pasted) | `artifact mirrors` (strict start-up refuses drift) |
| `calibration_fit_trailing_weeks`, `information_increment`, `calibration_gate_band` | the REPORT that derives each (`tune` names it) | `config mirrors reports` — these ship fixture values and cannot be null, so the check is the only thing between a pulled repo and a foreign number |
| `exploration.tau_initial` | `reports/shadow.json` → `tau_initial_derivation` (backtest = cross-check only) | `tau_provenance_error` — shadow refuses a stale paste |
| `scrap/margin_deterioration_pct`, `min_detectable_effect_pct` | OWNER, from `reports/thresholds.json` — `TOO TIGHT`, `BLOCKED` and `LIKELY INERT` are all blocking | `guardrail floors` |
| `ab_test.active` | OWNER — `false` until the A/B starts; the arm labels cannot say which regime is in force | `monitor.*_deterioration.basis` |
| `data.launch_date` | OWNER — null until launch day; once set, `--fit-calibration` schedules through the latest data (the weekly cron) while every sealed fit keeps its pre-launch scope. Never move `split.test_end` for this | `launch blockers`; `calibration_schedule_current` on every `--apply` |

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
| docs | `docs/maintaining_docs.md` |
| operating the chain end to end, phase order | `pipeline/advance.py` (`--plan`); rule 1b |
| operator runbook, review tiers | `RUNBOOK.md`, `REVIEW_GUIDE.md` |
| why is it not done the other way? | `docs/learnings.md` |

## Repo conventions

- Run modules as `python3 -m package.module` from the repo root.
- `--workers N` (`0` = all cores but one) parallelises backtest and shadow;
  reports are byte-identical serial or parallel — results return in
  submission order and only the parent touches the event store.
- Each episode draws from its own RNG seeded by episode id — draws are
  order-independent by design.
- Tests: shared builders live in `tests/conftest.py` (`cfg`,
  `decision_event`/`outcome_event`, `episode_frame`, `_reports`,
  `reports_dir`) — extend those, never add a per-file copy. Test a
  behaviour by calling the function; an `inspect.getsource` assertion is
  reserved for an architecture ban (no second copy of X, no shared state in
  a worker) that no behavioural test can express.
- Synthetic fixtures: `tools/make_dummy_flc.py` (`--policy randomized` =
  recoverable elasticity, `--policy legacy` = the production confound). It
  must keep emitting both source inventory conventions — regenerate the
  fixture after any change to them, and read the two printed counts.
