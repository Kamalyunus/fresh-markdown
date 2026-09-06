# Runbook — operating the Perishable Markdown MVP

For the engineering team and the product owner. The order of operations is
code — `python3 -m ops.advance` — so this document is about the parts
the process cannot do: what engineering builds, what the owner decides, and
what a stop or a red line means. The authoritative spec is
`docs/design.md`; the integration contract is `docs/event_contract.html`;
`REVIEW_GUIDE.md` maps the code by risk tier; `AGENTS.md` is what an agent
reads before touching the repo.

All commands run from the repo root. `data/`, `reports/`, `artifacts/`,
`events_store*/` are run outputs and never committed. Credentials
(`REDSHIFT_*`) live in `~/.env`, never in config or code.

## Who does what

- **Owner** — tell your agent: *read `AGENTS.md`, then run `ops.advance`
  until `reports/launch_readiness.md` says it is waiting on
  `data.launch_date`.* It pulls the extract for the config's split and
  hold-out dates (`download_flc` exits non-zero if a pull does not cover
  `train_start` through the hold-out's end — the file is still written, the
  chain does not proceed on it), trains once, derives and pastes every MEASURED value,
  runs shadow on the hold-out, and stops at each decision only you can
  make, printing the evidence. Read the section *What the owner decides*.
  For leadership, `python3 -m tools.scenario_deck --workers 0` writes
  `reports/scenarios.html`: twelve situations (heavy stock, hours left, high
  COGS, exploration cost, legacy ramp, demand shock, restock, dead stock,
  learning, refusals) answered by the production solver on the config in
  force. Demand there is a slider, not a forecast; the pilot's own
  outcomes are the evidence.
- **Engineering** — build Lane B (below) against the event contract,
  choose the pilot episodes (**spanning FCs and categories** — several of
  each, so no single site or category carries the read and exploration is
  tested across the catalogue at small scale; there is no A/B and no
  control arm), then run the daily lane on a cron and read its stop:

```bash
python3 -m ops.advance --plan       # where the chain is, what runs next; touches nothing
python3 -m ops.advance              # run to the next human decision, then stop
python3 -m ops.advance --feed <yesterday's hourly parquet>   # the daily lane
python3 -m ops.advance --report     # regenerate reports/launch_readiness.md
```

`advance` recomputes the state from disk every run, so running it again
after any action is always safe. It never retrains unless the model is
absent or `--retrain` is given — a moved training input (`data.split`,
`exclusion_window`, the model's own keys) is a STOP naming `--retrain`, and
that retrain re-runs the stale shadow itself before reading any report; it
re-runs a report only when its bundle moved or a config key that report
reads moved (a paste of what the report itself measured invalidates
nothing; a key only the backtest reads re-runs the backtest alone); it
never invents a value. Every stop writes
`reports/launch_readiness.md` — what ran per phase, every config value the
process changed (before, after, why, source), the config in force, status,
and what is waited on. Its stops, in order: a tune BLOCK · a failed shadow
gate · the owner keys · `data.launch_date` · a stale extract · a red
`status` · and, daily, **`daily.update --apply`**, which stays a human's.

---

## What the owner decides, and how to read it

The process stops with the evidence printed; these are the readings behind
each decision.

1. **Level diagnostic (a review, not a gate).** `reports/backtest.json` →
   `calibration_gate_value` against `calibration_gate_band`. Out of band is
   a drift/staleness reading — the decision tree in `docs/design.md` §9.2
   separates wobble from trend. WARN in `status`, never a launch blocker.
   While there, read `policy_deltas.intra_episode_moves`: how often the
   agent steps after entry on its own path, by cost band, against the share
   of episodes above the deepening bar. Near zero with the bar unreached is
   enter-and-hold at the launch prior (design §5.7), not a pinned price;
   `pct_dp_deepened` answers a different question (episode mean vs legacy).
2. **Prior gate.** `artifacts/prior.json`, in this order:
   `wrong_sign_categories` → per-category `mean/std/std_basis` →
   `holdout_comparison` (read `information_available_per_row` first). There
   is no pass flag; a pooled or uniform prior is a designed outcome.
3. **Shadow gate.** `reports/shadow.json`: completeness ≥ 99% and zero
   cost-floor violations, then `exploration_budget.spend_over_budget`
   (over 2× → do not launch at this tau) and `tau_controller_trace` — day
   one is an out-of-sample test of the derived tau. `learning_yield_would_be`
   says how fast the pilot can learn and whether evidence or the calendar
   binds; `delta_min` on the decision events says which categories the
   forced-move floor binds. `exploration_would_be.forced_rate` is what the
   budget buys (it equals 1 − `affordable_set_empty_rate`); to change it
   read `exploration_budget_sweep` — one row per (`budget_share_of_il`,
   `delta_min_bias_multiple`) with forced rate, spend, mean move and
   `information_rel` — and set the pair, then re-run shadow once. A
   smaller share forces less at the same depth; a larger multiple forces
   less but deeper.
4. **The owner keys** (`advance` stops here; the values in force are the
   table in design §12 — shrink 0.10, step 0.796, both guardrail series
   smoothed 7 days, k 0.5, set 2026-09-06): `max_std_shrink` first
   (`information_increment` derives from it), then `max_mean_step` when
   `backtest.step_sensitivity` says the re-price exceeds the auto-apply
   gate (inside it, `tune` pastes it). The stop thresholds
   `scrap_deterioration_pct` and `margin_deterioration_pct` are PASTED at
   the 3σ trailing floor `thresholds.json` stamps; they come to you only
   on `TOO TIGHT`, `BLOCKED`, `LIKELY INERT` or `insufficient history`,
   and the answer there is the basis or the metric, never a number.
   `posterior.cold_start_shift_std` (0.5) is how aggressive the day-one
   belief is: launch |ε| = prior |ε| + k·std per cell. Read
   `backtest.policy_deltas`: `intra_episode_deepening` (prior vs launch
   median against the bar), `dp_clearance`, `dp_il_reduction_pct_of_legacy`
   and `intra_episode_moves` — more k buys clearance and movement and pays
   IL under the model. Change it before launch only; after the first
   consumed outcome the learner owns the mean.
5. **`data.launch_date`**, on launch day. It lets the weekly level re-fit
   schedule past `split.test_end`; never move `split.test_end` for this.
   `advance` then re-fits, re-seals, and `status` must be green.

**What the live run reads, and what pins it.** `engine.decide` reads the
model, factors, `r_lookup` and `rho` from the paths in `config.yaml`, the
runtime knobs from `config.yaml` itself, and cells plus tau from
`artifacts/posterior.json`; nothing reads a report. The seal pins all of
it: the artifact hashes, the config (digest and snapshot) and the library
versions — `status`'s `artifact bundle` row is red when any of them moved,
and every decision event carries `config_digest`, so an hour maps to one
`artifacts/history/<bundle>/<sealed_at>/` snapshot. `artifacts/` and `config.yaml` are not in git: production runs
from the directory `advance` ran in, or ships the latest snapshot whole.

**The daily `--apply` gate.** One human approves at most one posterior step
per cell per day; each cell triggers on its own batch. Before approving:

| Field | Approve when | Hold when |
| --- | --- | --- |
| `predictive_check` | `worse_than_a_flat_prior: false`, or a one-off | it persists across batches — the belief tightened faster than the evidence; escalate before more updates |
| `bound_clipped` | occasional | most updates clip — step cap or increment mis-sized; escalate |
| `batch_oldest_outcome_age_days` | near the expected cadence | growing without a trigger — the loop is stalling; check volumes and tau |
| event-quality gates | green (the command refuses on red) | never work around a refusal |
| `calibration_schedule_current` | green | red means the weekly re-fit was missed — `--apply` refuses, because learning from prices set on stale factors banks evidence about a model that is not the one running |

`tau` needs no approval: `advance --feed` walks it one clipped step per
closed day (`update --calibrate-tau`); a second run on the same day is a
no-op, and a missed day is graded, not skipped.

---

## Lane B — Price (hourly; the only lane where engineering builds code)

The engine is `engine.decide`: state in, price + decision event out, or
`StateRejected` — it never returns a best-effort price for a state it cannot
validate. Engineering owns everything on the other side of the event
contract:

- the hourly scheduler and transport that call `decide` per SKU × FC
  (the 12-field request in `docs/event_contract.html` §03);
- applying the returned price (the applied price must be the returned one —
  the mismatch rate is gated at 1%);
- reporting **failed price pushes** — one row per failed hour, as a table
  (parquet/CSV) or JSONL (`sku_id`, `fc`, `date`, `hour_of_day`, `reason`);
  no row means the push succeeded; keys are normalised on read (a datetime
  `date`, an id read back as a float);
- holding ONE `PosteriorStore` per process is fine, but **reload it once
  per decision batch** (`store.reload()`): the monitor writes a suspension
  and `--apply` writes updates into the same file from another process, and
  a handle that never reloads keeps drawing on a suspended pilot;
- a defined fallback for `StateRejected` (hold the current price; alert on
  rate);
- the daily cron: `advance --feed <yesterday's hourly parquet>`, which
  ingests outcomes, walks tau, writes monitor/assurance/status/exports and
  stops at `--apply`.

Outcomes are NOT engineering's to produce: `daily.ingest_outcomes`
builds them from the hourly FLC feed, matched to decisions by (SKU, FC,
date, hour), deriving `adjustment_reason`, `is_stockout` and the offered
price itself. §08 of the contract page is the pre-build feasibility
checklist, and §01 — deliberately first — is the definitions and claims
register: every derivation stands on source-data meanings only engineering
can confirm, so align on §01 before anything else.

The caller reads `tau` from `PosteriorStore.tau(cfg)` — **not** from
`config.exploration.tau_initial`, which is only the launch value and never
moves. Safety properties (cost floor, price monotonicity) are structural in
the engine's action set: there is nothing to configure and nothing to check
in the caller.

---

## Red-line table — what a red `status` line means and the response

| Line | Response |
| --- | --- |
| stop condition fired (overspend >2× on `persistence_days` consecutive days, mismatch, duplicates, guardrail) | the monitor suspends exploration in the posterior state; `decide` stops drawing and **exploitation pricing continues**; `status` shows `exploration SUSPENDED since …`. Investigate, then a human resumes with `python3 -m daily.update --resume-exploration` — never restart blindly. A resumed pilot is not re-suspended by the days it spent suspended (they read as zero spend); a fresh fire is a fresh two-day streak. The posterior file is production state from the first walked τ: `advance` never re-initialises it after launch |
| `config mirrors reports` FAIL | a MEASURED paste disagrees with the report that derives it, or the report could not measure it (NOT RUN). `python3 -m ops.tune` prints the reason; `advance` re-pastes what it can |
| `guardrail floors` WARN | "insufficient history" — nobody measured the floor, so the stop was not checked. Not a pass: more closed-episode history, then re-run `derive_thresholds` |
| `assurance · reproduction` FAIL | something moved under the solver (config edit, artifact swap, deploy, library). Diff the bundle first: `artifact bundle` line, then `artifact mirrors`, then the live `artifacts/` against the latest `artifacts/history/<bundle>/<sealed_at>/` snapshot (every seal leaves one, with the config and posterior of the moment; its `MANIFEST.json` names the reason — `bootstrap`, `check-only`, `retrain`, `weekly-refit`, `config`, `libraries` — the config and library versions in force, and every copy was re-hashed against the seal when written). The failing decisions name their own `config_digest` |
| `artifact bundle` FAIL — `config moved` / `libraries moved since sealing` | the seal covers the environment too. A config edit or a library upgrade changed what the next hour is priced with; nothing prices on it until it is sealed. If the change was deliberate: `python3 -m ops.seal --reason config|libraries` (`advance` does it once nothing is left to paste) — the snapshot it leaves is the record. If it was not, the `MANIFEST.json` of the latest snapshot holds the config and versions that were in force |
| `artifact mirrors` FAIL | config paste and its source disagree (rho). Read the **bundle** line before re-pasting — the stale side is not always config |
| `report vintages` FAIL | a report was produced against a model no longer on disk — its gate rows grade a ghost. `advance` re-runs it; do not launch on it |
| `calibration_coverage` says `STALE FACTORS IN USE` | the weekly re-fit was missed: rows were priced on frozen factors. `advance` re-fits and re-seals; re-run the report |
| posterior std flat ≥ alert days | the loop is dead: no committed update. Check batch age, tau, volumes — in that order |
| guardrail breach (scrap/margin, 2 consecutive days) | business decision, not a code fix — escalate to the owner with the monitor's trailing comparison |
| `INSUFFICIENT` verdicts | not a pass. A thin window said so; widen or wait. Assurance's top line stays `INSUFFICIENT` until every check ran |

**Never** retrain between two runs you intend to compare (comparisons are
valid only when `baseline_model_version` matches); never tune anything on
the hold-out window; never hand-edit `artifacts/posterior.json`; never
re-derive filter logic outside `prepare_data.population`; never drive a
quarantine count to zero with a catch-all reason.

---

## RACI

| Decision / step | Engineering | Product owner |
| --- | --- | --- |
| Run `advance`, CI, deploys | **R/A** (owner may run it via their agent) | — |
| Prior gate verdict; level-diagnostic review | run & present | **A** |
| MEASURED pastes into config | — (the process, from named report fields only) | informed |
| `SET BY OWNER` thresholds, rails, `budget_share_of_il`, `launch_date` | — | **A** |
| Lane B service, push-failure feed, daily cron | **R/A** | — |
| Daily `--apply` approval | **R** (pilot: owner may retain) | consulted on escalations |
| Guardrail breach response, pilot readout | informed | **A** (decision table in design §11.2) |
| Pilot episode set spans FCs × categories | **R/A** | consulted |
