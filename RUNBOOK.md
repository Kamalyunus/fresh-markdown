# Runbook — operating the Perishable Markdown MVP

For the engineering team and the product owner. The order of operations is
code — `python3 -m ops.advance` — so this document is about the parts
the process cannot do: what engineering builds, what the owner decides, and
what a stop or a red line means. The authoritative spec is
`docs/design.md`; engineering's page is `docs/engineering_handover.html`
(the process, then the integration contract as its appendices);
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
  outcomes are the evidence. Before launch day, rehearse the weeks after
  it: `python3 -m evaluate.pilot_sim` walks the hourly engine and this
  whole daily lane against a simulated shop (design §11.3) in a workspace
  under `sim/`, and `reports/pilot_sim.json` grades what a healthy launch
  shows. The shop and the run are set in `pilot_sim.yaml` (never in
  `config.yaml`, which the sim rehearses as it stands); `--fault
  mismatch:0.05` and friends override it for one run to check that the
  gates and stops fire when they should.
- **Engineering** — build Lane B (below) against the event contract
  from `integration/`, the standalone folder maintained in place (its
  code, `pricing/`, is the folder's own and stateless — a row priced
  from itself, the id its opening tag, the price in force piped back —
  held equal to the repository's hour by the parity test; three commands,
  the hourly price (exploit only), the morning pull from the warehouse and
  the morning table from it; its `README.md`
  is the handoff; every seal and every `advance` run syncs the five
  artifacts and the pruned config into it; the checks
  run HERE on the files they send; the id is theirs,
  nothing there assigns one; the learning lane stays in the repository and
  reads the store that folder writes),
  choose the pilot episodes (**spanning FCs and categories** — several of
  each, so no single site or category carries the read and exploration is
  tested across the catalogue at small scale; there is no A/B and no
  control arm), then run the daily lane on a cron and read its stop:

```bash
python3 -m ops.advance --plan       # where the chain is, what runs next; touches nothing
python3 -m ops.advance              # run to the next human decision, then stop
python3 -m ops.advance --feed <yesterday's hourly parquet> [--failures <failed pushes>]   # the daily lane
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
and what is waited on. Its stops, in order: a moved training input
(`--retrain` is yours) · a tune BLOCK · a MEASURED value a report could not
derive · a failed shadow gate · the owner keys · `data.launch_date` · a
the weekly extract refresh (after launch it is the cron's own step, not a stop: when the factor schedule falls behind the week being priced, `advance` pulls the extract through yesterday, prepares it, re-fits and re-seals in the same run; it stops only when an extract that already reaches yesterday still cannot reach the week) · then the daily lane
(ingest, tau walk, monitor, assurance, export, status — it runs even
while `status` is red, because the rows that go red after launch, a fired
stop or an assurance verdict, are refreshed only by this lane; stopping
before it once deadlocked the lane on a fired stop) · a red `status`,
naming the rows and the resume path · and, daily, **`daily.update --apply`**,
which stays a human's. A step that fails, a plan that loops or a round
budget that runs out are stops too: journaled, reported, exit 1.

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
   less but deeper — and deeper is what keeps a level error from tilting
   into ε (design 5.8: a short forced move leaves the model right at one
   price and multiples wrong at the deep markdowns). With a fixed IL
   budget the owner's posture is the larger gap and the lower forced
   rate; `delta_min_bias_multiple` is that lever, yours to set from the
   sweep.
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
   `baseline_model.calibration_fit_trailing_weeks` (W, the level factors'
   trailing fit window) is yours too: the backtest's rolling-origin sweep
   RECOMMENDS a W with its evidence, `tune` reports it as an owner
   decision, and nothing pastes it -- a pasted W turned the calibration
   loop, re-ran the backtest, re-scored the sweep and re-ran shadow, the
   heaviest re-run short of a retrain, and a near-tie could cycle it.
   Move it only on a material win; the change still turns the loop once.
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
`artifacts/history/<bundle>/<sealed_at>/` snapshot. `artifacts/` is not in
git (`config.yaml` IS: it ships the owner's production readings): production
runs from the directory `advance` ran in, or ships the latest snapshot whole.
The handover page's task 0.7 is the ship list, the two cron lines, the time
zone rule, the exit codes and the backup rule.

**The daily `--apply` gate.** One human approves at most one posterior step
per cell per day; each cell triggers on its own batch. Before approving:

| Field | Approve when | Hold when |
| --- | --- | --- |
| `predictive_check` | `worse_than_a_flat_prior: false`, or a one-off | it persists across batches — the belief tightened faster than the evidence; escalate before more updates |
| `bound_clipped` | occasional | most updates clip — step cap or increment mis-sized; escalate |
| `batch_oldest_outcome_age_days` | near the expected cadence | growing without a trigger — the loop is stalling; check volumes and tau |
| event-quality gates | green (the command refuses on red) | never work around a refusal |
| `calibration_schedule_current` | green | red means the weekly re-fit was missed — `--apply` refuses, because learning from prices set on stale factors banks evidence about a model that is not the one running (the tau walk still commits: it moves on spend, not factors). A week the re-fit ran and HELD at the anchor (too few anchor rows) is green with `held_at_anchor: true`: production prices on the anchor factors that week — read it, it is not a refusal |

`tau` needs no approval: `advance --feed` walks it one clipped step per
closed day (`update --calibrate-tau`) once the trailing IL base spans
`budget_il_window_days` — the first week after launch holds it, and the
walk rows say `held`; a day exploration was suspended on is held too
(`exploration suspended`: nothing was drawn, so its zero spend is no
reading), and only pushes that executed count as spend. A second run on
the same day is a no-op, and a missed day is graded, not skipped — a day
whose outcomes arrived after a later day was walked is graded on the
next walk and printed (`day(s) whose outcomes arrived late`).

---

## Lane B — Price (hourly; the only lane where engineering builds code)

The engine is `engine.decide`: state in, price + decision event out, or
`StateRejected` — it never returns a best-effort price for a state it cannot
validate. Engineering owns everything on the other side of the event
contract:

- the hourly cron: the shelf at the top of the hour in the feed's own
  schema (the snapshot) carrying the `episode_id` THEY assign — the
  producers own the data and, in time, the pipeline, so the id is theirs;
  `ops.assign_episode_ids` is the chain's rule (`EPISODE_RULE`) as a
  script for one hour against the hour before, to run or port — through
  **`ops.price_hour`** — which reads the id as given (a new id is an
  entry, the id the store last priced on the shelf continues it), takes
  the anchor from the price in force, evaluates the rule only to count
  the ids that disagree (`episode_ids_disagreeing_with_the_rule`,
  `LIVE_RULE`; from the store alone only a counter reset is decisive,
  the rest is `episode_ids_not_decidable_from_the_store` and vanishes
  when the closed rows ride along; a shelf with no hour to step from is
  `episode_ids_the_rule_could_not_check`, a gap, never a contradiction),
  builds the 12-field requests and prices them through
  `ops.price_batch` (the caller beneath it: requests in, a price per
  request out) — then applying the `apply_price` column it returns.
  `ops.check_inputs` checks their three tables (snapshot, feed, failed
  pushes) before launch, the id column included. The nightly feed carries
  the same `episode_id` too, so a window is one group-by for every reader
  and the checker can read a whole day of their ids against `EPISODE_RULE`
  at once — on the window BOUNDARIES, never the id's spelling, since the
  scheme is theirs. The posterior is
  read once per batch; every decision is in the store before its price
  returns; an hour already priced is refused (`already_priced`). Every request is read in one spelling (ids as the
  hour key spells them, the day as `YYYY-MM-DD`), so a parquet timestamp
  or an id read back as `7.0` prices and writes. Call it as it is (a file
  drop per hour) or lift the service out of it: `engine.state.build_states`
  is the one request → state (an entry request: the frozen model's
  `mu_ref_path` on its two demand-rate features, read off the day's
  feature table `features/<today>.parquet` — `daily.features` writes it
  every morning from the rolling feed, `engine.state.ref_rate_table`;
  both features read strictly before the opening date, so the table's
  row is the batch's number and the batch reads no history; a later hour
  of a known episode: the entry's stored path SLICED to the hour and the
  features it recorded, extended only when a restock grew the window;
  `r` from the lookup) — never re-derived. `--history` (the trailing feed
  itself) computes the same features inside the batch, for the day
  before the first morning. Read the batch report's
  `requests_with_unknown_features` (fresh forecasts with no history
  behind them) and `non_entry_requests_without_stored_path` (later hours
  of episodes the store never priced) after the first batches: a whole
  batch of either means the history table or the episode ids do not meet
  the requests. `python3 -m tools.e2e_cycle` runs one whole cycle —
  requests, decisions, the shop's feed, ingest, exports — in a workspace
  under `sim/e2e`, before any of your code exists (the pilot simulator's
  shop, priced through `ops.price_batch`; a rejected request holds the
  shelf and is priced again next hour, as your fallback should);
- applying the returned price (the applied price must be the returned one —
  the mismatch rate is gated at 1%);
- reporting **failed price pushes** — one row per failed hour, as a table
  (parquet/CSV) or JSONL (`sku_id`, `fc`, `date`, `hour_of_day`, `reason`);
  no row means the push succeeded; keys are normalised on read (a datetime
  `date`, an id read back as a float); a reported failure is counted apart
  (`push_failures_reported`), never as a price mismatch — the mismatch gate
  catches only the pushes you did not report;
- holding ONE `PosteriorStore` per process is fine, but **reload it once
  per decision batch** (`store.reload()`): the monitor writes a suspension
  and `--apply` writes updates into the same file from another process, and
  a handle that never reloads keeps drawing on a suspended pilot;
- a defined fallback for `StateRejected` (hold the current price; alert on
  rate);
- the daily cron: `advance --feed <yesterday's hourly parquet> --failures
  <that day's failed pushes>` (omit `--failures` on a day with none), which
  ingests outcomes, walks tau, writes monitor/assurance/status/exports and
  stops at `--apply`; after ingest it writes today's feature table
  (`daily.features`), which every batch that day joins on SKU × FC. Once
  a week the same call refreshes the extract
  (`download_flc` through yesterday, `prepare_data`, the level re-fit, the
  re-seal) before the lane runs, so the cron host holds `REDSHIFT_*` in
  `~/.env` and the pull's runtime lands in that morning.

Outcomes are NOT engineering's to produce: `daily.ingest_outcomes`
builds them from the hourly FLC feed, matched to decisions by (SKU, FC,
date, hour), deriving `adjustment_reason`, `is_stockout` and the offered
price itself. Every event id is the hour's key —
`feed-<sku>|<fc>|<date>T<hh>`, `dec-<sku>|<fc>|<date>T<hh>` and, for a
shelf-hour seen but NOT priced, `rej-<sku>|<fc>|<date>T<hh>` — so
engineering can name any of them from the feed row and they differ only in
their prefix. A rejection holds no price, never enters `priced_hours` (a
corrected hour stays free to price) and never reaches the evidence; it is
there so a refused hour is distinguishable from an hour that was never
sent, which is what makes `episode_ids_the_rule_could_not_check` mean a
missing hour. The exported tables (`daily.export_events`) carry the
shelf-hour in the FEED'S own spelling and types beside the event's fields
— `skuseq`, `fc`, `date` as a date, `hour` — so appending a night's
decisions to the hourly feed is a four-column join with no rename and no
cast; a SKU id that is not an integer is null in `skuseq` and counted.
Two decisions priced for one hour (a retried batch) match neither and
are counted (`decisions_colliding_on_hour`); the store itself refuses a
second decision for a priced hour — under the store's lock, with the tail re-read first, so two
overlapping hourly runs cannot both price an hour — and a second outcome for a decision
(`outcomes_per_decision_over_one`), and an outcome without `is_stockout`
never lands (`missing_stockout_field`). The monitor's safety block
carries the decision side of completeness — `decisions_colliding_on_hour`
and `decisions_without_outcome` over the days the feed has answered for —
beside the outcome side. §08 of the contract page is the pre-build feasibility
checklist, and §01 — deliberately first — is the definitions and claims
register: every derivation stands on source-data meanings only engineering
can confirm, so align on §01 before anything else.

The caller reads `tau` from `PosteriorStore.tau()` — **not** from
`config.exploration.tau_initial`, which is only the launch value and never
moves. Safety properties (cost floor, price monotonicity) are structural in
the engine's action set: there is nothing to configure and nothing to check
in the caller.

---

## Red-line table — what a red `status` line means and the response

| Line | Response |
| --- | --- |
| stop condition fired (overspend >2× on `persistence_days` consecutive days — no reading while the IL base is shorter than `budget_il_window_days`, the first week after launch, `engine.budget.budget_base_ready` — mismatch, duplicates, guardrail) | the monitor suspends exploration in the posterior state; `decide` stops drawing and **exploitation pricing continues**; `status` shows `exploration SUSPENDED since …`. `advance --feed` keeps running the lane (ingest, tau walk, monitor, assurance, export) while the row is red — that is what lets the windowed rate dilute and the monitor re-read — and stops before the operator gate naming the row; `fired` clears on its own once the window or the streak is back under the threshold, the SUSPENSION only when a human runs `python3 -m daily.update --resume-exploration` after investigating — never restart blindly. A resumed pilot is not re-suspended by the days it spent suspended (they read as zero spend); a fresh fire is a fresh two-day streak. The posterior file is production state from the first walked τ: `advance` never re-initialises it after launch |
| `config mirrors reports` FAIL | a MEASURED paste disagrees with the report that derives it, or the report could not measure it (NOT RUN). `python3 -m ops.tune` prints the reason; `advance` re-pastes what it can |
| `guardrail floors` WARN | "insufficient history" — nobody measured the floor, so the stop was not checked. Not a pass: more closed-episode history, then re-run `derive_thresholds` |
| `assurance · reproduction` FAIL | something moved under the solver (config edit, artifact swap, deploy, library). Diff the bundle first: `artifact bundle` line, then `artifact mirrors`, then the live `artifacts/` against the latest `artifacts/history/<bundle>/<sealed_at>/` snapshot (every seal leaves one, with the config and posterior of the moment; its `MANIFEST.json` names the reason — `bootstrap`, `check-only`, `retrain`, `weekly-refit`, `config`, `libraries` — the config and library versions in force, and every copy was re-hashed against the seal when written). The failing decisions name their own `config_digest` |
| `artifact bundle` FAIL — `config moved` / `libraries moved since sealing` | the seal covers the environment too. A config edit or a library upgrade changed what the next hour is priced with; nothing prices on it until it is sealed. If the change was deliberate: `python3 -m ops.seal --reason config|libraries` (`advance` does it once nothing is left to paste) — the snapshot it leaves is the record. If it was not, the `MANIFEST.json` of the latest snapshot holds the config and versions that were in force. A `config`/`libraries` re-seal REFUSES when an artifact also moved since the previous seal (it names which): it may not bless an edited or re-fitted artifact as the record. Seal that under the reason that changed it (`check-only`, `weekly-refit` — the calibration alone — `retrain`), or restore the artifact from the latest snapshot, then re-run `advance` |
| `boundary solutions` WARN | a fit is pinned at a search bound (rule 3): a prior category pooled off `epsilon_min`, a level factor at a bracket end (`calibration.json → pinned_cells`, the GLOBAL parent), an `r` at the search bound. Not an estimate — investigate the cell; widening `epsilon_min` or the bracket is a config decision, never a paste |
| `artifact mirrors` FAIL | config paste and its source disagree (rho). Read the **bundle** line before re-pasting — the stale side is not always config |
| `report vintages` FAIL | a report was produced against a model no longer on disk — its gate rows grade a ghost. `advance` re-runs it; do not launch on it |
| `calibration_schedule_current` red in `daily.update`'s batch summary (`--apply` refuses on it) | the weekly re-fit was missed: rows are being priced on frozen factors. `advance` re-fits and re-seals on its own rule (the schedule must reach the week being priced); run it, then `--apply`. `status` has no row for this: it is read at the operator gate. The backtest and shadow reports carry their own reading of the same thing, `artifact_versions.calibration_coverage` (`STALE FACTORS IN USE` when rows fell past the schedule's end) — a report figure, read in the report, never a `status` line |
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
