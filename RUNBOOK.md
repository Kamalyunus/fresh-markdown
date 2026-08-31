# Runbook — operating the Perishable Markdown MVP

For the engineering team running this system. Three lanes, each with its own
cadence and its own definition of done. The authoritative spec is
`docs/design.md`; the integration contract is `docs/event_contract.html`;
`REVIEW_GUIDE.md` maps the code by risk tier. Nothing in this document
requires reading the spec first.

All commands run from the repo root. `data/`, `reports/`, `artifacts/`,
`events_store*/` are run outputs and never committed. Credentials
(`REDSHIFT_*`) live in `~/.env`, never in config or code.

---

## Lane A — Train & freeze (at launch; on each retrain)

Produces the sealed artifact bundle every price stands on. Runs offline; no
price is touched until Lane B. One step is a HUMAN GATE (the prior) and one
is an owner review (the level diagnostic) — an engineer runs the commands,
the owner reads the verdicts.

```bash
pip install -r requirements.txt
python3 -m bootstrap.download_flc --days 120          # -> data/flc_raw.parquet
python3 -m bootstrap.run --input data/flc_raw.parquet
# trains ONCE, iterates calibration -> prior -> dispersion to CONVERGED,
# then backtest, thresholds, seal, status. Production settles in ~8-9 turns.
```

Then, in order:

1. **Level diagnostic (review, not a gate).** Calibration is always
   fitted and applied (the script's step 3b). Read `reports/backtest.json`
   → `calibration_gate_value` against the band `[0.90, 1.10]`: out of band
   is a drift/staleness reading — follow the decision tree in `docs/design.md` §9.2
   to separate wobble from trend — surfaced as WARN in `status`, never a
   launch blocker.
2. **GATE — prior (owner).** Read `artifacts/prior.json` in this order:
   `wrong_sign_categories` → per-category
   `mean/std/std_basis` → `holdout_comparison` (read
   `information_available_per_row` first). There is no pass flag; a pooled
   or uniform prior is a designed outcome, not a failure.
3. **Paste MEASURED values into `config.yaml`** — the only hand step:
   - `dispersion.rho`, `dispersion.mean_forced_hours_per_episode` — from
     `artifacts/rho.json`, after **every** retrain;
   - `learning.information_increment` — from `reports/thresholds.json` →
     `information_increment_recommendation.recommended`, re-derived after
     **every** prior change (it is the ceiling above which a bounded update
     discards the evidence it waited for; the report stamps `TOO LARGE`);
   - `exploration.tau_initial` — from `reports/shadow.json` →
     `tau_initial_derivation.tau_initial` (shadow derives it itself on the
     trailing pre-window week, so this paste happens AFTER step 6 and feeds
     the pilot, not the shadow run; a stale or mismatched paste is refused,
     by design).
4. **Owner sets the `SET BY OWNER` keys** (`scrap_deterioration_pct`,
   `margin_deterioration_pct`, `min_detectable_effect_pct`) from
   `reports/thresholds.json` — never invented, and never below a floor the
   report stamps `TOO TIGHT`. The two learning rails
   (`learning.max_std_shrink`, then `learning.max_mean_step`) are owner calls
   too: read `bounded_step_recommendation` for which one binds first, and
   `backtest.step_sensitivity` for what a mean step does to prices. Set
   `max_std_shrink` first — `information_increment` is derived from it.
5. `python3 -m bootstrap.init_posterior` (once, at launch — refuses to
   overwrite production learning state without `--force`).
6. **Shadow, on the hold-out** (default window):
   `python3 -m pipeline.shadow --input data/prepared.parquet --max-episodes 0`
   for the launch record. It derives its own launch tau on the trailing
   pre-window week (`tau_initial_derivation` in the report — this is the
   value to paste in step 3 for the pilot). Exit gate: completeness ≥ 99%,
   matched ≥ 99%, **zero** cost-floor violations. Also read
   `exploration_budget.spend_over_budget` (over 2× → do not launch at this
   tau; re-derive) and `tau_controller_trace` — day one is an out-of-sample
   test of the derived tau.
7. `python3 -m pipeline.status` — **done means every line green.** Exit code
   1 on any FAIL, so it can gate a deploy.

**Never** retrain between two runs you intend to compare; comparisons are
valid only when `baseline_model_version` matches. **Never** tune anything on
the hold-out window — it is a one-shot resource.

---

## Lane B — Price (hourly; the only lane where engineering builds code)

The engine is `inference.decide`: state in, price + decision event out, or
`StateRejected` — it never returns a best-effort price for a state it cannot
validate. Engineering owns everything on the other side of the event
contract:

- the hourly scheduler and transport that call `decide` per SKU × FC
  (the 12-field request in `docs/event_contract.html` §02);
- applying the returned price (the applied price must be the returned one —
  the mismatch rate is gated at 1%);
- reporting **failed price pushes** — one row per failed hour, as a table
  (parquet/CSV) or JSONL (`sku_id`, `fc`, `date`, `hour_of_day`, `reason`);
  no row means the push succeeded;
- a defined fallback for `StateRejected` (hold the current price; alert on
  rate).

Outcomes are NOT engineering's to produce: `pipeline.ingest_outcomes`
(Lane C) builds them from the hourly FLC feed, matched to decisions by
(SKU, FC, date, hour), deriving `adjustment_reason`, `is_stockout` and the
offered price itself. §07 of the contract page is the pre-build
feasibility checklist, and §08 is the conventions register: every
derivation stands on source-data meanings only engineering can confirm —
walk them through it before the pilot.

The caller reads `tau` from `PosteriorStore.tau(cfg)` — **not** from
`config.exploration.tau_initial`, which is only the launch value and never
moves. Safety properties (cost floor, price monotonicity) are structural in
the engine's action set: there is nothing to configure and nothing to check
in the caller.

---

## Lane C — Learn & watch (daily: one cron, one human decision)

Run after midnight, in this order:

```bash
python3 -m pipeline.ingest_outcomes --feed <hourly parquet> \
    [--failures failures.parquet]      # yesterday's table rows -> outcomes;
                                       # idempotent, failures table optional
python3 -m pipeline.update             # monitor only -- always safe
python3 -m pipeline.update --apply     # OPERATOR GATE -- see below
python3 -m pipeline.monitor            # business / learning / safety series
python3 -m pipeline.assurance          # the frozen artifacts vs the live world
python3 -m pipeline.status             # the only screen that must be read daily
python3 -m pipeline.export_events      # decision/outcome tables for the
                                       # warehouse -- derived, never the record
```

**Weekly, in the same lane — re-fit the level factors.** The calibration
schedule only covers weeks it was fitted on, and a row past its end falls
back to the frozen factors **silently**. That is the staleness the point-in-
time change exists to remove, so the re-fit is an operational step, not a
launch-time one:

```bash
python3 -m bootstrap.train_baseline --input data/prepared.parquet --fit-calibration
python3 -m bootstrap.seal        # calibration.json changed -- re-seal the bundle
```

- **This is NOT a retrain.** The model, `r`, `rho` and the prior do not move,
  so before/after comparisons stay valid under hard rule 1 — but say which
  factor vintage a report ran under, because the factors did move.
- Read `artifact_versions.calibration_coverage` in the next backtest or
  shadow report: `rows_on_fallback` should be 0 and the verdict `OK`.
  `STALE FACTORS IN USE` means the schedule ran out and the run was priced on
  frozen factors.
- If `calibration_window_sweep` starts preferring a different trailing
  window, that is an owner decision — re-read it, do not drift the config.

**The `--apply` gate.** One human approves at most one posterior step per
cell per day. Before approving, read each cell's block:

| Field | Approve when | Hold when |
| --- | --- | --- |
| `predictive_check` | `worse_than_a_flat_prior: false`, or a one-off | it persists across batches — the belief tightened faster than the evidence; escalate before more updates |
| `bound_clipped` | occasional | most updates clip — step cap or increment mis-sized; escalate |
| `batch_oldest_outcome_age_days` | near the expected cadence | growing without a trigger — the loop is stalling; check volumes and tau |
| event-quality gates | green (the command refuses on red) | never work around a refusal |
| `calibration_schedule_current` | green | red means the weekly re-fit above was missed — `--apply` refuses, because learning from prices set on stale factors banks evidence about a model that is not the one running |

`tau` recalibrates on the same `--apply`, on spend rather than evidence,
exactly once per day — a second run in the same day is a no-op, not a bug.

**Red-line table** — what a red `status` line means and the response:

| Line | Response |
| --- | --- |
| stop condition fired (overspend >2×, mismatch, duplicates, missing stockout field) | exploration suspends for the cohort automatically; **exploitation pricing continues**. Investigate, don't restart blindly |
| `assurance · reproduction` FAIL | something moved under the solver (config edit, artifact swap, deploy, library). Diff the bundle first: `artifact bundle` line, then `artifact mirrors` |
| `artifact mirrors` FAIL | config paste and its source disagree (rho, forced hours). Read the **bundle** line before re-pasting — the stale side is not always config |
| `report vintages` FAIL | backtest/shadow report was produced against a model no longer on disk — its gate rows grade a ghost. Re-run that report; do not launch on it |
| `calibration_coverage` says `STALE FACTORS IN USE` | the weekly re-fit was missed: the schedule ran out and rows were priced on frozen factors. Re-run `--fit-calibration`, re-seal, re-run the report |
| posterior std flat ≥ alert days | the loop is dead: no committed update. Check batch age, tau, volumes — in that order |
| guardrail breach (scrap/margin, 2 consecutive days) | business decision, not a code fix — escalate to the owner with the monitor's arm comparison |
| `INSUFFICIENT` verdicts | not a pass. A thin window said so; widen or wait |

**Never** hand-edit `artifacts/posterior.json`, re-derive filter logic
outside `prepare_data.population`, or drive a quarantine count to zero by
adding a catch-all reason — the quarantine file is where uninterpretable
outcomes stay visible.

---

## RACI

| Decision / step | Engineering | Product owner |
| --- | --- | --- |
| Run Lane A commands, CI, deploys | **R/A** | — |
| Prior gate verdict; level-diagnostic review | run & present | **A** |
| MEASURED pastes into config | **R** (from named report fields only) | informed |
| `SET BY OWNER` thresholds, MDE, gate band | — | **A** |
| Lane B service, push-failure feed, SLA | **R/A** | — |
| Daily `--apply` approval | **R** (pilot: owner may retain) | consulted on escalations |
| Guardrail breach response, A/B readout | informed | **A** (decision table in design §11.2) |
| A/B duration & no-early-reads | enforce | **A** |


