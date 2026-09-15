# Engineering handover — Lane B and the daily cron

This is the one page for the engineering team. It says what you receive,
what you build, the constraints the engine will hold you to, how to prove
the integration end to end before any of your code exists, and who owns
what. The binding integration contract is `docs/event_contract.html`
(field by field, with a worked episode); `RUNBOOK.md` is the operator's
document. Where this page and the contract disagree, the contract wins.

## The big picture

The markdown engine is a black box to you. Once per clock hour you send
one price request per open clearance episode (SKU × FC), get back a
price or a row-scoped rejection with a reason, put exactly that price on
the shelf, and report the pushes that failed. The engine forecasts,
optimises, explores, learns and monitors itself. You never produce
outcomes or forecasts: outcomes are built by the engine from the hourly
FLC feed you already produce.

## What you receive from us

| What | It is | How it reaches you |
| --- | --- | --- |
| `docs/event_contract.html` | The contract: request fields (§03), the decision the engine logs (§04), the outcome it builds from your feed (§05), one worked episode (§06), the gates the events feed (§07), the checklist (§08) | The HTML file is self-contained; share it |
| `ops.price_batch` | The reference caller: one hour's requests in, one response row per request out. Run it as a file drop per hour, or lift `engine.state.build_states` into your service | In the repo |
| `tools.e2e_cycle` | A runnable rehearsal of the whole cycle (requests → decisions → shop feed → ingest → exports) against the real artifacts, in a throwaway workspace | In the repo |
| `artifacts/` | The sealed bundle the engine prices with: model, calibration, prior, dispersion lookup, posterior, thresholds, `bundle.json` | Deployed beside the engine, never edited by hand |
| `config.yaml` | Every tunable the engine reads (tier step, entry offsets, `max_window_hours`, the event store path, …) | Deployed beside the engine; the owner's file |
| the event store (`events.store_dir` in config) | The decision and outcome log the engine reads and writes: one decision per hour, one outcome per decision, duplicates refused and counted | Shared state on disk (or its equivalent) reachable by the hourly caller and the daily cron |
| `data.launch_date` in config | Null until the owner sets it on launch day | The owner sets it |

## What you build

### 1. The hourly caller

One batch per clock hour, one request per open episode, in the twelve
fields of contract §03, in this order:

```
episode_id, sku_id, fc, category, subcategory,
date, hour_of_day, hours_remaining, q,
original_price, cost, current_discount
```

The reference caller:

```
python3 -m ops.price_batch --requests <hour>.jsonl|.parquet|.csv \
    --history <trailing hourly FLC table> \
    --out <hour>.response.jsonl --report <hour>.report.json
```

Every request is read in one spelling before anything else: ids as text
(a `7`, a `"7"` and a `7.0` are one item), the day as `YYYY-MM-DD` (a
parquet timestamp is fine), counts as integers. The response has one row
per request, in the request's order:

```
{"episode_id": ..., "sku_id": ..., "fc": ..., "date": ..., "hour_of_day": ...,
 "decision_id": "<id>", "applied_discount": 0.15, "applied_price": 8500.0,
 "is_exploration": false, "rejected": null}
```

or, for a request the engine will not price:

```
{..., "decision_id": null, "applied_discount": null, "applied_price": null,
 "is_exploration": null, "rejected": "<reason>"}
```

A rejection is always row-scoped: the rest of the batch prices. Reasons
you will see while integrating: a missing or null field, `date` that
names no day, `hours_remaining` outside `1..max_window_hours`, a price or
cost that is not a number, `duplicate_request` (two rows for one hour in
one batch), `already_priced` (the store already holds that hour's
decision — a resend is refused, never re-priced), and the engine's own
economic refusals (a cost above the price, an anchor the action set
cannot honour).

The batch report (`--report`) is your integration dashboard. Read these
after every early batch:

| Key | Meaning | Healthy |
| --- | --- | --- |
| `rejected_before_the_engine`, `rejected_by_the_engine` | Rows refused, by reason | Explained per reason |
| `requests_with_unknown_features` | Entry requests whose SKU has no trailing history behind them, so the forecast starts from "unknown" | A few new SKUs; a whole batch means the history table or the id spelling does not meet the requests |
| `non_entry_requests_without_stored_path` | Later hours of episodes the store never priced at entry | Zero once you are live; non-zero means an episode's entry hour was skipped or its id changed mid-window |
| `exploration_suspended`, `tau_in_force` | Whether the monitor has suspended exploration (exploitation pricing continues) | Read, do not act |

### 2. Apply the returned price

`applied_price` is what goes on the shelf, exactly that number. The
engine matches its decision against the price your feed shows in force;
the mismatch rate is gated (contract §07) and a breach suspends
exploration. A price you could not apply is a failed push (next item),
never a silent mismatch.

### 3. Report failed price pushes

The one outcome fact only you know. One row per failed hour, as a table
(parquet or CSV) or JSONL:

```
sku_id, fc, date, hour_of_day, reason
```

No row means the push succeeded. Keys are normalised on read the same way
requests are. A reported failure is counted apart
(`push_failures_applied` in the ingest report) and excluded from learning; an unreported one
is caught by the mismatch gate as if it were your bug.

### 4. A fallback for a rejected request

When a row comes back rejected, hold the current shelf price and alert on
the rejection rate. Do not retry the same hour with altered fields to get
a price out; the next hour's request is the retry.

### 5. The daily cron

Once per day, after the previous day's hourly feed has landed:

```
python3 -m ops.advance --feed <yesterday's hourly FLC parquet> --failures <yesterday's failed pushes>
```

Omit `--failures` on a day with none. One command runs the lane in order:
ingest outcomes (from the feed, matched to decisions on
`sku_id, fc, date, hour_of_day`), walk the exploration budget, monitor
(guardrails and stop conditions), assurance, export the paired tables,
status. It stops at `daily.update --apply`, the operator's gate, and
never runs it. It keeps running the lane while status is red; a red stop
names the row and what clears it.

## Constraints the engine holds you to

- **`episode_id` is stable across midnight.** An episode is one clearance
  window of one SKU at one FC, from the hour it opens to the hour the
  listing closes; windows cross midnight. Never derive the id from the
  calendar date: a date-keyed id splits one window in two, the second
  half arrives with `current_discount = null`, the engine treats it as a
  fresh entry, and the price can move back up on shoppers who saw
  yesterday's — the one move the system guarantees never happens inside a
  window. The full assignment rule is the box under contract §03.
- **A restock keeps the id.** Stock arriving mid-window, even one that
  extends the window so `hours_remaining` jumps up, is the same listing:
  send the new, larger counter and the engine re-plans over the longer
  horizon. The counter moves from the hour after the arrival.
- **A sell-out closes the window.** When the shelf empties, that hour is
  the last hour of the episode; if stock later arrives under the same
  listing, those hours are a new episode with a new id and a fresh entry
  (`current_discount = null`). Rows that continue an emptied window
  under the old id are flagged and not priced.
- **`hours_remaining` means hours.** Hours left in the window, this one
  included, at least 1 and at most `data.max_window_hours` (config); a
  larger value is rejected as a feed defect. Within one id it decrements
  by exactly one per hour, with the one restock exception above. Build
  that consistency check into your side: any other jump means the id is
  stitching two windows together or a feed hour is missing.
- **`current_discount`** is `null` on the first hour of an episode and a
  fraction (`0.15`, not `15`) on every later one. Prices only hold or
  deepen within an episode; the engine's action set enforces it.
- **`cost` and `original_price`** are per-request inputs, per hour, per
  FC. Cost sets the price floor; a stale or missing cost either blocks
  legal discounts or rejects the row.
- **`--history`** is the trailing hourly FLC table the two demand-rate
  features read (the prepared extract plus every hour fed since); it
  must reach back at least `data.ref_rate_window_days` (config) before
  the batch's day, in the same id spelling as the requests.
- **The feed keeps every priced hour**, zero-sale hours included.
  Outcomes are built from the feed, so a priced hour the feed drops is
  a completeness miss (contract §07), and the zero-sale rows are most of
  the evidence.
- **One request per hour per episode.** A second request for a priced
  hour is refused (`already_priced`); two rows for one hour in one batch
  are both refused (`duplicate_request`).

## What you do not build

- **Outcomes.** `daily.ingest_outcomes` builds them from the feed and
  matches them to decisions on `(sku_id, fc, date, hour_of_day)`; the
  outcome id is that key. You keep the feed landing.
- **Forecasts.** The engine resolves demand from the frozen model; a
  later hour of an episode is priced from the entry decision's stored
  forecast path, so nothing is recomputed on your side.
- **Monitoring and guardrails.** The monitor, assurance and the stop
  conditions run inside `advance --feed`.

## How to prove the integration end to end

Nothing below needs your code. It needs the sealed bundle in
`artifacts/` and a prepared extract under `data/`; the owner's agent
produces both (`ops.bootstrap_loop`, then `ops.init_posterior`).

**Step 1 — watch one cycle.**

```
python3 -m tools.e2e_cycle --input data/prepared.parquet --episodes 50 --hours 6 --workers 0
```

This wipes and rebuilds `sim/e2e/` and writes every file you will handle
in production:

| File under `sim/e2e/` | What it represents |
| --- | --- |
| `requests/<date>T<hh>.jsonl` | What you send, one file per hour, twelve fields |
| `decisions/<date>T<hh>.jsonl` | What you get back, one response row per request |
| `feed/<date>.parquet` | The hourly FLC feed you already produce, in its source column names |
| `exports/decisions.parquet`, `exports/outcomes.parquet` | The paired tables the daily lane exports for the warehouse |
| `e2e_report.json` | The cycle's counts |

**Step 2 — read the counts.** In `e2e_report.json` and the printed
summary, these must be zero: `price_mismatches`,
`decisions_colliding_on_hour`, `outcomes_per_decision_over_one`,
`missing_stockout_field`. `requests_with_unknown_features` and
`non_entry_requests_without_stored_path` should be zero here too (the
harness prices every entry hour and feeds the history it built); a
non-zero count in your own integration is a plumbing fault — an id
spelling, a history table cut short, an entry hour skipped. Every
rejection is listed with its reason.

**Step 3 — price a batch from real data.** Build one hour's requests from
the prepared extract, one row per open episode, and write it twice: as
JSONL and as parquet with a datetime `date` column. Price each with the
full source table as history, then send the JSONL batch a second time:

```
python3 -m ops.price_batch --requests <file> --history data/flc_raw.parquet \
    --out sim/batch/<name>.jsonl --report sim/batch/<name>.json --workers 0
```

Check: ids echo the request ids; every priced row sits on the discount
grid (`pricing.tier_step` in config); JSONL and parquet price
identically; the resend returns `already_priced` on every row;
unknown-feature requests are few and are genuinely new SKUs. Steps 1 and
3 write to the event store the config names, so clear that store before
launch.

**Step 4 — the checklist.** Contract §08, in the order things go wrong:
have you watched one cycle run; is `episode_id` stable across midnight;
does `hours_remaining` mean hours; does the feed keep every priced hour,
zero-sale hours included; can you report a failed push; are `cost` and
`original_price` available per hour, per FC.

## Operational rules

- **Reload the posterior once per batch** if you lift the engine into a
  long-lived service (`PosteriorStore.reload()` before pricing). The
  monitor writes suspensions and `--apply` writes updates into the same
  file from other processes; a handle that never reloads keeps exploring
  on a suspended pilot. `ops.price_batch` does this for you.
- **The exploration budget comes from the store** (`PosteriorStore.tau`),
  never from `exploration.tau_initial` in config, which is only the
  launch value.
- **Never run `daily.update --apply`.** It is the operator's gate.
- **Safety is structural**, not configured on your side: the cost floor
  and the never-rises-within-a-window rule live in the engine's action
  set. There is nothing for the caller to check or enforce.
- **The bundle is sealed.** Do not edit `artifacts/` or `config.yaml`; a
  moved file shows as a red seal row in `ops.status` and the lane stops.

## Who owns what

| Responsibility | Owner |
| --- | --- |
| Artifacts, model, config, the prior gate, `data.launch_date` | The product owner |
| Hourly caller, price application, failed-push reporting, the daily cron, the feed | Engineering |
| Outcomes, forecasts, learning, monitoring, guardrails, stop conditions | The engine |
