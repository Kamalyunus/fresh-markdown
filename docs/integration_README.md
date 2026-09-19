# The pricing folder

Everything the hourly price needs, and nothing else. This folder does not
import from, read from or write to anything outside itself. Put it on the
host, install the pinned libraries, and the two cron lines below run from
it. It prices; it does not learn (see *What is not here*).

The full handover — the loop, who does what, every field, every rejection
reason — is `docs/engineering_handover.html`, in this folder. This page is
the short version for the person setting the host up.

## Three commands

| When | Command | In | Out |
| --- | --- | --- | --- |
| Every clock hour | `python3 price_hour.py --snapshot snapshots/<day>T<hh>.parquet --features features/<day>.parquet --workers 0 --out decisions/<day>T<hh>.csv --report reports/hours/<day>T<hh>.json` | the shelf at the top of the hour in the feed's schema, with the `episode_id` you assign | one row per snapshot row: the discount to apply as a percent and as a price, or `rejected` with the reason; the report JSON with the hour's counts |
| Every morning, after yesterday's feed lands | `python3 build_features.py --feed feed/<yesterday>.parquet` | yesterday's hourly feed | `features/<today>.parquet`, the two demand-rate features every episode opening today reads |
| Before launch, whenever a table's shape changes, and on any hour that looks wrong | `python3 check_inputs.py [--snapshot <file>] [--feed <file>] [--failures <file>] [--response <file>]` | any of your three tables; the hour's response, with its snapshot | PASS / WARN / FAIL per check, the count behind each and what to fix; exit 1 on any FAIL |

Each command sets this folder as its working directory before it runs, so
call it from anywhere; a relative path in an argument is relative to this
folder. `--help` on each lists every flag. The code behind them is in
`src/`; the command files are the only place to start reading.

The hourly command exits 0 whenever it ran, even if every shelf came back
`rejected`: alert on the report's `rejected` count, not the exit code. It
exits non-zero only when it could not run at all — the console output
says which of the host, the config or the artifacts moved.

## Setting the host up

1. Linux, Python 3.11. `pip install -r requirements.lock` — the exact
   library versions the owner's sealed bundle was verified with.
2. `artifacts/` and `data/` are filled by the owner's own chain: every
   time the owner seals a bundle or runs their driver, the sealed
   artifacts, the posterior, the extract and the config land here
   (`artifacts/synced.json` names the bundle and lists what arrived and
   when). You never copy an artifact by hand, and the hourly command
   refuses to start when the config disagrees with them, on purpose. If
   the folder reaches you without them, ask the owner for a sync.
3. `config.yaml` is the owner's, with their readings in it. The one value
   they set on launch day is `data.launch_date`; until it is set the hourly
   command refuses to start, which is the intended pre-launch state.
4. Check your tables against the examples' shape, then your own files:

   ```
   python3 check_inputs.py --snapshot examples/snapshot_2026-08-29T11.csv \
       --feed examples/feed_2026-08-29.parquet --failures examples/failures_2026-08-29.csv
   python3 check_inputs.py --response examples/decisions_2026-08-29T11.csv \
       --snapshot examples/snapshot_2026-08-29T11.csv
   ```

   The examples were written against a synthetic shop: the shape, the
   names and the units, nothing about production.
5. Dry-run one real hour: add `--dry-run` to the hourly command. It prices
   against a scratch copy of the store and commits nothing. Then run the
   response check on what it wrote, with the snapshot it read.

## The episode id is yours

Every snapshot row carries `episode_id`, the name of the listing the row
belongs to, assigned by you before the row reaches this folder. The rule
for it is in the handover page (a row continues the shelf's previous
episode when last hour's row is exactly one hour earlier, did not close
the shelf, and the counter stepped as stock allows; otherwise it opens a
new one). Nothing here assigns an id. The hourly command reads yours as
given — a new id on a shelf is an entry, the id it last priced on that
shelf continues the episode — and only *counts* the ids that disagree
with the rule (`episode_ids_disagreeing_with_the_rule` in the hour's
report, with a sample). A row without an id is not priced.

## The folders

| Folder | Who writes | What |
| --- | --- | --- |
| `snapshots/` | you, hourly | the top-of-hour rows, `<day>T<hh>.parquet` (CSV and JSONL are read too) |
| `feed/` | you, nightly | the day's hourly feed, `<day>.parquet`, every shelf-hour, `episode_id` carried through |
| `failures/` | you, nightly | one row per hour whose returned price did not reach the shelf; a header alone is a valid file |
| `features/` | `build_features.py` | `<day>.parquet`, the day's feature table the hourly command joins on |
| `decisions/`, `reports/hours/` | `price_hour.py` | the response and the hour's counts |
| `events_store/` | `price_hour.py` | the decision and rejection streams, append-only, locked on every write: the pilot's record. Back it up daily |
| `artifacts/` | the owner's chain | the sealed bundle, the prior and the posterior; `synced.json` says which bundle and when |
| `data/` | the owner's chain, then `build_features.py` | the extract, then the rolling feed history the feature table is built from |
| `logs/` | cron | stdout of both commands, collected by the cron lines |
| `examples/`, `docs/` | the owner | the four example tables; the handover page |
| `src/` | the owner | the engine. Generated from the owner's repository; a change goes there and the next build carries it here |

## The cron lines

```
# hourly, a few minutes past the hour, once the snapshot is written
5 * * * *  TZ=<the feed's zone> flock -n /opt/pricing/.hourly.lock \
  python3 /opt/pricing/price_hour.py --snapshot snapshots/$(date +\%Y-\%m-\%dT\%H).parquet \
    --features features/$(date +\%Y-\%m-\%d).parquet --workers 0 \
    --out decisions/$(date +\%Y-\%m-\%dT\%H).csv --report reports/hours/$(date +\%Y-\%m-\%dT\%H).json \
    >> /opt/pricing/logs/hourly.log 2>&1

# every morning, after yesterday's feed has landed
30 6 * * *  TZ=<the feed's zone> flock -n /opt/pricing/.morning.lock \
  python3 /opt/pricing/build_features.py --feed feed/$(date -d yesterday +\%Y-\%m-\%d).parquet \
    >> /opt/pricing/logs/morning.log 2>&1
```

`TZ` is the feed's own wall clock, so file names line up with the feed's
day. `flock -n` skips a run that would overlap the previous one; the event
store also locks itself on every write, so an overlap that slips through
is refused rather than double-priced.

## What is not here, deliberately

The learning lane — building outcomes from your feed, the posterior
update, the monitor, the assurance checks, the nightly export — runs in
the owner's repository. It reads `events_store/` and `feed/` from here and
its result comes back as the next sync into `artifacts/`. How the store
and the feed reach the owner is agreed with them; nothing in this folder
assumes an answer.

## What you send back

The hour's report JSON, the console output under `logs/`, and — before
the first cron fires — the output of `check_inputs.py` on your own three
tables, of one `--dry-run` hour, and of the response check on its output.
The handover page's Phase 0 lists each item and what it proves.

---

*This folder is generated by the owner's `python3 -m ops.integration` and
never edited in place; `MANIFEST.json` lists every generated file with its
digest.*
