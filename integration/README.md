# The pricing folder — standalone, for the pricing host

This folder is everything the hourly price needs and nothing else. It does
not import from, read from or write to anything outside itself. Copy it to
the host, put the owner's artifacts in `artifacts/`, and the cron lines
below run from here.

It is **generated** from the owner's repository by
`python3 -m tools.build_integration` (the exact import closure of the four
scripts below, copied verbatim; `MANIFEST.json` lists every file with its
digest). Do not edit a file in it: the fix goes into the repository and the
next build carries it here. A folder that differs from its build fails the
owner's test suite.

The full handover — the loop, who does what, every field, every rejection
reason — is `docs/engineering_handover.html` in this folder. This page is
the short version for the person setting the host up.

## What you run

| When | Command | In | Out |
| --- | --- | --- | --- |
| Every clock hour | `python3 -m ops.price_hour --snapshot snapshots/<day>T<hh>.parquet --features features/<day>.parquet --workers 0 --out decisions/<day>T<hh>.csv --report reports/hours/<day>T<hh>.json` | the shelf at the top of the hour, in the feed's schema, with `episode_id` | one row per snapshot row: the discount to apply as a percent and a price, or `rejected` with the reason |
| Every morning, after yesterday's feed lands | `python3 -m daily.features --feed feed/<yesterday>.parquet` | yesterday's hourly feed | `features/<today>.parquet`, the two demand-rate features every episode opening today reads |
| Before launch, and whenever a table's shape changes | `python3 -m ops.check_inputs --snapshot <file> --feed <file> --failures <file>` | any of your three tables | PASS/FAIL per check on stdout; exit 1 on a FAIL |
| Each hour, before the snapshot is written (or ported into your pipeline) | `python3 -m ops.assign_episode_ids --hour <this hour's rows> --previous <last hour's rows> --out <rows with episode_id>` | this hour's rows and last hour's | the rows with `episode_id`, by the rule the handover states |

Every command runs from this folder. Run from anywhere else and a script
reads the wrong `config.yaml` and the wrong `artifacts/`, silently.

The hourly script exits 0 whenever it ran, even if every shelf came back
`rejected`: alert on the report's `rejected` count, not the exit code. It
exits non-zero only when it could not run at all — the console output says
which of the host, the config or the artifacts moved.

## Setting the host up

1. Linux, Python 3.11. `pip install -r requirements.lock` — the exact
   library versions the owner's sealed bundle was verified with.
2. Put the owner's artifacts in `artifacts/`, whole: `bundle.json`,
   `baseline_model.txt`, `feature_schema.json`, `calibration.json`,
   `r_lookup.json`, `rho.json`, `prior.json`, `posterior.json`,
   `split_manifest.json`, and the `history/` directory. They come from
   the owner's pre-launch run, never from here. The hourly script refuses
   to start when the config disagrees with them, on purpose.
3. Put the owner's extract at `data/flc_raw.parquet`. The first morning's
   feature run seeds the rolling feed history from it
   (`data/feed_history.parquet`, kept at `features.history_days` days
   from then on).
4. `config.yaml` is the owner's, with their readings in it. The one value
   they set on launch day is `data.launch_date`; until it is set the hourly
   script refuses to start, which is the intended pre-launch state.
5. Check your tables against the examples' shape:
   `python3 -m ops.check_inputs --snapshot examples/snapshot_2026-08-29T11.csv --feed examples/feed_2026-08-29.parquet --failures examples/failures_2026-08-29.csv`
   then the same on your own files. The examples were written against a
   synthetic shop: the shape, the names and the units, nothing about
   production.
6. Dry-run one real hour: add `--dry-run` to the hourly command. It prices
   against a scratch copy of the store and commits nothing.

## The folders

| Folder | Who writes | What |
| --- | --- | --- |
| `snapshots/` | you, hourly | the top-of-hour rows, `<day>T<hh>.parquet` (CSV and JSONL are read too) |
| `feed/` | you, nightly | the day's hourly feed, `<day>.parquet`, every shelf-hour, `episode_id` carried through |
| `failures/` | you, nightly | one row per hour whose returned price did not reach the shelf; a header alone is a valid file |
| `features/` | `daily.features` | `<day>.parquet`, the day's feature table the hourly script joins on |
| `decisions/`, `reports/hours/` | `ops.price_hour` | the response and the hour's counts |
| `events_store/` | `ops.price_hour` | the decision and rejection streams, append-only, locked on every write. This is the pilot's record; back it up daily |
| `artifacts/` | the owner | the sealed bundle above; `posterior.json` is the one file in it that moves after launch, and only the owner's daily lane moves it |
| `data/` | the owner, then `daily.features` | the extract, then the rolling feed history |
| `logs/` | cron | stdout of both scripts, collected by the cron lines |

## The cron lines

```
# hourly, a few minutes past the hour, once the snapshot is written
5 * * * *  cd /opt/pricing && TZ=<the feed's zone> flock -n /opt/pricing/.hourly.lock \
  python3 -m ops.price_hour --snapshot snapshots/$(date +\%Y-\%m-\%dT\%H).parquet \
    --features features/$(date +\%Y-\%m-\%d).parquet --workers 0 \
    --out decisions/$(date +\%Y-\%m-\%dT\%H).csv --report reports/hours/$(date +\%Y-\%m-\%dT\%H).json \
    >> logs/hourly.log 2>&1

# every morning, after yesterday's feed has landed
30 6 * * *  cd /opt/pricing && TZ=<the feed's zone> flock -n /opt/pricing/.morning.lock \
  python3 -m daily.features --feed feed/$(date -d yesterday +\%Y-\%m-\%d).parquet \
    >> logs/morning.log 2>&1
```

`TZ` is the feed's own wall clock, so file names line up with the feed's
day. `flock -n` skips a run that would overlap the previous one; the event
store also locks itself on every write, so an overlap that slips through
is refused rather than double-priced.

## What is not here, deliberately

The learning lane — building outcomes from your feed, the posterior update,
the monitor, the assurance checks, the nightly export — runs in the owner's
repository, not in this folder. It reads `events_store/` and `feed/` from
here and writes `artifacts/posterior.json` back. How those three move
between the two is agreed with the owner; nothing in this folder assumes
an answer. This folder prices.

## What you send back

The hour's report JSON, the console output under `logs/`, and — before
the first cron fires — the output of the `check_inputs` run on your own
three tables and of one `--dry-run` hour. The handover page's Phase 0
lists each item and what it proves.
