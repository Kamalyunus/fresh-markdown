# The pricing folder

Three commands and what they need. Every clock hour, `price_hour.py` prices
the shelf at the engine's optimal price. Every morning, `download_flc.py`
pulls the trailing days of the hourly table from the warehouse and
`build_features.py` turns that extract into the day's feature table the
hour joins on. This folder does not import from, read from or write to
anything outside itself but the warehouse, and it does nothing else: no
exploration, no learning, no model fitting, no checks. Those run in the
owner's repository; their results arrive here as files.

The full handover — the loop, who does what, every field, every rejection
reason — is `docs/engineering_handover.html`, in this folder. This page is
the short version for the person setting the host up.

## The layout

Every directory below exists in the folder as you receive it, each with a
`README.md` saying what lands there and who writes it.

```
integration/
  price_hour.py        every clock hour: the snapshot in, a price per shelf out
  download_flc.py      every morning, first: the trailing days of the hourly table, from the warehouse
  build_features.py    every morning, then: the day's extract in, the day's feature table out
  config.yaml          the owner's config pruned to what these commands read (synced)
  requirements.lock    the pinned libraries
  pricing/             the code behind the three commands
  artifacts/           IN, from the owner: the five model files the hour opens
  snapshots/           IN, from you, hourly: the shelf at the top of the hour
  data/                the day's extract, written by download_flc.py: the feed itself
  features/            the daily feature tables, written by build_features.py
  decisions/           OUT: the hour's response, what you push to the shelf
  reports/hours/       OUT: the hour's counts
  events_store/        OUT: the append-only log of decisions and refusals, for the owner
  logs/                the cron lines' console output
  examples/  docs/     the four example tables and the handover page
  MANIFEST.json        every file in the folder and where it came from
```

## The three commands

| When | Command | In | Out |
| --- | --- | --- | --- |
| Every clock hour | `python3 price_hour.py --snapshot snapshots/<day>T<hh>.parquet --workers 0 --out decisions/<day>T<hh>.csv --report reports/hours/<day>T<hh>.json` | the shelf at the top of the hour: the twelve request fields (below) | one row per snapshot row: the discount to apply as a percent and as a price, or `rejected` with the reason; the hour's counts |
| Every morning, first | `python3 download_flc.py` | the warehouse: the trailing 45 days of the hourly table through yesterday, `episode_id` on every row, `REDSHIFT_*` from `~/.env` | `data/flc.parquet`, the day's extract |
| Every morning, then | `python3 build_features.py` | the day's extract | `features/<today>.parquet`, the two demand-rate features every episode opening today reads |

Each command sets this folder as its working directory before it runs, so
call it from anywhere; a relative path in an argument is relative to this
folder. `--help` lists every flag; `--dry-run` on the hourly command writes
the response and the report and appends nothing to the log.

The hourly command exits 0 whenever it ran, even if every shelf came back
`rejected`: alert on the report's `rejected` count, not the exit code. It
exits non-zero only when it could not run at all — the console output
says which of the host, the config, the artifacts or the feature table is
missing or moved.

## The hourly service is stateless

Every row is priced from the row itself, the artifacts and the feature
table of its episode's opening day. Nothing is looked up from an earlier
hour, no store is read, and a re-sent hour gets the identical answer. The
snapshot is the twelve request fields of the handover's Appendix C, as you
send them: `episode_id, sku_id, fc, category, subcategory, date, hour_of_day,
hours_remaining` (this hour included), `q, original_price, cost,
current_discount` (a fraction). Nothing is renamed or converted on the way
in. Two things in the row make the statelessness possible, and both are
yours to supply:

- **`episode_id` is the opening tag** of the row's episode:
  `<skuseq>|<fc>|<day>T<hh>`, the shelf-hour the episode began. A row whose
  tag names its own shelf-hour is the episode's first hour (an entry: the
  price is set freely). Any later row of the episode carries the same tag
  and is a later hour (the price may only step deeper). An id that is not
  a tag, names another shelf, or opens after the row's hour is refused
  with the reason, and counted (`episode_ids_that_place_no_hour`).
- **`current_discount` is the price in force**, a fraction. Null on an
  entry row. On every later row of the episode it is the discount this
  service applied last hour (the response's `apply_discount_pct` divided
  by 100), piped back by you — the anchor the hour steps from. A later row
  without it is refused (`later_hours_without_the_price_in_force`), never
  priced as an entry.

The forecast is re-made every hour over the remaining hours, on the two
demand-rate features from `features/<opening day>.parquet`, so every hour
of an episode is forecast on what its first hour was. A day whose table is
not there reads the batch day's table and is listed in the report
(`opening_days_on_the_batch_day_table`); a day with neither refuses its
rows.

## The morning is a pull and a table

`download_flc.py` runs one SELECT over the trailing days of the hourly
table (its own columns, with your `episode_id` on every row) and writes `data/flc.parquet`. It reads
the `REDSHIFT_*` values from `~/.env` on the host; nothing in this folder
holds a credential or a hostname, and a missing variable is named in the
error. `pip install psycopg2-binary python-dotenv` on the host for it.

`build_features.py` reads that extract as it is: your ids are the
episodes, no rule re-derives them. It keeps a row only when it carries
its key, its id and its counts, drops the earlier copy of a re-fed hour,
and drops a row whose discount is outside 0..100, whose count is
negative, whose category is null or whose base price is not positive.
From what remains it writes the trailing 30-day anchor-hour sales rate
and the previous episode's rate per SKU × FC, plus one pooled row per
SKU, as of today. Nothing rolls over from yesterday: each morning starts
from the day's pull.

## What arrives, and from whom

Nothing in this folder fits a model or reads a seal. The owner's chain
does that in the repository and copies the result in on every seal and
every run of their driver. `artifacts/synced.json` names the bundle and
lists what arrived and when. You never copy a file by hand; if the folder
reaches you without one, ask the owner for a sync.

| Folder | What | Who puts it there |
| --- | --- | --- |
| `artifacts/` | the five files the hour opens: the model, its feature schema, the level factors, the dispersion table, the posterior | the owner's sync |
| `data/` | `flc.parquet`, the day's extract: the trailing 45 days of the hourly table -- the feed itself, pulled rather than delivered -- overwritten every morning (a trailing 30-day rate needs 30 days of rows, and the previous episode's rate reaches a little further) | `download_flc.py` |
| `config.yaml` | the owner's config pruned to the keys these two commands read, generated by the sync: the horizon cap, the category anchors, the five artifact paths, the table folder, the tier grid, the log folder. No switch: the folder prices whenever it is run | the owner's sync |
| `snapshots/` | the top-of-hour rows | you, hourly |
| `features/` | `<day>.parquet`, the day's feature table; the hour reads the table of each episode's opening day | the morning command |
| `decisions/`, `reports/hours/` | the response and the hour's counts | the hourly command |
| `events_store/` | the append-only log of every priced and every refused shelf-hour, one JSON line each; never read here, collected by the owner's learning lane. Back it up daily | the hourly command |
| `logs/` | stdout, collected by the cron lines | cron |
| `examples/`, `docs/` | the four example tables (the shape, the names and the units, nothing about production) and the handover page | the owner |
| `pricing/` | the code behind the three commands, written for this folder: fourteen small modules, the map at the top of `pricing/__init__.py`. The two feature parameters (the 30-day window, the anchor band) are constants in `pricing/features.py`, fixed with the model | the owner |

The code is this folder's own, not a copy of the repository's: it is the
hourly path and the morning table and nothing else, and the owner's suite
prices the same hours through both and refuses a difference in any field
of the decision.

The failed-pushes table is yours to deliver as before; it goes to the
owner's repository, where the morning lane reads it, not here. The owner
runs the table checks there on the files you send, the check of this
command's response against its snapshot included; send the first hour's
snapshot and response for that.

## This phase: exploit only

This folder prices exploit-only, by construction: `pricing/decide.py`
records the engine's optimal price as the applied price, nothing is
drawn, no budget is read, `is_exploration` is false on every row and the
hourly line ends in `EXPLOIT ONLY`. That is deliberate for the
integration phase: the loop is being tested, not the learner. Turning the
draw on is the owner's change to this folder, not a setting of yours.

## Setting the host up

1. Linux, Python 3.11. `pip install -r requirements.lock`, plus
   `psycopg2-binary` and `python-dotenv` for the pull; `~/.env` with the
   five `REDSHIFT_*` values.
2. Wait for the owner's sync: `artifacts/` holds the five files,
   `config.yaml` the pruned config.
3. Run `download_flc.py` once, then `build_features.py --as-of <today>`:
   the day's extract and today's table.
4. Dry-run one real hour with `--dry-run` and send the owner the
   snapshot, the response and the report.
5. Install the two cron lines.

## The cron lines

```
# hourly, a few minutes past the hour, once the snapshot is written
5 * * * *  TZ=<the feed's zone> flock -n /opt/pricing/.hourly.lock \
  python3 /opt/pricing/price_hour.py --snapshot snapshots/$(date +\%Y-\%m-\%dT\%H).parquet \
    --workers 0 --out decisions/$(date +\%Y-\%m-\%dT\%H).csv \
    --report reports/hours/$(date +\%Y-\%m-\%dT\%H).json \
    >> /opt/pricing/logs/hourly.log 2>&1

# every morning, once yesterday's rows are in the warehouse: the pull, then the table
30 6 * * *  TZ=<the feed's zone> flock -n /opt/pricing/.morning.lock \
  sh -c 'python3 /opt/pricing/download_flc.py && python3 /opt/pricing/build_features.py' \
    >> /opt/pricing/logs/morning.log 2>&1
```

`TZ` is the feed's own wall clock, so file names line up with the feed's
day. `flock -n` skips a run that would overlap the previous one; the log
also locks itself on every append, so two runs cannot interleave lines.

## What you send back

The hour's report JSON and the console output under `logs/`; before the
first cron fires, one `--dry-run` hour's snapshot, response and report.

---

*The owner maintains this folder: `MANIFEST.json` lists every file, the
owner's suite refuses a copy that fell behind its source and a decision
that differs from the repository's. Do not edit a file here; tell the
owner what is wrong.*
