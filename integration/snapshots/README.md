# snapshots/

`<day>T<hh>.parquet` (or `.csv`, `.jsonl`): the shelf at the top of the
hour, one row per SKU x FC on clearance, in the feed's schema with your
`episode_id` (the opening tag) and `discount` (the price in force: null on
an entry row, the price applied last hour on a later row). Written by you
before the hourly cron fires; read by `price_hour.py`. The shape is
`examples/snapshot_2026-08-29T11.csv`.
