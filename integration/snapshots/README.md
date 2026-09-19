# snapshots/

`<day>T<hh>.parquet` (or `.csv`, `.jsonl`): the shelf at the top of the
hour, one row per SKU x FC on clearance, written by you before the hourly
cron fires and read by `price_hour.py`. The twelve request fields of the
handover's Appendix C, as sent: `episode_id, sku_id, fc, category,
subcategory, date, hour_of_day, hours_remaining` (this hour included),
`q, original_price, cost, current_discount` (a fraction). `episode_id` is
the opening tag `<sku_id>|<fc>|<day>T<hh>`; `current_discount` is null on an
entry row and last hour's applied discount (the response's
`apply_discount_pct` / 100) on every later row. The shape is
`examples/snapshot_2026-08-29T11.csv`.
