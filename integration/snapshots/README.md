# snapshots/

`<day>T<hh>.parquet` (or `.csv`, `.jsonl`): the shelf at the top of the
hour, one row per SKU x FC on clearance, written by you before the hourly
cron fires and read by `price_hour.py`. One spelling per file, either:

- the hourly table's columns: `skuseq, fc, date, hour, episode_id,
  inventory, discount` (percent), `normal_asp, cogs_wo_vat, flc_window`
  (hours still to come after this one), `category, subcategory` -- the
  shape of `examples/snapshot_2026-08-29T11.csv`; or
- the request's twelve fields (handover Appendix C): `episode_id, sku_id,
  fc, category, subcategory, date, hour_of_day, hours_remaining` (this hour
  included), `q, original_price, cost, current_discount` (a fraction).

In both, `episode_id` is the opening tag `<skuseq>|<fc>|<day>T<hh>` and the
price in force is null on an entry row and last hour's applied discount on
every later row.
