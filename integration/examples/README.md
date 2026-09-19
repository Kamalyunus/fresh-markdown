# Example files — the shape of the hourly interface

Written by the owner's rehearsal against a **synthetic** shop, so every id,
price and count here is a fixture's. They show the shape, the column names,
the units and the ids; they say nothing about production.

| File | Who writes it | What it is |
| --- | --- | --- |
| `snapshot_2026-08-29T11.csv` | you, every hour | The shelf at the top of hour 11: the twelve request fields (handover Appendix C). `episode_id` is the opening tag; `hours_remaining` counts this hour; `q` is the opening stock; `current_discount` is a fraction, null on an entry row and last hour's applied discount on a later one. |
| `decisions_2026-08-29T11.csv` | `price_hour.py`, every hour | One row per snapshot row: `sku_id, fc, date, hour_of_day` and the id echoed, `decision_id` (`dec-<sku>\|<fc>\|<date>T<hh>`), the discount to apply as a percent and as a price, or `rejected` with the reason. |
| `feed_2026-08-29.parquet` | the warehouse | A day of the hourly table in its own columns (`skuseq, hour, inventory, discount` in percent, `flc_window`, …), `episode_id` on every row. `download_flc.py` pulls the columns the feature table reads from it, under the engine's names: `date, hour_of_day, sku_id, fc, episode_id, starting_inventory, units_sold, total_discount` (percent), `category`. |
| `failures_2026-08-29.csv` | you, every morning, to the owner | One row per hour whose returned price did not reach the shelf. Empty here: the header alone is a valid file. |
