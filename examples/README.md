# Example files — the four tables of the hourly interface

Written by the rehearsal (`python3 -m tools.e2e_cycle`) against the
**synthetic** shop, so every id, price and count here is a fixture's. They
show the shape, the column names, the units and the ids; they say nothing
about production.

| File | Who writes it | What it is |
| --- | --- | --- |
| `snapshot_2026-08-29T11.csv` | engineering, every hour | The shelf at the top of hour 11 in the feed's own schema, plus the rows of hour 10 that just closed. `discount` is a percent; `flc_window` is hours still to come; `episode_id` is the producer's (here the shop's). |
| `decisions_2026-08-29T11.csv` | `ops.price_hour`, every hour | One row per snapshot row: the id echoed, `decision_id` (`dec-<sku>\|<fc>\|<date>T<hh>`), the discount to apply as a percent and as a price, or `rejected` with the reason. |
| `feed_2026-08-29.parquet` | engineering, every morning | The day's hourly feed, every shelf-hour, the write-off zero in `ending_inventory` on a window's last row, and `episode_id` carried through from the snapshot. |
| `failures_2026-08-29.csv` | engineering, every morning | One row per hour whose returned price did not reach the shelf. Empty here: the header alone is a valid file. |

Check any of them the way the page says to:

```bash
python3 -m ops.check_inputs --snapshot examples/snapshot_2026-08-29T11.csv \
    --feed examples/feed_2026-08-29.parquet --failures examples/failures_2026-08-29.csv
```
