# data/

`flc.parquet`: the day's extract, the trailing 45 days of the hourly table
pulled from the warehouse by `download_flc.py` every morning and overwritten
each time. This IS the feed -- every shelf-hour with what sold, the closing
stock and the price that was on the shelf, your `episode_id` on every row
-- pulled rather than delivered as files. `build_features.py` reads it.
Nothing else lives here; the shape is `examples/feed_2026-08-29.parquet`.
