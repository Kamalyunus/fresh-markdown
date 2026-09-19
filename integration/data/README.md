# data/

`flc.parquet`: the day's extract, the trailing 45 days of the hourly table
pulled from the warehouse by `download_flc.py` every morning and overwritten
each time. Only the columns the feature table reads, under the engine's
names: `date, hour_of_day, sku_id, fc, episode_id, starting_inventory,
units_sold, total_discount` (percent), `category`. `build_features.py`
reads it. Nothing else lives here.
