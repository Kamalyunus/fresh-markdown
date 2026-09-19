# data/

`flc.parquet`: the day's extract, the trailing 45 days of the hourly table
pulled from the warehouse by `download_flc.py` every morning and overwritten
each time. `build_features.py` reads it. Nothing else lives here.
