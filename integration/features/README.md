# features/

`<day>.parquet`: the day's feature table, one row per SKU x FC plus a pooled
row per SKU, written by `build_features.py` every morning. `price_hour.py`
reads the table of each episode's OPENING day, so keep at least the last
week of tables here (an episode lasts at most five days).
