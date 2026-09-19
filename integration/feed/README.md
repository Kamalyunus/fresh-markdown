# feed/

`<day>.parquet`: yesterday's hourly feed as a file -- every shelf-hour with
what sold, the closing stock and the price that was on the shelf -- if you
deliver it as a file for the owner's learning lane. None of this folder's
commands reads it: the morning pull reads the same rows from the warehouse.
The shape is `examples/feed_2026-08-29.parquet`.
