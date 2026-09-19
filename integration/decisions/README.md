# decisions/

`<day>T<hh>.csv`: the hour's response, one row per snapshot row -- the
discount to apply as a percent and as a price, the decision id, or
`rejected` with the reason. Written by `price_hour.py`; this is what you
push to the shelf. The shape is `examples/decisions_2026-08-29T11.csv`.
