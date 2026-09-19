# decisions/

`<day>T<hh>.csv`: the hour's response, one row per snapshot row -- `sku_id,
fc, date, hour_of_day, episode_id` echoed, then `decision_id`
(`dec-<sku_id>|<fc>|<day>T<hh>`), `apply_discount_pct` (the discount to
apply, in percent), `apply_price` (the same as a price: put exactly this on
the shelf), `is_exploration` (always false here), `rejected` (null when
priced, else the reason). Written by `price_hour.py`; this is what you push
to the shelf, and `apply_discount_pct / 100` is next hour's
`current_discount` for the same shelf. The shape is
`examples/decisions_2026-08-29T11.csv`.
