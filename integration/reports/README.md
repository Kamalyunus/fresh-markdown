# reports/

`hours/<day>T<hh>.json`: the hour's counts, written by `price_hour.py`
with `--report`. Alert on its `rejected` count. A morning report lands here
too when `build_features.py` is given `--report`.
