# events_store/

`decisions.jsonl` and `rejections.jsonl`: the append-only log, one JSON
line per priced shelf-hour and one per refused shelf-hour, appended by
`price_hour.py` (never on a `--dry-run`). Nothing in this folder reads
them; the owner's learning lane collects them. Back this folder up daily.
