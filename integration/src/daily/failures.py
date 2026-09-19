"""daily.failures -- the failed-pushes table, read once for two readers.

The one fact only engineering knows -- did the returned price reach the
shelf -- arrives as a table of the hours where it did not: one row per
failed push, in the contract's names (sku_id, fc, date, hour_of_day,
reason) or the feed's (skuseq, hour). `daily.ingest_outcomes` reads it to
keep a failed push out of the evidence; `ops.check_inputs` reads it to
check the file before launch. Neither needs the other, so the loader
lives here and the pricing folder carries this file alone.
"""

from common.io import read_rows
from events.pairs import hour_key
from fit.prepare_data import SOURCE_TO_CANONICAL


class Failures(dict):
    """{hour key: reason} plus the rows that named no hour."""
    unkeyable = 0


def load_failures(path):
    """{key: reason} from the failures input -- a parquet/CSV table or
    JSONL (common.io.read_rows), in the contract's names or the feed's
    (skuseq, hour) -- or {} when no file is given. One unkeyable row (a
    NaN id, a blank line, a line that is not an object) costs that row,
    never the batch: it is counted in `push_failures_unkeyable` on the
    returned dict's `.unkeyable`."""
    out = Failures()
    if not path:
        return out
    for r in read_rows(path, rename=SOURCE_TO_CANONICAL):
        try:
            k = hour_key(r["sku_id"], r["fc"], r["date"], r["hour_of_day"])
        except (KeyError, TypeError, ValueError):
            out.unkeyable += 1
            continue
        out[k] = r.get("reason") or "unspecified"
    return out
