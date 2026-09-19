"""The producers' episode-id rule, for one shelf and one clock step. The
hourly command never assigns an id with it; it evaluates it to COUNT the ids
that disagree, and reports them."""
from pricing.keys import as_number, hour_int, hours_between, iso_day

RULE = ("continue last hour's episode on the shelf when last hour's row is one "
        "hour earlier, did not end at the write-off zero, and the counter "
        "stepped down by one or stepped up/flat while stock arrived "
        "(ending > inventory - units_sold); otherwise open "
        "skuseq|fc|<date>T<hour>")


def continues(prev, row):
    """RULE for one shelf: `prev` is last hour's row (feed names), `row` this
    hour's; either may be None."""
    if prev is None or row is None:
        return False
    try:
        one = hours_between(iso_day(prev["date"]), hour_int(prev["hour"]),
                            iso_day(row["date"]), hour_int(row["hour"])) == 1
    except (KeyError, TypeError, ValueError):
        return False
    if not one:
        return False
    ending, start, sold = (as_number(prev.get("ending_inventory")), as_number(prev.get("inventory")),
                           as_number(prev.get("units_sold")))
    if ending is not None and ending == 0:
        return False                                       # the write-off zero closed it
    c_prev, c_now = as_number(prev.get("flc_window")), as_number(row.get("flc_window"))
    if c_prev is None or c_now is None:
        return False
    step = c_now - c_prev
    if step == -1:
        return True
    if step < -1:
        return False
    if None in (ending, start, sold):
        return False
    return ending > start - sold                           # stock arrived: the same window
