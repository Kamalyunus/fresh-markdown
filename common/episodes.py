"""Episode endings and what each one means for scrap.

An episode does not necessarily run the length of its window. `hours_remaining`
(source `flc_window`) counts the window down; rows stop either when the window
ends or when inventory reaches zero, whichever comes first. Those two endings
are economically opposite and must not be pooled:

  completed        hours_remaining reached 0. The window is over and whatever
                   inventory is left IS scrapped. This is the only ending that
                   contributes scrap.
  sold_out_early   inventory hit zero with window time left. A success, and
                   scrap is zero -- not because we assumed it, but because
                   there is nothing left to scrap.
  truncated        window time left AND inventory left, but no more rows. The
                   window's outcome is NOT IN THE DATA. Scrap is unknown, and
                   assuming the leftover was scrapped overstates it.

Counting every episode's last ending_inventory as scrap -- which every metric
here used to do -- charges the truncated episodes' unsold units to scrap on no
evidence. Those episodes are excluded from scrap-based aggregates instead, and
the excluded share is reported so the exclusion is visible rather than silent.
"""

import numpy as np
import pandas as pd

COMPLETED = "completed"
SOLD_OUT_EARLY = "sold_out_early"
TRUNCATED = "truncated"


def last_rows(d, order=("date", "hour_of_day")):
    """Final row of each episode, in window order."""
    return d.sort_values(list(order)).groupby("episode_id").tail(1)


def classify(hours_remaining, ending_inventory):
    """Ending type per row of a last-rows frame."""
    hr = np.asarray(hours_remaining, dtype=float)
    inv = np.asarray(ending_inventory, dtype=float)
    return pd.Series(
        np.where(hr <= 0, COMPLETED,
                 np.where(inv <= 0, SOLD_OUT_EARLY, TRUNCATED)),
        index=getattr(hours_remaining, "index", None))


def scrap_units(hours_remaining, ending_inventory):
    """Units scrapped, NaN where the window's outcome is not in the data.

    NaN propagates: a sum over a frame containing truncated episodes must be
    taken with the truncated ones dropped, not silently treated as zero.
    """
    kind = classify(hours_remaining, ending_inventory)
    inv = pd.Series(np.asarray(ending_inventory, dtype=float),
                    index=kind.index).clip(lower=0)
    return inv.where(kind == COMPLETED,
                     pd.Series(np.where(kind == SOLD_OUT_EARLY, 0.0, np.nan),
                               index=kind.index))


def ending_summary(d):
    """Share of each ending type, and the scrap at stake in the unknown one."""
    last = last_rows(d)
    kind = classify(last.hours_remaining, last.ending_inventory)
    counts = kind.value_counts()
    n = max(len(last), 1)
    inv = last.ending_inventory.clip(lower=0)
    return {
        "episodes": int(len(last)),
        "shares": {k: round(float(counts.get(k, 0)) / n, 4)
                   for k in (COMPLETED, SOLD_OUT_EARLY, TRUNCATED)},
        "scrap_units_completed": int(inv[kind == COMPLETED].sum()),
        "scrap_units_unknown_truncated": int(inv[kind == TRUNCATED].sum()),
        "share_of_naive_scrap_that_is_unknown": round(
            float(inv[kind == TRUNCATED].sum() / inv.sum()), 4)
            if inv.sum() > 0 else None,
        "note": ("Only `completed` windows contribute scrap. `sold_out_early` "
                 "has none by construction. `truncated` episodes have no "
                 "recorded window end, so their leftover units are unknown "
                 "rather than scrapped -- they are excluded from scrap and IL "
                 "aggregates. share_of_naive_scrap_that_is_unknown is how "
                 "much a last-row-inventory scrap figure would have "
                 "overstated."),
    }
