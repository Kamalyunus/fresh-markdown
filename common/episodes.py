"""Episode endings and what each one means for scrap.

An episode does not necessarily run the length of its window. `hours_remaining`
(source `flc_window`) counts the window down; rows stop either when the window
ends or when inventory reaches zero, whichever comes first. Those two endings
are economically opposite and must not be pooled.

DATA QUIRK -- `ending_inventory` IS ALWAYS ZERO ON AN EPISODE'S LAST ROW.
The source writes off whatever remains when the window closes, so the last
hour breaks the inventory chain by design: `ending_inventory == 0` regardless
of whether it equals `starting_inventory - units_sold`. Verified upstream
across 953K episodes -- it is never positive there, and ~49.5% of episodes end
by write-off rather than clean sellout. Two consequences, both severe if
missed:

  * Reading the last row's `ending_inventory` as scrap reports ZERO SCRAP FOR
    EVERY EPISODE. IL collapses to discount cost, the planner's reward loses
    the term that makes markdown worth doing, and nothing looks broken.
  * Treating the broken chain as a data error and dropping those episodes
    discards essentially all the genuine waste, keeping only guaranteed
    sellouts -- the exact opposite of the sample a scrap-cost signal needs.

True leftover at the window close is therefore `max(0, starting_inventory -
units_sold)` on the last row, never `ending_inventory`. That formula is also
correct where the chain is honest, so it is the only one used here.

  completed        hours_remaining reached 0. The window is over and the
                   leftover IS scrapped. The only ending that contributes
                   scrap.
  sold_out_early   inventory hit zero with window time left. Scrap is zero
                   because there is nothing left, not by assumption.
  truncated        window time left AND stock left, but no more rows. The
                   window's outcome is NOT IN THE DATA, so scrap is unknown
                   and those episodes are excluded from scrap aggregates
                   rather than counted as zero.
"""

import numpy as np
import pandas as pd

COMPLETED = "completed"
SOLD_OUT_EARLY = "sold_out_early"
TRUNCATED = "truncated"


def last_rows(d, order=("date", "hour_of_day")):
    """Final row of each episode, in window order."""
    return d.sort_values(list(order)).groupby("episode_id").tail(1)


def leftover_units(starting_inventory, units_sold):
    """Stock still on hand at the close of the last hour.

    `max(0, starting_inventory - units_sold)`. NOT `ending_inventory`, which
    the source zeroes on the last row when it writes the remainder off.
    """
    start = np.asarray(starting_inventory, dtype=float)
    sold = np.asarray(units_sold, dtype=float)
    out = np.clip(start - sold, 0, None)
    idx = getattr(starting_inventory, "index", None)
    return pd.Series(out, index=idx)


def classify(hours_remaining, starting_inventory, units_sold):
    """Ending type per row of a last-rows frame."""
    hr = np.asarray(hours_remaining, dtype=float)
    left = leftover_units(starting_inventory, units_sold)
    return pd.Series(
        np.where(hr <= 0, COMPLETED,
                 np.where(left.to_numpy() <= 0, SOLD_OUT_EARLY, TRUNCATED)),
        index=left.index)


def scrap_units(hours_remaining, starting_inventory, units_sold):
    """Units scrapped, NaN where the window's outcome is not in the data.

    NaN propagates: a sum over a frame containing truncated episodes must be
    taken with the truncated ones dropped, not silently treated as zero.
    """
    kind = classify(hours_remaining, starting_inventory, units_sold)
    left = leftover_units(starting_inventory, units_sold)
    return left.where(kind == COMPLETED,
                      pd.Series(np.where(kind == SOLD_OUT_EARLY, 0.0, np.nan),
                                index=kind.index))


def ending_summary(d):
    """Share of each ending type, and the scrap at stake in the unknown one."""
    last = last_rows(d)
    kind = classify(last.hours_remaining, last.starting_inventory,
                    last.units_sold)
    left = leftover_units(last.starting_inventory, last.units_sold)
    counts = kind.value_counts()
    n = max(len(last), 1)
    return {
        "episodes": int(len(last)),
        "shares": {k: round(float(counts.get(k, 0)) / n, 4)
                   for k in (COMPLETED, SOLD_OUT_EARLY, TRUNCATED)},
        "scrap_units_completed": int(left[kind == COMPLETED].sum()),
        "scrap_units_unknown_truncated": int(left[kind == TRUNCATED].sum()),
        "share_episodes_ending_by_write_off": round(float(
            ((kind == COMPLETED) & (left > 0)).mean()), 4),
        "last_row_ending_inventory_ever_positive": bool(
            (last.ending_inventory > 0).any()),
        "note": ("Scrap is max(0, starting_inventory - units_sold) on the last "
                 "row, NOT ending_inventory, which the source zeroes when it "
                 "writes off the remainder. If "
                 "last_row_ending_inventory_ever_positive is false, the "
                 "write-off convention is in force and any metric reading "
                 "ending_inventory directly is reporting zero scrap."),
    }


def extend_to_window(d, feature_cols=(), max_tail_hours=None):
    """Append the rows a window has but the data does not.

    Rows stop at zero inventory, so an episode's row count understates its
    window whenever it sold out. A planner handed the row count as its horizon
    is being told the future: the horizon is short *because* the item sold
    out. Every episode is therefore extended to its full window --
    hours_remaining running down to 0 -- with synthetic rows marked
    `is_observed = False`.

    The extension is exact, not an approximation: every baseline feature is
    either episode-constant (category, fc, price, the velocity features, which
    are keyed to the episode's first date) or a function of the advancing
    timestamp (hour_of_day, dow, day_of_month). Synthetic rows carry no sales
    and must be filtered out of fidelity, likelihood and IL -- they exist only
    so `mu_ref` can be predicted for the hours the DP must plan over.
    """
    d = d.copy()
    d["is_observed"] = True
    last = last_rows(d)
    need = last[last.hours_remaining > 0]
    if not len(need):
        return d.sort_values(["episode_id", "date", "hour_of_day"])

    if max_tail_hours is not None and len(need):
        worst = float(need.hours_remaining.max())
        if worst > max_tail_hours:
            raise ValueError(
                f"episode window of {worst:.0f}h exceeds max_window_hours "
                f"({max_tail_hours}); prepare_data should have dropped it. "
                "Refusing to generate an unbounded synthetic tail.")

    carry = [c for c in feature_cols if c in d.columns]
    # emit dates in the column's own type. Downstream split filters compare
    # `date.astype(str)`, and a Timestamp stringifies with a time component
    # that would silently fall outside every configured window.
    as_date = not isinstance(d.date.iloc[0], pd.Timestamp)
    rows = []
    for r in need.itertuples():
        base = pd.Timestamp(r.date) + pd.Timedelta(hours=int(r.hour_of_day))
        tail_inv = max(int(r.starting_inventory) - int(r.units_sold), 0)
        for k in range(1, int(r.hours_remaining) + 1):
            ts = base + pd.Timedelta(hours=k)
            row = {c: getattr(r, c) for c in carry}
            row.update({
                "episode_id": r.episode_id,
                "date": ts.date() if as_date else ts.normalize(),
                "hour_of_day": ts.hour,
                "hours_remaining": r.hours_remaining - k,
                # true leftover, not the written-off zero
                "starting_inventory": tail_inv,
                "ending_inventory": tail_inv,
                "units_sold": 0,
                "is_observed": False,
            })
            rows.append(row)
    out = pd.concat([d, pd.DataFrame(rows)], ignore_index=True)
    return out.sort_values(["episode_id", "date", "hour_of_day"])
