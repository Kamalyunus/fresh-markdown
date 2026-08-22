"""How an episode ends, and what that means for scrap.

TWO ENDINGS, ONE ELIGIBILITY CHECK. The scrap rule is as simple as the data:

    leftover = max(0, starting_inventory - units_sold)   on the LAST row

    leftover == 0  ->  sold out          scrap = 0
    leftover  > 0  ->  ended with stock  scrap = leftover

That is the whole of it for any episode that has FINISHED. The third state
below is not a third way to end -- it is "this episode is not finished yet".

DATA QUIRK -- `ending_inventory` IS ZEROED ON AN EPISODE'S LAST ROW.
The source writes off whatever remains when a listing closes, so the last hour
breaks the inventory chain by design: `ending_inventory == 0` regardless of
whether it equals `starting_inventory - units_sold`. Measured on the filtered
production extract, 356,228 of 356,228 final rows carry the zero; 48,280
(13.55%) of them still had stock on hand. Two consequences, both severe if
missed:

  * Reading the last row's `ending_inventory` as scrap reports ZERO SCRAP FOR
    EVERY EPISODE. IL collapses to discount cost, the planner's reward loses
    the term that makes markdown worth doing, and nothing looks broken.
  * Treating the broken chain as a data error and dropping those episodes
    discards essentially all the genuine waste, keeping only guaranteed
    sellouts -- the exact opposite of the sample a scrap-cost signal needs.

So leftover is COMPUTED, never read. Note that the zero is on EVERY final row,
sold-out and leftover alike, which makes it useless as a scrap figure and
useless for telling the two endings apart. Only `leftover` does that.

DO NOT KEY ANY OF THIS TO `hours_remaining`. The counter (`flc_window`) is
NOMINAL and is still positive on ~99.9% of final rows. An earlier version
treated "counter reached zero" as the end-of-window signal; it fired on ~0.1%
of episodes and pushed ~99% of all real leftover into the unknown bucket,
emptying every scrap aggregate in the system. Confirmed with the business:
when a listing ends with stock on hand, those units are DISPOSED and counted
as scrap, whatever the counter says.

WHERE THE ELIGIBILITY CHECK EARNS ITS PLACE: live monitoring. Offline every
episode in the extract has finished, so the two cases above are the whole
story. In production, episodes are still in flight -- and an in-flight
episode's most recent row is not a final row, so it carries an honest,
non-zero `ending_inventory`. Its leftover is stock ON THE SHELF, not in the
bin. Booking it as scrap would count it today and count something different
tomorrow. The closure sentinel -- the source's own zero -- is what separates
"finished" from "still running", and it is the same test in both worlds.

  sold_out_early   leftover is zero. Nothing left to scrap, by fact rather
                   than assumption. Tested first: unambiguous either way.
  completed        leftover > 0 on a closed episode. Those units were
                   disposed of; this is where essentially all scrap lives.
  not_closed       leftover > 0 and NO closure sentinel. NOT an ending --
                   the episode is still running (or the feed cut it), so
                   scrap is unknown and it is excluded from scrap aggregates
                   rather than counted as zero. Empty on a closed extract.

The sentinel is DETECTED, not assumed (`write_off_convention`): a feed that
reports honest ending_inventory throughout has no sentinel to read, and
treating every episode as unfinished would move all scrap into UNKNOWN -- the
same silent emptying this module already suffered once. `ending_summary`
reports whether the convention was found.
"""

import numpy as np
import pandas as pd

COMPLETED = "completed"
SOLD_OUT_EARLY = "sold_out_early"
NOT_CLOSED = "not_closed"


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


def adjustment_reason(starting_inventory, units_sold, ending_inventory):
    """Why an outcome's inventory does not reconcile, or None.

    The event store quarantines any non-reconciling outcome that carries no
    reason, and a quarantined outcome never lands -- so an unnamed but
    legitimate break sinks event completeness and fails the shadow gate.

      restock     ending EXCEEDS what was left: stock was added.
      write-off   ending is exactly ZERO while stock remained. That is the
                  source's own convention -- it writes the remainder off and
                  reports 0 -- and it is recognised BY THE ZERO ITSELF, not
                  by position in the episode.

    Keying the write-off to "our last observed hour" was wrong and quarantined
    real outcomes in bulk: the source zeroes at ITS episode boundary, and once
    a window is merged across midnight that row sits in the MIDDLE of ours.
    Position is our bookkeeping; the zero is the source's fact.

    A PARTIAL shortfall -- ending above zero but below the leftover -- is
    unexplained inventory loss, matches no convention, and returns None on
    purpose so it quarantines and stays visible.
    """
    leftover = max(starting_inventory - units_sold, 0)
    if ending_inventory > leftover:
        return "intraday_restock"
    if ending_inventory == 0 and leftover > 0:
        return "episode_close_write_off"
    return None


def write_off_convention(last):
    """Is the source's closure sentinel present in this frame?

    A final row reporting `ending_inventory == 0` while stock remained can only
    be a write-off, so a single such row proves the convention is in force.
    Detected rather than assumed: a feed that reports honest ending_inventory
    throughout has no sentinel to read, and treating every episode as unclosed
    would move ALL scrap into UNKNOWN -- the same silent emptying this module
    already suffered once.
    """
    left = leftover_units(last.starting_inventory, last.units_sold).to_numpy()
    return bool(((last.ending_inventory.to_numpy() == 0) & (left > 0)).any())


def classify_last(last):
    """Ending type per episode, from a frame of FINAL rows.

    The source zeroes `ending_inventory` when a listing closes, so THE ZERO IS
    THE CLOSURE SIGNAL and its absence means the episode did not close inside
    this data. That is the source's own fact; earlier versions inferred closure
    from `hours_remaining == 0` (fires on ~0.1% of episodes -- the counter is
    nominal) and then from proximity to the extract's last timestamp (a
    heuristic with a tolerance constant). The sentinel needs neither.
    """
    left = leftover_units(last.starting_inventory, last.units_sold).to_numpy()
    closed = last.ending_inventory.to_numpy() == 0
    if not write_off_convention(last):
        # nothing to read: fall back to treating every episode as closed, which
        # is what this data can support. `ending_summary` reports the fallback.
        closed = np.ones(len(last), dtype=bool)
    return pd.Series(
        np.where(left <= 0, SOLD_OUT_EARLY,
                 np.where(closed, COMPLETED, NOT_CLOSED)),
        index=last.episode_id.to_numpy() if "episode_id" in last else last.index)


def classify(d):
    """Ending type per episode, indexed by episode_id."""
    return classify_last(last_rows(d))


def scrap_units(d):
    """Units scrapped per episode, NaN where the episode did not close here.

    NaN propagates: a sum over a frame containing unfinished episodes must be
    taken with those dropped, not silently treated as zero.
    """
    return scrap_units_last(last_rows(d))


def scrap_units_last(last):
    """scrap_units from a frame of FINAL rows."""
    kind = classify_last(last)
    left = pd.Series(
        leftover_units(last.starting_inventory, last.units_sold).to_numpy(),
        index=kind.index)
    return left.where(kind == COMPLETED,
                      pd.Series(np.where(kind == SOLD_OUT_EARLY, 0.0, np.nan),
                                index=kind.index))


def ending_summary(d):
    """Share of each ending type, and the scrap at stake in the unknown one."""
    last = last_rows(d)
    kind = classify_last(last)
    left = pd.Series(
        leftover_units(last.starting_inventory, last.units_sold).to_numpy(),
        index=last.episode_id.to_numpy())
    hr = pd.Series(last.hours_remaining.to_numpy(),
                   index=last.episode_id.to_numpy())
    counts = kind.value_counts()
    n = max(len(last), 1)
    return {
        "episodes": int(len(last)),
        # if this is false the source is NOT marking closure, so truncation is
        # undetectable and every episode is treated as closed. Read it before
        # trusting the not_closed share.
        "write_off_convention_in_force": write_off_convention(last),
        "final_rows_without_closure_sentinel": int(
            (last.ending_inventory.to_numpy() != 0).sum()),
        "shares": {k: round(float(counts.get(k, 0)) / n, 4)
                   for k in (SOLD_OUT_EARLY, COMPLETED, NOT_CLOSED)},
        "scrap_units_completed": int(left[kind == COMPLETED].sum()),
        "scrap_units_unknown_not_closed": int(left[kind == NOT_CLOSED].sum()),
        "share_episodes_ending_by_write_off": round(float(
            ((kind == COMPLETED) & (left > 0)).mean()), 4),
        # the diagnostic that caught the original misclassification: if the
        # counter almost never reaches zero, any rule keyed to it is wrong
        "share_last_row_counter_at_zero": round(float((hr <= 0).mean()), 4),
        "share_completed_with_counter_still_positive": round(float(
            ((kind == COMPLETED) & (hr > 0)).mean()), 4),
        "last_row_ending_inventory_ever_positive": bool(
            (last.ending_inventory > 0).any()),
        "note": ("Scrap is max(0, starting_inventory - units_sold) on the last "
                 "row, NOT ending_inventory, which the source zeroes when it "
                 "writes off the remainder. An episode ends when its listing "
                 "ends, NOT when hours_remaining reaches zero -- the counter is "
                 "nominal and usually still positive, so "
                 "share_completed_with_counter_still_positive is expected to be "
                 "large. Closure is read from the source's own sentinel -- "
                 "ending_inventory zeroed on the final row -- so an episode "
                 "whose final row still reports honest inventory is the only "
                 "kind with an unknown outcome."),
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
