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


def window_slice(d, start=None, end=None):
    """Episodes whose WINDOW STARTED in [start, end] -- whole, never sliced.

    The one rule for cutting this data by date, because a row-level cut is
    wrong in exactly the way the episode definition exists to prevent. FLC
    windows routinely run past midnight, so `d[d.date >= start]` keeps the
    tail of a window that opened the day before: the entry decision that set
    the whole price path is gone, the opening inventory is gone, and
    `hours_remaining` starts mid-countdown. The episode survives as a short
    one that never existed.

    Assignment is by the window's FIRST date, so every episode lands in
    exactly one slice and no boundary runs through the middle of one.
    """
    if start is None and end is None:
        return d
    opened = d.groupby("episode_id")["date"].transform("min").astype(str)
    keep = pd.Series(True, index=d.index)
    if start is not None:
        keep &= opened.ge(str(start))
    if end is not None:
        keep &= opened.le(str(end))
    return d[keep]


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


# --------------------------------------------------------------------------
# THE SOURCE'S INVENTORY CONVENTION, stated once.
#
#   `ending_inventory` is the FINAL quantity on hand at the close of the hour,
#   AFTER any restock that arrived during it. It is not `starting - sold`; it
#   is what the source counted at the end.
#
# Everything about hour-level data quality follows from that one sentence:
#
#   ending == starting - sold     the ordinary hour. Nothing arrived.
#   ending >  starting - sold     STOCK ARRIVED. Note this holds whenever a
#                                 restock happened, including the case where
#                                 the hour sold MORE than it opened with --
#                                 `starting - sold` goes negative and any
#                                 ending at all exceeds it.
#   ending == 0 and net > 0       the source wrote the remainder off, which is
#                                 how a listing closes.
#   0 < ending < starting - sold  stock left without being sold and without
#                                 being written off. Nothing explains this.
#
# And ACROSS hours, with no exceptions inside an episode:
#
#   starting[t+1] == ending[t]    the chain is continuous, because ending
#                                 already carries the restock forward.
#
# `units_sold > starting_inventory` is therefore NOT an impossible quantity.
# It is the restock signal in its plainest form, and a filter that deletes it
# is deleting the restocks it is meant to detect.
# --------------------------------------------------------------------------

RECONCILES = "reconciles"
RESTOCK = "intraday_restock"
WRITE_OFF = "episode_close_write_off"
SHORTFALL = "unexplained_shortfall"


def leftover_units(starting_inventory, units_sold):
    """Stock still on hand at the close of the last hour.

    `max(0, starting_inventory - units_sold)`. NOT `ending_inventory`, which
    the source zeroes on the last row when it writes the remainder off.

    EXACT except on a last hour that also took a restock, where the arriving
    quantity is not recoverable from (starting, sold, ending) once the ending
    has been zeroed -- see `restock_on_final_hour`, which counts them so the
    ambiguity is measured rather than assumed away.
    """
    start = np.asarray(starting_inventory, dtype=float)
    sold = np.asarray(units_sold, dtype=float)
    out = np.clip(start - sold, 0, None)
    idx = getattr(starting_inventory, "index", None)
    return pd.Series(out, index=idx)


def hour_status(starting_inventory, units_sold, ending_inventory):
    """Classify every hour against the source's convention. Vectorised.

    The single source of truth for hour-level DQ: `adjustment_reason` is the
    scalar form of the same four rules, and `bootstrap.prepare_data` filters
    on this. Test ORDER is load-bearing -- restock must be asked before
    write-off, or an hour that restocked and then sold out to exactly zero
    reads as an unexplained zero instead of the restock it was.
    """
    start = np.asarray(starting_inventory, dtype="int64")
    sold = np.asarray(units_sold, dtype="int64")
    end = np.asarray(ending_inventory, dtype="int64")
    net = start - sold
    return np.where(end == net, RECONCILES,
                    np.where(end > net, RESTOCK,
                             np.where((end == 0) & (net > 0),
                                      WRITE_OFF, SHORTFALL)))


def episode_flow(d):
    """Per-episode supply accounting: what came in, what went missing.

    `starting_inventory` on the opening row is NOT an episode's supply once a
    restock is possible, and every ratio built on it breaks in the same
    direction. A window that opened with 3, took 10 mid-flight and sold 9
    reads as 300% cleared AND "fully cleared" while it scrapped 4 units. That
    is not a rounding problem, it is a nonsense number reaching a chart.

    The hour discrepancy is `(starting - ending) - sold`: what the inventory
    says left, minus what sales claim. Summed over the episode it NETS, and
    the netting is the point --

      net <  0   stock genuinely ARRIVED. supply = opening + |net|.
      net >  0   stock genuinely VANISHED, unsold and unwritten-off.
      net == 0   internally consistent. Any hour-level wobble is the sales
                 feed bucketing a sale an hour off the inventory snapshot:
                 a +1 at one hour and a -1 at the next cancel exactly, and
                 the episode neither gained nor lost anything.

    The write-off row is excluded because the source zeroes its ending, which
    would otherwise read as the whole remainder vanishing.

    Returns a frame indexed by episode_id with `opening`, `arrived`,
    `vanished`, `supply`, `sold` and `clearance`. `clearance` cannot exceed 1
    by construction, which is the property the old ratio lacked.
    """
    d = d.sort_values(["date", "hour_of_day"])
    status = hour_status(d.starting_inventory, d.units_sold, d.ending_inventory)
    disc = ((d.starting_inventory.to_numpy() - d.ending_inventory.to_numpy())
            - d.units_sold.to_numpy())
    disc = np.where(status == WRITE_OFF, 0, disc)

    g = pd.DataFrame({"episode_id": d.episode_id.to_numpy(), "disc": disc,
                      "sold": d.units_sold.to_numpy(),
                      "start": d.starting_inventory.to_numpy()})
    agg = g.groupby("episode_id", sort=False).agg(
        net=("disc", "sum"), sold=("sold", "sum"), opening=("start", "first"))
    agg["arrived"] = np.clip(-agg.net, 0, None)
    agg["vanished"] = np.clip(agg.net, 0, None)
    agg["supply"] = agg.opening + agg.arrived
    agg["clearance"] = np.divide(
        agg.sold.to_numpy(), agg.supply.to_numpy(),
        out=np.zeros(len(agg)), where=agg.supply.to_numpy() > 0)
    return agg.drop(columns=["net"])


def censored_hours(starting_inventory, units_sold, ending_inventory):
    """Hours where demand was only observed as a LOWER bound. Vectorised.

    The rule is `starting == sold` AND the hour took no restock. Both halves
    matter and the second one is why this cannot be `sold >= starting`:

      - `starting == sold`, no restock -- the shelf emptied through sales and
        stayed empty. Whoever arrived next bought nothing and left no trace,
        so true demand is `>= sold`. CENSORED.
      - a RESTOCK hour ending with stock on the shelf -- e.g. opened with 3,
        sold 2, ended with 5 because 4 arrived. Nothing ran out and demand was
        observed exactly. NOT censored, even though the hour has a restock.
      - `sold > starting` -- e.g. opened with 1, sold 3. Under `sold >=
        starting` this counted as censored, and it is the opposite: the hour
        sold MORE than it opened with precisely because stock arrived.

    Four call sites carried `units_sold >= starting_inventory` independently
    -- m5, the dispersion fit, the prior fit and the live posterior update.
    All four marked every restock hour censored, which inflates demand on
    exactly the hours that had the most stock to sell.

    ONE CASE IS LEFT OUT DELIBERATELY: a restock hour ending at zero, which
    did run out and arguably is censored. It is excluded because the arrival
    time inside the hour is unknown, so the bound is not `>= sold` for a
    well-defined stock level. `censoring_edge_cases` counts them.
    """
    start = np.asarray(starting_inventory, dtype="int64")
    sold = np.asarray(units_sold, dtype="int64")
    status = hour_status(start, sold, ending_inventory)
    return (status == RECONCILES) & (start - sold == 0)


def censoring_edge_cases(starting_inventory, units_sold, ending_inventory):
    """The hours the censoring rule deliberately does not claim."""
    start = np.asarray(starting_inventory, dtype="int64")
    sold = np.asarray(units_sold, dtype="int64")
    end = np.asarray(ending_inventory, dtype="int64")
    status = hour_status(start, sold, end)
    restock_to_zero = (status == RESTOCK) & (end == 0)
    n = max(len(start), 1)
    return {
        "restock_hours_ending_at_zero": int(restock_to_zero.sum()),
        "share_of_hours": round(float(restock_to_zero.sum()) / n, 6),
        "sold_over_starting_hours": int((sold > start).sum()),
        "note": ("A restock hour ending at zero DID run out, but the arrival "
                 "time inside the hour is unknown, so there is no stock level "
                 "the bound `demand >= sold` is taken against. Treated as "
                 "uncensored, which understates demand on those hours. If "
                 "share_of_hours is not small, the rule needs the arrival "
                 "time from the source rather than a convention."),
    }


def continuity_breaks(d):
    """Hours whose ending is not the next hour's starting, inside an episode.

    `ending` already carries any restock forward, so the chain has NO
    legitimate exception here -- unlike the within-hour test, which has three.
    A violation means the two fields disagree about the same instant, which no
    business event explains.

    Returns a boolean Series aligned to `d`, False on each episode's last hour
    (there is no next hour to disagree with). `d` must be in window order.
    """
    nxt = d.groupby("episode_id")["starting_inventory"].shift(-1)
    return (nxt.notna() & (nxt != d.ending_inventory)).to_numpy()


def restock_on_final_hour(d):
    """Episodes whose LAST hour took a restock -- where scrap goes soft.

    Scrap is `max(starting - sold, 0)` on the last row, which is exact until
    stock arrives during that row. Then the true leftover is `ending`, and if
    the source also zeroed `ending` to write the remainder off, the arriving
    quantity is gone and the leftover is genuinely unrecoverable.

    Counted, not corrected: changing what IL means in the same breath as
    changing what "broken" means would make neither movement attributable.
    """
    last = last_rows(d)
    kind = hour_status(last.starting_inventory, last.units_sold,
                       last.ending_inventory)
    hit = kind == RESTOCK
    zeroed = hit & (last.ending_inventory.to_numpy() == 0)
    return {
        "episodes_with_restock_on_final_hour": int(hit.sum()),
        "of_those_ending_zeroed_so_scrap_unrecoverable": int(zeroed.sum()),
        "share_of_episodes": round(float(hit.mean()), 6) if len(last) else 0.0,
        "note": ("On these, leftover_units reads max(starting - sold, 0) and "
                 "the true leftover is `ending`. Where ending is also zero "
                 "the restock quantity cannot be recovered at all, so scrap "
                 "is unknown rather than zero. Read this before quoting a "
                 "scrap or IL figure to the decimal."),
    }


def adjustment_reason(starting_inventory, units_sold, ending_inventory):
    """Why an outcome's inventory does not reconcile, or None.

    Scalar form of `hour_status`, and the rule `events.store` enforces on
    every live outcome. The event store quarantines any non-reconciling
    outcome that carries no reason, and a quarantined outcome never lands --
    so an unnamed but legitimate break sinks event completeness and fails the
    shadow gate.

      restock     ending EXCEEDS what was left over. `starting - sold` is
                  used UNCLIPPED here: an hour that sold more than it opened
                  with has a negative net, and clipping it to zero hid the
                  most obvious restock there is -- one that arrived and then
                  sold out to exactly zero, which read as an unexplained zero
                  and quarantined in bulk.
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
    net = starting_inventory - units_sold
    if ending_inventory > net:
        return RESTOCK
    if ending_inventory == 0 and net > 0:
        return WRITE_OFF
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


def _unknown_by(last, kind, left, by, top=8):
    """Unclosed episodes and the units they hold, grouped by month or category.

    Answers the question a single share cannot: is the unknown scrap one
    incident, a standing property of the feed, or one corner of the
    catalogue. Months are returned in order; categories by units descending,
    capped, with the remainder folded into `other` rather than dropped.
    """
    ids = last.episode_id.to_numpy() if "episode_id" in last else last.index
    key = (pd.Series(pd.to_datetime(last.date).dt.strftime("%Y-%m").to_numpy(),
                     index=ids) if by == "month"
           else pd.Series(last[by].astype(str).to_numpy(), index=ids))
    sel = kind == NOT_CLOSED
    if not sel.any():
        return {}
    g = pd.DataFrame({"key": key[sel], "units": left[sel]}).groupby("key")
    out = [(k, {"episodes": int(len(v)), "leftover_units": int(v.units.sum())})
           for k, v in g]
    if by == "month":
        return dict(sorted(out))
    out.sort(key=lambda kv: -kv[1]["leftover_units"])
    head, tail = out[:top], out[top:]
    if tail:
        head.append(("other", {
            "episodes": sum(v["episodes"] for _, v in tail),
            "leftover_units": sum(v["leftover_units"] for _, v in tail)}))
    return dict(head)


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
        # WHERE the unknowns sit in time and in the catalogue. The extract
        # boundary explains some of them and nothing can be done about those;
        # the rest are unclosed for a reason a longer extract will NOT fix --
        # a gap in the hourly feed, or a subset whose feed never writes off.
        # The split is `share_of_unclosed_explained_by_edge` in the
        # dp_eligible waterfall row. Here: concentrated in the LAST month
        # reads as the boundary; concentrated in one earlier month reads as
        # an incident; spread evenly across every month reads as a standing
        # property of the feed; concentrated in a few categories names the
        # subset.
        "not_closed_by_month": _unknown_by(last, kind, left, "month"),
        "not_closed_by_category": _unknown_by(last, kind, left, "category"),
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
                 "kind with an unknown outcome. Unclosed episodes are KEPT in "
                 "the population -- their observed hours are good demand data "
                 "and only the ending is missing -- so not_closed here counts "
                 "both kinds: the ones the extract boundary cut off, and the "
                 "ones unclosed for a reason a longer extract will not fix "
                 "(a gap in the hourly feed, or a subset whose feed never "
                 "writes off). The split is edge_truncated."
                 "share_of_unclosed_explained_by_edge in the dp_eligible "
                 "waterfall row; not_closed_by_month and "
                 "not_closed_by_category say where the residue sits."),
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
