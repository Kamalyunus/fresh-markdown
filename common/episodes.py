"""Episode identification, closure/scrap/censoring conventions, window extension.

Identity: opening + restocked == sold + scrap (scrap = leftover + shrink).
`ending_inventory` is the counted end-of-hour quantity, restocks included, and
`starting[t+1] == ending[t]` has no exception. The source zeroes
`ending_inventory` on an episode's LAST ROW at close: leftover is COMPUTED,
never read; closure is that zero, never `hours_remaining`; censoring is
`starting == sold`, last row only. design.md sections 5.2/12a for the rationale."""

import numpy as np
import pandas as pd

# DID IT CLOSE -- decided by `ending_inventory == 0` on the last row, alone.
COMPLETED = "completed"
SOLD_OUT_EARLY = "sold_out_early"
NOT_CLOSED = "not_closed"


def is_anchor_row(d, tier_step):
    """Rows priced AT the reference discount (within half a tier): the level
    fit, the fidelity anchor ratio and the gate's anchor share all mean
    this one mask. (`ref_rate_anchor_band` in prepare_data is a wider,
    separately configured band for the demand-rate features.)"""
    return (d.total_discount - d.d_ref).abs() <= tier_step / 2


def calendar_days(dates):
    """Days spanned by `dates`, inclusive, never below 1 -- the one n_days
    rule for "spend per day" (shadow and the backtest used two)."""
    ts = pd.to_datetime(pd.Series(dates))
    return max((ts.max() - ts.min()).days + 1, 1)


def week_start(ts):
    """The ISO week (Mon-Sun) holding `ts`, as its Monday."""
    return pd.Timestamp(ts).to_period("W").start_time


def week_key(dates):
    """`week_start` per row, as the "YYYY-MM-DD" key the factor schedules use."""
    return (pd.to_datetime(dates).dt.to_period("W").dt.start_time
            .dt.strftime("%Y-%m-%d"))


def trailing_weeks_window(d, week_start, weeks_back):
    """The rows a factor fit for the week starting `week_start` may read:
    WHOLE episodes that opened in the `weeks_back` weeks strictly before it,
    plus how many distinct opening weeks that window actually holds.

    The one cut shared by the artifact schedule (train_baseline) and the
    shadow re-fit; a row-level week cut in either truncated windows at the
    midnight seam and the two solved on different rows."""
    w0 = pd.Timestamp(week_start)
    lo = w0 - pd.Timedelta(weeks=int(weeks_back))
    window = window_slice(d, lo.strftime("%Y-%m-%d"),
                          (w0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if not len(window):
        return window, 0
    opened = window.groupby("episode_id")["date"].transform("min")
    return window, int(week_key(opened).nunique())


def window_slice(d, start=None, end=None):
    """Episodes whose WINDOW STARTED in [start, end] -- whole, never sliced.

    Assignment is by the window's FIRST date, so every episode lands in exactly
    one slice; a row-level date cut truncates windows that cross midnight.
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


# Hour statuses vs the source convention (`ending` is the counted end-of-hour
# quantity, restocks included; `units_sold > starting` is a restock, not junk).

RECONCILES = "reconciles"
RESTOCK = "intraday_restock"
WRITE_OFF = "episode_close_write_off"
SHORTFALL = "unexplained_shortfall"


def net_leftover(starting_inventory, units_sold):
    """`starting_inventory - units_sold`, UNCLIPPED. Its SIGN is the answer.

    On a last row: > 0 scrap; == 0 censored (true demand only known `>= sold`);
    < 0 stock ARRIVED during the final hour. Clipping folds the third case into
    the second, making a restocked close read as a clean sell-out.
    """
    start = np.asarray(starting_inventory, dtype=float)
    sold = np.asarray(units_sold, dtype=float)
    return pd.Series(start - sold, index=getattr(starting_inventory, "index", None))


def leftover_units(starting_inventory, units_sold):
    """Stock still on hand at the close: `max(0, net_leftover)`.

    NOT `ending_inventory`, which the source zeroes on the last row. Exact
    except on a final hour that also restocked (those fail `final_hour_clean`
    and are gated out); read the sign from `net_leftover`, never from this.
    """
    return net_leftover(starting_inventory, units_sold).clip(lower=0)


def shrink_by_hour(starting_inventory, units_sold, ending_inventory, is_last_row):
    """Units that vanished in each hour: `(starting - ending) - sold`, clipped
    at zero, with the write-off exemption on the LAST ROW ONLY.

    The one home for shrink. `episode_flow` aggregates it as `vanished` and
    `daily.monitor` measures the live guardrail with it -- a second
    hand-rolled copy is how the trigger came to run looser than the floor it
    is compared against. Mid-episode a zeroed ending with stock still owed is
    shrink, not a close (learnings.md); restock hours clip to zero.
    """
    disc = ((np.asarray(starting_inventory) - np.asarray(ending_inventory))
            - np.asarray(units_sold))
    status = hour_status(starting_inventory, units_sold, ending_inventory)
    return np.where((status == WRITE_OFF) & np.asarray(is_last_row), 0,
                    np.clip(disc, 0, None))


def hour_status(starting_inventory, units_sold, ending_inventory):
    """Classify every hour against the source's convention. Vectorised.

    Single source of truth for hour-level DQ (`adjustment_reason` is its scalar
    form). Test ORDER is load-bearing: restock must be asked before write-off,
    or a restock that sold out to exactly zero misreads as an unexplained zero.
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

    Returns a frame indexed by episode_id with `opening`, `arrived`,
    `vanished`, `supply`, `sold`, `clearance`, `leftover`, `scrap` and the
    eligibility flags. `clearance` cannot exceed 1 by construction, because
    supply (= opening + arrived) counts everything that arrived (design 12a).
    """
    d = d.sort_values(["date", "hour_of_day"])
    disc = ((d.starting_inventory.to_numpy() - d.ending_inventory.to_numpy())
            - d.units_sold.to_numpy())
    # `d` is in time order, so the last occurrence of each id is its final hour.
    is_last_row = ~d.episode_id.duplicated(keep="last").to_numpy()
    # the write-off exemption lives in shrink_by_hour, the one home; `disc`
    # keeps its negatives here because `arrived` is read off them below
    status = hour_status(d.starting_inventory, d.units_sold, d.ending_inventory)
    disc = np.where((status == WRITE_OFF) & is_last_row, 0, disc)

    g = pd.DataFrame({"episode_id": d.episode_id.to_numpy(), "disc": disc,
                      "shrink": shrink_by_hour(d.starting_inventory,
                                               d.units_sold,
                                               d.ending_inventory, is_last_row),
                      "sold": d.units_sold.to_numpy(),
                      "start": d.starting_inventory.to_numpy()})
    agg = g.groupby("episode_id", sort=False).agg(
        net=("disc", "sum"), sold=("sold", "sum"), opening=("start", "first"))
    # `arrived`/`vanished` are GROSS, never netted: netting a shortfall against
    # a same-size restock would hide both real events (design 12a).
    agg["arrived"] = (-g[g.disc < 0].groupby("episode_id", sort=False).disc.sum()
                      ).reindex(agg.index).fillna(0).astype("int64")
    agg["vanished"] = g.groupby("episode_id", sort=False).shrink.sum(
        ).reindex(agg.index).fillna(0).astype("int64")
    agg["supply"] = agg.opening + agg.arrived
    agg["clearance"] = np.divide(
        agg.sold.to_numpy(), agg.supply.to_numpy(),
        out=np.zeros(len(agg)), where=agg.supply.to_numpy() > 0)

    # Leftover at the close: `ending_inventory` (the authoritative count),
    # except on a write-off row, where the source zeroed it and clipped
    # `starting - sold` is what was written off.
    last = d.groupby("episode_id", sort=False).tail(1)
    last_status = hour_status(last.starting_inventory, last.units_sold,
                              last.ending_inventory)
    agg["leftover"] = pd.Series(
        np.where(last_status == WRITE_OFF,
                 np.clip(last.starting_inventory.to_numpy()
                         - last.units_sold.to_numpy(), 0, None),
                 last.ending_inventory.to_numpy()),
        index=last.episode_id.to_numpy()).reindex(agg.index).to_numpy()

    # Scrap = leftover + shrink: both are units paid for with no revenue.
    agg["scrap"] = agg.leftover + agg.vanished

    # THE EPISODE IDENTITY: opening + restocked == sold + scrap (equivalently
    # supply == sold + leftover + shrink). It balances or the arithmetic is
    # wrong; `flow_identity_violations` enforces it.
    agg["accounting_closes"] = agg.supply == agg.sold + agg.scrap

    # Final hour clean: `starting >= sold` on the last row. `sold > starting`
    # proves a final-hour restock, making the scrap quantity unknowable.
    agg["final_hour_clean"] = pd.Series(
        (last.starting_inventory.to_numpy() >= last.units_sold.to_numpy()),
        index=last.episode_id.to_numpy()).reindex(agg.index).to_numpy()

    # Closed: `ending_inventory == 0` on the last row. Without it the window
    # was still running when the extract stopped, so `leftover` may yet sell.
    agg["closed"] = pd.Series(
        (last.ending_inventory.to_numpy() == 0),
        index=last.episode_id.to_numpy()).reindex(agg.index).to_numpy()

    # ELIGIBLE, stated once: reconciles AND final-hour clean AND closed.
    # NOT dp_eligible -- the DP has further requirements of its own.
    agg["eligible"] = agg.accounting_closes & agg.final_hour_clean & agg.closed
    return agg.drop(columns=["net"])


def hour_adjustment(d):
    """Per-hour EXOGENOUS inventory change: `+n` arrived, `-n` went missing.

    Closes `q_next = q - sold + adjustment` into the source's own
    `ending_inventory`. Zero on a write-off row: the zeroed ending is a
    disposal, not stock leaving during the window. `d` must be in window order.
    """
    d = d.sort_values(["date", "hour_of_day"])
    status = hour_status(d.starting_inventory, d.units_sold,
                         d.ending_inventory)
    is_last = ~d.episode_id.duplicated(keep="last").to_numpy()
    disc = ((d.starting_inventory.to_numpy() - d.ending_inventory.to_numpy())
            - d.units_sold.to_numpy())
    return pd.Series(np.where((status == WRITE_OFF) & is_last, 0, -disc),
                     index=d.index)


def flow_identity_violations(d):
    """Episodes failing `opening + restocked == sold + shrink + leftover`.

    Not a tolerance check: on a continuous chain the identity is provable, so
    a violation means `episode_flow` has a bug. Returns the offending rows of
    `episode_flow`, empty when all is well.
    """
    flow = episode_flow(d)
    lhs = flow.opening + flow.arrived
    rhs = flow.sold + flow.scrap
    return flow[lhs != rhs]


def censored_hours(d):
    """Hours where demand was only observed as a LOWER bound. Frame in.

    Decided at the LAST ROW ONLY, where `starting == sold` is the whole test
    (true demand is `>= sold`). It cannot happen elsewhere -- the source stops
    emitting rows once inventory reaches zero; `censoring_off_last_row` checks.
    """
    order = ["date", "hour_of_day"]
    idx = d.sort_values(order).groupby("episode_id").tail(1).index
    is_last = pd.Series(False, index=d.index)
    is_last.loc[idx] = True
    return (is_last.to_numpy()
            & (d.starting_inventory.to_numpy() == d.units_sold.to_numpy()))


def is_censored_hour(starting_inventory, units_sold, ending_inventory):
    """Row-level censoring, for the LIVE path where there is no episode frame.

    Rows stop at zero inventory, so `starting == sold` with no restock IS the
    close; offline this coincides with `censored_hours`'s stronger form.
    """
    start = np.asarray(starting_inventory, dtype="int64")
    sold = np.asarray(units_sold, dtype="int64")
    status = hour_status(start, sold, ending_inventory)
    return (status == RECONCILES) & (start - sold == 0)


def censoring_off_last_row(d):
    """Rows where the shelf emptied but the episode carried on. Should be 0.

    Non-zero means the feed emits rows after inventory reached zero, and
    `censored_hours` is then deciding on the wrong row.
    """
    order = ["date", "hour_of_day"]
    idx = d.sort_values(order).groupby("episode_id").tail(1).index
    is_last = pd.Series(False, index=d.index)
    is_last.loc[idx] = True
    empty = d.starting_inventory.to_numpy() == d.units_sold.to_numpy()
    off = empty & ~is_last.to_numpy()
    return {
        "rows_shelf_emptied_mid_episode": int(off.sum()),
        "rows_with_zero_starting_inventory": int(
            (d.starting_inventory == 0).sum()),
        "note": ("Both should be zero: the source stops emitting rows once "
                 "inventory reaches zero, so an empty shelf ends the episode. "
                 "Non-zero means censoring is being decided on the wrong row."),
    }


def continuity_breaks(d):
    """Hours whose ending is not the next hour's starting, inside an episode.

    `ending` already carries any restock forward, so the chain has NO
    legitimate exception here. Returns a boolean mask aligned to `d`, False on
    each episode's last hour. `d` must be in window order.
    """
    nxt = d.groupby("episode_id")["starting_inventory"].shift(-1)
    return (nxt.notna() & (nxt != d.ending_inventory)).to_numpy()


def adjustment_reason(starting_inventory, units_sold, ending_inventory):
    """Why an outcome's inventory does not reconcile, or None. Scalar `hour_status`.

    restock: ending exceeds UNCLIPPED `starting - sold`. write-off: ending
    exactly zero with stock remaining -- recognised BY THE ZERO ITSELF, never
    by position in the episode (learnings.md). shrink: 0 < ending < net --
    named, NOT quarantined; it is an interpreted, monitorable event.
    """
    net = starting_inventory - units_sold
    if ending_inventory > net:
        return RESTOCK
    if ending_inventory == 0 and net > 0:
        return WRITE_OFF
    if 0 < ending_inventory < net:
        return SHORTFALL
    return None


def write_off_convention(last):
    """Is the source's closure sentinel present in this frame? DIAGNOSTIC ONLY.

    Never a fallback into `classify_last`: closure is read from the zero and
    nothing else, so a feed without the sentinel reports every episode
    unclosed -- loudly, which is the point (learnings.md).
    """
    left = leftover_units(last.starting_inventory, last.units_sold).to_numpy()
    return bool(((last.ending_inventory.to_numpy() == 0) & (left > 0)).any())


def _last_index(last):
    return last.episode_id.to_numpy() if "episode_id" in last else last.index


def classify_last(last):
    """Did the episode CLOSE, and with what, from a frame of FINAL rows.

    Closure is `ending_inventory == 0` on the last row, alone -- never
    `hours_remaining` (nominal), never a frame-wide fallback. NOT_CLOSED:
    ending != 0, scrap unknown. SOLD_OUT_EARLY: leftover == 0, censored.
    COMPLETED: leftover != 0, including the leftover < 0 final-hour-restock
    close, whose scrap quantity is unknowable and gated via `final_hour_clean`.
    """
    net = net_leftover(last.starting_inventory, last.units_sold).to_numpy()
    closed = last.ending_inventory.to_numpy() == 0
    return pd.Series(
        np.where(~closed, NOT_CLOSED,
                 np.where(net == 0, SOLD_OUT_EARLY, COMPLETED)),
        index=_last_index(last))


def classify(d):
    """Ending type per episode, indexed by episode_id."""
    return classify_last(last_rows(d))


def scrap_units(d):
    """Units scrapped per episode: the leftover at the close PLUS the shrink.

    NaN exactly where `~eligible` (not closed, dirty final hour, or identity
    does not balance) -- the figure cannot be trusted there. NaN propagates:
    sum with those dropped, never silently treated as zero.
    """
    flow = episode_flow(d)
    return flow.scrap.astype(float).where(flow.eligible, np.nan)


def extend_to_window(d, feature_cols=(), max_tail_hours=None):
    """Append the rows a window has but the data does not.

    Rows stop at zero inventory, so sold-out episodes are extended to their
    full window (`hours_remaining` down to 0) with synthetic rows marked
    `is_observed = False`. Exact: carried features are episode-constant or
    functions of the advancing timestamp. Synthetic rows carry no sales and
    must be filtered out of fidelity, likelihood and IL.
    """
    d = d.copy()
    d["is_observed"] = True
    last = last_rows(d)
    need = last[last.hours_remaining > 0]
    if not len(need):
        return d.sort_values(["episode_id", "date", "hour_of_day"])

    if max_tail_hours is not None:
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
