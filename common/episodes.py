"""What an episode was given, where it went, and what that means for scrap.

THE IDENTITY. Every unit an episode ever had has exactly one fate:

    opening + restocked  ==  sold + scrap        scrap = leftover + shrink

There is no third fate, so this is not a heuristic with a tolerance -- it
balances or the arithmetic is wrong. `episode_flow` computes it,
`flow_identity_violations` enforces it, and it has caught two bugs in this
module already.

THE SOURCE'S CONVENTION, which everything above rests on. `ending_inventory`
is the FINAL quantity on hand at the close of the hour, AFTER anything that
arrived during it. It is not `starting - sold`; it is what the source counted.
`hour_status` is the only place the four cases are written down, and the one
that surprises people is `units_sold > starting_inventory`: that is not an
impossible quantity, it is a RESTOCK, because `starting - sold` goes negative
and any ending at all exceeds it.

Across hours the chain is continuous -- `starting[t+1] == ending[t]` -- with
no exception, because `ending` already carries the restock forward. That is
the one hour-level rule that still drops an episode, and it is also why a
restock needs no special handling downstream: the DP simply sees a bigger `q`
when it re-solves at the next hour, exactly as it does in production.

TWO ENDINGS, AND ONE THAT IS NOT AN ENDING.

  sold_out_early   leftover is zero. Nothing left to scrap, by fact.
  completed        leftover > 0 on a closed episode. Those units were
                   disposed of; this is where essentially all scrap lives.
  not_closed       leftover > 0 and NO closure sentinel. NOT an ending -- the
                   episode is still running (or the feed cut it), so scrap is
                   unknown and excluded rather than counted as zero.

DATA QUIRK -- `ending_inventory` IS ZEROED ON AN EPISODE'S LAST ROW. The
source writes off whatever remains when a listing closes, so the last hour
breaks the chain by design. On the production extract 356,228 of 356,228 final
rows carry the zero and 48,280 (13.55%) still had stock on hand. Reading that
zero as scrap reports ZERO SCRAP EVERYWHERE -- IL collapses to discount cost
and nothing looks broken. So leftover is COMPUTED, never read. The exemption
applies to the LAST ROW ONLY: mid-episode a zero ending with stock still owed
is shrink, and exempting it there lost those units (282 of 4,000 random
continuous episodes failed the identity until this was scoped correctly).

DO NOT KEY ANY OF THIS TO `hours_remaining`. The counter is NOMINAL and still
positive on ~99.9% of final rows. An earlier version treated "counter reached
zero" as the end-of-window signal; it fired on ~0.1% of episodes and pushed
~99% of all real leftover into the unknown bucket. Confirmed with the
business: when a listing ends with stock on hand, those units are DISPOSED.

CENSORING is decided at the LAST ROW only, where `starting == sold` means the
shelf emptied. It cannot happen anywhere else -- the source stops emitting
rows once inventory reaches zero -- and `censoring_off_last_row` checks that
the feed keeps holding to it.

CLOSURE IS ONE CONDITION: `ending_inventory == 0` on the last row. That is the
source zeroing the field when a listing ends, and its absence means the window
was still running when the extract stopped. `write_off_convention` reports whether the
sentinel appears anywhere in a frame -- a DIAGNOSTIC only, never a fallback: a
feed with no sentinel is a broken feed, and every episode reads not_closed
until the feed says otherwise (docs/learnings.md).

Closure and OUTCOME are separate axes. Once closed, the sign of the UNCLIPPED
`starting - sold` on the last row says which close it was: positive is scrap,
zero is censored, negative means stock arrived during the final hour. Clipping
folds the third into the second, so a restocked close reads as a clean sell-out
-- see `net_leftover`.
"""

import numpy as np
import pandas as pd

# DID IT CLOSE -- decided by `ending_inventory == 0` on the last row, alone.
COMPLETED = "completed"
SOLD_OUT_EARLY = "sold_out_early"
NOT_CLOSED = "not_closed"

# WHAT THE CLOSE WAS -- decided by the SIGN of the unclipped last-row leftover.
# A separate axis from the one above, on purpose: an episode can be censored
# and unclosed at once, and one three-way label could not say so.
CLOSE_SCRAP = "scrap"                    # leftover > 0
CLOSE_CENSORED = "censored"              # leftover == 0
CLOSE_RESTOCK = "final_hour_restock"     # leftover < 0 -- same name as the DP gate


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


def net_leftover(starting_inventory, units_sold):
    """`starting_inventory - units_sold`, UNCLIPPED. Its SIGN is the answer.

    On a last row the three cases are exhaustive and mutually exclusive, and
    together they are the whole of what a close can be:

        > 0   stock was left. It is scrap.
        == 0  the shelf emptied through sales. CENSORED -- true demand is only
              known to be `>= sold`.
        < 0   the hour sold more than it opened with, so stock ARRIVED during
              it. How much arrived and how much was left are two unknowns with
              one equation once the ending has been zeroed.

    Clipping destroys the third case by folding it into the second, making a
    restocked close read as a clean sell-out -- hence UNCLIPPED.
    """
    start = np.asarray(starting_inventory, dtype=float)
    sold = np.asarray(units_sold, dtype=float)
    return pd.Series(start - sold, index=getattr(starting_inventory, "index", None))


def leftover_units(starting_inventory, units_sold):
    """Stock still on hand at the close of the last hour.

    `max(0, net_leftover)`. NOT `ending_inventory`, which the source zeroes on
    the last row when it writes the remainder off.

    EXACT except on a last hour that also took a restock, where the arriving
    quantity is not recoverable once the ending has been zeroed. Those
    episodes fail `final_hour_clean` and are gated out of `eligible`, so no
    figure rests on the ambiguity -- but read the sign from `net_leftover`,
    never from this, or the ambiguous case is invisible.
    """
    return net_leftover(starting_inventory, units_sold).clip(lower=0)


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
    by construction, because supply counts everything that arrived.
    """
    d = d.sort_values(["date", "hour_of_day"])
    status = hour_status(d.starting_inventory, d.units_sold, d.ending_inventory)
    disc = ((d.starting_inventory.to_numpy() - d.ending_inventory.to_numpy())
            - d.units_sold.to_numpy())
    # The write-off exemption applies to the LAST ROW ONLY. The source zeroes
    # `ending` when a listing closes, so that row's arithmetic must not be
    # read as stock vanishing -- but mid-episode a zero ending with stock
    # still owed is not a close, it is shrink, and exempting it loses those
    # units. Found by brute force: 282 of 4,000 randomly generated CONTINUOUS
    # episodes failed the identity, all of them because a mid-episode row was
    # being waved through as a write-off.
    # `d` is already in time order, so the last occurrence of each id is its
    # final hour.
    is_last_row = ~d.episode_id.duplicated(keep="last").to_numpy()
    disc = np.where((status == WRITE_OFF) & is_last_row, 0, disc)

    g = pd.DataFrame({"episode_id": d.episode_id.to_numpy(), "disc": disc,
                      "sold": d.units_sold.to_numpy(),
                      "start": d.starting_inventory.to_numpy()})
    agg = g.groupby("episode_id", sort=False).agg(
        net=("disc", "sum"), sold=("sold", "sum"), opening=("start", "first"))
    # GROSS, never netted. An hour where `ending` exceeds `starting - sold`
    # is a restock; an hour where it falls short is stock gone missing. Both
    # are real events the source is reporting, and they are counted as they
    # happen.
    #
    # GROSS, NEVER NETTED. Netting a shortfall against a same-size restock is
    # an inference dressed as arithmetic -- it reads a window with 2 restocked
    # and 2 shrunk as having neither, letting a restocked episode into the
    # DP-side population with its clearance priced against the wrong supply.
    # The gross figures are what the data says.
    agg["arrived"] = (-g[g.disc < 0].groupby("episode_id", sort=False).disc.sum()
                      ).reindex(agg.index).fillna(0).astype("int64")
    agg["vanished"] = g[g.disc > 0].groupby("episode_id", sort=False).disc.sum(
        ).reindex(agg.index).fillna(0).astype("int64")
    agg["supply"] = agg.opening + agg.arrived
    agg["clearance"] = np.divide(
        agg.sold.to_numpy(), agg.supply.to_numpy(),
        out=np.zeros(len(agg)), where=agg.supply.to_numpy() > 0)

    # THE STOCK LEFT AT THE CLOSE. `ending_inventory` is the authoritative
    # final count -- that is the whole convention -- so it is the answer
    # except on a write-off row, where the source zeroed it and
    # `starting - sold` is what was written off. Reading `starting - sold`
    # unconditionally was wrong on a last hour that RESTOCKED: one episode
    # opened its final hour with 27, sold 30 and ended holding 26, and the
    # subtraction gave 0.
    last = d.groupby("episode_id", sort=False).tail(1)
    last_status = hour_status(last.starting_inventory, last.units_sold,
                              last.ending_inventory)
    agg["leftover"] = pd.Series(
        np.where(last_status == WRITE_OFF,
                 np.clip(last.starting_inventory.to_numpy()
                         - last.units_sold.to_numpy(), 0, None),
                 last.ending_inventory.to_numpy()),
        index=last.episode_id.to_numpy()).reindex(agg.index).to_numpy()

    # SCRAP IS THE LEFTOVER PLUS THE SHRINK. Both are units the business paid
    # cost for and got no revenue from, which is the only thing scrap means.
    # Keeping shrink out of it left an episode's economics with a hole in the
    # middle -- units neither sold nor scrapped nor on the shelf -- and no way
    # to close the books on it.
    agg["scrap"] = agg.leftover + agg.vanished

    #
    #   THE EPISODE IDENTITY
    #
    #       opening + restocked  ==  sold + scrap
    #
    # equivalently  supply == sold + leftover + shrink. Every unit an episode
    # ever had has exactly one fate: it sold, or it is scrap. There is no
    # third, so this is not a heuristic with a tolerance -- it balances or the
    # arithmetic is wrong. `flow_identity_violations` enforces it.
    agg["accounting_closes"] = agg.supply == agg.sold + agg.scrap

    # THE FINAL HOUR MUST BE CLEAN: `starting >= sold` on the last row, so the
    # episode ends in exactly one of two states and nothing else --
    #
    #     sold == starting   the shelf emptied. CENSORED, no leftover.
    #     sold <  starting   `starting - sold` is left, and that is scrap.
    #
    # `sold > starting` on the last row proves stock arrived during it, and
    # then the close is ambiguous: if the source also zeroed `ending` to write
    # the remainder off, how much arrived and how much was scrapped are two
    # unknowns with one equation. The identity still balances -- it infers the
    # arrival from the ending -- but on an assumption rather than on evidence,
    # and a scrap figure resting on that should not reach IL.
    agg["final_hour_clean"] = pd.Series(
        (last.starting_inventory.to_numpy() >= last.units_sold.to_numpy()),
        index=last.episode_id.to_numpy()).reindex(agg.index).to_numpy()

    # AND THE EPISODE MUST HAVE CLOSED: `ending_inventory == 0` on the last
    # row. Without it the window was still running when the extract stopped,
    # so `sold` is sold-SO-FAR and `leftover` is stock still on the shelf that
    # may yet sell. The identity balances either way -- it is arithmetic on
    # what was observed -- which is exactly why closure has to be asked
    # separately.
    agg["closed"] = pd.Series(
        (last.ending_inventory.to_numpy() == 0),
        index=last.episode_id.to_numpy()).reindex(agg.index).to_numpy()

    # ELIGIBLE: three conditions, stated ONCE, here.
    #
    #   1. the episode reconciles      opening + restocked == sold + scrap
    #   2. the final hour is clean     starting - sold >= 0 on the last row
    #   3. the episode closed          ending_inventory == 0 on the last row
    #
    # All three were live before this; the third was assembled independently
    # at each consumer (`prepare_data` ANDed `outcome_known` in, `eda` and
    # `scrap_units` each re-derived it), which is three chances to forget it
    # and no single place to read the definition. It is one flag now.
    #
    # NOT the same as dp_eligible -- the DP has further requirements of its
    # own (a feasible tier, a horizon it can read, one inventory pool) that
    # say nothing about whether the demand observations are sound.
    agg["eligible"] = agg.accounting_closes & agg.final_hour_clean & agg.closed
    return agg.drop(columns=["net"])


def hour_adjustment(d):
    """Per-hour EXOGENOUS inventory change: `+n` arrived, `-n` went missing.

    The quantity that closes `q_next = q - sold + adjustment` into the
    source's own `ending_inventory`, and the reason a restocked episode needs
    no special handling anywhere. A replay does not have to MODEL a delivery;
    it applies what happened and re-solves, which is exactly what production
    does. The DP is never asked to anticipate stock arriving -- live it finds
    out at the next hour, because `ending[t]` is `starting[t+1]`, and in
    replay it finds out the same way.

    Zero on a write-off row: the source zeroing `ending` at the close is a
    disposal, not stock leaving during the window.

    `d` must be in window order.
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

    Every unit an episode ever had -- what it opened with, plus anything that
    arrived -- ends up in exactly one of three places: sold, shrunk, or still
    on the shelf at the last hour. There is no fourth option, so the identity
    is not a heuristic with a tolerance; it either balances or the arithmetic
    is wrong.

    And it IS the arithmetic that would be wrong, not the feed. Once the chain
    is continuous -- `starting[t+1] == ending[t]`, which `episode_universe`
    guarantees -- the two sides are provably equal, because the per-hour
    discrepancies telescope to exactly the last row's leftover. A violation
    therefore means `episode_flow` has a bug.

    That is worth checking rather than assuming: it caught one. Reading the
    last row's leftover as `starting - sold` is wrong on a final hour that
    restocked, and an episode that opened its last hour with 27, sold 30 and
    ended holding 26 came out as an anomaly it had no business being.

    Returns the offending rows of `episode_flow`, empty when all is well.
    """
    flow = episode_flow(d)
    lhs = flow.opening + flow.arrived
    rhs = flow.sold + flow.scrap
    return flow[lhs != rhs]


def censored_hours(d):
    """Hours where demand was only observed as a LOWER bound. Frame in.

    Decided at the LAST ROW ONLY, and `starting == sold` there is the whole
    test. It cannot happen anywhere else: the source stops emitting rows once
    inventory reaches zero -- which is why `extend_to_window` exists at all --
    so a shelf that empties ends the episode. Measured on the extract, 259 of
    259 rows with `starting == sold` are final rows, and no row anywhere has
    `starting_inventory == 0`. The last-row restriction is therefore a safety
    net over an invariant the feed already keeps, not a change of definition;
    `censoring_off_last_row` reports any row that breaks it.

    An eligible episode has `starting >= sold` on its last row, so the close
    is exactly one of two states and nothing else:

        sold == starting   the shelf emptied through sales. Whoever came next
                           bought nothing and left no trace, so true demand is
                           `>= sold`. CENSORED.
        sold <  starting   stock was left. Demand was observed exactly, and
                           the leftover is scrap. NOT censored.

    Four call sites carried `units_sold >= starting_inventory` independently
    -- m5, the dispersion fit, the prior fit and the live posterior update.
    All four marked every restock hour censored, which inflates demand on
    exactly the hours that had the most stock to sell.
    """
    order = ["date", "hour_of_day"]
    idx = d.sort_values(order).groupby("episode_id").tail(1).index
    is_last = pd.Series(False, index=d.index)
    is_last.loc[idx] = True
    return (is_last.to_numpy()
            & (d.starting_inventory.to_numpy() == d.units_sold.to_numpy()))


def is_censored_hour(starting_inventory, units_sold, ending_inventory):
    """Row-level censoring, for the LIVE path where there is no episode frame.

    `pipeline.update` sees one outcome per decision hour and cannot know
    whether it is an episode's last. It does not need to: rows stop at zero
    inventory, so `starting == sold` with no restock IS the close. Offline the
    two definitions coincide -- `censored_hours` asserts the stronger form.
    """
    start = np.asarray(starting_inventory, dtype="int64")
    sold = np.asarray(units_sold, dtype="int64")
    status = hour_status(start, sold, ending_inventory)
    return (status == RECONCILES) & (start - sold == 0)


def censoring_off_last_row(d):
    """Rows where the shelf emptied but the episode carried on. Should be 0.

    If this is ever non-zero the feed has started emitting rows after
    inventory reached zero, and `censored_hours` is then deciding on the wrong
    row -- so it is counted rather than assumed away.
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
    legitimate exception here -- unlike the within-hour test, which has three.
    A violation means the two fields disagree about the same instant, which no
    business event explains.

    Returns a boolean Series aligned to `d`, False on each episode's last hour
    (there is no next hour to disagree with). `d` must be in window order.
    """
    nxt = d.groupby("episode_id")["starting_inventory"].shift(-1)
    return (nxt.notna() & (nxt != d.ending_inventory)).to_numpy()


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
      shrink      ending is ABOVE zero but below the leftover. Stock left
                  without being sold and without being written off.

    Keying the write-off to "our last observed hour" was wrong and quarantined
    real outcomes in bulk: the source zeroes at ITS episode boundary, and once
    a window is merged across midnight that row sits in the MIDDLE of ours.
    Position is our bookkeeping; the zero is the source's fact.

    SHRINK IS NAMED, NOT QUARANTINED. The live path treats shrink exactly as
    the offline chain does: an ordinary event, counted gross, booked into
    scrap, gating nothing. The reason is on the event, `units_shrink` and
    `episode_scrap` carry the quantity, and the rate is monitorable.
    Quarantine is for what the system CANNOT interpret, and a shrink is
    interpreted (docs/learnings.md on the shadow gate this consistency fixed).
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

    A final row reporting `ending_inventory == 0` while stock remained can only
    be a write-off, so a single such row proves the convention is in force.

    A DIAGNOSTIC, never a fallback into `classify_last`: closure is read from
    the zero and nothing else, so a feed without the sentinel reports every
    episode unclosed -- loudly, which is the point. Read this before believing
    that number.
    """
    left = leftover_units(last.starting_inventory, last.units_sold).to_numpy()
    return bool(((last.ending_inventory.to_numpy() == 0) & (left > 0)).any())


def _last_index(last):
    return last.episode_id.to_numpy() if "episode_id" in last else last.index


def close_outcome(last):
    """What the close WAS, from the sign of the unclipped leftover.

    Orthogonal to `classify_last`, which answers whether the episode closed at
    all -- the two facts are independent axes, so, say, a censored close on a
    window still running is expressible.
    """
    net = net_leftover(last.starting_inventory, last.units_sold).to_numpy()
    return pd.Series(np.where(net > 0, CLOSE_SCRAP,
                              np.where(net == 0, CLOSE_CENSORED,
                                       CLOSE_RESTOCK)),
                     index=_last_index(last))


def classify_last(last):
    """Did the episode CLOSE, and with what, from a frame of FINAL rows.

    ONE condition decides closure: `ending_inventory == 0` on the last row.
    The source zeroes that field when a listing ends, whatever remained, so the
    zero IS the closure and its absence means the window is still running at
    the edge of this extract. Nothing else is consulted -- not
    `hours_remaining` (nominal, still positive on ~99.9% of final rows), not
    proximity to the extract's last timestamp, and never a frame-wide fallback
    for feeds missing the sentinel.

    The leftover then says which kind of close it was, and it is read UNCLIPPED
    because the sign carries the whole answer:

        NOT_CLOSED       ending != 0. Still running; scrap unknown.
        SOLD_OUT_EARLY   closed, leftover == 0. The shelf emptied. CENSORED.
        COMPLETED        closed, leftover != 0. The remainder was written off.

    COMPLETED covers the `leftover < 0` close too -- stock arrived during the
    final hour -- because it did close, and refusing to say so would be a
    second lie on top of the unknown quantity. What is NOT knowable there is
    HOW MUCH was scrapped, and that is gated separately and everywhere it
    matters: `final_hour_clean` is False, so `eligible`, `scrap_units` and the
    DP population all exclude it. `close_outcome` names it CLOSE_RESTOCK.
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

    Both are units the business paid cost for and got no revenue from, which
    is the only thing scrap means. Keeping shrink out left an episode's
    economics with a hole in the middle -- units neither sold, nor scrapped,
    nor on the shelf -- and no way to close the books. With it in, the
    identity holds: `supply == sold + scrap`.

    NaN where the number cannot be trusted, which is exactly `~eligible` --
    the same three conditions, not a second list that has to be kept in step:

      NOT CLOSED         `ending != 0` on the last row. The episode did not
                         finish inside this data, so the stock is on the shelf
                         rather than in the bin.
      DIRTY FINAL HOUR   the last row sold more than it opened with, proving
                         stock arrived during it. If the source also zeroed
                         `ending` to write the remainder off, how much arrived
                         and how much was scrapped are two unknowns with one
                         equation, and any answer is a guess.
      DOES NOT RECONCILE the identity does not balance, so `scrap` would be
                         the residue of arithmetic already known to be wrong
                         -- an unbalanced episode never yields a
                         confident-looking figure.

    NaN propagates: a sum over a frame containing such episodes must be taken
    with those dropped, not silently treated as zero. Callers that report an
    excluded count -- `bootstrap.measure.m6_il_pct` does -- keep the exclusion
    visible instead of quietly shrinking the population.
    """
    flow = episode_flow(d)
    return flow.scrap.astype(float).where(flow.eligible, np.nan)


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
    outcome = close_outcome(last).value_counts()
    n = max(len(last), 1)
    return {
        "episodes": int(len(last)),
        # if this is FALSE the source is not marking closure at all, and every
        # episode below will read not_closed -- deliberately, with no
        # fallback. Read this FIRST: false here means the feed changed or the
        # extract is wrong, not that nothing ever closed.
        "write_off_convention_in_force": write_off_convention(last),
        # the outcome axis, independent of closure: scrap / censored /
        # final_hour_restock by the SIGN of the unclipped last-row leftover
        "close_outcomes": {str(k): int(v) for k, v in outcome.items()},
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
