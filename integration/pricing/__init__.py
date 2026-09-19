"""pricing -- the hourly price and the morning table it joins on, standalone
and stateless.

    hour.py       the hourly command: the snapshot in, a price per shelf out
    features.py   the morning command: yesterday's feed in, the day's table out
    state.py      a snapshot row becomes the engine's state, from the row alone
    decide.py     the state is judged, solved and recorded as a decision event
    dp.py         the monotone DP over the tier grid; demand.py its NB demand
    model.py      the frozen demand model, applied at the reference discount
    log.py        the append-only log of decisions and refusals, never read here
    feed.py       files and names: the extract's columns, the snapshot, the response; keys.py the one spelling of every key
    config.py     the pruned config, the launch gate, the category anchors
    pool.py       the worker pool for an hour's batch

Stateless: a row is priced from itself, the artifacts and the feature table
of its episode's opening day. The row's `episode_id` is the opening tag
(`<sku>|<fc>|<day>T<hh>`), so an entry is the row whose tag is its own
shelf-hour; a later hour is anchored on the price in force the row carries,
the price applied last hour piped back by the producers. Exploit only: every
request is priced at the engine's optimal price; nothing is drawn and no
budget is read. The learning lane (outcomes, the posterior update, the
monitor) runs in the owner's repository and reads the log this package
appends to; the decision event carries every field it needs.
"""
