"""pricing -- the hourly price and the morning table it joins on, standalone.

    hour.py       the hourly command: the snapshot in, a price per shelf out
    features.py   the morning command: yesterday's feed in, the day's table out
    state.py      a validated request becomes the engine's state
    decide.py     the state is judged, solved and recorded as a decision event
    dp.py         the monotone DP over the tier grid; demand.py its NB demand
    model.py      the frozen demand model, applied at the reference discount
    store.py      the append-only record: decisions and refusals, one per shelf-hour
    feed.py       the feed's schema and files; keys.py the one spelling of every key
    rule.py       the producers' episode-id rule, evaluated only to count disagreements
    config.py     the pruned config, the launch gate, the category anchors
    pool.py       the worker pool for an hour's batch

Exploit only: every request is priced at the engine's optimal price; nothing
is drawn and no budget is read. The learning lane (outcomes, the posterior
update, the monitor) runs in the owner's repository and reads the store this
package writes; the decision event carries every field it needs.
"""
