"""ops.price_batch -- price one batch of requests through engine.decide.

Lane B's reference caller (design 5.10, RUNBOOK "Lane B"): the 12-field
requests of one hour in, a decision per request out -- the price to apply,
or a rejection with its reason, never a best-effort price. Everything the
engine needs beyond the request is resolved here, by the one home for
each: the state (engine.state.build_states -- the frozen model's
`mu_ref_path`: a fresh forecast for an entry request on the two
demand-rate features computed point-in-time over the trailing history, the
stored forecast sliced for a later hour of a known episode; `r` from the
lookup), the posterior and tau (ONE `PosteriorStore` read per batch, so a
suspension or an --apply landed by another process is seen before the
first decision), the config digest (computed once per batch), the event
store (every priced decision is committed before its price is returned --
an hour that is not in the record is not priced; the store itself refuses
a second decision for a priced hour).

Row-scoped, never batch-scoped: a request that cannot become a state, a
state the engine rejects, a duplicate of another request's hour, or an
hour the store already holds a decision for, costs that request and is
returned as `rejected` with the reason; the rest of the batch prices.
Every request is read in one spelling (engine.state.canonical_request),
so a parquet timestamp or an id read back as 7.0 prices and writes.

Run: python3 -m ops.price_batch --requests <hour.jsonl|.parquet|.csv>
        --history <hourly FLC parquet, the table ingest reads>
        --out decisions.jsonl [--report report.json] [--workers N] [--seed S]
"""

from collections import Counter


from common.parallel import map_episodes
from engine.state import (batch_context, build_states, table_as_of, canonical_request, price_one, validate_request)
from events.contract import rejection_event
from events.pairs import colliding_keys, hour_key, ident_series
from events.store import EventStore
from fit.artifacts import load_bundle

# what a caller gets back per request: the hour, the id the outcome will
# name, the price to put on the shelf -- or why there is none
RESPONSE_FIELDS = ("episode_id", "sku_id", "fc", "date", "hour_of_day",
                   "decision_id", "applied_discount", "applied_price",
                   "is_exploration", "rejected")


# ----------------------------------------------------------------- inputs

# `load_history` is engine.state's (daily.features reads it every morning);
# the name stays here for callers


def plan(requests, priced_keys, cfg):
    """Which requests become states, and which are refused before the
    engine sees them: a request missing what a state needs
    (validate_request), two requests for one hour (two states for one
    hour -- neither is preferable, the rule ingest applies to the feed:
    events.pairs.colliding_keys), and an hour the store already holds a
    decision for (`priced_keys`, the store's own index; a retry of a
    priced hour would put two decisions on one feed row). Returns
    (to_price [(index, canonical request, key)], rejected {index: reason})
    -- the requests handed on are in the one spelling
    (engine.state.canonical_request)."""
    rejected, keyed = {}, []
    for i, r in enumerate(requests):
        problems = validate_request(r, cfg)
        if problems:
            rejected[i] = "; ".join(problems)
            continue
        c = canonical_request(r)
        keyed.append((i, c, hour_key(c["sku_id"], c["fc"], c["date"], c["hour_of_day"])))
    twice = colliding_keys(k for _, _, k in keyed)
    to_price = []
    for i, r, k in keyed:
        if k in twice:
            rejected[i] = "duplicate_request: two requests for one hour"
        elif k in priced_keys:
            rejected[i] = "already_priced: the store holds a decision for this hour"
        else:
            to_price.append((i, r, k))
    return to_price, rejected


# ------------------------------------------------------------------ batch

# THIS FOLDER'S PHASE: exploit only. Every request is priced at p*; nothing
# is drawn and tau is not read, so each decision records tau_current null
# the way a suspended day does and the learning lane holds those days.
# The budgeted draw is the repository's; this copy is the integration
# phase -- the hourly loop tested alone, before any learning.
EXPLOIT_ONLY = True


def run(cfg, requests, history=None, workers=None, seed=0, store=None, model=None,
        posterior=None, r_lookup=None, features=None):
    """Price `requests` (dicts in REQUEST_FIELDS) against the day's feature
    table `features` (engine.state.ref_rate_table, the production path:
    one join, no history read) or, without one, against `history`
    (HISTORY_COLS, the features computed here). Returns (rows, events, report): one response row per
    request in request order (RESPONSE_FIELDS), the committed decision
    events, and the batch's counts. `store`, `model`, `posterior`,
    `r_lookup` are built from `cfg` unless a long-lived caller holds them
    (fit.artifacts.load_bundle: a missing lookup is a FileNotFoundError
    naming its path); the posterior is RELOADED here regardless -- one
    read per batch."""
    bundle = load_bundle(cfg)
    store = store or EventStore(cfg)
    model = model or bundle.model
    posterior = (posterior or bundle.posterior).reload()
    r_lookup = r_lookup or bundle.r_lookup

    to_price, rejected = plan(requests, store.priced_hours, cfg)
    canon = {i: r for i, r, _ in to_price}

    # the history the batch's features read: this batch's SKUs only (the
    # trailing table is every SKU ever fed; the features are per SKU),
    # matched in the one id spelling whatever dtype the caller's table has
    skus = {r["sku_id"] for r in canon.values()}
    if features is None and history is None:
        raise ValueError("a batch prices against the day's feature table or a history")
    mine = (history[ident_series(history.sku_id).isin(skus)]
            if history is not None and len(history) else history)
    states, notes = build_states(list(canon.values()), mine, cfg, model, r_lookup,
                                 episode_paths=store.episode_paths, features=features)
    # an entry request priced on a table built for an EARLIER day read the
    # features as of that day, not its own: counted, never refused (a day
    # late is nearly the same window; a week late is a cron that stopped)
    as_of = table_as_of(features)
    stale = sum(1 for r in canon.values()
                if r["current_discount"] is None and as_of is not None and as_of < r["date"])
    # the one context every worker reads (engine.state.batch_context): the
    # cells in category order, so the report's posterior_versions read so
    cats = sorted({r["category"] for r in canon.values()})
    # tau settled here as None (batch_context's `fixed`): no draw, and the
    # posterior's tau -- exploration.tau_initial -- is never read
    ctx = batch_context(cfg, posterior, model, cats, seed, tau=None)
    mode = "exploit" if EXPLOIT_ONLY else "explore"
    results = map_episodes(price_one, [(s, k) for s, (_, _, k) in zip(states, to_price)],
                           ctx, workers=workers)

    priced, engine_rejected, quarantined = {}, Counter(), 0
    for (i, r, k), res in zip(to_price, results):
        if res["evt"] is None:
            rejected[i] = res["rejected"]
            engine_rejected[res["rejected"]] += 1
            continue
        if not store.emit_decision(res["evt"]):
            # the store refused the event and says why (another run
            # committed the hour between plan() and here, or quarantine):
            # an hour not in the record is not priced
            rejected[i] = store.last_refusal or "the store refused the decision event"
            quarantined += 1
            continue
        priced[i] = res["evt"]              # request order: to_price is

    # every refused request is RECORDED as a rejection (events.store:
    # seen, not priced), so the hour is not mistaken later for one
    # engineering never sent. The store itself skips an hour it holds a
    # decision for, which is what `already_priced` means.
    rejections, recorded = 0, set()
    for i, reason in sorted(rejected.items()):
        evt = rejection_event(canon.get(i, requests[i]), reason)
        # one record per shelf-hour: two requests for one hour
        # (duplicate_request) are one refused shelf-hour, not a duplicate
        if evt is None or evt["rejection_id"] in recorded:
            continue
        recorded.add(evt["rejection_id"])
        if store.emit_rejection(evt):
            rejections += 1

    rows = []
    for i, r in enumerate(requests):
        base = {f: canon.get(i, r).get(f)
                for f in ("episode_id", "sku_id", "fc", "date", "hour_of_day")}
        evt = priced.get(i)
        rows.append({**base, **{f: (evt[f] if evt else None) for f in
                                ("decision_id", "applied_discount", "applied_price",
                                 "is_exploration")},
                     "rejected": None if evt else rejected[i]})
    events = list(priced.values())
    report = {
        "requests": len(requests),
        "decisions": len(events),
        "rejected": len(requests) - len(events),
        "rejected_before_the_engine": len(requests) - len(to_price),
        "rejected_by_the_engine": dict(engine_rejected),
        # refused hours RECORDED as seen-not-priced, so next hour's rule can
        # step from them: fewer than `rejected` when a refusal names no
        # shelf-hour, or the hour already holds a decision (`already_priced`)
        "rejections_recorded": rejections,
        "quarantined": quarantined,
        "explored": sum(1 for e in events if e["is_exploration"]),
        # a fresh forecast with no history behind it prices as "unknown";
        # a batch where every request reads so is a history that does not
        # meet its requests (an id spelling, a table cut too short)
        **notes,
        "features_as_of": as_of,
        "entry_requests_on_stale_features": int(stale),
        "tau_in_force": None if ctx["suspended"] else ctx["tau"],
        "exploration_suspended": ctx["suspended"],
        "exploration_mode": mode,
        "model_version": ctx["model_version"],
        "config_digest": ctx["digest"],
        "posterior_versions": {c: int(v["version"]) for c, v in ctx["cells"].items()},
        "history_rows": int(len(history)) if history is not None else 0,
        "history_dates": ([str(history.date.min()), str(history.date.max())]
                          if history is not None and len(history) else None),
    }
    return rows, events, report

