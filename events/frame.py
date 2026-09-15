"""events.frame -- live events as the prepared-frame vocabulary.

The one adapter from matched (decision, outcome) pairs to the hourly rows
common.metrics reads, and the one settled episode frame built from them:
the business metrics, the guardrail series and the tau controller's IL
base all read `settled_episodes`, so floor, trigger and budget measure
one thing (design 5.12). Records in, a frame out -- nothing here prices
or learns.
"""

import pandas as pd

from common import metrics
from events.pairs import decision_day, match_pairs


def event_frame(decisions, outcomes, pairs=None):
    """Matched (decision, outcome) pairs as HOURLY rows in the prepared-frame
    vocabulary, so metrics.episode_economics is the one episode-grain
    definition on live events too -- floor and trigger measure one thing.
    `pairs` is events.pairs.match_pairs(decisions, outcomes) if the caller
    already built it."""
    if pairs is None:
        pairs = match_pairs(decisions, outcomes)
    return pd.DataFrame([{
        "episode_id": d["episode_id"], "date": decision_day(d),
        "hour_of_day": d["hour_of_day"],
        "category": d.get("category"), "fc": d["fc"], "sku_id": d["sku_id"],
        "original_price": d["original_price"], "offered_price": o["applied_price"],
        "cost": d["cost"], "starting_inventory": o["starting_inventory"],
        "units_sold": o["units_sold"], "ending_inventory": o["ending_inventory"],
    } for d, o in pairs])


def settled_episodes(decisions, outcomes, pairs=None):
    """The ONE episode frame the business metrics, the guardrail series and
    the tau controller's IL base all read: metrics.settled over
    metrics.episode_economics on the live pairs. (settled frame, exclusion
    counts), or None when no outcome matches a decision -- built once per
    run (daily.monitor.build_report) and passed down, never per metric
    family."""
    df = event_frame(decisions, outcomes, pairs)
    if df.empty:
        return None
    return metrics.settled(metrics.episode_economics(df))


def il_by_close_day(episodes):
    """Realised IL by CLOSE DAY over a `settled_episodes` result: the
    trailing base the tau controller prices a day's budget from
    (engine.budget.trailing_daily_il). Settled episodes only -- an open one
    contributes nothing until it closes, which is what makes the base
    knowable at the start of each day. {} when there is no frame."""
    if episodes is None:
        return {}
    ep, _ = episodes
    return {str(k): round(float(v), 2) for k, v
            in ep.groupby("close_day").il.sum().items()}
