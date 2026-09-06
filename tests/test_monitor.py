"""daily.monitor: the live guardrail (persistence over calendar days), the
series it triggers on, the business metrics, and the safety metrics -- all
read off the one episode frame."""

import pandas as pd
import pytest


def test_guardrail_fires_only_after_persistence():
    """The owner thresholds must actually be evaluated -- and must not fire on
    a single day over, which is what the noise floor makes routine."""
    from daily.monitor import evaluate_guardrail

    block = {"basis": "trailing_28d_mean", "latest": 0.30,
             "by_day": {"2026-09-01": 0.05, "2026-09-02": 0.30}}

    one_day = evaluate_guardrail(block, threshold=0.20, persistence_days=2)
    assert not one_day["fired"] and one_day["consecutive_days_over"] == 1

    block["by_day"]["2026-09-03"] = 0.25
    two_days = evaluate_guardrail(block, threshold=0.20, persistence_days=2)
    assert two_days["fired"] and two_days["consecutive_days_over"] == 2

    # a day back under the threshold breaks the streak
    block["by_day"]["2026-09-04"] = 0.01
    assert not evaluate_guardrail(block, 0.20, 2)["fired"]

    # a null threshold is blocked, never silently passing
    blocked = evaluate_guardrail(block, None, 2)
    assert not blocked["fired"] and "BLOCKED" in blocked["status"]
    # one result shape, whatever the branch
    for r in (one_day, two_days, blocked):
        assert set(r) >= {"fired", "threshold", "persistence_days", "basis",
                          "latest", "status"}


def test_persistence_counts_calendar_days_not_observed_days():
    """Two days over with a silent day between them are not two CONSECUTIVE
    days: a missing reading is not a reading over the threshold."""
    from daily.monitor import evaluate_guardrail

    gap = {"by_day": {"2026-09-01": 0.30, "2026-09-03": 0.30}}
    r = evaluate_guardrail(gap, threshold=0.20, persistence_days=2)
    assert not r["fired"] and r["consecutive_days_over"] == 1

    contiguous = {"by_day": {"2026-09-02": 0.30, "2026-09-03": 0.30}}
    assert evaluate_guardrail(contiguous, 0.20, 2)["fired"]

    # the streak is counted back from the LATEST day only
    stale = {"by_day": {"2026-09-01": 0.30, "2026-09-02": 0.30,
                        "2026-09-05": 0.30}}
    assert evaluate_guardrail(stale, 0.20, 2)["consecutive_days_over"] == 1


def test_overspend_reads_zero_on_a_priced_day_with_no_forced_spend(cfg):
    """After a stop suspended exploration, the following days have NO forced
    spend. Without a reading for them the streak was still counted from the
    last two over-budget days, and the next --feed re-suspended what a human
    had just resumed. A priced day with nothing spent is a day at 0."""
    from daily.monitor import overspend_series, evaluate_guardrail
    cfg["exploration"]["budget_il_window_days"] = 1      # a one-day base reads
    il = {f"2026-09-{d:02d}": 1000.0 for d in range(1, 8)}
    learning = {"posterior_by_cell": {"GLOBAL": {"std": 0.5}},
                "cell_of": {"MEAT": "GLOBAL"},
                "exploration_cost_by_day": {"2026-09-05": 1e6, "2026-09-06": 1e6},
                "priced_days": ["2026-09-05", "2026-09-06", "2026-09-07"],
                "latest_priced_day": "2026-09-07"}
    series = overspend_series(learning, {"il_by_close_day": il}, cfg)
    assert series["by_day"]["2026-09-07"] == 0.0 and series["latest"] == 0.0
    assert not evaluate_guardrail(series, 2.0, 2)["fired"]
    # the two over days alone still fire
    learning["priced_days"] = learning["priced_days"][:2]
    learning["latest_priced_day"] = "2026-09-06"
    fired = evaluate_guardrail(overspend_series(learning, {"il_by_close_day": il}, cfg),
                               2.0, 2)
    assert fired["fired"] and fired["consecutive_days_over"] == 2


def test_business_metrics_counts_shrink_like_the_guardrail_and_il_pct_do(cfg):
    """Scrap = leftover + shrink has ONE definition. business_metrics read
    leftover only, so IL, waste_units, sell-through -- the pilot's read and
    production's exploration-budget base -- were all a different IL from the
    one the noise floors are measured on."""
    from daily.monitor import business_metrics, guardrail_series

    hours = [(3, 6, 1, 3),      # 6 - 1 = 5 expected, 3 reported -> shrink 2
             (2, 3, 1, 2),
             (1, 2, 1, 0)]      # write-off row: leftover 1, NOT shrink
    decisions, outcomes = [], []
    for hr, start, sold, end in hours:
        decisions.append({
            "decision_id": f"b{hr}", "episode_id": "EP-B", "sku_id": "s",
            "fc": "f", "category": "VEG", "subcategory": "LEAFY",
            "date": "2026-08-19", "hour_of_day": 24 - hr,
            "hours_remaining": hr, "original_price": 1000.0, "cost": 100.0})
        outcomes.append({
            "decision_id": f"b{hr}", "starting_inventory": start,
            "units_sold": sold, "ending_inventory": end,
            "applied_price": 800.0})

    b = business_metrics(decisions, outcomes)
    g = guardrail_series(decisions, outcomes, cfg)
    # leftover 1 + shrink 2 = 3 units of scrap, on both sides
    assert b["waste_units"] == 3
    assert g["daily_scrap_rate"]["2026-08-19"] == pytest.approx(3 / 6)
    # IL charges cost x scrap, not cost x leftover
    assert b["il_pct_aggregate"]["il_absolute"] == pytest.approx(
        3 * 200.0 + 3 * 100.0)
    assert b["sell_through"] == pytest.approx(3 / 6)


def test_the_guardrail_fires_on_a_catalogue_wide_deterioration(cfg):
    """The basis is the trailing mean of the same system-priced episodes:
    there is no control arm, so a catalogue-wide scrap doubling must show as
    a positive deterioration (an arm comparison of two system-priced halves
    once cancelled it to exactly zero)."""
    from daily.monitor import guardrail_series

    # the trailing basis needs guardrail_noise_window_days of history before
    # it produces a comparison at all, plus the smoothing shift
    span = cfg["monitoring"]["guardrail_noise_window_days"] + 20
    decisions, outcomes, i = [], [], 0
    for day in range(span):
        # scrap doubles for the last stretch, across the WHOLE catalogue
        sold, end = (2, 0) if day < span - 10 else (1, 0)
        for unit in range(40):
            i += 1
            decisions.append({
                "decision_id": f"g{i}", "episode_id": f"E{i}",
                "sku_id": f"S{unit}", "fc": "F1", "category": "VEG",
                "subcategory": "LEAFY", "hours_remaining": 1, "hour_of_day": 23,
                "cost": 100.0, "original_price": 1000.0,
                "timestamp": (pd.Timestamp("2026-06-01")
                              + pd.Timedelta(days=day)).isoformat() + "+00:00",
                "date": str((pd.Timestamp("2026-06-01")
                             + pd.Timedelta(days=day)).date())})
            outcomes.append({
                "decision_id": f"g{i}", "starting_inventory": 4,
                "units_sold": sold, "ending_inventory": end,
                "applied_price": 800.0})

    pre = guardrail_series(decisions, outcomes, cfg)
    assert pre["scrap_deterioration"]["basis"].startswith("trailing_")
    assert pre["scrap_deterioration"]["latest"] > 0, pre["scrap_deterioration"]


def test_the_guardrail_series_is_keyed_on_the_trading_day(cfg):
    """An hour-23 decision on day D carries a UTC timestamp that may read as
    D+1. Bucketing on the wall clock split the daily scrap series at
    midnight UTC while IL and the budget were keyed on the trading day."""
    from daily.monitor import guardrail_series

    decisions = [{
        "decision_id": "late", "episode_id": "EP-L", "sku_id": "s", "fc": "f",
        "date": "2026-08-19", "hour_of_day": 23,
        "timestamp": "2026-08-20T02:00:00+00:00",      # UTC is already D+1
        "hours_remaining": 1, "cost": 100.0, "original_price": 1000.0}]
    outcomes = [{"decision_id": "late", "starting_inventory": 4,
                 "units_sold": 1, "ending_inventory": 0,
                 "applied_price": 500.0}]
    g = guardrail_series(decisions, outcomes, cfg)
    assert list(g["daily_scrap_rate"]) == ["2026-08-19"]


def test_daily_rates_are_keyed_on_the_close_day_not_the_opening_day(cfg):
    """Bucketed by OPENING day over settled episodes, the newest days hold
    only the episodes that closed early (sold out: low scrap) -- the
    long-running ones are still open -- so the series read as improving
    exactly where the persistence rule evaluates. A close-day bucket is
    complete once its episodes settle."""
    from common import metrics
    from conftest import episode_frame

    # two episodes open on the 19th: one sells out that day, one runs into
    # the 20th and writes off two units there
    d = episode_frame(
        episode_id=["quick", "quick", "slow", "slow", "slow"],
        date=["2026-08-19", "2026-08-19", "2026-08-19", "2026-08-19", "2026-08-20"],
        hour_of_day=[20, 21, 20, 21, 1],
        starting_inventory=[2, 1, 4, 3, 2],
        units_sold=[1, 1, 1, 1, 0],
        ending_inventory=[1, 0, 3, 2, 0],
        original_price=1000.0, offered_price=800.0, cost=100.0)
    ep, _ = metrics.settled(metrics.episode_economics(d))
    day = metrics.daily_rates(ep)
    assert list(day.index) == ["2026-08-19", "2026-08-20"]
    # the 19th carries only the sold-out episode; the slow one lands where
    # its scrap became known
    assert day.loc["2026-08-19", "scrap"] == 0 and day.loc["2026-08-19", "opening"] == 2
    assert day.loc["2026-08-20", "scrap"] == 2 and day.loc["2026-08-20", "opening"] == 4
    assert day.loc["2026-08-20", "scrap_rate"] == pytest.approx(0.5)
    # the monitor's series is the same one, on the same key
    from daily.monitor import guardrail_series
    decisions, outcomes = [], []
    for i, row in d.iterrows():
        decisions.append({"decision_id": f"d{i}", "episode_id": row.episode_id,
                          "sku_id": "s", "fc": "f", "category": "VEG",
                          "date": row.date, "hour_of_day": int(row.hour_of_day),
                          "cost": row.cost, "original_price": row.original_price})
        outcomes.append({"decision_id": f"d{i}",
                         "starting_inventory": int(row.starting_inventory),
                         "units_sold": int(row.units_sold),
                         "ending_inventory": int(row.ending_inventory),
                         "applied_price": row.offered_price})
    g = guardrail_series(decisions, outcomes, cfg)
    assert g["day_key"] == "close_day"
    assert g["daily_scrap_rate"] == {"2026-08-19": 0.0, "2026-08-20": 0.5}


def test_the_report_pairs_the_events_once_and_reads_one_episode_frame(cfg, tmp_path, monkeypatch):
    """Four metric families used to pair decisions to outcomes four or five
    times and settle the episode frame twice. build_report does each once
    and hands them down; each family, called on its own, still answers the
    same."""
    from conftest import decision_event, outcome_event
    from daily import monitor as mon
    from daily import update as upd
    from events.store import EventStore
    from engine.posterior import PosteriorStore

    store = EventStore(cfg, root=str(tmp_path / "events"))
    for i in range(4):
        store.emit_decision(decision_event(decision_id=f"D{i}", episode_id=f"EP{i}",
                                           sku_id=f"S{i}"))
        store.emit_outcome(outcome_event(outcome_id=f"O{i}", decision_id=f"D{i}",
                                         adjustment_reason="episode_close_write_off"))
    posterior = PosteriorStore.initialise(
        cfg, {"vegetables": {"mean": -1.0, "std": 0.6}}, {"vegetables": 500},
        path=str(tmp_path / "posterior.json"))

    calls = {"pairs": 0, "economics": 0}
    real_pairs, real_econ = mon.match_pairs, mon.metrics.episode_economics

    def counting_pairs(*a, **k):
        calls["pairs"] += 1
        return real_pairs(*a, **k)

    def counting_econ(*a, **k):
        calls["economics"] += 1
        return real_econ(*a, **k)

    monkeypatch.setattr(mon, "match_pairs", counting_pairs)
    monkeypatch.setattr(upd, "match_pairs", counting_pairs)
    monkeypatch.setattr(mon.metrics, "episode_economics", counting_econ)
    report = mon.build_report(store, posterior, cfg)
    assert calls == {"pairs": 1, "economics": 1}, calls

    decisions, outcomes = store.load_decisions(), store.load_outcomes()
    assert report["business"] == mon.business_metrics(decisions, outcomes)
    assert report["guardrails"] == mon.guardrail_series(decisions, outcomes, cfg)
    assert report["safety"] == mon.safety_metrics(store, decisions, outcomes)
    assert report["learning"]["priced_days"] == ["2026-08-19"]
    assert report["business"]["waste_units"] == 4
    # the store-wide deff is named for what it is, apart from update's
    # per-batch `deff_applied`
    assert "deff_applied_all_time" in report["learning"]
    assert "deff_applied" not in report["learning"]


def test_price_mismatch_is_a_rate_over_compared_pairs(cfg):
    """Dividing by every outcome let unmatched outcomes dilute the rate
    below the stop threshold."""
    from daily.monitor import safety_metrics, stop_conditions
    from events.pairs import quality_rates

    class _Store:
        duplicate_counts = {"decision": 0, "outcome": 0}

        def load_quarantine(self):
            return []

    decisions = [{"decision_id": "D1", "date": "2026-08-19", "applied_price": 100.0,
                  "expected_denominator": 1.0, "original_price": 100.0}]
    outcomes = [{"decision_id": "D1", "applied_price": 90.0, "units_sold": 1,
                 "is_stockout": False}]
    # undated orphans have no day to age out on: counted in the window
    outcomes += [{"decision_id": f"orphan{i}", "applied_price": 1.0,
                  "units_sold": 0, "is_stockout": False} for i in range(9)]
    s = safety_metrics(_Store(), decisions, outcomes, cfg=cfg)
    assert s["applied_vs_recommended_price_mismatch"] == 1.0
    assert s["unmatched_outcome_count"] == 9 and s["outcomes_in_window"] == 10
    assert s["duplicate_or_unmatched_rate"] == 0.9
    # the stop condition reads the block's COUNTS through the one rate
    # definition, so an override of the counts moves the verdict
    fired = stop_conditions(s, {}, {}, {}, cfg)["fired"]
    assert fired["price_mismatch"] and fired["duplicate_or_unmatched"]
    calm = dict(s, price_mismatch_count=0, unmatched_outcome_count=0)
    assert quality_rates(calm) == {"duplicate_or_unmatched_rate": 0.0,
                                   "price_mismatch_rate": 0.0}
    assert not stop_conditions(calm, {}, {}, {}, cfg)["fired"]["price_mismatch"]


def test_the_overspend_stop_takes_no_reading_while_the_il_base_is_short(cfg):
    """The stop compares the same budget the controller prices from, by the
    same rule (explore.budget_base_ready): a base shorter than its window
    is no reading, so a launch's first over-budget mornings cannot fire it
    -- the owner's rehearsal lost the whole pilot to that on day three."""
    from daily.monitor import overspend_series, evaluate_guardrail

    cfg["exploration"]["budget_il_window_days"] = 7
    il = {f"2026-09-{d:02d}": 1000.0 for d in range(1, 12)}
    learning = {"posterior_by_cell": {"GLOBAL": {"std": 0.5}},
                "cell_of": {"MEAT": "GLOBAL"},
                "exploration_cost_by_day": {f"2026-09-{d:02d}": 1e6 for d in range(2, 12)},
                "priced_days": [f"2026-09-{d:02d}" for d in range(2, 12)],
                "latest_priced_day": "2026-09-11"}
    series = overspend_series(learning, {"il_by_close_day": il}, cfg)
    # 09-02 .. 09-07 have a base shorter than seven days: no reading
    assert sorted(series["by_day"]) == [f"2026-09-{d:02d}" for d in range(8, 12)]
    assert evaluate_guardrail(series, 2.0, 2)["fired"]        # once it can read, it does
    learning["priced_days"] = learning["priced_days"][:4]
    learning["latest_priced_day"] = "2026-09-05"
    early = overspend_series(learning, {"il_by_close_day": il}, cfg)
    assert early["by_day"] == {} and not evaluate_guardrail(early, 2.0, 2)["fired"]
