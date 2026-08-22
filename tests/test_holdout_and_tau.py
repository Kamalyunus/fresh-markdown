"""The hold-out slice, the shared tau derivation, and the controller trace.

Three properties, each of which was violated by the code these tests replace:

  1. A date cut must take whole episodes. Row-level slicing kept the tail of
     a cross-midnight window as its own episode -- no entry decision, wrong
     opening inventory, a countdown starting mid-window.
  2. tau must be solved on every decision hour. The replay collected spreads
     at entry only, funding ~1 exploration per episode against a system that
     explores every hour -- and its own bisection reported 1.00x regardless.
  3. A single spend-over-budget multiple cannot say whether a pilot survives
     its first day. The controller only ever sees yesterday.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from common import episodes
from common.config import load_config as _load_config
from pricing.explore import SpreadLedger, budget_today, tau_next

# By path, not by CWD. The end-to-end tests chdir into a temp workspace and
# stay there, so a bare load_config() here reads whichever config ran last --
# and the assertions below are about the one this repo SHIPS.
REPO_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def load_config():
    return _load_config(REPO_CONFIG)


# ---------------------------------------------------------------- window_slice

def _frame():
    """Two episodes: one opens 08-03 22:00 and runs past midnight into 08-04,
    one opens 08-04 09:00. Only the second belongs to a 08-04 hold-out."""
    rows = []
    for h in range(22, 24):
        rows.append(("crosses", "2026-08-03", h))
    for h in range(0, 4):
        rows.append(("crosses", "2026-08-04", h))
    for h in range(9, 13):
        rows.append(("inside", "2026-08-04", h))
    return pd.DataFrame(rows, columns=["episode_id", "date", "hour_of_day"])


def test_window_slice_takes_whole_episodes_or_none():
    out = episodes.window_slice(_frame(), "2026-08-04", "2026-08-21")
    assert set(out.episode_id) == {"inside"}
    assert len(out) == 4          # all four of its rows, none of the other's


def test_row_level_slicing_is_what_this_prevents():
    d = _frame()
    naive = d[d.date.astype(str).ge("2026-08-04")]
    # the naive cut keeps 4 orphan hours of an episode that opened the day
    # before -- a "short episode" that never existed
    assert (naive.episode_id == "crosses").sum() == 4
    assert "crosses" not in set(
        episodes.window_slice(d, "2026-08-04", None).episode_id)


def test_window_slice_assigns_every_episode_to_exactly_one_slice():
    d = _frame()
    a = episodes.window_slice(d, None, "2026-08-03")
    b = episodes.window_slice(d, "2026-08-04", None)
    assert set(a.episode_id) | set(b.episode_id) == {"crosses", "inside"}
    assert not set(a.episode_id) & set(b.episode_id)
    assert len(a) + len(b) == len(d)


def test_window_slice_is_a_noop_without_bounds():
    d = _frame()
    assert episodes.window_slice(d) is d


def test_split_frames_uses_the_shared_rule():
    import inspect
    from bootstrap import prepare_data
    src = inspect.getsource(prepare_data.split_frames)
    assert "window_slice" in src
    assert ".transform(\"min\")" not in src      # not a second copy of it


# ------------------------------------------------------------------- ledger

def test_solve_tau_lands_just_under_budget():
    rng = np.random.default_rng(0)
    led = SpreadLedger()
    for i in range(400):
        led.add(f"2026-08-{1 + i % 4:02d}", rng.lognormal(6, 1, rng.integers(2, 12)))
    budget = 4000.0
    tau = led.solve_tau(budget, n_days=4)
    spend = led.implied_daily_spend(tau, 4)
    assert spend <= budget
    assert spend > 0.95 * budget          # under, but not by much


def test_spend_is_monotone_in_tau():
    rng = np.random.default_rng(1)
    led = SpreadLedger()
    for i in range(200):
        led.add("d", rng.lognormal(5, 1, 6))
    seq = [led.implied_daily_spend(t, 1) for t in range(0, 2000, 100)]
    assert seq == sorted(seq)
    assert led.implied_daily_spend(0.0, 1) == 0.0


def test_spend_by_day_sums_to_the_total():
    rng = np.random.default_rng(2)
    led = SpreadLedger()
    for i in range(120):
        led.add(f"d{i % 3}", rng.lognormal(5, 1, 4))
    tau = 500.0
    by_day = led.spend_by_day(tau)
    assert len(by_day) == 3
    assert by_day.sum() == pytest.approx(led.implied_daily_spend(tau, 1))


def test_chunked_flush_does_not_lose_costs():
    rng = np.random.default_rng(3)
    a, b = SpreadLedger(), SpreadLedger()
    b._FLUSH = 64                       # force many chunks
    for _ in range(200):
        c = rng.lognormal(5, 1, 5)
        a.add("d", c)
        b.add("d", c)
    assert a.decisions == b.decisions == 200
    assert a.implied_daily_spend(1e12, 1) == pytest.approx(
        b.implied_daily_spend(1e12, 1))


def test_empty_ledger_is_not_a_crash():
    led = SpreadLedger()
    assert led.decisions == 0
    assert led.solve_tau(100.0) is None
    assert led.distribution() == {}
    assert led.quantile_of(10.0) is None


def test_entry_only_collection_understates_the_funded_tau():
    """The bug this class exists to prevent, reproduced.

    Funding one decision per episode buys a much larger tau than funding
    every hour of it, because the same daily budget is spread over ~8x fewer
    decisions. Solve on entry only, measure on all hours: over budget.
    """
    rng = np.random.default_rng(4)
    entry_only, all_hours = SpreadLedger(), SpreadLedger()
    for ep in range(300):
        for hour in range(8):
            costs = rng.lognormal(6, 0.8, 6)
            all_hours.add("d", costs)
            if hour == 0:
                entry_only.add("d", costs)

    budget = 20000.0
    tau_entry = entry_only.solve_tau(budget, n_days=1)
    tau_all = all_hours.solve_tau(budget, n_days=1)
    assert tau_entry > tau_all
    # launching at the entry-only tau overspends on the real decision count
    assert all_hours.implied_daily_spend(tau_entry, 1) > budget


def test_replay_collects_every_decision_hour():
    import inspect
    from backtest import replay
    src = inspect.getsource(replay.policy_replay)
    assert "ledger.add" in src
    assert "if t == 0 and costs" not in src


# ------------------------------------------------------- controller trace

def test_controller_cannot_correct_before_it_has_seen_a_day(cfg=None):
    """tau_next only ever reads the day just closed, so day 1 is spent at the
    launch tau whatever that is -- which is why the trace exists."""
    cfg = cfg or load_config()
    tau0 = 10_000.0
    budget, spend = 1_000.0, 8_700.0      # the 8.7x the shadow report measured
    assert tau_next(tau0, budget, spend, cfg) == tau0 * 0.5   # clip floor
    # three halvings to get under a 2.0x stop, if spend fell proportionally
    tau, over = tau0, spend / budget
    days_over = 0
    for _ in range(5):
        if over > cfg["monitoring"]["stop_conditions"]["exploration_cost_vs_budget"]:
            days_over += 1
        tau = tau_next(tau, budget, budget * over, cfg)
        over = over / 2
    assert days_over >= 3


def test_budget_scales_down_as_the_posterior_narrows():
    cfg = load_config()
    ec = cfg["exploration"]
    wide = budget_today(1_000_000.0, ec["budget_scale_ref_std"], cfg)
    narrow = budget_today(1_000_000.0, 0.0, cfg)
    assert wide == ec["budget_share_of_il"] * 1_000_000.0
    assert narrow == pytest.approx(wide * ec["budget_scale_floor"])


def test_shadow_budget_uses_the_production_budget_function():
    import inspect
    from pipeline import shadow
    src = inspect.getsource(shadow.run_shadow)
    assert "explore.budget_today(" in src


# ------------------------------------------------- pre-launch containment

def test_pre_launch_stops_at_the_gate_window():
    from bootstrap.prepare_data import pre_launch
    cfg = load_config()
    end = cfg["data"]["split"]["test_end"]
    d = pd.DataFrame({
        "episode_id": ["before", "before", "straddles", "straddles",
                       "holdout", "after_holdout"],
        "date": [end, end, end, cfg["data"]["holdout"]["start"],
                 cfg["data"]["holdout"]["start"], "2026-09-01"],
        "hour_of_day": [22, 23, 23, 0, 9, 9],
    })
    kept = set(pre_launch(d, cfg).episode_id)
    assert kept == {"before", "straddles"}, \
        "an episode that OPENED before the gate window closed belongs to " \
        "pre-launch whole; one that opened after does not belong at all"


def test_the_backtest_cannot_reach_past_the_gate_window():
    """Two paths reached the hold-out and neither announced itself.

    `policy_replay` and `derive_tau_initial` ran on the whole frame, so
    tau_initial -- a MEASURED launch value -- was being fitted on the window
    reserved for grading it. And `calibration_fit_window: "all"` resolved to
    the whole frame, one config edit from fitting the level factors there.
    """
    import inspect
    from backtest import __main__ as bt
    from bootstrap import train_baseline

    src = inspect.getsource(bt.main)
    assert "pre_launch(d, cfg)" in src
    assert src.index("pre_launch(d, cfg)") < src.index("fidelity("), \
        "the slice must happen before anything reads the frame"

    fit = inspect.getsource(train_baseline.fit_level_calibration)
    all_branch = fit[fit.index('fit_window == "all"'):]
    assert "pre_launch(d, cfg)" in all_branch.split("elif")[0], \
        'calibration_fit_window "all" must mean all PRE-LAUNCH data'


def test_the_three_artifact_fits_stay_inside_their_own_splits():
    """Bounded already -- asserted so they stay that way."""
    import inspect
    from bootstrap import train_baseline, fit_dispersion, estimate_prior
    assert 'splits["train"]' in inspect.getsource(train_baseline.train)
    assert 'split_frames(d, cfg)["calib"]' in inspect.getsource(
        fit_dispersion.fit_dispersion)
    assert 'split_frames(d, cfg)["train"]' in inspect.getsource(
        estimate_prior.estimate_prior)


# -------------------------------------------------------- tau provenance

def _cfg_with_tau(tau):
    cfg = load_config()
    return dict(cfg, exploration=dict(cfg["exploration"], tau_initial=tau))


def _derivation(tau, scoped=True):
    block = {"tau_initial": tau}
    if scoped:
        block["spread_decisions"] = 12345
    return {"tau_initial_derivation": block}


def test_a_matching_paste_from_a_current_backtest_is_clean():
    from pricing.explore import tau_provenance_error
    assert tau_provenance_error(_cfg_with_tau(410.74),
                                _derivation(410.74)) is None


def test_a_null_tau_is_not_this_check_s_business():
    from pricing.explore import tau_provenance_error
    # the null case is _require_shadow_config's, and it is louder
    assert tau_provenance_error(_cfg_with_tau(None), None) is None


def test_a_paste_with_no_derivation_on_disk_is_refused():
    from pricing.explore import tau_provenance_error
    assert "no backtest derivation" in tau_provenance_error(
        _cfg_with_tau(410.74), None)


def test_a_derivation_predating_the_scoping_fix_is_refused():
    from pricing.explore import tau_provenance_error
    err = tau_provenance_error(_cfg_with_tau(410.74),
                               _derivation(410.74, scoped=False))
    assert "ENTRY decisions only" in err


def test_a_paste_that_no_longer_matches_its_source_is_refused():
    from pricing.explore import tau_provenance_error
    err = tau_provenance_error(_cfg_with_tau(500.0), _derivation(410.74))
    assert "500.0" in err and "410.74" in err


def test_shadow_refuses_to_start_on_a_stale_tau(tmp_path):
    from common.config import ConfigError
    from pipeline import shadow
    cfg = load_config()
    cfg = dict(cfg,
               baseline_model=dict(cfg["baseline_model"],
                                   apply_level_calibration=False),
               exploration=dict(cfg["exploration"], tau_initial=410.74))
    path = tmp_path / "backtest.json"
    path.write_text(json.dumps(_derivation(410.74, scoped=False)))
    with pytest.raises(ConfigError, match="stale tau"):
        shadow._require_shadow_config(cfg, backtest_path=str(path))
    path.write_text(json.dumps(_derivation(410.74)))
    shadow._require_shadow_config(cfg, backtest_path=str(path))   # now fine


def test_status_fails_a_stale_tau_rather_than_passing_it():
    from pipeline import status
    cfg = _cfg_with_tau(410.74)
    row = status._tau(cfg, _derivation(410.74, scoped=False))
    assert row["verdict"] == status.FAIL
    assert status._tau(cfg, _derivation(410.74))["verdict"] == status.PASS


def test_config_ships_tau_initial_null():
    # it is void until re-derived: the scoping fix changed what the backtest
    # produces, so any value carried over from before is wrong
    assert load_config()["exploration"]["tau_initial"] is None


# ------------------------------------------------------------------- config

def test_holdout_window_is_after_the_test_window():
    cfg = load_config()
    h, s = cfg["data"]["holdout"], cfg["data"]["split"]
    assert h["start"] > s["test_end"], \
        "a hold-out that overlaps the gate window is not a hold-out"
    assert h["end"] > h["start"]


def test_holdout_is_disjoint_from_every_fitting_window():
    cfg = load_config()
    h, s = cfg["data"]["holdout"], cfg["data"]["split"]
    for lo, hi in [(s["train_start"], s["train_end"]),
                   (s["calib_start"], s["calib_end"]),
                   (s["test_start"], s["test_end"])]:
        assert not (h["start"] <= hi and lo <= h["end"])
