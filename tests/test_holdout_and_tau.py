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


# --------------------------------------------- shadow defaults to holdout

def test_shadow_runs_on_the_holdout_unless_told_otherwise():
    """The default matters more than the flag.

    Every frozen artifact is fit on data up to test_end, so a shadow run that
    includes that data grades the pipeline on rows it already saw. Making the
    hold-out opt-IN meant the honest run was the one someone had to remember.
    """
    import inspect
    from pipeline import shadow

    src = inspect.getsource(shadow.main)
    # the hold-out branch is the fall-through, not a flag test
    assert "elif args.all:" in src
    assert 'basis = HOLDOUT_BASIS' in src
    assert src.index("args.all") < src.index("basis = HOLDOUT_BASIS")
    # and run_shadow assumes it too, so a programmatic caller gets the same
    assert inspect.signature(shadow.run_shadow).parameters[
        "window_basis"].default == shadow.HOLDOUT_BASIS


def test_a_non_holdout_run_says_which_numbers_it_flatters():
    import inspect
    from pipeline import shadow
    src = inspect.getsource(shadow.run_shadow)
    assert "in_sample_caveat" in src
    # the caveat has to name what it does NOT undermine, or it reads as
    # "ignore this whole report" and gets ignored itself
    assert "cost-floor" in src[src.index("in_sample_caveat"):]


def test_missing_holdout_config_is_an_error_not_a_silent_full_run():
    """The dangerous failure is running on everything and not saying so."""
    import inspect
    from pipeline import shadow
    src = inspect.getsource(shadow.main)
    branch = src[src.index("no data.holdout"):]
    assert "--all" in branch          # names the deliberate alternative
    assert "SystemExit" in src[:src.index("no data.holdout")]


def test_the_trace_says_when_a_sample_is_too_thin_to_read_daily():
    """Sampling degrades exactly one figure, and it has to name itself.

    The gate reads rates; tau_recommended equates two quantities that both
    scale with the sample, so the tau solving them is invariant. The
    day-by-day trace is the exception -- it divides the sample across the
    window's days, so the controller looks jumpier than it is.
    """
    import inspect
    from pipeline import shadow
    src = inspect.getsource(shadow._controller_trace)
    assert "episodes_per_day_sampled" in src
    assert "episodes_per_day_population" in src
    # and it must point at the figure that DOES survive sampling, or the
    # caveat leaves the reader with nothing to quote
    assert "sample-invariant" in src and "spend_over_budget" in src


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
    from bootstrap import prior_density
    assert 'split_frames(d, cfg)[window]' in inspect.getsource(
        prior_density.build_curves), (
        "the prior must take its rows from a named split, so the fit window "
        "and the held-out window cannot silently be the same one")
    assert '"train"' in inspect.getsource(prior_density.estimate), \
        "the prior fit must be built on the TRAIN window"
    hold = inspect.getsource(prior_density.holdout_comparison)
    assert 'cfg["posterior"]["prior"]["holdout_window"]' in hold, \
        "the held-out comparison must score the CONFIGURED held-out window"
    assert 'build_curves(d, cfg, model, grid, "train")' not in hold, \
        "the held-out comparison must not score the window the prior was fitted on"


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
    none = str(tmp_path / "missing.json")
    path.write_text(json.dumps(_derivation(410.74, scoped=False)))
    with pytest.raises(ConfigError, match="stale tau"):
        shadow._require_shadow_config(cfg, backtest_path=str(path),
                                      shadow_path=none)
    path.write_text(json.dumps(_derivation(410.74)))
    shadow._require_shadow_config(cfg, backtest_path=str(path),
                                  shadow_path=none)   # now fine


def test_a_shadow_derivation_is_the_trusted_paste_source():
    """The anchored-path derivation outranks the backtest's exploit-only one:
    a paste matching shadow is clean even when the backtest disagrees, and a
    paste matching only the backtest is refused once shadow has derived."""
    from pricing.explore import tau_provenance_error
    shadow = {"tau_initial_derivation": {"tau_initial": 257.48,
                                         "fallback": False}}
    assert tau_provenance_error(_cfg_with_tau(257.48),
                                _derivation(410.74), shadow) is None
    err = tau_provenance_error(_cfg_with_tau(410.74),
                               _derivation(410.74), shadow)
    assert "257.48" in err and "shadow" in err


def test_a_fallback_shadow_block_defers_to_the_backtest_checks():
    # a shadow run that itself fell back to the paste is not a paste source
    from pricing.explore import tau_provenance_error
    shadow = {"tau_initial_derivation": {"tau_initial": None, "fallback": True}}
    assert tau_provenance_error(_cfg_with_tau(410.74),
                                _derivation(410.74), shadow) is None


def test_shadow_derives_its_launch_tau_on_the_pre_window_week():
    """The tau in force comes from derive_tau0 over the trailing week -- the
    same span as the day-one budget base -- on the run's own ANCHORED path.
    The config paste is only the fallback, still behind the provenance gate."""
    import inspect
    from pipeline import shadow
    cfg = load_config()
    assert cfg["exploration"]["tau0_derivation_min_decisions"] > 0
    src = inspect.getsource(shadow.run_shadow)
    assert "derive_tau0(" in src
    assert src.index("derive_tau0(") < src.index("_require_shadow_config"), \
        "the paste (and its provenance gate) must be the FALLBACK"
    der = inspect.getsource(shadow.derive_tau0)
    assert "budget_il_window_days" in der       # the budget's own span
    assert "budget_today(" in der and "solve_tau(" in der
    # a sampled week carries only its fraction of the population's spend, so
    # the bisection must target the budget scaled by the same fraction
    assert "budget * frac" in der
    # and main hands run_shadow the FULL frame, before the window slice
    assert "pre_window_frame=full" in inspect.getsource(shadow.main)


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


def test_the_budget_base_is_the_trailing_realised_il():
    """The IL base for a day's budget is the mean of REALISED daily IL over
    the trailing budget_il_window_days, ending YESTERDAY -- never the same
    day's own IL.

    Same-day IL is unobservable when the budget is needed (scrap lands only
    at episode close), and it funds exploration hardest on exactly the days
    already losing most, since IL is discount plus scrap. The trace and the
    aggregate gate must grade against the budget production would actually
    apply, or the stop condition tests a counterfactual.
    """
    import inspect

    from pricing.explore import trailing_daily_il
    from pipeline import shadow

    cfg = load_config()
    assert cfg["exploration"]["budget_il_window_days"] == 7

    il = {f"2026-08-{d:02d}": 700.0 for d in range(1, 8)}   # 7 flat days
    # day 8's base is the mean of days 1-7; day 8's own IL must not enter
    il["2026-08-08"] = 99_999.0
    assert trailing_daily_il(il, "2026-08-08", cfg) == pytest.approx(700.0)

    # a zero-IL calendar day inside the window counts as ZERO, not skipped:
    # the budget is a share of the actual run-rate
    il2 = {"2026-08-01": 700.0, "2026-08-07": 700.0}
    assert trailing_daily_il(il2, "2026-08-08", cfg) == pytest.approx(200.0)

    # no history at all -> no budget. The conservative side to start on.
    assert trailing_daily_il({}, "2026-08-08", cfg) == 0.0

    # and shadow actually uses it, in the trace and in the aggregate gate
    src = inspect.getsource(shadow)
    assert src.count("trailing_daily_il(") >= 2, \
        "both the controller trace and the aggregate gate must budget on " \
        "the trailing basis, or they grade different quantities"


def test_the_first_shadow_day_carries_the_pre_window_trailing_base():
    """Day one of the hold-out must not read budget 0: production's launch
    day holds the trailing legacy IL of the days before it. The history is
    computed from the full frame BEFORE the window slice, keyed by close day
    -- discount plus scrap, closed episodes only."""
    import pandas as pd
    from pipeline.shadow import pre_window_il_history

    cfg = load_config()

    def episode(eid, day, sold, start, end_last):
        return pd.DataFrame({
            "episode_id": [eid] * 2, "date": [day] * 2, "hour_of_day": [9, 10],
            "total_discount": [0.30] * 2, "original_price": [10_000.0] * 2,
            "cost": [4000.0] * 2, "starting_inventory": [start, start - sold],
            "units_sold": [sold, 0], "ending_inventory": [start - sold, end_last],
        })

    d = pd.concat([
        # closes 08-01 with 2 left (write-off): IL = 3000*1... discount 0.3*10000*1 + scrap 2*4000
        episode("pre", "2026-08-01", 1, 3, 0),
        # unclosed pre-window episode: contributes NOTHING
        episode("open", "2026-08-02", 1, 3, 2),
        # closes INSIDE the window: not part of the seed
        episode("in", "2026-08-05", 1, 2, 0),
    ])
    h = pre_window_il_history(d, cfg, "2026-08-04")
    assert set(h) == {"2026-08-01"}
    assert h["2026-08-01"] == pytest.approx(0.30 * 10_000 * 1 + 2 * 4000)
    # outside the trailing window -> empty; no `before` -> empty
    assert pre_window_il_history(d, cfg, "2026-08-20") == {}
    assert pre_window_il_history(d, cfg, None) == {}


def test_the_pre_window_seed_is_scaled_to_the_sample():
    """The seed and the spend must describe the SAME slice of the business.

    The pre-window IL has to be measured on the full frame -- those episodes
    are outside the window and are never sampled -- while the window's own IL
    and the exploration spend it is graded against are measured on the sample.
    Left unscaled the first `budget_il_window_days` of budget come in at
    1/sample_fraction too large, and `spend_over_budget` reads near zero on
    any sampled run: a 2.9% sample inflated the reported budget ~14x and
    turned a ~0.7x into 0.05x. The bug is invisible at --max-episodes 0,
    which is exactly why it needs a test.
    """
    import inspect
    from pipeline import shadow

    src = inspect.getsource(shadow.run_shadow)
    assert "seed_scale" in src, "the pre-window seed is not scaled at all"
    seeded = src[src.index("seed_scale ="):]
    assert "len(groups)" in seeded.split("\n")[0] and \
        "len(population)" in seeded.split("\n")[0], \
        "the scale must be the WINDOW's sample fraction, sampled/population"
    # and it must be applied to the seed, not merely computed
    applied = seeded[:400]
    assert "amount * seed_scale" in applied, \
        "seed_scale is computed but never multiplied into il_by_day"
    # a full run must be unaffected: scale is exactly 1 when nothing sampled
    assert "max(len(population), 1)" in seeded.split("\n")[0]


def test_the_controller_holds_tau_on_a_zero_budget():
    """A zero budget from an EMPTY trailing history is an absence of signal,
    not an overspend: the controller must hold tau, not halve it. The moment
    history exists, calibration resumes."""
    from pipeline.shadow import _controller_trace
    from pricing.explore import SpreadLedger

    cfg = load_config()

    led = SpreadLedger()
    led.add("2026-08-04", [50.0, 80.0])          # spend exists on day 1
    led.add("2026-08-05", [50.0, 80.0])
    # no IL history at all for day 1; day 2 sees day 1's realised IL
    il_by_day = {"2026-08-04": 1_000_000.0}
    trace = _controller_trace(led, il_by_day, tau0=100.0, widest_std=1.0,
                              cfg=cfg)
    d1, d2 = trace["by_day"][0], trace["by_day"][1]
    assert d1["budget"] == 0.0 and d1["over_budget"] is None
    assert d2["tau"] == pytest.approx(100.0), \
        "tau moved on a day whose budget was 0 -- empty history is not signal"
    assert d2["budget"] > 0


# ------------------------------------------------- point-in-time calibration

def test_calibration_factors_never_see_their_own_week_or_later():
    """The whole point of the rolling schedule: week W's factors are fit on
    the trailing window ENDING STRICTLY BEFORE W.

    This is the same discipline as the point-in-time velocity features (design
    12a) and it fails the same way -- silently, flattering every downstream
    number, with no error anywhere. So it is asserted on the ARTIFACT rather
    than trusted to the code that wrote it: for every fitted week, the window
    the factors claim to come from must end before that week begins.
    """
    import json
    import os

    import pandas as pd

    cfg = load_config()
    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        pytest.skip("no calibration artifact on disk")
    with open(path) as f:
        cal = json.load(f)
    sched = cal.get("schedule")
    if not sched:
        pytest.skip("calibration is not on the rolling schedule")

    weeks = sorted(sched["by_week"])
    assert weeks, "a rolling schedule with no fitted weeks is not a schedule"
    n = int(sched["trailing_weeks"])
    for w in weeks:
        start = pd.Timestamp(w)
        # the window is [start - n weeks, start): its LAST instant is strictly
        # before the week the factors are applied to
        window_end = start - pd.Timedelta(seconds=1)
        assert window_end < start
        assert (start - (start - pd.Timedelta(weeks=n))).days == 7 * n

    # and the applier must key on the row's OWN week, never the nearest or
    # the latest available -- borrowing forward is the leak
    import inspect
    from bootstrap.train_baseline import BaselineModel
    src = inspect.getsource(BaselineModel._factor_vector)
    assert 'to_period("W")' in src, "factors are not selected by row week"
    assert "self.calibration.get(key, 1.0)" in src, \
        "an unfitted week must fall back to the frozen set, not to a later week"


def test_both_harnesses_get_point_in_time_factors_without_their_own_code():
    """backtest and shadow must not each re-implement factor selection: they
    call predict_mu_ref on frames carrying `date`, so the schedule reaches
    them through the one applier. A second copy would drift."""
    import inspect
    from backtest import replay
    from pipeline import shadow

    for mod, name in ((replay, "backtest.replay"), (shadow, "pipeline.shadow")):
        src = inspect.getsource(mod)
        assert "predict_mu_ref(" in src, f"{name} does not predict mu_ref"
        # `by_week` is deliberately NOT banned: fidelity has its own weekly
        # series. What must not appear is the calibration schedule's own names
        for banned in ("calibration_schedule", "_factor_vector",
                       "calibration_factor_path"):
            assert banned not in src, (
                f"{name} reaches into the calibration schedule itself -- "
                "factor selection belongs to BaselineModel alone")


def test_a_level_shift_does_not_leak_into_its_own_weeks_factor(tmp_path):
    """The functional half of the point-in-time guarantee.

    A schedule built by hand where demand jumps during the week of 07-13. The
    factor APPLIED during that week must be the one fit BEFORE it -- if the
    jump leaked into its own week's factor the model would appear to have
    tracked a shift it could not have seen, and every backtest and shadow
    figure downstream would be flattered by hindsight. The same failure mode
    as the point-in-time velocity features (design 12a), and just as silent.
    """
    import json

    import pandas as pd

    from bootstrap.train_baseline import BaselineModel

    quiet, jumped, fallback = 1.00, 1.80, 1.33
    artifact = {
        "grain": "category",
        "factors": {"FRUIT": fallback},           # frozen set, distinct value
        "schedule": {
            "mode": "rolling_trailing", "trailing_weeks": 4,
            "by_week": {
                "2026-07-06": {"FRUIT": quiet},   # before the jump
                "2026-07-13": {"FRUIT": quiet},   # THE JUMP WEEK: cannot see it
                "2026-07-20": {"FRUIT": jumped},  # first week that can
            },
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact))

    cfg = load_config()
    cfg = dict(cfg, baseline_model=dict(cfg["baseline_model"],
                                        calibration_factor_path=str(path)))
    model = BaselineModel.__new__(BaselineModel)      # applier only, no booster
    model.cfg = cfg
    model.calibration = artifact["factors"]
    model.calibration_grain = "category"
    model.calibration_schedule = artifact["schedule"]["by_week"]
    model._reset_calibration_counters()

    rows = pd.DataFrame({
        "category": ["FRUIT"] * 4,
        "date": ["2026-07-15",   # DURING the jump week
                 "2026-07-22",   # the week after, which may see it
                 "2026-07-08",   # the week before
                 "2026-06-29"],  # no fitted week -> the frozen fallback
    })
    f = model._factor_vector(rows)

    assert f[0] == pytest.approx(quiet), \
        "the jump leaked into the factor applied during its own week"
    assert f[1] == pytest.approx(jumped), \
        "the week after the jump never picked it up"
    assert f[2] == pytest.approx(quiet)
    assert f[3] == pytest.approx(fallback), \
        "an unfitted week must fall back to the frozen set, never borrow " \
        "a later week's factors"


def test_running_past_the_calibration_schedule_is_reported_not_silent():
    """The schedule only covers weeks it was fitted on. A row past its end
    takes the frozen fallback and nothing errors -- so production drifts back
    onto stale factors the moment the weekly re-fit is missed, invisibly.
    That is the failure the point-in-time change exists to remove, so it has
    to be counted and named."""
    import pandas as pd

    from bootstrap.train_baseline import BaselineModel

    model = BaselineModel.__new__(BaselineModel)
    model.cfg = load_config()
    model.calibration = {"FRUIT": 1.33}
    model.calibration_grain = "category"
    model.calibration_schedule = {"2026-07-06": {"FRUIT": 1.10}}
    model._reset_calibration_counters()

    rows = pd.DataFrame({"category": ["FRUIT"] * 3,
                         "date": ["2026-07-08",     # inside the schedule
                                  "2026-07-20",     # PAST its end
                                  "2026-07-27"]})   # past its end
    f = model._factor_vector(rows)
    assert f[0] == pytest.approx(1.10)
    assert f[1] == pytest.approx(1.33) and f[2] == pytest.approx(1.33)

    cov = model.calibration_coverage()
    assert cov["rows_on_schedule"] == 1
    assert cov["rows_on_fallback"] == 2
    assert cov["weeks_after_schedule_end"] == ["2026-07-20", "2026-07-27"]
    assert cov["verdict"].startswith("STALE FACTORS IN USE")
    # the verdict must name the remedy, not merely report the count
    assert "--fit-calibration" in cov["verdict"]

    # a run entirely inside the schedule says so, and says nothing alarming
    model._cal_rows_fallback, model._cal_fallback_weeks = 0, set()
    assert model.calibration_coverage()["verdict"].startswith("OK")


def test_both_reports_carry_the_calibration_coverage():
    """It is only useful where the numbers are read."""
    import inspect
    from backtest import __main__ as bt
    from pipeline import shadow
    for mod, name in ((bt, "backtest"), (shadow, "pipeline.shadow")):
        assert "calibration_coverage()" in inspect.getsource(mod), \
            f"{name} does not report calibration coverage"


def test_apply_refuses_when_the_weekly_refit_was_missed(tmp_path):
    """Detection after the fact is not enough. Learning from prices set on
    stale factors banks evidence about a model that is not the one running,
    so a schedule that no longer reaches today is a HARD gate on --apply --
    the same standing as a duplicate-event breach."""
    import json

    from pipeline.update import calibration_current

    cfg = load_config()
    path = tmp_path / "calibration.json"
    cfg = dict(cfg, baseline_model=dict(cfg["baseline_model"],
                                        calibration_factor_path=str(path)))
    path.write_text(json.dumps({
        "grain": "category", "factors": {"FRUIT": 1.2},
        "schedule": {"mode": "rolling_trailing", "trailing_weeks": 4,
                     "by_week": {"2026-07-06": {"FRUIT": 1.1},
                                 "2026-07-13": {"FRUIT": 1.1}}}}))

    # inside the schedule -> passes
    assert calibration_current(cfg, today="2026-07-15")["pass"]
    # the week AFTER the last fitted one -> refuses, and names the remedy
    late = calibration_current(cfg, today="2026-07-21")
    assert not late["pass"]
    assert "--fit-calibration" in late["note"]

    # static calibration is a legitimate configuration: nothing to outrun
    path.write_text(json.dumps({"grain": "category", "factors": {"FRUIT": 1.2}}))
    assert calibration_current(cfg, today="2027-01-01")["pass"]

    # and no artifact at all means factors are 1.0 -- also not a failure
    path.unlink()
    assert calibration_current(cfg, today="2027-01-01")["pass"]


def test_the_calibration_gate_is_wired_into_the_apply_refusal():
    """It only bites if `run` treats it as hard, beside the event-quality
    gates, and reports it on the monitor-only pass too."""
    import inspect
    from pipeline import update

    src = inspect.getsource(update.run)
    assert 'gates["calibration_schedule_current"] = calibration_current' in src
    # hard_fail is computed AFTER the gate is added, or it can never refuse
    assert src.index("calibration_schedule_current") < src.index("hard_fail = ")


def test_a_partial_trailing_window_is_counted_not_passed_off_as_full():
    """`trailing_weeks: 4` does not mean every week HAS four behind it.

    Two places have less: the start of the extract, and the weeks just after
    the exclusion window, where the gap leaves only post-gap history. Those
    weeks still fit -- a large extract clears min_anchor on one week -- but a
    factor fit on 1 of 4 intended weeks is noisier than its label claims, and
    the second group sits mid-extract rather than harmlessly at the start.
    """
    import json
    import os

    cfg = load_config()
    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        pytest.skip("no calibration artifact on disk")
    sched = (json.load(open(path)).get("schedule") or {})
    if not sched.get("by_week"):
        pytest.skip("calibration is not on the rolling schedule")

    assert "weeks_on_partial_window" in sched, \
        "a short trailing window must be counted, not silently labelled full"
    for row in sched["weeks_on_partial_window"]:
        assert row["weeks_in_window"] < sched["trailing_weeks"]
        assert row["week"] in sched["by_week"], \
            "a partial week is still fitted -- it is flagged, not dropped"


def test_the_gate_freezes_calibration_even_though_the_schedule_runs_past_it():
    """Two questions, one artifact, and they must not be confused.

    The schedule re-fits every week and runs through the whole extract,
    because production re-fits weekly and a forward-time replay (shadow, the
    DP walk) should see exactly that -- at week k only weeks < k were read, so
    there is no look-ahead. But the LAUNCH GATE grades a frozen artifact, and
    a factor re-fit inside the hold-out has read the rows it is graded on. So
    fidelity calls `freeze_calibration_from(gate_start)` and prices the gate
    window off the anchor, while `weekly_refit` reports the mechanism reading
    beside it. Freezing the ARTIFACT instead would answer the gate correctly
    and silently stop shadow mirroring production.
    """
    import json
    import os

    cfg = load_config()
    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        pytest.skip("no calibration artifact on disk")
    sched = (json.load(open(path)).get("schedule") or {})
    if not sched.get("by_week"):
        pytest.skip("calibration is not on the rolling schedule")

    split = cfg["data"]["split"]
    gate_start = pd.Timestamp(
        split["test_start"]
        if cfg["baseline_model"]["calibration_gate_window"] == "test"
        else split["calib_start"])
    assert sched.get("gate_freezes_at") == str(gate_start.date()), (
        "the artifact must record where the gate freezes, or coverage cannot "
        "tell a deliberate freeze from production running onto stale factors")

    # the gate does the freezing, not the artifact
    import inspect

    from backtest import replay

    src = inspect.getsource(replay.fidelity)
    assert "freeze_calibration_from(gate_start)" in src, \
        "the fidelity gate must freeze calibration at the gate window start"
    assert '"weekly_refit"' in src, \
        "the mechanism reading must be reported beside the frozen gate"

    # frozen rows take the anchor, scheduled rows take their own week
    from bootstrap.train_baseline import BaselineModel
    m = BaselineModel.__new__(BaselineModel)
    m.calibration_grain = "category"
    m.calibration = {"A": 2.0}
    m.calibration_schedule = {"2026-07-27": {"A": 5.0},
                              "2026-08-03": {"A": 9.0}}
    m._reset_calibration_counters()
    frame = pd.DataFrame({"date": ["2026-07-28", "2026-08-03"],
                          "category": ["A", "A"]})

    m.freeze_calibration_from(None)
    assert list(m._factor_vector(frame)) == [5.0, 9.0], \
        "unfrozen, every row takes its own week -- production's mechanism"

    m._reset_calibration_counters()
    m.freeze_calibration_from(gate_start)
    assert list(m._factor_vector(frame)) == [2.0, 2.0], \
        "frozen at the gate, every graded row takes the anchor"
    assert m._cal_rows_frozen == 2 and m._cal_rows_fallback == 0, \
        "a deliberate freeze must not be counted as a fallback gap"


def test_convergence_check_flags_drift_and_never_commits_the_resolve(
        tmp_path, monkeypatch):
    """The f <-> r cycle is broken by iteration, and this is the assertion
    that the iteration settled. Dry run is load-bearing: committing the
    re-solve while prior/dispersion lag it would CREATE the inconsistency
    being tested for."""
    import copy

    import bootstrap.train_baseline as tb

    cfg = copy.deepcopy(load_config())
    path = str(tmp_path / "cal.json")
    cfg["baseline_model"]["calibration_factor_path"] = path
    cfg["baseline_model"]["calibration_convergence_tol_log"] = 0.02
    old = {"factors": {"A": 1.0, "B": 1.2},
           "schedule": {"by_week": {"2026-07-06": {"A": 1.05}}}}
    json.dump(old, open(path, "w"))

    def fake(art):
        def _fit(d, c):
            json.dump(art, open(path, "w"))
        return _fit

    # a factor moved 1.2 -> 1.3 under the current r/prior -> NOT CONVERGED,
    # named, and the disk artifact still holds iteration k
    monkeypatch.setattr(tb, "fit_level_calibration", fake(
        {"factors": {"A": 1.0, "B": 1.3},
         "schedule": {"by_week": {"2026-07-06": {"A": 1.05}}}}))
    block = tb.check_calibration_convergence(None, cfg)
    assert not block["converged"]
    assert block["worst_cell"] == "anchor:B"
    assert abs(block["max_abs_dlog"] - abs(np.log(1.3 / 1.2))) < 1e-5
    disk = json.load(open(path))
    assert disk["factors"] == old["factors"], "dry run must restore"
    assert disk["convergence"]["converged"] is False

    # identical re-solve -> converged; a schedule week's cell moving is
    # caught the same way as the anchor's
    monkeypatch.setattr(tb, "fit_level_calibration", fake(old))
    assert tb.check_calibration_convergence(None, cfg)["converged"]
    monkeypatch.setattr(tb, "fit_level_calibration", fake(
        {"factors": {"A": 1.0, "B": 1.2},
         "schedule": {"by_week": {"2026-07-06": {"A": 1.30}}}}))
    block = tb.check_calibration_convergence(None, cfg)
    assert not block["converged"] and block["worst_cell"] == "2026-07-06:A"

    # a cell appearing or vanishing is never averaged away
    monkeypatch.setattr(tb, "fit_level_calibration", fake(
        {"factors": {"A": 1.0}, "schedule": {"by_week": {}}}))
    block = tb.check_calibration_convergence(None, cfg)
    assert not block["converged"]
    assert "anchor:B" in block["cells_appeared_or_gone"]


def test_rho_is_fit_on_the_calib_window_not_the_full_frame():
    """rho once read the whole input frame: ~83% training rows, where the
    model fits its own residuals and between-episode variance reads small
    (fixture: in-train rho 0.081 vs out-of-sample 0.230) -- plus, on a longer
    extract, rows past test_end (hard rule 16). An understated rho understates
    deff, and deff deflates every posterior update. calib is the one window
    both out-of-train and pre-gate, and r already lives there."""
    import inspect

    from bootstrap import fit_dispersion as fd

    src = inspect.getsource(fd.fit_dispersion)
    rho_part = src[src.index("rho against fitted residuals"):]
    assert "d.copy()" not in rho_part, \
        "rho must not be fit on the raw input frame"
    assert 'calib.groupby("episode_id")' in rho_part

    art = fd and __import__("json").load(
        open(load_config()["dispersion"]["rho_path"]))
    if "fit_window" in art:                 # artifact written post-change
        assert art["fit_window"] == "calib"


def test_shadow_refits_calibration_forward_only_and_reports_both_regimes():
    """Shadow must answer BOTH calibration questions, and the re-fit one must
    not cheat.

    The artifact's schedule is built over pre-launch data and stops at
    test_end, so every hold-out row falls back to the anchor -- shadow alone
    measures "launch and never re-calibrate". Production re-fits weekly, so
    the report carries that reading too. It is fitted in SHADOW, not in the
    artifact, so the pre-launch bundle stays clean of hold-out rows (rule 16),
    and each week may read only weeks STRICTLY BEFORE it -- a forward replay
    has no later data, and borrowing it would make the comparison meaningless.
    """
    import inspect

    from pipeline import shadow

    src = inspect.getsource(shadow.weekly_refit_schedule)
    assert "wk.dt.start_time < w0" in src, \
        "each week must fit on data strictly before it -- no look-ahead"
    assert "_solve_level_factors" in src, \
        "must reuse the one factor solve, not a second copy"

    run = inspect.getsource(shadow.run_shadow)
    assert '"calibration_regimes"' in run
    for key in ("frozen_anchor", "weekly_refit", "spread"):
        assert f'"{key}"' in run, f"the report must carry {key}"
    # the artifact is never written by shadow: the bundle stays pre-launch
    assert "fit_level_calibration" not in run

    # both ratios are computed on the SAME rows -- only the factor differs
    assert "mu_arr * scale" in run, \
        "the re-fit reading must rescale mu, not re-predict it"
