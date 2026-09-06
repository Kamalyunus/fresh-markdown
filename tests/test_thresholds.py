"""derive_thresholds: one verdict per metric, its tunables in config, and
the noise floor measured on the same series the monitor triggers on."""

import copy
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from evaluate import derive_thresholds as dt
from conftest import CFG, episode_frame


def _trailing(three_sigma=0.2):
    return {"scrap_rate": {"three_sigma": three_sigma,
                           "three_sigma_robust": three_sigma,
                           "outlier_dominated": False},
            "margin_rate": {"three_sigma": three_sigma,
                            "three_sigma_robust": three_sigma,
                            "outlier_dominated": False}}


def test_the_inert_multiple_is_read_from_config(cfg):
    cfg = copy.deepcopy(cfg)
    floor = 0.2
    cfg["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = floor * 5
    rec = dt.recommend_thresholds(_trailing(floor), cfg)["scrap_rate"]
    assert "LIKELY INERT" in rec["verdict"]                 # shipped multiple 3
    assert "guardrail_inert_floor_multiple" in rec["verdict"]

    cfg["tuning"]["guardrail_inert_floor_multiple"] = 10
    rec = dt.recommend_thresholds(_trailing(floor), cfg)["scrap_rate"]
    assert rec["verdict"].startswith("OK")
    # the keys tune and status read are still there, and constant by design
    assert rec["binding_floor"] == rec["trailing_floor"] == floor
    assert rec["binding_basis"] == "trailing" and rec["binding_label"] == "3-sigma"
    assert "3 * floor" not in inspect.getsource(dt.recommend_thresholds)


def test_the_noise_block_measures_and_points_at_the_one_verdict(cfg):
    """guardrail_noise carried a second `verdict()` that re-graded the
    threshold with its own wording; recommend_thresholds is the grader."""
    src = inspect.getsource(dt.guardrail_noise)
    assert "def verdict" not in src and "TOO TIGHT" not in src
    assert "from common import episodes" not in inspect.getsource(dt)

    cfg = copy.deepcopy(cfg)
    # three closed one-hour episodes on three days: far too short a series
    d = episode_frame(
        [("e1", "2026-03-01"), ("e2", "2026-03-02"), ("e3", "2026-03-03")],
        columns=["episode_id", "date"], hour_of_day=10, starting_inventory=5,
        units_sold=3, ending_inventory=0, original_price=1000.0,
        offered_price=800.0, cost=500.0, hours_remaining=0, dp_eligible=True)
    out = dt.guardrail_noise(d, cfg)
    for metric in ("scrap_rate", "margin_rate"):
        block = out[metric]
        assert "verdict" not in block
        assert block["verdict_in"] == f"guardrail_threshold_recommendation.{metric}"
        assert block["config_key"].startswith("monitoring.stop_conditions.")
        need = (cfg["monitoring"]["guardrail_noise_window_days"]
                + cfg["monitoring"]["guardrail_noise_min_extra_days"])
        assert block["note"] == f"needs at least {need} days"
    # the extra-days margin is config, not `window + 7`
    cfg["monitoring"]["guardrail_noise_min_extra_days"] = 100
    assert "at least 128 days" in dt.guardrail_noise(d, cfg)["scrap_rate"]["note"]
    assert "window + 7" not in inspect.getsource(dt.guardrail_noise)


def test_outlier_dominance_uses_the_configured_sigma_ratio():
    rel = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 2.0])
    loose = dt._sigma_summary(rel, outlier_ratio=2.0)
    tight = dt._sigma_summary(rel, outlier_ratio=1e6)
    assert loose["outlier_dominated"] and not tight["outlier_dominated"]
    assert loose["three_sigma"] == tight["three_sigma"]
    assert "2 * sigma_robust" not in inspect.getsource(dt._sigma_summary)


def test_the_consistent_band_is_read_from_config(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps({"per_category": {
        "A": {"mean": -1.0, "std": 0.6}}}))
    cfg["posterior"]["prior"]["path"] = str(prior_path)
    # shrink 0.25 -> pull 0.4375; mean step 0.15 clips at 0.15/(0.4375*0.6)
    cfg["learning"]["max_std_shrink"] = 0.25
    cfg["learning"]["max_mean_step"] = 0.15
    clips_at = 0.15 / (0.4375 * 0.6)
    assert clips_at < 0.7
    assert dt.bounded_step(cfg)["verdict"].startswith("MEAN RAIL BINDS FIRST")

    cfg["tuning"]["bounded_step_consistent_band"] = [0.0, 10.0]
    out = dt.bounded_step(cfg)
    assert out["verdict"].startswith("CONSISTENT")
    assert out["consistent_band_std"] == [0.0, 10.0]
    src = inspect.getsource(dt.bounded_step)
    assert "0.7 <=" not in src and "<= 1.4" not in src


def test_noise_floor_and_monitor_use_the_same_smoothing():
    """The floor the owner sets a threshold from and the series the monitor
    triggers on must be averaged identically, or the threshold is graded
    against a yardstick nothing uses. (That both sides read the shared
    comparison and smoothing is asserted where the shared module is.)"""
    sm = CFG["monitoring"]["stop_conditions"]["deterioration_smoothing_days"]
    assert set(sm) == {"scrap", "margin"}
    # both series are averaged over the same week (owner, 2026-09-06)
    assert sm["scrap"] > 1 and sm["margin"] == sm["scrap"]


def _daily_frame(days=70, skus=60, seed=0, start="2026-01-01", dp=True):
    """Episode-hour rows over many days with a shared day effect, so the
    trailing-mean floor has genuine day-to-day swing to measure."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, day in enumerate(pd.date_range(start, periods=days)):
        day_effect = 1.0 + 0.35 * np.sin(i / 3.0)      # shared by both arms
        for sku in range(skus):
            sold = int(np.clip(rng.poisson(9 * day_effect), 0, 20))
            rows.append(dict(episode_id=f"{day.date()}|{sku}|{dp}", date=day.date(),
                             hour_of_day=10, sku_id=sku, fc="FC1",
                             starting_inventory=20, units_sold=sold,
                             ending_inventory=0, hours_remaining=0,
                             offered_price=1000.0, original_price=1000.0,
                             cost=600.0, dp_eligible=dp))
    return pd.DataFrame(rows)


def test_the_floor_is_measured_on_a_calendar_so_a_gap_is_not_a_seam():
    """The pre-launch series has data.exclusion_window cut out of it. Rolled
    over ROWS, the first post-gap days were smoothed with, and graded
    against, days six weeks earlier -- while the basis text said "trailing
    N-day mean". On the calendar each side of the gap warms up on its own:
    the scored days are exactly the two segments' scored days added."""
    pre = _daily_frame(70, start="2026-01-01")
    post = _daily_frame(70, start="2026-05-01", seed=1)     # a 50-day hole
    both = pd.concat([pre, post], ignore_index=True)
    for metric in ("scrap_rate", "margin_rate"):
        a, b = dt.guardrail_noise(pre, CFG)[metric], dt.guardrail_noise(post, CFG)[metric]
        gapped = dt.guardrail_noise(both, CFG)[metric]
        assert gapped["days"] == a["days"] + b["days"]
        assert gapped["days_scored"] == a["days_scored"] + b["days_scored"]
        # a row-rolled series would have scored every post-gap day but the
        # first: more scored days, seamed across the hole
        assert gapped["days_scored"] < a["days_scored"] + b["days"]
    assert "CALENDAR" in dt.guardrail_noise(both, CFG)["basis"]


def test_the_floor_is_measured_on_the_population_the_trigger_runs_on():
    """The monitor's series is the system-priced episodes -- the ones the DP
    could price. Measured on the integrity population, the floor carried
    the noise of episodes the guardrail will never see."""
    d = _daily_frame()
    # dp-INELIGIBLE episodes whose scrap swings wildly day to day
    wild = _daily_frame(seed=3, dp=False, skus=20)
    wild["units_sold"] = np.where(pd.to_datetime(wild.date).dt.day % 2 == 0, 0, 20)
    floor = dt.guardrail_noise(d, CFG)["scrap_rate"]["three_sigma"]
    assert dt.guardrail_noise(pd.concat([d, wild]), CFG)["scrap_rate"]["three_sigma"] == floor
    assert dt.guardrail_noise(pd.concat([d, wild.assign(dp_eligible=True)]),
                              CFG)["scrap_rate"]["three_sigma"] != floor


def test_a_relative_deviation_from_a_zero_control_is_no_reading():
    """t / 0 is not an infinite deterioration, it is undefined: the shared
    comparison returns NaN so neither the floor's sigma nor the trigger's
    streak ever sees +-inf (the monitor once replaced it locally)."""
    from common import guardrail
    t, c = pd.Series([0.1, 0.2, 0.3]), pd.Series([0.0, 0.1, 0.0])
    for worse_high in (True, False):
        dev = guardrail.deviation(t, c, worse_high, guardrail.RELATIVE)
        assert dev.isna().tolist() == [True, False, True]
        assert np.isfinite(dev.dropna()).all()
        # the absolute basis has no such singularity
        assert np.isfinite(guardrail.deviation(t, c, worse_high,
                                               guardrail.ABSOLUTE_PP)).all()


def test_trailing_floor_is_measured_on_the_smoothed_series():
    """Smoothing must actually be applied, not just mentioned. Measuring the
    same data at smoothing 1 must give a strictly wider floor -- if the two
    agree, the smoothing is being ignored and a threshold set from this floor
    sits several times above its true operating noise."""
    import copy
    from evaluate import derive_thresholds as dt

    d = _daily_frame()
    cfg_smoothed = copy.deepcopy(CFG)
    cfg_flat = copy.deepcopy(CFG)
    cfg_flat["monitoring"]["stop_conditions"]["deterioration_smoothing_days"] \
        ["scrap"] = 1

    smoothed = dt.guardrail_noise(d, cfg_smoothed)["scrap_rate"]
    flat = dt.guardrail_noise(d, cfg_flat)["scrap_rate"]

    assert smoothed["smoothing_days"] == \
        CFG["monitoring"]["stop_conditions"]["deterioration_smoothing_days"]["scrap"]
    assert flat["smoothing_days"] == 1
    assert smoothed["three_sigma"] < flat["three_sigma"]


def test_threshold_recommendation_grades_against_the_trailing_floor():
    """The monitor compares against the trailing mean of the same
    system-priced episodes (no control arm), so the threshold must clear
    that floor -- and a value far above it is called out rather than
    blessed."""
    import copy
    from evaluate import derive_thresholds as dt

    d = _daily_frame()
    cfg = copy.deepcopy(CFG)
    trailing = dt.guardrail_noise(d, cfg)

    rec = dt.recommend_thresholds(trailing, cfg)["scrap_rate"]
    assert rec["binding_floor"] == rec["trailing_floor"] > 0
    assert rec["binding_basis"] == "trailing"

    floor = rec["binding_floor"]
    sc = cfg["monitoring"]["stop_conditions"]

    sc["scrap_deterioration_pct"] = floor / 2
    assert "TOO TIGHT" in dt.recommend_thresholds(trailing, cfg)["scrap_rate"]["verdict"]

    sc["scrap_deterioration_pct"] = floor * 1.5
    assert dt.recommend_thresholds(trailing, cfg)["scrap_rate"]["verdict"].startswith("OK")

    # a guardrail that cannot fire is a failure mode of its own, not a pass
    sc["scrap_deterioration_pct"] = floor * 20
    assert "INERT" in dt.recommend_thresholds(trailing, cfg)["scrap_rate"]["verdict"]


def test_a_relative_floor_above_one_is_reported_as_blocked_not_as_a_number():
    """A floor at or above 1.0 means the series' ordinary daily swing exceeds
    its own level. No threshold can clear it without also clearing the failure
    the guardrail exists to catch -- so it is BLOCKED, and the report has to
    say the word."""
    import copy
    from evaluate import derive_thresholds as dt
    from common import guardrail

    assert guardrail.floor_is_unusable(1.0, guardrail.RELATIVE)
    assert guardrail.floor_is_unusable(3.5853, guardrail.RELATIVE)
    assert not guardrail.floor_is_unusable(0.4156, guardrail.RELATIVE)
    # absolute-pp floors are in the metric's own units and have no such bound
    assert not guardrail.floor_is_unusable(3.5853, guardrail.ABSOLUTE_PP)

    cfg = copy.deepcopy(CFG)
    trailing = {"margin_rate": {"three_sigma": 3.5853,
                                "three_sigma_robust": 3.5853,
                                "outlier_dominated": True,
                                "mean_level": 0.0308, "days": 134,
                                "days_at_or_below_zero": 36}}
    # on the wrong (relative) basis the floor is unusable and must BLOCK...
    saved = guardrail.BASIS["margin"]
    try:
        guardrail.BASIS["margin"] = guardrail.RELATIVE
        v = dt.recommend_thresholds(trailing, cfg)["margin_rate"]["verdict"]
    finally:
        guardrail.BASIS["margin"] = saved
    assert v.startswith("BLOCKED"), v
    assert "absolute_pp" in v, "the verdict must name the remedy, not just complain"

    # ...and on the shipped absolute_pp basis it is an ordinary, settable floor
    v2 = dt.recommend_thresholds(trailing, cfg)["margin_rate"]["verdict"]
    assert not v2.startswith("BLOCKED"), v2


def test_the_information_increment_is_derived_from_the_posterior_arithmetic(
        tmp_path):
    """`information_increment` is the evidence a bounded update may USE, and
    that is fixed by the posterior's own algebra rather than judgment."""
    import copy
    import json
    import numpy as np

    from evaluate import derive_thresholds as dt

    cfg = copy.deepcopy(CFG)
    prior_path = tmp_path / "prior.json"
    # two cells an octave apart in width: 4x the precision, 4x the cost
    prior_path.write_text(json.dumps({"per_category": {
        "WIDE": {"mean": -1.0, "std": 1.0},
        "NARROW": {"mean": -1.0, "std": 0.5}}}))
    cfg["posterior"]["prior"]["path"] = str(prior_path)
    cfg["learning"]["max_std_shrink"] = 0.25

    out = dt.information_increment(cfg)
    k = 1.0 / 0.75 ** 2 - 1.0
    need = out["information_to_saturate_cap_by_category"]
    assert need["WIDE"] == pytest.approx(k, abs=1e-3)
    assert need["NARROW"] == pytest.approx(k / 0.25, abs=1e-3)
    assert need["NARROW"] == pytest.approx(4 * need["WIDE"], rel=1e-3), \
        "halving the std must quadruple the information -- precision is 1/s^2"

    # the ceiling is checkable against the posterior step it claims to fund:
    # spending exactly I* on the wide cell lands on the shrink cap
    s0 = 1.0
    s1 = 1.0 / np.sqrt(1.0 / s0 ** 2 + need["WIDE"])
    # tolerance is what the report's 3dp rounding permits, nothing looser
    assert s1 == pytest.approx(s0 * (1 - cfg["learning"]["max_std_shrink"]),
                               rel=1e-3)

    # and an over-sized configured value is named, with the std it was
    # implicitly sized for -- the number that says what it was fitted to
    cfg["learning"]["information_increment"] = 12.0
    over = dt.information_increment(cfg)
    assert over["verdict"].startswith("TOO LARGE")
    assert over["configured_implied_std"] == pytest.approx(
        np.sqrt(k / 12.0), abs=1e-3)
    assert over["wastes_at_launch"] > 1

    cfg["learning"]["information_increment"] = over["recommended"]
    assert dt.information_increment(cfg)["verdict"].startswith("OK")


def test_the_two_bounded_step_rails_are_graded_against_each_other(tmp_path):
    """`max_mean_step` and `max_std_shrink` are one decision expressed twice.
    A cap-sized update moves the mean by [1-(1-shrink)^2] x pull, so if the
    mean rail sits far below that it clips every ordinary batch while the
    shrink rail never binds -- and `bound_clipped` stops meaning anything.
    The report has to say which rail binds first, and at what surprise."""
    import copy
    import json

    from evaluate import derive_thresholds as dt

    cfg = copy.deepcopy(CFG)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(
        {"per_category": {"A": {"mean": -1.0, "std": 1.0}}}))
    cfg["posterior"]["prior"]["path"] = str(prior_path)
    cfg["learning"]["max_std_shrink"] = 0.25

    # a mean rail far under the cap-sized move: clips on ordinary batches
    cfg["learning"]["max_mean_step"] = 0.15
    tight = dt.bounded_step(cfg)
    assert tight["mean_move_fraction_of_pull_at_cap"] == pytest.approx(0.4375)
    assert tight["consistent_max_mean_step"] == pytest.approx(0.4375, abs=1e-3)
    assert tight["mean_rail_clips_above_pull_of_std"] == pytest.approx(
        0.15 / 0.4375, abs=1e-2)
    assert tight["verdict"].startswith("MEAN RAIL BINDS FIRST")
    # the verdict must name the owner's actual options, not just complain
    assert "step_sensitivity" in tight["verdict"]

    # set to the consistent value and both rails trip at a 1-std surprise
    cfg["learning"]["max_mean_step"] = tight["consistent_max_mean_step"]
    assert dt.bounded_step(cfg)["verdict"].startswith("CONSISTENT")

    # and the shrink cap alone fixes convergence: 25%/update from 1.0 to the
    # 0.05 floor is log(0.05)/log(0.75) updates, nothing to do with the mean
    assert dt.bounded_step(cfg)["updates_to_min_std_by_category"]["A"] == \
        pytest.approx(np.log(cfg["posterior"]["min_std"]) / np.log(0.75), abs=0.1)


def test_the_floor_and_the_trigger_compute_the_same_quantity():
    """`derive_thresholds` measures the floor and `daily.monitor` evaluates
    the trigger. If they compute different things the threshold is graded
    against a yardstick nothing uses, and nothing downstream would notice --
    so both import the comparison from `common.guardrail` rather than keeping
    two implementations that resemble each other."""
    import inspect
    from evaluate import derive_thresholds as dt
    from daily import monitor

    for mod in (dt, monitor):
        src = inspect.getsource(mod)
        assert "guardrail.deviation(" in src or "guard.deviation(" in src, \
            f"{mod.__name__} is not using the shared comparison"
        assert "guardrail.smooth(" in src or "guard.smooth(" in src \
            or "_smooth = guardrail.smooth" in src, \
            f"{mod.__name__} is not using the shared smoothing"

    # the sign convention is the caller's, and it must be opposite per metric
    from common import guardrail
    import pandas as pd
    t, c = pd.Series([0.12]), pd.Series([0.10])
    for basis in (guardrail.RELATIVE, guardrail.ABSOLUTE_PP):
        # scrap: higher is worse -> positive
        assert guardrail.deviation(t, c, True, basis).iloc[0] > 0
        # margin: higher is BETTER -> negative
        assert guardrail.deviation(t, c, False, basis).iloc[0] < 0
