"""Tests for backtest.replay's calibration window sweep.

The sweep ranks the calibration fit window, and pipeline.tune pastes the
winner -- so the comparison it runs has to be a fair one.
"""
import copy

import numpy as np
import pandas as pd

from backtest.replay import calibration_window_sweep


def _anchor_rows(weeks, categories=("VEG", "FRUIT"), seed=7):
    """Anchor rows (total_discount == d_ref) the sweep can group by week."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-01-05", periods=7 * weeks, freq="D")
    return pd.DataFrame([
        {"date": str(d.date()), "category": c,
         "total_discount": 0.30, "d_ref": 0.30,
         "units_sold": float(rng.integers(40, 60)), "predicted_units": 50.0}
        for d in days for c in categories])


def test_every_sweep_row_is_scored_on_the_same_weeks(cfg):
    """Per-window burn-in judged an 8w window on 11 weeks and a 2w window on
    17 DIFFERENT weeks, so the ranking read which weeks, not which window.
    One common eval set, and `uncalibrated` ranked with the rest."""
    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [1, 2, 4]
    out = calibration_window_sweep(_anchor_rows(12), cfg)

    scored = {k: v["eval_weeks"] for k, v in out.items()
              if isinstance(v, dict) and "eval_weeks" in v}
    assert len(set(scored.values())) == 1, scored
    assert {"uncalibrated", "trailing_1w", "trailing_2w",
            "trailing_4w"} <= set(scored)
    assert out["eval_weeks_common_from"]
    # uncalibrated is ranked, but the PASTE target stays a real window: W=0
    # is not a config value
    assert out["recommended_fit_window"].startswith("trailing_")
    assert isinstance(out["uncalibrated_beats_all_windows"], bool)


def test_no_factors_winning_is_flagged_not_hidden(cfg):
    """On flat data the factors only add estimation noise, so uncalibrated
    wins -- and the sweep must say so instead of silently ranking the
    least-bad window."""
    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [1, 2]
    out = calibration_window_sweep(_anchor_rows(10), cfg)

    unc, best = out["uncalibrated"], out[out["recommended_fit_window"]]
    beats = ((-unc["share_weeks_in_band"], unc["mean_abs_log_error"])
             < (-best["share_weeks_in_band"], best["mean_abs_log_error"]))
    assert out["uncalibrated_beats_all_windows"] is beats
    assert beats, "flat anchor data: factors cannot beat no factors"
    assert "NO-FACTORS WINS" in out["verdict"]


def test_the_sweep_refuses_rather_than_score_a_stub_eval_set(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [8]
    out = calibration_window_sweep(_anchor_rows(3), cfg)
    assert isinstance(out, str) and out.startswith("NOT RUN")


def test_the_sweep_says_when_it_cannot_tell(cfg):
    """The ranking compares aggregates over ~10 weeks and then turns on a
    lexicographic tie-break, so ONE week of share_weeks_in_band can decide
    which window 'wins'. The paired test asks the question that matters --
    same week, did the factors move the ratio closer to 1 -- and says so when
    the answer is undecidable."""
    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [1, 2]
    out = calibration_window_sweep(_anchor_rows(10), cfg)

    for key in ("trailing_1w", "trailing_2w"):
        pv = out[key]["paired_vs_uncalibrated"]
        assert pv["weeks_paired"] == out[key]["eval_weeks"]
        assert 0 <= pv["weeks_calibration_helped"] <= pv["weeks_paired"]
        assert 0.0 <= pv["sign_test_p"] <= 1.0

    # flat anchor data: nothing to correct, so no window can separate
    assert out["calibration_earns_its_keep"].startswith("UNDECIDED")
    assert "tie-break, not a measurement" in out["calibration_earns_its_keep"]


def test_a_window_that_genuinely_helps_is_called_out(cfg):
    """A persistent per-category level offset is exactly what the factors
    exist to remove; the paired test must find it."""
    import numpy as np
    import pandas as pd

    cfg = copy.deepcopy(cfg)
    cfg["baseline_model"]["calibration_window_sweep_weeks"] = [2]
    rng = np.random.default_rng(3)
    days = pd.date_range("2026-01-05", periods=7 * 14, freq="D")
    rows = []
    for d in days:
        for c, bias in (("VEG", 1.6), ("FRUIT", 0.6)):   # stable, large offset
            rows.append({"date": str(d.date()), "category": c,
                         "total_discount": 0.30, "d_ref": 0.30,
                         "units_sold": 50.0 * bias + rng.normal(0, 1.0),
                         "predicted_units": 50.0})
    out = calibration_window_sweep(pd.DataFrame(rows), cfg)

    pv = out["trailing_2w"]["paired_vs_uncalibrated"]
    assert pv["verdict"] == "calibration helps", pv
    assert pv["median_abs_log_delta"] < 0            # error moved toward zero
    assert out["calibration_earns_its_keep"].startswith("YES")
