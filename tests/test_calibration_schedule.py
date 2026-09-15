"""The rolling level-factor schedule: point-in-time factors that never see
their own week, the freeze at the gate, convergence of the f <-> r loop, and
the --apply gate on a schedule that no longer reaches today."""

import json

import numpy as np
import pandas as pd
import pytest

from common import windows
from conftest import _Applier, _calib_frame, _harness_cfg, _prepared, load_config
from fit import calibrate as cal
from fit import train_baseline as tb



def _schedule_artifact(cfg, monkeypatch, cells_by_day, weeks_back=1, k_shrink=0.0):
    """Fit the calibration artifact on a constructed frame under tmp paths:
    `cells_by_day(day) -> {sub: (cat, sold)}` gives each day's cells (one
    4-hour anchor episode per cell per day, mu_ref 1.0), over the three
    weeks before the gate and the gate week. Returns (artifact, frame)."""
    cfg["baseline_model"]["calibration_fit_trailing_weeks"] = weeks_back
    cfg["baseline_model"]["calibration_shrinkage_units"] = k_shrink
    gate = pd.Timestamp(cfg["data"]["split"]["test_start"])
    days = [str(x.date()) for x in
            pd.date_range(gate - pd.Timedelta(days=21), gate + pd.Timedelta(days=6))]
    d = pd.concat([_prepared(cells_by_day(day), [day]) for day in days],
                  ignore_index=True)
    monkeypatch.setattr(cal, "BaselineModel", lambda c: _Applier(cfg, 1.0))
    cal.fit_level_calibration(d, cfg)
    return json.load(open(cfg["baseline_model"]["calibration_factor_path"])), d


def test_calibration_factors_never_see_their_own_week_or_later(scratch_cfg, monkeypatch):
    """The whole point of the rolling schedule: week W's factors are fit on
    the trailing window ENDING STRICTLY BEFORE W. Checked against a factor
    computed independently from the rows each week may read (raw basis,
    no shrinkage: sold over predicted on the trailing window's whole
    episodes) -- on a frame where the level jumps mid-way, so the jump
    week's table cannot contain it."""
    cfg = scratch_cfg
    gate = pd.Timestamp(cfg["data"]["split"]["test_start"])
    jump = str((gate - pd.Timedelta(days=10)).date())

    def cells(day):
        return {"A": ("C", 6 if day >= jump else 2)}
    art, d = _schedule_artifact(cfg, monkeypatch, cells)
    sched = art["schedule"]
    assert sched["trailing_weeks"] == 1 and len(sched["by_week"]) >= 3
    opened = windows.opening_dates(d)
    for week, table in sched["by_week"].items():
        window, _ = windows.trailing_weeks_window(d, week, 1, opened=opened)
        assert (opened.loc[window.index] < week).all(), "a row of its own week"
        want = float(window.units_sold.sum()) / float(len(window))   # mu_ref 1.0
        assert table["A"] == pytest.approx(want, abs=1e-3), week
    weeks = sorted(sched["by_week"])
    jump_week = windows.week_key(pd.Series([jump])).iloc[0]
    before = [w for w in weeks if w <= jump_week]
    after = [w for w in weeks if w > jump_week]
    assert before and after
    # the jump week's table is fit on the week before it: the old level
    assert sched["by_week"][before[-1]]["A"] == pytest.approx(2.0, abs=1e-3)
    # only a week AFTER the jump can carry it, and the last one carries it whole
    assert sched["by_week"][after[-1]]["A"] == pytest.approx(6.0, abs=1e-3)
    assert all(sched["by_week"][w]["A"] < 6.0 for w in before)


def test_both_harnesses_get_point_in_time_factors_without_their_own_code():
    """backtest and shadow must not each re-implement factor selection: they
    call predict_mu_ref on frames carrying `date`, so the schedule reaches
    them through the one applier. A second copy would drift."""
    import inspect
    from evaluate import backtest as replay
    from evaluate import level
    from evaluate import shadow

    # one prediction path: `evaluate.level.predict_frame` calls the applier,
    # both harnesses predict through it (each once carried a copy of
    # extend/lookup/predict)
    assert "predict_mu_ref(" in inspect.getsource(level.predict_frame)
    assert "predict_frame(" in inspect.getsource(shadow._prepare_items)
    assert "predict_frame(" in inspect.getsource(replay)
    for mod, name in ((replay, "evaluate.backtest"), (shadow, "evaluate.shadow"),
                      (level, "evaluate.level")):
        src = inspect.getsource(mod)
        # `by_week` is deliberately NOT banned: fidelity has its own weekly
        # series. What must not appear is the selection itself or the
        # artifact path. (Shadow swaps the model's schedule for the table it
        # fits locally -- rule 16 keeps that table out of the artifact --
        # and reads BOTH regimes' factors through `level_factors`, the one
        # selection; that is using the applier, not re-implementing it.)
        for banned in ("calibration_factor_path",):
            assert banned not in src, (
                f"{name} reaches into the calibration schedule itself -- "
                "factor selection belongs to BaselineModel alone")
        # the applier is read directly, or through evaluate.level -- the one
        # home of the frozen-vs-refit reading both harnesses share
        assert "level_factors(" in src or "refit_scale(" in src, \
            f"{name} does not read the applier"


def test_a_level_shift_does_not_leak_into_its_own_weeks_factor(tmp_path, cfg):
    """The functional half of the point-in-time guarantee."""
    import json

    import pandas as pd

    from fit.train_baseline import BaselineModel

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
    f = model.level_factors(rows)

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

    from fit.train_baseline import BaselineModel

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
    f = model.level_factors(rows)
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


def test_apply_refuses_when_the_weekly_refit_was_missed(tmp_path, cfg):
    """Detection after the fact is not enough. Learning from prices set on
    stale factors banks evidence about a model that is not the one running,
    so a schedule that no longer reaches today is a HARD gate on --apply --
    the same standing as a duplicate-event breach."""
    import json

    from daily.update import calibration_current

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


def test_a_partial_trailing_window_is_counted_not_passed_off_as_full(scratch_cfg, monkeypatch):
    """`trailing_weeks: 2` does not mean every week HAS two behind it: the
    first fitted weeks have one, and are flagged, not dropped."""
    art, _ = _schedule_artifact(scratch_cfg, monkeypatch,
                                lambda day: {"A": ("C", 2)}, weeks_back=2)
    sched = art["schedule"]
    assert sched["weeks_on_partial_window"], \
        "a short trailing window must be counted, not silently labelled full"
    for row in sched["weeks_on_partial_window"]:
        assert row["weeks_in_window"] < sched["trailing_weeks"]
        assert row["week"] in sched["by_week"], \
            "a partial week is still fitted -- it is flagged, not dropped"
    full = [w for w in sched["by_week"]
            if w not in {r["week"] for r in sched["weeks_on_partial_window"]}]
    assert full, "later weeks have the whole window behind them"


def test_the_gate_freezes_calibration_even_though_the_schedule_runs_past_it(scratch_cfg, monkeypatch):
    """Two questions, one artifact, and they must not be confused."""
    cfg, BaselineModel = scratch_cfg, tb.BaselineModel
    art, _ = _schedule_artifact(cfg, monkeypatch, lambda day: {"A": ("C", 2)})
    sched = art["schedule"]
    gate_start = pd.Timestamp(cfg["data"]["split"]["test_start"])
    assert sched.get("gate_freezes_at") == str(gate_start.date()), (
        "the artifact must record where the gate freezes, or coverage cannot "
        "tell a deliberate freeze from production running onto stale factors")
    # the schedule itself runs past the gate (pre-launch scope: to test_end)
    assert max(sched["by_week"]) >= str(gate_start.date())

    # (that the fidelity gate does the freezing is exercised in
    # test_fidelity_grades_the_frozen_artifact_and_reports_the_refit_beside_it)

    # frozen rows take the anchor, scheduled rows take their own week
    m = BaselineModel.__new__(BaselineModel)
    m.calibration_grain = "category"
    m.calibration = {"A": 2.0}
    m.calibration_schedule = {"2026-07-27": {"A": 5.0},
                              "2026-08-03": {"A": 9.0}}
    m._reset_calibration_counters()
    frame = pd.DataFrame({"date": ["2026-07-28", "2026-08-03"],
                          "category": ["A", "A"]})

    m.freeze_calibration_from(None)
    assert list(m.level_factors(frame)) == [5.0, 9.0], \
        "unfrozen, every row takes its own week -- production's mechanism"

    m._reset_calibration_counters()
    m.freeze_calibration_from(gate_start)
    assert list(m.level_factors(frame)) == [2.0, 2.0], \
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
    monkeypatch.setattr(cal, "fit_level_calibration", fake(
        {"factors": {"A": 1.0, "B": 1.3},
         "schedule": {"by_week": {"2026-07-06": {"A": 1.05}}}}))
    block = cal.check_calibration_convergence(None, cfg)
    assert not block["converged"]
    assert block["worst_cell"] == "anchor:B"
    assert abs(block["max_abs_dlog"] - abs(np.log(1.3 / 1.2))) < 1e-5
    disk = json.load(open(path))
    assert disk["factors"] == old["factors"], "dry run must restore"
    assert disk["convergence"]["converged"] is False

    # identical re-solve -> converged; a schedule week's cell moving is
    # caught the same way as the anchor's
    monkeypatch.setattr(cal, "fit_level_calibration", fake(old))
    assert cal.check_calibration_convergence(None, cfg)["converged"]
    monkeypatch.setattr(cal, "fit_level_calibration", fake(
        {"factors": {"A": 1.0, "B": 1.2},
         "schedule": {"by_week": {"2026-07-06": {"A": 1.30}}}}))
    block = cal.check_calibration_convergence(None, cfg)
    assert not block["converged"] and block["worst_cell"] == "2026-07-06:A"

    # a cell appearing or vanishing is never averaged away
    monkeypatch.setattr(cal, "fit_level_calibration", fake(
        {"factors": {"A": 1.0}, "schedule": {"by_week": {}}}))
    block = cal.check_calibration_convergence(None, cfg)
    assert not block["converged"]
    assert "anchor:B" in block["cells_appeared_or_gone"]


def test_rho_is_fit_on_the_calib_window_not_the_full_frame(cfg, monkeypatch):
    """rho once read the whole input frame: mostly training rows, where the
    model fits its own residuals and between-episode variance reads small
    -- plus, on a longer extract, rows past test_end (hard rule 16). An
    understated rho understates deff, and deff deflates every posterior
    update. calib is the one window both out-of-train and pre-gate, and r
    already lives there. Observed on the rows the ICC is handed."""
    import copy

    from fit import fit_dispersion as fd
    cfg = copy.deepcopy(cfg)
    cfg["dispersion"]["min_rows_per_group"] = 8
    monkeypatch.setattr(fd, "fit_r", lambda k, mu, cen, b: (2.0, True))
    monkeypatch.setattr(fd, "_working_elasticity", lambda c: ({}, -1.0))
    monkeypatch.setattr(fd, "pearson_dispersion", lambda k, mu: 5.0)
    calib = _calib_frame(cfg, {"S1": ("C1", 16), "S2": ("C2", 32)})
    train = calib.assign(date=cfg["data"]["split"]["train_start"],
                         episode_id=lambda f: "train-" + f.episode_id)
    seen = []
    real = fd.intraclass_correlation

    def spy(resid, groups, clip_max=None):
        seen.append(len(resid))
        return real(resid, groups, clip_max)
    monkeypatch.setattr(fd, "intraclass_correlation", spy)
    _, rho_out = fd.fit_dispersion(pd.concat([train, calib], ignore_index=True),
                                   cfg, model=_Applier(cfg))
    assert rho_out["fit_window"] == "calib"
    assert seen == [len(calib)], "rho read rows outside the calib window"
    assert rho_out["fit_rows"] == len(calib)


def test_convergence_carries_its_trajectory_and_the_worst_cell_s_evidence(tmp_path, cfg):
    """A single reading cannot tell a contracting loop from a stuck one, and
    an unweighted max cannot tell an unsettled chain from one thin cell."""
    import copy

    cfg = copy.deepcopy(cfg)
    path = str(tmp_path / "cal.json")
    cfg["baseline_model"]["calibration_factor_path"] = path
    cfg["baseline_model"]["calibration_convergence_tol_log"] = 0.02
    json.dump({"factors": {"A": 1.0, "B": 1.2},
               "detail": {"B": {"anchor_rows": 37}},
               "schedule": {"by_week": {}}}, open(path, "w"))
    moves = iter([1.4, 1.3, 1.25])                  # a contracting loop

    def refit(d, c):
        json.dump({"factors": {"A": 1.0, "B": next(moves)},
                   "schedule": {"by_week": {}}}, open(path, "w"))
    import unittest.mock as um
    with um.patch.object(cal, "fit_level_calibration", refit):
        blocks = [cal.check_calibration_convergence(None, cfg) for _ in range(3)]
    for block in blocks:
        assert isinstance(block.get("history"), list) and block["history"]
        assert len(block["history"]) <= 6, "the history is bounded"
        assert block["history"][-1] == block["max_abs_dlog"], \
            "the last reading must be this run's"
        # the worst cell's evidence is reported, so a shrinkage-dominated
        # cell is distinguishable from a genuinely unsettled loop
        assert block["worst_cell"] == "anchor:B"
        assert block["worst_cell_anchor_rows"] == 37
    assert [len(b["history"]) for b in blocks] == [1, 2, 3]
    assert blocks[-1]["history"] == sorted(blocks[-1]["history"], reverse=True)
    assert "contracting" in blocks[-1]["verdict"]


def test_the_prior_fast_path_drops_only_what_cannot_move_the_fixed_point():
    """`fold_spread` only widens the std FLOOR -- while the loop compares
    FACTORS, which follow `r`, which is fitted at the prior MEAN. Skipping it
    is most of the prior's in-loop cost, now that `design_comparison` (which
    fed nothing at all) is gone rather than merely skipped."""
    import inspect

    from fit import prior_density

    assert "fast" in inspect.signature(prior_density.estimate).parameters
    assert "design_comparison" not in inspect.getsource(prior_density), \
        "the alternative-design block was deleted, not made conditional"


def test_apply_is_refused_when_the_calibration_schedule_is_stale(cfg, tmp_path):
    """The gate must bite inside update.run, beside the event-quality gates:
    a schedule that no longer reaches today refuses --apply."""
    from daily.update import run as update_run
    cfg = _harness_cfg(cfg, tmp_path)
    cal = tmp_path / "calibration.json"
    cfg["baseline_model"] = dict(cfg["baseline_model"],
                                 calibration_factor_path=str(cal))
    cal.write_text(json.dumps({
        "grain": "category", "factors": {"FRUIT": 1.2},
        "schedule": {"mode": "rolling_trailing", "trailing_weeks": 4,
                     "by_week": {"2026-07-06": {"FRUIT": 1.1}}}}))
    late = update_run(cfg, apply=True, events_root=str(tmp_path / "ev"),
                      posterior_path=cfg["posterior"]["path"], today="2026-07-21")
    assert not late["event_quality_gates"]["calibration_schedule_current"]["pass"]
    assert "calibration_schedule_current" in late["refused"]
    assert not late["applied"]
    current = update_run(cfg, apply=True, events_root=str(tmp_path / "ev"),
                         posterior_path=cfg["posterior"]["path"], today="2026-07-08")
    assert current["event_quality_gates"]["calibration_schedule_current"]["pass"]
    assert "refused" not in current
