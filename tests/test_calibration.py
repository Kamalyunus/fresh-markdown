"""Level factors: a bracket end is not a solve; one encoder; the loop stops
when its check fails."""

import copy
import inspect
import json

import numpy as np
import pandas as pd
import pytest

import bootstrap.train_baseline as tb


class _Model:
    """A fixed raw mu_ref per row; carries cfg like the real model."""
    def __init__(self, cfg, mu):
        self.cfg, self.mu = cfg, mu

    def predict_mu_ref(self, d, raw=False):
        return np.full(len(d), self.mu)


def _anchor_frame(cells):
    """`cells`: {subcategory: (category, rows, units_sold per row)}; every
    row at the anchor with plenty of stock."""
    rows = []
    for sub, (cat, n, sold) in cells.items():
        for i in range(n):
            rows.append(dict(episode_id=f"{sub}-{i}", subcategory=sub,
                             category=cat, total_discount=0.25, d_ref=0.25,
                             starting_inventory=100, units_sold=sold))
    return pd.DataFrame(rows)


R_LOOKUP = {"subcategory": {}, "category": {}, "global": 5.0,
            "fallback_order": ["subcategory", "category", "global"]}


def test_a_factor_pinned_at_the_bracket_is_flagged_not_returned_silently(cfg):
    """`_solve_level_factors` returned the literal bound when the bisection
    bracket did not contain the sold total. The value is unchanged (the
    caller still gets a number); the detail now says it is a bound."""
    cfg = copy.deepcopy(cfg)
    f_lo, f_hi = cfg["baseline_model"]["calibration_factor_search_bounds"]
    # HOT sells 60 an hour against mu 1.0: no factor inside the bracket
    # reaches it. COLD sells 1 against mu 1.0: interior.
    calib = _anchor_frame({"HOT": ("C", 30, 60), "COLD": ("C", 30, 1)})
    factors, detail, f_global = tb._solve_level_factors(
        calib, _Model(cfg, 1.0), k_shrink=0.0, min_anchor=10,
        tier_step=cfg["pricing"]["tier_step"], max_k=cfg["pricing"]["negbin_max_k"],
        r_lookup=R_LOOKUP)
    assert detail["HOT"]["raw_factor"] == pytest.approx(f_hi)
    assert detail["HOT"]["at_bound"] == "upper"
    assert "not a solved value" in detail["HOT"]["at_bound_note"]
    assert "at_bound" not in detail["COLD"]
    assert f_lo < detail["COLD"]["raw_factor"] < f_hi

    # the bracket is config, not a literal: narrow it and the pin moves
    cfg["baseline_model"]["calibration_factor_search_bounds"] = [0.5, 2.0]
    _, detail2, _ = tb._solve_level_factors(
        calib.copy(), _Model(cfg, 1.0), 0.0, 10, cfg["pricing"]["tier_step"],
        cfg["pricing"]["negbin_max_k"], R_LOOKUP)
    assert detail2["HOT"]["raw_factor"] == pytest.approx(2.0)
    assert detail2["HOT"]["at_bound"] == "upper"
    src = inspect.getsource(tb._solve_level_factors)
    for literal in ("0.1, 10.0", "range(20)"):
        assert literal not in src, f"{literal} is still a literal"
    assert "calibration_factor_bisection_steps" in src


def test_a_lower_pin_is_named_too(cfg):
    calib = _anchor_frame({"DEAD": ("C", 30, 0), "LIVE": ("C", 30, 1)})
    # DEAD sells nothing at all: solve_factor short-circuits to 1.0 (no
    # evidence), never a bound...
    _, detail, _ = tb._solve_level_factors(
        calib, _Model(cfg, 1.0), 1.0, 10, cfg["pricing"]["tier_step"],
        cfg["pricing"]["negbin_max_k"], R_LOOKUP)
    assert detail["DEAD"]["raw_factor"] == 1.0 and "at_bound" not in detail["DEAD"]
    # ...while a cell selling far LESS than the lowest factor predicts pins low
    f_lo = cfg["baseline_model"]["calibration_factor_search_bounds"][0]
    calib = _anchor_frame({"SLOW": ("C", 30, 1), "LIVE": ("C", 30, 50)})
    _, detail, _ = tb._solve_level_factors(
        calib, _Model(cfg, 50.0), 1.0, 10, cfg["pricing"]["tier_step"],
        cfg["pricing"]["negbin_max_k"], R_LOOKUP)
    assert detail["SLOW"]["at_bound"] == "lower"
    assert detail["SLOW"]["raw_factor"] == pytest.approx(f_lo)


def test_thin_cells_are_shrunk_toward_the_parent_not_held_at_one(cfg):
    """The payload used to SAY thin cells were 'left at 1.0'. They are not:
    every cell above the window floor follows its parent by k_shrink."""
    # THIN wants twice the factor FAT does, on a fortieth of the evidence
    calib = _anchor_frame({"THIN": ("C", 12, 6), "FAT": ("C", 300, 3)})
    factors, detail, f_global = tb._solve_level_factors(
        calib, _Model(cfg, 1.0), k_shrink=1000.0, min_anchor=10,
        tier_step=cfg["pricing"]["tier_step"], max_k=cfg["pricing"]["negbin_max_k"],
        r_lookup=R_LOOKUP)
    thin = detail["THIN"]
    assert thin["shrinkage_weight_on_self"] < 0.1
    assert thin["raw_factor"] > thin["parent_factor"] > 1.0
    assert factors["THIN"] != 1.0
    assert abs(factors["THIN"] - thin["parent_factor"]) < \
        0.2 * abs(thin["raw_factor"] - thin["parent_factor"])
    doc = tb.fit_level_calibration.__doc__
    assert "left at 1.0" not in doc and "stay 1.0" not in doc
    assert "shrunk toward its parent" in doc
    src = inspect.getsource(tb.fit_level_calibration)
    assert "left at 1.0" not in src and "shrunk toward" in src


def test_the_convergence_method_label_says_whether_the_resolve_was_kept(
        tmp_path, monkeypatch, cfg):
    """The label hard-coded 'artifact restored (dry run)' even under
    --commit-convergence, when the re-solve is kept on disk."""
    cfg = copy.deepcopy(cfg)
    path = str(tmp_path / "cal.json")
    cfg["baseline_model"]["calibration_factor_path"] = path
    old = {"factors": {"A": 1.0}, "schedule": {"by_week": {}}}
    json.dump(old, open(path, "w"))
    new = {"factors": {"A": 1.1}, "schedule": {"by_week": {}}}
    monkeypatch.setattr(tb, "fit_level_calibration",
                        lambda d, c: json.dump(new, open(path, "w")))

    dry = tb.check_calibration_convergence(None, cfg)
    assert "dry run" in dry["method"] and "KEPT" not in dry["method"]
    assert json.load(open(path))["factors"] == old["factors"]

    kept = tb.check_calibration_convergence(None, cfg, commit=True)
    assert "KEPT" in kept["method"] and "dry run" not in kept["method"]
    assert json.load(open(path))["factors"] == new["factors"]
    # the digests come from the one provenance walk, not a second one
    src = inspect.getsource(tb.check_calibration_convergence)
    assert "collect(cfg)" in src and "file_digest(" not in src
    assert set(dry["checked_against"]) <= {"prior", "r_lookup", "rho"}


def test_training_and_inference_share_one_feature_encoder(cfg):
    levels = {"category": ["A", "B"]}
    d = pd.DataFrame({"category": ["B", "A", "Z"], "x": ["1", "2.5", "3"]})
    X = tb.encode_features(d, ["category", "x"], ["category"], levels)
    assert list(X.category) == [1, 0, -1]           # unseen level -> -1
    assert list(X.x) == [1.0, 2.5, 3.0]
    assert list(X.columns) == ["category", "x"]
    for fn in (tb.train, tb.BaselineModel._matrix):
        assert "encode_features(" in inspect.getsource(fn), fn.__qualname__
        assert "pd.Categorical(" not in inspect.getsource(fn), fn.__qualname__


def test_the_vectorised_factor_vector_matches_the_row_by_row_rule(cfg):
    """Each row takes its own week's table, frozen rows the anchor, unfitted
    weeks the anchor -- checked against a plain per-row reference on rows
    that exercise every branch, counters included."""
    anchor = {"A": 1.5, "B": 0.8}
    schedule = {"2026-07-06": {"A": 1.1},               # B missing -> 1.0
                "2026-07-20": {"A": 1.3, "B": 0.9}}
    m = tb.BaselineModel.__new__(tb.BaselineModel)
    m.calibration, m.calibration_grain = anchor, "category"
    m.calibration_schedule = schedule
    m._reset_calibration_counters()
    m.freeze_calibration_from("2026-07-22")

    rng = np.random.default_rng(3)
    dates = pd.to_datetime("2026-07-01") + pd.to_timedelta(
        rng.integers(0, 30, 60), unit="D")
    rows = pd.DataFrame({"category": rng.choice(["A", "B", "C"], 60),
                         "date": dates.strftime("%Y-%m-%d")})
    got = m._factor_vector(rows)

    from common import episodes
    weeks = episodes.week_key(pd.to_datetime(rows.date))
    want, n_frozen, n_fallback, n_sched = [], 0, 0, 0
    for key, wk, dt in zip(rows.category, weeks, pd.to_datetime(rows.date)):
        if dt >= pd.Timestamp("2026-07-22"):
            n_frozen += 1
            want.append(anchor.get(key, 1.0))
        elif wk not in schedule:
            n_fallback += 1
            want.append(anchor.get(key, 1.0))
        else:
            n_sched += 1
            want.append(schedule[wk].get(key, 1.0))
    assert np.allclose(got, want)
    assert (m._cal_rows_frozen, m._cal_rows_fallback, m._cal_rows_scheduled) \
        == (n_frozen, n_fallback, n_sched)
    assert n_frozen and n_fallback and n_sched, "every branch must be hit"
    # no per-row Python loop over the frame
    src = inspect.getsource(tb.BaselineModel._factor_vector)
    assert "enumerate(zip(" not in src


def test_a_failing_convergence_check_stops_the_loop_with_its_own_message(
        monkeypatch, cfg):
    """When 5b itself crashed, the loop read no verdict, the stall test had
    nothing to fire on and the run went to --max-turns doing nothing."""
    import bootstrap.run as br

    calls = []

    def fake_step(label, args, fatal=True, quiet=False):
        calls.append(label)
        return 3 if "5b" in label else 0
    monkeypatch.setattr(br, "step", fake_step)
    monkeypatch.setattr(br, "convergence", lambda c: None)

    with pytest.raises(SystemExit) as exc:
        br.settle(cfg, max_turns=20)
    msg = str(exc.value)
    assert "5b convergence FAILED (exit 3)" in msg
    assert "--max-turns" in msg and "--check-only" in msg
    assert sum("5b" in c for c in calls) == 1, "it must not iterate on"
