"""fit_dispersion: a pinned r is a failed fit, and the residual frame has one
home."""

import copy
import inspect

import numpy as np
import pandas as pd
import pytest

from fit import fit_dispersion as fd


def test_an_r_at_a_search_bound_is_recognised_by_the_configured_tolerance(cfg):
    lo, hi = cfg["dispersion"]["r_search_bounds"]
    tol = cfg["dispersion"]["r_bound_tolerance_rel"]
    assert fd.r_at_bound(hi, cfg) and fd.r_at_bound(hi * (1 - tol / 2), cfg)
    assert fd.r_at_bound(lo, cfg) and fd.r_at_bound(lo * (1 + tol / 2), cfg)
    assert not fd.r_at_bound(hi * (1 - 2 * tol), cfg)
    assert not fd.r_at_bound(lo * (1 + 2 * tol), cfg)
    assert not fd.r_at_bound((lo + hi) / 2, cfg)
    # the drift measurement uses the SAME test, not its own 0.99 / 1.01
    src = inspect.getsource(fd.drift_by_window)
    assert "r_at_bound(" in src and "0.99" not in src and "1.01" not in src


class _FlatModel:
    """mu_ref = 2.0 everywhere; enough to form residuals."""
    @staticmethod
    def predict_mu_ref(rows, raw=False):
        return np.full(len(rows), 2.0)


def _calib_frame(cfg, groups):
    """`groups`: {subcategory: (category, rows)}; every row inside the calib
    window, stocked, eligible, at the anchor."""
    s = cfg["data"]["split"]
    rng = np.random.default_rng(0)
    rows = []
    for sub, (cat, n) in groups.items():
        for i in range(n):
            rows.append(dict(
                episode_id=f"{sub}-{i // 4}", date=s["calib_start"],
                hour_of_day=10 + i % 4, sku_id=sub, fc="F", category=cat,
                subcategory=sub, starting_inventory=10,
                units_sold=int(rng.negative_binomial(1.0, 1.0 / 3.0)),
                total_discount=0.25, d_ref=0.25,
                episode_eligible=True, dp_eligible=True))
    d = pd.DataFrame(rows)
    d["ending_inventory"] = (d.starting_inventory - d.units_sold).clip(lower=0)
    return d


def test_a_pinned_r_is_flagged_and_kept_out_of_the_clamp_percentile(
        cfg, monkeypatch):
    """A group whose MLE ran to r_search_bounds used to be stored like any
    other converged value AND to enter the clamp percentile -- so one failed
    fit at the ceiling dragged the cap toward the ceiling for everyone."""
    cfg = copy.deepcopy(cfg)
    cfg["dispersion"]["min_rows_per_group"] = 8
    lo, hi = cfg["dispersion"]["r_search_bounds"]
    # group identity reaches fit_r only through the arrays, so fits are keyed
    # on group size: S1 (8 rows) pins at the ceiling, the others are interior
    groups = {"S1": ("C1", 8), "S2": ("C1", 16), "S3": ("C2", 32)}
    by_size = {8: hi, 16: 2.0, 32: 3.0, 24: 4.0, 56: 2.5}   # C1=24, all=56
    monkeypatch.setattr(fd, "fit_r", lambda k, mu, cen, b: (by_size[len(k)], True))
    monkeypatch.setattr(fd, "BaselineModel", lambda c: _FlatModel())
    monkeypatch.setattr(fd, "_working_elasticity", lambda c: ({}, -1.0))
    # every group over-dispersed, so nothing is clamp-exempt on Pearson
    monkeypatch.setattr(fd, "pearson_dispersion", lambda k, mu: 5.0)

    r_lookup, rho_out = fd.fit_dispersion(_calib_frame(cfg, groups), cfg)

    assert r_lookup["at_bound"] == {"subcategory:S1": hi}
    assert r_lookup["global_at_bound"] is False
    interior = [2.0, 3.0, 4.0, 3.0, 2.5]         # S2, S3, C1, C2, global
    cap = float(np.percentile(interior, cfg["dispersion"]["clamp_percentile"] * 100))
    assert r_lookup["clamp_at"] == pytest.approx(cap)
    assert cap < hi / 2, "the pinned value must not pull the cap upward"
    # the pinned group is still in the lookup (the fallback chain needs a
    # value) -- clamped like any over-dispersed group, and flagged
    assert r_lookup["subcategory"]["S1"] == pytest.approx(cap)
    assert r_lookup["subcategory"]["S2"] == pytest.approx(2.0)
    assert "at_bound_note" in r_lookup
    assert rho_out["fit_window"] == "calib"
    # the residual basis is reported as what was USED: the per-category
    # means, with the constant named as the fallback for unseen categories
    # -- never a bare `working_elasticity` that was always the constant
    assert "working_elasticity" not in r_lookup
    assert r_lookup["working_elasticity_fallback"] == -1.0
    assert r_lookup["working_elasticity_by_category"] == {}


def test_lookup_r_walks_the_fallback_chain_and_treats_zero_as_a_value():
    r = {"subcategory": {"S": 0.0}, "category": {"C": 2.0}, "global": 9.0,
         "fallback_order": ["subcategory", "category", "global"]}
    assert fd.lookup_r(r, "S", "C") == 0.0          # 0.0 is a value, not a miss
    assert fd.lookup_r(r, "X", "C") == 2.0
    assert fd.lookup_r(r, "X", "Y") == 9.0
    # the vectorised form is the row rule, row for row
    sub = pd.Series(["S", "X", "X", "S"], index=[7, 3, 9, 1])
    cat = pd.Series(["C", "C", "Y", "Y"], index=[7, 3, 9, 1])
    got = fd.lookup_r_vec(r, sub, cat)
    assert list(got) == [fd.lookup_r(r, s, c) for s, c in zip(sub, cat)]
    assert got.dtype == float


def test_drift_windows_hold_whole_episodes_keyed_on_the_opening_week(
        cfg, monkeypatch):
    """A row-level `to_period` bucket split every window crossing a week
    seam into two clusters (rule 15) -- its Monday rows became a phantom
    one-hour episode in the next window's rho."""
    cfg = copy.deepcopy(cfg)
    cfg["dispersion"]["min_rows_per_group"] = 4
    monkeypatch.setattr(fd, "BaselineModel", lambda c: _FlatModel())
    monkeypatch.setattr(fd, "_working_elasticity", lambda c: ({}, -1.0))
    monkeypatch.setattr(fd, "pre_launch", lambda d, c: d)
    monkeypatch.setattr(fd, "population", lambda d, c: d)
    monkeypatch.setattr(fd, "fit_r", lambda k, mu, cen, b: (2.0, True))

    def hours(eid, day, hrs):
        return [dict(episode_id=eid, date=day, hour_of_day=h, category="A",
                     starting_inventory=10, units_sold=1, ending_inventory=9,
                     total_discount=0.25, d_ref=0.25) for h in hrs]
    rows = (hours("w1a", "2026-07-07", range(10, 14))          # Tue, week 1
            + hours("w1b", "2026-07-08", range(10, 14))
            + hours("w2a", "2026-07-14", range(10, 14))        # Tue, week 2
            + hours("w2b", "2026-07-15", range(10, 14))
            # opens Sunday 22:00 of week 1, runs into Monday of week 2
            + hours("SEAM", "2026-07-12", [22, 23])
            + hours("SEAM", "2026-07-13", [0, 1, 2]))
    d = pd.DataFrame(rows)

    drift = fd.drift_by_window(d, cfg)
    assert sorted(drift["by_window"]) == ["2026-07-06", "2026-07-13"]
    # all five SEAM rows sit in the week it OPENED in, none in the next
    assert drift["by_window"]["2026-07-06"]["rows"] == 8 + 5
    assert drift["by_window"]["2026-07-13"]["rows"] == 8


def test_the_working_elasticity_fallback_is_a_config_key(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg["posterior"]["prior"]["path"] = str(tmp_path / "absent.json")
    cfg["dispersion"]["working_elasticity_fallback"] = -1.7
    by_cat, fallback = fd._working_elasticity(cfg)
    assert by_cat == {} and fallback == -1.7
    assert "-1.0" not in inspect.getsource(fd._working_elasticity)


def test_the_residual_frame_has_one_home(cfg):
    """The frozen fit and drift_by_window built mu_hat / censored / resid in
    two hand-synced blocks; both now read `_residual_frame`."""
    for fn in (fd.fit_dispersion, fd.drift_by_window):
        src = inspect.getsource(fn)
        assert "_residual_frame(" in src, fn.__name__
        assert "ratio **" not in src, f"{fn.__name__} rebuilds mu_hat itself"

    d = pd.DataFrame({
        "episode_id": ["e"] * 3, "date": ["2026-07-01"] * 3,
        "hour_of_day": [10, 11, 12], "category": ["A", "A", "B"],
        "starting_inventory": [4, 0, 4], "units_sold": [1, 0, 4],
        "ending_inventory": [3, 0, 0], "total_discount": [0.5, 0.5, 0.5],
        "d_ref": [0.25, 0.25, 0.25]})
    f = fd._residual_frame(d, cfg, _FlatModel(), {"A": -2.0}, -1.0)
    # the zero-stock row is gone; A uses its own eps, B the fallback
    assert list(f.index) == [0, 2]
    ratio = 0.5 / 0.75
    assert f.mu_hat.iloc[0] == pytest.approx(2.0 * ratio ** -2.0)
    assert f.mu_hat.iloc[1] == pytest.approx(2.0 * ratio ** -1.0)
    assert (f.resid == f.units_sold - f.mu_hat).all()
    assert list(f.censored) == [False, True]


def test_an_under_dispersed_group_is_exempt_from_the_clamp():
    """The clamp must not make the steadiest cells claim variance they lack."""
    from fit.fit_dispersion import pearson_dispersion

    rng = np.random.default_rng(4)
    mu = 6.0
    # binomial with the same mean is UNDER-dispersed: var = mu(1-p) < mu
    tight = rng.binomial(12, mu / 12, 4000)
    assert pearson_dispersion(tight, np.full(len(tight), mu)) < 1.0
    # negative binomial at the same mean is over-dispersed
    loose = rng.negative_binomial(1.5, 1.5 / (1.5 + mu), 4000)
    assert pearson_dispersion(loose, np.full(len(loose), mu)) > 1.0
    # Poisson sits at 1 either side of noise
    poi = rng.poisson(mu, 20000)
    assert 0.9 < pearson_dispersion(poi, np.full(len(poi), mu)) < 1.1


def _weekly_frame(cfg, weeks_before, weeks_after, rows_per_week=8):
    """Whole episodes opening in `weeks_before` weeks up to train_end and
    `weeks_after` weeks past it, 4 stocked hours each, at the anchor."""
    train_end = pd.Timestamp(cfg["data"]["split"]["train_end"])
    monday = train_end.to_period("W").start_time
    rng = np.random.default_rng(1)
    rows = []
    for k in range(-weeks_before + 1, weeks_after + 1):
        day = monday + pd.Timedelta(weeks=k, days=1)      # a Tuesday
        for e in range(rows_per_week // 4):
            for h in range(10, 14):
                rows.append(dict(
                    episode_id=f"w{k}e{e}", date=str(day.date()),
                    hour_of_day=h, category="A", starting_inventory=10,
                    units_sold=int(rng.poisson(2.0)), ending_inventory=8,
                    total_discount=0.25, d_ref=0.25))
    return pd.DataFrame(rows)


def test_drift_is_graded_on_post_train_windows_and_says_so(cfg, monkeypatch):
    """drift_by_window fitted every pre-launch week and pooled in-train and
    post-train windows in one spread: the model fits its own residuals in
    train, so the seam read as drift. Each window names its `basis`; the
    medians, spread and verdict take the post-train windows alone, and the
    thresholds are config keys, not literals."""
    cfg = copy.deepcopy(cfg)
    cfg["dispersion"]["min_rows_per_group"] = 4
    cfg["dispersion"]["drift_min_windows"] = 2
    monkeypatch.setattr(fd, "_working_elasticity", lambda c: ({}, -1.0))
    monkeypatch.setattr(fd, "pre_launch", lambda d, c: d)
    monkeypatch.setattr(fd, "population", lambda d, c: d)
    # r keyed on the window's rows so the two sides are told apart: the
    # in-train weeks (12 rows) fit r=1.0, post-train (8 rows) r=3.0 / 5.0
    seen = iter([3.0, 5.0])
    monkeypatch.setattr(fd, "fit_r", lambda k, mu, cen, b:
                        (1.0, True) if len(k) == 12 else (next(seen), True))
    monkeypatch.setattr(fd, "pearson_dispersion", lambda k, mu: 2.0)
    d = pd.concat([_weekly_frame(cfg, 3, 0, rows_per_week=12),
                   _weekly_frame(cfg, 0, 2, rows_per_week=8)])
    # the model is passed in, not rebuilt (main shares one with fit_dispersion)
    drift = fd.drift_by_window(d, cfg, model=_FlatModel())

    basis = {w: v["basis"] for w, v in drift["by_window"].items()}
    train_end = cfg["data"]["split"]["train_end"]
    assert all((b == "post_train") == (w > train_end) for w, b in basis.items())
    assert list(basis.values()).count("post_train") == 2
    assert drift["stats_basis"] == "post_train"
    assert drift["windows_graded"] == 2 and drift["windows_fitted"] == 5
    assert drift["r_median"] == 4.0 and drift["r_spread"] == 2.0
    post = [v["rho"] for v in drift["by_window"].values()
            if v["basis"] == "post_train" and v["rho"] is not None]
    assert drift["rho_median"] == pytest.approx(float(np.median(post)))

    # fewer post-train windows than the key asks for: every window, and a
    # note that says the spread is then a floor
    cfg["dispersion"]["drift_min_windows"] = 3
    seen = iter([3.0, 5.0])
    fallback = fd.drift_by_window(d, cfg, model=_FlatModel())
    assert fallback["stats_basis"].startswith("all windows")
    assert "drift_min_windows" in fallback["stats_basis"]
    assert fallback["windows_graded"] == 5 and fallback["r_median"] == 1.0

    src = inspect.getsource(fd.drift_by_window)
    assert "drift_min_windows" in src and "drift_max_unusable_share" in src
    assert "< 3" not in src and "0.34" not in src


def test_dispersion_drift_separates_a_failed_fit_from_a_moved_parameter(
        cfg, monkeypatch, tmp_path):
    """The measurement that says whether freezing r and rho is defensible --
    and the trap it has to avoid: a window under-dispersed (Pearson < 1) or
    pinned at a bound has no r, and the verdict must say THAT, not "drift".
    The block is built here and round-tripped through the artifact writer,
    never read off the working tree."""
    from common.io import read_json, write_json

    steady = np.full(400, 2.0)
    assert fd.pearson_dispersion(steady, np.full(400, 2.0)) < 1.0
    rng = np.random.default_rng(0)
    bursty = rng.negative_binomial(0.7, 0.7 / (0.7 + 2.0), 400)
    assert fd.pearson_dispersion(bursty, np.full(400, 2.0)) > 1.0

    cfg = copy.deepcopy(cfg)
    cfg["dispersion"]["min_rows_per_group"] = 4
    cfg["dispersion"]["drift_min_windows"] = 3
    monkeypatch.setattr(fd, "_working_elasticity", lambda c: ({}, -1.0))
    monkeypatch.setattr(fd, "pre_launch", lambda d, c: d)
    monkeypatch.setattr(fd, "population", lambda d, c: d)
    monkeypatch.setattr(fd, "fit_r", lambda k, mu, cen, b: (2.0, True))
    # two of five post-train windows are under-dispersed
    pears = iter([0.8, 0.9, 1.5, 1.6, 1.7])
    monkeypatch.setattr(fd, "pearson_dispersion", lambda k, mu: next(pears))
    d = _weekly_frame(cfg, 0, 5)
    path = tmp_path / "rho.json"
    write_json(str(path), {"rho": 0.1, "drift_by_window":
                           fd.drift_by_window(d, cfg, model=_FlatModel())})
    drift = read_json(str(path))["drift_by_window"]

    for w, v in drift["by_window"].items():
        assert set(("basis", "pearson", "nb_expressible", "r_at_search_bound",
                    "r_usable")) <= set(v), w
        assert isinstance(v["nb_expressible"], bool)     # a JSON bool, not "False"
        if v["pearson"] < 1.0:
            assert not v["r_usable"], \
                f"{w}: Pearson < 1 but its r is being treated as a measurement"
    usable = [v["r"] for v in drift["by_window"].values() if v["r_usable"]]
    assert len(usable) == 3 and drift["r_windows_usable"] == 3
    assert drift["r_spread"] == pytest.approx(max(usable) - min(usable), abs=1e-3)
    # 2 of 5 unusable (0.4) is above the configured share: not a drift verdict
    assert drift["r_unusable_share"] > cfg["dispersion"]["drift_max_unusable_share"]
    assert "NOT FITTABLE AT THIS CADENCE" in drift["verdict"]
    # ...and below it, the verdict is about rho
    cfg["dispersion"]["drift_max_unusable_share"] = 0.5
    pears = iter([0.8, 0.9, 1.5, 1.6, 1.7])
    ok = fd.drift_by_window(d, cfg, model=_FlatModel())
    assert "rho varies" in ok["verdict"]


def test_a_global_r_that_does_not_converge_is_a_refusal_not_a_value(
        cfg, monkeypatch):
    """Groups that fail to converge are skipped; the global r ends every
    fallback and cannot be, so `r_global, _ = fit_group(calib)` banked a
    failed fit as the chain's last resort."""
    cfg = copy.deepcopy(cfg)
    cfg["dispersion"]["min_rows_per_group"] = 8
    monkeypatch.setattr(fd, "fit_r", lambda k, mu, cen, b: (2.0, len(k) < 40))
    monkeypatch.setattr(fd, "_working_elasticity", lambda c: ({}, -1.0))
    d = _calib_frame(cfg, {"S1": ("C1", 16), "S2": ("C2", 32)})    # 48 rows
    with pytest.raises(RuntimeError, match="global r fit did not converge"):
        fd.fit_dispersion(d, cfg, model=_FlatModel())
