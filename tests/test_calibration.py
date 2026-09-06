"""Level factors: a bracket end is not a solve; one encoder; the loop stops
when its check fails."""

import copy
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from fit import train_baseline as tb


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
    factors, detail, f_global, global_at_bound, _ = tb._solve_level_factors(
        calib, _Model(cfg, 1.0), k_shrink=0.0, min_anchor=10,
        tier_step=cfg["pricing"]["tier_step"], max_k=cfg["pricing"]["negbin_max_k"],
        r_lookup=R_LOOKUP)
    assert detail["HOT"]["raw_factor"] == pytest.approx(f_hi)
    assert detail["HOT"]["at_bound"] == "upper"
    assert "not a solved value" in detail["HOT"]["at_bound_note"]
    assert "at_bound" not in detail["COLD"]
    assert f_lo < detail["COLD"]["raw_factor"] < f_hi
    # the GLOBAL solve pools both cells: ~30 sold per unit predicted, so it
    # pins too -- and that used to be discarded (rule 3)
    assert global_at_bound == "upper" and f_global == pytest.approx(f_hi)

    # the bracket is config, not a literal: narrow it and the pin moves
    cfg["baseline_model"]["calibration_factor_search_bounds"] = [0.5, 2.0]
    _, detail2, _, _, _ = tb._solve_level_factors(
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
    _, detail, _, _, _ = tb._solve_level_factors(
        calib, _Model(cfg, 1.0), 1.0, 10, cfg["pricing"]["tier_step"],
        cfg["pricing"]["negbin_max_k"], R_LOOKUP)
    assert detail["DEAD"]["raw_factor"] == 1.0 and "at_bound" not in detail["DEAD"]
    # ...while a cell selling far LESS than the lowest factor predicts pins low
    f_lo = cfg["baseline_model"]["calibration_factor_search_bounds"][0]
    calib = _anchor_frame({"SLOW": ("C", 30, 1), "LIVE": ("C", 30, 50)})
    _, detail, _, _, _ = tb._solve_level_factors(
        calib, _Model(cfg, 50.0), 1.0, 10, cfg["pricing"]["tier_step"],
        cfg["pricing"]["negbin_max_k"], R_LOOKUP)
    assert detail["SLOW"]["at_bound"] == "lower"
    assert detail["SLOW"]["raw_factor"] == pytest.approx(f_lo)


def test_a_pinned_global_factor_is_flagged_not_discarded(cfg):
    """`f_global, _, _ = solve_factor(anchor_all)` threw the global solve's
    at_bound away, so every cell shrank toward a bracket end that nothing
    reported as one (rule 3). It is returned, written to the artifact and
    printed."""
    f_lo, f_hi = cfg["baseline_model"]["calibration_factor_search_bounds"]
    # EVERY cell sells far more than mu predicts: the global solve pins high
    calib = _anchor_frame({"HOT": ("C", 30, 60), "HOTTER": ("C", 30, 80)})
    factors, detail, f_global, global_at_bound, _ = tb._solve_level_factors(
        calib, _Model(cfg, 1.0), 1.0, 10, cfg["pricing"]["tier_step"],
        cfg["pricing"]["negbin_max_k"], R_LOOKUP)
    assert global_at_bound == "upper" and f_global == pytest.approx(f_hi)
    assert all(v["at_bound"] == "upper" for v in detail.values())

    art = {"grain": "subcategory", "detail": detail, "factors": factors,
           "global_factor": f_global, "global_factor_at_bound": global_at_bound,
           "fit_window": "w", "fit_window_dates": ["a", "b"], "fit_rows": 60,
           "fit_basis": "b", "fit_in_sample_share": 0.0}
    text = tb._describe_calibration(art, factors)
    assert "AT UPPER BOUND" in text.split("\n")[0], "the global pin is on line 1"
    # the artifact writer carries the flag by name (payload key, not detail)
    src = inspect.getsource(tb.fit_level_calibration)
    assert '"global_factor_at_bound": global_at_bound' in src
    assert '"weeks_global_at_bound"' in src


def test_the_vectorised_r_lookup_is_used_for_the_censored_basis(cfg):
    """One Python `lookup_r` call per row per schedule window was the cost;
    the vectorised chain gives the same r per row."""
    from fit.fit_dispersion import lookup_r, lookup_r_vec
    r = {"subcategory": {"HOT": 3.0}, "category": {"C": 1.5}, "global": 5.0,
         "fallback_order": ["subcategory", "category", "global"]}
    calib = _anchor_frame({"HOT": ("C", 3, 6), "COLD": ("C", 3, 1),
                           "ODD": ("Z", 3, 1)})
    got = lookup_r_vec(r, calib.subcategory, calib.category)
    assert list(got) == [lookup_r(r, s, c)
                         for s, c in zip(calib.subcategory, calib.category)]
    src = inspect.getsource(tb.attach_fit_basis)
    assert "lookup_r_vec(" in src and "for s, c in zip(" not in src
    assert "lookup_r" not in inspect.getsource(tb._solve_level_factors)


def test_thin_cells_are_shrunk_toward_the_parent_not_held_at_one(cfg):
    """The payload used to SAY thin cells were 'left at 1.0'. They are not:
    every cell above the window floor follows its parent by k_shrink."""
    # THIN wants twice the factor FAT does, on a fortieth of the evidence
    calib = _anchor_frame({"THIN": ("C", 12, 6), "FAT": ("C", 300, 3)})
    factors, detail, f_global, _, _ = tb._solve_level_factors(
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
    got = m.level_factors(rows)

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
    src = inspect.getsource(tb.BaselineModel.level_factors)
    assert "enumerate(zip(" not in src
    # the pre-rename name survives only as an alias for the harness applier


def test_a_failing_convergence_check_stops_the_loop_with_its_own_message(
        monkeypatch, cfg):
    """When 5b itself crashed, the loop read no verdict, the stall test had
    nothing to fire on and the run went to --max-turns doing nothing."""
    from ops import bootstrap_loop as br

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


def test_a_pinned_category_marks_every_subcategory_it_is_the_parent_of(cfg):
    """`cat_factors, _ = fit_level(...)` threw the category detail away, so a
    category pinned at the bracket was the parent every thin subcategory
    shrank toward, unflagged (rule 3). The category detail is returned and
    each child carries `parent_at_bound`."""
    f_lo, f_hi = cfg["baseline_model"]["calibration_factor_search_bounds"]
    # category C: HOT sells 60/unit-mu (pins), THIN sells 1 on little
    # evidence; category Z is interior. Global pools C's heat and pins too.
    calib = _anchor_frame({"HOT": ("C", 300, 60), "THIN": ("C", 12, 1),
                           "CALM": ("Z", 300, 1)})
    factors, detail, f_global, g_bound, cat_detail = tb._solve_level_factors(
        calib, _Model(cfg, 1.0), k_shrink=100.0, min_anchor=10,
        tier_step=cfg["pricing"]["tier_step"],
        max_k=cfg["pricing"]["negbin_max_k"], r_lookup=R_LOOKUP)
    assert cat_detail["C"]["at_bound"] == "upper"
    assert cat_detail["C"]["raw_factor"] == pytest.approx(f_hi)
    assert "at_bound" not in cat_detail["Z"]
    # THIN's own solve is interior, but its parent is a bound
    assert "at_bound" not in detail["THIN"]
    assert detail["THIN"]["parent_at_bound"] == "upper"
    assert detail["THIN"]["shrinkage_weight_on_self"] < 0.2
    # CALM's parent (Z) converged: no flag, even though the GLOBAL pinned
    assert g_bound == "upper" and "parent_at_bound" not in detail["CALM"]
    # the category's parent is the global: its flag is the global's
    assert cat_detail["Z"]["parent_at_bound"] == "upper"
    assert tb.pinned_cells(detail) == {"HOT": "upper"}
    assert tb.pinned_cells(cat_detail) == {"C": "upper"}


class _CountingModel(_Model):
    """`_Model` that counts raw predictions and carries a version."""
    version = "counting-model"

    def __init__(self, cfg, mu):
        super().__init__(cfg, mu)
        self.calls = 0

    def predict_mu_ref(self, d, raw=False):
        self.calls += 1
        return super().predict_mu_ref(d, raw)


def _prepared(cells, days):
    """A prepared-frame lookalike: one 4-hour anchor episode per cell per
    day over `days`, every row eligible. `cells`: {sub: (cat, sold)}."""
    rows = []
    for day in days:
        for sub, (cat, sold) in cells.items():
            for h in range(10, 14):
                rows.append(dict(
                    episode_id=f"{sub}|{day}|{h}", date=day, hour_of_day=h,
                    sku_id=sub, fc="F", category=cat, subcategory=sub,
                    total_discount=0.25, d_ref=0.25, starting_inventory=100,
                    units_sold=sold, ending_inventory=100 - sold,
                    episode_eligible=True, dp_eligible=True))
    return pd.DataFrame(rows)


@pytest.fixture
def scratch_cfg(cfg, tmp_path):
    """Every artifact path under tmp_path (no r_lookup: raw basis), a thin
    anchor floor, and W=1 so the anchor window is the week before the gate."""
    cfg = copy.deepcopy(cfg)
    for key, name in (("model_path", "m.txt"), ("feature_schema_path", "s.json"),
                      ("calibration_factor_path", "cal.json")):
        cfg["baseline_model"][key] = str(tmp_path / name)
    cfg["data"]["split_manifest_path"] = str(tmp_path / "split.json")
    cfg["dispersion"]["r_lookup_path"] = str(tmp_path / "r.json")
    cfg["dispersion"]["rho_path"] = str(tmp_path / "rho.json")
    cfg["posterior"]["prior"]["path"] = str(tmp_path / "prior.json")
    cfg["baseline_model"]["calibration_min_anchor_rows"] = 10
    cfg["baseline_model"]["calibration_fit_trailing_weeks"] = 1
    return cfg


def test_the_weekly_production_refit_carries_the_convergence_verdict(
        scratch_cfg, monkeypatch):
    """`--fit-calibration` wrote no `convergence` block, so after the weekly
    production re-fit ops.tune read "never checked" and BLOCKED the daily
    lane forever. Once `launch_date` is set the re-fit moves no prior/r/rho,
    so the previous verdict is carried forward, marked; before launch (the
    bootstrap, which runs --check-convergence after each fit) nothing is."""
    cfg = scratch_cfg
    gate = pd.Timestamp(cfg["data"]["split"]["test_start"])
    days = [str(x.date()) for x in
            pd.date_range(gate - pd.Timedelta(days=21), gate + pd.Timedelta(days=6))]
    d = _prepared({"A": ("C", 2), "B": ("C", 1)}, days)
    model = _CountingModel(cfg, 1.0)
    monkeypatch.setattr(tb, "BaselineModel", lambda c: model)
    path = cfg["baseline_model"]["calibration_factor_path"]

    # bootstrap path: fit, check, fit again -> no carry (5b re-checks)
    tb.fit_level_calibration(d, cfg)
    assert "convergence" not in json.load(open(path))
    block = tb.check_calibration_convergence(d, cfg)
    assert block["converged"] and "carried_from" not in block
    tb.fit_level_calibration(d, cfg)
    assert "convergence" not in json.load(open(path))

    # production path: the same three steps keep the verdict
    cfg["data"]["launch_date"] = str(gate.date())
    tb.fit_level_calibration(d, cfg)
    first = json.load(open(path))
    assert "convergence" not in first            # nothing to carry yet
    tb.check_calibration_convergence(d, cfg)
    checked = json.load(open(path))["convergence"]
    tb.fit_level_calibration(d, cfg)             # the weekly cron
    art = json.load(open(path))
    conv = art["convergence"]
    assert conv["converged"] and conv["history"] == checked["history"]
    assert conv["checked_against"] == checked["checked_against"]
    assert conv["carried_from"] == first["provenance"]["created_at"]
    assert "carried forward" in conv["carried_note"]
    # ...and the origin survives a second carry
    tb.fit_level_calibration(d, cfg)
    assert json.load(open(path))["convergence"]["carried_from"] == \
        first["provenance"]["created_at"]
    # the launched schedule reaches the week being priced
    assert max(art["schedule"]["by_week"]) > str(gate.date())


def test_the_artifact_lists_every_pin_and_predicts_the_scope_once(
        scratch_cfg, monkeypatch):
    """Two things the schedule used to lose: the per-week at_bound flags
    (`by_week[w] = f[0]` kept factors only) and one prediction per window
    (re-predicting the same rows W times). `pinned_cells` is the one field
    status reads; the raw mu is attached to the scope once."""
    cfg = scratch_cfg
    gate = pd.Timestamp(cfg["data"]["split"]["test_start"])
    days = [str(x.date()) for x in
            pd.date_range(gate - pd.Timedelta(days=21), gate - pd.Timedelta(days=1))]
    # HOT pins at the upper bracket in every window; category C pins with it
    # (a bracket exists on the censored basis only, so r_lookup is present)
    d = _prepared({"HOT": ("C", 60), "COLD": ("C", 1), "CALM": ("Z", 1)}, days)
    json.dump(R_LOOKUP, open(cfg["dispersion"]["r_lookup_path"], "w"))
    model = _CountingModel(cfg, 1.0)
    monkeypatch.setattr(tb, "BaselineModel", lambda c: model)
    tb.fit_level_calibration(d, cfg)
    art = json.load(open(cfg["baseline_model"]["calibration_factor_path"]))

    assert art["detail"]["HOT"]["at_bound"] == "upper"
    assert art["detail_category"]["C"]["at_bound"] == "upper"
    assert art["detail"]["COLD"]["parent_at_bound"] == "upper"
    weeks = sorted(art["schedule"]["by_week"])
    assert len(weeks) >= 2
    pinned_weeks = art["schedule"]["pinned_by_week"]
    assert set(pinned_weeks) == set(weeks)
    assert pinned_weeks[weeks[0]]["subcategory:HOT"] == "upper"
    assert pinned_weeks[weeks[0]]["category:C"] == "upper"
    pins = art["pinned_cells"]
    anchor = {(p["level"], p["cell"]) for p in pins if p["scope"] == "anchor"}
    assert anchor == {("subcategory", "HOT"), ("category", "C")}
    assert {p["scope"] for p in pins} == {"anchor", *weeks}
    assert all(p["at_bound"] == "upper" for p in pins)
    # the console summary names the parent pin
    text = tb._describe_calibration(art, art["factors"])
    assert "CATEGORY solve(s) pinned" in text and "schedule week(s)" in text
    # one raw prediction for the anchor fit, one for the whole scope --
    # not one per schedule week
    assert model.calls == 2
    assert art["fit_window"] == "trailing_before_gate"
    lo, hi = art["schedule"]["anchor_fit_window"]
    assert hi == str((gate - pd.Timedelta(days=1)).date())
    assert art["fit_window_dates"] == [lo, hi]
