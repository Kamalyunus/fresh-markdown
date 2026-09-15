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
    # the halvings are config too: one halving resolves the interior cell
    # to the bracket's midpoint, twenty to well inside a percent
    cfg["baseline_model"]["calibration_factor_bisection_steps"] = 1
    _, coarse, _, _, _ = tb._solve_level_factors(
        calib.copy(), _Model(cfg, 1.0), 0.0, 10, cfg["pricing"]["tier_step"],
        cfg["pricing"]["negbin_max_k"], R_LOOKUP)
    assert coarse["COLD"]["raw_factor"] in (pytest.approx(0.875), pytest.approx(1.625))
    assert abs(coarse["COLD"]["raw_factor"] - detail2["COLD"]["raw_factor"]) > 0.05


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
    # (the written artifact carries the flag by name: exercised on a fitted
    # artifact in test_the_artifact_lists_every_pin_and_predicts_the_scope_once)


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
    # the fit basis carries exactly that r per row, attached once
    attached = tb.attach_fit_basis(calib.copy(), _Model(cfg, 1.0), r)
    assert list(attached.r_val) == list(got)


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
    assert "held_at_parent" not in thin and thin["factor"] == factors["THIN"]


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


def scratch_config(cfg, tmp_path):
    """`cfg` with every artifact path under tmp_path (no r_lookup: raw
    basis), a thin anchor floor, and W=1 so the anchor window is the week
    before the gate -- the builder behind `scratch_cfg` (test_calibration_
    schedule builds its artifacts on it too)."""
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


@pytest.fixture
def scratch_cfg(cfg, tmp_path):
    return scratch_config(cfg, tmp_path)


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


def test_a_cell_with_no_anchor_rows_takes_its_parent_never_one(cfg):
    """A subcategory with ZERO anchor rows in the fit window priced at raw
    mu x 1.0 while its category solved to something else -- a cell with
    one anchor row and no sales was shrunk to its parent, one with none
    was silently 1.0. Every cell of the window's population gets a
    factor: no anchor rows means the parent's, said so."""
    calib = _anchor_frame({"PORK": ("MEAT", 300, 2), "LEAF": ("VEG", 300, 3)})
    # BEEF is in the window's population but never at the anchor; FISH's
    # whole category is off the anchor
    off = _anchor_frame({"BEEF": ("MEAT", 20, 1), "FISH": ("SEAFOOD", 20, 1)})
    off["total_discount"] = 0.45
    calib = pd.concat([calib, off], ignore_index=True)
    factors, detail, f_global, _, cat_detail = tb._solve_level_factors(
        calib, _Model(cfg, 1.0), k_shrink=0.0, min_anchor=10,
        tier_step=cfg["pricing"]["tier_step"], max_k=cfg["pricing"]["negbin_max_k"],
        r_lookup=R_LOOKUP)
    assert factors["PORK"] > 1.5 and factors["LEAF"] > factors["PORK"]
    assert factors["BEEF"] == pytest.approx(cat_detail["MEAT"]["factor"])
    assert factors["BEEF"] == pytest.approx(factors["PORK"], abs=1e-3)   # k_shrink 0
    assert detail["BEEF"]["held_at_parent"] and detail["BEEF"]["anchor_rows"] == 0
    # a category with no anchor rows is held at the global, and its child at it
    assert cat_detail["SEAFOOD"]["held_at_parent"]
    assert cat_detail["SEAFOOD"]["factor"] == pytest.approx(round(f_global, 4))
    assert factors["FISH"] == pytest.approx(round(f_global, 4))
    assert tb.keys_held_at_parent(detail) == ["BEEF", "FISH"]
    assert tb.category_factors(cat_detail)["MEAT"] == cat_detail["MEAT"]["factor"]
    assert 1.0 not in factors.values()


def test_the_applier_waterfalls_a_cell_the_window_never_saw_to_its_category(cfg):
    """A subcategory absent from the fit window entirely (new assortment)
    prices at its CATEGORY's factor, then 1.0 -- in the anchor table and in
    every schedule week -- never at raw mu beside a category at 1.4."""
    m = tb.BaselineModel.__new__(tb.BaselineModel)
    m.cfg, m.calibration_grain = cfg, "subcategory"
    m.calibration = {"PORK": 1.5}
    m.calibration_category = {"MEAT": 1.4}
    m.calibration_schedule = {"2026-07-06": {"PORK": 1.2}}
    m.calibration_schedule_category = {"2026-07-06": {"MEAT": 1.1}}
    m._reset_calibration_counters()
    rows = pd.DataFrame({"subcategory": ["PORK", "BEEF", "FISH"] * 2,
                         "category": ["MEAT", "MEAT", "SEAFOOD"] * 2,
                         "date": ["2026-07-08"] * 3 + ["2026-06-29"] * 3})
    got = list(m.level_factors(rows))
    assert got[:3] == [1.2, 1.1, 1.0]          # the week's tables
    assert got[3:] == [1.5, 1.4, 1.0]          # the frozen anchor
    # without a parent table the waterfall ends at 1.0, as before
    m.calibration_category, m.calibration_schedule_category = None, None
    assert list(m.level_factors(rows)) == [1.2, 1.0, 1.0, 1.5, 1.0, 1.0]


def test_factors_apply_by_the_week_the_episode_opened_not_the_rows_week(cfg):
    """The fit windows cut WHOLE episodes by opening week, but the factors
    were applied by row week: the Monday rows of a Sunday-opened episode
    sat inside week w's fit window AND took week w's table -- a self-fit at
    every seam. An episode takes the table of the week it opened in; a
    frame without episode_id (the live forecast rows) reads the row date."""
    m = tb.BaselineModel.__new__(tb.BaselineModel)
    m.cfg, m.calibration_grain = cfg, "category"
    m.calibration = {"A": 9.0}
    m.calibration_schedule = {"2026-08-03": {"A": 1.1}, "2026-08-10": {"A": 1.3}}
    m._reset_calibration_counters()
    seam = pd.DataFrame({"category": ["A"] * 3,
                         "episode_id": ["x", "x", "y"],
                         "date": ["2026-08-09", "2026-08-10", "2026-08-10"],
                         "hour_of_day": [23, 0, 9]})
    assert list(m.level_factors(seam)) == [1.1, 1.1, 1.3]
    assert m._cal_rows_scheduled == 3
    m._reset_calibration_counters()
    assert list(m.level_factors(seam.drop(columns="episode_id"))) == [1.1, 1.3, 1.3]


def test_the_artifact_carries_the_parent_tables_and_the_held_cells(
        scratch_cfg, monkeypatch):
    """`factors_category` / `schedule.by_week_category` are what the
    applier waterfalls to; `keys_held_at_parent` (anchor and per week)
    names the cells priced at their parent; the held weeks are named for
    what they are held at -- the anchor -- with the old key kept for
    readers not yet moved."""
    cfg = scratch_cfg
    gate = pd.Timestamp(cfg["data"]["split"]["test_start"])
    days = [str(x.date()) for x in
            pd.date_range(gate - pd.Timedelta(days=21), gate - pd.Timedelta(days=1))]
    d = _prepared({"A": ("C", 2), "B": ("C", 1)}, days)
    # an off-anchor subcategory of C, and one of a category never at the
    # anchor, inside the anchor window (the last trailing week)
    off = _prepared({"OFF": ("C", 1), "FAR": ("Z", 1)}, days[-3:])
    off["total_discount"] = 0.45
    d = pd.concat([d, off], ignore_index=True)
    model, Applier = _CountingModel(cfg, 1.0), tb.BaselineModel
    monkeypatch.setattr(tb, "BaselineModel", lambda c: model)
    tb.fit_level_calibration(d, cfg)
    art = json.load(open(cfg["baseline_model"]["calibration_factor_path"]))
    assert art["keys_held_at_parent"] == ["FAR", "OFF"]
    assert art["factors"]["OFF"] == art["factors_category"]["C"]
    assert art["factors"]["FAR"] == art["factors_category"]["Z"] == art["global_factor"]
    sched = art["schedule"]
    assert set(sched["by_week_category"]) == set(sched["by_week"])
    for w, table in sched["by_week"].items():
        assert set(table) >= {"A", "B"}
        assert set(sched["by_week_category"][w]) >= {"C"}
    assert "weeks_unfitted_held_at_anchor" in sched
    assert "weeks_unfitted_held_at_1" not in sched      # the former name is read, never written
    assert tb.schedule_reaches({"by_week": {}, "weeks_unfitted_held_at_anchor": ["2026-09-07"]}) \
        == "2026-09-07"
    assert tb.weeks_held_at_anchor({"weeks_unfitted_held_at_1": ["2026-09-07"]}) == ["2026-09-07"]
    # the loaded applier reads the parent tables
    applier = Applier.__new__(Applier)
    applier.cfg = cfg
    applier.calibration = art["factors"]
    applier.calibration_category = art["factors_category"]
    applier.calibration_grain = art["grain"]
    applier.calibration_schedule = sched["by_week"]
    applier.calibration_schedule_category = sched["by_week_category"]
    applier._reset_calibration_counters()
    new = pd.DataFrame({"subcategory": ["NEW"], "category": ["C"],
                        "date": [days[-1]], "episode_id": ["n"]})
    wk = tb.episodes.week_key(pd.Series([days[-1]])).iloc[0]
    assert applier.level_factors(new)[0] == sched["by_week_category"][wk]["C"]
