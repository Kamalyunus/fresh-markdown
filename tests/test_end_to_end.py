"""End-to-end smoke over the synthetic generator: bootstrap chain, decision
loop, learning update. Exercises the bootstrap pipeline order on a small
randomized-policy dataset."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import ROOT


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("mvp")
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    # shrink the expensive knobs for test speed; semantics unchanged
    cfg["baseline_model"]["num_boost_round"] = 60
    cfg["posterior"]["prior"]["search_grid_size"] = 41
    # the fixture's dataset is a fraction of production size; scale the
    # anchor-row guard with it rather than disabling it
    cfg["baseline_model"]["calibration_min_anchor_rows"] = 20
    (ws / "config.yaml").write_text(yaml.safe_dump(cfg))

    env = {**os.environ, "PYTHONPATH": ROOT}

    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=ws, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    # no --days: the generator derives the span from config so it covers
    # every split INCLUDING the hold-out -- a pinned day count silently
    # stops short of the hold-out whenever the split moves
    run(os.path.join(ROOT, "tools", "make_dummy_flc.py"),
        "--skus", "120", "--policy", "randomized",
        "--seed", "3", "--out", "data/flc.parquet")
    run("-m", "fit.prepare_data", "--input", "data/flc.parquet",
        "--out", "data/prepared.parquet")
    run("-m", "fit.train_baseline", "--input", "data/prepared.parquet")
    run("-m", "fit.fit_dispersion", "--input", "data/prepared.parquet")
    run("-m", "fit.estimate_prior", "--input", "data/prepared.parquet")
    run("-m", "evaluate.backtest", "--input", "data/prepared.parquet",
        "--out", "reports/backtest.json", "--policy-episodes", "150")
    # every test here chdirs into the workspace; RESTORE on teardown, or the
    # whole rest of the session runs against this fixture's artifacts and
    # config instead of the repo's -- silently, and only for whoever happens
    # to run after this module
    yield ws
    os.chdir(_ORIGINAL_CWD)


_ORIGINAL_CWD = os.getcwd()


def _chdir(ws):
    os.chdir(ws)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def test_prepared_data_is_priceable_and_self_consistent(workspace):
    """Postconditions of the filter chain."""
    _chdir(workspace)
    from common import episodes as episodes_mod
    full = pd.read_parquet("data/prepared.parquet")
    cfg = yaml.safe_load(open("config.yaml"))

    # integrity: true of everything
    assert full.total_discount.between(0, 1).all()
    assert (full.starting_inventory >= 0).all() and (full.units_sold >= 0).all()
    assert full.category.notna().all() and full.subcategory.notna().all()
    assert (full.original_price > 0).all()
    assert full.dp_eligible.dtype == bool
    # and the flag agrees with the reason column, in both directions
    assert (full.dp_ineligible_reason.isna() == full.dp_eligible).all()
    assert not full[~full.dp_eligible].dp_ineligible_reason.isna().any()

    # priceability: true of the subset the DP acts on -- a non-empty subset
    d = full[full.dp_eligible]
    assert 0 < len(d) <= len(full)

    # every surviving episode has at least one feasible discount tier, and a
    # cost we actually know. A zero cost is a MISSING cost -- it reads as
    # maximally priceable (d_max = 1.0), and it contributes no scrap to IL,
    # so it deflates every figure measured over it.
    assert (d.cost > 0).all()
    assert (d.cost < d.original_price).all() and (d.d_max > 0).all()
    assert (d.d_max < 1.0).all()
    # A dp_eligible episode MAY contain an hour the LEGACY policy priced under
    # cost -- that is history, and the agent is constrained never to repeat
    # it. What must hold is that such an hour is FLAGGED, so the refusal it
    # causes in shadow is expected rather than a surprise.
    under = d.offered_price < d.cost - 1e-9
    assert (~under | d.below_cost_hours).all(), \
        "a below-cost hour survived without below_cost_hours being set"
    assert d.category.notna().all() and d.subcategory.notna().all()
    assert (d.hours_remaining >= 0).all()
    assert (d.hours_remaining <= cfg["data"]["max_window_hours"]).all()

    # No episode may sell more than it was SUPPLIED -- opening stock plus
    # anything that arrived. Against opening stock alone this is simply false
    # for a restocked window, and it was: 13 episodes tripped it the moment
    # restocks became dp_eligible, every one of them correctly.
    #
    # It follows from the episode identity (`supply == sold + scrap`, scrap
    # non-negative) rather than from any filter, which is why it is asserted
    # on the output rather than argued in a comment.
    flow = episodes_mod.episode_flow(d)
    assert (flow.sold <= flow.supply).all(), (
        f"{int((flow.sold > flow.supply).sum())} episodes sell more than they "
        "were supplied -- the supply accounting has broken")
    assert (flow.clearance <= 1.0).all()

    # the exclusion window is removed whole-episode, so no survivor may have
    # ANY hour inside it
    excl = cfg["data"]["exclusion_window"]
    ds = d.date.astype(str)
    assert not (ds.ge(excl["start"]) & ds.le(excl["end"])).any()


def test_every_episode_has_a_monotone_window_counter(workspace):
    """The episode rule's own postcondition: inside an episode, flc_window
    (hours_remaining) steps down exactly one per row, and the window is at
    least as long as the rows we hold. A violation means two runs collided
    into one id -- which is what duplicate (sku, fc, date, hour) rows did."""
    _chdir(workspace)
    d = pd.read_parquet("data/prepared.parquet")
    g = d.sort_values(["date", "hour_of_day"]).groupby("episode_id")
    for eid, s in g.hours_remaining:
        steps = set(np.diff(s.to_numpy()))
        assert steps <= {-1.0}, f"{eid} has window-counter steps {steps}"
    ge = (d[d.dp_eligible].sort_values(["date", "hour_of_day"])
          .groupby("episode_id"))
    first, n = ge.hours_remaining.first(), ge.size()
    assert (first >= n - 1).all()
    assert not d.duplicated(
        subset=["sku_id", "fc", "date", "hour_of_day"]).any()


def test_prior_artifact_within_bounds(workspace):
    _chdir(workspace)
    with open("artifacts/prior.json") as f:
        prior = json.load(f)
    assert prior["source"] in ("bracket", "fallback", "mixed",
                              "profile_density")
    lo, hi = prior["search_bounds"]
    for cat, v in prior["per_category"].items():
        assert lo <= v["epsilon_naive"] <= hi
        assert lo <= v["epsilon_controlled"] <= hi
        assert v["std"] > 0 and v["mean"] < 0

        # WHAT THE DATA SAID BEFORE ANYTHING OVERWROTE IT. Both methods must
        # carry it, or the cost of the overwrite is invisible in the artifact.
        # The two methods overwrite differently, so the field differs: the
        # bracket REPLACES a rejected category with a constant, while
        # profile_density SHRINKS toward a pooled density and never replaces.
        if prior["source"] == "profile_density":
            assert v["own_mean"] < 0 and v["own_std"] > 0
            assert 0.0 <= v["own_information_weight"] <= 1.0
            # a category standing on its own data carries its own density's
            # MEAN unchanged -- but only if its likelihood is right-signed. A
            # wrong-sign category has its own density discarded whatever its
            # information weight says, because the weight measures how SHARP
            # the curve is and the sign says it points the wrong way.
            if v["own_information_weight"] >= 0.999 and not v.get("wrong_sign"):
                assert abs(v["mean"] - v["own_mean"]) < 1e-9, \
                    "a right-signed category standing on its own data must " \
                    "carry its own density's mean unchanged"
            if v.get("wrong_sign"):
                assert abs(v["mean"] - prior["pooled"]["pooled_mean"]) < 1e-3, \
                    "a wrong-sign category must take the POOLED density"
                assert max(v["unconstrained_argmax"].values()) >= -0.05, \
                    "wrong_sign must be decided on the UNCONSTRAINED peak"
            # a prior of zero width is a frozen posterior -- bounded_step can
            # never move it, whatever evidence arrives
            assert v["std"] > 0, f"{cat} has a zero-width prior"
            assert v["std_basis"] in ("density", "grid_resolution",
                                      "fold_spread")
            # the whole point of the method: no constant anywhere in it
            assert "using" not in v and "rejected_for" not in v
        else:
            assert v["bracket_mean"] < 0 and v["bracket_std"] > 0
            assert v["using"] in ("bracket", "fallback")
            if v["using"] == "bracket":
                assert v["mean"] == v["bracket_mean"], \
                    "a category using its bracket must carry the bracket's numbers"


def test_the_density_prior_has_no_constant_in_it(workspace):
    """The objection that motivated the method: -1.00 +- 0.60 is a config
    constant nothing measured produced, and it was overwriting measured
    brackets. Under `profile_density` it must not appear at all."""
    _chdir(workspace)
    with open("artifacts/prior.json") as f:
        prior = json.load(f)
    if prior["source"] != "profile_density":
        pytest.skip("bracket method configured")

    from common.config import load_config

    cfg = load_config()
    pc = cfg["posterior"]["prior"]
    lo, hi = prior["search_bounds"]
    u = prior["uniform_limit"]
    assert abs(u["mean"] - (lo + hi) / 2) < 1e-6
    assert abs(u["std"] - (hi - lo) / np.sqrt(12)) < 1e-3

    # the config carries NO prior constant at all any more -- the strongest
    # form of "no fallback": there is nothing to land on
    for gone in ("fallback_mean", "fallback_std", "std_floor", "reference_r"):
        assert gone not in pc, f"{gone} is back in config -- the constant " \
            "this method removed has been reintroduced"
    for cat, v in prior["per_category"].items():
        # a category with NO price variation has a flat likelihood, so its OWN
        # density must be the uniform, to the grid's resolution
        if v["log_ratio_sd"] < 1e-9:
            assert abs(v["own_mean"] - u["mean"]) < 0.05, (
                f"{cat} has no price variation, so its own density must be "
                f"the uniform ({u['mean']}), got {v['own_mean']}")
            assert v["own_information_weight"] == 0.0

    # and the artifact must carry the evidence for its own method
    c = prior["holdout_comparison"]
    assert c["window"] != "train"


def test_backtest_blocks_reported_separately(workspace):
    _chdir(workspace)
    with open("reports/backtest.json") as f:
        bt = json.load(f)
    assert "fidelity" in bt and "policy_deltas" in bt
    assert bt["fidelity"]["fidelity_episode_sold_ratio"] > 0
    pol = bt["policy_deltas"]
    # absolute IL reported alongside every IL% figure (design 2.3)
    assert "actual_il" in pol and "actual_il_pct" in pol
    assert "dp_il" in pol and "dp_il_pct" in pol
    # policy comparison must be apples-to-apples: both arms under the model
    assert "legacy_model_il" in pol
    gap = pol["policy_gap_like_for_like"]
    assert abs(gap["dp_minus_legacy_il"]
               - (pol["dp_il"] - pol["legacy_model_il"])) < 1.0
    # the DP optimises expected IL under this same model; like-for-like it
    # should not lose materially (small slack: tier grid, entry band, and
    # the deterministic transition differ slightly from the DP's objective)
    if pol["legacy_model_il"] > 0:
        assert gap["dp_minus_legacy_il"] <= 0.05 * pol["legacy_model_il"]
    tau = bt["tau_initial_derivation"]
    assert tau is None or tau["tau_initial"] >= 0


def test_decision_loop_and_exactly_once_update(workspace):
    _chdir(workspace)
    from common.config import load_config
    from fit.train_baseline import BaselineModel
    from evaluate.backtest import _attach_predictions
    from engine.posterior import PosteriorStore
    from events.store import EventStore, DECISION_REQUIRED
    from engine.decide import decide
    from daily.update import run as update_run

    cfg = load_config("config.yaml")
    with open(cfg["posterior"]["prior"]["path"]) as f:
        prior = json.load(f)
    with open(cfg["dispersion"]["r_lookup_path"]) as f:
        r_lookup = json.load(f)
    model = BaselineModel(cfg)
    d = _attach_predictions(pd.read_parquet("data/prepared.parquet"),
                            cfg, model, prior, r_lookup)

    store = PosteriorStore.initialise(cfg, prior["per_category"],
                                      prior["episodes_per_week"])
    events = EventStore(cfg)
    rng = np.random.default_rng(5)
    tau = 5000.0

    n = 0
    replayed = []
    for eid, g in list(d.groupby("episode_id"))[:80]:
        g = g.sort_values("hour_of_day")
        q = int(g.starting_inventory.iloc[0])
        if q <= 0:
            continue
        anchor = None
        mu_path = list(g.mu_ref_hat.to_numpy())
        for t in range(len(g)):
            if q <= 0:
                break
            row = g.iloc[t]
            evt = decide({
                "episode_id": eid, "sku_id": int(row.sku_id), "fc": row.fc,
                "category": row.category, "subcategory": row.subcategory,
                "date": str(row.date),
                "hour_of_day": int(row.hour_of_day),
                "hours_remaining": len(g) - t, "q": q,
                "original_price": float(row.original_price),
                "cost": float(row.cost), "r": float(row.r),
                "mu_ref_path": mu_path[t:], "current_discount": anchor,
            }, store, events, cfg, rng, tau, model.version)
            n += 1

            assert all(f in evt for f in DECISION_REQUIRED)
            # the contract must be sufficient, not merely complete: a decision
            # has to be recomputable from its own event, or daily.assurance
            # cannot tell a drifted artifact from a correct one
            replayed.append(evt)
            # safety invariants: cost floor and monotonicity on every path
            assert evt["applied_price"] >= evt["cost"] - 1e-6
            if anchor is not None:
                assert evt["applied_discount"] >= anchor - 1e-9
            anchor = evt["applied_discount"]

            mu = mu_path[t] * ((1 - anchor)
                               / (1 - evt["reference_discount"])) ** -1.3
            sold = min(int(rng.poisson(mu)), q)
            assert events.emit_outcome({
                "outcome_id": f"o-{evt['decision_id']}",
                "decision_id": evt["decision_id"], "units_sold": sold,
                "starting_inventory": q, "ending_inventory": q - sold,
                "applied_price": evt["applied_price"],
                "is_stockout": sold >= q, "execution_status": "ok",
                "finalized_at": pd.Timestamp.now("UTC").isoformat(),
            })
            q -= sold
    assert n > 50

    # Every decision the real path just emitted must re-solve to itself. This
    # is the production check running against production events, and it is the
    # test that would fail if the event contract ever stopped carrying enough
    # to recompute a price.
    from daily import assurance
    repro = assurance.reproduction(replayed, cfg)
    assert repro["verdict"] == "PASS", repro
    # `reproduction` re-solves the most recent `reproduction_sample` only, so
    # the count to expect is the sample cap once the loop above outgrows it --
    # which it did the moment the fixture gained shrink and this slice of 80
    # episodes started yielding 511 decisions against a cap of 500. What the
    # assertion is FOR is that nothing was silently dropped between emitting
    # and checking; the cap is a deliberate bound, not a drop.
    assert repro["decisions_checked"] == min(
        len(replayed), cfg["assurance"]["reproduction_sample"])
    assert repro["decisions_skipped_no_inputs"] == 0

    # `today` is pinned inside the fixture's span: the calibration-currency
    # gate compares the week being priced against the schedule's last fitted
    # week, and a test that read the wall clock would start failing the week
    # after it was written
    day = cfg["data"]["split"]["test_end"]
    report = update_run(cfg, apply=True, today=day)
    assert all(g["pass"] for g in report["event_quality_gates"].values())
    assert report["applied"]
    for cell, c in report["cells"].items():
        assert abs(c["proposed_mean"] - c["mean_before"]) \
            <= cfg["learning"]["max_mean_step"] + 1e-9
        assert c["proposed_std"] >= cfg["posterior"]["min_std"]

    # a SUB-THRESHOLD batch must be banked, not burned. Under the old
    # semantics the outcomes were marked processed even when no revision
    # committed, so the evidence was destroyed and the posterior later stepped
    # on an information count it no longer had the observations to justify.
    cells = report["cells"]
    if not any(c["update_triggered"] for c in cells.values()):
        before = {k: (v["forced_outcomes"], v["effective_information"])
                  for k, v in cells.items()}
        again = update_run(cfg, apply=True, today=day)["cells"]
        assert {k: (v["forced_outcomes"], v["effective_information"])
                for k, v in again.items()} == before
        for c in again.values():
            assert c["batch_oldest_outcome_age_days"] is not None

    # exactly-once, on a batch that DOES commit: lower the bar so the same
    # evidence triggers, then confirm a second apply consumes nothing
    cfg["learning"]["information_increment"] = 1e-9
    triggered = update_run(cfg, apply=True, today=day)
    assert triggered["applied"]
    assert all(c["update_triggered"] for c in triggered["cells"].values())
    assert not update_run(cfg, apply=True, today=day)["cells"]

    # the design 12 guardrails must be computed from these events, not
    # merely declared in config: with both thresholds null they report BLOCKED,
    # and once set they evaluate a real deterioration series
    from daily.monitor import (guardrail_series, stop_conditions,
                                  business_metrics, learning_metrics,
                                  safety_metrics)
    decisions, outcomes = events.load_decisions(), events.load_outcomes()
    guard = guardrail_series(decisions, outcomes, cfg)
    assert guard["days_observed"] >= 1
    assert set(guard["scrap_deterioration"]) >= {"basis", "by_day", "latest"}

    args = (safety_metrics(events, decisions, outcomes),
            learning_metrics(decisions, store, cfg, outcomes),
            business_metrics(decisions, outcomes), guard)
    # null thresholds report BLOCKED -- set HERE, not read from the shipped
    # config, so the assertion states its own precondition
    null_cfg = copy.deepcopy(cfg)
    null_cfg["monitoring"]["stop_conditions"].update(
        scrap_deterioration_pct=None, margin_deterioration_pct=None)
    blocked = stop_conditions(*args, null_cfg)
    assert "BLOCKED" in blocked["fired"]["scrap_deterioration_pct"]
    assert "BLOCKED" in blocked["fired"]["margin_deterioration_pct"]

    cfg["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = 0.20
    cfg["monitoring"]["stop_conditions"]["margin_deterioration_pct"] = 0.15
    live = stop_conditions(*args, cfg)
    # the price-mismatch stop compares COUNTS, never the 4dp rate written
    # for reading: at the boundary the rounded rate read "not fired" while
    # update's unrounded gate refused the same log
    safety = dict(args[0], price_mismatch_count=10_001,
                  compared_pair_count=1_000_000)          # 0.010001 > 0.01
    assert stop_conditions(safety, *args[1:], cfg)["fired"]["price_mismatch"]
    for key in ("scrap_deterioration_pct", "margin_deterioration_pct"):
        assert live["fired"][key] in (True, False)     # evaluated, not skipped
        g = live["guardrails"][key]
        # the result shape is the same whether or not comparable days exist,
        # so a caller never has to branch on it
        assert g["threshold"] is not None
        assert "persistence_days" in g and "consecutive_days_over" in g


def test_fit_calibration_cli(workspace):
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": ROOT}
    r = subprocess.run(
        [sys.executable, "-m", "fit.train_baseline",
         "--input", "data/prepared.parquet", "--fit-calibration"],
        cwd=workspace, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open("artifacts/calibration.json") as f:
        calib = json.load(f)
    factors = calib["factors"]
    assert factors and all(v > 0 for v in factors.values())
    assert calib["grain"] in ("subcategory", "category")
    # every cell must sit between its own raw fit and its parent: shrinkage
    # pulls toward the parent, it never extrapolates past either
    for key, info in calib["detail"].items():
        lo, hi = sorted([info["raw_factor"], info["parent_factor"]])
        assert lo - 1e-6 <= factors[key] <= hi + 1e-6, key
    # a thin cell must sit nearer its parent than a data-rich one does
    weights = [i["shrinkage_weight_on_self"] for i in calib["detail"].values()]
    assert all(0.0 <= w <= 1.0 for w in weights)
    # the trained model must still be loadable with factors applied
    from common.config import load_config
    from fit.train_baseline import BaselineModel
    cfg = load_config("config.yaml")
    d = pd.read_parquet("data/prepared.parquet").head(50)
    mu = BaselineModel(cfg).predict_mu_ref(d)
    assert (mu >= cfg["pricing"]["demand_floor"]).all()


def test_duplicate_and_malformed_events_quarantined(workspace):
    _chdir(workspace)
    from common.config import load_config
    from events.store import EventStore

    cfg = load_config("config.yaml")
    events = EventStore(cfg)
    good = {
        "outcome_id": "dup-1", "decision_id": "nonexistent", "units_sold": 1,
        "starting_inventory": 2, "ending_inventory": 1, "applied_price": 900.0,
        "is_stockout": False, "execution_status": "ok",
        "finalized_at": "2026-08-08T00:00:00+00:00",
    }
    assert events.emit_outcome(dict(good))
    assert not events.emit_outcome(dict(good))          # duplicate detected
    assert events.duplicate_counts["outcome"] == 1

    bad = dict(good, outcome_id="bad-1", units_sold=-2)
    before = len(events.load_quarantine())
    assert not events.emit_outcome(bad)                 # quarantined, not dropped
    assert len(events.load_quarantine()) == before + 1


@pytest.fixture(scope="module")
def shadow_reports(workspace):
    """evaluate.shadow, twice, for every shadow test in this module: the
    --all SAMPLED run (the sampling and in-sample caveats need more episodes
    than the hold-out holds, and its tau is the config paste) and the
    hold-out run over EVERY episode with the tau0 floor within reach (its tau
    is derived). Each test reads the report it needs; nothing re-runs."""
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": ROOT}

    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=workspace, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    # Paste tau the way an operator has to: from the backtest's own
    # derivation. A hand-typed number is refused -- see
    # engine.explore.tau_provenance_error.
    with open("config.yaml") as f:
        cfg_raw = yaml.safe_load(f)
    with open("reports/backtest.json") as f:
        derived = json.load(f)["tau_initial_derivation"]
    cfg_raw["exploration"]["tau_initial"] = derived["tau_initial"]
    with open("config.yaml", "w") as f:
        f.write(yaml.safe_dump(cfg_raw))
    # the hold-out run must DERIVE its tau: the fixture's pre-window week is
    # thin, so the floor is lowered to within reach
    cfg_raw["exploration"]["tau0_derivation_min_decisions"] = 1
    with open("config_tau0.yaml", "w") as f:
        f.write(yaml.safe_dump(cfg_raw))

    run("-m", "ops.init_posterior", "--force")
    run("-m", "evaluate.shadow", "--input", "data/prepared.parquet",
        "--out", "reports/shadow.json", "--all", "--max-episodes", "60")
    run("-m", "evaluate.shadow", "--input", "data/prepared.parquet",
        "--config", "config_tau0.yaml", "--out", "reports/shadow_holdout.json",
        "--max-episodes", "0")
    with open("reports/shadow.json") as f:
        sampled_all = json.load(f)
    with open("reports/shadow_holdout.json") as f:
        holdout = json.load(f)
    return {"derived": derived, "all": sampled_all, "holdout": holdout}


def test_shadow_phase_harness(workspace, shadow_reports):
    _chdir(workspace)
    derived, report = shadow_reports["derived"], shadow_reports["all"]
    gate = report["shadow_gate"]
    # decisions logged, no prices applied: completeness 1:1, zero cost-floor
    assert gate["cost_floor_violations"]["value"] == 0
    assert gate["event_completeness"]["value"] == 1.0
    assert report["decision_count"] > 50

    # a sampled gate must say so, and must not let a zero violation COUNT
    # read as a proof over the whole window
    w = report["window"]
    assert w["sampled"] and w["episodes"] == 60
    assert w["population_episodes"] > w["episodes"]
    assert "sampling_caveat" in gate
    assert gate["verdict"].startswith("PASS")   # caveat is not a gate row

    # and a run that is NOT the hold-out has to say which numbers it flatters
    assert w["basis"] == "full extract" and not w["out_of_sample"]
    assert "in_sample_caveat" in gate
    assert "cost-floor" in gate["in_sample_caveat"]   # names what still holds

    # every window row is graded on the frozen anchor ON PURPOSE, and the
    # coverage block says so instead of reading STALE by construction
    cov = report["artifact_versions"]["calibration_coverage"]
    assert cov["verdict"].startswith("OK"), cov["verdict"]

    # the budget check answers "is this tau affordable on the ANCHORED path",
    # which the backtest's derivation cannot: it solves on the exploit-only
    # replay path, so the same tau buys a different amount of exploration
    b = report["exploration_budget_would_be"]
    spend = report["exploration_would_be"]["would_be_cost_total"] / b["days"]
    assert b["implied_daily_spend"] == pytest.approx(spend, rel=1e-3)
    # the budget is on the TRAILING realised-IL basis -- the budget production
    # would apply -- so it need not equal share x window-mean IL. It must be
    # positive, carry its basis, and sit within the range trailing means can
    # reach (bounded by the window's own worst and best trailing days).
    assert b["daily_budget"] > 0
    assert "trailing" in b["budget_basis"]
    whole_window = b["budget_share_of_il"] * b["markdown_il_total"] / b["days"]
    assert b["daily_budget"] < whole_window * 2.5, \
        "trailing-basis budget wildly above the window mean -- check the basis"
    assert b["markdown_il_total"] == pytest.approx(
        b["markdown_il_discount"] + b["markdown_il_scrap"])
    # `spend_over_budget` is computed from UNROUNDED quantities and reported
    # to 2dp; the recomputation here divides two fields each already rounded
    # to 1dp. The tolerance is therefore the propagated error, derived rather
    # than picked -- a constant was used before and passed by luck until the
    # population changed underneath it.
    tol = 0.005 + 0.05 / b["daily_budget"] * (1 + b["spend_over_budget"])
    assert b["spend_over_budget"] == pytest.approx(
        b["implied_daily_spend"] / b["daily_budget"], abs=tol)

    # SCRAP MUST BE IN THE PROJECTION. An inline copy of classify_last was
    # tried here and dropped every scrap won on a feed with no write-off
    # sentinel -- the fallback that function carries precisely to stop that.
    # It understated the budget 10x and turned "within budget" into "WOULD
    # SUSPEND", which is the wrong direction to be wrong in: it would have
    # sent someone to re-derive tau against a budget that was never real.
    assert b["markdown_il_scrap"] > 0, \
        "scrap vanished from the budget projection -- classify_last's " \
        "no-sentinel fallback is being bypassed again"
    assert b["markdown_il_scrap"] > b["markdown_il_discount"]

    # tau re-derived on THIS path, by the same bisection the replay runs.
    # Reported, never applied: tau_initial is a MEASURED value and goes
    # through the paste gate like rho and mean_forced_hours_per_episode.
    assert b["tau_recommended"] > 0
    assert b["tau_recommended_implied_spend"] <= b["daily_budget"]
    assert b["spread_decisions"] > 0
    assert b["spread_decisions_per_episode"] > 1.0, \
        "spreads collected once per episode -- the entry-only scoping is back"
    assert b["tau"] == pytest.approx(derived["tau_initial"])   # not rewritten
    assert "tau" not in report["exploration_would_be"], "tau is reported once"

    # the trace the single multiple cannot give: does the pilot survive day 1
    tr = b["tau_controller_trace"]
    # three day counts, none of them interchangeable: the calendar span the
    # budget divides by, the days that produced a decision, the days walked
    assert tr["window_days"] == b["days"]
    assert tr["days_with_decisions"] <= tr["window_days"]
    assert tr["days_simulated"] + tr["days_truncated"] == tr["days_with_decisions"]
    max_days = yaml.safe_load(open("config.yaml"))["tuning"]["controller_trace_max_days"]
    assert tr["days_truncated"] == max(tr["days_with_decisions"] - max_days, 0)
    assert tr["tau_start"] == pytest.approx(b["tau"])
    assert tr["days_stop_condition_fires"] <= tr["days_simulated"]
    assert len(tr["by_day"]) == tr["days_simulated"]
    assert [r["day"] for r in tr["by_day"]] == sorted(r["day"] for r in tr["by_day"])
    assert tr["by_day"][0]["tau"] == pytest.approx(tr["tau_start"])
    row = next(r for r in tr["by_day"] if r["over_budget"] is not None)
    assert row["over_budget"] == pytest.approx(   # the field is rounded to 2dp
        row["spend"] / row["budget"], abs=0.01)
    # the aggregate budget is the mean of the trace's own per-day budgets
    # over the window's decision days (no day truncated here)
    if not tr["days_truncated"]:
        assert b["daily_budget"] == pytest.approx(
            np.mean([r["budget"] for r in tr["by_day"]]), abs=0.1)

    # shadow outcomes are NOT learning evidence: update must consume nothing
    from common.config import load_config
    from daily.update import run as update_run
    cfg = load_config("config.yaml")
    report2 = update_run(cfg, apply=False,
                         events_root=cfg["events"]["shadow_store_dir"])
    assert not report2["cells"]


def test_shadow_defaults_to_the_holdout_window(workspace, shadow_reports):
    """No flag, no window arguments -- it must land on the hold-out."""
    _chdir(workspace)
    report = shadow_reports["holdout"]
    w, cfg = report["window"], yaml.safe_load(open("config.yaml"))
    assert w["basis"] == "holdout" and w["out_of_sample"]
    assert w["date_min"] >= cfg["data"]["holdout"]["start"]
    assert "in_sample_caveat" not in report["shadow_gate"]
    # and it really is a different, later population than the --all run
    assert w["date_min"] > shadow_reports["all"]["window"]["date_min"]


def test_shadow_derives_tau0_when_the_week_is_thick_enough(workspace, shadow_reports):
    """With the floor within reach, the tau in force must be the DERIVED one
    -- the config paste is only the fallback -- and the derivation block must
    reconcile: implied spend at the derived tau sits at or under its target."""
    report = shadow_reports["holdout"]
    td = report["tau_initial_derivation"]
    assert not td["fallback"]
    assert td["tau_initial"] > 0 and td["decisions"] >= 1
    assert td["day_one_budget"] > 0
    assert td["implied_daily_spend"] <= td["budget_target"]
    b = report["exploration_budget_would_be"]
    assert b["tau"] == pytest.approx(td["tau_initial"])
    assert b["tau_source"].startswith("derived")


def test_parallel_and_serial_produce_the_same_reports(workspace):
    """The only claim parallelism is allowed to make: it is faster."""
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": ROOT}

    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=workspace, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

    def compare(a, b, drop):
        with open(a) as f:
            x = json.load(f)
        with open(b) as f:
            y = json.load(f)
        for k in drop:
            x.pop(k, None), y.pop(k, None)
        assert json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)

    run("-m", "evaluate.backtest", "--input", "data/prepared.parquet", "--out",
        "reports/bt_s.json", "--policy-episodes", "80")
    run("-m", "evaluate.backtest", "--input", "data/prepared.parquet", "--out",
        "reports/bt_p.json", "--policy-episodes", "80", "--workers", "3")
    compare("reports/bt_s.json", "reports/bt_p.json", ())

    for out, extra in (("reports/sh_s.json", []),
                       ("reports/sh_p.json", ["--workers", "3"])):
        run("-m", "evaluate.shadow", "--input", "data/prepared.parquet",
            "--out", out, "--all", "--max-episodes", "80", *extra)
    # solver latency is wall-clock, not a result
    compare("reports/sh_s.json", "reports/sh_p.json", ("solver_latency_p95_s",))


def test_derive_thresholds_cli(workspace):
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": ROOT}
    r = subprocess.run(
        [sys.executable, "-m", "evaluate.derive_thresholds",
         "--input", "data/prepared.parquet",
         "--out", "reports/thresholds.json"],
        cwd=workspace, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open("reports/thresholds.json") as f:
        report = json.load(f)
    assert "ab_duration" not in report          # the A/B module is gone
    assert "scrap_rate" in report["guardrail_noise"]
    assert "margin_rate" in report["guardrail_noise"]
    # the sign-off block: the trailing floor the monitor compares against
    rec = report["guardrail_threshold_recommendation"]
    for metric in ("scrap_rate", "margin_rate"):
        assert "trailing_floor" in rec[metric]
        assert rec[metric]["verdict"]


def test_zero_cost_episodes_are_flagged_whole_not_dropped(workspace, tmp_path):
    """A zero cost is a MISSING cost -- nobody gives perishable stock away."""
    _chdir(workspace)
    from fit.prepare_data import load_and_filter

    cfg = yaml.safe_load(open("config.yaml"))
    raw = pd.read_parquet("data/flc.parquet")

    def episodes_at(wf, label):
        return next(t[2] for t in wf if t[0] == label)   # (label, rows, eps, cogs, ...)

    _, clean_wf = load_and_filter("data/flc.parquet", cfg)

    # zero the cost on ONE hour of one window -- the whole episode must go
    holed = raw.copy()
    victim = holed.index[len(holed) // 2]
    holed.loc[victim, "cogs_wo_vat"] = 0.0
    path = tmp_path / "holed.parquet"
    holed.to_parquet(path)
    d, wf = load_and_filter(str(path), cfg)

    # STILL PRESENT -- that is the change. It is in the population every
    # frozen artifact trains on, and out of the one the DP acts on.
    assert not (d.cost > 0).all(), "the zero-cost episode was dropped again"
    assert episodes_at(wf, "negative_quantities_dropped") == \
        episodes_at(clean_wf, "negative_quantities_dropped"), \
        "a missing cost is economic, not an integrity defect -- it must not " \
        "be dropped by negative_quantities"

    # flagged WHOLE: every hour of that window, not just the zeroed row
    holed_eps = d.loc[d.cost <= 0, "episode_id"].unique()
    assert len(holed_eps) == 1
    ep = d[d.episode_id == holed_eps[0]]
    assert len(ep) > 1 and not ep.dp_eligible.any()
    assert (ep.dp_ineligible_reason == "cost_missing").all()

    # and the DP-eligible subset is exactly one episode smaller than a clean run
    clean, _ = load_and_filter("data/flc.parquet", cfg)
    assert int(d[d.dp_eligible].episode_id.nunique()) == \
        int(clean[clean.dp_eligible].episode_id.nunique()) - 1


def test_a_restock_survives_the_real_chain_and_stays_dp_eligible(
        workspace, tmp_path):
    """The flag through the whole pipeline, not just the detector."""
    _chdir(workspace)
    from fit.prepare_data import load_and_filter

    cfg = yaml.safe_load(open("config.yaml"))
    raw = pd.read_parquet("data/flc.parquet")
    clean, _ = load_and_filter("data/flc.parquet", cfg)

    # pick a window that survives a CLEAN run, so the only thing this test can
    # be measuring is the restock
    sizes = clean.groupby("episode_id").size()
    eid = sizes[sizes >= 4].index[0]
    g = (clean[clean.episode_id == eid]
         .sort_values(["date", "hour_of_day"]))
    tail = set(zip(g.date.astype(str), g.hour_of_day))  # from its 3rd hour on
    tail -= set(zip(g.date.astype(str)[:2], g.hour_of_day[:2]))

    # 50 units arrive during the episode's 3rd hour. The source reports the
    # FINAL count, so that hour's `ending_inventory` carries the arrival and
    # every LATER hour opens 50 higher -- the chain stays continuous. Only
    # the arriving hour has ending != starting - sold, and that IS the
    # restock. A write-off zero must stay zero or chain_break takes the
    # episode before the restock flag ever sees it.
    holed = raw.copy()
    key = pd.Series(list(zip(holed.date.astype(str), holed.hour)),
                    index=holed.index)
    same_sku = holed.skuseq.eq(g.sku_id.iloc[0]) & holed.fc.eq(g.fc.iloc[0])
    arrives = same_sku & key.isin([sorted(tail)[0]])
    later = same_sku & key.isin(sorted(tail)[1:])
    assert arrives.sum() == 1 and later.sum() >= 1
    holed.loc[arrives | later, "ending_inventory"] += \
        50 * holed.loc[arrives | later, "ending_inventory"].ne(0)
    holed.loc[later, "inventory"] += 50
    path = tmp_path / "restocked.parquet"
    holed.to_parquet(path)

    d, wf = load_and_filter(str(path), cfg)
    hit = d[(d.sku_id == g.sku_id.iloc[0]) & (d.fc == g.fc.iloc[0])
            & d.date.astype(str).isin(g.date.astype(str))]
    assert len(hit), "the restocked episode was dropped from the population"
    # STAYS dp_eligible. The replay re-solves each hour against the stock on
    # hand and applies the episode's own per-hour adjustment, so the DP finds
    # out about the arrival at the next hour -- which is what happens live.
    assert hit.dp_eligible.all()
    assert hit.units_restocked.gt(0).all()

    # and it cost the frozen artifacts nothing: same population, one fewer
    # DP-eligible episode
    assert d.episode_id.nunique() == clean.episode_id.nunique()
    detail = next(t[4] for t in wf if t[0] == "dp_eligible")
    assert detail["restocked"]["episodes"] >= 1
    assert detail["restocked"]["cogs_at_risk"] > 0
    assert detail["restocked"]["still_dp_eligible"] >= 1


def test_negative_entry_window_is_recovered_not_dropped(workspace, tmp_path):
    """A window counter that enters ALREADY negative is a known source
    pattern, not a defect, and dropping it is not neutral."""
    _chdir(workspace)
    from fit.prepare_data import load_and_filter

    cfg = yaml.safe_load(open("config.yaml"))
    cap = cfg["data"]["manufacturing_window_hours"]
    raw = pd.read_parquet("data/flc.parquet")

    # make one whole episode enter negative, keeping it short enough to qualify
    keys = ["skuseq", "fc", "date"]
    victim = raw[keys].iloc[0]
    mask = (raw.skuseq == victim.skuseq) & (raw.fc == victim.fc) \
        & (raw.date == victim.date)
    assert 0 < mask.sum() <= cap, mask.sum()
    holed = raw.copy()
    holed.loc[mask, "flc_window"] = -242        # the observed constant, roughly
    path = tmp_path / "neg.parquet"
    holed.to_parquet(path)

    d, wf = load_and_filter(str(path), cfg)
    rows = {t[0]: t for t in wf}
    rec = rows["negative_window_recovered"][4]   # (label, rows, eps, cogs, detail)
    assert rec["episodes_recovered"] >= 1
    assert rec["window_hours_assumed"] == cap

    # recovered rows survive, and carry a real countdown rather than a clamp:
    # episode identification, the DP horizon and extend_to_window all
    # difference this column, so a flat value would re-segment every hour.
    # Asserted on dp_eligible: an UNrecovered negative is kept in the
    # population now, flagged rather than dropped.
    assert (d.loc[d.dp_eligible, "hours_remaining"] >= 0).all()
    # (the reason column is first-match, so a negative window that also has a
    #  missing cost reads `cost_missing` -- what must hold is the gating)
    assert not d.loc[d.hours_remaining < 0, "dp_eligible"].any()
    surv = d[(d.sku_id == victim.skuseq) & (d.fc == victim.fc)]
    if len(surv):
        for _, g in surv.groupby("episode_id"):
            steps = set(g.sort_values(["date", "hour_of_day"])
                        .hours_remaining.diff().dropna())
            assert steps <= {-1.0}, steps


def test_an_unreconciled_hour_becomes_shrink_not_a_drop(workspace, tmp_path):
    """Stock that vanishes is COUNTED, not deleted with its episode."""
    _chdir(workspace)
    from fit.prepare_data import load_and_filter
    from common import episodes as E

    cfg = yaml.safe_load(open("config.yaml"))
    raw = pd.read_parquet("data/flc.parquet")
    clean, clean_wf = load_and_filter("data/flc.parquet", cfg)
    clean_eps = {t[0]: t[2] for t in clean_wf}

    # one unit vanishes mid-episode
    holed = raw.copy()
    row = holed.index[len(holed) // 2]
    start, sold = holed.at[row, "inventory"], holed.at[row, "units_sold"]
    holed.at[row, "ending_inventory"] = max(start - sold - 1, 0)
    # keep the chain continuous, or the CONTINUITY rule takes it first
    nxt = holed.index[holed.index.get_loc(row) + 1]
    same = (holed.at[nxt, "skuseq"] == holed.at[row, "skuseq"]
            and holed.at[nxt, "fc"] == holed.at[row, "fc"])
    if same:
        holed.at[nxt, "inventory"] = holed.at[row, "ending_inventory"]
    path = tmp_path / "shrunk.parquet"
    holed.to_parquet(path)

    d, wf = load_and_filter(str(path), cfg)
    got = {t[0]: t[2] for t in wf}

    # the episode SURVIVES the universe -- shrink is not a drop
    assert got["episode_universe"] == clean_eps["episode_universe"], \
        "a shortfall dropped its episode again; it is shrink, and shrink is scrap"

    # ...and the units are accounted for rather than lost
    flow = E.episode_flow(d)
    assert (flow.opening + flow.arrived == flow.sold + flow.scrap).all()
    assert len(E.flow_identity_violations(d)) == 0

def test_the_episode_identity_holds_on_every_episode(workspace):
    """opening + restocked == sold + shrink + leftover_at_last_hour."""
    _chdir(workspace)
    from common import episodes as E
    d = pd.read_parquet("data/prepared.parquet")

    bad = E.flow_identity_violations(d)
    assert len(bad) == 0, bad.head(10).to_string()

    flow = E.episode_flow(d)
    assert (flow.opening + flow.arrived == flow.sold + flow.scrap).all()
    # scrap is the last hour's leftover PLUS the shrink -- both are units paid
    # for that returned no revenue
    assert (flow.scrap == flow.leftover + flow.vanished).all()

    # the two consequences the owner named, on the whole population
    assert (flow.clearance <= 1.0).all(), "clearance above 1 is never valid"
    assert (flow[flow.clearance >= 1.0].scrap == 0).all(), \
        "an episode that sold everything it had cannot also carry scrap"

    # censoring is decided at the LAST ROW, and can only happen there: the
    # source stops emitting rows once inventory reaches zero
    off = E.censoring_off_last_row(d)
    assert off["rows_shelf_emptied_mid_episode"] == 0
    assert off["rows_with_zero_starting_inventory"] == 0

    # three nested populations
    assert (set(d[d.dp_eligible].episode_id)
            <= set(d[d.episode_eligible].episode_id)
            <= set(d.episode_id))

    # and the frame carries the columns the identity is built from
    for col in ("units_restocked", "units_shrink", "episode_supply",
                "episode_scrap", "episode_clearance", "episode_eligible"):
        assert col in d.columns, col
    per_ep = d.groupby("episode_id").first()
    assert (per_ep.episode_supply
            == flow.opening.reindex(per_ep.index)
            + flow.arrived.reindex(per_ep.index)).all()


def test_the_manifest_reports_the_identity(workspace):
    """It is checked on every run, not only in the test suite."""
    _chdir(workspace)
    # asserted on episode_flow itself now that the published waterfall (which
    # used to carry this block) is gone. The identity is the property; the
    # report was only where it happened to be printed.
    import pandas as pd

    from common import episodes as E
    d = pd.read_parquet("data/prepared.parquet")
    flow = E.episode_flow(d)
    elig = flow[flow.eligible]
    assert len(elig) > 0
    residual = (elig.supply - elig.sold - elig.leftover - elig.vanished).abs()
    assert float(residual.max()) < 1e-6, "the flow identity does not close"


def test_re_segmentation_is_a_no_op_and_says_so_if_it_stops_being_one(workspace):
    """`contiguous_episodes_built` guards an invisible invariant."""
    _chdir(workspace)
    from fit.prepare_data import load_and_filter, assign_episode_ids
    from common.config import load_config as _lc

    _, wf = load_and_filter("data/flc.parquet", _lc())
    steps = [t[0] for t in wf]
    i = steps.index("contiguous_episodes_built")
    assert wf[i][1] == wf[i - 1][1], "re-segmentation must not drop rows"
    # episodes MAY rise here (one source window splits into two); the point is
    # that no row leaves the frame
    assert wf[i][2] >= wf[i - 1][2]

    # and the ids on the output are their own fixed point
    d = pd.read_parquet("data/prepared.parquet")
    d = d.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert (assign_episode_ids(d) == d.episode_id).all()


def test_a_row_scoped_drop_is_caught_rather_than_silently_re_segmenting(
        workspace, tmp_path, monkeypatch):
    """The guard fires. Simulated by punching a hole the way a row-scoped
    filter would -- which is exactly the failure mode it exists for."""
    _chdir(workspace)
    from fit import prepare_data as pdm

    real = pdm.assign_episode_ids
    calls = {"n": 0}

    def holed(df):
        calls["n"] += 1
        out = real(df)
        # from the second call on, pretend a row went missing mid-window so
        # the ids come back different -- and different from the PREVIOUS
        # call's, so the re-segmentation check disagrees with the ids in
        # force however many assignments the chain makes before it
        return (out.str.replace("T1", f"T{8 + calls['n']}", regex=False)
                if calls["n"] > 1 else out)

    monkeypatch.setattr(pdm, "assign_episode_ids", holed)
    cfg = yaml.safe_load(open("config.yaml"))
    with pytest.raises(AssertionError, match="re-segmentation moved"):
        pdm.load_and_filter("data/flc.parquet", cfg)


def test_a_missing_hour_drops_the_whole_window_not_just_a_fragment(
        workspace, tmp_path):
    """A fragment is not an episode, and neither fragment is usable."""
    _chdir(workspace)
    from fit.prepare_data import load_and_filter

    cfg = yaml.safe_load(open("config.yaml"))
    raw = pd.read_parquet("data/flc.parquet")
    clean, clean_wf = load_and_filter("data/flc.parquet", cfg)
    base = next((t[4] for t in clean_wf
                 if t[0] == "gap_split_windows_dropped" and len(t) > 4), {})

    # punch one hour out of the middle of a surviving multi-hour window
    sizes = clean.groupby("episode_id").size()
    eid = sizes[sizes >= 6].index[0]
    g = clean[clean.episode_id == eid].sort_values(["date", "hour_of_day"])
    mid = g.iloc[len(g) // 2]
    holed = raw[~((raw.skuseq == g.sku_id.iloc[0]) & (raw.fc == g.fc.iloc[0])
                  & (raw.date == mid.date) & (raw.hour == mid.hour_of_day))]
    path = tmp_path / "gap.parquet"
    holed.to_parquet(path)

    d, wf = load_and_filter(str(path), cfg)
    got = next(t[4] for t in wf if t[0] == "gap_split_windows_dropped")

    # exactly one more window broken, and BOTH its fragments are gone
    assert got["windows_split_by_a_feed_gap"] == \
        base.get("windows_split_by_a_feed_gap", 0) + 1
    assert got["fragments_dropped"] == base.get("fragments_dropped", 0) + 2
    assert got["missing_hours"] == base.get("missing_hours", 0) + 1

    surviving = d[(d.sku_id == g.sku_id.iloc[0]) & (d.fc == g.fc.iloc[0])
                  & d.date.astype(str).isin(g.date.astype(str))]
    assert len(surviving) == 0, \
        "a fragment of the split window survived -- on its own it looks like " \
        "a whole episode, which is the failure this filter exists to prevent"


def test_no_pre_launch_artifact_reads_past_test_end(workspace):
    """Rule 16: the hold-out is read once, by evaluate.shadow. derive_thresholds
    PASTES guardrail floors into config via tune, and
    fit_dispersion's drift_by_window sets the retrain cadence -- both were
    measuring on the full extract, i.e. tuning config on the window that
    exists to grade it."""
    import pandas as pd

    from fit.prepare_data import pre_launch

    import yaml

    ws = workspace
    cfg = yaml.safe_load((ws / "config.yaml").read_text())
    d = pd.read_parquet(ws / "data" / "prepared.parquet")
    test_end = cfg["data"]["split"]["test_end"]
    assert (d.date.astype(str) > test_end).any(), "fixture has no hold-out rows"
    assert (pre_launch(d, cfg).date.astype(str) <= test_end).all()

    rho = json.loads((ws / "artifacts" / "rho.json").read_text())
    windows = (rho.get("drift_by_window") or {}).get("by_window") or []
    seen = [w.get("window") or w.get("start") for w in windows
            if isinstance(w, dict)]
    assert all(str(s) <= test_end for s in seen if s), seen


def test_the_manifest_persists_the_waterfall_it_computed(workspace):
    """Every stage's rows, episodes, COGS at risk and detail dict were
    computed and then printed as three columns -- so flow_identity.holds
    going False, the restock/edge diagnostics and the shrink-vs-skew reading
    all landed on the floor while the run succeeded. AGENTS and design 5.2
    describe an artifact that did not exist."""
    ws = workspace
    raw = (ws / "artifacts" / "split_manifest.json").read_text()

    def bare(token):                       # a NaN/Infinity TOKEN, not the word in prose
        raise AssertionError(f"bare {token} is not JSON and most parsers refuse it")

    manifest = json.loads(raw, parse_constant=bare)

    stages = manifest["waterfall"]
    assert len(stages) > 1
    assert [s["stage"] for s in stages] == [s["stage"] for s in stages if s["stage"]]
    assert all(isinstance(s["rows"], int) and isinstance(s["episodes"], int)
               for s in stages)
    # the COGS-at-risk column the docs tell the owner to read
    assert any(s["cogs_at_risk"] for s in stages)
    # and at least one stage carries its detail dict, the identity among them
    details = [s["detail"] for s in stages if s["detail"]]
    assert details
    assert any("flow_identity" in d for d in details), list(details[0])


def test_a_set_launch_date_schedules_factors_past_the_gate(workspace, tmp_path):
    """Before launch the factor schedule stops at split.test_end (rule 16).
    After launch the same command IS the weekly cron: it must reach the
    week being priced, or calibration_current refuses every --apply. Moving
    split.test_end instead rescopes every sealed fit."""
    from fit.train_baseline import fit_level_calibration
    from common import episodes
    from common.config import load_config

    ws = workspace
    _chdir(ws)
    cfg = load_config(str(ws / "config.yaml"))
    cfg["baseline_model"] = dict(cfg["baseline_model"],
                                 calibration_factor_path=str(tmp_path / "cal.json"))
    d = pd.read_parquet(ws / "data" / "prepared.parquet")
    test_end_week = episodes.week_key(pd.Series([cfg["data"]["split"]["test_end"]]))[0]
    last_data_week = episodes.week_key(d.date).max()
    assert last_data_week > test_end_week, "fixture has no post-gate weeks"

    fit_level_calibration(d, cfg)
    pre = json.loads((tmp_path / "cal.json").read_text())["schedule"]
    assert max(pre["by_week"]) <= test_end_week
    assert pre["scope"].startswith("pre-launch")

    cfg["data"]["launch_date"] = str(d.date.max())
    fit_level_calibration(d, cfg)
    live = json.loads((tmp_path / "cal.json").read_text())["schedule"]
    priced_week = (episodes.week_start(last_data_week)
                   + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    assert max(live["by_week"]) == priced_week, live["weeks_unfitted_held_at_1"]
    assert live["scope"].startswith("production")


def test_advance_reads_the_workspace_and_names_the_phase(workspace):
    """The driver's probe runs against a real bootstrapped workspace and
    --plan touches nothing: same files before and after."""
    ws = workspace
    before = {p: os.path.getmtime(ws / p) for p in
              ("config.yaml", "artifacts/bundle.json", "reports/backtest.json")
              if (ws / p).exists()}
    r = subprocess.run([sys.executable, "-m", "ops.advance", "--plan"],
                       cwd=ws, env={**os.environ, "PYTHONPATH": ROOT},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("phase   ")
    assert "[" in r.stdout.splitlines()[0]            # a phase is marked
    assert {p: os.path.getmtime(ws / p) for p in before} == before


def test_the_pilot_simulator_walks_past_launch_date(workspace, tmp_path):
    """Two simulated days after launch through the real engine and the real
    daily lane, in a workspace of its own: decisions priced and ingested,
    the tau walk committed, Lane C re-fit and re-sealed under sim/, the
    expectations graded -- and not one production artifact touched."""
    from evaluate import pilot_sim
    from common.provenance import file_digest

    ws = workspace
    _chdir(ws)
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    # the RUNTIME_REQUIRED values a launch needs; the sim sets launch_date
    cfg["dispersion"]["rho"] = json.load(open(cfg["dispersion"]["rho_path"]))["rho"]
    cfg["exploration"]["tau_initial"] = 500.0
    cfg["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = 0.5
    cfg["monitoring"]["stop_conditions"]["margin_deterioration_pct"] = 0.1
    (ws / "sim_config.yaml").write_text(yaml.safe_dump(cfg))
    if not os.path.exists(cfg["baseline_model"]["calibration_factor_path"]):
        subprocess.run([sys.executable, "-m", "fit.train_baseline", "--input",
                        "data/prepared.parquet", "--fit-calibration",
                        "--config", "sim_config.yaml"], check=True,
                       cwd=ws, env={**os.environ, "PYTHONPATH": ROOT})
    if os.path.exists(cfg["posterior"]["path"]):
        os.remove(cfg["posterior"]["path"])          # the sim initialises one
    frozen = {p: file_digest(p) for p in (
        cfg["baseline_model"]["model_path"], cfg["baseline_model"]["calibration_factor_path"],
        cfg["dispersion"]["r_lookup_path"], cfg["posterior"]["prior"]["path"])}

    sim_dir, out = str(tmp_path / "sim"), str(tmp_path / "pilot_sim.json")
    rc = pilot_sim.main(["--config", "sim_config.yaml", "--input", "data/prepared.parquet",
                         "--raw", "data/flc.parquet", "--days", "2",
                         "--episodes-per-day", "4", "--sim-dir", sim_dir, "--out", out,
                         "--fault", "push_fail:0.2"])
    assert rc == 0
    rep = json.load(open(out))

    assert rep["engine"]["decisions"] > 0 and rep["engine"]["rejected_total"] == 0
    assert rep["engine"]["violations"] == {"price_rose_within_episode": 0, "below_cost": 0}
    day = rep["days"][0]
    assert day["ingest"]["outcomes_built"] > 0 and day["ingest"]["emitted"] > 0
    assert day["ingest"]["push_failures_applied"] >= 0
    # the walk commits once an episode has closed; a day with none says so
    assert day["tau"]["committed"] or day["tau"]["skipped"]
    assert rep["days"][-1]["tau"]["committed"], rep["days"][-1]["tau"]
    assert day["lane_c"] and day["lane_c"]["scope"].startswith("production")
    assert day["assurance"]["reproduction"] in ("PASS", "INSUFFICIENT")
    assert "apply" in day and day["apply"]["calibration_schedule_current"]
    assert {x["name"] for x in rep["expectations"]} == {n for n, _ in pilot_sim.EXPECTATIONS}
    assert {"pilot", "legacy"} <= set(rep["economics"])
    # the world's truth and the store agree on what was priced
    assert rep["engine"]["pilot_hours"] >= rep["engine"]["decisions"]

    # every write went under sim_dir; production is byte-identical
    assert {p: file_digest(p) for p in frozen} == frozen
    assert not os.path.exists(cfg["posterior"]["path"])
    assert os.path.exists(os.path.join(sim_dir, "events_store", "decisions.jsonl"))
    assert os.path.exists(os.path.join(sim_dir, "events_store", "outcomes.jsonl"))
    assert os.path.exists(os.path.join(sim_dir, "posterior.json"))
    assert os.path.exists(os.path.join(sim_dir, "bundle.json"))
    assert len(os.listdir(os.path.join(sim_dir, "history"))) == 1      # one bundle
