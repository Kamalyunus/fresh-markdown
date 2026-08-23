"""End-to-end smoke over the synthetic generator: bootstrap chain, decision
loop, learning update. Exercises the PRD section 1a order on a small
randomized-policy dataset."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("mvp")
    with open(os.path.join(REPO, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    # shrink the expensive knobs for test speed; semantics unchanged
    cfg["baseline_model"]["num_boost_round"] = 60
    cfg["posterior"]["prior"]["search_grid_size"] = 41
    # the fixture's dataset is a fraction of production size; scale the
    # anchor-row guard with it rather than disabling it
    cfg["baseline_model"]["calibration_min_anchor_rows"] = 20
    (ws / "config.yaml").write_text(yaml.safe_dump(cfg))

    env = {**os.environ, "PYTHONPATH": REPO}

    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=ws, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    run(os.path.join(REPO, "tools", "make_dummy_flc.py"),
        "--skus", "120", "--days", "160", "--policy", "randomized",
        "--seed", "3", "--out", "data/flc.parquet")
    run("-m", "bootstrap.prepare_data", "--input", "data/flc.parquet",
        "--out", "data/prepared.parquet")
    run("-m", "bootstrap.measure", "--input", "data/flc.parquet",
        "--out", "reports/phase0.json")
    run("-m", "bootstrap.train_baseline", "--input", "data/prepared.parquet")
    run("-m", "bootstrap.fit_dispersion", "--input", "data/prepared.parquet")
    run("-m", "bootstrap.estimate_prior", "--input", "data/prepared.parquet")
    run("-m", "backtest", "--input", "data/prepared.parquet",
        "--out", "reports/backtest.json", "--policy-episodes", "150")
    return ws


def _chdir(ws):
    os.chdir(ws)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def test_filter_chain_waterfall(workspace):
    _chdir(workspace)
    with open("artifacts/split_manifest.json") as f:
        manifest = json.load(f)
    wf = manifest["data_quality_waterfall"]
    rows = [s["rows"] for s in wf if s["step"] != "contiguous_episodes_built"]
    assert rows == sorted(rows, reverse=True)
    # every stage is named and counted -- 8 drops, the raw row, the
    # re-segmentation row, and the dp_eligible SUMMARY row, which drops
    # nothing. The count is asserted so that adding or removing a filter has
    # to be a deliberate edit here, and so the figure quoted in the
    # walkthrough and design doc has something holding it to the code.
    assert wf[0]["step"] == "raw"
    assert wf[-1]["step"] == "dp_eligible"
    assert len(wf) == 11, [s["step"] for s in wf]
    # Everything that is not an integrity or scope rule is a FLAG. None of
    # these may come back as a drop: each one removes population from every
    # frozen artifact while the thing it protects (the DP's state space, or a
    # scrap figure) reads the flag anyway.
    for gone in ("below_cost_dropped", "non_priceable_dropped",
                 "window_too_long_dropped", "cost_missing_dropped",
                 "negative_window_dropped", "restocked_episodes_dropped",
                 "edge_truncated_episodes_dropped",
                 "units_gt_inventory_dropped"):
        assert gone not in [x["step"] for x in wf], \
            f"{gone} is back to dropping -- it must flag, or every frozen " \
            "artifact loses that population again"
    d = pd.read_parquet("data/prepared.parquet")
    assert d.category.notna().all()
    assert d.total_discount.between(0, 1).all()      # percent -> fraction once
    assert (d.original_price > 0).all()
    # `sold <= starting` holds on dp_eligible ONLY. In the integrity
    # population an hour selling more than it opened with is a RESTOCK -- the
    # source reports the final count in `ending_inventory` -- and asserting it
    # on the whole frame is how a restock got deleted as an impossible value.
    assert (d[d.dp_eligible].units_sold
            <= d[d.dp_eligible].starting_inventory).all()


def test_waterfall_reports_the_money_each_filter_removes(workspace):
    """Rows are not the unit the business cares about.

    IL is discount given away plus scrap at cost, so what a filter costs is
    measured in exposure -- unit cost x opening stock -- not in rows. The two
    diverge: a stage can take a small share of rows and a large share of the
    money, and only the second says whether the surviving population still
    represents the business.
    """
    _chdir(workspace)
    with open("artifacts/split_manifest.json") as f:
        wf = json.load(f)["data_quality_waterfall"]

    raw = wf[0]["cogs_at_risk"]
    assert raw > 0
    assert "cogs_dropped" not in wf[0]              # nothing precedes raw
    for prev, row in zip(wf, wf[1:]):
        assert row["cogs_dropped"] == pytest.approx(
            prev["cogs_at_risk"] - row["cogs_at_risk"], abs=0.2)
        assert row["cogs_dropped_pct_of_raw"] == pytest.approx(
            row["cogs_dropped"] / raw, abs=1e-5)
        assert row["cogs_pct_of_raw"] == pytest.approx(
            row["cogs_at_risk"] / raw, abs=1e-5)

    # every stage removes money or leaves it alone -- except re-segmentation,
    # which splits windows so one opening row becomes two and the same stock
    # is counted twice. That stage ADDS, in episodes and in money alike.
    for row in wf[1:]:
        if row["step"] == "contiguous_episodes_built":
            continue
        assert row["cogs_dropped"] >= -0.2, row["step"]

    # and the measure is per EPISODE, not per row: summing cost x inventory
    # over hours would multiply the same stock by the window length
    d = pd.read_parquet("data/prepared.parquet")
    per_row = float((d.cost * d.starting_inventory).sum())
    assert wf[-1]["cogs_at_risk"] < per_row


def test_edge_truncation_is_flagged_and_split_from_the_feed_residue(workspace):
    """Unclosed episodes stay, and the two reasons stay told apart.

    Two things make an episode's outcome unknown. The extract cutting a window
    mid-flight is unavoidable and a longer extract is the only fix. Everything
    else is a feed problem a longer extract will NOT fix. Neither is dropped
    -- the observed hours are good demand data either way, and every scrap
    figure already excludes an unclosed ending on its own -- so what this test
    protects is the SPLIT: `edge_truncated` must mark exactly the
    boundary-explained ones, leaving m11's not_closed residue readable.
    """
    _chdir(workspace)
    from common import episodes as E
    with open("artifacts/split_manifest.json") as f:
        wf = json.load(f)["data_quality_waterfall"]
    steps = [s["step"] for s in wf]
    stage = wf[steps.index("dp_eligible")]["edge_truncated"]
    assert 0.0 <= stage["share_of_unclosed_explained_by_edge"] <= 1.0

    d = pd.read_parquet("data/prepared.parquet")
    kind = E.classify(d)
    unknown_ids = set(kind.index[kind == E.NOT_CLOSED])
    flagged = set(d.loc[d.edge_truncated, "episode_id"].unique())

    assert stage["episodes_unclosed"] == len(unknown_ids)
    assert stage["episodes_edge_truncated"] == len(flagged)
    assert stage["episodes_unclosed_not_edge"] == len(unknown_ids - flagged)
    assert flagged <= unknown_ids, \
        "a CLOSED episode was flagged edge_truncated -- the flag has stopped " \
        "meaning 'outcome missing because the extract stopped'"
    # nothing was removed for being unclosed: the frame still carries both
    # kinds, and the flag does not gate the DP
    assert stage["still_dp_eligible"] > 0 or not flagged, \
        "edge-truncated episodes are being kept out of dp_eligible -- only " \
        "their ENDING is unknown, and replay zeroes that scrap already"

    # the flag is exactly the boundary test: an episode NOT flagged has a
    # window that ended before the extract's last hour
    last = E.last_rows(d)
    ts = pd.to_datetime(last.date) + pd.to_timedelta(last.hour_of_day, unit="h")
    ends = ts + pd.to_timedelta(last.hours_remaining.clip(lower=0), unit="h")
    residue = last.episode_id.isin(unknown_ids - flagged).to_numpy()
    if residue.any():
        assert (ends[residue] <= ts.max()).all()
        assert (ts[residue] < ts.max()).all()


def test_m11_still_reports_where_the_unknown_scrap_sits(workspace):
    """The whole point of keeping the non-edge ones: they stay countable."""
    _chdir(workspace)
    with open("reports/phase0.json") as f:
        m11 = json.load(f)["m11_episode_endings"]
    by_month = m11["not_closed_by_month"]
    assert isinstance(by_month, dict)
    if m11["shares"]["not_closed"] > 0:
        assert by_month, "not_closed episodes exist but are not broken out"
        assert sum(v["episodes"] for v in by_month.values()) == \
            round(m11["shares"]["not_closed"] * m11["episodes"])
        assert sum(v["leftover_units"] for v in by_month.values()) == \
            m11["scrap_units_unknown_not_closed"]
        assert list(by_month) == sorted(by_month)       # chronological


def test_prepared_data_is_priceable_and_self_consistent(workspace):
    """Postconditions of the filter chain.

    Two tiers now, and the split is the point. INTEGRITY properties hold on
    every surviving row -- a discount in range, non-negative quantities, a
    real category. PRICEABILITY properties hold only on `dp_eligible`, because
    an episode the DP cannot price is deliberately still in the population:
    the demand model cannot see cost or hours_remaining, so it is an ordinary
    observation to every frozen artifact.
    """
    _chdir(workspace)
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

    # priceability: true of the subset the DP acts on
    d = full[full.dp_eligible]
    assert len(d) and len(d) < len(full) or len(d) == len(full)

    assert (d.units_sold <= d.starting_inventory).all()
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

    # No episode may sell more than it opened with. This is NOT asserted
    # directly anywhere -- it follows from the `restocked` flag gating
    # dp_eligible. Outside a restock the source's convention is
    # `ending == starting - sold` with a non-negative ending, so every hour
    # has `sold <= starting`, and the chain never increases. By induction
    # `start_t <= start_1 - sum(sold before t)`, so the episode total cannot
    # exceed the opening stock. It holds on THIS subset only: a restocked
    # episode is kept in the population and sells more than it opened with as
    # a matter of course -- that IS the restock signal.
    #
    # An earlier hand-written cleaner checked the episode total directly. It
    # is redundant here ONLY while the restock flag holds, and the
    # subsumption breaks silently if the flag is computed before
    # re-segmentation, so the invariant is asserted on the output rather than
    # argued in a comment.
    g = d.sort_values(["episode_id", "date", "hour_of_day"]).groupby("episode_id")
    over = g.units_sold.sum() > g.starting_inventory.first()
    assert not over.any(), (
        f"{int(over.sum())} episodes sell more than they opened with -- the "
        "per-hour and restock stages no longer cover the episode total")

    # the exclusion window is removed whole-episode, so no survivor may have
    # ANY hour inside it
    excl = cfg["data"]["exclusion_window"]
    ds = d.date.astype(str)
    assert not (ds.ge(excl["start"]) & ds.le(excl["end"])).any()


def test_every_episode_has_a_monotone_window_counter(workspace):
    """The episode rule's own postcondition: inside an episode, flc_window
    (hours_remaining) steps down exactly one per row, and the window is at
    least as long as the rows we hold. A violation means two runs collided
    into one id -- which is what duplicate (sku, fc, date, hour) rows did.

    The STEP is asserted on everything, because episode identification rests
    on it. The window LENGTH is asserted on `dp_eligible` only: an episode
    entering with an already-negative counter that recovery could not repair
    is kept and flagged `negative_window`, and its counter is exactly the
    thing that cannot be trusted."""
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


def test_phase0_report_complete(workspace):
    _chdir(workspace)
    with open("reports/phase0.json") as f:
        res = json.load(f)
    for key in ["m1_cost_ratio", "m2_same_hour_variation",
                "m3_intra_episode_correlation", "m4_demand_density",
                "m5_censoring", "m6_il_pct", "m7_learning_rate",
                "m8_entry_hour", "reassessment_gates"]:
        assert key in res
    assert res["m5_censoring"]["episodes_reaching_zero_inventory"] < 1.0
    m3 = res["m3_intra_episode_correlation"]
    assert m3["implied_deff"] >= 1.0


def test_prior_artifact_within_bounds(workspace):
    _chdir(workspace)
    with open("artifacts/prior.json") as f:
        prior = json.load(f)
    assert prior["source"] in ("bracket", "fallback")
    lo, hi = prior["search_bounds"]
    for cat, v in prior["per_category"].items():
        assert lo <= v["epsilon_naive"] <= hi
        assert lo <= v["epsilon_controlled"] <= hi
        assert v["std"] > 0 and v["mean"] < 0


def test_backtest_blocks_reported_separately(workspace):
    _chdir(workspace)
    with open("reports/backtest.json") as f:
        bt = json.load(f)
    assert "fidelity" in bt and "policy_deltas" in bt
    assert bt["fidelity"]["fidelity_episode_sold_ratio"] > 0
    pol = bt["policy_deltas"]
    # absolute IL reported alongside every IL% figure (PRD 3.6 / 17.4)
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
    from bootstrap.train_baseline import BaselineModel
    from backtest.replay import _attach_predictions
    from pricing.posterior import PosteriorStore
    from events.store import EventStore, DECISION_REQUIRED
    from inference.decide import decide
    from pipeline.update import run as update_run

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
                "hour_of_day": int(row.hour_of_day),
                "hours_remaining": len(g) - t, "q": q,
                "original_price": float(row.original_price),
                "cost": float(row.cost), "r": float(row.r),
                "mu_ref_path": mu_path[t:], "current_discount": anchor,
            }, store, events, cfg, rng, tau, model.version)
            n += 1

            assert all(f in evt for f in DECISION_REQUIRED)
            # the contract must be sufficient, not merely complete: a decision
            # has to be recomputable from its own event, or pipeline.assurance
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
    from pipeline import assurance
    repro = assurance.reproduction(replayed, cfg)
    assert repro["verdict"] == "PASS", repro
    assert repro["decisions_checked"] == len(replayed)
    assert repro["decisions_skipped_no_inputs"] == 0

    report = update_run(cfg, apply=True)
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
        again = update_run(cfg, apply=True)["cells"]
        assert {k: (v["forced_outcomes"], v["effective_information"])
                for k, v in again.items()} == before
        for c in again.values():
            assert c["batch_oldest_outcome_age_days"] is not None

    # exactly-once, on a batch that DOES commit: lower the bar so the same
    # evidence triggers, then confirm a second apply consumes nothing
    cfg["learning"]["information_increment"] = 1e-9
    triggered = update_run(cfg, apply=True)
    assert triggered["applied"]
    assert all(c["update_triggered"] for c in triggered["cells"].values())
    assert not update_run(cfg, apply=True)["cells"]

    # the section 15.4 guardrails must be computed from these events, not
    # merely declared in config: with both thresholds null they report BLOCKED,
    # and once set they evaluate a real deterioration series
    from pipeline.monitor import (guardrail_series, stop_conditions,
                                  business_metrics, learning_metrics,
                                  safety_metrics)
    decisions, outcomes = events.load_decisions(), events.load_outcomes()
    guard = guardrail_series(decisions, outcomes, cfg)
    assert guard["days_observed"] >= 1
    assert set(guard["scrap_deterioration"]) >= {"basis", "by_day", "latest"}

    args = (safety_metrics(events, decisions, outcomes),
            learning_metrics(decisions, store, cfg),
            business_metrics(decisions, outcomes, cfg), guard)
    blocked = stop_conditions(*args, cfg)
    assert "BLOCKED" in blocked["fired"]["scrap_deterioration_pct"]
    assert "BLOCKED" in blocked["fired"]["margin_deterioration_pct"]

    cfg["monitoring"]["stop_conditions"]["scrap_deterioration_pct"] = 0.20
    cfg["monitoring"]["stop_conditions"]["margin_deterioration_pct"] = 0.15
    live = stop_conditions(*args, cfg)
    for key in ("scrap_deterioration_pct", "margin_deterioration_pct"):
        assert live["fired"][key] in (True, False)     # evaluated, not skipped
        g = live["guardrails"][key]
        # the result shape is the same whether or not comparable days exist,
        # so a caller never has to branch on it
        assert g["threshold"] is not None
        assert "persistence_days" in g and "consecutive_days_over" in g


def test_fit_calibration_cli(workspace):
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": REPO}
    r = subprocess.run(
        [sys.executable, "-m", "bootstrap.train_baseline",
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
    from bootstrap.train_baseline import BaselineModel
    cfg = load_config("config.yaml")
    cfg["baseline_model"]["apply_level_calibration"] = True
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


def test_shadow_phase_harness(workspace):
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": REPO}

    # shadow needs the section 9.3 decision and a tau; supply them in config
    with open("config.yaml") as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw["baseline_model"]["apply_level_calibration"] = False
    # Paste tau the way an operator has to: from the backtest's own
    # derivation. A hand-typed number is now refused -- see
    # pricing.explore.tau_provenance_error.
    with open("reports/backtest.json") as f:
        derived = json.load(f)["tau_initial_derivation"]
    cfg_raw["exploration"]["tau_initial"] = derived["tau_initial"]
    with open("config.yaml", "w") as f:
        f.write(yaml.safe_dump(cfg_raw))

    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=workspace, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    run("-m", "bootstrap.init_posterior", "--force")
    # --all here on purpose: this test exercises the harness and the SAMPLING
    # caveat, which needs more episodes than the hold-out window holds. The
    # hold-out default is covered by its own run below.
    run("-m", "pipeline.shadow", "--input", "data/prepared.parquet",
        "--out", "reports/shadow.json", "--all", "--max-episodes", "60")

    with open("reports/shadow.json") as f:
        report = json.load(f)
    gate = report["shadow_gate"]
    # decisions logged, no prices applied: completeness 1:1, zero cost-floor
    assert gate["cost_floor_violations"]["value"] == 0
    assert gate["event_completeness"]["value"] == 1.0
    assert gate["matched_decision_rate"]["pass"]
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

    # the budget check answers "is this tau affordable on the ANCHORED path",
    # which the backtest's derivation cannot: it solves on the exploit-only
    # replay path, so the same tau buys a different amount of exploration
    b = report["exploration_budget_would_be"]
    spend = report["exploration_would_be"]["would_be_cost_total"] / b["days"]
    assert b["implied_daily_spend"] == pytest.approx(spend, rel=1e-3)
    assert b["daily_budget"] == pytest.approx(
        b["budget_share_of_il"] * b["markdown_il_total"] / b["days"], rel=1e-3)
    assert b["markdown_il_total"] == pytest.approx(
        b["markdown_il_discount"] + b["markdown_il_scrap"])
    assert b["spend_over_budget"] == pytest.approx(   # reported to 2dp
        b["implied_daily_spend"] / b["daily_budget"], abs=0.005)

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

    # the trace the single multiple cannot give: does the pilot survive day 1
    tr = b["tau_controller_trace"]
    # three day counts, none of them interchangeable: the calendar span the
    # budget divides by, the days that produced a decision, the days walked
    assert tr["window_days"] == b["days"]
    assert tr["days_with_decisions"] <= tr["window_days"]
    assert tr["days_simulated"] + tr["days_truncated"] == tr["days_with_decisions"]
    if tr["days_truncated"]:
        assert "TRUNCATED" in tr["note"]
    assert tr["tau_start"] == pytest.approx(b["tau"])
    assert len(tr["clip"]) == 2
    assert tr["days_stop_condition_fires"] <= tr["days_simulated"]
    assert len(tr["by_day"]) == tr["days_simulated"]
    assert [r["day"] for r in tr["by_day"]] == sorted(r["day"] for r in tr["by_day"])
    assert tr["by_day"][0]["tau"] == pytest.approx(tr["tau_start"])
    row = next(r for r in tr["by_day"] if r["over_budget"] is not None)
    assert row["over_budget"] == pytest.approx(   # the field is rounded to 2dp
        row["spend"] / row["budget"], abs=0.01)

    # shadow outcomes are NOT learning evidence: update must consume nothing
    from common.config import load_config
    from pipeline.update import run as update_run
    cfg = load_config("config.yaml")
    report2 = update_run(cfg, apply=False,
                         events_root=cfg["events"]["shadow_store_dir"])
    assert not report2["cells"]



def test_shadow_defaults_to_the_holdout_window(workspace):
    """No flag, no window arguments -- it must land on the hold-out.

    The honest run was opt-in until now, which is the wrong way round: every
    artifact is fit up to test_end, so a run including that data grades the
    pipeline on rows it already saw.
    """
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": REPO}
    r = subprocess.run(
        [sys.executable, "-m", "pipeline.shadow", "--input",
         "data/prepared.parquet", "--out", "reports/shadow_holdout.json",
         "--max-episodes", "0"],
        cwd=workspace, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    with open("reports/shadow_holdout.json") as f:
        report = json.load(f)
    w, cfg = report["window"], yaml.safe_load(open("config.yaml"))
    assert w["basis"] == "holdout" and w["out_of_sample"]
    assert w["date_min"] >= cfg["data"]["holdout"]["start"]
    assert "in_sample_caveat" not in report["shadow_gate"]
    # and it really is a different, later population than the --all run
    with open("reports/shadow.json") as f:
        assert w["date_min"] > json.load(f)["window"]["date_min"]


def test_parallel_and_serial_produce_the_same_reports(workspace):
    """The only claim parallelism is allowed to make: it is faster.

    Run both harnesses each way and compare the JSON. Anything that differs
    is a shared-state leak -- an accumulator folded in completion order, a
    draw that depended on position, an event committed by a worker.
    """
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": REPO}

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

    run("-m", "backtest", "--input", "data/prepared.parquet", "--out",
        "reports/bt_s.json", "--policy-episodes", "80")
    run("-m", "backtest", "--input", "data/prepared.parquet", "--out",
        "reports/bt_p.json", "--policy-episodes", "80", "--workers", "3")
    compare("reports/bt_s.json", "reports/bt_p.json", ())

    for out, extra in (("reports/sh_s.json", []),
                       ("reports/sh_p.json", ["--workers", "3"])):
        run("-m", "pipeline.shadow", "--input", "data/prepared.parquet",
            "--out", out, "--all", "--max-episodes", "80", *extra)
    # solver latency is wall-clock, not a result
    compare("reports/sh_s.json", "reports/sh_p.json", ("solver_latency_p95_s",))


def test_derive_thresholds_cli(workspace):
    _chdir(workspace)
    env = {**os.environ, "PYTHONPATH": REPO}
    r = subprocess.run(
        [sys.executable, "-m", "bootstrap.derive_thresholds",
         "--input", "data/prepared.parquet", "--mde", "0.075",
         "--out", "reports/thresholds.json"],
        cwd=workspace, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open("reports/thresholds.json") as f:
        report = json.load(f)
    ab = report["ab_duration"]
    assert ab["by_duration"], "no candidate duration produced an SE"
    for row in ab["by_duration"].values():
        assert row["se_pooled"] > 0
        # difference SE must exceed the pooled SE (arm split loses precision)
        assert row["se_arm_difference"] > row["se_pooled"]
    assert "scrap_rate" in report["guardrail_noise"]
    assert "margin_rate" in report["guardrail_noise"]
    # the sign-off block: both bases side by side, so a threshold can never be
    # signed off against the trailing floor alone
    rec = report["guardrail_threshold_recommendation"]
    for metric in ("scrap_rate", "margin_rate"):
        assert "trailing_floor" in rec[metric]
        assert "control_arm_floor" in rec[metric]
        assert rec[metric]["verdict"]


def test_state_rejected_not_priced(workspace):
    _chdir(workspace)
    from common.config import load_config
    from inference.decide import decide, StateRejected

    cfg = load_config("config.yaml")
    with pytest.raises(StateRejected):
        decide({
            "episode_id": "x", "sku_id": 1, "fc": "F", "category": "MEAT",
            "subcategory": "PORK", "hour_of_day": 12, "hours_remaining": 2,
            "q": 3, "original_price": -5.0, "cost": 10.0, "r": 1.0,
            "mu_ref_path": [1.0, 1.0], "current_discount": None,
        }, None, None, cfg, np.random.default_rng(0), 100.0, "v")


def _window(sku, fc, start, hours, base_hr=None):
    """One selling window as hourly rows, counting hours_remaining down."""
    hr = hours - 1 if base_hr is None else base_hr
    ts = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame({
        "sku_id": sku, "fc": fc,
        "date": ts.normalize(), "hour_of_day": ts.hour,
        "hours_remaining": [hr - i for i in range(hours)],
    })


def test_episode_spans_midnight_as_one_window():
    """FLC windows commonly run past midnight -- 36 hours is common. A
    date-keyed episode would split one economic window into three, resetting
    the monotonicity anchor and charging carried inventory to scrap twice."""
    from bootstrap.prepare_data import assign_episode_ids

    long_window = _window(1, "FC1", "2026-03-01 10:00", 36)
    d = long_window.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    ids = assign_episode_ids(d)
    assert ids.nunique() == 1, "a 36-hour window must be ONE episode"
    assert d.date.nunique() == 2, "and it must genuinely cross midnight"
    assert ids.iloc[0] == "1|FC1|2026-03-01T10"
    assert len(d) == 36 and d.hours_remaining.iloc[-1] == 0

    # a window long enough to cross twice is still one episode
    three = _window(1, "FC1", "2026-03-01 20:00", 36)
    three = three.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert assign_episode_ids(three).nunique() == 1
    assert three.date.nunique() == 3


def test_back_to_back_windows_and_gaps_still_split():
    from bootstrap.prepare_data import assign_episode_ids

    # two windows abutting with no time gap: only the counter reset separates
    # them, so time-contiguity alone would wrongly merge these
    a = _window(1, "FC1", "2026-03-01 10:00", 6)
    b = _window(1, "FC1", "2026-03-01 16:00", 6)
    d = pd.concat([a, b]).sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert assign_episode_ids(d).nunique() == 2

    # a missing hour inside a window splits it, so an episode's row count
    # always equals its clock -- validate_state rejects any mismatch
    g = _window(1, "FC1", "2026-03-01 10:00", 6).drop(index=3)
    assert assign_episode_ids(g).nunique() == 2

    # different sku x fc never merge
    two = pd.concat([_window(1, "FC1", "2026-03-01 10:00", 4),
                     _window(2, "FC1", "2026-03-01 10:00", 4)])
    two = two.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    assert assign_episode_ids(two).nunique() == 2


def test_split_assigns_straddling_episode_by_start_date():
    """A window that starts in train and ends in calib belongs wholly to
    train -- otherwise the boundary runs through the middle of an episode."""
    from bootstrap.prepare_data import split_frames

    cfg = {"data": {"split": {
        "train_start": "2026-03-01", "train_end": "2026-03-02",
        "calib_start": "2026-03-03", "calib_end": "2026-03-04",
        "test_start": "2026-03-05", "test_end": "2026-03-06"}}}
    d = _window(1, "FC1", "2026-03-02 10:00", 36)     # crosses into 03-03/04
    d["episode_id"] = "1|FC1|2026-03-02T10"
    frames = split_frames(d, cfg)
    assert len(frames["train"]) == len(d)
    assert len(frames["calib"]) == 0 and len(frames["test"]) == 0


def _observed_episode():
    """The worked example from the production extract: a MEAT episode that
    closes with stock left, while ending_inventory reports zero."""
    hours = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    inv = [8, 8, 8, 8, 8, 5, 5, 5, 5, 4]
    sold = [0, 0, 0, 0, 3, 0, 0, 0, 1, 3]
    end = [8, 8, 8, 8, 5, 5, 5, 5, 4, 0]      # zeroed at the close
    return pd.DataFrame({
        "episode_id": ["m"] * 10,
        "date": [pd.Timestamp("2026-03-01").date()] * 10,
        "hour_of_day": hours,
        "hours_remaining": list(range(9, -1, -1)),
        "starting_inventory": inv, "units_sold": sold, "ending_inventory": end,
    })


def test_true_leftover_on_the_production_worked_example():
    from common import episodes

    d = _observed_episode()
    last = episodes.last_rows(d)
    assert int(last.ending_inventory.iloc[0]) == 0     # what the source says

    # 4 units enter the final hour, 3 sell -> 1 is written off, not zero
    left = episodes.leftover_units(last.starting_inventory, last.units_sold)
    assert float(left.iloc[0]) == 1.0
    assert int(d.starting_inventory.iloc[0]) == 8 and int(d.units_sold.sum()) == 7

    # the final row carries the closure sentinel (ending_inventory zeroed with
    # a unit still on hand), so the listing ended and that unit IS scrap
    assert episodes.write_off_convention(last)
    kind = episodes.classify(d)
    assert kind.iloc[0] == episodes.COMPLETED
    assert float(episodes.scrap_units(d).iloc[0]) == 1.0

    # and the counter is at zero here only because the fixture makes it so;
    # on production it is still positive on ~99.9% of final rows, which is
    # why scrap must not be keyed to it
    assert int(episodes.last_rows(d).hours_remaining.iloc[0]) == 0


def test_zero_cost_episodes_are_flagged_whole_not_dropped(workspace, tmp_path):
    """A zero cost is a MISSING cost -- nobody gives perishable stock away.

    It is fatal to exactly two consumers and harmless to the rest, which is
    why it FLAGS rather than drops. Fatal to the action set (`d_max` reads
    1.0, i.e. maximally priceable, which raised ZeroDivisionError out of the
    demand model on the first full-population shadow run) and to IL (scrap is
    `cost x leftover`, so it contributes discount cost and no scrap). Harmless
    to every frozen artifact, because FEATURES carries no `cost` at all.

    Flagged WHOLE, at episode grain: one bad hour poisons the window, since
    the monotonicity anchor carries its price into every later hour.
    """
    _chdir(workspace)
    from bootstrap.prepare_data import load_and_filter

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


def test_restocked_episodes_are_flagged_not_dropped():
    """The detector, and the fact that its output only ever sets a flag.

    A restock breaks the DP's state transition -- one pool draining
    monotonically -- and nothing else. The hours themselves are honest demand
    observations, each censored against its own opening stock, so dropping
    them cost the demand fit 2.7% of the extract's COGS to protect a solver
    that reads `dp_eligible` anyway.
    """
    from bootstrap.prepare_data import restocked_episodes

    clean = _observed_episode()
    assert len(restocked_episodes(clean)) == 0

    # 4 units arrive during hour 16, which opened with 5 and sold none. The
    # source reports the FINAL count, so ending goes to 9 and hour 17 opens
    # with 9 -- the chain stays continuous, which is what makes this a
    # restock rather than a break.
    restocked = clean.copy()
    h16 = restocked.hour_of_day == 16
    restocked.loc[h16, "ending_inventory"] += 4
    restocked.loc[restocked.hour_of_day > 16, "starting_inventory"] += 4
    restocked.loc[restocked.hour_of_day.between(17, 19), "ending_inventory"] += 4
    assert list(restocked_episodes(restocked)) == ["m"]

    # the plainest restock of all: an hour selling MORE than it opened with.
    # `sold >= starting` is not an impossible quantity, it is stock arriving.
    oversell = clean.copy()
    h15 = oversell.hour_of_day == 15
    oversell.loc[h15, "units_sold"] = 12          # opened with 8
    oversell.loc[h15, "ending_inventory"] = 5     # so 9 arrived
    assert list(restocked_episodes(oversell)) == ["m"]

    # selling stock down is never a restock, however steep the drop
    steep = clean.copy()
    steep.loc[steep.hour_of_day == 15, "units_sold"] = 8
    steep.loc[steep.hour_of_day == 15, "ending_inventory"] = 0
    steep.loc[steep.hour_of_day > 15, "starting_inventory"] = 0
    steep.loc[steep.hour_of_day > 15, "ending_inventory"] = 0
    steep.loc[steep.hour_of_day > 15, "units_sold"] = 0
    assert len(restocked_episodes(steep)) == 0


def test_a_restock_survives_the_real_chain_and_lands_out_of_dp_eligible(
        workspace, tmp_path):
    """The flag through the whole pipeline, not just the detector.

    Two things have to hold together and only an end-to-end run shows both:
    the episode is STILL THERE for the demand fit, and it is OUT of the subset
    the DP acts on. A restock is also the one break `adjustment_reason` names
    (`intraday_restock`), so the chain-break stage must let it through -- if
    that stage took it first, the flag would be measuring an empty set.
    """
    _chdir(workspace)
    from bootstrap.prepare_data import load_and_filter

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
    assert not hit.dp_eligible.any()
    assert (hit.dp_ineligible_reason == "restocked").all()

    # and it cost the frozen artifacts nothing: same population, one fewer
    # DP-eligible episode
    assert d.episode_id.nunique() == clean.episode_id.nunique()
    assert int(d[d.dp_eligible].episode_id.nunique()) < \
        int(clean[clean.dp_eligible].episode_id.nunique())
    detail = next(t[4] for t in wf if t[0] == "dp_eligible")
    assert detail["restocked"]["episodes"] >= 1
    assert detail["restocked"]["cogs_at_risk"] > 0


def test_negative_entry_window_is_recovered_not_dropped(workspace, tmp_path):
    """A window counter that enters ALREADY negative is a known source
    pattern, not a defect, and dropping it is not neutral.

    Those episodes concentrate in a handful of categories, so dropping them
    selects on category and biases every per-category figure -- the prior, the
    per-subcategory `r`, the category IL split. They behave like a standard
    short window, so they are recovered with a synthetic countdown instead.

    The claim is CHECKED, not trusted: an episode entering negative that runs
    LONGER than the assumed window is not the pattern, is not recovered, and
    is flagged `negative_window` -- kept in the population, out of
    `dp_eligible`, since the counter it cannot supply is a DP input and not a
    demand feature.
    """
    _chdir(workspace)
    from bootstrap.prepare_data import load_and_filter

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


def test_an_unreconciled_hour_drops_the_episode(workspace, tmp_path):
    """Every hour must reconcile or name why not, using the SAME function
    production enforces -- so the analysis population cannot contain a break
    `events.store` would quarantine.

    Recognised by the ZERO, never by position: the source writes stock off at
    ITS window boundary, and once a window is merged across midnight that row
    sits in the MIDDLE of ours. A "only the last hour may break the chain"
    test drops those episodes in bulk.
    """
    _chdir(workspace)
    from bootstrap.prepare_data import load_and_filter
    from common.episodes import adjustment_reason

    cfg = yaml.safe_load(open("config.yaml"))
    raw = pd.read_parquet("data/flc.parquet")
    _, clean_wf = load_and_filter("data/flc.parquet", cfg)
    clean = {t[0]: t[2] for t in clean_wf}

    # a PARTIAL shortfall mid-episode: matches no convention, so it is
    # unexplained inventory loss and the episode must go
    holed = raw.copy()
    row = holed.index[len(holed) // 2]
    start, sold = holed.at[row, "inventory"], holed.at[row, "units_sold"]
    holed.at[row, "ending_inventory"] = max(start - sold - 1, 0)
    assert adjustment_reason(start, sold, holed.at[row, "ending_inventory"]) is None
    path = tmp_path / "broken.parquet"
    holed.to_parquet(path)

    _, wf = load_and_filter(str(path), cfg)
    got = {t[0]: t[2] for t in wf}
    assert got["chain_break_dropped"] < got["negative_window_recovered"], \
        "the unexplained shortfall did not drop its episode"
    assert got["chain_break_dropped"] == clean["chain_break_dropped"] - 1

    # A write-off zero at the SAME position is now ALSO a break, and that is
    # the corrected convention rather than a regression. `adjustment_reason`
    # still names the row -- the zero is the source's own sentinel and nothing
    # about that changed -- but the row is mid-episode, so the NEXT hour opens
    # from a positive figure this hour says was written off. The two fields
    # contradict each other about one instant, and no business event explains
    # that.
    #
    # This is what makes the write-off effectively last-hour-only without a
    # position test anywhere: a genuine close has no next hour to disagree
    # with, so continuity cannot fire on it. Position used to be rejected as
    # "our bookkeeping"; the continuity rule derives it from the source's own
    # arithmetic instead.
    ok = raw.copy()
    ok.at[row, "ending_inventory"] = 0
    assert adjustment_reason(start, sold, 0) == "episode_close_write_off"
    path2 = tmp_path / "writeoff.parquet"
    ok.to_parquet(path2)
    _, wf2 = load_and_filter(str(path2), cfg)
    got2 = {t[0]: t[2] for t in wf2}
    assert got2["chain_break_dropped"] == clean["chain_break_dropped"] - 1, \
        "a mid-episode write-off leaves the next hour opening from stock it " \
        "just wrote off -- the source contradicting itself, not a convention"

    # ...and the same zero on the episode's LAST hour is the real thing: no
    # next hour, so nothing to contradict, and it survives untouched
    last_ix = (raw[(raw.skuseq == raw.at[row, "skuseq"])
                   & (raw.fc == raw.at[row, "fc"])]
               .sort_values(["date", "hour"]).index[-1])
    closing = raw.copy()
    l_start = closing.at[last_ix, "inventory"]
    l_sold = min(closing.at[last_ix, "units_sold"], max(l_start - 1, 0))
    closing.at[last_ix, "units_sold"] = l_sold
    closing.at[last_ix, "ending_inventory"] = 0
    if l_start - l_sold > 0:
        path3 = tmp_path / "closing.parquet"
        closing.to_parquet(path3)
        _, wf3 = load_and_filter(str(path3), cfg)
        got3 = {t[0]: t[2] for t in wf3}
        assert got3["chain_break_dropped"] == clean["chain_break_dropped"], \
            "a write-off on the FINAL hour is how a listing closes and must " \
            "never be treated as a break"


def test_the_episode_identity_holds_on_every_episode(workspace):
    """opening + restocked == sold + shrink + leftover_at_last_hour.

    The whole specification for episode-level data quality, in one line. Every
    unit an episode ever had ends up sold, shrunk, or still on the shelf at
    the last hour -- there is no fourth option, so this is not a heuristic
    with a tolerance.

    Asserted on the REAL prepared frame rather than a fixture, because what it
    protects is the arithmetic: chain continuity makes the two sides provably
    equal, so a violation means the supply accounting is broken, and a silent
    break there moves every scrap, clearance and IL figure at once.
    """
    _chdir(workspace)
    from common import episodes as E
    d = pd.read_parquet("data/prepared.parquet")

    bad = E.flow_identity_violations(d)
    assert len(bad) == 0, bad.head(10).to_string()

    flow = E.episode_flow(d)
    assert (flow.opening + flow.arrived
            == flow.sold + flow.vanished + flow.remaining_from_last_row).all()

    # the two consequences the owner named, on the whole population
    assert (flow.clearance <= 1.0).all(), "clearance above 1 is never valid"
    assert (flow[flow.clearance >= 1.0].remaining_from_last_row == 0).all(), \
        "an episode that sold everything it had cannot also carry scrap"

    # and the frame carries the columns the identity is built from
    for col in ("units_restocked", "units_shrink", "episode_supply",
                "episode_clearance", "flow_reconciled"):
        assert col in d.columns, col
    per_ep = d.groupby("episode_id").first()
    assert (per_ep.episode_supply
            == flow.opening.reindex(per_ep.index)
            + flow.arrived.reindex(per_ep.index)).all()


def test_the_manifest_reports_the_identity(workspace):
    """It is checked on every run, not only in the test suite."""
    _chdir(workspace)
    with open("artifacts/split_manifest.json") as f:
        wf = json.load(f)["data_quality_waterfall"]
    ident = [s for s in wf if s["step"] == "dp_eligible"][0]["flow_identity"]
    assert ident["holds"] is True
    assert ident["violations"] == 0
    assert ident["episodes_checked"] > 0
    assert "opening + restocked" in ident["rule"]
