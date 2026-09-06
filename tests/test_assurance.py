"""Tests for daily.assurance."""
import copy
import uuid

import numpy as np
import pytest
from scipy.stats import nbinom

from conftest import P0, COST, decision_event
from daily import assurance
from engine import dp as dp_mod
from engine import explore
from engine.demand import mu_at


D_REF, R = 0.30, 0.919


def _decision(cfg, q, path, eps=-1.0, anchor=0.0, tau=None, rng=None,
              episode="ep", entry=False):
    """A decision event built the way engine.decide builds one: the
    shared contract builder, with the solver's own answer for this state
    and a uniform draw from the affordable set when a tau is in force."""
    anchor = None if entry else anchor
    res = dp_mod.solve(P0, COST, q, path, D_REF, eps, R, cfg,
                       anchor_discount=anchor, entry=entry)
    star = res.optimal_index
    chosen = star
    is_expl, affordable, cost = False, [], 0.0
    if tau is not None:
        affordable, costs = explore.affordable_set(res, tau)
        if affordable:
            chosen = affordable[int(rng.integers(0, len(affordable)))]
            is_expl, cost = True, float(costs[chosen])
    return decision_event(
        decision_id=str(uuid.uuid4()), episode_id=episode, is_entry=entry,
        q_remaining=q, hours_remaining=len(path),
        mu_ref_path=[float(m) for m in path], anchor_discount=anchor,
        reference_discount=D_REF, reference_mu=float(path[0]),
        dispersion_r=R, epsilon_posterior_mean=eps,
        optimal_discount=float(res.tiers[star]),
        optimal_price=P0 * (1 - float(res.tiers[star])),
        applied_discount=float(res.tiers[chosen]),
        applied_price=P0 * (1 - float(res.tiers[chosen])),
        expected_il=float(-res.q_by_tier[chosen]),
        is_exploration=is_expl, exploration_cost=cost,
        affordable_set_size=len(affordable), tau_current=tau, delta_min=0.0)


def _outcome(dec, sold, q, end=None):
    """An outcome for `dec`: `end` defaults to the reconciling count."""
    return {"outcome_id": str(uuid.uuid4()), "decision_id": dec["decision_id"],
            "units_sold": int(sold), "starting_inventory": int(q),
            "ending_inventory": int(q - sold if end is None else end)}


# Solve sets are expensive (hundreds of DP solves) and every test reads the
# same one for the same arguments: built once per module, handed out as
# copies so a test that edits its events cannot leak into the next. The
# shipped config is identical across tests (conftest's `cfg` reloads it),
# which is what makes the argument tuple a sufficient key.
_SOLVED = {}


def _cached(key, build):
    if key not in _SOLVED:
        _SOLVED[key] = build()
    return copy.deepcopy(_SOLVED[key])


# ------------------------------------------------------------- 1 · reproduce
def test_reproduction_passes_on_untouched_events(cfg):
    rng = np.random.default_rng(0)
    decs = [_decision(cfg, q=int(rng.integers(1, 5)), path=[0.8] * 4)
            for _ in range(12)]
    out = assurance.reproduction(decs, cfg)
    assert out["verdict"] == "PASS"
    assert out["decisions_checked"] == 12 and out["mismatch_count"] == 0


def test_reproduction_catches_a_drifted_artifact(cfg):
    """The failure this exists for: the logged price no longer follows from the
    logged inputs, because something moved underneath the solver."""
    decs = [_decision(cfg, q=3, path=[0.8] * 4) for _ in range(4)]
    decs[2]["expected_il"] += 500.0            # as a config or artifact change
    out = assurance.reproduction(decs, cfg)    # would present itself
    assert out["verdict"] == "FAIL"
    assert out["mismatch_count"] == 1
    assert out["failures"][0]["decision_id"] == decs[2]["decision_id"]


def test_reproduction_catches_a_changed_decision(cfg):
    decs = [_decision(cfg, q=3, path=[0.8] * 4) for _ in range(4)]
    decs[1]["optimal_discount"] = decs[1]["optimal_discount"] + 0.05
    assert assurance.reproduction(decs, cfg)["verdict"] == "FAIL"


def test_reproduction_reports_events_it_cannot_replay(cfg):
    decs = [_decision(cfg, q=2, path=[0.8] * 3) for _ in range(3)]
    stripped = copy.deepcopy(decs[0])
    del stripped["mu_ref_path"]                 # a pre-schema event
    out = assurance.reproduction([stripped] + decs, cfg)
    assert out["decisions_skipped_no_inputs"] == 1
    assert out["verdict"] == "PASS"             # skipped, not silently counted


def test_a_solver_that_raises_on_every_decision_is_a_fail_not_insufficient(cfg, monkeypatch):
    """An exception in the re-solve counted as a mismatch but not as a check,
    so a broken solver read INSUFFICIENT (status WARN) instead of FAIL."""
    decs = [_decision(cfg, q=3, path=[0.8] * 4) for _ in range(4)]

    def boom(evt, cfg):
        raise RuntimeError("solver broken")
    monkeypatch.setattr(assurance, "_resolve", boom)
    out = assurance.reproduction(decs, cfg)
    assert out["verdict"] == "FAIL"
    assert out["decisions_checked"] == 4 and out["mismatch_count"] == 4
    assert out["mismatch_rate"] == 1.0
    assert all("RuntimeError" in f["error"] for f in out["failures"])
    # and the top-line verdict follows
    assert assurance.run(decs, [], cfg)["verdict"] == "FAIL"


# ------------------------------------------------------------ 2 · dispersion
def _demand_pairs(cfg, n, r_true, seed=0):
    """Decisions plus outcomes drawn from NB(mu, r_true) and censored."""
    def build():
        rng = np.random.default_rng(seed)
        decs, outs = [], []
        for i in range(n):
            q = int(rng.integers(1, 4))
            d = _decision(cfg, q=q, path=[0.8] * 2, episode=f"ep{i}")
            mu = mu_at(d["reference_mu"], d["applied_discount"], D_REF,
                       d["epsilon_posterior_mean"], cfg["pricing"]["demand_floor"])
            demand = nbinom.rvs(r_true, r_true / (r_true + mu), random_state=rng)
            decs.append(d)
            outs.append(_outcome(d, min(demand, q), q))
        return decs, outs
    return _cached(("demand", n, r_true, seed), build)


def test_dispersion_passes_when_the_world_matches_r(cfg):
    decs, outs = _demand_pairs(cfg, 1500, r_true=R, seed=1)
    out = assurance.dispersion_fit(decs, outs, cfg)
    assert out["verdict"] == "PASS", out


def test_dispersion_catches_demand_lumpier_than_frozen_r(cfg):
    """The dangerous direction: real demand burstier than r claims makes every
    bounded update overconfident, and nothing else in the system would say so."""
    decs, outs = _demand_pairs(cfg, 1500, r_true=0.15, seed=2)
    out = assurance.dispersion_fit(decs, outs, cfg)
    assert out["verdict"] == "FAIL"
    assert out["bins_flagged"] > 0


def test_dispersion_reports_insufficient_rather_than_guessing(cfg):
    decs, outs = _demand_pairs(cfg, 20, r_true=R, seed=3)
    assert assurance.dispersion_fit(decs, outs, cfg)["verdict"] == "INSUFFICIENT"


def test_dispersion_reads_a_stockout_by_the_shared_censoring_rule(cfg):
    """`sold >= q` called a restocked hour a stockout: the shelf never sat
    empty, demand was observed exactly. The check uses the ONE censoring
    rule (episodes.is_censored_hour) and leaves restocked hours out."""
    decs, outs = _demand_pairs(cfg, 1500, r_true=R, seed=1)
    n_min = cfg["assurance"]["dispersion_min_outcomes"]
    # restock every hour that would otherwise read as a stockout
    restocked = 0
    for o in outs:
        if o["units_sold"] >= o["starting_inventory"]:
            o["ending_inventory"] = o["starting_inventory"] + 3   # stock arrived
            o["adjustment_reason"] = "intraday_restock"
            restocked += 1
    assert restocked > 0
    out = assurance.dispersion_fit(decs, outs, cfg)
    assert out["outcomes"] == len(outs) - restocked        # left out, not misread
    assert out["outcomes"] >= n_min
    # and a write-off row (ending 0 with stock left) is not a stockout either:
    # with every sell-out gone the observed stockout rate is exactly zero
    assert all(b["observed"] == 0.0 for b in out["stockout"]["bins"])


# ----------------------------------------------------------- 3 · correlation
def _episodes(cfg, n_ep, hours, episode_shift, seed=0):
    """episode_shift > 0 gives every hour of an episode a shared offset --
    exactly the structure rho measures."""
    def build():
        rng = np.random.default_rng(seed)
        decs, outs = [], []
        for e in range(n_ep):
            shared = rng.normal(0, episode_shift)
            for _ in range(hours):
                d = _decision(cfg, q=3, path=[0.8] * 2, episode=f"ep{e}")
                mu = mu_at(d["reference_mu"], d["applied_discount"], D_REF,
                           -1.0,
                           cfg["pricing"]["demand_floor"])
                sold = max(0, int(round(mu + shared + rng.normal(0, 0.3))))
                decs.append(d)
                outs.append(_outcome(d, min(sold, 3), 3))
        return decs, outs
    return _cached(("episodes", n_ep, hours, episode_shift, seed), build)


def _frozen_at_live_rho(cfg, decs, outs):
    """cfg whose frozen rho IS what this world measures -- "the world has not
    moved", stated directly instead of tuning a generator constant until it
    happens to match whatever config ships today."""
    live = assurance.correlation_drift(decs, outs, cfg)["rho_live"]
    return {**cfg, "dispersion": {**cfg["dispersion"], "rho": live}}, live


def test_correlation_passes_when_the_world_has_not_moved(cfg):
    hours = cfg["assurance"]["rho_min_hours_per_episode"] * 2
    decs, outs = _episodes(cfg, 300, hours=hours, episode_shift=0.25, seed=4)
    matched, live = _frozen_at_live_rho(cfg, decs, outs)
    out = assurance.correlation_drift(decs, outs, matched)
    assert out["verdict"] == "PASS", out
    assert out["rho_live"] == pytest.approx(live)
    # rho_live is reported to 4dp, so feeding it back leaves only rounding
    assert out["deff_drift_rel"] < cfg["assurance"]["deff_drift_alert_rel"] / 100


def test_deff_is_measured_at_the_live_clustering_not_a_frozen_paste(cfg):
    """m is measured wherever deff is applied, so the forced-hours channel
    cannot drift: the same world at half the hours per episode is not an
    alarm, it is a different (correctly computed) divisor on both sides."""
    base_hours = cfg["assurance"]["rho_min_hours_per_episode"] * 2
    for hours in (base_hours, base_hours // 2):
        decs, outs = _episodes(cfg, 300, hours=hours, episode_shift=0.25,
                               seed=4)
        matched, _ = _frozen_at_live_rho(cfg, decs, outs)
        out = assurance.correlation_drift(decs, outs, matched)
        assert out["mean_forced_hours_live"] == pytest.approx(hours, abs=0.01)
        assert out["verdict"] == "PASS", out


def test_correlation_catches_drift_that_would_rescale_every_update(cfg):
    """Hours almost perfectly correlated within an episode: deff should climb
    far above the frozen value, and evidence is being over-counted until it
    does."""
    decs, outs = _episodes(cfg, 300, hours=4, episode_shift=3.0, seed=5)
    out = assurance.correlation_drift(decs, outs, cfg)
    assert out["verdict"] == "FAIL"
    assert out["rho_live"] > cfg["dispersion"]["rho"]
    assert out["deff_live"] > out["deff_frozen"]


def test_correlation_reports_insufficient_on_a_thin_window(cfg):
    decs, outs = _episodes(cfg, 5, hours=4, episode_shift=0.5, seed=6)
    assert assurance.correlation_drift(decs, outs, cfg)["verdict"] == "INSUFFICIENT"


# ----------------------------------------------------------- 4 · exploration
def _exploration_events(cfg, n, seed, biased=False):
    def build():
        rng = np.random.default_rng(seed)
        decs = []
        for i in range(n):
            d = _decision(cfg, q=1, path=[0.8], tau=2000.0,
                          rng=rng, episode=f"ep{i}")
            if biased and d["is_exploration"]:
                # a draw that always takes the shallowest affordable tier:
                # prices stay legal, IL stays reported, and the evidence stops
                # being causal
                res = dp_mod.solve(P0, COST, 1, [0.8], D_REF, -1.0, R, cfg,
                                   anchor_discount=0.0, entry=False)
                affordable, _ = explore.affordable_set(res, 2000.0)
                if affordable:
                    d["applied_discount"] = float(res.tiers[affordable[0]])
                    d["expected_il"] = float(-res.q_by_tier[affordable[0]])
            decs.append(d)
        return decs
    return _cached(("exploration", n, seed, biased), build)


def test_uniformity_passes_on_an_honest_uniform_draw(cfg):
    out = assurance.exploration_uniformity(_exploration_events(cfg, 600, 7), cfg)
    assert out["verdict"] == "PASS", out
    assert out["exploration_draws"] >= cfg["assurance"]["uniformity_min_draws"]


def test_uniformity_catches_a_biased_draw(cfg):
    out = assurance.exploration_uniformity(
        _exploration_events(cfg, 600, 8, biased=True), cfg)
    assert out["verdict"] == "FAIL", out


def test_uniformity_catches_an_affordable_set_that_never_explored(cfg):
    """select() draws whenever the affordable set is non-empty, so a decision
    reporting a set and no exploration means the two disagree."""
    decs = _exploration_events(cfg, 300, 9)
    for d in decs[:5]:
        d["is_exploration"] = False
        d["affordable_set_size"] = 3
    out = assurance.exploration_uniformity(decs, cfg)
    assert out["affordable_but_not_explored"] == 5
    assert out["verdict"] == "FAIL"


def test_a_suspended_decision_is_not_a_contradiction(cfg):
    """While exploration is suspended decide() records tau_current None and an
    empty affordable set; that is exploitation by design, not a draw that
    failed to happen."""
    decs = _exploration_events(cfg, 300, 9)
    for d in decs[:20]:
        d.update(is_exploration=False, affordable_set_size=0, tau_current=None,
                 applied_discount=d["optimal_discount"])
    out = assurance.exploration_uniformity(decs, cfg)
    assert out["affordable_but_not_explored"] == 0


# ------------------------------------------------------------------- wiring
def test_run_aggregates_and_names_the_failing_checks(cfg):
    decs = [_decision(cfg, q=3, path=[0.8] * 4) for _ in range(4)]
    decs[0]["expected_il"] += 900.0
    report = assurance.run(decs, [], cfg)
    assert report["verdict"] == "FAIL"
    assert "reproduction" in report["failing"]
    # thin families report INSUFFICIENT and must not be counted as failures
    assert report["dispersion"]["verdict"] == "INSUFFICIENT"
    assert "dispersion" not in report["failing"]


def test_uniformity_needs_size_as_well_as_significance(cfg):
    """chi-square power grows with n and the event store is append-only with
    no window, so a p-value alone tightens every day the system runs. The
    effect size carries the meaning; p only stops noise being called bias."""
    import numpy as np
    from scipy.stats import chi2 as chi2_dist

    from daily.assurance import exploration_uniformity

    bins = cfg["assurance"]["uniformity_bins"]
    size_gate = cfg["assurance"]["uniformity_max_bin_deviation"]

    def verdict(n, dev):
        """n draws whose first bin sits `dev` above uniform."""
        counts = [round(n / bins)] * bins
        counts[0] = round(counts[0] * (1 + dev))
        counts[-1] = round(counts[-1] * (1 - dev))
        u, edges = [], np.linspace(0, 1, bins + 1)
        for j, c in enumerate(counts):
            u += [float((edges[j] + edges[j + 1]) / 2)] * c
        obs = np.array(counts, dtype=float)
        exp = obs.sum() / bins
        stat = float(((obs - exp) ** 2 / exp).sum())
        p = float(chi2_dist.sf(stat, bins - 1))
        max_dev = float(np.max(np.abs(obs - exp)) / exp)
        return ("FAIL" if (p < cfg["assurance"]["uniformity_alert_p"]
                           and max_dev > size_gate) else "PASS")

    tiny = size_gate / 3
    # a deviation too small to matter stays PASS at every scale
    assert verdict(10_000, tiny) == "PASS"
    assert verdict(1_000_000, tiny) == "PASS"
    # a real bias still fails once there is enough data to be sure of it
    assert verdict(100_000, size_gate * 3) == "FAIL"

    # and a contradiction FAILs on its own, whatever the draw looks like
    out = exploration_uniformity(
        [{"affordable_set_size": 3, "is_exploration": False,
          "mu_ref_path": [1.0], "epsilon_posterior_mean": -1.0}], cfg)
    assert out["affordable_but_not_explored"] == 1


def test_a_stale_rho_still_fails_weighted_by_todays_clustering(cfg):
    """What remains frozen is rho, and the verdict prices its staleness at
    the clustering in force -- the same absolute rho error matters more when
    episodes contribute more correlated hours."""
    from common.config import design_effect

    # a fixed rho, not the shipped paste: the claim is about how the
    # verdict scales with clustering, not about the owner's extract
    rho = 0.12
    gate = cfg["assurance"]["deff_drift_alert_rel"]
    stale = rho + cfg["assurance"]["rho_drift_alert"]
    m_small = cfg["assurance"]["rho_min_hours_per_episode"]
    m_large = m_small * 4

    drift = [abs(design_effect(stale, m) - design_effect(rho, m))
             / design_effect(rho, m) for m in (m_small, m_large)]
    assert drift[0] < drift[1]          # same rho error, larger consequence
    assert drift[1] > gate              # and it trips the gate where it bites


def test_run_is_insufficient_until_every_check_ran(cfg):
    """Nothing failing and one check that saw almost nothing is not a PASS:
    the operator gate reads the top-line verdict."""
    decs = [_decision(cfg, q=3, path=[0.8] * 4) for _ in range(4)]
    report = assurance.run(decs, [], cfg)
    assert report["reproduction"]["verdict"] == "PASS"
    assert not report["failing"]
    assert "dispersion" in report["insufficient"]
    assert report["verdict"] == "INSUFFICIENT"


def test_icc_survives_a_nan_residual():
    """One NaN must not poison every sum into 0.0 -- 'no clustering', deff 1,
    every posterior step over-weighted."""
    import numpy as np
    from common.config import intraclass_correlation
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 6)
    shared = rng.normal(size=40)[groups]
    resid = shared + rng.normal(size=240) * 0.5
    clean = intraclass_correlation(resid, groups)
    dirty = resid.copy()
    dirty[7] = np.nan
    assert clean > 0.5
    assert intraclass_correlation(dirty, groups) == pytest.approx(clean, abs=0.05)


def test_assurance_grades_only_prices_that_were_actually_charged(cfg):
    """A failed push sold at a price we did not choose. Grading r on it
    indicts the model for the integration's miss."""
    from events.pairs import match_pairs
    decs, outs = _demand_pairs(cfg, 30, r_true=R, seed=4)
    for o in outs[:10]:
        o["execution_status"] = "failed"
    assert len(match_pairs(decs, outs)) == 30
    assert len(match_pairs(decs, outs, learnable=True)) == 20
    assert len(assurance._pairs(decs, outs)) == 20


def test_run_pairs_the_events_once_for_both_live_checks(cfg, monkeypatch):
    """dispersion and correlation grade the same learnable pairs; run builds
    them once and hands them down, and each check answers the same on its
    own."""
    decs, outs = _demand_pairs(cfg, 30, r_true=R, seed=4)
    calls = []
    real = assurance._pairs

    def counting(d, o):
        calls.append(1)
        return real(d, o)

    monkeypatch.setattr(assurance, "_pairs", counting)
    report = assurance.run(decs, outs, cfg)
    assert len(calls) == 1
    assert report["dispersion"] == assurance.dispersion_fit(decs, outs, cfg)
    assert report["correlation"] == assurance.correlation_drift(decs, outs, cfg)


def test_uniformity_reconstructs_the_set_with_the_decision_own_delta_min(cfg):
    """The affordable set the chooser drew from excluded tiers below the
    decision's delta_min; a reconstruction without it would call every
    honest draw non-uniform (the near tiers never appear)."""
    rng = np.random.default_rng(5)
    decs = []
    for _ in range(150):
        d = _decision(cfg, q=4, path=[0.8] * 4, tau=1e9, rng=rng)
        res = assurance._resolve(d, cfg)
        dmin = 0.10
        kept, costs = explore.affordable_set(res, 1e9, dmin)
        j = kept[int(rng.integers(0, len(kept)))]
        d.update(applied_discount=res.tiers[j], is_exploration=True,
                 exploration_cost=costs[j], affordable_set_size=len(kept),
                 delta_min=dmin)
        decs.append(d)
    rep = assurance.exploration_uniformity(decs, cfg)
    assert rep["verdict"] == "PASS", rep
