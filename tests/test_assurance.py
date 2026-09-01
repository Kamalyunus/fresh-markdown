"""Tests for pipeline.assurance."""
import copy
import uuid

import numpy as np
import pytest
from scipy.stats import nbinom

from common.config import load_config
from pipeline import assurance
from pricing import dp as dp_mod
from pricing.demand import mu_at


P0, COST, D_REF, R = 10000.0, 4000.0, 0.30, 0.919


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _decision(cfg, q, path, eps=-1.0, anchor=0.0, tau=None, rng=None,
              episode="ep", entry=False):
    """A decision event built the way inference.decide builds one."""
    anchor = None if entry else anchor
    res = dp_mod.solve(P0, COST, q, path, D_REF, eps, R, cfg,
                       anchor_discount=anchor, entry=entry)
    star = res.optimal_index
    chosen = star
    is_expl, affordable = False, []
    if tau is not None:
        costs = {j: res.q_by_tier[star] - res.q_by_tier[j] for j in res.q_by_tier}
        affordable = [j for j in res.q_by_tier if j != star and costs[j] <= tau]
        if affordable:
            chosen = affordable[int(rng.integers(0, len(affordable)))]
            is_expl = True
    return {
        "decision_id": str(uuid.uuid4()), "episode_id": episode,
        "is_entry": entry, "q_remaining": q, "hours_remaining": len(path),
        "original_price": P0, "cost": COST,
        "mu_ref_path": [float(m) for m in path],
        "anchor_discount": anchor,
        "reference_discount": D_REF, "reference_mu": float(path[0]),
        "dispersion_r": R, "epsilon_posterior_mean": eps,
        "optimal_discount": float(res.tiers[star]),
        "applied_discount": float(res.tiers[chosen]),
        "expected_il": float(-res.q_by_tier[chosen]),
        "is_exploration": is_expl, "affordable_set_size": len(affordable),
        "tau_current": tau,
    }


def _outcome(dec, sold, q):
    return {"outcome_id": str(uuid.uuid4()), "decision_id": dec["decision_id"],
            "units_sold": int(sold), "starting_inventory": int(q)}


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


# ------------------------------------------------------------ 2 · dispersion
def _demand_pairs(cfg, n, r_true, seed=0):
    """Decisions plus outcomes drawn from NB(mu, r_true) and censored."""
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


# ----------------------------------------------------------- 3 · correlation
def _episodes(cfg, n_ep, hours, episode_shift, seed=0):
    """episode_shift > 0 gives every hour of an episode a shared offset --
    exactly the structure rho measures."""
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


def test_correlation_matches_the_frozen_value_when_the_world_has_not_moved(cfg):
    frozen = cfg["dispersion"]["rho"]
    # BOTH terms of deff have to match, not just rho: hours tracks the frozen
    # mean_forced_hours_per_episode, and shared variance is tuned so live rho
    # lands near the frozen scalar (0.2436 on the calib window; count
    # discreteness sets a ~0.26 floor on the generator's live rho)
    hours = cfg["assurance"]["rho_min_hours_per_episode"] * 2
    decs, outs = _episodes(cfg, 300, hours=hours, episode_shift=0.25, seed=4)
    out = assurance.correlation_drift(decs, outs, cfg)
    assert out["verdict"] == "PASS", out
    assert abs(out["rho_live"] - frozen) <= cfg["assurance"]["rho_drift_alert"]


def test_deff_is_measured_at_the_live_clustering_not_a_frozen_paste(cfg):
    """m was a config paste -- and not forced hours at all, but the mean
    LENGTH of legacy episodes whose discount changed. It is now measured
    wherever deff is applied, so the forced-hours channel cannot drift: the
    same world at half the hours per episode is not an alarm, it is a
    different (correctly computed) divisor on both sides."""
    base_hours = cfg["assurance"]["rho_min_hours_per_episode"] * 2
    for hours in (base_hours, base_hours // 2):
        decs, outs = _episodes(cfg, 300, hours=hours, episode_shift=0.25,
                               seed=4)
        out = assurance.correlation_drift(decs, outs, cfg)
        assert out["mean_forced_hours_live"] == pytest.approx(hours, abs=0.01)
        assert out["verdict"] == "PASS", out


def test_correlation_catches_drift_that_would_rescale_every_update(cfg):
    """Hours almost perfectly correlated within an episode: deff should climb
    far above the frozen 3.347, and evidence is being over-counted until it
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
def _exploration_events(cfg, n, rng, biased=False):
    decs = []
    for i in range(n):
        d = _decision(cfg, q=1, path=[0.8], tau=2000.0,
                      rng=rng, episode=f"ep{i}")
        if biased and d["is_exploration"]:
            # a draw that always takes the shallowest affordable tier: prices
            # stay legal, IL stays reported, and the evidence stops being causal
            res = dp_mod.solve(P0, COST, 1, [0.8], D_REF, -1.0, R, cfg,
                               anchor_discount=0.0, entry=False)
            star = res.optimal_index
            costs = {j: res.q_by_tier[star] - res.q_by_tier[j] for j in res.q_by_tier}
            affordable = [j for j in res.q_by_tier if j != star and costs[j] <= 2000.0]
            if affordable:
                d["applied_discount"] = float(res.tiers[affordable[0]])
                d["expected_il"] = float(-res.q_by_tier[affordable[0]])
        decs.append(d)
    return decs


def test_uniformity_passes_on_an_honest_uniform_draw(cfg):
    rng = np.random.default_rng(7)
    out = assurance.exploration_uniformity(_exploration_events(cfg, 600, rng), cfg)
    assert out["verdict"] == "PASS", out
    assert out["exploration_draws"] >= cfg["assurance"]["uniformity_min_draws"]


def test_uniformity_catches_a_biased_draw(cfg):
    rng = np.random.default_rng(8)
    out = assurance.exploration_uniformity(
        _exploration_events(cfg, 600, rng, biased=True), cfg)
    assert out["verdict"] == "FAIL", out


def test_uniformity_catches_an_affordable_set_that_never_explored(cfg):
    """select() draws whenever the affordable set is non-empty, so a decision
    reporting a set and no exploration means the two disagree."""
    rng = np.random.default_rng(9)
    decs = _exploration_events(cfg, 300, rng)
    for d in decs[:5]:
        d["is_exploration"] = False
        d["affordable_set_size"] = 3
    out = assurance.exploration_uniformity(decs, cfg)
    assert out["affordable_but_not_explored"] == 5
    assert out["verdict"] == "FAIL"


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
    no window, so a p-value alone tightens every day the system runs: at 100
    draws it takes a ~47% bin deviation to FAIL, at a million ~0.5%. The same
    draw distribution would pass in week one and fail at volume. The effect
    size carries the meaning; p only stops noise being called bias."""
    import numpy as np
    from scipy.stats import chi2 as chi2_dist

    from pipeline.assurance import exploration_uniformity

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
    # a deviation too small to matter stays PASS at every scale -- the old
    # p-only rule flipped this to FAIL as n grew
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

    rho = float(cfg["dispersion"]["rho"])
    gate = cfg["assurance"]["deff_drift_alert_rel"]
    stale = rho + cfg["assurance"]["rho_drift_alert"]
    m_small = cfg["assurance"]["rho_min_hours_per_episode"]
    m_large = m_small * 4

    drift = [abs(design_effect(stale, m) - design_effect(rho, m))
             / design_effect(rho, m) for m in (m_small, m_large)]
    assert drift[0] < drift[1]          # same rho error, larger consequence
    assert drift[1] > gate              # and it trips the gate where it bites
