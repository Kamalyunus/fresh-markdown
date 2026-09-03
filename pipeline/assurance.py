"""pipeline.assurance -- does live production still match what we froze?

Tests the frozen artifacts against the live world (design section 16): four
checks -- reproduction, dispersion, correlation, exploration -- each built to
fail LOUDLY on an assumption break that would otherwise be silent. None
suspends pricing on its own; verdicts are read at the operator gate.
Run: python3 -m pipeline.assurance --out reports/assurance.json
"""

import argparse

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import nbinom

from common.config import (design_effect, intraclass_correlation,
                           load_config)
from events.store import EventStore
from events.pairs import match_pairs
from common.io import write_json
from common.provenance import config_fingerprint
from pricing import dp as dp_mod
from pricing.explore import affordable_set
from pricing.demand import mu_at


def _resolve(evt, cfg):
    """Re-solve one logged decision from nothing but its own event payload."""
    return dp_mod.solve(
        evt["original_price"], evt["cost"], int(evt["q_remaining"]),
        list(evt["mu_ref_path"]), evt["reference_discount"],
        evt["epsilon_posterior_mean"], evt["dispersion_r"], cfg,
        anchor_discount=evt.get("anchor_discount"),
        entry=bool(evt["is_entry"]))


def _tier_index(tiers, discount, step):
    """Index of the tier a logged discount refers to, or None."""
    for j, d in enumerate(tiers):
        if abs(d - discount) <= step / 2:
            return j
    return None


def _replayable(decisions):
    """Events carrying the inputs a re-solve needs (section 16.1)."""
    return [d for d in decisions
            if d.get("mu_ref_path") and "epsilon_posterior_mean" in d]


# --------------------------------------------------------------- 1 · reproduce
def reproduction(decisions, cfg):
    """Re-solve recent decisions and assert they come out the same. The most
    RECENT decisions, not a random sample: a deploy or config break shows
    there, and a uniform sample would dilute it."""
    ac = cfg["assurance"]
    step = cfg["pricing"]["tier_step"]
    pool = _replayable(decisions)
    skipped = len(decisions) - len(pool)
    sample = pool[-ac["reproduction_sample"]:]

    checked = mismatches = 0
    worst_il = 0.0
    failures = []
    for evt in sample:
        try:
            res = _resolve(evt, cfg)
        except Exception as exc:  # a decision that no longer solves
            failures.append({"decision_id": evt.get("decision_id"),
                             "error": f"{type(exc).__name__}: {exc}"})
            mismatches += 1
            continue
        checked += 1

        d_opt = res.tiers[res.optimal_index]
        j_applied = _tier_index(res.tiers, evt["applied_discount"], step)
        il_re = (-res.q_by_tier[j_applied]
                 if j_applied in res.q_by_tier else None)

        price_off = abs(d_opt - evt["optimal_discount"]) > ac["reproduction_discount_tol"]
        il_off = il_re is None or abs(il_re - evt["expected_il"]) > max(
            ac["reproduction_il_tol_abs"],
            ac["reproduction_il_tol_rel"] * abs(evt["expected_il"]))
        if il_re is not None:
            worst_il = max(worst_il, abs(il_re - evt["expected_il"]))
        if price_off or il_off:
            mismatches += 1
            if len(failures) < ac["reproduction_report_max"]:
                failures.append({
                    "decision_id": evt.get("decision_id"),
                    "logged_optimal_discount": evt["optimal_discount"],
                    "resolved_optimal_discount": round(float(d_opt), 6),
                    "logged_expected_il": round(float(evt["expected_il"]), 2),
                    "resolved_expected_il": (round(float(il_re), 2)
                                             if il_re is not None else None),
                    "baseline_model_version": evt.get("baseline_model_version"),
                    "config_version": evt.get("config_version"),
                })

    return {
        "decisions_checked": checked,
        "decisions_skipped_no_inputs": skipped,
        "mismatch_count": mismatches,
        "mismatch_rate": round(mismatches / checked, 6) if checked else None,
        "worst_expected_il_delta": round(worst_il, 6),
        "failures": failures,
        # a deterministic solver that does not reproduce is never benign
        "verdict": ("PASS" if checked and not mismatches
                    else "INSUFFICIENT" if not checked else "FAIL"),
    }


# -------------------------------------------------------------- 2 · dispersion
def _pairs(decisions, outcomes):
    """Pairs with stock on hand whose push SUCCEEDED: a failed push sold at
    a price we did not choose, and grading r or rho on it indicts the model
    for the integration's miss."""
    return [(d, o) for d, o in match_pairs(decisions, outcomes, learnable=True)
            if o.get("starting_inventory", 0) >= 1]


def dispersion_fit(decisions, outcomes, cfg):
    """Is live demand as lumpy as the frozen r says? Two statistics are EXACT
    under censoring (with stock on hand): P(sold=0) = P(D=0) and
    P(sold>=q) = P(D>=q), so both compare against the NB with no correction.
    Binned by mu: only miscalibration that grows with mu indicts r."""
    ac = cfg["assurance"]
    pcfg = cfg["pricing"]
    pairs = _pairs(decisions, outcomes)
    if len(pairs) < ac["dispersion_min_outcomes"]:
        return {"outcomes": len(pairs), "verdict": "INSUFFICIENT",
                "required": ac["dispersion_min_outcomes"]}

    mu, r, q, sold = [], [], [], []
    for d, o in pairs:
        mu.append(mu_at(d["reference_mu"], d["applied_discount"],
                        d["reference_discount"], d["epsilon_posterior_mean"],
                        pcfg["demand_floor"]))
        r.append(d["dispersion_r"])
        q.append(int(o["starting_inventory"]))
        sold.append(int(o["units_sold"]))
    mu, r, q, sold = map(np.asarray, (mu, r, q, sold))

    p = r / (r + mu)
    p_zero = nbinom.pmf(0, r, p)                    # P(D = 0)
    p_out = nbinom.sf(np.maximum(q, 1) - 1, r, p)   # P(D >= q)
    obs_zero = (sold == 0).astype(float)
    obs_out = (sold >= q).astype(float)

    edges = np.quantile(mu, np.linspace(0, 1, ac["dispersion_bins"] + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(mu, edges[1:-1])

    def band(name, pred, obs):
        rows, flagged = [], 0
        for b in range(ac["dispersion_bins"]):
            m = idx == b
            n = int(m.sum())
            if n < ac["dispersion_min_bin"]:
                continue
            e, a = float(pred[m].mean()), float(obs[m].mean())
            # binomial SE on the EXPECTED rate: the null is "the frozen r is
            # right", so the null's variance belongs in the denominator
            se = float(np.sqrt(max(e * (1 - e), 1e-12) / n))
            z = (a - e) / se
            flagged += abs(z) > ac["dispersion_alert_z"]
            rows.append({"bin": b, "n": n,
                         "mu_range": [round(float(mu[m].min()), 4),
                                      round(float(mu[m].max()), 4)],
                         "expected": round(e, 4), "observed": round(a, 4),
                         "z": round(float(z), 2)})
        return {"statistic": name, "bins": rows, "bins_flagged": flagged}

    zero, out = band("P(sold = 0)", p_zero, obs_zero), band("P(sold out)", p_out, obs_out)
    flagged = zero["bins_flagged"] + out["bins_flagged"]
    return {
        "outcomes": len(pairs),
        "zero_sale": zero,
        "stockout": out,
        "bins_flagged": flagged,
        # demand lumpier than r says makes every bounded update overconfident
        "verdict": "PASS" if not flagged else "FAIL",
    }


# ------------------------------------------------------------- 3 · correlation
def correlation_drift(decisions, outcomes, cfg):
    """Re-measure rho on live residuals, against the frozen artifact. Same
    basis as bootstrap.fit_dispersion: residuals against raw mu at the working
    elasticity (per-category prior means via the shared helper), never the
    moved posterior -- a basis mismatch would read as drift and fire the alert."""
    from bootstrap.fit_dispersion import _working_elasticity

    ac = cfg["assurance"]
    pcfg = cfg["pricing"]
    eps_by_cat, eps_fallback = _working_elasticity(cfg)
    pairs = _pairs(decisions, outcomes)

    rows = {}
    for d, o in pairs:
        eps0 = eps_by_cat.get(str(d.get("category")), eps_fallback)
        mu = mu_at(d["reference_mu"], d["applied_discount"],
                   d["reference_discount"], eps0, pcfg["demand_floor"])
        rows.setdefault(d["episode_id"], []).append(
            (float(o["units_sold"]) - mu, d["applied_discount"]))

    usable = {k: v for k, v in rows.items() if len(v) >= ac["rho_min_hours_per_episode"]}
    if len(usable) < ac["rho_min_episodes"]:
        return {"episodes": len(usable), "required": ac["rho_min_episodes"],
                "verdict": "INSUFFICIENT"}

    resid = np.array([x for v in usable.values() for x, _ in v])
    groups = np.array([eid for eid, v in usable.items() for _ in v])
    rho_live = intraclass_correlation(resid, groups,
                                      cfg["dispersion"]["rho_clip_max"])

    moved = [v for v in usable.values() if len({round(dd, 6) for _, dd in v}) > 1]
    hours = float(np.mean([len(v) for v in (moved or list(usable.values()))]))
    deff_live = design_effect(rho_live, hours)

    rho_frozen = cfg["dispersion"]["rho"]
    # BOTH sides at the LIVE clustering: m is now measured per batch wherever
    # deff is applied (common.config.deff_from_episodes), so the forced-hours
    # channel cannot drift by construction and this check isolates what IS
    # still frozen -- rho -- weighted by the consequence it has today.
    deff_frozen = design_effect(rho_frozen, hours)
    # deff = 1 + (m - 1) * rho divides accumulated information, and it drifts
    # through BOTH terms. Judging on rho alone was blind to the m channel:
    # forced hours per episode can move (an exploration-rate change moves it
    # by design) and rescale every update while rho sits still. Judge the
    # quantity with the consequence; rho_drift stays as the diagnostic that
    # says WHICH term moved.
    deff_drift = abs(deff_live - deff_frozen) / max(deff_frozen, 1e-9)
    return {
        "episodes": len(usable),
        "rho_live": round(rho_live, 4),
        "rho_frozen": round(float(rho_frozen), 4),
        "rho_drift": round(abs(rho_live - rho_frozen), 4),
        "mean_forced_hours_live": round(hours, 3),
        "deff_live": round(deff_live, 3),
        "deff_frozen": round(deff_frozen, 3),
        "deff_drift_rel": round(deff_drift, 4),
        "verdict": ("PASS" if deff_drift <= ac["deff_drift_alert_rel"]
                    else "FAIL"),
    }


# ------------------------------------------------------------- 4 · exploration
def exploration_uniformity(decisions, cfg):
    """Is the applied price a uniform draw from the affordable set?
    Reconstructs the set by re-solving and maps the applied tier's rank onto
    [0, 1) -- uniform whatever the set size, so all sets pool into one test.
    Invariant also checked: a non-empty affordable set MUST explore."""
    ac = cfg["assurance"]
    step = cfg["pricing"]["tier_step"]
    u, contradictions, unreconstructed = [], 0, 0

    for evt in _replayable(decisions):
        if evt.get("affordable_set_size", 0) > 0 and not evt.get("is_exploration"):
            contradictions += 1
        if not evt.get("is_exploration") or evt.get("tau_current") is None:
            continue
        try:
            res = _resolve(evt, cfg)
        except Exception:
            unreconstructed += 1
            continue
        # the SAME function the chooser used, so this cannot drift from it
        affordable, _ = affordable_set(res, evt["tau_current"],
                                       evt.get("delta_min", 0.0))
        j_applied = _tier_index(res.tiers, evt["applied_discount"], step)
        if j_applied is None or j_applied not in affordable or len(affordable) < 2:
            # size-1 sets say nothing about uniformity -- not evidence
            if j_applied is not None and j_applied not in affordable:
                unreconstructed += 1
            continue
        u.append((affordable.index(j_applied) + 0.5) / len(affordable))

    n = len(u)
    if n < ac["uniformity_min_draws"]:
        return {"exploration_draws": n, "required": ac["uniformity_min_draws"],
                "affordable_but_not_explored": contradictions,
                "verdict": "INSUFFICIENT"}

    bins = ac["uniformity_bins"]
    counts, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    expected = n / bins
    stat = float(((counts - expected) ** 2 / expected).sum())
    pval = float(chi2_dist.sf(stat, bins - 1))
    # SIGNIFICANT **and** LARGE. chi-square power grows with n, and this store
    # is append-only with no window, so a p-value alone tightens every day the
    # system runs: the same draw distribution that passes in week one fails at
    # volume with nothing about the draw having changed. The effect size is
    # scale-free and carries the meaning -- how far a bin actually sits from
    # uniform -- while p still guards against calling noise a bias at low n.
    max_dev = float(np.max(np.abs(counts - expected)) / expected)
    biased = pval < ac["uniformity_alert_p"] and max_dev > ac["uniformity_max_bin_deviation"]
    return {
        "exploration_draws": n,
        "unreconstructed": unreconstructed,
        "affordable_but_not_explored": contradictions,
        "bin_counts": counts.tolist(),
        "expected_per_bin": round(expected, 1),
        "chi_square": round(stat, 3),
        "p_value": round(pval, 4),
        "max_bin_deviation": round(max_dev, 4),
        # a biased draw leaves no trace in any business metric -- it only
        # corrupts the causal claim
        "verdict": "FAIL" if biased or contradictions else "PASS",
    }


def run(decisions, outcomes, cfg):
    report = {
        "config": config_fingerprint(cfg, "production"),
        "reproduction": reproduction(decisions, cfg),
        "dispersion": dispersion_fit(decisions, outcomes, cfg),
        "correlation": correlation_drift(decisions, outcomes, cfg),
        "exploration": exploration_uniformity(decisions, cfg),
    }
    report["failing"] = sorted(k for k, v in report.items()
                               if isinstance(v, dict) and v.get("verdict") == "FAIL")
    report["insufficient"] = sorted(
        k for k, v in report.items()
        if isinstance(v, dict) and v.get("verdict") == "INSUFFICIENT")
    # a check that saw almost nothing is not one that looked and found
    # nothing: the whole report is INSUFFICIENT until every check ran
    report["verdict"] = ("FAIL" if report["failing"]
                         else "INSUFFICIENT" if report["insufficient"]
                         else "PASS")
    return report


def main():
    ap = argparse.ArgumentParser(prog="pipeline.assurance", description=__doc__)
    ap.add_argument("--out", default="reports/assurance.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = EventStore(cfg)
    report = run(store.load_decisions(), store.load_outcomes(), cfg)

    write_json(args.out, report)
    for name in ("reproduction", "dispersion", "correlation", "exploration"):
        print(f"{name:14s} {report[name]['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
