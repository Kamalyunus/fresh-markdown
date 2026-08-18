"""pipeline.assurance -- does live production still match what we froze?

The unit suite checks logic against fixtures. It cannot check the thing that
has actually broken this system every time: an ASSUMPTION about real data. The
censoring basis, the horizon taken from a row count, scrap read as zero, a
stale `rho` paste -- none of those were logic bugs, and none would have been
caught by a test that supplies its own inputs.

So this module tests the frozen artifacts against the live world instead, and
every check is built to fail LOUDLY on the failure mode that would otherwise be
silent:

  reproduction   Re-solve logged decisions from their own event payload. The DP
                 is deterministic, so a mismatch means something moved
                 underneath it -- config edit, artifact swap, bad deploy,
                 library upgrade. One check, four causes.

  dispersion     Is live demand as lumpy as the frozen `r` claims? Every
                 learning bound is derived assuming it is. Two statistics are
                 exact under censoring and so can be tested directly: the
                 chance of selling nothing, and the chance of selling out.

  correlation    Re-measure `rho` on live residuals. It divides ALL accumulated
                 evidence through deff, and drift here has no symptom -- the
                 loop looks healthy while being wrong about how much it knows.

  exploration    Is the applied price really a uniform draw from the affordable
                 set? The causal claim rests entirely on that. A biased draw
                 breaks nothing visible: prices stay legal, IL stays reported,
                 and the evidence quietly stops being evidence.

None of these suspend pricing on their own. They are reported beside the
section 15 families and read at the operator gate, because the right response
to "the world stopped matching the model" is a human decision, not an
automatic one.

Usage:
    python3 -m pipeline.assurance --out reports/assurance.json
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import nbinom

from common.config import load_config, deff, design_effect
from events.store import EventStore
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
    """Re-solve recent decisions and assert they come out the same.

    The most RECENT decisions, not a random sample: if a deploy or a config
    edit broke something, that is where it shows, and a uniform sample over the
    whole history would dilute it.
    """
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
        except Exception as exc:                      # a decision that no
            failures.append({"decision_id": evt.get("decision_id"),   # longer
                             "error": f"{type(exc).__name__}: {exc}"})  # solves
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
        # A deterministic solver that does not reproduce is the one failure
        # here that is never benign.
        "verdict": ("PASS" if checked and not mismatches
                    else "INSUFFICIENT" if not checked else "FAIL"),
    }


# -------------------------------------------------------------- 2 · dispersion
def _pairs(decisions, outcomes):
    """(decision, outcome) for finalized outcomes that name a known decision."""
    dec = {d["decision_id"]: d for d in decisions}
    out = []
    for o in outcomes:
        d = dec.get(o.get("decision_id"))
        if d is not None and o.get("starting_inventory", 0) >= 1:
            out.append((d, o))
    return out


def dispersion_fit(decisions, outcomes, cfg):
    """Is live demand as lumpy as the frozen r says?

    Two statistics survive censoring exactly, which is why these two and not a
    variance comparison. With at least one unit on the shelf,

        P(sold = 0)  =  P(D = 0)          -- selling nothing is never censored
        P(sold >= q) =  P(D >= q)         -- selling out is exactly the tail

    so both can be compared against the negative binomial directly, with no
    correction and no bias. Binned by predicted demand, because miscalibration
    that is flat in mu is a level problem and miscalibration that grows with mu
    is a shape problem -- and only the second one indicts r.
    """
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
            # binomial standard error on the EXPECTED rate: the null is "the
            # frozen r is right", so the variance under the null is the one
            # that belongs in the denominator
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
        # Demand lumpier than r says makes every bounded update overconfident,
        # which is the direction that costs correctness rather than speed.
        "verdict": "PASS" if not flagged else "FAIL",
    }


# ------------------------------------------------------------- 3 · correlation
def correlation_drift(decisions, outcomes, cfg):
    """Re-measure rho on live residuals, against the frozen artifact.

    Measured on the SAME basis bootstrap.fit_dispersion used -- residuals
    against raw mu at the WORKING elasticity (the prior fallback), not against
    the posterior mean. Using the moved posterior would make rho drift for a
    reason that has nothing to do with the world, and the number would no
    longer be comparable to the one deff was frozen from.
    """
    ac = cfg["assurance"]
    pcfg = cfg["pricing"]
    eps0 = cfg["posterior"]["prior"]["fallback_mean"]
    pairs = _pairs(decisions, outcomes)

    rows = {}
    for d, o in pairs:
        mu = mu_at(d["reference_mu"], d["applied_discount"],
                   d["reference_discount"], eps0, pcfg["demand_floor"])
        rows.setdefault(d["episode_id"], []).append(
            (float(o["units_sold"]) - mu, d["applied_discount"]))

    usable = {k: v for k, v in rows.items() if len(v) >= ac["rho_min_hours_per_episode"]}
    if len(usable) < ac["rho_min_episodes"]:
        return {"episodes": len(usable), "required": ac["rho_min_episodes"],
                "verdict": "INSUFFICIENT"}

    resid = np.array([x for v in usable.values() for x, _ in v])
    means = np.array([np.mean([x for x, _ in v]) for v in usable.values()])
    total = float(resid.var(ddof=1))
    rho_live = float(np.clip(means.var(ddof=1) / total, 0.0, 0.95)) if total > 0 else 0.0

    moved = [v for v in usable.values() if len({round(dd, 6) for _, dd in v}) > 1]
    hours = float(np.mean([len(v) for v in (moved or list(usable.values()))]))
    deff_live = design_effect(rho_live, hours)

    rho_frozen = cfg["dispersion"]["rho"]
    drift = abs(rho_live - rho_frozen)
    return {
        "episodes": len(usable),
        "rho_live": round(rho_live, 4),
        "rho_frozen": round(float(rho_frozen), 4),
        "rho_drift": round(drift, 4),
        "mean_forced_hours_live": round(hours, 3),
        "deff_live": round(deff_live, 3),
        "deff_frozen": round(float(deff(cfg)), 3),
        # deff divides accumulated information, so a drift here silently
        # rescales every update in the window -- in whichever direction nobody
        # is watching.
        "verdict": "PASS" if drift <= ac["rho_drift_alert"] else "FAIL",
    }


# ------------------------------------------------------------- 4 · exploration
def exploration_uniformity(decisions, cfg):
    """Is the applied price a uniform draw from the affordable set?

    Reconstructs the affordable set by re-solving, finds where the applied tier
    sits inside it, and maps that rank onto [0, 1). Under a uniform draw the
    mapped value is uniform whatever the set size, so sets of two and sets of
    nine can be pooled into one test instead of needing one test each.

    Also checks the invariant that makes the rate meaningful: a non-empty
    affordable set MUST produce an exploration, since select() draws whenever
    one exists.
    """
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
        affordable, _ = affordable_set(res, evt["tau_current"])
        j_applied = _tier_index(res.tiers, evt["applied_discount"], step)
        if j_applied is None or j_applied not in affordable or len(affordable) < 2:
            # size-1 sets carry no information about uniformity: the draw was
            # forced. Not a failure, just not evidence.
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
    return {
        "exploration_draws": n,
        "unreconstructed": unreconstructed,
        "affordable_but_not_explored": contradictions,
        "bin_counts": counts.tolist(),
        "expected_per_bin": round(expected, 1),
        "chi_square": round(stat, 3),
        "p_value": round(pval, 4),
        # A biased draw is the one failure that leaves no trace in prices, IL,
        # or any business metric -- it only corrupts the causal claim.
        "verdict": ("FAIL" if pval < ac["uniformity_alert_p"] or contradictions
                    else "PASS"),
    }


def run(decisions, outcomes, cfg):
    report = {
        "reproduction": reproduction(decisions, cfg),
        "dispersion": dispersion_fit(decisions, outcomes, cfg),
        "correlation": correlation_drift(decisions, outcomes, cfg),
        "exploration": exploration_uniformity(decisions, cfg),
    }
    report["failing"] = sorted(k for k, v in report.items()
                               if isinstance(v, dict) and v.get("verdict") == "FAIL")
    report["verdict"] = "FAIL" if report["failing"] else "PASS"
    return report


def main():
    ap = argparse.ArgumentParser(prog="pipeline.assurance", description=__doc__)
    ap.add_argument("--out", default="reports/assurance.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = EventStore(cfg)
    report = run(store.load_decisions(), store.load_outcomes(), cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    for name in ("reproduction", "dispersion", "correlation", "exploration"):
        print(f"{name:14s} {report[name]['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
