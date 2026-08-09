"""tools.deck_numbers -- pull every number the design doc and the technical
deck quote, out of the run reports, into one pasteable block.

The deck and docs/design.md quote roughly two dozen measured quantities. They
go stale every time the pipeline is re-run (a retrain at the launch freeze
changes most of them). This prints exactly those quantities, labelled with the
slide that carries each one, so refreshing the deck is a copy-paste rather
than a hunt through four JSON files.

Missing files are reported, not fatal -- run it with whatever you have.

Usage:
    python3 -m tools.deck_numbers \
        --backtest reports/backtest.json \
        [--shadow reports/shadow.json] \
        [--phase0 reports/phase0.json] \
        [--thresholds reports/thresholds.json]
"""

import argparse
import json
import os


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def g(obj, *keys, default="--"):
    """Nested get that tolerates any missing level."""
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return default
        obj = obj[k]
    return obj


def section(title):
    print()
    print(title)
    print("-" * len(title))


def row(label, value, slides=""):
    tag = f"  [{slides}]" if slides else ""
    print(f"{label:<44} {value}{tag}")


def main():
    ap = argparse.ArgumentParser(prog="tools.deck_numbers")
    ap.add_argument("--backtest", required=True,
                    help="the GATE-PASSING backtest report")
    ap.add_argument("--shadow", default="reports/shadow.json")
    ap.add_argument("--phase0", default="reports/phase0.json")
    ap.add_argument("--thresholds", default="reports/thresholds.json")
    args = ap.parse_args()

    bt, sh = load(args.backtest), load(args.shadow)
    p0, th = load(args.phase0), load(args.thresholds)

    for name, obj, path in (("backtest", bt, args.backtest),
                            ("shadow", sh, args.shadow),
                            ("phase0", p0, args.phase0),
                            ("thresholds", th, args.thresholds)):
        if obj is None:
            print(f"note: no {name} report at {path} -- those rows show '--'")

    fid = g(bt, "fidelity", default={})
    pol = g(bt, "policy_deltas", default={})

    section("CALIBRATION GATE -- slides 1, 11, 12; design 8, 9.2, 10")
    row("gate metric", g(fid, "calibration_gate_metric"), "12")
    row("gate band", g(fid, "calibration_gate_band"), "1, 11, 12")
    row("gate value", g(fid, "calibration_gate_value"), "1, 12")
    row("gate verdict", g(fid, "calibration_gate"), "1, 12, 13")
    row("post-calibration episode sold ratio",
        g(fid, "fidelity_episode_sold_ratio"), "11")
    row("hourly MAE (gate window)", g(fid, "fidelity_hourly_mae"), "11")
    row("sold ratio by window", g(fid, "by_window"), "11, 14")
    row("share of hours non-zero", g(fid, "fidelity_pct_nonzero"), "2")

    section("POLICY -- slides 2, 12; design 8")
    row("episodes replayed", g(pol, "episodes_replayed"), "2, 12")
    row("actual IL%", g(pol, "actual_il_pct"), "2, 12")
    row("actual IL (currency)", g(pol, "actual_il"), "2, 12")
    row("actual clearance", g(pol, "actual_clearance"), "2")
    row("DP IL reduction vs legacy (like-for-like)",
        g(pol, "policy_gap_like_for_like", "dp_il_reduction_pct_of_legacy"),
        "12")
    row("clearance delta",
        g(pol, "policy_gap_like_for_like", "clearance_delta"), "12")

    section("EXPLORATION -- slide 9; design 8")
    tau = g(bt, "tau_initial_derivation", default={})
    row("tau_initial", g(tau, "tau_initial"), "9, 12")
    row("implied daily spend", g(tau, "implied_daily_spend"), "9")
    row("daily budget", g(tau, "daily_budget"), "9")
    row("cost-distribution quantile", g(tau, "cost_distribution_quantile"), "9")

    section("DISPERSION / POWER -- slides 7, 10, 12; design 8")
    row("rho", g(p0, "m3_intra_episode_correlation", "rho"), "12")
    row("mean forced hours",
        g(p0, "m3_intra_episode_correlation", "mean_forced_hours_per_episode"),
        "12")
    row("implied deff",
        g(p0, "m3_intra_episode_correlation", "implied_deff"), "7, 10, 12")
    row("IL% clustered SE",
        g(p0, "m6_il_pct", "il_pct_ratio_se_clustered"), "12")

    section("SHADOW GATE -- slides 1, 13; design 9.4, 10")
    sg = g(sh, "shadow_gate", default={})
    row("event completeness", g(sg, "event_completeness", "value"), "13")
    row("matched decision rate", g(sg, "matched_decision_rate", "value"), "13")
    row("cost-floor violations", g(sg, "cost_floor_violations", "value"), "13")
    row("verdict", g(sg, "verdict"), "1, 13")
    row("drift ratio at legacy price",
        g(sh, "realised_vs_predicted_sold_ratio_at_legacy_price"), "13")
    row("decisions", g(sh, "decision_count"), "13")
    row("would-be forced rate",
        g(sh, "exploration_would_be", "forced_rate"), "13")
    row("solver latency p95 (s)", g(sh, "solver_latency_p95_s"), "13")

    section("OWNER THRESHOLDS -- slide 15; design 12")
    gn = g(th, "guardrail_noise", default={})
    row("scrap 3-sigma daily noise",
        g(gn, "scrap_rate", "three_sigma"), "15")
    row("scrap verdict at current config",
        g(gn, "scrap_rate", "verdict"), "15")
    row("margin 3-sigma daily noise",
        g(gn, "margin_rate", "three_sigma"), "15")
    row("margin verdict at current config",
        g(gn, "margin_rate", "verdict"), "15")
    ab = g(th, "ab_duration", default={})
    row("recommended A/B duration (weeks)",
        g(ab, "recommended_duration_weeks"), "15")
    for label, r in (g(ab, "by_duration", default={}) or {}).items():
        row(f"  detectable MDE @ {label}",
            f"{g(r, 'detectable_mde_rel')} "
            f"({g(r, 'blocks_measured')} blocks)"
            + ("  <-- meets target" if g(r, "meets_target") is True else ""),
            "15")

    section("VERSIONS -- design appendix")
    for k, v in (g(bt, "artifact_versions", default={}) or {}).items():
        row(k, v)


if __name__ == "__main__":
    main()
