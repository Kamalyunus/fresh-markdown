"""Every measured figure printed on the walkthrough, and where it came from.

The panels are prose with the numbers typed into them. That is the right way
to write a document and the wrong way to keep one true: after a re-run every
figure on the replay tab is stale and nothing says so. It has happened
already -- the v3 deck still shows 36.68% where the report gives 38.68%.

So each figure is registered here against the JSON path it was read from, and
against the `baseline_model_version` of the run it was read from. Two checks
fall out, and they are different questions:

  * Does the page still SAY what this ledger says? Guards editing a panel
    without updating the ledger, and vice versa. Always checkable, no report
    needed.
  * Does the page match the report on disk? Only answerable when the report
    is from the SAME model version. A report from a different run is not
    evidence the page is wrong -- it is a different measurement of a
    different thing, and comparing across it is the mistake hard rule 1
    exists to prevent. That case is "stale, cannot verify", which is what
    `pipeline.status` reports after a re-run.

PENDING marks a figure the page has reserved a slot for but has no
measurement behind yet -- the shadow tab, until the hold-out run. Registering
the slot before the number exists is deliberate: it fixes how the result will
be read before anyone can see it.
"""

import json
import os

PENDING = "PENDING"


def won_m(x):
    return f"₩{x / 1e6:.2f}M"


def pct(x):
    return f"{x * 100:.2f}%"


def pct_of(x):
    """A reduction reported as a positive percentage of the other arm."""
    return f"{x * 100:.2f}%"


def dec4(x):
    return f"{x:.4f}"


def count(x):
    return f"{int(x):,}"


# (tab, printed literal, dotted path into the report, formatter)
REPLAY_FIGURES = [
    ("replay", "₩17.11M", "policy_deltas.actual_il", won_m),
    ("replay", "₩13.96M", "policy_deltas.actual_discount_cost", won_m),
    ("replay", "₩3.15M", "policy_deltas.actual_scrap_cost", won_m),
    ("replay", "93.28%", "policy_deltas.actual_clearance", pct),
    ("replay", "0.3094", "policy_deltas.actual_mean_discount", dec4),
    ("replay", "₩19.51M", "policy_deltas.legacy_model_il", won_m),
    ("replay", "₩13.27M", "policy_deltas.legacy_model_discount_cost", won_m),
    ("replay", "₩6.24M", "policy_deltas.legacy_model_scrap_cost", won_m),
    ("replay", "77.58%", "policy_deltas.legacy_model_clearance", pct),
    ("replay", "0.2935", "policy_deltas.legacy_model_mean_discount", dec4),
    ("replay", "₩12.09M", "policy_deltas.dp_il", won_m),
    ("replay", "₩5.26M", "policy_deltas.dp_discount_cost", won_m),
    ("replay", "₩6.84M", "policy_deltas.dp_scrap_cost", won_m),
    ("replay", "76.61%", "policy_deltas.dp_clearance", pct),
    ("replay", "0.1285", "policy_deltas.dp_mean_discount", dec4),
    ("replay", "38.02%",
     "policy_deltas.policy_gap_like_for_like.dp_il_reduction_pct_of_legacy",
     pct_of),
]

SHADOW_FIGURES = [
    ("shadow", PENDING, "window.episodes", count),
    ("shadow", PENDING, "shadow_gate.event_completeness.value", pct),
    ("shadow", PENDING, "shadow_gate.matched_decision_rate.value", pct),
    ("shadow", PENDING, "shadow_gate.cost_floor_violations.value", count),
    ("shadow", PENDING, "recommendation_vs_legacy.mean_recommended_discount", dec4),
    ("shadow", PENDING, "recommendation_vs_legacy.mean_legacy_discount", dec4),
    ("shadow", PENDING, "recommendation_vs_legacy.share_hours_differing", pct),
    ("shadow", PENDING, "realised_vs_predicted_sold_ratio_at_legacy_price", dec4),
    ("shadow", PENDING, "exploration_budget_would_be.spend_over_budget", dec4),
    ("shadow", PENDING, "exploration_budget_would_be.tau_recommended", dec4),
    ("shadow", PENDING,
     "exploration_budget_would_be.tau_controller_trace.days_stop_condition_fires",
     count),
    ("shadow", PENDING, "learning_yield_would_be.episodes_per_bounded_update", dec4),
]

# Which report each tab's figures were read from, and the model version of the
# run that produced them. Bump the version in the same commit that refreshes
# the numbers -- that pairing is the whole mechanism.
SOURCES = {
    "replay": {
        "report": "reports/backtest.json",
        "model_version": "baseline-20260811043259",
        "figures": REPLAY_FIGURES,
    },
    "shadow": {
        "report": "reports/shadow.json",
        "model_version": None,          # set when the hold-out run lands
        "figures": SHADOW_FIGURES,
    },
}


def _dig(obj, path):
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def _report_version(report):
    return ((report.get("artifact_versions") or {}).get("baseline_model_version")
            or (report.get("provenance") or {}).get("baseline_model_version"))


def check(tab, root="."):
    """Compare one tab's registered figures against the report on disk.

    Returns (verdict, detail, problems):
      "pending"   the tab has no measurement behind it yet
      "no report" nothing on disk to check against
      "stale"     the report is from a DIFFERENT run; cannot verify
      "drift"     same run, and a figure disagrees -- a real contradiction
      "ok"        same run, every figure matches
    """
    src = SOURCES[tab]
    pending = [f for f in src["figures"] if f[1] is PENDING]
    if pending and len(pending) == len(src["figures"]):
        return ("pending", f"{len(pending)} figures awaiting a run", [])

    path = os.path.join(root, src["report"])
    if not os.path.exists(path):
        return ("no report", f"{src['report']} not on disk", [])
    with open(path) as f:
        report = json.load(f)

    got = _report_version(report)
    want = src["model_version"]
    if want and got != want:
        return ("stale",
                f"page figures are from {want}, {src['report']} is from {got}",
                [])

    problems = []
    for _, literal, jpath, fmt in src["figures"]:
        if literal is PENDING:
            continue
        value = _dig(report, jpath)
        if value is None:
            problems.append(f"{jpath}: absent from {src['report']}")
        elif fmt(value) != literal:
            problems.append(f"{jpath}: page says {literal}, report says {fmt(value)}")
    if problems:
        return ("drift", f"{len(problems)} figures disagree", problems)
    return ("ok", f"{len(src['figures'])} figures match {got}", [])
