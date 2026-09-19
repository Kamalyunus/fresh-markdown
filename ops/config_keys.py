"""ops.config_keys -- what each config key is to the chain: who pastes it,
who reads it, what a move re-runs.

The registry (`KEYS`, `DERIVED_IN`, `RERUN`, `READ_BY`, `INERT_PREFIXES`),
the routing built on it (`rerun_for`, `rerun_classes`, `stale_keys`), the
one line editor a paste goes through (`set_scalar`), the report-vintage
judgement both drivers read (`report_staleness`) and the tau paste's
provenance gate (`tau_provenance_error`). `ops.tune` derives the pastes
from the reports; `ops.advance` and `ops.status` route by this table and
never keep a second copy of it.
"""

import os

from common import provenance

# Every config key --apply may touch, one row each.
#   anchor   -> the unique line anchor that carries its scalar. Targeted line
#               edits rather than a YAML round-trip: every value in config.yaml
#               carries the reasoning for it in a comment, and a round-trip
#               would drop them all.
#   measured -> the value is DERIVED from a report rather than chosen by the
#               owner. A number here that disagrees with its report is stale
#               or FOREIGN -- from another run, or from the repo's synthetic
#               fixture -- never a preference. ops.status refuses the
#               chain on any of them, whatever class the finding ended up in:
#               a W the split rule downgrades to OWNER is still a value nobody
#               chose.
#   rerun    -> what a paste invalidates: "none" = read at runtime or mirrors
#               the artifact; "calibration" = the loop turns (3b-5b, NO
#               retrain). tune never pastes a "retrain" key (the training
#               inputs are SET / SET BY OWNER); READ_BY routes them.
KEYS = {
    # derive_thresholds grades the configured increment against I*; shadow's
    # bounded_updates_supported is eff_info / increment, a division a reader
    # can redo, so the increment re-derives thresholds only
    ("learning", "information_increment"):
        {"anchor": "  information_increment:", "measured": True, "rerun": "thresholds"},
    # bounded_step (thresholds) and learning_yield (shadow) both read the
    # rail; the backtest's step_sensitivity grades a rail paste under the
    # step in force on purpose (the paste is gated on that measurement)
    ("learning", "max_mean_step"):
        {"anchor": "  max_mean_step:", "measured": False, "rerun": "thresholds+shadow"},
    ("learning", "max_std_shrink"):
        {"anchor": "  max_std_shrink:", "measured": False, "rerun": "thresholds"},
    ("exploration", "tau_initial"):
        {"anchor": "  tau_initial:", "measured": True, "rerun": "none"},
    ("exploration", "delta_min_log_bias"):
        {"anchor": "  delta_min_log_bias:", "measured": True, "rerun": "shadow"},
    ("dispersion", "rho"):
        {"anchor": "  rho:", "measured": True, "rerun": "none"},
    # W is the OWNER's: the backtest's rolling-origin sweep recommends one
    # (tune reports it as an owner decision) and never pastes it. Pasted, a
    # W move turned the calibration loop, re-ran the backtest, re-scored
    # the sweep and re-ran shadow -- the heaviest re-run class short of a
    # retrain, and one the sweep's own near-ties could cycle
    ("baseline_model", "calibration_fit_trailing_weeks"):
        {"anchor": "  calibration_fit_trailing_weeks:", "measured": False,
         "rerun": "calibration"},
    ("monitoring", "stop_conditions", "scrap_deterioration_pct"):
        {"anchor": "    scrap_deterioration_pct:", "measured": False,
         "rerun": "thresholds"},
    ("monitoring", "stop_conditions", "margin_deterioration_pct"):
        {"anchor": "    margin_deterioration_pct:", "measured": False,
         "rerun": "thresholds"},
    # the sweep ranks windows by share_weeks_in_band, so the band is an input
    # to W -- left "none" DELIBERATELY: re-grading on every band paste would
    # reopen the W oscillation the hysteresis exists for. The band is a
    # reported diagnostic, never a launch gate.
    ("baseline_model", "calibration_gate_band"):
        {"anchor": "  calibration_gate_band:", "measured": True, "rerun": "none"},
}
MEASURED_KEYS = {k for k, v in KEYS.items() if v["measured"]}
# where each MEASURED value comes from -- what advance names when a report
# ran and still produced nothing for it
DERIVED_IN = {
    "learning.information_increment": "reports/thresholds.json -> information_increment_recommendation",
    "exploration.tau_initial": "reports/shadow.json -> tau_initial_derivation",
    "exploration.delta_min_log_bias": "reports/backtest.json -> fidelity (level error at W)",
    "dispersion.rho": "artifacts/rho.json (fit_dispersion)",
    "baseline_model.calibration_gate_band": "reports/backtest.json -> fidelity.calibration_window_sweep",
}

# The re-run classes, weakest to strongest: (reports the class invalidates,
# what to run). A key is an INPUT to the reports its class names and nothing
# else (design 5.14a); classes do not nest, so staleness is the UNION over
# moved keys (stale_keys), never the strongest alone.
RERUN = {
    "none": (set(), "nothing to re-run: every value written is read at runtime "
                    "or mirrors an artifact that already holds it"),
    "thresholds": ({"thresholds"},
                   "a stop threshold moved -- re-derive its verdict:\n"
                   "    python3 -m evaluate.derive_thresholds --input "
                   "data/prepared.parquet"),
    "shadow": ({"shadow"},
               "shadow's own inputs moved -- re-run evaluate.shadow so its "
               "tau derivation and gate are graded under the value in force"),
    "thresholds+shadow": ({"thresholds", "shadow"},
                          "a learning rail moved: re-derive thresholds AND "
                          "re-run evaluate.shadow (both read it)"),
    "backtest": ({"backtest"},
                 "the backtest's own inputs moved -- re-run it (no artifact "
                 "reads them):\n    python3 -m evaluate.backtest --input "
                 "data/prepared.parquet"),
    "backtest+shadow": ({"backtest", "shadow"},
                        "the launch belief moved: re-run evaluate.backtest AND "
                        "re-initialise the posterior, then evaluate.shadow "
                        "(advance does all three in order)"),
    "calibration": ({"backtest", "thresholds", "shadow"},
                    "the calibration loop turned -- settle it WITHOUT retraining:\n"
                    "    python3 -m ops.bootstrap_loop --check-only\n"
                    "  (iterates 3b -> 4 -> 5 -> 5b to CONVERGED and refreshes "
                    "the reports), then re-run evaluate.shadow"),
    "retrain": ({"backtest", "thresholds", "shadow"},
                "the model's own training data or hyper-parameters changed -- a "
                "deliberate `python3 -m ops.advance --retrain` is required; "
                "nothing from before is comparable (rule 1)"),
}
RERUN_ORDER = list(RERUN)
INVALIDATES = {k: v[0] for k, v in RERUN.items()}
RERUN_STEPS = {k: v[1] for k, v in RERUN.items()}
ROUTED_REPORTS = set().union(*INVALIDATES.values())

# keys tune does not paste but a report or a fit READS: prefix -> class.
# Checked before INERT_PREFIXES, so a read key inside an otherwise inert
# section is listed here. Everything in none of the three tables is an edit
# nobody classified and turns the loop ("calibration").
READ_BY = (
    # the training run itself: a new model, never a loop turn (rule 1)
    ("data.split.", "retrain"),
    ("data.exclusion_window", "retrain"),
    ("data.max_window_hours", "retrain"),
    ("data.manufacturing_window_hours", "retrain"),
    ("baseline_model.objective", "retrain"),
    ("baseline_model.tweedie_variance_power", "retrain"),
    ("baseline_model.ref_rate_", "retrain"),
    ("baseline_model.learning_rate", "retrain"),
    ("baseline_model.num_boost_round", "retrain"),
    ("baseline_model.num_leaves", "retrain"),
    ("baseline_model.min_data_in_leaf", "retrain"),
    # the category anchors: d_ref drives the anchor rows and the SKU rate
    # features the model is trained on (prepare_data), not only the factors
    ("reference_discount.", "retrain"),
    # the cell routing: init_posterior re-routes (advance re-inits while
    # unlearned) and shadow prices from that file
    ("posterior.min_episodes_per_week_for_cell", "shadow"),
    # fit inputs inside the assurance section (fit_dispersion, prior_density)
    ("assurance.rho_min_hours_per_episode", "calibration"),
    ("assurance.rho_drift_alert", "calibration"),
    # shadow's inputs
    ("exploration.budget_", "shadow"),           # budget base, scale, window
    ("exploration.tau_adjust_clip", "shadow"),   # the controller trace
    ("exploration.tau_spend_guard", "shadow"),
    ("exploration.tau0_derivation_min_decisions", "shadow"),
    ("exploration.delta_min_bias_multiple", "shadow"),
    ("monitoring.shadow_gate.", "shadow"),
    ("monitoring.stop_conditions.exploration_cost_vs_budget", "shadow"),
    ("learning.update_cadence_days", "shadow"),  # learning_yield's calendar floor
    ("tuning.controller_trace_max_days", "shadow"),
    # derive_thresholds' inputs
    ("monitoring.guardrail_noise_", "thresholds"),
    ("monitoring.guardrail_outlier_sigma_ratio", "thresholds"),
    ("monitoring.stop_conditions.deterioration_smoothing_days", "thresholds"),
    ("monitoring.stop_conditions.persistence_days", "thresholds"),
    ("posterior.min_std", "thresholds"),
    ("tuning.guardrail_inert_floor_multiple", "thresholds"),
    ("tuning.bounded_step_consistent_band", "thresholds"),
    # the launch belief: the backtest prices its DP arm at it, and shadow
    # prices from the re-initialised posterior file (advance re-inits while
    # unlearned, then re-runs shadow on the moved file)
    ("posterior.cold_start_shift_std", "backtest+shadow"),
    # the backtest's own tables

    ("tuning.cost_ratio_bands", "backtest"),
    ("tuning.step_sensitivity_episodes", "backtest"),
    ("tuning.backtest_policy_episodes", "backtest"),
)

# config keys no report reads: runtime-only, paths, the driver's own knobs
INERT_PREFIXES = ("meta.", "events.", "artifacts.", "tuning.", "assurance.",
                  "data.launch_date", "data.split_manifest_path",
                  "monitoring.alert_posterior_std_flat_days",
                  "monitoring.stop_conditions.duplicate_or_unmatched_rate",
                  "monitoring.stop_conditions.price_mismatch_rate",
                  "monitoring.stop_conditions.event_quality_window_days",
                  "exploration.tau_paste_tolerance_rel",
                  "exploration.mode",           # Lane B reads it; shadow always explores
                  "dispersion.rho_paste_tolerance_rel",
                  "posterior.path", "posterior.prior.path",
                  "baseline_model.model_path", "baseline_model.feature_schema_path",
                  "baseline_model.calibration_factor_path",
                  "dispersion.r_lookup_path", "dispersion.rho_path")


def _class_of(key):
    """One dotted key's re-run class. A per-category paste diffs as
    `exploration.delta_min_log_bias.MEAT`, so KEYS is matched on the LONGEST
    prefix -- a miss there once routed every category move to `calibration`
    and turned a floor re-round into a full loop."""
    parts = key.split(".")
    for n in range(len(parts), 1, -1):
        entry = KEYS.get(tuple(parts[:n]))
        if entry is not None:
            return entry["rerun"]
    for prefix, cls in READ_BY:
        if key.startswith(prefix):
            return cls
    # neither pasted, read by a named report, nor inert: an edit nobody
    # classified turns the loop, which re-grades every report
    return "none" if key.startswith(INERT_PREFIXES) else "calibration"


def rerun_for(keys):
    """The strongest re-run a set of moved dotted config keys demands."""
    return max((_class_of(k) for k in keys), key=RERUN_ORDER.index, default="none")


def rerun_classes(keys):
    """Every distinct re-run class the moved keys demand, strongest first
    ("none" only when nothing else is)."""
    found = set()
    for k in keys:
        found.update(_class_of(k).split("+"))
    found -= {"none"}
    return sorted(found, key=RERUN_ORDER.index, reverse=True) or ["none"]


def stale_keys(report, moved):
    """The moved dotted keys that invalidate `report` -- the ONE routing both
    readers (ops.advance and ops.status) use. The backtest's
    exploration ledger reads delta_min too, but nothing pasted comes from it
    (tau_initial is shadow's), so a delta_min paste re-runs shadow only."""
    return [k for k in moved if report in INVALIDATES[rerun_for([k])]]


def set_scalar(text, path, value):
    """Replace one scalar in config.yaml, keeping its comment and every
    other line byte-identical (a YAML round-trip drops comments)."""
    anchor = (KEYS.get(tuple(path)) or {}).get("anchor")
    if anchor is None:
        raise KeyError(f"no line anchor for {'.'.join(path)}")
    hits = [i for i, ln in enumerate(text.splitlines()) if ln.startswith(anchor)]
    if len(hits) != 1:
        raise RuntimeError(
            f"anchor {anchor!r} matched {len(hits)} lines -- refusing to guess")
    lines = text.splitlines(keepends=True)
    i = hits[0]
    line = lines[i]
    head, _, tail = line.partition(":")
    # the anchor must OWN its value on the one line: a block mapping
    # (`key:` with indented children below) replaced line by line leaves
    # the children behind and the file unparseable. Refuse; the config
    # ships every mapping in one-line flow form for this reason
    indent = len(line) - len(line.lstrip(" "))
    nxt = lines[i + 1] if i + 1 < len(lines) else ""
    if not tail.split("#")[0].strip() and nxt.strip() \
            and len(nxt) - len(nxt.lstrip(" ")) > indent:
        raise RuntimeError(
            f"{'.'.join(path)} is a block mapping in config.yaml -- refusing "
            "to paste over it; write it as a one-line {...} mapping first")
    comment = ""
    if "#" in tail:
        comment = "  " + tail[tail.index("#"):].rstrip("\n")
    if isinstance(value, dict):
        # a per-category mapping pastes as a one-line YAML flow mapping, so
        # the anchor still owns exactly one line
        value = "{" + ", ".join(f'"{str(k).replace(chr(34), "")}": {v}'
                                for k, v in value.items()) + "}"
    lines[i] = f"{head}: {value}{comment}\n"
    return "".join(lines)


# ---------------------------------------------------------- report vintages

def report_staleness(cfg, bundle, reports):
    """What each report on disk is evidence about, judged once for both
    drivers: `ops.advance` (which re-runs) and `ops.status` (which prints
    the `report vintages` line). Returns {name: vintage} over the reports
    present, where a vintage carries the facts and each caller keeps its
    own precedence and wording:

      bundle_mismatch   the report names a bundle that is not `bundle`
      posterior_moved   shadow only: it priced from a posterior file whose
                        digest is no longer on disk (a re-init, a rho paste
                        after a retrain) -- no config key says so
      fingerprint       the report carries a `config` block at all
      snapshot          ... and the snapshot the diff is taken against
      routed            a config move can re-grade it (ROUTED_REPORTS);
                        a production report reads the live config every run
      phase, digest     what the block says it ran under
      moved             the "key: before -> after" lines of the config diff
                        this report READS (stale_keys), in diff order
      keys, rerun       those keys alone, and the re-run they demand

    A MEASURED paste that only writes back what a report measured moves no
    key the report reads, so it invalidates nothing."""
    posterior_now = {}                      # read once, and only if a report asks

    def posterior_digest():
        if "digest" not in posterior_now:
            path = cfg["posterior"]["path"]
            posterior_now["digest"] = (provenance.file_digest(path)
                                       if os.path.exists(path) else None)
        return posterior_now["digest"]

    out = {}
    live = provenance.config_fingerprint(cfg, phase=None)["digest"]
    for name, rep in reports.items():
        if not rep:
            continue
        av = rep.get("artifact_versions") or {}
        fp = rep.get("config") or {}
        v = {"bundle": av.get("baseline_model_version"),
             "bundle_mismatch": bool(bundle) and av.get("baseline_model_version") not in (None, bundle),
             "posterior_moved": bool(name == "shadow" and av.get("posterior_digest")
                                     and av["posterior_digest"] != posterior_digest()),
             "fingerprint": bool(rep.get("config")),
             "snapshot": bool(fp.get("snapshot")),
             "routed": name in ROUTED_REPORTS,
             "phase": fp.get("phase"), "digest": fp.get("digest"),
             "moved": [], "keys": [], "rerun": None}
        # the diff walks the whole snapshot: taken only when the report's
        # digest is not the live one (a current report has nothing to diff)
        if v["fingerprint"] and fp.get("digest") != live:
            diff = provenance.config_diff(fp.get("snapshot") or {}, cfg)
            keys = stale_keys(name, [d.split(":")[0] for d in diff])
            mine = set(keys)
            v["moved"] = [d for d in diff if d.split(":")[0] in mine]
            v["keys"] = keys
            v["rerun"] = rerun_for(keys) if keys else None
        out[name] = v
    return out


# ------------------------------------------------------------ the tau paste

def tau_provenance_error(cfg, backtest, shadow=None):
    """Why the pasted `exploration.tau_initial` cannot be trusted, or None.

    A PROVENANCE check, not a precision one: the paste must agree (within
    `tau_paste_tolerance_rel`) with shadow's own derivation, or -- only when
    no shadow derivation exists -- with a backtest derivation that carries
    `spread_decisions`. `backtest`/`shadow` are the loaded reports, or None.
    """
    tau = cfg["exploration"]["tau_initial"]
    if tau is None:
        return None                 # null is a separate, louder failure
    tol = float(cfg["exploration"]["tau_paste_tolerance_rel"])

    def agrees(derived):
        return abs(float(derived) - float(tau)) <= tol * abs(float(derived))

    sh = (shadow or {}).get("tau_initial_derivation") or {}
    if sh.get("tau_initial") is not None:
        if agrees(sh["tau_initial"]):
            return None             # sourced from the anchored-path derivation
        return (f"exploration.tau_initial is {tau} but the shadow run derived "
                f"{sh['tau_initial']} on its own anchored path. Re-paste from "
                "reports/shadow.json -> tau_initial_derivation.tau_initial, "
                "or re-run shadow if the paste is the newer of the two.")
    der = (backtest or {}).get("tau_initial_derivation") or {}
    if not der:
        return (f"exploration.tau_initial is {tau} but no backtest derivation "
                "or shadow derivation is on disk to source it from. Run "
                "`python3 -m evaluate.shadow` (preferred: it derives tau on "
                "the anchored path) and paste "
                "tau_initial_derivation.tau_initial.")
    if "spread_decisions" not in der:
        return (f"exploration.tau_initial ({tau}) came from a backtest that "
                "predates the entry-only scoping fix: it solved tau on ENTRY "
                "decisions only, funding ~1 exploration per episode against a "
                "system that explores every hour. Re-run `python3 -m evaluate.backtest` "
                "and re-paste.")
    derived = float(der["tau_initial"])
    if not agrees(derived):
        return (f"exploration.tau_initial is {tau} but the backtest derived "
                f"{derived}. Re-paste, or re-run the backtest if the paste is "
                "the newer of the two.")
    return None
