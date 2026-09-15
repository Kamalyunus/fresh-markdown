"""evaluate.pilot_grade -- how a simulated pilot is read and graded.

The home of the simulator's readings -- the arm and paired economics
(through the one episode frame, `common.metrics`), the posterior against
the truth it was learning, the agent's level against the world's -- and of
the fixed list of expectations (`EXPECTATIONS`) `grade` scores a run on.
Everything here reads the run's record (the truth frame, the store's
decisions, the lane's mornings) and nothing here drives the shop
(`evaluate.pilot_shop`). Every number is about the WORLD that was
simulated (rule 19): a PASS says the machinery does what it claims on a
shop with that elasticity, never that the shop has it.
"""

import numpy as np
import pandas as pd

from common import episodes, metrics
from common import guardrail as guard
from engine.posterior import PosteriorStore

# the simulator's own grading and driving knobs (pilot_sim.yaml `grading`;
# no flag: they shape how a run is read, not what it rehearses)
GRADING_KEYS = ("spend_over_budget_band", "tau_week_days", "starve_days",
                "lane_hour", "feature_history_margin_days")

# what a healthy run shows, each graded in `grade()`; the fault that turns
# an expectation around is named so a fault run reads PASS when it fires
EXPECTATIONS = (
    ("hourly_engine", "every hour with stock is priced and stored; no state rejected, no event quarantined"),
    ("price_monotone_within_episode", "no applied price rises within an episode"),
    ("never_below_cost", "no applied price under cost"),
    ("outcome_completeness", "outcomes land for >= shadow_gate.min_event_completeness of decisions (faults: missing, duplicate -- a duplicated hour matches neither row)"),
    ("event_quality_gates", "each event-quality gate passes every day unless a fault's rate exceeds its threshold (price_mismatch_rate: mismatch, discount_rounding; duplicate_or_unmatched_rate: no sim fault reaches it)"),
    ("learning_moves_toward_truth", "every cell that updated ends closer to epsilon_true than it launched"),
    ("posterior_narrows", "every cell that updated ends with a smaller std"),
    ("tau_walks_on_spend", "tau moved and the last week's spend sits within grading.spend_over_budget_band of its budget"),
    ("stops_only_on_faults", "no stop condition fires without the fault that causes it; with it, the stop fires"),
    ("exploration_never_starves", "no grading.starve_days consecutive days with a budget in force and nothing forced (an empty affordable set is exploration off without a stop)"),
    ("agent_level_tracks_world", "every week's mean log(agent mu_ref / world mu_ref) over the pilot's hours sits inside the calibration gate band AND the elasticity bias it implies (level error / mean forced move) inside the posterior's std (the learner has no level term and reads a level error as elasticity)"),
    ("assurance_holds", "reproduction and exploration never FAIL; dispersion never FAILs with the world's marginal untouched (correlation is reported: the world's rho is a knob)"),
    ("lane_c_keeps_the_schedule_current", "the weekly re-fit reaches every week priced, so --apply is never refused on calibration_schedule_current"),
    ("apply_ran_on_cadence", "--apply ran every learning.update_cadence_days and was refused only under a fault"),
)

# the event-quality gates by name and the sim faults whose RATE reaches
# each: a mismatch lands on the compared pair; the rounding fault moves
# every tier off the grid. `missing` and `duplicate` drop the hour's
# OUTCOME (ingest matches neither row of a duplicated hour), which is a
# completeness gap -- the unmatched/duplicate gate counts outcomes without
# a decision and duplicate ids, which no sim fault produces
EVENT_GATE_FAULTS = {"duplicate_or_unmatched_rate": (),
                     "price_mismatch_rate": ("mismatch",)}
COMPLETENESS_FAULTS = ("missing", "duplicate")


# ---------------------------------------------------------------- readers

# the precision each arm figure is reported at (common.metrics.summary
# leaves rounding to its readers)
_ARM_ROUNDING = {"il_absolute": 1, "il_pct": 6, "il_pct_denominator": 1,
                 "scrap_rate": 4, "sell_through": 4, "margin": 1,
                 "mean_discount": 4}


def _arm_economics(hours, ep, excluded):
    """One arm's figures from its hourly frame and its SETTLED episode
    frame (common.metrics.summary, the block the monitor reads too); the
    mean discount is over the settled episodes' hours only."""
    s = metrics.summary(ep, hours, rounding=_ARM_ROUNDING, discount_col="shelf_discount")
    return {
        "episodes": s["episodes"], "hours": s["hours"],
        "il_absolute": s["il_absolute"],
        "il_pct": s["il_pct"],
        "il_pct_denominator": s["il_pct_denominator"],
        "scrap_units": s["scrap_units"],
        "scrap_rate": s["scrap_rate"],
        "sell_through": s["sell_through"],
        "margin": s["margin"],
        "mean_discount": s["mean_discount"],
        "excluded": excluded,
    }


def economics(truth):
    """Both arms through the one episode frame (metrics.episode_economics
    over metrics.settled): per arm over everything settled, and PAIRED
    over the templates settled under both arms (a twin postponed past
    the run's end, or a template re-picked, leaves the arms' template
    sets unequal -- `unpaired_templates` counts them). `truth` is the
    run's hourly truth frame (evaluate.pilot_shop.TRUTH_COLS)."""
    df = truth
    settled = {}
    for arm, g in df.groupby("arm"):
        ep, excluded = metrics.settled(metrics.episode_economics(g))
        settled[arm] = (g, ep, excluded)
    out = {arm: _arm_economics(g, ep, excluded)
           for arm, (g, ep, excluded) in settled.items()}
    template_of = df.drop_duplicates("episode_id").set_index("episode_id").template_id
    by_arm = {arm: set(template_of.reindex(ep.index)) for arm, (_, ep, _) in settled.items()}
    both = set.intersection(*by_arm.values()) if len(by_arm) == 2 else set()
    paired = {"templates": len(both),
              "unpaired_templates": len(set.union(*by_arm.values()) - both)
              if by_arm else 0}
    for arm, (g, ep, _) in settled.items():
        mine = ep[template_of.reindex(ep.index).isin(both).to_numpy()]
        paired[arm] = _arm_economics(g, mine, {})
    out["paired"] = paired
    return out


def learning(cells, cell_of, launch_cells, epsilon_true):
    """Every posterior cell against the truth it was learning: `cells` and
    `cell_of` are the posterior's state, `launch_cells` the cells as the
    run launched, `epsilon_true` the world's {category: epsilon}."""
    truth = epsilon_true
    out = {}
    for c, rec in cells.items():
        members = [cat for cat, cell in cell_of.items() if cell == c] or list(truth)
        known = [truth[m] for m in members if m in truth]
        # a cell none of whose categories the world simulates has no
        # truth to grade against: reported, never averaged over nothing
        eps = float(np.mean(known)) if known else None
        launch = launch_cells[c]
        out[c] = {"members": members,
                  "epsilon_true": round(eps, 4) if eps is not None else None,
                  "launch_mean": launch["mean"], "launch_std": launch["std"],
                  "mean": rec["mean"], "std": rec["std"], "n_obs": rec["n_obs"],
                  "version": rec["version"],
                  "abs_error_at_launch": (round(abs(launch["mean"] - eps), 4)
                                          if eps is not None else None),
                  "abs_error_now": (round(abs(rec["mean"] - eps), 4)
                                    if eps is not None else None),
                  "accumulated_information": round(rec["accumulated_information"], 3)}
    return out


def level_tracking(decisions, truth):
    """Per ISO week, the mean log ratio of the agent's mu_ref to the
    world's over the pilot's priced hours: 0 when the weekly re-fit
    reproduces the world's level, off by the re-fit's error otherwise.
    The elasticity learner reads every outcome against the agent's
    mu_ref and carries no level term, so a level error this size is
    read as elasticity -- the diagnostic that tells a learning FAIL
    from a re-fit artefact. `decisions` is the store's list, loaded
    once by the caller; `truth` the run's hourly truth frame."""
    df = truth[truth.arm == "pilot"]
    if df.empty:
        return {}
    df = df.assign(log_ratio=np.log(df.mu_ref_agent.astype(float)
                                    / df.mu_ref_world.astype(float)),
                   week=episodes.week_key(df.date))
    # the lever the learner identifies elasticity with: the forced
    # moves' SIGNED log price ratio, log((1 - applied) / (1 - reference))
    # -- negative for a deeper move. A level error of e read against
    # moves of mean L is an elasticity error of about -e / L (e > 0
    # and L < 0 bias the belief toward zero); small moves make the
    # learner hypersensitive to the level
    forced = pd.DataFrame([(d["date"], d["reference_discount"], d["applied_discount"])
                           for d in decisions if d["is_exploration"]],
                          columns=["date", "reference_discount", "applied_discount"])
    if len(forced):
        forced["move"] = np.log((1.0 - forced.applied_discount.astype(float))
                                / (1.0 - forced.reference_discount.astype(float)))
        moves = forced.groupby(episodes.week_key(forced.date)).move.mean()
    else:
        moves = pd.Series(dtype=float)
    out = {}
    for wk, g in df.groupby("week"):
        e = float(g.log_ratio.mean())
        move = float(moves.get(wk, np.nan))
        out[wk] = {"hours": int(len(g)),
                   "mean_log_ratio": round(e, 4),
                   "p10_p90": [round(float(g.log_ratio.quantile(q)), 4)
                               for q in (0.1, 0.9)],
                   "mean_forced_log_move": round(move, 4) if np.isfinite(move) else None,
                   "implied_elasticity_bias": (round(-e / move, 3)
                                               if np.isfinite(move) and move != 0
                                               else None)}
    return out


# ------------------------------------------------------------------ grading

def _verdict(ok, measured=True):
    if not measured:
        return "NOT MEASURED"
    return "PASS" if ok else "FAIL"


def _fault_rate(faults, *names):
    """The summed rate of the named faults in force (a rate-less fault --
    `discount_rounding`, `demand_shock` -- contributes nothing here)."""
    return sum(float(faults[n]) for n in names
               if n in faults and isinstance(faults[n], (int, float))
               and not isinstance(faults[n], bool))


def expected_gate_failures(faults, cfg):
    """{gate name: should it fail}: an event-quality gate is expected to
    fail only when the rate of the faults that reach it exceeds ITS
    threshold (`EVENT_GATE_FAULTS`); `discount_rounding` always moves the
    price-mismatch gate (every tier off the grid)."""
    sc = cfg["monitoring"]["stop_conditions"]
    out = {name: _fault_rate(faults, *fs) > sc[name]
           for name, fs in EVENT_GATE_FAULTS.items()}
    out["price_mismatch_rate"] |= bool(faults.get("discount_rounding"))
    return out


def grade(rep, cfg, sim_settings):
    """EXPECTATIONS against the run. A fault that is present turns its
    expectation around: the gate/stop it targets must fire. `cfg` is the
    system's config (its thresholds are what the lane compared against);
    `sim_settings` carries the simulator's own grading knobs
    (pilot_sim.yaml `grading`)."""
    faults = rep["world"]["faults"]
    days = rep["days"]
    eng = rep["engine"]
    gr = sim_settings
    out = []

    def add(name, ok, observed, measured=True):
        out.append({"name": name, "expected": dict(EXPECTATIONS)[name],
                    "verdict": _verdict(ok, measured), "observed": observed})

    # every pilot hour is a decision, a rejection or a quarantined event --
    # on BOTH sides of the store: a decision it refused, and an outcome
    # the ingester built that it refused (a contract defect the shadow
    # gate would read as incompleteness must read the same here)
    ing = [d["ingest"] for d in days]
    quarantined = int(eng.get("quarantined", 0))
    outcomes_quarantined = sum(int(i.get("quarantined", 0)) for i in ing)
    accounted = eng["decisions"] + eng["rejected_total"] + quarantined
    add("hourly_engine", eng["decisions"] > 0 and eng["rejected_total"] == 0
        and quarantined == 0 and outcomes_quarantined == 0
        and eng["pilot_hours"] == accounted,
        {"decisions": eng["decisions"], "rejected": eng["rejected"],
         "quarantined": quarantined, "outcomes_quarantined": outcomes_quarantined,
         "pilot_hours": eng["pilot_hours"]})
    add("price_monotone_within_episode", eng["violations"]["price_rose_within_episode"] == 0,
        eng["violations"])
    add("never_below_cost", eng["violations"]["below_cost"] == 0, eng["violations"])

    # completeness on the shadow gate's population: outcomes the store
    # ACCEPTED per decision the feed could answer for -- whatever kept an
    # outcome from landing (no feed row, two decisions claiming one hour,
    # an unusable row, a quarantined outcome) is a gap, and the ingester's
    # counts are summed generically so a newly named cause reaches the
    # report without a list here to extend
    due = sum(i["decisions"] - i["decisions_outside_feed_range"] for i in ing)
    landed = sum(i["emitted"] for i in ing)
    counts = {}
    for i in ing:
        for key, v in i.items():
            if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
                counts[key] = counts.get(key, 0) + int(v)
    completeness = landed / due if due else None
    floor = cfg["monitoring"]["shadow_gate"]["min_event_completeness"]
    # a missing row and a duplicated hour (ingest matches neither state)
    # both cost the decision its outcome: a gap is expected once their
    # summed rate exceeds what the floor admits
    gap_rate = _fault_rate(faults, *COMPLETENESS_FAULTS)
    expect_gap = gap_rate > 1 - floor
    ok = completeness is not None and ((completeness >= floor) != expect_gap)
    add("outcome_completeness", ok,
        {"completeness": round(completeness, 4) if completeness is not None else None,
         "floor": floor, "fault_expects_a_gap": expect_gap,
         "fault_gap_rate": round(gap_rate, 4), "decisions_due": due,
         "outcomes_landed": landed, "ingest_counts": counts},
        measured=completeness is not None)

    # per gate, by name: the calibration gate is graded by
    # lane_c_keeps_the_schedule_current, not here
    expect_by_gate = expected_gate_failures(faults, cfg)
    failed_by_gate = {name: [d["date"] for d in days if d["gates"].get(name) is False]
                      for name in expect_by_gate}
    gates_off = [name for name, exp in expect_by_gate.items()
                 if bool(failed_by_gate[name]) != exp]
    expect_fail = any(expect_by_gate.values())
    add("event_quality_gates", not gates_off,
        {"days_a_gate_failed": failed_by_gate, "fault_expects_a_failure": expect_by_gate,
         "gates_off_expectation": gates_off},
        measured=bool(days))

    learned = {c: r for c, r in rep["learning"].items()
               if r["version"] > 0 and r["epsilon_true"] is not None}
    add("learning_moves_toward_truth",
        all(r["abs_error_now"] < r["abs_error_at_launch"] for r in learned.values()),
        {c: {"launch": r["abs_error_at_launch"], "now": r["abs_error_now"]}
         for c, r in rep["learning"].items()}, measured=bool(learned))
    add("posterior_narrows", all(r["std"] < r["launch_std"] for r in learned.values()),
        {c: {"launch_std": r["launch_std"], "std": r["std"]}
         for c, r in rep["learning"].items()}, measured=bool(learned))

    # the level error is graded by what it does to the learner: the
    # elasticity bias it implies (level error / mean forced move) must stay
    # inside the posterior's own uncertainty at that week's end, or the
    # re-fit is steering the belief. The gate band bounds the level itself
    band = cfg["baseline_model"]["calibration_gate_band"]
    tol = max(abs(np.log(band[0])), abs(np.log(band[1])))
    level = rep.get("level_tracking") or {}
    std_by_week = {}
    if days:
        weeks = episodes.week_key(pd.Series([d["date"] for d in days]))
        for d, wk in zip(days, weeks):
            # the widest std among the cells a category REACHES (the one
            # reading the budget and the flat-std alert take): an
            # unrouted GLOBAL keeps its launch std and would excuse any bias
            std_by_week[wk] = PosteriorStore.widest_active_std(
                d["posterior"], d.get("cell_of") or {})
    off = {}
    for wk, v in level.items():
        bias, std = v.get("implied_elasticity_bias"), std_by_week.get(wk)
        if abs(v["mean_log_ratio"]) > tol or (
                bias is not None and std is not None and abs(bias) > std):
            off[wk] = {"mean_log_ratio": v["mean_log_ratio"],
                       "implied_elasticity_bias": bias, "posterior_std": std}
    add("agent_level_tracks_world", not off,
        {"weeks_off": off, "band_tolerance_log": round(float(tol), 4),
         "by_week": {wk: {"mean_log_ratio": v["mean_log_ratio"],
                          "implied_elasticity_bias": v.get("implied_elasticity_bias")}
                     for wk, v in level.items()}},
        measured=bool(level))

    # the controller moves tau by at most the clip per day, so a launch tau
    # far from this world's budget needs a week or more to arrive: graded
    # on the last `tau_week_days` days the controller actually MOVED on
    # (held days -- no base yet -- are not walks), once there are that
    # many, against the sim's own spend-over-budget band (the clip bounds
    # the daily step, not the ratio)
    walked = {}
    for d in days:
        if d["tau"].get("committed"):
            for w in d["tau"].get("walked") or []:
                walked[w["day"]] = w                  # a day walked once
    walked = [walked[day] for day in sorted(walked)]
    live = [w for w in walked if not w.get("held")]
    week_days = int(gr["tau_week_days"])
    lo, hi = gr["spend_over_budget_band"]
    week = live[-week_days:]
    ratios = [w["spend"] / w["budget"] for w in week if w["budget"] > 0]
    ratio = float(np.mean(ratios)) if ratios else None
    moved = eng["tau_now"] != eng["tau_at_launch"]
    held = [w["day"] for w in walked if w.get("held")]
    suspended = [d["date"] for d in days if d.get("suspended")]
    add("tau_walks_on_spend", moved and ratio is not None and lo <= ratio <= hi,
        {"tau_at_launch": eng["tau_at_launch"], "tau_now": eng["tau_now"],
         "last_week_spend_over_budget": round(ratio, 3) if ratio is not None else None,
         "band": [lo, hi], "week_days": week_days, "days_walked": len(live),
         "days_held": held, "days_exploration_suspended": suspended},
        measured=len(live) >= week_days)

    fired = {}
    for d in days:
        for name, v in d["stops"].items():
            if v is True:
                fired.setdefault(name, []).append(d["date"])
    # the stop behind each event gate fires on the gate's own rate: the
    # same per-gate expectation
    stop_of_gate = {"duplicate_or_unmatched_rate": "duplicate_or_unmatched",
                    "price_mismatch_rate": "price_mismatch"}
    causes = {stop_of_gate[g]: EVENT_GATE_FAULTS[g] + (("discount_rounding",)
                                                       if g == "price_mismatch_rate" else ())
              for g in stop_of_gate}
    causes.update({"scrap_deterioration_pct": ("demand_shock",),
                   "margin_deterioration_pct": ("demand_shock",)})
    unexpected = [n for n in fired if not any(f in faults for f in causes.get(n, ()))]
    # the scrap guardrail can only have fired once its series (which
    # starts at launch; one close day per lane morning) holds
    # persistence_days readings -- common.guardrail.stop_ready_close_days,
    # the series' own arithmetic, never a second count of it -- and the
    # shock has filled that many fully-shocked smoothed readings, counted
    # from the first close day wholly after the shock day
    mc, sc = cfg["monitoring"], cfg["monitoring"]["stop_conditions"]
    shock_day = faults["demand_shock"][0] if "demand_shock" in faults else None
    smooth, persist = sc["deterioration_smoothing_days"]["scrap"], sc["persistence_days"]
    ready_days = guard.stop_ready_close_days(smooth, mc["guardrail_noise_window_days"], persist)
    guardrail_ready = (shock_day is not None
                       and len(days) >= ready_days
                       and len(days) >= shock_day + 1
                       + guard.change_visible_close_days(smooth, persist))
    expected_missing = [stop_of_gate[g] for g, exp in expect_by_gate.items()
                        if exp and stop_of_gate[g] not in fired]
    if "demand_shock" in faults and guardrail_ready \
            and "scrap_deterioration_pct" not in fired:
        expected_missing.append("scrap_deterioration_pct")
    # a shock the series SAW (the scrap deviation moved worse) that still
    # sits under the owner's floor is this world's reach, not a silent
    # stop: reported with the reading, not graded
    scrap = (days[-1].get("guardrails") or {}).get("scrap_deterioration_pct") or {} if days else {}
    under_floor = ("scrap_deterioration_pct" in expected_missing
                   and scrap.get("latest") is not None and scrap.get("threshold") is not None
                   and 0 < scrap["latest"] < scrap["threshold"])
    if under_floor:
        expected_missing.remove("scrap_deterioration_pct")
    add("stops_only_on_faults", not unexpected and not expected_missing,
        {"fired": fired, "unexpected": unexpected, "expected_but_silent": expected_missing,
         "guardrail_window_reached": guardrail_ready if shock_day is not None else None,
         "guardrail_ready_after_close_days": ready_days,
         "scrap_deviation_latest": scrap.get("latest"), "scrap_floor": scrap.get("threshold"),
         "shock_seen_but_under_the_floor": under_floor},
        measured=bool(days) and (shock_day is None or guardrail_ready or bool(unexpected))
        and not under_floor)

    # forced per day from the monitor's cumulative count; a day with a tau
    # in force (not suspended) and no forced decision is exploration off
    starve_days = int(gr["starve_days"])
    starved, streak, worst = [], 0, 0
    prev = 0
    for d in days:
        forced_today = d["learning"]["forced_decision_count"] - prev
        prev = d["learning"]["forced_decision_count"]
        streak = streak + 1 if forced_today == 0 and not d["suspended"] else 0
        worst = max(worst, streak)
        if streak >= starve_days:
            starved.append(d["date"])
    add("exploration_never_starves", not starved,
        {"days_starved": starved, "longest_streak": worst, "starve_days": starve_days,
         "affordable_set_empty_rate_latest": days[-1]["learning"]
         ["affordable_set_empty_rate"] if days else None}, measured=bool(days))

    verdicts = {n: sorted({d["assurance"][n] for d in days}) for n in
                ("reproduction", "dispersion", "correlation", "exploration")} if days else {}
    # correlation grades the world's within-episode shock against the
    # frozen rho -- a knob of the world (`episode_shock_sd`), not a claim
    # about the machinery: reported with the live reading, never graded
    # a world whose hours share a shock is over-dispersed against an r
    # fitted without one: dispersion is then a reading of the knob too
    # ... and the check bins by PREDICTED mu, so while a gate fault keeps
    # --apply refused the belief never converges and a wrong elasticity
    # reads as a shape problem: dispersion is excused while any event
    # gate is expected to fail
    marginal_moved = (rep["world"]["r_scale"] != 1.0
                      or rep["world"].get("episode_shock_sd", 0) > 0
                      or expect_fail)
    bad = [n for n, vs in verdicts.items() if "FAIL" in vs and n != "correlation"
           and not (n == "dispersion" and marginal_moved)]
    rho = [d["assurance_detail"]["rho_live"] for d in days
           if d["assurance_detail"].get("rho_live") is not None]
    add("assurance_holds", not bad,
        {"verdicts": verdicts, "failing": bad,
         "rho_live_latest": rho[-1] if rho else None,
         "rho_frozen": cfg["dispersion"]["rho"]}, measured=bool(days))

    applies = [d["apply"] for d in days if "apply" in d]
    stale = [d["date"] for d in days if not d["calibration_current"]["pass"]]
    held = [d["date"] for d in days if d["calibration_current"].get("held_at_anchor")]
    add("lane_c_keeps_the_schedule_current", not stale,
        {"re_fits": len(rep["lane_c"]), "mornings_schedule_stale": stale,
         "mornings_priced_on_the_held_anchor": held,
         "schedule_reaches": [r["schedule_end"] for r in rep["lane_c"]]},
        measured=bool(days))

    cadence = int(cfg["learning"]["update_cadence_days"])
    expected_applies = sum(1 for d in days if d["day"] % cadence == 0)
    # a refusal is the fault's doing only when EVERY gate that failed that
    # morning is one a fault in force is expected to fail (the same
    # per-gate expectation the gates and the stops are graded on); a
    # refusal on the calibration gate, or on a gate no fault reaches, is a
    # lane that does not apply. And an apply that neither applied nor
    # refused is silent, whatever the cadence count says
    excused = {g for g, exp in expect_by_gate.items() if exp}
    refused, unexplained, silent = [], [], []
    for d in days:
        app = d.get("apply")
        if not app:
            continue
        if app.get("refused"):
            refused.append(d["date"])
            failing = [g for g, ok in d["gates"].items() if ok is False]
            if not failing or not set(failing) <= excused:
                unexplained.append({"date": d["date"], "gates_failed": failing,
                                    "refused": app["refused"]})
        elif not app.get("applied"):
            silent.append(d["date"])
    add("apply_ran_on_cadence", len(applies) == expected_applies
        and not unexplained and not silent,
        {"applies": len(applies), "expected": expected_applies, "refused_on": refused,
         "refusals_no_fault_explains": unexplained, "neither_applied_nor_refused": silent,
         "gates_a_fault_excuses": sorted(excused)}, measured=bool(days))
    return out
