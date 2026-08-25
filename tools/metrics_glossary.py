"""tools.metrics_glossary -- every measured quantity, and the component it sits in.

    python3 -m tools.metrics_glossary        # writes docs/metrics.html

The four reports carry ~725 fields between them. `pipeline.status` prints the
ten that gate a decision, which is the right tier-one view and deliberately
not a reference. This is the reference: what a quantity means, what unit it is
in, which component writes it, and whether anything downstream reads it.

The catalogue below is data, not prose -- short strings in a table -- so it
lives in Python rather than in partial files. What keeps it honest is
`tests/test_metrics_glossary.py`, which cross-checks the entries that HAVE a
machine-readable source of truth: the two event schemas, the artifact list,
and the status gate names. Those three drift silently; the rest is prose that
a human must keep true, and the entries say so by living in one place.

Units are stated because getting one wrong has already cost this project a
launch value: `tau` is a CURRENCY amount and phase 0 first reported a rate for
it, which is dimensionally meaningless against `Q(p*) - Q(p)`.
"""

import html
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "metrics.html"

# unit -> how it is displayed. Kept short: the badge is scanned, not read.
UNITS = {
    "won": "₩", "rate": "0–1", "pct": "%", "count": "n", "ratio": "×",
    "pp": "pp", "days": "d", "hours": "h", "secs": "s", "exp": "ε",
    "id": "id", "verdict": "✓/✗", "text": "…",
}

# (name, unit, meaning, read_by)  -- read_by "" means diagnostic only.
# GATE marks a quantity that blocks or suspends something.
CATALOGUE = [

("Population", "bootstrap.prepare_data → artifacts/split_manifest.json",
 "The rows every other number is measured on. Frozen at launch.", [
  ("data_quality_waterfall", "count",
   "Rows, episodes and COGS at risk after each of the 13 waterfall rows, in order. "
   "The first entry is `raw`. Every row carries `kind` and `used_by`: `hard_drop` "
   "rows leave the frame and are gone for every consumer, while the last two are "
   "`population_gate` rows that drop nothing and say WHO reads them -- `eligible` "
   "for the demand model and every scrap/IL figure, `dp_eligible` for the solver, "
   "backtest and A/B. The three populations are NESTED, so their exclusions must "
   "never be added together.", "every figure traces back through it"),
  ("cogs_at_risk", "won",
   "Unit cost × SUPPLY -- opening stock plus gross arrivals -- ONCE per episode, "
   "reported at every waterfall stage "
   "with `cogs_dropped` and `cogs_dropped_pct_of_raw`. Never summed over hours — "
   "inventory persists, so a per-row sum multiplies the same stock by the window "
   "length. Rows and money diverge, and that is the point: a filter taking 1% of rows "
   "and 15% of the exposure has changed what the population represents. Negative "
   "`cogs_dropped` appears at exactly one stage, `contiguous_episodes_built`, where "
   "re-segmentation turns one opening row into two.",
   "whether the surviving population still looks like the business"),
  ("unreconciled_anomalies", "count",
   "Episodes with SHRINK -- stock left the shelf that no sale or write-off accounts "
   "for. Kept and dp_eligible: the shrink settles into scrap at the episode level "
   "(scrap = leftover + shrink), so the identity still closes and the units stay in "
   "every scrap/IL figure. The block breaks them out by category and month: one "
   "incident, one corner of the catalogue, or a standing feed property are three "
   "different investigations.", "a report for the business, not a gate"),
  ("units_restocked / episode_supply", "count",
   "Units that ARRIVED mid-window, and the resulting supply = opening + arrivals. "
   "Opening stock stopped being an episode's supply once restocked episodes were "
   "kept, and clearance against it read 300% on a window that scrapped 4 units. "
   "Counted GROSS: an episode that takes 2 units in and loses 2 to shrink has 2 of "
   "each, not zero of both. `shrink_and_restock_together` reports the ones carrying "
   "both.",
   "clearance, and the DP state the solver cannot model"),
  ("dp_eligible · edge_truncated", "count",
   "Episodes the extract cut off mid-window, FLAGGED not dropped: only the ENDING is "
   "unknown, the observed hours are ordinary priced demand, and these are the LARGEST "
   "episodes in the extract (~24.9 units of opening stock against ~3.05). They stay "
   "dp_eligible, because every consumer of an outcome already excludes an unclosed "
   "one -- scrap_units returns NaN, replay zeroes scrap under outcome_known, shadow "
   "charges scrap only on COMPLETED. "
   "`share_of_unclosed_explained_by_edge` near 1.0 means the whole unknown-scrap "
   "problem was the extract boundary; `episodes_unclosed_not_edge` is the residue a "
   "longer extract will NOT fix.", "every scrap and IL figure"),
  ("episode_rule", "text",
   "The persisted definition of an episode: a maximal run of consecutive hourly "
   "rows for one SKU × FC over which the source hours-remaining counter "
   "decrements by one per elapsed hour. NOT keyed by date.",
   "production must derive identical boundaries"),
  ("split.train / calib / test", "text",
   "Date windows, fixed before any model was fit. An episode belongs wholly to "
   "the split its window STARTED in.", "the level diagnostic, which must not grade its own fit"),
 ]),

("Phase 0 — is this problem tractable?", "bootstrap.measure → reports/phase0.json",
 "Measured once on raw history, before anything is modelled. Each `m` is a "
 "specified deliverable (design 5.3), so none can be dropped.", [
  ("m1 · cost_ratio_pct", "pct",
   "Unit cost as a share of full price. Sets how much discounting room exists at all.", ""),
  ("m1 · d_max_pct", "pct",
   "The feasible discount ceiling, `1 − cost/price`, in percent. The cost floor "
   "expressed in discount space.", "the action set is built from it"),
  ("m1 · share_non_explorable", "rate",
   "Share of episodes with too few feasible tiers to experiment on at all.",
   "learning throughput"),
  ("m2 · discount_std_pp", "pp",
   "Spread of discounts within one SKU × FC × hour cell. Near zero means the "
   "legacy rule was deterministic and history carries no price variation to learn from.",
   "why elasticity is unidentifiable from history"),
  ("m3 · rho", "rate",
   "Correlation between hours inside one episode, on a category × hour proxy "
   "computed BEFORE any model exists. Diagnostic only — never paste this one.",
   "superseded by artifacts/rho.json"),
  ("m3 · implied_deff", "ratio",
   "The evidence deflator implied by m3's rho. Same caveat.", ""),
  ("m4 · zero_sale_rate", "rate",
   "Share of hours selling nothing. Around three quarters. This is why accuracy "
   "is gated on an aggregate ratio rather than on MAE.", ""),
  ("m5 · overall_censored_hour_rate", "rate",
   "Share of hours where sales hit inventory, so demand is known only as a lower bound.",
   "the censored likelihood everywhere"),
  ("m5 · episodes_reaching_zero_inventory", "rate",
   "Share of episodes that sold out. Their windows are short BECAUSE they sold out, "
   "which is the lookahead bias `extend_to_window` exists to remove.", ""),
  ("m6 · il_pct_aggregate", "rate",
   "Inventory Loss as a ratio of sums: (discount given away + scrap at cost) ÷ "
   "(full-price value of units sold). Never averaged per episode — the denominator "
   "is zero for a zero-sale episode.", "the business baseline"),
  ("m6 · il_absolute_total", "won",
   "The same loss in currency, always reported alongside the ratio.", ""),
  ("m6 · il_pct_ratio_se_clustered", "rate",
   "Standard error of IL%, clustered by SKU × FC. Informational — A/B power is "
   "sized by `derive_thresholds`, which re-measures the SE on actual T-week "
   "blocks rather than scaling this by √T.", ""),
  ("m7 · episodes_per_category_per_week", "count",
   "Volume per category. Below `min_episodes_per_week_for_cell` a category has no "
   "posterior cell of its own and pools into the global one.", "cell assignment"),
  ("m8 · entry_hour", "text",
   "Distribution of the episode's first priced hour. Elasticity identification uses "
   "entry rows ONLY, so this is the population that estimate is drawn from.", ""),
  ("m11 · episode_endings", "count",
   "Splits episodes three ways — `completed` (leftover IS scrap), `sold_out_early` "
   "(no scrap by construction), `not_closed` (scrap UNKNOWN). Unclosed episodes are "
   "KEPT in the population, so `not_closed` counts BOTH kinds: the ones the extract "
   "boundary cut off and the ones unclosed for a reason a longer extract will not "
   "fix. Read it beside `dp_eligible.edge_truncated."
   "share_of_unclosed_explained_by_edge`, which is the split.",
   "every scrap and IL figure"),
  ("m11 · not_closed_by_month / by_category", "count",
   "Where the residual unknown scrap sits. Concentrated in one month reads as an "
   "incident; spread evenly across every month reads as a standing property of the "
   "feed; concentrated in a few categories names the subset. This is what tells you "
   "whether a longer extract will help.", "the data-quality decision"),
  ("m11 · share_last_row_counter_at_zero", "rate",
   "How often `hours_remaining` actually reaches 0 on a final row. Measured at 0.52% "
   "— the counter is NOMINAL, so any rule keyed to it is measuring a rounding error. "
   "This is the tripwire for the bug that once excluded ~99% of real leftover.", ""),
 ]),

("Demand model", "bootstrap.train_baseline → artifacts/baseline_model.txt, calibration.json",
 "The frozen forecaster. It answers exactly one question and is graded on exactly that.", [
  ("mu_ref", "count",
   "Expected units next hour AT THE REFERENCE DISCOUNT. The price feature is "
   "overwritten to `d_ref` at inference, always, so the model never expresses a "
   "price effect — that is ε's job.", "the DP, every hour"),
  ("level factor", "ratio",
   "One multiplicative correction per subcategory, blended toward its parent "
   "category with a pseudo-count of 100 anchor units. SOLVED on the censored "
   "basis, not divided out. ALWAYS applied -- with no calibration artifact "
   "on disk every factor reads 1.0.", "μ_ref at inference, every call"),
  ("anchor row", "text",
   "A row priced within ±1.25pp (half a tier step) of the category reference "
   "discount, so the price-response term is exactly 1. **The velocity features "
   "use a wider ±2.5pp band** — same word, two bands, deliberately.",
   "the calibration solve and the gate"),
  ("sku_ref_sales_rate_30d", "rate",
   "Trailing 30-day anchor-hour sales rate at SKU × FC, falling back to SKU pooled "
   "across FCs. Point-in-time: an episode never sees its own day.", "the model"),
  ("prior_episode_ref_sales_rate", "rate",
   "The same signal from that SKU × FC's previous clearance window. Computed at "
   "episode grain, NaN when that episode had no anchor hours.", "the model"),
 ]),

("Dispersion and correlation", "bootstrap.fit_dispersion → artifacts/r_lookup.json, rho.json",
 "Two numbers that never touch a price. They govern how much a piece of evidence "
 "is worth, and how confident the agent may be.", [
  ("r", "count",
   "Negative-binomial dispersion, `Var[D] = μ + μ²/r`. Small is lumpy, large is "
   "near-Poisson. Fitted per subcategory by maximum likelihood with censored hours "
   "entering as `P(D ≥ q)`. Clamped only on the HIGH side — and groups steadier "
   "than Poisson (Pearson dispersion < 1) are EXEMPT and listed in "
   "`under_dispersed_groups`: no NB can express Var < mean, so their fit rides the "
   "search ceiling by necessity, not optimism.",
   "every probability the DP uses, and every posterior update"),
  ("rho", "rate",
   "Correlation between hours within one episode, measured on the FITTED MODEL'S "
   "residuals at the working elasticity (the per-category prior means). "
   "**0.3103.** A changed prior invalidates it.", "GATE — mirrored into config"),
  ("mean_forced_hours_per_episode", "hours",
   "Average number of hours per episode that carry forced (randomised) prices. "
   "**8.563.** The cluster size in the design effect.", "GATE — mirrored into config"),
  ("deff", "ratio",
   "Design effect `1 + (m − 1)·ρ` = **3.347**. Divides Fisher information before it "
   "is compared to the learning threshold: 8.563 correlated hours carry the "
   "information of about 2.56 independent ones.", "every posterior update"),
 ]),

("Elasticity prior", "bootstrap.estimate_prior → artifacts/prior.json",
 "The profile likelihood read as a density (design 5.6). No fallback constant: "
 "a flat likelihood degrades to the uniform on the support, a wrong-signed one "
 "takes the measured pooled density.", [
  ("mean / std", "exp",
   "Moments of each category's density. `std_basis` names which measured floor "
   "bound the width (density / grid_resolution / fold_spread) — a zero-width "
   "prior would freeze the posterior, so the std can never be zero. ε is always "
   "negative; `epsilon_max` (−0.05) is a sign constraint, never a bound to widen.",
   "the DP until production learns better"),
  ("own_information_weight", "rate",
   "How much of the category's prior is its own data vs the pooled density — "
   "min(1, likelihood span / own_information_saturation).", ""),
  ("wrong_sign_categories", "text",
   "Likelihoods whose UNCONSTRAINED peak (searched past the bounds) sits at or "
   "above zero — demand rising with price. Their own densities are discarded; "
   "they take the pooled one. Usually the ramp confound at full strength.",
   "GATE — read before trusting any category"),
  ("design_comparison", "text",
   "All rows × hour-control combinations scored for sign every run. Fewer "
   "wrong-signed first, then median span; then median_rows_per_time_cell, since "
   "a thin time cell absorbs the price response it should control for.", ""),
  ("holdout_comparison", "ratio",
   "Log marginal predictive on a window the fit never saw, bracketed by `oracle` "
   "and `uniform`. Read information_available_per_row FIRST — a method gap that "
   "is a large share of a tiny number is still tiny. A candidate below `uniform` "
   "is worse than knowing nothing.", "the prior-acceptance gate evidence"),
 ]),

("Replay — fidelity", "backtest → reports/backtest.json → fidelity",
 "Does the frozen model still describe the world? This is the backtest's most "
 "important job, and it is NOT evidence the policy works.", [
  ("fidelity_episode_sold_ratio", "ratio",
   "Actual ÷ predicted sales on the gate window. Above 1 the model under-predicts. "
   "The pooled ratio embeds the unidentifiable prior's slope, so it is reported as "
   "a diagnostic rather than used as the verdict.", ""),
  ("level_bias_at_anchor", "ratio",
   "The same ratio restricted to anchor rows, where the price term is 1. This is "
   "the model's only production job.", "the level diagnostic metric — WARN out of band, not a gate"),
  ("calibration_gate_band", "ratio",
   "**[0.90, 1.10]** — roughly 2σ of measured week-to-week demand volatility. Wide "
   "enough not to fire on noise, tight enough to catch a stale model. A "
   "DIAGNOSTIC band: calibration is always applied, and out of band reads as "
   "drift to investigate, never a launch blocker.", ""),
  ("gate_window", "text",
   "Which split the gate is read on. Must be DISJOINT from the calibration fit "
   "window, or the gate grades its own fit. Read the field; do not assume.", "GATE"),
  ("slope_ratio_by_discount_gap", "ratio",
   "Actual ÷ predicted bucketed by distance from the reference discount. Flat and "
   "offset means LEVEL error (calibration fixes it). Right at the anchor and wrong "
   "either side means SLOPE error (only the prior or learning fixes it).",
   "which remedy the gate decision tree sends you to"),
  ("fidelity_hourly_mae / rmse / bias", "count",
   "Per-hour error in units. On a series that is ~78% zeroes an MAE near the mean is "
   "expected; this is why the gate is on a ratio, not on MAE.", ""),
  ("fidelity_nz_mae / nz_bias", "count",
   "The same errors restricted to hours that actually sold something.", ""),
  ("fidelity_zero_acc", "rate",
   "How often the model calls a zero-sale hour correctly (predicting under half a unit).", ""),
  ("sold_over_censored_prediction", "ratio",
   "Realised sales ÷ `E[min(D, q)]` — the CENSORED basis. Everything that compares "
   "predictions to reality must use this.", "the level factor solve"),
  ("sold_over_raw_mu", "ratio",
   "The same against raw μ. Shown only to make the gap visible: a true correction of "
   "1.45 fits as 0.68 on this basis — the wrong side of 1.", ""),
  ("censoring_shrinkage", "ratio",
   "Censored total ÷ raw μ total. How much the inventory ceiling removes.", ""),
  ("anchor_ratio_by_rate_history", "ratio",
   "The anchor ratio split by whether the SKU had velocity history. `no_history` far "
   "above `with_history` means new assortment, not a macro trend.",
   "distinguishing wobble from trend"),
  ("level_mix_decomposition", "ratio",
   "Is weekly movement in the anchor ratio demand drift, or a change in which SKUs "
   "entered the cohort? Per-SKU ratios held fixed while composition varies.", ""),
  ("calibration_window_sweep", "ratio",
   "Rolling-origin test of the calibration mechanism across candidate trailing "
   "windows. Answers 'how long should the fit window be' with data.", ""),
 ]),

("Replay — policy", "backtest → reports/backtest.json → policy_deltas",
 "What the DP would have done. **Replay is never evidence the policy works** — the "
 "A/B is. Read these as internal consistency only.", [
  ("actual_il / actual_il_pct", "won",
   "Observed-world inventory loss under the legacy prices that really ran.", ""),
  ("legacy_model_il", "won",
   "Legacy prices scored UNDER THE MODEL. The honest comparator.", "the policy verdict"),
  ("dp_il", "won",
   "DP prices scored under the same model.", "the policy verdict"),
  ("policy_gap_like_for_like", "won",
   "`dp_il − legacy_model_il`. Same demand generator both arms, so model bias "
   "cancels. **Never compare `actual_*` against `dp_*`** — that charges all model "
   "bias to the DP.", "the only defensible policy statement in replay"),
  ("actual_clearance / dp_clearance", "rate",
   "Share of stock sold before the window closed, per arm.", ""),
  ("actual_mean_discount / dp_mean_discount", "rate",
   "Average discount applied per arm. The DP opens far shallower and holds.", ""),
  ("step_sensitivity", "won",
   "What one bounded posterior step (`learning.max_mean_step`) is worth on real "
   "episodes: the DP arm re-solved at ε ± step, reporting the share of episodes "
   "whose prices move at all and the IL delta under the same demand model. Below "
   "the deepening bar a step changes nothing — the measured insensitivity that "
   "makes a wrong-direction update cheap; `crossers` isolates the episodes where "
   "the cap is load-bearing.", "the evidence behind the step cap"),
  ("tau_initial", "won",
   "The currency quantile of the `Q(p*) − Q(p)` distribution whose implied daily "
   "spend matches `budget_share_of_il`, on the EXPLOIT-ONLY replay path. **A "
   "CURRENCY AMOUNT, never a rate**, and a CROSS-CHECK: the launch paste comes "
   "from shadow's own anchored-path `tau_initial_derivation`; this block is "
   "accepted as a paste source only while no shadow derivation exists.", ""),
  ("cost_distribution_quantile", "rate",
   "Where `tau_initial` landed in the cost distribution. Sanity check on the solve.", ""),
 ]),

("Thresholds", "bootstrap.derive_thresholds → reports/thresholds.json",
 "Evidence for the values only the owner may set.", [
  ("target_mde_rel", "rate",
   "The relative effect size the A/B must be able to detect.", "GATE — owner-set"),
  ("recommended_duration_weeks", "days",
   "How long the A/B must run to reach that effect size. The duration curve is flat, "
   "so more weeks buy little.", ""),
  ("three_sigma / three_sigma_robust", "rate",
   "Noise floor of a guardrail series — raw and outlier-resistant. A floor above 1.0 "
   "means the series swings by more than its own level, and NO threshold on that "
   "basis is both safe and useful.", "GATE — guardrail floors"),
  ("guardrail_noise (trailing basis)", "rate",
   "Each day against a trailing 28-day mean. Applies only BEFORE an A/B is running.", ""),
  ("guardrail_noise_control_arm_basis", "rate",
   "Same-day treatment vs control, using the identical arm hash the monitor uses and "
   "smoothed the same way. This is what binds once both arms are populated.", ""),
  ("guardrail_threshold_recommendation", "verdict",
   "Reports BOTH floors, names the binding one, and stamps a verdict. `TOO TIGHT` "
   "and `CLEARS THE FLOOR BUT LIKELY INERT` are both blocking, not advisory.",
   "GATE — read by pipeline.status"),
 ]),

("Shadow", "pipeline.shadow → reports/shadow.json",
 "The full decision path against live data with no price applied. The last gate "
 "before anything is priced for real.", [
  ("event_completeness", "rate",
   "Outcomes landing per decision. **≥ 0.99.** Quarantined outcomes do not land, so "
   "a missing `adjustment_reason` shows up here first.", "GATE — shadow exit"),
  ("matched_decision_rate", "rate",
   "Decisions that got an outcome back. **≥ 0.99.**", "GATE — shadow exit"),
  ("cost_floor_violations", "count",
   "Prices below cost. **Must be exactly zero** — the action set makes them "
   "unrepresentable, so any count is a defect, not a tolerance.", "GATE — shadow exit"),
  ("window.sampled / population_episodes", "count",
   "Whether the run sampled and out of how many. **Quote the sampling caveat "
   "whenever you quote the violation count**: the 3,000-episode default once passed "
   "on a sample that hid a crash the full run found.", ""),
  ("state_rejected_count", "count",
   "States refused rather than priced. Refusal is the designed response to an "
   "implausible state, so a non-zero count is information, not failure.", ""),
  ("effective_information_total", "count",
   "Fisher information for ε the run would have bought, after deff deflation — "
   "in NB units, `μ·L²·r/(r+μ)`, the same accumulation `pipeline.update` runs. "
   "Accumulated on EXPLORATION decisions only.", ""),
  ("exploration_budget · implied_daily_spend", "won",
   "What exploration would have spent per day, on SHADOW'S OWN anchored path. "
   "The backtest reports the same pair, but solves on the exploit-only replay "
   "path where each hour is scored independently — the anchored action sets "
   "differ, so the same tau buys a different amount of exploration.",
   "GATE — whether tau is affordable before the pilot"),
  ("exploration_budget · daily_budget", "won",
   "`budget_share_of_il` × the trailing 7-day mean of realised daily IL, where a "
   "day's realised IL is the whole-episode IL (discount AND scrap) of episodes "
   "that CLOSED that day — settled at midnight, never a forecast; `budget_basis` "
   "names the basis. Scrap is charged via `classify_last`, only on a CLOSED "
   "episode — `ending_inventory == 0` on the last row. If the feed ever stops "
   "emitting that sentinel every episode reads unclosed, ALL scrap drops out "
   "and this reads ~10× too small; `ending_summary."
   "write_off_convention_in_force` is the flag that says so.", ""),
  ("exploration_budget · spend_over_budget", "ratio",
   "Spend ÷ budget. Above `exploration_cost_vs_budget` (2×) the stop condition "
   "would suspend exploration on day one of the pilot; between 1× and 2× the "
   "tau controller shrinks tau at the operator gate, capped at halving a day.",
   "GATE — read before launch, not after"),
  ("tau_initial_derivation", "won",
   "Shadow's own launch tau: the same bisection, run on the run's anchored path "
   "over the trailing `budget_il_window_days` before its window — the span the "
   "day-one budget base reads — against day one's budget. Used as the tau in "
   "force for the run (day one of the controller trace is an out-of-sample test "
   "of it) and the paste source for the pilot's `exploration.tau_initial`. "
   "`fallback: true` means the week was missing or under "
   "`tau0_derivation_min_decisions` and the config paste was used instead.",
   "GATE — the launch value for exploration.tau_initial"),
  ("exploration_budget · tau_recommended", "won",
   "The same bisection pooled over shadow's WHOLE window. Reported, never "
   "applied — a cross-check on the launch derivation, not its source. Check "
   "`tau_recommended_implied_spend` sits just UNDER `daily_budget` — spend "
   "steps as each cost crosses tau rather than sliding, so no tau lands "
   "exactly on the budget.", ""),
  ("exploration_budget · spread_decisions_per_episode", "count",
   "Decisions whose Q-spread funded the tau derivation, per episode. Near 1 "
   "means the entry-only scoping is back: the replay once collected spreads at "
   "`t == 0` only, funding one exploration per episode against a system that "
   "explores every hour, and its own bisection reported 1.00× regardless.", ""),
  ("tau_controller_trace · days_stop_condition_fires", "count",
   "Days in the window on which realised spend would have exceeded the 2× stop. "
   "`tau_next` reads only the day just closed, so day one is spent at whatever "
   "tau was launched with and no correction can precede it — a single "
   "spend_over_budget multiple cannot show this.",
   "GATE — does the pilot survive its own launch"),
  ("tau_controller_trace · episodes_per_day_sampled", "count",
   "The ONE figure a sample degrades. The gate reads rates, and `tau_recommended` / "
   "`spend_over_budget` equate two quantities that both scale with the sample, so they "
   "are sample-INVARIANT. The daily series is not — it divides the sample across the "
   "window's days, so 3,000 episodes over an 18-day hold-out leaves ~167 behind each "
   "day's budget and the controller looks jumpier than it is. Quote the pooled "
   "`spend_over_budget`; raise `--max-episodes` only to read this series closely.", ""),
  ("tau_controller_trace · window_days", "count",
   "Calendar span the budget divides by. NOT `days_with_decisions` (days that "
   "produced a decision) and NOT `days_simulated` (walked, capped at 60, with "
   "`days_truncated` reporting the rest). Reading \"N of M days\" against the "
   "wrong M is off by the gap, which on a sampled run is large.", ""),
  ("episodes_per_bounded_update", "count",
   "`information_increment ÷ effective information per episode`. Divide by the "
   "pilot's daily episode count for the evidence-side floor; the binding constraint "
   "is whichever is larger, that or the calendar.", "weeks-to-convergence"),
  ("realised_vs_predicted_sold_ratio_at_legacy_price", "ratio",
   "The production continuation of the calibration gate, and the first place "
   "frozen-baseline drift shows.", ""),
  ("solver_latency_p95_s", "secs",
   "95th percentile DP solve time.", ""),
 ]),

("Decision event", "inference.decide → events_store/decisions.jsonl",
 "36 required fields per priced hour. Written by the service; an integrating team "
 "produces none of it. Full field list in the event contract.", [
  ("expected_il", "won",
   "Expected inventory loss under the APPLIED action — the objective. Positive here; "
   "the DP's internal value is its negative, because the solver maximises value and "
   "every term is a cost.", "monitoring, and the reproduction check"),
  ("mu_ref_path", "count",
   "The FULL remaining-hours forecast, one entry per remaining hour. Its length must "
   "equal `hours_remaining`. Without it a decision cannot be recomputed.",
   "GATE — assurance reproduction"),
  ("anchor_discount", "rate",
   "The discount already in force, which fixed the action set. `null` at entry.",
   "GATE — assurance reproduction"),
  ("action_set_size", "count",
   "Actions allowed at THIS decision after the no-price-increase constraint. Distinct "
   "from the 25-tier grid, and what explorability is judged on.", ""),
  ("exploration_cost", "won",
   "`Q(p*) − Q(p)` for the applied tier: expected IL given up to run the experiment. "
   "Zero when not exploring.", "the tau controller and the budget stop condition"),
  ("affordable_set_size", "count",
   "How many alternatives cost no more than `tau`. **If non-empty, the draw happens** "
   "— there is no exploration rate.", "assurance uniformity"),
  ("tau_current", "won",
   "The exploration budget in force at that decision.", ""),
 ]),

("Outcome event", "the integrating team → events_store/outcomes.jsonl",
 "9 required fields, one per decision. This is the integration.", [
  ("units_sold", "count",
   "Zero-sale hours MUST be sent — around three quarters of hours, and the "
   "observations that identify demand at a price.", "the censored likelihood"),
  ("starting_inventory", "count",
   "Units at the top of the hour. `units_sold ≥ starting_inventory` is what marks "
   "the observation censored.", "the censored likelihood"),
  ("applied_price", "won",
   "The price ACTUALLY in force, not the one recommended. The gap between the two is "
   "monitored.", "GATE — price mismatch ≤ 1%"),
  ("is_stockout", "verdict",
   "Whether demand hit the inventory ceiling. Any missing rate above zero fires a "
   "stop condition. False on an episode-close write-off — that is not demand.",
   "GATE — stop condition"),
  ("execution_status", "text",
   "Only `ok`, `success` or absent make the outcome eligible for learning. Anything "
   "else is stored and excluded.", "learning eligibility"),
  ("adjustment_reason", "text",
   "Required when inventory does not reconcile. Exactly three are legitimate: "
   "`intraday_restock`, `episode_close_write_off` (~13.5% of episodes end holding "
   "stock) and `unexplained_shortfall` (shrink — named, not quarantined).",
   "GATE — event completeness"),
 ]),

("Learning", "pipeline.update → artifacts/posterior.json",
 "The only file production writes. Two things move in it, on two different kinds "
 "of evidence.", [
  ("information_pending", "count",
   "Effective information in the UNCONSUMED batch, recomputed each run. There is no "
   "stored counter — a sub-threshold batch is left unconsumed and re-evaluated whole.", ""),
  ("information_required", "count",
   "**12.0** — the threshold a cell's batch must clear. SET, not derived from data.",
   "GATE — whether the posterior moves"),
  ("batch_oldest_outcome_age_days", "days",
   "How long the batch has been accumulating without firing. Surfaces a stalled "
   "learning loop long before the 21-day flat-posterior alert.", ""),
  ("accumulated_information", "count",
   "Running total across COMMITTED revisions. The artifact-level survivor of the "
   "old pending counter.", ""),
  ("predictive_check", "ratio",
   "The unconsumed batch scored against the PRE-update posterior — an "
   "out-of-sample grade of the current belief, since every outcome in it "
   "arrived after that belief was set. Bracketed by `oracle` and `uniform` "
   "like the prior's holdout_comparison; `worse_than_a_flat_prior` persisting "
   "across batches means the posterior tightened faster than the evidence "
   "justified. Correlated hours inflate all three scores alike — read the "
   "differences, never the absolutes.", "the operator gate, before approving"),
  ("bound_clipped", "verdict",
   "Whether the bounded step bound. Mean moves at most 0.15, std shrinks at most 25%, "
   "floored at `min_std`. A clipped cell is flagged for operator review.", ""),
  ("tau", "won",
   "The exploration budget in force. Recalibrated on **spend, not evidence**, on "
   "every `--apply` — a day that explored and learned nothing still cost money.", ""),
  ("tau_calibrated_through", "text",
   "The last date tau consumed. The exactly-once guard: two runs in a day would "
   "otherwise move tau by the square of the ratio.", ""),
  ("processed_outcome_ids", "id",
   "The exactly-once ledger. An outcome is marked processed ONLY when a revision "
   "actually consumes it.", "GATE — exactly-once"),
 ]),

("Monitoring — business", "pipeline.monitor → reports/monitor.json → business",
 "Is the business all right? Section 15 family one.", [
  ("il_pct_aggregate", "rate",
   "Live IL%, always a ratio of sums and always reported with its denominator.", ""),
  ("il_pct_by_arm", "rate",
   "The same split by A/B arm, using the shared arm hash. The comparison the "
   "experiment reads out on.", ""),
  ("episodes_excluded_still_running", "count",
   "In-flight episodes held out. Their leftover is stock on the shelf, not scrap in "
   "the bin — booking it would count it today and something different tomorrow.", ""),
  ("sell_through", "rate", "Units sold ÷ units that entered the window.", ""),
  ("waste_units", "count", "Units scrapped at the close.", ""),
 ]),

("Monitoring — learning", "pipeline.monitor → reports/monitor.json → learning",
 "Is the loop actually learning? Section 15 family two.", [
  ("posterior_by_cell", "exp",
   "Mean, std, n_obs, accumulated information and version per cell.", ""),
  ("posterior_std_flat_alert", "days",
   "Cells whose std has not moved for N days. Std moves only when an update commits, "
   "so 'flat for N days' is exactly 'no committed update in N days'.",
   "the 21-day dead-loop alert"),
  ("forced_decision_count", "count", "Decisions that explored.", ""),
  ("affordable_set_empty_rate", "rate",
   "Share of decisions with nothing affordable to explore. High means tau is too "
   "tight or the action sets are too narrow.", ""),
  ("mean_forced_log_price_ratio", "ratio",
   "Average |log price ratio| on forced decisions. Information goes as its SQUARE, "
   "so this drives learning speed more than the count does.", ""),
  ("realised_exploration_cost", "won",
   "What exploration actually spent.", "GATE — vs 2× budget, and the tau controller"),
 ]),

("Monitoring — safety", "pipeline.monitor → reports/monitor.json → safety",
 "Is the plumbing sound? Section 15 family three. Three of these can suspend "
 "exploration.", [
  ("duplicate_or_unmatched_rate", "rate",
   "Repeated ids plus outcomes with no matching decision. **≤ 1%.**", "GATE — stop condition"),
  ("applied_vs_recommended_price_mismatch", "rate",
   "Outcomes whose applied price differs from the decision's by more than 1e-6. "
   "**≤ 1%.** The 'did our price go live' check.", "GATE — stop condition"),
  ("missing_stockout_field_rate", "rate",
   "Outcomes without `is_stockout`. **Must be exactly zero.**", "GATE — stop condition"),
  ("quarantined_event_count", "count",
   "Events written to quarantine with their validation failure attached. Nothing is "
   "ever silently dropped. **Scope differs by report:** in `pipeline.monitor` it is the "
   "whole store — the standing production log, which is what a monitor should show — "
   "while in `pipeline.shadow` it is THIS RUN. Shadow read the whole file until two "
   "runs over identical input disagreed; ids are minted per run, so the file grows "
   "every time and the figure was not a property of the run the gate was judging.", ""),
  ("realised_vs_predicted_sold_ratio", "ratio",
   "Realised revenue base ÷ predicted. Production's continuation of the calibration gate.", ""),
 ]),

("Assurance", "pipeline.assurance → reports/assurance.json",
 "Are the FROZEN ARTIFACTS still a description of the world we price in? Verdicts "
 "are PASS / FAIL / **INSUFFICIENT** — a thin window says so rather than passing.", [
  ("reproduction · mismatch_rate", "rate",
   "Share of logged decisions that do not re-solve to themselves from their own "
   "event payload. The DP is deterministic, so any mismatch means config, artifact, "
   "code or a library moved underneath it.", "GATE — operator gate"),
  ("reproduction · decisions_skipped_no_inputs", "count",
   "Events that cannot be replayed because they predate `mu_ref_path`. Skipped, "
   "never silently counted as passing.", ""),
  ("dispersion · bins_flagged", "count",
   "Predicted-μ bins where realised zero-sale or stockout rates diverge from "
   "`NB(μ, r)`. Both statistics are EXACT under censoring; a variance comparison "
   "would not be.", "GATE — operator gate"),
  ("correlation · rho_live vs rho_frozen", "rate",
   "ρ re-measured on live residuals AT THE WORKING ELASTICITY — the same basis "
   "`fit_dispersion` used. Measuring at the posterior mean would show drift that is "
   "not there.", "GATE — operator gate"),
  ("exploration · chi_square / p_value", "ratio",
   "Tests that the applied tier is a uniform draw from the reconstructed affordable "
   "set. Uniformity is what makes the evidence causal.", "GATE — operator gate"),
  ("exploration · affordable_but_not_explored", "count",
   "Decisions reporting a non-empty affordable set and no exploration. The two "
   "disagree, and `select()` cannot produce that.", "GATE"),
 ]),

("Status board", "pipeline.status",
 "The checks that gate a decision, each with the figure behind it and where to "
 "look when red. Computes nothing. Exit code 1 on any FAIL. **Start here.**", [
  ("launch blockers", "verdict", "Config values strict mode refuses to start without.", "GATE"),
  ("artifact bundle", "verdict",
   "Do the frozen artifacts form ONE bundle, unedited since sealing? Mixed vintages "
   "and edited files are separate failures with different remedies.", "GATE"),
  ("artifact mirrors", "verdict",
   "Do the config pastes still match their source — the frozen artifacts (rho, "
   "forced hours) and phase 0's A/B power SE? Read the bundle line FIRST — the "
   "check says they disagree, not which is stale.", "GATE"),
  ("report vintages", "verdict",
   "Were backtest.json and shadow.json produced against the artifacts on disk? "
   "After a retrain the old reports still show green gates while grading a model "
   "that no longer exists (hard rule 1). Model mismatch FAILs — re-run the "
   "report; a moved config_version WARNs.", "GATE"),
  ("calibration level", "verdict", "The level at the anchor, in band. Diagnostic — WARN out of band, since calibration is always applied.", ""),
  ("elasticity prior", "verdict", "How many categories stand on their own data, and how many are wrong-signed.", ""),
  ("exploration tau", "verdict", "What is in force, and the latest derivation.", "GATE"),
  ("shadow gate", "verdict", "Completeness, matched rate, cost-floor violations.", "GATE"),
  ("guardrail floors", "verdict", "The two owner thresholds against their noise floors.", "GATE"),
  ("stop conditions", "verdict",
   "How many evaluated, how many fired, and how many CANNOT fire because their "
   "threshold is null.", "GATE"),
  ("assurance", "verdict", "The four live-data checks, with thin windows warned not passed.", "GATE"),
  ("walkthrough · replay", "verdict",
   "Do the figures typed into the leadership walkthrough still match the report they "
   "were read from? WARN when the report on disk is from a DIFFERENT model version — "
   "that cannot be compared at all (hard rule 1), so it is 'stale, unverifiable', not "
   "'wrong'. FAIL only on a disagreement WITHIN one run.", ""),
  ("walkthrough · shadow", "verdict",
   "Same check for the shadow tab. 'not run' until the hold-out run fills its "
   "reserved figure slots — registered before the numbers exist, so how the result "
   "gets read is fixed before anyone can see it.", ""),
 ]),
]


# ------------------------------------------------------------------ rendering
CSS = """
:root {
  --ground:#F3F3F1; --surface:#FFFFFF; --sunk:#EAEAE6; --rule:#DEDED8;
  --rule-firm:#C2C2B8; --ink:#1A1A18; --muted:#65645D; --faint:#8A897F;
  --accent:#94282C; --accent-w:#F4E4E3; --gate:#7A5B12; --gate-w:#F6EEDA;
  --display:"Source Serif 4",Georgia,serif;
  --body:"Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;
  --data:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#131311; --surface:#1B1B18; --sunk:#212120; --rule:#2E2E2A;
  --rule-firm:#43433C; --ink:#E9E8E2; --muted:#9C9B91; --faint:#807F76;
  --accent:#E08A82; --accent-w:#331A19; --gate:#D9B36A; --gate-w:#332A14;
}}
:root[data-theme="dark"]{
  --ground:#131311; --surface:#1B1B18; --sunk:#212120; --rule:#2E2E2A;
  --rule-firm:#43433C; --ink:#E9E8E2; --muted:#9C9B91; --faint:#807F76;
  --accent:#E08A82; --accent-w:#331A19; --gate:#D9B36A; --gate-w:#332A14;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--body);
  font-size:16px;line-height:1.6;margin:0;padding:0 20px 110px;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto}
header.mast{padding:54px 0 26px}
.eyebrow{font-family:var(--data);font-size:11px;font-weight:500;letter-spacing:.17em;
  text-transform:uppercase;color:var(--accent);margin:0 0 16px}
h1{font-family:var(--display);font-weight:600;font-size:clamp(30px,5vw,44px);
  line-height:1.08;letter-spacing:-.015em;text-wrap:balance;margin:0 0 18px}
.standfirst{font-size:17.5px;line-height:1.55;color:var(--muted);max-width:62ch;margin:0}
.standfirst strong{color:var(--ink);font-weight:600}

/* filter bar */
.bar{position:sticky;top:0;z-index:30;background:var(--ground);
  border-bottom:1px solid var(--rule-firm);padding:12px 0;margin-bottom:8px}
.bar .inner{max-width:1000px;margin:0 auto;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
#q{flex:1 1 260px;min-width:0;font-family:var(--data);font-size:14px;color:var(--ink);
  background:var(--surface);border:1px solid var(--rule-firm);border-radius:5px;
  padding:9px 12px}
#q:focus{outline:2px solid var(--accent);outline-offset:-1px}
#q::placeholder{color:var(--faint)}
#count{font-family:var(--data);font-size:12.5px;color:var(--muted);white-space:nowrap}
.toggle{font-family:var(--data);font-size:11.5px;font-weight:500;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);background:var(--surface);cursor:pointer;
  border:1px solid var(--rule-firm);border-radius:5px;padding:8px 12px}
.toggle[aria-pressed="true"]{background:var(--gate-w);color:var(--gate);border-color:var(--gate)}

/* sections */
section{margin-top:40px;scroll-margin-top:74px}
section[hidden]{display:none}
h2{font-family:var(--display);font-weight:600;font-size:24px;line-height:1.2;
  letter-spacing:-.01em;margin:0 0 4px;text-wrap:balance}
.src{font-family:var(--data);font-size:12px;color:var(--accent);margin:0 0 8px;
  word-break:break-word}
.intro{font-size:14.5px;color:var(--muted);max-width:70ch;margin:0 0 4px}

/* rows */
.rows{border:1px solid var(--rule);border-radius:6px;background:var(--surface);
  margin-top:16px;overflow:hidden}
.row{display:grid;grid-template-columns:minmax(200px,300px) 1fr;gap:0 22px;
  padding:14px 18px;border-top:1px solid var(--rule)}
.row:first-child{border-top:0}
.row[hidden]{display:none}
.row:target,.row.hit{background:var(--accent-w)}
.nm{font-family:var(--data);font-size:13.5px;font-weight:500;color:var(--ink);
  word-break:break-word;display:flex;flex-wrap:wrap;gap:7px;align-items:baseline}
.u{font-family:var(--data);font-size:10px;font-weight:500;color:var(--muted);
  background:var(--sunk);border-radius:3px;padding:2px 6px;white-space:nowrap}
.mn{font-size:14.5px;line-height:1.52}
.mn code{font-family:var(--data);font-size:.86em;background:var(--sunk);
  border-radius:3px;padding:1px 5px}
.mn strong{font-weight:600}
.rd{display:block;margin-top:6px;font-size:12.5px;color:var(--muted)}
.rd b{font-family:var(--data);font-size:10px;font-weight:600;letter-spacing:.09em;
  text-transform:uppercase;color:var(--gate);background:var(--gate-w);
  border-radius:3px;padding:2px 6px;margin-right:7px}
.empty{padding:30px 18px;color:var(--muted);font-size:14.5px}
.empty[hidden]{display:none}
footer{margin-top:60px;padding-top:20px;border-top:1px solid var(--rule);
  font-size:13px;color:var(--faint);max-width:70ch}
footer code{font-family:var(--data)}
@media (max-width:680px){
  .row{grid-template-columns:1fr;gap:8px}
  body{padding:0 15px 80px}
}
"""

JS = """
(function(){
  var q=document.getElementById("q"),cnt=document.getElementById("count"),
      gate=document.getElementById("gateonly"),empty=document.getElementById("empty"),
      rows=[].slice.call(document.querySelectorAll(".row")),
      secs=[].slice.call(document.querySelectorAll("section[data-sec]"));
  rows.forEach(function(r){r._t=r.textContent.toLowerCase();});
  function apply(){
    var s=q.value.trim().toLowerCase(), g=gate.getAttribute("aria-pressed")==="true", n=0;
    rows.forEach(function(r){
      var ok=(!s||r._t.indexOf(s)>=0)&&(!g||r.dataset.gate==="1");
      r.hidden=!ok; if(ok)n++;
    });
    secs.forEach(function(sec){
      sec.hidden=!sec.querySelector(".row:not([hidden])");
    });
    cnt.textContent=n+(n===1?" metric":" metrics");
    empty.hidden=n>0;
  }
  q.addEventListener("input",apply);
  gate.addEventListener("click",function(){
    gate.setAttribute("aria-pressed",gate.getAttribute("aria-pressed")==="true"?"false":"true");
    apply();
  });
  q.addEventListener("keydown",function(e){if(e.key==="Escape"){q.value="";apply();}});
  apply();
})();
"""


def _md(text):
    """The only markup the entries use: **bold**, `code`."""
    out, parts = [], html.escape(text).split("**")
    for i, part in enumerate(parts):
        out.append(f"<strong>{part}</strong>" if i % 2 else part)
    text = "".join(out)
    parts, out = text.split("`"), []
    for i, part in enumerate(parts):
        out.append(f"<code>{part}</code>" if i % 2 else part)
    return "".join(out)


def render():
    total = sum(len(entries) for _, _, _, entries in CATALOGUE)
    body = []
    for i, (title, source, intro, entries) in enumerate(CATALOGUE):
        rows = []
        for name, unit, meaning, read in entries:
            is_gate = read.startswith("GATE")
            read_html = ""
            if read:
                label = "<b>gates</b>" if is_gate else ""
                txt = read[5:].lstrip(" —-") if is_gate else read
                read_html = (f'<span class="rd">{label}{_md(txt)}</span>'
                             if txt else f'<span class="rd">{label}</span>')
            rows.append(
                f'<div class="row" data-gate="{"1" if is_gate else "0"}">'
                f'<div class="nm">{_md(name)}'
                f'<span class="u">{html.escape(UNITS.get(unit, unit))}</span></div>'
                f'<div class="mn">{_md(meaning)}{read_html}</div></div>')
        body.append(
            f'<section data-sec id="s{i}">\n<h2>{html.escape(title)}</h2>\n'
            f'<p class="src">{html.escape(source)}</p>\n'
            f'<p class="intro">{_md(intro)}</p>\n'
            f'<div class="rows">\n' + "\n".join(rows) + "\n</div>\n</section>")

    OUT.write_text(f"""<title>Markdown Metrics Index</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=JetBrains+Mono:wght@400;500;600&\
family=Source+Sans+3:wght@400;600&\
family=Source+Serif+4:wght@600&display=swap">
<style>{CSS}</style>

<div class="wrap">
<header class="mast">
  <p class="eyebrow">Perishable Markdown MVP · reference</p>
  <h1>Every number the system measures, and who owns it</h1>
  <p class="standfirst">
    The four reports carry around <strong>725 fields</strong> between them, which is the right
    number to write and the wrong number to read. This is the index: what a quantity means,
    what unit it is in, the component that writes it, and whether anything downstream is
    gated on it. Type to filter, or show only the checks that block something.
  </p>
</header>
</div>

<div class="bar"><div class="inner">
  <input id="q" type="search" placeholder="filter — try  il, tau, censor, rho, gate" aria-label="Filter metrics">
  <button class="toggle" id="gateonly" aria-pressed="false">gates only</button>
  <span id="count">{total} metrics</span>
</div></div>

<div class="wrap">
{chr(10).join(body)}
<p class="empty" id="empty" hidden>Nothing matches that filter.</p>

<footer>
  <p>
    Generated by <code>tools.metrics_glossary</code>. Entries whose source of truth is
    machine-readable — the two event schemas, the frozen-artifact list, and the status
    gate names — are cross-checked against the code by
    <code>tests/test_metrics_glossary.py</code>, so those cannot drift silently. The rest is
    prose held true by hand, which is why it lives in one file rather than scattered across
    the reports it describes.
  </p>
  <p>
    Figures quoted are the <code>baseline-20260811043259</code> production run. Where a value
    is owner-set rather than measured, the entry says so.
  </p>
</footer>
</div>

<script>{JS}</script>
""")
    return total


if __name__ == "__main__":
    n = render()
    print(f"wrote {OUT.relative_to(ROOT)}: {n} metrics, {len(CATALOGUE)} components")
