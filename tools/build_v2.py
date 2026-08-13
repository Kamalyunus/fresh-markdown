"""Build deck v2: problem -> solution -> novelty -> components -> results -> A/B.

Rebuilds the NARRATIVE from the existing deck's slides rather than from
scratch: every reused slide keeps its vetted content, layout and speaker
notes, so the restructure cannot reintroduce a fixed error. Seven new slides
fill genuine gaps. All figures refreshed from the latest deck_numbers run.
"""
import os

from tools import deckkit as k

# The original 34-slide deck. It is no longer a deliverable -- v3 is the only
# deck we present -- but it is still the SOURCE every reused slide is lifted
# from, so it lives in tools/ rather than docs/ and must not be deleted.
SRC = "tools/deck_source.pptx"

# v2 is now an intermediate rather than a deliverable: v3 is the deck to
# present, and it is built from v2's slides. Retiring v2 as a file must not
# retire the step, so this writes into build/ (gitignored) and tools.build_v3
# runs it. To look at v2 on its own, run this module and open the result.
OUT = "build/perishable_markdown_deck_v2.pptx"
os.makedirs("build", exist_ok=True)

k.unpack(SRC)

# ------------------------------------------------ number refresh (file nums)
k.sub_in_slide(2, [
    ("32.3%", "36.7%"),
    ("Inventory Loss as a share of full-price sales value — 356,114 episodes",
     "Inventory Loss as a share of full-price sales value, on a 2,000-episode replay"),
    ("clearance under legacy — scrap is ~7% of IL today",
     "clearance under legacy — the bar the system must not break"),
    ("₩14.7M", "₩17.1M"),
    ("IL on a 2,000-episode replay sample alone",
     "IL on that same 2,000-episode sample — 334k episodes in the full window"),
])
k.sub_in_slide(14, [("0.10 s", "0.09 s")])
k.sub_in_slide(18, [
    ("₩1,271 / day", "₩1,476 / day"),
    ("implied spend vs a ₩1,271/day budget on the replay sample — calibration lands on target",
     "implied spend against a ₩1,475/day budget — the derivation lands on target by construction"),
])
k.sub_in_slide(19, [("38.1%", "43.4%"),
                    ("of shadow decisions would have been forced — measured before any price was applied",
                     "of shadow decisions would have been forced, measured before any price was applied")])
k.sub_in_slide(23, [
    ("1,837 episodes per step", "1,381 episodes per step"),
    ("0.0065 units of effective information", "0.0087 units of effective information"),
])
k.sub_in_slide(28, [
    ("Measured on production data · 2026-08-09",
     "Measured on production data · model baseline-20260811043259"),
    ("32.27%  (₩14.27M / 2,000 episodes)", "36.68%  (₩17.11M / 2,000 episodes)"),
    ("₩409.87 from the passing run · spend ≈ budget",
     "₩447.78 from the passing run · spend ≈ budget"),
    ("IL% clustered SE (18 weeks)", "IL% clustered SE (SKU × FC, 71,559 units)"),
    ("0.000875", "0.002915"),
    ("6× the original assumption — drives the A/B duration decision",
     "3× the pre-scrap-fix figure — a real scrap term makes IL lumpier, not smoother"),
])
k.sub_in_slide(25, [("100% · 100% · zero", "100% · 100% · zero")])

# --------------------------------------------------------------- new slides
N = {}
for key, tmpl in [("legacy", 15), ("system", 25), ("demand", 16),
                  ("calib", 21), ("variance", 6), ("result", 27), ("abdesign", 30)]:
    N[key] = k.dup(tmpl)

# --- 3 · What the legacy policy does ------------------------------- [T3]
k.Slide(N["legacy"]) \
 .runs("Text 2", ["THE POLICY TODAY"]) \
 .runs("Text 3", ["A rule that prices by the clock, not by demand"]) \
 .runs("Text 6", ["What it does"]) \
 .paras("Text 7", [
     "Enter every episode at a fixed reference discount, set per category.",
     "Deepen roughly one percentage point per hour, every hour, to a cap.",
     "Hold at the cap until the window closes, then scrap whatever is left."]) \
 .runs("Text 10", ["What it never looks at"]) \
 .paras("Text 11", [
     "How much stock is on the shelf, or how fast it is moving.",
     "How many hours the window actually has left.",
     "Whether the price it is charging has fallen below cost."]) \
 .runs("Text 12", ["Legible and safe — and demand-blind:  ",
     "it is also the reason price and hour are the same variable in five months "
     "of history, which is the constraint the next two slides are about. Measured "
     "result: 36.7% of full-price sales value lost, at 93% clearance."]) \
 .save()

# --- 6 · What the system does -------------------------------------- [T15]
k.Slide(N["system"]) \
 .runs("Text 2", ["THE SYSTEM IN ONE SLIDE"]) \
 .runs("Text 3", ["Price for the window ahead, and pay to learn"]) \
 .runs("Text 5", ["Every hour"]) \
 .runs("Text 6", ["choose the price"]) \
 .runs("Text 7", [
     "For each SKU × fulfilment centre, take the stock on hand and the hours left "
     "and pick the discount that minimises expected loss over the WHOLE remaining "
     "window, not just the next hour. A below-cost price is not among the options."]) \
 .runs("Text 9", ["Some hours"]) \
 .runs("Text 10", ["pay to learn"]) \
 .runs("Text 11", [
     "Deliberately choose a different price and record what happened. The cost of "
     "each experiment is known in won before it is made, and total spend is capped "
     "at 1% of the loss the system exists to reduce."]) \
 .runs("Text 13", ["Every day"]) \
 .runs("Text 14", ["update one belief"]) \
 .runs("Text 15", [
     "Use ONLY those randomised outcomes to update a single number — how customers "
     "respond to price. A human approves each step, the step is bounded, and every "
     "outcome is spent exactly once."]) \
 .runs("Text 17", ["Nothing else moves:  ",
     "demand levels, seasonality and variance are fit offline and frozen. The only "
     "thing that changes in production is the price-response belief — so if the "
     "system's behaviour changes, it learned something."]) \
 .save()

# --- 17 · Price-blind demand model --------------------------------- [T5]
k.Slide(N["demand"]) \
 .runs("Text 2", ["COMPONENT · DEMAND MODEL"]) \
 .runs("Text 3", ["A demand model deliberately blind to price"]) \
 .runs("Text 6", ["WHAT IT PREDICTS"]) \
 .paras("Text 7", [
     "Units sold in the next hour at the REFERENCE discount — one number, mu_ref",
     "LightGBM on a Tweedie objective, fit on 2.07M hourly rows and frozen at launch",
     # NOT "recent 1h/3h sales": within-episode lags are post-treatment
     # mediators of the episode's own price path and are excluded by design
     # (design 5.4). The slide claimed a feature set the model does not have.
     "Features: SKU × FC velocity measured at the reference price, hour, day of week, category, FC — no within-episode lags",
     "Every economic quantity the planner computes is denominated in this prediction"]) \
 .runs("Text 10", ["WHY IT MUST NOT SEE PRICE"]) \
 .runs("Text 11", ["✕"]) \
 .paras("Text 12", ["One price feature, overwritten to the",
                    "reference discount at inference time"]) \
 .paras("Text 13", [
     "Legacy ties price to the clock, so a model free to use price would learn the evening peak and report it as elasticity",
     "Its price gradient is therefore never queried — the slope comes from ε alone, and ε is learned from randomisation",
     "That split IS the design: the model owns the LEVEL, the posterior owns the SLOPE, and neither is allowed the other's job"]) \
 .runs("Text 14", ["Measured fidelity: 0.9940 sold ratio over 2.07M rows, hourly MAE 0.4399 — accurate at the level, and never asked about the slope."]) \
 .save()

# --- 18 · Calibration ---------------------------------------------- [T3]
k.Slide(N["calib"]) \
 .runs("Text 2", ["COMPONENT · CALIBRATION"]) \
 .runs("Text 3", ["Correct the level, never the slope"]) \
 .runs("Text 6", ["What may be corrected"]) \
 .paras("Text 7", [
     "One multiplicative factor per subcategory, fit on ANCHOR rows only.",
     "Anchor rows are those priced at the reference discount, where the elasticity term is exactly 1.",
     "Fit on train+calib, gated on test — disjoint, so the gate never grades its own fit."]) \
 .runs("Text 10", ["What may never be"]) \
 .paras("Text 11", [
     "A sold ratio that degrades as the discount moves away from the anchor is SLOPE error.",
     "Scaling mu_ref cannot fix slope error. It is fixed by re-estimating the prior, or by learning.",
     "Five successive gate failures were each traced to a distinct cause, ending in a censoring-basis bug."]) \
 .runs("Text 12", ["The basis is the trap:  ",
     "the factor is solved against E[min(D, q)], never raw mu. Measured, a true "
     "correction of 1.45 fits as 0.68 on the raw basis — the wrong side of 1, which "
     "is why calibration used to leave the gate unmoved or worse. Gate now PASSES at "
     "1.0389 in a [0.90, 1.10] band."]) \
 .save()

# --- 19 · Dispersion and correlation ------------------------------- [T8]
k.Slide(N["variance"]) \
 .runs("Text 2", ["COMPONENT · VARIANCE STRUCTURE"]) \
 .runs("Text 3", ["Two frozen numbers that decide how fast we may believe"]) \
 .runs("Text 6", ["Negative binomial, not Poisson"]) \
 .runs("Text 7", [
     "Hourly demand is far more variable than Poisson — bursty shoppers, basket effects. "
     "Var[D] = mu + mu²/r. A Poisson likelihood would make every learning update "
     "overconfident, and each update is bounded on the assumption that the variance is "
     "honest."]) \
 .runs("Text 10", ["r by subcategory, with a fallback"]) \
 .runs("Text 11", [
     "Dispersion genuinely differs by product type and the data supports estimating it "
     "there: censored MLE per subcategory on the calibration weeks, 200 rows required, "
     "falling back subcategory → category → global. High converged values are clamped at "
     "the 90th percentile and low ones preserved — high r means near-Poisson, which is the "
     "dangerous direction."]) \
 .runs("Text 14", ["rho global, and deliberately so"]) \
 .runs("Text 15", [
     "Within-episode residual correlation is ONE scalar, 0.3103. Per category it would rest "
     "on much weaker signal, and a noisy per-cell rho deflates evidence unevenly. Episode-level "
     "random effects replace the scalar in phase 2."]) \
 .runs("Text 18", ["deff = 3.347 — the honesty tax"]) \
 .runs("Text 19", [
     "1 + (8.563 − 1) × 0.3103. Hours in one episode share an inventory pool and a demand "
     "shock, so ten forced hours are worth about three independent ones. Accumulated "
     "information is divided by this before any update may commit — and start-up refuses to "
     "run if config drifts from the frozen artifact."]) \
 .save()

# --- 34 · What the -38% is and isn't ------------------------------- [T9]
k.Slide(N["result"]) \
 .runs("Text 2", ["READING THE RESULT"]) \
 .runs("Text 3", ["What the −38% is, and what it costs"]) \
 .runs("Text 5", ["Both arms simulated under the SAME demand model:"]) \
 .paras("Text 6", [
     "legacy-under-model IL   vs   DP-under-model IL",
     "same mu_ref, same r, same ε  → model bias cancels",
     "actual_* figures are FIDELITY, never policy",
 ]) \
 .paras("Text 7", [
     "The DP does not clear more — it clears slightly LESS, 76.61% against legacy's 77.58%, and scraps more: ₩6.84M against ₩6.24M.",
     "The entire gain comes from not over-discounting: mean discount 0.1285 against legacy's 0.2935. The legacy ramp buys clearance at a price well above what the units are worth.",
     "Say this before someone finds it. A markdown system that cuts loss by discounting LESS is the right answer here, and it is not the answer most people expect.",
 ]) \
 .runs("Text 8", ["−38.0%"]) \
 .runs("Text 9", ["Inventory Loss, like-for-like against legacy under the same demand model"]) \
 .runs("Text 10", ["−0.97pp"]) \
 .runs("Text 11", ["clearance, 77.58% → 76.61%, with scrap cost up ₩0.59M — the price of the win"]) \
 .runs("Text 12", ["0.1285"]) \
 .runs("Text 13", ["DP mean discount against legacy's 0.2935 — the mechanism, in one number"]) \
 .save()

# --- 36 · A/B design ----------------------------------------------- [T8]
k.Slide(N["abdesign"]) \
 .runs("Text 2", ["HOW WE WILL KNOW"]) \
 .runs("Text 3", ["The experiment that decides it"]) \
 .runs("Text 6", ["Randomise on SKU × FC, not episode"]) \
 .runs("Text 7", [
     "Consecutive episodes of the same unit share inventory carryover, so episode-level "
     "assignment would leak treatment into control. 71,559 units, split 50/50 by a stable "
     "hash — the same hash the monitor uses, so the split measured in the derivation is the "
     "split that runs."]) \
 .runs("Text 10", ["A ratio of sums, clustered"]) \
 .runs("Text 11", [
     "IL% is a ratio of sums, never a mean of per-episode ratios — a zero-sale episode has no "
     "defined ratio. Analysed by delta-method linearisation with standard errors clustered on "
     "the assignment unit, and absolute IL reported beside every IL%."]) \
 .runs("Text 14", ["It cannot measure elasticity"]) \
 .runs("Text 15", [
     "Both arms are POLICIES, not price randomisations, so the A/B answers “does the policy "
     "beat legacy” and nothing else. Elasticity learning happens inside the treatment arm from "
     "forced exploration — two separate mechanisms answering two separate questions."]) \
 .runs("Text 18", ["Decided before it starts"]) \
 .runs("Text 19", [
     "Duration fixed in advance and honoured — no early reads. The four-cell decision table is "
     "pre-committed. Guardrails on scrap, margin, sell-through and stockout rate; a breach halts "
     "the treatment arm, and it stops LEARNING, never pricing."]) \
 .save()


# Speaker notes for the seven new slides. dup() deliberately does not
# inherit its template's notes, so each gets its own.
for key, text in [
    ("legacy",
     "Sets up the whole argument. The rule is legible and safe, which is "
     "why it has survived -- but it prices on the clock and never looks at "
     "demand, stock or the cost floor. Two things to land: it is the "
     "baseline the -38% is measured against, and it is the REASON history "
     "cannot identify elasticity, because it makes price a deterministic "
     "function of the hour."),
    ("system",
     "The plain-language product, before any mechanism. If someone leaves "
     "after one slide, this is the one. Three verbs: choose, pay to learn, "
     "update. The footer is the attribution argument -- everything else is "
     "frozen, so a change in behaviour IS learning, not drift. Do not go "
     "into the DP or tau here; both come later."),
    ("demand",
     "The component most people assume must use price, and the one where "
     "NOT using it is the novelty. The model predicts demand at the "
     "reference discount only; its single price feature is overwritten at "
     "inference so the gradient is never queried. If it could use price it "
     "would learn the legacy clock ramp and report it as elasticity. Model "
     "owns the LEVEL, posterior owns the SLOPE -- neither is allowed the "
     "other's job."),
    ("calib",
     "The most-debugged part of the system: five successive gate failures, "
     "each a distinct cause. The one worth telling is the basis bug -- the "
     "factor must be solved against the CENSORED expectation, not raw mu. A "
     "true correction of 1.45 fit as 0.68 on the raw basis, the wrong side "
     "of 1, which is why calibration used to make the gate worse. Also "
     "state the rule: level may be corrected multiplicatively, slope never."),
    ("variance",
     "Two frozen numbers most audiences will not ask about, but they set "
     "how fast we are allowed to believe anything. Negative binomial "
     "because Poisson would make every update overconfident. r per "
     "subcategory because dispersion genuinely differs and the data "
     "supports it; rho as one global scalar because per-cell it would be "
     "noise. deff = 3.347 is the honesty tax: ten forced hours in one "
     "episode are worth about three independent ones."),
    ("result",
     "Say this before someone finds it in the table. The DP does not clear "
     "more -- it clears slightly less and scraps more. The entire gain is "
     "not over-discounting: 0.1285 mean discount against legacy's 0.2935. A "
     "markdown system that cuts loss by discounting LESS is the correct "
     "answer here and it is not what people expect, so lead with it rather "
     "than letting it be discovered. Also flag that both arms run under the "
     "same demand model, which is what makes the comparison honest."),
    ("abdesign",
     "The design, not the power -- power is the next slide. Three things: "
     "randomisation is on SKU x FC because consecutive episodes of the same "
     "unit share inventory carryover; the estimator is a ratio of sums with "
     "clustered errors, never a mean of per-episode ratios; and the A/B "
     "CANNOT measure elasticity, because both arms are policies. Elasticity "
     "learning happens inside the treatment arm from forced exploration. "
     "Two mechanisms, two questions."),
]:
    k.notes(N[key], text)

# ------------------------------------------------------------------- order
ORDER = [
    1, 2, N["legacy"], 4, 5,                       # 1 problem
    N["system"], 3, 9, 11, 10,                     # 2 solution
    12, 29, 30,                                    # 3 novelty
    6, 7, 8,                                       # 4a data
    N["demand"], N["calib"], N["variance"],        # 4b demand model
    14, 13, 15, 16, 17,                            # 4c decision core
    18, 19,                                        # 4d exploration
    20, 21, 22, 23,                                # 4e learning
    26,                                            # 4f monitoring
    24, 28, N["result"], 25,                       # 5 backtest
    N["abdesign"], 27, 31, 32, 33, 34,             # 6 A/B + production
]
assert len(ORDER) == len(set(ORDER)), "duplicate slide in ORDER"
k.set_order(ORDER)

# page numbers follow the new positions; slides 1 and 34 have no page number
for pos, n in enumerate(ORDER, start=1):
    if n in (1, 34):
        continue
    k.Slide(n).runs("Text 1", [str(pos)]).save()

k.pack(OUT)
# v1 file numbers equal v1 positions, so ORDER is also the v1 -> v2 position
# map. build_v3 composes it with its own to keep deck_numbers' slide tags --
# which are v1 positions -- usable against the deck people actually read.
V1_TO_V2 = {n: i for i, n in enumerate(ORDER, start=1) if n <= 34}
print(f"built {OUT} with {len(ORDER)} slides")
print("new slides:", {kk: vv for kk, vv in N.items()})
