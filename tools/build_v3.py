"""Build deck v3: a seventeen-slide presented core, everything else indexed
behind an appendix divider in the SAME file.

v2 is right and complete but 41 slides is more than an hour of leadership
attention will absorb. v3 changes nothing about the argument: it promotes the
seventeen slides that carry it, moves the other twenty-five behind an appendix
divider and an index keyed by the question a room actually asks, and tightens
four core slides that were written to be read rather than presented.

Same principle as build_v2 -- restructure, never retype. Every slide here is a
v2 slide; the three genuinely new ones are a headline-results slide and the two
navigation slides. Nothing is dropped, which tools.deck_diff enforces.

Every slide below is named by its POSITION in v2 -- slide 8 means the eighth
slide of the deck everyone reviewed. Slide FILE numbers do not track positions
and are resolved once, at the top, through v2's own slide-id list.

Usage:  python3 -m tools.build_v3
"""
from tools import deckkit as k

SRC = "build/perishable_markdown_deck_v2.pptx"
OUT = "docs/perishable_markdown_deck_v3.pptx"

# v2 is no longer checked in -- v3 contains all of it, and two decks to keep
# in sync is how a number goes stale in one of them. It is still the stage
# this build reads from, so importing build_v2 RUNS it: the whole chain is
# v1 -> v2 (build/) -> v3, from one command, with no leftover to go stale.
from tools.build_v2 import V1_TO_V2

k.unpack(SRC)

_V2 = k.order()          # v2 presentation order, as slide-file numbers


def f(pos):
    """v2 position -> slide file number."""
    return _V2[pos - 1]


# ========================================================= tighten for a room
# Four core slides carry a second clause on every line -- right for a reader,
# too much for a listener. The argument is unchanged; the supporting detail
# these lines carried is still on the appendix slide that owns it.

# 8 · DESIGN THESIS -- the cell-count reasoning belongs to the learning section
k.sub_in_slide(f(8), [
    ("Baseline demand at the reference discount (LightGBM / Tweedie) — its price gradient is never queried",
     "Baseline demand at the reference discount — its price gradient is never queried"),
    ("~10 learning cells: high-volume categories plus one pooled cell. NOT subcategory — a finer ε changes no price until a cell crosses the deepening bar (2.43), and splitting cells divides the same evidence, so finer grain would only slow the one thing that gets us there",
     "~10 learning cells — high-volume categories plus one pooled cell. NOT subcategory: a finer ε changes no price until a cell crosses the deepening bar (2.43), and splitting cells only divides the same evidence"),
])

# 10 · OBJECTIVE & METRIC -- the worked table carries it; the prose can halve
k.sub_in_slide(f(10), [
    ("IL% has an endogenous denominator: deeper markdowns sell more units and grow it. The two metrics can disagree by design:",
     "IL% has an endogenous denominator — deeper markdowns sell more units and grow it. So the two can disagree by design:"),
    ("Original price 10,000 · cost 2,000 · 10 units. The planner picks A — 16,000 of loss beats 28,000. IL% prefers B because its own denominator grew. A ratio objective can be gamed by discounting harder; the planner optimises the currency amount instead.",
     "Price 10,000 · cost 2,000 · 10 units. The planner picks A: 16,000 of loss beats 28,000. IL% prefers B because its own denominator grew. A ratio objective is gameable by discounting harder — so the planner optimises the currency amount."),
    ("Stated now, before the experiment — so the readout is not decided by whoever reads the dashboard.",
     "Pre-committed, so the readout is not decided by whoever reads the dashboard first."),
])

# 12 · NOVELTY -- same seven rows, cells short enough to read from a seat
k.sub_in_slide(f(12), [
    ("Deliberately blind to price — its one price feature is overwritten at inference",
     "Blind to price by construction — its price feature is overwritten at inference"),
    ("The cost floor constructs the actions; unsafe prices are unrepresentable",
     "The cost floor constructs the action set — unsafe prices are unrepresentable"),
    ("A finance owner can sign an exploration budget; nobody can sign an ε",
     "A finance owner can sign an exploration budget. Nobody can sign an ε"),
    ("Behaviour is explained rather than observed — and the bar names what must change",
     "Behaviour is explained, not just observed — and the bar names what must change"),
])

# 40 · DECISIONS NEEDED -- keep the recommendation and the one fact behind it
k.sub_in_slide(f(40), [
    ("Three thresholds block launch — recommendations, with evidence",
     "Three thresholds block launch — and the floor each clears"),
    ("Measured on nine real 2-week blocks: 6.75% detectable, so 7.5% is met with 0.75pp to spare. Thin against the target — but the target is ours to choose and the measured policy effect is 38%, a 5.6× margin. Duration barely helps: six weeks only reaches 5.74% where √T promised 3.90%, and past six weeks the block count collapses to 1–3.",
     "Nine real 2-week blocks measure 6.75% detectable. Thin against the 7.5% target — but the target is ours to choose, and against a measured 38% effect that is a 5.6× margin. Duration does not buy it back: six weeks reaches only 5.74% where √T promised 3.90%."),
    ("Now a real guardrail. Once the floor was measured on the same smoothing AND the same population the monitor triggers on, it fell from an unusable 480% to a binding 39.6% (control-arm basis; trailing 38.9% raw, 40.1% robust, not outlier-dominated, mean scrap level 13.3%). 50% clears the floor by 26% and stays well inside the 3× line past which a guardrail cannot fire at all.",
     "Now a real guardrail. Measuring the floor on the same smoothing and the same population the monitor triggers on took it from an unusable 480% to a binding 39.6%. 50% clears that by 26% and stays inside the 3× line past which a guardrail cannot fire at all."),
    ("Well behaved throughout: binding 3σ floor 13.94% on the trailing basis, not outlier-dominated. 15% clears it by 7.6%, so the 2-day persistence rule is covering the gap rather than headroom — implemented and tested, not an aspiration. Guardrails suspend exploration only; pricing continues.",
     "Well behaved throughout: binding 3σ floor 13.94%, not outlier-dominated. 15% clears it by 7.6%, so the 2-day persistence rule is covering the gap rather than headroom. A breach suspends exploration only — pricing continues."),
])

# 20 · THE DECISION CORE -- the recursion as the solver actually evaluates it.
# The old line, m(p)·E[min(D,q)] + V(q', t-1), collapsed the demand distribution
# to its mean and then carried ONE next state. The solver does neither: it sums
# over every sales count, and V is non-linear in the leftover, so the mean-field
# form is not equal to it. It also hid the exact step this deck is asked about
# most -- where dispersion enters.
k.Slide(f(20)).paras("Text 6", [
    "V(q, 0)   =  − cost × q     ← scrap the rest",
    "Q(q,t,p)  =  Σₖ P(D=k)·[ m(p)·min(k,q)",
    "             + V(q−min(k,q), t−1) ]",
    "V(q, t)   =  max over feasible p of Q(q,t,p)",
]).save()
k.sub_in_slide(f(20), [
    ("Demand enters as the censored expectation E[min(D, q)], never the raw mean. At a median starting inventory of two units the two differ substantially, and every place that confused them produced a real bug.",
     "Demand enters as a distribution, never an average. The sum runs over every sales count, so censoring — an hour cannot sell more than the shelf holds — and the non-linear value of the leftover both sit inside it, rather than being applied to a mean afterwards."),
])

# ================================================================ new slides
N = {}
for key, tmpl in [("results", f(2)), ("divider", f(13)), ("index", f(12))]:
    N[key] = k.dup(tmpl)

# --- 11 · Headline results ------------------------------- [T2: four figures]
k.Slide(N["results"]) \
 .runs("Text 2", ["WHERE WE ARE"]) \
 .runs("Text 3", ["Four numbers, and what each one is evidence of"]) \
 .paras("Text 4", [
     "The −38% is a like-for-like replay: both policies run through the same frozen "
     "demand model, so model error hits both arms and cancels. It is a strong sanity "
     "check on the planner — it is not, on its own, evidence that the policy wins in "
     "the market. Only the A/B can say that.",
     "The calibration gate is the blocking check on that frozen model: is it accurate "
     "at the reference price, measured on a window disjoint from the one its own "
     "correction was fitted on. 1.0389 inside a [0.90, 1.10] band.",
     "Shadow ran the full production decision path against real state for 3,000 "
     "episodes and applied no prices: 18,846 decisions, complete and matched events, "
     "zero cost-floor violations, p95 latency 0.09 s.",
     "6.75% is measured on nine real two-week blocks of history, not extrapolated by "
     "√T. Everything standing behind these four numbers is in the appendix, indexed "
     "on slide 19.",
 ]) \
 .runs("Text 6", ["−38.0%"]) \
 .runs("Text 7", ["Inventory Loss vs legacy, like-for-like — won by discounting less, not by clearing more"]) \
 .runs("Text 9", ["1.0389"]) \
 .runs("Text 10", ["calibration gate on a held-out window, band [0.90, 1.10] — PASS"]) \
 .runs("Text 12", ["100%"]) \
 .runs("Text 13", ["shadow event completeness and matched rate, zero cost-floor violations — PASS"]) \
 .runs("Text 15", ["6.75%"]) \
 .runs("Text 16", ["effect the A/B can detect in two weeks, against a measured 38%"]) \
 .colour("Text 6", "B3402A", "2C5F2D") \
 .save()

# --- 18 · Appendix divider ------------------------------- [T13: four blocks]
k.Slide(N["divider"]) \
 .runs("Text 2", ["APPENDIX"]) \
 .runs("Text 3", ["Everything else, kept and indexed"]) \
 .runs("Text 6", ["Why it is back here"]) \
 .runs("Text 7",
       ["The seventeen slides above are the argument. The twenty-five that follow the "
        "index are the evidence for it: the data nuances, the mechanics of each component, "
        "the measurement discipline, and the alternatives considered and declined. "
        "Nothing has been removed — it has been moved out of the path."]) \
 .runs("Text 10", ["How to use it"]) \
 .runs("Text 11",
       ["The next slide indexes the appendix by question — “is the data trustworthy”, "
        "“how does it choose a price”, “why not reinforcement learning”. Each group "
        "names its slide range, so a question from the room is one jump rather than a "
        "hunt."]) \
 .runs("Text 14", ["What is not here"]) \
 .runs("Text 15",
       ["docs/design.md is the standalone reference and goes deeper than any slide "
        "can: the identification argument in full, every component's rationale, the "
        "nine-row risk register, and the open owner decisions with the evidence "
        "behind each recommendation."]) \
 .runs("Text 18", ["Every number here is regenerable"]) \
 .runs("Text 19",
       ["Every figure in this deck comes from a report artifact and is tagged with the "
        "slide that carries it. A pipeline re-run regenerates all of them, and the "
        "refresh tool refuses any replacement whose target text does not match exactly "
        "once — so a stale number cannot survive a successful-looking update."]) \
 .save()

# --- 19 · Appendix index --------------------------------- [T12: 3×8 table]
k.Slide(N["index"]) \
 .runs("Text 2", ["APPENDIX INDEX"]) \
 .runs("Text 3", ["Jump by the question you are asked"]) \
 .table("Table 0", [
     "Slides", "The question", "What is there",

     "20–22", "Why can’t history answer it?",
     "The obvious shortcut and why it fails; the problem class; the taxonomy against "
     "deep RL, bandits and vendor rules",

     "23–26", "Is the data trustworthy?",
     "Scope; four things the source data does not mean literally; the filter "
     "waterfall; how an episode ends and what is scrap",

     "27–30", "How does the model work?",
     "The versioned architecture; the price-blind demand model; what calibration may "
     "not correct; dispersion and the design effect",

     "31–34", "How does it choose a price?",
     "The dynamic program in three lines; the horizon trap a gate caught; entry vs "
     "hourly action sets; when deepening pays",

     "35–39", "How does it learn?",
     "τ and the affordable set; the censored likelihood; the bounded step and human "
     "gate; the ledger; the two floors on learning speed",

     "40–44", "How do you know it works?",
     "Threshold discipline; validation on production data; the full measured table; "
     "the shadow dress rehearsal; the experiment design",

     "—", "Where is the detail?",
     "docs/design.md — the identification argument, per-component rationale, the risk "
     "register, and the open owner decisions",
 ]) \
 .save()

k.notes(N["results"],
        "The results slide. Lead with -38% but qualify it in the same breath: it is a "
        "like-for-like replay through the SAME frozen demand model, so it grades the "
        "planner, not the market. The other three are the ones that make it "
        "believable -- a blocking calibration gate on a held-out window, a shadow run "
        "of the real decision path that applied nothing, and a detectable effect "
        "measured on real two-week blocks rather than extrapolated. If someone wants "
        "the mechanism behind the -38%, it is the next slide: less discount, not more "
        "clearance.")
k.notes(N["divider"],
        "Transition slide. Say plainly that the argument is finished and everything "
        "from here is on demand. Worth naming why the deck was cut rather than "
        "rewritten: the content was reviewed and is not wrong, it is just more than an "
        "hour holds. If the room has no questions, stop here.")
k.notes(N["index"],
        "Navigation, not content. Do not read it. Its only job is that when someone "
        "asks 'how do you know the data means what you think it means', the answer is "
        "slides 23 to 26 and it takes one jump to get there. The last row is the "
        "escape hatch: anything deeper than a slide lives in docs/design.md.")

# ==================================================================== order
CORE = [
    f(1),                # title
    f(2),                # THE PROBLEM
    f(3),                # THE POLICY TODAY
    f(4),                # THE CORE DIFFICULTY -- prediction vs identification
    f(6),                # THE SYSTEM IN ONE SLIDE
    f(8),                # DESIGN THESIS
    f(10),               # OBJECTIVE & METRIC
    f(12),               # NOVELTY
    f(21),               # SAFETY
    f(25),               # EXPLORATION
    N["results"],     # WHERE WE ARE
    f(34),               # READING THE RESULT
    f(36),               # HOW WE WILL KNOW -- A/B design
    f(38),               # LAUNCH PLAN
    f(39),               # RISK REGISTER
    f(40),               # DECISIONS NEEDED
    f(41),               # close
]

APPENDIX = [
    N["divider"], N["index"],
    f(5), f(7), f(11),              # why not history, what problem class, why not RL
    f(13), f(14), f(15), f(16),     # scope and the data
    f(9), f(17), f(18), f(19),      # architecture and the demand model
    f(20), f(22), f(23), f(24),     # the decision core
    f(26), f(27), f(28), f(29), f(30),  # exploration and learning
    f(31), f(32), f(33), f(35), f(37),  # measurement, validation, shadow, experiment
]

ORDER = CORE + APPENDIX
assert len(ORDER) == len(set(ORDER)), "duplicate slide in ORDER"
assert set(ORDER) == set(k.slide_files()), (
    f"slides not placed: {sorted(set(k.slide_files()) - set(ORDER))}")

# the index quotes slide ranges; a reorder that invalidates them is a defect
INDEX_CLAIMS = {"20–22": [5, 7, 11], "23–26": [13, 14, 15, 16],
                "27–30": [9, 17, 18, 19], "31–34": [20, 22, 23, 24],
                "35–39": [26, 27, 28, 29, 30], "40–44": [31, 32, 33, 35, 37]}
pos = {n: i for i, n in enumerate(ORDER, start=1)}
for rng, files in INDEX_CLAIMS.items():
    lo, hi = (int(x) for x in rng.split("–"))
    got = [pos[f(v2pos)] for v2pos in files]
    assert got == list(range(lo, hi + 1)), f"index says {rng}, slides land at {got}"
assert pos[N["index"]] == 19, "the results slide points at slide 19 for the index"

k.set_order(ORDER)

# page numbers follow the new positions; the title and the close carry none
for p, n in enumerate(ORDER, start=1):
    if n in (f(1), f(41)):
        continue
    k.Slide(n).runs("Text 1", [str(p)]).save()

k.pack(OUT)
print(f"built {OUT}: {len(CORE)} presented + {len(APPENDIX)} appendix "
      f"= {len(ORDER)} slides")
print("new slides:", N)

# deck_numbers tags slides by v1 position; this is how to find them here.
print("v1 -> v3 position map:")
print("  " + "  ".join(f"{a}->{pos[f(b)]}" for a, b in sorted(V1_TO_V2.items())))
