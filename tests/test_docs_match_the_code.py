"""The docs describe a filter chain. This checks it is THIS filter chain.

Prose rots quietly. Over one long session the chain gained a stage, lost two,
and reclassified four flags -- and `AGENTS.md` kept up while `design.md`
was left describing `restocked` as a DP gate months after it stopped
being one. Nobody reading it would have known.

So the names are cross-checked against the code that defines them. This cannot
verify that the PROSE is true -- only a human can -- but it can guarantee that
every stage and every flag is at least MENTIONED wherever the chain is
documented, and that nothing removed is still being described as live. Those
are the two failures that actually happened.

`tools.metrics_glossary` has its own cross-check for the report fields; this is
the same idea for the population chain.
"""

import pathlib

import pytest

from bootstrap.prepare_data import DP_INELIGIBLE, load_and_filter
from common.config import load_config

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every doc that describes the filter chain and therefore has to keep up. A
# doc that merely mentions the pipeline is not listed -- the point is to catch
# the ones a reader would trust for the chain itself. AGENTS.md is no longer
# listed: it is a router now, pointing at design.md for the chain.
CHAIN_DOCS = ["docs/design.md"]

# Names the chain USED to carry. Any of these appearing as a live stage or
# flag means a doc is describing a rule that no longer exists.
RETIRED = [
    "units_gt_inventory_dropped",
    "chain_break_dropped",
    "below_cost_dropped",
    "non_priceable_dropped",
    "window_too_long_dropped",
    "negative_window_dropped",
    "restocked_episodes_dropped",
    "edge_truncated_episodes_dropped",
    "cost_missing_dropped",
    "drop_edge_truncated_episodes",
    "filter_forensics",
]


@pytest.fixture(scope="module")
def stages():
    _, wf = load_and_filter(str(ROOT / "data" / "flc_synth.parquet"),
                            load_config(str(ROOT / "config.yaml")))
    return [t[0] for t in wf]


@pytest.fixture(scope="module")
def docs():
    return {f: (ROOT / f).read_text() for f in CHAIN_DOCS}


def test_every_waterfall_stage_is_documented(stages, docs):
    missing = [(f, s) for f in CHAIN_DOCS for s in stages
               if s != "raw" and s not in docs[f]]
    assert not missing, (
        "waterfall stages absent from a doc that describes the chain: "
        + ", ".join(f"{s} not in {f}" for f, s in missing))


def test_every_dp_ineligible_reason_is_documented(docs):
    names = [n for n, _ in DP_INELIGIBLE]
    missing = [(f, n) for f in CHAIN_DOCS for n in names if n not in docs[f]]
    assert not missing, (
        "dp_eligible gates absent from a doc: "
        + ", ".join(f"{n} not in {f}" for f, n in missing))


def test_the_reported_only_flags_are_documented(docs):
    """These gate nothing, which is exactly why they need saying out loud --
    a reader has no other way to learn that a restock does NOT exclude an
    episode from the backtest."""
    for flag in ("below_cost_hours", "edge_truncated", "restocked", "shrink"):
        missing = [f for f in CHAIN_DOCS if flag not in docs[f]]
        assert not missing, f"{flag} is undocumented in {missing}"


def test_no_doc_still_describes_a_retired_rule(stages, docs):
    """The failure that actually happened: `design.md` went on calling
    `restocked` a DP gate long after it stopped being one."""
    live = set(stages) | {n for n, _ in DP_INELIGIBLE}
    stale = [(f, name) for f in CHAIN_DOCS for name in RETIRED
             if name in docs[f] and name not in live]
    # a doc MAY name a retired rule while explaining that it was retired, so
    # the test allows it only next to a word that marks it as history
    unexplained = []
    for f, name in stale:
        # skip fenced code blocks: a doc may keep a historical listing on
        # purpose, with a note above it saying what is superseded
        fenced, body = False, []
        for line in docs[f].split("\n"):
            if line.startswith("```"):
                fenced = not fenced
                continue
            if not fenced:
                body.append(line)
        for line in body:
            if name not in line:
                continue
            if not any(w in line.lower() for w in
                       ("used to", "no longer", "retired", "removed", "was ",
                        "superseded", "stopped", "former", "since ", "not ",
                        "deleted", "reclassif")):
                unexplained.append(f"{name} in {f}: {line[:90]}")
    assert not unexplained, (
        "a doc describes a retired rule as if it were live:\n  "
        + "\n  ".join(unexplained))


def test_the_three_populations_are_named_everywhere(docs):
    for pop in ("integrity", "eligible", "dp_eligible"):
        missing = [f for f in CHAIN_DOCS if pop not in docs[f]]
        assert not missing, f"population {pop!r} undocumented in {missing}"


def test_the_episode_identity_is_stated_in_every_chain_doc(docs):
    """One line, and everything downstream is arithmetic on it. If a doc
    describes the chain without it, the reader cannot check any figure."""
    for f in CHAIN_DOCS:
        t = docs[f]
        assert "opening + restocked" in t and "sold + scrap" in t.replace(
            "sold + shrink + leftover_at_last_hour", "sold + scrap"), \
            f"{f} describes the chain without stating the episode identity"


def test_agents_md_stays_a_router_not_a_reference():
    """AGENTS.md was 1,564 lines and small-model agents missed instructions
    buried mid-file. It is a router now: one-line non-negotiables up front,
    pointers to design.md for everything else. This budget is what keeps it
    that way -- new material goes to design.md (spec), learnings.md
    (history), or docs/maintaining_docs.md (doc tooling), with at most a
    one-liner and a pointer here."""
    text = (ROOT / "AGENTS.md").read_text()
    lines = text.count("\n") + 1
    assert lines <= 400, (
        f"AGENTS.md is {lines} lines, over the 400-line router budget -- "
        "move the new content to its reference home and point to it")
    # the non-negotiables must stay at the top, ahead of everything else
    assert text.index("## Non-negotiables") < text.index("## Setup")


def test_the_pipeline_table_matches_the_executable_script():
    """AGENTS.md's step table and run_bootstrap.sh are two copies of one
    order, and they silently drifted: the table left calibration at step 7
    (after backtest) while the script had long moved it to 3b (before prior,
    because prior and dispersion are fitted against CALIBRATED mu_ref). An
    agent following the table would run --check-convergence before
    calibration.json existed. The script is the authority; this pins the
    table to it on the steps whose ORDER carries a dependency."""
    import re

    text = (ROOT / "AGENTS.md").read_text()
    script = (ROOT / "scripts" / "run_bootstrap.sh").read_text()

    def table_pos(token):
        i = text.find("## Pipeline order")
        j = text.find(token, i)
        assert j > 0, f"{token} missing from the AGENTS.md pipeline table"
        return j

    def script_pos(token):
        j = script.find(token)
        assert j > 0, f"{token} missing from run_bootstrap.sh"
        return j

    # the dependency chain: calibration -> prior -> dispersion -> convergence.
    # Matched on INVOCATIONS, not mentions: the script explains the ordering
    # in a comment that names later steps before it runs them.
    table_chain = ["--fit-calibration", "estimate_prior", "fit_dispersion",
                   "--check-convergence"]
    script_chain = ["--fit-calibration",
                    "python3 -m bootstrap.estimate_prior",
                    "python3 -m bootstrap.fit_dispersion",
                    "--check-convergence"]
    for where, pos, chain in (("AGENTS.md", table_pos, table_chain),
                              ("run_bootstrap.sh", script_pos, script_chain)):
        offsets = [pos(t) for t in chain]
        assert offsets == sorted(offsets), (
            f"{where} orders the calibration/prior/dispersion/convergence "
            f"chain wrongly: {chain} must appear in that order")

    # and the table must not still promise a post-backtest calibration step
    tail = text[table_pos("--fit-calibration"):table_pos("--check-convergence")]
    assert "backtest" not in tail, \
        "calibration must precede backtest in the table, as it does in the script"
    # the step label the script uses for it
    assert re.search(r"3b\.\s+bootstrap\.train_baseline --fit-calibration", text)
