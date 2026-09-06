"""The docs describe a filter chain. This checks it is THIS filter chain."""

import os
import pathlib

import pytest

from fit.prepare_data import DP_INELIGIBLE, load_and_filter
from common.config import load_config
from conftest import ROOT

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
def stages(synth_flc):
    _, wf = load_and_filter(synth_flc, load_config(os.path.join(ROOT, "config.yaml")))
    return [t[0] for t in wf]


@pytest.fixture(scope="module")
def docs():
    return {f: pathlib.Path(ROOT, f).read_text() for f in CHAIN_DOCS}


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
    (history), with at most a
    one-liner and a pointer here."""
    text = pathlib.Path(ROOT, "AGENTS.md").read_text()
    lines = text.count("\n") + 1
    assert lines <= 400, (
        f"AGENTS.md is {lines} lines, over the 400-line router budget -- "
        "move the new content to its reference home and point to it")
    # the non-negotiables must stay at the top, ahead of everything else
    assert text.index("## Non-negotiables") < text.index("## Setup")




def test_every_sim_config_key_is_in_the_design_table():
    """pilot_sim.yaml is the simulator's settings file; design 11.3 carries
    the key table an operator reads. A key in the file the table does not
    name is an undocumented knob."""
    import yaml
    from conftest import ROOT

    with open(os.path.join(ROOT, "pilot_sim.yaml")) as f:
        sim = yaml.safe_load(f)
    with open(os.path.join(ROOT, "docs", "design.md")) as f:
        design = f.read()
    table = design[design.index("### 11.3"):design.index("## 12.")]
    keys = [k for section in ("run", "world", "paths", "grading") for k in sim[section]] + ["faults"]
    missing = [k for k in keys if f"`{k}`" not in table]
    assert not missing, missing
    # and the loader's own key list is the file's
    from evaluate.pilot_sim import SIM_KEYS
    assert {k for ks in SIM_KEYS.values() for k in ks} == set(keys) - {"faults"}
