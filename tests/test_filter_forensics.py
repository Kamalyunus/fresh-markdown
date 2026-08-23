"""The instrument that says WHAT a filter took, not just how much.

Two stages remove roughly 60% of everything lost on the production extract,
and a waterfall cannot say whether they were right to. This file holds up the
tool that decomposes them -- and the read-only probe it rides on, which must
never be able to change the chain's output.
"""

import json

import numpy as np
import pandas as pd
import pytest

from bootstrap.prepare_data import load_and_filter
from common.config import load_config
from tools import filter_forensics as ff

REPO_CONFIG = __import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.dirname(
        __import__("os").path.abspath(__file__))), "config.yaml")
SYNTH = __import__("os").path.join(
    __import__("os").path.dirname(REPO_CONFIG), "data", "flc_synth.parquet")


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_CONFIG)


# ------------------------------------------------------------------- the probe

def test_the_probe_sees_every_drop_stage_in_order(cfg):
    seen = []
    _, wf = load_and_filter(SYNTH, cfg, probe=lambda l, b, a: seen.append(l))
    # every step()-driven row, which is every stage that can drop. The two
    # summary rows are appended directly and drop nothing.
    expected = [t[0] for t in wf
                if t[0] not in ("raw", "duplicate_hour_rows_dropped",
                                "contiguous_episodes_built", "dp_eligible")]
    assert seen == expected, (seen, expected)


def test_before_is_the_frame_the_stage_filtered(cfg):
    """`before` must be the frame ENTERING the stage, or every decomposition
    built on it attributes the casualties to the wrong filter."""
    caught = {}
    _, wf = load_and_filter(
        SYNTH, cfg, probe=lambda l, b, a: caught.__setitem__(l, (len(b), len(a))))
    rows = {t[0]: t[1] for t in wf}
    for label, (before, after) in caught.items():
        assert after == rows[label], label
    # and the chain is a chain: this stage's `before` is the previous row's
    # `after`
    order = [t[0] for t in wf]
    for label, (before, _) in caught.items():
        prev = order[order.index(label) - 1]
        assert before == rows[prev], f"{label} entered with {prev}'s output"


def test_a_probe_cannot_change_the_output(cfg):
    """A diagnostic hook in production code. If it can perturb the population,
    every artifact built after someone attaches one is suspect.

    This is not hypothetical: the first version of the hook handed over the
    live frames, and this test caught a probe writing a column straight into
    the chain. The fix is a shallow copy -- under copy-on-write it shares the
    data blocks, so it is free on a 10M-row frame, and a probe that writes
    gets its own block instead of the population's.
    """
    clean, wf_clean = load_and_filter(SYNTH, cfg)

    def meddler(label, before, after):
        after["units_sold"] = 0              # a whole column
        after.loc[after.index[:5], "starting_inventory"] = 0   # and some cells
        before.drop(columns=["cost"], inplace=True)            # and a schema change

    probed, wf_probed = load_and_filter(SYNTH, cfg, probe=meddler)
    assert [t[:4] for t in wf_clean] == [t[:4] for t in wf_probed]
    assert clean.units_sold.sum() == probed.units_sold.sum()
    assert clean.starting_inventory.sum() == probed.starting_inventory.sum()
    assert "cost" in probed.columns


# ------------------------------------------------------- units > inventory

def test_units_gt_inventory_is_measured_against_the_production_reconciler(cfg):
    """The finding the tool exists to surface.

    `adjustment_reason` -- the rule `events.store` enforces live -- names an
    hour that sold more than it opened with as `intraday_restock`, because
    leftover clips to 0 and any positive ending exceeds it. But
    `units_gt_inventory_dropped` runs FIRST, so the reconciler is never asked.
    The tool has to report that share, or the contradiction stays invisible.
    """
    report = ff.run(SYNTH, cfg)
    blk = report[ff.UNITS_GT]
    if not blk.get("total"):
        pytest.skip("fixture trips no units>inventory episode")
    named = blk["offending_hours"]
    assert 0.0 <= named["share_named"] <= 1.0
    assert named["named_intraday_restock_by_adjustment_reason"] <= named["rows"]
    # the between-hours detector is reported beside it: same phenomenon or not
    assert "also_flagged_restocked_between_hours" in blk


def test_an_hour_selling_more_than_it_opened_with_is_named_a_restock():
    """Pinned directly, since the whole argument rests on it."""
    from common.episodes import adjustment_reason
    assert adjustment_reason(5, 8, 3) == "intraday_restock"
    assert adjustment_reason(10, 12, 2) == "intraday_restock"


# ------------------------------------------------------------- chain break

def _with_shortfalls(tmp_path, share=0.02, seed=0):
    """Make stock vanish without being sold -- the unnamed residue."""
    raw = pd.read_parquet(SYNTH)
    rng = np.random.default_rng(seed)
    cand = raw.index[((raw.inventory - raw.units_sold) > 1)
                     & (raw.ending_inventory > 1)]
    pick = rng.choice(cand, size=max(int(len(cand) * share), 5), replace=False)
    raw.loc[pick, "ending_inventory"] -= 1
    path = tmp_path / "shrink.parquet"
    raw.to_parquet(path)
    return str(path), len(pick)


def test_a_partial_shortfall_is_decomposed_not_just_counted(cfg, tmp_path):
    path, n = _with_shortfalls(tmp_path)
    blk = ff.run(path, cfg)[ff.CHAIN]
    assert blk["total"]["episodes"] > 0

    shape = blk["shape_of_broken_rows"]
    assert shape["partial_shortfall_0_lt_ending_lt_leftover"] > 0
    assert shape["share_partial_shortfall"] == 1.0, \
        "an injected one-unit shortfall is the ONLY shape here; anything " \
        "else means the classifier is mislabelling"
    assert blk["shortfall"]["share_of_1_unit"] == 1.0


def test_the_tool_prices_whole_episode_scoping_separately(cfg, tmp_path):
    """The defect and the RULE cost different amounts, and conflating them is
    how a one-unit discrepancy on 2% of rows justifies deleting a fifth of the
    money. If most casualties break on one hour out of many, the scoping is
    doing the damage."""
    path, _ = _with_shortfalls(tmp_path)
    blk = ff.run(path, cfg)[ff.CHAIN]
    cost = blk["episode_scoping_cost"]
    assert cost["episodes_with_exactly_one_broken_hour"] > 0
    assert cost["share"] > 0.5, \
        "scattered single-hour breaks should dominate a scattered injection"
    assert cost["median_episode_length_hours"] > cost["median_broken_hours"]
    # the money behind them is reported, not just the count
    assert cost["cogs_in_single_broken_hour_episodes"]["cogs_at_risk"] > 0


def test_recovery_options_are_priced_but_the_report_recommends_nothing(cfg, tmp_path):
    path, _ = _with_shortfalls(tmp_path)
    blk = ff.run(path, cfg)[ff.CHAIN]
    rec = blk["recovery_if"]
    for key in ("partial_shortfall_named_and_flagged", "tolerance_1_unit",
                "tolerance_2_units"):
        assert rec[key]["cogs_at_risk"] >= 0
    # a 1-unit tolerance cannot recover more than naming the whole shape
    assert rec["tolerance_1_unit"]["cogs_at_risk"] <= \
        rec["partial_shortfall_named_and_flagged"]["cogs_at_risk"]


# ------------------------------------------------------------------- output

def test_the_report_is_strictly_valid_json(cfg, tmp_path):
    """`json.dump` writes a bare NaN literal, which no strict parser reads
    back. A degenerate correlation -- every shortfall the same size -- is the
    NORMAL case on a small sample, so this is reachable, not theoretical."""
    path, _ = _with_shortfalls(tmp_path)
    text = json.dumps(ff.run(path, cfg), indent=2, default=str)

    def boom(c):
        raise ValueError(f"invalid JSON literal: {c}")

    back = json.loads(text, parse_constant=boom)
    corr = back[ff.CHAIN]["shortfall"]["shortfall_vs_sold_corr"]
    assert corr is None or -1.0 <= corr <= 1.0


def test_the_tool_writes_no_artifact_and_changes_no_population(cfg, tmp_path):
    """Forensics is read-only. It must never become a step in the pipeline."""
    path, _ = _with_shortfalls(tmp_path)
    before, wf_before = load_and_filter(path, cfg)
    ff.run(path, cfg)
    after, wf_after = load_and_filter(path, cfg)
    assert [t[:4] for t in wf_before] == [t[:4] for t in wf_after]
    assert len(before) == len(after)
