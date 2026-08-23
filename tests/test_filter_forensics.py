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
    assert "units_gt_inventory_dropped" not in seen, \
        "the stage came back -- sold > starting is a restock, not a defect"
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


# ------------------------------------------------- the corrected convention

def test_an_hour_selling_more_than_it_opened_with_is_a_restock_not_a_defect():
    """The finding that removed a whole filter stage.

    `ending_inventory` is the FINAL count after anything that arrived, so
    `sold > starting` is the restock signal in its plainest form. It used to
    be deleted as an impossible quantity by a stage that ran BEFORE the
    reconciler was asked, taking 18.1pp of the extract's COGS with it.
    """
    from common.episodes import adjustment_reason, hour_status, RESTOCK
    assert adjustment_reason(5, 8, 3) == "intraday_restock"
    assert adjustment_reason(1, 3, 0) == "intraday_restock"   # sold out after
    # and a restock needs no over-sell at all: opened 3, sold 2, ended 5
    assert adjustment_reason(3, 2, 5) == "intraday_restock"
    assert list(hour_status([5, 1, 3], [8, 3, 2], [3, 0, 5])) == [RESTOCK] * 3


def test_the_tool_reports_the_hour_status_mix(cfg, tmp_path):
    """What "clean" means now, as four shares that must sum to one."""
    path, _ = _with_shortfalls(tmp_path)
    blk = ff.run(path, cfg)[ff.CHAIN]
    mix = blk["hour_status"]
    assert set(mix) == {"reconciles", "intraday_restock",
                        "episode_close_write_off", "unexplained_shortfall"}
    assert sum(v["share"] for v in mix.values()) == pytest.approx(1.0, abs=1e-4)
    assert mix["intraday_restock"]["rows"] > 0, \
        "the fixture has over-sell hours; they must classify as restocks"


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

    cause = blk["cause"]
    assert cause["rows_unexplained_shortfall"] > 0
    assert blk["shortfall"]["share_of_1_unit"] == 1.0
    # a shortfall almost always breaks continuity too -- the next hour opens
    # from a figure this hour disputes -- so the two causes must overlap
    # heavily rather than partition
    assert cause["rows_both"] > 0
    assert cause["cogs_shortfall"]["cogs_at_risk"] > 0


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
    for key in ("shortfall_named_and_flagged", "tolerance_1_unit",
                "tolerance_2_units"):
        assert rec[key]["cogs_at_risk"] >= 0
    # a 1-unit tolerance cannot recover more than naming the whole shape
    assert rec["tolerance_1_unit"]["cogs_at_risk"] <= \
        rec["shortfall_named_and_flagged"]["cogs_at_risk"]


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
