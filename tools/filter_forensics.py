"""tools.filter_forensics -- what the two big drop stages actually removed.

    python3 -m tools.filter_forensics --input data/flc.parquet

A waterfall says how much a filter took. It does not say WHAT it took, or
whether it was right to. On the production extract two stages between them
account for roughly 60% of everything lost -- `units_gt_inventory_dropped`
(18.1pp of COGS) and `chain_break_dropped` (33.6pp) -- and both drop the WHOLE
EPISODE on a single offending hour. At that size the question stops being "is
the rule defensible" and becomes "what population does it select".

This tool answers that. It re-runs the real chain with a read-only probe,
catches the frame entering each stage, and decomposes the casualties into
named causes with COGS attached. It changes nothing and writes no artifact.

Three things it is built to test, each of which implies a different fix:

  1. IS `units_gt_inventory` A RESTOCK? `common.episodes.adjustment_reason`
     -- the same function `events.store` enforces in production -- classifies
     `start=5, sold=8, ending=3` as `intraday_restock`. But
     `units_gt_inventory_dropped` runs FIRST and deletes the episode as an
     impossible quantity, so the reconciler never sees it. If most of the
     dropped episodes would have been named, the stage is contradicting the
     project's own rule and the ordering is hiding it.

  2. IS THE CHAIN BREAK SHRINK, OR TIMING SKEW? The unnamed residue is a
     partial shortfall, `0 < ending < leftover`: stock left without being
     sold. Shrink is roughly independent of velocity. Skew between a
     transaction feed and a stock snapshot is NOT -- it grows with
     `units_sold`. `shortfall_vs_sold_corr` separates them. Skew is a join
     defect to fix; shrink is a real event to name.

  3. HOW MUCH IS THE WHOLE-EPISODE RULE COSTING? If most casualties break on
     one hour out of many, the episode-scoping is doing more damage than the
     defect, and the survival rate falls with episode length -- which selects
     against exactly the long, fast, heavily-stocked windows the markdown
     system exists for.

Everything is vectorised: the production chain's `adjustment_reason` call is a
Python loop over every row, which is fine once but not for a decomposition.
"""

import argparse
import json

import numpy as np
import pandas as pd

from bootstrap.prepare_data import cogs_at_risk, restocked_episodes
from common.config import load_config

UNITS_GT = "units_gt_inventory_dropped"
CHAIN = "chain_break_dropped"


def _reasons(start, sold, ending):
    """`common.episodes.adjustment_reason`, vectorised. Same rule, same order."""
    leftover = np.clip(start - sold, 0, None)
    restock = ending > leftover
    write_off = (ending == 0) & (leftover > 0)
    return leftover, restock, write_off


def _slice(d, ids):
    return d[d.episode_id.isin(ids)]


def _money(d, ids, raw_cogs):
    sub = _slice(d, ids)
    c = cogs_at_risk(sub) if len(sub) else 0.0
    return {"episodes": int(len(ids)), "rows": int(len(sub)),
            "cogs_at_risk": round(c, 1),
            "pct_of_raw_cogs": round(c / raw_cogs, 6) if raw_cogs else 0.0}


# --------------------------------------------------------------- units > stock

def units_gt_inventory(before, raw_cogs):
    """An hour selling more than it opened with. Impossible, or a restock?"""
    start = before.starting_inventory.to_numpy()
    sold = before.units_sold.to_numpy()
    ending = before.ending_inventory.to_numpy()
    over = sold > start
    if not over.any():
        return {"episodes": 0, "note": "no episode trips this stage"}

    hit = before.episode_id.isin(before.loc[over, "episode_id"].unique())
    ids = before.loc[hit, "episode_id"].unique()
    out = {"total": _money(before, ids, raw_cogs)}

    _, restock, _ = _reasons(start, sold, ending)
    # the central question: on the offending hours themselves, would the
    # production reconciler have NAMED this rather than called it impossible?
    named = restock[over]
    out["offending_hours"] = {
        "rows": int(over.sum()),
        "named_intraday_restock_by_adjustment_reason": int(named.sum()),
        "share_named": round(float(named.mean()), 4),
        "note": ("adjustment_reason(start, sold, ending) with sold > start "
                 "gives leftover = 0, so any ending > 0 is already a "
                 "DOCUMENTED restock. This stage runs first and deletes the "
                 "episode before the reconciler is asked."),
    }

    # is the overage a rounding-scale nuisance or a real inventory jump?
    amt = (sold - start)[over]
    out["overage_units"] = {
        "p50": float(np.median(amt)), "p90": float(np.percentile(amt, 90)),
        "max": int(amt.max()),
        "share_of_1_unit": round(float((amt == 1).mean()), 4),
        "share_le_2_units": round(float((amt <= 2).mean()), 4),
    }

    # does the BETWEEN-hours detector independently agree stock arrived? if so
    # the two tests are one phenomenon and only the labels differ
    also = set(restocked_episodes(_slice(before, ids)))
    out["also_flagged_restocked_between_hours"] = {
        "episodes": len(also),
        "share": round(len(also) / max(len(ids), 1), 4),
    }

    # one bad hour, or a broken window?
    per_ep = before.loc[over].groupby("episode_id").size()
    length = before.loc[hit].groupby("episode_id").size()
    out["episode_scoping_cost"] = {
        "episodes_with_exactly_one_offending_hour": int((per_ep == 1).sum()),
        "share": round(float((per_ep == 1).mean()), 4),
        "median_episode_length_hours": float(length.median()),
        "median_offending_hours": float(per_ep.median()),
    }
    return out


# ----------------------------------------------------------------- chain break

def chain_break(before, raw_cogs):
    """`ending != starting - sold` with no reason `adjustment_reason` names."""
    start = before.starting_inventory.to_numpy()
    sold = before.units_sold.to_numpy()
    ending = before.ending_inventory.to_numpy()

    leftover, restock, write_off = _reasons(start, sold, ending)
    reconciles = (start - sold) == ending
    broken = ~(reconciles | restock | write_off)
    if not broken.any():
        return {"episodes": 0, "note": "no episode trips this stage"}

    hit = before.episode_id.isin(before.loc[broken, "episode_id"].unique())
    ids = before.loc[hit, "episode_id"].unique()
    out = {"total": _money(before, ids, raw_cogs)}

    # SHAPE. The rule names two conventions and leaves one residue: the
    # partial shortfall. Anything else here is a surprise worth seeing.
    short = broken & (ending > 0) & (ending < leftover)
    neg = broken & (ending < 0)
    out["shape_of_broken_rows"] = {
        "rows_broken": int(broken.sum()),
        "partial_shortfall_0_lt_ending_lt_leftover": int(short.sum()),
        "ending_negative": int(neg.sum()),
        "other": int(broken.sum() - short.sum() - neg.sum()),
        "share_partial_shortfall": round(float(short.sum() / broken.sum()), 4),
    }

    # SHRINK OR SKEW. Shrink is roughly velocity-independent; a join between a
    # transaction feed and a stock snapshot is not. This is the discriminator.
    gap = (leftover - ending)[short]
    s_sold = sold[short]
    # None, never NaN: `json.dump` writes a bare `NaN` literal, which no
    # strict parser will read back. A degenerate correlation -- every
    # shortfall the same size -- is the normal case on a small sample.
    corr = None
    if len(gap) > 2 and gap.std() > 0 and s_sold.std() > 0:
        c = np.corrcoef(gap, s_sold)[0, 1]
        corr = round(float(c), 4) if np.isfinite(c) else None
    out["shortfall"] = {
        "units_p50": float(np.median(gap)) if len(gap) else 0.0,
        "units_p90": float(np.percentile(gap, 90)) if len(gap) else 0.0,
        "share_of_1_unit": round(float((gap == 1).mean()), 4) if len(gap) else 0.0,
        "share_le_2_units": round(float((gap <= 2).mean()), 4) if len(gap) else 0.0,
        "share_of_leftover_p50": round(float(np.median(
            gap / np.clip(leftover[short], 1, None))), 4) if len(gap) else 0.0,
        "shortfall_vs_sold_corr": corr,
        "mean_sold_on_broken_rows": round(float(s_sold.mean()), 3) if len(gap) else 0.0,
        "mean_sold_on_clean_rows": round(float(sold[~broken].mean()), 3),
        "reading": ("corr near 0 with small absolute gaps reads as SHRINK -- a "
                    "real event to NAME. corr well above 0, or broken rows "
                    "selling far more than clean ones, reads as TIMING SKEW "
                    "between the sales feed and the stock snapshot -- a join "
                    "defect to fix upstream, not a population to delete."),
    }

    # WHAT THE WHOLE-EPISODE RULE COSTS on top of the defect itself
    per_ep = before.loc[broken].groupby("episode_id").size()
    length = before.loc[hit].groupby("episode_id").size()
    one_hour = per_ep[per_ep == 1].index
    out["episode_scoping_cost"] = {
        "episodes_with_exactly_one_broken_hour": int(len(one_hour)),
        "share": round(float((per_ep == 1).mean()), 4),
        "median_episode_length_hours": float(length.median()),
        "median_broken_hours": float(per_ep.median()),
        "cogs_in_single_broken_hour_episodes": _money(before, one_hour, raw_cogs),
    }

    # SIZE SELECTION -- the reason this matters more than its episode share
    kept = before[~hit]
    n_hit, n_kept = len(ids), kept.episode_id.nunique()
    c_hit = cogs_at_risk(_slice(before, ids))
    c_kept = cogs_at_risk(kept) if n_kept else 0.0
    out["size_selection"] = {
        "cogs_per_dropped_episode": round(c_hit / max(n_hit, 1), 1),
        "cogs_per_kept_episode": round(c_kept / max(n_kept, 1), 1),
        "ratio": round((c_hit / max(n_hit, 1)) / max(c_kept / max(n_kept, 1), 1e-9), 2),
        "reading": ("a ratio well above 1 means the filter is selecting the "
                    "LARGEST episodes. Combined with whole-episode scoping, "
                    "survival falls with window length, which selects against "
                    "exactly the long heavily-stocked windows markdown is for."),
    }

    # WHAT A DIFFERENT RULE WOULD RETURN. Reported, not recommended.
    out["recovery_if"] = {
        "partial_shortfall_named_and_flagged": _money(
            before,
            before.loc[broken & short, "episode_id"].unique(), raw_cogs),
        "tolerance_1_unit": _money(
            before,
            before.loc[broken & (leftover - ending == 1)
                       & short, "episode_id"].unique(), raw_cogs),
        "tolerance_2_units": _money(
            before,
            before.loc[broken & (leftover - ending <= 2)
                       & short, "episode_id"].unique(), raw_cogs),
        "note": ("COGS that returns to the INTEGRITY population under each "
                 "candidate rule. Episode sets overlap with each other, not "
                 "with the survivors. A tolerance is the weakest option and is "
                 "priced here only so the stronger ones have a floor to beat."),
    }
    return out


# -------------------------------------------------------------------- assembly

def run(path, cfg):
    caught = {}

    def probe(label, before, after):
        if label in (UNITS_GT, CHAIN):
            caught[label] = before

    from bootstrap.prepare_data import load_and_filter
    _, wf = load_and_filter(path, cfg, probe=probe)

    by_step = {t[0]: t for t in wf}
    raw_cogs = by_step["raw"][3]
    report = {
        "raw_cogs_at_risk": round(raw_cogs, 1),
        "waterfall_summary": [
            {"step": t[0], "rows": t[1], "episodes": t[2],
             "cogs_pct_of_raw": round(t[3] / raw_cogs, 6) if raw_cogs else None}
            for t in wf],
    }
    if UNITS_GT in caught:
        report[UNITS_GT] = units_gt_inventory(caught[UNITS_GT], raw_cogs)
    if CHAIN in caught:
        report[CHAIN] = chain_break(caught[CHAIN], raw_cogs)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="data/flc.parquet")
    ap.add_argument("--out", default="reports/filter_forensics.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    report = run(args.input, load_config(args.config))

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"raw COGS at risk: {report['raw_cogs_at_risk']:,.0f}\n")
    for stage in (UNITS_GT, CHAIN):
        blk = report.get(stage)
        if not blk or not blk.get("total"):
            continue
        t = blk["total"]
        print(f"{stage}")
        print(f"  removed {t['episodes']:,} episodes, {t['rows']:,} rows, "
              f"{t['pct_of_raw_cogs'] * 100:.2f}pp of raw COGS")
        for key, sub in blk.items():
            if key == "total" or not isinstance(sub, dict):
                continue
            print(f"  {key}")
            for k, v in sub.items():
                if k in ("note", "reading"):
                    continue
                print(f"    {k:52} {v}")
        print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
