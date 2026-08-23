"""tools.filter_forensics -- what the chain-break stage actually removed.

    python3 -m tools.filter_forensics --input data/flc.parquet

A waterfall says how much a filter took. It does not say WHAT it took, or
whether it was right to. `chain_break_dropped` is the last big drop left in
the chain, and it drops the WHOLE EPISODE on a single offending hour, so at
any material size the question stops being "is the rule defensible" and
becomes "what population does it select".

This tool answers that. It re-runs the real chain with a read-only probe,
catches the frame entering the stage, and decomposes the casualties with COGS
attached. It changes nothing and writes no artifact.

The stage now drops exactly two things, and they mean different things:

  UNEXPLAINED SHORTFALL   `0 < ending < starting - sold`. Stock left without
                          being sold and without being written off. Shrink is
                          roughly independent of velocity; skew between a
                          transaction feed and a stock snapshot is NOT -- it
                          grows with `units_sold`. `shortfall_vs_sold_corr`
                          separates them, and they need opposite responses:
                          shrink is a real event to NAME, skew is a join
                          defect to fix upstream.

  CHAIN DISCONTINUITY     `ending[t] != starting[t+1]`. The two fields
                          disagree about the same instant, which no business
                          event explains. Nothing tested this before the
                          source convention was understood -- restock was
                          INFERRED from the next hour's opening, which quietly
                          assumed the continuity it should have checked.

It also prices what the whole-episode rule costs on top of the defect itself.
If most casualties break on one hour out of many, the scoping is doing more
damage than the defect, and survival falls with episode length -- which
selects against exactly the long, heavily-stocked windows markdown is for.

Everything is vectorised, and the hour classifier is `common.episodes`
`hour_status` -- the same one the chain filters on, so this cannot drift into
measuring a different rule than the one in force.
"""

import argparse
import json

import numpy as np
import pandas as pd

from bootstrap.prepare_data import cogs_at_risk
from common import episodes
from common.config import load_config

CHAIN = "chain_break_dropped"


def _slice(d, ids):
    return d[d.episode_id.isin(ids)]


def _money(d, ids, raw_cogs):
    sub = _slice(d, ids)
    c = cogs_at_risk(sub) if len(sub) else 0.0
    return {"episodes": int(len(ids)), "rows": int(len(sub)),
            "cogs_at_risk": round(c, 1),
            "pct_of_raw_cogs": round(c / raw_cogs, 6) if raw_cogs else 0.0}


# ----------------------------------------------------------------- chain break

def chain_break(before, raw_cogs):
    """The two things the stage drops, told apart and priced."""
    before = before.sort_values(["sku_id", "fc", "date", "hour_of_day"])
    start = before.starting_inventory.to_numpy()
    sold = before.units_sold.to_numpy()
    ending = before.ending_inventory.to_numpy()
    net = start - sold

    status = episodes.hour_status(start, sold, ending)
    short = status == episodes.SHORTFALL
    disc = episodes.continuity_breaks(before)
    broken = short | disc
    if not broken.any():
        return {"episodes": 0, "note": "no episode trips this stage"}

    hit = before.episode_id.isin(before.loc[broken, "episode_id"].unique())
    ids = before.loc[hit, "episode_id"].unique()
    out = {"total": _money(before, ids, raw_cogs)}

    # what the whole population looks like under the source's convention. Any
    # movement in these four is the headline: they are what "clean" means now.
    n = max(len(before), 1)
    out["hour_status"] = {
        k: {"rows": int((status == k).sum()),
            "share": round(float((status == k).sum()) / n, 6)}
        for k in (episodes.RECONCILES, episodes.RESTOCK,
                  episodes.WRITE_OFF, episodes.SHORTFALL)}

    # WHICH of the two causes, and do they overlap? A shortfall usually breaks
    # continuity too -- the next hour opens from a figure this hour disputes --
    # so the overlap says whether they are one defect or two.
    out["cause"] = {
        "rows_unexplained_shortfall": int(short.sum()),
        "rows_chain_discontinuous": int(disc.sum()),
        "rows_both": int((short & disc).sum()),
        "episodes_shortfall_only": int(
            len(set(before.loc[short, "episode_id"])
                - set(before.loc[disc, "episode_id"]))),
        "episodes_discontinuity_only": int(
            len(set(before.loc[disc, "episode_id"])
                - set(before.loc[short, "episode_id"]))),
        "cogs_shortfall": _money(
            before, before.loc[short, "episode_id"].unique(), raw_cogs),
        "cogs_discontinuity": _money(
            before, before.loc[disc, "episode_id"].unique(), raw_cogs),
    }

    # SHRINK OR SKEW. Shrink is roughly velocity-independent; a join between a
    # transaction feed and a stock snapshot is not. This is the discriminator.
    gap = (net - ending)[short]
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
            gap / np.clip(net[short], 1, None))), 4) if len(gap) else 0.0,
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
        "shortfall_named_and_flagged": _money(
            before, before.loc[short & ~disc, "episode_id"].unique(), raw_cogs),
        "tolerance_1_unit": _money(
            before, before.loc[short & ~disc & (net - ending == 1),
                               "episode_id"].unique(), raw_cogs),
        "tolerance_2_units": _money(
            before, before.loc[short & ~disc & (net - ending <= 2),
                               "episode_id"].unique(), raw_cogs),
        "note": ("COGS that returns to the INTEGRITY population under each "
                 "candidate rule. Only episodes whose ONLY defect is the "
                 "shortfall are counted -- one that also breaks continuity "
                 "would still be dropped. A tolerance is the weakest option "
                 "and is priced here only so the stronger ones have a floor "
                 "to beat."),
    }

    # the censoring rule's own blind spot, on the same population
    out["censoring_edge_cases"] = episodes.censoring_edge_cases(
        start, sold, ending)
    return out


# -------------------------------------------------------------------- assembly

def run(path, cfg):
    caught = {}

    def probe(label, before, after):
        if label == CHAIN:
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
    for stage in (CHAIN,):
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
