"""tools.eda -- what the population looks like, before anything is modelled.

    python3 -m tools.eda --input data/prepared.parquet

Writes `reports/eda.json` (every number) and `docs/eda.html` (the same
numbers, drawn). Reads the prepared parquet and `config.yaml` and nothing
else: no artifacts, no model, no DP. It runs in seconds, which is the point --
a check nobody re-runs is worse than no check.

WHAT THIS IS NOT. `bootstrap.measure` produces the MEASURED config values and
decides the reassessment gates; `pipeline.status` prints what gates a
decision. This decides nothing and produces no config value. Two sources of
truth for `rho` is exactly the failure `artifact_mirror_drift` exists to
catch, so nothing here is ever pasted anywhere.

What makes it more than a notebook: every panel names the config keys or
artifacts it informs, and `tests/test_eda.py` asserts those keys actually
resolve. A renamed key breaks the claim instead of leaving it stale.

Each panel's `chart` block carries the series as data; `docs/eda.html` is a
pure view over it. So the JSON is complete on its own, and the page can never
show a number the report does not contain.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config
from common import episodes

PANELS = []


def panel(key, title, informs=(), lede=""):
    """Register a panel. `informs` is the point of the whole module: a config
    key or artifact this figure should change your mind about."""
    def wrap(fn):
        PANELS.append({"key": key, "title": title, "informs": list(informs),
                       "lede": lede, "fn": fn})
        return fn
    return wrap


# ------------------------------------------------------------------ helpers

def _openings(d):
    """One row per episode, its opening hour. The episode-level unit of
    analysis: inventory persists across hours, so anything about stock or
    exposure is counted here and never summed over rows."""
    return d[~d.episode_id.duplicated()]


def _closings(d):
    return episodes.last_rows(d)


def _share_table(frame, by, weights, top=12):
    """Grouped totals, biggest first, with a labelled `other` rather than a
    silent truncation."""
    g = frame.groupby(by).agg(**{k: (c, "sum") for k, c in weights.items()})
    g = g.sort_values(list(weights)[0], ascending=False)
    head, tail = g.iloc[:top], g.iloc[top:]
    rows = [dict({"name": str(i)}, **{k: float(v) for k, v in r.items()})
            for i, r in head.iterrows()]
    if len(tail):
        rows.append(dict({"name": f"other ({len(tail)})"},
                         **{k: float(tail[k].sum()) for k in weights}))
    return rows


def _hist(values, bins, lo=None, hi=None):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(
        v, bins=bins, range=(lo if lo is not None else float(v.min()),
                             hi if hi is not None else float(v.max())))
    return {"edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts]}


def _pcts(v, pcts=(10, 25, 50, 75, 90, 99)):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {}
    return {f"p{p}": round(float(np.percentile(v, p)), 4) for p in pcts}


# --------------------------------------------------------- 1 scale & coverage

@panel("volumes", "Daily volumes",
       informs=["posterior.min_episodes_per_week_for_cell",
                "ab_test.min_episodes_per_block",
                "monitoring.shadow_gate.sample_episodes"],
       lede="How much of everything there is, per day. Sets whether any cell "
            "can leave the global bucket, how long an A/B has to run, and "
            "whether a 3,000-episode shadow sample is a large or small "
            "fraction of a day.")
def p_volumes(d, cfg):
    op = _openings(d)
    daily = op.groupby(op.date.astype(str)).agg(
        episodes=("episode_id", "size"),
        skus=("sku_id", "nunique"),
        fcs=("fc", "nunique"))
    daily["sku_fc"] = op.groupby(op.date.astype(str)).apply(
        lambda g: g.sku_id.astype(str).str.cat(g.fc.astype(str), sep="|")
        .nunique(), include_groups=False)
    rows_daily = d.groupby(d.date.astype(str)).size()
    days = sorted(daily.index)
    return {
        "days": len(days),
        "episodes_total": int(len(op)),
        "episodes_per_day": _pcts(daily.episodes),
        "unique_skus_per_day": _pcts(daily.skus),
        "unique_sku_fc_per_day": _pcts(daily.sku_fc),
        "unique_skus_overall": int(op.sku_id.nunique()),
        "fcs_overall": int(op.fc.nunique()),
        "rows_per_day_p50": int(np.percentile(rows_daily, 50)),
        "hours_per_episode": _pcts(d.groupby("episode_id").size()),
        "chart": {"kind": "line", "x": days,
                  "series": {"episodes": [int(daily.episodes[k]) for k in days],
                             "unique SKUs": [int(daily.skus[k]) for k in days]},
                  "y_label": "count per day"},
    }


@panel("coverage", "Calendar coverage",
       informs=["data.split", "data.holdout", "data.exclusion_window"],
       lede="Gaps are not cosmetic: a missing hour splits one window into two "
            "fragments, and the first fragment ends with no closure sentinel.")
def p_coverage(d, cfg):
    ds = pd.to_datetime(d.date.astype(str))
    present = set(ds.dt.strftime("%Y-%m-%d"))
    full = pd.date_range(ds.min(), ds.max(), freq="D").strftime("%Y-%m-%d")
    excl = cfg["data"]["exclusion_window"]
    missing = [x for x in full if x not in present]
    inside_excl = [x for x in missing if excl["start"] <= x <= excl["end"]]
    return {
        "first_date": str(ds.min().date()), "last_date": str(ds.max().date()),
        "calendar_days": len(full), "days_with_data": len(present),
        "missing_days": len(missing),
        "missing_days_inside_exclusion_window": len(inside_excl),
        "missing_days_unexplained": [x for x in missing if x not in inside_excl][:40],
        "note": ("Days inside the exclusion window are missing on purpose. "
                 "Anything in missing_days_unexplained is a feed gap, and each "
                 "one fragments every window that spans it."),
    }


@panel("splits", "Split occupancy",
       informs=["data.split", "baseline_model.calibration_gate_window",
                "baseline_model.calibration_fit_window"],
       lede="A gate window carrying one week of episodes carries one week of "
            "demand noise with it.")
def p_splits(d, cfg):
    from bootstrap.prepare_data import split_frames
    s, h = cfg["data"]["split"], cfg["data"].get("holdout")
    frames = dict(split_frames(d, cfg))
    if h:
        frames["holdout"] = episodes.window_slice(d, h["start"], h["end"])
    out, order = {}, ["train", "calib", "test", "holdout"]
    for name in order:
        g = frames.get(name)
        if g is None or not len(g):
            out[name] = {"episodes": 0}
            continue
        op = _openings(g)
        out[name] = {
            "episodes": int(len(op)), "rows": int(len(g)),
            "skus": int(op.sku_id.nunique()),
            "units_sold": int(g.units_sold.sum()),
            "days": int(g.date.astype(str).nunique()),
            "cogs_at_risk": round(float(
                (op.cost * op.starting_inventory).sum()), 1),
        }
    return {"by_window": out,
            "chart": {"kind": "bars", "labels": order,
                      "values": [out[k].get("episodes", 0) for k in order],
                      "y_label": "episodes"}}


# ------------------------------------------------------------ 2 concentration

@panel("pareto", "Where the money is",
       informs=["exploration.tau_initial",
                "posterior.min_episodes_per_week_for_cell"],
       lede="IL is currency, and tau is a currency budget. A draw on a tail "
            "SKU costs the same attention and buys almost no measurable "
            "evidence, so concentration decides where exploration is worth "
            "spending.")
def p_pareto(d, cfg, top=25):
    op = _openings(d).copy()
    op["cogs"] = op.cost * op.starting_inventory
    sold = d.groupby("sku_id").units_sold.sum()
    g = op.groupby("sku_id").agg(cogs=("cogs", "sum"),
                                 episodes=("episode_id", "size"))
    g["units"] = sold.reindex(g.index).fillna(0)
    g = g.sort_values("cogs", ascending=False)
    total = float(g.cogs.sum()) or 1.0
    cum = (g.cogs.cumsum() / total).to_numpy()
    n = len(g)

    def at(share):
        k = max(int(round(n * share)), 1)
        return round(float(cum[k - 1]), 4)

    return {
        "skus": n,
        "cogs_at_risk_total": round(total, 1),
        "cogs_share_of_top_1pct": at(0.01),
        "cogs_share_of_top_5pct": at(0.05),
        "cogs_share_of_top_10pct": at(0.10),
        "cogs_share_of_top_25pct": at(0.25),
        "skus_covering_half_the_cogs": int(np.searchsorted(cum, 0.5) + 1),
        "skus_covering_80pct_of_cogs": int(np.searchsorted(cum, 0.8) + 1),
        "top_skus": [{"sku_id": int(i), "cogs_at_risk": round(float(r.cogs), 1),
                      "episodes": int(r.episodes), "units_sold": int(r.units)}
                     for i, r in g.head(top).iterrows()],
        "chart": {"kind": "pareto",
                  "x": [round(float(i + 1) / n, 5) for i in range(n)][::max(n // 400, 1)],
                  "y": [round(float(c), 5) for c in cum][::max(n // 400, 1)],
                  "x_label": "share of SKUs", "y_label": "share of COGS at risk"},
    }


@panel("mix", "Category and facility mix",
       informs=["reference_discount", "dispersion.fallback_order"],
       lede="Every anchor, every dispersion cell and every posterior cell is "
            "keyed on these. A category holding 40% of the exposure and 3% of "
            "the episodes is a different problem from the reverse.")
def p_mix(d, cfg):
    op = _openings(d).copy()
    op["cogs"] = op.cost * op.starting_inventory
    op["units"] = d.groupby("episode_id").units_sold.sum().reindex(
        op.episode_id).to_numpy()
    op["one"] = 1.0
    cats = _share_table(op, "category",
                        {"cogs_at_risk": "cogs", "episodes": "one",
                         "units_sold": "units"})
    fcs = _share_table(op, "fc",
                       {"cogs_at_risk": "cogs", "episodes": "one",
                        "units_sold": "units"})
    ref = cfg["reference_discount"]
    covered = [c for c in op.category.unique()
               if str(c).replace(" ", "_") in ref]
    return {
        "categories": int(op.category.nunique()),
        "subcategories": int(op.subcategory.nunique()),
        "by_category": cats,
        "by_fc": fcs,
        "categories_with_an_explicit_reference_discount": sorted(map(str, covered)),
        "categories_falling_back_to_default": sorted(
            str(c) for c in op.category.unique()
            if str(c).replace(" ", "_") not in ref),
        "chart": {"kind": "bars", "labels": [r["name"] for r in cats],
                  "values": [r["cogs_at_risk"] for r in cats],
                  "y_label": "COGS at risk"},
    }


# ------------------------------------------------------------------ 3 window

@panel("window", "The clearance window",
       informs=["data.max_window_hours", "data.manufacturing_window_hours"],
       lede="The window is the DP's horizon. Its shape decides how much "
            "prediction work each episode costs and how long the agent thinks "
            "it has to sell.")
def p_window(d, cfg):
    op = _openings(d)
    entry = op.hours_remaining.to_numpy()
    observed = d.groupby("episode_id").size()
    cap = cfg["data"]["max_window_hours"]
    return {
        "entry_hours_remaining": _pcts(entry),
        "observed_hours": _pcts(observed),
        "share_entry_at_or_above_cap": round(float((entry >= cap).mean()), 4),
        "common_entry_values": {
            str(int(k)): int(v) for k, v in
            pd.Series(entry).round().value_counts().head(10).items()},
        "chart": {"kind": "hist", "y_label": "episodes",
                  "x_label": "hours_remaining at entry",
                  **_hist(entry, bins=40, lo=0, hi=float(min(cap, entry.max())))},
    }


@panel("entry_hour", "When windows open",
       informs=["baseline_model.ref_rate_window_days"],
       lede="Elasticity identification uses ENTRY rows only, so this is the "
            "population that estimate is drawn from. It is also the confound: "
            "if entry hour and entry discount move together, the two cannot "
            "be told apart from history.")
def p_entry_hour(d, cfg):
    op = _openings(d)
    by_hour = op.groupby(op.hour_of_day).size().reindex(range(24), fill_value=0)
    depth = op.groupby(op.hour_of_day).total_discount.mean().reindex(range(24))
    # np.corrcoef divides by the standard deviation, so a constant entry hour
    # or a constant discount yields nan -- and json.dump writes bare NaN,
    # which is not valid JSON and breaks every consumer downstream
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (float(np.corrcoef(op.hour_of_day, op.total_discount)[0, 1])
                if len(op) > 2 else float("nan"))
    corr = round(corr, 4) if np.isfinite(corr) else None
    return {
        "episodes_by_entry_hour": {str(h): int(by_hour[h]) for h in range(24)},
        "mean_entry_discount_by_hour": {
            str(h): (round(float(depth[h]), 4) if pd.notna(depth[h]) else None)
            for h in range(24)},
        "corr_entry_hour_vs_entry_discount": corr,
        "note": ("A strong correlation here is the clock confound in one "
                 "number: history cannot separate 'later' from 'cheaper'."),
        "chart": {"kind": "bars", "labels": [str(h) for h in range(24)],
                  "values": [int(by_hour[h]) for h in range(24)],
                  "y_label": "windows opened", "x_label": "hour of day"},
    }


# ---------------------------------------------------------------- 4 outcomes

@panel("clearance", "Clearance",
       informs=["monitoring.stop_conditions.scrap_deterioration_pct"],
       lede="Units sold over everything the episode had to sell. The quantity "
            "the whole system trades against loss, and the one the business "
            "will watch first if the agent holds prices higher.")
def p_clearance(d, cfg):
    """Denominator is SUPPLY, not opening stock.

    Opening stock stopped being an episode's supply the moment restocked
    episodes were kept in the population, and the failure is not subtle: a
    window that opened with 3, took 10 mid-flight and sold 9 read as 300%
    cleared and counted in `share_fully_cleared`, while it scrapped 4 units.
    The histogram clipped at 1.0, so nothing showed.

    Clearance cannot exceed 1: supply counts everything that arrived.
    """
    op = _openings(d)
    all_flow = episodes.episode_flow(d).reindex(op.episode_id)
    kind = episodes.classify(d).reindex(op.episode_id)

    # Two exclusions, both counted rather than quietly averaged in.
    #
    #   NOT ELIGIBLE   the close is ambiguous, so no figure here is safe.
    #   NOT CLOSED     the window has not ended. Its "clearance" is only
    #                  sold-so-far, and the bias runs ONE way -- an unfinished
    #                  episode has by definition sold less than it will. These
    #                  are also the largest episodes in the extract, so the
    #                  drag is far bigger than their count suggests: on a
    #                  two-episode example one unclosed window pulled the mean
    #                  from 0.70 to 0.45.
    unclosed = (kind == episodes.NOT_CLOSED).to_numpy()
    keep = all_flow.eligible.to_numpy() & ~unclosed
    flow = all_flow[keep]
    rate = flow.clearance.to_numpy()
    counts = episodes.classify_last(_closings(d)).value_counts()
    frame = pd.DataFrame({"category": op.category.to_numpy()[keep],
                          "rate": rate})
    n = max(len(flow), 1)
    return {
        "clearance_rate": _pcts(rate),
        "mean_clearance": round(float(rate.mean()), 4),
        "share_fully_cleared": round(float((rate >= 1.0).mean()), 4),
        "share_sold_nothing": round(float((rate <= 0).mean()), 4),
        "endings": {str(k): int(v) for k, v in counts.items()},
        "by_category": {str(k): round(float(g.rate.mean()), 4)
                        for k, g in frame.groupby("category")},
        # what the denominator correction is worth, so the number above can be
        # compared with anything quoted before supply was accounted for
        "supply": {
            "episodes_restocked": int((flow.arrived > 0).sum()),
            "share_restocked": round(float((flow.arrived > 0).sum()) / n, 4),
            "units_arrived": int(flow.arrived.sum()),
            "units_vanished": int(flow.vanished.sum()),
            "supply_over_opening": round(
                float(flow.supply.sum() / max(flow.opening.sum(), 1)), 4),
            "max_clearance": round(float(rate.max()) if len(rate) else 0.0, 4),
            "episodes_excluded_not_eligible": int(
                (~all_flow.eligible.to_numpy()).sum()),
            "episodes_excluded_unclosed": int(
                (kind == episodes.NOT_CLOSED).sum()),
            "note": ("clearance = sold / (opening + net arrivals), over the "
                     "episodes whose flow identity holds. Against opening "
                     "stock alone a restocked episode reads above 1.0 -- and "
                     "can read above 1.0 while scrapping units, which is why "
                     "max_clearance is reported: it must never exceed 1. "
                     "Episodes whose final hour is ambiguous -- it sold more "
                     "than it opened with, so the close rests on an assumption "
                     "-- are excluded and counted. Scrap is the last hour's "
                     "leftover PLUS the shrink, so supply = sold + scrap."),
        },
        "chart": {"kind": "hist", "y_label": "episodes",
                  "x_label": "units sold / supply",
                  **_hist(rate, bins=25, lo=0, hi=1.0)},
    }


@panel("timing", "When the units move",
       informs=["pricing.entry_offsets"],
       lede="If most units go in the last hours, waiting is cheap and the DP "
            "should hold. If they go early, the entry price is nearly the "
            "whole decision.")
def p_timing(d, cfg):
    obs = d[d.units_sold > 0] if "units_sold" in d else d
    hr = obs.hours_remaining.to_numpy()
    buckets = pd.cut(hr, [-0.1, 3, 6, 12, 24, 48, 1e9],
                     labels=["0-3", "4-6", "7-12", "13-24", "25-48", "48+"])
    by = obs.groupby(buckets, observed=False).units_sold.sum()
    total = float(by.sum()) or 1.0
    labels = list(by.index.astype(str))
    return {
        "units_by_hours_remaining": {k: int(v) for k, v in by.items()},
        "share_of_units_in_final_6h": round(
            float(by.get("0-3", 0) + by.get("4-6", 0)) / total, 4),
        "chart": {"kind": "bars", "labels": labels,
                  "values": [round(float(by[k]) / total, 4) for k in by.index],
                  "y_label": "share of units", "x_label": "hours remaining"},
    }


# ------------------------------------------------- 5 price and identification

@panel("discount_path", "The legacy price path",
       informs=["pricing.entry_offsets", "reference_discount"],
       lede="The agent's entry arms run from 15pp shallower than the category "
            "reference to one 5pp step deeper. If history never priced there, "
            "the arms are extrapolation rather than interpolation.")
def p_discount_path(d, cfg):
    op = _openings(d)
    gap = (op.total_discount - op.d_ref).to_numpy()
    offsets = cfg["pricing"]["entry_offsets"]
    step = cfg["pricing"]["tier_step"]
    support = {f"{o:+.2f}": round(float(
        (np.abs(gap - o) <= step / 2).mean()), 4) for o in offsets}
    ramp = d.groupby(pd.cut(d.hours_remaining,
                            [-0.1, 3, 6, 12, 24, 48, 1e9],
                            labels=["0-3", "4-6", "7-12", "13-24", "25-48", "48+"],
                            ), observed=False).total_discount.mean()
    return {
        "entry_discount": _pcts(op.total_discount),
        "entry_gap_to_reference": _pcts(gap),
        "share_of_entries_within_half_a_tier_of_each_arm": support,
        "mean_discount_by_hours_remaining": {
            str(k): (round(float(v), 4) if pd.notna(v) else None)
            for k, v in ramp.items()},
        "note": ("An arm with near-zero support is one the demand model has "
                 "never seen priced; the parametric layer extrapolates there "
                 "on the elasticity prior alone."),
        "chart": {"kind": "bars", "labels": list(support),
                  "values": list(support.values()),
                  "y_label": "share of entries", "x_label": "entry offset"},
    }


@panel("anchors", "Anchor rows",
       informs=["baseline_model.calibration_min_anchor_rows",
                "baseline_model.mix_decomposition_min_unit_rows",
                "baseline_model.ref_rate_anchor_band", "pricing.tier_step"],
       lede="Two bands, same word, different jobs: calibration uses "
            "tier_step/2 (±1.25pp) and the velocity features use "
            "ref_rate_anchor_band (±2.5pp). Calibration is fit ENTIRELY on "
            "the first, and nothing showed how many rows exist per "
            "subcategory before the fit runs.")
def p_anchors(d, cfg):
    gap = (d.total_discount - d.d_ref).abs()
    tight = cfg["pricing"]["tier_step"] / 2
    wide = cfg["baseline_model"]["ref_rate_anchor_band"]
    need = cfg["baseline_model"]["calibration_min_anchor_rows"]
    per_sub = d[gap <= tight].groupby("subcategory").size()
    subs = d.subcategory.nunique()
    return {
        "calibration_band_pp": round(tight * 100, 3),
        "velocity_band_pp": round(wide * 100, 3),
        "rows_within_calibration_band": int((gap <= tight).sum()),
        "rows_within_velocity_band": int((gap <= wide).sum()),
        "share_of_rows_within_calibration_band": round(float((gap <= tight).mean()), 4),
        "subcategories": int(subs),
        "subcategories_with_anchor_rows": int(len(per_sub)),
        "subcategories_clearing_min_anchor_rows": int((per_sub >= need).sum()),
        "share_of_subcategories_clearing_min": round(
            float((per_sub >= need).sum()) / max(subs, 1), 4),
        "anchor_rows_per_subcategory": _pcts(per_sub) if len(per_sub) else {},
        "note": ("Subcategories below the minimum inherit their parent "
                 "factor. A low share here is not a failure -- the hierarchy "
                 "is built for it -- but it means calibration is effectively "
                 "happening at category level."),
        "chart": {"kind": "hist", "y_label": "subcategories",
                  "x_label": "anchor rows (calibration band)",
                  **_hist(per_sub, bins=25)} if len(per_sub) else None,
    }


# ----------------------------------------------------------- 6 cost geometry

@panel("cost_geometry", "How much room there is to discount",
       informs=["reassessment_gates.max_share_non_explorable",
                "exploration.min_feasible_tiers", "pricing.tier_step"],
       lede="The cost floor decides the action set before any model speaks. "
            "An episode with fewer than min_feasible_tiers discounts cannot "
            "be experimented on at any price.")
def p_cost_geometry(d, cfg):
    op = _openings(d)
    ratio = (op.cost / op.original_price).to_numpy()
    d_max = op.d_max.to_numpy()
    step = cfg["pricing"]["tier_step"]
    tiers = np.floor(d_max / step + 1e-9) + 1
    need = cfg["exploration"]["min_feasible_tiers"]
    frame = pd.DataFrame({"category": op.category.to_numpy(),
                          "ratio": ratio, "tiers": tiers})
    return {
        "cost_ratio": _pcts(ratio),
        "d_max": _pcts(d_max),
        "feasible_tiers": _pcts(tiers),
        "share_non_explorable": round(float((tiers < need).mean()), 4),
        "gate_threshold": cfg["reassessment_gates"]["max_share_non_explorable"],
        "by_category": {str(k): {"cost_ratio_p50": round(float(g.ratio.median()), 4),
                                 "tiers_p50": int(g.tiers.median()),
                                 "share_non_explorable": round(
                                     float((g.tiers < need).mean()), 4)}
                        for k, g in frame.groupby("category")},
        "chart": {"kind": "hist", "y_label": "episodes",
                  "x_label": "cost / full price",
                  **_hist(ratio, bins=30, lo=0, hi=1.0)},
    }


@panel("entry_arms", "Which entry arms actually exist",
       informs=["pricing.entry_offsets"],
       lede="config asserts the deepest arm disappears above a ~0.65 cost "
            "ratio. Nothing measured how often. The deep arm is the escape "
            "valve for the clearance/scrap trade, so its absence is a "
            "capability the DP silently does not have.")
def p_entry_arms(d, cfg):
    op = _openings(d)
    d_max = op.d_max.to_numpy()
    d_ref = op.d_ref.to_numpy()
    step = cfg["pricing"]["tier_step"]
    out = {}
    for o in cfg["pricing"]["entry_offsets"]:
        arm = np.round((d_ref + o) / step) * step
        feasible = (arm >= 0) & (arm <= d_max + 1e-9)
        out[f"{o:+.2f}"] = round(float(feasible.mean()), 4)
    return {
        "share_of_episodes_where_arm_is_feasible": out,
        "arms_available": _pcts(np.array([
            sum((np.round((dr + o) / step) * step >= 0)
                and (np.round((dr + o) / step) * step <= dm + 1e-9)
                for o in cfg["pricing"]["entry_offsets"])
            for dr, dm in zip(d_ref, d_max)])),
        "chart": {"kind": "bars", "labels": list(out),
                  "values": list(out.values()),
                  "y_label": "share of episodes", "x_label": "entry offset"},
    }


# ------------------------------------------------- 7 group sizes vs thresholds

@panel("cells", "Will the hierarchy have anything to work with?",
       informs=["dispersion.min_rows_per_group",
                "baseline_model.calibration_min_anchor_rows",
                "baseline_model.mix_decomposition_min_unit_rows",
                "posterior.min_episodes_per_week_for_cell"],
       lede="One table for the question every fallback rule turns on: how "
            "many groups clear their own minimum, and how many fall through "
            "to the parent.")
def p_cells(d, cfg):
    op = _openings(d)
    weeks = max(pd.to_datetime(d.date.astype(str)).dt.to_period("W").nunique(), 1)
    gap = (d.total_discount - d.d_ref).abs()
    anchors = d[gap <= cfg["pricing"]["tier_step"] / 2]
    unit = d.sku_id.astype(str) + "|" + d.fc.astype(str)
    checks = [
        ("subcategory rows", d.groupby("subcategory").size(),
         cfg["dispersion"]["min_rows_per_group"], "dispersion.min_rows_per_group"),
        ("subcategory anchor rows", anchors.groupby("subcategory").size(),
         cfg["baseline_model"]["calibration_min_anchor_rows"],
         "baseline_model.calibration_min_anchor_rows"),
        ("sku x fc anchor rows", anchors.groupby(unit[anchors.index]).size(),
         cfg["baseline_model"]["mix_decomposition_min_unit_rows"],
         "baseline_model.mix_decomposition_min_unit_rows"),
        ("category episodes per week", op.groupby("category").size() / weeks,
         cfg["posterior"]["min_episodes_per_week_for_cell"],
         "posterior.min_episodes_per_week_for_cell"),
    ]
    rows = []
    for name, sizes, threshold, key in checks:
        n = int(len(sizes))
        clears = int((sizes >= threshold).sum()) if n else 0
        rows.append({"grouping": name, "groups": n, "threshold": threshold,
                     "config_key": key, "clearing": clears,
                     "share_clearing": round(clears / n, 4) if n else None,
                     "p50_size": round(float(np.median(sizes)), 1) if n else None})
    return {"weeks_in_window": int(weeks), "checks": rows,
            "chart": {"kind": "bars",
                      "labels": [r["grouping"] for r in rows],
                      "values": [r["share_clearing"] or 0.0 for r in rows],
                      "y_label": "share of groups clearing"}}


# ------------------------------------------------------------------- 8 drift

@panel("drift", "Week by week",
       informs=["baseline_model.calibration_fit_window",
                "baseline_model.calibration_gate_band"],
       lede="The panel that would have caught it: the calibration fortnight "
            "turned out to be the most anomalous stretch in five months, and "
            "factors fit on it inherited the anomaly. One plot, one second.")
def p_drift(d, cfg):
    dd = d.copy()
    dd["week"] = pd.to_datetime(dd.date.astype(str)).dt.to_period("W").astype(str)
    op = _openings(dd)
    g = dd.groupby("week")
    og = op.groupby("week")
    weeks = sorted(g.groups)
    units_per_ep = (g.units_sold.sum() / og.size()).reindex(weeks)
    depth = g.total_discount.mean().reindex(weeks)
    cost_ratio = (og.cost.sum() / og.original_price.sum()).reindex(weeks)
    eps = og.size().reindex(weeks)
    s = cfg["data"]["split"]
    return {
        "weeks": weeks,
        "episodes_by_week": {w: int(eps[w]) for w in weeks},
        "units_per_episode_by_week": {w: round(float(units_per_ep[w]), 4)
                                      for w in weeks},
        "mean_discount_by_week": {w: round(float(depth[w]), 4) for w in weeks},
        "cost_ratio_by_week": {w: round(float(cost_ratio[w]), 4) for w in weeks},
        "split_boundaries": {"train_end": s["train_end"],
                             "calib_end": s["calib_end"],
                             "test_end": s["test_end"]},
        "note": ("Read units_per_episode against the split boundaries. A fit "
                 "window that sits on an unrepresentative stretch pushes every "
                 "factor derived from it the wrong way."),
        "chart": {"kind": "line", "x": weeks,
                  "series": {"units / episode": [round(float(units_per_ep[w]), 3)
                                                 for w in weeks],
                             "mean discount": [round(float(depth[w]), 3)
                                               for w in weeks]},
                  "y_label": "weekly level"},
    }


# ------------------------------------------------------------------- assembly

def build(d, cfg):
    """Describe the INTEGRITY population -- everything that survived the filter
    chain, DP-ineligible episodes included.

    That is the right frame for a description of the business, and it is not
    the frame the DP acts on, so the header states the split. It matters for
    two panels in particular: the window counter carries negatives
    (`negative_window`) and values above the cap (`window_too_long`), so their
    tails are real data about what was flagged rather than noise to clip.
    """
    pop = {"rows": int(len(d)), "episodes": int(d.episode_id.nunique())}
    if "dp_eligible" in d:
        pop["episodes_dp_eligible"] = int(
            d.loc[d.dp_eligible, "episode_id"].nunique())
        pop["episodes_by_dp_ineligible_reason"] = {
            str(k): int(v) for k, v in
            d[~d.dp_eligible].groupby("dp_ineligible_reason")
            .episode_id.nunique().items()}
        pop["note"] = (
            "The INTEGRITY population: every episode that survived the filter "
            "chain, including those the DP cannot price. Panels below describe "
            "all of it. The DP, the calibration gate and the A/B read "
            "dp_eligible only.")
    out = {"population": pop, "panels": {}}
    for p in PANELS:
        body = p["fn"](d, cfg)
        out["panels"][p["key"]] = dict(
            {"title": p["title"], "lede": p["lede"], "informs": p["informs"]},
            **body)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="data/prepared.parquet")
    ap.add_argument("--out", default="reports/eda.json")
    ap.add_argument("--html", default="docs/eda.html")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)
    report = build(d, cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    from tools.eda_page import render
    os.makedirs(os.path.dirname(args.html) or ".", exist_ok=True)
    with open(args.html, "w") as f:
        f.write(render(report, args.input))

    print(f"{report['population']['episodes']:,} episodes, "
          f"{report['population']['rows']:,} rows")
    for key, p in report["panels"].items():
        print(f"  {key:16} {p['title']}")
    print(f"wrote {args.out} and {args.html}")


if __name__ == "__main__":
    main()
