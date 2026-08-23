"""
Synthetic FLC data generator.

Emits data matching the flc_filtered.parquet schema exactly, with a known
ground-truth elasticity so that estimators can be validated against it.

Two policy modes:

  --policy legacy      Reproduces the current production behaviour: discount
                       ramps ~1pp/hour from entry to a cap. Price is collinear
                       with hour-of-day, so elasticity is NOT identifiable.
                       Use this to confirm an estimator DETECTS the confound.

  --policy randomized  Entry discount randomized across the feasible range and
                       hourly perturbations drawn from feasible deeper tiers.
                       Elasticity IS identifiable. Use this to confirm an
                       estimator RECOVERS epsilon_true.

Usage:
    python3 make_dummy_flc.py --skus 400 --days 60 --policy legacy
    python3 make_dummy_flc.py --skus 400 --days 60 --policy randomized \
        --out data/flc_filtered_randomized.parquet

THE SOURCE'S TWO INVENTORY CONVENTIONS ARE MODELLED HERE, and both must stay
modelled. A fixture missing one does not fail -- it passes, quietly, with the
code path that reads it never executed:

  write-off sentinel   `ending_inventory` zeroed on the row a listing closes.
                       This is the ONLY test for closure. A fixture without it
                       reads every episode as unclosed; with the old
                       `write_off_convention` fallback in place it read every
                       episode as CLOSED instead, and a stale fixture carrying
                       no sentinel at all went unnoticed for months.
  shrink               `0 < ending < starting - sold` -- stock gone without a
                       sale and without a write-off. Real, and the chain is
                       specified to KEEP it: scrap is leftover PLUS shrink.

`main` counts both on every run. If either prints zero, the fixture is not
exercising what it claims to.

DIRT IS INJECTED AT THE SCOPE THE DEFECT REALLY HAS. Null category, null
subcategory, zero base price and the multi-lot over-sell are ROW properties
and are injected per row. A negative `flc_window` is a WINDOW property -- an
episode entering already negative -- and is injected across whole windows,
because `assign_episode_ids` differences the counter hour to hour and a single
bad value reads as a window boundary rather than as bad data.
"""

import argparse
import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# Catalog definition. Reference discounts mirror PRD section 7.1.
# --------------------------------------------------------------------------

CATALOG = {
    "MEAT": {
        "subcats": ["CHICKEN", "PORK", "BEEF"],
        "ref_discount": 0.25,
        "asp": (12000, 26000),
        "cost_ratio": (0.62, 0.72),
        "base_rate": 0.55,
        "epsilon": -1.10,
    },
    "SIDE DISH": {
        "subcats": ["KIMCHI", "BANCHAN"],
        "ref_discount": 0.25,
        "asp": (4000, 11000),
        "cost_ratio": (0.55, 0.68),
        "base_rate": 0.70,
        "epsilon": -0.85,
    },
    "SEAFOOD": {
        "subcats": ["FISH", "SHELLFISH"],
        "ref_discount": 0.30,
        "asp": (9000, 32000),
        "cost_ratio": (0.66, 0.78),
        "base_rate": 0.40,
        "epsilon": -1.45,
    },
    "FRUIT": {
        "subcats": ["BERRY", "CITRUS", "MELON"],
        "ref_discount": 0.30,
        "asp": (5000, 20000),
        "cost_ratio": (0.60, 0.74),
        "base_rate": 0.80,
        "epsilon": -1.70,
    },
    "VEGETABLE": {
        "subcats": ["LEAF", "ROOT"],
        "ref_discount": 0.30,
        "asp": (2500, 9000),
        "cost_ratio": (0.58, 0.70),
        "base_rate": 0.95,
        "epsilon": -1.25,
    },
}

FCS = ["BUC2", "DAJ1", "ICN3", "BSN1", "GWJ2"]

# Time-of-day demand multiplier. This is the confounder: it rises through the
# evening independently of price, and the legacy policy ramps price on the
# same clock.
HOUR_FACTOR = {
    10: 0.45, 11: 0.55, 12: 0.75, 13: 0.80, 14: 0.85,
    15: 0.95, 16: 1.20, 17: 1.55, 18: 1.95, 19: 2.20,
}

EXCLUSION_START = dt.date(2026, 4, 25)
EXCLUSION_END = dt.date(2026, 6, 3)

SCHEMA = pa.schema([
    ("date", pa.date32()),
    ("hour", pa.int64()),
    ("skuseq", pa.int64()),
    ("fc", pa.string()),
    ("inventory", pa.float64()),
    ("discount", pa.float64()),
    ("units_sold", pa.int64()),
    ("normal_asp", pa.float64()),
    ("final_price", pa.float64()),
    ("cogs_wo_vat", pa.float64()),
    ("ending_inventory", pa.float64()),
    ("flc_window", pa.float64()),
    ("category", pa.string()),
    ("subcategory", pa.string()),
])


def round_to(x, step=10):
    return float(np.round(x / step) * step)


def build_sku_master(n_skus, rng):
    cats = list(CATALOG)
    weights = np.array([3, 2, 2, 2, 2], dtype=float)
    weights /= weights.sum()
    rows = []
    for i in range(n_skus):
        cat = rng.choice(cats, p=weights)
        spec = CATALOG[cat]
        asp = round_to(rng.uniform(*spec["asp"]), 10)
        cost = round_to(asp * rng.uniform(*spec["cost_ratio"]), 10)
        rows.append({
            "skuseq": 15000000 + int(rng.integers(0, 900000)),
            "category": cat,
            "subcategory": rng.choice(spec["subcats"]),
            "normal_asp": asp,
            "cogs_wo_vat": cost,
            # per-SKU demand scale, lognormal so a few SKUs move much faster
            "sku_scale": float(np.exp(rng.normal(0, 0.55))),
        })
    return pd.DataFrame(rows).drop_duplicates("skuseq").reset_index(drop=True)


def legacy_discount_path(entry_d, n_hours, d_max):
    """Deterministic ~1pp/hour ramp, flat for the first two hours, capped."""
    path = []
    d = entry_d
    for h in range(n_hours):
        if h >= 2:
            d = min(d + 0.01, min(0.30, d_max))
        path.append(min(d, d_max))
    return path


def randomized_discount_path(entry_d, n_hours, d_max, rng, tier=0.025):
    """Entry randomized by caller; hourly draws from feasible deeper tiers."""
    path = []
    d = entry_d
    for h in range(n_hours):
        if h >= 1 and rng.random() < 0.35:
            deeper = np.arange(d + tier, d_max + 1e-9, tier)
            if len(deeper):
                # bias toward the deep end: information scales with (log ratio)^2
                w = np.linspace(1.0, 3.0, len(deeper))
                d = float(rng.choice(deeper, p=w / w.sum()))
        path.append(min(d, d_max))
    return path


def generate(n_skus, n_days, policy, seed, dirty_frac, shrink_rate=0.02):
    rng = np.random.default_rng(seed)
    master = build_sku_master(n_skus, rng)
    start = dt.date(2026, 3, 1)

    records = []
    windows = []          # (start, end) row range of each emitted window
    for _, sku in master.iterrows():
        spec = CATALOG[sku.category]
        eps_true = spec["epsilon"] * float(np.exp(rng.normal(0, 0.12)))
        d_ref = spec["ref_discount"]
        d_max = 1.0 - sku.cogs_wo_vat / sku.normal_asp
        if d_max < 0.05:
            continue

        r_disp = float(rng.uniform(0.8, 4.0))  # NB dispersion
        n_episodes = int(rng.integers(2, 14))

        for _ in range(n_episodes):
            day = start + dt.timedelta(days=int(rng.integers(0, n_days)))
            fc = rng.choice(FCS)
            start_hour = int(rng.integers(10, 14))
            n_hours = int(rng.integers(4, 20 - start_hour + 1))
            if n_hours < 2:
                continue

            inv = int(rng.integers(1, 32))
            window_start = len(records)   # for whole-window dirt injection

            if policy == "legacy":
                entry_d = min(d_ref, d_max)
                path = legacy_discount_path(entry_d, n_hours, d_max)
            else:
                lo, hi = max(0.05, d_ref - 0.12), min(d_max, d_ref + 0.12)
                entry_d = float(rng.uniform(lo, hi)) if hi > lo else min(d_ref, d_max)
                entry_d = float(np.round(entry_d / 0.025) * 0.025)
                entry_d = min(max(entry_d, 0.0), d_max)
                path = randomized_discount_path(entry_d, n_hours, d_max, rng)

            for h_idx in range(n_hours):
                hour = start_hour + h_idx
                if inv <= 0:
                    break
                d = path[h_idx]
                price_ratio = (1 - d) / (1 - d_ref)
                mu = (
                    spec["base_rate"]
                    * sku.sku_scale
                    * HOUR_FACTOR.get(hour, 1.0)
                    * price_ratio ** eps_true
                )
                mu = max(mu, 0.01)
                p = r_disp / (r_disp + mu)
                demand = int(rng.negative_binomial(r_disp, p))
                sold = min(demand, inv)

                # SHRINK -- stock that left without being sold and without
                # being written off, `0 < ending < starting - sold`. A real
                # business event the source reports, not dirt: the old chain
                # rule DELETED these episodes and took 33.6pp of the extract's
                # COGS with them, selecting the largest windows 4.5 to 1.
                #
                # Kept strictly interior. `ending == 0` would read as the
                # write-off sentinel by `hour_status`, which is a CLOSE, not a
                # shrink -- so the draw leaves at least one unit standing and
                # the episode carries on. And never on the final hour, where
                # shrink and leftover are indistinguishable once the source
                # zeroes the ending: `starting - sold` is the whole remainder
                # there and it is booked as leftover.
                #
                # Chain continuity is preserved BY CONSTRUCTION -- `inv` is
                # assigned from `ending` below, so `ending[t] == starting[t+1]`
                # whatever leaves during the hour.
                shrink = 0
                if (h_idx < n_hours - 1 and inv - sold >= 2
                        and rng.random() < shrink_rate):
                    shrink = int(rng.integers(1, min(inv - sold, 3)))
                ending = inv - sold - shrink

                if sold == 0:
                    final_price = 0.0
                else:
                    # realized ASP: small intra-hour drift from the offered price
                    realized = sku.normal_asp * (1 - d) * (1 + rng.normal(0, 0.004))
                    final_price = round_to(realized, 10)

                records.append((
                    day, hour, int(sku.skuseq), fc,
                    float(inv), float(np.round(d * 100, 0)), int(sold),
                    float(sku.normal_asp), final_price, float(sku.cogs_wo_vat),
                    float(ending), float(n_hours - h_idx - 1),
                    sku.category, sku.subcategory,
                ))
                inv = ending
            # WRITE-OFF SENTINEL. The source zeroes ending_inventory on the
            # row where a listing closes, whatever remained. Emulating it is
            # not cosmetic: it is the signal common.episodes reads to tell a
            # closed episode from one still open, so a fixture without it
            # exercises a code path production never takes.
            if records and records[-1][0] == day:
                last = list(records[-1])
                last[10] = 0.0
                records[-1] = tuple(last)
            if len(records) > window_start:
                windows.append((window_start, len(records)))

    df = pd.DataFrame(records, columns=[f.name for f in SCHEMA])

    # ---- inject the dirt the canonical filter chain is specified to remove ----
    n = len(df)
    if n and dirty_frac > 0:
        k = max(1, int(n * dirty_frac))
        idx = rng.choice(n, size=k * 4, replace=False)
        a, b, c, e = np.array_split(idx, 4)
        df.loc[a, "category"] = None                      # null category
        df.loc[b, "subcategory"] = None                   # null subcategory
        df.loc[c, "normal_asp"] = 0.0                     # zero base price
        df.loc[e, "units_sold"] = df.loc[e, "inventory"].astype(int) + 3  # multi-lot

        # NEGATIVE WINDOW -- injected on WHOLE WINDOWS, from the first hour.
        #
        # It used to be written onto RANDOM ROWS, and that did not model the
        # source at all: a single corrupted counter mid-window reads as a
        # window BOUNDARY to `assign_episode_ids`, which differences the
        # counter hour to hour. One bad value shredded a clean 4-hour window
        # into three fragments -- 3, 2, [-1], 0 -- and the fragments that did
        # not carry the closing row came out `not_closed`. It manufactured
        # 62 of the fixture's 65 unclosed episodes, so `edge_truncated` read 0
        # against them and `share_of_unclosed_explained_by_edge` read 0: the
        # diagnostic that is supposed to say "the extract boundary explains
        # these" was being answered by an injection artifact instead. With
        # dirt off, unclosed fell from 3.9% to 0.2%.
        #
        # The real pattern is an episode ENTERING already negative, and it
        # stays an episode: the counter still decrements by one per hour, so
        # segmentation keeps the window whole, `negative_window_recovered`
        # rewrites it as a countdown from `manufacturing_window_hours`, and
        # the episode keeps its closing row.
        #
        # The source's other negative shape -- a flat negative CONSTANT on
        # every row -- is deliberately NOT injected. A flat counter differences
        # to zero, so it segments into one-row episodes; only the one holding
        # the window's final row would close, and the rest would come out
        # `not_closed`. That is the same artifact this change removes, so
        # injecting it here would put the noise back under a better name.
        if windows:
            k_win = max(1, int(len(windows) * dirty_frac))
            for w in rng.choice(len(windows), size=min(k_win, len(windows)),
                                replace=False):
                s, t = windows[w]
                entry = -float(rng.integers(1, 40))       # enters below zero
                df.loc[s:t - 1, "flc_window"] = entry - np.arange(t - s)

    df = df.sort_values(["skuseq", "fc", "date", "hour"]).reset_index(drop=True)
    return df, master


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skus", type=int, default=400)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--policy", choices=["legacy", "randomized"], default="legacy")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dirty-frac", type=float, default=0.004)
    # NOT dirt, and deliberately not folded into --dirty-frac: shrink is a
    # real event the source reports and the chain is specified to KEEP. It
    # has its own knob so a run can turn it off to isolate the leftover half
    # of the identity, which is the only reason to want a fixture without it.
    ap.add_argument("--shrink-rate", type=float, default=0.02,
                    help="per-hour probability of an interior inventory "
                         "shortfall (0 < ending < starting - sold)")
    ap.add_argument("--out", default="data/flc_filtered_synthetic.parquet")
    args = ap.parse_args()

    df, master = generate(args.skus, args.days, args.policy, args.seed,
                          args.dirty_frac, args.shrink_rate)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, args.out)

    excl = df[(df.date >= EXCLUSION_START) & (df.date <= EXCLUSION_END)]
    # The two convention signals, counted at the source. Both were silently
    # ABSENT from a committed fixture at some point, and each absence hid a
    # whole code path while every test passed -- the write-off sentinel for
    # months. Reported on every run so the next one cannot be silent.
    net = df.inventory - df.units_sold
    sentinel = int(((df.ending_inventory == 0) & (net > 0)).sum())
    shrink_rows = int(((df.ending_inventory > 0)
                       & (df.ending_inventory < net)).sum())
    print(f"policy            : {args.policy}")
    print(f"rows              : {len(df):,}")
    print(f"skus              : {df.skuseq.nunique():,}")
    print(f"episodes          : {df.groupby(['skuseq','fc','date']).ngroups:,}")
    print(f"zero-sale rows    : {(df.units_sold == 0).mean():.1%}")
    print(f"mean units_sold   : {df.units_sold.mean():.3f}")
    print(f"discount range    : {df.discount.min():.0f}-{df.discount.max():.0f} (percent)")
    print(f"final_price==0    : {(df.final_price == 0).mean():.1%} "
          f"(zero-sale rows: {(df.units_sold == 0).mean():.1%})")
    print(f"in exclusion win  : {len(excl):,} rows")
    print(f"write-off rows    : {sentinel:,} (ending==0 while stock remained "
          f"— MUST be > 0 or every episode reads unclosed)")
    print(f"shrink rows       : {shrink_rows:,} "
          f"(0 < ending < starting-sold, {shrink_rows/max(len(df),1):.2%})")
    print(f"negative-window   : {int((df.flc_window < 0).sum()):,} rows "
          f"(whole windows entering below zero, never a single row)")
    print(f"median d_max      : {(1 - df.cogs_wo_vat/df.normal_asp.replace(0,np.nan)).median():.3f}")
    print(f"corr(discount,hour) within episode: {corr_within(df):.3f}")
    print(f"wrote             : {args.out}")


def corr_within(df):
    """Mean within-episode correlation of discount and hour -- the confound."""
    g = df[df.normal_asp > 0].groupby(["skuseq", "fc", "date"])
    cs = []
    for _, sub in g:
        if len(sub) > 3 and sub.discount.std() > 0:
            cs.append(np.corrcoef(sub.discount, sub.hour)[0, 1])
    return float(np.nanmean(cs)) if cs else float("nan")


if __name__ == "__main__":
    main()
