"""tools.export_backtest -- the backtest's three arms, hour by hour, as a
workbook.

`reports/backtest.json` reports aggregates. This exports the rows underneath
them: one line per episode-hour with the observed world, the legacy policy
under the model, and the DP side by side, so a category owner can look at an
episode and see where the DP would have priced differently and what the model
thought would happen.

It does NOT re-run the pricing logic. `backtest.replay._replay_one` grew an
opt-in per-hour trace and this reads it, because the replay protocol (design 5.14) forbids a parallel
implementation and an exporter with its own copy of the arms would drift the
first time either one changed.

READ THE ARMS CORRECTLY, or the sheets will mislead:

  actual   the observed world at the prices the legacy policy actually set.
           `actual_units` is what really sold.
  legacy   the SAME prices, but demand from the model. Differs from `actual`
           only by model error, so `actual_units` vs `legacy_units` is a
           FIDELITY read, not a policy one.
  dp       the DP's own prices, same demand model as `legacy`.

So the policy comparison is legacy vs dp -- both simulated, so model bias hits
them identically. Comparing dp against ACTUAL mixes policy difference with
model error and flatters whichever arm the model happens to favour; design 5.14
is explicit that a replay under-predicting demand always flatters a
price-holding policy.

INVENTORY, per arm, per hour: `*_start_inv` opens the hour, `*_end_inv` closes
it -- after sales AND after that hour's restock/shrink (`hour_adjustment`), the
source's own convention, so `*_end_inv[t]` carries into `*_start_inv[t+1]`. The
arms hold DIFFERENT stock from the first hour they price differently, which is
the point. Three things a reader will notice and none is a defect:

  * `actual_end_inv` is 0 on a closed episode's LAST row whatever remained --
    the source writes the remainder off there, and that zero IS the scrap
    (design 12a). Read the leftover off `actual_start_inv - actual_units`.
  * the simulated arms carry FRACTIONAL inventory, because each hour removes
    E[min(D,q)] rather than a whole-unit draw (docs/stockout_gap.html).
  * on those arms `*_start_inv` is the INTEGER shelf the solver priced
    (`int(round(q))`) while `*_end_inv` is the exact carry, so consecutive
    hours can differ by up to half a unit. The DP prices integer inventory;
    rounding once at the decision is what that costs.

WHERE THE UNITS WENT, per arm: every unit an episode had either sold, shrank,
or was scrapped at the close -- there is no fourth fate, so the hourly sheet
accounts for all of them:

    sum(*_units) + sum(*_shrink) + sum(*_scrap_at_close)  ==  supply

`*_shrink` is an HOURLY event (the exogenous loss in `hour_adjustment`), and
each simulated arm absorbs only what its own shelf still held -- units an arm
already sold cannot also shrink, so the arms can differ here and their totals
differ from the observed `shrink`. `*_scrap_at_close` is a TERMINAL event and
is non-zero only on the episode's LAST row; it is zero throughout an unfinished
episode, whose stock is still on the shelf when the extract ends. The episode
sheet's `*_scrap_units` is the sum of the two.

Usage:
    python3 -m tools.export_backtest --input data/prepared.parquet \
        --out reports/backtest_episodes.xlsx --episodes 300 [--html]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config
from bootstrap.prepare_data import population, pre_launch
from bootstrap.train_baseline import BaselineModel
from backtest.replay import fidelity, policy_replay

# Column order for the hourly sheet: identity, then the three arms in blocks,
# so the eye can run down one arm without hopping between columns.
HOUR_COLS = [
    "episode_id", "date", "hour_of_day", "t", "is_observed",
    "sku_id", "fc", "category",
    "original_price", "cost", "d_ref", "mu_ref", "hour_adjustment",
    # each arm: what it opened the hour with, what it sold, what it closed
    # with. `*_end_inv` is AFTER sales and AFTER the hour's restock/shrink, so
    # `*_end_inv[t] == *_start_inv[t+1]` down every arm.
    "actual_start_inv", "actual_discount", "actual_price", "actual_units",
    "actual_shrink", "actual_scrap_at_close", "actual_end_inv",
    "legacy_start_inv", "legacy_discount", "legacy_price", "legacy_mu",
    "legacy_units", "legacy_shrink", "legacy_scrap_at_close", "legacy_end_inv",
    "dp_start_inv", "dp_discount", "dp_price", "dp_mu", "dp_units",
    "dp_shrink", "dp_scrap_at_close", "dp_end_inv",
    "dp_is_entry", "dp_feasible_tiers", "dp_minus_actual_discount",
]

EP_COLS = [
    "episode_id", "date", "sku_id", "fc", "category",
    "q0", "supply", "shrink", "end_inv", "hours", "n_observed",
    "outcome_known", "original_price", "cost", "d_ref", "eps", "r",
    "deepening_threshold",
    # UNITS BEFORE MONEY in each arm's block. A category owner checks the
    # physical story first -- how much sold, how much was left, how much was
    # scrapped -- and the currency figures are those quantities priced.
    "actual_sold_units", "actual_leftover_units", "actual_scrap_units",
    "actual_il", "actual_discount_cost", "actual_scrap_cost",
    "actual_denom", "actual_cleared", "actual_mean_discount",
    "legacy_model_sold_units", "legacy_model_leftover_units",
    "legacy_model_shrink_applied", "legacy_model_scrap_units",
    "legacy_model_il", "legacy_model_discount_cost", "legacy_model_scrap_cost",
    "legacy_model_denom", "legacy_model_cleared", "legacy_model_mean_discount",
    "dp_sold_units", "dp_leftover_units", "dp_shrink_applied",
    "dp_scrap_units",
    "dp_il", "dp_discount_cost", "dp_scrap_cost",
    "dp_denom", "dp_cleared", "dp_mean_discount",
]

# Per arm: the episode total, the column on the hourly sheet that must sum to
# it, and the identity residual. One list so the episode sheet, the summary and
# the reconciliation sheet cannot disagree about what reconciles with what.
ARMS = (("actual", "actual_units"),
        ("legacy_model", "legacy_units"),
        ("dp", "dp_units"))


def build(input_path, cfg, episodes_n, seed=0, workers=None):
    """Run the backtest with tracing on. Same setup as `backtest.__main__`."""
    d = pd.read_parquet(input_path)
    # the backtest is a PRE-LAUNCH artifact: it must see nothing past the gate
    # window, and the DP cannot price an ineligible episode
    d = pre_launch(d, cfg)
    d = population(d, cfg, "dp_eligible")
    if d.empty:
        raise SystemExit("no dp_eligible episodes on or before split.test_end")

    model = BaselineModel(cfg)
    with open(cfg["posterior"]["prior"]["path"]) as f:
        prior = json.load(f)
    with open(cfg["dispersion"]["r_lookup_path"]) as f:
        r_lookup = json.load(f)

    _, d_pred = fidelity(d, cfg, model, prior, r_lookup)
    _, ep, _, hourly = policy_replay(d_pred, cfg, max_episodes=episodes_n,
                                     seed=seed, workers=workers, trace=True)
    return ep, hourly


def _episode_view(ep):
    e = ep.copy()
    # what the workbook is FOR: the like-for-like policy gap, per episode.
    # Both simulated arms, so model bias cancels -- see the module docstring.
    e["dp_minus_legacy_il"] = e.dp_il - e.legacy_model_il
    e["dp_minus_legacy_cleared"] = e.dp_cleared - e.legacy_model_cleared
    e["dp_minus_legacy_mean_discount"] = (e.dp_mean_discount
                                          - e.legacy_model_mean_discount)
    # and the bar the DP has to clear before deepening ever pays, per episode
    e["eps_clears_deepening_bar"] = e.eps.abs() > e.deepening_threshold
    cols = [c for c in EP_COLS if c in e] + [
        "dp_minus_legacy_il", "dp_minus_legacy_cleared",
        "dp_minus_legacy_mean_discount", "eps_clears_deepening_bar"]
    return e[cols].sort_values("dp_minus_legacy_il")


def _reconciliation(ep, hourly, tol=1e-6):
    """Does the episode sheet agree with the hourly sheet, row by row?

    THE POINT OF PUTTING IT IN THE WORKBOOK rather than in a test: a reader
    who spots a number they do not believe needs to be able to check it
    themselves, on the episode in front of them, without running anything. Two
    identities per arm, per episode:

      hourly sum of units  ==  the episode's *_sold_units
      supply               ==  sold + leftover + shrink

    The second is `accounting_closes` for the observed arm and holds by
    construction for the two simulated ones, since the replay loop moves every
    unit it starts with. So a non-zero residual is a real defect, never
    rounding -- which is why the tolerance is 1e-6 and not something forgiving.

    An UNCLOSED episode reconciles on the hourly identity but not the supply
    one, and that is correct rather than a failure: nothing is scrapped
    because the outcome was never observed, so the units are still on the
    shelf when the extract ends. Those rows are marked rather than flagged.
    """
    sums = hourly.groupby("episode_id").agg(
        **{f"hourly_{col}": (col, "sum") for _, col in ARMS},
        hourly_rows=("t", "size"))
    e = ep.set_index("episode_id").join(sums, how="left")
    out = pd.DataFrame({
        "episode_id": e.index, "category": e.category.to_numpy(),
        "outcome_known": e.outcome_known.to_numpy(),
        "hours": e.hours.to_numpy(), "hourly_rows": e.hourly_rows.to_numpy(),
        "supply": e.supply.to_numpy(), "shrink": e.shrink.to_numpy(),
    })
    ok = np.ones(len(e), bool)
    for arm, col in ARMS:
        ep_units = e[f"{arm}_sold_units"].to_numpy()
        hr_units = e[f"hourly_{col}"].fillna(0.0).to_numpy()
        out[f"{arm}_episode_sold"] = ep_units
        out[f"{arm}_hourly_sold"] = hr_units
        out[f"{arm}_hourly_gap"] = ep_units - hr_units
        out[f"{arm}_supply_residual"] = e[f"{arm}_supply_residual"].to_numpy()
        ok &= np.abs(ep_units - hr_units) <= tol
        # the supply identity is only claimed where the outcome is known
        ok &= (np.abs(e[f"{arm}_supply_residual"].to_numpy()) <= tol) | \
              (~e.outcome_known.to_numpy())
    out["reconciles"] = ok
    out["note"] = np.where(
        ok, "",
        np.where(~e.outcome_known.to_numpy(),
                 "unclosed episode -- supply identity not claimed",
                 "MISMATCH -- episode and hourly sheets disagree"))
    return out.sort_values(["reconciles", "episode_id"])


def _summary(ep):
    """Ratio-of-sums, never a mean of per-episode ratios, and always over
    outcome_known -- the same rule policy_replay applies. An unfinished
    episode has a truncated actual arm graded against two full-horizon
    simulated ones."""
    k = ep[ep.outcome_known]
    rows = []

    def add(name, value, note=""):
        rows.append({"metric": name, "value": value, "note": note})

    add("episodes_traced", int(len(ep)))
    add("episodes_outcome_known", int(len(k)),
        "every figure below is over these only")
    add("episodes_excluded_unclosed", int(len(ep) - len(k)),
        "truncated actual arm; excluded rather than averaged in")
    for arm in ("actual", "legacy_model", "dp"):
        il = float(k[f"{arm}_il"].sum())
        den = float(k[f"{arm}_denom"].sum())
        add(f"{arm}_il", round(il, 1))
        add(f"{arm}_il_pct", round(il / den, 6) if den else None,
            "ratio of sums; denominator is original_price x units_sold")
        add(f"{arm}_cleared_weighted",
            round(float((k[f"{arm}_cleared"] * k.supply).sum()
                        / max(k.supply.sum(), 1)), 4))
    add("dp_minus_legacy_il", round(float(k.dp_il.sum()
                                          - k.legacy_model_il.sum()), 1),
        "THE policy comparison -- both arms simulated, model bias cancels")
    add("dp_minus_actual_il", round(float(k.dp_il.sum()
                                          - k.actual_il.sum()), 1),
        "NOT a clean policy read: mixes policy difference with model error")
    add("share_episodes_clearing_deepening_bar",
        round(float((k.eps.abs() > k.deepening_threshold).mean()), 4),
        "|eps| > (1-d)/(gamma-d). At 0 the DP enters and holds by design")

    # UNITS, beside the money. Scrap is what the whole system exists to
    # reduce, and an owner reading an IL delta in won wants to know how many
    # physical units moved to produce it.
    for arm in ("actual", "legacy_model", "dp"):
        add(f"{arm}_sold_units", round(float(k[f"{arm}_sold_units"].sum()), 1))
        add(f"{arm}_scrap_units", round(float(k[f"{arm}_scrap_units"].sum()), 1),
            "leftover at the close PLUS shrink -- units paid for that "
            "returned no revenue")
    add("dp_minus_legacy_scrap_units",
        round(float(k.dp_scrap_units.sum()
                    - k.legacy_model_scrap_units.sum()), 1),
        "THE physical policy comparison, the units counterpart of "
        "dp_minus_legacy_il")
    add("shrink_units", round(float(k.shrink.sum()), 1),
        "EXOGENOUS, observed-world total. Each simulated arm absorbs only "
        "what its own shelf still held when the shrink landed "
        "(*_shrink_applied) -- units an arm already sold cannot also shrink")
    return pd.DataFrame(rows)


def _autosize(writer, sheet, frame):
    ws = writer.sheets[sheet]
    for i, col in enumerate(frame.columns):
        width = max(len(str(col)),
                    int(frame[col].astype(str).str.len().max() or 0))
        ws.set_column(i, i, min(max(width + 2, 10), 42))
    ws.freeze_panes(1, 0)


def write_workbook(path, ep, hourly):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sheets = {
        "summary": _summary(ep),
        "episodes": _episode_view(ep),
        "hourly": hourly[[c for c in HOUR_COLS if c in hourly]],
        # so a reader who doubts a number can check it on the episode in
        # front of them, without running anything
        "reconciliation": _reconciliation(ep, hourly),
    }
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        for name, frame in sheets.items():
            frame.to_excel(w, sheet_name=name, index=False)
            _autosize(w, name, frame)
    return sheets


def write_html(path, sheets, episodes_shown=40):
    """A browsable version of the same thing. Deliberately small: the full
    hourly frame is tens of thousands of rows and a single HTML table of that
    size is unusable, so this shows the summary, every episode, and the hours
    of the episodes where the DP differed MOST from legacy -- which is what
    anyone opening this is looking for."""
    ep = sheets["episodes"]
    hourly = sheets["hourly"]
    focus = ep.reindex(ep.dp_minus_legacy_il.abs().sort_values(
        ascending=False).index).head(episodes_shown)
    hrs = hourly[hourly.episode_id.isin(focus.episode_id)]

    def table(df, cls=""):
        return df.to_html(index=False, float_format=lambda x: f"{x:,.4g}",
                          classes=cls, border=0, na_rep="")

    html = f"""<title>Backtest episodes</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --line:#e3e3e3; --head:#f6f6f6;
           --muted:#666; --accent:#0b5fff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#151515; --fg:#e8e8e8; --line:#333; --head:#1f1f1f;
             --muted:#9a9a9a; --accent:#6ea8ff; }} }}
  body {{ background:var(--bg); color:var(--fg); margin:0 auto; max-width:1600px;
          padding:2rem 1.25rem; font:14px/1.5 ui-sans-serif,system-ui,sans-serif; }}
  h1 {{ font-size:1.4rem; margin:0 0 .25rem; }}
  h2 {{ font-size:1.05rem; margin:2rem 0 .5rem; }}
  p.note {{ color:var(--muted); max-width:70ch; }}
  div.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; }}
  table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
  th,td {{ padding:.35rem .55rem; text-align:right; white-space:nowrap;
           border-bottom:1px solid var(--line); }}
  th {{ background:var(--head); position:sticky; top:0; text-align:right;
        font-weight:600; }}
  td:nth-child(-n+3), th:nth-child(-n+3) {{ text-align:left; }}
  tbody tr:hover {{ background:var(--head); }}
</style>
<h1>Backtest episodes</h1>
<p class="note">Three arms. <b>actual</b> is the observed world at the prices
legacy really set. <b>legacy</b> is those same prices under the model's
demand, so actual-vs-legacy is a <i>fidelity</i> read. <b>dp</b> is the DP's
own prices under that same model. The policy comparison is
<b>legacy vs dp</b> — both simulated, so model bias cancels. Comparing dp
against actual mixes policy difference with model error.</p>

<h2>Summary</h2>
<div class="scroll">{table(sheets['summary'])}</div>

<h2>Episodes ({len(ep):,}), by policy gap</h2>
<div class="scroll">{table(ep)}</div>

<h2>Hourly — the {len(focus):,} episodes where the arms diverged most</h2>
<p class="note">{len(hourly):,} hourly rows in total; the workbook has all of
them. Rows with <code>is_observed = False</code> are the synthetic window
extension — both simulated arms run the full horizon, the actual arm does
not.</p>
<div class="scroll">{table(hrs)}</div>
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(prog="export_backtest")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="reports/backtest_episodes.xlsx")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--episodes", type=int, default=300,
                    help="episodes to trace. The hourly sheet is roughly this "
                         "x the mean window length, and Excel stops at "
                         "1,048,576 rows per sheet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--html", nargs="?", const="docs/backtest_episodes.html",
                    default=None, help="also write a browsable HTML view")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ep, hourly = build(args.input, cfg, args.episodes, args.seed, args.workers)
    sheets = write_workbook(args.out, ep, hourly)

    print(f"episodes traced : {len(ep):,}")
    print(f"hourly rows     : {len(hourly):,}")
    print(f"wrote           : {args.out}")
    if args.html:
        write_html(args.html, sheets)
        print(f"wrote           : {args.html}")


if __name__ == "__main__":
    main()
