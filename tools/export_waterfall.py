"""tools.export_waterfall -- the data-quality waterfall as a workbook, with
worked examples and the definitions beside it.

`artifacts/split_manifest.json` already carries the counts. What it cannot do
is answer the question anyone actually asks on being shown them: *what does one
of these removals look like?* A reader told that 24,002 episodes went for
`window_too_long` is entitled to see one, whole, hour by hour, and to check for
themselves that it deserved to go.

Three sheets, in the order someone reads them:

  waterfall     one row per stage, with rows, episodes and COGS at risk, what
                the stage cost in money and as a share of the raw exposure,
                and -- the part counts alone never gave -- whether the stage is
                a HARD DROP (rows leave the frame, gone for every consumer) or
                a POPULATION GATE (nothing is dropped; the consumers differ),
                plus who reads the population it produces.

  examples      three whole episodes per stage, every hour of them, drawn from
                the RAW feed as it arrived. Not a sample of rows: the entire
                episode, so the defect is visible in context rather than
                asserted. The removal reason travels with each block.

  definitions   every stage and every gate reason, in prose, written for
                someone who has not read the code.

The episode ids come from `bootstrap.prepare_data.load_and_filter` itself,
which records them as it drops them. This tool re-derives nothing: an exporter
carrying its own copy of the filter chain would disagree with the real one the
first time either changed, and it would do so silently, in the document meant
to establish trust.

Usage:
    python3 -m tools.export_waterfall --input data/flc.parquet \
        --out reports/data_quality.xlsx [--html] [--examples 3]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from common.config import load_config
from bootstrap.prepare_data import (
    load_and_filter, waterfall_rows, SOURCE_TO_PRD, assign_episode_ids,
    WATERFALL_STEPS, DP_INELIGIBLE, BELOW_COST_HOURS)

# What a reader needs to see to judge a removal, in the order they read it:
# which episode and when, then the inventory chain, then the price.
# PRD names, because the frame is renamed on load -- `flc_window` is
# `hours_remaining` here, which is also what every rule in the definitions
# sheet calls it, so the two sheets use one vocabulary.
RAW_COLS = ["episode_id", "removed_at_step", "why", "date", "hour_of_day",
            "sku_id", "fc", "category", "subcategory", "hours_remaining",
            "starting_inventory", "units_sold", "ending_inventory",
            "chain_break", "hours_claimed_by_counter", "hours_present",
            "original_price", "total_discount", "cost"]


def _raw(path):
    """The feed as it arrived, with episode ids assigned the same way the
    pipeline assigns them -- otherwise the ids recorded during the drop would
    not match anything here."""
    df = pd.read_parquet(path).rename(columns=SOURCE_TO_PRD)
    # returns a Series, same as the pipeline's own use of it at load time
    df["episode_id"] = assign_episode_ids(df)
    return df


def examples_frame(raw, examples, gate_examples):
    """Whole episodes, every hour, with the reason attached.

    `chain_break` is computed and shown rather than left implicit: for the
    continuity drops it IS the defect, and a reader scanning inventory columns
    by eye will not spot `ending[t] != starting[t+1]` reliably. For every other
    stage it is a useful negative -- the chain is fine, something else was
    wrong.
    """
    why = dict((label, text) for label, _, text in WATERFALL_STEPS)
    why.update((name, text) for name, text in DP_INELIGIBLE)
    blocks = []
    for step, ids in list(examples.items()) + list(gate_examples.items()):
        for eid in ids:
            g = raw[raw.episode_id == eid].sort_values(["date", "hour_of_day"])
            if g.empty:
                continue
            g = g.copy()
            g["removed_at_step"] = step
            g["why"] = why.get(step, "")
            end = g.ending_inventory.to_numpy()[:-1]
            nxt = g.starting_inventory.to_numpy()[1:]
            g["chain_break"] = np.append(end != nxt, False)
            # THE GAP-SPLIT DEFECT IS INVISIBLE IN A FRAGMENT SHOWN ALONE.
            # Nothing in a two-row block looks wrong until you notice its
            # counter opens at 6, which claims a seven-hour window. This is the
            # same disagreement `validate_state` refuses at run time, and
            # printing both numbers is what lets a reader see it without
            # being told.
            if "hours_remaining" in g:
                g["hours_claimed_by_counter"] = float(
                    g.hours_remaining.iloc[0]) + 1
                g["hours_present"] = len(g)
            blocks.append(g[[c for c in RAW_COLS if c in g]])
    if not blocks:
        return pd.DataFrame(columns=RAW_COLS)
    return pd.concat(blocks, ignore_index=True)


def gate_examples_from(prepared, n):
    """Examples for the two GATE rows, which drop nothing and so never appear
    in the drop-time capture. Their reason is a column on the prepared frame,
    so these are read straight off it rather than reconstructed."""
    out = {}
    if "dp_ineligible_reason" in prepared:
        for reason, g in prepared[
                prepared.dp_ineligible_reason.notna()].groupby(
                    "dp_ineligible_reason"):
            out[str(reason)] = sorted(g.episode_id.unique())[:n]
    if "episode_eligible" in prepared:
        # ONLY THE CASES THE DP REASONS DO NOT ALREADY SHOW. Two of the three
        # `eligible` conditions have a dp_eligible reason of their own
        # (`outcome_unknown`, `final_hour_restock`), so picking freely here
        # would print the same episode twice under two headings and teach a
        # reader nothing the second time. What is left is the third condition,
        # `accounting_closes` -- and where the extract has none of those, the
        # heading is dropped rather than filled with a duplicate.
        shown = {e for ids in out.values() for e in ids}
        bad = prepared[~prepared.episode_eligible]
        fresh = [e for e in sorted(bad.episode_id.unique()) if e not in shown]
        if fresh:
            out["eligible"] = fresh[:n]
    return out


def definitions_frame(cfg):
    """Every stage and every gate reason, in prose. Sourced from the tables in
    `bootstrap.prepare_data`, so a rule that changes in the code changes here
    without anyone remembering to edit a document."""
    rows = []
    for label, scope, text in WATERFALL_STEPS:
        rows.append({"kind": "population_gate" if "GATE" in scope
                     else "waterfall stage",
                     "name": label, "scope": scope, "definition": text})
    for name, text in DP_INELIGIBLE:
        rows.append({"kind": "dp_eligible reason", "name": name,
                     "scope": "episode -- FLAGGED, not dropped",
                     "definition": text})
    rows.append({"kind": "reported only", "name": "below_cost_hours",
                 "scope": "episode -- reported, NOT gating",
                 "definition": BELOW_COST_HOURS + ". Stays dp_eligible: a "
                 "below-cost price is one the LEGACY policy set, and the DP "
                 "refusing to match it is the cost floor working rather than "
                 "a reason to delete the episode"})
    rows.append({"kind": "how to read it", "name": "counts overlap",
                 "scope": "--",
                 "definition": "The per-reason counts on the dp_eligible row "
                 "are UNCONDITIONAL: an episode tripping three tests is "
                 "counted in all three, so they do not sum to "
                 "episodes_dp_ineligible. Only the `dp_ineligible_reason` "
                 "column is first-match, in the order listed above"})
    rows.append({"kind": "how to read it", "name": "nested populations",
                 "scope": "--",
                 "definition": "integrity > eligible > dp_eligible. The three "
                 "are NESTED, not disjoint, so their exclusions must never be "
                 "added together. `integrity` has no row of its own: it is the "
                 "last hard-drop row, negative_window_recovered"})
    rows.append({"kind": "how to read it", "name": "cogs_at_risk",
                 "scope": "won",
                 "definition": "unit cost x SUPPLY -- opening stock PLUS gross "
                 "arrivals -- counted ONCE per "
                 "episode, never summed over hours -- inventory persists, so a "
                 "per-row sum would multiply the same stock by the window "
                 "length. Reported beside row and episode counts because the "
                 "two disagree and the disagreement is the point: a filter can "
                 "take 1% of the rows and 15% of the money"})
    return pd.DataFrame(rows)


# The waterfall sheet's readable core. Every stage also reports its own detail
# block -- gap counts, restock counts, the six dp_eligible reasons -- and
# flattening those into columns gave a 45-column sheet where 30 columns were
# blank on any given row. They go into one `details` column as compact JSON
# instead: nothing is lost, and the sheet is legible without horizontal
# scrolling past thirty empty cells.
WF_CORE = ("step", "kind", "rows", "episodes", "cogs_at_risk",
           "cogs_pct_of_raw", "cogs_dropped", "cogs_dropped_pct_of_raw",
           "used_by")


def waterfall_sheet(rows):
    out = []
    for r in rows:
        keep = {k: r.get(k) for k in WF_CORE}
        extra = {k: v for k, v in r.items() if k not in WF_CORE}
        keep["details"] = json.dumps(extra, default=str) if extra else ""
        out.append(keep)
    return pd.DataFrame(out)


def build(input_path, cfg, n_examples=3):
    examples = {}
    prepared, wf = load_and_filter(input_path, cfg, examples=examples,
                                   examples_per_step=n_examples)
    raw = _raw(input_path)
    gates = gate_examples_from(prepared, n_examples)
    return {
        "waterfall": waterfall_sheet(waterfall_rows(wf, cfg)),
        "examples": examples_frame(raw, examples, gates),
        "definitions": definitions_frame(cfg),
    }


def _autosize(writer, sheet, frame):
    ws = writer.sheets[sheet]
    for i, col in enumerate(frame.columns):
        width = max(len(str(col)),
                    int(frame[col].astype(str).str.len().max() or 0))
        # the definition column is prose and would otherwise be one very long
        # line; cap it and let Excel wrap
        ws.set_column(i, i, min(max(width + 2, 10), 60))
    ws.freeze_panes(1, 0)


def write_workbook(path, sheets):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        for name, frame in sheets.items():
            frame.to_excel(w, sheet_name=name, index=False)
            _autosize(w, name, frame)
    return path


def write_html(path, sheets):
    """The same three sheets as one page. Unlike the backtest export this one
    is small enough to show whole -- that is the point of it."""
    def table(df):
        return df.to_html(index=False, border=0,
                          float_format=lambda x: f"{x:,.6g}")

    wf = sheets["waterfall"].copy()
    # `used_by` is the column the reader most needs and the one Excel hides
    # behind a scrollbar; on the page it gets its own line under each stage
    show = [c for c in ("step", "kind", "rows", "episodes", "cogs_at_risk",
                        "cogs_pct_of_raw", "cogs_dropped",
                        "cogs_dropped_pct_of_raw") if c in wf]
    rows = []
    for _, r in wf.iterrows():
        cells = "".join(f"<td>{r[c]:,}</td>" if isinstance(r[c], (int, float))
                        and not pd.isna(r[c]) else f"<td>{r.get(c, '')}</td>"
                        for c in show)
        rows.append(f"<tr class='{r.get('kind', '')}'>{cells}</tr>")
        if r.get("kind") == "population_gate":
            rows.append(f"<tr class='used'><td colspan='{len(show)}'>"
                        f"read by: {r.get('used_by', '')}</td></tr>")
    head = "".join(f"<th>{c}</th>" for c in show)

    ex = sheets["examples"]
    blocks = []
    for (step, eid), g in ex.groupby(["removed_at_step", "episode_id"],
                                     sort=False):
        blocks.append(
            f"<h3>{step} &mdash; <code>{eid}</code></h3>"
            f"<p class='why'>{g.why.iloc[0]}</p>"
            + table(g.drop(columns=["why", "removed_at_step"], errors="ignore")))

    html = f"""<meta charset="utf-8"><title>Data quality waterfall</title>
<style>
 body{{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
      max-width:1200px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:2.5rem}}
 h3{{font-size:.95rem;margin:1.6rem 0 .2rem;font-weight:600}}
 table{{border-collapse:collapse;width:100%;font-size:12px;margin:.4rem 0 1rem}}
 th{{text-align:left;border-bottom:2px solid #1a1a1a;padding:4px 8px}}
 td{{border-bottom:1px solid #e6e2d8;padding:3px 8px}}
 tr.population_gate td{{background:#f4efe4;font-weight:600}}
 tr.used td{{background:#f4efe4;font-size:11px;font-weight:400;color:#5a5348}}
 p.why{{font-size:12px;color:#5a5348;margin:.1rem 0 .4rem;max-width:80ch}}
 code{{font-size:12px}}
</style>
<h1>Data quality waterfall</h1>
<p>Rows above the two shaded lines are <b>hard drops</b> &mdash; those rows
leave the frame and no consumer ever sees them. The two shaded rows are
<b>population gates</b>: they drop nothing at all, they only decide who reads
what. The three populations are <b>nested</b>
(integrity &sup; eligible &sup; dp_eligible), so their exclusions must never be
added together.</p>
<table><tr>{head}</tr>{''.join(rows)}</table>
<h2>What a removal looks like</h2>
<p>Whole episodes from the raw feed, every hour of them, so the defect can be
seen in context rather than taken on trust. <code>chain_break</code> marks an
hour whose <code>ending_inventory</code> does not match the next hour's
<code>starting_inventory</code>.</p>
{''.join(blocks)}
<h2>Definitions</h2>
{table(sheets['definitions'])}
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="the RAW flc parquet")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="reports/data_quality.xlsx")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--examples", type=int, default=3,
                    help="whole episodes to show per removal reason")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sheets = build(args.input, cfg, args.examples)
    write_workbook(args.out, sheets)

    wf = sheets["waterfall"]
    print(f"waterfall stages : {len(wf)}")
    print(f"example episodes : "
          f"{sheets['examples'].episode_id.nunique() if len(sheets['examples']) else 0}"
          f"  across {sheets['examples'].removed_at_step.nunique() if len(sheets['examples']) else 0} reasons")
    print(f"definitions      : {len(sheets['definitions'])}")
    print(f"wrote            : {args.out}")
    if args.html:
        p = os.path.splitext(args.out)[0] + ".html"
        write_html(p, sheets)
        print(f"wrote            : {p}")


if __name__ == "__main__":
    main()
