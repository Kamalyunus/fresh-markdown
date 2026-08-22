"""tools.eda_page -- docs/eda.html, drawn from reports/eda.json and nothing else.

A pure view. Every number and every series comes from the report, so the page
cannot show a figure the JSON does not contain -- the failure the walkthrough
needed a whole provenance ledger to prevent, avoided here by not having a
second source in the first place.

Charts are inline SVG rather than PNGs: one self-contained file, no image
directory to fall out of step with the data, and it scales. Four kinds cover
every panel -- line, bars, hist, pareto -- and a panel declaring anything else
renders its numbers and no picture, which is the right failure.
"""

import html
import json

W, H = 760, 240
PAD = {"l": 62, "r": 16, "t": 14, "b": 34}


def _esc(x):
    return html.escape(str(x), quote=True)


def _fmt(v):
    if isinstance(v, bool) or v is None:
        return "—" if v is None else str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v and (abs(v) >= 100_000):
            return f"{v:,.0f}"
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return _esc(v)


def _axes(ymax, y_label="", x_label="", ticks=4):
    """Horizontal gridlines with labels, plus the two axis captions."""
    iw, ih = W - PAD["l"] - PAD["r"], H - PAD["t"] - PAD["b"]
    out = []
    for i in range(ticks + 1):
        y = PAD["t"] + ih - ih * i / ticks
        out.append(f'<line class="grid" x1="{PAD["l"]}" y1="{y:.1f}" '
                   f'x2="{W - PAD["r"]}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{PAD["l"] - 8}" y="{y + 3.5:.1f}" '
                   f'text-anchor="end">{_fmt(ymax * i / ticks)}</text>')
    if y_label:
        out.append(f'<text class="axis" x="6" y="{PAD["t"] + 10}">'
                   f'{_esc(y_label)}</text>')
    if x_label:
        out.append(f'<text class="axis" x="{W - PAD["r"]}" y="{H - 4}" '
                   f'text-anchor="end">{_esc(x_label)}</text>')
    return "".join(out)


def _svg(body):
    return (f'<svg viewBox="0 0 {W} {H}" class="chart" '
            f'preserveAspectRatio="xMidYMid meet" role="img">{body}</svg>')


def _line(c):
    xs, series = c["x"], c["series"]
    if not xs:
        return ""
    iw, ih = W - PAD["l"] - PAD["r"], H - PAD["t"] - PAD["b"]
    ymax = max((max(v) for v in series.values() if v), default=1) or 1
    body = [_axes(ymax, c.get("y_label", ""), c.get("x_label", ""))]
    for i, (name, vals) in enumerate(series.items()):
        pts = " ".join(
            f'{PAD["l"] + iw * (j / max(len(xs) - 1, 1)):.1f},'
            f'{PAD["t"] + ih - ih * (v / ymax):.1f}'
            for j, v in enumerate(vals))
        body.append(f'<polyline class="s{i}" points="{pts}"/>')
        body.append(f'<text class="legend s{i}t" x="{PAD["l"] + 6}" '
                    f'y="{PAD["t"] + 12 + i * 14}">{_esc(name)}</text>')
    # first and last x label only: 160 dates will not fit and do not need to
    body.append(f'<text class="tick" x="{PAD["l"]}" y="{H - 16}">{_esc(xs[0])}</text>')
    body.append(f'<text class="tick" x="{W - PAD["r"]}" y="{H - 16}" '
                f'text-anchor="end">{_esc(xs[-1])}</text>')
    return _svg("".join(body))


def _bars(c):
    labels, vals = c["labels"], c["values"]
    if not labels:
        return ""
    iw, ih = W - PAD["l"] - PAD["r"], H - PAD["t"] - PAD["b"]
    ymax = max(vals) or 1
    step = iw / len(labels)
    body = [_axes(ymax, c.get("y_label", ""), c.get("x_label", ""))]
    for i, (lab, v) in enumerate(zip(labels, vals)):
        h = ih * (v / ymax)
        x = PAD["l"] + i * step + step * 0.15
        body.append(f'<rect class="bar" x="{x:.1f}" y="{PAD["t"] + ih - h:.1f}" '
                    f'width="{step * 0.7:.1f}" height="{max(h, 0.5):.1f}"><title>'
                    f'{_esc(lab)}: {_fmt(v)}</title></rect>')
        if len(labels) <= 26:
            body.append(f'<text class="tick" x="{x + step * 0.35:.1f}" '
                        f'y="{H - 18}" text-anchor="middle">{_esc(lab)}</text>')
    return _svg("".join(body))


def _hist(c):
    edges, counts = c.get("edges", []), c.get("counts", [])
    if not counts:
        return ""
    labels = [f"{edges[i]:g}–{edges[i + 1]:g}" for i in range(len(counts))]
    return _bars({"labels": labels, "values": counts,
                  "y_label": c.get("y_label", ""),
                  "x_label": c.get("x_label", "")}) if len(counts) <= 26 else \
        _bars({"labels": [""] * len(counts), "values": counts,
               "y_label": c.get("y_label", ""),
               "x_label": f'{c.get("x_label", "")}  '
                          f'({edges[0]:g} → {edges[-1]:g})'})


def _pareto(c):
    xs, ys = c["x"], c["y"]
    if not xs:
        return ""
    iw, ih = W - PAD["l"] - PAD["r"], H - PAD["t"] - PAD["b"]
    body = [_axes(1.0, c.get("y_label", ""), c.get("x_label", ""))]
    # the line a perfectly even distribution would trace, for scale
    body.append(f'<line class="ref" x1="{PAD["l"]}" y1="{PAD["t"] + ih}" '
                f'x2="{W - PAD["r"]}" y2="{PAD["t"]}"/>')
    pts = " ".join(f'{PAD["l"] + iw * x:.1f},{PAD["t"] + ih - ih * y:.1f}'
                   for x, y in zip(xs, ys))
    body.append(f'<polyline class="s0" points="{pts}"/>')
    return _svg("".join(body))


KINDS = {"line": _line, "bars": _bars, "hist": _hist, "pareto": _pareto}


def _chart(c):
    if not c or c.get("kind") not in KINDS:
        return ""
    return f'<figure class="ch">{KINDS[c["kind"]](c)}</figure>'


def _value_block(body):
    """Everything in the panel that is not the chart, as definition rows.

    Nested dicts become their own sub-table rather than JSON in a cell: these
    are read, not parsed.
    """
    skip = {"title", "lede", "informs", "chart"}
    out = []
    for k, v in body.items():
        if k in skip:
            continue
        if k == "note":
            out.append(f'<p class="note">{_esc(v)}</p>')
        elif isinstance(v, dict) and v:
            rows = "".join(
                f"<tr><th>{_esc(kk)}</th><td>"
                + (", ".join(f"{_esc(a)} {_fmt(b)}" for a, b in vv.items())
                   if isinstance(vv, dict) else _fmt(vv)) + "</td></tr>"
                for kk, vv in v.items())
            out.append(f'<div class="sub"><h4>{_esc(k)}</h4>'
                       f'<div class="scroller"><table>{rows}</table></div></div>')
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            cols = list(v[0])
            head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
            rows = "".join("<tr>" + "".join(f"<td>{_fmt(r.get(c))}</td>"
                                            for c in cols) + "</tr>" for r in v)
            out.append(f'<div class="sub"><h4>{_esc(k)}</h4><div class="scroller">'
                       f'<table><thead><tr>{head}</tr></thead>{rows}</table>'
                       f'</div></div>')
        elif isinstance(v, list):
            out.append(f'<div class="kv"><span>{_esc(k)}</span>'
                       f'<b>{_esc(", ".join(map(str, v)) or "none")}</b></div>')
        else:
            out.append(f'<div class="kv"><span>{_esc(k)}</span>'
                       f'<b>{_fmt(v)}</b></div>')
    return "".join(out)


CSS = """
:root{--bg:#fbfaf8;--ink:#1a1a1a;--muted:#6b6b6b;--line:#e2ded8;--card:#fff;
--accent:#1f6f4a;--accent2:#9a6b1f;--grid:#eeeae4;}
:root:not([data-theme=light]){}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#131312;--ink:#eceae6;--muted:#9c9890;--line:#2c2b28;--card:#1a1a18;
--accent:#5fbf8f;--accent2:#d6a34e;--grid:#242320;}}
:root[data-theme=dark]{--bg:#131312;--ink:#eceae6;--muted:#9c9890;--line:#2c2b28;
--card:#1a1a18;--accent:#5fbf8f;--accent2:#d6a34e;--grid:#242320;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:48px 22px 90px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 6px}
.sub-title{color:var(--muted);margin:0 0 8px}
.warn{border-left:3px solid var(--accent2);padding:10px 14px;margin:22px 0;
background:var(--card);color:var(--muted);font-size:13.5px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:22px 22px 8px;margin:22px 0}
.panel h2{font-size:19px;margin:0 0 4px;letter-spacing:-.01em}
.lede{color:var(--muted);margin:0 0 14px;font-size:14px}
.informs{font-size:12px;color:var(--muted);margin:0 0 16px}
.informs code{background:var(--grid);padding:1px 5px;border-radius:4px;
font-size:11.5px;margin-right:4px;display:inline-block}
.kv{display:flex;justify-content:space-between;gap:16px;padding:5px 0;
border-bottom:1px dotted var(--line);font-size:14px}
.kv span{color:var(--muted)} .kv b{font-variant-numeric:tabular-nums}
.sub{margin:16px 0} .sub h4{margin:0 0 6px;font-size:13px;color:var(--muted);
text-transform:uppercase;letter-spacing:.05em;font-weight:600}
.scroller{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:5px 10px 5px 0;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
thead th{color:var(--muted);font-weight:600}
td:not(:first-child),th:not(:first-child){text-align:right}
.note{color:var(--muted);font-size:13.5px;border-left:2px solid var(--line);
padding-left:12px;margin:14px 0}
.ch{margin:6px 0 14px} .chart{width:100%;height:auto;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.ref{stroke:var(--line);stroke-width:1;stroke-dasharray:4 4}
.tick{fill:var(--muted);font-size:10px}
.axis{fill:var(--muted);font-size:10px;text-transform:uppercase;
letter-spacing:.06em}
.bar{fill:var(--accent);opacity:.85} .bar:hover{opacity:1}
polyline{fill:none;stroke-width:1.8}
.s0{stroke:var(--accent)} .s1{stroke:var(--accent2)}
.legend{font-size:11px} .s0t{fill:var(--accent)} .s1t{fill:var(--accent2)}
footer{color:var(--muted);font-size:12.5px;margin-top:36px}
"""


def render(report, source="data/prepared.parquet"):
    pop = report["population"]
    panels = []
    for key, body in report["panels"].items():
        informs = "".join(f"<code>{_esc(i)}</code>" for i in body["informs"])
        panels.append(
            f'<section class="panel" id="{_esc(key)}">'
            f'<h2>{_esc(body["title"])}</h2>'
            f'<p class="lede">{_esc(body["lede"])}</p>'
            + (f'<p class="informs">informs {informs}</p>' if informs else "")
            + _chart(body.get("chart")) + _value_block(body) + "</section>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Population EDA</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Population EDA</h1>
<p class="sub-title">{pop['episodes']:,} episodes · {pop['rows']:,} rows ·
from <code>{_esc(source)}</code></p>
<div class="warn"><b>This page decides nothing.</b> It produces no config
value and no gate. <code>bootstrap.measure</code> owns the MEASURED values and
the reassessment gates; <code>pipeline.status</code> owns what blocks a
launch. Every panel names the keys it should change your mind about, and a
test asserts those keys exist — but the changing of minds is yours.</div>
{''.join(panels)}
<footer>Built by <code>python3 -m tools.eda</code> from
<code>reports/eda.json</code>. Re-run it after every re-extract: it reads the
prepared parquet and config only, so it costs seconds and is the first thing
worth looking at on a new population.</footer>
</div></body></html>
"""
