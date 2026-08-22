"""Tab content for the system walkthrough. Imported by tools.walkthrough.build.

The prose lives in `panels/*.html`, one file per tab. It is a document, not
code -- 1,700 lines of it -- and holding it in Python string literals meant
every literal brace in the markup had to be doubled for the f-string, which
silently broke the page twice. The files are now read verbatim.

Every figure in them is either (a) quoted from docs/design.md, i.e. the
baseline-20260811043259 production run, or (b) computed from this repo's own
code with the inputs stated inline, so a reader can re-run it. Where a figure
is a constructed illustration rather than a measurement, it says so.

Some fragments are constructed rather than written, because they carry values
that must not be typed twice. The files call for them with a custom tag:

    <x-filecard path=… holds=… state=… reader=… [moves="1"]></x-filecard>
    <x-pmfbars></x-pmfbars>
    <x-eda-chips></x-eda-chips>
    <x-eda-chart key="pareto"></x-eda-chart>

The two `x-eda-*` tags read `reports/eda.json` at BUILD time and render from
it, reusing `tools.eda_page`'s chart renderers so the walkthrough and
docs/eda.html cannot draw the same series two different ways. `reports/` is
gitignored, so on a fresh clone there is no report to read: the tags then
render a visible "not built yet" note naming the command, rather than an empty
space that reads like a finding of zero.

Anything else in a panel file is markup and reaches the page unchanged.
"""

import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
PANEL_DIR = HERE / "panels"

# tab key -> panels/<key>.html. "dp" is absent on purpose: that tab is lifted
# from _source.html by the builder, not authored here.
KEYS = ("map", "data", "population", "model", "calib", "var", "prior",
        "learn", "replay", "shadow", "assure")

EDA_REPORT = HERE.parent.parent / "reports" / "eda.json"

# (label, panel key, field path, formatter) -- the handful of figures from the
# EDA that change a decision, as opposed to the fifteen panels that describe
# the population. The full set is docs/eda.html; this is the summary strip.
CHIPS = [
    ("Episodes", "volumes", ("episodes_total",), "{:,.0f}"),
    ("SKUs", "volumes", ("unique_skus_overall",), "{:,.0f}"),
    ("Episodes / day", "volumes", ("episodes_per_day", "p50"), "{:,.0f}"),
    ("SKUs holding 80% of exposure", "pareto",
     ("skus_covering_80pct_of_cogs",), "{:,.0f}"),
    ("Cost ratio, median", "cost_geometry", ("cost_ratio", "p50"), "{:.0%}"),
    ("Non-explorable", "cost_geometry", ("share_non_explorable",), "{:.1%}"),
    ("Mean clearance", "clearance", ("mean_clearance",), "{:.0%}"),
    ("Subcats clearing anchor min", "anchors",
     ("share_of_subcategories_clearing_min",), "{:.0%}"),
]


def filecard(path, holds, state, reader, moves=False):
    tag = "moves-tag" if moves else ""
    return f"""  <div class="filecard">
    <b>{path}</b>
    <span><i>holds</i>{holds}</span>
    <span><i>read by</i>{reader}</span>
    <span class="frozen-tag {tag}">{state}</span>
  </div>"""


def _bar_row(label, vals, hi):
    """One sales-count row across the three dispersions."""
    cells = "".join(
        f'<div class="track"><div class="bar {cls}" style="width:{v / hi * 100:.1f}%"></div>'
        f'<b>{v:.1f}%</b></div>'
        for v, cls in zip(vals, ("lumpy", "nb", "po")))
    return (f'<div class="row"><span>{label}</span>'
            f'<div class="bar-pair">{cells}</div></div>')


_PMF = [("sells 0", [71.6, 64.6, 57.6]), ("sells 1", [15.4, 22.5, 31.4]),
        ("sells 2", [6.4, 8.2, 9.0]), ("sells 3", [3.1, 3.0, 1.8]),
        ("sells 4+", [3.5, 1.7, 0.2])]


def _pmf_bars():
    return "".join(_bar_row(label, vals, 75) for label, vals in _PMF)


def _eda():
    """The report, or None. Never raises: a walkthrough that cannot be built
    without a run of the pipeline is a walkthrough nobody can build."""
    if not EDA_REPORT.exists():
        return None
    try:
        return json.loads(EDA_REPORT.read_text())
    except (ValueError, OSError):
        return None


def _eda_missing(what):
    return (f'<div class="callout"><h3>{what} not built yet</h3><p>Run '
            f'<code>python3 -m tools.eda --input data/prepared.parquet</code> '
            f'and rebuild this page. <code>reports/</code> is gitignored, so a '
            f'fresh clone has no report to read.</p></div>')


def _eda_chips():
    report = _eda()
    if not report:
        return _eda_missing("Population figures")
    out = []
    for label, key, path, fmt in CHIPS:
        node = report["panels"].get(key, {})
        for part in path:
            node = node.get(part, {}) if isinstance(node, dict) else None
        if node is None or isinstance(node, dict):
            continue
        out.append(f'<li><span class="k">{label}</span>'
                   f'<span class="v">{fmt.format(node)}</span></li>')
    return f'<ul class="chips" style="margin-top:26px">{"".join(out)}</ul>'


def _eda_chart(key):
    report = _eda()
    if not report:
        return _eda_missing(f"Chart <code>{key}</code>")
    body = report["panels"].get(key)
    if not body or not body.get("chart"):
        return _eda_missing(f"Chart <code>{key}</code>")
    from tools.eda_page import KINDS
    chart = body["chart"]
    if chart.get("kind") not in KINDS:
        return ""
    svg = KINDS[chart["kind"]](chart)
    return (f'<figure><div class="scroller">{svg}</div>'
            f'<figcaption>{body["lede"]}</figcaption></figure>')


_ATTR = re.compile(r'(\w+)="([^"]*)"')
_FILECARD = re.compile(r"<x-filecard\b([^>]*)></x-filecard>")
_EDA_CHART = re.compile(r"<x-eda-chart\b([^>]*)></x-eda-chart>")


def _expand(html):
    def card(m):
        a = dict(_ATTR.findall(m.group(1)))
        return filecard(a["path"], a["holds"], a["state"], a["reader"],
                        moves=bool(a.get("moves")))
    html = _FILECARD.sub(card, html)
    html = _EDA_CHART.sub(lambda m: _eda_chart(dict(_ATTR.findall(m.group(1)))["key"]),
                          html)
    html = html.replace("<x-eda-chips></x-eda-chips>", _eda_chips())
    return html.replace("<x-pmfbars></x-pmfbars>", _pmf_bars())


def load():
    """Every tab's HTML, expanded. Raises if a panel file is missing -- a tab
    silently rendering empty is the one failure mode worth being loud about."""
    return {k: _expand((PANEL_DIR / f"{k}.html").read_text()) for k in KEYS}


PANELS = load()
