"""Tab content for the system walkthrough. Imported by tools.walkthrough.build.

The prose lives in `panels/*.html`, one file per tab. It is a document, not
code -- 1,700 lines of it -- and holding it in Python string literals meant
every literal brace in the markup had to be doubled for the f-string, which
silently broke the page twice. The files are now read verbatim.

Every figure in them is either (a) quoted from docs/design.md, i.e. the
baseline-20260811043259 production run, or (b) computed from this repo's own
code with the inputs stated inline, so a reader can re-run it. Where a figure
is a constructed illustration rather than a measurement, it says so.

Two fragments are constructed rather than written, because they carry values
that must not be typed twice. The files call for them with a custom tag:

    <x-filecard path=… holds=… state=… reader=… [moves="1"]></x-filecard>
    <x-pmfbars></x-pmfbars>

Anything else in a panel file is markup and reaches the page unchanged.
"""

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
PANEL_DIR = HERE / "panels"

# tab key -> panels/<key>.html. "dp" is absent on purpose: that tab is lifted
# from _source.html by the builder, not authored here.
KEYS = ("map", "data", "model", "calib", "var", "prior", "learn", "replay",
        "shadow", "assure")


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


_ATTR = re.compile(r'(\w+)="([^"]*)"')
_FILECARD = re.compile(r"<x-filecard\b([^>]*)></x-filecard>")


def _expand(html):
    def card(m):
        a = dict(_ATTR.findall(m.group(1)))
        return filecard(a["path"], a["holds"], a["state"], a["reader"],
                        moves=bool(a.get("moves")))
    html = _FILECARD.sub(card, html)
    return html.replace("<x-pmfbars></x-pmfbars>", _pmf_bars())


def load():
    """Every tab's HTML, expanded. Raises if a panel file is missing -- a tab
    silently rendering empty is the one failure mode worth being loud about."""
    return {k: _expand((PANEL_DIR / f"{k}.html").read_text()) for k in KEYS}


PANELS = load()
