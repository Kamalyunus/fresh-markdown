"""tools.walkthrough.build -- assemble docs/system_walkthrough.html.

The leadership-facing walkthrough: one tab per frozen artifact, one for the
hourly decision, one for the learning loop. Run from the repo root:

    python3 -m tools.walkthrough.build

`_source.html` is the original single-topic DP page, kept pristine so this is
re-runnable; its sections are lifted verbatim because they are already
number-checked against the solver. The artifact tabs live in `panels.py`.

Figures on the artifact tabs come from the v3 deck and docs/design.md -- the
baseline-20260811043259 production run -- so the whole page quotes one vintage.
The decision tab is a self-contained solve whose inputs are stated on it.

To publish it as an artifact, deploy the built file and pass the existing
artifact URL so the same link is updated rather than a new one created.
"""
import pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "docs" / "system_walkthrough.html"

src = (HERE / "_source.html").read_text().split("\n")

style = "\n".join(src[2:247])                 # inside <style> ... </style>, tags excluded
dp_header = ("\n".join(src[251:275])          # <header> ... </header>
             .replace("<h1>", '<h2 class="tab-title">').replace("</h1>", "</h2>"))
dp_body = "\n".join(src[276:842])             # the DP sections
_OLD_MU = dp_body[dp_body.index("    <p>\n      The machine-learned model supplies"):
                  dp_body.index("      But you never sell 0.56 of anything.")]
dp_body = dp_body.replace(_OLD_MU, """    <p>
      For any candidate price the two frozen inputs give an <em class="term">average</em>.
      The demand model supplies this hour's base level, and the price-response exponent
      moves it to the price being considered — both have their own tabs. At full price that
      average is 0.56 units this hour; at 30% off, 0.80.
    </p>
  </div>
  <div class="prose">
    <p>
""")
fan_script = "\n".join(src[857:875])          # the fan hover wiring

TAB_CSS = """
  /* ------------------------------------------------------------------ tabs */
  .tabwrap {
    position: sticky; top: 0; z-index: 20; background: var(--paper);
    border-bottom: 1px solid var(--rule); margin: 34px 0 0;
  }
  .tabs {
    display: flex; gap: 2px; overflow-x: auto; scrollbar-width: none;
    max-width: 940px; margin: 0 auto;
  }
  .tabs::-webkit-scrollbar { display: none; }
  .tabs button {
    appearance: none; background: none; border: 0; cursor: pointer;
    font-family: var(--body); font-size: 13.5px; font-weight: 550;
    color: var(--muted); white-space: nowrap; padding: 13px 11px 11px;
    border-bottom: 2.5px solid transparent; margin-bottom: -1px;
    transition: color .12s ease, border-color .12s ease;
  }
  .tabs button:hover { color: var(--ink); }
  .tabs button[aria-selected="true"] { color: var(--green); border-bottom-color: var(--green); }
  .tabs button:focus-visible { outline: 2px solid var(--green); outline-offset: -3px; }
  /* the base table style right-aligns for figures; these carry prose instead */
  .lefty tbody td:nth-child(2), .lefty thead th:nth-child(2) { text-align: left; }
  .tl tbody td, .tl thead th { text-align: left; }
  .tl tbody td { font-family: var(--body); }
  .panel[hidden] { display: none; }
  @media (prefers-reduced-motion: reduce) { .tabs button { transition: none; } }

  /* the file an artifact lives in, stated once at the top of its tab */
  .filecard {
    display: flex; flex-wrap: wrap; gap: 6px 26px; align-items: baseline;
    background: var(--sunk); border: 1px solid var(--rule); border-radius: 6px;
    padding: 12px 16px; margin: 22px 0 0; font-size: 13.5px;
  }
  .filecard b {
    font-family: var(--data); font-weight: 600; font-size: 13px; color: var(--ink);
  }
  .filecard span { color: var(--muted); }
  .filecard span i {
    font-style: normal; text-transform: uppercase; letter-spacing: .08em;
    font-size: 10.5px; margin-right: 7px;
  }
  /* these must out-specify `.filecard span`, which sets the muted label colour */
  .filecard .frozen-tag {
    font-family: var(--body); font-size: 10.5px; letter-spacing: .09em;
    text-transform: uppercase; font-weight: 650; color: var(--green);
    background: var(--green-w); border-radius: 3px; padding: 3px 7px;
  }
  .filecard .moves-tag {
    color: var(--loss); background: transparent;
    box-shadow: inset 0 0 0 1px var(--loss);
  }

  /* a panel's own opening title -- h1 belongs to the page, not to a tab */
  .panel header { padding-top: 30px; }
  .tab-title {
    font-family: var(--display); font-weight: 400;
    font-size: clamp(31px, 5.2vw, 46px); line-height: 1.06;
    letter-spacing: -.015em; text-wrap: balance; margin: 0 0 20px;
  }

  /* uppercasing maths is destructive: ρ→P, τ→T, μ→M, f→F. Opt those out. */
  thead th.raw { text-transform: none; font-size: 12.5px; letter-spacing: .02em; }
  .sym { text-transform: none; }

  /* a third distribution series alongside the existing .nb / .po pair */
  .bar.lumpy { background: var(--loss); }
  .key i.lumpy { background: var(--loss); }

  /* the architecture diagram on the first tab */
  .arch { width: 100%; min-width: 720px; height: auto; display: block; color: var(--ink); }
  .arch .box { fill: var(--surface); stroke: var(--rule); stroke-width: 1; }
  .arch .box-key { fill: var(--green-w); stroke: var(--green); stroke-width: 1.5; }
  .arch .frozen .box { fill: var(--sunk); }
  .arch .t {
    font-family: var(--body); font-size: 13px; font-weight: 600;
    fill: currentColor; text-anchor: middle;
  }
  .arch .s {
    font-family: var(--body); font-size: 11px; fill: var(--muted); text-anchor: middle;
  }
  .arch .band {
    font-family: var(--body); font-size: 10.5px; letter-spacing: .1em;
    text-transform: uppercase; font-weight: 650; fill: var(--muted);
  }
  .arch .wire { fill: none; stroke: var(--muted); stroke-width: 1.4; }
  .arch .head { fill: var(--muted); }
  .arch .loop { stroke: var(--green); stroke-width: 1.8; }
  .arch .head-key { fill: var(--green); }
  .arch .note {
    font-family: var(--body); font-size: 11px; font-style: italic;
    fill: var(--green); text-anchor: middle;
  }

  /* the level-vs-slope diagnostic pair */
  .diag { width: 100%; height: auto; display: block; color: var(--ink); }
  .diag .ax { stroke: var(--rule); stroke-width: 1; }
  .diag .ref { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; opacity: .75; }
  .diag .flat { stroke: var(--green); stroke-width: 2.5; }
  .diag .tilt { fill: none; stroke: var(--loss); stroke-width: 2.5; }
  .diag .pt { fill: var(--ink); }
  .diag .cap {
    font-family: var(--body); font-size: 10.5px; letter-spacing: .09em;
    text-transform: uppercase; font-weight: 650; fill: var(--muted);
  }
  .diag .tick { font-family: var(--data); font-size: 10.5px; fill: var(--muted); }
  .diag .note { font-family: var(--body); font-size: 11px; fill: var(--muted); }
"""

PAGE_HEADER = """<header>
  <p class="kicker">Perishable Markdown MVP · how the system works</p>
  <h1>Every hour, one price — and everything standing behind it</h1>
  <p class="standfirst">
    Fresh stock enters a clearance window with a clock on it, and whatever does not sell is
    thrown away at cost. This is the whole of what we built to price it: the
    <strong>architecture</strong> first, then <strong>each frozen input</strong> in turn,
    the <strong>exact calculation</strong> that turns them into a price, the
    <strong>one number learned in production</strong>, and what five months of
    <strong>replayed history</strong> says it would have done — and how we will know it is
    still working once real prices start moving. Read the tabs in order, or jump to the one
    you are being asked about.
  </p>
</header>"""

TABS = [
    # key stays "map" so any shared #map link keeps working
    ("map",   "Architecture",     "the whole system"),
    ("data",  "Data",             "split_manifest.json"),
    ("model", "Demand",           "baseline_model.txt"),
    ("calib", "Calibration",      "calibration.json"),
    ("var",   "Variance",         "r_lookup · rho"),
    ("prior", "Elasticity",       "prior.json"),
    ("dp",    "Decision",         "pricing/dp.py"),
    ("learn", "Learning",         "posterior.json"),
    ("replay","Replay",           "what we measured"),
    ("assure","Assurance",        "how we know it keeps working"),
]


from tools.walkthrough.panels import PANELS

FOOTER = """<footer class="prose">
  <p>
    Figures on the artifact tabs are measured on production data against model
    <code>baseline-20260811043259</code>. The decision tab is a single episode solved by
    <code>pricing/dp.py</code> with the inputs named on it — ₩10,000 original price,
    ₩4,000 cost, 3 units, a 4-hour window, a SKU forecast at 0.8 units an hour at the
    category reference discount of 30%, dispersion 0.919 and price response −1.0 — and
    every number on that tab is the solver's own output for that state, reproducible and
    moving when the inputs do.
  </p>
</footer>"""

# ------------------------------------------------------------------- assemble
bar = "\n".join(
    f'    <button role="tab" id="t-{k}" aria-controls="p-{k}" '
    f'aria-selected="{"true" if i == 0 else "false"}" tabindex="{"0" if i == 0 else "-1"}">'
    f'{label}</button>'
    for i, (k, label, _sub) in enumerate(TABS))

panels = []
for i, (k, label, _sub) in enumerate(TABS):
    body = dp_header + "\n" + dp_body if k == "dp" else PANELS[k]
    panels.append(
        f'<div class="panel" id="p-{k}" role="tabpanel" aria-labelledby="t-{k}"'
        f'{"" if i == 0 else " hidden"}>\n{body}\n</div>')

TAB_JS = """
  (function () {
    var bar = document.getElementById("tabbar");
    var tabs = [].slice.call(bar.querySelectorAll('[role="tab"]'));

    function show(id, push) {
      tabs.forEach(function (t) {
        var on = t.id === "t-" + id;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        document.getElementById("p-" + t.id.slice(2)).hidden = !on;
      });
      if (push && history.replaceState) history.replaceState(null, "", "#" + id);
    }

    tabs.forEach(function (t) {
      t.addEventListener("click", function () {
        show(t.id.slice(2), true);
        window.scrollTo({ top: 0, behavior: "instant" });
      });
    });

    bar.addEventListener("keydown", function (e) {
      var i = tabs.indexOf(document.activeElement);
      if (i < 0) return;
      var j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1
            : e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : -1;
      if (j < 0) return;
      e.preventDefault();
      j = (j + tabs.length) % tabs.length;
      tabs[j].focus(); show(tabs[j].id.slice(2), true);
    });

    var h = (location.hash || "").slice(1);
    if (h && document.getElementById("p-" + h)) show(h, false);
  })();
"""

def _protect_symbols(html):
    """h3 and caption are rendered uppercase, which mangles Greek letters into
    Latin lookalikes (ρ reads as P, τ as T, μ as M). Wrap them so the transform
    skips them, rather than relying on every author remembering."""
    GREEK = "ρτμεσλγβα"

    def fix(m):
        head, body, tail = m.group(1), m.group(2), m.group(3)
        for ch in GREEK:
            body = body.replace(ch, f'<span class="sym">{ch}</span>')
        return head + body + tail

    for pat in (r"(<caption>)([^<]*)(</caption>)",
                r"(<h3[^>]*>)([^<]*)(</h3>)",
                r"(<th[^>]*>)([^<]*)(</th>)"):
        html = re.sub(pat, fix, html)
    return html


out = f"""<title>Inside the Markdown Agent</title>
<style>
{style}
{TAB_CSS}
</style>

<div class="wrap">
{PAGE_HEADER}
</div>

<div class="tabwrap">
  <div class="tabs" id="tabbar" role="tablist" aria-label="System walkthrough">
{bar}
  </div>
</div>

<div class="wrap">

{chr(10).join(panels)}

{FOOTER}

</div>

<script>
{TAB_JS}
{fan_script}
</script>
"""

OUT.write_text(_protect_symbols(out))
print(f"wrote {OUT.relative_to(ROOT)}: "
      f"{len(out.splitlines())} lines, {len(TABS)} tabs")
