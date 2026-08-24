"""tools.refresh_figures -- keep the numbers in the docs equal to the artifacts.

The documents are prose with measurements typed into them. That is the right
way to write a document and the wrong way to keep one true: after a re-run
every figure is stale and nothing says so. It has happened repeatedly in this
repo -- `rho 0.3103`, `deff 3.347` and the IL figures all outlived the runs
that produced them, and were quoted for weeks afterwards as if current.

The fix is not discipline, it is an anchor. A figure in a document is written

    <!--f:rho.rho|dec4-->0.3103<!--/f-->

which renders as `0.3103` and nothing else -- HTML comments do not display, in
GitHub or in any markdown viewer. `rho` names an artifact, `.rho` is the path
inside it, `dec4` is the formatter. Then:

    python3 -m tools.refresh_figures            # check, non-zero on drift
    python3 -m tools.refresh_figures --write    # rewrite from the artifacts

THE DOCUMENT IS THE LEDGER. `tools/walkthrough/figures.py` keeps a separate
registry pairing each printed literal with its JSON path, which buys a check
that works with no report on disk -- but it also means two places can disagree
about what the page says. Here the number and its source sit in the same
place, so that disagreement cannot exist, and the only question left is
whether the document matches the artifact.

WHAT MUST NOT BE ANCHORED. A figure describing what a PAST decision cost --
"deleting restocked episodes took 18.1pp of the extract's COGS" -- is a
historical measurement, not a current one. Refreshing it against today's
artifacts would silently replace a fact about a decision with an unrelated
number and destroy the argument it supports. Those stay as plain prose and
carry their own date. Anchor only figures that describe the CURRENT state of
the pipeline and are re-measured on every run.

Run it from `scripts/run_bootstrap.sh`, which does: the agent holding the real
data is the one that has the numbers, so refreshing the docs is part of the
run rather than a thing someone remembers to do afterwards.
"""

import argparse
import json
import os
import re

# artifact key -> path, relative to the repo root. Adding a source here is the
# only step needed to make a new artifact's fields anchorable.
SOURCES = {
    "rho": "artifacts/rho.json",
    "r_lookup": "artifacts/r_lookup.json",
    "prior": "artifacts/prior.json",
    "manifest": "artifacts/split_manifest.json",
    "backtest": "reports/backtest.json",
    "phase0": "reports/phase0.json",
    "shadow": "reports/shadow.json",
    "thresholds": "reports/thresholds.json",
    "eda": "reports/eda.json",
}

DOCS = ("AGENTS.md", "README.md", "docs/design.md",
        "docs/perishable_markdown_mvp_prd.md")

FORMATTERS = {
    "raw": lambda x: str(x),
    "dec2": lambda x: f"{float(x):.2f}",
    "dec3": lambda x: f"{float(x):.3f}",
    "dec4": lambda x: f"{float(x):.4f}",
    "pct": lambda x: f"{float(x) * 100:.2f}%",
    "pct1": lambda x: f"{float(x) * 100:.1f}%",
    # already a percentage in the artifact, printed as percentage points
    "pp": lambda x: f"{float(x):.1f}pp",
    "count": lambda x: f"{int(x):,}",
    "won_m": lambda x: f"₩{float(x) / 1e6:.2f}M",
    "won_b": lambda x: f"₩{float(x) / 1e9:.2f}bn",
    "signed3": lambda x: f"{float(x):+.3f}",
}

ANCHOR = re.compile(
    r"<!--f:(?P<src>[a-z_]+)\.(?P<path>[A-Za-z0-9_.\[\]| -]+?)"
    r"(?:\|(?P<fmt>[a-z0-9_]+))?-->(?P<text>.*?)<!--/f-->",
    re.DOTALL)

STAMP = re.compile(r"<!--figures-from:(?P<run>[^>]*)-->")


def dig(obj, path):
    """Walk a dotted path, with `[k]` for a dict key that contains dots or
    spaces -- category names do, and they are exactly what a per-category
    figure needs to reach."""
    for part in re.findall(r"\[([^\]]+)\]|([^.\[\]]+)", path):
        key = part[0] or part[1]
        if isinstance(obj, list):
            try:
                obj = obj[int(key)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def load_sources(root="."):
    out = {}
    for key, rel in SOURCES.items():
        p = os.path.join(root, rel)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    out[key] = json.load(f)
            except (ValueError, OSError):
                pass
    return out


# A dataset whose name says it is not real. Writing production documents from
# a fixture run is the exact failure this tool could most easily cause -- the
# numbers are plausible, the mechanism is silent, and the result reads as a
# measurement. It happened once during development, which is why the guard
# exists rather than a note asking people to be careful.
FIXTURE_HINTS = ("dummy", "synth", "fixture", "sample", "test", "tmp", "mock")


def looks_like_a_fixture(label):
    low = str(label).lower()
    return any(h in low for h in FIXTURE_HINTS)


def run_stamp(sources):
    """Which run the artifacts on disk are from. Written into each refreshed
    document so a reader can tell at a glance whether its figures belong to the
    run they are looking at, rather than assuming."""
    for key in ("backtest", "prior", "rho"):
        d = sources.get(key) or {}
        v = ((d.get("artifact_versions") or {}).get("baseline_model_version")
             or (d.get("provenance") or {}).get("baseline_model_version"))
        if v:
            return str(v)
    return "unknown"


def resolve(match, sources):
    """(formatted value, problem). Absent sources and bad paths are REPORTED,
    never guessed and never crashed on -- a doc refresh that dies because one
    report is missing is a refresh nobody runs."""
    src, path = match.group("src"), match.group("path")
    fmt = match.group("fmt") or "raw"
    if src not in SOURCES:
        return None, f"unknown source {src!r}"
    if src not in sources:
        return None, f"{SOURCES[src]} not on disk"
    if fmt not in FORMATTERS:
        return None, f"unknown formatter {fmt!r}"
    value = dig(sources[src], path)
    if value is None:
        return None, f"{src}.{path} absent from {SOURCES[src]}"
    try:
        return FORMATTERS[fmt](value), None
    except (TypeError, ValueError):
        return None, f"{src}.{path} is {value!r}, which {fmt} cannot format"


def scan(doc, sources, root="."):
    """(anchors, drift, problems) for one document."""
    p = os.path.join(root, doc)
    if not os.path.exists(p):
        return 0, [], []
    with open(p) as f:
        text = f.read()
    anchors, drift, problems = 0, [], []
    for m in ANCHOR.finditer(text):
        anchors += 1
        want, problem = resolve(m, sources)
        if problem:
            problems.append(f"{doc}: {problem}")
        elif want != m.group("text"):
            drift.append((doc, f"{m.group('src')}.{m.group('path')}",
                          m.group("text"), want))
    return anchors, drift, problems


def rewrite(doc, sources, stamp, root="."):
    """Replace every anchored figure with the artifact's value. Returns the
    number changed."""
    p = os.path.join(root, doc)
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        text = f.read()
    changed = [0]

    def sub(m):
        want, problem = resolve(m, sources)
        if problem or want == m.group("text"):
            return m.group(0)
        changed[0] += 1
        head = m.group(0)[:m.start("text") - m.start(0)]
        return head + want + "<!--/f-->"

    new = ANCHOR.sub(sub, text)
    if changed[0]:
        # the stamp only moves when a figure did, so an untouched document does
        # not claim to be from a run it was not refreshed against
        new = STAMP.sub(f"<!--figures-from:{stamp}-->", new)
        with open(p, "w") as f:
            f.write(new)
    return changed[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="rewrite the documents from the artifacts on disk")
    ap.add_argument("--dataset", default=None,
                    help="what the artifacts were built from; REQUIRED with "
                         "--write, and written into each refreshed document")
    ap.add_argument("--allow-fixture", action="store_true",
                    help="permit --write from a dataset whose name says it is "
                         "synthetic. Only for exercising this tool itself")
    ap.add_argument("--root", default=".")
    ap.add_argument("--docs", nargs="*", default=None)
    args = ap.parse_args()

    if args.write:
        if not args.dataset:
            print("refusing to write without --dataset: a document refreshed "
                  "from artifacts nobody can identify is worse than a stale "
                  "one, because it reads as current")
            return 2
        if looks_like_a_fixture(args.dataset) and not args.allow_fixture:
            print(f"refusing to write production documents from "
                  f"{args.dataset!r}, whose name says it is synthetic. Fixture "
                  f"numbers are plausible and silent -- they read as "
                  f"measurements. Pass --allow-fixture only to exercise this "
                  f"tool, never to refresh a document anyone will read.")
            return 2

    docs = args.docs or DOCS
    sources = load_sources(args.root)
    stamp = run_stamp(sources)

    if not sources:
        print("no artifacts on disk -- nothing to check against")
        return 0

    total_anchors, all_drift, all_problems = 0, [], []
    for doc in docs:
        a, d, pr = scan(doc, sources, args.root)
        total_anchors += a
        all_drift += d
        all_problems += pr

    if args.write:
        full = f"{stamp} on {args.dataset}"
        n = sum(rewrite(doc, sources, full, args.root) for doc in docs)
        print(f"refreshed {n} of {total_anchors} anchored figures "
              f"across {len(docs)} documents, from {full}")
        for p in all_problems:
            print(f"  ! {p}")
        # a problem is not drift: it means the anchor could not be resolved at
        # all, which is a broken anchor rather than a stale number
        return 1 if all_problems else 0

    print(f"{total_anchors} anchored figures, run {stamp}")
    for doc, path, says, should in all_drift:
        print(f"  DRIFT {doc}: {path} -- says {says}, artifact says {should}")
    for p in all_problems:
        print(f"  ! {p}")
    if all_drift:
        print(f"\n{len(all_drift)} stale. Run with --write to refresh.")
    return 1 if (all_drift or all_problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
