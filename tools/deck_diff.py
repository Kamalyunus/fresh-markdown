"""tools.deck_diff -- regression check between two versions of the deck.

Restructuring a deck is exactly when vetted content gets silently dropped or
mangled. This pairs slides by their first two text runs (eyebrow + title),
then reports what is NEW, what was DROPPED, and any reused slide whose text
changed for a reason not on the INTENDED list. Exits non-zero on the last two,
so it can gate a rebuild.

Usage:
    python3 -m tools.deck_diff [--old A.pptx] [--new B.pptx]

The defaults compare the original deck against the one we present, which is
the comparison that matters now that the intermediate v2 is not checked in.
"""
import difflib, re, sys, zipfile

RUN = re.compile(r"<a:t>([^<]*)</a:t>")


def slides(path):
    z = zipfile.ZipFile(path)
    pres = z.read("ppt/presentation.xml").decode()
    rels = z.read("ppt/_rels/presentation.xml.rels").decode()
    tgt = {m.group(1): m.group(2) for m in
           re.finditer(r'Id="(rId\d+)"[^>]*Target="(slides/slide\d+\.xml)"', rels)}
    lst = re.search(r"<p:sldIdLst>(.*?)</p:sldIdLst>", pres, re.S).group(1)
    out = []
    for m in re.finditer(r'r:id="(rId\d+)"', lst):
        xml = z.read("ppt/" + tgt[m.group(1)]).decode()
        runs = [t for t in RUN.findall(xml) if t.strip()]
        # drop the page-number run: it legitimately changes with position
        out.append([t for t in runs if not t.strip().isdigit()])
    return out


import argparse
ap = argparse.ArgumentParser(prog="tools.deck_diff", description=__doc__)
ap.add_argument("--old", default="docs/perishable_markdown_tech_deck.pptx")
ap.add_argument("--new", default="docs/perishable_markdown_deck_v3.pptx")
args = ap.parse_args()
v1, v2 = slides(args.old), slides(args.new)

INTENDED = {  # substrings whose change is a deliberate number refresh
    "32.3%", "36.7%", "356,114", "2,000-episode replay", "₩14.7M", "₩17.1M",
    "scrap is ~7%", "the bar the system must not break", "0.10 s", "0.09 s",
    "₩1,271", "₩1,476", "₩1,475", "38.1%", "43.4%", "1,837", "1,381",
    "0.0065", "0.0087", "2026-08-09", "baseline-20260811043259",
    "32.27%", "36.68%", "₩14.27M", "₩17.11M", "409.87", "447.78",
    "18 weeks", "71,559", "0.000875", "0.002915", "6× the original",
    "3× the pre-scrap-fix", "measured before any price was applied",
    # v2 -> v3: the four core slides tightened for a presented setting. Each
    # entry is a phrase unique to the v2 wording, so a change these do not
    # cover still fails the check.
    "(LightGBM / Tweedie)", "splitting cells divides the same evidence",
    "The two metrics can disagree by design", "Original price 10,000",
    "Stated now, before the experiment", "Deliberately blind to price",
    "constructs the actions", "budget; nobody can sign",
    "explained rather than observed", "recommendations, with evidence",
    "Measured on nine real 2-week blocks", "Once the floor was measured",
    "on the trailing basis, not outlier-dominated",
    # v3: the decision-core recursion, corrected to what the solver evaluates
    "m(p)·E[min(D,q)]", "the censored expectation E[min(D, q)], never the raw mean",
    "Σₖ P(D=k)", "V(q−min(k,q), t−1)", "Demand enters as a distribution",
}

v1_by_key = {tuple(s[:2]): (i, s) for i, s in enumerate(v1, 1)}
missing, changed, new = [], [], []
seen = set()

for j, s in enumerate(v2, 1):
    key = tuple(s[:2])
    if key not in v1_by_key:
        new.append((j, s[1] if len(s) > 1 else s[0]))
        continue
    i, old = v1_by_key[key]
    seen.add(key)
    if old != s:
        # Align the two run lists before judging them. A straight zip would
        # treat one inserted line as "everything after it changed"; aligning
        # first separates a rewritten run (a pair -- either side may carry the
        # INTENDED phrase) from a run that was genuinely added or deleted
        # (which must carry one itself, since that is the case worth catching).
        pairs, gone, came = [], [], []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, old, s, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            n = min(i2 - i1, j2 - j1)
            pairs += list(zip(old[i1:i1 + n], s[j1:j1 + n]))
            gone += old[i1 + n:i2]
            came += s[j1 + n:j2]
        # A rewritten run is authorised by a phrase from the text being
        # REPLACED, not from its replacement. Accepting a match on either side
        # lets a deleted line hide behind the allowed rewrite next to it --
        # the tool then passes exactly the case it exists to catch.
        ok = (all(any(x in a for x in INTENDED) for a, _ in pairs)
              and all(any(x in t for x in INTENDED) for t in gone + came))
        if not ok:
            changed.append((i, j, pairs, gone, came))

for key, (i, s) in v1_by_key.items():
    if key not in seen:
        missing.append((i, s[1] if len(s) > 1 else s[0]))

print(f"v1 slides: {len(v1)}   v2 slides: {len(v2)}")
print(f"\nNEW in v2 ({len(new)}):")
for j, t in new:
    print(f"  pos {j:>2}  {t[:70]}")
print(f"\nDROPPED from v1 ({len(missing)}):")
for i, t in missing:
    print(f"  was {i:>2}  {t[:70]}")
print(f"\nUNINTENDED CHANGES ({len(changed)}):")
for i, j, pairs, gone, came in changed:
    print(f"  v1 {i} -> v2 {j}")
    for a, b in pairs[:4]:
        print(f"      ~ {a[:86]}\n        {b[:86]}")
    for t in gone[:4]:
        print(f"      - {t[:88]}")
    for t in came[:4]:
        print(f"      + {t[:88]}")
sys.exit(1 if (changed or missing) else 0)
