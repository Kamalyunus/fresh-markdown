"""tools.deck_diff -- regression check between two versions of the deck.

Restructuring a deck is exactly when vetted content gets silently dropped or
mangled. This pairs slides by their first two text runs (eyebrow + title),
then reports what is NEW, what was DROPPED, and any reused slide whose text
changed for a reason not on the INTENDED list. Exits non-zero on the last two,
so it can gate a rebuild.

Usage:
    python3 -m tools.deck_diff [--old A.pptx] [--new B.pptx]
"""
import re, sys, zipfile

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
ap.add_argument("--new", default="docs/perishable_markdown_deck_v2.pptx")
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
        diffs = [(a, b) for a, b in zip(old, s) if a != b]
        extra = abs(len(old) - len(s))
        ok = all(any(t in a or t in b for t in INTENDED) for a, b in diffs)
        if not ok or extra:
            changed.append((i, j, diffs, extra))

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
for i, j, diffs, extra in changed:
    print(f"  v1 {i} -> v2 {j}  extra_runs={extra}")
    for a, b in diffs[:4]:
        print(f"      - {a[:80]}\n      + {b[:80]}")
sys.exit(1 if (changed or missing) else 0)
