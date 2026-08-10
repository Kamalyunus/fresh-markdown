"""tools.deck_text -- inspect and patch the text runs in the technical deck.

Refreshing a measured number in a .pptx by hand means unzipping it, finding
the right run of XML, editing it, and zipping it back without disturbing
anything else. That is easy to get subtly wrong -- a mis-zipped archive still
has the right bytes in it and simply will not open. This tool does the whole
round trip and refuses to guess:

  --list                  every text run, tagged with slide and shape, so the
                          exact string to replace can be copied out
  --patch patch.json      apply replacements; every entry MUST match exactly
                          once on its slide or nothing is written

Refusing on a miss is the point. A silent no-op leaves a stale number on the
slide while the run log says the refresh succeeded, which is the failure this
tool exists to prevent.

Patch file format -- a list of objects:

    [
      {"slide": 18, "old": "\\u20a91,271 / day", "new": "\\u20a91,231 / day"},
      {"slide": 2,  "old": "\\u20a914.7M",       "new": "\\u20a914.27M"}
    ]

`slide` is the 1-based position in the deck as presented, not the slideN.xml
file number -- those diverge as soon as slides are reordered.

Usage:
    python3 -m tools.deck_text --list
    python3 -m tools.deck_text --list --slide 18
    python3 -m tools.deck_text --patch refresh.json
    python3 -m tools.deck_text --patch refresh.json --dry-run
"""

import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile

DECK = "docs/perishable_markdown_tech_deck.pptx"

RUN_RE = re.compile(r"<a:t>([^<]*)</a:t>")
SHAPE_RE = re.compile(r'<p:(sp|graphicFrame)>.*?</p:\1>', re.S)
NAME_RE = re.compile(r'name="([^"]*)"')


def _unescape(t):
    return (t.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))


def _escape(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def slide_order(z):
    """Presentation order -> slideN.xml part name.

    The <p:sldIdLst> holds relationship ids in presentation order; the
    presentation rels map those to parts. Reading the parts directly and
    sorting by number gives the WRONG order on any reordered deck.
    """
    pres = z.read("ppt/presentation.xml").decode("utf8")
    rels = z.read("ppt/_rels/presentation.xml.rels").decode("utf8")
    target = {m.group(1): m.group(2) for m in
              re.finditer(r'Id="(rId\d+)"[^>]*Target="(slides/slide\d+\.xml)"', rels)}
    lst = re.search(r"<p:sldIdLst>(.*?)</p:sldIdLst>", pres, re.S).group(1)
    return ["ppt/" + target[m.group(1)]
            for m in re.finditer(r'r:id="(rId\d+)"', lst)]


def runs_of(xml):
    """(shape name, run text) for every text run, in document order."""
    out = []
    for m in SHAPE_RE.finditer(xml):
        body = m.group(0)
        nm = NAME_RE.search(body)
        name = nm.group(1) if nm else "?"
        for r in RUN_RE.finditer(body):
            out.append((name, _unescape(r.group(1))))
    return out


def do_list(deck, only):
    with zipfile.ZipFile(deck) as z:
        for i, part in enumerate(slide_order(z), start=1):
            if only and i != only:
                continue
            xml = z.read(part).decode("utf8")
            print(f"\n--- slide {i}  ({os.path.basename(part)})")
            for name, text in runs_of(xml):
                if text.strip():
                    print(f"  [{name:<12s}] {text}")


def apply_patch(deck, entries, dry_run):
    with zipfile.ZipFile(deck) as z:
        order = slide_order(z)
        parts = {n: z.read(n) for n in z.namelist()}

    by_slide = {}
    for e in entries:
        for k in ("slide", "old", "new"):
            if k not in e:
                raise SystemExit(f"patch entry missing '{k}': {e}")
        if not 1 <= e["slide"] <= len(order):
            raise SystemExit(f"slide {e['slide']} out of range 1..{len(order)}")
        by_slide.setdefault(e["slide"], []).append(e)

    problems, applied = [], []
    for n, group in sorted(by_slide.items()):
        part = order[n - 1]
        xml = parts[part].decode("utf8")
        for e in group:
            old, new = _escape(e["old"]), _escape(e["new"])
            # count occurrences inside text runs only -- never match markup
            hits = sum(t.count(old) for t in RUN_RE.findall(xml))
            if hits != 1:
                problems.append(f"slide {n}: {hits} matches for {e['old']!r} "
                                "(need exactly 1)")
                continue
            xml = RUN_RE.sub(
                lambda m: "<a:t>" + m.group(1).replace(old, new) + "</a:t>", xml)
            applied.append(f"slide {n}: {e['old']!r} -> {e['new']!r}")
        parts[part] = xml.encode("utf8")

    if problems:
        raise SystemExit("REFUSED -- nothing written:\n  " + "\n  ".join(problems))

    for line in applied:
        print(line)
    if dry_run:
        print(f"\ndry run -- {len(applied)} replacement(s) would be made")
        return

    fd, tmp = tempfile.mkstemp(suffix=".pptx",
                               dir=os.path.dirname(deck) or ".")
    os.close(fd)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for name in parts:
            out.writestr(name, parts[name])
    with zipfile.ZipFile(tmp) as check:      # refuse to ship an unopenable deck
        bad = check.testzip()
        if bad or "ppt/presentation.xml" not in check.namelist():
            os.remove(tmp)
            raise SystemExit(f"REFUSED -- rebuilt archive is broken ({bad})")
    shutil.move(tmp, deck)
    print(f"\nwrote {len(applied)} replacement(s) to {deck}")


def main():
    ap = argparse.ArgumentParser(prog="tools.deck_text")
    ap.add_argument("--deck", default=DECK)
    ap.add_argument("--list", action="store_true",
                    help="dump every text run with its slide and shape")
    ap.add_argument("--slide", type=int, help="restrict --list to one slide")
    ap.add_argument("--patch", help="JSON file of {slide, old, new} entries")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    if not os.path.exists(args.deck):
        raise SystemExit(f"no deck at {args.deck}")
    if args.list:
        do_list(args.deck, args.slide)
    elif args.patch:
        with open(args.patch) as f:
            apply_patch(args.deck, json.load(f), args.dry_run)
    else:
        ap.error("one of --list or --patch is required")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass          # `--list | head` is the expected way to use this
