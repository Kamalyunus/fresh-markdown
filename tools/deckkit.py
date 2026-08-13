"""Slide-package surgery for the v2 rebuild.

Self-contained: the pptx skill's add_slide.py is not available in this
environment, so slide duplication does its own package bookkeeping --
content types, presentation rels, and the slide id list.
"""
import os, re, shutil, subprocess, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
UNP = os.path.join(HERE, "unpacked")
SLIDE_CT = ("application/vnd.openxmlformats-officedocument."
            "presentationml.slide+xml")
SLIDE_REL = ("http://schemas.openxmlformats.org/officeDocument/2006/"
             "relationships/slide")
NOTES_REL = ("http://schemas.openxmlformats.org/officeDocument/2006/"
             "relationships/notesSlide")
NOTES_CT = ("application/vnd.openxmlformats-officedocument."
            "presentationml.notesSlide+xml")


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unpack(src):
    if os.path.isdir(UNP):
        shutil.rmtree(UNP)
    with zipfile.ZipFile(src) as z:
        z.extractall(UNP)


def _read(p):
    return open(os.path.join(UNP, p), encoding="utf8").read()


def _write(p, x):
    open(os.path.join(UNP, p), "w", encoding="utf8").write(x)


def slide_files():
    d = os.path.join(UNP, "ppt", "slides")
    return sorted((int(re.search(r"\d+", f).group()) for f in os.listdir(d)
                   if f.startswith("slide") and f.endswith(".xml")))


def order():
    """Presentation order as slide-file numbers."""
    pres, rels = _read("ppt/presentation.xml"), _read("ppt/_rels/presentation.xml.rels")
    tgt = {m.group(1): int(m.group(2)) for m in
           re.finditer(r'Id="(rId\d+)"[^>]*Target="slides/slide(\d+)\.xml"', rels)}
    lst = re.search(r"<p:sldIdLst>(.*?)</p:sldIdLst>", pres, re.S).group(1)
    return [tgt[m.group(1)] for m in re.finditer(r'r:id="(rId\d+)"', lst)]


def dup(template_n, keep_notes=False):
    """Copy slideN into a new slide part, fully registered. Returns its number."""
    n = max(slide_files()) + 1
    shutil.copy(os.path.join(UNP, "ppt/slides", f"slide{template_n}.xml"),
                os.path.join(UNP, "ppt/slides", f"slide{n}.xml"))
    src_rels = os.path.join(UNP, "ppt/slides/_rels", f"slide{template_n}.xml.rels")
    dst_rels = os.path.join(UNP, "ppt/slides/_rels", f"slide{n}.xml.rels")
    r = open(src_rels, encoding="utf8").read()
    if not keep_notes:
        # a duplicate must not inherit its template's speaker notes
        r = re.sub(r'<Relationship[^>]*Type="' + re.escape(NOTES_REL) + r'"[^>]*/>',
                   "", r)
    open(dst_rels, "w", encoding="utf8").write(r)

    ct = _read("[Content_Types].xml")
    ct = ct.replace("</Types>", f'<Override PartName="/ppt/slides/slide{n}.xml" '
                                f'ContentType="{SLIDE_CT}"/></Types>')
    _write("[Content_Types].xml", ct)

    pr = _read("ppt/_rels/presentation.xml.rels")
    rid = "rId" + str(max(int(m) for m in re.findall(r'Id="rId(\d+)"', pr)) + 1)
    pr = pr.replace("</Relationships>",
                    f'<Relationship Id="{rid}" Type="{SLIDE_REL}" '
                    f'Target="slides/slide{n}.xml"/></Relationships>')
    _write("ppt/_rels/presentation.xml.rels", pr)

    pres = _read("ppt/presentation.xml")
    sid = max(int(m) for m in re.findall(r'<p:sldId id="(\d+)"', pres)) + 1
    pres = pres.replace("</p:sldIdLst>",
                        f'<p:sldId id="{sid}" r:id="{rid}"/></p:sldIdLst>')
    _write("ppt/presentation.xml", pres)
    return n


def notes(slide_n, text, template=None):
    """Attach speaker notes to a slide that has none.

    A duplicated slide cannot simply share its template's notes part: a notes
    slide carries a back-reference to the slide it belongs to, so the part is
    cloned and rewired rather than reused.
    """
    d = os.path.join(UNP, "ppt", "notesSlides")
    nums = sorted(int(re.search(r"\d+", f).group()) for f in os.listdir(d)
                  if f.startswith("notesSlide") and f.endswith(".xml"))
    src, n = template or nums[0], max(nums) + 1

    x = _read(f"ppt/notesSlides/notesSlide{src}.xml")
    body = re.compile(r'(<p:ph type="body".*?<a:lstStyle/>).*?(</p:txBody>)', re.S)
    if not body.search(x):
        raise SystemExit(f"notesSlide{src}: no body placeholder to write into")
    x = body.sub(lambda m: (m.group(1) + "<a:p><a:r><a:t>" + esc(text)
                            + "</a:t></a:r></a:p>" + m.group(2)), x, count=1)
    _write(f"ppt/notesSlides/notesSlide{n}.xml", x)

    r = _read(f"ppt/notesSlides/_rels/notesSlide{src}.xml.rels")
    r = re.sub(r'Target="\.\./slides/slide\d+\.xml"',
               f'Target="../slides/slide{slide_n}.xml"', r)
    _write(f"ppt/notesSlides/_rels/notesSlide{n}.xml.rels", r)

    ct = _read("[Content_Types].xml")
    ct = ct.replace("</Types>",
                    f'<Override PartName="/ppt/notesSlides/notesSlide{n}.xml" '
                    f'ContentType="{NOTES_CT}"/></Types>')
    _write("[Content_Types].xml", ct)

    p = f"ppt/slides/_rels/slide{slide_n}.xml.rels"
    sr = _read(p)
    if NOTES_REL in sr:
        raise SystemExit(f"slide{slide_n} already has notes")
    rid = "rId" + str(max(int(m) for m in re.findall(r'Id="rId(\d+)"', sr)) + 1)
    sr = sr.replace("</Relationships>",
                    f'<Relationship Id="{rid}" Type="{NOTES_REL}" '
                    f'Target="../notesSlides/notesSlide{n}.xml"/></Relationships>')
    _write(p, sr)
    return n


def set_order(seq):
    pres = _read("ppt/presentation.xml")
    rels = _read("ppt/_rels/presentation.xml.rels")
    rid = {int(m.group(2)): m.group(1) for m in
           re.finditer(r'Id="(rId\d+)"[^>]*Target="slides/slide(\d+)\.xml"', rels)}
    missing = [n for n in seq if n not in rid]
    if missing:
        raise SystemExit(f"no relationship for slide files {missing}")
    lst = "".join(f'<p:sldId id="{256+i}" r:id="{rid[n]}"/>'
                  for i, n in enumerate(seq))
    pres = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>",
                  "<p:sldIdLst>" + lst + "</p:sldIdLst>", pres, flags=re.S)
    _write("ppt/presentation.xml", pres)


class Slide:
    def __init__(self, n):
        self.p = f"ppt/slides/slide{n}.xml"
        self.x = _read(self.p)

    def _shape(self, name):
        pat = re.compile(r'<p:(sp|graphicFrame)>(?:(?!<p:(?:sp|graphicFrame)>).)*?'
                         r'name="' + re.escape(name) + r'".*?</p:\1>', re.S)
        m = pat.search(self.x)
        if not m:
            raise KeyError(f"{name} not in {self.p}")
        return m

    def runs(self, name, texts):
        m = self._shape(name); body = m.group(0)
        found = re.findall(r"<a:t>[^<]*</a:t>", body)
        if len(found) != len(texts):
            raise ValueError(f"{self.p} {name}: {len(found)} runs, got {len(texts)}")
        it = iter(texts)
        body = re.sub(r"<a:t>[^<]*</a:t>",
                      lambda _: "<a:t>" + esc(next(it)) + "</a:t>", body)
        self.x = self.x[:m.start()] + body + self.x[m.end():]
        return self

    def paras(self, name, texts):
        m = self._shape(name); body = m.group(0)
        tb = re.search(r"(<p:txBody>.*?<a:lstStyle/>)(.*)(</p:txBody>)", body, re.S)
        first = re.search(r"<a:p>.*?</a:p>", tb.group(2), re.S).group(0)
        out = [re.sub(r"<a:t>[^<]*</a:t>", "<a:t>" + esc(t) + "</a:t>", first, count=1)
               for t in texts]
        body = body[:tb.start(2)] + "".join(out) + body[tb.end(2):]
        self.x = self.x[:m.start()] + body + self.x[m.end():]
        return self

    def colour(self, name, old, new):
        """Recolour one shape. A duplicated card keeps its template's palette,
        and the template's red 'this is what it costs us' number is the wrong
        signal on a slide where the same figure is the win."""
        m = self._shape(name); body = m.group(0)
        if f'val="{old}"' not in body:
            raise KeyError(f"{self.p} {name}: no {old} to recolour")
        body = body.replace(f'val="{old}"', f'val="{new}"')
        self.x = self.x[:m.start()] + body + self.x[m.end():]
        return self

    def table(self, name, cells):
        return self.runs(name, cells)

    def save(self):
        _write(self.p, self.x)


def pack(out):
    out = os.path.abspath(out)   # zip runs from the unpacked tree, not the cwd
    if os.path.exists(out):
        os.remove(out)
    subprocess.run(["zip", "-Xrq", out, "."], cwd=UNP, check=True)


def sub_in_slide(n, pairs):
    """Exact-once substring replacement inside text runs of slide file n."""
    s = Slide(n)
    for old, new in pairs:
        o, w = esc(old), esc(new)
        hits = sum(t.count(o) for t in re.findall(r"<a:t>([^<]*)</a:t>", s.x))
        if hits != 1:
            raise SystemExit(f"slide{n}: {hits} matches for {old!r} (need 1)")
        s.x = re.sub(r"<a:t>[^<]*</a:t>",
                     lambda m: m.group(0).replace(o, w), s.x)
    s.save()
