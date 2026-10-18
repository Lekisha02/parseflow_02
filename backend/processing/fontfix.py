"""Repair of broken text-layer character maps, and detection of what cannot be repaired.

Some PDFs (typically TeX books set with commercial math fonts) carry a wrong or missing ToUnicode map for their
math fonts, so the text layer reports '=' as 'D', '+' as 'C', '[' as 'Œ', '−' as a control code, and so on. The
glyph *shapes* are right — only the declared character codes are wrong. Two layers handle this:

1. Repair from the font program itself. Every embedded font names its glyphs ('equal', 'plus', 'bracketleft',
   'minus', 'uni2212' ...). For each glyph the text layer reports (PyMuPDF's text trace gives the glyph id), the
   glyph name is looked up in the embedded font (fontTools) and mapped to Unicode with the Adobe Glyph List. When
   the text layer's character is *suspicious* (a control code, U+FFFD, private use, or a letter/'Œ' where the
   glyph is named as an operator or bracket) and the glyph name gives a different, ordinary character, the glyph
   name wins. Characters that already look right are never changed.
2. Detection. Whatever still looks like a broken map (control codes, U+FFFD, private-use codes, 'Œ' inside
   formula-like tokens) is reported per segment so blocks are flagged for review and never marked verified.

Everything is cached per document and font; fonts are only parsed for pages that show suspicious characters or
use a font already found to be broken.
"""
import io
import logging
import re
import unicodedata

logging.getLogger("fontTools").setLevel(logging.ERROR)     # subset fonts trigger harmless table warnings

SUSPICIOUS = re.compile("[\u0000-\u0008\u000e-\u001f\u007f-\u009f�-]")
OE_IN_FORMULA = re.compile(r"[A-Za-z]Œ|Œ[A-Za-z0-9]")


ODD_IN_TEXT = set("<=>^`|~{}\\")
LATIN1_LETTERS = set("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ")


def suspicious_char(c):
    return bool(c) and (SUSPICIOUS.match(c) is not None or c in "Œœ")


def garbled(text):
    """True when text shows the fingerprints of a broken character map."""
    t = text or ""
    return bool(SUSPICIOUS.search(t)) or bool(OE_IN_FORMULA.search(t))


def _agl(name):
    try:
        from fontTools.agl import toUnicode
    except Exception:
        return ""
    if not name or name in (".notdef",):
        return ""
    base = name.split(".")[0]
    try:
        u = toUnicode(base)
    except Exception:
        return ""
    return u if len(u) == 1 else ""


def _glyph_order(ext, buf):
    """gid -> glyph name for an embedded font program, or None."""
    try:
        if ext in ("ttf", "otf", "ttc"):
            from fontTools.ttLib import TTFont
            return TTFont(io.BytesIO(buf), lazy=True).getGlyphOrder()
        if ext == "cff":
            from fontTools.cffLib import CFFFontSet
            cs = CFFFontSet()
            cs.decompile(io.BytesIO(buf), None)
            return cs[cs.fontNames[0]].charset
        if ext in ("pfa", "pfb"):
            from fontTools.t1Lib import T1Font
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix="." + ext, delete=False) as f:
                f.write(buf); p = f.name
            try:
                t1 = T1Font(p); t1.parse()
                return list(t1.font["CharStrings"].keys())
            finally:
                os.unlink(p)
    except Exception:
        return None
    return None


_REPLACEABLE = re.compile(r"[A-Za-zŒœ]")


def _wants(reported, named):
    """Replace the reported char by the glyph-name char?"""
    if not named or named == reported:
        return False
    if suspicious_char(reported):
        return True
    cat = unicodedata.category(named)
    # a letter reported where the font draws an operator, bracket or punctuation ('D' drawn as '=')
    return bool(_REPLACEABLE.fullmatch(reported or "")) and (cat.startswith("S") or cat.startswith("P")) and named not in "'\""


class DocFonts:
    """Per-document cache: font name -> {gid: unicode-from-glyph-name}; names of fonts found broken."""

    def __init__(self, doc):
        self.doc = doc
        self.maps = {}          # font basename -> {gid: char} or None
        self.broken = set()
        self.t3 = {}            # Type 3 font xref -> {code: CharProc xref}
        self.gids = {}          # font basename -> {char: set(gid)}   (several glyphs for one char = broken map)
        self.suspect = set()    # fonts whose map gives one character to different glyphs
        self.tt = {}            # font basename -> (glyphset, glyph order, upem) or None
        self.tt_cache = {}
        self.shape_cache = {}   # (outline hash, reported char) -> replacement or None
        self.t3_ext = {}        # (Type 3 xref, code) -> (ymin, ymax) of the glyph's ink in em, y up from the baseline
        self.last_ext = {}      # for the page just repaired: (origin x, origin y) -> (top, bottom) in page coordinates

    def export_state(self):
        """What earlier pages taught about this document's fonts (JSON-safe), for checkpoints."""
        return dict(broken=sorted(map(str, self.broken)), suspect=sorted(int(x) for x in self.suspect),
                    gids=[[int(fx), {c: sorted(v) for c, v in g.items()}] for fx, g in self.gids.items()])

    def import_state(self, st):
        self.broken |= set(st.get("broken") or [])
        self.suspect |= set(st.get("suspect") or [])
        for fx, g in st.get("gids") or []:
            d = self.gids.setdefault(int(fx), {})
            for c, v in g.items():
                d.setdefault(c, set()).update(v)

    @staticmethod
    def _base(name):
        """Comparable font name: subset prefix and punctuation removed ('ABCDEF+DejaVu Serif Book' ->
        'dejavuserifbook'). The text trace and the font list spell names differently, so lookups also accept
        a name that is a prefix of the other."""
        return re.sub(r"[^a-z0-9]", "", re.sub(r"^[A-Z]{6}\+", "", name or "").lower())

    def _page_font_xref(self, page_fonts, base):
        """The embedded font this page's text trace means by `base`. Subsets of one typeface on different pages
        share a name but number their glyphs differently, so everything per font is keyed by the font object."""
        exact = [x for x, b in page_fonts if b == base]
        if len(exact) > 1:
            return None          # two embedded fonts with one name on this page: the trace cannot say which is meant
        if exact:
            return exact[0]
        near = [x for x, b in page_fonts if b and base and (b.startswith(base) or base.startswith(b) or b.endswith(base) or base.endswith(b))]
        return near[0] if near else None

    def _map_for(self, xref):
        return self.maps.get(xref)

    def _load_page_fonts(self, page):
        for f in page.get_fonts(full=True):
            xref, ext, name = f[0], f[1], f[3]
            if xref in self.maps or not xref:
                continue
            m = None
            try:
                _, fext, _, buf = self.doc.extract_font(xref)
                order = _glyph_order(fext or ext, buf) if buf else None
                if order:
                    m = {gid: _agl(n) for gid, n in enumerate(order)}
            except Exception:
                m = None
            self.maps[xref] = m

    def _type3(self, xref):
        """code -> CharProc stream for a Type 3 font."""
        if xref in self.t3:
            return self.t3[xref]
        procs = {}
        try:
            o = self.doc.xref_object(xref)
            cp = dict(re.findall(r"/([^\s/<>\[\]()]+)\s+(\d+)\s+0\s+R", o.split("/CharProcs", 1)[1].split(">>", 1)[0]))
            enc = {}
            m = re.search(r"/Encoding\s+(\d+)\s+0\s+R", o)
            eo = self.doc.xref_object(int(m.group(1))) if m else o
            dm = re.search(r"/Differences\s*\[(.*?)\]", eo, re.S)
            if dm:
                code = 0
                for tok in re.findall(r"\d+|/[^\s/\[\]]+", dm.group(1)):
                    if tok.isdigit():
                        code = int(tok)
                    else:
                        enc[code] = tok[1:]; code += 1
            for code, name in enc.items():
                if name in cp:
                    procs[code] = int(cp[name])
            fm = re.search(r"/FontMatrix\s*\[\s*([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)", o)
            procs["_flip"] = (float(fm.group(4)) < 0) if fm else False
            procs["_ys"] = abs(float(fm.group(4))) if fm else 0.001       # glyph units -> text space (font size 1)
        except Exception:
            procs = {}
        self.t3[xref] = procs
        return procs

    def type3_char(self, xref, code, mapped):
        """Replacement for a Type 3 glyph identified by its outline, or None."""
        from . import glyphshape
        procs = self._type3(xref)
        sx = procs.get(code)
        if not sx:
            return None
        try:
            stream = self.doc.xref_stream(sx)
        except Exception:
            return None
        key = (hash(stream), mapped)
        if key in self.shape_cache:
            return self.shape_cache[key]
        r = None
        try:
            ol = glyphshape.outline(stream)
            if ol:
                r = glyphshape.identify(ol[0], ol[1], mapped, flip_y=procs.get("_flip", False))
        except Exception:
            r = None
        self.shape_cache[key] = r
        return r

    def type3_extent(self, xref, code):
        """Vertical ink extent of a Type 3 glyph (em, y up). The text layer's box for Type 3 glyphs is the font's
        box, the same for every glyph, so a big delimiter and a small parenthesis cannot be told apart from it."""
        key = (xref, code)
        if key in self.t3_ext:
            return self.t3_ext[key]
        r = None
        procs = self._type3(xref)
        sx = procs.get(code)
        if sx:
            try:
                from . import glyphshape
                ol = glyphshape.outline(self.doc.xref_stream(sx))
                if ol and ol[0]:
                    k = procs.get("_ys") or 0.001
                    ys = [(-y if procs.get("_flip") else y) * k for p in ol[0] for _, y in p]
                    r = (min(ys), max(ys))
            except Exception:
                r = None
        self.t3_ext[key] = r
        return r

    def _ttfont(self, xref):
        if xref in self.tt:
            return self.tt[xref]
        r = None
        if xref:
            if True:
                try:
                    _, ext, _, buf = self.doc.extract_font(xref)
                    if buf and ext in ("ttf", "otf", "cff"):
                        from fontTools.ttLib import TTFont
                        import io as _io
                        if ext == "cff":
                            from fontTools.cffLib import CFFFontSet
                            cs = CFFFontSet(); cs.decompile(_io.BytesIO(buf), None)
                            top = cs[cs.fontNames[0]]
                            r = (top.CharStrings, list(top.charset), 1000)
                        else:
                            tt = TTFont(_io.BytesIO(buf))
                            r = (tt.getGlyphSet(), tt.getGlyphOrder(), tt["head"].unitsPerEm)
                except Exception:
                    r = None
        self.tt[xref] = r
        return r

    def tt_char(self, xref, gid, mapped):
        key = (xref, gid, mapped)
        if key in self.tt_cache:
            return self.tt_cache[key]
        r = None
        f = self._ttfont(xref)
        if f:
            gs, order, upem = f
            try:
                from . import glyphshape
                paths = glyphshape.tt_outline(gs, order[gid], upem)
                if paths:
                    r = glyphshape.identify(paths, True, mapped, scale=1.0, flip_y=False, math=False)
            except Exception:
                r = None
        self.tt_cache[key] = r
        return r

    def page_repairs(self, page, chars_text, has_type3=False):
        """{(rounded origin x, rounded origin y): char} for glyphs whose text-layer char should be replaced. Also
        fills self.last_ext with the true vertical extent of the page's Type 3 glyphs."""
        self.last_ext = {}
        if not (has_type3 or SUSPICIOUS.search(chars_text) or "Œ" in chars_text or "œ" in chars_text or self.broken or self.suspect
                or re.search(r"[a-z][0-9][a-z]|[^\x00-\x7f\u2010-\u203a\u00a0-\u00ff]", chars_text)):
            return {}
        try:
            trace = page.get_texttrace()
        except Exception:
            return {}
        if not trace:
            return {}
        if any(not (sp.get("font") or "").startswith("Type3") for sp in trace) and (SUSPICIOUS.search(chars_text) or "Œ" in chars_text or self.broken):
            self._load_page_fonts(page)
        out = {}
        page_fonts = None
        # pass 1: which glyphs each font uses for each character. A font that draws one character with two glyphs
        # ('3' for both the digit and the minus sign) is suspect for ALL its characters on the page, including those
        # met before the second glyph shows up (one repair used to depend on the order of the text runs).
        for sp in trace:
            fname = sp.get("font") or ""
            if re.match(r"Type3 \((\d+) 0 R\)", fname):
                continue
            if page_fonts is None:
                page_fonts = [(f[0], self._base(f[3])) for f in page.get_fonts(full=True)]
            fx = self._page_font_xref(page_fonts, self._base(fname))
            if fx is None:
                continue
            g = self.gids.setdefault(fx, {})
            chs = list(sp.get("chars", ()))
            lig = {k for k, ch in enumerate(chs) if k + 1 < len(chs) and isinstance(chs[k + 1][1], int) and chs[k + 1][1] < 0}
            for k, ch in enumerate(chs):
                if isinstance(ch[0], int) and ch[0] > 32 and k not in lig and isinstance(ch[1], int) and ch[1] >= 0:
                    g.setdefault(chr(ch[0]), set()).add(ch[1])
            if fx not in self.suspect and any(len(v) > 1 for v in g.values()):
                self.suspect.add(fx)
        for sp in trace:
            fname = sp.get("font") or ""
            t3 = re.match(r"Type3 \((\d+) 0 R\)", fname)
            if t3:
                xref = int(t3.group(1))
                size = sp.get("size") or 0
                for ch in sp.get("chars", ()):
                    u, code, origin = ch[0], ch[1], ch[2]
                    ext = self.type3_extent(xref, code) if size else None
                    if ext:
                        self.last_ext[(round(origin[0], 1), round(origin[1], 1))] = (origin[1] - ext[1] * size, origin[1] - ext[0] * size)
                    rep = chr(u) if isinstance(u, int) and u >= 0 else ""
                    new = self.type3_char(xref, code, rep)
                    if new and new != rep:
                        out[(round(origin[0], 1), round(origin[1], 1))] = new
                        self.broken.add(fname)
                continue
            base = self._base(fname)
            if page_fonts is None:
                page_fonts = [(f[0], self._base(f[3])) for f in page.get_fonts(full=True)]
            fx = self._page_font_xref(page_fonts, base)
            if fx is None:
                continue
            g = self.gids.setdefault(fx, {})
            chs = list(sp.get("chars", ()))
            # a ligature glyph ('t_i', 'f_f') is reported as its first letter followed by letters with no glyph of
            # their own (gid -1) at the same origin: it is a correct mapping, not a second glyph for that letter
            lig = {k for k, ch in enumerate(chs) if k + 1 < len(chs) and isinstance(chs[k + 1][1], int) and chs[k + 1][1] < 0}
            sus = fx in self.suspect
            m = self._map_for(fx)
            for k, ch in enumerate(chs):
                u, gid, origin = ch[0], ch[1], ch[2]
                if k in lig or (isinstance(gid, int) and gid < 0):
                    continue
                rep = chr(u) if isinstance(u, int) and u >= 0 else ""
                at = (round(origin[0], 1), round(origin[1], 1))
                # 1. the font's own glyph name ('equal' drawn where the map says 'D')
                named = m.get(gid, "") if m else ""
                if _wants(rep, named):
                    out[at] = named
                    self.broken.add(base)
                    continue
                # 2. the glyph outline, for suspicious characters and fonts whose map gives one char to several glyphs
                if not rep or (ord(rep) <= 32 and not suspicious_char(rep)):
                    continue
                odd = suspicious_char(rep) or rep in ODD_IN_TEXT or (not rep.isascii() and rep.isalpha() and rep in LATIN1_LETTERS)
                if not (odd or (sus and not (len(g.get(rep, ())) < 2 and rep.isascii() and rep.isalnum()))):
                    continue        # ordinary characters of a font without signs of a broken map are trusted
                new = self.tt_char(fx, gid, rep)
                if new and new != rep:
                    out[at] = new
        return out


_CACHE = {}


def for_doc(doc):
    key = id(doc)
    d = _CACHE.get(key)
    if d is None or d.doc is not doc:
        if len(_CACHE) > 4:
            _CACHE.clear()
        d = _CACHE[key] = DocFonts(doc)
    return d
