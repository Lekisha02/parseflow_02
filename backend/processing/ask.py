"""Ask Your Document: extractive question answering over a processed ParseFlow document.

No model is involved. Answers are copied from extracted blocks, never generated, and every answer part carries the
block it came from so the viewer can show the source region. Four strategies, tried in this order:

  page     "What does page 4 explain?"            -> the page's headings and opening content, in reading order
  cell     "What was the total revenue in FY24?"  -> one table cell, found by matching row label and column header
  section  "Algorithm for AVL deletion"            -> a heading that matches the question, returned with its content
  passage  "What is the normalization factor?"     -> the best-matching sentences (BM25 over blocks, phrase and
                                                      definition bonuses), plus other passages that use the same phrase

Answers are returned as typed parts (text with inline LaTeX, display math, table, code, list, heading) so the UI can
render math and tables instead of flattening them. If nothing matches well enough, found=False with the
closest candidates for checking. Nothing is invented or summarised.
"""
import math
import re
from collections import Counter, defaultdict

STOP = set("""a an the of in on at to for from by with about into over under between through during before after above below
and or but nor so yet if then than that this these those there here which who whom whose what when where why how is are was
were be been being am do does did done doing have has had having it its it's they them their i me my we our you your he she
his her him can could should would may might must shall will page pages document doc pdf file please tell show give explain
explains explained describe describes described mean means meaning say says said shown shows contain contains contained
list lists find get let us any some all each much many also just only very more most other such as not no yes one
summarize summarise summary overview define definition according""".split())
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota",
         "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau",
         "υ": "upsilon", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "gamma", "Δ": "delta", "Θ": "theta",
         "Λ": "lambda", "Ξ": "xi", "Π": "pi", "Σ": "sigma", "Φ": "phi", "Ψ": "psi", "Ω": "omega", "ϕ": "phi", "ϵ": "epsilon"}
SUFFIXES = ("izations", "ization", "isation", "ational", "ations", "ation", "izing", "ising", "ized", "ised", "izes", "ises", "ize", "ise",
            "nesses", "ness", "ments", "ment", "ings", "ing", "edly", "ions", "ion", "ed", "ies", "es", "ly")
FURNITURE_ROLES = ("header", "footer", "page_number", "margin")
TEXT_TYPES = ("paragraph", "list", "caption", "heading", "code", "handwritten_text", "handwritten_code")
DEF_PAT = r"(?:is|are|was|were)\s+(?:called|known\s+as|termed|named|defined\s+as|referred\s+to\s+as)"


def stem(w):
    if not w.isascii() or len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 4:
        w = w[:-1]
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 4:
            w = w[: -len(s)]
            if s == "ies":
                w += "y"
            elif s in ("ing", "ings", "ed", "edly") and len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "lsz" and w[-1] not in "aeiou":
                w = w[:-1]               # debugging -> debug, running -> run, stopped -> stop
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


_SOFT_WORDS = """happen happens happened compute computed computing calculate calculated calculation determine determined
work works working use used using obtain obtained find found given done perform performed handle handled handles occur occurs
need needed needs required require role purpose main key important""".split()


def latex_words(s):
    """Searchable words from LaTeX: \\psi -> psi, \\mathbf{r} -> r, \\frac{a}{b} -> a b."""
    s = re.sub(r"\\(?:mathbf|mathrm|mathit|mathcal|mathbb|boldsymbol|text|textbf|textit|operatorname|left|right|big|Big|bigg|Bigg)\b", " ", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: " " + m.group(1) + " ", s)
    return re.sub(r"[{}_^\\$]", " ", s)


ABBR = {"docs": "documents", "doc": "document", "no": "number", "num": "number", "nos": "numbers", "avg": "average",
        "pct": "percent", "yr": "year", "yrs": "years", "fig": "figure", "figs": "figures", "eq": "equation", "eqn": "equation",
        "approx": "approximately", "govt": "government", "dept": "department", "intl": "international", "mln": "million",
        "mn": "million", "bn": "billion", "tn": "trillion", "qty": "quantity", "amt": "amount", "temp": "temperature",
        "info": "information", "min": "minimum", "max": "maximum", "est": "estimated", "proj": "projected"}


def tokens(text, keep_stop=False):
    t = latex_words(text or "")
    t = "".join(" " + GREEK[c] + " " if c in GREEK else c for c in t)
    words = [ABBR.get(w, w) for w in re.findall(r"[^\W_]+", t.lower())]
    return [stem(w) for w in words if keep_stop or (w not in STOP and len(w) > 1) or w.isdigit()]


def plain(text):
    """Readable plain text for snippets: inline-math delimiters removed."""
    t = re.sub(r"\\\(\s*|\s*\\\)", "", text or "")
    t = re.sub(r"\\\[\s*|\s*\\\]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def snippet(b, n=170):
    """A short preview of a block that keeps inline math intact (\\( ... \\) is rendered by the viewer):
    cut at a word boundary outside math, never inside a formula."""
    if b.get("type") == "equation" and b.get("latex"):
        return "\\(" + b["latex"] + "\\)" if len(b["latex"]) <= 300 else ""
    if b.get("type") == "table" and b.get("rows"):
        t = " · ".join(str(c) for c in b["rows"][0] if str(c).strip())
    else:
        t = re.sub(r"\s+", " ", b.get("content") or "").strip()
    t = re.sub(r"^[•‣●○▪■∙·➢⁃◦]\s*", "", t)
    if len(t) <= n:
        return t
    cut, depth, i = 0, 0, 0
    while i < len(t) and i <= n:
        if t.startswith("\\(", i):
            depth += 1; i += 2; continue
        if t.startswith("\\)", i):
            depth = max(0, depth - 1); i += 2
            if depth == 0:
                cut = i
            continue
        if depth == 0 and t[i] == " ":
            cut = i
        i += 1
    return t[:cut].rstrip(" ,;:") + " …"


def block_text(b):
    t = b.get("content") or ""
    if b.get("type") == "table" and b.get("rows"):
        t = "\n".join(" ".join(str(c) for c in r) for r in b["rows"])
    if b.get("type") == "equation" and b.get("latex"):
        t = b["latex"]
    extra = " ".join(str(x) for x in (b.get("elements") or []))
    if b.get("ocr_text"):
        extra += " " + b["ocr_text"]
    if b.get("title"):
        extra += " " + str(b["title"])
    return (t + " " + extra).strip()


SOFT = {stem(w) for w in _SOFT_WORDS}


class Index:
    """Per-document index: blocks in reading order, section context, BM25 statistics."""

    def __init__(self, doc):
        self.doc = doc
        bl = [b for b in doc.get("blocks", []) if (b.get("role") not in FURNITURE_ROLES) and not ((b.get("furniture") or {}).get("excluded"))]
        bl.sort(key=lambda b: (b.get("page", 0), b.get("reading_order", 0)))
        self.blocks = bl
        self.by_id = {b["block_id"]: b for b in bl}
        self.toks = [tokens(block_text(b)) for b in bl]
        # section context: the nearest heading above each block (and the heading chain)
        self.section_of, stack = [], []
        for i, b in enumerate(bl):
            if b.get("type") == "heading":
                lvl = b.get("level") or 1
                stack = [s for s in stack if (bl[s].get("level") or 1) < lvl] + [i]
                self.section_of.append(None)
            else:
                self.section_of.append(stack[-1] if stack else None)
        self.ctx_toks = [tokens(" ".join(bl[s].get("content") or "" for s in ([self.section_of[i]] if self.section_of[i] is not None else []))) for i in range(len(bl))]
        self.N = len(bl)
        df = Counter()
        for t in self.toks:
            df.update(set(t))
        self.df = df
        self.avgdl = (sum(len(t) for t in self.toks) / self.N) if self.N else 1.0
        self.tf = [Counter(t) for t in self.toks]
        self.ctx_tf = [Counter(t) for t in self.ctx_toks]

    def idf(self, w):
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def bm25(self, q, i, k1=1.2, bb=0.75):
        tf, dl = self.tf[i], len(self.toks[i]) or 1
        s = 0.0
        for w in q:
            f = tf.get(w, 0)
            if f:
                s += self.idf(w) * f * (k1 + 1) / (f + k1 * (1 - bb + bb * dl / self.avgdl))
        return s


_CACHE = {}


def index_for(doc, key=None):
    if key is not None and key in _CACHE:
        return _CACHE[key]
    ix = Index(doc)
    if key is not None:
        if len(_CACHE) >= 4:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = ix
    return ix


# ---------------------------------------------------------------------------------------------
# Answer parts
# ---------------------------------------------------------------------------------------------
def part_for(b, excerpt=None):
    t, c = b.get("type"), b.get("content")
    base = dict(block_id=b["block_id"], page=b.get("page"))
    if t == "heading":
        return dict(base, kind="heading", text=c or "")
    if t == "table" and b.get("rows"):
        return dict(base, kind="table", rows=b["rows"], header_rows=b.get("header_rows") or 1, html=b.get("markdown") if (b.get("markdown") or "").lstrip().startswith("<table") else None)
    if t == "equation":
        lx = b.get("latex") or ""
        if lx:
            return dict(base, kind="math", latex=lx)
        return dict(base, kind="text", text=c or "")
    if t in ("code", "handwritten_code"):
        if excerpt is not None and excerpt.strip() != (c or "").strip():
            return dict(base, kind="code", text=excerpt, full=c)
        return dict(base, kind="code", text=c or "")
    if t == "list" and c and excerpt is None:
        items, ordered = list_items(c)
        if len(items) > 1:
            return dict(base, kind="list", items=items, ordered=ordered)
        return dict(base, kind="text", text=items[0] if items else _unwrap(c))
    if c is None:
        return None
    if excerpt is not None and excerpt.strip() != (c or "").strip():
        return dict(base, kind="text", text=_unwrap(excerpt), full=c)
    return dict(base, kind="text", text=_unwrap(c))


_ITEM = re.compile(r"^\s*(?:[•‣●○▪■–—∙·➢⁃*◦-]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+")


def list_items(c):
    """List items from a list block: a line without its own bullet/enumerator continues the item above it
    (wrapped lines are not separate items). Returns (items, ordered)."""
    items, marks = [], []
    for l in (c or "").split("\n"):
        if not l.strip():
            continue
        m = _ITEM.match(l)
        if m or not items:
            items.append(_ITEM.sub("", l).strip()); marks.append(m.group(0).strip() if m else "")
        else:
            items[-1] = (items[-1] + " " + l.strip()).strip()
    ordered = bool(marks) and all(re.match(r"^\(?\d{1,3}[.)]", m) for m in marks)
    return [i for i in items if i], ordered


_BUL = r"(?:[•‣●○▪■–—∙·➢⁃*◦-]|\(?\d{1,3}[.)])\s"


def _unwrap(t):
    """Join the visual line breaks of a wrapped paragraph; keep breaks before bullets. A lone leading bullet is dropped."""
    t = re.sub(r"\s*\n\s*(?!" + _BUL + ")", " ", t or "").strip()
    t = re.sub(r"\s+¦\s+", "\n• ", t)            # a bullet glyph some fonts map to '¦' (inline bullet runs)
    if len(re.findall(r"(?:^|\n)\s*" + _BUL, t)) <= 1:
        t = re.sub(r"^\s*(?:[•‣●○▪■∙·➢⁃◦¦]|\\\(-\\\))\s*", "", t)
    return t


def sentences(text):
    """Split prose into sentences without breaking inside \\( ... \\) math."""
    maths = []

    def keep(m):
        maths.append(m.group(0)); return f"\x00{len(maths) - 1}\x00"
    t = re.sub(r"\\\(.*?\\\)", keep, text or "", flags=re.S)
    t = re.sub(r"\s*\n\s*(?=[•‣●○▪■∙·➢◦*-]\s)", " \x01", t)
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'“(\[]?[A-Z0-9\x00•])|\s*\x01|\s+(?=[•‣●○▪■∙➢◦]\s)", t)
    out = []
    for p in parts:
        p = re.sub(r"\x00(\d+)\x00", lambda m: maths[int(m.group(1))], p).strip()
        if p:
            out.append(p)
    return out


def best_excerpt(text, q, phrase_res, max_chars=650):
    """The sentence(s) of a block that answer the question; the whole block when it is short."""
    if len(plain(text)) <= max_chars:
        return text
    ss = sentences(text)
    if len(ss) <= 1:
        return text
    qs = set(q)
    scores = []
    for s in ss:
        st = set(tokens(s))
        sc = len(qs & st) + 1.5 * sum(1 for r in phrase_res if r.search(plain(s).lower())) + (1.0 if re.search(DEF_PAT, s, re.I) else 0)
        scores.append(sc)
    i = max(range(len(ss)), key=lambda k: (scores[k], -k))
    lo, hi = i, i
    if i > 0 and re.match(r"^(it|this|these|they|such|thus|hence|therefore|here)\b", plain(ss[i]), re.I):
        lo = i - 1
    while hi + 1 < len(ss) and scores[hi + 1] > 0 and hi - lo < 2:
        hi += 1
    if len(plain(" ".join(ss[lo:hi + 1]))) < 120 and hi + 1 < len(ss):
        hi += 1
    ex = " ".join(ss[lo:hi + 1])
    return ("… " if lo > 0 else "") + ex + (" …" if hi < len(ss) - 1 else "")


# ---------------------------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------------------------
def _coverage(ix, q, toks_set):
    # a word the document never uses would otherwise get the largest idf of all; weigh it like the rarest word it has
    present = [ix.idf(w) for w in set(q) if ix.df.get(w)] if hasattr(ix, "df") else []
    cap = max(present) if present else 1.0
    wt = {w: min(ix.idf(w), cap) * (0.25 if w in SOFT else 1.0) for w in set(q)}
    tot = sum(wt.values()) or 1.0
    return sum(v for w, v in wt.items() if w in toks_set) / tot


def _assigns(line, qs):
    """'24. balance = height(...)' assigns to a question word: the line that defines it."""
    m = re.match(r"^\s*(?:\(?\d{1,3}[.):]\s*)?(?:(?:int|float|double|var|let|const|auto|long)\s+)?([A-Za-z_]\w*)(?:\s*->\s*\w+|\.\w+|\[[^\]]*\])?\s*(?::=|=|←|<-)(?!=)", line)
    return bool(m) and stem(m.group(1).lower()) in qs


def code_excerpt(text, q, max_lines=14):
    """The lines of a code block that answer the question: a commented step ('// Right Heavy Case') with the lines
    under it, or a matching statement with one line of context. None when the whole block is short."""
    lines = (text or "").split("\n")
    if len(lines) <= max_lines:
        return None
    qs = set(q)
    sc = [len(qs & set(tokens(l))) + (1.5 if _assigns(l, qs) else 0) for l in lines]
    i = max(range(len(lines)), key=lambda k: (sc[k], -k))
    if sc[i] == 0:
        return None
    body = lambda l: re.sub(r"^\s*\(?\d{1,3}[.):]?\s?", "", l)
    indent = lambda l: len(body(l)) - len(body(l).lstrip())
    is_comment = lambda l: re.match(r"^\s*(//|#|/\*|--)", body(l)) is not None
    if is_comment(lines[i]):
        j = i + 1
        while j < len(lines) and j - i < max_lines and not (is_comment(lines[j]) and indent(lines[j]) <= indent(lines[i])) \
                and not (body(lines[j]).strip() and indent(lines[j]) < indent(lines[i])):   # dedent: the step is over
            j += 1
        lo, hi = i, j - 1
    else:
        lo, hi = max(0, i - 1), min(len(lines) - 1, i + 1)
    return "\n".join(lines[lo:hi + 1])


def _phrases(question):
    """Multi-word phrases of the question (content words that are adjacent in the question), as regexes on plain text."""
    words = re.findall(r"[^\W_]+", question.lower())
    out, cur = [], []
    for w in words + ["the"]:
        if w in STOP or len(w) <= 1:
            if len(cur) >= 2:
                out.append(cur)
            cur = []
        else:
            cur.append(w)
    res = []
    for ph in out:
        pat = r"\b" + r"\W+".join(re.escape(stem(w)) + r"\w*" for w in ph)
        res.append(re.compile(pat))
    return res


def page_answer(ix, n, max_parts=8):
    bl = [b for b in ix.blocks if b.get("page") == n]
    if not bl:
        return None
    parts, used, chars = [], [], 0
    for b in bl:
        if len(parts) >= max_parts or chars > 1600:
            break
        t = b.get("type")
        if not parts and t in ("paragraph", "list") and re.match(r"^[a-z(,;]", plain(b.get("content") or "")) and len(bl) > 1:
            continue                     # the end of a sentence from the previous page
        if t in ("paragraph", "caption", "handwritten_text") and b.get("content"):
            ss = sentences(b["content"])
            ex = ss[0] if ss else b["content"]
            if len(ss) > 1:
                ex += " …"
            p = part_for(b, ex)
        elif t == "table" and b.get("rows"):
            p = dict(block_id=b["block_id"], page=n, kind="text", text=f"Table with {len(b['rows'])} rows: " + " · ".join(str(c) for c in b["rows"][0] if c)[:200])
        elif t in ("figure", "chart", "diagram", "unknown") and not b.get("content"):
            continue
        else:
            p = part_for(b)
        if p:
            parts.append(p); used.append(b)
            chars += len(p.get("text") or p.get("latex") or "")
    return parts, used


def _header_rows(b):
    """Header rows of a table for lookup: the declared count, extended over leading rows that have no row label
    or no numbers (e.g. 'Year Ended' / 'Jan 26, 2025' / '(In millions)' above the first data row)."""
    rows = b["rows"]
    hr = max(1, min(b.get("header_rows") or 1, len(rows) - 1))
    while hr < len(rows) - 1:
        r = rows[hr]
        label = str(r[0]).strip() if r else ""
        nums = sum(1 for c in r[1:] if re.search(r"\d", str(c)))
        filled = sum(1 for c in r[1:] if str(c).strip())
        if label and nums:
            break
        unit_row = filled == 1 and re.match(r"^\(?\s*(?:in\s+)?(?:millions|thousands|billions|percent|%|us\$|usd)", " ".join(str(c) for c in r[1:]).strip(), re.I)
        if not label and filled < 2 and not unit_row:
            break                        # a category row ('Pretraining - OLMo 2 Mix') starts the body
        if label and not nums and any(re.search(r"\d", str(c)) for rr in rows[hr + 1:] for c in rr[1:]) and len(rows) - hr > 2:
            # a label row without values (a category row) inside the body: stop
            break
        hr += 1
    return hr


def _col_header(rows, hr, c):
    parts = []
    for k in range(hr):
        r = rows[k]
        v = str(r[c]).strip() if c < len(r) else ""
        if not v:                       # a spanning header written in an earlier column
            for cc in range(c - 1, 0, -1):
                if cc < len(r) and str(r[cc]).strip():
                    if all(not str(rows[k2][cc2]).strip() for k2 in range(k + 1, hr) for cc2 in [cc] if cc2 < len(rows[k2])) or k == 0:
                        v = str(r[cc]).strip()
                    break
        if v and not re.match(r"^\(?in\s+(millions|thousands|billions)", v, re.I):
            parts.append(v)
    return parts


def _row_labels(rows, hr):
    """Row labels with wrapped label lines merged in: a label-only row joins the next data row, or the previous one
    when it reads as a continuation ('from Dolma 1.7', 'and ...', '(...)')."""
    labels = {r: str(rows[r][0]).strip() if rows[r] else "" for r in range(hr, len(rows))}
    data = [r for r in range(hr, len(rows)) if any(re.search(r"\d", str(c)) for c in rows[r][1:])]
    out = {r: labels[r] for r in data}
    for r in range(hr, len(rows)):
        if r in out or not labels[r] or any(str(c).strip() for c in rows[r][1:]):
            continue
        prev = max((d for d in data if d < r), default=None)
        nxt = min((d for d in data if d > r), default=None)
        cont = re.match(r"^(?:[a-z(\[]|and\b|or\b|of\b|from\b|in\b)", labels[r])
        tgt = prev if (cont and prev is not None) or nxt is None else nxt
        if tgt is not None:
            out[tgt] = (out[tgt] + " " + labels[r]) if tgt < r else (labels[r] + " " + out[tgt])
    return out


def cell_answer(ix, q, question):
    """A table cell whose row label and column header together match the question."""
    qs = set(q)
    if len(qs) < 2:
        return None
    best = None
    for b in ix.blocks:
        rows = b.get("rows") if b.get("type") == "table" else None
        if not rows or len(rows) < 2:
            continue
        hr = _header_rows(b)
        ncol = max(len(r) for r in rows)
        heads = [set(tokens(" ".join(_col_header(rows, hr, c)))) for c in range(ncol)]
        cap = set()
        for rel in b.get("relationships") or []:
            if rel.startswith("caption:") and rel[8:] in ix.by_id:
                cap |= set(tokens(ix.by_id[rel[8:]].get("content") or ""))
        rlab = _row_labels(rows, hr)
        for r in range(hr, len(rows)):
            row = rows[r]
            label = rlab.get(r, "")
            lt = set(tokens(label))
            if not lt & qs:
                continue
            for c in range(1, len(row)):
                val = str(row[c]).strip()
                if not val or not re.search(r"\d", val):
                    continue
                ht = heads[c] if c < len(heads) else set()
                if not ht & qs:
                    continue
                matched = (lt | ht | cap) & qs
                cov = _coverage(ix, q, matched)
                sc = cov + 0.3 * len(lt & qs) / max(1, len(lt)) + 0.2 * len(ht & qs) / max(1, len(ht))
                if cov >= 0.6 and (best is None or sc > best[0]):
                    best = (sc, b, r, c, cov, hr, matched, label)
    if not best:
        return None
    _, b, r, c, cov, hr, matched, label = best
    rows = b["rows"]
    head = " · ".join(_col_header(rows, hr, c))
    unit = next((str(x).strip() for k in range(hr) for x in rows[k] if re.match(r"^\(?in\s+(millions|thousands|billions)", str(x).strip(), re.I)), "")
    text = f"{label} — {head}: **{rows[r][c]}**" + (f" {unit}" if unit else "")
    sub = [rows[k] for k in range(hr)] + [rows[r]]
    parts = [dict(block_id=b["block_id"], page=b.get("page"), kind="text", text=text, markdown=True),
             dict(block_id=b["block_id"], page=b.get("page"), kind="table", rows=sub, header_rows=hr, highlight=[hr, c],
                  note=f"Row {r + 1 - hr} of {len(rows) - hr} of the table on page {b.get('page')}")]
    return parts, [b], cov, matched


def section_blocks(ix, hi, max_blocks=16, max_chars=4500):
    h = ix.blocks[hi]
    lvl = h.get("level") or 1
    out, chars = [h], 0
    for b in ix.blocks[hi + 1:]:
        if b.get("type") == "heading" and (b.get("level") or 1) <= lvl:
            break
        if b.get("type") in ("figure", "unknown") and not b.get("content"):
            continue
        out.append(b)
        chars += len(block_text(b))
        if len(out) > max_blocks or chars > max_chars:
            break
    return out


def answer(doc, question, key=None):
    question = (question or "").strip()
    res = dict(question=question, found=False, strategy=None, parts=[], evidence=[], confidence=None, matched_terms=[], missing_terms=[], note=None)
    if not question:
        res["note"] = "Empty question."
        return res
    ix = index_for(doc, key)
    doc = ix.doc
    if not ix.N:
        res["note"] = "This document has no extracted text to search."
        return res
    m = re.search(r"\bpage\s*(\d+)\b", question, re.I)
    page = int(m.group(1)) if m else None
    q = tokens(re.sub(r"\bpage\s*\d+\b", " ", question, flags=re.I))
    raw_words = [w for w in re.findall(r"[^\W_]+", question.lower())]
    phrase_res = _phrases(question)

    def finish(strategy, parts, used, cov=None, matched=None):
        parts = [p for p in parts if p and (p.get("kind") != "text" or plain(p.get("text") or "").strip())]
        res.update(found=True, strategy=strategy, parts=parts)
        seen = []
        for b in used:
            if b["block_id"] not in [e["block_id"] for e in seen]:
                seen.append(dict(block_id=b["block_id"], page=b.get("page"), type=b.get("type"), confidence=b.get("confidence"),
                                 verification=b.get("verification"), snippet=snippet(b)))
        res["evidence"] = seen
        confs = [b.get("confidence") for b in used if b.get("confidence") is not None]
        res["confidence"] = round(min(confs), 3) if confs else None
        if matched is not None:
            res["matched_terms"] = sorted({w for w in raw_words if stem(w) in matched and w not in STOP})
            res["missing_terms"] = sorted({w for w in raw_words if stem(w) in set(q) - matched and w not in STOP})
        if cov is not None:
            res["coverage"] = round(cov, 2)
        return res

    # page questions ---------------------------------------------------------------------------
    if page is not None:
        if not any(b.get("page") == page for b in doc.get("blocks", [])):
            if page > len(doc.get("pages", [])):
                res["note"] = f"The document has {len(doc.get('pages', []))} pages; there is no page {page}."
            else:
                res["note"] = f"No content was extracted from page {page}."
            return res
        if not q:
            pa = page_answer(ix, page)
            if pa:
                return finish("page", *pa)
    cand = [i for i in range(ix.N) if page is None or ix.blocks[i].get("page") == page]
    if not q:
        res["note"] = "Ask about something in the document (the question has no searchable words)."
        return res

    # table cell ---------------------------------------------------------------------------------
    ca = cell_answer(ix if page is None else _sub(ix, page), q, question)
    if ca and ca[2] >= 0.999:
        return finish("cell", *ca)

    # score blocks ---------------------------------------------------------------------------------
    scored = []
    for i in cand:
        b = ix.blocks[i]
        body = ix.bm25(q, i)
        ctx = 0.0
        if ix.section_of[i] is not None:
            tf = ix.ctx_tf[i]
            ctx = 0.5 * sum(ix.idf(w) for w in set(q) if tf.get(w))
        txt = plain(block_text(b)).lower()
        ph = sum(1 for r in phrase_res if r.search(txt))
        dfn = 1.0 if ph and re.search(DEF_PAT + r"|\bmeans\b|\brefers\s+to\b|\bis\s+defined\b", txt) else 0.0
        if b.get("type") in ("code", "handwritten_code") and any(_assigns(l, set(q)) for l in (b.get("content") or "").split("\n")):
            dfn = 1.5
        if body <= 0 and ctx <= 0:
            continue
        matched = set(ix.toks[i]) & set(q)
        cov = _coverage(ix, q, matched | (set(ix.ctx_toks[i]) & set(q)))
        if b.get("type") in ("figure", "unknown") and not b.get("content"):
            continue
        sc = body + ctx + 1.5 * ph + dfn + 0.6 * cov
        if b.get("type") == "table":
            sc *= 0.6
        scored.append((sc, i, cov, matched, ph))
    if ca:
        good = [x for x in scored if x[2] >= 0.6 and ix.blocks[x[1]].get("type") != "table"]
        if not good or ca[2] >= max(x[2] for x in good) - 1e-6:
            return finish("cell", *ca)      # no passage covers more of the question than the table cell does
    if not scored:
        res["note"] = "None of the question's key words occur in the extracted text."
        res["missing_terms"] = sorted({w for w in raw_words if w not in STOP and len(w) > 1})
        return res
    scored.sort(key=lambda x: -x[0])

    # section: a heading that the question names -------------------------------------------------
    heads = [(sc, i, cov, mt, ph) for sc, i, cov, mt, ph in scored if ix.blocks[i].get("type") == "heading"]
    if heads:
        hs = max(heads, key=lambda x: (x[2], x[0]))
        sc, hi, cov, mt, ph = hs
        hcov = _coverage(ix, q, mt)
        htoks = set(ix.toks[hi])
        hq = len(htoks & set(q)) / max(1, len(htoks))           # how much of the heading the question names
        if hcov >= 0.75 and hq >= 0.5:
            sb = section_blocks(ix, hi)
            if len(sb) > 1:
                return finish("section", [part_for(b) for b in sb], sb, hcov, mt)

    top = scored[0]
    sc0, i0, cov0, mt0, ph0 = top
    if cov0 < 0.5 or (len(set(q)) >= 3 and cov0 < 0.6):
        res["note"] = "No passage matches enough of the question."
        res["closest"] = [dict(block_id=ix.blocks[i]["block_id"], page=ix.blocks[i].get("page"), snippet=snippet(ix.blocks[i]))
                          for _, i, *_ in scored[:3]]
        res["missing_terms"] = sorted({w for w in raw_words if stem(w) in set(q) - mt0 and w not in STOP})
        return res
    b0 = ix.blocks[i0]
    if b0.get("type") == "heading":
        sb = section_blocks(ix, i0)
        if len(sb) > 1:
            return finish("section", [part_for(b) for b in sb], sb, cov0, mt0 | (set(ix.toks[i0]) & set(q)))
    used = [b0]
    if b0.get("type") in ("code", "handwritten_code"):
        ex = code_excerpt(b0.get("content") or "", q)
    elif b0.get("type") in ("paragraph", "caption", "handwritten_text", "list") and b0.get("content"):
        ex = best_excerpt(b0.get("content") or "", q, phrase_res)
    else:
        ex = None
    parts = [part_for(b0, ex)]
    # complete a passage that the layout split: a block that starts mid-sentence (or mid-line, '∫ Ψ*Ψ dv = 1,   1/√N is
    # called ...') is joined to the block(s) it continues; a short term ('ClrErr') gets the text beside it on its row.
    if b0.get("type") not in ("heading", "table", "code") and parts[0] and parts[0].get("kind") == "text":
        lead_blocks = []
        k = i0
        while k > 0 and len(lead_blocks) < 2:
            pb, cur = ix.blocks[k - 1], ix.blocks[k]
            if pb.get("page") != cur.get("page") or pb.get("type") in ("table", "code", "equation") or not pb.get("bbox") or not cur.get("bbox"):
                break
            ptxt = plain(pb.get("content") or "")
            mid_line = _same_line(pb["bbox"], cur["bbox"]) and len(ptxt) < 200
            continues = re.match(r"^[a-z(,;]", plain(cur.get("content") or "")) and not re.search(r"[.!?:]\s*$", ptxt) \
                and abs(pb["bbox"][0] - cur["bbox"][0]) < 60 and 0 <= cur["bbox"][1] - pb["bbox"][3] < 30
            if not (mid_line or continues) or (pb.get("type") == "heading" and not continues):
                break
            lead_blocks.insert(0, pb); k -= 1
            if not re.match(r"^[a-z(,;]", ptxt):
                break
        if lead_blocks:
            lead = " ".join(_unwrap(re.sub(r"^\s*\\\(-\\\)\s*", "", b.get("content") or "")) for b in lead_blocks)
            used[:0] = lead_blocks
            parts[0] = dict(parts[0], text=lead + " " + parts[0]["text"].lstrip("… "), joined=[b["block_id"] for b in lead_blocks] + [b0["block_id"]])
        elif len(plain(parts[0]["text"]).split()) <= 6 and b0.get("bbox"):
            row = [b for b in ix.blocks if b.get("page") == b0.get("page") and b is not b0 and b.get("bbox") and b.get("type") in ("paragraph", "list")
                   and b["bbox"][0] > b0["bbox"][2] and _row_overlap(b0["bbox"], b["bbox"])]
            row.sort(key=lambda b: b["bbox"][0])
            if row:
                used.extend(row[:2])
                parts[0] = dict(parts[0], text="**" + parts[0]["text"] + "** — " + " ".join(_unwrap(b.get("content") or "") for b in row[:2]), markdown=True)
    if ix.section_of[i0] is not None and b0.get("type") != "heading":
        res["section"] = ix.blocks[ix.section_of[i0]].get("content")
    # a display equation right after a passage that introduces it ('... is given by')
    if b0.get("type") in ("paragraph", "list") and i0 + 1 < ix.N and ix.blocks[i0 + 1].get("type") == "equation" \
            and re.search(r"(:|given by|as follows|defined as|is|=)\s*$", plain(b0.get("content") or "")):
        used.append(ix.blocks[i0 + 1]); parts.append(part_for(ix.blocks[i0 + 1]))
    # other passages that use the same phrase (e.g. two places that define the term)
    if ph0:
        for sc, i, cov, mt, ph in scored[1:6]:
            if len(used) >= 3:
                break
            b = ix.blocks[i]
            if ph and b.get("type") != "heading" and sc >= 0.6 * sc0 and cov >= cov0 - 0.01 and b["block_id"] not in [u["block_id"] for u in used]:
                used.append(b)
                parts.append(part_for(b, best_excerpt(b.get("content") or "", q, phrase_res) if b.get("type") in ("paragraph", "caption", "list") and b.get("content") else None))
    return finish("passage", parts, used, cov0, mt0 | (set(ix.ctx_toks[i0]) & set(q)))


def _row_overlap(a, b):
    """b's first line sits on a's row (a term and its description in a two-column layout)."""
    h = a[3] - a[1]
    return b[1] <= a[1] + 0.8 * h and b[1] >= a[1] - 0.8 * h - 6


def _same_line(a, b):
    """b starts on the same text line as a ends, to its right."""
    h = min(a[3] - a[1], b[3] - b[1]) or 1
    lasttop = a[3] - min(h, b[3] - b[1])
    ov = min(a[3], b[3]) - max(lasttop, b[1])
    return ov > 0.5 * (b[3] - b[1]) and b[0] >= a[0] and b[1] >= a[1] - 2


def _sub(ix, page):
    class _V:
        pass
    v = _V()
    v.blocks = [b for b in ix.blocks if b.get("page") == page]
    v.by_id = ix.by_id
    v.idf = ix.idf
    v.df = ix.df
    return v
