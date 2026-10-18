"""Layout analysis on text-line segments: column-aware reading order, paragraph assembly and block typing.

Works on any objects with a .bbox (x0, y0, x1, y1) in a common coordinate space; text items additionally
carry .text/.size/.baseline. Used for native-text pages and for OCR pages (OCR lines are wrapped as Segments),
so both paths share the same reading-order logic.
"""
import re
import statistics
from dataclasses import dataclass, field

BULLET = re.compile(r"^\s*(?:[•‣●○▪■–—∙·➢⁃*•◦▪-]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\(?[ivxIVX]{1,4}[.)])\s+\S")
CAPTION = re.compile(r"^\s*(?i:fig(?:ure)?|table|tab|chart|plate|exhibit|scheme|algorithm|listing)\.?\s*(?:\d+[A-Za-z]?|[IVX]+|[A-Z])(?:\.\d+)*[.:\-\u2013\u2014 ]")
HYPHEN_KEEP_PREFIX = {"self", "non", "well", "ill", "ex", "all", "half", "cross", "high", "low", "long", "short", "full", "semi",
                      "quasi", "co", "inter", "intra", "multi", "post", "pre", "pro", "anti", "counter", "over", "under", "state", "real",
                      "time", "user", "data", "end", "open", "close", "first", "second", "third", "one", "two", "three", "four", "five"}


ENUM_ONLY = re.compile(r"^\s*\(?\d{1,3}[.):]\s*$")
_CODE_OPS = [r"->", r"==|!=", r"<=|>=", r"&&|\|\|", r"\+\+|--(?=\w)|(?<=\w)--", r"::", r";\s*$", r"[{}]", r"\w\[[^\]]*\]"]
_CODE_KEYWORDS = re.compile(r"^\s*(?:\d{1,3}[.):]\s*)?(?:if|else|elif|else if|for|foreach|while|do|return|break|continue|switch|case|def|class|"
                            r"int|void|float|double|char|struct|public|private|static|import|from|print|printf|cout|let|var|const|function|"
                            r"begin|end|then|repeat|until|procedure|algorithm|input|output|null|true|false|endif|endwhile|endfor)\b", re.I)


def code_score(line):
    """How code-like one line of text is (0..1). Pseudo-code ('If (root == NULL)', 'root -> left = f(x)') scores high,
    prose ('the root of the tree is ...') low. Used to keep code out of the inline-math converter and out of tables."""
    t = (line or "").strip()
    t = re.sub(r"^\(?\d{1,3}[.):]\s+", "", t)          # a leading line number says nothing either way
    if not t:
        return 0.0
    s = 0.0
    ops = sum(1 for o in _CODE_OPS if re.search(o, t))
    s += min(0.6, 0.3 * ops)
    if re.match(r"^(//|/\*|#\s|--\s)", t):
        s += 0.45                                     # a comment line
    if _CODE_KEYWORDS.match(t):
        s += 0.35
    if re.search(r"\b[A-Za-z_]\w*\(", t):
        s += 0.15                                     # call / function application
    if re.search(r"\b[a-z]+_[a-z_]+\b|\b[a-z]+[A-Z]\w*\(|\b[A-Z][a-z]+[A-Z]\w*\(", t):   # snake_case / camelCase calls
        s += 0.3
    if re.search(r"\b\w+\s*=\s*[\w(]", t) and not re.search(r"[.!?]\s*$", t):
        s += 0.15
    hard = re.search(r"->|==|!=|&&|\|\||;\s*$|[{}]|//", t)
    if len(re.findall(r"\b[A-Za-z]{2,}\b", t)) > 8 and not hard:
        s -= 0.5                                     # a sentence
    elif re.search(r"[a-z]{2,}[.!?]\s*$", t) or (re.search(r"[.!?]\s*$", t) and len(re.findall(r"\b[a-z]{2,}\b", t)) >= 3 and not hard):
        s -= 0.4                                     # ends like a sentence
    return max(0.0, min(1.0, s))


def is_code_lines(lines):
    """True when (nearly) all lines are code-like."""
    ls = [l for l in lines if l and l.strip() and not ENUM_ONLY.match(l)]
    if not ls:
        return False
    sc = [code_score(l) for l in ls]
    return all(v >= 0.3 for v in sc) and sum(1 for v in sc if v >= 0.45) >= max(1, 0.5 * len(ls))


@dataclass
class Region:
    """A non-text item placed in reading order (table, figure, display equation)."""
    kind: str
    bbox: tuple
    payload: object = None
    text: str = ""
    size: float = 0.0

    @property
    def baseline(self):
        return self.bbox[3]


def _w(b):
    return b[2] - b[0]


def _h(b):
    return b[3] - b[1]


def _is_text(it):
    return hasattr(it, "chars") or getattr(it, "kind", None) == "text"


def _len(it):
    return len(getattr(it, "text", "") or "") if _is_text(it) else 80


def _gaps(intervals, min_gap):
    """Merged-interval gaps: returns list of (gap_start, gap_end)."""
    iv = sorted(intervals)
    out, cur_end = [], None
    for a, b in iv:
        if cur_end is None:
            cur_end = b
            continue
        if a - cur_end >= min_gap:
            out.append((cur_end, a))
        cur_end = max(cur_end, b)
    return out


def reading_order(items, body_size=10.0, _depth=0):
    """Recursive XY-cut with column preference.

    1. Look for vertical gutters among the 'narrow' items (items narrower than 55% of the region).
       Items that cross a gutter are separators: the region is split into horizontal bands at them.
    2. Inside a band with a clean gutter, read columns left to right when they look like text columns
       (or when no horizontal cut is available); otherwise cut horizontally into rows.
    3. Fall back to top-to-bottom, left-to-right.
    Returns (ordered_items, multi_column_flag).
    """
    items = list(items)
    if len(items) <= 1 or _depth > 400:
        return sorted(items, key=lambda i: (i.bbox[1], i.bbox[0])), False
    x0 = min(i.bbox[0] for i in items); x1 = max(i.bbox[2] for i in items)
    W = max(1.0, x1 - x0)
    min_gutter = max(4.0, 0.55 * body_size)
    # a bare list number ('12.') belongs to the text after it: it must not create a 'gutter' of its own
    narrow = [i for i in items if _w(i.bbox) < 0.8 * W and not (_is_text(i) and ENUM_ONLY.match(i.text or ""))]
    gutters = []
    if len(narrow) >= 2:
        for a, b in _gaps([(i.bbox[0], i.bbox[2]) for i in narrow], min_gutter):
            left = [i for i in narrow if i.bbox[2] <= a + 0.5]
            right = [i for i in narrow if i.bbox[0] >= b - 0.5]
            if left and right:
                gutters.append((a, b))
    if gutters:
        def crosses(i):
            return any(i.bbox[0] < a - 0.5 and i.bbox[2] > b + 0.5 for a, b in gutters)
        seps = [i for i in items if crosses(i)]
        if seps:
            return _split_bands_at(items, seps, body_size, _depth)
        cols = _split_columns(items, gutters)
        if len(cols) > 1:
            textlike = all(_column_is_text(c, body_size) for c in cols)
            if not textlike and any(_column_is_text(c, body_size) for c in cols) and all(len(c) >= 2 for c in cols):
                # sidebar beside a text column (not a label/value list: labels would sit on the text's baselines)
                textlike = _row_alignment(cols, body_size) < 0.6
            ygaps = _gaps([(i.bbox[1], i.bbox[3]) for i in items], 0.15 * body_size)
            if textlike or not ygaps:
                out, multi = [], textlike
                for c in cols:
                    o, m = reading_order(c, body_size, _depth + 1)
                    out += o; multi = multi or m
                return out, multi
    # No usable gutter here. If nothing in this region sits side by side, plain top-to-bottom is the order.
    side_by_side = _any_side_by_side(items, min_gutter)
    ygaps = _gaps([(i.bbox[1], i.bbox[3]) for i in items], 0.15 * body_size)
    if not side_by_side or not ygaps:
        return sorted(items, key=lambda i: (round(getattr(i, "baseline", i.bbox[3]) / max(1.0, 0.5 * body_size)), i.bbox[0])), False
    # Otherwise cut at the WIDEST horizontal gap only and re-examine both halves (a gutter that was hidden by a
    # full-width element - a running header, a title - becomes visible once that element is cut away).
    a, b = max(ygaps, key=lambda g: g[1] - g[0])
    cut = (a + b) / 2
    top = [i for i in items if (i.bbox[1] + i.bbox[3]) / 2 < cut]
    bot = [i for i in items if (i.bbox[1] + i.bbox[3]) / 2 >= cut]
    if not top or not bot:
        return sorted(items, key=lambda i: (i.bbox[1], i.bbox[0])), False
    o1, m1 = reading_order(top, body_size, _depth + 1)
    o2, m2 = reading_order(bot, body_size, _depth + 1)
    return o1 + o2, m1 or m2


def _any_side_by_side(items, min_gap):
    """True if two items overlap vertically while being horizontally separated (a sweep over y)."""
    srt = sorted(items, key=lambda i: i.bbox[1])
    active = []
    for it in srt:
        y0, y1 = it.bbox[1], it.bbox[3]
        active = [a for a in active if a.bbox[3] > y0]
        for a in active:
            ov = min(a.bbox[3], y1) - max(a.bbox[1], y0)
            if ov > 0.3 * min(_h(a.bbox), _h(it.bbox)) and (it.bbox[0] - a.bbox[2] >= min_gap or a.bbox[0] - it.bbox[2] >= min_gap):
                return True
        active.append(it)
    return False


def _bands(items, min_gap):
    srt = sorted(items, key=lambda i: i.bbox[1])
    bands, cur, cur_end = [], [], None
    for i in srt:
        if cur and i.bbox[1] - cur_end >= min_gap:
            bands.append(cur); cur, cur_end = [], None
        cur.append(i)
        cur_end = i.bbox[3] if cur_end is None else max(cur_end, i.bbox[3])
    if cur:
        bands.append(cur)
    return bands


def _split_bands_at(items, seps, body_size, depth):
    seps = sorted(seps, key=lambda i: i.bbox[1])
    rest = [i for i in items if i not in seps]
    out, multi = [], False
    prev_y = float("-inf")
    for s in seps:
        top = s.bbox[1]
        above = [i for i in rest if prev_y - 1 <= (i.bbox[1] + i.bbox[3]) / 2 < top + 0.5 * _h(s.bbox)]
        rest = [i for i in rest if i not in above]
        if above:
            o, m = reading_order(above, body_size, depth + 1)
            out += o; multi = multi or m
        out.append(s)
        prev_y = s.bbox[3]
    if rest:
        o, m = reading_order(rest, body_size, depth + 1)
        out += o; multi = multi or m
    return out, multi


def _split_columns(items, gutters):
    edges = [g[0] for g in gutters]
    cols = [[] for _ in range(len(edges) + 1)]
    for i in items:
        cx = (i.bbox[0] + i.bbox[2]) / 2
        k = sum(1 for e in edges if cx > e)
        cols[k].append(i)
    return [c for c in cols if c]


def _row_alignment(cols, body_size):
    """Fraction of items in the non-text columns whose baseline coincides with an item in a text column."""
    text_cols = [c for c in cols if _column_is_text(c, body_size)]
    other = [i for c in cols if c not in text_cols for i in c]
    if not other:
        return 0.0
    bases = [getattr(i, "baseline", i.bbox[3]) for c in text_cols for i in c]
    hit = sum(1 for i in other if any(abs(getattr(i, "baseline", i.bbox[3]) - b) < 0.3 * body_size for b in bases))
    return hit / len(other)


def _column_is_text(col, body_size):
    if any(not _is_text(i) for i in col):
        return True
    col = [i for i in col if not ENUM_ONLY.match(i.text or "")] or col
    lens = [_len(i) for i in col]
    widths = [_w(i.bbox) for i in col]
    return len(col) >= 2 and statistics.median(lens) >= 18 and statistics.median(widths) >= 8 * body_size


# ---------------------------------------------------------------------------------------------
# Paragraph assembly
# ---------------------------------------------------------------------------------------------
@dataclass
class Para:
    lines: list = field(default_factory=list)   # Segments
    kind: str = "paragraph"
    code: bool = False                           # set by merge_code: consecutive pseudo-code lines

    def visual_lines(self):
        """Segments grouped into visual lines (top to bottom), each sorted left to right."""
        out = []
        for l in sorted(self.lines, key=lambda l: (l.baseline, l.bbox[0])):
            if out and abs(out[-1][0].baseline - l.baseline) < 0.35 * max(l.size, out[-1][0].size, 0.1):
                out[-1].append(l)
            else:
                out.append([l])
        return [sorted(v, key=lambda l: l.bbox[0]) for v in out]

    @property
    def bbox(self):
        return (min(l.bbox[0] for l in self.lines), min(l.bbox[1] for l in self.lines),
                max(l.bbox[2] for l in self.lines), max(l.bbox[3] for l in self.lines))

    @property
    def size(self):
        return statistics.median([l.size for l in self.lines]) if self.lines else 0

    @property
    def bold(self):
        return sum(1 for l in self.lines if l.bold) >= 0.6 * len(self.lines)

    @property
    def mono(self):
        return sum(1 for l in self.lines if l.mono) >= 0.6 * len(self.lines)


def assemble_paragraphs(ordered, body_size):
    """Walk text segments in reading order and merge consecutive lines of the same paragraph.
    Non-text Regions pass through unchanged. Returns list of Para | Region."""
    out, cur = [], None

    def flush():
        nonlocal cur
        if cur and cur.lines:
            out.append(cur)
        cur = None

    for it in ordered:
        if not _is_text(it):
            flush(); out.append(it); continue
        if cur is None:
            cur = Para([it]); continue
        last = cur.lines[-1]
        if _continues(cur, last, it, body_size):
            cur.lines.append(it)
        else:
            flush(); cur = Para([it])
    flush()
    return merge_code(out, body_size)


def merge_code(items, body_size):
    """Merge runs of consecutive code-like paragraphs (numbered pseudo-code lines arrive as one 'list item' each)
    into a single code paragraph, so an algorithm stays one block in reading order. Needs >= 3 lines."""
    def texts(p):
        return [" ".join(s.text for s in v) for v in p.visual_lines()]

    def codey(p):          # every line at least weakly code-like
        if not isinstance(p, Para) or len(p.lines) > 60:
            return False
        ts = [t for t in texts(p) if not ENUM_ONLY.match(t)]
        return bool(ts) and all(code_score(t) >= 0.3 for t in ts)
    out, i = [], 0
    while i < len(items):
        p = items[i]
        if not codey(p):
            out.append(p); i += 1; continue
        run = [p]
        j = i + 1
        while j < len(items) and codey(items[j]):
            a, b = run[-1].bbox, items[j].bbox
            sz = max(run[-1].size, items[j].size, 0.1)
            gap = b[1] - a[3]
            if gap > 1.6 * sz or gap < -0.6 * sz or abs(items[j].size - run[-1].size) > 0.18 * sz:
                break
            if min(a[2], b[2]) - max(a[0], b[0]) < -2 * sz and abs(b[0] - a[0]) > 6 * sz:
                break                                  # another column
            run.append(items[j]); j += 1
        lines_ = [t for r in run for t in texts(r)]
        if len(lines_) >= 3 and is_code_lines(lines_):
            out.append(Para([l for r in run for l in r.lines], kind="code", code=True))
        else:
            out.extend(run)
        i = j
    return out


def code_text(para):
    """Code paragraph text with the source indentation kept (leading spaces, 4 per indent step);
    a leading line number stays at the start of its line."""
    rows = []
    for v in para.visual_lines():
        num, rest = "", v
        if len(v) > 1 and (ENUM_ONLY.match(v[0].text) or re.fullmatch(r"\s*\d{1,3}\s*", v[0].text)):
            num, rest = v[0].text.strip(), v[1:]
        x = rest[0].bbox[0]
        m = re.match(r"^\s*(\d{1,3}[.):]?)\s+(\S.*)$", rest[0].text) if not num else None
        text = " ".join(s.text.strip() for s in rest)
        if m:
            num, text = m.group(1), " ".join([m.group(2)] + [s.text.strip() for s in rest[1:]])
            # the indent of an inline-numbered line is where its text starts: estimate from the number's width
            x = rest[0].bbox[0] + (len(m.group(0)) - len(m.group(2))) * 0.5 * rest[0].size
            chars = getattr(rest[0], "chars", None) or []
            seen_num = False
            for ch in chars:              # exact: where the first character after the number is drawn
                if ch.c.strip() and not seen_num:
                    seen_num = True
                elif seen_num and ch.c.strip() and not (ch.c.isdigit() or ch.c in ".):"):
                    x = ch.x0
                    break
        rows.append((num, x, text, max(s.size for s in v)))
    xs = sorted({round(r[1], 1) for r in rows if r[2].strip()})
    base = xs[0] if xs else 0
    steps = sorted(d for d in {round(x - base, 1) for x in xs} if d > 1.5)
    step = steps[0] if steps else 1
    out = []
    numw = max((len(r[0]) for r in rows), default=0)
    for num, x, text, sz in rows:
        lvl = int(round((x - base) / step)) if step > 1 else 0
        pre = (num.ljust(numw) + " ") if numw else ""
        out.append((pre + "    " * max(0, lvl) + text).rstrip())
    return "\n".join(out)


def _continues(para, last, seg, body):
    sz = max(seg.size, 0.1); lsz = max(last.size, 0.1)
    same_line = abs(seg.baseline - last.baseline) < 0.35 * max(sz, lsz) and seg.bbox[0] >= last.bbox[2] - 0.5 * sz
    if same_line:
        reach = 8.0 if ENUM_ONLY.match(last.text or "") else 1.5     # '12.'  <hanging indent>  text
        return seg.bbox[0] - last.bbox[2] < reach * max(sz, lsz)
    small_attached = sz < 0.85 * lsz and seg.bbox[1] < last.bbox[3] and seg.bbox[3] > last.bbox[1]
    if small_attached:     # sub/superscript run hanging off the previous line
        return True
    if abs(sz - lsz) > 0.18 * max(sz, lsz):
        return False
    if (BULLET.match(seg.text) and len(seg.text) > 2) or ENUM_ONLY.match(seg.text or ""):
        return False
    first = para.lines[0]
    if BULLET.match(first.text or "") and getattr(first, "chars", None):
        # a list item's wrapped lines hang under its text; a line starting left of that text is the next paragraph
        cs = [c for c in first.chars if c.c.strip()]
        m = BULLET.match(first.text)
        k = len(re.sub(r"\s+", "", first.text[:m.end() - 1]))
        if len(cs) > k and seg.bbox[0] < cs[k].x0 - 0.6 * sz:
            return False
    if last.bold and not seg.bold and all(l.bold for l in para.lines):
        return False                                   # bold heading line(s) followed by body text
    if seg.bold and not last.bold and len(seg.text) < 120:
        return False                                   # body text followed by a bold heading
    if last.mono != seg.mono:
        return False
    if _caps_title(last.text) and not _caps_title(seg.text):
        return False                                   # ALL-CAPS section title above body text
    gap = seg.bbox[1] - last.bbox[3]
    lead = 0.9 * max(sz, lsz)
    if gap > lead or gap < -0.6 * max(sz, lsz):
        return False
    pb = para.bbox
    ovl = min(pb[2], seg.bbox[2]) - max(pb[0], seg.bbox[0])
    if ovl < 0.25 * min(_w(pb), _w(seg.bbox)) and abs(seg.bbox[0] - pb[0]) > 2 * sz:
        return False
    # first-line indent of a new paragraph (only when the previous line ended a sentence)
    if seg.bbox[0] - pb[0] > 1.2 * sz and re.search(r"[.!?:\"'”)]\s*$", last.text) and len(para.lines) >= 1:
        return False
    return True


# ---------------------------------------------------------------------------------------------
# Text joining with evidence-based dehyphenation
# ---------------------------------------------------------------------------------------------
def normalize_quotes(t):
    """TeX-style double quotes typeset as two single quotes -> proper double quotes."""
    t = t.replace("\u2018\u2018", "\u201c").replace("\u2019\u2019", "\u201d").replace("``", "\u201c")
    return t


def join_lines(lines, vocab=None):
    """Join line texts into a paragraph. A trailing hyphen is removed only when the evidence supports it:
    both fragments are lowercase letters, and either the joined word occurs elsewhere in the document
    vocabulary or the hyphenated form does not, and the prefix is not a common compound-forming prefix.
    Returns (text, transforms)."""
    vocab = vocab or set()
    parts, transforms = [], []
    for k, t in enumerate(lines):
        t = t.strip()
        if not parts:
            parts.append(t); continue
        prev = parts[-1]
        m = re.search(r"([^\W\d_]+)-$", prev)
        n = re.match(r"([^\W\d_]+)", t)
        if m and n and m.group(1)[-1].islower() and n.group(1)[0].islower():
            a, b = m.group(1), n.group(1)
            joined, hyph = (a + b).lower(), (a + "-" + b).lower()
            if hyph in vocab and joined not in vocab:
                parts[-1] = prev + t; transforms.append(f"kept hyphen '{a}-{b}' (hyphenated form used elsewhere)")
            elif joined in vocab or (a.lower() not in HYPHEN_KEEP_PREFIX and len(a) >= 2):
                parts[-1] = prev[:-1] + t; transforms.append(f"rejoined '{a}-' + '{b}'")
            else:
                parts[-1] = prev + t; transforms.append(f"kept hyphen '{a}-{b}' (compound prefix)")
        else:
            parts[-1] = prev + " " + t
    return normalize_quotes(re.sub(r"[ \t]+", " ", parts[0] if parts else "").strip()), transforms


def doc_vocab(texts):
    v = set()
    for t in texts:
        for w in re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", t):
            lw = w.lower()
            v.add(lw)
    return v


def body_like_sizes(paras):
    """Font sizes that carry >= 15% of the page's text: never headings by size alone."""
    tot = {}
    for p in paras:
        k = round(p.size, 1)
        tot[k] = tot.get(k, 0) + sum(len(l.text) for l in p.lines)
    allc = sum(tot.values()) or 1
    return {k for k, v in tot.items() if v >= 0.15 * allc}


def _caps_title(t):
    t = t.strip()
    letters = [c for c in t if c.isalpha()]
    return 4 <= len(letters) and len(t.split()) <= 10 and all(not c.islower() for c in letters) and not re.search(r"[.,;]$", t)


def classify(para, body_size, heading_sizes, body_sizes=()):
    """Return (type, level) for an assembled paragraph."""
    text = " ".join(l.text for l in para.lines).strip()
    n = len(para.lines)
    if (para.mono and n >= 2) or getattr(para, "code", False):
        return "code", 0
    if CAPTION.match(text) and len(text) < 400:
        return "caption", 0
    if all(BULLET.match(l.text) for l in para.lines[:1]) and len(text) > 2:
        return "list", 0
    sz = para.size
    if n <= 3 and len(text) < 200 and not text.endswith((",", ";")):
        if sz >= 1.15 * body_size and round(sz, 1) not in body_sizes and re.search(r"[^\W\d_]", text):
            lvl = 1 + sum(1 for s in heading_sizes if s > sz * 1.03)
            return "heading", min(lvl, 4)
        if para.bold and n <= 2 and len(text) < 120 and not text.endswith(".") and re.search(r"[^\W\d_]{2}", text):
            return "heading", min(len(heading_sizes) + 1, 4)
        if n <= 2 and _caps_title(text) and sz >= 0.95 * body_size and re.search(r"[^\W\d_]{3}", text):
            return "heading", min(len(heading_sizes) + 2, 4)
    return "paragraph", 0


def heading_size_levels(paras, body_size):
    bl = body_like_sizes(paras)
    sizes = sorted({round(p.size, 1) for p in paras if p.size >= 1.15 * body_size and len(p.lines) <= 3 and round(p.size, 1) not in bl}, reverse=True)
    merged = []
    for s in sizes:
        if not merged or merged[-1] - s > 0.06 * merged[-1]:
            merged.append(s)
    return merged[:3]
