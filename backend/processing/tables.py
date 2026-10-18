"""Table detection and structure reconstruction (local, deterministic).

Two detectors feed one structured model:

* ruled  - PyMuPDF ``find_tables()`` (vector ruling lines). Cell geometry is mapped onto the grid of row/column
           edges so merged cells get real rowspan/colspan instead of being flattened.
* aligned - borderless / booktabs tables found from text geometry: consecutive text rows that split into
           several horizontally-aligned fragments whose column positions agree. Multi-column prose (two long
           fragments per row) and code are rejected. Horizontal rules, when present, mark the header block.

Every table is a ``Table`` with cell bboxes, row/col indices, spans, header flags, method, confidence and
warnings. Text is never invented: an empty cell is "" (``filled=False``); a cell the detector could not read
is reported in warnings.

Output: a GitHub-style Markdown table when the grid is rectangular with a single header row, otherwise an
HTML table with <th>, rowspan and colspan.
"""
import html
import re
import statistics
from dataclasses import dataclass, field

from .schema import overlap_frac

NUMERIC = re.compile(r"^[\s(\[]*[-+−±~<>≈]?\s*[$€£¥]?\s*\d[\d,.\s]*%?[)\]]*\s*[a-zA-Z%]{0,3}\*{0,3}$")


@dataclass
class Cell:
    row: int
    col: int
    text: str
    bbox: tuple
    rowspan: int = 1
    colspan: int = 1
    header: bool = False


@dataclass
class Table:
    bbox: tuple
    n_rows: int
    n_cols: int
    cells: list
    method: str
    confidence: float
    header_rows: int = 1
    warnings: list = field(default_factory=list)

    @property
    def has_spans(self):
        return any(c.rowspan > 1 or c.colspan > 1 for c in self.cells)

    def grid(self, fill_spans=False):
        g = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            for r in range(c.row, min(self.n_rows, c.row + c.rowspan)):
                for k in range(c.col, min(self.n_cols, c.col + c.colspan)):
                    if (r, k) == (c.row, c.col) or fill_spans:
                        g[r][k] = c.text
        return g

    def rows(self):
        """v1-compatible list-of-lists (span text repeated in covered positions is NOT done: blanks stay blank)."""
        return self.grid(False)

    def to_dict(self):
        return dict(n_rows=self.n_rows, n_cols=self.n_cols, header_rows=self.header_rows, method=self.method,
                    cells=[dict(row=c.row, col=c.col, rowspan=c.rowspan, colspan=c.colspan, text=c.text,
                                bbox=[round(v, 1) for v in c.bbox], header=c.header, filled=bool(c.text)) for c in self.cells])


# ---------------------------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------------------------
def md_cell(t):
    t = re.sub(r"\s*\n\s*", " ", t or "").strip()
    return t.replace("\\", "\\\\").replace("|", "\\|")


def to_markdown(tb):
    g = tb.grid(False)
    if not g:
        return ""
    lines = ["| " + " | ".join(md_cell(c) for c in g[0]) + " |", "|" + "|".join(["---"] * tb.n_cols) + "|"]
    for r in g[1:]:
        lines.append("| " + " | ".join(md_cell(c) for c in r) + " |")
    return "\n".join(lines)


def to_html(tb):
    by_row = {}
    for c in tb.cells:
        by_row.setdefault(c.row, []).append(c)
    out = ["<table>"]
    hdr = [r for r in range(tb.n_rows) if r < tb.header_rows]
    if hdr:
        out.append("<thead>")
    for r in range(tb.n_rows):
        if r == tb.header_rows and hdr:
            out.append("</thead><tbody>")
        elif r == 0 and not hdr:
            out.append("<tbody>")
        cells = sorted(by_row.get(r, []), key=lambda c: c.col)
        tds = []
        for c in cells:
            tag = "th" if c.header else "td"
            attrs = (f' rowspan="{c.rowspan}"' if c.rowspan > 1 else "") + (f' colspan="{c.colspan}"' if c.colspan > 1 else "")
            txt = html.escape(re.sub(r"\s*\n\s*", " ", c.text or "").strip(), quote=False)
            tds.append(f"<{tag}{attrs}>{txt}</{tag}>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def serialize(tb):
    if tb.has_spans or tb.header_rows != 1:
        return to_html(tb)
    return to_markdown(tb)


def from_rows(rows, method="model", confidence=0.5, header_rows=1):
    """Build a Table from a plain list-of-lists (e.g. a vision model's rows). No geometry, no spans."""
    rows = [list(r) for r in rows if r]
    nc = max((len(r) for r in rows), default=0)
    cells = []
    for i, r in enumerate(rows):
        for j in range(nc):
            cells.append(Cell(i, j, str(r[j]).strip() if j < len(r) and r[j] is not None else "", (0, 0, 0, 0), header=i < header_rows))
    return Table((0, 0, 0, 0), len(rows), nc, cells, method, confidence, header_rows)


# ---------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------
def rules_from_drawings(drawings, page_w, page_h):
    """Horizontal and vertical ruling segments from page.get_drawings(): list of (x0, y0, x1, y1)."""
    hs, vs = [], []
    for d in drawings or []:
        for it in d.get("items", []):
            op = it[0]
            if op == "l":
                p, q = it[1], it[2]
                if abs(p.y - q.y) < 1.0 and abs(p.x - q.x) > 6:
                    hs.append((min(p.x, q.x), p.y, max(p.x, q.x), p.y))
                elif abs(p.x - q.x) < 1.0 and abs(p.y - q.y) > 6:
                    vs.append((p.x, min(p.y, q.y), p.x, max(p.y, q.y)))
            elif op == "re":
                r = it[1]
                w, h = r.x1 - r.x0, r.y1 - r.y0
                if h <= 2.0 and w > 6:
                    hs.append((r.x0, (r.y0 + r.y1) / 2, r.x1, (r.y0 + r.y1) / 2))
                elif w <= 2.0 and h > 6:
                    vs.append(((r.x0 + r.x1) / 2, r.y0, (r.x0 + r.x1) / 2, r.y1))
    big = 0.98
    hs = [h for h in hs if (h[2] - h[0]) < big * page_w]
    return hs, vs


def fragments(seg, gap_em=0.75):
    """Split a text segment into cell fragments at wide gaps. Returns list of (x0, x1, text, chars)."""
    from .textlayer import _chars_to_text
    chars = [c for c in seg.chars]
    out, cur, last = [], [], None
    for c in chars:
        if c.c.isspace():
            if cur:
                cur.append(c)
            continue
        if last is not None and c.x0 - last.x1 > gap_em * max(c.size, last.size, 1):
            out.append(cur); cur = []
        cur.append(c); last = c
    if cur:
        out.append(cur)
    res = []
    for f in out:
        vis = [c for c in f if not c.c.isspace()]
        if vis:
            res.append((min(c.x0 for c in vis), max(c.x1 for c in vis), _chars_to_text(f), vis))
    return res


@dataclass
class _Row:
    y0: float
    y1: float
    base: float
    frags: list      # (x0, x1, text, chars)
    segs: list


def _rows(segs):
    """Group horizontal segments into visual rows by baseline."""
    rows = []
    for s in sorted([s for s in segs if s.horizontal], key=lambda s: s.baseline):
        sz = max(s.size, 1)
        if rows and abs(rows[-1].base - s.baseline) < 0.4 * sz:
            r = rows[-1]
            r.y0, r.y1 = min(r.y0, s.bbox[1]), max(r.y1, s.bbox[3])
            r.frags += fragments(s); r.segs.append(s)
        else:
            rows.append(_Row(s.bbox[1], s.bbox[3], s.baseline, fragments(s), [s]))
    for r in rows:
        # leader dots / stray bullets between cells are layout, not cells
        r.frags = [f for f in r.frags if re.search(r"[\w%$€£¥]", f[2])]
        r.frags.sort(key=lambda f: f[0])
    return [r for r in rows if r.frags]


# ---------------------------------------------------------------------------------------------
# Aligned (borderless / booktabs) detector
# ---------------------------------------------------------------------------------------------
def detect_aligned(segs, hrules, body_size, page_w, exclude=()):
    segs = [s for s in segs if not any(overlap_frac(s.bbox, e) > 0.5 for e in exclude)]
    rows = _rows(segs)
    tables, used = [], set()
    i = 0
    while i < len(rows):
        if len(rows[i].frags) < 2 or _proselike(rows[i], body_size):
            i += 1; continue
        j = i
        multi = 1
        sizes0 = [statistics.median([s.size for s in rows[i].segs])]
        while j + 1 < len(rows):
            nxt, cur = rows[j + 1], rows[j]
            lh = max(1.0, cur.y1 - cur.y0)
            gap = nxt.y0 - cur.y1
            if gap > max(2.2 * lh, 2.0 * body_size) and not (_rule_between(hrules, cur.y1, nxt.y0) and gap < 4.5 * lh):
                break
            nsz = statistics.median([s.size for s in nxt.segs])
            if nsz > 1.2 * statistics.median(sizes0) or nsz < 0.7 * statistics.median(sizes0):
                break           # a heading or footnote in another font size ends the table
            id_row = multi >= 3 and len(nxt.frags) >= 2 and len(nxt.frags[0][2]) <= 4 and \
                any(abs(nxt.frags[0][0] - r.frags[0][0]) <= 3 or abs(nxt.frags[0][1] - r.frags[0][1]) <= 3 for r in rows[i:j + 1] if len(r.frags) >= 2)
            if _proselike(nxt, body_size) and not id_row:
                # a category row ('Translation  Rosetta-stone style ...') is allowed when the table continues after it
                after = rows[j + 2] if j + 2 < len(rows) else None
                if not (after and len(after.frags) >= 3 and not _proselike(after, body_size) and after.y0 - nxt.y1 < 2.2 * lh and multi >= 2):
                    break
            sizes0.append(nsz)
            if len(nxt.frags) >= 2:
                multi += 1
            elif not _short(nxt, body_size) and not _continuation(nxt, rows[i:j + 1], body_size):
                break
            j += 1
        block = rows[i:j + 1]
        # trim single-fragment rows at the bottom
        while len(block) > 1 and len(block[-1].frags) < 2:
            block.pop()
        tb = _build_aligned(block, hrules, body_size) if multi >= 3 and len(block) >= 3 and not _listing(block) else None
        if tb:
            tables.append(tb)
            for r in block:
                for s in r.segs:
                    used.add(id(s))
            i = j + 1
        else:
            i += 1
    return tables, used


LEAD_NUM = re.compile(r"^\s*\(?(\d{1,3})[.):]?\s+\S")


def _id_column_table(rows):
    """'2 | isMovable | Can it be moved?': a number column followed by at least two text columns that start at the
    same x on every row (code would be indented) and do not read as code."""
    from .layout import code_score
    fr = [sorted(r.frags, key=lambda f: f[0]) for r in rows]
    multi = [f for f in fr if len(f) >= 3]
    if len(multi) < 0.7 * len(fr) or len(multi) < 3:
        return False

    def same(xs):
        return max(sum(1 for y in xs if abs(y - x) <= 3) for x in xs) >= 0.85 * len(xs)
    texts = [g[2] for f in fr for g in f[1:]]
    code = sum(1 for t in texts if code_score(t) >= 0.45)
    return same([f[1][0] for f in multi]) and same([f[2][0] for f in multi]) and code < 0.3 * max(1, len(texts))


def _listing(block):
    """Numbered lines laid out in one or more columns (an algorithm, a numbered list, pseudo-code) align like a table
    but are not one: most rows carry a bare enumerator fragment ('12.') and the numbers run consecutively, or the
    row text is code. Such blocks are left to the text/list/code path."""
    from .layout import ENUM_ONLY, code_score
    rows = [r for r in block if r.frags]
    if len(rows) < 3:
        return False
    nums, enum_rows = [], 0
    for r in rows:
        e = [f for f in r.frags if ENUM_ONLY.match(f[2])]
        if e:
            enum_rows += 1
            nums.append(int(re.sub(r"\D", "", e[0][2]) or 0))
            continue
        m = LEAD_NUM.match(sorted(r.frags, key=lambda f: f[0])[0][2])     # '1 for i = 2 to n' (bare line number)
        if m:
            enum_rows += 1
            nums.append(int(m.group(1)))
    if enum_rows >= 0.6 * len(rows):
        steps = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)
        if steps >= 0.6 * max(1, len(nums) - 1) and not _id_column_table(rows):
            return True
    texts = [f[2] for r in rows for f in r.frags if not ENUM_ONLY.match(f[2])]
    code = sum(1 for t in texts if code_score(t) >= 0.45)
    return len(texts) >= 4 and code >= 0.6 * len(texts)


def _proselike(row, body):
    if len(row.frags) == 1:
        f = row.frags[0]
        return (f[1] - f[0]) > 22 * body and len(f[2]) > 60
    lens = [len(f[2]) for f in row.frags]
    if len(row.frags) <= 3 and min(lens) > 35:
        return True
    ext = row.frags[-1][1] - row.frags[0][0]
    cover = sum(f[1] - f[0] for f in row.frags) / ext if ext > 0 else 1
    words = sum(len(f[2].split()) for f in row.frags)
    gaps = [b[0] - a[1] for a, b in zip(row.frags, row.frags[1:])]
    if len(row.frags) >= 3 and max(gaps) >= 2.5 * body and all(len(f[2].split()) <= 3 for f in row.frags[:-1]):
        return False        # short cells then a long description ('13 | heatSourceMaxTemp | If it is a heat ...')
    return cover > 0.85 and words >= 8 and sum(lens) > 40


def _continuation(row, block, body):
    """A wrapped cell's second line: one or more fragments that start where a later column of the rows above
    starts (not the first column), close below the previous row."""
    if not block or not row.frags or sum(1 for r in block if len(r.frags) >= 2) < 2:
        return False
    prev = block[-1]
    lh = max(1.0, prev.y1 - prev.y0)
    if row.y0 - prev.y1 >= 1.4 * lh:
        return False
    starts = [round(f[0]) for r in block if len(r.frags) >= 2 for f in r.frags[1:]]
    first = min(f[0] for r in block for f in r.frags)
    for f in row.frags:
        x = f[0]
        if x <= first + 0.5 * body:
            return False
        if not any(abs(x - s0) <= 6 or 0 <= x - s0 <= 18 for s0 in starts):
            return False
    return True


def merge_wrapped_rows(tb):
    """Fold wrapped cell lines back into their row. A line whose first column is empty, that fills fewer columns
    than most rows and holds text (not values), belongs either to the row above (a wrapped cell: 'If it requires
    a key ...' / 'the key that opens it.') or, when the cell is vertically centred on its row, to the row below
    whose cell in that column is still empty. The nearer row wins; a value or code never wraps."""
    if tb.n_rows < 3 or tb.n_cols < 2:
        return tb
    grid = {(c.row, c.col): c for c in tb.cells}
    text = {k: c.text.strip() for k, c in grid.items()}
    filled = {r: [c for c in tb.cells if c.row == r and c.text.strip()] for r in range(tb.n_rows)}
    counts = sorted(len(v) for v in filled.values())
    med = counts[len(counts) // 2]
    has_first = lambda r: bool(text.get((r, 0)))
    if sum(1 for r in range(tb.n_rows) if has_first(r)) < 0.6 * tb.n_rows:
        return tb
    valueish = lambda t: bool(t) and (NUMERIC.match(t) or sum(ch.isdigit() for ch in t) > 0.4 * len(t))
    cy = lambda c: (c.bbox[1] + c.bbox[3]) / 2
    moved = {}                                   # row -> (destination row, 'up' | 'down')
    for r in range(tb.header_rows, tb.n_rows):
        cs = filled[r]
        if not cs or has_first(r) or len(cs) >= med or any(valueish(c.text.strip()) for c in cs):
            continue
        up = r - 1
        while up in moved:
            up = moved[up][0]
        down = r + 1 if r + 1 < tb.n_rows and has_first(r + 1) else None
        ok_up = up >= tb.header_rows and has_first(up) and all(
            not valueish(text.get((up, c.col), "")) and (not text.get((up, c.col)) or c.text.strip()[:1].islower()
                                                         or not re.search(r"[.!?:]$", text[(up, c.col)]))
            for c in cs)
        ok_down = down is not None and all(not text.get((down, c.col)) for c in cs)
        if not (ok_up or ok_down):
            continue
        if ok_up and ok_down:
            ref_u, ref_d = grid.get((up, 0)), grid.get((down, 0))
            y = sum(cy(c) for c in cs) / len(cs)
            lh = max(1.0, max(c.bbox[3] - c.bbox[1] for c in cs))
            go_down = ref_u is not None and ref_d is not None and abs(cy(ref_d) - y) < abs(cy(ref_u) - y) - 0.25 * lh
        else:
            go_down = ok_down
        dest = down if go_down else up
        for c in cs:
            k = (dest, c.col)
            text[k] = (c.text.strip() + " " + text.get(k, "")).strip() if go_down else (text.get(k, "") + " " + c.text.strip()).strip()
        moved[r] = (dest, "down" if go_down else "up")
    if not moved:
        return tb
    for r, (dest, how) in moved.items():
        for c in filled[r]:
            a = grid.get((dest, c.col))
            if a is None:
                c.row = dest; grid[(dest, c.col)] = c; c2 = c
            else:
                a.bbox = (min(a.bbox[0], c.bbox[0]), min(a.bbox[1], c.bbox[1]), max(a.bbox[2], c.bbox[2]), max(a.bbox[3], c.bbox[3])) \
                    if a.text.strip() else c.bbox
    for k, c in grid.items():
        if k[0] not in moved:
            c.text = text.get(k, c.text)
    keep = [r for r in range(tb.n_rows) if r not in moved]
    newidx = {r: k for k, r in enumerate(keep)}
    cells = []
    for k, c in grid.items():
        if k[0] in moved:
            continue
        c.row = newidx[k[0]]
        cells.append(c)
    tb.cells = sorted(cells, key=lambda c: (c.row, c.col))
    tb.n_rows = len(keep)
    tb.warnings.append(f"{len(moved)} wrapped cell line(s) joined to their row.")
    return tb


def _short(row, body):
    return all(len(f[2]) <= 40 for f in row.frags)


def _rule_between(hrules, y0, y1):
    return any(y0 - 1 <= h[1] <= y1 + 1 for h in hrules)


def _column_edges(block, body):
    """Columns from x-coverage of fragments; spanning fragments (covering several columns) are ignored first."""
    frs = [f for r in block for f in r.frags]
    if not frs:
        return []
    widths = sorted(f[1] - f[0] for f in frs)
    med = widths[len(widths) // 2]
    narrow = [f for f in frs if (f[1] - f[0]) <= max(3 * med, 6 * body)] or frs
    narrow_ids = {id(f) for f in narrow}
    iv = sorted((f[0], f[1]) for f in narrow)
    cols, cur = [], list(iv[0])
    for a, b in iv[1:]:
        if a <= cur[1] + 0.3 * body:
            cur[1] = max(cur[1], b)
        else:
            cols.append(tuple(cur)); cur = [a, b]
    cols.append(tuple(cur))
    # wide fragments that overlap no column are a real (text) column, not a spanning header
    extra = [(f[0], f[1]) for f in frs if id(f) not in narrow_ids and not any(min(f[1], c[1]) - max(f[0], c[0]) > 0 for c in cols)]
    if extra:
        allc = sorted(cols + extra)
        cols, cur = [], list(allc[0])
        for a, b in allc[1:]:
            if a <= cur[1] + 0.3 * body:
                cur[1] = max(cur[1], b)
            else:
                cols.append(tuple(cur)); cur = [a, b]
        cols.append(tuple(cur))
    return cols


def _split_rivers(block, cols, body):
    """Split a column at a vertical whitespace channel that runs through (nearly) every row of the column -
    e.g. a narrow '#' column whose numbers sit one word-space left of the next column ('1 Easy')."""
    out = []
    for c0, c1 in cols:
        glyphs, rows_in = [], 0
        for r in block:
            cs = [ch for f in r.frags for ch in f[3] if c0 - 0.5 <= (ch.x0 + ch.x1) / 2 <= c1 + 0.5]
            if cs:
                rows_in += 1
                glyphs.append([(ch.x0, ch.x1) for ch in cs])
        if rows_in < 3 or c1 - c0 < 3 * body:
            out.append((c0, c1)); continue
        step = 0.5
        n = int((c1 - c0) / step) + 1
        occ = [0] * n
        for gl in glyphs:
            mark = [False] * n
            for a, b in gl:
                for k in range(max(0, int((a - c0) / step)), min(n, int((b - c0) / step) + 1)):
                    mark[k] = True
            for k in range(n):
                occ[k] += mark[k]
        best, run = None, []
        for k in range(n + 1):
            if k < n and occ[k] <= 0.3 * rows_in:
                run.append(k); continue
            if run and len(run) * step >= 0.25 * body and run[0] > 0 and k < n:
                x = c0 + (run[0] + run[-1]) / 2 * step
                lft = sum(1 for gl in glyphs if any(b <= x for a, b in gl))
                rgt = sum(1 for gl in glyphs if any(a >= x for a, b in gl))
                if lft >= 0.6 * rows_in and rgt >= 0.6 * rows_in and (best is None or len(run) > best[1]):
                    best = (x, len(run))
            run = []
        if best:
            out += [(c0, best[0] - 0.01), (best[0] + 0.01, c1)]
        else:
            out.append((c0, c1))
    return out


def _resplit_frags(block, cols, rivers=True):
    """Cut fragments that straddle a column boundary created by river splitting."""
    from .textlayer import _chars_to_text
    bounds_all = sorted(b[1] for b in cols[:-1])
    starts = sorted(b[0] for b in cols[1:])
    for r in block:
        nf = []
        for f in r.frags:
            vis0 = [c for c in f[3] if c.c.strip()]
            cut = [b for b in bounds_all if f[0] < b < f[1]] if rivers else []
            # only cut where the fragment really has whitespace at the boundary (a category label that runs
            # across both columns stays whole and becomes a spanning cell)
            cut = [b for b in cut if any(a.x1 <= b <= c.x0 and c.x0 - a.x1 > 0.15 * max(a.size, 1) for a, c in zip(vis0, vis0[1:]))]
            # a long name that runs up to the next column ('antirequirementsNutrientLevels A list of ...'): cut at
            # a word gap right where the next column starts
            for s0 in starts:
                if f[0] + 1 < s0 < f[1] and not any(abs(s0 - b) < 3 for b in cut):
                    gap = next(((a, c) for a, c in zip(vis0, vis0[1:]) if a.x1 <= s0 + 1 and abs(c.x0 - s0) <= 1.5
                                and c.x0 - a.x1 > 0.15 * max(a.size, 1)), None)
                    left = _chars_to_text([c for c in f[3] if c.x1 <= s0 + 1]).strip()     # spaces may be implicit
                    if gap and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-.]{9,}", left):     # one long identifier, not prose or a value
                        cut.append((gap[0].x1 + gap[1].x0) / 2)
            cut.sort()
            if not cut:
                nf.append(f); continue
            pieces, cur = [], []
            for ch in f[3]:
                if cur and cut and (ch.x0 + ch.x1) / 2 > cut[0]:
                    pieces.append(cur); cur = []
                    cut.pop(0)
                cur.append(ch)
            if cur:
                pieces.append(cur)
            for pc in pieces:
                vis = [c for c in pc if c.c.strip()]
                if vis:
                    nf.append((min(c.x0 for c in vis), max(c.x1 for c in vis), _chars_to_text(pc), vis))
        r.frags = nf


def _build_aligned(block, hrules, body):
    cols = _column_edges(block, body)
    if len(cols) < 2:
        return None
    rc = _split_rivers(block, cols, body)
    rivers = len(rc) > len(cols)
    if rivers:
        cols = rc
    _resplit_frags(block, cols, rivers)
    nrows, ncols = len(block), len(cols)
    # reject multi-column prose: few columns, long fragments
    lens = [len(f[2]) for r in block for f in r.frags]
    if ncols <= 3 and statistics.median(lens) > 28:
        return None
    filled_rows = sum(1 for r in block if len(r.frags) >= 2)
    if filled_rows < 0.5 * nrows:
        return None
    # cells of a real table line up: each fragment starts, ends or centres on its column's edge/centre.
    # Justified or ragged prose split at wide word gaps does not.
    frs = [f for r in block for f in r.frags]
    aligned = 0
    for f in frs:
        col = max(cols, key=lambda c: min(f[1], c[1]) - max(f[0], c[0]))
        tol = 1.2 * body
        if abs(f[0] - col[0]) < tol or abs(f[1] - col[1]) < tol or abs((f[0] + f[1]) / 2 - (col[0] + col[1]) / 2) < tol:
            aligned += 1
    align_ratio = aligned / max(1, len(frs))
    # prose split at word gaps covers almost the whole row; table rows have real whitespace between cells
    cover = []
    for r in block:
        if len(r.frags) >= 2:
            ext = r.frags[-1][1] - r.frags[0][0]
            cover.append(sum(f[1] - f[0] for f in r.frags) / ext if ext > 0 else 1)
    if cover and statistics.median(cover) > 0.82:
        return None
    words_per_cell = statistics.mean(len(f[2].split()) for f in frs)
    if align_ratio < 0.75 or (words_per_cell > 5.5 and ncols <= 3):
        return None
    cells, grid_used = [], set()
    for ri, r in enumerate(block):
        for f in r.frags:
            cover = [k for k, c in enumerate(cols) if min(f[1], c[1]) - max(f[0], c[0]) > 0.3 * min(f[1] - f[0], c[1] - c[0])]
            if not cover:
                cx = (f[0] + f[1]) / 2
                cover = [min(range(ncols), key=lambda k: abs((cols[k][0] + cols[k][1]) / 2 - cx))]
            c0, c1 = min(cover), max(cover)
            if any((ri, k) in grid_used for k in range(c0, c1 + 1)):
                # two fragments landed in one column: append text to the existing cell
                ex = next((c for c in cells if c.row == ri and c.col <= c0 < c.col + c.colspan), None)
                if ex:
                    ex.text = (ex.text + " " + f[2]).strip()
                    ex.bbox = (min(ex.bbox[0], f[0]), ex.bbox[1], max(ex.bbox[2], f[1]), ex.bbox[3])
                continue
            for k in range(c0, c1 + 1):
                grid_used.add((ri, k))
            cells.append(Cell(ri, c0, f[2], (f[0], r.y0, f[1], r.y1), 1, c1 - c0 + 1))
    for ri in range(nrows):
        for k in range(ncols):
            if (ri, k) not in grid_used:
                cells.append(Cell(ri, k, "", (cols[k][0], block[ri].y0, cols[k][1], block[ri].y1)))
    # header rows: rows above the first horizontal rule that lies inside the table; else first row
    y_top, y_bot = block[0].y0, block[-1].y1
    x0, x1 = min(c[0] for c in cols), max(c[1] for c in cols)
    x0 = min([x0] + [f[0] for r in block for f in r.frags])       # long cells (wider than their column's core)
    x1 = max([x1] + [f[1] for r in block for f in r.frags])
    inner = sorted(h[1] for h in hrules if y_top + 2 < h[1] < y_bot - 2 and min(h[2], x1) - max(h[0], x0) > 0.5 * (x1 - x0))
    hdr = 1
    if inner:
        hdr = sum(1 for r in block if r.y1 <= inner[0] + 1) or 1
        if hdr > 4 or hdr >= nrows - 1:
            hdr = 1
    for c in cells:
        c.header = c.row < hdr
    # numeric share boosts confidence
    texts = [c.text for c in cells if c.text]
    num = sum(1 for t in texts if NUMERIC.match(t)) / max(1, len(texts))
    align = filled_rows / nrows * align_ratio
    conf = round(min(0.9, 0.45 + 0.25 * align + 0.2 * num + (0.05 if inner else 0)), 2)
    tb = Table((x0, y_top, x1, y_bot), nrows, ncols, sorted(cells, key=lambda c: (c.row, c.col)), "aligned_text", conf, hdr)
    merge_wrapped_rows(tb)
    cells = tb.cells
    empties = sum(1 for c in cells if not c.text)
    if empties:
        tb.warnings.append(f"{empties} empty cell(s): left blank, nothing inferred.")
    if any(c.colspan > 1 for c in cells):
        tb.warnings.append("Some cells span several columns (merged headers); emitted as HTML colspan.")
    return tb


# ---------------------------------------------------------------------------------------------
# Ruled tables (PyMuPDF find_tables) -> structured model with spans
# ---------------------------------------------------------------------------------------------
def grid_regions(hrules, vrules, pad=4.0):
    """Regions where vector rules actually form a grid: >= 2 vertical rules overlapping in y, crossed by
    >= 2 horizontal rules. find_tables() is expensive, so it only runs inside these regions."""
    # a grid drawn cell by cell (Word, PowerPoint, HTML renderers) has its rules in pieces: join collinear pieces
    hrules = _join_rules(hrules, 1)
    vrules = _join_rules(vrules, 0)
    vs = sorted(vrules, key=lambda v: v[1])
    clusters = []
    for v in vs:
        for c in clusters:
            if v[1] <= c["y1"] + pad and v[3] >= c["y0"] - pad:
                c["v"].append(v); c["y0"] = min(c["y0"], v[1]); c["y1"] = max(c["y1"], v[3]); break
        else:
            clusters.append(dict(v=[v], y0=v[1], y1=v[3]))
    out = []
    for c in clusters:
        xs = sorted({round(v[0]) for v in c["v"]})
        if len(xs) < 2:
            continue
        x0, x1 = xs[0], xs[-1]
        cross = [h for h in hrules if c["y0"] - pad <= h[1] <= c["y1"] + pad and h[0] <= x0 + pad and h[2] >= x1 - pad]
        if len(cross) >= 2:
            out.append((x0 - pad, c["y0"] - pad, x1 + pad, c["y1"] + pad))
    return out


def _join_rules(rules, axis, tol=1.0, gap=2.5):
    """Join rules lying on one line (axis 1: horizontal rules (x0, y, x1, y); axis 0: vertical (x, y0, x, y1))."""
    if not rules:
        return list(rules or [])
    lo, hi = (0, 2) if axis == 1 else (1, 3)
    pos = 1 if axis == 1 else 0
    out = []
    for r in sorted(rules, key=lambda r: (round(r[pos]), r[lo])):
        r = list(r)
        if out and abs(out[-1][pos] - r[pos]) <= tol and r[lo] <= out[-1][hi] + gap:
            out[-1][hi] = max(out[-1][hi], r[hi])
        else:
            out.append(r)
    return [tuple(r) for r in out]


def detect_ruled(page, segs, has_vrules, hrules=None, vrules=None):
    if not has_vrules:
        return []
    out = []
    regions = grid_regions(hrules or [], vrules or []) if vrules is not None else [None]
    found = []
    for clip in regions:
        try:
            found += page.find_tables(strategy="lines", clip=clip).tables if clip else page.find_tables(strategy="lines").tables
        except Exception:
            continue
    for t in found:
        cb = [tuple(c) for c in (t.cells or []) if c]
        if len(cb) < 4:
            continue
        xs = _cluster(sorted({round(v, 1) for c in cb for v in (c[0], c[2])}))
        ys = _cluster(sorted({round(v, 1) for c in cb for v in (c[1], c[3])}))
        ncols, nrows = len(xs) - 1, len(ys) - 1
        if ncols < 2 or nrows < 2:
            continue
        cells, seen = [], set()
        for c in sorted(set(cb), key=lambda c: (c[1], c[0])):
            c0, c1 = _idx(xs, c[0]), _idx(xs, c[2])
            r0, r1 = _idx(ys, c[1]), _idx(ys, c[3])
            if c1 <= c0 or r1 <= r0 or (r0, c0) in seen:
                continue
            for r in range(r0, r1):
                for k in range(c0, c1):
                    seen.add((r, k))
            cells.append(Cell(r0, c0, _text_in(segs, c), c, r1 - r0, c1 - c0))
        for r in range(nrows):
            for k in range(ncols):
                if (r, k) not in seen:
                    cells.append(Cell(r, k, "", (xs[k], ys[r], xs[k + 1], ys[r + 1])))
        filled = sum(1 for c in cells if c.text) / max(1, len(cells))
        if filled < 0.25:
            continue
        hdr = 1
        try:
            if t.header and t.header.external:
                hdr = 0
        except Exception:
            pass
        top_span = [c for c in cells if c.row == 0 and c.rowspan > 1]
        if top_span:
            hdr = max(hdr, max(c.rowspan for c in top_span))
        for c in cells:
            c.header = c.row < hdr
        tb = Table(tuple(t.bbox), nrows, ncols, sorted(cells, key=lambda c: (c.row, c.col)), "ruled_grid",
                   round(min(0.95, 0.6 + 0.35 * filled), 2), hdr)
        if any(c.rowspan > 1 or c.colspan > 1 for c in cells):
            tb.warnings.append("Merged cells detected from ruling geometry; emitted as HTML rowspan/colspan.")
        out.append(tb)
    return out


def _cluster(vals, tol=2.0):
    out = []
    for v in vals:
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _idx(edges, v):
    return min(range(len(edges)), key=lambda i: abs(edges[i] - v))


def _text_in(segs, box):
    from .textlayer import _chars_to_text
    lines = []
    for s in segs:
        cs = [c for c in s.chars if box[0] - 1 <= (c.x0 + c.x1) / 2 <= box[2] + 1 and box[1] - 1 <= (c.y0 + c.y1) / 2 <= box[3] + 1]
        if cs:
            lines.append((s.baseline, cs[0].x0, _chars_to_text(cs)))
    lines.sort()
    return " ".join(t for _, _, t in lines if t).strip()


# ---------------------------------------------------------------------------------------------
# Post-processing shared by both detectors
# ---------------------------------------------------------------------------------------------
_SYMBOL_ONLY = re.compile(r"^\s*[$€£¥%]\s*$")


def _col_intervals(tb):
    iv = {}
    for c in tb.cells:
        if c.colspan == 1 and c.text and c.bbox[2] > c.bbox[0]:
            a, b = iv.get(c.col, (c.bbox[0], c.bbox[2]))
            iv[c.col] = (min(a, c.bbox[0]), max(b, c.bbox[2]))
    for k in range(tb.n_cols):
        if k not in iv:
            xs = [c.bbox for c in tb.cells if c.col == k and c.bbox[2] > c.bbox[0]]
            iv[k] = (min(b[0] for b in xs), max(b[2] for b in xs)) if xs else (0, 0)
    return [iv[k] for k in range(tb.n_cols)]


def merge_symbol_columns(tb):
    """A column holding only currency/percent signs is typography, not data: fold '$' into the value to its right
    ('$178') and '%' into the value to its left. Nothing is dropped."""
    changed = True
    while changed and tb.n_cols > 2:
        changed = False
        for k in range(tb.n_cols):
            col = [c for c in tb.cells if c.col == k and c.colspan == 1]
            filled = [c for c in col if c.text]
            if not filled or not all(_SYMBOL_ONLY.match(c.text) for c in filled):
                continue
            pct = all("%" in c.text for c in filled)
            tgt = k - 1 if pct else k + 1
            if not (0 <= tgt < tb.n_cols):
                continue
            for c in filled:
                other = next((o for o in tb.cells if o.row == c.row and o.col <= tgt < o.col + o.colspan), None)
                if other is not None:
                    other.text = (other.text + c.text.strip()) if pct else (c.text.strip() + other.text)
            # remove column k: cells in it disappear, spans crossing it shrink, later columns shift left
            keep = []
            for c in tb.cells:
                if c.col == k and c.colspan == 1:
                    continue
                if c.col <= k < c.col + c.colspan:
                    c.colspan -= 1
                elif c.col > k:
                    c.col -= 1
                keep.append(c)
            tb.cells = keep
            tb.n_cols -= 1
            changed = True
            break
    return tb


def extend_header(tb, segs, hrules, body):
    """Short, column-aligned rows directly above a table are its header (e.g. 'Year Ended' over date columns).
    A horizontal rule right under a header cell defines how many columns it spans."""
    if tb.n_cols < 2:
        return tb, set()
    cols = _col_intervals(tb)
    x0 = min(c[0] for c in cols if c[1] > c[0]); x1 = max(c[1] for c in cols)
    cand = [s for s in segs if s.horizontal and s.bbox[3] <= tb.bbox[1] + 1 and s.bbox[0] >= x0 - 2 * body and s.bbox[2] <= x1 + 2 * body]
    rows = sorted(_rows(cand), key=lambda r: -r.y1)
    # collinear rule pieces with small gaps (one underline broken at column boundaries) act as one rule
    merged = []
    for h in sorted(hrules, key=lambda h: (round(h[1], 0), h[0])):
        if merged and abs(merged[-1][1] - h[1]) < 0.6 and h[0] - merged[-1][2] < max(1.5 * body, 12.0):
            m = merged[-1]; merged[-1] = (m[0], m[1], max(m[2], h[2]), m[3])
        else:
            merged.append(tuple(h))
    pieces = [tuple(h) for h in hrules]
    hrules = merged
    first_col_end = cols[0][1]
    # a ruled grid closed at the top by a rule across its full width already contains its header row: text above
    # that rule (a section heading, a sheet name) is not part of the table
    closed = tb.method == "ruled_grid" and any(abs(h[1] - tb.bbox[1]) < 2.0 and h[0] <= tb.bbox[0] + 3 and h[2] >= tb.bbox[2] - 3 for h in hrules)
    top, added, used = tb.bbox[1], [], set()
    for r in rows:
        lh = max(1.0, r.y1 - r.y0)
        if closed:
            break
        if any(sg.size > 1.3 * max(body, 1) for sg in r.segs):
            break       # a heading above the table
        if top - r.y1 > max(2.2 * lh, 1.8 * body) and not _rule_between(hrules, r.y1, top):
            break
        if _proselike(r, body) or any(len(f[2]) > 45 for f in r.frags):
            break
        if not any(f[1] > first_col_end + 0.5 * body for f in r.frags):
            break       # a header row labels data columns; text confined to the stub column is prose/caption
        added.insert(0, r)
        top = r.y0
        for s in r.segs:
            used.add(id(s))
        if len(added) >= 4:
            break
    # rotated column labels (e.g. 45-degree headers) directly above the table: one header row, each label
    # assigned to the column under its starting point
    rot = [s for s in segs if not s.horizontal and s.bbox[3] <= tb.bbox[1] + 0.8 * body and tb.bbox[1] - s.bbox[3] < 3 * body
           and x0 - 2 * body <= s.chars[0].ox <= x1 + 2 * body]
    if rot:
        ry0, ry1 = min(s.bbox[1] for s in rot), max(s.bbox[3] for s in rot)
        frs = [(s.chars[0].ox, s.chars[0].ox + 0.1, s.text, s.chars) for s in rot]
        r = _Row(ry0, ry1, ry1, sorted(frs, key=lambda f: f[0]), rot)
        added = [a for a in added if a.y1 <= ry0 + 1] + [r] + [a for a in added if a.y1 > ry0 + 1]
        for sg in rot:
            used.add(id(sg))
        # horizontal rows further up (group labels like 'ReACT') become reachable once the rotated band is in
        more = sorted(_rows([sg for sg in cand if sg.bbox[3] <= ry0 + 1 and id(sg) not in used]), key=lambda x: -x.y1)
        top2 = ry0
        for x in more:
            lh = max(1.0, x.y1 - x.y0)
            if top2 - x.y1 > max(2.2 * lh, 1.8 * body) or _proselike(x, body) or any(len(f[2]) > 45 for f in x.frags):
                break
            if any(id(sg) in used for sg in x.segs):
                continue
            added.append(x); top2 = x.y0
            for sg in x.segs:
                used.add(id(sg))
            if len(added) >= 5:
                break
    if not added:
        return tb, set()
    added = sorted(added, key=lambda a: a.y0)
    k = len(added)
    for c in tb.cells:
        c.row += k
        c.header = False
    new = []
    for ri, r in enumerate(added):
        occupied = set()
        for f in r.frags:
            fc = (f[0] + f[1]) / 2
            near = lambda h: r.y1 - 1 <= h[1] <= r.y1 + 0.9 * (r.y1 - r.y0) + 2 and h[0] - 1 <= fc <= h[2] + 1
            rotated_row = all(not sg.horizontal for sg in r.segs)
            tw = max(1.0, x1 - x0)
            rule = None if rotated_row else next((h for h in hrules if near(h) and h[2] - h[0] > (f[1] - f[0]) * 1.2 and (h[2] - h[0]) < 0.9 * tw), None)
            if rule and sum(1 for g in r.frags if rule[0] - 1 <= (g[0] + g[1]) / 2 <= rule[2] + 1) > 1:
                # several labels under one long underline: each label owns only its own piece
                rule = next((h for h in pieces if near(h) and h[2] - h[0] > (f[1] - f[0]) * 1.2), None)
            if rule:
                cover = [j for j, c in enumerate(cols) if rule[0] - 2 <= (c[0] + c[1]) / 2 <= rule[2] + 2]
            else:
                cover = [j for j, c in enumerate(cols) if min(f[1], c[1]) - max(f[0], c[0]) > 0]
            if not cover:
                cover = [min(range(len(cols)), key=lambda j: abs((cols[j][0] + cols[j][1]) / 2 - fc))]
            c0, c1 = min(cover), max(cover)
            if any(j in occupied for j in range(c0, c1 + 1)):
                ex = next((c for c in new if c.row == ri and c.col <= c0 < c.col + c.colspan), None)
                if ex:
                    ex.text = (ex.text + " " + f[2]).strip()
                continue
            occupied.update(range(c0, c1 + 1))
            new.append(Cell(ri, c0, f[2], (f[0], r.y0, f[1], r.y1), 1, c1 - c0 + 1, True))
        for j in range(tb.n_cols):
            if j not in occupied:
                new.append(Cell(ri, j, "", (cols[j][0], r.y0, cols[j][1], r.y1), 1, 1, True))
    tb.cells = sorted(new + tb.cells, key=lambda c: (c.row, c.col))
    tb.n_rows += k
    tb.header_rows = k
    tb.bbox = (min(tb.bbox[0], min(f[0] for r in added for f in r.frags)), added[0].y0, tb.bbox[2], tb.bbox[3])
    tb.warnings.append(f"{k} header row(s) above the ruled/aligned body were attached by column alignment.")
    return tb, used


# ---------------------------------------------------------------------------------------------
# Plausibility checks: things that align like a table but are not one
# ---------------------------------------------------------------------------------------------
FIG_CAP = re.compile(r"^\s*(fig(?:ure)?\.?|plate|chart|diagram)\s*[\dA-Z]", re.I)
TAB_CAP = re.compile(r"^\s*table\s*[\dA-Z]", re.I)


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


INDEX_CELL = re.compile(r",\s*(?:(?:\d{1,4}(?:[–-]\d{1,4})?(?:\s*(?:ex|pr|n|fig)\.)?)(?:,\s*|$))+$|,\s*continued$|^see(?: also)?\b|^\d{1,4}(?:[–-]\d{1,4})?\s*(?:ex|pr)\.$")


def reject_reason(tb, segs, drawings, image_boxes, body):
    """Why a detected table is not a table (None if it is plausible):
    picture inside it, a vector diagram (curves / slanted arrows), a mathematical display, a table of contents,
    a numbered listing (pseudo-code), or a region captioned as a figure that contains graphics."""
    from . import mathtex
    box = tb.bbox
    A = _area(box) or 1.0
    # 1. a raster picture covers a real part of it
    if sum(_inter(box, ib) for ib in image_boxes) > 0.15 * A:
        return "a picture lies inside the region"
    # 2. vector diagram: curves or slanted lines strictly inside (rounded table corners sit on the border)
    curves = slanted = shapes = 0
    inner = (box[0] + 3, box[1] + 3, box[2] - 3, box[3] - 3)
    for d in drawings or ():
        r = d.get("rect")
        if r is None or _inter(tuple(r), inner) <= 0:
            continue
        shapes += 1
        for it in d.get("items", ()):
            if it[0] == "c":
                if inner[0] <= it[1].x <= inner[2] and inner[1] <= it[1].y <= inner[3]:
                    curves += 1
            elif it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.x - p2.x) > 2 and abs(p1.y - p2.y) > 2 and inner[0] <= p1.x <= inner[2] and inner[1] <= p1.y <= inner[3]:
                    slanted += 1
    if curves >= 2 or slanted >= 2:
        return "vector diagram (curves or arrows) inside the region"
    inside = [s for s in segs if _inter(s.bbox, box) > 0.6 * (_area(s.bbox) or 1)]
    chars = [c for s in inside for c in s.chars if c.c.strip()]
    # 3. mathematical display: relations, big operators, Greek, math-font glyphs (signs and percentages excluded)
    if chars:
        sizes = sorted(c.size for c in chars)
        main = sizes[len(sizes) // 2]
        math = sum(1 for c in chars if c.c not in "+−-–%$€£.,()" and mathtex._is_strong(c, main))
        if math >= 4 and math >= 0.08 * len(chars):
            return "mathematical display (equation), not tabular data"
    rows = [r for r in tb.grid() if any(str(x).strip() for x in r)]
    # 4. table of contents: rows end in increasing page numbers
    if len(rows) >= 5:
        ends = []
        for r in rows:
            cells = [str(x).strip() for x in r if str(x).strip()]
            m = re.search(r"(?<![\d.,])(\d{1,4})$", cells[-1]) if cells else None
            # one integer page number per entry; the rest of the row is the title (no other numeric cells)
            others = [c for c in cells[:-1] if re.fullmatch(r"[-+−]?[\d.,%]+", c)]
            ends.append(int(m.group(1)) if m and not others and re.search(r"[A-Za-z]{3}", " ".join(cells)) else None)
        nums = [e for e in ends if e is not None]
        if len(nums) >= 0.7 * len(rows):
            inc = sum(1 for a, b in zip(nums, nums[1:]) if b >= a)
            if inc >= 0.85 * (len(nums) - 1) and nums[-1] > nums[0]:
                return "table of contents (entries with page numbers)"
    # 5. numbered listing (pseudo-code / algorithm)
    if len(rows) >= 3:
        lead = []
        for r in rows:
            first = next((str(x).strip() for x in r if str(x).strip()), "")
            m = re.match(r"^\(?(\d{1,3})[.):]?(?:\s|$)", first)
            lead.append(int(m.group(1)) if m else None)
        nums = [x for x in lead if x is not None]
        if len(nums) >= 0.6 * len(rows) and len(nums) >= 3:
            if sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1) >= 0.7 * (len(nums) - 1):
                code_rows = sum(1 for r in rows if re.search(r"\b(if|for|while|return|else|then|do|repeat|until|let|call|print|def|end)\b|[=;{}←]|//|\w\(|\[",
                                                       " ".join(str(x) for x in r[1:])))
                id_table = max(len(r) for r in rows) >= 3 and code_rows < 0.3 * len(rows)   # '2 | isMovable | Can it be moved?'
                if not id_table and (not any(re.search(r"\d", str(x)) for r in rows for x in r[2:]) or len(rows[0]) <= 3):
                    return "numbered listing (algorithm / pseudo-code)"
    filled = [[str(x).strip() for x in r if str(x).strip()] for r in rows]
    cells_all = [c for r in filled for c in r]
    # 7. aligned equations: an equation-number column '(3.18)' or rows starting with '='
    if len(rows) >= 2:
        eqno = sum(1 for r in filled if any(re.fullmatch(r"\(\s*[A-Z]?\d+(?:\.\d+)*[a-z]?\s*\)", c) for c in r))
        rel = sum(1 for r in filled if len(r) >= 2 and re.match(r"^[=<>≤≥≈∼]|^D\b", r[1] if len(r) > 1 else ""))
        if eqno >= 0.5 * len(rows) or rel >= 0.5 * len(rows):
            return "aligned equations (equation numbers / relation column)"
    # 8. figure panels: '(a)', '(b)' labels inside
    if sum(1 for c in cells_all if re.fullmatch(r"\([a-h]\)", c)) >= 2:
        return "figure panels labelled (a), (b), ..."
    # 9. an index: most cells are 'entry, page' items
    if cells_all and len(cells_all) >= 4 and sum(1 for c in cells_all if INDEX_CELL.search(c)) >= 0.45 * len(cells_all):
        return "book index (entries with page numbers)"
    # 10. coloured boxes (array cells drawn as a figure)
    colored = 0
    for d in drawings or ():
        f = d.get("fill")
        r = d.get("rect")
        if f and r is not None and _inter(tuple(r), box) > 0.8 * _area(tuple(r)) and _area(tuple(r)) < 0.05 * A \
                and r.width < 0.4 * (box[2] - box[0]) and r.width < 2.5 * r.height:
            # square-ish boxes (array cells in a diagram); wide flat fills are table cell or row shading
            if max(f) - min(f) > 0.08:
                colored += 1
    total_cells = sum(len(r) for r in rows) or 1
    fill_ratio = len(cells_all) / total_cells
    fig_near = any(s.horizontal and FIG_CAP.match(s.text or "") and min(abs(s.bbox[1] - box[3]), abs(box[1] - s.bbox[3])) < 6 * body
                   for s in segs)
    if colored >= 4 and (fill_ratio < 0.7 or fig_near):
        # coloured cells in a sparse grid, or under a 'Figure' caption: array diagrams (a heat-mapped data
        # table is dense and keeps its table)
        return "coloured boxes drawn as a figure (array / diagram)"
    # 6. captioned as a figure and containing graphics
    near = [s for s in segs if s.horizontal and FIG_CAP.match(s.text or "")
            and min(abs(s.bbox[1] - box[3]), abs(box[1] - s.bbox[3])) < 6 * body
            and min(s.bbox[2], box[2]) - max(s.bbox[0], box[0]) > 0]
    if near and not any(TAB_CAP.match(s.text or "") for s in segs if min(abs(s.bbox[1] - box[3]), abs(box[1] - s.bbox[3])) < 3 * body):
        if shapes >= 5 or any(_inter(box, ib) > 0 for ib in image_boxes):
            return "captioned as a figure and drawn with graphics"
    return None


RUNHEAD = re.compile(r"^\s*(?:\d{1,4}\s+)?(?:chapter|appendix|part)\s+[\dA-Z]+\b|^\s*\d+(?:\.\d+)+\s+[A-Z][^|]{2,80}$", re.I)
SECTIONISH = re.compile(r"^\s*(?:exercises?|problems?(?:\s+for\s+chapter\s+\d+)?|notes|chapter notes|solutions?|examples?)\s*$|^\s*\d+(?:\.\d+)*-\d+\s*$", re.I)


def _row_cells(tb):
    rows = {}
    for c in tb.cells:
        rows.setdefault(c.row, []).append(c)
    return rows


def trim_table(tb, page_h, body):
    """Drop rows that a table detector picked up around a table but that are not part of it: the running header
    at the top of the page, a lead-in sentence ('... as follows:'), a section label ('Exercises', '4.2-1').
    Returns the trimmed table, or None if fewer than two rows remain."""
    rows = _row_cells(tb)
    order = sorted(rows)

    def junk(r):
        cs = [c for c in rows[r] if c.text.strip()]
        if not cs:
            return True
        txt = " ".join(c.text.strip() for c in sorted(cs, key=lambda c: c.col))
        y0 = min(c.bbox[1] for c in cs)
        if y0 < 0.09 * page_h and (RUNHEAD.search(txt) or re.match(r"^\s*\d{1,4}\s", txt) or re.search(r"\s\d{1,4}\s*$", txt)):
            return True                       # running header of the page
        if len(cs) == 1:
            t = cs[0].text.strip()
            if SECTIONISH.match(t) or (len(t.split()) >= 4 and re.search(r"[a-z]{3}", t) and not NUMERIC.match(t)):
                return True                   # a heading or a sentence leading into the table
        return False
    k0, k1 = 0, len(order)
    while k0 < k1 and junk(order[k0]):
        k0 += 1
    while k1 > k0 and junk(order[k1 - 1]) and len([c for c in rows[order[k1 - 1]] if c.text.strip()]) <= 1:
        k1 -= 1
    if k0 == 0 and k1 == len(order):
        return tb
    keep = set(order[k0:k1])
    if len(keep) < 2:
        return None
    base = order[k0]
    cells = [c for c in tb.cells if c.row in keep]
    for c in cells:
        c.row -= base
    tb.cells = cells
    tb.n_rows = len(keep)
    tb.header_rows = max(0, min(tb.header_rows - k0, tb.n_rows - 1)) if tb.header_rows else 0
    bb = [c.bbox for c in cells if c.bbox]
    if bb:
        tb.bbox = (min(b[0] for b in bb), min(b[1] for b in bb), max(b[2] for b in bb), max(b[3] for b in bb))
    tb.warnings.append(f"{k0 + (len(order) - k1)} row(s) around the table (page header, heading or lead-in text) left out.")
    return tb


# ---------------------------------------------------------------------------------------------
# Tables continued on the next page
# ---------------------------------------------------------------------------------------------
_CONT_CAPTION = re.compile(r"\(?\s*(continued|cont(?:'|’)?d\.?|contd\.?|fortsetzung|suite)\s*\)?", re.I)


def _col_starts(b):
    cols = {}
    for c in b.get("cells") or []:
        if c.get("colspan", 1) == 1 and c.get("text"):
            cols.setdefault(c["col"], []).append(c["bbox"][0])
    if not cols:
        return None
    x0 = min(v for vs in cols.values() for v in vs)
    return [statistics.median(cols[k]) - x0 if k in cols else None for k in range(max(cols) + 1)]


def merge_page_tables(blocks, page_widths):
    """Join a table that ends a page with a table that starts the next page when they have the same columns.
    Both blocks stay (each keeps its own page evidence); the first gets `merged` (rows, cells with their page,
    Markdown) and `continued_on`, the second `continuation_of`. A repeated header row is dropped from the merge.
    Returns the number of joins."""
    by_page = {}
    for b in blocks:
        by_page.setdefault(b["page"], []).append(b)
    body = lambda b: not (b.get("furniture") or {}).get("excluded") and b.get("role") not in ("header", "footer", "page_number")
    joins = 0
    chain_head = {}
    for p in sorted(by_page):
        nxt = by_page.get(p + 1)
        if not nxt:
            continue
        cur = sorted([b for b in by_page[p] if body(b)], key=lambda b: b.get("reading_order", 0))
        nb = sorted([b for b in nxt if body(b)], key=lambda b: b.get("reading_order", 0))
        # the table must be the last thing on its page (footnotes / captions under it are allowed) ...
        tail = [b for b in cur if b["type"] not in ("caption",) and not re.match(r"^\s*(\*|†|‡|\d+\s|[a-z]\))", b.get("content") or "")]
        if not tail or tail[-1]["type"] != "table":
            continue
        a = tail[-1]
        # ... and the next table the first thing on the next page (a '(continued)' caption may come first)
        head = [b for b in nb if not (b["type"] in ("caption", "paragraph", "heading") and _CONT_CAPTION.search(b.get("content") or "") and len(b.get("content") or "") < 120)]
        if not head or head[0]["type"] != "table":
            continue
        c = head[0]
        ra, rc = a.get("rows") or [], c.get("rows") or []
        if not ra or not rc or len(ra[0]) != len(rc[0]) or len(ra[0]) < 2:
            continue
        sa, sc = _col_starts(a), _col_starts(c)
        if sa is None or sc is None or len(sa) != len(sc):
            continue
        w = max(page_widths.get(p, 612), 1)
        diffs = [abs(x - y) for x, y in zip(sa, sc) if x is not None and y is not None]
        if not diffs or max(diffs) > 0.03 * w:                 # cell bboxes are in PDF points
            continue
        head_a = a.get("header_rows", 1)
        skip = head_a if [r for r in rc[:head_a]] == [r for r in ra[:head_a]] else 0
        root = chain_head.get(a["block_id"], a)
        m = root.get("merged") or dict(rows=list(ra), cells=[dict(cc, page=root["page"]) for cc in root.get("cells") or []], pages=[root["page"]])
        off = len(m["rows"]) - skip
        m["rows"] += rc[skip:]
        for cc in c.get("cells") or []:
            if cc["row"] >= skip:
                m["cells"].append(dict(cc, row=cc["row"] + off, page=c["page"], header=False))
        m["pages"].append(c["page"])
        cl = [Cell(x["row"], x["col"], x.get("text", ""), tuple(x.get("bbox", (0, 0, 0, 0))), x.get("rowspan", 1), x.get("colspan", 1), x.get("header", False)) for x in m["cells"]]
        m["markdown"] = serialize(Table(tuple(root["bbox"]), max(x.row + x.rowspan for x in cl), max(x.col + x.colspan for x in cl), cl,
                                        root.get("table_method", ""), root.get("confidence", 0.5), root.get("header_rows", 1)))
        root["merged"] = m
        root.setdefault("continued_on", []).append(c["block_id"])
        c["continuation_of"] = root["block_id"]
        c.setdefault("warnings", []).append(f"Continues the table {root['block_id']} from page {root['page']}; merged there.")
        chain_head[c["block_id"]] = root
        joins += 1
    return joins
