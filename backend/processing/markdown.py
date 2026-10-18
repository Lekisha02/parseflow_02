"""Canonical Markdown generation from the Digital Twin (backend-side, deterministic).

render(doc, include_furniture=False, annotate=True)

* blocks in page order, then corrected reading order
* headings (#..####), paragraphs, lists, fenced code, captions, tables (Markdown or HTML with spans),
  display equations as \\[ ... \\], inline math already embedded in paragraph content as \\( ... \\)
* confidently detected running headers/footers/page numbers are left out of the body (kept in JSON);
  include_furniture=True puts them back
* annotate=True adds HTML-comment warnings for uncertain / not-extracted regions (invisible when rendered);
  annotate=False gives clean text (used for benchmarking - content is otherwise identical)
* duplicate content from overlapping extractors is emitted once
"""
import re
import unicodedata

from . import tables as T

_BULLET = re.compile(r"^\s*[•‣●○▪■∙·➢⁃•◦▪]\s*")


def _nfc(s):
    s = (s or "").encode("utf-8", "replace").decode("utf-8")     # never emit invalid code points (lone surrogates)
    return unicodedata.normalize("NFC", s)


def _escape_line_start(t):
    # Only escape what would otherwise change document structure. Never touch backslashes (LaTeX).
    if re.match(r"^#{1,6}\s", t):
        return "\\" + t
    if re.match(r"^>\s", t):
        return "\\" + t
    if re.match(r"^(?:-{3,}|={3,}|\*{3,})\s*$", t):
        return "\\" + t
    return t


def _key(b):
    return re.sub(r"\W+", "", (b.get("content") or "").lower())[:400]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0


def block_markdown(b, annotate=True):
    t, c = b["type"], b.get("content")
    if t == "table":
        if b.get("continuation_of"):
            return ""                      # merged into the table it continues
        if b.get("merged"):
            m = b["merged"]
            return T.serialize(_table_from_block(dict(b, cells=m["cells"], rows=m["rows"])))
        if b.get("cells"):
            tb = _table_from_block(b)
            body = T.serialize(tb)
        elif b.get("rows"):
            body = T.serialize(T.from_rows(b["rows"]))
        else:
            body = ""
        return body
    if t == "equation":
        if b.get("latex"):
            lx = b["latex"].strip()
            out = f"\\[{lx}\\]" if b.get("math_display", True) else f"\\({lx}\\)"
            if b.get("equation_number"):
                out += f" {b['equation_number']}"
            return out
        return _escape_line_start(c or "") if c else ""
    if c is None:
        if not annotate:
            return b.get("ocr_text") or ""
        kind = "Visual region not interpreted" if t in ("figure", "unknown", "chart", "diagram") else "Unreadable region"
        return f"<!-- parseflow: {kind} on page {b['page']} ({b['block_id']}) - flagged for review, confidence {b['confidence']} -->" + \
            (f"\n{b['ocr_text']}" if b.get("ocr_text") else "")
    c = _nfc(c)
    if t == "heading":
        lvl = max(1, min(6, int(b.get("level") or 2)))
        return "#" * lvl + " " + re.sub(r"\s+", " ", c).strip()
    if t == "list":
        lines = [l for l in c.split("\n") if l.strip()]
        return "\n".join(_BULLET.sub("- ", l.strip()) if _BULLET.match(l) else l.strip() for l in lines)
    if t in ("code", "handwritten_code"):
        return "```\n" + c.rstrip("\n") + "\n```"
    if t == "chart" and b.get("rows") and len(b["rows"]) >= 2:
        return c + "\n\n" + T.serialize(T.from_rows(b["rows"]))
    if t in ("chart", "diagram"):
        extra = ""
        if b.get("relationships") and t == "diagram":
            rel = [r for r in b["relationships"] if not r.startswith(("caption", "caption_of"))]
            if rel:
                extra = "\n\n" + "; ".join(rel)
        return c + extra
    lines = [_escape_line_start(l.strip()) for l in c.split("\n")]
    return "\n".join(l for l in lines if l)


def _table_from_block(b):
    cells = [T.Cell(x["row"], x["col"], x.get("text", ""), tuple(x.get("bbox", (0, 0, 0, 0))), x.get("rowspan", 1), x.get("colspan", 1), x.get("header", False))
             for x in b["cells"]]
    nr = max((x.row + x.rowspan for x in cells), default=0)
    nc = max((x.col + x.colspan for x in cells), default=0)
    return T.Table(tuple(b["bbox"]), nr, nc, cells, b.get("table_method", ""), b["confidence"], b.get("header_rows", 1))


def render(doc, include_furniture=False, annotate=True):
    blocks = sorted(doc.get("blocks", []), key=lambda b: (b["page"], b.get("reading_order", 0)))
    out, seen = [], []
    if annotate and doc.get("filename"):
        out.append(f"<!-- ParseFlow · {doc['filename']} -->")
    last_page = None
    for b in blocks:
        fur = b.get("furniture") or {}
        if fur.get("excluded") and not include_furniture:
            continue
        if b["page"] != last_page:
            if annotate and last_page is not None:
                out.append(f"<!-- page {b['page']} -->")
            last_page = b["page"]
        k = _key(b)
        if k and len(k) > 12 and any(sk == k and sp == b["page"] and _iou(sb, b["bbox"]) > 0.3 for sk, sp, sb in seen):
            continue
        md = block_markdown(b, annotate)
        if not md.strip():
            continue
        seen.append((k, b["page"], b["bbox"]))
        if annotate and b.get("verification") == "failed" and b.get("content") is not None:
            md = f"<!-- parseflow: low-confidence extraction ({b['block_id']}, {b['extractor']}) -->\n" + md
        out.append(md)
    if annotate and doc.get("warnings"):
        out.append("\n".join(f"<!-- warning: {w} -->" for w in doc["warnings"]))
    text = "\n\n".join(out).strip()
    # an empty document yields an empty string (not a lone newline): nothing should look like content
    return _nfc(text + "\n") if text else ""
