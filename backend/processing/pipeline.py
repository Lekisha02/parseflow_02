"""DETECT -> ANALYZE -> ROUTE -> EXTRACT -> ASSEMBLE -> VERIFY -> OUTPUT. Every processor emits Digital Twin blocks.

Routing is per page and per region, cheapest suitable method first:
  digital page with a trustworthy text layer   -> native text + layout analysis (no rendering needed)
  page whose text layer is broken/implausible  -> adaptive OCR (text layer kept as evidence)
  scanned page with an embedded OCR layer      -> embedded layer when it is plausible, else Tesseract
  scanned page                                 -> adaptive OCR
  tables                                       -> ruled-grid or aligned-text reconstruction
  equations                                    -> LaTeX rebuilt from glyph geometry (digital) / optional local
                                                  recognizer (scans) / OCR text kept + flagged
  figures, handwriting, diagrams               -> vision model when configured, else OCR labels / flagged
Bounding boxes are in page-image pixels (PDF points * Z), the same space as v1 and the frontend viewer.
"""
import json
import os
import re
import statistics
import time
from contextlib import contextmanager

import numpy as np
import pymupdf
from PIL import Image
from scipy import ndimage as ndi

from . import charts, diagrams, engines, fontfix, headers_footers, layout, mathtex, ocr, recognizers, tables, textlayer, verifier, vision
from . import markdown as mdgen
from .schema import Z, make_block, overlap_frac, scale, union, verdict
from . import formats

STAGES = ["DETECT", "ANALYZE", "ROUTE", "EXTRACT", "ASSEMBLE", "VERIFY", "OUTPUT"]
VIS_EXT = {"handwritten_text": "vision_handwriting", "handwritten_code": "vision_handwriting", "paragraph": "vision_handwriting", "heading": "vision_handwriting",
           "list": "vision_handwriting", "equation": "math_vision", "diagram": "vision_diagram", "figure": "vision_diagram", "chart": "vision_chart", "unknown": "vision_diagram", "table": "vision_table"}
PREFIX = {"table": "tbl", "chart": "chart", "diagram": "diag", "equation": "eq", "handwritten_text": "hand", "handwritten_code": "code", "figure": "fig",
          "unknown": "unk", "caption": "cap", "header": "hdr", "footer": "ftr", "page_number": "pgn", "margin": "mrg", "code": "code"}


class PFError(Exception):
    def __init__(self, code, title, message):
        super().__init__(message); self.code, self.title, self.message = code, title, message

    def dict(self):
        return dict(code=self.code, title=self.title, message=self.message)


class Progress:
    def __init__(self):
        self.stages = [dict(name=s, state="pending", detail="") for s in STAGES]
        self.status = "running"; self.error = None; self.timings = {}

    def set(self, name, state, detail=""):
        for s in self.stages:
            if s["name"] == name:
                s["state"], s["detail"] = state, detail or s["detail"]


@contextmanager
def _stage(prog, name, timings):
    prog.set(name, "running")
    t0 = time.perf_counter()
    yield
    timings[name] = round(timings.get(name, 0.0) + time.perf_counter() - t0, 4)


def _env_int(k, d):
    try:
        return int(os.getenv(k, str(d)))
    except ValueError:
        return d


def open_source(path, filename, kind=None):
    # the file on disk decides (a converted Word file is a PDF named after the original); the name is the fallback
    pext = path.lower().rsplit(".", 1)[-1] if "." in os.path.basename(path) else ""
    ext = pext if pext in ("pdf", "png", "jpg", "jpeg") else (filename.lower().rsplit(".", 1)[-1] if "." in filename else "")
    if kind in ("pdf", "image"):
        ext = "pdf" if kind == "pdf" else "png"
    if ext not in ("pdf", "png", "jpg", "jpeg"):
        raise PFError("PF-204", "Unsupported Document", "This file must be converted first (formats.prepare). Supported: " + formats.LABEL + ".")
    if os.path.getsize(path) == 0:
        raise PFError("PF-205", "Empty Document", "The file contains no data.")
    if ext == "pdf":
        try:
            d = pymupdf.open(path)
        except Exception:
            raise PFError("PF-201", "Corrupt Document", "The PDF could not be opened. It may be damaged or not a real PDF.")
        if d.needs_pass:
            raise PFError("PF-202", "Encrypted Document", "The PDF is password protected.")
        if d.page_count == 0:
            raise PFError("PF-205", "Empty Document", "The PDF has no pages.")
        return "pdf", d
    try:
        im = Image.open(path); im.load(); im = im.convert("RGB")
    except Exception:
        raise PFError("PF-201", "Corrupt Document", "The image could not be decoded.")
    if max(im.size) > 3400:
        im.thumbnail((3400, 3400))
    return "image", im


def page_count(path, filename):
    kind, src = open_source(path, filename, "pdf" if formats.sniff(path) == "pdf" else None)
    try:
        return src.page_count if kind == "pdf" else 1
    finally:
        if kind == "pdf":
            src.close()


# ---------------------------------------------------------------------------------------------
# Page analysis helpers
# ---------------------------------------------------------------------------------------------
def analyze_page(page):
    """Cheap page metrics (no rendering): text amount, hidden-text layer, image coverage, text-layer quality."""
    W, H = page.rect.width, page.rect.height
    segs, invisible = textlayer.read_segments(page)
    vis_text = " ".join(s.text for s in segs)
    chars = len(re.sub(r"\s", "", vis_text))
    cov, best_img_zoom = 0.0, 0.0
    try:
        for i in page.get_image_info():
            b = i["bbox"]
            a = max(0, min(b[2], W) - max(b[0], 0)) * max(0, min(b[3], H) - max(b[1], 0))
            cov += a
            bw = max(1e-3, b[2] - b[0])
            if a > 0.3 * W * H and i.get("width"):
                best_img_zoom = max(best_img_zoom, i["width"] / bw)
    except Exception:
        pass
    cov = min(1.0, cov / (W * H))
    q, qd = textlayer.text_quality(vis_text)
    hidden = ""
    if invisible > 20:
        hidden = page.get_text("text")
    return dict(segs=segs, chars=chars, invisible=invisible, image_coverage=round(cov, 2), text_quality=q, quality_detail=qd,
                img_zoom=best_img_zoom, hidden_text=hidden, W=W, H=H)


def route_page(a, ocr_ok):
    """Decide how to read a page. Returns (route, reason)."""
    if a["chars"] >= 30 and a["text_quality"] >= 0.6 and not (a["image_coverage"] > 0.85 and a["chars"] < 80):
        return "digital", "visible text layer passes quality checks"
    if 0 < a["chars"] < 30 and a["text_quality"] >= 0.6 and a["image_coverage"] < 0.3 and a["invisible"] <= 20:
        return "digital", "sparse page: short but trustworthy text layer and no large images"
    if a["chars"] >= 30 and a["text_quality"] < 0.6:
        return ("ocr" if ocr_ok else "digital"), f"visible text layer looks broken (quality {a['text_quality']:.2f})"
    if a["invisible"] > 20:
        hq, _ = textlayer.text_quality(a["hidden_text"])
        if hq >= 0.75 or not ocr_ok:
            return "embedded_ocr", f"scanned page with a plausible embedded OCR layer (quality {hq:.2f})"
        return "ocr", f"scanned page; embedded OCR layer implausible (quality {hq:.2f})"
    return ("ocr" if ocr_ok else "unreadable"), "no usable text layer"


def detect_ink_regions(img, ocr_boxes, max_side=1800):
    """Regions with ink that OCR could not explain (handwriting, diagrams, charts). Ruled lines are removed first.
    Large images are analysed on a downsampled copy (morphology cost grows with pixel count)."""
    k = min(1.0, max_side / max(img.size))
    if k < 1.0:
        small = img.convert("L").resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.BILINEAR)
        res = detect_ink_regions(small, [tuple(v * k for v in b) for b in ocr_boxes], max_side)
        return [tuple(min(lim, v / k) for v, lim in zip(r, (img.width, img.height, img.width, img.height))) for r in res]
    g = np.array(img.convert("L")); thr = min(170, np.median(g) * 0.65)
    mask = g < thr
    mask &= ~ndi.binary_dilation(ndi.binary_opening(mask, structure=np.ones((1, 90))) | ndi.binary_opening(mask, structure=np.ones((90, 1))), iterations=2)
    for (x0, y0, x1, y1) in ocr_boxes:
        mask[max(0, int(y0) - 3):int(y1) + 3, max(0, int(x0) - 3):int(x1) + 3] = False
    f = 4; h, w = (mask.shape[0] // f) * f, (mask.shape[1] // f) * f
    coarse = mask[:h, :w].reshape(h // f, f, w // f, f).max(axis=(1, 3))
    lab, n = ndi.label(ndi.binary_dilation(coarse, structure=np.ones((3, 3)), iterations=10))
    out = []
    for sl in ndi.find_objects(lab):
        y0, y1, x0, x1 = sl[0].start * f, sl[0].stop * f, sl[1].start * f, sl[1].stop * f
        ink = int(mask[y0:y1, x0:x1].sum()); bw, bh = x1 - x0, y1 - y0
        if ink < 400 or bw < 24 or bh < 24 or bw * bh > 0.9 * img.size[0] * img.size[1]:
            continue
        out.append((max(0, x0 - 8), max(0, y0 - 8), min(img.size[0], x1 + 8), min(img.size[1], y1 + 8)))
    return _merge_near(out, 0.03 * max(img.size))


def _merge_near(rects, gap):
    """Pieces of one picture (an illustration broken up by text drawn over it) become one region."""
    rects = list(rects)
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] - gap < b[2] and b[0] - gap < a[2] and a[1] - gap < b[3] and b[1] - gap < a[3]:
                    rects[i] = union(a, b); del rects[j]; merged = True; break
            if merged:
                break
    return rects


def figures(page, avoid, segs, drawings=None):
    """Image and vector-drawing regions large enough to be figures (merged first, filtered after)."""
    W, H = page.rect.width, page.rect.height
    rects, raster = [], []
    try:
        raster = [tuple(i["bbox"]) for i in page.get_image_info()]
        rects += raster
    except Exception:
        pass
    try:
        dr = drawings if drawings is not None else page.get_drawings()
        # page-sized backgrounds (a white or tinted page rectangle) would merge every drawing into one cluster
        dr = [d for d in dr if not (d.get("rect") is not None and (d["rect"].width * d["rect"].height) > 0.6 * W * H)]
        if dr:
            rects += [tuple(r) for r in page.cluster_drawings(drawings=dr)]
    except Exception:
        pass
    rects = [r for r in rects if (r[2] - r[0]) >= 3 and (r[3] - r[1]) >= 3 and not any(overlap_frac(r, a) > 0.5 for a in avoid)]
    if len(rects) > 300:            # extremely busy vector pages (maps, plots): one merged region per cluster pass
        rects = sorted(rects, key=lambda r: (r[1], r[0]))[:300]
    g, merged = 12, True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] - g < b[2] and b[0] - g < a[2] and a[1] - g < b[3] and b[1] - g < a[3]:
                    rects[i] = union(a, b); del rects[j]; merged = True; break
            if merged:
                break
    out = []
    for r in rects:
        w, h = r[2] - r[0], r[3] - r[1]
        if w < 40 or h < 30 or not (0.015 * W * H <= w * h <= 0.92 * W * H):
            continue
        # vector boxes that hold prose (cards, shaded sidebars, framed text) are text containers, not figures
        prose = sum(1 for s in segs if overlap_frac(s.bbox, r) > 0.8 and len(s.text) > 30 and len(s.text.split()) >= 5)
        is_raster = any(overlap_frac(r, ib) > 0.5 for ib in raster)
        if prose >= 3 and not is_raster:
            continue
        out.append(r)
    return out


class OcrLine:
    """An OCR text line shaped like textlayer.Segment (PDF-point coordinates), so layout/table/furniture code is shared."""

    def __init__(self, words, k):
        # words: list of dict(text, conf, box) in image px; k = px per PDF point
        ws = sorted(words, key=lambda w: w["box"][0])
        self.words = ws
        hs = [w["box"][3] - w["box"][1] for w in ws]
        h = statistics.median(hs) / k if hs else 10
        sz = max(4.0, 0.8 * h)
        self.chars = [textlayer.Char(w["text"] + " ", w["box"][0] / k, w["box"][1] / k, w["box"][2] / k, w["box"][3] / k,
                                     w["box"][0] / k, w["box"][3] / k, sz, "ocr", 0) for w in ws]
        self.bbox = (min(c.x0 for c in self.chars), min(c.y0 for c in self.chars), max(c.x1 for c in self.chars), max(c.y1 for c in self.chars))
        self.dir = (1.0, 0.0)
        self.gaps = []
        self.conf = sum(w["conf"] for w in ws) / len(ws)
        self._size = sz
        self.text = " ".join(w["text"] for w in ws)
        self.baseline = statistics.median(c.oy for c in self.chars)
        self.bold = self.italic = self.mono = False
        self.horizontal = True

    @property
    def size(self):
        return self._size

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]


_GOOD = re.compile(r"^[^\W\d_]{3,}[.,;:!?)\]'\"]*$|^\(?\d[\d,.:/%-]*\)?[.,;:]?$", re.U)


def ocr_lines(words, k, min_conf=0.25):
    """Group OCR words into lines and drop noise: texture, photos and blank paper produce short,
    low-confidence, implausible 'words'. A line is kept when it carries real evidence of text."""
    groups = {}
    for w in words:
        if w["conf"] < min_conf and not re.search(r"[A-Za-z]{3,}", w["text"]):
            continue
        if not re.search(r"\w", w["text"]) and w["conf"] < 0.5:
            continue
        groups.setdefault((w["block"], w["par"], w["line"]), []).append(w)
    out = []
    for ws in groups.values():
        good = [w for w in ws if _GOOD.match(w["text"]) and w["conf"] >= 0.6]
        mean = sum(w["conf"] for w in ws) / len(ws)
        plaus = sum(1 for w in ws if _GOOD.match(w["text"])) / len(ws)
        formula = mean >= 0.6 and any(re.search(r"[=<>≤≥≈]", w["text"]) for w in ws) and any(re.search(r"\w", w["text"]) for w in ws)
        if len(good) >= 2 or (len(good) == 1 and (len(ws) <= 2 and mean >= 0.7)) or (mean >= 0.8 and plaus >= 0.5) or formula:
            out.append(OcrLine(ws, k))
    return out


# ---------------------------------------------------------------------------------------------
# Text with inline math
# ---------------------------------------------------------------------------------------------
def _visual_lines(lines):
    """Merge small attached segments (sub/superscripts split off by the PDF) into their host line."""
    lines = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))
    hosts = []
    for l in lines:
        h = None
        for hl in hosts:
            off = abs(l.baseline - hl["base"])
            script_like = (len(l.text) <= 15 or (l.bbox[2] - l.bbox[0]) < 0.35 * (hl["bbox"][2] - hl["bbox"][0])) and 0.1 * hl["size"] < off < 0.75 * hl["size"]
            if l.size < 0.85 * hl["size"] and l.bbox[1] < hl["bbox"][3] and l.bbox[3] > hl["bbox"][1] and script_like:
                h = hl; break
            if abs(l.baseline - hl["base"]) < 0.35 * max(l.size, hl["size"]):
                h = hl; break
        if h is None:
            hosts.append(dict(chars=list(getattr(l, "chars", [])), size=l.size, base=l.baseline, bbox=l.bbox, segs=[l]))
        else:
            h["chars"] += list(getattr(l, "chars", [])); h["segs"].append(l)
            h["bbox"] = union(h["bbox"], l.bbox)
    return hosts


def para_text(para, hrules, use_math=True):
    """Return (text, raw_text, has_math, transforms) for a Para of Segments / OcrLines."""
    if isinstance(para.lines[0], OcrLine):
        raw = [l.text for l in para.lines]
        return None, raw, False, []
    vlines, raw_lines, has_math = [], [], False
    for vl in _visual_lines(para.lines):
        chars = sorted(vl["chars"], key=lambda c: c.x0)
        raw_lines.append(textlayer._chars_to_text(chars))
        if not use_math:
            vlines.append(raw_lines[-1]); continue
        parts = []
        for is_m, cs in mathtex.inline_runs(chars):
            if is_m:
                lx, conf, _ = mathtex.build_latex(cs, [r for r in hrules if any(abs(r[1] - c.oy) < 2 * c.size and r[0] - 2 <= c.x0 <= r[2] + 2 for c in cs)])
                if lx and conf >= 0.5:
                    parts.append(" \\(" + lx + "\\) "); has_math = True; continue
            parts.append(textlayer._chars_to_text(cs))
        vlines.append(re.sub(r"\s+", " ", "".join(parts)).strip())
    return vlines, raw_lines, has_math, []


# ---------------------------------------------------------------------------------------------
# Display equations on digital pages
# ---------------------------------------------------------------------------------------------
def display_equations(segs, hrules, body_size, col_ranges):
    """Group math-dense, isolated lines (plus their scripts/limits and fraction rules) into equation regions."""
    cand = []
    for s in segs:
        if not s.horizontal:
            continue
        strong, weak, text = mathtex.line_math_profile(s.chars)
        if strong == 0:
            continue
        if text == 0:
            cand.append(s); continue
        if text <= 1 and strong >= 2:
            # one connective word ('where', 'and') is allowed only on an indented line (displayed, not prose)
            ov = [o.bbox[0] for o in segs if len(o.text) >= 40 and min(o.bbox[2], s.bbox[2]) - max(o.bbox[0], s.bbox[0]) > 0]
            margin = min(ov) if ov else min(o.bbox[0] for o in segs)
            if s.bbox[0] > margin + 2.0 * body_size:
                cand.append(s)
    # display-size delimiters and integrals: everything inside their vertical span on that display belongs to the
    # formula (matrix rows like 'a11 a12 ... a1n' carry no math symbol of their own)
    talls = [c for s in segs if s.horizontal for c in s.chars if mathtex.is_tall(c)]
    if talls:
        have = {id(x) for x in cand}
        for s in segs:
            if not s.horizontal or id(s) in have or len(s.text) > 80:
                continue
            cy = (s.bbox[1] + s.bbox[3]) / 2
            for c in talls:
                if c.y0 - 0.2 * c.size <= cy <= c.y1 + 0.2 * c.size and c.x0 - 30 * c.size <= s.bbox[0] <= c.x1 + 30 * c.size:
                    st, wk, tx = mathtex.line_math_profile(s.chars)
                    if tx == 0:
                        cand.append(s); have.add(id(s))
                    break
    if not cand:
        return []
    # group candidates that belong to one display (same band, close horizontally), then attach nearby
    # math pieces (limits, scripts, fraction parts) - each piece to exactly one group
    groups = []
    for s in sorted(cand, key=lambda s: (s.bbox[1], s.bbox[0])):
        sz = max(s.size, 6)
        best = None
        for g in groups:
            gb = g["bbox"]
            vgap = max(s.bbox[1] - gb[3], gb[1] - s.bbox[3])
            hgap = max(s.bbox[0] - gb[2], gb[0] - s.bbox[2])
            if vgap < 0.8 * sz and hgap < 6 * sz:
                best = g; break
        if best:
            best["segs"].append(s); best["bbox"] = union(best["bbox"], s.bbox)
        else:
            groups.append(dict(segs=[s], bbox=s.bbox))
    # transitive merge of groups on the same display band
    def _close(a, b):
        sz = max(6, body_size)
        vov = min(a[3], b[3]) - max(a[1], b[1])
        hgap = max(b[0] - a[2], a[0] - b[2])
        vgap = max(b[1] - a[3], a[1] - b[3])
        if vov > 0.5 * min(a[3] - a[1], b[3] - b[1]):
            return hgap < 12 * sz
        return vgap < 0.8 * sz and hgap < 2 * sz
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _close(groups[i]["bbox"], groups[j]["bbox"]):
                    groups[i]["segs"] += groups[j]["segs"]; groups[i]["bbox"] = union(groups[i]["bbox"], groups[j]["bbox"])
                    del groups[j]; merged = True; break
            if merged:
                break
    taken = {id(x) for g in groups for x in g["segs"]}
    pool = [s for s in segs if s.horizontal and id(s) not in taken and len(s.text) < 80]
    changed = True
    while changed:
        changed = False
        for s in list(pool):
            sz = max(s.size, 6)
            st, wk, tx = mathtex.line_math_profile(s.chars)
            fits = []
            for g in groups:
                gb = g["bbox"]
                grown = (gb[0] - 1.5 * sz, gb[1] - 1.3 * sz, gb[2] + 1.5 * sz, gb[3] + 1.3 * sz)
                row = (min(gb[3], s.bbox[3]) - max(gb[1], s.bbox[1])) / max(1e-3, s.bbox[3] - s.bbox[1])
                hgap = max(s.bbox[0] - gb[2], gb[0] - s.bbox[2])
                if (tx <= 1 and overlap_frac(s.bbox, grown) > 0.6) or (st >= 1 and tx <= 2 and row >= 0.6 and hgap < 3 * sz):
                    fits.append((overlap_frac(s.bbox, gb) + row, g))
            if fits:
                g = max(fits, key=lambda t: t[0])[1]
                g["segs"].append(s); g["bbox"] = union(g["bbox"], s.bbox)
                pool.remove(s); changed = True
    # an equation number set far to the right ('(28.8)') joins the display it labels
    for g in list(groups):
        txt = "".join(c.c for s in g["segs"] for c in s.chars if c.c.strip())
        if not mathtex.EQNO.match(txt):
            continue
        cy = (g["bbox"][1] + g["bbox"][3]) / 2
        host = [h for h in groups if h is not g and h["bbox"][2] <= g["bbox"][0] and h["bbox"][1] - body_size <= cy <= h["bbox"][3] + body_size]
        if host:
            h = max(host, key=lambda h: h["bbox"][2])
            h["segs"] += g["segs"]; h["bbox"] = union(h["bbox"], g["bbox"]); groups.remove(g)
    out = []
    for g in groups:
        box = g["bbox"]
        chars = [c for s in g["segs"] for c in s.chars]
        rules = [r for r in hrules if box[0] - 3 <= r[0] and r[2] <= box[2] + 3 and box[1] - 2 <= r[1] <= box[3] + 2]
        out.append(dict(segs=g["segs"], bbox=box, chars=chars, rules=rules))
    return out


def _split_rows(chars, rules, size):
    """Split a display region into separate equation lines at clear vertical gaps in the ink
    (glyph boxes + fraction rules), so numerators/denominators stay with their own line."""
    vis = [c for c in chars if c.c.strip()]
    if not vis:
        return []
    def vbox(c):
        if mathtex.is_tall(c):
            return (c.y0, c.y1)
        if c.c in mathtex.BIGOP_CHARS:          # room for limits above and below a big operator
            return (min(c.y0, c.oy - 0.8 * c.size) - 0.9 * c.size, max(c.y1, c.oy) + 0.9 * c.size)
        return (c.oy - 0.72 * c.size, c.oy + 0.22 * c.size)
    boxes = [vbox(c) for c in vis] + [(r[1] - 0.5, r[1] + 0.5) for r in rules]
    boxes.sort()
    cuts, end = [], None
    for a, b in boxes:
        if end is not None and a - end > 0.35 * size:
            cuts.append((end + a) / 2)
        end = b if end is None else max(end, b)
    if not cuts:
        return [(vis, list(rules))]
    edges = [-1e9] + cuts + [1e9]
    rows = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        rc = [c for c in vis if lo <= ((c.y0 + c.y1) / 2 if mathtex.is_tall(c) else c.oy - 0.25 * c.size) < hi]
        rr = [r for r in rules if lo <= r[1] < hi]
        if rc:
            rows.append((rc, rr))
    return rows


# ---------------------------------------------------------------------------------------------
# Vision / fallback for visual regions (unchanged contract from v1)
# ---------------------------------------------------------------------------------------------
def _raster_diagram(crop, bb, state):
    """Diagram structure read from the pixels of a picture region (scans, photos, screenshots, embedded images)."""
    if os.getenv("PARSEFLOW_DIAGRAMS", "1") == "0" or state.get("raster_left", 1) <= 0:
        return None
    if "raster_left" not in state:
        state["raster_left"] = _env_int("PARSEFLOW_RASTER_DIAGRAMS", 30)
    if crop.width < 80 or crop.height < 50:
        return None
    # a time budget per document, so picture-heavy documents (slides of photos) stay fast
    budget = float(os.getenv("PARSEFLOW_RASTER_SECONDS", "20"))
    if state.get("raster_seconds", 0.0) >= budget:
        return None
    state["raster_left"] -= 1
    t0 = time.perf_counter()
    try:
        from . import diagrams_raster
        return diagrams_raster.extract(crop, origin=(bb[0], bb[1]), zoom=Z)
    except Exception:
        return None
    finally:
        state["raster_seconds"] = state.get("raster_seconds", 0.0) + time.perf_counter() - t0


def _raster_block(st, n, bb):
    b = make_block("", "diagram", n, scale(bb), round(max(0.5, diagrams.estimate(st) - 0.15), 2), "image geometry + OCR (estimated)",
                   "raster_diagram", None, structure=st, structure_text=diagrams.describe(st))
    b["content"] = b["structure_text"]
    b["warnings"].append("Diagram structure read from the image (outlines, lines and arrowheads in the pixels; text by OCR). "
                         "Faint or touching lines and misread labels are possible: check against the source.")
    return b


def vision_blocks(crop_img, bbox, hint, state, tag):
    """Specialised extraction for one visual region. Falls back to a flagged, evidence-preserving block."""
    cx0, cy0, cx1, cy1 = bbox; cw, ch = cx1 - cx0, cy1 - cy0
    err = None
    if state["vision_left"] > 0 and vision.available():
        state["vision_left"] -= 1; state["vision_used"] += 1
        items, err = vision.analyze_crop(crop_img)
        blocks = []
        for it in items:
            t = it.get("type") if it.get("type") in vision.VALID else "unknown"
            rows = None
            if t == "table":
                rr = it.get("rows")
                rows = [[str(c or "").strip() for c in r] for r in rr if isinstance(r, list)] if isinstance(rr, list) else None
                rows = [r for r in (rows or []) if any(r)]
                if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
                    t, rows = "unknown", None
            content = it.get("content") or (it.get("description") if t in ("diagram", "chart", "figure", "unknown") else None)
            if t == "table":
                content = f"Table with {len(rows)} rows"
            try:
                conf = max(0.0, min(0.95, float(it.get("confidence", 0.5))))
            except Exception:
                conf = 0.5
            warns, bb = [], bbox
            raw = it.get("bbox")
            if isinstance(raw, list) and len(raw) == 4 and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in raw) and raw[2] - raw[0] > 8 and raw[3] - raw[1] > 8:
                bb = (cx0 + raw[0] / 1000 * cw, cy0 + raw[1] / 1000 * ch, cx0 + raw[2] / 1000 * cw, cy0 + raw[3] / 1000 * ch)
            elif len(items) > 1:
                warns.append("Model gave no valid sub-region box; using the whole analysed region.")
            if not content:
                conf = min(conf, 0.45); warns.append("Low confidence — review source. Content withheld rather than guessed.")
            ch_ = it.get("chart") if isinstance(it.get("chart"), dict) else {}
            data = ch_.get("data") if isinstance(ch_.get("data"), list) and ch_.get("data") else None
            b = make_block("", t, 0, bb, conf, "model (self-reported)", VIS_EXT[t], content, latex=it.get("latex") if t == "equation" else None,
                           elements=it.get("elements") or None, relationships=[str(r) for r in it.get("relationships") or []] if t == "diagram" else None,
                           chart_type=ch_.get("chart_type") if t == "chart" else None, title=ch_.get("title") if t == "chart" else None,
                           data=data if t == "chart" else None, rows=rows, axes=ch_.get("axes") if t == "chart" else None, legend=ch_.get("legend") if t == "chart" else None)
            if t == "chart" and not data:
                b["warnings"].append("Chart values not extracted — trend described only; no values invented.")
            if t == "table" and rows:
                tb = tables.from_rows(rows, "vision", conf)
                b["cells"] = tb.to_dict()["cells"]; b["table_method"] = "vision"; b["header_rows"] = 1
            b["warnings"] += warns; blocks.append(b)
        if blocks:
            return blocks
        err = err or "vision model returned no regions"
    # optional local models (never downloaded; see engines.py): handwriting for unexplained ink, chart data for figures
    if tag == "ink" and engines.handwriting_ready():
        r = engines.read_handwriting(crop_img)
        if r and r[0]:
            b = make_block("", "handwritten_text", 0, bbox, min(0.7, r[1]), "model (self-reported)", "handwriting_model", r[0],
                           preserved_image=True)
            b["warnings"].append(f"Read by the local handwriting model '{engines.handwriting_name()}'; review against the image.")
            return [b]
    if engines.chart_ready() and hint in ("figure", "unknown"):
        r = engines.read_chart(crop_img)
        if r and r.get("rows"):
            rows = r["rows"]
            b = make_block("", "chart", 0, bbox, 0.55, "model (self-reported)", "chart_model", f"Chart with {len(rows) - 1} data rows",
                           chart_type=r.get("chart_type"), rows=rows, data=[dict(zip(rows[0], x)) for x in rows[1:]])
            tb = tables.from_rows(rows, "chart_model", 0.55)
            b["cells"] = tb.to_dict()["cells"]; b["header_rows"] = 1
            b["warnings"].append(f"Values read from the chart image by the local model '{engines.chart_name()}'; check them.")
            return [b]
    why = f"Vision extraction failed ({err}) — region preserved, flagged." if err else "Visual analysis unavailable: no VISION_API_KEY configured — region preserved, flagged."
    oc, ot = (0, "")
    if state.get("ocr_ok") and state["region_ocr_left"] > 0 and tag != "ink":   # ink regions: page already OCR'd
        state["region_ocr_left"] -= 1
        try:
            oc, ot = ocr.quick_conf(crop_img)
        except Exception:
            oc, ot = 0, ""
    b = make_block("", "figure" if hint == "figure" else "unknown", 0, bbox, 0.3, "heuristic (estimated)", "fallback", None,
                   ocr_text=ot if oc >= 0.6 and len(ot) >= 3 else None, preserved_image=True)
    b["warnings"] += [why] + (["Legible printed text found by OCR (see ocr_text); not treated as the region's meaning."] if b.get("ocr_text") else [])
    return [b]


# ---------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------
def process(path, filename, doc_id, out_dir, prog, save_images=True, include_furniture=None):
    path, filename, out_dir = str(path), str(filename), str(out_dir)
    timings = prog.timings if hasattr(prog, "timings") else {}
    t_all = time.perf_counter()
    state = dict(vision_left=_env_int("MAX_VISION_REGIONS", 12), vision_used=0, region_ocr_left=_env_int("MAX_REGION_OCR", 6))
    max_pages = _env_int("MAX_PAGES", 0); warnings = []          # MAX_PAGES 0 = no limit
    ocr_ok = bool(ocr.available()); state["ocr_ok"] = ocr_ok
    os.makedirs(out_dir, exist_ok=True)

    # DETECT ---------------------------------------------------------------------------------
    with _stage(prog, "DETECT", timings):
        conv = formats.conversion_info(out_dir, path)          # already converted at upload
        kind_hint = conv.get("kind") if conv else None
        if conv is None:
            try:
                prep = formats.prepare(path, filename, out_dir, doc_id)
            except formats.ConvertError as e:
                raise PFError(e.code, e.title, e.message)
            path, kind_hint = prep.path, prep.kind
            conv = dict(fmt=prep.fmt, method=prep.method, notes=prep.notes) if prep.method != "as-is" else None
        if conv:
            warnings.extend(conv.get("notes") or [])
        kind, src = open_source(path, filename, kind_hint)
        n_pages = src.page_count if kind == "pdf" else 1
        if max_pages and n_pages > max_pages:
            warnings.append(f"Document has {n_pages} pages; only the first {max_pages} were processed."); n_pages = max_pages
    prog.set("DETECT", "done", f"{kind.upper()} · {n_pages} page{'s' if n_pages > 1 else ''}")

    # ANALYZE / ROUTE / EXTRACT, in slices of pages -------------------------------------------
    # Pages are independent until ASSEMBLE, so a long document is handled CHUNK pages at a time: each slice is
    # analysed, routed and extracted, then its glyph data and page images are released. Memory stays bounded by the
    # slice, not the document (a 400-page report no longer holds 400 pages of glyphs and bitmaps at once).
    chunk = max(1, _env_int("PARSEFLOW_PAGE_CHUNK", 25))
    ocr_workers = max(1, _env_int("PARSEFLOW_OCR_WORKERS", min(4, os.cpu_count() or 1)))
    page_ctx, per_page = [], {}
    cnt = dict.fromkeys(("digital", "embedded_ocr", "ocr", "unreadable", "error"), 0)
    n_regions = n_blocks = 0
    of = f" of {n_pages}" if n_pages > 1 else ""
    # checkpoints: every finished slice is saved, so a stopped or crashed job resumes where it was
    ck = _Checkpoint(out_dir, path, n_pages) if kind == "pdf" and n_pages > 1 else None
    first = 0
    if ck is not None:
        rs = ck.restore()
        if rs:
            first = rs["end"]
            page_ctx.extend(rs["pages"]); per_page.update(rs["blocks"]); warnings.extend(rs["warnings"])
            state.update(rs["state"]); cnt.update(rs["cnt"]); n_regions, n_blocks = rs["n_regions"], rs["n_blocks"]
            try:
                fontfix.for_doc(src).import_state(rs.get("fonts") or {})
            except Exception:
                pass
            prog.set("ANALYZE", "running", f"resuming at page {first + 1}{of}")
    for start in range(first, n_pages, chunk):
        w0 = len(warnings)
        sl = []
        with _stage(prog, "ANALYZE", timings):
            for i in range(start, min(n_pages, start + chunk)):
                prog.set("ANALYZE", "running", f"page {i + 1}{of}")
                sl.append(_analyze_one(i, kind, src, ocr_ok, warnings))
        n_regions += sum(len(c["items"]) for c in sl)
        with _stage(prog, "ROUTE", timings):
            for c in sl:
                cnt[c["route"]] = cnt.get(c["route"], 0) + 1
        prog.set("ROUTE", "running", " · ".join(f"{k} {v}" for k, v in cnt.items() if v))
        with _stage(prog, "EXTRACT", timings):
            # scanned pages: OCR runs on several pages at once (Tesseract is a separate process per call), a few pages
            # ahead of the page being assembled; rendering stays on this thread (the PDF library is not thread-safe)
            ocr_q = [c for c in sl if c["route"] == "ocr"]
            pool = None
            if len(ocr_q) > 1 and ocr_workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                pool = ThreadPoolExecutor(ocr_workers)
            ahead = 0
            for ctx in sl:
                if pool is not None:
                    while ahead < len(ocr_q) and sum(1 for c in ocr_q[:ahead] if c.get("ocr_future") is not None) < ocr_workers:
                        c = ocr_q[ahead]; ahead += 1
                        try:
                            c["img"] = _render(c, kind, src)
                            oimg, zoom = _ocr_input(c)
                            c["ocr_future"], c["ocr_zoom"] = pool.submit(ocr.ocr_image, oimg), zoom
                        except Exception:
                            c.pop("ocr_future", None)
                prog.set("EXTRACT", "running", f"page {ctx['n']}{of}")
                _test_exit(ctx["n"], out_dir)
                per_page[ctx["n"]] = _extract_one(ctx, kind, src, out_dir, save_images, warnings, state)
                n_blocks += len(per_page[ctx["n"]])
                _release(ctx, kind)
            if pool is not None:
                pool.shutdown(wait=True)
        page_ctx.extend(sl)
        if ck is not None:
            try:
                fonts = fontfix.for_doc(src).export_state()
            except Exception:
                fonts = {}
            ck.save(start, min(n_pages, start + chunk), sl, per_page, warnings[w0:], state, cnt, n_regions, n_blocks, fonts)
    prog.set("ANALYZE", "done", f"{n_regions} regions identified on digital pages")
    prog.set("ROUTE", "done", " · ".join(f"{k} {v}" for k, v in cnt.items() if v))
    prog.set("EXTRACT", "done", f"{n_blocks} blocks")

    # ASSEMBLE (furniture classification across the document, ids, captions) ----------------------
    blocks = []
    with _stage(prog, "ASSEMBLE", timings):
        body_sizes = [ctx.get("body_size", 10) for ctx in page_ctx]
        body = statistics.median(body_sizes) if body_sizes else 10
        fpages = []
        for ctx in page_ctx:
            items = []
            for k, b in enumerate(per_page.get(ctx["n"], [])):
                if b["type"] in ("paragraph", "heading", "list", "caption", "code") and b.get("content"):
                    bb = tuple(v / Z for v in b["bbox"])
                    items.append(dict(id=(ctx["n"], k), bbox=bb, text=b.get("raw_text") or b["content"], size=b.get("font_size", body),
                                      rotated=b.get("rotated", False), kind="text"))
            fpages.append(dict(width=ctx["W"], height=ctx["H"], items=items))
        fur = headers_footers.classify(fpages, body)
        for ctx in page_ctx:
            n = ctx["n"]
            bs = per_page.get(n, [])
            for k, b in enumerate(bs):
                f = fur.get((n, k))
                if f:
                    b["furniture"] = f
                    if f["excluded"] or f["score"] >= headers_footers.REMOVE:
                        b["role"] = f["role"]
            ctr = {}
            for ro_i, b in enumerate(sorted(bs, key=lambda b: b["reading_order"]), 1):
                b["reading_order"] = ro_i
                pre = PREFIX.get(b.get("role")) if b.get("role") else None
                pre = pre or PREFIX.get(b["type"]) or ("ocr" if b["extractor"] == "ocr" else "txt")
                if b["type"] in ("heading", "paragraph", "list") and not b.get("role") and b["extractor"] not in ("ocr", "digital_text_parser"):
                    pre = "hand"
                ctr[pre] = ctr.get(pre, 0) + 1; b["block_id"] = f"{pre}_{n}_{ctr[pre]:02d}"; b["page"] = n
                blocks.append(b)
            targets = [b for b in bs if b["type"] in ("table", "chart", "diagram", "figure")]
            for c in [b for b in bs if b["type"] == "caption"]:
                near = sorted(targets, key=lambda t: min(abs(t["bbox"][1] - c["bbox"][3]), abs(c["bbox"][1] - t["bbox"][3])))
                if near and min(abs(near[0]["bbox"][1] - c["bbox"][3]), abs(c["bbox"][1] - near[0]["bbox"][3])) < 160:
                    c["relationships"].append("caption_of:" + near[0]["block_id"]); near[0]["relationships"].append("caption:" + c["block_id"])
        excluded = sum(1 for b in blocks if (b.get("furniture") or {}).get("excluded"))
        joined = tables.merge_page_tables(blocks, {ctx["n"]: ctx["W"] for ctx in page_ctx})
        if joined:
            warnings.append(f"{joined} table(s) continued across a page break were joined (see 'merged' on the first part).")
    prog.set("ASSEMBLE", "done", f"{len(blocks)} blocks ordered · {excluded} running header/footer element(s) set aside")

    # VERIFY ---------------------------------------------------------------------------------
    with _stage(prog, "VERIFY", timings):
        by_page = {}
        for b in blocks:
            by_page.setdefault(b["page"], []).append(b)
        for ctx in page_ctx:
            n = ctx["n"]; a = ctx.get("a", {})
            if len(page_ctx) > 1:
                prog.set("VERIFY", "running", f"page {n}{of}")
            ptxt = ""
            if kind == "pdf" and ctx["route"] != "error":
                try:
                    ptxt = src[n - 1].get_text("text")
                except Exception:
                    ptxt = ""
            info = dict(width=ctx["W"] * Z, height=ctx["H"] * Z, norm_text=re.sub(r"[^a-z0-9]+", "", ptxt.lower()),
                        text_quality=a.get("text_quality", 1.0) if ctx["route"] == "digital" else textlayer.text_quality(a.get("hidden_text", ""))[0],
                        hidden_text_raw=a.get("hidden_text", "") if ctx["route"] == "ocr" else "",
                        hidden_text_norm=re.sub(r"[^a-z0-9]+", "", (a.get("hidden_text", "") if ctx["route"] == "ocr" else "").lower()), ocr_ok=ocr_ok)
            crop = _lazy_crop(ctx, kind, src, out_dir)
            for b in by_page.get(n, []):
                verifier.verify(b, info, crop)
                txt = (b.get("content") or "") + " " + " ".join(" ".join(map(str, r)) for r in (b.get("rows") or []))
                if fontfix.garbled(txt):
                    b["warnings"].append("Some characters come from a font whose character map is broken and could not be "
                                         "repaired (e.g. '=' shown as a letter); check View Source.")
                    if b.get("verification") == "verified":
                        b["verification"] = "review"
        flagged = sum(1 for b in blocks if b["verification"] != "verified")
    prog.set("VERIFY", "done", f"{len(blocks) - flagged} verified · {flagged} flagged")

    # OUTPUT ---------------------------------------------------------------------------------
    with _stage(prog, "OUTPUT", timings):
        if not vision.available() and any(b["extractor"] == "fallback" for b in blocks):
            warnings.append("Vision API not configured: visual/handwritten regions were preserved and flagged, not interpreted.")
        if not blocks:
            warnings.append("No content could be extracted from this document.")
        pages = []
        for ctx in page_ctx:
            pg = dict(n=ctx["n"], kind=_kind_label(ctx), width=round(ctx["W"] * Z), height=round(ctx["H"] * Z),
                      flags=ctx.get("flags", {}), chars=ctx.get("a", {}).get("chars", 0), image_coverage=ctx.get("a", {}).get("image_coverage", 0),
                      route=ctx["route"], route_reason=ctx["reason"], text_quality=ctx.get("a", {}).get("text_quality"),
                      ocr=ctx.get("ocr_meta"), seconds=round(ctx["per_page_s"], 4))
            pages.append(pg)
        vis = ("chart", "figure", "diagram", "equation", "handwritten_text", "handwritten_code", "unknown")
        lanes = dict(local=sum(1 for b in blocks if b["extractor"] in ("digital_text_parser", "table_engine")), ocr=sum(1 for b in blocks if b["extractor"] == "ocr"),
                     vision=state["vision_used"], flagged=sum(b["extractor"] == "fallback" for b in blocks))
        timings["total"] = round(time.perf_counter() - t_all, 4)
        if conv:
            formats.annotate(blocks, conv)
        doc = dict(document_id=doc_id, filename=filename, format=kind, real=True, schema_version=2, pages=pages, blocks=blocks, warnings=warnings,
                   processing_status="complete",
                   router=[dict(block_id=b["block_id"], region_type=b["type"], extractor=b["extractor"]) for b in blocks],
                   stats=dict(pages=n_pages, blocks=len(blocks), tables=sum(b["type"] == "table" for b in blocks), visuals=sum(b["type"] in vis for b in blocks),
                              verified=len(blocks) - flagged, flagged=flagged, furniture_excluded=excluded, lanes=lanes),
                   engines=dict(ocr=ocr.available(), vision=vision.info(), math_recognizer=recognizers.info()),
                   timings=timings, config=dict(remove_furniture=headers_footers.enabled() if include_furniture is None else not include_furniture))
        if conv:
            doc["source"] = dict(format=conv.get("fmt"), converted_with=conv.get("method"))
        doc["markdown"] = mdgen.render(doc, include_furniture=bool(include_furniture))
        with open(os.path.join(out_dir, "document.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=True)       # ASCII-escaped: valid even if a PDF carried invalid code points
    prog.set("OUTPUT", "done", "Markdown + JSON ready")
    if ck is not None:
        ck.clear()
    if kind == "pdf":
        src.close()
    return doc


def _plain_text_fallback(ctx, n, ex):
    """A page whose full analysis failed keeps its text: the text layer's own paragraphs, in reading order, marked
    for review (no tables / maths reconstruction on this page). Only a page without usable text becomes one flagged
    image region."""
    out = []
    pg = ctx.get("pdf_page")
    try:
        if pg is not None and ctx.get("route") == "digital":
            for k, blk in enumerate(pg.get_text("blocks", sort=True)):
                x0, y0, x1, y1, txt = blk[:5]
                txt = re.sub(r"\s*\n\s*", " ", txt or "").strip()
                if blk[6] != 0 or not txt:
                    continue
                b = make_block("", "paragraph", n, scale((x0, y0, x1, y1)), 0.6, "text layer (fallback)", "digital_text_parser", txt,
                               raw_text=txt)
                b["reading_order"] = k + 1
                b["warnings"].append(f"The full analysis of this page failed ({type(ex).__name__}); its text is kept as plain "
                                     "paragraphs (tables and maths not reconstructed on this page).")
                out.append(b)
    except Exception:
        out = []
    if not out:
        b = make_block("", "unknown", n, scale((0, 0, ctx["W"], ctx["H"])), 0.0, "heuristic (estimated)", "fallback", None, preserved_image=True)
        b["warnings"].append(f"Extractor error ({type(ex).__name__}) — page preserved, flagged.")
        out = [b]
    return out


def _test_exit(n, out_dir):
    """Test hook (off unless PARSEFLOW_TEST_EXIT_AT_PAGE is set): the worker process dies once at that page, as an
    out-of-memory kill would, so crash recovery can be tested end to end."""
    t = os.getenv("PARSEFLOW_TEST_EXIT_AT_PAGE")
    if t and t == str(n):
        mark = os.path.join(out_dir, ".test_exited")
        if not os.path.exists(mark):
            open(mark, "w").close()
            os._exit(137)


class _Checkpoint:
    """out_dir/checkpoint/: one JSON file per finished slice of pages (their blocks and the little per-page data
    ASSEMBLE / VERIFY / OUTPUT need). Valid only for the same source file and page count. Off with
    PARSEFLOW_CHECKPOINT=0."""
    PAGE_KEYS = ("n", "W", "H", "body_size", "route", "reason", "flags", "ocr_meta", "per_page_s", "had_img")
    A_KEYS = ("text_quality", "hidden_text", "chars", "image_coverage")

    def __init__(self, out_dir, src_path, n_pages):
        self.on = os.getenv("PARSEFLOW_CHECKPOINT", "1").lower() not in ("0", "false", "no", "off")
        self.dir = os.path.join(out_dir, "checkpoint")
        try:
            st = os.stat(src_path)
            self.key = f"{st.st_size}:{int(st.st_mtime)}:{n_pages}:{_CK_VERSION}"
        except OSError:
            self.on = False

    def _slim(self, ctx):
        d = {k: ctx.get(k) for k in self.PAGE_KEYS if k in ctx}
        a = ctx.get("a") or {}
        d["a"] = {k: a.get(k) for k in self.A_KEYS if k in a}
        d["img"] = None
        return d

    def save(self, start, end, sl, per_page, warnings, state, cnt, n_regions, n_blocks, fonts=None):
        if not self.on:
            return
        try:
            os.makedirs(self.dir, exist_ok=True)
            man = os.path.join(self.dir, "manifest.json")
            if not os.path.exists(man):
                with open(man, "w") as f:
                    json.dump(dict(key=self.key), f)
            data = dict(start=start, end=end, pages=[self._slim(c) for c in sl],
                        blocks={str(c["n"]): per_page.get(c["n"], []) for c in sl}, warnings=list(warnings),
                        state={k: v for k, v in state.items() if isinstance(v, (int, float, str, bool, type(None)))},
                        cnt=cnt, n_regions=n_regions, n_blocks=n_blocks, fonts=fonts or {})
            tmp = os.path.join(self.dir, f"slice_{start:06d}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=True)
            os.replace(tmp, os.path.join(self.dir, f"slice_{start:06d}.json"))
        except (TypeError, ValueError, OSError):
            self.on = False                 # something in this document is not serialisable: run without checkpoints

    def restore(self):
        """The contiguous slices from page 1, merged; None when there is nothing valid to resume."""
        if not self.on or not os.path.isdir(self.dir):
            return None
        try:
            with open(os.path.join(self.dir, "manifest.json")) as f:
                if json.load(f).get("key") != self.key:
                    self.clear(); return None
            files = sorted(f for f in os.listdir(self.dir) if f.startswith("slice_") and f.endswith(".json"))
            out = dict(end=0, pages=[], blocks={}, warnings=[], state={}, cnt={}, n_regions=0, n_blocks=0)
            for fn in files:
                with open(os.path.join(self.dir, fn), encoding="utf-8") as f:
                    d = json.load(f)
                if d["start"] != out["end"]:
                    break
                out["end"] = d["end"]; out["pages"] += d["pages"]; out["warnings"] += d["warnings"]
                out["blocks"].update({int(k): v for k, v in d["blocks"].items()})
                out["state"], out["cnt"], out["n_regions"], out["n_blocks"] = d["state"], d["cnt"], d["n_regions"], d["n_blocks"]
                out["fonts"] = d.get("fonts") or {}
            return out if out["end"] else None
        except Exception:
            return None

    def clear(self):
        import shutil
        shutil.rmtree(self.dir, True)


_CK_VERSION = 1


def _analyze_one(i, kind, src, ocr_ok, warnings):
    n = i + 1
    ctx = dict(n=n, pdf_page=None, img=None, items=[], route=None, reason="", per_page_s=0.0)
    t0 = time.perf_counter()
    try:
        if kind == "pdf":
            p = src[i]; ctx["pdf_page"] = p
            a = analyze_page(p)
            route, reason = route_page(a, ocr_ok)
            W, H = p.rect.width, p.rect.height
        else:
            a = dict(segs=[], chars=0, invisible=0, image_coverage=1.0, text_quality=0.0, quality_detail={}, img_zoom=0, hidden_text="",
                     W=src.width / Z, H=src.height / Z)
            route, reason = ("ocr" if ocr_ok else "unreadable"), "image file"
            W, H = a["W"], a["H"]
        ctx.update(a=a, route=route, reason=reason, W=W, H=H)
        if route in ("digital", "embedded_ocr"):
            _analyze_digital(ctx, a["segs"] if route == "digital" else _hidden_segments(ctx["pdf_page"]), warnings)
        ctx["per_page_s"] += time.perf_counter() - t0
    except Exception as e:
        ctx.update(route="error", reason=f"analysis failed ({type(e).__name__})", W=ctx.get("W", 612), H=ctx.get("H", 792), a=dict(chars=0, image_coverage=0, text_quality=0))
        warnings.append(f"Page {n}: analysis failed ({type(e).__name__}); page preserved.")
    return ctx


def _extract_one(ctx, kind, src, out_dir, save_images, warnings, state):
    t0 = time.perf_counter()
    n = ctx["n"]
    # save_images: True saves every page bitmap; "needed" saves only the bitmaps extraction renders anyway (scans,
    # figures) and leaves the rest to be rendered on demand by the viewer; False saves none.
    need_img = save_images is True or ctx["route"] in ("ocr",) or any(it["kind"] in ("figure",) for it in ctx["items"])
    ctx["had_img"] = need_img
    if need_img and ctx["route"] != "error":
        if ctx.get("img") is None:                  # (OCR pages may have been rendered ahead for parallel OCR)
            ctx["img"] = _render(ctx, kind, src)
        if save_images:
            ctx["img"].save(os.path.join(out_dir, f"page_{n}.png"), compress_level=1)
    try:
        if ctx["route"] == "ocr":
            _extract_ocr(ctx, warnings, state)
        elif ctx["route"] == "unreadable":
            warnings.append(f"Page {n}: OCR engine unavailable; scanned content cannot be read.")
            ctx["items"].append(dict(kind="figure", bbox=(0, 0, ctx["W"], ctx["H"]), hint="unknown"))
        out = _blocks_for_page(ctx, state)
    except Exception as ex:
        out = _plain_text_fallback(ctx, n, ex)
    ctx["per_page_s"] += time.perf_counter() - t0
    return out


def _release(ctx, kind):
    """Drop what a finished page no longer needs: glyph segments, layout items, the page object and its bitmap.
    VERIFY and the viewer get the page image again on demand."""
    a = ctx.get("a")
    if isinstance(a, dict):
        a.pop("segs", None)
    ctx["items"] = []
    ctx["pdf_page"] = None
    if kind == "pdf":
        ctx["img"] = None
    for k in ("hrules", "text_segs", "vocab", "hybrid", "_dwords"):
        ctx.pop(k, None)
    for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] == "_diag"]:
        ctx.pop(k, None)


def _lazy_crop(ctx, kind, src, out_dir):
    """crop(bbox) for the verifier: the page bitmap is loaded (saved PNG, else a fresh render) only if a check needs it."""
    cache = {}

    def get():
        if "img" not in cache:
            im = ctx.get("img")
            if im is None and ctx.get("had_img"):    # only pages that had a bitmap during extraction get one here
                fp = os.path.join(out_dir, f"page_{ctx['n']}.png")
                try:
                    if os.path.exists(fp):
                        im = Image.open(fp); im.load()
                    elif kind == "pdf" and ctx["route"] != "error":
                        im = render_page(src, ctx["n"])
                except Exception:
                    im = None
            cache["img"] = im
        return cache["img"]

    def crop(bb):
        im = get()
        return im.crop(tuple(int(v) for v in bb)) if im is not None else None
    return crop


def render_page(src, n, zoom=None):
    """Page n (1-based) of an open PDF as an RGB image at the viewer scale Z (or `zoom`)."""
    z = Z if zoom is None else zoom
    pix = src[n - 1].get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _kind_label(ctx):
    r = ctx["route"]
    fl = ctx.get("flags", {})
    if r == "digital":
        return "mixed" if (fl.get("tables") or fl.get("figures")) else "digital"
    if r == "embedded_ocr":
        return "scanned (embedded OCR layer)"
    if r == "ocr":
        return "scanned" + (" + visual/handwriting" if fl.get("visual_regions") else "") if not ctx.get("a", {}).get("chars") else "digital (broken text layer) -> OCR"
    return r


def _render(ctx, kind, src):
    if kind != "pdf":
        return src
    pix = ctx["pdf_page"].get_pixmap(matrix=pymupdf.Matrix(Z, Z), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _hidden_segments(page):
    """Segments of an invisible (OCR) text layer: read with alpha ignored."""
    d = page.get_text("rawdict", flags=textlayer.RAW_FLAGS)
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                sp["alpha"] = 255
    # re-run the normal reader on the patched dict
    class _P:
        def get_text(self, *_a, **_k):
            return d
    segs, _ = textlayer.read_segments(_P(), ocr_layer=True)
    return segs


def _body_size(segs):
    sizes = []
    for s in segs:
        n = len(s.text)
        sizes += [round(s.size, 1)] * max(1, n)
    if not sizes:
        return 10.0
    return statistics.mode(sizes)


_BULLETS = "•·▪◦‣●○■□–-*\uf0b7\uf0a7\uf076\uf0d8\uf0fc\uf06e\uf0a8"
_PUA_BULLET = {"·": "•", "\uf0b7": "•", "\uf0a7": "▪", "\uf076": "•", "\uf0d8": "•", "\uf0fc": "✓", "\uf06e": "■", "\uf0a8": "•"}


def _merge_bullets(segs):
    """Word processors (LibreOffice, Word) write a list bullet as its own text run, far left of the item's text, often
    in a symbol font whose code is a private-use character. Join the bullet to its item on the same baseline, so the
    item reads '• text' and the bullet is not taken for a one-symbol equation."""
    import dataclasses
    from .textlayer import Segment
    out, used = [], set()
    for s in segs:
        t = s.text.strip()
        if id(s) in used or len(t) != 1 or t not in _BULLETS or s.dir != (1.0, 0.0):
            continue
        right = [o for o in segs if o is not s and id(o) not in used and abs(o.baseline - s.baseline) < max(2.0, 0.35 * max(o.size, 1))
                 and 0 <= o.bbox[0] - s.bbox[2] <= max(40.0, 4 * max(o.size, 1)) and len(o.text.strip()) > 1]
        if not right:
            continue
        o = min(right, key=lambda o: o.bbox[0])
        chars = [dataclasses.replace(c, c=_PUA_BULLET.get(c.c, c.c)) for c in s.chars] + list(o.chars)
        merged = Segment(chars=chars, bbox=(min(s.bbox[0], o.bbox[0]), min(s.bbox[1], o.bbox[1]), max(s.bbox[2], o.bbox[2]), max(s.bbox[3], o.bbox[3])),
                         dir=o.dir)
        used.add(id(s)); used.add(id(o))
        out.append((o, merged))
    if not used:
        return segs
    repl = {id(o): m for o, m in out}
    return [repl.get(id(x), x) for x in segs if id(x) not in used or id(x) in repl]


def _merge_drop_caps(segs, body):
    """A single oversized initial letter beside the first line of a paragraph is a drop cap: rejoin it."""
    out = list(segs)
    for d in [s for s in segs if s.horizontal and len(s.text) == 1 and s.text.isalpha() and s.size >= 2.0 * body]:
        right = [s for s in out if s is not d and s.horizontal and 0 <= s.bbox[0] - d.bbox[2] < 1.5 * body
                 and d.bbox[1] - 0.5 * body <= s.bbox[1] <= d.bbox[1] + 1.2 * body]
        if not right:
            continue
        host = min(right, key=lambda s: s.bbox[1])
        f = host.chars[0]
        cap = textlayer.Char(d.text, f.x0 - 0.6 * f.size, f.y0, f.x0, f.y1, f.x0 - 0.6 * f.size, f.oy, f.size, f.font, f.flags)
        host.chars.insert(0, cap)
        host._text = None
        host.bbox = union(host.bbox, d.bbox)
        out.remove(d)
    return out


LINE_NO = re.compile(r"^\s*(\d{1,3})(?:\s+\S.*)?$")          # bare line numbers; '150.' is a numbered list, not code


def find_listings(segs, body, page_w):
    """Algorithms and pseudo-code printed with bare line numbers ('1  for i = 2 to n', '2    key = A[i]'):
    >= 4 rows whose leftmost piece starts with consecutive numbers at the same x. Returns
    [dict(segs, bbox)] with every segment of those rows (and unnumbered continuation rows between them)."""
    hs = [s for s in segs if s.horizontal and s.text.strip()]
    rows = []
    for s in sorted(hs, key=lambda s: (s.baseline, s.bbox[0])):
        if rows and abs(rows[-1][0].baseline - s.baseline) < 0.4 * max(s.size, 1):
            rows[-1].append(s)
        else:
            rows.append([s])
    cands = []          # (row index, number, x, size)
    for i, r in enumerate(rows):
        r.sort(key=lambda s: s.bbox[0])
        for s in r:
            m = LINE_NO.match(s.text)
            if m and (s.bbox[2] - s.bbox[0] < 0.6 * page_w):
                cands.append((i, int(m.group(1)), s.bbox[0], s.size, s))
    by_x = {}
    for c in cands:
        by_x.setdefault(round(c[2] / 3), []).append(c)
    runs = []
    for _, cs in sorted(by_x.items()):
        cs.sort(key=lambda c: c[0])
        run = [cs[0]]
        for c in cs[1:] + [None]:
            prev = run[-1]
            if c is not None:
                gap = rows[c[0]][0].baseline - rows[prev[0]][0].baseline
                if c[1] == prev[1] + 1 and 0 < gap < 3.2 * max(c[3], 1):
                    run.append(c); continue
            consecutive = sum(1 for a, b in zip(run, run[1:]) if b[1] == a[1] + 1) >= 0.75 * (len(run) - 1)
            codey = len(run) == 3 and sum(1 for c in run if re.search(r"\b(if|for|while|return|else|then|do|repeat|until|let|call|print)\b|\s=\s|\w\(.*\)", " ".join(x.text for x in rows[c[0]]))) >= 2
            if (len(run) >= 4 or codey) and consecutive:
                runs.append(dict(x0=min(x[2] for x in run), r0=run[0][0], r1=run[-1][0]))
            if c is not None:
                run = [c]
    out, used = [], set()
    for R in runs:
        # the listing ends where another listing column (side by side) begins
        right = [o["x0"] for o in runs if o is not R and o["x0"] > R["x0"] + 20 and not (o["r1"] < R["r0"] or o["r0"] > R["r1"])]
        x1 = (min(right) - 2) if right else R["x0"] + 0.62 * page_w
        mem = [s for k in range(R["r0"], R["r1"] + 1) for s in rows[k]
               if R["x0"] - 2 <= s.bbox[0] < x1 and id(s) not in used]
        # a table with an ID column also has consecutive numbers: listing rows hold code/prose, not many
        # separate numeric cells
        per_row = {}
        for s in mem:
            per_row.setdefault(round(s.baseline), []).append(s)
        numeric = sum(1 for s in mem if re.fullmatch(r"[-+−]?[\d.,%$€£]+", s.text.strip()) and not LINE_NO.fullmatch(s.text.strip()))
        rows_many = sum(1 for v in per_row.values() if len(v) >= 4)
        if numeric > 0.25 * len(mem) or rows_many > 0.4 * len(per_row):
            continue
        # an ID column followed by fixed text columns ('2 | isMovable | Can it be moved?'): every row has the same
        # column starts (code is indented), and the rows do not read as code
        rr = [sorted(v, key=lambda s: s.bbox[0]) for v in per_row.values()]
        multi = [v for v in rr if len(v) >= 3]
        if rr and len(multi) >= 0.7 * len(rr):
            x2 = [round(v[1].bbox[0] / 3) for v in multi]
            x3 = [round(v[2].bbox[0] / 3) for v in multi]
            same = lambda xs: max(xs.count(x) + xs.count(x - 1) + xs.count(x + 1) for x in set(xs)) >= 0.85 * len(xs)
            code_rows = sum(1 for v in rr if re.search(r"\b(if|for|while|return|else|then|do|repeat|until|let|call|print|def|end)\b|[=;{}]|//|\w\(", " ".join(x.text for x in v[1:])))
            if same(x2) and same(x3) and code_rows < 0.3 * len(rr):
                continue
        if mem:
            used.update(id(s) for s in mem)
            out.append(dict(segs=mem, bbox=(min(s.bbox[0] for s in mem), min(s.bbox[1] for s in mem),
                                            max(s.bbox[2] for s in mem), max(s.bbox[3] for s in mem))))
    return out


TOC_END = re.compile(r"(?:^|\s)(\d{1,4}|[ivxlc]{1,7})\s*$")
TOC_WORD = re.compile(r"\b(contents|table of contents|inhalt|sommaire|índice|indice)\b", re.I)


def find_toc(segs, body, W, H):
    """Region of a table-of-contents page: most rows are titles that end in a page number, the page numbers
    increase, and the page says 'Contents'. Returned as one box (rows re-joined later), or None."""
    hs = [s for s in segs if s.horizontal and s.text.strip()]
    if len(hs) < 8 or not any(TOC_WORD.search(s.text) for s in hs if len(s.text) < 40):
        return None
    rows = []
    for s in sorted(hs, key=lambda s: (s.baseline, s.bbox[0])):
        if rows and abs(rows[-1][0].baseline - s.baseline) < 0.4 * max(s.size, 1):
            rows[-1].append(s)
        else:
            rows.append([s])
    body_rows = [r for r in rows if 0.06 * H < r[0].baseline < 0.94 * H]
    numbered = []
    for r in body_rows:
        r.sort(key=lambda s: s.bbox[0])
        txt = " ".join(s.text.strip() for s in r)
        m = TOC_END.search(txt)
        words = re.findall(r"[A-Za-z]{3,}", txt)
        if m and words and txt.strip() != m.group(1):
            numbered.append((r, m.group(1)))
    if len(numbered) < 6 or len(numbered) < 0.45 * len(body_rows):
        return None
    aligned = numbered          # page numbers may be right-aligned or follow the title directly
    nums = [int(n) for _, n in aligned if n.isdigit()]
    if len(nums) >= 4 and sum(1 for a, b in zip(nums, nums[1:]) if b >= a) < 0.75 * (len(nums) - 1):
        return None
    y0 = min(s.bbox[1] for r, _ in aligned for s in r)
    y1 = max(s.bbox[3] for r, _ in aligned for s in r)
    inside = [s for r in body_rows for s in r if s.bbox[1] >= y0 - 0.5 * body and s.bbox[3] <= y1 + 0.5 * body]
    return (min(s.bbox[0] for s in inside), y0, max(s.bbox[2] for s in inside), y1)


def _plain_box(d):
    """A drawing that is just a rectangle (stroked and/or filled), possibly with rounded corners."""
    items = d.get("items") or []
    if not items or d.get("rect") is None:
        return False
    kinds = {it[0] for it in items}
    return kinds <= {"re", "qu"} or (kinds <= {"l", "c"} and len(items) <= 12 and d.get("closePath", True))


def find_card_grid(drawings, segs, W, H, body):
    """Boxes of similar size, at least two side by side, each holding text: a card grid. Returns their boxes."""
    boxes = []
    for d in drawings or ():
        r = d.get("rect")
        if r is None or not _plain_box(d):
            continue
        a = r.width * r.height
        if not (0.01 * W * H <= a <= 0.3 * W * H) or r.width < 6 * body or r.height < 3 * body:
            continue
        b = (r.x0, r.y0, r.x1, r.y1)
        if any(overlap_frac(b, o) > 0.9 and overlap_frac(o, b) > 0.9 for o in boxes):
            continue                                    # the same box drawn twice (fill + stroke)
        inside = [s for s in segs if overlap_frac(s.bbox, b) > 0.8]
        if sum(len(s.text) for s in inside) < 20:
            continue
        others = [o for o in drawings if o is not d and o.get("rect") is not None and overlap_frac(tuple(o["rect"]), b) > 0.9
                  and not _plain_box(o)]
        if others:
            continue                                    # a box with a drawing inside is a figure panel
        boxes.append(b)
    if len(boxes) < 3:
        return []
    ws = sorted(b[2] - b[0] for b in boxes); hs = sorted(b[3] - b[1] for b in boxes)
    mw, mh = ws[len(ws) // 2], hs[len(hs) // 2]
    boxes = [b for b in boxes if abs((b[2] - b[0]) - mw) < 0.2 * mw and abs((b[3] - b[1]) - mh) < 0.35 * mh]
    side = sum(1 for a in boxes for b in boxes if a is not b and a[0] < b[0]
               and min(a[3], b[3]) - max(a[1], b[1]) > 0.5 * min(a[3] - a[1], b[3] - b[1]))
    return boxes if len(boxes) >= 3 and side >= 1 else []


def find_index(segs, body, W, H):
    """Body region of a book-index page: most lines are 'entry, 123–125' items (or 'see ...' cross references),
    usually in two or three columns. Returns the box of the index lines, or None."""
    hs = [s for s in segs if s.horizontal and s.text.strip() and 0.06 * H < (s.bbox[1] + s.bbox[3]) / 2 < 0.94 * H]
    if len(hs) < 15:
        return None
    hit = [s for s in hs if tables.INDEX_CELL.search(s.text.strip()) or re.search(r",\s*see\b", s.text)]
    if len(hit) < 0.45 * len(hs):
        return None
    # the index lines and the sub-entries between them (one column's text block)
    y0 = min(s.bbox[1] for s in hit); y1 = max(s.bbox[3] for s in hit)
    inside = [s for s in hs if s.bbox[1] >= y0 - 2 * body and s.bbox[3] <= y1 + 2 * body]
    return (min(s.bbox[0] for s in inside), min(s.bbox[1] for s in inside), max(s.bbox[2] for s in inside), max(s.bbox[3] for s in inside))


def _text_columns(segs, box, body):
    """Split box at vertical white gaps (at least 1.5 em wide, free of text over the box's whole height)."""
    inside = sorted((s.bbox[0], s.bbox[2]) for s in segs if s.horizontal and overlap_frac(s.bbox, box) > 0.6)
    if not inside:
        return [box]
    cols, (a, b) = [], inside[0]
    for x0, x1 in inside[1:]:
        if x0 > b + 1.5 * body:
            cols.append((a, b)); a, b = x0, x1
        else:
            b = max(b, x1)
    cols.append((a, b))
    if len(cols) == 1:
        return [box]
    return [(max(box[0], c0 - 0.5), box[1], min(box[2], c1 + 0.5), box[3]) for c0, c1 in cols]


def _join_rows(segs, box, body):
    """Re-join the pieces of each row inside box into one segment (a table of contents line keeps its title and
    page number together instead of falling apart into three paragraphs)."""
    from .textlayer import Char, Segment
    inside = [s for s in segs if s.horizontal and overlap_frac(s.bbox, box) > 0.6]
    if not inside:
        return segs
    rows = []
    for s in sorted(inside, key=lambda s: (s.baseline, s.bbox[0])):
        if rows and abs(rows[-1][0].baseline - s.baseline) < 0.4 * max(s.size, 1):
            rows[-1].append(s)
        else:
            rows.append([s])
    joined = []
    for r in rows:
        r.sort(key=lambda s: s.bbox[0])
        chars = []
        for k, s in enumerate(r):
            if k:
                a = chars[-1]
                chars.append(Char(" ", a.x1, a.y0, s.chars[0].x0, a.y1, a.x1, a.oy, a.size, a.font, a.flags))
            chars.extend(s.chars)
        seg = Segment(chars, (min(s.bbox[0] for s in r), min(s.bbox[1] for s in r), max(s.bbox[2] for s in r), max(s.bbox[3] for s in r)), r[0].dir)
        prev = joined[-1] if joined else None
        if prev is not None and not TOC_END.search(prev.text) and seg.baseline - prev.baseline < 2.2 * max(seg.size, 1) \
                and (re.fullmatch(r"[\divxlc][\d–\-, ]*(?:\s*(?:ex|pr|n|fig)\.)?", seg.text.strip())
                     or (prev.text.rstrip().endswith((",", "-", "–")) and seg.text.strip()[:1].isalnum())):
            # an entry wrapped onto a second line ('... indicator random variables' / '140'): one entry
            import dataclasses
            a = prev.chars[-1]
            dx = a.x1 + 0.3 * a.size - seg.chars[0].x0
            dy = a.oy - seg.chars[0].oy
            moved = [dataclasses.replace(c, x0=c.x0 + dx, x1=c.x1 + dx, ox=c.ox + dx, y0=c.y0 + dy, y1=c.y1 + dy, oy=c.oy + dy)
                     for c in seg.chars]       # laid out after the first line so the text reads in order
            cs = prev.chars + [Char(" ", a.x1, a.y0, moved[0].x0, a.y1, a.x1, a.oy, a.size, a.font, a.flags)] + moved
            joined[-1] = Segment(cs, union(prev.bbox, seg.bbox), prev.dir)
            continue
        joined.append(seg)
    ids = {id(s) for s in inside}
    return [s for s in segs if id(s) not in ids] + joined


def _analyze_digital(ctx, segs, warnings):
    p = ctx["pdf_page"]; W, H = ctx["W"], ctx["H"]
    body = _body_size(segs); ctx["body_size"] = body
    segs = _merge_drop_caps(segs, body)
    segs = _merge_bullets(segs)
    try:
        drawings = p.get_drawings()
    except Exception:
        drawings = []
    busy = len(drawings) > int(os.getenv("PARSEFLOW_BUSY_VECTOR_PATHS", "2500"))
    if busy:
        # maps, CAD, dense plots: thousands of vector paths. Table-grid search and path clustering would dominate
        # the runtime; the vector art is treated as one figure region instead (bounded cost).
        ctx.setdefault("notes", []).append(f"{len(drawings)} vector paths: busy-graphics mode (no grid-table search)")
        hr, vr = [], []
    else:
        hr, vr = tables.rules_from_drawings(drawings, W, H)
    ctx["hrules"] = hr
    # numbered listings (algorithms / pseudo-code with bare line numbers) first: their '=' lines are not display
    # equations and their aligned columns are not tables
    toc_found = find_toc(segs, body, W, H)
    pre_tocs = []
    regions = [toc_found] if toc_found else _text_columns(segs, find_index(segs, body, W, H), body) if find_index(segs, body, W, H) else []
    for reg in regions:
        before = {id(x) for x in segs}
        segs = _join_rows(segs, reg, body)
        lines = [x for x in segs if id(x) not in before]
        if lines:
            pre_tocs.append(dict(segs=lines, bbox=reg))
            segs = [x for x in segs if id(x) in before]
    listings = find_listings(segs, body, W)
    lst_ids = {id(x) for L in listings for x in L["segs"]}
    segs = [s for s in segs if id(s) not in lst_ids]
    # display equations next (their aligned pieces must not be mistaken for table cells)
    eqs = display_equations(segs, hr, body, None) if ctx["route"] == "digital" else []
    # tables: a ruled grid is found from its lines, so a short cell line ('n', 'x + 1') inside it is cell text, not a
    # display equation
    ruled = tables.detect_ruled(p, segs, len(vr) >= 2, hr, vr) if ctx["route"] == "digital" else []
    if ruled and eqs:
        eqs = [e for e in eqs if not any(overlap_frac(tuple(e["bbox"]), tuple(t.bbox)) > 0.6 for t in ruled)]
    eq_ids = {id(s) for e in eqs for s in e["segs"]}
    noneq = [s for s in segs if id(s) not in eq_ids]
    # text-aligned tables are found from the glyphs alone, so they are searched for on busy vector pages too (only the
    # ruled-grid search, which walks every path, is skipped there)
    aligned, used = tables.detect_aligned(noneq, hr, body, W, exclude=[t.bbox for t in ruled] + [e["bbox"] for e in eqs])
    tbs = ruled + aligned
    try:
        img_boxes = [tuple(i["bbox"]) for i in p.get_image_info()]
    except Exception:
        img_boxes = []
    rejected = []
    kept = []
    for t in tbs:
        t2 = tables.trim_table(t, H, body)
        if t2 is None:
            rejected.append((t, "only one row left after removing the page header / lead-in text")); continue
        t = t2
        why = tables.reject_reason(t, noneq, drawings, img_boxes, body)
        (rejected if why else kept).append((t, why))
    tbs = [t for t, _ in kept]
    hdr_used = set()
    final = []
    for t in tbs:
        t, u = tables.extend_header(t, [s for s in noneq if not any(overlap_frac(s.bbox, o.bbox) > 0.6 for o in tbs)], hr, body)
        t2 = tables.trim_table(t, H, body)
        if t2 is None:
            rejected.append((t, "only one row left after removing the page header / lead-in text")); continue
        t = t2
        why = tables.reject_reason(t, noneq, drawings, img_boxes, body)      # the header may have pulled in a figure
        if why:
            rejected.append((t, why)); continue
        tables.merge_symbol_columns(t)
        # header lines that trimming removed again ('Table 2 Station readings') stay text
        byid = {id(x): x for x in noneq}
        hdr_used |= {k for k in u if k in byid and overlap_frac(byid[k].bbox, t.bbox) > 0.5}
        final.append(t)
    tbs = final
    extra_figs, tocs = [], pre_tocs
    if rejected:
        ctx.setdefault("notes", []).extend(f"table candidate rejected: {why}" for _, why in rejected)
        for t, why in rejected:
            if why.startswith(("vector diagram", "captioned as a figure", "a picture", "figure panels", "coloured boxes")):
                # the figure is where the graphics are (the table candidate may have grown over a header line)
                b = tuple(t.bbox)
                g = [tuple(d["rect"]) for d in drawings if d.get("rect") is not None and overlap_frac(tuple(d["rect"]), b) > 0.5
                     and d["rect"].width * d["rect"].height < 0.6 * W * H]
                g += [ib for ib in img_boxes if overlap_frac(ib, b) > 0.3]
                if g:
                    b = (min(r[0] for r in g), min(r[1] for r in g), max(r[2] for r in g), max(r[3] for r in g))
                    # keep the labels around the graphics (indices above the boxes, '(a)' tags)
                    lab = [s.bbox for s in noneq if overlap_frac(s.bbox, t.bbox) > 0.6 and len(s.text) < 40
                           and s.bbox[1] >= b[1] - 2.5 * body and s.bbox[3] <= b[3] + 2.5 * body]
                    for lb in lab:
                        b = union(b, lb)
                extra_figs.append(b)
            elif why.startswith(("table of contents", "book index")):
                # an index is set in several columns: each column is its own list, read top to bottom
                for cb in _text_columns(noneq, tuple(t.bbox), body):
                    before = {id(x) for x in noneq}
                    noneq = _join_rows(noneq, cb, body)
                    lines = [x for x in noneq if id(x) not in before]
                    if lines:
                        tocs.append(dict(segs=lines, bbox=cb))
                        noneq = [x for x in noneq if id(x) in before]
    tboxes = [t.bbox for t in tbs]
    in_table = lambda s: any(overlap_frac(s.bbox, tb) > 0.6 for tb in tboxes)
    rest = [s for s in noneq if not in_table(s) and id(s) not in hdr_used]
    # figures
    if busy:
        xs = [d["rect"] for d in drawings if d.get("rect") is not None]
        art = (min(r.x0 for r in xs), min(r.y0 for r in xs), max(r.x1 for r in xs), max(r.y1 for r in xs)) if xs else None
        figs = [tuple(art)] if art and (art[2] - art[0]) * (art[3] - art[1]) > 0.05 * W * H else []
    else:
        figs = figures(p, tboxes + [e["bbox"] for e in eqs] + [L["bbox"] for L in listings], rest, drawings)
        for fb in extra_figs:
            over = [f for f in figs if overlap_frac(fb, f) > 0.2 or overlap_frac(f, fb) > 0.2]
            for f in over:
                figs.remove(f); fb = union(fb, f)
            figs.append(fb)
    figs = [(max(0, f[0]), max(0, f[1]), min(W, f[2]), min(H, f[3])) for f in figs if min(W, f[2]) > max(0, f[0]) and min(H, f[3]) > max(0, f[1])]
    # cards: plain boxes laid out in a grid, each holding text. They are read box by box, row by row, and are
    # not figures.
    grid = [] if busy else find_card_grid(drawings, rest, W, H, body)
    if grid:
        figs = [f for f in figs if not any(overlap_frac(c, f) > 0.5 or overlap_frac(f, c) > 0.5 for c in grid)]
    # vector charts: measure their data; the axis, category and legend labels belong to the chart
    chart_of = {}
    if not busy:
        all_text = [s for s in segs if s.horizontal]
        grown = []
        for f in figs:
            try:
                ch = charts.vector_chart(f, drawings, all_text)
            except Exception:
                ch = None
            if ch:
                for ub in ch.get("used") or []:
                    f = union(f, ub)
                chart_of[f] = ch
            grown.append(f)
        figs = grown
    fig_label = {}
    keep = []
    for s in rest:
        f = next((f for f in figs if overlap_frac(s.bbox, f) > 0.6), None)
        if f is not None and len(s.text) < 120:
            fig_label.setdefault(f, []).append(s)
        else:
            keep.append(s)
    rest = keep
    items = []
    for t in tbs:
        items.append(dict(kind="table", bbox=t.bbox, data=t))
    for e in eqs:
        items.append(dict(kind="equation", bbox=e["bbox"], data=e))
    for L in listings:
        items.append(dict(kind="listing", bbox=L["bbox"], data=L["segs"]))
    for T in tocs:
        items.append(dict(kind="toc", bbox=T["bbox"], data=T["segs"]))
    if grid:
        cards = []
        for c in grid:
            mine = [s for s in rest if overlap_frac(s.bbox, c) > 0.8]
            if mine:
                cards.append(dict(bbox=c, segs=mine))
        ids = {id(s) for cd in cards for s in cd["segs"]}
        rest = [s for s in rest if id(s) not in ids]
        if cards:
            gb = (min(c["bbox"][0] for c in cards), min(c["bbox"][1] for c in cards), max(c["bbox"][2] for c in cards), max(c["bbox"][3] for c in cards))
            items.append(dict(kind="cardgrid", bbox=gb, data=cards))
    for f in figs:
        items.append(dict(kind="figure", bbox=f, labels=fig_label.get(f, []), hint="figure", chart=chart_of.get(f)))
    ctx["items"] = items
    ctx["text_segs"] = rest
    ctx["flags"] = dict(text_layer=True, scanned=ctx["route"] != "digital", tables=len(tbs), figures=len(figs), equations=len(eqs))


def _extract_ocr(ctx, warnings, state):
    img = ctx["img"]; n = ctx["n"]
    if ctx.get("ocr_future") is not None:
        res, zoom = ctx.pop("ocr_future").result(), ctx.pop("ocr_zoom")
    else:
        oimg, zoom = _ocr_input(ctx)
        res = ocr.ocr_image(oimg)
    _extract_ocr_result(ctx, warnings, state, res, zoom)


def _ocr_input(ctx):
    """The image OCR reads: the scan at its own resolution (bounded), never upsampled."""
    img = ctx["img"]
    a = ctx.get("a", {})
    # re-render at the scan's native resolution (bounded) for small print
    zoom = Z
    if ctx.get("pdf_page") is not None:
        native = a.get("img_zoom", 0) or Z
        # render at the scan's own resolution: never upsample a low-resolution scan (wasted OCR time), and cap
        # very high resolutions / very large pages to a bounded pixel budget
        zoom = min(max(native, 1.0), float(os.getenv("OCR_MAX_ZOOM", "4.17")))
        max_side = float(os.getenv("OCR_MAX_SIDE_PX", "4200"))
        zoom = min(zoom, max_side / max(ctx["W"], ctx["H"]))
        if abs(zoom - Z) < 0.05:
            oimg, zoom = img.convert("L"), Z
        else:
            pix = ctx["pdf_page"].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, colorspace=pymupdf.csGRAY)
            oimg = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    else:
        oimg = img
    return oimg, zoom


def _extract_ocr_result(ctx, warnings, state, res, zoom):
    img = ctx["img"]; n = ctx["n"]
    a = ctx.get("a", {})
    k = zoom   # px per PDF point
    lines = ocr_lines(res["words"], k)
    lenient = False
    if not lines and res["metrics"].get("contrast", 0) >= 60 and res["metrics"].get("dark_ratio", 0) >= 0.005:
        # real ink but nothing passed the noise filter (typically handwriting): keep the plausible words, flagged
        good = [w for w in res["words"] if w["conf"] >= 0.45 and re.fullmatch(r"[^\W\d_]{3,}[.,;:]?|\d[\d.,/:-]*", w["text"])]
        groups = {}
        for w in good:
            groups.setdefault((w["block"], w["par"], w["line"]), []).append(w)
        lines = [OcrLine(ws, k) for ws in groups.values()]
        lenient = bool(lines)
    # hybrid page: visible native text (titles over a photo, stamps on a scan) is exact - keep it and drop
    # OCR lines that duplicate it
    native = [sg for sg in a.get("segs", []) if sg.text.strip()] if a.get("text_quality", 0) >= 0.6 else []
    if native:
        lines = [l for l in lines if not any(overlap_frac(l.bbox, sg.bbox) > 0.3 or overlap_frac(sg.bbox, l.bbox) > 0.3 for sg in native)]
        lines = lines + native
        ctx["hybrid"] = True
    ctx["ocr_meta"] = dict(conf=res["conf"], plausibility=res["plausibility"], steps=res["steps"], attempts=res["attempts"],
                           metrics={k2: v for k2, v in res["metrics"].items() if k2 != "size"}, render_zoom=round(zoom, 2), seconds=res["seconds"])
    body = statistics.median([l.size for l in lines]) if lines else 10
    ctx["body_size"] = body
    ctx["hrules"] = []
    aligned, used = tables.detect_aligned(lines, [], body, ctx["W"])
    tboxes = [t.bbox for t in aligned]
    rest = [l for l in lines if not any(overlap_frac(l.bbox, tb) > 0.6 for tb in tboxes)]
    ctx["text_segs"] = rest
    items = [dict(kind="table", bbox=t.bbox, data=t, ocr=True) for t in aligned]
    boxes_px = [tuple(v * Z for v in l.bbox) for l in lines]
    for r in detect_ink_regions(img, boxes_px):
        items.append(dict(kind="figure", bbox=tuple(v / Z for v in r), hint="unknown", ink=True))
    ctx["items"] = items
    ctx["flags"] = dict(text_layer=bool(a.get("chars")), scanned=True, tables=len(aligned), figures=0,
                        visual_regions=sum(1 for i in items if i.get("ink")))
    if lenient:
        ctx["lenient_ocr"] = True
        warnings.append(f"Page {n}: only isolated low-confidence OCR words were found (likely handwriting); they are kept, "
                        "flagged, and need review — a vision model is required to transcribe handwriting.")
    if res["conf"] < 0.6:
        warnings.append(f"Page {n}: low OCR confidence ({res['conf']:.2f}) after preprocessing ({', '.join(res['steps']) or 'none needed'}).")


def _blocks_for_page(ctx, state):
    n = ctx["n"]; body = ctx.get("body_size", 10); hr = ctx.get("hrules", [])
    segs = ctx.get("text_segs", [])
    route = ctx["route"]
    regions = [layout.Region(it["kind"], it["bbox"], it) for it in ctx["items"]]
    horiz = [s for s in segs if getattr(s, "horizontal", True)]
    rotated = [s for s in segs if not getattr(s, "horizontal", True)]
    ordered, multi = layout.reading_order(horiz + regions, body)
    if multi:
        ctx.setdefault("flags", {})["multi_column"] = True
    expanded = []
    for it in ordered:
        if isinstance(it, layout.Region) and it.kind == "cardgrid":
            cards = sorted(it.payload["data"], key=lambda c: (c["bbox"][1], c["bbox"][0]))
            rows = []
            for c in cards:                     # rows of cards (tops within half a card), each row left to right
                if rows and c["bbox"][1] < rows[-1][0]["bbox"][1] + 0.5 * (rows[-1][0]["bbox"][3] - rows[-1][0]["bbox"][1]):
                    rows[-1].append(c)
                else:
                    rows.append([c])
            for row in rows:
                for c in sorted(row, key=lambda c: c["bbox"][0]):
                    o, _ = layout.reading_order(c["segs"], body)
                    expanded += o + [layout.Region("break", c["bbox"], None)]
            ctx.setdefault("flags", {})["cards"] = len(cards)
        else:
            expanded.append(it)
    ordered = expanded
    paras = layout.assemble_paragraphs(ordered, body)
    paras = [x for x in paras if not (isinstance(x, layout.Region) and x.kind == "break")]     # card boundaries
    paras = [layout.Para(list(it.payload["data"]), kind="code", code=True) if isinstance(it, layout.Region) and it.kind == "listing"
             else layout.Para(sorted(it.payload["data"], key=lambda l: l.baseline), kind="list") if isinstance(it, layout.Region) and it.kind == "toc"
             else it for it in paras]
    tparas = [p for p in paras if isinstance(p, layout.Para)]
    hsizes = layout.heading_size_levels(tparas, body)
    bsizes = layout.body_like_sizes(tparas)
    vocab = ctx.setdefault("vocab", layout.doc_vocab([s.text for s in segs]))
    blocks, ro = [], 0
    is_ocr = route == "ocr"
    for it in paras:
        ro += 1
        if isinstance(it, layout.Para):
            typ, lvl = layout.classify(it, body, hsizes, bsizes)
            if it.kind == "code" and it.code:
                typ, lvl = "code", None
            elif it.kind == "list" and typ not in ("heading",):
                typ, lvl = "list", None
            is_code = typ == "code" or layout.is_code_lines([l.text for l in it.lines])
            vlines, raw_lines, has_math, _ = para_text(it, hr, use_math=not is_code and not any(isinstance(l, OcrLine) for l in it.lines))
            if vlines is None:
                vlines = raw_lines
            if getattr(it, "code", False) and not any(isinstance(l, OcrLine) for l in it.lines):
                vlines = layout.code_text(it).split("\n")
            if typ in ("list", "code"):
                text = "\n".join(vlines); tr = []
            else:
                text, tr = layout.join_lines(vlines, vocab)
            raw = "\n".join(raw_lines)
            if is_ocr and all(isinstance(l, OcrLine) for l in it.lines) and _looks_like_formula(text):
                eqb = _scanned_equation(ctx, it, text, n, ro, state)
                if eqb is not None:
                    blocks.append(eqb)
                    continue
            if is_ocr and any(isinstance(l, OcrLine) for l in it.lines):
                conf = statistics.mean(getattr(l, "conf", 0.99) for l in it.lines)
                b = make_block("", typ, n, scale(it.bbox), conf, "ocr", "ocr", text, level=lvl if typ == "heading" else None)
            else:
                b = make_block("", typ, n, scale(it.bbox), 0.99, "native text layer", "digital_text_parser", text,
                               level=lvl if typ == "heading" else None)
                if route == "embedded_ocr":
                    b["extractor"] = "ocr"; b["confidence_type"] = "embedded OCR layer"; b["confidence"] = 0.8
                    b["evidence_source"] = "embedded OCR text layer of the PDF (producer unknown)"
            b["raw_text"] = raw
            if tr:
                b["transforms"] = tr
            if has_math:
                b["transforms"] = b.get("transforms", []) + ["inline math converted to LaTeX \\( \\) from glyph geometry"]
                b["inline_math"] = True
            b["font_size"] = round(it.size, 2)
            b["reading_order"] = ro
            blocks.append(b)
            continue
        reg = it.payload
        if it.kind == "table":
            tb = reg["data"]
            rows = tb.rows()
            ext, ctype = ("ocr", "ocr") if reg.get("ocr") else ("table_engine", "heuristic (estimated)")
            b = make_block("", "table", n, scale(tb.bbox), tb.confidence, ctype, ext, f"Table with {tb.n_rows} rows", rows=rows,
                           cells=tb.to_dict()["cells"], table_method=tb.method, header_rows=tb.header_rows,
                           alignment=round(tb.confidence, 2), markdown=tables.serialize(tb))
            b["warnings"] += tb.warnings
            b["reading_order"] = ro
            blocks.append(b)
        elif it.kind == "equation":
            e = reg["data"]
            blocks += _equation_blocks(e, n, body, ro)
        else:
            bb = reg["bbox"]
            labels = reg.get("labels") or []
            img = ctx.get("img")
            ch = reg.get("chart")
            if ch:
                rows = [ch["columns"]] + ch["rows"]
                data = [dict(zip(ch["columns"], r)) for r in ch["rows"]]
                summary = f"{ch['chart_type'].capitalize()} chart: " + "; ".join(
                    f"{r[0]}: " + ", ".join(f"{c} {v}" for c, v in zip(ch["columns"][1:], r[1:]) if v != "") for r in ch["rows"][:12])
                b = make_block("", "chart", n, scale(bb), 0.75, "heuristic (estimated)", "vector_chart", summary, chart_type=ch["chart_type"],
                               data=data, rows=rows, axes=ch["axes"], legend=ch["columns"][1:], transforms=ch["notes"])
                tb = tables.from_rows(rows, "vector_chart", 0.75)
                b["cells"] = tb.to_dict()["cells"]; b["header_rows"] = 1
                b["warnings"].append("Chart values measured from the vector drawing against its axis labels (not read from a data "
                                     "table): check them before relying on exact figures.")
                b["reading_order"] = ro
                blocks.append(b)
            elif labels or _diagram(ctx, bb, reg):
                st = _diagram(ctx, bb, reg)
                lbl_text = " ".join(s.text for s in sorted(labels, key=lambda s: (round(s.bbox[1] / 4), s.bbox[0])))
                if st:
                    b = make_block("", "diagram", n, scale(bb), diagrams.estimate(st), "vector geometry (estimated)",
                                   "vector_diagram", lbl_text or None,
                                   figure_labels=bool(labels), structure=st, structure_text=diagrams.describe(st))
                    b["warnings"].append("Diagram structure (nodes, edges, cells, pointers) read from the vector drawing; "
                                         "unreadable parts are reported, not guessed.")
                else:
                    b = make_block("", "figure", n, scale(bb), 0.99, "native text layer", "digital_text_parser", lbl_text,
                                   figure_labels=True)
                    b["warnings"].append("Figure: only the printed labels inside it were extracted (text layer); the graphic itself is not interpreted.")
                b["reading_order"] = ro
                blocks.append(b)
            elif img is not None:
                crop = img.crop(tuple(int(v * Z) for v in bb))
                rst = None if reg.get("ink") else _raster_diagram(crop, bb, state)      # handwriting regions: OCR / vision
                if rst is not None and not vision.available():
                    vb = [_raster_block(rst, n, bb)]
                else:
                    vb = vision_blocks(crop, scale(bb), reg.get("hint", "figure"), state, "ink" if reg.get("ink") else "figure")
                    if rst is not None:
                        # the model's reading and the geometry read from the pixels, side by side and compared
                        dg = [b for b in vb if b["type"] in ("diagram", "figure", "unknown")]
                        if dg:
                            model = diagrams.from_model(dg[0].get("elements"), dg[0].get("relationships"))
                            rb = _raster_block(rst, n, bb)
                            rb["model_reading"] = dict(elements=dg[0].get("elements"), relationships=dg[0].get("relationships"),
                                                       content=dg[0].get("content"))
                            rb["agreement"] = diagrams.agreement(rst, model) if model else None
                            vb = [rb] + [b for b in vb if b is not dg[0]]
                        else:
                            vb = [_raster_block(rst, n, bb)] + vb
                for b in vb:
                    b["reading_order"] = ro
                blocks += vb
    # rotated text (side-margin stamps, rotated labels) goes last; furniture classification decides its fate
    for s in rotated:
        ro += 1
        b = make_block("", "paragraph", n, scale(s.bbox), 0.99, "native text layer", "digital_text_parser", s.text, rotated=True)
        b["raw_text"] = s.text; b["font_size"] = round(s.size, 2); b["reading_order"] = ro
        blocks.append(b)
    return blocks


_MATHY = set("=+−-×÷^_∑∫∏√∞≤≥≠≈±∂∇∈∉⊂⊆∪∩→←⇒⇔()[]{}|/<>'") | set("αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")


def _diagram(ctx, bb, reg):
    """Structure of a vector diagram in this figure region (trees, graphs, flowcharts, arrays, linked lists), cached."""
    key = ("_diag", tuple(round(v, 1) for v in bb))
    if key in ctx:
        return ctx[key]
    st = None
    pg = ctx.get("pdf_page")
    if pg is not None and ctx.get("route") == "digital" and not reg.get("ink") and os.getenv("PARSEFLOW_DIAGRAMS", "1") != "0":
        try:
            if "_dwords" not in ctx:
                ctx["_dwords"] = diagrams.page_words(pg)
            st = diagrams.extract(pg, bb, ctx["_dwords"])
        except Exception:
            st = None
    ctx[key] = st
    return st


def _looks_like_formula(t):
    vis = [c for c in t if not c.isspace()]
    if len(vis) < 3 or len(vis) > 160:
        return False
    words = re.findall(r"[A-Za-z]{4,}", t)
    sym = sum(1 for c in vis if c in _MATHY)
    return sym / len(vis) >= 0.2 and len(words) <= 2 and any(c in "=<>≤≥≈∑∫" for c in vis)


def _scanned_equation(ctx, para, text, n, ro, state):
    """Equation-looking OCR line on a scan: run the optional local recognizer on its crop only."""
    img = ctx.get("img")
    rinfo = recognizers.info()
    if rinfo["ready"] and img is not None and state.get("math_left", 20) > 0:
        state["math_left"] = state.get("math_left", 20) - 1
        bb = para.bbox
        pad = 4
        crop = img.crop((max(0, int(bb[0] * Z) - pad), max(0, int(bb[1] * Z) - pad), int(bb[2] * Z) + pad, int(bb[3] * Z) + pad))
        r = recognizers.recognize(crop)
        if r:
            lx, conf = r
            b = make_block("", "equation", n, scale(bb), conf, "model (self-reported)", "math_model", text, latex=lx.strip(), math_display=True,
                           raw_text=text, transforms=[f"LaTeX from local recognizer '{rinfo['name']}' on the region crop; OCR text kept as raw_text"])
            b["reading_order"] = ro
            return b
    b = make_block("", "paragraph", n, scale(para.bbox), statistics.mean(getattr(l, "conf", 0.5) for l in para.lines), "ocr", "ocr", text, raw_text=text)
    b["warnings"].append("Possible equation in a scan: OCR text kept, not converted to LaTeX" +
                         ("" if rinfo["ready"] else " (no local formula recognizer configured: PARSEFLOW_MATH_RECOGNIZER)") + ".")
    b["possible_equation"] = True
    b["reading_order"] = ro
    return b


def _equation_blocks(e, n, body, ro):
    chars, rules = e["chars"], e["rules"]
    chars, labels = mathtex.take_equation_numbers(chars)
    size = statistics.median([c.size for c in chars if c.c.strip()] or [body])
    rows = _split_rows(chars, rules, size)
    built = []
    for k, (rc, rr) in enumerate(rows):
        rc2, num = mathtex.split_equation_number(rc, e["bbox"][2])
        lx, conf, notes = mathtex.build_latex(rc2, rr, display=True)
        built.append(dict(rc=rc, lx=lx, conf=conf, notes=notes, num=num))
    if labels and built and not any(x["num"] for x in built):
        built[-1]["num"] = ", ".join(labels)            # labels of a multi-line display go with it
    # lines that continue one derivation ('x = Ix', '= (A^{-1}A)x', ...) become one aligned display
    groups = []
    for x in built:
        g = groups[-1] if groups else None
        if g and x["lx"] and x["conf"] >= 0.45 and mathtex.starts_with_relation(x["lx"]) and not g[-1]["num"] \
                and all(y["lx"] and y["conf"] >= 0.45 for y in g):
            g.append(x)
        else:
            groups.append([x])
    out = []
    for k, g in enumerate(groups):
        rc = [c for x in g for c in x["rc"]]
        lx = g[0]["lx"] if len(g) == 1 else mathtex.join_aligned([x["lx"] for x in g])
        conf = min(x["conf"] for x in g)
        notes = [m for x in g for m in x["notes"]] + (["aligned lines"] if len(g) > 1 else [])
        num = next((x["num"] for x in reversed(g) if x["num"]), None)
        raw = textlayer._chars_to_text(sorted(rc, key=lambda c: (round(c.oy / 3), c.x0)))
        box = (min(c.x0 for c in rc), min(c.y0 for c in rc), max(c.x1 for c in rc), max(c.y1 for c in rc))
        reconstructed = any(x.startswith(("fraction", "script", "radical", "big", "matrix", "cases", "aligned", "integral", "lines"))
                            for x in notes)
        if lx and conf >= 0.45:
            b = make_block("", "equation", n, scale(box), conf, "heuristic (estimated)", "digital_text_parser", raw, latex=lx, math_display=True,
                           equation_number=num, reconstructed=reconstructed, raw_text=raw, transforms=notes)
        else:
            b = make_block("", "paragraph", n, scale(box), 0.6, "native text layer", "digital_text_parser", raw, raw_text=raw)
            b["warnings"].append("Equation could not be converted to LaTeX confidently; original glyph text kept.")
        b["reading_order"] = ro + k * 0.01
        out.append(b)
    return out
