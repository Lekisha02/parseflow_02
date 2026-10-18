"""Diagram structure from an IMAGE (a scanned page, a photo, a screenshot, a picture inside a PDF).

The image is turned into the same primitives the vector reader uses, and the same interpretation runs on them
(diagrams.build). Nothing is learned or guessed from appearance alone:

* nodes   - closed outlines (background regions enclosed by ink whose shape matches an ellipse, a box or a diamond)
            and solid blobs (filled circles), at node size;
* strokes - the ink left after removing nodes and text; each connected stroke that touches two nodes is an edge
            (a stroke touching several nodes is split into the straight pieces that ink actually covers);
* arrows  - extra ink mass where a stroke meets a node (an arrowhead) makes the edge directed;
* text    - OCR words (Tesseract or the configured OCR engine), placed in nodes or used as edge labels and pointers.

Coordinates are converted to PDF points so the shared rules (sizes, distances) apply unchanged. What the image does
not show clearly - a faint line, a broken outline, unreadable text - is left out or reported as unlabelled.
"""
import math
import os

import numpy as np
from scipy import ndimage as ndi

from . import diagrams


def _otsu(g):
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    tot = g.size
    sum_all = np.dot(np.arange(256), hist)
    wb = sb = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        wb += hist[t]
        if wb == 0 or wb == tot:
            continue
        sb += t * hist[t]
        mb, mf = sb / wb, (sum_all - sb) / (tot - wb)
        v = wb * (tot - wb) * (mb - mf) ** 2
        if v > best:
            best, thr = v, t
    return thr


def _ideal(shape, h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    if shape == "ellipse":
        return ((yy - cy) / max(h / 2, 1)) ** 2 + ((xx - cx) / max(w / 2, 1)) ** 2 <= 1.0
    if shape == "diamond":
        return np.abs(yy - cy) / max(h / 2, 1) + np.abs(xx - cx) / max(w / 2, 1) <= 1.0
    return np.ones((h, w), bool)


def _classify(mask):
    """Which simple shape a filled region is (by overlap with the ideal shape in its box), or None."""
    h, w = mask.shape
    best, score = None, 0.0
    for shp in ("rect", "ellipse", "diamond"):
        ide = _ideal(shp, h, w)
        inter = np.logical_and(mask, ide).sum()
        uni = np.logical_or(mask, ide).sum()
        iou = inter / uni if uni else 0
        if iou > score:
            best, score = shp, iou
    need = 0.80 if best == "diamond" else 0.84
    return (best, score) if score >= need else (None, score)


def read_words(img, ocr_fn=None):
    """OCR words of the image: [(x0, y0, x1, y1, text, conf)] in pixels."""
    if ocr_fn is not None:
        return ocr_fn(img)
    from . import ocr
    if not ocr.available():
        return []
    out = []
    for psm in (11,):                                   # sparse text: labels scattered over a figure
        try:
            for w in ocr.words(img, psm=psm):
                if w.get("conf", 0) >= 0.3 and w["text"].strip():
                    out.append((*w["box"], w["text"], w.get("conf", 0)))
        except Exception:
            pass
    return out


def _sheet_ocr(gray, items, ocr_fn=None):
    """OCR many small crops in one call: each crop (already cut to its ink) goes on its own line of a white sheet,
    scaled to a comfortable text height; words are mapped back to their crop by line. items: [(key, array)]."""
    from PIL import Image
    if not items:
        return {}
    rows, keys, y = [], [], 16
    target = int(os.getenv("PF_OCR_TARGET", "34"))
    for key, a in items:
        h, w = a.shape
        f = max(1.0, min(4.0, target / max(h, 1)))
        im = Image.fromarray(a).resize((max(1, int(w * f)), max(1, int(h * f))), Image.LANCZOS)
        rows.append((im, y)); keys.append((key, y, y + im.height))
        y += im.height + 28
    W = max(im.width for im, _ in rows) + 40
    sheet = Image.new("L", (W, y + 10), 255)
    for im, yy in rows:
        sheet.paste(im, (20, yy))
    out = {}
    if ocr_fn is not None:
        found = ocr_fn(sheet)
    else:
        from . import ocr
        if not ocr.available():
            return {}
        try:
            found = [(*w["box"], w["text"], w.get("conf", 0)) for w in ocr.words(sheet, psm=6)]
        except Exception:
            found = []
    for (x0, y0, x1, y1, t, c) in found:
        cy = (y0 + y1) / 2
        for key, a, b in keys:
            if a - 6 <= cy <= b + 6 and t.strip() and c >= 0.2:
                out.setdefault(key, []).append((x0, t))
                break
    res = {k: " ".join(t for _, t in sorted(v)) for k, v in out.items()}
    # what the block reading skipped (often a lone character such as '7'): read on its own, as a word or a character
    if ocr_fn is None:
        from . import ocr
        tries = 0
        for (key, a), (im, _) in zip(items, rows):
            if key in res or a.shape[1] > 2.2 * a.shape[0] or tries >= 8 or len(items) > 60:
                continue
            tries += 1
            canvas = Image.new("L", (im.width + 40, im.height + 40), 255)
            canvas.paste(im, (20, 20))
            for psm in (8, 10):
                try:
                    ws = [w for w in ocr.words(canvas, psm=psm) if w["text"].strip() and w.get("conf", 0) >= 0.3]
                except Exception:
                    ws = []
                if ws:
                    res[key] = " ".join(w["text"] for w in sorted(ws, key=lambda w: w["box"][0]))
                    break
    return res


def _links(cands, lines_ink, sw, shape):
    """How many line strokes join two different candidate nodes (cheap: no OCR). A page of text or handwriting has
    letter loops that look like small nodes but no lines between them."""
    if len(cands) < 2:
        return 0
    owner = np.zeros(shape, np.int32)
    grow = max(2, int(round(sw)) + 2)
    H, W = shape
    for i, c in enumerate(cands, start=1):
        y0, y1 = max(0, c["sl"][0].start - grow), min(H, c["sl"][0].stop + grow)
        x0, x1 = max(0, c["sl"][1].start - grow), min(W, c["sl"][1].stop + grow)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[c["sl"][0].start - y0:c["sl"][0].stop - y0, c["sl"][1].start - x0:c["sl"][1].stop - x0] = c["mask"]
        win = owner[y0:y1, x0:x1]
        win[ndi.binary_dilation(m, iterations=grow) & (win == 0)] = i
    rest = lines_ink & (owner == 0)
    lab, n = ndi.label(rest, structure=np.ones((3, 3)))
    if n == 0:
        return 0
    touch = ndi.grey_dilation(owner, size=(5, 5))
    links = 0
    size = float(np.median([max(c["sl"][0].stop - c["sl"][0].start, c["sl"][1].stop - c["sl"][1].start) for c in cands]))
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        # an edge is a line at least about as long as a node is wide (strokes between letters are short)
        if sl is None or max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) < max(4 * sw, 0.8 * size):
            continue
        ids = np.unique(touch[sl][lab[sl] == k])
        if len([x for x in ids if x]) >= 2:
            links += 1
            if links >= 2:
                break
    return links


def _iou_sl(a, b):
    y0, y1 = max(a[0].start, b[0].start), min(a[0].stop, b[0].stop)
    x0, x1 = max(a[1].start, b[1].start), min(a[1].stop, b[1].stop)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    inter = (y1 - y0) * (x1 - x0)
    ua = (a[0].stop - a[0].start) * (a[1].stop - a[1].start) + (b[0].stop - b[0].start) * (b[1].stop - b[1].start) - inter
    return inter / ua if ua else 0.0


def _is_slash(dark):
    """One thin stroke from corner to corner of a box (the NIL mark of linked structures)."""
    lab, n = ndi.label(dark, structure=np.ones((3, 3)))
    if n == 0:
        return False
    sizes = ndi.sum(dark, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == k)
    h, w = dark.shape
    if ys.max() - ys.min() < 0.55 * h or xs.max() - xs.min() < 0.45 * w:
        return False
    if sizes[k - 1] / max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) > 0.3:
        return False
    c = np.corrcoef(xs, ys)[0, 1] if len(xs) > 3 else 0
    return abs(c) > 0.8


def _long_lines(lines_ink, ink, sw, zoom):
    """Thin strokes much longer than a letter (the edges of a diagram drawn without shapes)."""
    lab, n = ndi.label(lines_ink, structure=np.ones((3, 3)))
    cnt = 0
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) > 25 * zoom and (lab[sl] == k).sum() < 3.0 * sw * max(h, w):
            ys, xs = np.nonzero(lab[sl] == k)
            pts = np.stack([xs, ys], 1).astype(float)
            pts -= pts.mean(0)
            # straight: all pixels close to the main axis (handwriting strokes curl)
            _, sv, vt = np.linalg.svd(pts, full_matrices=False)
            dev = np.abs(pts @ vt[1])
            if np.percentile(dev, 95) <= 2.5 * sw:
                cnt += 1
    return cnt


def _text_nodes(img, g, ink, thr, sw, zoom, words, ocr_fn):
    """Clusters of letters (OCR'd) used as nodes when a diagram draws no shapes around its node labels."""
    from PIL import Image as _I
    clab, cn = ndi.label(ink, structure=np.ones((3, 3)))
    objs = ndi.find_objects(clab)
    hs = sorted(sl[0].stop - sl[0].start for sl in objs if sl is not None)
    ch = float(np.median([h for h in hs if h >= 4])) if hs else 9 * zoom
    keep = np.zeros_like(ink)
    for k, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        comp = clab[sl] == k
        if h <= 2.6 * ch and w <= 2.0 * ch and comp.sum() / max(1, h * w) >= 0.08:
            keep[sl] |= comp
    # join letters into clusters (words, and stacked parts such as a fraction n over 3)
    grown = ndi.binary_dilation(keep, structure=np.ones((3, 3)), iterations=max(2, int(0.35 * ch)))
    lab, n = ndi.label(grown)
    items, boxes = [], []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = (lab[sl] == k) & keep[sl]
        if m.sum() < 6:
            continue
        sub = np.where(m, g[sl], 255).astype(np.uint8)
        boxes.append(sl)
        items.append((len(boxes) - 1, np.pad(sub, 4, constant_values=255)))
    got = _sheet_ocr(g, items, ocr_fn)
    out = []
    for i, sl in enumerate(boxes):
        txt = got.get(i)
        if not txt:
            continue
        m = np.ones((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), bool)
        words.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, txt, 0.8))
        out.append(dict(sl=sl, mask=m, area=int(m.sum()), shape="rect", text=True, fill=None, textnode=True))
    return out


_DIGITISH = {"©": "∞", "o": "0", "O": "0", "i": "1", "l": "1", "I": "1", "|": "1", "z": "2", "Z": "2", "s": "5", "S": "5"}


def _fix_confusions(words, nodes_px):
    """OCR reads 'q0' as 'qo' and 'q1' as 'qi'. When most node names are a letter followed by digits, the characters
    after the letter in the other names are read as the digits they resemble."""
    import re
    names = [w for w in words if re.fullmatch(r"[A-Za-z][A-Za-z0-9|]{1,3}", w[4])]
    if len(names) < 2:
        return words
    good = sum(1 for w in names if re.fullmatch(r"[A-Za-z]\d{1,3}", w[4]))
    letter = {w[4][0] for w in names if re.fullmatch(r"[A-Za-z]\d{1,3}", w[4])}
    if good < max(1, 0.34 * len(names)) or len(letter) != 1:
        return words
    L = letter.pop()
    out = []
    for w in words:
        t = w[4]
        if len(t) >= 2 and t[0] == L and all(ch.isdigit() or ch in _DIGITISH for ch in t[1:]) and not t[1:].isdigit():
            t = L + "".join(ch if ch.isdigit() else _DIGITISH[ch] for ch in t[1:])
        out.append((*w[:4], t, *w[5:]))
    return out


def extract(img, origin=(0.0, 0.0), zoom=2.0, words=None, ocr_fn=None):
    """Structure of the diagram in a PIL image whose top-left is at `origin` (PDF points) rendered at `zoom` pixels
    per point. Returns the same dict as diagrams.extract (source says it was read from an image) or None."""
    # work at a bounded resolution (a whole scanned page at 300 dpi is 30 megapixels)
    maxside = int(os.getenv("PARSEFLOW_RASTER_MAX_PX", "1800"))
    if max(img.size) > maxside:
        f = maxside / max(img.size)
        img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))))
        zoom *= f
        if words is not None:
            words = [(x0 * f, y0 * f, x1 * f, y1 * f, *rest) for (x0, y0, x1, y1, *rest) in words]
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    H, W = g.shape
    if H < 30 or W < 30:
        return None
    # a diagram is drawn on paper: mostly background. A photo or a screenshot full of colour is not read as one
    paper0 = np.median(rgb.reshape(-1, 3), axis=0)
    if (np.abs(rgb.astype(int) - paper0.astype(int)).max(axis=2) <= 24).mean() < 0.55 or float(paper0.mean()) < 150:
        return None
    thr = min(200, max(90, _otsu(g)))
    ink = g < thr
    if ink.mean() < 0.003 or ink.mean() > 0.35:
        return None
    own_ocr = words is None
    words = [] if words is None else list(words)
    px2pt = lambda x, y: (origin[0] + x / zoom, origin[1] + y / zoom)

    # stroke width: twice the typical distance from a stroke pixel to the background
    dist = ndi.distance_transform_edt(ink)
    sw = max(1.0, 2.0 * float(np.median(dist[ink])) if ink.any() else 1.0)

    # text boxes (from OCR) are taken out of the ink so letters are not read as strokes or outlines
    textmask = np.zeros_like(ink)
    for (x0, y0, x1, y1, t, c) in words:
        textmask[max(0, int(y0) - 1):int(y1) + 1, max(0, int(x0) - 1):int(x1) + 1] = True
    # lines: thin anti-aliased strokes are lighter than letters, so they get a more generous threshold than outlines,
    # and one-pixel breaks are closed
    pap = float(np.percentile(g, 90))
    lines_ink = (g < max(thr, min(225, pap - 40)))
    lines_ink = ndi.binary_closing(lines_ink, structure=np.ones((3, 3))) | ink
    # a word's box (especially a PDF font box) can reach over a line running just under or beside it: long strokes
    # are never text
    wh = [y1 - y0 for (x0, y0, x1, y1, t, c) in words] or [9 * zoom]
    longest = 2.5 * float(np.median(wh))
    llab, ln = ndi.label(lines_ink, structure=np.ones((3, 3)))
    long_ids = [k for k, sl in enumerate(ndi.find_objects(llab), start=1)
                if sl is not None and max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) > longest]
    long_mask = np.isin(llab, long_ids) if long_ids else np.zeros_like(ink)
    textmask &= ~long_mask
    strokes_all = lines_ink & ~textmask

    # ---------------------------------------------------------------- nodes: enclosed regions with a node shape
    bg = ~ink
    lab, n = ndi.label(bg)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    objs = ndi.find_objects(lab)
    min_px = (diagrams.MIN_NODE * zoom) ** 2
    max_px = (diagrams.MAX_NODE * zoom) ** 2 * 3
    cand = []
    for k, sl in enumerate(objs, start=1):
        if sl is None or k in border:
            continue
        m = ndi.binary_fill_holes(lab[sl] == k)
        area = int(m.sum())
        h, w = m.shape
        if area < min_px or area > max_px or h < diagrams.MIN_NODE * zoom or w < diagrams.MIN_NODE * zoom or max(h, w) > 6 * min(h, w):
            continue
        shp, sc = _classify(m)
        if shp is None:
            continue
        has_text = any(sl[1].start <= (x0 + x1) / 2 <= sl[1].stop and sl[0].start <= (y0 + y1) / 2 <= sl[0].stop for (x0, y0, x1, y1, t, c) in words)
        inner = rgb[sl][m]
        med = np.median(inner, axis=0) / 255.0 if inner.size else np.array([1, 1, 1])
        fill = None if (med > 0.93).all() else tuple(round(float(v), 2) for v in med)
        cand.append(dict(sl=sl, mask=m, area=area, shape=shp, text=has_text, fill=fill, k=k))
    # two concentric outlines (an accepting state): the ring between them and the inner disc are both enclosed;
    # keep the outer one, marked double, and read its text from the inner disc
    def _contains(o, c):
        oy0, oy1, ox0, ox1 = o["sl"][0].start, o["sl"][0].stop, o["sl"][1].start, o["sl"][1].stop
        cy = (c["sl"][0].start + c["sl"][0].stop) / 2; cx = (c["sl"][1].start + c["sl"][1].stop) / 2
        oc = ((oy0 + oy1) / 2, (ox0 + ox1) / 2)
        return o is not c and oy0 <= cy <= oy1 and ox0 <= cx <= ox1 and c["area"] < o["area"] and \
            abs(cy - oc[0]) < 0.12 * (oy1 - oy0) + 2 and abs(cx - oc[1]) < 0.12 * (ox1 - ox0) + 2 and c["area"] > 0.35 * o["area"]
    for o in cand:
        for c in cand:
            if _contains(o, c) and not c.get("inner_of"):
                o["double"] = True; o["inner"] = c; c["inner_of"] = o
    text_h = []
    if len(cand) > 150:
        cand = sorted(cand, key=lambda c: -c["area"])[:150]
    hole_links = _links([c for c in cand if c.get("inner_of") is None], lines_ink, sw, ink.shape)
    rects = sum(1 for c in cand if c["shape"] == "rect")
    if own_ocr and cand and (hole_links or rects >= 3):        # node-and-edge diagrams, or arrays of boxes
        # the text inside each candidate node, read crop by crop (whole-figure OCR loses digits inside circles)
        items = []
        for i, c in enumerate(cand):
            if c.get("inner_of") is not None:
                continue
            src_c = c.get("inner") or c
            sl, m = src_c["sl"], src_c["mask"]
            inner = ndi.binary_erosion(m, iterations=max(1, int(round(sw))))
            sub = np.where(inner, g[sl], 255).astype(np.uint8)
            dark = sub < thr
            if dark.sum() < 4:
                continue
            if c["shape"] == "rect" and _is_slash(dark):
                c["nil"] = True                          # a diagonal stroke across a box: NIL, not a letter
                continue
            ys_, xs_ = np.nonzero(dark)
            y0_, y1_, x0_, x1_ = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
            text_h.append(y1_ - y0_)
            items.append((i, np.pad(sub[y0_:y1_, x0_:x1_], 4, constant_values=255)))
        got = _sheet_ocr(g, items, ocr_fn)
        for i, txt in got.items():
            c = cand[i]
            sl = c["sl"]
            c["text"] = True
            words.append((sl[1].start + 1, sl[0].start + 1, sl[1].stop - 1, sl[0].stop - 1, txt, 0.8))
    # a self-loop drawn on a node closes a small unlabelled lobe next to it: not a node
    for c in cand:
        if c["text"] or c["shape"] != "ellipse":
            continue
        grow = ndi.binary_dilation(np.pad(c["mask"], 4), iterations=int(2 * sw) + 2)
        for o in cand:
            if o is c or not o["text"]:
                continue
            oy, ox = o["sl"][0].start - c["sl"][0].start + 4, o["sl"][1].start - c["sl"][1].start + 4
            y0_, x0_ = max(0, oy), max(0, ox)
            y1_, x1_ = min(grow.shape[0], oy + o["mask"].shape[0]), min(grow.shape[1], ox + o["mask"].shape[1])
            if y1_ > y0_ and x1_ > x0_ and (grow[y0_:y1_, x0_:x1_] & o["mask"][y0_ - oy:y1_ - oy, x0_ - ox:x1_ - ox]).any():
                c["lobe"] = o
                break
    lobes = [c for c in cand if c.get("lobe")]
    cand = [c for c in cand if not c.get("lobe") and c.get("inner_of") is None]
    # regions closed off by edges (a triangle of three edges) have no label and are larger than real nodes
    texted = [c["area"] for c in cand if c["text"]]
    ref = float(np.median(texted)) if texted else None
    nodes_px = []
    for c in cand:
        if c["text"] or (ref is None and c["area"] <= (30 * zoom) ** 2) or (ref is not None and c["area"] <= 4.0 * ref):
            nodes_px.append(c)
    # filled nodes (tan, grey, red or black discs and boxes): paint that differs from the paper and survives an opening
    # wider than a line (lines and letters do not), with a node shape
    paper = np.median(rgb[g >= np.percentile(g, 60)], axis=0) if (g >= np.percentile(g, 60)).any() else np.array([255, 255, 255])
    paint = (np.abs(rgb.astype(int) - paper.astype(int)).max(axis=2) > 28) | ink
    r = max(2, int(round(1.6 * sw)))
    st = ndi.generate_binary_structure(2, 1)
    # boxes of different fill colours side by side are separate nodes: open each colour on its own
    # dominant fill colours (anti-aliasing spreads a fill over neighbouring shades: group shades near a dominant one)
    pm = paint & ~ink
    code = np.full(ink.shape, -1, np.int32)
    if pm.any():
        q = (rgb[pm] // 16).astype(np.int32)
        keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
        vals, cnts = np.unique(keys, return_counts=True)
        doms = []
        for v in vals[np.argsort(-cnts)]:
            c = np.array([(v // 256) * 16 + 8, ((v // 16) % 16) * 16 + 8, (v % 16) * 16 + 8])
            if cnts[list(vals).index(v)] < min_px / 4 or len(doms) >= 8:
                break
            if all(np.abs(c - d).max() > 40 for d in doms):
                doms.append(c)
        if doms:
            D = np.stack(doms).astype(int)
            px = rgb[pm].astype(int)
            dd = np.abs(px[:, None, :] - D[None, :, :]).max(axis=2)
            best = dd.argmin(axis=1)
            ok = dd[np.arange(len(best)), best] <= 48
            sub = np.full(len(best), -1); sub[ok] = best[ok]
            code[pm] = sub
    code[ink] = -2                                         # dark outlines and letters belong to every colour
    blab = np.zeros(ink.shape, np.int32)
    nxt = 0
    vals, cnts = np.unique(code[code >= 0], return_counts=True)
    for v, cnt in zip(vals, cnts):
        if cnt < min_px:
            continue
        mcol = ndi.binary_opening((code == v) | (code == -2) & ndi.binary_dilation(code == v, iterations=int(2 * sw) + 1),
                                  structure=st, iterations=r)
        lab_v, nv = ndi.label(mcol & (blab == 0))
        blab[lab_v > 0] = lab_v[lab_v > 0] + nxt
        nxt += nv
    solid = []
    for k, sl in enumerate(ndi.find_objects(blab), start=1):
        if sl is None:
            continue
        m = ndi.binary_fill_holes(blab[sl] == k)
        area = int(m.sum()); h, w = m.shape
        if area < min_px or area > max_px or max(h, w) > 4 * min(h, w) or min(h, w) < diagrams.MIN_NODE * zoom:
            continue
        shp, sc = _classify(m)
        if shp is None:
            continue
        if any(_iou_sl(sl, o["sl"]) > 0.5 for o in nodes_px):
            continue                                       # the same node, already found by its outline
        if shp == "diamond" and min(h, w) < 1.6 * diagrams.MIN_NODE * zoom:
            continue
        er = ndi.binary_erosion(m, iterations=max(1, int(round(sw))))
        inner = rgb[sl][er] if er.sum() > 9 else rgb[sl][m]
        med = np.median(inner, axis=0) / 255.0 if inner.size else np.array([0.5, 0.5, 0.5])
        # a filled node carries a label: letters of contrasting brightness inside (a thick pen stroke or a solid
        # blot has none)
        gi = g[sl][er] if er.sum() > 9 else g[sl][m]
        contrast = (gi > 170).sum() if float(med.mean()) < 0.45 else (gi < thr).sum()
        if contrast < max(5, 0.012 * gi.size):
            continue
        solid.append(dict(sl=sl, mask=m, area=area, shape=shp, text=False, fill=tuple(round(float(v), 2) for v in med), solid=True,
                          dark=float(med.mean()) < 0.45))
    solid_links = _links(solid + [c for c in cand if c.get("inner_of") is None], lines_ink, sw, ink.shape) if solid else 0
    if not hole_links and not solid_links and own_ocr:
        # shapes but no lines joining any two of them: not a node-and-edge diagram (a photo, text, handwriting);
        # arrays and records (boxes side by side) are still read below
        boxes = [c for c in cand if c["shape"] == "rect"]
        if len(boxes) < 3:
            return None
    if own_ocr and solid and solid_links:
        # their text: light letters on a dark fill, dark letters on a light fill
        items = []
        for i, nd in enumerate(solid):
            sl = nd["sl"]
            inner = ndi.binary_erosion(nd["mask"], iterations=max(1, int(round(sw))))
            letters = ((g[sl] > 160) if nd["dark"] else (g[sl] < min(thr, 140))) & inner
            if letters.sum() < 4:
                continue
            ys_, xs_ = np.nonzero(letters)
            a_ = np.where(letters, 0, 255).astype(np.uint8)[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
            text_h.append(ys_.max() + 1 - ys_.min())
            items.append((i, np.pad(a_, 4, constant_values=255)))
        for i, txt in _sheet_ocr(g, items, ocr_fn).items():
            sl = solid[i]["sl"]
            solid[i]["text"] = True
            words.append((sl[1].start + 1, sl[0].start + 1, sl[1].stop - 1, sl[0].stop - 1, txt, 0.8))
    # an unlabelled solid blob much bigger than the labelled nodes is a picture or a shaded area, not a node
    sref = [c["area"] for c in solid if c["text"]] or [c["area"] for c in nodes_px if c["text"]]
    sref = float(np.median(sref)) if sref else None
    nodes_px += [c for c in solid if c["text"] or (sref is not None and c["area"] <= 2.5 * sref) or (sref is None and c["area"] <= (30 * zoom) ** 2)]
    # a region enclosed by edges (or a frame around several nodes) contains other nodes: it is not a node itself
    def _holds(o, c):
        cy = (c["sl"][0].start + c["sl"][0].stop) / 2; cx = (c["sl"][1].start + c["sl"][1].stop) / 2
        return o is not c and o["sl"][0].start <= cy <= o["sl"][0].stop and o["sl"][1].start <= cx <= o["sl"][1].stop and c["area"] < 0.6 * o["area"]
    nodes_px = [o for o in nodes_px if not any(_holds(o, c) for c in nodes_px)]
    if not any(c["text"] for c in nodes_px):
        # nothing labelled: keep only node-sized shapes (large empty regions are spaces between edges)
        nodes_px = [c for c in nodes_px if c["area"] <= (30 * zoom) ** 2]
    if os.getenv("PF_RASTER_DEBUG"):
        print("DEBUG holes", len(cand), "solid", len(solid), "nodes", len(nodes_px), [ (c["shape"], c["area"], c["text"]) for c in solid][:12])
    if len(nodes_px) < 2 and own_ocr:
        # no drawn shapes: the nodes are the words themselves, joined by lines (recursion trees, expression trees,
        # mind maps). Each cluster of text touched by a line becomes a node.
        nodes_px = _text_nodes(img, g, ink, thr, sw, zoom, words, ocr_fn) if _long_lines(lines_ink, ink, sw, zoom) >= 3 else []
        text_nodes = True
    else:
        text_nodes = False
    if len(nodes_px) < 2:
        return None

    # node masks (with their outline) and the strokes outside them
    nodemask = np.zeros_like(ink)
    owner = np.zeros(ink.shape, np.int32)
    shapes = []
    for i, nd in enumerate(nodes_px, start=1):
        sl = nd["sl"]
        it_ = max(2, int(round(sw)) + 1) + (int(3 * sw) + 4 if nd.get("textnode") else 0)
        y0w, y1w = max(0, sl[0].start - it_), min(H, sl[0].stop + it_)
        x0w, x1w = max(0, sl[1].start - it_), min(W, sl[1].stop + it_)
        m = np.zeros((y1w - y0w, x1w - x0w), bool)
        m[sl[0].start - y0w:sl[0].stop - y0w, sl[1].start - x0w:sl[1].stop - x0w] = nd["mask"]
        grown = ndi.binary_dilation(m, iterations=it_)
        nodemask[y0w:y1w, x0w:x1w] |= grown
        ow = owner[y0w:y1w, x0w:x1w]
        ow[grown & (ow == 0)] = i
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        half = sw / 2
        a, b = px2pt(x0 - half, y0 - half)
        c_, d = px2pt(x1 + half, y1 + half)
        extra = {}
        if nd.get("double"):
            extra["double"] = True
        if nd.get("nil"):
            extra["nil_px"] = True
        shapes.append(dict(shape=nd["shape"], bbox=(a, b, c_, d), fill=nd["fill"], **extra))
    # a lobe next to a node is the inside of a self-loop: the loop is an edge from the node to itself
    loop_segs, lobe_ring = [], np.zeros_like(ink)
    for lb in lobes:
        o = lb["lobe"]
        if o not in nodes_px:
            continue
        ys_, xs_ = np.nonzero(lb["mask"])
        ys_ = ys_ + lb["sl"][0].start; xs_ = xs_ + lb["sl"][1].start
        oc = ((o["sl"][1].start + o["sl"][1].stop) / 2, (o["sl"][0].start + o["sl"][0].stop) / 2)
        d = np.hypot(xs_ - oc[0], ys_ - oc[1])
        near = d <= d.min() + 2 * sw
        xn, yn = xs_[near], ys_[near]
        p1, p2 = (float(xn.min()), float(yn[np.argmin(xn)])), (float(xn.max()), float(yn[np.argmax(xn)]))
        j = int(np.argmax(d))
        loop_segs.append(dict(pts=[px2pt(*p1), px2pt(*p2)], curved=True, mid=px2pt(float(xs_[j]), float(ys_[j])), loop=False))
        m_ = np.zeros_like(ink); m_[lb["sl"]] = lb["mask"]
        lobe_ring |= ndi.binary_dilation(m_, iterations=int(2 * sw) + 2)
    strokes = strokes_all & ~nodemask
    if own_ocr:
        # text outside nodes (edge weights, pointer names, captions): keep only character-sized ink, then OCR
        clab, cn = ndi.label(ink & ~nodemask, structure=np.ones((3, 3)))
        keep = np.zeros_like(ink)
        ch = float(np.median(text_h)) if text_h else 9.0 * zoom      # letter height: from the node labels
        for k, sl in enumerate(ndi.find_objects(clab), start=1):
            if sl is None:
                continue
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            comp = clab[sl] == k
            fill = comp.sum() / max(1, h * w)
            # letters are about letter-height and fairly dense; a thin diagonal line is sparse in its box
            if 0.25 * ch <= h <= 1.5 * ch and w <= 1.6 * ch and (fill >= 0.12 or max(h, w) <= 0.6 * ch):
                keep[sl] |= comp
        if keep.any():
            # letters close together on one line form a word; each word is read on the OCR sheet
            klab, kn = ndi.label(keep, structure=np.ones((3, 3)))
            boxes = [[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop] for sl in ndi.find_objects(klab) if sl is not None]
            boxes.sort(key=lambda b: (b[1], b[0]))
            merged = True
            while merged:
                merged = False
                for i_ in range(len(boxes)):
                    for j_ in range(i_ + 1, len(boxes)):
                        A_, B_ = boxes[i_], boxes[j_]
                        vo = min(A_[3], B_[3]) - max(A_[1], B_[1])
                        gap = max(A_[0], B_[0]) - min(A_[2], B_[2])
                        if vo > 0.3 * ch and gap < 0.45 * ch:
                            boxes[i_] = [min(A_[0], B_[0]), min(A_[1], B_[1]), max(A_[2], B_[2]), max(A_[3], B_[3])]
                            del boxes[j_]; merged = True; break
                    if merged:
                        break
            items = []
            for i_, (x0, y0, x1, y1) in enumerate(boxes):
                sub = np.where(keep[y0:y1, x0:x1], g[y0:y1, x0:x1], 255).astype(np.uint8)
                items.append((i_, np.pad(sub, 4, constant_values=255)))
            for i_, txt in _sheet_ocr(g, items, ocr_fn).items():
                if not any(ch_.isalnum() or ch_ in "∞−–-" for ch_ in txt):
                    continue
                x0, y0, x1, y1 = boxes[i_]
                words.append((x0, y0, x1, y1, txt, 0.8))
                textmask[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1] = True
        strokes = strokes_all & ~nodemask & ~textmask
    slab, sn = ndi.label(strokes, structure=np.ones((3, 3)))
    touch = ndi.binary_dilation(nodemask, iterations=2)
    dil = ndi.grey_dilation(owner, size=(5, 5))
    segs, heads = [], []
    min_len = 3 * sw
    for k, sl in enumerate(ndi.find_objects(slab), start=1):
        if sl is None:
            continue
        comp = slab[sl] == k
        npx = int(comp.sum())
        if npx < max(6, 2 * sw * sw):
            continue
        ys, xs = np.nonzero(comp)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        if max(ys.max() - ys.min(), xs.max() - xs.min()) < min_len:
            continue
        if lobe_ring.any() and lobe_ring[ys, xs].mean() > 0.6:
            continue                                     # the curve of a self-loop (already an edge)
        # where this stroke meets nodes: one contact point per node (centroid of the stroke pixels next to it)
        near = touch[ys, xs]
        contacts = {}
        if near.any():
            # the node each contact pixel belongs to: the owner of the nearest grown-node pixel
            for y, x in zip(ys[near], xs[near]):
                o = int(dil[y, x])
                if o:
                    contacts.setdefault(o, []).append((x, y))
        pts = {o: (float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v]))) for o, v in contacts.items()}

        def headness(p):
            """Ink near the contact compared with a plain line of this width: an arrowhead adds a filled triangle."""
            R = 5.0 * sw
            d2 = (xs - p[0]) ** 2 + (ys - p[1]) ** 2
            cnt = int((d2 <= R * R).sum())
            return cnt / max(1.0, R * sw)

        def head_for(p, other):
            if headness(p) < 1.75:
                return None
            ang = math.atan2(other[1] - p[1], other[0] - p[0])
            back = (p[0] + 3 * sw * math.cos(ang), p[1] + 3 * sw * math.sin(ang))
            return dict(tip=px2pt(*p), back=px2pt(*back))

        def covered(p, q):
            L = math.hypot(q[0] - p[0], q[1] - p[1])
            if L < 2:
                return 0.0
            n_ = max(8, int(L / max(1.0, sw)))
            hit = 0
            r_ = max(1, int(math.ceil(sw)))
            for t in np.linspace(0.12, 0.88, n_):
                x, y = int(round(p[0] + t * (q[0] - p[0]))), int(round(p[1] + t * (q[1] - p[1])))
                y0_, y1_, x0_, x1_ = max(0, y - r_), min(H, y + r_ + 1), max(0, x - r_), min(W, x + r_ + 1)
                if (strokes[y0_:y1_, x0_:x1_] | textmask[y0_:y1_, x0_:x1_]).any():
                    hit += 1
            return hit / n_

        keys = list(pts)
        if len(keys) == 2:
            p, q = pts[keys[0]], pts[keys[1]]
            # a curved stroke: its point farthest from the straight line between the contacts
            dx, dy = q[0] - p[0], q[1] - p[1]
            L = math.hypot(dx, dy) or 1.0
            off = np.abs((xs - p[0]) * dy - (ys - p[1]) * dx) / L
            j = int(np.argmax(off))
            curved = float(off[j]) > 2.5 * sw
            seg = dict(pts=[px2pt(*p), px2pt(*q)], curved=curved)
            if curved:
                seg["mid"] = px2pt(float(xs[j]), float(ys[j]))
            segs.append(seg)
            for a_, b_ in ((p, q), (q, p)):
                h_ = head_for(a_, b_)
                if h_:
                    heads.append(h_)
        elif len(keys) > 2:
            # crossing lines merged into one stroke: keep the straight pieces between contacts that ink covers
            used = set()
            pairs = sorted(((covered(pts[a_], pts[b_]), a_, b_) for i_, a_ in enumerate(keys) for b_ in keys[i_ + 1:]), reverse=True)
            for cov, a_, b_ in pairs:
                if cov < 0.9:
                    break
                segs.append(dict(pts=[px2pt(*pts[a_]), px2pt(*pts[b_])], curved=False))
                for x_, y_ in ((pts[a_], pts[b_]), (pts[b_], pts[a_])):
                    h_ = head_for(x_, y_)
                    if h_:
                        heads.append(h_)
                used |= {a_, b_}
        elif len(keys) == 1:
            # a stroke from one node to free space: a pointer or an entry arrow; its far end is the tail
            p = pts[keys[0]]
            d2 = (xs - p[0]) ** 2 + (ys - p[1]) ** 2
            j = int(np.argmax(d2))
            tail = (float(xs[j]), float(ys[j]))
            if math.sqrt(float(d2[j])) > 4 * sw:
                segs.append(dict(pts=[px2pt(*tail), px2pt(*p)], curved=False))
                h_ = head_for(p, tail)
                if h_:
                    heads.append(h_)
    segs += loop_segs
    if os.getenv("PF_RASTER_DEBUG"):
        print("DEBUG segs", len(segs), "heads", len(heads), "shapes", len(shapes), segs[:3])
    words = _fix_confusions(words, nodes_px)
    wpt = [(*px2pt(x0, y0), *px2pt(x1, y1), t) for (x0, y0, x1, y1, t, c) in words]
    region = (*px2pt(0, 0), *px2pt(W, H))
    st = diagrams.build(shapes, segs, heads, region, wpt, source="image (shapes, lines and arrowheads found in the pixels; text by OCR)")
    if st:
        # a few marks with punctuation for names (a letter, a scribble) are not a diagram: keep parts with at least
        # three nodes, two of them with readable names
        import re as _re
        good = []
        for p_ in st["parts"]:
            names = [x for x in (p_.get("level_order") or p_.get("vertices") or p_.get("states") or p_.get("cells") or
                                 [c.get("name") for c in p_.get("classes") or []] or
                                 [n for L in p_.get("lists") or [] for n in L.get("nodes", [])] or
                                 [s_.get("node") for s_ in p_.get("steps") or []]) if x]
            readable = [x for x in names if _re.search(r"[A-Za-z0-9]", str(x)) and "unlabelled" not in str(x)]
            if (len(readable) >= 2 and (len(names) >= 3 or len(readable) == len(names))) or p_["type"] == "linked lists" and readable:
                good.append(p_)
        if not good:
            return None
        st["parts"] = good
        st["notes"] = list(st.get("notes") or []) + ["Read from an image: thin, faint or touching lines can be missed, and labels are only as good as the OCR."]
    return st
