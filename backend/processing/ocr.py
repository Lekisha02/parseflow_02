"""OCR via Tesseract (local, offline) with adaptive, measured preprocessing.

Pipeline for one page/region image:
  1. assess()      - measure contrast, background unevenness, speckle noise, skew and text size
  2. preprocess()  - apply only the steps those measurements call for (illumination normalisation,
                     contrast stretch, despeckle, deskew, upscale, Sauvola binarisation); the original image
                     is always kept for fallback/comparison and every applied step is recorded
  3. ocr_image()   - OCR the most promising representation first; if the result is weak (low mean confidence or
                     implausible words) try ONE alternative representation and keep the better result.
                     Orientation (OSD) is only checked when both attempts are very poor. Attempts are bounded.

Raw OCR confidence is recorded but never treated as proof of correctness (see verifier.py).
Results are cached by image hash so the same region is never OCR'd twice.
"""
import hashlib
import os
import re
import threading
import time
from collections import OrderedDict

import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pytesseract import Output
from scipy import ndimage as ndi

_CACHE, _CACHE_LOCK, _CACHE_MAX = OrderedDict(), threading.Lock(), 64


def available():
    from . import engines
    if engines.ocr_engine() != "tesseract" and engines.ocr_ready():
        return engines.ocr_engine()
    try:
        return "tesseract " + str(pytesseract.get_tesseract_version())
    except Exception:
        return None


def lang():
    return os.getenv("OCR_LANG", "eng")


def timeout_s():
    try:
        return float(os.getenv("OCR_TIMEOUT", "60"))
    except ValueError:
        return 60.0


# ---------------------------------------------------------------------------------------------
# v1-compatible helpers (kept: used by pipeline fallbacks and the verifier)
# ---------------------------------------------------------------------------------------------
def words(img, psm=3, config_extra="", timeout=None):
    g = ImageOps.autocontrast(img.convert("L")) if img.mode != "L" else img
    return _run(g, psm, config_extra, timeout)


def _run(g, psm, extra="", timeout=None):
    key = (hashlib.sha1(g.tobytes()).hexdigest(), g.size, psm, extra, lang())
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    from . import engines
    if engines.ocr_engine() != "tesseract":
        alt = engines.ocr_words(g)               # a configured local engine (PaddleOCR, Surya, a command plugin)
        if alt is not None:
            with _CACHE_LOCK:
                _CACHE[key] = alt
            return alt
    d = pytesseract.image_to_data(g, lang=lang(), output_type=Output.DICT, config=f"--psm {psm} {extra}".strip(),
                                  timeout=timeout or timeout_s())
    out = []
    for i, t in enumerate(d["text"]):
        try:
            c = float(d["conf"][i])
        except Exception:
            continue
        if t.strip() and c >= 0:
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append(dict(text=t.strip(), conf=c / 100, box=(x, y, x + w, y + h), key=(d["block_num"][i], d["par_num"][i]),
                            line=d["line_num"][i], h=h, block=d["block_num"][i], par=d["par_num"][i]))
    with _CACHE_LOCK:
        _CACHE[key] = out
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return out


def paragraphs(ws, min_conf=0.6):
    groups = {}
    for w in ws:
        if w["conf"] >= min_conf:
            groups.setdefault(w["key"], []).append(w)
    res = []
    for g in groups.values():
        lines = {}
        for w in g:
            lines.setdefault(w["line"], []).append(w)
        text = "\n".join(" ".join(x["text"] for x in sorted(l, key=lambda z: z["box"][0])) for _, l in sorted(lines.items()))
        bb = (min(w["box"][0] for w in g), min(w["box"][1] for w in g), max(w["box"][2] for w in g), max(w["box"][3] for w in g))
        res.append(dict(text=text, bbox=bb, conf=sum(w["conf"] for w in g) / len(g), height=sum(w["h"] for w in g) / len(g), words=len(g)))
    return res


def quick_text(img):
    from . import engines
    if engines.ocr_engine() != "tesseract":
        return " ".join(w["text"] for w in words(img, 6))
    try:
        return pytesseract.image_to_string(ImageOps.autocontrast(img.convert("L")), lang=lang(), config="--psm 6", timeout=timeout_s()).strip()
    except Exception:
        return ""


def quick_conf(img):
    ws = words(img, 6)
    return (sum(w["conf"] for w in ws) / len(ws), " ".join(w["text"] for w in ws)) if ws else (0.0, "")


# ---------------------------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------------------------
def _small(g, max_side=900):
    s = max(g.size)
    if s <= max_side:
        return np.asarray(g, dtype=np.float32), 1.0
    k = max_side / s
    return np.asarray(g.resize((max(1, int(g.width * k)), max(1, int(g.height * k))), Image.BILINEAR), dtype=np.float32), k


def assess(img):
    """Measure scan characteristics on a downsampled copy. Returns a dict of metrics."""
    g = img.convert("L")
    a, k = _small(g)
    p50 = float(np.percentile(a, 50))
    # contrast = background level minus ink level; percentiles of the whole page fail when text is sparse
    flat = np.sort(a, axis=None)
    nd = max(30, int(0.0005 * flat.size))
    ink_level = float(flat[:nd].mean())            # the darkest few hundred pixels: robust when text is sparse
    bg_level = float(np.percentile(a, 60))
    contrast = float(bg_level - ink_level)
    win = max(15, int(min(a.shape) / 12) | 1)
    bg = ndi.grey_closing(a, size=(7, 7))
    bg = ndi.uniform_filter(bg, size=win)
    light = a > p50
    bg_var = float(np.std(bg[light])) if light.any() else 0.0
    # normalise illumination before binarising so gradients do not masquerade as ink
    norm = a / np.maximum(bg, 1) * 255.0 if bg_var > 10 else a
    thr = min(_otsu(norm), float(np.percentile(norm, 60)) - 0.35 * max(contrast, 1))
    ink = norm < thr
    dark_ratio = float(ink.mean())
    lab, n = ndi.label(ink)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        speckle = float((sizes <= 2).sum() / max(1, n))
        clean = ndi.binary_opening(ink, structure=np.ones((2, 2))) if speckle > 0.3 else ink
        lab2, n2 = ndi.label(clean)
        objs = ndi.find_objects(lab2)
        sizes2 = np.bincount(lab2.ravel())[1:] if n2 else np.array([])
        hs = np.array([sl[0].stop - sl[0].start for sl, sz in zip(objs, sizes2) if sz > 6])
        hs = hs[(hs >= 3) & (hs < a.shape[0] / 8)] if hs.size else hs
        char_h = float(np.percentile(hs, 70) / k) if hs.size >= 5 else 0.0
    else:
        speckle, char_h = 0.0, 0.0
    skew = _skew(ink) if 0.002 < dark_ratio < 0.35 else 0.0
    # speckle must be measured at full resolution (downsampling averages isolated noise pixels away)
    full = np.asarray(g, dtype=np.float32)
    h0, w0 = full.shape
    cy, cx = h0 // 2, w0 // 2
    crop = full[max(0, cy - 300):cy + 300, max(0, cx - 300):cx + 300]
    if crop.size:
        cink = crop < min(_otsu(crop), float(np.percentile(crop, 60)) - 0.35 * max(contrast, 1))
        lab3, n3 = ndi.label(cink)
        if n3:
            sz3 = np.bincount(lab3.ravel())[1:]
            speckle = max(speckle, float((sz3 <= 2).sum() / n3))
    return dict(contrast=round(contrast, 1), bg_unevenness=round(bg_var, 1), dark_ratio=round(dark_ratio, 4),
                speckle=round(speckle, 3), char_height_px=round(char_h, 1), skew_deg=round(skew, 2), size=img.size)


def _otsu(a):
    hist, edges = np.histogram(a, bins=256, range=(0, 256))
    w = np.cumsum(hist); w2 = w[-1] - w
    m = np.cumsum(hist * np.arange(256)); mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w / w[-1] - m) ** 2 / (w * w2)
    between[~np.isfinite(between)] = 0
    return float(np.argmax(between))


def _skew(ink, max_deg=6.0):
    """Projection-profile skew estimate on a binary image (coarse then fine)."""
    h, w = ink.shape
    if h < 50 or w < 50:
        return 0.0
    sm = ink[:: max(1, h // 600), :: max(1, w // 600)].astype(np.float32)

    def score(deg):
        r = ndi.rotate(sm, deg, reshape=False, order=0, mode="constant")
        prof = r.sum(axis=1)
        return float(np.var(prof))
    s0 = score(0.0)
    best = max(np.arange(-max_deg, max_deg + 0.01, 1.0), key=score)
    fine = max(np.arange(best - 1.0, best + 1.01, 0.2), key=score)
    if score(fine) < 1.05 * s0:          # no clear improvement: do not rotate on a weak estimate
        return 0.0
    return -float(fine) if abs(fine) > 0.05 else 0.0


# ---------------------------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------------------------
def preprocess(img, q=None, target_char_px=16, binarize=None, minimal=False):
    """Return (image_L, steps). Steps are chosen from measured metrics, never applied blindly."""
    q = q or assess(img)
    g = img.convert("L")
    steps = []
    if minimal:      # the conservative fallback: only a contrast stretch (and deskew), nothing that can destroy strokes
        out = ImageOps.autocontrast(g, cutoff=1)
        steps.append("minimal: autocontrast only")
        if abs(q["skew_deg"]) >= 0.4:
            out = out.rotate(-q["skew_deg"], resample=Image.BICUBIC, expand=True, fillcolor=255)
            steps.append(f"deskewed {-q['skew_deg']:+.1f} deg")
        return out, steps
    a = np.asarray(g, dtype=np.float32)
    if q["bg_unevenness"] > 10:
        size = int(max(15, 3 * max(q["char_height_px"], 8)))
        bg = ndi.grey_closing(a[::2, ::2], size=(size // 2 | 1, size // 2 | 1))
        bg = ndi.zoom(bg, (a.shape[0] / bg.shape[0], a.shape[1] / bg.shape[1]), order=1)[: a.shape[0], : a.shape[1]]
        a = np.clip(a / np.maximum(bg, 1) * 255.0, 0, 255)
        steps.append("illumination normalised (uneven background)")
    if q["contrast"] < 140:
        lo, hi = np.percentile(a, [1, 99])
        if hi - lo > 10:
            a = np.clip((a - lo) * 255.0 / (hi - lo), 0, 255)
            steps.append(f"contrast stretched ({q['contrast']:.0f} -> full range)")
    out = Image.fromarray(a.astype(np.uint8), "L")
    if q["speckle"] > 0.45 and q["char_height_px"] >= 8:
        out = out.filter(ImageFilter.MedianFilter(3))
        steps.append("despeckled (median 3x3)")
    if abs(q["skew_deg"]) >= 0.4:      # skew_deg = page rotation (counter-clockwise); undo it
        out = out.rotate(-q["skew_deg"], resample=Image.BICUBIC, expand=True, fillcolor=255)
        steps.append(f"deskewed {-q['skew_deg']:+.1f} deg")
    ch = q["char_height_px"]
    if 0 < ch < 0.62 * target_char_px and q["contrast"] >= 25:   # near-blank paper: 'characters' are texture
        f = min(3.0, target_char_px / ch)
        out = out.resize((int(out.width * f), int(out.height * f)), Image.LANCZOS)
        steps.append(f"upscaled x{f:.2f} (small text, ~{ch:.0f}px)")
    if binarize is None:
        binarize = q["bg_unevenness"] > 10 or q["contrast"] < 110
    if binarize:
        out = sauvola(out)
        steps.append("Sauvola binarisation")
    return out, steps


def sauvola(img, window=None, k=0.2, r=128.0):
    a = np.asarray(img.convert("L"), dtype=np.float64)
    w = window or max(15, (min(a.shape) // 40) | 1)
    mean = ndi.uniform_filter(a, w)
    sq = ndi.uniform_filter(a * a, w)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    th = mean * (1 + k * (std / r - 1))
    return Image.fromarray(((a > th) * 255).astype(np.uint8), "L")


# ---------------------------------------------------------------------------------------------
# Adaptive OCR
# ---------------------------------------------------------------------------------------------
_WORDISH = re.compile(r"^[^\W\d_]{2,}$|^\d[\d,.:/%-]*$", re.U)


def _quality(ws):
    if not ws:
        return 0.0, 0.0, 0.0
    conf = sum(w["conf"] for w in ws) / len(ws)
    plaus = sum(1 for w in ws if _WORDISH.match(re.sub(r"[^\w,.:/%-]", "", w["text"]) or "")) / len(ws)
    return conf * (0.4 + 0.6 * plaus), conf, plaus


def ocr_image(img, psm=3, max_attempts=None, min_good=0.72):
    """Adaptive OCR. Returns dict(words, conf, plausibility, steps, attempts, metrics, scale, rotation, timing).
    Word boxes are mapped back to the coordinates of the input image."""
    t0 = time.perf_counter()
    max_attempts = max_attempts or int(os.getenv("OCR_MAX_ATTEMPTS", "2"))
    q = assess(img)
    tries = []

    def attempt(binarize, label, minimal=False):
        pim, steps = preprocess(img, q, binarize=binarize, minimal=minimal)
        ws = _run(pim, psm)
        score, conf, plaus = _quality(ws)
        tries.append(dict(label=label, steps=steps, n=len(ws), conf=round(conf, 3), plausibility=round(plaus, 3), score=round(score, 3)))
        return dict(ws=ws, img=pim, steps=steps, score=score, conf=conf, plaus=plaus)

    first_bin = q["bg_unevenness"] > 10 or q["contrast"] < 110
    best = attempt(first_bin, "binarised" if first_bin else "grey")
    if best["score"] < min_good and max_attempts >= 2:
        risky = any(st.startswith(("upscaled", "Sauvola", "despeckled", "illumination")) for st in best["steps"])
        if risky:
            # the chosen transforms may have hurt: compare with the conservative representation
            alt = attempt(False, "minimal", minimal=True)
        elif best["score"] >= float(os.getenv("OCR_RETRY_FLOOR", "0.3")):
            alt = attempt(not first_bin, "grey" if first_bin else "binarised")
        else:
            alt = None    # near-hopeless with an already minimal input (handwriting, photos): a retry only costs time
        if alt and alt["score"] > best["score"] + 0.02:
            best = alt
    rotation = 0
    if best["score"] < 0.35 and max_attempts >= 3:
        try:
            osd = pytesseract.image_to_osd(img.convert("L"), config="--psm 0", timeout=timeout_s())
            m = re.search(r"Rotate:\s*(\d+)", osd)
            rot = int(m.group(1)) if m else 0
            if rot:
                rimg = img.rotate(-rot, expand=True)
                ws = _run(preprocess(rimg, assess(rimg))[0], psm)
                sc = _quality(ws)
                tries.append(dict(label=f"rotated {rot}", n=len(ws), score=round(sc[0], 3)))
                if sc[0] > best["score"]:
                    rotation = rot
                    best = dict(ws=ws, img=rimg, steps=[f"rotated {rot} deg (OSD)"], score=sc[0], conf=sc[1], plaus=sc[2])
        except Exception:
            pass
    # map coordinates back to the input image (undo upscale / deskew expansion approximately)
    sx = img.width / best["img"].width if not rotation else 1.0
    sy = img.height / best["img"].height if not rotation else 1.0
    ws = []
    for w in best["ws"]:
        b = w["box"]
        ws.append(dict(w, box=(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), h=w["h"] * sy))
    return dict(words=ws, conf=round(best["conf"], 3), plausibility=round(best["plaus"], 3), steps=best["steps"], attempts=tries,
                metrics=q, rotation=rotation, seconds=round(time.perf_counter() - t0, 3))


def clear_cache():
    with _CACHE_LOCK:
        _CACHE.clear()
