"""Identify glyphs by their shape.

Type 3 fonts (and some broken embedded fonts) carry no usable glyph names and a wrong ToUnicode map, so the text
layer can report '=' as 'D' or '[' as 'Œ'. The glyph outline itself is in the PDF (a Type 3 CharProc is a small
content stream). This module rasterises such an outline and compares it with reference renderings of ordinary
characters (operators, brackets, arrows, Greek, Latin, digits; generated once by tools/make_glyph_templates.py from
DejaVu fonts and shipped as data/glyph_templates.npz), using both the normalised shape and where the ink sits
relative to the baseline (a minus sign and an underscore look alike but sit at different heights).
"""
import os
import re

import numpy as np

N = 24                       # normalised bitmap size
# rarer look-alikes lose ties against the common character ('[' over '⌊', '−' over '—')
RARE = {c: 0.06 for c in "⌊⌋⌈⌉⟨⟩‖∣ϖϱϑϵς—–_⋮⋯∓≃≅∝↦∮"}
RARE.update({c: 0.03 for c in "∇∆"})
RARE.update({c: 0.05 for c in "‚„"})      # low quotes are rare; a comma is not     # rarer than the Greek capitals they resemble (Ω, Δ)
_T = None


def templates():
    global _T
    if _T is None:
        p = os.path.join(os.path.dirname(__file__), "data", "glyph_templates.npz")
        z = np.load(p, allow_pickle=False)
        bits = np.unpackbits(z["bits"], axis=1)[:, : N * N].reshape(-1, N, N).astype(np.float32)
        _T = dict(chars=[str(c) for c in z["chars"]], bits=bits, metrics=z["metrics"].astype(np.float32))
        _T["soft"] = np.stack([_soft(b) for b in bits]) if len(bits) else bits
    return _T


def _soft(b):
    """Blurred, zero-mean, unit-norm bitmap: compares shapes by where the ink is, tolerant of stroke weight."""
    try:
        from scipy import ndimage
        f = ndimage.gaussian_filter(b.astype(np.float32), 1.0)
    except Exception:
        f = b.astype(np.float32)
    f = f - f.mean()
    return f / (np.sqrt((f * f).sum()) or 1.0)


# ------------------------------------------------------------------------------------------- shape features
def features(mask, box):
    """mask: 2-D bool array of the ink, box: ink bbox in em units (x0, y0, x1, y1), y up from the baseline.
    Returns (normalised NxN float bitmap, metrics [w, h, ycenter, aspect])."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    m = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1].astype(np.float32)
    h, w = m.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = m
    # area-average down to N x N
    idx = (np.arange(side) * N // side)
    small = np.zeros((N, N), np.float32)
    np.add.at(small, (idx[:, None].repeat(side, 1), idx[None, :].repeat(side, 0)), sq)
    cnt = np.zeros((N, N), np.float32)
    np.add.at(cnt, (idx[:, None].repeat(side, 1), idx[None, :].repeat(side, 0)), 1)
    small = (small / np.maximum(cnt, 1) > 0.35).astype(np.float32)
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    try:
        from scipy import ndimage
        ncomp = ndimage.label(mask, structure=np.ones((3, 3)))[1]
    except Exception:
        ncomp = 1
    return small, np.array([bw, bh, (y0 + y1) / 2, bw / bh, min(ncomp, 4)], np.float32)


def classify(bitmap, metrics, candidates=None, top=3):
    """[(char, score)] best first. Score combines bitmap overlap (IoU) with size/position agreement."""
    T = templates()
    bits, mets = T["bits"], T["metrics"]
    inter = (bits * bitmap).sum(axis=(1, 2))
    union = np.maximum(bits, bitmap).sum(axis=(1, 2))
    iou = inter / np.maximum(union, 1)
    w = float(os.getenv("PARSEFLOW_GLYPH_NCC", "0.5"))
    if w > 0:
        ncc = (T["soft"] * _soft(bitmap)).sum(axis=(1, 2))
        iou = (1 - w) * iou + w * (ncc - 0.15)          # on IoU's scale: unrelated shapes ~0, same shape ~0.7-0.9
    bw, bh, yc, asp, nc = metrics
    pen = (1.0 * np.minimum(1.0, np.abs(mets[:, 4] - nc))
           + 0.9 * np.abs(np.log(np.maximum(mets[:, 3], 1e-3) / max(asp, 1e-3)))
           + 1.4 * np.abs(mets[:, 2] - yc)
           + 0.9 * np.abs(np.log(np.maximum(mets[:, 1], 1e-3) / bh))
           + 0.5 * np.abs(np.log(np.maximum(mets[:, 0], 1e-3) / bw)))
    score = iou - 0.25 * pen - np.array([RARE.get(c, 0.0) for c in T["chars"]], np.float32)
    best = {}
    for k in np.argsort(-score):
        c = T["chars"][k]
        if candidates is not None and c not in candidates:
            continue
        if c not in best:
            best[c] = float(score[k])
            if len(best) >= top:
                break
    return sorted(best.items(), key=lambda t: -t[1])


def score_for(bitmap, metrics, ch):
    T = templates()
    ks = [k for k, c in enumerate(T["chars"]) if c == ch]
    if not ks:
        return None
    r = classify(bitmap, metrics, candidates={ch}, top=1)
    return r[0][1] if r else None


# ------------------------------------------------------------------------------------------- Type 3 outlines
_NUM = re.compile(rb"[-+]?(?:\d+\.?\d*|\.\d+)")
_TOK = re.compile(rb"[-+]?(?:\d+\.?\d*|\.\d+)|[A-Za-z*'\"]+|\[|\]")


def _bez(p0, p1, p2, p3, n=10):
    t = np.linspace(0, 1, n)[1:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3).tolist()


def outline(stream):
    """Parse a Type 3 glyph procedure. Returns (subpaths [list of (x,y)], filled: bool) in glyph space, or None for
    procedures that draw images or text."""
    if b" BI" in stream or b"\nBI" in stream or b" Do" in stream or b"Tj" in stream:
        return None
    stack, paths, cur, ctm = [], [], [], np.array([1, 0, 0, 1, 0, 0], float)
    stroked = False

    def T(x, y):
        a, b, c, d, e, f = ctm
        return (a * x + c * y + e, b * x + d * y + f)
    last = (0.0, 0.0)
    for tok in _TOK.findall(stream):
        if _NUM.fullmatch(tok):
            stack.append(float(tok)); continue
        op = tok.decode("latin-1")
        try:
            if op == "cm" and len(stack) >= 6:
                a, b, c, d, e, f = stack[-6:]
                m = np.array([[a, b, 0], [c, d, 0], [e, f, 1]]) @ np.array([[ctm[0], ctm[1], 0], [ctm[2], ctm[3], 0], [ctm[4], ctm[5], 1]])
                ctm = np.array([m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[2, 0], m[2, 1]])
            elif op == "m" and len(stack) >= 2:
                if len(cur) > 1:
                    paths.append(cur)
                last = (stack[-2], stack[-1]); cur = [T(*last)]
            elif op == "l" and len(stack) >= 2:
                last = (stack[-2], stack[-1]); cur.append(T(*last))
            elif op in ("c", "v", "y") and len(stack) >= 4:
                if op == "c":
                    p1, p2, p3 = stack[-6:-4], stack[-4:-2], stack[-2:]
                elif op == "v":
                    p1, p2, p3 = last, stack[-4:-2], stack[-2:]
                else:
                    p1, p2, p3 = stack[-4:-2], stack[-2:], stack[-2:]
                pts = _bez(np.array(T(*last)), np.array(T(*p1)), np.array(T(*p2)), np.array(T(*p3)))
                cur.extend(tuple(p) for p in pts); last = tuple(p3)
            elif op == "re" and len(stack) >= 4:
                x, y, w, h = stack[-4:]
                if len(cur) > 1:
                    paths.append(cur)
                paths.append([T(x, y), T(x + w, y), T(x + w, y + h), T(x, y + h)]); cur = []
            elif op == "h":
                if len(cur) > 1:
                    paths.append(cur)
                cur = []
            elif op in ("S", "s"):
                stroked = True
                if len(cur) > 1:
                    paths.append(cur)
                cur = []
            elif op in ("f", "F", "f*", "B", "B*", "b", "b*"):
                if len(cur) > 1:
                    paths.append(cur)
                cur = []
        except Exception:
            return None
        if op not in ("",) and not _NUM.fullmatch(tok):
            stack = []
    if len(cur) > 1:
        paths.append(cur)
    return (paths, not stroked) if paths else None


def rasterize(paths, filled=True, flip_y=True, scale=1.0, res=96):
    """Rasterise subpaths (even-odd fill) -> (mask, bbox in input units with y up)."""
    from PIL import Image, ImageDraw, ImageChops
    pts = [(x * scale, (-y if flip_y else y) * scale) for p in paths for x, y in p]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span = max(x1 - x0, y1 - y0, 1e-6)
    k = (res - 4) / span
    acc = Image.new("1", (res, res), 0)
    for p in paths:
        poly = [((x * scale - x0) * k + 2, (y1 - (-y if flip_y else y) * scale) * k + 2) for x, y in p]
        if len(poly) < 2:
            continue
        lay = Image.new("1", (res, res), 0)
        d = ImageDraw.Draw(lay)
        if filled and len(poly) >= 3:
            d.polygon(poly, fill=1)
        else:
            d.line(poly, fill=1, width=max(1, res // 24))
        acc = ImageChops.logical_xor(acc, lay) if filled else ImageChops.logical_or(acc, lay)
    return np.array(acc, dtype=bool), (x0, y0, x1, y1)


def _prime_stroke(mask):
    """One short stroke slanting from upper right to lower left that tapers downwards (a prime), not a ring."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return False
    m = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    H, W = m.shape
    rows = [np.nonzero(r)[0] for r in m if r.any()]
    if len(rows) < 4 or H < 1.2 * W:
        return False
    try:
        from scipy import ndimage
        lab, n = ndimage.label(~m)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        holes = sum(1 for k in range(1, n + 1) if k not in border)      # a '0' encloses a counter
    except Exception:
        holes = 0
    if holes > 0:
        return False
    centres = [r.mean() / W for r in rows]
    widths = [(r[-1] - r[0] + 1) / W for r in rows]
    q = len(rows) // 4
    slant = np.mean(centres[:q + 1]) - np.mean(centres[-q - 1:])
    return slant > 0.3 and max(widths) < 0.75 and np.mean(widths[:q + 1]) >= np.mean(widths[-q - 1:])


def big_delimiter(mask):
    """Name a tall delimiter (or one piece of an extensible one) from its ink profile, independent of font:
    '(' ')' '[' ']' '{' '}' '|' '‖' '⌊' '⌋' '⌈' '⌉', or the pieces '⎛⎜⎝⎞⎟⎠⎡⎢⎣⎤⎥⎦⎧⎨⎩⎫⎬⎭'. None if unsure."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    m = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    H, W = m.shape
    if H < 2.2 * W or H < 12:
        return None
    try:
        from scipy import ndimage
        ncols = ndimage.label(m.any(axis=0))[1]          # separate vertical strokes side by side ('‖')
    except Exception:
        ncols = 1

    def band(a, b):
        rows = m[int(a * H): max(int(a * H) + 1, int(b * H))]
        on = rows.any(axis=0)
        if not on.any():
            return None
        idx = np.nonzero(on)[0]
        lefts = [np.nonzero(r)[0][0] for r in rows if r.any()]
        rights = [np.nonzero(r)[0][-1] for r in rows if r.any()]
        return (float(np.median(lefts)) / W, float(np.median(rights)) / W, (idx[-1] - idx[0] + 1) / W)
    top, q1, mid, q3, bot = band(0, 0.08), band(0.2, 0.3), band(0.45, 0.55), band(0.7, 0.8), band(0.92, 1.0)
    if None in (top, q1, mid, q3, bot):
        return None

    def edge(a, b, side):
        rows = [r for r in m[int(a * H): max(int(a * H) + 1, int(b * H))] if r.any()]
        if not rows:
            return None
        v = [np.nonzero(r)[0][0] / W if side == "L" else 1 - np.nonzero(r)[0][-1] / W for r in rows]
        return min(v)
    # braces: the stroke swings back out between the ends and the cusp ('{' has its cusp in the middle)
    for side, name in (("L", "{"), ("R", "}")):
        e = [edge(a, b, side) for a, b in ((0.1, 0.18), (0.28, 0.36), (0.46, 0.54), (0.64, 0.72), (0.82, 0.9))]
        if None not in e and e[2] < min(e[1], e[3]) - 0.2 and e[1] > e[0] + 0.08 and e[3] > e[4] + 0.08:
            return name
    if ncols >= 2 and all(b[2] > 0.8 for b in (top, mid, bot)):
        return "‖"
    widths = [b[2] for b in (top, q1, mid, q3, bot)]
    lefts = [b[0] for b in (top, q1, mid, q3, bot)]
    rights = [b[1] for b in (top, q1, mid, q3, bot)]
    # ends bent away from the middle: parentheses (both ends to one side), braces (cusp in the middle), integrals
    # (top end right, bottom end left)
    if lefts[0] - lefts[4] > 0.45 and rights[0] - rights[4] > 0.45 and widths[2] < 0.6:
        return "∫"
    if lefts[0] - lefts[2] > 0.35 and lefts[4] - lefts[2] > 0.35:
        cusp = lefts[1] - lefts[2] > 0.25 and lefts[3] - lefts[2] > 0.25 and widths[2] < 0.6 and rights[2] < 0.7
        return "{" if cusp and rights[1] > 0.8 and rights[3] > 0.8 else "("
    if rights[2] - rights[0] > 0.35 and rights[2] - rights[4] > 0.35:
        cusp = rights[2] - rights[1] > 0.25 and rights[2] - rights[3] > 0.25 and widths[2] < 0.6 and lefts[2] > 0.3
        return "}" if cusp and lefts[1] < 0.2 and lefts[3] < 0.2 else ")"
    straight = max(lefts[1:4]) - min(lefts[1:4]) < 0.2 and max(rights[1:4]) - min(rights[1:4]) < 0.2
    # a straight stem with horizontal bars: brackets, floor, ceiling (and their pieces)
    if straight:
        stem_left = lefts[2] < 0.35
        stem_right = rights[2] > 0.65
        top_bar = widths[0] > 0.7 and widths[2] < 0.6
        bot_bar = widths[4] > 0.7 and widths[2] < 0.6
        if top_bar and bot_bar:
            return "[" if stem_left else "]" if stem_right else None
        if top_bar:
            return ("⎡" if stem_left else "⎤") if H > 4 * W else ("⌈" if stem_left else "⌉")
        if bot_bar:
            return ("⎣" if stem_left else "⎦") if H > 4 * W else ("⌊" if stem_left else "⌋")
        if max(widths) - min(widths) < 0.3:
            return "|"
        return None
    # curves: parentheses bulge outwards in the middle; braces have a cusp in the middle
    if lefts[2] < lefts[0] - 0.25 and lefts[2] < lefts[4] - 0.25:
        if lefts[1] - lefts[2] < 0.25 and lefts[3] - lefts[2] < 0.25:
            return "("
        if lefts[1] > lefts[2] + 0.3 and lefts[3] > lefts[2] + 0.3 and widths[2] < 0.7:
            return "{"
        return "("
    if rights[2] > rights[0] + 0.25 and rights[2] > rights[4] + 0.25:
        if rights[2] - rights[1] > 0.3 and rights[2] - rights[3] > 0.3 and widths[2] < 0.7:
            return "}"
        return ")"
    # extensible pieces: only one end curves
    if lefts[0] > lefts[4] + 0.3:
        return "⎛"
    if lefts[4] > lefts[0] + 0.3:
        return "⎝"
    if rights[0] < rights[4] - 0.3:
        return "⎞"
    if rights[4] < rights[0] - 0.3:
        return "⎠"
    return None


ALIAS = {"⩾": "≥", "⩽": "≤"}
SAME = {"ϕ": "φ", "ϵ": "ε", "ϑ": "θ", "ϱ": "ρ", "ς": "σ"}   # glyph variants of one letter
BIG = set("()[]{}⟨⟩|‖∑∏∫√⌊⌋⌈⌉/\\")
BARS = set("—–-_−")


LIGS = {"fi", "fl", "ff", "ffi", "ffl"}
UNDERBAR = {">": "≥", "<": "≤", "⊃": "⊇", "⊂": "⊆", "∼": "≃", "~": "≃", "≻": "⪰", "≺": "⪯", "=": "≡", "+": "±", "−": "="}


def _with_underbar(paths, filled, scale, flip_y):
    """'≥' drawn as '>' plus a separate bar underneath (math fonts often leave a wide gap): classify the upper part
    alone and add the bar."""
    if len(paths) < 2:
        return None
    sgn = -1 if flip_y else 1

    def bb(p):
        xs = [x for x, _ in p]; ys = [sgn * y for _, y in p]
        return min(xs), min(ys), max(xs), max(ys)
    boxes = [bb(p) for p in paths]
    low = min(range(len(paths)), key=lambda k: boxes[k][1])
    x0, y0, x1, y1 = boxes[low]
    if not (x1 - x0 > 0 and (y1 - y0) < 0.2 * (x1 - x0)):
        return None
    rest = [p for k, p in enumerate(paths) if k != low]
    top = min(b[1] for k, b in enumerate(boxes) if k != low)
    if top <= y1:                       # the bar must be below everything else
        return None
    mask, box = rasterize(rest, filled, scale=scale, flip_y=flip_y)
    f = features(mask, box)
    if f is None:
        return None
    r = classify(f[0], f[1], top=1)
    if r and r[0][1] >= 0.4 and r[0][0] in UNDERBAR:
        return UNDERBAR[r[0][0]]
    return None


def identify(paths, filled, mapped, scale=0.001, flip_y=True, math=True):
    """Decide the character for a glyph outline whose text-layer character is `mapped`.
    Returns the replacement character, or None to keep `mapped`. math=False (text fonts): dashes stay dashes."""
    if mapped and len(mapped) == 1 and (0x370 <= ord(mapped) <= 0x3ff or 0x2010 <= ord(mapped) < 0xe000) and mapped not in "‚„":
        return None          # a Greek letter or a maths symbol in the text layer is what the font says; broken maps
        #                      report ASCII, Latin-1, control or private-use codes instead
    comp = _with_underbar(paths, filled, scale, flip_y)
    if comp and (not mapped or not mapped.isalnum() or ord(mapped[0]) > 0x7f):
        return comp
    mask, box = rasterize(paths, filled, scale=scale, flip_y=flip_y)
    f = features(mask, box)
    if f is None:
        return None
    bmp, met = f
    if math and met[1] > 1.15 and met[3] < 0.55:
        # a tall, narrow glyph in a maths font: a display-size delimiter, an integral, or a piece of one. Named
        # from its ink profile at a resolution fine enough for thin strokes (font-independent).
        hi, _ = rasterize(paths, filled, scale=scale, flip_y=flip_y, res=300)
        bd = big_delimiter(hi)
        if bd:
            return None if bd == mapped else bd
    if math and mapped == "0" and _prime_stroke(mask):
        return "′"            # TeX's maths-symbol font keeps the prime at the code of '0'
    top = classify(bmp, met, top=3)
    if not top:
        return None
    best, bs = top[0]
    big = False
    if bs < 0.35 and met[1] > 0.9:
        # big delimiters / operators (display sizes): compare shape and proportions only
        T = templates()
        cand = [k for k, c in enumerate(T["chars"]) if c in BIG]
        iou = [(float((T["bits"][k] * bmp).sum() / max(1, np.maximum(T["bits"][k], bmp).sum()))
                - 0.25 * abs(np.log(max(T["metrics"][k][3], 1e-3) / max(met[3], 1e-3))), T["chars"][k]) for k in cand]
        if iou:
            iou.sort(reverse=True)
            sc, ch = iou[0]
            second = next((v for v, c in iou if c != ch), -1)
            if sc >= 0.28 and sc - second >= 0.02:
                best, bs, big = ch, max(sc, 0.36), True
    if math and mapped and 0xe000 <= ord(mapped[0]) <= 0xf8ff:
        # a private-use code in a maths font stands for a symbol or Greek letter the font has no Unicode for: never
        # an ASCII letter, digit or text ligature, and only a clear winner is taken (else it stays flagged)
        if big and not (best.isascii() and best.isalnum()):
            return best
        T = templates()
        cands = {c for c in T["chars"] if not (c.isascii() and c.isalnum()) and c not in LIGS}
        tops = classify(bmp, met, candidates=cands, top=6)
        if not tops:
            return None
        c0, s0 = tops[0]
        c0 = ALIAS.get(c0, c0)
        if c0 in BARS:
            c0 = "−" if met[2] > 0.12 else "_"
        famof = lambda c: "bar" if c in BARS or c in "—–" else "dot" if c in "·.∙" else "tilde" if c in "∼~" else SAME.get(ALIAS.get(c, c), c)
        fam = famof(tops[0][0])
        second = next((sc for c, sc in tops[1:] if famof(c) != fam), -1)
        if s0 >= 0.35 or (s0 >= 0.15 and s0 - second >= 0.04):
            return c0
        return None
    if best in BARS and math:
        best = "−" if met[2] > 0.12 else "_"
    elif best in ("—", "–") and not math:
        best = "—" if met[0] > 0.75 else "–"          # em dash about 1 em wide, en dash about half
    if mapped and len(mapped) == 1 and not mapped.isascii() and mapped.isalpha():
        # a ligature reported as an accented letter ('ﬁ' as 'û') - a classic broken map in TrueType subsets
        lig = next(((c, sc) for c, sc in top if c in LIGS and sc >= bs - 0.05), None)
        if lig and lig[1] >= 0.45:
            return lig[0]
    if best in LIGS:
        ms = score_for(bmp, met, mapped) if mapped and len(mapped) == 1 else None
        return best if bs >= 0.45 and (ms is None or bs - ms >= 0.2) and mapped not in LIGS else None
    best = ALIAS.get(best, best)
    ms = score_for(bmp, met, mapped) if mapped else None
    if mapped and mapped.isalnum() and ms is not None and ms < 0 and best.isalnum():
        # the reported letter clearly isn't this shape and the closest letter is only a look-alike of a symbol
        alt = next(((c, sc) for c, sc in top[1:] if not c.isalnum() and sc >= bs - 0.1), None)
        if alt and alt[1] >= 0.45:
            return ALIAS.get(alt[0], alt[0])
    bad = (not mapped) or ord(mapped[0]) < 32 or 0x7f <= ord(mapped[0]) <= 0x9f or mapped in "\ufffdŒœŠšŸŽž" \
        or 0xe000 <= ord(mapped[0]) <= 0xf8ff or mapped in "ÍÎÏÌ"
    if bad:
        if bs >= 0.35:
            return best
        # a private-use / control code is wrong anyway: accept a weaker match when it clearly beats the others
        fam = SAME.get(best, best)
        second = next((sc for c, sc in top[1:] if SAME.get(c, c) != fam), -1)
        return best if bs >= 0.15 and bs - second >= 0.04 else None
    if best == mapped or ms is None and not mapped.isalnum():
        return None if best == mapped else (best if bs >= 0.5 else None)
    if mapped.isalnum():
        # letters and digits are only replaced by a clearly different symbol
        if big and (ms is None or ms < 0.1):
            return best
        if not best.isalnum() and bs >= 0.45 and (ms is None or (ms < 0.3 and bs - ms >= 0.3)):
            return best
        return None
    # punctuation / operators reported wrongly ('(' as '.', ',' as ';')
    if bs >= 0.4 and (ms is None or bs - ms >= 0.25):
        return best
    return None


def tt_outline(glyphset, name, upem):
    """Outline of a TrueType/CFF glyph (fontTools glyph set) as subpaths in em units, y up."""
    from fontTools.pens.basePen import BasePen

    class P(BasePen):
        def __init__(self, gs):
            super().__init__(gs); self.paths = []; self.cur = []

        def _moveTo(self, pt):
            if len(self.cur) > 1:
                self.paths.append(self.cur)
            self.cur = [pt]

        def _lineTo(self, pt):
            self.cur.append(pt)

        def _curveToOne(self, p1, p2, p3):
            p0 = self.cur[-1]
            self.cur.extend(tuple(x) for x in _bez(np.array(p0), np.array(p1), np.array(p2), np.array(p3)))

        def _qCurveToOne(self, p1, p2):
            p0 = np.array(self.cur[-1]); a, b = np.array(p1), np.array(p2)
            t = np.linspace(0, 1, 8)[1:, None]
            self.cur.extend(tuple(x) for x in ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * a + t ** 2 * b).tolist())

        def _closePath(self):
            if len(self.cur) > 1:
                self.paths.append(self.cur)
            self.cur = []
        _endPath = _closePath
    pen = P(glyphset)
    glyphset[name].draw(pen)
    pen._closePath()
    return [[(x / upem, y / upem) for x, y in p] for p in pen.paths]
