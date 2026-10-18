"""Build backend/processing/data/glyph_templates.npz: reference renderings of ordinary characters used by
processing/glyphshape.py to identify glyphs of fonts whose character map is wrong (Type 3 fonts in TeX books).
Run once (needs the DejaVu fonts): python3 tools/make_glyph_templates.py"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from processing.glyphshape import features, N

# Several type designs, so a glyph is compared with more than one way of drawing it: DejaVu (sans/serif, upright,
# italic, bold), STIX (Times-like text and maths, the look of most journal and textbook PDFs) and Liberation Serif
# (Times metrics). STIX ships with matplotlib; missing fonts are skipped. FreeSerif/FreeSans are kept out on purpose:
# tools/eval_glyphs.py measures accuracy on them as unseen fonts.
FONTS = ["DejaVuSerif.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSans.ttf", "DejaVuSerif-Bold.ttf",
         "STIXGeneral.ttf", "STIXGeneralItalic.ttf", "STIXGeneralBol.ttf", "LiberationSerif-Regular.ttf", "LiberationSerif-Italic.ttf"]
try:
    import matplotlib
    _MPL = [os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")]
except Exception:
    _MPL = []
DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts/liberation", "/Library/Fonts"] + _MPL
CHARS = ("=+−×·∗★≫≪/\\|‖:;,.'′″!?()[]{}⟨⟩⌊⌋⌈⌉<>≤≥≠≈≡∼≃≅∝←→↑↓↔⇐⇒⇔↦∞∈∉∋⊂⊆⊃⊇∪∩∅∑∏∫∮√∂∇¬∧∨±∓…⋯⋮°%#&*@$~^_-–—"
         "αβγδεϵζηθϑικλμνξπϖρϱσςτυφϕχψωΓΔΘΛΞΠΣΥΦΨΩ"
         "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
LIGS = ["fi", "fl", "ff", "ffi", "ffl"]
LIGCH = {"fi": "\ufb01", "fl": "\ufb02", "ff": "\ufb00", "ffi": "\ufb03", "ffl": "\ufb04"}
QUOTES = "“”‘’„‚«»\"'⩽⩾"
EM = 400


def render(font, ch):
    img = Image.new("L", (3 * EM, 3 * EM), 0)
    d = ImageDraw.Draw(img)
    base = (EM, 2 * EM)
    d.text(base, ch, font=font, fill=255, anchor="ls")
    a = np.array(img) > 127
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    box = ((xs.min() - base[0]) / EM, (base[1] - ys.max()) / EM, (xs.max() - base[0]) / EM, (base[1] - ys.min()) / EM)
    return features(a, box)


chars, bits, mets = [], [], []
for fn in FONTS:
    path = next((os.path.join(d, fn) for d in DIRS if os.path.exists(os.path.join(d, fn))), None)
    if not path:
        print("missing", fn); continue
    font = ImageFont.truetype(path, EM)
    for ch in list(CHARS) + list(QUOTES) + LIGS:
        glyph = LIGCH.get(ch, ch)
        if font.getmask(glyph).getbbox() is None:
            continue
        f = render(font, glyph)
        if f is None:
            continue
        chars.append(ch); bits.append(np.packbits(f[0].astype(np.uint8).ravel())); mets.append(f[1])
out = os.path.join(os.path.dirname(__file__), "..", "backend", "processing", "data", "glyph_templates.npz")
np.savez_compressed(out, chars=np.array(chars), bits=np.array(bits), metrics=np.array(mets, np.float32))
print(len(chars), "templates ->", out, os.path.getsize(out), "bytes")
