"""Accuracy of glyph identification on fonts that are NOT in the templates (held-out fonts).
For every symbol / Greek letter the held-out font draws:
  * repair rate: the glyph reported as a private-use code (a broken map) is identified as the right character
  * false changes: the glyph reported correctly is left alone (letters and symbols alike)
Run: python3 tools/eval_glyphs.py [font.ttf ...]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from fontTools.ttLib import TTFont
from processing import glyphshape as G

HELD_OUT = ["/usr/share/fonts/truetype/freefont/FreeSerif.ttf", "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf"]
SYMS = "=+−×·∗/|‖:;,()[]{}⟨⟩⌊⌋⌈⌉<>≤≥≠≈≡∼≃≅∝←→↔⇐⇒⇔↦∞∈∉∋⊂⊆⊃⊇∪∩∅∑∏∫√∂∇¬∧∨±∓…°%#&*@$~^_–—′"
GREEK = "αβγδεζηθικλμνξπρσςτυφχψωΓΔΘΛΞΠΣΥΦΨΩ"
SAME = {"ϕ": "φ", "ϵ": "ε", "ϑ": "θ", "ϱ": "ρ", "ς": "σ", "⩾": "≥", "⩽": "≤", "−": "−", "-": "−", "–": "−", "—": "−", "_": "−", "∼": "~"}


def norm(c):
    return SAME.get(c, c) if c else c


def run(paths):
    tot = ok = 0; keep_tot = keep_bad = 0; misses = {}; bad = {}
    for path in paths:
        tt = TTFont(path); cmap = tt.getBestCmap(); gs = tt.getGlyphSet(); upem = tt["head"].unitsPerEm
        for ch in SYMS + GREEK + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            g = cmap.get(ord(ch))
            if not g:
                continue
            paths_ = G.tt_outline(gs, g, upem)
            if not paths_:
                continue
            if ch in SYMS + GREEK:
                tot += 1
                r = G.identify(paths_, True, "", scale=1.0, flip_y=False, math=True)
                if norm(r) == norm(ch):
                    ok += 1
                else:
                    misses[ch] = misses.get(ch, []) + [r]
            keep_tot += 1
            r2 = G.identify(paths_, True, ch, scale=1.0, flip_y=False, math=True)
            if r2 is not None and norm(r2) != norm(ch):
                keep_bad += 1; bad[ch] = bad.get(ch, []) + [r2]
    print(f"repair {ok}/{tot} = {ok / max(1, tot):.1%}   false changes {keep_bad}/{keep_tot} = {keep_bad / max(1, keep_tot):.1%}")
    if "-v" in sys.argv:
        print("misses:", {k: v for k, v in misses.items()})
        print("false:", bad)
    return ok / max(1, tot), keep_bad / max(1, keep_tot)


if __name__ == "__main__":
    run([a for a in sys.argv[1:] if not a.startswith("-")] or HELD_OUT)
