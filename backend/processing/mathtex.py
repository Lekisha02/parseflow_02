"""Mathematical content: detection on the native text layer and LaTeX reconstruction from glyph geometry.

No model is needed for digital PDFs: TeX-produced (and most other) PDFs keep every math glyph with its font,
size and baseline. Structure is rebuilt deterministically:

  * fractions      - a horizontal rule (vector line) with glyph groups directly above and below it
  * big operators  - sum/prod/integral/union... with smaller glyphs stacked above/below -> \\sum_{..}^{..}
  * radicals       - a radical sign followed by an overbar rule -> \\sqrt{..}
  * scripts        - smaller glyphs whose baseline is raised/lowered relative to the main baseline
  * delimiters     - tall brackets -> \\left( .. \\right)
  * symbols        - Unicode -> LaTeX command table; TeX font families -> \\mathbb, \\mathcal, \\mathfrak, \\mathbf

Every reconstruction returns (latex, confidence, notes). When glyphs cannot be mapped (unmapped/private-use
codes) the confidence drops and the caller keeps the original text and flags the block instead of inventing.

Scanned math needs an image recognizer; see recognizers.py (optional, pluggable, never downloads weights).
"""
import re
import statistics
import unicodedata

MATH_FONT = re.compile(r"(cmmi|cmsy|cmex|cmbsy|cmmib|msbm|msam|eufm|eusm|eusb|rsfs|stmary|wasy|esint|lmmath|latinmodern-math|"
                       r"math|symbol|stix|cambria\s*math|mtextra|mt\s*extra|euclid|txsy|pxsy|txex|pxex|txmi|pxmi|zxmi|ntxmi|"
                       r"xits|asana|texgyre.*math|mathjax|libertinusmath|fira.?math|dejavumath)", re.I)
TEXTY_SYMBOL_FONT = re.compile(r"(wingding|dingbat|zapf)", re.I)
URLISH = re.compile(r"(https?://|www\.|ftp://|[\w.+-]+@[\w-]+\.[a-z]{2,}|doi\.org|\bdoi:|\.(?:com|org|net|edu|gov|io|ac|in|uk)(?:/|\b))", re.I)

GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ϵ": "epsilon", "ζ": "zeta", "η": "eta",
         "θ": "theta", "ϑ": "vartheta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi",
         "ϖ": "varpi", "ρ": "rho", "ϱ": "varrho", "σ": "sigma", "ς": "varsigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
         "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda",
         "Ξ": "Xi", "Π": "Pi", "Σ": "Sigma", "Υ": "Upsilon", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega", "ε": "varepsilon"}
# U+03B5 'ε' is \varepsilon in TeX output, U+03F5 'ϵ' is \epsilon
GREEK["ε"] = "varepsilon"; GREEK["φ"] = "varphi"
SYMBOLS = {
    "∑": r"\sum", "∏": r"\prod", "∐": r"\coprod", "∫": r"\int", "∬": r"\iint", "∭": r"\iiint", "∮": r"\oint",
    "⋃": r"\bigcup", "⋂": r"\bigcap", "⨁": r"\bigoplus", "⨂": r"\bigotimes", "⨀": r"\bigodot", "⋁": r"\bigvee", "⋀": r"\bigwedge",
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx", "≡": r"\equiv", "∼": r"\sim", "≃": r"\simeq", "≅": r"\cong",
    "∝": r"\propto", "≪": r"\ll", "≫": r"\gg", "≺": r"\prec", "≻": r"\succ", "⪯": r"\preceq", "⪰": r"\succeq", "≼": r"\preceq",
    "∈": r"\in", "∉": r"\notin", "∋": r"\ni", "⊂": r"\subset", "⊃": r"\supset", "⊆": r"\subseteq", "⊇": r"\supseteq",
    "⊊": r"\subsetneq", "∪": r"\cup", "∩": r"\cap", "∖": r"\setminus", "∅": r"\emptyset", "∀": r"\forall", "∃": r"\exists",
    "∄": r"\nexists", "¬": r"\neg", "∧": r"\wedge", "∨": r"\vee", "⊕": r"\oplus", "⊗": r"\otimes", "⊙": r"\odot", "⊖": r"\ominus",
    "∂": r"\partial", "∇": r"\nabla", "∞": r"\infty", "ℓ": r"\ell", "ℏ": r"\hbar", "℘": r"\wp", "ℜ": r"\Re", "ℑ": r"\Im", "ℵ": r"\aleph",
    "×": r"\times", "÷": r"\div", "·": r"\cdot", "⋅": r"\cdot", "∗": r"\ast", "∘": r"\circ", "∙": r"\bullet", "•": r"\bullet",
    "±": r"\pm", "∓": r"\mp", "−": "-", "–": "-", "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "↦": r"\mapsto", "⟶": r"\longrightarrow",
    "⟹": r"\Longrightarrow", "⟺": r"\Longleftrightarrow", "↑": r"\uparrow", "↓": r"\downarrow", "↪": r"\hookrightarrow",
    "⊥": r"\perp", "∥": r"\parallel", "∣": r"\mid", "|": "|", "‖": r"\|", "⟨": r"\langle", "⟩": r"\rangle", "〈": r"\langle", "〉": r"\rangle",
    "⌊": r"\lfloor", "⌋": r"\rfloor", "⌈": r"\lceil", "⌉": r"\rceil", "√": r"\sqrt", "′": "'", "″": "''", "∠": r"\angle",
    "△": r"\triangle", "□": r"\square", "◦": r"\circ", "⋯": r"\cdots", "…": r"\ldots", "⋮": r"\vdots", "⋱": r"\ddots",
    "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}", "ℙ": r"\mathbb{P}",
    "⊤": r"\top", "⊢": r"\vdash", "⊨": r"\models", "†": r"\dagger", "‡": r"\ddagger", "≔": r":=", "∶": ":", "⊲": r"\triangleleft",
    "⋆": r"\star", "★": r"\star", "♯": r"\sharp", "♭": r"\flat", "°": r"^{\circ}", "{": r"\{", "}": r"\}", "#": r"\#", "%": r"\%",
    "&": r"\&", "_": r"\_", "$": r"\$", "\\": r"\backslash", "~": r"\sim", "^": r"\hat{}", "‰": r"\permil",
}
BIG_OPS = {r"\sum", r"\prod", r"\coprod", r"\bigcup", r"\bigcap", r"\bigoplus", r"\bigotimes", r"\bigodot", r"\bigvee", r"\bigwedge"}
INTEGRALS = {r"\int", r"\iint", r"\iiint", r"\oint"}
FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth", "arcsin", "arccos", "arctan", "log", "ln", "lg",
         "exp", "lim", "liminf", "limsup", "max", "min", "sup", "inf", "det", "dim", "ker", "deg", "gcd", "lcm", "arg", "Pr", "hom", "mod"}
OPEN, CLOSE = set("([{⟨⌊⌈|‖"), set(")]}⟩⌋⌉|‖")
MATH_UNICODE = set(GREEK) | {k for k in SYMBOLS if k not in "{}#%&_$\\~^|•·…°–"} | set("=<>+") | set("∑∏∫")


def is_math_char(ch):
    c = ch.c
    if not c.strip():
        return False
    if TEXTY_SYMBOL_FONT.search(ch.font):
        return False
    if MATH_FONT.search(ch.font):
        return True
    if c in MATH_UNICODE or c in GREEK:
        return True
    if len(c) != 1:
        return False
    cp = ord(c)
    return 0x2200 <= cp <= 0x22FF or 0x1D400 <= cp <= 0x1D7FF or 0x27C0 <= cp <= 0x27EF or 0x2980 <= cp <= 0x2AFF


def math_density(chars):
    vis = [c for c in chars if c.c.strip()]
    if not vis:
        return 0.0
    return sum(1 for c in vis if is_math_char(c)) / len(vis)


# ---------------------------------------------------------------------------------------------
# Glyph -> token
# ---------------------------------------------------------------------------------------------
def _font_cmd(ch):
    f = ch.font.lower()
    if "msbm" in f or "bbold" in f or "dsrom" in f:
        return "mathbb"
    if "eufm" in f or "eufb" in f or "fraktur" in f:
        return "mathfrak"
    if ("cmsy" in f or "eusm" in f or "rsfs" in f or "calligraph" in f) and "A" <= ch.c <= "Z":
        return "mathcal"
    if ("cmbx" in f or "bold" in f or "cmmib" in f) and ch.c.isalnum():
        return "mathbf"
    return None


def _math_alnum(c):
    """Map Mathematical Alphanumeric Symbols (U+1D400..) back to (letter, style)."""
    if len(c) != 1:                     # an expanded ligature ('fi') or a repaired multi-letter glyph
        return None
    cp = ord(c)
    if 0x1D400 <= cp <= 0x1D7FF:
        name = unicodedata.name(c, "")
        m = re.match(r"MATHEMATICAL (.*?) (CAPITAL|SMALL|DIGIT) (\w+)", name)
        if m:
            style, kind, letter = m.groups()
            base = letter if kind == "CAPITAL" else letter.lower() if kind == "SMALL" else str("ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE".split().index(letter)) if letter in "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE".split() else letter
            if len(base) > 1:   # greek names
                base = "\\" + (base.lower() if kind == "SMALL" else base.capitalize())
            cmd = "mathbb" if "DOUBLE-STRUCK" in style else "mathfrak" if "FRAKTUR" in style else "mathcal" if "SCRIPT" in style else "mathbf" if "BOLD" in style else None
            return base, cmd
    return None


SUPERS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", "0123456789+-=()ni"))
SUBS = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₙₘₖₗₚₛₜ", "0123456789+-=()aeoxnmklpst"))


def token(ch):
    """Return (latex, ok) for one glyph. ok=False for glyphs we cannot map."""
    c = ch.c
    if len(c) != 1:                     # an expanded ligature or a repaired multi-letter glyph: letters as they are
        return (c, True) if c.isalnum() else (c, False)
    if c in SUPERS:
        return "^{" + SUPERS[c] + "}", True
    if c in SUBS:
        return "_{" + SUBS[c] + "}", True
    if c in GREEK:
        return "\\" + GREEK[c], True
    if c in SYMBOLS:
        return SYMBOLS[c], True
    ma = _math_alnum(c)
    if ma:
        base, cmd = ma
        return (f"\\{cmd}{{{base}}}" if cmd else base), True
    if c == "\ufffd" or 0xE000 <= ord(c) <= 0xF8FF:
        return "", False
    cmd = _font_cmd(ch)
    if cmd and c.isalnum():
        return f"\\{cmd}{{{c}}}", True
    if c.isalnum() or c in "+-=<>()[],.;:!?'/*|@":
        return c, True
    cat = unicodedata.category(c)
    if cat.startswith(("L", "N")):
        return c, True
    return c, cat.startswith(("S", "P"))


# ---------------------------------------------------------------------------------------------
# Structure reconstruction
# ---------------------------------------------------------------------------------------------
class _Atom:
    __slots__ = ("x0", "y0", "x1", "y1", "base", "size", "latex", "kind", "ok")

    def __init__(self, x0, y0, x1, y1, base, size, latex, kind="ord", ok=True):
        self.x0, self.y0, self.x1, self.y1, self.base, self.size, self.latex, self.kind, self.ok = x0, y0, x1, y1, base, size, latex, kind, ok

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2


def is_tall(ch):
    """A display-size delimiter, integral or extensible piece: its glyph box is much taller than its font size."""
    if ch.size < 3:                       # degenerate font sizes (text drawn through a scaling matrix)
        return False
    return (ch.y1 - ch.y0) > 2.0 * ch.size and (PIECES.get(ch.c, ch.c) in DELIMS or ch.c in "∫∮∑∏") or ch.c in PIECES


def _glyph_box(ch):
    """Tighter vertical box than the font ascender/descender box: [baseline - 0.72 size, baseline + 0.22 size]."""
    return (ch.x0, ch.oy - 0.72 * ch.size, ch.x1, ch.oy + 0.22 * ch.size)


def _atoms_from_chars(chars):
    out = []
    for ch in chars:
        if not ch.c.strip():
            continue
        lx, ok = token(ch)
        if not ok and not lx:
            out.append(None)          # unmapped glyph (e.g. a big delimiter with no Unicode): counted, not placed
            continue
        x0, y0, x1, y1 = _glyph_box(ch)
        base_c = PIECES.get(ch.c, ch.c)
        if (base_c in DELIMS and (ch.y1 - ch.y0) > 2.0 * max(ch.size, 1)) or ch.c in PIECES:
            # a display-size delimiter or a piece of one: its real height says what it encloses
            out.append(_Atom(ch.x0, ch.y0, ch.x1, ch.y1, ch.oy, ch.size, base_c, "piece" if ch.c in PIECES else "delim", True))
            continue
        kind = "op" if lx in BIG_OPS else "int" if lx in INTEGRALS else "rad" if lx == r"\sqrt" else "ord"
        if kind in ("op", "int", "rad"):
            y0, y1 = ch.y0, ch.y1    # big glyphs: use the font box
        a = _Atom(x0, y0, x1, y1, ch.oy, ch.size, lx, kind, ok)
        out.append(a)
    return out


def build_latex(chars, hrules=(), display=False):
    """Reconstruct LaTeX for a set of glyphs (one inline run or one display equation).
    hrules: horizontal rule segments (x0, y, x1, y) near the glyphs (fraction bars, radical overbars).
    Returns (latex, confidence, notes). Never raises: an unexpected glyph combination gives confidence 0, so the
    caller keeps the plain text of that run instead of losing the page."""
    try:
        return _build_latex(chars, hrules, display)
    except Exception as e:
        return "", 0.0, [f"reconstruction failed ({type(e).__name__}); text kept"]


def _build_latex(chars, hrules=(), display=False):
    raw_atoms = _atoms_from_chars(chars)
    atoms = [a for a in raw_atoms if a is not None]
    if not atoms:
        return "", 0.0, ["no glyphs"]
    notes = []
    bad = sum(1 for a in raw_atoms if a is None or not a.ok)
    rules = sorted([r for r in hrules], key=lambda r: r[2] - r[0])   # shortest first: inner fractions before outer
    used_rules = []
    atoms = _dot_stacks(atoms)
    atoms = _collapse_radicals(atoms, rules, used_rules, notes)
    atoms = _collapse_fractions(atoms, [r for r in rules if r not in used_rules], notes)
    atoms = _collapse_bigops(atoms, notes)
    atoms = _collapse_integrals(atoms, notes)
    atoms = _merge_pieces(atoms)
    size = _main_level(atoms)[1]            # the main text size (scripts and limits are smaller)
    atoms = _structures(atoms, size, notes)
    lines = _lines(atoms, size)
    if display and len(lines) >= 2:
        latex = _multiline(lines, size, notes)
    else:
        latex = _linear(atoms, notes)
    latex = _post(latex)
    conf = 0.9 - 0.5 * (bad / len(atoms)) - (0.1 if any(n.startswith("script") for n in notes) else 0) - (0.1 if any(n.startswith("fraction") for n in notes) else 0)
    if bad:
        notes.append(f"{bad} glyph(s) without a Unicode mapping")
    return latex, round(max(0.05, conf), 2), notes


DELIMS = set("()[]{}|‖⌊⌋⌈⌉⟨⟩")
PIECES = {"⎛": "(", "⎜": "(", "⎝": "(", "⎞": ")", "⎟": ")", "⎠": ")", "⎡": "[", "⎢": "[", "⎣": "[", "⎤": "]", "⎥": "]",
          "⎦": "]", "⎧": "{", "⎨": "{", "⎩": "{", "⎪": "{", "⎫": "}", "⎬": "}", "⎭": "}"}
OPENERS, CLOSERS = "([{⌊⌈⟨", ")]}⌋⌉⟩"
PAIR = {"(": ")", "[": "]", "{": "}", "⌊": "⌋", "⌈": "⌉", "⟨": "⟩", "|": "|", "‖": "‖"}
DL = {"(": "(", ")": ")", "[": "[", "]": "]", "{": r"\{", "}": r"\}", "|": "|", "‖": r"\|", "⌊": r"\lfloor", "⌋": r"\rfloor",
      "⌈": r"\lceil", "⌉": r"\rceil", "⟨": r"\langle", "⟩": r"\rangle"}
ENV = {"(": "pmatrix", "[": "bmatrix", "{": "Bmatrix", "|": "vmatrix", "‖": "Vmatrix"}
RELS = {"=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv", r"\sim", r"\simeq", r"\cong", r"\le", r"\ge",
        r"\ll", r"\gg", r"\prec", r"\succ", r"\preceq", r"\succeq", r"\subset", r"\subseteq", r"\supset", r"\supseteq",
        r"\in", r"\notin", r"\propto", r"\to", r"\Rightarrow", r"\Leftrightarrow", r"\iff", ":="}
TEXT_WORDS = re.compile(r"(?<![\\A-Za-z{])(if|otherwise|for|and|or|where|when|then|else|all|some|each|such|that|with|as|is|not)(?![A-Za-z}])")


def _dot_stacks(atoms):
    """Three dots drawn one above the other (or diagonally) are one symbol: \\vdots or \\ddots."""
    dots = sorted([a for a in atoms if a.latex in (".", r"\cdot", "·") and a.kind == "ord"], key=lambda a: (a.x0, a.y0))
    used, out = set(), []
    for a in dots:
        if id(a) in used:
            continue
        for b in dots:
            if id(b) in used or b is a or not (0.15 * a.size < b.base - a.base < 0.7 * a.size):
                continue
            dx = b.x0 - a.x0
            c = next((c for c in dots if c is not a and c is not b and id(c) not in used
                      and abs((c.base - b.base) - (b.base - a.base)) < 0.12 * a.size and abs((c.x0 - b.x0) - dx) < 0.12 * a.size), None)
            if c is None:
                continue
            if abs(dx) < 0.12 * a.size:
                lx = r"\vdots"
            elif 0.15 * a.size < dx < 0.7 * a.size:
                lx = r"\ddots"
            else:
                continue
            used.update((id(a), id(b), id(c)))
            out.append(_Atom(min(a.x0, c.x0), a.y0, max(a.x1, c.x1), c.y1, c.base, a.size, lx, "ord"))
            break
    if not out:
        return atoms
    return [a for a in atoms if id(a) not in used] + out


def _merge_pieces(atoms):
    """Stack the pieces of an extensible delimiter (top, extension, bottom) into one delimiter."""
    ds = sorted([a for a in atoms if a.kind == "piece"], key=lambda a: (a.x0, a.y0))
    rest = [a for a in atoms if a.kind != "piece"]
    out = []
    for d in ds:
        prev = next((o for o in out if o.latex == d.latex and min(o.x1, d.x1) - max(o.x0, d.x0) > 0.5 * min(o.x1 - o.x0, d.x1 - d.x0)
                     and -0.5 * d.size < d.y0 - o.y1 < 0.4 * d.size), None)
        if prev:
            prev.y1 = max(prev.y1, d.y1); prev.x0 = min(prev.x0, d.x0); prev.x1 = max(prev.x1, d.x1)
        else:
            out.append(_Atom(d.x0, d.y0, d.x1, d.y1, d.base, d.size, d.latex, "delim", True))
    return rest + out


def _vov(a, b):
    return (min(a.y1, b.y1) - max(a.y0, b.y0)) / max(1e-3, min(a.y1 - a.y0, b.y1 - b.y0))


def _lines(atoms, size):
    """Text lines of a display: clusters of main-size baselines at least 0.7 em apart; every atom joins the line
    whose band holds its centre (scripts stay with their line)."""
    mains = [a for a in atoms if a.kind not in ("delim",) and (a.size >= 0.85 * size or a.kind in ("frac", "opl", "env"))]
    if not mains:
        return [atoms] if atoms else []
    bases = sorted(a.base for a in mains)
    clusters = [[bases[0]]]
    for b in bases[1:]:
        if b - clusters[-1][-1] > 0.7 * size:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    cb = [statistics.median(c) for c in clusters]
    # a cluster holding only scripts or limits next to a fuller line is not a line of its own
    if len(cb) > 1:
        counts = [sum(1 for a in mains if min(range(len(cb)), key=lambda k: abs(a.base - cb[k])) == k) for k in range(len(cb))]
        cb = [b for b, n in zip(cb, counts) if n >= 1] or cb[:1]
    lines = [[] for _ in cb]
    for a in atoms:
        k = min(range(len(cb)), key=lambda k: abs((a.base if a.kind != "delim" else a.cy) - cb[k]))
        lines[k].append(a)
    return [l for l in lines if l]


def _columns(rows, size):
    """Column boundaries shared by all rows: x ranges free of ink in every row, at least 0.6 em wide."""
    iv = sorted((a.x0, a.x1) for r in rows for a in r)
    if not iv:
        return []
    cuts, end = [], iv[0][1]
    for x0, x1 in iv[1:]:
        if x0 - end > 0.6 * size:
            cuts.append((end + x0) / 2)
        end = max(end, x1)
    return cuts


def _split_row_entries(r, size):
    """Split a row of atoms into candidate column cells based on horizontal gaps."""
    r = sorted(r, key=lambda a: a.x0)
    if not r:
        return []
    entries = [[r[0]]]
    for a, b in zip(r, r[1:]):
        if b.x0 - a.x1 > 0.55 * size:
            entries.append([b])
        else:
            entries[-1].append(b)
    return entries


def _grid(rows, size, notes):
    cuts = _columns(rows, size)
    if cuts:
        out = []
        for r in rows:
            cells = [[] for _ in range(len(cuts) + 1)]
            for a in r:
                k = sum(1 for c in cuts if a.cx > c)
                cells[k].append(a)
            out.append(" & ".join(_linear(c, notes) for c in cells))
        return out

    row_entries = [_split_row_entries(r, size) for r in rows]
    counts = [len(e) for e in row_entries]
    max_c = max(counts) if counts else 0
    if max_c >= 2 and all(c == max_c for c in counts):
        out = []
        for entries in row_entries:
            out.append(" & ".join(_linear(e, notes) for e in entries))
        return out

    if max_c >= 2:
        centers = []
        for entries in row_entries:
            for e in entries:
                cx = (min(a.x0 for a in e) + max(a.x1 for a in e)) / 2
                centers.append(cx)
        centers.sort()
        cuts = []
        if len(centers) >= 2:
            col_clusters = [[centers[0]]]
            for c in centers[1:]:
                if c - col_clusters[-1][-1] > 1.2 * size:
                    col_clusters.append([c])
                else:
                    col_clusters[-1].append(c)
            if len(col_clusters) >= 2:
                for c1, c2 in zip(col_clusters, col_clusters[1:]):
                    cuts.append((max(c1) + min(c2)) / 2)
        if cuts:
            out = []
            for r in rows:
                cells = [[] for _ in range(len(cuts) + 1)]
                for a in r:
                    k = sum(1 for c in cuts if a.cx > c)
                    cells[k].append(a)
                out.append(" & ".join(_linear(c, notes) for c in cells))
            return out

    return [_linear(r, notes) for r in rows]


def _structures(atoms, size, notes):
    """Matrices, \\left..\\right groups and cases from display-size delimiters (innermost first)."""
    atoms = list(atoms)
    for _ in range(40):
        ds = sorted([a for a in atoms if a.kind == "delim"], key=lambda a: a.x0)
        pair = None
        for i, d in enumerate(ds):
            if d.latex not in OPENERS and d.latex not in "|‖":
                continue
            want = PAIR[d.latex]
            for e in ds[i + 1:]:
                if _vov(d, e) < 0.6:
                    continue
                if e.latex == want or (e.latex in CLOSERS and d.latex in OPENERS):
                    between = [x for x in ds if d.x1 <= x.x0 and x.x1 <= e.x0 and _vov(d, x) >= 0.6]
                    if not between:
                        pair = (d, e)
                    break
            if pair:
                break
        if not pair:
            break
        d, e = pair
        opens = [x for x in ds if x.latex in OPENERS or x.latex in "|‖"]

        def owner(a):
            # stacked displays touch: a glyph belongs to the delimiter whose middle is nearest
            cands = [x for x in opens if x.x0 < a.cx and x.y0 - 0.2 * size <= a.cy <= x.y1 + 0.2 * size]
            return min(cands, key=lambda x: (abs(a.cy - x.cy) // (0.5 * size), -x.x0)) if cands else None
        inner = [a for a in atoms if a is not d and a is not e and a.kind != "delim" and d.x1 - 1 <= a.cx <= e.x0 + 1
                 and d.y0 - 0.2 * size <= a.cy <= d.y1 + 0.2 * size and owner(a) in (d, None)]
        rows = _lines(inner, size)
        if len(rows) >= 2:
            env = ENV.get(d.latex, "matrix") if PAIR.get(d.latex) == e.latex else "matrix"
            body = " \\\\ ".join(_grid(rows, size, notes))
            lx = f"\\begin{{{env}}} {body} \\end{{{env}}}"
            if env == "matrix":
                lx = f"\\left{DL[d.latex]} {lx} \\right{DL[e.latex]}"
            notes.append("matrix reconstructed from bracketed grid")
        else:
            lx = f"\\left{DL[d.latex]} {_linear(inner, notes)} \\right{DL[e.latex]}"
            notes.append("big delimiters paired")
        # scripts on the closing delimiter ('[ ... ]_1^n'): small glyphs at its upper / lower right corner
        h = e.y1 - e.y0
        corner = [a for a in atoms if a.kind == "ord" and a.size < 0.85 * size and e.x1 - 0.3 * size <= a.x0 <= e.x1 + 0.9 * size
                  and a not in inner]
        sup = sorted([a for a in corner if abs(a.cy - (e.y0 + 0.12 * h)) < 0.3 * h], key=lambda a: a.x0)
        sub = sorted([a for a in corner if abs(a.cy - (e.y1 - 0.12 * h)) < 0.3 * h and a not in sup], key=lambda a: a.x0)
        if sub:
            lx += f"_{{{_linear(sub, notes)}}}"
        if sup:
            lx += f"^{{{_linear(sup, notes)}}}"
        mid = (d.y0 + max(d.y1, e.y1)) / 2 + 0.25 * size       # delimiters are centred on the maths axis
        m = _Atom(d.x0, d.y0, max([e.x1] + [a.x1 for a in sup + sub]), max(d.y1, e.y1), mid, size, lx, "env")
        atoms = [a for a in atoms if a is not d and a is not e and a not in inner and a not in sup and a not in sub] + [m]
    # cases: a tall '{' with no partner and two or more lines to its right
    for d in sorted([a for a in atoms if a.kind == "delim" and a.latex == "{"], key=lambda a: a.x0):
        inner = [a for a in atoms if a is not d and a.kind != "delim" and a.x0 >= d.x1 - 1 and d.y0 - 0.2 * size <= a.cy <= d.y1 + 0.2 * size]
        rows = _lines(inner, size)
        if len(rows) < 2:
            continue
        body = []
        starts = []
        for r in rows:
            r = sorted(r, key=lambda a: a.x0)
            gaps = [(b.x0 - a.x1, b.x0) for a, b in zip(r, r[1:])]
            g = max(gaps) if gaps else (0, None)
            if g[0] > 1.2 * size:
                starts.append(g[1])
        col = min(starts) if starts else None          # the conditions start in one column
        for r in rows:
            if col is None:
                body.append(_linear(r, notes)); continue
            v = [a for a in r if a.x0 < col - 0.3 * size]
            c = [a for a in r if a.x0 >= col - 0.3 * size]
            cl = TEXT_WORDS.sub(lambda m: chr(92) + "text{" + m.group(1) + " }", _linear(c, notes)) if c else ""
            body.append(f"{_linear(v, notes)} & {cl}" if c else _linear(v, notes))
        lx = "\\begin{cases} " + " \\\\ ".join(body) + " \\end{cases}"
        m = _Atom(d.x0, d.y0, max(a.x1 for a in inner), d.y1, d.cy + 0.25 * size, size, lx, "env")
        atoms = [a for a in atoms if a is not d and a not in inner] + [m]
        notes.append("cases reconstructed from brace")
    # any delimiter left alone keeps a size command (KaTeX accepts \Bigg( without a partner)
    out = []
    for a in atoms:
        if a.kind == "delim":
            a.kind, a.latex = "ord", ("\\Bigg" + DL.get(a.latex, a.latex)) if a.latex in DL else a.latex
            a.base = a.cy + 0.25 * size
            a.size = size
        out.append(a)
    return out


def _multiline(lines, size, notes):
    """Several lines in one display: aligned at their first relation when they have one, else gathered."""
    built = []
    for ln in lines:
        ln = sorted(ln, key=lambda a: a.x0)
        rel = next((k for k, a in enumerate(ln) if a.latex in RELS and a.size >= 0.85 * size), None)
        if rel is None:
            built.append((None, _linear(ln, notes)))
        else:
            built.append((_linear(ln[:rel], notes), _linear(ln[rel:], notes)))
    if sum(1 for b in built if b[0] is not None) >= max(2, len(built) - 1):
        rows = [f"{l} & {r}" if l is not None else f"& {r}" for l, r in built]
        notes.append("aligned lines")
        return "\\begin{aligned} " + " \\\\ ".join(rows) + " \\end{aligned}"
    notes.append("lines gathered")
    return "\\begin{gathered} " + " \\\\ ".join((r if l is None else f"{l} {r}") for l, r in built) + " \\end{gathered}"


def _post(s):
    s = re.sub(r"\s+", " ", s).strip()
    # numbers arrive glyph by glyph: '2 8 . 1' -> '28.1', 'a_{1 1}' -> 'a_{11}'
    for _ in range(3):
        s = re.sub(r"(?<![\\A-Za-z])(\d) (?=\d)", r"\1", s)
        s = re.sub(r"(\d) \. (?=\d)", r"\1.", s)
    s = s.replace(r"\cdot \cdot \cdot", r"\cdots").replace(". . .", r"\ldots")
    s = re.sub(r"_\{([0-9A-Za-z ]{1,12})\}", lambda m: "_{" + m.group(1).replace(" ", "") + "}", s)     # a_{1 n} -> a_{1n}
    s = re.sub(r"([\^_])\{([-−]) (\w)", r"\1{\2\3", s)                                                    # ^{- 1} -> ^{-1}
    # Unicode script characters arrive one by one: x ^{2} ^{3} -> x^{23}; and attach to the preceding atom
    for k in ("^", "_"):
        s = re.sub(r"(?:\s*\%s\{([^{}]*)\}){2,}" % k, lambda m: k + "{" + "".join(re.findall(r"\%s\{([^{}]*)\}" % k, m.group(0))) + "}", s)
        s = re.sub(r"\s+\%s\{" % k, k + "{", s)
    s = re.sub(r"\\operatorname\{(" + "|".join(sorted(FUNCS, key=len, reverse=True)) + r")\}", r"\\\1", s)
    s = s.replace(r"\Pr", r"\Pr").replace(r"\mod", r"\bmod")
    return s


def _collapse_radicals(atoms, rules, used, notes):
    out = list(atoms)
    for a in [x for x in atoms if x.kind == "rad"]:
        bar = None
        for r in rules:
            if abs(r[0] - a.x1) < 0.6 * a.size + 1 and a.y0 - 0.4 * a.size <= r[1] <= a.y0 + 0.6 * a.size:
                bar = r; break
        if bar is None:
            # no overbar: a bare radical sign applies to the next atom
            continue
        inside = [x for x in out if x is not a and bar[0] - 1 <= x.cx <= bar[2] + 1 and x.y0 >= bar[1] - 1.5 and x.y1 <= a.y1 + 1]
        index = [x for x in out if x is not a and x.x1 <= a.cx and x.y1 < a.cy and x.size < 0.8 * a.size and x.x0 >= a.x0 - a.size]
        body = _linear(inside, notes)
        idx = _linear(index, notes) if index else ""
        lx = f"\\sqrt[{idx}]{{{body}}}" if idx else f"\\sqrt{{{body}}}"
        used.append(bar)
        keep = [x for x in out if x is not a and x not in inside and x not in index]
        m = _Atom(a.x0, min([a.y0] + [x.y0 for x in inside]), bar[2], a.y1, max([a.base] + [x.base for x in inside]), max(x.size for x in inside) if inside else a.size, lx, "ord")
        out = keep + [m]
        notes.append("radical reconstructed from overbar rule")
    return out


def _collapse_fractions(atoms, rules, notes):
    out = list(atoms)
    for r in rules:
        x0, y, x1 = r[0], r[1], r[2]
        span = [a for a in out if x0 - 1.5 <= a.cx <= x1 + 1.5 and a.kind not in ("delim", "int", "op")]
        if not span:
            continue
        sz = statistics.median(a.size for a in span)
        span = [a for a in span if abs(a.cy - y) < 2.2 * sz]       # numerator and denominator sit on the rule
        num = _adjacent(sorted([a for a in span if a.cy < y], key=lambda a: -a.y1), y, sz, up=True)
        den = _adjacent(sorted([a for a in span if a.cy > y], key=lambda a: a.y0), y, sz, up=False)
        if not num or not den:
            continue
        def spread(g):
            g = sorted(g, key=lambda a: a.x0)
            return sum(1 for a, b in zip(g, g[1:]) if b.x0 - a.x1 > 0.6 * sz)
        if spread(num) >= 2 and spread(den) >= 2:
            continue            # a partition line in a block matrix, not a fraction bar
        nl, dl = _linear(num, notes), _linear(den, notes)
        if not nl or not dl:
            continue
        rest = [a for a in out if a not in num and a not in den]
        base = y + 0.25 * sz
        m = _Atom(min(x0, *(a.x0 for a in num + den)), min(a.y0 for a in num), max(x1, *(a.x1 for a in num + den)), max(a.y1 for a in den),
                  base, sz, f"\\frac{{{nl}}}{{{dl}}}", "frac")
        out = rest + [m]
        notes.append("fraction reconstructed from rule")
    return out


def _adjacent(cands, y, sz, up):
    """Take the glyphs vertically connected to the rule (stop at the first vertical gap > 0.6 size)."""
    picked, edge = [], y
    for a in cands:
        d = (edge - a.y1) if up else (a.y0 - edge)
        if d > 0.75 * sz and picked:
            break
        if d > 1.2 * sz and not picked:
            break
        picked.append(a)
        edge = min(edge, a.y0) if up else max(edge, a.y1)
    return picked


def _collapse_bigops(atoms, notes):
    out = list(atoms)
    for a in sorted([x for x in atoms if x.kind == "op"], key=lambda x: x.x0):
        if a not in out:
            continue
        w = a.x1 - a.x0
        lim = [x for x in out if x is not a and a.x0 - 0.6 * w <= x.cx <= a.x1 + 0.6 * w and x.size < 0.9 * a.size]
        below = _adjacent(sorted([x for x in lim if x.cy > a.y1 - 0.15 * a.size], key=lambda x: x.y0), a.y1, a.size * 0.6, up=False)
        above = _adjacent(sorted([x for x in lim if x.cy < a.y0 + 0.15 * a.size], key=lambda x: -x.y1), a.y0, a.size * 0.6, up=True)
        below = _extend_row(below, out, a)
        above = _extend_row(above, out, a)
        if not below and not above:
            continue
        lx = a.latex + (f"_{{{_linear(below, notes)}}}" if below else "") + (f"^{{{_linear(above, notes)}}}" if above else "")
        rest = [x for x in out if x is not a and x not in below and x not in above]
        all_ = [a] + below + above
        main_base = a.base if a.y1 - a.y0 < 1.3 * a.size else (a.y0 + a.y1) / 2 + 0.25 * a.size    # display size: on the axis
        m = _Atom(min(x.x0 for x in all_), min(x.y0 for x in all_), max(x.x1 for x in all_), max(x.y1 for x in all_), main_base, a.size, lx, "opl")
        out = rest + [m]
        notes.append("big operator limits reconstructed")
    return out


def _collapse_integrals(atoms, notes):
    """A display integral carries its limits at its upper and lower right corners (TeX's default placement). Each
    small glyph near a corner goes to the nearest corner of the nearest integral (stacked rows of integrals touch)."""
    ints = [x for x in atoms if x.kind == "int" and x.y1 - x.y0 >= 1.5 * x.size]
    if not ints:
        return atoms
    claim = {}
    for x in atoms:
        if x.kind != "ord":
            continue
        best = None
        for a in ints:
            h = a.y1 - a.y0
            if not (a.cx <= x.x0 <= a.x1 + 0.9 * a.size and x.size < 0.85 * a.size):
                continue
            for role, yref in (("up", a.y0 + 0.12 * h), ("lo", a.y1 - 0.12 * h)):
                d = abs(x.cy - yref)
                if d < 0.3 * h and (best is None or d < best[0]):
                    best = (d, id(a), role)
        if best:
            claim[id(x)] = best[1:]
    out = list(atoms)
    for a in sorted(ints, key=lambda x: x.x0):
        mine = lambda role: sorted([x for x in out if claim.get(id(x)) == (id(a), role)], key=lambda x: x.x0)
        up, lo = mine("up"), mine("lo")
        if not up and not lo:
            continue
        lx = a.latex + (f"_{{{_linear(lo, notes)}}}" if lo else "") + (f"^{{{_linear(up, notes)}}}" if up else "")
        grp = [a] + up + lo
        m = _Atom(min(x.x0 for x in grp), a.y0, max(x.x1 for x in grp), a.y1, (a.y0 + a.y1) / 2 + 0.25 * a.size, a.size, lx, "opl")
        out = [x for x in out if x not in grp] + [m]
        notes.append("integral limits reconstructed")
    return out


def _extend_row(group, pool, op):
    """Limits are often wider than the operator glyph: grow the group sideways along its own band."""
    if not group:
        return group
    group = list(group)
    changed = True
    while changed:
        changed = False
        gy0 = min(x.y0 for x in group); gy1 = max(x.y1 for x in group)
        gx0 = min(x.x0 for x in group); gx1 = max(x.x1 for x in group)
        for x in pool:
            if x is op or x in group or x.size >= 0.9 * op.size or x.kind in ("op", "opl"):
                continue
            if gy0 - 0.2 * x.size <= x.cy <= gy1 + 0.2 * x.size and (gx0 - 0.6 * x.size <= x.x1 and x.x0 <= gx1 + 0.6 * x.size):
                group.append(x); changed = True
    return group


def _main_level(atoms):
    norm = [a for a in atoms if a.kind not in ("op", "int", "rad", "opl", "delim", "env")]
    pool = norm or atoms
    big = max(a.size for a in pool)
    cand = [a for a in pool if a.size >= 0.9 * big]
    bases = sorted(a.base for a in cand)
    # most populated baseline cluster
    best, best_n = bases[0], 0
    for b in bases:
        n = sum(1 for x in bases if abs(x - b) < 0.12 * big)
        if n > best_n:
            best, best_n = b, n
    return best, big


def _linear(atoms, notes):
    """Left-to-right serialisation with sub/superscripts relative to the main baseline."""
    if not atoms:
        return ""
    base, size = _main_level(atoms)
    atoms = sorted(atoms, key=lambda a: (a.x0, a.y0))
    mains, scripts = [], []
    for a in atoms:
        if a.kind in ("frac", "opl", "op", "int", "env") or a.size >= 0.9 * size and abs(a.base - base) < 0.18 * size:
            mains.append(a)
        elif abs(a.base - base) < 0.12 * size and a.size >= 0.8 * size:
            mains.append(a)
        else:
            scripts.append(a)
    # assign each script atom to the main atom immediately to its left
    attach = {}
    for s in scripts:
        host = None
        for m in mains:
            if m.x0 <= s.x0 + 0.2 * size:
                host = m
        key = id(host) if host else None
        pos = "sup" if s.base < base - 0.15 * size else "sub"
        attach.setdefault(key, {"sup": [], "sub": []})[pos].append(s)
    parts = []
    if None in attach:      # scripts before any main atom (pre-scripts): emit as {}^{..}
        pre = attach[None]
        parts.append("{}" + (f"_{{{_linear(pre['sub'], notes)}}}" if pre["sub"] else "") + (f"^{{{_linear(pre['sup'], notes)}}}" if pre["sup"] else ""))
    prev = None
    for m in mains:
        lx = m.latex
        if prev is not None:
            gap = m.x0 - prev.x1
            if _wordy(prev.latex) and _wordy(lx) and gap < 0.12 * size:
                parts[-1] = parts[-1] + lx      # same word (e.g. upright function name)
                prev = m
                sc = attach.get(id(m))
                if sc:
                    parts[-1] += _scripts(sc, notes)
                continue
        parts.append(lx)
        sc = attach.get(id(m))
        if sc:
            parts[-1] += _scripts(sc, notes)
            notes.append("script placement from baseline offsets")
        prev = m
    out = " ".join(p for p in parts if p)
    out = _functions(out)
    return out


def _scripts(sc, notes):
    s = ""
    if sc["sub"]:
        s += "_{" + _linear(sc["sub"], notes) + "}"
    if sc["sup"]:
        sup = _linear(sc["sup"], notes)
        s += "^{" + sup + "}" if sup not in ("'", "''") else sup
    return s


def _wordy(lx):
    return bool(re.fullmatch(r"[A-Za-z]+", lx))


def _functions(s):
    def rep(m):
        w = m.group(0)
        if w in FUNCS:
            return "\\" + w if w not in ("mod",) else r"\bmod"
        if len(w) >= 3 and w[0].isupper() and w[1:].islower():
            return r"\operatorname{" + w + "}"
        return w
    return re.sub(r"(?<!\\operatorname\{)(?<![\\A-Za-z{])[A-Za-z]{2,}(?![A-Za-z])", rep, s)


# ---------------------------------------------------------------------------------------------
# Detection helpers used by the pipeline
# ---------------------------------------------------------------------------------------------
_TEXT_PUNCT = set(".,;:!?'\"’‘“”")


def _tokens(chars):
    """Split a line's chars into whitespace-separated tokens: list of lists of chars (spaces kept separately)."""
    toks, cur = [], []
    for c in chars:
        if not c.c.strip():
            if cur:
                toks.append(cur); cur = []
            toks.append([c]); continue
        if cur and c.x0 - cur[-1].x1 > 0.25 * max(c.size, cur[-1].size, 1):
            toks.append(cur); cur = []
        cur.append(c)
    if cur:
        toks.append(cur)
    return toks


_FOOT = dict(zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
DINGBATS = set("★☆•◦▪■□●○◆◇✓✗†‡§¶")       # section stars, bullets, footnote marks: not mathematics


def _is_strong(c, main):
    if c.c in DINGBATS:
        return False
    if c.c in GREEK or (c.c in SYMBOLS and c.c not in "{}#%&_$\\~^|•·…°–—-" ) or c.c in "=<>+−":
        return True
    if MATH_FONT.search(c.font) and not TEXTY_SYMBOL_FONT.search(c.font) and c.c.strip() and c.c not in _TEXT_PUNCT:
        return True
    if len(c.c) != 1:                 # multi-code-point glyphs (ligatures, combining sequences) are text
        return False
    cp = ord(c.c)
    return 0x2200 <= cp <= 0x22FF or 0x1D400 <= cp <= 0x1D7FF or 0x27C0 <= cp <= 0x27EF or 0x2980 <= cp <= 0x2AFF


def token_class(tok, main):
    """'strong' (contains a math glyph), 'weak' (math-compatible: italic letters, digits, brackets, scripts) or 'text'."""
    vis = [c for c in tok if c.c.strip()]
    if not vis:
        return "space"
    word = "".join(c.c for c in vis)
    if URLISH.search(word):
        return "text"      # links, e-mail addresses, DOIs: '=' '?' '&' '/' inside a URL are not mathematics
    # operator glyphs only at the edges of a real word: quotation marks that a font maps to '<' '=' ('“problems.”'
    # extracted as '<problems.=') or a word glued to punctuation; the word itself carries no math
    k0, k1 = 0, len(vis)
    while k0 < k1 and (_is_strong(vis[k0], main) or vis[k0].c in "\"'“”‘’«»"):
        k0 += 1
    while k1 > k0 and (_is_strong(vis[k1 - 1], main) or vis[k1 - 1].c in "\"'“”‘’«».,;:!?"):
        k1 -= 1
    core = vis[k0:k1]
    if (k0 or k1 < len(vis)) and core and not any(_is_strong(c, main) for c in core):
        m = re.fullmatch(r"[\"'“‘(\-‐–]?([A-Za-z]{4,})[.,;:!?'\"”’)]*", "".join(c.c for c in core))
        if m and m.group(1).lower() not in FUNCS and not any(MATH_FONT.search(c.font) for c in core) \
                and not any(c.size < 0.85 * main for c in core):
            return "text"
    letters0 = [c for c in vis if c.c.isalpha()]
    if len(letters0) >= 3 and all(c.c in GREEK or "\u0370" <= c.c <= "\u03ff" or "\u1f00" <= c.c <= "\u1fff" for c in letters0) \
            and not any(c.c in SYMBOLS or c.c in "=<>+" for c in vis) and not MATH_FONT.search(vis[0].font):
        return "text"      # a Greek-language word, not a formula
    if any(_is_strong(c, main) for c in vis):
        return "strong"
    letters = [c for c in vis if c.c.isalpha()]
    small = [c for c in vis if c.size < 0.85 * main]
    if not letters:
        return "weak" if all(c.c.isdigit() or c.c in "()[]/,.'|′:" or c.size < 0.85 * main for c in vis) else "text"
    # glyphs from Type 3 fonts (how TeX math fonts often end up in converted PDFs) count as math-compatible
    ital = all((c.flags & 2) or re.search(r"(ital|oblique|cmmi|-it)", c.font, re.I) or c.font.startswith("Type3") for c in letters)
    big_letters = [c for c in letters if c.size >= 0.85 * main]
    if ital and len(big_letters) <= 2:
        return "weak"
    if small and big_letters and len(big_letters) <= 2 and any(c.size < 0.85 * main for c in letters):
        return "weak"      # e.g. 'Rs' with a subscript
    if small and any(c.c in "()[]/=^_" for c in vis) and len(big_letters) <= 6:
        return "weak"      # e.g. 'R/Ann_R(x_i)': scripts plus brackets/slash only occur in formulas
    return "text"


def _unmatched_close(cs):
    """The run's last char closes a bracket that the run never opened (prose '(that is, A[i − 1])')."""
    t = "".join(c.c for c in cs)
    pair = {")": "(", "]": "["}[t[-1]]
    return t.count(t[-1]) > t.count(pair)


def _short_operand(tok):
    w = "".join(c.c for c in tok if c.c.strip())
    return bool(w) and not URLISH.search(w) and not re.search(r"[A-Za-z]{3,}", w) and len(w) <= 6


def _balance_runs(toks, cls):
    """A math run that closes a bracket it never opened ('A[1 : i − 1]' where only '− 1]' looked like math) is
    extended over the short operands before it, up to the opening bracket; likewise to the right for an opener."""
    n = len(toks)
    i = 0
    while i < n:
        if cls[i] not in ("strong", "weak"):
            i += 1
            continue
        j = i
        while j + 1 < n and (cls[j + 1] in ("strong", "weak") or (cls[j + 1] == "space" and j + 2 < n and cls[j + 2] in ("strong", "weak"))):
            j += 1
        if any(cls[k] == "strong" for k in range(i, j + 1)):
            txt = "".join(c.c for k in range(i, j + 1) for c in toks[k])
            need = sum(txt.count(b) for b in ")]") - sum(txt.count(b) for b in "([")
            k = i - 1
            ext = []
            while need > 0 and k >= 0 and (cls[k] == "space" or _short_operand(toks[k])):
                ext.append(k)
                w = "".join(c.c for c in toks[k])
                need -= sum(w.count(b) for b in "([") - sum(w.count(b) for b in ")]")
                k -= 1
            if need <= 0 and ext:
                for e in ext:
                    if cls[e] != "space":
                        cls[e] = "weak"
            need = sum(txt.count(b) for b in "([") - sum(txt.count(b) for b in ")]")
            k = j + 1
            ext = []
            while need > 0 and k < n and (cls[k] == "space" or _short_operand(toks[k])):
                ext.append(k)
                w = "".join(c.c for c in toks[k])
                need -= sum(w.count(b) for b in ")]") - sum(w.count(b) for b in "([")
                k += 1
            if need <= 0 and ext:
                for e in ext:
                    if cls[e] != "space":
                        cls[e] = "weak"
                j = max(ext)
        i = j + 1


def inline_runs(chars):
    """Split a text line's chars into [(is_math, [chars])]. A math run is a maximal sequence of strong/weak tokens
    containing at least one strong token (a real math glyph); prose words always end a run, so ordinary italic
    words or footnote markers are never wrapped. Trailing sentence punctuation stays outside the math."""
    if not chars:
        return []
    sizes = [c.size for c in chars if c.c.strip()]
    main = statistics.median(sizes) if sizes else 10
    toks = _tokens(chars)
    cls = [token_class(t, main) for t in toks]
    _balance_runs(toks, cls)
    out = []
    i = 0
    n = len(toks)
    while i < n:
        if cls[i] in ("strong", "weak"):
            j = i
            while j + 1 < n and (cls[j + 1] in ("strong", "weak") or (cls[j + 1] == "space" and j + 2 < n and cls[j + 2] in ("strong", "weak"))):
                j += 1
            run = [k for k in range(i, j + 1)]
            if any(cls[k] == "strong" for k in run):
                cs = [c for k in run for c in toks[k]]
                # keep sentence punctuation that ends the run as text
                tail = []
                # a footnote marker after the sentence's full stop ('... when i = 2.²'): small raised digits that
                # follow sentence punctuation are text, not an exponent
                vis = [c for c in cs if c.c.strip()]
                k = len(vis)
                while k > 0 and vis[k - 1].c.isdigit() and vis[k - 1].size < 0.8 * main:
                    k -= 1
                if 0 < k < len(vis) and vis[k - 1].c in ".,;:" and vis[k - 1].size >= 0.85 * main \
                        and all(c.oy < vis[k - 1].oy - 0.2 * main for c in vis[k:]):
                    import dataclasses
                    cut = cs.index(vis[k - 1]) + 1
                    tail = [dataclasses.replace(c, c=_FOOT.get(c.c, c.c)) for c in cs[cut:] if c.c.strip()]
                    cs = cs[:cut]
                while cs and (not cs[-1].c.strip() or (cs[-1].c in ".,;:" and cs[-1].size >= 0.85 * main)
                              or (cs[-1].c in ")]" and _unmatched_close(cs))):
                    tail.insert(0, cs.pop())
                # a lone '=' or '+' between prose words is still math, but a run made only of punctuation is not
                if [c for c in cs if c.c.strip()]:
                    out.append((True, cs))
                if tail:
                    out.append((False, tail))
            else:
                out.append((False, [c for k in run for c in toks[k]]))
            i = j + 1
        else:
            out.append((False, toks[i]))
            i += 1
    merged = []
    for m, cs in out:
        if merged and merged[-1][0] == m and not m:
            merged[-1] = (m, merged[-1][1] + cs)
        else:
            merged.append((m, cs))
    return merged


def line_math_profile(chars):
    """(n_strong, n_weak, n_text) token counts for a line - used to spot display equations."""
    sizes = [c.size for c in chars if c.c.strip()]
    main = statistics.median(sizes) if sizes else 10
    cls = [token_class(t, main) for t in _tokens(chars)]
    return cls.count("strong"), cls.count("weak"), cls.count("text")


EQ_NUMBER = re.compile(r"^\(\s*(?:[A-Z]?\d+(?:\.\d+)*[a-z]?|[ivx]+|\*+|[A-Z])\s*\)$")


EQNO = re.compile(r"^\(\s*[A-Z]?\d+(?:\.\d+)*[a-z]?\s*\)$|^\(\s*[ivx]+\s*\)$")


def take_equation_numbers(chars):
    """Remove every '(28.8)'-style label that ends a line of a display (set apart by a wide gap). Returns
    (remaining chars, [labels top to bottom])."""
    vis = [c for c in chars if c.c.strip()]
    lines = {}
    for c in vis:
        if is_tall(c):
            continue
        lines.setdefault(round(c.oy / 2.5), []).append(c)
    drop, labels = set(), []
    for key in sorted(lines):
        ln = sorted(lines[key], key=lambda c: c.x0)
        k = len(ln)
        while k > 0 and ln[k - 1].c != "(":
            k -= 1
        if k == 0:
            continue
        tail = ln[k - 1:]
        txt = "".join(c.c for c in tail)
        if not EQNO.match(txt):
            continue
        left = [c for c in vis if c.x1 <= tail[0].x0 and abs(c.oy - tail[0].oy) < 3 * tail[0].size]
        if left and tail[0].x0 - max(c.x1 for c in left) > 1.5 * tail[0].size:
            drop.update(id(c) for c in tail)
            labels.append(txt.replace(" ", ""))
    if not drop:
        return chars, []
    return [c for c in chars if id(c) not in drop], labels


BIGOP_CHARS = set("∑∏∐⋃⋂⨁⨂⨀⋁⋀∫∬∭∮")


def _rel_split(lx):
    """'T(n) = ...' -> ('T(n)', '= ...') at the first relation outside braces and environments; (None, lx) if none."""
    depth, env, i = 0, 0, 0
    while i < len(lx):
        ch = lx[i]
        if lx.startswith("\\begin{", i):
            env += 1
        elif lx.startswith("\\end{", i):
            env -= 1
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0 and env == 0:
            for r in sorted(RELS, key=len, reverse=True):
                if lx.startswith(r, i) and (not r[0].isalpha() and r[0] != "\\" or not lx[i + len(r):i + len(r) + 1].isalpha()):
                    if r[0] == "\\" or r in "=<>" or r == ":=":
                        return lx[:i].strip(), lx[i:].strip()
        i += 1
    return None, lx


def join_aligned(rows):
    """Consecutive display lines that continue one derivation ('x = Ix', '= (A^{-1}A)x', ...) -> one aligned block."""
    body = []
    for lx in rows:
        m = re.fullmatch(r"\\begin\{aligned\} (.*) \\end\{aligned\}", lx.strip(), re.S)
        if m:
            body += [r.strip() for r in m.group(1).split(" \\\\ ")]
            continue
        l, r = _rel_split(lx)
        body.append(f"{l} & {r}" if l is not None else f"& {r}" if r.startswith(("=", "<", ">", "\\le", "\\ge")) else r)
    return "\\begin{aligned} " + " \\\\ ".join(body) + " \\end{aligned}"


def starts_with_relation(lx):
    l, r = _rel_split(lx)
    return l == ""


def split_equation_number(chars, region_x1):
    """Separate a trailing '(3.2)' equation label from a display equation."""
    vis = [c for c in chars if c.c.strip()]
    if len(vis) < 4:
        return chars, None
    txt = "".join(c.c for c in vis)
    m = re.search(r"\(\s*[A-Z]?\d+(?:\.\d+)*[a-z]?\s*\)$|\(\s*[ivx]+\s*\)$", txt)
    if not m:
        return chars, None
    n = len(m.group(0).replace(" ", ""))
    tail = vis[-n:]
    rest = [c for c in chars if c not in tail]
    restv = [c for c in rest if c.c.strip()]
    if restv and tail[0].x0 - max(c.x1 for c in restv) > 1.5 * tail[0].size:
        return rest, "".join(c.c for c in tail)
    return chars, None
