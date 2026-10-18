"""v2.4 fixes for the documented limitations."""
import json

import pymupdf
import pytest

from conftest import LOREM, PDF, paragraph, text


def _row(p, y, cells, xs=(72, 110, 260), font="tiro"):
    for x, c in zip(xs, cells):
        if c:
            text(p, x, y, c, 9, font)


def test_wrapped_cell_lines_join_their_row(tmp, convert):
    rows = [("1", "gridLocation", "Initial world location (X, Y)"),
            ("2", "isMovable", "Can it be moved?"),
            ("3", "requiresKey", "If it requires a key to open or use, then this is the special ID for"),
            (None, None, "the key that opens it."),
            ("4", "isPassable", "Can an agent walk over this object?"),
            ("5", "keyID", "If this object acts as a key, here is its ID (0 by default)"),
            ("6", "density", "Object density in grams per cubic centimetre, used for")
            , (None, None, "buoyancy and weight."),
            ("7", "color", "Colour description of the object.")]

    def pg(p):
        text(p, 110, 80, "Property", 9, "tibo"); text(p, 260, 80, "Description", 9, "tibo")
        y = 95
        for r in rows:
            _row(p, y, r); y += 11
    d = convert(PDF().page(pg).save(tmp / "wrap.pdf"))
    t = [b for b in d["blocks"] if b["type"] == "table"]
    assert len(t) == 1
    body = t[0]["rows"]
    assert ["3", "requiresKey", "If it requires a key to open or use, then this is the special ID for the key that opens it."] in body
    assert ["6", "density", "Object density in grams per cubic centimetre, used for buoyancy and weight."] in body
    assert len(body) == 8


def test_two_column_borderless_table_wrapped_lines(tmp, convert):
    rows = [("gridLocation", "Initial world location (X, Y)"),
            ("isMovable", "Can it be moved?"),
            ("requiresKey", "If it requires a key to open or use, then this is the special ID for"),
            (None, "the key that opens it."),
            ("isPassable", "Can an agent walk over this object?"),
            ("keyID", "If this object acts as a key, here is its ID (0 by default)")]

    def pg(p):
        text(p, 110, 80, "Property", 9, "tibo"); text(p, 260, 80, "Description", 9, "tibo")
        y = 95
        for r in rows:
            if r[0]:
                text(p, 110, y, r[0], 9, "tiro")
            if r[1]:
                text(p, 260, y, r[1], 9, "tiro")
            y += 11
    d = convert(PDF().page(pg).save(tmp / "wrap2.pdf"))
    t = [b for b in d["blocks"] if b["type"] == "table"]
    assert len(t) == 1
    body = t[0]["rows"]
    assert ["requiresKey", "If it requires a key to open or use, then this is the special ID for the key that opens it."] in body
    assert len(body) == 6



def test_table_continued_on_next_page_is_joined(tmp, convert):
    data = [(f"Region {k}", f"{10 + k}.5", f"{20 + k}.1") for k in range(1, 23)]

    def page(p, part, first):
        if first:                                   # prose, then the table runs to the bottom of the page
            paragraph(p, 72, 100, 450, LOREM + " " + LOREM)
            y = 560
        else:                                       # the table continues at the top, header repeated
            y = 80
        _row(p, y, ("Region", "FY23", "FY24"), xs=(72, 260, 400), font="tibo"); y += 16
        for r in part:
            _row(p, y, r, xs=(72, 260, 400)); y += 16
        if not first:
            paragraph(p, 72, y + 40, 450, LOREM)
    pdf = PDF().page(lambda p: page(p, data[:11], True)).page(lambda p: page(p, data[11:], False))
    d = convert(pdf.save(tmp / "span.pdf"))
    t = [b for b in d["blocks"] if b["type"] == "table"]
    assert len(t) == 2 and t[1]["continuation_of"] == t[0]["block_id"]
    m = t[0]["merged"]
    assert m["pages"] == [1, 2] and len(m["rows"]) == 23 and m["rows"][0] == ["Region", "FY23", "FY24"]
    assert m["rows"][-1] == ["Region 22", "32.5", "42.1"]
    md = d["markdown"]
    assert md.count("| Region | FY23 | FY24 |") == 1 and "| Region 22 | 32.5 | 42.1 |" in md


def test_vertically_centred_cell_joins_its_own_row(tmp, convert):
    def row(p, y, cells, size=8):
        for x, c in zip((72, 110, 400), cells):
            if c:
                text(p, x, y, c, size, "tiro")

    def pg(p):
        row(p, 80, ("No.", "Substance", "EC Number"))
        y = 100
        for k, name in ((153, "Sodium peroxometaborate"), (152, "Cadmium chloride"), (151, "Cadmium sulphide"),
                        (149, "Dihexyl phthalate"), (148, "Trixylyl phosphate")):
            if k == 149:      # a two-line name centred on its row: one line above the number line, one below
                row(p, y, (None, "Disodium 4-amino-3-[[4'-(2,4-diaminophenyl)azo]", None))
                row(p, y + 11, ("150", None, "217-710-3"))
                row(p, y + 22, (None, "naphthalene-2,7-disulphonate (C.I. Direct Black 38)", None))
                y += 36
            row(p, y, (str(k), name, f"20{k}-5")); y += 14
    d = convert(PDF().page(pg).save(tmp / "centred.pdf"))
    rows = [b for b in d["blocks"] if b["type"] == "table"][0]["rows"]
    assert ["150", "Disodium 4-amino-3-[[4'-(2,4-diaminophenyl)azo] naphthalene-2,7-disulphonate (C.I. Direct Black 38)", "217-710-3"] in rows
    assert ["151", "Cadmium sulphide", "20151-5"] in rows


# --------------------------------------------------------------------------------------------- maths structures
from processing import mathtex
from processing.textlayer import Char


def _c(c, x, oy, size=10.0, h=None, w=None):
    """A glyph at baseline oy; h makes it a display-size glyph hanging below its origin (as TeX's cmex does)."""
    w = w or 0.55 * size
    if h:
        return Char(c, x, oy, x + w, oy + h, x, oy, size, "CMEX10", 0)
    return Char(c, x, oy - 0.75 * size, x + w, oy + 0.25 * size, x, oy, size, "CMMI10", 2)


def _word(s, x, oy, size=10.0, step=6.0):
    return [_c(ch, x + k * step, oy, size) for k, ch in enumerate(s) if ch != " "]


def test_matrix_from_display_delimiters():
    chars = _word("A=", 0, 125) + [_c("(", 20, 98, h=36)] + _word("a", 30, 115) + _word("b", 60, 115) + \
        _word("c", 30, 133) + _word("d", 60, 133) + [_c(")", 75, 98, h=36)]
    lx = mathtex.build_latex(chars, display=True)[0]
    assert lx == r"A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}", lx


def test_matrix_with_varying_width_elements_and_overlapping_x_spans():
    chars = _word("M=", 0, 125) + [_c("[", 20, 98, h=36)] + \
        _word("x11111", 30, 115) + _word("0", 80, 115) + \
        _word("0", 30, 133) + _word("x22222", 55, 133) + \
        [_c("]", 95, 98, h=36)]
    lx = mathtex.build_latex(chars, display=True)[0]
    assert lx == r"M = \begin{bmatrix} x 11111 & 0 \\ 0 & x 22222 \end{bmatrix}", lx



def test_cases_and_conditions():
    chars = _word("T=", 0, 125) + [_c("{", 20, 100, h=34)] + _word("c", 30, 116) + _word("if", 90, 116) + _word("n=1", 105, 116) + \
        _word("2n", 30, 134) + _word("if", 90, 134) + _word("n>1", 105, 134)
    lx = mathtex.build_latex(chars, display=True)[0]
    assert lx == r"T = \begin{cases} c & \text{if } n = 1 \\ 2 n & \text{if } n > 1 \end{cases}", lx


def test_display_integral_limits_and_big_parens():
    chars = [_c("(", 0, 100, h=30)] + [_c("∫", 10, 100, h=30)] + [_c("n", 18, 104, size=7), _c("1", 18, 128, size=7)] + \
        _word("x", 28, 118) + [_c(")", 40, 100, h=30)]
    lx = mathtex.build_latex(chars, display=True)[0]
    assert lx == r"\left( \int_{1}^{n} x \right)", lx


def test_dot_stacks_digits_and_aligned_derivation():
    dots = [_c(".", 30, 110 + 4 * k) for k in range(3)]
    chars = _word("x", 0, 100) + _word("x", 0, 125) + dots
    atoms = mathtex._dot_stacks(mathtex._atoms_from_chars(chars))
    assert any(a.latex == r"\vdots" for a in atoms)
    assert mathtex._post("( 2 8 . 1 ) a_{1 n} A^{- 1}") == "( 28.1 ) a_{1n} A^{-1}"
    assert mathtex.join_aligned(["x = I x", "= y"]) == r"\begin{aligned} x & = I x \\  & = y \end{aligned}"


def test_footnote_marker_after_full_stop_is_text():
    chars = _word("i=2.", 0, 100) + [_c("2", 26, 95, size=6)]
    runs = mathtex.inline_runs(chars)
    math = ["".join(c.c for c in cs) for m, cs in runs if m]
    text = "".join(c.c for m, cs in runs if not m for c in cs)
    assert math == ["i=2"] and "²" in text


def test_prime_and_delimiter_shapes():
    import numpy as np
    from processing import glyphshape
    m = np.zeros((40, 20), bool)
    for y in range(40):                       # a slanted, tapering stroke
        x = int(16 - 0.35 * y); m[y, max(0, x - (3 if y < 20 else 1)): x + 2] = True
    assert glyphshape._prime_stroke(m)
    ring = np.zeros((40, 30), bool); ring[2:38, 2:28] = True; ring[8:32, 8:22] = False
    assert not glyphshape._prime_stroke(ring)
    paren = np.zeros((120, 20), bool)
    for y in range(120):
        x = int(2 + 14 * ((y - 60) / 60) ** 2); paren[y, x: x + 3] = True
    assert glyphshape.big_delimiter(paren) == "("


def test_glyph_identification_on_unseen_fonts():
    """Held-out fonts (not used for the templates): broken-map glyphs are repaired, correct ones left alone."""
    import os, sys
    held = [p for p in ("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf")
            if os.path.exists(p)]
    if not held:
        pytest.skip("FreeSerif not installed")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import eval_glyphs
    repair, false = eval_glyphs.run(held)
    assert repair >= 0.88 and false <= 0.01


# --------------------------------------------------------------------------------------------- layout
def test_card_grid_reads_row_by_row(tmp, convert):
    def pg(p):
        text(p, 72, 60, "Our services", 18, "hebo")
        k = 0
        for r in range(3):
            for c in range(2):
                x, y = 72 + c * 240, 90 + r * 200
                p.draw_rect(pymupdf.Rect(x, y, x + 220, y + 180), color=(0.6, 0.6, 0.6), fill=(0.95, 0.95, 0.98))
                k += 1
                text(p, x + 12, y + 28, f"Card {k} title", 13, "hebo")
                paragraph(p, x + 12, y + 50, 196, f"Card {k} body text explains this service in a few short sentences.", 10)
    d = convert(PDF().page(pg).save(tmp / "cards.pdf"))
    order = [b["content"] for b in d["blocks"] if (b.get("content") or "").startswith("Card")]
    assert [o.split()[1] for o in order] == ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5", "6", "6"], order
    assert not [b for b in d["blocks"] if b["type"] == "figure"]


def test_busy_vector_page_still_finds_text_tables_and_keeps_caption(tmp, convert):
    import random
    rnd = random.Random(1)

    def pg(p):
        sh = p.new_shape()
        for _ in range(3000):
            x, y = rnd.uniform(40, 560), rnd.uniform(380, 760)
            sh.draw_line((x, y), (x + rnd.uniform(-8, 8), y + rnd.uniform(-8, 8)))
        sh.finish(color=(0.5, 0.6, 0.7)); sh.commit()
        text(p, 72, 80, "Table 2 Station readings", 11, "hebo")
        y = 100
        for r in (("Station", "Temp", "Rain"), ("North", "12.1", "30"), ("East", "14.3", "12"), ("South", "17.8", "8"), ("West", "11.0", "41")):
            for k, c in enumerate(r):
                text(p, 72 + 120 * k, y, c, 10, "tiro")
            y += 15
    d = convert(PDF().page(pg).save(tmp / "busy.pdf"))
    t = [b for b in d["blocks"] if b["type"] == "table"]
    assert t and t[0]["rows"][1] == ["North", "12.1", "30"]
    assert any(b.get("content") == "Table 2 Station readings" for b in d["blocks"])


def test_book_index_page_becomes_one_list_per_column(tmp, convert):
    left = ["binary heap, see heap", "binary search, 44 ex.", "with fast insertion, 472 pr.", "in insertion sort, 45 ex.",
            "binary search tree, 312-330", "AVL trees, 357 pr.", "deletion from, 322-325", "insertion into, 321-322",
            "left-leaning red-black binary search trees,", "358", "maximum key of, 317-318", "minimum key of, 317-318"]
    right = ["binomial distribution, 1198-1201", "and balls and bins, 143", "maximum value of, 1202 ex.", "tails of, 1203-1210",
             "binomial expansion, 1181", "binomial theorem, 1181", "bin packing, 1131 pr.", "bipartite graph, 1167",
             "complete, 716", "matching in, 693-697", "birthday paradox, 140-143", "bisection of a tree, 1147 pr."]

    def pg(p):
        text(p, 72, 60, "Index", 11, "tiro")
        for k, (a, b) in enumerate(zip(left, right)):
            text(p, 72 if not a.startswith(("with", "in ", "AVL", "deletion", "insertion", "left", "358", "maximum k", "minimum")) else 82, 90 + 13 * k, a, 9, "tiro")
            text(p, 320 if not b.startswith(("and", "maximum", "tails", "complete", "matching")) else 330, 90 + 13 * k, b, 9, "tiro")
    d = convert(PDF().page(pg).save(tmp / "index.pdf"))
    lists = [b for b in d["blocks"] if b["type"] == "list"]
    assert len(lists) == 2 and not [b for b in d["blocks"] if b["type"] == "table"]
    assert "left-leaning red-black binary search trees, 358" in lists[0]["content"]
    assert lists[1]["content"].split("\n")[0] == "binomial distribution, 1198-1201"



# --------------------------------------------------------------------------------------------- vector charts
def bar_chart(p, x0=100, y0=150, h=200, w=360):
    base = y0 + h
    p.draw_line((x0, y0), (x0, base), color=(0, 0, 0)); p.draw_line((x0, base), (x0 + w, base), color=(0, 0, 0))
    for k, v in enumerate((0, 10, 20, 30, 40)):
        y = base - v / 40 * h
        text(p, x0 - 22, y + 3, str(v), 8, "helv"); p.draw_line((x0 - 3, y), (x0, y), color=(0, 0, 0))
    data = {"2023": (12, 25, 31, 18), "2024": (15, 28, 36, 22)}
    cols = {"2023": (0.2, 0.4, 0.8), "2024": (0.9, 0.5, 0.1)}
    for g, cat in enumerate(("North", "East", "South", "West")):
        for s, (name, vals) in enumerate(data.items()):
            bx = x0 + 20 + g * 85 + s * 26
            p.draw_rect(pymupdf.Rect(bx, base - vals[g] / 40 * h, bx + 24, base), color=None, fill=cols[name])
        text(p, x0 + 30 + g * 85, base + 14, cat, 8, "helv")
    for s, name in enumerate(data):
        p.draw_rect(pymupdf.Rect(x0 + w - 70, y0 + s * 14, x0 + w - 62, y0 + 8 + s * 14), color=None, fill=cols[name])
        text(p, x0 + w - 58, y0 + 7 + s * 14, name, 8, "helv")
    text(p, x0, y0 - 20, "Figure 1 Sales by region (units)", 10, "helv")

def hbar_chart(p, x0=120, y0=450, h=150, w=300):
    p.draw_line((x0, y0), (x0, y0 + h), color=(0, 0, 0)); p.draw_line((x0, y0 + h), (x0 + w, y0 + h), color=(0, 0, 0))
    for v in (0, 25, 50, 75, 100):
        x = x0 + v / 100 * w
        text(p, x - pymupdf.get_text_length(str(v), "helv", 8) / 2, y0 + h + 12, str(v), 8, "helv")
    for k, (lab, v) in enumerate((("Apples", 64), ("Pears", 38), ("Plums", 81))):
        y = y0 + 15 + k * 45
        p.draw_rect(pymupdf.Rect(x0, y, x0 + v / 100 * w, y + 25), color=None, fill=(0.3, 0.6, 0.3))
        text(p, x0 - 45, y + 16, lab, 8, "helv")

def line_chart(p, x0=100, y0=150, h=200, w=360):
    base = y0 + h
    p.draw_line((x0, y0), (x0, base), color=(0, 0, 0)); p.draw_line((x0, base), (x0 + w, base), color=(0, 0, 0))
    for v in (0, 5, 10, 15, 20):
        y = base - v / 20 * h
        text(p, x0 - 18, y + 3, str(v), 8, "helv")
    months = ("Jan", "Feb", "Mar", "Apr", "May")
    vals = (4, 7.5, 12, 9, 16)
    pts = [pymupdf.Point(x0 + 30 + k * 75, base - v / 20 * h) for k, v in enumerate(vals)]
    p.draw_polyline(pts, color=(0.8, 0.1, 0.1), width=1.5)
    for k, m in enumerate(months):
        text(p, x0 + 22 + k * 75, base + 14, m, 8, "helv")



def test_vector_charts_give_their_data(tmp, convert):
    d = convert(PDF().page(lambda p: (bar_chart(p), hbar_chart(p))).page(line_chart).save(tmp / "charts.pdf"))
    ch = [b for b in d["blocks"] if b["type"] == "chart"]
    assert [b["chart_type"] for b in ch] == ["bar", "horizontal bar", "line"]
    assert ch[0]["rows"] == [["Category", "2023", "2024"], ["North", "12", "15"], ["East", "25", "28"], ["South", "31", "36"], ["West", "18", "22"]]
    assert ch[1]["rows"][1:] == [["Apples", "64"], ["Pears", "38"], ["Plums", "81"]]
    assert [r[1] for r in ch[2]["rows"][1:]] == ["4", "7.5", "12", "9", "16"]
    assert all(b["verification"] != "verified" for b in ch)
    # the axis and category labels belong to the chart, not to the text around it
    assert not [b for b in d["blocks"] if b["type"] == "paragraph" and b["content"] in ("North", "40", "Jan", "Apples")]
    assert "| North | 12 | 15 |" in d["markdown"]


# --------------------------------------------------------------------------------------------- optional local models
def _script(tmp, name, body):
    p = tmp / name
    p.write_text("import sys, json\n" + body)
    return f"python3 {p}"


@pytest.mark.ocr
def test_ocr_engine_command_plugin_replaces_tesseract(tmp, convert, monkeypatch):
    from processing import ocr
    ocr.clear_cache()
    cmd = _script(tmp, "fake_ocr.py", "print(json.dumps([{'text': w, 'conf': 0.97, 'box': [100 + 90 * k, 200, 180 + 90 * k, 230]}"
                                      " for k, w in enumerate('plugin engine words here'.split())]))\n")
    monkeypatch.setenv("PARSEFLOW_OCR_ENGINE", "command"); monkeypatch.setenv("PARSEFLOW_OCR_CMD", cmd)
    img = __import__("PIL.Image", fromlist=["Image"]).new("L", (1200, 900), 255)
    from PIL import ImageDraw
    ImageDraw.Draw(img).rectangle((100, 200, 460, 230), fill=0)
    f = tmp / "scan.png"; img.save(f)
    d = convert(f)
    assert "plugin engine words here" in d["markdown"]
    ocr.clear_cache()


def test_handwriting_and_chart_plugins(tmp, convert, monkeypatch):
    from processing import engines
    from PIL import Image, ImageDraw
    monkeypatch.setenv("PARSEFLOW_CHART_MODEL", "command")
    monkeypatch.setenv("PARSEFLOW_CHART_CMD", _script(tmp, "fake_chart.py", "print('Year,Sales\\n2022,10\\n2023,14')\n"))
    monkeypatch.setenv("PARSEFLOW_HANDWRITING", "command")
    monkeypatch.setenv("PARSEFLOW_HANDWRITING_CMD", _script(tmp, "fake_hw.py", "print(json.dumps({'text': 'call Ana at 5', 'conf': 0.8}))\n"))
    assert engines.chart_ready() and engines.handwriting_ready()
    pic = Image.new("RGB", (400, 300), "white"); dr = ImageDraw.Draw(pic)
    for k, h in enumerate((80, 120, 160)):
        dr.rectangle((60 + 90 * k, 260 - h, 120 + 90 * k, 260), fill=(40, 90, 200))
    pf = tmp / "chart.png"; pic.save(pf)
    d = convert(PDF().page(lambda p: (text(p, 72, 80, "Figure 3 Sales", 11, "hebo"), p.insert_image(pymupdf.Rect(72, 100, 472, 400), filename=str(pf)))).save(tmp / "imgchart.pdf"))
    ch = [b for b in d["blocks"] if b["type"] == "chart"]
    assert ch and ch[0]["rows"] == [["Year", "Sales"], ["2022", "10"], ["2023", "14"]] and ch[0]["verification"] != "verified"
    r = engines.read_handwriting(pic)
    assert r == ("call Ana at 5", 0.8)
    assert engines._table_text("TITLE | Sales <0x0A> Year | Value <0x0A> 2022 | 10") == [["Year", "Value"], ["2022", "10"]]


@pytest.mark.ocr
def test_handwriting_plugin_reads_unexplained_ink(tmp, convert, monkeypatch):
    import math
    from PIL import Image, ImageDraw
    img = Image.new("L", (1600, 1200), 255); d = ImageDraw.Draw(img)
    d.line([(300 + k * 4, 500 + 40 * math.sin(k / 6.0)) for k in range(200)], fill=0, width=6)
    d.line([(300 + k * 4, 600 + 30 * math.cos(k / 4.0)) for k in range(180)], fill=0, width=6)
    f = tmp / "scribble.png"; img.save(f)
    monkeypatch.setenv("PARSEFLOW_HANDWRITING", "command")
    monkeypatch.setenv("PARSEFLOW_HANDWRITING_CMD", _script(tmp, "hw.py", "print(json.dumps({'text': 'call Ana at 5', 'conf': 0.8}))\n"))
    blocks = convert(f)["blocks"]
    hw = [b for b in blocks if b["type"] == "handwritten_text"]
    assert hw and hw[0]["content"] == "call Ana at 5" and hw[0]["extractor"] == "handwriting_model" and hw[0]["verification"] != "verified"


def test_a_failing_page_keeps_its_text(tmp, convert, monkeypatch):
    """Found on the full Cormen run: one unexpected glyph must not cost a whole page."""
    from processing import pipeline
    f = PDF().page(lambda p: (text(p, 72, 80, "Fibonacci numbers", 18, "hebo"), paragraph(p, 72, 120, 450, LOREM))).save(tmp / "fail.pdf")

    def boom(ctx, state):
        raise TypeError("simulated")
    monkeypatch.setattr(pipeline, "_blocks_for_page", boom)
    d = convert(f)
    assert [b["type"] for b in d["blocks"]] == ["paragraph", "paragraph"] and "Fibonacci numbers" in d["markdown"]
    assert all(b["verification"] != "verified" and "failed" in b["warnings"][0] for b in d["blocks"])


def test_multi_letter_glyph_inside_maths_does_not_break_latex():
    from processing import mathtex
    from processing.textlayer import Char
    cs = [Char(c, 10 * k, 0, 10 * k + 8, 10, 10 * k, 8, 10, "CMMI10", 2) for k, c in enumerate(["x", "=", "fi", "+", "1"])]
    lx, conf, notes = mathtex.build_latex(cs)
    assert "fi" in lx and conf > 0
