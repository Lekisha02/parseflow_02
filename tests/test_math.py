"""Mathematical region detection, LaTeX reconstruction and serialisation, recognizer fallback."""
import json
import os
import stat

from conftest import PDF, paragraph, text, unicode_text
from processing import markdown as mdgen
from processing import mathtex, recognizers
from processing.textlayer import Char


def g(c, x, base, size=10, font="CMMI10", w=None):
    w = w or 0.55 * size
    return Char(c, x, base - 0.75 * size, x + w, base + 0.25 * size, x, base, size, font, 0)


def test_superscript_and_subscript():
    chars = [g("a", 0, 100), g("2", 5.6, 96, 7, "CMR7"), g("+", 10, 100, font="CMR10"), g("b", 18, 100), g("i", 23.6, 102.5, 7)]
    lx, conf, notes = mathtex.build_latex(chars)
    assert lx == "a^{2} + b_{i}"
    assert conf >= 0.6


def test_fraction_from_rule():
    num = [g("∂", 2, 95), g("B", 8, 95)]
    den = [g("∂", 2, 109), g("t", 8, 109)]
    lead = [g("E", -20, 102), g("=", -12, 102, font="CMR10")]
    lx, conf, notes = mathtex.build_latex(lead + num + den, hrules=[(1, 99.5, 15, 99.5)])
    assert lx == r"E = \frac{\partial B}{\partial t}"
    assert any(n.startswith("fraction") for n in notes)


def test_sum_with_limits_and_greek():
    chars = [g("λ", 0, 100), g("g", 5.5, 102, 7), g("=", 10, 100, font="CMR10"), g("∑", 18, 102, 15, "CMEX10", 10),
             g("i", 18.5, 112, 7), g("=", 21, 112, 7, "CMR7"), g("1", 25.5, 112, 7, "CMR7"), g("k", 21, 89, 7), g("x", 30, 100), g("i", 35.5, 102, 7)]
    lx, conf, _ = mathtex.build_latex(chars)
    assert lx == r"\lambda_{g} = \sum_{i = 1}^{k} x_{i}"


def test_radical_with_overbar():
    chars = [g("√", 0, 101, 10, "CMSY10"), g("x", 7, 100), g("+", 12, 100, font="CMR10"), g("1", 18, 100, font="CMR10")]
    lx, _, _ = mathtex.build_latex(chars, hrules=[(6, 92.5, 24, 92.5)])
    assert lx == r"\sqrt{x + 1}"


def test_unmapped_glyph_lowers_confidence_not_invented():
    chars = [g("x", 0, 100), g("�", 6, 100, font="CMEX10"), g("y", 12, 100)]
    lx, conf, notes = mathtex.build_latex(chars)
    assert "�" not in lx and lx == "x y"
    assert conf < 0.9 and any("without a Unicode mapping" in n for n in notes)


def test_inline_runs_do_not_wrap_prose_or_footnote_markers():
    prose = [g(c, i * 5, 100, 10, "Times-Roman") for i, c in enumerate("products")] + [g("1", 41, 96, 7, "Times-Roman")]
    assert not any(m for m, _ in mathtex.inline_runs(prose))
    italic_word = [Char(c, i * 5, 92, i * 5 + 5, 102, i * 5, 100, 10, "Times-Italic", 2) for i, c in enumerate("Proof")]
    assert not any(m for m, _ in mathtex.inline_runs(italic_word))
    greek_word = [g(c, i * 6, 100, 10, "Arial") for i, c in enumerate("Ελληνικά")]
    assert not any(m for m, _ in mathtex.inline_runs(greek_word))


def test_equation_number_split():
    chars = [g("x", 0, 100), g("=", 8, 100, font="CMR10"), g("1", 16, 100, font="CMR10")] + \
            [g(c, 200 + i * 5, 100, font="CMR10") for i, c in enumerate("(3.2)")]
    rest, num = mathtex.split_equation_number(chars, 230)
    assert num == "(3.2)" and len(rest) == 3


def test_display_equation_end_to_end(tmp, convert):
    def draw(p):
        paragraph(p, 72, 100, 450, "The identity below connects five constants and is shown as a display equation.")
        unicode_text(p, 250, 160, "e", 14)
        unicode_text(p, 258, 153, "iπ", 10)
        unicode_text(p, 272, 160, "+ 1 = 0", 14)
        paragraph(p, 72, 220, 450, "After the equation the paragraph continues with ordinary prose text.")
    d = convert(PDF().page(draw).save(tmp / "eq.pdf"))
    eq = [b for b in d["blocks"] if b["type"] == "equation"]
    assert eq and eq[0]["latex"] == r"e^{i \pi} + 1 = 0"
    assert r"\[e^{i \pi} + 1 = 0\]" in d["markdown"]
    assert eq[0]["raw_text"]                        # original glyph text kept as evidence
    assert eq[0]["verification"] == "review"        # structure inferred -> not 'verified'


def test_latex_survives_json_and_markdown_roundtrip(tmp, convert):
    def draw(p):
        unicode_text(p, 200, 160, "∑ α ≤ β", 14)
    d = convert(PDF().page(draw).save(tmp / "rt.pdf"))
    d2 = json.loads(json.dumps(d))
    md = mdgen.render(d2, annotate=False)
    assert r"\sum \alpha \leq \beta" in md and "\\\\" not in md


def test_recognizer_is_optional_and_never_downloads(monkeypatch, tmp):
    monkeypatch.setenv("PARSEFLOW_MATH_RECOGNIZER", "pix2tex")
    monkeypatch.setenv("PARSEFLOW_MATH_MODEL_DIR", str(tmp / "missing"))
    recognizers._STATE["name"] = None
    info = recognizers.info()
    assert info["ready"] is False and "weights are never downloaded" in (info["error"] or "") or "No module" in (info["error"] or "")
    from PIL import Image
    assert recognizers.recognize(Image.new("RGB", (40, 20), "white")) is None


def test_command_recognizer_plugin(monkeypatch, tmp):
    script = tmp / "fake_recognizer.sh"
    script.write_text("#!/bin/sh\necho 'x^{2}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PARSEFLOW_MATH_RECOGNIZER", "command")
    monkeypatch.setenv("PARSEFLOW_MATH_CMD", str(script))
    recognizers._STATE["name"] = None
    from PIL import Image
    assert recognizers.recognize(Image.new("RGB", (40, 20), "white")) == ("x^{2}", 0.6)
    monkeypatch.setenv("PARSEFLOW_MATH_RECOGNIZER", "none")
    recognizers._STATE["name"] = None


def _scanned_equation_pdf(tmp):
    import io
    import pymupdf
    src = pymupdf.open()
    p = src.new_page(width=612, height=300)
    p.insert_text((72, 80), "The relation used in this section is", fontsize=12, fontname="tiro")
    p.insert_text((200, 140), "y = 2x + 3z", fontsize=16, fontname="tiro")
    pix = p.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    doc = pymupdf.open()
    q = doc.new_page(width=612, height=300)
    q.insert_image(q.rect, stream=pix.tobytes("png"))
    f = tmp / "scan_eq.pdf"
    doc.save(str(f))
    return f


import pytest  # noqa: E402


@pytest.mark.ocr
def test_scanned_equation_uses_recognizer_only_on_region(tmp, convert, monkeypatch):
    script = tmp / "rec.sh"
    script.write_text("#!/bin/sh\necho 'y = 2x + 3z'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PARSEFLOW_MATH_RECOGNIZER", "command")
    monkeypatch.setenv("PARSEFLOW_MATH_CMD", str(script))
    recognizers._STATE["name"] = None
    d = convert(_scanned_equation_pdf(tmp))
    eq = [b for b in d["blocks"] if b["type"] == "equation"]
    assert eq and eq[0]["extractor"] == "math_model" and eq[0]["latex"] == "y = 2x + 3z"
    assert eq[0]["verification"] == "review" and eq[0]["raw_text"]
    assert "\\[y = 2x + 3z\\]" in d["markdown"]
    monkeypatch.setenv("PARSEFLOW_MATH_RECOGNIZER", "none")
    recognizers._STATE["name"] = None
    d2 = convert(_scanned_equation_pdf(tmp))
    flagged = [b for b in d2["blocks"] if b.get("possible_equation")]
    assert flagged and any("not converted to LaTeX" in w for w in flagged[0]["warnings"])
