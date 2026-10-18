"""v2.3 accuracy fixes: broken character maps repaired from the glyph shapes / names, links and quoted words are not
equations, bracketed index expressions stay whole, tables of contents become lists, numbered pseudo-code becomes a
code block (not a table), diagrams are not tables, real tables still are, and isolated processing."""
import json
import os
from pathlib import Path

import pymupdf
import pytest

from conftest import LOREM, PDF, paragraph, text
from processing import mathtex, runner
from processing.textlayer import Char

DEJAVU = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
CODE = ["Insertion sort pseudo-code:", "1 for i = 2 to n", "2     key = A[i]", "3     j = i − 1",
        "4     while j > 0 and A[j] > key", "5         A[j + 1] = A[j]", "6         j = j − 1", "7     A[j + 1] = key",
        "The subarray A[1 : i − 1] is sorted."]


def _corrupt_cmap_pdf(path, lines, bad):
    """A PDF whose ToUnicode map lies the way TeX math fonts often do ('=' -> 'D', '[' -> 'Œ', '−' -> control code)."""
    fontTools = pytest.importorskip("fontTools.ttLib")
    doc = pymupdf.open(); pg = doc.new_page(width=612, height=792)
    pg.insert_font(fontname="dv", fontfile=str(DEJAVU))
    y = 90
    for t in lines:
        pg.insert_text((72, y), t, fontname="dv", fontsize=11); y += 16
    doc.save(str(path)); doc.close()
    tt = fontTools.TTFont(str(DEJAVU)); cmap = tt.getBestCmap(); order = tt.getGlyphOrder()
    ent = ["<%04x> <%04x>" % (order.index(cmap[ord(ch)]), ord(bad.get(ch, ch))) for ch in sorted(set("".join(lines))) if ord(ch) in cmap]
    cm = ("/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /X def /CMapType 2 def 1 begincodespacerange "
          "<0000> <FFFF> endcodespacerange\n%d beginbfchar\n%s\nendbfchar\nendcmap CMapName currentdict /CMap defineresource pop end end"
          % (len(ent), "\n".join(ent)))
    doc = pymupdf.open(str(path))
    for x in range(1, doc.xref_length()):
        try:
            st = doc.xref_stream(x)
        except Exception:
            continue
        if st and b"beginbfchar" in st:
            doc.update_stream(x, cm.encode("latin-1"))
    out = Path(str(path) + ".c.pdf"); doc.save(str(out)); doc.close()
    return out


@pytest.mark.skipif(not DEJAVU.exists(), reason="DejaVu Serif not installed")
def test_broken_character_map_is_repaired(tmp, convert):
    f = _corrupt_cmap_pdf(tmp / "b.pdf", CODE, {"=": "D", "+": "C", "[": "Œ", "]": "\u008d", "−": "\u0002", ":": "W"})
    assert "AŒi" in pymupdf.open(str(f))[0].get_text()                  # the text layer really is broken
    md = convert(f)["markdown"]
    assert "Œ" not in md and "\u008d" not in md and "\u0002" not in md
    assert "pseudo-code:" in md and "1 for i = 2 to n" in md and "A[j + 1] = key" in md
    # code indentation follows the drawn positions
    assert "\n2     key = A[i]" in md and "\n5         A[j + 1] = A[j]" in md
    # the bracketed index expression stays one piece of mathematics
    assert r"\(A [ 1 : i - 1 ]\)" in md


def test_links_are_text_not_equations(tmp, convert):
    def pg(p):
        paragraph(p, 72, 200, 450, LOREM)
        text(p, 72, 300, "See https://example.com/search?q=a+b&page=2 for the data.", 11, "tiro")
        text(p, 72, 330, "https://doi.org/10.1000/xyz=123", 11, "tiro")
        text(p, 72, 360, "Write to someone@example.org or visit www.parseflow.ai/docs?x=1.", 11, "tiro")
    d = convert(PDF().page(pg).save(tmp / "links.pdf"))
    assert not [b for b in d["blocks"] if b["type"] == "equation"]
    md = d["markdown"]
    assert "https://example.com/search?q=a+b&page=2" in md and "\\(" not in md


def _chars(word, font="TimesNewRoman", size=10.0, flags=0):
    out, x = [], 0.0
    for c in word:
        out.append(Char(c, x, 0, x + 5, 10, x, 8, size, font, flags)); x += 5
    return out


def test_quoted_word_and_notation_words_are_text():
    for w in ("<problems.=", "Ω-notation", "“problems.”"):
        assert mathtex.token_class(_chars(w), 10.0) == "text", w
    assert mathtex.token_class(_chars("x=y"), 10.0) == "strong"
    runs = mathtex.inline_runs(_chars("★") + _chars(" ") + _chars("5.4"))
    assert not any(m for m, _ in runs)                                    # a section star is not mathematics


def test_table_of_contents_becomes_a_list(tmp, convert):
    entries = [("1", "The Role of Algorithms in Computing", 5), ("1.1", "Algorithms", 5), ("1.2", "Algorithms as a technology", 12),
               ("2", "Getting Started", 17), ("2.1", "Insertion sort", 17), ("2.2", "Analyzing algorithms", 25),
               ("2.3", "Designing algorithms", 34), ("3", "Characterizing Running Times", 49), ("3.1", "Asymptotic notation", 50)]

    def pg(p):
        text(p, 72, 80, "Contents", 20, "hebo")
        y = 130
        for num, title, page in entries:
            text(p, 72, y, num, 11, "tiro"); text(p, 110, y, title, 11, "tiro"); text(p, 500, y, str(page), 11, "tiro"); y += 18
    d = convert(PDF().page(pg).save(tmp / "toc.pdf"))
    assert not [b for b in d["blocks"] if b["type"] == "table"]
    lst = [b for b in d["blocks"] if b["type"] == "list"]
    assert lst and "2.1 Insertion sort 17" in lst[0]["content"] and "1 The Role of Algorithms in Computing 5" in lst[0]["content"]


def test_numbered_pseudocode_with_cost_column_is_code_not_table(tmp, convert):
    rows = ["for i = 2 to n", "    key = A[i]", "    j = i - 1", "    while j > 0 and A[j] > key", "        A[j + 1] = A[j]",
            "        j = j - 1", "    A[j + 1] = key"]

    def pg(p):
        text(p, 72, 80, "INSERTION-SORT(A, n)", 11, "hebo")
        y = 110
        for k, r in enumerate(rows, 1):
            text(p, 72, y, str(k), 10, "cour"); text(p, 90 + 6 * (len(r) - len(r.lstrip())), y, r.strip(), 10, "cour")
            text(p, 400, y, f"c{k}", 10, "tiro"); y += 15
        paragraph(p, 72, y + 30, 450, LOREM)
    d = convert(PDF().page(pg).save(tmp / "code.pdf"))
    assert not [b for b in d["blocks"] if b["type"] == "table"]
    code = [b for b in d["blocks"] if b["type"] == "code"]
    assert code and "1 for i = 2 to n" in code[0]["content"] and "key = A[i]" in code[0]["content"]


def test_box_diagram_is_a_figure_and_real_table_is_kept(tmp, convert):
    def diagram(p):
        text(p, 72, 70, "The array before and after one step:", 11, "tiro")
        for row, vals in enumerate(("524613", "254613")):
            y = 120 + row * 90
            for k, v in enumerate(vals):
                r = pymupdf.Rect(100 + k * 40, y, 136 + k * 40, y + 30)
                p.draw_rect(r, color=(0, 0, 0), fill=(0.9, 0.85, 0.6) if k else (0.6, 0.75, 0.95))
                text(p, r.x0 + 14, r.y0 + 20, v, 12, "tiro")
                text(p, r.x0 + 14, r.y0 - 4, str(k + 1), 8, "tiro")
            for k in range(3):
                p.draw_bezier((118 + k * 40, y + 32), (125 + k * 40, y + 60), (150 + k * 40, y + 60), (158 + k * 40, y + 32), color=(0.9, 0.4, 0))
        text(p, 72, 330, "Figure 2.2 The operation of insertion sort on a small array.", 10, "tiro")

    def table(p):
        text(p, 72, 80, "Table 1 Revenue by region (USD bn)", 11, "hebo")
        y = 110
        for r in (("Region", "FY23", "FY24"), ("Americas", "18.0", "19.1"), ("Europe", "11.2", "11.2"), ("APAC", "14.4", "17.9"), ("Total", "43.6", "48.2")):
            for k, c in enumerate(r):
                text(p, 72 + 140 * k, y, c, 10, "tiro")
            y += 16
    d = convert(PDF().page(diagram).page(table).save(tmp / "mix.pdf"))
    t = [b for b in d["blocks"] if b["type"] == "table"]
    assert len(t) == 1 and t[0]["page"] == 2 and "Americas" in json.dumps(t[0])
    assert [b for b in d["blocks"] if b["page"] == 1 and b["type"] in ("figure", "diagram", "chart")]


def test_isolated_processing_complete_failed_and_timeout(tmp):
    from processing import pipeline
    f = PDF().page(lambda p: (text(p, 72, 90, "Isolated", 18, "hebo"), paragraph(p, 72, 130, 450, LOREM))).save(tmp / "iso.pdf")
    out = tmp / "job"; out.mkdir()
    r = runner.run_isolated(str(f), "iso.pdf", "abcdefabcdef", str(out), pipeline.Progress(), poll=0.05)
    assert r["status"] == "complete" and (out / "document.json").exists()
    bad = tmp / "bad.pdf"; bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
    out2 = tmp / "job2"; out2.mkdir()
    r = runner.run_isolated(str(bad), "bad.pdf", "abcdefabcdee", str(out2), pipeline.Progress(), poll=0.05)
    assert r["status"] == "failed" and r.get("error")
    out3 = tmp / "job3"; out3.mkdir()
    r = runner.run_isolated(str(f), "iso.pdf", "abcdefabcded", str(out3), pipeline.Progress(), timeout=0.01, poll=0.02)
    assert r["status"] == "failed" and r["error"]["code"] == "PF-504"
