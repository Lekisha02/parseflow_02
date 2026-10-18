"""Native text extraction, reading order, paragraphs, hyphenation, headings, Unicode."""
import pymupdf

from conftest import LOREM, PDF, paragraph, text, unicode_text
from processing import layout, textlayer


def _segs(path):
    p = pymupdf.open(str(path))[0]
    return textlayer.read_segments(p)[0]


def test_native_text_extraction(tmp, convert):
    f = PDF().page(lambda p: (text(p, 72, 100, "Quarterly Financial Review", 20, "hebo"), paragraph(p, 72, 140, 450, LOREM))).save(tmp / "a.pdf")
    d = convert(f)
    assert d["pages"][0]["route"] == "digital"
    md = d["markdown"]
    assert "# Quarterly Financial Review" in md
    assert "ParseFlow keeps the evidence behind every extracted block" in md
    para = [b for b in d["blocks"] if b["type"] == "paragraph"][0]
    assert para["extractor"] == "digital_text_parser" and para["verification"] == "verified"
    assert para["raw_text"] and para["bbox"][2] > para["bbox"][0]


def test_two_column_reading_order_with_spanning_header(tmp, convert):
    left = "Alpha column sentence one continues here. " * 6 + "LEFTEND."
    right = "Beta column sentence two continues here. " * 6 + "RIGHTEND."

    def draw(p):
        text(p, 150, 30, "Downloaded from http://example.org/ on July 12, 2017 - Published by example.org", 8)
        text(p, 72, 80, "A Title Spanning Both Columns For The Article", 16, "hebo")
        paragraph(p, 72, 120, 220, left)
        paragraph(p, 320, 120, 220, right)
    d = convert(PDF().page(draw).save(tmp / "cols.pdf"))
    md = d["markdown"]
    # each column is read completely before the other; lines of the two columns are never interleaved
    assert md.index("LEFTEND") < md.index("Beta column sentence two")
    assert "Alpha column sentence one continues here. Alpha column" in md.replace("\n", " ")
    assert d["pages"][0]["flags"].get("multi_column")


def test_sidebar_beside_text_column(tmp, convert):
    def draw(p):
        y = 100
        for s in ["Correspondence to:", "Dr A Person,", "Some Street 6,", "Helsinki;", "a@b.org"]:
            text(p, 40, y, s, 7); y += 9
        paragraph(p, 150, 100, 300, "Main column text that should stay together in one paragraph when read. " * 4)
    md = convert(PDF().page(draw).save(tmp / "side.pdf"))["markdown"]
    flat = " ".join(md.split())
    assert "Main column text that should stay together in one paragraph when read. Main column" in flat


def test_dehyphenation_rules():
    vocab = layout.doc_vocab(["the well-known result", "information"])
    t, tr = layout.join_lines(["We studied the infor-", "mation flow of a well-", "known result and self-", "contained parts."], vocab)
    assert "information flow" in t
    assert "well-known" in t           # hyphenated form used elsewhere in the document
    assert "self-contained" in t       # compound-forming prefix keeps its hyphen
    assert any("rejoined" in x for x in tr)


def test_quotes_normalised():
    assert layout.normalize_quotes("‘‘ethics’’") == "“ethics”"


def test_unicode_and_accents(tmp, convert):
    def draw(p):
        unicode_text(p, 72, 100, "Ján Mináč — naïve façade, Ελληνικά, 日本語のテキスト")
        # an accent drawn as a separate glyph over the letter (as TeX does)
        text(p, 72, 140, "Jan", 12, "tiro")
        a_x = 72 + pymupdf.Font("tiro").text_length("J", fontsize=12)
        text(p, a_x + 1.2, 140, "´", 12, "tiro")
    d = convert(PDF().page(draw).save(tmp / "u.pdf"))
    md = d["markdown"]
    assert "Ελληνικά" in md and "日本語" in md and "naïve" in md
    assert "Ján" in md   # spacing accent composed with its letter


def test_adaptive_spacing_without_space_glyphs(tmp):
    def draw(p):
        x = 72
        f = pymupdf.Font("helv")
        for w in "Composition of the pretraining data".split():
            text(p, x, 100, w, 10)
            x += f.text_length(w, fontsize=10) + 1.4    # word gap ~0.14 em, no space glyph
    segs = _segs(PDF().page(draw).save(tmp / "sp.pdf"))
    joined = " ".join(s.text for s in segs)
    assert "Composition of the pretraining data" in joined


def test_text_quality_flags_garbage():
    good, _ = textlayer.text_quality(LOREM)
    bad, _ = textlayer.text_quality("��(cid:12)(cid:44) xqzv bcdf  rrrrr")
    assert good > 0.8 and bad < 0.5


def test_heading_levels_and_lists(tmp, convert):
    def draw(p):
        text(p, 72, 80, "Main Title", 22, "hebo")
        text(p, 72, 120, "Section Heading", 15, "hebo")
        paragraph(p, 72, 150, 450, LOREM)
        text(p, 80, 220, "• first bullet item", 10)
        text(p, 80, 233, "• second bullet item", 10)
    d = convert(PDF().page(draw).save(tmp / "h.pdf"))
    md = d["markdown"]
    assert "# Main Title" in md and "## Section Heading" in md
    assert "- first bullet item" in md and "- second bullet item" in md


def test_deterministic_output(tmp, convert):
    f = PDF().page(lambda p: paragraph(p, 72, 100, 450, LOREM * 3)).save(tmp / "det.pdf")
    a, b = convert(f), convert(f)
    strip = lambda d: [{k: v for k, v in x.items()} for x in d["blocks"]]
    assert a["markdown"] == b["markdown"] and strip(a) == strip(b)
