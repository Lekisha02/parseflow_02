"""Verifier independence/provenance rules and the canonical Markdown serializer."""
from processing import markdown as mdgen
from processing import verifier
from processing.schema import make_block

INFO = dict(width=1224, height=1584, norm_text="regionfy23fy24americas180191", text_quality=0.98, hidden_text_norm="", hidden_text_raw="", ocr_ok=False)
crop = lambda bb: None


def _b(**kw):
    b = make_block("x_1_01", kw.pop("type", "paragraph"), 1, kw.pop("bbox", (10, 10, 200, 40)), kw.pop("conf", 0.99),
                   kw.pop("ctype", "native text layer"), kw.pop("ext", "digital_text_parser"), kw.pop("content", "Hello world"), **kw)
    return b


def test_native_text_verified_with_explicit_basis():
    b = verifier.verify(_b(), INFO, crop)
    assert b["verification"] == "verified"
    d = b["evidence_detail"]
    assert d["independent"] is False and "document-encoded" in d["basis"]


def test_native_text_from_broken_layer_is_review():
    b = verifier.verify(_b(), dict(INFO, text_quality=0.3), crop)
    assert b["verification"] == "review" and b["warnings"]


def test_ocr_alone_never_verified():
    b = verifier.verify(_b(ext="ocr", ctype="ocr", conf=0.97), INFO, crop)
    assert b["verification"] == "review"
    assert b["evidence_detail"]["independent"] is False
    assert any("independent" in w for w in b["warnings"])


def test_ocr_confirmed_by_embedded_layer_is_verified_and_independent():
    info = dict(INFO, hidden_text_raw="Hello world from the producer OCR layer", hidden_text_norm="helloworld")
    b = verifier.verify(_b(ext="ocr", ctype="ocr", conf=0.9), info, crop)
    assert b["verification"] == "verified" and b["evidence_detail"]["independent"] is True


def test_model_output_without_evidence_is_review_with_warning():
    b = verifier.verify(_b(ext="vision_diagram", ctype="model (self-reported)", conf=0.95, type="diagram", content="A tree"), INFO, crop)
    assert b["verification"] == "review"
    assert any("self-reported" in w for w in b["warnings"])


def test_model_table_confirmed_by_text_layer_is_independent():
    b = _b(ext="vision_table", ctype="model (self-reported)", conf=0.9, type="table", content="Table with 2 rows",
           rows=[["Region", "FY23", "FY24"], ["Americas", "18.0", "19.1"]])
    b = verifier.verify(b, INFO, crop)
    assert b["verification"] == "verified" and b["evidence_detail"]["independent"]


def test_aligned_table_is_review_ruled_table_verified():
    cells = [dict(row=0, col=0, text="a"), dict(row=0, col=1, text="")]
    al = verifier.verify(_b(ext="table_engine", ctype="heuristic (estimated)", conf=0.8, type="table", cells=cells, table_method="aligned_text", alignment=0.8), INFO, crop)
    ru = verifier.verify(_b(ext="table_engine", ctype="heuristic (estimated)", conf=0.9, type="table", cells=cells, table_method="ruled_grid"), INFO, crop)
    assert al["verification"] == "review" and ru["verification"] == "verified"
    assert any("empty cell" in s for s in ru["evidence_detail"]["signals"])


def test_fallback_failed_and_bbox_out_of_page_failed():
    assert verifier.verify(_b(ext="fallback", ctype="heuristic (estimated)", conf=0.3, content=None, type="figure"), INFO, crop)["verification"] == "failed"
    assert verifier.verify(_b(bbox=(10, 10, 5000, 40)), INFO, crop)["verification"] == "failed"


def test_reconstructed_equation_is_review():
    b = _b(type="equation", conf=0.8, ctype="heuristic (estimated)", latex=r"\frac{a}{b}", reconstructed=True)
    assert verifier.verify(b, INFO, crop)["verification"] == "review"


# ---------------------------------------------------------------------------------------------- markdown
def _doc(blocks):
    for i, b in enumerate(blocks, 1):
        b.setdefault("reading_order", i)
    return dict(filename="t.pdf", blocks=blocks, warnings=[])


def test_markdown_structures():
    blocks = [
        _b(type="heading", content="Results", level=2),
        _b(type="paragraph", content="Energy is \\(E = mc^{2}\\) here."),
        _b(type="list", content="• one\n• two"),
        _b(type="code", content="def f():\n    return 1"),
        _b(type="equation", latex=r"\int_{0}^{1} x \, dx = \frac{1}{2}", content="raw", math_display=True, equation_number="(3)"),
        _b(type="table", content="Table", cells=[dict(row=0, col=0, text="a|b", header=True), dict(row=0, col=1, text="c", header=True),
                                                  dict(row=1, col=0, text="1"), dict(row=1, col=1, text="")]),
    ]
    md = mdgen.render(_doc(blocks), annotate=False)
    assert "## Results" in md
    assert "\\(E = mc^{2}\\)" in md                         # backslashes and braces intact
    assert "- one\n- two" in md
    assert "```\ndef f():\n    return 1\n```" in md
    assert "\\[\\int_{0}^{1} x \\, dx = \\frac{1}{2}\\] (3)" in md
    assert "| a\\|b | c |" in md and "| 1 |  |" in md
    assert "<!--" not in md


def test_markdown_excludes_furniture_unless_requested_and_dedupes():
    hdr = _b(content="Running Header Text")
    hdr["furniture"] = dict(role="header", score=1.4, reasons=["x"], excluded=True)
    p1 = _b(content="Duplicate paragraph text that is long enough", bbox=(10, 10, 200, 40))
    p2 = _b(content="Duplicate paragraph text that is long enough", bbox=(12, 11, 201, 41), ext="ocr")
    d = _doc([hdr, p1, p2])
    md = mdgen.render(d, annotate=False)
    assert "Running Header Text" not in md and md.count("Duplicate paragraph") == 1
    assert "Running Header Text" in mdgen.render(d, include_furniture=True, annotate=False)


def test_markdown_unicode_nfc_and_empty_doc():
    md = mdgen.render(_doc([_b(content="café 日本")]), annotate=False)
    assert "café" in md and "日本" in md
    assert mdgen.render(dict(filename="e.pdf", blocks=[], warnings=[]), annotate=False).strip() == ""


def test_annotations_flag_uncertain_regions():
    fig = _b(type="figure", content=None, ext="fallback", conf=0.3)
    md = mdgen.render(_doc([fig]), annotate=True)
    assert "flagged for review" in md
    assert mdgen.render(_doc([fig]), annotate=False).strip() == ""


def test_markdown_deterministic():
    d = _doc([_b(content="a"), _b(type="heading", content="b", level=1)])
    assert mdgen.render(d) == mdgen.render(d)
