"""Every input format: Word, PowerPoint, Excel/CSV, images, text/Markdown/HTML, e-books. Each document is generated
here with known content and read back through the normal pipeline; LibreOffice paths run when it is installed,
ParseFlow's own renderers are tested with LibreOffice switched off."""
import importlib
import json
import sys
import time

import pymupdf
import pytest

import docmaker as D
from conftest import HAS_TESSERACT
from processing import formats, runner

LO = bool(formats.soffice())
needs_lo = pytest.mark.skipif(not LO, reason="LibreOffice not installed")


@pytest.fixture
def no_lo(monkeypatch):
    monkeypatch.setattr(formats, "soffice", lambda: None)


def _blocks(doc, kind=None):
    return [b for b in doc["blocks"] if kind is None or b["type"] == kind]


def _text(doc):
    return " ".join(str(b.get("content") or "") for b in doc["blocks"]) + " " + json.dumps([b.get("rows") for b in doc["blocks"]])


def _check_word(doc):
    heads = [b["content"] for b in _blocks(doc, "heading")]
    assert "Sorting Algorithms" in heads and "Comparison" in heads
    t = _blocks(doc, "table")
    assert t and t[0]["rows"][1:] == [r for r in D.TABLE]                     # header row, then the data rows
    assert t[0]["rows"][0][0] == "Running times"                              # the merged title cell
    items = [b["content"] for b in _blocks(doc, "list")]
    assert any("Stable sorts" in i for i in items) and any("In-place sorts" in i for i in items)
    assert any("Insertion sort builds the sorted array" in str(b["content"]) for b in _blocks(doc, "paragraph"))
    assert not any("In-place" in i and "running time" in i for i in items)   # the list ends where the paragraph begins


@needs_lo
def test_docx_with_libreoffice(tmp):
    doc = runner.convert(D.make_docx(tmp / "notes.docx"))
    assert doc["source"] == {"format": "DOCX", "converted_with": "libreoffice"}
    _check_word(doc)


def test_docx_without_libreoffice(tmp, no_lo):
    doc = runner.convert(D.make_docx(tmp / "notes.docx"))
    assert doc["source"]["converted_with"] == "native"
    _check_word(doc)
    assert _blocks(doc, "figure")                                            # the embedded picture
    assert any("rendered by ParseFlow" in w for w in doc["warnings"])


def _check_slides(doc):
    heads = [b["content"] for b in _blocks(doc, "heading")]
    assert {"Graph Algorithms", "Breadth-first search", "Running times", "Edges per graph"} <= set(heads)
    assert any(t["rows"] == D.TABLE for t in _blocks(doc, "table"))
    chart = [t for t in _blocks(doc, "table") if t["rows"][0] == ["", "Edges"]]
    assert chart and chart[0]["rows"][1:] == [["G1", "12"], ["G2", "30"], ["G3", "7"]]   # chart values, not pixels
    assert "FIFO queue" in _text(doc)


@needs_lo
def test_pptx_with_libreoffice(tmp):
    doc = runner.convert(D.make_pptx(tmp / "lecture.pptx"))
    assert doc["source"]["converted_with"] == "libreoffice"
    _check_slides(doc)


def test_pptx_without_libreoffice(tmp, no_lo):
    doc = runner.convert(D.make_pptx(tmp / "lecture.pptx"))
    _check_slides(doc)
    assert "BFS gives shortest paths" in _text(doc)                          # speaker notes kept


def _check_sheet(doc, calculated):
    heads = [b["content"] for b in _blocks(doc, "heading")]
    assert "Sales" in heads and "Notes" in heads
    t = _blocks(doc, "table")
    sales = next(x for x in t if x["rows"][0][0] == "Quarterly sales")
    assert sales["rows"][1:4] == [["Region", "Q1", "Q2"], ["North", "10", "12.5"], ["South", "7", "9"]]
    assert sales["rows"][4][:3] == (["Total", "17", "21.5"] if calculated else ["Total", "=SUM(B3:B4) (not calculated)", "=SUM(C3:C4) (not calculated)"])
    notes = next(x for x in t if x["rows"][0] == ["Item", "Value"])
    assert notes["rows"][1] == ["Growth", "25%"]                              # the cell's number format


@needs_lo
def test_xlsx_formulas_calculated_with_libreoffice(tmp):
    doc = runner.convert(D.make_xlsx(tmp / "sales.xlsx"))
    _check_sheet(doc, True)
    assert any("LibreOffice calculated" in w for w in doc["warnings"])


def test_xlsx_without_libreoffice_shows_formulas(tmp, no_lo):
    doc = runner.convert(D.make_xlsx(tmp / "sales.xlsx"))
    _check_sheet(doc, False)


def test_csv_semicolon(tmp):
    doc = runner.convert(D.make_csv(tmp / "grades.csv"))
    assert _blocks(doc, "table")[0]["rows"] == [["name", "score", "grade"], ["Asha", "91", "A"], ["Ravi", "78", "B"]]
    assert "grades" in [b["content"] for b in _blocks(doc, "heading")]


def test_markdown_html_text(tmp):
    md = runner.convert(D.make_md(tmp / "heaps.md"))
    assert [b["content"] for b in _blocks(md, "heading")] == ["Heaps"]
    assert _blocks(md, "table")[0]["rows"] == [["Op", "Cost"], ["build", "O(n)"]]
    assert len(_blocks(md, "list")) == 2
    (tmp / "s.html").write_text("<h1>Stacks</h1><p>LIFO order.</p><script>document.write('x')</script>")
    h = runner.convert(tmp / "s.html")
    assert "Stacks" in [b["content"] for b in _blocks(h, "heading")] and "document.write" not in _text(h)
    (tmp / "n.txt").write_text("First paragraph.\n\nSecond paragraph about queues.")
    t = runner.convert(tmp / "n.txt")
    assert "Second paragraph about queues." in _text(t)


@pytest.mark.ocr
def test_images_of_every_kind(tmp):
    tif = runner.convert(D.make_image(tmp / "scan.tiff", "TIFF", frames=2))
    assert len(tif["pages"]) == 2 and "Page 2 scanned text" in _text(tif)
    for fmt, ext in (("BMP", "bmp"), ("WEBP", "webp"), ("GIF", "gif")):
        d = runner.convert(D.make_image(tmp / f"x.{ext}", fmt))
        assert "Binary search halves the range" in _text(d), ext
    up = runner.convert(D.make_image(tmp / "phone.jpg", "JPEG", exif_rotate=True))
    assert "Binary search halves the range" in _text(up)                     # read upright, not sideways
    assert any("upright" in w for w in up["warnings"])


def test_epub_via_mupdf(tmp):
    import zipfile
    p = tmp / "book.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="c.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr("c.opf", '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title><dc:identifier id="id">x</dc:identifier></metadata>'
                   '<manifest><item id="c" href="c.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c"/></spine></package>')
        z.writestr("c.xhtml", '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Queues</h1><p>First in, first out.</p></body></html>')
    d = runner.convert(p)
    assert "First in, first out." in _text(d)


def test_mislabelled_and_unsupported_files(tmp, no_lo):
    pdf = tmp / "real.pdf"
    doc = pymupdf.open(); doc.new_page().insert_text((72, 72), "A PDF named .docx"); doc.save(pdf)
    (tmp / "renamed.docx").write_bytes(pdf.read_bytes())
    d = runner.convert(tmp / "renamed.docx")
    assert "A PDF named .docx" in _text(d)
    with pytest.raises(formats.ConvertError) as e:
        formats.check_supported("song.mp3")
    assert e.value.code == "PF-204"
    (tmp / "old.doc").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 600)
    with pytest.raises(formats.ConvertError) as e:
        formats.prepare(str(tmp / "old.doc"), "old.doc", str(tmp), "x")
    assert e.value.code == "PF-207" and "LibreOffice" in e.value.message


@needs_lo
def test_legacy_binary_formats_with_libreoffice(tmp):
    import subprocess
    D.make_docx(tmp / "a.docx"); D.make_pptx(tmp / "b.pptx"); D.make_xlsx(tmp / "c.xlsx")
    lo = formats.soffice()
    for src, to in (("a.docx", "doc"), ("b.pptx", "ppt"), ("c.xlsx", "xls")):
        subprocess.run([lo, f"-env:UserInstallation=file://{tmp}/lo", "--headless", "--convert-to", to, "--outdir", str(tmp), str(tmp / src)],
                       capture_output=True, timeout=180)
    _check_word(runner.convert(tmp / "a.doc"))
    assert any(t["rows"] == D.TABLE for t in _blocks(runner.convert(tmp / "b.ppt"), "table"))
    _check_sheet(runner.convert(tmp / "c.xls"), True)


def test_api_upload_word_file(tmp, monkeypatch, no_lo):
    monkeypatch.setenv("PARSEFLOW_DATA_DIR", str(tmp / "data"))
    monkeypatch.setenv("PARSEFLOW_ISOLATE", "0")
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    from fastapi.testclient import TestClient
    c = TestClient(app.app)
    f = D.make_docx(tmp / "Lecture Notes.docx")
    r = c.post("/api/process", files={"file": (f.name, f.read_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}).json()
    did = r["document_id"]
    assert r["type"] == "DOCX" and r["pages"] >= 1
    for _ in range(400):
        if c.get(f"/api/status/{did}").json().get("status") in ("complete", "failed"):
            break
        time.sleep(0.05)
    doc = c.get(f"/api/document/{did}").json()
    assert doc["filename"] == "Lecture Notes.docx" and doc["source"]["format"] == "DOCX"
    _check_word(doc)
    files = sorted(p.name for p in (tmp / "data" / did).iterdir())
    assert any(n.startswith("original_") for n in files) and any(n.startswith("source_") and n.endswith(".pdf") for n in files)
    assert c.get(f"/api/page/{did}/1.png").status_code == 200
    bad = c.post("/api/process", files={"file": ("x.mp3", b"ID3....", "audio/mpeg")})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "PF-204"
    h = c.get("/api/health").json()["formats"]
    assert "docx" in h["supported"] and "xlsx" in h["supported"] and h["libreoffice"] is False
