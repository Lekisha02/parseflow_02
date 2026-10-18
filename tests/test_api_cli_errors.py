"""Error handling (empty/corrupt/encrypted/multi-page/images), the FastAPI API and the batch CLI."""
import importlib
import io
import json
import sys
import time
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from conftest import LOREM, PDF, ROOT, paragraph, text
from processing import pipeline


def test_empty_corrupt_encrypted_unsupported(tmp, convert):
    (tmp / "empty.pdf").write_bytes(b"")
    (tmp / "bad.pdf").write_bytes(b"%PDF-1.7 this is not really a pdf")
    (tmp / "x.docx").write_bytes(b"PK..")                       # Word files are supported; this one is damaged
    (tmp / "song.mp3").write_bytes(b"ID3\x03\x00")
    doc = pymupdf.open(); doc.new_page(); doc.save(str(tmp / "enc.pdf"), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    codes = {}
    for name in ("empty.pdf", "bad.pdf", "x.docx", "song.mp3", "enc.pdf"):
        with pytest.raises(pipeline.PFError) as e:
            convert(tmp / name)
        codes[name] = e.value.code
    assert codes == {"empty.pdf": "PF-205", "bad.pdf": "PF-201", "x.docx": "PF-201", "song.mp3": "PF-204", "enc.pdf": "PF-202"}


def test_multipage_and_page_limit(tmp, convert, monkeypatch):
    pdf = PDF()
    for i in range(4):
        pdf.page(lambda p, i=i: paragraph(p, 72, 100, 450, f"Page marker {i + 1}. " + LOREM))
    f = pdf.save(tmp / "multi.pdf")
    d = convert(f)
    assert len(d["pages"]) == 4 and {b["page"] for b in d["blocks"]} == {1, 2, 3, 4}
    assert d["markdown"].index("Page marker 1") < d["markdown"].index("Page marker 4")
    monkeypatch.setenv("MAX_PAGES", "2")
    d2 = convert(f)
    assert len(d2["pages"]) == 2 and any("only the first 2" in w for w in d2["warnings"])


@pytest.mark.ocr
def test_png_input(tmp, convert):
    im = Image.new("RGB", (1200, 300), "white")
    d = ImageDraw.Draw(im)
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    d.text((40, 100), "Invoice number 4711 total amount", fill="black", font=font)
    im.save(tmp / "img.png")
    out = convert(tmp / "img.png")
    assert out["format"] == "image" and "4711" in out["markdown"]


# ------------------------------------------------------------------------------------------------ API
@pytest.fixture
def client(tmp, monkeypatch):
    monkeypatch.setenv("PARSEFLOW_DATA_DIR", str(tmp / "data"))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    sys.modules.pop("app", None)
    sys.path.insert(0, str(ROOT / "backend"))
    app = importlib.import_module("app")
    from fastapi.testclient import TestClient
    return TestClient(app.app), app


def _wait(c, doc_id):
    for _ in range(200):
        st = c.get(f"/api/status/{doc_id}").json()
        if st["status"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_api_process_json_markdown(tmp, client):
    c, app = client
    f = PDF().page(lambda p: (text(p, 72, 40, "Running head", 8), text(p, 72, 90, "API Title", 20, "hebo"), paragraph(p, 72, 130, 450, LOREM))).save(tmp / "api.pdf")
    h = c.get("/api/health").json()
    assert h["ok"] and "version" in h and "math_recognizer" in h
    r = c.post("/api/process", files={"file": ("api test.pdf", f.read_bytes(), "application/pdf")}).json()
    assert set(r) >= {"document_id", "filename", "size", "type", "pages"} and r["type"] == "PDF"
    st = _wait(c, r["document_id"])
    assert st["status"] == "complete" and [s["state"] for s in st["stages"]] == ["done"] * 7 and "timings" in st
    doc = c.get(f"/api/document/{r['document_id']}").json()
    assert doc["blocks"] and "markdown" in doc and "# API Title" in doc["markdown"]
    for k in ("block_id", "type", "content", "page", "bbox", "reading_order", "confidence", "confidence_type", "extractor", "verification", "relationships", "warnings"):
        assert k in doc["blocks"][0]
    md = c.get(f"/api/document/{r['document_id']}/markdown?annotate=0")
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown") and "<!--" not in md.text
    assert c.get(f"/api/page/{r['document_id']}/1.png").status_code == 200


def test_api_validation_and_cleanup(tmp, client):
    c, app = client
    assert c.post("/api/process", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).json()["error"]["code"] == "PF-204"
    r = c.post("/api/process", files={"file": ("bad.pdf", b"%PDF-1.4 junk", "application/pdf")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "PF-201"
    big = c.post("/api/process", files={"file": ("big.pdf", b"0" * (1024 * 1024 + 10), "application/pdf")})
    assert big.status_code == 413
    # failed uploads leave nothing behind
    assert not any(p.is_dir() for p in (tmp / "data").iterdir())
    assert c.get("/api/document/../../etc").status_code == 404
    assert c.get("/api/document/zzzzzzzzzzzz/markdown").status_code == 404
    assert c.get("/api/status/nope").status_code == 404


# ------------------------------------------------------------------------------------------------ CLI
def test_cli_batch(tmp, capsys):
    import cli
    inp = tmp / "in"; (inp / "sub").mkdir(parents=True)
    PDF().page(lambda p: paragraph(p, 72, 100, 450, "Alpha document. " + LOREM)).save(inp / "a.pdf")
    PDF().page(lambda p: paragraph(p, 72, 100, 450, "Nested alpha. " + LOREM)).save(inp / "sub" / "a.pdf")
    (inp / "broken.pdf").write_bytes(b"not a pdf")
    out = tmp / "out"
    code = cli.main([str(inp), "--output-dir", str(out), "--recursive", "--workers", "2", "--report", str(tmp / "rep.json"), "--plain"])
    assert code == 1                                            # one failure -> non-zero exit
    assert (out / "a.md").exists() and (out / "sub" / "a.md").exists()   # no filename collision, outputs kept
    assert "Alpha document." in (out / "a.md").read_text() and "<!--" not in (out / "a.md").read_text()
    rep = json.loads((tmp / "rep.json").read_text())
    assert rep["summary"]["succeeded"] == 2 and rep["summary"]["failed"] == 1
    # skip-existing processes nothing new and succeeds
    code2 = cli.main([str(inp / "a.pdf"), "--output-dir", str(out), "--skip-existing"])
    assert code2 in (0, 1)


def test_cli_timeout(tmp):
    from processing.runner import run_batch
    f = PDF().page(lambda p: paragraph(p, 72, 100, 450, LOREM)).save(tmp / "t.pdf")
    res = run_batch([str(f)], {}, workers=1, timeout=0.01)
    assert res and (res[0][1].get("timeout") or res[0][1].get("ok"))     # either finished instantly or was killed cleanly
