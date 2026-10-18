"""Scanned-page OCR: adaptive preprocessing on synthetic degradations, fallback, timeouts, hybrid pages."""
import io
import re
from difflib import SequenceMatcher

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageFilter

from conftest import PDF, paragraph, text
from processing import ocr

pytestmark = pytest.mark.ocr

TRUTH = ("Old scans are often faded skewed and noisy but the words on the page still carry the meaning "
         "that a reader expects to recover from the document without guessing anything that is not there")


def _clean_image(size=11, zoom=2.0, width=470):
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=300)
    paragraph(p, 60, 60, width, TRUTH, size=size, font="tiro")
    pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def _acc(words):
    got = " ".join(w["text"] for w in words)
    got = re.sub(r"[^a-z ]", "", got.lower())
    return SequenceMatcher(None, got.split(), TRUTH.lower().split()).ratio()


def _degrade(img, kind, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(img, dtype=np.float32)
    if kind == "skew":
        return img.rotate(3.0, resample=Image.BICUBIC, expand=True, fillcolor=255)
    if kind == "noise":
        a = a.copy(); m = rng.random(a.shape)
        a[m < 0.03] = 0; a[m > 0.97] = 255
        return Image.fromarray(a.astype(np.uint8))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(1.2))
    if kind == "low_contrast":
        return Image.fromarray((150 + a * (80 / 255)).astype(np.uint8))
    if kind == "uneven":
        h, w = a.shape
        grad = np.linspace(0, 110, w)[None, :]
        return Image.fromarray(np.clip(a - grad, 0, 255).astype(np.uint8))
    raise ValueError(kind)


def test_assess_measures_skew_and_contrast():
    q = ocr.assess(_degrade(_clean_image(), "skew"))
    assert 2.0 <= abs(q["skew_deg"]) <= 4.0
    q2 = ocr.assess(_degrade(_clean_image(), "low_contrast"))
    assert q2["contrast"] < 110


@pytest.mark.parametrize("kind", ["skew", "noise", "blur", "low_contrast", "uneven"])
def test_preprocessing_not_worse_and_reads_text(kind):
    img = _degrade(_clean_image(), kind)
    raw = _acc(ocr.words(img, 3))
    res = ocr.ocr_image(img)
    adaptive = _acc(res["words"])
    # adaptive preprocessing must never make things worse than plain OCR, and must read most of the text
    assert adaptive >= raw - 0.02, (kind, raw, adaptive, res["steps"])
    assert adaptive >= 0.75, (kind, adaptive, res["steps"], res["attempts"])
    assert len(res["attempts"]) <= 2                       # bounded fallback


def test_small_text_is_upscaled():
    img = _clean_image(size=7, zoom=1.4, width=470)       # small print scanned at ~100 dpi
    raw = _acc(ocr.words(img, 3))
    res = ocr.ocr_image(img)
    assert any("upscaled" in s for s in res["steps"]), res["steps"]
    assert _acc(res["words"]) >= raw - 0.02 and _acc(res["words"]) >= 0.6, (raw, _acc(res["words"]))


def test_scanned_pdf_page_end_to_end(tmp, convert):
    img = _degrade(_clean_image(), "skew")
    buf = io.BytesIO(); img.convert("RGB").save(buf, "PNG")
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=320)
    p.insert_image(p.rect, stream=buf.getvalue())
    f = tmp / "scan.pdf"; doc.save(str(f))
    d = convert(f)
    pg = d["pages"][0]
    assert pg["route"] == "ocr" and pg["ocr"]["steps"]
    md = d["markdown"].lower()
    assert "faded" in md and "meaning" in md
    ocr_blocks = [b for b in d["blocks"] if b["extractor"] == "ocr"]
    assert ocr_blocks and all(b["verification"] in ("review", "failed") for b in ocr_blocks)   # OCR alone is never 'verified'


def test_ocr_timeout_is_reported_not_fatal(tmp, convert, monkeypatch):
    img = _clean_image()
    buf = io.BytesIO(); img.convert("RGB").save(buf, "PNG")
    doc = pymupdf.open(); p = doc.new_page(); p.insert_image(p.rect, stream=buf.getvalue())
    f = tmp / "to.pdf"; doc.save(str(f))
    monkeypatch.setenv("OCR_TIMEOUT", "0.001")
    ocr.clear_cache()
    d = convert(f)
    assert d["processing_status"] == "complete"
    assert d["blocks"] and any(b["extractor"] == "fallback" for b in d["blocks"])
    assert any("Extractor error" in w for b in d["blocks"] for w in b["warnings"])
    ocr.clear_cache()


def test_hybrid_page_keeps_native_text_over_image(tmp, convert):
    noise = Image.fromarray((np.random.default_rng(1).random((400, 300)) * 255).astype(np.uint8)).convert("RGB")
    buf = io.BytesIO(); noise.save(buf, "PNG")

    def draw(p):
        p.insert_image(p.rect, stream=buf.getvalue())
        text(p, 100, 400, "THE POWER OF NATIVE TEXT", 24, "hebo")
    d = convert(PDF().page(draw).save(tmp / "hyb.pdf"))
    assert "THE POWER OF NATIVE TEXT" in d["markdown"]
