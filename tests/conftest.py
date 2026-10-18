"""Shared fixtures. Every PDF used by the unit tests is generated here with PyMuPDF, so the suite needs no binary
fixtures, no network and no model downloads. Integration tests against the official olmOCR-Bench sample data
are marked `bench` and skipped unless OLMOCR_PATH points at an olmocr checkout."""
import os
import shutil
import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from processing import ocr, runner  # noqa: E402

HAS_TESSERACT = bool(ocr.available())
OLMOCR = os.getenv("OLMOCR_PATH")
SAMPLE = Path(OLMOCR) / "olmocr" / "bench" / "sample_data" if OLMOCR else None


def pytest_collection_modifyitems(config, items):
    for it in items:
        if "ocr" in it.keywords and not HAS_TESSERACT:
            it.add_marker(pytest.mark.skip(reason="tesseract binary not installed"))
        if "bench" in it.keywords and not (SAMPLE and SAMPLE.exists()):
            it.add_marker(pytest.mark.skip(reason="set OLMOCR_PATH to an olmocr checkout to run official-scorer regression tests"))


LOREM = ("ParseFlow keeps the evidence behind every extracted block so that a reviewer can always return to the "
         "original page region and check what was read and how it was read before trusting it.")


class PDF:
    """Tiny builder: PDF().page(fn).page(fn).save(path) where fn(page) draws content."""

    def __init__(self, w=612, h=792):
        self.doc = pymupdf.open()
        self.w, self.h = w, h

    def page(self, fn=None, w=None, h=None):
        p = self.doc.new_page(width=w or self.w, height=h or self.h)
        if fn:
            fn(p)
        return self

    def save(self, path, **kw):
        self.doc.save(str(path), **kw)
        self.doc.close()
        return Path(path)


def text(page, x, y, s, size=10, font="helv"):
    page.insert_text((x, y), s, fontsize=size, fontname=font)


def unicode_text(page, x, y, s, size=11):
    """Insert text with PyMuPDF's built-in fallback font (covers Greek and math symbols; ships with PyMuPDF)."""
    tw = pymupdf.TextWriter(page.rect)
    tw.append((x, y), s, font=pymupdf.Font("cjk"), fontsize=size)
    tw.write_text(page)


def paragraph(page, x, y, width, s, size=10, lead=1.25, font="helv"):
    """Greedy-wrap s into lines of at most `width` points; returns the y after the last line."""
    f = pymupdf.Font(font if font != "hebo" else "hebo")
    words, line = s.split(), ""
    for w in words:
        cand = (line + " " + w).strip()
        if f.text_length(cand, fontsize=size) > width and line:
            text(page, x, y, line, size, font); y += size * lead; line = w
        else:
            line = cand
    if line:
        text(page, x, y, line, size, font); y += size * lead
    return y


@pytest.fixture
def convert():
    def _c(path, **kw):
        return runner.convert(str(path), **kw)
    return _c


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def require_olmocr():
    if not (SAMPLE and SAMPLE.exists()):
        pytest.skip("OLMOCR_PATH not set")
    if OLMOCR not in sys.path:
        sys.path.insert(0, OLMOCR)


def copy_sample(name, dst):
    shutil.copy(SAMPLE / "pdfs" / name, dst)
    return Path(dst)
