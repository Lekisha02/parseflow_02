"""Deterministic regression check on the official olmOCR-Bench *sample* data, using the OFFICIAL test classes
(olmocr.bench.tests). Every test that passed when the expectation file was recorded must still pass.

Run:  OLMOCR_PATH=/path/to/olmocr pytest -m bench tests/test_regression_olmocr_sample.py
Re-record after an intentional change:  PARSEFLOW_RECORD_EXPECTED=1 OLMOCR_PATH=... pytest -m bench ...
"""
import json
import os
from pathlib import Path

import pytest

from conftest import ROOT, SAMPLE, require_olmocr
from processing import markdown as mdgen
from processing import runner

EXPECTED = ROOT / "bench" / "baselines" / "sample_expected_pass.json"

pytestmark = [pytest.mark.bench, pytest.mark.slow]


def _page_md(doc, page):
    sub = dict(doc, blocks=[b for b in doc["blocks"] if b["page"] == page], warnings=[], filename=None)
    return mdgen.render(sub, annotate=False)


def test_sample_regression():
    require_olmocr()
    from olmocr.bench.tests import load_tests
    tests = []
    for jf in sorted(SAMPLE.glob("*.jsonl")):
        tests += load_tests(str(jf))
    by_pdf = {}
    for t in tests:
        by_pdf.setdefault(t.pdf, []).append(t)
    results, cache = {}, {}
    for pdf, ts in sorted(by_pdf.items()):
        doc = runner.convert(str(SAMPLE / "pdfs" / pdf))
        for t in ts:
            md = cache.setdefault((pdf, t.page), _page_md(doc, t.page))
            try:
                ok, why = t.run(md)
            except Exception as e:      # the scorer itself failing counts as a failed test
                ok, why = False, f"{type(e).__name__}: {e}"
            results[t.id] = dict(ok=ok, why=why, pdf=pdf, page=t.page, type=t.type)
    passed = sorted(k for k, v in results.items() if v["ok"])
    if os.getenv("PARSEFLOW_RECORD_EXPECTED"):
        EXPECTED.parent.mkdir(parents=True, exist_ok=True)
        EXPECTED.write_text(json.dumps(dict(passed=passed, total=len(results)), indent=1))
        pytest.skip(f"recorded {len(passed)}/{len(results)} passing tests")
    exp = json.loads(EXPECTED.read_text())["passed"]
    lost = [k for k in exp if k in results and not results[k]["ok"]]
    msg = "\n".join(f"{results[k]['pdf']} p{results[k]['page']} [{results[k]['type']}] {k}: {results[k]['why']}" for k in lost)
    assert not lost, f"{len(lost)} previously passing official tests now fail:\n{msg}"
