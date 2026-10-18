"""Opt-in check against a REAL language model (not run by default; needs network or a local model).

    PARSEFLOW_LIVE_LLM=1 ASK_PROVIDER=anthropic ASK_API_KEY=... pytest -m live tests/test_live_llm.py
    PARSEFLOW_LIVE_LLM=1 ASK_PROVIDER=ollama ASK_MODEL=llama3.1 pytest -m live tests/test_live_llm.py
"""
import os

import pytest

from conftest import LOREM, PDF, paragraph, text
from processing import assistant, llm

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.getenv("PARSEFLOW_LIVE_LLM") != "1", reason="set PARSEFLOW_LIVE_LLM=1 and configure ASK_PROVIDER")]


def test_live_answer_cites_pages_and_explains(tmp, convert):
    assert llm.available(), "ASK_PROVIDER is not configured"
    pdf = PDF()
    pdf.page(lambda p: (text(p, 72, 80, "1 Insertion sort", 18, "hebo"),
                        paragraph(p, 72, 120, 450, "Insertion sort builds the sorted array one element at a time. " + LOREM)))
    pdf.page(lambda p: (text(p, 72, 80, "2 Merge sort", 18, "hebo"),
                        paragraph(p, 72, 120, 450, "Merge sort divides the array into halves, sorts each half recursively and "
                                                   "merges them in linear time, so it runs in n log n time. " + LOREM)))
    doc = convert(pdf.save(tmp / "live.pdf"))
    r = assistant.respond(doc, "Explain merge sort in simple words with a small example.")
    assert r["mode"] == "llm", r.get("note")
    assert "From the document" in r["answer"] and "Explanation" in r["answer"]
    assert any(c["cited"] and c["page"] == 2 for c in r["citations"])
    f = assistant.respond(doc, "and how fast is it?", history=[dict(role="user", content="Explain merge sort"),
                                                             dict(role="assistant", content=r["answer"], chunk_ids=r["chunk_ids"])])
    assert f["mode"] == "llm" and ("log" in f["answer"].lower())


def test_live_block_answer_about_a_drawn_automaton(tmp):
    """The real model, asked about a vector DFA: it must agree with ParseFlow's computed run and cite the page."""
    import json
    import shutil
    from test_block_ask import _dfa_book
    from processing import block_ask, pipeline
    assert llm.available(), "ASK_PROVIDER is not configured"
    folder = tmp / "abcabcabcabc"; folder.mkdir()
    src = _dfa_book(tmp / "dfa.pdf")
    shutil.copy(src, folder / "source_abcabcabcabc_dfa.pdf")
    doc = pipeline.process(str(src), "dfa.pdf", "abcabcabcabc", str(folder), pipeline.Progress(), save_images=False)
    b = next(x for x in doc["blocks"] if x["type"] == "diagram")
    r = block_ask.respond(doc, b["block_id"], "Does this automaton accept the string 0110? Trace it step by step.", folder=str(folder))
    assert r["mode"] == "llm", r.get("note")
    assert any("REJECTED" in f for f in r["computed"])                  # 0110 ends in 0
    ans = r["answer"].lower()
    assert "reject" in ans or "not accept" in ans or "does not" in ans, r["answer"]
    assert "[p. 1" in r["answer"]
    print(json.dumps(dict(used_image=r["used_image"], answer=r["answer"][:1500]), indent=1))
