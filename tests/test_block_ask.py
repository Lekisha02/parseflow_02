"""Ask about one block: the evidence sent to the model (block, structure read from the drawing, computed facts,
caption, surrounding text, other pages that refer to the figure, the image of the region), extraction kept separate
from the answer, page references checked against the evidence, the offline answer, and the per-block chat API."""
import base64
import importlib
import json
import math
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from conftest import LOREM, PDF, paragraph, text
from processing import assistant, block_ask, pipeline

ENV = ("ASK_PROVIDER", "ASK_API_KEY", "VISION_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
       "ASK_BASE_URL", "ASK_VISION")


def _dfa_book(path):
    from test_diagrams import _dfa
    pdf = PDF()

    def p1(p):
        text(p, 72, 50, "1 Finite automata", 16, "hebo")
        _dfa(p)
        text(p, 72, 200, "Figure 1.1 A DFA that accepts binary strings ending in 1.", 9)
        paragraph(p, 72, 240, 450, "The machine starts in state q0. Reading a 1 moves it to q1 and reading a 0 moves it "
                                   "back to q0, so it only remembers the last symbol. " + LOREM)

    def p2(p):
        text(p, 72, 60, "2 Why it works", 16, "hebo")
        paragraph(p, 72, 100, 450, "As Figure 1.1 shows, the automaton is in the accepting state exactly when the last "
                                   "symbol read was a 1. " + LOREM)
    pdf.page(p1).page(p2)
    return pdf.save(path)


@pytest.fixture
def no_llm(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def dfa_doc(tmp, no_llm):
    folder = tmp / "abcabcabcabc"
    folder.mkdir()
    src = _dfa_book(tmp / "dfa.pdf")
    shutil.copy(src, folder / "source_abcabcabcabc_dfa.pdf")
    doc = pipeline.process(str(src), "dfa.pdf", "abcabcabcabc", str(folder), pipeline.Progress(), save_images=False)
    (folder / "document.json").write_text(json.dumps(doc))
    diag = [b for b in doc["blocks"] if b["type"] == "diagram"]
    assert diag, [b["type"] for b in doc["blocks"]]
    return doc, diag[0], folder


def test_diagram_block_carries_its_structure(dfa_doc):
    doc, b, _ = dfa_doc
    assert b["extractor"] == "vector_diagram" and b["page"] == 1
    a = [p for p in b["structure"]["parts"] if p["type"] == "automaton"][0]
    assert a["start"] == "q0" and a["accepting"] == ["q1"]
    assert "deterministic finite automaton" in b["structure_text"]


def test_offline_block_answer_is_evidence_and_computed_run(dfa_doc):
    doc, b, folder = dfa_doc
    r = block_ask.respond(doc, b["block_id"], "Does it accept the string 0101?", folder=str(folder))
    assert r["mode"] == "extractive" and r["block"] == dict(id=b["block_id"], page=1, type="diagram")
    assert any("“0101”" in f and "ACCEPTED" in f for f in r["computed"])
    assert "What the drawing shows" in r["answer"] and "q0" in r["answer"]
    assert "Figure 1.1" in r["answer"]                                  # the caption
    assert "p. 2" in r["answer"] or any(c["page"] == 2 for c in r["citations"])   # the page that refers to it
    # extraction quality is reported on its own, never as the confidence of the answer
    assert set(r["extraction"]) >= {"verification", "confidence", "method"} and "confidence" not in r
    assert "No AI model configured" in r["note"]
    rej = block_ask.respond(doc, b["block_id"], "trace the input 110", folder=str(folder))
    assert any("“110”" in f and "REJECTED" in f for f in rej["computed"])


def test_unknown_block_is_an_error(dfa_doc):
    doc, b, folder = dfa_doc
    assert block_ask.respond(doc, "nope_1", "x", folder=str(folder))["error"]["code"] == "PF-404"


class _Vision(BaseHTTPRequestHandler):
    seen = []
    reply = ""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Vision.seen.append(body)
        out = json.dumps({"choices": [{"message": {"content": _Vision.reply}}]}).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def vision_llm(monkeypatch, no_llm):
    srv = HTTPServer(("127.0.0.1", 0), _Vision)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Vision.seen.clear()
    monkeypatch.setenv("ASK_PROVIDER", "openai")
    monkeypatch.setenv("ASK_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.setenv("ASK_MODEL", "local-vision-model")
    monkeypatch.setenv("ASK_VISION", "1")
    yield _Vision
    srv.shutdown()


def test_model_gets_block_structure_facts_context_and_image(dfa_doc, vision_llm):
    doc, b, folder = dfa_doc
    vision_llm.reply = ("**From the document**\nThe DFA starts in q0 and accepts in q1 [p. 1]; page 2 explains why [p. 2]. "
                        "See also [p. 9].\n\n**Explanation**\n1. Read 0 → q0. 2. Read 1 → q1 ...")
    hist = [dict(role="user", content="What is this?"), dict(role="assistant", content="A DFA [p. 1].")]
    r = block_ask.respond(doc, b["block_id"], "Does it accept 0101? Explain step by step.", history=hist, folder=str(folder))
    assert r["mode"] == "llm" and r["used_image"] is True and r["llm"]["model"] == "local-vision-model"
    msgs = vision_llm.seen[-1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]              # follow-up context kept
    assert "Never invent" in msgs[0]["content"] or "never invent" in msgs[0]["content"].lower()
    last = msgs[-1]["content"]
    parts = last if isinstance(last, list) else [dict(type="text", text=last)]
    txt = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
    img = [p for p in parts if p.get("type") == "image_url"]
    assert img and img[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(img[0]["image_url"]["url"].split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"
    assert "SELECTED BLOCK [p. 1]" in txt and "STRUCTURE (source: vector drawing (exact geometry))" in txt
    assert "COMPUTED FACTS" in txt and "ACCEPTED" in txt
    assert "Figure 1.1 A DFA" in txt and "OTHER PAGES REFERRING TO" in txt and "[p. 2]" in txt
    assert "step" in txt.lower()                                         # the requested style reaches the model
    # a page reference outside the evidence is removed and reported
    assert "[p. 9]" not in r["answer"] and "p. 9" in r["note"] and "[p. 1]" in r["answer"]


def test_no_image_when_the_model_cannot_see(dfa_doc, vision_llm, monkeypatch):
    monkeypatch.setenv("ASK_VISION", "0")
    doc, b, folder = dfa_doc
    vision_llm.reply = "It is a DFA [p. 1]."
    r = block_ask.respond(doc, b["block_id"], "What is this?", folder=str(folder))
    assert r["mode"] == "llm" and r["used_image"] is False
    assert isinstance(vision_llm.seen[-1]["messages"][-1]["content"], str)


def test_check_citations():
    t, removed = assistant.check_citations("A [p. 1]. B [pp. 2–3]. C [p. 7]. D [p. 3, 8].", {1, 2, 3})
    assert removed == [7, 8] or set(removed) == {7, 8}
    assert "[p. 1]" in t and "[p. 7]" not in t and "8" not in t.split("D")[-1]


def test_block_chat_api(tmp, monkeypatch, no_llm):
    monkeypatch.setenv("PARSEFLOW_DATA_DIR", str(tmp / "data"))
    monkeypatch.setenv("PARSEFLOW_ISOLATE", "0")
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    from fastapi.testclient import TestClient
    c = TestClient(app.app)
    src = _dfa_book(tmp / "dfa.pdf")
    did = c.post("/api/process", files={"file": ("dfa.pdf", src.read_bytes(), "application/pdf")}).json()["document_id"]
    for _ in range(400):
        if c.get(f"/api/status/{did}").json().get("status") in ("complete", "failed"):
            break
        time.sleep(0.05)
    doc = c.get(f"/api/document/{did}").json()
    bid = next(b["block_id"] for b in doc["blocks"] if b["type"] == "diagram")
    r = c.post(f"/api/ask/{did}/block/{bid}", json=dict(question="Does it accept 11?", history=[])).json()
    assert r["mode"] == "extractive" and any("ACCEPTED" in f for f in r["computed"])
    c.post(f"/api/ask/{did}", json=dict(question="What is a DFA?", history=[]))
    blk = c.get(f"/api/chat/{did}?block={bid}").json()["messages"]
    main = c.get(f"/api/chat/{did}").json()["messages"]
    assert [m["role"] for m in blk] == ["user", "assistant"] and blk[0]["text"] == "Does it accept 11?"
    assert all(not m.get("block") for m in main) and len(main) == 2
    assert c.post(f"/api/ask/{did}/block/../../x", json=dict(question="x")).status_code in (400, 404, 405)
    assert c.post(f"/api/ask/{did}/block/nope_9", json=dict(question="x")).status_code == 404
    c.delete(f"/api/chat/{did}?block={bid}")
    assert c.get(f"/api/chat/{did}?block={bid}").json()["messages"] == []
    assert len(c.get(f"/api/chat/{did}").json()["messages"]) == 2           # the main chat is untouched
    other = TestClient(app.app)                                              # another visitor cannot ask about it
    assert other.post(f"/api/ask/{did}/block/{bid}", json=dict(question="x")).status_code in (403, 404)
