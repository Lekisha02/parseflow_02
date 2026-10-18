"""Ask assistant: retrieval across the whole document, summaries, follow-ups, the language-model path (with a local
stub of an OpenAI-compatible server and mocked Anthropic / Gemini transports), and the offline fallback."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from conftest import LOREM, PDF, paragraph, text
from processing import assistant, llm

TOPICS = [
    ("Insertion sort", "Insertion sort builds the sorted array one element at a time. Each new key is compared with the "
                       "elements to its left and inserted where it belongs. Its worst-case running time is quadratic in n."),
    ("Merge sort", "Merge sort divides the array into two halves, sorts each half recursively and merges the two sorted "
                   "halves. The merge procedure takes linear time, so merge sort runs in n log n time."),
    ("Loop invariants", "A loop invariant is a property that holds before and after every iteration of a loop. "
                        "To use one you show initialization, maintenance and termination."),
    ("Heaps", "A binary heap is an array that can be viewed as a nearly complete binary tree. In a max-heap every parent "
              "is at least as large as its children, so the largest element is at the root."),
]


@pytest.fixture
def book(tmp, convert):
    pdf = PDF()
    for k, (title, body) in enumerate(TOPICS):
        def pg(p, title=title, body=body, k=k):
            text(p, 72, 80, f"{k + 1} {title}", 18, "hebo")
            y = paragraph(p, 72, 120, 450, body + " " + LOREM)
            paragraph(p, 72, y + 20, 450, "Some unrelated filler text about the history of computing and early machines. " * 3)
        pdf.page(pg)
    return convert(pdf.save(tmp / "book.pdf"))


@pytest.fixture
def no_llm(monkeypatch):
    for k in ("ASK_PROVIDER", "ASK_API_KEY", "VISION_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ASK_BASE_URL"):
        monkeypatch.delenv(k, raising=False)


def test_retrieval_finds_the_explaining_passage_not_the_first_mention(book, no_llm):
    ix = assistant.DocIndex(book)
    hits = ix.search(assistant.ask.tokens("what is a loop invariant"), k=3, phrases=assistant.ask._phrases("what is a loop invariant"))
    top = ix.chunks[hits[0][1]]
    assert 3 in top["pages"] and "property that holds" in top["text"]


def test_offline_question_summary_and_follow_up(book, no_llm):
    r = assistant.respond(book, "How does merge sort work?")
    assert r["mode"] == "extractive" and r["llm"]["configured"] is False
    assert r["extract"]["found"] and any(e["page"] == 2 for e in r["extract"]["evidence"])
    assert "AI model" in r["note"]
    s = assistant.respond(book, "Summarize this document")
    assert s["intent"] == "summary" and "**Outline**" in s["answer"] and "Merge sort" in s["answer"]
    assert {c["page"] for c in s["citations"]} >= {1, 2}
    hist = [dict(role="user", content="What is a binary heap?"), dict(role="assistant", content="...", chunk_ids=r.get("chunk_ids"))]
    f = assistant.respond(book, "explain it more simply", history=hist)
    assert f["extract"]["found"] and any(e["page"] == 4 for e in f["extract"]["evidence"])


class _Stub(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Stub.seen.append(dict(path=self.path, body=body, auth=self.headers.get("authorization")))
        user = body["messages"][-1]["content"]
        page = "2" if "Merge sort divides" in user else "1"
        reply = (f"**From the document**\nMerge sort splits the array and merges the halves [p. {page}].\n\n"
                 "**Explanation**\n1. Split.\n2. Sort each half.\n3. Merge: \\(O(n \\log n)\\).")
        out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(out)))
        self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub_llm(monkeypatch, no_llm):
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    _Stub.seen.clear()
    monkeypatch.setenv("ASK_PROVIDER", "openai")
    monkeypatch.setenv("ASK_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.setenv("ASK_MODEL", "local-test-model")
    yield _Stub.seen
    srv.shutdown()


def test_llm_answer_with_citations_context_and_history(book, stub_llm):
    hist = [dict(role="user", content="What does chapter 1 cover?"), dict(role="assistant", content="Insertion sort [p. 1].")]
    r = assistant.respond(book, "How does merge sort work?", history=hist)
    assert r["mode"] == "llm" and "From the document" in r["answer"] and r["llm"]["model"] == "local-test-model"
    req = stub_llm[-1]
    assert req["path"] == "/v1/chat/completions"
    msgs = req["body"]["messages"]
    assert msgs[0]["role"] == "system" and "From the document" in msgs[0]["content"] and "Explanation" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]          # conversation kept
    assert "[p. 2" in msgs[-1]["content"] and "Merge sort divides" in msgs[-1]["content"]   # excerpts with page headers
    assert any(c["cited"] and c["page"] == 2 for c in r["citations"])


def test_llm_summary_uses_outline_and_key_sentences(book, stub_llm):
    r = assistant.respond(book, "Give me a summary of the document")
    assert r["mode"] == "llm" and r["intent"] == "summary"
    user = stub_llm[-1]["body"]["messages"][-1]["content"]
    assert "Outline (headings)" in user and "Key sentences" in user


def test_llm_failure_falls_back_to_quotes(book, monkeypatch, no_llm):
    monkeypatch.setenv("ASK_PROVIDER", "openai")
    monkeypatch.setenv("ASK_BASE_URL", "http://127.0.0.1:9/v1")           # nothing listens there
    monkeypatch.setenv("ASK_TIMEOUT", "2")
    r = assistant.respond(book, "How does merge sort work?")
    assert r["mode"] == "extractive" and "could not answer" in r["note"] and r["extract"]["found"]


def test_provider_request_formats(monkeypatch, no_llm):
    calls = []

    def handler(req):
        calls.append(req)
        if "anthropic" in str(req.url):
            return httpx.Response(200, json={"content": [{"type": "text", "text": "A"}]})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "G"}]}}]})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("ASK_PROVIDER", "anthropic"); monkeypatch.setenv("ASK_API_KEY", "k1")
    assert llm.chat("sys", [dict(role="user", content="hi")], client=client) == "A"
    b = json.loads(calls[-1].content)
    assert calls[-1].headers["x-api-key"] == "k1" and b["system"] == "sys" and b["messages"][0]["content"] == "hi"
    monkeypatch.setenv("ASK_PROVIDER", "gemini")
    assert llm.chat("sys", [dict(role="user", content="q"), dict(role="assistant", content="a"), dict(role="user", content="q2")], client=client) == "G"
    b = json.loads(calls[-1].content)
    assert b["systemInstruction"]["parts"][0]["text"] == "sys" and [c["role"] for c in b["contents"]] == ["user", "model", "user"]
    monkeypatch.setenv("ASK_PROVIDER", "none")
    with pytest.raises(llm.LLMError):
        llm.chat("s", [dict(role="user", content="x")])


def test_api_post_ask(tmp, monkeypatch, no_llm):
    import importlib, sys, time
    from conftest import ROOT
    monkeypatch.setenv("PARSEFLOW_DATA_DIR", str(tmp / "data"))
    monkeypatch.setenv("PARSEFLOW_ISOLATE", "0")
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    from fastapi.testclient import TestClient
    c = TestClient(app.app)
    pdf = PDF()
    for title, body in TOPICS[:2]:
        pdf.page(lambda p, title=title, body=body: (text(p, 72, 80, title, 18, "hebo"), paragraph(p, 72, 120, 450, body)))
    f = pdf.save(tmp / "two.pdf")
    did = c.post("/api/process", files={"file": ("two.pdf", f.read_bytes(), "application/pdf")}).json()["document_id"]
    for _ in range(200):
        if c.get(f"/api/status/{did}").json()["status"] != "running":
            break
        time.sleep(0.05)
    r = c.post(f"/api/ask/{did}", json=dict(question="What does merge sort do?", history=[])).json()
    assert r["mode"] == "extractive" and r["extract"]["found"]
    assert c.get("/api/health").json()["ask_llm"]["configured"] is False


def test_book_outline_and_chapter_summary_follow_the_table_of_contents(tmp, convert, no_llm):
    pdf = PDF()

    def contents(p):
        text(p, 72, 80, "Contents", 20, "hebo")
        y = 130
        for e, pg in (("Preface", 1), ("1 Insertion sort", 2), ("1.1 Running time", 2), ("2 Merge sort", 3),
                      ("3 Loop invariants", 4), ("4 Heaps", 5), ("Index", 6)):
            text(p, 72, y, f"{e} {pg}", 11, "tiro"); y += 18
    pdf.page(contents)
    for k, (title, body) in enumerate(TOPICS):
        def pg(p, title=title, body=body, k=k):
            text(p, 72, 80, f"{k + 1} {title}", 18, "hebo")
            paragraph(p, 72, 120, 450, body + " " + LOREM)
        pdf.page(pg)
    doc = convert(pdf.save(tmp / "toc_book.pdf"))
    ix = assistant.DocIndex(doc)
    assert [c[0] for c in ix.chapters] == ["1", "2", "3", "4"] and ix.chapters[1][3] == 3
    s = assistant.respond(doc, "Summarize this document")
    assert "- 2 Merge sort *(p. 3)*" in s["answer"]
    c = assistant.respond(doc, "Summarise chapter 2")
    assert c["intent"] == "summary" and {x["page"] for x in c["citations"]} == {3}


# --------------------------------------------------------------------------------------------- v2.4
def _many_chunks_doc(extra_target):
    """A document of 40 short sections: 'priority queue' and 'heap' appear together in many of them; the target
    section speaks of a priority queue but never says 'heap'."""
    blocks = []
    filler = ["sorting arrays in place", "hashing keys into tables", "walking graphs breadth first", "dynamic programming tables",
              "greedy choices for schedules", "matrix multiplication cost", "string matching automata", "network flow augmenting paths"]
    k = 0
    for i in range(40):
        if i % 3 == 0:
            body = (f"A binary heap implements a priority queue. The heap keeps the largest key at the root, so the priority "
                    f"queue returns it first. Section {i} studies heap insertion and priority queue updates.")
        else:
            body = f"Section {i} is about {filler[i % len(filler)]}. " * 3
        if i == 37:
            body = extra_target
        k += 1
        blocks.append(dict(block_id=f"h{i}", type="heading", content=f"Topic {i}", page=i + 1, reading_order=0, level=2))
        blocks.append(dict(block_id=f"b{i}", type="paragraph", content=body, page=i + 1, reading_order=1))
    return dict(filename="t.pdf", pages=[{} for _ in range(40)], blocks=blocks)


def test_meaning_based_retrieval_finds_passages_without_the_question_words(no_llm):
    doc = _many_chunks_doc("Schedulers keep waiting jobs in a priority queue and always run the most urgent job next. "
                           "Updating a job's urgency moves it within the priority queue.")
    ix = assistant.DocIndex(doc)
    assert ix.lsa is not None
    sim = ix.semantic(assistant.ask.tokens("heap"))
    target = next(c["id"] for c in ix.chunks if "Schedulers" in c["text"])
    unrelated = [c["id"] for c in ix.chunks if "hashing keys" in c["text"]]
    assert all(sim[target] > sim[u] for u in unrelated)
    hits = [i for _, i in ix.search(assistant.ask.tokens("heap scheduling of urgent jobs"), k=5)]
    assert target in hits


def test_index_is_saved_and_reloaded(tmp, no_llm):
    import json as _json
    doc = _many_chunks_doc("A scheduler keeps jobs in a priority queue.")
    (tmp / "document.json").write_text(_json.dumps(doc))
    assert assistant.prepare(doc, str(tmp))
    assert (tmp / "ask_index.json").exists() and (tmp / "ask_lsa.npz").exists()
    mt = (tmp / "document.json").stat().st_mtime
    ix = assistant.DocIndex.load(str(tmp), doc, mt)
    fresh = assistant.DocIndex(doc)
    q = assistant.ask.tokens("priority queue scheduler")
    assert [i for _, i in ix.search(q, k=5)] == [i for _, i in fresh.search(q, k=5)]


def test_offline_example_answer(book, no_llm):
    import copy
    d = copy.deepcopy(book)
    b = next(b for b in d["blocks"] if "Merge sort divides" in (b.get("content") or ""))
    b["content"] += " For example, merge sort splits [5, 2, 4, 6] into [5, 2] and [4, 6], sorts both and merges them into [2, 4, 5, 6]."
    r = assistant.respond(d, "Give me an example of merge sort")
    assert r["intent"] == "example" and "[5, 2, 4, 6]" in r["answer"] and r["citations"][0]["page"] == 2


def test_book_summary_offline_has_one_line_per_chapter(tmp, convert, no_llm):
    pdf = PDF()

    def contents(p):
        text(p, 72, 80, "Contents", 20, "hebo")
        y = 130
        for e, pg in (("Preface", 1), ("1 Insertion sort", 2), ("1.1 Running time", 2), ("2 Merge sort", 3),
                      ("3 Loop invariants", 4), ("4 Heaps", 5), ("5 Graphs", 6), ("Index", 7)):
            text(p, 72, y, f"{e} {pg}", 11, "tiro"); y += 18
    pdf.page(contents)
    for k, (title, body) in enumerate(TOPICS + [("Graphs", "A graph has vertices and edges. Breadth-first search visits vertices in order of distance from the source.")]):
        pdf.page(lambda p, title=title, body=body, k=k: (text(p, 72, 80, f"{k + 1} {title}", 18, "hebo"), paragraph(p, 72, 120, 450, body + " " + LOREM)))
    doc = convert(pdf.save(tmp / "book5.pdf"))
    s = assistant.respond(doc, "Summarize this document")
    assert "**Chapters**" in s["answer"] and "**2 Merge sort** *(p. 3)* —" in s["answer"] and "breadth-first" in s["answer"].lower()


class _EmbStub(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _EmbStub.calls.append(self.path)
        texts = body["input"]
        vec = lambda t: [1.0 if "urgent" in t.lower() else 0.0, 1.0 if "heap" in t.lower() else 0.0, 0.1]
        out = json.dumps({"data": [{"index": k, "embedding": vec(t)} for k, t in enumerate(texts)]}).encode()
        self.send_response(200); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(out)))
        self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_optional_neural_embeddings(tmp, monkeypatch, no_llm):
    srv = HTTPServer(("127.0.0.1", 0), _EmbStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("ASK_EMBED_MODEL", "stub-embed")
        monkeypatch.setenv("ASK_EMBED_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
        doc = _many_chunks_doc("Schedulers run the most urgent job next.")
        ix = assistant.DocIndex(doc); ix.folder = str(tmp)
        hits = [i for _, i in ix.search(assistant.ask.tokens("urgent"), k=3)]
        target = next(c["id"] for c in ix.chunks if "Schedulers" in c["text"])
        assert target in hits and "/v1/embeddings" in _EmbStub.calls and (tmp / "ask_emb.npz").exists()
    finally:
        srv.shutdown()


def test_chat_is_saved_with_the_document(tmp, monkeypatch, no_llm):
    import importlib, sys, time
    monkeypatch.setenv("PARSEFLOW_DATA_DIR", str(tmp / "data"))
    monkeypatch.setenv("PARSEFLOW_ISOLATE", "0")
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    from fastapi.testclient import TestClient
    c = TestClient(app.app)
    pdf = PDF()
    for title, body in TOPICS[:2]:
        pdf.page(lambda p, title=title, body=body: (text(p, 72, 80, title, 18, "hebo"), paragraph(p, 72, 120, 450, body)))
    f = pdf.save(tmp / "two.pdf")
    did = c.post("/api/process", files={"file": ("two.pdf", f.read_bytes(), "application/pdf")}).json()["document_id"]
    for _ in range(200):
        if c.get(f"/api/status/{did}").json()["status"] != "running":
            break
        time.sleep(0.05)
    assert (app.DATA / did / "ask_index.json").exists()                 # indexed at processing time
    c.post(f"/api/ask/{did}", json=dict(question="What does merge sort do?", history=[]))
    m = c.get(f"/api/chat/{did}").json()["messages"]
    assert [x["role"] for x in m] == ["user", "assistant"] and m[0]["text"] == "What does merge sort do?" and m[1]["s"]["mode"] == "extractive"
    assert c.delete(f"/api/chat/{did}").json()["ok"] and c.get(f"/api/chat/{did}").json()["messages"] == []


def test_long_document_summary_is_map_reduce_over_real_text(book, stub_llm, monkeypatch):
    """Too long to send whole: each part is summarised from its own excerpts, then the parts are combined."""
    monkeypatch.setenv("ASK_CONTEXT_CHARS", "1500")
    r = assistant.respond(book, "Summarize this document", key="map-reduce-test")
    assert r["mode"] == "llm" and r["intent"] == "summary"
    systems = [req["body"]["messages"][0]["content"] for req in stub_llm]
    part_calls = [req for req, s in zip(stub_llm, systems) if s == assistant.SECTION_SYSTEM]
    assert len(part_calls) >= 3                                          # one per part of the book
    assert any("Merge sort divides" in req["body"]["messages"][-1]["content"] for req in part_calls)
    final = stub_llm[-1]["body"]["messages"][-1]["content"]
    assert systems[-1] == assistant.SYSTEM and "Summaries of each part" in final
    n = len(stub_llm)
    assistant.respond(book, "Summarize this document", key="map-reduce-test")   # part summaries are cached
    assert len(stub_llm) == n + 1


def test_comparison_retrieves_both_sides(book, stub_llm):
    r = assistant.respond(book, "Compare insertion sort and merge sort")
    assert r["intent"] == "analysis"
    sent = stub_llm[-1]["body"]["messages"][-1]["content"]
    assert "Insertion sort builds" in sent and "Merge sort divides" in sent


def test_answer_style_hints_reach_the_model(book, stub_llm):
    assistant.respond(book, "Explain heaps simply with an example")
    sent = stub_llm[-1]["body"]["messages"][-1]["content"]
    assert "How to answer:" in sent and "very simply" in sent and "worked example" in sent


def test_ai_selftest_and_gemini_edge_cases(monkeypatch, no_llm):
    calls = []

    def handler(req):
        calls.append(req)
        body = json.loads(req.content)
        has_img = any("inline_data" in p for p in body["contents"][-1]["parts"])
        if len(calls) == 3:          # a thinking model that spent its budget
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "…", "thought": True}]}, "finishReason": "MAX_TOKENS"}]})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "7" if has_img else "OK"}]}}]})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("ASK_PROVIDER", "gemini"); monkeypatch.setenv("ASK_API_KEY", "k")
    r = llm.selftest(client=client)
    assert r["ok"] and r["vision"] and r["vision_ok"] and r["provider"] == "gemini"
    assert "inline_data" in json.dumps(json.loads(calls[1].content))         # the image test really sends an image
    with pytest.raises(llm.LLMError) as e:
        llm.chat("s", [dict(role="user", content="x")], client=client)
    assert "MAX_TOKENS" in str(e.value) and "ASK_MAX_TOKENS" in str(e.value)
    bad = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(400, json={"error": {"message": "API key not valid"}})))
    r2 = llm.selftest(client=bad)
    assert r2["ok"] is False and "API key not valid" in r2["error"]
    monkeypatch.setenv("ASK_PROVIDER", "none")
    assert llm.selftest()["ok"] is False
