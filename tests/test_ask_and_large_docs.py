"""Ask Your Document (extractive answers with renderable math, code and tables), the /api/ask endpoint, on-demand page
images and thumbnails, slice-wise processing of long documents, and running headers/footers in long documents."""
import importlib
import json
import sys
import time

import pymupdf
import pytest

from conftest import LOREM, PDF, ROOT, paragraph, text, unicode_text
from processing import ask, pipeline

AVL_L = ["If (root == NULL)", "    return root", "If (key < root -> data)", "    root -> left = AVL_Delete(root -> left, key)",
         "Else if (key > root -> data)", "    root -> right = AVL_Delete(root -> right, key)", "Else // Node found",
         "    // Case 1 : Node with no child", "    If (root -> left == NULL AND root -> right == NULL)", "        return NULL",
         "    // Case 2 : Node with one child", "    Else if (root -> left == NULL)", "        return root -> right",
         "    Else if (root -> right == NULL)", "        return root -> left", "    // Case 3 : Node with two children", "    Else",
         "        successor = Minimum(root -> right)", "        root -> data = successor -> data",
         "        root -> right = AVL_Delete(root -> right, successor -> data)", "If (root == NULL)", "    return root",
         "height(root) = 1 + max(height(root -> left), height(root -> right))", "balance = height(root -> left) - height(root -> right)"]
AVL_R = ["// Left Heavy Case", "If (balance > 1)", "    If (height(root -> left -> left) >= height(root -> left -> right))",
         "        return RightRotate(root)   // L1 or L0", "    Else", "        root -> left = LeftRotate(root -> left)",
         "        return RightRotate(root)   // L-1", "// Right Heavy Case", "If (balance < -1)",
         "    If (height(root -> right -> right) >= height(root -> right -> left))", "        return LeftRotate(root)   // R-1 or R0",
         "    Else", "        root -> right = RightRotate(root -> right)", "        return LeftRotate(root)   // R1", "return root"]


def _avl_slide(p):
    p.insert_text((300, 50), "Algorithm for AVL deletion", fontsize=26, fontname="hebo")
    for lines, x, start in ((AVL_L, 30, 1), (AVL_R, 500, 25)):
        y = 90
        for i, t in enumerate(lines):
            ind = (len(t) - len(t.lstrip())) / 4 * 18
            p.insert_text((x, y), f"{start + i}.", fontsize=10.5, fontname="helv")
            p.insert_text((x + 26 + ind, y), t.strip(), fontsize=10.5, fontname="helv")
            y += 18


def _norm_slide(p):
    text(p, 180, 30, "Normalization of Wavefunction", 20, "hebo")
    unicode_text(p, 20, 64, "•  Total probability P = ∫ |ψ(r,t)|² dv = 1", 15)
    unicode_text(p, 20, 100, "•  ψ(r,t) satisfying the above condition is called normalized.", 15)
    text(p, 20, 136, "•  If the wave function is not normalized, it must be multiplied by a constant", 15, "tiro")
    text(p, 36, 160, "factor called normalization factor.", 15, "tiro")
    unicode_text(p, 20, 330, "•  ∫ Ψ*Ψ dv = 1,     1/√N is called normalization factor.", 15)


@pytest.fixture
def slides(tmp, convert):
    avl = PDF(960, 540).page(_avl_slide).save(tmp / "avl.pdf")
    norm = PDF(692, 390).page(_norm_slide).save(tmp / "norm.pdf")
    return convert(avl), convert(norm)


def test_numbered_two_column_pseudocode_is_code_not_a_table(slides):
    avl, _ = slides
    types = [b["type"] for b in avl["blocks"]]
    assert "table" not in types and types.count("code") == 2
    code = [b for b in avl["blocks"] if b["type"] == "code"]
    assert code[0]["content"].startswith("1.") and "24. balance = height(root -> left) - height(root -> right)" in code[0]["content"]
    assert code[1]["content"].startswith("25. // Left Heavy Case")
    assert "\\(" not in code[0]["content"]           # code is never turned into inline math


def test_ask_section_and_code_excerpts(slides):
    avl, _ = slides
    r = ask.answer(avl, "algorithm for AVL deletion")
    assert r["found"] and r["strategy"] == "section" and [p["kind"] for p in r["parts"]] == ["heading", "code", "code"]
    r = ask.answer(avl, "what happens in the right heavy case")
    assert r["parts"][0]["kind"] == "code" and r["parts"][0]["text"].startswith("32. // Right Heavy Case")
    assert r["parts"][0]["text"].rstrip().endswith("39. return root")
    r = ask.answer(avl, "what is a node with two children")
    ex = r["parts"][0]["text"]
    assert ex.startswith("16.") and "successor" in ex and "21." not in ex       # the step ends where the indentation does
    r = ask.answer(avl, "how is balance computed")
    assert "balance = height(root -> left) - height(root -> right)" in r["parts"][0]["text"]
    for e in r["evidence"]:
        assert e["block_id"] and e["page"] == 1


def test_ask_answers_keep_math_renderable(slides):
    _, norm = slides
    r = ask.answer(norm, "what is normalization factor")
    assert r["found"] and r["confidence"] is not None
    first = r["parts"][0]["text"]
    assert "normalization factor" in first and "\\(" in first and "\\sqrt" in first      # LaTeX kept with delimiters for KaTeX
    assert len(r["evidence"]) >= 2 and all("snippet" in e for e in r["evidence"])
    other = [p["text"] for p in r["parts"][1:] if p["kind"] == "text"]
    assert any(t.startswith("If the wave function is not normalized") and "\n" not in t for t in other)   # wrapped lines joined
    r = ask.answer(norm, "what is total probability")
    assert "\\int" in r["parts"][0]["text"] and r["evidence"][0]["snippet"].count("\\(") == r["evidence"][0]["snippet"].count("\\)")


def test_ask_not_found_and_page_questions(slides):
    avl, norm = slides
    r = ask.answer(norm, "what is the capital of peru")
    assert not r["found"] and r["note"]
    r = ask.answer(avl, "What does page 1 explain?")
    assert r["found"] and r["strategy"] == "page" and r["parts"][0]["text"] == "Algorithm for AVL deletion"
    r = ask.answer(avl, "What does page 7 explain?")
    assert not r["found"] and "1 page" in r["note"]


def test_ask_term_and_description_rows(tmp, convert):
    def pg(p):
        text(p, 72, 60, "Error-Handling Commands", 16, "hebo")
        rows = [("Try...EndTry", "Defines a program block that lets the program execute a command."),
                ("ClrErr", "Clears the error status and sets the error number to zero."),
                ("PassErr", "Passes an error to the next level of the Try...EndTry block.")]
        y = 110
        for a, b in rows:
            text(p, 72, y, a, 10); paragraph(p, 220, y, 320, b, 10); y += 48
    d = convert(PDF().page(pg).save(tmp / "cmds.pdf"))
    r = ask.answer(d, "what does ClrErr do")
    t = r["parts"][0]["text"]
    assert "ClrErr" in t and "Clears the error status" in t and "PassErr" not in t


def test_stemming_and_list_items():
    assert ask.stem("debugging") == ask.stem("debug") == "debug"
    assert ask.stem("running") == "run" and ask.stem("called") == "call" and ask.stem("passes") == "pass"
    items, ordered = ask.list_items("• The agent first talks to one, who\nspeaks an utterance.\n• It then explores\nthe village.")
    assert items == ["The agent first talks to one, who speaks an utterance.", "It then explores the village."] and not ordered


# ------------------------------------------------------------------------------------------------ API
@pytest.fixture
def client(tmp, monkeypatch):
    monkeypatch.setenv("PARSEFLOW_DATA_DIR", str(tmp / "data"))
    sys.modules.pop("app", None)
    sys.path.insert(0, str(ROOT / "backend"))
    app = importlib.import_module("app")
    from fastapi.testclient import TestClient
    return TestClient(app.app), app


def _process(c, path):
    r = c.post("/api/process", files={"file": (path.name, path.read_bytes(), "application/pdf")}).json()
    for _ in range(400):
        st = c.get(f"/api/status/{r['document_id']}").json()
        if st["status"] != "running":
            break
        time.sleep(0.05)
    assert st["status"] == "complete"
    return r["document_id"]


def test_api_ask_and_on_demand_page_images(tmp, client):
    c, app = client
    f = PDF(692, 390).page(_norm_slide).page(_norm_slide).save(tmp / "norm2.pdf")
    did = _process(c, f)
    d = app.DATA / did
    assert not list(d.glob("page_*.png"))                  # digital pages are not rasterised up front
    j = c.get(f"/api/ask/{did}", params={"q": "what is normalization factor"}).json()
    assert j["found"] and j["evidence"] and "normalization factor" in j["parts"][0]["text"]
    assert c.get(f"/api/ask/{did}", params={"q": "zebra quantum lasagne"}).json()["found"] is False
    assert c.get("/api/ask/nothere12345", params={"q": "x"}).status_code == 404
    r = c.get(f"/api/page/{did}/2.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and (d / "page_2.png").exists()
    t = c.get(f"/api/thumb/{did}/1.jpg")
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg" and len(t.content) < len(r.content)
    assert c.get(f"/api/page/{did}/3.png").status_code == 404 and c.get(f"/api/thumb/{did}/0.jpg").status_code == 404
    h = c.get("/api/health").json()
    assert h["max_pages"] in (0, None) or h["max_pages"] >= 400      # 0 = no limit
    assert h["max_upload_mb"] == 0 or h["max_upload_mb"] >= 200


# ------------------------------------------------------------------------------------------------ long documents
def _book(tmp, n=24):
    """A small 'book': chapter running footers whose page number swaps sides on even/odd pages, and a different
    heading-sized page title at the top of every page."""
    topics = ("Variables Loops Strings Menus Graphs Tables Lists Matrices Functions Errors Debugging Input Output Files Memory "
              "Links Programs Editors Folders Units Statistics Sequences Vectors Polar Parametric Complex Calculus Algebra").split()
    pdf = PDF()
    for i in range(n):
        num = 101 + i
        def pg(p, i=i, num=num):
            text(p, 72, 50, f"Working with {topics[i % len(topics)]}" + (" in Depth" if i >= len(topics) else ""), 14, "hebo")
            paragraph(p, 72, 100, 450, f"Body {i + 1}. " + LOREM * 3)
            if num % 2 == 0:
                text(p, 72, 760, f"{num} Chapter 3: Programming", 8)
            else:
                text(p, 400, 760, f"Chapter 3: Programming {num}", 8)
        pdf.page(pg)
    return pdf.save(tmp / "book.pdf")


def test_slices_give_identical_output_with_bounded_memory(tmp, monkeypatch):
    f = _book(tmp, 30)
    outs = []
    for chunk in ("1000", "7"):
        monkeypatch.setenv("PARSEFLOW_PAGE_CHUNK", chunk)
        prog = pipeline.Progress()
        d = pipeline.process(str(f), "book.pdf", "x", str(tmp / ("o" + chunk)), prog, save_images=False)
        assert [s["state"] for s in prog.stages] == ["done"] * 7
        for p in d["pages"]:
            p.pop("seconds")
        d.pop("timings")
        outs.append(d)
    assert outs[0] == outs[1]


def test_progress_reports_pages(tmp, monkeypatch):
    f = _book(tmp, 6)
    seen = []

    class P(pipeline.Progress):
        def set(self, name, state, detail=""):
            seen.append((name, state, detail)); super().set(name, state, detail)
    pipeline.process(str(f), "book.pdf", "x", str(tmp / "o"), P(), save_images=False)
    assert ("EXTRACT", "running", "page 6 of 6") in seen and ("ANALYZE", "running", "page 3 of 6") in seen


def test_long_document_running_footers_removed_page_titles_kept(tmp, convert):
    d = convert(_book(tmp, 24))
    footers = [b for b in d["blocks"] if "Chapter 3: Programming" in (b.get("content") or "")]
    assert len(footers) == 24 and all((b.get("furniture") or {}).get("excluded") for b in footers)
    titles = [b for b in d["blocks"] if (b.get("content") or "").startswith("Working with")]
    assert len(titles) == 24 and not any((b.get("furniture") or {}).get("excluded") for b in titles)
    assert "Chapter 3: Programming" not in d["markdown"] and "Working with Programs" in d["markdown"]
