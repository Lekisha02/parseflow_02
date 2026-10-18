"""Very large documents: light/compressed delivery and per-page details, resumable processing (checkpoints),
automatic retry after a worker crash, per-worker memory cap, job queue, resume after a server restart."""
import gzip
import importlib
import json
import os
import re
import shutil
import sys
import time

import pytest

from conftest import LOREM, PDF, paragraph, text
from processing import docfiles, fontfix, pipeline, runner


def _book(path, n=12):
    pdf = PDF()
    for k in range(n):
        def pg(p, k=k):
            text(p, 72, 80, f"{k + 1} Chapter {k + 1}", 18, "hebo")
            y = paragraph(p, 72, 120, 450, f"Page {k + 1} explains topic {k + 1}. " + LOREM)
            for r in range(4):
                for c, v in enumerate((f"row{r}", str(10 * r + k), str(r * k))):
                    text(p, 72 + 140 * c, y + 30 + 16 * r, v, 10, "tiro")
        pdf.page(pg)
    return pdf.save(path)


def _strip(d):
    return re.sub(r'"seconds": [0-9.e-]+', "", json.dumps({k: v for k, v in d.items() if k != "timings"}, sort_keys=True))


def test_variants_lite_and_page_blocks(tmp, convert):
    out = tmp / "out"; out.mkdir()
    doc = pipeline.process(str(_book(tmp / "b.pdf", 4)), "b.pdf", "abcdefabcdef", str(out), pipeline.Progress(), save_images=False)
    docfiles.write_variants(str(out), doc)
    assert docfiles.fresh(str(out))
    lite = json.loads(gzip.decompress((out / "document.lite.json.gz").read_bytes()))
    assert lite["lite"] and lite["markdown"] is None and len(lite["blocks"]) == len(doc["blocks"])
    assert not any(k in b for b in lite["blocks"] for k in docfiles.HEAVY)
    full = docfiles.page_blocks(str(out), [2])
    assert full and all(b["page"] == 2 for b in full) and any("evidence_detail" in b for b in full)
    assert json.loads(gzip.decompress((out / "document.json.gz").read_bytes()))["blocks"] == json.loads((out / "document.json").read_text())["blocks"]
    assert (out / "document.md").read_text() == doc["markdown"]


def test_resume_after_a_crash_gives_the_same_document(tmp, monkeypatch):
    src = _book(tmp / "c.pdf", 12)
    monkeypatch.setenv("PARSEFLOW_PAGE_CHUNK", "3")
    a_dir, b_dir = tmp / "a", tmp / "b"
    a_dir.mkdir(); b_dir.mkdir()
    a = pipeline.process(str(src), "c.pdf", "abc", str(a_dir), pipeline.Progress(), save_images=False)
    orig = pipeline._extract_one

    def boom(ctx, *x, **k):
        if ctx["n"] == 8:
            raise KeyboardInterrupt("simulated crash")
        return orig(ctx, *x, **k)
    monkeypatch.setattr(pipeline, "_extract_one", boom)
    fontfix._CACHE.clear()
    with pytest.raises(KeyboardInterrupt):
        pipeline.process(str(src), "c.pdf", "abc", str(b_dir), pipeline.Progress(), save_images=False)
    assert sorted(f for f in os.listdir(b_dir / "checkpoint") if f.startswith("slice_")) == ["slice_000000.json", "slice_000003.json"]
    seen = []

    def count(ctx, *x, **k):
        seen.append(ctx["n"]); return orig(ctx, *x, **k)
    monkeypatch.setattr(pipeline, "_extract_one", count)
    fontfix._CACHE.clear()
    prog = pipeline.Progress()
    b = pipeline.process(str(src), "c.pdf", "abc", str(b_dir), prog, save_images=False)
    assert seen == list(range(7, 13))                       # pages 1-6 came from the checkpoint
    assert _strip(a) == _strip(b) and not (b_dir / "checkpoint").exists()


def test_worker_crash_is_retried_with_smaller_slices(tmp, monkeypatch):
    src = _book(tmp / "r.pdf", 10)
    out = tmp / "job"; out.mkdir()
    monkeypatch.setenv("PARSEFLOW_PAGE_CHUNK", "4")
    monkeypatch.setenv("PARSEFLOW_TEST_EXIT_AT_PAGE", "7")          # the worker process dies once, at page 7
    r = runner.run_isolated(str(src), "r.pdf", "abcdefabcdef", str(out), pipeline.Progress(), poll=0.05)
    assert r["status"] == "complete" and "retry" in " ".join(r.get("notes", [])) and (out / ".test_exited").exists()
    assert len(json.loads((out / "document.json").read_text())["pages"]) == 10


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS kernel does not enforce RLIMIT_DATA")
def test_memory_cap_stops_only_the_worker(tmp, monkeypatch):
    out = tmp / "m"; out.mkdir()
    monkeypatch.setenv("PARSEFLOW_WORKER_MEMORY_MB", "150")
    monkeypatch.setenv("PARSEFLOW_RETRIES", "0")
    r = runner.run_isolated(str(_book(tmp / "m.pdf", 3)), "m.pdf", "abcdefabcdea", str(out), pipeline.Progress(), poll=0.05)
    assert r["status"] == "failed" and r["error"]["code"] in ("PF-507", "PF-503")
    monkeypatch.setenv("PARSEFLOW_WORKER_MEMORY_MB", "auto")
    assert runner.memory_limit_bytes() >= 1536 * 1024 * 1024


def _app(tmp, monkeypatch, **env):
    monkeypatch.setenv("PARSEFLOW_DATA_DIR", str(tmp / "data"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def _wait(c, did, n=600):
    for _ in range(n):
        st = c.get(f"/api/status/{did}").json()
        if st.get("status") in ("complete", "failed"):
            return st
        time.sleep(0.05)
    return st


def test_api_light_document_blocks_and_queue(tmp, monkeypatch):
    app = _app(tmp, monkeypatch, PARSEFLOW_ISOLATE="0", PARSEFLOW_WORKERS="1", PARSEFLOW_LITE_MB="0")
    from fastapi.testclient import TestClient
    c = TestClient(app.app)
    files = [_book(tmp / f"q{i}.pdf", 6 if i else 25) for i in range(3)]
    ids = [c.post("/api/process", files={"file": (f.name, f.read_bytes(), "application/pdf")}).json()["document_id"] for f in files]
    st = c.get(f"/api/status/{ids[2]}").json()
    assert st["status"] in ("queued", "running", "complete")
    if st["status"] == "queued":
        assert st["queue_position"] >= 1
    for did in ids:
        assert _wait(c, did)["status"] == "complete"
    did = ids[1]
    r = c.get(f"/api/document/{did}?lite=auto", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200 and r.json()["lite"] is True                 # LITE_MB=0: every document is 'large'
    assert "evidence_detail" not in r.json()["blocks"][0]
    full = c.get(f"/api/document/{did}").json()
    assert "lite" not in full and "evidence_detail" in full["blocks"][0]
    pb = c.get(f"/api/document/{did}/blocks?page=2").json()
    assert pb["pages"] == [2] and pb["blocks"] and "raw_text" in pb["blocks"][0]
    assert c.get(f"/api/document/{did}/blocks?pages=1-3").json()["pages"] == [1, 2, 3]
    assert c.get(f"/api/document/{did}/markdown").text == full["markdown"]


def test_interrupted_documents_resume_after_restart(tmp, monkeypatch):
    data = tmp / "data"; data.mkdir()
    did = "0123456789ab"
    (data / did).mkdir()
    shutil.copy(_book(tmp / "x.pdf", 5), data / did / f"source_{did}_x.pdf")
    (data / "fedcba987654").mkdir()                                       # a final failure is not retried
    shutil.copy(tmp / "x.pdf", data / "fedcba987654" / "source_fedcba987654_x.pdf")
    (data / "fedcba987654" / "failed.json").write_text("{}")
    app = _app(tmp, monkeypatch, PARSEFLOW_ISOLATE="0")
    assert app.RESUMED == [did]
    from fastapi.testclient import TestClient
    c = TestClient(app.app)
    assert _wait(c, did)["status"] == "complete" and (data / did / "document.json").exists()
