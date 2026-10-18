"""Batch execution shared by the CLI and the benchmark harness.

Each document runs in its own worker process so that a malformed PDF, a native-library crash or a timeout
affects only that file. Workers are started with the 'spawn' method (no inherited state, deterministic),
at most `workers` run at once, and documents are streamed one by one (never all loaded in memory).
"""
import multiprocessing as mp
import os
import queue
import tempfile
import time
import traceback


def convert(path, include_furniture=False, save_images=False, out_dir=None):
    """Convert one document in-process. Returns the Digital Twin dict (with doc['markdown'])."""
    from . import pipeline
    name = os.path.basename(path)
    prog = pipeline.Progress()
    tmp = None
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="parseflow_")
        out_dir = tmp.name
    try:
        doc = pipeline.process(path, name, "cli", out_dir, prog, save_images=save_images, include_furniture=include_furniture)
        return doc
    finally:
        if tmp is not None:
            tmp.cleanup()


def _worker(path, opts, q):
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")       # one Tesseract thread per worker process
    try:
        from . import markdown as md, pipeline  # noqa: F401  (imports are not part of the measured conversion time)
    except BaseException as e:
        q.put(dict(ok=False, seconds=0.0, error=f"import failed: {type(e).__name__}: {e}"))
        return
    t0 = time.perf_counter()
    try:
        doc = convert(path, include_furniture=opts.get("include_furniture", False))
        try:
            import resource
            scale = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
        except Exception:
            rss = None
        res = dict(ok=True, seconds=time.perf_counter() - t0, pages=len(doc["pages"]), max_rss_mb=rss,
                   page_seconds=[p.get("seconds") for p in doc["pages"]], timings=doc.get("timings"),
                   routes=[p.get("route") for p in doc["pages"]], warnings=doc.get("warnings", []))
        if opts.get("per_page_markdown"):
            res["page_markdown"] = {p["n"]: md.render(dict(doc, blocks=[b for b in doc["blocks"] if b["page"] == p["n"]], warnings=[], filename=None),
                                                      include_furniture=opts.get("include_furniture", False), annotate=opts.get("annotate", True))
                                    for p in doc["pages"]}
        res["markdown"] = md.render(doc, include_furniture=opts.get("include_furniture", False), annotate=opts.get("annotate", True))
        if opts.get("want_json"):
            res["doc"] = doc
        q.put(res)
    except BaseException as e:   # noqa: report every failure, including SystemExit from native code
        code = getattr(e, "code", None)
        q.put(dict(ok=False, seconds=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}", code=code if isinstance(code, str) else None,
                   trace=traceback.format_exc(limit=4)))


def run_batch(paths, opts=None, workers=None, timeout=300, on_result=None):
    """Run documents in isolated processes. Yields (path, result) as each finishes; results are dicts with ok/error."""
    opts = opts or {}
    workers = max(1, workers or os.cpu_count() or 1)
    ctx = mp.get_context("spawn")
    pending = list(paths)
    running = {}       # path -> (proc, queue, start)
    results = []
    while pending or running:
        while pending and len(running) < workers:
            p = pending.pop(0)
            q = ctx.Queue()
            pr = ctx.Process(target=_worker, args=(p, opts, q), daemon=True)
            pr.start()
            running[p] = (pr, q, time.perf_counter())
        done = []
        for p, (pr, q, t0) in running.items():
            res = None
            try:
                res = q.get_nowait()
            except queue.Empty:
                pass
            if res is not None:
                pr.join(5)
                done.append((p, res))
            elif not pr.is_alive():
                done.append((p, dict(ok=False, seconds=time.perf_counter() - t0, error=f"worker exited with code {pr.exitcode} (crash)")))
            elif time.perf_counter() - t0 > timeout:
                pr.terminate(); pr.join(5)
                if pr.is_alive():
                    pr.kill()
                done.append((p, dict(ok=False, seconds=time.perf_counter() - t0, error=f"timeout after {timeout}s", timeout=True)))
        for p, res in done:
            running.pop(p, None)
            results.append((p, res))
            if on_result:
                on_result(p, res)
        if not done:
            time.sleep(0.02)
    return results


# ---------------------------------------------------------------------------------------------
# Web app: one isolated process per uploaded document
# ---------------------------------------------------------------------------------------------
def _write_json(path, obj):
    import json
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True)
    os.replace(tmp, path)


def memory_limit_bytes():
    """Per-worker memory cap. PARSEFLOW_WORKER_MEMORY_MB: a number, 0/off for none, or auto (default): 80% of the
    memory available to this machine or container, shared by PARSEFLOW_WORKERS workers, at least 1.5 GB."""
    v = (os.getenv("PARSEFLOW_WORKER_MEMORY_MB", "auto") or "auto").strip().lower()
    if v in ("0", "off", "none", "no"):
        return 0
    if v != "auto":
        try:
            return int(float(v) * 1024 * 1024)
        except ValueError:
            return 0
    total = 0
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            c = open(f).read().strip()
            if c.isdigit() and 0 < int(c) < (total or 1 << 62):
                total = int(c)
        except OSError:
            pass
    if not total:
        return 0
    workers = max(1, int(os.getenv("PARSEFLOW_WORKERS", "2") or 2))
    return max(1536 * 1024 * 1024, int(0.8 * total / workers))


def _near_memory_limit():
    lim = memory_limit_bytes()
    if not lim:
        return False
    try:
        import resource
        scale = 1 if sys.platform == "darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale > 0.7 * lim
    except Exception:
        return False


def _apply_memory_limit():
    lim = memory_limit_bytes()
    if not lim:
        return
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_DATA, (lim, lim))     # heap + private mappings; a runaway page fails
    except (ImportError, ValueError, OSError):                      # with MemoryError instead of exhausting the host
        pass


def _job_child(src, name, doc_id, out_dir, env=None):
    """Child process: process one document, publishing progress to out_dir/progress.json."""
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    if env:
        os.environ.update(env)
    _apply_memory_limit()
    ppath = os.path.join(out_dir, "progress.json")
    try:
        from . import pipeline
    except BaseException as e:
        _write_json(ppath, dict(status="failed", stages=[], error=dict(code="PF-500", title="Processing Error", message=f"Worker could not start ({type(e).__name__}).")))
        return

    class FileProgress(pipeline.Progress):
        def __init__(self):
            super().__init__(); self._last = 0.0

        def set(self, name, state, detail=""):
            super().set(name, state, detail)
            if _near_memory_limit():
                raise MemoryError("worker exceeded memory limit")
            now = time.time()
            if now - self._last > 0.4 or state == "done":
                self._last = now
                try:
                    _write_json(ppath, dict(status="running", stages=self.stages, timings=self.timings))
                except Exception:
                    pass
    prog = FileProgress()
    try:
        doc = pipeline.process(src, name, doc_id, out_dir, prog, save_images="needed")
        from . import assistant
        assistant.prepare(doc, out_dir)                 # the Ask index is built once, here, not on the first question
        try:
            from . import docfiles
            docfiles.write_variants(out_dir, doc)       # compressed / lite / per-page files for the viewer
        except Exception:
            pass
        stats = None
        try:
            import accounts
            stats = accounts.doc_stats(doc)
        except Exception:
            try:
                import sys as _s
                _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                import accounts
                stats = accounts.doc_stats(doc)
            except Exception:
                stats = None
        _write_json(ppath, dict(status="complete", stages=prog.stages, timings=prog.timings, stats=stats))
    except pipeline.PFError as e:
        _write_json(ppath, dict(status="failed", stages=prog.stages, error=e.dict()))
    except BaseException as e:
        if isinstance(e, MemoryError) or _near_memory_limit():
            # libraries report a failed allocation in their own words (MuPDF: 'zlib error', 'malloc ...')
            _write_json(ppath, dict(status="failed", stages=prog.stages, error=dict(
                code="PF-507", title="Not Enough Memory", message="The document needed more memory than one worker may "
                "use (PARSEFLOW_WORKER_MEMORY_MB). The server is still running.")))
            return
        _write_json(ppath, dict(status="failed", stages=prog.stages,
                                error=dict(code="PF-500", title="Processing Error", message=f"Unexpected failure ({type(e).__name__}); the server is still running.")))


def run_isolated(src, name, doc_id, out_dir, prog, timeout=None, poll=0.5, retries=None):
    """Run _job_child in a spawned process and mirror its progress into `prog` (a pipeline.Progress). A crash or an
    out-of-memory stop of the worker fails only this document. When the worker dies or runs out of memory, the job
    is retried (PARSEFLOW_RETRIES, default 1) with smaller page slices, resuming from the last saved slice.
    Returns the final progress dict."""
    retries = int(os.getenv("PARSEFLOW_RETRIES", "1")) if retries is None else retries
    t0 = time.time()
    chunk = max(1, int(os.getenv("PARSEFLOW_PAGE_CHUNK", "25") or 25))
    env = {}
    for attempt in range(retries + 1):
        left = (timeout - (time.time() - t0)) if timeout else None
        r = _run_once(src, name, doc_id, out_dir, prog, left if left is None or left > 0 else 0.01, poll, env)
        code = (r.get("error") or {}).get("code")
        if r.get("status") == "complete" or code not in ("PF-503", "PF-507") or attempt == retries:
            if attempt and r.get("status") == "complete":
                r.setdefault("notes", []).append(f"completed after {attempt} automatic retr{'y' if attempt == 1 else 'ies'} with smaller page slices")
            return r
        chunk = max(1, chunk // 4)
        env = dict(PARSEFLOW_PAGE_CHUNK=str(chunk), PARSEFLOW_OCR_WORKERS="1")
        try:
            prog.set("EXTRACT", "running", f"worker stopped ({code}); resuming with {chunk}-page slices")
        except Exception:
            pass
    return r


def _run_once(src, name, doc_id, out_dir, prog, timeout, poll, env):
    import json
    ppath = os.path.join(out_dir, "progress.json")
    try:
        os.unlink(ppath)                 # a previous attempt's final status must not end this one
    except OSError:
        pass
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_job_child, args=(src, name, doc_id, out_dir, env), daemon=True)
    p.start()
    t0 = time.time()
    last = None
    while True:
        alive = p.is_alive()
        try:
            with open(ppath, encoding="utf-8") as f:
                last = json.load(f)
            prog.stages = last.get("stages") or prog.stages
            prog.timings = last.get("timings") or getattr(prog, "timings", {})
        except Exception:
            pass
        if last and last.get("status") in ("complete", "failed"):
            p.join(5)
            return last
        if not alive:
            p.join(1)
            code = p.exitcode
            return dict(status="failed", stages=prog.stages, error=dict(
                code="PF-503", title="Processing Stopped",
                message=f"The processing worker stopped unexpectedly (exit code {code}; often not enough memory). The server is still running."))
        if timeout and time.time() - t0 > timeout:
            p.terminate(); p.join(5)
            return dict(status="failed", stages=prog.stages, error=dict(code="PF-504", title="Processing Timeout", message=f"Processing took longer than {int(timeout)} s and was stopped."))
        time.sleep(poll)
