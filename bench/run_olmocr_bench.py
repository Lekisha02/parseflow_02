#!/usr/bin/env python3
"""Reproducible olmOCR-Bench run for ParseFlow, scored ONLY by the official scorer (olmocr.bench.benchmark).

What it does
  1. reads the official layout: <bench-dir>/*.jsonl + <bench-dir>/pdfs/**.pdf
  2. optional fixed regression subset: --sample-per-category N --seed S picks N PDFs per .jsonl file by a
     deterministic hash ordering and writes the subset (symlinked PDFs + filtered .jsonl) plus a manifest
  3. converts every PDF (current ParseFlow, or the original v1 implementation with --impl legacy) in isolated
     worker processes with a per-document timeout, writing <candidate>/<pdf path>_pg{page}_repeat1.md
     (a failed/timed-out document gets an EMPTY .md, so its tests fail - nothing is skipped)
  4. runs:  python -m olmocr.bench.benchmark --dir <dir> --candidate <candidate>   (official CLI, unmodified)
  5. writes results JSON + a Markdown summary: official overall / per-jsonl / per-test-type scores (parsed from the
     scorer's own output, raw output saved alongside), per-page latency (median, p90, p95, max), throughput,
     errors, timeouts, hardware and configuration.

Examples
  # full benchmark (download allenai/olmOCR-bench from Hugging Face first; see README)
  python bench/run_olmocr_bench.py --bench-dir ~/olmOCR-bench/bench_data --results bench/results/full.json
  # fixed regression sample, 25 PDFs per category
  python bench/run_olmocr_bench.py --bench-dir ~/olmOCR-bench/bench_data --sample-per-category 25 --seed 0
  # bundled official sample data (ships inside the olmocr repo) - used for CI
  python bench/run_olmocr_bench.py --bench-dir $OLMOCR/olmocr/bench/sample_data
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))


def hardware():
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    mem = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem = round(int(line.split()[1]) / 1024 / 1024, 1); break
    except OSError:
        pass
    try:
        import pymupdf
        pv = pymupdf.VersionBind
    except Exception:
        pv = None
    try:
        from processing import ocr
        tv = ocr.available()
    except Exception:
        tv = None
    return dict(cpu=cpu, logical_cores=os.cpu_count(), ram_gb=mem, os=platform.platform(), python=platform.python_version(), pymupdf=pv, tesseract=tv)


def load_tests(bench_dir):
    tests = {}
    for jf in sorted(glob.glob(os.path.join(bench_dir, "*.jsonl"))):
        for line in open(jf, encoding="utf-8"):
            if line.strip():
                t = json.loads(line)
                tests.setdefault(os.path.basename(jf), []).append(t)
    return tests


def make_subset(bench_dir, out_dir, n, seed):
    tests = load_tests(bench_dir)
    os.makedirs(os.path.join(out_dir, "pdfs"), exist_ok=True)
    manifest = {}
    for jf, ts in tests.items():
        pdfs = sorted({t["pdf"] for t in ts}, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())[:n]
        keep = set(pdfs)
        manifest[jf] = sorted(pdfs)
        with open(os.path.join(out_dir, jf), "w", encoding="utf-8") as f:
            for t in ts:
                if t["pdf"] in keep:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
        for p in pdfs:
            src = os.path.join(bench_dir, "pdfs", p)
            dst = os.path.join(out_dir, "pdfs", p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(dict(source=os.path.abspath(bench_dir), per_category=n, seed=seed, pdfs=manifest), f, indent=1)
    return out_dir


def pages_needed(bench_dir):
    need = {}
    for ts in load_tests(bench_dir).values():
        for t in ts:
            need.setdefault(t["pdf"], set()).add(int(t.get("page", 1)))
    # every PDF in pdfs/ gets a baseline test from the scorer (page 1) unless skipped
    for p in glob.glob(os.path.join(bench_dir, "pdfs", "**", "*.pdf"), recursive=True):
        rel = os.path.relpath(p, os.path.join(bench_dir, "pdfs"))
        need.setdefault(rel, set()).add(1)
    return need


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * q
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run_current(bench_dir, cand_dir, need, workers, timeout, annotate):
    from processing.runner import run_batch
    jobs = {os.path.join(bench_dir, "pdfs", p): (p, pages) for p, pages in need.items()}
    rec = {}

    def on_result(path, res):
        rel, pages = jobs[path]
        base = os.path.join(cand_dir, rel[:-4])
        os.makedirs(os.path.dirname(base), exist_ok=True)
        pm = res.get("page_markdown") or {}
        for pg in pages:
            with open(f"{base}_pg{pg}_repeat1.md", "w", encoding="utf-8") as f:
                f.write(pm.get(pg, pm.get(str(pg), "")) if res.get("ok") else "")
        rec[rel] = dict(ok=res.get("ok"), seconds=res.get("seconds"), pages=res.get("pages"), error=res.get("error"), timeout=res.get("timeout", False),
                        routes=res.get("routes"), page_seconds=res.get("page_seconds"), max_rss_mb=res.get("max_rss_mb"))
        sys.stderr.write(f"{'ok ' if res.get('ok') else 'ERR'} {res.get('seconds', 0):6.2f}s {rel}\n")

    t0 = time.perf_counter()
    run_batch(list(jobs), dict(per_page_markdown=True, annotate=annotate), workers=workers, timeout=timeout, on_result=on_result)
    return rec, time.perf_counter() - t0


def run_legacy(bench_dir, cand_dir, need, timeout, legacy_backend, variant, workers):
    from concurrent.futures import ThreadPoolExecutor
    rec = {}

    def one(item):
        rel, pages = item
        base = os.path.join(cand_dir, rel[:-4])
        os.makedirs(os.path.dirname(base), exist_ok=True)
        cmd = [sys.executable, os.path.join(ROOT, "bench", "legacy_runner.py"), "--backend", legacy_backend, "--pdf", os.path.join(bench_dir, "pdfs", rel),
               "--out", base, "--variant", variant, "--pages", *map(str, sorted(pages))]
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=dict(os.environ, OMP_THREAD_LIMIT="1"))
            info = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
            ok = info is not None
            res = dict(ok=ok, seconds=info["seconds"] if ok else time.perf_counter() - t0, pages=info["pages"] if ok else None,
                       error=None if ok else (r.stderr.strip().splitlines() or ["failed"])[-1][:300])
        except subprocess.TimeoutExpired:
            res = dict(ok=False, seconds=timeout, error=f"timeout after {timeout}s", timeout=True)
        if not res["ok"]:
            for pg in pages:
                open(f"{base}_pg{pg}_repeat1.md", "w").close()
        rec[rel] = res
        sys.stderr.write(f"{'ok ' if res['ok'] else 'ERR'} {res['seconds']:6.2f}s {rel}\n")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, sorted(need.items())))
    return rec, time.perf_counter() - t0


def run_scorer(bench_dir, candidate, olmocr_path, extra):
    env = dict(os.environ)
    if olmocr_path:
        env["PYTHONPATH"] = olmocr_path + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("RAPIDFUZZ_IMPLEMENTATION", env.get("RAPIDFUZZ_IMPLEMENTATION", ""))
    if not env["RAPIDFUZZ_IMPLEMENTATION"]:
        env.pop("RAPIDFUZZ_IMPLEMENTATION")
    cmd = [sys.executable, "-m", "olmocr.bench.benchmark", "--dir", bench_dir, "--candidate", candidate] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return cmd, r.returncode, r.stdout + ("\n[stderr]\n" + r.stderr[-4000:] if r.returncode else "")


def parse_scorer(out, candidate):
    res = dict(overall=None, ci=None, per_jsonl={}, per_test_type={})
    m = re.search(rf"^{re.escape(candidate)}\s*:\s*Average Score:\s*([\d.]+)%\s*±\s*([\d.]+)%", out, re.M)
    if m:
        res["overall"], res["ci"] = float(m.group(1)), float(m.group(2))
    tail = out.split("Final Summary", 1)[-1]
    for m in re.finditer(r"^\s{4}(\w+)\s*:\s*([\d.]+)% average pass rate over (\d+) tests", tail, re.M):
        res["per_test_type"][m.group(1)] = dict(score=float(m.group(2)), tests=int(m.group(3)))
    for m in re.finditer(r"^\s{8}(\S+)\s*:\s*([\d.]+)% \((\d+)/(\d+) tests\)", tail, re.M):
        res["per_jsonl"][m.group(1)] = dict(score=float(m.group(2)), passed=int(m.group(3)), tests=int(m.group(4)))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", required=True, help="official olmOCR-Bench layout (jsonl files + pdfs/)")
    ap.add_argument("--candidate", default=None, help="candidate folder name (default parseflow / parseflow_legacy)")
    ap.add_argument("--impl", choices=["current", "legacy"], default="current")
    ap.add_argument("--legacy-backend", help="path to the v1 backend/ directory (for --impl legacy)")
    ap.add_argument("--legacy-md", choices=["frontend", "plain"], default="frontend")
    ap.add_argument("--work-dir", default=None, help="where subsets/candidate outputs go (default: bench-dir for full runs)")
    ap.add_argument("--sample-per-category", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--timeout", type=int, default=180, help="per-document timeout in seconds")
    ap.add_argument("--annotated", action="store_true", help="keep HTML-comment warnings in the Markdown (default: clean)")
    ap.add_argument("--olmocr", default=os.getenv("OLMOCR_PATH"), help="path to an olmocr checkout if the package is not installed")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--scorer-args", default="", help="extra args passed verbatim to olmocr.bench.benchmark")
    ap.add_argument("--results", default=None, help="results JSON path (default bench/results/<name>.json)")
    a = ap.parse_args()

    bench_dir = os.path.abspath(a.bench_dir)
    cand = a.candidate or ("parseflow" if a.impl == "current" else f"parseflow_legacy_{a.legacy_md}")
    if a.sample_per_category:
        work = os.path.abspath(a.work_dir or os.path.join(ROOT, "bench", "work"))
        bench_dir = make_subset(bench_dir, os.path.join(work, f"subset_n{a.sample_per_category}_s{a.seed}"), a.sample_per_category, a.seed)
    elif a.work_dir:
        work = os.path.abspath(a.work_dir)
        os.makedirs(work, exist_ok=True)
        for jf in glob.glob(os.path.join(bench_dir, "*.jsonl")):
            shutil.copy(jf, work)
        if not os.path.exists(os.path.join(work, "pdfs")):
            os.symlink(os.path.join(bench_dir, "pdfs"), os.path.join(work, "pdfs"))
        bench_dir = work
    cand_dir = os.path.join(bench_dir, cand)
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir)
    need = pages_needed(bench_dir)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    if a.impl == "current":
        rec, wall = run_current(bench_dir, cand_dir, need, a.workers, a.timeout, a.annotated)
    else:
        if not a.legacy_backend:
            ap.error("--legacy-backend is required for --impl legacy")
        rec, wall = run_legacy(bench_dir, cand_dir, need, a.timeout, os.path.abspath(a.legacy_backend), a.legacy_md, a.workers)

    per_page = [r["seconds"] / max(1, r.get("pages") or 1) for r in rec.values() if r.get("ok")]
    total_pages = sum((r.get("pages") or 0) for r in rec.values() if r.get("ok"))
    latency = dict(per_page_median=pct(per_page, .5), per_page_p90=pct(per_page, .9), per_page_p95=pct(per_page, .95),
                   per_page_max=max(per_page) if per_page else None, per_page_mean=statistics.mean(per_page) if per_page else None,
                   documents=len(rec), pages=total_pages,
                   worker_peak_rss_mb=max((r.get("max_rss_mb") or 0 for r in rec.values()), default=None), wall_seconds=round(wall, 2), workers=a.workers,
                   throughput_pages_per_s=round(total_pages / wall, 3) if wall else None,
                   method="time inside each worker around the conversion call (excludes process spawn and imports); "
                          "each document runs in a fresh process (cold caches); outputs written after timing")
    slow = sorted(((r["seconds"], p) for p, r in rec.items() if r.get("ok")), reverse=True)[:10]
    result = dict(started=started, candidate=cand, impl=a.impl, legacy_md=a.legacy_md if a.impl == "legacy" else None,
                  bench_dir=bench_dir, jsonl_files=sorted(load_tests(bench_dir)), n_pdfs=len(need),
                  n_tests=sum(len(v) for v in load_tests(bench_dir).values()), hardware=hardware(),
                  config={k: os.getenv(k) for k in ("MAX_PAGES", "OCR_LANG", "OCR_TIMEOUT", "OCR_MAX_ATTEMPTS", "PARSEFLOW_REMOVE_FURNITURE",
                                                    "PARSEFLOW_MATH_RECOGNIZER", "VISION_PROVIDER") if os.getenv(k)},
                  vision_configured=bool(os.getenv("VISION_API_KEY")), annotated_markdown=a.annotated, timeout_s=a.timeout,
                  latency=latency, slowest=[dict(pdf=p, seconds=round(s, 3)) for s, p in slow],
                  errors=[dict(pdf=p, error=r.get("error")) for p, r in rec.items() if not r.get("ok")],
                  timeouts=sum(1 for r in rec.values() if r.get("timeout")))
    if os.path.exists(os.path.join(bench_dir, "manifest.json")):
        result["manifest"] = json.load(open(os.path.join(bench_dir, "manifest.json")))
    if not a.no_score:
        cmd, code, out = run_scorer(bench_dir, cand, a.olmocr, a.scorer_args.split())
        result["scorer"] = dict(command=" ".join(cmd), exit_code=code, **parse_scorer(out, cand))
        raw_path = (a.results or os.path.join(ROOT, "bench", "results", f"{cand}.json")).rsplit(".", 1)[0] + ".scorer.txt"
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        open(raw_path, "w").write(out)
        result["scorer"]["raw_output"] = os.path.relpath(raw_path, ROOT)
    out_path = a.results or os.path.join(ROOT, "bench", "results", f"{cand}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=1)
    open(out_path.rsplit(".", 1)[0] + ".md", "w").write(summary_md(result))
    print(summary_md(result))


def summary_md(r):
    s = r.get("scorer", {})
    L = r["latency"]
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    lines = [f"# olmOCR-Bench run: {r['candidate']}", "",
             f"- scorer: `{s.get('command', 'not run')}`",
             f"- dataset: {r['bench_dir']} - {r['n_pdfs']} PDFs, {r['n_tests']} tests, jsonl: {', '.join(r['jsonl_files'])}",
             f"- hardware: {r['hardware']['cpu']} · {r['hardware']['logical_cores']} logical cores · {r['hardware']['ram_gb']} GB · {r['hardware']['os']}",
             f"- engines: PyMuPDF {r['hardware'].get('pymupdf')} · {r['hardware'].get('tesseract')} · vision {'on' if r['vision_configured'] else 'off'}",
             "", f"**Overall (official): {s.get('overall', 'n/a')}%** (± {s.get('ci', 'n/a')})", "", "| JSONL / category | score | passed / tests |", "|---|---|---|"]
    for k, v in sorted(s.get("per_jsonl", {}).items()):
        lines.append(f"| {k} | {v['score']:.1f} | {v['passed']}/{v['tests']} |")
    lines += ["", "| test type | score | tests |", "|---|---|---|"]
    for k, v in sorted(s.get("per_test_type", {}).items()):
        lines.append(f"| {k} | {v['score']:.1f} | {v['tests']} |")
    lines += ["", f"Latency per page (s): median {fmt(L['per_page_median'])} · p90 {fmt(L['per_page_p90'])} · p95 {fmt(L['per_page_p95'])} · "
                  f"max {fmt(L['per_page_max'])} · throughput {L['throughput_pages_per_s']} pages/s with {L['workers']} workers",
              f"Errors: {len(r['errors'])} · timeouts: {r['timeouts']}", ""]
    if r["slowest"]:
        lines += ["Slowest documents: " + ", ".join(f"{x['pdf']} ({x['seconds']}s)" for x in r["slowest"][:5])]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
