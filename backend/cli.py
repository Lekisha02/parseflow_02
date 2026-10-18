"""ParseFlow command line: convert every PDF in a folder to one Markdown file per PDF (offline, no API keys).

    parseflow input_pdfs/ --output-dir output_markdown/

Options: --recursive, --workers N, --timeout S, --json (also write the Digital Twin JSON), --report report.json,
--log-json (machine-readable progress on stdout), --include-headers (keep running headers/footers),
--plain (no HTML-comment warnings in the Markdown), --skip-existing, --images (also PNG/JPG).

Exit codes: 0 = every file converted; 1 = at least one file failed (successful outputs are kept); 2 = usage error.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def discover(inputs, recursive, images):
    """Files named directly are converted whatever their supported type; in folders: PDFs, Office, text and
    e-book files, plus images with --images."""
    from processing import formats
    img = {"." + e for e in formats.IMAGE_DIRECT | formats.IMAGE_CONVERT}
    allx = {"." + e for e in formats.SUPPORTED}
    exts = (allx - img) | (img if images else set())
    out = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            if p.suffix.lower() in allx:
                out.append((p, p.parent))
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out += [(f, p) for f in sorted(it) if f.is_file() and f.suffix.lower() in exts]
    return out


def plan_outputs(files, out_dir, multi_root):
    """Map each input to a unique .md path, mirroring sub-folders to avoid name collisions."""
    used, plan = set(), []
    for f, root in files:
        rel = f.relative_to(root).with_suffix("")
        if multi_root:
            rel = Path(root.name) / rel
        cand = out_dir / rel.with_suffix(".md")
        k = 2
        while str(cand).lower() in used:
            cand = out_dir / f"{rel}_{k}.md"; k += 1
        used.add(str(cand).lower())
        plan.append((f, cand))
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(prog="parseflow", description="Convert documents (PDF, Word, PowerPoint, Excel, images, text) to Markdown with ParseFlow (offline).")
    ap.add_argument("inputs", nargs="+", help="PDF files and/or folders")
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("-r", "--recursive", action="store_true", help="search folders recursively")
    ap.add_argument("-w", "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("-t", "--timeout", type=int, default=300, help="per-document timeout in seconds")
    ap.add_argument("--json", action="store_true", help="also write <name>.json (Digital Twin) next to each .md")
    ap.add_argument("--report", default=None, help="write a JSON report of all results to this path")
    ap.add_argument("--log-json", action="store_true", help="print one JSON line per finished document on stdout")
    ap.add_argument("--include-headers", action="store_true", help="keep running headers/footers/page numbers in the Markdown")
    ap.add_argument("--plain", action="store_true", help="omit HTML-comment warnings from the Markdown")
    ap.add_argument("--skip-existing", action="store_true", help="skip inputs whose .md already exists")
    ap.add_argument("--images", action="store_true", help="also convert image files found in folders (PNG, JPG, TIFF, BMP, GIF, WebP)")
    a = ap.parse_args(argv)

    files = discover(a.inputs, a.recursive, a.images)
    missing = [i for i in a.inputs if not Path(i).exists()]
    for m in missing:
        print(f"parseflow: input not found: {m}", file=sys.stderr)
    if not files:
        print("parseflow: no supported files found", file=sys.stderr)
        return 2 if missing or not a.inputs else 1
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = {r for _, r in files}
    plan = plan_outputs(files, out_dir, len(roots) > 1)
    if a.skip_existing:
        plan = [(f, o) for f, o in plan if not o.exists()]
    dest = {str(f): o for f, o in plan}

    from processing.runner import run_batch
    results, t0 = [], time.perf_counter()

    def on_result(path, res):
        o = dest[path]
        rec = dict(input=path, output=str(o) if res.get("ok") else None, ok=bool(res.get("ok")), seconds=round(res.get("seconds", 0), 3),
                   pages=res.get("pages"), error=res.get("error"), warnings=res.get("warnings", []))
        if res.get("ok"):
            try:
                o.parent.mkdir(parents=True, exist_ok=True)
                tmp = o.with_suffix(".md.part")
                tmp.write_text(res["markdown"], encoding="utf-8", errors="replace")
                os.replace(tmp, o)
                if a.json and res.get("doc") is not None:
                    jp = o.with_suffix(".json")
                    jp.write_text(json.dumps(res["doc"], ensure_ascii=True), encoding="utf-8")
                    rec["json"] = str(jp)
            except Exception as e:      # one unwritable result must not stop the batch
                rec.update(ok=False, output=None, error=f"could not write output: {type(e).__name__}: {e}")
        results.append(rec)
        if a.log_json:
            print(json.dumps(rec, ensure_ascii=True), flush=True)
        else:
            status = "ok " if rec["ok"] else "ERR"
            print(f"[{len(results)}/{len(plan)}] {status} {rec['seconds']:7.2f}s  {path}" + ("" if rec["ok"] else f"  -> {rec['error']}"), file=sys.stderr, flush=True)

    opts = dict(include_furniture=a.include_headers, annotate=not a.plain, want_json=a.json)
    run_batch([str(f) for f, _ in plan], opts, workers=a.workers, timeout=a.timeout, on_result=on_result)
    wall = time.perf_counter() - t0
    ok = sum(r["ok"] for r in results)
    failed = [r for r in results if not r["ok"]]
    summary = dict(total=len(results), succeeded=ok, failed=len(failed), wall_seconds=round(wall, 2),
                   pages=sum(r.get("pages") or 0 for r in results if r["ok"]), output_dir=str(out_dir))
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(dict(summary=summary, results=results), indent=1, ensure_ascii=True), encoding="utf-8")
    print(f"parseflow: {ok}/{len(results)} converted, {len(failed)} failed, {summary['pages']} pages in {wall:.1f}s -> {out_dir}", file=sys.stderr)
    for r in failed:
        print(f"  FAILED {r['input']}: {r['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
