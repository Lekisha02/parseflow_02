"""Run the ORIGINAL (v1) ParseFlow pipeline on one PDF and write per-page Markdown, so the old and new
implementations can be benchmarked under identical conditions.

v1 had no backend Markdown: the browser built it with md() in frontend/index.html. md_frontend() below is a
line-for-line Python port of that function (including its provenance HTML comments). md_plain() is the same
without the provenance comments, for a fairer text-only comparison.

Usage (normally called by run_olmocr_bench.py):
    python bench/legacy_runner.py --backend /path/to/v1/backend --pdf file.pdf --out out_base --pages 1 2 --variant frontend
"""
import argparse
import json
import os
import sys
import tempfile
import time


def _js_num(v):
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def md_frontend(doc, blocks):
    o = f"<!-- ParseFlow · {doc['filename']} -->\n"
    for b in sorted(blocks, key=lambda b: (b["page"], b["reading_order"])):
        prov = (f"<!-- {b['block_id']} · p{b['page']} · bbox {json.dumps(b['bbox'], separators=(',', ':'))} · {b['extractor']} · "
                f"{_js_num(b['confidence'])} ({b['confidence_type']}) · {b['verification']} -->")
        o += f"\n{prov}\n{_body(b)}\n"
    return o


def md_plain(doc, blocks):
    return "\n\n".join(_body(b) for b in sorted(blocks, key=lambda b: (b["page"], b["reading_order"]))) + "\n"


def _body(b):
    c, t = b.get("content"), b["type"]
    if c is None:
        return f"> ⚠ {'Visual region not interpreted' if t in ('figure', 'unknown') else 'Unreadable region'} — flagged for review (confidence {_js_num(b['confidence'])})."
    if t == "heading" or (t == "handwritten_text" and b["reading_order"] == 1):
        return "# " + c
    if t == "table":
        rows = b.get("rows") or []
        lines = []
        for i, r in enumerate(rows):
            line = "| " + " | ".join(r) + " |"
            if i == 0:
                line += "\n|" + "|".join("---" for _ in r) + "|"
            lines.append(line)
        return f"**{c}**\n\n" + "\n".join(lines)
    if t == "handwritten_code":
        return "### Handwritten Code\n```text\n" + c + "\n```"
    if t == "diagram":
        return "### Diagram\n" + c + "\n\nRelationships: " + ", ".join(b.get("relationships", []))
    if t == "chart":
        return "### Chart\n" + c
    if t == "equation":
        return "### Equation\n$$" + str(b.get("latex")) + "$$"
    if t == "handwritten_text":
        return "### Handwritten Notes\n" + c
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True, help="output path prefix; files are <out>_pg{N}_repeat1.md")
    ap.add_argument("--pages", type=int, nargs="+", required=True)
    ap.add_argument("--variant", choices=["frontend", "plain"], default="frontend")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    from processing import pipeline   # the v1 module
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        doc = pipeline.process(a.pdf, os.path.basename(a.pdf), "legacy", td, pipeline.Progress())
    dt = time.perf_counter() - t0
    fn = md_frontend if a.variant == "frontend" else md_plain
    for p in a.pages:
        with open(f"{a.out}_pg{p}_repeat1.md", "w") as f:
            f.write(fn(doc, [b for b in doc["blocks"] if b["page"] == p]))
    print(json.dumps(dict(ok=True, seconds=dt, pages=len(doc["pages"]))))


if __name__ == "__main__":
    main()
