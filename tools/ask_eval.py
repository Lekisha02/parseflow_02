"""Live evaluation of Ask and Ask-about-this-block with YOUR configured model (Gemini, Claude, OpenAI or a local one).

    python tools/ask_eval.py path/to/document.pdf [--questions q.txt] [--blocks 8] [--out ask_eval]

What it does (nothing is scored by hand, nothing is invented):
  1. processes the PDF with the normal pipeline (or reuses <out>/document.json),
  2. asks document questions (a summary, the questions in --questions, one per line, or a few defaults),
  3. asks about up to --blocks visual blocks (diagrams first, then tables, equations, figures),
  4. records for every answer: mode (llm / extractive), latency, page references removed because they were
     not in the evidence, whether the region image was sent, and - where ParseFlow computed an exact fact
     (automaton run, shortest path, cycle) - whether the model's answer agrees with it,
  5. writes <out>/ask_eval.json and <out>/ask_eval.md so you can read every answer next to its checks.

Agreement with computed facts is a mechanical check (the computed path / ACCEPTED-REJECTED verdict appears in
the answer); it does not grade explanation quality - read the answers for that.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _l in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    if "=" in _l and not _l.lstrip().startswith("#"):
        _k, _v = _l.split("=", 1)
        _v = re.sub(r"\s+#.*$", "", _v).strip()
        if _v:
            os.environ.setdefault(_k.strip(), _v)
sys.path.insert(0, str(ROOT / "backend"))

from processing import assistant, block_ask, llm, pipeline  # noqa: E402

DEFAULT_Q = ["Summarize this document", "What are the most important ideas, and how are they related?",
             "Explain the hardest concept in this document simply, with an example",
             "Give me 3 practice questions on this document with answers"]


def _block_questions(b):
    st = b.get("structure_text") or ""
    q = ["Explain this " + b["type"] + " step by step"]
    if "automaton" in st:
        q.append("Does it accept the string 0110? Trace the run.")
    elif " graph " in st:
        vs = re.search(r"vertices \((.*?)\) and \d+ edges", st)
        names = [v.strip().split(" (")[0] for v in (vs.group(1).split(", ") if vs else [])]
        names = [n for n in names if n and "unlabelled" not in n and not n.startswith("(")]
        if len(names) >= 2:
            q.append(f"Find the shortest path from {names[0]} to {names[-1]} and explain it")
    elif " tree " in st:
        q.append("Give the in-order and level-order traversals and explain them")
    elif b["type"] == "table":
        q.append("What pattern do the values in this table show?")
    elif b["type"] == "equation":
        q.append("Explain each symbol in this equation")
    return q


def _agrees(answer, facts):
    """For each computed fact: does the answer state the same result?"""
    out = []
    a = (answer or "").replace("->", "→")
    for f in facts:
        m = re.search(r"Shortest path from \S+ to \S+: (.+?) \(total", f)
        if m:
            path = [x.strip() for x in m.group(1).split("→")]
            pat = r"\s*(?:→|->|,|to|then)\s*".join(re.escape(p) for p in path)
            out.append(dict(fact=f, agrees=bool(re.search(pat, a))))
            continue
        m = re.search(r"Result: (ACCEPTED|REJECTED)", f)
        if m:
            yes = re.search(r"\baccept(?:s|ed)?\b", a, re.I) and not re.search(r"\b(not accept|reject)", a, re.I)
            out.append(dict(fact=f, agrees=bool(yes) == (m.group(1) == "ACCEPTED")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--questions")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--out", default="ask_eval")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    info = llm.info()
    print("model:", json.dumps(info))
    if not info.get("configured"):
        print("No model configured (set ASK_PROVIDER / ASK_API_KEY, see README). Running the offline path only.")
    src = Path(a.pdf)
    did = "evalevaleval"
    folder = out / did
    folder.mkdir(exist_ok=True)
    tgt = folder / f"source_{did}_{src.name}"
    if not tgt.exists():
        tgt.write_bytes(src.read_bytes())
    dj = folder / "document.json"
    if dj.exists():
        doc = json.loads(dj.read_text())
    else:
        t = time.time()
        doc = pipeline.process(str(tgt), src.name, did, str(folder), pipeline.Progress(), save_images=False)
        dj.write_text(json.dumps(doc))
        print(f"processed {len(doc['pages'])} pages in {time.time() - t:.1f}s")
    qs = [l.strip() for l in open(a.questions)] if a.questions else DEFAULT_Q
    qs = [q for q in qs if q]
    rows = []
    hist = []
    for q in qs:
        t = time.time()
        r = assistant.respond(doc, q, history=hist[-6:], key=did, folder=str(folder))
        dt = time.time() - t
        ans = r.get("answer") or json.dumps(r.get("extract"))[:2000]
        rows.append(dict(kind="document", question=q, mode=r.get("mode"), seconds=round(dt, 2), note=r.get("note"),
                         cited_pages=sorted({c["page"] for c in r.get("citations") or [] if c.get("cited")}), answer=ans))
        hist += [dict(role="user", content=q), dict(role="assistant", content=ans[:1500])]
        print(f"[doc] {q[:60]!r}: {r.get('mode')} {dt:.1f}s")
    order = {"diagram": 0, "table": 1, "equation": 2, "chart": 3, "figure": 4}
    vis = sorted((b for b in doc["blocks"] if b["type"] in order), key=lambda b: (order[b["type"]], b["page"]))[:a.blocks]
    for b in vis:
        bh = []
        for q in _block_questions(b):
            t = time.time()
            r = block_ask.respond(doc, b["block_id"], q, history=bh, key=did, folder=str(folder))
            dt = time.time() - t
            rows.append(dict(kind="block", block=b["block_id"], page=b["page"], type=b["type"], question=q, mode=r.get("mode"),
                             seconds=round(dt, 2), used_image=r.get("used_image"), computed=r.get("computed"),
                             agreement=_agrees(r.get("answer"), r.get("computed") or []) if r.get("mode") == "llm" else None,
                             note=r.get("note"), answer=r.get("answer")))
            bh += [dict(role="user", content=q), dict(role="assistant", content=(r.get("answer") or "")[:1500])]
            print(f"[{b['block_id']}] {q[:50]!r}: {r.get('mode')} {dt:.1f}s")
    lat = sorted(r["seconds"] for r in rows)
    ag = [x["agrees"] for r in rows for x in (r.get("agreement") or [])]
    summary = dict(model=info, answers=len(rows), llm_answers=sum(1 for r in rows if r["mode"] == "llm"),
                   median_seconds=lat[len(lat) // 2] if lat else None, max_seconds=lat[-1] if lat else None,
                   answers_with_removed_references=sum(1 for r in rows if "Removed page reference" in (r.get("note") or "")),
                   computed_fact_checks=len(ag), computed_fact_agreement=sum(ag))
    (out / "ask_eval.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, ensure_ascii=False))
    md = ["# Ask evaluation", "", "```", json.dumps(summary, indent=1), "```", ""]
    for r in rows:
        head = f"## {r['kind']}: {r['question']}" + (f"  ({r['block']}, p. {r['page']}, {r['type']})" if r["kind"] == "block" else "")
        md += [head, f"mode {r['mode']} · {r['seconds']} s" + (" · image sent" if r.get("used_image") else ""),
               *(["computed: " + "; ".join(r["computed"])] if r.get("computed") else []),
               *([f"agreement with computed facts: {r['agreement']}"] if r.get("agreement") else []),
               *([f"note: {r['note']}"] if r.get("note") else []), "", r["answer"] or "", ""]
    (out / "ask_eval.md").write_text("\n".join(md))
    print(json.dumps(summary, indent=1))
    print("wrote", out / "ask_eval.md")


if __name__ == "__main__":
    main()
