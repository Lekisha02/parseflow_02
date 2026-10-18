"""How well does the IMAGE diagram reader (diagrams_raster) recover what the VECTOR reader reads exactly?

    python tools/eval_raster_diagrams.py book.pdf [--dpi 150] [--limit 0] [--out raster_eval.json]

For every diagram block that the vector reader finds in a born-digital PDF, the same region is rendered to pixels
(no text layer, no drawing commands) and read again by the image reader with OCR. The vector reading is the
reference - it was hand-audited (bench/diagram_audit_v25.md) and is exact about geometry, though not perfect.

Reported, summed over all figures:
  node labels   - recall: labelled vector nodes whose label the image reader also found on a node
  edges         - recall / precision of edges between labelled nodes, compared as unordered label pairs
  directions    - of the directed edges of the vector reading, how many the image reader found with the same
                  direction (overall), and among the edges it found at all (direction_on_found_edges)
  structure     - figures whose kinds of parts (tree / graph / automaton / sequence ...) agree
  time          - seconds per figure for the image reader (rendering excluded)
"""
import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402

from processing import diagrams, diagrams_raster, pipeline  # noqa: E402


def norm(s):
    return re.sub(r"\s+", "", str(s or "")).replace("−", "-").replace("–", "-").lower()


def edges_of(st):
    out = collections.Counter()
    for p in st.get("parts", []):
        es = p.get("edges") or []
        for e in es:
            m = re.match(r"^(.*?) (→|—) (.*?)(?: \((.*)\))?$", e)
            if m:
                a, b = norm(m.group(1).split(" (")[0]), norm(m.group(3).split(" (")[0])
                if "unlabelled" in a or "unlabelled" in b or not a or not b:
                    continue
                out[(tuple(sorted((a, b))), m.group(2) == "→" and (a, b))] += 1
        for parent, kids in p.get("children") or []:
            for k in kids:
                k = re.sub(r"^(left|right): ", "", k).split(" (edge label")[0]
                a, b = norm(parent), norm(k)
                if "unlabelled" in a or "unlabelled" in b or a == "nil" or b == "nil":
                    continue
                out[(tuple(sorted((a, b))), False)] += 1
        if p.get("type") == "automaton":
            for a, x, tos in p.get("transitions") or []:
                for b in tos:
                    out[(tuple(sorted((norm(a), norm(b)))), (norm(a), norm(b)))] += 1
    return out


def node_labels(st):
    out = collections.Counter()
    for p in st.get("parts", []):
        if p.get("type") in ("graph", "directed graph"):
            out.update(norm(v.split(" (")[0]) for v in p.get("vertices") or [] if "unlabelled" not in v)
        elif p.get("type") in ("tree", "binary tree"):
            out.update(norm(x) for x in p.get("level_order") or [] if "unlabelled" not in x and x != "NIL")
        elif p.get("type") == "automaton":
            out.update(norm(x) for x in p.get("states") or [] if "unlabelled" not in x)
        elif p.get("type") == "sequence":
            out.update(norm(x) for x in p.get("cells") or [] if x)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--dpi", type=float, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="raster_eval.json")
    ap.add_argument("--doc", help="an already processed document.json for this PDF (saves processing time)")
    ap.add_argument("--oracle-text", action="store_true",
                    help="give the image reader the PDF's own words instead of OCR: measures the geometry alone")
    a = ap.parse_args()
    if a.doc:
        doc = json.load(open(a.doc))
    else:
        doc = pipeline.process(a.pdf, Path(a.pdf).name, "rastereval000", "/tmp/rastereval", pipeline.Progress(), save_images=False)
    pdf = pymupdf.open(a.pdf)
    zoom = a.dpi / 72.0
    blocks = [b for b in doc["blocks"] if b["type"] == "diagram"]
    if a.limit:
        blocks = blocks[:a.limit]
    tot = collections.Counter()
    rows, secs = [], []
    for b in blocks:
        page = pdf[b["page"] - 1]
        reg = tuple(v / 2.0 for v in b["bbox"])
        ref = diagrams.extract(page, reg)
        if not ref:
            continue
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(*reg), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        t = time.time()
        try:
            words = None
            if a.oracle_text:
                words = [((w[0] - reg[0]) * zoom, (w[1] - reg[1]) * zoom, (w[2] - reg[0]) * zoom, (w[3] - reg[1]) * zoom, w[4], 1.0)
                         for w in diagrams.page_words(page) if reg[0] - 2 <= (w[0] + w[2]) / 2 <= reg[2] + 2 and reg[1] - 2 <= (w[1] + w[3]) / 2 <= reg[3] + 2]
            got = diagrams_raster.extract(img, origin=reg[:2], zoom=zoom, words=words) or {"parts": []}
        except Exception as e:
            got = {"parts": [], "error": f"{type(e).__name__}: {e}"}
        secs.append(time.time() - t)
        rn, gn = node_labels(ref), node_labels(got)
        re_, ge = edges_of(ref), edges_of(got)
        und = lambda c: collections.Counter({k[0]: 0 for k in c} | {}) if False else collections.Counter(k[0] for k in c.elements())
        ru, gu = und(re_), und(ge)
        match_e = sum((ru & gu).values())
        dirs_ref = collections.Counter(k[1] for k in re_.elements() if k[1])
        dirs_got = collections.Counter(k[1] for k in ge.elements() if k[1])
        matched_pairs = set((ru & gu).keys())
        dir_on_matched = collections.Counter(k[1] for k in re_.elements() if k[1] and k[0] in matched_pairs)
        r = dict(block=b["block_id"], page=b["page"], nodes_ref=sum(rn.values()), nodes_found=sum((rn & gn).values()),
                 directed_on_matched=sum(dir_on_matched.values()), directed_on_matched_same=sum((dir_on_matched & dirs_got).values()),
                 edges_ref=sum(ru.values()), edges_got=sum(gu.values()), edges_match=match_e,
                 directed_ref=sum(dirs_ref.values()), directed_match=sum((dirs_ref & dirs_got).values()),
                 kinds_ref=sorted({p["type"] for p in ref["parts"]}), kinds_got=sorted({p["type"] for p in got["parts"]}),
                 seconds=round(secs[-1], 2), error=got.get("error"))
        rows.append(r)
        for k in ("nodes_ref", "nodes_found", "edges_ref", "edges_got", "edges_match", "directed_ref", "directed_match",
                  "directed_on_matched", "directed_on_matched_same"):
            tot[k] += r[k]
        tot["figures"] += 1
        tot["kinds_agree"] += int(r["kinds_ref"] == r["kinds_got"])
        print(f"{b['block_id']}: nodes {r['nodes_found']}/{r['nodes_ref']} edges {r['edges_match']}/{r['edges_ref']} (got {r['edges_got']}) {r['seconds']}s", flush=True)
    secs.sort()
    summary = dict(figures=tot["figures"], dpi=a.dpi, text="PDF words (geometry only)" if a.oracle_text else "OCR",
                   node_label_recall=round(tot["nodes_found"] / max(1, tot["nodes_ref"]), 3),
                   edge_recall=round(tot["edges_match"] / max(1, tot["edges_ref"]), 3),
                   edge_precision=round(tot["edges_match"] / max(1, tot["edges_got"]), 3),
                   direction_agreement=round(tot["directed_match"] / max(1, tot["directed_ref"]), 3),
                   direction_on_found_edges=round(tot["directed_on_matched_same"] / max(1, tot["directed_on_matched"]), 3),
                   structure_kinds_agree=round(tot["kinds_agree"] / max(1, tot["figures"]), 3),
                   median_seconds=secs[len(secs) // 2] if secs else None, max_seconds=secs[-1] if secs else None,
                   errors=sum(1 for r in rows if r["error"]))
    json.dump(dict(summary=summary, rows=rows), open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
