"""Ask about one block: a paragraph, an equation, a table, a chart, a diagram, a figure, a code listing.

Three stages, kept separate so a wrong answer can be traced:

  1. extraction   what ParseFlow read for the block (text, LaTeX, table cells, chart data, diagram structure) with
                  its own verification status and confidence - shown to the reader as *extraction* confidence only
  2. evidence     the block, the original image of the region, its caption, the text around it, other pages that
                  refer to it (e.g. "Figure 6.1" elsewhere) and passages retrieved for the question, plus facts
                  COMPUTED from an extracted structure (traversals, shortest paths, automaton runs) - exact, not model
                  output
  3. explanation  a language model (vision-capable when available, so it sees the region itself) answers from
                  that evidence and separates what the document shows from general explanation. Without a model the
                  answer is the evidence itself, clearly labelled.
"""
import io
import json
import os
import re

from . import assistant, formats, ask, diagrams, llm

VISUAL = ("figure", "diagram", "chart", "table", "equation", "handwritten_text", "handwritten_code", "unknown", "code")
Z = 2.0

BLOCK_SYSTEM = """You are the ParseFlow tutor. The reader selected ONE block of their document and asks about it. You
receive: the block as ParseFlow extracted it (type, page, text / LaTeX / table / chart data), possibly an image of the
original region, a STRUCTURE (nodes, edges, cells, pointers) read from the vector drawing, from the image pixels or
from an AI model's reading (its source is stated), COMPUTED FACTS
derived from it by code (traversals, paths, automaton runs), the caption and text around the block, and related passages
from other pages (each with its page in square brackets).

Rules:
1. Ground every statement about the document in this material and cite pages like [p. 41]. Use only pages given.
2. Answer in Markdown with: **What the block shows** (grounded, cited) then **Explanation** (simple, student-friendly;
   general knowledge allowed and labelled as such; numbered steps, a small example or an analogy when they help;
   mathematics in LaTeX \\( \\) / \\[ \\]). For questions about practice, write **Practice** questions with answers.
3. A STRUCTURE read from a vector drawing is exact for what was read; one read from an image (pixels + OCR) can miss
   faint lines or misread labels; one from an AI model's reading is unverified. Say which you rely on. COMPUTED FACTS
   are exact for the structure they came from. Prefer a geometric structure over your own reading of the image for
   node labels, edges, values and traversal orders; where the image clearly shows something different (a label the
   OCR misread), say so as an observation from the image. Never invent labels, edges, values or symbols you cannot
   read; say what is unreadable.
4. Interpret, don't just describe: for a tree give root, parent/child relations, leaves, levels and traversals; for a
   stack the top and what push/pop would do; for a queue head/tail and enqueue/dequeue; for a linked list the order and
   pointers; for a graph vertices, edges, weights, paths, cycles; for a flowchart the steps, decisions and outcomes;
   for an automaton states, start/accepting states and transitions (trace inputs only with the computed run); for a
   table the meaning of rows/columns and the patterns the values support; for a chart axes, units, trends and
   extremes; for a formula each symbol, the assumptions and how it is used or derived; for code what it does step by
   step.
5. The extraction may be wrong. If the block is marked for review, if the text looks garbled, or if the image and the
   extracted text disagree, say so plainly and tell the reader what to check. Extraction confidence is not evidence
   that your explanation is right.
6. Follow-up questions keep this block as their subject unless the reader clearly asks about something else.
7. Reply in the language of the question."""


def _find(doc, block_id):
    return next((b for b in doc.get("blocks", []) if b.get("block_id") == block_id), None)


def _block_material(b):
    """The block as extracted, as text for the model."""
    t = b.get("type")
    out = [f"Type: {t}. Page {b.get('page')}. Extraction: {b.get('verification')} "
           f"(confidence {b.get('confidence')}, {b.get('confidence_type')}; method: {b.get('extractor')})."]
    if b.get("warnings"):
        out.append("Extraction warnings: " + " | ".join(b["warnings"][:4]))
    c = b.get("content")
    if t == "equation":
        out.append("LaTeX: " + (b.get("latex") or c or "(none)"))
    if t in ("table", "chart") and b.get("rows"):
        rows = b["rows"][:60]
        out.append("Table (row by row, cells separated by |):\n" + "\n".join(" | ".join(str(x) for x in r) for r in rows)
                   + (f"\n… {len(b['rows']) - 60} more rows" if len(b["rows"]) > 60 else ""))
    if t == "chart":
        for k in ("chart_type", "axes", "legend"):
            if b.get(k):
                out.append(f"{k}: {b[k]}")
    if c and t not in ("equation",):
        out.append("Text: " + str(c)[:4000])
    if b.get("raw_text") and b.get("raw_text") != c:
        out.append("Raw text-layer text: " + str(b["raw_text"])[:1500])
    for k in ("description", "ocr_text"):
        if b.get(k):
            out.append(f"{k}: {str(b[k])[:1500]}")
    if b.get("elements"):
        out.append("Elements (read by the vision model during processing): " + "; ".join(map(str, b["elements"][:60])))
    rel = [r for r in (b.get("relationships") or []) if not str(r).startswith(("caption", "caption_of"))]
    if rel:
        out.append("Relationships (read by the vision model during processing): " + "; ".join(map(str, rel[:60])))
    return "\n".join(out)


def _context(doc, b):
    """Caption, neighbouring blocks on the page, and other pages that refer to the same figure / table / equation."""
    page = [x for x in doc.get("blocks", []) if x.get("page") == b.get("page") and not (x.get("furniture") or {}).get("excluded")]
    page.sort(key=lambda x: x.get("reading_order", 0))
    k = next((i for i, x in enumerate(page) if x is b), None)
    cap = None
    for r in b.get("relationships") or []:
        if str(r).startswith("caption:"):
            cap = _find(doc, str(r).split(":", 1)[1])
    if cap is None and k is not None:
        for x in page[max(0, k - 2):k + 3]:
            if x.get("type") == "caption" or re.match(r"^(Figure|Fig\.|Table)\s+\d", x.get("content") or ""):
                cap = x; break
    around = []
    if k is not None:
        for x in page[max(0, k - 3):k] + page[k + 1:k + 4]:
            if x is not cap and x.get("content") and x.get("type") not in ("header", "footer", "page_number"):
                around.append(f"[p. {x['page']}] ({x['type']}) {str(x.get('latex') or x['content'])[:900]}")
    refs = []
    label = None
    if cap is not None:
        m = re.match(r"^((?:Figure|Fig\.|Table)\s+[\dA-Z]+(?:\.\d+)*)", cap.get("content") or "")
        label = m.group(1) if m else None
    if label:
        pat = re.compile(r"\b" + re.escape(label).replace(r"Figure", r"(?:Figure|Fig\.)") + r"\b(?!\.\d)")
        for x in doc.get("blocks", []):
            if x is cap or x is b or not x.get("content") or (x.get("furniture") or {}).get("excluded"):
                continue
            if pat.search(x["content"]):
                refs.append(f"[p. {x['page']}] {x['content'][:700]}")
            if len(refs) >= 4:
                break
    return cap, around, refs, label


def _crop(folder, b, max_side=1600):
    """(png bytes, zoom, origin) of the block's region with a small margin, or (None, 0, (0, 0))."""
    try:
        import pymupdf
        path = formats.source_file(folder)
        if not path:
            return None, 0, (0, 0)
        x0, y0, x1, y1 = [v / Z for v in b["bbox"]]
        pad = 6
        if path.lower().endswith(".pdf"):
            d = pymupdf.open(path)
            try:
                pg = d[b["page"] - 1]
                clip = pymupdf.Rect(max(0, x0 - pad), max(0, y0 - pad), min(pg.rect.width, x1 + pad), min(pg.rect.height, y1 + pad))
                zoom = min(3.0, max_side / max(clip.width, clip.height, 1))
                pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
                return pix.tobytes("png"), zoom, (clip.x0, clip.y0)
            finally:
                d.close()
        from PIL import Image
        im = Image.open(path).convert("RGB")
        box = (int(max(0, x0 - pad) * Z), int(max(0, y0 - pad) * Z), int((x1 + pad) * Z), int((y1 + pad) * Z))
        buf = io.BytesIO(); im.crop(box).save(buf, "PNG")
        return buf.getvalue(), Z, (box[0] / Z, box[1] / Z)
    except Exception:
        return None, 0, (0, 0)


def _crop_png(folder, b, max_side=1600):
    """The original region of the block as PNG bytes (rendered from the source PDF, or cut from the image)."""
    try:
        from PIL import Image
        import pymupdf
        path = formats.source_file(folder)
        if not path:
            return None
        src = os.path.basename(path)
        x0, y0, x1, y1 = [v / Z for v in b["bbox"]]
        pad = 10
        if src.lower().endswith(".pdf"):
            d = pymupdf.open(path)
            try:
                pg = d[b["page"] - 1]
                clip = pymupdf.Rect(max(0, x0 - pad), max(0, y0 - pad), min(pg.rect.width, x1 + pad), min(pg.rect.height, y1 + pad))
                zoom = min(3.0, max_side / max(clip.width, clip.height, 1))
                pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
                return pix.tobytes("png")
            finally:
                d.close()
        im = Image.open(path).convert("RGB")
        sx = im.width / max(1, (b.get("_page_w") or im.width / Z))
        crop = im.crop((int(max(0, x0 - pad) * Z), int(max(0, y0 - pad) * Z), int((x1 + pad) * Z), int((y1 + pad) * Z)))
        crop.thumbnail((max_side, max_side))
        buf = io.BytesIO(); crop.save(buf, "PNG")
        return buf.getvalue()
    except Exception:
        return None


def _structure(folder, b):
    """The block's diagram structure: saved at processing time; else read now - from the PDF's vector drawing, then
    from the pixels of the region (scans, photos, embedded pictures), then from an AI model's own list of nodes and
    edges (marked as such)."""
    if b.get("structure"):
        return b["structure"]
    if b.get("type") not in ("figure", "diagram", "unknown", "chart", "handwritten_text", "handwritten_code"):
        return None
    st = None
    try:
        import pymupdf
        src = formats.source_file(folder) if folder else None
        if src and src.lower().endswith(".pdf"):
            d = pymupdf.open(src)
            try:
                st = diagrams.extract(d[b["page"] - 1], tuple(v / Z for v in b["bbox"]))
            finally:
                d.close()
        if st is None and src:
            png, zoom, origin = _crop(folder, b)
            if png:
                from PIL import Image
                from . import diagrams_raster
                st = diagrams_raster.extract(Image.open(io.BytesIO(png)).convert("RGB"), origin=origin, zoom=zoom)
    except Exception:
        st = None
    if st is None and b.get("relationships"):
        st = diagrams.from_model(b.get("elements"), b.get("relationships"))
    return st


WORD = re.compile(r"""(?:string|input|word)s?\s*["'“‘]?([^\s"'”’?,.]+)|["'“‘]([01ab]+)["'”’]""", re.I)
PAIR = re.compile(r"\bfrom\s+(\S+?)\s+to\s+(\S+?)(?=[\s?.,]|$)", re.I)


def _word_for(part, question):
    """The input string the question asks about: quoted or after 'string/input/word', else a token made only of
    the automaton's alphabet ('does it accept 0101?'). Letter tokens must be 2+ symbols so 'a' in 'is a' is not
    taken for an input."""
    m = WORD.search(question)
    if m:
        return m.group(1) or m.group(2)
    alpha = set(part.get("alphabet") or [])
    if not alpha or not all(len(x) == 1 for x in alpha):
        return None
    if re.search(r"(?i)empty (string|word|input)|\bε\b|\bepsilon\b", question):
        return ""
    for tok in re.findall(r"[^\s?.,;:!()]+", question):
        if set(tok) <= alpha and (len(tok) >= 2 or not tok.isalpha()):
            return tok
    return None


def computed_facts(st, question):
    """Exact answers code can give from the structure for this question (a run of an automaton on the asked string,
    a shortest path between the asked vertices, connectivity and cycles), as sentences."""
    if not st:
        return []
    out = []
    for p in st.get("parts", []):
        if p["type"] == "automaton":
            w = _word_for(p, question or "")
            if w is not None:
                r = diagrams.run_automaton(p, w)
                if r:
                    trail = " → ".join("{" + ", ".join(s["states"]) + "}" + (f" (read {s['read']})" if s["read"] else "") for s in r["steps"])
                    out.append(f"Computed run of the automaton on “{w}”: {trail}. Result: {'ACCEPTED' if r['accepted'] else 'REJECTED'}"
                               + (" (no transition for a symbol: the run dies)" if r["dead"] else "") + ".")
        if p["type"] in ("graph", "directed graph"):
            m = PAIR.search(question or "")
            names = {v.split(" (")[0]: v for v in p["vertices"]} | {v: v for v in p["vertices"]}
            src = dst = None
            if m:
                src, dst = names.get(m.group(1)), names.get(m.group(2))
            f = diagrams.graph_facts(p, src, dst)
            if f:
                s = f"Computed: the graph has {f['connected_components']} connected component(s) and {'a' if f['has_cycle'] else 'no'} cycle."
                sp = f.get("shortest_path")
                if sp and sp.get("path"):
                    s += f" Shortest path from {src} to {dst}: {' → '.join(sp['path'])} (total {'weight' if sp['weighted'] else 'edges'} {sp['length']:g})."
                elif sp:
                    s += " " + sp["note"] + "."
                out.append(s)
    return out


def _offline(b, st, facts, cap, around, refs, ctx_hits, ix):
    """Answer without a model: the evidence itself, labelled."""
    parts = []
    if st:
        parts.append("**What the drawing shows** *(read from the vector geometry, exact)*\n\n" + diagrams.describe(st))
    elif b.get("type") == "table" and b.get("rows"):
        rows = b["rows"]
        hdr = rows[0] if rows else []
        parts.append(f"**The table** has {max(0, len(rows) - 1)} data rows and {len(hdr)} columns: " + ", ".join(str(h) for h in hdr) + ".")
        nums = []
        for j, h in enumerate(hdr):
            col = [assistant_num(r[j]) for r in rows[1:] if j < len(r)]
            col = [v for v in col if v is not None]
            if len(col) >= 2:
                nums.append(f"{h}: from {min(col):g} to {max(col):g}")
        if nums:
            parts.append("Numeric columns (computed): " + "; ".join(nums) + ".")
    elif b.get("type") == "equation":
        parts.append("**The equation** as extracted:\n\n\\[" + (b.get("latex") or b.get("content") or "") + "\\]")
    elif b.get("content"):
        parts.append("**The block**\n\n" + str(b["content"])[:1500])
    if facts:                                   # the exact answer to the question comes first
        parts.insert(0, "**Computed from the structure** *(exact, by code)*\n\n" + "\n".join("- " + f for f in facts))
    if cap is not None:
        parts.append(f"**Caption** *(p. {cap['page']})*: " + (cap.get("content") or ""))
    if refs:
        parts.append("**Where the document refers to it**\n" + "\n".join("- " + r for r in refs[:3]))
    elif ctx_hits:
        parts.append("**Related passages**\n" + "\n".join(f"- {re.sub(r'\s+', ' ', ix.chunks[i]['text'])[:300]} *({assistant._pages_str(ix.chunks[i]['pages'])})*" for i in ctx_hits[:3]))
    return "\n\n".join(parts)


def assistant_num(x):
    try:
        return float(str(x).replace(",", "").replace("−", "-").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def respond(doc, block_id, question, history=None, key=None, folder=None, doc_mtime=None, use_llm=True):
    ix = assistant.index_for(doc, key, folder=folder, doc_mtime=doc_mtime)
    doc = ix.doc
    b = _find(doc, block_id)
    if b is None:
        return dict(error=dict(code="PF-404", title="Not Found", message="No such block in this document."))
    question = (question or "").strip()[:2000] or "Explain this block."
    st = _structure(folder, b) if folder else b.get("structure")
    facts = computed_facts(st, question)
    cap, around, refs, label = _context(doc, b)
    # related passages: the question plus the block's own words, anywhere in the document
    qt = ask.tokens(question) + ask.tokens(str(b.get("content") or "")[:400])[:20]
    hits = [i for _, i in ix.search(qt, k=6)] if ix.chunks else []
    hits = [i for i in hits if b.get("page") not in ix.chunks[i]["pages"]][:4]
    extraction = dict(verification=b.get("verification"), confidence=b.get("confidence"), method=b.get("extractor"),
                      basis=b.get("confidence_type"), warnings=b.get("warnings") or [])
    base = dict(block=dict(id=block_id, page=b.get("page"), type=b.get("type")), question=question, extraction=extraction,
                structure=diagrams.describe(st) if st else None, computed=facts, llm=llm.info(), used_image=False)
    pages = {b.get("page")} | ({cap["page"]} if cap else set()) | {int(p) for r in refs + around for p in re.findall(r"^\[p\. (\d+)\]", r)}
    pages |= {p for i in hits for p in ix.chunks[i]["pages"]}
    if use_llm and llm.available():
        img = None
        if folder and (b.get("type") in VISUAL or st) and llm.vision_ok():
            img = _crop_png(folder, b)
        mat = [f"Document: {doc.get('filename') or 'document'}",
               f"SELECTED BLOCK [p. {b.get('page')}]\n" + _block_material(b)]
        if st:
            mat.append(f"STRUCTURE (source: {st.get('source', 'vector drawing')}):\n" + diagrams.describe(st) +
                       ("\nRaw: " + json.dumps(st["parts"], ensure_ascii=False)[:5000]))
        if b.get("model_reading"):
            mr = b["model_reading"]
            mat.append("AN AI VISION MODEL'S READING OF THE SAME IMAGE (made at processing time; not verified):\n" +
                       json.dumps(dict(elements=mr.get("elements"), relationships=mr.get("relationships")), ensure_ascii=False)[:3000])
        if b.get("agreement"):
            mat.append("CHECK OF THAT READING AGAINST THE IMAGE GEOMETRY: " + b["agreement"]["summary"])
        if facts:
            mat.append("COMPUTED FACTS (exact, by code from the structure):\n" + "\n".join("- " + f for f in facts))
        if cap is not None:
            mat.append(f"CAPTION [p. {cap['page']}]: {cap.get('content')}")
        if around:
            mat.append("TEXT AROUND THE BLOCK:\n" + "\n".join(around))
        if refs:
            mat.append(f"OTHER PAGES REFERRING TO {label}:\n" + "\n".join(refs))
        if hits:
            mat.append("RELATED PASSAGES:\n" + "\n\n".join(assistant._excerpt(ix, i, limit=1500) for i in hits))
        if img:
            mat.append("An image of the original region is attached.")
        style = [txt for rx, txt in assistant.STYLE if rx.search(question)]
        user = "\n\n".join(mat) + f"\n\nQuestion about the selected block: {question}" + (("\n\nHow to answer: " + " ".join(style)) if style else "")
        try:
            text = llm.chat(BLOCK_SYSTEM, assistant._history(history) + [dict(role="user", content=user)],
                            max_tokens=int(os.getenv("ASK_MAX_TOKENS", "1600")), images=[("image/png", img)] if img else None)
            if text:
                text, removed = assistant.check_citations(text, pages)
                out = dict(base, mode="llm", answer=text, used_image=bool(img), citations=_cites(b, cap, hits, ix, text))
                if removed:
                    out["note"] = "Removed page reference(s) not among the evidence: " + ", ".join(f"p. {r}" for r in removed) + "."
                return out
        except llm.LLMError as e:
            base["note"] = f"The language model could not answer ({e}); showing what ParseFlow read instead."
    text = _offline(b, st, facts, cap, around, refs, hits, ix) or "Nothing was extracted for this block."
    return dict(base, mode="extractive", answer=text, citations=_cites(b, cap, hits, ix, text),
                note=base.get("note") or "No AI model configured: this is what ParseFlow read and computed for the block. "
                                         "Explanations need ASK_PROVIDER (Gemini, Claude, OpenAI, or a local model).")


def _cites(b, cap, hits, ix, text):
    out = [dict(page=b.get("page"), pages=[b.get("page")], block_id=b.get("block_id"), section="selected block",
                snippet=str(b.get("content") or b.get("latex") or "")[:160], cited=True)]
    if cap is not None:
        out.append(dict(page=cap["page"], pages=[cap["page"]], block_id=cap["block_id"], section="caption",
                        snippet=str(cap.get("content") or "")[:160], cited=True))
    out += assistant._citations(ix, hits, text)
    return out
