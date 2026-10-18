"""Ask assistant: contextual Q&A over one processed document.

    question (+ conversation so far)
        -> intent (summary / scoped summary / question)
        -> retrieval over the WHOLE document: passages ("chunks") ranked by BM25 with section titles, definition
           and phrase bonuses, neighbouring context, and the passages behind the previous answer for follow-ups
        -> with a language model configured (llm.py): the model answers from those passages, citing pages,
           and keeps "From the document" separate from "Explanation" (general knowledge, examples, steps)
        -> without one (or if the call fails): an extractive answer — the best passages quoted from across the
           document with page numbers, or an extractive summary — clearly labelled as such.

No training is involved: the model reads the relevant pages at question time (retrieval-augmented generation),
which works for any document and keeps every statement traceable to a page.
"""
import math
import os
import re
from collections import Counter

from . import ask, embeddings, llm

FURN = ("header", "footer", "page_number", "margin")
SUMMARY = re.compile(r"\b(summar(?:y|i[sz]e|i[sz]ing)|overview|tl;?dr|gist|main (?:points|ideas|topics)|key (?:points|takeaways|ideas)|"
                     r"what (?:is|are) (?:this|the) (?:document|book|paper|file|report|pdf|text)\s+(?:about|on)|outline of)\b", re.I)
FOLLOW = re.compile(r"^\s*(and|but|so|also|why|how come|what about|explain (?:it|that|this|more|again)|simpler|in simple|"
                    r"give (?:me )?an example|example|more detail|elaborate|continue|go on)\b|\b(it|its|this|that|these|those|they|them|"
                    r"he|she|above|previous|earlier|same)\b", re.I)
PAGE_SCOPE = re.compile(r"\bpages?\s*(\d{1,5})(?:\s*(?:-|–|to|through)\s*(\d{1,5}))?", re.I)
CHAPTER_SCOPE = re.compile(r"\b(chapter|section|part|appendix)\s+([0-9]+(?:\.[0-9]+)*|[A-Z]|[ivxlc]+)\b", re.I)
EXAMPLE = re.compile(r"\b(examples?|for instance|illustrat\w*|show me how|worked|sample)\b", re.I)
EX_SENT = re.compile(r"\b(for example|for instance|e\.g\.|example\b|such as|consider|suppose|illustrat\w*)", re.I)
CH_INTRO = re.compile(r"\b(this (?:chapter|part|appendix)|in this (?:chapter|part)|the (?:chapter|part) (?:begins|starts|shows|presents)|"
                      r"we (?:will |’ll |'ll )?(?:begin|start|study|examine|introduce|present|show|explore|focus|look at|consider))\b", re.I)
CHUNK_CHARS = 1100
FOLLOW_SOFT = set(ask.tokens("explain explanation simple simply simpler example examples detail details elaborate more again "
                             "understand meaning means clarify step steps easy easier words plain please why how"))


def _block_text(b):
    t, c = b.get("type"), b.get("content") or ""
    if t == "heading":
        return "## " + c
    if t == "table" and b.get("rows"):
        rows = [" | ".join(str(x).strip() for x in r) for r in b["rows"][:40]]
        return "Table:\n" + "\n".join(rows)
    if t == "equation":
        return "\\[" + (b.get("latex") or c) + "\\]"
    if t in ("code", "handwritten_code"):
        return "```\n" + c + "\n```"
    if t in ("figure", "chart", "diagram", "unknown"):
        bits = []
        if b.get("structure_text"):
            bits.append(b["structure_text"][:1500])
        elif c:
            bits.append(c[:300])
        if t == "chart" and b.get("rows"):
            bits.append("data: " + "; ".join(" | ".join(str(x) for x in r) for r in b["rows"][:15]))
        if b.get("relationships"):
            rel = [r for r in b["relationships"] if not str(r).startswith(("caption", "caption_of"))]
            if rel:
                bits.append("relationships: " + "; ".join(map(str, rel[:30])))
        return f"[{t}: {' '.join(bits)}]" if bits else ""
    return c


def _flag(b):
    """Short reason a block's extraction is uncertain (shown to the model so it can say so), or None."""
    if b.get("verification") == "verified":
        return None
    w = " ".join(b.get("warnings") or []).lower()
    if "character map" in w:
        return "some characters come from a broken font map"
    if b.get("extractor") == "ocr" or b.get("confidence_type") == "ocr":
        return "read by OCR from a scan"
    if b.get("type") == "equation" or b.get("inline_math"):
        return "mathematics rebuilt from glyph positions"
    if b.get("type") in ("table",):
        return "table structure inferred"
    if b.get("type") in ("diagram", "chart"):
        return "read from the drawing"
    if b.get("extractor") == "fallback":
        return "region not interpreted"
    return None


class DocIndex:
    def __init__(self, doc):
        self.doc = doc
        self.n_pages = len(doc.get("pages") or [])
        bl = [b for b in doc.get("blocks", []) if b.get("role") not in FURN and not (b.get("furniture") or {}).get("excluded")]
        bl.sort(key=lambda b: (b.get("page", 0), b.get("reading_order", 0)))
        self.blocks = bl
        self.headings = []                 # (index into chunks at the heading, level, text, page)
        chunks, cur, stack = [], None, []

        def close():
            nonlocal cur
            if cur and cur["text"].strip():
                chunks.append(cur)
            cur = None
        for b in bl:
            txt = _block_text(b).strip()
            if not txt:
                continue
            if b.get("type") == "heading":
                lvl = b.get("level") or 1
                stack = [h for h in stack if h[0] < lvl] + [(lvl, (b.get("content") or "").strip())]
                close()
                self.headings.append((len(chunks), lvl, (b.get("content") or "").strip(), b.get("page")))
            sec = " › ".join(h[1] for h in stack)[-160:]
            if cur is None or len(cur["text"]) + len(txt) > CHUNK_CHARS or cur["section"] != sec or b.get("page", 0) - cur["pages"][-1] > 1:
                close()
                cur = dict(text="", pages=[b.get("page", 0)], blocks=[], section=sec)
            cur["text"] += ("\n\n" if cur["text"] else "") + txt
            fl = _flag(b)
            if fl and fl not in cur.setdefault("flags", []):
                cur["flags"].append(fl)
            if b.get("page", 0) not in cur["pages"]:
                cur["pages"].append(b.get("page", 0))
            cur["blocks"].append(b["block_id"])
        close()
        for i, c in enumerate(chunks):
            c["id"] = i
        self.chunks = chunks
        self._derive()
        self.digests = self._chapter_digests()
        self.chapter_summaries = {}

    def _derive(self, lsa=None):
        """Term statistics, the meaning space (latent semantic analysis) and the chapters from the chunks."""
        chunks = self.chunks
        self.toks = [ask.tokens(c["text"]) for c in chunks]
        self.sec_toks = [set(ask.tokens(c["section"])) for c in chunks]
        self.tf = [Counter(t) for t in self.toks]
        self.N = len(chunks) or 1
        df = Counter()
        for t in self.toks:
            df.update(set(t))
        self.df = df
        self.avgdl = (sum(len(t) for t in self.toks) / self.N) if chunks else 1.0
        self.chapters = self._toc_chapters()
        self.lsa = lsa if lsa is not None else _lsa(self)
        self.emb = None                  # optional neural embeddings (embeddings.py), loaded lazily

    # ---------------------------------------------------------------- meaning-based similarity
    def semantic(self, q_tokens):
        """Cosine similarity of the question to every chunk in the document's own concept space (LSA), or None."""
        L = self.lsa
        if not L or not q_tokens:
            return None
        import numpy as np
        v = np.zeros(len(L["vocab"]), np.float32)
        for w, f in Counter(q_tokens).items():
            k = L["index"].get(w)
            if k is not None:
                v[k] = (1 + math.log(f)) * L["idf"][k]
        if not v.any():
            return None
        qv = v @ L["proj"]
        n = float(np.linalg.norm(qv))
        if n == 0:
            return None
        direct = L["vecs"] @ (qv / n)
        # pseudo-relevance feedback: the passages that use the question's words most define what it is about;
        # passages close to them in the concept space match even without sharing a word with the question
        qs = set(q_tokens)
        lex = [(sum(self.tf[i].get(w, 0) * self.idf(w) for w in qs) / (1 + len(self.toks[i])) ** 0.5, i)
               for i in range(len(self.chunks))]
        top = [i for sc, i in sorted(lex, reverse=True)[:15] if sc > 0]
        if not top:
            return direct
        cen = L["vecs"][top].mean(axis=0)
        cn = float(np.linalg.norm(cen))
        if cn == 0:
            return direct
        return 0.5 * direct + 0.5 * (L["vecs"] @ (cen / cn))

    def _chapter_digests(self, per=2):
        """Two central sentences per chapter: the offline summary of a book covers every chapter."""
        out = {}
        for k, (label, title, ci, page) in enumerate(self.chapters):
            end = self.chapters[k + 1][2] if k + 1 < len(self.chapters) else len(self.chunks)
            ids = list(range(ci, max(ci + 1, end)))
            # a chapter's opening usually says what it is about ('This chapter shows how ...')
            intro = []
            for i in ids[:8]:
                for snt in _sentences(self.chunks[i]["text"]):
                    if CH_INTRO.search(snt) and not re.search(r"\b(exercise|problem)s?\b", snt, re.I):
                        intro.append((i, snt))
            # prefer openings that name the chapter's subject; 'This chapter will answer these questions' says little
            tw = set(ask.tokens(title))
            vague = re.compile(r"\b(these|those|such) (questions|ideas|issues|things)\b|^(it|this|they)\b(?! (chapter|part))", re.I)
            intro.sort(key=lambda t: -(2 * len(tw & set(ask.tokens(t[1]))) + (2.5 if re.search(r"\b(this|in this) (chapter|part|appendix)\b", t[1], re.I) else 0)
                                        + min(len(t[1].split()), 25) / 25 - (3 if vague.search(t[1]) else 0)))
            intro = [x for x in intro if not vague.search(x[1])] or intro
            if len(intro) < per:
                picked, _ = extractive_summary(self, ids, n=per, outline=False)
                intro += [x for x in picked if x not in intro]
            out[label] = sorted(intro[:per], key=lambda t: t[0])
        return out

    # ---------------------------------------------------------------- persistence
    def save(self, folder, doc_mtime=None):
        """ask_index.json (+ ask_lsa.npz) next to document.json, so a large document's index is built once."""
        import json
        import numpy as np
        data = dict(version=3, doc_mtime=doc_mtime, n_pages=self.n_pages, chunks=self.chunks, headings=self.headings,
                    digests=self.digests, chapter_summaries=self.chapter_summaries)
        tmp = os.path.join(folder, "ask_index.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(folder, "ask_index.json"))
        if self.lsa:
            np.savez_compressed(os.path.join(folder, "ask_lsa.npz"), vecs=self.lsa["vecs"], proj=self.lsa["proj"],
                                idf=self.lsa["idf"], vocab=np.array(self.lsa["vocab"]))

    @classmethod
    def load(cls, folder, doc, doc_mtime=None):
        import json
        import numpy as np
        p = os.path.join(folder, "ask_index.json")
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        if data.get("version") != 3 or (doc_mtime is not None and data.get("doc_mtime") not in (None, doc_mtime)):
            return None
        ix = cls.__new__(cls)
        ix.doc = doc if doc is not None else {}
        ix.n_pages = data.get("n_pages") or len((doc or {}).get("pages") or [])
        bl = [b for b in (doc or {}).get("blocks", []) if b.get("role") not in FURN and not (b.get("furniture") or {}).get("excluded")]
        bl.sort(key=lambda b: (b.get("page", 0), b.get("reading_order", 0)))
        ix.blocks = bl
        ix.chunks = data["chunks"]
        ix.headings = [tuple(h) for h in data["headings"]]
        lsa = None
        q = os.path.join(folder, "ask_lsa.npz")
        if os.path.exists(q):
            try:
                z = np.load(q, allow_pickle=False)
                vocab = [str(w) for w in z["vocab"]]
                lsa = dict(vecs=z["vecs"], proj=z["proj"], idf=z["idf"], vocab=vocab, index={w: k for k, w in enumerate(vocab)})
            except Exception:
                lsa = None
        ix._derive(lsa=lsa if lsa is not None else False)
        if ix.lsa is False:
            ix.lsa = _lsa(ix)
        ix.digests = {k: [tuple(x) for x in v] for k, v in (data.get("digests") or {}).items()}
        ix.chapter_summaries = data.get("chapter_summaries") or {}
        ix.folder = folder
        return ix

    def _toc_chapters(self):
        """[(label, title, chunk index, page)] for the top-level entries of the document's own table of contents
        ('2 Getting Started 17', 'Part I Foundations', 'A Summations 1140'), located at their headings."""
        limit = max(40, int(0.06 * self.n_pages))
        entries, toc_page = [], 0
        for b in self.blocks:
            if b.get("type") != "list" or b.get("page", 0) > limit:
                continue
            lines = [l.strip() for l in (b.get("content") or "").split("\n") if l.strip()]
            if len(lines) < 6 or sum(1 for l in lines if re.search(r"\s(\d{1,4}|[ivxlc]{1,7})$", l)) < 0.5 * len(lines):
                continue
            toc_page = max(toc_page, b.get("page", 0))
            for l in lines:
                m = re.match(r"^(?:★\s*)?((?:Part\s+)?[IVXL]{1,5}|\d{1,2}|[A-H])\s+([^\d\s].*?)(?:\s+(?:\d{1,4}|[ivxlc]{1,7}))?$", l)
                if m:
                    entries.append((m.group(1), m.group(2).strip()))
        if len(entries) < 4:
            return []
        norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
        out, start = [], 0
        for label, title in entries:
            nt = norm(title)
            for j in range(start, len(self.headings)):
                ci, lvl, text, page = self.headings[j]
                if page and page > toc_page and (norm(text) == nt or norm(text).endswith(" " + nt)):
                    out.append((label, title, ci, page)); start = j + 1
                    break
        return out if len(out) >= 4 else []

    def chapter_range(self, number, kind=None):
        """Chunk ids of chapter / part / appendix `number` from the table of contents, or None."""
        num = number.strip().lower()
        for k, (label, title, ci, page) in enumerate(self.chapters):
            lab = label.lower().replace("part", "").strip()
            is_part = label.lower().startswith("part") or bool(re.fullmatch(r"[ivxl]+", lab))
            if lab != num or (kind and kind.lower() == "part") != is_part and not (kind or "").lower() in ("", "section"):
                continue
            end = len(self.chunks)
            for label2, _, ci2, _ in self.chapters[k + 1:]:
                lab2 = label2.lower().replace("part", "").strip()
                is_part2 = label2.lower().startswith("part") or bool(re.fullmatch(r"[ivxl]+", lab2))
                if ci2 > ci and (is_part2 or not is_part):
                    end = ci2; break
            return list(range(ci, end))
        return None

    def idf(self, w):
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, q, k=8, pages=None, phrases=()):
        if not q or not self.chunks:
            return []
        qs = set(q)
        out = []
        for i, c in enumerate(self.chunks):
            if pages and not (set(c["pages"]) & pages):
                continue
            tf, dl = self.tf[i], len(self.toks[i]) or 1
            s = 0.0
            for w in qs:
                f = tf.get(w, 0)
                if f:
                    s += self.idf(w) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl / self.avgdl))
            s += 0.8 * sum(self.idf(w) for w in qs & self.sec_toks[i])          # the section is about it
            low = c["text"].lower()
            ph = sum(1 for r in phrases if r.search(low))
            s += 1.5 * ph
            if ph and re.search(ask.DEF_PAT + r"|\bmeans\b|\brefers\s+to\b|\bis\s+defined\b|\bis\s+an?\b|\bare\b", low):
                s += 1.2                                                         # a definition, not just a mention
            if s > 0:
                cov = len(qs & set(self.toks[i])) / len(qs)
                out.append((s * (0.6 + 0.4 * cov), i))
        # meaning-based retrieval: passages about the same thing in other words (LSA over this document, plus neural
        # embeddings when configured) join the keyword hits
        sem = self.semantic(list(q))
        emb = embeddings.similarity(self, " ".join(q)) if embeddings.available() else None
        if sem is not None or emb is not None:
            import numpy as np
            top = max((t[0] for t in out), default=0.0) or 1.0
            mix = np.zeros(len(self.chunks), np.float32)
            if sem is not None:
                mix += 0.45 * np.clip(sem, 0, None)
            if emb is not None:
                mix += 0.6 * np.clip(emb, 0, None)
            cand = {i for _, i in out} | {int(i) for i in np.argsort(-mix)[: max(40, 4 * k)]}
            lex = {i: v / top for v, i in out}
            out = [(lex.get(i, 0.0) + float(mix[i]), i) for i in cand
                   if not pages or (set(self.chunks[i]["pages"]) & pages)]
        out.sort(key=lambda t: (-round(t[0], 6), t[1]))
        return out[:k]

    def section_chunks(self, title_tokens=None, number=None, kind=None):
        """Chunks of the section whose heading matches 'Chapter 2' / '2.3' / a title."""
        if number and "." not in number and (kind or "chapter").lower() in ("chapter", "part", "appendix"):
            r = self.chapter_range(number, kind)
            if r:
                return r
        best = None
        for j, (ci, lvl, text, page) in enumerate(self.headings):
            t = text.lower()
            ok = False
            if number:
                num = number.lower()
                ok = bool(re.match(r"^(?:%s\s+)?%s\b" % (re.escape(kind.lower()) if kind else "(?:chapter|section|part|appendix)", re.escape(num)), t)) \
                    or bool(re.match(r"^%s(?:\s|\.|$)" % re.escape(num), t))
            elif title_tokens:
                ok = title_tokens <= set(ask.tokens(text))
            if ok:
                best = j; break
        if best is None:
            return []
        ci, lvl, _, _ = self.headings[best]
        end = len(self.chunks)
        for ci2, lvl2, _, _ in self.headings[best + 1:]:
            if lvl2 <= lvl and ci2 > ci:
                end = ci2; break
        return list(range(ci, end))


_CACHE = {}


def _lsa(ix, k=96):
    """Latent semantic analysis of the document's own chunks (TF-IDF + truncated SVD). Lets a question find a
    passage that says the same thing in other words. No model, no download; skipped for very short documents."""
    n = len(ix.chunks)
    if n < 30:
        return None
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds
    except Exception:
        return None
    vocab = sorted(w for w, d in ix.df.items() if 2 <= d <= 0.5 * n and len(w) > 2)
    if len(vocab) < 20:
        return None
    index = {w: j for j, w in enumerate(vocab)}
    idf = np.array([math.log(n / ix.df[w]) for w in vocab], np.float32)
    rows, cols, vals = [], [], []
    for i, tf in enumerate(ix.tf):
        for w, f in tf.items():
            j = index.get(w)
            if j is not None:
                rows.append(i); cols.append(j); vals.append((1 + math.log(f)) * idf[j])
    X = csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n, len(vocab)))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()) + 1e-9
    X = csr_matrix(X.multiply(1 / norms[:, None]))
    kk = max(2, min(k, n // 4, len(vocab) - 1))      # well below full rank, so related words share dimensions
    try:
        U, S, Vt = svds(X, k=kk)
    except Exception:
        return None
    vecs = (U * S).astype(np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    return dict(vecs=vecs, proj=Vt.T.astype(np.float32), idf=idf, vocab=vocab, index=index)


def index_for(doc, key=None, folder=None, doc_mtime=None):
    if key is not None and key in _CACHE:
        return _CACHE[key]
    ix = DocIndex.load(folder, doc, doc_mtime) if folder else None
    if ix is None:
        ix = DocIndex(doc)
        if folder:
            try:
                ix.save(folder, doc_mtime)
                ix.folder = folder
            except Exception:
                pass
    if key is not None:
        if len(_CACHE) >= 3:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = ix
    return ix


def prepare(doc, folder):
    """Build and save the Ask index right after processing (a large document is indexed once, not on the first
    question)."""
    try:
        mt = os.stat(os.path.join(folder, "document.json")).st_mtime
        DocIndex(doc).save(folder, mt)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------ helpers
def _sentences(text):
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)
    t = re.sub(r"\[(?:figure|chart|diagram|unknown): .*?\]", " ", t, flags=re.S)
    t = re.sub(r"^#+\s.*$|^Table:(?:\n[^\n]*\|[^\n]*)*", " ", t, flags=re.M)
    out = [re.sub(r"^[•‣●○▪■∙·➢◦*-]\s*", "", s.strip()) for s in ask.sentences(re.sub(r"\s+", " ", t))]
    return [s for s in out if 40 <= len(s) <= 400 and _prose(s)]


def _prose(s):
    """A sentence worth quoting in a summary: mostly words (not figure labels, number runs or broken glyphs)."""
    if re.match(r"^(Figure|Fig\.|Table|Exercise|Problem)s?\s+[\dA-Z][\d.\-]*\b", s) or re.search(r"\([a-h]\)\s*\([a-h]\)", s):
        return False
    if re.search(r"[\ue000-\uf8ff\ufffd˘ˇ]", s):
        return False
    toks = s.split()
    words = [w for w in toks if re.fullmatch(r"[\"'“‘(]?[A-Za-z][A-Za-z’'\-]*[.,;:!?)\"”’]*", w)]
    return len(toks) >= 6 and len(words) >= 0.65 * len(toks)


def extractive_summary(ix, chunk_ids=None, n=8, outline=True):
    """Central sentences (TF-IDF similarity to the document centroid, spread across sections) + outline."""
    ids = chunk_ids if chunk_ids is not None else list(range(len(ix.chunks)))
    if len(ids) > 4000:
        step = len(ids) / 4000.0
        ids = [ids[int(k * step)] for k in range(4000)]
    cand = []
    for i in ids:
        for s in _sentences(ix.chunks[i]["text"]):
            cand.append((i, s))
    if not cand:
        return [], []
    vecs = []
    cent = Counter()
    for _, s in cand:
        v = Counter(ask.tokens(s))
        vecs.append(v)
        for w, f in v.items():
            cent[w] += f * ix.idf(w)
    norm_c = math.sqrt(sum(x * x for x in cent.values())) or 1.0
    scored = []
    for k, ((i, s), v) in enumerate(zip(cand, vecs)):
        dot = sum(f * ix.idf(w) * cent[w] for w, f in v.items())
        nv = math.sqrt(sum((f * ix.idf(w)) ** 2 for w, f in v.items())) or 1.0
        scored.append((dot / (nv * norm_c) + (0.08 if k < 3 else 0.0), k))
    scored.sort(reverse=True)
    picked, used_sec = [], Counter()
    for sc, k in scored:
        i, s = cand[k]
        sec = ix.chunks[i]["section"]
        if used_sec[sec] >= 2:
            continue
        if any(len(set(ask.tokens(s)) & set(ask.tokens(p[1]))) > 0.7 * max(1, len(set(ask.tokens(s)))) for p in picked):
            continue
        picked.append((i, s)); used_sec[sec] += 1
        if len(picked) >= n:
            break
    picked.sort(key=lambda t: t[0])
    if not outline:
        return picked, []
    sel = set(ids)
    if chunk_ids is None and len(ix.chapters) >= 5:
        # the document's own table of contents gives the best outline of a book
        return picked, [(f"{lab} {title}", page, 1) for lab, title, ci, page in ix.chapters[:45]]
    seen, outline = set(), []
    for ci, lvl, text, page in ix.headings:
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if lvl <= 2 and (chunk_ids is None or ci in sel) and key not in seen and len(key) > 2:
            seen.add(key); outline.append((text, page, lvl))
    if len(outline) > 25:
        # a long document: chapter-level headings, spread over the whole document
        top = [o for o in outline if o[2] == min(x[2] for x in outline)]
        outline = top if len(top) >= 8 else outline
        if len(outline) > 25:
            outline = [outline[int(k * len(outline) / 25)] for k in range(25)]
    return picked, outline


def _pages_str(pages):
    pages = sorted(set(p for p in pages if p))
    if not pages:
        return ""
    return f"p. {pages[0]}" if len(pages) == 1 else f"pp. {pages[0]}–{pages[-1]}"


def _excerpt(ix, i, limit=None):
    c = ix.chunks[i]
    t = c["text"] if limit is None else c["text"][:limit]
    fl = f" | extraction note: {'; '.join(c['flags'])}" if c.get("flags") else ""
    return f"[{_pages_str(c['pages'])}{' | ' + c['section'] if c['section'] else ''}{fl}]\n{t}"


CITE = re.compile(r"\[(pp?)\.\s*(\d{1,5}(?:\s*[–-]\s*\d{1,5})?(?:\s*(?:,|and|&)\s*\d{1,5}(?:\s*[–-]\s*\d{1,5})?)*)\]")


def check_citations(text, allowed):
    """Page citations in a model answer that point outside the pages it was given are removed (and reported), so
    a reader is never sent to a page that was not part of the evidence. In a list ([p. 3, 8]) only the pages
    outside the evidence are dropped."""
    allowed = set(allowed)
    removed = []

    def fix(m):
        keep = []
        for item in re.split(r"\s*(?:,|and|&)\s*", m.group(2)):
            r = re.match(r"(\d+)(?:\s*[–-]\s*(\d+))?$", item.strip())
            if not r:
                continue
            a = int(r.group(1)); b = int(r.group(2) or a)
            if any(p in allowed for p in range(a, min(b, a + 60) + 1)):
                keep.append(item.strip())
            else:
                removed.append(a if a == b else f"{a}–{b}")
        if not keep:
            return ""
        many = len(keep) > 1 or "–" in keep[0] or "-" in keep[0]
        return f"[{'pp' if many else 'p'}. {', '.join(keep)}]"
    out = CITE.sub(fix, text or "")
    return re.sub(r"[ \t]+([.,;:])", r"\1", out), removed


SYSTEM = """You are the ParseFlow document assistant. You help a reader understand ONE uploaded document.
You receive excerpts from the document (each starts with its page numbers in square brackets) and the question.

Rules:
1. Statements about what the document says must come from the excerpts. Put the page after each such statement, like [p. 41] or [pp. 41–42]. Use only page numbers that appear in the excerpt headers. Never invent quotes, numbers or pages.
2. Write the answer in Markdown with these parts:
   **From the document** — what the document says that answers the question, with page citations.
   **Explanation** — a clear explanation in simple language that may use general knowledge: define terms, give a small example, walk through steps (numbered) when that helps, and explain equations symbol by symbol. Make clear this part adds to the document.
   Keep short questions short. Leave out the Explanation only when it adds nothing.
3. If the excerpts do not answer the question, say so plainly first ("The document doesn't cover …"), then give a general explanation labelled as general knowledge. Never pretend the document says something it does not.
4. For a summary request: explain the CONTENT, never just list headings. Give **Overview** (what the document is about and why it matters), **Key ideas** (each explained in a sentence or two, with page citations), **Important formulas or definitions** (in LaTeX, each with what it means), **Examples** (if the document has them) and **Conclusions / takeaways**. Skip a part the material does not support.
5. Excerpt headers may carry an "extraction note" (for example mathematics rebuilt from glyph positions, OCR, a broken font map). When your answer depends on such text, say that the extracted text may be imperfect and check it against the page.
6. For questions that combine several parts of the document (compare, relate, why/how across sections), use excerpts from all the relevant pages and say how they connect.
7. Write mathematics in LaTeX: inline \\( … \\), display \\[ … \\]. Put code in fenced code blocks.
8. Follow-up questions refer to the conversation so far ("it", "that algorithm"): resolve them from the previous turns. Adapt to requests such as "simpler", "an example", "step by step", "derive it", "in more detail".
9. Reply in the language of the question."""


def _history(history, max_turns=6):
    out = []
    for h in (history or [])[-2 * max_turns:]:
        role = "assistant" if h.get("role") == "assistant" else "user"
        txt = str(h.get("content") or "")[:2500]
        if txt.strip():
            if out and out[-1]["role"] == role:
                out[-1]["content"] += "\n\n" + txt
            else:
                out.append(dict(role=role, content=txt))
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def _citations(ix, ids, answer_text=None):
    cited = set(int(x) for x in re.findall(r"\bpp?\.\s*(\d{1,5})", answer_text or ""))
    out = []
    for i in ids:
        c = ix.chunks[i]
        out.append(dict(pages=c["pages"], page=c["pages"][0], block_id=c["blocks"][0], section=c["section"],
                        snippet=re.sub(r"\s+", " ", re.sub(r"[#`\\\[\]]", " ", c["text"]))[:180].strip(),
                        cited=bool(cited & set(c["pages"]))))
    out.sort(key=lambda d: (not d["cited"], d["page"]))
    return out


# ------------------------------------------------------------------------------------------------ main entry
def _examples(ix, ids, q_tokens, limit=4):
    """Sentences of the retrieved passages (and the rest of their sections) that give an example."""
    qs = set(q_tokens)
    seen, out = set(), []
    pool = list(ids)
    for i in ids[:4]:
        sec = ix.chunks[i]["section"]
        pool += [j for j in range(max(0, i - 3), min(len(ix.chunks), i + 4)) if ix.chunks[j]["section"] == sec]
    for i in dict.fromkeys(pool):
        t = re.sub(r"\[(?:figure|chart|diagram|unknown): .*?\]|```.*?```", " ", ix.chunks[i]["text"], flags=re.S)
        for snt in [x.strip() for x in ask.sentences(re.sub(r"\s+", " ", t)) if 25 <= len(x.strip()) <= 500]:
            if EX_SENT.search(snt) and (not qs or qs & set(ask.tokens(snt))) and snt not in seen:
                seen.add(snt); out.append((i, snt))
                if len(out) >= limit:
                    return out
    return out


SECTION_SYSTEM = """You summarise ONE part of a document for a student, from excerpts of that part (each excerpt starts with its
page numbers in square brackets). Explain the content - do not list headings. Write 3-7 bullet points covering, when present:
the main ideas and why they matter, key definitions, important formulas (LaTeX, with what each symbol means), algorithms or
methods (what they do, briefly how), examples, and results or conclusions. End every bullet with its page citation like
[p. 41] using only pages from the excerpt headers. Use only the excerpts; do not add outside facts. No preamble."""


def sections(ix, max_sections=None):
    """Parts of the document a whole-document summary is built from: its own chapters (table of contents), else its
    top-level headings, else equal runs of passages. [(label, title, page, [chunk ids])]."""
    n = len(ix.chunks)
    out = []
    if len(ix.chapters) >= 5:
        for k, (label, title, ci, page) in enumerate(ix.chapters):
            end = ix.chapters[k + 1][2] if k + 1 < len(ix.chapters) else n
            out.append((label, title, page, list(range(ci, max(ci + 1, end)))))
    else:
        tops = [h for h in ix.headings if h[1] == min((x[1] for x in ix.headings), default=1)]
        if 3 <= len(tops) <= 60:
            starts = [0] + [h[0] for h in tops if h[0] > 0]
            for k, st in enumerate(starts):
                end = starts[k + 1] if k + 1 < len(starts) else n
                if end > st:
                    t = next((h for h in tops if h[0] == st), None)
                    out.append((str(len(out) + 1), t[2] if t else "Opening", ix.chunks[st]["pages"][0], list(range(st, end))))
        if not out:
            per = max(6, -(-n // 12))
            for st in range(0, n, per):
                ids = list(range(st, min(n, st + per)))
                pg = ix.chunks[st]["pages"][0]
                out.append((str(len(out) + 1), f"Pages {pg}–{ix.chunks[ids[-1]]['pages'][-1]}", pg, ids))
    if max_sections and len(out) > max_sections:                 # merge neighbours so every part is still covered
        per = -(-len(out) // max_sections)
        merged = []
        for k in range(0, len(out), per):
            grp = out[k:k + per]
            merged.append((grp[0][0], " / ".join(g[1] for g in grp)[:200], grp[0][2], [i for g in grp for i in g[3]]))
        out = merged
    return out


def _section_material(ix, ids, budget):
    """Excerpts that represent a part: its opening, its most central passages, and passages holding its formulas,
    tables and diagrams, in page order, within the budget."""
    if not ids:
        return []
    pick = list(ids[:2])
    central, _ = extractive_summary(ix, ids, n=6, outline=False)
    pick += [i for i, _ in central]
    pick += [i for i in ids if re.search(r"\\\[|^Table:|\[(?:diagram|chart):", ix.chunks[i]["text"], re.M)][:3]
    out, total = [], 0
    for i in sorted(dict.fromkeys(pick)):
        ex = _excerpt(ix, i, limit=1400)
        if total + len(ex) > budget:
            break
        out.append(ex); total += len(ex)
    return out


def _section_summaries(ix, parts, max_calls):
    """Map step of a document summary: one model summary per part from its real text (cached in the saved index),
    computed a few at a time."""
    from concurrent.futures import ThreadPoolExecutor
    key = lambda p: "|".join(("v3", p[0], p[1]))
    todo = [p for p in parts if key(p) not in ix.chapter_summaries][:max_calls]

    def run(p):
        label, title, page, ids = p
        mat = _section_material(ix, ids, int(os.getenv("ASK_SECTION_CHARS", "6000")))
        msg = f"Part {label}: {title}\n\nExcerpts:\n\n" + "\n\n---\n\n".join(mat)
        try:
            return key(p), llm.chat(SECTION_SYSTEM, [dict(role="user", content=msg)], max_tokens=500).strip()
        except llm.LLMError:
            return key(p), None
    new = 0
    if todo:
        with ThreadPoolExecutor(int(os.getenv("ASK_PARALLEL", "4"))) as ex:
            for k, v in ex.map(run, todo):
                if v:
                    ix.chapter_summaries[k] = v; new += 1
    if new and getattr(ix, "folder", None):
        try:
            ix.save(ix.folder, getattr(ix, "doc_mtime", None))
        except Exception:
            pass
    return [(p[0], p[1], p[2], ix.chapter_summaries.get(key(p))) for p in parts]


COMPARE = re.compile(r"\b(compare|comparison|differ(?:ence|ent)?|vs\.?|versus|relat(?:e|ed|ion|ionship)|connection|between|contrast|similar(?:ity|ities)?)\b", re.I)
STYLE = [(re.compile(r"\b(simpl(?:e|er|y)|eli5|like i'?m|beginner|easy|plain (?:words|english))\b", re.I),
          "Explain very simply, as to a first-year student: short sentences, an everyday analogy, a tiny worked example."),
         (re.compile(r"\b(step[- ]by[- ]step|walk me through|trace|dry run)\b", re.I),
          "Go step by step with numbered steps; trace a small concrete example through every step."),
         (re.compile(r"\b(deriv\w*|prove|proof|why does .* hold|show that)\b", re.I),
          "Give the derivation or proof in full, line by line, justifying each step; say which steps come from the document."),
         (re.compile(r"\b(more detail|in depth|elaborate|detailed)\b", re.I),
          "Give a detailed, thorough explanation."),
         (re.compile(r"\b(examples?|for instance)\b", re.I),
          "Include at least one concrete worked example (say whether it is from the document or your own).")]


def _parts_of(question):
    """'compare A and B' -> ['A', 'B'] so each side gets its own retrieval."""
    q = re.sub(COMPARE, " ", question)
    bits = [b.strip(" ?.,") for b in re.split(r"\band\b|\bvs\.?\b|\bversus\b|\bwith\b|,|\bor\b", q, flags=re.I)]
    return [b for b in bits if len(ask.tokens(b)) >= 1][:4]


def respond(doc, question, history=None, key=None, use_llm=True, folder=None, doc_mtime=None):
    question = (question or "").strip()[:2000]
    ix = index_for(doc, key, folder=folder, doc_mtime=doc_mtime)
    ix.doc_mtime = doc_mtime
    doc = ix.doc
    base = dict(question=question, mode=None, answer=None, citations=[], llm=llm.info(), note=None, intent=None)
    if not question:
        return dict(base, note="Ask a question about the document.")
    if not ix.chunks:
        return dict(base, note="This document has no extracted text to answer from.")
    history = history or []
    is_summary = bool(SUMMARY.search(question))
    pages = None
    m = PAGE_SCOPE.search(question)
    if m:
        a = int(m.group(1)); b = int(m.group(2) or a)
        if a > b:
            a, b = b, a
        pages = set(range(a, min(b, a + 60) + 1))
    sec_ids = None
    cm = CHAPTER_SCOPE.search(question)
    if cm:
        sec_ids = ix.section_chunks(number=cm.group(2), kind=cm.group(1)) or None
    # follow-ups inherit the previous question's terms and passages
    prev_q = next((h.get("content") for h in reversed(history) if h.get("role") == "user"), "")
    prev_ids = next((h.get("chunk_ids") for h in reversed(history) if h.get("role") == "assistant" and h.get("chunk_ids")), None) or []
    q = ask.tokens(re.sub(r"\bpages?\s*\d+(?:\s*(?:-|–|to|through)\s*\d+)?", " ", question, flags=re.I))
    follow = bool(history) and (len(set(q)) <= 2 or FOLLOW.search(question))
    if follow and prev_q:
        q = q + [w for w in ask.tokens(prev_q) if w not in q]
    phrases = ask._phrases(question) + (ask._phrases(prev_q) if follow and prev_q else [])

    # ---- choose passages
    if is_summary:
        base["intent"] = "summary"
        scope = sec_ids if sec_ids else ([c["id"] for c in ix.chunks if set(c["pages"]) & pages] if pages else None)
        picked, outline = extractive_summary(ix, scope, n=10)
        ids = sorted(set(i for i, _ in picked))
        if scope is not None and len(scope) <= 40:
            ids = list(scope)
    else:
        base["intent"] = "question"
        hits = ix.search(q, k=8, pages=pages, phrases=phrases)
        if COMPARE.search(question):
            # several parts of the document: retrieve for each side, then interleave so both are represented
            base["intent"] = "analysis"
            side_hits = [ix.search(ask.tokens(p), k=5, pages=pages, phrases=ask._phrases(p)) for p in _parts_of(question)]
            if len(side_hits) >= 2:
                mixed = [h for grp in zip(*[sh + [None] * (5 - len(sh)) for sh in side_hits]) for h in grp if h]
                hits = list(dict.fromkeys([h for h in mixed] + hits))[:12]
        if sec_ids:
            hits = [h for h in hits if h[1] in set(sec_ids)] or hits
        ids = [i for _, i in hits]
        if follow:
            ids = list(dict.fromkeys(list(prev_ids)[:4] + ids))
        if not ids and pages:
            ids = [c["id"] for c in ix.chunks if set(c["pages"]) & pages][:8]
        # neighbours give the passage its context (the sentence before a formula, the steps after a heading)
        nb = []
        for i in ids[:3]:
            for j in (i - 1, i + 1):
                if 0 <= j < len(ix.chunks) and ix.chunks[j]["section"] == ix.chunks[i]["section"]:
                    nb.append(j)
        ids = list(dict.fromkeys(ids + nb))

    # ---- language model
    if use_llm and llm.available():
        budget = int(os.getenv("ASK_CONTEXT_CHARS", "14000"))
        parts, used, total = [], [], 0
        if is_summary:
            digest = []
            whole = sum(len(c["text"]) for c in ix.chunks) if scope is None else sum(len(ix.chunks[i]["text"]) for i in scope)
            if whole <= budget:
                ids = list(range(len(ix.chunks))) if scope is None else list(scope)     # short: the model reads all of it
            else:
                # map-reduce: each part of the document is summarised from its own text, then combined here
                secs = sections(ix, int(os.getenv("ASK_SUMMARY_PARTS", "40"))) if scope is None else \
                    [("1", "Requested part", ix.chunks[scope[0]]["pages"][0], list(scope))] if len(scope) <= 40 else \
                    [(str(k + 1), f"Part {k + 1}", ix.chunks[scope[k * 30]]["pages"][0], scope[k * 30:(k + 1) * 30]) for k in range(-(-len(scope) // 30))]
                done = _section_summaries(ix, secs, int(os.getenv("ASK_SUMMARY_CHAPTERS", "40")))
                got = [(lab, t, p, sm) for lab, t, p, sm in done if sm]
                if got:
                    digest.append("Summaries of each part, in order (written from the part's own text, with page citations):\n\n" +
                                  "\n\n".join(f"Part {lab} — {t} (from p. {p}):\n{sm}" for lab, t, p, sm in got))
                    used_pages = {int(x) for _, _, _, sm in got for x in re.findall(r"\bpp?\.\s*(\d{1,5})", sm)}
                    for _, _, _, sm in got:
                        for a, b2 in re.findall(r"pp\.\s*(\d{1,5})\s*[–-]\s*(\d{1,5})", sm):
                            used_pages |= set(range(int(a), int(b2) + 1))
                    base["_pages"] = used_pages
            if not digest and outline:
                digest.append("Outline (headings):\n" + "\n".join(f"- {t} [p. {p}]" for t, p, _ in outline))
            if digest:
                digest.append("Key sentences:\n" + "\n".join(f"- {s} [{_pages_str(ix.chunks[i]['pages'])}]" for i, s in picked[:6]))
                parts.append("\n\n".join(digest)); total = len(parts[0])
        for i in ids:
            ex = _excerpt(ix, i, limit=2200)
            if total + len(ex) > budget:
                break
            parts.append(ex); used.append(i); total += len(ex)
        meta = f"Document: {doc.get('filename') or 'document'} ({ix.n_pages} pages)"
        style = [txt for rx, txt in STYLE if rx.search(question)]
        user = f"{meta}\n\nExcerpts:\n\n" + "\n\n---\n\n".join(parts) + f"\n\nQuestion: {question}" + \
            (("\n\nHow to answer: " + " ".join(style)) if style else "")
        try:
            text = llm.chat(SYSTEM, _history(history) + [dict(role="user", content=user)], max_tokens=int(os.getenv("ASK_MAX_TOKENS", "1600")))
            if text:
                allowed = {p for i in used for p in ix.chunks[i]["pages"]} | base.pop("_pages", set())
                text, removed = check_citations(text, allowed)
                out = dict(base, mode="llm", answer=text, chunk_ids=used or ids[:8], citations=_citations(ix, used or ids[:8], text))
                if removed:
                    out["note"] = ("Removed page reference(s) the model gave that were not among the passages it was shown: "
                                   + ", ".join(f"p. {r}" for r in removed) + ".")
                return out
        except llm.LLMError as e:
            base["note"] = f"The language model could not answer ({e}); showing passages from the document instead."
    base.pop("_pages", None)

    # ---- extractive fallback
    if is_summary:
        lines = []
        if scope is None and ix.digests and len(ix.chapters) >= 5:
            rows = []
            for label, title, ci, page in ix.chapters[:45]:
                d = ix.digests.get(label) or []
                rows.append(f"- **{label} {title}** *(p. {page})*" + (f" — {d[0][1]}" if d else ""))
            lines.append("**Chapters**\n" + "\n".join(rows))
            text = "\n\n".join(lines)
            cits = [i for label, *_ in ix.chapters[:45] for i, _ in (ix.digests.get(label) or [])[:1]]
            return dict(base, mode="extractive", answer=text, chunk_ids=cits[:12], citations=_citations(ix, cits[:12]),
                        note=base["note"] or "Summary built from the document's own contents and one key sentence per chapter "
                                              "(no AI model configured).")
        if outline:
            lines.append("**Outline**\n" + "\n".join(f"- {t} *(p. {p})*" for t, p, _ in outline[:15]))
        if picked:
            lines.append("**Key sentences**\n" + "\n".join(f"- {s} *({_pages_str(ix.chunks[i]['pages'])})*" for i, s in picked))
        text = "\n\n".join(lines) or "No summary could be extracted."
        return dict(base, mode="extractive", answer=text, chunk_ids=ids,
                    citations=_citations(ix, sorted(set(i for i, _ in picked))),
                    note=base["note"] or "Summary built from sentences quoted from the document (no AI model configured).")
    if EXAMPLE.search(question):
        base["intent"] = "example"
        topic = [w for w in q if w not in FOLLOW_SOFT and w not in ("example", "examples", "instance", "show")]
        exs = _examples(ix, ids, topic)
        if exs:
            text = "**Examples in the document**\n" + "\n".join(f"- {snt} *({_pages_str(ix.chunks[i]['pages'])})*" for i, snt in exs)
            return dict(base, mode="extractive", answer=text, chunk_ids=[i for i, _ in exs], citations=_citations(ix, [i for i, _ in exs]),
                        note=base["note"] or "Examples quoted from the document. A worked example in simple language needs an AI "
                                              "model (set ASK_PROVIDER — a local model through Ollama works offline).")
    ask_key = ("ask",) + tuple(key) if isinstance(key, tuple) else None
    own = [w for w in ask.tokens(question) if w not in FOLLOW_SOFT]
    eq = question if not follow else (prev_q if not own else question + " " + prev_q)
    r = ask.answer(doc, eq, key=ask_key)
    more = []
    primary_pages = {e.get("page") for e in (r.get("evidence") or [])}
    for i in ids:
        c = ix.chunks[i]
        if set(c["pages"]) & primary_pages and r.get("found"):
            continue
        more.append(dict(pages=c["pages"], page=c["pages"][0], block_id=c["blocks"][0], section=c["section"],
                         snippet=re.sub(r"\s+", " ", re.sub(r"[#`]", " ", c["text"]))[:260].strip()))
        if len(more) >= 3:
            break
    out = dict(base, mode="extractive", extract=r, more=more, chunk_ids=ids[:6],
               citations=_citations(ix, ids[:6]),
               note=base["note"] or ("Quoted from the document. Explanations in simple language need an AI model "
                                     "(set ASK_PROVIDER — a local model through Ollama works offline)."))
    if not r.get("found") and not more:
        out["answer"] = "I couldn't find this in the document."
    return out
