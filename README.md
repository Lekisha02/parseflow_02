# ParseFlow

**ParseFlow** is a document processing system developed by **Team DEADLOCK** for the **DQCL ParseAnything** problem statement (DataQuest 3.0).

It reads PDFs and images that mix digital text, tables, scans, handwriting, equations, diagrams and charts, uses the right extractor for each region, and returns a structured **Digital Twin** (JSON) plus **canonical Markdown**, keeping the evidence behind every result: page, bounding box, extractor, confidence, verification status and warnings.

> **Version 2** (this revision) rebuilt the extraction core for accuracy and speed while keeping the architecture, API and frontend: column-aware layout, running header/footer detection, table structure with merged cells, LaTeX from the PDF text layer, adaptive OCR preprocessing, backend Markdown, a stricter evidence-based verifier, a one-command CLI, a pytest suite and an official olmOCR-Bench harness. See [CHANGES.md](CHANGES.md) and [Benchmark results](#benchmark-results).

---

## Problem Statement

Most extraction systems work on clean digital text, but real documents mix digital text, tables, scanned pages, handwriting, handwritten code, equations, diagrams, charts and images. One technique for everything loses or invents information, and AI models can produce content that is not in the document. ParseFlow processes each kind of content with a suitable method and records where every result came from.

---

## Our Approach

```text
Document → Detect → Analyze → Route → Extract → Assemble → Verify → Structured Output (JSON + Markdown)
```

| Stage | What happens (v2) |
|---|---|
| **Detect** | File type, page count, encryption, size limits. Per page (no rendering): visible text amount, hidden OCR text layer, image coverage, **text-layer quality** (broken font maps and garbage OCR layers are detected). |
| **Analyze** | Character-level reading of the text layer (font, size, baseline, bbox for every glyph); display equations; ruled and borderless tables; figures; drop caps; split accents. Scanned pages: adaptive OCR. |
| **Route** | Per page and per region, cheapest suitable method first (table below). |
| **Extract** | Paragraphs, headings, lists, code, captions, tables (cell model), equations (LaTeX), figures (vision or flagged). |
| **Assemble** | Column-aware reading order, **document-level header/footer/page-number classification**, block ids, caption links. |
| **Verify** | Evidence-based status per block (independence rules below). |
| **Output** | Digital Twin JSON (backward compatible, additive fields) + canonical Markdown generated in the backend. |

### Routing

| Content | Method |
|---|---|
| Digital text (trustworthy text layer) | Native text layer (PyMuPDF), column-aware layout |
| Text layer present but broken (unmapped glyphs, implausible words) | Adaptive OCR; the text layer is kept as evidence |
| Scanned page with an embedded OCR layer | Embedded layer when plausible, otherwise Tesseract |
| Scanned page | Adaptive preprocessing + Tesseract |
| Photo/graphic page with a few real text lines | **Hybrid**: native lines kept, OCR for the rest |
| Ruled tables | Vector ruling grid → cells with rowspan/colspan |
| Borderless / booktabs tables | Column alignment + horizontal rules for header blocks |
| Equations (digital) | LaTeX rebuilt from glyph geometry (no model) |
| Equations (scanned) | Optional local recognizer; otherwise OCR text kept and flagged |
| Handwriting, diagrams, charts, figures | Vision model when configured; otherwise preserved and flagged |

Vision models are optional and never required. Everything else runs **offline without API keys**.

---

## Input formats (`processing/formats.py`)

| You upload | How it is read |
|---|---|
| PDF | directly (text layer, vector drawings, scans) |
| Word: DOCX, DOC, ODT, RTF | LibreOffice → PDF when installed (Word's own layout and pages). Without it, DOCX is rendered by ParseFlow from its content: headings, lists, tables with merged cells, images, sub/superscripts. DOC/ODT/RTF need LibreOffice. |
| PowerPoint: PPTX, PPT, ODP | LibreOffice → PDF when installed, plus a page with the **data of every chart** (read from the file, not from the drawn bars). Without it, PPTX is rendered slide by slide (text boxes, tables, pictures, chart data) with speaker notes after their slide. Slide titles become headings. |
| Excel: XLSX, XLSM, XLS, ODS, CSV, TSV | read **cell by cell** (every sheet, merged cells, number formats such as 25 %, dates); formulas show their saved results, are calculated with LibreOffice when the file has none, and otherwise show the formula marked "(not calculated)". Each sheet becomes a ruled table (`verified`, exact cell text). XLS/ODS need LibreOffice. |
| Images: PNG, JPG/JPEG, TIFF (multi-page), BMP, GIF, WebP, HEIC (with `pillow-heif`) | OCR; phone photos are turned upright from their EXIF orientation; each TIFF page is a page |
| Text, Markdown, HTML, JSON | typeset into a PDF (headings, lists, tables, code); scripts and remote images are left out |
| EPUB, XPS, FB2, MOBI, CBZ, SVG | converted by MuPDF |

The original is kept (`original_<id>_<name>`), the converted PDF is the document's source, and the document records `source: {format, converted_with}` and a warning saying how it was converted. A damaged file (a `.docx` that is not a Word file) is refused as corrupt, a renamed file is read as what it really is, and a format that needs LibreOffice says so (`PF-207`). In-browser mode (no server) reads PDFs and images only.

## What is extracted and how

### Digital text and reading order
Glyphs are read with their font, size and baseline. Lines are split at wide gaps; spaces are synthesised adaptively when a PDF positions words without space glyphs; ligatures are expanded; accents drawn as separate glyphs are composed (`a` + `´` → `á`); drop caps are re-attached.
Reading order uses a recursive XY-cut that prefers real text gutters, treats full-width elements (titles, running headers) as separators, and handles sidebars next to text columns. Hyphenated line breaks are rejoined only with evidence (the joined word occurs in the document, or the prefix is not a compound-forming prefix such as `self-`, `well-`, `non-`).

### Broken character maps (`processing/fontfix.py`, `processing/glyphshape.py`)
Many TeX-produced books (CLRS, for example) embed math fonts whose ToUnicode map is wrong: the text layer says `D` where the page shows `=`, `C` for `+`, `Œ`/U+FFFD for `[`/`]`, `.`/`/` for `(`/`)`, a private-use code for `−`, `4` for an em dash or `û` for the *fi* ligature. ParseFlow repairs these from the font itself, never by guessing from context:
1. the glyph's **name** in the embedded font (`equal`, `bracketleft`, `uni2212`) when the font has names;
2. otherwise the glyph's **outline** (Type 3 CharProcs or the TrueType/CFF outline) is rasterised and matched against bundled templates rendered from DejaVu, STIX and Liberation fonts in regular, italic and bold (`data/glyph_templates.npz`, 55 KB), scored by overlap plus blurred correlation (robust to stroke weight) and size/position metrics. Measured on fonts that are *not* in the templates (`python3 tools/eval_glyphs.py`, FreeSerif/FreeSans): **91.6 %** of glyphs reported with a broken code are identified correctly (414/452), and **0 of 700** correctly reported glyphs are changed. A reported letter is replaced only by a clearly different symbol; ordinary characters of fonts without signs of a broken map are never touched. A font is suspected when its map gives one character to several different glyphs.
Whatever cannot be repaired stays flagged (`review` plus a block warning), never silently marked `verified`.

### Running headers, footers and page numbers
A document-level classifier scores each edge-of-page line with independent signals and keeps the reasons: position (top/bottom band, extreme edge, rotated side-margin stamps), isolation from the body, **repetition across pages** (digits normalised), page-number patterns (`12`, `- 12 -`, `xii`, `Page 3 of 20`), publication boilerplate (URLs, DOI, ©, "Downloaded from", journal/volume/issue, revision/part numbers, dates), small font, rows of short items spread across the line, items sharing a line with a page number, and content that ends far from the paper edge. Protective signals keep one-off titles, footnotes, captions and section headings. A structural signal (isolation, repetition, page number, margin, edge row) is required before anything is removed; a fixed margin alone never removes text.
Removed items stay in the JSON (`role`, `furniture.score`, `furniture.reasons`, `furniture.excluded`). Removal is configurable (`PARSEFLOW_REMOVE_FURNITURE=0`, CLI `--include-headers`, API `?include_furniture=1`, frontend toggle).

### Tables
* **Ruled tables**: PyMuPDF `find_tables()` runs only inside regions where vector rules form a grid; cell geometry is mapped onto row/column edges so merged cells get real `rowspan`/`colspan`.
* **Borderless/booktabs tables**: consecutive rows that split into aligned fragments; columns from x-coverage, plus whitespace "rivers" (e.g. a `#` column next to a text column); rejection of multi-column prose (long fragments, high row coverage, misaligned cell edges) and leader dots; category rows inside tables are allowed.
* **Header blocks** above a table (e.g. *Year Ended* over date columns, rotated 45° column labels, group labels such as *ReACT*) are attached by column alignment; a rule under a header cell defines its span.
* Currency-only columns are folded into their values (`$178`), nothing is dropped.
* **Plausibility checks** reject table candidates that are really something else, with the reason kept in the page notes: a picture or vector diagram inside (curves, slanted lines), display mathematics or aligned equations, tables of contents and book indexes (become lists), numbered pseudo-code (becomes a code block), figure panels `(a) (b)`, coloured box diagrams, regions captioned as a figure. Running headers, lead-in sentences and section labels caught at a table's edge are trimmed off. Rejected diagrams become figures (Visuals tab).
* **Wrapped cells**: a cell whose text wraps onto a second line (or is vertically centred against a taller neighbour) is joined to its own row instead of creating a new row.
* **Tables continued on the next page** (same columns, the header repeated or not) are joined into one table; the first part records `merged`.
* Output: Markdown for rectangular tables with one header row, **HTML with `<th>`, `rowspan`, `colspan`** otherwise. Pipes are escaped, in-cell line breaks normalised, empty cells kept empty (`filled: false`). Values are never invented.

### Equations
On digital PDFs the LaTeX is rebuilt from the text layer: Unicode→LaTeX symbol table (Greek, operators, relations, arrows, ℝ/ℕ, math alphanumerics), TeX font families (`\mathbb`, `\mathcal`, `\mathfrak`, `\mathbf`), sub/superscripts from baseline offsets and sizes, fractions from vector rules, big operators with limits, radicals with overbars, `\operatorname` for upright function names, multi-line displays split into separate equations, equation numbers separated. Big delimiters are identified from their glyph shape and true extent (also when drawn from pieces) and become `\left( … \right)`; a grid between tall brackets becomes a **matrix** (`pmatrix`/`bmatrix`/`vmatrix`), a left brace with value/condition rows becomes **`cases`**, integral limits are attached, multi-line derivations aligned on their relation become **`aligned`**, dot stacks become `\vdots`/`\cdots`, primes are `'`, numbers are joined (`28.1`, not `2 8 . 1`) and a small raised number after a full stop is a footnote marker, not an exponent. Display math is emitted as `\[ … \]`, inline math as `\( … \)` (token-level detection: italic words, footnote markers, Greek-language words, **URLs, e-mail addresses, DOIs**, quoted words whose quotes a font maps to `<` `=` and words such as *Θ-notation* are not wrapped; an index expression such as `A[1 : i − 1]` is kept whole). Glyphs without a Unicode mapping lower the confidence and are reported instead of guessed. Reconstructed structure is marked `review`, never `verified`. Original glyph text is kept in `raw_text`.

### Charts (`processing/charts.py`)
Bar and line charts drawn as **vector graphics** (most charts exported from Excel, matplotlib, R or PowerPoint) give their data without any model: the numeric tick labels define the value axis (least-squares fit, rejected unless it fits closely), bars are measured against it and grouped into series by colour, line vertices are read the same way, categories come from the axis labels and series names from the legend. The chart block gets `data` (a category × series table) and stays `review` (measurements, not printed values). Chart *images* need a model (below).

### Local models (optional, `processing/engines.py`)
Off by default, local only, never downloaded; a missing model is reported in `/api/health` and the region stays flagged.
* `PARSEFLOW_OCR_ENGINE=paddleocr | surya | command` replaces Tesseract for scanned pages (command: any executable that prints words with boxes as JSON).
* `PARSEFLOW_HANDWRITING=trocr | command` reads handwriting regions and handwriting-only pages.
* `PARSEFLOW_CHART_MODEL=deplot | command` turns chart images into a data table.
* Weights go in `PARSEFLOW_MODEL_DIR` (`paddleocr/`, `trocr/`, `deplot/`). The built-in adapters follow each project's documented local-inference API; they could not be run here (no model downloads in the build environment) — the command plugins are covered by tests.

### OCR for scans (`processing/ocr.py`)
1. **Measure**: contrast (ink vs background), background unevenness, speckle (full resolution), skew (projection profile, applied only when the estimate is clearly better than 0°), character height.
2. **Apply only what the measurements call for**: illumination normalisation, contrast stretch, median despeckle, deskew, upscaling of small print, Sauvola binarisation.
3. **Bounded fallback**: if the result is weak, one alternative representation (the conservative "minimal" image if risky steps were used); never more than 2 attempts by default; OSD rotation only on request.
4. Pages are rendered at the scan's native resolution (bounded), never upsampled needlessly. Results are cached by image hash. Noise lines from textures/photos are filtered; handwriting-only pages keep isolated words, flagged.
5. Several scanned pages are OCR'd at once (`PARSEFLOW_OCR_WORKERS`, default up to 4): about 2× faster on 2 vCPUs.

### Verification (stricter than v1)
* `verified` requires **independent evidence**, or content read directly from the document's encoded text layer (no recognition step) with passing quality checks — that basis is recorded as *document-encoded*, not as independent.
* Outputs derived from the same extraction never count as independent (table cells are not "verified" by the text layer they were read from; two Tesseract passes are not independent).
* OCR alone is `review`; it becomes `verified` when an independent recogniser (the PDF's embedded OCR layer) agrees. Model output is `review` unless independently confirmed (e.g. table cells found in the text layer, OCR token overlap).
* Ruled tables are `verified` (grid geometry is independent of the text); borderless tables are `review` (structure inferred).
* Every block carries `evidence` (short string, v1-compatible) and `evidence_detail` (`basis`, `independent`, `signals`, `confidence_method`), plus `raw_text` and `transforms` whenever normalisation/reconstruction was applied. Statuses: `verified`, `review` (uncertain), `failed`.

---

### Ask assistant (`processing/assistant.py`, `processing/llm.py`)
The Ask tab is a chat. Each question is answered from the **whole document**: the document is split into passages (≈1,100 characters, kept within their section), and the passages that explain the question are retrieved with BM25 plus section-heading, phrase and definition bonuses and their neighbouring passages — not the first place a word occurs. Follow-up questions ("explain it more simply", "give an example") reuse the previous question's terms and passages; the last 12 turns are sent as conversation context. "Summarise the document" (or *chapter 2*, *part II*, *pages 40–60*) builds an outline — from the document's own table of contents when it has one, otherwise from its headings — plus key sentences from every part of the document.
* **With a language model configured** (`ASK_PROVIDER`, see below) the answer is written by the model from the retrieved passages, in two clearly separated parts: **From the document** (facts from the passages, each with `[p. N]` citations that jump to the page) and **Explanation** (simple language, examples, step-by-step working, general knowledge — marked as not coming from the document). Anthropic, Gemini, any OpenAI-compatible API, or a **local model through Ollama / LM Studio / llama.cpp (fully offline, no key)**. No training is needed: the model reads the relevant passages at question time.
* **Without a model**, or if the model fails, the answer falls back to the passages themselves (quoted, with page numbers) and says so; nothing is generated. "Give an example" quotes the document's own examples; a book summary gives one line per chapter, taken from the chapter's opening.
* **Retrieval by meaning**: besides keywords, passages are compared in a concept space learned from the document itself (latent semantic analysis with pseudo-relevance feedback), so a passage that says the same thing in other words is found. Optionally a neural embedding model adds to this (`ASK_EMBED_MODEL`, any OpenAI-compatible `/embeddings` endpoint; `nomic-embed-text` through Ollama works offline).
* The search index is built once, when the document is processed, and saved next to it (`ask_index.json`, `ask_lsa.npz`); conversations are saved with the document and come back when it is reopened. With a model, a whole-book summary is made from one short summary per chapter (cached), then combined.
The older single-shot extractive engine (`processing/ask.py`, `GET /api/ask/{id}?q=`) is kept and still used for the passage quotes:
answers are **copied from extracted blocks, never generated**. The question is matched with BM25 over blocks plus the heading each block sits under, phrase and definition bonuses (`... is called ...`), and stemming that handles `debugging`/`debug`. Four strategies: *page* ("What does page 4 explain?"), *table cell* (row label × column header), *section* (the question names a heading → the heading with its content) and *passage* (the best sentences, a code step such as `// Right Heavy Case` with the lines under it, a term with the description beside it, or a passage the layout split across blocks joined back together). Math stays LaTeX with `\( \)` delimiters and the viewer renders it with a vendored KaTeX (`frontend/vendor/katex`, works offline); code keeps its line numbers and indentation; tables are rendered as tables. Every answer lists its evidence blocks with **View Source**.

### Ask about this block (`processing/block_ask.py`) and diagram understanding (`processing/diagrams.py`)
In the Inspector, directly under a block's **extraction confidence**, *Ask about this block* opens a conversation about that one block (paragraph, equation, table, code, figure or diagram). The block's page stays highlighted in the viewer while you ask, follow-ups keep the block's context, and the thread is saved per block (`GET /api/chat/{id}?block=…`). The Visuals tab has an *Ask about this* button on every card.

What the model receives, as separate, labelled stages so an error can be traced to where it happened:
1. **Extraction** — the block as extracted (text, LaTeX, table rows, chart data), its extraction status and warnings. This confidence is about how the block was *read*; it is shown apart from the answer and never presented as the answer's confidence.
2. **Structure** (for vector figures) — read from the PDF's drawing commands, not guessed from pixels: circles/boxes/diamonds become nodes with the text inside them, lines and arrowheads become directed or undirected edges (polylines, curved edges, self-loops), text beside an edge becomes its label (weights, yes/no, transition symbols), text at an arrow's tail becomes a pointer (`S.top`, `Q.head`, `L.head`), rows/columns of boxes become arrays, stacks and queues (with indices, fill colours and Gantt time boundaries), boxes of 2–3 fields chained by arrows become linked lists (NIL slashes kept). Then it is interpreted: **tree** (root, children left/right, leaves, height, pre-/in-/post-/level-order, max-/min-heap and BST checks; trees drawn with child *and* parent pointers), **automaton** (start state from the entry arrow, accepting states from double circles, transitions; only when most edges carry a symbol), **flowchart** (start, steps, decisions with their branch labels, unreachable steps), **graph** (vertices, edges, weights, degrees) and **sorting traces** (which cells changed between consecutive rows). Shapes with no readable text are reported as unlabelled — never named.
3. **Computed facts** — exact answers code can give from the structure for the question asked: an automaton run on the string in the question (accept/reject with every state set), a shortest path between the two vertices named (Dijkstra, or Bellman-Ford with negative weights), connectivity, cycles.
4. **Context** — the caption, the text around the block on its page, other pages that refer to the figure ("Figure 22.1"), and related passages from anywhere in the document.
5. **The image** of the original region (cropped from the PDF at up to 1600 px) when the model accepts images (`ASK_VISION`), so raster figures, scans and handwriting are seen too.

The model explains; page references in its answer that are not among the evidence are removed and reported. Without a model, the answer is the evidence itself: what the drawing shows, the computed facts first, the caption and the pages that refer to it.

Measured on *Introduction to Algorithms* figures (vector drawings): the heap (tree + array with indices, max-heap property), stack and queue arrays (cells, `S.top`/`Q.head`/`Q.tail` pointers, colours, the change between states), two BSTs and the same trees drawn with parent pointers, red-black trees with NIL leaves, the adjacency-list figures 22.1/22.2 (every list and every edge, including the self-loop on vertex 6) and the weighted Bellman-Ford graphs were read correctly; a negative weight was read once as "34" instead of "–4" (a text-layer glyph issue). Regression tests draw each structure with known answers (`tests/test_diagrams.py`). A hand audit of random CLRS figures (`bench/diagram_audit_v25.md`) found, on 10 held-out figures before tuning: 3 correct, 1 mostly correct, 4 partial (missing or garbled labels, a highlight drawn as an extra edge), 2 wrong (Venn diagrams — overlapping shapes are now left uninterpreted).

### Diagrams that are images (`processing/diagrams_raster.py`)
Scanned pages, photos, screenshots and pictures inside PDFs get the same structure, read from the pixels: closed outlines and filled discs/boxes (by colour) are nodes, the remaining strokes are edges (crossing lines are split into the straight pieces ink covers), extra ink where a line meets a node is an arrowhead, an empty lobe on a node is a self-loop, two concentric outlines are an accepting state, a diagonal stroke in a box is NIL, and when no shapes are drawn the words joined by lines are the nodes (recursion trees). The text inside each node is OCR'd crop by crop. The block is `raster_diagram`, `review`, with an *estimated* confidence. When a vision model is configured, its own list of nodes and edges is kept next to it and compared edge by edge (`agreement`: "k of n edges in the model's reading are also in the image geometry").

Measured on all 153 vector figures of *Introduction to Algorithms*, rendered to 150-dpi images and read back without the PDF's drawing commands (`tools/eval_raster_diagrams.py`; the vector reading is the reference):

| 154 figures | node labels found | edges found (recall) | edges correct (precision) | direction right (of edges found) |
|---|---|---|---|---|
| geometry only (the PDF's own words given) | 70.6 % | 55.1 % | 93.9 % | 54 % |
| end to end with Tesseract OCR | 41.3 % | 18.9 % | 38.3 % | 45 % |

So the geometry is mostly right when it reports something, and OCR of node names (single italic letters, subscripts such as v₁) is the main source of misses: a misread name also makes its edges count as wrong. Clean figures (the synthetic tree, DFA and weighted graph in `tests/test_diagrams_raster.py`) are read completely, including with OCR; busy figures with crossing coloured curves are not. Median time per figure: 0.16 s without OCR, 1.1 s with it. Pages that are photos or text are recognised and skipped; picture-heavy documents have a time budget (`PARSEFLOW_RASTER_SECONDS`, 20 s).

### More diagram kinds
* **B-trees** — nodes holding several keys, each internal node with one child more than it has keys: keys per node, the keys in order, and a check of the B-tree order (sorted keys, every subtree between its separating keys, all leaves at one depth). All CLRS chapter 18 trees pass.
* **Huffman / prefix-code trees and tries** — labels on the edges are read along each root-to-leaf path (`codes` for 0/1 edges, `paths` otherwise); an unreadable edge label shows as `?`.
* **ER diagrams** — entities (boxes; double box = weak), relationships (diamonds, with the cardinality written beside each line), attributes (ellipses; underlined = key, double = multi-valued, attributes of attributes).
* **UML class diagrams** — boxes split by full-width lines into name / attributes / operations; outlined triangles are generalisation, outlined/filled diamonds aggregation/composition.

### Ask: summaries, comparisons and grounding
* A document that fits the context budget is sent whole. A longer one is summarised **map-reduce**: each part (the document's own chapters, else its top-level headings, else page windows) is summarised by the model from that part's real text — opening, most central passages, and passages with formulas, tables and diagrams — a few at a time (`ASK_PARALLEL`), cached with the index, and the part summaries are combined into an explanation of the content (overview, key ideas, formulas, examples, conclusions), not a list of headings.
* Questions that compare or relate things ("compare A and B", "difference between", "how is X related to Y") retrieve passages for each side and interleave them, so both are represented.
* "Explain simply", "step by step", "derive/prove", "in detail", "with an example" each add an instruction on how to answer.
* Passages carry extraction notes (OCR text, reconstructed maths, inferred table structure, figure labels only) so the model is told when the text itself may be wrong.

### Long documents
* **Opening**: after processing, ParseFlow also writes `document.json.gz`, a light copy without per-block provenance details (`document.lite.json.gz`), the blocks per page (`blocks.jsonl` + index) and the Markdown (`document.md`). The viewer opens a large document (JSON > `PARSEFLOW_LITE_MB`, default 8 MB) from the light copy and loads a page's provenance details when a block is inspected and the Markdown when its tab is opened. Cormen: **1.5 MB** to download instead of 25 MB; the viewer opens in about 2 s.
* **Resuming**: every finished slice of pages is saved (`checkpoint/`). A job that stops — worker crash, out of memory, server restart — continues from the last saved slice; the result is identical to an uninterrupted run (tested). A worker that dies or runs out of memory is retried automatically with smaller slices (`PARSEFLOW_RETRIES`, default 1); after a restart, interrupted documents are queued again.
* **Memory**: each worker has a memory cap (`PARSEFLOW_WORKER_MEMORY_MB`; `auto` = 80 % of the machine's or container's memory shared by the workers, at least 1.5 GB). Hitting it fails only that worker (`PF-507`), then the retry above.
* **Queue**: `PARSEFLOW_WORKERS` documents are processed at once; the rest wait, and the processing screen and library show their place in line.

There is **no page or size limit** by default (`MAX_PAGES=0`, `MAX_UPLOAD_MB=0`). Each document is processed in its own worker process (`PARSEFLOW_ISOLATE=1`): if a document exhausts memory or crashes the worker, only that document fails (`PF-503`) and the server keeps running; `PARSEFLOW_JOB_TIMEOUT` optionally stops very long jobs (`PF-504`). Measured: the 1,312-page *Introduction to Algorithms* (12 MB) processed through the API in about 3 minutes on 2 vCPUs (JSON ≈ 25 MB).
A 400-page PDF is processed in slices of `PARSEFLOW_PAGE_CHUNK` pages: each slice is analysed, routed and extracted, then its glyph data and page bitmaps are released; the processing screen shows `page N of M` for each stage. Digital pages are not rasterised during processing — the viewer requests page images and thumbnails on demand (lazy-loaded), and the long lists in the viewer (Blocks, Tables, Visuals, JSON) render in pages with *Show more*. Measured on a 400-page, 13 MB mixed test PDF (398 digital pages, 6,644 blocks, 2 vCPU container): peak memory **362 MB** (was 3.7 GB with every page bitmap held), API upload → viewer open **71 s**, document JSON 9 MB. Scanned pages are OCR'd one by one and take far longer than digital pages.

## Digital Twin (JSON)

All v1 fields are unchanged. Example block:

```json
{
  "block_id": "eq_1_01", "type": "equation", "content": "λg = ∑ s∈S ζ ψ(gs) n", "page": 1,
  "bbox": [412, 288, 794, 364], "reading_order": 2, "confidence": 0.8,
  "confidence_type": "heuristic (estimated)", "extractor": "digital_text_parser", "verification": "review",
  "relationships": [], "warnings": [],
  "latex": "\\lambda_{g} = \\sum_{s \\in S} \\zeta_{n}^{\\psi ( gs )}", "math_display": true, "reconstructed": true,
  "raw_text": "λg = ∑ s∈S ζ ψ(gs) n", "transforms": ["big operator limits reconstructed", "script placement from baseline offsets"],
  "evidence": "text-layer quality 1.00; LaTeX structure reconstructed from glyph geometry (inferred, not encoded)",
  "evidence_detail": {"basis": "document-encoded text layer (no recognition step)", "independent": false, "signals": ["…"], "confidence_method": "text-layer quality checks"}
}
```

Additive document fields: `markdown`, `schema_version: 2`, `timings` (per stage, seconds), `pages[].route`, `pages[].route_reason`, `pages[].text_quality`, `pages[].ocr` (preprocessing steps, attempts, metrics), `pages[].seconds`, `stats.furniture_excluded`, `engines.math_recognizer`, `config`. Table blocks add `cells` (row, col, rowspan, colspan, text, bbox, header, filled), `header_rows`, `table_method`, `markdown`.

---

## Installation

Requirements: Python ≥ 3.10, the Tesseract binary (for scans), Linux or macOS (Windows works via `run.bat`). Optional: LibreOffice (`brew install --cask libreoffice` · `sudo apt install libreoffice`) for Word/PowerPoint's exact layout and for DOC, PPT, XLS, ODT, ODP, ODS and RTF files.

```bash
# system OCR engine
sudo apt install tesseract-ocr        # Debian/Ubuntu
brew install tesseract                # macOS

git clone https://github.com/ADII6769/PARSERFLOWS.git
cd PARSERFLOWS
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt               # app + CLI
pip install -r requirements-dev.txt           # tests and benchmark tooling (optional)
```

Or install as a package (provides the `parseflow` command): `pip install -e ".[server]"`.

### Web app
```bash
bash run.sh          # Linux/macOS  (run.bat on Windows)  → http://localhost:8000
```

**Screens.** The landing page has the logo top-left, *Try Live Demo* and *Log in / Sign up* top-right, an auto-playing feature tour (7 slides, pauses on hover, arrows/dots to step; it replaces the old "Present 90s demo" button), the upload card, and — further down — features, the seven stages and a drop zone. After an upload, the processing screen and the document viewer are exactly the v2 screens.

**Guests and accounts.** Without an account, a file is processed and opened in the viewer, and deleted from the server after `PARSEFLOW_JOB_TTL_HOURS`. With an account (email + password, stored on this server only — no third-party login), every upload is also saved to *My documents*: a workspace with a left sidebar (upload, My documents, Needs review, recent files, Settings, Privacy & data, storage used, log out), filters, search by name, a preview panel (first page, summary, open in viewer, Markdown/JSON download, rename, delete) and account settings (name, password, delete account). *New document* in the viewer returns guests to the landing page and logged-in users to their library. Passwords are salted scrypt hashes; sessions are random tokens in an HttpOnly, SameSite=Lax cookie (only their SHA-256 is stored); saved documents are readable only by their owner. Set `PARSEFLOW_SECURE_COOKIE=1` when the app is served over HTTPS.

### One-command CLI (offline, no API keys)
```bash
./parseflow input_pdfs/ --output-dir output_markdown/
# or, after `pip install -e .`:
parseflow input_pdfs/ --output-dir output_markdown/
```
| Option | Meaning |
|---|---|
| `-r/--recursive` | search folders recursively (sub-folders are mirrored, so names never collide) |
| `-w/--workers N` | parallel worker processes (default min(4, CPUs)) |
| `-t/--timeout S` | per-document timeout (the worker is killed, the batch continues) |
| `--json` | also write the Digital Twin `<name>.json` |
| `--report r.json` | machine-readable summary of every file |
| `--log-json` | one JSON line per finished document on stdout |
| `--include-headers` | keep running headers/footers/page numbers |
| `--plain` | no HTML-comment warnings in the Markdown |
| `--skip-existing`, `--images` | skip done files; also convert PNG/JPG |

Each document runs in its own process (a malformed PDF cannot take the batch down); successful outputs are written atomically and kept when other files fail; the exit code is `1` if any file failed, `0` otherwise, `2` for usage errors.

### Python
```python
import sys; sys.path.insert(0, "backend")
from processing.runner import convert
doc = convert("paper.pdf")          # Digital Twin dict
print(doc["markdown"])
```

---

## API (FastAPI)

| Endpoint | Description |
|---|---|
| `GET /api/health` | backend status, OCR engine, vision config, math recognizer, PyMuPDF version, limits |
| `POST /api/process` | upload (PDF/PNG/JPG; type checked before storing; size limit `MAX_UPLOAD_MB`); returns `document_id, filename, size, type, pages` |
| `GET /api/status/{id}` | `status`, `stages`, `error`, **`timings`** |
| `GET /api/document/{id}` | Digital Twin JSON (now includes `markdown`) |
| `GET /api/document/{id}/markdown` | **new**: canonical Markdown; `?include_furniture=1` restores headers/footers, `?annotate=0` drops HTML-comment warnings |
| `GET /api/page/{id}/{n}.png` | page image for the viewer; rendered on first request and cached (digital pages are not rasterised during processing) |
| `GET /api/thumb/{id}/{n}.jpg` | **new**: small JPEG thumbnail for the page strip, rendered on demand and cached |
| `POST /api/auth/signup` · `POST /api/auth/login` · `POST /api/auth/logout` · `GET /api/auth/me` | **new**: accounts; JSON bodies `{name, email, password}` / `{email, password}`; session cookie `pf_session` |
| `GET /api/my/documents` · `PATCH /api/my/documents/{id}` · `DELETE /api/my/documents/{id}` | **new**: the logged-in user's library (status, live progress, per-document stats, usage), rename (`{filename}`), delete |
| `PATCH /api/account` · `POST /api/account/password` · `POST /api/account/delete` | **new**: change name, change password (signs out other sessions), delete account and all its documents |
| `POST /api/ask/{id}` | **new**: Ask assistant; body `{question, history:[{role, content, chunk_ids?}]}`; returns `mode` (`llm` or `extractive`), `answer` (Markdown with `[p. N]` citations and LaTeX), `citations` (page, section, excerpt, cited), `chunk_ids` (send back with the next turn's history), `extract` (the quoted passages), `intent` (`question` · `summary`), `note`, `llm` (provider, model) |
| `GET /api/chat/{id}` · `DELETE /api/chat/{id}` | the saved Ask conversation of a document (owner only) / start a new one |
| `POST /api/llm/test` | **new**: one short real request to the configured AI model (and, if it takes images, a picture of a digit): `{ok, provider, model, seconds, vision, vision_ok, error}`. The Ask tab's *Test AI* button calls it |
| `POST /api/ask/{id}/block/{block_id}` | **new**: ask about one block; body `{question, history}`; returns `answer`, `mode`, `block` (id, page, type), `extraction` (status, confidence, method, warnings — kept separate from the answer), `structure` (what the drawing shows), `computed` (exact facts), `used_image`, `citations`, `note`. `GET`/`DELETE /api/chat/{id}?block={block_id}` read / clear that block's thread |
| `GET /api/document/{id}?lite=1\|auto` | the light copy (no `evidence_detail`, `raw_text`, `transforms`, Markdown); `auto` = only when the JSON is large. Sent gzip-compressed when the client accepts it (also the full JSON) |
| `GET /api/document/{id}/blocks?page=N` · `?pages=a-b` | full blocks of one page or a range (≤ 50 pages) |
| `GET /api/document/{id}/access` · `POST /api/document/{id}/share` | who may open it; owner turns the read-only share link on/off (`{shared: true\|false}` → `{shared, link}`) |
| `POST /api/auth/verify` · `POST /api/auth/verify/resend` | confirm the email address with the emailed token (logs the browser in) / send the link again |
| `POST /api/auth/reset/request` · `POST /api/auth/reset` | email a password-reset link (same answer whether or not the address has an account) / set the new password with the token (signs out every other session) |
| `GET /api/ask/{id}?q=...` | Ask Your Document (single-shot, extractive) — extractive answer (`parts` typed as text with inline LaTeX, math, code, list, table, heading), `strategy` (page · table cell · section · passage), `evidence` with block ids, `confidence`, matched/missing terms; `found=false` with the closest passages when nothing matches |

Uploads that fail validation are deleted immediately; guest documents expire after `PARSEFLOW_JOB_TTL_HOURS` (default 24); documents uploaded while logged in are kept until their owner deletes them. Document routes (`/api/document`, `/markdown`, `/blocks`, `/status`, `/page`, `/thumb`, `/ask`, `/chat`) answer 404 unless the request comes from the document's owner — the account that saved it, or for a guest upload the browser that uploaded it (a private `pf_guest` cookie) — or the owner has turned its share link on (read-only; a viewer's questions are not saved to the owner's conversation). `PARSEFLOW_GUEST_LINKS=open` restores the old behaviour (anyone with the id). The frontend renders the backend Markdown (with an *include running headers/footers* toggle) and keeps its own `md()` only for demo/browser mode.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VISION_PROVIDER`, `VISION_API_KEY`, `VISION_MODEL`, `MAX_VISION_REGIONS` | gemini, –, provider default, 12 | optional vision model (never required) |
| `MAX_PAGES` | 0 | page limit per document; 0 = no limit |
| `MAX_UPLOAD_MB` | 0 | API upload limit; 0 = no limit |
| `PARSEFLOW_ISOLATE`, `PARSEFLOW_JOB_TIMEOUT` | 1, – | process each document in its own worker process (crash isolation); optional per-document time limit in seconds |
| `ASK_PROVIDER` | – | Ask assistant model: `anthropic` · `gemini` · `openai` (any OpenAI-compatible server) · `ollama` (local, offline) · `none`. Empty: reuse the vision provider when `VISION_API_KEY` is set |
| `ASK_API_KEY`, `ASK_MODEL`, `ASK_BASE_URL` | –, provider default, – | key (falls back to `VISION_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY`), model (claude-sonnet-5-5, gemini-2.5-flash, gpt-4o-mini, llama3.1), OpenAI-compatible endpoint (Ollama default `http://localhost:11434/v1`) |
| `ASK_TIMEOUT`, `ASK_CONTEXT_CHARS`, `ASK_MAX_TOKENS` | 120, 14000, 1600 | model timeout (s), document text sent per question, answer length |
| `ASK_VISION` | auto | send the image of a selected visual block to the model: `auto` = on for Anthropic and Gemini and for OpenAI's own API, off for other OpenAI-compatible servers (set `1` for a local vision model, e.g. `llava` through Ollama); `0` = never |
| `ASK_SECTION_CHARS`, `ASK_PARALLEL`, `ASK_SUMMARY_PARTS` | 6000, 4, 40 | text per part in a map-reduce summary, part summaries made at once, maximum number of parts |
| `PARSEFLOW_DIAGRAMS` | 1 | read the structure of vector diagrams (0 = keep figures as label text only, as before v2.5) |
| `PARSEFLOW_RASTER_DIAGRAMS` | 30 | picture regions per document whose diagram structure is read from the pixels |
| `PARSEFLOW_OFFICE` | auto | `auto` = LibreOffice when installed, else ParseFlow's own DOCX/PPTX renderers; `libreoffice` or `native` to force one |
| `PARSEFLOW_SOFFICE`, `PARSEFLOW_CONVERT_TIMEOUT` | found automatically, 180 | path to LibreOffice's `soffice`; seconds per conversion |
| `PARSEFLOW_PAGE_CHUNK` | 25 | pages analysed + extracted per slice; memory is bounded by the slice, not the document |
| `OCR_LANG` | eng | Tesseract language(s), e.g. `eng+deu` |
| `OCR_TIMEOUT` | 60 | seconds per Tesseract call |
| `OCR_MAX_ATTEMPTS` | 2 | bounded OCR fallback attempts (3 enables OSD rotation) |
| `OCR_MAX_ZOOM`, `OCR_MAX_SIDE_PX` | 4.17, 4200 | render-resolution bounds for scans |
| `MAX_REGION_OCR` | 6 | OCR of picture regions on digital pages (per document) |
| `PARSEFLOW_REMOVE_FURNITURE` | 1 | exclude confidently detected headers/footers from Markdown |
| `PARSEFLOW_MATH_RECOGNIZER` | none | `none` · `pix2tex` · `rapid_latex_ocr` · `command` (scanned equations) |
| `PARSEFLOW_MATH_MODEL_DIR`, `PARSEFLOW_MATH_CMD` | – | local weights folder / external recognizer command |
| `PARSEFLOW_DATA_DIR`, `PARSEFLOW_JOB_TTL_HOURS`, `PARSEFLOW_WORKERS` | data/, 24, 2 | API storage (documents + `parseflow.db` with accounts), guest expiry, concurrency |
| `PARSEFLOW_SESSION_DAYS`, `PARSEFLOW_SECURE_COOKIE` | 30, 0 | login session lifetime; always set the cookie's `Secure` flag (it is set automatically on HTTPS, also behind a proxy that sends `X-Forwarded-Proto`) |
| `PARSEFLOW_WORKER_MEMORY_MB`, `PARSEFLOW_RETRIES` | auto, 1 | memory cap per worker; automatic retries (smaller slices) after a crash or out-of-memory |
| `PARSEFLOW_CHECKPOINT`, `PARSEFLOW_RESUME`, `PARSEFLOW_LITE_MB` | 1, 1, 8 | save finished page slices; queue interrupted documents again after a restart; size above which the viewer opens the light copy |
| `PARSEFLOW_OCR_WORKERS` | up to 4 | scanned pages OCR'd at once |
| `ASK_EMBED_MODEL`, `ASK_EMBED_BASE_URL`, `ASK_EMBED_API_KEY` | – | optional neural embeddings for Ask retrieval (OpenAI-compatible endpoint; default base URL = `ASK_BASE_URL`, else Ollama) |
| `PARSEFLOW_OCR_ENGINE`, `PARSEFLOW_HANDWRITING`, `PARSEFLOW_CHART_MODEL`, `PARSEFLOW_MODEL_DIR`, `PARSEFLOW_*_CMD` | tesseract, none, none, –, – | optional local models / command plugins (see *Local models*) |
| `PARSEFLOW_PUBLIC_URL` | – | the address users open ParseFlow at, used in emailed links (set it in production) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_SECURITY` | –, 587, –, –, –, starttls | outgoing email; without `SMTP_HOST` emails are written to `<data>/outbox/` and the server log |
| `PARSEFLOW_REQUIRE_VERIFIED` | 0 | 1 = users must confirm their email before logging in |
| `PARSEFLOW_GUEST_LINKS` | private | `open` = guest documents reachable by anyone with the id (v2.3 behaviour) |
| `PARSEFLOW_TRUST_PROXY` | 0 | 1 = build links from `X-Forwarded-Host/-Proto` (only behind your own proxy) |
| `DATABASE_URL` | – | `postgresql://…` to use PostgreSQL instead of the SQLite file (`pip install -r requirements-postgres.txt`) |

### Formula recognition for scanned equations (optional)
Digital PDFs need no model. For equations in **scans**, a recognizer can be plugged in; it runs only on detected equation regions and **never downloads weights at runtime** (missing weights are reported via `/api/health` and block warnings):

* `pix2tex` (LaTeX-OCR, MIT): `pip install "pix2tex[cli]"` (needs PyTorch; CPU works, roughly seconds per equation). Put `weights.pth` and `image_resizer.pth` from the project's GitHub release into `PARSEFLOW_MATH_MODEL_DIR`.
* `rapid_latex_ocr` (Apache-2.0, ONNX Runtime only): `pip install rapid_latex_ocr`; put `encoder.onnx`, `decoder.onnx`, `image_resizer.onnx`, `tokenizer.json` into `PARSEFLOW_MATH_MODEL_DIR`.
* `command`: any local executable that takes an image path and prints LaTeX (`PARSEFLOW_MATH_CMD`).

These adapters could not be exercised here (no PyPI/model access in the build environment); only the `command` plugin and the "weights missing" path are covered by tests. Evaluate a model on old-scans math before relying on it.

---

## Tests

```bash
pytest -q                                   # 200 unit/integration tests on generated fixtures (~105 s; OCR tests need tesseract)
PARSEFLOW_LIVE_LLM=1 ASK_PROVIDER=... pytest -m live   # optional: the Ask assistant against a real model
python tools/ask_eval.py your.pdf --out ask_eval   # optional: real answers from your model, with latency and checks
python tools/check_ai.py                            # optional: does the configured AI model answer (and read images)?
bash bench/full_benchmark.sh                        # optional: the full 1,403-PDF olmOCR-Bench on this machine
OLMOCR_PATH=/path/to/olmocr pytest -m bench # regression on the official sample with the official test classes
```
Covered: native text, scanned OCR (skew, noise, blur, low contrast, uneven light, small text; preprocessing never worse than plain OCR), two-column and sidebar reading order, header/footer classification (repeated headers, page numbers, first-page title, footnotes, section headings, documents without headers, configurability), equation boundaries/LaTeX/symbols/fallback, ruled tables with merged cells, borderless tables, prose rejection, Markdown escaping, hyphenation, Unicode/accents, empty/corrupt/encrypted/multi-page/image inputs, OCR timeout fallback, verifier independence/provenance, JSON and Markdown API responses, upload validation and cleanup, CLI batch/timeout, determinism; diagram structure (trees with traversals and heap/BST checks, stacks and queues with pointers, linked lists with NIL, weighted graphs with shortest paths, flowcharts with branches, DFAs with runs, Gantt charts, sorting traces, self-loops written as shuffled pieces, unreadable labels reported), Ask about this block (evidence sent to the model, image attached, citations outside the evidence removed, follow-ups, per-block chat API and access), map-reduce summaries and comparisons. Failures name the PDF, page, test id and the scorer's explanation.

---

## Benchmark (official olmOCR-Bench scorer)

`bench/run_olmocr_bench.py` converts every PDF, writes `<candidate>/<pdf>_pg{page}_repeat1.md`, then runs the **unmodified official CLI** and records scores, latency (median/p90/p95/max, throughput), errors, timeouts, hardware and configuration as JSON + Markdown. Failed or timed-out documents get an empty Markdown file, so their tests fail; nothing is skipped.

```bash
git clone https://github.com/allenai/olmocr.git          # scorer (we used commit f7cfe4c)
pip install fuzzysearch rapidfuzz tqdm beautifulsoup4 pypdf playwright && playwright install chromium

# full benchmark: 1,403 PDFs from Hugging Face
huggingface-cli download --repo-type dataset --resume-download allenai/olmOCR-bench --local-dir ./olmOCR-bench
python bench/run_olmocr_bench.py --bench-dir olmOCR-bench/bench_data --olmocr olmocr --results bench/results/full.json

# fixed regression sample: N PDFs per category, deterministic by hash (manifest saved)
python bench/run_olmocr_bench.py --bench-dir olmOCR-bench/bench_data --sample-per-category 25 --seed 0 --olmocr olmocr

# original v1 implementation under identical conditions
python bench/run_olmocr_bench.py --bench-dir … --impl legacy --legacy-backend /path/to/v1/backend --legacy-md plain

# regression gate (CI): fail if a protected category drops by more than 1 point
python bench/compare.py --baseline bench/baselines/sample_v2.json --current bench/results/parseflow.json --tolerance 1.0 --protected dataset.jsonl,type:math,type:table,type:absent
```
The scorer command that is run is exactly: `python -m olmocr.bench.benchmark --dir <bench-dir> --candidate parseflow`.

### Benchmark results

**Only the official scorer's bundled sample (21 PDFs, 100 tests) could be run in the build environment**: Hugging Face (and therefore the 1,403-PDF dataset) was not reachable. The sample is also the set this version was developed against, so these are **in-sample numbers, not a held-out or full-benchmark score**. The full run has to be done with the commands above.

Hardware: Intel Xeon @ 2.80 GHz, 2 logical cores, 7.8 GB RAM, Linux; PyMuPDF 1.28.2, Tesseract 5.3.4; 2 workers; vision off; no math model; Python `rapidfuzz` implementation in the scorer. Latency = time inside each worker around the conversion call (imports and process spawn excluded, cold caches, one fresh process per document), identical method for v1 and v2.

| Official sample, `dataset.jsonl` (99 tests) | v1 original | v2 |
|---|---|---|
| **Main score** | **34.3 %** (34/99) | **83.8 %** (83/99) |
| absent (headers/footers) | 33.3 – 37.5 | 100.0 |
| math | 0.0 | 100.0 |
| table | 50.0 | 100.0 |
| order | 46.2 | 84.6 |
| present | 30.0 – 33.3 | 53.3 |
| baseline / blank-page checks | 100 / 0–100 | 100 / 100 |
| median latency (s/page) | 0.50 | **0.17** |
| p90 / p95 / max (s/page) | 1.6–1.8 / 3.3 / 11.8–12.2 | 2.2 / 2.5 / 6.8 |
| errors / timeouts | 0 / 0 | 0 / 0 (peak worker RSS 421 MB) |

(v1 ranges cover the two ways of turning v1 output into Markdown: the frontend `md()` with provenance comments, and plain text. The scorer's printed "overall" averages the three JSONL groups of the sample — 44.8/78.1 vs 94.6 — and is not meaningful on this sample.) Robustness check (not scored): the CLI converted all 37 PDFs in olmocr's `tests/gnarly_pdfs` (498 pages: rotated, image-only, broken text layers, maps, forms) with no failures, median 0.21 s/page, in 82 s on the same 2 cores. The remaining sample failures are handwriting (needs a vision model), errors already present in a scan's OCR, logo text inside an image, and one ambiguous card-layout order.

**Is the score above 80 %?** On the official *sample* the main test file reaches 83.8 %, but that is in-sample and not the benchmark. A reproducible **full olmOCR-Bench score above 80 % has not been demonstrated**. For context, the olmocr README reports Marker 76.1, MinerU 75.2, DeepSeek-OCR 75.7 and Mistral OCR 72.0; the results above 80 (PaddleOCR-VL 80.0, olmOCR v0.4 82.4, Infinity-Parser 82.5, Chandra 83.1) come from neural vision-language models. The full set contains old-scan, handwriting and scanned-math material where an offline Tesseract pipeline is fundamentally limited. Run the full benchmark to get the real number: `bash bench/full_benchmark.sh` (set `WORKERS` to your core count) downloads the scorer and data and writes the official overall score with median/p90 page latency to `bench/results/full.md`. Latency on the 2-vCPU build machine: median 0.22–0.29 s/page on the sample (v2.6); it has not been measured on an 8-core laptop, and no other parser was timed on the same machine.

---

## Project structure

```text
PARSERFLOWS/
├── backend/
│   ├── app.py                FastAPI app (v1 API + markdown, ask, thumbnails, accounts, library)
│   ├── accounts.py           local accounts, sessions, saved documents, email tokens (SQLite or PostgreSQL)
│   ├── mailer.py             verification / reset emails (SMTP, or an outbox folder)
│   ├── cli.py                batch CLI
│   └── processing/
│       ├── pipeline.py       stages, routing, assembly, timings
│       ├── textlayer.py      glyph-level text layer, spacing, accents, quality
│       ├── layout.py         reading order, paragraphs, headings, dehyphenation
│       ├── headers_footers.py running header/footer/page-number classifier
│       ├── tables.py         table detection + cell model + Markdown/HTML
│       ├── mathtex.py        math detection + LaTeX reconstruction
│       ├── recognizers.py    optional local formula recognizers
│       ├── ocr.py            adaptive preprocessing + bounded OCR
│       ├── verifier.py       evidence-based verification
│       ├── markdown.py       canonical Markdown serializer
│       ├── runner.py         isolated execution (CLI, benchmark, web jobs): memory cap, retries
│       ├── fontfix.py, glyphshape.py  broken character maps repaired from glyph names / outlines
│       ├── charts.py         data from vector bar/line charts
│       ├── engines.py        optional local OCR / handwriting / chart models
│       ├── assistant.py, llm.py, embeddings.py  Ask assistant: retrieval, summaries, model providers
│       ├── block_ask.py          Ask about this block: evidence, computed facts, image, citation checks
│       ├── diagrams.py           structure of vector diagrams (trees, graphs, automata, flowcharts, arrays, lists)
│       ├── docfiles.py       compressed / light / per-page files for large documents
│       ├── vision.py         optional Gemini/Claude vision (unchanged)
│       ├── schema.py         block schema helpers
│       └── pdf_parser.py     v1 helpers (kept for compatibility, unused by v2 pipeline)
├── frontend/index.html
├── deploy/                   docker-compose (Caddy HTTPS, optional PostgreSQL), Caddyfile, systemd unit
├── Dockerfile
├── tools/                    glyph template generator, glyph evaluation on held-out fonts, ask_eval.py (live Ask evaluation)
├── bench/                    official-scorer harness, v1 runner, compare, baselines
├── tests/                    pytest suite
├── parseflow                 CLI launcher
├── requirements.txt, requirements-dev.txt, pyproject.toml, run.sh, run.bat, .env.example
└── .github/workflows/ci.yml
```

---

## Deployment notes

* **HTTPS**: `deploy/docker-compose.yml` runs ParseFlow behind Caddy, which obtains and renews a certificate automatically (`PARSEFLOW_DOMAIN=your.domain docker compose up -d --build` in `deploy/`). Without Docker, use `deploy/parseflow.service` with Caddy or nginx in front. Set `PARSEFLOW_PUBLIC_URL`, and SMTP for real emails. These files were written for this release but Docker was not available in the build environment, so they have not been run here.
* **Several servers**: point them at one PostgreSQL (`DATABASE_URL`) and one shared data folder (NFS/EFS); a job's status is read from its files when another server started it. (PostgreSQL support is written against `psycopg` 3 and unit-tested for its SQL, but not run against a live PostgreSQL here.)
* ParseFlow needs the **Tesseract binary** for scans and runs background jobs in a thread pool with page images on local disk. Serverless platforms (e.g. Vercel functions) typically lack the binary, cap execution time and do not keep local files between invocations; **this was not tested on any serverless platform**, so use a normal server or container there.
* Licences: PyMuPDF/MuPDF is **AGPL-3.0** (or commercial licence from Artifex); Tesseract is Apache-2.0. Check this before distributing ParseFlow as a hosted service.

## Current limitations

What still has limits after v2.6 (and what was not tested here):
* **AI answers were not run against a real model here**: the build environment's network policy blocks Google's API (and Hugging Face) - the key in `.env` was present, the requests were refused by the sandbox proxy. Run `python tools/check_ai.py` (or press *Test AI* in the Ask tab) to confirm your model answers, then `python tools/ask_eval.py your.pdf`. The whole path — evidence, image attachment for Anthropic / Gemini / OpenAI formats, citation checks, follow-ups — is tested with local stand-in servers, and the offline answers were checked on real textbook figures. Run `python tools/ask_eval.py your.pdf` with your key to evaluate real answers: it records every answer with its latency, removed references, and whether the model agrees with the facts ParseFlow computed.
* **Diagrams in images** are read from the pixels, but much less completely than vector figures (see the measured numbers above: with OCR, about a fifth of the edges of the CLRS figures are recovered with the right names). Handwritten diagrams depend on OCR of handwriting, which Tesseract reads poorly; circuits/schematics, sequence and use-case diagrams have no dedicated interpretation (they are read as generic graphs or left to the image). With a vision model configured, its reading is shown and checked against the geometry, never taken as verified.
* **Labels**: rotated, two-line and slightly distant edge labels are now assigned; labels written far from their edge, or between two edges at equal distance, are still left unassigned (reported, not guessed).
* Parent/child in an undirected tree is taken from the vertical layout; left/right from horizontal position.
* **Without any model**, Ask quotes the document instead of explaining it, handwriting and chart *images* are kept and flagged, and scanned maths is OCR text. Each has a plug-in (Ask: `ASK_PROVIDER`, `ollama` for offline; handwriting/charts/OCR: *Local models*; scanned maths: *Formula recognition*). None of these models could be downloaded in the build environment: the Ask providers are tested against local stand-in servers (`pytest -m live` runs the real thing), and the local-model adapters through their command plugins.
* **Glyphs from broken fonts**: about 8 % of broken-code glyphs on unseen fonts are still not identified; they are flagged, never guessed. Very unusual maths layouts (commutative diagrams, nested arrays) are reconstructed approximately and marked `review`.
* **Tables**: irregular borderless tables (cells spanning rows without rules) can still be split imperfectly; they are marked `review`.
* **Deployment**: Docker/Caddy files and PostgreSQL support were not run here (no Docker or PostgreSQL available); SMTP was tested against a local stub server.
* **Benchmark**: the official sample regression runs in CI; the full 1,403-PDF olmOCR-Bench is downloaded from Hugging Face, which was not reachable from the build environment, so it was not re-run for v2.5; the official sample regression gives the same scores as v2.4 (main 83.8 %, 83/99).
* **Disk**: the extra files written after processing (compressed and light copies, per-page blocks, Markdown, Ask index) add about 1.4× the size of the JSON (Cormen: 21 MB JSON + 30 MB).
* The CLI's default per-document timeout (`-t 300`) can be too short for long scanned documents; raise it for those.

## Team

Team DEADLOCK · Project: ParseFlow · Problem Statement: DQCL ParseAnything · Event: DataQuest 3.0 · Repository: https://github.com/ADII6769/PARSERFLOWS

## In Short

Don't treat every part of a document the same. Understand the content first, use the right extractor, and keep the evidence with the result.
