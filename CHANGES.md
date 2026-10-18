# ParseFlow v2 — changes

Baseline: the uploaded `PARSERFLOWS-main` (commit `c729770` in this history). All v1 API routes, response fields,
frontend features, upload workflow and downloads are kept; JSON changes are additive.

## Modified files
| File | Change |
|---|---|
| `backend/processing/pipeline.py` | Rewritten around the same 7 stages: per-page routing by measured text-layer quality (digital / embedded OCR / OCR / hybrid / sparse), per-stage timings, render-on-demand, native-resolution OCR rendering, display-equation, table, figure, drop-cap handling, document-level furniture pass, scanned-equation recognizer hook, busy-vector-page mode, invalid-code-point safety, backend Markdown in the output. |
| `backend/processing/ocr.py` | Adaptive preprocessing from measured metrics (contrast, background, speckle, skew, text size), bounded fallback (conservative image when risky steps were used), optional OSD, image-hash cache, configurable language/timeout. v1 helpers kept. |
| `backend/processing/verifier.py` | Evidence rules: independence, document-encoded basis, OCR/model never verified alone, ruled vs borderless tables, `evidence_detail`. |
| `backend/processing/schema.py` | Documented contract, `Z`, `union`, `scale` helpers. |
| `backend/app.py` | Markdown endpoint, upload type check before storing, safe filenames, cleanup on failure, job expiry, timings in status, richer health. Existing routes/fields unchanged. |
| `frontend/index.html` | Markdown tab renders backend Markdown (+ include-headers toggle; browser/demo mode keeps `md()`), Inspector shows evidence basis, furniture reasons, transforms, original text, HTML tables with spans; timings in Overview; dashed overlays for excluded headers/footers; Header/Footer filter. |
| `README.md`, `.env.example`, `pyproject.toml`, `requirements-dev.txt` (new), `.gitignore` (new) | Documentation, configuration, packaging (`parseflow` console script), dev/benchmark dependencies. |

## New files
| File | Purpose |
|---|---|
| `backend/processing/textlayer.py` | Glyph-level text layer reader: segments with font/size/baseline, adaptive spacing, ligatures, accents, stray-accent repair, zero-width/surrogate cleanup, text-quality score. |
| `backend/processing/layout.py` | Recursive XY-cut reading order with gutter preference, sidebars, paragraphs, headings (size rarity, bold, ALL-CAPS), lists, captions, code, evidence-based dehyphenation, quote normalisation. |
| `backend/processing/headers_footers.py` | Document-level running header/footer/page-number classifier with scores and reasons; configurable removal. |
| `backend/processing/tables.py` | Cell model; ruled-grid tables (clipped `find_tables`, spans); aligned/booktabs tables; header extension (rotated labels, underline spans); currency columns; Markdown/HTML serialisation. |
| `backend/processing/mathtex.py` | Math detection (token level) and LaTeX reconstruction from glyph geometry. |
| `backend/processing/recognizers.py` | Optional local formula recognizers (pix2tex, rapid_latex_ocr, command plugin); never downloads weights. |
| `backend/processing/markdown.py` | Canonical deterministic Markdown serializer. |
| `backend/processing/runner.py` | Isolated per-document worker processes with timeouts (CLI + benchmark). |
| `backend/cli.py`, `parseflow`, `backend/__init__.py` | One-command batch CLI. |
| `bench/run_olmocr_bench.py`, `bench/legacy_runner.py`, `bench/compare.py`, `bench/baselines/*` | Official-scorer harness, v1 runner (same conditions), regression gate, recorded baselines. |
| `tests/*` | 67 pytest tests + official-scorer regression test on the bundled sample. |
| `.github/workflows/ci.yml` | CI: tests + sample regression gate (written, not executed here). |

`backend/processing/pdf_parser.py` and `vision.py` are unchanged (pdf_parser is no longer used by the pipeline).


## v2.1 — Ask, readable math, long documents
| File | Change |
|---|---|
| `backend/processing/ask.py` (new) | Extractive Ask engine (BM25 + section context, phrase/definition bonuses, table-cell lookup, page overviews, code-step excerpts, term/description rows, split-passage joining); typed answer parts with LaTeX kept for rendering. |
| `backend/app.py` | `GET /api/ask/{id}`; page images and thumbnails rendered on demand (`/api/page`, new `/api/thumb`); limits raised to 1000 pages / 300 MB. |
| `backend/processing/pipeline.py` | Pages processed in slices with per-page progress; glyph data and bitmaps released after each slice; verifier crops loaded lazily (same results); `save_images="needed"`. |
| `backend/processing/headers_footers.py` | Repetition measured at the same height and capped at 4 pages for long documents; page numbers that swap sides on even/odd pages; heading-sized one-off page titles kept in long documents. |
| `backend/processing/layout.py`, `tables.py`, `mathtex.py` | Numbered pseudo-code / enumerated listings are not tables; code paragraphs keep indentation and skip inline math; Unicode super/subscripts in LaTeX; crash on multi-code-point glyphs fixed. |
| `frontend/index.html`, `frontend/vendor/katex/` | Ask tab uses the API and renders math (KaTeX, vendored), code, lists and tables, with evidence snippets and suggested questions from the document; math rendered in Inspector/Blocks/Tables; lazy thumbnails, paged long lists, server-side upload limit. The post-upload screens look the same. |
| `tests/test_ask_and_large_docs.py` (new) | Ask answers, API, on-demand images, slice equivalence, page progress, long-document furniture. |

## v2.2 — new landing page, accounts and saved documents
| File | Change |
|---|---|
| `backend/accounts.py` (new) | SQLite users / sessions / documents; scrypt passwords; hashed session tokens; login rate limit; library stats. |
| `backend/app.py` | Auth, account and library endpoints; uploads by a logged-in user are saved; owner-only access to saved documents; guest expiry skips saved documents. |
| `frontend/index.html` | New landing page (logo, Try Live Demo, Log in / Sign up, auto-playing 7-slide feature tour replacing "Present 90s demo", upload card with guest / save options, moving feature ticker, features, stages, drop zone, footer); sign-in dialog; logged-in workspace (sidebar, library with filters, search, preview panel, rename/delete/download, settings, privacy). Processing screen and viewer unchanged; *New document* returns to the landing page or the library. Logo gradient now renders in every view. |
| `tests/test_accounts.py` (new) | Sign-up/login/logout, password + session storage, rate limit, privacy of saved documents, guest behaviour, rename/delete/expiry, settings, account deletion, interrupted jobs. |

## v2.3 — accuracy on real books, any document size, Ask assistant
Found on the 1,312-page *Introduction to Algorithms* (4th ed.).
| File | Change |
|---|---|
| `backend/processing/fontfix.py`, `glyphshape.py` (new), `data/glyph_templates.npz`, `tools/make_glyph_templates.py` (new) | Repair of broken character maps from glyph names or glyph outlines (Type 3 and TrueType/CFF): `=` `+` `[` `]` `(` `)` `−` `:` `,` `≤` `≥` `⌊⌋` `⌈⌉` `‖` `≫` `★`, dashes, ligatures, curly quotes. Unrepairable glyphs stay flagged. |
| `backend/processing/textlayer.py` | Applies the repairs per glyph; no false space where the next glyph is drawn over a space (`I NSERTION`); ligatures reported as a letter plus a zero-width letter keep their order (`fni ite` → `finite`); floor/ceiling brackets paired. |
| `backend/processing/mathtex.py` | URLs, e-mail addresses and DOIs are text; quoted words (`“problems.”` read as `<problems.=`) and words such as *Θ-notation* are text; stars, bullets and footnote marks are not math; an index expression such as `A[1 : i − 1]` stays whole and an unmatched closing bracket of the prose stays outside. |
| `backend/processing/tables.py` | ID-column tables ('2 \| isMovable \| Can it be moved?') are not mistaken for numbered listings; plausibility checks with reasons (picture, vector diagram, display maths, aligned equations, table of contents, book index, numbered listing, figure panels, coloured boxes, figure caption); trimming of running headers / lead-in lines / section labels. |
| `backend/processing/pipeline.py`, `layout.py` | Tables of contents detected directly and emitted as lists (wrapped entries rejoined); book indexes split into their text columns; numbered pseudo-code becomes a code block with exact indentation (column-aware, cost columns kept); rejected diagrams become figures; page-size background drawings ignored; garbled blocks flagged; no default page limit. |
| `backend/processing/runner.py`, `backend/app.py` | Each document processed in its own spawned worker process with progress mirroring (`PARSEFLOW_ISOLATE`, `PARSEFLOW_JOB_TIMEOUT`): a crash or out-of-memory fails only that document (`PF-503`/`PF-504`). No upload or page limit by default. `POST /api/ask/{id}`. |
| `backend/processing/assistant.py`, `llm.py` (new) | Ask assistant: passage retrieval across the document, follow-ups with conversation context, document/chapter/page summaries (outline and chapter ranges from the document's own table of contents), LLM answers separated into *From the document* (with `[p. N]` citations) and *Explanation*; Anthropic, Gemini, OpenAI-compatible and local Ollama; labelled extractive fallback. |
| `frontend/index.html` | Ask tab is a chat (citations jump to the page, KaTeX math, sources list, suggested follow-ups, model/fallback badge); Handwritten tab removed (handwriting is shown in Visuals, marked as handwritten); "any size" upload text. Other screens unchanged. |
| `requirements.txt`, `pyproject.toml`, `.env.example`, `README.md` | `fonttools`; Ask and isolation settings; limits default to 0 (none). |
| `tests/test_assistant.py`, `tests/test_v23_accuracy.py` (new) | Retrieval, summaries, follow-ups, model request formats (stub server / mock transports), fallback, API; broken-map repair, links vs equations, quoted words, TOC, pseudo-code vs table, diagram vs table, isolated processing. |

## v2.4 — the documented limitations, worked through
| Area | Change |
|---|---|
| Tables (`tables.py`, `pipeline.py`) | Wrapped and vertically centred cell lines join their row; tables continued on the next page are joined; row shading is not taken for a diagram; ID-column tables stay tables. |
| Maths (`mathtex.py`) | Big delimiters from their glyph shape and true extent (`\left…\right`); matrices, `cases`, integral limits, `aligned` derivations, equation numbers, dot stacks, primes, joined numbers, footnote markers after a full stop. A run whose LaTeX cannot be rebuilt keeps its text instead of failing the page. |
| Broken fonts (`glyphshape.py`, `fontfix.py`, `tools/`) | STIX and Liberation templates (regular/italic/bold), blurred-correlation scoring, correctly mapped maths symbols trusted; repairs keyed by font object (subsets sharing a name no longer mix glyph ids). On held-out fonts: 91.6 % of broken glyphs repaired, 0 of 700 correct glyphs changed (`tools/eval_glyphs.py`). |
| Layout (`pipeline.py`, `layout.py`) | Book-index pages become one list per column; card grids read box by box, row by row; pieces of one picture merged; text tables searched on busy vector pages; a page whose analysis fails keeps its text as plain paragraphs (flagged) instead of becoming one image region. |
| Charts and scans (`charts.py`, `engines.py`, `ocr.py`) | Data tables measured from vector bar/line charts; optional local OCR engine (PaddleOCR, Surya, command), handwriting (TrOCR, command) and chart-image (DePlot, command) adapters, never downloaded; scanned pages OCR'd in parallel. |
| Ask (`assistant.py`, `embeddings.py`, `app.py`, frontend) | Retrieval by meaning (LSA over the document + pseudo-relevance feedback; optional neural embeddings); index built at processing time and saved; offline "example" answers; book summaries one line per chapter (offline) or per-chapter model summaries combined (with a model); conversations saved with the document (`/api/chat/{id}`); opt-in live-model test. |
| Large documents (`docfiles.py`, `pipeline.py`, `runner.py`, `app.py`, frontend) | Compressed, light and per-page files; the viewer opens large documents from the light copy and loads details on demand (Cormen: 1.5 MB instead of 25 MB); checkpoints per page slice and resume (output identical to an uninterrupted run); automatic retry with smaller slices after a crash or out-of-memory; per-worker memory cap; queue with place in line; interrupted documents resumed after a restart; status read from disk when the job is not in memory. |
| Accounts and deployment (`accounts.py`, `mailer.py`, `app.py`, frontend, `deploy/`, `Dockerfile`) | Email confirmation and password reset (SMTP or outbox), optional confirmation before login; guest documents private to the uploading browser with an owner-controlled read-only share link; Secure cookies on HTTPS behind proxies; links from `PARSEFLOW_PUBLIC_URL`; Docker + Caddy (automatic HTTPS) + optional PostgreSQL, systemd unit; `DATABASE_URL` for PostgreSQL. |
| Tests | 139 (+38): `test_v24_fixes.py`, `test_scale.py`, `test_accounts_mail_share.py`, `test_assistant.py` additions, `test_live_llm.py` (opt-in). |

Not run in the build environment (no network to model hubs, no Docker, no PostgreSQL): the Ask providers against live accounts, the PaddleOCR/Surya/TrOCR/DePlot adapters with real weights, the Docker/Caddy deployment, PostgreSQL, and the full 1,403-PDF benchmark. The official sample regression, the 37 "gnarly" PDFs and the full 1,312-page Cormen book were run after the changes.

## v2.4.1 — matrix layout, borderless tables, Gemini vision & Ask
| Area | Change |
|---|---|
| Maths (`mathtex.py`) | Matrix columns reconstructed cleanly when entries have varying widths or overlapping x-spans (row-level gap clustering and centroid matching). Added unit tests in `test_v24_fixes.py`. |
| Tables (`tables.py`) | Borderless table continuation generalized to support 2-column tables, hanging indents, and multi-fragment wrapped lines without premature truncation. Added unit tests in `test_v24_fixes.py`. |
| Execution (`runner.py`, `test_scale.py`) | Process rusage scaling normalized for macOS Darwin; memory limit check tied into progress updates. |
| AI Models (`vision.py`, `llm.py`, `.env.example`, `.env`) | Updated default Gemini model to `gemini-3.5-flash`; enabled multimodal vision and Ask assistant with live API authentication. |


## v2.5 — Ask becomes a document tutor: ask about any block, diagrams understood, grounded answers
| Area | Change |
|---|---|
| Diagram understanding (`processing/diagrams.py`, new; `pipeline.py`) | The structure of a vector figure is read from the PDF's drawing commands. **Nodes** are circles, boxes, diamonds and ovals with the text inside them, with subscripts kept (`a_0`). **Edges** are lines, polylines, curves and self-loops, with directions taken from the arrowheads. **Labels** are weights, branch words and transition symbols, one per edge, nearest first. **Pointers** are `S.top`, `Q.head` and `L.head`. **Arrays, stacks and queues** come with indices, colour groups and Gantt time boundaries. **Singly, doubly and circular linked lists** keep their NIL fields. On top of the structure ParseFlow interprets **trees** (traversals, heap/BST checks, trees drawn with parent pointers, edge labels), **automata** (start and accepting states; only when the transitions carry symbols), **flowcharts** (decisions and their branches), **graphs** and **sorting traces**. Shape groups with no text and overlapping shapes such as Venn diagrams are reported and left uninterpreted rather than guessed. A highlight stroke drawn over an edge is merged with that edge. A stroke split into shuffled pieces is joined back together. Diagram blocks have their own extractor (`vector_diagram`), evidence and an *estimated* confidence. Their status is always `review`. `PARSEFLOW_DIAGRAMS=0` turns this off. |
| Ask about this block (`processing/block_ask.py`, new; `app.py`; frontend) | `POST /api/ask/{id}/block/{block_id}` answers about one block. The model gets the extraction (status and warnings, labelled as being about the extraction), the structure, **exact computed facts** (automaton runs on the string in the question, shortest paths, cycles, connectivity), the caption, the surrounding text, other pages that mention the figure, related passages, and **the cropped image** when the model accepts images (`ASK_VISION`). Page references outside the evidence are removed and reported. Without a model, the answer shows the evidence and computed facts. Each block keeps its own saved thread (`/api/chat/{id}?block=`). In the Inspector, a section directly under the extraction confidence has suggested questions per block type, follow-ups, *Show on page* and *New*. Visuals cards show the structure and have an *Ask about this* button. |
| Ask (`assistant.py`, `llm.py`) | Images go to Anthropic, Gemini and OpenAI-compatible APIs (`llm.chat(images=…)`). Summaries of long documents are map-reduce: each part is summarised from its own text, a few in parallel, and cached. A bug that stopped the part summaries from being reused was fixed. Comparison questions retrieve passages for each side. Style hints are added for *simply*, *step by step*, *derive*, *in detail* and *example*. Excerpts carry extraction notes. Page citations are checked, including lists such as `[p. 3, 8]`. |
| Evaluation (`tools/ask_eval.py`, `tests/test_live_llm.py`, `bench/diagram_audit_v25.md`) | `ask_eval.py` runs real questions against your configured model and records latency, removed references and agreement with the computed facts. A live block test is opt-in. A manual audit of 24 random CLRS figures, including 10 held out, was done before tuning on them: 3 correct, 1 mostly correct, 4 partial, 2 wrong (both Venn diagrams, which are now left uninterpreted). |
| Tests | 173 (+34): `test_diagrams.py` (22) covers synthetic figures with known answers. `test_block_ask.py` (8) covers the evidence sent, the image, citation checks and the per-block chat API with access control. `test_assistant.py` adds map-reduce, comparison and style tests. |

Measured on this machine (2 vCPUs):

* **Whole Cormen book** (1,312 pages, one process): 219 s (0.167 s/page), peak RSS 378 MB, 153 diagram blocks.
* **Official olmOCR sample:** main score 83.8 % (83/99), the same as v2.4. Median latency was 0.29 s/page with diagram reading on and 0.25 s/page with it off, one run each, so this is noisy.
* **Gnarly set:** 37/37 converted, 552 pages in 122 s.

Not run here, for lack of an API key or model download: answers from a real model (Gemini, Claude, OpenAI or a local model). That whole path is tested against local stand-in servers. Run `tools/ask_eval.py` with your key for real answers.

## v2.6 — every document type, diagrams in images, more diagram kinds, label fixes
| Area | Change |
|---|---|
| Input formats (`formats.py`, new; `pipeline.py`, `app.py`, `cli.py`, frontend) | The app now reads Word (DOCX/DOC/ODT/RTF), PowerPoint (PPTX/PPT/ODP), Excel (XLSX/XLSM/XLS/ODS/CSV/TSV), images (PNG/JPG, multi-page TIFF, BMP, GIF, WebP, HEIC), text, Markdown, HTML, JSON, EPUB, XPS, FB2, MOBI, CBZ and SVG. **Documents and slides:** LibreOffice converts them when installed; otherwise ParseFlow renders DOCX and PPTX from their content. **Spreadsheets:** read cell by cell, with merged cells and number formats. Formulas show their saved result; if the file has none, LibreOffice calculates it, and otherwise the formula is shown. **Charts:** PowerPoint chart data is kept as tables. **Photos and slides:** photos are turned upright from their EXIF orientation, and slide titles become headings. **Safety:** a damaged or renamed file is detected. The original file is kept, and the converted PDF is the document's source. The CLI discovers all formats, and the Health endpoint lists them. |
| Extraction fixes found with these files | **Lists:** a bullet written as its own run (Word, LibreOffice, symbol-font bullets) is joined to its item, and a list item ends where a non-indented paragraph starts. **Tables:** a line inside a ruled table cell is no longer taken for an equation; a heading above a table closed by a full-width rule is no longer pulled into it; and grids drawn cell by cell are found. **Ligatures:** ligature glyphs ('ti') are no longer "repaired" into other letters. **Glyph repair:** suspect fonts are found in two passes, so a repair no longer depends on the order of the text runs. This fixed the CLRS "–4 read as 34". Held-out glyph evaluation is unchanged: 91.6 % repaired, 0/700 false changes. |
| Diagram labels (`diagrams.py`) | Labels rotated along slanted edges, two-line labels, and labels up to 30 pt from their edge are now read. Each edge takes at most one label, nearest first. Node labels keep sub/superscripts across lines (ω_2n^0, c_0), and a size field under a key stays separate (26 20). On the CLRS flow network all 14 weights are now read (6 were missing). |
| Diagrams in images (`diagrams_raster.py`, new) | **Nodes:** outlines, colour-filled nodes (they must contain a label), double circles, self-loop lobes and NIL slashes. **Edges:** lines (crossings split into the straight pieces that ink covers) and arrowheads. When no shapes are drawn, the labels joined by lines become the nodes. Node text is OCR'd crop by crop. Photos and text pages are rejected, and there is a time budget per document. **Pipeline:** picture regions become `raster_diagram` blocks. With a vision model, its node/edge reading is parsed (`from_model`) and compared with the geometry (`agreement`). Measured on 154 CLRS figures rendered at 150 dpi (`tools/eval_raster_diagrams.py`): with the PDF's own words, 55 % edge recall and 94 % precision; end to end with Tesseract OCR, 19 % recall and 38 % precision, limited by OCR of node names. |
| More diagram kinds | **B-trees:** keys per node and an order check (all 13 CLRS chapter-18 trees pass). **Code trees and tries:** Huffman codes and trie paths read along labelled edges. **ER diagrams:** entities, weak entities, keys, multi-valued attributes, relationships with cardinalities. **UML class diagrams:** compartments, plus generalisation, aggregation and composition arrows. **Overlapping shapes:** groups such as Venn diagrams are not read as graphs. |
| AI model check | `llm.selftest()` sends one real request, plus a check that the model reads an image. It is exposed as `POST /api/llm/test`, the *Test AI* button in the Ask tab, and `tools/check_ai.py`. Gemini answers that come back empty, blocked, or cut short by thinking are now reported clearly instead of failing to parse. Block Ask tells the model where its structure came from (vector, image or model). |
| Benchmark tooling | `bench/full_benchmark.sh` downloads the official scorer and the 1,403-PDF dataset and runs the full olmOCR-Bench on your machine with latency. |
| Tests | 200 (+27): `test_formats.py` (13), `test_diagrams_raster.py` (8), plus B-tree, Huffman, ER and UML tests, rotated/stacked/far labels, and the AI self-test. |

Measured here (2 vCPUs, nothing else running):

* **Official olmOCR sample:** main score 83.8 % (83/99), unchanged. Median page latency was 0.29 s with diagram reading and 0.22 s without (p90 2.0 s either way).
* **Gnarly set:** 37/37 converted, 552 pages in 129 s.
* **Whole Cormen book:** 224 s (0.17 s/page), peak RSS 384 MB, 171 diagram blocks.

Not run here, because the sandbox's network policy refused Google's API and Hugging Face with HTTP 403:

* real Gemini answers, even though the key in `.env` was present;
* the full olmOCR-Bench.
