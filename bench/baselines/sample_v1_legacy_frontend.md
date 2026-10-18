# olmOCR-Bench run: parseflow_legacy_frontend

- scorer: `/usr/bin/python3 -m olmocr.bench.benchmark --dir /home/claude/benchwork/sample --candidate parseflow_legacy_frontend`
- dataset: /home/claude/benchwork/sample - 21 PDFs, 100 tests, jsonl: blanks.jsonl, dataset.jsonl
- hardware: Intel(R) Xeon(R) Processor @ 2.80GHz · 2 logical cores · 7.8 GB · Linux-6.18.44-fc-v114-x86_64-with-glibc2.39
- engines: PyMuPDF 1.28.2 · tesseract 5.3.4 · vision off

**Overall (official): 44.8%** (± 3.2)

| JSONL / category | score | passed / tests |
|---|---|---|
| baseline | 100.0 | 20/20 |
| blanks.jsonl | 0.0 | 0/1 |
| dataset.jsonl | 34.3 | 34/99 |

| test type | score | tests |
|---|---|---|
| absent | 37.5 | 24 |
| baseline | 95.2 | 21 |
| math | 0.0 | 12 |
| order | 46.2 | 13 |
| present | 30.0 | 30 |
| table | 50.0 | 20 |

Latency per page (s): median 0.509 · p90 1.573 · p95 3.278 · max 12.180 · throughput 0.881 pages/s with 2 workers
Errors: 0 · timeouts: 0

Slowest documents: lincoln_letter.pdf (12.18s), mattsnotes.pdf (4.271s), buildingnotes.pdf (3.278s), test-graphical-text.pdf (1.573s), mathfuncs.pdf (0.913s)
