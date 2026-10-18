# olmOCR-Bench run: parseflow_legacy_plain

- scorer: `/usr/bin/python3 -m olmocr.bench.benchmark --dir /home/claude/benchwork/sample --candidate parseflow_legacy_plain`
- dataset: /home/claude/benchwork/sample - 21 PDFs, 100 tests, jsonl: blanks.jsonl, dataset.jsonl
- hardware: Intel(R) Xeon(R) Processor @ 2.80GHz · 2 logical cores · 7.8 GB · Linux-6.18.44-fc-v114-x86_64-with-glibc2.39
- engines: PyMuPDF 1.28.2 · tesseract 5.3.4 · vision off

**Overall (official): 78.1%** (± 3.0)

| JSONL / category | score | passed / tests |
|---|---|---|
| baseline | 100.0 | 20/20 |
| blanks.jsonl | 100.0 | 1/1 |
| dataset.jsonl | 34.3 | 34/99 |

| test type | score | tests |
|---|---|---|
| absent | 33.3 | 24 |
| baseline | 100.0 | 21 |
| math | 0.0 | 12 |
| order | 46.2 | 13 |
| present | 33.3 | 30 |
| table | 50.0 | 20 |

Latency per page (s): median 0.496 · p90 1.818 · p95 3.247 · max 11.815 · throughput 0.888 pages/s with 2 workers
Errors: 0 · timeouts: 0

Slowest documents: lincoln_letter.pdf (11.815s), mattsnotes.pdf (3.994s), buildingnotes.pdf (3.247s), test-graphical-text.pdf (1.818s), blank_book_pg1.pdf (0.902s)
