# olmOCR-Bench run: parseflow

- scorer: `/usr/bin/python3 -m olmocr.bench.benchmark --dir /home/claude/benchwork/sample --candidate parseflow`
- dataset: /home/claude/benchwork/sample - 21 PDFs, 100 tests, jsonl: blanks.jsonl, dataset.jsonl
- hardware: Intel(R) Xeon(R) Processor @ 2.80GHz · 2 logical cores · 7.8 GB · Linux-6.18.44-fc-v114-x86_64-with-glibc2.39
- engines: PyMuPDF 1.28.2 · tesseract 5.3.4 · vision off

**Overall (official): 94.6%** (± 2.4)

| JSONL / category | score | passed / tests |
|---|---|---|
| baseline | 100.0 | 20/20 |
| blanks.jsonl | 100.0 | 1/1 |
| dataset.jsonl | 83.8 | 83/99 |

| test type | score | tests |
|---|---|---|
| absent | 100.0 | 24 |
| baseline | 100.0 | 21 |
| math | 100.0 | 12 |
| order | 84.6 | 13 |
| present | 53.3 | 30 |
| table | 100.0 | 20 |

Latency per page (s): median 0.173 · p90 2.242 · p95 2.485 · max 6.839 · throughput 1.181 pages/s with 2 workers
Errors: 0 · timeouts: 0

Slowest documents: lincoln_letter.pdf (6.839s), blank_book_pg1.pdf (2.485s), buildingnotes.pdf (2.242s), test-graphical-text.pdf (2.014s), openstax_caculus_pg_273.pdf (0.539s)
