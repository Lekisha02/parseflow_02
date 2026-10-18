#!/usr/bin/env bash
# Full olmOCR-Bench run on YOUR machine, scored by the unmodified official scorer, with per-page latency.
#
#   bash bench/full_benchmark.sh            # all CPU cores as workers
#   WORKERS=8 bash bench/full_benchmark.sh
#
# Downloads (once): the olmocr scorer (git, pinned commit) and the 1,403-PDF benchmark from Hugging Face (~1.4 GB).
# Writes bench/results/full.json and bench/results/full.md: official overall score, per-category scores, median /
# p90 / p95 page latency, throughput, errors, timeouts, hardware. Nothing is skipped: a failed or timed-out PDF
# gets an empty answer, so its tests fail.
set -euo pipefail
cd "$(dirname "$0")/.."
EXT="${EXT:-ext}"
mkdir -p "$EXT" bench/results
if [ -z "${WORKERS:-}" ]; then
  WORKERS=$(python3 -c "import os; print(os.cpu_count() or 2)")
fi

python3 -m pip install -q -r requirements.txt -r requirements-dev.txt huggingface_hub
python3 -m playwright install chromium >/dev/null 2>&1 || echo "note: playwright browser install failed; math tests need it"

if [ ! -d "$EXT/olmocr" ]; then
  git clone -q https://github.com/allenai/olmocr.git "$EXT/olmocr"
  (cd "$EXT/olmocr" && git checkout -q f7cfe4c 2>/dev/null || true)
fi
if [ ! -d "$EXT/olmOCR-bench/bench_data" ]; then
  python3 - "$EXT" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="allenai/olmOCR-bench", repo_type="dataset", local_dir=f"{sys.argv[1]}/olmOCR-bench")
PY
fi

echo "Running ParseFlow on $(ls "$EXT"/olmOCR-bench/bench_data/pdfs | wc -l | tr -d ' ') PDF folders with $WORKERS workers..."
python3 bench/run_olmocr_bench.py --bench-dir "$EXT/olmOCR-bench/bench_data" --olmocr "$EXT/olmocr" \
  --workers "$WORKERS" --results bench/results/full.json
echo
echo "Done. Summary: bench/results/full.md"
echo "To compare latency with another parser on the same machine, time it on the same PDFs, for example:"
echo "  /usr/bin/time -p your_parser ext/olmOCR-bench/bench_data/pdfs/arxiv_math/<file>.pdf"
