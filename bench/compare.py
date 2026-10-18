#!/usr/bin/env python3
"""Compare two run_olmocr_bench.py result files and fail (exit 1) on regressions beyond a tolerance.

    python bench/compare.py --baseline bench/baselines/sample_v2.json --current bench/results/parseflow.json \
        --tolerance 1.0 --protected overall,dataset.jsonl --latency-tolerance 0.5

Scores are the official scorer's numbers (percent). Protected keys are 'overall', any JSONL name (a category),
or 'type:<test type>'. Latency regression is checked on the per-page median (seconds, relative tolerance).
"""
import argparse
import json
import sys


def scores(r):
    s = r.get("scorer") or {}
    out = {"overall": s.get("overall")}
    for k, v in (s.get("per_jsonl") or {}).items():
        out[k] = v["score"]
    for k, v in (s.get("per_test_type") or {}).items():
        out["type:" + k] = v["score"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--tolerance", type=float, default=1.0, help="allowed drop in percentage points")
    ap.add_argument("--protected", default="overall", help="comma list, or 'all'")
    ap.add_argument("--latency-tolerance", type=float, default=None, help="allowed relative increase of median latency (0.5 = +50%%)")
    a = ap.parse_args()
    b, c = json.load(open(a.baseline)), json.load(open(a.current))
    sb, sc = scores(b), scores(c)
    keys = sorted(set(sb) | set(sc)) if a.protected == "all" else [k.strip() for k in a.protected.split(",")]
    rows, failed = [], False
    for k in sorted(set(sb) | set(sc)):
        x, y = sb.get(k), sc.get(k)
        d = None if x is None or y is None else round(y - x, 2)
        bad = k in keys and d is not None and d < -a.tolerance
        failed |= bad
        rows.append(f"{k:32s} {x if x is not None else '-':>8} -> {y if y is not None else '-':>8}  {'' if d is None else f'{d:+.2f}'}{'  REGRESSION' if bad else ''}")
    lb, lc = b["latency"].get("per_page_median"), c["latency"].get("per_page_median")
    rows.append(f"{'latency median (s/page)':32s} {lb!s:>8} -> {lc!s:>8}")
    if a.latency_tolerance is not None and lb and lc and lc > lb * (1 + a.latency_tolerance):
        rows[-1] += "  REGRESSION"; failed = True
    print("\n".join(rows))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
