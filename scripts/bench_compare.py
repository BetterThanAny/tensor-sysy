#!/usr/bin/env python3
"""Compare current bench CSV vs baseline CSV with per-primitive thresholds.

Exit codes:
  0  — no rows exceed the fail threshold (WARNs allowed)
  1  — at least one row exceeds its primitive's fail threshold

Usage:
  scripts/bench_compare.py --baseline <b.csv> --current <c.csv>
  scripts/bench_compare.py --baseline <b.csv> --current <c.csv> --update-baseline
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

# (fail_ratio, warn_ratio) — current/baseline ratio crossing fail_ratio → FAIL.
# Calibrated against observed cross-run noise on RTX 3080 Laptop + WSL2:
# only 1024^3 matmul stayed within 10% across ~10 captures under the same
# power/thermal state. 512^3 cublas drifts 15-25% due to cublas kernel
# heuristics; transformer_block cuda_adapter is bimodal (1.8-3ms typical,
# occasional 5ms+ outliers). Sub-ms shapes are dominated by CUDA launch
# overhead jitter (~20-50us). Only tracked (baseline) rows are gated;
# tsy-bench's full sweep is kept for correctness + informational timing.
THRESHOLDS = {
    "matmul":            (1.10, 1.05),
    "transformer_block": (1.15, 1.10),
}
DEFAULT = (1.10, 1.05)  # fallback for unknown primitives


def load(path: Path) -> dict[tuple, float]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["primitive"], row["M"], row["K"], row["N"], row["variant"])
            out[key] = float(row["ms_median"])
    return out


def classify(ratio: float, primitive: str) -> str:
    fail, warn = THRESHOLDS.get(primitive, DEFAULT)
    if ratio >= fail: return "FAIL"
    if ratio >= warn: return "WARN"
    if ratio < 0.95:  return "IMPROVED"
    return "OK"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--current",  required=True, type=Path)
    ap.add_argument("--update-baseline", action="store_true",
                    help="overwrite baseline with current after printing diff")
    args = ap.parse_args()

    baseline = load(args.baseline)
    current  = load(args.current)

    fail_count = warn_count = imp_count = missing_count = 0
    # Only compare rows tracked in baseline. tsy-bench may sweep more shapes
    # than the regression gate tracks (e.g. tiny shapes kept for tiled-alignment
    # correctness coverage but excluded from baseline because measurement
    # noise dominates their <0.5ms median). Extra current rows are silently
    # ignored — `cat /tmp/tsy_bench_current.csv` to inspect them.
    tracked_keys = sorted(baseline.keys())

    for key in tracked_keys:
        prim, M, K, N, var = key
        b = baseline[key]
        c = current.get(key)
        if c is None:
            print(f"MISSING  {prim:18s} {M:>5}x{K:>5}x{N:>5} {var:12s} "
                  f"baseline={b:.3f}ms")
            missing_count += 1
            continue
        ratio = c / b
        status = classify(ratio, prim)
        print(f"{status:8s} {prim:18s} {M:>5}x{K:>5}x{N:>5} {var:12s} "
              f"baseline={b:7.3f}ms current={c:7.3f}ms ratio={ratio:.3f}")
        if status == "FAIL": fail_count += 1
        elif status == "WARN": warn_count += 1
        elif status == "IMPROVED": imp_count += 1

    ok_count = len(tracked_keys) - fail_count - warn_count - imp_count - missing_count
    print(f"\nsummary: {fail_count} FAIL, {warn_count} WARN, {imp_count} IMPROVED, "
          f"{missing_count} MISSING, {ok_count} OK")

    if imp_count > 0 and not args.update_baseline:
        print("hint: confirmed improvements? re-run with --update-baseline")

    if args.update_baseline:
        shutil.copyfile(args.current, args.baseline)
        print(f"baseline updated: {args.baseline}\n"
              f"\u2192 review the diff and commit: git add {args.baseline}")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
