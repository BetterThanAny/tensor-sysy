#!/usr/bin/env python3
"""Thin driver over build/tsy-bench. No external deps — stdlib only.

Usage:
    python benchmarks/run_shapes.py             # full sweep + table
    python benchmarks/run_shapes.py --smoke     # ctest entrypoint
    python benchmarks/run_shapes.py --check-scheduler
                                                # assert tiled > naive at 1024^3
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path


BENCH = Path("build/tsy-bench")

# Empirical threshold: on WSL + RTX 3080 we've measured tiled ~1.26x
# faster than naive at 1024^3. Use 1.2x as the regression guard: looser
# than observed, tight enough to catch a scheduler-broken-kernel case.
SCHEDULER_SPEEDUP_MIN = 1.2


def run(bench_args: list[str]) -> list[dict]:
    if not BENCH.exists():
        print(f"error: {BENCH} not found (build it first with cmake --build)",
              file=sys.stderr)
        sys.exit(1)
    cmd = [str(BENCH), *bench_args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return list(csv.DictReader(io.StringIO(result.stdout)))


def print_table(rows: list[dict]) -> None:
    for r in rows:
        print(f"{r['primitive']:>7}  "
              f"{r['M']:>5}x{r['K']:>5}x{r['N']:>5}  "
              f"{r['variant']:>8}  "
              f"{float(r['ms_median']):8.3f} ms  "
              f"{float(r['gflops']):9.1f} GFLOPS")


def check_scheduler(rows: list[dict]) -> int:
    by = {}
    for r in rows:
        key = (r['M'], r['K'], r['N'], r['variant'])
        by[key] = float(r['ms_median'])

    big = ('1024', '1024', '1024')
    naive_key = (*big, 'naive')
    tiled_key = (*big, 'tiled')

    if naive_key not in by or tiled_key not in by:
        print("skip: 1024^3 not in default sweep (did you --shapes override?)",
              file=sys.stderr)
        return 0

    speedup = by[naive_key] / by[tiled_key]
    print(f"1024^3: tiled/naive speedup = {speedup:.2f}x "
          f"(min required {SCHEDULER_SPEEDUP_MIN:.2f}x)")
    if speedup < SCHEDULER_SPEEDUP_MIN:
        print(f"FAIL: below threshold", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one shape, exit 0 if tsy-bench runs cleanly")
    ap.add_argument("--check-scheduler", action="store_true",
                    help="assert tiled > naive at 1024^3 (not in ctest)")
    args = ap.parse_args()

    bench_args = ["--smoke"] if args.smoke else []
    rows = run(bench_args)
    if not rows:
        print("no rows from tsy-bench", file=sys.stderr)
        return 1

    print_table(rows)

    if args.check_scheduler:
        return check_scheduler(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
