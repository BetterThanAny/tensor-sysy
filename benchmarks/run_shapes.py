#!/usr/bin/env python3
"""Thin driver over tsy-bench. No external deps — stdlib only.

Usage:
    python benchmarks/run_shapes.py             # full sweep + table
    python benchmarks/run_shapes.py --smoke     # ctest entrypoint
    python benchmarks/run_shapes.py --check-scheduler
                                                # assert scheduled variants are covered
    python benchmarks/run_shapes.py --bench /tmp/build/tsy-bench
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path


def default_bench() -> Path:
    if os.environ.get("TSY_BENCH"):
        return Path(os.environ["TSY_BENCH"])
    if os.environ.get("TSY_BUILD_DIR"):
        return Path(os.environ["TSY_BUILD_DIR"]) / "tsy-bench"
    return Path("build/tsy-bench")


def run(bench: Path, bench_args: list[str]) -> list[dict]:
    if not bench.exists():
        print(f"error: {bench} not found (set --bench, TSY_BENCH, or TSY_BUILD_DIR)",
              file=sys.stderr)
        sys.exit(1)
    cmd = [str(bench), *bench_args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return list(csv.DictReader(io.StringIO(result.stdout)))


def print_table(rows: list[dict]) -> None:
    for r in rows:
        print(f"{r['primitive']:>7}  "
              f"{r['M']:>5}x{r['K']:>5}x{r['N']:>5}  "
              f"{r['variant']:>8}  "
              f"{float(r['ms_median']):8.3f} ms  "
              f"{float(r['gflops']):9.1f} GFLOPS")


def scheduled_variant(m: int, k: int, n: int) -> str:
    if m * n < 1024:
        return "naive"
    aligned = (m % 128 == 0) and (n % 128 == 0) and (k % 8 == 0)
    large_enough = (m >= 128) and (n >= 128) and (k >= 128)
    if aligned and large_enough and m * n <= 256 * 256:
        return "tiled"
    return "cublas"


def check_scheduler(rows: list[dict]) -> int:
    by_shape: dict[tuple[str, str, str], set[str]] = {}
    for r in rows:
        key = (r['M'], r['K'], r['N'])
        by_shape.setdefault(key, set()).add(r['variant'])

    failures = 0
    for shape, variants in sorted(by_shape.items(), key=lambda x: tuple(map(int, x[0]))):
        m, k, n = (int(x) for x in shape)
        expected = scheduled_variant(m, k, n)
        label = f"{shape[0]}x{shape[1]}x{shape[2]}"
        if expected not in variants:
            print(f"FAIL: scheduler picks {expected} for {label}, "
                  f"but benchmark rows only contain {sorted(variants)}",
                  file=sys.stderr)
            failures += 1
        else:
            print(f"scheduler {label}: {expected} covered")

    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one shape, exit 0 if tsy-bench runs cleanly")
    ap.add_argument("--check-scheduler", action="store_true",
                    help="assert each swept shape includes the variant selected "
                         "by ScheduleCudaPass")
    ap.add_argument("--bench", type=Path, default=default_bench(),
                    help="path to tsy-bench (default: TSY_BENCH, "
                         "TSY_BUILD_DIR/tsy-bench, or build/tsy-bench)")
    args = ap.parse_args()

    bench_args = ["--smoke"] if args.smoke else []
    rows = run(args.bench, bench_args)
    if not rows:
        print("no rows from tsy-bench", file=sys.stderr)
        return 1

    print_table(rows)

    if args.check_scheduler:
        return check_scheduler(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
