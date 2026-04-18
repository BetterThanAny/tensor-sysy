#!/usr/bin/env bash
# W11: one-command local gate for GPU bench regression.
# Requires: built ./build/tsy-bench, populated benchmarks/baseline/rtx3080_wsl.csv.
set -euo pipefail

cd "$(dirname "$0")/.."

BENCH=./build/tsy-bench
BASELINE=benchmarks/baseline/rtx3080_wsl.csv
CURRENT=/tmp/tsy_bench_current.csv

if [[ ! -x "$BENCH" ]]; then
    echo "error: $BENCH not found. Build first: cmake --build build -j" >&2
    exit 1
fi

echo "[1/3] matmul sweep..."
"$BENCH" --primitive matmul > "$CURRENT"

echo "[2/3] transformer_block (append, strip duplicate header)..."
"$BENCH" --primitive transformer_block | tail -n +2 >> "$CURRENT"

echo "[3/3] compare vs baseline..."
python3 scripts/bench_compare.py \
    --baseline "$BASELINE" \
    --current  "$CURRENT"
