# tensor-sysy GPU benchmark baseline

## Source of truth

`benchmarks/baseline/rtx3080_wsl.csv` is the canonical reference. Everything
in this doc describes how to reproduce and when to update that file.

Current content:

```
primitive,M,K,N,variant,ms_median,gflops
matmul,1024,1024,1024,naive,6.9577,308.649
matmul,1024,1024,1024,tiled,4.91114,437.268
matmul,1024,1024,1024,cublas,4.97181,431.932
```

## Scope

Current baseline tracks **only 3 rows**: `matmul 1024x1024x1024` × {naive,
tiled, cublas}. This is NOT what the original W11 spec anticipated (18 rows
including 512^3, sub-ms shapes, and `transformer_block`). The scope was
shrunk during T10 capture after empirical evidence showed:

- 1024^3 matmul: <5% cross-run drift — gate-able at 10%/5% FAIL/WARN
- 512^3 matmul cublas: 15-25% drift from cublas kernel heuristics
- Sub-ms shapes (128x16x8, 7x13x11): 40-200% drift, dominated by CUDA
  launch overhead jitter (~20-50us per launch)
- transformer_block cuda_adapter: bimodal (1.8-3.1 ms typical, occasional
  5 ms+ outliers)

tsy-bench still sweeps all shapes + `transformer_block` + 3 backends for
correctness coverage and informational timing. `bench_compare.py` only
gates on baseline-tracked rows; other current rows are silently ignored.

## Hardware

- GPU: NVIDIA GeForce RTX 3080 Laptop GPU (16384 MiB)
- Driver CUDA version: 566.36
- CUDA toolkit: release 12.0, V12.0.140 (build cuda_12.0.r12.0/compiler.32267302_0)
- Host: Ubuntu 24.04.4 LTS (Noble Numbat)
- Kernel: 6.6.87.2-microsoft-standard-WSL2 (WSL2 on Windows)

## Software at time of capture

- g++: 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)
- CMake: 3.28.3
- bison: 3.8.2
- flex: 2.6.4
- Python: 3.11.15 (from `.venv/bin/python --version`)
- Repo commit: `7f6d468a17e54ba13acb6ad2a0a1c626a74509f6`
  (`7f6d468 bench: 1024^3 matmul baseline + tracked-rows-only comparison`)

## Physical conditions required

The baseline is valid only when captured under:

- Laptop plugged in (AC power, not battery)
- Windows power mode: **Performance** (not Balanced, not Battery Saver)
- No browser tabs playing video, no Docker Desktop, no IDE indexer running
- `nvidia-smi` shows GPU idle before capture (utilization < 10%, temp < 60°C)

These conditions shift `ms_median` by 5-15% each. Deviating silently
triggers the 10% gate as a false positive. If a regression appears after
a legitimate code change, first verify physical conditions before
investigating the code.

## Reproduction

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake --build build -j
bash scripts/bench_local.sh
```

Expected: `summary: 0 FAIL, ...`. WARN rows (5-10% drift) are acceptable.

## Noise profile (captured during W11 T10, ~10 runs)

| Row | Typical ms_median | Cross-run ratio range | Notes |
|-----|-------------------|------------------------|-------|
| matmul 1024^3 naive | 6.8-7.4 ms | 0.97-1.06 | stable |
| matmul 1024^3 tiled | 4.9-5.4 ms | 0.97-1.09 | stable |
| matmul 1024^3 cublas | 4.9-5.3 ms | 0.95-1.04 | stable |
| matmul 512^3 (all) | 1.7-2.3 ms | 0.82-1.21 | **excluded** — cublas heuristic drift |
| matmul 256^3 (all) | 0.4-1.4 ms | 0.69-1.34 | **excluded** — below noise floor |
| matmul 128x16x8 / 7x13x11 | 0.14-0.47 ms | 0.39-2.31 | **excluded** — launch overhead dominates |
| transformer_block cuda_adapter | 1.8-5.2 ms | 0.63-1.68 | **excluded** — bimodal |
| transformer_block native/cpu_adapter | 2-3 us | 0.81-1.77 | **excluded** — sub-clock-tick |

Thresholds in `scripts/bench_compare.py`:

- matmul: FAIL ≥ 10%, WARN ≥ 5%
- transformer_block: FAIL ≥ 15%, WARN ≥ 10% (thresholds left in place
  even though no `transformer_block` row is currently baselined — future
  re-inclusion after improved timing infrastructure should not require
  re-calibrating)

## When to update the baseline

Trigger an update when:

1. Scheduler / layout / codegen changes yield real performance movement
   (IMPROVED rows printed; confirm it's genuine improvement, not
   measurement-scope change)
2. CUDA toolkit or driver upgrade changes numbers system-wide
3. Hardware environment changes (new machine → new baseline file, not
   an update to this one)
4. New baselined shapes get promoted after measurement infrastructure
   becomes less noisy (e.g., if future work lets `transformer_block` use
   a timing methodology that tames the bimodal noise)

## How to update

```bash
# Confirm a regression or improvement is legitimate. Verify physical
# conditions (AC power, performance mode, idle). Re-run bench_local.sh
# 2-3 times to ensure numbers are stable.

bash scripts/bench_local.sh

# Once confirmed, capture fresh baseline:
./build/tsy-bench --primitive matmul > /tmp/newbase.csv
# Filter to tracked shapes (only 1024^3 currently):
(head -1 /tmp/newbase.csv; grep -E "^matmul,1024,1024,1024," /tmp/newbase.csv) \
    > benchmarks/baseline/rtx3080_wsl.csv
rm /tmp/newbase.csv

git add benchmarks/baseline/rtx3080_wsl.csv
git commit -m "bench: refresh baseline for <REASON>

- <describe code change or environment shift>
- Median change: <summarize row ratios, e.g. 1024^3 cublas 4.97 -> 4.52>"
```

Commit messages for baseline updates must state WHY numbers moved. If a
future reader can't distinguish optimization from measurement-scope
change from a later `git log`, baseline updates become blind.

## Future work (not in W11)

- Measurement infrastructure improvements: per-kernel `cudaEvent` timing
  + pinning to a specific CUDA stream could tame sub-ms noise
- Median-of-N with larger N (currently N=5) for bimodal workloads
- Self-hosted runner or cloud GPU CI to gate PRs against GPU regressions
- Re-include 512^3 + `transformer_block` cuda_adapter once noise is
  measurable — they're the natural next targets to re-baseline
