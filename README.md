# TensorSysY

A tensor-oriented compiler that extends **SysY** with first-class tensor
types and builtin operators (`matmul`, `add`, `softmax`, `rmsnorm`,
`transpose`, `relu`), lowering through HIR → LIR → a thin runtime adapter
layer. The CPU adapter bridges to `mini-llm-engine`'s `ops_cpu` entry points;
the CUDA adapter is project-local FP32 CUDA kernels plus cuBLAS.

> End-to-end path: `.tsy` source → AST → HIR (with passes) → LIR (with
> CUDA scheduling; `layout-lowering` is currently a registered no-op
> placeholder) → C++/CUDA codegen → linked against the runtime adapter →
> native binary.

## 30-Second Summary

| Signal | Details |
| --- | --- |
| Positioning | Tensor compiler project that connects frontend/compiler work with backend-style runtime integration and GPU execution. |
| Stack | C++17, flex/bison, HIR/LIR, CMake/CTest, C++/CUDA codegen, CPU adapter over `mini-llm-engine`, CUDA adapter with FP32 kernels/cuBLAS. |
| Hard parts | Shape/type verifier, optimization passes, CUDA scheduling, generated binary build/run, numpy numerical checks. |
| Quick start | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`; `cmake --build build -j`; `ctest --test-dir build --output-on-failure`. |
| Validation | CPU-only CTest by default; CUDA and pytest gates are registered only when the toolchain/dependencies are available. |
| Benchmark / result | `scripts/bench_local.sh` gates tracked 1024^3 matmul rows against the RTX 3080 WSL2 baseline when `tsy-bench` is built. |

## Status — current checkout

- ✅ Full CPU pipeline closed (W7): `examples/mlp.tsy` compiles and runs,
  matches interpreter / numpy reference.
- CUDA support is optional at configure time. Hosts with a CUDA compiler build
  `adapter_cuda`, CUDA examples, CUDA tests, and `tsy-bench`; CPU-only hosts
  skip those targets.
- The local CPU-only test matrix for this checkout registers 33 CTest tests.
  Additional pytest and CUDA tests depend on Python packages and CUDA
  availability.
- `scripts/bench_local.sh` gates the tracked 1024³ matmul rows against the
  recorded RTX 3080 WSL2 baseline.

Full roadmap: [PLAN.md](PLAN.md).

## Quick start

```bash
# Build (Release; CUDA auto-detected if available)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run the registered test suite. CUDA/pytest tests are conditional.
ctest --test-dir build --output-on-failure

# Run generated demos when their targets are built
./build/out/mlp                   # CPU MLP forward
./build/out/transformer_block     # CPU transformer block
./build/out/matmul_cuda_demo      # CUDA single-op, if CUDA is available
./build/out/transformer_block_cuda # CUDA, if CUDA is available

# Benchmark gate (1024³ matmul × {naive, tiled, cublas})
bash scripts/bench_local.sh
```

See [`docs/demo.md`](docs/demo.md) for a one-shot reproduction guide.

## CLI

```
tsc <command> [--opt=O0|O1] [--disable-pass=<name>] <input.tsy>
```

| Command      | Produces                                                             |
|--------------|----------------------------------------------------------------------|
| `parse`      | AST build check; exit code reflects parse success                    |
| `dump-ast`   | Pretty-printed AST                                                   |
| `emit-hir`   | MLIR-style HIR dump (post passes)                                    |
| `emit-lir`   | LIR dump (post LIR passes — CUDA scheduling; layout lowering is no-op) |
| `emit-cpp`   | Self-contained C++ host source (links against `adapter_cpu`)         |
| `emit-cu`    | Self-contained CUDA source (links against `adapter_cuda`)            |
| `run-lir`    | Runs the LIR interpreter with deterministic inputs, prints tensors   |

Pipeline flags (apply to `emit-hir` / `emit-lir` / `run-lir`):

- `--opt=O0` — verifier only.
- `--opt=O1` — verifier → const-fold → dce → verify-post. LIR stage
  additionally runs `layout-lowering` (currently no-op) and `schedule-cuda`.
- `--disable-pass=<name>` — repeatable; skip a named pass. Useful for
  round-trip pass testing.

## Layout

```
src/frontend/   flex/bison grammar (sysy.l/y), AST, SourceLocation, diagnostic engine
src/hir/        HIR ops, shape/type inference, verifier, lowering from AST, printer
src/lir/        loop-level IR, interpreter, printer, module utilities
src/passes/     PassManager, const-fold, DCE, layout-lowering placeholder, schedule-cuda
src/runtime/    adapter_cpu.cpp bridge + adapter_cuda.cu local CUDA kernels/cuBLAS
src/codegen/    C++ (cpp.cpp) and CUDA (cuda.cpp) emitters
src/tools/      tsc CLI + tsy-bench sweep harness
examples/       .tsy sources — smoke, matmul variants, mlp, transformer_block, neg cases
tests/
  parse/        L2 frontend regression (bad_*.tsy, tensor_all_ops.tsy)
  shape/        type/shape verifier positive + negative
  passes/       L4 pass semantics (O0/O1 round trip, --disable-pass)
  adapter/      L3 runtime adapter (matmul layouts, softmax axis, add residual, transpose+relu)
  run/          CLI smoke plus generated source/binary checks
  e2e/          pytest — transformer_block numerical parity via numpy
benchmarks/
  run_shapes.py       shape sweep (3 sizes × {naive,tiled,cublas} + 2 edge shapes)
  baseline/rtx3080_wsl.csv   canonical 3-row baseline for W11 gate
scripts/
  bench_local.sh      wraps sweep → compare → exit code
  bench_compare.py    FAIL/WARN threshold logic (10% / 5%)
  compare_numpy.py    CPU-path vs numpy reference helper
  compare_pytorch.py  optional PyTorch comparison helper
docs/
  architecture.md     data flow, pass pipeline, adapter contract
  benchmarks/baseline.md   baseline reproduction + noise analysis
  blog/writeup.md     project writeup — decisions, lessons, what was shrunk
  demo.md             one-shot reproduction guide
third_party/
  README.md                documents sibling reference checkout locations
```

## Architecture in one picture

```
.tsy  ──(flex/bison)──►  AST  ──(lowering)──►  HIR  ──(O0/O1 passes)──►  HIR'
                                                                           │
                                                                           ▼
                                                                         LIR
                                                                           │
                                    ┌───────────────┬──────────────────────┤
                                    ▼               ▼                      ▼
                             LIR interpreter    C++ codegen          CUDA codegen
                                    │               │                      │
                                    │               ▼                      ▼
                                    │        adapter_cpu          adapter_cuda
                                    │               │                      │
                                    │               ▼                      ▼
                                    │        mini-llm-engine        local FP32 kernels
                                    │         ops_cpu.*              + cuBLAS
                                    │
                                    └───► reference output for three-way compare
```

Full details: [`docs/architecture.md`](docs/architecture.md).

## Testing layers

| Layer | Scope                                               | Gate                              |
|-------|-----------------------------------------------------|-----------------------------------|
| L0    | Build + CLI smoke                                   | `ctest` — cli_* targets           |
| L1    | AST / HIR / LIR / diagnostics text smoke            | `ctest` — parse/run/cli targets   |
| L2    | Frontend + verifier (positive + negative)           | `ctest -R "parse\|shape"`         |
| L3    | Runtime adapter (matmul layout, broadcast, ReLU)    | `adapter_cpu_cases`, `adapter_cuda_cases` |
| L4    | Passes (structure, semantics, idempotence, disable) | `pass_cases`, `schedule_cuda_cases` |
| L5    | Interpreter / codegen three-way compare             | `cli_*` + `e2e_transformer_block_pytest` |
| L6    | E2E vs numpy                                        | `tests/e2e/`                      |
| L7    | Performance regression                              | `scripts/bench_local.sh`          |

Rationale behind the layers: see PLAN.md §测试方案 v2.

## Benchmarks

Current local gate covers the baseline-tracked rows of `matmul
1024×1024×1024` (`naive`/`tiled`/`cublas`) on an RTX 3080 Laptop GPU under
WSL2. FAIL >10% slower than baseline, WARN 5–10%; missing tracked rows fail,
and extra current rows are reported as informational. The 18-row aspiration
from the original W11 spec was empirically shrunk (sub-ms shapes drift
40–200% run-to-run, transformer_block is bimodal). Full noise analysis and
shrinking evidence:
[`docs/benchmarks/baseline.md`](docs/benchmarks/baseline.md).

## Dependencies

- CMake ≥ 3.20, a C++17 compiler, flex/bison (system packages).
- CUDA 12.x toolkit (optional — CPU-only build skips `adapter_cuda` and
  the CUDA tests automatically).
- Python ≥ 3.10 with `numpy`, `pytest` (only for the E2E pytest and the
  numpy reference). Install via `uv venv && uv pip install numpy pytest`
  — CMake uses `.venv/bin/python` when it exists, otherwise the detected
  system Python (override with `-DTSY_PYTHON_EXECUTABLE=...`).

## External repos

- [`sysy-compiler`](https://github.com/BetterThanAny/sysy-compiler) —
  optional sibling checkout documented in `third_party/README.md`; used as a
  read-only grammar reference, not vendored into this repository.
- [`mini-llm-engine`](https://github.com/BetterThanAny/mini-llm-engine) —
  sibling checkout used by `adapter_cpu` for `ops_cpu.cpp` when present.
  `adapter_cuda` does not call `mini-llm-engine`; it uses local FP32 kernels
  and cuBLAS.
