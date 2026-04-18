# TensorSysY

A tensor-oriented compiler that extends **SysY** with first-class tensor
types and builtin operators (`matmul`, `add`, `softmax`, `rmsnorm`,
`transpose`, `relu`), lowering through HIR → LIR → a thin runtime adapter
onto `mini-llm-engine`'s CPU and CUDA kernels.

> End-to-end path: `.tsy` source → AST → HIR (with passes) → LIR (with
> scheduling/layout lowering) → C++/CUDA codegen → linked against the
> runtime adapter → native binary.

## Status — W11 complete (2026-04-18)

- ✅ Full CPU pipeline closed (W7): `examples/mlp.tsy` compiles and runs,
  matches interpreter / PyTorch reference.
- ✅ CUDA path closed (W8–W10): single-op and `transformer_block.tsy`
  run end-to-end on RTX 3080; CUDA vs CPU agrees within `atol=1e-4,
  rtol=1e-3`.
- ✅ CI + local bench gate (W11): **32/32** local ctest (CPU + CUDA),
  **~20/20** CPU-only ctest in GHA, `scripts/bench_local.sh` gates three
  1024³ matmul variants against a recorded baseline.
- ⏳ W12 (this week): docs pass — see [`docs/`](docs/) below.

Full roadmap: [PLAN.md](PLAN.md).

## Quick start

```bash
# Build (Release; CUDA auto-detected if available)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run the full test suite (CPU + CUDA, 32 tests)
ctest --test-dir build --output-on-failure

# Run the demos
./build/out/mlp                   # CPU MLP forward
./build/out/matmul_cuda_demo      # CUDA single-op
./build/out/transformer_block     # CPU transformer block
./build/out/transformer_block_cuda

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
| `emit-lir`   | LIR dump (post LIR passes — layout lowering, CUDA scheduling)        |
| `emit-cpp`   | Self-contained C++ host source (links against `adapter_cpu`)         |
| `emit-cu`    | Self-contained CUDA source (links against `adapter_cuda`)            |
| `run-lir`    | Runs the LIR interpreter with deterministic inputs, prints tensors   |

Pipeline flags (apply to `emit-hir` / `emit-lir` / `run-lir`):

- `--opt=O0` — verifier only.
- `--opt=O1` — verifier → const-fold → dce → verify-post. LIR stage
  additionally runs `layout-lowering` and `schedule-cuda`.
- `--disable-pass=<name>` — repeatable; skip a named pass. Useful for
  round-trip pass testing.

## Layout

```
src/frontend/   flex/bison grammar (sysy.l/y), AST, SourceLocation, diagnostic engine
src/hir/        HIR ops, shape/type inference, verifier, lowering from AST, printer
src/lir/        loop-level IR, interpreter, printer, module utilities
src/passes/     PassManager, const-fold, DCE, layout-lowering, schedule-cuda
src/runtime/    adapter_cpu.cpp + adapter_cuda.cu — thin bridge to mini-llm-engine
src/codegen/    C++ (cpp.cpp) and CUDA (cuda.cpp) emitters
src/tools/      tsc CLI + tsy-bench sweep harness
examples/       .tsy sources — smoke, matmul variants, mlp, transformer_block, neg cases
tests/
  parse/        L2 frontend regression (bad_*.tsy, tensor_all_ops.tsy)
  shape/        type/shape verifier positive + negative
  passes/       L4 pass semantics (O0/O1 round trip, --disable-pass)
  adapter/      L3 runtime adapter (matmul layouts, softmax axis, add residual, transpose+relu)
  codegen/      codegen produces source + binary compiles + binary runs
  run/          CLI smoke — emit-hir / emit-lir / run-lir snapshots
  e2e/          pytest — transformer_block vs numpy/torch triangular check
  golden/       AST/HIR/LIR/diagnostics text golden files
benchmarks/
  run_shapes.py       shape sweep (3 sizes × {naive,tiled,cublas} + 2 edge shapes)
  baseline/rtx3080_wsl.csv   canonical 3-row baseline for W11 gate
scripts/
  bench_local.sh      wraps sweep → compare → exit code
  bench_compare.py    FAIL/WARN threshold logic (10% / 5%)
  compare_numpy.py    CPU-path vs numpy reference helper
  compare_pytorch.py  CPU/CUDA path vs PyTorch reference helper
docs/
  architecture.md     data flow, pass pipeline, adapter contract
  benchmarks/baseline.md   baseline reproduction + noise analysis
  blog/writeup.md     project writeup — decisions, lessons, what was shrunk
  demo.md             one-shot reproduction guide
third_party/
  sysy-compiler-ref/       frozen snapshot of the original grammar for regression
  mini-llm-engine-ref/     frozen snapshot of the ops library we adapt onto
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
                                    │        mini-llm-engine        mini-llm-engine
                                    │         ops_cpu.*              ops_cuda.*
                                    │
                                    └───► reference output for three-way compare
```

Full details: [`docs/architecture.md`](docs/architecture.md).

## Testing layers

| Layer | Scope                                               | Gate                              |
|-------|-----------------------------------------------------|-----------------------------------|
| L0    | Build + CLI smoke                                   | `ctest` — cli_* targets           |
| L1    | Golden (AST / HIR / LIR / diagnostics text)         | `ctest` — `tests/golden/`         |
| L2    | Frontend + verifier (positive + negative)           | `ctest -R "parse\|shape"`         |
| L3    | Runtime adapter (matmul layout, broadcast, ReLU)    | `adapter_cpu_cases`, `adapter_cuda_cases` |
| L4    | Passes (structure, semantics, idempotence, disable) | `pass_cases`, `schedule_cuda_cases` |
| L5    | Interpreter / codegen three-way compare             | `cli_*` + `e2e_transformer_block_pytest` |
| L6    | E2E vs PyTorch / numpy                              | `tests/e2e/`                      |
| L7    | Performance regression                              | `scripts/bench_local.sh`          |

Rationale behind the layers: see PLAN.md §测试方案 v2.

## Benchmarks

Current W11 gate covers three rows of `matmul 1024×1024×1024`
(`naive`/`tiled`/`cublas`) on an RTX 3080 Laptop GPU under WSL2. FAIL >10%
slower than baseline, WARN 5–10%. The 18-row aspiration from the original
W11 spec was empirically shrunk (sub-ms shapes drift 40–200% run-to-run,
transformer_block is bimodal). Full noise analysis and shrinking evidence:
[`docs/benchmarks/baseline.md`](docs/benchmarks/baseline.md).

## Dependencies

- CMake ≥ 3.20, a C++17 compiler, flex/bison (system packages).
- CUDA 12.x toolkit (optional — CPU-only build skips `adapter_cuda` and
  the CUDA tests automatically).
- Python ≥ 3.10 with `numpy`, `pytest` (only for the E2E pytest and the
  numpy reference). Install via `uv venv && uv pip install numpy pytest`
  — ctest uses `.venv/bin/python` by default (override with
  `-DTSY_PYTHON_EXECUTABLE=...`).

## External repos

- [`sysy-compiler`](https://github.com/BetterThanAny/sysy-compiler) —
  frozen under `third_party/sysy-compiler-ref/` as the regression anchor
  for the original SysY grammar.
- [`mini-llm-engine`](https://github.com/BetterThanAny/mini-llm-engine) —
  frozen under `third_party/mini-llm-engine-ref/`; `adapter_cpu` and
  `adapter_cuda` are the only code allowed to call into it.
