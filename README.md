# TensorSysY

A tensor-oriented compiler that extends SysY with first-class tensor types and
builtin operators (`matmul`, `add`, `softmax`, `rmsnorm`), compiling to C++/CUDA
through a runtime adapter over `mini-llm-engine`.

See [PLAN.md](PLAN.md) for the full 12-week roadmap.

## Status

W0 (bootstrap) — CLI skeleton, CMake, flex/bison frontend migrated from
`sysy-compiler`, `SourceLocation`, diagnostic engine, gtest/ctest wired up.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## CLI

```bash
./build/tsc --help
./build/tsc parse    examples/smoke.tsy
./build/tsc dump-ast examples/smoke.tsy
./build/tsc emit-hir examples/smoke.tsy   # stub (W2)
./build/tsc run-lir  examples/smoke.tsy   # stub (W4)
```

## Layout

```
src/frontend/  flex/bison + AST + SourceLocation + Diagnostics
src/hir/       (W2)  HIR ops, verifier, shape inference
src/lir/       (W4)  loop-level IR + interpreter
src/passes/    (W5)  pass manager, constant folding, DCE, fusion
src/runtime/   (W6)  adapter over mini-llm-engine ops_cpu/ops_cuda
src/codegen/   (W7)  C++ and CUDA emitters
src/tools/     CLI entrypoint (tsc)
tests/         gtest + golden + e2e layers (see PLAN.md §测试方案 v2)
third_party/   local references to sysy-compiler and mini-llm-engine
```
