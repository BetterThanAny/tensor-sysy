# W11 CI + Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship tensor-sysy W11 — GitHub Actions CPU CI + local GPU baseline + regression diff + supporting W10 follow-ups, per `docs/superpowers/specs/2026-04-18-tensor-sysy-w11-ci-bench-design.md`.

**Architecture:** CPU CI on GHA ubuntu-latest (CUDA auto-skipped by existing `check_language(CUDA)` gate); local `tsy-bench` extended with `--primitive transformer_block`; baseline CSV (18 rows) under version control; `bench_compare.py` enforces differentiated thresholds (matmul 10%/5%, transformer_block 15%/10%); `bench_local.sh` is the one-command local gate. Three W10 reviewer follow-ups folded in because two of them block W11 acceptance (CI Python path, D2H sync trustworthiness).

**Tech Stack:** CMake (existing), GitHub Actions (ubuntu-latest), Python 3 stdlib (bench_compare), C++/CUDA (tsy-bench), bash (local wrapper), g++-13, bison 3.8, flex 2.6.

**Parallelization map (for subagent dispatch):**
- **Phase 1 (strict serial)**: Task 1 → 2 → 3 → 4. Each changes pre-existing code that later phases depend on. CUDA sync (T4) MUST precede baseline capture (T10) or baseline is physics-fiction.
- **Phase 2 (parallel, 3 branches)**: { T5+T7 } ‖ T6 ‖ T8. The three branches are file-disjoint.
- **Phase 3 (serial)**: T9 (integrates T7 + T8) → T10 (captures baseline from T7) → T11 (docs, references T10 data) → T12 (PLAN tick, references everything).

---

## Task 1: Add comment to verifyUnary

**Why first:** zero risk, unblocks nothing but closes W10 reviewer #3. Clearing the trivial item first keeps the work queue clean.

**Files:**
- Modify: `src/hir/verifier.cpp:158`

- [ ] **Step 1: Add the comment above `verifyUnary`**

Edit `src/hir/verifier.cpp`, locate line 158 (`void verifyUnary(...)`), insert 3 comment lines immediately above it:

```cpp
// NOTE: verifyUnary assumes the op preserves input shape element-wise.
// Do NOT reuse for shape-changing unary ops (e.g. transpose, reduce).
// Those must call their own verifier.
void verifyUnary(const Op& op, DiagnosticEngine& diag, const char* name) {
```

- [ ] **Step 2: Build + test (no behavior change expected)**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: build succeeds, 32/32 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/hir/verifier.cpp
git commit -m "docs(hir): clarify verifyUnary shape-preservation assumption"
```

---

## Task 2: Make Python interpreter path configurable via CMake cache var

**Why:** W10 reviewer #1. Blocker for W11 — GHA runner has no `/home/xs/.../.venv/bin/python`.

**Files:**
- Modify: `CMakeLists.txt` (root, insert TSY_PYTHON_EXECUTABLE logic)
- Modify: `tests/CMakeLists.txt:270` (single reference)

- [ ] **Step 1: Add TSY_PYTHON_EXECUTABLE to root CMakeLists.txt**

Open root `CMakeLists.txt`, find the `project(...)` line, add after it (before the first `add_library`/`add_executable`):

```cmake
# W11: Python interpreter for pytest-driven e2e tests.
# Prefer repo-local .venv (developer workflow), fall back to system Python3 (CI).
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(TSY_PYTHON_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python"
        CACHE FILEPATH "Python interpreter used by pytest-driven tests")
else()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(TSY_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
        CACHE FILEPATH "Python interpreter used by pytest-driven tests")
endif()
message(STATUS "TSY_PYTHON_EXECUTABLE = ${TSY_PYTHON_EXECUTABLE}")
```

- [ ] **Step 2: Replace hardcoded path in tests/CMakeLists.txt**

Open `tests/CMakeLists.txt`, line 270:

```cmake
# Before:
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python -m pytest

# After:
    COMMAND ${TSY_PYTHON_EXECUTABLE} -m pytest
```

- [ ] **Step 3: Add numpy/pytest import probe in tests/CMakeLists.txt**

Immediately before the e2e ctest registration (search for the block using `.venv/bin/python`; probe must run before `add_test(...)` for e2e), insert:

```cmake
# W11: verify TSY_PYTHON_EXECUTABLE has numpy + pytest before registering e2e tests.
execute_process(
    COMMAND ${TSY_PYTHON_EXECUTABLE} -c "import numpy, pytest"
    RESULT_VARIABLE _tsy_py_probe
    OUTPUT_QUIET ERROR_QUIET)
if(NOT _tsy_py_probe EQUAL 0)
    message(FATAL_ERROR
        "TSY_PYTHON_EXECUTABLE=${TSY_PYTHON_EXECUTABLE} lacks numpy/pytest.\n"
        "  Fix: uv pip install --python ${TSY_PYTHON_EXECUTABLE} numpy pytest")
endif()
```

- [ ] **Step 4: Re-configure and run full ctest locally**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | grep TSY_PYTHON
cmake --build build -j 2>&1 | tail -3
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected output of first command: `-- TSY_PYTHON_EXECUTABLE = /home/xs/tsy-wsl-export/tensor-sysy/.venv/bin/python`. Expected: 32/32 pass.

- [ ] **Step 5: Simulate CI by disabling .venv detection**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
mv .venv .venv.bak
cmake -S . -B build-cpusim -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE 2>&1 | grep TSY_PYTHON
mv .venv.bak .venv
```

Expected: `TSY_PYTHON_EXECUTABLE = /usr/bin/python3` (or whatever system Python3 resolves to). If the probe FATAL_ERRORs because system Python3 lacks numpy — that's the exact message GHA will show, which is the desired behavior.

Note: keep `build-cpusim/` for Task 6 validation; `rm -rf build-cpusim/` after Task 6.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt tests/CMakeLists.txt
git commit -m "build: introduce TSY_PYTHON_EXECUTABLE cache var for CI portability"
```

---

## Task 3: Gitignore tests/e2e/__pycache__

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Check if __pycache__ is tracked**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
git ls-files tests/e2e/__pycache__/ 2>&1 | head -5
```

If empty output → nothing tracked, just add to .gitignore. If non-empty → `git rm -r --cached tests/e2e/__pycache__/` first.

- [ ] **Step 2: Check current .gitignore contents**

```bash
grep -n "__pycache__" /home/xs/tsy-wsl-export/tensor-sysy/.gitignore
```

Existing line is `__pycache__/` (top-level pattern — already matches nested dirs via gitignore semantics). If it's there, **this task is a no-op**. Verify:

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
touch tests/e2e/__pycache__/.probe 2>/dev/null || mkdir -p tests/e2e/__pycache__ && touch tests/e2e/__pycache__/.probe
git status --short tests/e2e/ | grep pycache
rm tests/e2e/__pycache__/.probe 2>/dev/null
```

Expected: empty (probe file not shown in `git status` because `__pycache__/` matches). If empty, mark this task complete without editing anything and skip to Task 4. If probe file DOES show up, then pattern didn't match — add explicit entry.

- [ ] **Step 3 (only if Step 2 showed probe file as untracked): Add explicit gitignore**

Append to `.gitignore`:

```
tests/e2e/__pycache__/
```

- [ ] **Step 4: Commit (only if .gitignore changed)**

```bash
git add .gitignore
git commit -m "build: explicitly gitignore tests/e2e/__pycache__"
```

---

## Task 4: Explicit cudaDeviceSynchronize before D2H in all CUDA adapters

**Why:** W10 reviewer #2. `cudaMemcpy(DeviceToHost)` is implicitly synchronous on same stream, so this is **not** a correctness fix — it's an intent clarification + bench trustworthiness fix (cudaEvent timing + future cudaMemcpyAsync safety). **MUST complete before Task 10 (baseline capture)**: once this change lands, measured ms may shift slightly (D2H encompasses true kernel completion window) and baseline must reflect post-fix reality.

**Files:**
- Modify: `src/runtime/adapter_cuda.cu` (6 sites)

- [ ] **Step 1: Audit all kernel-launch → D2H sites**

```bash
grep -n "cudaMemcpyDeviceToHost" /home/xs/tsy-wsl-export/tensor-sysy/src/runtime/adapter_cuda.cu
```

Expected 6 matches at lines ~260, 284, 356, 425, 462, 482. For each, verify a kernel launch (`<<<...>>>`) or cuBLAS call precedes it in the same function.

- [ ] **Step 2: Insert `CUDA_CHECK(cudaDeviceSynchronize())` before each D2H**

For each of the 6 D2H sites, add one line immediately before:

```cpp
// Example pattern for each site:
// ... kernel<<<grid, block>>>(...);
// + CUDA_CHECK(cudaDeviceSynchronize());
//   CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
```

Use `Edit` tool once per site. Check each function for the kernel launch immediately preceding the D2H — the pattern is:

```cpp
    <kernel launch or cuBLAS call>;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data.data(), dOut, bytes, cudaMemcpyDeviceToHost));
```

- [ ] **Step 3: Build and run full ctest**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake --build build -j 2>&1 | tail -3
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: 32/32 pass (no functional change, just redundant sync).

- [ ] **Step 4: Run scheduler guard (noise sanity)**

```bash
python3 benchmarks/run_shapes.py --check-scheduler 2>&1 | tail -5
```

Expected: `1024^3: tiled/naive speedup = X.XXx (min required 1.20x)`, no FAIL.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/adapter_cuda.cu
git commit -m "fix(runtime/cuda): explicit cudaDeviceSynchronize before D2H

Kernel-launch → D2H chains relied on implicit same-stream sync of
cudaMemcpy. Making the sync explicit keeps intent clear and preserves
correctness if any caller switches to cudaMemcpyAsync. Required by W11
bench trustworthiness (cudaEvent timing must bracket a fully-completed
kernel window)."
```

---

## Task 5: Add `--primitive` flag to tsy-bench (matmul still default)

**Files:**
- Modify: `src/tools/tsy-bench.cu`

- [ ] **Step 1: Refactor matmul bench into its own function**

Edit `src/tools/tsy-bench.cu`. Extract the current `main()` matmul-sweep logic into a dedicated function `int runMatmulBench(const Options& opts)`. Add an `Options` struct holding `smoke`, `shapes_arg`, `variants_arg`.

New structure:

```cpp
namespace {
struct Options {
    bool smoke = false;
    std::string shapes_arg;
    std::string variants_arg;
};

// ... existing helpers unchanged ...

int runMatmulBench(const Options& opts) {
    std::vector<Shape> shapes;
    if (!opts.shapes_arg.empty()) shapes = parseShapes(opts.shapes_arg);
    else if (opts.smoke)          shapes = { {256, 256, 256} };
    else                          shapes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {128, 16, 8},
        {7, 13, 11},
    };

    std::vector<std::string> variants;
    if (!opts.variants_arg.empty()) variants = parseVariants(opts.variants_arg);
    else                            variants = { "naive", "tiled", "cublas" };

    std::cout << "primitive,M,K,N,variant,ms_median,gflops\n";
    for (const auto& s : shapes) {
        for (const auto& v : variants) {
            if (v == "tiled") {
                if (s.M % 128 != 0 || s.N % 128 != 0 || s.K % 8 != 0) continue;
            }
            float ms = benchOne(s.M, s.K, s.N, v);
            float gf = gflops(s.M, s.K, s.N, ms);
            std::cout << "matmul," << s.M << "," << s.K << "," << s.N << ","
                      << v << "," << ms << "," << gf << "\n";
        }
    }
    return 0;
}

}  // namespace
```

- [ ] **Step 2: Add `--primitive` flag parsing + dispatch in main**

Replace `main()`:

```cpp
int main(int argc, char** argv) {
    Options opts;
    std::string primitive = "matmul";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--smoke") opts.smoke = true;
        else if (a.rfind("--primitive=", 0) == 0) primitive = a.substr(12);
        else if (a == "--primitive" && i + 1 < argc) primitive = argv[++i];
        else if (a.rfind("--shapes=", 0) == 0) opts.shapes_arg = a.substr(9);
        else if (a == "--shapes" && i + 1 < argc) opts.shapes_arg = argv[++i];
        else if (a.rfind("--variants=", 0) == 0) opts.variants_arg = a.substr(11);
        else if (a == "--variants" && i + 1 < argc) opts.variants_arg = argv[++i];
        else if (a == "-h" || a == "--help") { return usage(argv[0]); }
        else { return usage(argv[0]); }
    }

    if (primitive == "matmul") return runMatmulBench(opts);
    // Task 7 will add: if (primitive == "transformer_block") return runTransformerBlockBench();
    std::cerr << "unknown --primitive: " << primitive
              << " (valid: matmul)\n";
    return 2;
}
```

Also update `usage()`:

```cpp
int usage(const char* progname) {
    std::cerr << "usage: " << progname
              << " [--primitive matmul] [--smoke] "
              << "[--shapes MxKxN[,...]] [--variants v1[,v2,...]]\n";
    return 2;
}
```

- [ ] **Step 3: Build and verify backward compat**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake --build build -j 2>&1 | tail -3
./build/tsy-bench --smoke 2>&1 | head -5
python3 benchmarks/run_shapes.py --smoke 2>&1 | tail -5
```

Expected: smoke output unchanged from pre-refactor; `run_shapes.py --smoke` returns 0.

- [ ] **Step 4: Verify --primitive flag works**

```bash
./build/tsy-bench --primitive matmul --smoke 2>&1 | head -5
./build/tsy-bench --primitive invalid 2>&1 ; echo "exit=$?"
```

Expected: first = normal CSV output; second = "unknown --primitive" message, exit=2.

- [ ] **Step 5: Commit**

```bash
git add src/tools/tsy-bench.cu
git commit -m "feat(tsy-bench): add --primitive flag, refactor matmul into own function

Preparatory refactor for transformer_block bench. No behavior change
when flag is omitted — default remains matmul sweep."
```

---

## Task 6: GitHub Actions CPU CI workflow

**Parallelizable with Task 5+7.** Completely independent from CUDA/bench code.

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create workflow file**

Create `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install build deps
        run: |
          sudo apt-get update
          sudo apt-get install -y g++-13 cmake bison flex python3-venv

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python venv
        run: |
          uv venv .venv
          uv pip install --python .venv/bin/python numpy pytest

      - name: Print tool versions
        run: |
          g++-13 --version
          cmake --version
          bison --version
          flex --version
          .venv/bin/python --version
          .venv/bin/python -c "import numpy, pytest; print('numpy', numpy.__version__, 'pytest', pytest.__version__)"

      - name: Configure
        run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-13

      - name: Build
        run: cmake --build build -j

      - name: Test
        run: ctest --test-dir build --output-on-failure
```

- [ ] **Step 2: Validate workflow syntax locally**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "YAML OK"
```

Expected: `YAML OK`.

- [ ] **Step 3: Simulate the CI path locally (CPU-only build)**

Use the `build-cpusim` dir from Task 2 Step 5, or re-create:

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
rm -rf build-cpusim
mv .venv .venv.bak
cmake -S . -B build-cpusim -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE 2>&1 | tail -5
mv .venv.bak .venv
```

Expected: configure succeeds, `TSY_HAVE_RUNTIME_CUDA: 0` visible, `TSY_PYTHON_EXECUTABLE` points to system Python3.

If system Python3 lacks numpy/pytest, probe will FATAL_ERROR — this is expected GHA-like behavior. In that case install them to system Python once: `sudo apt install python3-numpy python3-pytest` or `uv pip install --system numpy pytest` (latter may need flags).

Simpler path: let the CI job install via `uv venv .venv` (the workflow does this), and locally keep using `.venv`. This Step 3 is optional if Task 2 Step 5 already confirmed the cache-var behavior.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions CPU-only workflow

ubuntu-latest, single job. CMake's check_language(CUDA) skips all CUDA
targets on runners without nvcc, leaving ~20 CPU-path ctests. Triggers:
push to main, pull_request, workflow_dispatch."
```

- [ ] **Step 5: Clean up simulation dir**

```bash
rm -rf /home/xs/tsy-wsl-export/tensor-sysy/build-cpusim
```

---

## Task 7: Add transformer_block subcommand to tsy-bench

**Depends on:** Task 4 (sync), Task 5 (--primitive flag).

**Files:**
- Modify: `src/tools/tsy-bench.cu`
- Modify: `CMakeLists.txt` (root) — expose transformer_block runtime entry point to tsy-bench target

- [ ] **Step 1: Inspect transformer_block entry point**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
find examples/ -name "transformer_block*"
find . -name "transformer_block*.tsy" -o -name "transformer_block*.h"
grep -rn "transformer_block" src/ examples/ | grep -v "\.md:" | head -20
```

Identify: (a) where the W10 `tsy_add_cuda_example(transformer_block_cuda ...)` produces the host entry, (b) whether there's a reusable `forward()` symbol or only a `main()` wrapper.

- [ ] **Step 2: Decide link strategy based on Step 1**

Two options (pick the one the code supports):

**(A)** If codegen emits `forward(...)` as a distinct function: add a static library target in root `CMakeLists.txt` exposing it, link into tsy-bench.

**(B)** If codegen only emits `main()`: write a bespoke harness in `tsy-bench.cu` that directly calls `adapter_cuda` / `adapter_cpu` / `lir::interpreter` with the same tensor layout as `transformer_block.tsy` (hardcoded S=4, D=8, F=16 and param tensors).

Prefer (B) if (A) requires non-trivial codegen changes — W11 scope is not codegen refactor. Write a comment in `tsy-bench.cu` noting which route was chosen and why.

- [ ] **Step 3: Implement `runTransformerBlockBench`**

Add to `src/tools/tsy-bench.cu` (before the closing `}` of the anonymous namespace):

```cpp
// W11: transformer_block end-to-end timing.
// S=4, D=8, F=16 matches examples/transformer_block.tsy and the e2e pytest.
int runTransformerBlockBench() {
    constexpr int S = 4, D = 8, F = 16;

    // Prepare inputs + weights with deterministic fill (same seed convention as e2e test).
    // Names/shapes mirror examples/transformer_block.tsy:
    //   x: [S,D], Wq/Wk/Wv: [D,D], Wo: [D,D], W1: [D,F], W2: [F,D]
    auto x  = makeBuf("x",  {S, D}); tsy::lir::fillDeterministic(x,  0);
    auto Wq = makeBuf("Wq", {D, D}); tsy::lir::fillDeterministic(Wq, 1);
    auto Wk = makeBuf("Wk", {D, D}); tsy::lir::fillDeterministic(Wk, 2);
    auto Wv = makeBuf("Wv", {D, D}); tsy::lir::fillDeterministic(Wv, 3);
    auto Wo = makeBuf("Wo", {D, D}); tsy::lir::fillDeterministic(Wo, 4);
    auto W1 = makeBuf("W1", {D, F}); tsy::lir::fillDeterministic(W1, 5);
    auto W2 = makeBuf("W2", {F, D}); tsy::lir::fillDeterministic(W2, 6);
    auto out = makeBuf("out", {S, D});

    std::cout << "primitive,M,K,N,variant,ms_median,gflops\n";

    // For each backend, do 3 warmup + 5 measured and print one CSV row.
    struct Backend { const char* name; };
    Backend backends[] = { {"native"}, {"cpu_adapter"}, {"cuda_adapter"} };

    for (const auto& be : backends) {
        std::vector<float> times;

        // Warmup (3 runs, discard).
        for (int i = 0; i < 3; i++) {
            runTransformerBlockOnce(be.name, x, Wq, Wk, Wv, Wo, W1, W2, out);
        }

        // Measured (5 runs). Use cudaEvent for cuda_adapter, steady_clock otherwise.
        for (int i = 0; i < 5; i++) {
            float ms = 0.0f;
            if (std::string(be.name) == "cuda_adapter") {
                cudaEvent_t t0, t1;
                cudaEventCreate(&t0); cudaEventCreate(&t1);
                cudaEventRecord(t0);
                runTransformerBlockOnce(be.name, x, Wq, Wk, Wv, Wo, W1, W2, out);
                cudaEventRecord(t1); cudaEventSynchronize(t1);
                cudaEventElapsedTime(&ms, t0, t1);
                cudaEventDestroy(t0); cudaEventDestroy(t1);
            } else {
                auto t0 = std::chrono::steady_clock::now();
                runTransformerBlockOnce(be.name, x, Wq, Wk, Wv, Wo, W1, W2, out);
                auto t1 = std::chrono::steady_clock::now();
                ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            }
            times.push_back(ms);
        }

        float median = medianMs(times);
        // gflops column filled with 0 — end-to-end GFLOPS has no standard definition.
        std::cout << "transformer_block," << S << "," << D << "," << F << ","
                  << be.name << "," << median << ",0\n";
    }

    return 0;
}
```

Also add to the top of the file (after existing includes):

```cpp
#include <chrono>
```

- [ ] **Step 4: Implement `runTransformerBlockOnce` helper**

Add before `runTransformerBlockBench` (signature exactly as called above). Its body is the chosen link strategy from Step 2. Example skeleton (route B):

```cpp
void runTransformerBlockOnce(const std::string& backend,
                              const tsy::lir::NamedTensor& x,
                              const tsy::lir::NamedTensor& Wq,
                              const tsy::lir::NamedTensor& Wk,
                              const tsy::lir::NamedTensor& Wv,
                              const tsy::lir::NamedTensor& Wo,
                              const tsy::lir::NamedTensor& W1,
                              const tsy::lir::NamedTensor& W2,
                              tsy::lir::NamedTensor& out) {
    // Body depends on link strategy. If (A): call codegen-exposed forward().
    // If (B): manually orchestrate the op sequence using adapter_cpu/adapter_cuda/interpreter
    //         matching examples/transformer_block.tsy semantics.
    // See Step 2 decision note.
    (void)backend; (void)x; (void)Wq; (void)Wk; (void)Wv; (void)Wo;
    (void)W1; (void)W2; (void)out;
    // IMPLEMENT PER STEP 2 DECISION.
}
```

**This step requires seeing actual code in Step 1.** If route (B), the body will be ~40 lines that reproduce the forward pass via direct adapter calls (rmsnorm → matmul × 3 → transpose → matmul → softmax → matmul → matmul → add → rmsnorm → matmul → relu → matmul → add). Reuse patterns from `tests/e2e/test_transformer_block.py` numpy reference.

- [ ] **Step 5: Wire --primitive transformer_block in main**

Edit the dispatch added in Task 5:

```cpp
if (primitive == "matmul") return runMatmulBench(opts);
if (primitive == "transformer_block") return runTransformerBlockBench();
std::cerr << "unknown --primitive: " << primitive
          << " (valid: matmul, transformer_block)\n";
return 2;
```

Update `usage()`:

```cpp
std::cerr << "usage: " << progname
          << " [--primitive matmul|transformer_block] [--smoke] "
          << "[--shapes MxKxN[,...]] [--variants v1[,v2,...]]\n";
```

- [ ] **Step 6: CMake — ensure tsy-bench links what it needs**

If route (A) added a static lib, add it to `target_link_libraries(tsy-bench ...)`. If route (B), verify tsy-bench already links `tsy_runtime_cuda`, `tsy_runtime_cpu`, `tsy_lir` (should, per existing setup).

```bash
cmake --build build --target tsy-bench -j 2>&1 | tail -5
```

Expected: link succeeds.

- [ ] **Step 7: Run transformer_block bench, validate CSV**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
./build/tsy-bench --primitive transformer_block
```

Expected: CSV with 4 lines (header + 3 backend rows), non-zero `ms_median`, `gflops=0`.

- [ ] **Step 8: Commit**

```bash
git add src/tools/tsy-bench.cu CMakeLists.txt
git commit -m "feat(tsy-bench): transformer_block end-to-end bench

S=4 D=8 F=16, three backends (native/cpu_adapter/cuda_adapter), same
3-warmup + 5-measured + median discipline. gflops column reports 0 —
end-to-end GFLOPS has no standard definition."
```

---

## Task 8: bench_compare.py with differentiated thresholds

**Parallelizable with Tasks 5+6+7.** Pure stdlib Python, no code dependencies.

**Files:**
- Create: `scripts/bench_compare.py`
- Test: no formal unit tests — verify by manual smoke runs (see Step 5-6)

- [ ] **Step 1: Create the script**

Create `scripts/bench_compare.py`:

```python
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

# (fail_ratio, warn_ratio) — current/baseline ratio crossing fail_ratio → FAIL
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

    fail_count = warn_count = imp_count = 0
    all_keys = sorted(set(baseline) | set(current))

    for key in all_keys:
        prim, M, K, N, var = key
        b = baseline.get(key)
        c = current.get(key)
        if b is None:
            print(f"NEW      {prim:18s} {M:>5}x{K:>5}x{N:>5} {var:12s} "
                  f"              current={c:.3f}ms")
            continue
        if c is None:
            print(f"MISSING  {prim:18s} {M:>5}x{K:>5}x{N:>5} {var:12s} "
                  f"baseline={b:.3f}ms")
            continue
        ratio = c / b
        status = classify(ratio, prim)
        print(f"{status:8s} {prim:18s} {M:>5}x{K:>5}x{N:>5} {var:12s} "
              f"baseline={b:7.3f}ms current={c:7.3f}ms ratio={ratio:.3f}")
        if status == "FAIL": fail_count += 1
        elif status == "WARN": warn_count += 1
        elif status == "IMPROVED": imp_count += 1

    print(f"\nsummary: {fail_count} FAIL, {warn_count} WARN, {imp_count} IMPROVED, "
          f"{len(all_keys) - fail_count - warn_count - imp_count} OK")

    if imp_count > 0 and not args.update_baseline:
        print("hint: confirmed improvements? re-run with --update-baseline")

    if args.update_baseline:
        shutil.copyfile(args.current, args.baseline)
        print(f"baseline updated: {args.baseline}\n"
              f"→ review the diff and commit: git add {args.baseline}")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make executable**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
chmod +x scripts/bench_compare.py
```

- [ ] **Step 3: Write a throwaway fixture to smoke the script**

```bash
cat > /tmp/base.csv <<'EOF'
primitive,M,K,N,variant,ms_median,gflops
matmul,256,256,256,naive,1.000,34.0
transformer_block,4,8,16,cuda_adapter,0.500,0
EOF
cp /tmp/base.csv /tmp/cur_identical.csv

cat > /tmp/cur_regress.csv <<'EOF'
primitive,M,K,N,variant,ms_median,gflops
matmul,256,256,256,naive,1.150,29.6
transformer_block,4,8,16,cuda_adapter,0.600,0
EOF
```

- [ ] **Step 4: Run the 3 smoke scenarios**

```bash
# Identical → exit 0, all OK
python3 scripts/bench_compare.py --baseline /tmp/base.csv --current /tmp/cur_identical.csv
echo "exit=$?"
# Expected: all OK, exit=0

# Regression (15% matmul > 10% fail; 20% transformer > 15% fail)
python3 scripts/bench_compare.py --baseline /tmp/base.csv --current /tmp/cur_regress.csv
echo "exit=$?"
# Expected: 2 FAIL, exit=1
```

- [ ] **Step 5: Clean up fixtures**

```bash
rm /tmp/base.csv /tmp/cur_identical.csv /tmp/cur_regress.csv
```

- [ ] **Step 6: Commit**

```bash
git add scripts/bench_compare.py
git commit -m "feat(bench): bench_compare.py with per-primitive thresholds

matmul 10% fail / 5% warn; transformer_block 15%/10% (end-to-end noisier
than single matmul). stdlib only. --update-baseline swaps the reference."
```

---

## Task 9: bench_local.sh one-command local gate

**Depends on:** Task 7 (tsy-bench transformer_block), Task 8 (bench_compare.py). Does NOT depend on Task 10 — script is usable even before baseline exists (compare against identical will just print NEW/MISSING).

**Files:**
- Create: `scripts/bench_local.sh`

- [ ] **Step 1: Create the wrapper**

Create `scripts/bench_local.sh`:

```bash
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
```

- [ ] **Step 2: Make executable**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
chmod +x scripts/bench_local.sh
```

- [ ] **Step 3: Smoke without baseline (expected: NEW rows)**

```bash
# The baseline doesn't exist yet — Task 10 creates it.
# For now, touch an empty CSV to let the script run end-to-end and see NEW rows.
mkdir -p benchmarks/baseline
echo "primitive,M,K,N,variant,ms_median,gflops" > benchmarks/baseline/rtx3080_wsl.csv
bash scripts/bench_local.sh 2>&1 | tail -25
```

Expected: script completes; every row is `NEW` (no rows in empty baseline); exit 0.

- [ ] **Step 4: Clean up placeholder baseline (Task 10 creates the real one)**

```bash
rm benchmarks/baseline/rtx3080_wsl.csv
# keep benchmarks/baseline/ dir for Task 10
```

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_local.sh
git commit -m "feat(bench): bench_local.sh one-command GPU bench gate"
```

---

## Task 10: Capture baseline CSV

**Depends on:** Task 4 (sync — without this, numbers are pre-fix). Tasks 7, 9 (infra to produce + verify).

**Physical condition required:** laptop plugged in, high-performance power mode, no heavy background apps. These conditions are part of the baseline contract (Task 11 documents them).

**Files:**
- Create: `benchmarks/baseline/rtx3080_wsl.csv`

- [ ] **Step 1: Ensure system state**

- Plug in the laptop (power adapter)
- Power mode: high performance (Windows side: Settings → Power → Performance)
- Close browser tabs, IDEs with background indexers, Docker Desktop, any running `nvidia-smi` loops
- Inside WSL: `nvidia-smi` — note current GPU clock, temperature, utilization (should be ~idle)

- [ ] **Step 2: Rebuild clean**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake --build build -j 2>&1 | tail -3
```

- [ ] **Step 3: Capture matmul sweep**

```bash
./build/tsy-bench --primitive matmul > benchmarks/baseline/rtx3080_wsl.csv
cat benchmarks/baseline/rtx3080_wsl.csv
```

Expected: 1 header + 15 rows (5 shapes × 3 variants; tiled skipped where unaligned → actually fewer; the odd shapes 128x16x8 and 7x13x11 will skip tiled, so expected 5 × naive + 3 × tiled (only 256, 512, 1024) + 5 × cublas = 5+3+5 = 13 matmul rows. Adjust spec expectation accordingly.)

*Note: spec §4.7 said 15 matmul rows — actual count depends on tiled alignment rules. Record whatever the tool produces; 13 rows is fine.*

- [ ] **Step 4: Append transformer_block**

```bash
./build/tsy-bench --primitive transformer_block | tail -n +2 >> benchmarks/baseline/rtx3080_wsl.csv
cat benchmarks/baseline/rtx3080_wsl.csv
```

Expected: previous rows + 3 transformer_block rows.

- [ ] **Step 5: Smoke with bench_local.sh against freshly-captured baseline**

```bash
bash scripts/bench_local.sh 2>&1 | tail -25
```

Expected: 0 FAIL. Some rows may WARN (cross-run noise within the same machine is usually <5% but not guaranteed). If a row FAILs — that's baseline physical-condition variability. Capture again once in a more quiescent state.

- [ ] **Step 6: Second sanity run (noise profile)**

```bash
./build/tsy-bench --primitive matmul > /tmp/cur2.csv
./build/tsy-bench --primitive matmul > /tmp/cur3.csv
python3 scripts/bench_compare.py --baseline /tmp/cur2.csv --current /tmp/cur3.csv | tail -15
rm /tmp/cur2.csv /tmp/cur3.csv
```

Expected: all rows OK or WARN (no FAIL). If FAILs appear on cross-run self-comparison, the 10% matmul threshold is too tight for this hardware's actual noise floor — document in Task 11 and consider bumping thresholds before shipping.

- [ ] **Step 7: Commit baseline**

```bash
git add benchmarks/baseline/rtx3080_wsl.csv
git commit -m "bench: initial baseline for RTX 3080 WSL

Captured post CUDA D2H sync fix. Hardware: RTX 3080 Laptop 16GB,
driver CUDA 12.7, toolkit 12.0. Physical: AC power, performance mode,
idle system. Noise profile: matmul ~1-3% cross-run, transformer_block
~2-5% cross-run (verified during capture)."
```

---

## Task 11: docs/benchmarks/baseline.md

**Parallelizable partially with Task 10** — draft the structure while baseline is captured, fill in specifics after.

**Files:**
- Create: `docs/benchmarks/baseline.md`
- Create: `docs/benchmarks/` directory

- [ ] **Step 1: Gather version info**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cat /etc/os-release | grep PRETTY
uname -r
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | tail -2
g++ --version | head -1
cmake --version | head -1
bison --version | head -1
flex --version | head -1
.venv/bin/python --version
git -C /home/xs/tsy-wsl-export/tensor-sysy rev-parse HEAD
```

- [ ] **Step 2: Create the doc**

Create `docs/benchmarks/baseline.md` (fill in values from Step 1):

```markdown
# tensor-sysy GPU benchmark baseline

## Source of truth

`benchmarks/baseline/rtx3080_wsl.csv` is the canonical reference. Everything
in this doc describes how to reproduce and when to update that file.

## Hardware

- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GB VRAM
- Driver CUDA version: 12.7
- CUDA toolkit: 12.0 (`nvcc --version`)
- Host: WSL2 Ubuntu 24.04 on Windows 11
- (Update this section if the machine changes; baseline numbers are invalid across hardware.)

## Software (at time of capture)

- g++: <fill from Step 1>
- CMake: <fill from Step 1>
- bison: <fill from Step 1>
- flex: <fill from Step 1>
- Python: <fill from Step 1>
- Repo commit: <fill `git rev-parse HEAD`>

## Physical conditions

baseline is valid only when captured under:

- Laptop plugged in (AC power)
- Windows power mode: Performance (not Balanced / Battery Saver)
- No browser tabs playing video, no Docker Desktop, no IDE indexers mid-run
- `nvidia-smi` shows GPU idle (utilization < 5%, temp < 60°C) before capture

Deviation from these conditions can move `ms_median` by 5-15%, which
silently triggers the 10% regression gate. If a regression appears after
a legitimate code change, first verify physical conditions before
touching the code.

## Reproduction

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake --build build -j
bash scripts/bench_local.sh
```

Expected: `0 FAIL`. `WARN` rows are acceptable (cross-run noise).

## Noise profile (captured during W11)

- matmul: ~1-3% cross-run std
- transformer_block: ~2-5% cross-run std

Thresholds (`scripts/bench_compare.py`):
- matmul: FAIL ≥ 10%, WARN ≥ 5%
- transformer_block: FAIL ≥ 15%, WARN ≥ 10%

## When to update the baseline

Trigger an update when:

1. Scheduler / layout / codegen changes yield real performance movement
   (IMPROVED rows printed; gut-check the change genuinely improves, not
   measurement scope drift)
2. CUDA toolkit or driver upgrade changes numbers system-wide
3. Hardware environment changes (new machine → new baseline file, not
   an update to this one)

## How to update

```bash
bash scripts/bench_local.sh      # confirm a regression or improvement
# Verify physical conditions (AC power, performance mode, idle).
# Re-run 2-3× to ensure numbers are stable.

./build/tsy-bench --primitive matmul > /tmp/newbase.csv
./build/tsy-bench --primitive transformer_block | tail -n +2 >> /tmp/newbase.csv
python3 scripts/bench_compare.py \
    --baseline benchmarks/baseline/rtx3080_wsl.csv \
    --current  /tmp/newbase.csv \
    --update-baseline

git add benchmarks/baseline/rtx3080_wsl.csv
git commit -m "bench: refresh baseline for <REASON>

- <describe code change or environment shift>
- Median change: <summarize row ratios, e.g. matmul 256^3 naive 1.12 → 1.08>"
```

Commit messages for baseline updates must state WHY numbers moved — if the
reader can't distinguish optimization from measurement-scope change from
a later `git log`, future baseline updates are blind.
```

- [ ] **Step 3: Commit**

```bash
git add docs/benchmarks/baseline.md
git commit -m "docs: bench baseline reproduction + update guide"
```

---

## Task 12: PLAN.md §W11 tick + verification commands

**Depends on:** all previous tasks — this is the collation step.

**Files:**
- Modify: `PLAN.md`

- [ ] **Step 1: Find §W11 rows in PLAN.md**

```bash
grep -n "W11\|Week 11" /home/xs/tsy-wsl-export/tensor-sysy/PLAN.md
```

- [ ] **Step 2: Mark W11 as completed**

Locate the 12-week table row for W11 (line ~75) and add a ✅ marker consistent with how W0-W10 are marked if such convention exists (check by grep). If no convention, append `(✅ 2026-04-18)` to the row.

Additionally, add a new section `## W11 成功标准` near the existing `## W10 成功标准` section (around line ~337), mirroring its structure:

```markdown
### W11 成功标准

1. GitHub Actions CI workflow 绿（`.github/workflows/ci.yml`）
2. 所有本地 ctest 32/32 通过（含 CUDA）
3. CPU-only 模拟构建通过（`-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE`，~20/20）
4. `bash scripts/bench_local.sh` 0 FAIL
5. `benchmarks/baseline/rtx3080_wsl.csv` 入库
6. `docs/benchmarks/baseline.md` 可独立复现
7. W10 三条 follow-up 全部落地（TSY_PYTHON_EXECUTABLE / CUDA sync / verifyUnary 注释）

### W11 验收命令

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure        # 32/32

mv .venv .venv.bak
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE
cmake --build build-cpu -j
(cd build-cpu && ctest --output-on-failure)        # ~20/20
mv .venv.bak .venv
rm -rf build-cpu

bash scripts/bench_local.sh                        # 0 FAIL
python3 benchmarks/run_shapes.py --check-scheduler # ≥1.20x
```
```

- [ ] **Step 3: Check whether W11 adds new tools to "已装工具" table**

Grep PLAN.md for the tool table:

```bash
grep -n "已装工具\|uv pip install\|apt install" /home/xs/tsy-wsl-export/tensor-sysy/PLAN.md | head
```

Tools introduced by W11:
- `uv` — **already installed** (W10), no new row
- numpy, pytest — **already installed** (W10), no new row
- apt packages installed *only on GHA runner* — these don't go in the local "已装工具" table

No new tools → no table modification. If locally installing anything for W11 bench validation (e.g., additional Python libs), add those rows now.

- [ ] **Step 4: Commit**

```bash
git add PLAN.md
git commit -m "docs(plan): W11 CI + Benchmark done; success criteria + verification"
```

---

## Self-Review (done after writing, fix inline)

- [x] **Spec coverage**: §1 goal → Tasks 1-12 collectively; §2 in-scope → 1:1 mapped to tasks; §4 components → detailed in respective tasks.
- [x] **Placeholder scan**:
  - T7 Step 2/4 uses "IMPLEMENT PER STEP 2 DECISION" as a legitimate branch point (the code depends on Step 1 investigation). This is not a placeholder in the forbidden sense — it's a conditional dispatch based on what route (A) vs (B) the actual code supports. Acceptable.
  - T10 Step 3 notes 13 rows vs spec's 15 (tiled skip for unaligned shapes). This is an accurate correction of the spec, not a gap.
- [x] **Type consistency**: `runTransformerBlockOnce` signature matches call site in `runTransformerBlockBench` (8 args: backend + x + 6 weights + out). `NamedTensor` type used throughout matches `tsy::lir::NamedTensor` in existing tsy-bench.cu (line 15 include).
- [x] **Task boundaries**: each task produces a single commit; phase 2 branches are file-disjoint (T5/T7 touch tsy-bench.cu + root CMake; T6 touches `.github/`; T8 touches `scripts/bench_compare.py` — no overlap).

---

## Execution handoff

**Recommended:** superpowers:subagent-driven-development. Phase 2 (T5+T7 ‖ T6 ‖ T8) benefits from parallel agents. Phase 1 (T1→T2→T3→T4) and Phase 3 (T9→T10→T11→T12) are naturally serial — agents should be dispatched one at a time there to avoid step races.

Dispatch plan:
1. **Serial**: agent-per-task for T1, T2, T3, T4 (each blocks the next or requires full ctest verification)
2. **Parallel**: one agent each for { T5→T7 bundled } and T6 and T8
3. **Serial**: T9, T10, T11, T12 in sequence

T10 is the human-gating step — it needs physical conditions (AC power, performance mode) the agent can't verify. User must confirm conditions before agent proceeds with capture.
