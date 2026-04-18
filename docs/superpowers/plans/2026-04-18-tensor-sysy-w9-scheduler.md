# W9 — CUDA Scheduler + Layout-Lowering Skeleton + Benchmark · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a shape-aware, compile-time CUDA matmul scheduler with 3 kernel variants (naive / register-tiled / cuBLAS); the scheduler runs as a new LIR pass that annotates each matmul Call with `attrs["variant"]`. Include a no-op `LayoutLoweringPass` placeholder (W10 fills in the body). Ship a CUDA-event precision benchmark: C++ `tsy-bench` binary + thin Python driver `benchmarks/run_shapes.py`.

**Architecture:** Extend `Stmt` with a generic `attrs` string-map. Extend `PassManager` with a parallel `LirPassFn` container so passes can now operate on LIR modules (not just HIR). The scheduler pass reads shapes from LIR Buffers, consults a hard-coded lookup table, writes `attrs["variant"]`. The CUDA adapter / codegen both read that same field. Everything is gated on `TSY_HAVE_RUNTIME_CUDA` so non-CUDA machines see zero regression.

**Tech Stack:** C++17, CUDA 12 / cuBLAS, CMake 3.18+, Python 3.8+ standard library only (no numpy/pandas), pre-existing flex/bison/nvcc toolchain.

**Spec:** `docs/superpowers/specs/2026-04-18-tensor-sysy-w9-scheduler-design.md`
**Expected final state:** ctest **29/29 green** on WSL + RTX 3080 (25 from W0-W8 + 4 new); still 21/21 on non-CUDA machines.
**Starting HEAD:** commit `f1f62fa` (W8 complete).

---

## File Structure

### NEW files (under `/home/xs/tsy-wsl-export/tensor-sysy`)
- `src/passes/layout_lowering.cpp` — no-op LIR pass, W10 fills in body
- `src/passes/schedule_cuda.cpp` — LIR pass that picks matmul variant via `pickMatmulVariant(M, K, N)`
- `src/tools/tsy-bench.cu` — C++ benchmark binary, CUDA-event timing
- `benchmarks/run_shapes.py` — Python driver over tsy-bench, stdlib only
- `tests/passes/test_schedule_cuda_cases.cpp` — programmatic LIR module construction + scheduler assertions
- `examples/run_matmul_medium.tsy` — 128×128×128 matmul (triggers `variant=tiled`)
- `examples/run_matmul_large.tsy` — 512×512×512 matmul (triggers `variant=cublas`)

### MODIFIED files
- `src/lir/ir.h` — add `attrs` field to `Stmt`
- `src/lir/printer.cpp` — print attrs `{key="val", ...}` when non-empty
- `src/passes/pass_manager.h` — add `LirPassFn` type, `addLir()`, `runLir()` method declarations
- `src/passes/pass_manager.cpp` — implement them, wire into `buildPipelineO1`
- `src/tools/tsc.cpp` — call `pm.runLir(*lmod, diag)` after lowerHirToLir in emit-lir / emit-cpp / emit-cu / run-lir
- `src/runtime/adapter_cuda.h` — matmul signature gains `const std::string& variant = ""`
- `src/runtime/adapter_cuda.cu` — add two new kernels + variant dispatch + executor reads `attrs["variant"]`
- `src/codegen/cuda.cpp` — emit variant string as last arg in `adapterMatMulCuda(...)` call
- `CMakeLists.txt` — add `schedule_cuda.cpp` + `layout_lowering.cpp` to `tsy_passes`; add `tsy-bench` target gated on `TSY_HAVE_RUNTIME_CUDA`
- `tests/CMakeLists.txt` — register 4 new ctests (`pass_schedule_cuda_cases`, `cli_emit_lir_schedule_shows_variant` × 2, `cli_bench_smoke`)

### UNCHANGED (but referenced for parallelism)
- `src/runtime/adapter_cpu.*` — CPU adapter doesn't change; variant concept is CUDA-only
- `src/codegen/cpp.cpp` — CPU codegen doesn't change
- `src/hir/*` — HIR unchanged
- `src/passes/{verify,const_fold,dce}.cpp` — existing HIR passes unchanged

---

## Commit Plan

Eight logical milestones per spec §9, one commit each:

1. `feat(lir): add Stmt.attrs + printer support` (Tasks 1–2)
2. `feat(passes): extend PassManager with LirPassFn + stub layout-lowering/schedule-cuda` (Tasks 3–6)
3. `feat(runtime): add matmul naive + reg-tiled kernels with variant dispatch` (Tasks 7–10)
4. `feat(passes): implement ScheduleCudaPass shape lookup` (Tasks 11–14)
5. `feat(tools): tsy-bench C++ binary` (Tasks 15–17)
6. `feat(tools): benchmarks/run_shapes.py driver + CLI ctest` (Tasks 18–20)
7. `feat(codegen): emit variant arg in emit-cu` (Tasks 21–23)
8. `docs: W9 spec` (Task 24 — spec file is already present, just clean-ups + final verify)

---

## Task 1: Add `Stmt.attrs` to LIR + printer support

**Goal:** Teach the LIR `Stmt` struct to carry arbitrary string-string metadata, and have the printer render it at call sites when non-empty. Preserves W8 behaviour when attrs are empty (so no existing golden / regex test breaks).

**Files:**
- Modify: `src/lir/ir.h`
- Modify: `src/lir/printer.cpp`

- [ ] **Step 1: Modify `src/lir/ir.h` — add `#include <unordered_map>` and the `attrs` field**

Find the `struct Stmt` declaration (currently around line 39). Add `<unordered_map>` to the includes at the top of the file if not already there. Replace the struct body with:

```cpp
struct Stmt {
    StmtKind kind = StmtKind::Call;
    std::string primitive;         // "matmul" / "add" / "softmax" / ...
    std::vector<int> operand_bufs; // indices into Function::buffers.
    int result_buf = -1;           // index into Function::buffers; -1 = none.
    tsy::SourceLocation loc;

    // Generic attribute map. Passes write into this (e.g. ScheduleCudaPass
    // sets "variant" for matmul calls). The LIR printer emits these as a
    // trailing `{k="v", ...}` group, sorted by key for golden stability.
    // Default-empty preserves pre-W9 behaviour (printouts / codegen are
    // byte-identical to W8 when no pass writes into attrs).
    std::unordered_map<std::string, std::string> attrs;
};
```

- [ ] **Step 2: Modify `src/lir/printer.cpp` — emit attrs after the operand list**

At the top of the file, add `#include <algorithm>` and `#include <vector>` if missing (std::sort + std::pair need them).

Find the call-printing loop (currently lines 46–60). Replace the body of the `for (const auto& s : f.body)` loop (from the `spaces(os, indent + 2);` line through the final `os << "\n";` inside the loop) with:

```cpp
    for (const auto& s : f.body) {
        spaces(os, indent + 2);
        if (s.kind == StmtKind::Return) {
            os << "return\n";
            continue;
        }
        if (s.result_buf >= 0) {
            os << "%" << f.buffers[s.result_buf].name << " = ";
        }
        os << "call " << s.primitive;
        for (size_t i = 0; i < s.operand_bufs.size(); ++i) {
            os << (i == 0 ? " " : ", ");
            os << "%" << f.buffers[s.operand_bufs[i]].name;
        }
        if (!s.attrs.empty()) {
            // Sort by key for golden stability (unordered_map iteration order
            // is not portable).
            std::vector<std::pair<std::string, std::string>> kv(
                s.attrs.begin(), s.attrs.end());
            std::sort(kv.begin(), kv.end());
            os << " {";
            for (size_t i = 0; i < kv.size(); ++i) {
                if (i) os << ", ";
                os << kv[i].first << "=\"" << kv[i].second << "\"";
            }
            os << "}";
        }
        os << "\n";
    }
```

- [ ] **Step 3: Build + run the full suite — zero-attrs path must stay byte-identical**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: `100% tests passed, 0 tests failed out of 25`.
(Because no pass currently writes to `attrs`, every `call` still prints exactly as before.)

- [ ] **Step 4: Self-verify with `tsc emit-lir`**

```bash
./build/tsc emit-lir examples/run_matmul_tiny.tsy | head -12
```

Expected: output identical to pre-W9 (no trailing `{...}` group on call lines).

---

## Task 2: Commit 1

- [ ] **Step 1: Commit**

```bash
git add src/lir/ir.h src/lir/printer.cpp
git commit -m "$(cat <<'EOF'
feat(lir): add Stmt.attrs + printer support

Extend LIR Stmt with a generic std::unordered_map<string,string> attrs
field that later passes (ScheduleCudaPass) will write into. Printer
emits a trailing {k="v", ...} group when attrs is non-empty, sorted by
key for golden stability. Defaults to empty so pre-W9 printouts are
byte-identical and all existing golden / regex tests stay green.
EOF
)"
git log --oneline -1
```

Expected: new commit hash; `git log` shows the new commit at HEAD.

---

## Task 3: Extend `PassManager` with `LirPassFn` + `runLir`

**Goal:** Add a parallel LIR-pass container to `PassManager` so LIR passes can be registered and run by name, sharing the `--disable-pass` mechanism with HIR passes.

**Files:**
- Modify: `src/passes/pass_manager.h`
- Modify: `src/passes/pass_manager.cpp`

- [ ] **Step 1: Modify `src/passes/pass_manager.h` — add `LirPassFn` type and API**

Add `#include "../lir/ir.h"` near the top (after the existing `#include "../hir/ops.h"`).

Inside the `tsy::passes` namespace, add:

```cpp
using LirPassFn =
    std::function<void(tsy::lir::Module&, tsy::DiagnosticEngine&)>;
```

Modify the `PassManager` class so it owns BOTH pass vectors. The final class body should read:

```cpp
class PassManager {
   public:
    void add(std::string name, PassFn fn);
    void addLir(std::string name, LirPassFn fn);

    void disable(const std::string& name);
    void enable(const std::string& name);
    bool isDisabled(const std::string& name) const;

    std::vector<std::string> names() const;     // includes both HIR + LIR
    std::vector<std::string> lirNames() const;  // LIR-only subset

    void run(tsy::hir::Module& m, tsy::DiagnosticEngine& diag) const;
    void runLir(tsy::lir::Module& m, tsy::DiagnosticEngine& diag) const;

   private:
    struct HirEntry { std::string name; PassFn fn; };
    struct LirEntry { std::string name; LirPassFn fn; };
    std::vector<HirEntry> passes_;
    std::vector<LirEntry> lir_passes_;
    std::unordered_set<std::string> disabled_;
};
```

Below the existing declarations of `runVerifier` / `runConstFold` / `runDCE`, add:

```cpp
// LIR passes (W9).
void runLayoutLowering(tsy::lir::Module& m, tsy::DiagnosticEngine& diag);
void runScheduleCuda(tsy::lir::Module& m, tsy::DiagnosticEngine& diag);
```

Keep `buildPipelineO0()` / `buildPipelineO1()` signatures unchanged.

- [ ] **Step 2: Modify `src/passes/pass_manager.cpp` — implement the new members**

Replace the whole file with:

```cpp
#include "pass_manager.h"

namespace tsy::passes {

void PassManager::add(std::string name, PassFn fn) {
    passes_.push_back({std::move(name), std::move(fn)});
}

void PassManager::addLir(std::string name, LirPassFn fn) {
    lir_passes_.push_back({std::move(name), std::move(fn)});
}

void PassManager::disable(const std::string& name) { disabled_.insert(name); }
void PassManager::enable(const std::string& name) { disabled_.erase(name); }
bool PassManager::isDisabled(const std::string& name) const {
    return disabled_.count(name) != 0;
}

std::vector<std::string> PassManager::names() const {
    std::vector<std::string> out;
    out.reserve(passes_.size() + lir_passes_.size());
    for (const auto& e : passes_) out.push_back(e.name);
    for (const auto& e : lir_passes_) out.push_back(e.name);
    return out;
}

std::vector<std::string> PassManager::lirNames() const {
    std::vector<std::string> out;
    out.reserve(lir_passes_.size());
    for (const auto& e : lir_passes_) out.push_back(e.name);
    return out;
}

void PassManager::run(tsy::hir::Module& m, tsy::DiagnosticEngine& diag) const {
    for (const auto& e : passes_) {
        if (disabled_.count(e.name)) continue;
        e.fn(m, diag);
        if (diag.hasErrors()) break;
    }
}

void PassManager::runLir(tsy::lir::Module& m,
                         tsy::DiagnosticEngine& diag) const {
    for (const auto& e : lir_passes_) {
        if (disabled_.count(e.name)) continue;
        e.fn(m, diag);
        if (diag.hasErrors()) break;
    }
}

PassManager buildPipelineO0() {
    PassManager pm;
    pm.add("verify", runVerifier);
    return pm;
}

PassManager buildPipelineO1() {
    PassManager pm;
    pm.add("verify", runVerifier);
    pm.add("const-fold", runConstFold);
    pm.add("dce", runDCE);
    pm.add("verify-post", runVerifier);
    // LIR passes run AFTER hir-to-lir lowering (callers handle that order).
    pm.addLir("layout-lowering", runLayoutLowering);
    pm.addLir("schedule-cuda", runScheduleCuda);
    return pm;
}

}  // namespace tsy::passes
```

- [ ] **Step 3: Build just the passes library to verify declarations compile**

```bash
cmake --build build -j --target tsy_passes 2>&1 | tail -10
```

Expected: **link will FAIL** with unresolved references to `runLayoutLowering` and `runScheduleCuda`. That's the red stage — Tasks 4 + 5 provide the definitions.

---

## Task 4: No-op `LayoutLoweringPass` stub

**Goal:** Create the no-op placeholder that W10 will fill in.

**Files:**
- Create: `src/passes/layout_lowering.cpp`

- [ ] **Step 1: Create `src/passes/layout_lowering.cpp`**

Exact content:

```cpp
// W9: no-op placeholder. W10 fills in real layout transformations.
//
// Where this pass is going (W10 transformer block):
//   - Recognise non-canonical matmul operand layouts and emit transposes
//     that make them canonical before ScheduleCudaPass picks a kernel.
//   - Expand View/Permute ops (HIR's currently-reserved enums) into
//     adjacent buffer reshape + copy statements.
//
// For W9 this pass is registered in PassManager O1 pipeline so the
// structural hookup is done once. W10 can drop in the body here without
// touching pass_manager / CLI / tsc.

#include "pass_manager.h"

namespace tsy::passes {

void runLayoutLowering(tsy::lir::Module& /*m*/,
                       tsy::DiagnosticEngine& /*diag*/) {
    // intentionally empty
}

}  // namespace tsy::passes
```

---

## Task 5: Declaration-only `ScheduleCudaPass` stub

**Goal:** Unblock linking. Real shape-lookup logic lands in Task 11.

**Files:**
- Create: `src/passes/schedule_cuda.cpp`

- [ ] **Step 1: Create `src/passes/schedule_cuda.cpp`** (empty body — Task 11 fills in)

```cpp
// W9: empty stub — linker needs this symbol to exist so Task 6's CMake
// change and the PassManager registration compile and link. Task 11
// replaces the body with the real shape-lookup.

#include "pass_manager.h"

namespace tsy::passes {

void runScheduleCuda(tsy::lir::Module& /*m*/,
                     tsy::DiagnosticEngine& /*diag*/) {
    // intentionally empty — Task 11 replaces this.
}

}  // namespace tsy::passes
```

---

## Task 6: Wire LIR passes into build + caller sites

**Goal:** Add the new sources to `tsy_passes`, then make every consumer of the pipeline in `tsc.cpp` call `pm.runLir(*lmod, diag)` right after `lowerHirToLir`.

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `src/tools/tsc.cpp`

- [ ] **Step 1: Modify `CMakeLists.txt` — add the two new sources to `tsy_passes`**

Find the existing `tsy_passes` library block. It currently reads roughly:

```cmake
add_library(tsy_passes STATIC
    src/passes/pass_manager.cpp
    src/passes/verify.cpp
    src/passes/const_fold.cpp
    src/passes/dce.cpp
)
```

Replace with:

```cmake
add_library(tsy_passes STATIC
    src/passes/pass_manager.cpp
    src/passes/verify.cpp
    src/passes/const_fold.cpp
    src/passes/dce.cpp
    src/passes/layout_lowering.cpp
    src/passes/schedule_cuda.cpp
)
```

- [ ] **Step 2: Modify `src/tools/tsc.cpp` — insert `pm.runLir` after every `lowerHirToLir` call**

There are 3 call sites that invoke `lowerHirToLir`: `cmdEmitLir`, `cmdEmitCpp`, `cmdEmitCu`, `cmdRunLir`. (That's 4 places.) In each, right after `auto lmod = tsy::lir::lowerHirToLir(*hmod, diag);` and the error check, add a `pm.runLir` call.

But — the current `parseAndRunPipeline` helper builds the `PassManager` locally and doesn't return it. We need the `PassManager` to be reachable at the caller. Refactor `parseAndRunPipeline` so it also returns the pass manager. Find the helper (around lines 115–146) and change its signature + body.

Before:
```cpp
std::unique_ptr<tsy::hir::Module> parseAndRunPipeline(const Options& o,
                                                      tsy::DiagnosticEngine& diag) {
    // ... returns the HIR module only
}
```

After — keep behaviour identical for callers but expose the pass manager via an out-param:

```cpp
std::unique_ptr<tsy::hir::Module> parseAndRunPipeline(
        const Options& o,
        tsy::DiagnosticEngine& diag,
        tsy::passes::PassManager* out_pm = nullptr) {
    auto r = tsy::parseFile(o.path);
    if (!r.ok) {
        r.diagnostics.print(std::cerr);
        std::cerr << "parse failed: " << o.path << "\n";
        return nullptr;
    }
    auto mod = tsy::hir::lowerAstToHir(*r.ast, r.diagnostics);
    if (!mod || r.diagnostics.hasErrors()) {
        r.diagnostics.print(std::cerr);
        std::cerr << "lowering failed: " << o.path << "\n";
        return nullptr;
    }
    auto pm = buildPipeline(o);
    pm.run(*mod, r.diagnostics);
    for (const auto& d : r.diagnostics.diagnostics()) {
        diag.report(d.level, d.loc, d.message);
    }
    if (r.diagnostics.hasErrors()) {
        std::cerr << "pipeline failed: " << o.path << "\n";
        return nullptr;
    }
    if (out_pm) *out_pm = std::move(pm);
    return mod;
}
```

Now update each of the 4 LIR-consuming commands. For `cmdEmitLir`:

```cpp
int cmdEmitLir(const Options& o) {
    tsy::DiagnosticEngine diag;
    tsy::passes::PassManager pm;
    auto hmod = parseAndRunPipeline(o, diag, &pm);
    if (!hmod) return 1;
    auto lmod = tsy::lir::lowerHirToLir(*hmod, diag);
    if (!lmod || diag.hasErrors()) {
        diag.print(std::cerr);
        std::cerr << "lir lowering failed: " << o.path << "\n";
        return 1;
    }
    pm.runLir(*lmod, diag);
    if (diag.hasErrors()) {
        diag.print(std::cerr);
        std::cerr << "lir pipeline failed: " << o.path << "\n";
        return 1;
    }
    tsy::lir::printModule(std::cout, *lmod);
    return 0;
}
```

Apply the **same pattern** to `cmdEmitCpp`, `cmdEmitCu`, `cmdRunLir`: pass `&pm` to `parseAndRunPipeline`, and add the `pm.runLir(*lmod, diag);` block (with its own error check) right after the lowerHirToLir error check but before the consumer's action (printing, codegen, or execution).

- [ ] **Step 3: Build + run full suite — still 25/25 green, LIR passes are currently no-ops so no behaviour change**

```bash
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: `100% tests passed, 0 tests failed out of 25`.

- [ ] **Step 4: Sanity — confirm `names()` lists LIR passes too**

```bash
./build/tsc emit-hir --opt=O1 examples/run_matmul_tiny.tsy >/dev/null && echo OK
```

(This exercises the O1 pipeline's HIR portion. Nothing prints about LIR passes since they're no-op, but the code path is now live.)

Expected: `OK`.

---

## Task 7: Commit 2

- [ ] **Step 1: Commit**

```bash
git add src/passes/pass_manager.h src/passes/pass_manager.cpp \
        src/passes/layout_lowering.cpp src/passes/schedule_cuda.cpp \
        CMakeLists.txt src/tools/tsc.cpp
git commit -m "$(cat <<'EOF'
feat(passes): extend PassManager with LirPassFn + stub layout-lowering/schedule-cuda

Adds a parallel LIR-pass container to PassManager. pm.add()/addLir()
register different fn types; pm.run()/runLir() execute them; a single
disabled_ set makes --disable-pass=<name> work across both stages.

Two stubs ship registered in buildPipelineO1():
- layout-lowering: no-op placeholder (W10 Permute/View lowering)
- schedule-cuda:   no-op placeholder (W9 Task 11 fills in the body)

tsc.cpp now runs pm.runLir() after lowerHirToLir() in emit-lir / emit-cpp
/ emit-cu / run-lir. Because both stubs are no-op the W8 test suite is
25/25 green unchanged.
EOF
)"
git log --oneline -1
```

---

## Task 8: Add `variant` parameter to `adapterMatMulCuda`

**Goal:** Extend the adapter API so callers can request a specific kernel variant. Default (empty string) preserves W8 behaviour (= cuBLAS).

**Files:**
- Modify: `src/runtime/adapter_cuda.h`
- Modify: `src/runtime/adapter_cuda.cu`

- [ ] **Step 1: Modify `src/runtime/adapter_cuda.h`**

Find the existing `adapterMatMulCuda` declaration and change it to accept a trailing `variant` parameter with default value `""`:

```cpp
// MatMul: C[M,N] = A[M,K] @ B[K,N], FP32, row-major.
// variant: "" (default) or "cublas" → cuBLAS sgemm (W8 path).
//          "naive" → one-thread-per-output naive kernel.
//          "tiled" → 128x128 register-tiled kernel; requires
//                    M%128==0 && N%128==0 && K%8==0 — caller must
//                    pre-check (the schedule-cuda pass does this).
void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c,
                       const std::string& variant = "");
```

- [ ] **Step 2: Modify `src/runtime/adapter_cuda.cu` — signature only (kernels come in Task 9)**

Find the `adapterMatMulCuda` definition. Change the first line of its signature to match the header (accept `variant` with no default here; default only lives in the header):

```cpp
void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c,
                       const std::string& variant) {
```

Leave the body unchanged for now (still only cuBLAS). Task 9 adds the dispatch.

**Important:** the parameter is currently unused — add `(void)variant;` as the first line of the body to silence `-Wunused-parameter` if it fires. Task 9 removes that line.

- [ ] **Step 3: Build — expect clean rebuild with no test regressions**

```bash
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: `100% tests passed, 0 tests failed out of 25` (no behaviour change — caller in executor still doesn't pass variant, which means default "" → cuBLAS).

---

## Task 9: Implement naive + register-tiled kernels and variant dispatch

**Goal:** Two new CUDA kernels (naive, reg-tiled) + dispatch logic in `adapterMatMulCuda`.

**Files:**
- Modify: `src/runtime/adapter_cuda.cu`

- [ ] **Step 1: Add the naive kernel above `adapterMatMulCuda`**

Just above the `adapterMatMulCuda` function (which currently contains only the cuBLAS code), add:

```cpp
// One thread computes one output element via a dot-product over K.
// Launch: block(32,32), grid(ceil(N/32), ceil(M/32)). Block has 1024
// threads (CUDA max) to maximise occupancy on small shapes where the
// scheduler picks this variant.
__global__ void matmulNaiveKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
```

- [ ] **Step 2: Add the register-tiled kernel**

Right after `matmulNaiveKernel`, add:

```cpp
// Register-tiled GEMM — 128x128 output tile per block, 8x8 register tile
// per thread, BK=8 k-strip. Ported from mini-llm-engine/cuda-kernels/gemm/
// gemm.cu (gemm_register_tiled). Requires M%128==0 && N%128==0 && K%8==0.
// Launch: block(16,16)=256 threads, grid(N/128, M/128).
//
// Design notes (detail in the source file it's ported from):
//   - As[BM][BK+1]: +1 col pad kills bank conflicts on the inner A loads.
//   - Bs[BK][BN+4]: +4 col pad reduces the inherent 4-way conflict on B.
//   - Each thread loads one float4 from A and one float4 from B per
//     k-strip; registers hold TM=TN=8 so the outer-product loop has 64
//     FMAs per k step (high arithmetic intensity).
#define TSY_BM 128
#define TSY_BN 128
#define TSY_BK 8
#define TSY_TM 8
#define TSY_TN 8
__global__ void matmulTiledKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * (TSY_BN / TSY_TN) + tx;

    const int bm = blockIdx.y * TSY_BM;
    const int bn = blockIdx.x * TSY_BN;

    __shared__ float As[TSY_BM][TSY_BK + 1];
    __shared__ float Bs[TSY_BK][TSY_BN + 4];

    float C_reg[TSY_TM][TSY_TN];
    #pragma unroll
    for (int i = 0; i < TSY_TM; i++)
        #pragma unroll
        for (int j = 0; j < TSY_TN; j++)
            C_reg[i][j] = 0.f;

    const int a_row  = tid / 2;
    const int a_col4 = (tid % 2) * 4;
    const int b_row  = tid / (TSY_BN / 4);
    const int b_col4 = (tid % (TSY_BN / 4)) * 4;

    for (int k_tile = 0; k_tile < K; k_tile += TSY_BK) {
        // Load A tile (float4 with boundary guard).
        {
            const int gm = bm + a_row;
            const int gk = k_tile + a_col4;
            if (gm < M && gk + 3 < K) {
                float4 a4 = *reinterpret_cast<const float4*>(A + gm * K + gk);
                As[a_row][a_col4 + 0] = a4.x;
                As[a_row][a_col4 + 1] = a4.y;
                As[a_row][a_col4 + 2] = a4.z;
                As[a_row][a_col4 + 3] = a4.w;
            } else {
                #pragma unroll
                for (int dk = 0; dk < 4; dk++) {
                    int k = gk + dk, m = gm;
                    As[a_row][a_col4 + dk] = (m < M && k < K) ? A[m * K + k] : 0.f;
                }
            }
        }
        // Load B tile (float4 with boundary guard).
        {
            const int gk = k_tile + b_row;
            const int gn = bn + b_col4;
            if (gk < K && gn + 3 < N) {
                float4 b4 = *reinterpret_cast<const float4*>(B + gk * N + gn);
                Bs[b_row][b_col4 + 0] = b4.x;
                Bs[b_row][b_col4 + 1] = b4.y;
                Bs[b_row][b_col4 + 2] = b4.z;
                Bs[b_row][b_col4 + 3] = b4.w;
            } else {
                #pragma unroll
                for (int dn = 0; dn < 4; dn++) {
                    int k = gk, n = gn + dn;
                    Bs[b_row][b_col4 + dn] = (k < K && n < N) ? B[k * N + n] : 0.f;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TSY_BK; k++) {
            float a_reg[TSY_TM], b_reg[TSY_TN];
            #pragma unroll
            for (int tm = 0; tm < TSY_TM; tm++)
                a_reg[tm] = As[ty * TSY_TM + tm][k];
            #pragma unroll
            for (int tn = 0; tn < TSY_TN; tn++)
                b_reg[tn] = Bs[k][tx * TSY_TN + tn];
            #pragma unroll
            for (int tm = 0; tm < TSY_TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < TSY_TN; tn++)
                    C_reg[tm][tn] += a_reg[tm] * b_reg[tn];
        }
        __syncthreads();
    }

    // Write C_reg to global memory (two float4 stores per row).
    #pragma unroll
    for (int tm = 0; tm < TSY_TM; tm++) {
        const int gm = bm + ty * TSY_TM + tm;
        if (gm >= M) continue;
        {
            const int gn = bn + tx * TSY_TN;
            if (gn + 3 < N) {
                float4 c4 = {C_reg[tm][0], C_reg[tm][1],
                             C_reg[tm][2], C_reg[tm][3]};
                *reinterpret_cast<float4*>(C + gm * N + gn) = c4;
            } else {
                for (int i = 0; i < 4 && gn + i < N; i++)
                    C[gm * N + gn + i] = C_reg[tm][i];
            }
        }
        {
            const int gn = bn + tx * TSY_TN + 4;
            if (gn + 3 < N) {
                float4 c4 = {C_reg[tm][4], C_reg[tm][5],
                             C_reg[tm][6], C_reg[tm][7]};
                *reinterpret_cast<float4*>(C + gm * N + gn) = c4;
            } else {
                for (int i = 0; i < 4 && gn + i < N; i++)
                    C[gm * N + gn + i] = C_reg[tm][4 + i];
            }
        }
    }
}
#undef TSY_BM
#undef TSY_BN
#undef TSY_BK
#undef TSY_TM
#undef TSY_TN
```

- [ ] **Step 3: Replace `adapterMatMulCuda` body with variant dispatch**

Find the current `adapterMatMulCuda` (W8 version: cuBLAS only). Replace the entire function body with:

```cpp
void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c,
                       const std::string& variant) {
    assert(a.dims.size() == 2 && b.dims.size() == 2 && c.dims.size() == 2);
    const int M = static_cast<int>(a.dims[0]);
    const int K = static_cast<int>(a.dims[1]);
    const int N = static_cast<int>(b.dims[1]);
    assert(b.dims[0] == K);
    assert(c.dims[0] == M && c.dims[1] == N);

    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, a.data.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data.data(), bytesB, cudaMemcpyHostToDevice));

    if (variant == "naive") {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmulNaiveKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    } else if (variant == "tiled") {
        assert(M % 128 == 0 && N % 128 == 0 && K % 8 == 0 &&
               "tiled variant requires M%128==0 && N%128==0 && K%8==0");
        dim3 block(16, 16);
        dim3 grid(N / 128, M / 128);
        matmulTiledKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // "" or "cublas": W8 path.
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            getCublasHandle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, N,
            dA, K,
            &beta,
            dC, N));
    }

    c.data.assign(static_cast<size_t>(M) * N, 0.0f);
    CUDA_CHECK(cudaMemcpy(c.data.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}
```

(Remove the `(void)variant;` line from Task 8 if present.)

- [ ] **Step 4: Update the executor to read `attrs["variant"]` and pass it through**

Find `runFunctionCudaAdapter` in the same file. Locate the matmul branch inside the per-Stmt dispatch loop (it currently reads something like):

```cpp
if (s.primitive == "matmul") {
    if (s.operand_bufs.size() != 2) { ... }
    adapterMatMulCuda(r.buffers[s.operand_bufs[0]],
                      r.buffers[s.operand_bufs[1]], out);
}
```

Replace that branch with:

```cpp
if (s.primitive == "matmul") {
    if (s.operand_bufs.size() != 2) {
        diag.error(s.loc, "cuda-adapter matmul: expected 2 operands");
        r.ok = false; continue;
    }
    std::string variant;
    auto it = s.attrs.find("variant");
    if (it != s.attrs.end()) variant = it->second;
    adapterMatMulCuda(r.buffers[s.operand_bufs[0]],
                      r.buffers[s.operand_bufs[1]], out, variant);
}
```

- [ ] **Step 5: Build + run full suite — still 25/25, because no pass writes attrs yet**

```bash
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: `100% tests passed, 0 tests failed out of 25`.

---

## Task 10: Per-variant parity test + Commit 3

**Goal:** Extend `tests/adapter/test_adapter_cuda_cases.cpp` with a per-variant sub-test (matmul shape × each of naive / tiled / cublas), then commit.

**Files:**
- Modify: `tests/adapter/test_adapter_cuda_cases.cpp`

- [ ] **Step 1: Add `testMatmulVariant` helper**

Inside the existing anonymous namespace (just before the closing `}  // namespace`), add:

```cpp
void testMatmulVariant(const std::string& label,
                        int64_t M, int64_t K, int64_t N,
                        const std::string& variant) {
    auto A = makeTensor("A", {M, K}, linspace(M * K, 0.0f, 0.1f));
    auto B = makeTensor("B", {K, N}, linspace(K * N, 0.5f, 0.1f));
    auto C_cpu  = zeros("C", {M, N});
    auto C_cuda = zeros("C", {M, N});
    adapterMatMul(A, B, C_cpu);
    adapterMatMulCuda(A, B, C_cuda, variant);
    compareTensors(label, C_cpu, C_cuda);
}
```

- [ ] **Step 2: Add the per-variant assertions at the END of `main()`**

Just before the final `if (g_failures == 0) { ... }` block in `main()`, add:

```cpp
    // W9 per-variant parity: two shapes × three variants = 6 cases.
    // 64x64x64: fully aligned but below the scheduler's "tiled"
    // threshold — still exercises all three kernels manually.
    // 128x128x128: the scheduler's tiled sweet spot.
    for (auto shape : std::vector<std::tuple<int64_t, int64_t, int64_t>>{
             {64, 64, 64}, {128, 128, 128}}) {
        auto [M, K, N] = shape;
        std::string base = "matmul/" + std::to_string(M) + "x"
                          + std::to_string(K) + "x" + std::to_string(N);
        testMatmulVariant(base + "/naive",  M, K, N, "naive");
        testMatmulVariant(base + "/tiled",  M, K, N, "tiled");
        testMatmulVariant(base + "/cublas", M, K, N, "cublas");
    }
```

- [ ] **Step 3: Build + run — expect 6 new sub-case labels, all PASS**

```bash
cmake --build build -j --target test_adapter_cuda_cases
ctest --test-dir build -R adapter_cuda_cases --output-on-failure 2>&1 | tail -10
```

Expected: `adapter_cuda_cases: ALL PASS`, with internal output confirming 20 cases (14 previous + 6 new).

- [ ] **Step 4: Full suite**

```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: `100% tests passed, 0 tests failed out of 25`.

- [ ] **Step 5: Commit 3**

```bash
git add src/runtime/adapter_cuda.h src/runtime/adapter_cuda.cu \
        tests/adapter/test_adapter_cuda_cases.cpp
git commit -m "$(cat <<'EOF'
feat(runtime): add matmul naive + reg-tiled kernels with variant dispatch

adapterMatMulCuda gains an optional std::string variant parameter:
- "naive"  → matmulNaiveKernel (one thread per output element).
- "tiled"  → matmulTiledKernel (128x128 block, 8x8 register tile,
              BK=8 k-strip, float4 loads). Ported from
              mini-llm-engine/cuda-kernels/gemm/gemm.cu. Requires
              M%128==0 && N%128==0 && K%8==0.
- ""/cublas → cuBLAS sgemm (W8 path, unchanged).

Executor now reads Stmt.attrs["variant"] and passes it through; default
empty → cuBLAS so W8 behaviour is preserved pre-scheduler. Six new
adapter_cuda parity sub-cases (64^3 and 128^3 × 3 variants) all within
atol=1e-4, rtol=1e-3 of the CPU adapter.
EOF
)"
git log --oneline -1
```

---

## Task 11: Implement `ScheduleCudaPass` shape lookup + unit test

**Goal:** Fill in the real `runScheduleCuda` body with a shape-based variant picker. Add a unit test that constructs LIR modules programmatically and asserts the pass writes the expected variant.

**Files:**
- Modify: `src/passes/schedule_cuda.cpp`
- Create: `tests/passes/test_schedule_cuda_cases.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Replace `src/passes/schedule_cuda.cpp` with the real implementation**

Entire file:

```cpp
// W9 ScheduleCudaPass — shape-based kernel variant picker for matmul.
//
// The picker rules (initial thresholds, tuned after tsy-bench runs):
//   M*N < 1024                              → "naive"
//   aligned && M,N,K >= 128 && M*N <= 256^2 → "tiled"
//   otherwise                               → "cublas"
// where aligned == (M%128==0 && N%128==0 && K%8==0).
//
// Only matmul calls are scheduled. Other primitives (add/softmax/rmsnorm)
// have a single CUDA kernel and need no variant attr.

#include "pass_manager.h"

#include <string>

namespace tsy::passes {

namespace {

std::string pickMatmulVariant(int64_t M, int64_t K, int64_t N) {
    if (M * N < 1024) return "naive";
    const bool aligned =
        (M % 128 == 0) && (N % 128 == 0) && (K % 8 == 0);
    const bool large_enough =
        (M >= 128) && (N >= 128) && (K >= 128);
    if (aligned && large_enough && M * N <= 256 * 256) return "tiled";
    return "cublas";
}

}  // namespace

void runScheduleCuda(tsy::lir::Module& m, tsy::DiagnosticEngine& /*diag*/) {
    for (auto& f : m.funcs) {
        for (auto& s : f->body) {
            if (s.kind != tsy::lir::StmtKind::Call) continue;
            if (s.primitive != "matmul") continue;
            if (s.operand_bufs.size() != 2 || s.result_buf < 0) continue;
            const auto& A = f->buffers[s.operand_bufs[0]];
            const auto& B = f->buffers[s.operand_bufs[1]];
            if (A.dims.size() != 2 || B.dims.size() != 2) continue;
            const int64_t M = A.dims[0];
            const int64_t K = A.dims[1];
            const int64_t N = B.dims[1];
            s.attrs["variant"] = pickMatmulVariant(M, K, N);
        }
    }
}

}  // namespace tsy::passes
```

- [ ] **Step 2: Create `tests/passes/test_schedule_cuda_cases.cpp`**

Exact content:

```cpp
// W9 ScheduleCudaPass unit test — constructs LIR modules programmatically
// (no parser involved) and asserts runScheduleCuda writes the expected
// variant for each shape class.

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../../src/frontend/diagnostics.h"
#include "../../src/lir/ir.h"
#include "../../src/passes/pass_manager.h"

using tsy::DiagnosticEngine;
using tsy::lir::Buffer;
using tsy::lir::Function;
using tsy::lir::Module;
using tsy::lir::Stmt;
using tsy::lir::StmtKind;

namespace {

int g_failures = 0;

void fail(const std::string& label, const std::string& why) {
    std::cerr << "FAIL[" << label << "]: " << why << "\n";
    ++g_failures;
}

// Build a minimal LIR module with one function, three buffers (A[M,K],
// B[K,N], C[M,N]), and a single matmul Call. The module is owned by the
// caller and returned by value for simplicity.
std::unique_ptr<Module> makeMatmulModule(int64_t M, int64_t K, int64_t N) {
    auto mod = std::make_unique<Module>();
    auto f = std::make_unique<Function>();
    f->name = "matmul_case";
    f->return_type = "void";

    Buffer bufA; bufA.id = 0; bufA.name = "A"; bufA.dims = {M, K};
    Buffer bufB; bufB.id = 1; bufB.name = "B"; bufB.dims = {K, N};
    Buffer bufC; bufC.id = 2; bufC.name = "C"; bufC.dims = {M, N};
    f->buffers = {bufA, bufB, bufC};
    f->params = {0, 1};

    Stmt s;
    s.kind = StmtKind::Call;
    s.primitive = "matmul";
    s.operand_bufs = {0, 1};
    s.result_buf = 2;
    f->body.push_back(std::move(s));

    Stmt ret;
    ret.kind = StmtKind::Return;
    f->body.push_back(std::move(ret));

    mod->funcs.push_back(std::move(f));
    return mod;
}

void checkVariant(const std::string& label,
                   int64_t M, int64_t K, int64_t N,
                   const std::string& expected) {
    auto mod = makeMatmulModule(M, K, N);
    DiagnosticEngine diag;
    tsy::passes::runScheduleCuda(*mod, diag);
    if (diag.hasErrors()) {
        fail(label, "diagnostic engine reported errors");
        return;
    }
    const auto& s = mod->funcs[0]->body[0];
    auto it = s.attrs.find("variant");
    if (it == s.attrs.end()) {
        fail(label, "no variant attr written");
        return;
    }
    if (it->second != expected) {
        fail(label, "got variant='" + it->second +
                     "', expected '" + expected + "'");
    }
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    // Tiny: M*N < 1024 → naive
    checkVariant("tiny-4x4x4",   4,   4,  4, "naive");
    checkVariant("row-1x32x8",   1,  32,  8, "naive");

    // Sweet spot: aligned + large_enough + M*N <= 256^2 → tiled
    checkVariant("sweet-128x128x128",  128, 128, 128, "tiled");
    checkVariant("sweet-256x128x256",  256, 128, 256, "tiled");

    // Too big: aligned but M*N > 256^2 → cublas
    checkVariant("large-512x512x512", 512, 512, 512, "cublas");

    // Odd shape: alignment fails → cublas
    checkVariant("odd-7x13x11", 7, 13, 11, "cublas");

    // Aligned but K<128 (not large_enough) → cublas
    checkVariant("small-K-128x64x128", 128, 64, 128, "cublas");

    if (g_failures == 0) {
        std::cout << "pass_schedule_cuda_cases: ALL PASS\n";
        return 0;
    }
    std::cerr << "pass_schedule_cuda_cases: " << g_failures
              << " failure(s)\n";
    return 1;
}
```

- [ ] **Step 3: Register the test in `tests/CMakeLists.txt`**

Find the existing W5 passes block (around `add_executable(test_pass_cases passes/test_pass_cases.cpp)`). After that block, add:

```cmake
# W9: ScheduleCudaPass unit test.
add_executable(test_schedule_cuda_cases passes/test_schedule_cuda_cases.cpp)
target_link_libraries(test_schedule_cuda_cases PRIVATE tsy_passes)

add_test(NAME pass_schedule_cuda_cases
    COMMAND test_schedule_cuda_cases
)
```

(Note: this test does NOT need `TARGET tsy_runtime_cuda` — it only exercises the scheduler pass, which is CPU code that writes strings. It works on non-CUDA boxes too.)

- [ ] **Step 4: Build + run**

```bash
cmake -S . -B build
cmake --build build -j --target test_schedule_cuda_cases 2>&1 | tail -5
ctest --test-dir build -R pass_schedule_cuda_cases --output-on-failure
```

Expected:
```
pass_schedule_cuda_cases: ALL PASS
...
100% tests passed, 0 tests failed out of 1
```

- [ ] **Step 5: Full suite check**

```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: `100% tests passed, 0 tests failed out of 26` (25 + new pass_schedule_cuda_cases).

---

## Task 12: Example .tsy files + CLI regex ctests

**Goal:** Add two new `.tsy` examples that together exercise the medium (tiled) and large (cublas) shape classes, and register two CLI regex tests that confirm schedule-cuda writes the attr in emit-lir output.

**Files:**
- Create: `examples/run_matmul_medium.tsy`
- Create: `examples/run_matmul_large.tsy`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Create `examples/run_matmul_medium.tsy`**

```
// W9 scheduler fixture: 128x128x128 matmul. The aligned + large_enough
// + M*N <= 256^2 condition picks variant="tiled" in the schedule-cuda
// pass. Used by cli_emit_lir_schedule_shows_variant ctest.
const int M = 128;
const int K = 128;
const int N = 128;

void matmul_layer(tensor<f32>[M, K] A, tensor<f32>[K, N] B) {
    tensor<f32>[M, N] C = @matmul(A, B);
    return;
}

int main() { return 0; }
```

- [ ] **Step 2: Create `examples/run_matmul_large.tsy`**

```
// W9 scheduler fixture: 512x512x512 matmul. Aligned + large_enough
// pass but M*N=262144 > 256^2=65536 so scheduler picks variant="cublas".
const int M = 512;
const int K = 512;
const int N = 512;

void matmul_layer(tensor<f32>[M, K] A, tensor<f32>[K, N] B) {
    tensor<f32>[M, N] C = @matmul(A, B);
    return;
}

int main() { return 0; }
```

- [ ] **Step 3: Register two CLI regex ctests in `tests/CMakeLists.txt`**

Append to the end of `tests/CMakeLists.txt` (OUTSIDE any `if(TARGET tsy_runtime_cuda)` block — these tests don't need CUDA runtime, just the scheduler pass):

```cmake
# W9: --opt=O1 with schedule-cuda enabled must write `variant="tiled"`
# for the 128^3 medium fixture.
add_test(NAME cli_emit_lir_schedule_shows_variant
    COMMAND tsc emit-lir --opt=O1
            ${CMAKE_SOURCE_DIR}/examples/run_matmul_medium.tsy
)
set_tests_properties(cli_emit_lir_schedule_shows_variant PROPERTIES
    PASS_REGULAR_EXPRESSION "variant=\"tiled\""
)

# W9: --disable-pass=schedule-cuda must suppress the variant attr.
add_test(NAME cli_emit_lir_disable_schedule_no_variant
    COMMAND tsc emit-lir --opt=O1 --disable-pass=schedule-cuda
            ${CMAKE_SOURCE_DIR}/examples/run_matmul_medium.tsy
)
set_tests_properties(cli_emit_lir_disable_schedule_no_variant PROPERTIES
    FAIL_REGULAR_EXPRESSION "variant="
)
```

- [ ] **Step 4: Configure + build + run**

```bash
cmake -S . -B build
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -10
```

Expected: `100% tests passed, 0 tests failed out of 28` (26 + 2 new CLI tests).

- [ ] **Step 5: Manual sanity**

```bash
./build/tsc emit-lir --opt=O1 examples/run_matmul_medium.tsy | grep variant
./build/tsc emit-lir --opt=O1 --disable-pass=schedule-cuda examples/run_matmul_medium.tsy | grep variant || echo "no variant (expected)"
./build/tsc emit-lir --opt=O1 examples/run_matmul_large.tsy | grep variant
```

Expected:
```
    %C = call matmul %A, %B {variant="tiled"}
no variant (expected)
    %C = call matmul %A, %B {variant="cublas"}
```

---

## Task 13: Commit 4

- [ ] **Step 1: Commit**

```bash
git add src/passes/schedule_cuda.cpp \
        tests/passes/test_schedule_cuda_cases.cpp tests/CMakeLists.txt \
        examples/run_matmul_medium.tsy examples/run_matmul_large.tsy
git commit -m "$(cat <<'EOF'
feat(passes): implement ScheduleCudaPass shape lookup

Walks each LIR function's body, picks a matmul variant via
pickMatmulVariant(M,K,N), and writes it into Stmt.attrs["variant"].
Lookup rules (initial thresholds):
- M*N < 1024                              → naive
- aligned && M,N,K >= 128 && M*N <= 256^2 → tiled
- otherwise                               → cublas

Seven-case programmatic unit test (pass_schedule_cuda_cases) asserts
each rule class. Two CLI regex tests (cli_emit_lir_schedule_shows_variant
and cli_emit_lir_disable_schedule_no_variant) confirm end-to-end
integration and --disable-pass=schedule-cuda behaviour. Two new .tsy
fixtures: run_matmul_medium.tsy (128^3 → tiled) and run_matmul_large.tsy
(512^3 → cublas).

Total ctest count: 28/28.
EOF
)"
git log --oneline -1
```

---

## Task 14: `src/tools/tsy-bench.cu` — CUDA-event bench binary

**Goal:** Standalone C++/CUDA binary that iterates over shapes × variants, times each with CUDA events (3 warmup + 5 measured, take median), emits CSV to stdout.

**Files:**
- Create: `src/tools/tsy-bench.cu`

- [ ] **Step 1: Create `src/tools/tsy-bench.cu`**

```cpp
// tsy-bench — CUDA-event precision matmul benchmark.
//
// Usage:
//   tsy-bench                    # default 5 shape × 3 variant = 15 rows
//   tsy-bench --smoke            # single shape × all variants (ctest use)
//   tsy-bench --shapes 1024x1024x1024  # single shape
//   tsy-bench --variants naive,tiled   # variant filter
//
// Output: CSV on stdout with header
//   primitive,M,K,N,variant,ms_median,gflops
//
// Timing: 3 warmup + 5 measured runs via cudaEvent, take the median.

#include "../runtime/adapter_cuda.h"
#include "../lir/interpreter.h"  // NamedTensor / fillDeterministic

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Shape { int M, K, N; };

std::vector<Shape> parseShapes(const std::string& csv_spec) {
    std::vector<Shape> out;
    std::stringstream ss(csv_spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        Shape s;
        if (std::sscanf(item.c_str(), "%dx%dx%d", &s.M, &s.K, &s.N) == 3) {
            out.push_back(s);
        }
    }
    return out;
}

std::vector<std::string> parseVariants(const std::string& csv_spec) {
    std::vector<std::string> out;
    std::stringstream ss(csv_spec);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

// Host-side NamedTensor factory mirroring adapter tests.
tsy::lir::NamedTensor makeBuf(const std::string& name,
                               const std::vector<int64_t>& dims) {
    tsy::lir::NamedTensor t;
    t.name = name;
    t.dims = dims;
    int64_t n = 1; for (auto d : dims) n *= d;
    t.data.assign(n, 0.0f);
    return t;
}

float medianMs(std::vector<float> ms) {
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

// Run one (shape, variant) benchmark, return median ms.
float benchOne(int M, int K, int N, const std::string& variant) {
    auto A = makeBuf("A", {M, K});
    auto B = makeBuf("B", {K, N});
    auto C = makeBuf("C", {M, N});
    tsy::lir::fillDeterministic(A, 0);
    tsy::lir::fillDeterministic(B, 1);

    // Warmup (3 runs, results discarded).
    for (int i = 0; i < 3; i++) {
        tsy::runtime::adapterMatMulCuda(A, B, C, variant);
    }
    cudaDeviceSynchronize();

    // Measured (5 runs).
    std::vector<float> times;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(t0);
        tsy::runtime::adapterMatMulCuda(A, B, C, variant);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        times.push_back(ms);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return medianMs(times);
}

float gflops(int M, int K, int N, float ms) {
    if (ms <= 0.0f) return 0.0f;
    const double ops = 2.0 * static_cast<double>(M) *
                              static_cast<double>(K) *
                              static_cast<double>(N);
    return static_cast<float>(ops / (static_cast<double>(ms) * 1e6));
}

int usage(const char* progname) {
    std::cerr << "usage: " << progname << " [--smoke] "
              << "[--shapes MxKxN[,...]] [--variants v1[,v2,...]]\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    bool smoke = false;
    std::string shapes_arg;
    std::string variants_arg;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--smoke") { smoke = true; }
        else if (a.rfind("--shapes=", 0) == 0) shapes_arg = a.substr(9);
        else if (a == "--shapes" && i + 1 < argc) shapes_arg = argv[++i];
        else if (a.rfind("--variants=", 0) == 0) variants_arg = a.substr(11);
        else if (a == "--variants" && i + 1 < argc) variants_arg = argv[++i];
        else if (a == "-h" || a == "--help") { return usage(argv[0]); }
        else { return usage(argv[0]); }
    }

    // Defaults
    std::vector<Shape> shapes;
    if (!shapes_arg.empty()) shapes = parseShapes(shapes_arg);
    else if (smoke)          shapes = { {256, 256, 256} };
    else                     shapes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {128, 16, 8},
        {7, 13, 11},
    };

    std::vector<std::string> variants;
    if (!variants_arg.empty()) variants = parseVariants(variants_arg);
    else                       variants = { "naive", "tiled", "cublas" };

    std::cout << "primitive,M,K,N,variant,ms_median,gflops\n";
    for (const auto& s : shapes) {
        for (const auto& v : variants) {
            // Tiled kernel requires aligned dims; skip cleanly if not.
            if (v == "tiled") {
                if (s.M % 128 != 0 || s.N % 128 != 0 || s.K % 8 != 0) {
                    continue;
                }
            }
            float ms = benchOne(s.M, s.K, s.N, v);
            float gf = gflops(s.M, s.K, s.N, ms);
            std::cout << "matmul," << s.M << "," << s.K << "," << s.N << ","
                      << v << "," << ms << "," << gf << "\n";
        }
    }
    return 0;
}
```

- [ ] **Step 2: Add `tsy-bench` target in `CMakeLists.txt`**

Find the `tsy_runtime_cuda` block (already gated inside `if(CMAKE_CUDA_COMPILER)`). Inside the same `if(CMAKE_CUDA_COMPILER) ... endif()`, after `tsy_add_cuda_example(matmul_cuda_demo ...)`, add:

```cmake
    add_executable(tsy-bench src/tools/tsy-bench.cu)
    set_source_files_properties(src/tools/tsy-bench.cu PROPERTIES LANGUAGE CUDA)
    target_link_libraries(tsy-bench PRIVATE tsy_runtime_cuda)
    set_target_properties(tsy-bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CUDA_ARCHITECTURES 86)
```

(`tsy_runtime_cuda` already exposes its include dirs PUBLIC, so `adapter_cuda.h` and `interpreter.h` will resolve.)

- [ ] **Step 3: Configure + build**

```bash
cmake -S . -B build
cmake --build build -j --target tsy-bench 2>&1 | tail -10
```

Expected: clean build, `Built target tsy-bench`.

- [ ] **Step 4: Manual smoke**

```bash
./build/tsy-bench --smoke
```

Expected output (ms/gflops values will vary):
```
primitive,M,K,N,variant,ms_median,gflops
matmul,256,256,256,naive,0.123,272.6
matmul,256,256,256,tiled,0.087,385.2
matmul,256,256,256,cublas,0.056,598.4
```
(Three rows if tiled is skipped due to alignment.)

- [ ] **Step 5: Full-sweep sanity check**

```bash
./build/tsy-bench | head -20
```

Expected: CSV header + up to 15 rows (some tiled rows skipped when shape not aligned). No abort / no CUDA errors.

---

## Task 15: Commit 5

- [ ] **Step 1: Commit**

```bash
git add src/tools/tsy-bench.cu CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(tools): tsy-bench C++ binary

Standalone CUDA-event benchmark. Iterates over shape x variant, runs
3 warmup + 5 measured calls of adapterMatMulCuda per combo, emits
a single CSV row per combo (primitive,M,K,N,variant,ms_median,gflops).

Default sweep: {256, 512, 1024}^3 + tall + odd × {naive, tiled, cublas};
tiled skipped when shape isn't 128/8-aligned. --smoke runs the 256^3
shape only. --shapes=MxKxN[,...] and --variants=v1[,v2,...] override
the defaults.

Built only when TSY_HAVE_RUNTIME_CUDA; target output in build/.
EOF
)"
git log --oneline -1
```

---

## Task 16: `benchmarks/run_shapes.py` + ctest

**Goal:** Python driver that shells out to tsy-bench, prints a human-readable table, and optionally asserts the scheduler-critical invariant (tiled faster than naive at 1024^3). Only `--smoke` goes into ctest (stable).

**Files:**
- Create: `benchmarks/run_shapes.py`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Create `benchmarks/run_shapes.py`**

```python
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
    print(f"1024^3: tiled/naive speedup = {speedup:.2f}x")
    if speedup < 2.0:
        print(f"FAIL: expected >= 2x, got {speedup:.2f}x", file=sys.stderr)
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
```

- [ ] **Step 2: Register `cli_bench_smoke` ctest in `tests/CMakeLists.txt`**

Append to the end of `tests/CMakeLists.txt`, inside the existing outer `if(TARGET tsy_runtime_cuda)` block (needs tsy-bench target, which needs CUDA). OUTSIDE the inner `tsy_runtime_cpu` gate:

```cmake
    # W9: Python driver smoke — confirms tsy-bench runs + CSV is parseable.
    add_test(NAME cli_bench_smoke
        COMMAND python3 ${CMAKE_SOURCE_DIR}/benchmarks/run_shapes.py --smoke
    )
    set_tests_properties(cli_bench_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "matmul.*256.*256.*256"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
    )
```

(`WORKING_DIRECTORY` is the project root so the Python script's `Path("build/tsy-bench")` resolves correctly.)

- [ ] **Step 3: Reconfigure + build + run**

```bash
cmake -S . -B build
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -10
```

Expected: `100% tests passed, 0 tests failed out of 29`.

- [ ] **Step 4: Manual full-sweep check** (informational, not ctest)

```bash
python3 benchmarks/run_shapes.py
```

Expected: table with 10-15 rows. Visually inspect that at 1024×1024×1024 the `tiled` row's `ms_median` is at least ~2× smaller than `naive`'s. If not, we'll tune thresholds after commit (that work lives outside this task).

---

## Task 17: Commit 6

- [ ] **Step 1: Commit**

```bash
git add benchmarks/run_shapes.py tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(tools): benchmarks/run_shapes.py driver + cli_bench_smoke ctest

Pure-stdlib Python driver over build/tsy-bench. --smoke runs one shape
and exits 0 if the subprocess emits valid CSV. --check-scheduler runs
the full sweep and asserts 1024^3 tiled is at least 2x faster than
naive (kept out of ctest to avoid GPU-timing flakiness; used by humans
for regression-checking).

Total ctest count: 29/29 on CUDA boxes, 21/21 on non-CUDA.
EOF
)"
git log --oneline -1
```

---

## Task 18: Codegen emits variant arg in emit-cu

**Goal:** When LIR `Stmt.attrs["variant"]` is set, the generated `.cu` should pass that variant as the last arg to `adapterMatMulCuda(...)`. Non-matmul primitives and empty attrs → unchanged output.

**Files:**
- Modify: `src/codegen/cuda.cpp`

- [ ] **Step 1: Modify `src/codegen/cuda.cpp` — body-call emission**

Find the `emitCudaModule` function's body-call loop (roughly the section with `const char* sym = adapterSymbolFor(...)`). Replace the loop body with:

```cpp
    for (const auto& s : f->body) {
        if (s.kind == StmtKind::Return) break;
        if (s.kind != StmtKind::Call) continue;
        const char* sym = adapterSymbolFor(s.primitive);
        if (!sym) {
            os << "    // skipped unsupported primitive: " << s.primitive << "\n";
            continue;
        }
        os << "    tsy::runtime::" << sym << "(";
        for (size_t i = 0; i < s.operand_bufs.size(); ++i) {
            if (i) os << ", ";
            os << identFor(f->buffers[s.operand_bufs[i]]);
        }
        os << ", " << identFor(f->buffers[s.result_buf]);
        // W9: pass variant as trailing arg to adapterMatMulCuda when set.
        // Only matmul takes a variant; ignore attrs on other primitives.
        if (s.primitive == "matmul") {
            auto it = s.attrs.find("variant");
            if (it != s.attrs.end()) {
                os << ", \"" << it->second << "\"";
            }
        }
        os << ");\n";
    }
```

- [ ] **Step 2: Build + manual smoke (verify generated code)**

```bash
cmake --build build -j 2>&1 | tail -5

# W8 default (no --opt=O1 → no scheduler → no variant arg) — unchanged
./build/tsc emit-cu examples/matmul_tiny_cuda.tsy | grep adapterMatMulCuda

# W9 O1 (scheduler picks naive for 2x2 since M*N=4 < 1024) — variant arg present
./build/tsc emit-cu --opt=O1 examples/matmul_tiny_cuda.tsy | grep adapterMatMulCuda

# W9 O1 medium — variant="tiled"
./build/tsc emit-cu --opt=O1 examples/run_matmul_medium.tsy | grep adapterMatMulCuda
```

Expected outputs:
```
    tsy::runtime::adapterMatMulCuda(buf_A, buf_B, buf_C);
    tsy::runtime::adapterMatMulCuda(buf_A, buf_B, buf_C, "naive");
    tsy::runtime::adapterMatMulCuda(buf_A, buf_B, buf_C, "tiled");
```

- [ ] **Step 3: Run full suite — W8 codegen_cuda_matmul_binary_runs must stay green**

```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -10
```

Expected: `100% tests passed, 0 tests failed out of 29`.

**Important check:** `codegen_cuda_matmul_binary_runs` (W8) runs `./build/out/matmul_cuda_demo` which is built via `tsy_add_cuda_example(matmul_cuda_demo examples/matmul_tiny_cuda.tsy)`. That CMake custom command invokes `tsc emit-cu` WITHOUT `--opt=O1` — so no scheduler, no variant suffix, byte-identical to W8's output, same `0.0700 0.0800 / 0.3100 0.3600` result. Regression-safe.

---

## Task 19: Commit 7

- [ ] **Step 1: Commit**

```bash
git add src/codegen/cuda.cpp
git commit -m "$(cat <<'EOF'
feat(codegen): emit variant arg in emit-cu

When a matmul call carries attrs["variant"] (written by ScheduleCudaPass
under --opt=O1), the generated .cu now emits
    tsy::runtime::adapterMatMulCuda(..., "naive"|"tiled"|"cublas")
so the compiled binary honours the compile-time scheduler decision.

Non-matmul primitives and attr-less matmul calls emit byte-identical
code to W8, so the codegen_cuda_matmul_binary_runs regression test
(which compiles under default --opt=O0) is unchanged.
EOF
)"
git log --oneline -1
```

---

## Task 20: Final verification + Commit 8 (docs)

**Goal:** Clean rebuild, confirm 29/29, manually verify the acceptance matrix from spec §8, commit the spec.

**Files:**
- None modified (spec is already in-tree from earlier).

- [ ] **Step 1: Clean rebuild from scratch**

```bash
rm -rf build
cmake -S . -B build 2>&1 | tail -10
cmake --build build -j 2>&1 | tail -10
```

Expected: clean configure + build, including `Built target tsy-bench` and all targets from W8.

- [ ] **Step 2: Full ctest suite**

```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: `100% tests passed, 0 tests failed out of 29`.

- [ ] **Step 3: Acceptance matrix from spec §8**

```bash
# (1) Build + ctest already done above.
# (2) Already green.

# (3) emit-lir shows variant="tiled" for medium
./build/tsc emit-lir --opt=O1 examples/run_matmul_medium.tsy | grep -q 'variant="tiled"' && echo OK3 || echo FAIL3

# (4) --disable-pass=schedule-cuda removes variant
./build/tsc emit-lir --opt=O1 --disable-pass=schedule-cuda examples/run_matmul_medium.tsy | grep -q 'variant=' && echo FAIL4 || echo OK4

# (5) tsy-bench --smoke succeeds, emits >=1 CSV row
./build/tsy-bench --smoke | tail -n +2 | wc -l | xargs -I{} test {} -ge 1 && echo OK5 || echo FAIL5

# (6) Python driver smoke succeeds
python3 benchmarks/run_shapes.py --smoke > /dev/null && echo OK6 || echo FAIL6

# (7) Full-sweep: tiled at 1024^3 at least 2x faster than naive
python3 benchmarks/run_shapes.py --check-scheduler > /tmp/w9_scheduler_check.out 2>&1
grep -q "speedup" /tmp/w9_scheduler_check.out && cat /tmp/w9_scheduler_check.out | grep speedup

# (8) W7 MLP unchanged
./build/out/mlp | tail -3 | grep -q "0.4502" && echo OK8 || echo FAIL8
```

Expected: `OK3 OK4 OK5 OK6`, a `speedup = X.XXx` line showing ≥ 2, and `OK8`.

- [ ] **Step 4: Commit the spec + plan under a docs commit**

The spec and plan files already exist from the brainstorming / plan-writing phases. Commit them now as the formal W9 documentation record.

```bash
git add docs/superpowers/specs/2026-04-18-tensor-sysy-w9-scheduler-design.md \
        docs/superpowers/plans/2026-04-18-tensor-sysy-w9-scheduler.md
git commit -m "$(cat <<'EOF'
docs: W9 scheduler spec + implementation plan

Design doc: CUDA matmul scheduler with 3 variants (naive / reg-tiled /
cuBLAS), shape-based lookup table via new ScheduleCudaPass, LIR
Stmt.attrs field for carrying compile-time metadata, placeholder
LayoutLoweringPass for W10 to fill in. Benchmark harness: C++ tsy-bench
with CUDA events + Python driver with stdlib only.

Plan doc: 8 commits, 20 tasks. Spec coverage table at the end of the
plan maps each spec section to its implementing task.
EOF
)"
git log --oneline -8
```

- [ ] **Step 5: Echo acceptance**

```bash
echo "W9 ACCEPTANCE: GREEN"
git log --oneline | head -12
```

Expected: `W9 ACCEPTANCE: GREEN` and a log showing 8 new commits (4 W8 + 8 W9 + 1 initial import = 13 total, though exact count depends on commit granularity).

---

## Rollback / recovery

If any task fails irrecoverably:
```bash
git reset --hard HEAD~1   # undo last commit
git stash                 # preserve uncommitted work
```

Non-CUDA machines: `tsy_runtime_cuda`, `tsy-bench`, `matmul_cuda_demo`, and all `if(TARGET tsy_runtime_cuda)` tests skip cleanly. W0-W7 + the new `pass_schedule_cuda_cases` (CPU-only) + the two CLI regex tests (CPU-only — they exercise `tsc emit-lir` which doesn't need CUDA) stay green at 24/24.

---

## Spec coverage

| Spec section | Implementing task |
|---|---|
| §4.1 LIR Stmt.attrs + printer | Task 1 |
| §4.2 ScheduleCudaPass + pickMatmulVariant | Tasks 5 (stub), 11 (real body) |
| §4.2 Threshold table | Task 11 Step 1 |
| §4.3 LayoutLoweringPass stub | Task 4 |
| §4.3 O1 pipeline registration | Task 3 Step 2 (`buildPipelineO1`) |
| §4.4 adapterMatMulCuda variant parameter | Task 8 |
| §4.4 matmulNaiveKernel / matmulTiledKernel | Task 9 Steps 1+2 |
| §4.4 Executor reads attrs["variant"] | Task 9 Step 4 |
| §4.5 Codegen emits variant string | Task 18 |
| §4.6 tsy-bench CLI + CSV schema | Task 14 |
| §4.7 run_shapes.py driver | Task 16 |
| §5 matmul_tiny_cuda / medium / large fixtures | Task 12 (medium, large); W8 already has tiny |
| §6.1 pass_schedule_cuda_cases (5 shapes) | Task 11 Step 2 (7 cases — adds K<128 and 256x128x256 on top of spec's 5) |
| §6.2 cli_emit_lir_schedule_shows_variant + disable variant | Task 12 Step 3 |
| §6.3 adapter_cuda_variants_parity | Task 10 |
| §6.4 cli_bench_smoke | Task 16 Step 2 |
| §7 Risk: reg-tiled kernel port bug | Task 9 Step 2 (verbatim port from gemm.cu, asserts on alignment in adapter) |
| §7 Risk: PassManager schema change | Task 3 (HIR path unchanged, LIR vector is additive) |
| §7 Risk: printer golden regressions | Task 1 Step 2 (empty-attrs guard keeps byte-identical output) |
| §7 Risk: GPU timing flakiness | Task 14 (3 warmup + 5 measured + median), Task 16 Step 2 (only --smoke in ctest) |
| §8 Acceptance (8 checks) | Task 20 Step 3 |
| §9 Commits 1-8 | Tasks 2 / 7 / 10 / 13 / 15 / 17 / 19 / 20 |

All spec requirements map to at least one task. No orphan requirements.
