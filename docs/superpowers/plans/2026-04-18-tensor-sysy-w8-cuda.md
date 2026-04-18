# W8 — CUDA Adapter + 单算子 Codegen · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring tensor-sysy's CUDA path online end-to-end: `run-lir --backend=cuda-adapter` and `tsc emit-cu` both work, with CPU/CUDA cross-check tests and a compiled standalone CUDA binary for a single-operator example.

**Architecture:** Self-written FP32 adapter layer calling cuBLAS sgemm for matmul and three hand-written CUDA kernels (add / softmax / rmsnorm). Mirrors the existing `src/runtime/adapter_cpu.{h,cpp}` and `src/codegen/cpp.{h,cpp}` patterns exactly; gated at CMake-configure time via `check_language(CUDA)` so non-CUDA machines still see 21/21 green on W0–W7.

**Tech Stack:** C++17, CMake 3.18+, CUDA 12 toolkit, cuBLAS, flex/bison (unchanged), nvcc sm_86 (RTX 3080 Ampere).

**Spec:** `docs/superpowers/specs/2026-04-18-tensor-sysy-w8-cuda-design.md`

**Expected final state:** `ctest --test-dir build --output-on-failure` → **25 / 25 green** (21 existing + 4 new) on WSL + RTX 3080; still 21/21 green on any non-CUDA machine.

---

## File Structure

### NEW files
- `src/runtime/adapter_cuda.h` — public signatures (parallel to `adapter_cpu.h`)
- `src/runtime/adapter_cuda.cu` — kernels + cuBLAS wrapper + executor
- `src/codegen/cuda.h` — `emitCudaModule` signature
- `src/codegen/cuda.cpp` — `.cu` source generator (parallel to `codegen/cpp.cpp`)
- `tests/adapter/test_adapter_cuda_cases.cpp` — CPU-vs-CUDA adapter parity harness
- `examples/matmul_tiny_cuda.tsy` — one-operator fixture for CUDA codegen test

### MODIFIED files
- `CMakeLists.txt` — CUDA detection, `tsy_runtime_cuda` target, WSL libcuda RPATH fix, `tsy_add_cuda_example()` helper, conditional link to `tsc`
- `src/tools/tsc.cpp` — `--backend=cuda-adapter` branch, `emit-cu` subcommand, `--help` text update
- `tests/CMakeLists.txt` — four new ctest entries gated on `TARGET tsy_runtime_cuda`

### UNCHANGED (referenced for parallelism)
- `src/runtime/adapter_cpu.{h,cpp}` — structural template
- `src/codegen/cpp.{h,cpp}` — structural template
- `tests/adapter/test_adapter_cpu_cases.cpp` — test harness template

---

## Commit Plan

Four logical milestones, one commit each (per `~/.claude/CLAUDE.md` "每个逻辑里程碑一个 commit"):

1. `feat(runtime): add CUDA FP32 adapter + tsy_runtime_cuda` (Tasks 1–4)
2. `feat(runtime): implement add/softmax/rmsnorm CUDA kernels + parity matrix` (Tasks 5–9)
3. `feat(cli): run-lir --backend=cuda-adapter` (Tasks 10–12)
4. `feat(codegen): add emit-cu + tsy_add_cuda_example` (Tasks 13–17)

**Note on git state:** the WSL export tarball has no `.git`. Before the first commit, run `git init && git add -A && git commit -m "snapshot: W0–W7 on WSL"` to create a baseline. Or if you prefer to reattach to origin per `SETUP_WSL.md` Method B, do that first (outside this plan).

---

## Task 1: CMake CUDA detection + empty `tsy_runtime_cuda` target

**Goal:** Configure CUDA at build time with graceful fallback. After this task, `cmake -S . -B build` succeeds on machines both with and without CUDA; on CUDA-capable machines `tsy_runtime_cuda` builds as an empty static lib.

**Files:**
- Create: `src/runtime/adapter_cuda.cu` (stub — 3 lines)
- Modify: `CMakeLists.txt` (append CUDA block before `# --- tsc executable ---` section around line 163)

- [ ] **Step 1: Create stub `src/runtime/adapter_cuda.cu`**

```cpp
// Stub — real kernels added in Task 3 onwards.
// Required to give tsy_runtime_cuda at least one source file so the target
// links correctly before any adapter entry points exist.
namespace tsy::runtime { void adapterCudaStub() {} }
```

- [ ] **Step 2: Modify `CMakeLists.txt` — append CUDA detection block**

Insert this **before** the `# --- tsc executable ---` comment (after the `set_source_files_properties` block that relaxes warnings on flex/bison output, i.e. after line 154):

```cmake
# --- Runtime adapter (CUDA, W8) ------------------------------------------
#
# Self-written FP32 kernels + cuBLAS sgemm for matmul. Independent of
# mini-llm-engine (their ops_cuda.cu is FP16-only). Gated at configure
# time so machines without CUDA still see 21/21 on W0-W7 tests.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

    # WSL2 libcuda stub fix: apt's libcuda.so shadows the real driver in
    # /usr/lib/wsl/lib/. DT_RPATH (not RUNPATH) propagates to dlopen inside
    # cudart. Borrowed verbatim from mini-llm-engine/CMakeLists.txt.
    if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
        set(CMAKE_BUILD_RPATH "/usr/lib/wsl/lib")
        add_link_options("LINKER:--disable-new-dtags")
        set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -L/usr/lib/wsl/lib -lcuda -Wl,--as-needed")
    endif()

    find_package(CUDAToolkit REQUIRED)

    add_library(tsy_runtime_cuda STATIC src/runtime/adapter_cuda.cu)
    target_include_directories(tsy_runtime_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime)
    target_link_libraries(tsy_runtime_cuda PUBLIC
        tsy_lir CUDA::cudart CUDA::cublas)

    set(TSY_HAVE_RUNTIME_CUDA 1)
    message(STATUS "tsy_runtime_cuda: enabled (arch sm_86)")
else()
    set(TSY_HAVE_RUNTIME_CUDA 0)
    message(STATUS "tsy_runtime_cuda: CUDA not detected; CUDA tests skipped")
endif()
```

Also extend the `tsc` executable link block (currently lines 163–172) so it conditionally pulls in the CUDA adapter. Replace those lines with:

```cmake
# --- tsc executable -------------------------------------------------------

add_executable(tsc src/tools/tsc.cpp)
target_link_libraries(tsc PRIVATE tsy_passes tsy_codegen)
if(TSY_HAVE_RUNTIME_CPU)
    target_link_libraries(tsc PRIVATE tsy_runtime_cpu)  # brings tsy_lir
    target_compile_definitions(tsc PRIVATE TSY_HAVE_RUNTIME_CPU=1)
else()
    target_link_libraries(tsc PRIVATE tsy_lir)
endif()
if(TSY_HAVE_RUNTIME_CUDA)
    target_link_libraries(tsc PRIVATE tsy_runtime_cuda)
    target_compile_definitions(tsc PRIVATE TSY_HAVE_RUNTIME_CUDA=1)
endif()
```

- [ ] **Step 3: Run configure + build; verify `tsy_runtime_cuda` target exists and builds**

Run:
```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
rm -rf build
cmake -S . -B build
cmake --build build -j --target tsy_runtime_cuda
```

Expected output includes:
```
-- tsy_runtime_cuda: enabled (arch sm_86)
...
[100%] Built target tsy_runtime_cuda
```

- [ ] **Step 4: Rebuild everything and run the full ctest to ensure W0–W7 regression is clean**

Run:
```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 21`.

- [ ] **Step 5: Do NOT commit yet** — Task 1 produces only scaffolding. Commit after Task 4 together.

---

## Task 2: `src/runtime/adapter_cuda.h` — public signatures

**Goal:** Declare the full CUDA adapter API so test code can `#include "adapter_cuda.h"` even before implementations exist. Parallel to `src/runtime/adapter_cpu.h`.

**Files:**
- Create: `src/runtime/adapter_cuda.h`

- [ ] **Step 1: Write the header file**

```cpp
#pragma once

#include <string>

#include "../frontend/diagnostics.h"
#include "../lir/interpreter.h"   // NamedTensor / RunResult
#include "../lir/ir.h"

// CUDA runtime adapter — parallel to adapter_cpu.h. Self-contained FP32
// implementations using cuBLAS for GEMM and hand-written CUDA kernels
// for add/softmax/rmsnorm. Each adapter* entry point manages its own
// host<->device staging and allocations; callers pass plain host-side
// NamedTensor and receive host-side NamedTensor back.
//
// Layout + semantics match adapter_cpu.h verbatim. Tolerance when
// comparing outputs against adapter_cpu: atol=1e-4, rtol=1e-3.

namespace tsy::runtime {

using Tensor = tsy::lir::NamedTensor;

// MatMul: C[M,N] = A[M,K] @ B[K,N], FP32, row-major.
// Internally: cuBLAS sgemm via the col-major trick
//   row-major (A @ B) == col-major (B^T^T @ A^T) == col-major (B @ A^T)
void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c);

// Elementwise add, same shape required.
void adapterAddCuda(const Tensor& a, const Tensor& b, Tensor& c);

// Softmax along the innermost dim. Numerically-stable two-pass
// max/sum/normalize kernel, one block per outer row.
void adapterSoftmaxCuda(const Tensor& x, Tensor& y);

// RMSNorm along the innermost dim with eps=1e-6 and an implicit
// ones-weight vector (the HIR doesn't expose gain yet — matches CPU
// adapter behaviour). Warp-shuffle reduce over x^2.
void adapterRMSNormCuda(const Tensor& x, Tensor& y);

// Executor entry points (parallel to runWithCpuAdapter).
tsy::lir::RunResult runWithCudaAdapter(const tsy::lir::Module& m,
                                        tsy::DiagnosticEngine& diag);

tsy::lir::RunResult runNamedWithCudaAdapter(const tsy::lir::Module& m,
                                             const std::string& name,
                                             tsy::DiagnosticEngine& diag);

}  // namespace tsy::runtime
```

- [ ] **Step 2: Build to verify the header compiles with the stub**

Run:
```bash
cmake --build build -j --target tsy_runtime_cuda
```

Expected: builds clean (no errors — the header isn't included by anything yet).

---

## Task 3: TDD `adapterMatMulCuda` — red / green / refactor

**Goal:** One failing test → minimal passing cuBLAS implementation. The matmul is the highest-risk primitive because of the row-major/col-major trick, so it gets full TDD treatment before any others.

**Files:**
- Create: `tests/adapter/test_adapter_cuda_cases.cpp`
- Modify: `tests/CMakeLists.txt` (append new ctest entries gated on `TARGET tsy_runtime_cuda`)
- Modify: `src/runtime/adapter_cuda.cu` (replace stub with real impl)

- [ ] **Step 1: Create the test file with one tiny 2×2×2 matmul case**

```cpp
// W8 adapter_cuda cases — parity with adapter_cpu on a handful of shapes.
// Each test runs the same primitive on the CPU adapter and on the CUDA
// adapter with identical inputs; both outputs must agree within
// atol=1e-4, rtol=1e-3 (FP32 cuBLAS vs FP32 CPU reference).

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/runtime/adapter_cpu.h"
#include "../../src/runtime/adapter_cuda.h"

using tsy::lir::NamedTensor;
using tsy::runtime::adapterAdd;
using tsy::runtime::adapterAddCuda;
using tsy::runtime::adapterMatMul;
using tsy::runtime::adapterMatMulCuda;
using tsy::runtime::adapterRMSNorm;
using tsy::runtime::adapterRMSNormCuda;
using tsy::runtime::adapterSoftmax;
using tsy::runtime::adapterSoftmaxCuda;

namespace {

int g_failures = 0;

void fail(const std::string& label, const std::string& why) {
    std::cerr << "FAIL[" << label << "]: " << why << "\n";
    ++g_failures;
}

bool closeEnough(float a, float b, float atol, float rtol) {
    float diff = std::fabs(a - b);
    float tol  = atol + rtol * std::fabs(b);
    return diff <= tol;
}

NamedTensor makeTensor(const std::string& name,
                       const std::vector<int64_t>& dims,
                       std::vector<float> data) {
    NamedTensor t;
    t.name = name;
    t.dims = dims;
    t.data = std::move(data);
    return t;
}

NamedTensor zeros(const std::string& name, const std::vector<int64_t>& dims) {
    NamedTensor t;
    t.name = name;
    t.dims = dims;
    int64_t n = 1;
    for (auto d : dims) n *= d;
    t.data.assign(n, 0.0f);
    return t;
}

std::vector<float> linspace(int64_t n, float start = 0.0f, float step = 0.1f) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = start + step * static_cast<float>(i);
    return v;
}

bool compareTensors(const std::string& label,
                    const NamedTensor& cpu, const NamedTensor& cuda,
                    float atol = 1e-4f, float rtol = 1e-3f) {
    if (cpu.data.size() != cuda.data.size()) {
        fail(label, "size mismatch: cpu=" + std::to_string(cpu.data.size()) +
                     " cuda=" + std::to_string(cuda.data.size()));
        return false;
    }
    for (size_t i = 0; i < cpu.data.size(); ++i) {
        if (!closeEnough(cuda.data[i], cpu.data[i], atol, rtol)) {
            std::ostringstream oss;
            oss << "idx=" << i << " cpu=" << cpu.data[i]
                << " cuda=" << cuda.data[i]
                << " diff=" << std::fabs(cuda.data[i] - cpu.data[i]);
            fail(label, oss.str());
            return false;
        }
    }
    return true;
}

// MatMul parity: random-ish inputs, compare CPU and CUDA outputs.
void testMatmul(const std::string& label, int64_t M, int64_t K, int64_t N) {
    auto A = makeTensor("A", {M, K}, linspace(M * K, 0.0f, 0.1f));
    auto B = makeTensor("B", {K, N}, linspace(K * N, 0.5f, 0.1f));
    auto C_cpu  = zeros("C", {M, N});
    auto C_cuda = zeros("C", {M, N});
    adapterMatMul(A, B, C_cpu);
    adapterMatMulCuda(A, B, C_cuda);
    compareTensors(label, C_cpu, C_cuda);
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    testMatmul("matmul/2x2x2", 2, 2, 2);

    if (g_failures == 0) {
        std::cout << "adapter_cuda_cases: ALL PASS\n";
        return 0;
    }
    std::cerr << "adapter_cuda_cases: " << g_failures << " failure(s)\n";
    return 1;
}
```

- [ ] **Step 2: Register the test in `tests/CMakeLists.txt`**

Append this block after the existing W6 `tsy_runtime_cpu` gated block (after line 78):

```cmake
# W8: CUDA adapter tests. Outer gate is just `tsy_runtime_cuda`.
# The parity test has an inner gate because it links against BOTH
# runtimes (it compares CPU vs CUDA outputs).
if(TARGET tsy_runtime_cuda)
    if(TARGET tsy_runtime_cpu)
        add_executable(test_adapter_cuda_cases adapter/test_adapter_cuda_cases.cpp)
        target_link_libraries(test_adapter_cuda_cases
            PRIVATE tsy_runtime_cpu tsy_runtime_cuda)

        add_test(NAME adapter_cuda_cases
            COMMAND test_adapter_cuda_cases ${CMAKE_SOURCE_DIR}/examples
        )
    endif()

    # ⋯ (subsequent tasks append CLI + codegen tests here, outside the
    # inner tsy_runtime_cpu gate — they only need the CUDA runtime.)
endif()
```

- [ ] **Step 3: Configure + build + run test — verify it FAILS with a link error**

Run:
```bash
cmake -S . -B build
cmake --build build -j --target test_adapter_cuda_cases 2>&1 | tail -20
```

Expected: **build failure** with undefined reference to `adapterMatMulCuda`. That's the "red" state — test compiles but doesn't link because the stub file doesn't define it yet.

- [ ] **Step 4: Replace stub `src/runtime/adapter_cuda.cu` with the first real implementation**

Replace the stub file's content with:

```cpp
#include "adapter_cuda.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace tsy;
using namespace tsy::lir;

namespace tsy::runtime {

namespace {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(_err));        \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _st = (call);                                           \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",         \
                         __FILE__, __LINE__, static_cast<int>(_st));           \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Lazy-initialised, process-singleton cuBLAS handle. No locking because the
// executor is single-threaded.
cublasHandle_t getCublasHandle() {
    static cublasHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() { CUBLAS_CHECK(cublasCreate(&handle)); });
    return handle;
}

}  // namespace

// C[M,N] = A[M,K] @ B[K,N] (row-major, FP32).
// cuBLAS is col-major; use the identity
//   row-major (A @ B) == col-major (B @ A)  (since col-major M is row-major M^T)
// so call sgemm(OP_N, OP_N, N, M, K, B, A, C).
void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c) {
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

    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        getCublasHandle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB, N,       // leading dim of B in col-major view is N
        dA, K,       // leading dim of A in col-major view is K
        &beta,
        dC, N));

    c.data.assign(static_cast<size_t>(M) * N, 0.0f);
    CUDA_CHECK(cudaMemcpy(c.data.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// Other primitives added in later tasks — linker-time stubs until then.
void adapterAddCuda(const Tensor&, const Tensor&, Tensor&) {
    std::fprintf(stderr, "adapterAddCuda: not implemented (Task 5)\n");
    std::abort();
}
void adapterSoftmaxCuda(const Tensor&, Tensor&) {
    std::fprintf(stderr, "adapterSoftmaxCuda: not implemented (Task 6)\n");
    std::abort();
}
void adapterRMSNormCuda(const Tensor&, Tensor&) {
    std::fprintf(stderr, "adapterRMSNormCuda: not implemented (Task 7)\n");
    std::abort();
}

// Executor added in Task 10.
tsy::lir::RunResult runWithCudaAdapter(const tsy::lir::Module&,
                                        tsy::DiagnosticEngine& diag) {
    diag.error({}, "cuda-adapter executor not implemented (Task 10)");
    return {};
}
tsy::lir::RunResult runNamedWithCudaAdapter(const tsy::lir::Module&,
                                             const std::string&,
                                             tsy::DiagnosticEngine& diag) {
    diag.error({}, "cuda-adapter executor not implemented (Task 10)");
    return {};
}

}  // namespace tsy::runtime
```

- [ ] **Step 5: Build and run the test — verify it PASSES**

Run:
```bash
cmake --build build -j --target test_adapter_cuda_cases
ctest --test-dir build -R adapter_cuda_cases --output-on-failure
```

Expected output:
```
adapter_cuda_cases: ALL PASS
...
100% tests passed, 1 test ran out of 1
```

If the single test fails with a numerical mismatch, the col-major trick in `cublasSgemm` is wrong — re-check the `N, M, K` argument order.

---

## Task 4: Commit 1 — runtime matmul milestone

- [ ] **Step 1: Verify full suite green (22/22 on CUDA box, 21/21 on non-CUDA)**

```bash
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 22`.

- [ ] **Step 2: Initialize git repo if missing, then commit**

If `.git` doesn't exist:
```bash
git init -b main
git add -A
git commit -m "snapshot: W0-W7 on WSL (21/21 green)"
```

Then commit the W8-stage-1 work:
```bash
git add CMakeLists.txt tests/CMakeLists.txt \
        src/runtime/adapter_cuda.h src/runtime/adapter_cuda.cu \
        tests/adapter/test_adapter_cuda_cases.cpp \
        docs/superpowers/
git commit -m "$(cat <<'EOF'
feat(runtime): add CUDA FP32 matmul adapter + tsy_runtime_cuda

First W8 milestone: tsy_runtime_cuda target builds under CUDA 12 / sm_86
with a WSL libcuda RPATH fix borrowed from mini-llm-engine. adapterMatMulCuda
calls cuBLAS sgemm using the row-major↔col-major identity so we don't
materialise a transpose buffer. adapter_cuda_cases pins a 2x2x2 parity
case against adapter_cpu (atol=1e-4, rtol=1e-3). Other primitives stubbed
with abort() until Tasks 5-7.
EOF
)"
```

---

## Task 5: TDD `adapterAddCuda`

**Goal:** Second primitive — the simplest (flat elementwise). Confirms the "host copy in / kernel / copy out" pattern before we tackle softmax/rmsnorm.

**Files:**
- Modify: `src/runtime/adapter_cuda.cu`
- Modify: `tests/adapter/test_adapter_cuda_cases.cpp`

- [ ] **Step 1: Add failing test case**

Inside the anonymous namespace of `tests/adapter/test_adapter_cuda_cases.cpp`, just before the closing `}  // namespace`, add:

```cpp
void testAdd(const std::string& label,
             const std::vector<int64_t>& dims) {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    auto A = makeTensor("A", dims, linspace(n, 0.0f, 0.1f));
    auto B = makeTensor("B", dims, linspace(n, 1.0f, 0.2f));
    auto C_cpu  = zeros("C", dims);
    auto C_cuda = zeros("C", dims);
    adapterAdd(A, B, C_cpu);
    adapterAddCuda(A, B, C_cuda);
    compareTensors(label, C_cpu, C_cuda);
}
```

And add a test call inside `main()`:

```cpp
    testAdd("add/8x8", {8, 8});
```

- [ ] **Step 2: Run — expect test process to abort() with "adapterAddCuda: not implemented"**

Run:
```bash
cmake --build build -j --target test_adapter_cuda_cases
ctest --test-dir build -R adapter_cuda_cases --output-on-failure
```

Expected: test ABORTS (non-zero exit) because the stub `adapterAddCuda` calls `std::abort()`.

- [ ] **Step 3: Implement the add kernel in `src/runtime/adapter_cuda.cu`**

Above the stub `adapterAddCuda` definition, add:

```cpp
__global__ void addKernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

Replace the stub `adapterAddCuda` (the one that aborts) with:

```cpp
void adapterAddCuda(const Tensor& a, const Tensor& b, Tensor& c) {
    assert(a.data.size() == b.data.size());
    const int n = static_cast<int>(a.data.size());
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, a.data.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid  = (n + block - 1) / block;
    addKernel<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());

    c.data.assign(static_cast<size_t>(n), 0.0f);
    CUDA_CHECK(cudaMemcpy(c.data.data(), dC, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
```

- [ ] **Step 4: Build + re-run test — expect PASS**

Run:
```bash
cmake --build build -j --target test_adapter_cuda_cases
ctest --test-dir build -R adapter_cuda_cases --output-on-failure
```

Expected: `adapter_cuda_cases: ALL PASS`.

---

## Task 6: TDD `adapterSoftmaxCuda`

**Goal:** Third primitive — numerically-stable two-pass softmax with shared-memory block reduce.

**Files:**
- Modify: `src/runtime/adapter_cuda.cu`
- Modify: `tests/adapter/test_adapter_cuda_cases.cpp`

- [ ] **Step 1: Add failing test case**

Inside the anonymous namespace, before the closing `}`, add:

```cpp
void testSoftmax(const std::string& label,
                 const std::vector<int64_t>& dims) {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    auto X = makeTensor("X", dims, linspace(n, -1.0f, 0.3f));
    auto Y_cpu  = zeros("Y", dims);
    auto Y_cuda = zeros("Y", dims);
    adapterSoftmax(X, Y_cpu);
    adapterSoftmaxCuda(X, Y_cuda);
    compareTensors(label, Y_cpu, Y_cuda);
}
```

And add inside `main()`:

```cpp
    testSoftmax("softmax/8x8", {8, 8});
```

- [ ] **Step 2: Run — expect abort() from the softmax stub**

```bash
cmake --build build -j --target test_adapter_cuda_cases
ctest --test-dir build -R adapter_cuda_cases --output-on-failure
```

Expected: test ABORTS on `testSoftmax`.

- [ ] **Step 3: Add the softmax kernel in `src/runtime/adapter_cuda.cu`**

Above the stub `adapterSoftmaxCuda`, add:

```cpp
__global__ void softmaxRowKernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int inner) {
    extern __shared__ float smem[];   // blockDim.x floats
    const int row = blockIdx.x;
    const float* xr = x + row * inner;
    float*       yr = y + row * inner;

    // Pass 1: max reduction
    float tmax = -1e30f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        tmax = fmaxf(tmax, xr[i]);
    }
    smem[threadIdx.x] = tmax;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x],
                                       smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float row_max = smem[0];
    __syncthreads();

    // Pass 2: exp + sum reduction
    float tsum = 0.0f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float v = expf(xr[i] - row_max);
        yr[i] = v;
        tsum += v;
    }
    smem[threadIdx.x] = tsum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / smem[0];
    __syncthreads();

    // Pass 3: normalise
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        yr[i] *= inv;
    }
}
```

Replace the aborting `adapterSoftmaxCuda` with:

```cpp
void adapterSoftmaxCuda(const Tensor& x, Tensor& y) {
    assert(!x.dims.empty());
    const int64_t inner = x.dims.back();
    int64_t outer = 1;
    for (size_t i = 0; i + 1 < x.dims.size(); ++i) outer *= x.dims[i];
    const size_t n = x.data.size();
    const size_t bytes = n * sizeof(float);

    float *dX = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dY, bytes));
    CUDA_CHECK(cudaMemcpy(dX, x.data.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const size_t smem_bytes = static_cast<size_t>(block) * sizeof(float);
    softmaxRowKernel<<<static_cast<int>(outer), block, smem_bytes>>>(
        dX, dY, static_cast<int>(inner));
    CUDA_CHECK(cudaGetLastError());

    y.data.assign(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(y.data.data(), dY, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dY);
}
```

- [ ] **Step 4: Build + run — expect PASS**

```bash
cmake --build build -j --target test_adapter_cuda_cases
ctest --test-dir build -R adapter_cuda_cases --output-on-failure
```

Expected: `ALL PASS`.

---

## Task 7: TDD `adapterRMSNormCuda`

**Goal:** Fourth primitive. Warp-shuffle reduction for `sum(x^2)` + device-side ones-weight cache keyed by inner dim.

**Files:**
- Modify: `src/runtime/adapter_cuda.cu`
- Modify: `tests/adapter/test_adapter_cuda_cases.cpp`

- [ ] **Step 1: Add failing test case**

Inside the anonymous namespace, before the closing `}`, add:

```cpp
void testRMSNorm(const std::string& label,
                 const std::vector<int64_t>& dims) {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    auto X = makeTensor("X", dims, linspace(n, 0.5f, 0.25f));
    auto Y_cpu  = zeros("Y", dims);
    auto Y_cuda = zeros("Y", dims);
    adapterRMSNorm(X, Y_cpu);
    adapterRMSNormCuda(X, Y_cuda);
    compareTensors(label, Y_cpu, Y_cuda);
}
```

And inside `main()`:

```cpp
    testRMSNorm("rmsnorm/8x8", {8, 8});
```

- [ ] **Step 2: Run — expect abort()**

```bash
cmake --build build -j --target test_adapter_cuda_cases
ctest --test-dir build -R adapter_cuda_cases --output-on-failure
```

Expected: ABORT.

- [ ] **Step 3: Add rmsnorm kernel + ones cache in `src/runtime/adapter_cuda.cu`**

Above the stub `adapterRMSNormCuda`, add:

```cpp
__global__ void rmsnormRowKernel(const float* __restrict__ x,
                                  const float* __restrict__ w,
                                  float* __restrict__ y,
                                  int inner, float eps) {
    extern __shared__ float smem[];   // 1 float per warp
    const int row = blockIdx.x;
    const float* xr = x + row * inner;
    float*       yr = y + row * inner;

    // Phase 1: sum(x^2) in FP32
    float sum = 0.0f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    // Intra-warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    const int warp_id  = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    const int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) smem[0] = sum;
    }
    __syncthreads();

    const float rms_inv = rsqrtf(smem[0] / static_cast<float>(inner) + eps);

    // Phase 2: normalise + scale
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        yr[i] = xr[i] * rms_inv * w[i];
    }
}

// Device-side ones-weight cache keyed by inner dim.
const float* onesDeviceVector(int64_t inner) {
    static std::unordered_map<int64_t, float*> cache;
    auto it = cache.find(inner);
    if (it != cache.end()) return it->second;
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, inner * sizeof(float)));
    std::vector<float> host(inner, 1.0f);
    CUDA_CHECK(cudaMemcpy(d, host.data(), inner * sizeof(float),
                          cudaMemcpyHostToDevice));
    cache[inner] = d;
    return d;
}
```

Replace the aborting `adapterRMSNormCuda` with:

```cpp
void adapterRMSNormCuda(const Tensor& x, Tensor& y) {
    constexpr float kEps = 1e-6f;
    assert(!x.dims.empty());
    const int64_t inner = x.dims.back();
    int64_t outer = 1;
    for (size_t i = 0; i + 1 < x.dims.size(); ++i) outer *= x.dims[i];
    const size_t n = x.data.size();
    const size_t bytes = n * sizeof(float);

    float *dX = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dY, bytes));
    CUDA_CHECK(cudaMemcpy(dX, x.data.data(), bytes, cudaMemcpyHostToDevice));
    const float* dW = onesDeviceVector(inner);

    const int block = 128;  // 4 warps
    const size_t smem_bytes = 4 * sizeof(float);
    rmsnormRowKernel<<<static_cast<int>(outer), block, smem_bytes>>>(
        dX, dW, dY, static_cast<int>(inner), kEps);
    CUDA_CHECK(cudaGetLastError());

    y.data.assign(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(y.data.data(), dY, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dY);
}
```

- [ ] **Step 4: Build + run — expect PASS**

```bash
cmake --build build -j --target test_adapter_cuda_cases
ctest --test-dir build -R adapter_cuda_cases --output-on-failure
```

Expected: `ALL PASS`.

---

## Task 8: Fill in the full shape matrix

**Goal:** Per spec §5.6, exercise all 5 classes on matmul and 3 classes on add/softmax/rmsnorm. Lock in tolerance.

**Files:**
- Modify: `tests/adapter/test_adapter_cuda_cases.cpp`

- [ ] **Step 1: Expand `main()` to cover the full shape matrix**

Replace the body of `main()` with:

```cpp
int main(int /*argc*/, char** /*argv*/) {
    // MatMul — 5 shape classes (M, K, N)
    testMatmul("matmul/square-8",    8,   8,  8);
    testMatmul("matmul/tall-128x16", 128, 16, 8);
    testMatmul("matmul/odd-7x13",    7,   13, 11);
    testMatmul("matmul/1xK-row",     1,   32, 8);
    testMatmul("matmul/Kx1-col",     8,   32, 1);

    // Add / Softmax / RMSNorm — 3 shape classes
    for (auto dims : std::vector<std::vector<int64_t>>{
             {8, 8}, {7, 13}, {1, 32}}) {
        std::string label = "shape-" + std::to_string(dims[0]) + "x"
                                      + std::to_string(dims[1]);
        testAdd    ("add/"     + label, dims);
        testSoftmax("softmax/" + label, dims);
        testRMSNorm("rmsnorm/" + label, dims);
    }

    if (g_failures == 0) {
        std::cout << "adapter_cuda_cases: ALL PASS\n";
        return 0;
    }
    std::cerr << "adapter_cuda_cases: " << g_failures << " failure(s)\n";
    return 1;
}
```

- [ ] **Step 2: Build + run full matrix**

```bash
cmake --build build -j --target test_adapter_cuda_cases
ctest --test-dir build -R adapter_cuda_cases --output-on-failure
```

Expected: `adapter_cuda_cases: ALL PASS` (14 sub-cases — 5 matmul + 3×3 others).

- [ ] **Step 3: If any case fails with tolerance overshoot**

Try first: raise to `atol=5e-4, rtol=1e-3` per spec §3 fallback. Edit `compareTensors` default args in the test file. If still failing on a specific shape, diagnose: for rmsnorm, check `inner < 32` edge case (warp reduce needs `num_warps >= 1`). The block=128 gives `num_warps=4`, so inner<128 still works, but we still need `inner >= 1`. For the `1x32` case inner=32 is fine.

---

## Task 9: Commit 2 — all primitives milestone

- [ ] **Step 1: Verify full suite (22/22 on CUDA box)**

```bash
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 22`.

- [ ] **Step 2: Commit**

```bash
git add src/runtime/adapter_cuda.cu tests/adapter/test_adapter_cuda_cases.cpp
git commit -m "$(cat <<'EOF'
feat(runtime): implement add/softmax/rmsnorm CUDA kernels + parity matrix

adapterAddCuda: flat elementwise, block=256.
adapterSoftmaxCuda: two-pass max/sum/normalise, one block per outer row,
  block=256 with shared-mem reduce.
adapterRMSNormCuda: warp-shuffle reduce of sum(x^2) with device-side
  ones-weight cache (no gain vector in HIR yet, matches CPU adapter).

adapter_cuda_cases now exercises 14 sub-cases: 5 matmul shape classes
(square/tall/odd/1xK/Kx1) + 3x3 (add, softmax, rmsnorm) across
square/odd/1xK. All within atol=1e-4, rtol=1e-3 of the CPU adapter.
EOF
)"
```

---

## Task 10: Implement `runWithCudaAdapter` executor

**Goal:** Mirror `runWithCpuAdapter` so `tsc run-lir --backend=cuda-adapter` can execute whole LIR modules.

**Files:**
- Modify: `src/runtime/adapter_cuda.cu`

- [ ] **Step 1: Replace the placeholder executor in `src/runtime/adapter_cuda.cu`**

Remove the two stubs (`runWithCudaAdapter` and `runNamedWithCudaAdapter`) that currently report errors and insert this block above them:

```cpp
namespace {

RunResult runFunctionCudaAdapter(const Function& f, DiagnosticEngine& diag) {
    RunResult r;
    r.function_name = f.name;
    r.buffers.reserve(f.buffers.size());
    for (const auto& b : f.buffers) {
        NamedTensor t;
        t.name = b.name;
        t.dims = b.dims;
        t.data.assign(b.numElements(), 0.0f);
        r.buffers.push_back(std::move(t));
    }

    for (size_t i = 0; i < f.params.size(); ++i) {
        fillDeterministic(r.buffers[f.params[i]], static_cast<int>(i));
    }

    for (const auto& s : f.body) {
        if (s.kind == StmtKind::Return) break;
        if (s.kind != StmtKind::Call) continue;
        if (s.result_buf < 0) {
            diag.error(s.loc, "cuda-adapter: call has no result buffer");
            r.ok = false;
            continue;
        }
        auto& out = r.buffers[s.result_buf];

        if (s.primitive == "matmul") {
            if (s.operand_bufs.size() != 2) {
                diag.error(s.loc, "cuda-adapter matmul: expected 2 operands");
                r.ok = false; continue;
            }
            adapterMatMulCuda(r.buffers[s.operand_bufs[0]],
                              r.buffers[s.operand_bufs[1]], out);
        } else if (s.primitive == "add") {
            if (s.operand_bufs.size() != 2) {
                diag.error(s.loc, "cuda-adapter add: expected 2 operands");
                r.ok = false; continue;
            }
            adapterAddCuda(r.buffers[s.operand_bufs[0]],
                           r.buffers[s.operand_bufs[1]], out);
        } else if (s.primitive == "softmax") {
            if (s.operand_bufs.size() != 1) {
                diag.error(s.loc, "cuda-adapter softmax: expected 1 operand");
                r.ok = false; continue;
            }
            adapterSoftmaxCuda(r.buffers[s.operand_bufs[0]], out);
        } else if (s.primitive == "rmsnorm") {
            if (s.operand_bufs.size() != 1) {
                diag.error(s.loc, "cuda-adapter rmsnorm: expected 1 operand");
                r.ok = false; continue;
            }
            adapterRMSNormCuda(r.buffers[s.operand_bufs[0]], out);
        } else {
            diag.error(s.loc, "cuda-adapter: unsupported primitive '" +
                                   s.primitive + "'");
            r.ok = false;
        }
    }

    for (size_t i = 0; i < f.params.size(); ++i) {
        r.buffers[f.params[i]].is_param = true;
    }
    return r;
}

const Function* pickFirstTensorFunction(const Module& m) {
    for (const auto& f : m.funcs) {
        if (f->name == "main") continue;
        if (!f->params.empty()) return f.get();
    }
    if (!m.funcs.empty()) return m.funcs.front().get();
    return nullptr;
}

}  // namespace

RunResult runWithCudaAdapter(const Module& m, DiagnosticEngine& diag) {
    const Function* f = pickFirstTensorFunction(m);
    if (!f) {
        diag.error({}, "cuda-adapter: module has no runnable function");
        RunResult r; r.ok = false; return r;
    }
    return runFunctionCudaAdapter(*f, diag);
}

RunResult runNamedWithCudaAdapter(const Module& m, const std::string& name,
                                   DiagnosticEngine& diag) {
    for (const auto& f : m.funcs) {
        if (f->name == name) return runFunctionCudaAdapter(*f, diag);
    }
    diag.error({}, "cuda-adapter: function '" + name + "' not found");
    RunResult r; r.ok = false; return r;
}
```

- [ ] **Step 2: Build to verify compilation**

```bash
cmake --build build -j --target tsy_runtime_cuda
```

Expected: clean build.

- [ ] **Step 3: Run adapter tests to confirm nothing regressed**

```bash
ctest --test-dir build -R adapter_cuda_cases --output-on-failure
```

Expected: `ALL PASS` (same 14 cases).

---

## Task 11: `tsc run-lir --backend=cuda-adapter` CLI + test

**Goal:** Wire the executor through the CLI and register a ctest that runs the 2×2 matmul tiny fixture.

**Files:**
- Modify: `src/tools/tsc.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Modify `src/tools/tsc.cpp` — add include guard block and backend dispatch**

Below the existing `#if TSY_HAVE_RUNTIME_CPU` block near the top (around lines 17–19), add:

```cpp
#if TSY_HAVE_RUNTIME_CUDA
#include "../runtime/adapter_cuda.h"
#endif
```

In the `kUsage` string (lines 23–48), in the `run-lir backends:` section, add a new line after the `cpu-adapter` entry:

```cpp
    "  --backend=cuda-adapter W8 self-written FP32 CUDA kernels + cuBLAS.\n"
```

In `cmdRunLir` (around lines 198–212 — search for `if (o.backend == "cpu-adapter")`), insert a new branch between the `cpu-adapter` branch and the `native` branch:

```cpp
    } else if (o.backend == "cuda-adapter") {
#if TSY_HAVE_RUNTIME_CUDA
        result = tsy::runtime::runWithCudaAdapter(*lmod, diag);
#else
        std::cerr << "run-lir: --backend=cuda-adapter requires tsy_runtime_cuda, "
                     "which was not built. Check CUDA toolchain at CMake time.\n";
        return 1;
#endif
```

Full resulting `cmdRunLir` backend dispatch should read:
```cpp
    if (o.backend == "cpu-adapter") {
#if TSY_HAVE_RUNTIME_CPU
        result = tsy::runtime::runWithCpuAdapter(*lmod, diag);
#else
        ...
#endif
    } else if (o.backend == "cuda-adapter") {
#if TSY_HAVE_RUNTIME_CUDA
        result = tsy::runtime::runWithCudaAdapter(*lmod, diag);
#else
        std::cerr << "run-lir: --backend=cuda-adapter requires tsy_runtime_cuda, "
                     "which was not built. Check CUDA toolchain at CMake time.\n";
        return 1;
#endif
    } else if (o.backend == "native" || o.backend.empty()) {
        result = tsy::lir::runFirstTensorFunction(*lmod, diag);
    } else {
        std::cerr << "run-lir: unknown backend '" << o.backend << "'\n";
        return 1;
    }
```

- [ ] **Step 2: Register the CLI ctest in `tests/CMakeLists.txt`**

Inside the existing `if(TARGET tsy_runtime_cuda)` block (added in Task 3), append:

```cmake
    add_test(NAME cli_cuda_adapter_run_matmul_tiny
        COMMAND tsc run-lir --backend=cuda-adapter
                ${CMAKE_SOURCE_DIR}/examples/run_matmul_tiny.tsy
    )
    set_tests_properties(cli_cuda_adapter_run_matmul_tiny PROPERTIES
        PASS_REGULAR_EXPRESSION "local C .*0.0700.*0.0800"
    )
```

- [ ] **Step 3: Build + run**

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 23` — 21 original + `adapter_cuda_cases` + `cli_cuda_adapter_run_matmul_tiny`.

- [ ] **Step 4: Manual sanity — run the 2×2 matmul via the CUDA backend and eyeball the output**

```bash
./build/tsc run-lir --backend=cuda-adapter examples/run_matmul_tiny.tsy
```

Expected final block:
```
  local C shape=[2,2]:
      0.0700   0.0800
      0.3100   0.3600
```

(Exact same values as the `native` and `cpu-adapter` backends — FP32 cuBLAS and CPU agree bit-approximately on 2×2×2.)

---

## Task 12: Commit 3 — run-lir CUDA backend

- [ ] **Step 1: Commit**

```bash
git add src/runtime/adapter_cuda.cu src/tools/tsc.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(cli): run-lir --backend=cuda-adapter

runWithCudaAdapter executor mirrors runWithCpuAdapter and dispatches
every LIR Call through adapter*Cuda. tsc.cpp picks up the new backend
under a TSY_HAVE_RUNTIME_CUDA compile gate — machines without CUDA get
a clear error message at runtime rather than a link failure.

cli_cuda_adapter_run_matmul_tiny locks the 2x2x2 output against the
same {0.0700, 0.0800, 0.3100, 0.3600} the CPU backends produce.
EOF
)"
```

---

## Task 13: `src/codegen/cuda.{h,cpp}` — `.cu` source generator

**Goal:** Generate a self-contained `.cu` host file that `#include`s `adapter_cuda.h` and calls the `adapterXxxCuda` entry points — exact parallel to `emitCppModule` in `src/codegen/cpp.cpp`.

**Files:**
- Create: `src/codegen/cuda.h`
- Create: `src/codegen/cuda.cpp`
- Modify: `CMakeLists.txt` (add `cuda.cpp` to `tsy_codegen` sources)

- [ ] **Step 1: Create `src/codegen/cuda.h`**

```cpp
#pragma once

#include <ostream>
#include <string>

#include "../lir/ir.h"

namespace tsy::codegen {

// Write a self-contained .cu source that runs the LIR module through
// adapter_cuda at runtime. The generated binary is expected to link
// against tsy_runtime_cuda. The output is pure host C++ with .cu
// extension so nvcc handles the one-step compile + link.
bool emitCudaModule(std::ostream& os, const tsy::lir::Module& m,
                    const std::string& source_path);

}  // namespace tsy::codegen
```

- [ ] **Step 2: Create `src/codegen/cuda.cpp`**

```cpp
#include "cuda.h"

#include <sstream>
#include <unordered_set>

using namespace tsy::lir;

namespace tsy::codegen {

namespace {

std::string identFor(const Buffer& b) {
    return "buf_" + b.name;
}

const Function* pickFirstTensorFunction(const Module& m) {
    for (const auto& f : m.funcs) {
        if (f->name == "main") continue;
        if (!f->params.empty()) return f.get();
    }
    return m.funcs.empty() ? nullptr : m.funcs.front().get();
}

const char* adapterSymbolFor(const std::string& primitive) {
    if (primitive == "matmul")  return "adapterMatMulCuda";
    if (primitive == "add")     return "adapterAddCuda";
    if (primitive == "softmax") return "adapterSoftmaxCuda";
    if (primitive == "rmsnorm") return "adapterRMSNormCuda";
    return nullptr;
}

void writeHeader(std::ostream& os, const std::string& source_path) {
    os << "// AUTO-GENERATED by `tsc emit-cu`. Do not edit.\n"
       << "// Source: " << source_path << "\n"
       << "//\n"
       << "// Links against tsy_runtime_cuda; every builtin dispatches through\n"
       << "// the CUDA adapter (cuBLAS sgemm + hand-written FP32 kernels).\n"
       << "\n"
       << "#include <cstdint>\n"
       << "#include <initializer_list>\n"
       << "#include <iostream>\n"
       << "#include <utility>\n"
       << "#include <vector>\n"
       << "\n"
       << "#include \"adapter_cuda.h\"\n"
       << "#include \"interpreter.h\"\n"
       << "\n"
       << "namespace {\n"
       << "\n"
       << "tsy::lir::NamedTensor makeBuf(const char* name,\n"
       << "                              std::initializer_list<int64_t> dims) {\n"
       << "    tsy::lir::NamedTensor t;\n"
       << "    t.name = name;\n"
       << "    t.dims.assign(dims.begin(), dims.end());\n"
       << "    std::int64_t n = 1;\n"
       << "    for (auto d : dims) n *= d;\n"
       << "    t.data.assign(static_cast<std::size_t>(n), 0.0f);\n"
       << "    return t;\n"
       << "}\n"
       << "\n"
       << "}  // namespace\n"
       << "\n";
}

}  // namespace

bool emitCudaModule(std::ostream& os, const Module& m,
                    const std::string& source_path) {
    writeHeader(os, source_path);

    const Function* f = pickFirstTensorFunction(m);
    if (!f) {
        os << "int main() { return 0; }\n";
        return true;
    }

    os << "int main() {\n";

    for (const auto& b : f->buffers) {
        os << "    auto " << identFor(b) << " = makeBuf(\"" << b.name << "\", {";
        for (size_t i = 0; i < b.dims.size(); ++i) {
            if (i) os << ", ";
            os << b.dims[i];
        }
        os << "});\n";
    }
    os << "\n";

    std::unordered_set<int> param_set(f->params.begin(), f->params.end());
    for (size_t i = 0; i < f->params.size(); ++i) {
        const auto& b = f->buffers[f->params[i]];
        os << "    " << identFor(b) << ".is_param = true;\n";
        os << "    tsy::lir::fillDeterministic(" << identFor(b) << ", "
           << static_cast<int>(i) << ");\n";
    }
    if (!f->params.empty()) os << "\n";

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
        os << ", " << identFor(f->buffers[s.result_buf]) << ");\n";
    }
    os << "\n";

    os << "    tsy::lir::RunResult result;\n"
       << "    result.function_name = \"" << f->name << "\";\n"
       << "    result.ok = true;\n";
    for (const auto& b : f->buffers) {
        os << "    result.buffers.push_back(std::move(" << identFor(b) << "));\n";
    }
    os << "    tsy::lir::printRunResult(std::cout, result);\n"
       << "    return 0;\n"
       << "}\n";

    return true;
}

}  // namespace tsy::codegen
```

- [ ] **Step 3: Modify `CMakeLists.txt` — add cuda.cpp to tsy_codegen**

Find the `tsy_codegen` library block (around lines 103–113) and change:
```cmake
add_library(tsy_codegen STATIC
    src/codegen/cpp.cpp
)
```
to:
```cmake
add_library(tsy_codegen STATIC
    src/codegen/cpp.cpp
    src/codegen/cuda.cpp
)
```

- [ ] **Step 4: Build to confirm tsy_codegen still compiles**

```bash
cmake --build build -j --target tsy_codegen
```

Expected: clean build.

---

## Task 14: `tsc emit-cu` CLI command

**Goal:** Wire emit-cu into `tsc` with the same output-path logic as emit-cpp.

**Files:**
- Modify: `src/tools/tsc.cpp`

- [ ] **Step 1: Add the cuda codegen include near the top of `tsc.cpp`**

After the existing `#include "../codegen/cpp.h"` line, add:

```cpp
#include "../codegen/cuda.h"
```

- [ ] **Step 2: Extend `kUsage` — add `emit-cu` to the Commands section**

In the `kUsage` string (near the top), find the `emit-cpp` line and add after it:

```cpp
    "  emit-cu   Generate a self-contained CUDA .cu host binary source.\n"
```

- [ ] **Step 3: Add `cmdEmitCu` function**

Just after `cmdEmitCpp` (currently ending around line 172), add a parallel function:

```cpp
int cmdEmitCu(const Options& o) {
    tsy::DiagnosticEngine diag;
    auto hmod = parseAndRunPipeline(o, diag);
    if (!hmod) return 1;
    auto lmod = tsy::lir::lowerHirToLir(*hmod, diag);
    if (!lmod || diag.hasErrors()) {
        diag.print(std::cerr);
        std::cerr << "lir lowering failed: " << o.path << "\n";
        return 1;
    }

    std::ostream* out = &std::cout;
    std::ofstream ofs;
    if (!o.output_path.empty()) {
        ofs.open(o.output_path);
        if (!ofs) {
            std::cerr << "emit-cu: cannot write to '" << o.output_path << "'\n";
            return 1;
        }
        out = &ofs;
    }
    tsy::codegen::emitCudaModule(*out, *lmod, o.path);
    return 0;
}
```

- [ ] **Step 4: Register the dispatch in `main()`**

Near the end of `main()`, right after the `emit-cpp` line:
```cpp
    if (cmd == "emit-cpp") return cmdEmitCpp(opts);
```
add:
```cpp
    if (cmd == "emit-cu") return cmdEmitCu(opts);
```

- [ ] **Step 5: Build + manual smoke**

```bash
cmake --build build -j
./build/tsc emit-cu examples/run_matmul_tiny.tsy | head -30
```

Expected output begins with:
```
// AUTO-GENERATED by `tsc emit-cu`. Do not edit.
// Source: examples/run_matmul_tiny.tsy
...
#include "adapter_cuda.h"
...
int main() {
    auto buf_A = makeBuf("A", {2, 2});
    ...
    tsy::runtime::adapterMatMulCuda(buf_A, buf_B, buf_C);
```

---

## Task 15: `examples/matmul_tiny_cuda.tsy` + `tsy_add_cuda_example` + compile smoke

**Goal:** Add a tiny fixture for CUDA codegen and the CMake helper that mirrors `tsy_add_example`. Register the build-side compile + run test.

**Files:**
- Create: `examples/matmul_tiny_cuda.tsy`
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Create the example `.tsy` file (can be a literal copy of `run_matmul_tiny.tsy`)**

Look up the content of `examples/run_matmul_tiny.tsy` and duplicate it to `examples/matmul_tiny_cuda.tsy`. These two files have identical content — we keep separate names so the codegen test has a stable target name (`matmul_cuda_demo`) that doesn't fight the W7 `run_matmul_tiny.tsy` usage. Run:

```bash
cp examples/run_matmul_tiny.tsy examples/matmul_tiny_cuda.tsy
```

Verify:
```bash
cat examples/matmul_tiny_cuda.tsy
```

Should show the same `matmul_layer(A,B,C)` definition.

- [ ] **Step 2: Add `tsy_add_cuda_example` helper in `CMakeLists.txt`**

Append this function just after the existing `tsy_add_example` function (around line 200) and call it once at the bottom:

```cmake
# --- W8: helper to compile a .tsy example into a CUDA host binary --------
function(tsy_add_cuda_example NAME TSY_FILE)
    if(NOT TSY_HAVE_RUNTIME_CUDA)
        message(STATUS "tsy_add_cuda_example(${NAME}): skipped (no CUDA).")
        return()
    endif()
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${TSY_FILE}")
    set(_gen "${CMAKE_CURRENT_BINARY_DIR}/gen/${NAME}.cu")
    add_custom_command(
        OUTPUT ${_gen}
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_BINARY_DIR}/gen
        COMMAND $<TARGET_FILE:tsc> emit-cu ${_src} -o ${_gen}
        DEPENDS tsc ${_src}
        COMMENT "tsc emit-cu ${TSY_FILE} -> gen/${NAME}.cu"
        VERBATIM
    )
    add_executable(${NAME} ${_gen})
    set_source_files_properties(${_gen} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${NAME} PRIVATE tsy_runtime_cuda)
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/out"
        CUDA_ARCHITECTURES 86)
endfunction()

tsy_add_cuda_example(matmul_cuda_demo examples/matmul_tiny_cuda.tsy)
```

(Place the `tsy_add_cuda_example(matmul_cuda_demo ...)` call **after** the existing `tsy_add_example(mlp examples/mlp.tsy)` line.)

- [ ] **Step 3: Add the compile + run ctest in `tests/CMakeLists.txt`**

Inside the `if(TARGET tsy_runtime_cuda)` block, append:

```cmake
    # Compile smoke: the generated matmul_cuda_demo must link and produce
    # the expected 2x2x2 output when run.
    add_test(NAME codegen_cuda_matmul_binary_runs
        COMMAND $<TARGET_FILE:matmul_cuda_demo>
    )
    set_tests_properties(codegen_cuda_matmul_binary_runs PROPERTIES
        PASS_REGULAR_EXPRESSION
        "function: matmul_layer.*local C shape=\\[2,2\\].*0\\.0700.*0\\.0800"
    )

    # CLI-level: emit-cu must produce source that references the adapter.
    add_test(NAME codegen_emit_cu_contains_adapter_calls
        COMMAND tsc emit-cu ${CMAKE_SOURCE_DIR}/examples/matmul_tiny_cuda.tsy
    )
    set_tests_properties(codegen_emit_cu_contains_adapter_calls PROPERTIES
        PASS_REGULAR_EXPRESSION "tsy::runtime::adapterMatMulCuda"
    )
```

- [ ] **Step 4: Configure + full build**

```bash
cmake -S . -B build
cmake --build build -j
```

Expected: clean build, including `[100%] Built target matmul_cuda_demo`.

- [ ] **Step 5: Manual sanity — run the compiled CUDA binary**

```bash
./build/out/matmul_cuda_demo
```

Expected final block:
```
  local C shape=[2,2]:
      0.0700   0.0800
      0.3100   0.3600
```

- [ ] **Step 6: Run the full ctest suite**

```bash
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 25`.

Break-down (CUDA machine):
- 21 from W0–W7
- `adapter_cuda_cases` (1 ctest entry, 14 internal cases)
- `cli_cuda_adapter_run_matmul_tiny`
- `codegen_cuda_matmul_binary_runs`
- `codegen_emit_cu_contains_adapter_calls`

---

## Task 16: Commit 4 — codegen milestone

- [ ] **Step 1: Commit**

```bash
git add src/codegen/cuda.h src/codegen/cuda.cpp \
        src/tools/tsc.cpp \
        examples/matmul_tiny_cuda.tsy \
        CMakeLists.txt tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(codegen): add emit-cu + tsy_add_cuda_example

tsc emit-cu foo.tsy [-o foo.cu] writes a self-contained .cu host binary
that calls tsy::runtime::adapterXxxCuda; generated source mirrors
emit-cpp verbatim aside from header includes and adapter symbol names.
tsy_add_cuda_example() is the build-time companion: `make matmul_cuda_demo`
runs tsc and nvcc in one shot, dropping the binary into build/out/.

Two new ctest entries lock the compile-smoke (codegen_cuda_matmul_binary_runs)
and the generator (codegen_emit_cu_contains_adapter_calls). Total ctest
count on a CUDA box is now 25/25.
EOF
)"
```

---

## Task 17: Final verification + PLAN.md update

**Goal:** End-to-end sanity. Update PLAN.md marking W8 complete.

**Files:**
- Modify: `PLAN.md`

- [ ] **Step 1: Rebuild from scratch to prove the CMake flow is idempotent**

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: `25/25 green`.

- [ ] **Step 2: Manual W8 acceptance command matrix**

Run each of these and verify the output matches the paired CPU path:

```bash
./build/tsc run-lir --backend=native        examples/run_matmul_tiny.tsy
./build/tsc run-lir --backend=cpu-adapter   examples/run_matmul_tiny.tsy
./build/tsc run-lir --backend=cuda-adapter  examples/run_matmul_tiny.tsy

./build/out/mlp                          # W7 CPU MLP — unchanged
./build/out/matmul_cuda_demo             # W8 new

./build/tsc emit-cu examples/matmul_tiny_cuda.tsy | head -20
```

All three `run-lir` backends must print `0.0700 0.0800 / 0.3100 0.3600` for C.
`matmul_cuda_demo` must print the same values.
`emit-cu` output must start with `// AUTO-GENERATED by 'tsc emit-cu'`.

- [ ] **Step 3: Update `PLAN.md` "项目期间安装的工具" table — no new entries**

On the current WSL image, `cmake 3.28.3 / bison 3.8.2 / flex 2.6.4 / g++ 13.3 / nvcc 12.0 / RTX 3080 driver` were all pre-installed. No tool-install rows to add. Leave the table unchanged. The git log is the source of truth for what was built this milestone; PLAN.md's 12-week forecast table and SETUP_WSL.md's commit-hash table are snapshots and don't need edits mid-execution.

- [ ] **Step 4: Final proof — ctest + acceptance re-run in one shot**

```bash
ctest --test-dir build --output-on-failure && \
./build/out/matmul_cuda_demo && \
./build/tsc run-lir --backend=cuda-adapter examples/run_matmul_tiny.tsy && \
echo "W8 ACCEPTANCE: GREEN"
```

Expected: `W8 ACCEPTANCE: GREEN` tail line.

- [ ] **Step 5: (Optional) commit any PLAN.md tweaks**

If you added a W8 completion note manually or updated the tools table with something unexpected:
```bash
git add PLAN.md
git commit -m "docs: note W8 completion"
```
Otherwise skip — nothing to commit.

---

## Rollback / recovery

If any task fails irrecoverably, roll back with:
```bash
git reset --hard HEAD~1   # undo last commit
# -- or --
git stash                 # stash uncommitted work
```

Non-CUDA machines: the plan degrades gracefully — `tsy_runtime_cuda` is skipped at configure time and only the 21 W0–W7 tests run. The design explicitly avoids turning W8 into a hard dependency.

---

## Spec coverage check

| Spec section | Implemented by |
|---|---|
| §3 Kernel source choice (A) | Task 1 (empty target), Tasks 3/5/6/7 (impls) |
| §4 Architecture | Tasks 2 (header), 10 (executor), 13 (codegen) |
| §5.1 adapter_cuda primitives | Tasks 3, 5, 6, 7 |
| §5.1 Ones-weight device cache | Task 7 Step 3 |
| §5.1 cuBLAS col-major trick | Task 3 Step 4 |
| §5.2 codegen/cuda + emit-cu | Tasks 13, 14 |
| §5.3 tsc CLI extensions | Tasks 11, 14 |
| §5.4 CMake detection + WSL libcuda fix | Task 1 |
| §5.4 tsy_add_cuda_example | Task 15 |
| §5.5 matmul_tiny_cuda.tsy | Task 15 |
| §5.6 Shape matrix (5 matmul + 3×3 others) | Task 8 |
| §5.6 cli_cuda_adapter_run_matmul_tiny | Task 11 |
| §5.6 codegen_cuda_matmul_binary_runs | Task 15 |
| §5.6 codegen_emit_cu_contains_adapter_calls | Task 15 |
| §6 CLI surface | Tasks 11, 14 |
| §7 Risk: cuBLAS layout | Task 3 (2×2 sanity before bigger shapes) |
| §7 Risk: sm_86 unsupported | Task 1 (check_language fallback) |
| §7 Risk: WSL libcuda stub | Task 1 (RPATH block) |
| §7 Risk: FP32 tolerance overshoot | Task 8 Step 3 fallback |
| §8 Acceptance (25/25 ctest + run_cuda + manual inspect) | Task 17 |
| §9 Commit划分 (4 commits) | Tasks 4, 9, 12, 16 |

All spec requirements are wired to at least one task. No orphan requirements.
