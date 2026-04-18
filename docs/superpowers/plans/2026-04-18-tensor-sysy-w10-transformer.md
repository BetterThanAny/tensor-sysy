# W10 — Transformer Block E2E (Toy Single-Head) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a toy single-head transformer block (rmsnorm → attention → residual → rmsnorm → FFN(ReLU) → residual) as an end-to-end executable tensor-sysy fixture, with all three backends (native / cpu-adapter / cuda-adapter) validated against a numpy reference via pytest.

**Architecture:** Add two new HIR/LIR primitives (`@transpose`, `@relu`) through the full stack (HIR enum, verifier, LIR lowering, interpreter, cpu-adapter, cuda-adapter, codegen). Factor the duplicated `pickFirstTensorFunction` into a shared `module_utils` header. Write the block as a single `.tsy` function with 7 tensor params and one derived `out` buffer. Set up a `.venv` with numpy + pytest for the first external-reference e2e test.

**Tech Stack:** C++17, CUDA 12 / cuBLAS (reused from W8), CMake 3.18+, uv 0.11.6, Python 3.12 stdlib + numpy + pytest.

**Spec:** `docs/superpowers/specs/2026-04-18-tensor-sysy-w10-transformer-design.md`
**Expected final state:** ctest **32/32 green** on WSL + RTX 3080 (29 from W0-W9 + 3 new). Non-CUDA boxes: 29/29 (e2e pytest skips cuda-adapter parametrize; CLI smoke + transpose_relu_cases stay green).
**Starting HEAD:** commit `4b67d1f` (W9 complete).

---

## File Structure

### NEW files (under `/home/xs/tsy-wsl-export/tensor-sysy`)
- `src/lir/module_utils.h` — `pickFirstTensorFunction` declaration
- `src/lir/module_utils.cpp` — single authoritative definition
- `examples/transformer_block.tsy` — 8-matmul toy transformer fixture
- `tests/adapter/test_transpose_relu_cases.cpp` — C++ unit driver for the two new primitives
- `tests/e2e/__init__.py` — pytest package marker (empty)
- `tests/e2e/conftest.py` — `run_backend` + `parse_run_lir_output` helpers
- `tests/e2e/reference.py` — numpy forward reference mirroring `fillDeterministic`
- `tests/e2e/test_transformer_block.py` — pytest-parametrized 3-backend test
- `docs/superpowers/specs/2026-04-18-tensor-sysy-w10-transformer-design.md` (already exists)
- `docs/superpowers/plans/2026-04-18-tensor-sysy-w10-transformer.md` (this file)

### MODIFIED files
- `src/hir/ops.h` — add `Transpose`, `ReLU` to `OpKind`
- `src/hir/ops.cpp` — `toString` + `builtinKindFromName` cases
- `src/hir/verifier.cpp` — `verifyTranspose` + reuse `verifyUnary` for relu
- `src/lir/lowering.cpp` — add `Transpose` / `ReLU` to the supported-ops switch
- `src/lir/interpreter.cpp` — `kernelTranspose` + `kernelReLU` + executor dispatch; remove local `pickFirstTensorFunction`
- `src/runtime/adapter_cpu.h` + `adapter_cpu.cpp` — `adapterTranspose` + `adapterReLU` + executor dispatch; remove local `pickFirstTensorFunction`
- `src/runtime/adapter_cuda.h` + `adapter_cuda.cu` — `transposeKernel` + `reluKernel` + adapter wrappers + executor dispatch; remove local `pickFirstTensorFunction`
- `src/codegen/cpp.cpp` — `adapterSymbolFor` + remove local `pickFirstTensorFunction`
- `src/codegen/cuda.cpp` — `adapterSymbolFor` + remove local `pickFirstTensorFunction`
- `CMakeLists.txt` — `tsy_lir` gets `module_utils.cpp`; `tsy_add_example(transformer_block ...)` + `tsy_add_cuda_example(transformer_block_cuda ...)`
- `tests/CMakeLists.txt` — register 3 new ctest entries
- `PLAN.md` — append numpy + pytest rows to "项目期间安装的工具" table
- `.gitignore` — confirm `.venv/` is present (already is per W9 check; no action if so)

### UNCHANGED (but referenced for parallelism)
- `src/hir/lowering.cpp` — HIR's `emitBuiltinOp` already uses `builtinKindFromName`, so once the enum grows, no code change here
- `src/lir/ir.h` — Stmt.attrs from W9 still there; transpose/relu don't need attrs

---

## Commit Plan

Eight logical milestones per spec §9:

1. `refactor(lir): extract pickFirstTensorFunction to module_utils` (Tasks 1–2)
2. `feat(hir): add @transpose and @relu ops (enum + verifier + lowering)` (Tasks 3–6)
3. `feat(lir): implement transpose/relu in interpreter` (Tasks 7–8)
4. `feat(runtime): transpose + relu for cpu-adapter and cuda-adapter` (Tasks 9–12)
5. `feat(codegen): emit adapterTranspose + adapterReLU in emit-cpp and emit-cu` (Tasks 13–14)
6. `feat(examples): transformer_block.tsy + CMake example targets + unit ctest + CLI smoke ctest` (Tasks 15–18)
7. `test(e2e): numpy reference + pytest + uv venv + PLAN.md tools table` (Tasks 19–22)
8. `docs: W10 spec + implementation plan` (Task 23 — final verify + commit spec/plan)

---

## Task 1: Factor `pickFirstTensorFunction` into `src/lir/module_utils.{h,cpp}`

**Goal:** One authoritative definition; 5 files stop defining their own copy. This is pure refactoring — no behaviour change. Verifies no regression on the full 29-test suite.

**Files:**
- Create: `src/lir/module_utils.h`
- Create: `src/lir/module_utils.cpp`
- Modify: `src/lir/interpreter.cpp`, `src/runtime/adapter_cpu.cpp`, `src/runtime/adapter_cuda.cu`, `src/codegen/cpp.cpp`, `src/codegen/cuda.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `src/lir/module_utils.h`**

```cpp
#pragma once

#include "ir.h"

namespace tsy::lir {

// Pick the first "interesting" function to execute / codegen.
// Skips `main`, prefers functions with non-empty tensor params, falls back to
// module.funcs.front() so run-lir always has something to do. Previously
// duplicated verbatim across interpreter, adapter_cpu, adapter_cuda, and
// both codegen modules; this is now the single authoritative copy.
const Function* pickFirstTensorFunction(const Module& m);

}  // namespace tsy::lir
```

- [ ] **Step 2: Create `src/lir/module_utils.cpp`**

```cpp
#include "module_utils.h"

namespace tsy::lir {

const Function* pickFirstTensorFunction(const Module& m) {
    for (const auto& f : m.funcs) {
        if (f->name == "main") continue;
        if (!f->params.empty()) return f.get();
    }
    return m.funcs.empty() ? nullptr : m.funcs.front().get();
}

}  // namespace tsy::lir
```

- [ ] **Step 3: Add `module_utils.cpp` to `tsy_lir` in `CMakeLists.txt`**

Find the `tsy_lir` library block. Currently:
```cmake
add_library(tsy_lir STATIC
    src/lir/printer.cpp
    src/lir/lowering.cpp
    src/lir/interpreter.cpp
)
```
Change to:
```cmake
add_library(tsy_lir STATIC
    src/lir/printer.cpp
    src/lir/lowering.cpp
    src/lir/interpreter.cpp
    src/lir/module_utils.cpp
)
```

- [ ] **Step 4: Remove `pickFirstTensorFunction` from 5 files, add include**

For EACH of these 5 files, (a) delete the local `pickFirstTensorFunction` definition and (b) add `#include "module_utils.h"` (use the correct relative path based on each file's location):

- `src/lir/interpreter.cpp` — delete the local `pickFirstTensorFunction` function (currently lines ~144-153 inside the anon namespace). Add `#include "module_utils.h"` near the other `#include`s at the top. Replace the call site `pickFirstTensorFunction(m)` with `tsy::lir::pickFirstTensorFunction(m)` if it resolves ambiguously; if the file is already inside `namespace tsy::lir`, the bare call resolves correctly.
  - Important: since `runFirstTensorFunction` currently calls `pickFirstTensorFunction(m)` from inside `namespace tsy::lir`, and our new header puts the function in the same namespace, the bare name resolves — no rename needed.

- `src/runtime/adapter_cpu.cpp` — delete the local `pickFirstTensorFunction` inside the anon namespace. Add `#include "../lir/module_utils.h"` near the other includes. The call site (`pickFirstTensorFunction(m)` inside `runWithCpuAdapter`) is inside `namespace tsy::runtime`, so change to `tsy::lir::pickFirstTensorFunction(m)`.

- `src/runtime/adapter_cuda.cu` — same pattern as adapter_cpu.cpp. Add `#include "../lir/module_utils.h"`. Update the call site to `tsy::lir::pickFirstTensorFunction(m)`.

- `src/codegen/cpp.cpp` — delete the local `pickFirstTensorFunction` inside the anon namespace. Add `#include "../lir/module_utils.h"`. The call site is inside `namespace tsy::codegen` so change to `tsy::lir::pickFirstTensorFunction(m)`.

- `src/codegen/cuda.cpp` — same as codegen/cpp.cpp.

- [ ] **Step 5: Configure + build + full regression test**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake -S . -B build 2>&1 | tail -5
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: clean build (no duplicate-symbol linker errors, no missing-symbol errors), `100% tests passed, 0 tests failed out of 29`.

If a linker error `multiple definition of ... pickFirstTensorFunction` appears, one of the 5 files still has a local copy — grep it down:
```bash
grep -rn "pickFirstTensorFunction" src/ | grep -v "module_utils"
# Expected: only call-site lines, no function definitions
```

---

## Task 2: Commit 1

- [ ] **Step 1: Commit**

```bash
git add src/lir/module_utils.h src/lir/module_utils.cpp CMakeLists.txt \
        src/lir/interpreter.cpp src/runtime/adapter_cpu.cpp \
        src/runtime/adapter_cuda.cu src/codegen/cpp.cpp src/codegen/cuda.cpp
git commit -m "$(cat <<'EOF'
refactor(lir): extract pickFirstTensorFunction to module_utils

Previously duplicated verbatim across 5 translation units (interpreter,
adapter_cpu, adapter_cuda, codegen/cpp, codegen/cuda). W8 and W9 code
reviewers both flagged this as a pre-W10 debt item. Consolidated into
src/lir/module_utils.{h,cpp} inside tsy::lir namespace; callers now
include the header and qualify the name.

Zero behavioural change — all 29 existing tests remain green.
EOF
)"
git log --oneline -1
```

---

## Task 3: Extend `OpKind` enum + name mapping

**Goal:** HIR knows two more ops. Nothing actually uses them yet; this is table-extension only.

**Files:**
- Modify: `src/hir/ops.h`
- Modify: `src/hir/ops.cpp`

- [ ] **Step 1: Extend `OpKind` in `src/hir/ops.h`**

Find `enum class OpKind { ... }`. Add `Transpose` and `ReLU` before `FuncCall`:

```cpp
enum class OpKind {
    Param,
    MatMul,
    Add,
    Softmax,
    RMSNorm,
    View,       // W1 syntax not yet exposed; reserved.
    Permute,    // ditto.
    Transpose,  // W10: 2-D transpose, result shape = [in.dim1, in.dim0].
    ReLU,       // W10: elementwise relu, result shape = in.shape.
    FuncCall,
    Return,
    Unknown,
};
```

- [ ] **Step 2: Update `toString` + `builtinKindFromName` in `src/hir/ops.cpp`**

Full replacement:

```cpp
#include "ops.h"

namespace tsy::hir {

const char* toString(OpKind k) {
    switch (k) {
        case OpKind::Param:     return "param";
        case OpKind::MatMul:    return "matmul";
        case OpKind::Add:       return "add";
        case OpKind::Softmax:   return "softmax";
        case OpKind::RMSNorm:   return "rmsnorm";
        case OpKind::View:      return "view";
        case OpKind::Permute:   return "permute";
        case OpKind::Transpose: return "transpose";
        case OpKind::ReLU:      return "relu";
        case OpKind::FuncCall:  return "call";
        case OpKind::Return:    return "return";
        case OpKind::Unknown:   return "unknown";
    }
    return "?";
}

OpKind builtinKindFromName(const std::string& name) {
    if (name == "matmul")    return OpKind::MatMul;
    if (name == "add")       return OpKind::Add;
    if (name == "softmax")   return OpKind::Softmax;
    if (name == "rmsnorm")   return OpKind::RMSNorm;
    if (name == "view")      return OpKind::View;
    if (name == "permute")   return OpKind::Permute;
    if (name == "transpose") return OpKind::Transpose;
    if (name == "relu")      return OpKind::ReLU;
    return OpKind::Unknown;
}

}  // namespace tsy::hir
```

- [ ] **Step 3: Build to verify enum extension compiles**

```bash
cmake --build build -j --target tsy_hir 2>&1 | tail -5
```

Expected: clean build. No test breaks (existing verifier's `switch (op.kind)` just defaults-through the two new enum values; the LIR lowering switch defaults-errors them but nothing emits those ops yet).

Verify full suite still passes:
```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```
Expected: 29/29 green.

---

## Task 4: HIR verifier rules for transpose and relu

**Goal:** `@transpose(X)` must reject non-2D and must pin result shape. `@relu(X)` reuses the existing `verifyUnary` (same contract as softmax/rmsnorm).

**Files:**
- Modify: `src/hir/verifier.cpp`

- [ ] **Step 1: Add `verifyTranspose` and wire into the verifier switch**

In `src/hir/verifier.cpp`, inside the anonymous namespace where `verifyMatMul` / `verifyAdd` / `verifyUnary` live, add a new function right after `verifyUnary`:

```cpp
void verifyTranspose(const Op& op, DiagnosticEngine& diag) {
    if (op.operands.size() != 1) {
        std::ostringstream oss;
        oss << "transpose expects 1 operand, got " << op.operands.size();
        diag.error(op.loc, oss.str());
        return;
    }
    const auto& a = *op.operands[0];
    if (!requireTensor(a, diag, op.loc, "input", "transpose")) return;
    if (!requireResolved(a, diag, op.loc, "transpose")) return;

    if (a.type.shape.rank() != 2) {
        std::ostringstream oss;
        oss << "transpose expects a 2-D tensor, got rank "
            << a.type.shape.rank();
        diag.error(op.loc, oss.str());
        return;
    }
    if (op.results.size() != 1) {
        diag.error(op.loc, "transpose must produce exactly 1 result");
        return;
    }
    const auto& r = *op.results[0];
    if (!requireTensor(r, diag, op.loc, "result", "transpose")) return;
    if (!requireResolved(r, diag, op.loc, "transpose result")) return;

    Shape expected;
    expected.dims.push_back(a.type.shape.dims[1]);
    expected.dims.push_back(a.type.shape.dims[0]);
    if (!shapesEqual(r.type.shape, expected)) {
        std::ostringstream oss;
        oss << "transpose result shape " << shapeAsString(r.type.shape)
            << " does not match expected " << shapeAsString(expected);
        diag.error(op.loc, oss.str());
    }
}
```

- [ ] **Step 2: Wire `Transpose` and `ReLU` into the verifier dispatch**

Still in `src/hir/verifier.cpp`, find `verifyModule` with its `switch (op.kind)` block. Add two new cases:

```cpp
case OpKind::Softmax:   verifyUnary(op, diag, "softmax"); break;
case OpKind::RMSNorm:   verifyUnary(op, diag, "rmsnorm"); break;
case OpKind::Transpose: verifyTranspose(op, diag);        break;
case OpKind::ReLU:      verifyUnary(op, diag, "relu");    break;
case OpKind::Unknown:   verifyUnknown(op, diag);          break;
```

- [ ] **Step 3: Build + quick test**

```bash
cmake --build build -j --target tsy_hir test_shape_cases 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: 29/29 green (existing shape tests untouched; new verifier rules have no fixtures yet — that arrives in Task 17).

---

## Task 5: LIR lowering for Transpose and ReLU

**Goal:** The HIR→LIR switch that today errors on Transpose/ReLU now lowers them like the existing four builtins.

**Files:**
- Modify: `src/lir/lowering.cpp`

- [ ] **Step 1: Extend the supported-ops switch in `src/lir/lowering.cpp`**

Find the per-op lowering switch (around lines 73-88). Add `Transpose` and `ReLU` to the list of handled cases:

```cpp
switch (op.kind) {
    case hir::OpKind::MatMul:
    case hir::OpKind::Add:
    case hir::OpKind::Softmax:
    case hir::OpKind::RMSNorm:
    case hir::OpKind::Transpose:
    case hir::OpKind::ReLU:
        lowerBuiltinCall(*lf, vm, op);
        break;
    default: {
        std::ostringstream oss;
        oss << "LIR lowering skipped " << toString(op.kind)
            << " op (not yet supported in W4)";
        diag.error(op.loc, oss.str());
        break;
    }
}
```

(`lowerBuiltinCall` already calls `toString(op.kind)` to produce the primitive string, which our new enum entries now handle as "transpose" / "relu".)

- [ ] **Step 2: Build + regression**

```bash
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: 29/29 green (no test exercises transpose/relu yet — their LIR still has no backend impl; this is fine because no `.tsy` uses these builtins yet).

---

## Task 6: Commit 2

- [ ] **Step 1: Commit**

```bash
git add src/hir/ops.h src/hir/ops.cpp src/hir/verifier.cpp src/lir/lowering.cpp
git commit -m "$(cat <<'EOF'
feat(hir): add @transpose and @relu ops (enum + verifier + lowering)

Extends OpKind with Transpose and ReLU, wires builtinKindFromName,
toString, verifier dispatch (transpose validates 2-D input + swapped
result shape; relu reuses verifyUnary), and LIR lowering switch.

No backend implementation yet — Task 7+ add interpreter/adapter/codegen
support. Existing 29 tests remain green because no .tsy fixture uses
these builtins.
EOF
)"
git log --oneline -1
```

---

## Task 7: Interpreter kernels for transpose and relu

**Goal:** The native backend can execute `@transpose` and `@relu`.

**Files:**
- Modify: `src/lir/interpreter.cpp`

- [ ] **Step 1: Add `kernelTranspose` and `kernelReLU` + executor dispatch**

In `src/lir/interpreter.cpp`, add two new kernels inside the anon namespace, right after `kernelRMSNorm`:

```cpp
// 2-D transpose: y[j, i] = x[i, j]. Caller ensures x is rank 2 (verifier's job).
void kernelTranspose(const NamedTensor& x, NamedTensor& y) {
    int64_t M = x.dims[0];
    int64_t N = x.dims[1];
    y.data.assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            y.data[j * M + i] = x.data[i * N + j];
        }
    }
}

// Elementwise ReLU: y[i] = max(0, x[i]). Shape preserved.
void kernelReLU(const NamedTensor& x, NamedTensor& y) {
    const int64_t n = static_cast<int64_t>(x.data.size());
    y.data.assign(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        float v = x.data[i];
        y.data[i] = v < 0.0f ? 0.0f : v;
    }
}
```

In the same file, locate the `runFunctionImpl`'s primitive-dispatch if/else chain (around lines 120-135). Insert two new branches between the rmsnorm branch and the else clause:

```cpp
} else if (s.primitive == "rmsnorm") {
    if (s.operand_bufs.size() != 1) { diag.error(s.loc, "rmsnorm: expected 1 operand"); r.ok = false; continue; }
    kernelRMSNorm(r.buffers[s.operand_bufs[0]], out);
} else if (s.primitive == "transpose") {
    if (s.operand_bufs.size() != 1) { diag.error(s.loc, "transpose: expected 1 operand"); r.ok = false; continue; }
    kernelTranspose(r.buffers[s.operand_bufs[0]], out);
} else if (s.primitive == "relu") {
    if (s.operand_bufs.size() != 1) { diag.error(s.loc, "relu: expected 1 operand"); r.ok = false; continue; }
    kernelReLU(r.buffers[s.operand_bufs[0]], out);
} else {
    diag.error(s.loc, "interpreter: unknown primitive '" + s.primitive + "'");
    r.ok = false;
}
```

- [ ] **Step 2: Build + regression**

```bash
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: 29/29 green (new kernels compile; nothing exercises them yet — Task 17 adds unit tests and Task 18 adds the transformer fixture).

---

## Task 8: Commit 3

- [ ] **Step 1: Commit**

```bash
git add src/lir/interpreter.cpp
git commit -m "$(cat <<'EOF'
feat(lir): implement transpose/relu in interpreter

kernelTranspose: nested i/j loop over a 2-D input, y[j*M+i] = x[i*N+j].
kernelReLU: flat max(0, x[i]) elementwise.

Plugs into the existing primitive-dispatch chain in runFunctionImpl
alongside matmul/add/softmax/rmsnorm. 29/29 tests still green — unit
tests for the new kernels arrive with the adapter layer (Task 11 + 17).
EOF
)"
git log --oneline -1
```

---

## Task 9: CPU adapter — adapterTranspose + adapterReLU

**Goal:** cpu-adapter backend supports the two new primitives. This is the second of three backends.

**Files:**
- Modify: `src/runtime/adapter_cpu.h`
- Modify: `src/runtime/adapter_cpu.cpp`

- [ ] **Step 1: Declare new adapters in `src/runtime/adapter_cpu.h`**

Find the existing `adapterRMSNorm` declaration. Add two new declarations right below it, inside `namespace tsy::runtime`:

```cpp
// 2-D transpose. x:[M,N] → c:[N,M]. Used by W10 attention for Q @ K^T.
void adapterTranspose(const Tensor& x, Tensor& c);

// Elementwise ReLU. Shape preserved. Used by W10 FFN activation.
void adapterReLU(const Tensor& x, Tensor& c);
```

- [ ] **Step 2: Implement them in `src/runtime/adapter_cpu.cpp`**

Just after the `adapterRMSNorm` function definition, add:

```cpp
void adapterTranspose(const Tensor& x, Tensor& c) {
    assert(x.dims.size() == 2);
    const int64_t M = x.dims[0];
    const int64_t N = x.dims[1];
    c.data.assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            c.data[j * M + i] = x.data[i * N + j];
        }
    }
}

void adapterReLU(const Tensor& x, Tensor& c) {
    const size_t n = x.data.size();
    c.data.assign(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float v = x.data[i];
        c.data[i] = v < 0.0f ? 0.0f : v;
    }
}
```

- [ ] **Step 3: Wire into the executor's primitive dispatch**

Still in `src/runtime/adapter_cpu.cpp`, find `runFunctionAdapter`'s primitive if/else chain. Right after the `else if (s.primitive == "rmsnorm")` branch, insert:

```cpp
} else if (s.primitive == "transpose") {
    if (s.operand_bufs.size() != 1) {
        diag.error(s.loc, "cpu-adapter transpose: expected 1 operand");
        r.ok = false; continue;
    }
    adapterTranspose(r.buffers[s.operand_bufs[0]], out);
} else if (s.primitive == "relu") {
    if (s.operand_bufs.size() != 1) {
        diag.error(s.loc, "cpu-adapter relu: expected 1 operand");
        r.ok = false; continue;
    }
    adapterReLU(r.buffers[s.operand_bufs[0]], out);
```

- [ ] **Step 4: Build + regression**

```bash
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: 29/29 green.

---

## Task 10: CUDA adapter — transposeKernel + reluKernel + wrappers

**Goal:** cuda-adapter backend supports the two new primitives. Mirror the cpu-adapter API.

**Files:**
- Modify: `src/runtime/adapter_cuda.h`
- Modify: `src/runtime/adapter_cuda.cu`

- [ ] **Step 1: Declare new adapters in `src/runtime/adapter_cuda.h`**

Just after the existing `adapterRMSNormCuda` declaration, add:

```cpp
// 2-D transpose on GPU. x:[M,N] → c:[N,M]. Naive one-thread-per-element
// kernel, block(32,32). Used by W10 attention for Q @ K^T.
void adapterTransposeCuda(const Tensor& x, Tensor& c);

// Elementwise ReLU on GPU. Shape preserved. Flat block=256 launch.
void adapterReLUCuda(const Tensor& x, Tensor& c);
```

- [ ] **Step 2: Implement kernels and wrappers in `src/runtime/adapter_cuda.cu`**

Add the kernels BEFORE the first adapter wrapper (anywhere after the `CUDA_CHECK` macros at the top of the file is fine — follow the existing pattern where `addKernel`, `softmaxRowKernel`, etc., are at file scope):

```cpp
__global__ void transposeKernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row of x
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col of x
    if (i >= M || j >= N) return;
    y[j * M + i] = x[i * N + j];
}

__global__ void reluKernel(const float* __restrict__ a,
                            float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = fmaxf(0.0f, a[i]);
}
```

Just after the existing `adapterRMSNormCuda` definition, add:

```cpp
void adapterTransposeCuda(const Tensor& x, Tensor& c) {
    assert(x.dims.size() == 2);
    const int M = static_cast<int>(x.dims[0]);
    const int N = static_cast<int>(x.dims[1]);
    const size_t bytes = static_cast<size_t>(M) * N * sizeof(float);

    float *dX = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dY, bytes));
    CUDA_CHECK(cudaMemcpy(dX, x.data.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    transposeKernel<<<grid, block>>>(dX, dY, M, N);
    CUDA_CHECK(cudaGetLastError());

    c.data.assign(static_cast<size_t>(M) * N, 0.0f);
    CUDA_CHECK(cudaMemcpy(c.data.data(), dY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
}

void adapterReLUCuda(const Tensor& x, Tensor& c) {
    const int n = static_cast<int>(x.data.size());
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *dA = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, x.data.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid  = (n + block - 1) / block;
    reluKernel<<<grid, block>>>(dA, dC, n);
    CUDA_CHECK(cudaGetLastError());

    c.data.assign(static_cast<size_t>(n), 0.0f);
    CUDA_CHECK(cudaMemcpy(c.data.data(), dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dC));
}
```

- [ ] **Step 3: Wire into the executor's primitive dispatch**

Still in `src/runtime/adapter_cuda.cu`, find `runFunctionCudaAdapter`'s primitive dispatch. Add two branches after rmsnorm:

```cpp
} else if (s.primitive == "transpose") {
    if (s.operand_bufs.size() != 1) {
        diag.error(s.loc, "cuda-adapter transpose: expected 1 operand");
        r.ok = false; continue;
    }
    adapterTransposeCuda(r.buffers[s.operand_bufs[0]], out);
} else if (s.primitive == "relu") {
    if (s.operand_bufs.size() != 1) {
        diag.error(s.loc, "cuda-adapter relu: expected 1 operand");
        r.ok = false; continue;
    }
    adapterReLUCuda(r.buffers[s.operand_bufs[0]], out);
```

- [ ] **Step 4: Build + regression**

```bash
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: 29/29 green.

---

## Task 11: Commit 4

- [ ] **Step 1: Commit**

```bash
git add src/runtime/adapter_cpu.h src/runtime/adapter_cpu.cpp \
        src/runtime/adapter_cuda.h src/runtime/adapter_cuda.cu
git commit -m "$(cat <<'EOF'
feat(runtime): transpose + relu for cpu-adapter and cuda-adapter

CPU: naive 2-D loop transpose; elementwise max(0, x) relu.
CUDA: transposeKernel launches block(32,32) with grid((N+31)/32, (M+31)/32);
reluKernel launches block=256 flat. Both follow the W8 pattern:
cudaMalloc → memcpy H2D → launch → memcpy D2H → CUDA_CHECK(cudaFree).

Executors in both adapters now dispatch "transpose" / "relu" primitives
alongside the original four. All 29 pre-W10 tests remain green; unit
coverage arrives with Task 17's test_transpose_relu_cases.cpp.
EOF
)"
git log --oneline -1
```

---

## Task 12: Codegen emits adapterTranspose + adapterReLU

**Goal:** Generated `.cpp` and `.cu` sources know how to call the new adapters.

**Files:**
- Modify: `src/codegen/cpp.cpp`
- Modify: `src/codegen/cuda.cpp`

- [ ] **Step 1: Extend `adapterSymbolFor` in `src/codegen/cpp.cpp`**

Find the `adapterSymbolFor` function (an anon-namespace helper). Add two new clauses:

```cpp
const char* adapterSymbolFor(const std::string& primitive) {
    if (primitive == "matmul")    return "adapterMatMul";
    if (primitive == "add")       return "adapterAdd";
    if (primitive == "softmax")   return "adapterSoftmax";
    if (primitive == "rmsnorm")   return "adapterRMSNorm";
    if (primitive == "transpose") return "adapterTranspose";
    if (primitive == "relu")      return "adapterReLU";
    return nullptr;
}
```

- [ ] **Step 2: Extend `adapterSymbolFor` in `src/codegen/cuda.cpp`**

Same shape, but with `Cuda` suffix on the new names:

```cpp
const char* adapterSymbolFor(const std::string& primitive) {
    if (primitive == "matmul")    return "adapterMatMulCuda";
    if (primitive == "add")       return "adapterAddCuda";
    if (primitive == "softmax")   return "adapterSoftmaxCuda";
    if (primitive == "rmsnorm")   return "adapterRMSNormCuda";
    if (primitive == "transpose") return "adapterTransposeCuda";
    if (primitive == "relu")      return "adapterReLUCuda";
    return nullptr;
}
```

- [ ] **Step 3: Build + regression + manual sanity**

```bash
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: 29/29 green. No existing fixture uses transpose/relu so codegen output is byte-identical for all current `.tsy` files.

Manual sanity (optional — just reassures nothing regressed on existing emit paths):
```bash
./build/tsc emit-cpp examples/mlp.tsy | grep adapterMat | head -3
./build/tsc emit-cu examples/matmul_tiny_cuda.tsy | grep adapterMat | head -3
```

Expected: same output as before (no new adapter symbols appear because the fixtures don't call them).

---

## Task 13: Commit 5

- [ ] **Step 1: Commit**

```bash
git add src/codegen/cpp.cpp src/codegen/cuda.cpp
git commit -m "$(cat <<'EOF'
feat(codegen): emit adapterTranspose + adapterReLU in emit-cpp and emit-cu

adapterSymbolFor in both codegen files now maps "transpose"→adapterTranspose[Cuda]
and "relu"→adapterReLU[Cuda]. No other codegen logic changes; the call
emission loop already handles 1-operand + 1-result primitives (same shape
as softmax/rmsnorm), so the new builtins flow through without special
cases.

W8/W9 existing fixtures don't use transpose/relu, so generated output
for them is byte-identical — regression-safe.
EOF
)"
git log --oneline -1
```

---

## Task 14: `examples/transformer_block.tsy` + CMake targets

**Goal:** Ship the .tsy fixture and wire it into the build so `build/out/transformer_block` and `build/out/transformer_block_cuda` get generated + compiled automatically.

**Files:**
- Create: `examples/transformer_block.tsy`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `examples/transformer_block.tsy`**

```
// W10 fixture: toy single-head transformer block (no MHA / mask / RoPE).
//   x → rmsnorm → Q/K/V projs → Q @ Kᵀ → softmax → @V → out_proj → residual
//   x1 → rmsnorm → fc1 → ReLU → fc2 → residual → out
//
// Shapes chosen small enough for hand-verifiable unit tests while still
// exercising every new W10 primitive. Dot-product attention (not scaled)
// — the numpy reference in tests/e2e/reference.py matches the omission.
const int S = 4;   // sequence length
const int D = 8;   // model dim
const int F = 16;  // ffn hidden dim

void transformer_block(
    tensor<f32>[S, D] x,
    tensor<f32>[D, D] Wq,
    tensor<f32>[D, D] Wk,
    tensor<f32>[D, D] Wv,
    tensor<f32>[D, D] Wo,
    tensor<f32>[D, F] W1,
    tensor<f32>[F, D] W2
) {
    tensor<f32>[S, D] x_n   = @rmsnorm(x);
    tensor<f32>[S, D] Q     = @matmul(x_n, Wq);
    tensor<f32>[S, D] K     = @matmul(x_n, Wk);
    tensor<f32>[S, D] V     = @matmul(x_n, Wv);
    tensor<f32>[D, S] Kt    = @transpose(K);
    tensor<f32>[S, S] attn_scores = @matmul(Q, Kt);
    tensor<f32>[S, S] attn  = @softmax(attn_scores);
    tensor<f32>[S, D] ctx   = @matmul(attn, V);
    tensor<f32>[S, D] a_out = @matmul(ctx, Wo);
    tensor<f32>[S, D] x1    = @add(x, a_out);

    tensor<f32>[S, D] x1_n  = @rmsnorm(x1);
    tensor<f32>[S, F] h     = @matmul(x1_n, W1);
    tensor<f32>[S, F] h1    = @relu(h);
    tensor<f32>[S, D] f_out = @matmul(h1, W2);
    tensor<f32>[S, D] out   = @add(x1, f_out);
    return;
}

int main() { return 0; }
```

- [ ] **Step 2: Register two CMake example targets**

Find the existing `tsy_add_example(mlp examples/mlp.tsy)` line in `CMakeLists.txt`. Just after it (and still outside the CUDA block, since `tsy_add_example` itself is CUDA-agnostic), add:

```cmake
tsy_add_example(transformer_block examples/transformer_block.tsy)
```

Then find the `tsy_add_cuda_example(matmul_cuda_demo ...)` call inside the CUDA-gated block. Right after it (and before that block's `endif()`), add:

```cmake
    tsy_add_cuda_example(transformer_block_cuda examples/transformer_block.tsy)
```

- [ ] **Step 3: Configure + build — transformer_block target must build successfully**

```bash
cmake -S . -B build 2>&1 | tail -5
cmake --build build -j 2>&1 | tail -10
```

Expected: additional lines `[ X%] tsc emit-cpp examples/transformer_block.tsy -> gen/transformer_block.cpp`, `Built target transformer_block`, and (on CUDA box) `Built target transformer_block_cuda`.

- [ ] **Step 4: Manual sanity — run the produced binaries**

```bash
./build/out/transformer_block | tail -8
./build/out/transformer_block_cuda | tail -8   # only if CUDA available
```

Expected last line group for both:
```
  local out shape=[4,8]:
      <32 floats in 4 rows of 8>
```

Both should print 32 finite float values. Exact numbers are determined by `fillDeterministic` and the forward computation; we'll validate them against numpy in Task 17.

---

## Task 15: C++ unit test — `test_transpose_relu_cases.cpp`

**Goal:** Lock in the CPU-adapter transpose/relu primitives against hand-computed expected values. This runs on every ctest pass, gated only on tsy_runtime_cpu (not CUDA).

**Files:**
- Create: `tests/adapter/test_transpose_relu_cases.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Create the test driver**

```cpp
// W10 unit tests for the two new primitives added to the CPU adapter.
// Follows the same "g_failures + fail() + tiny helpers" pattern as
// test_adapter_cpu_cases.cpp and test_adapter_cuda_cases.cpp.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/runtime/adapter_cpu.h"

using tsy::lir::NamedTensor;
using tsy::runtime::adapterReLU;
using tsy::runtime::adapterTranspose;

namespace {

int g_failures = 0;

void fail(const std::string& label, const std::string& why) {
    std::cerr << "FAIL[" << label << "]: " << why << "\n";
    ++g_failures;
}

bool approx(float a, float b, float atol = 1e-6f) {
    return std::fabs(a - b) <= atol;
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

NamedTensor zeros(const std::string& name,
                  const std::vector<int64_t>& dims) {
    NamedTensor t;
    t.name = name;
    t.dims = dims;
    int64_t n = 1;
    for (auto d : dims) n *= d;
    t.data.assign(n, 0.0f);
    return t;
}

void checkVec(const std::string& label, const NamedTensor& t,
              const std::vector<float>& expected) {
    if (t.data.size() != expected.size()) {
        fail(label, "size mismatch: got " + std::to_string(t.data.size()) +
                     ", expected " + std::to_string(expected.size()));
        return;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!approx(t.data[i], expected[i])) {
            std::ostringstream oss;
            oss << "idx=" << i << " got " << t.data[i]
                << " expected " << expected[i];
            fail(label, oss.str());
            return;
        }
    }
}

void testTranspose2x2() {
    auto x = makeTensor("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto y = zeros("y", {2, 2});
    adapterTranspose(x, y);
    checkVec("transpose/2x2", y, {1.0f, 3.0f, 2.0f, 4.0f});
}

void testTranspose2x3() {
    // input  shape [2, 3]:  [[1, 2, 3], [4, 5, 6]]
    // output shape [3, 2]:  [[1, 4], [2, 5], [3, 6]]
    auto x = makeTensor("x", {2, 3}, {1, 2, 3, 4, 5, 6});
    auto y = zeros("y", {3, 2});
    adapterTranspose(x, y);
    checkVec("transpose/2x3", y, {1, 4, 2, 5, 3, 6});
}

void testReluFlat() {
    auto x = makeTensor("x", {4}, {-1.0f, 0.0f, 1.0f, 2.0f});
    auto y = zeros("y", {4});
    adapterReLU(x, y);
    checkVec("relu/flat", y, {0.0f, 0.0f, 1.0f, 2.0f});
}

void testRelu2D() {
    // -0.5 → 0, 0.5 stays, etc.
    auto x = makeTensor("x", {2, 3},
                        {-0.5f, 0.0f, 0.5f, 1.0f, -1.0f, 2.0f});
    auto y = zeros("y", {2, 3});
    adapterReLU(x, y);
    checkVec("relu/2x3", y, {0.0f, 0.0f, 0.5f, 1.0f, 0.0f, 2.0f});
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    testTranspose2x2();
    testTranspose2x3();
    testReluFlat();
    testRelu2D();
    if (g_failures == 0) {
        std::cout << "transpose_relu_cases: ALL PASS\n";
        return 0;
    }
    std::cerr << "transpose_relu_cases: " << g_failures << " failure(s)\n";
    return 1;
}
```

- [ ] **Step 2: Register ctest in `tests/CMakeLists.txt`**

Append (inside the existing `if(TARGET tsy_runtime_cpu)` block, near the other adapter_cpu tests):

```cmake
    add_executable(test_transpose_relu_cases adapter/test_transpose_relu_cases.cpp)
    target_link_libraries(test_transpose_relu_cases PRIVATE tsy_runtime_cpu)
    add_test(NAME transpose_relu_cases
        COMMAND test_transpose_relu_cases
    )
```

- [ ] **Step 3: Configure + run**

```bash
cmake -S . -B build 2>&1 | tail -5
cmake --build build -j --target test_transpose_relu_cases 2>&1 | tail -5
ctest --test-dir build -R transpose_relu_cases --output-on-failure
```

Expected:
```
transpose_relu_cases: ALL PASS
...
100% tests passed, 0 tests failed out of 1
```

Full suite check:
```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: `100% tests passed, 0 tests failed out of 30`.

---

## Task 16: CLI smoke ctest — `cli_run_transformer_block_native`

**Goal:** Regression-guard the W10 fixture: if any future change breaks `tsc run-lir examples/transformer_block.tsy`, this test fails.

**Files:**
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Append ctest (outside CUDA gate — this uses native backend only)**

Append to the end of `tests/CMakeLists.txt`:

```cmake
# W10: transformer_block native run — must print "local out shape=[4,8]".
add_test(NAME cli_run_transformer_block_native
    COMMAND tsc run-lir ${CMAKE_SOURCE_DIR}/examples/transformer_block.tsy
)
set_tests_properties(cli_run_transformer_block_native PROPERTIES
    PASS_REGULAR_EXPRESSION "local out shape=\\[4,8\\]"
)
```

- [ ] **Step 2: Run**

```bash
cmake -S . -B build
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build -R cli_run_transformer_block_native --output-on-failure
```

Expected: passes (regex finds `local out shape=[4,8]` in stdout).

Full suite: `ctest --test-dir build --output-on-failure 2>&1 | tail -5` → `31/31 green`.

---

## Task 17: Commit 6

- [ ] **Step 1: Commit**

```bash
git add examples/transformer_block.tsy \
        tests/adapter/test_transpose_relu_cases.cpp \
        CMakeLists.txt tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(examples): transformer_block.tsy + CMake targets + unit + CLI ctests

examples/transformer_block.tsy is the first multi-op fixture: toy
single-head attention (rmsnorm → Q/K/V → Q@Kᵀ → softmax → A@V → Wo)
with residual, plus FFN (rmsnorm → fc1 → ReLU → fc2) with residual.
Shapes S=4, D=8, F=16 — small enough to hand-verify, large enough to
exercise every W10 primitive.

tsy_add_example / tsy_add_cuda_example wire it into build/out. New
unit ctest transpose_relu_cases locks transpose/relu CPU adapter
against hand-computed expected values. New CLI ctest
cli_run_transformer_block_native regression-guards the end-to-end
native run. Total ctest count: 31/31.
EOF
)"
git log --oneline -1
```

---

## Task 18: Set up Python venv + install numpy + pytest

**Goal:** Create a project-local `.venv` with minimum Python deps for the e2e test. Document in PLAN.md.

**Files:**
- Create: `.venv/` (via uv)
- Modify: `PLAN.md` (append tools table rows)

- [ ] **Step 1: Confirm `.venv/` is in `.gitignore`**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
grep -c "^.venv/" .gitignore
```

Expected: `1`. If `0`, add it:
```bash
echo ".venv/" >> .gitignore
```

- [ ] **Step 2: Create venv + install deps**

```bash
uv venv .venv 2>&1 | tail -5
uv pip install --python .venv/bin/python numpy pytest 2>&1 | tail -10
```

If the install times out or fails on network, prefix with `proxy_on &&`:
```bash
proxy_on && uv pip install --python .venv/bin/python numpy pytest
```

Expected final line from `uv pip install`: `Successfully installed ... numpy-X pytest-Y ...`.

Verify:
```bash
.venv/bin/python -c "import numpy, pytest; print('numpy', numpy.__version__, 'pytest', pytest.__version__)"
```

Expected: `numpy 1.XX.X pytest 7.X.X` or newer.

- [ ] **Step 3: Append to `PLAN.md` "项目期间安装的工具" table**

Find the existing table in `PLAN.md` (last section). Append two rows after the existing `cmake` row:

```
| numpy >=1.24 | `uv pip install --python .venv/bin/python numpy` | 2026-04-18 W10 | e2e transformer_block 的 numpy 参考实现 | `rm -rf .venv` |
| pytest >=7 | `uv pip install --python .venv/bin/python pytest` | 2026-04-18 W10 | e2e 测试驱动（ctest 通过 python3 -m pytest 调用） | `rm -rf .venv` |
```

- [ ] **Step 4: Sanity — `.venv` is not staged**

```bash
git status --short | grep -c "^.. .venv"
```

Expected: `0` (directory ignored by .gitignore).

---

## Task 19: Create `tests/e2e/` Python package — reference + conftest + test

**Goal:** The three Python files that drive pytest comparison against numpy reference.

**Files:**
- Create: `tests/e2e/__init__.py` (empty)
- Create: `tests/e2e/reference.py`
- Create: `tests/e2e/conftest.py`
- Create: `tests/e2e/test_transformer_block.py`

- [ ] **Step 1: Create `tests/e2e/__init__.py`**

Empty file:
```bash
mkdir -p /home/xs/tsy-wsl-export/tensor-sysy/tests/e2e
touch /home/xs/tsy-wsl-export/tensor-sysy/tests/e2e/__init__.py
```

- [ ] **Step 2: Create `tests/e2e/reference.py`**

```python
"""Numpy reference forward pass for examples/transformer_block.tsy.

Fill rule mirrors src/lir/interpreter.cpp:
    value[elem_idx] = buf_idx * 0.5 + elem_idx * 0.1
applied to each parameter buffer before reshape.

Param order matches the transformer_block(...) signature:
    0: x      [S, D]
    1: Wq     [D, D]
    2: Wk     [D, D]
    3: Wv     [D, D]
    4: Wo     [D, D]
    5: W1     [D, F]
    6: W2     [F, D]
"""
from __future__ import annotations

import numpy as np

S, D, F = 4, 8, 16


def det_fill(buf_idx: int, shape: tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape))
    flat = np.array(
        [buf_idx * 0.5 + i * 0.1 for i in range(n)],
        dtype=np.float32,
    )
    return flat.reshape(shape)


def rmsnorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sq = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(sq + eps)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax_lastdim(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def forward() -> np.ndarray:
    x = det_fill(0, (S, D))
    Wq = det_fill(1, (D, D))
    Wk = det_fill(2, (D, D))
    Wv = det_fill(3, (D, D))
    Wo = det_fill(4, (D, D))
    W1 = det_fill(5, (D, F))
    W2 = det_fill(6, (F, D))

    x_n = rmsnorm(x)
    Q = x_n @ Wq
    K = x_n @ Wk
    V = x_n @ Wv
    Kt = K.T  # (D, S)
    scores = Q @ Kt  # (S, S) — no sqrt(d) scaling, matches the fixture
    attn = softmax_lastdim(scores)
    ctx = attn @ V  # (S, D)
    a_out = ctx @ Wo
    x1 = x + a_out

    x1_n = rmsnorm(x1)
    h = x1_n @ W1  # (S, F)
    h1 = relu(h)
    f_out = h1 @ W2  # (S, D)
    out = x1 + f_out
    return out
```

- [ ] **Step 3: Create `tests/e2e/conftest.py`**

```python
"""Shared helpers for tensor-sysy e2e tests.

Provides:
  REPO_ROOT / BUILD_DIR / TSC — filesystem anchors.
  run_backend(backend, tsy_file) — run tsc subprocess, return stdout.
  parse_run_lir_output(stdout, buf_name) — pull a named tensor out of
      the printed run-lir report as a numpy array.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
TSC = BUILD_DIR / "tsc"


def run_backend(backend: str, tsy_file: Path) -> str:
    if not TSC.exists():
        pytest.skip(f"tsc not built at {TSC}; run cmake --build build first")
    cmd = [str(TSC), "run-lir", f"--backend={backend}", str(tsy_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if "requires tsy_runtime_cuda" in proc.stderr:
            pytest.skip(f"backend={backend} not built (no CUDA runtime)")
        raise RuntimeError(
            f"tsc run-lir --backend={backend} failed with rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def _parse_floats(tokens: list[str]) -> list[float]:
    out = []
    for tok in tokens:
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def parse_run_lir_output(stdout: str, buf_name: str) -> np.ndarray:
    """Extract buffer named `buf_name` from tsc run-lir stdout.

    Format (see src/lir/interpreter.cpp printRunResult):
        (local|input) <name> shape=[d0,d1,...]:
            <rows of floats>
    """
    header_pat = rf"(?:local|input)\s+{re.escape(buf_name)}\s+shape=\[([0-9,\s]+)\]:"
    m = re.search(header_pat, stdout)
    assert m, f"buffer {buf_name!r} not found in run-lir output"
    shape = tuple(int(x.strip()) for x in m.group(1).split(",") if x.strip())
    expected = int(np.prod(shape))

    tail = stdout[m.end():]
    # Body ends at next "local"/"input"/"function:" header or EOF.
    stop = re.search(r"\n\s*(?:input |local |function:)", tail)
    body = tail[: stop.start()] if stop else tail
    nums = _parse_floats(body.split())
    assert len(nums) >= expected, (
        f"buffer {buf_name!r}: parsed {len(nums)} floats, expected {expected}"
    )
    return np.asarray(nums[:expected], dtype=np.float32).reshape(shape)
```

- [ ] **Step 4: Create `tests/e2e/test_transformer_block.py`**

```python
"""W10 e2e: compare 3 backends against numpy reference."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import REPO_ROOT, parse_run_lir_output, run_backend
from .reference import forward as ref_forward


TSY = REPO_ROOT / "examples" / "transformer_block.tsy"


@pytest.mark.parametrize("backend", ["native", "cpu-adapter", "cuda-adapter"])
def test_transformer_block_matches_numpy(backend):
    stdout = run_backend(backend, TSY)
    actual = parse_run_lir_output(stdout, "out")
    expected = ref_forward()

    assert actual.shape == expected.shape, (
        f"shape mismatch: got {actual.shape}, expected {expected.shape}"
    )
    np.testing.assert_allclose(
        actual, expected,
        atol=1e-3, rtol=1e-2,
        err_msg=f"backend={backend} differs from numpy reference",
    )
```

- [ ] **Step 5: Verify the pytest suite runs manually**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
.venv/bin/python -m pytest tests/e2e/test_transformer_block.py -v 2>&1 | tail -20
```

Expected: three test lines (native, cpu-adapter, cuda-adapter), all PASSED. If cuda-adapter is unavailable (no CUDA build), it shows as SKIPPED, which is also fine.

If `assert_allclose` fails on ANY backend:
1. First check the det_fill pattern matches `src/lir/interpreter.cpp:detValue` — they must be byte-identical (`buf_idx * 0.5 + i * 0.1`).
2. Second, examine the actual vs expected diff magnitude. If <5e-3 on all elements, widen the tolerance to `atol=5e-3, rtol=1e-2` (matches spec §7 fallback).
3. If the diff is huge (>0.1), there's a real op bug — debug per op. Print intermediate buffers (`x_n`, `Q`, `attn`, ...) from stdout and compare to numpy step-by-step.

---

## Task 20: Wire e2e pytest into ctest

**Goal:** `ctest` now includes the pytest run as test #32.

**Files:**
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Append ctest**

Append to the end of `tests/CMakeLists.txt`:

```cmake
# W10: e2e pytest — compares 3 backends against numpy reference.
add_test(NAME e2e_transformer_block_pytest
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python -m pytest
            -xvs ${CMAKE_SOURCE_DIR}/tests/e2e/test_transformer_block.py
)
set_tests_properties(e2e_transformer_block_pytest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
```

- [ ] **Step 2: Full regression**

```bash
cmake -S . -B build
cmake --build build -j 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -10
```

Expected: `100% tests passed, 0 tests failed out of 32`.

The breakdown:
- 29 from W0-W9
- `transpose_relu_cases` (unit)
- `cli_run_transformer_block_native` (CLI smoke)
- `e2e_transformer_block_pytest` (pytest wraps 3 parametrized sub-tests into 1 ctest)

---

## Task 21: Commit 7

- [ ] **Step 1: Commit**

```bash
git add .gitignore PLAN.md \
        tests/e2e/__init__.py tests/e2e/reference.py \
        tests/e2e/conftest.py tests/e2e/test_transformer_block.py \
        tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
test(e2e): numpy reference + pytest + uv venv + PLAN.md tools table

First external-framework validation in the project. Previously all
cross-backend checks were tensor-sysy vs tensor-sysy (same-error
blind spots). e2e_transformer_block_pytest compares 3 backends
(native / cpu-adapter / cuda-adapter) against a 30-line numpy
reference at atol=1e-3, rtol=1e-2. cuda-adapter is gracefully
skipped on non-CUDA machines.

reference.py mirrors src/lir/interpreter.cpp:detValue byte-for-byte
(buf_idx*0.5 + i*0.1). conftest.py's parse_run_lir_output regex
matches the printRunResult format exactly.

numpy + pytest installed via uv inside project-local .venv;
.venv/ is .gitignore'd (already was). PLAN.md "项目期间安装的工具"
table gets the two new rows.

Total ctest count: 32/32.
EOF
)"
git log --oneline -1
```

---

## Task 22: Final verification + Commit 8 (spec + plan)

**Goal:** Clean rebuild, run the W10 acceptance matrix (spec §7), commit the spec + plan docs.

**Files:**
- None modified (docs already exist).

- [ ] **Step 1: Clean rebuild from scratch**

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
rm -rf build
cmake -S . -B build 2>&1 | tail -10
cmake --build build -j 2>&1 | tail -10
```

Expected: full clean configure + build, including `transformer_block`, `transformer_block_cuda`, `matmul_cuda_demo`, `mlp`, `tsy-bench` targets.

- [ ] **Step 2: Full ctest**

```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: `100% tests passed, 0 tests failed out of 32`.

- [ ] **Step 3: Acceptance matrix from spec §7**

```bash
# (1) rebuild — already done above.

# (2) ctest 32/32 — already done.

# (3) transformer_block binary prints local out shape=[4,8]
./build/out/transformer_block | grep -q "local out shape=\[4,8\]" && echo OK3 || echo FAIL3

# (4) transformer_block_cuda produces atol=1e-3-consistent numbers
./build/out/transformer_block | tail -4 > /tmp/w10_cpu_out.txt
./build/out/transformer_block_cuda | tail -4 > /tmp/w10_cuda_out.txt
diff /tmp/w10_cpu_out.txt /tmp/w10_cuda_out.txt > /tmp/w10_diff.txt || true
wc -l /tmp/w10_diff.txt
# Small diffs expected (atol 1e-3); ensure no WILD mismatch by showing max line delta
echo "CPU vs CUDA output (last 4 lines each):"
paste /tmp/w10_cpu_out.txt /tmp/w10_cuda_out.txt

# (5) pytest green
.venv/bin/python -m pytest tests/e2e/test_transformer_block.py -v 2>&1 | tail -5

# (6) W7 MLP unchanged
./build/out/mlp | tail -3 | grep -q "0.4502" && echo OK6 || echo FAIL6

# (7) emit-lir --opt=O1 on transformer_block shows 8 matmul calls with variant attr
./build/tsc emit-lir --opt=O1 --disable-pass=dce examples/transformer_block.tsy | grep -c "call matmul.*variant=" 
# Expected: 8
```

Expected: `OK3`, paste output shows matching numbers line-by-line (approximate), pytest PASSED (all backends), `OK6`, grep count `8`.

- [ ] **Step 4: Commit spec + plan docs**

```bash
git add docs/superpowers/specs/2026-04-18-tensor-sysy-w10-transformer-design.md \
        docs/superpowers/plans/2026-04-18-tensor-sysy-w10-transformer.md
git commit -m "$(cat <<'EOF'
docs: W10 transformer block spec + implementation plan

Spec: toy single-head transformer block (no MHA/mask/RoPE), 8 matmul
chain + 2 rmsnorm + 1 transpose + 1 softmax + 1 relu + 2 add, running
as examples/transformer_block.tsy. First external-framework validation
for the project — numpy reference compared against all 3 backends at
atol=1e-3, rtol=1e-2.

Plan: 8 commit milestones, 22 tasks. Coverage table at the end maps
each spec section to its implementing task.
EOF
)"
git log --oneline | head -10
```

- [ ] **Step 5: Echo W10 acceptance**

```bash
echo "W10 ACCEPTANCE: GREEN"
```

---

## Rollback / recovery

If any task fails irrecoverably:
```bash
git reset --hard HEAD~1   # undo last commit
git stash                  # preserve uncommitted work
```

Non-CUDA machines: `transformer_block_cuda` target + `e2e_transformer_block_pytest`'s cuda-adapter parametrize skip cleanly. W0-W7 + W8-CPU + transpose_relu_cases + cli_run_transformer_block_native + 2 of 3 pytest parametrizes stay green.

---

## Spec coverage

| Spec section | Implementing task |
|---|---|
| §4.1 `Transpose` + `ReLU` `OpKind` entries | Task 3 |
| §4.2 HIR verifier for transpose (2-D input, swapped result shape) | Task 4 |
| §4.2 HIR verifier for relu (reuses verifyUnary) | Task 4 Step 2 |
| §4.3 HIR→LIR lowering switch extension | Task 5 |
| §4.4 Interpreter `kernelTranspose` + `kernelReLU` + dispatch | Task 7 |
| §4.5 CPU adapter `adapterTranspose` + `adapterReLU` + dispatch | Task 9 |
| §4.6 CUDA adapter `transposeKernel` + `reluKernel` + wrappers + dispatch | Task 10 |
| §4.7 Codegen `adapterSymbolFor` for transpose/relu | Task 12 |
| §4.8 `module_utils.{h,cpp}` + 5 callers deduped | Task 1 |
| §4.9 `examples/transformer_block.tsy` | Task 14 Step 1 |
| §4.10 CMake tsy_add_example + tsy_add_cuda_example for transformer_block | Task 14 Step 2 |
| §4.11 `.venv` + numpy + pytest installed, documented in PLAN.md | Task 18 |
| §4.11 `tests/e2e/reference.py` | Task 19 Step 2 |
| §4.11 `tests/e2e/conftest.py` | Task 19 Step 3 |
| §4.11 `tests/e2e/test_transformer_block.py` | Task 19 Step 4 |
| §4.12 ctest `transpose_relu_cases` | Task 15 |
| §4.12 ctest `cli_run_transformer_block_native` | Task 16 |
| §4.12 ctest `e2e_transformer_block_pytest` | Task 20 |
| §5 Test matrix (32/32 green) | Tasks 15, 16, 20 + full regression each step |
| §6 Risk: transpose semantics | Task 15 fixes 2×2 and 2×3 goldens early |
| §6 Risk: det_fill mismatch | Task 19 Step 2 mirrors interpreter.cpp byte-for-byte |
| §6 Risk: tolerance overshoot | Task 19 Step 5 documents the 5e-3 fallback |
| §6 Risk: 5-way dedup missed one | Task 1 Step 5 includes grep verification |
| §7 Acceptance (7 checks) | Task 22 Step 3 |
| §9 Commits 1-8 (spec lists 9; plan consolidates to 8 by merging cpu+cuda adapter into one commit) | Tasks 2 / 6 / 8 / 11 / 13 / 17 / 21 / 22 |

All spec requirements map to at least one task.

---

## Notes on the 8-vs-9 commit discrepancy

The spec §9 lists 9 commit messages but says "最终 8 个 commit"; this plan merges spec's "feat(runtime): cpu-adapter ..." (spec #4) and "feat(runtime): cuda-adapter ..." (spec #5) into a single "feat(runtime): transpose + relu for cpu-adapter and cuda-adapter" (Task 11). This preserves the one-logical-milestone-per-commit principle (both adapters ship the same two primitives) while matching spec's "8 commit" headline count.
