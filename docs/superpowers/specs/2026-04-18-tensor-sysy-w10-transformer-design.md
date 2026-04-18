# W10 — Transformer Block E2E (Toy Single-Head) · 设计文档

**日期**：2026-04-18
**项目**：tensor-sysy（W9 已落地：scheduler + layout stub + benchmark, 29/29 绿）
**对应 PLAN.md 行**：§W10「Transformer Block E2E」
**前置**：`docs/superpowers/specs/2026-04-18-tensor-sysy-w9-scheduler-design.md`

## 1. 目标

在 tensor-sysy 上跑通一个**toy 单头 transformer block**：rmsnorm → attention → residual → rmsnorm → FFN(ReLU) → residual，三个 backend（native / cpu-adapter / cuda-adapter）输出都和 numpy reference 在 `atol=1e-3, rtol=1e-2` 内一致。

这是 12 阶段 PLAN.md 的**第一个端到端架构对拍**：第一次让编译器的输出被独立参考框架（numpy）验证，而不是三个自家 backend 互相对拍。

## 2. 范围（用户 2026-04-18 四轮确认）

### In-scope
- 单 `transformer_block(...)` HIR 函数，串行 op 链
- 两个新 HIR / LIR primitive：`@transpose`（2-D only）+ `@relu`（任意 rank）
- 三个 backend（native interpreter / cpu-adapter / cuda-adapter）全部支持新 primitive
- 一个 `.tsy` fixture + CMake `tsy_add_example` + `tsy_add_cuda_example` 各编一个产物
- numpy-based pytest e2e 测试，覆盖 3 backend
- 顺手清理 `pickFirstTensorFunction` 5 份复制 → 统一到 `src/lir/module_utils.{h,cpp}`

### Out-of-scope（YAGNI，推后）
- MHA / GQA / multi-head reshape（需 `@view` + `@permute` + MHA 调度）
- RoPE（需专用 kernel）
- Causal mask / 其它 attention mask
- GELU / SiLU（W10 仅 ReLU）
- LayerNorm（继续用 rmsnorm）
- KV cache（inference 概念，forward-only 无需）
- LayoutLoweringPass 真实 body（继续 W9 的 stub）
- torch 依赖（numpy 即够）
- Scaled-dot-product 缩放（不除 √d_head，参考 numpy 也不缩放，语义一致）
- `@mul` / `@scale` 等 elementwise 新 op
- transformer 进 tsy-bench（推 W11）

## 3. 架构

```
x → rmsnorm → Q=x_n@Wq, K=x_n@Wk, V=x_n@Wv → Kt=transpose(K) → S=Q@Kt → A=softmax(S)
     → ctx=A@V → a_out=ctx@Wo → x1 = x + a_out
x1 → rmsnorm → h=x1_n@W1 → h1=relu(h) → f_out=h1@W2 → out = x1 + f_out
```

8 次 matmul（Q/K/V projs 3 + Q@Kt + A@V + ctx@Wo + x1_n@W1 + h1@W2） + 2 rmsnorm + 1 softmax + 1 transpose + 1 relu + 2 add。Shapes S=4, D=8, F=16（小但非平凡，足够覆盖 scheduler 的不同 shape class）。

HIR 保持现有结构：`OpKind` 加 2 枚。LIR 无变化（只是 primitive 字符串多 2 个）。runtime adapter / codegen 每处加 2 条分派。

## 4. 组件详表

### 4.1 HIR: `Transpose` + `ReLU` (`src/hir/ops.{h,cpp}`)

扩 `OpKind`:
```cpp
enum class OpKind {
    // ... existing
    Transpose,  // W10: 2-D transpose. result shape = [in.dim1, in.dim0].
    ReLU,       // W10: elementwise relu. result shape = in.shape.
};
```

`toString` 加两行；`builtinKindFromName` 加:
```cpp
if (name == "transpose") return OpKind::Transpose;
if (name == "relu")      return OpKind::ReLU;
```

### 4.2 HIR verifier (`src/hir/verifier.cpp`)

- `@transpose(X)`：assert X is 2-D (`X.shape.rank() == 2`)；result shape = `[X.dims[1], X.dims[0]]`。诊断：
  - `transpose expects a 2-D tensor, got rank N`
  - result shape 不匹配源码声明时报 `transpose result shape [...] does not match declared [...]`
- `@relu(X)`：任意 rank；result shape = X.shape；result dtype = X.dtype

### 4.3 HIR → LIR lowering (`src/hir/lowering.cpp`)

Transpose / ReLU 下降为 `Stmt{ primitive="transpose"/"relu", operand_bufs = {X.id}, result_buf = Y.id }`。和 existing matmul/add/softmax/rmsnorm lowering 同一分派结构。

### 4.4 Interpreter (`src/lir/interpreter.cpp`)

- `runTranspose(Buf& x, Buf& y)`：assert `x.dims.size() == 2`；`M = x.dims[0], N = x.dims[1]`；双循环 `y[j*M+i] = x[i*N+j]`
- `runReLU(Buf& x, Buf& y)`：flat `for i: y[i] = x[i] < 0 ? 0 : x[i]`
- executor 分派加两条 case

### 4.5 CPU adapter (`src/runtime/adapter_cpu.{h,cpp}`)

- `adapterTranspose(const Tensor& x, Tensor& y)`：和 interpreter 同实现，但走 adapter_cpu 的 executor path
- `adapterReLU(const Tensor& x, Tensor& y)`：同上
- executor `runFunctionAdapter` 分派加 `if (s.primitive == "transpose") { ... adapterTranspose(...); } else if (s.primitive == "relu") { ... adapterReLU(...); }`

### 4.6 CUDA adapter (`src/runtime/adapter_cuda.{h,cu}`)

两个新 kernel:
```cuda
__global__ void transposeKernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int M, int N) {
    // naive: one thread per (i, j). Block (32, 32), grid (ceil(N/32), ceil(M/32)).
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row of x
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col of x
    if (i >= M || j >= N) return;
    y[j * M + i] = x[i * N + j];
}

__global__ void reluKernel(const float* __restrict__ a, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = fmaxf(0.0f, a[i]);
}
```

两个 adapter：`adapterTransposeCuda` / `adapterReLUCuda`，和 `adapterAddCuda` / `adapterSoftmaxCuda` 同 pattern（cudaMalloc → memcpy H2D → launch → memcpy D2H → cudaFree，全包 CUDA_CHECK）。

Executor 分派加两条。

### 4.7 Codegen (`src/codegen/cpp.cpp` + `cuda.cpp`)

两份都在 `adapterSymbolFor`:
```cpp
if (primitive == "transpose") return "adapterTranspose";  // or adapterTransposeCuda in cuda.cpp
if (primitive == "relu")      return "adapterReLU";       // or adapterReLUCuda
```

其它 emit 逻辑不变；新 primitive 也是 "1 operand + 1 result" 形式，adapter call 签名和 softmax/rmsnorm 同形。

### 4.8 `src/lir/module_utils.{h,cpp}`（新）

```cpp
// src/lir/module_utils.h
#pragma once
#include "ir.h"

namespace tsy::lir {

// Pick the first "interesting" function to execute — skip main, prefer
// functions with tensor params; fallback to module.funcs.front().
// Previously duplicated across interpreter, adapter_cpu, adapter_cuda,
// codegen/cpp, codegen/cuda.
const Function* pickFirstTensorFunction(const Module& m);

}  // namespace tsy::lir
```

```cpp
// src/lir/module_utils.cpp
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

5 处 caller 全换成 `#include "module_utils.h"` + `tsy::lir::pickFirstTensorFunction(m)`：
- `src/lir/interpreter.cpp`
- `src/runtime/adapter_cpu.cpp`
- `src/runtime/adapter_cuda.cu`
- `src/codegen/cpp.cpp`
- `src/codegen/cuda.cpp`

`CMakeLists.txt`：
```cmake
add_library(tsy_lir STATIC
    src/lir/printer.cpp
    src/lir/lowering.cpp
    src/lir/interpreter.cpp
    src/lir/module_utils.cpp
)
```

### 4.9 Fixture `examples/transformer_block.tsy`

完整内容见 §3 架构图对应的 tsy 代码（已在草案里给出）。

S=4, D=8, F=16。无返回值，通过 param 绑定捕获 `out` buffer。

### 4.10 CMake example wiring

```cmake
tsy_add_example(transformer_block examples/transformer_block.tsy)
tsy_add_cuda_example(transformer_block_cuda examples/transformer_block.tsy)
```

(两个 target 共用同一个 .tsy source。)

### 4.11 Python 测试 infra

#### uv venv 创建
```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
uv venv .venv
uv pip install --python .venv/bin/python numpy pytest
```

PLAN.md 末尾「已安装工具」表追加：
| 工具 | 方式 | 时间 | 原因 | 卸载 |
|---|---|---|---|---|
| numpy >=1.24 | `uv pip install --python .venv/bin/python numpy` | 2026-04-18 W10 | e2e 参考实现 | `rm -rf .venv` |
| pytest >=7 | `uv pip install --python .venv/bin/python pytest` | 2026-04-18 W10 | e2e 测试驱动 | `rm -rf .venv` |

`.venv/` 进 `.gitignore`（已有 `.venv/` 条目，确认）。

#### `tests/e2e/reference.py`

```python
"""Numpy reference for transformer_block.tsy.

Fill rule mirrors src/lir/interpreter.cpp: fillDeterministic(buf_idx, n):
  value[i] = buf_idx * 0.5 + i * 0.1
The caller reshapes after fill.
"""
import numpy as np

S, D, F = 4, 8, 16

def det_fill(buf_idx: int, shape: tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape))
    flat = np.array([buf_idx * 0.5 + i * 0.1 for i in range(n)],
                    dtype=np.float32)
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
    # Param order matches transformer_block(...) signature:
    # 0: x, 1: Wq, 2: Wk, 3: Wv, 4: Wo, 5: W1, 6: W2
    x  = det_fill(0, (S, D))
    Wq = det_fill(1, (D, D))
    Wk = det_fill(2, (D, D))
    Wv = det_fill(3, (D, D))
    Wo = det_fill(4, (D, D))
    W1 = det_fill(5, (D, F))
    W2 = det_fill(6, (F, D))

    x_n  = rmsnorm(x)
    Q    = x_n @ Wq
    K    = x_n @ Wk
    V    = x_n @ Wv
    Kt   = K.T                      # (D, S)
    S_   = Q @ Kt                   # (S, S)  — no scaling, dot-product attention
    A_   = softmax_lastdim(S_)
    ctx  = A_ @ V                   # (S, D)
    a_out= ctx @ Wo
    x1   = x + a_out

    x1_n = rmsnorm(x1)
    h    = x1_n @ W1                # (S, F)
    h1   = relu(h)
    f_out= h1 @ W2                  # (S, D)
    out  = x1 + f_out
    return out
```

#### `tests/e2e/conftest.py`

```python
"""pytest fixtures and helpers for tensor-sysy e2e tests."""
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
TSC = BUILD_DIR / "tsc"


def parse_run_lir_output(stdout: str, buf_name: str) -> np.ndarray:
    """Extract a named buffer from tsc run-lir stdout.

    Format (from src/lir/printer.cpp + interpreter.cpp):
        local out shape=[4,8]:
            <space-separated floats>
            <space-separated floats>
            ...
    Returns a numpy array reshaped to the declared shape.
    """
    pat = rf"(?:local|input)\s+{re.escape(buf_name)}\s+shape=\[([0-9, ]+)\]:"
    m = re.search(pat, stdout)
    assert m, f"buffer {buf_name!r} not found in run-lir output"
    shape = tuple(int(x) for x in m.group(1).split(","))
    # Grab numbers following the header line up to the next buf header or blank.
    tail = stdout[m.end():]
    # Next "input X" or "local X" or "function:" marks end of this buffer block.
    stop = re.search(r"\n\s*(?:input|local|function:|$)", tail)
    body = tail[:stop.start()] if stop else tail
    floats = [float(tok) for tok in body.split() if tok.replace("-", "").replace(".", "").replace("e", "").isdigit() or tok.startswith("-")]
    n = int(np.prod(shape))
    assert len(floats) >= n, f"found {len(floats)} floats, expected {n}"
    return np.asarray(floats[:n], dtype=np.float32).reshape(shape)


def run_backend(backend: str, tsy_file: Path) -> str:
    if not TSC.exists():
        pytest.skip(f"tsc not built at {TSC}")
    cmd = [str(TSC), "run-lir", f"--backend={backend}", str(tsy_file)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
```

(The float-extraction in `parse_run_lir_output` is a bit regex-heavy; simpler approach: just whitespace-split the body and float-parse tokens that successfully convert. Cleaned up below for the plan.)

#### `tests/e2e/test_transformer_block.py`

```python
"""W10 e2e: compare 3 backends against numpy reference."""
import numpy as np
import pytest

from .reference import forward as ref_forward
from .conftest import REPO_ROOT, run_backend, parse_run_lir_output

TSY = REPO_ROOT / "examples" / "transformer_block.tsy"


@pytest.mark.parametrize("backend", ["native", "cpu-adapter", "cuda-adapter"])
def test_transformer_block_matches_numpy(backend):
    try:
        stdout = run_backend(backend, TSY)
    except FileNotFoundError:
        pytest.skip(f"tsc missing (build first)")
    actual = parse_run_lir_output(stdout, "out")
    expected = ref_forward()
    assert actual.shape == expected.shape, (
        f"shape mismatch: got {actual.shape}, expected {expected.shape}")
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-2)
```

CUDA backend 的 skip-on-missing 逻辑：`tsc run-lir --backend=cuda-adapter` 在未编 CUDA runtime 时会 exit 1 with "requires tsy_runtime_cuda" 消息。pytest 要捕获并 skip，而不是 fail：

```python
@pytest.mark.parametrize("backend", ["native", "cpu-adapter", "cuda-adapter"])
def test_transformer_block_matches_numpy(backend):
    import subprocess
    try:
        stdout = run_backend(backend, TSY)
    except subprocess.CalledProcessError as e:
        if "requires tsy_runtime_cuda" in e.stderr:
            pytest.skip("cuda-adapter not built")
        raise
    # ... (same as above)
```

### 4.12 ctest 新增 4 条

```cmake
# W10: HIR transpose + relu unit tests (CPU-only, gated on tsy_runtime_cpu).
if(TARGET tsy_runtime_cpu)
    add_executable(test_transpose_relu_cases adapter/test_transpose_relu_cases.cpp)
    target_link_libraries(test_transpose_relu_cases PRIVATE tsy_runtime_cpu)
    add_test(NAME transpose_relu_cases
        COMMAND test_transpose_relu_cases
    )
endif()

# W10: transformer_block native run — must print "local out shape=[4,8]".
add_test(NAME cli_run_transformer_block_native
    COMMAND tsc run-lir ${CMAKE_SOURCE_DIR}/examples/transformer_block.tsy
)
set_tests_properties(cli_run_transformer_block_native PROPERTIES
    PASS_REGULAR_EXPRESSION "local out shape=\\[4,8\\]"
)

# W10: e2e pytest — compare 3 backends against numpy reference.
add_test(NAME e2e_transformer_block_pytest
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python -m pytest
            -xvs ${CMAKE_SOURCE_DIR}/tests/e2e/test_transformer_block.py
)
set_tests_properties(e2e_transformer_block_pytest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
```

`hir_transpose_case` / `hir_relu_case` 合成一个 `transpose_relu_cases` 驱动，每个 op 3-4 个 assert。内部测：
- transpose([[1,2],[3,4]]) == [[1,3],[2,4]]
- transpose((2,3) random) 形状和元素对应
- relu([-1, 0, 1, 2]) == [0, 0, 1, 2]
- relu(shape (2, 3)) 形状保留、负值清零

## 5. 测试矩阵（29 + 4 = 33）

| ID | 测试 | 验证 |
|---|---|---|
| #30 | `transpose_relu_cases` | 两个新 primitive 的 CPU adapter 单算子正确性 |
| #31 | `cli_run_transformer_block_native` | fixture 能跑通，产物中带 `local out shape=[4,8]` |
| #32 | (合并在 #31) | — |
| #33 | `e2e_transformer_block_pytest` | 3 backend vs numpy 全过（pytest 内 3 条 parametrize） |

（`hir_transpose_case` + `hir_relu_case` 合成一个 driver，ctest 看就是 1 条；e2e pytest 的 3 条 parametrize 也合成 1 条 ctest。所以总数 29 + 3 = 32 可能；我在 §8 验收里按 32 算，不强求 33。）

**更正**：按 **32/32 green** 作为验收标准（29 W0-W9 + `transpose_relu_cases` + `cli_run_transformer_block_native` + `e2e_transformer_block_pytest` = 32）。

## 6. 风险 + mitigation

| 风险 | Mitigation |
|---|---|
| `@transpose` 语义被 HIR/LIR/adapter 三层实现不同（行列主混淆） | 先写 transpose_relu_cases unit test 固化 2×2 → 2×2 数值，跨 3 backend 同断言 |
| numpy `det_fill` 和 C++ `fillDeterministic` 算法微差（round-trip 整数 vs float） | 用相同 `buf_idx * 0.5 + i * 0.1` 浮点公式；test 包含一个 "空 transformer，只 dump input buffer" 的 smoke 对比，确保种子齐 |
| 8 次 matmul 链累积误差超 atol=1e-3 | pytest 默认 `atol=1e-3, rtol=1e-2`；若 CUDA 路径超 1e-3，放到 5e-3。在 spec §7 里说明这是可调。不调就 FAIL_LOOSER。 |
| `pickFirstTensorFunction` 5 份删除漏掉一处导致多定义链接错误 | 5 处 grep 后统一删；tsy_lir 只保留 module_utils 的那一份；重新 cmake configure + build 如果有 multiple definition 报错，立刻知道 |
| `uv venv` 依赖安装在 WSL 网络受限时失败 | 用户有 proxy_on 函数；spec 在命令里不强制 proxy，README 说明如有问题先 `proxy_on` |
| pytest ctest 的 `WORKING_DIRECTORY` + `.venv/bin/python` 路径在 clean-rebuild 后相对 cmake binary dir 丢失 | `${CMAKE_SOURCE_DIR}/.venv/bin/python` 用绝对路径解析；ctest 只负责执行 |
| cuda-adapter backend 在非 CUDA 机器 skip 但 pytest 总数跟预期不齐 | pytest.skip 不算 fail，ctest 收集 exit code 仍然 0；tsc exit 1 的非 CUDA-missing 错误要照常 fail |
| emit-cpp / emit-cu 默认 `--opt=O0` 不跑 scheduler，codegen 不注入 variant arg，和 W9 测试不冲突 | W8+W9 已验证；transformer_block E2E 也用默认 O0，回归面安全 |
| `fmaxf` 在 -Wfloat-conversion 下警告 | ReLU kernel 用 `fmaxf(0.0f, a[i])`（已是 float 字面量）；-Wall -Wextra 不触发 |

## 7. 验收口径（§8 对应 §6 最终统计）

**同时满足**才算 W10 完成：

1. `rm -rf build && cmake -S . -B build && cmake --build build -j` 全量 clean（含 `transformer_block` + `transformer_block_cuda`）
2. `ctest --test-dir build --output-on-failure` → **32/32 green**
3. `./build/out/transformer_block` 输出含 `local out shape=[4,8]`
4. `./build/out/transformer_block_cuda` 输出数值（同样 32 float）和 CPU 版 atol=1e-3 一致
5. `.venv/bin/python -m pytest tests/e2e/test_transformer_block.py -v` 三条参数化全过
6. `./build/out/mlp` 仍出 `0.4502 0.5498`（W7 回归护身符）
7. `tsc emit-lir --opt=O1 examples/transformer_block.tsy` 可见 8 个 matmul，多数应标 `{variant="naive"}`（shape 较小），无 abort

## 8. 交付物清单

### 新文件
- `src/lir/module_utils.h` / `module_utils.cpp`
- `examples/transformer_block.tsy`
- `tests/adapter/test_transpose_relu_cases.cpp`
- `tests/e2e/__init__.py`
- `tests/e2e/conftest.py`
- `tests/e2e/reference.py`
- `tests/e2e/test_transformer_block.py`
- `docs/superpowers/specs/2026-04-18-tensor-sysy-w10-transformer-design.md`（本文件）

### 改动文件
- `src/hir/ops.h` / `ops.cpp`（OpKind + builtinKindFromName）
- `src/hir/verifier.cpp`（新 2 条验证规则）
- `src/hir/lowering.cpp`（Transpose / ReLU 下降分支）
- `src/lir/interpreter.cpp`（transpose / relu 实现 + executor dispatch + 删 1 份 `pickFirstTensorFunction`）
- `src/runtime/adapter_cpu.h` / `adapter_cpu.cpp`（adapterTranspose / adapterReLU + executor dispatch + 删 1 份）
- `src/runtime/adapter_cuda.h` / `adapter_cuda.cu`（两个 kernel + 两 adapter + executor dispatch + 删 1 份）
- `src/codegen/cpp.cpp`（adapterSymbolFor 增 2 条 + 删 1 份）
- `src/codegen/cuda.cpp`（同上 + 删 1 份）
- `CMakeLists.txt`（tsy_lir 加 module_utils.cpp；`tsy_add_example(transformer_block ...)` + `tsy_add_cuda_example(transformer_block_cuda ...)`）
- `tests/CMakeLists.txt`（3 条新 ctest）
- `PLAN.md`（末尾工具表追加 numpy + pytest 行）
- `.gitignore`（确认 `.venv/` 已在）

### 预期 commit 划分（8 个 milestone，仿 W9 节奏）

1. `refactor(lir): extract pickFirstTensorFunction to module_utils`（纯重构，0 功能）
2. `feat(hir): add @transpose and @relu ops (enum + verifier + lowering)`
3. `feat(lir): implement transpose/relu in interpreter`
4. `feat(runtime): transpose + relu for cpu-adapter`
5. `feat(runtime): transpose + relu for cuda-adapter`
6. `feat(codegen): emit adapterTranspose + adapterReLU in emit-cpp and emit-cu`
7. `feat(examples): transformer_block.tsy + CMake example targets + CLI + unit ctest`
8. `test(e2e): numpy reference + pytest + uv venv + PLAN.md tools table`
9. `docs: W10 spec`

（9 条 commit — commit #4 + #5 可合并为一条若都很小，最终 8 个 commit。）

## 9. 为什么这是「第一个真端到端对拍」

W4-W9 的所有对拍都是三个自家 backend（native / cpu-adapter / cuda-adapter）之间互相对比。同错会全绿。W10 是第一次：
- 让 numpy（独立参考）算同一份前向
- 让 tsc 产物的输出被第三方数学校验

这对未来 W11+ 的 transformer block 扩展（加 MHA / mask / RoPE）和整个 compiler correctness 边界都是**基础信任**。

## 10. 不做（重申 YAGNI）

- MHA / GQA 的 reshape（需 view+permute）
- RoPE
- Causal mask
- GELU / SiLU / gating-based FFN
- LayerNorm
- KV cache
- LayoutLoweringPass 真实 body
- torch 依赖
- transformer 进 benchmark

---

**前提约束**：
- W7 MLP 可复现输出作为"无回归"锚点
- W9 scheduler 在 O1 下可见 `{variant="..."}`，但本 W10 默认 O0，scheduler 不入 codegen，W8 `codegen_cuda_matmul_binary_runs` 继续绿
- `tsy_runtime_cuda` / CUDA 测试继续 gate 在 `TARGET tsy_runtime_cuda`，非 CUDA 机器自动跳过 cuda-adapter 相关断言
