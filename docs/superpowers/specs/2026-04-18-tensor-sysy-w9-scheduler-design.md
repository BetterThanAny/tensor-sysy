# W9 — CUDA Scheduler + Layout-Lowering 骨架 + Benchmark · 设计文档

**日期**：2026-04-18
**项目**：tensor-sysy（W8 已落地：adapter_cuda FP32 + emit-cu + 25/25 green）
**对应 PLAN.md 行**：§W9「CUDA 调度和 layout lowering」
**前置**：`docs/superpowers/specs/2026-04-18-tensor-sysy-w8-cuda-design.md`

## 1. 目标

围绕 matmul 建立一条**编译期 shape-aware kernel scheduler**，同时放好**layout lowering 的结构占位**（真正工作延到 W10），并交付一条**CUDA-event 精度的 benchmark harness**（C++ `tsy-bench` + 轻量 Python 驱动），使

- `tsc emit-lir --opt=O1 foo.tsy` 能在 LIR 输出里观察到 matmul 的 `variant=naive|tiled|cublas` 决策
- `--disable-pass=schedule-cuda` 能让决策退回默认（variant 缺省 = cuBLAS，和 W8 行为一致）
- `python benchmarks/run_shapes.py --smoke` 能跑通，产生 CSV 数据

## 2. 范围

**In-scope**（用户 2026-04-18 四轮确认）：
- Layout lowering：**骨架占位**（LIR `Stmt.attrs` 字段 + 空 `LayoutLoweringPass` 注册到 PassManager）。真正的 Permute/View lowering 推到 W10。
- Scheduler：**matmul 三变体**（naive / register-tiled 128×128 / cuBLAS），其他 3 个原语（add/softmax/rmsnorm）保持 W8 的单 kernel。
- Scheduler 位置：**编译期 LIR pass**（`ScheduleCudaPass`），写入 `Stmt.attrs["variant"]`，adapter/codegen 都读同一字段。
- Benchmark：**C++ `tsy-bench` 新工具**（CUDA event 计时，3 warmup + 5 measured，median）+ Python `benchmarks/run_shapes.py` 轻量驱动（纯标准库，无 numpy/pandas 依赖）。

**Out-of-scope**：
- Permute / View 的真实 lowering 工作（W10 transformer block）
- softmax / rmsnorm 的 online / warp-shuffle 变体选择（YAGNI 本周期）
- FP16 / INT8（W11+）
- 多算子 fusion（W10/W11）
- 运行时 autotuner / 动态重查表（静态查表即可）
- Bench 结果的 regression gate（W11 才引入中位数阈值）

## 3. 架构

```
.tsy → AST → HIR → LIR  (new: Stmt.attrs map)
                   │
                   ├── PassManager O1:
                   │     HIR: verify → const-fold → dce → verify  (W5 unchanged)
                   │     ↓ (HIR → LIR lowering, existing)
                   │     LIR: layout-lowering  (W9 NEW: stub, no-op)
                   │        → schedule-cuda    (W9 NEW: picks matmul variant)
                   │
                   ├── run-lir --backend=cuda-adapter
                   │     → executor reads attrs["variant"]
                   │     → adapterMatMulCuda(a, b, c, variant)
                   │         ├─ variant="naive"  → matmulNaiveKernel
                   │         ├─ variant="tiled"  → matmulTiledKernel
                   │         └─ variant=""/cublas → cuBLAS sgemm (W8)
                   │
                   └── emit-cu
                         → codegen writes adapterMatMulCuda(..., "tiled") calls
                         → nvcc compile → ./build/out/<name>
```

并行：
```
src/tools/tsy-bench.cu
   │ iterates shapes × variants,
   │ CUDA events time each (3 warmup + 5 measured, median),
   ▼ emits CSV (primitive,M,K,N,variant,ms_median,gflops)

benchmarks/run_shapes.py
   │ spawns tsy-bench, parses CSV,
   ▼ prints table / writes last_run.csv / --check-scheduler asserts tiled > naive
```

HIR 保持不变。所有 W9 改动局部在 LIR 及其下游。

## 4. 组件详表

### 4.1 LIR `Stmt.attrs` (`src/lir/ir.h` / `printer.cpp` / `lowering.cpp`)

```cpp
struct Stmt {
    StmtKind kind = StmtKind::Call;
    std::string primitive;
    std::vector<int> operand_bufs;
    int result_buf = -1;
    tsy::SourceLocation loc;
    std::unordered_map<std::string, std::string> attrs;  // NEW
};
```

**printer 行为**：attrs 为空 → 旧格式原样（`call matmul %A, %B`）。attrs 非空 → 在 `call primitive args` 后追加 `{k1="v1", k2="v2"}`，key 按字典序排序以保 golden 稳定。

**lowering 初始化**：`lowerHirToLir` 构造 Stmt 时不填 attrs（保持默认空 map），passes 再写入。

**W8 兼容**：现有 golden 测试全部在 attrs 为空的情况下写的，自动兼容。

### 4.2 Pass `ScheduleCudaPass` (`src/passes/schedule_cuda.cpp`)

**签名**：`void runScheduleCuda(tsy::lir::Module&, tsy::DiagnosticEngine&);`

注意：这是 LIR pass，不是 HIR pass。`pass_manager` 目前只接受 `tsy::hir::Module&` 签名的 PassFn。两条路：
- (a) 新增 LIR pass 机制（`LirPassFn`）
- (b) 让 pipeline 在 LIR 层显式调用 schedule pass，不走 PassManager

**采用 (a)**：扩展 `pass_manager.{h,cpp}`，加一个平行的 `LirPassFn = std::function<void(tsy::lir::Module&, DiagnosticEngine&)>` 和 `addLir(name, fn)`。`PassManager::run` 分前后两段：HIR passes 先跑，然后 HIR→LIR 下降（已有流程），再跑 LIR passes。`--disable-pass=schedule-cuda` 一个机制统一管两类。

（这个结构改动是 W9 真正的 architectural 代价。写得小心，保证现有 O0/O1 流程不变。）

**算法**：走 LIR module，对每个 Function 的每个 Stmt 如果 `primitive=="matmul"`，读取三个 operand buffer 的 dim（A[M,K], B[K,N] → C[M,N]），查表后写入 `s.attrs["variant"]`。其他 primitive 跳过。

**查表**（硬编码在 `schedule_cuda.cpp`）：
```cpp
std::string pickMatmulVariant(int64_t M, int64_t K, int64_t N) {
    // Tiny: use naive — cuBLAS launch overhead dominates.
    if (M * N < 1024) return "naive";

    // Medium aligned: register-tiled kernel is best.
    // Tiled kernel requires M,N,K divisible by 128 / 8 respectively;
    // also demands all three >= 128 for the tile to make sense.
    const bool aligned = (M % 128 == 0) && (N % 128 == 0) && (K % 8 == 0);
    const bool large_enough = (M >= 128) && (N >= 128) && (K >= 128);
    if (aligned && large_enough && M * N <= 256 * 256) return "tiled";

    // Fallback: cuBLAS (best for odd shape, very large, mixed dims).
    return "cublas";
}
```

**阈值说明**：初版数字是占位，bench 结果出来后同 PR 内调整。阈值表的来源见 §7 风险。

### 4.3 Pass `LayoutLoweringPass` stub (`src/passes/layout_lowering.cpp`)

W9 占位，**body 里什么都不做**，但结构完整（同样是 LIR pass）：

```cpp
void runLayoutLowering(tsy::lir::Module& /*m*/, tsy::DiagnosticEngine& /*diag*/) {
    // Stub. Real work arrives in W10 when Permute/View becomes live:
    //   - Recognise non-canonical matmul operand layouts and emit transposes
    //   - Expand View ops into adjacent buffer reshape + copy
    // For W9 this pass is registered in PassManager O1 pipeline so the
    // structural hookup is done once and W10 can fill in the body.
}
```

注册顺序在 `buildPipelineO1` 里：HIR 部分 `verify → const-fold → dce → verify`（W5 原样），下降到 LIR 后再 `layout-lowering → schedule-cuda`。Layout 先，scheduler 后（调度依赖终态 layout）。LIR 后不再 verify —— attrs 只加元数据不改结构，且当前没有 LIR verifier（留给 W11 CI 里再做）。

### 4.4 adapter_cuda 扩展 (`src/runtime/adapter_cuda.{h,cu}`)

#### `adapter_cuda.h`

```cpp
void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c,
                       const std::string& variant = "");
```

`variant`：`""` 或 `"cublas"` → cuBLAS sgemm（W8 行为，保留所有 W8 测试绿）；`"naive"` → 一 thread 一 output；`"tiled"` → BM=BN=128 BK=8 TM=TN=8 reg-tiled。

#### `adapter_cuda.cu`

新增 `matmulNaiveKernel`：
```cu
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
Launch：`block(32,32), grid(ceil(N/32), ceil(M/32))`。

新增 `matmulTiledKernel`：直接从 `mini-llm-engine/cuda-kernels/gemm/gemm.cu` 行 100–234 一字一句移植，改 `#include` 头和命名空间。BM/BN=128, BK=8, TM/TN=8, 256 threads/block。

`adapterMatMulCuda(a, b, c, variant)` 实现：
```cpp
void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c,
                      const std::string& variant) {
    // ... (shape extract, cudaMalloc, cudaMemcpy H2D — unchanged from W8)
    if (variant == "naive") {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmulNaiveKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    } else if (variant == "tiled") {
        // BM=BN=128, block=256. Requires aligned dims; caller (schedule
        // pass) is responsible for picking this variant only when safe.
        dim3 block(16, 16);
        dim3 grid(N / 128, M / 128);
        matmulTiledKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // "" or "cublas": the W8 path.
        // ... cublasSgemm call unchanged
    }
    // ... cudaMemcpy D2H, cudaFree — unchanged
}
```

executor `runFunctionCudaAdapter`：
```cpp
if (s.primitive == "matmul") {
    // ...
    std::string variant;
    auto it = s.attrs.find("variant");
    if (it != s.attrs.end()) variant = it->second;
    adapterMatMulCuda(r.buffers[s.operand_bufs[0]],
                      r.buffers[s.operand_bufs[1]], out, variant);
}
```

### 4.5 Codegen `emit-cu` 更新 (`src/codegen/cuda.cpp`)

在 `emitCudaModule` 的 body-calls 循环里，对 matmul 附加 variant 字符串参数（如果 attrs 有）。对其他 primitive 保持原样。

```cpp
if (s.primitive == "matmul") {
    os << "    tsy::runtime::adapterMatMulCuda(";
    // ... operands
    auto it = s.attrs.find("variant");
    if (it != s.attrs.end()) {
        os << ", \"" << it->second << "\"";
    }
    os << ");\n";
}
```

注：emit-cpp 不变（CPU adapter 不分 variant）。

### 4.6 `src/tools/tsy-bench.cu`（新工具）

单独一个 `.cu` 文件，不走 lowering，纯算子级 bench。结构从 `mini-llm-engine/cuda-kernels/gemm/gemm.cu` 移植：
- `fill_random(float*, int n)` 用固定 seed
- `check_correctness(ref, test, n, atol)`
- `bench(lambda)` → 3 warmup + 5 measured via cudaEvent，median
- `main()`：parse args (`--shapes`, `--variants`, `--smoke`)，对每组 (shape, variant) 跑 bench，emit 一行 CSV

CLI：
```bash
tsy-bench                               # 默认 5 shape × 3 variant = 15 行
tsy-bench --shapes 1024x1024x1024       # 单 shape × 所有 variant
tsy-bench --smoke                       # 1 shape × 3 variant，ret=0 即 OK
tsy-bench --variants naive,tiled        # 过滤 variant
```

输出 CSV 头固定：
```
primitive,M,K,N,variant,ms_median,gflops
```

默认 shape 集：`256x256x256, 512x512x512, 1024x1024x1024, 128x16x8 (tall), 7x13x11 (odd)`. 最后两个覆盖 scheduler 的边界 case。

### 4.7 `benchmarks/run_shapes.py`（纯标准库）

```python
#!/usr/bin/env python3
"""Thin driver over build/tsy-bench. No external deps."""
import argparse, csv, io, subprocess, sys
from pathlib import Path

BENCH = Path("build/tsy-bench")

def run(args: list[str]) -> list[dict]:
    cmd = [str(BENCH), *args]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(io.StringIO(out)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--check-scheduler", action="store_true")
    args = p.parse_args()

    bench_args = ["--smoke"] if args.smoke else []
    rows = run(bench_args)

    if not rows:
        print("no rows returned from tsy-bench", file=sys.stderr)
        return 1

    for r in rows:
        print(f"{r['primitive']} {r['M']:>4}x{r['K']:>4}x{r['N']:>4} "
              f"{r['variant']:>8}  {float(r['ms_median']):7.3f} ms  "
              f"{float(r['gflops']):8.1f} GFLOPS")

    if args.check_scheduler:
        # Expect: at 1024^3, tiled faster than naive (hardware-independent invariant)
        by = {(r['M'], r['variant']): float(r['ms_median']) for r in rows}
        if ('1024', 'tiled') in by and ('1024', 'naive') in by:
            speedup = by[('1024', 'naive')] / by[('1024', 'tiled')]
            if speedup < 2.0:
                print(f"FAIL: tiled only {speedup:.1f}x faster than naive "
                      f"at 1024^3 (expected >= 2x)", file=sys.stderr)
                return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**不引入 numpy / pandas**。

## 5. 新 Example 文件

- `examples/run_matmul_medium.tsy`：128×128×128 matmul（推 variant=tiled）
- `examples/run_matmul_large.tsy`：512×512×512 matmul（推 variant=cublas —— 不 % 128 == 0 之外的 case 也走 cublas；这里 512 % 128 == 0 且 >= 128*2=256 范围外，根据 §4.2 阈值应落到 cublas）

（注：根据 §4.2 阈值，512×512×512 满足 aligned 且 large_enough，但 M*N = 262144 > 256*256 = 65536，所以落 `cublas`。留一条医疗级注释在 .tsy 里写清 intent。）

## 6. 测试矩阵

新增 4 条 ctest（gate 在 `TARGET tsy_runtime_cuda`）。

### 6.1 `pass_schedule_cuda_cases` — 新 C++ driver `tests/passes/test_schedule_cuda_cases.cpp`

每个 shape 一条 sub-case：
| Shape (M×K×N) | 预期 variant | 理由 |
|---|---|---|
| 4 × 4 × 4 | `naive` | M*N=16 < 1024 |
| 128 × 128 × 128 | `tiled` | 对齐 + 在阈值内 |
| 512 × 512 × 512 | `cublas` | M*N > 256² 阈值 |
| 7 × 13 × 11 | `cublas` | odd shape 不对齐 |
| 1 × 32 × 8 | `naive` | M*N=8 < 1024 |

Driver 以 programmatic 方式构造 LIR module（不经 parser），调 `runScheduleCuda`，断言 `attrs["variant"]`.

### 6.2 `cli_emit_lir_schedule_shows_variant` — ctest shell
- `tsc emit-lir --opt=O1 examples/run_matmul_medium.tsy` → PASS_REGULAR_EXPRESSION `variant="tiled"`
- 第二条：`tsc emit-lir --opt=O1 --disable-pass=schedule-cuda examples/run_matmul_medium.tsy` → FAIL_REGULAR_EXPRESSION `variant=` (即禁用后不该打印)

### 6.3 `adapter_cuda_variants_parity` — 扩展 `test_adapter_cuda_cases.cpp`
对 `(64,64,64)` 和 `(128,128,128)` 两个 shape，循环 `{"naive", "tiled", "cublas"}` 三个 variant，每次 adapterMatMul(CPU) vs adapterMatMulCuda(CUDA, variant=X)，atol=1e-4 rtol=1e-3。6 条新 assert。

### 6.4 `cli_bench_smoke` — ctest
```
python benchmarks/run_shapes.py --smoke
```
退出码 0，stdout 含 `variant`。（`--check-scheduler` 不进 ctest：GPU 抖动 + 可能 flaky）

## 7. 风险 + mitigation

| 风险 | Mitigation |
|---|---|
| reg-tiled kernel 移植 bug（共享内存 bank 冲突、register tile 索引、float4 unroll）| 先只移植 + 跑 (64,64,64) 最小 case 做 smoke，parity 过才扩展到 (128,128,128)。失败降级：推迟 tiled 到 W10，scheduler 只在 naive / cublas 之间选 |
| `PassManager` 扩 `LirPassFn` 破坏 W5 的 HIR pass 语义 | 保守扩展：HIR passes 原路径不变、LIR passes 是新容器；`buildPipelineO0` / `buildPipelineO1` 仍返回同样的 `PassManager` 对象，只是 O1 里多两个 LIR 条目 |
| shape 阈值实测后不合理 | bench 出结果后同 PR 调阈值；spec §4.2 数字标"初版占位" |
| `Stmt.attrs` 改 IR schema 影响 codegen/printer/所有 pass | 只在 attrs 非空时才影响 printer / codegen 输出；现有 golden 全部为空 attrs，自动兼容。扩展一个 printer unit test 专门测 attrs 打印 |
| bench 结果抖动（GPU 其他进程）| 3 warmup + 5 measured + median + `--smoke` 不 assert 时延绝对值；`--check-scheduler` 只做相对加速比，不进 ctest |
| Python 脚本环境 | 纯标准库（subprocess / csv / argparse），任何 python 3.8+ 都能跑。ctest 用 `python3` 显式 |
| tsy-bench `.cu` 里调 `adapter_cuda.h` 的 symbol 而不走 LIR 层 | bench 为纯算子级，直接 link tsy_runtime_cuda，调用 `adapterMatMulCuda(..., variant)`。不经 parser / lowering |
| variant="tiled" 在不对齐 shape 上 crash | scheduler pass 负责过滤；但 adapter_cuda 也加 `assert(M%128==0 && N%128==0 && K%8==0)`，错误配也 fail-fast |
| emit-cu 生成的代码在 W8 已有测试不受影响 | 若 attrs 为空则生成和 W8 字节一致的代码（通过 codegen_emit_cu_contains_adapter_calls 回归确认） |

## 8. 验收口径

**同时满足**才算 W9 完成：

1. `cmake --build build -j` 全量 clean（含 `tsy-bench` target）
2. `ctest --test-dir build --output-on-failure` → **29/29 green**
3. `tsc emit-lir --opt=O1 examples/run_matmul_medium.tsy` 输出里可见 `variant="tiled"`
4. `tsc emit-lir --opt=O1 --disable-pass=schedule-cuda examples/run_matmul_medium.tsy` 输出里**没有** `variant=` 字样
5. `./build/tsy-bench --smoke` 成功退出，emit ≥1 行 CSV
6. `python benchmarks/run_shapes.py --smoke` 成功退出
7. `./build/tsy-bench` 全量跑（15 行 CSV）手动检查：1024×1024×1024 上 tiled 的 ms_median 比 naive 至少快 2x（成文档证据，不做成 ctest）
8. W7 MLP 仍跑 `0.4502 0.5498`（无回归）

## 9. 交付物清单

### 新文件
- `src/passes/schedule_cuda.cpp`
- `src/passes/layout_lowering.cpp`（no-op stub）
- `src/tools/tsy-bench.cu`
- `benchmarks/run_shapes.py`
- `tests/passes/test_schedule_cuda_cases.cpp`
- `examples/run_matmul_medium.tsy`
- `examples/run_matmul_large.tsy`
- `docs/superpowers/specs/2026-04-18-tensor-sysy-w9-scheduler-design.md`（本文件）

### 改动文件
- `src/lir/ir.h`（加 `attrs` 字段）
- `src/lir/printer.cpp`（打印 attrs）
- `src/lir/lowering.cpp`（不用改代码，默认空 map 已生效；但检查一下）
- `src/passes/pass_manager.{h,cpp}`（加 `LirPassFn` + `addLir()` + 在 `run()` 里分段执行 + 注册 layout-lowering / schedule-cuda 到 O1 pipeline；声明 runLayoutLowering / runScheduleCuda）
- `src/runtime/adapter_cuda.h`（matmul 签名加 variant 默认参数）
- `src/runtime/adapter_cuda.cu`（两个新 kernel + adapterMatMulCuda 分派 + executor 读 attrs）
- `src/codegen/cuda.cpp`（emit variant string 参数到 matmul call）
- `CMakeLists.txt`（加 `schedule_cuda.cpp` / `layout_lowering.cpp` 到 `tsy_passes`；加 `tsy-bench` target gated in TSY_HAVE_RUNTIME_CUDA）
- `tests/CMakeLists.txt`（4 条新 ctest）

### 预期 commit 划分（每个 ≤ 一个小时工）
1. **feat(lir): add Stmt.attrs + printer support**（仅 LIR schema）
2. **feat(passes): extend PassManager with LirPassFn + stub LayoutLowering/ScheduleCuda**（骨架）
3. **feat(runtime): add matmul naive + reg-tiled kernels with variant dispatch**（kernel 移植 + adapter signature）
4. **feat(passes): implement ScheduleCudaPass shape lookup**（填真正逻辑 + ctest）
5. **feat(tools): tsy-bench C++ binary**（bench 骨架）
6. **feat(tools): benchmarks/run_shapes.py driver + CLI ctest**（Python wrapper + ctest）
7. **feat(codegen): emit variant arg in emit-cu**（codegen 集成 + 回归测试绿）
8. **docs: W9 spec**（本文件入库）

## 10. 不做（重申 YAGNI）

- Permute / View 真实 lowering（W10）
- softmax/rmsnorm 变体选择（YAGNI）
- FP16 / INT8（W11+）
- 多算子 fusion
- 运行时 autotuner
- Bench regression threshold gate
- 跨 GPU device 选择
- numpy / pandas / matplotlib（bench driver 纯标准库）
- benchmark 结果进 PLAN.md 工具表（那是 W11）

---

**前提约束**：所有 CUDA-dependent 组件（tsy-bench、schedule pass、variant kernels、adapter 分派）gate 在 `TSY_HAVE_RUNTIME_CUDA`。非 CUDA 机器上 `cmake` configure 仍成功，W0-W7 + W8-CPU 部分测试（21 + 部分）仍全绿。
