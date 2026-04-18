# W11 — CI + Benchmark · 设计文档

**日期**：2026-04-18
**项目**：tensor-sysy（W10 已落地：transformer block E2E，32/32 ctest 绿）
**对应 PLAN.md 行**：§W11「CI + Benchmark」
**前置**：`docs/superpowers/specs/2026-04-18-tensor-sysy-w10-transformer-design.md`

## 1. 目标

把 tensor-sysy 从「本地能跑」推到「每次改动都有自动化质量门槛」：

1. **CPU CI**：GitHub Actions ubuntu-latest，每次 `push` 到 main / `pull_request` / 手动 dispatch 都跑一次 CPU-path `ctest`。CMake 已有的 `check_language(CUDA)` + `TSY_HAVE_RUNTIME_CUDA` 门控让 CUDA target/test 在无 nvcc 的 CI runner 自然跳过。
2. **本地 GPU baseline**：`tsy-bench` 产出固定 shape + 固定 warmup/run 的 median ms，入 `benchmarks/baseline/rtx3080_wsl.csv`；`scripts/bench_compare.py` 做分 primitive 差异化阈值回归检查：matmul 10% 硬失败 / 5% warning、transformer_block 15% / 10%（端到端测量噪声更大，阈值相应放宽）。
3. **基线文档**：`docs/benchmarks/baseline.md` 解释数据来源、硬件条件、复现步骤、更新规范。

这是 12 阶段 PLAN.md 的**第一次自动化质量门槛**：W0-W10 的正确性纪律依赖手动 `ctest`；从 W11 起 push 到 main 即跑 CI。

## 2. 范围（用户 2026-04-18 五轮确认）

### In-scope

- `.github/workflows/ci.yml`（单 job，ubuntu-latest，CPU-only）
- `tests/CMakeLists.txt` 的 `.venv/bin/python` 绝对路径 → `TSY_PYTHON_EXECUTABLE` CMake cache var
- 根 `CMakeLists.txt` 加 Python 探测 + fallback
- `.gitignore` 加 `tests/e2e/__pycache__/`（并 `git rm -r --cached` 若已 track）
- 所有 CUDA adapter kernel launch 后、D2H 前补 `cudaDeviceSynchronize`（W8/W9/W10 全量审）
- `src/hir/verifier.cpp:verifyUnary` 注释补一行约束
- `src/tools/tsy-bench.cu` 加 `--primitive {matmul,transformer_block}` flag
- 新增 `benchmarks/baseline/rtx3080_wsl.csv`（18 行基线）
- 新增 `scripts/bench_compare.py`（stdlib only）
- 新增 `scripts/bench_local.sh`（一键复现 baseline diff）
- 新增 `docs/benchmarks/baseline.md`
- `PLAN.md §W11` 打勾 + 必要的已装工具记录

### Out-of-scope（YAGNI，推后）

- self-hosted runner / 云端 GPU runner
- GHA 装 nvcc 做编译期 CUDA 回归（本地构建兜底；W12 或之后评估）
- OS matrix（只 ubuntu-latest）
- cron nightly job
- 在 GHA 里跑 baseline diff（CI 无 GPU，跑不出 cuda 变体）
- odd shape / softmax / rmsnorm micro-bench（baseline 只要 matmul + transformer_block E2E）
- Release 产物构建 / 发布流程
- markdown baseline 表双写（CSV 是脚本消费源，baseline.md 只讲复现）
- 多硬件 baseline（只 `rtx3080_wsl.csv` 一份）

## 3. 架构

```
┌─────────────────────┐       ┌──────────────────────────┐
│  GitHub Actions     │       │  本地 WSL + RTX 3080     │
│  (ubuntu-latest,    │       │                           │
│   no GPU, no nvcc)  │       │  scripts/bench_local.sh  │
│                     │       │    ↓                      │
│  apt + uv venv      │       │  ./build/tsy-bench       │
│    ↓                │       │    --primitive matmul    │
│  cmake build        │       │    --primitive           │
│    (CUDA skipped)   │       │      transformer_block   │
│    ↓                │       │    ↓ (CSV 18 rows)       │
│  ctest (~20 pass)   │       │  bench_compare.py        │
│                     │       │    vs baseline.csv       │
│                     │       │    10% fail / 5% warn    │
└─────────────────────┘       └──────────────────────────┘
```

**CI 与本地的分工是明确的**：CI 只看 CPU-path 回归，GPU 回归靠本地 baseline diff。两条路径独立，不互相阻塞。

## 4. 组件详表

### 4.1 `.github/workflows/ci.yml`

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

      - name: Configure
        run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-13

      - name: Build
        run: cmake --build build -j

      - name: Test
        run: ctest --test-dir build --output-on-failure
```

CMake 的 `check_language(CUDA)` 找不到 nvcc → `TSY_HAVE_RUNTIME_CUDA=0` → 所有 `tsy_runtime_cuda`、`tsy_add_cuda_example`、`tsy-bench`、CUDA-only ctest 全部自动跳过。CPU-path 预期剩约 20 项 ctest：parse / diagnostics / shape / adapter_cpu / emit-cpp / pytest e2e 的 native + cpu-adapter 分支。

**不加 cache**（初版）：GHA 免费层 cache 配置复杂度 vs 收益不匹配（build 本身 2-3min），留给后续。

### 4.2 `tests/CMakeLists.txt` + 根 CMake 的 Python 路径

**现状**（W10 reviewer #1）：`tests/CMakeLists.txt` 里 `.venv/bin/python` 硬编码为绝对或相对路径，GHA runner 上根本不存在。

**改法**：

根 `CMakeLists.txt` 在 `project()` 后加：
```cmake
# Python for e2e / pytest driven tests.
# Prefer repo-local .venv if present, else fall back to system Python3.
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

`tests/CMakeLists.txt` 里所有 `.venv/bin/python` 字面量替换为 `${TSY_PYTHON_EXECUTABLE}`。并在测试注册前加一次运行时探测：
```cmake
execute_process(
    COMMAND ${TSY_PYTHON_EXECUTABLE} -c "import numpy, pytest"
    RESULT_VARIABLE _py_probe
    OUTPUT_QUIET ERROR_QUIET)
if(NOT _py_probe EQUAL 0)
    message(FATAL_ERROR
        "TSY_PYTHON_EXECUTABLE=${TSY_PYTHON_EXECUTABLE} lacks numpy/pytest.\n"
        "  uv pip install --python ${TSY_PYTHON_EXECUTABLE} numpy pytest")
endif()
```

### 4.3 `.gitignore`

- 加 `tests/e2e/__pycache__/`
- 实现阶段先 `git ls-files tests/e2e/__pycache__/` 确认是否误 track；若已 track，`git rm -r --cached tests/e2e/__pycache__/` 再提交

### 4.4 CUDA kernel D2H 前同步

**现状**（W10 reviewer #2）：`adapter_cuda*.cu` 里每个 adapter 的 pattern 是：
```cpp
kernel<<<grid, block>>>(...);
cudaMemcpy(host, device, n, cudaMemcpyDeviceToHost);  // ← kernel 可能还没跑完
```

严格讲 `cudaMemcpy(cudaMemcpyDeviceToHost)` 是阻塞同步的，**会隐式等待同流 pending kernel**。所以正确性上现状不挂。但：

1. 显式 `cudaDeviceSynchronize()` 让意图清晰、和 `cudaEvent` timing 协同（bench 侧必须显式同步才能拿到可信 ms）
2. 如果后续换 async memcpy（`cudaMemcpyAsync`）这个隐式保证就没了
3. 统一 pattern 方便 review

**改法**：`src/runtime/adapter_cuda*.{cpp,cu}` 每个 adapter function 内，kernel launch 之后、cudaMemcpy D2H 之前，加 `CUDA_CHECK(cudaDeviceSynchronize());`。总点位预计 6-8 处（W8 matmul/add/softmax/rmsnorm、W9 无新 kernel、W10 transpose/relu）。

### 4.5 `src/hir/verifier.cpp:verifyUnary` 注释

**现状**（W10 reviewer #3）：`verifyUnary` 当前对 softmax / rmsnorm / relu 复用同一模板（result shape = input shape），但没有写清楚「这条假设对所有 unary 成立是因为它们都是 elementwise 形状保留 op」。

**改法**：在 `verifyUnary` 函数定义上方加一行：
```cpp
// NOTE: verifyUnary assumes the op preserves input shape element-wise.
// Do NOT reuse for shape-changing unary ops (e.g. transpose, reduce).
// Those must call their own verifier.
```

### 4.6 `src/tools/tsy-bench.cu` 的 `--primitive` flag

**CLI 扩展**：
```
tsy-bench [--primitive matmul|transformer_block]
          [--smoke]
          [--shapes MxKxN[,...]]  (matmul only)
          [--variants v1[,v2,...]]  (matmul only)
```

默认 `--primitive matmul`，向后兼容（`run_shapes.py` / `tsy-bench --smoke` 一行不改）。

**实现结构**：`tsy-bench.cu` 的 `main()` 解析出 `primitive` 后分派：
- `primitive == "matmul"`：保持现有 matmul sweep 逻辑不变
- `primitive == "transformer_block"`：调用新函数 `runTransformerBlockBench()`

`runTransformerBlockBench()`：
- 硬编码 W10 的 shape：S=4, D=8, F=16（和 `examples/transformer_block.tsy` / `tests/e2e/test_transformer_block.py` 一致）
- 三个 backend 各跑一次 forward end-to-end，3 warmup + 5 run + median：
  - `native`：调 `runModuleNative(module)`，用 `std::chrono::steady_clock`
  - `cpu_adapter`：调 adapter_cpu executor，用 `std::chrono`
  - `cuda_adapter`：调 adapter_cuda executor，cudaEvent 包裹（start → forward → end → elapsed → synchronize）
- 每 backend 输出一行 CSV：`transformer_block,4,8,16,<backend>,<median_ms>,0`（gflops 列填 0，transformer 端到端 GFLOPS 计算无标准约定，留空义务更清楚）

**link 方式**：transformer_block 的 forward 逻辑通过 `tsy_add_cuda_example(transformer_block_cuda examples/transformer_block.tsy)` 生成 .cu 产物。为了让 tsy-bench 复用这份逻辑而不是把 example 二进制 `dlopen` 进来，采用：

1. CMake 里把 transformer_block 的生成产物同时编译成 static lib `tsy_example_transformer_block`（`add_library(... STATIC ...)`），头文件导出 `forward(...)` 签名
2. `tsy-bench` `target_link_libraries(... tsy_example_transformer_block ...)`

如果 example 产物 link 模型过复杂（比如 codegen 生成的 `main()` 冲突），退路是在 `src/tools/` 里单独写一份 transformer_block 的 host harness（W10 已有的 runtime API 够用）。实现阶段 pick 简单的那条。

### 4.7 `benchmarks/baseline/rtx3080_wsl.csv`

18 行，CSV header 沿用 `tsy-bench` 现有格式：

```csv
primitive,M,K,N,variant,ms_median,gflops
matmul,256,256,256,naive,<measured>,<measured>
matmul,256,256,256,tiled,<measured>,<measured>
matmul,256,256,256,cublas,<measured>,<measured>
matmul,512,512,512,naive,...
... (15 matmul rows, covering 256/512/1024/1536/2048 cubed × 3 variants)
transformer_block,4,8,16,native,<measured>,0
transformer_block,4,8,16,cpu_adapter,<measured>,0
transformer_block,4,8,16,cuda_adapter,<measured>,0
```

具体数值在实现阶段由 `tsy-bench` 在已修完 W10 follow-up #2（D2H 前同步）之后采集，而不是先采集后修同步——否则 baseline 首次提交即自相矛盾。

### 4.8 `scripts/bench_compare.py`

```python
#!/usr/bin/env python3
"""Compare current bench output vs baseline CSV.

Usage:
    scripts/bench_compare.py --baseline <b.csv> --current <c.csv>
    scripts/bench_compare.py --baseline <b.csv> --current <c.csv> --update-baseline
"""
```

- 输入：两份 CSV，header 同 `tsy-bench` 输出
- key：`(primitive, M, K, N, variant)`
- metric：`ms_median`
- thresholds（dict by primitive）：
  - `matmul`：fail ≥1.10, warn ≥1.05
  - `transformer_block`：fail ≥1.15, warn ≥1.10
- 状态分类：
  - `IMPROVED`：ratio < 0.95（print + 建议 `--update-baseline`，不 fail）
  - `OK`：0.95 ≤ ratio < warn
  - `WARN`：warn ≤ ratio < fail
  - `FAIL`：ratio ≥ fail → exit 1
- `--update-baseline`：把 current 直接覆盖 baseline 路径，打印「请 git add + commit」
- stdlib only（`csv` + `argparse` + `sys` + `pathlib`）
- 缺 key（baseline 有但 current 没有，或反之）：print warning，不影响 exit code

### 4.9 `scripts/bench_local.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake --build build -j
./build/tsy-bench --primitive matmul > /tmp/bench_current.csv
./build/tsy-bench --primitive transformer_block >> /tmp/bench_current.csv
# strip duplicate CSV header from the second append
awk '!/^primitive,M,K,N,variant/ || NR==1' /tmp/bench_current.csv > /tmp/bench_dedup.csv
python3 scripts/bench_compare.py \
    --baseline benchmarks/baseline/rtx3080_wsl.csv \
    --current /tmp/bench_dedup.csv
```

（「append 后去重 header」是因为 tsy-bench 每次都输出 CSV header；实现时也可以加 `--no-header` flag 让第二次调用不打 header。pick 简单的那条。）

### 4.10 `docs/benchmarks/baseline.md`

内容大纲：

- **硬件环境**：CPU 型号、GPU 型号（RTX 3080 Laptop 16GB）、driver 版本、CUDA toolkit 版本（12.0）、OS（WSL2 Ubuntu 24.04）
- **软件环境**：CMake / g++ / bison / flex / nvcc / Python 版本（实际从 `bench_local.sh` 运行时打印一次采集）
- **物理条件**：接电源 + 高性能模式 + 关闭无关负载（浏览器、大后台进程）
- **复现步骤**：git clone → build → `bash scripts/bench_local.sh` → 若 0 FAIL 0 WARN 即复现成功
- **基线更新触发条件**：
  - 工具链升级（CUDA toolkit / driver 更新 → 数字会变）
  - 调度策略改变（W9 scheduler 或后续 LayoutLoweringPass 真正落地）
  - 主动性能工作（某阶段专门优化某 kernel）
- **基线更新流程**：`bash scripts/bench_local.sh` 失败 → 确认改动意图（优化 or 回归）→ 若优化：`python3 scripts/bench_compare.py ... --update-baseline` → commit CSV + 在 commit message 里说明数值变化

### 4.11 `PLAN.md` 更新

- §W11 行打勾、补上实际验收命令清单（见 §5）
- 若 W11 装了新工具（预计无——uv 已装、GHA 不装新东西），补到「已装工具」表

## 5. 验收

### 5.1 本地必过

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy

# [1] 全 ctest 不回归（CPU + CUDA 全绿）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
# 预期：32/32 pass

# [2] CI 等价（强制 CPU-only，模拟 GHA runner）
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE
cmake --build build-cpu -j
(cd build-cpu && ctest --output-on-failure)
# 预期：~20/20 CPU-path tests pass

# [3] baseline 自比对
bash scripts/bench_local.sh
# 预期：0 FAIL 0 WARN（baseline 自己和自己比）

# [4] noise sanity check
./build/tsy-bench --primitive matmul > /tmp/a.csv
./build/tsy-bench --primitive matmul > /tmp/b.csv
python3 scripts/bench_compare.py --baseline /tmp/a.csv --current /tmp/b.csv
# 预期：全 OK/WARN，无 FAIL（验证 10% 阈值相对跨跑噪声够宽）

# [5] W9 scheduler guard 继续有效
python3 benchmarks/run_shapes.py --check-scheduler
# 预期：tiled/naive speedup ≥ 1.2x
```

### 5.2 CI 必过

- 第一次 push 到 main（或开 PR）后，Actions tab 里 CI workflow 绿
- 日志里 CPU-path ctest ~20 项全 pass
- 后续 PR 每次都跑、且可阻塞 merge

### 5.3 文档必过

- 新开一个 terminal，按 `docs/benchmarks/baseline.md` 的复现步骤 cold start → `bash scripts/bench_local.sh` 0 FAIL
- `PLAN.md §W11` 行打勾可见

## 6. 风险 & 缓解

| 风险 | 缓解 |
|---|---|
| baseline 数字在不同 power state / thermal 下抖动 | `tsy-bench` 已 3 warmup + 5 run + median；`baseline.md` 明确「接电 + 高性能 + 关浏览器」产出条件；阈值 10% 留了充分余量 |
| 修 D2H 前同步（#4.4）会让现有 bench 数字变化（测量范围变真实） | 顺序：先改 sync 代码 → 跑 tsy-bench 采集 baseline → commit baseline。反过来基线首次入库即自相矛盾 |
| `TSY_PYTHON_EXECUTABLE` 改后 ctest 找不到 numpy/pytest | `tests/CMakeLists.txt` 加 import 探测 + `FATAL_ERROR` 带人类可读修复提示 |
| GHA ubuntu-latest 工具链和本地 drift（bison/flex/g++ 版本差异致 parser 生成差异） | CI 步骤 print 所有工具版本便于定位；ubuntu-latest 当前 24.04（g++-13、bison 3.8、flex 2.6）和本地对齐 |
| `tsy-bench` link transformer_block 产物的 CMake 集成可能踩坑（codegen 生成的 `main()` 冲突） | 实现阶段若 static lib 方案不通，退路是在 `src/tools/` 里写一份独立 harness 复用 runtime API |
| CI 挂了 block 所有 PR | ubuntu-latest + CPU-only 是最稳组合；失败先查 apt / uv cache；workflow_dispatch 手动重跑兜底 |
| `--update-baseline` 被误用刷掉真实回归 | `bench_compare.py` 打印 `IMPROVED` 时提示「只在确认是性能优化而非测量范围变化时才刷」；commit message 规范要求说明数值变化原因 |

### 6.1 不解决但明确承认的限制

- CI **不防** GPU 路径回归（无 self-hosted runner / 云 GPU）；靠本地 `bench_local.sh` + baseline diff
- CI **不防** 编译期 CUDA 语法错误（未装 nvcc）；本地构建兜底
- baseline 只代表 **这一台机 + 这一次物理条件**；别人复现会不一样，这是 baseline 本质
- baseline 没有历史版本管理（只记当前 ground truth，git log 即历史）

## 7. 实施顺序建议

纯依赖顺序（不是时间估算）：

1. **W10 follow-up #3**（verifier 注释）—— 零风险，先清单
2. **W10 follow-up #1**（Python 路径 CMake 化）—— 本地 `ctest` 必须继续绿
3. **W10 follow-up #2**（CUDA sync）—— 改完跑 `ctest` 确认不挂；**此时 bench 数字会变**
4. **tsy-bench `--primitive` + transformer_block**（§4.6）—— 新增能力
5. **采集 baseline CSV**（§4.7）—— 必须在 3 之后，4 完成时
6. **`bench_compare.py` + `bench_local.sh`**（§4.8-4.9）—— 本地工具
7. **`docs/benchmarks/baseline.md`**（§4.10）—— 文档可以边做边写
8. **`.github/workflows/ci.yml`**（§4.1）—— 独立于 GPU 侧；任何时候可加；放最后是因为要在本地所有东西绿之后推上去
9. **`PLAN.md §W11` 打勾 + commit** —— 收官

每步单独 commit 更清楚；2、3、4 每步后 `ctest` 确认绿。
