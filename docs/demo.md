# TensorSysY — Demo 复现指南

这一份文档的目标：**照着从上到下跑一遍，得到 README 首页声称的所有
结果**。每一条命令都标了预期的输出片段，跟实际输出对不上就说明环境
有问题，先排查再往下。

## 0. 先决条件

| 工具 | 最低版本 | 检查命令 | 备注 |
|---|---|---|---|
| CMake | 3.20 | `cmake --version` | Release 构建 |
| C++ 编译器 | C++17 | `c++ --version` | 项目用 gcc-13 |
| flex / bison | —  | `flex --version && bison --version` | 前端生成 |
| CUDA toolkit | 12.x | `nvcc --version` | 可选，没有则跳过 CUDA 段 |
| Python | 3.10+ | `python3 --version` | E2E pytest 和 numpy 参考 |
| uv | 0.11+ | `uv --version` | Python 依赖管理 |
| NVIDIA GPU | 8GB+ VRAM | `nvidia-smi -L` | 验收 baseline 录在 RTX 3080 Laptop |

## 1. 克隆 / 准备

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy   # 这是本仓库的本地路径
```

建立 `.venv`（E2E pytest 需要 numpy/pytest；ctest 通过 `.venv/bin/python`
调用）：

```bash
uv venv
uv pip install --python .venv/bin/python numpy pytest
```

**验证**：`.venv/bin/python -c "import numpy, pytest; print(numpy.__version__, pytest.__version__)"`
应打印版本号，不报错。

## 2. 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**预期产物**：

- `build/tsc` — 编译器 CLI
- `build/out/mlp` — CPU MLP demo
- `build/out/matmul_cuda_demo` — CUDA 单算子 demo
- `build/out/transformer_block` — CPU transformer block
- `build/out/transformer_block_cuda` — CUDA transformer block

没有 CUDA 的环境，后两个 `*_cuda` 产物不会构建，cmake 阶段会打印
`-- TSY_ENABLE_CUDA=OFF`。其余一切正常。

## 3. 全量测试 — W0–W11 一键验收

```bash
ctest --test-dir build --output-on-failure
```

**预期**：有 CUDA 的机器 `100% tests passed, 0 tests failed out of 32`；
无 CUDA 约 `out of 20`（`adapter_cuda_*` / `codegen_cuda_*` /
`cli_cuda_*` / `pass_schedule_cuda_*` / `*_cuda` 自动跳过）。

总耗时参考：RTX 3080 Laptop / WSL2 / Release 下 ~5 秒。

## 4. 按里程碑逐项复现（可选 —— 与 §3 重叠，但便于对号入座）

```bash
# W0: CLI + parse smoke
./build/tsc --help
./build/tsc parse    examples/smoke.tsy        # "parse ok: examples/smoke.tsy"
./build/tsc dump-ast examples/smoke.tsy | head

# W1: 前端 + 张量语法
ctest --test-dir build -R parse --output-on-failure      # 4/4

# W2: AST → HIR lowering
./build/tsc emit-hir examples/matmul.tsy                  # 看到 matmul/return

# W3: type/shape/verifier
ctest --test-dir build -R shape --output-on-failure      # 1/1

# W4: LIR + interpreter
./build/tsc run-lir examples/run_matmul_tiny.tsy          # 打印 A, B, C 张量

# W5: passes（O1 + DCE）
./build/tsc emit-hir --opt=O1 examples/dead_matmul.tsy   # dead_block 中的 matmul 被 dce 掉

# W6: CPU runtime adapter
ctest --test-dir build -R adapter_cpu --output-on-failure  # 2/2

# W7: CPU codegen 闭环
./build/out/mlp                                           # MLP 前向，打印中间张量

# W8: CUDA adapter + 单算子 codegen
./build/out/matmul_cuda_demo                              # 与 CPU 等价的 matmul 输出
ctest --test-dir build -R "adapter_cuda|codegen_cuda|cli_cuda"  # 3/3

# W9: scheduler + layout lowering
.venv/bin/python benchmarks/run_shapes.py --check-scheduler
# 末行应类似：1024^3: tiled/naive speedup = 1.2x+ (min required 1.20x)
ctest --test-dir build -R "schedule|transpose_relu"       # 4/4

# W10: transformer block e2e
./build/out/transformer_block                             # CPU
./build/out/transformer_block_cuda                        # CUDA
.venv/bin/python -m pytest tests/e2e/ -q                  # 3 passed

# W11: CI + 本地 bench gate
ctest --test-dir build --output-on-failure                # 32/32
bash scripts/bench_local.sh                               # 目标 0 FAIL
```

## 5. CI 等价运行（模拟 GHA 的 CPU-only runner）

GHA runner 没有 CUDA，但有 `.venv` 和 numpy/pytest——因为 workflow
会先装。完整流程直接照抄 `.github/workflows/ci.yml`：

```bash
# 干净隔离：备份当前 .venv 和 build，保证从零
mv .venv .venv.bak 2>/dev/null || true
rm -rf build-cpu

# Step 1: 建 venv 并装依赖（GHA 的 "Set up Python venv" step）
uv venv .venv
uv pip install --python .venv/bin/python numpy pytest

# Step 2: 配置 + 构建（GHA 的 "Configure" + "Build" steps）
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j

# Step 3: 跑测试（GHA 的 "Test" step）
(cd build-cpu && ctest --output-on-failure)

# 收尾：还原之前的 .venv
rm -rf .venv && mv .venv.bak .venv 2>/dev/null || true
rm -rf build-cpu
```

**预期**：本地有 CUDA 时 CMake 会探到 nvcc，因此构建的是全量 32/32
测试；真正的 GHA 跑在 CPU-only 机器上是 ~20/20（所有 `*_cuda*`
条目不注册）。要真复现 CPU-only 行为，需要把 `nvcc` 从 PATH 摘掉
（超出本地验证范围——以 GHA 跑的数字为准）。

**为什么不能偷懒不重建 `.venv`**：CMake 在 `tests/CMakeLists.txt:271`
会 probe `TSY_PYTHON_EXECUTABLE` 能否 `import numpy, pytest`。如果
`.venv` 缺失它会回落到 `/usr/bin/python3`，而系统 Python 默认没
这两个包——configure 就会 FATAL_ERROR。这是有意的安全门，避免
"e2e 测试静默跳过"。

## 6. Benchmark 门禁

```bash
bash scripts/bench_local.sh
```

**预期**：

- 1024³ matmul × {naive, tiled, cublas} 三行。
- FAIL 判定阈值 ±10%、WARN ±5%。
- 因笔记本 GPU / WSL2 的物理噪声，**允许最多 3 次重跑命中一次
  0 FAIL** —— 详见 `docs/benchmarks/baseline.md` 的 "Noise and
  acceptance policy" 段。

如果需要 scheduler 证据：

```bash
.venv/bin/python benchmarks/run_shapes.py --check-scheduler
```

**预期**：末行 `1024^3: tiled/naive speedup = 1.2x+`（门槛 1.20×）。

如果确认性能真的变好了（不是噪声）、想重录 baseline：

```bash
# 连续跑 5 次确认都 IMPROVED
for i in 1 2 3 4 5; do bash scripts/bench_local.sh | tail -5; done
# 再写入 baseline
bash scripts/bench_local.sh --update-baseline
```

（`--update-baseline` 由 `bench_compare.py` 的 hint 指导，不是每天都该按的按钮。）

## 7. 故障排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `tsc: command not found` | 忘了 `cmake --build` 或构建被拒 | 重新 build，检查 flex/bison 是否装了 |
| ctest 有 `*_cuda*` 全 skip | `nvcc` 不在 PATH / cmake 探测失败 | `nvidia-smi -L` 看 GPU；装 CUDA 12.x toolkit |
| `e2e_transformer_block_pytest` FAIL 但单跑 `pytest` 过 | ctest 拿不到 `TSY_PYTHON_EXECUTABLE` | 重新 configure：`cmake -S . -B build -DTSY_PYTHON_EXECUTABLE=$(pwd)/.venv/bin/python` |
| `bench_local.sh` 连续 3 次都 FAIL 且 ratio 稳定 >1.15 | 真回归 | 先 git bisect 上一个绿 commit；再看 `benchmarks/run_shapes.py` 明细 |
| `bench_local.sh` 1-2 次 FAIL 夹带 1 次 OK | 噪声 | 记录为 noise，不是回归 |
| `run_shapes --check-scheduler` < 1.20× | 真回归（scheduler 没工作）| 检查 `schedule-cuda` pass 是否被 `--disable-pass` 影响到；看 `emit-lir` 是否仍带 variant |

## 8. 产物清单（跑完这份文档应有的东西）

- `build/tsc` 能跑 §4 所有命令。
- `build/out/{mlp, matmul_cuda_demo, transformer_block, transformer_block_cuda}` 能独立运行。
- `ctest` 在本地机器 32/32，在 CPU-only 模拟 ~20/20。
- `bench_local.sh` 3 次内命中 0 FAIL。
- `benchmarks/baseline/rtx3080_wsl.csv` 未被修改（除非你按了 `--update-baseline`）。

这份文档跑不通 = W12 的"文档命令全跑通"验收没通过。请开 issue 或
本地排查。
