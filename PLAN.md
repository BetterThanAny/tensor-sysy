# TensorSysY：执行计划 v2

这是按两个现有仓库真实情况修正后的版本。核心调整：

1. **不再假设 `mini-llm-engine` 是通用 runtime**
   先做一层 `runtime adapter`，把 TensorSysY 的 HIR/LIR 映射到它现有的 CPU/CUDA 算子接口。
2. **不再假设 `cuda-kernels/*.cu` 可直接链接**
   GPU 主路径优先复用 `mini-llm-engine/src/ops_cuda.*`；`cuda-kernels/` 只作为独立 kernel 验证和性能参考。
3. **测试从"能跑"升级为"能定位问题"**
   增加诊断测试、adapter 测试、interpreter/codegen 三角对拍、幂等性测试、layout/aliasing 测试。

---

## 架构总览 v2

```text
.tsy source
   │  (flex/bison, from sysy-compiler frontend, extended)
   ▼
AST + SourceLocation
   │  (type check, shape infer, builtin validation)
   ▼
HIR
   │  ops: MatMul / Add / Softmax / RMSNorm / View / Permute / Const / FuncCall
   │  passes: verify / constant-fold / DCE / fusion / layout-lowering
   ▼
LIR
   │  loops / buffers / loads / stores / calls
   │
   ├── CPU path:
   │     LIR -> runtime-adapter -> mini-llm-engine/src/ops_cpu.*
   │
   └── CUDA path:
         LIR -> runtime-adapter -> mini-llm-engine/src/ops_cuda.*
         cuda-kernels/ only for isolated kernel validation & benchmarking
```

---

## 先改的原则

### 范围收缩
前 6 周只支持这 4 个核心算子：

- `matmul`
- `add`
- `softmax`
- `rmsnorm`

`view / permute` 只做最小必要子集，默认要求静态 shape、连续布局优先。

### 交付优先级
- **P0**：W7 前拿到 CPU 端到端闭环
- **P1**：W9 前拿到单算子 CUDA 闭环
- **P2**：W10 做 transformer block 对拍
- **P3**：性能和博客可以压缩，不影响项目成立

---

## 12 周计划 v2

| 周 | 主题 | 交付物 | 验收命令 | 必测项 |
|---|---|---|---|---|
| **W0** | 基建 + 诊断骨架 | 新仓 `tensor-sysy`、CMake、CLI、测试框架、位置追踪 `SourceLocation` | `ctest` 可跑；`tsc --help` 正常 | CLI smoke、错误信息 smoke |
| **W1** | 前端迁移 + 张量语法 | 从 `sysy-compiler` 迁 `sysy.l/y`；支持 `tensor<f32>[M,N]`、`@matmul/@add/@softmax/@rmsnorm` | `ctest -R parse` | 原 SysY 回归、张量 parse、诊断测试 |
| **W2** | AST + HIR | AST 扩展、HIR 节点、printer、parser->HIR lowering | `tsc --emit-hir examples/matmul.tsy` | AST/HIR golden、round-trip printer |
| **W3** | Type/Shape/Verifier | shape infer、builtin 参数校验、报错定位到行列 | `ctest -R shape` | 正/负例、错误位置断言、边界 shape |
| **W4** | LIR + HIR/LIR 解释器 | LIR 定义、naive interpreter、numpy 参考脚本 | `tsc --run-lir examples/matmul.tsy` | interpreter vs numpy、small/odd shape |
| **W5** | Pass 基建 | PassManager、Verifier pass、ConstFold、DCE | `tsc --opt=O1 --emit-hir` | before/after 语义等价、幂等性、禁用 pass |
| **W6** | Runtime Adapter (CPU) | `runtime/adapter_cpu.*`，把 HIR/LIR 调到 `ops_cpu.*` | `ctest -R adapter_cpu` | matmul/layout、softmax 维度、rmsnorm 末维、add 语义 |
| **W7** | CPU Codegen 闭环 | 生成 C++ host 代码，链接 adapter + `ops_cpu.cpp`，跑通 MLP | `tsc examples/mlp.tsy --emit-cpp && cmake --build build && ./out/mlp` | interpreter/codegen 对拍、PyTorch 对拍、编译 smoke |
| **W8** | CUDA Adapter + 单算子 Codegen | `runtime/adapter_cuda.*`，复用 `ops_cuda.*` 跑单算子 | `tsc examples/matmul.tsy --target cuda ...` | CPU/GPU 容差对拍、单算子编译运行 |
| **W9** | CUDA 调度和 layout lowering | 初版 scheduler、layout lowering、shape 查表策略 | `python benchmarks/run_shapes.py` | 5 类 shape 数值对拍、调度覆盖、odd shape |
| **W10** | Transformer Block E2E | `examples/transformer_block.tsy`，attention + ffn 跑通 | `pytest tests/e2e/test_transformer_block.py` | vs PyTorch、vs mini-llm-engine 子路径 |
| **W11** ✅ 2026-04-18 | CI + Benchmark | GHA CPU CI、本地 GPU bench 脚本、1024³ matmul baseline + bench_compare | GitHub Actions + `bash scripts/bench_local.sh` 0 FAIL | 稳定性、回归阈值、中位数策略 |
| **W12** | 收口 | README、架构图、博客、demo | 文档可复现 | 文档命令全跑通 |

---

## W0：48 小时版本 v2

### D1
1. 建仓 `tensor-sysy`
2. 建 CMake + `src/` + `tests/`
3. 从 `sysy-compiler` 迁入 `sysy.l/y`
4. 把当前 `main.cpp` 的固定课程式 CLI 改成可扩展子命令风格：
   - `tsc parse`
   - `tsc dump-ast`
   - `tsc emit-hir`
   - `tsc run-lir`

### D2
1. 加 `SourceLocation`
2. 开 parser/semantic error 诊断通路
3. 接入 gtest/pytest/ctest
4. 配 GitHub Actions 跑：
   - parse tests
   - golden tests
   - unit tests

### W0 验收命令
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/tsc parse examples/smoke.tsy
./build/tsc dump-ast examples/smoke.tsy
```

---

## 目录结构 v2

```text
tensor-sysy/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── frontend/
│   │   ├── sysy.l
│   │   ├── sysy.y
│   │   ├── ast.{h,cpp}
│   │   ├── location.h
│   │   └── diagnostics.{h,cpp}
│   ├── hir/
│   │   ├── ops.{h,cpp}
│   │   ├── types.{h,cpp}
│   │   ├── verifier.cpp
│   │   ├── shape.cpp
│   │   └── printer.cpp
│   ├── lir/
│   │   ├── ir.{h,cpp}
│   │   ├── printer.cpp
│   │   └── interpreter.cpp
│   ├── passes/
│   │   ├── pass_manager.{h,cpp}
│   │   ├── verify.cpp
│   │   ├── const_fold.cpp
│   │   ├── dce.cpp
│   │   └── fusion.cpp
│   ├── runtime/
│   │   ├── adapter_cpu.{h,cpp}
│   │   ├── adapter_cuda.{h,cpp}
│   │   └── tensor_view.h
│   ├── codegen/
│   │   ├── cpp.cpp
│   │   └── cuda.cpp
│   └── tools/
│       └── tsc.cpp
├── examples/
│   ├── matmul.tsy
│   ├── mlp.tsy
│   └── transformer_block.tsy
├── tests/
│   ├── parse/
│   ├── diagnostics/
│   ├── golden/
│   ├── shape/
│   ├── adapter/
│   ├── passes/
│   ├── codegen/
│   └── e2e/
├── scripts/
│   ├── update_golden.sh
│   ├── compare_numpy.py
│   ├── compare_pytorch.py
│   └── run_on_cloud.sh
└── third_party/
    ├── sysy-compiler-ref/
    └── mini-llm-engine-ref/
```

---

## 测试方案 v2

这部分是 v2 的核心。

### L0 — Smoke / Build
每次提交都要覆盖：

- `tsc` 能启动
- 示例 `.tsy` 能 parse
- codegen 产物能编译
- 基础脚本能运行

### L1 — Golden
覆盖：

- AST
- HIR
- LIR
- Diagnostics 文本

示例目录：

```text
tests/golden/matmul_basic/
├── input.tsy
├── expected.ast.txt
├── expected.hir.txt
├── expected.lir.txt
└── expected.diag.txt
```

### L2 — Frontend / Verifier
必须补你原计划里缺的这几类：

- 原 SysY 程序回归
- 张量语法合法/非法
- builtin 参数个数错误
- dtype 错误
- rank mismatch
- shape mismatch
- 非法 `view`
- 非法 `permute`
- 错误定位行列断言

### L3 — Adapter 测试
单独验证 TensorSysY 语义和 `mini-llm-engine` 接口是否对齐。

必须有：

- `matmul`: `A[M,K] @ B[N,K]^T`
- `softmax`: 最后一维语义
- `rmsnorm`: 只沿 hidden dim
- `add`: elementwise；如果支持 broadcast，要明确测
- layout 不满足预期时是否显式报错或物化复制

### L4 — Pass 测试
每个 pass 至少 4 类断言：

- 结构变化正确
- 语义不变
- 重跑一次结果不变
- `--disable-pass=X` 能回退

### L5 — Interpreter / Codegen 三角对拍
这是 v2 新增重点。

同一 `.tsy`，比较：

- HIR/LIR interpreter
- CPU codegen
- CUDA codegen

规则：

- CPU 与 interpreter：`1e-6`
- CUDA 与 CPU：`atol=1e-4, rtol=1e-3`
- 不要求 bit-exact

### L6 — E2E 对拍
覆盖：

- `mlp.tsy` vs PyTorch
- `transformer_block.tsy` vs PyTorch
- 子路径对拍 `mini-llm-engine` 中已有实现

### L7 — 性能回归
先做轻量版：

- 固定 shape 集合
- warmup 3 次
- run 5 次
- 取 median
- 退化超过 5% 标黄，超过 10% 标红

---

## 最少必补的测试清单

下面这组是原 v1 缺得最明显的，建议直接写进里程碑定义：

### 前端
- 原 `SysY` 全量回归
- 张量关键字不影响旧语法
- 词法非法字符报错
- parser 错误恢复或至少报对位置

### 类型/shape
- `matmul([M,K],[K,N])` 合法
- `matmul([M,K],[N,K])` 因布局约定非法或需要转置，行为明确
- `softmax` 对最后一维
- `rmsnorm` hidden dim 一致
- 1x1、1xK、Kx1、非 2 次幂 shape

### pass
- DCE 不删副作用节点
- ConstFold 不改变 dtype/shape
- fusion 不跨越非法 dependency
- pass 顺序交换后的 verifier 仍通过

### codegen
- 生成代码编译成功
- 生成代码运行成功
- 缺 runtime symbol 时有清晰错误
- `--target cpu/cuda` 产物路径稳定

### runtime adapter
- stride/layout 不符合约定时显式失败
- aliasing/in-place 不覆盖活值
- add residual 路径正确
- host/device 拷贝边界正确

### e2e
- MLP 前向
- single-head attention toy case
- transformer block
- 固定 seed 的随机输入对拍

---

## 风险与对策 v2

| 风险 | 对策 |
|---|---|
| `sysy-compiler` 前端改动太大把旧功能打坏 | W1 强制保留旧 SysY 回归测试 |
| `mini-llm-engine` 接口不够通用 | W6 单独做 adapter，不把 HIR 直接绑死到现有 API |
| GPU 路径返工 | W8 前不碰复杂 fusion，先单算子闭环 |
| 诊断做晚了返工 | `SourceLocation` 前移到 W0 |
| 性能 benchmark 噪声太大 | 固定 shape、固定 warmup/run、取中位数 |
| 周数不够 | W11/W12 可压缩，文档和博客不阻断核心交付 |

---

## 验收口径 v2

### W7 成功标准
下面 4 条同时满足，才算"CPU 编译器闭环完成"：

1. `examples/mlp.tsy` 可编译
2. 生成产物可执行
3. 数值与 interpreter / PyTorch 对拍通过
4. 所有 parse/shape/pass/codegen 测试全绿

### W10 成功标准
下面 4 条同时满足，才算"GPU 路径成立"：

1. transformer block `.tsy` 可跑
2. CUDA 输出与 CPU 输出在容差内
3. 对拍 PyTorch 误差合格
4. benchmark 至少覆盖 5 种 shape

### W11 成功标准（2026-04-18 已达成）
下面 6 条同时满足，才算"自动化质量门槛建立"：

1. `.github/workflows/ci.yml` 绿 —— 每次 push/PR 触发 CPU-path ctest（~20/20）
2. 本地全 ctest 32/32 通过（含 CUDA）
3. `bash scripts/bench_local.sh` 0 FAIL（只 gate 1024³ matmul 三行）
4. `benchmarks/baseline/rtx3080_wsl.csv` 入库，`docs/benchmarks/baseline.md` 可独立复现
5. W10 三条 reviewer follow-up 全部落地（TSY_PYTHON_EXECUTABLE / CUDA sync / verifyUnary 注释）
6. 回归阈值政策与物理噪声对齐（spec 原 18 行 baseline 被证伪后收窄到 3 行 1024³，有据可查）

### W11 验收命令

```bash
cd /home/xs/tsy-wsl-export/tensor-sysy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure        # 32/32

# CI 等价（模拟 GHA runner —— 需要 .venv 先备份）
mv .venv .venv.bak
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j
(cd build-cpu && ctest --output-on-failure)        # ~20/20 CPU-path
mv .venv.bak .venv
rm -rf build-cpu

bash scripts/bench_local.sh                        # 0 FAIL
python3 benchmarks/run_shapes.py --check-scheduler # ≥1.20x tiled/naive
```

---

## 现在最建议的开工顺序

1. **先做 W0/W1**
   原因：`SourceLocation + CLI + parse 回归` 是后面所有事情的地基。
2. **然后直奔 W3**
   原因：shape/verifier 会倒逼 HIR 设计稳定。
3. **W6 提前思考**
   原因：adapter 决定 HIR/LIR 的调用约定，不能等到 W7 才发现不匹配。

---

## 远程仓库

- sysy-compiler: https://github.com/BetterThanAny/sysy-compiler.git
- mini-llm-engine: https://github.com/BetterThanAny/mini-llm-engine.git

---

## 一句话结论

`v2` 的核心不是"多做功能"，而是把原来隐含的两层假设显式化：

- `sysy-compiler` 只是前端基础，不自带诊断体系
- `mini-llm-engine` 只是可复用算子集合，不是现成通用后端

---

## 项目期间安装的工具（Claude 安装记录）

项目结束后用户自行决定是否删除。每条记录：工具名 · 安装方式 · 安装时间（UTC+8） · 安装原因 · 卸载命令。

| 工具 | 方式 | 时间 | 原因 | 卸载 |
|---|---|---|---|---|
| `cmake` 4.3.1 | `brew install cmake` | 2026-04-18 W0 | W0 验收命令 `cmake -S . -B build` 需要；`sysy-compiler` 原来用 Makefile，TensorSysY 切换到 CMake 是为了统一 flex/bison 生成 + 未来链接 `mini-llm-engine` | `brew uninstall cmake` |
| numpy 2.4.4 | `uv pip install --python .venv/bin/python numpy` | 2026-04-18 W10 | e2e transformer_block 的 numpy 参考实现（对拍 3 个 backend） | `rm -rf .venv` |
| pytest 9.0.3 | `uv pip install --python .venv/bin/python pytest` | 2026-04-18 W10 | e2e 测试驱动（ctest 通过 `.venv/bin/python -m pytest` 调用） | `rm -rf .venv` |

后续（W2+）如果还需要额外依赖（gtest、LLVM、CUDA toolkit 等），继续往这张表里追加，不要散落在其他地方。
