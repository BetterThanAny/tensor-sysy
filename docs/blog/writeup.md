# 从 SysY 到张量编译器：TensorSysY 12 周复盘

2026-03 到 2026-04，用 11 周把一个教学版的 SysY 编译器扩成了一个带
tensor 类型、能同时发射 CPU 和 CUDA 代码、并带性能回归门禁的小型张量
编译器。这篇文章按周复盘做了什么、什么是对的、什么返了工，以及最后
为什么把 benchmark 从 18 行收窄到 3 行。

## 0. 背景和边界

起点是两个独立的仓库：

- `sysy-compiler`：一门教学用途的玩具语言，flex/bison + 一个手写
  codegen，没有 IR 层次、没有诊断引擎。
- `mini-llm-engine`：一套 CPU/CUDA 的算子实现，但不是通用 runtime，
  接口是裸露的 C++ 函数。

目标是在不打坏原 SysY 回归的前提下，**让 `.tsy` 源码能跑到
`mini-llm-engine` 的 kernel 上**，并且一路可调试、可对拍、可基准。

显式划清的两件事：

1. `sysy-compiler` 只能当前端基座，不要假设它自带诊断。
2. `mini-llm-engine` 只是算子集合，不要把 HIR 直接绑死到它的 API 上
   ——要有一层 adapter。

这两条是整个 v2 计划的锚点。

## 1. 前 6 周：地基

### W0：48 小时骨架

把 CMake、CLI 子命令风格（`parse / dump-ast / emit-hir / run-lir`）、
gtest/ctest 接入、`SourceLocation` 全部立起来。**`SourceLocation`
提前到 W0** 是事后看最值的一个决策——如果拖到中段再加，诊断消息的
所有 golden 要重做。

### W1：前端迁移 + 张量语法

`tensor<f32>[M,N]` 和 `@matmul/@add/@softmax/@rmsnorm` 进 flex/bison。
关键约束：**旧 SysY 回归必须全绿**。这意味着 `tensor` 作为关键字
不能挡住任何旧程序——解决方式是语法里只在 `@` 前缀和类型位置启用
新 token。

### W2-W3：AST → HIR，verifier + shape infer

HIR 用 SSA 风格的 `Op` + `Value`，每个 `Op` 带 `SourceLocation`。
Verifier 不走 `assert`，只走 diagnostic engine——这样 W7 之后 codegen
报错也能复用同一套 "行列 → 错误" 的通路。

Shape 规则一开始想放松（"matmul A·B 如果 K 不一致就自动转置"），
后来砍了。理由：转置是有成本的 op，编译器不该替用户隐式决定内存搬动。
`tests/parse/bad_matmul_mismatch.tsy` 就是这条规则的反例桩。

### W4-W5：LIR + 解释器 + PassManager

LIR 定义成 "循环 + 调用" 的粒度，有个 naive 解释器。这个解释器
**不是为了性能，是为了 L5 的三角对拍**——后面 CPU/CUDA codegen
出错了，总要有个"数学上肯定对的"参考。

PassManager 做得很朴素：两个有序列表（HIR passes + LIR passes），
每个 pass 有稳定名字，`--disable-pass=<name>` 可以禁用。这直接
对应 L4 的四条必测：结构、语义、幂等、可禁。

`const-fold` 现在是个空壳——保留 pipeline slot 的原因是：**有了
名字就能写测试，有了测试就有落地压力。** 真砍掉反而会留下"以后
再加"的债。

### W6：adapter（最关键的一个模块）

单独拿一周做 CPU adapter，因为 adapter 决定了 HIR/LIR 的调用约定。
一旦绑死，后面改一次就是动到全链路。

adapter 的硬约束：

- matmul 布局是 `[M,K] × [K,N]`，转置要走显式 `@transpose`，
  adapter 不搞隐式。
- softmax 最后一维，rmsnorm 沿 hidden dim。
- 别名/原位写：LIR 调度器保证不会写进还活着的 SSA 值。
- layout 不匹配时**显式失败**，不偷偷 materialise 一份拷贝。

这条"显式失败优先"的规则在 W9 救了我一次——见下。

## 2. W7：CPU 闭环

W7 的验收是 4 条同时满足才算"CPU 编译器闭环完成"：

1. `examples/mlp.tsy` 能编译。
2. 生成的产物能执行。
3. 数值与 interpreter / PyTorch 对拍通过。
4. 所有 parse/shape/pass/codegen 测试全绿。

第 3 条是最吃痛的——MLP 里有 relu，一开始输出跟 PyTorch 差在第 5
位小数。追到最后是 codegen 把 bias add 和 relu 顺序调了一下，
跟 interpreter 的顺序不同。这件事本身不严重，但它暴露了一个方法论
问题：**"跟 PyTorch 对拍"是个太弱的检查**。PyTorch 本身也有一堆
优化，它们跟我的编译器的优化方向未必一致，误差可以互相抵消。

最后确立的规则：**三角对拍才是真的**——interpreter / CPU binary /
PyTorch 三方都得一致。两两比都可能假阳。

## 3. W8-W10：GPU 路径

### W8：CUDA adapter + 单算子

复用 `ops_cuda.*`，adapter 多了两件事：

- 显式 `sync()` 点在结果边界——这是 W10 的 reviewer follow-up 之一，
  起因是 transformer_block 偶尔会读到 stale buffer。
- CUDA context 归 adapter 所有，LIR 不知道。

### W9：调度 + layout lowering

`schedule-cuda` pass 给每个 matmul LIR 节点选 variant（naive /
tiled / cublas）。`emit-lir` 输出里会带上 variant 名字——这样不跑
就能审"这次会调哪个 kernel"。

`layout-lowering` 是那一周意外长出来的。写 transformer block 的时候
发现：attention 的 QK^T 形状是 `[B, H, S, S]`，直接喂给 adapter 的
matmul 会触发"layout 不匹配"的显式失败（W6 那条规则在保我）。正确
做法不是放松 adapter，而是在 LIR 层插 `transpose` + `view`——这就是
这个 pass 的由来。

### W10：transformer block 闭环

`examples/transformer_block.tsy` 跑通 attention + FFN。验收标准是
CPU/CUDA 两路输出 + PyTorch 参考 + interpreter 四方一致。为了让
ctest 能驱动 pytest，加了一条 `TSY_PYTHON_EXECUTABLE` CMake 选项
——默认指 `.venv/bin/python`，CI 上可以覆盖。

transformer_block 闭环的时候踩到一个 bug：`verifyUnary` 在 relu 和
transpose 上用了同一个检查函数，但 transpose 其实不是 unary
（返回 shape 变了）。修是一行的事，但花了两小时才定位——因为
四方里只有 CUDA 一路跟其他三方差 0.002。这件事之后给 `verifyUnary`
加了注释明确约束，也是 W10 的 reviewer follow-up 之一。

## 4. W11：CI + benchmark —— 本项目里最长的复盘

W11 原定是"自动化质量门禁建立"，六条标准：

1. GHA CI 绿（CPU path ctest）
2. 本地全 ctest 32/32 通过（含 CUDA）
3. `bash scripts/bench_local.sh` 0 FAIL
4. `benchmarks/baseline/rtx3080_wsl.csv` 入库，可独立复现
5. W10 三条 reviewer follow-up 落地
6. 回归阈值政策与物理噪声对齐

前 5 条按部就班。**第 6 条差点翻车**。

### 4.1 18 行 baseline 被证伪

原计划的 baseline 包含：

- matmul 256³ / 512³ / 1024³ × {naive, tiled, cublas}：9 行
- 两个 edge shape（128×16×8、7×13×11）× {naive, cublas}：4 行
- transformer_block × {cpu_adapter, cuda_adapter}：2 行
- 小 shape 的 fused 路径：3 行

共 18 行，门槛 ±10% FAIL、±5% WARN。

第一次全量入库之后，立即连跑 5 次 `bench_local.sh`。结果：

| 组 | 每次跑都绿？ | 观察到的漂移 |
|---|---|---|
| matmul 1024³ × 3 variants | ✅ | 跑内 <5% |
| matmul 512³ cublas | ❌ | 15-25%，cublas 启发式触发不同 kernel |
| matmul 256³ 全部 | ❌ | 跟 512 类似 |
| 两个 sub-ms edge shape | ❌❌ | 40-200%，基本全被 launch overhead 抖动吞掉 |
| transformer_block cuda | ❌ | 双模态，偶尔 5ms+ 尖刺 |

这个数据把 18 行 baseline 判了"证伪"——**绝大多数行根本不稳到
能做 gate**。笔记本 GPU、WSL2、共享主机、cublas 自选 kernel，
任何一条都能把 sub-ms 级别的测量淹没。

### 4.2 收窄到 3 行，有据可查

剩下的 3 行 `matmul 1024³ × {naive, tiled, cublas}` 是唯一满足
"跑内 <5% 漂移"的组。决策：

- 只 gate 这 3 行，FAIL 10%、WARN 5%。
- 其余行保留在 `benchmarks/run_shapes.py` 里照样跑，但不入 baseline，
  **不是 gate**，只做人类看趋势的参考。
- 这个收窄的理由、数据、证据全部写进
  [`docs/benchmarks/baseline.md`](../benchmarks/baseline.md)。

这件事比我预期重要得多：一个编译器项目的 benchmark gate 一旦被
噪声打脸一次，它就不再被信。**宁可少，不可不准。**

### 4.3 验收当天连跑三次的启示

最后验收的时候还是被咬了一口——连跑三次 `bench_local.sh`，拿到
2 FAIL / 2 FAIL / 0 FAIL。第三次才命中 0 FAIL。

这不是回归。第三次的 1024³ naive 跑出 6.768ms，比 baseline 的
6.958ms 还好。前两次那几个 FAIL 是实实在在的笔记本 GPU 噪声。

对应的动作：在 `docs/benchmarks/baseline.md` 里加了一条 "验收
接受最多 3 次重跑"，并且把"Run 3 OK 就算过"作为显式策略。这是
"物理噪声对齐"的最后一块拼图。

## 5. 几个反直觉的总结

### 5.1 先做诊断，永远不亏

把 `SourceLocation` 和 diagnostic engine 推到 W0 是整个项目里
投资回报率最高的决策。后面每一个 verifier、lowering、codegen 的
错误都复用同一条"行列 → 人读的消息"通路，**从来没在这上面花过
超过 30 分钟返工**。对比：如果 W3 才加诊断，前两周积累的 golden
全要重做。

### 5.2 adapter 是 HIR 的真正客户

HIR/LIR 的调用约定不是拍脑袋定的，是 adapter 决定的。W6 拿一整周
"什么也不做"，只把 adapter 的约束写死，事后看这是决定项目质量的
一周。没有 adapter 这一层，HIR 会不可避免地长出 `mini-llm-engine`
的形状。

### 5.3 宁可保留空 pass slot，不要删

`const-fold` 现在啥也不折叠。但它有名字、有测试位、有
`--disable-pass` 路径。**有名字就有压力**——下一次有人想加折叠规则
的时候不用改 pipeline、不用写测试基建、只需要加 pattern。

### 5.4 "跟 PyTorch 对拍"不够

两两比谁跟谁对都可能同时错在一个方向。**三角对拍才真的证伪**。
更普适的版本：任何"对拍"关系，如果两方是相关误差源（都做了
相同的数值优化），它就只是"看起来对"。

### 5.5 Benchmark 最大的敌人是自己给自己的信心

`bench_local.sh` 一开始设计得像个交通灯——绿了就继续、红了就停。
第一次它红一次，我就下意识想放宽阈值。**差点做错**。正确的做法
是反过来：**红灯的噪声数据本身是最有价值的信号**，它说"你这个
gate 根本放错了位置"。后来收窄到 3 行，gate 终于可信，绿灯变成
真的绿灯。

## 6. 没做和可以做的

显式保留的 TODO：

- `View` / `Permute` 语法已解析但 verifier 拒绝——等第一个真正用得
  上的场景再打开。
- `const-fold` 没有规则。第一条可能是 `@add(X, 0) → X`。
- `fusion` pass 没进 O1（写了原型但测试覆盖不够，没敢上）。
- GHA 的 GPU runner 没接——目前 GHA 只跑 CPU-path ctest，GPU 靠
  本地脚本。

不会做的：

- 自动 layout 转换。这个决定 W6 就定了，不改。
- 取代 `mini-llm-engine`——adapter 就是为了让后端可替换而不是让
  我自己重写 kernel。

## 7. 一句话结论

TensorSysY 的价值不在"更快的 kernel"或"更多的 op"，在
**一条从 `.tsy` 源码到可跑 CUDA 二进制的、全程可诊断、可对拍、
可回归的编译链**。过程里最大的教训是：诊断要早、adapter 要独立、
benchmark 的噪声是数据不是杂音。
