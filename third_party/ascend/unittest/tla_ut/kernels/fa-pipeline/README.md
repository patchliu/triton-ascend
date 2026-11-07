# FA8：可读 task/stage 流水

完整外部规格：B=4、H=8、S=1024、D=64，dense BF16 Q/K/V，返回 BF16 O 及
FP32 max/sum `[4,8,1024,8]`。与 `../fa` 中 优化 FA 的 FA8 工作量一致。
**本轮未编译、未执行精度/性能测试。它是可读的流水用例，不宣称性能超过 优化 FA 版。**

## 选择独立 query tile

一次 `tla.pipeline(num_query_tiles)` 迭代处理 16 行 Q，覆盖全部 1024 个 K/V。
28 个 program 按 `core + query_tile * 28` 分配总共 2048 个 query tile，完整覆盖 B/H/S。
内部 BM16、KV1024 与性能参考的 BM128/BN256 不同；没有缩小外部输入或忽略有效 key。

全 KV softmax 消除了跨 KV 块的 `m/l/acc` 递推。每个 scratch 在同一逻辑迭代内先写后读，
不同逻辑迭代没有数学依赖，因而可以安全地表达 stage；不依赖“单槽等于持久状态”的假设。
BF16 P 的舍入发生在全行 max 下，与 online-softmax 的逐块舍入不保证逐位相同，按同一 FP32 参考验收。

| Task | Core | Stage | 内容 |
|---|---|---:|---|
| load_qk | Cube | 0 | Q/K GM→L1 |
| qk | Cube | 0 | Q @ K.T，Fixpipe 按 query 行拆给两个 AIV |
| load_v | Cube | 1 | V GM→独立 L1 存储 |
| softmax | Vector | 1 | 全行最大值、exp、FP32 sum、BF16 P |
| publish_probability | Vector | 1 | 两个 AIV 分别将自己的 P 行写入 L1 |
| pv | Cube | 1 | P @ V，Fixpipe 输出分子 |
| output | Vector | 1 | 除分母、BF16 输出、八份 max/sum |

stage 0 在 tick n 处理 query n；stage 1 处理 query n−1。
同 stage 保持依赖顺序；stage 不意味着同一引擎内的指令能够并行。
没有手写同步、物理槽号、软件 fill/drain，也没有源码宏或六相位展开。

## 存储与同步要求

K/V 使用独立 L1 allocation，QK/PV 使用独立且按结果大小分配的 L0C allocation。
每个逻辑迭代的 scratch 先写后读；全 KV 计算没有 online-softmax 跨迭代递推。
调用者保证 GM 迭代独立：host 分配独立输出，两个 AIV 写不同 query 行。
编译器负责异步 ready/free 和存储复用，不验证任意 GM 参数别名下的迭代独立性。

两个 AIV 各写 P 的一半，PV 必须等待两方；单槽也必须在最后消费者完成后才能复用。
当前 N=last_stage-first_stage+1，所以只有 score 使用两份，其余 root 各一份。
显式数组预算：

| 存储 | 版本化后大小 |
|---|---:|
| L1：Q + K + V + P | 290 KiB |
| L0C：QK + PV | 68 KiB |
| UB：每 AIV 的 score/P/state/output | 85.5625 KiB |

这不包含后端临时量、额外对齐或 VF 栈。拆分 QK/PV 使原来共享 root 的
128 KiB L0C 预算降为 68 KiB；这是静态预算，不是已测性能收益。

需要配套固定 stage 的 NPU-IR：LowerTLAStages→CCGSS→CreatePreload→GSS/多缓冲。
所有 task 显式给 stage，暂不支持 order。11 月 6 日日志是修改前的旧产物，
本次源码改动尚未构建、执行精度或性能验证。

## 运行与对比

```bash
TLA_TEST_STAGE=run pytest -q -s third_party/ascend/unittest/tla_ut/kernels/fa-pipeline -k test_run
# 将 test_fa_pipeline.py 的 RUN_PERF 改为 True 后：
TLA_TEST_STAGE=run pytest -q -s third_party/ascend/unittest/tla_ut/kernels/fa-pipeline -k test_perf
```

每个入口都对比 candidate、优化 FA8、AscendC，同样检查全部输出、统计复制、normal/uniform/peaked
输入和重复调用。性能独立做精度预检查，再轮换采样，保存 `fa-pipeline-results` 下的 JSON 与 trace。
候选/参考/原生共用 `../fa/testing.py` 和独立 FP32 参考；详细精度门限和计时口径见 `../fa/README.md`。

本阶段只支持 FA8。开放 causal、更多 head dimension 或 KV streaming 前，需要重新设计 mask、
片上容量及递推生命周期，不能仅修改上方常量。

## v2：BM128 / BN256 / DN softmax

已有设备反馈显示 v2 存在严重数值错误，其性能数字不能作为有效基线。
`maximum[0,0]` 的 NORM_B32 写入覆盖 64 个 FP32 lane，并非只写第零行；
已发现的同步异常也不能仅凭源码归因于 static_range 本身。v2 保留用于对照。

`fa_pipeline_kernel_v2.py` 保持完整 dense FA8 输入输出，内部改为 BM128、BN256。
每个 query tile 遍历四块 K/V；同一份 K/V 服务 128 行 Q，源码 K/V 逻辑请求量
从 v1 的 512 MiB 降到 64 MiB，不代表 HBM 实际流量必定按此比例下降。

QK 使用 K @ Q.T，COLUMN_SPLIT 向两个 AIV 各分配 64 个 query 列；softmax 每个
FP32 lane 对应一行 query，沿 key 方向使用四条独立 max/sum 链。PV 使用 P.T @ V。
输出统计量用向量 interleave 展开为八份，避免逐元素复制。

| Stage | Task | 存储与依赖 |
|---:|---|---|
| 0 | load_query | 预取 Q，供当前逻辑 query 的 stage 1 读取 |
| 1 | qk_b → load_value_b → softmax_b → publish_probability_b → pv_b → accumulate_b | b=0..3；先初始化，再按块递推 m/l/acc |
| 2 | output | 读取同一 query 最终 acc、m、l，归一化并存储 |

pipeline 的迭代单位始终是 query tile。四个 KV 块在同一 stage、同一逻辑迭代内顺序执行，
因此没有跨 pipeline 迭代的数学递推。首块直接初始化 m/l/acc，后续块更新：
`m'=max(m,max(score))`，`alpha=exp(m-m')`，`l'=alpha*l+sum(exp(score-m'))`，
`acc'=alpha*acc+bf16(exp(score-m')) @ V`。

源码用 `tl.static_range(4)` 复用一份计算链；TLA 前端识别其编译期展开，生成 24 个
stage-1 直接子任务，名称包含块号。普通设备循环内嵌 task 仍然拒绝。无需改动 C++ builder，
也没有另外的源码生成器。静态展开后总计 26 个 task，stage 顺序为 0、24 个 1、2。

按 stage 跨度，Q、acc、m/l 各需两份，其余 scratch 各一份。显式数组的静态预算为
L1 160 KiB、L0C 160 KiB、每 AIV 的 UB 157.5 KiB；不包含后端临时量、额外对齐或 VF 栈。
QK/PV、K/V、score/P/partial 的复用必须由自动同步保护；PV 必须等待两个 AIV 发布完整 P。

v2 优先改善数据复用、MMAD 形状与 SIMD 布局。当前跨 query 的流水只重叠 Q 预取、
计算链和最终输出；没有将 KV 的 QK/softmax/PV 强行分到不同 stage。进一步重叠 KV 块
需要明确支持递推与版本生命周期，不能只调整 stage 数字。

测试入口参数化 v1/v2，两者都与优化 FA、AscendC 及 FP32 reference 比较：

```bash
TLA_TEST_STAGE=run pytest -q -s third_party/ascend/unittest/tla_ut/kernels/fa-pipeline -k 'test_run and v2'
# 启用 RUN_PERF 后：
TLA_TEST_STAGE=run pytest -q -s third_party/ascend/unittest/tla_ut/kernels/fa-pipeline -k 'test_perf and v2'
```

本次只做源码、静态展开结构和 Python 语法检查，未进行编译、设备精度或性能验证。
不能从逻辑读取量下降直接推断实际加速比。

## v3：外层 query 循环，内层 KV pipeline

`fa_pipeline_kernel_v3.py` 延续 v2 的 BM128/BN256、DN softmax、BF16 P 和完整 FA8
输入输出。外层普通 `tl.range` 遍历 query tile，内层 `tla.pipeline(4)` 遍历 KV 块。
不展开四份 task，也不使用嵌套 pipeline。每个 query 只加载一次 Q。

```python
for query in tl.range(num_query_tiles):
    # 外部分配 Q、m/l/acc；普通 core scope 加载 Q、初始化状态。
    for block in tla.pipeline(S // BN):
        # 在这里分配该 KV 块的 scratch。
        with tla.scope.cube().task("load_kv", stage=0): ...
        with tla.scope.cube().task("qk", stage=1): ...
        with tla.scope.vector().task("softmax", stage=1): ...
        with tla.scope.vector().task("publish_probability", stage=1): ...
        with tla.scope.cube().task("pv", stage=1): ...
        with tla.scope.vector().task("accumulate", stage=1): ...
    # 排空后在普通 vector scope 内归一化、写 O 和 max/sum。
```

Stage 0 提前加载下一块 K/V；stage 1 顺序执行当前块的 QK、softmax、P 发布、PV 和
累积。`m/l/acc` 的所有域内访问都在 stage 1，因此 `softmax(k) → accumulate(k)
→ softmax(k+1)` 的递推保持顺序。stage 是时间安排，task 本身不是全引擎 barrier；
RAW/WAR/WAW、两个 AIV 的 P 发布完成和单槽复用仍由编译器的自动同步负责。

初始化 `m=-inf, l=0, acc=0`，随后所有 KV 块使用同一公式，不需要首块分支。
对于有限的 dense 输入，首块 `alpha=exp(-inf-m')=0`，不会读取未初始化状态。
packed P 保留 v2 的 pitch 和两个 AIV 各自的 query 列映射；统计量仍输出八份复制。
出口的小型 `static_range(2)` 只展开 max/sum 的 SIMD 复制，不生成 task 或 KV 流水。

| 对象 | 生命周期 | 物理份数 |
|---|---|---:|
| Q | query 内只读，KV pipeline 外分配 | 1 |
| maximum / denominator / accumulator | query 内持久 UB，全部域内访问在 stage 1 | 各 1 |
| K / V | KV scratch，stage 0 写、stage 1 读 | 各 2 |
| P / score / alpha / packed P / partial / QK / PV | KV scratch，stage 1 内使用 | 各 1 |
| output / max_copies / sum_copies | query 出口临时存储 | 各 1 |

显式数组预算：L1 **208 KiB**、L0C **160 KiB**、UB **141 KiB / AIV**，不包含
编译器临时量、额外对齐和 VF 栈。K/V 逻辑请求量仍为 64 MiB，预取增加的是 L1 版本，
没有增加算法读取量。这个版本优先提供简单的 KV 预取流水；跨 query 重叠、QK 与下一块
softmax/PV 的更深交叠不在本版范围内，不能据此声称已经达到优化 FA 的速度。

### 使用约束

1. `tla.pipeline(num_iters)` 使用非负标量次数、从 0 开始、步长 1；不支持 SSA
   loop carry、break/continue/return。外层可以是携带 SSA 值的普通 for，不能是
   运行时 if、while、core scope 或另一个 pipeline。
2. task 必须直接位于 pipeline 循环体中，使用唯一名称、显式整数 stage（0..63），
   stage 按源码非递减。禁止将 task 包在运行时循环/分支里。SIMD 扫描等计算循环
   可以放在 task 内，不把每轮计算都变成一个 task。
3. pipeline 内分配的静态 UB/L1/L0C 是迭代 scratch：每轮先写后读，别名 view 共享
   root 和版本选择，份数由 stage 跨度决定。task 内不分配 scratch。
4. pipeline 外分配的静态 UB/L1/L0C 对象保持单份存储，可以跨 stage、跨核读写；
   不再用 FA v3 的单 stage UB 写法限制所有 kernel。调用者负责初始化和递推的时序正确性。
   例如 stage 0 读取上轮状态、stage 1 才更新该状态，仍可能读早；自动同步不会替你修正 stage。
5. task 的内存效果必须可追踪，GM view 源于函数参数；GM 迭代独立性及别名安全由调用者
   保证。pipeline 外的普通代码不再经过 TLA 专用整函数结构检查，服从通用编译链规则。
6. task 内不混用手写同步、手工多缓冲/preload 标记或不透明调用；自动同步必须开启。
   不再因为函数含 pipeline 就禁止外围手写同步，但这不保证任意组合已获验证。

第 1、2 项中的放置位置、stage 非递减和 0..63 上限属于 Python 前端策略。
NPU-IR 保留域/循环/task 的必要 IR 结构、可识别内存效果、存储来源及自动同步前提；
后端不再限制普通外层循环携带值、外部状态的核/阶段/UB 类型，也不再硬编码 64 阶段上限。
跨 stage 递推的准入不等于正确性证明。v3 的六个 task 和状态安排没有改变。

### 配套修改与运行

本版同时修改 TLA 前端和 NPU-IR 的 `LowerTLAStages`：允许普通外层 for，
支持外部持久状态；复用目标分支已有的统一 affinity VF outlining，
在 CCGSS 前处理入口和出口的 SIMD VF。
**包含 `cf35f1e1d` 的已有包可继续用于 v3，不需要为本轮放宽重新出包。**
`49466d3ef` 或仅包含 `89e13a0ff` 的包缺少外部持久状态支持，不能直接运行当前 v3。
使用新放宽的用法需要更新后端；Python 不能绕过旧二进制中的限制。
没有修改 `tla.builder.cc` 的 `create_tla_pipeline`，也没有修改 CCGSS 求解算法。

```bash
TLA_TEST_STAGE=run pytest -q -s third_party/ascend/unittest/tla_ut/kernels/fa-pipeline -k 'test_run and v3'
# 精度通过并启用 RUN_PERF 后：
TLA_TEST_STAGE=run pytest -q -s third_party/ascend/unittest/tla_ut/kernels/fa-pipeline -k 'test_perf and v3'
```

测试入口已加入 v3，沿用全部输出及 max/sum、normal/uniform/peaked、重复调用、优化 FA
和 AscendC 的对比。补充了前端和 NPU-IR 合同回归源码。按本轮要求未执行这些测试，
未构建编译器、未上卡，也未测性能；同步正确性仍须通过生成 IR 与设备结果确认。
