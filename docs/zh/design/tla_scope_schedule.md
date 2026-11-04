# TLA scope 与 NPU-IR Solver 调度设计

状态：接口与 IR 设计提案，尚未实现。示例中的链式 `.task()`、`scope.schedule()` 和 `ScheduleHints` 均为新增接口。
源码依据：Triton-Ascend `1987910ec`、AscendNPU-IR `4eb12142e`、TileLang `fe6d319c`。

## 1. 目标与结论

默认由用户描述计算、数据布局和核分工，由 NPU-IR 联合选择流水阶段、任务顺序和缓冲版本，并生成同步。
用户通过局部约束介入，也可以保留完整手写流水。高层算子和低层指令共享这条调度链。

**复用 `scope.scope` 作为任务容器，复用 preload scope 作为已求解流水的物化格式；不要直接把 preload 编号作为用户接口。**
前端不运行 Solver、不分配 flag、不展开 prologue/epilogue。Solver 与合法性检查放在 NPU-IR，避免绑定到 TLA Python。

首期范围：一个 AIC 与配对的两个 AIV、静态 tile、正步长规则循环、可分析的片上 buffer/view。
任意 block 间共享 GM workspace、数据相关可选任务和复杂动态循环后续扩展；不支持的模式应明确拒绝或退回保序方案。

## 2. 前端接口

```python
with tla.scope.schedule(mode="auto", group=(1, 2), hints=hints):
    for t in tl.range(begin, end, step):
        with tla.scope.cube().task("gemm"):
            ...
        with tla.scope.simd().task("activation"):
            ...
        with tla.scope.vector().task("store"):
            ...
```

- `cube/vector/simd` 保留核类型与 SIMD/VF 语义。`.task(name)` 只增加任务边界与稳定标识，不隐含 stage、barrier 或 buffer 数量。
- `.task()` 可省略，由编译器按操作与 VF 边界推导任务；名称在当前调度域内唯一。
- 显式普通 task 为单核亲和、单 pipe 的调度单元，内部保序。跨 pipe 的 load→pack→copy 应拆开，不能把整个 Vector scope 一律原子化。
- SIMD task 保持原 VF 内部指令顺序，不允许自动拆分寄存器计算。初期任务内部不插入跨 pipe 同步。
- 调度域中的源码顺序定义正确结果；编译器依据 SSA、内存依赖和跨迭代递推决定合法重排，而非把所有语句变成独立节点。

可选约束与计算代码分离：

```python
hints = tla.ScheduleHints()
hints.prefer_versions("a", 2)       # 软偏好，可不满足并说明原因
hints.limit_versions("acc", 1, 2)   # 硬约束：1 至 2 个物理槽
hints.require_stage("store", 1)     # 硬约束，其他 task 仍自动求解
```

硬约束无解时返回冲突原因；不能静默忽略。地址不重叠等正确性断言单独建模，不能混入性能偏好。
未标 stage 的任务保持自由，不采用“只要一处有 Stage，其他兄弟任务默认 stage 0”的隐式切换。

缓冲版本必须区分三个状态：未指定/自动、明确版本数、用户已手动构造 ring。
现有 `alloc(..., versions=1)` 的默认值与显式 `1` 无法区分，需改为内部 sentinel：调度域外保持原行为，域内未指定表示自动；显式 `1` 固定单槽。
已有 `(3 * D, ...)` 加 `% 3` 的 ring 属于 manual storage，不能再次自动扩容或重写索引。
`name="a"` 是提案中的稳定存储标识；所有 subview/reshape 共享根存储的版本计划。

### 手动与自动的责任边界

| 模式 | 顺序和缓冲 | 同步 |
| --- | --- | --- |
| `auto` | Solver 选择，接受部分硬约束 | 编译器管理整个调度域 |
| `manual` | 保留内部编排与 ring | 保留内部同步，验证接口边界 |

固定 stage 仍属于 `auto`，不要求用户手写 flag。
首期禁止 auto 域直接混入原始 set/wait；已有手写流水应整体放进 manual 域。
嵌套 manual 域作为复合任务接入：输入在进入前可用，对外读写在退出前完成；内部 flag 必须预留且不能参与域外握手。
异步退出需要后续 completion-token 接口，不能通过隐藏 flag 表达。

自动同步覆盖数据就绪和最后使用后的存储释放，包括两个 AIV 的参与者集合。
仅预留手写 flag 编号不能证明协议兼容；手工 UnitFlag 同样需要明确所有权。

## 3. Cube–Vector 示例

计算一批固定形状的 `C[t] = relu(A[t] @ B[t])`，每组为 `32×64 @ 64×128`，输出 FP32。
每个 AIC 配两个 AIV，Fixpipe 按行分发；省略 host 入口。下列代码是接口草案，不可直接运行。

```python
@tla.jit
def batched_gemm_relu(A, B, C, BATCH: tl.constexpr, CORES: tl.constexpr):
    with tla.scope.schedule(mode="auto", group=(1, 2)):
        for t in tl.range(tl.program_id(0), BATCH, CORES):
            # 单个逻辑迭代的 buffer；物理槽复用由 NPU-IR 决定。
            a = tla.alloc((4, 2, 16, 16), tl.bfloat16, space="l1", name="a")
            b = tla.alloc((8, 4, 16, 16), tl.bfloat16, space="l1", name="b")
            acc = tla.alloc((8, 2, 16, 16), tl.float32, space="l0c", name="acc")
            out = tla.alloc((16, 128), tl.float32, space="ub", name="out")

            with tla.scope.cube().task("load_a"):
                tla.load_gm_to_l1(tla.view(A + t * 32 * 64, (32, 64)), a)
            with tla.scope.cube().task("load_b"):
                tla.load_gm_to_l1(tla.view(B + t * 64 * 128, (64, 128)), b)
            with tla.scope.cube().task("gemm"):
                tla.cube.mmad(a, b, acc, clear_accum=True)
            with tla.scope.cube().task("fixpipe"):
                tla.copy_l0c_to_ub(
                    acc, out,
                    tla.FixpipeL0C2UBParams(
                        dma_mode=tla.FixpipeDMAMode.NZ2ND,
                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                        sub_block_idx=None,
                    ),
                )
            with tla.scope.simd().task("relu"):
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                for row in tl.range(16):
                    for col in tl.static_range(2):
                        x = tla.simd.vlds(out[row, col * 64])
                        tla.simd.vsts(out[row, col * 64], tla.simd.vmax(x, zero))
            with tla.scope.vector().task("store"):
                sub = tla.sub_block_id()
                tla.store_ub_to_gm(
                    out,
                    tla.view(C + t * 32 * 128 + sub * 16 * 128, (16, 128)),
                )
```

依赖链为 `load_a/load_b → gemm → fixpipe → relu → store`。
候选方案可让 load 处理迭代 n、Cube 处理 n−1、Vector 处理 n−2；是否值得如此划分由代价与容量约束决定。
`out` 是两个 AIV 各自的 UB 存储实例，不能当作全核共享的一份 buffer；ROW_SPLIT 必须进入参与者与读写区域模型。
用户无需写 `% versions`、ready/free flag 或首尾排空代码。

## 4. NPU-IR 表达：先逻辑任务，再已求解计划

优先扩展现有 `scope.scope`，不先增加另一套 task op。新增属性放在 HIVM 命名空间，并实现 verifier。
以下是结构示意，省略类型和操作参数，**不是可直接交给解析器的 MLIR 测试**。

### 4.1 前端输出

```mlir
scf.for %t = %begin to %end step %step {
  %a = memref.alloc() ...
  annotation.mark %a {hivm.schedule_buffer = {id = "a", versions = "auto"}} ...

  scope.scope : () -> () {
    hivm.hir.nd2nz ...
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>,
     hivm.schedule_task = {id = "load_a", atomic = true}}

  // load_b、gemm、fixpipe、relu、store 同理。
} {hivm.schedule_domain = {id = "cv", mode = "auto", state = "logical",
                          aic = 1 : i32, aiv = 2 : i32}}
```

不在前端写入 `hivm.preload_num`，因为此时尚无调度结果。
域 scope 的约束在规范化时落到其拥有的循环；首期一个域只拥有一个流水循环，嵌套域单独处理。
TLA 原有 SIMD `outline/no_inline/vector_mode` 标记继续保留；这些标记不等于任务调度状态。

`scope.scope` 已有 `RecursiveMemoryEffects`，可作为分析入口，但还需要存储根、访问区域、参与核和递推距离。
VF 被 outline 后，任务标识应转移到调用点，并保留可追溯的读写摘要；不能因函数边界丢失依赖。
可分析的高层函数先展开再形成任务；不透明外部调用没有可信副作用摘要时禁止跨越重排。

### 4.2 Solver 输出合同

计划至少包含以下信息，持久化为 NPU-IR 属性；求解过程中的图不必全部变成独立 op。

| 对象 | 计划字段 |
| --- | --- |
| 域 | `state=planned`、目标架构、II、计划版本/指纹 |
| 任务 | 稳定 ID、stage、每 pipe 发射顺序、模型起始时间、资源/核归属 |
| 存储根 | 物理版本数、索引方式、生命周期、对齐后的容量 |
| 依赖边 | 生产/消费任务、访问区域、迭代距离、RAW/WAR/WAW、参与者 |
| 同步 | 逻辑 ready/free 通道及 owner；物理 flag 在后续分配 |

模型起始时间用于选序，不等于硬件按周期定时执行。实际程序由各 pipe 顺序及同步约束推进。
同一个 stage 可以包含多个任务，stage 数也不等于任何一个 buffer 的版本数。

## 5. preload scope 能复用什么

当前源码的真实路径包含：

```text
CVPipelining → 部分 preload 重排/多缓冲标记
→ 跨核同步 STEP1 → SplitMixKernel
→ 对齐/额外缓冲/PlanMemory → 跨核同步 STEP2
→ LowerToLoops → CreatePreload → 核内同步
→ PeelPreload → EnableMultiBuffer / LowerMultiBufferCounter
```

此图省略其他 pass，仅说明关键相对顺序。不能直接再塞一个独立 Solver，让两条路径各自生成流水。

| 现有机制 | 复用方式与限制 |
| --- | --- |
| `scope.scope` + `hivm.preload_num/max_preload_num` | 用于表达已选择的阶段偏移；不是完整任务图或约束模型 |
| `CreatePreload` | 复用偏移迭代、边界 guard、view 重建和物理地址轮换；必须校验输入合同 |
| `PeelPreload` | 复用首尾展开与 guard 简化，覆盖短循环、空循环和递推值 |
| `MarkMultiBuffer` / `PlanMemory` | 接收 Solver 版本数，不能再启发式改写；按实际 padding、scratch、地址空间检查容量 |
| 跨核/核内同步与 UnitFlag | 作为计划的实现后端；共享依赖/版本信息，不能各自创建冲突的协议 |
| `CVPipelining`、Reposition/Merge/Sink preload | 对 Solver-owned 域跳过选序；若继续使用，必须证明保持计划并更新任务映射 |

### stage 到 preload_num 的转换

建议对用户定义：stage s 的任务在流水 tick n 处理逻辑迭代 `n−s`。
当前 `CreatePreload` 在 P=`max_preload_num` 时，对编号 p 的 scope 使用：

```text
logical_iv = tick_iv − (P − 1 − p) × step
因此：preload_num = P − 1 − stage，P = max(stage) + 1
```

例如 load/gemm/vector 分别处于 stage 0/1/2：

```mlir
scf.for %t = %begin to %end step %step {
  scope.scope { /* load_a / load_b，保留各自任务边界 */ }
    {hivm.preload_num = 2 : i32, hivm.max_preload_num = 3 : i32, ...}
  scope.scope { /* gemm / fixpipe */ }
    {hivm.preload_num = 1 : i32, hivm.max_preload_num = 3 : i32, ...}
  scope.scope { /* relu / store */ }
    {hivm.preload_num = 0 : i32, hivm.max_preload_num = 3 : i32, ...}
}
```

版本槽以逻辑迭代序号 `epoch = (logical_iv − begin) / step` 选择，而不是直接使用 `iv % versions`。
消费者必须读取对应生产者 epoch 的槽，不能统一按当前物理 tick 取模；示例中的 `step=CORES` 尤其需要覆盖。

上述表示分组概念，不能为凑一个 stage scope 而合并不同核的 task。
该示例中 Fixpipe 也可独立选 stage；只有合法性和性能求解结果才能决定最终划分。

需要补齐以下兼容条件：

1. `CreatePreload` 当前只收集直接位于 `scf.for` 内的 preload scope；先规范化域/任务嵌套。
2. 同编号多 scope 在当前实现中受 shape-registry gate 控制，legacy 路径仅保留每编号一个槽位。
   对验证通过的 Solver 计划应提供显式支持，不能依赖函数名注册或让部分任务被覆盖。
3. `max_preload_num` 是该路径中的阶段槽数，不是最高编号，也不是统一 buffer 数量。
   本地 buffer 地址旋转已读取 PlanMemory 产生的实际地址数量；GM workspace 子视图仍有按 P 处理的路径，不能直接泛化。
4. 当前普通路径会扩展上界 `P × step`，再由 guard/peel 处理；不要擅自换成理论上的 `(P−1) × step`。
   空循环、末尾无效 tick 与同步 token 平衡必须一起测试。
5. scope 的物理排列顺序必须服从每 pipe 的求解顺序；preload 编号只负责迭代偏移，不表达全部顺序。

结论：首期复用现有格式与物化代码，但为 Solver 增加显式 verified-plan 入口及专用验证器。
一般数据流、动态 epoch 或当前 preload 无法承载的递推，后续使用独立 materializer；不把它们硬塞进现有编号。

## 6. Solver 与 pass 接入方案

建议新增一条按域选择的 NPU-IR 分支，旧 manual/legacy 路径保持原行为：

```text
NormalizeScheduleScopes
→ InferTaskEffects（alias、区域、核分工、跨迭代依赖）
→ NormalizeResources（布局、尺寸、对齐、已知 scratch）
→ SolveSchedule（顺序、stage、版本数、容量/通道预算）
→ VerifySchedulePlan
→ LowerScheduleToPreload + 固定版本标记
→ 现有 SplitMix / PlanMemory / 两阶段跨核同步 / preload / 核内同步
→ VerifyScheduledPipeline
```

这是目标顺序，不是对当前 pipeline 的描述。首期需要把相关资源分析前移或提取为可共享分析，
使 Solver 在 SplitMix 前仍能看到完整 Cube–Vector 图。不能等拆核后再分别求解互相依赖的任务。
晚期 PlanMemory 若因额外 scratch、对齐或碎片导致不可行，回到保存的逻辑 IR 降低版本数重求解；
不能只改物理槽数而保留旧同步计划。

Solver 核心约束：

- 依赖：任务 u(i) 的结果被 v(i+d) 使用时，`S_v − S_u + d×II ≥ latency_u`。
- 资源：同 pipe 的 issue interval、占用能力、共享带宽与任务串行约束；latency 不等于 issue interval。
- 存储：同一物理槽再次写入之前，所有消费者必须完成；覆盖 RAW、WAR、WAW 与跨迭代距离。
- 容量：分开计算 L1、L0C、每个 AIV 的 UB，包含 padding、对齐、scratch 和版本数。
- 同步：满足参与者与控制流 guard，预估通道数量；最终 flag 分配失败需反馈重求解。
- 正确性：真实 loop-carried 状态保留递推边；版本化只能消除存储假依赖，不能复制出独立的 softmax 状态。

目标以稳定吞吐 II 为主，兼顾短循环总延迟和内存占用；超时保留已验证可行解，无解则保序 fallback 或报告硬约束冲突。
代价模型只用于排序候选，不能用于替代数据依赖同步。可导出少量方案上卡验证，再固化计划重放。

下游 pass 通过 `owner/state/fingerprint` 识别已求解区域。改变任务、buffer 生命周期或参与者的变换必须更新并验证计划，
否则使计划失效并回到求解边界；仅保持语义而改变时延的变换也应使性能模型失效。

## 7. TLA 前端需要的改动

当前 scope handler 读取类对象并调用类方法 `mlir_attrs`，不能直接处理 `vector().task(...)`。
建议引入不可变、仅编译期存在的 ScopeSpec：`vector()` 创建 spec，`.task()` 返回附带任务属性的新 spec。
`handle_scope_with` 解析完整 context expression，再产生一个携带核属性和 task 属性的 `scope.scope`。

必须保留无 `.task()` 的旧写法；不执行 Python 运行时 context manager，不把链式调用 lower 成普通 device 函数。
`scope.schedule` 单独描述调度域，不继承 Vector 核属性。前端只验证静态参数；资源、依赖和 schedule 的合法性由 NPU-IR 验证。

## 8. 实施与验收

| 阶段 | 交付与验收 |
| --- | --- |
| 1：接口与保序调度 | 链式 scope、任务推导、读写摘要；示例计算正确，明确拒绝 auto 域原始 flag |
| 2：静态 Solver + preload 适配 | GEMM–ReLU、多版本容量约束、同 stage 多任务；空/短循环和任意正步长正确 |
| 3：DSA | 保留数值递推，支持分组 gather 与不同 buffer 版本数；比较手写 opt2 的精度、内存和真实耗时 |
| 4：可组合 manual 区域 | 外部效果/完成合同、私有 flag 预留、跨边界同步；随后再扩展动态 epoch 和跨 block 通信 |

必测：subview/reshape 别名、覆盖写与部分写、两个 AIV 的参与者、读取旧值、最后消费者、flag 耗尽、
stage 约束冲突、内存不足、VF outlining 后的副作用、版本数与 stage 数不等、非零下界、非单位步长、零/少量迭代。

DSA 中应允许 Solver 搜索 latent 三槽/RoPE 两槽及滚动 gather，但不硬编码它们为“最优”。
若 gather 被封成不可拆的大任务，Solver 无法恢复组间重叠；必须暴露 load/pack/L1-copy 的合法任务粒度。
完成标志为：正确性验证通过、调度报告可解释、同环境性能测量有收益。不能仅用 Solver 目标值宣称达到手工性能。

## 源码依据

路径相对本文件；TileLang 为本机参考仓库，不是本仓构建依赖。

- [TLA scope](../../../third_party/ascend/language/tla/scope.py) 与 [with handler](../../../third_party/ascend/language/tla/code_generator.py)。
- [ScopeOp 定义](../../../third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/Scope/IR/ScopeOps.td)：region 与递归内存副作用。
- [preload 属性](../../../third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)：`PreloadNum/MaxPreloadNum/PreloadLocalBuffer`。
- [CreatePreload](../../../third_party/ascend/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/CreatePreload.cpp)：`rewritePreloadLoop`、`collectPreloadLocalBuffers`、`getPreloadCondition`、`runOnOperation`。
- [regbase pipeline](../../../third_party/ascend/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp)：preload、拆核、内存规划与同步的实际顺序。
- [空循环回归](../../../third_party/ascend/AscendNPU-IR/bishengir/test/Dialect/HIVM/hivm-create-preload-empty-loop.mlir) 与 [preload 同步测试](../../../third_party/ascend/AscendNPU-IR/bishengir/test/Dialect/HIVM/SyncSolver/hivm-ccgss-preload.mlir)。
- `/Users/ayame/workspace/tilelang/tilelang/ascend/language/schedule_hint.py`：Task/Stage 合同。
- `/Users/ayame/workspace/tilelang/tilelang/ascend/pipeline.py`、`transform/z3_scheduler.py`：调度、版本和同步分阶段实现。
