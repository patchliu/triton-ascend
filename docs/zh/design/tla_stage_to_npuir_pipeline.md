# 手动 stage 的 TLA 流水落地方案

本文承接 [TLA task/stage v2](tla_scope_schedule_v2.md)，收缩为一个明确目标：**所有参与流水的 task 都已手动指定 stage，NPU-IR 只验证并物化该安排，不自动选择 stage，也不做性能搜索。**

一句话方案：**新增 `hivm-lower-tla-stages` pass，把已标注 stage 的任务验证并转换为现有 preload 正规形；在 TLA affinity pipeline 的 SplitMix 前运行它并接入跨核同步，在 PlanMemory 后接入 CreatePreload，随后复用现有核内 GSS 和多缓冲物化。**

状态：实现设计，未新增编译器 pass，本文 IR 为结构示意，未作为完整用例编译。源码依据为 2026-11-05 本地 AscendNPU-IR `9780591de`。

## 1. 首版只承诺什么

输入继续使用现有属性：

```mlir
scope.scope : () -> () {
  scf.for %i = %c0 to %T step %c1 {
    scope.scope : () -> () {
      // load(i)
      scope.return
    } {hivm.task_name = "load", hivm.required_stage = 0 : i32}
    scope.scope : () -> () {
      // compute(i)
      scope.return
    } {hivm.task_name = "compute", hivm.required_stage = 1 : i32}
  }
  scope.return
} {hivm.schedule_domain = "auto", hivm.schedule_state = "logical"}
```

为兼容前端，暂时保留 `schedule_domain="auto"` 的拼写；本实现中它只表示由编译器物化流水及同步，不表示自动求 stage。未来可以另行讨论命名，不作为本轮前置条件。

首版要求：

- 每个 task 都有非负整数 stage；缺省 stage 直接诊断，不默认填 0。
- 每个域仅调度一层 `scf.for`，下界为 0、步长为 1；上界可以是运行时非负整数。其他循环形式先拒绝，不新增循环归一化功能。
- 每个 task 是循环直接子 scope，保持原 Cube/Vector/SIMD 属性。同 stage 的 task 保持源码顺序，不做重排或任务拆分。
- 跨阶段传值通过明确的、固定大小且有地址空间的 buffer；常量及 iv 派生的安全纯地址表达式可以重算。
- 不支持跨阶段 scalar SSA、`iter_args` 数值递推、循环结果逃逸或跨 stage 的条件分支。发现这些情况明确报错。
- 首版 auto 域所在入口不混入手写同步；跨核参与者和分片必须明确。
- 不支持的硬约束只报错，不调整 stage、不退回会忽略标注的旧 pipeline。

这些限制能覆盖独立 tile 的 load→compute→store 和当前单 KV 块 attention 骨架；不声称已经支持含 online-softmax 递推的完整 FA。

## 2. 端到端最少需要的动作

| 动作 | 为什么不能省略 | 最小实现 |
| --- | --- | --- |
| 检查任务和依赖 | 手动 stage 也可能把消费者排在生产者之前 | 验证固定顺序；无法证明安全就报错 |
| 确定 buffer 版本 | 提前加载会覆盖尚未消费的上一 tile | 保守固定版本数，不搜索最优值 |
| stage 转 preload | 现有展开器不认识 `required_stage` | 生成已有 preload 属性与受保护的 scope |
| 核内同步 | 源码发射顺序不代表 DMA/Cube/VF 已完成 | 复用现有 GraphSyncSolver |
| 跨核同步 | AIC 与两个 AIV 不共享一个指令顺序 | SplitMix 前分析并建立匹配的 ready/free 协议 |
| 内存规划和版本轮换 | 逻辑版本必须落成真实地址和一致槽号 | 复用 PlanMemory、CreatePreload、EnableMultiBuffer |
| 首尾 guard | warmup/drain 不能访问不存在的 tile | 复用 CreatePreload；首版无需 PeelPreload |

**可以省掉 Solver、成本模型、重求解、自动重排和 PeelPreload；不能省掉跨核同步、跨迭代存储安全和首尾 guard。**

## 3. 只新增一个主 pass

建议名称：`hivm-lower-tla-stages`，作用于 module 或入口函数，内部按下面的顺序处理，不必为每一步另建 pass。

```text
收集并验证 domain / loop / task
→ 分析 buffer root、alias、读写及执行核
→ 验证用户 stage 和原任务顺序
→ 设置保守版本数、准备跨核依赖锚点
→ 转成 preload scope，消费 domain wrapper
```

pass 不负责重新实现事件分配、物理内存分配或循环展开；这些复用已有设施。但仍需要适配现有 pass 的输入合同，不能只增加一个属性改写 pass 就宣称端到端完成。

### 3.1 最小依赖验证

记录每个 task 的 buffer read/write、访问分片和参与者。追踪 `subview/reinterpret_cast/pointer_cast` 等到 storage root；不同 SSA value 不代表不同内存。VF 必须读取内存效果摘要，不能作为纯调用跳过。

对于已证明的边 `producer(k) → consumer(k+d)`，至少满足：

```text
stage(producer) <= stage(consumer) + d
```

等号时还要符合该 tick 的源码顺序；异步完成由同步保证。首版拒绝一般跨迭代真依赖，允许的跨迭代关系主要是固定槽复用形成的 WAR/WAW。

对独立输出 tile，须证明不同 k 的 GM 写不冲突；未知 GM alias 或无法证明的迭代独立性直接诊断。纯只读 gather 不因离散地址而天然非法，但与可能 alias 的写仍需检查。

### 3.2 不搜索版本数，先用保守策略

令 `P = max(stage) + 1`。对于随 tile 变化、在多个阶段间使用的临时 root，首版可统一设 **N=P**；只在一个 task 内使用且无跨迭代在飞使用的临时存储可维持原行为。循环不变量存储需单独证明只读稳定，不能自动把所有 K/V 都归为不变量。

N=P 是保守起点，不是最优结果，也不替代同步。槽复用仍需保证：

```text
producer(k) 完成 → consumers(k)
last-consumer(k) 完成 → producer(k+N)
```

以实际对齐后的字节数检查容量；容量不足就报告 root、槽数和所需空间，提示用户减少 stage 深度或改变 kernel。首版不自动减槽、不重求解。

循环内 alloc 可在证明不逃逸、大小固定后提升为循环拥有的版本池；每 tile 的初始化仍留在原任务中。引用同一 root 的 view 共享版本池，不能各自复制一份。

这项策略使首版无需修改 workspace 的 P 槽公式：epoch 从 0 开始、step=1 且 N=P。为了进一步缩小实现，**首版可先拒绝需要编译器新增 GM workspace 的计划，只支持已有 ABI 和局部/紧耦合存储**。若 kernel 确实需要 workspace 通道，则其容量、ABI 和槽划分必须补齐，不能隐藏在“直接复用”里。

### 3.3 输出 preload 正规形

转换关系：

```text
max_preload_num = P
preload_num = P - 1 - required_stage
```

例如 stage 0、1、2 对应：

```mlir
scf.for %k = %c0 to %T step %c1 {
  scope.scope : () -> () {
    // load(k)
    scope.return
  } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
  scope.scope : () -> () {
    // compute(k)
    scope.return
  } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
  scope.scope : () -> () {
    // store(k)
    scope.return
  } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
}
```

示意省略了 buffer、算子和原执行核属性，实际转换必须保留它们。stage 有空洞时保留，不压缩用户指定的数字。

保留一个内部“手动 stage 已验证”标记和稳定 loop/task/buffer ID，供 SplitMix、CreatePreload 和错误诊断识别。该标记是新实现合同，不是现有属性。源码 task 名可以保留在 location 或诊断信息中；只有完成物化后才清理未解决标记。

## 4. 现有 TLA pipeline 怎么改

当前 `buildAffinityPipeline()` 已有核内 GSS 与已声明多缓冲的物化，但没有 preload 和跨核自动同步链。

建议最小接入顺序：

```text
现有设备/入口 ABI 归一化
→ LowerTLAStages                       [新增，SplitMix 前]
→ MarkTightlyCoupledBuffer
→ 跨核同步准备/注入                     [对新流水域接入]
→ SplitMixKernel
→ Outline SIMD scope / Inline 普通 scope
→ 现有 AVE 归一化与 InferMemScope
→ PlanMemory                           [固定 N，不静默降槽]
→ 核对实际地址版本数及跨核对应关系
→ CreatePreload                        [新增接入，保留 guard]
→ 现有核内 GraphSyncSolver
→ 现有 EnableMultiBuffer / LowerMultiBufferCounter
→ 必要的跨核同步 ABI 初始化/finalize
→ 现有 LLVM lowering
```

首版不运行 PeelPreload：保留循环内 guard 即可表达正确流水，减少 prologue/epilogue 克隆和 counter 衔接的改动。后续再接 PeelPreload 优化性能。

普通无 task/stage 的 TLA kernel 沿用当前路径。不要把通用 CVPipelining、自动布局、RepositionPreloadWork、Sink/Merge 启发式一起开启，它们会重新选择用户已经确定的安排。

必须阻止 InlineScope 或 canonicalization 在消费 stage 前抹掉任务边界；preload scope 保持 `no_inline` 到 CreatePreload。SIMD scope 的 outline 标记与调度外壳职责不同，不能把整个多 pipe task 当作 VF outline。

### 4.1 跨核部分的最小边界

对于纯 AIV 用例，可先不接 CCGSS，复用核内 GSS 完成第一条编译闭环。但 **Cube/Vector attention 端到端需要跨核协议**，不能把纯 AIV 闭环当成最终完成。

复用 CrossCoreGSS 的分析和事件分配能力，在 SplitMix 前保留完整 AIC/AIV 上下文。必须核对其生成事件的 stage、版本和 guard；普通源码序列的同步结果不一定能原样用于 skew 后的流水。首版优先采用同步在 SplitMix 前确定的路径，避免为了通用向量化而引入 delayed 的 anchors/backup 全套机制；若目标依赖规划后地址才能确定同步，则需要补规划后绑定步骤。

每个事件属于明确 task/edge：ready 跟随生产 epoch，wait 跟随消费 epoch，free 跟随最后消费。不能把所有 set/wait 都留在循环顶层交给 CreatePreload 随意映射。两个 AIV 写 P 的不同 L1 分片时，Cube 的 PV 必须等待两个参与者都 ready；槽释放也必须与实际最后使用匹配。

TLA 当前跳过通用 sync-block-lock finalize。若新增同步使用该机制，就按实际生成的同步恢复必要的 lock 绑定、初始化和 finalize；若选用其他机制，不应无条件引入 lock 流程。

关闭核内自动同步时首版拒绝这类自动物化域。手写同步与生成同步的组合暂不开放。

## 5. 必须适配现有代码的三个地方

### 5.1 CreatePreload 接受同阶段多个 scope

当前非 registry 路径通过 `uniqueSlots[preloadNum]` 收集 scope，同编号后来的 scope 可能替换之前的记录。当前 attention 恰好有多个同 stage task，不能沿用这个限制。

最小修改：对带新验证标记的 loop 使用可容纳多个 scope 的收集路径，保持原顺序，不依赖函数名 registry。不必合并 scope，也不需要启用全局 shape bypass。

### 5.2 scope 外操作不能重复产生副作用

CreatePreload 对普通 scope 外操作可能按阶段克隆，对同步则走特殊路径。新增 pass 必须先保证：

- load/store、初始化、状态更新等有副作用操作归属某个 task。
- 可以重复生成的仅为已证明安全的纯地址/view 表达式。
- 分配和多缓冲 root 由存储物化路径处理。
- 不可归属的操作直接报错。

`isMemoryEffectFree` 不是足够条件；还需考虑除零和不安全推测执行。不要无条件把可能越界的索引读取移到 guard 外。

### 5.3 PlanMemory 不能悄悄减少固定槽数

当前 PlanMemory 重试会移除部分非 preload-local 的 multi_buffer 标记。新路径需保护固定版本计划，并在规划后核对 pointer_cast 地址个数与 N 一致。容量不足返回明确失败，不允许沿用按 P 槽建立的同步却只分配更少槽。

保持现有 affinity 的 `disableTightlyCoupledBufferReuse=true`，首版不增加存储复用优化。

## 6. 展开后的 IR 和首尾语义

stage s 在 tick n 执行 `k=n-s`，有效条件为 `0<=k<T`。stage 1 的结构如下，buffer 选择和算子体省略：

```mlir
%k1 = arith.subi %n, %c1 : index
%ge0 = arith.cmpi sge, %k1, %c0 : index
%ltT = arith.cmpi slt, %k1, %T : index
%valid = arith.andi %ge0, %ltT : i1
scf.if %valid {
  // 选择 k1 对应的 buffer 版本。
  // compute(k1) 及该 epoch 对应的同步。
}
```

算法上最短 tick 数为非空时 `T+P-1`；当前 CreatePreload 扩到 `T+P`，多一个全 inactive tick。首版沿用现有行为，不自行改上界；验证额外 tick 不产生真实计算副作用、不更新逻辑状态、不发生无匹配 wait。T=0 保持空循环，不应只因 drain 而发射同步。

物理 slot 由 CreatePreload 的地址旋转和 EnableMultiBuffer 的 counter 联合生成，前端不再手写 `%N`。实际物理编号允许有统一常量相位，但 producer/consumer 必须使用同一个逻辑 epoch，k 与 k+N 才复用。

当前 attention 的 stage 已全手动标注时，可直接得到：

| task | stage | preload_num，P=5 |
| --- | ---: | ---: |
| load_qk / load_v | 0 | 4 |
| qk / score | 1 | 3 |
| softmax / prob | 2 | 2 |
| pv / output | 3 | 1 |
| store | 4 | 0 |

现有 v2 示例的 `load_v` 没有 stage，落地这条路径前需要在测试 kernel 中显式补上；这里选 0 只作为示例，不代表性能最优。P=5 的保守存储方案也可能超过容量，应如实失败，不宣称该例必然可直接运行。

## 7. 控制流暂时怎么处理

| 场景 | 最小实现行为 |
| --- | --- |
| 动态 T，T=0 或很短 | 支持，复用 CreatePreload guard |
| 非零 lb / 非单位 step | 拒绝；以后再增加 epoch 归一化 |
| task 内部顺序循环 | 整体留在该 task，禁止内部再标 stage；分析全部 effects |
| task 内部单核 if | 可作为整体保留，前提是无跨 task 结果、effect 可分析、无条件跨核协议问题 |
| 一个 if 包围多个 stage | 拒绝，不拆分谓词通道 |
| 跨核条件执行 | 首版要求生产/消费参与者和谓词可证明匹配，否则拒绝 |
| 跨 stage scalar SSA | 拒绝，不生成 shift register |
| iter_args 递推、loop 返回状态 | 首版拒绝，不借多缓冲掩盖真依赖 |
| 嵌套层同时流水化、while、非结构化 CFG | 拒绝自动物化 |

手动 stage 不会自动解决这些问题。例如 stage 0 产生的 `%idx(k)` 被 stage 1 使用，不能直接连接当前 tick 的 SSA 值，否则消费者可能拿到 `%idx(k+1)`；需要通道或重新组织 kernel。又如 `if cond(k)` 中两个阶段应分别使用 `cond(n)` 和 `cond(n-1)`，不能用一个 `cond(n)` 包住全部阶段。

第一版宁可接受范围小，也不能把“scope 具有 stage 属性”当成足够的正确性证明。

## 8. 最小实现与测试顺序

1. **AIV 闭环**：新增主 pass，适配同 stage 多 scope 和固定版本检查；接入 CreatePreload，复用现有 GSS/多缓冲。验证 load→VF→store。
2. **Cube/Vector 闭环**：适配跨核 ready/free、两个 AIV 参与者和必要 ABI，验证 QK→softmax→PV。至此才满足 Cube/Vector 端到端目标。
3. **后续优化**：PeelPreload、减少 buffer 槽数、更复杂控制流。均不作为本轮最少功能。

至少覆盖以下测试：

- T=0、1、小于 P、长循环、运行时 T；每个 `(task,k)` 恰好执行一次。
- 同 stage 多任务、stage 空洞、全部 stage=0；不丢 scope，不复制 effect。
- 固定版本轮换、容量不足、alias/view；复用前完成全部消费者。
- 缺省 stage、逆向依赖、跨 stage scalar、递推和不支持控制流的报错。
- AIC→两个 AIV、两个 AIV→AIC 的 ready/free 和短循环排空。
- 原有无 stage 的 TLA 编译、精度不受影响。

FileCheck 检查属性和 guard 结构，MLIR verifier 检查 SSA/类型，NPU 测试检查数值与运行完成；仅编译出合法 IR 不能证明不会跨核死锁。首版无需以性能搜索结果作为验收，但应测量 guard 和保守 P 槽方案的实际开销。

## 9. 源码定位

- [Affinity pipeline](/Users/ayame/workspace/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:163)：新增主 pass 和 CreatePreload 的主要接入位置；同文件含 TLA lock finalize 分支。
- [RegBase 通用 pipeline](/Users/ayame/workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp)：参考跨核同步、规划、preload、核内同步的顺序，不整体搬入。
- [CreatePreload](/Users/ayame/workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/CreatePreload.cpp)：scope 收集、guard、地址旋转和循环上界。
- [PlanMemory](/Users/ayame/workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp:3267)：固定槽保护和重试行为。
- [CrossCoreGSS](/Users/ayame/workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/CrossCoreGSS.cpp)、[GSS translator](/Users/ayame/workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIRTranslator.cpp)：跨核协议与核内效果分析的复用基础。
- [EnableMultiBuffer](/Users/ayame/workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/EnableMultiBuffer.cpp)：版本选择和循环 counter。
- [PeelPreload](/Users/ayame/workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/PeelPreload.cpp)：后续性能优化，首版不接入。
- [已有 affinity 自动同步测试](/Users/ayame/workspace/AscendNPU-IR/bishengir/test/bishengir-compile/affinity-auto-sync-ave.mlir)、[多缓冲测试](/Users/ayame/workspace/AscendNPU-IR/bishengir/test/bishengir-compile/affinity-multi-buffer.mlir)：扩充现有测试的起点。
