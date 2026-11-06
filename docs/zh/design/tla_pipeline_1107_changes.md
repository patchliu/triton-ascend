# TLA pipeline 固定 stage：实施记录与维护边界

日期：2026-11-07。本次修改基于 `triton-ascend:dev/tla-pipeline-1107` 的
`efa0a959c` 和 `AscendNPU-IR:tla-pipeline-1107` 的 `4600c7185`。
本文记录实际实施结果；[此前评审](tla_pipeline_1107_review.md)保留为修改前的分析快照，
其中的建议不等于本次全部实现。当前接口以[接口说明](tla_scope_schedule_v2.md)为准。

本次将功能收敛为：用户显式指定每个 task 的固定 stage，后端将其转换为既有 preload
正规形，自动完成存储版本、地址轮换与同步。没有实现自动选 stage、缺省 stage 或 order，
没有修改 PlanMemory、跨核 GSS、核内 GSS 的算法，也没有修改 CreatePreload 的展开主体。

## 1. 当前语义与责任分界

```python
for k in tla.pipeline(TILES):
    tile = tla.alloc((64,), tl.float32, space="ub")
    with tla.scope.vector().task("load", stage=0):
        tla.load_gm_to_ub(tla.view(X + k * 64, (64,)), tile)
    with tla.scope.vector().task("store", stage=1):
        tla.store_ub_to_gm(tile, tla.view(Y + k * 64, (64,)))
```

`pipeline(stop)` 的下界是 0，步长是 1。task 必须是循环的直接子 scope，名字非空且
在域内唯一；每个 stage 都是 0..63 的编译期整数，按源码非递减。同 stage 的 task
保持源码顺序。stage s 在 tick n 执行逻辑迭代 n−s，仅当 0≤n−s<stop 时执行其工作。
SIMD 写在 vector task 内的 `scope.simd()` 中；外层 task 保留调度边界，内层才 outline。

用户保证不同逻辑迭代没有算法内存依赖，包括不同 GM 参数发生别名时产生的依赖。
循环内 scratch 每轮必须先写后读；N=1 只是一个物理槽，不赋予跨轮持久状态。
后端维护异步生产/消费及复用顺序，不证明任意 GM 地址表达式的跨迭代独立性。
例如 `X[k+1]=f(X[k])` 或 online-softmax 的跨 KV 块 m/l/acc 递推不能直接按本接口交错。

这是对旧评审建议的明确取舍：没有恢复通用 GM 别名/迭代独立性证明器。当前接口是
程序员提供显式合法调度的受限接口。保留的 GM root 检查只要求 view 追溯到函数参数，
以满足提前执行 CCGSS 的存储身份分析，不能将这项检查理解为 noalias 证明。
将来若要自动调度原本没有独立性保证的一般循环，必须先补依赖分析。

## 2. 前端的收敛

`scope.py` 中 `task(name, *, stage)` 的 stage 现在必填，删除 order 参数和属性生成。
`None` 不再表示待推断 stage。Python 调用缺少 stage 或传入 order 时直接报错，
无需再维护可选字段及其后续补全路径。

`code_generator.py` 不再把普通 vector/cube scope 合成为匿名 task。pipeline 内的直接
scope 必须显式调用 `.task(...)`；pipeline 外普通 core/SIMD scope 的语义不变。
前端仍只发出 domain、task、stage 和 core/VF 属性，不生成物理槽、同步事件或 fill/drain。

删除了早期 `scope/fa_stage.py`、`fa_stage_shape.py`、`fa_stage_v2.py`、`fa_stage_v3.py`
及 `fa_stage_v3.md`。这些实验不是当前功能的维护目标。保留独立的前端合同测试和
正式 `kernels/fa-pipeline` 用例；删除可选 stage、order、匿名任务的正向测试。
`task_stage.py` 还移除了对仓内不存在的 `attention_stages.ttadapter.mlir` 的读取依赖，
继续直接检查实际生成的 IR，不新增静态大快照。

## 3. LowerTLAStages 的实际修改

### 3.1 用同一次 effect 遍历计算 buffer 生命周期

旧实现先遍历 task 检查 effects，再从每个 allocation 沿 use/view 链递归收集 stage。
现在逐操作读取 `MemoryEffectOpInterface`，walk 同时访问结构化控制流的子操作；
既不重复汇总子树，也不丢弃带递归效果 trait 的操作自身的 effects。
每次读写都沿 `ViewLikeOpInterface` 追到 root，并更新该 root 的 first/last stage。

版本数为 `N = last_stage − first_stage + 1`；无访问的 root 取 1。
删除了独立的访问收集器、访问 stage 向上查找、版本回退到 P 的分支，以及
在无结果 task、无 iter_args 循环的有效 SSA 下重复做的 buffer 逃逸遍历。
仍保留 allocation 的源码顺序，避免 DenseMap 的遍历顺序影响生成标记的顺序。

这个算法是保守的 stage span，不是复杂的 live-range 分段优化。
把两个逻辑对象手工绑到同一个 root，会把它们的访问跨度合在一起；前端应优先表达
独立逻辑 allocation，再由内存规划处理物理资源，而不是为人为别名增加调度特例。

### 3.2 修复 domain helper 的 SSA 支配关系

旧实现删除 domain 时将 helper 移入 loop。如果 helper 定义了 loop upper bound，
它会在 scf.for 使用之后才定义，破坏 dominance。
现在将 domain 中的 helper 和 loop 按源码顺序整体移到 domain 之前，再删除空域。
loop-invariant helper 保持在 loop 外，既不重复执行，也不依赖特殊的补救 lowering。

### 3.3 保留必要准入，移除旧形态补偿

保留静态局部分配、显式核归属、已知 effects、无手写同步/不透明调用、无外部局部
scratch 等约束。这些约束决定既有 preload 与同步分析能否处理输入。
本次关键修正是：非 Read/Write effect 不再直接忽略，释放等操作被拒绝；GM 读写必须
保留函数参数来源；显式拒绝不支持的 `hivm.task_order`，避免把 order 静默当成有效提示。

不再兼容“整个 task 就是 outline VF”的旧形态并临时补 shell。
前端已经生成外层 task、内层 SIMD scope，后端只接受这种清晰边界。
嵌套归约和条件仍可存在于 task 内，但用户需保证分支内初始化及跨核参与者协议正确。

### 3.4 LowerTLAStages 的输出合同

| 输入事实 | LowerTLAStages 输出 |
|---|---|
| 所有 task 的最大 stage | P=max(stage)+1 |
| task 的 stage=s | max_preload_num=P，preload_num=P−1−s |
| root 的访问跨度 | multi_buffer=N、preload_local_buffer、mem_unique |
| task scope | 保留 no_inline/core，消费 task_name/pipeline_stage |
| pipeline domain | 删除域壳，loop 标记 hivm.preload_pipeline |

这个 pass 不展开迭代、不规划地址、不插入事件、不重新选 stage。
SyncTLAStages 仍只是选中 MIX 函数调用现有 CCGSS 的适配器。

## 4. CreatePreload、OutlineScope 与其他 pass

编译链继续采用：

```text
LowerTLAStages → 提前 outline 流水内 VF → MarkTCB → SyncTLAStages/CCGSS
→ SplitMix → 原有 affinity 归一化 → InferMemScope/PlanMemory
→ CreatePreload → 核内 GSS → EnableMultiBuffer → LowerMultiBufferCounter
```

OutlineScope 本次没有代码改动。已有的 ancestor 过滤确保提前 outline 仅处理 pipeline
内部标记的 VF；外层 task 留在原循环中，preload 属性不会随 VF 抽取而消失。
拆核后保留共同的 P 和时间原点，不能按每个核剩余 task 重新编号。

CreatePreload 本次只增加两项一致性约束：同一循环所有 scope 的 max_preload_num
必须一致；preload-local root 若仍保留 multi_buffer 计数，它必须与规划地址数一致。
前者避免按最大值或最后一个值重解释不一致的输入，后者避免期望多槽却只得到少槽。
沿用原有“必须有规划地址”的检查，不引入重复计划 ID、全图标记审计或新的调度格式。

既有 loop-filter 入口和 bypass-shape-registry 路径保留：前者选择显式流水，后者保留
同一 preload 编号的所有 task。过滤展开时保留未处理 scope 所需的 workspace 标记。
这些是通用入口语义，不是 FA 算子特例；不应把这部分连同必要支持整体回退。

展开、guard、地址相位、counter 物化都复用原实现。若某 root 有 N 个地址 A，
task 的 p=P−1−s，则既有相位为 `A'[j]=A[(j+p+1)%N]`；结合 tick n 的 counter，
同一逻辑迭代 k 最终访问 `A[(k+P)%N]`。因此 N 不需要等于 P。
buffer 槽数也不决定 flag ID 数；同步算法仍负责 ready/free 和事件生命周期。

本次不修改 T+P 循环上界。最后没有算子工作的 tick 仍可能维护事件和 counter，
不能仅凭 stage 公式把它删掉。PlanMemory、跨核/核内求解器、SplitMix 和通用
OutlineScope 本次均没有新增 TLA 分支。

## 5. FA 主用例的变化与预期 IR

FA 保持七个 task、两阶段和完整 KV 计算：stage 0 是 load_qk/qk；stage 1 是
load_v/softmax/publish_probability/pv/output。tick n 的前一组处理 query n，
后一组处理 query n−1；fill/drain 的无效逻辑迭代由 guard 屏蔽。

原来 K/V 共用 kv_l1，QK/PV 共用 product_l0c，并为 PV 建立较小 subview。
现在分别使用 k_l1/v_l1、qk_l0c/pv_l0c，PV allocation 直接采用其结果尺寸。
计算、stage 和 GM 布局保持不变。旧写法不一定错误，但会把无需共同版本化的对象绑在一起。

| Root | 访问 stage | N | 版本化容量 |
|---|---|---:|---:|
| q_l1 / k_l1 | 0 | 各 1 | 2 / 128 KiB |
| v_l1 / p_l1 | 1 | 各 1 | 128 / 32 KiB |
| qk_l0c / pv_l0c | 0 / 1 | 各 1 | 64 / 4 KiB |
| score | 0、1 | 2 | 每 AIV 64 KiB |
| probability、maximum、denominator、numerator、result、统计输出 scratch | 1 | 各 1 | 每 AIV 合计 21.5625 KiB |

因此显式数组预算为 L1 290 KiB、L0C 68 KiB、每 AIV UB 85.5625 KiB。
L0C 从共享大 root 的两份 128 KiB 降为两个独立 root 合计 68 KiB。
这些是源码与 stage span 推导，不含后端临时量、额外对齐或 VF 栈，不代表已测性能。

后续看 IR 时，应观察 LowerTLAStages 后所有 shell 的 P=2，stage 0 的 p=1，
stage 1 的 p=0，只有 score 的 N=2；OutlineScope 后 softmax/output 的 VF call
仍位于各自的 shell；PlanMemory 后实际地址数满足这些版本；CreatePreload 后
读写同一 query 的 score 使用相同槽；单槽 K/V/P/L0C 仍有复用同步。
两个 AIV 发布 P 后，PV 必须等待两个参与者，不能只看到一个 set/wait 就认定协议正确。

## 6. 保留的测试与验证状态

保留已有固定 stage、同 stage 多 task、VF 边界、控制流、跨核同步及完整 affinity
编译链测试。针对本次修改，已有文件中补充/调整了 domain bound helper、view stage
span、禁止 order/Free/不透明 GM root、preload 深度不一致和规划地址缺槽的用例。
共享 counter 的重复 TLA 用例删除，通用 `preload-shared-counter.mlir` 保留。

已完成的检查仅包括修改后 Python 的 AST 语法解析、两个仓库的 diff 空白检查、
修改过的 C++ 按仓库 `.clang-format` 格式化，以及关键 interface/IR/属性传递的静态复核。
按本次要求，没有构建、运行 pytest/lit、上卡或执行 simulator。新增/调整的测试没有运行。

2026-11-06 的 pass dump 使用旧的 preload_fixed_depth 标记，且对应修改前共享 buffer
的 kernel，不能作为本次最终提交验证通过的证据。恢复验证时，应先生成与两个仓库提交
严格对应的新产物，再检查 T=0/1/短循环、N<P、重复调用、容量失败和 1 AIC : 2 AIV 协议，
随后按 FA 用例原有精度标准比较全部输出，再进行性能采样。

## 7. 提交、清理与后续扩展

AscendNPU-IR 的独立仓和 triton-ascend 子模块使用相同的三个提交：

| 提交 | 内容 |
|---|---|
| `6154df2e6` | LowerTLAStages effect/span 合并、dominance 修复和固定输入合同 |
| `d52e1d843` | CreatePreload 一致性修复与重复 counter 用例清理 |
| `49466d3ef` | 汇总当前后端设计，删除旧方案与临时 IR |

triton-ascend 的 `0ca5c6d2f` 收敛前端及测试，`edb97bbd1` 拆分 FA scratch；
本文件所在的集成提交更新子模块到 `49466d3ef`，并提交评审快照、当前 pass 边界说明和本文。
所有提交都在用户指定分支上完成；没有推送远端。原有 `.gitignore` 用户修改未纳入提交。

后端删除 `preload_unified_contract.md`、`tla_stage_required_changes.md`、
`docs/issue/stage.md`、仓根 `tla_scope_schedule_v2.md` 和临时 `tla.mlir`，统一以
[后端实现合同](../../../third_party/ascend/AscendNPU-IR/docs/design/tla_manual_stages_implementation.md)
为准。旧设计仍可从 Git 历史查看。
依照该仓 AGENTS.md，删除文件后的归属确认请参考
[CODEOWNERS 开发指南的 workflow 2](../../../third_party/ascend/AscendNPU-IR/tools/codeowners-check/DEVELOPER.md)。

将来扩展“手工 hint + 自动调度”，适合在 LowerTLAStages 之前增加从约束和依赖生成完整
固定计划的步骤，再共用目前的物化链。本次没有为尚未实现的功能保留空参数或推断分支。
自动 stage 选择需要依赖合法性、片上容量、核/pipe 资源和性能反馈；这些决策不应下沉成
CreatePreload、PlanMemory 或 GSS 中针对某个算子的特殊规则。
