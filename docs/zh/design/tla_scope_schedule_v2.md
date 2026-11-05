# TLA pipeline 接口设计（最新版）

## 1. 接口收敛

流水域与循环合并为 `tla.pipeline(stop)`，移除 `scope.schedule` 和 `auto` 模式。Python 仍使用 for 语法，但无需再套调度域：

```python
@tla.jit
def kernel(X, Y, TILES: tl.constexpr):
    for t in tla.pipeline(TILES):
        tile = tla.alloc((64,), tl.float32, space="ub")
        with tla.scope.vector().task("load", stage=0, order=1):
            tla.load_gm_to_ub(tla.view(X + t * 64, (64,)), tile)
        with tla.scope.vector().task("compute", stage=1):
            with tla.scope.simd():
                # SIMD 指令
                ...
        with tla.scope.vector().task("store", stage=2, order=3):
            tla.store_ub_to_gm(tile, tla.view(Y + t * 64, (64,)))
```

接口定义：

| 写法 | 含义 |
|---|---|
| `for t in tla.pipeline(N)` | 0 起点、步长 1 的流水循环；N 为标量 signed int32/int64，可为运行时值 |
| `with tla.scope.vector():` / `cube()` | pipeline 直接子 scope，声明一个执行任务，由前端生成内部名称 |
| `.task("name")` | 显式指定任务名，不指定 stage/order |
| `.task("name", stage=s, order=o)` | 分别指定可选阶段及 mainloop 排序偏好 |
| `with tla.scope.simd():` | vector task 内部的 VF，不承载 task hint |

pipeline 至少要有一个直接的 vector/cube scope hint。`.task()` 仅属于 vector/cube；普通 scope 在 pipeline 外仍保持原有语义，不自动变成任务。显式 task 不能在 pipeline 外使用，也不能嵌套在另一个任务或内层控制流中。

## 2. stage 与 order

二者独立且均可省略；省略或传 None 时不输出对应属性，不默认填 0。

- **stage**：编译期整数 0..63。在 tick n 执行逻辑迭代 n−stage；不直接决定同一 tick 的指令排列。
- **order**：非负编译期 i32，留给后续 mainloop 排序使用；较小值表达优先安排的偏好，允许重复或不连续。
- bool、浮点数、负数和超范围值拒绝。task 名称是非空编译期字符串，同一 pipeline 内唯一。
- 未命名 scope 使用 `__scope_<line>_<column>` 作为稳定的内部标识；循环分析多次访问同一 AST 节点不会被误判重复。
- 当前前端不推断 stage、不按 order 重排，不插入同步，也不将 hint 解释为越过依赖的许可。

## 3. ttadapter 合同

Python 层删除 schedule 包装，不要求立即改变 NPU-IR 的结构化合同：前端内部生成一个仅包含循环的 `scope.scope {hivm.pipeline}`。

```mlir
// bounds 在域外生成。
scope.scope : () -> () {
  scf.for %iv = %c0 to %N step %c1 {
    %t = arith.index_cast %iv : index to i64
    // 静态本地 alloc、可重算地址等。
    scope.scope : () -> () {
      // load
      scope.return
    } {hivm.task_name = "load", hivm.pipeline_stage = 0 : i32,
       hivm.task_order = 1 : i32, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      // stage/order 未指定。
      scope.return
    } {hivm.task_name = "compute", hivm.tcore_type = #hivm.tcore_type<CUBE>}
  }
  scope.return
} {hivm.pipeline}
```

| 属性 | 归属 |
|---|---|
| `hivm.pipeline` | 前端生成的流水域，unit |
| `hivm.task_name` | 每个直接任务 scope，string；显式或内部生成 |
| `hivm.pipeline_stage` | 任务 scope，可选 i32 |
| `hivm.task_order` | 任务 scope，可选 i32 |
| `hivm.tcore_type` | 执行核，沿用 CUBE/VECTOR |

VF/outline 属性只属于内层 SIMD scope。旧 schedule_domain/schedule_state/required_stage 不再生成。preload_fixed_depth、preload_num、multi_buffer 仍是后端物化结果，不由 Python 接口生成。

## 4. 实现与形态限制

- `pipeline` 标记在现有 scope.py 定义，并导出为 `tla.pipeline`；旧 schedule 类及 dispatch 删除。
- TLA 专用 CodeGenerator 子类识别 pipeline；普通 range/static_range 仍交给原有 codegen。在线 @tla.jit 与离线入口共用该子类，不修改通用 Triton 的 for lowering。
- TLA builder 新增 `create_tla_pipeline`：先在域外生成 index bounds，再创建 scope 和无 iter_args 的 index 循环，向 Python 标量运算暴露 i64 IV。
- 复用已有 loop-carry 分析，拒绝跨迭代 SSA carry。作用域退出恢复 Python 符号表，不允许循环局部值意外逃出 resultless pipeline 域。
- 当前只接受 `pipeline(stop)`，不接受 start/step、for-else、多变量解包、break/continue/return、嵌套 pipeline，或嵌在其他控制流/核 scope 中的 pipeline。
- scope hint 至少一个；buffer 分配和地址计算可在循环顶层，内存效果必须归入 task。完整的内存效果、别名和调度合法性由后端检查。

## 5. 与当前后端的边界

**允许省略 hint 是前端接口能力，不表示当前固定 stage 后端已经实现了无 stage 的调度。**

当前 NPU-IR 的 LowerTLAStages 仍要求所有 task 显式给出 stage，并限制源码 stage 非递减。缺省 stage 的输入可以生成 ttadapter，但设备编译会被该 pass 拒绝；本次不通过偷偷填 0 或移除错误检查来伪装支持。后续支持无 stage 时，需要单独实现依赖合法的阶段分配规则。

order 当前仅透传至 ttadapter，后端尚未据此重排，也未承诺物化过程中保留它。固定 stage 后端还限制任务内部控制流、buffer 形态与 GM 根来源；改进 Python 外形不放宽这些语义限制。

设备编译入口仍限于配套 NPU-IR 的 RegBase TLA 路径。TLA AutoBlockify 继续关闭，避免 launcher 在没有覆盖循环时截断 grid。

## 6. 用例与验证范围

在现有 `scope/task_stage.py` 中更新 attention、copy 和 order 用例，并增加：

- stage/order 的四种有无组合；匿名 vector scope；旧 schedule 接口不存在。
- 缺少直接 scope hint、直接 SIMD hint、嵌套 pipeline、loop-carried 值等负例。
- task 仍只标注 vector/cube，SIMD 嵌套并保留自身 VF 属性。

attention fixture 手工更新为 index pipeline 循环及 i64 逻辑 IV，不是本轮重新编译产生的输出。它仍含 SIMD row 循环、1:2 分工及 pointer_cast GM 路径，只用于前端合同展示，不能宣称当前受限 NPU-IR 能端到端运行。

本次不新增文件。按要求未构建、未运行测试；由于新增了 C++ builder binding，使用新接口前需要后续重建 Python 扩展。

## 7. 待用户评审：如何更好用，同时保持编译器简单

以下是设计提案，尚未实现，也不改变上文的当前合同。示例中的 `...` 代表任务内容。优先评审语义，再决定是否增加语法糖；同一种语义尽量只保留一种写法。

### 7.1 缺省 stage：先支持确定性的退化行为，还是立即支持部分推导？

```python
for t in tla.pipeline(TILES):
    with tla.scope.vector().task("load"):
        ...
    with tla.scope.cube().task("compute"):
        ...
    with tla.scope.vector().task("store"):
        ...
```

用户可能期待“先跑通，之后逐步调优”。最简单的后端实现是：**当全部 task 都未指定 stage 时，按原始顺序执行，物化为一个 stage，不进行跨迭代重叠**。IR 输入仍保持缺省，默认策略由后端显式执行并在调度报告中说明，不由前端偷偷填值。

较复杂的是允许混合指定：

```python
with tla.scope.vector().task("load", stage=0):
    ...
with tla.scope.cube().task("compute"):   # 应放在哪个 stage？
    ...
with tla.scope.vector().task("store", stage=2):
    ...
```

仅凭源码相邻关系不能确定最合适的 stage。给缺省任务填 0、复制前一个 stage 或按序递增，都可能违背用户的性能预期；更一般的补全需要依赖和资源分析。

**建议首期采用“全指定 / 全缺省”两种模式，混合指定给出诊断。** 这能让无 hint 程序具备确定的正确性基线，再支持显式流水调优；不需要新参数或求解器。单 stage 仍然需要正确的同步，不能理解为删除同步。

评审问题：是否接受无 stage 时默认不重叠，还是必须支持“只标最关键的一个 stage”？后者应作为单独的后端能力设计。

### 7.2 order：是硬性位置，还是依赖约束下的排序偏好？

```python
with tla.scope.vector().task("load", stage=0, order=20):
    ...
with tla.scope.cube().task("compute", stage=1, order=10):
    ...
```

在一个稳态 tick 中，这两个 task 可能对应不同逻辑迭代。因此不能仅根据源码的 load→compute 关系禁止交换，也不能直接按照整数排序。需要考察 stage 偏移、当前 buffer 槽、同步和资源依赖。

**建议 order 是偏好，不是数组下标，也不意味着同步或强制先后。** 后端在已经可执行的候选任务中按 order 选择：

- 显式值越小越优先；相同值按源码顺序稳定选择。
- 未指定 order 的任务排在同一候选集合中显式指定的任务之后，并保留彼此的源码顺序。
- 不满足依赖的任务不能因 order 较小而提前；约束未满足时报告实际顺序及阻挡原因，不报精度无关的编译错误。
- scope 的 order 作用于整个 task，首期不拆分 task 内部指令进行交错排布。

只在合法候选集合中增加稳定排序规则，比提供 `before/after` 列表、绝对时间或多级优先级简单。这里仍以正确的依赖图为前提，不能把排序器当成依赖分析。

评审问题：用户是否需要“无法满足该顺序就报错”的硬约束？若只是性能探索，先保留一种软偏好通常足够。

### 7.3 调优一个物理循环体：允许一套 order，还是区分 prologue/mainloop/epilogue？

```python
with tla.scope.cube().task("qk", stage=1, order=0):
    ...
with tla.scope.vector().task("gather_next", stage=0, order=1):
    ...
```

**建议首期只有一套 task 顺序**，用于生成带有效性 guard 的物理循环体；启动和排空阶段通过已有 guard 跳过无效 task，不增加 `prologue_order`、`epilogue_order`。

这与当前“一个展开循环加 guard”的结构更一致。未来若单独 peel 首尾，再以保持既有顺序为默认策略，避免用户为三个区域重复写 hint。

评审问题：是否已有明确案例要求首尾采用不同顺序，且收益足以抵消接口和验证成本？没有实测需求时建议不开放。

### 7.4 scope 的名字是否必填？保留一种简写即可

```python
# 快速书写：仅说明执行位置。
with tla.scope.vector():
    ...

# 性能调优：指定名字，方便定位 IR、诊断和 profiling。
with tla.scope.vector().task("gather", stage=0, order=1):
    ...
```

**建议保留当前两种写法，但不再增加 `.stage()`、`.order()` 或外部字典等同义接口。** 匿名 scope 的位置生成名称仅作为内部标识；源码插行后该名称可能变化，不承诺跨版本稳定。需要长期比对性能时使用显式名称。

评审问题：评审、profile 或 agent 调优是否会频繁引用 task 名字？若会，文档推荐显式命名即可，无须让所有初学者都必须命名。

### 7.5 SIMD 的嵌套是否值得保留？

```python
with tla.scope.vector().task("softmax", stage=2):
    # 可包含属于同一 task 的 vector 核搬运。
    ...
    with tla.scope.simd():
        # VF 内寄存器运算。
        ...
    ...
```

**建议保留这一层嵌套。** 外层定义执行核和调度边界，内层定义 SIMD/VF 代码生成边界，编译器不需要猜测整个 task 是否都能 outline。

不要因缩进较多就重新开放 `simd().task(...)`，否则相同的 task 语义又会出现两种 IR 形态。后续可改善编辑器模板和示例，而不是增加另一种 scope 类型。

评审问题：典型算子中，一个 vector task 是否常包含搬运与多个 SIMD 片段？如果大多如此，显式区分两种边界尤其有用。

### 7.6 buffer 生命周期：是否保持“循环内临时量，循环外持久量”？

```python
for t in tla.pipeline(TILES):
    tile = tla.alloc(...)
    with tla.scope.vector().task("load", stage=0):
        # 写 tile
        ...
    with tla.scope.cube().task("consume", stage=1):
        # 读同一逻辑迭代的 tile
        ...
```

**建议 pipeline 内 alloc 明确属于一次逻辑迭代，物理多缓冲由后端物化；不要求用户手写 `% versions`。** 首期继续拒绝跨迭代 SSA carry，以及直接把循环外本地 buffer 当作流水临时量。

例如在线 softmax 的 `m/l/acc` 是真正的循环递推状态，不能仅靠多分配几个槽解决。未来应单独设计递推状态支持，不以“允许外部 buffer”模糊放宽当前合同。

评审问题：首批目标是否必须覆盖带递推的长序列 FA？若必须，需要优先设计 carry，而不是继续增加更多排序 hint。

### 7.7 先提升可观察性，比再加 hint 更有价值

建议后端输出可选的调度摘要，复用已有 IR dump 或诊断机制，不新增用户算子参数：

```text
Task         Stage  Requested order  Emitted position
qk           1      0                0
gather_next  0      1                1
store        2      unspecified      2

Buffer tile: 3 versions
```

表中阶段、顺序和版本数必须来自实际编译结果；若某个 order 没有实现，附上阻挡它的依赖。让用户看得到“hint 如何被消费”，比不断试填参数更容易调优，也方便 agent 判断下一步修改。

评审问题：最需要的是任务顺序表、buffer 版本表，还是 task 到 profiling 指令段的映射？建议先做前两项，不立即引入复杂的可视化协议。

### 7.8 建议评审优先级

| 优先级 | 决策 | 推荐起点 | 复杂度 |
|---|---|---|---|
| P0 | 缺省 stage 如何执行 | 全缺省串行、全指定流水；暂拒绝混合 | 较低，但仍需后端实现 |
| P0 | order 的确定语义 | 可执行候选集合内的软偏好，稳定处理同值/缺省 | 中等，依赖现有正确依赖分析 |
| P1 | 首尾是否单独配置 | 不配置，复用一套带 guard 的顺序 | 低 |
| P1 | task 名称和 SIMD 嵌套 | 保留当前写法，不扩展同义接口 | 低 |
| P1 | 调度结果是否可观察 | 先输出任务和 buffer 摘要 | 较低 |
| P2 | 递推状态、混合 stage 补全 | 有具体目标算子后单独设计 | 高 |

建议评审时先回答前两个 P0 问题。它们决定无 hint 程序的行为和调优参数的含义，其他语法变化都应围绕这两个稳定合同展开。
