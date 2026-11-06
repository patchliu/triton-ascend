# TLA pipeline 控制流与跨迭代依赖路线图

本文规定建议的支持边界，并以逐场景配对的 TLA 与 MLIR 表达说明固定 stage 到自动调度的演进方向。
核心原则是：**限制调度结构，不限制 task 内必要的计算结构；先分析依赖和选择调度，再物化 stage。**
跨迭代依赖不等于禁止流水，但算法递推不能通过增加 buffer 版本数消除。

## 先看结论

- pipeline 内可以有复杂计算：普通循环和 if 放在 task 内，通常可以保留（S2）。
- 如果循环或 if 包住多个 task，调度器就需要理解整组任务的执行次数和条件，不能直接放开限制（U1/U2）。
- 上一轮的结果可以供下一轮使用，但消费者必须等生产者完成；可以提前的是不依赖该结果的工作（S4/S5）。
- pipeline 携带标量状态值得近期支持。TileLang 用可变标量 buffer 表达它，再分析跨轮依赖；TLA 也可以保留 SSA 表示（U4）。

建议先读 S1、S4、S5 和 U4，再按支持矩阵查其他场景。每节先说明结论，再给 TLA/MLIR 对照，最后讨论编译器需要做什么。

### 术语速查

| 术语 | 本文中的意思 |
|---|---|
| 逻辑迭代 | 原始算法中的第 i 轮 |
| 流水步 / tick | 展开后的一轮执行，其中可能混有不同逻辑迭代的 task |
| stage | task 相对所在逻辑迭代的调度位置；不是自动完成同步的屏障 |
| SSA carry / iter_args | 循环本轮接收上轮结果，并把新结果传给下一轮 |
| scratch / 版本化 | 每轮临时数据 / 给同时在途的迭代分配不同存储槽 |
| guard | 决定某次任务是否执行的条件 |
| fill / drain | 流水开始时逐步填满 / 结束时完成尚未执行完的任务 |
| II | 相邻逻辑迭代的启动间隔；越小通常交叠越多 |
| alias | 两个不同的 buffer 或 view 可能访问同一块存储 |
| 物化 | 把调度决定改写成实际循环、存储槽和同步操作 |

## 1 状态与阅读约定

源码基线：2026-11-07，Triton Ascend `17065bdff`、NPU-IR `830ae9ba1`。
本文是 roadmap，不是已验证能力清单；TLA/MLIR 为设计审查示例，未进行编译、lit 或设备验证。
“当前准入”只表示接口或 verifier 接受该结构，不代表已经证明其同步、数值和性能正确。

状态分为：

- **当前保留**：现有结构有明确用途，应继续支持。
- **条件准入**：结构已能表达，但调用者仍承担未被证明的语义前提。
- **近期补齐**：优先增加针对真实依赖的检查或有限表达能力。
- **后续复合场景**：需要新的调度表示和同步协议，不能只删除 verifier 检查。
- **明确拒绝**：无合法语义或已违反生命周期、执行核、同步契约。

每个场景按 **TLA 表达 → 对应 MLIR → 语义与限制** 排列。两侧对应同一控制流和依赖，
不是从编译器逐字导出的 golden IR；省略常量、index/i64 转换、GM view 描述符和部分重复属性。
例如 Python `scope.vector()` 会增加 tcore_type，纯 AIV 的 MLIR 示例可省略该冗余属性。

所有 Python 片段假定放在 `@tla.jit` kernel 内，使用以下导入；函数参数、启动和返回 ABI
由测试外壳提供。n、queries 等参数分别对应 MLIR 中带 `%` 的同名值。

```python
import triton.language as tl
import triton.language.extra.tla as tla
```

**访存简写约定很重要：** 原文的标量 memref 例子保留，以突出依赖。配套 Python 中
`read_scalar(buf, i)`、`write_scalar(buf, i, value)` 是本文定义的语义记号，分别对应
`memref.load`、`memref.store`，不是已有 TLA API，也不是应该留在 IR 中的不透明 func.call。
GM 指针的标量访存可以写成 `tl.load(GM + i)` / `tl.store(GM + i, value)`；但 UB/L1 的
`tla.buffer` 不能据此当成 Triton pointer。当前 `buf[i]` 返回的是 buffer_access，不是标量。
因此带这些记号的代码是 **TLA 结构示意，不能直接复制执行**。实现 kernel 应使用对应的
DMA/copy 或 VF 操作；S6 提供真实 `tla.simd` API 与 AVE IR 的配对。

这一差别不影响对 task/pipeline/control-flow 层级的讨论，但不能把“后端接受某段 scalar
MLIR”误写成“前端已有完全相同的标量 API”。示例使用现有 `scope.scope`、`scf`、`memref` 表示。
第 4 节 MLIR 为完整函数；之后 MLIR 多为替换片段，Python 通常补出外围循环帮助定位。
两侧仅对照注明的区域；前置上下文均在文字中说明。
`roadmap.*` 属性只表示未来设计意图，当前后端不识别；不能作为可运行接口使用。
`memref.load/store` 用来简化依赖说明，不代表建议用标量操作实现高性能 FA。

TLA 存储声明与 MLIR 类型的公共对应如下（声明片段，不放进每个 task）：

```python
ub_buffer = tla.alloc((1,), tl.float32, space="ub")  # memref<1xf32, #ub>
l1_buffer = tla.alloc((2,), tl.float32, space="l1")  # memref<2xf32, #l1>
```

MLIR 片段共用如下类型别名，测试文件需放在最前面：

```mlir
#ub = #hivm.address_space<ub>
#l1 = #hivm.address_space<cbuf>
```

除明确讨论 MIX 的片段外，Python 示例只使用 VECTOR task，或由 VECTOR task 内嵌 SIMD VF；
调用侧应将 kernel 配置为 AIV。`scope.vector()` 表达的是区域核归属，不能将它与函数
`func_core_type` 的设置视作一条逐字转换规则。相应 MLIR 所在函数使用：

```mlir
attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}
```

pipeline lowering 入口是 `hivm.pipeline` domain 内的 `scf.for`；
`hivm.preload_pipeline` 是 lowering 后的选择标记，用户输入示例不预先添加它。

## 2 FA 三个版本给出的边界

| 版本 | 算法和调度安排 | 对接口的启示 |
|---|---|---|
| v1 | 独立 query tile，BM16，全 KV softmax | 容易满足迭代独立，但不能为了流水限制牺牲分块和数据复用 |
| v2 | BM128/BN256，四块 KV 静态展开为 26 个 task | 静态展开可作有限工具，不应成为表达内部循环的必经方式 |
| v3 | 外层 query 普通循环，内层四次 KV pipeline；状态在外部 | 自然的生命周期表达；当前主要重叠 KV 预取，不能等同更深 FA 流水 |

v2 已有数值错误反馈，不能用其性能数字评判调度设计。v3 的设计也不构成设备正确性保证。
向量 store 的实际 lane 覆盖、view/layout 和跨 AIV 分片必须进入内存效果模型；task 边界不会
自动纠正 task 内的越界或错误写入。详见 [FA 版本说明](README.md)。

## 3 支持矩阵

| 编号 | 场景 | 当前 Python / 后端 | 建议 |
|---|---|---|---|
| S1 | 平坦 task，独立迭代 scratch | 可表达 / 准入 | 当前基础能力 |
| S2 | task 内普通 for、归约、if | 普通结构可表达 / 效果须可追踪 | 保留，检查条件初始化和访问范围 |
| S3 | 外层普通 for 携带 SSA，内层 pipeline | 可表达 / 准入 | 保留，内层完成后再进入下一外层迭代 |
| S4 | 外部持久状态在同一 stage 递推 | 可表达 / 条件准入 | 近期验证递推顺序、入口出口与复用 |
| S5 | 外部状态跨 stage 或跨核访问 | 可表达 / 条件准入 | 必须补依赖分析；准入不代表合法 |
| S6 | task 内嵌 VF | 可表达 / 准入 | 保留 VF 边界，统一 outline |
| U1 | 顶层运行时 if 包 task | Python 拒绝 / 后端不支持该顶层结构 | 后续支持带 guard 的任务组 |
| U2 | 顶层普通 for 包多个 task | Python 拒绝 / 后端不支持该顶层结构 | 后续支持复合循环单元 |
| U3 | 嵌套 pipeline | 拒绝 | 远期，明确双层版本所有权后再开放 |
| U4 | pipeline 自身的 iter_args / results | 拒绝 | 先支持受限标量递推，再考虑一般 SSA |
| U5 | pipeline 放在运行时 if/while/core scope 内 | Python 拒绝；后端不统一禁止这些祖先 | 不能把后端放行等同端到端支持 |
| U6 | break、continue、return、提前退出 | Python 拒绝 | 远期，需要退出和 drain 协议 |
| U7 | task 内 alloc/free、手写同步、不透明调用 | 后端拒绝 | 保留；有调用效果摘要或独立手动区域后另行设计 |
| U8 | task 名称重复、跨 task SSA result、未分类 MIX task | 拒绝 | 保留明确边界；名称仅诊断用途，不作为依赖身份 |
| F1 | 全自动 stage / 部分固定 stage | 当前 Python 必须显式 stage | 共用依赖模型，分阶段开放 |
| F2 | 同一个逻辑 task 的分核条件候选 | 当前不提供专门契约 | 后续明确参与者、exact-once 和 join |

Python 当前还要求 stage 在 `[0,64)` 且源码非递减；NPU-IR 已不要求这两个策略限制，
但 stage 仍需为非负 i32 且小于 INT32_MAX。后续取消全局单调限制，应与依赖及同 pipe 顺序
检查一起做，不能仅删 Python 报错。

`static_range` 会生成直接 task，可在有限规模使用。当前 `direct_scope_nodes` 不递归进入 if，
因此不能承诺“constexpr if 包 task”已被支持；即使理论上可以先折叠，也需先完成前端规范化。

### TLA 与 MLIR 的结构映射

| TLA | MLIR 结构或语义 |
|---|---|
| `for i in tla.pipeline(n)` | `scope.scope {hivm.pipeline}` 内的零起点、单位步长 `scf.for` |
| `scope.vector().task("x", stage=s)` | 带 task_name、pipeline_stage、VECTOR 属性的直接子 scope |
| `scope.cube().task("x", stage=s)` | 同上，core 为 CUBE |
| `scope.simd()` | task 内部带 outline、vector_function 的 VF scope |
| `tla.alloc(..., space="ub")` | UB memref.alloc；位置决定此例的 scratch/持久生命周期 |
| `for j in tl.range(...)` | 普通 scf.for；有循环携带值时使用 iter_args / yield |
| 运行时 `if` / `while` | scf.if / scf.while；结构可表示不等于 pipeline 准入 |
| `read_scalar` / `write_scalar` | **文档语义记号**，分别对应 memref.load / memref.store |

## 4 S1 独立迭代与平坦 task

**当前保留。** 下例每轮处理不同 GM 元素；scratch 每轮先写后读。相同 GM root 的第 i 轮和
第 j 轮访问不同元素，是示例的独立性前提，不是一般 GM alias 自动证明。

**TLA 表达**

TLA 结构示意；标量读写使用第 1 节的语义记号。

```python
for i in tla.pipeline(n):
    tile = tla.alloc((1,), tl.float32, space="ub")
    with tla.scope.vector().task("load", stage=0):
        x = read_scalar(gm, i)
        write_scalar(tile, 0, x)
    with tla.scope.vector().task("compute_store", stage=1):
        x = read_scalar(tile, 0)
        write_scalar(gm, i, x + x)
```

**对应 MLIR**

```mlir
func.func @independent(%gm: memref<?xf32>, %n: index)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scope.scope : () -> () {
    scf.for %i = %c0 to %n step %c1 {
      %tile = memref.alloc() : memref<1xf32, #ub>
      scope.scope : () -> () {
        %x = memref.load %gm[%i] : memref<?xf32>
        memref.store %x, %tile[%c0] : memref<1xf32, #ub>
        scope.return
      } {hivm.task_name = "load", hivm.pipeline_stage = 0 : i32}
      scope.scope : () -> () {
        %x = memref.load %tile[%c0] : memref<1xf32, #ub>
        %y = arith.addf %x, %x : f32
        memref.store %y, %gm[%i] : memref<?xf32>
        scope.return
      } {hivm.task_name = "compute_store", hivm.pipeline_stage = 1 : i32}
    }
    scope.return
  } {hivm.pipeline}
  return
}
```

期望：stage 0 处理 i=tick，stage 1 处理 i=tick-1；P=2，tile 的访问跨度为 2，
两个逻辑版本映射到两个物理槽。编译器维护 fill/drain、有效 guard 和异步复用保护。
同 stage 多 task 仍按约定保序；task 结束不表示全引擎 barrier。

## 5 S2 task 内循环与条件计算

**当前保留。** 下例替换 S1 的循环体，`%flag` 是函数传入的 i1。
循环、分支不改变 task 的动态实例数；if 两条路径均初始化 scratch。

**TLA 表达**

TLA 结构示意；flag 是运行时标量条件。对应 MLIR 只展示循环体。

```python
for i in tla.pipeline(n):
    tile = tla.alloc((1,), tl.float32, space="ub")
    with tla.scope.vector().task("reduce_and_select", stage=0):
        acc = tl.full((), 0.0, tl.float32)
        for j in tl.range(n):
            acc = acc + 1.0
        if flag:
            write_scalar(tile, 0, acc)
        else:
            write_scalar(tile, 0, 0.0)
        value = read_scalar(tile, 0)
        write_scalar(gm, i, value)
```

**对应 MLIR**

```mlir
%tile = memref.alloc() : memref<1xf32, #ub>
scope.scope : () -> () {
  %seed = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %sum = scf.for %j = %c0 to %n step %c1
      iter_args(%acc = %seed) -> (f32) {
    %next = arith.addf %acc, %one : f32
    scf.yield %next : f32
  }
  scf.if %flag {
    memref.store %sum, %tile[%c0] : memref<1xf32, #ub>
  } else {
    memref.store %seed, %tile[%c0] : memref<1xf32, #ub>
  }
  %value = memref.load %tile[%c0] : memref<1xf32, #ub>
  memref.store %value, %gm[%i] : memref<?xf32>
  scope.return
} {hivm.task_name = "reduce_and_select", hivm.pipeline_stage = 0 : i32}
```

内层 `iter_args` 是 task 内部局部 SSA，不是 pipeline 的 loop carry。VF 内的归约循环也属于
此类。调度器可以先将整个 task 当成复合计算，不要求把每次归约迭代展开为 task。

反例：删除 else 分支后，在 `%flag=false` 的路径无条件读取 tile，违反先写后读。
当前效果检查不能据此宣称已完成 definite-assignment 证明。近期应补初始化与 guard 分析，
而不是禁止所有 if。条件有跨核效果时还需检查参与者协议。

## 6 S3 外层普通循环与内层流水

**当前保留。** 完整结构片段，置于 AIV 函数体；参数为 `%queries`、`%blocks` 和 `%gm`。
`%qnext` 表示外层 SSA 状态，内层流水本身不带 iter_args。

**TLA 表达**

TLA 结构示意；qstate 是外层 SSA carry，内层 pipeline 不修改它。

```python
qstate = tl.full((), 0, tl.int64)
for q in tl.range(queries):
    for k in tla.pipeline(blocks):
        with tla.scope.vector().task("kv_step", stage=0):
            offset = q * blocks + k
            x = read_scalar(gm, offset)
            write_scalar(gm, offset, x)
    qstate = qstate + 1
# qstate 对应 MLIR 的 %last；index 转换省略。
```

**对应 MLIR**

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%last = scf.for %q = %c0 to %queries step %c1
    iter_args(%qstate = %c0) -> (index) {
  scope.scope : () -> () {
    scf.for %k = %c0 to %blocks step %c1 {
      scope.scope : () -> () {
        %base = arith.muli %q, %blocks : index
        %offset = arith.addi %base, %k : index
        %x = memref.load %gm[%offset] : memref<?xf32>
        memref.store %x, %gm[%offset] : memref<?xf32>
        scope.return
      } {hivm.task_name = "kv_step", hivm.pipeline_stage = 0 : i32}
    }
    scope.return
  } {hivm.pipeline}
  %qnext = arith.addi %qstate, %c1 : index
  scf.yield %qnext : index
}
```

这是 FA v3 的骨架。它不表示不同 query 已经交叠；每次内层流水排空后外层才推进。
将来若要跨 query 重叠，需要同时安排两层生命周期，不能偷偷改变这段普通外层循环的语义。

## 7 S4 持久状态递推与独立预取

**条件准入，近期重点验证。** 以下是 AIV 函数体片段，参数 `%gm: memref<?xf32>`、`%n: index`。
输入是只读数据，state 在流水外初始化；tile 是每迭代 scratch。

**TLA 表达**

TLA 结构示意；state 在循环外分配，tile 在循环内分配。

```python
state = tla.alloc((1,), tl.float32, space="ub")
write_scalar(state, 0, 0.0)
for i in tla.pipeline(n):
    tile = tla.alloc((1,), tl.float32, space="ub")
    with tla.scope.vector().task("prefetch", stage=0):
        write_scalar(tile, 0, read_scalar(gm, i))
    with tla.scope.vector().task("update", stage=1):
        old = read_scalar(state, 0)
        x = read_scalar(tile, 0)
        write_scalar(state, 0, old + x)
result = read_scalar(state, 0)
```

**对应 MLIR**

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%zero = arith.constant 0.0 : f32
%state = memref.alloc() : memref<1xf32, #ub>
memref.store %zero, %state[%c0] : memref<1xf32, #ub>
scope.scope : () -> () {
  scf.for %i = %c0 to %n step %c1 {
    %tile = memref.alloc() : memref<1xf32, #ub>
    scope.scope : () -> () {
      %x = memref.load %gm[%i] : memref<?xf32>
      memref.store %x, %tile[%c0] : memref<1xf32, #ub>
      scope.return
    } {hivm.task_name = "prefetch", hivm.pipeline_stage = 0 : i32}
    scope.scope : () -> () {
      %old = memref.load %state[%c0] : memref<1xf32, #ub>
      %x = memref.load %tile[%c0] : memref<1xf32, #ub>
      %next = arith.addf %old, %x : f32
      memref.store %next, %state[%c0] : memref<1xf32, #ub>
      scope.return
    } {hivm.task_name = "update", hivm.pipeline_stage = 1 : i32}
  }
  scope.return
} {hivm.pipeline}
%result = memref.load %state[%c0] : memref<1xf32, #ub>
```

依赖为 `update(i) → update(i+1)`，而 `prefetch(i+1)` 不依赖 state。
因此可以保留递推顺序，同时预取下一轮输入。state 为一份，tile 按跨度版本化。
同 stage 并不替代异步同步；入口初始化、update 的完成、出口读取都必须处于依赖图中。

FA v3 的 m/l/acc 属于这种持久状态，但其 task 跨多个引擎，协议比标量示例复杂得多。
生命周期默认可由 allocation 词法位置推断，进入优化前应固化到 IR；不能让后续 hoist
allocation 无意改变 scratch/持久状态的含义。

## 8 S5 递推依赖与固定 stage 顺序冲突

**算法合法，但当前指定的 stage 和展开顺序冲突，应由依赖检查拒绝。** 替换 S4 的循环体，state 仍在外面且已初始化：

**TLA 表达**

反例的 TLA 结构示意。对应 MLIR 替换 S4 循环体；初始化和最终读取沿用 S4。

```python
state = tla.alloc((1,), tl.float32, space="ub")
write_scalar(state, 0, 0.0)
for i in tla.pipeline(n):
    old_copy = tla.alloc((1,), tl.float32, space="ub")
    with tla.scope.vector().task("read_previous", stage=0):
        write_scalar(old_copy, 0, read_scalar(state, 0))
    with tla.scope.vector().task("write_current", stage=1):
        old = read_scalar(old_copy, 0)
        write_scalar(state, 0, old + 1.0)
result = read_scalar(state, 0)
```

**对应 MLIR**

```mlir
%old_copy = memref.alloc() : memref<1xf32, #ub>
scope.scope : () -> () {
  %old = memref.load %state[%c0] : memref<1xf32, #ub>
  memref.store %old, %old_copy[%c0] : memref<1xf32, #ub>
  scope.return
} {hivm.task_name = "read_previous", hivm.pipeline_stage = 0 : i32}
scope.scope : () -> () {
  %old = memref.load %old_copy[%c0] : memref<1xf32, #ub>
  %one = arith.constant 1.0 : f32
  %next = arith.addf %old, %one : f32
  memref.store %next, %state[%c0] : memref<1xf32, #ub>
  scope.return
} {hivm.task_name = "write_current", hivm.pipeline_stage = 1 : i32}
```

这段算法每轮执行 `state = state + 1`。初始值为 0，执行两轮应得到 2。
但当前展开保留 task 的源码顺序：同一流水步先读本轮 state，再写回上一轮的结果。
即使所有读写都立即完成，且 old_copy 已经正确分成两个存储槽，仍会出错：

| 流水步 | 按顺序执行的操作 | state |
|---|---|---|
| 0：填充 | 第 0 轮读取 0，保存到 old_copy[0] | 0 |
| 1：交叠 | 第 1 轮先读取 0；第 0 轮再写回 1 | 1 |
| 2：排空 | 第 1 轮用先前读到的 0 加 1，写回 1 | 1，预期为 2 |

第 1 轮需要读取第 0 轮产生的 1，却读早了。增加 old_copy 的槽数不能产生尚未算出的值。
在读取前插等待、却把同一执行序列中的生产者仍留在后面，也不能解决问题。

可行的修改有三种：两个 task 都用 stage 0 并保持先读后写；合成一个更新 task，像 S4
那样只提前预取独立输入；或由未来调度器改成先写上一轮、再读本轮，并保证写入完成。
因此，这不是禁止所有跨 stage 递推，而是拒绝这份具体调度。

**责任边界：** 用户指定 stage，编译器负责检查它是否破坏原始循环的依赖。
本例应报调度冲突，不能要求用户自行发现错误，也不能仅靠展开后的同步插入补救。

近期诊断应明确列出：`state`、生产者 `write_current(i)`、消费者 `read_previous(i+1)`、
距离 1、指定 stage 和冲突顺序。固定模式报错；自动模式可重新选择 stage/发射顺序，
或退化为顺序执行。仅仅“跨 stage”不构成拒绝理由；本例冲突还依赖实际发射顺序。

## 9 S6 task 与 VF 两层边界

**当前保留。** 以下是 pipeline 循环内的 task；`%tile` 是循环直接分配的静态 UB scratch，
并在本逻辑迭代已初始化。这个场景用实际 SIMD API 对照 AVE，区别于其他场景的标量语义简写。

**TLA 表达**

真实 TLA SIMD API 片段。tile 改为 64 个 FP32 元素，且本逻辑迭代的 64 个 lane 均已初始化；下方 MLIR 同步改为 AVE，不再用标量 load/store 替代。两侧均只展示 task。

```python
# 上下文：pipeline 的直接子 allocation 为
# tile = tla.alloc((64,), tl.float32, space="ub")
# 前面的任务已初始化 tile 的全部 64 个元素。
with tla.scope.vector().task("vector_compute", stage=1):
    with tla.scope.simd():
        x = tla.simd.vlds(tile[0])
        y = tla.simd.vadd(x, x)
        tla.simd.vsts(tile[0], y)
```

**对应 MLIR**

```mlir
// 文件头还需：#b32 = #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>
// 上下文：%tile : memref<64xf32, #ub>，%c0 : index。
scope.scope : () -> () {
  scope.scope : () -> () {
    %x = ave.hir.vload <NORM> %tile[%c0]
        : memref<64xf32, #ub> into vector<64xf32, #b32>
    %mask = ave.hir.pge <ALL> : vector<64xi1, #b32>
    %y = ave.hir.vadd %x, %x, %mask
        : vector<64xf32, #b32>, vector<64xi1, #b32>
    ave.hir.masked_store <NORM_B32> %tile[%c0], %mask, %y
        : memref<64xf32, #ub>, vector<64xi1, #b32>, vector<64xf32, #b32>
    scope.return
  } {outline, no_inline, hivm.vector_function,
     hivm.func_core_type = #hivm.func_core_type<AIV>}
  scope.return
} {hivm.task_name = "vector_compute", hivm.pipeline_stage = 1 : i32,
   hivm.tcore_type = #hivm.tcore_type<VECTOR>}
```

task 保留 preload 边界，内层 VF outline 成函数调用；CCGSS 通过调用汇总读写。
不应提前 flatten stage scope，也不应让同一个 scope 同时承担 task 和 VF。
当前用户输入中的不透明 call 被拒绝，与经过验证后由 OutlineScope 产生的 VF call 不矛盾。

## 10 U1 与 U2 顶层分支和循环不能只删除检查

### U1 条件 task

**当前不支持；后续以带 guard 的任务组支持。** 以下位于 pipeline 循环体，tile 为直接 alloc：

**TLA 表达**

当前前端拒绝的 TLA 结构示意；active 是运行时条件。两侧 MLIR/Python 对照 if 区域。

```python
for i in tla.pipeline(n):
    tile = tla.alloc((1,), tl.float32, space="ub")
    if active:  # task 不再是 pipeline 的直接子节点
        with tla.scope.vector().task("load_if_active", stage=0):
            write_scalar(tile, 0, read_scalar(gm, i))
        with tla.scope.vector().task("store_if_active", stage=1):
            write_scalar(gm, i, read_scalar(tile, 0))
```

**对应 MLIR**

```mlir
scf.if %active {
  scope.scope : () -> () {
    %x = memref.load %gm[%i] : memref<?xf32>
    memref.store %x, %tile[%c0] : memref<1xf32, #ub>
    scope.return
  } {hivm.task_name = "load_if_active", hivm.pipeline_stage = 0 : i32}
  scope.scope : () -> () {
    %x = memref.load %tile[%c0] : memref<1xf32, #ub>
    memref.store %x, %gm[%i] : memref<?xf32>
    scope.return
  } {hivm.task_name = "store_if_active", hivm.pipeline_stage = 1 : i32}
}
```

需要为两个 task 保存“同一逻辑迭代的 active”，即 stage 1 使用 active(i)，不是物理
tick 对应的 active(i+1)。若条件读取可变状态，还要保留求值时点和依赖，不能在每个
stage 随意重算。guard 必须延续到同步插入；跳过生产者的迭代不能留下无条件等待。

优先支持：由 IV/只读参数决定、在参与核间一致、producer/consumer 成组激活的条件。
数据相关、不同核各自判断的条件推迟到参与者协议可分析之后。

### U2 顶层复合循环

**当前不支持；后续以复合调度单元支持。** 以下位于 pipeline 的第 i 轮：

**TLA 表达**

当前前端拒绝的 TLA 结构示意；chunks 是内层循环次数。MLIR 仅展示 j 循环。

```python
for i in tla.pipeline(n):
    tile = tla.alloc((1,), tl.float32, space="ub")
    for j in tl.range(chunks):
        with tla.scope.vector().task("chunk_load", stage=0):
            offset = i * chunks + j
            write_scalar(tile, 0, read_scalar(gm, offset))
        with tla.scope.vector().task("chunk_consume", stage=1):
            write_scalar(out, j, read_scalar(tile, 0))
```

**对应 MLIR**

```mlir
scf.for %j = %c0 to %chunks step %c1 {
  scope.scope : () -> () {
    %base = arith.muli %i, %chunks : index
    %offset = arith.addi %base, %j : index
    %x = memref.load %gm[%offset] : memref<?xf32>
    memref.store %x, %tile[%c0] : memref<1xf32, #ub>
    scope.return
  } {hivm.task_name = "chunk_load", hivm.pipeline_stage = 0 : i32}
  scope.scope : () -> () {
    %x = memref.load %tile[%c0] : memref<1xf32, #ub>
    memref.store %x, %out[%j] : memref<?xf32>
    scope.return
  } {hivm.task_name = "chunk_consume", hivm.pipeline_stage = 1 : i32}
}
```

该输入首先存在 stage 归属歧义：0/1 描述外层 i 还是内层 j？不能默认解释。
若调度内层 j，应建立独立的循环调度域；若只将整个内层循环放在外层某个 stage，
内部应保持串行语义，作为带读写、资源和代价摘要的复合单元。
本例 out[j] 在外层之间复用，还需要外层依赖，不能假定外层独立。

当前替代方式：只有单核普通计算循环时放进一个 task（S2）；需要跨核多 task 时将其
作为独立内层 pipeline，并让外层保持普通循环（S3）。不要用无限静态展开替代分层表示。

## 11 U3 U4 U5 后续的嵌套和循环携带值

### U3 两层 pipeline

**当前明确拒绝，远期考虑。** `%outer_count`、`%inner_count` 和 buffer 是函数上下文：

**TLA 表达**

当前明确拒绝的 TLA 结构示意。两个 tla.pipeline 对应两个 hivm.pipeline domain。

```python
for i in tla.pipeline(outer_count):
    for j in tla.pipeline(inner_count):
        with tla.scope.vector().task("inner", stage=0):
            write_scalar(out, j, read_scalar(gm, j))
```

**对应 MLIR**

```mlir
scope.scope : () -> () {
  scf.for %i = %c0 to %outer_count step %c1 {
    scope.scope : () -> () {
      scf.for %j = %c0 to %inner_count step %c1 {
        scope.scope : () -> () {
          %x = memref.load %gm[%j] : memref<?xf32>
          memref.store %x, %out[%j] : memref<?xf32>
          scope.return
        } {hivm.task_name = "inner", hivm.pipeline_stage = 0 : i32}
      }
      scope.return
    } {hivm.pipeline}
  }
  scope.return
} {hivm.pipeline}
```

本例 out 跨外层迭代共享，不能自动认为外层可交叠。完整支持需明确二元迭代身份 (i,j)、
版本所有者、外层复用何时等待内层 drain，以及两个层级的事件作用域。
首期只实现 S3，不承诺两个 pipeline 可任意嵌套。

### U4 pipeline 自身携带 SSA

**当前拒绝；后续先支持受限标量递推。** 函数上下文含 `%zero: f32` 和 `%n: index`：

**TLA 表达**

当前拒绝的未来表达草图；acc 从 task 逃逸并成为 pipeline carry。下方 MLIR 是拟议数据流形态，不是当前 codegen 对这段 Python 的输出。

```python
acc = tl.full((), 0.0, tl.float32)
for i in tla.pipeline(n):
    with tla.scope.vector().task("accumulate", stage=0):
        acc = acc + read_scalar(gm, i)
write_scalar(out, 0, acc)
```

**对应 MLIR**

```mlir
scope.scope : () -> () {
  %sum = scf.for %i = %c0 to %n step %c1
      iter_args(%acc = %zero) -> (f32) {
    %next = scope.scope : () -> f32 {
      %x = memref.load %gm[%i] : memref<?xf32>
      %y = arith.addf %acc, %x : f32
      scope.return %y : f32
    } {hivm.task_name = "accumulate", hivm.pipeline_stage = 0 : i32}
    scf.yield %next : f32
  }
  memref.store %sum, %out[%c0] : memref<?xf32>
  scope.return
} {hivm.pipeline}
```

它同时超出当前无结果 task、无 iter_args pipeline 和 domain 顶层效果的约束。
未来不能靠把 iter_args 转成单槽 memref 就宣称支持；仍须保留距离 1 的真实依赖、
初值、空循环结果、排空后的最终值。该例应能退化为普通顺序求和，不能自动改成改变
浮点结合顺序的树形归约。

#### 外提 alloc 能解决什么

可以把 SSA carry 转成 pipeline 外的一份独立 alloc：循环前写入初值，每轮读取、更新
并写回，排空后读取最终值。这解决的是状态如何保存和传递的问题，也能自然保留零次
迭代返回初值的语义。

**外提 alloc 不会自动保证调度正确。** state 必须作为持久状态保留，不能按每轮独立
scratch 多缓冲；下一轮读取前，上轮更新必须已经完成。S5 的 state 本来就在循环外，
仍因下一轮先读、上一轮后写而出错。

因此，U4 的 alloc 转换应与跨迭代依赖检查一起交付。首期可以限定单核、同一个 task
完成读—修改—写，并验证实际执行顺序和完成条件。不能仅因转换成功，就放行任意
跨 task、跨 stage 或跨核状态更新。用户可以继续写 SSA 形式，由编译器完成转换。
完整责任划分见第 14 节。

#### TileLang 如何处理同类状态

TileLang Ascend 已有标量递推的测试，结构如下（省略 kernel 外壳）：

```python
accum = T.alloc_var("float32", init=0)
out = T.alloc_shared((16,), "float32")
for i in T.Pipelined(16, num_stages=2):
    previous = accum
    for _ in T.serial(2):
        accum += A[i]
    out[i] = previous
T.copy(out, O)
```

输出应为 `out[i] = 2 * sum(A[0:i])`：previous 必须保存本轮更新前的值，下一轮必须接着
本轮更新后的 accum 计算。仓库中有这个设备测试；本文没有重新运行它。

它与 U4 的区别在表示方式。`T.alloc_var` 创建一个 `[1]`、`local.var` 类型的 buffer，
初始化和更新在 TIR 中是 buffer 读写。因此自动调度器看到的是可变状态，而非直接接收
`scf.for iter_args`。这只是 IR 的选择，不意味着最终一定要占一个 UB 槽。

调度器从读写中建立“读取旧值(i) → 更新(i) → 读取旧值(i+1)”的依赖。
local.var 不进入自动多缓冲候选，其跨轮依赖使用距离 1；不会通过多复制几份累加器
来消除递推。求解器可以增大迭代启动间隔，同时提前执行不依赖 accum 的输入预取。
自动调度无可行解时有顺序回退，手动约束不可行时则报错。

PTO codegen 后续还可以通过 PTODSL 将变量更新重新推导成循环携带 SSA。
因此，TileLang 的做法不是禁止 SSA，而是在调度阶段用 buffer 读写表达这类状态。

**对 TLA 的建议：近期先支持单核、受限标量 carry。** 可以保留 iter_args/yield 并直接
提取距离 1 的依赖，也可以转成显式状态存储。两条路线都必须保留初值、零次迭代结果、
逐轮更新顺序和排空后的最终值；转成 memref 本身不等于已经解决调度问题。
跨核 carry、多 task 传递及复杂条件合并可放到后续阶段。

源码依据：

- [标量快照与流水更新测试](/Users/ayame/workspace/tilelang/testing/ascend/auto_schedule/test_schedule_integration.py:819)
- [alloc_var 的存储表示](/Users/ayame/workspace/tilelang/tilelang/language/allocate.py:103)
- [跨轮依赖转换](/Users/ayame/workspace/tilelang/src/ascend/transform/auto_schedule.cc:1008)
- [PTO 循环携带 SSA 生成说明](/Users/ayame/workspace/tilelang/src/ascend/codegen/codegen_pto.cc:7036)

### U5 整个 pipeline 条件执行

**Python 当前拒绝；后端祖先检查已放宽，仍需独立验收。** 它与 U1 的条件 task 不同：

**TLA 表达**

当前 Python 拒绝的 TLA 结构示意；enabled 是运行时条件，不是 constexpr。

```python
if enabled:
    for i in tla.pipeline(n):
        with tla.scope.vector().task("copy", stage=0):
            write_scalar(out, i, read_scalar(gm, i))
```

**对应 MLIR**

```mlir
scf.if %enabled {
  scope.scope : () -> () {
    scf.for %i = %c0 to %n step %c1 {
      scope.scope : () -> () {
        %x = memref.load %gm[%i] : memref<?xf32>
        memref.store %x, %out[%i] : memref<?xf32>
        scope.return
      } {hivm.task_name = "copy", hivm.pipeline_stage = 0 : i32}
    }
    scope.return
  } {hivm.pipeline}
}
```

优先考虑所有参与核一致且在进入前确定的 enabled：初始化、循环、drain、退出都受同一个
激活域控制。跨核不一致的 enabled 不能仅靠条件包裹解决，仍可能出现单边等待。

### U6 数据相关 while 与提前退出

**当前 Python 拒绝，远期考虑。** 以下完整循环片段位于函数体，`%keep: memref<1xi1>`、
`%gm: memref<?xf32>`、`%out: memref<?xf32>` 为函数参数，c0/c1 是 index 常量。
每次检查 keep 再执行一次 pipeline；示例不是允许提前越过已经发出的工作。

**TLA 表达**

当前 Python 拒绝的 TLA 结构示意；keep 是可读取的布尔状态。此例仅表示整个 pipeline 位于 while 内，中途 break 另需退出协议。

```python
while read_scalar(keep, 0):
    for i in tla.pipeline(n):
        with tla.scope.vector().task("copy", stage=0):
            write_scalar(out, i, read_scalar(gm, i))
```

**对应 MLIR**

```mlir
scf.while : () -> () {
  %continue = memref.load %keep[%c0] : memref<1xi1>
  scf.condition(%continue)
} do {
  scope.scope : () -> () {
    scf.for %i = %c0 to %n step %c1 {
      scope.scope : () -> () {
        %x = memref.load %gm[%i] : memref<?xf32>
        memref.store %x, %out[%i] : memref<?xf32>
        scope.return
      } {hivm.task_name = "copy", hivm.pipeline_stage = 0 : i32}
    }
    scope.return
  } {hivm.pipeline}
  scf.yield
}
```

整域 while 是第一步；在 pipeline 中途数据相关退出还需要 active 状态、在途任务排空、
事件归还和最终状态选择。SCF 没有可随意插入 scf.for 的通用 break op，不能用一个早退
terminator 假装已有 lowering。未来需专门的退出表示和物化方案，不应偷偷省略 drain。

### U7 task 内分配和不透明调用

**当前拒绝；先保持限制。** 以下替换 S1 的一个 task。假设模块声明了
`func.func private @unknown(memref<1xf32, #ub>)`，调用仍没有可供调度的内存效果摘要。

**TLA 表达**

拒绝场景的 TLA 表达草图。unknown_external 表示未内联调用；release_storage 是本文生命周期记号，不是现有 TLA API。普通 @tla.jit helper 若已内联，不属于这里的不透明 call。

```python
for i in tla.pipeline(n):
    with tla.scope.vector().task("opaque", stage=0):
        temporary = tla.alloc((1,), tl.float32, space="ub")
        unknown_external(temporary)  # 拟议外部调用，当前无对应准入能力
        release_storage(temporary)  # 语义记号，对应 memref.dealloc
```

**对应 MLIR**

```mlir
scope.scope : () -> () {
  %temporary = memref.alloc() : memref<1xf32, #ub>
  func.call @unknown(%temporary) : (memref<1xf32, #ub>) -> ()
  memref.dealloc %temporary : memref<1xf32, #ub>
  scope.return
} {hivm.task_name = "opaque", hivm.pipeline_stage = 0 : i32}
```

这里有两个独立问题：分配生命周期没有进入循环 scratch 规划；call 的读写、资源和同步
效果未知。将 alloc 提到 pipeline 顶层只能解决前者。支持 helper 复用应优先在调度前
内联可见实现；不内联调用需要效果摘要。手写 wait/set 与自动协议混用同样需要明确的
所有权边界，不能通过把 op 包进一个 scope 来规避检查。

### U8 task 名称重复

**当前明确拒绝。** 名称用于诊断和调度报告，两个不同 task 不能重名。这里仅例示 U8 的
名称问题；跨 task SSA result 已在 U4 展示。MIX 核分类由 cube()/vector() 显式表达。

**TLA 表达**

当前会被前端拒绝；标量读写仍使用第 1 节语义记号。

```python
for i in tla.pipeline(n):
    with tla.scope.vector().task("same_name", stage=0):
        write_scalar(out, i, read_scalar(gm, i))
    with tla.scope.vector().task("same_name", stage=1):
        write_scalar(out, i, read_scalar(gm, i))
```

**对应 MLIR**

下列为 pipeline 循环体，后端也应拒绝名称重复。

```mlir
scope.scope : () -> () {
  %x = memref.load %gm[%i] : memref<?xf32>
  memref.store %x, %out[%i] : memref<?xf32>
  scope.return
} {hivm.task_name = "same_name", hivm.pipeline_stage = 0 : i32}
scope.scope : () -> () {
  %x = memref.load %gm[%i] : memref<?xf32>
  memref.store %x, %out[%i] : memref<?xf32>
  scope.return
} {hivm.task_name = "same_name", hivm.pipeline_stage = 1 : i32}
```

## 12 F1 自动调度与部分固定

**未来提案，非当前接口。** 在 S1 循环体中可考虑如下形态：

**TLA 表达**

未来 TLA API 提案：省略 stage 表示自动选择。当前 task() 要求 stage，此写法不能直接使用；具体 API 尚未确定。MLIR 只展示循环体。

```python
for i in tla.pipeline(n):
    tile = tla.alloc((1,), tl.float32, space="ub")
    with tla.scope.vector().task("load"):  # 提案：auto stage
        write_scalar(tile, 0, read_scalar(gm, i))
    with tla.scope.vector().task("store"):  # 提案：auto stage
        write_scalar(out, i, read_scalar(tile, 0))
```

**对应 MLIR**

```mlir
%tile = memref.alloc() : memref<1xf32, #ub>
scope.scope : () -> () {
  %x = memref.load %gm[%i] : memref<?xf32>
  memref.store %x, %tile[%c0] : memref<1xf32, #ub>
  scope.return
} {hivm.task_name = "load", roadmap.auto_stage}
scope.scope : () -> () {
  %x = memref.load %tile[%c0] : memref<1xf32, #ub>
  memref.store %x, %out[%i] : memref<?xf32>
  scope.return
} {hivm.task_name = "store", roadmap.auto_stage}
```

auto_stage 仅示意“决策未固定”，不是要求最终新增此属性。实现时可用专门调度 IR 的
可选 stage 字段；不能让缺少 stage 的拼写错误静默变成 stage 0。
先支持全固定和全自动；混合约束后做。两者共用依赖、存储与同步物化，固定模式不悄悄
改用户 stage。自动模式在无有益可行流水时可保序退化，并报告原因。

用户 task 可继续是算法分组；内部可分解为 MTE/Cube/Fixpipe 等节点，但 VF 边界及明确
原子区域应保留。stage、各 pipe 的发射顺序、成本模型不是同一个概念。
自动调度不默认改变 BM/BN、数值路径或 reduction 结合顺序；这些属于另一层优化。

## 13 F2 分核候选与条件同步

**后续复合场景。** 以下放在 MIX 函数中的 VECTOR task 内，%sid 为 AIV sub-block id，
%p 是共享 L1，两个 AIV 写不同元素。此处只用标量 store 描述分片关系：

**TLA 表达**

TLA 分核控制流示意，置于已有 VECTOR task 内；p 是共享 L1。write_scalar 只说明分片语义，不能据此声称设备支持直接标量写 L1；真实实现用 UB→L1 搬运。

```python
sid = tla.sub_block_id()
if sid == 0:
    write_scalar(p, 0, a)
else:
    write_scalar(p, 1, b)
```

**对应 MLIR**

```mlir
%is_aiv0 = arith.cmpi eq, %sid, %c0 : index
scf.if %is_aiv0 {
  memref.store %a, %p[%c0] : memref<2xf32, #l1>
} else {
  memref.store %b, %p[%c1] : memref<2xf32, #l1>
}
```

真实 FA 中对应两个 AIV 发布 P 的不同分片；AIC 消费必须等待两个参与者都完成。
需要表示逻辑访问角色、参与者集合、候选 guard、exact-once 和全部到达的 join。
只知道执行核为 VECTOR 不够；不同 AIV 的 UB 还必须按不同物理存储建模。

本例并不是要求给每个普通 if 添加手工协议。如果双方都必然写各自分片，应从布局和核映射
推导。只有无法推导的分核 dispatch 才考虑专门契约，不能把所有 task 变成全核 barrier。
若某一分支完全不发布而消费者仍等待两方，应拒绝或要求另一种明确的可选生产协议。

## 14 跨迭代依赖的调度规则

### 正确性由谁负责

**用户描述算法和调度意图，编译器保证流水变换保持原始循环语义。** 用户需要理解 stage
的含义，但不应靠人工推演展开代码，才能避免读到旧值。以下是建议建立的接口契约，
不代表当前实现已经具备完整检查：

> `tla.pipeline` 保留原始循环的计算语义；显式 stage 是需要通过依赖检查的调度约束。
> 如果指定安排与依赖冲突，编译器应报错，不能静默生成错误代码。

| 模式 | 用户负责 | 编译器负责 |
|---|---|---|
| 手动 stage | 提供期望的任务安排 | 验证依赖；冲突时报告原因，不擅自改 stage |
| 自动 stage | 表达算法及数据访问 | 选择合法顺序；必要时减少交叠或按约定顺序回退 |
| 访问关系无法推断 | 必要时提供明确的别名、访问范围等契约 | 保守处理或拒绝相关变换，不能默认各轮独立 |

因此，S5 是编译器应诊断的调度冲突，不应仅作为“算子作者注意事项”。诊断应指出
state、前后两个 task、相差一轮的依赖，以及具体哪次读取被提前了。

### 先判断顺序，再插同步

两类问题需要不同处理：

- 生产者已排在消费者前面，但异步执行尚未完成：可以插入同步，等待结果就绪。
- 消费者被排在生产者前面：需要调整调度或拒绝；不能只加 wait，同时把生产者留在后面。

依赖应在流水展开前提取，或以等价方式保留原始逻辑迭代关系。例如 S5 必须记录：

```text
write_current(i) → read_previous(i+1)，迭代距离为 1
```

如果先展开，再只按新代码顺序推导普通内存依赖，分析器可能只是维护了错误的新顺序。
因此，后端“已有同步插入”并不能单独证明跨迭代递推是安全的。

### 依赖如何约束调度

先区分两种等待：临时存储槽还没用完，可以增加槽数；下一轮需要上一轮算出的结果，必须等计算完成。
对 u(i) → v(i+d)，令 t 为相对开始时刻、II 为迭代启动间隔、L 为依赖所需延迟：

```text
t(v) - t(u) + d × II >= L
```

同轮 d=0，下一轮 d=1。递推环给出 II 的下界，资源与存储容量可进一步增大该下界。
固定 stage 仍需检查周期内顺序；相同 stage 不表示同步完成，不同 stage 也不必然非法。

| 情况 | 固定 stage | 自动 stage |
|---|---|---|
| 合法且可交叠 | 物化原安排 | 选择可行安排 |
| 存在递推但部分工作独立 | 保持递推与指定顺序 | 增大 II 或调整任务安排，保留独立重叠 |
| 用户安排与依赖冲突 | 报告 buffer、两端 task、迭代距离和冲突顺序 | 搜索其他合法安排 |
| 没有有益的流水安排 | 不静默修改用户 stage | 顺序退化，报告原因 |
| 效果或 alias 无法分析 | 局部契约或拒绝相应变换 | 保守保序或拒绝，不能猜测独立 |

未来非法示例的预期诊断应是依赖证据，而不是“包含 if”“状态不在同 stage”这类形状判断。
对一般 GM alias，不要求第一阶段完成全程序证明；明确接受调用者提供的事实及其适用范围。

## 15 TileLang Ascend 的参考实现

这里描述本地 TileLang 源码机制，不把它等同于任意控制流都已被证明正确。

1. `NormalizeControlFlowForSchedule` 提取复杂条件；读取 buffer 的 loop bounds 显式快照，
   保持 evaluate-on-entry 语义。调度树中的 while 有专门临时循环表示和恢复路径；VF 等
   不参与外层调度的叶子内部不随意改写。不是把 while 固定执行有限次数。
2. `MaterializeScheduleUnits` 把 serial/unrolled 循环变成分层单元，把 if 两分支变成带
   cond / !cond guard 的单元；不是无条件执行两边。
3. `AnnotateMultiBufferEligible` 判断每个候选存储是否每轮先写后读，考虑 guard 的蕴含与
   可能跳过的写，区分自动版本化资格和算法状态。
4. `AnalyzeDependencies` 分析 RAW/WAR/WAW、访问区域、alias、迭代距离及部分 subcore 信息。
   未版本化存储的跨轮冲突保留距离约束；合法 scratch 的复用距离可关联版本变量。
5. `ScheduleRecursive` 先调度子循环，再向父层提供摘要；Z3 用 stage×II+周期内偏移表示时间，
   加入资源、依赖、版本数和容量约束。手动 stage 也进入同一模型。
6. 循环求解未找到可行模型时，固定模式报错，自动模式可退回顺序安排。求解器搜索失败
   不等于对任意程序不可调度的数学证明；成本模型的最优也不等于硬件实测最优。
7. `InsertSync` 在 guard、参与者和版本语义上插入协议，再物化多缓冲和循环；不能仅复制
   TileLang 的语法，而跳过这些中间契约。

TLA 应先借鉴这一责任划分，不必立即迁移全部求解器、while 变换和复杂控制流机制。
手动与自动模式共用依赖模型，比新增一个 schedule 选择参数更重要。

## 16 分阶段交付与验收

| 阶段 | 范围 | MLIR 场景 | 验收重点 |
|---|---|---|---|
| R0 固化现有边界 | 平坦 task、内部控制流、普通外层循环、VF、持久状态 | S1–S4、S6 | 短循环、同 stage 多 task、初始化/出口、view alias、1 AIC:2 AIV |
| R1 固定调度检查 | 本地状态跨轮依赖与诊断；受限单核标量 carry；保留独立预取 | S5、U4 子集 | 接受合法递推，拒绝具体冲突；不靠增加状态槽绕过依赖 |
| R2 有限自动调度 | 固定分块/VF 边界，自动 stage/版本/pipe 顺序 | F1 | 容量和资源约束，失败保序退化，可读调度报告 |
| R3 受限控制流 | 整域一致 if、成组 guarded task、内层复合循环 | U1、U2、U5 | guard 版本、条件快照、空分支/零 trip、动态次数 |
| R4 复合流水 | 部分固定、一般 SSA carry、参与者候选、嵌套域 | U3、U4、F2 | 双层生命周期、状态初末值、事件作用域、完整 drain |
| R5 数据相关退出 | while、break/continue 等 | 单独设计 | 退出时在途工作、事件归还、counter 连续性；无未配对等待 |

每个阶段先建立 MLIR 回归，再接 Python 语法。R0/R1 不需要新增复杂前端参数。
R3/R4 先按受限场景交付，不宣称“支持任意复杂控制流”。

最低回归矩阵：T=0/1/<P/长循环；条件全真/全假/交替；内层次数 0/1/动态；两个 AIV
不同分片；同一 root 多个 view；初值与最终状态；容量不足；固定安排不可行；重复调用。
数值验证区分正常、极端输入；FA 同时比较完整 O、max/sum，不使用精度错误的性能数据。

四次 KV 迭代的 FA v3 还应评估 fill/drain 成本，不能只优化无限稳态 II。
最先优化的目标是正确保留算法递推，并让用户理解为什么某项工作不能提前，而不是让
用户继续拆 task、复制 kernel 或增加更多 stage 数字。

## 17 源码索引

Triton Ascend 仓内相对链接：

- [FA 版本与约束](README.md)
- [FA v3](fa_pipeline_kernel_v3.py)
- [Python scope 与 pipeline](../../../../language/tla/scope.py)
- [Python task 直接子节点检查](../../../../language/tla/code_generator.py)

以下是本机分析源码位置；分享文档时应替换成对应仓库固定版本链接：

- [NPU-IR 设计合同](/Users/ayame/workspace/AscendNPU-IR/docs/design/tla_manual_stages_implementation.md)
- [LowerTLAStages](/Users/ayame/workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/LowerTLAStages.cpp)
- [TileLang 控制流规范化](/Users/ayame/workspace/tilelang/src/ascend/transform/normalize_control_flow_for_schedule.cc)
- [TileLang 调度单元](/Users/ayame/workspace/tilelang/src/ascend/transform/materialize_schedule_units.cc)
- [TileLang 版本化资格](/Users/ayame/workspace/tilelang/src/ascend/transform/annotate_multi_buffer_eligible.cc)
- [TileLang 依赖分析](/Users/ayame/workspace/tilelang/src/ascend/transform/auto_schedule/dependency_analysis.cc)
- [TileLang 递归调度](/Users/ayame/workspace/tilelang/src/ascend/transform/auto_schedule.cc)
- [TileLang Z3 约束和退化路径](/Users/ayame/workspace/tilelang/tilelang/ascend/transform/z3_scheduler.py)
- [TileLang 条件同步](/Users/ayame/workspace/tilelang/src/ascend/transform/insert_sync.cc)
