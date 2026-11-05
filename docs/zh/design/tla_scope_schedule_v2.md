# TLA 固定 stage 接口与 NPU-IR 属性合同

本版与 AscendNPU-IR 的 LowerTLAStages / SyncTLAStages 接口对齐。用户手动指定 stage，后端物化流水并调用已有同步求解器；不包含 stage 或 latency 自动搜索。

## Python 接口

```python
with tla.scope.schedule():
    for t in tl.range(TILES):
        tile = tla.alloc((64,), tl.float32, space="ub")
        with tla.scope.vector().task("load", stage=0):
            tla.load_gm_to_ub(tla.view(X + t * 64, (64,)), tile)
        with tla.scope.vector().task("store", stage=1):
            tla.store_ub_to_gm(tile, tla.view(X + t * 64, (64,)))
```

- 保留 `schedule()` / `schedule("auto")` 写法；`auto` 只指流水和同步物化，不表示求解 stage。
- `.task(...)` 只属于 `scope.vector()` 和 `scope.cube()`；`scope.simd()` 不提供 task 接口。
- `.task(name, stage=..., order=None)` 必须显式给出 0–63 的编译期整数，拒绝 None、bool、负数和浮点数。名称必须非空、域内唯一。
- tick n 时，stage s 执行逻辑迭代 n−s；源码地址继续使用逻辑迭代变量，不手写 n−s。
- 同 stage 保留源码顺序，task 不是 barrier，也不要求只包含一个 pipe。
- 不新增外部 hint 文件、环境变量或前端版本数参数。

## 前端输出

| 属性 | 位置 | 类型与含义 |
|---|---|---|
| `hivm.pipeline` | 外层 scope | unit，标识需要物化的流水域 |
| `hivm.task_name` | task scope | 非空 string，域内唯一 |
| `hivm.pipeline_stage` | task scope | 必填 i32，范围 [0,64) |
| `hivm.task_order` | task scope | 可选非负 i32，后续 mainloop 排序偏好 |
| `hivm.tcore_type` | task scope | 沿用 CUBE / VECTOR 执行位置 |

```mlir
scope.scope : () -> () {
  scf.for %i = %c0 to %T step %c1 {
    // 循环内的静态本地 alloc。
    scope.scope : () -> () {
      // load
      scope.return
    } {hivm.pipeline_stage = 0 : i32, hivm.task_name = "load",
       hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      // store
      scope.return
    } {hivm.pipeline_stage = 1 : i32, hivm.task_name = "store",
       hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  }
  scope.return
} {hivm.pipeline}
```

不再输出 `hivm.schedule_domain`、`hivm.schedule_state` 或 `hivm.required_stage`。旧 IR 在 backend 入口明确报错，需重新生成。

task 的外层 scope 只标注 task/stage 和执行核；SIMD scope 嵌套其中，单独携带 VF/outline 属性：

```python
with tla.scope.vector().task("softmax", stage=2, order=1):
    with tla.scope.simd():
        # SIMD 运算；此 scope 不携带 task/stage。
        ...
```

这样 outlining 只提取内层 SIMD，外层 task 保留完整的阶段边界。`hivm.preload_fixed_depth`、preload_num、multi_buffer 由 NPU-IR lowering 生成，前端不生成这些内部调度属性，也不提前插入 set/wait。

## order：mainloop 内的排序偏好

`task(..., stage=2, order=1)` 输出 `hivm.pipeline_stage = 2 : i32` 和 `hivm.task_order = 1 : i32`。两者独立：stage 决定逻辑迭代偏移，order 表达后续 mainloop 内任务先后安排的偏好，不改变迭代归属。

- order 接受编译期整数 0..2³¹−1；拒绝 bool、浮点数、负数及越界值。
- 省略或传 None 时不输出属性，不自动填 0；0 是有效的显式值。
- 预期较小的 order 倾向优先安排；允许相同值，也不要求连续。不根据这些数值在前端移动任务。
- 本次仅实现 Python → ttadapter 属性透传。当前 NPU-IR 尚未消费该属性进行 mainloop 重排，因此设置 order 暂不改变执行顺序，也不保证后续各次 lowering 保留该属性。
- 后续排序实现必须遵守数据依赖、同步与 buffer 生命周期；order 不能授权越过这些约束。未指定值、相同值和无法满足的偏好如何处理，需在后端排序实现时明确。

## 责任与编译路径

前端检查 task 名称、stage 值和嵌套使用；NPU-IR 检查完整 IR 形态、核类型、别名、跨迭代依赖及规划结果。

当前后端仅支持受限形态：函数直接子 pipeline scope、一个从 0 开始步长 1 的 index 循环、无 iter_args、stage 按源码非递减；本地静态 alloc 在循环顶层，任务中无动态控制流、手写同步或 opaque call。GM 必须来自函数参数及其 view。

设备编译只为 RegBase 的 ttadapter → bishengir-compile TLA 路径放行。A2/A3、直接 TTIR 设备编译及不支持该入口的编译器路径继续报错。需要部署含当前固定 stage 实现的 NPU-IR；Python 路径检查不是编译器版本能力探测，不能将它视为对旧二进制的兼容承诺。

TLA affinity 不运行 AutoBlockify，因此 backend 同时关闭其 metadata，避免 launcher 将完整 grid 截断到物理核数。普通 kernel 保持原策略。

## 测试输入与限制

[task_stage.py](../../../third_party/ascend/unittest/tla_ut/scope/task_stage.py) 更新了固定 stage、非法值、旧属性拒绝和编译入口检查。

[attention_stages.ttadapter.mlir](../../../third_party/ascend/unittest/tla_ut/scope/attention_stages.ttadapter.mlir) 是预期 IR fixture，本轮手工更新属性，并非重新执行编译生成的结果。load_v 明确为 stage 0。attention 仍有嵌套 SIMD scope 内的 row 循环及 1:2 分工，因此仅用于前端属性测试，不能声称当前受限后端已端到端支持它。

本轮未构建、未运行测试或 NPU 精度/性能验证。
