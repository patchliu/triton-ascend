# TLA task/stage 调度接口 v2

状态：stage-only 前端原型。实现任务属性到 ttadapter 的透传；NPU-IR Solver、流水物化与同步生成尚未实现。
不引入 latency、group、外部 hints 配置或性能结论。v1 文档保留作为讨论记录，本版接口以本文为准。

## 1. 最小接口

```python
with tla.scope.schedule("auto"):
    for t in tl.range(TILES):
        with tla.scope.vector().task("gather", stage=0):
            ...
        with tla.scope.cube().task("qk"):
            ...
        with tla.scope.simd().task("softmax", stage=1):
            ...
```

- `cube/vector/simd`：执行位置和形式，保持现有语义。
- `.task(name, stage=None)`：命名一个逻辑任务组；`stage` 是可选硬约束，省略表示待求解。
- `scope.schedule("auto")`：圈定自动调度区域。源码仍按逻辑依赖顺序书写；用户不写前取、排空和 `% versions`。
- 普通 scope 不要求任务标注。未来 NPU-IR 将按操作/VF 边界推导未命名任务。
- 任务组可以包含多个 pipe，例如 scalar 索引加载和 DMA；Solver 建图时拆分子任务，继承相同 stage 约束并保留依赖。
  `.task()` 不隐含单 pipe、不可拆分或 barrier。SIMD VF 内部保持原指令顺序。

task 名称是当前域内唯一的静态位置标识，循环中的动态执行仍是同一个任务。
`stage` 只接受非负编译期 i32 整数；布尔值、浮点数、运行时值均拒绝。
首期不支持嵌套调度域或嵌套任务标注；任务内部可以嵌套普通 SIMD scope。

没有 `group=(1,2)`：核配对来自目标与 launch，数据分工来自 ROW_SPLIT、buffer view 和 sub_block_id。
仅知道核的数量不足以生成同步；无法证明参与者或别名关系时必须保守处理或报错。

## 2. stage 的准确语义

在流水 tick n，stage s 处理逻辑迭代 n−s。原始代码中的 `t` 始终指同一个逻辑 tile，不手动写 `t−s`。
同 stage 的任务仍有先后依赖，不代表同时执行。固定一个任务的 stage 不会固定其他任务。
硬约束无解应报告冲突；前端不凭 stage 数字判断依赖已满足。

未指定 stage 的任务继续自由求解；不能默认填 0。
即使只提供 stage，后续 Solver 仍需推导依赖、资源和操作成本，只是成本不作为本版用户接口。
buffer 版本数与 stage 数独立。单槽复用需要等待最后消费者，多槽只能消除存储假依赖，不能消除真正的数值递推。

## 3. 测试用例：单 KV 块 attention

完整代码：[task_stage.py](../../../third_party/ascend/unittest/tla_ut/scope/task_stage.py)。
预期完整 ttadapter：[attention_stages.ttadapter.mlir](../../../third_party/ascend/unittest/tla_ut/scope/attention_stages.ttadapter.mlir)。

每次迭代处理 Q[16,64]，共享 K[128,64]、V[128,64]，输出 O[16,64]：

```text
scores = Q @ K.T / 8
P = softmax(scores, axis=-1)
O = bf16(P) @ V，累加及输出 FP32
```

这是固定长度的小型 attention，而非已完成长序列 online-softmax 流水的 FlashAttention 实现。
测试覆盖了 FA 所需的 QK→softmax→PV 路径，并保留显式 NZ 打包；不以占位函数隐藏数据搬运。
两个 AIV 各处理 8 行，P 写入 L1 的不同分片。

| task | scope | required_stage | 内容 |
| --- | --- | ---: | --- |
| load_qk | cube | 0 | Q、K 的 GM→L1 |
| load_v | cube | 未指定 | V 的 GM→L1，验证缺省 stage 不变成 0 |
| qk | cube | 1 | 转置 B 的 MMAD |
| score | cube | 1 | Fixpipe ROW_SPLIT 到两个 AIV |
| softmax | simd | 2 | FP32 softmax、BF16 P、NZ 排列 |
| prob | vector | 2 | 两个 AIV 将各自的 P 分片写入 L1 |
| pv | cube | 3 | P×V |
| output | cube | 3 | Fixpipe ROW_SPLIT |
| store | vector | 4 | 各 AIV 写回自己的输出行 |

这些阶段用于验证接口，不声称是性能最优解。K/V 重用与跨迭代优化由后续调度分析决定。
本轮只测试前端与 IR，未执行 NPU 数值/性能测试。

## 4. ttadapter 合同

前端输出保留 `scope.scope`，使用以下四个属性。它们是本原型新定义的合同，当前 NPU-IR 尚无消费它们的 Solver。

| 属性 | 位置 | 类型与含义 |
| --- | --- | --- |
| `hivm.schedule_domain` | 外层 scope | string `"auto"` |
| `hivm.schedule_state` | 外层 scope | string `"logical"`，未求解 |
| `hivm.task_name` | 任务 scope | 非空 string |
| `hivm.required_stage` | 任务 scope | 可选 i32 硬约束 |

节选如下，省略算子参数；完整可解析的输出见链接中的 `.mlir` 文件。

```mlir
scope.scope : () -> () {
  scf.for %t = %c0 to %c3 step %c1 {
    // 原始逻辑 buffer 分配与数据访问。
    scope.scope : () -> () {
      // QK MMAD
      scope.return
    } {hivm.required_stage = 1 : i32, hivm.task_name = "qk",
       hivm.tcore_type = #hivm.tcore_type<CUBE>}

    scope.scope : () -> () {
      // V 的 GM→L1
      scope.return
    } {hivm.task_name = "load_v",
       hivm.tcore_type = #hivm.tcore_type<CUBE>}
  }
  scope.return
} {hivm.schedule_domain = "auto", hivm.schedule_state = "logical"}
```

此片段仅示意属性，不代表示例完整任务顺序。真实输出中 load_v 位于 qk 之前。
SIMD scope 仍有原来的 `outline/no_inline/vector_mode/hivm.vector_function` 标记。
没有生成 `preload_num`、物理版本索引或 set/wait，也没有把任务 scope 当作同步屏障。

当前自动调度域不是设备可执行合同：npubin 入口检查未解决的 domain/task 标记并报错。
不能让旧版 NPU-IR 静默忽略属性后运行。未来接入 Solver 时再替换此 guard，为支持该合同的编译器开放入口。

## 5. NPU-IR 求解与 preload 复用

建议处理顺序：

```text
VerifyLogicalSchedule
→ NormalizeTaskGroups / InferEffects
→ SolveStagesOrderAndVersions
→ VerifySchedulePlan
→ LowerScheduleToPreload
→ 内存规划、同步、CreatePreload、PeelPreload
→ VerifyPhysicalSchedule
```

Solver 对同一 AIC/AIV 协作单元联合建图，需在 SplitMix 丢失跨核上下文前求解，或保留可重建的完整任务图。
读写分析必须覆盖 buffer 根与别名、分片、RAW/WAR/WAW、跨迭代递推、真实最后使用点和两个 AIV 的参与者。
VF outlining 后通过调用点摘要保留这些信息。

复用现有 preload 机制来物化已求解的循环偏移与首尾 guard。映射为：

```text
P = max(solved_stage) + 1
preload_num = P − 1 − solved_stage
max_preload_num = P
```

例如最终 stage 0..4 对应 preload_num 4..0；本例 load_v 未求解，前端不能现在就输出该结果。
每个存储根的物理版本数单独决定；槽号基于 `(logical_iv − begin) / step`，不能直接用 `iv % versions`。

适配现有实现时必须解决：

1. 当前 CreatePreload 只收集循环直接子 scope；域 wrapper 需由规范化 pass 消费。
2. 同 preload 编号多 scope 受现有 gate 控制；新计划应有显式 verified-plan 入口，不能依赖函数名 registry。
3. 原有 CVPipelining/Reposition/Merge/多缓冲启发式不得再次改写 Solver-owned 计划。
4. 同步需同时覆盖 ready 和 free，按实际版本数与参与者生成；已有手写 flag 不能仅靠编号避让判断兼容。
5. PlanMemory 最终容量不足时回到逻辑 IR 重求解，不直接削减槽数后沿用旧同步。

不在本轮实现 Solver 和上述后端适配。已有 preload 语义的详细源码分析见 [v1](tla_scope_schedule.md)。

## 6. 第一版实现边界

已实现路径：

- scope 的 `.task(name, stage=...)` 返回不可变编译期描述；`schedule("auto")` 返回独立域描述，不带 Vector 核属性。
- with handler 解析完整链式表达式，沿用现有 dispatch registry，无需修改社区 Triton patch 或 C++ builder。
- 校验名称、stage 类型/范围、域外 task、重复静态名称、嵌套域/任务与 `as` 绑定。
- 既有不带 task 的 scope 写法保持原属性；新属性透传至 ttadapter。
- 二进制编译入口拒绝未求解的 domain/task，明确提示需要 NPU-IR 调度与同步 lowering。

本版还没有验证 stage 约束是否可行，也没有自动插入任何同步；原始手写同步与 auto 域的兼容性留给 NPU-IR verifier。
实际接入后必须拒绝未封装的手写同步，或采用具有完整输入/输出完成合同的 manual 任务。
本版不开放 manual 模式，避免声称已经支持该组合。

## 7. 验证与复现

在安装当前 Python 前端、具有兼容原生 TLA 扩展的环境中执行：

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/scope/task_stage.py
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/scope \
    third_party/ascend/unittest/tla_ut/cube/mmad.py
```

IR 测试检查 task 名称/顺序/阶段、未指定 stage、Cube/Vector/SIMD 属性及 QK/PV/DMA 结构。
预期 IR 去除源码 location，保留完整函数 ABI、buffer 类型和算术；比较任务合同而非脆弱的 SSA 名称。
负例与二进制阻断测试确保不会将“产生标注”误认为“已经完成调度”。

2026-11-04 在 blue 的 `tla-dev` 环境中以隔离 Python 源码覆盖方式验证：scope 与 MMAD 回归 **31 passed, 38 skipped**（`TLA_TEST_STAGE=ir`）。保存的预期 IR 已通过 MLIR 解析与 verifier；未进行 NPU 精度、性能或 Solver 调度验证。
