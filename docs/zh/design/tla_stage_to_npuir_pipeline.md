# TLA 固定阶段到 NPU-IR 的物化

当前用户合同见 [pipeline 接口](tla_scope_schedule_v2.md)。所有 stage 必须显式固定，
不支持 order；调用者保证逻辑迭代之间没有算法内存依赖。后端管理异步执行和存储复用，
不额外引入通用 GM 别名证明器或调度搜索。

LowerTLAStages 负责必要的结构/effect 准入、view root 归并、stage span 版本计划和
preload 转换。域内 helper 与 loop 按源码顺序内联到函数，不能把 loop bound 下沉。
每个 task 是调度 shell，内层 simd 才 outline；无需兼容“整个 task 就是 VF”的旧形态。

```text
LowerTLAStages → 提前 outline VF → MarkTCB → 选中的 MIX 函数执行 CCGSS
→ SplitMix → 原有 affinity 归一化 → PlanMemory → CreatePreload
→ 核内 GSS → EnableMultiBuffer → LowerMultiBufferCounter → LLVM
```

P=max(stage)+1，preload_num=P-1-stage；root 版本数 N=last_stage-first_stage+1。
不强制 N=P，不强制事件 ID 数=N。拆核后保留共同时间坐标。
CreatePreload 使用既有循环展开、地址相位与 guard；同 stage 多 scope 通过已有 bypass
路径保留。通用入口过滤和少量正规形检查保留，不修改 PlanMemory 或同步求解算法。

FA8 主用例使用独立 K/V、QK/PV 存储，仅 score 跨 stage 需要两个版本。
显式数组预算为 L1 290 KiB、L0C 68 KiB、每 AIV UB 85.5625 KiB；
不含后端临时量与额外对齐，不代表已测性能。

详细后端合同以子模块中的
[实现说明](../../../third_party/ascend/AscendNPU-IR/docs/design/tla_manual_stages_implementation.md)
为准。此前评审文档保留为锁定旧提交的分析记录；本次具体变更见
[修改记录](tla_pipeline_1107_changes.md)。本次未构建或上卡。
