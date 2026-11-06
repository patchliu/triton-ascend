# TLA pipeline：显式固定 stage 接口

当前只支持所有 stage 显式固定；不支持 order、缺省 stage 或匿名 task。

```python
for k in tla.pipeline(TILES):
    tile = tla.alloc((64,), tl.float32, space="ub")
    with tla.scope.vector().task("load", stage=0):
        tla.load_gm_to_ub(tla.view(X + k * 64, (64,)), tile)
    with tla.scope.vector().task("store", stage=1):
        tla.store_ub_to_gm(tile, tla.view(Y + k * 64, (64,)))
```

`pipeline(stop)` 从 0 开始，步长 1。每个 task 是循环直接子 scope，名称非空且唯一；
stage 为编译期整数 0..63，按源码非递减。同 stage 多 task 保留源码顺序。
stage s 在 tick n 执行逻辑迭代 n-s，有效范围为 0<=n-s<stop。
SIMD 代码写在 vector task 内部的 scope.simd() 中，外层 task 不作为 VF outline。
普通 pipeline 外的 vector/cube/simd scope 保持原语义。

调用者必须保证不同逻辑迭代没有算法内存依赖，包括通过 GM 参数别名产生的依赖。
循环内 scratch 每轮先写后读；物理单槽不提供跨轮持久状态。
编译器负责 scratch 版本、地址轮换和异步同步，不证明任意 GM 迭代独立性。
例如 X[k+1]=f(X[k]) 的内存递推不能通过不同 stage 直接交错。

pipeline 不支持 loop-carried SSA 值、嵌套 pipeline、break/continue/return、for-else，
也不能嵌套在其他控制流/核 scope 内。循环顶层允许静态局部分配和安全的纯地址计算，
内存效果放入 task。GM view 保持参数 provenance；task 内 VF 可以包含归约循环。

前端仅发出 hivm.pipeline、hivm.task_name、hivm.pipeline_stage 和核/VF 属性。
NPU-IR 以 P=max(stage)+1、preload_num=P-1-stage 适配已有展开器，
各 buffer 的版本由访问 stage 跨度确定，不由 Python 直接指定。
`enable_preload=False` 不取消显式 tla.pipeline 的物化。

未来自动调度可在固定计划之前增加，但当前不保留未实现的参数或自动补 stage 路径。
配套后端及 pass 边界见 [后端实现](tla_stage_to_npuir_pipeline.md)。
