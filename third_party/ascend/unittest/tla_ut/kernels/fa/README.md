# FlashAttention

八种规格的 BF16 FlashAttention，包含六种优化模板、shape-only 分发和显式流水同步。
当前实现尚未编译、上卡或测量性能。

## 实现与接口

- `kernel.py`：可读的源模板，具名 `inline_*` 表达 QK、softmax、PV、output 等步骤；由解析器读取，不直接导入。
- `expand_kernel.py`：展开具名步骤与六相位槽位轮转；不重排算术。
- `fa_kernel.py`：公共 `prepare_call` / `forward` / `flash_attention`，按 shape 选择模板；源码生成只在准备阶段执行。
- `reference.py`、`testing.py`：共享全输出 FP32 参考、原生基线、精度和性能入口。
- `test_fa.py`：按 DSA 目录约定使用 pytest 和顶部配置变量。

使用当前仓的 `tla.cube.mmad`（包括 transpose-A/B、real extents）、`tla.simd.vexpdif` 和
`tla.simd.vmuls`。

紧凑 causal 尾块通过现有接口构造连续前缀：

```text
连续 p_ub → reshape 为一维 → subview 截取零偏移前缀 → reshape 为紧凑目标形状
```

这保留 allocation 和元素顺序，并改变目标行距。直接在二维 buffer 上取较小 subview
会保留原 pitch，不等价。两个 compact causal 模板均按上述方式处理。

## 规格

所有输入为连续 BF16 BNSD，B=4、H=8；输出为 BF16 O 和 FP32 max/sum `[B,H,S,8]`，
八份统计量全部有定义。默认 BM128/BN256、28 AIC/56 AIV，不支持 dropout、PSE 或任意 tail shape。

| Case | S | D | causal | 模板 |
|---|---:|---:|---|---|
| FA1 | 8192 | 128 | true | long_d128 |
| FA2 | 8192 | 128 | false | long_d128 |
| FA3 | 8192 | 64 | true | long_d64 |
| FA4 | 8192 | 64 | false | long_d64 |
| FA5 | 1024 | 128 | true | short_causal_d128 |
| FA6 | 1024 | 128 | false | short_dense_d128 |
| FA7 | 1024 | 64 | true | short_causal_d64 |
| FA8 | 1024 | 64 | false | short_dense_d64 |

优化重点：K @ Q.T 产生 DN score，每个 SIMD lane 对应一个 query；FP32 ExpSub 与 scalar multiply；
P.T @ V 使用 MMAD transpose-A。两槽 score、三槽 P/alpha 的最小公倍数为六，因此展开六相位。
D64 使用双 PV 槽；长序列使用相应 K/V 槽策略；短 causal 模板采用紧凑尾块和 shape-only 调度。
六相位展开将物理槽位与 ready/free 事件对应起来，保留各阶段的执行顺序。
可读 task/stage 示例另见 `../fa-pipeline`，不把该参考 kernel 的 flag 改写成未经证明的 stage 数字。

## 运行

需要匹配的 Ascend950、CANN、torch_npu 和包含当前 TLA 公共接口的 wheel。
从仓库根目录运行（沿用父目录设备 fixture 和 `ASCEND_RT_VISIBLE_DEVICES`）：

```bash
TLA_TEST_STAGE=run pytest -q -s third_party/ascend/unittest/tla_ut/kernels/fa -k test_run
# 先把 test_fa.py 的 RUN_PERF 改为 True
TLA_TEST_STAGE=run pytest -q -s third_party/ascend/unittest/tla_ut/kernels/fa -k test_perf
```

顶部 `CASES` 选择规格。精度默认覆盖两个 normal seed、Q=0 的 uniform attention、Q/K 放大四倍的
peaked 输入，每种重复三次。每次对 TLA 输出预填 NaN，检查完整 O 和全部八份 max/sum，
关闭 HF32，按张量归一化最大误差 O≤0.005、max/sum≤0.0001 验收，并记录 relative RMS。
这些是测试验收门限，不是任意输入的数学误差上界。

性能入口独立执行精度预检查，随后轮换实现顺序，默认 3×40 次采样；使用共享
`kernels/utils.py` 保存环境、源码信息、每轮 profiler trace、原始样本、device median 与相对加速比。
`wall_samples_us` 是无 profiler 的已准备 callable 完整调用时间；TLA 输出已预分配，不能将其称为
包含公共 `forward` 分配的端到端耗时。AscendC 的输出分配由原生 API 管理。
`PERF_COUNTERS=True` 可额外导出流水线计数器。采样不使用 xdist，性能默认关闭。

当前没有性能测量数据，待实际验证后更新。
