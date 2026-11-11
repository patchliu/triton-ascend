# FlashMLA V4.1 sparse decode

目标规格 `(B,Hq,Hkv,Dq/Dv,Sq,TopK) = (64,64,1,512/512,1,256+2048)`。Q 和 O 是 BSND BF16，LSE 为 `[B,Hq,1]` FP32。两份量化 KV 各使用 `[pages,page_size,1,528]` 布局，每条 token 记录含 512 个 E4M3FN 字节和 16 个 E8M0 scale 字节。两份索引分别引用各自缓存内的物理 token，`-1` 屏蔽。两个 TopK 集合合并做一次 softmax，重复索引保留权重。可选 FP32 attention sink 只缩放输出。全屏蔽时返回零输出和 `+inf` LSE。

首版选双 FP8 缓存和有限量化值。测试的页大小为 64，每请求两份缓存分别分配 256 和 4096 个 token，并在后者中选择 2048 个。这个选择区分稀疏选择数与实际缓存占用。性能使用同一份缓存、索引、Q、sink 和输出合同，对照 DeepSeek 的 `flash_mla_with_kvcache` Ascend C 扩展。上游当前声明 CANN ≥ 9.2、PyTorch ≥ 2.10，运行前需核对实际环境及 `_build_info.BACKEND`。 [接口与构建说明](https://github.com/deepseek-ai/FlashMLA) · [原生精度参考](https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py)

## 调度与优化

Ascend C 报告的重点是按 token 连续存储数据和 scale、用 `gather2` 减少 MTE2 请求、用带 padding 的 NZ 排布控制 UB bank 冲突，以及通过 64-head tile 复用 KV。它也报告了延迟 softmax rescale 和 L1 分 bank 的做法。这里沿用 token 布局、64-head 复用和概率 NZ 的 padding。 [DeepSeek Ascend 优化报告](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20260930-ascend-prefill-deep-dive.md)

第一次实现中，一个 Triton kernel 每次 gather 双缓存并反量化到 BF16 workspace。随后 TLA 每轮预取 128 个选中 token，复用同一 L1 KV 完成 QK 和 PV。六个任务表达 load、QK、softmax、概率发布、PV 和累加。通过参数切换 stage 0 对照；TLA 源码共用。FP32 在线状态在循环外，行级 LSE 和 sink 在输出时计算。

这个首版在主规格上有 144 MiB BF16 KV workspace 和 576 KiB 有效索引 workspace，写入和读取都在同一次端到端测试内。下一步优化是在流水 stage 0 将稀疏 gather、FP8 反量化和 NZ 排布一起搬到 UB→L1 路径，消除整个中间 KV 的 GM 往返。再比较 gather2、64/128 token tile、延迟 rescale 和 L1 两个 bank 的读写分布。原生支持的额外缓存 FP4 需另验相同数据合同。

编译器诉求为带 token 地址的两段 DMA 和内存效果标注，便于流水推断 gather 依赖。对 FP8/E8M0 反量化和布局转换，希望能在同一 stage 中表达动态 scale，并用资源报告核对 KV 版本、UB/L1 分配及 MTE2 指令数。

## 验证

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/flashmla-pipeline -k test_ir
TLA_TEST_STAGE=compile pytest -q third_party/ascend/unittest/tla_ut/kernels/flashmla-pipeline -k test_compile_gather
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/flashmla-pipeline -k 'test_run and not test_run_native'
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/flashmla-pipeline -k test_run_native
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q third_party/ascend/unittest/tla_ut/kernels/flashmla-pipeline -k test_perf
```

精度用 normal 双 seed、peaked 和带全屏蔽请求的无效前缀，每组重复三次，输出与 LSE 都对比独立 FP32 参考。通过同输入精度检查后，同卡轮换 3×40 个样本，统计两种 TLA 配置和原生 Ascend C 的全部设备任务时长。结果记录实际 wheel、编译器、原生扩展哈希与占用，性能比例按 Ascend C 设备中位耗时除以 pipeline 耗时计算。

FP8 gather 初版的向量指针选择触发 `TritonToUnstructure` 的 rank 断言。拆成两个带 mask 的读取后，FP8 gather 和两种 TLA attention stage 均生成了 NPU 二进制。首版和 16-token 分缓存 gather + 提前 QK 候选分别在 blue2 通过 4 项设备精度，O、LSE 和全屏蔽合同一致。

官方源码在隔离的 CANN 9.2 工具链构建。x86 适配将 SDK 路径改为 x86_64，设备源码使用经校验的 GCC 11 头文件，主机 C++ 源码使用 GCC 13。上游固定 32 AIC 的调度网格改为本机实际的 28 AIC，并保留运行时核数断言。生成的原生扩展在系统 CANN 9.1、Torch 2.10 上通过四项精度。原生计时前延后可选的设备信息采集，避免驱动处理过的显存查询状态被原生 launch 读取。

| 实现 | 设备任务总时长中位数 (µs) | 同步 wall time 中位数 (µs) | 相对 Ascend C |
|---|---:|---:|---:|
| TLA pipeline | 2152.389 | 2193.948 | 4.26% |
| Ascend C，28 AIC 适配 | 91.590 | 105.974 | 100.00% |
| TLA stage 0 调度诊断 | 2181.582 | 2223.977 | — |

三方共九轮的卡 1 占用检查全部通过。TLA 的两项任务中，`_gather_fp8` 约 1946.4 µs，占总耗时约 90.4%，`_attention` 约 206.0 µs。证据在 blue2 的 `~/workspace/npuir/temp/20261110-tla-14-validation/flashmla-x86-replay.6NTIDy/measure-grid28.YJYl3p/card1-perf.P7PPf5/results/v41-decode-eg570a3p/results.json`。

用 `TLA_FLASHMLA_FUSED_GATHER=0 TLA_FLASHMLA_GATHER_TILE=16 TLA_FLASHMLA_GROUPED_GATHER=1 TLA_FLASHMLA_EARLY_QK=1` 复现原始计时。一个 gather program 只访问一个缓存，保留输入 token 顺序。

### 融合 gather 候选

两路 TopK 的交界对齐到 128-token tile 时，调用默认启用融合 gather。合法但不对齐的切分自动回到分步实现，也可以设 `TLA_FLASHMLA_FUSED_GATHER=0` 强制分步。融合路径让 stage 0 的两个 AIV 从两路原始缓存搬运各自的 528 B token 记录。每组记录在 UB 内按 32 个元素共享一个 E8M0 scale 反量化，转为带行 padding 的 NZ，再发布到 Cube 使用的同一 L1 tile。softmax 复用本 tile 的 UB 索引生成有效性掩码。该配置移除了 BF16 和有效标记的 GM workspace。

融合版通过三套 Ascend950 离线二进制验证。卡 2 上四项精度通过，主规格输出相对 RMS 为 0.1841%，LSE 最大绝对误差为 9.54e-7。占用检查通过的三轮同栈计时给出 TLA pipeline **1279.887 µs** 和 Ascend C **92.024 µs**，相对性能 **7.19%**。TLA 端到端耗时比前版低 40.5%，单一设备任务内包含 gather、反量化和 attention。stage 0 调度诊断为 1252.253 µs。结果位于 blue2 的 `flashmla-fused.LBU4F5/card2-perf.ujzyp0/results/v41-decode-ri75165c/results.json`。

当前每个 token 仍需一次标量索引读取和一次 DMA。原生 gather2 调用 `asc_copy_gm2ub_align`，以动态 GM 起始偏移和两条记录的地址差一次发出两个 burst。源记录为 528 B，UB 行步长为 544 B。离线对照用静态 GM 行步长 1056 B 生成一个 `hivm.hir.load` 并通过设备二进制编译。改为运行时地址差时，`tla.view` 在校验 `strides` 时要求正的编译期整数。`EvictFirst` 可以到达 TTAdapter，在该探针中与默认策略生成相同的设备二进制。证据在 blue2 的 `flashmla-fused.LBU4F5/gather2-evidence/`。

编译器需要 GM 视图的运行时行步长及对应的双 burst lowering，并能表达无效索引、重复索引和 `NOTALLOC_CLEAN` 一类 L2 分配提示。还需让 JIT helper 能接收 TLA buffer 和 SIMD register，现有子函数生成器会落入普通 Triton builder，无法构造 TLA buffer 类型。

### 32-token UB 聚合筛选

单独试了每次聚合 16 或 32 个 token 后再发布到 L1，保持同一套 pipeline 和 task。8、16、32 三档的 stage 0 和 stage 1 共 6 组均完成 Ascend950 离线编译。32 档每个 tile 的 UB→L1 发布从 8 次降至 2 次，单 AIV 临时 UB 从 13,568 B 增至 51,200 B。卡 2 的 32 档配对计时在 pipeline 第 1 轮被占用校验拒绝，正式表沿用 7.19%。该候选仅在独立实验目录，交付默认实现使用已验收的融合路径。
