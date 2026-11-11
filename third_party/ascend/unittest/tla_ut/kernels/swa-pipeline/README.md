# Sliding-window GQA pipeline

主规格 `(B,Hq,Hkv,D,Sq,Skv) = (1,32,16,256,8192,8192)`，连续 BNSD BF16，窗口 1024 含当前位置。第 `q` 行可见的 key 为 `[max(0,q-1023),q]`。用 `torch_npu.npu_fused_infer_attention_score` 的 band 模式做 Ascend C 对照，`sparse_mode=4`、`pre_tokens=1023`、`next_tokens=0`、2048² 压缩 mask。设备精度先用 `window=1` 检查是否只取对角线。 [CANN FIA V5](https://github.com/hicann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md) · [vLLM-Ascend 滑窗调用](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/attention_v1.py)

每个工作项处理一个 Q head 的 64 或 128 行，KV tile 固定 128 行。只遍历该 Q tile 覆盖的窗口与 KV tiles 的交集，在 Vector softmax 中做逐行左右边界。`load_kv` 在 stage 0，其余五项在同一 compute stage，在线 FP32 状态留在流水外。stage 0 对照与流水版共用源码。窗口和分块边界直接从逻辑坐标计算。

首版配置为 Q64 和逐行 mask。`TLA_SWA_Q_TILE=128` 让主规格每个 Q head 的 KV tile 访问由 1080 次减到 540 次。加宽 Q tile 会增加 L0C 的 QK/PV 缓冲，以及各 Vector 子核的 FP32 累加缓冲。`TLA_SWA_SKIP_INTERIOR_MASK=1` 允许跳过两侧边界检查。当 Q tile 从 `q_start` 开始时，每一行共同可见的 key 区间是 `[max(0,q_start+QT-WINDOW),q_start]`。KV tile 完全位于该区间才跳过 mask，主规格有 420/540 个 128 行工作项中的 KV tile 满足这个条件。两个选项可以独立启用。

历史 causal FA 把 QK 变为 `K @ Q.T`，用 `COLUMN_SPLIT` 将 128 行 Q 均分给两个 Vector 子核，每个 SIMD lane 处理一个 query。这个布局可以复用 Q/K 搬运和调度。滑窗需要在 lane 上比较左右边界，并保留有限的 FP32 初始最大值，避免首批 KV 对部分 query 全掩码时产生 NaN。移植也需要转置概率打包和 PV 输入，并重排 D256 的输出累加。当前候选沿用逐行 softmax，先单独测量 Q tile 加宽和内部 tile 免 mask 的收益。希望编译器提供不同 Q tile 的 UB/L0C 峰值与流水缓冲版本，支持报告 task 内分支的同步和变长任务的分核利用率。

精度比较 normal 双 seed、uniform 和 peaked，补测单点窗口及非对齐窗口 193。FP32 参考按 256 行分块生成数学 mask。通过精度后，同卡轮换采集三轮各 40 个样本，按 Ascend C 的设备耗时中位数除以 pipeline 耗时计算比例。`results.json` 保存 tiling 和 mask 选项。

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/swa-pipeline -k test_ir
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/swa-pipeline -k test_run
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q third_party/ascend/unittest/tla_ut/kernels/swa-pipeline -k test_perf
TLA_SWA_Q_TILE=128 TLA_SWA_SKIP_INTERIOR_MASK=1 TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/swa-pipeline -k test_run
```

窗口边界经 CPU 检查。原始 64 行逐行 mask 版本的两种 stage 配置通过 1109 离线编译。Q128 加完整 tile 免 mask 通过主规格、单 tile 和非对齐窗口精度，设备中位数 2161.2375 / 1289.0475 µs（59.64%）。

当前默认采用 Q128/KV128 列式 softmax、重叠调度、PV task 搬 V。这样保留 K 的提前搬运，同时减少 V 的流水缓冲版本。六项精度通过，正式中位数 1162.3855 / 1268.8795 µs（109.16%）。KV256、同样延后搬 V 的结果为 89.50%。可用 `TLA_SWA_KV_TILE`、`TLA_SWA_LATE_VALUE` 和 `TLA_SWA_SCHEDULE` 复现消融。
