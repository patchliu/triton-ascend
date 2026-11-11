# FlexAttention forward pipeline

目标为连续 BNSD BF16 `(1,16,8,128,52000,52000)`。参考 Mojo `dd8ca8c5` 的 `sparse_2000_22000` 多模态用例。两个 document 各含 2000 文本、22000 图像、2000 文本 token。文本采用同 document 的因果局部距离 1024 和 4 个起始 token。每段图像另有段内双向可见性。前向接口接收预建的 128×128 packed BlockMask，支持 batch/head 共享的任意 full/partial 布局。当前输出为 BF16 O，score 使用标准缩放点积。通用 `score_mod` 和反向尚未接入。 [PyTorch BlockMask 语义](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html)

参考的 Mojo A5 Triton 实现会区分 full 与 partial、裁剪空块并用持久任务调度。此实现保留压缩 partial 数据，按有效块数分桶。每个桶将循环次数作为普通标量传入，Cube 和 Vector 各在所属 task 读取自己的块信息，绕开跨核 GM 标量决定流水边界的问题。每个逻辑工作项计算一个 head 的 64 或 128 行，使用六个任务。在线 softmax 状态由各工作项独立维护。

`TLA_FLEX_SKIP_FULL_MASK=1` 在同一个 `softmax` task 内给 full block 省去 UB mask 填写、读取和逐行 predicate。partial block 沿用逐行 mask，空槽继续产生零贡献。当前默认启用，性能报告记录此开关。CPU 枚举 52K 的有效 tile 得到 58,923 个 full 和 1,882 个 partial，full 占 96.9%。52000 尾部填到 52096。每次调用的 Q/K/V padding、所有桶和输出裁剪后拷贝均纳入性能。可复用的 mask 与分桶表在计时前生成。

`TLA_FLEX_MERGE_BUCKETS=1` 按宽度从小到大尝试合并相邻桶，额外槽总数最多占原计划的 0.5%。52K 的 19 个 launch 合为 10、174 和 182 槽三桶，每 head 增加 137 个填充 tile，占 60,805 个有效 tile 的 0.23%。只要有填充槽，K/V 尾部就预留一个合法的全零 tile。`-2` 槽使用全掩码，更新时 `alpha=1`、概率和 PV 贡献均为零。`TLA_FLEX_Q_TILE=128` 让两个 Vector 子核各处理 64 行，对同一个 128 行 BlockMask query block 只加载一次 K/V，当前默认 128 行。三个开关可独立启用。

列式 softmax 用 `K @ Q.T`，Fixpipe 按 query 列分发，每个 SIMD lane 对应一个 query。只在 partial 块加载 mask，晚到 PV task 再加载 V，Q128 与 K128 共用原六个任务。默认 Q128 采用列式调度，设置 `TLA_FLEX_COLUMN_SOFTMAX=0` 可复现旧版。

`TLA_FLEX_PACKED_PREDICATE=1` 将每个 AIV 对一个 key 的 64 个查询位放入 8 字节，再填充到 PLD 所需的 32 字节对齐。每个 partial tile 从 64 KiB 降至 8 KiB。开关 `TLA_FLEX_DEEP=1` 将输出累加从 PV 所在的 stage 2 移到 stage 3。两项在 Q128 默认启用，通过 7 项设备精度。正式同卡 3×40 为 TLA 32.739 ms、Mojo 33.728 ms、Ascend C FIA 组合 31.862 ms，分别为两个对照的 103.02% 和 97.32%。

桶内全块前缀候选按同一真实 KV 索引读取，仅在公共前缀省去 mask slot 读取。候选通过 7 项 TLA 和 6 项组合对照精度，正式同卡结果为 32.358 / 31.423 ms（97.11%），与默认的 97.32% 相近。保留较简单的原调度，候选源码和数据归档在 `flex-prefix-final.KYOf3M/`。

两种 stage 配置使用同一源码。小规格对全部输出核对独立 FP32 参考，覆盖 normal 双 seed、uniform、peaked，并检查整块空与 partial 块内全空行输出精确为零。52K 对 Mojo 全量输出并用独立 FP32 检查跨文档、文本/图像边界和定点采样行。各方轮换三轮各 40 次，Mojo 单列 `pipeline_pct_of_mojo`。

`TLA_FLEX_FIA_BASELINE=1` 启用针对当前多模态 mask 的 Ascend C 组合对照。文本走带完整 4D mask 的 FIA。图像走双向 FIA，再把与图像块互斥的前导文本贡献按 FP32 LSE 合并。原生调用、切片拷贝、合并和输出拼接计入计时，静态 mask 计划与候选一致地复用。组合通过五组 CPU 对照和六项设备精度，正式性能证据在 `~/workspace/npuir/temp/20261110-tla-14-validation/flex-predicate.I88PjI/`。通用 BlockSparseAttentionV2 当前仅接收块级 mask。 [FIA V5](https://github.com/hicann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md) · [BlockSparseAttentionV2](https://github.com/hicann/ops-transformer/blob/master/attention/block_sparse_attention/docs/aclnnBlockSparseAttentionV2.md)

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/flex-attention-pipeline -k test_ir
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/flex-attention-pipeline -k test_run
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q third_party/ascend/unittest/tla_ut/kernels/flex-attention-pipeline -k test_perf
TLA_TEST_STAGE=run TLA_RUN_PERF=1 TLA_FLEX_FIA_BASELINE=1 pytest -q third_party/ascend/unittest/tla_ut/kernels/flex-attention-pipeline -k test_perf
TLA_FLEX_SKIP_FULL_MASK=1 TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/flex-attention-pipeline -k test_run
TLA_FLEX_SKIP_FULL_MASK=1 TLA_FLEX_MERGE_BUCKETS=1 TLA_FLEX_Q_TILE=128 TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/flex-attention-pipeline -k test_run
```

两种 stage 配置通过 1109 离线编译，首轮 4 个短规格在 blue2 通过。52K 主规格包含单槽分桶，真实 JIT 原本会把 `SLOTS=1` 专化为常量，折叠流水区域要求保留的循环。通过 `TLAJITFunction` 保持该参数动态后，128-token 单槽回归与 52K 主规格均在 blue2 卡 1 通过。两种 TLA 配置都通过了与 Mojo 的全量比较及独立 FP32 参考检查。复测证据保存在 `~/workspace/npuir/temp/20261110-tla-14-validation/flex-class.SpKZq4/pytest.log`，三种调优配置均通过 7 项设备精度。Q128、完整块免 mask 和合并分桶的默认组合在 3×40 正式计时为 88.302 ms，Mojo 为 35.059 ms（39.70%）。这条对照不计作 Ascend C。

列式默认路径通过 7 项设备精度，52K 的 3×40 设备中位数为 35.879 ms，Mojo 为 33.800 ms（94.21%）。证据在 `~/workspace/npuir/temp/20261110-tla-14-validation/flex-lightning-binary.NdSYZR/`。编译时内联 SIMD softmax helper 会在普通 Triton builder 丢失 TLA buffer 类型，改为 task 内表达。将 mask 分配移到 pipeline 循环体的任务外之后生成了目标二进制。
