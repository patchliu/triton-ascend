# Lightning Indexer

主规格为 `(bs,Hq,Hkv,D,Sq,Skv) = (1,32,1,128,1024,32768)`。TopK 采用 DeepSeek-V3.2 官方配置的 2048。Q、K、W 都是 BF16，采用连续 BSND，输出 INT32 `[B,Sq,1,2048]`。本版使用完整物理长度和右对齐因果 `sparse_mode=3`，只返回 indices。原生对照为 `torch_npu.npu_lightning_indexer`，使用相同布局、精度和默认 `return_value=False`。量化接口使用独立的 INT8 或 FP8 输入及反量化 scale，应单列对照。[DeepSeek-V3.2 配置](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) · [CANN BF16 接口](https://github.com/hicann/ops-transformer/blob/master/attention/lightning_indexer/docs/aclnnLightningIndexer.md) · [CANN 量化接口](https://github.com/hicann/ops-transformer/blob/master/attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md)

## 参考实现与调度

分数为 `sum_head(ReLU(Q @ K.T) * W)`，右对齐后第 `q` 行的可见数为 `max(0, Skv - Sq + q + 1)`。Ascend C 950 按 query 分组复用 K，先算分数再按 8192-key 块做直方图 TopK 并合并。当可见数小于 TopK 时，直接返回升序有效位置并用 `-1` 补齐。可见数等于 TopK 时仍走选择路径。公开接口规定长行的 TopK 集合，未规定呈现顺序或同分优先级。[Ascend C 950 Vector 实现](https://github.com/hicann/ops-transformer/blob/master/attention/lightning_indexer/op_kernel/arch35/lightning_indexer_service_vector_arch35.h) · [Ascend C 950 TopK 实现](https://github.com/hicann/ops-transformer/blob/master/attention/lightning_indexer/op_kernel/arch35/vf/lightning_indexer_topk.h)

TLA 主规格把四个 query 的 32 heads 合并为 128 行，共享每次搬入的 128 个 K。Q 和 W 常驻当前工作项，每次 `tla.pipeline` 迭代处理一个 K tile。`load_key`、`dot`、`reduce_heads` 分别处于 stage 0、1、2。Fixpipe 按列分发，两个 AIV 各负责连续 64 个 key。每个 SIMD lane 对一个 key 做 ReLU、带权 head 归约及因果 mask。stage 0 对照保留同一切分和任务顺序。

首版把完整 FP32 分数写到 GM，使用 `torch.topk` 做设备侧选择，所有选择、短行处理和转换开销均计入候选。主规格分数为 128 MiB，端到端会包含这次写回及再次读取。精度额外单测分数，性能比例只使用返回最终索引的完整调用。CANN 的模型优化报告也将 TopK 外置开销列为融合动机。[CANN DeepSeek-V3.2-Exp 算子指南](https://github.com/hicann/cann-recipes-infer/blob/master/docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_ascendc_operator_guide.md)

## 验证与后续优化

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/lightning-indexer-pipeline -k test_ir
TLA_TEST_STAGE=compile pytest -q third_party/ascend/unittest/tla_ut/kernels/lightning-indexer-pipeline -k test_compile
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/lightning-indexer-pipeline -k 'test_run and not test_run_native'
TLA_INDEXER_NATIVE=ascendc_v2 TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/lightning-indexer-pipeline -k test_run_native
TLA_INDEXER_NATIVE=ascendc_v2 TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q -s third_party/ascend/unittest/tla_ut/kernels/lightning-indexer-pipeline -k test_perf
```

设备精度覆盖短行补位、阈值等于 TopK 的边界、同分和 32K 主规格。分数逐元素与独立 FP32 比较，长行验证每个索引的范围、唯一性及对应分数是否达到 CPU TopK 给出的阈值容差，短行逐位置比较。性能使用同一主规格输入，三方轮换三轮各 40 次，先通过精度门槛，记录所有设备任务、wall time、设备占用及实际工具链。设置 `TLA_INDEXER_TUNING=1` 可附加 Cube/Vector 同 stage、四条 head 累加链及其组合。

- 比较 stage 0 与 stage 1/2 的 MTE2、Cube 和 Vector 重叠，评估多条独立累加链和一次性把 W 预转为 FP32。
- 评估 tile 内保留有界候选以及多级合并，缩小长上下文全量分数的 GM 流量。TopK 截断、同分和短行补位均需要同一接口验证。
- 编译器可提供 SIMD 横跨 head 的加权归约、稳定携带索引的局部 select/merge、以及 Fixpipe split 和跨 stage 缓冲资源报告。后续与局部 TopK 融合时，再验证候选缓冲的跨迭代依赖。

首版主规格通过 1109 离线编译。用运行时 tile 数保留单次循环后，短行、等于 TopK 的边界、同分、长行及三种流水候选共 7 项通过 blue2 精度。主规格 seed 1104 上，早期两种 TLA 调度的索引集合都与 CPU FP32 TopK 在全部 1024 行一致。CANN 9.1.0 V1 在排序前将分数舍入到 BF16，有 730 行超阈值，显式传入实际 Q/K 长度未改变结果。日志位于 `~/workspace/npuir/temp/20261110-tla-14-validation/li-topk-diagnostic.4C9INw/` 和 `lightning-resume.qm4yyE/`。

同一安装包的 950 V2 使用 32-bit 排序键。四条 indices-only 精度各跑三次并全部通过，cutoff 差为零。`ascendc_v2.py` 接收同一 BF16 输入，并在每次调用时无损扩宽 W。固定规格可在准备阶段生成并复用 metadata，动态计划把 metadata 生成计入调用。V2 使用 `return_value=0`，因为安装版的 32-bit 路径没有写回可选 scores。 [CANN V2 接口](https://github.com/hicann/ops-transformer/blob/master/attention/lightning_indexer_v2/docs/aclnnLightningIndexerV2.md)

blue2 卡 1 的动态 V2 对照通过完整占用检查。TLA 默认、stage 0、Fixpipe ReLU 与 Ascend C V2 依次为 2523.6105、2312.8050、2885.5765、13928.1420 µs。V2 逐任务中位数为 metadata 43.3570 µs、W 转换 1.6265 µs、主核 13883.1640 µs。结果与原始 trace 位于 `~/workspace/npuir/temp/20261110-tla-14-validation/lightning-v2-perf.NRTFol/device-1.IGWVfs/`。

固定规格复用 metadata 的正式对照在 blue2 卡 2 通过 12 项精度和同卡 3×40。TLA 默认、stage 0、Fixpipe ReLU 与 Ascend C V2 依次为 2508.4145、2287.5225、2878.0240、13896.9665 µs。默认流水版达到 Ascend C V2 的 554.01%。V2 每调用保留 W 转换和主核，metadata 在准备时生成一次。默认流水版与 V2 的热调用 wall 分别为 2539.8010 和 13934.1935 µs，冷首调用分别为 2641.001 和 190.006 ms，TLA 冷调用包含 JIT。V2 的一次性 metadata 准备为 15.314 ms。证据在 `~/workspace/npuir/temp/20261110-tla-14-validation/lightning-v2-cached.nlnOpY/device-2.WbnBS1/`。

新候选在读取 Q/W 时将 W 转成一次 FP32，供所有 key tile 复用。可选 Fixpipe ReLU 把原本逐 head 的激活移到 Cube 输出搬运，保持 FP32 score 和 TopK 合同。
