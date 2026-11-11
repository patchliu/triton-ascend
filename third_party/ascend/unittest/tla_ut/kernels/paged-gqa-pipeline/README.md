# Paged GQA pipeline

同一个算子覆盖 Paged GQA Decode 和 Chunked Paged GQA Prefill。共有的验证协议和归档复盘见 [PIPELINE_VALIDATION.md](../PIPELINE_VALIDATION.md)。

| 规格 | Q，BNSD | 实际 KV 长度 | mask | Q tile |
|---|---|---|---|---|
| Decode | `[16,32,1,128]` | 每请求 4096 | 全部有效 KV | 16 heads |
| Chunked prefill | `[4,32,512,128]` | 每请求 8192，包含当前 chunk | 右对齐 causal | 128 tokens |
| 精度补测 | `[4,32,1,128]` | `[65,128,257,511]` | 按请求屏蔽尾页 | 16 heads |

Q/K/V 和输出是 BF16，K/V 各为 `[num_pages,8,128,128]`，页表与有效长度是 int32。每组 Q heads 共享同一个 KV head。Prefill 第 `q` 行的最后可见 KV 位置为 `length - Sq + q`。测试打乱物理页分配，用相同页面、页表和实际长度调用 TLA 和原生 FIA。

## 参考实现与调度

参考 vLLM-Ascend 的 `attention_v1.py`、Mojo A5 `flash_attention.py` 的分页读取，以及 FA/DSA 归档里的 page/tile 预取、在线 softmax、状态寿命与输出缓冲分离。CANN 对照使用 `torch_npu.npu_fused_infer_attention_score`。Prefill 用 `sparse_mode=3` 与 2048×2048 压缩 causal mask。 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/attention_v1.py) · [CANN FIA 规格](https://github.com/hicann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md)

默认每次迭代处理一个 KV 页。`load_kv` 在 stage 0 查页表并搬 K。Decode 默认同 stage 并同时搬 V；Prefill 的 QK、softmax/概率发布、PV 和累加分别在 `(0,1,2,3)`，V 延后到 PV 搬运。Q 和在线 softmax 状态在流水外，当前页的 K/V、score、P 和 partial 在流水内。Decode 与 prefill 的差异集中在 Q tile 与输出映射、搬运调度和右对齐 mask。

为适配当前编译器，Cube 和 Vector 在各自任务中读取实际 KV 长度，公共循环上界由页表容量计算。超出实际长度的迭代复用最后一个有效页，Vector 按逻辑页位置屏蔽其贡献。满页表的主规格保持原来的迭代数，短请求会多跑完全屏蔽的页，并且每页增加标量读取。保留动态循环上界且能为共享标量分配明确的核归属，是后续的编译器诉求。

Decode 先用 16 行 Cube 对齐 tile 读取现成的 Q 行，存储该 KV head 对应的 4 行。这省去额外的 Q 打包任务，同时引入了多余的矩阵乘与 Vector 行计算。Prefill 默认用 128 个连续 query token，每个工作项负责一个 Q head。列式 score 为 K @ Qᵀ，每个 AIV 的 64 个 SIMD lane 各处理一个 query。

## 优化与编译器检查

1. 优先检查 Decode 的 4/16 有效 Q 行、两 AIV 分工和页表标量读取。下一步比较小 M 路径、工作项分组和 split-KV，记录减少的填充计算、重复读及额外归约。
2. Prefill 先比较 Q tile 大小和 QK 输出布局，再评估把 mask/softmax 改成每 lane 一个 query。统计每个 KV 页服务的有效 Q 行以及 KV 重读量。
3. 从生成 IR 确认页表 load 的 Cube 归属、K/V 自动版本、两个 AIV 的 P 发布与 Cube 消费顺序。若条件写回或标量效果限制编译，保留最小 IR 与实际失败 pass。
4. 编译器方向先记录两项待验证需求。提供小 Q group 的 Cube/Vector 映射能力，减少对齐填充。提供可追踪的页表读取和跨 stage 缓冲版本诊断。

## 测试

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/paged-gqa-pipeline -k test_ir
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/paged-gqa-pipeline -k test_run
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q -s third_party/ascend/unittest/tla_ut/kernels/paged-gqa-pipeline -k test_perf
```

精度用 normal 双 seed、uniform、peaked，每组重复三次。流水版、stage 0 对照和 Ascend C 都与 FP32 参考比较。归一化最大误差和相对 RMS 的首版阈值均为 `0.005`。性能只对两条主规格采样，三方轮换，3×40 次，每轮核对占用。设备任务时间和 Python 调用时间分别记录，性能百分比以 Ascend C 耗时除以流水版耗时。

当前版本在 `dev/tla-pipeline-1108` 和 `tla-pipeline-1109` 组合上完成离线编译。2026-11-10 在 blue2 的 Ascend950PR 卡 2 上，首版 decode、prefill 和不等长尾页的三方精度共 108 次对比全部通过。流水版最差归一化最大误差为 `0.00353725`，最差相对 RMS 为 `0.00229818`。随后 Decode 同 stage 达到 98.42%，Prefill Q128 单页列式从 66.21% 提升到延后 V + deep 的 90.77%。Decode 的多 stage 候选未达 95%；Prefill 的双页候选通过精度，性能为 56.01%。 [共同验证记录](../PIPELINE_VALIDATION.md)

调优入口：`TLA_PAGED_GQA_SCHEDULE` 支持 serial/prefetch/overlap/deep/wide/early_qk；`TLA_PAGED_GQA_Q_TILE`、`TLA_PAGED_GQA_COLUMN_SOFTMAX`、`TLA_PAGED_GQA_PAGES_PER_TILE`、`TLA_PAGED_GQA_LATE_VALUE` 和 `TLA_PAGED_GQA_COMPACT_DECODE` 用于复现消融。
