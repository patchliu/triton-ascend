# Dense MLA Decode

各规格共用同一个 BF16 paged MLA 内核。Q 和 KV 的 512 维 latent 与 64 维 RoPE 可分别输入，也可用连续的 576 维 packed 张量。一个 KV head 的 latent 同时用于 QK 与 PV，RoPE 只参与 QK。输入是完成投影和位置编码后的张量，输出为 512 维 latent attention，可选 FP32 LSE。

| 规格 | bs | num_head | kv_head | head_dim QK/V | q_len | kv_len | 分段数 |
|---|---:|---:|---:|---|---:|---:|---:|
| 单 token | 16 | 128 | 1 | 576/512 | 1 | 32768 | 4 |
| 多 token | 4 | 64 | 1 | 576/512 | 8 | 131072 | 16 |
| B8 H24 | 8 | 24 | 1 | 576/512 | 4 | 143360 | 7 |
| 不等长补测 | 4 | 64 | 1 | 576/512 | 8 | 8/129/261/513 | 3 |
| H24 不等长补测 | 4 | 24 | 1 | 576/512 | 4 | 4/129/260/1031 | 5 |

Q 与 Q-RoPE 为连续 `BSND`，KV 与 K-RoPE 分别为 `[pages,128,512]` 和 `[pages,128,64]`。KV 加入一个单元素 head 维即可传给 CANN 的 `BnNBsD`，两种视图共用同一存储。页表与实际长度均为 int32。实际 KV 长度包含当前 query chunk。第 `q` 个 query token 可见的最后一个 key 是 `length - Sq + q`。单 token 时该条件覆盖全部有效 KV。

## 参考实现与切分

- CANN FIA V5 定义了 Decode MLA 分离 RoPE、分页布局与右对齐 causal mask。Ascend op-plugin 在 950 通过 V5 实现 `torch_npu.npu_fused_infer_attention_score`。 [CANN FIA V5](https://github.com/hicann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md) · [Ascend op-plugin](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp)
- vLLM-Ascend decode 把相同的 latent 作为 K/V 传入 FIA，另传 Q/K-RoPE。DeepSeek 的 Dense FlashMLA 性能报告给出了共享 latent、搬运与矩阵乘交叠、split-KV 合并的实现思路。 [vLLM-Ascend MLA](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/mla_v1.py) · [FlashMLA 性能报告](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md)

一个工作项负责一个请求内的 `(query token, Q head tile, KV 分段)`。Q tile 首版为 32 个 heads、每段 128 页；当前主规格为 64 个 heads、每段 64 页。`load_page` 在 stage 0 查页表并把 latent 和 RoPE 搬到 L1。`qk`、`softmax`、`publish_probability`、`pv` 和 `accumulate` 在可选的后续 stage，默认使用 overlap 调度。Cube 将 Fixpipe 输出均分给两个 AIV。Q 与 FP32 在线状态跨页复用，latent 页在 QK 与 PV 间共用同一 L1 分配。每页的内存版本与同步由编译器生成。

流水次数由页表容量与分段大小计算。Cube 与 Vector 在各自的任务中读取实际长度。超出当前 query 可见范围时复用最后一个可见页的物理地址，Vector 将该迭代完全屏蔽。全空分段保留零分子和零分母。合并 kernel 将各分段的 FP32 分子与分母乘以 `exp(分段最大值 - 全局最大值)` 后分别求和，最后除法并转 BF16。

## 性能和编译器检查

| 观察点 | 调优和编译器诉求 |
|---|---|
| 多头和多 token 重复搬 KV | Q group 可让两个 64 行子块共享一次 KV 加载。单 token 合并两个相邻 head tile，多 token 合并相邻两个 token。宽 QK 可在一次 M128 QK 中计算这两块，按 Vector 子核保持各自的 causal 边界。PV 沿 latent 特征拆成两块 256 列，以复用 UB 和 L0C scratch。 |
| KV 页同时用于 QK 与 PV | 检查 L1 版本数、bank 地址和 Fixpipe 等待时间。希望调度报告给出跨 Cube/Vector 的生产者、最后消费者与峰值资源分配。 |
| 短请求额外执行全掩码页 | 希望支持供 Cube 和 Vector 共同使用的动态循环界，并保留 scalar load 的核归属。当前以静态页表容量保证编译语义。 |
| 分段归约和 workspace | 比较不同 `split_pages` 的同轮样本，拆开记录主 kernel 与 merge。合并 kernel 重复读取行的 max/sum，长序列可以评估合并访存与 kernel 间启动间隔。 |

## 测试

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/mla-decode-pipeline -k test_ir
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/mla-decode-pipeline -k test_run
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q -s third_party/ascend/unittest/tla_ut/kernels/mla-decode-pipeline -k test_perf
```

默认 M64；`TLA_MLA_HEAD_TILE=32` 可复现较小的 Q tile。`TLA_MLA_SPLIT_PAGES` 覆盖两条主规格每段的页数，例如设为 `64` 或 `256`。同一轮的 stage 0 和 stage 1 共用这两个参数。不等长补测保留每段 2 页，继续覆盖完全空的分段。

`TLA_MLA_Q_GROUP=2` 可启用相邻 Q tile 共享 KV，默认在 stage 0 加载、stage 1 计算。`TLA_MLA_WIDE_QK=1` 在两块共享配置下用一次宽 QK 和两次窄 PV。加载使用组内最后一个 query 的合法页边界，每个子块仍按自己的 query 位置做因果裁剪。ragged 的长度 129 使同组最后两个 query 分处第 127 和第 128 个 key，覆盖共享加载的跨页边界。

随机打乱物理页号，三方共用输入、页表、实际长度与 `1/24` scale。多 token 的 CANN 基线传压缩 causal mask，输出与候选同为 BF16 BSND，关闭 LSE。FP32 参考按行直接计算完整 KV softmax。正常输入双 seed、等概率输入和尖峰输入各跑三次，归一化最大误差阈值为 0.008，相对 RMS 阈值为 0.005。不等长补测覆盖 causal 尾页和全空分段。

精度全部通过后才计时。同卡三方轮换，3×40 次，检查设备占用。TLA 性能计入流水主 kernel 和合并 kernel，设备任务时长与端到端调用时间分别保存。性能百分比为 Ascend C 设备耗时中位数除以 TLA pipeline 设备耗时中位数，再乘 100%。记录编译器和源码哈希。

首版 32-head tile 在 1109 离线编译通过，blue2 的单 token、多 token 和不等长精度也已通过。M64 也通过三条精度用例。主规格默认每段 64 页，QK 提前到 stage 0，完整页跳过 mask，PV 和累加在 stage 2。使用 512 列合并后，正式同卡结果为单 token 75.17%、多 token 87.96%。短测把 partial 改为 query-major 没有稳定额外收益，保留原 workspace 布局。 [共同验证协议](../PIPELINE_VALIDATION.md)

默认按 KV 分段优先派发，使相邻工作项能在 L2 复用同一个分段的 KV。保持 M64、64 页及 512 列合并后，同卡正式结果为单 token 983.6050 / 742.3560 µs（75.47%），多 token 3484.8450 / 3260.6580 µs（93.57%）。576 维单次 QK 两规格均较慢。BF16 partial 通过精度但未显示加速，保留 FP32。多 token 改为 144 页时，干净复跑为 3413.4015 / 3280.1105 µs（96.10%）。证据在 `~/workspace/npuir/temp/20261110-tla-14-validation/mla-144-clean.NIAsOS/`。

两个 M64 子块共享 KV 页通过 1109 目标编译和 blue2 单 token、多 token、ragged 精度。blue2 卡 1 同卡 3×40 得到单 token 1213.0610 / 739.3785 µs（60.95%）和多 token 3916.4365 / 3274.5330 µs（83.61%），默认保留较快的单块 overlap。证据在 `~/workspace/npuir/temp/20261110-tla-14-validation/mla-qgroup.KWfsyJ/`。

宽 QK 的两种调度通过 1109 目标编译和 blue2 的三条精度。宽 QK prefetch 的正式结果为单 token 1098.8170 / 739.7825 µs（67.33%）和多 token 3525.7540 / 3278.1915 µs（92.98%）。同轮 stage 0 为 1016.4480 和 3284.4230 µs。证据在 `~/workspace/npuir/temp/20261110-tla-14-validation/mla-wide.HAw4Dx/`。将 64 页调成单 token 37 页、多 token 147 页，可把 28 核中最繁忙核的页数分别从 192 降到 148、从 640 降到 588。两项 stage 0 主核与 7-split、512 列合并已通过离线编译，实机结果待测。将宽 QK 提到 stage 0、其余计算保留 stage 1 的配置在 `hivm-plan-memory` 失败。

## B8 H24 长序列

[官方实现拆解、TLA 调优顺序及编译器诉求](FLASH_MLA_ANALYSIS.md)。

PR #2973 和 1104 附件的 H24 规格是 B1、Sq4、KV131072、page128、BF16，scale 0.04，右对齐 causal，返回 O 和 FP32 LSE。它把四个 query 的 96 行一起计算，以一次 KV 搬运服务全部行，并用满宽 PV512、一页延迟 PV、消费流水线等待和每个 split 权重只算一次的归约减少空隙。B1 H24 完整调用从 209.494 µs 降到 191.918 µs，历史同轮专用 Ascend C prepared 为 180.495 µs。 [PR #2973](https://github.com/triton-lang/triton-ascend/pull/2973)

B8 规格每请求 1120 页。M96 覆盖四个 query，两个 AIV 各处理两组 24 行。尾页按组生成各自的因果谓词，完整页跳过谓词。每段 160 页，在 28 个 Cube 上得到 56 个等长工作项。默认 `late_pv` 将加载、QK 和 softmax 放在 stage 0，PV 与更新放在 stage 1。P、KV 和 rescale 权重的轮转由编译器生成。合并对全 512 列复用同一份 split 权重，可同时产出 LSE。

B8 H24 测试默认给 KV 的 GM→L1 设置 `l2_cache_ctrl=4`，其他规格使用编译器默认策略。用 `TLA_MLA_KV_L2=default` 复跑未开启 bypass 的对照。卡 2 的正式 3×40 对比中，开启 bypass 从 1241.656 µs 降到 1165.487 µs，达到自编 FlashMla 的 68.18%。112-token 三槽通过精度，设备中位数为 1551.018 µs。

正式对照使用从 `ops-transformer/attention/flash_mla` 和 `attention/flash_mla_metadata` 源码构建的算子包。双方读取相同的 packed Q `[8,4,24,576]`、KV `[8960,128,1,576]`、页表和序列长度，默认同时验证 BF16 O 与 FP32 LSE。prepared 使用计时外生成的、仅依赖形状和长度的 metadata。regenerate 单独记录，并将完整调用时间与设备任务时间区分开。设置 `TLA_MLA_RETURN_LSE=0` 时双方只校验 O。FIA 只作为已有历史和可选诊断，设 `TLA_MLA_BASELINE=fia` 可复跑。 [Flash MLA 源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/flash_mla)

在 `ops-transformer` 的 `897efb917ea657c759c4f7ce79a162e4729ebdd4` 和 CANN 9.2.0 构建环境执行以下命令。blue2 的整包构建用时 2 分 09 秒，其中两个并行设备变体分别用时约 35 秒、37 秒。

```bash
bash build.sh --pkg --soc=ascend950 --ops=flash_mla,flash_mla_metadata -j16
bash build.sh --torch_extension --ops=flash_mla,flash_mla_metadata -j8
```

将生成的 .run 安装到私有目录，加载其 `vendors/custom_transformer/bin/set_env.bash`，并在测试 Python 环境安装 `cann_ops_transformer_custom` wheel。首次导入会编译 Torch C++ 封装。测试调用 custom wheel 的接口，并把模块和 vendor 路径写入结果。blue2 的设备侧验证使用系统 CANN 9.1.0。

```bash
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/mla-decode-pipeline -k 'test_run and batch8-h24'
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q -s third_party/ascend/unittest/tla_ut/kernels/mla-decode-pipeline -k 'test_perf and batch8-h24'
```

历史 FIA 诊断中，分离输入、O-only 的 M96 预取流水为 1468.733 µs，延迟 PV 流水为 1203.341 µs。延迟 PV 的同轮 stage 0 和 FIA 分别为 1373.357、1800.637 µs。主核 Level2 中位数约 1168 µs，Cube MAC 672 µs、MTE2 550 µs，合并约 26 µs。计数器可交叠。双页 256-token tile 的两种 PV 切分均通过精度，满宽 PV 的筛选为 1366.606 µs。提前 QK 的满宽配置需要 2,072,576 bit UB，超过当前 2,031,616 bit。

编译器改进集中在两个位置。第一是让跨页 GM→L1 支持可证明边界的动态 DMA 分段长度，从而简洁表达 112-token tile。第二是提供跨核依赖的消费流水线与 scratch 版本报告，定位 QK、P 发布和 PV 的等待，并让接近 UB 上限的调度可以用生命周期区间核查分配。

## 112-token 三阶段流水与 L0 参考

`prepare_call(..., kv_tile=112, split_pv=True)` 使用 packed BF16 Q/KV。物理页仍为 128 tokens，七页对齐到八个计算 tile。每个 tile 从至多两个物理页搬运。默认调度的三个阶段分别是 KV 搬运，QK、在线 softmax 和 P 发布，以及 PV 和 FP32 累加。K/V/P 共用一个 L1 分配，QK 读完 RoPE 后由 P 覆盖其空间。两个 256 列 PV 分块共用一个 UB 交接缓冲。流水版本数、填充、排空与同步由编译器生成，分段结果由独立的 `merge_splits` 归约。

`TLA_MLA_PIPELINE_112=1` 加入流水候选。`TLA_MLA_L0=native112_merge_dynflags_stream_k` 加入显式 BF16 L0 参考，包含 QK 和 PV 的 `mmad_l0`、跨请求的 Stream-K 和单次设备任务内的归约。L0 参考需要 BF16 L0、动态 ND2NZ 和动态 block flag 配套包。设置 `TLA_MLA_PROBE_O2=1` 使用实验包的 O2 bitcode，复现性能对照。

```bash
TLA_MLA_PIPELINE_112=1 TLA_TEST_STAGE=ir python -m pytest -q third_party/ascend/unittest/tla_ut/kernels/mla-decode-pipeline -k test_ir
export TLA_MLA_PIPELINE_112=1 TLA_MLA_L0=native112_merge_dynflags_stream_k TLA_MLA_PROBE_O2=1
TLA_TEST_STAGE=run python -m pytest -q third_party/ascend/unittest/tla_ut/kernels/mla-decode-pipeline -k 'test_run and (batch8-h24 or ragged-h24)'
TLA_TEST_STAGE=run TLA_RUN_PERF=1 python -m pytest -q -s third_party/ascend/unittest/tla_ut/kernels/mla-decode-pipeline -k 'test_perf and batch8-h24'
```

## 显式调度与 CANNBot DSL 对照

`TLA_MLA_EXPLICIT=1` 将 `mla_decode_explicit.py` 加入精度和性能对照，也可在 `test_ir` 单独核查该实现的交接旗标。显式版与流水版共用 host 校验和 split 归约，支持单块 Q tile 的 packed 与分离 RoPE 输入。

`TLA_MLA_L0=all` 在同一套显式调度和归约里加入使用 BF16 MMAD L0 配套包的常规候选。指定模式名可单测一项。`TLA_MLA_L0=extended` 可用于收集全候选 IR，需要额外的动态 ND2NZ 和动态 block flag 实验补丁。实卡结果和已通过的模式见性能分析。

| 模式 | 调度变化 |
|---|---|
| `reload` | 直接调度 L1→L0A/L0B 和 `mmad_l0`，每页重搬 Q，每个 PV 列带各搬一次 P。 |
| `q` | 首个 96 列 Q 分块在分段循环外搬到 L0A，并跨 KV 页复用。 |
| `qp` | Q 首块驻留，两个 256 列 PV 带再共用一次 P 搬运。 |
| `prefetch` | 复用 Q 和 P，并在上一页 PV 发出最后一次 L1 读取后，把下一页预取到回收的 L1 槽。 |
| `pingpong` | 基于 `qp`，为 QK 和 PV 交替使用两个 L0B 槽，并让 P 复用 QK 的 L0A 工作区。 |
| `native112` | 用三个 112-token 计算槽重叠预取、QK 和 PV，P 覆盖 QK 读完的 RoPE 区域。Vector 读取带 128 列行距的 score，分段边界按七个物理页对齐。 |
| `l1_112` | 使用相同的 112-token 三槽和 P 覆盖布局，矩阵乘使用 L1 MMAD 模板。 |
| `native112_k5` | QK 按标杆的 96/112/128/112/128 切成五段。两块独立的 L0A 工作区在 PV 阶段分别存 64/48 列 P，并用两块独立 L0B 交替搬运。 |
| `native112_bands` | 基于五段 QK，每个 256 列 PV 带独立交给 Vector，以便首带累加与第二带计算重叠。 |
| `native112_manual` | 基于分带交接，显式管理 L1 槽、L0 bank 与 Vector P 缓冲的就绪和释放。运行需要支持 `disable_auto_inject_sync` 的 Triton 实验包。 |
| `native112_phase` | 在显式事件版上跨 tile 延续 L0B 双 bank 相位。 |
| `native112_shared_p` | 延续 L0B 相位，把完整 P 搬入一块 L0A 共用缓冲，两个 PV 带复用。 |
| `native112_fused_pv` | 保留显式事件，用一次 Fixpipe 把两个 PV 带一起交给 Vector。 |
| `l1_112_dynamic` | 在三槽 L1 对照中，用运行时行窗口把跨页 KV 变成至多两条 ND2NZ，需要支持 `LoadGM2L1Params.row_count` 的实验包。 |
| `native112_dynamic` | 同一动态 DMA 窗口配五段 QK 的 L0 MMAD。 |
| `native112_manual_dma` | 动态 DMA 窗口配显式事件，需同时启用两个实验补丁。 |
| `native112_merge` | 在融合 PV 基础上，在同一 kernel 的 AIV 全局同步后归约 split。 |
| `native112_merge_wide` | 在 UB 容量允许时，把片上归约从 8 行增到 16 行。 |
| `native112_merge_sum` | 在融合归约版上，延后 FP32 分母更新。 |
| `native112_merge_vec` | 让每个 SIMD lane 计算一行的 split 权重，再按列合并分子。 |
| `native112_balanced` | 按逻辑 112-token tile 均分 split，保持原有物理页表与跨页搬运。 |
| `native112_merge_banks` | 标杆式 L0C bank 和 PV 分带交接，随后在同一任务内归约。 |
| `native112_merge_dma_banks` | 在 L0C bank 隔离后复验动态 ND2NZ。 |
| `native112_merge_tail` | 每个 job 预置 score 的 16 列 padding，省去完整 tile 的尾列选择。 |
| `native112_merge_layout` | 将同一输出行块的 split workspace 排在一起，减少归约阶段的 GM 读请求。 |
| `native112_merge_arena` | 在 AIV 同步点后复用主核的 UB arena，支持 16 行归约。 |
| `native112_merge_arena_banks` | 用独立 L0C bank 和分带交接复验 UB arena。 |
| `native112_merge_arena_vec` | 在已复用的 UB arena 内，向量化 16 行 split 权重。 |
| `native112_merge_qreuse` | 同一 Cube 连续处理同一 Q tile 的 split 时，复用 L1 Q 和 L0A 首块。 |
| `native112_merge_fast_exit` | 末尾 job 结束后直接等待 AIV 归约 fence，保留相邻 job 的缓冲交接。 |
| `native112_merge_dynflags` | 用运行时 P 槽编号交接，省去静态 block flag 分支。需要动态 block flag 实验包。 |
| `native112_merge_dynflags_phase` | 在独立 L0C bank 与动态 flag 下复验 L0B 相位轮转。 |
| `native112_merge_dynflags_shared` | 在相位轮转上复刻一次完整 P 搬运，由两个 PV 带共用。 |
| `native112_merge_dynflags_first_pv` | 首个 PV 分块直接写 FP32 累加器，空 split 仍写零。 |
| `native112_merge_dynflags_balanced` | 在共享 P 版上按逻辑 KV tile 切分，降低最长 split 的 tile 数。 |
| `native112_merge_dynflags_dma` | 在首 PV 直接写入版上改用运行时 ND2NZ 行窗口。 |
| `native112_merge_dynflags_stream` | 用双缓冲交叠 FlashDecode 的 split 分子读取与 FP32 累加。 |
| `native112_merge_dynflags_vecsum` | 将 online softmax 的 FP32 分母更新合并到一个跨行 SIMD 寄存器。 |
| `native112_merge_dynflags_hybrid` | 合并运行时 ND2NZ、跨行分母和双缓冲 FlashDecode。 |
| `native112_merge_dynflags_stream_k` | 进一步按实际 Cube 核数分配跨请求的连续 KV tile，并按请求归约有效 split。 |
| `native112_late_sum` | 保留 FP32 tile sum，将运行分母的更新移到首带 PV 累加。 |
| `native112_l0c` | 复刻标杆的两个 128 KiB L0C bank，按 bank 交接 MMAD 与 Fixpipe。 |

这些模式共用 128-token 物理页、在线 softmax 和归约，保持 packed BF16 O/LSE 的验证口径。比较显式 L1 对照和 L0 候选时，同时设置 `TLA_MLA_EXPLICIT=1`。设备计时包含主核及 `merge_splits`。

`TLA_MLA_CANNBOT=1` 为 `flash_mla` 基线规格加入 CANNBot DSL。评测需要 `cannbot-dsl==0.7.0`，并将样例提交 `6e5bef20c29aad8bc74c66ae995aeeb01a4789f4` 下的 `samples/flash_mla_with_kvcache` 放入 `PYTHONPATH`。原样校验接受 H64、H96。H24 实验在隔离副本的 `checker.py` 允许列表加入 `24`，并设置 `TLA_MLA_CANNBOT_ADAPTED_H24=1`。Q 由 BSND 零拷贝转为 TND，O 与 LSE 从 NTD 转回公用布局。计时外生成和复用 metadata，记录三轮各 40 次的最小值和全部样本中位数。 [固定版本样例](https://gitcode.com/cann/cannbot-dsl/tree/6e5bef20c29aad8bc74c66ae995aeeb01a4789f4/samples/flash_mla_with_kvcache)
