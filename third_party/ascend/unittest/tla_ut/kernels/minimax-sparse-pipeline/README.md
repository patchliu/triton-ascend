# MiniMax-M3 Sparse Attention

MiniMax-M1 使用混合 Lightning Attention，MiniMax-M2 的注意力是密集 GQA。这里实现 MiniMax-M3 稀疏层中选块之后的 BF16 GQA。M3 官方配置为 64 个 Q heads、4 个 KV heads、128 维 head、每 token 每 KV group 选 16 个 128-token 块，并保留 query 所在块。Q 长 4096、KV 长 32768 是使用该模型维度的代表性 chunk prefill。 [MiniMax-M1](https://github.com/MiniMax-AI/MiniMax-M1) · [MiniMax-M2 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m2/modeling_minimax_m2.py) · [MiniMax-M3 配置](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json)

| 用例 | bs | num_head | kv_head | head_dim | q_len | kv_len | KV block | TopK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chunk prefill | 1 | 64 | 4 | 128 | 4096 | 32768 | 128 | 16 |
| 短前缀精度补测 | 2 | 64 | 4 | 128 | 96 | 131/521 | 128 | 16 |

Q 为连续 `TND`，K/V 为 `[physical_pages,128,Hkv,128]` 即 `PA_BBND`。输入索引和计数为 `[Hkv,T,TopK]` 和 `[Hkv,T]`。索引的有效前缀按逻辑块号升序，页表将逻辑块号映射为物理页。TopK 选块、QK norm 和 RoPE 都在输入构造边界之前。右对齐 causal 的最后可见位置为 `实际 KV 长度 - Sq + token_position`。输出为 BF16 `TND`。

## 实现参考

MiniMax 的报告采用 KV-outer，使多个 query 重用已经搬入的 KV 块，并合并不同 KV 分片的结果。vLLM-Ascend A5 prefill 同样提供由 Q→K 转换为 K→Q CSR 的路径。首版采用 Q-outer，每个工作项计算一个 query token 的一个 KV group，将 16 个 query heads 组成一次 Cube tile。它可直接消费模型的选块结果，便于建立精度和性能基线。 [MiniMax MSA 报告](https://arxiv.org/abs/2606.13392) · [MiniMax MSA 实现](https://github.com/MiniMax-AI/MSA) · [vLLM-Ascend MSA](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/minimax_m3/ops/msa_m3_npu.py)

`load_block` 根据索引和页表把 K/V 搬入 L1。`qk`、`softmax`、`publish_probability`、`pv`、`accumulate` 组成下一 stage。两个 AIV 分担 16 行，每个工作项保留 FP32 在线最大值、分母与分子。每页 buffer 版本和 Cube/Vector 同步由编译器生成。Stage 0 对照保留同样的数据布局和切分。

Cube 和 Vector 分别在对应 task 内读取稀疏元数据。短请求在多余的迭代中重复搬一个合法块，Vector 将其贡献屏蔽。此路径依赖 M3 每行至少有一个选块并包含本地块的合同。

`TLA_MINIMAX_RESOLVE_METADATA=1` 在设备上预先解析每个选块的物理页和可见 token 数。Cube 和 Vector 在相应 task 中各读取一个标量，共用同一套 pipeline。主规格的两个 int32 中间张量共 2 MiB，预处理计入每次调用耗时。

## 优化与编译器诉求

| 观察点 | 优化和编译器诉求 |
|---|---|
| 多 token 重复读相同 KV 块 | 统计块热度和重复读取量。评估选块局部重排，以及 KV-outer 与分段归约。后者需要能表达不规则的 K→Q 工作列表和跨任务局部归约。 |
| 逻辑块到物理页的两次标量读取 | 比较 task 内解析与设备预处理的总成本。编译器需要追踪共享标量的核归属和同步，并显示 stage 0 的 Scalar 依赖。 |
| Q 外切，K/V 按 head 跨步搬运 | 测量 ND2NZ 指令数、L1 双缓冲和 Fixpipe 等待。希望编译器报告跨步搬运降级原因和各 stage 的资源、依赖。 |
| 实际选块数小于 TopK | 测量 padding 的吞吐损失。希望支持依赖动态计数的混合流水循环界，编译器能处理 Scalar load 的核归属。 |

## 精度和性能验证

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/minimax-sparse-pipeline -k test_ir
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/minimax-sparse-pipeline -k 'test_run or test_native'
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q -s third_party/ascend/unittest/tla_ut/kernels/minimax-sparse-pipeline -k test_perf
TLA_TEST_STAGE=run TLA_MINIMAX_NATIVE=splitkv pytest -q third_party/ascend/unittest/tla_ut/kernels/minimax-sparse-pipeline -k test_native
```

测试打乱物理页，按 KV group 和 query token 构造不同的选块列表并保留本地块。FP32 参考按页表收集已选 KV，对单行所有可见 key 直接做 softmax。候选两种 stage 配置和 CANN 都比对独立参考。正常输入双 seed、等概率输入和尖峰输入各重复三次。归一化最大误差阈值 0.008，相对 RMS 阈值 0.005。

CANN 对照有两个入口。默认 `gbsa` 使用 `cann_ops_transformer.ops.generic_block_sparse_attention`，设置 `TND/PA_BBND`、`block_shape=[1,128]`、layoutSparsePattern=4、causal、无量化和无 LSE。`splitkv` 使用 vLLM-Ascend A5 的 `aclnnMinimaxSparseAttentionSplitKv`，设置 `inner_precise=0`，每次调用先把 Q→K 索引转换为 K→Q CSR。两个入口都使用和 TLA 一样的选块结果、计数、长度及页表，并把设备元数据转换计入耗时。只有 CPU CSR 包装可用时，`splitkv` 只运行精度检查。 [Ascend GBSA 接口](https://github.com/hicann/ops-transformer/blob/master/attention/generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md) · [vLLM-Ascend 的 M3 调用](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/minimax_m3/ops/msa_m3_npu.py)

仅精度通过后计时。同卡三方轮换，3×40 次，记录实际设备任务、占用和 Python 调用时间。缺少 CANN Python 扩展时，仍可运行候选精度测试。性能百分比使用 Ascend C 设备耗时中位数除以 TLA pipeline 设备耗时中位数，再乘 100%。

blue2 的首版与完整块免 mask/重叠调度候选均通过主规格和短前缀精度。新增物理页与可见长度预处理已通过 1109 的六项目标离线编译，覆盖两种规格下的预处理核和两种 stage 配置，实机待测。

### Ascend C 短规格复现

隔离 CANN 9.2 GBSA 的设备初始化被 AICPU 签名检查阻止。CANN 9.1 用官方 ops-transformer `762e432` 构建了 split-KV 和设备 CSR。以下日志和构建目录都相对于 blue2 的 `~/workspace/npuir/temp/20261110-tla-14-validation/`。

用 `TLA_MINIMAX_NATIVE=splitkv`、仅加载官方 split-KV Torch 包装，并运行 `-k 'test_native and short-prefix'`，可复现 CPU CSR 路径。双 batch 正常输入的归一化最大误差和相对 RMS 分别为 0.6062 和 0.3821，原门槛为 0.008 和 0.005。详情见 `minimax-splitkv-eval.TQQo2D/short-prefix.log`。

均匀 Q 的交叉检查固定 Q 长为 96，逐个 batch 比较独立 FP32 参考，见同目录 `short-crosscheck.log`。三个形状、四个 KV 头的 CPU CSR 行指针、全局 query 索引和原 rank slot 均与逐边枚举完全一致。

| KV 长度 | 相对 RMS，inner_precise=0 | 相对 RMS，inner_precise=1 |
|---|---:|---:|
| 131 / 521 | 0.4047 / 0.2069 | 0.4031 / 0.2065 |
| 131 | 0.4877 | 0.4864 |
| 521 | 0.1895 | 0.1891 |

同一轮的设备 CSR 在 `K2qCsrMeta` 报 `EZ1009`。检查发现增量包的 ascend950 `aic-ascend950-ops-info.json` 和 `binary_info_config.json` 只登记了上一轮的 split-KV。按父算子 `--ops=minimax_build_k2q_csr` 在 `minimax-csr-clean.Efv9KY/` 独立重建后，两份实际打包清单均包含 Meta、Hist、RowPrefix、TilePrefix 和 Scatter，五个设备二进制也齐全。干净包实机待测。两条原生路径目前均无可配对性能数据。 [共同验证协议](../PIPELINE_VALIDATION.md)

用 `TLA_MINIMAX_OVERLAP=1 TLA_MINIMAX_SKIP_FULL_MASK=1` 复现重叠调度候选。仅在所有选中 key 都可见时跳过 mask，尾页和 padding 仍走原因果逻辑。
