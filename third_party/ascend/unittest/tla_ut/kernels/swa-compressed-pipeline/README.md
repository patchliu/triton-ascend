# SWA + 压缩 KV HCA Prefill pipeline

| batch | Q heads | KV heads | Dq / Dv | Q 长度 | 原始历史长度 | 压缩 KV 长度 | 原始窗口 | 压缩比 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 1 | 512 / 512 | 4096 | 32768 | 256 | 128 | 128 |

规格按 DeepSeek V4 文本 HCA 解释。32768 表示含当前 chunk 的原始 token 数。TND 输入分别为 `Q[4096,64,512]`、`original[32768,1,512]`、`compressed[256,1,512]` 和 `sink[64]`。Q、两个共享 K/V 池和输出为 BF16，sink 为 FP32。比较从已有的两份 KV 和已应用 RoPE 的 Q 开始，输出是模型后处理之前的 BF16 attention，关闭可选 LSE。压缩器、RoPE 和输出投影属于上游或下游接口。 [DeepSeek 的 prefill 数据组装](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/v4_modules/attn.py) · [CANN SparseFlashMLA](https://github.com/hicann/ops-transformer/blob/master/attention/sparse_flash_mla/docs/aclnnSparseFlashMla.md)

第 `i` 个 query 的绝对位置为 `p = 32768 - 4096 + i`。原始 KV 可见区间为 `[max(0,p-127), p+1)`，压缩块 `j` 代表原始 `[128*j, 128*(j+1))`，可见区间为 `[0,(p+1)//128)`。已完成的压缩块可与最近窗口覆盖部分原始位置，两份 KV 仍分别计权。HCA 遍历完整可见的压缩前缀。两路以相同 `QKᵀ / sqrt(512)` 计算，使用同一分母，FP32 sink 对每个 head 增加 `exp(sink)` 并贡献零值。 [CANN HCA 因果边界与公式](https://github.com/hicann/ops-transformer/blob/master/attention/sparse_flash_mla/docs/aclnnSparseFlashMla.md)

## TLA 实现与优化

- 每个 work item 处理一个 query 的 64 个 head，共享读入的 KV。一个 `tla.pipeline` 首轮读 128 条原始窗口，之后读压缩前缀，右边界在 Vector `softmax` task 屏蔽。Q 和在线 softmax 状态跨迭代常驻，sink 在初始化时只计一次。`load_kv` 在 stage 0，QK、softmax、概率发布、PV 和累加同处 stage 1，保证更新顺序。stage 0 对照共用同一源码。
- 主规格每个 query 读取三个 KV tile，共 384 条 KV，最后一块按 query 位置屏蔽。连续布局可直接 DMA 两路数据，最后不足 128 行的压缩缓存才在每次调用中用零填充。循环边界由 query 的逻辑位置计算，减少公共标量加载的核归属问题。
- CANN 参考路径使用任务 metadata、S2 分块、共享 K/V 和在线归一化。后续优先比较多 query 共享压缩 tile、同组内相同压缩前缀的调度、QK/PV 的 L0C 容量和尾块利用率。64-head tile 保留了一次载入供全部 head 复用；进一步把不同 query 的 head 维与 token 维联合分块前，需确认资源与同步。
- 1109 的 TLA 转换暂不支持指针类型的 `arith.select`。修订版在同一 Cube load task 内用分支直接发出 DMA。编译器诉求包括支持动态地址选择、追踪两来源共享 buffer 的依赖、输出每级缓冲实际分配及 stage 版本，并支持带独立有效长度的尾块 DMA 以去掉预填充。可以评估把 NZ 概率发布抽成带内存效果的布局原语。

## 验证与对照

主规格和 `(Sq,history)=(129,129)` 都有精度入口。边界用例覆盖首个压缩块生效、原始窗口首次移位和压缩尾块填充。两种 TLA stage 配置与原生输出分别对比独立 CPU FP32 联合 softmax。分段汇总整张输出的最大误差和 RMS，并分别检查参考与设备输出是否为有限值。CPU 检查还包含已知解析结果和两种规格的 BF16 概率流水模拟。

原生对照依次尝试 TorchNPU、CANN 9.2 随 Transformer Ops 安装的 `cann_ops_transformer.ops.attention.sparse_flash_mla` 和 vLLM-Ascend BF16 接口，并使用对应的 metadata 算子。测试也可通过 `aclnn_hca.py` 直接调用 CANN 9.1 `libopapi.so`。这个适配器核对本机头文件签名，使用 1024 项 int32 metadata 并按 Torch 当前 NPU stream 提交。两个未设置的 `seqused` 参数传入形状 `[0]` 的 int32 描述符，供旧 metadata 接口内部的连续化处理。该旧 ABI 只用于原始长度整除 128 的规格。 [vLLM-Ascend BF16 算子包装](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/attention/sparse_flash_mla/sparse_flash_mla_torch_adpt.h)

对照设置 `cmp_ratio=128`，原始 mask 为 4、左宽 127、右宽 0，压缩 mask 为 3。两边接收同样的原始与压缩 KV、sink、score scale 和输出要求。性能入口在相同输入通过精度后，先比较固定规格且原生 metadata 已缓存的设备调用，3×40 轮换。另记录包含原生 metadata 生成的同步 wall time。TLA 的设备填充等每次调用成本始终计入。记录占用、实际 kernel、wheel、原生库和编译器哈希。

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/swa-compressed-pipeline -k test_ir
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/swa-compressed-pipeline -k test_run
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q third_party/ascend/unittest/tla_ut/kernels/swa-compressed-pipeline -k test_perf
```

本地已收集 14 项测试，3 项 CPU 检查通过。修订后四组均通过 1109 离线编译。blue2 首轮通过两个边界精度用例，主规格改用 CPU FP32 参考后两种 TLA stage 都通过。首轮直接 ACLNN 对照在 CANN 9.1 metadata workspace 查询时以 139 退出。按官方包装调整空序列描述符并传入压缩余数 0 后，CANN 9.1 的原生 metadata 已发往设备，返回 AICPU 21008。CANN 9.2 包提供官方入口，但当前设备拒绝缺签名的 AICPU 包，未产生有效原生性能。

CANN 9.1 自带的主机 metadata 实现按每核 8 个整数和每调度项 4 个 query 生成数据，实际设备内核读取每核 9 个整数并按单 query 调度。使用公开 v9.1.0 scheduler 生成与设备布局一致的 1024-int32 数据，按实际核数和长度选取预计算文件。此修订仍在原生 Ascend C 主核报 507015，底层记录为标量访存地址非法。日志在 blue2 的 `hca-fixed-metadata.lryibh/card1-native.OLsY1v/`。只有原生通过同一精度门槛后才进行计时。 [CANN v9.1.0 metadata 源码](https://github.com/hicann/ops-transformer/tree/v9.1.0/attention/sparse_flash_mla_metadata)
