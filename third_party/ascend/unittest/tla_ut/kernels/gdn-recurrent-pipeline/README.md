# GDN Recurrent Decode pipeline

| batch | Q/K heads | V heads | Dk / Dv | query length | state |
|---:|---:|---:|---:|---:|---|
| 64 | 16 | 16 | 128 / 128 | 1 | BF16 `[slots,Hv,Dv,Dk]` |
| 64 | 16 | 32 | 128 / 128 | 1 | BF16 `[slots,Hv,Dv,Dk]` |

输入是连续的 TND。`q` 和 `k` 共享头数，按 `value_head // (Hv / Hk)` 选 Q/K head。`g` 和 `beta` 按 value head 存储，分别为 FP32 和 BF16。每个请求有一个有效且唯一的 int32 state slot。算子输出 BF16 `o` 并原地更新对应的 BF16 state。Q/K 归一化若由模型需要，则在接口外完成。原生规格见 [TorchNPU Recurrent Gated Delta Rule](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_recurrent_gated_delta_rule.md)。

递推先算 `S_decay = exp(g) * S`、`delta = beta * (v - S_decay @ k)`，再算 `S_new = S_decay + delta ⊗ k` 和 `o = S_new @ (q * scale)`。FP32 SIMD 完成两次点积和秩一更新，最后分别转换 state 和输出。模型端的 Q/K 布局及 FP32 状态选择参考 [vLLM-Ascend GDN 调度](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/gdn.py)，当前只比较原生接口同规格的 BF16 状态。

## 流水与优化

- 一个 work item 负责 `(request, value_head)` 内互不重叠的一段 value 行。默认一段包含 64 行。`load` stage 0 读取 state/Q/K/V 和连续的 beta head 行，`update` stage 1 做门控与递推，`store` 默认在 stage 2 写回 state 和输出。跨 work item 流水。单 token 内的递推依赖保留。
- 优先测 GM 读写和 Vector 归约瓶颈。默认一块状态 tile 为 16 KiB，满规格主要开销来自每步读写所有 state。可比较 32/64/128 行切分、减少 Q/K 复读，以及在同一 UB tile 上原地更新的版本。先用同 tiling 的 stage 0 对照隔离流水贡献。
- 需要查看按任务的 UB 多版本、MTE 等待和向量寄存器压力。若设备编译阻止间接地址上的原地流水，需要让编译器接受基于唯一 slot 的分区合同，并能报告具体的冲突任务及地址。

## 验证

`test_ir` 检查两种 head 数的三任务结构。精度用 BF16 输入和独立 FP32 递推参考，对比输出和写回状态，并验证闲置 slot 原样保留。每个输入族重复三次。性能使用同输入、状态布局、dtype 和 scale，轮换 TLA、stage 0 和原生 `torch_npu.npu_recurrent_gated_delta_rule`。每轮从相同 state 开始，warmup 与测量均执行连续原地更新。每轮的 state reset 在计时之前。默认三轮各 40 个设备样本，保留设备占用、算子列表、编译器哈希和 wall time。原生 API 缺失时不给出性能比例。

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/gdn-recurrent-pipeline -k test_ir
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/gdn-recurrent-pipeline -k test_run
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q -s third_party/ascend/unittest/tla_ut/kernels/gdn-recurrent-pipeline -k test_perf
```

调优时可设置 `TLA_GDN_VALUE_TILE=128` 或 `32`。`TLA_GDN_STORE_STAGE=1` 将写回与更新放在同一 stage，可与默认的 stage 2 独立对比。stage 0 对照跟随同一 value tile。测量记录实际分块和各 stage，状态布局、精度及原地语义一致。

本地已收集 13 项测试，CPU 参考检查通过。初版在 1109 完成四组前端 lowering，CANN 9.2.1 链接时缺少 BF16 标量扩展产生的 `_mlir_ciface_vcast_bfloat16_t_to_float_1d_with_mode`。切换至 Vector SIMD `vextf` 后，默认 128 行的两种 head 数与两种 stage 配置都通过 1109 离线编译。随后两种 head 数的 TLA、stage 0 和原生共六项设备精度通过。BV64、写回 stage 2 在 blue2 通过两种 Hv 的精度。3×40 同卡结果为 Hv16：54.2275 / 77.4000 µs（142.73%），Hv32：110.9010 / 145.4610 µs（131.16%）。
