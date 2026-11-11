# GDN Chunk Prefill pipeline

| batch | Q/K heads | V heads | Dk / Dv | token 数 | chunk |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 128 / 128 | 32768 | 64 |
| 1 | 16 | 32 | 128 / 128 | 32768 | 64 |

输入输出是 TND。Q/K 使用同一 head 数，按 `value_head // (Hv / Hk)` 分组。Q/K/V、beta、输出及起止状态为 BF16，g 为 FP32，状态布局是 `[B,Hv,Dv,Dk]`。本版输入为一条由完整 64-token chunk 组成的连续序列，Q/K 已归一化。Ascend C 对照使用 `torch_npu.npu_chunk_gated_delta_rule`，调用前核对本机 CANN 是否提供该入口。 [CANN Chunk Gated Delta Rule](https://github.com/hicann/ops-transformer/blob/master/attention/chunk_gated_delta_rule/docs/aclnnChunkGatedDeltaRule.md) · [vLLM-Ascend GDN](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/gdn.py)

## 分块与调度

设 `c` 为当前块内 `g` 的前缀和，`L[i,j] = beta[i] * dot(k[i], k[j]) * exp(c[i]-c[j])`（`i > j`）。先分别求 `U = (I+L)^-1 (beta*V)` 和 `W = (I+L)^-1 (beta*exp(c)*K)`。块状态在计算中使用 `[Dv,Dk]`，故残差为 `Rᵀ = Uᵀ - S @ Wᵀ`。块输出由 `S @ Q_dᵀ + Rᵀ @ Aᵀ` 给出，下一块状态为 `exp(c[-1]) * S + Rᵀ @ K_d`。其中 `Q_d` 包含 query scale 和 `exp(c)`，`K_d` 包含到块尾的衰减，`A` 是块内带衰减的 causal QK 矩阵。参考 [Flash Linear Attention 的分块公式与实现](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py)。

- PyTorch 在同一设备上准备每块系数。默认 WY 用严格下三角矩阵的幂倍增展开，另有 FLA `solve_tril` 候选。两种前处理共用同一 TLA recurrence。
- TLA 中每个 job 负责一个 value head 的连续 value 行，默认 32 行，候选 64 行。`load_coefficients` 和 `load_values` 在 stage 0 搬运独立块数据。`predict`、`form_residual`、`apply_chunk`、`advance_state` 同处 stage 1，按 chunk 顺序推进状态。FP32 state 在 UB，Cube 读取 BF16 L1 镜像。输出先按 `[chunk,Hv,Dv,64]` 连续落地，再转回 TND。
- 同一源码覆盖两种 head 数和较短精度用例。对照只把计算 stage 设为 0。
- 64 行分块让每个 Vector 子块接收连续 32 行。状态、残差和输出沿同一 value 轴切分，NZ 发布按每子块的独立行区间写入。UB 内相邻 NZ 分组的步长为 528 个 BF16 元素，按 32 字节对齐。每个 value head 的 W、Q、K、A 读取由四份减为两份，每个 Vector 子块显式常驻和循环缓冲约 76.5 KiB，另一个 U 预取槽为 8 KiB。实际缓冲分配以编译结果为准，耗时需上卡测量。

## 优化与编译器诉求

1. WY 倍增每个 chunk 和 value head 执行 6 次 `64×64 @ 64×256` 和 5 次 `64×64 @ 64×64`，共约 760 万次乘加，另有 6 次 RHS 加法。FLA 候选用 16×16 对角求解及 16 次 `16×16 @ 16×16` 合并，随后对拼接的 U/W RHS 做一次 `64×64 @ 64×256`，矩阵乘合计约 111 万次乘加，另有对角块求解的 Vector 运算。这里的算术量只覆盖 WY 求解和 RHS 计算。系数与逆矩阵借 `[1,chunk_count*Hv*64,1,64]` 视图对应 FLA 的 `[B,T,H,64]`，每 64 行还是一个独立系统。FLA 接收严格下三角的 `L` 并返回 FP32 `(I+L)^-1`，U 保持 FP32 直到 recurrence 残差，W 按原接口转为 BF16。后续可融合局部 cumsum、缩放和打包，并分别测量 WY 和 recurrence 的设备耗时。 [FLA 三角解](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/solve_tril.py)
2. 对比 32 和 64 行分块的系数流量，检查 Cube 预取和 Vector 残差转换间的空档。进一步合并同一 Q/K head 的重复读，并评估直接输出 TND 以减少一次转置。
3. 编译器需要清晰呈现跨 chunk 的持久状态依赖、stage 内 Cube↔Vector 的同步、各级缓冲的版本和实占空间。检查能否通过布局变换直接发布 NZ 或同时对 Cube/Vector 保留高精状态，减少手动 pack 和低精度镜像转换。

## 验证

短序列用逐 token FP32 golden，覆盖两种 head 数和四类输入，输出和 final state 各重复三次。32768 token 的精度测试及性能预检用独立的 FP32 三角解 golden，精度通过后才记时。候选侧每次都重算系数并恢复输出 TND，所有设备任务及 wall time 都计入。两种 TLA stage 配置与 CANN 同流轮换，默认三轮各 40 个设备样本，并记录包、编译器、源文件和设备占用。

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/gdn-chunk-pipeline -k test_ir
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/gdn-chunk-pipeline -k test_run
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q -s third_party/ascend/unittest/tla_ut/kernels/gdn-chunk-pipeline -k test_perf
```

给这些命令加 `TLA_GDN_CHUNK_VALUE_TILE=64` 可独立运行候选。同次测试的 pipeline 和 stage 0 使用相同分块，性能报告记录实际 value tile。两种分块的 CPU 切片模拟均通过。

给设备测试加 `TLA_GDN_CHUNK_WY=fla` 可启用 FLA 前处理，默认 `doubling`。该候选按 blue2 上 FLA `bf9ff21e3124` 接口接入，需在运行环境中导入对应的 `fla` 包。性能报告记录实际三角解模块路径及 SHA256，全部设备前处理照常计时。CPU 测试用独立三角解核对两种 head 数的打包、符号、U 精度及完整输出和最终状态。

### Fused WY 前处理

`fla` 与 64 行 value tile 的 blue2 同卡测量，Hv16 为 17.20 ms 对 SGL Ascend C 的 5.77 ms，Hv32 为 34.06 ms 对 11.24 ms。首轮设备 trace 每次记录 46 和 48 项任务。三角逆 kernel 的中位数分别为 4.93 和 9.89 ms，TLA recurrence 为 1.54 和 2.35 ms，末尾输出转置为 0.28 和 0.60 ms。其余时间主要花在 Q/K/V 转置和 FP32 转换、逐元素 gate、拼接 RHS 以及系数打包。

设置 `TLA_GDN_CHUNK_WY=fused` 启用 FLA `bf9ff21e3124` 的融合 KKT 与求逆。两个 Triton kernel 从 BF16 TND 和 FP32 gate 直接写入 recurrence 所需系数。融合版本的 12 项设备精度通过，Hv16 和 Hv32 的耗时对应 SGL 性能的 22.79% 和 22.17%。融合 KKT 与求逆分别用时 22.2 和 45.3 ms，W/U 重算用时 0.75 和 1.46 ms，QD/KD/A 准备用时 0.47 和 0.80 ms。 [FLA 融合 KKT 与求逆](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk_fwd.py)

设置 `TLA_GDN_CHUNK_WY=split` 将求逆拆回 `fla` 路径已测的独立 `solve_tril`。QD/KD/A kernel 复用已经读到的 K 累加 KKT，并把严格下三角 `L` 写为 `[1,tokens*Hv,1,64]`。W/U kernel 直接读取同样布局的 FP32 逆矩阵并写最终系数。拆分版在 blue2 通过 12 项设备精度。补跑的同卡配对记录在 `gdn-resume.cnGcpV/results/`，包含全部设备前处理和输出转置。 [FLA 独立三角解](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/solve_tril.py)

| value heads | split TLA µs | SGL Ascend C µs | TLA 相对性能 |
|---:|---:|---:|---:|
| 16 | 8183.0155 | 5806.5895 | 70.96% |
| 32 | 15483.2435 | 11258.5195 | 72.71% |

首轮独立求逆与清零合计约 4.98 和 10.04 ms。其余任务约 3.20 和 5.44 ms。如果其他任务耗时相近，求逆需要降至约 2.61 和 5.82 ms，才有空间超过同卡 SGL 结果。

`TLA_GDN_CHUNK_WY=split32` 将每个 64×64 系统拆成两个独立的 32×32 对角块。Vector 核用动态连续切片取得 pivot 行并做 FP32 FMA 更新，Cube 核用 `−R⁻¹ L₂₁ D⁻¹` 恢复左下块，`D`、`R` 表示 `I+L` 的两个对角块。复用 split 的前后处理和 packed inverse 布局。设备通过 12 项精度，卡 2 的三轮配对结果在 `gdn-split32-card2.6TOoK8/results/`。Hv16 为 8887.621 µs 对 5844.181 µs（65.76%），Hv32 为 16809.555 µs 对 11276.7135 µs（67.09%）。首轮 Vector 前代用时 5.39 和 10.79 ms，Cube 合并只用 0.28 和 0.57 ms。 [SGL 950 的 32 行分块](https://github.com/sgl-project/sgl-kernel-npu/blob/2026.11.0.post1/csrc/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage1.h)

`TLA_GDN_CHUNK_WY=split16` 将 Vector 前代改为四个 16×16 对角块，再用同一 Cube 核从 16 合并至 32、从 32 合并至 64。每个系统的 Vector 更新元素数由 `2×31×32²` 减至 `4×15×16²`，代价为一次额外的 16 行 Cube 合并和中间缓冲。两种分块保留 FP32 三角解和 W/U 乘法，使用同一 recurrence 和独立精度阈值。CPU 参考在两种 value tile 下各通过 12 项检查。实际目标 `Ascend950PR_9579` 的 1109 离线编译为两种对角 tile 各生成两份二进制，记录在 `gdn-split16-offline.fb4WBK/offline.log`。

`TLA_GDN_CHUNK_DIRECT_TND=1` 让 Cube 的 `apply_chunk` task 用 `NZ2DN + F322BF16` 将输出写入目标 TND view。每个 job 只写入其 head 的 value 行区间，token 维使用 `Hv×Dv` 的跨步。按输出张量大小计算，可省去转置对 128 或 256 MiB 中间结果的整张读取和写回。CPU 地址映射检查覆盖两种 head 数和两种 value tile，1109 为两种 head 数、两种 tile 和两种 stage 生成八份二进制，记录在 `gdn-direct-tnd-offline.Rsvb4t/offline.log`。

设置 `TLA_GDN_CHUNK_WY=split16 TLA_GDN_CHUNK_VALUE_TILE=64 TLA_GDN_CHUNK_DIRECT_TND=1 TLA_GDN_NATIVE=sgl`。两种 head 数的输出和最终状态共 12 项设备精度通过。正式同卡三轮各 40 次配对记录在 `gdn-direct-formal.gclwia/results/`。

| value heads | split16 + direct TND µs | SGL Ascend C µs | TLA 相对性能 |
|---:|---:|---:|---:|
| 16 | 6452.0515 | 5761.7585 | 89.30% |
| 32 | 11949.3745 | 11286.1060 | 94.45% |

| TLA 设备任务中位数（µs） | Hv16 | Hv32 |
|---|---:|---:|
| 块内 gate 前缀和 | 10.6940 | 10.8075 |
| QD/KD/A 和 KKT | 577.6865 | 994.7370 |
| 16×16 对角块求逆 | 2634.8180 | 5261.6915 |
| Cube 合并至 32×32 | 621.9220 | 1246.9085 |
| Cube 合并至 64×64 | 286.0320 | 575.5875 |
| W/U 重算 | 760.8870 | 1474.4040 |
| TLA recurrence 和 TND 输出 | 1559.4585 | 2384.2030 |

对角块求逆约占 TLA 设备耗时的 41% 和 44%。连同两次合并，求逆阶段约占 55% 和 59%。同卡追平 SGL 还需减少约 690 和 663 µs。如果仅优化对角求逆，该核分别需要提速约 26% 和 13% 的当前耗时。下一轮应先根据 Vector trace 查明动态 pivot 切片、FMA 和 UB 读写各占多少时间，并核对编译器能否将连续 pivot 行读取与行更新放入更紧凑的布局。

本地 CPU 验证了两种 head 数。独立三角解与逐 token FP32 参考一致，按算子精度模拟的分块路径也满足误差门槛。默认 32 行配置已用 1109 为两种 head 数和两种 stage 生成四份 NPU 二进制，blue2 上短序列及 32768 token 精度共 8 项通过。CANN 9.1 的四项原生验证在发射前返回 561103，日志显示缺少 `ChunkGatedDeltaRule` 动态 JSON 或设备二进制。隔离 CANN 9.2.0 包含该算子的 Ascend 950 二进制，blue2 当前拒绝加载该隔离环境的 `aicpu_hcomm`。64 行候选在 blue2 通过两种 Hv、长短规格、两种 TLA stage 共 8 项精度。通过隔离 SGL 官方 950 扩展恢复同合同原生对照，设置 `TLA_GDN_NATIVE=sgl`。FLA 三角解 + BV64 的三方 12 项精度通过，正式比值为 Hv16 33.56%、Hv32 33.01%。实际时间和同轮证据见总表。
