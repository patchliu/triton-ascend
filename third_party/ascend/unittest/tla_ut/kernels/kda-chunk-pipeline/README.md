# KDA Chunk Prefill

保留规格为 `(B,Hq,Hkv,Dk/Dv,Sq,Skv) = (1,12,12,128/128,32768,32768)`，计算 chunk 为 64。`Q/K/V` 是连续 BSND BF16，`g` 是逐 key channel 的 FP32 自然对数衰减，`beta` 是每 token/head 的 BF16 系数，初始和最终状态均为 FP32 `[B,H,K,V]`。输入已经完成 Q/K 归一化及 gate、beta 激活。算子返回 BF16 输出和 FP32 最终状态。首版支持 K/V 等维 128、等头数和完整 64-token chunk。

旧 KDA H 记录验证的是 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 的部分仿真 trace，规格为 B1、T8192、H96、K=V128。这里的工作覆盖完整前向调用，TLA 融合历史 H 和后续 O 阶段，输入规格也不同。[Kimi Linear 技术报告](https://arxiv.org/abs/2510.26692) · [FLA 前向实现](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_fwd.py)

## 参考实现与切分

Ascend C 对照用 SGLang `torch.ops.npu.chunk_kda_fwd`。原生 950 实现启用 SSBUF 传递 Cube/Vector 数据。vLLM-Ascend 的单 L0 方案把 gate、块内求解、WY、状态递推和最终输出放在同一物理 kernel，减少中间张量写回和 launch。这里先复用 FLA 的块内 cumsum、Aqk/Akk、三角求解及 WY 变换，避免重复实现已经存在的数值敏感代码。原生与候选均从相同的预激活 gate 开始计时，候选的完整 FLA 前处理也计入。[SGLang Ascend 接口](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py) · [Ascend C 构建配置](https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/CMakeLists.txt) · [单 L0 方案](https://github.com/vllm-project/vllm-ascend/pull/13954)

状态推进和输出在一个 `tla.pipeline` 循环内完成。每个 `(batch, head)` 的 value 维分成两组连续的 64 列，两个工作项独立推进各自状态。每个工作项的 Cube 执行四次矩阵乘，两个 AIV 分摊状态 K 行和 token 行。UB 中的 FP32 状态跨 chunk 保留，L1 BF16 状态只供当前 chunk 使用。

```text
correction = U - W @ S
output     = scale * QG @ S + Aqk @ correction
next_S     = exp2(last_gate)[:, None] * S + KG.T @ correction
```

`Aqk` 包含 scale，`G` 为 FLA 的 chunk-local log2 累积衰减。Stage 0 预取 chunk 的 W、KG、QG、Aqk、U 及末行 gate。状态发布、计算、更新和输出放在同一计算 stage，保留 chunk 间状态依赖。对照版将计算也放在 stage 0。两个版本共享同一实现、tiling 和精度规则。

## 验证与后续优化

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/kda-chunk-pipeline -k test_ir
TLA_TEST_STAGE=compile pytest -q third_party/ascend/unittest/tla_ut/kernels/kda-chunk-pipeline -k test_compile
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/kda-chunk-pipeline -k 'test_run and not test_run_native'
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/kda-chunk-pipeline -k test_run_native
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q -s third_party/ascend/unittest/tla_ut/kernels/kda-chunk-pipeline -k test_perf
```

候选精度需要匹配的 Triton-Ascend、CANN、torch_npu 及带 Ascend 内核的 FLA。原生精度另用 `torch.ops.npu.chunk_kda_fwd` 的注册，两类精度各自可运行。两者都用独立 FP32 逐 token 递推比较每个输出元素和最终状态。192 token 用例覆盖常规 gate、强衰减、零更新和非零初态，32768 token 用例覆盖主规格。每路径重复三次。定长 BSND 对照省略可选的 `cu_seqlens`，避免对原生路径引入变长调度。性能以同输入和精度门槛进行三方轮换，每方三轮各 40 次，记录占用、全接口设备任务中位数、wall time、编译器和原生 schema。[原生定长与变长分派](https://github.com/sgl-project/sgl-kernel-npu/blob/3de19ee0f87b01734f3ab3c19c317bb367238be9/csrc/chunk_kda_fwd/op_host/chunk_kda_fwd.cpp)

### 优化方向与编译器诉求

- 32K 的 FLA 前处理占流水版设备时间约 94%，块内 token 运算和块间三角求解分别约 5.41 ms 和 5.53 ms。优先分析这两个内核的向量化与矩阵乘切分，再尝试把求解结果直接交给 WY 生成，减少 Aqk、Akk 的中间流量。
- 同切分的 H/O scan 在 stage 1 约 799 µs，stage 0 约 664 µs。分离预取阶段引入的开销大于本规格的搬运收益。结合 MTE2 和同步事件检查双缓冲，比较两个 value tile 共享 W、KG、QG、Aqk 和一个工作项处理 128 value 列的开销。
- 编译器需要可追踪的跨 chunk UB 状态依赖、双 AIV 合写 L1 的同步与资源报告，以及 UB 到 NZ 的窄向量打包原语。进一步融合块内求解时，需要在普通循环内表达局部流水并保留跨 chunk 的状态顺序。

本地通过 Python 语法与测试收集。独立 NumPy 公式检查覆盖三个 64-token chunk、非零初态和三种 gate/update，FP32 输出最大差 `2.09e-7`，状态最大差 `6.56e-7`。首版 TLA H/O 的两种 stage 配置通过 1109 离线编译。首轮设备验证四项失败，32K 输出出现 NaN。

在 blue2 的 192-token 诊断中，使用 one-hot QG 和全零 WY、gate，排除了 FLA 前处理。两种 stage 的首个 chunk 输出逐位一致，第二个 chunk 相对 RMS 为 `0.4573`，最终状态最大差为 `0.9765625`。修订版让 AIV 直接广播各自的只读 log2 gate 并计算衰减，消除了同一 gate UB 的先写后读依赖。修订后候选六项通过，包括两项逐位精确的零更新状态保持测试。定长 Ascend C 对照四项通过。

同栈验证使用 blue2 Ascend 950、系统 CANN 9.1、Torch 2.10.0+cpu、torch_npu 2.10.0.post1.dev20260810、Triton-Ascend `ee7cc433` 和 FLA `bf9ff21e3124`。原生注册库来自 SGL Kernel NPU `2026.11.0.post1` 的 950 x86_64 公开包，解包到独立目录后通过 PyTorch dispatcher 加载。精度与计时日志位于 blue2 的 `~/workspace/npuir/temp/20261110-tla-14-validation/kda-native-sgl.dC6NvN/`。[SGL Kernel NPU 官方发布](https://github.com/sgl-project/sgl-kernel-npu/releases/tag/2026.11.0.post1)

### 32K 定长计时

| 实现 | 设备任务总时长中位数 (µs) | 同步 wall time 中位数 (µs) | 相对 Ascend C |
| --- | ---: | ---: | ---: |
| TLA pipeline | 12887.906 | 12945.06 | 164.35% |
| Ascend C | 21181.757 | 21239.54 | 100.00% |
| TLA stage 0 调度诊断 | 12756.572 | 12802.81 | — |

卡 1 的三方共九轮占用检查全部通过。定长性能报告为上述目录下的 `perf-results/prefill-0ucz3rfg/results.json`。
