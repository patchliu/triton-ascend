# DSA / Sparse MLA：pipeline + task

`dsa_kernel.py` 是算子，`test_dsa_pipeline.py` 是统一的精度、三方性能和报表入口。
复用相邻 `kernels/dsa/test_dsa.py` 的输入生成、独立 FP32 参考和 AscendC API，
不改动原始 DSA 实现。四个规格均已通过完整离线编译及真实 NPU 精度验收：6 类分布 × 2 个种子 × 4 个 heads，
共 48 组输入；另有 normal/invalid_prefix × 4 heads × 两种 TLA 实现各 10 次逐位一致性检查。

## 范围与语义

B=1，SQ=KV=8192，TopK=2048；Q heads 为 16/32/64/128，内容维 512、RoPE 维 64。
输入与原始 DSA 一致：连续 BSND BF16 Q/KV/QR/KR、连续 int32 indices、一个 KV head。
输出为 BF16 `[1,8192,H,512]`，以及 FP32 max/sum `[1,1,8192,H]`。

按 indices 做稀疏 attention，scale=1/24；重复索引保留重复权重，负数及大于等于
8192 的索引屏蔽。每个 query 至少需要一个有效索引。不存在额外 causal mask，
不包含 indexer、TopK 生成、paged cache 或后续投影。

每次调用重新执行 KV/RoPE packing，不缓存选中的 KV、索引结果或 reference。
性能始终包含 `_pack_kv` 和 `_sparse_mla` 两个设备任务。

## 正式性能结果

2026-11-08，blue2 物理卡 3，Ascend950PR、28 AIC / 56 AIV，CANN 9.1.0。
三方使用相同 normal 输入，3 轮 × 40 个样本；以下是全部设备任务时长之和的
中位数，包含 KV packing。实际 AscendC launch 也确认是 28/56。

| H | Pipeline ms | 原始 TLA ms | AscendC ms | AscendC / Pipeline |
|---:|---:|---:|---:|---:|
| 128 | 45.110 | 27.906 | 28.984 | 0.643x |
| 64 | 22.586 | 13.976 | 17.238 | 0.763x |
| 32 | 11.601 | 11.418 | 16.274 | 1.403x |
| 16 | 10.553 | 11.148 | 16.312 | 1.546x |

H16 比原始 TLA 快约 5.3%，H32 接近原始 TLA，两者分别约为 AscendC 的
1.546/1.403 倍速度。H64/H128 仍比原始 TLA 慢约 61.6%，也慢于 AscendC；
当前版本尚未达到这些大 heads 规格的高性能目标。

完整调用时长、120 个设备原始样本、计数器和精度指标均在 results JSON/CSV。
表格只使用当前统一源码的数据，没有将旧候选的较快测量混入当前表。

## Task 表达

| task | 核 | stage | 输入/输出 |
|---|---|---:|---|
| gather | Vector | 0 | indices + packed GM KV → latent/RoPE L1，保存原始索引 |
| qk | Cube | 1 | 两次 MMAD，latent + RoPE 累加到同一 QK L0C，Fixpipe → score |
| softmax | Vector | 2 | 索引 mask、online m/l、FP32 correction、BF16/NZ P |
| publish_probability | Vector | 2 | P 从 UB 发布到 L1 |
| pv | Cube | 2 | P × latent KV，完整 512 维 PV，Fixpipe → partial |
| accumulate | Vector | 2 | FP32 `acc = partial + alpha * acc` |

Q、m/l 和 acc 有 query/head-tile 生命周期；KV、score、P、alpha、partial
声明在 `tla.pipeline(TopK/128)` 内。源码没有物理槽号、buffer version 注解、
ready/free flags、手写首取或最后一轮 PV 排空，也没有四游标状态机。

Gather 领先 QK 一拍，latent 又被后续 PV 使用，因此编译器自动得出：

| 逻辑数据 | 自动版本数 |
|---|---:|
| latent KV | 3 |
| RoPE KV | 2 |
| 原始 indices | 3 |
| score | 2 |
| P / PV partial / correction | 各 1 |

这些数字来自完整 IR，不是源码手工指定。BM64 的 L1 地址范围上界为 504 KiB，
保留 BM64 的同时容纳三份 latent、两份 RoPE、Q 与 P；BM32/BM16 分别为
460/438 KiB。直接将 PV 再推后一拍会延长整份 latent 的生命周期，资源代价很大。

## 可控硬件细节

| H | BM | BN | Gather rows / AIV batch |
|---:|---:|---:|---:|
| 128 | 64 | 128 | 32 |
| 64 | 64 | 128 | 32 |
| 32 | 32 | 128 | 64 |
| 16 | 16 | 128 | 64 |

每个 AIV gather 64 行；普通 group 循环按 GR 分批搬运和打包。小 heads 留出更多
UB，因此只调整形状策略就能增大批量，无需重写 pipeline 或管理两套物理 scratch。
输出 staging 只在 KV pipeline 结束后分配，缩短生命周期，让编译器复用已释放
的 UB 空间，容纳较大的 gather 批量。H128 仍按两个 BM64 作业执行，每次重复 gather，与原始 DSA 的范围一致。

保留原实现的重要细节：int64 load 一次拆两个 int32 索引（仅存储别名）；安全
搬运地址用 `index & 8191`，有效性用 unsigned `<8192` 判断；640 个 BF16 的行
padding 保护 RoPE SIMD load，NZ pitch 保留 `GR+1`；softmax 保留显式
`vsub + vexp`，状态和累加器均为 FP32。

初始最大值使用有限的最小 FP32 数，保证前面若干 KV tile 全部无效时，指数与
correction 仍有定义；不能将这一步直接改为 `-inf` 并计算 `-inf - -inf`。

## 编译器要求与复现

本轮原编译器在 task 内的标量索引 `memref.load` 触发
`SplitMixKernel: unsupported op to get out operands`。已构建一个隔离修复版：
保留 AIV 的真实 scalar load，对移除该操作的 AIC 半边提供类型正确的占位值。
用户现有源码、构建目录和安装包均未替换；旧失败 IR 与补丁、完整构建命令均保留。

- 基础编译器 SHA256：`6c1d57a4b5240f4566d2ae10f429807a4d2702ae9642838bd8b61b197d386170`。
- 本 DSA 实测修复版：`b31e9139fcbea795bf8eefccc62f207dd6206c9321ae04e0e6543a92082626c9`。
- wheel：`git676cbc44`。FA 八规格性能表继续使用基础编译器。

blue2 已部署的独立目录：
`/home/l00898962/workspace/npuir/temp/20261108-082146-dsa-pipeline-c5d8`。

```bash
# run-on-blue2.sh 选择配套 wheel、修复编译器、独立 cache，并检查实际编译器哈希和占用。
cd /home/l00898962/workspace/npuir/temp/20261108-082146-dsa-pipeline-c5d8
bash run-on-blue2.sh --mode all --rounds 3 --samples 40
```

在已经加载对应 CANN/wheel/compiler 的环境中，也可直接使用仓库入口：

```bash
ENTRY=third_party/ascend/unittest/tla_ut/kernels/dsa-pipeline/test_dsa_pipeline.py
TLA_TEST_STAGE=run python "$ENTRY" --mode accuracy
TLA_TEST_STAGE=run python "$ENTRY" --mode performance
python "$ENTRY" --mode table
```

CLI 支持 `--heads 16 32`、`--output PATH`、`--rounds`、`--samples`。
`--mode smoke --rounds 1 --samples 8` 是初测，不能生成正式表。
pytest 同样提供四个 heads 的 `test_run` 和 `test_perf`；后者在入口设置
`RUN_PERF=True`，且必须串行运行。CPU/offline 环境会跳过设备测试。

## 精度与性能口径

精度使用两个 seed、六种输入：normal、uniform、peaked、duplicates、invalid、
invalid_prefix。最后一种包含整段无效的前缀 tile，以及正数越界和有效索引混合，
用于检查最初几轮没有任何有效 key 的情况。检查所有 query/head/输出元素，HF32
关闭，独立分块 FP32 参考；归一化最大误差阈值保持 O=0.005、max/sum=0.0001。

正常四种分布对 Task、原始 TLA、AscendC 都进行检查。两种 invalid 分布检查
Task 和原始 TLA；本机 CANN 的原生 `-1` 语义与屏蔽参考不同，未将其数值混入
正确性结论或放宽阈值。正式性能只使用相同的 normal 输入和 seed=1103。

每种实现 3 轮 × 40 个设备样本，逐轮轮换顺序，取全部任务时长之和的中位数；
包含每次 KV/RoPE packing 和 attention，不含输入、reference 和编译准备。
另行记录无 profiler 的完整 Python 调用时间，包含分配、核数查询和同步。
两组采样独立，不能相减推算 host 开销。

Level2 counters 仅作管线和实际 native launch 的诊断，不混入正式 Level1 时长。
原始样本、Chrome trace、环境、源码哈希及占用快照保留。表格导出要求当前三个
源码/参考哈希、完整精度覆盖、3×40 样本、占用检查和 AscendC 诊断数据均有效。
输出为 `performance.csv`、`performance.json`、`performance.md`。

## 收益、限制和后续方向

收益是数据依赖直接映射到命名 task；原实现的三槽 KV、两槽 RoPE 和复杂 flag
可以由生命周期推导，同时保留 tile、gather 批量、向量数学、布局和 MMAD 控制。
小 heads 的 gather 批量调整只改变配置，没有修改物理槽位或流水展开。

限制是当前不支持嵌套 pipeline：gather 的 GM→UB、SIMD packing、UB→L1
在一个 task 的普通 group 循环内，原来的滚动两批 DMA 重叠未被自动完整恢复。
更细 stage 又会延长整份 KV 生命周期，受到 512 KiB L1 的硬限制。当前
softmax 与 PV 都在 stage 2；将 softmax 和 `publish_probability` 一起提前一拍，
P 会从一份变为两份，BM64 的离线编译确认 L1 需要 520 KiB，超过 512 KiB。
仅提前 softmax、保留概率发布在 stage 2 则可编译：P 仍为一份，prob UB 变为两份，
L1 保持 504 KiB。单独的 H64 normal/seed1103 诊断 A/B（3×40 样本）中，
当前版本为 22.561 ms，该候选为 22.002 ms，时延降低约 2.5%；三方均通过该输入的
独立 FP32 精度检查。此候选未覆盖 H128 和完整输入分布，不替换上面的正式表或算子。
需要联合改变 gather 局部流水、P 发布/释放、调度和存储策略；实际各项损失占比仍需
细粒度流水验证。标量 GM
load 的分核缺口也表明 side effect 和 core ownership 的通用支持还需完善。

后续优先方向：支持 task 内的局部搬运流水或分层 pipeline；按真实生命周期
复用 UB scratch；提供可解释的缓冲预算和 stage 诊断；支持读写效果明确的
buffer helper 与 gather/layout API。H128 的重复 gather 可作为进一步算法
方向，但需要独立验证资源和收益，不应通过缓存输入结果掩盖成本。

## 证据位置

yun 主归档：
`/home/liupengcheng/workspace/npuir/temp/20261108-082146-dsa-pipeline-c5d8`。

`runs/offline-v1/`、`runs/split-debug/` 保存旧编译器崩溃；
`source/compiler-fix/`、`build/`、`package/provenance.json` 保存隔离修复；
`runs/offline-scalar-load-fix/`、`runs/offline-three-stages/`、
`runs/offline-wide-gather/` 保存各版完整 before/after IR 和真实退出码。
blue2 同名目录的 `runs/device/` 保存真实精度、三方 profiling 及计数器。
完整实机数据、编译命令、IR 和性能表已有 Mac 副本；run-id 为
`20261108-082146-dsa-pipeline-c5d8`。最终算子 SHA256 为 `8e47a49b7f463c6aaf13860e6dc9f9651739f2384b6720b84da78fc3590eba62`。
