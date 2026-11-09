# FlashAttention：pipeline + task

一个算子脚本 `fa_kernel.py`、一个验证入口 `test_fa_pipeline.py`，覆盖相邻
`kernels/fa` 的八个规格。复用 `kernels/fa/reference.py`、`testing.py` 和
`kernels/utils.py` 的精度、AscendC 基线及 profiler 口径。

**八个规格已经通过真实 NPU 精度检查和正式三方性能对比。**
2026-11-08，blue2 Ascend950PR、物理卡 3，CANN 9.1.0；使用已锁定的
`git676cbc44` wheel 与 yun 编译器。每个规格的 Task、原始 FA、AscendC 均通过
4 组输入 × 3 次重复的 O/max/sum 精度检查，再进行 3 轮 × 40 个设备样本计时。
AscendC 的实际启动维度也核验为 28 AIC / 56 AIV。

## 输入输出与规格

Q/K/V 是连续的 BF16 BNSD，B=4、H=8，Q/KV 等长；无 dropout、无额外偏置，
缩放 D^-0.5。输出为 BF16 O，以及 FP32 max/sum `[B,H,S,8]` 的全部八份副本。
目标是 Ascend950、28 AIC / 56 AIV；BM=128、BN=256。

| Case | S | D | Causal | Pipeline µs | Original FA µs | AscendC µs | AscendC / Pipeline |
|---|---:|---:|:---:|---:|---:|---:|---:|
| FA1 | 8192 | 128 | True | 1775.577 | 2109.668 | 1694.379 | 0.954x |
| FA2 | 8192 | 128 | False | 3289.095 | 4089.466 | 3350.671 | 1.019x |
| FA3 | 8192 | 64 | True | 1403.535 | 1338.626 | 1271.907 | 0.906x |
| FA4 | 8192 | 64 | False | 2538.700 | 2547.891 | 2254.335 | 0.888x |
| FA5 | 1024 | 128 | True | 54.730 | 44.693 | 39.916 | 0.729x |
| FA6 | 1024 | 128 | False | 76.507 | 72.937 | 59.791 | 0.782x |
| FA7 | 1024 | 64 | True | 41.201 | 31.484 | 34.273 | 0.832x |
| FA8 | 1024 | 64 | False | 56.064 | 45.971 | 44.696 | 0.797x |
本表单位为 µs；加速比为 AscendC/Task，大于 1 表示 Task 在本轮样本中位数更快。
与原始 FA 相比，FA1/FA2 延迟分别降低约 15.8%/19.6%，FA4 接近；FA3 仍慢约
4.8%，四个短序列规格仍慢约 4.9%～30.9%。这套实现尚未在所有规格上达到
手写 FA 的性能。CSV/原始样本均保留，较小的速度差应结合波动判断。

## 运行

在有真实 NPU 的机器上，先加载配套 CANN；从源码树根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PY=/home/l00898962/.conda/envs/tla-lpc/bin/python
ENTRY=third_party/ascend/unittest/tla_ut/kernels/fa_pipeline/test_fa_pipeline.py
# 在已确认空闲、且获得分配的卡上运行；例如既有实验使用的物理卡 3。
export ASCEND_RT_VISIBLE_DEVICES=3
TLA_TEST_STAGE=run "$PY" "$ENTRY" --mode all --output fa-pipeline-results
```

也可以分开运行 `--mode accuracy`、`--mode performance`、`--mode table`。
`--cases FA3 FA8` 可以缩小范围；`--rounds 3 --samples 40` 是默认计时设置。
`--mode smoke --rounds 1 --samples 8` 只做初步检查，其结果不足以导出正式表格。

Pytest 入口分别是 `test_run`、`test_perf`；后者需要将入口内 `RUN_PERF` 改为
`True`，并串行运行，不能使用 pytest-xdist。CLI 的 performance/all 不受这个
pytest 开关限制。解释器必须加载配套的 Triton wheel 和本轮 npuir 编译器；
仅 source CANN 不会自动选择本轮编译器。

精度检查包含两个 normal seed、Q=0 的 uniform 和 Q/K×4 的 peaked，每组重复
三次；Task、原始 FA、AscendC 都与独立分块 FP32 reference 比较。HF32 关闭，
检查 O、max、sum 的形状、类型、有限性和全部副本；归一化最大误差阈值分别为
0.005、0.0001、0.0001，与 `kernels/fa` 一致，不为新实现放宽阈值。

AscendC 使用 `torch_npu.npu_fusion_attention`：BNSD、`inner_precise=0`、
`keep_prob=1`、同一 scale；dense 使用 `sparse_mode=0`，causal 使用
`sparse_mode=2` 和 2048×2048 压缩上三角 bool mask。

性能每轮轮换三种实现的顺序，默认每种实现 3×40 个样本，取全部 device task
耗时之和的中位数，包含算子内部所有核，不包含准备、编译、reference 和拷贝。
原始 Chrome trace、样本、环境和占用快照均保留。额外的 Level2 counters 用于
核验 AscendC 的真实 `Block Num=28`、`Mix Block Num=56`，其诊断耗时不进入
性能表。若实际原生调度使用不同核数，表格导出会拒绝比较，需先确认比较规格。

`--mode table/all` 输出 `performance.csv`、`performance.json`、`performance.md`。
要求全部输入族和重复次数通过、占用检查通过、来源 SHA256 与当前算子完全
一致、AscendC 核数经过实测确认；缺少任一项都会报错。

## 算子表达与当前调度

Q、m/l、输出累加器具有 query 生命周期；K、V、P、score、alpha、partial
在 KV pipeline 内声明，由编译器根据任务依赖和 stage 生命周期分配版本。

| task | 核 | S1024 stage | S8192 stage | 语义 |
|---|---|---:|---:|---|
| load_key | Cube | 0 | 0 | 加载本轮 K |
| qk | Cube | 0 | 1 | QK 和 Fixpipe COLUMN_SPLIT |
| softmax | Vector | 1 | 2 | online softmax、causal mask、P 转 BF16/NZ |
| publish_probability | Vector | 1 | 2 | 将 P 发布到 L1 |
| load_value | Cube | 空 task，编译时消除 | 2 | 提前加载 V |
| pv | Cube | 2 | 3 | PV 和 Fixpipe ROW_SPLIT；短序列也在此加载 V |
| accumulate | Vector | D128=2，D64=3 | D128=3，D64=4 | 修正并累加 O；末轮融合归一化 |

源码没有显式物理槽号、ready/free flags、多缓冲注解或展开后的六相流水。
`SPECIALIZE` 控制首尾数学特化，`COMPACT` 控制短 causal 半块，`PREFETCH` 控制长
序列 K/V 提前加载，均由 shape 决定，
不根据输入值选择实现。保留开关便于做有明确自变量的对照实验。

- 首个 KV block 直接初始化 m/l/acc，末个 block 融合除法及 BF16 输出；单轮
  causal query 同时执行 FIRST/LAST 特化，避免额外清零和累加器往返。
- causal 只遍历当前 query 可见的 KV；完整历史块不执行逐元素 mask，只有
  对角块执行 mask。静态数学变体位于完整 SIMD region 内，避免让 vector
  返回值跨越动态控制流，适配当前 Vector Layout 转换限制。
- S1024 causal 的半块使用 WIDTH=128 的 K/QK/P/V 活跃前缀 view，其余块为
  WIDTH=256。底层根缓冲仍为完整容量，减少的是计算和搬运，不是分配上限。
- 长序列通过 load_key→qk、load_value→pv 的 stage 距离自动生成 K/V 双缓冲；
  短序列保持单缓冲和较少填充阶段。
- 长 causal 使用 head 相关旋转；短 causal 使用相邻 head 的正/反 query 顺序，
  减小核间工作量不均衡。没有在源码里手动展开跨 query 的连续流水。
- `enable_preload=False` 与原始 FA 的全局选项一致；TLA pipeline 降低产生的
  `hivm.preload_pipeline` 仍经过 CreatePreload 转换，不能据此选项推断没有流水。

本轮离线 IR 中的自动缓冲版本：P=2、score=2；D64 的 alpha=3、partial=2，
D128 的 alpha=2、partial=1。PlanMemory 地址范围上界为 D64 UB=221.50 KiB、
L1=短/长序列 208/272 KiB，D128 UB=245.25 KiB、L1=短/长序列
288/416 KiB。这是静态编译信息，不是实测
运行时占用或耗时。D128 改成 accumulate stage=3 的独立实验需要
277.5 KiB UB，超过 248 KiB 限制，完整错误 IR 已保留。

## 方法的收益与限制

收益：算子工程师直接决定 tile、数据布局、向量数学、causal 工作裁剪和任务
先后关系，编译器负责流水填充/排空、多缓冲、槽位轮转和同步。八种 shape
共用一个语义内核，可以保留硬件控制能力，同时减少手工流水状态机。

当前限制：stage 仍是需要结合资源预算选择的低层参数；相同 stage 数不能保证
相同性能。当前 query 之间会排空/重新填充，短 KV 的固定开销可能更突出；P
只有两槽，仍少于原始 FA 的三槽，可能限制重叠；长序列的 K/V 双缓冲虽然已由
IR 确认，本轮长 D128 已快于原始 FA，长 D64 尚未显示一致收益。短 causal
的完整 task 特化会增加分支和 SIMD 函数数量。布局前缀 view 仍较繁琐，当前
wheel 的 buffer 参数 helper 支持及动态 SIMD 返回值转换也限制了代码复用；task
必须是 pipeline 的直接子级，编译期条件也应放在 task 内部。
实测的短序列差距与首尾开销、P 缓冲容量等限制相符；各项损失占比仍需细粒度
流水证据区分，不能仅根据静态 IR 归因。

## 后续改进方向

1. **资源感知的 stage/版本选择。** 输出每个根缓冲的 producer/consumer、版本
   数、UB/L1 预算和同步开销；在 D128 的 248 KiB 约束内联合优化布局与复用，
   而不是直接增加 stage。为 P 的额外 slack 提供编译器管理的策略，不暴露槽位。
2. **编译器跨 query 调度。** 在保持嵌套 query/KV 源码形式的前提下，允许
   编译器跨相邻 query 重叠 Q 加载、上一轮尾部和下一轮头部；尤其关注短 KV。
   不要求算子工程师写四游标或手动连续流状态机。
3. **K/V 提前加载的细分选择。** 已保留长序列独立 load task 和未预加载两版；
   评估搬运距离、额外 L1 版本、填充成本和 P/PV 重叠。先做精度检查，再用
   流水与计时证据选择策略，不以缓冲数增加直接判定性能改善。
4. **边界 task 和布局 API。** 自动生成 FIRST/LAST、masked/unmasked、半块
   变体，简化 NZ 前缀 view 和 GM→L1/Fixpipe 的接口；完善带 buffer 参数的
   JIT helper 与 Vector Layout 控制流支持，减少重复和代码膨胀。
5. **工作分配与诊断。** 用实际每核时长验证长 causal 旋转、短 causal 镜像
   与原生 FA5 打包映射的差别；提供 task 名到最终 IR/事件的稳定映射。区分
   核间尾部不均、流水首尾、计算、搬运、同步后，再决定优化优先级。

## 本轮证据

run-id：`20261108-010123-fa-pipeline-all-c83d`。
主编译机为用户指定的 yun，归档在
`/home/liupengcheng/workspace/npuir/temp/20261108-010123-fa-pipeline-all-c83d`；
Mac 同名副本保留在本次 visualization 工作目录。编译器 SHA256 为
`6c1d57a4b5240f4566d2ae10f429807a4d2702ae9642838bd8b61b197d386170`；
配套 wheel 为 `triton_ascend-3.6.0.dev0+git676cbc44-cp311-cp311-linux_x86_64.whl`。

`runs/device/` 包含完整精度记录、每种实现 120 个原始设备样本和诊断计数器；
`runs/device-ir-replay/FA*/{pipeline,original}/` 是基于真实 blue2 编译输入，在 yun
重放的完整 before/after IR，16 个重放全部通过。

`runs/offline-prefetch-direct/FA*/`、`runs/offline-v3b/FA*/`（未预加载候选）和
`runs/offline-original-fa/FA*/` 包含实际 argv、输入
IR、完整 before/after pass dump 和 save-temps；`analysis/ir-comparison.md`
是静态对照，`analysis/ir-slices/` 是便于文档引用的关键 pass 快照。早期候选与
失败尝试独立保留。没有为本次任务生成新的 msprof simulator 流水图。
