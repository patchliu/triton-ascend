# Groupwise W4A16 pipeline

规格 `(M,K,N) = (128,4096,12288)`。激活、scale、输出均为 BF16，权重是沿 N 每八个 signed INT4 装入一个 int32 的 ND `[K,N/8]` 张量。K 分组大小 128，每组每列独立 scale，无 offset 或 bias。两方共享 `npu_convert_weight_to_int4pack` 产生的同一权重，构造输入时另验证 nibble 顺序。预打包在权重加载期完成。反量化仍计入每次调用。 [Ascend C W4A16 接口](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_weight_quant_batchmatmul.md) · [INT4 打包](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_convert_weight_to_int4pack.md)

一个工作项计算由 M/N tile 决定的输出块；主规格为 M128/N512。每次流水迭代处理一组 K。Cube stage 0 搬激活，两个 Vector 子核在 stage 0 分别解各自一半 K 行，按列组 scale 反量化并直接写带 padding 的 NZ。Cube `matmul` 消费这一组，FP32 累加器跨组留在 L0C。输出在流水排空后转 BF16 写回。两种 stage 配置共用 tiling 和代码。

原生接口还支持 950 上的 NZ 常驻权重。首轮的总表单列相同 ND 权重的对照。后续可用同一 NZ 合同对比预排布收益，检查 UB gather 和 shift 是否吃掉 INT4 搬运收益，再比较融合 unpack、scale、ND→NZ 与更宽 N tile。希望编译器提供 packed INT4 解码和带 scale 的布局转换，报告每阶段的 L1/UB 版本、L0C 持久累加寿命及 Cube/Vector 空闲时长。

验证覆盖 normal 双 seed 和有符号 nibble 的所有值。用未打包的独立权重构造 BF16 反量化值，在 FP32 参考中做矩阵乘。精度通过后，同卡轮换三轮各 40 个样本，用同一权重和 scale 对照 Ascend C。设备耗时与 Python 调用耗时分别记录。

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/w4a16-pipeline -k test_ir
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/w4a16-pipeline -k test_run
TLA_TEST_STAGE=run TLA_RUN_PERF=1 pytest -q third_party/ascend/unittest/tla_ut/kernels/w4a16-pipeline -k test_perf
```

首版离线和设备精度通过。复用反量化权重、两次算术移位和扩大 N tile 后，主规格从 156.0350 µs / 36.42% 降到 65.4395 µs / 86.75%。N256 双组 K tile 为 84.36%，单组 N256 为 85.31%。

最终候选在调用入口用同一 packed Tensor 的字节视图，避免 TLA 指针元素位转换。一次 `UNPK4_B8` 把 64 个连续字节展开到 32 位 lanes，移位分别解出偶数和奇数列，再乘各自的 BF16 scale。原始权重、每次调用的反量化、输出合同都一致。全部精度通过，3×40 中位数 56.7755 / 57.0075 µs，为 Ascend C 的 100.41%，两者接近。旧的 word gather 可以用 `TLA_W4A16_UNPACK_BYTES=0` 复现。完整轮次和同轮基线见总表。

最终默认配置复测为 56.3045 / 56.7605 µs（100.81%）。设备耗时两轮均接近，端到端调用时间另行记录。
