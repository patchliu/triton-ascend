# TLA pipeline 算子验证记录

本轮保留此前排序的第 1–14 项。每个算子目录用自己的 README 固化输入输出、参考实现、调优记录和编译器诉求。此页记录共同环境、验收口径和总表，结构化性能摘要见 [PIPELINE_RESULTS.json](PIPELINE_RESULTS.json)。

## 2026-11-08 FA / DSA 归档复盘

来源为 `20261108-final-fa-dsa-391be9.tar.gz`，SHA256 为 `d89e4322c009c2713888d2577644eb1b2c0f0822f1af9f7daae38a2623ce8523`。`FINAL_SUMMARY.md` 是入口，`VALIDATION.json` 列出资产和源文件校验，`FILE_MANIFEST.json` 与 `SHA256SUMS` 用于回查，`conversation/transcript.md` 有构建和排障记录。

| 项目 | 归档记录 |
|---|---|
| 设备与软件 | 正式轮次主要在 blue2 的物理卡 3，Ascend950PR、28 AIC / 56 AIV、CANN 9.1.0、Torch 2.7.1。yun 用于 x86_64 构建和真实设备编译输入的离线重放。 |
| Triton wheel | `676cbc4498b1e89e321b451089ceda192ad3ac46`，CPython 3.11，wheel SHA256 `57435c04485048ec056f557a2455a7008c3e1b2b5450f609514e1691c01bc616`。 |
| FA 编译器 | `6c1d57a4b5240f4566d2ae10f429807a4d2702ae9642838bd8b61b197d386170`。统一入口 `fa_pipeline/test_fa_pipeline.py`，16 份实际编译输入离线重放通过。 |
| DSA 编译器 | 在基础版本 `b0b20b66e78759be6edf3159c6f105b122b8d7a2` 上隔离修复 scalar load 的 MIX 拆分。单独编译目标对象，复制静态库后重新链接，编译器 SHA256 `b31e9139fcbea795bf8eefccc62f207dd6206c9321ae04e0e6543a92082626c9`。 |
| 二进制留存 | 压缩归档保存源码、IR、结果和来源记录。可安装 wheel、编译工具链、模板 bitcode 需要从远端包或相同版本构建恢复。blue2 的历史 DSA 二进制已复核到相同 SHA256。 |

FA 的八组统一入口都完成了精度与三方性能验证，性能为同轮 Ascend C 的 72.93%–101.87%。DSA 通用流水的 H16/H32 分别为 154.57% 和 140.28%，H64 为 76.32%，H128 通用版为 64.25%。H128 通过共享 gather、提早发布 P 和直接搬到 NZ 逐步改善，最终同轮中位数为 28.9175745 ms，Ascend C 为 28.994333 ms。中位数之比是 100.27%，逐样本配对比值的中位数为 99.97%，因此归档判为持平。按规格取最佳的交接表保留每行自己的同轮基线。

复用以下验证方法。

- 同一公开接口比较相同形状、精度、布局、有效长度、mask、scale、输出和必需的前后处理。DSA 对每次调用都计 KV packing。原生接口与候选的无效索引语义有差异时，分开标注精度覆盖范围。
- 先用独立 FP32 参考检查全部输出。每个算子在 README 固化输入族、随机种子、误差阈值和重复次数。状态类算子同时检查输出和最终状态。性能预检使用计时的同一输入。
- 在同卡同流交错或轮换执行，预热后默认三轮、每轮 40 个样本。记录可见卡映射、占用快照、实际 kernel 列表和分核、二进制与源文件哈希。设备任务时长、端到端 wall time、Level2 计数和部分 simulator trace 单独记录。
- `pipeline 性能 / Ascend C = Ascend C 设备耗时中位数 / pipeline 设备耗时中位数 × 100%`。超过 100% 代表候选中位耗时更短。差异接近波动时同时报告分位数和逐调用配对比值。
- 精度、完整编译、性能和诊断各自有状态。发布表由同源码、通过精度和占用校验的记录生成。测试前后保留输入、实际编译 argv、完整 pass dump 和汇总 CSV/JSON。失败候选的日志保留在对应实验目录。

## 本轮工具链

本地在 `dev/tla-pipeline-1108`，基线 `ee7cc4332af0d9fff53540935d653fc5afc6dabd`。相对归档 wheel 的 `676cbc449`，该提交区间的变更在算子用例目录，前端源码相同。当前工作区的 AscendNPU-IR 子模块有用户已有的指针差异。编译器选择独立记录。

yun 上的 `AscendNPU-IR:tla-pipeline-1109` 已核对到 `ddc12f369e95c6192afe681cdee0d7805f700693`，现有 `bishengir-compile` SHA256 为 `e61f0202eb179d51981d2accef2d8c2269c8240b66f7dd8483c3d65240d981c7`，最高 glibc 符号需求为 2.38。blue2 经 Mac 直连可达，glibc 2.39，CANN 9.1.0，`tla-lpc` 中 Torch 2.7.1 与 torch_npu 2.7.1.post7.dev20260707 可导入，探测时卡 2 空闲。每次上卡前重新检查占用。

出包按 yun 的 `build/BUILD_AND_PACKAGE.md`。在独立目录锁定前端、编译器和模板，CPython 3.11 使用 `python -m pip wheel . --no-build-isolation --no-deps -w dist`，通过 `TRITON_ASCEND_BISHENGIR_PATH` 指向有 `bin/` 和 `lib/` 的 payload。CANN 构建侧与运行侧版本分别留档。运行时校验实际 `triton.__file__` 和编译器路径及哈希。若回退，历史 DSA 修复版单列结果和哈希。

本轮 wheel 已在 yun 用 CANN 9.2.1 构建并隔离安装。产物为 `~/workspace/npuir/temp/20261109-tla-ops-paged-gqa-43df/package/wheels/triton_ascend-3.6.0.dev0+gitee7cc433-cp311-cp311-linux_x86_64.whl`，SHA256 为 `2710be1cb1c884ea7c36e8afc28fead176fe70f383f715f3ce295d53341b2aab`。已核对实际导入目录和包内编译器哈希。yun 构建环境没有 `torch_npu`。离线 smoke 只将设备核数查询替换为 28 AIC / 56 AIV，仓库自带的 `query_local_recurrence` 和 `attention_stages` 都生成了 NPU 二进制。实际编译命令和 IR 分别在同一实验目录的 `analysis/package-offline.log` 和 `ir/package-offline/`。

2026-11-09 的 blue2 主机重启后曾出现 `npu-smi` 超时和 `dcmi module initialize failed`（`-8005`）。2026-11-10 经用户提供的备用地址连通，卡 2 恢复健康并通过空闲检查。指定 wheel 在 blue2 的独立 Python 目录隔离安装，源码、wheel 和包内编译器哈希均已复核。

用于复现断言的 Paged GQA 源文件和测试文件保留在 yun 和 blue2 的 `~/workspace/npuir/temp/20261109-tla-ops-paged-gqa-43df/input-258625e/`。

首轮编译发现 `tla.simd.vsplat` 只接受常量，已改用 `tla.simd.vdup` 广播运行时 mask 边界。修订后，decode、prefill 和不等长尾页的 stage 0/1 共六种组合都完成前端 lowering，均未生成二进制。1109 编译器在 `CrossCoreGSS` 的 `memref.load` 翻译处触发 `CUBE_OR_VECTOR` 核归属断言。替换为 SHA256 已复核的历史 DSA 编译器后，六组也都触发同一断言。日志分别为 yun 实验目录下的 `analysis/paged-gqa-258625e.log` 和 `analysis/paged-gqa-258625e-dsa.log`。用 DSA 编译器重放 decode stage 1 时，同步前 IR 的两个标量 load 中，页表读取已标记为 `CUBE`，位于流水外、供循环边界和 Vector mask 共用的 `LENGTHS[batch]` 缺少单核归属。重放证据在 `ir/paged-gqa-gss-replay/gss-before.log`。

编译通过的修订版内核 SHA256 为 `00ec4c0f266b1a0b6e7a8f37ba750df0ddb846366f80d6eb91e21659b35ed6a9`，测试 SHA256 为 `8d11a29dd4581fc8588874419cc368edf02a6a42fad2d589aa6bdfd9e5a58897`。本版以页表容量确定循环上界，Cube 和 Vector 分别在各自 task 内读取实际长度。短请求的额外迭代复用有效页并屏蔽贡献。yun 的 1109 离线编译六组全通过并生成二进制，日志在 `analysis/paged-gqa-00ec4c0.log`，IR 在 `ir/paged-gqa-00ec4c0/`。本地另检查了 2565 行的页表索引和可见 key 范围。

其余 11 个实现目录的 23 个 Python 文件已在 yun 同步并核对 SHA256。批量实验根目录为 `~/workspace/npuir/temp/20261109-tla-ops-batch-11-1109`。`input.FalPYk/` 保存首轮源码，`analysis/` 保存复跑脚本、每组结果、日志和退出码。修订后的 FlexAttention、FlashMLA/GDN Recurrent/压缩滑窗，以及校正合并核规格后的 Dense MLA 分别在 `attempt-flex.RVnCB9/`、`attempt-fixes.Dtzj9Z/`、`attempt-mla.GyB2aY/`。各修订独立保留输入和产物。编译通过的 Paged GQA 版本已同步到 blue2 的 `input-00ec4c0/`。

在 `ee7cc433` 前端、`ddc12f36` 1109 后端和 CANN 9.2.1 下，最终 12 个实现目录的 44 个离线编译项均生成了 `.npubin`。其中其余 11 个目录的 38 项包含 FP8 gather、Dense MLA split 合并与多个尾页、stage 0/1 配置，保留了逐项二进制哈希。编译侧使用离线 28 AIC / 56 AIV 核数设置。PyTorch 或 FLA 的前处理、设备 TopK、原生对照接口和真实设备上的 JIT、精度及性能另行验证。

## 首批规格与状态

GDN 的 `Hkv` 栏分别列 Hk 和可选 Hv。W4A16 的 `(M,K,N)` 列在额外规格。各行都已有本地实现和测试入口。表中性能取通过同规格、同精度和占用校验的最佳正式轮次，时间单位为 µs。以 Ascend C 时间除以 TLA 时间，最新目标为超过 100%。FlexAttention 的 Mojo A5 Triton 对照单独记录。

14 行主规格的 TLA 路径均已通过设备精度。12 行有可比较的 Ascend C 性能，其中 5 行超过 100%。FlexAttention 的 Ascend C 行对应本表的 52K 多模态 mask 组合。GDN 的两个 Hv 分别验收，表中按原排序各算一行。逐轮配置、设备任务耗时和对照类型见 [原始汇总](PIPELINE_RESULTS.json)。

| # | 算子 | bs | Hq | Hkv | Dq/Dv | Sq | Skv | 额外规格 | 当前验证 | TLA µs | Ascend C µs | 性能比例 |
|---:|---|---:|---:|---|---|---:|---|---|---|---:|---:|---:|
| 1 | [Paged GQA Decode](paged-gqa-pipeline/README.md) | 16 | 32 | 8 | 128 | 1 | 4096 | 页 128 | 双方精度通过，同 stage 调度 | 176.2700 | 173.4880 | 98.42% |
| 2 | [Chunked Paged GQA Prefill](paged-gqa-pipeline/README.md) | 4 | 32 | 8 | 128 | 512 | 8192 | 页 128，右对齐因果 | 双方精度通过；Q128 列式、延后搬 V、deep | 971.8130 | 882.0745 | 90.77% |
| 3 | [GDN Chunk Prefill](gdn-chunk-pipeline/README.md) | 1 | 16 | Hk16，Hv16 或 32 | 128/128 | 32768 | 32768 | 64-token chunk，拆分 WY + TLA | 16 行三角解、直接 TND，双方精度通过 | 6452.0515 / 11949.3745 | 5761.7585 / 11286.1060 | 89.30% / 94.45% |
| 4 | [Dense MLA Decode](mla-temp/original/README.md) | 16 | 128 | 1 | 576/512 | 1 | 32768 | 页 128，split-KV | 双方精度通过；M64，KV 分段优先 | 983.6050 | 742.3560 | 75.47% |
| 5 | [KDA Chunk Prefill](kda-chunk-pipeline/README.md) | 1 | 12 | 12 | 128/128 | 32768 | 32768 | FLA 前处理 + TLA H/O | TLA 6/6、SGL Ascend C 4/4；定长达标 | 12887.9060 | 21181.7570 | **164.35%** |
| 6 | [FlashMLA V4.1 Sparse Decode](flashmla-pipeline/README.md) | 64 | 64 | 1 | 512/512 | 1 | 256+4096 | 双 FP8，选择 256+2048，带 sink 与 LSE | 融合 gather，原生与 TLA 各 4 项精度通过 | 1279.8870 | 92.0240 | 7.19% |
| 7 | [Sliding Window Attention](swa-pipeline/README.md) | 1 | 32 | 16 | 256 | 8192 | 8192 | 因果窗口 1024，含当前位置 | 双方精度通过；Q128 列式、KV128、PV task 搬 V，达标 | 1162.3855 | 1268.8795 | **109.16%** |
| 8 | [Lightning Indexer](lightning-indexer-pipeline/README.md) | 1 | 32 | 1 | 128 | 1024 | 32768 | BF16，TopK 2048 | V2 indices-only，复用 metadata，计入 W 扩宽 | 2508.4145 | 13896.9665 | **554.01%** |
| 9 | [MiniMax M3 Sparse Attention](minimax-sparse-pipeline/README.md) | 1 | 64 | 4 | 128 | 4096 | 32768 | 块 128，Top16 | TLA 精度通过，原生 split-KV 短规格精度失败 | — | — | — |
| 10 | [GDN Recurrent Decode](gdn-recurrent-pipeline/README.md) | 64 | 16 | Hk16，Hv16 或 32 | 128/128 | 1 | 1 | BF16 状态原地更新 | 双方精度通过；BV64 两规格均达标 | 54.2275 / 110.9010 | 77.4000 / 145.4610 | **142.73% / 131.16%** |
| 11 | [Dense MLA 多 token decode](mla-temp/original/README.md) | 4 | 64 | 1 | 576/512 | 8 | 131072 | 与 #4 共用内核 | 双方精度通过，M64、144 页 KV 分段优先 | 3413.4015 | 3280.1105 | 96.10% |
| 12 | [SWA + 压缩 KV Attention](swa-compressed-pipeline/README.md) | 1 | 64 | 1 | 512/512 | 4096 | 32768 原始历史 | C128 HCA，256 压缩 KV，窗口 128 | TLA 主规格和边界通过。原生 host metadata 成功，设备报 507015 | — | — | — |
| 13 | [W4A16 MatMul](w4a16-pipeline/README.md) | — | — | — | — | — | — | `(128,4096,12288)`，ND INT4，G128 | 双方精度通过；M128/N512 字节解包达标 | 56.3045 | 56.7605 | **100.81%** |
| 14 | [FlexAttention Forward](flex-attention-pipeline/README.md) | 1 | 16 | 8 | 128 | 52000 | 52000 | 128×128 packed mask，多模态 | 位图与深流水过精度，FIA 组合对照，同轮 Mojo 103.02% | 32739.1120 | 31862.1640 | 97.32% |

## 距超过 Ascend C 的差距

以当前各自的同轮 Ascend C 中位数为参照，以下耗时是下一轮优化目标；验收仍以新一轮同卡配对结果为准。

| 规格 | 同轮 Ascend C（µs） | 当前 TLA 耗时至少需下降 | 优先验证 |
|---|---:|---:|---|
| Paged GQA Decode | 173.49 | 1.58% | 进一步核对每页的 Cube/Vector 同步和对齐空行成本 |
| FlexAttention 52K | 31862.16 | 2.68% | 进一步核对满块的 QK、PV、Vector 累加重叠 |
| Dense MLA 多 token | 3280.11 | 3.90% | 隔离宽 QK 同 stage 调度与分段参数的影响 |
| Chunked Paged GQA Prefill | 882.07 | 9.24% | 减少 Vector 累加内循环控制开销，分别测算术和同步 |
| Dense MLA 单 token | 742.36 | 24.53% | 同一页复用到不同的 Q head 组，拆分 PV 输出列以控制资源 |
| GDN Chunk Hv16 | 5761.76 | 10.70% | 分别压缩 16 行对角块求解和 W/U 系数准备 |
| GDN Chunk Hv32 | 11286.11 | 5.55% | 与 Hv16 共用 WY 和 recurrence，核对三角解随 head 并发扩展 |
| FlashMLA V4.1 Sparse Decode | 92.02 | 92.81% | 在已融合流水内合并间接 DMA，进一步核对解码与 Cube 等待 |

## 第二轮已验收轮次

| 算子与候选 | TLA µs | Ascend C µs | 比例 | 记录 |
|---|---:|---:|---:|---|
| Paged GQA Decode，按页清零 P 的 compact + 同 stage | 179.1010 | 172.0985 | 96.09% | `paged-resume.PlgrYq/paged-gqa-results/decode-f1tiyltg/` |
| Paged GQA Prefill，KV head 优先派发 | 1006.5780 | 900.2095 | 89.43% | `paged-resume.PlgrYq/paged-gqa-results/prefill-_z27symd/` |
| Paged GQA Decode，常驻 P 缓冲 | 181.4290 | 176.5785 | 97.33% | `resume-sweep.HJA4XR/paged-gqa-results/decode-8db4i4is/` |
| Paged GQA Prefill，四行累加 | 975.7925 | 876.5030 | 89.82% | `resume-sweep.HJA4XR/paged-gqa-results/prefill-_49zb49_/` |
| Paged GQA Decode，双页 | 205.2395 | 173.6920 | 84.63% | `paged-two-pages.02Lz4E/paged-gqa-results/decode-bcszu3vy/` |
| Paged GQA Prefill，QK 与 softmax 均在 stage 1 | 1126.7620 | 909.1155 | 80.68% | `paged-two-pages.02Lz4E/paged-gqa-results/prefill-85mqgz6t/` |
| Paged GQA Decode，完整页免 mask | 181.3360 | 174.8150 | 96.40% | `paged-skip-full.5rwi0W/paged-gqa-results/decode-oct0wr60/` |
| Paged GQA Prefill，Q256 与 BF16 PV 暂存 | 1070.5560 | 874.7235 | 81.71% | `paged-compact-pv.RpBBAD/paged-gqa-results/prefill-n7_hpgyl/` |
| GDN Chunk Hv16，拆分 WY | 8183.0155 | 5806.5895 | 70.96% | `gdn-resume.cnGcpV/results/hv16/` |
| GDN Chunk Hv32，拆分 WY | 15483.2435 | 11258.5195 | 72.71% | `gdn-resume.cnGcpV/results/hv32/` |
| GDN Chunk Hv16，32×32 对角块 | 8887.6210 | 5844.1810 | 65.76% | `gdn-split32-card2.6TOoK8/results/hv16/` |
| GDN Chunk Hv32，32×32 对角块 | 16809.5550 | 11276.7135 | 67.09% | `gdn-split32-card2.6TOoK8/results/hv32/` |
| GDN Chunk Hv16，16 行求解与直接 TND | 6452.0515 | 5761.7585 | 89.30% | `gdn-direct-formal.gclwia/results/hv16/` |
| GDN Chunk Hv32，16 行求解与直接 TND | 11949.3745 | 11286.1060 | 94.45% | `gdn-direct-formal.gclwia/results/hv32/` |
| Dense MLA 单 token，两个串行 M64 query tile | 1213.0610 | 739.3785 | 60.95% | `mla-qgroup.KWfsyJ/` |
| Dense MLA 多 token，两个串行 M64 query tile | 3916.4365 | 3274.5330 | 83.61% | `mla-qgroup.KWfsyJ/` |
| Dense MLA 单 token，宽 QK 与分列 PV | 1098.8170 | 739.7825 | 67.33% | `mla-wide.HAw4Dx/` |
| Dense MLA 多 token，宽 QK 与分列 PV | 3525.7540 | 3278.1915 | 92.98% | `mla-wide.HAw4Dx/` |
| Dense MLA 多 token，宽 QK 同 stage 控制组 | 3284.4230 | 3278.1915 | 99.81% | `mla-wide.HAw4Dx/` |
| Dense MLA 多 token，144 页 KV 分段优先 | 3413.4015 | 3280.1105 | 96.10% | `mla-144-clean.NIAsOS/results/multi-token-u2zl18wp/` |
| FlexAttention 52K，列式 softmax、晚搬 V | 35878.5310 | 33800.4855 Mojo Triton | 94.21% Mojo Triton | `flex-lightning-binary.NdSYZR/flex-attention-results/multimodal-hcnjykvm/` |
| FlexAttention 52K，packed predicate 与深流水 | 32739.1120 | 31862.1640 Ascend C FIA 组合 | 97.32% | `flex-predicate.I88PjI/flex-attention-results/multimodal-qjpvpv4g/` |
| FlexAttention 52K，packed predicate 与深流水 | 32739.1120 | 33727.8345 Mojo Triton | 103.02% Mojo Triton | 同上 |
| FlexAttention 52K，跳过桶内全块前缀的 mask 元数据读取 | 32358.0530 | 31423.1590 Ascend C FIA 组合 | 97.11% | `flex-prefix-final.KYOf3M/flex-attention-results/multimodal-eijocx5i/` |
| FlashMLA，16-token 分缓存 gather 与 early QK | 2152.3890 | 91.5900 | 4.26% | `flashmla-x86-replay.6NTIDy/measure-grid28.YJYl3p/card1-perf.P7PPf5/results/v41-decode-eg570a3p/` |
| FlashMLA，融合 FP8 gather 与 attention | 1279.8870 | 92.0240 | 7.19% | `flashmla-fused.LBU4F5/card2-perf.ujzyp0/results/v41-decode-ri75165c/` |
| Lightning Indexer V2，动态 metadata | 2523.6105 | 13928.1420 | 551.91% | `lightning-v2-perf.NRTFol/device-1.IGWVfs/` |
| Lightning Indexer V2，固定规格复用 metadata | 2508.4145 | 13896.9665 | 554.01% | `lightning-v2-cached.nlnOpY/device-2.WbnBS1/` |

Lightning Indexer 的短行修订在 `lightning-resume.qm4yyE/` 通过 7 项设备检查。再次诊断 CANN 9.1 原生时，显式 Q/K 长度与默认长度的索引精度相同，730/1024 行超过原阈值。安装版 V1 的 Vector 源码在 TopK 排序前把 FP32 分数转换成 BF16。公开源码版本已使用 32 位编码，不能据此推断已安装 V1 二进制的行为。V2 的 950 入口保留 FP32 排序分数，在相同 BF16 Q/K/W、indices-only 合同下四种输入各重复三次，cutoff 差均为 0。固定计划版通过 12 项精度，同轮每调用只计 W 扩宽与原生主核，任务中位数分别为 1.4865 和 13895.5160 µs。动态计划版的 metadata 设备耗时为 43.3570 µs。固定计划的首次准备墙钟耗时单列。已安装 V2 的可选分数输出故障另行记录，indices-only 测试关闭该输出。

FlexAttention 的组合基线按可见区间做原生 FIA，对不相交的部分按 LSE 合并。4D mask 修订过四种小输入、单槽及 52K 主规格精度。正式 3×40 包含设备切片拷贝、所有原生 FIA、合并和输出拼接，TLA 也包含真实 padding 和输出裁剪拷贝。Mojo Triton 是通用 BlockMask 对照，Ascend C 组合只对应当前多模态 mask。

压缩滑窗通过 CANN 9.1 自带的 host 调度器生成原生 metadata，绕过 AICPU metadata 的 21008。主规格进入设备内核后返回 507015，当前未通过原生精度。日志在 `hca-host-plan.TLcTGn/card2-native.B7C5M4/`。

MiniMax 的官方 split-KV 路径可发射，短规格归一化最大误差 0.6062、相对 RMS 0.3821，原精度门槛为 0.005。三种短规格的 CPU CSR 均逐项与独立选块枚举一致。干净重建的设备 CSR 包已包含 950 的两份清单和五个设备二进制，实机待验。

FlashMLA 用原版 C++20 和设备编译所需的隔离 GCC 11 头文件构建了官方 x86 wheel，Torch 2.10 和系统 CANN 9.1 可加载。官方 32 核 grid 与 blue2 实际 28 核不同。调整构建 grid 并保留运行时核数校验后，原生及 TLA 各通过 4 项精度。首轮配对预检读到环境采集留下的 `507899 rtsDeviceGetInfo`。环境采集移到性能段之后，保留设备状态检查，9 组计时全部通过占用校验。TLA gather 约 1946.4 µs，占 90.4%，主流水约 206.0 µs。

融合版把两路 528 B FP8 记录的间接搬运、反量化和 NZ 发布放进同一 `tla.pipeline`，消除 144 MiB BF16 临时张量。保留 O/LSE 合同和全部四组精度，9 组计时占用一致，设备时间下降 40.5%。融合 trace 只记录一项设备任务。官方内核 Gather2 的 `asc_copy_gm2ub_align` 用一次指令搬两个记录，支持动态源差分步长、544 B UB 目标步长和 L2 cache hint。TLA 当前逐 token 搬运。两条记录按地址规范化后，动态 stride 的 `tla.view` 在前端要求正的编译期整数，静态对照生成一条 load 并出二进制。cache 属性探针保留到 TTAdapter，但得到相同目标二进制。证据位于 `flashmla-fused.LBU4F5/gather2-evidence/`。

## 2026-11-10 blue2 Paged GQA 精度

物理卡 2 映射为逻辑卡 0，Ascend950PR_9579，CANN 9.1.0，Torch 2.7.1 和 torch_npu 2.7.1.post7.dev20260707。使用已出包的 `ee7cc433 + ddc12f36`，HF32 关闭。先用两张交换物理顺序的 KV 页验证原生接口的 decode 及右对齐 prefill，再用 3 种规格、4 组输入和 3 个实现各重复 3 次，共 108 次与独立 FP32 参考比较，均通过 `0.005` 的归一化最大误差和相对 RMS 门槛。实际输入族为 normal 双 seed、uniform 和 peaked。

| 规格 | TLA 流水最差归一化最大误差 | TLA 流水最差相对 RMS | 同 stage 0 / Ascend C |
|---|---:|---:|---|
| Paged Decode | 0.00353725 | 0.00227161 | 均通过 |
| Chunked Prefill | 0.00306277 | 0.00229818 | 均通过 |
| Decode 不等长尾页 `[65,128,257,511]` | 0.00343773 | 0.00210117 | 均通过 |

blue2 实验路径为 `~/workspace/npuir/temp/20261109-tla-ops-paged-gqa-43df/precision.PoUFiw/`，保留 `accuracy.jsonl`、`accuracy.log`、`accuracy.exit`、`environment.json`、`cache/` 和 `ir/`。本轮只做精度。

## 2026-11-10 blue2 全量复测

本轮实验根目录为 `~/workspace/npuir/temp/20261110-tla-14-validation/`。原始源码在 `input.eZEZIu/`，全量精度在 `accuracy.2HHF8c/`。Dense MLA M64（3 项）、GDN Recurrent BV64（6 项）和压缩滑窗 CPU FP32 主规格（两项 TLA）在 `recheck.4PtIeZ/` 通过。压缩滑窗把大规格参考及误差归约移到 CPU，两项阈值仍为 0.006。Flex 修订在 `flex-class.SpKZq4/` 通过；必须保留单槽循环的运行时边界。Lightning Indexer 的 `li-topk-diagnostic.4C9INw/diagnostic.log` 记录 TLA 两个版本各 1024/1024 行集合匹配，原生 730/1024 行超过原始精度阈值，最大 cutoff 差 0.806198。KDA 候选失败和 GDN Chunk 原生 561103 的日志均保留。

Paged GQA 首版性能在卡 2 采集，每个实现轮转顺序并测量 3 轮 × 40 次。Decode stage 0 为 182.0830 µs（96.02%），Prefill stage 0 为 2945.2860 µs（29.62%）。首版结果在 `paged-perf.ZKrGFX/results/`；stage 0 独立记录。W4A16 首版正式结果在 `qualified-perf.aPFAnt/w4a16/w4a16-results/nd-groupwise-x__ae4dl/results.json`，stage 0 为 156.0750 µs。卡 0 或卡 1 上占用变化的轮次不计入总表。

| Paged GQA 优化轮次 | TLA µs | 同轮 Ascend C µs | 性能比例 | 采样 |
|---|---:|---:|---:|---|
| Decode 首版流水 | 255.3385 | 174.8465 | 68.48% | 3×40 |
| Decode overlap | 213.3530 | 180.6470 | 84.67% | 1×24 |
| Decode deep | 212.8990 | 173.9590 | 81.71% | 1×24 |
| Decode compact + overlap | 208.5485 | 172.6495 | 82.79% | 3×40 |
| Decode 同 stage | 176.2700 | 173.4880 | 98.42% | 3×40 |
| Prefill Q64 首版流水 | 4367.2540 | 872.5165 | 19.98% | 3×40 |
| Prefill Q64 overlap | 2414.2185 | 871.4265 | 36.10% | 1×24 |
| Prefill Q64 deep | 2369.6145 | 869.9870 | 36.71% | 1×24 |
| Prefill Q128 serial | 2684.2810 | 870.9735 | 32.45% | 1×24 |
| Prefill Q128 overlap | 1563.7470 | 872.1435 | 55.77% | 1×24 |
| Prefill Q128 deep | 1566.6445 | 871.9625 | 55.66% | 1×24 |
| Prefill Q128 单页列式 | 1330.4965 | 880.9595 | 66.21% | 3×40 |
| Prefill Q128 双页列式 | 1559.8485 | 873.6265 | 56.01% | 3×40 |
| Prefill Q256 单页列式 | 1152.7020 | 874.5705 | 75.87% | 3×40 |
| Prefill Q128 列式 PV，含输出转置 | 1364.9610 | 887.0790 | 64.99% | 3×40 |
| Prefill Q256 列式 PV，含输出转置 | 1269.1995 | 876.0780 | 69.03% | 3×40 |
| Prefill Q256 列式，延后搬 V | 1124.5025 | 888.7920 | 79.04% | 3×40 |
| Prefill Q128 列式，延后搬 V | 1034.2785 | 874.1075 | 84.51% | 3×40 |
| Prefill Q128 列式，延后搬 V，deep | 971.8130 | 882.0745 | 90.77% | 3×40 |
| Prefill Q128 列式，延后搬 V，wide | 1092.0540 | 881.8650 | 80.75% | 3×40 |
| Prefill Q128 列式，延后搬 V，early QK | 1753.4015 | 889.2270 | 50.71% | 3×40 |
| Prefill Q128 列式，独立 V stage 1 | 989.0065 | 879.5380 | 88.93% | 3×40 |

筛选数据在 `paged-tune.jtvscv/`。Q128 单页、双页及 Q256 单页正式数据分别在 `paged-column.2M2s1R/paged-gqa-results/prefill-s5y8hrnt/`、`paged-column.4Mo2IJ/paged-gqa-results/prefill-_3n3n7ot/` 和 `paged-column.S2ocNK/paged-gqa-results/prefill-mo8j1kns/`。三轮均完成原有 4 个输入族的精度检查。

| 其他正式轮次（3×40） | TLA µs | 对照 µs | 比例 | 记录 |
|---|---:|---:|---:|---|
| GDN Recurrent Hv16，BV64 | 54.2275 | 77.4000 | 142.73% Ascend C | `card2-tune.furEvS/gdn-bv64/gdn-results/recurrent-hv16-bv64-store2-a4m81od5/` |
| GDN Recurrent Hv32，BV64 | 110.9010 | 145.4610 | 131.16% Ascend C | `card2-tune.furEvS/gdn-bv64/gdn-results/recurrent-hv32-bv64-store2-v45jrwc4/` |
| GDN Chunk Hv16，FLA solve + BV64 | 17195.1770 | 5771.3960 | 33.56% Ascend C | `gdn-fla.nvRkQX/gdn-results/chunk-hv16-bv64-sumbd3dy/` |
| GDN Chunk Hv32，FLA solve + BV64 | 34056.9965 | 11241.0925 | 33.01% Ascend C | `gdn-fla.nvRkQX/gdn-results/chunk-hv32-bv64-d0imtrz0/` |
| GDN Chunk Hv16，全融合 WY + BV64 | 25354.6975 | 5777.4385 | 22.79% Ascend C | `gdn-fused-card2.dJiZ0b/gdn-results/chunk-hv16-bv64-9kxiu3l9/` |
| GDN Chunk Hv32，全融合 WY + BV64 | 50680.4090 | 11233.9120 | 22.17% Ascend C | `gdn-fused-card2.dJiZ0b/gdn-results/chunk-hv32-bv64-hz5ytud0/` |
| KDA 定长，含 FLA 前处理 | 12887.9060 | 21181.7570 | 164.35% Ascend C | `kda-native-sgl.dC6NvN/perf-results/prefill-0ucz3rfg/` |
| Dense MLA 单 token，M64 | 1515.4600 | 740.3780 | 48.86% Ascend C | `card2-tune.furEvS/mla-m64/mla-decode-results/single-token-ugovfdxb/` |
| Dense MLA 多 token，M64 | 4775.4685 | 3264.5970 | 68.36% Ascend C | `card2-tune.furEvS/mla-m64/mla-decode-results/multi-token-8rt15tj4/` |
| Dense MLA 单 token，64 页 split，early QK | 1069.5420 | 739.7735 | 69.17% Ascend C | `card2-refine.GNSQd1/mla-early/mla-decode-results/single-token-2ii67sg8/` |
| Dense MLA 多 token，64 页 split，early QK | 3791.3465 | 3260.0120 | 85.99% Ascend C | `card2-refine.GNSQd1/mla-early/mla-decode-results/multi-token-rs5h5d3n/` |
| Dense MLA 单 token，64 页 split，overlap | 1065.8665 | 740.4050 | 69.47% Ascend C | `card2-refine.GNSQd1/mla-overlap/mla-decode-results/single-token-i57zh7d0/` |
| Dense MLA 多 token，64 页 split，overlap | 3783.8300 | 3259.3195 | 86.14% Ascend C | `card2-refine.GNSQd1/mla-overlap/mla-decode-results/multi-token-9f7748ui/` |
| Dense MLA 单 token，512 列合并 | 988.3450 | 742.9785 | 75.17% Ascend C | `card2-acceptance.nkbKdd/mla-final/mla-decode-results/single-token-_sn5m57s/` |
| Dense MLA 多 token，512 列合并 | 3708.8835 | 3262.4225 | 87.96% Ascend C | `card2-acceptance.nkbKdd/mla-final/mla-decode-results/multi-token-_qtua2wy/` |
| Dense MLA 单 token，576 维单次 QK | 1128.1955 | 741.5715 | 65.73% Ascend C | `paged-value-direct.sb0lnP/mla-rope/mla-decode-results/single-token-d4wrlr3_/` |
| Dense MLA 多 token，576 维单次 QK | 4316.0495 | 3262.4690 | 75.59% Ascend C | `paged-value-direct.sb0lnP/mla-rope/mla-decode-results/multi-token-6zsygdgm/` |
| Dense MLA 单 token，KV 分段优先 | 983.6050 | 742.3560 | 75.47% Ascend C | `mla-kv-major.vQ3G4Y/mla-decode-results/single-token-ix0eykfo/` |
| Dense MLA 多 token，KV 分段优先 | 3484.8450 | 3260.6580 | 93.57% Ascend C | `mla-kv-major.vQ3G4Y/mla-decode-results/multi-token-7urrdfey/` |
| SWA Q64，卡 2 | 3132.7545 | 1258.2175 | 40.16% Ascend C | `card2-tune.furEvS/swa/swa-results/band-1_va5v91/` |
| SWA Q64，卡 1 | 3134.7405 | 1298.4930 | 41.42% Ascend C | `card1-tune.a4r6gu/swa-base/swa-results/band-q66b4e5r/` |
| SWA Q128，完整 tile 免 mask | 2161.2375 | 1289.0475 | 59.64% Ascend C | `card1-refine.pMGicR/swa-128-full/swa-results/band-dv45f51n/` |
| SWA Q128，列式，prefetch | 2012.7985 | 1259.9940 | 62.60% Ascend C | `card2-final-layout.cCkpIS/swa-column/swa-results/band-hlmvth8r/` |
| SWA Q128，列式，overlap | 1444.9035 | 1239.7510 | 85.80% Ascend C | `card2-final-layout.cCkpIS/swa-column-overlap/swa-results/band-0084wpln/` |
| SWA Q128，列式，overlap，复测 | 1444.8840 | 1263.4320 | 87.44% Ascend C | `card2-pv.PWYXsz/swa-overlap/swa-results/band-s5aiz6u7/` |
| SWA Q128，列式，early PV | 1469.3890 | 1266.9030 | 86.22% Ascend C | `card2-pv.PWYXsz/swa-early-pv/swa-results/band-m7pltyo1/` |
| SWA Q128/KV256，延后搬 V | 1408.2465 | 1260.3590 | 89.50% Ascend C | `swa-wide.lc39Tv/kv256-late/swa-results/band-ye_8hms5/` |
| SWA Q128/KV128，延后搬 V | 1162.3855 | 1268.8795 | 109.16% Ascend C | `swa-wide.lc39Tv/kv128-late/swa-results/band-uxil6pb3/` |
| W4A16 M64/N128，首版 | 156.0350 | 56.8265 | 36.42% Ascend C | `qualified-perf.aPFAnt/w4a16/w4a16-results/nd-groupwise-x__ae4dl/` |
| W4A16 M128/N128，首版解包 | 93.4660 | 56.8505 | 60.82% Ascend C | `card2-tune.furEvS/w4a16-m128/w4a16-results/nd-groupwise-m_e8e9xs/` |
| W4A16 M128/N128，两次移位 | 77.0880 | 56.8215 | 73.71% Ascend C | `card1-tune.a4r6gu/w4a16-shifts/w4a16-results/nd-groupwise-tk5rnpok/` |
| W4A16 M128/N256，两次移位 | 66.6350 | 56.8445 | 85.31% Ascend C | `card1-tune.a4r6gu/w4a16-n256/w4a16-results/nd-groupwise-95n2xvp5/` |
| W4A16 M128/N128，交错复制 packed word | 100.7100 | 56.8175 | 56.42% Ascend C | `card1-tune.a4r6gu/w4a16-interleave/w4a16-results/nd-groupwise-vfunpt6z/` |
| W4A16 M128/N512，两次移位 | 65.4395 | 56.7670 | 86.75% Ascend C | `card1-refine.pMGicR/w4a16-512/w4a16-results/nd-groupwise-qfwmwgg2/` |
| W4A16 M128/N256，K 双组 | 67.3825 | 56.8425 | 84.36% Ascend C | `card1-refine.pMGicR/w4a16-g2/w4a16-results/nd-groupwise-1xevhidc/` |
| W4A16 M128/N512，字节解包 | 56.7755 | 57.0075 | 100.41% Ascend C | `card2-final-layout.cCkpIS/w4-byte/w4a16-results/nd-groupwise-45_4l07c/` |
| W4A16 M128/N512，最终默认复测 | 56.3045 | 56.7605 | 100.81% Ascend C | `card2-pv.PWYXsz/w4-final/w4a16-results/nd-groupwise-8fn73m2o/` |
| FlexAttention 52K | 127509.9105 | 33616.3115 | 26.36% Mojo Triton | `card2-tune.furEvS/flex/flex-attention-results/multimodal-7j3e1pzh/` |
| FlexAttention 52K，完整块免 mask | 123029.4690 | 34150.5180 | 27.76% Mojo Triton | `card1-refine.pMGicR/flex-full/flex-attention-results/multimodal-rrpll6yz/` |
| FlexAttention 52K，合并分桶 | 120695.9150 | 34051.4610 | 28.21% Mojo Triton | `flex-buckets.38oZAp/buckets/flex-attention-results/multimodal-aq31qs8f/` |
| FlexAttention 52K，Q128 | 91460.4200 | 34558.5075 | 37.79% Mojo Triton | `flex-buckets.38oZAp/q128/flex-attention-results/multimodal-r54725yp/` |
| FlexAttention 52K，Q128 + 合并分桶 | 88302.2325 | 35059.1675 | 39.70% Mojo Triton | `flex-buckets.38oZAp/combined/flex-attention-results/multimodal-1fqcjszf/` |

任务分解显示 Dense MLA overlap 的单 token 首轮约为 942 µs 主核 + 125 µs 合并，多 token 约为 3631 µs 主核 + 152 µs 合并。512 列合并短测中，调整 partial 为 query-major 无稳定额外收益，split 缩到 32 页也未改善。正式版本保留原 workspace 排布和 64 页 split。Flex 逐块 mask 优化后的首轮包括 19 个内核与真实 padding/copy；合并分桶和 Q128 都通过包含空行的精度。W4A16 首个字节解包正式轮次的设备中位数接近，端到端 wall time 分别约 81.1 / 72.9 µs；总表百分比按约定只用设备任务耗时。

Paged Decode 精度及两次正式结果在 `card2-refine.GNSQd1/paged-{compact,serial}/`。同 stage 调度不手动展开流水，六个任务仍由编译器生成；独立的多 stage 候选尚未达到 95%。列式 PV 的两轮证据在 `card2-pv.PWYXsz/paged-{q128,q256}-pvt/`；公平计入输出转置后较慢，已从默认实现去掉。KDA 修订在 `card2-refine.GNSQd1/kda-scan/`，两个精确零更新状态回归和四组 FP32 对比均通过。FlashMLA、MiniMax 和 GDN Chunk BV64 新候选在 `last-candidates.ffJDwD/`。

已在 `cann92-native.7tmBBZ/` 隔离安装 CANN 9.2 toolkit 和公开 Ascend 950 Ops 包，后者的 conda SHA256 为 `106662601980f0030013b8d73cb55ba1825a6256104b3cd3aa03e5b6974bd56c`，含 GDN Chunk 和 GBSA 的 950 二进制。该环境可导入 PyTorch，但 blue2 初始化 NPU 返回 507033，设备拒绝缺少数字签名的 `aicpu_hcomm.tar.gz`（E30009）。GDN Chunk、MiniMax 和 Lightning Indexer 的 CANN 9.2 精度因此未运行，未更改设备签名策略。CANN 9.1 的 HCA 薄适配器补齐空描述符和压缩余数后，原生设备 AICPU metadata 返回 21008；没有用 TLA 数据代替原生结果。

KDA 和 GDN Chunk 另在 CANN 9.1 上隔离使用 [SGL 2026.11.0.post1 的 950 x86 发布包](https://github.com/sgl-project/sgl-kernel-npu/releases/tag/2026.11.0.post1)、PyTorch 2.10.0+cpu 与 TorchNPU 2.10.0.post1.dev20260810。KDA 同卡共九组占用校验通过；TLA 前处理占约 94%，流水 H/O scan 约 799 µs，同 stage 约 664 µs。表中定长原生入口不传可选的 `cu_seqlens`，避免把变长额外成本计给 Ascend C。GDN Chunk 使用同 dtype、同 shape 的 `npu::chunk_gated_delta_rule`，没有回退到缺二进制的 ACLNN。FLA 版首轮 profile 显示 WY inverse 约 4.93 / 9.89 ms，TLA recurrence 约 1.54 / 2.35 ms，系数准备是主要开销。

GDN 全融合 WY 三方 12 项精度均通过。慢点是 FLA 融合的 KKT/求逆，单任务 22.20 / 45.26 ms。直接写 recurrence 布局的 RHS 重算用时 0.75 / 1.46 ms，QD/KD/A 用时 0.47 / 0.80 ms。拆分版保留直接写布局的系数核，改用独立 `solve_tril`，通过 12 项精度。16 行块三角解加直接 TND 写回在 `gdn-combined.tb41lV/` 通过 12 项设备精度，随后在 `gdn-direct-formal.gclwia/` 完成两种 Hv 的有效配对。相比拆分 WY 原版，设备时间分别下降 21.15% 和 22.83%。

Paged 的独立 V stage 2 通过精度，Dense MLA 的 BF16 partial 工作区通过精度，但没有观察到加速。Dense MLA 多 token 的早期 144 页试测未通过完整占用校验，第二轮取得了正式结果。Dense MLA 单 token 仍用 64 页，两个规格均用 FP32 工作区。

FlashMLA 原生侧另外核对了公开源码和 Torch 2.10 环境。Ascend 构建默认使用 aarch64 SDK 路径；在隔离的 x86 路径适配后，bisheng 仍在本机 GCC 13 标准库头报重定义。设备侧 CANN 9.2 签名问题独立存在。FlashMLA、MiniMax 与 HCA 的候选单方诊断也尝试过采样，卡 0 占用改变导致作废；它们不产生可比比值。

## 各算子的优化和编译器诉求

| 算子 | 已验证或下一步的优化 | 编译器诉求 |
|---|---|---|
| Paged GQA Decode | 减少无效 Q 行的 softmax 和累加；比较同 stage 与细粒度重叠 | 保持有效 GQA 组的 Cube/Vector 分工，报告小 tile 的同步开销 |
| Chunked Paged GQA Prefill | Q128、每 lane 一个 query 已测；双页增加开销，继续检查 GQA 组内 KV 重用 | 对独立 P 分区和更大 KV tile 给出多缓冲与 bank 使用诊断 |
| GDN Chunk | FP32 WY 前处理是主要耗时；验证独立三角解和直接写 recurrence 布局的系数核，优先减少中间拷贝 | 表达块内三角求解、持久状态与跨 task 的精度边界 |
| Dense MLA 两规格 | M64、KV 分段优先和 512 列合并提高复用；下一步让同一 Cube tile 内的多个 token 直接复用 KV 页 | 报告同一 KV buffer 供 QK/PV 两次消费时的 L1 版本；支持带步长的 Q 搬运和 split 合并 |
| KDA Chunk | 先验证持久状态跨 chunk 正确；再减少前处理 GM 数据和扫描开销 | 支持清晰表达跨迭代 state 与向量写后读依赖 |
| FlashMLA | 融合 FP8 gather、scale、NZ 发布，已移除 144 MiB BF16 workspace；后续合并动态索引 DMA | 支持带动态源 stride、对齐 UB 目标与 L2 hint 的间接多 burst 搬运 |
| Sliding Window | Q128 与完整 KV tile 省 mask；再比较列式 softmax | 保持运行时单次循环，报告边界分支对 task 同步的影响 |
| Lightning Indexer | 保留正确的长行打分与 TopK，按 tile 局部筛选可进一步减少 GM | 动态 tile 数下保留 pipeline 域，支持融合并准确处理并列分数 |
| MiniMax M3 Sparse | 分析 K→Q 的 KV 外循环、稀疏索引重排和局部归约 | 支持不规则 worklist、共享页元数据和分块归约 |
| GDN Recurrent | BV64 已超过目标；对照阶段顺序和原地状态写回 | 报告状态排布、L0C/UB buffer 版本和异步 store 的生命周期 |
| SWA + 压缩 KV | 对连续 query 重用压缩页；保留两路独立 mask 与共享分母 | 支持动态指针来源、可见长度与 UB/L1 依赖 |
| W4A16 | M128/N512 与连续字节展开；替换两次 gather 后达标 | 支持 TLA 指针元素位转换；提供带 scale 的 packed INT4 解码与 ND→NZ 合并，报告 L0C 临界容量 |
| FlexAttention | 19 个桶收敛到 3 个，列式 softmax、晚搬 V 和跳过 full block mask | 支持内联接收 TLA buffer 的 SIMD helper，允许编译器识别只在 partial 分支使用的 mask 缓冲 |

## 编译器诉求收集方式

每项诉求关联能重现的 shape、所用编译器、实际 IR 或 profile、期望行为和受影响的算子。归档已确认的共性包括分阶段版本与资源报告、持久状态的依赖合同、带可见内存效果的 gather/排布 helper，以及嵌套局部流水和跨工作项持续调度。具体到新算子时，按实际证据区分实现问题、编译器缺口和调优候选。

| 编译器诉求 | 涉及算子 | 验证状态 |
|---|---|---|
| 对共同循环界的 GM 标量读取推断核归属，或允许显式表达两侧副本 | Paged GQA、分页 MLA、FlexAttention、块稀疏 Attention | Paged GQA 已在 1109 和历史 DSA 编译器重现；改为任务内独立读取后离线通过 |
| 报告跨 task 的 L1/UB 版本、L0C 和 UB 持久状态依赖，以及实际分核 | GDN、KDA、Dense MLA、W4A16 | 首版候选，待 pass dump 和设备验证 |
| 将稀疏 gather、解量化和带 padding 的 NZ 写合到同一可分析的 task | FlashMLA、W4A16、SWA + 压缩 KV | 根据原生实现与当前临时缓冲设计提出 |
| 保留或显式降低单次迭代的流水域 | FlexAttention、Sliding Window、Lightning Indexer | 1109 会在后续 pipeline pass 前折叠单次循环，运行时上界已在三者上通过设备验证 |
| 允许 constexpr 启用可选 task，并按依赖安排独立 task | Paged GQA、Sliding Window | 1109 要求 task 直接位于 pipeline 下，且 stage 随源码非递减。已用直接 task 内条件和合法顺序绕过；建议诊断指出实际依赖边 |
| 支持内联 helper 接收 TLA buffer 并在 SIMD 作用域中使用 | FlexAttention、FlashMLA | 1109 child JIT 选择普通 Triton builder，找不到 `create_tla_buffer_type`。两者改为 task 内就地表达后目标编译通过 |
| 给出 stage 版本导致的 UB 峰值及可选切分建议 | Paged GQA Prefill、Dense MLA 宽 QK | Paged Q256 的提前 K 配置要求 2383872 bits，超出 2031616 bits。将 PV 暂存缩为 BF16 后目标编译和设备精度通过，配对性能为 81.71% |
| 保持同地址的指针元素类型转换及字节布局信息 | W4A16、FP8 gather | W4A16 上 `tt.bitcast` 被 Affinity lowering 拒绝；同一权重的 host 字节视图已通过并达标 |
| 支持流水内按边界或有效性跳过工作，减少 full tile 的 mask 成本 | 滑窗、MiniMax M3、FlexAttention | 待验证动态分支的同步和指令开销 |
| 可表达并合并流水化打分与分块 TopK，保持短行索引语义 | Lightning Indexer | 本地首版将 TopK 作为单独设备步骤，全部计入对照 |
| 支持指针 `select` 的 TLA lowering，以及普通 Triton 向量间接指针广播的 rank 处理 | SWA + 压缩 KV、FlashMLA | 1109 已重现；分别改为 task 内分支 DMA 和独立 masked load 后通过 |
| 提供或避免生成缺失的 BF16 标量扩展链接符号 | GDN Recurrent | CANN 9.2.1 已重现；改为 SIMD `vextf` 后通过 |
