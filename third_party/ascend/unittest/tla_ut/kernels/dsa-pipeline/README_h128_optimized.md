# H128 optimized DSA pipeline

`dsa_kernel_h128_optimized.py` is the H128 operator; `test_dsa_h128_optimized.py`
reuses the existing common DSA precision/performance driver. It supports the
same B1/SQ8192/KV8192/TopK2048/latent512/RoPE64 contract as the H128 baseline,
with BF16 inputs/P/output and FP32 scores, online state and accumulator.

The query reuse tile covers128 heads. Cube processes two64-head partitions,
with KV tile128 and full512-column PV. One gathered KV tile serves both head
partitions. Their tuple entries are different logical head products, not
physical pipeline slots. The compiler still owns versions, fill/drain and sync.

Each selected KV row is viewed as36 contiguous blocks of16 BF16 elements.
A strided GM-to-UB DMA writes those blocks directly into NZ scratch columns,
removing the raw-key scratch and SIMD packing loop. No selected-KV result is
cached across calls. `_pack_kv` remains included in each timed invocation.

## Measured H128 result

2026-11-08, blue2 physical NPU3, Ascend950PR,28AIC/56AIV, CANN9.1.0.
Interleaved same-stream launches,3 rounds x40 samples/implementation,
normal input seed1103, all device tasks included; occupancy checks passed.

| Implementation | Median ms | AscendC time / implementation time |
|---|---:|---:|
| Previous best pipeline | 31.558464 | 91.9% |
| Direct NZ DMA, BM128/BN64 | 29.304686 | 98.9% |
| This head-partition pipeline | 28.917575 | 100.3% |
| Original manual TLA | 27.822214 | 104.2% |
| AscendC | 28.994333 | 100.0% |

The new operator reduces latency by8.4% relative to the previous best pipeline.
Its small difference from AscendC is within the observed variation; treat them
as approximately equal. The median paired per-cycle AscendC/operator ratio is
0.9997 for this kernel. Do not compare absolute timings from different runs.

Full precision:6 input families x2 seeds, all output elements against the
independent FP32 reference, unchanged thresholds. normal and invalid_prefix
were each repeated10 times with bitwise-identical outputs. This result is for
H128; the other head specifications retain their existing implementations.

## Run

After loading CANN and the matching wheel/compiler:

```bash
ENTRY=third_party/ascend/unittest/tla_ut/kernels/dsa-pipeline/test_dsa_h128_optimized.py
TLA_TEST_STAGE=run python "$ENTRY" --mode all --output dsa-h128-optimized-results
```

The CLI also supports `--mode accuracy`, `--mode performance`, `--mode table`
and `--mode smoke`. pytest provides H128 `test_run`/`test_perf`; performance
is opt-in through RUN_PERF and must run serially. The common driver compares
with the original TLA and AscendC and includes packing in its timings.

The measured environment uses wheel git676cbc44 and the isolated compiler
SHA256 b31e9139fcbea795bf8eefccc62f207dd6206c9321ae04e0e6543a92082626c9
(the earlier scalar-load MIX split fix). The frozen environment and complete
IR/results are archived under run20261108-115952-dsa-opt-8aeb in
/home/liupengcheng/workspace/npuir/temp on yun and
/home/l00898962/workspace/npuir/temp on blue2.

The upstream AscendC source was inspected at commit
5adf800c91c9375e44720d38265a6e7fd433900c. It informed the paired-transfer,
ND-staging and64x128 Cube tiling experiments; the installed CANN9.1 baseline
is a separately versioned binary, not a newly built copy of that commit.
