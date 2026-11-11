# Paged MLA Decode with explicit L0

This implementation keeps the validated 112-token Stream-K schedule. It maps
the global KV tile stream onto the runtime Cube grid and intersects each
partition with the current request's visible length.

The attention kernel follows one loop. At tick t, Cube prefetches KV(t+1),
computes QK(t), then consumes P/V(t-1). The two Vector sub-blocks apply causal
masking, update the FP32 online softmax, and accumulate the two PV bands.

- Three L1 slots hold packed K/V. After QK reads RoPE, P uses that region.
- The first 96 columns of Q stay in L0A. QK consumes five K fragments of
  96/112/128/112/128 columns. QK offsets are measured in 16-column NZ blocks.
- The two 256-column PV bands reuse P in L0A. L0B banks alternate operand
  loads, and L0C banks alternate Cube accumulation and Fixpipe.
- Score and PV-band ready/free flags protect the respective UB buffers.
  Probability flags are indexed by the runtime L1 slot. The final AIV
  barrier protects the FP32 split workspace before the fused reduction.

If the runtime tiling exceeds the fused workspace budget, the host selects
page-aligned splits and the independent FP32 reduction.

## Input and output

| Tensor | Shape | Dtype |
|---|---|---|
| Q | [batch, query length, query heads, 576] | BF16 |
| KV | [physical pages, 128, 576] | BF16 |
| Block table | [batch, max pages] | int32 |
| KV lengths | [batch] | int32 |
| Output | [batch, query length, query heads, 512] | BF16 |
| LSE | [batch, query length, query heads] | FP32 |

The first 512 columns are the shared latent K/V and the last 64 columns are
RoPE K. KV lengths include the query chunk. Query position q sees keys through
`kv_length - query_length + q`. Page entries below each actual length must
refer to valid physical pages. Inputs remain on one NPU. A prepared call
reads the current inputs on every invocation and allocates fresh workspace.

```python
from mla import prepare_call

call = prepare_call(q, kv, block_table, kv_lengths, scale=0.04)
output, lse = call()
```

The benchmark specification is B8, Hq24, Hkv1, Sq4, Skv143360, QK/V 576/512,
page128, scale 0.04, and bottom-right causal. The ragged case uses KV lengths
4/129/260/1031. Both schedules query the available Cube and Vector grid at
runtime. `head_tile` and `split_pages` are tiling controls.

## Run and compare

Use Ascend 950, the tested CANN runtime, PyTorch and torch_npu, and the
matching Triton Ascend and AscendNPU-IR builds. The packaged reproduction
uses CANN 9.1.0, torch 2.7.1, and the compiler/bitcode in the
`flashmla-l0-20261111` toolchain bundle. Both implementations explicitly
select optimized metaop bitcode. The L0 implementation also needs BF16
MMAD L0, runtime ND2NZ row windows, dynamic block flags, and per-kernel
manual synchronization support. The pipeline implementation needs
`pipeline/task` lowering.

Ascend C is built from `ops-transformer` commit
`897efb917ea657c759c4f7ce79a162e4729ebdd4` using CANN 9.2.0.

```bash
bash build.sh --pkg --soc=ascend950 --ops=flash_mla,flash_mla_metadata -j16
bash build.sh --torch_extension --ops=flash_mla,flash_mla_metadata -j8
```

Install the custom vendor and its Python wheel, then load the vendor
environment in the benchmark shell. Use CANNBot DSL 0.7.0 and the
`samples/flash_mla_with_kvcache` directory from commit
`6e5bef20c29aad8bc74c66ae995aeeb01a4789f4`. Point
`CANNBOTDSL_BISHENG_PATH` at the compatible CANN 9.2 compiler when the runtime
uses CANN 9.1. The benchmark copies the sample and adds H24 to the checker
allowlist. It records source and checker hashes. CANNBot's compute source
and tiling come from the pinned sample.

Run from this directory after selecting an idle physical device.

```bash
python -m pytest -q
python benchmark.py --dsl-source /path/to/samples/flash_mla_with_kvcache \
    --output /path/to/new-results
```

Set `--baselines ascendc` to run only the Ascend C comparison, or
`--baselines '' --check-only` for standalone FP32 reference checks.

All compared implementations receive the same packed BF16 Q/KV and shuffled
page table, and produce BF16 O and FP32 LSE. DSL layout adapters are tensor
views. Accuracy precedes timing. The native baselines use the full workload, and
ragged validation uses the FP32 reference. The reference materializes a full FP32
softmax per query row, independent of the kernels' splits. Benchmarks reuse
only input tensors and shape/length metadata across invocations. The device
metric sums every task in a call, including a separate split merge when used.
The wall metric includes allocations and launch gaps. Samples collected while
another process occupied the card are discarded and retained as rejected
evidence. The reported speed ratio is Ascend C device latency divided by
candidate device latency.

[Ascend C source](https://gitcode.com/cann/ops-transformer/tree/897efb917ea657c759c4f7ce79a162e4729ebdd4/attention/flash_mla)
· [CANNBot DSL source](https://gitcode.com/cann/cannbot-dsl/tree/6e5bef20c29aad8bc74c66ae995aeeb01a4789f4/samples/flash_mla_with_kvcache)

## Verified results

On blue2 Ascend950PR_9579, six rounds of 40 samples produced a device median of 818.277 µs and a full-call median of 867.638 µs. The frozen pre-cleanup device median was 818.312 µs in the same comparison. Ascend C measured 824.594 µs and the H24 CANNBot adaptation measured 861.276 µs. See [the results](results.json) and [raw samples](../mla-temp/evidence/blue2/before-after-final/results.json).

The runtime supplies core counts and the target architecture. Offline validation also generated full and ragged ELF objects for Ascend950DT_9581 using the 32-Cube configuration. Device accuracy and timing were measured on 950PR.
