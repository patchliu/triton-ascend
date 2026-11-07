"""Readable task/stage attention for the complete FA8 workload.

B4/H8/S1024/D64, dense, BF16 Q/K/V/P/O, FP32 scores and normalization.
Each pipeline iteration owns an independent 16-row query tile. Processing all
1024 keys together removes online-softmax state and inter-iteration recurrence.
Stage 0: load Q/K, QK. Stage 1: load V, softmax, publish P, PV, output.
No handwritten synchronization, ring selection, prologue or drain.

Requires NPU-IR fixed-stage lowering with automatic C/V synchronization.
Not compiled or device-tested as part of this port.
"""
from pathlib import Path
import hashlib

import torch
import torch_npu
import triton.language as tl
import triton.language.extra.tla as tla
from triton.runtime import driver

# External workload is exactly FA8. These are internal tiles, not smaller inputs.
B, H, S, D = 4, 8, 1024, 64
BM, VM, CORES = 16, 8, 28
QUERY_TILES = B * H * S // BM
PROB_PITCH = VM * 16 + 16
OPTIONS = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)


@tla.jit
def _fa8_pipeline(Q, K, V, O, MAXIMUM, DENOMINATOR,
                  S: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, CORES: tl.constexpr):
    VM: tl.constexpr = BM // 2
    QUERY_TILES: tl.constexpr = 4 * 8 * S // BM
    PROB_PITCH: tl.constexpr = VM * 16 + 16
    core = tl.program_id(0)
    num_query_tiles = (QUERY_TILES - core + CORES - 1) // CORES
    for query_tile in tla.pipeline(num_query_tiles):
        tile = core + query_tile * CORES
        head = tile // (S // BM)
        row = (tile % (S // BM)) * BM
        base = head * S * D
        sub = tla.sub_block_id()

        q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        k_l1 = tla.alloc((D // 16, S // 16, 16, 16), tl.bfloat16, space="l1")
        v_l1 = tla.alloc((D // 16, S // 16, 16, 16), tl.bfloat16, space="l1")
        p_l1 = tla.alloc((S // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        # Independent logical buffers let stage lifetimes determine versions.
        qk_l0c = tla.alloc((S // 16, BM // 16, 16, 16), tl.float32, space="l0c")
        pv_l0c = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space="l0c")
        score = tla.alloc((VM, S), tl.float32, space="ub")
        probability = tla.alloc((S // 16, PROB_PITCH), tl.bfloat16, space="ub")
        maximum = tla.alloc((VM, 1), tl.float32, space="ub")
        denominator = tla.alloc((VM, 1), tl.float32, space="ub")
        numerator = tla.alloc((VM, D), tl.float32, space="ub")
        output = tla.alloc((VM, D), tl.bfloat16, space="ub")
        maximum_copies = tla.alloc((VM, 8), tl.float32, space="ub")
        denominator_copies = tla.alloc((VM, 8), tl.float32, space="ub")

        with tla.scope.cube().task("load_qk", stage=0):
            tla.load_gm_to_l1(tla.view(Q + base + row * D, (BM, D)), q_l1)
            tla.load_gm_to_l1(tla.view(K + base, (S, D)), k_l1)

        with tla.scope.cube().task("qk", stage=0):
            tla.cube.mmad(q_l1, k_l1, qk_l0c, transpose_b=True, clear_accum=True)
            tla.copy_l0c_to_ub(
                qk_l0c, score,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                        sub_block_idx=None))

        with tla.scope.cube().task("load_v", stage=1):
            tla.load_gm_to_l1(tla.view(V + base, (S, D)), v_l1)

        with tla.scope.vector().task("softmax", stage=1):
            with tla.scope.simd():
                for r in tl.range(VM):
                    # First pass: stable maximum over every key in the row.
                    lanes_max = tla.simd.vfull(float("-inf"), tl.float32, 64)
                    for chunk in tl.range(S // 64):
                        x = tla.simd.vmuls(tla.simd.vlds(score[r, chunk * 64]), D**-0.5)
                        lanes_max = tla.simd.vmax(lanes_max, x)
                    m = tla.simd.vreduce_max(lanes_max)
                    tla.simd.vsts(maximum[r, 0], m, dist="ONEPT_B32")
                    # Second pass: FP32 denominator and BF16 unnormalized P.
                    lanes_sum = tla.simd.vfull(0.0, tl.float32, 64)
                    for pair in tl.range(S // 128):
                        x0 = tla.simd.vmuls(tla.simd.vlds(score[r, pair * 128]), D**-0.5)
                        x1 = tla.simd.vmuls(tla.simd.vlds(score[r, pair * 128 + 64]), D**-0.5)
                        e0 = tla.simd.vexpdif(x0, m)
                        e1 = tla.simd.vexpdif(x1, m)
                        lanes_sum = tla.simd.vadd(lanes_sum, tla.simd.vadd(e0, e1))
                        p0 = tla.simd.vtruncf(e0, tl.bfloat16)
                        p1 = tla.simd.vtruncf(e1, tl.bfloat16)
                        packed, unused = tla.simd.vdintlv(p0, p1)
                        tla.simd.vsstb(packed, probability[pair * 8, r * 16], PROB_PITCH)
                    l = tla.simd.vbroadcast(tla.simd.vreduce_sum(lanes_sum))
                    tla.simd.vsts(denominator[r, 0], l, dist="ONEPT_B32")

        with tla.scope.vector().task("publish_probability", stage=1):
            # Exclude staging padding; each AIV writes its own eight Q rows.
            packed_p = tla.subview(probability, (0, 0), (S // 16, VM * 16))
            p_rows = tla.reshape(p_l1, (S // 16, BM * 16))
            destination = tla.subview(p_rows, (0, sub * VM * 16), (S // 16, VM * 16))
            tla.copy_ub_to_l1(packed_p, destination)

        with tla.scope.cube().task("pv", stage=1):
            # Automatic sync must wait for BOTH AIV probability partitions.
            tla.cube.mmad(p_l1, v_l1, pv_l0c, clear_accum=True)
            tla.copy_l0c_to_ub(
                pv_l0c, numerator,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                        sub_block_idx=None))

        with tla.scope.vector().task("output", stage=1):
            with tla.scope.simd():
                for r in tl.range(VM):
                    m = tla.simd.vlds(maximum[r, 0], dist="BRC_B32")
                    l = tla.simd.vlds(denominator[r, 0], dist="BRC_B32")
                    value = tla.simd.vdiv(tla.simd.vlds(numerator[r, 0]), l)
                    bits = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), "i32")
                    tla.simd.vsts(output[r, 0], bits, dist="PK_B32")
                    for replica in tl.static_range(8):
                        tla.simd.vsts(maximum_copies[r, replica], m, dist="ONEPT_B32")
                        tla.simd.vsts(denominator_copies[r, replica], l, dist="ONEPT_B32")
            output_row = row + sub * VM
            tla.store_ub_to_gm(output, tla.view(O + base + output_row * D, (VM, D)))
            stats_offset = (head * S + output_row) * 8
            tla.store_ub_to_gm(maximum_copies, tla.view(MAXIMUM + stats_offset, (VM, 8)))
            tla.store_ub_to_gm(denominator_copies, tla.view(DENOMINATOR + stats_offset, (VM, 8)))


def device_cores(device):
    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, k, v, causal=False):
    if type(causal) is not bool:
        raise TypeError("causal must be bool")
    if causal:
        raise ValueError("This readable pipeline implements dense FA8 only")
    if not all(tuple(x.shape) == (B, H, S, D) and x.dtype == torch.bfloat16 and
               x.device.type == "npu" and x.device == q.device and x.is_contiguous()
               for x in (q, k, v)):
        raise ValueError("FA8 requires contiguous BF16 B4/H8/S1024/D64 inputs on the same NPU")
    if device_cores(q.device) != (28, 56):
        raise ValueError("FA8 comparison requires 28 Cube and 56 Vector cores")
    outputs = (torch.empty_like(q),
               torch.empty((B, H, S, 8), dtype=torch.float32, device=q.device),
               torch.empty((B, H, S, 8), dtype=torch.float32, device=q.device))
    def call():
        call.compiled = _fa8_pipeline[(CORES,)](
            q, k, v, *outputs, S=S, D=D, BM=BM, CORES=CORES, **OPTIONS)
        return outputs
    call.outputs = outputs
    call.metadata = dict(case="FA8", query_tile=BM, keys_per_iteration=S, stages=2,
                         options=OPTIONS,
                         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return call


def forward(q, k, v, causal=False):
    return prepare_call(q, k, v, causal)()
