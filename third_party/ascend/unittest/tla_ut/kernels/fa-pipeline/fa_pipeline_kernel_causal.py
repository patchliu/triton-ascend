"""FA3 causal attention: B4/H8/S8192/D64 BF16, BM128/BN256.

QK -> online softmax (causal diagonal mask) -> PV -> accumulation use four
semantic task stages. Query-local state stays outside the KV pipeline. Each
query's KV bound is causal; rotation=62 matches the original FA3 dispatcher.
Physical ring slots, multibuffering and synchronization remain compiler-owned.
"""
import hashlib
from pathlib import Path

import torch
import torch_npu
import triton.language as tl
import triton.language.extra.tla as tla
from triton.runtime import driver

B, H, S, D = 4, 8, 8192, 64
BM, BN, CORES = 128, 256, 28
OPTIONS = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)


@tla.jit
def _fa3_causal_pipeline(Q, K, V, O, MAXIMUM, DENOMINATOR, BM: tl.constexpr, BN: tl.constexpr, CORES: tl.constexpr):
    S: tl.constexpr = 8192
    ROTATION: tl.constexpr = 62
    D: tl.constexpr = 64
    VM: tl.constexpr = BM // 2
    PITCH: tl.constexpr = (BN // 2 + 1) * 16
    QUERY_TILES: tl.constexpr = 4 * 8 * S // BM
    tl.static_assert(BM == 128 and BN == 256)
    core = tl.program_id(0)
    num_query_tiles = (QUERY_TILES - core + CORES - 1) // CORES
    for query_tile in tl.range(num_query_tiles):
        tile = core + query_tile * CORES
        head = tile // (S // BM)
        row = (tile % (S // BM) + head * ROTATION) % (S // BM) * BM
        base = head * S * D
        sub = tla.sub_block_id()

        # Query lifetime: Q is read-only inside the pipeline; m/l/acc are
        # single-buffer recurrences: m/l at stage 1 and accumulator at stage 3.
        q = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        maximum = tla.alloc((VM, 1), tl.float32, space="ub")
        denominator = tla.alloc((VM, 1), tl.float32, space="ub")
        accumulator = tla.alloc((VM, D), tl.float32, space="ub")
        output = tla.alloc((VM, D), tl.bfloat16, space="ub")
        max_copies = tla.alloc((VM, 8), tl.float32, space="ub")
        sum_copies = tla.alloc((VM, 8), tl.float32, space="ub")

        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + base + row * D, (BM, D)), q)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                # NORM_B32 stores 64 consecutive lanes, covering all VM rows.
                tla.simd.vsts(maximum[0, 0], tla.simd.vfull(float("-inf"), tl.float32, 64))
                tla.simd.vsts(denominator[0, 0], zero)
                for r in tl.range(VM):
                    tla.simd.vsts(accumulator[r, 0], zero)

        # Stage 0: QK(t), stage 1: softmax(t-1), stage 2: PV(t-2),
        # stage 3: update(t-3). Scratch versions follow their stage lifetimes.
        # Causal work stops after the last key visible to this query tile.
        # Every task shares this bound; the compiler owns short-loop fill/drain.
        blocks = (row + BM + BN - 1) // BN
        for block in tla.pipeline(blocks):
            k = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
            v = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
            p = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
            qk = tla.alloc((BM // 16, BN // 16, 16, 16), tl.float32, space="l0c")
            pv = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space="l0c")
            score = tla.alloc((BN, VM), tl.float32, space="ub")
            packed_p = tla.alloc((VM // 16 * 2, PITCH), tl.bfloat16, space="ub")
            alpha = tla.alloc((VM, 1), tl.float32, space="ub")
            partial = tla.alloc((VM, D), tl.float32, space="ub")

            with tla.scope.cube().task("load_key", stage=0):
                tla.load_gm_to_l1(tla.view(K + base + block * BN * D, (BN, D)), k)

            with tla.scope.cube().task("qk", stage=0):
                # K @ Q.T gives [keys,queries]. COLUMN_SPLIT assigns 64 query
                # columns to each AIV, exactly one query per FP32 SIMD lane.
                tla.cube.mmad(k, q, qk, transpose_b=True, clear_accum=True)
                tla.copy_l0c_to_ub(
                    qk, score,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task("softmax", stage=1):
                # Specialize the math for full history and diagonal blocks.
                # Exactly one complete SIMD region runs per logical KV block.
                # These two cases are unrelated to physical pipeline slots.
                for MASKED in tl.static_range(2):
                    if (block * BN + BN > row) == MASKED:
                        with tla.scope.simd():
                            neg_inf = tla.simd.vfull(float("-inf"), tl.float32, 64)
                            if MASKED:
                                queries = tla.simd.vci((row + sub * VM).to(tl.int32))
                            # Four independent chains; no horizontal query reduction.
                            mx0 = tla.simd.vfull(float("-inf"), tl.float32, 64)
                            mx1 = mx0
                            mx2 = mx0
                            mx3 = mx0
                            for i in tl.range(BN // 4):
                                x0 = tla.simd.vlds(score[i * 4, 0])
                                x1 = tla.simd.vlds(score[i * 4 + 1, 0])
                                x2 = tla.simd.vlds(score[i * 4 + 2, 0])
                                x3 = tla.simd.vlds(score[i * 4 + 3, 0])
                                if MASKED:
                                    valid0 = tla.simd.vcmp(tla.simd.vdup(block * BN + i * 4 + 0, tl.int32), queries,
                                                           op=tla.simd.CMP.LE)
                                    x0 = tla.simd.vsel(valid0, x0, neg_inf)
                                    valid1 = tla.simd.vcmp(tla.simd.vdup(block * BN + i * 4 + 1, tl.int32), queries,
                                                           op=tla.simd.CMP.LE)
                                    x1 = tla.simd.vsel(valid1, x1, neg_inf)
                                    valid2 = tla.simd.vcmp(tla.simd.vdup(block * BN + i * 4 + 2, tl.int32), queries,
                                                           op=tla.simd.CMP.LE)
                                    x2 = tla.simd.vsel(valid2, x2, neg_inf)
                                    valid3 = tla.simd.vcmp(tla.simd.vdup(block * BN + i * 4 + 3, tl.int32), queries,
                                                           op=tla.simd.CMP.LE)
                                    x3 = tla.simd.vsel(valid3, x3, neg_inf)
                                mx0 = tla.simd.vmax(mx0, x0)
                                mx1 = tla.simd.vmax(mx1, x1)
                                mx2 = tla.simd.vmax(mx2, x2)
                                mx3 = tla.simd.vmax(mx3, x3)
                            m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), D**-0.5)
                            old_m = tla.simd.vlds(maximum[0, 0])
                            m = tla.simd.vmax(m, old_m)
                            correction = tla.simd.vexpdif(old_m, m)
                            tla.simd.vsts(alpha[0, 0], correction)
                            tla.simd.vsts(maximum[0, 0], m)
                            sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                            sum1 = sum0
                            sum2 = sum0
                            sum3 = sum0
                            for i in tl.range(BN // 4):
                                x0 = tla.simd.vmuls(tla.simd.vlds(score[i, 0]), D**-0.5)
                                x1 = tla.simd.vmuls(tla.simd.vlds(score[i + BN // 4, 0]), D**-0.5)
                                x2 = tla.simd.vmuls(tla.simd.vlds(score[i + BN // 2, 0]), D**-0.5)
                                x3 = tla.simd.vmuls(tla.simd.vlds(score[i + BN * 3 // 4, 0]), D**-0.5)
                                if MASKED:
                                    valid0 = tla.simd.vcmp(tla.simd.vdup(block * BN + i, tl.int32), queries,
                                                           op=tla.simd.CMP.LE)
                                    x0 = tla.simd.vsel(valid0, x0, neg_inf)
                                    valid1 = tla.simd.vcmp(tla.simd.vdup(block * BN + i + BN // 4, tl.int32), queries,
                                                           op=tla.simd.CMP.LE)
                                    x1 = tla.simd.vsel(valid1, x1, neg_inf)
                                    valid2 = tla.simd.vcmp(tla.simd.vdup(block * BN + i + BN // 2, tl.int32), queries,
                                                           op=tla.simd.CMP.LE)
                                    x2 = tla.simd.vsel(valid2, x2, neg_inf)
                                    valid3 = tla.simd.vcmp(tla.simd.vdup(block * BN + i + BN * 3 // 4, tl.int32),
                                                           queries, op=tla.simd.CMP.LE)
                                    x3 = tla.simd.vsel(valid3, x3, neg_inf)
                                e0 = tla.simd.vexpdif(x0, m)
                                e1 = tla.simd.vexpdif(x1, m)
                                e2 = tla.simd.vexpdif(x2, m)
                                e3 = tla.simd.vexpdif(x3, m)
                                sum0 = tla.simd.vadd(sum0, e0)
                                sum1 = tla.simd.vadd(sum1, e1)
                                sum2 = tla.simd.vadd(sum2, e2)
                                sum3 = tla.simd.vadd(sum3, e3)
                                # Pack keys (i,i+128) and (i+64,i+192) into NZ.
                                pa, unused_a = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                                tla.simd.vtruncf(e2, tl.bfloat16))
                                pb, unused_b = tla.simd.vdintlv(tla.simd.vtruncf(e1, tl.bfloat16),
                                                                tla.simd.vtruncf(e3, tl.bfloat16))
                                tla.simd.vsstb(pa, packed_p[0, i * 16], PITCH)
                                tla.simd.vsstb(pb, packed_p[0, (i + BN // 4) * 16], PITCH)
                            l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                            old_l = tla.simd.vlds(denominator[0, 0])
                            l = tla.simd.vadd(tla.simd.vmul(old_l, correction), l)
                            tla.simd.vsts(denominator[0, 0], l)

            with tla.scope.vector().task("publish_probability", stage=1):
                # Each AIV owns disjoint query rows of P.T. Exclude UB padding;
                # preserve its pitch. PV requires both publications completed.
                p_rows = tla.reshape(p, (BM // 16, BN * 16))
                first = tla.subview(packed_p, (0, 0), (VM // 16, BN // 2 * 16))
                second = tla.subview(packed_p, (VM // 16, 0), (VM // 16, BN // 2 * 16))
                dst_first = tla.subview(p_rows, (sub * (VM // 16), 0), (VM // 16, BN // 2 * 16))
                dst_second = tla.subview(p_rows, (sub * (VM // 16), BN // 2 * 16), (VM // 16, BN // 2 * 16))
                tla.copy_ub_to_l1(first, dst_first)
                tla.copy_ub_to_l1(second, dst_second)

            with tla.scope.cube().task("pv", stage=2):
                tla.load_gm_to_l1(tla.view(V + base + block * BN * D, (BN, D)), v)
                tla.cube.mmad(p, v, pv, transpose_a=True, clear_accum=True)
                tla.copy_l0c_to_ub(
                    pv, partial,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task("accumulate", stage=3):
                with tla.scope.simd():
                    for r in tl.range(VM):
                        contribution = tla.simd.vlds(partial[r, 0])
                        correction = tla.simd.vlds(alpha[r, 0], dist="BRC_B32")
                        previous = tla.simd.vlds(accumulator[r, 0])
                        contribution = tla.simd.vadd(tla.simd.vmul(previous, correction), contribution)
                        tla.simd.vsts(accumulator[r, 0], contribution)

        # The KV pipeline drains before this epilogue. Normal data dependencies
        # also protect state reset and Q reuse on the next query iteration.
        with tla.scope.vector():
            with tla.scope.simd():
                for r in tl.range(VM):
                    den = tla.simd.vlds(denominator[r, 0], dist="BRC_B32")
                    value = tla.simd.vdiv(tla.simd.vlds(accumulator[r, 0]), den)
                    bits = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), "i32")
                    tla.simd.vsts(output[r, 0], bits, dist="PK_B32")
                # Expand FP32 statistics to eight copies with vector interleaves.
                for statistic in tl.static_range(2):
                    if statistic == 0:
                        bits = tla.simd.vbitcast(tla.simd.vlds(maximum[0, 0]), "i32")
                    else:
                        bits = tla.simd.vbitcast(tla.simd.vlds(denominator[0, 0]), "i32")
                    a0, a1 = tla.simd.vintlv(bits, bits)
                    b0, b1 = tla.simd.vintlv(a0, a0)
                    b2, b3 = tla.simd.vintlv(a1, a1)
                    c0, c1 = tla.simd.vintlv(b0, b0)
                    c2, c3 = tla.simd.vintlv(b1, b1)
                    c4, c5 = tla.simd.vintlv(b2, b2)
                    c6, c7 = tla.simd.vintlv(b3, b3)
                    if statistic == 0:
                        copies = max_copies
                    else:
                        copies = sum_copies
                    tla.simd.vsts(copies[0, 0], tla.simd.vbitcast(c0, "f32"))
                    tla.simd.vsts(copies[8, 0], tla.simd.vbitcast(c1, "f32"))
                    tla.simd.vsts(copies[16, 0], tla.simd.vbitcast(c2, "f32"))
                    tla.simd.vsts(copies[24, 0], tla.simd.vbitcast(c3, "f32"))
                    tla.simd.vsts(copies[32, 0], tla.simd.vbitcast(c4, "f32"))
                    tla.simd.vsts(copies[40, 0], tla.simd.vbitcast(c5, "f32"))
                    tla.simd.vsts(copies[48, 0], tla.simd.vbitcast(c6, "f32"))
                    tla.simd.vsts(copies[56, 0], tla.simd.vbitcast(c7, "f32"))
            output_row = row + sub * VM
            tla.store_ub_to_gm(output, tla.view(O + base + output_row * D, (VM, D)))
            stats_offset = (head * S + output_row) * 8
            tla.store_ub_to_gm(max_copies, tla.view(MAXIMUM + stats_offset, (VM, 8)))
            tla.store_ub_to_gm(sum_copies, tla.view(DENOMINATOR + stats_offset, (VM, 8)))


def device_cores(device):
    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, k, v, causal=True):
    if type(causal) is not bool or not causal:
        raise ValueError("This operator implements causal FA3 only")
    if not all(
            tuple(x.shape) == (B, H, S, D) and x.dtype == torch.bfloat16 and x.device.type == "npu"
            and x.device == q.device and x.is_contiguous() for x in (q, k, v)):
        raise ValueError("FA3 requires contiguous BF16 B4/H8/S8192/D64 on the same NPU")
    if device_cores(q.device) != (28, 56):
        raise ValueError("FA3 comparison requires 28 Cube and 56 Vector cores")
    outputs = (torch.empty_like(q), torch.empty((B, H, S, 8), dtype=torch.float32, device=q.device),
               torch.empty((B, H, S, 8), dtype=torch.float32, device=q.device))

    def call():
        call.compiled = _fa3_causal_pipeline[(CORES, )](q, k, v, *outputs, BM=BM, BN=BN, CORES=CORES, **OPTIONS)
        return outputs

    call.outputs = outputs
    call.metadata = dict(case="FA3", variant="causal-task-pipeline", rotation=62, causal=True, query_tile=BM,
                         key_tile=BN, stages=4,
                         recurrence="query-local m/l/acc, softmax at stage 1, accumulator at stage 3", options=OPTIONS,
                         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return call


def forward(q, k, v, causal=True):
    return prepare_call(q, k, v, causal)()
