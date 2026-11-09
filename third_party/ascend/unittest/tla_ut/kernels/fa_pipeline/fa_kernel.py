"""Eight-shape BF16 FlashAttention through semantic pipeline tasks.

The operator controls tiles, layouts, causal work and task stages. The compiler
owns physical buffer versions, pipeline fill/drain and all synchronization.
"""
import hashlib
from pathlib import Path

import triton.language as tl
import triton.language.extra.tla as tla

CASES = {
    f"FA{i + 1}": (s, d, causal)
    for i, (s, d, causal) in enumerate((s, d, c) for s in (8192, 1024) for d in (128, 64) for c in (True, False))
}
BM, BN, CORES = 128, 256, 28
OPTIONS = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)


@tla.jit
def _fa_pipeline(Q, K, V, O, MAXIMUM, DENOMINATOR, S: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, CORES: tl.constexpr, ROTATION: tl.constexpr,
                 ACC_STAGE: tl.constexpr, SPECIALIZE: tl.constexpr, COMPACT: tl.constexpr, PREFETCH: tl.constexpr):
    VM: tl.constexpr = BM // 2
    PITCH: tl.constexpr = (BN // 2 + 1) * 16
    SHIFT: tl.constexpr = 1 if PREFETCH else 0
    QUERY_TILES: tl.constexpr = 4 * 8 * S // BM
    tl.static_assert(BM == 128 and BN == 256)
    tl.static_assert(not PREFETCH or not COMPACT, "Prefetch requires full-width K/V")
    core = tl.program_id(0)
    num_query_tiles = (QUERY_TILES - core + CORES - 1) // CORES
    for query_tile in tl.range(num_query_tiles):
        tile = core + query_tile * CORES
        head = tile // (S // BM)
        row_index = tile % (S // BM)
        if CAUSAL and S == 1024:
            # Pair early/late causal rows across neighboring heads.
            row_index = S // BM - 1 - row_index if head % 2 else row_index
        else:
            row_index = (row_index + head * ROTATION) % (S // BM)
        row = row_index * BM
        base = head * S * D
        sub = tla.sub_block_id()

        # Query lifetime: Q is read-only inside the pipeline; m/l/acc are
        # single-buffer recurrences: m/l at stage 1 and accumulator at ACC_STAGE.
        q = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        maximum = tla.alloc((VM, 1), tl.float32, space="ub")
        denominator = tla.alloc((VM, 1), tl.float32, space="ub")
        accumulator = tla.alloc((VM, D), tl.float32, space="ub")
        output = tla.alloc((VM, D), tl.bfloat16, space="ub")
        max_copies = tla.alloc((VM, 8), tl.float32, space="ub")
        sum_copies = tla.alloc((VM, 8), tl.float32, space="ub")

        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + base + row * D, (BM, D)), q)
        if not SPECIALIZE:
            with tla.scope.vector():
                with tla.scope.simd():
                    zero = tla.simd.vfull(0.0, tl.float32, 64)
                    # NORM_B32 stores 64 consecutive lanes, covering all VM rows.
                    tla.simd.vsts(maximum[0, 0], tla.simd.vfull(float("-inf"), tl.float32, 64))
                    tla.simd.vsts(denominator[0, 0], zero)
                    for r in tl.range(VM):
                        for col in tl.static_range(D // 64):
                            tla.simd.vsts(accumulator[r, col * 64], zero)

        # Long queries give K/V their own load tasks one stage before use.
        # Relative compute stages stay QK -> softmax -> PV -> accumulate;
        # buffer versions follow lifetimes rather than explicit ring slots.
        # Causal work stops after the last key visible to this query tile.
        # Every task shares this bound; the compiler owns short-loop fill/drain.
        blocks = (row + BM + BN - 1) // BN if CAUSAL else S // BN
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
                for HALF in tl.static_range(2 if COMPACT else 1):
                    if not COMPACT or (row + BM - block * BN < BN) == HALF:
                        WIDTH: tl.constexpr = BM if HALF else BN
                        key_active_size: tl.constexpr = D * WIDTH
                        key_active_flat = tla.reshape(k, (D * BN, ))
                        key_active_prefix = tla.subview(key_active_flat, (0, ), (key_active_size, ))
                        key_active = tla.reshape(key_active_prefix, (D // 16, WIDTH // 16, 16, 16))
                        tla.load_gm_to_l1(tla.view(K + base + block * BN * D, (WIDTH, D)), key_active)

            with tla.scope.cube().task("qk", stage=SHIFT):
                for HALF in tl.static_range(2 if COMPACT else 1):
                    if not COMPACT or (row + BM - block * BN < BN) == HALF:
                        WIDTH: tl.constexpr = BM if HALF else BN
                        key_active_size: tl.constexpr = D * WIDTH
                        key_active_flat = tla.reshape(k, (D * BN, ))
                        key_active_prefix = tla.subview(key_active_flat, (0, ), (key_active_size, ))
                        key_active = tla.reshape(key_active_prefix, (D // 16, WIDTH // 16, 16, 16))
                        qk_active_size: tl.constexpr = BM * WIDTH
                        qk_active_flat = tla.reshape(qk, (BM * BN, ))
                        qk_active_prefix = tla.subview(qk_active_flat, (0, ), (qk_active_size, ))
                        qk_active = tla.reshape(qk_active_prefix, (BM // 16, WIDTH // 16, 16, 16))
                        score_active = tla.subview(score, (0, 0), (WIDTH, VM))
                        tla.cube.mmad(key_active, q, qk_active, transpose_b=True, clear_accum=True)
                        tla.copy_l0c_to_ub(
                            qk_active, score_active,
                            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                    dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT,
                                                    sub_block_idx=None))

            with tla.scope.vector().task("softmax", stage=1 + SHIFT):
                # Specialize the math for full history and diagonal blocks.
                # Exactly one complete SIMD region runs per logical KV block.
                # These two cases are unrelated to physical pipeline slots.
                for HALF in tl.static_range(2 if COMPACT else 1):
                    if not COMPACT or (row + BM - block * BN < BN) == HALF:
                        WIDTH: tl.constexpr = BM if HALF else BN
                        ACTIVE_PITCH: tl.constexpr = (WIDTH // 2 + 1) * 16
                        packed_active_size: tl.constexpr = (VM // 16 * 2) * ACTIVE_PITCH
                        packed_active_flat = tla.reshape(packed_p, ((VM // 16 * 2) * PITCH, ))
                        packed_active_prefix = tla.subview(packed_active_flat, (0, ), (packed_active_size, ))
                        packed_active = tla.reshape(packed_active_prefix, (VM // 16 * 2, ACTIVE_PITCH))
                        score_active = tla.subview(score, (0, 0), (WIDTH, VM))
                        for FIRST in tl.static_range(2 if SPECIALIZE else 1):
                            for MASKED in tl.static_range(2 if CAUSAL else 1):
                                if (not SPECIALIZE or (block == 0) == FIRST) and (not CAUSAL or
                                                                                  (block * BN + BN > row) == MASKED):
                                    with tla.scope.simd():
                                        neg_inf = tla.simd.vfull(float("-inf"), tl.float32, 64)
                                        if MASKED:
                                            queries = tla.simd.vci((row + sub * VM).to(tl.int32))
                                        # Four independent chains; no horizontal query reduction.
                                        mx0 = tla.simd.vfull(float("-inf"), tl.float32, 64)
                                        mx1 = mx0
                                        mx2 = mx0
                                        mx3 = mx0
                                        for i in tl.range(WIDTH // 4):
                                            x0 = tla.simd.vlds(score_active[i * 4, 0])
                                            x1 = tla.simd.vlds(score_active[i * 4 + 1, 0])
                                            x2 = tla.simd.vlds(score_active[i * 4 + 2, 0])
                                            x3 = tla.simd.vlds(score_active[i * 4 + 3, 0])
                                            if MASKED:
                                                valid0 = tla.simd.vcmp(tla.simd.vdup(block * BN + i * 4 + 0, tl.int32),
                                                                       queries, op=tla.simd.CMP.LE)
                                                x0 = tla.simd.vsel(valid0, x0, neg_inf)
                                                valid1 = tla.simd.vcmp(tla.simd.vdup(block * BN + i * 4 + 1, tl.int32),
                                                                       queries, op=tla.simd.CMP.LE)
                                                x1 = tla.simd.vsel(valid1, x1, neg_inf)
                                                valid2 = tla.simd.vcmp(tla.simd.vdup(block * BN + i * 4 + 2, tl.int32),
                                                                       queries, op=tla.simd.CMP.LE)
                                                x2 = tla.simd.vsel(valid2, x2, neg_inf)
                                                valid3 = tla.simd.vcmp(tla.simd.vdup(block * BN + i * 4 + 3, tl.int32),
                                                                       queries, op=tla.simd.CMP.LE)
                                                x3 = tla.simd.vsel(valid3, x3, neg_inf)
                                            mx0 = tla.simd.vmax(mx0, x0)
                                            mx1 = tla.simd.vmax(mx1, x1)
                                            mx2 = tla.simd.vmax(mx2, x2)
                                            mx3 = tla.simd.vmax(mx3, x3)
                                        m = tla.simd.vmuls(
                                            tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), D**-0.5)
                                        correction = tla.simd.vfull(0.0, tl.float32, 64)
                                        if not FIRST:
                                            old_m = tla.simd.vlds(maximum[0, 0])
                                            m = tla.simd.vmax(m, old_m)
                                            correction = tla.simd.vexpdif(old_m, m)
                                        tla.simd.vsts(alpha[0, 0], correction)
                                        tla.simd.vsts(maximum[0, 0], m)
                                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                                        sum1 = sum0
                                        sum2 = sum0
                                        sum3 = sum0
                                        for i in tl.range(WIDTH // 4):
                                            x0 = tla.simd.vmuls(tla.simd.vlds(score_active[i, 0]), D**-0.5)
                                            x1 = tla.simd.vmuls(tla.simd.vlds(score_active[i + WIDTH // 4, 0]), D**-0.5)
                                            x2 = tla.simd.vmuls(tla.simd.vlds(score_active[i + WIDTH // 2, 0]), D**-0.5)
                                            x3 = tla.simd.vmuls(tla.simd.vlds(score_active[i + WIDTH * 3 // 4, 0]),
                                                                D**-0.5)
                                            if MASKED:
                                                valid0 = tla.simd.vcmp(tla.simd.vdup(block * BN + i, tl.int32), queries,
                                                                       op=tla.simd.CMP.LE)
                                                x0 = tla.simd.vsel(valid0, x0, neg_inf)
                                                valid1 = tla.simd.vcmp(
                                                    tla.simd.vdup(block * BN + i + WIDTH // 4, tl.int32), queries,
                                                    op=tla.simd.CMP.LE)
                                                x1 = tla.simd.vsel(valid1, x1, neg_inf)
                                                valid2 = tla.simd.vcmp(
                                                    tla.simd.vdup(block * BN + i + WIDTH // 2, tl.int32), queries,
                                                    op=tla.simd.CMP.LE)
                                                x2 = tla.simd.vsel(valid2, x2, neg_inf)
                                                valid3 = tla.simd.vcmp(
                                                    tla.simd.vdup(block * BN + i + WIDTH * 3 // 4, tl.int32), queries,
                                                    op=tla.simd.CMP.LE)
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
                                            tla.simd.vsstb(pa, packed_active[0, i * 16], ACTIVE_PITCH)
                                            tla.simd.vsstb(pb, packed_active[0, (i + WIDTH // 4) * 16], ACTIVE_PITCH)
                                        l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                                        if not FIRST:
                                            old_l = tla.simd.vlds(denominator[0, 0])
                                            l = tla.simd.vadd(tla.simd.vmul(old_l, correction), l)
                                        tla.simd.vsts(denominator[0, 0], l)

            with tla.scope.vector().task("publish_probability", stage=1 + SHIFT):
                for HALF in tl.static_range(2 if COMPACT else 1):
                    if not COMPACT or (row + BM - block * BN < BN) == HALF:
                        WIDTH: tl.constexpr = BM if HALF else BN
                        ACTIVE_PITCH: tl.constexpr = (WIDTH // 2 + 1) * 16
                        packed_active_size: tl.constexpr = (VM // 16 * 2) * ACTIVE_PITCH
                        packed_active_flat = tla.reshape(packed_p, ((VM // 16 * 2) * PITCH, ))
                        packed_active_prefix = tla.subview(packed_active_flat, (0, ), (packed_active_size, ))
                        packed_active = tla.reshape(packed_active_prefix, (VM // 16 * 2, ACTIVE_PITCH))
                        probability_size: tl.constexpr = BM * WIDTH
                        probability_flat = tla.reshape(p, (BM * BN, ))
                        probability_prefix = tla.subview(probability_flat, (0, ), (probability_size, ))
                        probability = tla.reshape(probability_prefix, (BM // 16, WIDTH // 16, 16, 16))
                        p_rows = tla.reshape(probability, (BM // 16, WIDTH * 16))
                        first = tla.subview(packed_active, (0, 0), (VM // 16, WIDTH // 2 * 16))
                        second = tla.subview(packed_active, (VM // 16, 0), (VM // 16, WIDTH // 2 * 16))
                        dst_first = tla.subview(p_rows, (sub * (VM // 16), 0), (VM // 16, WIDTH // 2 * 16))
                        dst_second = tla.subview(p_rows, (sub * (VM // 16), WIDTH // 2 * 16),
                                                 (VM // 16, WIDTH // 2 * 16))
                        tla.copy_ub_to_l1(first, dst_first)
                        tla.copy_ub_to_l1(second, dst_second)

            with tla.scope.cube().task("load_value", stage=1 + SHIFT):
                if PREFETCH:
                    tla.load_gm_to_l1(tla.view(V + base + block * BN * D, (BN, D)), v)

            with tla.scope.cube().task("pv", stage=2 + SHIFT):
                for HALF in tl.static_range(2 if COMPACT else 1):
                    if not COMPACT or (row + BM - block * BN < BN) == HALF:
                        WIDTH: tl.constexpr = BM if HALF else BN
                        value_active_size: tl.constexpr = D * WIDTH
                        value_active_flat = tla.reshape(v, (D * BN, ))
                        value_active_prefix = tla.subview(value_active_flat, (0, ), (value_active_size, ))
                        value_active = tla.reshape(value_active_prefix, (D // 16, WIDTH // 16, 16, 16))
                        probability_size: tl.constexpr = BM * WIDTH
                        probability_flat = tla.reshape(p, (BM * BN, ))
                        probability_prefix = tla.subview(probability_flat, (0, ), (probability_size, ))
                        probability = tla.reshape(probability_prefix, (BM // 16, WIDTH // 16, 16, 16))
                        if not PREFETCH:
                            tla.load_gm_to_l1(tla.view(V + base + block * BN * D, (WIDTH, D)), value_active)
                        tla.cube.mmad(probability, value_active, pv, transpose_a=True, clear_accum=True)
                        tla.copy_l0c_to_ub(
                            pv, partial,
                            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task("accumulate", stage=ACC_STAGE + SHIFT):
                # First/last are mathematical boundary cases, not buffer slots.
                # A one-block causal query selects FIRST=LAST=1.
                for FIRST in tl.static_range(2 if SPECIALIZE else 1):
                    for LAST in tl.static_range(2 if SPECIALIZE else 1):
                        if not SPECIALIZE or ((block == 0) == FIRST and (block + 1 == blocks) == LAST):
                            with tla.scope.simd():
                                for r in tl.range(VM):
                                    if not FIRST:
                                        correction = tla.simd.vlds(alpha[r, 0], dist="BRC_B32")
                                    for col in tl.static_range(D // 64):
                                        contribution = tla.simd.vlds(partial[r, col * 64])
                                        if not FIRST:
                                            previous = tla.simd.vlds(accumulator[r, col * 64])
                                            contribution = tla.simd.vadd(tla.simd.vmul(previous, correction),
                                                                         contribution)
                                        if LAST:
                                            den = tla.simd.vlds(denominator[r, 0], dist="BRC_B32")
                                            value = tla.simd.vdiv(contribution, den)
                                            bits = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), "i32")
                                            tla.simd.vsts(output[r, col * 64], bits, dist="PK_B32")
                                        else:
                                            tla.simd.vsts(accumulator[r, col * 64], contribution)

        # The KV pipeline drains before this epilogue. Normal data dependencies
        # also protect state reset and Q reuse on the next query iteration.
        with tla.scope.vector():
            with tla.scope.simd():
                if not SPECIALIZE:
                    for r in tl.range(VM):
                        den = tla.simd.vlds(denominator[r, 0], dist="BRC_B32")
                        for col in tl.static_range(D // 64):
                            value = tla.simd.vdiv(tla.simd.vlds(accumulator[r, col * 64]), den)
                            bits = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), "i32")
                            tla.simd.vsts(output[r, col * 64], bits, dist="PK_B32")
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
    import torch
    from triton.runtime import driver
    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def default_config(s, d, causal):
    """Shape-only hardware policy; no tensor-value-dependent choices."""
    return dict(BM=128, BN=256, CORES=28, ROTATION=(62 if s == 8192 else 0) if causal else 0,
                ACC_STAGE=2 if d == 128 else 3, SPECIALIZE=True, COMPACT=causal and s == 1024, PREFETCH=s == 8192)


def prepare_call(q, k, v, causal=False):
    import torch
    import torch_npu
    if type(causal) is not bool:
        raise TypeError("causal must be bool")
    if q.ndim != 4 or q.device.type != "npu":
        raise ValueError("Expected BNSD tensors on an Ascend NPU")
    if not all(x.shape == q.shape and x.dtype == torch.bfloat16 and x.device == q.device and x.is_contiguous()
               for x in (q, k, v)):
        raise ValueError("Q/K/V must have the same contiguous BF16 shape and device")
    b, h, s, d = q.shape
    if (b, h) != (4, 8) or s not in (1024, 8192) or d not in (64, 128):
        raise ValueError("Supported scope: B4/H8, S1024/8192, D64/128")
    if device_cores(q.device) != (28, 56):
        raise ValueError("This implementation requires 28 AIC and 56 AIV")
    config = default_config(s, d, causal)
    outputs = (torch.empty_like(q), torch.empty((b, h, s, 8), dtype=torch.float32, device=q.device),
               torch.empty((b, h, s, 8), dtype=torch.float32, device=q.device))

    def call():
        call.compiled = _fa_pipeline[(config['CORES'], )](q, k, v, *outputs, S=s, D=d, CAUSAL=causal, **config,
                                                          **OPTIONS)
        return outputs

    call.outputs = outputs
    call.metadata = dict(case=next(name for name, spec in CASES.items() if spec == (s, d, causal)),
                         constants=dict(S=s, D=d, CAUSAL=causal, **config), options=OPTIONS,
                         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return call


def forward(q, k, v, causal=False):
    return prepare_call(q, k, v, causal)()


flash_attention = forward
