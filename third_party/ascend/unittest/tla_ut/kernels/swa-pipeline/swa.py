"""Causal sliding-window GQA with tile pruning and exact row masks."""

from functools import partial

import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction


SCHEDULES = {'serial': (0, 0, 0), 'prefetch': (1, 1, 1), 'overlap': (0, 1, 2),
             'early_qk': (0, 1, 1), 'early_pv': (0, 0, 1)}


@partial(TLAJITFunction, do_not_specialize=['kv_tiles'])
def sliding_attention(Q, K, V, O, kv_tiles, B: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, S: tl.constexpr,
                      D: tl.constexpr, WINDOW: tl.constexpr, CORES: tl.constexpr, STAGES: tl.constexpr,
                      QT: tl.constexpr = 64, SKIP_INTERIOR_MASK: tl.constexpr = False,
                      COLUMN_SOFTMAX: tl.constexpr = False, KT: tl.constexpr = 128,
                      LATE_VALUE: tl.constexpr = False):
    ROWS: tl.constexpr = QT // 2
    QTILES: tl.constexpr = S // QT
    KVGROUP: tl.constexpr = HQ // HKV
    tl.static_assert(QT == 64 or QT == 128)
    tl.static_assert(KT == 128 or (COLUMN_SOFTMAX and KT == 256))
    tl.static_assert(S % KT == 0 and S % QT == 0 and D % 128 == 0 and HQ % HKV == 0)
    if COLUMN_SOFTMAX:
        tl.static_assert(ROWS == 64)
        P_NZ: tl.constexpr = (QT // 16, KT // 16, 16, 16)
        SCORE_SHAPE: tl.constexpr = (KT, ROWS)
        PACK_PITCH: tl.constexpr = (KT // 2 + 1) * 16
        PACK_SHAPE: tl.constexpr = (2 * (ROWS // 16), PACK_PITCH)
        QK_SPLIT: tl.constexpr = tla.FixpipeDualDstMode.COLUMN_SPLIT
    else:
        P_NZ: tl.constexpr = (KT // 16, QT // 16, 16, 16)
        SCORE_SHAPE: tl.constexpr = (ROWS, KT)
        PACK_SHAPE: tl.constexpr = (KT // 16, (ROWS + 1) * 16)
        QK_SPLIT: tl.constexpr = tla.FixpipeDualDstMode.ROW_SPLIT

    for job in tl.range(tl.program_id(0), B * HQ * QTILES, CORES):
        head = job // QTILES
        q_start = job % QTILES * QT
        kv_head = head // HQ * HKV + head % HQ // KVGROUP
        q_offset = (head * S + q_start) * D
        sub = tla.sub_block_id()
        first_tile = tl.maximum(q_start - WINDOW + 1, 0) // KT
        # A runtime capacity also keeps the loop when S is only one KV tile.
        end_tile = tl.minimum(tl.cdiv(q_start + QT, KT), kv_tiles)

        query = tla.alloc((D // 16, QT // 16, 16, 16), tl.bfloat16, space='l1')
        maximum = tla.alloc((64,), tl.float32, space='ub')
        denominator = tla.alloc((64,), tl.float32, space='ub')
        accumulator = tla.alloc((ROWS, D), tl.float32, space='ub')
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + q_offset, (QT, D)), query)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vsplat(0.0, tl.float32)
                tla.simd.vsts(maximum[0], tla.simd.vsplat(-3.4028234663852886e38, tl.float32))
                tla.simd.vsts(denominator[0], zero)
                for row in tl.range(ROWS):
                    for col in tl.static_range(D // 64):
                        tla.simd.vsts(accumulator[row, col * 64], zero)

        # Only KV tiles intersecting the union of this Q tile's windows.
        for tile in tla.pipeline(end_tile - first_tile):
            key = tla.alloc((D // 16, KT // 16, 16, 16), tl.bfloat16, space='l1')
            value = tla.alloc((D // 16, KT // 16, 16, 16), tl.bfloat16, space='l1')
            p = tla.alloc(P_NZ, tl.bfloat16, space='l1')
            qk = tla.alloc(P_NZ, tl.float32, space='l0c')
            pv = tla.alloc((D // 16, QT // 16, 16, 16), tl.float32, space='l0c')
            scores = tla.alloc(SCORE_SHAPE, tl.float32, space='ub')
            packed = tla.alloc(PACK_SHAPE, tl.bfloat16, space='ub')
            correction = tla.alloc((64,), tl.float32, space='ub')
            partial = tla.alloc((ROWS, D), tl.float32, space='ub')
            k_start = (first_tile + tile) * KT

            with tla.scope.cube().task('load_kv', stage=0):
                offset = (kv_head * S + k_start) * D
                tla.load_gm_to_l1(tla.view(K + offset, (KT, D)), key)
                if not LATE_VALUE:
                    tla.load_gm_to_l1(tla.view(V + offset, (KT, D)), value)

            with tla.scope.cube().task('qk', stage=STAGES[0]):
                if COLUMN_SOFTMAX:
                    tla.cube.mmad(key, query, qk, transpose_b=True, clear_accum=True)
                else:
                    tla.cube.mmad(query, key, qk, transpose_b=True, clear_accum=True)
                tla.copy_l0c_to_ub(
                    qk, scores,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=QK_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('softmax', stage=STAGES[1]):
                if COLUMN_SOFTMAX:
                    # Each AIV owns a contiguous half of the query columns.
                    # Test the intersection of their windows before masking.
                    first = (q_start + sub * ROWS - k_start).to(tl.int32)
                    if first < KT - 1 or first + ROWS > WINDOW:
                        with tla.scope.simd():
                            queries = tla.simd.vci(first)
                            neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                            for key_id in tl.range(KT):
                                left = tla.simd.vdup(key_id, tl.int32)
                                right = tla.simd.vdup(key_id + WINDOW - 1, tl.int32)
                                valid = tla.simd.pand(tla.simd.vcmp(queries, left, op=tla.simd.CMP.GE),
                                                      tla.simd.vcmp(queries, right, op=tla.simd.CMP.LE))
                                tla.simd.vsts(scores[key_id, 0], tla.simd.vsel(valid, tla.simd.vlds(scores[key_id, 0]), neg))
                    with tla.scope.simd():
                        # Each lane carries one query's online statistics.
                        # Four independent chains keep the vector units fed.
                        mx0 = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx1, mx2, mx3 = mx0, mx0, mx0
                        for i in tl.range(KT // 4):
                            mx0 = tla.simd.vmax(mx0, tla.simd.vlds(scores[4 * i, 0]))
                            mx1 = tla.simd.vmax(mx1, tla.simd.vlds(scores[4 * i + 1, 0]))
                            mx2 = tla.simd.vmax(mx2, tla.simd.vlds(scores[4 * i + 2, 0]))
                            mx3 = tla.simd.vmax(mx3, tla.simd.vlds(scores[4 * i + 3, 0]))
                        old_max = tla.simd.vlds(maximum[0])
                        tile_max = tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3))
                        new_max = tla.simd.vmax(old_max, tla.simd.vmuls(tile_max, D**-0.5))
                        alpha = tla.simd.vexpdif(old_max, new_max)
                        tla.simd.vsts(correction[0], alpha)
                        tla.simd.vsts(maximum[0], new_max)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1, sum2, sum3 = sum0, sum0, sum0
                        for i in tl.range(KT // 4):
                            e0 = tla.simd.vexpdif(tla.simd.vmuls(tla.simd.vlds(scores[i, 0]), D**-0.5), new_max)
                            e1 = tla.simd.vexpdif(
                                tla.simd.vmuls(tla.simd.vlds(scores[i + KT // 4, 0]), D**-0.5), new_max)
                            e2 = tla.simd.vexpdif(
                                tla.simd.vmuls(tla.simd.vlds(scores[i + KT // 2, 0]), D**-0.5), new_max)
                            e3 = tla.simd.vexpdif(
                                tla.simd.vmuls(tla.simd.vlds(scores[i + 3 * KT // 4, 0]), D**-0.5), new_max)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            # Interleave the two key halves into padded P.T.
                            lo, unused_lo = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                             tla.simd.vtruncf(e2, tl.bfloat16))
                            hi, unused_hi = tla.simd.vdintlv(tla.simd.vtruncf(e1, tl.bfloat16),
                                                             tla.simd.vtruncf(e3, tl.bfloat16))
                            tla.simd.vsstb(lo, packed[0, i * 16], PACK_PITCH)
                            tla.simd.vsstb(hi, packed[0, (i + KT // 4) * 16], PACK_PITCH)
                        total = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        total = tla.simd.vadd(tla.simd.vmul(tla.simd.vlds(denominator[0]), alpha), total)
                        tla.simd.vsts(denominator[0], total)
                else:
                    # Every row can see the intersection [q_start + QT - WINDOW, q_start].
                    edge = (k_start < q_start + QT - WINDOW) | (k_start + KT - 1 > q_start)
                    first_mask: tl.constexpr = 0 if SKIP_INTERIOR_MASK else 1
                    for MASKED in tl.static_range(first_mask, 2):
                        if not SKIP_INTERIOR_MASK or edge == MASKED:
                            with tla.scope.simd():
                                scale = tla.simd.vsplat(D**-0.5, tl.float32)
                                if MASKED:
                                    neg_inf = tla.simd.vsplat(float('-inf'), tl.float32)
                                    k0, k1 = tla.simd.vci(0), tla.simd.vci(64)
                                for row in tl.range(ROWS):
                                    x0 = tla.simd.vmul(tla.simd.vlds(scores[row, 0]), scale)
                                    x1 = tla.simd.vmul(tla.simd.vlds(scores[row, 64]), scale)
                                    if MASKED:
                                        right = (q_start + sub * ROWS + row - k_start).to(tl.int32)
                                        lo = tla.simd.vdup(right - WINDOW + 1, tl.int32)
                                        hi = tla.simd.vdup(right, tl.int32)
                                        valid0 = tla.simd.pand(tla.simd.vcmp(k0, lo, op=tla.simd.CMP.GE),
                                                               tla.simd.vcmp(k0, hi, op=tla.simd.CMP.LE))
                                        valid1 = tla.simd.pand(tla.simd.vcmp(k1, lo, op=tla.simd.CMP.GE),
                                                               tla.simd.vcmp(k1, hi, op=tla.simd.CMP.LE))
                                        x0 = tla.simd.vsel(valid0, x0, neg_inf)
                                        x1 = tla.simd.vsel(valid1, x1, neg_inf)
                                    old_max = tla.simd.vlds(maximum[row], dist='BRC_B32')
                                    new_max = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_max)
                                    alpha = tla.simd.vexp(tla.simd.vsub(old_max, new_max))
                                    e0 = tla.simd.vexp(tla.simd.vsub(x0, new_max))
                                    e1 = tla.simd.vexp(tla.simd.vsub(x1, new_max))
                                    total = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(e0, e1)))
                                    total = tla.simd.vfma(total, tla.simd.vlds(denominator[row], dist='BRC_B32'), alpha)
                                    tla.simd.vsts(maximum[row], new_max, dist='ONEPT_B32')
                                    tla.simd.vsts(denominator[row], total, dist='ONEPT_B32')
                                    tla.simd.vsts(correction[row], alpha, dist='ONEPT_B32')
                                    probabilities, _ = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                                       tla.simd.vtruncf(e1, tl.bfloat16))
                                    tla.simd.vsstb(probabilities, packed[0, row * 16], (ROWS + 1) * 16)

            with tla.scope.vector().task('publish_probability', stage=STAGES[1]):
                if COLUMN_SOFTMAX:
                    rows = tla.reshape(p, (QT // 16, KT * 16))
                    HALF: tl.constexpr = (ROWS // 16, KT // 2 * 16)
                    for half in tl.static_range(2):
                        source = tla.subview(packed, (half * (ROWS // 16), 0), HALF)
                        target = tla.subview(rows, (sub * (ROWS // 16), half * (KT // 2 * 16)), HALF)
                        tla.copy_ub_to_l1(source, target)
                else:
                    tla.copy_ub_to_l1(
                        tla.view(packed, (KT // 16, ROWS * 16)),
                        tla.view(tla.reshape(p, (KT // 16, QT * 16)), (KT // 16, ROWS * 16),
                                 offsets=(0, sub * ROWS * 16)))

            with tla.scope.cube().task('pv', stage=STAGES[2]):
                if LATE_VALUE:
                    # Keep a single V buffer when a wider K tile needs more L1.
                    offset = (kv_head * S + k_start) * D
                    tla.load_gm_to_l1(tla.view(V + offset, (KT, D)), value)
                tla.cube.mmad(p, value, pv, transpose_a=COLUMN_SOFTMAX, clear_accum=True)
                tla.copy_l0c_to_ub(
                    pv, partial,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('accumulate', stage=STAGES[2]):
                with tla.scope.simd():
                    for row in tl.range(ROWS):
                        alpha = tla.simd.vlds(correction[row], dist='BRC_B32')
                        for col in tl.static_range(D // 64):
                            update = tla.simd.vfma(tla.simd.vlds(partial[row, col * 64]),
                                                   tla.simd.vlds(accumulator[row, col * 64]), alpha)
                            tla.simd.vsts(accumulator[row, col * 64], update)

        output = tla.alloc((ROWS, D), tl.bfloat16, space='ub')
        with tla.scope.vector():
            with tla.scope.simd():
                one = tla.simd.vsplat(1.0, tl.float32)
                for row in tl.range(ROWS):
                    reciprocal = tla.simd.vdiv(one, tla.simd.vlds(denominator[row], dist='BRC_B32'))
                    for col in tl.static_range(D // 128):
                        lo = tla.simd.vmul(tla.simd.vlds(accumulator[row, col * 128]), reciprocal)
                        hi = tla.simd.vmul(tla.simd.vlds(accumulator[row, col * 128 + 64]), reciprocal)
                        values, unused = tla.simd.vdintlv(tla.simd.vtruncf(lo, tl.bfloat16),
                                                          tla.simd.vtruncf(hi, tl.bfloat16))
                        tla.simd.vsts(output[row, col * 128], values)
            tla.store_ub_to_gm(output, tla.view(O + q_offset + sub * ROWS * D, (ROWS, D)))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, k, v, window, *, prefetch=True, q_tile=128, skip_interior_mask=True, column_softmax=None,
                 schedule=None, kv_tile=128, late_value=None):
    """Prepare causal BNSD GQA, with the current token included in window."""
    import torch

    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('Expected BNSD tensors')
    b, hq, s, d = q.shape
    hkv = k.shape[1]
    if not b or not hq or not hkv or hq % hkv or k.shape != (b, hkv, s, d):
        raise ValueError('Expected equal Q/KV sequence lengths and integral GQA groups')
    if not s or s % 128 or d not in (128, 256) or not isinstance(window, int) or not 1 <= window <= s:
        raise ValueError('Expected S divisible by 128, D=128 or 256, and 1 <= window <= S')
    if q_tile not in (64, 128):
        raise ValueError('Expected a Q tile of 64 or 128 rows')
    if column_softmax is None:
        column_softmax = q_tile == 128
    if column_softmax and q_tile != 128:
        raise ValueError('Column softmax needs 128 query rows')
    if kv_tile not in (128, 256) or (kv_tile == 256 and not column_softmax):
        raise ValueError('Expected 128 KV rows, or 256 for column softmax')
    # The narrow tile handles sequence lengths that are not aligned to 256.
    if s % kv_tile:
        kv_tile = 128
    if late_value is None:
        late_value = column_softmax
    if q.device.type != 'npu' or any(x.dtype != torch.bfloat16 or x.device != q.device or not x.is_contiguous()
                                    for x in (q, k, v)):
        raise ValueError('Expected contiguous BF16 tensors on the same Ascend NPU')
    cube, vector = device_cores(q.device)
    if cube <= 0 or vector != 2 * cube:
        raise ValueError(f'Expected a 1:2 Cube/Vector ratio, got {cube}:{vector}')
    cores = min(cube, b * hq * (s // q_tile))
    if schedule is None:
        schedule = 'early_pv' if kv_tile == 256 else ('overlap' if column_softmax else 'prefetch')
    if not prefetch:
        schedule = 'serial'
    stages = SCHEDULES[schedule]

    def call():
        out = torch.empty_like(q)
        sliding_attention[(cores,)](q, k, v, out, s // kv_tile, b, hq, hkv, s, d, window, cores, stages,
                                    QT=q_tile, SKIP_INTERIOR_MASK=skip_interior_mask,
                                    COLUMN_SOFTMAX=column_softmax, KT=kv_tile, LATE_VALUE=late_value,
                                    sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
        return out

    call.metadata = dict(q_tile=q_tile, skip_interior_mask=skip_interior_mask, column_softmax=column_softmax,
                         kv_tile=kv_tile, late_value=late_value, schedule=schedule, task_stages=stages)
    return call
