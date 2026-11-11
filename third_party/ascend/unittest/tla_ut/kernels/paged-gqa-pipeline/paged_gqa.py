"""BF16 paged GQA with page prefetch and online softmax."""

import triton.language as tl
import triton.language.extra.tla as tla


# Stages for QK, softmax and P publication, PV, and the output update.
_SCHEDULES = {
    'serial': (0, 0, 0, 0),
    'prefetch': (1, 1, 1, 1),
    'overlap': (0, 1, 2, 2),
    'deep': (0, 1, 2, 3),
    'wide': (0, 0, 1, 1),
    'early_qk': (0, 1, 1, 1),
}


@tla.jit
def paged_gqa(Q, K, V, PAGE_TABLE, LENGTHS, O, B: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, SQ: tl.constexpr,
              Q_TILE: tl.constexpr, TABLE_STRIDE: tl.constexpr, CORES: tl.constexpr, SCHEDULE: tl.constexpr,
              COLUMN_SOFTMAX: tl.constexpr, PAGES_PER_TILE: tl.constexpr, COMPACT_DECODE: tl.constexpr,
              LATE_VALUE: tl.constexpr):
    D: tl.constexpr = 128
    PAGE: tl.constexpr = 128
    KV_TILE: tl.constexpr = PAGE * PAGES_PER_TILE
    PAGE_NZ: tl.constexpr = (D // 16, PAGE // 16, 16, 16)
    Q_PER_AIV: tl.constexpr = Q_TILE // 2
    STATS: tl.constexpr = 128 if Q_PER_AIV == 128 else 64
    GROUP: tl.constexpr = HQ // HKV
    DECODE: tl.constexpr = SQ == 1
    OUT_ROWS: tl.constexpr = GROUP if DECODE else Q_PER_AIV
    Q_TILES: tl.constexpr = 1 if DECODE else SQ // Q_TILE
    JOBS_PER_BATCH: tl.constexpr = HKV if DECODE else HQ * Q_TILES
    tl.static_assert(HQ % HKV == 0 and Q_TILE % 16 == 0 and Q_PER_AIV <= 128)
    tl.static_assert(not COMPACT_DECODE or DECODE)
    tl.static_assert(not LATE_VALUE or (not DECODE and PAGES_PER_TILE == 1))
    if DECODE:
        tl.static_assert(HQ % Q_TILE == 0 and Q_PER_AIV % GROUP == 0)
    else:
        tl.static_assert(SQ % Q_TILE == 0)
    if COLUMN_SOFTMAX:
        tl.static_assert(not DECODE and Q_PER_AIV % 64 == 0)
        P_ROWS: tl.constexpr = Q_TILE
        P_COLS: tl.constexpr = KV_TILE
        PACK_ROWS: tl.constexpr = 2 * (Q_PER_AIV // 16)
        PACK_PITCH: tl.constexpr = (KV_TILE // 2 + 1) * 16
    else:
        tl.static_assert(PAGES_PER_TILE == 1 and Q_PER_AIV <= 64)
        P_ROWS: tl.constexpr = PAGE
        P_COLS: tl.constexpr = Q_TILE
        PACK_ROWS: tl.constexpr = PAGE // 16
        PACK_PITCH: tl.constexpr = (Q_PER_AIV + 1) * 16

    for job in tl.range(tl.program_id(0), B * JOBS_PER_BATCH, CORES):
        batch = job // JOBS_PER_BATCH
        local_job = job % JOBS_PER_BATCH
        sub = tla.sub_block_id()
        if DECODE:
            kv_head = local_job
            first_head = kv_head * GROUP
            q_offset = batch * HQ + first_head // Q_TILE * Q_TILE
            out_offset = batch * HQ + first_head
            out_row = first_head % Q_PER_AIV
            owns_output = sub == (first_head % Q_TILE) // Q_PER_AIV
            max_pages = TABLE_STRIDE
        else:
            head = local_job // Q_TILES
            kv_head = head // GROUP
            token_start = local_job % Q_TILES * Q_TILE
            q_offset = (batch * HQ + head) * SQ + token_start
            out_offset = q_offset + sub * Q_PER_AIV
            out_row = 0
            owns_output = True
            max_pages = tl.cdiv(TABLE_STRIDE * PAGE - SQ + token_start + Q_TILE, PAGE)

        # Decode reads a full Cube-aligned block of heads from Q and stores
        # its own group. Prefill reads consecutive tokens for one Q head.
        q = tla.alloc((D // 16, Q_TILE // 16, 16, 16), tl.bfloat16, space='l1')
        maximum = tla.alloc((STATS,), tl.float32, space='ub')
        denominator = tla.alloc((STATS,), tl.float32, space='ub')
        accumulator = tla.alloc((Q_PER_AIV, D), tl.float32, space='ub')
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + q_offset * D, (Q_TILE, D)), q)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                for stat in tl.static_range(0, STATS, 64):
                    tla.simd.vsts(maximum[stat], tla.simd.vfull(float('-inf'), tl.float32, 64))
                    tla.simd.vsts(denominator[stat], zero)
                for row in tl.range(Q_PER_AIV):
                    for col in tl.static_range(0, D, 64):
                        tla.simd.vsts(accumulator[row, col], zero)

        # Bound the shared loop by the table capacity. Each core reads the
        # actual length in its own task, where the compiler knows its owner.
        for page in tla.pipeline(tl.cdiv(max_pages, PAGES_PER_TILE)):
            k = tla.alloc((D // 16, KV_TILE // 16, 16, 16), tl.bfloat16, space='l1')
            v = tla.alloc((D // 16, KV_TILE // 16, 16, 16), tl.bfloat16, space='l1')
            p = tla.alloc((P_ROWS // 16, P_COLS // 16, 16, 16), tl.bfloat16, space='l1')
            qk = tla.alloc((P_ROWS // 16, P_COLS // 16, 16, 16), tl.float32, space='l0c')
            pv = tla.alloc((D // 16, Q_TILE // 16, 16, 16), tl.float32, space='l0c')
            if COLUMN_SOFTMAX:
                score = tla.alloc((KV_TILE, Q_PER_AIV), tl.float32, space='ub')
            else:
                score = tla.alloc((Q_PER_AIV, PAGE), tl.float32, space='ub')
            packed_p = tla.alloc((PACK_ROWS, PACK_PITCH), tl.bfloat16, space='ub')
            correction = tla.alloc((STATS,), tl.float32, space='ub')
            partial = tla.alloc((Q_PER_AIV, D), tl.float32, space='ub')

            with tla.scope.cube().task('load_kv', stage=0):
                length = tl.load(LENGTHS + batch)
                # Padded table entries are undefined. Reuse the last valid
                # page for surplus iterations, which softmax masks out.
                last_page = tl.cdiv(length, PAGE) - 1
                for slot in tl.static_range(PAGES_PER_TILE):
                    logical_page = tl.minimum(page * PAGES_PER_TILE + slot, last_page)
                    physical = tl.load(PAGE_TABLE + batch * TABLE_STRIDE + logical_page)
                    offset = (physical.to(tl.int64) * HKV + kv_head) * PAGE * D
                    if PAGES_PER_TILE == 1:
                        page_k, page_v = k, v
                    else:
                        page_k = tla.subview(k, (0, slot * (PAGE // 16), 0, 0), PAGE_NZ)
                        page_v = tla.subview(v, (0, slot * (PAGE // 16), 0, 0), PAGE_NZ)
                    tla.load_gm_to_l1(tla.view(K + offset, (PAGE, D)), page_k)
                    if not LATE_VALUE:
                        tla.load_gm_to_l1(tla.view(V + offset, (PAGE, D)), page_v)

            with tla.scope.cube().task('qk', stage=SCHEDULE[0]):
                if COLUMN_SOFTMAX:
                    # K @ Q.T puts one query in each FP32 SIMD lane.
                    tla.cube.mmad(k, q, qk, transpose_b=True, clear_accum=True)
                    SPLIT: tl.constexpr = tla.FixpipeDualDstMode.COLUMN_SPLIT
                else:
                    tla.cube.mmad(q, k, qk, transpose_b=True, clear_accum=True)
                    SPLIT: tl.constexpr = tla.FixpipeDualDstMode.ROW_SPLIT
                tla.copy_l0c_to_ub(
                    qk, score,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('softmax', stage=SCHEDULE[1]):
                length = tl.load(LENGTHS + batch)
                if COLUMN_SOFTMAX:
                    for vector in tl.static_range(Q_PER_AIV // 64):
                        lane_base = vector * 64
                        # Only the boundary page needs a causal mask. Apply it
                        # to scores in place before the common max/sum loops.
                        first_query = length - SQ + token_start + sub * Q_PER_AIV + lane_base - page * KV_TILE
                        if first_query < KV_TILE - 1:
                            with tla.scope.simd():
                                query = tla.simd.vci(first_query.to(tl.int32))
                                neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                                for key in tl.range(KV_TILE):
                                    valid = tla.simd.vcmp(tla.simd.vdup(key, tl.int32), query, op=tla.simd.CMP.LE)
                                    value = tla.simd.vsel(valid, tla.simd.vlds(score[key, lane_base]), neg)
                                    tla.simd.vsts(score[key, lane_base], value)
                        with tla.scope.simd():
                            # Each lane carries one query's online statistics.
                            # Four independent chains keep the vector units fed.
                            mx0 = tla.simd.vfull(float('-inf'), tl.float32, 64)
                            mx1, mx2, mx3 = mx0, mx0, mx0
                            for i in tl.range(KV_TILE // 4):
                                mx0 = tla.simd.vmax(mx0, tla.simd.vlds(score[4 * i, lane_base]))
                                mx1 = tla.simd.vmax(mx1, tla.simd.vlds(score[4 * i + 1, lane_base]))
                                mx2 = tla.simd.vmax(mx2, tla.simd.vlds(score[4 * i + 2, lane_base]))
                                mx3 = tla.simd.vmax(mx3, tla.simd.vlds(score[4 * i + 3, lane_base]))
                            old_max = tla.simd.vlds(maximum[lane_base])
                            tile_max = tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3))
                            new_max = tla.simd.vmax(old_max, tla.simd.vmuls(tile_max, D**-0.5))
                            alpha = tla.simd.vexpdif(old_max, new_max)
                            tla.simd.vsts(correction[lane_base], alpha)
                            tla.simd.vsts(maximum[lane_base], new_max)
                            sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                            sum1, sum2, sum3 = sum0, sum0, sum0
                            for i in tl.range(KV_TILE // 4):
                                e0 = tla.simd.vexpdif(tla.simd.vmuls(tla.simd.vlds(score[i, lane_base]), D**-0.5), new_max)
                                e1 = tla.simd.vexpdif(
                                    tla.simd.vmuls(tla.simd.vlds(score[i + KV_TILE // 4, lane_base]), D**-0.5), new_max)
                                e2 = tla.simd.vexpdif(
                                    tla.simd.vmuls(tla.simd.vlds(score[i + KV_TILE // 2, lane_base]), D**-0.5), new_max)
                                e3 = tla.simd.vexpdif(
                                    tla.simd.vmuls(tla.simd.vlds(score[i + 3 * KV_TILE // 4, lane_base]), D**-0.5), new_max)
                                sum0 = tla.simd.vadd(sum0, e0)
                                sum1 = tla.simd.vadd(sum1, e1)
                                sum2 = tla.simd.vadd(sum2, e2)
                                sum3 = tla.simd.vadd(sum3, e3)
                                # Interleave the two key halves into padded P.T.
                                lo, unused_lo = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                                 tla.simd.vtruncf(e2, tl.bfloat16))
                                hi, unused_hi = tla.simd.vdintlv(tla.simd.vtruncf(e1, tl.bfloat16),
                                                                 tla.simd.vtruncf(e3, tl.bfloat16))
                                tla.simd.vsstb(lo, packed_p[vector * 8, i * 16], PACK_PITCH)
                                tla.simd.vsstb(hi, packed_p[vector * 8, (i + KV_TILE // 4) * 16], PACK_PITCH)
                            total = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                            total = tla.simd.vadd(tla.simd.vmul(tla.simd.vlds(denominator[lane_base]), alpha), total)
                            tla.simd.vsts(denominator[lane_base], total)
                else:
                    if COMPACT_DECODE:
                        flat_p = tla.reshape(packed_p, (PACK_ROWS * PACK_PITCH,))
                        begin = tl.where(owns_output, out_row, 0)
                        end = tl.where(owns_output, out_row + OUT_ROWS, 0)
                    else:
                        begin, end = 0, Q_PER_AIV
                    with tla.scope.simd():
                        if COMPACT_DECODE:
                            zeros = tla.simd.vfull(0.0, tl.bfloat16, 128)
                            for offset in tl.range(0, PACK_ROWS * PACK_PITCH, 128):
                                tla.simd.vsts(flat_p[offset], zeros)
                        neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        scale = tla.simd.vsplat(D**-0.5, tl.float32)
                        lane0 = tla.simd.vci(0)
                        lane1 = tla.simd.vci(64)
                        if DECODE:
                            limit = tla.simd.vdup(length - page * PAGE, tl.int32)
                            valid0 = tla.simd.vcmp(lane0, limit, op=tla.simd.CMP.LT)
                            valid1 = tla.simd.vcmp(lane1, limit, op=tla.simd.CMP.LT)
                        for row in tl.range(begin, end):
                            if not DECODE:
                                # The chunk is right aligned to the full KV length.
                                row_end = length - SQ + token_start + sub * Q_PER_AIV + row + 1
                                limit = tla.simd.vdup(row_end - page * PAGE, tl.int32)
                                valid0 = tla.simd.vcmp(lane0, limit, op=tla.simd.CMP.LT)
                                valid1 = tla.simd.vcmp(lane1, limit, op=tla.simd.CMP.LT)
                            x0 = tla.simd.vsel(valid0, tla.simd.vmul(tla.simd.vlds(score[row, 0]), scale), neg)
                            x1 = tla.simd.vsel(valid1, tla.simd.vmul(tla.simd.vlds(score[row, 64]), scale), neg)
                            old_max = tla.simd.vlds(maximum[row], dist='BRC_B32')
                            new_max = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_max)
                            alpha = tla.simd.vexp(tla.simd.vsub(old_max, new_max))
                            e0 = tla.simd.vexp(tla.simd.vsub(x0, new_max))
                            e1 = tla.simd.vexp(tla.simd.vsub(x1, new_max))
                            total = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(e0, e1)))
                            total = tla.simd.vadd(total, tla.simd.vmul(
                                tla.simd.vlds(denominator[row], dist='BRC_B32'), alpha))
                            tla.simd.vsts(maximum[row], new_max, dist='ONEPT_B32')
                            tla.simd.vsts(denominator[row], total, dist='ONEPT_B32')
                            tla.simd.vsts(correction[row], alpha, dist='ONEPT_B32')
                            values, unused = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                              tla.simd.vtruncf(e1, tl.bfloat16))
                            tla.simd.vsstb(values, packed_p[0, row * 16], PACK_PITCH)

            with tla.scope.vector().task('publish_probability', stage=SCHEDULE[1]):
                if COLUMN_SOFTMAX:
                    # Preserve the padded UB pitch and give each AIV its
                    # own 64 query columns of P.T.
                    rows = tla.reshape(p, (Q_TILE // 16, KV_TILE * 16))
                    HALF_SHAPE: tl.constexpr = (4, KV_TILE // 2 * 16)
                    for vector in tl.static_range(Q_PER_AIV // 64):
                        for half in tl.static_range(2):
                            source = tla.subview(packed_p, (vector * 8 + half * 4, 0), HALF_SHAPE)
                            target = tla.subview(rows, (sub * (Q_PER_AIV // 16) + vector * 4, half * (KV_TILE // 2 * 16)),
                                                 HALF_SHAPE)
                            tla.copy_ub_to_l1(source, target)
                else:
                    tla.copy_ub_to_l1(
                        tla.view(packed_p, (PACK_ROWS, Q_PER_AIV * 16)),
                        tla.view(tla.reshape(p, (PACK_ROWS, Q_TILE * 16)), (PACK_ROWS, Q_PER_AIV * 16),
                                 offsets=(0, sub * Q_PER_AIV * 16)))

            with tla.scope.cube().task('pv', stage=SCHEDULE[2]):
                if LATE_VALUE:
                    # Read the page again in Cube and keep only one live V
                    # tile while the earlier stages work on K and scores.
                    length = tl.load(LENGTHS + batch)
                    logical_page = tl.minimum(page, tl.cdiv(length, PAGE) - 1)
                    physical = tl.load(PAGE_TABLE + batch * TABLE_STRIDE + logical_page)
                    offset = (physical.to(tl.int64) * HKV + kv_head) * PAGE * D
                    tla.load_gm_to_l1(tla.view(V + offset, (PAGE, D)), v)
                tla.cube.mmad(p, v, pv, transpose_a=COLUMN_SOFTMAX, clear_accum=True)
                tla.copy_l0c_to_ub(
                    pv, partial,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('accumulate', stage=SCHEDULE[3]):
                if COMPACT_DECODE:
                    begin = tl.where(owns_output, out_row, 0)
                    end = tl.where(owns_output, out_row + OUT_ROWS, 0)
                else:
                    begin, end = 0, Q_PER_AIV
                with tla.scope.simd():
                    for row in tl.range(begin, end):
                        alpha = tla.simd.vlds(correction[row], dist='BRC_B32')
                        for col in tl.static_range(D // 64):
                            update = tla.simd.vfma(tla.simd.vlds(partial[row, col * 64]),
                                                   tla.simd.vlds(accumulator[row, col * 64]), alpha)
                            tla.simd.vsts(accumulator[row, col * 64], update)

        output = tla.alloc((OUT_ROWS, D), tl.bfloat16, space='ub')
        with tla.scope.vector():
            if owns_output:
                with tla.scope.simd():
                    for row in tl.range(OUT_ROWS):
                        local = out_row + row
                        den = tla.simd.vlds(denominator[local], dist='BRC_B32')
                        lo = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(accumulator[local, 0]), den), tl.bfloat16)
                        hi = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(accumulator[local, 64]), den), tl.bfloat16)
                        values, unused = tla.simd.vdintlv(lo, hi)
                        tla.simd.vsts(output[row, 0], values)
                tla.store_ub_to_gm(output, tla.view(O + out_offset * D, (OUT_ROWS, D)))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, k, v, table, lengths, *, prefetch=True, schedule=None, prefill_q_tile=None, column_softmax=None,
                 pages_per_tile=1, compact_decode=False, late_value=None):
    """Prepare paged GQA for contiguous BNSD Q and [page, head, token, dim] KV.

    Lengths include the current query chunk and are at least Sq. Every page
    entry referenced by a length names an allocated page. The caller keeps
    the inputs and page table stable per call.
    """
    import torch

    b, hq, sq, d = q.shape
    _, hkv, page, dk = k.shape
    if (d, page, dk) != (128, 128, 128) or k.shape != v.shape:
        raise ValueError('Expected Q [B,Hq,Sq,128] and KV [pages,Hkv,128,128]')
    if b <= 0 or hkv <= 0 or hq <= 0 or hq % hkv:
        raise ValueError('Expected positive B and Hq divisible by Hkv')
    if sq == 1:
        if hq % 16 or 8 % (hq // hkv):
            raise ValueError('Decode requires Hq divisible by 16 and a GQA group of 1, 2, 4 or 8')
        q_tile, jobs = 16, b * hkv
    else:
        if prefill_q_tile is None:
            prefill_q_tile = 128 if sq % 128 == 0 else 64
        if prefill_q_tile not in (64, 128, 256) or sq <= 0 or sq % prefill_q_tile:
            raise ValueError('Prefill requires Sq to be a positive multiple of its 64-, 128- or 256-row Q tile')
        q_tile, jobs = prefill_q_tile, b * hq * (sq // prefill_q_tile)
    if column_softmax is None:
        column_softmax = sq > 1 and (q_tile == 256 or prefetch and q_tile == 128)
    if (column_softmax and (sq == 1 or q_tile not in (128, 256))) or (q_tile == 256 and not column_softmax):
        raise ValueError('Column softmax uses 128 or 256 prefill rows; row softmax uses at most 128')
    if pages_per_tile != 1 and (pages_per_tile != 2 or not column_softmax or q_tile != 128):
        raise ValueError('A two-page KV tile requires 128-row column softmax')
    if compact_decode and sq != 1:
        raise ValueError('Compact query rows apply to decode')
    if late_value is None:
        late_value = prefetch and column_softmax and pages_per_tile == 1
    if late_value and (sq == 1 or pages_per_tile != 1):
        raise ValueError('Late V applies to single-page prefill tiles')
    if table.ndim != 2 or table.shape[0] != b or table.shape[1] == 0 or lengths.shape != (b,):
        raise ValueError('Expected page table [B,max_pages] and lengths [B]')
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)) or any(x.dtype != torch.int32 for x in (table, lengths)):
        raise ValueError('Expected BF16 Q/K/V and int32 page table/lengths')
    if any(x.device != q.device or not x.is_contiguous() for x in (q, k, v, table, lengths)) or q.device.type != 'npu':
        raise ValueError('Expected contiguous inputs on the same Ascend device')

    aic, aiv = device_cores(q.device)
    if aic <= 0 or aiv != 2 * aic:
        raise ValueError(f'Expected a 1:2 Cube/Vector ratio, got {aic}:{aiv}')
    cores = min(aic, jobs)
    if schedule is None:
        if sq == 1:
            schedule = 'serial'
        elif q_tile == 256:
            schedule = 'wide'
        else:
            schedule = 'deep' if late_value else ('overlap' if column_softmax else 'prefetch')
    if not prefetch:
        schedule = 'serial'
    stages = _SCHEDULES[schedule]

    def call():
        out = torch.empty_like(q)
        paged_gqa[(cores,)](q, k, v, table, lengths, out, b, hq, hkv, sq, q_tile, table.shape[1], cores, stages,
                            column_softmax, pages_per_tile, compact_decode, late_value,
                            sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
        return out

    call.metadata = dict(grid=cores, schedule=schedule, task_stages=stages, q_tile=q_tile, kv_tile=128 * pages_per_tile,
                         column_softmax=column_softmax, compact_decode=compact_decode, late_value=late_value)
    return call
