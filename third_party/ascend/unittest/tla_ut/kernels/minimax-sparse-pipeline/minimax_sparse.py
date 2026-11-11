"""MiniMax-M3 block-sparse GQA over preselected logical KV blocks."""

import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def _resolve_blocks(TABLE, LENGTHS, INDICES, COUNTS, PHYSICAL, VISIBLE, TOKENS: tl.constexpr, SQ: tl.constexpr,
                    HKV: tl.constexpr, TABLE_STRIDE: tl.constexpr, TOPK: tl.constexpr, PAGE: tl.constexpr,
                    BLOCK: tl.constexpr):
    item = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = item < TOKENS * HKV * TOPK
    row, rank = item // TOPK, item % TOPK
    token = row % TOKENS
    batch = token // SQ
    count = tl.load(COUNTS + row, mask=active, other=1)
    logical = tl.load(INDICES + row * TOPK + tl.minimum(rank, count - 1), mask=active, other=0)
    physical = tl.load(TABLE + batch * TABLE_STRIDE + logical, mask=active, other=0)
    length = tl.load(LENGTHS + batch, mask=active, other=0)
    remaining = length - SQ + token % SQ + 1 - logical * PAGE
    remaining = tl.where(rank < count, tl.maximum(0, tl.minimum(PAGE, remaining)), 0)
    tl.store(PHYSICAL + item, physical, mask=active)
    tl.store(VISIBLE + item, remaining, mask=active)


@tla.jit
def sparse_attention(Q, K, V, TABLE, LENGTHS, INDICES, COUNTS, OUT, PHYSICAL, VISIBLE,
                     B: tl.constexpr, SQ: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
                     TABLE_STRIDE: tl.constexpr, TOPK: tl.constexpr, CORES: tl.constexpr,
                     COMPUTE_STAGE: tl.constexpr, OVERLAP: tl.constexpr = False, SKIP_FULL_MASK: tl.constexpr = False,
                     RESOLVED_METADATA: tl.constexpr = False):
    D: tl.constexpr = 128
    PAGE: tl.constexpr = 128
    HEAD_TILE: tl.constexpr = 16
    ROWS: tl.constexpr = HEAD_TILE // 2
    TOKENS: tl.constexpr = B * SQ
    tl.static_assert(HQ == HKV * HEAD_TILE)

    for job in tl.range(tl.program_id(0), TOKENS * HKV, CORES):
        token = job // HKV
        kv_head = job % HKV
        batch = token // SQ
        query_position = token % SQ
        q_offset = token * HQ + kv_head * HEAD_TILE
        index_row = kv_head * TOKENS + token
        sub = tla.sub_block_id()

        q = tla.alloc((D // 16, HEAD_TILE // 16, 16, 16), tl.bfloat16, space='l1')
        max_state = tla.alloc((64,), tl.float32, space='ub')
        sum_state = tla.alloc((64,), tl.float32, space='ub')
        accumulator = tla.alloc((ROWS, D), tl.float32, space='ub')
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + q_offset * D, (HEAD_TILE, D)), q)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                tla.simd.vsts(max_state[0], tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64))
                tla.simd.vsts(sum_state[0], zero)
                for row in tl.range(ROWS):
                    tla.simd.vsts(accumulator[row, 0], zero)
                    tla.simd.vsts(accumulator[row, 64], zero)

        # The mask depends on the selected logical block and the query token,
        # while the memory addresses also use the request's physical page table.
        for rank in tla.pipeline(TOPK):
            k = tla.alloc((D // 16, PAGE // 16, 16, 16), tl.bfloat16, space='l1')
            v = tla.alloc((D // 16, PAGE // 16, 16, 16), tl.bfloat16, space='l1')
            probability = tla.alloc((PAGE // 16, HEAD_TILE // 16, 16, 16), tl.bfloat16, space='l1')
            qk = tla.alloc((PAGE // 16, HEAD_TILE // 16, 16, 16), tl.float32, space='l0c')
            pv = tla.alloc((D // 16, HEAD_TILE // 16, 16, 16), tl.float32, space='l0c')
            score = tla.alloc((ROWS, PAGE), tl.float32, space='ub')
            packed_p = tla.alloc((PAGE // 16, (ROWS + 1) * 16), tl.bfloat16, space='ub')
            partial = tla.alloc((ROWS, D), tl.float32, space='ub')
            correction = tla.alloc((64,), tl.float32, space='ub')

            with tla.scope.cube().task('load_block', stage=0):
                if RESOLVED_METADATA:
                    physical = tl.load(PHYSICAL + index_row * TOPK + rank)
                else:
                    count = tl.load(COUNTS + index_row)
                    # Clamp padded ranks to a legal page. Vector masks their entire contribution.
                    safe_rank = tl.minimum(rank, count - 1)
                    logical = tl.load(INDICES + index_row * TOPK + safe_rank)
                    physical = tl.load(TABLE + batch * TABLE_STRIDE + logical)
                offset = (physical.to(tl.int64) * PAGE * HKV + kv_head) * D
                tla.load_gm_to_l1(tla.view(K + offset, (PAGE, D), strides=(HKV * D, 1)), k)
                tla.load_gm_to_l1(tla.view(V + offset, (PAGE, D), strides=(HKV * D, 1)), v)

            with tla.scope.cube().task('qk', stage=0 if OVERLAP else COMPUTE_STAGE):
                tla.cube.mmad(q, k, qk, transpose_b=True, clear_accum=True)
                tla.copy_l0c_to_ub(
                    qk, score,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('softmax', stage=COMPUTE_STAGE):
                if RESOLVED_METADATA:
                    remaining = tl.load(VISIBLE + index_row * TOPK + rank)
                else:
                    count = tl.load(COUNTS + index_row)
                    length = tl.load(LENGTHS + batch)
                    safe_rank = tl.minimum(rank, count - 1)
                    logical = tl.load(INDICES + index_row * TOPK + safe_rank)
                    remaining = tl.where(rank < count, length - SQ + query_position + 1 - logical * PAGE, 0)
                for MASKED in tl.static_range(0 if SKIP_FULL_MASK else 1, 2):
                    if not SKIP_FULL_MASK or (remaining < PAGE) == MASKED:
                        with tla.scope.simd():
                            if MASKED:
                                limit = tla.simd.vdup(remaining, tl.int32)
                                valid0 = tla.simd.vcmp(tla.simd.vci(0), limit, op=tla.simd.CMP.LT)
                                valid1 = tla.simd.vcmp(tla.simd.vci(64), limit, op=tla.simd.CMP.LT)
                                neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                            for row in tl.range(ROWS):
                                x0 = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), D**-0.5)
                                x1 = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), D**-0.5)
                                if MASKED:
                                    x0 = tla.simd.vsel(valid0, x0, neg)
                                    x1 = tla.simd.vsel(valid1, x1, neg)
                                old_max = tla.simd.vlds(max_state[row], dist='BRC_B32')
                                maximum = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_max)
                                alpha = tla.simd.vexpdif(old_max, maximum)
                                e0 = tla.simd.vexpdif(x0, maximum)
                                e1 = tla.simd.vexpdif(x1, maximum)
                                denominator = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(e0, e1)))
                                denominator = tla.simd.vfma(denominator, tla.simd.vlds(sum_state[row], dist='BRC_B32'), alpha)
                                tla.simd.vsts(max_state[row], maximum, dist='ONEPT_B32')
                                tla.simd.vsts(sum_state[row], denominator, dist='ONEPT_B32')
                                tla.simd.vsts(correction[row], alpha, dist='ONEPT_B32')
                                p0 = tla.simd.vtruncf(e0, tl.bfloat16)
                                p1 = tla.simd.vtruncf(e1, tl.bfloat16)
                                packed, unused = tla.simd.vdintlv(p0, p1)
                                tla.simd.vsstb(packed, packed_p[0, row * 16], (ROWS + 1) * 16)

            with tla.scope.vector().task('publish_probability', stage=COMPUTE_STAGE):
                tla.copy_ub_to_l1(
                    tla.view(packed_p, (PAGE // 16, ROWS * 16)),
                    tla.view(tla.reshape(probability, (PAGE // 16, HEAD_TILE * 16)),
                             (PAGE // 16, ROWS * 16), offsets=(0, sub * ROWS * 16)))

            with tla.scope.cube().task('pv', stage=2 if OVERLAP else COMPUTE_STAGE):
                tla.cube.mmad(probability, v, pv, clear_accum=True)
                tla.copy_l0c_to_ub(
                    pv, partial,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('accumulate', stage=2 if OVERLAP else COMPUTE_STAGE):
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
                for row in tl.range(ROWS):
                    den = tla.simd.vlds(sum_state[row], dist='BRC_B32')
                    lo = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(accumulator[row, 0]), den), tl.bfloat16)
                    hi = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(accumulator[row, 64]), den), tl.bfloat16)
                    packed, unused = tla.simd.vdintlv(lo, hi)
                    tla.simd.vsts(output[row, 0], packed)
            tla.store_ub_to_gm(output, tla.view(OUT + (q_offset + sub * ROWS) * D, (ROWS, D)))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        properties = driver.active.utils.get_device_properties(device)
    return properties['num_aicore'], properties['num_vectorcore']


def prepare_call(q, k, v, table, lengths, indices, counts, *, prefetch=True, overlap=False, skip_full_mask=False,
                 resolve_metadata=False):
    """Prepare TND Q and PA_BBND KV for fixed-size query chunks per request.

    Each [head, token] index row contains a sorted, unique prefix of 1..TopK
    logical block IDs, including the block containing the query. Entries in
    this prefix and the corresponding page table must be valid. Each actual
    length covers the entire query chunk and fits the supplied table.
    """
    import torch

    if q.ndim != 3 or k.ndim != 4 or table.ndim != 2 or indices.ndim != 3:
        raise ValueError('Expected TND Q, PA_BBND K/V, B×pages table and KV-head×T×TopK indices')
    tokens, hq, dim = q.shape
    pages, page_size, hkv, kv_dim = k.shape
    b = table.shape[0]
    if min(b, tokens, pages, hkv) <= 0 or tokens % b or hq != hkv * 16 or (page_size, dim, kv_dim) != (128,) * 3:
        raise ValueError('Expected equal positive Q chunk lengths, 16 Q heads per KV head and page/D of 128')
    sq = tokens // b
    if table.shape[1] * page_size < sq or v.shape != k.shape or lengths.shape != (b,):
        raise ValueError('Expected valid table capacity, matching K/V and [B] lengths')
    topk = indices.shape[2]
    if topk <= 0 or indices.shape[:2] != (hkv, tokens) or counts.shape != (hkv, tokens):
        raise ValueError('Expected [Hkv,T,TopK] indices and [Hkv,T] counts')
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)) or any(
            t.dtype != torch.int32 for t in (table, lengths, indices, counts)):
        raise ValueError('Expected BF16 Q/K/V and int32 metadata')
    if q.device.type != 'npu' or any(t.device != q.device or not t.is_contiguous()
                                     for t in (q, k, v, table, lengths, indices, counts)):
        raise ValueError('Expected contiguous inputs on the same Ascend NPU')

    aic, aiv = device_cores(q.device)
    if aic <= 0 or aiv != 2 * aic:
        raise ValueError(f'Expected 1:2 Cube/Vector ratio, got {aic}:{aiv}')
    cores = min(aic, tokens * hkv)
    compute_stage = int(prefetch)
    options = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)

    def call():
        out = torch.empty_like(q)
        with torch.npu.device(q.device):
            physical, visible = indices, indices
            if resolve_metadata:
                physical, visible = torch.empty_like(indices), torch.empty_like(indices)
                block = 256
                _resolve_blocks[(triton.cdiv(indices.numel(), block),)](
                    table, lengths, indices, counts, physical, visible, tokens, sq, hkv, table.shape[1], topk,
                    page_size, block)
            sparse_attention[(cores,)](q, k, v, table, lengths, indices, counts, out, physical, visible, b, sq, hq, hkv,
                                        table.shape[1], topk, cores, compute_stage, prefetch and overlap,
                                        skip_full_mask, resolve_metadata, **options)
        return out

    call.metadata = dict(cube_grid=cores, compute_stage=compute_stage, head_tile=16, kv_page=128, topk=topk,
                         overlap=prefetch and overlap, skip_full_mask=skip_full_mask, resolve_metadata=resolve_metadata)
    return call
