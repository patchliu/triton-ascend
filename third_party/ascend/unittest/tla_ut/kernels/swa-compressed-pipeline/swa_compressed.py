"""DeepSeek V4 HCA with shared BF16 K/V and one online softmax for two sources."""

import triton.language as tl
import triton.language.extra.tla as tla

WINDOW = 128
RATIO = 128


@tla.jit
def hca_attention(Q, ORIGINAL, COMPRESSED, SINK, O, SQ: tl.constexpr, HISTORY: tl.constexpr,
                  CORES: tl.constexpr, COMPUTE_STAGE: tl.constexpr, EARLY_QK: tl.constexpr = False,
                  SKIP_FULL_MASK: tl.constexpr = False):
    D: tl.constexpr = 512
    H: tl.constexpr = 64
    ROWS: tl.constexpr = H // 2
    TILE: tl.constexpr = 128
    SCALE: tl.constexpr = D**-0.5

    for query_index in tl.range(tl.program_id(0), SQ, CORES):
        position = HISTORY - SQ + query_index
        # The original window includes the query's current token. A compressed
        # row is visible as soon as its entire 128-token group is complete.
        window_start = tl.maximum(position + 1 - TILE, 0)
        visible_compressed = (position + 1) // TILE
        sub = tla.sub_block_id()
        q = tla.alloc((D // 16, H // 16, 16, 16), tl.bfloat16, space='l1')
        maximum = tla.alloc((64,), tl.float32, space='ub')
        denominator = tla.alloc((64,), tl.float32, space='ub')
        accumulator = tla.alloc((ROWS, D), tl.float32, space='ub')
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + query_index * H * D, (H, D)), q)
        with tla.scope.vector():
            # A sink adds exp(sink) to the denominator and zero to the
            # numerator, exactly once across both sources.
            tla.load_gm_to_ub(tla.view(SINK + sub * ROWS, (ROWS,)), tla.view(maximum, (ROWS,)))
            with tla.scope.simd():
                zero = tla.simd.vsplat(0.0, tl.float32)
                tla.simd.vsts(denominator[0], tla.simd.vsplat(1.0, tl.float32))
                for row in tl.range(ROWS):
                    for col in tl.static_range(D // 64):
                        tla.simd.vsts(accumulator[row, col * 64], zero)

        # The first iteration reads the original window. Subsequent iterations
        # read complete and partial compressed tiles in chronological order.
        for tile in tla.pipeline(1 + tl.cdiv(visible_compressed, TILE)):
            kv = tla.alloc((D // 16, TILE // 16, 16, 16), tl.bfloat16, space='l1')
            p = tla.alloc((TILE // 16, H // 16, 16, 16), tl.bfloat16, space='l1')
            qk = tla.alloc((TILE // 16, H // 16, 16, 16), tl.float32, space='l0c')
            pv = tla.alloc((D // 16, H // 16, 16, 16), tl.float32, space='l0c')
            score = tla.alloc((ROWS, TILE), tl.float32, space='ub')
            packed = tla.alloc((TILE // 16, (ROWS + 1) * 16), tl.bfloat16, space='ub')
            correction = tla.alloc((64,), tl.float32, space='ub')
            partial = tla.alloc((ROWS, D), tl.float32, space='ub')
            compact_start = tl.maximum(tile - 1, 0) * TILE

            with tla.scope.cube().task('load_kv', stage=0):
                if tile == 0:
                    tla.load_gm_to_l1(tla.view(ORIGINAL + window_start * D, (TILE, D)), kv)
                else:
                    tla.load_gm_to_l1(tla.view(COMPRESSED + compact_start * D, (TILE, D)), kv)

            with tla.scope.cube().task('qk', stage=0 if EARLY_QK else COMPUTE_STAGE):
                tla.cube.mmad(q, kv, qk, transpose_b=True, clear_accum=True)
                tla.copy_l0c_to_ub(
                    qk, score,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('softmax', stage=COMPUTE_STAGE):
                visible = tl.where(tile == 0, position + 1 - window_start, visible_compressed - compact_start)
                for MASKED in tl.static_range(0 if SKIP_FULL_MASK else 1, 2):
                    if not SKIP_FULL_MASK or (visible < TILE) == MASKED:
                        with tla.scope.simd():
                            if MASKED:
                                neg_inf = tla.simd.vsplat(float('-inf'), tl.float32)
                                limit = tla.simd.vdup(visible.to(tl.int32), tl.int32)
                                valid0 = tla.simd.vcmp(tla.simd.vci(0), limit, op=tla.simd.CMP.LT)
                                valid1 = tla.simd.vcmp(tla.simd.vci(64), limit, op=tla.simd.CMP.LT)
                            for row in tl.range(ROWS):
                                lo = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                                hi = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                                if MASKED:
                                    lo = tla.simd.vsel(valid0, lo, neg_inf)
                                    hi = tla.simd.vsel(valid1, hi, neg_inf)
                                old_max = tla.simd.vlds(maximum[row], dist='BRC_B32')
                                new_max = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(lo, hi)), old_max)
                                alpha = tla.simd.vexpdif(old_max, new_max)
                                exp_lo = tla.simd.vexpdif(lo, new_max)
                                exp_hi = tla.simd.vexpdif(hi, new_max)
                                total = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(exp_lo, exp_hi)))
                                total = tla.simd.vfma(total, tla.simd.vlds(denominator[row], dist='BRC_B32'), alpha)
                                tla.simd.vsts(maximum[row], new_max, dist='ONEPT_B32')
                                tla.simd.vsts(denominator[row], total, dist='ONEPT_B32')
                                tla.simd.vsts(correction[row], alpha, dist='ONEPT_B32')
                                values, unused = tla.simd.vdintlv(tla.simd.vtruncf(exp_lo, tl.bfloat16),
                                                                  tla.simd.vtruncf(exp_hi, tl.bfloat16))
                                tla.simd.vsstb(values, packed[0, row * 16], (ROWS + 1) * 16)

            with tla.scope.vector().task('publish_probability', stage=COMPUTE_STAGE):
                tla.copy_ub_to_l1(
                    tla.view(packed, (TILE // 16, ROWS * 16)),
                    tla.view(tla.reshape(p, (TILE // 16, H * 16)), (TILE // 16, ROWS * 16),
                             offsets=(0, sub * ROWS * 16)))

            with tla.scope.cube().task('pv', stage=COMPUTE_STAGE):
                tla.cube.mmad(p, kv, pv, clear_accum=True)
                tla.copy_l0c_to_ub(
                    pv, partial,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('accumulate', stage=COMPUTE_STAGE):
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
                        packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(lo, tl.bfloat16),
                                                          tla.simd.vtruncf(hi, tl.bfloat16))
                        tla.simd.vsts(output[row, col * 128], packed, dist='NORM_B16')
            tla.store_ub_to_gm(output, tla.view(O + (query_index * H + sub * ROWS) * D, (ROWS, D)))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, original, compressed, sink, *, prefetch=True, early_qk=False, skip_full_mask=False):
    """Return a single-sequence HCA attention call from the two BF16 KV pools."""
    import torch

    if q.ndim != 3 or q.shape[1:] != (64, 512) or original.ndim != 3 or original.shape[1:] != (1, 512):
        raise ValueError('Expected Q [Sq,64,512] and original KV [history,1,512]')
    sq, history = q.shape[0], original.shape[0]
    if history < WINDOW or not 1 <= sq <= history or compressed.shape != (history // RATIO, 1, 512):
        raise ValueError('Expected history >= 128, right-aligned Q and one compressed row per full group')
    if any(x.dtype != torch.bfloat16 for x in (q, original, compressed)):
        raise ValueError('Expected BF16 Q and both KV sources')
    if sink.shape != (64,) or sink.dtype != torch.float32:
        raise ValueError('Expected one finite FP32 sink per Q head')
    if q.device.type != 'npu' or any(x.device != q.device or not x.is_contiguous()
                                    for x in (q, original, compressed, sink)):
        raise ValueError('Expected contiguous inputs on the same Ascend NPU')
    cube, vector = device_cores(q.device)
    if cube <= 0 or vector != 2 * cube:
        raise ValueError(f'Expected a 1:2 Cube/Vector ratio, got {cube}:{vector}')
    cores = min(cube, sq)
    pad_rows = -compressed.shape[0] % WINDOW

    def call():
        # Preserve the native input contract. Only a partial tail tile needs
        # padding, and that device work is part of each measured call.
        kv = torch.nn.functional.pad(compressed, (0, 0, 0, 0, 0, pad_rows)) if pad_rows else compressed
        out = torch.empty_like(q)
        hca_attention[(cores,)](q, original, kv, sink, out, sq, history, cores, int(prefetch), early_qk,
                                 skip_full_mask, sanitize_overflow=False,
                                 enable_auto_bind_sub_block=False, enable_preload=False)
        return out

    call.metadata = dict(grid=cores, compute_stage=int(prefetch), tile=WINDOW, pad_rows=pad_rows,
                         early_qk=early_qk, skip_full_mask=skip_full_mask)
    return call
