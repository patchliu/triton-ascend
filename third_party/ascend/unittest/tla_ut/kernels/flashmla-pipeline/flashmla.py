"""DeepSeek V4.1 sparse decode with two token-packed FP8 KV caches."""

import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def _read_fp8_rows(CACHE, token, column, TOKENS: tl.constexpr):
    valid = (token >= 0) & (token < TOKENS)
    row = tl.maximum(token, 0).to(tl.int64) * 528
    data = tl.load(CACHE + row[:, None] + column[None, :], valid[:, None], 0)
    scale = tl.load(CACHE + row[:, None] + 512 + column[None, :] // 32, valid[:, None], 0)
    return data, scale, valid


@triton.jit
def _gather_fp8(MAIN, EXTRA, MAIN_IDS, EXTRA_IDS, KV, VALID, B: tl.constexpr, MAIN_TOP: tl.constexpr,
                EXTRA_TOP: tl.constexpr, MAIN_TOKENS: tl.constexpr, EXTRA_TOKENS: tl.constexpr,
                TILE: tl.constexpr, GROUPED_GATHER: tl.constexpr = False):
    DIM: tl.constexpr = 512
    TOP: tl.constexpr = MAIN_TOP + EXTRA_TOP
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    column = tl.arange(0, DIM)
    batch, index = row // TOP, row % TOP
    if GROUPED_GATHER and MAIN_TOP % TILE == 0 and EXTRA_TOP % TILE == 0:
        # An aligned tile belongs to exactly one cache. Select it once per
        # program instead of emitting two indirect byte/scale loads per lane.
        if tl.program_id(0) % (TOP // TILE) < MAIN_TOP // TILE:
            token = tl.load(MAIN_IDS + batch * MAIN_TOP + index)
            bits, exponents, valid = _read_fp8_rows(MAIN, token, column, MAIN_TOKENS)
        else:
            token = tl.load(EXTRA_IDS + batch * EXTRA_TOP + index - MAIN_TOP)
            bits, exponents, valid = _read_fp8_rows(EXTRA, token, column, EXTRA_TOKENS)
    else:
        main = index < MAIN_TOP
        main_token = tl.load(MAIN_IDS + batch * MAIN_TOP + index, (batch < B) & main, -1)
        extra_token = tl.load(EXTRA_IDS + batch * EXTRA_TOP + index - MAIN_TOP, (batch < B) & ~main, -1)
        # Keep the two base pointers distinct through the indirect loads.
        main_data, main_scale, main_valid = _read_fp8_rows(MAIN, main_token, column, MAIN_TOKENS)
        extra_data, extra_scale, extra_valid = _read_fp8_rows(EXTRA, extra_token, column, EXTRA_TOKENS)
        bits, exponents, valid = main_data | extra_data, main_scale | extra_scale, main_valid | extra_valid
    magnitude = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale = (exponents.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    finite = ((bits & 127) != 127) & (exponents != 255)
    dequantized = tl.where(finite, magnitude * scale, 0.0)
    tl.store(KV + row[:, None] * DIM + column[None, :], dequantized, (batch < B)[:, None])
    tl.store(VALID + row, valid.to(tl.int32), batch < B)


@tla.jit
def _attention(Q, KV, VALID, MAIN, EXTRA, MAIN_IDS, EXTRA_IDS, SINK, O, LSE, B: tl.constexpr, H: tl.constexpr,
               TOP: tl.constexpr, CORES: tl.constexpr, HAS_SINK: tl.constexpr, COMPUTE_STAGE: tl.constexpr,
               EARLY_QK: tl.constexpr = False, FUSED_GATHER: tl.constexpr = False, MAIN_TOP: tl.constexpr = 0,
               MAIN_TOKENS: tl.constexpr = 0, EXTRA_TOKENS: tl.constexpr = 0):
    DIM: tl.constexpr = 512
    HEAD_TILE: tl.constexpr = 64
    KV_TILE: tl.constexpr = 128
    GATHER_GROUP: tl.constexpr = 8
    NZ_PITCH: tl.constexpr = (GATHER_GROUP + 1) * 16
    ROWS: tl.constexpr = HEAD_TILE // 2
    HEAD_TILES: tl.constexpr = H // HEAD_TILE
    tl.static_assert(H % HEAD_TILE == 0 and TOP % KV_TILE == 0)
    if FUSED_GATHER:
        tl.static_assert(MAIN_TOP % KV_TILE == 0)

    for job in tl.range(tl.program_id(0), B * HEAD_TILES, CORES):
        batch = job // HEAD_TILES
        head = job % HEAD_TILES * HEAD_TILE
        sub = tla.sub_block_id()
        query = tla.alloc((DIM // 16, HEAD_TILE // 16, 16, 16), tl.bfloat16, space='l1')
        maximum = tla.alloc((64,), tl.float32, space='ub')
        denominator = tla.alloc((64,), tl.float32, space='ub')
        accumulator = tla.alloc((ROWS, DIM), tl.float32, space='ub')
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + (batch * H + head) * DIM, (HEAD_TILE, DIM)), query)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vsplat(0.0, tl.float32)
                # A finite initial max also handles a completely masked prefix.
                tla.simd.vsts(maximum[0], tla.simd.vsplat(-3.4028234663852886e38, tl.float32))
                tla.simd.vsts(denominator[0], zero)
                for row in tl.range(ROWS):
                    for col in tl.static_range(DIM // 64):
                        tla.simd.vsts(accumulator[row, col * 64], zero)

        for tile in tla.pipeline(TOP // KV_TILE):
            key_value = tla.alloc((DIM // 16, KV_TILE // 16, 16, 16), tl.bfloat16, space='l1')
            probability = tla.alloc((KV_TILE // 16, HEAD_TILE // 16, 16, 16), tl.bfloat16, space='l1')
            qk = tla.alloc((KV_TILE // 16, HEAD_TILE // 16, 16, 16), tl.float32, space='l0c')
            pv = tla.alloc((DIM // 16, HEAD_TILE // 16, 16, 16), tl.float32, space='l0c')
            score = tla.alloc((ROWS, KV_TILE), tl.float32, space='ub')
            valid = tla.alloc((KV_TILE,), tl.int32, space='ub')
            packed = tla.alloc((KV_TILE // 16, (ROWS + 1) * 16), tl.bfloat16, space='ub')
            correction = tla.alloc((64,), tl.float32, space='ub')
            partial = tla.alloc((ROWS, DIM), tl.float32, space='ub')
            offset = batch * TOP + tile * KV_TILE

            if FUSED_GATHER:
                raw = tla.alloc((GATHER_GROUP, 544), tl.uint8, space='ub')
                packed_kv = tla.alloc((32, NZ_PITCH), tl.bfloat16, space='ub')
            with tla.scope.vector().task('gather_kv', stage=0):
                if FUSED_GATHER:
                    main_tile = tile < MAIN_TOP // KV_TILE
                    main_ids = MAIN_IDS + batch * MAIN_TOP + tile * KV_TILE
                    extra_ids = EXTRA_IDS + batch * (TOP - MAIN_TOP) + tile * KV_TILE - MAIN_TOP
                    if main_tile:
                        tla.load_gm_to_ub(tla.view(main_ids, (KV_TILE,)), valid)
                    else:
                        tla.load_gm_to_ub(tla.view(extra_ids, (KV_TILE,)), valid)
                    for group in tl.range(KV_TILE // (2 * GATHER_GROUP)):
                        first = sub * (KV_TILE // 2) + group * GATHER_GROUP
                        for row in tl.range(GATHER_GROUP):
                            dst = tla.view(raw, (1, 528), offsets=(row, 0))
                            if main_tile:
                                token = tl.load(main_ids + first + row).to(tl.uint32)
                                safe = tl.where(token < MAIN_TOKENS, token, 0).to(tl.int64)
                                tla.load_gm_to_ub(tla.view(MAIN + safe * 528, (1, 528)), dst)
                            else:
                                token = tl.load(extra_ids + first + row).to(tl.uint32)
                                safe = tl.where(token < EXTRA_TOKENS, token, 0).to(tl.int64)
                                tla.load_gm_to_ub(tla.view(EXTRA + safe * 528, (1, 528)), dst)
                        with tla.scope.simd():
                            low_half = tla.simd.pge(32, 64)
                            byte_mask = tla.simd.vsplat(255, tl.uint32)
                            magnitude_mask = tla.simd.vsplat(127, tl.uint32)
                            zero_bits = tla.simd.vsplat(0, tl.uint32)
                            zero = tla.simd.vsplat(0.0, tl.float32)
                            for row in tl.range(GATHER_GROUP):
                                for pair in tl.static_range(4):
                                    for half in tl.static_range(2):
                                        chunk = pair * 2 + half
                                        scale0 = tla.simd.vlds(raw[row, DIM + chunk * 2], dist='BRC_B8')
                                        scale1 = tla.simd.vlds(raw[row, DIM + chunk * 2 + 1], dist='BRC_B8')
                                        exponent = tla.simd.vsel(low_half, tla.simd.vbitcast(scale0, 'uint32'),
                                                                tla.simd.vbitcast(scale1, 'uint32'))
                                        exponent = tla.simd.vand(exponent, byte_mask)
                                        valid_scale = tla.simd.vcmps(exponent, 255, op=tla.simd.CMP.NE)
                                        exponent = tla.simd.vsel(valid_scale, exponent, zero_bits)
                                        scale = tla.simd.vbitcast(tla.simd.vshls(exponent, 23), 'fp32')
                                        bits = tla.simd.vlds(raw[row, chunk * 64], dist='UNPK4_B8')
                                        magnitude = tla.simd.vand(tla.simd.vbitcast(bits, 'uint32'), magnitude_mask)
                                        finite = tla.simd.vcmps(magnitude, 127, op=tla.simd.CMP.NE)
                                        values = tla.simd.vextf(tla.simd.vbitcast(bits, 'fp8e4nv'), tl.float32)
                                        values = tla.simd.vsel(finite, values, zero)
                                        dequant = tla.simd.vtruncf(tla.simd.vmul(values, scale), tl.bfloat16)
                                        if half == 0:
                                            lo = dequant
                                        else:
                                            hi = dequant
                                    values, unused = tla.simd.vdintlv(lo, hi)
                                    tla.simd.vsstb(values, packed_kv[pair * 8, row * 16], NZ_PITCH)
                        tla.copy_ub_to_l1(
                            tla.view(packed_kv, (32, GATHER_GROUP * 16)),
                            tla.view(tla.reshape(key_value, (32, KV_TILE * 16)), (32, GATHER_GROUP * 16),
                                     offsets=(0, first * 16)))
            with tla.scope.cube().task('load_kv', stage=0):
                if not FUSED_GATHER:
                    tla.load_gm_to_l1(tla.view(KV + offset * DIM, (KV_TILE, DIM)), key_value)

            with tla.scope.cube().task('qk', stage=0 if EARLY_QK else COMPUTE_STAGE):
                tla.cube.mmad(query, key_value, qk, transpose_b=True, clear_accum=True)
                tla.copy_l0c_to_ub(
                    qk, score,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('softmax', stage=COMPUTE_STAGE):
                if FUSED_GATHER:
                    limit = tl.where(tile < MAIN_TOP // KV_TILE, MAIN_TOKENS, EXTRA_TOKENS)
                else:
                    tla.load_gm_to_ub(tla.view(VALID + offset, (KV_TILE,)), valid)
                with tla.scope.simd():
                    neg_inf = tla.simd.vsplat(float('-inf'), tl.float32)
                    scale = tla.simd.vsplat(DIM**-0.5, tl.float32)
                    if FUSED_GATHER:
                        valid_limit = tla.simd.vdup(limit, tl.int32)
                        lo_valid = tla.simd.vcmp(tla.simd.vlds(valid[0]), valid_limit, op=tla.simd.CMP.ULT)
                        hi_valid = tla.simd.vcmp(tla.simd.vlds(valid[64]), valid_limit, op=tla.simd.CMP.ULT)
                    else:
                        lo_valid = tla.simd.vcmps(tla.simd.vlds(valid[0]), 0, op=tla.simd.CMP.NE)
                        hi_valid = tla.simd.vcmps(tla.simd.vlds(valid[64]), 0, op=tla.simd.CMP.NE)
                    for row in tl.range(ROWS):
                        lo = tla.simd.vsel(lo_valid, tla.simd.vmul(tla.simd.vlds(score[row, 0]), scale), neg_inf)
                        hi = tla.simd.vsel(hi_valid, tla.simd.vmul(tla.simd.vlds(score[row, 64]), scale), neg_inf)
                        old_max = tla.simd.vlds(maximum[row], dist='BRC_B32')
                        new_max = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(lo, hi)), old_max)
                        alpha = tla.simd.vexp(tla.simd.vsub(old_max, new_max))
                        exp_lo = tla.simd.vexp(tla.simd.vsub(lo, new_max))
                        exp_hi = tla.simd.vexp(tla.simd.vsub(hi, new_max))
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
                    tla.view(packed, (KV_TILE // 16, ROWS * 16)),
                    tla.view(tla.reshape(probability, (KV_TILE // 16, HEAD_TILE * 16)),
                             (KV_TILE // 16, ROWS * 16), offsets=(0, sub * ROWS * 16)))

            with tla.scope.cube().task('pv', stage=COMPUTE_STAGE):
                tla.cube.mmad(probability, key_value, pv, clear_accum=True)
                tla.copy_l0c_to_ub(
                    pv, partial,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('accumulate', stage=COMPUTE_STAGE):
                with tla.scope.simd():
                    for row in tl.range(ROWS):
                        alpha = tla.simd.vlds(correction[row], dist='BRC_B32')
                        for col in tl.static_range(DIM // 64):
                            value = tla.simd.vfma(tla.simd.vlds(partial[row, col * 64]),
                                                  tla.simd.vlds(accumulator[row, col * 64]), alpha)
                            tla.simd.vsts(accumulator[row, col * 64], value)

        output = tla.alloc((ROWS, DIM), tl.bfloat16, space='ub')
        logsumexp = tla.alloc((ROWS,), tl.float32, space='ub')
        if HAS_SINK:
            sink = tla.alloc((ROWS,), tl.float32, space='ub')
        first_head = head + sub * ROWS
        out_row = batch * H + first_head
        with tla.scope.vector():
            if HAS_SINK:
                tla.load_gm_to_ub(tla.view(SINK + first_head, (ROWS,)), sink)
            with tla.scope.simd():
                one = tla.simd.vsplat(1.0, tl.float32)
                zero = tla.simd.vsplat(0.0, tl.float32)
                infinity = tla.simd.vsplat(float('inf'), tl.float32)
                for row in tl.range(ROWS):
                    denominator_row = tla.simd.vlds(denominator[row], dist='BRC_B32')
                    nonempty = tla.simd.vcmps(denominator_row, 0.0, op=tla.simd.CMP.GT)
                    total = tla.simd.vsel(nonempty, denominator_row, one)
                    lse = tla.simd.vadd(tla.simd.vlds(maximum[row], dist='BRC_B32'), tla.simd.vln(total))
                    lse = tla.simd.vsel(nonempty, lse, infinity)
                    tla.simd.vsts(logsumexp[row], lse, dist='ONEPT_B32')
                    factor = tla.simd.vdiv(one, total)
                    if HAS_SINK:
                        sink_lse = tla.simd.vsel(nonempty, lse, zero)
                        attenuation = tla.simd.vexp(tla.simd.vsub(tla.simd.vlds(sink[row], dist='BRC_B32'), sink_lse))
                        factor = tla.simd.vdiv(factor, tla.simd.vadd(one, attenuation))
                    for col in tl.static_range(DIM // 128):
                        lo = tla.simd.vmul(tla.simd.vlds(accumulator[row, col * 128]), factor)
                        hi = tla.simd.vmul(tla.simd.vlds(accumulator[row, col * 128 + 64]), factor)
                        values, unused = tla.simd.vdintlv(tla.simd.vtruncf(lo, tl.bfloat16),
                                                          tla.simd.vtruncf(hi, tl.bfloat16))
                        tla.simd.vsts(output[row, col * 128], values)
            tla.store_ub_to_gm(output, tla.view(O + out_row * DIM, (ROWS, DIM)))
            tla.store_ub_to_gm(logsumexp, tla.view(LSE + out_row, (ROWS,)))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, main_cache, main_ids, extra_cache, extra_ids, sink=None, *, prefetch=True, gather_tile=8,
                 grouped_gather=False, early_qk=False, fused_gather=None):
    """Return a call that gathers from the two FP8 caches and computes O and LSE."""
    import torch

    if q.ndim != 4 or not q.shape[0] or q.shape[1] != 1 or q.shape[2] not in (64, 128) or q.shape[3] != 512:
        raise ValueError('Expected Q [B,1,64 or 128,512]')
    batch, _, heads, dim = q.shape
    for cache, ids in ((main_cache, main_ids), (extra_cache, extra_ids)):
        if cache.ndim != 4 or cache.shape[2:] != (1, 528) or not cache.numel():
            raise ValueError('Expected token-packed FP8 cache [pages,page_size,1,528]')
        if cache.dtype not in (torch.uint8, torch.int8, torch.float8_e4m3fn):
            raise ValueError('Expected byte-addressable FP8 KV cache')
        if ids.ndim != 3 or ids.shape[:2] != (batch, 1) or ids.dtype != torch.int32 or not ids.shape[2]:
            raise ValueError('Expected indices [B,1,TopK] in int32')
    if q.dtype != torch.bfloat16 or (sink is not None and (sink.shape != (heads,) or sink.dtype != torch.float32)):
        raise ValueError('Expected BF16 Q and optional FP32 sink [H]')
    inputs = (q, main_cache, main_ids, extra_cache, extra_ids) + (() if sink is None else (sink,))
    if q.device.type != 'npu' or any(x.device != q.device or not x.is_contiguous() for x in inputs):
        raise ValueError('Expected contiguous inputs on the same Ascend NPU')

    main_top, extra_top = main_ids.shape[2], extra_ids.shape[2]
    top = main_top + extra_top
    if top % 128:
        raise ValueError('Expected total TopK to be a multiple of 128')
    fused_gather = main_top % 128 == 0 and (fused_gather is None or fused_gather)
    if gather_tile not in (8, 16, 32):
        raise ValueError('Expected a gather tile of 8, 16 or 32 tokens')
    cube, vector = device_cores(q.device)
    if cube <= 0 or vector != 2 * cube:
        raise ValueError(f'Expected a 1:2 Cube/Vector ratio, got {cube}:{vector}')
    cores = min(cube, batch * (heads // 64))
    main = main_cache.view(torch.uint8)
    extra = extra_cache.view(torch.uint8)

    def call():
        kv = valid = None
        output = torch.empty_like(q)
        lse = torch.empty((batch, heads, 1), dtype=torch.float32, device=q.device)
        if not fused_gather:
            kv = torch.empty((batch, top, dim), dtype=q.dtype, device=q.device)
            valid = torch.empty((batch, top), dtype=torch.int32, device=q.device)
            _gather_fp8[(triton.cdiv(batch * top, gather_tile),)](main, extra, main_ids, extra_ids, kv, valid, batch,
                                                                main_top, extra_top, main.numel() // 528,
                                                                extra.numel() // 528, gather_tile, grouped_gather)
        _attention[(cores,)](q, kv, valid, main, extra, main_ids, extra_ids, lse if sink is None else sink, output, lse,
                             batch, heads, top, cores, sink is not None, int(prefetch), early_qk, fused_gather,
                             main_top, main.numel() // 528, extra.numel() // 528, sanitize_overflow=False,
                             enable_auto_bind_sub_block=False, enable_preload=False)
        return output, lse

    call.metadata = dict(grid=cores, fused_gather=fused_gather, qk_stage=0 if early_qk else int(prefetch),
                         compute_stage=int(prefetch))
    if fused_gather:
        call.metadata['gather_group'] = 8
    else:
        call.metadata.update(gather_tile=gather_tile, grouped_gather=grouped_gather)
    return call
