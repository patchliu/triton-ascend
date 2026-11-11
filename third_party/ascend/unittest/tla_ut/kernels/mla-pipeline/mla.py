"""BF16 paged FlashMLA, with FP32 online state and LSE."""

import math
import triton
import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def attention_splits(
    Q,
    KV,
    BLOCK_TABLE,
    KV_LENGTHS,
    NUMERATOR,
    MAXIMUM,
    DENOMINATOR,
    BATCH: tl.constexpr,
    QUERY_LENGTH: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    SPLIT_PAGES: tl.constexpr,
    SPLITS: tl.constexpr,
    CUBE_CORES: tl.constexpr,
    SCALE: tl.constexpr,
    Q_TILE_ROWS: tl.constexpr,
    KV_L2: tl.constexpr,
):
    PAGE: tl.constexpr = 128
    VECTOR_ROWS: tl.constexpr = Q_TILE_ROWS // 2
    QUERY_TILES: tl.constexpr = QUERY_LENGTH * QUERY_HEADS // Q_TILE_ROWS
    ALL_ROWS: tl.constexpr = BATCH * QUERY_LENGTH * QUERY_HEADS
    VALUE_DIM: tl.constexpr = 512
    ROPE_DIM: tl.constexpr = 64
    KEY_DIM: tl.constexpr = VALUE_DIM + ROPE_DIM
    Q_LATENT_NZ: tl.constexpr = (KEY_DIM // 16, Q_TILE_ROWS // 16, 16, 16)
    KV_LATENT_NZ: tl.constexpr = (KEY_DIM // 16, 8, 16, 16)
    PROB_NZ: tl.constexpr = (8, Q_TILE_ROWS // 16, 16, 16)
    tl.static_assert(Q_TILE_ROWS == 32 or Q_TILE_ROWS == 64 or Q_TILE_ROWS == 96)
    tl.static_assert(QUERY_LENGTH * QUERY_HEADS % Q_TILE_ROWS == 0)
    tl.static_assert(VECTOR_ROWS <= 64)
    tl.static_assert(QUERY_HEADS % Q_TILE_ROWS == 0 or VECTOR_ROWS % QUERY_HEADS == 0)
    tl.static_assert(128 == PAGE)

    # Persistent Cube jobs partition KV into independently reducible splits.
    for job in tl.range(tl.program_id(0), BATCH * QUERY_TILES * SPLITS, CUBE_CORES):
        batch = job // (QUERY_TILES * SPLITS)
        split = job // QUERY_TILES % SPLITS
        tile = job % QUERY_TILES
        first_page = split * SPLIT_PAGES
        num_pages = tl.minimum(SPLIT_PAGES, MAX_PAGES - first_page)
        request_row = tile * Q_TILE_ROWS
        q_row = batch * QUERY_LENGTH * QUERY_HEADS + request_row
        sub = tla.sub_block_id()
        first_token = first_page * PAGE
        num_tiles = num_pages
        query = tla.alloc((KEY_DIM // 16, Q_TILE_ROWS // 16, 16, 16), tl.bfloat16, space='l1')
        maximum_state = tla.alloc((64,), tl.float32, space='ub')
        denominator_state = tla.alloc((64,), tl.float32, space='ub')
        accumulator = tla.alloc((VECTOR_ROWS, VALUE_DIM), tl.float32, space='ub')
        with tla.scope.cube():
            row = q_row
            tla.load_gm_to_l1(
                tla.view(Q + row * KEY_DIM, (Q_TILE_ROWS, KEY_DIM)),
                tla.view(query, Q_LATENT_NZ, offsets=(0, 0, 0, 0)),
            )
        with tla.scope.vector():
            with tla.scope.simd():
                tla.simd.vsts(maximum_state[0], tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64))
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                tla.simd.vsts(denominator_state[0], zero)
                for row in tl.range(VECTOR_ROWS):
                    for col in tl.static_range(VALUE_DIM // 64):
                        tla.simd.vsts(accumulator[row, col * 64], zero)

        # Stage 0 produces QK/P, while stage 1 consumes the preceding tile.
        # Buffer versions, fill/drain, and synchronization belong to the compiler.
        for kv_index in tla.pipeline(num_tiles):
            token = first_token + kv_index * 128
            page = first_page + kv_index
            key_value = tla.alloc(KV_LATENT_NZ, tl.bfloat16, space='l1')
            probability = tla.alloc((8, Q_TILE_ROWS // 16, 16, 16), tl.bfloat16, space='l1')
            qk = tla.alloc(PROB_NZ, tl.float32, space='l0c')
            pv = tla.alloc((VALUE_DIM // 16, Q_TILE_ROWS // 16, 16, 16), tl.float32, space='l0c')
            score = tla.alloc((VECTOR_ROWS, PAGE), tl.float32, space='ub')
            packed_p = tla.alloc((PAGE // 16, (VECTOR_ROWS + 1) * 16), tl.bfloat16, space='ub')
            partial = tla.alloc((VECTOR_ROWS, VALUE_DIM), tl.float32, space='ub')
            correction = tla.alloc((64,), tl.float32, space='ub')
            with tla.scope.cube().task('load_page', stage=0):
                length = tl.load(KV_LENGTHS + batch)
                if Q_TILE_ROWS % QUERY_HEADS == 0:
                    last_query = (tile + 1) * (Q_TILE_ROWS // QUERY_HEADS) - 1
                else:
                    last_query = (request_row + Q_TILE_ROWS - 1).to(tl.int32) // QUERY_HEADS
                last_page = (length - QUERY_LENGTH + last_query) // PAGE
                physical = tl.load(BLOCK_TABLE + batch * MAX_PAGES + tl.minimum(page, last_page)).to(tl.int64)
                tla.load_gm_to_l1(
                    tla.view(KV + physical * PAGE * KEY_DIM, (PAGE, KEY_DIM)),
                    key_value,
                    tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2),
                )
            with tla.scope.cube().task('qk', stage=0):
                latent = tla.view(query, Q_LATENT_NZ, offsets=(0, 0, 0, 0))
                tla.cube.mmad(latent, key_value, qk, transpose_b=True, clear_accum=True)
                tla.copy_l0c_to_ub(
                    qk,
                    tla.view(score, (VECTOR_ROWS, 128), offsets=(0, 0)),
                    tla.FixpipeL0C2UBParams(
                        dma_mode=tla.FixpipeDMAMode.NZ2ND,
                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                        sub_block_idx=None,
                    ),
                )

            # FP32 max/sum state maintains one global softmax per query row.
            # Causal tails and empty splits use the masked path.
            with tla.scope.vector().task('softmax', stage=0):
                length = tl.load(KV_LENGTHS + batch)
                causal_base = (length - QUERY_LENGTH + 1 - token).to(tl.int32)
                if QUERY_HEADS % Q_TILE_ROWS == 0:
                    row_start = request_row + sub * VECTOR_ROWS
                    remaining = causal_base + row_start.to(tl.int32) // QUERY_HEADS
                else:
                    remaining = (
                        causal_base + tile * (Q_TILE_ROWS // QUERY_HEADS) + sub * (VECTOR_ROWS // QUERY_HEADS)
                    )
                for MASKED in tl.static_range(0, 2):
                    if (remaining < 128) == MASKED:
                        MASK_ROWS: tl.constexpr = (
                            QUERY_HEADS if MASKED and QUERY_HEADS % Q_TILE_ROWS != 0 else VECTOR_ROWS
                        )
                        with tla.scope.simd():
                            if MASKED:
                                neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                            for query in tl.static_range(VECTOR_ROWS // MASK_ROWS):
                                if MASKED:
                                    end = tla.simd.vdup(remaining + query, tl.int32)
                                    valid0 = tla.simd.vcmp(tla.simd.vci(0), end, op=tla.simd.CMP.LT)
                                if MASKED:
                                    valid1 = tla.simd.vcmp(tla.simd.vci(64), end, op=tla.simd.CMP.LT)
                                for row_in_query in tl.range(MASK_ROWS):
                                    row = query * MASK_ROWS + row_in_query
                                    state = row
                                    x0 = tla.simd.vmuls(tla.simd.vlds(score[state, 0]), SCALE)
                                    x1 = tla.simd.vmuls(tla.simd.vlds(score[state, 64]), SCALE)
                                    if MASKED:
                                        x0 = tla.simd.vsel(valid0, x0, neg)
                                    if MASKED:
                                        x1 = tla.simd.vsel(valid1, x1, neg)
                                    old_max = tla.simd.vlds(maximum_state[state], dist='BRC_B32')
                                    maximum = tla.simd.vmax(
                                        tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_max
                                    )
                                    alpha = tla.simd.vexpdif(old_max, maximum)
                                    e0 = tla.simd.vexpdif(x0, maximum)
                                    e1 = tla.simd.vexpdif(x1, maximum)
                                    denominator = tla.simd.vbroadcast(
                                        tla.simd.vreduce_sum(tla.simd.vadd(e0, e1))
                                    )
                                    denominator = tla.simd.vfma(
                                        denominator,
                                        tla.simd.vlds(denominator_state[state], dist='BRC_B32'),
                                        alpha,
                                    )
                                    tla.simd.vsts(maximum_state[state], maximum, dist='ONEPT_B32')
                                    tla.simd.vsts(denominator_state[state], denominator, dist='ONEPT_B32')
                                    tla.simd.vsts(correction[state], alpha, dist='ONEPT_B32')
                                    p0 = tla.simd.vtruncf(e0, tl.bfloat16)
                                    p1 = tla.simd.vtruncf(e1, tl.bfloat16)
                                    packed, unused = tla.simd.vdintlv(p0, p1)
                                    tla.simd.vsstb(packed, packed_p[0, row * 16], (VECTOR_ROWS + 1) * 16)
            with tla.scope.vector().task('publish_probability', stage=0):
                all_prob = tla.reshape(probability, (8, Q_TILE_ROWS * 16))
                tla.copy_ub_to_l1(
                    tla.view(packed_p, (8, VECTOR_ROWS * 16)),
                    tla.view(all_prob, (8, VECTOR_ROWS * 16), offsets=(0, sub * VECTOR_ROWS * 16)),
                )
            with tla.scope.cube().task('pv', stage=1):
                product = pv
                handoff = partial
                prob_part = tla.view(probability, PROB_NZ, offsets=(0, 0, 0, 0))
                value = tla.view(key_value, (VALUE_DIM // 16, 8, 16, 16), offsets=(0, 0, 0, 0))
                tla.cube.mmad(prob_part, value, product, clear_accum=True)
                tla.copy_l0c_to_ub(
                    product,
                    handoff,
                    tla.FixpipeL0C2UBParams(
                        dma_mode=tla.FixpipeDMAMode.NZ2ND,
                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                        sub_block_idx=None,
                    ),
                )
            with tla.scope.vector().task('accumulate', stage=1):
                handoff = partial
                with tla.scope.simd():
                    for row in tl.range(VECTOR_ROWS):
                        state = row
                        alpha = tla.simd.vlds(correction[state], dist='BRC_B32')
                        for col in tl.static_range(VALUE_DIM // 64):
                            column = col * 64
                            update = tla.simd.vfma(
                                tla.simd.vlds(handoff[row, col * 64]),
                                tla.simd.vlds(accumulator[state, column]),
                                alpha,
                            )
                            tla.simd.vsts(accumulator[state, column], update)
        with tla.scope.vector():
            workspace_row = split * ALL_ROWS + q_row + sub * VECTOR_ROWS
            tla.store_ub_to_gm(
                tla.view(accumulator, (VECTOR_ROWS, VALUE_DIM), offsets=(0, 0)),
                tla.view(NUMERATOR + workspace_row * VALUE_DIM, (VECTOR_ROWS, VALUE_DIM)),
            )
            tla.store_ub_to_gm(
                tla.view(maximum_state, (VECTOR_ROWS,), offsets=(0,)),
                tla.view(MAXIMUM + workspace_row, (VECTOR_ROWS,)),
            )
            tla.store_ub_to_gm(
                tla.view(denominator_state, (VECTOR_ROWS,), offsets=(0,)),
                tla.view(DENOMINATOR + workspace_row, (VECTOR_ROWS,)),
            )


@triton.jit
def merge_splits(
    NUMERATOR,
    MAXIMUM,
    DENOMINATOR,
    OUT,
    LSE,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT_TILE: tl.constexpr,
    COL_TILE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    LATENT: tl.constexpr = 512
    col_tiles: tl.constexpr = LATENT // COL_TILE
    job = tl.program_id(0)
    row = job // col_tiles
    col = job % col_tiles * COL_TILE + tl.arange(0, COL_TILE)
    split = tl.arange(0, SPLIT_TILE)
    split_row = split * ROWS + row
    maxima = tl.load(MAXIMUM + split_row, split < SPLITS, -float('inf'))
    sums = tl.load(DENOMINATOR + split_row, split < SPLITS, 0.0)
    maximum = tl.max(maxima, 0)
    weight = tl.exp(maxima - maximum)
    denominator = tl.sum(sums * weight, 0)
    numerators = tl.load(NUMERATOR + split_row[:, None] * LATENT + col[None, :], split[:, None] < SPLITS, 0.0)
    result = tl.sum(numerators * weight[:, None], 0) / denominator
    tl.store(OUT + row * LATENT + col, result)
    if RETURN_LSE:
        tl.store(LSE + row, maximum + tl.log(denominator), job % col_tiles == 0)


def device_cores(device):
    """Read the available hardware grid through the active Triton driver."""
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        properties = driver.active.utils.get_device_properties(device)
    return (properties['num_aicore'], properties['num_vectorcore'])


def prepare_call(
    q, kv, block_table, kv_lengths, *, scale=0.04, return_lse=True, head_tile=None, split_pages=None, kv_l2=4
):
    """Prepare paged MLA for packed BF16 Q and a single latent KV head.

    Q is [batch, query_length, query_heads, 576], KV is [pages, 128, 576].
    The first 512 KV columns supply both latent K and V, and the final
    64 columns supply RoPE K. Causality is aligned to the end of each KV
    sequence. Every actual KV length must be in [query_length, table capacity].
    The table must address valid physical pages below that actual length.
    A prepared call captures the tensor objects, and reads their current
    contents on every invocation. It allocates fresh output and workspace.
    """
    import torch

    if q.ndim != 4 or kv.ndim != 3:
        raise ValueError('Expected Q [B,Sq,H,576] and KV [pages,128,576]')
    batch, query_length, heads, width = q.shape
    if (
        min(batch, query_length, heads) <= 0
        or width != 576
        or kv.shape[0] <= 0
        or (kv.shape[1:] != (128, 576))
    ):
        raise ValueError('Expected nonempty packed Q/KV with 576 columns and 128-token pages')
    if block_table.ndim != 2 or block_table.shape[0] != batch or kv_lengths.shape != (batch,):
        raise ValueError('Expected block table [B,max_pages] and KV lengths [B]')
    if block_table.shape[1] * 128 < query_length:
        raise ValueError('The page table capacity must cover the query length')
    if q.dtype != torch.bfloat16 or kv.dtype != q.dtype:
        raise ValueError('Q and KV must both be BF16')
    if block_table.dtype != torch.int32 or kv_lengths.dtype != torch.int32:
        raise ValueError('Page metadata must be int32')
    tensors = (q, kv, block_table, kv_lengths)
    if q.device.type != 'npu' or any((t.device != q.device or not t.is_contiguous() for t in tensors)):
        raise ValueError('Inputs must be contiguous and on the same Ascend NPU')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Softmax scale must be finite and positive')
    if head_tile is None:
        head_tile = 96 if heads % 32 and query_length * heads % 96 == 0 else 64 if heads % 64 == 0 else 32
    if head_tile not in (32, 64, 96) or query_length * heads % head_tile:
        raise ValueError('Query rows must form complete 32-, 64- or 96-row tiles')
    if heads % head_tile and head_tile // 2 % heads:
        raise ValueError('Each Vector sub-block must cover complete query head groups')
    if split_pages is not None and (not isinstance(split_pages, int) or split_pages <= 0):
        raise ValueError('split_pages must be a positive integer')
    cube_cores, vector_cores = device_cores(q.device)
    if cube_cores <= 0 or vector_cores != 2 * cube_cores:
        raise ValueError(f'Expected two Vector cores per Cube core, got {cube_cores}:{vector_cores}')
    rows = batch * query_length * heads
    query_tiles = rows // head_tile
    max_pages = block_table.shape[1]
    if split_pages is None:
        groups = cube_cores // math.gcd(query_tiles, cube_cores)
        split_pages = triton.cdiv(max_pages, groups) if head_tile == 96 else 64
    splits = triton.cdiv(max_pages, split_pages)
    fuse_merge = False
    launch_cores = cube_cores if fuse_merge else min(cube_cores, query_tiles * splits)
    options = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False,
                   enable_optimized_metaop=True)

    def call():
        numerator = torch.empty((splits, rows, 512), dtype=torch.float32, device=q.device)
        maximum = torch.empty((splits, rows), dtype=torch.float32, device=q.device)
        denominator = torch.empty_like(maximum)
        output = torch.empty((batch, query_length, heads, 512), dtype=q.dtype, device=q.device)
        lse = (
            torch.empty((batch, query_length, heads), dtype=torch.float32, device=q.device)
            if return_lse
            else maximum
        )
        args = (
            q,
            kv,
            block_table,
            kv_lengths,
            numerator,
            maximum,
            denominator,
            batch,
            query_length,
            heads,
            max_pages,
            split_pages,
            splits,
            launch_cores,
            scale,
            head_tile,
            kv_l2,
        )
        with torch.npu.device(q.device):
            attention_splits[launch_cores,](*args, **options)
            if not fuse_merge:
                merge_splits[rows,](
                    numerator,
                    maximum,
                    denominator,
                    output,
                    lse,
                    rows,
                    splits,
                    triton.next_power_of_2(splits),
                    512,
                    return_lse,
                    sanitize_overflow=False,
                )
        return (output, lse) if return_lse else output

    call.metadata = dict(
        cube_grid=launch_cores,
        vector_cores=vector_cores,
        head_tile=head_tile,
        split_pages=split_pages,
        splits=splits,
        kv_tile=128,
        fuse_merge=fuse_merge,
        return_lse=return_lse,
        scale=scale,
        kv_l2=kv_l2,
    )
    return call
