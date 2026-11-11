"""Paged MLA with an explicit one-page lead between QK and PV."""

import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def mla_split_explicit(Q, QR, KV, KR, TABLE, LENGTHS, NUMERATOR, MAXIMUM, DENOMINATOR,
                       B: tl.constexpr, SQ: tl.constexpr, H: tl.constexpr, TABLE_STRIDE: tl.constexpr,
                       SPLIT_PAGES: tl.constexpr, SPLITS: tl.constexpr, CORES: tl.constexpr,
                       STAGES: tl.constexpr, SCALE: tl.constexpr, HEAD_TILE: tl.constexpr,
                       SKIP_FULL_MASK: tl.constexpr, KV_MAJOR: tl.constexpr, Q_GROUP: tl.constexpr,
                       WIDE_QK: tl.constexpr, PACKED: tl.constexpr, KV_L2: tl.constexpr, SPLIT_PV: tl.constexpr,
                       L0_MODE: tl.constexpr = None, OUT=None, LSE=None,
                       FUSE_MERGE: tl.constexpr = False, RETURN_LSE: tl.constexpr = False):
    tl.static_assert(Q_GROUP == 1 and not WIDE_QK and not SPLIT_PV)
    PAGE: tl.constexpr = 128
    LATENT: tl.constexpr = 512
    ROPE: tl.constexpr = 64
    WIDTH: tl.constexpr = LATENT + ROPE if PACKED else LATENT
    L1_MATMUL: tl.constexpr = L0_MODE is None or L0_MODE == 'l1_112' or L0_MODE == 'l1_112_dynamic'
    USE_L0: tl.constexpr = not L1_MATMUL
    STREAM_K_MODE: tl.constexpr = L0_MODE == 'native112_merge_dynflags_stream_k'
    HYBRID_MERGE: tl.constexpr = L0_MODE == 'native112_merge_dynflags_hybrid' or STREAM_K_MODE
    VECTOR_SUM: tl.constexpr = L0_MODE == 'native112_merge_dynflags_vecsum' or HYBRID_MERGE
    SHARED_DMA: tl.constexpr = L0_MODE == 'native112_merge_dynflags_dma' or VECTOR_SUM
    MANUAL_DMA: tl.constexpr = L0_MODE == 'native112_manual_dma' or L0_MODE == 'native112_merge_dma_banks' or SHARED_DMA
    DYNAMIC_DMA: tl.constexpr = L0_MODE == 'l1_112_dynamic' or L0_MODE == 'native112_dynamic' or MANUAL_DMA
    STREAM_MODE: tl.constexpr = L0_MODE == 'native112_merge_dynflags_stream' or HYBRID_MERGE
    STREAM_MERGE: tl.constexpr = STREAM_MODE and FUSE_MERGE
    FIRST_PV_COPY: tl.constexpr = L0_MODE == 'native112_merge_dynflags_first_pv' or SHARED_DMA or STREAM_MODE
    BALANCE_SHARED: tl.constexpr = L0_MODE == 'native112_merge_dynflags_balanced'
    SHARED_P: tl.constexpr = (L0_MODE == 'native112_shared_p' or L0_MODE == 'native112_merge_dynflags_shared'
                             or FIRST_PV_COPY or BALANCE_SHARED)
    ROTATE_BANK: tl.constexpr = L0_MODE == 'native112_phase' or L0_MODE == 'native112_merge_dynflags_phase' or SHARED_P
    BALANCE_TILES: tl.constexpr = L0_MODE == 'native112_balanced' or BALANCE_SHARED
    DYNAMIC_BLOCK_FLAGS: tl.constexpr = (L0_MODE == 'native112_merge_dynflags'
                                        or L0_MODE == 'native112_merge_dynflags_phase'
                                        or L0_MODE == 'native112_merge_dynflags_shared' or FIRST_PV_COPY
                                        or BALANCE_SHARED)
    ARENA_VECTOR_STATS: tl.constexpr = L0_MODE == 'native112_merge_arena_vec' or DYNAMIC_BLOCK_FLAGS
    VECTOR_MERGE_STATS: tl.constexpr = L0_MODE == 'native112_merge_vec' or ARENA_VECTOR_STATS
    PRESET_SCORE_TAIL: tl.constexpr = L0_MODE == 'native112_merge_tail'
    REUSE_QUERY_JOBS: tl.constexpr = L0_MODE == 'native112_merge_qreuse'
    FAST_EXIT: tl.constexpr = L0_MODE == 'native112_merge_fast_exit' and FUSE_MERGE
    ARENA_MODE: tl.constexpr = (L0_MODE == 'native112_merge_arena' or L0_MODE == 'native112_merge_arena_banks'
                               or ARENA_VECTOR_STATS)
    MERGE_ARENA: tl.constexpr = ARENA_MODE and FUSE_MERGE
    MERGE_MAJOR: tl.constexpr = L0_MODE == 'native112_merge_layout' or ARENA_MODE
    WIDE_MERGE: tl.constexpr = L0_MODE == 'native112_merge_wide'
    BANKED_L0C: tl.constexpr = (L0_MODE == 'native112_l0c' or L0_MODE == 'native112_merge_banks'
                               or L0_MODE == 'native112_merge_dma_banks' or L0_MODE == 'native112_merge_arena_banks'
                               or ARENA_VECTOR_STATS)
    FUSED_PV: tl.constexpr = not BANKED_L0C and (L0_MODE == 'native112_fused_pv' or L0_MODE == 'native112_merge'
                             or L0_MODE == 'native112_merge_sum' or WIDE_MERGE
                             or VECTOR_MERGE_STATS or BALANCE_TILES or PRESET_SCORE_TAIL or MERGE_MAJOR
                             or REUSE_QUERY_JOBS or L0_MODE == 'native112_merge_fast_exit')
    LATE_SUM: tl.constexpr = L0_MODE == 'native112_late_sum' or L0_MODE == 'native112_merge_sum'
    MANUAL_SYNC: tl.constexpr = (L0_MODE == 'native112_manual' or ROTATE_BANK or FUSED_PV
                                or MANUAL_DMA or LATE_SUM or BANKED_L0C)
    EARLY_BANDS: tl.constexpr = L0_MODE == 'native112_bands' or (MANUAL_SYNC and not FUSED_PV)
    NATIVE_CHUNKS: tl.constexpr = L0_MODE == 'native112_k5' or L0_MODE == 'native112_dynamic' or EARLY_BANDS or FUSED_PV
    THREE_SLOTS: tl.constexpr = L0_MODE == 'native112' or L0_MODE == 'l1_112' or NATIVE_CHUNKS or DYNAMIC_DMA
    KV_TILE: tl.constexpr = 112 if THREE_SLOTS else PAGE
    TILES_PER_SPLIT: tl.constexpr = (TABLE_STRIDE * PAGE + SPLITS * KV_TILE - 1) // (SPLITS * KV_TILE)
    SLOT_ELEMS: tl.constexpr = (LATENT + (HEAD_TILE if HEAD_TILE > ROPE else ROPE)) * KV_TILE
    RESIDENT_Q: tl.constexpr = USE_L0 and L0_MODE != 'reload'
    REUSE_P: tl.constexpr = L0_MODE == 'qp' or L0_MODE == 'prefetch' or L0_MODE == 'pingpong' or THREE_SLOTS
    PREFETCH_KV: tl.constexpr = L0_MODE == 'prefetch'
    PINGPONG_L0: tl.constexpr = L0_MODE == 'pingpong' or THREE_SLOTS
    QK_K: tl.constexpr = 96
    QK_CHUNKS: tl.constexpr = (96, 112, 128, 112, 128) if NATIVE_CHUNKS else (96, 96, 96, 96, 96, 96)
    QK_OFFSETS: tl.constexpr = (0, 6, 13, 21, 28)
    PV_K: tl.constexpr = 64
    PV_N: tl.constexpr = 256
    PV_CHUNKS: tl.constexpr = (PV_K, KV_TILE - PV_K)
    Q_L0_SHAPE: tl.constexpr = (QK_K // 16, HEAD_TILE // 16, 16, 16)
    P_L0_SHAPE: tl.constexpr = (KV_TILE // 16, HEAD_TILE // 16, 16, 16)
    K_L0_SHAPE: tl.constexpr = (QK_K // 16, KV_TILE // 16, 16, 16)
    V_L0_SHAPE: tl.constexpr = (PV_K // 16, PV_N // 16, 16, 16)
    L0B_BANK_ELEMENTS: tl.constexpr = PV_K * PV_N
    L0C_BANK_ELEMENTS: tl.constexpr = 128 * 1024 // 4
    L0C_EVENTS: tl.constexpr = 2 if BANKED_L0C else 3
    PAGE_L1_SHAPE: tl.constexpr = (WIDTH // 16, PAGE // 16, 16, 16)
    KV_L1_SHAPE: tl.constexpr = (WIDTH // 16, KV_TILE // 16, 16, 16)
    V_L1_SHAPE: tl.constexpr = (LATENT // 16, KV_TILE // 16, 16, 16)
    tl.static_assert(not USE_L0 or (PACKED and HEAD_TILE <= 96 and WIDTH % QK_K == 0))
    tl.static_assert(not THREE_SLOTS or (PACKED and SPLIT_PAGES * PAGE % KV_TILE == 0))
    ROWS: tl.constexpr = HEAD_TILE // 2
    MERGE_ROWS: tl.constexpr = 16 if MERGE_ARENA or (WIDE_MERGE and SPLITS <= 3) else (4 if VECTOR_MERGE_STATS else 8)
    QUERY_TILES: tl.constexpr = SQ * H // HEAD_TILE
    REQUEST_K_TILES: tl.constexpr = (TABLE_STRIDE * PAGE + KV_TILE - 1) // KV_TILE
    GLOBAL_K_TILES: tl.constexpr = B * QUERY_TILES * REQUEST_K_TILES
    CORE_K_TILES: tl.constexpr = (GLOBAL_K_TILES + CORES - 1) // CORES
    STREAM_K_SPLITS: tl.constexpr = min(CORES, (REQUEST_K_TILES + 2 * CORE_K_TILES - 2) // CORE_K_TILES)
    STREAM_K: tl.constexpr = STREAM_K_MODE and FUSE_MERGE and SPLITS >= STREAM_K_SPLITS
    JOBS: tl.constexpr = B * QUERY_TILES * SPLITS
    ALL_ROWS: tl.constexpr = B * SQ * H
    SCORE_READY: tl.constexpr = 0
    SCORE_FREE: tl.constexpr = 1
    PROB_READY: tl.constexpr = 2
    PROB_SLOT_READY: tl.constexpr = (PROB_READY, 10, 11)
    PROB_FLAG_BASE: tl.constexpr = 10
    PROB_FREE: tl.constexpr = 3
    PRODUCT_READY: tl.constexpr = 4
    PRODUCT_FREE: tl.constexpr = 5
    BAND_READY: tl.constexpr = (PRODUCT_READY, 8)
    BAND_FREE: tl.constexpr = (PRODUCT_FREE, 9)
    JOB_DONE: tl.constexpr = 6
    JOB_ACK: tl.constexpr = 7

    query = tla.alloc((WIDTH // 16, HEAD_TILE // 16, 16, 16), tl.bfloat16, space='l1')
    if THREE_SLOTS:
        slots = tla.alloc((3 * SLOT_ELEMS,), tl.bfloat16, space='l1')
    else:
        pages = tla.alloc((2 * (WIDTH // 16), PAGE // 16, 16, 16), tl.bfloat16, space='l1')
        probability = tla.alloc(P_L0_SHAPE, tl.bfloat16, space='l1')
        probability_flat = tla.reshape(probability, (PAGE // 16, HEAD_TILE * 16))
    if not PACKED:
        query_rope = tla.alloc((ROPE // 16, HEAD_TILE // 16, 16, 16), tl.bfloat16, space='l1')
        rope_page = tla.alloc((ROPE // 16, PAGE // 16, 16, 16), tl.bfloat16, space='l1')
    if USE_L0:
        if RESIDENT_Q:
            q_first = tla.alloc(Q_L0_SHAPE, tl.bfloat16, space='l0a')
        if NATIVE_CHUNKS:
            q_even = tla.alloc((7, HEAD_TILE // 16, 16, 16), tl.bfloat16, space='l0a')
            q_odd = tla.alloc((8, HEAD_TILE // 16, 16, 16), tl.bfloat16, space='l0a')
            if ROTATE_BANK:
                b_banks = tla.alloc((2 * L0B_BANK_ELEMENTS,), tl.bfloat16, space='l0b')
            else:
                b_even = tla.alloc((L0B_BANK_ELEMENTS,), tl.bfloat16, space='l0b')
                b_odd = tla.alloc((L0B_BANK_ELEMENTS,), tl.bfloat16, space='l0b')
        elif PINGPONG_L0:
            # QK and PV are consecutive on Cube. P reuses the two Q work
            # tiles, while the first Q tile stays resident for the job.
            q_work = tla.alloc((2 * (QK_K // 16), HEAD_TILE // 16, 16, 16), tl.bfloat16, space='l0a')
            p_l0 = tla.view(q_work, P_L0_SHAPE)
            b_banks = tla.alloc((2 * L0B_BANK_ELEMENTS,), tl.bfloat16, space='l0b')
        else:
            q_work = tla.alloc(Q_L0_SHAPE, tl.bfloat16, space='l0a')
            p_l0 = tla.alloc(P_L0_SHAPE, tl.bfloat16, space='l0a')
            k_l0 = tla.alloc(K_L0_SHAPE, tl.bfloat16, space='l0b')
            v_l0 = tla.alloc(V_L0_SHAPE, tl.bfloat16, space='l0b')
    if BANKED_L0C:
        c_banks = tla.alloc((2 * L0C_BANK_ELEMENTS,), tl.float32, space='l0c')
    else:
        qk = tla.alloc(P_L0_SHAPE, tl.float32, space='l0c')
        pv = tla.alloc((LATENT // 16, HEAD_TILE // 16, 16, 16), tl.float32, space='l0c')
    P_PACK_SHAPE: tl.constexpr = (PAGE // 16, (ROWS + 1) * 16)
    PARTIAL_SHAPE: tl.constexpr = (2 * ROWS, PV_N) if EARLY_BANDS else (ROWS, LATENT)
    if MERGE_ARENA:
        # Attention and merge are separated by the AIV fence. Their views
        # reuse one physical UB arena, as in the source baseline.
        SCORE_WORDS: tl.constexpr = ROWS * PAGE
        VALUE_WORDS: tl.constexpr = ROWS * LATENT
        STATE_OFFSET: tl.constexpr = SCORE_WORDS + 2 * VALUE_WORDS
        MERGE_NUM_WORDS: tl.constexpr = (3 if STREAM_MERGE else SPLITS) * MERGE_ROWS * LATENT
        MERGE_STAT_WORDS: tl.constexpr = SPLITS * MERGE_ROWS
        MERGE_READ_WORDS: tl.constexpr = MERGE_STAT_WORDS + (64 - MERGE_ROWS if VECTOR_MERGE_STATS else 0)
        WEIGHT_OFFSET: tl.constexpr = MERGE_NUM_WORDS + 2 * MERGE_READ_WORDS
        LSE_OFFSET: tl.constexpr = WEIGHT_OFFSET + MERGE_STAT_WORDS
        INVERSE_OFFSET: tl.constexpr = LSE_OFFSET + MERGE_ROWS
        F32_WORDS: tl.constexpr = max(STATE_OFFSET + 256, INVERSE_OFFSET + (MERGE_ROWS if VECTOR_MERGE_STATS else 0))
        BF16_WORDS: tl.constexpr = max(PAGE * (ROWS + 1), MERGE_ROWS * LATENT)
        float_arena = tla.alloc((F32_WORDS,), tl.float32, space='ub')
        bf16_arena = tla.alloc((BF16_WORDS,), tl.bfloat16, space='ub')
        score = tla.reshape(tla.view(float_arena, (SCORE_WORDS,)), (ROWS, PAGE))
        partial = tla.reshape(tla.view(float_arena, (VALUE_WORDS,), offsets=(SCORE_WORDS,)), PARTIAL_SHAPE)
        accumulator = tla.reshape(tla.view(float_arena, (VALUE_WORDS,), offsets=(SCORE_WORDS + VALUE_WORDS,)),
                                  (ROWS, LATENT))
        maximum = tla.view(float_arena, (64,), offsets=(STATE_OFFSET,))
        denominator = tla.view(float_arena, (64,), offsets=(STATE_OFFSET + 64,))
        corrections = tla.view(float_arena, (128,), offsets=(STATE_OFFSET + 128,))
        packed_p = tla.reshape(tla.view(bf16_arena, (PAGE * (ROWS + 1),)), P_PACK_SHAPE)
    else:
        score = tla.alloc((ROWS, PAGE), tl.float32, space='ub')
        packed_p = tla.alloc(P_PACK_SHAPE, tl.bfloat16, space='ub')
        partial = tla.alloc(PARTIAL_SHAPE, tl.float32, space='ub')
        maximum = tla.alloc((64,), tl.float32, space='ub')
        denominator = tla.alloc((64,), tl.float32, space='ub')
        corrections = tla.alloc((128,), tl.float32, space='ub')
        accumulator = tla.alloc((ROWS, LATENT), tl.float32, space='ub')
    if LATE_SUM or VECTOR_SUM:
        tile_sums = tla.alloc((128,), tl.float32, space='ub')
    sub = tla.sub_block_id()

    core = tl.program_id(0)
    if STREAM_K:
        # Divide the contiguous KV tile stream among the runtime Cube grid.
        core_first = tl.minimum(core * CORE_K_TILES, GLOBAL_K_TILES)
        core_end = tl.minimum(core_first + CORE_K_TILES, GLOBAL_K_TILES)
        job_start = core_first // REQUEST_K_TILES
        job_end = tl.where(core_first < core_end, tl.cdiv(core_end, REQUEST_K_TILES), job_start)
        job_stride: tl.constexpr = 1
    elif REUSE_QUERY_JOBS:
        jobs_per_core: tl.constexpr = (JOBS + CORES - 1) // CORES
        job_start = core * jobs_per_core
        job_end = tl.minimum(job_start + jobs_per_core, JOBS)
        job_stride: tl.constexpr = 1
    else:
        job_start = core
        job_end = JOBS
        job_stride: tl.constexpr = CORES
    for job in tl.range(job_start, job_end, job_stride):
        if STREAM_K:
            batch = job // QUERY_TILES
            tile = job % QUERY_TILES
            request_first = job * REQUEST_K_TILES
            split = core - request_first // CORE_K_TILES
        elif KV_MAJOR and not REUSE_QUERY_JOBS:
            batch = job // (QUERY_TILES * SPLITS)
            split = job // QUERY_TILES % SPLITS
            tile = job % QUERY_TILES
        else:
            batch = job // (QUERY_TILES * SPLITS)
            split = job % SPLITS
            tile = job // SPLITS % QUERY_TILES
        load_query = (job == job_start) | (split == 0) if REUSE_QUERY_JOBS else True
        request_row = tile * HEAD_TILE
        q_row = batch * SQ * H + request_row
        first_page = split * SPLIT_PAGES
        length = tl.load(LENGTHS + batch)
        last_query = ((request_row + HEAD_TILE - 1).to(tl.int32)) // H
        last_page = (length - SQ + last_query) // PAGE
        num_pages = tl.maximum(0, tl.minimum(SPLIT_PAGES, last_page + 1 - first_page))
        if STREAM_K:
            visible_tiles = tl.cdiv(length - SQ + last_query + 1, KV_TILE)
            intersection_first = tl.maximum(request_first, core_first)
            intersection_end = tl.minimum(request_first + visible_tiles, core_end)
            first_token = (intersection_first - request_first) * KV_TILE
            num_tiles = tl.maximum(0, intersection_end - intersection_first)
        elif BALANCE_TILES:
            first_tile = split * TILES_PER_SPLIT
            first_token = first_tile * KV_TILE
            visible_tiles = tl.cdiv(length - SQ + last_query + 1, KV_TILE)
            num_tiles = tl.maximum(0, tl.minimum(TILES_PER_SPLIT, visible_tiles - first_tile))
        else:
            first_token = first_page * PAGE
            num_tiles = tl.cdiv(num_pages * PAGE, KV_TILE) if THREE_SLOTS else num_pages
        with tla.scope.cube():
            if load_query:
                tla.load_gm_to_l1(tla.view(Q + q_row * WIDTH, (HEAD_TILE, WIDTH)), query)
                if not PACKED:
                    tla.load_gm_to_l1(tla.view(QR + q_row * ROPE, (HEAD_TILE, ROPE)), query_rope)
                if MANUAL_SYNC:
                    tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 3)
                    tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 3)
            if MANUAL_SYNC:
                # L1 and L0C begin free. Every stage consumes and returns
                # one credit for the slot it uses.
                for event in tl.static_range(3):
                    tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.MTE2, event)
                for event in tl.static_range(L0C_EVENTS):
                    tla.sync.set_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, event)
                for event in tl.static_range(2):
                    tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, event)
            if RESIDENT_Q and load_query:
                tla.copy_l1_to_l0(query, q_first, tla.CopyL1ToL0Params(
                    k_part_idx=0, k_part=QK_K, k_part_ceil=QK_K, k_part_loop=WIDTH // QK_K,
                    k_part_actual=QK_K, m=HEAD_TILE))
            if PREFETCH_KV and num_pages > 0:
                physical_first = tl.load(TABLE + batch * TABLE_STRIDE + first_page).to(tl.int64)
                tla.load_gm_to_l1(tla.view(KV + physical_first * PAGE * WIDTH, (PAGE, WIDTH)),
                                  tla.view(pages, PAGE_L1_SHAPE), tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2))
        with tla.scope.vector():
            if MANUAL_SYNC:
                tla.sync.set_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 0)
                # The previous job's output stores still use these UB views.
                tla.sync.set_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 1)
                tla.sync.wait_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 1)
            with tla.scope.simd():
                tla.simd.vsts(maximum[0], tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64))
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                tla.simd.vsts(denominator[0], zero)
                if PRESET_SCORE_TAIL:
                    pad_mask = tla.simd.pge(PAGE - KV_TILE, 64)
                    pad_value = tla.simd.vfull(float('-inf'), tl.float32, 64)
                if not FIRST_PV_COPY:
                    for row in tl.range(ROWS):
                        for col in tl.static_range(0, LATENT, 64):
                            tla.simd.vsts(accumulator[row, col], zero)
                        if PRESET_SCORE_TAIL:
                            tla.simd.vsts(score[row, KV_TILE], pad_value, mask=pad_mask)
            if FIRST_PV_COPY and num_tiles == 0:
                with tla.scope.simd():
                    zero = tla.simd.vfull(0.0, tl.float32, 64)
                    for row in tl.range(ROWS):
                        for col in tl.static_range(0, LATENT, 64):
                            tla.simd.vsts(accumulator[row, col], zero)
            if PRESET_SCORE_TAIL and num_tiles > 0:
                # Fixpipe only writes the first KV_TILE columns of each row.
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, SCORE_FREE)

        # Current QK can run before the previous PV. The extra iteration drains
        # the last PV and leaves every credit consumed before the next job.
        for step in tl.range(-1 if THREE_SLOTS else 0, num_tiles + 1):
            has_qk = (step >= 0) & (step < num_tiles) if THREE_SLOTS else step < num_tiles
            has_pv = step > 0
            slot = step % 3 if THREE_SLOTS else step % 2
            correction_slot = step % 2
            with tla.scope.cube():
                if THREE_SLOTS:
                    if step + 1 < num_tiles:
                        # The prologue is step -1. In steady state the next
                        # tile occupies a slot separate from QK and PV.
                        key_start = first_token + (step + 1) * KV_TILE
                        page = key_start // PAGE
                        in_page = key_start % PAGE
                        first_count = tl.minimum(KV_TILE, PAGE - in_page)
                        next_id = (step + 1) % 3
                        next_slot = next_id * SLOT_ELEMS
                        next_key = tla.reshape(tla.view(slots, (WIDTH * KV_TILE,), offsets=(next_slot,)), KV_L1_SHAPE)
                        first = tl.load(TABLE + batch * TABLE_STRIDE + tl.minimum(page, last_page)).to(tl.int64)
                        if MANUAL_SYNC:
                            tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.MTE2, next_id)
                        if DYNAMIC_DMA:
                            tla.load_gm_to_l1(
                                tla.view(KV + first * PAGE * WIDTH, (PAGE, WIDTH)), next_key,
                                tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2, row_count=first_count, source_row=in_page))
                            if first_count < KV_TILE:
                                second = tl.load(TABLE + batch * TABLE_STRIDE + tl.minimum(page + 1, last_page)).to(tl.int64)
                                tla.load_gm_to_l1(
                                    tla.view(KV + second * PAGE * WIDTH, (PAGE, WIDTH)), next_key,
                                    tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2, row_count=KV_TILE - first_count,
                                                       destination_row=first_count))
                        else:
                            for FRAGMENT in tl.static_range(16, KV_TILE + 1, 16):
                                if first_count == FRAGMENT:
                                    first_view = tla.view(next_key, (WIDTH // 16, FRAGMENT // 16, 16, 16))
                                    tla.load_gm_to_l1(tla.view(KV + (first * PAGE + in_page) * WIDTH, (FRAGMENT, WIDTH)),
                                                      first_view, tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2))
                                    if FRAGMENT < KV_TILE:
                                        second = tl.load(TABLE + batch * TABLE_STRIDE + tl.minimum(page + 1, last_page)).to(tl.int64)
                                        second_view = tla.view(next_key, (WIDTH // 16, (KV_TILE - FRAGMENT) // 16, 16, 16),
                                                               offsets=(0, FRAGMENT // 16, 0, 0))
                                        tla.load_gm_to_l1(tla.view(KV + second * PAGE * WIDTH, (KV_TILE - FRAGMENT, WIDTH)),
                                                          second_view, tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2))
                        if MANUAL_SYNC:
                            tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, next_id)
                if has_qk:
                    if THREE_SLOTS:
                        key = tla.reshape(tla.view(slots, (WIDTH * KV_TILE,), offsets=(slot * SLOT_ELEMS,)), KV_L1_SHAPE)
                    else:
                        key = tla.view(pages, PAGE_L1_SHAPE, offsets=(slot * (WIDTH // 16), 0, 0, 0))
                        if not PREFETCH_KV:
                            page = first_page + step
                            physical = tl.load(TABLE + batch * TABLE_STRIDE + page).to(tl.int64)
                            tla.load_gm_to_l1(tla.view(KV + physical * PAGE * WIDTH, (PAGE, WIDTH)), key,
                                              tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2))
                            if not PACKED:
                                tla.load_gm_to_l1(tla.view(KR + physical * PAGE * ROPE, (PAGE, ROPE)), rope_page)
                    if USE_L0:
                        if MANUAL_SYNC:
                            qk_event = step % 2 if BANKED_L0C else 0
                            tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, slot)
                            tla.sync.wait_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, qk_event)
                        if BANKED_L0C:
                            qk = tla.reshape(tla.view(c_banks, (HEAD_TILE * KV_TILE,),
                                                      offsets=(qk_event * L0C_BANK_ELEMENTS,)), P_L0_SHAPE)
                        for chunk in tl.static_range(5 if NATIVE_CHUNKS else WIDTH // QK_K):
                            if MANUAL_SYNC:
                                qk_bank = (step + chunk) % 2 if ROTATE_BANK else (chunk + 1) % 2
                                tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, qk_bank)
                            if RESIDENT_Q and chunk == 0:
                                q_operand = q_first
                            else:
                                q_fragment = query
                                if NATIVE_CHUNKS:
                                    q_operand = q_even if chunk % 2 else q_odd
                                    q_fragment = tla.view(query, (QK_CHUNKS[chunk] // 16, HEAD_TILE // 16, 16, 16),
                                                          offsets=(QK_OFFSETS[chunk], 0, 0, 0))
                                elif PINGPONG_L0:
                                    q_operand = tla.view(q_work, Q_L0_SHAPE,
                                                         offsets=(chunk % 2 * (QK_K // 16), 0, 0, 0))
                                else:
                                    q_operand = q_work
                                tla.copy_l1_to_l0(q_fragment, q_operand, tla.CopyL1ToL0Params(
                                    k_part_idx=0 if NATIVE_CHUNKS else chunk, k_part=QK_CHUNKS[chunk],
                                    k_part_ceil=QK_CHUNKS[chunk], k_part_loop=1 if NATIVE_CHUNKS else WIDTH // QK_K,
                                    k_part_actual=QK_CHUNKS[chunk], m=HEAD_TILE))
                            k_fragment = key
                            if NATIVE_CHUNKS:
                                if ROTATE_BANK:
                                    bank = tla.view(b_banks, (L0B_BANK_ELEMENTS,),
                                                    offsets=(qk_bank * L0B_BANK_ELEMENTS,))
                                else:
                                    bank = b_even if chunk % 2 else b_odd
                                k_operand = tla.reshape(tla.view(bank, (QK_CHUNKS[chunk] * KV_TILE,)),
                                                        (QK_CHUNKS[chunk] // 16, KV_TILE // 16, 16, 16))
                                k_fragment = tla.view(key, (QK_CHUNKS[chunk] // 16, KV_TILE // 16, 16, 16),
                                                      offsets=(QK_OFFSETS[chunk], 0, 0, 0))
                            elif PINGPONG_L0:
                                k_operand = tla.reshape(tla.view(b_banks, (QK_K * KV_TILE,),
                                                                offsets=(chunk % 2 * L0B_BANK_ELEMENTS,)), K_L0_SHAPE)
                            else:
                                k_operand = k_l0
                            tla.copy_l1_to_l0(k_fragment, k_operand, tla.CopyL1ToL0Params(
                                is_transpose=True, k_part_idx=0 if NATIVE_CHUNKS else chunk,
                                k_part=QK_CHUNKS[chunk], k_part_ceil=QK_CHUNKS[chunk],
                                k_part_loop=1 if NATIVE_CHUNKS else WIDTH // QK_K,
                                k_part_actual=QK_CHUNKS[chunk], m=KV_TILE))
                            if MANUAL_SYNC:
                                tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, qk_bank)
                                tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, qk_bank)
                            tla.cube.mmad_l0(q_operand, k_operand, qk, m=HEAD_TILE, n=KV_TILE, k=QK_CHUNKS[chunk],
                                             clear_accum=chunk == 0)
                            if MANUAL_SYNC:
                                tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, qk_bank)
                    else:
                        tla.cube.mmad(query, key, qk, transpose_b=True, clear_accum=True)
                        if not PACKED:
                            tla.cube.mmad(query_rope, rope_page, qk, transpose_b=True, clear_accum=False)
                    if MANUAL_SYNC:
                        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, qk_event)
                        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, qk_event)
                    if PRESET_SCORE_TAIL or has_pv:
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK,
                                                 tla.sync.PIPE.F if MANUAL_SYNC else tla.sync.PIPE.S, SCORE_FREE)
                    score_tile = tla.view(score, (ROWS, KV_TILE)) if THREE_SLOTS else score
                    tla.copy_l0c_to_ub(qk, score_tile, tla.FixpipeL0C2UBParams(
                        dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                        sub_block_idx=None))
                    if MANUAL_SYNC:
                        tla.sync.set_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, qk_event)
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, SCORE_READY)
                if has_pv:
                    if THREE_SLOTS:
                        previous = (step + 2) % 3 * SLOT_ELEMS
                        value = tla.reshape(tla.view(slots, (LATENT * KV_TILE,), offsets=(previous,)), V_L1_SHAPE)
                        probability_tile = tla.reshape(tla.view(slots, (HEAD_TILE * KV_TILE,),
                                                                offsets=(previous + LATENT * KV_TILE,)), P_L0_SHAPE)
                    else:
                        value = tla.view(pages, V_L1_SHAPE, offsets=((1 - slot) * (WIDTH // 16), 0, 0, 0))
                        probability_tile = probability
                    if MANUAL_SYNC:
                        # The Vector may publish the next P before Cube
                        # consumes the previous one. Each L1 slot owns a flag.
                        if DYNAMIC_BLOCK_FLAGS:
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1,
                                                     PROB_FLAG_BASE + (step + 2) % 3)
                        else:
                            for ready in tl.static_range(3):
                                if (step + 2) % 3 == ready:
                                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1,
                                                             PROB_SLOT_READY[ready])
                    else:
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, PROB_READY)
                    if USE_L0:
                        if ROTATE_BANK:
                            # The fifth QK chunk leaves the opposite bank free.
                            # During drain, wait for the previous PV's last read
                            # before replacing a shared P tile.
                            pv_phase = (step + ((step > 1) | has_qk).to(tl.int32)) % 2
                        for band in tl.static_range(LATENT // PV_N):
                            if MANUAL_SYNC:
                                pv_event = (step + band + has_qk.to(tl.int32)) % 2 if BANKED_L0C else band + 1
                                tla.sync.wait_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, pv_event)
                            if not NATIVE_CHUNKS and (not REUSE_P or band == 0):
                                tla.copy_l1_to_l0(probability_tile, p_l0, tla.CopyL1ToL0Params(
                                    k_part_idx=0, k_part=KV_TILE, k_part_ceil=KV_TILE, k_part_loop=1,
                                    k_part_actual=KV_TILE, m=HEAD_TILE))
                            v_band = tla.view(value, (PV_N // 16, KV_TILE // 16, 16, 16),
                                              offsets=(band * (PV_N // 16), 0, 0, 0))
                            if BANKED_L0C:
                                product = tla.reshape(tla.view(c_banks, (HEAD_TILE * PV_N,),
                                                                   offsets=(pv_event * L0C_BANK_ELEMENTS,)),
                                                      (PV_N // 16, HEAD_TILE // 16, 16, 16))
                            else:
                                product = tla.view(pv, (PV_N // 16, HEAD_TILE // 16, 16, 16),
                                                   offsets=(band * (PV_N // 16), 0, 0, 0))
                            for chunk in tl.static_range(2):
                                if MANUAL_SYNC:
                                    pv_bank = (pv_phase + chunk) % 2 if ROTATE_BANK else chunk
                                    tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, pv_bank)
                                if NATIVE_CHUNKS:
                                    # Once QK finishes, the two Q work banks
                                    # hold the 64- and 48-column P fragments.
                                    if SHARED_P:
                                        if band == 0 and chunk == 0:
                                            tla.copy_l1_to_l0(probability_tile, q_even, tla.CopyL1ToL0Params(
                                                k_part_idx=0, k_part=KV_TILE, k_part_ceil=KV_TILE, k_part_loop=1,
                                                k_part_actual=KV_TILE, m=HEAD_TILE))
                                        p_operand = tla.view(q_even, (PV_CHUNKS[chunk] // 16, HEAD_TILE // 16, 16, 16),
                                                             offsets=(chunk * (PV_K // 16), 0, 0, 0))
                                    else:
                                        work = q_even if chunk == 0 else q_odd
                                        p_operand = tla.view(work, (PV_CHUNKS[chunk] // 16, HEAD_TILE // 16, 16, 16))
                                        if band == 0:
                                            tla.copy_l1_to_l0(probability_tile, p_operand, tla.CopyL1ToL0Params(
                                                k_part_idx=chunk, k_part=PV_K, k_part_ceil=PV_CHUNKS[chunk],
                                                k_part_loop=2, k_part_actual=PV_CHUNKS[chunk], m=HEAD_TILE))
                                    if ROTATE_BANK:
                                        bank = tla.view(b_banks, (L0B_BANK_ELEMENTS,),
                                                        offsets=(pv_bank * L0B_BANK_ELEMENTS,))
                                    else:
                                        bank = b_even if chunk == 0 else b_odd
                                    v_operand = tla.reshape(tla.view(bank, (PV_CHUNKS[chunk] * PV_N,)),
                                                            (PV_CHUNKS[chunk] // 16, PV_N // 16, 16, 16))
                                else:
                                    p_operand = tla.view(p_l0, (PV_CHUNKS[chunk] // 16, HEAD_TILE // 16, 16, 16),
                                                         offsets=(chunk * (PV_K // 16), 0, 0, 0))
                                    if PINGPONG_L0:
                                        v_operand = tla.reshape(tla.view(b_banks, (PV_CHUNKS[chunk] * PV_N,),
                                                                        offsets=(chunk * L0B_BANK_ELEMENTS,)),
                                                                (PV_CHUNKS[chunk] // 16, PV_N // 16, 16, 16))
                                    else:
                                        v_operand = v_l0
                                tla.copy_l1_to_l0(v_band, v_operand, tla.CopyL1ToL0Params(
                                    k_part_idx=chunk, k_part=PV_K, k_part_ceil=PV_CHUNKS[chunk],
                                    k_part_loop=2, k_part_actual=PV_CHUNKS[chunk], m=PV_N))
                                if MANUAL_SYNC:
                                    if band == 1 and chunk == 1:
                                        tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.MTE2, (step + 2) % 3)
                                    tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, pv_bank)
                                    tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, pv_bank)
                                tla.cube.mmad_l0(p_operand, v_operand, product, m=HEAD_TILE, n=PV_N, k=PV_CHUNKS[chunk],
                                                 clear_accum=chunk == 0)
                                if MANUAL_SYNC:
                                    tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, pv_bank)
                            if EARLY_BANDS:
                                if MANUAL_SYNC:
                                    tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, pv_event)
                                    tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, pv_event)
                                if step > 1:
                                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, BAND_FREE[band])
                                band_partial = tla.view(partial, (ROWS, PV_N), offsets=(band * ROWS, 0))
                                tla.copy_l0c_to_ub(product, band_partial, tla.FixpipeL0C2UBParams(
                                    dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                    sub_block_idx=None))
                                if MANUAL_SYNC:
                                    tla.sync.set_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, pv_event)
                                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, BAND_READY[band])
                    else:
                        tla.cube.mmad(probability_tile, value, pv, clear_accum=True)
                    if not THREE_SLOTS and has_qk:
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.M, PROB_FREE)
                if PREFETCH_KV and step + 1 < num_tiles:
                    # PV has issued its last L1 read. The next DMA can use
                    # that slot while Cube drains L0 and Fixpipe publishes PV.
                    physical_next = tl.load(TABLE + batch * TABLE_STRIDE + first_page + step + 1).to(tl.int64)
                    next_key = tla.view(pages, PAGE_L1_SHAPE, offsets=((1 - slot) * (WIDTH // 16), 0, 0, 0))
                    tla.load_gm_to_l1(tla.view(KV + physical_next * PAGE * WIDTH, (PAGE, WIDTH)), next_key,
                                      tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2))
                if has_pv and not EARLY_BANDS:
                    if MANUAL_SYNC:
                        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 1)
                        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 1)
                    if step > 1:
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK,
                                                 tla.sync.PIPE.F if MANUAL_SYNC else tla.sync.PIPE.S, PRODUCT_FREE)
                    tla.copy_l0c_to_ub(pv, partial, tla.FixpipeL0C2UBParams(
                        dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                        sub_block_idx=None))
                    if MANUAL_SYNC:
                        for band in tl.static_range(2):
                            tla.sync.set_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, band + 1)
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, PRODUCT_READY)
            with tla.scope.vector():
                if has_qk:
                    first_row = request_row + sub * ROWS
                    remaining = (length - SQ + 1 - first_token - step * KV_TILE + first_row // H).to(tl.int32)
                    correction = tla.view(corrections, (64,), offsets=(correction_slot * 64,))
                    if LATE_SUM or VECTOR_SUM:
                        tile_sum = tla.view(tile_sums, (64,), offsets=(correction_slot * 64,))
                    if MANUAL_SYNC:
                        tla.sync.wait_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 0)
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK,
                                             tla.sync.PIPE.V if MANUAL_SYNC else tla.sync.PIPE.S, SCORE_READY)
                    for MASKED in tl.static_range(0 if SKIP_FULL_MASK else 1, 2):
                        if not SKIP_FULL_MASK or (remaining < KV_TILE) == MASKED:
                            QUERY_ROWS: tl.constexpr = H if MASKED and H % HEAD_TILE != 0 else ROWS
                            with tla.scope.simd():
                                if MASKED or (THREE_SLOTS and not PRESET_SCORE_TAIL):
                                    neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                                for query in tl.static_range(ROWS // QUERY_ROWS):
                                    if MASKED:
                                        end = tla.simd.vdup(remaining + query, tl.int32)
                                        valid0 = tla.simd.vcmp(tla.simd.vci(0), end, op=tla.simd.CMP.LT)
                                    if THREE_SLOTS and (MASKED or not PRESET_SCORE_TAIL):
                                        end1 = tl.minimum(remaining + query, KV_TILE) if MASKED else KV_TILE
                                        tail = tla.simd.vdup(end1, tl.int32)
                                        valid1 = tla.simd.vcmp(tla.simd.vci(64), tail, op=tla.simd.CMP.LT)
                                    elif MASKED:
                                        valid1 = tla.simd.vcmp(tla.simd.vci(64), end, op=tla.simd.CMP.LT)
                                    for i in tl.range(QUERY_ROWS):
                                        row = query * QUERY_ROWS + i
                                        x0 = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                                        x1 = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                                        if MASKED:
                                            x0 = tla.simd.vsel(valid0, x0, neg)
                                        if MASKED or (THREE_SLOTS and not PRESET_SCORE_TAIL):
                                            x1 = tla.simd.vsel(valid1, x1, neg)
                                        old_max = tla.simd.vlds(maximum[row], dist='BRC_B32')
                                        new_max = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_max)
                                        alpha = tla.simd.vexpdif(old_max, new_max)
                                        e0 = tla.simd.vexpdif(x0, new_max)
                                        e1 = tla.simd.vexpdif(x1, new_max)
                                        total = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(e0, e1)))
                                        tla.simd.vsts(maximum[row], new_max, dist='ONEPT_B32')
                                        if LATE_SUM or VECTOR_SUM:
                                            tla.simd.vsts(tile_sum[row], total, dist='ONEPT_B32')
                                        else:
                                            old_sum = tla.simd.vlds(denominator[row], dist='BRC_B32')
                                            tla.simd.vsts(denominator[row], tla.simd.vfma(total, old_sum, alpha),
                                                          dist='ONEPT_B32')
                                        tla.simd.vsts(correction[row], alpha, dist='ONEPT_B32')
                                        p0 = tla.simd.vtruncf(e0, tl.bfloat16)
                                        p1 = tla.simd.vtruncf(e1, tl.bfloat16)
                                        interleaved, _ = tla.simd.vdintlv(p0, p1)
                                        tla.simd.vsstb(interleaved, packed_p[0, row * 16], (ROWS + 1) * 16)
                                if VECTOR_SUM:
                                    # Update every row's online denominator in one register.
                                    tla.simd.membar('VST_VLD')
                                    row_mask = tla.simd.vcmp(tla.simd.vci(0), tla.simd.vfull(ROWS, tl.int32, 64),
                                                             op=tla.simd.CMP.LT)
                                    running = tla.simd.vfma(tla.simd.vlds(tile_sum[0]),
                                                           tla.simd.vlds(denominator[0]),
                                                           tla.simd.vlds(correction[0]))
                                    tla.simd.vsts(denominator[0], running, mask=row_mask)
                    if step + 1 < num_tiles:
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, SCORE_FREE)
                    if not THREE_SLOTS and has_pv:
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, PROB_FREE)
                    if THREE_SLOTS:
                        # QK has consumed the packed RoPE columns. P now
                        # takes the same address range without touching V.
                        destination = tla.reshape(tla.view(slots, (HEAD_TILE * KV_TILE,),
                                                           offsets=(slot * SLOT_ELEMS + LATENT * KV_TILE,)),
                                                  (KV_TILE // 16, HEAD_TILE * 16))
                    else:
                        destination = probability_flat
                    if MANUAL_SYNC:
                        tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 0)
                        tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 0)
                    tla.copy_ub_to_l1(tla.view(packed_p, (KV_TILE // 16, ROWS * 16)),
                                      tla.view(destination, (KV_TILE // 16, ROWS * 16),
                                               offsets=(0, sub * ROWS * 16)))
                    if MANUAL_SYNC:
                        tla.sync.set_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 0)
                        if DYNAMIC_BLOCK_FLAGS:
                            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, PROB_FLAG_BASE + slot)
                        else:
                            for ready in tl.static_range(3):
                                if slot == ready:
                                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3,
                                                            PROB_SLOT_READY[ready])
                    else:
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, PROB_READY)
                if has_pv:
                    correction = tla.view(corrections, (64,), offsets=((1 - correction_slot) * 64,))
                    if LATE_SUM:
                        tile_sum = tla.view(tile_sums, (64,), offsets=((1 - correction_slot) * 64,))
                    for band in tl.static_range(2 if EARLY_BANDS else 1):
                        if EARLY_BANDS:
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, BAND_READY[band])
                            result = tla.view(partial, (ROWS, PV_N), offsets=(band * ROWS, 0))
                        else:
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK,
                                                     tla.sync.PIPE.V if MANUAL_SYNC else tla.sync.PIPE.S, PRODUCT_READY)
                            result = partial
                        if FIRST_PV_COPY and step == 1:
                            # The first PV contributes the entire numerator so far.
                            with tla.scope.simd():
                                for row in tl.range(ROWS):
                                    for col in tl.static_range(0, PV_N, 64):
                                        tla.simd.vsts(accumulator[row, band * PV_N + col],
                                                      tla.simd.vlds(result[row, col]))
                        else:
                            with tla.scope.simd():
                                for row in tl.range(ROWS):
                                    alpha = tla.simd.vlds(correction[row], dist='BRC_B32')
                                    if LATE_SUM and band == 0:
                                        total = tla.simd.vlds(tile_sum[row], dist='BRC_B32')
                                        old_sum = tla.simd.vlds(denominator[row], dist='BRC_B32')
                                        tla.simd.vsts(denominator[row], tla.simd.vfma(total, old_sum, alpha),
                                                      dist='ONEPT_B32')
                                    for col in tl.static_range(0, PV_N if EARLY_BANDS else LATENT, 64):
                                        update = tla.simd.vfma(tla.simd.vlds(result[row, col]),
                                                               tla.simd.vlds(accumulator[row, band * PV_N + col]), alpha)
                                        tla.simd.vsts(accumulator[row, band * PV_N + col], update)
                        if has_qk:
                            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, BAND_FREE[band])

        with tla.scope.vector():
            if MANUAL_SYNC:
                tla.sync.wait_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 0)
                tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 0)
                tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 0)
            output_row = q_row + sub * ROWS
            if FUSE_MERGE and MERGE_MAJOR:
                tl.static_assert(ROWS % MERGE_ROWS == 0)
                # The consumer reads every split of one row group together.
                # Each producer writes its split into the group-strided view.
                workspace_row = (output_row // MERGE_ROWS * SPLITS + split) * MERGE_ROWS
                values: tl.constexpr = (ROWS // MERGE_ROWS, MERGE_ROWS, LATENT)
                stats: tl.constexpr = (ROWS // MERGE_ROWS, MERGE_ROWS)
                stats_stride: tl.constexpr = (SPLITS * MERGE_ROWS, 1)
                tla.store_ub_to_gm(
                    tla.reshape(accumulator, values),
                    tla.view(NUMERATOR + workspace_row * LATENT, values,
                             strides=(SPLITS * MERGE_ROWS * LATENT, LATENT, 1)))
                tla.store_ub_to_gm(tla.reshape(tla.view(maximum, (ROWS,)), stats),
                                   tla.view(MAXIMUM + workspace_row, stats, strides=stats_stride))
                tla.store_ub_to_gm(tla.reshape(tla.view(denominator, (ROWS,)), stats),
                                   tla.view(DENOMINATOR + workspace_row, stats, strides=stats_stride))
            else:
                workspace_row = split * ALL_ROWS + output_row
                tla.store_ub_to_gm(accumulator, tla.view(NUMERATOR + workspace_row * LATENT, (ROWS, LATENT)))
                tla.store_ub_to_gm(tla.view(maximum, (ROWS,)), tla.view(MAXIMUM + workspace_row, (ROWS,)))
                tla.store_ub_to_gm(tla.view(denominator, (ROWS,)), tla.view(DENOMINATOR + workspace_row, (ROWS,)))
            if not FAST_EXIT or job + job_stride < job_end:
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, JOB_DONE)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, JOB_ACK)
        with tla.scope.cube():
            if MANUAL_SYNC:
                for event in tl.static_range(3):
                    tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.MTE2, event)
                for event in tl.static_range(L0C_EVENTS):
                    tla.sync.wait_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, event)
                for event in tl.static_range(2):
                    tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, event)
            if not FAST_EXIT or job + job_stride < job_end:
                # No further job can reuse Cube or Vector storage after the
                # last PV. The merge fence still waits for every MTE3 store.
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, JOB_DONE)
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, JOB_ACK)

    if FUSE_MERGE:
        # The mixed grid launches every AIV. Wait until all split workspaces
        # are visible before redistributing the contiguous output row tiles.
        with tla.scope.vector():
            tla.sync.barrier(tla.sync.PIPE.MTE3)
            tla.sync.sync_all(False)
            worker = tl.program_id(0) * 2 + sub.to(tl.int32)
            # The planner retains some attention buffers through the fence.
            ROW_TILE: tl.constexpr = MERGE_ROWS
            STAT_VALUES: tl.constexpr = SPLITS * ROW_TILE
            STAT_ELEMENTS: tl.constexpr = STAT_VALUES + (64 - ROW_TILE if VECTOR_MERGE_STATS else 0)
            tl.static_assert(ALL_ROWS % ROW_TILE == 0)
            if MERGE_ARENA:
                if STREAM_MERGE:
                    numerators = tla.reshape(tla.view(float_arena, (2 * ROW_TILE * LATENT,)), (2, ROW_TILE, LATENT))
                    reduced = tla.reshape(tla.view(float_arena, (ROW_TILE * LATENT,), offsets=(2 * ROW_TILE * LATENT,)),
                                          (ROW_TILE, LATENT))
                else:
                    numerators = tla.reshape(tla.view(float_arena, (MERGE_NUM_WORDS,)), (SPLITS, ROW_TILE, LATENT))
                maxima = tla.view(float_arena, (STAT_ELEMENTS,), offsets=(MERGE_NUM_WORDS,))
                sums = tla.view(float_arena, (STAT_ELEMENTS,), offsets=(MERGE_NUM_WORDS + MERGE_READ_WORDS,))
                weights = tla.reshape(tla.view(float_arena, (STAT_VALUES,), offsets=(WEIGHT_OFFSET,)), (SPLITS, ROW_TILE))
                merged_output = tla.reshape(tla.view(bf16_arena, (ROW_TILE * LATENT,)), (ROW_TILE, LATENT))
            else:
                maxima = tla.alloc((STAT_ELEMENTS,), tl.float32, space='ub')
                sums = tla.alloc((STAT_ELEMENTS,), tl.float32, space='ub')
                numerators = tla.alloc((SPLITS, ROW_TILE, LATENT), tl.float32, space='ub')
                weights = tla.alloc((SPLITS, ROW_TILE), tl.float32, space='ub')
                merged_output = tla.alloc((ROW_TILE, LATENT), tl.bfloat16, space='ub')
            if VECTOR_MERGE_STATS:
                if MERGE_ARENA:
                    inverse_sums = tla.view(float_arena, (ROW_TILE,), offsets=(INVERSE_OFFSET,))
                else:
                    inverse_sums = tla.alloc((ROW_TILE,), tl.float32, space='ub')
                # Loads read a full register but only the first ROW_TILE lanes
                # belong to the current split. Pad the last load once.
                with tla.scope.simd():
                    zeros = tla.simd.vfull(0.0, tl.float32, 64)
                    tla.simd.vsts(maxima[(SPLITS - 1) * ROW_TILE], zeros)
                    tla.simd.vsts(sums[(SPLITS - 1) * ROW_TILE], zeros)
                tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE2, 2)
                tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE2, 2)
            if RETURN_LSE:
                if MERGE_ARENA:
                    result_lse = tla.view(float_arena, (ROW_TILE,), offsets=(LSE_OFFSET,))
                else:
                    result_lse = tla.alloc((ROW_TILE,), tl.float32, space='ub')

            for merge_tile in tl.range(worker, ALL_ROWS // ROW_TILE, 2 * CORES):
                row_start = merge_tile * ROW_TILE
                if STREAM_K:
                    request_first_tile = row_start // HEAD_TILE * REQUEST_K_TILES
                    request_last_tile = request_first_tile + REQUEST_K_TILES - 1
                    merge_splits = request_last_tile // CORE_K_TILES - request_first_tile // CORE_K_TILES + 1
                else:
                    merge_splits = SPLITS
                if MERGE_MAJOR:
                    start = merge_tile * SPLITS * ROW_TILE
                    tla.load_gm_to_ub(tla.view(MAXIMUM + start, (STAT_VALUES,)), tla.view(maxima, (STAT_VALUES,)))
                    tla.load_gm_to_ub(tla.view(DENOMINATOR + start, (STAT_VALUES,)), tla.view(sums, (STAT_VALUES,)))
                    if not STREAM_MERGE:
                        span: tl.constexpr = SPLITS * ROW_TILE * LATENT
                        tla.load_gm_to_ub(tla.view(NUMERATOR + start * LATENT, (span,)),
                                          tla.reshape(numerators, (span,)))
                else:
                    for merge_split in tl.range(SPLITS):
                        start = merge_split * ALL_ROWS + row_start
                        stat = merge_split * ROW_TILE
                        tla.load_gm_to_ub(tla.view(MAXIMUM + start, (ROW_TILE,)),
                                          tla.view(maxima, (ROW_TILE,), offsets=(stat,)))
                        tla.load_gm_to_ub(tla.view(DENOMINATOR + start, (ROW_TILE,)),
                                          tla.view(sums, (ROW_TILE,), offsets=(stat,)))
                        tla.load_gm_to_ub(tla.view(NUMERATOR + start * LATENT, (1, ROW_TILE, LATENT)),
                                          tla.view(numerators, (1, ROW_TILE, LATENT), offsets=(merge_split, 0, 0)))
                tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 2)
                if STREAM_MERGE:
                    # FlashDecode prefetches split numerators while V computes weights.
                    for preload_bank in tl.static_range(min(2, SPLITS)):
                        if not STREAM_K or preload_bank < merge_splits:
                            preload_source = NUMERATOR + (start + preload_bank * ROW_TILE) * LATENT
                            tla.load_gm_to_ub(tla.view(preload_source, (1, ROW_TILE, LATENT)),
                                              tla.view(numerators, (1, ROW_TILE, LATENT), offsets=(preload_bank, 0, 0)))
                            tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, preload_bank + 3)
                tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 2)
                with tla.scope.simd():
                    zero = tla.simd.vfull(0.0, tl.float32, 64)
                    one = tla.simd.vfull(1.0, tl.float32, 64)
                    if VECTOR_MERGE_STATS:
                        # One lane per output row computes all split weights.
                        # Numerator accumulation below still works by columns.
                        merge_mask = tla.simd.pge(ROW_TILE, 64)
                        row_maxima = tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64)
                        for merge_split in tl.range(merge_splits):
                            stat = merge_split * ROW_TILE
                            row_maxima = tla.simd.vmax(row_maxima, tla.simd.vlds(maxima[stat]), mask=merge_mask)
                        row_sums = zero
                        for merge_split in tl.range(merge_splits):
                            stat = merge_split * ROW_TILE
                            weight = tla.simd.vexpdif(tla.simd.vlds(maxima[stat]), row_maxima, mask=merge_mask)
                            row_sums = tla.simd.vfma(row_sums, tla.simd.vlds(sums[stat]), weight)
                            tla.simd.vsts(weights[merge_split, 0], weight, mask=merge_mask)
                        tla.simd.vsts(inverse_sums[0], tla.simd.vdiv(one, row_sums, mask=merge_mask), mask=merge_mask)
                        if RETURN_LSE:
                            lse = tla.simd.vadd(row_maxima, tla.simd.vln(row_sums, mask=merge_mask), mask=merge_mask)
                            tla.simd.vsts(result_lse[0], lse, mask=merge_mask)
                        tla.simd.membar('VST_VLD')
                    if not STREAM_MERGE:
                        for merge_row in tl.range(ROW_TILE):
                            if VECTOR_MERGE_STATS:
                                inv_denominator = tla.simd.vlds(inverse_sums[merge_row], dist='BRC_B32')
                            else:
                                max_row = tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64)
                                for merge_split in tl.range(SPLITS):
                                    stat = merge_split * ROW_TILE + merge_row
                                    max_row = tla.simd.vmax(max_row, tla.simd.vlds(maxima[stat], dist='BRC_B32'))
                                denominator_vec = zero
                                for merge_split in tl.range(SPLITS):
                                    stat = merge_split * ROW_TILE + merge_row
                                    split_max = tla.simd.vlds(maxima[stat], dist='BRC_B32')
                                    split_sum = tla.simd.vlds(sums[stat], dist='BRC_B32')
                                    weight = tla.simd.vexpdif(split_max, max_row)
                                    denominator_vec = tla.simd.vfma(denominator_vec, split_sum, weight)
                                    tla.simd.vsts(weights[merge_split, merge_row], weight, dist='ONEPT_B32')
                                if RETURN_LSE:
                                    lse = tla.simd.vadd(max_row, tla.simd.vln(denominator_vec))
                                    tla.simd.vsts(result_lse[merge_row], lse, dist='ONEPT_B32')
                                tla.simd.membar('VST_VLD')
                                inv_denominator = tla.simd.vdiv(one, denominator_vec)
                            for col in tl.static_range(0, LATENT, 128):
                                lo = zero
                                hi = zero
                                for merge_split in tl.range(SPLITS):
                                    weight = tla.simd.vlds(weights[merge_split, merge_row], dist='BRC_B32')
                                    lo = tla.simd.vfma(lo, tla.simd.vlds(numerators[merge_split, merge_row, col]), weight)
                                    hi = tla.simd.vfma(hi, tla.simd.vlds(numerators[merge_split, merge_row, col + 64]), weight)
                                lo_bf16 = tla.simd.vtruncf(tla.simd.vmul(lo, inv_denominator), tl.bfloat16)
                                hi_bf16 = tla.simd.vtruncf(tla.simd.vmul(hi, inv_denominator), tl.bfloat16)
                                packed, _ = tla.simd.vdintlv(lo_bf16, hi_bf16)
                                tla.simd.vsts(merged_output[merge_row, col], packed, dist='NORM_B16')
                if STREAM_MERGE:
                    # Each input bank can be refilled as soon as V has read it.
                    with tla.scope.simd():
                        zero = tla.simd.vfull(0.0, tl.float32, 64)
                        for row in tl.range(ROW_TILE):
                            for col in tl.static_range(0, LATENT, 64):
                                tla.simd.vsts(reduced[row, col], zero)
                    for merge_split in tl.range(merge_splits):
                        merge_bank = merge_split % 2
                        event = merge_bank + 3
                        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, event)
                        with tla.scope.simd():
                            for row in tl.range(ROW_TILE):
                                weight = tla.simd.vlds(weights[merge_split, row], dist='BRC_B32')
                                for col in tl.static_range(0, LATENT, 64):
                                    value = tla.simd.vlds(numerators[merge_bank, row, col])
                                    previous = tla.simd.vlds(reduced[row, col])
                                    tla.simd.vsts(reduced[row, col], tla.simd.vfma(previous, value, weight))
                        if merge_split + 2 < merge_splits:
                            tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE2, event)
                            tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE2, event)
                            refill_source = NUMERATOR + (start + (merge_split + 2) * ROW_TILE) * LATENT
                            tla.load_gm_to_ub(tla.view(refill_source, (1, ROW_TILE, LATENT)),
                                              tla.view(numerators, (1, ROW_TILE, LATENT), offsets=(merge_bank, 0, 0)))
                            tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, event)
                    with tla.scope.simd():
                        for row in tl.range(ROW_TILE):
                            inv_denominator = tla.simd.vlds(inverse_sums[row], dist='BRC_B32')
                            for col in tl.static_range(0, LATENT, 128):
                                lo = tla.simd.vmul(tla.simd.vlds(reduced[row, col]), inv_denominator)
                                hi = tla.simd.vmul(tla.simd.vlds(reduced[row, col + 64]), inv_denominator)
                                packed, _ = tla.simd.vdintlv(tla.simd.vtruncf(lo, tl.bfloat16),
                                                             tla.simd.vtruncf(hi, tl.bfloat16))
                                tla.simd.vsts(merged_output[row, col], packed, dist='NORM_B16')
                tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 2)
                tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 2)
                tla.store_ub_to_gm(merged_output, tla.view(OUT + row_start * LATENT, (ROW_TILE, LATENT)))
                if RETURN_LSE:
                    tla.store_ub_to_gm(result_lse, tla.view(LSE + row_start, (ROW_TILE,)))
                tla.sync.barrier(tla.sync.PIPE.ALL)
