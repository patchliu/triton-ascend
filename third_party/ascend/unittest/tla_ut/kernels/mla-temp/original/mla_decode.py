"""Paged BF16 MLA decode with a shared latent K/V tile and split-KV merge."""

import math

import triton
import triton.language as tl
import triton.language.extra.tla as tla


SCHEDULES = {
    'serial': (0, 0, 0),
    'prefetch': (1, 1, 1),
    'early_qk': (0, 1, 1),
    'late_pv': (0, 0, 1),
    'overlap': (0, 1, 2),
}
FUSED_MERGE_MODES = ('native112_merge', 'native112_merge_wide', 'native112_merge_sum',
                     'native112_merge_vec', 'native112_balanced',
                     'native112_merge_banks', 'native112_merge_dma_banks',
                     'native112_merge_tail', 'native112_merge_layout',
                     'native112_merge_arena', 'native112_merge_arena_banks', 'native112_merge_arena_vec',
                     'native112_merge_qreuse', 'native112_merge_fast_exit', 'native112_merge_dynflags',
                     'native112_merge_dynflags_phase', 'native112_merge_dynflags_shared',
                     'native112_merge_dynflags_first_pv', 'native112_merge_dynflags_balanced',
                     'native112_merge_dynflags_dma', 'native112_merge_dynflags_stream',
                     'native112_merge_dynflags_vecsum', 'native112_merge_dynflags_hybrid',
                     'native112_merge_dynflags_stream_k')
FUSED_MERGE_MAX_SPLITS = 7
MANUAL_L0_MODES = ('native112_manual', 'native112_phase', 'native112_shared_p', 'native112_fused_pv',
                   'native112_manual_dma', 'native112_late_sum', 'native112_l0c') + FUSED_MERGE_MODES
L1_112_MODES = ('l1_112', 'l1_112_dynamic')
DYNAMIC_ONLY_MODES = ('l1_112_dynamic', 'native112_dynamic')
DYNAMIC_112_MODES = DYNAMIC_ONLY_MODES + ('native112_manual_dma', 'native112_merge_dma_banks',
                                        'native112_merge_dynflags_dma', 'native112_merge_dynflags_vecsum',
                                        'native112_merge_dynflags_hybrid', 'native112_merge_dynflags_stream_k')
BASE_112_MODES = ('l1_112', 'native112', 'native112_k5', 'native112_bands')
KV112_MODES = BASE_112_MODES + MANUAL_L0_MODES + DYNAMIC_ONLY_MODES
BASE_L0_MODES = ('reload', 'q', 'qp', 'prefetch', 'pingpong') + BASE_112_MODES
L0_MODES = BASE_L0_MODES + MANUAL_L0_MODES + DYNAMIC_ONLY_MODES


@tla.jit
def mla_split(Q, QR, KV, KR, TABLE, LENGTHS, NUMERATOR, MAXIMUM, DENOMINATOR,
              B: tl.constexpr, SQ: tl.constexpr, H: tl.constexpr, TABLE_STRIDE: tl.constexpr,
              SPLIT_PAGES: tl.constexpr, SPLITS: tl.constexpr, CORES: tl.constexpr,
              STAGES: tl.constexpr, SCALE: tl.constexpr, HEAD_TILE: tl.constexpr, SKIP_FULL_MASK: tl.constexpr,
              KV_MAJOR: tl.constexpr, Q_GROUP: tl.constexpr = 1, WIDE_QK: tl.constexpr = False,
              PACKED: tl.constexpr = False, KV_L2: tl.constexpr = None, SPLIT_PV: tl.constexpr = False,
              KV_TILE: tl.constexpr = 128):
    PAGE: tl.constexpr = 128
    GROUP_ROWS: tl.constexpr = HEAD_TILE * Q_GROUP
    PARTS: tl.constexpr = 1 if WIDE_QK else Q_GROUP
    TILE_ROWS: tl.constexpr = GROUP_ROWS // PARTS
    ROWS: tl.constexpr = TILE_ROWS // 2
    STATE_ROWS: tl.constexpr = ROWS * PARTS
    QUERY_TILES: tl.constexpr = SQ * H // GROUP_ROWS
    ALL_ROWS: tl.constexpr = B * SQ * H
    LATENT: tl.constexpr = 512
    ROPE: tl.constexpr = 64
    KEY_DIM: tl.constexpr = LATENT + ROPE if PACKED else LATENT
    VALUE_TILE: tl.constexpr = LATENT // 2 if WIDE_QK or SPLIT_PV else LATENT
    VALUE_PARTS: tl.constexpr = LATENT // VALUE_TILE
    PV_TASKS: tl.constexpr = PARTS * VALUE_PARTS
    Q_LATENT_NZ: tl.constexpr = (KEY_DIM // 16, TILE_ROWS // 16, 16, 16)
    Q_ROPE_NZ: tl.constexpr = (ROPE // 16, TILE_ROWS // 16, 16, 16)
    SHARED_KVP: tl.constexpr = KV_TILE == 112
    DOUBLE_PV_HANDOFF: tl.constexpr = SPLIT_PV and not SHARED_KVP
    KV_LATENT_NZ: tl.constexpr = (KEY_DIM // 16, KV_TILE // 16, 16, 16)
    K_ROPE_NZ: tl.constexpr = (ROPE // 16, PAGE // 16, 16, 16)
    PROB_NZ: tl.constexpr = (KV_TILE // 16, TILE_ROWS // 16, 16, 16)
    KVP_ELEMENTS: tl.constexpr = KV_TILE * (LATENT + max(TILE_ROWS, ROPE))
    tl.static_assert(HEAD_TILE == 32 or HEAD_TILE == 64 or HEAD_TILE == 96)
    tl.static_assert((Q_GROUP == 1 or Q_GROUP == 2) and SQ * H % GROUP_ROWS == 0)
    tl.static_assert(STATE_ROWS <= 64)
    tl.static_assert(H % TILE_ROWS == 0 or ROWS % H == 0)
    tl.static_assert(not WIDE_QK or Q_GROUP == 2)
    tl.static_assert(not SPLIT_PV or Q_GROUP == 1)
    tl.static_assert(KV_TILE == PAGE or (SHARED_KVP and PACKED and Q_GROUP == 1 and SPLIT_PAGES % 7 == 0))

    for job in tl.range(tl.program_id(0), B * QUERY_TILES * SPLITS, CORES):
        batch = job // (QUERY_TILES * SPLITS)
        if KV_MAJOR:
            # Run the Q tiles for a given request and KV split together.
            # Their page loads can share L2, even across Cube cores.
            split = job // QUERY_TILES % SPLITS
            tile = job % QUERY_TILES
        else:
            split = job % SPLITS
            tile = job // SPLITS % QUERY_TILES
        first_page = split * SPLIT_PAGES
        num_pages = tl.minimum(SPLIT_PAGES, TABLE_STRIDE - first_page)
        request_row = tile * GROUP_ROWS
        q_row = batch * SQ * H + request_row
        sub = tla.sub_block_id()
        first_token = first_page * PAGE
        # Keep length reads in the owning tasks. Beyond the visible KV
        # length, clamped loads and the causal predicates make a split empty.
        num_tiles = tl.cdiv(num_pages * PAGE, KV_TILE) if SHARED_KVP else num_pages

        q_lat = tla.alloc((PARTS * KEY_DIM // 16, TILE_ROWS // 16, 16, 16), tl.bfloat16, space='l1')
        if not PACKED:
            q_rope = tla.alloc((PARTS * ROPE // 16, TILE_ROWS // 16, 16, 16), tl.bfloat16, space='l1')
        max_state = tla.alloc((64,), tl.float32, space='ub')
        sum_state = tla.alloc((64,), tl.float32, space='ub')
        accumulator = tla.alloc((STATE_ROWS, LATENT), tl.float32, space='ub')
        with tla.scope.cube():
            for part in tl.static_range(PARTS):
                row = q_row + part * TILE_ROWS
                tla.load_gm_to_l1(tla.view(Q + row * KEY_DIM, (TILE_ROWS, KEY_DIM)),
                                  tla.view(q_lat, Q_LATENT_NZ, offsets=(part * (KEY_DIM // 16), 0, 0, 0)))
                if not PACKED:
                    tla.load_gm_to_l1(tla.view(QR + row * ROPE, (TILE_ROWS, ROPE)),
                                      tla.view(q_rope, Q_ROPE_NZ, offsets=(part * (ROPE // 16), 0, 0, 0)))
        with tla.scope.vector():
            with tla.scope.simd():
                # Finite sentinel keeps exp(-inf - max) equal to zero in empty splits.
                tla.simd.vsts(max_state[0], tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64))
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                tla.simd.vsts(sum_state[0], zero)
                for row in tl.range(STATE_ROWS):
                    for col in tl.static_range(LATENT // 64):
                        tla.simd.vsts(accumulator[row, col * 64], zero)

        for kv_index in tla.pipeline(num_tiles):
            token = first_token + kv_index * KV_TILE
            page = token // PAGE if SHARED_KVP else first_page + kv_index
            if SHARED_KVP:
                # QK reads RoPE before the Vector task replaces its bytes with P.
                # The shared root spans all three stages, including the latent V read.
                kvp = tla.alloc((KVP_ELEMENTS,), tl.bfloat16, space='l1')
                kv_lat = tla.reshape(tla.view(kvp, (KV_TILE * KEY_DIM,)), KV_LATENT_NZ)
                probability = tla.reshape(tla.view(kvp, (KV_TILE * TILE_ROWS,), offsets=(KV_TILE * LATENT,)), PROB_NZ)
            else:
                kv_lat = tla.alloc(KV_LATENT_NZ, tl.bfloat16, space='l1')
                probability = tla.alloc((PARTS * KV_TILE // 16, TILE_ROWS // 16, 16, 16), tl.bfloat16, space='l1')
            if not PACKED:
                k_rope = tla.alloc(K_ROPE_NZ, tl.bfloat16, space='l1')
            qk = tla.alloc(PROB_NZ, tl.float32, space='l0c')
            pv = tla.alloc((VALUE_TILE // 16, TILE_ROWS // 16, 16, 16), tl.float32, space='l0c')
            score = tla.alloc((STATE_ROWS, PAGE), tl.float32, space='ub')
            packed_p = tla.alloc((PAGE // 16, (ROWS + 1) * 16), tl.bfloat16, space='ub')
            partial = tla.alloc((ROWS, VALUE_TILE), tl.float32, space='ub')
            if SPLIT_PV:
                pv_second = tla.alloc((VALUE_TILE // 16, TILE_ROWS // 16, 16, 16), tl.float32, space='l0c')
            if DOUBLE_PV_HANDOFF:
                partial_second = tla.alloc((ROWS, VALUE_TILE), tl.float32, space='ub')
            correction = tla.alloc((64,), tl.float32, space='ub')

            with tla.scope.cube().task('load_page', stage=0):
                length = tl.load(LENGTHS + batch)
                # A shared page can be visible to the latest query only.
                if GROUP_ROWS % H == 0:
                    last_query = (tile + 1) * (GROUP_ROWS // H) - 1
                else:
                    last_query = ((request_row + GROUP_ROWS - 1).to(tl.int32)) // H
                last_page = (length - SQ + last_query) // PAGE
                physical = tl.load(TABLE + batch * TABLE_STRIDE + tl.minimum(page, last_page)).to(tl.int64)
                if SHARED_KVP:
                    in_page = token % PAGE
                    first_count = tl.minimum(KV_TILE, PAGE - in_page)
                    for fragment in tl.static_range(16, KV_TILE + 1, 16):
                        if first_count == fragment:
                            first = tla.view(kv_lat, (KEY_DIM // 16, fragment // 16, 16, 16))
                            tla.load_gm_to_l1(tla.view(KV + (physical * PAGE + in_page) * KEY_DIM,
                                                       (fragment, KEY_DIM)), first,
                                              tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2))
                            if fragment < KV_TILE:
                                next_page = tl.minimum(page + 1, last_page)
                                second = tl.load(TABLE + batch * TABLE_STRIDE + next_page).to(tl.int64)
                                tail = tla.view(kv_lat, (KEY_DIM // 16, (KV_TILE - fragment) // 16, 16, 16),
                                                offsets=(0, fragment // 16, 0, 0))
                                tla.load_gm_to_l1(tla.view(KV + second * PAGE * KEY_DIM,
                                                           (KV_TILE - fragment, KEY_DIM)), tail,
                                                  tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2))
                else:
                    tla.load_gm_to_l1(tla.view(KV + physical * PAGE * KEY_DIM, (PAGE, KEY_DIM)), kv_lat,
                                      tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2))
                if not PACKED:
                    tla.load_gm_to_l1(tla.view(KR + physical * PAGE * ROPE, (PAGE, ROPE)), k_rope)

            # Parts are logical row tiles, sharing one loaded page. The compiler
            # owns all iteration overlap, scratch versioning and synchronization.
            for part in tl.static_range(PARTS):
                with tla.scope.cube().task(f'qk_{part}' if PARTS > 1 else 'qk', stage=STAGES[0]):
                    latent = tla.view(q_lat, Q_LATENT_NZ, offsets=(part * (KEY_DIM // 16), 0, 0, 0))
                    tla.cube.mmad(latent, kv_lat, qk, transpose_b=True, clear_accum=True)
                    if not PACKED:
                        rope = tla.view(q_rope, Q_ROPE_NZ, offsets=(part * (ROPE // 16), 0, 0, 0))
                        tla.cube.mmad(rope, k_rope, qk, transpose_b=True, clear_accum=False)
                    tla.copy_l0c_to_ub(
                        qk, tla.view(score, (ROWS, KV_TILE), offsets=(part * ROWS, 0)),
                        tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            for part in tl.static_range(PARTS):
                with tla.scope.vector().task(f'softmax_{part}' if PARTS > 1 else 'softmax', stage=STAGES[1]):
                    length = tl.load(LENGTHS + batch)
                    causal_base = (length - SQ + 1 - token).to(tl.int32)
                    if H % TILE_ROWS == 0:
                        row_start = request_row + part * TILE_ROWS + sub * ROWS
                        remaining = causal_base + row_start.to(tl.int32) // H
                    else:
                        remaining = causal_base + tile * (GROUP_ROWS // H) + part * (TILE_ROWS // H) + sub * (ROWS // H)
                    # Only the tail or an empty split needs predicates.
                    for MASKED in tl.static_range(0 if SKIP_FULL_MASK else 1, 2):
                        if not SKIP_FULL_MASK or (remaining < KV_TILE) == MASKED:
                            MASK_ROWS: tl.constexpr = H if MASKED and H % TILE_ROWS != 0 else ROWS
                            with tla.scope.simd():
                                if MASKED or SHARED_KVP:
                                    neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                                # Each contiguous head group shares a single causal predicate.
                                for query in tl.static_range(ROWS // MASK_ROWS):
                                    if MASKED:
                                        end = tla.simd.vdup(remaining + query, tl.int32)
                                        valid0 = tla.simd.vcmp(tla.simd.vci(0), end, op=tla.simd.CMP.LT)
                                    if SHARED_KVP:
                                        end1 = tl.minimum(remaining + query, KV_TILE) if MASKED else KV_TILE
                                        valid1 = tla.simd.vcmp(tla.simd.vci(64), tla.simd.vdup(end1, tl.int32),
                                                               op=tla.simd.CMP.LT)
                                    elif MASKED:
                                        valid1 = tla.simd.vcmp(tla.simd.vci(64), end, op=tla.simd.CMP.LT)
                                    for row_in_query in tl.range(MASK_ROWS):
                                        row = query * MASK_ROWS + row_in_query
                                        state = part * ROWS + row
                                        x0 = tla.simd.vmuls(tla.simd.vlds(score[state, 0]), SCALE)
                                        x1 = tla.simd.vmuls(tla.simd.vlds(score[state, 64]), SCALE)
                                        if MASKED:
                                            x0 = tla.simd.vsel(valid0, x0, neg)
                                        if MASKED or SHARED_KVP:
                                            x1 = tla.simd.vsel(valid1, x1, neg)
                                        old_max = tla.simd.vlds(max_state[state], dist='BRC_B32')
                                        maximum = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_max)
                                        alpha = tla.simd.vexpdif(old_max, maximum)
                                        e0 = tla.simd.vexpdif(x0, maximum)
                                        e1 = tla.simd.vexpdif(x1, maximum)
                                        denominator = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(e0, e1)))
                                        denominator = tla.simd.vfma(denominator,
                                                                   tla.simd.vlds(sum_state[state], dist='BRC_B32'), alpha)
                                        tla.simd.vsts(max_state[state], maximum, dist='ONEPT_B32')
                                        tla.simd.vsts(sum_state[state], denominator, dist='ONEPT_B32')
                                        tla.simd.vsts(correction[state], alpha, dist='ONEPT_B32')
                                        p0 = tla.simd.vtruncf(e0, tl.bfloat16)
                                        p1 = tla.simd.vtruncf(e1, tl.bfloat16)
                                        packed, unused = tla.simd.vdintlv(p0, p1)
                                        tla.simd.vsstb(packed, packed_p[0, row * 16], (ROWS + 1) * 16)

                with tla.scope.vector().task(f'publish_probability_{part}' if PARTS > 1 else 'publish_probability',
                                             stage=STAGES[1]):
                    all_prob = tla.reshape(probability, (PARTS * KV_TILE // 16, TILE_ROWS * 16))
                    tla.copy_ub_to_l1(
                        tla.view(packed_p, (KV_TILE // 16, ROWS * 16)),
                        tla.view(all_prob, (KV_TILE // 16, ROWS * 16),
                                 offsets=(part * (KV_TILE // 16), sub * ROWS * 16)))

            for task_index in tl.static_range(PV_TASKS):
                with tla.scope.cube().task(f'pv_{task_index}' if PV_TASKS > 1 else 'pv', stage=STAGES[2]):
                    product = pv_second if SPLIT_PV and task_index == 1 else pv
                    # The 112-token schedule shares one stage-2 UB handoff to fit ragged jobs.
                    handoff = partial_second if DOUBLE_PV_HANDOFF and task_index == 1 else partial
                    prob_part = tla.view(probability, PROB_NZ,
                                         offsets=(task_index // VALUE_PARTS * (KV_TILE // 16), 0, 0, 0))
                    value = tla.view(kv_lat, (VALUE_TILE // 16, KV_TILE // 16, 16, 16),
                                     offsets=(task_index % VALUE_PARTS * (VALUE_TILE // 16), 0, 0, 0))
                    tla.cube.mmad(prob_part, value, product, clear_accum=True)
                    tla.copy_l0c_to_ub(
                        product, handoff,
                        tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

                with tla.scope.vector().task(f'accumulate_{task_index}' if PV_TASKS > 1 else 'accumulate',
                                             stage=STAGES[2]):
                    handoff = partial_second if DOUBLE_PV_HANDOFF and task_index == 1 else partial
                    with tla.scope.simd():
                        for row in tl.range(ROWS):
                            state = task_index // VALUE_PARTS * ROWS + row
                            alpha = tla.simd.vlds(correction[state], dist='BRC_B32')
                            for col in tl.static_range(VALUE_TILE // 64):
                                column = task_index % VALUE_PARTS * VALUE_TILE + col * 64
                                update = tla.simd.vfma(tla.simd.vlds(handoff[row, col * 64]),
                                                       tla.simd.vlds(accumulator[state, column]), alpha)
                                tla.simd.vsts(accumulator[state, column], update)

        with tla.scope.vector():
            for part in tl.static_range(PARTS):
                workspace_row = split * ALL_ROWS + q_row + part * TILE_ROWS + sub * ROWS
                tla.store_ub_to_gm(tla.view(accumulator, (ROWS, LATENT), offsets=(part * ROWS, 0)),
                                   tla.view(NUMERATOR + workspace_row * LATENT, (ROWS, LATENT)))
                tla.store_ub_to_gm(tla.view(max_state, (ROWS,), offsets=(part * ROWS,)),
                                   tla.view(MAXIMUM + workspace_row, (ROWS,)))
                tla.store_ub_to_gm(tla.view(sum_state, (ROWS,), offsets=(part * ROWS,)),
                                   tla.view(DENOMINATOR + workspace_row, (ROWS,)))


@triton.jit
def merge_splits(NUMERATOR, MAXIMUM, DENOMINATOR, OUT, LSE, ROWS: tl.constexpr,
                 SPLITS: tl.constexpr, SPLIT_TILE: tl.constexpr, COL_TILE: tl.constexpr, RETURN_LSE: tl.constexpr):
    LATENT: tl.constexpr = 512
    col_tiles: tl.constexpr = LATENT // COL_TILE
    job = tl.program_id(0)
    row = job // col_tiles
    col = job % col_tiles * COL_TILE + tl.arange(0, COL_TILE)
    split = tl.arange(0, SPLIT_TILE)
    split_row = split * ROWS + row
    maxima = tl.load(MAXIMUM + split_row, split < SPLITS, -float('inf'))
    sums = tl.load(DENOMINATOR + split_row, split < SPLITS, 0.0)
    # Rebase each unnormalized partial at the same maximum before the sum.
    maximum = tl.max(maxima, 0)
    weight = tl.exp(maxima - maximum)
    denominator = tl.sum(sums * weight, 0)
    numerators = tl.load(NUMERATOR + split_row[:, None] * LATENT + col[None, :], split[:, None] < SPLITS, 0.0)
    result = tl.sum(numerators * weight[:, None], 0) / denominator
    tl.store(OUT + row * LATENT + col, result)
    if RETURN_LSE:
        tl.store(LSE + row, maximum + tl.log(denominator), job % col_tiles == 0)


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def stream_k_splits(table_pages, query_tiles, cores):
    request_tiles = triton.cdiv(table_pages * 128, 112)
    core_tiles = triton.cdiv(query_tiles * request_tiles, cores)
    return min(cores, triton.cdiv(request_tiles + core_tiles - 1, core_tiles))


def prepare_call(q, qr, kv, kr, table, lengths, *, prefetch=True, head_tile=None, split_pages=None, scale=1.0 / 24.0,
                 schedule=None, skip_full_mask=None, merge_columns=512, kv_major=True, q_group=1, wide_qk=False,
                 return_lse=False, kv_l2=None, split_pv=False, kv_tile=128, l0_mode=None, _split_kernel=None):
    """Prepare attention for BSND Q and page-major KV with right-aligned causal masking.

    The caller supplies one actual length in [Sq, table capacity] per request.
    Entries for pages below that length must address the physical KV cache.
    When both RoPE arguments are None, Q and KV contain the packed 576 columns.
    Inputs and page metadata remain stable during each invocation.
    kv_tile=112 selects the packed three-slot pipeline by default.
    """
    import torch

    split_kernel = mla_split if _split_kernel is None else _split_kernel
    packed = qr is None and kr is None
    if kv_tile not in (112, 128) or (kv_tile == 112 and (not packed or q_group != 1 or _split_kernel is not None)):
        raise ValueError('The 112-token pipeline requires packed inputs and one Q tile per group')
    if l0_mode is not None and (l0_mode not in L0_MODES or not packed or _split_kernel is None
                                or q_group != 1 or wide_qk or split_pv):
        raise ValueError('L0 modes require packed inputs, one Q tile, and the explicit split kernel')
    b, sq, heads, d = q.shape
    if b <= 0 or sq <= 0 or heads <= 0 or d != (576 if packed else 512):
        raise ValueError('Expected packed Q [B,Sq,H,576] or split Q [B,Sq,H,512]')
    if head_tile is None:
        if heads % 32 and sq * heads % 96 == 0:
            head_tile = 96
        else:
            head_tile = 32 if heads % 64 else 64
    if not isinstance(head_tile, int) or head_tile not in (32, 64, 96):
        raise ValueError('Expected a query row tile of 32, 64 or 96')
    if q_group not in (1, 2) or sq * heads % (head_tile * q_group):
        raise ValueError('Expected complete query row tiles within each request')
    if head_tile * q_group > 128:
        raise ValueError('A query group can cover at most 128 rows')
    if wide_qk and q_group != 2:
        raise ValueError('Wide QK requires a group of two head tiles')
    if split_pv and q_group != 1:
        raise ValueError('Independent PV bands require one query row tile per group')
    tile_rows = head_tile * (q_group if wide_qk else 1)
    if heads % tile_rows and (tile_rows // 2) % heads:
        raise ValueError('Query rows must align with whole head groups in each Vector sub-block')
    if kv.ndim != 3 or kv.shape[0] == 0 or kv.shape[1:] != (128, d):
        raise ValueError('Expected page-major KV [pages,128,D] with the same width as Q')
    if packed:
        qr, kr = q, kv  # The packed specialization does not read these arguments.
    elif qr is None or kr is None or qr.shape != (b, sq, heads, 64) or kr.shape != (kv.shape[0], 128, 64):
        raise ValueError('Expected QR [B,Sq,H,64] and KR [pages,128,64] with split Q/KV')
    if table.ndim != 2 or table.shape[0] != b or table.shape[1] * 128 < sq or lengths.shape != (b,):
        raise ValueError('Expected page table [B,max_pages] and lengths [B]')
    if any(t.dtype != torch.bfloat16 for t in (q, qr, kv, kr)) or table.dtype != torch.int32 or lengths.dtype != torch.int32:
        raise ValueError('Expected BF16 Q/KV and RoPE tensors and int32 page metadata')
    if q.device.type != 'npu' or any(t.device != q.device or not t.is_contiguous() for t in (q, qr, kv, kr, table, lengths)):
        raise ValueError('Expected contiguous tensors on the same Ascend NPU')
    if (split_pages is not None and (not isinstance(split_pages, int) or split_pages <= 0)
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Expected positive split_pages and finite positive scale')
    if merge_columns not in (128, 256, 512):
        raise ValueError('Expected a merge tile of 128, 256 or 512 columns')

    rows = b * sq * heads
    aic, aiv = device_cores(q.device)
    if aic <= 0 or aiv != 2 * aic:
        raise ValueError(f'Expected 1:2 Cube/Vector ratio, got {aic}:{aiv}')
    query_tiles = rows // (head_tile * q_group)
    aligned_splits = kv_tile == 112 or l0_mode in KV112_MODES
    if split_pages is None:
        split_pages = (triton.cdiv(table.shape[1], aic // math.gcd(query_tiles, aic)) if head_tile == 96 else 64)
        if aligned_splits:
            split_pages = triton.cdiv(split_pages, 7) * 7
    if aligned_splits and split_pages % 7:
        raise ValueError('The 112-token tile requires split boundaries aligned to seven physical pages')
    splits = triton.cdiv(table.shape[1], split_pages)
    if l0_mode == 'native112_merge_dynflags_stream_k':
        stream_splits = stream_k_splits(table.shape[1], query_tiles, aic)
        if stream_splits <= FUSED_MERGE_MAX_SPLITS:
            splits = stream_splits
    # Eight merge rows need 16 KiB of FP32 numerator storage per split.
    # Keep room for the buffers retained through the AIV fence.
    fuse_merge = l0_mode in FUSED_MERGE_MODES and splits <= FUSED_MERGE_MAX_SPLITS
    cores = aic if fuse_merge else min(aic, query_tiles * splits)
    if not prefetch:
        schedule = 'serial'
    elif schedule is None:
        if kv_tile == 112:
            schedule = 'overlap'
        else:
            schedule = 'late_pv' if head_tile == 96 else 'prefetch' if q_group == 2 else 'overlap'
    if skip_full_mask is None:
        skip_full_mask = prefetch
    stages = SCHEDULES[schedule]
    options = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
    if kv_tile == 112:
        options['KV_TILE'] = kv_tile
    if l0_mode is not None:
        options['L0_MODE'] = l0_mode
        if l0_mode not in L1_112_MODES:
            options.update(unit_flag=False, sync_solver=True)
        if l0_mode in MANUAL_L0_MODES:
            options['disable_auto_inject_sync'] = True

    def call():
        numerator = torch.empty((splits, rows, 512), dtype=torch.float32, device=q.device)
        maximum = torch.empty((splits, rows), dtype=torch.float32, device=q.device)
        denominator = torch.empty_like(maximum)
        output = torch.empty((b, sq, heads, 512), dtype=q.dtype, device=q.device)
        lse = torch.empty((b, sq, heads), dtype=torch.float32, device=q.device) if return_lse else maximum
        merge_args = dict(OUT=output, LSE=lse, FUSE_MERGE=True, RETURN_LSE=return_lse) if fuse_merge else {}
        with torch.npu.device(q.device):
            split_kernel[(cores,)](q, qr, kv, kr, table, lengths, numerator, maximum, denominator,
                                   b, sq, heads, table.shape[1], split_pages, splits, cores, stages, scale,
                                   head_tile, skip_full_mask, kv_major, q_group, wide_qk, packed, kv_l2, split_pv,
                                   **options, **merge_args)
            if not fuse_merge:
                merge_splits[(rows * (512 // merge_columns),)](
                    numerator, maximum, denominator, output, lse, rows, splits,
                    triton.next_power_of_2(splits), merge_columns, return_lse, sanitize_overflow=False)
        return (output, lse) if return_lse else output

    explicit = split_kernel is not mla_split
    call.metadata = dict(cube_grid=cores, task_stages=None if explicit else stages,
                         schedule='explicit' if explicit else schedule, split_pages=split_pages, splits=splits,
                         head_tile=head_tile, q_group=q_group, kv_page=128, scale=scale, skip_full_mask=skip_full_mask,
                         merge_columns=merge_columns, fuse_merge=fuse_merge, kv_major=kv_major, wide_qk=wide_qk, packed=packed,
                         return_lse=return_lse, kv_l2=kv_l2, split_pv=split_pv, l0_mode=l0_mode,
                         kv_tile=112 if aligned_splits else 128)
    return call
