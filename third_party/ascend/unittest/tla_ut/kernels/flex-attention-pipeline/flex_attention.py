"""FlexAttention forward with a packed full/partial block mask."""

from collections import defaultdict
from functools import partial

import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction


@partial(TLAJITFunction, do_not_specialize=['SLOTS'])
def flex_forward(Q, K, V, Q_BLOCKS, KV_BLOCKS, MASK_SLOTS, PARTIAL_MASK, O, SLOTS, PLAN_ROWS,
                 HQ: tl.constexpr, HKV: tl.constexpr, Q_PAD: tl.constexpr, KV_PAD: tl.constexpr,
                 D: tl.constexpr, CORES: tl.constexpr, COMPUTE_STAGE: tl.constexpr,
                 SKIP_FULL_MASK: tl.constexpr = False, QT: tl.constexpr = 64, COLUMN_SOFTMAX: tl.constexpr = False,
                 PACKED_PREDICATE: tl.constexpr = False, DEEP: tl.constexpr = False):
    KT: tl.constexpr = 128
    ROWS: tl.constexpr = QT // 2
    GROUP: tl.constexpr = HQ // HKV
    Q_SUBTILES: tl.constexpr = 128 // QT
    QK_STAGE: tl.constexpr = 0 if COLUMN_SOFTMAX else COMPUTE_STAGE
    PV_STAGE: tl.constexpr = 2 * COMPUTE_STAGE if COLUMN_SOFTMAX else COMPUTE_STAGE
    ACCUM_STAGE: tl.constexpr = PV_STAGE + COMPUTE_STAGE if DEEP else PV_STAGE
    tl.static_assert(QT == 64 or QT == 128)
    tl.static_assert(not COLUMN_SOFTMAX or QT == 128)
    tl.static_assert(not (PACKED_PREDICATE or DEEP) or COLUMN_SOFTMAX)

    for job in tl.range(tl.program_id(0), HQ * PLAN_ROWS * Q_SUBTILES, CORES):
        head = job // (PLAN_ROWS * Q_SUBTILES)
        plan_row = job // Q_SUBTILES % PLAN_ROWS
        q_subtile = job % Q_SUBTILES * QT
        kv_head = head // GROUP
        sub = tla.sub_block_id()

        query = tla.alloc((D // 16, QT // 16, 16, 16), tl.bfloat16, space='l1')
        maximum = tla.alloc((64,), tl.float32, space='ub')
        denominator = tla.alloc((64,), tl.float32, space='ub')
        accumulator = tla.alloc((ROWS, D), tl.float32, space='ub')
        with tla.scope.cube():
            q_block = tl.load(Q_BLOCKS + plan_row)
            q_offset = (head * Q_PAD + q_block * 128 + q_subtile) * D
            tla.load_gm_to_l1(tla.view(Q + q_offset, (QT, D)), query)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vsplat(0.0, tl.float32)
                tla.simd.vsts(maximum[0], tla.simd.vsplat(-3.4028234663852886e38, tl.float32))
                tla.simd.vsts(denominator[0], zero)
                for row in tl.range(ROWS):
                    for col in tl.static_range(D // 64):
                        tla.simd.vsts(accumulator[row, col * 64], zero)

        # Keep the bucket count dynamic even for one slot. Folding a
        # one-trip loop removes the scf.for required by the pipeline scope.
        for tile in tla.pipeline(SLOTS):
            key = tla.alloc((D // 16, KT // 16, 16, 16), tl.bfloat16, space='l1')
            value = tla.alloc((D // 16, KT // 16, 16, 16), tl.bfloat16, space='l1')
            p = tla.alloc((KT // 16, QT // 16, 16, 16), tl.bfloat16, space='l1')
            qk = tla.alloc((KT // 16, QT // 16, 16, 16), tl.float32, space='l0c')
            pv = tla.alloc((D // 16, QT // 16, 16, 16), tl.float32, space='l0c')
            if COLUMN_SOFTMAX:
                scores = tla.alloc((KT, ROWS), tl.float32, space='ub')
                if PACKED_PREDICATE:
                    mask = tla.alloc((KT * 32,), tl.uint8, space='ub')
                else:
                    mask = tla.alloc((KT, ROWS), tl.int32, space='ub')
            else:
                scores = tla.alloc((ROWS, KT), tl.float32, space='ub')
                mask = tla.alloc((ROWS, KT), tl.int32, space='ub')
            packed = tla.alloc((KT // 16, (ROWS + 1) * 16), tl.bfloat16, space='ub')
            correction = tla.alloc((64,), tl.float32, space='ub')
            partial = tla.alloc((ROWS, D), tl.float32, space='ub')
            plan_index = plan_row * SLOTS + tile

            with tla.scope.cube().task('load_kv', stage=0):
                kv_block = tl.load(KV_BLOCKS + plan_index)
                offset = (kv_head * KV_PAD + kv_block * KT) * D
                tla.load_gm_to_l1(tla.view(K + offset, (KT, D)), key)
                if not COLUMN_SOFTMAX:
                    tla.load_gm_to_l1(tla.view(V + offset, (KT, D)), value)

            with tla.scope.cube().task('qk', stage=QK_STAGE):
                if COLUMN_SOFTMAX:
                    tla.cube.mmad(key, query, qk, transpose_b=True, clear_accum=True)
                    SPLIT: tl.constexpr = tla.FixpipeDualDstMode.COLUMN_SPLIT
                else:
                    tla.cube.mmad(query, key, qk, transpose_b=True, clear_accum=True)
                    SPLIT: tl.constexpr = tla.FixpipeDualDstMode.ROW_SPLIT
                tla.copy_l0c_to_ub(
                    qk, scores,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('softmax', stage=COMPUTE_STAGE):
                partial_slot = tl.load(MASK_SLOTS + plan_index)
                # -1 marks full blocks. The -2 padding slot stays masked.
                if COLUMN_SOFTMAX:
                    if partial_slot >= 0:
                        if PACKED_PREDICATE:
                            mask_offset = (partial_slot * 2 + sub) * KT * 32
                            tla.load_gm_to_ub(tla.view(PARTIAL_MASK + mask_offset, (KT * 32,)), mask)
                        else:
                            mask_offset = partial_slot * KT * QT + sub * ROWS
                            tla.load_gm_to_ub(tla.view(PARTIAL_MASK + mask_offset, (KT, ROWS), (QT, 1)), mask)
                        with tla.scope.simd():
                            neg = tla.simd.vsplat(float('-inf'), tl.float32)
                            for key in tl.range(KT):
                                if PACKED_PREDICATE:
                                    valid = tla.simd.pld(mask[key * 32], width=32)
                                else:
                                    valid = tla.simd.vcmps(tla.simd.vlds(mask[key, 0]), 0, op=tla.simd.CMP.NE)
                                tla.simd.vsts(scores[key, 0],
                                              tla.simd.vsel(valid, tla.simd.vlds(scores[key, 0]), neg))
                    elif partial_slot == -2:
                        with tla.scope.simd():
                            neg = tla.simd.vsplat(float('-inf'), tl.float32)
                            for key in tl.range(KT):
                                tla.simd.vsts(scores[key, 0], neg)
                    with tla.scope.simd():
                        # Fixpipe split the columns: each lane owns one query and scans the
                        # 128 keys. Four independent chains reduce the loop dependency.
                        negative = tla.simd.vsplat(float('-inf'), tl.float32)
                        max0, max1, max2, max3 = negative, negative, negative, negative
                        for key in tl.range(0, 128, 4):
                            max0 = tla.simd.vmax(max0, tla.simd.vlds(scores[key, 0]))
                            max1 = tla.simd.vmax(max1, tla.simd.vlds(scores[key + 1, 0]))
                            max2 = tla.simd.vmax(max2, tla.simd.vlds(scores[key + 2, 0]))
                            max3 = tla.simd.vmax(max3, tla.simd.vlds(scores[key + 3, 0]))
                        tile_max = tla.simd.vmax(tla.simd.vmax(max0, max1), tla.simd.vmax(max2, max3))
                        old_max = tla.simd.vlds(maximum[0])
                        new_max = tla.simd.vmax(old_max, tla.simd.vmuls(tile_max, D**-0.5))
                        alpha = tla.simd.vexpdif(old_max, new_max)
                        tla.simd.vsts(correction[0], alpha)
                        tla.simd.vsts(maximum[0], new_max)
                        zero = tla.simd.vsplat(0.0, tl.float32)
                        sum0, sum1, sum2, sum3 = zero, zero, zero, zero
                        for key in tl.range(32):
                            e0 = tla.simd.vexpdif(tla.simd.vmuls(tla.simd.vlds(scores[key, 0]), D**-0.5), new_max)
                            e1 = tla.simd.vexpdif(tla.simd.vmuls(tla.simd.vlds(scores[key + 32, 0]), D**-0.5), new_max)
                            e2 = tla.simd.vexpdif(tla.simd.vmuls(tla.simd.vlds(scores[key + 64, 0]), D**-0.5), new_max)
                            e3 = tla.simd.vexpdif(tla.simd.vmuls(tla.simd.vlds(scores[key + 96, 0]), D**-0.5), new_max)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            low, _ = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16), tla.simd.vtruncf(e2, tl.bfloat16))
                            high, _ = tla.simd.vdintlv(tla.simd.vtruncf(e1, tl.bfloat16), tla.simd.vtruncf(e3, tl.bfloat16))
                            # Store the two key halves into the same padded NZ scratch.
                            tla.simd.vsstb(low, packed[0, key * 16], 65 * 16)
                            tla.simd.vsstb(high, packed[0, (key + 32) * 16], 65 * 16)
                        total = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        tla.simd.vsts(denominator[0], tla.simd.vfma(total, tla.simd.vlds(denominator[0]), alpha))
                else:
                    needs_mask = partial_slot != -1
                    first_mask: tl.constexpr = 0 if SKIP_FULL_MASK else 1
                    for MASKED in tl.static_range(first_mask, 2):
                        if not SKIP_FULL_MASK or needs_mask == MASKED:
                            if MASKED:
                                if partial_slot >= 0:
                                    mask_offset = (partial_slot * 128 + q_subtile + sub * ROWS) * KT
                                    tla.load_gm_to_ub(tla.view(PARTIAL_MASK + mask_offset, (ROWS, KT)), mask)
                                else:
                                    with tla.scope.simd():
                                        allowed = tla.simd.vdup((partial_slot == -1).to(tl.int32), tl.int32)
                                        for row in tl.range(ROWS):
                                            tla.simd.vsts(mask[row, 0], allowed)
                                            tla.simd.vsts(mask[row, 64], allowed)

                            with tla.scope.simd():
                                scale = tla.simd.vsplat(D**-0.5, tl.float32)
                                if MASKED:
                                    neg_inf = tla.simd.vsplat(float('-inf'), tl.float32)
                                for row in tl.range(ROWS):
                                    x0 = tla.simd.vmul(tla.simd.vlds(scores[row, 0]), scale)
                                    x1 = tla.simd.vmul(tla.simd.vlds(scores[row, 64]), scale)
                                    if MASKED:
                                        valid0 = tla.simd.vcmps(tla.simd.vlds(mask[row, 0]), 0, op=tla.simd.CMP.NE)
                                        valid1 = tla.simd.vcmps(tla.simd.vlds(mask[row, 64]), 0, op=tla.simd.CMP.NE)
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

            with tla.scope.vector().task('publish_probability', stage=COMPUTE_STAGE):
                if COLUMN_SOFTMAX:
                    p_rows = tla.reshape(p, (QT // 16, KT * 16))
                    HALF: tl.constexpr = (ROWS // 16, KT // 2 * 16)
                    for half in tl.static_range(2):
                        source = tla.subview(packed, (half * (ROWS // 16), 0), HALF)
                        target = tla.subview(p_rows, (sub * (ROWS // 16), half * (KT // 2 * 16)), HALF)
                        tla.copy_ub_to_l1(source, target)
                else:
                    tla.copy_ub_to_l1(
                        tla.view(packed, (KT // 16, ROWS * 16)),
                        tla.view(tla.reshape(p, (KT // 16, QT * 16)), (KT // 16, ROWS * 16),
                                 offsets=(0, sub * ROWS * 16)))

            with tla.scope.cube().task('pv', stage=PV_STAGE):
                if COLUMN_SOFTMAX:
                    # Keep V out of the earlier QK and softmax stages.
                    kv_block = tl.load(KV_BLOCKS + plan_index)
                    offset = (kv_head * KV_PAD + kv_block * KT) * D
                    tla.load_gm_to_l1(tla.view(V + offset, (KT, D)), value)
                tla.cube.mmad(p, value, pv, transpose_a=COLUMN_SOFTMAX, clear_accum=True)
                tla.copy_l0c_to_ub(
                    pv, partial,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

            with tla.scope.vector().task('accumulate', stage=ACCUM_STAGE):
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
                    denom = tla.simd.vlds(denominator[row], dist='BRC_B32')
                    denom = tla.simd.vmax(denom, tla.simd.vsplat(1.0, tl.float32))
                    for col in tl.static_range(D // 128):
                        lo = tla.simd.vdiv(tla.simd.vlds(accumulator[row, col * 128]), denom)
                        hi = tla.simd.vdiv(tla.simd.vlds(accumulator[row, col * 128 + 64]), denom)
                        values, unused = tla.simd.vdintlv(tla.simd.vtruncf(lo, tl.bfloat16),
                                                          tla.simd.vtruncf(hi, tl.bfloat16))
                        tla.simd.vsts(output[row, col * 128], values)
            q_block = tl.load(Q_BLOCKS + plan_row)
            q_offset = (head * Q_PAD + q_block * 128 + q_subtile + sub * ROWS) * D
            tla.store_ub_to_gm(output, tla.view(O + q_offset, (ROWS, D)))


def _merge_bucket_groups(groups):
    # Each widened slot still runs QK and PV. Limit extra tile work to 0.5%.
    remaining_padding = sum(width * len(rows) for width, rows in groups) // 200
    merged = []
    for width, rows in groups:
        if merged:
            previous_width, previous_rows = merged[-1]
            extra_slots = (width - previous_width) * len(previous_rows)
            if extra_slots <= remaining_padding:
                previous_rows.extend(rows)
                merged[-1] = (width, previous_rows)
                remaining_padding -= extra_slots
                continue
        merged.append((width, list(rows)))
    return merged


def make_plan(block_mask, *, merge_buckets=True, column_softmax=True, packed_predicate=None):
    """Compile reusable B=H=1 packed BlockMask metadata into bucket tables."""
    import torch

    if tuple(block_mask.BLOCK_SIZE) != (128, 128):
        raise ValueError('Expected 128x128 BlockMask tiles')
    if packed_predicate is None:
        packed_predicate = column_softmax
    if packed_predicate and not column_softmax:
        raise ValueError('Packed predicates need column softmax')
    if block_mask.kv_num_blocks.shape[:2] != (1, 1):
        raise ValueError('Expected a batch/head-independent mask')
    q_len, kv_len = block_mask.seq_lengths
    if q_len <= 0 or kv_len <= 0:
        raise ValueError('Expected positive query and KV lengths')
    q_blocks, kv_blocks = (q_len + 127) // 128, (kv_len + 127) // 128
    partial_counts = block_mask.kv_num_blocks[0, 0].cpu().tolist()
    partial_indices = block_mask.kv_indices[0, 0].cpu().tolist()
    full_counts, full_indices = [0] * q_blocks, [[] for _ in range(q_blocks)]
    if block_mask.full_kv_num_blocks is not None:
        full_counts = block_mask.full_kv_num_blocks[0, 0].cpu().tolist()
        full_indices = block_mask.full_kv_indices[0, 0].cpu().tolist()
    partial_slots = block_mask.partial_block_table.cpu().tolist()
    masks = block_mask.packed_partial_mask
    if masks.ndim != 3 or masks.shape[1:] != (128, 128):
        raise ValueError('Expected packed partial mask [tiles,128,128]')
    if len(partial_counts) != q_blocks or len(full_counts) != q_blocks or len(partial_slots) != q_blocks:
        raise ValueError('BlockMask shape does not match its sequence lengths')

    buckets = defaultdict(list)
    needs_dummy = False
    for row in range(q_blocks):
        full = full_indices[row][:full_counts[row]]
        partial = partial_indices[row][:partial_counts[row]]
        indices = full + partial
        if any(index < 0 or index >= kv_blocks for index in indices) or len(set(indices)) != len(indices):
            raise ValueError('Invalid or duplicate KV block index')
        slots = [partial_slots[row][index] for index in partial]
        if any(slot < 0 or slot >= masks.shape[0] for slot in slots):
            raise ValueError('Partial block is missing its packed mask')
        # All rows with the same trip count share a launch. An empty row uses
        # one padded slot so the output path writes its zeros normally.
        count = len(indices)
        if count:
            buckets[count].append((row, indices, [-1] * len(full) + slots))
        else:
            buckets[1].append((row, [kv_blocks], [-2]))
            needs_dummy = True

    device = block_mask.kv_indices.device
    groups = sorted(buckets.items())
    if merge_buckets:
        groups = _merge_bucket_groups(groups)
    planned_buckets = []
    for width, rows in groups:
        query_ids, key_ids, mask_ids = [], [], []
        for row, indices, slots in rows:
            missing = width - len(indices)
            needs_dummy |= missing > 0
            query_ids.append(row)
            key_ids.append(indices + [kv_blocks] * missing)
            mask_ids.append(slots + [-2] * missing)
        planned_buckets.append(tuple(torch.tensor(column, dtype=torch.int32, device=device)
                                     for column in (query_ids, key_ids, mask_ids)))

    # Empty slots read a dedicated zero K/V tile, including when real V contains NaN or Inf.
    kv_pad = (kv_blocks + int(needs_dummy)) * 128
    if masks.shape[0] == 0:
        masks = torch.empty((1, 128, 128), dtype=torch.int32, device=device)
    if packed_predicate:
        # Each AIV receives 64 query bits for a key. PLD requires an
        # aligned 32-byte slot even though only the first 8 bytes are used.
        bits = (masks != 0).transpose(1, 2).reshape(-1, 128, 2, 8, 8).to(torch.int32)
        weights = torch.tensor((1, 2, 4, 8, 16, 32, 64, 128), dtype=torch.int32, device=device)
        packed = (bits * weights).sum(dim=-1, dtype=torch.int32).to(torch.uint8).transpose(1, 2)
        masks = torch.nn.functional.pad(packed, (0, 24)).contiguous()
    elif column_softmax:
        masks = masks.transpose(1, 2)
    if not packed_predicate:
        masks = masks.to(torch.int32).contiguous()
    return q_len, kv_len, kv_pad, masks, column_softmax, tuple(planned_buckets)


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, k, v, plan, *, prefetch=True, skip_full_mask=True, q_tile=128, deep=None):
    """Prepare an output-only call from BNSD inputs and a reusable mask plan."""
    import torch
    import torch.nn.functional as functional

    q_len, kv_len, kv_pad, masks, column_softmax, buckets = plan
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError('Expected single-batch BNSD inputs')
    hq, hkv, dim = q.shape[1], k.shape[1], q.shape[-1]
    if not hq or not hkv or hq % hkv or q.shape[2] != q_len or k.shape[2:] != (kv_len, dim) or dim != 128:
        raise ValueError('Expected matching BlockMask lengths, D=128 and valid GQA heads')
    if q_tile not in (64, 128):
        raise ValueError('Expected a Q tile of 64 or 128 rows')
    if column_softmax and q_tile != 128:
        raise ValueError('Column softmax needs a complete 128-row BlockMask tile')
    if deep is None:
        deep = column_softmax and masks.dtype == torch.uint8
    if deep and not column_softmax:
        raise ValueError('Deep scheduling needs column softmax')
    if q.device.type != 'npu' or any(x.dtype != torch.bfloat16 or x.device != q.device or not x.is_contiguous()
                                    for x in (q, k, v)) or masks.device != q.device:
        raise ValueError('Expected contiguous BF16 tensors and a plan on the same NPU')
    cube, vector = device_cores(q.device)
    if cube <= 0 or vector != 2 * cube:
        raise ValueError(f'Expected a 1:2 Cube/Vector ratio, got {cube}:{vector}')
    q_pad = (q_len + 127) // 128 * 128

    def call():
        q_input = functional.pad(q, (0, 0, 0, q_pad - q_len)) if q_pad != q_len else q
        if kv_pad != kv_len:
            k_input = functional.pad(k, (0, 0, 0, kv_pad - kv_len))
            v_input = functional.pad(v, (0, 0, 0, kv_pad - kv_len))
        else:
            k_input, v_input = k, v
        output = torch.empty_like(q_input)
        for q_blocks, kv_blocks, slots in buckets:
            rows, count = kv_blocks.shape
            cores = min(cube, hq * rows * (128 // q_tile))
            flex_forward[(cores,)](q_input, k_input, v_input, q_blocks, kv_blocks, slots, masks, output,
                                   count, rows, hq, hkv, q_pad, kv_pad, dim, cores, int(prefetch),
                                   SKIP_FULL_MASK=skip_full_mask, QT=q_tile, COLUMN_SOFTMAX=column_softmax,
                                   PACKED_PREDICATE=masks.dtype == torch.uint8, DEEP=deep,
                                   sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
        return output[:, :, :q_len].contiguous()

    return call
