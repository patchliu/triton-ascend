"""Single-token gated delta rule with one in-place state update per request."""

import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def gdn_recurrent(Q, K, V, G, BETA, STATE, SLOTS, O, B: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr,
                  DK: tl.constexpr, DV: tl.constexpr, CORES: tl.constexpr, SCALE: tl.constexpr,
                  COMPUTE_STAGE: tl.constexpr, STORE_STAGE: tl.constexpr, BV: tl.constexpr = 128):
    tl.static_assert(DK == 128 and DV == 128 and HV % HK == 0 and DV % BV == 0 and BV % 32 == 0)
    TILES_PER_HEAD: tl.constexpr = DV // BV
    worker = tl.program_id(0)
    # Each job owns disjoint value rows of one state matrix. Only the work
    # across jobs is pipelined. Each state element is loaded and stored once.
    for tile in tla.pipeline(tl.cdiv(B * HV * TILES_PER_HEAD - worker, CORES)):
        job = worker + tile * CORES
        head_job = job // TILES_PER_HEAD
        row_start = job % TILES_PER_HEAD * BV
        batch = head_job // HV
        value_head = head_job % HV
        key_head = value_head // (HV // HK)
        qk_offset = (batch * HK + key_head) * DK
        value_offset = head_job * DV + row_start
        q = tla.alloc((DK,), tl.bfloat16, space='ub')
        k = tla.alloc((DK,), tl.bfloat16, space='ub')
        v = tla.alloc((BV,), tl.bfloat16, space='ub')
        beta = tla.alloc((HV,), tl.bfloat16, space='ub')
        old_state = tla.alloc((BV, DK), tl.bfloat16, space='ub')
        new_state = tla.alloc((BV, DK), tl.bfloat16, space='ub')
        out = tla.alloc((BV,), tl.bfloat16, space='ub')

        with tla.scope.vector().task('load', stage=0):
            slot = tl.load(SLOTS + batch).to(tl.int64)
            state_offset = ((slot * HV + value_head) * DV + row_start) * DK
            tla.load_gm_to_ub(tla.view(STATE + state_offset, (BV, DK)), old_state)
            tla.load_gm_to_ub(tla.view(Q + qk_offset, (DK,)), q)
            tla.load_gm_to_ub(tla.view(K + qk_offset, (DK,)), k)
            tla.load_gm_to_ub(tla.view(V + value_offset, (BV,)), v)
            tla.load_gm_to_ub(tla.view(BETA + batch * HV, (HV,)), beta)

        with tla.scope.vector().task('update', stage=COMPUTE_STAGE):
            g = tl.load(G + head_job)
            with tla.scope.simd():
                alpha = tla.simd.vexp(tla.simd.vdup(g, tl.float32))
                step = tla.simd.vextf(tla.simd.vlds(beta[value_head], dist='BRC_B16'), tl.float32)
                q_raw = tla.simd.vlds(q[0])
                k_raw = tla.simd.vlds(k[0])
                # Duplicate each BF16 lane into a pair so widening each
                # register produces one contiguous half of the 128 lanes.
                q0, q1 = tla.simd.vintlv(q_raw, q_raw)
                k0, k1 = tla.simd.vintlv(k_raw, k_raw)
                query0 = tla.simd.vmuls(tla.simd.vextf(q0, tl.float32), SCALE)
                query1 = tla.simd.vmuls(tla.simd.vextf(q1, tl.float32), SCALE)
                key0 = tla.simd.vextf(k0, tl.float32)
                key1 = tla.simd.vextf(k1, tl.float32)
                for row in tl.range(BV):
                    raw = tla.simd.vlds(old_state[row, 0])
                    s0, s1 = tla.simd.vintlv(raw, raw)
                    decayed0 = tla.simd.vmul(tla.simd.vextf(s0, tl.float32), alpha)
                    decayed1 = tla.simd.vmul(tla.simd.vextf(s1, tl.float32), alpha)
                    prediction = tla.simd.vreduce_sum(tla.simd.vadd(tla.simd.vmul(decayed0, key0),
                                                                    tla.simd.vmul(decayed1, key1)))
                    value = tla.simd.vextf(tla.simd.vlds(v[row], dist='BRC_B16'), tl.float32)
                    delta = tla.simd.vmul(tla.simd.vsub(value, tla.simd.vbroadcast(prediction)), step)
                    updated0 = tla.simd.vfma(decayed0, delta, key0)
                    updated1 = tla.simd.vfma(decayed1, delta, key1)
                    result = tla.simd.vreduce_sum(tla.simd.vadd(tla.simd.vmul(updated0, query0),
                                                                tla.simd.vmul(updated1, query1)))
                    packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(updated0, tl.bfloat16),
                                                       tla.simd.vtruncf(updated1, tl.bfloat16))
                    tla.simd.vsts(new_state[row, 0], packed, dist='NORM_B16')
                    tla.simd.vsts(out[row], tla.simd.vtruncf(result, tl.bfloat16), dist='ONEPT_B16')

        with tla.scope.vector().task('store', stage=STORE_STAGE):
            slot = tl.load(SLOTS + batch).to(tl.int64)
            state_offset = ((slot * HV + value_head) * DV + row_start) * DK
            tla.store_ub_to_gm(new_state, tla.view(STATE + state_offset, (BV, DK)))
            tla.store_ub_to_gm(out, tla.view(O + value_offset, (BV,)))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, k, v, g, beta, state, slots, *, scale=None, prefetch=True, value_tile=64, store_stage=None):
    """Return one TND decode step that updates `state` in place.

    Slots are pairwise distinct valid int32 indices. There is one token per
    request. Query and key are already normalized if the caller needs it.
    """
    import torch

    b, hk, dk = q.shape
    bv, hv, dv = v.shape
    if not (b > 0 and hk > 0 and hv > 0 and hv % hk == 0 and (dk, dv) == (128, 128)):
        raise ValueError('Expected positive batch, Hv divisible by Hk and Dk = Dv = 128')
    if bv != b or k.shape != q.shape or g.shape != (b, hv) or beta.shape != (b, hv):
        raise ValueError('Expected TND Q/K/V and per-token, per-value-head gates')
    if state.ndim != 4 or state.shape[0] < b or state.shape[1:] != (hv, dv, dk) or slots.shape != (b,):
        raise ValueError('Expected state [slots,Hv,Dv,Dk] and one slot index per request')
    if (any(x.dtype != torch.bfloat16 for x in (q, k, v, beta, state))
            or g.dtype != torch.float32 or slots.dtype != torch.int32):
        raise ValueError('Expected BF16 Q/K/V/beta/state, FP32 g and int32 slots')
    if q.device.type != 'npu' or any(x.device != q.device or not x.is_contiguous()
                                    for x in (q, k, v, g, beta, state, slots)):
        raise ValueError('Expected contiguous inputs on the same NPU')
    if type(value_tile) is not int or value_tile not in (32, 64, 128):
        raise ValueError('Expected value_tile of 32, 64 or 128')

    scale = dk**-0.5 if scale is None else float(scale)
    _, vectors = device_cores(q.device)
    cores = min(b * hv * (dv // value_tile), vectors)
    compute_stage = int(prefetch)
    if store_stage is None:
        store_stage = 2 if prefetch else 0
    if type(store_stage) is not int or store_stage not in ((1, 2) if prefetch else (0,)):
        raise ValueError('Expected store_stage 1 or 2 for prefetch, or 0 for stage0')

    def call():
        out = torch.empty_like(v)
        gdn_recurrent[(cores,)](q, k, v, g, beta, state, slots, out, b, hk, hv, dk, dv, cores, scale, compute_stage,
                               store_stage, value_tile, sanitize_overflow=False, enable_auto_bind_sub_block=False,
                               enable_preload=False)
        return out, state

    call.metadata = dict(grid=cores, value_tile=value_tile, compute_stage=compute_stage, store_stage=store_stage)
    return call
