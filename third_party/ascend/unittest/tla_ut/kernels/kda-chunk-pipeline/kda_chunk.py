"""KDA chunk prefill with the FLA prologue and a fused TLA state/output scan."""

import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def kda_scan(W, U, QG, KG, AQK, G, H0, O, HT, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
             BT: tl.constexpr, SCALE: tl.constexpr, COMPUTE_STAGE: tl.constexpr):
    BV: tl.constexpr = 64
    CK: tl.constexpr = K // 2
    CT: tl.constexpr = BT // 2
    tl.static_assert(T % BT == 0 and K % 32 == 0 and V % BV == 0)

    job = tl.program_id(0)
    batch_head = job // (V // BV)
    batch, head = batch_head // H, batch_head % H
    column = job % (V // BV) * BV
    sub = tla.sub_block_id()
    state = tla.alloc((CK, BV), tl.float32, space='ub')
    state_offset = batch_head * K * V + sub * CK * V + column
    with tla.scope.vector():
        tla.load_gm_to_ub(tla.view(H0 + state_offset, (CK, BV), (V, 1)), state)

    for chunk in tla.pipeline(T // BT):
        token_head = (batch * T + chunk * BT) * H + head
        w = tla.alloc((K // 16, BT // 16, 16, 16), tl.bfloat16, space='l1')
        kg = tla.alloc((K // 16, BT // 16, 16, 16), tl.bfloat16, space='l1')
        qg = tla.alloc((K // 16, BT // 16, 16, 16), tl.bfloat16, space='l1')
        aqk = tla.alloc((BT // 16, BT // 16, 16, 16), tl.bfloat16, space='l1')
        state_l1 = tla.alloc((BV // 16, K // 16, 16, 16), tl.bfloat16, space='l1')
        corr_l1 = tla.alloc((BV // 16, BT // 16, 16, 16), tl.bfloat16, space='l1')
        state_nz = tla.alloc((BV // 16, (CK + 1) * 16), tl.bfloat16, space='ub')
        corr_nz = tla.alloc((BV // 16, (CT + 1) * 16), tl.bfloat16, space='ub')
        u = tla.alloc((CT, BV), tl.bfloat16, space='ub')
        gate = tla.alloc((CK,), tl.float32, space='ub')
        correction = tla.alloc((CT, BV), tl.float32, space='ub')
        inter = tla.alloc((CT, BV), tl.float32, space='ub')
        intra = tla.alloc((CT, BV), tl.float32, space='ub')
        delta = tla.alloc((CK, BV), tl.float32, space='ub')
        output = tla.alloc((CT, BV), tl.bfloat16, space='ub')
        chunk_acc = tla.alloc((BV // 16, BT // 16, 16, 16), tl.float32, space='l0c')
        state_acc = tla.alloc((BV // 16, K // 16, 16, 16), tl.float32, space='l0c')

        with tla.scope.cube().task('load_cube', stage=0):
            tla.load_gm_to_l1(tla.view(W + token_head * K, (BT, K), (H * K, 1)), w)
            tla.load_gm_to_l1(tla.view(KG + token_head * K, (BT, K), (H * K, 1)), kg)
            tla.load_gm_to_l1(tla.view(QG + token_head * K, (BT, K), (H * K, 1)), qg)
            tla.load_gm_to_l1(tla.view(AQK + token_head * BT, (BT, BT), (H * BT, 1)), aqk)

        with tla.scope.vector().task('load_vector', stage=0):
            offset = (token_head + sub * CT * H) * V + column
            tla.load_gm_to_ub(tla.view(U + offset, (CT, BV), (H * V, 1)), u)
            gate_offset = (token_head + (BT - 1) * H) * K + sub * CK
            tla.load_gm_to_ub(tla.view(G + gate_offset, (CK,)), gate)

        # Persistent state is FP32. Cube consumes a BF16 copy of the state at
        # the start of this chunk. All state accesses belong to one stage.
        with tla.scope.vector().task('publish_state', stage=COMPUTE_STAGE):
            with tla.scope.simd():
                half = tla.simd.pge(BV, 128)
                for row in tl.range(CK):
                    values = tla.simd.vtruncf(tla.simd.vlds(state[row, 0]), tl.bfloat16)
                    packed, unused = tla.simd.vdintlv(values, values)
                    tla.simd.vsstb(packed, state_nz[0, row * 16], (CK + 1) * 16, mask=half)
            tla.copy_ub_to_l1(tla.view(state_nz, (BV // 16, CK * 16)),
                             tla.view(tla.reshape(state_l1, (BV // 16, K * 16)), (BV // 16, CK * 16),
                                      offsets=(0, sub * CK * 16)))

        with tla.scope.cube().task('project_state', stage=COMPUTE_STAGE):
            tla.cube.mmad(w, state_l1, chunk_acc, clear_accum=True)
            tla.copy_l0c_to_ub(chunk_acc, correction, tla.FixpipeL0C2UBParams(
                dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                sub_block_idx=None))
            tla.cube.mmad(qg, state_l1, chunk_acc, clear_accum=True)
            tla.copy_l0c_to_ub(chunk_acc, inter, tla.FixpipeL0C2UBParams(
                dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                sub_block_idx=None))

        with tla.scope.vector().task('publish_correction', stage=COMPUTE_STAGE):
            with tla.scope.simd():
                half = tla.simd.pge(BV, 128)
                for row in tl.range(CT):
                    value = tla.simd.vextf(tla.simd.vlds(u[row, 0], dist='UNPK_B16'), tl.float32)
                    residual = tla.simd.vsub(value, tla.simd.vlds(correction[row, 0]))
                    values = tla.simd.vtruncf(residual, tl.bfloat16)
                    packed, unused = tla.simd.vdintlv(values, values)
                    tla.simd.vsstb(packed, corr_nz[0, row * 16], (CT + 1) * 16, mask=half)
            tla.copy_ub_to_l1(tla.view(corr_nz, (BV // 16, CT * 16)),
                             tla.view(tla.reshape(corr_l1, (BV // 16, BT * 16)), (BV // 16, CT * 16),
                                      offsets=(0, sub * CT * 16)))

        with tla.scope.cube().task('form_delta', stage=COMPUTE_STAGE):
            tla.cube.mmad(kg, corr_l1, state_acc, transpose_a=True, clear_accum=True)
            tla.copy_l0c_to_ub(state_acc, delta, tla.FixpipeL0C2UBParams(
                dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                sub_block_idx=None))
            tla.cube.mmad(aqk, corr_l1, chunk_acc, clear_accum=True)
            tla.copy_l0c_to_ub(chunk_acc, intra, tla.FixpipeL0C2UBParams(
                dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                sub_block_idx=None))

        with tla.scope.vector().task('advance_state', stage=COMPUTE_STAGE):
            with tla.scope.simd():
                for row in tl.range(CK):
                    # Keep the prefetched gate read-only. Broadcasting a lane
                    # after writing the decay back into UB needs a dependency
                    # between the SIMD store and the following load.
                    log2_decay = tla.simd.vlds(gate[row], dist='BRC_B32')
                    decay = tla.simd.vexp(tla.simd.vmuls(log2_decay, 0.6931471805599453))
                    updated = tla.simd.vfma(tla.simd.vlds(delta[row, 0]), tla.simd.vlds(state[row, 0]), decay)
                    tla.simd.vsts(state[row, 0], updated)

        with tla.scope.vector().task('store_output', stage=COMPUTE_STAGE):
            with tla.scope.simd():
                scale = tla.simd.vsplat(SCALE, tl.float32)
                for row in tl.range(CT):
                    value = tla.simd.vfma(tla.simd.vlds(intra[row, 0]), tla.simd.vlds(inter[row, 0]), scale)
                    bits = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'uint32')
                    tla.simd.vsts(output[row, 0], bits, dist='PK_B32')
            offset = (token_head + sub * CT * H) * V + column
            tla.store_ub_to_gm(output, tla.view(O + offset, (CT, BV), (H * V, 1)))

    with tla.scope.vector():
        tla.store_ub_to_gm(state, tla.view(HT + state_offset, (CK, BV), (V, 1)))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, k, v, gate, beta, initial_state, *, compute_stage=1):
    """Return a full prefill call from normalized Q/K, log-decay and delta beta."""
    import torch
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    batch, tokens, heads, dk = q.shape
    dv = v.shape[-1]
    chunk = 64
    if tokens <= 0 or tokens % chunk or dk != 128 or dv != 128:
        raise ValueError('Expected T divisible by 64 with K = V = 128')
    if k.shape != q.shape or v.shape != q.shape or gate.shape != q.shape or beta.shape != q.shape[:3]:
        raise ValueError('Expected equal Q/K/V/G shapes [B,T,H,128] and beta [B,T,H]')
    if initial_state.shape != (batch, heads, dk, dv):
        raise ValueError('Expected state [B,H,K,V]')
    tensors = q, k, v, gate, beta, initial_state
    dtypes = torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.float32, torch.bfloat16, torch.float32
    if any(x.dtype != dtype or x.device != q.device or not x.is_contiguous() for x, dtype in zip(tensors, dtypes)):
        raise ValueError('Expected contiguous BF16 Q/K/V/beta and FP32 gate/state on one NPU')
    if q.device.type != 'npu' or compute_stage not in (0, 1):
        raise ValueError('Expected an NPU device and compute stage 0 or 1')
    aic, aiv = device_cores(q.device)
    if aiv != 2 * aic:
        raise ValueError(f'Expected a 1:2 Cube/Vector ratio, got {aic}:{aiv}')

    scale = dk**-0.5
    grid = (batch * heads * (dv // 64),)

    def call():
        g = chunk_local_cumsum(g=gate, chunk_size=chunk, scale=RCP_LN2)
        w, u, qg, kg, aqk, _ = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g, beta=beta, scale=scale, chunk_size=chunk, disable_recompute=True)
        output = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)
        kda_scan[grid](w, u, qg, kg, aqk, g, initial_state, output, final_state, tokens, heads, dk, dv,
                       chunk, scale, compute_stage, enable_auto_bind_sub_block=False, enable_preload=False)
        return output, final_state

    call.metadata = dict(grid=grid, chunk=chunk, value_tile=64, compute_stage=compute_stage)
    return call
