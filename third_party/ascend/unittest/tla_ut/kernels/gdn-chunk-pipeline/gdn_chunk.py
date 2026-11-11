"""64-token GDN with a Cube/Vector pipeline across chunks.

The first implementation uses device PyTorch to prepare independent WY
blocks. The TLA kernel consumes those blocks in sequence order and retains
the recurrent state on chip. All preparation is part of the public call.
"""

import triton.language as tl
import triton.language.extra.tla as tla

CHUNK = 64
VALUE_TILE = 32


@tla.jit
def chunk_recurrence(U, W, Q_DECAYED, K_DECAYED, A, END_DECAY, INITIAL, O, FINAL, CHUNKS: tl.constexpr,
                     HV: tl.constexpr, BT: tl.constexpr, DV: tl.constexpr, DK: tl.constexpr, BV: tl.constexpr,
                     CORES: tl.constexpr, COMPUTE_STAGE: tl.constexpr, DIRECT_TND: tl.constexpr = False):
    tl.static_assert(BT == 64 and DK == 128 and DV % BV == 0 and (BV == 32 or BV == 64))
    # Fixpipe row-splits the value tile between the two Vector sub-blocks.
    VM: tl.constexpr = BV // 2
    PITCH: tl.constexpr = (VM + 1) * 16
    sub = tla.sub_block_id()
    for job in tl.range(tl.program_id(0), HV * (DV // BV), CORES):
        head = job // (DV // BV)
        v_start = job % (DV // BV) * BV
        state_base = (head * DV + v_start) * DK
        state_l1 = tla.alloc((DK // 16, BV // 16, 16, 16), tl.bfloat16, space='l1')
        state = tla.alloc((VM, DK), tl.float32, space='ub')
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(INITIAL + state_base, (BV, DK)), state_l1)
        with tla.scope.vector():
            initial = tla.alloc((VM, DK), tl.bfloat16, space='ub')
            tla.load_gm_to_ub(tla.view(INITIAL + state_base + sub * VM * DK, (VM, DK)), initial)
            with tla.scope.simd():
                for row in tl.range(VM):
                    raw = tla.simd.vlds(initial[row, 0])
                    lo, hi = tla.simd.vintlv(raw, raw)
                    tla.simd.vsts(state[row, 0], tla.simd.vextf(lo, tl.float32))
                    tla.simd.vsts(state[row, 64], tla.simd.vextf(hi, tl.float32))

        for chunk in tla.pipeline(CHUNKS):
            index = chunk * HV + head
            qk_base = index * BT * DK
            out_base = (index * DV + v_start + sub * VM) * BT
            w = tla.alloc((DK // 16, BT // 16, 16, 16), tl.bfloat16, space='l1')
            q = tla.alloc((DK // 16, BT // 16, 16, 16), tl.bfloat16, space='l1')
            k = tla.alloc((DK // 16, BT // 16, 16, 16), tl.bfloat16, space='l1')
            a = tla.alloc((BT // 16, BT // 16, 16, 16), tl.bfloat16, space='l1')
            u = tla.alloc((VM, BT), tl.float32, space='ub')
            residual_l1 = tla.alloc((BT // 16, BV // 16, 16, 16), tl.bfloat16, space='l1')
            residual_nz = tla.alloc((8, PITCH), tl.bfloat16, space='ub')
            prediction = tla.alloc((VM, BT), tl.float32, space='ub')
            if not DIRECT_TND:
                output = tla.alloc((VM, BT), tl.float32, space='ub')
                out_bf16 = tla.alloc((VM, BT), tl.bfloat16, space='ub')
            update = tla.alloc((VM, DK), tl.float32, space='ub')
            state_nz = tla.alloc((DK // 16, PITCH), tl.bfloat16, space='ub')
            pred_acc = tla.alloc((BT // 16, BV // 16, 16, 16), tl.float32, space='l0c')
            out_acc = tla.alloc((BT // 16, BV // 16, 16, 16), tl.float32, space='l0c')
            update_acc = tla.alloc((DK // 16, BV // 16, 16, 16), tl.float32, space='l0c')

            with tla.scope.cube().task('load_coefficients', stage=0):
                tla.load_gm_to_l1(tla.view(W + qk_base, (BT, DK)), w)
                tla.load_gm_to_l1(tla.view(Q_DECAYED + qk_base, (BT, DK)), q)
                tla.load_gm_to_l1(tla.view(K_DECAYED + qk_base, (BT, DK)), k)
                tla.load_gm_to_l1(tla.view(A + index * BT * BT, (BT, BT)), a)
            with tla.scope.vector().task('load_values', stage=0):
                tla.load_gm_to_ub(tla.view(U + out_base, (VM, BT)), u)

            with tla.scope.cube().task('predict', stage=COMPUTE_STAGE):
                tla.cube.mmad(state_l1, w, pred_acc, transpose_b=True, clear_accum=True)
                tla.copy_l0c_to_ub(
                    pred_acc, prediction,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
            with tla.scope.vector().task('form_residual', stage=COMPUTE_STAGE):
                with tla.scope.simd():
                    zero = tla.simd.vsplat(0.0, tl.bfloat16)
                    for row in tl.range(VM):
                        delta = tla.simd.vsub(tla.simd.vlds(u[row, 0]), tla.simd.vlds(prediction[row, 0]))
                        packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(delta, tl.bfloat16), zero)
                        tla.simd.vsstb(packed, residual_nz[0, row * 16], PITCH)
                tla.copy_ub_to_l1(
                    tla.view(residual_nz, (BT // 16, VM * 16)),
                    tla.view(tla.reshape(residual_l1, (BT // 16, BV * 16)), (BT // 16, VM * 16),
                             offsets=(0, sub * VM * 16)))

            with tla.scope.cube().task('apply_chunk', stage=COMPUTE_STAGE):
                # O.T = S @ Q_d.T + R.T @ A.T and delta_S = R.T @ K_d.
                tla.cube.mmad(state_l1, q, out_acc, transpose_b=True, clear_accum=True)
                tla.cube.mmad(residual_l1, a, out_acc, transpose_b=True, clear_accum=False)
                if DIRECT_TND:
                    out_tnd = (chunk * BT * HV + head) * DV + v_start
                    tla.store_l0c_to_gm(
                        out_acc, tla.view(O + out_tnd, (BT, BV), strides=(HV * DV, 1)),
                        tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2DN,
                                                pre_quant=tla.FixpipePreQuantMode.F322BF16))
                tla.cube.mmad(residual_l1, k, update_acc, clear_accum=True)
                if not DIRECT_TND:
                    tla.copy_l0c_to_ub(
                        out_acc, output,
                        tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                tla.copy_l0c_to_ub(
                    update_acc, update,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
            with tla.scope.vector().task('advance_state', stage=COMPUTE_STAGE):
                alpha = tl.load(END_DECAY + index)
                with tla.scope.simd():
                    decay = tla.simd.vdup(alpha, tl.float32)
                    if not DIRECT_TND:
                        zero = tla.simd.vsplat(0.0, tl.bfloat16)
                    for row in tl.range(VM):
                        s0 = tla.simd.vfma(tla.simd.vlds(update[row, 0]), tla.simd.vlds(state[row, 0]), decay)
                        s1 = tla.simd.vfma(tla.simd.vlds(update[row, 64]), tla.simd.vlds(state[row, 64]), decay)
                        tla.simd.vsts(state[row, 0], s0)
                        tla.simd.vsts(state[row, 64], s1)
                        packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(s0, tl.bfloat16),
                                                           tla.simd.vtruncf(s1, tl.bfloat16))
                        tla.simd.vsstb(packed, state_nz[0, row * 16], PITCH)
                        if not DIRECT_TND:
                            result, unused = tla.simd.vdintlv(tla.simd.vtruncf(tla.simd.vlds(output[row, 0]), tl.bfloat16),
                                                               zero)
                            tla.simd.vsts(out_bf16[row, 0], result, dist='NORM_B16', mask=tla.simd.pge(BT, 128))
                tla.copy_ub_to_l1(
                    tla.view(state_nz, (DK // 16, VM * 16)),
                    tla.view(tla.reshape(state_l1, (DK // 16, BV * 16)), (DK // 16, VM * 16),
                             offsets=(0, sub * VM * 16)))
                if not DIRECT_TND:
                    tla.store_ub_to_gm(out_bf16, tla.view(O + out_base, (VM, BT)))

        with tla.scope.vector():
            final = tla.alloc((VM, DK), tl.bfloat16, space='ub')
            with tla.scope.simd():
                for row in tl.range(VM):
                    s0 = tla.simd.vtruncf(tla.simd.vlds(state[row, 0]), tl.bfloat16)
                    s1 = tla.simd.vtruncf(tla.simd.vlds(state[row, 64]), tl.bfloat16)
                    packed, unused = tla.simd.vdintlv(s0, s1)
                    tla.simd.vsts(final[row, 0], packed, dist='NORM_B16')
            tla.store_ub_to_gm(final, tla.view(FINAL + state_base + sub * VM * DK, (VM, DK)))


def device_cores(device):
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, k, v, g, beta, initial_state, *, scale=None, prefetch=True, value_tile=VALUE_TILE, wy='doubling',
                 direct_tnd=False):
    """Return one full single-sequence chunk call from raw TND inputs."""
    import torch
    from pathlib import Path
    from kernels.utils import load_kernel

    blocks = load_kernel(Path(__file__).with_name('wy_blocks.py'))
    tokens, hk, dk = q.shape
    tv, hv, dv = v.shape
    if not (tokens > 0 and tokens % CHUNK == 0 and hk > 0 and hv > 0 and hv % hk == 0 and (dk, dv) == (128, 128)):
        raise ValueError('Expected full 64-token chunks, Hv divisible by Hk and Dk = Dv = 128')
    if k.shape != q.shape or tv != tokens or g.shape != (tokens, hv) or beta.shape != (tokens, hv):
        raise ValueError('Expected contiguous TND inputs and one scalar gate per token and value head')
    if initial_state.shape != (1, hv, dv, dk):
        raise ValueError('Expected initial state [1,Hv,Dv,Dk]')
    if type(value_tile) is not int or value_tile not in (32, 64):
        raise ValueError('GDN chunk value_tile must be 32 or 64')
    if wy not in ('doubling', 'fla', 'fused', 'split', 'split16', 'split32'):
        raise ValueError('GDN chunk wy must be doubling, fla, fused, split, split16 or split32')
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, beta, initial_state)) or g.dtype != torch.float32:
        raise ValueError('Expected BF16 Q/K/V/beta/state and FP32 g')
    if q.device.type != 'npu' or any(x.device != q.device or not x.is_contiguous()
                                    for x in (q, k, v, g, beta, initial_state)):
        raise ValueError('Expected contiguous inputs on the same NPU')

    chunks = tokens // CHUNK
    scale = dk**-0.5 if scale is None else float(scale)
    aic, aiv = device_cores(q.device)
    if aic <= 0 or aiv != 2 * aic:
        raise ValueError(f'Expected a 1:2 Cube/Vector ratio, got {aic}:{aiv}')
    cores = min(hv * (dv // value_tile), aic)
    launch_options = dict(grid=cores, chunk=CHUNK, value_tile=value_tile, compute_stage=int(prefetch), wy=wy,
                          direct_tnd=direct_tnd)
    prepare_coefficients = blocks.prepare_wy
    wy_args = {}
    if wy != 'doubling':
        from hashlib import file_digest
        from importlib import import_module

        if wy == 'fla':
            solver = import_module('fla.ops.utils.solve_tril')
            wy_args = dict(solve_tril=solver.solve_tril)
            sources = dict(wy_solver=solver.__file__)
        else:
            from fla.ops.utils.cumsum import chunk_local_cumsum

            fused = load_kernel(Path(__file__).with_name('wy_fused.py'))
            prepare_coefficients = fused.prepare_wy
            if wy == 'split':
                solver = import_module('fla.ops.utils.solve_tril')
                wy_args = dict(cumsum=chunk_local_cumsum, solve_tril=solver.solve_tril)
            elif wy in ('split16', 'split32'):
                from functools import partial

                solver = load_kernel(Path(__file__).with_name('wy_tril.py'))
                diagonal_size = 16 if wy == 'split16' else 32
                wy_args = dict(cumsum=chunk_local_cumsum,
                               solve_tril=partial(solver.solve_tril, diagonal_size=diagonal_size))
            else:
                solver = import_module('fla.ops.gated_delta_rule.chunk_fwd')
                wy_args = dict(cumsum=chunk_local_cumsum, solve_kkt=solver.chunk_gated_delta_rule_fwd_kkt_solve_kernel)
            sources = dict(wy_solver=solver.__file__, wy_fused=fused.__file__)
        for name, filename in sources.items():
            source = Path(filename).resolve()
            with source.open('rb') as stream:
                launch_options[name] = str(source)
                launch_options[f'{name}_sha256'] = file_digest(stream, 'sha256').hexdigest()

    def call():
        coefficients = prepare_coefficients(q, k, v, g, beta, CHUNK, scale, **wy_args)
        out_shape = (tokens, hv, dv) if direct_tnd else (chunks, hv, dv, CHUNK)
        o = torch.empty(out_shape, dtype=q.dtype, device=q.device)
        final = torch.empty_like(initial_state)
        chunk_recurrence[(cores,)](*coefficients, initial_state, o, final, chunks, hv, CHUNK, dv, dk, value_tile, cores,
                                  int(prefetch), direct_tnd, sanitize_overflow=False, enable_auto_bind_sub_block=False,
                                  enable_preload=False)
        return (o if direct_tnd else o.permute(0, 3, 1, 2).reshape(tokens, hv, dv).contiguous()), final

    call.metadata = launch_options
    return call
