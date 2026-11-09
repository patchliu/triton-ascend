"""Standalone FA8-only optimized kernel, generated from fa8_optimized.template.py.

Fixed B4/H8/S1024/D64 BF16, dense; no dependency on the multi-shape dispatcher
or source expander at runtime. Static six-phase ring expansion is preserved.
"""
import hashlib
from pathlib import Path
import torch
import torch_npu
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction
from triton.runtime import driver

B, H, S, D = 4, 8, 1024, 64
OPTIONS = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)


@TLAJITFunction
def _fa8_optimized(Q, K, V, O, SM, SL):
    S: tl.constexpr = 1024
    D: tl.constexpr = 64
    H: tl.constexpr = 8
    BM: tl.constexpr = 128
    BN: tl.constexpr = 256
    TILES: tl.constexpr = 256
    CORES: tl.constexpr = 28
    PAD: tl.constexpr = 1
    VM: tl.constexpr = BM // 2
    SCALE: tl.constexpr = D**(-0.5)
    tl.static_assert(VM == 64 and BN >= 128 and (BN <= 512) and True)
    p_ub = tla.alloc((VM // 16 * 2, (BN // 2 + PAD) * 16), tl.bfloat16, space='ub')
    acc_ub = tla.alloc((VM, D), tl.float32, space='ub')
    out_ub = tla.alloc((VM, D), tl.bfloat16, space='ub')
    m_ub = tla.alloc((192, 1), tl.float32, space='ub')
    l_ub = tla.alloc((192, 1), tl.float32, space='ub')
    m_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    l_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    stat_m = tla.alloc((VM, 8), tl.float32, space='ub')
    stat_l = tla.alloc((VM, 8), tl.float32, space='ub')
    tl.static_assert(D == 64)
    pv_ub0 = tla.alloc((VM, D), tl.float32, space='ub')
    pv_ub1 = tla.alloc((VM, D), tl.float32, space='ub')
    q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
    k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    v_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    s0 = tla.alloc((BN, VM), tl.float32, space='ub')
    p0 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf0 = tla.reshape(p0, (BM // 16, BN * 16))
    alpha0 = tla.alloc((64, 1), tl.float32, space='ub')
    s1 = tla.alloc((BN, VM), tl.float32, space='ub')
    p1 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf1 = tla.reshape(p1, (BM // 16, BN * 16))
    alpha1 = tla.alloc((64, 1), tl.float32, space='ub')
    p2 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf2 = tla.reshape(p2, (BM // 16, BN * 16))
    alpha2 = tla.alloc((64, 1), tl.float32, space='ub')
    with tla.scope.vector():
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 12)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
        with tla.scope.simd():
            zeros = tla.simd.vfull(0.0, tl.float32, 64)
            for state in tl.static_range(3):
                tla.simd.vsts(m_ub[state * 64, 0], zeros)
                tla.simd.vsts(l_ub[state * 64, 0], zeros)
            tla.simd.vsts(m_tmp[0, 0], zeros)
            tla.simd.vsts(l_tmp[0, 0], zeros)
    query_count = (TILES - 1 - tl.program_id(0)) // CORES + 1
    task_count = 0
    for q_index in tl.range(query_count):
        count_pid = tl.program_id(0) + q_index * CORES
        count_bh = count_pid // (S // BM)
        count_row = count_pid % (S // BM) % (S // BM) * BM
        count_end = S
        task_count += (count_end + BN - 1) // BN
    c_query = 0
    c_key = 0
    p_query = 0
    p_key = 0
    sm_query = 0
    sm_key = 0
    u_query = 0
    u_key = 0
    for step in tl.range(0, task_count + 3, 6):
        t = step + 0
        if t < task_count:
            c_pid = tl.program_id(0) + c_query * CORES
            c_bh = c_pid // (S // BM)
            c_row = c_pid % (S // BM) % (S // BM) * BM
            c_base = c_bh * S * D
            c_end = S
            with tla.scope.cube():
                if c_key == 0:
                    tla.load_gm_to_l1(tla.view(Q + c_base, (BM, D), strides=(D, 1), offsets=(c_row, 0)), q_l1)
                tla.load_gm_to_l1(tla.view(K + c_base, (BN, D), strides=(D, 1), offsets=(c_key, 0)), k_l1)
                score_l0c = tla.alloc((BM // 16, BN // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(k_l1, q_l1, score_l0c, transpose_b=True, clear_accum=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
                tla.copy_l0c_to_ub(
                    score_l0c, s0,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 0)
            c_key += BN
            if c_key >= c_end:
                c_query += 1
                c_key = 0
        if t >= 2 and t < task_count + 2:
            p_pid = tl.program_id(0) + p_query * CORES
            p_bh = p_pid // (S // BM)
            p_row = p_pid % (S // BM) % (S // BM) * BM
            p_base = p_bh * S * D
            p_end = S
            with tla.scope.cube():
                tla.load_gm_to_l1(tla.view(V + p_base, (BN, D), strides=(D, 1), offsets=(p_key, 0)), v_l1)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 5)
                pv_l0c = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(p1, v_l1, pv_l0c, clear_accum=True, transpose_a=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
                tla.copy_l0c_to_ub(
                    pv_l0c, pv_ub0,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 2)
            p_key += BN
            if p_key >= p_end:
                p_query += 1
                p_key = 0
        if t >= 1 and t < task_count + 1:
            sm_pid = tl.program_id(0) + sm_query * CORES
            sm_bh = sm_pid // (S // BM)
            sm_row = sm_pid % (S // BM) % (S // BM) * BM
            sm_base = sm_bh * S * D
            sm_end = S
            sm_state = sm_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                if sm_key == 0:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i * 4, 0])
                            x1 = tla.simd.vlds(s1[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s1[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s1[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s1[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s1[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s1[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                else:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i * 4, 0])
                            x1 = tla.simd.vlds(s1[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s1[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s1[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        old_m = tla.simd.vlds(m_ub[sm_state, 0])
                        new_m = tla.simd.vmax(new_m, old_m)
                        alpha_v = tla.simd.vexpdif(old_m, new_m)
                        tla.simd.vsts(alpha2[0, 0], alpha_v)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s1[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s1[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s1[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        old_l = tla.simd.vlds(l_ub[sm_state, 0])
                        new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), new_l)
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(0, 0)),
                                  tla.view(pf2, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), 0)))
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(VM // 16, 0)),
                                  tla.view(pf2, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), BN // 2 * 16)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 13)
                if sm_key + BN >= sm_end:
                    with tla.scope.simd():
                        bits = tla.simd.vbitcast(tla.simd.vlds(m_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_m[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_m[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_m[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_m[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_m[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_m[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_m[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_m[56, 0], tla.simd.vbitcast(c7, 'f32'))
                        bits = tla.simd.vbitcast(tla.simd.vlds(l_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_l[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_l[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_l[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_l[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_l[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_l[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_l[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_l[56, 0], tla.simd.vbitcast(c7, 'f32'))
                    tla.store_ub_to_gm(
                        stat_m, tla.view(SM + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
                    tla.store_ub_to_gm(
                        stat_l, tla.view(SL + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
            sm_key += BN
            if sm_key >= sm_end:
                sm_query += 1
                sm_key = 0
        if t >= 3 and t < task_count + 3:
            u_pid = tl.program_id(0) + u_query * CORES
            u_bh = u_pid // (S // BM)
            u_row = u_pid % (S // BM) % (S // BM) * BM
            u_base = u_bh * S * D
            u_end = S
            u_state = u_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 6)
                if u_key + BN < u_end:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub1[r, col * 64])
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha0[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub1[r, col * 64]))
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                else:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub1[r, col * 64])
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha0[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub1[r, col * 64]))
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    tla.store_ub_to_gm(out_ub,
                                       tla.view(O + u_base, (VM, D), strides=(D, 1), offsets=(u_row + sub * VM, 0)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 12)
            u_key += BN
            if u_key >= u_end:
                u_query += 1
                u_key = 0
        t = step + 1
        if t < task_count:
            c_pid = tl.program_id(0) + c_query * CORES
            c_bh = c_pid // (S // BM)
            c_row = c_pid % (S // BM) % (S // BM) * BM
            c_base = c_bh * S * D
            c_end = S
            with tla.scope.cube():
                if c_key == 0:
                    tla.load_gm_to_l1(tla.view(Q + c_base, (BM, D), strides=(D, 1), offsets=(c_row, 0)), q_l1)
                tla.load_gm_to_l1(tla.view(K + c_base, (BN, D), strides=(D, 1), offsets=(c_key, 0)), k_l1)
                score_l0c = tla.alloc((BM // 16, BN // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(k_l1, q_l1, score_l0c, transpose_b=True, clear_accum=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)
                tla.copy_l0c_to_ub(
                    score_l0c, s1,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 4)
            c_key += BN
            if c_key >= c_end:
                c_query += 1
                c_key = 0
        if t >= 2 and t < task_count + 2:
            p_pid = tl.program_id(0) + p_query * CORES
            p_bh = p_pid // (S // BM)
            p_row = p_pid % (S // BM) % (S // BM) * BM
            p_base = p_bh * S * D
            p_end = S
            with tla.scope.cube():
                tla.load_gm_to_l1(tla.view(V + p_base, (BN, D), strides=(D, 1), offsets=(p_key, 0)), v_l1)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 13)
                pv_l0c = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(p2, v_l1, pv_l0c, clear_accum=True, transpose_a=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 12)
                tla.copy_l0c_to_ub(
                    pv_l0c, pv_ub1,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 6)
            p_key += BN
            if p_key >= p_end:
                p_query += 1
                p_key = 0
        if t >= 1 and t < task_count + 1:
            sm_pid = tl.program_id(0) + sm_query * CORES
            sm_bh = sm_pid // (S // BM)
            sm_row = sm_pid % (S // BM) % (S // BM) * BM
            sm_base = sm_bh * S * D
            sm_end = S
            sm_state = sm_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
                if sm_key == 0:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i * 4, 0])
                            x1 = tla.simd.vlds(s0[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s0[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s0[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s0[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s0[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s0[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                else:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i * 4, 0])
                            x1 = tla.simd.vlds(s0[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s0[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s0[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        old_m = tla.simd.vlds(m_ub[sm_state, 0])
                        new_m = tla.simd.vmax(new_m, old_m)
                        alpha_v = tla.simd.vexpdif(old_m, new_m)
                        tla.simd.vsts(alpha0[0, 0], alpha_v)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s0[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s0[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s0[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        old_l = tla.simd.vlds(l_ub[sm_state, 0])
                        new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), new_l)
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(0, 0)),
                                  tla.view(pf0, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), 0)))
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(VM // 16, 0)),
                                  tla.view(pf0, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), BN // 2 * 16)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 1)
                if sm_key + BN >= sm_end:
                    with tla.scope.simd():
                        bits = tla.simd.vbitcast(tla.simd.vlds(m_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_m[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_m[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_m[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_m[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_m[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_m[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_m[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_m[56, 0], tla.simd.vbitcast(c7, 'f32'))
                        bits = tla.simd.vbitcast(tla.simd.vlds(l_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_l[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_l[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_l[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_l[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_l[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_l[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_l[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_l[56, 0], tla.simd.vbitcast(c7, 'f32'))
                    tla.store_ub_to_gm(
                        stat_m, tla.view(SM + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
                    tla.store_ub_to_gm(
                        stat_l, tla.view(SL + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
            sm_key += BN
            if sm_key >= sm_end:
                sm_query += 1
                sm_key = 0
        if t >= 3 and t < task_count + 3:
            u_pid = tl.program_id(0) + u_query * CORES
            u_bh = u_pid // (S // BM)
            u_row = u_pid % (S // BM) % (S // BM) * BM
            u_base = u_bh * S * D
            u_end = S
            u_state = u_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2)
                if u_key + BN < u_end:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub0[r, col * 64])
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha1[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub0[r, col * 64]))
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                else:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub0[r, col * 64])
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha1[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub0[r, col * 64]))
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    tla.store_ub_to_gm(out_ub,
                                       tla.view(O + u_base, (VM, D), strides=(D, 1), offsets=(u_row + sub * VM, 0)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
            u_key += BN
            if u_key >= u_end:
                u_query += 1
                u_key = 0
        t = step + 2
        if t < task_count:
            c_pid = tl.program_id(0) + c_query * CORES
            c_bh = c_pid // (S // BM)
            c_row = c_pid % (S // BM) % (S // BM) * BM
            c_base = c_bh * S * D
            c_end = S
            with tla.scope.cube():
                if c_key == 0:
                    tla.load_gm_to_l1(tla.view(Q + c_base, (BM, D), strides=(D, 1), offsets=(c_row, 0)), q_l1)
                tla.load_gm_to_l1(tla.view(K + c_base, (BN, D), strides=(D, 1), offsets=(c_key, 0)), k_l1)
                score_l0c = tla.alloc((BM // 16, BN // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(k_l1, q_l1, score_l0c, transpose_b=True, clear_accum=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
                tla.copy_l0c_to_ub(
                    score_l0c, s0,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 0)
            c_key += BN
            if c_key >= c_end:
                c_query += 1
                c_key = 0
        if t >= 2 and t < task_count + 2:
            p_pid = tl.program_id(0) + p_query * CORES
            p_bh = p_pid // (S // BM)
            p_row = p_pid % (S // BM) % (S // BM) * BM
            p_base = p_bh * S * D
            p_end = S
            with tla.scope.cube():
                tla.load_gm_to_l1(tla.view(V + p_base, (BN, D), strides=(D, 1), offsets=(p_key, 0)), v_l1)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
                pv_l0c = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(p0, v_l1, pv_l0c, clear_accum=True, transpose_a=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
                tla.copy_l0c_to_ub(
                    pv_l0c, pv_ub0,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 2)
            p_key += BN
            if p_key >= p_end:
                p_query += 1
                p_key = 0
        if t >= 1 and t < task_count + 1:
            sm_pid = tl.program_id(0) + sm_query * CORES
            sm_bh = sm_pid // (S // BM)
            sm_row = sm_pid % (S // BM) % (S // BM) * BM
            sm_base = sm_bh * S * D
            sm_end = S
            sm_state = sm_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                if sm_key == 0:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i * 4, 0])
                            x1 = tla.simd.vlds(s1[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s1[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s1[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s1[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s1[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s1[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                else:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i * 4, 0])
                            x1 = tla.simd.vlds(s1[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s1[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s1[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        old_m = tla.simd.vlds(m_ub[sm_state, 0])
                        new_m = tla.simd.vmax(new_m, old_m)
                        alpha_v = tla.simd.vexpdif(old_m, new_m)
                        tla.simd.vsts(alpha1[0, 0], alpha_v)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s1[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s1[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s1[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        old_l = tla.simd.vlds(l_ub[sm_state, 0])
                        new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), new_l)
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(0, 0)),
                                  tla.view(pf1, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), 0)))
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(VM // 16, 0)),
                                  tla.view(pf1, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), BN // 2 * 16)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 5)
                if sm_key + BN >= sm_end:
                    with tla.scope.simd():
                        bits = tla.simd.vbitcast(tla.simd.vlds(m_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_m[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_m[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_m[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_m[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_m[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_m[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_m[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_m[56, 0], tla.simd.vbitcast(c7, 'f32'))
                        bits = tla.simd.vbitcast(tla.simd.vlds(l_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_l[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_l[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_l[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_l[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_l[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_l[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_l[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_l[56, 0], tla.simd.vbitcast(c7, 'f32'))
                    tla.store_ub_to_gm(
                        stat_m, tla.view(SM + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
                    tla.store_ub_to_gm(
                        stat_l, tla.view(SL + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
            sm_key += BN
            if sm_key >= sm_end:
                sm_query += 1
                sm_key = 0
        if t >= 3 and t < task_count + 3:
            u_pid = tl.program_id(0) + u_query * CORES
            u_bh = u_pid // (S // BM)
            u_row = u_pid % (S // BM) % (S // BM) * BM
            u_base = u_bh * S * D
            u_end = S
            u_state = u_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 6)
                if u_key + BN < u_end:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub1[r, col * 64])
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha2[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub1[r, col * 64]))
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                else:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub1[r, col * 64])
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha2[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub1[r, col * 64]))
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    tla.store_ub_to_gm(out_ub,
                                       tla.view(O + u_base, (VM, D), strides=(D, 1), offsets=(u_row + sub * VM, 0)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 12)
            u_key += BN
            if u_key >= u_end:
                u_query += 1
                u_key = 0
        t = step + 3
        if t < task_count:
            c_pid = tl.program_id(0) + c_query * CORES
            c_bh = c_pid // (S // BM)
            c_row = c_pid % (S // BM) % (S // BM) * BM
            c_base = c_bh * S * D
            c_end = S
            with tla.scope.cube():
                if c_key == 0:
                    tla.load_gm_to_l1(tla.view(Q + c_base, (BM, D), strides=(D, 1), offsets=(c_row, 0)), q_l1)
                tla.load_gm_to_l1(tla.view(K + c_base, (BN, D), strides=(D, 1), offsets=(c_key, 0)), k_l1)
                score_l0c = tla.alloc((BM // 16, BN // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(k_l1, q_l1, score_l0c, transpose_b=True, clear_accum=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)
                tla.copy_l0c_to_ub(
                    score_l0c, s1,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 4)
            c_key += BN
            if c_key >= c_end:
                c_query += 1
                c_key = 0
        if t >= 2 and t < task_count + 2:
            p_pid = tl.program_id(0) + p_query * CORES
            p_bh = p_pid // (S // BM)
            p_row = p_pid % (S // BM) % (S // BM) * BM
            p_base = p_bh * S * D
            p_end = S
            with tla.scope.cube():
                tla.load_gm_to_l1(tla.view(V + p_base, (BN, D), strides=(D, 1), offsets=(p_key, 0)), v_l1)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 5)
                pv_l0c = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(p1, v_l1, pv_l0c, clear_accum=True, transpose_a=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 12)
                tla.copy_l0c_to_ub(
                    pv_l0c, pv_ub1,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 6)
            p_key += BN
            if p_key >= p_end:
                p_query += 1
                p_key = 0
        if t >= 1 and t < task_count + 1:
            sm_pid = tl.program_id(0) + sm_query * CORES
            sm_bh = sm_pid // (S // BM)
            sm_row = sm_pid % (S // BM) % (S // BM) * BM
            sm_base = sm_bh * S * D
            sm_end = S
            sm_state = sm_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
                if sm_key == 0:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i * 4, 0])
                            x1 = tla.simd.vlds(s0[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s0[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s0[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s0[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s0[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s0[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                else:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i * 4, 0])
                            x1 = tla.simd.vlds(s0[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s0[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s0[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        old_m = tla.simd.vlds(m_ub[sm_state, 0])
                        new_m = tla.simd.vmax(new_m, old_m)
                        alpha_v = tla.simd.vexpdif(old_m, new_m)
                        tla.simd.vsts(alpha2[0, 0], alpha_v)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s0[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s0[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s0[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        old_l = tla.simd.vlds(l_ub[sm_state, 0])
                        new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), new_l)
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(0, 0)),
                                  tla.view(pf2, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), 0)))
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(VM // 16, 0)),
                                  tla.view(pf2, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), BN // 2 * 16)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 13)
                if sm_key + BN >= sm_end:
                    with tla.scope.simd():
                        bits = tla.simd.vbitcast(tla.simd.vlds(m_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_m[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_m[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_m[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_m[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_m[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_m[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_m[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_m[56, 0], tla.simd.vbitcast(c7, 'f32'))
                        bits = tla.simd.vbitcast(tla.simd.vlds(l_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_l[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_l[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_l[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_l[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_l[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_l[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_l[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_l[56, 0], tla.simd.vbitcast(c7, 'f32'))
                    tla.store_ub_to_gm(
                        stat_m, tla.view(SM + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
                    tla.store_ub_to_gm(
                        stat_l, tla.view(SL + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
            sm_key += BN
            if sm_key >= sm_end:
                sm_query += 1
                sm_key = 0
        if t >= 3 and t < task_count + 3:
            u_pid = tl.program_id(0) + u_query * CORES
            u_bh = u_pid // (S // BM)
            u_row = u_pid % (S // BM) % (S // BM) * BM
            u_base = u_bh * S * D
            u_end = S
            u_state = u_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2)
                if u_key + BN < u_end:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub0[r, col * 64])
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha0[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub0[r, col * 64]))
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                else:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub0[r, col * 64])
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha0[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub0[r, col * 64]))
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    tla.store_ub_to_gm(out_ub,
                                       tla.view(O + u_base, (VM, D), strides=(D, 1), offsets=(u_row + sub * VM, 0)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
            u_key += BN
            if u_key >= u_end:
                u_query += 1
                u_key = 0
        t = step + 4
        if t < task_count:
            c_pid = tl.program_id(0) + c_query * CORES
            c_bh = c_pid // (S // BM)
            c_row = c_pid % (S // BM) % (S // BM) * BM
            c_base = c_bh * S * D
            c_end = S
            with tla.scope.cube():
                if c_key == 0:
                    tla.load_gm_to_l1(tla.view(Q + c_base, (BM, D), strides=(D, 1), offsets=(c_row, 0)), q_l1)
                tla.load_gm_to_l1(tla.view(K + c_base, (BN, D), strides=(D, 1), offsets=(c_key, 0)), k_l1)
                score_l0c = tla.alloc((BM // 16, BN // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(k_l1, q_l1, score_l0c, transpose_b=True, clear_accum=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
                tla.copy_l0c_to_ub(
                    score_l0c, s0,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 0)
            c_key += BN
            if c_key >= c_end:
                c_query += 1
                c_key = 0
        if t >= 2 and t < task_count + 2:
            p_pid = tl.program_id(0) + p_query * CORES
            p_bh = p_pid // (S // BM)
            p_row = p_pid % (S // BM) % (S // BM) * BM
            p_base = p_bh * S * D
            p_end = S
            with tla.scope.cube():
                tla.load_gm_to_l1(tla.view(V + p_base, (BN, D), strides=(D, 1), offsets=(p_key, 0)), v_l1)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 13)
                pv_l0c = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(p2, v_l1, pv_l0c, clear_accum=True, transpose_a=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
                tla.copy_l0c_to_ub(
                    pv_l0c, pv_ub0,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 2)
            p_key += BN
            if p_key >= p_end:
                p_query += 1
                p_key = 0
        if t >= 1 and t < task_count + 1:
            sm_pid = tl.program_id(0) + sm_query * CORES
            sm_bh = sm_pid // (S // BM)
            sm_row = sm_pid % (S // BM) % (S // BM) * BM
            sm_base = sm_bh * S * D
            sm_end = S
            sm_state = sm_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                if sm_key == 0:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i * 4, 0])
                            x1 = tla.simd.vlds(s1[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s1[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s1[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s1[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s1[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s1[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                else:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i * 4, 0])
                            x1 = tla.simd.vlds(s1[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s1[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s1[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        old_m = tla.simd.vlds(m_ub[sm_state, 0])
                        new_m = tla.simd.vmax(new_m, old_m)
                        alpha_v = tla.simd.vexpdif(old_m, new_m)
                        tla.simd.vsts(alpha0[0, 0], alpha_v)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s1[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s1[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s1[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s1[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        old_l = tla.simd.vlds(l_ub[sm_state, 0])
                        new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), new_l)
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(0, 0)),
                                  tla.view(pf0, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), 0)))
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(VM // 16, 0)),
                                  tla.view(pf0, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), BN // 2 * 16)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 1)
                if sm_key + BN >= sm_end:
                    with tla.scope.simd():
                        bits = tla.simd.vbitcast(tla.simd.vlds(m_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_m[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_m[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_m[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_m[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_m[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_m[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_m[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_m[56, 0], tla.simd.vbitcast(c7, 'f32'))
                        bits = tla.simd.vbitcast(tla.simd.vlds(l_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_l[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_l[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_l[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_l[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_l[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_l[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_l[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_l[56, 0], tla.simd.vbitcast(c7, 'f32'))
                    tla.store_ub_to_gm(
                        stat_m, tla.view(SM + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
                    tla.store_ub_to_gm(
                        stat_l, tla.view(SL + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
            sm_key += BN
            if sm_key >= sm_end:
                sm_query += 1
                sm_key = 0
        if t >= 3 and t < task_count + 3:
            u_pid = tl.program_id(0) + u_query * CORES
            u_bh = u_pid // (S // BM)
            u_row = u_pid % (S // BM) % (S // BM) * BM
            u_base = u_bh * S * D
            u_end = S
            u_state = u_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 6)
                if u_key + BN < u_end:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub1[r, col * 64])
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha1[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub1[r, col * 64]))
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                else:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub1[r, col * 64])
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha1[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub1[r, col * 64]))
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    tla.store_ub_to_gm(out_ub,
                                       tla.view(O + u_base, (VM, D), strides=(D, 1), offsets=(u_row + sub * VM, 0)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 12)
            u_key += BN
            if u_key >= u_end:
                u_query += 1
                u_key = 0
        t = step + 5
        if t < task_count:
            c_pid = tl.program_id(0) + c_query * CORES
            c_bh = c_pid // (S // BM)
            c_row = c_pid % (S // BM) % (S // BM) * BM
            c_base = c_bh * S * D
            c_end = S
            with tla.scope.cube():
                if c_key == 0:
                    tla.load_gm_to_l1(tla.view(Q + c_base, (BM, D), strides=(D, 1), offsets=(c_row, 0)), q_l1)
                tla.load_gm_to_l1(tla.view(K + c_base, (BN, D), strides=(D, 1), offsets=(c_key, 0)), k_l1)
                score_l0c = tla.alloc((BM // 16, BN // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(k_l1, q_l1, score_l0c, transpose_b=True, clear_accum=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)
                tla.copy_l0c_to_ub(
                    score_l0c, s1,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 4)
            c_key += BN
            if c_key >= c_end:
                c_query += 1
                c_key = 0
        if t >= 2 and t < task_count + 2:
            p_pid = tl.program_id(0) + p_query * CORES
            p_bh = p_pid // (S // BM)
            p_row = p_pid % (S // BM) % (S // BM) * BM
            p_base = p_bh * S * D
            p_end = S
            with tla.scope.cube():
                tla.load_gm_to_l1(tla.view(V + p_base, (BN, D), strides=(D, 1), offsets=(p_key, 0)), v_l1)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
                pv_l0c = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space='l0c')
                tla.cube.mmad(p0, v_l1, pv_l0c, clear_accum=True, transpose_a=True)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 12)
                tla.copy_l0c_to_ub(
                    pv_l0c, pv_ub1,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 6)
            p_key += BN
            if p_key >= p_end:
                p_query += 1
                p_key = 0
        if t >= 1 and t < task_count + 1:
            sm_pid = tl.program_id(0) + sm_query * CORES
            sm_bh = sm_pid // (S // BM)
            sm_row = sm_pid % (S // BM) % (S // BM) * BM
            sm_base = sm_bh * S * D
            sm_end = S
            sm_state = sm_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
                if sm_key == 0:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i * 4, 0])
                            x1 = tla.simd.vlds(s0[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s0[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s0[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s0[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s0[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s0[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                else:
                    with tla.scope.simd():
                        scale = tla.simd.vfull(SCALE, tl.float32, 64)
                        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        mx0 = neg_inf
                        mx1 = neg_inf
                        mx2 = neg_inf
                        mx3 = neg_inf
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i * 4, 0])
                            x1 = tla.simd.vlds(s0[i * 4 + 1, 0])
                            x2 = tla.simd.vlds(s0[i * 4 + 2, 0])
                            x3 = tla.simd.vlds(s0[i * 4 + 3, 0])
                            mx0 = tla.simd.vmax(mx0, x0)
                            mx1 = tla.simd.vmax(mx1, x1)
                            mx2 = tla.simd.vmax(mx2, x2)
                            mx3 = tla.simd.vmax(mx3, x3)
                        new_m = tla.simd.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
                        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
                        old_m = tla.simd.vlds(m_ub[sm_state, 0])
                        new_m = tla.simd.vmax(new_m, old_m)
                        alpha_v = tla.simd.vexpdif(old_m, new_m)
                        tla.simd.vsts(alpha1[0, 0], alpha_v)
                        tla.simd.vsts(m_ub[sm_state, 0], new_m)
                        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
                        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.range(0, BN // 4):
                            x0 = tla.simd.vlds(s0[i + 0 * (BN // 4), 0])
                            x1 = tla.simd.vlds(s0[i + 1 * (BN // 4), 0])
                            x2 = tla.simd.vlds(s0[i + 2 * (BN // 4), 0])
                            x3 = tla.simd.vlds(s0[i + 3 * (BN // 4), 0])
                            x0 = tla.simd.vmuls(x0, SCALE)
                            x1 = tla.simd.vmuls(x1, SCALE)
                            x2 = tla.simd.vmuls(x2, SCALE)
                            x3 = tla.simd.vmuls(x3, SCALE)
                            e0 = tla.simd.vexpdif(x0, new_m)
                            e1 = tla.simd.vexpdif(x1, new_m)
                            e2 = tla.simd.vexpdif(x2, new_m)
                            e3 = tla.simd.vexpdif(x3, new_m)
                            sum0 = tla.simd.vadd(sum0, e0)
                            sum1 = tla.simd.vadd(sum1, e1)
                            sum2 = tla.simd.vadd(sum2, e2)
                            sum3 = tla.simd.vadd(sum3, e3)
                            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
                            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
                            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
                            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
                            pa, unused_a = tla.simd.vdintlv(pc0, pc2)
                            pb, unused_b = tla.simd.vdintlv(pc1, pc3)
                            tla.simd.vsstb(pa, p_ub[0, i * 16], (BN // 2 + PAD) * 16)
                            tla.simd.vsstb(pb, p_ub[0, (i + BN // 4) * 16], (BN // 2 + PAD) * 16)
                        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
                        old_l = tla.simd.vlds(l_ub[sm_state, 0])
                        new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), new_l)
                        tla.simd.vsts(l_ub[sm_state, 0], new_l)
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(0, 0)),
                                  tla.view(pf1, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), 0)))
                tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(VM // 16, 0)),
                                  tla.view(pf1, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), BN // 2 * 16)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 5)
                if sm_key + BN >= sm_end:
                    with tla.scope.simd():
                        bits = tla.simd.vbitcast(tla.simd.vlds(m_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_m[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_m[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_m[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_m[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_m[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_m[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_m[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_m[56, 0], tla.simd.vbitcast(c7, 'f32'))
                        bits = tla.simd.vbitcast(tla.simd.vlds(l_ub[sm_state, 0]), 'i32')
                        a0, a1 = tla.simd.vintlv(bits, bits)
                        b0, b1 = tla.simd.vintlv(a0, a0)
                        c0, c1 = tla.simd.vintlv(b0, b0)
                        tla.simd.vsts(stat_l[0, 0], tla.simd.vbitcast(c0, 'f32'))
                        tla.simd.vsts(stat_l[8, 0], tla.simd.vbitcast(c1, 'f32'))
                        c2, c3 = tla.simd.vintlv(b1, b1)
                        tla.simd.vsts(stat_l[16, 0], tla.simd.vbitcast(c2, 'f32'))
                        tla.simd.vsts(stat_l[24, 0], tla.simd.vbitcast(c3, 'f32'))
                        b2, b3 = tla.simd.vintlv(a1, a1)
                        c4, c5 = tla.simd.vintlv(b2, b2)
                        c6, c7 = tla.simd.vintlv(b3, b3)
                        tla.simd.vsts(stat_l[32, 0], tla.simd.vbitcast(c4, 'f32'))
                        tla.simd.vsts(stat_l[40, 0], tla.simd.vbitcast(c5, 'f32'))
                        tla.simd.vsts(stat_l[48, 0], tla.simd.vbitcast(c6, 'f32'))
                        tla.simd.vsts(stat_l[56, 0], tla.simd.vbitcast(c7, 'f32'))
                    tla.store_ub_to_gm(
                        stat_m, tla.view(SM + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
                    tla.store_ub_to_gm(
                        stat_l, tla.view(SL + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
            sm_key += BN
            if sm_key >= sm_end:
                sm_query += 1
                sm_key = 0
        if t >= 3 and t < task_count + 3:
            u_pid = tl.program_id(0) + u_query * CORES
            u_bh = u_pid // (S // BM)
            u_row = u_pid % (S // BM) % (S // BM) * BM
            u_base = u_bh * S * D
            u_end = S
            u_state = u_query % 3 * 64
            with tla.scope.vector():
                sub = tla.sub_block_id()
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2)
                if u_key + BN < u_end:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub0[r, col * 64])
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha2[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub0[r, col * 64]))
                                    tla.simd.vsts(acc_ub[r, col * 64], acc)
                else:
                    if u_key == 0:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vlds(pv_ub0[r, col * 64])
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    else:
                        with tla.scope.simd():
                            for r in tl.range(0, VM):
                                alpha = tla.simd.vlds(alpha2[r, 0], dist='BRC_B32')
                                for col in tl.static_range(D // 64):
                                    acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                                    acc = tla.simd.vadd(acc, tla.simd.vlds(pv_ub0[r, col * 64]))
                                    den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                                    value = tla.simd.vdiv(acc, den)
                                    packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                                    tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
                    tla.store_ub_to_gm(out_ub,
                                       tla.view(O + u_base, (VM, D), strides=(D, 1), offsets=(u_row + sub * VM, 0)))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
            u_key += BN
            if u_key >= u_end:
                u_query += 1
                u_key = 0
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 12)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)


def device_cores(device):
    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    return props['num_aicore'], props['num_vectorcore']


def prepare_call(q, k, v, causal=False):
    if type(causal) is not bool or causal:
        raise ValueError("This standalone operator implements dense FA8 only")
    if not all(
            tuple(x.shape) == (B, H, S, D) and x.dtype == torch.bfloat16 and x.device.type == "npu"
            and x.device == q.device and x.is_contiguous() for x in (q, k, v)):
        raise ValueError("FA8 requires contiguous BF16 B4/H8/S1024/D64 on the same NPU")
    if device_cores(q.device) != (28, 56):
        raise ValueError("FA8 comparison requires 28 Cube and 56 Vector cores")
    outputs = (torch.empty_like(q), torch.empty((B, H, S, 8), dtype=torch.float32, device=q.device),
               torch.empty((B, H, S, 8), dtype=torch.float32, device=q.device))

    def call():
        call.compiled = _fa8_optimized[(28, )](q, k, v, *outputs, **OPTIONS)
        return outputs

    call.outputs = outputs
    call.metadata = dict(case="FA8", variant="FA8-only optimized", query_tile=128, key_tile=256,
                         recurrence="original six-phase explicit ring schedule", options=OPTIONS,
                         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return call


def forward(q, k, v, causal=False):
    return prepare_call(q, k, v, causal)()
