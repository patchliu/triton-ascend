"""fa_best workload shape (B4 H8 S=1024 D=64) expressed with TLA pipeline stages.

The goal is a *same-workload* data point for
``issue104_fa_optimized_20261031/README_best.md`` (FA8: S=1024, D=64, dense,
80.69 us on a real Ascend950PR with 28 AIC / 56 AIV).  One program walks a
run of query tiles of one head, streaming the full 1024-key KV row, exactly
like fa_best's per-query-tile work.

Geometry
--------
fa_best uses BM=128 / BN=256 and one K slot, one V slot, two S/P slots, with
hand-written ``block_set_flag`` / ``block_wait_flag`` pairs.  This case keeps
BN=256, D=64 and the full 1024-key KV row (NKB=4 blocks), but has to halve the
query tile to BM=64 and collapse to depth 2; see "Why not BM=128" below.

Budget (per AIV / per AIC, depth = max(stage) + 1 = 2)
------------------------------------------------------
LowerTLAStages stamps every pipeline buffer with ``hivm.multi_buffer = depth``
and EnableMultiBuffer rotates the physical slot from a per-iteration counter,
so the local-buffer need is ``sum(bytes) * depth`` and a buffer reused by two
chunks inside one iteration is a real WAR hazard (serialised by SyncSolver),
not a second slot.

    L1  (cbuf) q 8192 + k 32768 + v 32768 + p 32768        = 131072 B  x2 = 262144 (50%)
    UB         score 32768 + prob 16896 + acc 8192 + pv 8192
             + state 384                                  =  66432 B  x2 = 132864 (65%)
    L0C        qk 65536 + pv 16384                        =  81920 B  x2 = 163840

Why not BM=128: the BM=128 working set is 2x the above (score 64 KB, prob
33 KB, q 16 KB, p 64 KB), so UB alone needs 231168 B per iteration and cannot
be multi-buffered at all (depth 1), i.e. no pipeline.  BM=64 is the largest
query tile that still allows depth 2.

Why depth 2: with one S slot, prob/pv and the next chunk's QK are ordered in
source order inside the same stage, so a third stage buys nothing while it
would push UB past its limit.

Dense attention only (no causal mask, no query rotation, no FP32 row max/sum
outputs), matching the FA8 comparison point.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM/AVE IR structure (8 mmadL1 / 8 fixpipe / 9 nd2nz /
            4 UB->L1 copies / 1 store, no manual sync)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : device launch + accuracy against the torch online-softmax reference
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, lower_offline,
                                  check_compile_artifacts, get_stages)

# fa_best FA8 geometry: KV length 1024 = NKB * BN, head dim 64.
BM = 64          # query rows per tile (fa_best: 128)
BN = 256         # keys per KV block (same as fa_best)
NKB = 4          # KV blocks -> 1024 keys (same as fa_best's S)
D = 64
TILES = 2        # query tiles per launch; only K/V length has to match fa_best
ROWS = TILES * BM
K_ROWS = NKB * BN
SCALE = 1.0 / D**0.5
VM = BM // 2     # rows owned by one AIV sub-block
PSTRIDE = VM * 16 + 16  # UB row stride of the NZ probability buffer


@tla.jit
def fa_shape_stages(Q, K, V, O, TILES: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    NKB: tl.constexpr, D: tl.constexpr, VM: tl.constexpr, SCALE: tl.constexpr,
                    PSTRIDE: tl.constexpr):
    for t in tla.pipeline(TILES):
        q = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        k = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        v = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        p = tla.alloc((BN // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        qk = tla.alloc((BN // 16, BM // 16, 16, 16), tl.float32, space="l0c")
        pv = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space="l0c")
        score = tla.alloc((VM, BN), tl.float32, space="ub")
        prob = tla.alloc((BN // 16, PSTRIDE), tl.bfloat16, space="ub")
        mrun = tla.alloc((VM, 1), tl.float32, space="ub")
        lrun = tla.alloc((VM, 1), tl.float32, space="ub")
        alpha = tla.alloc((VM, 1), tl.float32, space="ub")
        acc = tla.alloc((VM, D), tl.float32, space="ub")
        pvub = tla.alloc((VM, D), tl.float32, space="ub")
        sub = tla.sub_block_id()

        # ---- KV block 0: load Q/K/V, QK, first-block softmax, P->L1, PV ----
        with tla.scope.cube().task("load0", stage=0):
            tla.load_gm_to_l1(tla.view(Q + t * BM * D, (BM, D)), q)
            tla.load_gm_to_l1(tla.view(K, (BN, D)), k)
            tla.load_gm_to_l1(tla.view(V, (BN, D)), v)
        with tla.scope.cube().task("qk0", stage=0):
            tla.cube.mmad(q, k, qk, transpose_b=True, clear_accum=True)
            tla.copy_l0c_to_ub(
                qk, score,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        with tla.scope.vector().task("softmax0", stage=1):
            with tla.scope.simd():
                for row in tl.range(VM):
                    x0 = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                    x1 = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                    x2 = tla.simd.vmuls(tla.simd.vlds(score[row, 128]), SCALE)
                    x3 = tla.simd.vmuls(tla.simd.vlds(score[row, 192]), SCALE)
                    new_m = tla.simd.vreduce_max(
                        tla.simd.vmax(tla.simd.vmax(x0, x1), tla.simd.vmax(x2, x3)))
                    e0 = tla.simd.vexp(tla.simd.vsub(x0, new_m))
                    e1 = tla.simd.vexp(tla.simd.vsub(x1, new_m))
                    e2 = tla.simd.vexp(tla.simd.vsub(x2, new_m))
                    e3 = tla.simd.vexp(tla.simd.vsub(x3, new_m))
                    new_l = tla.simd.vbroadcast(
                        tla.simd.vreduce_sum(
                            tla.simd.vadd(tla.simd.vadd(e0, e1), tla.simd.vadd(e2, e3))))
                    pa, unused_a = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                   tla.simd.vtruncf(e1, tl.bfloat16))
                    pb, unused_b = tla.simd.vdintlv(tla.simd.vtruncf(e2, tl.bfloat16),
                                                   tla.simd.vtruncf(e3, tl.bfloat16))
                    tla.simd.vsstb(pa, prob[0, row * 16], PSTRIDE)
                    tla.simd.vsstb(pb, prob[BN // 32, row * 16], PSTRIDE)
                    tla.simd.vsts(mrun[row, 0], new_m, dist="ONEPT_B32")
                    tla.simd.vsts(lrun[row, 0], new_l, dist="ONEPT_B32")
        with tla.scope.vector().task("prob0", stage=1):
            tla.copy_ub_to_l1(tla.view(prob, (BN // 16, VM * 16)),
                              tla.view(tla.reshape(p, (BN // 16, BM * 16)), (BN // 16, VM * 16),
                                       offsets=(0, sub * VM * 16)))
        with tla.scope.cube().task("pv0", stage=1):
            tla.cube.mmad(p, v, pv, clear_accum=True)
            tla.copy_l0c_to_ub(
                pv, acc,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

        # ---- KV blocks 1..3: single K/V slot, online-softmax update ----
        with tla.scope.cube().task("load1", stage=1):
            tla.load_gm_to_l1(tla.view(K + BN * D, (BN, D)), k)
            tla.load_gm_to_l1(tla.view(V + BN * D, (BN, D)), v)
        with tla.scope.cube().task("qk1", stage=1):
            tla.cube.mmad(q, k, qk, transpose_b=True, clear_accum=True)
            tla.copy_l0c_to_ub(
                qk, score,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        with tla.scope.vector().task("softmax1", stage=1):
            with tla.scope.simd():
                for row in tl.range(VM):
                    old_m = tla.simd.vlds(mrun[row, 0], dist="BRC_B32")
                    old_l = tla.simd.vlds(lrun[row, 0], dist="BRC_B32")
                    x0 = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                    x1 = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                    x2 = tla.simd.vmuls(tla.simd.vlds(score[row, 128]), SCALE)
                    x3 = tla.simd.vmuls(tla.simd.vlds(score[row, 192]), SCALE)
                    new_m = tla.simd.vmax(
                        old_m, tla.simd.vreduce_max(
                            tla.simd.vmax(tla.simd.vmax(x0, x1), tla.simd.vmax(x2, x3))))
                    alpha_v = tla.simd.vexp(tla.simd.vsub(old_m, new_m))
                    e0 = tla.simd.vexp(tla.simd.vsub(x0, new_m))
                    e1 = tla.simd.vexp(tla.simd.vsub(x1, new_m))
                    e2 = tla.simd.vexp(tla.simd.vsub(x2, new_m))
                    e3 = tla.simd.vexp(tla.simd.vsub(x3, new_m))
                    local_l = tla.simd.vbroadcast(
                        tla.simd.vreduce_sum(
                            tla.simd.vadd(tla.simd.vadd(e0, e1), tla.simd.vadd(e2, e3))))
                    new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), local_l)
                    pa, unused_a = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                   tla.simd.vtruncf(e1, tl.bfloat16))
                    pb, unused_b = tla.simd.vdintlv(tla.simd.vtruncf(e2, tl.bfloat16),
                                                   tla.simd.vtruncf(e3, tl.bfloat16))
                    tla.simd.vsstb(pa, prob[0, row * 16], PSTRIDE)
                    tla.simd.vsstb(pb, prob[BN // 32, row * 16], PSTRIDE)
                    tla.simd.vsts(mrun[row, 0], new_m, dist="ONEPT_B32")
                    tla.simd.vsts(lrun[row, 0], new_l, dist="ONEPT_B32")
                    tla.simd.vsts(alpha[row, 0], alpha_v, dist="ONEPT_B32")
        with tla.scope.vector().task("prob1", stage=1):
            tla.copy_ub_to_l1(tla.view(prob, (BN // 16, VM * 16)),
                              tla.view(tla.reshape(p, (BN // 16, BM * 16)), (BN // 16, VM * 16),
                                       offsets=(0, sub * VM * 16)))
        with tla.scope.cube().task("pv1", stage=1):
            tla.cube.mmad(p, v, pv, clear_accum=True)
            tla.copy_l0c_to_ub(
                pv, pvub,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        with tla.scope.vector().task("rescale1", stage=1):
            with tla.scope.simd():
                for row in tl.range(VM):
                    a = tla.simd.vlds(alpha[row, 0], dist="BRC_B32")
                    tla.simd.vsts(acc[row, 0],
                                  tla.simd.vadd(tla.simd.vmul(tla.simd.vlds(acc[row, 0]), a),
                                                tla.simd.vlds(pvub[row, 0])))

        with tla.scope.cube().task("load2", stage=1):
            tla.load_gm_to_l1(tla.view(K + 2 * BN * D, (BN, D)), k)
            tla.load_gm_to_l1(tla.view(V + 2 * BN * D, (BN, D)), v)
        with tla.scope.cube().task("qk2", stage=1):
            tla.cube.mmad(q, k, qk, transpose_b=True, clear_accum=True)
            tla.copy_l0c_to_ub(
                qk, score,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        with tla.scope.vector().task("softmax2", stage=1):
            with tla.scope.simd():
                for row in tl.range(VM):
                    old_m = tla.simd.vlds(mrun[row, 0], dist="BRC_B32")
                    old_l = tla.simd.vlds(lrun[row, 0], dist="BRC_B32")
                    x0 = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                    x1 = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                    x2 = tla.simd.vmuls(tla.simd.vlds(score[row, 128]), SCALE)
                    x3 = tla.simd.vmuls(tla.simd.vlds(score[row, 192]), SCALE)
                    new_m = tla.simd.vmax(
                        old_m, tla.simd.vreduce_max(
                            tla.simd.vmax(tla.simd.vmax(x0, x1), tla.simd.vmax(x2, x3))))
                    alpha_v = tla.simd.vexp(tla.simd.vsub(old_m, new_m))
                    e0 = tla.simd.vexp(tla.simd.vsub(x0, new_m))
                    e1 = tla.simd.vexp(tla.simd.vsub(x1, new_m))
                    e2 = tla.simd.vexp(tla.simd.vsub(x2, new_m))
                    e3 = tla.simd.vexp(tla.simd.vsub(x3, new_m))
                    local_l = tla.simd.vbroadcast(
                        tla.simd.vreduce_sum(
                            tla.simd.vadd(tla.simd.vadd(e0, e1), tla.simd.vadd(e2, e3))))
                    new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), local_l)
                    pa, unused_a = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                   tla.simd.vtruncf(e1, tl.bfloat16))
                    pb, unused_b = tla.simd.vdintlv(tla.simd.vtruncf(e2, tl.bfloat16),
                                                   tla.simd.vtruncf(e3, tl.bfloat16))
                    tla.simd.vsstb(pa, prob[0, row * 16], PSTRIDE)
                    tla.simd.vsstb(pb, prob[BN // 32, row * 16], PSTRIDE)
                    tla.simd.vsts(mrun[row, 0], new_m, dist="ONEPT_B32")
                    tla.simd.vsts(lrun[row, 0], new_l, dist="ONEPT_B32")
                    tla.simd.vsts(alpha[row, 0], alpha_v, dist="ONEPT_B32")
        with tla.scope.vector().task("prob2", stage=1):
            tla.copy_ub_to_l1(tla.view(prob, (BN // 16, VM * 16)),
                              tla.view(tla.reshape(p, (BN // 16, BM * 16)), (BN // 16, VM * 16),
                                       offsets=(0, sub * VM * 16)))
        with tla.scope.cube().task("pv2", stage=1):
            tla.cube.mmad(p, v, pv, clear_accum=True)
            tla.copy_l0c_to_ub(
                pv, pvub,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        with tla.scope.vector().task("rescale2", stage=1):
            with tla.scope.simd():
                for row in tl.range(VM):
                    a = tla.simd.vlds(alpha[row, 0], dist="BRC_B32")
                    tla.simd.vsts(acc[row, 0],
                                  tla.simd.vadd(tla.simd.vmul(tla.simd.vlds(acc[row, 0]), a),
                                                tla.simd.vlds(pvub[row, 0])))

        with tla.scope.cube().task("load3", stage=1):
            tla.load_gm_to_l1(tla.view(K + 3 * BN * D, (BN, D)), k)
            tla.load_gm_to_l1(tla.view(V + 3 * BN * D, (BN, D)), v)
        with tla.scope.cube().task("qk3", stage=1):
            tla.cube.mmad(q, k, qk, transpose_b=True, clear_accum=True)
            tla.copy_l0c_to_ub(
                qk, score,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        with tla.scope.vector().task("softmax3", stage=1):
            with tla.scope.simd():
                for row in tl.range(VM):
                    old_m = tla.simd.vlds(mrun[row, 0], dist="BRC_B32")
                    old_l = tla.simd.vlds(lrun[row, 0], dist="BRC_B32")
                    x0 = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                    x1 = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                    x2 = tla.simd.vmuls(tla.simd.vlds(score[row, 128]), SCALE)
                    x3 = tla.simd.vmuls(tla.simd.vlds(score[row, 192]), SCALE)
                    new_m = tla.simd.vmax(
                        old_m, tla.simd.vreduce_max(
                            tla.simd.vmax(tla.simd.vmax(x0, x1), tla.simd.vmax(x2, x3))))
                    alpha_v = tla.simd.vexp(tla.simd.vsub(old_m, new_m))
                    e0 = tla.simd.vexp(tla.simd.vsub(x0, new_m))
                    e1 = tla.simd.vexp(tla.simd.vsub(x1, new_m))
                    e2 = tla.simd.vexp(tla.simd.vsub(x2, new_m))
                    e3 = tla.simd.vexp(tla.simd.vsub(x3, new_m))
                    local_l = tla.simd.vbroadcast(
                        tla.simd.vreduce_sum(
                            tla.simd.vadd(tla.simd.vadd(e0, e1), tla.simd.vadd(e2, e3))))
                    new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), local_l)
                    pa, unused_a = tla.simd.vdintlv(tla.simd.vtruncf(e0, tl.bfloat16),
                                                   tla.simd.vtruncf(e1, tl.bfloat16))
                    pb, unused_b = tla.simd.vdintlv(tla.simd.vtruncf(e2, tl.bfloat16),
                                                   tla.simd.vtruncf(e3, tl.bfloat16))
                    tla.simd.vsstb(pa, prob[0, row * 16], PSTRIDE)
                    tla.simd.vsstb(pb, prob[BN // 32, row * 16], PSTRIDE)
                    tla.simd.vsts(mrun[row, 0], new_m, dist="ONEPT_B32")
                    tla.simd.vsts(lrun[row, 0], new_l, dist="ONEPT_B32")
                    tla.simd.vsts(alpha[row, 0], alpha_v, dist="ONEPT_B32")
        with tla.scope.vector().task("prob3", stage=1):
            tla.copy_ub_to_l1(tla.view(prob, (BN // 16, VM * 16)),
                              tla.view(tla.reshape(p, (BN // 16, BM * 16)), (BN // 16, VM * 16),
                                       offsets=(0, sub * VM * 16)))
        with tla.scope.cube().task("pv3", stage=1):
            tla.cube.mmad(p, v, pv, clear_accum=True)
            tla.copy_l0c_to_ub(
                pv, pvub,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        with tla.scope.vector().task("rescale3", stage=1):
            with tla.scope.simd():
                for row in tl.range(VM):
                    a = tla.simd.vlds(alpha[row, 0], dist="BRC_B32")
                    tla.simd.vsts(acc[row, 0],
                                  tla.simd.vadd(tla.simd.vmul(tla.simd.vlds(acc[row, 0]), a),
                                                tla.simd.vlds(pvub[row, 0])))

        # pvub is dead after the last rescale, so it doubles as the output stage.
        with tla.scope.vector().task("output", stage=1):
            with tla.scope.simd():
                for row in tl.range(VM):
                    den = tla.simd.vlds(lrun[row, 0], dist="BRC_B32")
                    tla.simd.vsts(pvub[row, 0], tla.simd.vdiv(tla.simd.vlds(acc[row, 0]), den))
            tla.store_ub_to_gm(tla.view(pvub, (VM, D)),
                               tla.view(O + t * BM * D + sub * VM * D, (VM, D)))


SIGNATURE = dict(Q="*bf16", K="*bf16", V="*bf16", O="*fp32")
CONSTS = dict(TILES=TILES, BM=BM, BN=BN, NKB=NKB, D=D, VM=VM, SCALE=SCALE, PSTRIDE=PSTRIDE)
EXPECTED_STAGES = dict(load0=0, qk0=0,
                       softmax0=1, prob0=1, pv0=1,
                       load1=1, qk1=1, softmax1=1, prob1=1, pv1=1, rescale1=1,
                       load2=1, qk2=1, softmax2=1, prob2=1, pv2=1, rescale2=1,
                       load3=1, qk3=1, softmax3=1, prob3=1, pv3=1, rescale3=1,
                       output=1)


def task_contract(text):
    """Ignore names/locations while retaining ordered scope attributes and stages."""
    return [(name, int(stage) if stage else None)
            for stage, name in re.findall(r'(?:hivm.pipeline_stage = (\d+) : i32, )?hivm.task_name = "([^"]+)"', text)]


def test_ir(output_dir=None):
    module, ctx = compile_offline(fa_shape_stages, SIGNATURE, CONSTS, output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert task_contract(text) == list(EXPECTED_STAGES.items())
    assert re.search(r"\bhivm\.pipeline(?:[,}\s])", text)
    assert text.count("hivm.hir.mmadL1") == 8, "QK x4 + PV x4"
    assert text.count("hivm.hir.fixpipe") == 8, "S x2 + PV x4 + first PV"
    assert text.count("hivm.hir.nd2nz") == 9, "q + k/v x4"
    assert text.count("hivm.hir.copy ") == 4, "P UB->L1 x4"
    assert text.count("hivm.hir.store ") == 1, "O UB->GM x1"
    assert "hivm.vector_function" in text
    assert "sync_block" not in text and "tt." not in text
    print("IR PASS: fa_shape_stage HIVM structure ok"
          " (8 mmadL1 / 8 fixpipe / 9 nd2nz / 4 UB->L1 / 1 store / no manual sync)")


def _inputs():
    import torch
    q = torch.randn(ROWS * D, dtype=torch.float32, device="npu").to(torch.bfloat16)
    k = torch.randn(K_ROWS * D, dtype=torch.float32, device="npu").to(torch.bfloat16)
    v = torch.randn(K_ROWS * D, dtype=torch.float32, device="npu").to(torch.bfloat16)
    o = torch.zeros(ROWS * D, dtype=torch.float32, device="npu")
    return q, k, v, o


def _reference(q, k, v):
    """Online-softmax flash attention in FP32 with block-wise BF16 P rounding."""
    import torch
    qt = q.to(torch.float32).reshape(TILES, BM, D)
    kt = k.to(torch.float32).reshape(K_ROWS, D)
    vt = v.to(torch.float32).reshape(K_ROWS, D)
    outs = []
    for i in range(TILES):
        qi = qt[i]
        m = torch.full((BM, 1), float("-inf"))
        l = torch.zeros((BM, 1))
        acc = torch.zeros((BM, D))
        for b in range(NKB):
            s = (qi @ kt[b * BN:(b + 1) * BN].transpose(0, 1)) * SCALE
            new_m = torch.maximum(m, s.max(dim=1, keepdim=True).values)
            alpha = torch.exp(m - new_m)
            e = torch.exp(s - new_m)
            l = l * alpha + e.sum(dim=1, keepdim=True)
            p = e.to(torch.bfloat16).to(torch.float32)
            acc = acc * alpha + p @ vt[b * BN:(b + 1) * BN]
            m = new_m
        outs.append(acc / l)
    return torch.cat(outs, dim=0).reshape(-1)


def _launch():
    q, k, v, o = _inputs()
    fa_shape_stages[(1, )](q, k, v, o, **CONSTS)
    return o, _reference(q.cpu(), k.cpu(), v.cpu())


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: fa_shape_stage artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    import torch
    out, ref = _launch()
    torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)
    print("FA_SHAPE OK: fa_best workload (S=%d, D=%d, dense) through the stage pipeline" % (K_ROWS, D),
          "TILES", TILES, "BM", BM, "BN", BN, "NKB", NKB)


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: fa_shape_stage", "ir" if do_ir else "", "compile" if do_compile else "",
          "run" if do_run else "")


if __name__ == "__main__":
    main()
