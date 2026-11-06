"""Stage-scheduled FlashAttention forward, following the issue104 fa_best spec.

Algorithm taken from ``issue104_fa_optimized_20261031/fa_best_kernel.py``:

  * BF16 Q/K/V, P (the softmax numerator) rounded to BF16 before the PV matmul;
  * FP32 QK / PV accumulation and FP32 online-softmax state (running m and l);
  * KV traversal in BN-key blocks with alpha rescaling of the running state;
  * ``O = (sum_b alpha_b * P_b V_b) / l`` per query tile.

fa_best hand-rolls the software pipeline with two S/P slots, one K/V slot and
explicit ``tla.sync.block_set_flag`` / ``block_wait_flag`` pairs.  This case
keeps the same algorithm but writes it with the ``tla.pipeline`` stage contract
(``docs/zh/design/tla_scope_schedule_v2.md``) and inserts **no** sync: every
AIC/AIV handoff is left to the backend.

The stage contract rejects loop-carried SSA values and buffers allocated outside
the pipeline loop, and doc section 7.6 states that the online-softmax m/l/acc
recurrence cannot cross pipeline iterations yet.  The KV blocks are therefore
statically unrolled inside one logical iteration and the pipeline domain
iterates over query tiles only.

Deliberate reductions vs fa_best: no causal mask, no query rotation, one K/V and
one PV allocation per block (no double slot), FP32 O, and no eight-copy FP32 row
max/sum outputs.

The pipeline depth is `max(stage) + 1` and LowerTLAStages stamps every pipeline
buffer with `hivm.multi_buffer = depth`, so the local-buffer budget is
`sum(buffer bytes) * depth`.  With four 16 KB K/V tiles in L1 the depth has to
stay at 6, which is why the two KV blocks share stage 0..5 instead of getting one
stage per block.

Shapes: BM=16 query rows per tile (VM=8 rows per AIV sub-block), BN=128 keys per
KV block, NKB=2 blocks (256 keys), D=64 head dim, TILES=3 query tiles.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM/AVE IR structure (4 mmadL1 / 4 fixpipe / 5 nd2nz /
            2 UB->L1 copies / 1 store, no manual sync)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : device launch + accuracy (FA_STAGE OK vs torch online-softmax ref)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, lower_offline,
                                  check_compile_artifacts, get_stages)

# BM query rows per tile, BN keys per KV block, NKB KV blocks, D head dim.
BM = 16
BN = 128
NKB = 2
D = 64
TILES = 3
ROWS = TILES * BM
K_ROWS = NKB * BN
SCALE = 1.0 / D**0.5
VM = BM // 2  # rows owned by one AIV sub-block


@tla.jit
def fa_stages(Q, K, V, O, TILES: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
              VM: tl.constexpr, SCALE: tl.constexpr):
    for t in tla.pipeline(TILES):
        q = tla.alloc((4, 1, 16, 16), tl.bfloat16, space="l1")
        k0 = tla.alloc((4, 8, 16, 16), tl.bfloat16, space="l1")
        v0 = tla.alloc((4, 8, 16, 16), tl.bfloat16, space="l1")
        k1 = tla.alloc((4, 8, 16, 16), tl.bfloat16, space="l1")
        v1 = tla.alloc((4, 8, 16, 16), tl.bfloat16, space="l1")
        p0 = tla.alloc((8, 1, 16, 16), tl.bfloat16, space="l1")
        p1 = tla.alloc((8, 1, 16, 16), tl.bfloat16, space="l1")
        qk0 = tla.alloc((8, 1, 16, 16), tl.float32, space="l0c")
        qk1 = tla.alloc((8, 1, 16, 16), tl.float32, space="l0c")
        pv0 = tla.alloc((4, 1, 16, 16), tl.float32, space="l0c")
        pv1 = tla.alloc((4, 1, 16, 16), tl.float32, space="l0c")
        score0 = tla.alloc((8, BN), tl.float32, space="ub")
        score1 = tla.alloc((8, BN), tl.float32, space="ub")
        prob0 = tla.alloc((8, 9 * 16), tl.bfloat16, space="ub")
        prob1 = tla.alloc((8, 9 * 16), tl.bfloat16, space="ub")
        # Running online-softmax state, one scalar per row of this sub-block.
        mrun = tla.alloc((VM, 1), tl.float32, space="ub")
        lrun = tla.alloc((VM, 1), tl.float32, space="ub")
        alpha = tla.alloc((VM, 1), tl.float32, space="ub")
        # First block writes acc directly; later blocks rescale it in place.
        acc = tla.alloc((VM, D), tl.float32, space="ub")
        pvub = tla.alloc((VM, D), tl.float32, space="ub")
        out = tla.alloc((VM, D), tl.float32, space="ub")
        sub = tla.sub_block_id()

        with tla.scope.cube().task("load0", stage=0):
            tla.load_gm_to_l1(tla.view(Q + t * BM * D, (BM, D)), q)
            tla.load_gm_to_l1(tla.view(K, (BN, D)), k0)
            tla.load_gm_to_l1(tla.view(V, (BN, D)), v0)
        with tla.scope.cube().task("qk0", stage=0):
            tla.cube.mmad(q, k0, qk0, transpose_b=True, clear_accum=True)
            tla.copy_l0c_to_ub(
                qk0, score0,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        # First KV block initializes the running state: m = row max of S, no rescale.
        with tla.scope.vector().task("softmax0", stage=1):
            with tla.scope.simd():
                for row in tl.range(VM):
                    x0 = tla.simd.vmuls(tla.simd.vlds(score0[row, 0]), SCALE)
                    x1 = tla.simd.vmuls(tla.simd.vlds(score0[row, 64]), SCALE)
                    new_m = tla.simd.vreduce_max(tla.simd.vmax(x0, x1))
                    e0 = tla.simd.vexp(tla.simd.vsub(x0, new_m))
                    e1 = tla.simd.vexp(tla.simd.vsub(x1, new_m))
                    new_l = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(e0, e1)))
                    p0v = tla.simd.vtruncf(e0, tl.bfloat16)
                    p1v = tla.simd.vtruncf(e1, tl.bfloat16)
                    packed, unused = tla.simd.vdintlv(p0v, p1v)
                    tla.simd.vsstb(packed, prob0[0, row * 16], 9 * 16)
                    tla.simd.vsts(mrun[row, 0], new_m, dist="ONEPT_B32")
                    tla.simd.vsts(lrun[row, 0], new_l, dist="ONEPT_B32")
        with tla.scope.vector().task("prob0", stage=1):
            tla.copy_ub_to_l1(tla.view(prob0, (8, 8 * 16)),
                              tla.view(tla.reshape(p0, (8, 16 * 16)), (8, 8 * 16), offsets=(0, sub * 8 * 16)))
        with tla.scope.cube().task("pv0", stage=2):
            tla.cube.mmad(p0, v0, pv0, clear_accum=True)
            tla.copy_l0c_to_ub(
                pv0, acc,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))

        with tla.scope.cube().task("load1", stage=2):
            tla.load_gm_to_l1(tla.view(K + BN * D, (BN, D)), k1)
            tla.load_gm_to_l1(tla.view(V + BN * D, (BN, D)), v1)
        with tla.scope.cube().task("qk1", stage=2):
            tla.cube.mmad(q, k1, qk1, transpose_b=True, clear_accum=True)
            tla.copy_l0c_to_ub(
                qk1, score1,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        # Later blocks merge into the running state and keep exp(m_old - m_new).
        with tla.scope.vector().task("softmax1", stage=3):
            with tla.scope.simd():
                for row in tl.range(VM):
                    old_m = tla.simd.vlds(mrun[row, 0], dist="BRC_B32")
                    old_l = tla.simd.vlds(lrun[row, 0], dist="BRC_B32")
                    x0 = tla.simd.vmuls(tla.simd.vlds(score1[row, 0]), SCALE)
                    x1 = tla.simd.vmuls(tla.simd.vlds(score1[row, 64]), SCALE)
                    new_m = tla.simd.vmax(old_m, tla.simd.vreduce_max(tla.simd.vmax(x0, x1)))
                    alpha_v = tla.simd.vexp(tla.simd.vsub(old_m, new_m))
                    e0 = tla.simd.vexp(tla.simd.vsub(x0, new_m))
                    e1 = tla.simd.vexp(tla.simd.vsub(x1, new_m))
                    local_l = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(e0, e1)))
                    new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), local_l)
                    p0v = tla.simd.vtruncf(e0, tl.bfloat16)
                    p1v = tla.simd.vtruncf(e1, tl.bfloat16)
                    packed, unused = tla.simd.vdintlv(p0v, p1v)
                    tla.simd.vsstb(packed, prob1[0, row * 16], 9 * 16)
                    tla.simd.vsts(mrun[row, 0], new_m, dist="ONEPT_B32")
                    tla.simd.vsts(lrun[row, 0], new_l, dist="ONEPT_B32")
                    tla.simd.vsts(alpha[row, 0], alpha_v, dist="ONEPT_B32")
        with tla.scope.vector().task("prob1", stage=3):
            tla.copy_ub_to_l1(tla.view(prob1, (8, 8 * 16)),
                              tla.view(tla.reshape(p1, (8, 16 * 16)), (8, 8 * 16), offsets=(0, sub * 8 * 16)))
        with tla.scope.cube().task("pv1", stage=4):
            tla.cube.mmad(p1, v1, pv1, clear_accum=True)
            tla.copy_l0c_to_ub(
                pv1, pvub,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        with tla.scope.vector().task("rescale", stage=5):
            with tla.scope.simd():
                for row in tl.range(VM):
                    a = tla.simd.vlds(alpha[row, 0], dist="BRC_B32")
                    acc_v = tla.simd.vadd(tla.simd.vmul(tla.simd.vlds(acc[row, 0]), a),
                                          tla.simd.vlds(pvub[row, 0]))
                    tla.simd.vsts(acc[row, 0], acc_v)
        with tla.scope.vector().task("output", stage=5):
            with tla.scope.simd():
                for row in tl.range(VM):
                    den = tla.simd.vlds(lrun[row, 0], dist="BRC_B32")
                    tla.simd.vsts(out[row, 0], tla.simd.vdiv(tla.simd.vlds(acc[row, 0]), den))
            tla.store_ub_to_gm(out, tla.view(O + t * BM * D + sub * VM * D, (VM, D)))


SIGNATURE = dict(Q="*bf16", K="*bf16", V="*bf16", O="*fp32")
CONSTS = dict(TILES=TILES, BM=BM, BN=BN, D=D, VM=VM, SCALE=SCALE)
EXPECTED_STAGES = dict(load0=0, qk0=0, softmax0=1, prob0=1, pv0=2, load1=2, qk1=2, softmax1=3, prob1=3, pv1=4,
                       rescale=5, output=5)


def task_contract(text):
    """Ignore names/locations while retaining ordered scope attributes and stages."""
    return [(name, int(stage) if stage else None)
            for stage, name in re.findall(r'(?:hivm.pipeline_stage = (\d+) : i32, )?hivm.task_name = "([^"]+)"', text)]


def test_ir(output_dir=None):
    module, ctx = compile_offline(fa_stages, SIGNATURE, CONSTS, output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert task_contract(text) == list(EXPECTED_STAGES.items())
    assert re.search(r"\bhivm\.pipeline(?:[,}\s])", text)
    assert text.count("hivm.hir.mmadL1") == 4, "QK x2 + PV x2"
    assert text.count("hivm.hir.fixpipe") == 4, "S x2 + PV x2"
    assert text.count("hivm.hir.nd2nz") == 5, "q, k0, v0, k1, v1"
    assert text.count("hivm.hir.copy ") == 2, "P UB->L1 x2"
    assert text.count("hivm.hir.store ") == 1, "O UB->GM x1"
    assert "hivm.vector_function" in text
    assert "sync_block" not in text and "tt." not in text
    print("IR PASS: fa_stage HIVM structure ok"
          " (4 mmadL1 / 4 fixpipe / 5 nd2nz / 2 UB->L1 / 1 store / no manual sync)")


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
            kb = kt[b * BN:(b + 1) * BN]
            s = (qi @ kb.transpose(0, 1)) * SCALE
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
    fa_stages[(1, )](q, k, v, o, **CONSTS)
    return o, _reference(q.cpu(), k.cpu(), v.cpu())


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: fa_stage artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    import torch
    out, ref = _launch()
    torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)
    print("FA_STAGE OK: online-softmax flash attention through the task/stage pipeline",
          "TILES", TILES, "BM", BM, "BN", BN, "NKB", NKB, "D", D)


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: fa_stage", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
