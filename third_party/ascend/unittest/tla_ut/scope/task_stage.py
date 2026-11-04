"""Task/stage frontend contracts and a single-KV-tile attention IR fixture.

IR-only: no schedule solver or automatic synchronization has run yet.
The kernel computes softmax(Q K^T / 8) V for Q[16,64], K[128,64], V[128,64].
P is rounded to BF16 before PV; output is FP32. Each loop tile has its own Q/O.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from triton.compiler.errors import CompilationError
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, lower_offline


@tla.jit
def attention_stages(Q, K, V, O, TILES: tl.constexpr):
    with tla.scope.schedule("auto"):
        for t in tl.range(TILES):
            q = tla.alloc((4, 1, 16, 16), tl.bfloat16, space="l1")
            k = tla.alloc((4, 8, 16, 16), tl.bfloat16, space="l1")
            v = tla.alloc((4, 8, 16, 16), tl.bfloat16, space="l1")
            p = tla.alloc((8, 1, 16, 16), tl.bfloat16, space="l1")
            qk = tla.alloc((8, 1, 16, 16), tl.float32, space="l0c")
            pv = tla.alloc((4, 1, 16, 16), tl.float32, space="l0c")
            score = tla.alloc((8, 128), tl.float32, space="ub")
            prob = tla.alloc((8, 9 * 16), tl.bfloat16, space="ub")
            out = tla.alloc((8, 64), tl.float32, space="ub")
            sub = tla.sub_block_id()

            with tla.scope.cube().task("load_qk", stage=0):
                tla.load_gm_to_l1(tla.view(Q + t * 16 * 64, (16, 64)), q)
                tla.load_gm_to_l1(tla.view(K, (128, 64)), k)
            with tla.scope.cube().task("load_v"):
                tla.load_gm_to_l1(tla.view(V, (128, 64)), v)
            with tla.scope.cube().task("qk", stage=1):
                tla.cube.mmad(q, k, qk, transpose_b=True, clear_accum=True)
            with tla.scope.cube().task("score", stage=1):
                tla.copy_l0c_to_ub(
                    qk, score,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
            with tla.scope.simd().task("softmax", stage=2):
                for row in tl.range(8):
                    x0 = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), 0.125)
                    x1 = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), 0.125)
                    maximum = tla.simd.vreduce_max(tla.simd.vmax(x0, x1))
                    e0 = tla.simd.vexp(tla.simd.vsub(x0, maximum))
                    e1 = tla.simd.vexp(tla.simd.vsub(x1, maximum))
                    total = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(e0, e1)))
                    p0 = tla.simd.vtruncf(tla.simd.vdiv(e0, total), tl.bfloat16)
                    p1 = tla.simd.vtruncf(tla.simd.vdiv(e1, total), tl.bfloat16)
                    packed, unused = tla.simd.vdintlv(p0, p1)
                    tla.simd.vsstb(packed, prob[0, row * 16], 9 * 16)
            with tla.scope.vector().task("prob", stage=2):
                tla.copy_ub_to_l1(tla.view(prob, (8, 8 * 16)),
                                  tla.view(tla.reshape(p, (8, 16 * 16)), (8, 8 * 16), offsets=(0, sub * 8 * 16)))
            with tla.scope.cube().task("pv", stage=3):
                tla.cube.mmad(p, v, pv, clear_accum=True)
            with tla.scope.cube().task("output", stage=3):
                tla.copy_l0c_to_ub(
                    pv, out,
                    tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
            with tla.scope.vector().task("store", stage=4):
                tla.store_ub_to_gm(out, tla.view(O + t * 16 * 64 + sub * 8 * 64, (8, 64)))


SIGNATURE = dict(Q="*bf16", K="*bf16", V="*bf16", O="*fp32")
EXPECTED_STAGES = dict(load_qk=0, load_v=None, qk=1, score=1, softmax=2, prob=2, pv=3, output=3, store=4)


def task_contract(text):
    """Ignore names/locations while retaining ordered scope attributes and stages."""
    return [(name, int(stage) if stage else None)
            for stage, name in re.findall(r'(?:hivm.required_stage = (\d+) : i32, )?hivm.task_name = "([^"]+)"', text)]


def test_ir(output_dir=None):
    module, ctx = compile_offline(attention_stages, SIGNATURE, dict(TILES=3), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert task_contract(text) == list(EXPECTED_STAGES.items())
    assert 'hivm.schedule_domain = "auto"' in text
    assert 'hivm.schedule_state = "logical"' in text
    assert text.count("hivm.hir.mmadL1") == 2
    assert text.count("hivm.hir.fixpipe") == 2
    assert text.count("hivm.hir.nd2nz") == 3
    assert text.count("hivm.hir.copy ") == 1
    assert text.count("hivm.hir.store ") == 1
    assert "hivm.vector_function" in text
    assert "hivm.preload_num" not in text and "hivm.multi_buffer" not in text
    assert "sync_block" not in text and "tt." not in text
    expected = Path(__file__).with_name("attention_stages.ttadapter.mlir")
    assert task_contract(expected.read_text()) == task_contract(text)
    from triton._C.libtriton import ir
    assert ir.parse_mlir_module(str(expected), ctx).verify()


@tla.jit
def task_copy(X, Y, NAME: tl.constexpr, STAGE: tl.constexpr):
    with tla.scope.schedule():
        a = tla.alloc((64, ), tl.float32, space="ub")
        with tla.scope.vector().task(NAME, stage=STAGE):
            tla.load_gm_to_ub(tla.view(X, (64, )), a)
            tla.store_ub_to_gm(a, tla.view(Y, (64, )))


@pytest.mark.parametrize("name,stage,error", [
    ("", 0, "nonempty"),
    ("copy", -1, "nonnegative"),
    ("copy", True, "nonnegative"),
    ("copy", 1.5, "nonnegative"),
    ("copy", 2**31, "nonnegative"),
])
def test_invalid_task(name, stage, error):
    with pytest.raises(CompilationError, match=error):
        compile_offline(task_copy, dict(X="*fp32", Y="*fp32"), dict(NAME=name, STAGE=stage))


@tla.jit
def outside_domain(X):
    with tla.scope.vector().task("load", stage=0):
        a = tla.alloc((64, ), tl.float32, space="ub")
        tla.load_gm_to_ub(tla.view(X, (64, )), a)


def test_task_requires_domain():
    with pytest.raises(CompilationError, match="requires an enclosing"):
        compile_offline(outside_domain, dict(X="*fp32"), {})


@tla.jit
def duplicate_task(X):
    with tla.scope.schedule():
        a = tla.alloc((64, ), tl.float32, space="ub")
        with tla.scope.vector().task("load"):
            tla.load_gm_to_ub(tla.view(X, (64, )), a)
        with tla.scope.vector().task("load"):
            tla.load_gm_to_ub(tla.view(X + 64, (64, )), a)


def test_duplicate_task():
    with pytest.raises(CompilationError, match="duplicate task name"):
        compile_offline(duplicate_task, dict(X="*fp32"), {})


def test_device_compile_rejects_unresolved_schedule():
    from triton.backends.ascend.compiler import (linalg_to_bin_enable_npu_compile_910_95,
                                                 linalg_to_bin_enable_npu_compile_A2_A3, ttir_to_npubin)
    for compile_binary in (linalg_to_bin_enable_npu_compile_910_95, linalg_to_bin_enable_npu_compile_A2_A3, ttir_to_npubin):
        with pytest.raises(RuntimeError, match="ttadapter only"):
            compile_binary('hivm.schedule_domain = "auto"', {}, None)
