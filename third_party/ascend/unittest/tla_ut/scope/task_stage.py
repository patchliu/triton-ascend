"""Task/stage frontend contracts and a single-KV-tile attention IR fixture.

Frontend IR-only: no pipeline materialization or synchronization has run.
The SIMD row loop is outside the current NPU-IR straight-line task subset.
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
    for t in tla.pipeline(TILES):
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
        with tla.scope.cube().task("load_v", stage=0):
            tla.load_gm_to_l1(tla.view(V, (128, 64)), v)
        with tla.scope.cube().task("qk", stage=1):
            tla.cube.mmad(q, k, qk, transpose_b=True, clear_accum=True)
        with tla.scope.cube().task("score", stage=1):
            tla.copy_l0c_to_ub(
                qk, score,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
        with tla.scope.vector().task("softmax", stage=2, order=1):
            with tla.scope.simd():
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
EXPECTED_STAGES = dict(load_qk=0, load_v=0, qk=1, score=1, softmax=2, prob=2, pv=3, output=3, store=4)


def task_contract(text):
    """Ignore names/locations while retaining ordered scope attributes and stages."""
    return [(name, int(stage) if stage else None)
            for stage, name in re.findall(r'(?:hivm.pipeline_stage = (\d+) : i32, )?hivm.task_name = "([^"]+)"', text)]


def test_ir(output_dir=None):
    module, ctx = compile_offline(attention_stages, SIGNATURE, dict(TILES=3), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert task_contract(text) == list(EXPECTED_STAGES.items())
    assert re.search(r"\bhivm\.pipeline(?:[,}\s])", text)
    assert "hivm.schedule_domain" not in text and "hivm.schedule_state" not in text
    assert "hivm.required_stage" not in text and "hivm.preload_fixed_depth" not in text
    assert 'hivm.task_name = "softmax", hivm.task_order = 1 : i32' in text
    assert text.count("hivm.hir.mmadL1") == 2
    assert text.count("hivm.hir.fixpipe") == 2
    assert text.count("hivm.hir.nd2nz") == 3
    assert text.count("hivm.hir.copy ") == 1
    assert text.count("hivm.hir.store ") == 1
    assert "hivm.vector_function" in text
    assert "hivm.preload_num" not in text and "hivm.multi_buffer" not in text
    assert "sync_block" not in text and "tt." not in text
    # Task boundaries are core scopes; VF attributes belong only to the
    # nested SIMD scope and must not be combined with task/stage attributes.
    task_attrs = re.findall(r'\{([^{}\n]*hivm\.task_name[^{}\n]*)\}', text)
    assert len(task_attrs) == len(EXPECTED_STAGES)
    assert all("hivm.vector_function" not in attrs and "outline" not in attrs for attrs in task_attrs)
    expected = Path(__file__).with_name("attention_stages.ttadapter.mlir")
    assert task_contract(expected.read_text()) == task_contract(text)
    from triton._C.libtriton import ir
    assert ir.parse_mlir_module(str(expected), ctx).verify()


@tla.jit
def task_copy(X, Y, NAME: tl.constexpr, STAGE: tl.constexpr):
    for t in tla.pipeline(2):
        a = tla.alloc((64, ), tl.float32, space="ub")
        with tla.scope.vector().task(NAME, stage=STAGE):
            tla.load_gm_to_ub(tla.view(X, (64, )), a)
            tla.store_ub_to_gm(a, tla.view(Y, (64, )))


@pytest.mark.parametrize("name,stage,error", [
    ("", 0, "nonempty"),
    ("copy", -1, "compile-time integer"),
    ("copy", True, "compile-time integer"),
    ("copy", 1.5, "compile-time integer"),
    ("copy", 64, "compile-time integer"),
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
    for t in tla.pipeline(2):
        a = tla.alloc((64, ), tl.float32, space="ub")
        with tla.scope.vector().task("load", stage=0):
            tla.load_gm_to_ub(tla.view(X, (64, )), a)
        with tla.scope.vector().task("load", stage=0):
            tla.load_gm_to_ub(tla.view(X + 64, (64, )), a)


def test_duplicate_task():
    with pytest.raises(CompilationError, match="duplicate task name"):
        compile_offline(duplicate_task, dict(X="*fp32"), {})


@pytest.mark.parametrize("attr", ["hivm.pipeline", "hivm.pipeline_stage = 0 : i32", 'hivm.task_name = "load"'])
def test_pipeline_compile_gate(attr):
    from types import SimpleNamespace
    from triton.backends.ascend.compiler import _check_tla_pipeline_support
    tla_options = SimpleNamespace(enable_tla_mode=True)
    _check_tla_pipeline_support(attr, tla_options, allow_pipeline=True)
    with pytest.raises(RuntimeError, match="RegBase ttadapter"):
        _check_tla_pipeline_support(attr, tla_options)
    with pytest.raises(RuntimeError, match="RegBase ttadapter"):
        _check_tla_pipeline_support(attr, SimpleNamespace(enable_tla_mode=False), allow_pipeline=True)


@pytest.mark.parametrize("attr", ["hivm.schedule_domain", "hivm.schedule_state", "hivm.required_stage"])
def test_old_schedule_attributes_rejected(attr):
    from types import SimpleNamespace
    from triton.backends.ascend.compiler import _check_tla_pipeline_support
    with pytest.raises(RuntimeError, match="obsolete TLA"):
        _check_tla_pipeline_support(attr, SimpleNamespace(enable_tla_mode=True), allow_pipeline=True)


@pytest.mark.parametrize("kind", [tla.scope.vector, tla.scope.cube])
def test_only_core_scopes_own_tasks(kind):
    spec = kind().task("compute", stage=1)
    assert spec.kind is kind
    assert spec.stage == 1


def test_simd_does_not_own_tasks():
    assert not hasattr(tla.scope.simd(), "task")
    assert not hasattr(tla.scope(), "task")
    from triton.language.extra.tla.scope import _TaskScope
    with pytest.raises(ValueError, match="only supported on"):
        _TaskScope(tla.scope.simd, "compute", 1)


@tla.jit
def ordered_task_copy(X, Y, ORDER: tl.constexpr):
    for t in tla.pipeline(2):
        tile = tla.alloc((64,), tl.float32, space="ub")
        with tla.scope.vector().task("copy", stage=0, order=ORDER):
            tla.load_gm_to_ub(tla.view(X + t * 64, (64,)), tile)
            tla.store_ub_to_gm(tile, tla.view(Y + t * 64, (64,)))


@pytest.mark.parametrize("order", [None, 0, 7, 2**31 - 1])
def test_task_order_ir(order):
    module, ctx = compile_offline(ordered_task_copy, dict(X="*fp32", Y="*fp32"), dict(ORDER=order))
    text = str(lower_offline(module, ctx))
    if order is None:
        assert "hivm.task_order" not in text
    else:
        assert f"hivm.task_order = {order} : i32" in text
    assert task_contract(text) == [("copy", 0)]


@pytest.mark.parametrize("order", [-1, True, 1.5, "1", 2**31])
def test_invalid_task_order(order):
    with pytest.raises(CompilationError, match="task order must be"):
        compile_offline(ordered_task_copy, dict(X="*fp32", Y="*fp32"), dict(ORDER=order))


@tla.jit
def hinted_pipeline(X, STAGE: tl.constexpr, ORDER: tl.constexpr):
    for t in tla.pipeline(2):
        tile = tla.alloc((64,), tl.float32, space="ub")
        with tla.scope.vector().task("load", stage=STAGE, order=ORDER):
            tla.load_gm_to_ub(tla.view(X + t * 64, (64,)), tile)


@pytest.mark.parametrize("stage,order", [(None, None), (None, 2), (1, None), (1, 2)])
def test_optional_pipeline_hints(stage, order):
    module, ctx = compile_offline(hinted_pipeline, dict(X="*fp32"), dict(STAGE=stage, ORDER=order))
    text = str(lower_offline(module, ctx))
    assert task_contract(text) == [("load", stage)]
    assert ("hivm.pipeline_stage" in text) == (stage is not None)
    assert ("hivm.task_order" in text) == (order is not None)
    assert re.search(r"\bhivm\.pipeline(?:[,}\s])", text)
    assert "hivm.schedule_domain" not in text


@tla.jit
def unnamed_pipeline_scope(X):
    for t in tla.pipeline(2):
        tile = tla.alloc((64,), tl.float32, space="ub")
        with tla.scope.vector():
            tla.load_gm_to_ub(tla.view(X + t * 64, (64,)), tile)


def test_plain_core_scope_is_a_task_hint():
    module, ctx = compile_offline(unnamed_pipeline_scope, dict(X="*fp32"), {})
    text = str(lower_offline(module, ctx))
    tasks = task_contract(text)
    assert len(tasks) == 1 and tasks[0][0].startswith("__scope_")
    assert tasks[0][1] is None
    assert "hivm.task_order" not in text
    assert not hasattr(tla.scope, "schedule")


@tla.jit
def missing_pipeline_scope(X):
    for t in tla.pipeline(2):
        tile = tla.alloc((64,), tl.float32, space="ub")
        tla.load_gm_to_ub(tla.view(X + t * 64, (64,)), tile)


@tla.jit
def direct_simd_pipeline():
    for t in tla.pipeline(2):
        with tla.scope.simd():
            pass


@tla.jit
def nested_pipeline():
    for t in tla.pipeline(2):
        for k in tla.pipeline(2):
            with tla.scope.vector():
                pass


@tla.jit
def carried_pipeline():
    acc = tl.full((), 0, tl.int32)
    for t in tla.pipeline(2):
        acc += 1
        with tla.scope.vector():
            pass


@pytest.mark.parametrize("kernel,signature,error", [
    (missing_pipeline_scope, dict(X="*fp32"), "requires at least one"),
    (direct_simd_pipeline, {}, "scope hints must be vector"),
    (nested_pipeline, {}, "nested tla.pipeline"),
    (carried_pipeline, {}, "loop-carried values"),
])
def test_pipeline_structure_errors(kernel, signature, error):
    with pytest.raises(CompilationError, match=error):
        compile_offline(kernel, signature, {})
