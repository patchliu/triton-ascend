"""Explicit fixed-stage frontend contracts and single-KV-tile attention.

Frontend IR-only: no pipeline materialization or synchronization has run.
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
from utils.tla_test_utils import (compile_offline, lower_offline,
                                  check_compile_artifacts, get_stages)


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
        with tla.scope.vector().task("softmax", stage=2):
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
    return [(name, int(stage))
            for stage, name in re.findall(r'hivm.pipeline_stage = (\d+) : i32, hivm.task_name = "([^"]+)"', text)]


def test_ir(output_dir=None):
    module, ctx = compile_offline(attention_stages, SIGNATURE, dict(TILES=3), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert task_contract(text) == list(EXPECTED_STAGES.items())
    assert re.search(r"\bhivm\.pipeline(?:[,}\s])", text)
    assert "hivm.schedule_domain" not in text and "hivm.schedule_state" not in text
    assert "hivm.required_stage" not in text and "hivm.preload_fixed_depth" not in text
    assert "hivm.task_order" not in text
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


@tla.jit
def task_copy(X, Y, NAME: tl.constexpr, STAGE: tl.constexpr):
    for t in tla.pipeline(2):
        a = tla.alloc((64, ), tl.float32, space="ub")
        with tla.scope.vector().task(NAME, stage=STAGE):
            tla.load_gm_to_ub(tla.view(X, (64, )), a)
            tla.store_ub_to_gm(a, tla.view(Y, (64, )))


@pytest.mark.parametrize("name,stage,error", [
    ("", 0, "nonempty"),
    ("copy", None, "compile-time integer"),
    ("copy", -1, "compile-time integer"),
    ("copy", True, "compile-time integer"),
    ("copy", 1.5, "compile-time integer"),
    ("copy", 2**31 - 1, "compile-time integer"),
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


def test_task_requires_stage_and_rejects_order():
    with pytest.raises(TypeError, match="stage"):
        tla.scope.vector().task("load")
    with pytest.raises(TypeError, match="order"):
        tla.scope.vector().task("load", stage=0, order=1)


@tla.jit
def unnamed_pipeline_scope(X):
    for t in tla.pipeline(2):
        tile = tla.alloc((64,), tl.float32, space="ub")
        with tla.scope.vector():
            tla.load_gm_to_ub(tla.view(X + t * 64, (64,)), tile)


def test_pipeline_requires_explicit_tasks():
    with pytest.raises(CompilationError, match="pipeline scopes require"):
        compile_offline(unnamed_pipeline_scope, dict(X="*fp32"), {})


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
        with tla.scope.vector().task("work", stage=0):
            pass


@pytest.mark.parametrize("kernel,signature,error", [
    (missing_pipeline_scope, dict(X="*fp32"), "requires at least one"),
    (direct_simd_pipeline, {}, "pipeline scopes require"),
    (nested_pipeline, {}, "nested tla.pipeline"),
    (carried_pipeline, {}, "loop-carried values"),
])
def test_pipeline_structure_errors(kernel, signature, error):
    with pytest.raises(CompilationError, match=error):
        compile_offline(kernel, signature, {})


@tla.jit
def static_task_copies(X, Y):
    for t in tla.pipeline(2):
        tile = tla.alloc((64,), tl.float32, space="ub")
        for block in tl.static_range(4):
            with tla.scope.vector().task(f"copy_{block}", stage=0):
                offset = (t * 4 + block) * 64
                tla.load_gm_to_ub(tla.view(X + offset, (64,)), tile)
                tla.store_ub_to_gm(tile, tla.view(Y + offset, (64,)))


def test_static_range_expands_direct_tasks():
    module, ctx = compile_offline(static_task_copies, dict(X="*fp32", Y="*fp32"), {})
    text = str(lower_offline(module, ctx))
    assert task_contract(text) == [(f"copy_{i}", 0) for i in range(4)]
    assert text.count("scf.for ") == 1  # Only the enclosing pipeline loop.


@tla.jit
def runtime_task_copies(X):
    for t in tla.pipeline(2):
        tile = tla.alloc((64,), tl.float32, space="ub")
        for block in tl.range(4):
            with tla.scope.vector().task("load", stage=0):
                tla.load_gm_to_ub(tla.view(X + (t * 4 + block) * 64, (64,)), tile)


def test_runtime_loop_does_not_make_tasks_direct():
    with pytest.raises(CompilationError, match="direct children"):
        compile_offline(runtime_task_copies, dict(X="*fp32"), {})


@tla.jit
def task_inside_task():
    for t in tla.pipeline(2):
        with tla.scope.vector().task("outer", stage=0):
            with tla.scope.vector().task("inner", stage=0):
                pass


def test_task_cannot_nest_in_task():
    with pytest.raises(CompilationError, match="direct children"):
        compile_offline(task_inside_task, {}, {})


@tla.jit
def query_local_recurrence(X, Y, QUERIES: tl.constexpr, BLOCKS: tl.constexpr):
    for query in tl.range(QUERIES):
        state = tla.alloc((64,), tl.float32, space="ub")
        with tla.scope.vector():
            with tla.scope.simd():
                tla.simd.vsts(state[0], tla.simd.vfull(0.0, tl.float32, 64))
        for block in tla.pipeline(BLOCKS):
            tile = tla.alloc((64,), tl.float32, space="ub")
            with tla.scope.vector().task("load", stage=0):
                offset = (query * BLOCKS + block) * 64
                tla.load_gm_to_ub(tla.view(X + offset, (64,)), tile)
            with tla.scope.vector().task("update", stage=1):
                with tla.scope.simd():
                    value = tla.simd.vadd(tla.simd.vlds(state[0]), tla.simd.vlds(tile[0]))
                    tla.simd.vsts(state[0], value)
        with tla.scope.vector():
            tla.store_ub_to_gm(state, tla.view(Y + query * 64, (64,)))


@pytest.mark.parametrize("blocks", [0, 1, 4])
def test_pipeline_in_ordinary_for(blocks):
    module, _ = compile_offline(query_local_recurrence, dict(X="*fp32", Y="*fp32"),
                                dict(QUERIES=2, BLOCKS=blocks))
    # Frontend admission only, including zero trips; do not rely on how later
    # canonicalization simplifies an empty pipeline.
    text = str(module)
    assert task_contract(text) == [("load", 0), ("update", 1)]
    assert len(re.findall(r"\bhivm\.pipeline(?:[,}\s])", text)) == 1
    assert "sync_block" not in text


@tla.jit
def reversed_stages():
    for t in tla.pipeline(2):
        with tla.scope.vector().task("late", stage=1):
            pass
        with tla.scope.vector().task("early", stage=0):
            pass


def test_stage_order_is_preserved():
    module, _ = compile_offline(reversed_stages, {}, {})
    assert task_contract(str(module)) == [("late", 1), ("early", 0)]


@pytest.mark.parametrize("stage", [0, 63, 64, 2**31 - 2])
def test_stage_range_matches_backend(stage):
    assert tla.scope.vector().task("work", stage=stage).stage == stage


TILES = 3
ROWS = 16
K_ROWS = 128
COLS = 64


def _inputs():
    import torch
    q = torch.randn(TILES * ROWS * COLS, dtype=torch.float32, device="npu").to(torch.bfloat16)
    k = torch.randn(K_ROWS * COLS, dtype=torch.float32, device="npu").to(torch.bfloat16)
    v = torch.randn(K_ROWS * COLS, dtype=torch.float32, device="npu").to(torch.bfloat16)
    o = torch.zeros(TILES * ROWS * COLS, dtype=torch.float32, device="npu")
    return q, k, v, o


def _reference(q, k, v):
    """softmax(Q K^T / 8) V with P rounded to BF16, computed in FP32."""
    import torch
    qt = q.to(torch.float32).reshape(TILES, ROWS, COLS)
    kt = k.to(torch.float32).reshape(K_ROWS, COLS)
    vt = v.to(torch.float32).reshape(K_ROWS, COLS)
    s = torch.softmax((qt @ kt.transpose(-1, -2)) * 0.125, dim=-1)
    p = s.to(torch.bfloat16).to(torch.float32)
    return (p @ vt).reshape(-1)


def _launch():
    q, k, v, o = _inputs()
    attention_stages[(1, )](q, k, v, o, TILES=TILES)
    return o, _reference(q.cpu(), k.cpu(), v.cpu())


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: task_stage artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    import torch
    out, ref = _launch()
    torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)
    print("TASK_STAGE OK: softmax(Q K^T / 8) V through the task/stage pipeline")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: task_stage", "ir" if do_ir else "",
          "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
