"""TLA 3-in-1 test: device_print (route A: reuses tt.print).

Semantics: copy GM->UB, print two UB slices outside the SIMD scope, then copy
UB->GM inside it.

Validation (see device_print.py): calls inside a tla.scope and non-UB values
are rejected with RuntimeError.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline-compile, assert plain tt.print at AST stage and
            triton_print func calls after lower (npuir rewrites these to
            hivm.hir.debug -> _mlir_ciface_print_* libcalls)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch (prints go to the device debug tunnel)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tla_ut/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_compile_artifacts)

SHAPE = dict(M=8, N=64)

# offline compile signature (pointer entry takes pointer args).
SIGNATURE = {"A_ptr": "*fp32"}


@triton.jit
def device_print_kernel(A_ptr, M: tl.constexpr, N: tl.constexpr):
    a_ub = tla.alloc_buffer((M, N), tl.float32, space="ub")
    a_blk = tla.view(A_ptr, (M, N), strides=(N, 1))
    tla.copy(a_blk, a_ub)  # GM -> UB (hivm.hir.load)
    tla.device_print("ub[0, 0] = ", a_ub[0, 0])
    tla.device_print("ub[0, 1] = ", a_ub[0, 1], hex=True)
    with tla.scope.simd():
        tla.copy(a_ub, a_blk)  # UB -> GM (hivm.hir.store)


# negative kernels: must raise RuntimeError at compile time.
def _bad_inside_scope():
    @triton.jit
    def k(A_ptr, M: tl.constexpr, N: tl.constexpr):
        a_ub = tla.alloc_buffer((M, N), tl.float32, space="ub")
        a_blk = tla.view(A_ptr, (M, N), strides=(N, 1))
        tla.copy(a_blk, a_ub)
        with tla.scope.simd():
            tla.device_print("x = ", a_ub[0, 0])
        tla.copy(a_ub, a_blk)

    return k


def _bad_l1():
    @triton.jit
    def k(A_ptr, M: tl.constexpr, N: tl.constexpr):
        a_l1 = tla.alloc_buffer((M, N), tl.float32, space="l1")
        a_blk = tla.view(A_ptr, (M, N), strides=(N, 1))
        tla.copy(a_blk, a_l1)
        tla.device_print("x = ", a_l1[0, 0])
        tla.copy(a_l1, a_blk)

    return k


def _bad_gm():
    @triton.jit
    def k(A_ptr, M: tl.constexpr, N: tl.constexpr):
        a_ub = tla.alloc_buffer((M, N), tl.float32, space="ub")
        a_blk = tla.view(A_ptr, (M, N), strides=(N, 1))
        tla.copy(a_blk, a_ub)
        tla.device_print("x = ", a_blk[0, 0])  # GM value
        with tla.scope.simd():
            tla.copy(a_ub, a_blk)

    return k


def constexprs():
    s = SHAPE
    return {"M": s["M"], "N": s["N"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(device_print_kernel, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("tt.print") == 2, "route A keeps plain tt.print at AST stage"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    for forbidden in ("!tt.ptr", "tt.ptr_to_int", "tensor<"):
        assert forbidden not in text, forbidden
    assert text.count("triton_print") == 2, "print lowered to triton_print func calls"
    assert text.count("hivm.hir.load") == 1 and text.count("hivm.hir.store") == 1
    # validation: inside-scope, L1, and GM prints must be rejected.
    for bad in (_bad_inside_scope, _bad_l1, _bad_gm):
        try:
            compile_offline(bad(), SIGNATURE, constexprs())
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"expected RuntimeError from {bad.__name__}")
    print("IR PASS: device_print (tt.print x2 -> triton_print x2, 1 load / 1 store, validation ok)")


def _launch():
    s = SHAPE
    A = torch.randn(s["M"], s["N"], dtype=torch.float32, device="npu")
    device_print_kernel[(1, )](A, **constexprs())
    torch.npu.synchronize()
    return A


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: device_print generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    _launch()
    print("DEVICE_PRINT OK")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: device_print", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
