"""Sparse MLA implementation consolidated from Downloads/x/dsa; arithmetic unchanged."""
import sys

# ---- config ----
from dataclasses import dataclass


@dataclass(frozen=True)
class Tuning:
    """只描述编译期调度，不改变算子的数学含义。"""

    bm: int
    bn: int
    split: int
    direct_nz: bool = False
    pv_cols: int = 512
    share_score: bool = False


# (batch, query length, heads, KV length, number of selected indices)
CASE_SHAPES = {
    "decode_b1": (1, 1, 128, 8192, 2048),
    "decode_b8": (8, 1, 128, 32768, 2048),
    "decode_b32": (32, 1, 128, 32768, 2048),
    "decode_tp8": (8, 1, 16, 32768, 2048),
    "prefill_q16": (1, 16, 128, 8192, 2048),
    "prefill_q64": (1, 64, 128, 32768, 2048),
    "prefill_tp8": (1, 128, 16, 32768, 2048),
    "prefill_k512": (1, 128, 64, 32768, 512),
    "decode_b4_h128": (4, 1, 128, 8192, 2048),
    "prefill_b4_h128": (4, 8192, 128, 8192, 2048),
    "decode_b4_h64": (4, 1, 64, 8192, 2048),
    "prefill_b4_h64": (4, 8192, 64, 8192, 2048),
    "decode_b4_h32": (4, 1, 32, 8192, 2048),
    "prefill_b4_h32": (4, 8192, 32, 8192, 2048),
    "decode_b4_h16": (4, 1, 16, 8192, 2048),
    "prefill_b4_h16": (4, 8192, 16, 8192, 2048),
}

SELECTED = {
    "decode_b1": Tuning(64, 64, 14, direct_nz=True, pv_cols=256),
    "decode_b8": Tuning(128, 128, 2, pv_cols=256, share_score=True),
    "decode_b32": Tuning(128, 128, 4, pv_cols=256, share_score=True),
    "decode_tp8": Tuning(16, 128, 8),
    "prefill_q16": Tuning(128, 128, 4, pv_cols=256, share_score=True),
    "prefill_q64": Tuning(128, 128, 1, pv_cols=256, share_score=True),
    "prefill_tp8": Tuning(16, 256, 1),
    "prefill_k512": Tuning(64, 128, 1),
    "decode_b4_h128": Tuning(128, 128, 8, pv_cols=256, share_score=True),
    "prefill_b4_h128": Tuning(128, 128, 1, pv_cols=256, share_score=True),
    "decode_b4_h64": Tuning(64, 128, 8),
    "prefill_b4_h64": Tuning(64, 128, 1),
    "decode_b4_h32": Tuning(32, 128, 8),
    "prefill_b4_h32": Tuning(32, 128, 1),
    "decode_b4_h16": Tuning(16, 128, 8),
    "prefill_b4_h16": Tuning(16, 256, 1),
}

_BY_SHAPE = {CASE_SHAPES[name]: tuning for name, tuning in SELECTED.items()}


def select(shape):
    """只接受已有验证配置；额外规格需要调用者显式给出 Tuning。"""
    try:
        return _BY_SHAPE[tuple(shape)]
    except KeyError as error:
        raise ValueError(f"没有验证过的 DSA 配置: {tuple(shape)}") from error

# ---- compat ----
import hashlib
import os
import triton.language.extra.tla as tla
from pathlib import Path
import tempfile
import triton.language as tl
from triton.language.core import builtin, _unwrap_if_constexpr
from triton.language.extra.tla.core import vector
from triton.language.extra.tla.cube import _logical_shape
from triton.language.extra.tla.simd import _all_mask, _scalar_handle

# 在 import 前明确选择。默认 legacy 用于与附件逐指令复验；public 严格要求
# 上游 API 完整；auto 仅供交付使用，测试报告必须记录 SELECTED_API_MODE。
REQUESTED_API_MODE = os.environ.get("DSA_API_MODE", "legacy")
if REQUESTED_API_MODE not in ("legacy", "public", "auto"):
    raise ValueError("DSA_API_MODE must be legacy, public or auto")
_REQUIRED = {
    "scalar_load": hasattr(tla, "scalar_load"),
    "byte_view": hasattr(tla, "byte_view"),
    "load_gm_to_ub_strided": hasattr(tla, "load_gm_to_ub_strided"),
    "cube.mmad": hasattr(tla.cube, "mmad"),
    "simd.vmuls": hasattr(tla.simd, "vmuls"),
    "simd.vexpdif": hasattr(tla.simd, "vexpdif"),
}
MISSING_PUBLIC_APIS = tuple(name for name, available in _REQUIRED.items() if not available)
if REQUESTED_API_MODE == "public" and MISSING_PUBLIC_APIS:
    raise RuntimeError("Missing public TLA APIs: " + ", ".join(MISSING_PUBLIC_APIS))
_USE_PUBLIC = REQUESTED_API_MODE == "public" or (REQUESTED_API_MODE == "auto" and not MISSING_PUBLIC_APIS)
SELECTED_API_MODE = "public" if _USE_PUBLIC else "legacy"


def install():
    if _USE_PUBLIC:
        return
    from triton.compiler.code_generator import CodeGenerator

    if getattr(CodeGenerator, "_issue104_native_ext", False):
        return
    original = CodeGenerator.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.builder._tla_extension_module = self.module

    CodeGenerator.__init__ = init
    CodeGenerator._issue104_native_ext = True


def _call(semantic, arguments, body, returns=()):
    from triton._C.libtriton import ir

    builder = semantic.builder
    module = builder._tla_extension_module
    types = [str(a.get_type()) for a in arguments]
    signature = ", ".join(f"%a{i}: {ty}" for i, ty in enumerate(types))
    result = " -> (" + ", ".join(returns) + ")" if returns else ""
    key = hashlib.sha256((signature + result + body).encode()).hexdigest()[:20]
    name = "__tla_native_" + key
    if not module.has_function(name):
        text = (f"module {{ tt.func private @{name}({signature}){result} {{\n{body}\n}} }}")
        with tempfile.TemporaryDirectory(prefix="tla-native-op-") as directory:
            path = Path(directory) / "op.mlir"
            path.write_text(text)
            helper = ir.parse_mlir_module(str(path), arguments[0].get_context())
        # ModuleOp.push_back requires an already-detached operation. Splice
        # the helper module's zero-argument body instead, preserving ownership.
        source_block = helper.get_function(name).get_operation().get_block()
        destination_block = (module.get_function(module.get_entry_func_name()).get_operation().get_block())
        source_block.merge_block_before(destination_block)
    return builder.call(module.get_function(name), arguments)


@builtin
def vmuls(src, scalar, _semantic=None):
    """Native FP32 vector-scalar multiply through the existing TLA binding."""
    if src.dtype != tl.float32 or src.shape != (64, ):
        raise ValueError("vmuls currently requires vector<64xf32>")
    handle = _scalar_handle(src, scalar, _semantic)
    mask = _all_mask(64, _semantic)
    result = _semantic.builder.create_simd_vec_scalar("muls", src.handle, handle, mask.handle)
    return vector(result, src.type)


@builtin
def vexpdif(lhs, rhs, _semantic=None):
    if lhs.type != rhs.type or lhs.dtype != tl.float32 or lhs.shape != (64, ):
        raise ValueError("vexpdif currently requires two vector<64xf32> operands")
    mask = _all_mask(64, _semantic)
    ty, mt = str(lhs.handle.get_type()), str(mask.handle.get_type())
    # The installed AVE dialect has no inliner interface, unlike HIVM.
    # Splice a typed native-op region at this insertion point. The constant
    # condition is removed by the normal TTIR canonicalizer; final IR must
    # contain vexpdif directly, with no helper call or conditional remaining.
    from triton._C.libtriton import ir

    builder = _semantic.builder
    text = f"""module {{ tt.func private @primitive(%a: {ty}, %b: {ty}, %p: {mt}) -> {ty} {{
      %r = scf.execute_region -> {ty} {{
        %e = ave.hir.vexpdif %a, %b, %p : {ty}, {mt}
        scf.yield %e : {ty}
      }}
      tt.return %r : {ty}
    }} }}"""
    with tempfile.TemporaryDirectory(prefix="tla-native-exp-") as directory:
        path = Path(directory) / "op.mlir"
        path.write_text(text)
        helper = ir.parse_mlir_module(str(path), lhs.handle.get_context())
    fn = helper.get_function("primitive")
    for i, handle in enumerate((lhs.handle, rhs.handle, mask.handle)):
        fn.args(i).replace_all_uses_with(handle)
    native = []
    helper.walk(lambda op: native.append(op) if op.get_name() == "ave.hir.vexpdif" else None)
    condition = builder.get_int1(True)
    branch = builder.create_if_op([lhs.handle.get_type()], condition, True)
    native[0].get_block().merge_block_before(branch.get_then_block())
    point = builder.get_insertion_point()
    builder.set_insertion_point_to_end(branch.get_else_block())
    builder.create_yield_op([lhs.handle])
    builder.restore_insertion_point(point)
    return vector(branch.get_result(0), lhs.type)


@builtin
def matmul(a, b, acc, transpose_b=False, clear_accum=False, transpose_a=False, _semantic=None):
    ta, tb, clear = (bool(_unwrap_if_constexpr(x)) for x in (transpose_a, transpose_b, clear_accum))
    m, k = _logical_shape(a.shape)
    k2, n = _logical_shape(b.shape)
    if ta:
        m, k = k, m
    if tb:
        k2, n = n, k2
    if (a.space, b.space, acc.space) != ("l1", "l1", "l0c"):
        raise ValueError("MMAD requires L1/L1/L0C")
    if a.dtype != b.dtype or k != k2 or _logical_shape(acc.shape) != (m, n):
        raise ValueError("MMAD logical shapes or dtypes do not agree")
    attrs = ", ".join(x for x, enabled in [("a_transpose", ta), ("b_transpose", tb)] if enabled)
    attrs = " {" + attrs + "}" if attrs else ""
    at, bt, ct = (str(x.handle.get_type()) for x in (a, b, acc))
    body = f"""%init = arith.constant {str(clear).lower()}
%m = arith.constant {m} : index
%k = arith.constant {k} : index
%n = arith.constant {n} : index
hivm.hir.mmadL1{attrs} ins(%a0, %a1, %init, %m, %k, %n : {at}, {bt}, i1, index, index, index) outs(%a2 : {ct})
tt.return"""
    _call(_semantic, [a.handle, b.handle, acc.handle], body)


install()


@builtin
def scalar_load(buf, index, _semantic=None):
    if buf.space != "ub" or buf.dtype != tl.int32 or len(buf.shape) != 1:
        raise ValueError("scalar_load requires a rank-one int32 UB buffer")
    i = _semantic.to_tensor(index)
    bt = str(buf.handle.get_type())
    it = str(i.handle.get_type())
    body = f"""%i = arith.index_cast %a1 : {it} to index
%value = memref.load %a0[%i] : {bt}
tt.return %value : i32"""
    values = _call(_semantic, [buf.handle, i.handle], body, ("i32", ))
    return tl.tensor(values.get_result(0), tl.int32)


@builtin
def load_pair(base, offset, stride, dst, _semantic=None):
    """Two contiguous rows from GM with a runtime row stride (in elements).

    Caller ensures both rows are in bounds and stride is positive and fits DMA
    hardware limits. base is the complete, zero-offset rank-one GM buffer.
    """
    if (base.space != "gm" or dst.space != "ub" or base.dtype != dst.dtype or len(base.shape) != 1
            or dst.shape[0] != 2):
        raise ValueError("load_pair requires GM base and a two-row UB destination")
    off = _semantic.to_tensor(offset)
    st = _semantic.to_tensor(stride)
    bt, dt = str(base.handle.get_type()), str(dst.handle.get_type())
    ot, stt = str(off.handle.get_type()), str(st.handle.get_type())
    width = dst.shape[1]
    elem = "bf16"
    if dst.dtype != tl.bfloat16:
        raise ValueError("BF16 only")
    view = (f"memref<2x{width}x{elem}, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>")
    body = f"""%offset = arith.index_cast %a1 : {ot} to index
%stride = arith.index_cast %a2 : {stt} to index
%source = memref.reinterpret_cast %a0 to offset: [%offset], sizes: [2, {width}], strides: [%stride, 1] : {bt} to {view}
hivm.hir.load ins(%source : {view}) outs(%a3 : {dt})
tt.return"""
    _call(_semantic, [base.handle, off.handle, st.handle, dst.handle], body)


@builtin
def byte_view(base, shape, dtype, _semantic=None):
    """Zero-offset typed view into a byte allocation; aliases are explicit IR.

    The caller must respect lifetimes. Compiler alias and synchronization analysis
    see the memref.view operation; this is not a no-alias claim.
    """
    from triton.language.extra.tla.core import buffer, buffer_type
    import math

    shape = tuple(_unwrap_if_constexpr(x) for x in _unwrap_if_constexpr(shape))
    dtype = _unwrap_if_constexpr(dtype)
    if base.space != "ub" or base.dtype != tl.int8 or len(base.shape) != 1:
        raise ValueError("byte_view requires a rank-one UB byte allocation")
    bits = {"bf16": 16, "fp32": 32}[str(dtype)]
    if math.prod(shape) * bits // 8 > base.shape[0]:
        raise ValueError("view exceeds byte allocation")
    ty = buffer_type(shape, dtype, "ub")
    rt = str(ty.to_ir(_semantic.builder))
    bt = str(base.handle.get_type())
    body = f"""%zero = arith.constant 0 : index
%view = memref.view %a0[%zero][] : {bt} to {rt}
tt.return %view : {rt}"""
    result = _call(_semantic, [base.handle], body, (rt, ))
    return buffer(result.get_result(0), ty)


@builtin
def load_rows(base, offset, stride, count, dst, _semantic=None):
    """One or two BF16 rows; positive stride, runtime DMA burst count.

    One-row copies allow repeated indices without an out-of-bounds second read.
    The caller fills the duplicate row from row zero during the layout conversion.
    """
    if (base.space != "gm" or dst.space != "ub" or dst.shape[0] != 2 or base.dtype != tl.bfloat16):
        raise ValueError("load_rows expects BF16 GM and a two-row UB tile")
    off, st, n = (_semantic.to_tensor(x) for x in (offset, stride, count))
    bt, dt = str(base.handle.get_type()), str(dst.handle.get_type())
    width = dst.shape[1]
    src = f"memref<?x{width}xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>"
    full = (f"memref<2x{width}xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>")
    dst_offset = "?" if dst.offset is None else str(dst.offset)
    dest = f"memref<?x{width}xbf16, strided<[{dst.strides[0]}, 1], offset: {dst_offset}>, #hivm.address_space<ub>>"
    body = f"""%offset = arith.index_cast %a1 : {off.handle.get_type()} to index
%stride = arith.index_cast %a2 : {st.handle.get_type()} to index
%count = arith.index_cast %a4 : {n.handle.get_type()} to index
%full = memref.reinterpret_cast %a0 to offset: [%offset], sizes: [2, {width}], strides: [%stride, 1] : {bt} to {full}
%src = memref.subview %full[0, 0] [%count, {width}] [1, 1] : {full} to {src}
%dst = memref.subview %a3[0, 0] [%count, {width}] [1, 1] : {dt} to {dest}
hivm.hir.load ins(%src : {src}) outs(%dst : {dest})
tt.return"""
    _call(_semantic, [base.handle, off.handle, st.handle, dst.handle, n.handle], body)


if _USE_PUBLIC:
    # 主核仍使用同样的数学与调度；这里只替换前端构造 IR 的入口。
    # public 模式不会 monkeypatch CodeGenerator，也不解析/拼接 MLIR 文本。
    scalar_load = tla.scalar_load
    byte_view = tla.byte_view
    matmul = tla.cube.mmad
    vmuls = tla.simd.vmuls
    vexpdif = tla.simd.vexpdif

    @builtin
    def load_pair(base, offset, stride, dst, _semantic=None):
        return tla.load_gm_to_ub_strided(base, dst, offset, stride, _semantic=_semantic)

    @builtin
    def load_rows(base, offset, stride, count, dst, _semantic=None):
        return tla.load_gm_to_ub_strided(base, dst, offset, stride, rows=count, _semantic=_semantic)

ops = sys.modules[__name__]

# ---- merge ----
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction



@TLAJITFunction
def merge_partials(
    P,
    PM,
    PL,
    O,
    OM,
    OL,
    T: tl.constexpr,
    H: tl.constexpr,
    SPLIT: tl.constexpr,
    CORES: tl.constexpr,
):
    GH: tl.constexpr = 4 if T * H >= 512 else 1
    with tla.scope.vector():
        for job in tl.range(tl.program_id(0), T * (H // GH), CORES):
            token = job // (H // GH)
            head = (job % (H // GH)) * GH
            values = tla.alloc((SPLIT, GH * 512), tl.float32, space="ub")
            vals = tla.reshape(values, (SPLIT, GH, 512))
            maxes = tla.alloc((SPLIT, 8), tl.float32, space="ub")
            sums = tla.alloc((SPLIT, 8), tl.float32, space="ub")
            mout = tla.alloc((64, ), tl.float32, space="ub")
            lout = tla.alloc((64, ), tl.float32, space="ub")
            out = tla.alloc((GH, 512), tl.bfloat16, space="ub")
            tla.load_gm_to_ub(
                tla.view(
                    P + (token * SPLIT * H + head) * 512,
                    (SPLIT, GH * 512),
                    strides=(H * 512, 1),
                ),
                values,
            )
            tla.load_gm_to_ub(
                tla.view(PM + token * SPLIT * H + head, (SPLIT, GH), strides=(H, 1)),
                tla.view(maxes, (SPLIT, GH)),
            )
            tla.load_gm_to_ub(
                tla.view(PL + token * SPLIT * H + head, (SPLIT, GH), strides=(H, 1)),
                tla.view(sums, (SPLIT, GH)),
            )
            with tla.scope.simd():
                for h in tl.range(GH):
                    maximum = tla.simd.vfull(float("-inf"), tl.float32, 64)
                    for i in tl.static_range(SPLIT):
                        maximum = tla.simd.vmax(maximum, tla.simd.vlds(maxes[i, h], dist="BRC_B32"))
                    den = tla.simd.vfull(0.0, tl.float32, 64)
                    weights = ()
                    for i in tl.static_range(SPLIT):
                        weight = ops.vexpdif(tla.simd.vlds(maxes[i, h], dist="BRC_B32"), maximum)
                        den = tla.simd.vadd(
                            den,
                            tla.simd.vmul(weight, tla.simd.vlds(sums[i, h], dist="BRC_B32")),
                        )
                        weights = weights + (weight, )
                    tla.simd.vsts(mout[h], maximum, dist="ONEPT_B32")
                    tla.simd.vsts(lout[h], den, dist="ONEPT_B32")
                    for chunk in tl.range(4):
                        y0 = tla.simd.vfull(0.0, tl.float32, 64)
                        y1 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.static_range(SPLIT):
                            y0 = tla.simd.vadd(
                                y0,
                                tla.simd.vmul(weights[i], tla.simd.vlds(vals[i, h, chunk * 128])),
                            )
                            y1 = tla.simd.vadd(
                                y1,
                                tla.simd.vmul(
                                    weights[i],
                                    tla.simd.vlds(vals[i, h, chunk * 128 + 64]),
                                ),
                            )
                        b0 = tla.simd.vtruncf(tla.simd.vdiv(y0, den), tl.bfloat16)
                        b1 = tla.simd.vtruncf(tla.simd.vdiv(y1, den), tl.bfloat16)
                        packed, unused = tla.simd.vdintlv(b0, b1)
                        tla.simd.vsts(out[h, chunk * 128], packed)
            tla.store_ub_to_gm(out, tla.view(O + (token * H + head) * 512, (GH, 512)))
            tla.store_ub_to_gm(tla.view(mout, (GH, )), tla.view(OM + token * H + head, (GH, )))
            tla.store_ub_to_gm(tla.view(lout, (GH, )), tla.view(OL + token * H + head, (GH, )))

# ---- ragged ----
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction



@TLAJITFunction
def ragged_attention(
    Q,
    KV,
    QR,
    KR,
    IDS,
    O,
    MAX,
    SUM,
    B: tl.constexpr,
    SQ: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    CORES: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
):
    D: tl.constexpr = 512
    DR: tl.constexpr = 64
    VM: tl.constexpr = BM // 2
    tl.static_assert(BN == 64)
    VN: tl.constexpr = BN // 2
    SCALE: tl.constexpr = 576**-0.5
    # 把 TOPK/64 个 tile 均分到 SPLIT 份，允许分片含不同数量的 tile。
    for job in tl.range(tl.program_id(0), B * SQ * (H // BM) * SPLIT, CORES):
        token = job // ((H // BM) * SPLIT)
        split = job % SPLIT
        output_token = token * SPLIT + split
        head = ((job // SPLIT) % (H // BM)) * BM
        batch = token // SQ
        q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        qr_l1 = tla.alloc((DR // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        kv_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        kr_l1 = tla.alloc((DR // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        kv_next = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        kr_next = tla.alloc((DR // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        kflat_next = tla.reshape(kv_next, (D // 16, BN * 16))
        rflat_next = tla.reshape(kr_next, (DR // 16, BN * 16))
        p_l1 = tla.alloc((BN // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        kflat = tla.reshape(kv_l1, (D // 16, BN * 16))
        rflat = tla.reshape(kr_l1, (DR // 16, BN * 16))
        pflat = tla.reshape(p_l1, (BN // 16, BM * 16))
        shared = tla.alloc((VM * 256 * 4, ), tl.int8, space="ub")
        # 两行 DMA 暂存；按原顺序独立分配，避免改变 UB 布局与生命周期。
        kpairs = (
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
        )
        rpairs = (
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
        )
        kpack = tla.alloc((D // 16, (VN + 1) * 16), tl.bfloat16, space="ub")
        pack_shared = tla.alloc(
            (max((DR // 16) * (VN + 1) * 16, (BN // 16) * (VM + 1) * 16) * 2, ),
            tl.int8,
            space="ub",
        )
        rpack = ops.byte_view(pack_shared, (DR // 16, (VN + 1) * 16), tl.bfloat16)
        lows = tla.alloc((64, ), tl.int32, space="ub")
        strides = tla.alloc((64, ), tl.int32, space="ub")
        counts = tla.alloc((64, ), tl.int32, space="ub")
        any_duplicate = tla.alloc((8, ), tl.int32, space="ub")
        duplicate_flags = tla.alloc((64, ), tl.int32, space="ub")
        ids0 = tla.alloc((BN, ), tl.int32, space="ub")
        ids1 = tla.alloc((BN, ), tl.int32, space="ub")
        score = tla.alloc((VM, BN), tl.float32, space="ub")
        pub = ops.byte_view(pack_shared, (BN // 16, (VM + 1) * 16), tl.bfloat16)
        pv = ops.byte_view(shared, (VM, 256), tl.float32)
        acc = tla.alloc((VM, D), tl.float32, space="ub")
        acc_alt = tla.alloc((VM, D), tl.float32, space="ub")
        out = ops.byte_view(shared, (VM, 256), tl.bfloat16)
        m = tla.alloc((64, ), tl.float32, space="ub")
        l = tla.alloc((64, ), tl.float32, space="ub")
        m_alt = tla.alloc((64, ), tl.float32, space="ub")
        l_alt = tla.alloc((64, ), tl.float32, space="ub")
        alpha = tla.alloc((64, ), tl.float32, space="ub")
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + (token * H + head) * D, (BM, D)), q_l1)
            tla.load_gm_to_l1(tla.view(QR + (token * H + head) * DR, (BM, DR)), qr_l1)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                neginf = tla.simd.vfull(float("-inf"), tl.float32, 64)
                if SPLIT > 1:
                    neginf = tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64)
                tla.simd.vsts(m[0], neginf)
                tla.simd.vsts(l[0], zero)
                for row in tl.range(0, VM):
                    for col in tl.static_range(0, D // 64):
                        tla.simd.vsts(acc[row, col * 64], zero)
        pc = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space="l0c")
        # Seed the first ready slot and one free PV-fragment credit.
        # 按 tile 数切分，而非要求 TOPK 可整除 SPLIT；split14 覆盖全部32个tile。
        first_start = ((split * (TOPK // BN)) // SPLIT) * BN
        split_end = (((split + 1) * (TOPK // BN)) // SPLIT) * BN
        with tla.scope.vector():
            sub = tla.sub_block_id()
            tla.load_gm_to_ub(tla.view(IDS + token * TOPK + first_start, (BN, )), ids0)
            with tla.scope.simd():
                ii = tla.simd.vlds(ids0[0])
                even, odd = tla.simd.vdintlv(ii, ii)
                low = tla.simd.vmin(even, odd)
                high = tla.simd.vmax(even, odd)
                sorted_ids, unused = tla.simd.vintlv(low, high)
                tla.simd.vsts(ids0[0], sorted_ids)
                zero = tla.simd.vfull(0, tl.int32, 64)
                limit = tla.simd.vfull(L - 1, tl.int32, 64)
                low = tla.simd.vmax(zero, tla.simd.vmin(limit, low))
                high = tla.simd.vmax(zero, tla.simd.vmin(limit, high))
                equal = tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ)
                flags = tla.simd.vsel(equal, tla.simd.vfull(65537, tl.int32, 64), zero)
                tla.simd.vsts(duplicate_flags[0], flags)
                tla.simd.vsts(any_duplicate[0], tla.simd.vreduce_max(flags), dist="ONEPT_B32")
                one = tla.simd.vfull(1, tl.int32, 64)
                two = tla.simd.vfull(2, tl.int32, 64)
                tla.simd.vsts(lows[0], low)
                tla.simd.vsts(strides[0], tla.simd.vmax(one, tla.simd.vsub(high, low)))
                tla.simd.vsts(counts[0], tla.simd.vsel(equal, one, two))
            if ops.scalar_load(any_duplicate, 0) == 0:
                for pair in tl.static_range(0, VN // 2):
                    row = pair * 2
                    s0 = ops.scalar_load(lows, sub * (VN // 2) + pair)
                    stride = ops.scalar_load(strides, sub * (VN // 2) + pair)
                    ops.load_pair(
                        tla.view(KV, (B * L * D, )),
                        (batch * L + s0) * D,
                        stride * D,
                        kpairs[pair],
                    )
                    ops.load_pair(
                        tla.view(KR, (B * L * DR, )),
                        (batch * L + s0) * DR,
                        stride * DR,
                        tla.view(rpairs[pair], (2, DR)),
                    )
            else:
                for pair in tl.static_range(0, VN // 2):
                    row = pair * 2
                    s0 = ops.scalar_load(lows, sub * (VN // 2) + pair)
                    stride = ops.scalar_load(strides, sub * (VN // 2) + pair)
                    count = ops.scalar_load(counts, sub * (VN // 2) + pair)
                    ops.load_rows(
                        tla.view(KV, (B * L * D, )),
                        (batch * L + s0) * D,
                        stride * D,
                        count,
                        kpairs[pair],
                    )
                    ops.load_rows(
                        tla.view(KR, (B * L * DR, )),
                        (batch * L + s0) * DR,
                        stride * DR,
                        count,
                        tla.view(rpairs[pair], (2, DR)),
                    )
            if ops.scalar_load(any_duplicate, 0) == 0:
                with tla.scope.simd():
                    for pair in tl.static_range(16):
                        for localrow in tl.static_range(2):
                            row = pair * 2 + localrow
                            for chunk in tl.static_range(D // 128):
                                values = tla.simd.vlds(kpairs[pair][localrow, chunk * 128])
                                tla.simd.vsstb(values, kpack[chunk * 8, row * 16], (VN + 1) * 16)
                            rv = tla.simd.vlds(rpairs[pair][localrow, 0])
                            tla.simd.vsstb(
                                rv,
                                rpack[0, row * 16],
                                (VN + 1) * 16,
                                mask=tla.simd.pge(64, 128),
                            )
            else:
                with tla.scope.simd():
                    for pair in tl.static_range(16):
                        flag = tla.simd.vbitcast(
                            tla.simd.vlds(duplicate_flags[sub * (VN // 2) + pair], dist="BRC_B32"),
                            "i16",
                        )
                        duplicate = tla.simd.vcmp(flag, tla.simd.vfull(0, tl.int16, 128), op=tla.simd.CMP.NE)
                        for localrow in tl.static_range(2):
                            row = pair * 2 + localrow
                            for chunk in tl.static_range(D // 128):
                                values = tla.simd.vlds(kpairs[pair][localrow, chunk * 128])
                                if localrow == 1:
                                    values = tla.simd.vsel(
                                        duplicate,
                                        tla.simd.vlds(kpairs[pair][0, chunk * 128]),
                                        values,
                                    )
                                tla.simd.vsstb(values, kpack[chunk * 8, row * 16], (VN + 1) * 16)
                            rv = tla.simd.vlds(rpairs[pair][localrow, 0])
                            if localrow == 1:
                                rv = tla.simd.vsel(duplicate, tla.simd.vlds(rpairs[pair][0, 0]), rv)
                            tla.simd.vsstb(
                                rv,
                                rpack[0, row * 16],
                                (VN + 1) * 16,
                                mask=tla.simd.pge(64, 128),
                            )
            tla.copy_ub_to_l1(
                tla.view(kpack, (D // 16, VN * 16)),
                tla.view(kflat, (D // 16, VN * 16), offsets=(0, sub * VN * 16)),
            )
            tla.copy_ub_to_l1(
                tla.view(rpack, (DR // 16, VN * 16)),
                tla.view(rflat, (DR // 16, VN * 16), offsets=(0, sub * VN * 16)),
            )
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
        with tla.scope.vector():
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        # Flags: 0/5=KV slot ready, 1=score ready, 2=P ready,
        # 3=PV fragment ready, 4=PV fragment consumed/free.
        # 先发下一 tile 的 gather，等当前 score 后做 softmax，再在 PV 期间打包 NZ。
        # old/new 双份状态避免原地更新的 Vector hazard。
        for pair_start in tl.range(first_start, split_end, 2 * BN):
            for phase in tl.static_range(2):
                start = pair_start + phase * BN
                if start < split_end:
                    if phase == 0:
                        acc_old = acc
                        acc_new = acc_alt
                        m_old = m
                        m_new = m_alt
                        l_old = l
                        l_new = l_alt
                    else:
                        acc_old = acc_alt
                        acc_new = acc
                        m_old = m_alt
                        m_new = m
                        l_old = l_alt
                        l_new = l
                    if phase == 0:
                        # Distinct old/new states avoid in-place vector hazards.
                        key = kv_l1
                        rope = kr_l1
                        ids_cur = ids0
                        ids_prefetch = ids1
                        pack_key = kflat_next
                        pack_rope = rflat_next
                    else:
                        key = kv_next
                        rope = kr_next
                        ids_cur = ids1
                        ids_prefetch = ids0
                        pack_key = kflat
                        pack_rope = rflat
                    with tla.scope.cube():
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, phase * 5)
                        sc = tla.alloc((BN // 16, BM // 16, 16, 16), tl.float32, space="l0c")
                        ops.matmul(q_l1, key, sc, transpose_b=True, clear_accum=True)
                        ops.matmul(qr_l1, rope, sc, transpose_b=True, clear_accum=False)
                        tla.copy_l0c_to_ub(
                            sc,
                            score,
                            tla.FixpipeL0C2UBParams(
                                dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                sub_block_idx=None,
                            ),
                        )
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 1)
                    if start + BN < split_end:
                        with tla.scope.vector():
                            sub = tla.sub_block_id()
                            tla.load_gm_to_ub(
                                tla.view(IDS + token * TOPK + (start + BN), (BN, )),
                                ids_prefetch,
                            )
                            with tla.scope.simd():
                                ii = tla.simd.vlds(ids_prefetch[0])
                                even, odd = tla.simd.vdintlv(ii, ii)
                                low = tla.simd.vmin(even, odd)
                                high = tla.simd.vmax(even, odd)
                                sorted_ids, unused = tla.simd.vintlv(low, high)
                                tla.simd.vsts(ids_prefetch[0], sorted_ids)
                                zero = tla.simd.vfull(0, tl.int32, 64)
                                limit = tla.simd.vfull(L - 1, tl.int32, 64)
                                low = tla.simd.vmax(zero, tla.simd.vmin(limit, low))
                                high = tla.simd.vmax(zero, tla.simd.vmin(limit, high))
                                equal = tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ)
                                flags = tla.simd.vsel(equal, tla.simd.vfull(65537, tl.int32, 64), zero)
                                tla.simd.vsts(duplicate_flags[0], flags)
                                tla.simd.vsts(
                                    any_duplicate[0],
                                    tla.simd.vreduce_max(flags),
                                    dist="ONEPT_B32",
                                )
                                one = tla.simd.vfull(1, tl.int32, 64)
                                two = tla.simd.vfull(2, tl.int32, 64)
                                tla.simd.vsts(lows[0], low)
                                tla.simd.vsts(
                                    strides[0],
                                    tla.simd.vmax(one, tla.simd.vsub(high, low)),
                                )
                                tla.simd.vsts(counts[0], tla.simd.vsel(equal, one, two))
                            if ops.scalar_load(any_duplicate, 0) == 0:
                                for pair in tl.static_range(0, VN // 2):
                                    row = pair * 2
                                    s0 = ops.scalar_load(lows, sub * (VN // 2) + pair)
                                    stride = ops.scalar_load(strides, sub * (VN // 2) + pair)
                                    ops.load_pair(
                                        tla.view(KV, (B * L * D, )),
                                        (batch * L + s0) * D,
                                        stride * D,
                                        kpairs[pair],
                                    )
                                    ops.load_pair(
                                        tla.view(KR, (B * L * DR, )),
                                        (batch * L + s0) * DR,
                                        stride * DR,
                                        tla.view(rpairs[pair], (2, DR)),
                                    )
                            else:
                                for pair in tl.static_range(0, VN // 2):
                                    row = pair * 2
                                    s0 = ops.scalar_load(lows, sub * (VN // 2) + pair)
                                    stride = ops.scalar_load(strides, sub * (VN // 2) + pair)
                                    count = ops.scalar_load(counts, sub * (VN // 2) + pair)
                                    ops.load_rows(
                                        tla.view(KV, (B * L * D, )),
                                        (batch * L + s0) * D,
                                        stride * D,
                                        count,
                                        kpairs[pair],
                                    )
                                    ops.load_rows(
                                        tla.view(KR, (B * L * DR, )),
                                        (batch * L + s0) * DR,
                                        stride * DR,
                                        count,
                                        tla.view(rpairs[pair], (2, DR)),
                                    )
                    with tla.scope.vector():
                        sub = tla.sub_block_id()
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
                        with tla.scope.simd():
                            neginf = tla.simd.vfull(float("-inf"), tl.float32, 64)
                            izero = tla.simd.vfull(0, tl.int32, 64)
                            ilimit = tla.simd.vfull(L, tl.int32, 64)
                            i0 = tla.simd.vlds(ids_cur[0])
                            i1 = i0
                            if BN == 128:
                                i1 = tla.simd.vlds(ids_cur[64])
                            for row in tl.range(0, VM):
                                x0 = ops.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                                x1 = neginf
                                if BN == 128:
                                    x1 = ops.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                                x0 = tla.simd.vsel(
                                    tla.simd.vcmp(i0, izero, op=tla.simd.CMP.GE),
                                    x0,
                                    neginf,
                                )
                                x1 = tla.simd.vsel(
                                    tla.simd.vcmp(i1, izero, op=tla.simd.CMP.GE),
                                    x1,
                                    neginf,
                                )
                                x0 = tla.simd.vsel(
                                    tla.simd.vcmp(i0, ilimit, op=tla.simd.CMP.LT),
                                    x0,
                                    neginf,
                                )
                                x1 = tla.simd.vsel(
                                    tla.simd.vcmp(i1, ilimit, op=tla.simd.CMP.LT),
                                    x1,
                                    neginf,
                                )
                                new_m = tla.simd.vreduce_max(tla.simd.vmax(x0, x1))
                                a = tla.simd.vfull(1.0, tl.float32, 64)
                                old_m = tla.simd.vlds(m_old[row], dist="BRC_B32")
                                new_m = tla.simd.vmax(new_m, old_m)
                                a = ops.vexpdif(old_m, new_m)
                                tla.simd.vsts(alpha[row], a, dist="ONEPT_B32")
                                p0 = ops.vexpdif(x0, new_m)
                                p1 = ops.vexpdif(x1, new_m)
                                new_l = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(p0, p1)))
                                old_l = tla.simd.vlds(l_old[row], dist="BRC_B32")
                                new_l = tla.simd.vadd(new_l, tla.simd.vmul(old_l, a))
                                tla.simd.vsts(m_new[row], new_m, dist="ONEPT_B32")
                                tla.simd.vsts(l_new[row], new_l, dist="ONEPT_B32")
                                packed, unused = tla.simd.vdintlv(
                                    tla.simd.vtruncf(p0, tl.bfloat16),
                                    tla.simd.vtruncf(p1, tl.bfloat16),
                                )
                                if BN == 128:
                                    tla.simd.vsstb(packed, pub[0, row * 16], (VM + 1) * 16)
                                else:
                                    tla.simd.vsstb(
                                        packed,
                                        pub[0, row * 16],
                                        (VM + 1) * 16,
                                        mask=tla.simd.pge(64, 128),
                                    )
                        tla.copy_ub_to_l1(
                            tla.view(pub, (BN // 16, VM * 16)),
                            tla.view(pflat, (BN // 16, VM * 16), offsets=(0, sub * VM * 16)),
                        )
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 2)
                    with tla.scope.cube():
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2)
                        ops.matmul(p_l1, key, pc, clear_accum=True)
                    if start + BN < split_end:
                        with tla.scope.vector():
                            sub = tla.sub_block_id()
                            if ops.scalar_load(any_duplicate, 0) == 0:
                                with tla.scope.simd():
                                    for pair in tl.static_range(16):
                                        for localrow in tl.static_range(2):
                                            row = pair * 2 + localrow
                                            for chunk in tl.static_range(D // 128):
                                                values = tla.simd.vlds(kpairs[pair][localrow, chunk * 128])
                                                tla.simd.vsstb(
                                                    values,
                                                    kpack[chunk * 8, row * 16],
                                                    (VN + 1) * 16,
                                                )
                                            rv = tla.simd.vlds(rpairs[pair][localrow, 0])
                                            tla.simd.vsstb(
                                                rv,
                                                rpack[0, row * 16],
                                                (VN + 1) * 16,
                                                mask=tla.simd.pge(64, 128),
                                            )
                            else:
                                with tla.scope.simd():
                                    for pair in tl.static_range(16):
                                        flag = tla.simd.vbitcast(
                                            tla.simd.vlds(
                                                duplicate_flags[sub * (VN // 2) + pair],
                                                dist="BRC_B32",
                                            ),
                                            "i16",
                                        )
                                        duplicate = tla.simd.vcmp(
                                            flag,
                                            tla.simd.vfull(0, tl.int16, 128),
                                            op=tla.simd.CMP.NE,
                                        )
                                        for localrow in tl.static_range(2):
                                            row = pair * 2 + localrow
                                            for chunk in tl.static_range(D // 128):
                                                values = tla.simd.vlds(kpairs[pair][localrow, chunk * 128])
                                                if localrow == 1:
                                                    values = tla.simd.vsel(
                                                        duplicate,
                                                        tla.simd.vlds(kpairs[pair][0, chunk * 128]),
                                                        values,
                                                    )
                                                tla.simd.vsstb(
                                                    values,
                                                    kpack[chunk * 8, row * 16],
                                                    (VN + 1) * 16,
                                                )
                                            rv = tla.simd.vlds(rpairs[pair][localrow, 0])
                                            if localrow == 1:
                                                rv = tla.simd.vsel(
                                                    duplicate,
                                                    tla.simd.vlds(rpairs[pair][0, 0]),
                                                    rv,
                                                )
                                            tla.simd.vsstb(
                                                rv,
                                                rpack[0, row * 16],
                                                (VN + 1) * 16,
                                                mask=tla.simd.pge(64, 128),
                                            )
                            tla.copy_ub_to_l1(
                                tla.view(kpack, (D // 16, VN * 16)),
                                tla.view(
                                    pack_key,
                                    (D // 16, VN * 16),
                                    offsets=(0, sub * VN * 16),
                                ),
                            )
                            tla.copy_ub_to_l1(
                                tla.view(rpack, (DR // 16, VN * 16)),
                                tla.view(
                                    pack_rope,
                                    (DR // 16, VN * 16),
                                    offsets=(0, sub * VN * 16),
                                ),
                            )
                            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, (1 - phase) * 5)
                    for fragment in tl.range(2):
                        with tla.scope.cube():
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                            tla.copy_l0c_to_ub(
                                tla.view(
                                    pc,
                                    (16, BM // 16, 16, 16),
                                    offsets=(fragment * 16, 0, 0, 0),
                                ),
                                pv,
                                tla.FixpipeL0C2UBParams(
                                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                    sub_block_idx=None,
                                ),
                            )
                            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 3)
                        with tla.scope.vector():
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 3)
                            with tla.scope.simd():
                                for row in tl.range(VM):
                                    a = tla.simd.vlds(alpha[row], dist="BRC_B32")
                                    for chunk in tl.static_range(2):
                                        y0 = tla.simd.vlds(pv[row, chunk * 128])
                                        y1 = tla.simd.vlds(pv[row, chunk * 128 + 64])
                                        y0 = tla.simd.vadd(
                                            y0,
                                            tla.simd.vmul(
                                                tla.simd.vlds(acc_old[
                                                    row,
                                                    fragment * 256 + chunk * 128,
                                                ]),
                                                a,
                                            ),
                                        )
                                        y1 = tla.simd.vadd(
                                            y1,
                                            tla.simd.vmul(
                                                tla.simd.vlds(acc_old[
                                                    row,
                                                    fragment * 256 + chunk * 128 + 64,
                                                ]),
                                                a,
                                            ),
                                        )
                                        tla.simd.vsts(
                                            acc_new[row, fragment * 256 + chunk * 128],
                                            y0,
                                        )
                                        tla.simd.vsts(
                                            acc_new[row, fragment * 256 + chunk * 128 + 64],
                                            y1,
                                        )
                            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
        # 分片长度可能是奇数：最终状态位于不同的 ping-pong 缓冲区。
        if ((split_end - first_start) // BN) % 2 == 0:
            with tla.scope.vector():
                sub = tla.sub_block_id()
                if SPLIT == 1:
                    for fragment in tl.range(2):
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                den = tla.simd.vlds(l[row], dist="BRC_B32")
                                for chunk in tl.static_range(2):
                                    y0 = tla.simd.vlds(acc[row, fragment * 256 + chunk * 128])
                                    y1 = tla.simd.vlds(acc[row, fragment * 256 + chunk * 128 + 64])
                                    b0 = tla.simd.vtruncf(tla.simd.vdiv(y0, den), tl.bfloat16)
                                    b1 = tla.simd.vtruncf(tla.simd.vdiv(y1, den), tl.bfloat16)
                                    packed, unused = tla.simd.vdintlv(b0, b1)
                                    tla.simd.vsts(out[row, chunk * 128], packed)
                        tla.store_ub_to_gm(
                            out,
                            tla.view(
                                O + (output_token * H + head + sub * VM) * D + fragment * 256,
                                (VM, 256),
                                strides=(D, 1),
                            ),
                        )
                else:
                    tla.store_ub_to_gm(
                        acc,
                        tla.view(O + (output_token * H + head + sub * VM) * D, (VM, D)),
                    )
                tla.store_ub_to_gm(
                    tla.view(m, (VM, ), offsets=(0, )),
                    tla.view(MAX + output_token * H + head + sub * VM, (VM, )),
                )
                tla.store_ub_to_gm(
                    tla.view(l, (VM, ), offsets=(0, )),
                    tla.view(SUM + output_token * H + head + sub * VM, (VM, )),
                )
        else:
            with tla.scope.vector():
                sub = tla.sub_block_id()
                if SPLIT == 1:
                    for fragment in tl.range(2):
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                den = tla.simd.vlds(l_alt[row], dist="BRC_B32")
                                for chunk in tl.static_range(2):
                                    y0 = tla.simd.vlds(acc_alt[row, fragment * 256 + chunk * 128])
                                    y1 = tla.simd.vlds(acc_alt[row, fragment * 256 + chunk * 128 + 64])
                                    b0 = tla.simd.vtruncf(tla.simd.vdiv(y0, den), tl.bfloat16)
                                    b1 = tla.simd.vtruncf(tla.simd.vdiv(y1, den), tl.bfloat16)
                                    packed, unused = tla.simd.vdintlv(b0, b1)
                                    tla.simd.vsts(out[row, chunk * 128], packed)
                        tla.store_ub_to_gm(
                            out,
                            tla.view(
                                O + (output_token * H + head + sub * VM) * D + fragment * 256,
                                (VM, 256),
                                strides=(D, 1),
                            ),
                        )
                else:
                    tla.store_ub_to_gm(
                        acc_alt,
                        tla.view(O + (output_token * H + head + sub * VM) * D, (VM, D)),
                    )
                tla.store_ub_to_gm(
                    tla.view(m_alt, (VM, ), offsets=(0, )),
                    tla.view(MAX + output_token * H + head + sub * VM, (VM, )),
                )
                tla.store_ub_to_gm(
                    tla.view(l_alt, (VM, ), offsets=(0, )),
                    tla.view(SUM + output_token * H + head + sub * VM, (VM, )),
                )

# ---- workspace ----
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction



@TLAJITFunction
def workspace_attention(Q, KV, QR, KR, IDS, WORK, O, MAX, SUM, B: tl.constexpr, SQ: tl.constexpr,
                        L: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr, BM: tl.constexpr,
                        CORES: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr, PV_COLS: tl.constexpr,
                        SHARE_SCORE: tl.constexpr):
    tl.static_assert(BN == 128 or BN == 256)
    VM: tl.constexpr = BM // 2
    VP: tl.constexpr = PV_COLS
    HALF_N: tl.constexpr = BN // 2
    # 一个 job = 一个 query × 一个 head tile × 一个 KV 分片。
    for job in tl.range(tl.program_id(0), B * SQ * (H // BM) * SPLIT, CORES):
        token = job // (H // BM * SPLIT)
        split = job % SPLIT
        output_token = token * SPLIT + split
        head = job // SPLIT % (H // BM) * BM
        batch = token // SQ
        # L1/NZ 供 Cube 使用；UB 保存每个 Vector 子核负责的半个 head tile。
        query_l1 = tla.alloc((36, BM // 16, 16, 16), tl.bfloat16, space='l1')
        keys_l1 = tla.alloc((36, BN // 16, 16, 16), tl.bfloat16, space='l1')
        prob_l1 = tla.alloc((BN // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
        prob_flat_l1 = tla.reshape(prob_l1, (BN // 16, BM * 16))
        ids0 = tla.alloc((BN, ), tl.int32, space='ub')
        ids1 = tla.alloc((BN, ), tl.int32, space='ub')
        if not SHARE_SCORE:
            scores_ub = tla.alloc((VM, BN), tl.float32, space='ub')
        prob_padded_ub = tla.alloc((BN // 16, (VM + 1) * 16), tl.bfloat16, space='ub')
        shared_bytes = tla.alloc((VM * VP * 4, ), tl.int8, space='ub')
        pv_ub = ops.byte_view(shared_bytes, (VM, VP), tl.float32)
        if SHARE_SCORE:
            scores_ub = ops.byte_view(shared_bytes, (VM, BN), tl.float32)
        numerator_ub = tla.alloc((VM, 512), tl.float32, space='ub')
        output_bf16 = ops.byte_view(shared_bytes, (VM, VP), tl.bfloat16)
        need_safe_gather = tla.alloc((8, ), tl.int32, space='ub')
        gather_key_even = tla.alloc((16, 512), tl.bfloat16, space='ub')
        gather_key_odd = tla.alloc((16, 512), tl.bfloat16, space='ub')
        gather_rope_even = tla.alloc((16, 64), tl.bfloat16, space='ub')
        gather_rope_odd = tla.alloc((16, 64), tl.bfloat16, space='ub')
        running_max = tla.alloc((64, ), tl.float32, space='ub')
        running_sum = tla.alloc((64, ), tl.float32, space='ub')
        rescale = tla.alloc((64, ), tl.float32, space='ub')
        score_l0c = tla.alloc((BN // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        pv_l0c = tla.alloc((VP // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        # 此路径等长切分；每个分片含偶数个 BN tile。
        begin = split * (TOPK // SPLIT)
        end = (split + 1) * (TOPK // SPLIT)
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + (token * H + head) * 512, (BM, 512)),
                              tla.view(query_l1, (32, BM // 16, 16, 16)))
            tla.load_gm_to_l1(tla.view(QR + (token * H + head) * 64, (BM, 64)),
                              tla.view(query_l1, (4, BM // 16, 16, 16), offsets=(32, 0, 0, 0)))
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                negative = tla.simd.vfull(-3.4028234663852886e+38, tl.float32, 64)
                tla.simd.vsts(running_max[0], negative)
                tla.simd.vsts(running_sum[0], zero)
                for row in tl.range(VM):
                    for col in tl.static_range(8):
                        tla.simd.vsts(numerator_ub[row, col * 64], zero)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        # 预取第一槽。后面的重复 gather 块刻意放在 QK 后，以覆盖 Cube 工作。
        with tla.scope.vector():
            gather_ids = ids0
            gather_start = begin
            gather_slot = 0
            sub = tla.sub_block_id()
            tla.load_gm_to_ub(tla.view(IDS + token * TOPK + gather_start, (BN, )), gather_ids)
            with tla.scope.simd():
                combined_flags = tla.simd.vfull(0, tl.int32, 64)
                for index_half in tl.static_range(BN // 64):
                    ii = tla.simd.vlds(gather_ids[index_half * 64])
                    even, odd = tla.simd.vdintlv(ii, ii)
                    low = tla.simd.vmin(even, odd)
                    high = tla.simd.vmax(even, odd)
                    ordered, unused = tla.simd.vintlv(low, high)
                    tla.simd.vsts(gather_ids[index_half * 64], ordered)
                    zero = tla.simd.vfull(0, tl.int32, 64)
                    limit = tla.simd.vfull(L - 1, tl.int32, 64)
                    invalid_low = tla.simd.vsel(tla.simd.vcmp(low, zero, op=tla.simd.CMP.LT),
                                                tla.simd.vfull(1, tl.int32, 64), zero)
                    invalid_high = tla.simd.vsel(tla.simd.vcmp(high, limit, op=tla.simd.CMP.GT),
                                                 tla.simd.vfull(1, tl.int32, 64), zero)
                    flags = tla.simd.vsel(tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ),
                                          tla.simd.vfull(1, tl.int32, 64), zero)
                    combined_flags = tla.simd.vmax(
                        combined_flags, tla.simd.vmax(flags, tla.simd.vmax(invalid_low, invalid_high)))
                tla.simd.vsts(need_safe_gather[0], tla.simd.vreduce_max(combined_flags), dist='ONEPT_B32')
            for pair_chunk in tl.range(BN // 64):
                for gather_phase in tl.static_range(2):
                    chunk = (pair_chunk * 2 + gather_phase) * 16
                    if gather_phase == 0:
                        keys = gather_key_even
                        ropes = gather_rope_even
                    else:
                        keys = gather_key_odd
                        ropes = gather_rope_odd
                    if ops.scalar_load(need_safe_gather, 0) == 0:
                        for pair in tl.static_range(8):
                            i0 = ops.scalar_load(gather_ids, sub * HALF_N + chunk + pair * 2)
                            i1 = ops.scalar_load(gather_ids, sub * HALF_N + chunk + pair * 2 + 1)
                            s0 = i0
                            s1 = i1
                            ops.load_pair(tla.view(KV, (B * L * 512, )), (batch * L + s0) * 512,
                                          (s1 - s0) * 512, tla.view(keys, (2, 512), offsets=(pair * 2, 0)))
                            ops.load_pair(tla.view(KR, (B * L * 64, )), (batch * L + s0) * 64, (s1 - s0) * 64,
                                          tla.view(ropes, (2, 64), offsets=(pair * 2, 0)))
                    else:
                        for row in tl.static_range(16):
                            index = ops.scalar_load(gather_ids, sub * HALF_N + chunk + row)
                            safe = tl.maximum(0, tl.minimum(index, L - 1))
                            tla.load_gm_to_ub(tla.view(KV + (batch * L + safe) * 512, (1, 512)),
                                              tla.view(keys, (1, 512), offsets=(row, 0)))
                            tla.load_gm_to_ub(tla.view(KR + (batch * L + safe) * 64, (1, 64)),
                                              tla.view(ropes, (1, 64), offsets=(row, 0)))
                    tla.store_ub_to_gm(
                        keys,
                        tla.view(WORK +
                                 ((tl.program_id(0) * 2 + gather_slot) * BN + sub * HALF_N + chunk) * 576,
                                 (16, 512),
                                 strides=(576, 1)))
                    tla.store_ub_to_gm(
                        ropes,
                        tla.view(WORK +
                                 ((tl.program_id(0) * 2 + gather_slot) * BN + sub * HALF_N + chunk) * 576 +
                                 512, (16, 64),
                                 strides=(576, 1)))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
        # 稳态：QK → 下一个 tile 的 gather → softmax → PV → FP32 累加。
        # flag 0/5: GM 槽就绪；1: score 就绪；2: P 就绪；3/4: PV 就绪/已消费。
        for pair_start in tl.range(begin, end, 2 * BN):
            for phase in tl.static_range(2):
                start = pair_start + phase * BN
                if phase == 0:
                    current = keys_l1
                    indices = ids0
                    next_indices = ids1
                else:
                    current = keys_l1
                    indices = ids1
                    next_indices = ids0
                with tla.scope.cube():
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, phase * 5)
                    tla.load_gm_to_l1(tla.view(WORK + (tl.program_id(0) * 2 + phase) * BN * 576, (BN, 576)),
                                      current)
                    ops.matmul(query_l1, current, score_l0c, transpose_b=True, clear_accum=True)
                    if SHARE_SCORE:
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                    tla.copy_l0c_to_ub(
                        score_l0c, scores_ub,
                        tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                                sub_block_idx=None))
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 1)
                with tla.scope.vector():
                    if start + BN < end:
                        gather_ids = next_indices
                        gather_start = start + BN
                        gather_slot = 1 - phase
                        sub = tla.sub_block_id()
                        tla.load_gm_to_ub(tla.view(IDS + token * TOPK + gather_start, (BN, )), gather_ids)
                        with tla.scope.simd():
                            combined_flags = tla.simd.vfull(0, tl.int32, 64)
                            for index_half in tl.static_range(BN // 64):
                                ii = tla.simd.vlds(gather_ids[index_half * 64])
                                even, odd = tla.simd.vdintlv(ii, ii)
                                low = tla.simd.vmin(even, odd)
                                high = tla.simd.vmax(even, odd)
                                ordered, unused = tla.simd.vintlv(low, high)
                                tla.simd.vsts(gather_ids[index_half * 64], ordered)
                                zero = tla.simd.vfull(0, tl.int32, 64)
                                limit = tla.simd.vfull(L - 1, tl.int32, 64)
                                invalid_low = tla.simd.vsel(tla.simd.vcmp(low, zero, op=tla.simd.CMP.LT),
                                                            tla.simd.vfull(1, tl.int32, 64), zero)
                                invalid_high = tla.simd.vsel(tla.simd.vcmp(high, limit, op=tla.simd.CMP.GT),
                                                             tla.simd.vfull(1, tl.int32, 64), zero)
                                flags = tla.simd.vsel(tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ),
                                                      tla.simd.vfull(1, tl.int32, 64), zero)
                                combined_flags = tla.simd.vmax(
                                    combined_flags,
                                    tla.simd.vmax(flags, tla.simd.vmax(invalid_low, invalid_high)))
                            tla.simd.vsts(need_safe_gather[0],
                                          tla.simd.vreduce_max(combined_flags),
                                          dist='ONEPT_B32')
                        for pair_chunk in tl.range(BN // 64):
                            for gather_phase in tl.static_range(2):
                                chunk = (pair_chunk * 2 + gather_phase) * 16
                                if gather_phase == 0:
                                    keys = gather_key_even
                                    ropes = gather_rope_even
                                else:
                                    keys = gather_key_odd
                                    ropes = gather_rope_odd
                                if ops.scalar_load(need_safe_gather, 0) == 0:
                                    for pair in tl.static_range(8):
                                        i0 = ops.scalar_load(gather_ids, sub * HALF_N + chunk + pair * 2)
                                        i1 = ops.scalar_load(gather_ids, sub * HALF_N + chunk + pair * 2 + 1)
                                        s0 = i0
                                        s1 = i1
                                        ops.load_pair(tla.view(KV, (B * L * 512, )), (batch * L + s0) * 512,
                                                      (s1 - s0) * 512,
                                                      tla.view(keys, (2, 512), offsets=(pair * 2, 0)))
                                        ops.load_pair(tla.view(KR, (B * L * 64, )), (batch * L + s0) * 64,
                                                      (s1 - s0) * 64,
                                                      tla.view(ropes, (2, 64), offsets=(pair * 2, 0)))
                                else:
                                    for row in tl.static_range(16):
                                        index = ops.scalar_load(gather_ids, sub * HALF_N + chunk + row)
                                        safe = tl.maximum(0, tl.minimum(index, L - 1))
                                        tla.load_gm_to_ub(tla.view(KV + (batch * L + safe) * 512, (1, 512)),
                                                          tla.view(keys, (1, 512), offsets=(row, 0)))
                                        tla.load_gm_to_ub(tla.view(KR + (batch * L + safe) * 64, (1, 64)),
                                                          tla.view(ropes, (1, 64), offsets=(row, 0)))
                                tla.store_ub_to_gm(
                                    keys,
                                    tla.view(
                                        WORK +
                                        ((tl.program_id(0) * 2 + gather_slot) * BN + sub * HALF_N + chunk) *
                                        576, (16, 512),
                                        strides=(576, 1)))
                                tla.store_ub_to_gm(
                                    ropes,
                                    tla.view(
                                        WORK +
                                        ((tl.program_id(0) * 2 + gather_slot) * BN + sub * HALF_N + chunk) *
                                        576 + 512, (16, 64),
                                        strides=(576, 1)))
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, (1 - phase) * 5)
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
                    sub = tla.sub_block_id()
                    if BN == 128:
                        with tla.scope.simd():
                            negative = tla.simd.vfull(float('-inf'), tl.float32, 64)
                            zero = tla.simd.vfull(0, tl.int32, 64)
                            limit = tla.simd.vfull(L, tl.int32, 64)
                            i0 = tla.simd.vlds(indices[0])
                            i1 = tla.simd.vlds(indices[64])
                            for row in tl.range(VM):
                                x0 = ops.vmuls(tla.simd.vlds(scores_ub[row, 0]), 576**(-0.5))
                                x1 = ops.vmuls(tla.simd.vlds(scores_ub[row, 64]), 576**(-0.5))
                                x0 = tla.simd.vsel(tla.simd.vcmp(i0, zero, op=tla.simd.CMP.GE), x0, negative)
                                x1 = tla.simd.vsel(tla.simd.vcmp(i1, zero, op=tla.simd.CMP.GE), x1, negative)
                                x0 = tla.simd.vsel(tla.simd.vcmp(i0, limit, op=tla.simd.CMP.LT), x0, negative)
                                x1 = tla.simd.vsel(tla.simd.vcmp(i1, limit, op=tla.simd.CMP.LT), x1, negative)
                                old_m = tla.simd.vlds(running_max[row], dist='BRC_B32')
                                maximum = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_m)
                                scale_old = ops.vexpdif(old_m, maximum)
                                p0 = ops.vexpdif(x0, maximum)
                                p1 = ops.vexpdif(x1, maximum)
                                den = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(p0, p1)))
                                den = tla.simd.vadd(
                                    den,
                                    tla.simd.vmul(tla.simd.vlds(running_sum[row], dist='BRC_B32'), scale_old))
                                tla.simd.vsts(running_max[row], maximum, dist='ONEPT_B32')
                                tla.simd.vsts(running_sum[row], den, dist='ONEPT_B32')
                                tla.simd.vsts(rescale[row], scale_old, dist='ONEPT_B32')
                                packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p0, tl.bfloat16),
                                                                  tla.simd.vtruncf(p1, tl.bfloat16))
                                tla.simd.vsstb(packed, prob_padded_ub[0, row * 16], (VM + 1) * 16)
                    else:
                        with tla.scope.simd():
                            negative = tla.simd.vfull(float('-inf'), tl.float32, 64)
                            zero = tla.simd.vfull(0, tl.int32, 64)
                            limit = tla.simd.vfull(L, tl.int32, 64)
                            i0 = tla.simd.vlds(indices[0])
                            i1 = tla.simd.vlds(indices[64])
                            i2 = tla.simd.vlds(indices[128])
                            i3 = tla.simd.vlds(indices[192])
                            for row in tl.range(VM):
                                x0 = ops.vmuls(tla.simd.vlds(scores_ub[row, 0]), 576**(-0.5))
                                x0 = tla.simd.vsel(tla.simd.vcmp(i0, zero, op=tla.simd.CMP.GE), x0, negative)
                                x0 = tla.simd.vsel(tla.simd.vcmp(i0, limit, op=tla.simd.CMP.LT), x0, negative)
                                x1 = ops.vmuls(tla.simd.vlds(scores_ub[row, 64]), 576**(-0.5))
                                x1 = tla.simd.vsel(tla.simd.vcmp(i1, zero, op=tla.simd.CMP.GE), x1, negative)
                                x1 = tla.simd.vsel(tla.simd.vcmp(i1, limit, op=tla.simd.CMP.LT), x1, negative)
                                x2 = ops.vmuls(tla.simd.vlds(scores_ub[row, 128]), 576**(-0.5))
                                x2 = tla.simd.vsel(tla.simd.vcmp(i2, zero, op=tla.simd.CMP.GE), x2, negative)
                                x2 = tla.simd.vsel(tla.simd.vcmp(i2, limit, op=tla.simd.CMP.LT), x2, negative)
                                x3 = ops.vmuls(tla.simd.vlds(scores_ub[row, 192]), 576**(-0.5))
                                x3 = tla.simd.vsel(tla.simd.vcmp(i3, zero, op=tla.simd.CMP.GE), x3, negative)
                                x3 = tla.simd.vsel(tla.simd.vcmp(i3, limit, op=tla.simd.CMP.LT), x3, negative)
                                old_m = tla.simd.vlds(running_max[row], dist='BRC_B32')
                                maximum = tla.simd.vmax(
                                    tla.simd.vreduce_max(
                                        tla.simd.vmax(tla.simd.vmax(x0, x1), tla.simd.vmax(x2, x3))), old_m)
                                scale_old = ops.vexpdif(old_m, maximum)
                                p0 = ops.vexpdif(x0, maximum)
                                p1 = ops.vexpdif(x1, maximum)
                                p2 = ops.vexpdif(x2, maximum)
                                p3 = ops.vexpdif(x3, maximum)
                                den = tla.simd.vbroadcast(
                                    tla.simd.vreduce_sum(
                                        tla.simd.vadd(tla.simd.vadd(p0, p1), tla.simd.vadd(p2, p3))))
                                den = tla.simd.vadd(
                                    den,
                                    tla.simd.vmul(tla.simd.vlds(running_sum[row], dist='BRC_B32'), scale_old))
                                tla.simd.vsts(running_max[row], maximum, dist='ONEPT_B32')
                                tla.simd.vsts(running_sum[row], den, dist='ONEPT_B32')
                                tla.simd.vsts(rescale[row], scale_old, dist='ONEPT_B32')
                                packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p0, tl.bfloat16),
                                                                  tla.simd.vtruncf(p1, tl.bfloat16))
                                tla.simd.vsstb(packed, prob_padded_ub[0, row * 16], (VM + 1) * 16)
                                packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p2, tl.bfloat16),
                                                                  tla.simd.vtruncf(p3, tl.bfloat16))
                                tla.simd.vsstb(packed, prob_padded_ub[8, row * 16], (VM + 1) * 16)
                    tla.copy_ub_to_l1(tla.view(prob_padded_ub, (BN // 16, VM * 16)),
                                      tla.view(prob_flat_l1, (BN // 16, VM * 16), offsets=(0, sub * VM * 16)))
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 2)
                with tla.scope.cube():
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2)
                for fragment in tl.range(512 // VP):
                    with tla.scope.cube():
                        ops.matmul(prob_l1,
                                   tla.view(current, (VP // 16, BN // 16, 16, 16),
                                            offsets=(fragment * (VP // 16), 0, 0, 0)),
                                   pv_l0c,
                                   clear_accum=True)
                        if SHARE_SCORE:
                            if fragment > 0:
                                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                        else:
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                        tla.copy_l0c_to_ub(
                            pv_l0c, pv_ub,
                            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                                    sub_block_idx=None))
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 3)
                    with tla.scope.vector():
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 3)
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                scale_old = tla.simd.vlds(rescale[row], dist='BRC_B32')
                                for col in tl.static_range(VP // 64):
                                    value = tla.simd.vadd(
                                        tla.simd.vlds(pv_ub[row, col * 64]),
                                        tla.simd.vmul(
                                            tla.simd.vlds(numerator_ub[row, fragment * VP + col * 64]),
                                            scale_old))
                                    tla.simd.vsts(numerator_ub[row, fragment * VP + col * 64], value)
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        # 回收最后一个 PV credit，再输出；split>1 保留未归一化 FP32 分子。
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
        with tla.scope.vector():
            sub = tla.sub_block_id()
            if SPLIT == 1:
                for fragment in tl.range(512 // VP):
                    with tla.scope.simd():
                        for row in tl.range(VM):
                            den = tla.simd.vlds(running_sum[row], dist='BRC_B32')
                            for col in tl.static_range(VP // 128):
                                b0 = tla.simd.vtruncf(
                                    tla.simd.vdiv(tla.simd.vlds(numerator_ub[row, fragment * VP + col * 128]),
                                                  den), tl.bfloat16)
                                b1 = tla.simd.vtruncf(
                                    tla.simd.vdiv(
                                        tla.simd.vlds(numerator_ub[row, fragment * VP + col * 128 + 64]),
                                        den), tl.bfloat16)
                                packed, unused = tla.simd.vdintlv(b0, b1)
                                tla.simd.vsts(output_bf16[row, col * 128], packed)
                    tla.store_ub_to_gm(
                        output_bf16,
                        tla.view(O + (output_token * H + head + sub * VM) * 512 + fragment * VP, (VM, VP),
                                 strides=(512, 1)))
            else:
                tla.store_ub_to_gm(numerator_ub,
                                   tla.view(O + (output_token * H + head + sub * VM) * 512, (VM, 512)))
            tla.store_ub_to_gm(tla.view(running_max, (VM, )),
                               tla.view(MAX + output_token * H + head + sub * VM, (VM, )))
            tla.store_ub_to_gm(tla.view(running_sum, (VM, )),
                               tla.view(SUM + output_token * H + head + sub * VM, (VM, )))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 7)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 8)
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 7)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 8)

# ---- api ----
import torch
import torch_npu  # 注册 PyTorch 的 NPU 设备与 runtime。


# 与归档中已测实现一致；没有借重构增加近似或改变编译优化策略。
LAUNCH_OPTIONS = {
    "sanitize_overflow": False,
    "enable_auto_bind_sub_block": False,
    "enable_preload": False,
}


def _check_inputs(q, kv, q_rope, k_rope, indices):
    # Tensor 元数据仍逐次校验，但每个属性只读取所需次数。
    q_shape, kv_shape, ids_shape = q.shape, kv.shape, indices.shape
    if len(q_shape) != 4 or len(kv_shape) != 4 or len(ids_shape) != 4:
        raise ValueError("Q、KV 和 indices 必须是四维张量")
    batch, queries, heads, width = q_shape
    length, topk = kv_shape[1], ids_shape[-1]
    if (width != 512 or kv_shape != (batch, length, 1, 512) or q_rope.shape != (batch, queries, heads, 64)
            or k_rope.shape != (batch, length, 1, 64) or ids_shape != (batch, queries, 1, topk)
            or min(batch, queries, heads, length, topk) <= 0):
        raise ValueError("需要 Q[B,SQ,H,512]、KV[B,L,1,512]、RoPE64、indices[B,SQ,1,K]")
    device = q.device
    if (device.type != "npu" or kv.device != device or q_rope.device != device or k_rope.device != device
            or indices.device != device or not q.is_contiguous() or not kv.is_contiguous()
            or not q_rope.is_contiguous() or not k_rope.is_contiguous() or not indices.is_contiguous()):
        raise ValueError("所有输入必须位于同一 NPU，且连续存放")
    bf16 = torch.bfloat16
    if (q.dtype != bf16 or kv.dtype != bf16 or q_rope.dtype != bf16 or k_rope.dtype != bf16
            or indices.dtype != torch.int32):
        raise ValueError("Q/KV/RoPE 必须为 BF16；indices 必须为 int32")
    return (batch, queries, heads, length, topk), device


def _check_tuning(config, shape, cores):
    if not isinstance(config, Tuning):
        raise TypeError("config 必须是 dsa.Tuning")
    _, _, heads, _, topk = shape
    if type(cores) is not int or cores <= 0:
        raise ValueError("cores 必须是正整数")
    if any(type(v) is not int or v <= 0 for v in (config.bm, config.bn, config.split, config.pv_cols)):
        raise ValueError("BM/BN/split/PV 列数必须是正整数")
    if config.bm not in (16, 32, 64, 128) or heads % config.bm:
        raise ValueError("BM 必须为 16/32/64/128，且整除 head 数")
    if config.direct_nz:
        if config.bn != 64 or topk % 64 or config.split > topk // 64:
            raise ValueError("直接 NZ 路径需要 BN64、完整 tile 和非空 split")
    else:
        if config.bn not in (128, 256) or topk % (2 * config.bn * config.split):
            raise ValueError("GM 工作区路径需要 BN128/256，每个 split 含偶数个 tile")
        if config.pv_cols not in (128, 256, 512):
            raise ValueError("PV 列数必须为 128/256/512")
        if config.share_score and config.pv_cols < config.bn:
            raise ValueError("共享 UB 必须能装下 score tile")


def _validated_defaults():
    """固定形状与不可变 Tuning 只需校验一次，不缓存任何张量或输入值。"""
    defaults = {}
    for name, shape in CASE_SHAPES.items():
        config = SELECTED[name]
        _check_tuning(config, shape, 28)
        defaults[shape] = config
    return defaults


_DEFAULT_CONFIGS = _validated_defaults()


def forward(q, kv, q_rope, k_rope, indices, *, config=None, cores=28, return_kernels=False):
    """计算给定 indices 的稀疏注意力，返回 output、softmax_max、softmax_sum。

    output 为 BF16 [B,SQ,H,512]；两个统计量为 FP32 [B,1,SQ,H]。
    缩放系数固定为 1/sqrt(512+64)。重复索引保留重复权重，不会去重。
    每个 query 至少需要一个有效索引；无效项按 [0,L) 掩码处理。
    这不是 indexer/TopK，也不含 paged/causal 模式；不在这里扫描输入数据。

    return_kernels=True 时返回 (outputs, compiled_kernels)，便于核对 IR、
    二进制及完整的主核+merge 计时。默认调用方式只返回三个结果张量。
    """
    shape, device = _check_inputs(q, kv, q_rope, k_rope, indices)
    if config is None:
        try:
            config = _DEFAULT_CONFIGS[shape]
        except KeyError as error:
            raise ValueError(f"没有验证过的 DSA 配置: {shape}") from error
        # 配置已在导入时检查；调用者可变的 core 数仍须逐次检查。
        if type(cores) is not int or cores <= 0:
            raise ValueError("cores 必须是正整数")
    else:
        _check_tuning(config, shape, cores)
    batch, queries, heads, length, topk = shape
    split = config.split

    # 所有 gather 和分片合并均属于本次调用，没有跨调用缓存输入或结果。
    workspace = None
    if not config.direct_nz:
        workspace = torch.empty((cores, 2, config.bn, 576), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q)
    maximum = torch.empty((batch, 1, queries, heads), device=device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    if split == 1:
        numerator, partial_max, partial_sum = output, maximum, denominator
    else:
        numerator = torch.empty((batch, queries, split, heads, 512), device=device, dtype=torch.float32)
        partial_max = torch.empty((batch, queries, split, heads), device=device, dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)

    if config.direct_nz:
        main = ragged_attention[(cores, )](q, kv, q_rope, k_rope, indices, numerator, partial_max,
                                           partial_sum, batch, queries, length, heads, topk, config.bm, cores,
                                           config.bn, split, **LAUNCH_OPTIONS)
    else:
        main = workspace_attention[(cores, )](q, kv, q_rope, k_rope, indices, workspace, numerator,
                                              partial_max, partial_sum, batch, queries, length, heads, topk,
                                              config.bm, cores, config.bn, split, config.pv_cols,
                                              config.share_score, **LAUNCH_OPTIONS)
    if split > 1:
        # 这一步是运算的一部分；benchmark 必须包含，不能只量主核。
        merged = merge_partials[(cores * 2, )](numerator, partial_max, partial_sum, output, maximum,
                                               denominator, batch * queries, heads, split, cores * 2,
                                               **LAUNCH_OPTIONS)
    result = (output, maximum, denominator)
    if return_kernels:
        return result, (main, merged) if split > 1 else (main, )
    return result
