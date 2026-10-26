# TLA Cube 适配细化方案（v1）

> 目标：在 triton-ascend 的 TLA（`triton.language.extra.tla`）前端新增 cube 计算接口，
> 用纯 TLA 接口重写 tilelang ascend 的 gemm（cube）与 gemm+vector（cube+vector）两个算子。
>
> 范围约束：
> - 修改点全部在 TLA 模块内（`third_party/ascend/language/tla/`，`python/triton/language/extra/tla/` 是其软链，只需改一份）；
> - 不改 `cann/extension/builder.py`、不改通用 `ascendnpu_ir_builder` 绑定；
> - 排除：mmad_mx/scale、T.Pipelined 流水、T.Kernel/cthread/T.serialize 多核、L1→L0A/L0B 显式搬运、transpose_A。

## 1. tilelang → HIVM op 对照

| tilelang（ascend） | HIVM op | TLA v1 接口 |
|---|---|---|
| `T.copy(GM→L1)` | `hivm.hir.nd2nz`（ND→NZ，2D 逻辑形状就地转换） | `tla.copy(gm_blk, l1_buf)` |
| `T.gemm(A_L1, B_L1, C_L0C)` | `hivm.hir.mmadL1`（A/B 在 L1、C 在 L0C，DPS 就地累加） | `tla.mmad(a, b, acc, transpose_b, clear_accum)` |
| `T.copy(L0C→GM)` | `hivm.hir.fixpipe`（dma_mode=NZ2ND：NZ 转 ND 写回行主序 GM） | `tla.copy(l0c_buf, gm_blk)` |
| `T.copy(L0C→UB)`（cube+vector 中间） | `hivm.hir.fixpipe`（2D dst→NZ2ND 出 ND 到 UB；4D dst→NZ2NZ 保 NZ） | `tla.copy(l0c_buf, ub_buf)` |
| `T.copy(UB→GM)` | `hivm.hir.store`（MTE3） | `tla.copy(ub_buf, gm_blk)`（已有） |

三层 lower 链：Python `T.*` → TIR `tl.ascend_*` → CANN 编译。TLA 直接生成 HIVM op，跳过中间层。

## 2. 关键实现事实（核查结论）

1. **单一副本**：`python/triton/language/extra/tla` 是 `third_party/ascend/language/tla` 的软链，所有修改只需落一份。
2. **地址空间字符串映射缺口（必须先处理）**：前端 space 是 `gm/ub/l1/l0a/l0b/l0c`，
   但 `hivm::symbolizeAddressSpace` 只认 `gm/cbuf/ca/cb/cc/ub/...`（`HIVMAttrs.td` 的 `I32EnumAttrCase` 助记符）。
   `create_tla_buffer_type` 必须加映射：`l1→cbuf, l0a→ca, l0b→cb, l0c→cc`，否则 alloc l1/l0c 直接抛异常。
3. **2D 逻辑形状即可**：`nd2nz` 的 src/dst 就是 2D 逻辑形状 memref（`test/Dialect/HIVM/IR/dma-ops.mlir`），
   NZ 转换在搬运中完成；`mmadL1` 接受 2D memref（`test/Dialect/MemRef/test_copy_removal.mlir`：
   `ins(A: memref<29x128xf16>, B: memref<128x256xf16>, %true, %c29, %c128, %c256_index, C: memref<1x256xf32>) outs(C)`，
   无 transpose attr），`MmadL1Op::verify` 只查 sync_related_args，无形状/空间约束。
4. **fixpipe 必须新建 memref 版本**：基类 `create_fixpipe`（ascend_ir.cc:1288）强制 src 为
   `RankedTensorType`；TLA buffer 是 memref，需在 `tla_builder.cc` 按 `FixpipeOp` 自定义 builder
   （`result, src, dst, dma_mode, dual_dst_mode, sub_block_idx, pre_quant, pre_relu, channel_split`）新建绑定。
5. **fixpipe dma_mode 规则**：官方 HIVM 流水（`InsertLoadStoreForMixCV/Utils.cpp:640`
   `getInsertedFixpipeDmaMode`）按 dst rank 选模式——2D ND 形状 dst → `NZ2ND`（去 NZ 成 ND），
   否则 → `NZ2NZ(normal)`。`NZ2DN` 仅 Ascend950 特定 layout 场景使用，v1 不用。
6. **scope.cube 基础设施已就绪**：dispatch 已把 `cube → handle_scope_with`，已生成
   `#hivm.tcore_type<CUBE>`；唯一缺口是 `_vf_scope_attrs` 硬编码 `hivm.func_core_type<AIV>`，
   cube 需 `<AIC>`（该符号合法，见 `test/Target/LLVMIR/hivm.mlir`）。
7. **tilelang 的 transpose 暴露方式**（`tilelang/ascend/language/gemm_op.py`）：
   `gemm(A, B, C, transpose_A=False, transpose_B=False, ..., clear_accum=False, ...)`；
   但 L1 路径 codegen（`codegen_ascend.cc` 的 `tl.ascend_gemm_l1`）只消费 `transpose_B`，
   `transpose_A` 折叠到 GM→L1 copy 的 transpose 参数中。→ TLA v1 只暴露 `transpose_b`，不暴露 transpose_a。

## 3. 逐文件实现清单

### 3.1 `third_party/ascend/language/tla/cpp/tla_builder.cc`（C++ 绑定）

**a) `create_tla_buffer_type` 加 space 名映射**：

```cpp
auto memorySpace = hivm::symbolizeAddressSpace(space);
if (!memorySpace) {
  // TLA frontend names -> HIVM #hivm.address_space mnemonics.
  static const llvm::StringMap<llvm::StringRef> kTlaToHivm = {
      {"l1", "cbuf"}, {"l0a", "ca"}, {"l0b", "cb"}, {"l0c", "cc"}};
  auto it = kTlaToHivm.find(space);
  if (it != kTlaToHivm.end())
    memorySpace = hivm::symbolizeAddressSpace(it->second);
}
if (!memorySpace)
  throw std::invalid_argument("unknown TLA address space: " + space);
```

**b) 新增 3 个绑定**（照 `create_hivm_load/store` 的 memref 模式）：

```cpp
// GM(ND) -> L1(NZ)，dst_continuous 置上（库函数调用路径要求，ir 阶段可验证）。
.def("create_hivm_nd2nz",
     [](TLABuilder &self, Value src, Value dst) {
       self.create<hivm::ND2NZOp>(TypeRange{}, src, dst,
                                  self.getBuilder().getUnitAttr());
     }, py::arg("src"), py::arg("dst"))

// L0C -> GM/UB。基类 create_fixpipe 强制 src 为 RankedTensorType，
// TLA buffer 是 memref，这里走 FixpipeOp 自定义 builder 的 memref 形态。
.def("create_hivm_fixpipe",
     [](TLABuilder &self, Value src, Value dst, std::string &dma_mode) {
       auto &builder = self.getBuilder();
       auto *ctx = builder.getContext();
       auto dmaEnum = hivm::symbolizeFixpipeDMAMode(dma_mode);  // "nz2nd"/"normal"
       if (!dmaEnum)
         throw std::invalid_argument("unknown fixpipe dma_mode: " + dma_mode);
       auto dmaAttr = hivm::FixpipeDMAModeAttr::get(ctx, *dmaEnum);
       auto dualAttr = hivm::FixpipeDualDstModeAttr::get(
           ctx, hivm::FixpipeDualDstMode::NO_DUAL);
       auto quantAttr = hivm::FixpipePreQuantModeAttr::get(
           ctx, hivm::FixpipePreQuantMode::NO_QUANT);
       auto reluAttr = hivm::FixpipePreReluModeAttr::get(
           ctx, hivm::FixpipePreReluMode::NO_RELU);
       self.create<hivm::FixpipeOp>(
           TypeRange{}, src, dst, dmaAttr, dualAttr, nullptr, quantAttr,
           reluAttr, builder.getBoolAttr(false));
     }, py::arg("src"), py::arg("dst"), py::arg("dma_mode"))

// C = A x B（+C），A=(M,K) B=(K,N) 均在 L1，C 在 L0C(f32/i32)。
.def("create_hivm_mmadl1",
     [](TLABuilder &self, Value a, Value b, Value c, bool init_condition,
        int64_t real_m, int64_t real_k, int64_t real_n, bool transpose_b) {
       auto &builder = self.getBuilder();
       auto loc = self.getLastLoc();
       auto init = builder.create<arith::ConstantOp>(
           loc, builder.getI1Type(), builder.getBoolAttr(init_condition));
       auto m = builder.create<arith::ConstantIndexOp>(loc, real_m);
       auto k = builder.create<arith::ConstantIndexOp>(loc, real_k);
       auto n = builder.create<arith::ConstantIndexOp>(loc, real_n);
       auto bTrans = transpose_b ? builder.getUnitAttr() : UnitAttr{};
       self.create<hivm::MmadL1Op>(TypeRange{}, a, b, init, m, k, n, c,
                                   /*per_channel_bias=*/{}, bTrans);
     }, py::arg("a"), py::arg("b"), py::arg("c"),
        py::arg("init_condition"), py::arg("real_m"),
        py::arg("real_k"), py::arg("real_n"), py::arg("transpose_b"))
```

要点：
- `MmadL1Op` 自定义 builder 参数序：`result_tensors, a, b, init_condition, real_m, real_k, real_n, c, per_channel_bias{}, a_transpose{}, b_transpose{}, enable_HF32{}, enable_I4{}`；
  `TypeRange{}` + 8 个值参即得 `outs(%c)` 无结果的 DPS 形态。
- `init_condition`：`clear_accum=True` → true（使用前清 L0C）；`clear_accum=False` → false（累加进现有 L0C 内容，K 循环首轮须为 true）。

### 3.2 `third_party/ascend/language/tla/builder.py`

`ascend_methods` 追加 `'create_hivm_nd2nz', 'create_hivm_fixpipe', 'create_hivm_mmadl1'`。

### 3.3 `third_party/ascend/language/tla/copy.py`（扩展分派表）

```python
def _fixpipe_dma_mode(dst):
    # 对齐 InsertLoadStoreForMixCV::getInsertedFixpipeDmaMode：
    # 2D ND 形状 dst -> NZ2ND（去 NZ 成 ND）；4D fractal dst -> NZ2NZ（保 NZ）。
    return "nz2nd" if len(dst.shape) == 2 else "normal"


@builtin
def copy(src, dst, _semantic=None):
    if not isinstance(src, buffer) or not isinstance(dst, buffer):
        raise TypeError("tla.copy expects two tla.buffer operands; pointer/tensor arguments are unsupported")
    if src.shape != dst.shape or src.dtype != dst.dtype:
        raise ValueError("DMA source and destination must have identical shape and element type")
    dispatch = {
        ("gm", "ub"):  lambda: _semantic.builder.create_hivm_load(src.handle, dst.handle),
        ("ub", "gm"):  lambda: _semantic.builder.create_hivm_store(src.handle, dst.handle),
        ("gm", "l1"):  lambda: _semantic.builder.create_hivm_nd2nz(src.handle, dst.handle),
        ("l0c", "gm"): lambda: _semantic.builder.create_hivm_fixpipe(
            src.handle, dst.handle, _fixpipe_dma_mode(dst)),
        ("l0c", "ub"): lambda: _semantic.builder.create_hivm_fixpipe(
            src.handle, dst.handle, _fixpipe_dma_mode(dst)),
    }
    key = (src.space, dst.space)
    if key not in dispatch:
        raise ValueError(f"tla.copy unsupported space pair {key}")
    dispatch[key]()
```

### 3.4 `third_party/ascend/language/tla/cube.py`（新增）

```python
"""Cube-core primitives for TLA (hivm.hir.mmadL1)."""
from triton.language.core import builtin, _unwrap_if_constexpr
from .core import buffer

__all__ = ["mmad"]


@builtin
def mmad(a, b, acc, transpose_b=False, clear_accum=False, _semantic=None):
    if not (isinstance(a, buffer) and isinstance(b, buffer) and isinstance(acc, buffer)):
        raise TypeError("tla.mmad expects tla.buffer operands for a, b, acc")
    if a.space != "l1" or b.space != "l1":
        raise ValueError("tla.mmad a/b must be L1 buffers (fill via tla.copy(gm->l1))")
    if acc.space != "l0c":
        raise ValueError("tla.mmad acc must be an L0C buffer")
    if a.dtype != b.dtype:
        raise ValueError("tla.mmad a and b must have the same element type")
    if len(a.shape) != 2 or len(b.shape) != 2 or len(acc.shape) != 2:
        raise ValueError("tla.mmad expects 2-D buffers (M,K)/(K,N)/(M,N)")
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or acc.shape != (m, n):
        raise ValueError("tla.mmad shape mismatch: a=(M,K), b=(K,N), acc=(M,N)")
    _semantic.builder.create_hivm_mmadl1(
        a.handle, b.handle, acc.handle,
        bool(_unwrap_if_constexpr(clear_accum)),
        m, k, n,
        bool(_unwrap_if_constexpr(transpose_b)))
```

### 3.5 `third_party/ascend/language/tla/__init__.py`

`from . import cube` + `__all__` 追加 `"cube"`。

### 3.6 `third_party/ascend/language/tla/code_generator.py`

`_vf_scope_attrs` 按 core_mode 分支：

```python
func_core_type_str = ("#hivm.func_core_type<AIC>" if core_mode == "cube"
                      else "#hivm.func_core_type<AIV>")
attrs = {
    "no_inline": builder.get_unit_attr(),
    "outline": builder.get_unit_attr(),
    "tcore_type": builder.parse_attr(tcore_type_str),       # CUBE / VECTOR
    "hivm.func_core_type": builder.parse_attr(func_core_type_str),
    "hivm.vector_function": builder.get_unit_attr(),
    "tla.memory_effect": builder.get_array_attr(...),
}
if core_mode != "cube":
    attrs["vector_mode"] = builder.get_string_attr("vec")
```

`handle_scope_with` 中 `core_mode == "cube"` 时跳过 simd 的 `_VF_EFFECTS` 效应推断（cube op 的
读写效应由 op 自身的 `MemoryEffectsOpInterface` 承担，`tla.memory_effect` 传空数组）。

## 4. 开放决策（已确认）

| # | 决策 | 结论 |
|---|------|------|
| 1 | scope.cube 形态 | 真实 scope（outline 为 AIC func），不退化 no-op；仅需 code_generator.py 一处 AIC 分支 |
| 2 | fixpipe dma_mode | 由 dst.space + dst rank 自动推断（`_fixpipe_dma_mode`），不暴露显式参数 |
| 3 | L1 buffer 布局 | 2D 逻辑形状，nd2nz 搬运时自动转 NZ；无需 tla.view/reinterpret 表达布局切换 |
| 4 | L0C 复用 | v1 不做 multi_buffer，L0C 单缓冲（versions=1），K 循环串行无流水重叠；已知限制 |
| 5 | mmad transpose | 只暴露 `transpose_b`（bool → b_transpose attr），不暴露 transpose_a（对齐 tilelang L1 路径） |

## 5. 可适配算子示例（tla_ut 3-in-1 标准）

两个用例完整脚本（kernel + constexprs + Triton 参考 + ir/compile/run 三阶段），结构与
`tla_ut/vector/cast_back.py` 一致。**注意**：用例中的 IR 断言数字（nd2nz/mmadL1/fixpipe 计数、
`check_lowered_ir` 的 buffers 数）为首跑目标值，需在首次 `TLA_TEST_STAGE=ir` 跑通后锁定。

### 5.1 cube 单算子：`cube/gemm_basic.py`

语义：`C = A @ B`（A/B fp16 → L0C f32 累加，NZ2DN 写回行主序 GM）。

```python
"""TLA 3-in-1 test: gemm_basic (tilelang ascend gemm 的 TLA 直译).

Semantics: C[M, N] = A[M, K] @ B[K, N]  (fp16 输入, f32 在 L0C 累加)

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline-compile the HIVM IR handed to npuir and assert
            (2 nd2nz / 1 mmadL1 / 1 fixpipe, cbuf/cc alloc, AIC scope)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (GEMM_BASIC OK; standard Triton
            reference gemm_triton, same semantics/shape -> TRITON_GEMM_BASIC OK)

v1 范围: 单 block (M=N=K=64), clear_accum=True 一次结算; 无 K 循环流水/多核。
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: M, N, K (K 须为 16 的倍数)
SHAPE = dict(M=64, N=64, K=64)

# offline compile signature (pointer entry takes pointer args).
SIGNATURE = {"A_ptr": "*fp16", "B_ptr": "*fp16", "C_ptr": "*fp32"}


@triton.jit
def gemm_basic(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    tl.static_assert(K % 16 == 0 and M % 16 == 0 and N % 16 == 0,
                     "M/N/K must be multiples of 16 (cube tile)")
    a_l1 = tla.alloc_buffer((M, K), tl.float16, space="l1")
    b_l1 = tla.alloc_buffer((K, N), tl.float16, space="l1")
    c_l0c = tla.alloc_buffer((M, N), tl.float32, space="l0c")
    a_blk = tla.view(A_ptr, (M, K), strides=(K, 1))
    b_blk = tla.view(B_ptr, (K, N), strides=(N, 1))
    c_blk = tla.view(C_ptr, (M, N), strides=(N, 1))
    with tla.scope.cube():
        tla.copy(a_blk, a_l1)               # hivm.hir.nd2nz
        tla.copy(b_blk, b_l1)               # hivm.hir.nd2nz
        tla.mmad(a_l1, b_l1, c_l0c, clear_accum=True)  # hivm.hir.mmadL1
        tla.copy(c_l0c, c_blk)              # hivm.hir.fixpipe (2D dst -> nz2nd)


def constexprs():
    s = SHAPE
    return {"M": s["M"], "N": s["N"], "K": s["K"]}


# ---- standard Triton reference (same semantics/shape) ----


@triton.jit
def gemm_triton(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    a = tl.load(A_ptr + rows[:, None] * K + cols[None, :])   # (M, K)
    b = tl.load(B_ptr + cols[:, None] * K + rows[None, :])   # (N, K) -> 转置取 B(K,N)
    c = tl.sum(a[:, :, None] * b[None, :, :], axis=1)        # (M, N) f32 累加
    tl.store(C_ptr + rows[:, None] * N + cols[None, :], c)


def test_ir(output_dir=None):
    module, ctx = compile_offline(gemm_basic, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 3, "tla.view(ptr) x3"
    assert text.count("memref.reinterpret_cast") == 3, "tla.view(ptr) x3"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=0, stores=0, buffers=3,
                     extra=("#hivm.address_space<cbuf>", "#hivm.address_space<cc>",
                            "#hivm.func_core_type<AIC>"))
    assert text.count("hivm.hir.nd2nz") == 2, "GM->L1 x2"
    assert text.count("hivm.hir.mmadL1") == 1, "one gemm"
    assert text.count("hivm.hir.fixpipe") == 1, "L0C->GM"
    print("IR PASS: gemm_basic HIVM structure ok (2 nd2nz / 1 mmadL1 / 1 fixpipe)")


def _launch():
    s = SHAPE
    A = torch.randn(s["M"], s["K"], dtype=torch.float16, device="npu")
    B = torch.randn(s["K"], s["N"], dtype=torch.float16, device="npu")
    C = torch.full((s["M"], s["N"]), 7.0, dtype=torch.float32, device="npu")
    gemm_basic[(1,)](A, B, C, **constexprs())
    torch.npu.synchronize()
    return C, (A.float() @ B.float())


def _launch_triton():
    s = SHAPE
    A = torch.randn(s["M"], s["K"], dtype=torch.float16, device="npu")
    B = torch.randn(s["K"], s["N"], dtype=torch.float16, device="npu")
    C = torch.full((s["M"], s["N"]), 7.0, dtype=torch.float32, device="npu")
    gemm_triton[(1,)](A, B, C, **constexprs())
    torch.npu.synchronize()
    return C, (A.float() @ B.float())


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: gemm_basic generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    C, ref = _launch()
    oc, rc = C.cpu(), ref.cpu()
    print("C[:4,:4]  :", oc[:4, :4].tolist())
    print("ref[:4,:4]:", rc[:4, :4].tolist())
    ok = torch.allclose(oc, rc, atol=0.05)
    print("GEMM_BASIC", "OK" if ok else "MISMATCH", "M", SHAPE["M"], "N", SHAPE["N"], "K", SHAPE["K"])
    assert ok, "gemm_basic e2e mismatch"
    # standard Triton reference.
    Ct, rt = _launch_triton()
    oc_t, rc_t = Ct.cpu(), rt.cpu()
    bad_t = ((oc_t - rc_t).abs() > 0.05).nonzero().flatten()
    print("TRITON_GEMM_BASIC", "OK" if bad_t.numel() == 0 else "MISMATCH", "num_mismatch:", bad_t.numel())
    assert bad_t.numel() == 0, "standard triton gemm_triton mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: gemm_basic", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
```

### 5.2 cube+vector 混合：`cube/gemm_bias.py`

语义：`C = A @ B + bias`。cube 段计算到 L0C，fixpipe（2D UB dst → NZ2ND）把结果以 ND
行主序落到 UB，vector 段用 simd 寄存器逐 64-lane 加常量，再 MTE3 写回 GM。

```python
"""TLA 3-in-1 test: gemm_bias (tilelang ascend gemm + vector elementwise 直译).

Semantics: C[M, N] = A[M, K] @ B[K, N] + bias  (vector 段逐 64-lane 加常量)

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline-compile the HIVM IR handed to npuir and assert
            (2 nd2nz / 1 mmadL1 / 1 fixpipe, 1 store, cbuf/cc/ub alloc, AIC+AIV scope)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (GEMM_BIAS OK; standard Triton
            reference gemm_bias_triton, same semantics/shape ->
            TRITON_GEMM_BIAS OK)

v1 范围: 单 block, cube 段与 vector 段各自单 scope; 无流水/多核。
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: M, N, K (K 须为 16 的倍数; M*N 须为 64 的倍数)
SHAPE = dict(M=64, N=64, K=64, BIAS=1.0)

# offline compile signature (pointer entry takes pointer args).
SIGNATURE = {"A_ptr": "*fp16", "B_ptr": "*fp16", "C_ptr": "*fp32"}


@triton.jit
def gemm_bias(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BIAS: tl.constexpr):
    tl.static_assert(K % 16 == 0 and M % 16 == 0 and N % 16 == 0,
                     "M/N/K must be multiples of 16 (cube tile)")
    tl.static_assert((M * N) % 64 == 0, "M*N must be a multiple of 64 (vector lanes)")
    a_l1 = tla.alloc_buffer((M, K), tl.float16, space="l1")
    b_l1 = tla.alloc_buffer((K, N), tl.float16, space="l1")
    c_l0c = tla.alloc_buffer((M, N), tl.float32, space="l0c")
    c_ub = tla.alloc_buffer((M, N), tl.float32, space="ub")
    a_blk = tla.view(A_ptr, (M, K), strides=(K, 1))
    b_blk = tla.view(B_ptr, (K, N), strides=(N, 1))
    c_blk = tla.view(C_ptr, (M, N), strides=(N, 1))
    with tla.scope.cube():
        tla.copy(a_blk, a_l1)               # hivm.hir.nd2nz
        tla.copy(b_blk, b_l1)               # hivm.hir.nd2nz
        tla.mmad(a_l1, b_l1, c_l0c, clear_accum=True)  # hivm.hir.mmadL1
        tla.copy(c_l0c, c_ub)               # hivm.hir.fixpipe (2D ub dst -> nz2nd, ND 落 UB)
    with tla.scope.simd():
        bias = tla.simd.vfull(BIAS, tl.float32, 64)
        for i in tl.static_range((M * N) // 64):
            row = i // (N // 64)
            col = (i % (N // 64)) * 64
            v = tla.simd.vlds(c_ub[row, col])          # NORM, ND 行主序直读
            tla.simd.vsts(c_ub[row, col], tla.simd.vadd(v, bias))
    tla.copy(c_ub, c_blk)                   # hivm.hir.store (MTE3 UB->GM)


def constexprs():
    s = SHAPE
    return {"M": s["M"], "N": s["N"], "K": s["K"], "BIAS": s["BIAS"]}


# ---- standard Triton reference (same semantics/shape) ----


@triton.jit
def gemm_bias_triton(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BIAS: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    a = tl.load(A_ptr + rows[:, None] * K + cols[None, :])
    b = tl.load(B_ptr + cols[:, None] * K + rows[None, :])
    c = tl.sum(a[:, :, None] * b[None, :, :], axis=1) + BIAS
    tl.store(C_ptr + rows[:, None] * N + cols[None, :], c)


def test_ir(output_dir=None):
    module, ctx = compile_offline(gemm_bias, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 3, "tla.view(ptr) x3"
    assert text.count("memref.reinterpret_cast") == 3, "tla.view(ptr) x3"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=0, stores=1, buffers=4,
                     extra=("#hivm.address_space<cbuf>", "#hivm.address_space<cc>",
                            "#hivm.func_core_type<AIC>"))
    assert text.count("hivm.hir.nd2nz") == 2, "GM->L1 x2"
    assert text.count("hivm.hir.mmadL1") == 1, "one gemm"
    assert text.count("hivm.hir.fixpipe") == 1, "L0C->UB"
    print("IR PASS: gemm_bias HIVM structure ok (2 nd2nz / 1 mmadL1 / 1 fixpipe / 1 store)")


def _launch():
    s = SHAPE
    A = torch.randn(s["M"], s["K"], dtype=torch.float16, device="npu")
    B = torch.randn(s["K"], s["N"], dtype=torch.float16, device="npu")
    C = torch.full((s["M"], s["N"]), 7.0, dtype=torch.float32, device="npu")
    gemm_bias[(1,)](A, B, C, **constexprs())
    torch.npu.synchronize()
    return C, (A.float() @ B.float() + s["BIAS"])


def _launch_triton():
    s = SHAPE
    A = torch.randn(s["M"], s["K"], dtype=torch.float16, device="npu")
    B = torch.randn(s["K"], s["N"], dtype=torch.float16, device="npu")
    C = torch.full((s["M"], s["N"]), 7.0, dtype=torch.float32, device="npu")
    gemm_bias_triton[(1,)](A, B, C, **constexprs())
    torch.npu.synchronize()
    return C, (A.float() @ B.float() + s["BIAS"])


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: gemm_bias generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    C, ref = _launch()
    oc, rc = C.cpu(), ref.cpu()
    print("C[:4,:4]  :", oc[:4, :4].tolist())
    print("ref[:4,:4]:", rc[:4, :4].tolist())
    ok = torch.allclose(oc, rc, atol=0.05)
    print("GEMM_BIAS", "OK" if ok else "MISMATCH", "M", SHAPE["M"], "N", SHAPE["N"], "K", SHAPE["K"])
    assert ok, "gemm_bias e2e mismatch"
    # standard Triton reference.
    Ct, rt = _launch_triton()
    oc_t, rc_t = Ct.cpu(), rt.cpu()
    bad_t = ((oc_t - rc_t).abs() > 0.05).nonzero().flatten()
    print("TRITON_GEMM_BIAS", "OK" if bad_t.numel() == 0 else "MISMATCH", "num_mismatch:", bad_t.numel())
    assert bad_t.numel() == 0, "standard triton gemm_bias_triton mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: gemm_bias", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
```

## 6. 实施顺序与验证

1. C++ 绑定（tla_builder.cc）→ `make dev-install` 重建 wheel
2. Python 侧：builder.py / copy.py / cube.py / __init__.py / code_generator.py
3. 用例落地为 `cube/gemm_basic.py`、`cube/gemm_bias.py`，`TLA_TEST_STAGE=ir` 离线验证 IR 形态（无 NPU）
4. blue2 上 `compile` / `run` 阶段验证精度
5. 验证期间需重点确认：
   - `dst_continuous` 对 nd2nz 库函数化的影响（可切换重测）
   - 首跑后锁定各 IR 断言计数（nd2nz/mmadL1/fixpipe、`check_lowered_ir` 的 buffers 数）
   - L0C 单缓冲在 K 循环（clear_accum=False）中的串行正确性
   - cube scope 被 outline 为 AIC func 后与入口 AIV 的 mix 衔接（InsertLoadStoreForMixCV 行为）

## 7. v1 已知限制（后续版本）

- 无 K 循环流水（L0C/L1 均单缓冲，`clear_accum=False` 复用现有 buffer）
- 无多核切分（T.Kernel/cthread/T.serialize）
- 无 transpose_A（GM→L1 copy 不带 transpose 参数）
- 无 mmad_mx / scale / pre_quant 量化
- fixpipe 无 dual_dst（Ascend950 L0C→UB 双目标）
