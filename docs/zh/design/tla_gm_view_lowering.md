# TLA GM view：由 TLAToLinalg 维护的 descriptor lowering

状态：源码已调整，未构建、未运行测试。本文替代新增 `ascend.tla_pointer_view` 及向通用 TritonToLinalg 注册 TLA 桥接的旧方案。

## 边界

TLA 的前端入口和 `TLAToLinalgPass.cpp` 负责 GM view 的桥接及转换规则。不新增 dialect op，不在通用 `TritonOpConverter.cpp`、`TritonToLinalgPass.cpp` 注册或实现 TLA 专属逻辑。

可以调用已有的通用 patterns；复用实现不等于改变普通 Triton pipeline。通用头文件中的 TLA bridge 声明也已移除。

## 表示与转换

```text
TLA view:
  linear = sum(i64(offset[i]) * stride[i])
  base = tt.addptr(ptr, linear)
  block = tt.make_tensor_ptr(base, shape, strides, offsets=0)
  buffer = builtin.unrealized_conversion_cast(block) {tla.gm_view}
```

shape/strides 为静态正整数。总偏移放在 i64 标量 base 上，避免标准 make_tensor_ptr 维度 offsets 的 i32 限制及负值截断行为。parent shape 与 block shape 相同，不新增实际 load、padding 或边界检查。

builtin cast 只将 block pointer 桥接到 TLA buffer 的 memref 类型，不执行地址运算。目标类型保留静态 shape/strides 和动态 offset。

TLAToLinalg 内注册：

| Pattern | 职责 |
| --- | --- |
| 现有 AddPtrConverter | 使用 BlockDataParser 生成带元素 offset 的 memref descriptor |
| 现有 MakeTensorPtrConverter | 组合源 offset 与 block 信息，生成 reinterpret view |
| TLA 本地 TLABufferBridgeConversion | 消除带 tla.gm_view 标记的 builtin cast；必要时生成合法 memref.cast |
| 现有 PtrToIntConverter | 对确实需要整数地址的用户计算 aligned base + descriptor offset × 元素字节数 |

不再使用 TLA 旧的 addptr 字节地址加法路径，也删除旧的仅提取 aligned base 的 getPointerAddress。这两者与 descriptor 路径混用会丢失已有 offset。

TLA 的 target 将 MakeTensorPtrOp 和带标记的 bridge 判为必须转换。bridge 使用 adaptor 中已转换的 memref，类型不兼容时失败；不会静默保留到 NPU-IR。

bridge pattern 不绑定标量 pointer TypeConverter：block pointer 的 memref 由 MakeTensorPtrConverter 产生，不能套用 `ptr<T> -> memref<?xT>` 的签名规则。

## 正确性约束

对于 `view(K+p, ..., offsets=(r,c), strides=(S,1))`，目标元素地址为：

```text
aligned_base(K) + (source_offset + p + r*S + c + i*S + j) * sizeof(T)
```

source offset 由标准 descriptor lowering 维护。TLA 不再提取整数地址后恢复来源，不做 `pointer_cast(extract_aligned_pointer(view)) -> view` 这类会丢 offset 或改变布局的折叠。

GM 使用默认 memref memory space，与转换后的参数一致；后续推导为显式 GM。UB/L1/L0C 等继续使用显式空间。

静态 offset 可以通过 memref.cast 放宽为动态 offset，实际 descriptor 和地址不变。reinterpret/cast 均可被 ViewLikeOpInterface 追踪，当前 schedule 的参数根约束不需要修改。

指针 bitcast 不再无条件透传 descriptor：只接受转换后 backing element type 相同的情况，包括 ABI 的 i1 -> i8。不同元素类型会改变后续元素偏移的解释，当前 TLA pass 明确拒绝；不静默生成错误地址，也不在本次扩展字节重解释能力。

## 范围与限制

- 不新增 op，不修改普通 Triton 的指针转换算法、NPU-IR 或 launcher。
- TLA kernel ABI、program ID、返回值和已有 load 转换保持原路径。
- 本次未扩展 TLA 对 pointer select/if/int_to_ptr、循环携带指针等形式的支持。通用 adapter 支持某种指针，不代表 TLA 独立 pass 自动支持它。
- schedule 仍要求 GM 参数来源可证明，并检查跨迭代访问独立性；生成 reinterpret 并不自动满足全部流水约束。
- 前端继续要求字节可寻址元素，offset 乘法前扩展为 i64。调用者仍负责有效范围，已经在上游表达式中发生的溢出不由 view 修复。

## 后续验证项（未执行）

零/动态/负偏移、非连续 strides、超过 i32 的有效偏移、DSA 的 576 宽 view，以及 view 后 subview/reshape/DMA。检查 source offset 只累加一次；最终 ttadapter 无标记 bridge；普通参数 view 不引入 pointer_cast；ptr_to_int 的真实整数地址包含 descriptor offset。

还需回归入口参数及 workspace/lock 索引，并验证独立 TLAToLinalg 的完整编译路径。本文只描述实现与约束，不代表已经通过这些验证。
