# TLA GM view：保留指针来源的 lowering 设计

状态：已按本文实现首版，未更新测试、未构建验证。基于当前分支的 `view.py`、TLA builder、TritonToLinalg，以及 AscendNPU-IR 的 MemRef verifier 和 InferHIVMMemScope 源码分析。

## 1. 结论

采用修正后的方案 A：增加 TLA 专用的 `ascend.tla_pointer_view`，在现有 TritonToLinalg 的指针转换阶段将其转换成 `memref.reinterpret_cast`。保留原指针的 memref 来源和元素偏移，不经过整数地址中介。

实现集中在 TLA 前端及其专用转换 pattern；共享部分只增加 op 定义、注册和 legality。无需新增 pass，不修改通用 `tt.ptr_to_int` 的语义，不修改 NPU-IR 的 HIVM 方言、同步分析或 launcher。

方案 A 原提议有三个必须修正的地方：

1. 最终 offset 是原指针的 descriptor offset 加 `linear`，不能只使用 `linear`。
2. 当前参数 memref 使用默认 memory space，而 TLA GM buffer 显式使用 GM；两者不能直接进行 `reinterpret_cast`。需要统一 TLA GM 的中间表示。
3. 是否能保留来源必须在转换阶段判断；GM fallback 放在新 op 的 converter 内，不能交给 Python `_bind` 判断。

这项优化改善别名、同步和流水分析可见性，不直接减少 GM 读取量，也不意味着所有动态指针都能变成函数参数上的 view。

## 2. 当前实现与问题

`third_party/ascend/language/tla/view.py` 的 `_entry` 目前计算：

```text
linear = sum(offsets[i] * strides[i])
ptr -> tt.ptr_to_int -> hivm.hir.pointer_cast -> memref.reinterpret_cast
```

`_bind` 产生一个 identity layout 的一维 memref，然后最终 view 描述 shape、stride 和 offset。这样可以表达正确的地址，但经过整数地址后，普通 memref view 链不再直接通向函数参数。后端若只追踪 memref alias/view，会在 `pointer_cast` 处失去来源。

普通 Triton 的 `BlockPtrAnalysis.cpp` 已通过 `BlockData::createCastOp` 将结构化指针访问表示为 `reinterpret_cast`。TLA 应复用转换后的指针 descriptor，而不是再做一次地址整数化。

`TritonOpConverter.cpp` 的 `materializePointerAddress` 特别处理了 descriptor offset：完整地址是 aligned pointer 加元素偏移乘元素字节数。它也处理了标量指针的控制流 carrier。因此不能为了 TLA 改掉通用 `PtrToIntConverter` 的这一行为。

## 3. A/B 方案比较

| 项目 | A：显式 pointer-view op | B：在 ptr_to_int 中识别组合 |
| --- | --- | --- |
| 意图 | 显式表达“指针上的 buffer view” | 从整数转换的使用链猜测 |
| 修改边界 | TLA 专属 pattern，加少量共享注册 | 修改所有 Triton 都使用的转换入口 |
| 对 canonicalize 的依赖 | 无固定邻接/唯一用户要求 | 依赖中间链没有被改写或增加用户 |
| 类型转换 | 直接替换 memref 结果 | ptr_to_int 返回整数，不能直接用 memref 替换它 |
| 偏移和空间检查 | 在一个 op 中完成 | 需要跨 op 检查、改写用户、处理转换映射 |
| 长期维护 | 一个稳定的语义边界 | 多种等价写法容易漏匹配 |

B 并非不可实现，但“唯一用户为 pointer_cast”还不充分：仍需核对元素类型、memory space、layout 和后续 offset。即便独占使用，也必须替换消费方并正确删除整数链，不能把 `ptr_to_int` 自己替换成 memref。现有 dialect conversion 存在多阶段转换和 adaptor 映射，在 producer pattern 中主动改写用户会额外增加维护负担。

不建议以 B 为正式方案，也不建议把新 pattern 伪装成通用 ptr_to_int 优化。若以后要做通用组合折叠，应单独定义其合法性条件。

## 4. 新 op 的最小语义

建议名字：`ascend.tla_pointer_view`。

```mlir
// 示意语法；最终 assemblyFormat 可不同。
%v = ascend.tla_pointer_view %ptr, %linear
  : !tt.ptr<bf16>, index
    -> memref<128x64xbf16, strided<[64, 1], offset: ?>>
```

输入只需要 scalar typed pointer 和一个以元素为单位的 `linear`。当前 TLA 的 shape/strides 都是静态正整数，可以直接由结果 memref 类型读取，避免在 operand/attribute/type 中重复存储 sizes、strides。

验证要求：

- 输入是标量指针，不是 tensor-of-pointers 或 block pointer。
- `linear` 是 scalar index；前端 builder 负责整数转换。
- 结果是静态 shape、静态正 strides、动态 offset 的 ranked memref。
- 结果元素类型与指针指向的元素类型一致；首版优化限于转换后仍保持相同存储类型的常规字节可寻址元素。
- 结果使用默认 memory space；此 op 专用于 GM pointer entry。
- op 不执行 load/store，不分配内存，只创建 alias 描述。不得声明结果为独立分配。

`i1` 等可能由 type converter 归一化存储类型的情况、改变元素类型的 pointer bitcast，应明确走已有兼容路径或报错，不可忽略类型差异强行 reinterpret。不能宣称新增 op 自然覆盖所有 `TT_Ptr` 的 pointee 类型。

新 op 只负责 pointer entry。`tla.view(buffer, ...)` / `tla.subview` 的相对切片语义保持不变。

## 5. 统一 GM memory space

当前 `convertPointerType` 生成 `memref<?xT>`，没有显式空间；`create_tla_buffer_type` 对 GM 则生成 `#hivm.address_space<gm>`。本地 MLIR `ReinterpretCastOp::verify` 明确要求输入输出的元素类型和 memory space 相同，因此直接照抄现有 `ty.to_ir()` 会验证失败。

建议在 **TLA buffer 的 IR 类型构造入口**统一：

```text
Python buffer.space == "gm" -> 默认 memory space 的 memref
UB / L1 / L0C / ...        -> 保持现有显式 HIVM memory space
```

Python 的 `space="gm"` 语义不变。GM 的 view、subview、reshape、fallback 类型均使用同一个构造规则，不能只在新 converter 中删除结果的 GM 标注，否则后续 builder 仍会构造不一致的类型。

这样与普通 Triton 的输入表示一致，后续由现有 InferHIVMMemScope 根据函数参数及 view 链传播 GM。已核对该 pass 对函数参数/外部函数和 GM pointer_cast 的推导入口；落地仍需验证 TLA 实际 pipeline 的推导顺序及 DMA 操作的传播结果。

不建议为了省几行代码插入 `memref.memory_space_cast`：这会新增一种后端必须正确追踪的桥接，而目标正是复用现有参数到 view 的路径。也不能用 `memref.cast` 或未消除的 unrealized cast 掩盖 memory space 不一致。

这一步不改 kernel 的 Triton 指针签名和 launcher 参数，但会改变 TLA 中间 IR 类型，必须同步更新相关 IR 测试；不能把它描述为仅修改 `_entry` 三行。

## 6. 转换算法与正确性

### 6.1 快路径：保留 memref 来源

converter 使用 adaptor 中已经转换的 pointer memref，而不是重新解析一套 `tt.addptr` 算术。只复用保留 rank、元素类型和空间的 descriptor materialization；现有 `unwrapPointerDescriptorMaterialization` 是可参考的边界，不能无条件剥离所有 unrealized cast。

设转换后的源 descriptor 表示：

```text
pointer_address = aligned_base + source_offset * sizeof(T)
linear          = sum(offsets[i] * strides[i])
result_offset   = source_offset + linear
```

生成 view 后，元素 `(i, j)` 地址应为：

```text
aligned_base + (source_offset + linear + i*S0 + j*S1) * sizeof(T)
```

`memref.reinterpret_cast` 的 offset 相对底层 aligned base，**不会自动累加源 memref 的 offset**。因此即使直接以转换后的 memref 为 source，也必须显式传 `source_offset + linear`。

最小实现可先以适配后的 memref 为 source：从静态 layout 或 `memref.extract_strided_metadata` 获取 offset，生成一个正确的 reinterpret。对普通 `tt.addptr` 已产生的 reinterpret 链，复用现有折叠将 source 收敛到参数。不要再引入通用的指针来源分析器。

需要显式展开 view 链时，只处理确认保留 aligned base 的 memref 操作；已经取得最终 descriptor 的完整 offset 后，不得再逐层累加旧 offset。转换完成及已有 canonicalize 后，对计划支持的参数/标量 addptr 用例检查其最终根确实为参数。

注意：

- source offset、linear、结果 strides 都是元素单位；只有 fallback 的地址整数化涉及字节。
- linear 不再乘源 memref 的 stride；它已经是原始 pointer 上的元素偏移。
- 普通标量指针参数是主要快路径，不应归入“标量指针 fallback”。
- `linear == 0` 不代表 source offset 为零。是否折叠出常量零只影响 IR 简洁程度，不影响正确性。
- 不添加未经证明的 no-wrap/inbounds 假设；动态边界仍由调用者保证。

### 6.2 Fallback：来源已经是整数地址

当前 pointer `select`、某些 `scf.if` 返回值和 `int_to_ptr` 会通过 i64 carrier + pointer_cast 表达。新 op 不负责重写这些控制流，也不尝试从任意整数反推函数参数。

对这类源，可保留其现有 pointer_cast 根，再在其 descriptor 上生成正确的 reinterpret；如果需要重建绑定，则复用 `materializePointerAddress` 得到包含源 offset 的完整地址，创建 identity 一维 pointer_cast，再用 **linear** 生成最终 reinterpret。两种写法都不得重复累加 source offset。

GM 的 fallback 决策只放在 converter 内；Python `_bind` 可以继续服务非 GM 或暂不纳入新 op 的兼容路径。新 op 已发出后，不应再从 Python 选择 fallback。

无法证明快路径合法不等于可以任意 fallback：元素类型/存储宽度不兼容时，只有已有路径明确支持才能回退，否则应给出诊断。

普通 TLA view 的 fallback 合法，不代表它符合 fixed-stage pipeline 的可调度约束。若后者要求 GM root 是参数，就应继续拒绝 opaque pointer；本设计不会通过更换最后一个 op 名字绕过该约束。

### 6.3 偏移整数宽度

当前 `_entry` 在调用 builder 前计算 linear，可能在 i32 乘法中已经溢出。事后 cast 到 index 无法恢复。

实现时应明确地址偏移的宽度契约：在乘 stride 之前将 offset 扩展到足够宽的整数（当前设备按 64 位处理），再求和并转 index；有符号值保持符号扩展，无符号值按其语义处理。Python 大整数、超出可表示范围的值和静态 stride 乘积应有检查。

这是需要单独说明的语义修正，不应混入“原样替换且所有溢出行为不变”的承诺。首版若保持旧算术，则必须把快路径保证限制在原 linear 运算不溢出的输入，并将宽度修正列为后续工作。

## 7. 预期 IR

以下都是设计示意，**不是实际编译产物**。

输入：

```python
tile = tla.view(K + p, (128, 64), strides=(576, 1), offsets=(r, c))
```

在转换并折叠后，期望保留：

```mlir
// %K : memref<?xbf16>; %p, %r, %c 已按正确宽度转为 index。
%s = arith.constant 576 : index
%row = arith.muli %r, %s : index
%linear = arith.addi %row, %c : index
%offset = arith.addi %p, %linear : index
%tile = memref.reinterpret_cast %K to
  offset: [%offset], sizes: [128, 64], strides: [576, 1]
  : memref<?xbf16>
    to memref<128x64xbf16, strided<[576, 1], offset: ?>>
```

这里参数 descriptor offset 为零，因此结果为 `p + r*576 + c`。如果调用边界允许非零 descriptor offset，还必须加上该值，不能仅凭“是 block argument”就假定为零。

零偏移直接 view：

```mlir
%tile = memref.reinterpret_cast %K to
  offset: [0], sizes: [128, 64], strides: [64, 1]
  : memref<?xbf16> to memref<128x64xbf16>
```

这是允许 canonicalize 后得到的简化形式；转换初始结果保留动态 offset layout 同样正确。后续 memory scope 推导再统一为显式 GM。对 opaque pointer，IR 仍允许存在 pointer_cast，不能要求所有测试都全局 `CHECK-NOT: pointer_cast`。

## 8. 最小实现改动清单

| 位置 | 必要改动 |
| --- | --- |
| `include/Dialect/TritonAscend/IR/TritonAscendOps.td` 及对应 verifier 实现 | 新增 TLA 专用 op 和类型/layout 检查 |
| `language/tla/cpp/tla_builder.cc` | 注册 `create_tla_pointer_view`；GM buffer 类型统一使用默认空间 |
| `language/tla/builder.py` | 将新 native builder 纳入现有 builder 分发 |
| `language/tla/view.py` | GM entry 发新 op；buffer subview 不变；更新语义注释 |
| `include/TritonToLinalg/TritonOpConverter.h`、`lib/TritonToLinalg/TritonOpConverter.cpp` | 增加只匹配新 op 的 converter，复用 descriptor/address 工具 |
| `lib/TritonToLinalg/TritonToLinalgPass.cpp` | 在现有 memory conversion 阶段注册 pattern，并显式将新 op 标为 illegal |
| 对应 TLA/转换测试 | 检查地址、根、memory space、fallback 与诊断 |

现有 memory conversion 已对 TritonAscend 方言实施 pointer-free legality，但仍建议为新 op 显式列入 illegal 清单，确保不能因类型提前适配而漏转换。确认转换后没有新 op 或用于掩盖类型不匹配的 unrealized cast 遗留。

不增加 `lower_ttir` pipeline 步骤，不改通用 PtrToIntConverter、IfConverter、PointerSelectConverter，不改 kernel 参数顺序、workspace/lock 参数索引、launcher 和 NPU-IR passes。`global_dyn_memref_args` 等边界元数据仍按现有规则产生，需通过回归确认而不是仅凭设计断言。

建议将改动拆成两个便于审查的部分：先统一 TLA GM 类型表示并更新测试，再增加专用 op 和转换。偏移宽度修正若同时实现，单独列出行为变化和测试。

## 9. 落地验证清单

| 用例 | 核心断言 |
| --- | --- |
| `view(K, (128,64))` | 根为参数；无新增 pointer_cast；输入输出空间一致 |
| `view(K+p, ...)` | p 保留且只累加一次 |
| `K+p` 加动态 offsets、非连续 strides | 最终 offset 为 source offset + linear，stride 不被错误相乘 |
| 多层 addptr/reinterpret、非零 descriptor offset | 不漏偏移、不重复加偏移 |
| linear 为零、source offset 非零 | 不能错误折叠为 offset 0 |
| 同一个指针同时用于 ptr_to_int、load 和 view | 普通整数地址和 load 不受新 op 影响 |
| pointer select/if/int_to_ptr | 保留完整地址和兼容路径，不假装有 arg root |
| GM view 后 subview/reshape/DMA | 类型全链一致，推导后源/目的空间正确 |
| bf16/f16/f32、i1/类型改变的 bitcast | 常规类型正确；不支持组合明确处理 |
| 负偏移、超过 i32 范围的有效地址计算 | 符号和运算宽度符合声明的契约 |
| 非法 tensor pointer、shape/stride/type | verifier 或前端有明确诊断 |
| fixed-stage TLA 用例 | arg-root 快路径通过；opaque root 仍按原约束诊断 |

先做 op verifier 与转换 FileCheck，再做真实 ttadapter 输出和 GM DMA 数值验证。最终至少覆盖一个 DSA gather 风格的 `view(PACKED_KV + safe*576, (1,576))`，检查来源仍是 PACKED_KV 参数，并回归一个带 workspace/同步参数的 kernel。

上述测试为后续验证要求，不代表已经执行。

## 10. 首版实现范围

- 已实现专用 op、verifier、builder 及 memory conversion pattern，并统一 TLA GM memref 的默认空间表示。
- converter 直接复用适配后的 descriptor：源为 reinterpret 时直接读取其完整 offset，否则读取静态 offset 或提取动态 metadata。opaque pointer 保留现有 pointer_cast 根，不额外整数化或重复绑定。
- GM offsets 在乘 stride 前扩展到 i64，避免 view 内部的 i32 乘法溢出。调用者仍需保证完整元素偏移在有效的有符号 64 位范围内；已经在 `K + expr` 的 expr 中发生的溢出不由 view 修复。
- GM entry 明确拒绝非字节可寻址元素（如 i1）以及非整数/浮点 pointee；转换后元素类型或空间不匹配时报错，不做未经验证的类型重解释。
- 已核对当前 LowerTLAStages：其 `storageRoot` 沿 ViewLikeOpInterface 追踪，且明确将缺省空间的参数 view 视作 GM。因此普通参数/addptr 路径满足来源约束；仍需满足该 pass 对跨迭代访问独立性的其余检查。opaque pointer 不因此获得 schedule 支持。
- 未修改 NPU-IR、通用 ptr_to_int/控制流转换、launcher 和测试文件；未运行编译或用例。
