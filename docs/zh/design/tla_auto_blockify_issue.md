# [TLA] 对齐 AutoBlockify 编译策略与 launcher 的 grid 映射

## 问题概述

TLA 的 NPU-IR affinity pipeline 当前不运行 AutoBlockify，但 Triton backend 的 `auto_blockify_enabled` 决策没有显式排除 `enable_tla_mode`。如果该 metadata 为 true，launcher 仍会将实际启动的 block 数截断到物理核数。

这可能造成编译与启动语义不一致：kernel 没有遍历剩余逻辑 program 的循环，launcher 却按存在该循环的方式启动，导致部分 program 未执行。

**状态：根据当前源码确认存在策略缺口，尚未通过端到端运行复现。** 是否触发取决于 AutoBlockify 开关、其他禁用条件以及 launch grid。

## 当前行为与源码依据

| 位置 | 当前行为 |
|---|---|
| `third_party/ascend/backend/compiler.py`：AutoBlockify metadata 决策 | 根据全局开关、SIMT、黑名单和已有 grid transform 决定是否启用，没有显式排除 TLA |
| 同文件：NPU 编译参数生成 | metadata 为 true 时传入 `--enable-auto-blockify-loop`；TLA 另传入 `--enable-tla-mode=true` |
| `third_party/ascend/backend/launcher.py`：`make_launch_spec` | metadata 为 true 时设置 `AUTO_MAP` |
| `third_party/ascend/backend/launcher_src/launcher_args.h`：grid 计算 | `AUTO_MAP` 开启时使用 `min(logicalBlocks, physical_blocks)` |
| AscendNPU-IR：`bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp`：`buildAffinityPipeline` | TLA 路径没有运行 `AutoBlockifyParallelLoopPass` |

普通 HIVM AutoBlockify 会生成跨逻辑 program 的循环。以默认分组大小为例：

```python
# G：逻辑 program 总数；C：物理核数
# launcher 启动 min(G, C) 个 block
for pid in range(block_idx, G, C):
    original_kernel(pid)
```

该 pass 还会尝试提升满足安全条件的不变 load，并增加供同步分析使用的标记。它处理的是 program 到物理核的映射；TLA 的 stage/preload 处理的是 program 内部任务的流水，两者不是同一个机制。

## 触发场景与影响

例如 `G = 100`、`C = 24`：

1. TLA 编译结果中没有 AutoBlockify 循环。
2. backend 仍生成 `auto_blockify_enabled = true`。
3. launcher 只启动 24 个 block。
4. kernel 只能处理前 24 个 program，其余 program 没有执行。

可能表现为输出未写入、精度错误或不完整计算得到的虚假性能提升。`G <= C` 的测试通常无法暴露该问题。

以上是条件满足时的行为推导，不是实测报告。

## 预期行为

当前 TLA 默认保留用户指定的 program 划分，不启用 AutoBlockify：

- backend 输出 `auto_blockify_enabled = false`。
- 不向 NPU 编译器传入 `--enable-auto-blockify-loop`。
- launcher 不设置 `AUTO_MAP`，不因物理核数而截断 grid。
- NPU-IR 对 TLA 与显式 AutoBlockify 开启的组合报错，而不是静默忽略。
- 普通非 TLA kernel 的行为保持不变。

关闭 AutoBlockify 不要求 `grid <= 物理核数`；grid 仍须满足运行时与设备的 launch 限制。物理核数表示并发资源量，不等同于逻辑 block 总数上限。

## 最小修改范围

### 1. Triton backend：统一生成正确的 metadata

在 AutoBlockify 的公共策略决策处优先处理 TLA，避免仅在某一个编译参数分支关闭：

```python
if getattr(opt, "enable_tla_mode", False):
    auto_blockify_enabled = False
elif opt.is_pure_simt:
    # 保留现有 SIMT 策略
    ...
else:
    # 保留现有普通 kernel 策略
    ...
```

编译参数和 launcher 继续消费同一份 metadata，不在 launcher 中独立重算策略。核对该决策发生在 `enable_tla_mode` 确定之后，并覆盖所有 TLA 编译入口。

已有 persistent-grid transform 的授权是另一套机制，不能因为关闭 AutoBlockify 就默认授权。当前 TLA 若未支持这类 transform，应明确拒绝不支持的组合。

### 2. AscendNPU-IR：拒绝互相矛盾的选项

在 TLA 编译路径的配置检查处，对以下组合给出清晰诊断：

```text
--enable-tla-mode=true --enable-auto-blockify-loop=true
```

建议诊断：

```text
AutoBlockify is not supported by the TLA affinity pipeline;
launch the full program grid or use an explicitly supported persistent kernel.
```

该检查仅针对显式开启；普通 TLA 调用不应被一个通用默认值误拦截。实现前应核对选项默认值及配置来源。应在运行变换前失败，不生成一个与调用方预期不一致的产物。

### 3. 文档：明确 TLA 的执行合同

- 默认一个 launch block 对应一个逻辑 program，编译器不自动合并多个 program。
- 不依赖不同 program 的执行顺序或同时驻留；跨 program 通信需要另行定义协议。
- stage hint 仅作用于流水域内部，不授权修改 launch grid 或引入 persistent 外层循环。
- MIX 的 Cube/Vector 必须使用一致的逻辑工作编号、迭代范围和同步参与者映射。
- 手写 persistent kernel 负责完整且不重复地覆盖工作项，并正确处理切换工作项时的 buffer 与同步生命周期。当前固定 stage 首版要求流水域直接位于函数中，尚未支持把它嵌入任意 persistent 外层循环。

## 验收用例

| 层次 | 用例 | 预期 |
|---|---|---|
| backend 策略 | TLA + 全局 AutoBlockify 开关开启 | metadata 为 false，编译参数不含启用 AutoBlockify 的选项 |
| backend 策略 | 非 TLA 的既有 SIMT、普通 kernel 和 blacklist 场景 | 保持原有策略 |
| launcher | TLA 对应的 launch spec，`G > C` | 不含 `AUTO_MAP`，实际 launch block 数为 G |
| launcher | `G < C`、`G = C`、多维 grid | grid 乘积与 program ID 映射保持正确 |
| NPU-IR | 显式同时开启 TLA 与 AutoBlockify | 编译失败，并输出针对性诊断 |
| NPU-IR | 仅开启 TLA | 可进入正常 affinity 编译路径 |
| 端到端 | 每个 program 向独立输出位置写入自身 ID；输出预填哨兵值 | 所有位置均正确写入，无遗漏 |

端到端至少覆盖 `G = C + 1`、`G = 2C + 3` 和多维 grid，并分别覆盖受支持的 AIV、MIX 模式。不同模式的 C 应取各自实际使用的核资源数，不能统一硬编码。MIX 测试还需确认 Cube/Vector 的 program 配对。

## 不在本次修改范围

本 issue 不为 TLA 新增 AutoBlockify、自动 persistent 调度或 load 提升，也不调整 stage、多缓冲或 CCGSS 算法。优先修复编译与启动合同，后续若开放自动 program 调度，应同时设计覆盖证明、MIX 同步生命周期及 launcher metadata 协议。
