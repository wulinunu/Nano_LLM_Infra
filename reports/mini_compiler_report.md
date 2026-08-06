# Mini AI Compiler 端到端报告

## 测试环境

- 日期：2026-08-06
- 运行脚本：`evals/compiler_demo.py`
- 运行命令：

```bash
PYTHONPATH=. python evals/compiler_demo.py
```

- 验证目标模型：`Add -> RMSNorm -> Linear`
- 输入 shape：`(32, 128)`
- 验证方式：`torch.testing.assert_close(..., rtol=1e-3, atol=1e-3)`

## 总体结论

整条 Mini Compiler 链路已跑通：

`Graph Capture -> Fusion / Memory Planning -> Lowering -> Triton Codegen -> Dispatch 执行`

编译后执行结果与原生 PyTorch Eager 数值对齐，前 `3x3` 元素完全一致。

---

## Step 1：Graph Capture（图捕获）

### 结果

FX 捕获后的 Custom Graph IR：

```text
x, residual
  -> add
  -> pow_1 -> mean -> add_1 -> rsqrt -> mul -> mul_1
  -> linear
  -> output
```

完整节点列表：

| 节点 | 类型 | 含义 |
| --- | --- | --- |
| `x` / `residual` | placeholder | 模型输入 |
| `add` | call_function | residual add |
| `pow_1` | call_method | `x^2` |
| `mean` | call_method | 求均值 |
| `add_1` | call_function | `+ eps` |
| `rsqrt` | call_function | 开方倒数 |
| `mul` / `mul_1` | call_function | 缩放与乘 weight |
| `weight` | get_attr | RMSNorm 权重 |
| `linear` | call_module | 全连接层 |
| `output` | output | 返回值 |

### 解释

这一步做的是前端图捕获：

1. 用 `torch.fx` 把 Eager 代码变成静态计算图
2. 再导出成自定义 `GraphIR`
3. 每个节点只保留编译器真正需要的信息：`OpType`、`Inputs`、`Outputs`、`Shape/Dtype`

相比 Eager 模式，有了 Graph IR 之后，后续才能做全局优化，例如算子融合和静态内存规划。

---

## Step 2：Transformation Pass（融合与内存规划）

### 2.1 Fusion 结果

融合前 RMSNorm 被拆成 6 个细粒度算子：

```text
pow -> mean -> add -> rsqrt -> mul -> mul
```

融合后变成：

```text
add -> fused_rmsnorm -> linear -> output
```

优化后的图：

```text
IRNode(name='x', ...)
IRNode(name='residual', ...)
IRNode(name='add', inputs=['x', 'residual'])
IRNode(name='fused_rmsnorm_pow_1', inputs=['add', 'weight'])
IRNode(name='weight', ...)
IRNode(name='linear', inputs=['fused_rmsnorm_pow_1'])
IRNode(name='output', inputs=['linear'])
```

### 解释

Fusion Pass 不是按数组下标硬匹配，而是沿数据流出边（`users`）做拓扑匹配：

```text
pow.users -> mean.users -> add.users -> rsqrt.users -> mul.users -> mul
```

命中后，把这 6 个节点替换成一个 `fused_rmsnorm`。

收益：

- 中间张量消除：`pow/mean/rsqrt/...` 的中间结果不再物化到 HBM
- 访存减少：多个 kernel launch 变成一次 fused kernel
- 对 Memory Bandwidth Bound 算子尤其有效

### 2.2 Memory Planning 结果

| 张量 | 最后使用者 | 含义 |
| --- | --- | --- |
| `x` | `add` | add 后可释放/复用 |
| `residual` | `add` | add 后可释放/复用 |
| `add` | `fused_rmsnorm_pow_1` | fused RMSNorm 后可释放/复用 |
| `fused_rmsnorm_pow_1` | `linear` | linear 后可释放/复用 |
| `weight` | `fused_rmsnorm_pow_1` | 参数，不能覆盖写 |
| `linear` | `output` | 最终结果前一步 |

### 解释

Memory Planning Pass 做的是生命周期分析（Liveness Analysis）：

- 记录每个张量最后一次被谁使用
- 若两个中间张量生命周期不重叠，且 shape 兼容，则可静态复用同一块 buffer

在本 Demo 中，后续 Codegen 把这条分析结论落地为：

```python
out_add = residual
out_norm = out_add
```

也就是：

1. `add` 的输出覆盖写到 `residual`
2. `fused_rmsnorm` 的输出继续覆盖同一块显存

因此中间结果不需要反复 `torch.empty_like` 新开显存。

---

## Step 3：Lowering（降到 Kernel / Tile 级）

### 结果

```text
KernelIR(name='add', loads=2, computes=1, stores=1)
KernelIR(name='fused_rmsnorm_pow_1', loads=2, computes=4, stores=1)
```

### 解释

Lowering 把高层 Graph Op 拆成接近硬件执行的微指令：

- `loads`：HBM -> SRAM
- `computes`：寄存器/SRAM 内计算
- `stores`：SRAM -> HBM

同时决定并发参数：

- `BLOCK_SIZE = 128`
- `grid = (B,)`，即每行一个 Block

为什么只有这两个 Kernel？

- `add` 和 `fused_rmsnorm` 走 Triton Codegen
- `linear` 故意 Fallback 到 `torch.matmul`，避免 Demo 里展开复杂 2D MatMul tiling

这是工业编译器里常见的策略：能生成高效 kernel 的算子走 codegen，难写或已有极优库的算子走 fallback。

---

## Step 4：Codegen & Dispatch

### 生成结果摘要

编译器生成了：

1. `add_kernel`
2. `fused_rmsnorm_pow_1`
3. Host 侧 `dispatch_and_run`

Dispatcher 核心逻辑：

```python
out_add = residual      # 静态内存复用
out_norm = out_add      # 静态内存复用

add_kernel[grid](x, residual, out_add, N, BLOCK_SIZE=BLOCK_SIZE)
fused_rmsnorm_pow_1[grid](out_add, weight, out_norm, N, BLOCK_SIZE=BLOCK_SIZE)

out_final = torch.matmul(out_norm, linear_weight.t())
```

### 解释

这一步体现了 Codegen 与 Dispatch 的边界：

- **Codegen**：根据 KernelIR 拼出 Triton 源码
- **Dispatch**：在 CPU 侧分配/别名 buffer，并 launch kernel

`fused_rmsnorm` 内核内部把原本 6 个算子收成一次 load / 多次 compute / 一次 store，这正是融合收益的物理来源。

---

## Step 5：正确性验证

### 结果

```text
✅ 验证通过！生成的 Triton 内核跑出的结果与 PyTorch 原生计算在数值上完全对齐。
```

前 `3x3` 元素对比：

| 来源 | 输出 |
| --- | --- |
| 原生 PyTorch | `[[-0.0126, 0.0226, 0.5822], [-1.3907, -0.1709, -1.1625], [0.2167, 0.4074, -0.0838]]` |
| 编译器生成 | `[[-0.0126, 0.0226, 0.5822], [-1.3907, -0.1709, -1.1625], [0.2167, 0.4074, -0.0838]]` |

### 解释

这说明：

1. 图捕获没有丢算子语义
2. Fusion 替换后的 `fused_rmsnorm` 数学上等价
3. Triton Kernel + In-place 复用没有破坏数值
4. Linear fallback 与原模型一致

---

## 端到端链路总结

| 阶段 | 做了什么 | 本 Demo 的可见结果 |
| --- | --- | --- |
| Graph Capture | FX -> Custom IR | 抓到完整 `Add/RMSNorm/Linear` 图 |
| Fusion Pass | 拓扑 Pattern Matching | 6 个 RMSNorm 细算子合成 1 个 |
| Memory Planning | Liveness Analysis | 明确谁用完即可复用 |
| Lowering | Graph Op -> KernelIR | 拆成 loads/computes/stores |
| Codegen | KernelIR -> Triton 源码 | 动态生成可执行 kernel |
| Dispatch | Host 调度 + buffer 别名 | In-place 执行并得到正确输出 |

## 当前边界

这个 Demo 刻意做了规模精简，但流程是完整的：

1. Shape 传播在部分环境下未完全打印出来，不影响当前正确性验证
2. Linear 仍回退到 PyTorch，没有做完整 MatMul Triton tiling
3. 内存复用目前只把最关键的一条链落地成：

```python
out_add = residual
out_norm = out_add
```

尚未做成通用 buffer pool allocator

这些都不影响主结论：**Mini AI Compiler 的核心编译链路已经打通，并且结果与 Eager 对齐。**
