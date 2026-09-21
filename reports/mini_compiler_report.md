# Mini AI Compiler 测试报告

## 1. 结论

该 Demo 已跑通以下编译链路：

```text
PyTorch FX
  -> Custom Graph IR
  -> Fusion / Memory Planning
  -> Kernel IR
  -> Triton Codegen
  -> Dispatch
```

核心结果：

- 捕获 `Add -> RMSNorm -> Linear` 计算图。
- 将 RMSNorm 的 6 个细粒度算子融合为 1 个算子。
- 通过生命周期分析复用中间 Buffer。
- 为 Add 和 RMSNorm 生成并执行 Triton Kernel。
- 编译结果与 PyTorch Eager 在 `rtol=1e-3, atol=1e-3` 下对齐。

## 2. 测试配置

- 运行脚本：`evals/compiler_demo.py`
- 输入 Shape：`(32, 128)`
- 目标子图：`Add -> RMSNorm -> Linear`
- Kernel 后端：Triton
- Linear：Fallback 到 `torch.matmul`
- 正确性检查：`torch.testing.assert_close`

## 3. 编译流程

### 3.1 Graph Capture

`torch.fx` 将 Eager 模型捕获为静态图，再转换为自定义 Graph IR：

```text
x, residual
  -> add
  -> pow -> mean -> add -> rsqrt -> mul -> mul
  -> linear
  -> output
```

Graph IR 保存算子类型、输入输出、Shape 和 Dtype，为后续优化提供统一表示。

### 3.2 Fusion

Fusion Pass 沿数据流匹配 RMSNorm：

```text
pow -> mean -> add -> rsqrt -> mul -> mul
```

优化后：

```text
add -> fused_rmsnorm -> linear -> output
```

6 个算子被替换为一个 `fused_rmsnorm`，减少中间张量物化和 Kernel Launch。

### 3.3 Memory Planning

Memory Planning Pass 记录每个张量的最后使用位置。生命周期结束且 Shape 兼容的 Buffer 可以复用：

```python
out_add = residual
out_norm = out_add
```

Add 覆盖写入 `residual`，RMSNorm 继续复用同一块显存，不再为两个中间结果单独分配 Buffer。

### 3.4 Lowering

高层算子被转换为 Kernel IR：

```text
KernelIR(name='add', loads=2, computes=1, stores=1)
KernelIR(name='fused_rmsnorm_pow_1', loads=2, computes=4, stores=1)
```

其中：

- `loads`：从 HBM 读取数据
- `computes`：在片上完成计算
- `stores`：将结果写回 HBM

本例使用 `BLOCK_SIZE=128`，每行输入由一个 Triton Program 处理。

### 3.5 Codegen 与 Dispatch

Codegen 根据 Kernel IR 生成：

- `add_kernel`
- `fused_rmsnorm` Kernel
- Host 侧 `dispatch_and_run`

Dispatch 负责 Buffer 别名、Kernel Launch，以及 Linear 的 PyTorch Fallback。

## 4. 正确性

生成的 Triton Kernel 与 PyTorch Eager 输出通过：

```python
torch.testing.assert_close(
    expected_out,
    compiled_out,
    rtol=1e-3,
    atol=1e-3,
)
```

这验证了图捕获、算子融合、In-place Buffer 复用和 Triton 执行没有改变计算结果。

## 5. 复现命令

```bash
PYTHONPATH=. python evals/compiler_demo.py
```
