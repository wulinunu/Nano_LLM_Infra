# RMSNorm Benchmark 报告

## 测试环境

- 日期：2026-06-24
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- CUDA Compute Capability：12.0
- Benchmark 脚本：`benchmarks/bench_rmsnorm.py`
- 运行命令：

```bash
PYTHONPATH=src python benchmarks/bench_rmsnorm.py
```

## 输入配置

- 输入 shape：`(4, 128, 4096)`
- Warmup 次数：`20`
- Benchmark 迭代次数：`100`
- Epsilon：`1e-6`
- 数据类型：`float32`

## 正确性

两个自定义 CUDA 实现都通过了与 PyTorch reference 的数值对齐检查。

| 实现 | 是否通过 | 最大误差 |
| --- | --- | --- |
| shared memory 版本 | `True` | `1.907349e-06` |
| warp shuffle 版本 | `True` | `1.907349e-06` |

## 延迟

每次 RMSNorm 调用的平均耗时：

| 实现 | 平均延迟 |
| --- | --- |
| PyTorch reference | `0.1142 ms` |
| shared memory CUDA 版本 | `0.0250 ms` |
| warp shuffle CUDA 版本 | `0.0230 ms` |

## 加速比

| 对比项 | 加速比 |
| --- | --- |
| shared memory CUDA 版本 vs PyTorch reference | `4.57x` |
| warp shuffle CUDA 版本 vs PyTorch reference | `4.97x` |
| warp shuffle CUDA 版本 vs shared memory CUDA 版本 | `1.09x` |

## 结论

这次结果说明 Python 到 PyBind，再到自定义 CUDA kernel 的完整调用链路已经跑通。两个 CUDA 版本的最大误差都是 `1.907349e-06`，在当前 `float32` 测试下可以接受。

主要加速来自把 RMSNorm 融合成单个 CUDA kernel，避免 PyTorch reference 中多个 tensor operation 带来的额外 kernel launch 和中间张量读写。`warp shuffle` 版本比 `shared memory` 版本略快，因为它在 warp 内使用寄存器级 shuffle 做规约，减少了 shared memory 访问和同步开销。
