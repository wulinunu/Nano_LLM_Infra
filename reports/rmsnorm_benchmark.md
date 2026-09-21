# RMSNorm Benchmark 报告

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

两个 CUDA 版本的最大误差都是 `1.907349e-06`，可以接受。

主要加速来自把 RMSNorm 融合成单个 CUDA kernel，避免 PyTorch reference 中多个 tensor operation 带来的额外 kernel launch 和中间张量读写。`warp shuffle` 版本比 `shared memory` 版本略快，因为它在 warp 内使用寄存器级 shuffle 做规约，减少了 shared memory 访问和同步开销。

## NCU 分析

通过 NVTX 只采集 Warmup 后的 Shared Memory 和 Warp-shuffle 各一次正式执行。

采集命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /usr/local/cuda/bin/ncu --set full --nvtx \
  --nvtx-include "rmsnorm_shared/" \
  --nvtx-include "rmsnorm_warp_shuffle/" \
  -o reports/ncu_rmsnorm -f \
  python evals/bench_rmsnorm.py --ncu-profile
```

### Duration

| 实现 | Kernel Duration |
| --- | ---: |
| Shared Memory | `69.98 us` |
| Warp-shuffle | `66.05 us` |

Warp-shuffle 比 Shared Memory 版本快约 `1.06x`。NCU 会重放 Kernel 采集硬件计数器，因此这里的绝对耗时高于普通 Benchmark，应主要关注两个 Kernel 之间的相对差异。

### Shared Memory 指标

| 指标 | Warp-shuffle | Shared Memory | 对比 |
| --- | ---: | ---: | ---: |
| Shared Load Instructions | `4,608` | `16,896` | 降低 `72.7%` |
| Shared Store Instructions | `4,608` | `10,752` | 降低 `57.1%` |
| Shared Instructions Total | `9,216` | `27,648` | 降低 `66.7%` |
| Shared Wavefronts Total | `37,641` | `117,056` | 降低 `67.8%` |
| Shared Memory % Peak | `1.88%` | `5.49%` | 降低 `65.8%` |
| Load/Store Bank Conflicts | `273` | `494` | 降低 `44.7%` |

Warp-shuffle 版本在 Warp 内通过寄存器 Shuffle 完成规约，只在 Warp 之间交换部分和，因此 Shared Memory 指令和 Wavefront 数量都减少约三分之二。这与其更低的延迟一致，说明主要收益来自减少 Shared Memory 访问与同步。

Bank Conflict 从 `494` 降至 `273`，降低 `44.7%`；这为“Warp-shuffle 减少 Shared Memory 访问冲突”提供了硬件计数器证据。

