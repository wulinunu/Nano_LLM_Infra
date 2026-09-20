# Inference Engine Core 测试报告

## 1. 测试目标

本报告使用真实运行结果验证推理引擎的核心能力：

1. PagedAttention 能否根据 Block Table 正确读取非连续 KV Cache。
2. Continuous Batching 能否在运行过程中接收新请求。
3. KV Block 不足时能否抢占请求，并在资源释放后恢复执行。
4. 所有请求结束后，KV Block 能否完整归还。
5. Triton 和 CUDA PagedAttention 相对 PyTorch Reference 的实际性能。

## 2. 测试配置

### Streaming Engine

- GPU：单卡
- KV Block Size：4
- KV Block 总数：2
- Max Batch Size：2
- 初始请求数：2
- 运行中动态插入请求数：1
- Sampling：Greedy

两个 KV Block 是刻意设置的压力条件，用于触发真实的 Preemption。

### PagedAttention

- GPU：单卡
- 数据类型：FP32
- Heads：32
- Head Dimension：64
- Block Size：16
- Context Length：128、512、1024、2048、4096
- Warmup：20 次
- 正式迭代：100 次

## 3. Continuous Batching 与 Preemption

运行结果：

```text
step=00 running=2 waiting=0 preempted=0 finished=0 free_blocks=0 generated={'0': 1, '1': 7}
step=01 running=0 waiting=0 preempted=1 finished=1 free_blocks=2 generated={'1': 3}
inserted request=2 prompt=[20, 21, 22, 23, 24] max_new_tokens=2
step=02 running=1 waiting=1 preempted=0 finished=1 free_blocks=0 generated={'0': 11}
step=03 running=0 waiting=1 preempted=0 finished=2 free_blocks=2 generated={'0': 7}
step=04 running=1 waiting=0 preempted=0 finished=2 free_blocks=0 generated={'2': 31}
step=05 running=0 waiting=0 preempted=0 finished=3 free_blocks=2 generated={'2': 19}
```

### 3.1 Block 耗尽

Step 0 中两个初始请求同时运行，`free_blocks=0`，说明两个 KV Block 已全部占用。

### 3.2 请求抢占

Step 1 中 Request 0 需要扩展 Block，但当前没有空闲 Block，因此进入 `preempted` 队列。Request 1 正常完成并释放资源：

```text
preempted=1 finished=1 free_blocks=2
```

这证明抢占由真实的 KV Block 不足触发，而不是测试代码直接修改请求状态。

### 3.3 抢占恢复与 Waiting

Step 1 结束后动态插入 Request 2。Step 2 中调度器优先恢复 Request 0，此时两个 Block 再次被占满，因此 Request 2 保持 Waiting：

```text
running=1 waiting=1 preempted=0 free_blocks=0
```

Request 0 在 Step 3 完成并释放 Block后，Request 2 才在 Step 4 被调度。结果证明调度器能够区分 Running、Waiting 和 Preempted，并优先恢复被抢占请求。

### 3.4 KV Block 回收

Step 5 中三个请求全部结束：

```text
running=0 waiting=0 preempted=0 finished=3 free_blocks=2
```

最终空闲 Block 数恢复到初始值 2，说明本轮运行没有丢失 Block。

## 4. 生成结果一致性

资源受限并发生抢占后的最终结果：

```text
request=0 prompt=[1, 2, 3] generated=[1, 11, 7]
request=1 prompt=[10, 11] generated=[7, 3]
request=2 prompt=[20, 21, 22, 23, 24] generated=[31, 19]
```

这些结果与之前使用 8 个 KV Block、未发生抢占时的生成结果完全一致。

Request 0 被抢占后会清空 Block Table 和 `cached_tokens`，恢复时重新执行 Prefill。相同的最终 Token 表明抢占、KV 释放、重新 Prefill 和恢复生成没有改变推理结果。

## 5. PagedAttention 正确性

所有 Context Length 下，Triton 和 CUDA 输出都通过 `torch.allclose`：

```text
Triton 最大绝对误差：6.71e-8 ～ 1.79e-7
CUDA   最大绝对误差：1.86e-7 ～ 2.98e-7
```

最大误差低于 `3e-7`，说明两种 Kernel 都能正确完成：

- Logical Block 到 Physical Block 的映射。
- 跨物理 Block 的 K/V 读取。
- QK 计算。
- Online Softmax。
- Probability 与 V 的加权累加。

## 6. PagedAttention 性能

实测结果：

```text
Context 128:
  Reference 0.153ms
  Triton    0.065ms，2.36x
  CUDA      0.046ms，3.32x

Context 512:
  Reference 0.157ms
  Triton    0.050ms，3.16x
  CUDA      0.178ms，0.89x

Context 1024:
  Reference 0.158ms
  Triton    0.058ms，2.71x
  CUDA      0.354ms，0.45x

Context 2048:
  Reference 0.178ms
  Triton    0.114ms，1.56x
  CUDA      0.706ms，0.25x

Context 4096:
  Reference 0.354ms
  Triton    0.204ms，1.73x
  CUDA      1.280ms，0.28x
```

Triton 在全部测试长度上都快于 Reference，加速比为 `1.56x～3.16x`。它直接按 Block Table 读取 K/V，并在单个 Kernel 内完成分块计算和 Online Softmax，避免了 Reference 的 K/V 拼接和中间 Tensor。

CUDA 在 128 Token 时最快，但从 512 Token 开始慢于 Reference。当前 CUDA 实现在逐 Token 循环内执行归约和两次 `__syncthreads()`，Context Length 增长时同步次数同步增长，因此 4096 Token 时延迟达到 `1.280 ms`。

这些数据说明当前 Triton 实现更适合长上下文 Decode；CUDA 版本已经实现正确算法，但还需要扩大 Token Tile 并减少同步。

## 7. 框架有效性

真实数据分别验证了控制面和数据面：

- 控制面：动态插入的 Request 能够进入 Waiting，并在资源可用后继续执行。
- 资源管理：Block 不足会触发 Preemption，恢复后结果保持一致，最终全部 Block 被归还。
- 数据面：PagedAttention 能正确读取分页 KV Cache，最大误差低于 `3e-7`。
- 性能：Triton PagedAttention 在 128～4096 Context 范围内均快于 Reference。

因此，PagedAttention、Continuous Batching、KV Block Pool 和 Engine Step 已经组成可运行的最小推理引擎闭环。

### 7.1 torch.profiler Timeline

推理引擎统一使用 torch.profiler 做宏观 Timeline 分析。代码将每个 `Engine.step_N` 拆成以下范围：

```text
Scheduler
Prefill / Decode
Stack_and_Sample
Update_Request_State
```

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python evals/demo_inference_streaming.py --profile
```

脚本会打印按 `self_cpu_time_total` 排序的算子摘要，并导出：

```text
reports/traces/inference_engine.json
```

本次 RTX 4090 Trace 的关键结果：

- `Engine.step_0` 为 `228.8 ms`，其中首个 Prefill 为 `197.4 ms`；`Runtime Triggered Module Loading` 自耗时为 `159.7 ms`，说明首步主要是冷启动。
- `Engine.step_1` 仍受 Lazy Loading 影响，为 `11.5 ms`。
- `Engine.step_2~5` 稳定在 `1.82~1.95 ms`。
- 稳定阶段 Scheduler 只有 `12~29 us`，调度状态机本身不是主要瓶颈。
- 6 个 Step 的 CUDA Kernel Self Time 合计约 `0.49 ms`，明显小于 `247.9 ms` CPU Self Time。当前模型很小，端到端时间主要受模块加载、Python 循环和 Kernel Launch 开销影响，GPU 计算没有被充分喂满。

Trace 原文件体积较大且与 GPU/软件环境强相关，因此不提交仓库；报告保留埋点语义、复现命令和判读方法。

## 8. 实现边界

当前结果不能证明以下内容：

- 一次运行只能说明当前场景没有 Block 泄漏，严格的长期稳定性仍需循环压测。
- Prefill 和 Decode 仍按 Request 循环，不是生产级融合 Batch Kernel。
- PagedAttention Benchmark 只覆盖 Batch Size 1 和 FP32。
- 尚未提供 NCU 的 L1/L2 Cache、Memory Throughput 和 SM 利用率数据。
- torch.profiler 的绝对时间依赖目标 GPU，跨机器比较时应重新采集 Trace。

## 9. 原始测试命令与输出

### 9.1 Streaming、Preemption 与 KV 回收

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python evals/demo_inference_streaming.py
```

```text
step=00 running=2 waiting=0 preempted=0 finished=0 free_blocks=0 generated={'0': 1, '1': 7}
step=01 running=0 waiting=0 preempted=1 finished=1 free_blocks=2 generated={'1': 3}
inserted request=2 prompt=[20, 21, 22, 23, 24] max_new_tokens=2
step=02 running=1 waiting=1 preempted=0 finished=1 free_blocks=0 generated={'0': 11}
step=03 running=0 waiting=1 preempted=0 finished=2 free_blocks=2 generated={'0': 7}
step=04 running=1 waiting=0 preempted=0 finished=2 free_blocks=0 generated={'2': 31}
step=05 running=0 waiting=0 preempted=0 finished=3 free_blocks=2 generated={'2': 19}

Final requests:
request=0 prompt=[1, 2, 3] generated=[1, 11, 7] total=[1, 2, 3, 1, 11, 7]
request=1 prompt=[10, 11] generated=[7, 3] total=[10, 11, 7, 3]
request=2 prompt=[20, 21, 22, 23, 24] generated=[31, 19] total=[20, 21, 22, 23, 24, 31, 19]
```

### 9.2 PagedAttention

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python evals/bench_paged_attention.py
```

```text
=== num_tokens=128 block_size=16 num_heads=32 head_dim=64 dtype=torch.float32 ===
    ref: allclose=True  max_abs_diff=0.000000e+00
    ref: latency=0.153 ms
 triton: allclose=True  max_abs_diff=1.788139e-07
 triton: latency=0.065 ms
   cuda: allclose=True  max_abs_diff=2.980232e-07
   cuda: latency=0.046 ms
 triton: speedup_vs_ref=2.36x
   cuda: speedup_vs_ref=3.32x

=== num_tokens=512 block_size=16 num_heads=32 head_dim=64 dtype=torch.float32 ===
    ref: allclose=True  max_abs_diff=0.000000e+00
    ref: latency=0.157 ms
 triton: allclose=True  max_abs_diff=1.229346e-07
 triton: latency=0.050 ms
   cuda: allclose=True  max_abs_diff=2.980232e-07
   cuda: latency=0.178 ms
 triton: speedup_vs_ref=3.16x
   cuda: speedup_vs_ref=0.89x

=== num_tokens=1024 block_size=16 num_heads=32 head_dim=64 dtype=torch.float32 ===
    ref: allclose=True  max_abs_diff=0.000000e+00
    ref: latency=0.158 ms
 triton: allclose=True  max_abs_diff=8.940697e-08
 triton: latency=0.058 ms
   cuda: allclose=True  max_abs_diff=1.937151e-07
   cuda: latency=0.354 ms
 triton: speedup_vs_ref=2.71x
   cuda: speedup_vs_ref=0.45x

=== num_tokens=2048 block_size=16 num_heads=32 head_dim=64 dtype=torch.float32 ===
    ref: allclose=True  max_abs_diff=0.000000e+00
    ref: latency=0.178 ms
 triton: allclose=True  max_abs_diff=8.195639e-08
 triton: latency=0.114 ms
   cuda: allclose=True  max_abs_diff=1.862645e-07
   cuda: latency=0.706 ms
 triton: speedup_vs_ref=1.56x
   cuda: speedup_vs_ref=0.25x

=== num_tokens=4096 block_size=16 num_heads=32 head_dim=64 dtype=torch.float32 ===
    ref: allclose=True  max_abs_diff=0.000000e+00
    ref: latency=0.354 ms
 triton: allclose=True  max_abs_diff=6.705523e-08
 triton: latency=0.204 ms
   cuda: allclose=True  max_abs_diff=2.086163e-07
   cuda: latency=1.280 ms
 triton: speedup_vs_ref=1.73x
   cuda: speedup_vs_ref=0.28x
```

### 9.3 Engine Trace

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python evals/demo_inference_streaming.py --profile
```

Trace 输出路径：

```text
reports/traces/inference_engine.json
```
