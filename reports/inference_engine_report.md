# Inference Engine Core 测试报告

## 1. 测试目标

验证最小推理引擎的三条核心链路：

1. Continuous Batching 能否动态接收和调度请求。
2. KV Block 不足时能否抢占、恢复并完整回收资源。
3. PagedAttention 能否正确访问分页 KV Cache，并获得实际加速。

## 2. 测试配置

### 2.1 Streaming Engine

- GPU：单卡
- 模型：Training、Inference、RL 共用的 `TinyTransformerModel`
- 模型规模：Hidden Size 16、1 Layer、2 Heads、MLP Hidden Size 32
- KV Block Size：4
- Max Batch Size：2
- 请求：初始 2 个，运行中插入 1 个
- Sampling：Greedy（`do_sample=False`）
- KV Block：8 个作为无抢占基线，2 个作为压力测试

### 2.2 PagedAttention Benchmark

- 数据类型：FP32
- Heads：32
- Head Dimension：64
- Block Size：16
- Context Length：128、512、1024、2048、4096
- Warmup：20 次
- 正式迭代：100 次

## 3. Continuous Batching、Preemption 与 KV 回收

### 3.1 8 Block 基线

资源充足时没有发生抢占，三个请求在 4 个 Step 内完成：

```text
step=00 running=2 waiting=0 preempted=0 finished=0 free_blocks=6
step=01 running=1 waiting=0 preempted=0 finished=1 free_blocks=6
step=02 running=1 waiting=0 preempted=0 finished=2 free_blocks=6
step=03 running=0 waiting=0 preempted=0 finished=3 free_blocks=8
```

### 3.2 2 Block 压力测试

Step 0 后两个 Block 被占满。Step 1 中 Request 0 无法扩展 KV Block，被放入 `preempted` 队列；Request 1 完成后释放资源。调度器随后优先恢复 Request 0，新插入的 Request 2 保持 Waiting：

```text
step=00 running=2 waiting=0 preempted=0 finished=0 free_blocks=0
step=01 running=0 waiting=0 preempted=1 finished=1 free_blocks=2
step=02 running=1 waiting=1 preempted=0 finished=1 free_blocks=0
step=03 running=0 waiting=1 preempted=0 finished=2 free_blocks=2
step=04 running=1 waiting=0 preempted=0 finished=2 free_blocks=0
step=05 running=0 waiting=0 preempted=0 finished=3 free_blocks=2
```

### 3.3 生成结果一致性

8 Block 基线和 2 Block 压力测试的最终结果相同：

```text
request=0 generated=[1, 11, 7]
request=1 generated=[7, 3]
request=2 generated=[31, 19]
```

## 4. PagedAttention 正确性与性能

### 4.1 正确性

所有 Context Length 下，Triton 和 CUDA 输出均通过 `torch.allclose`：

```text
Triton 最大绝对误差：6.71e-8 ～ 1.79e-7
CUDA   最大绝对误差：1.86e-7 ～ 2.98e-7
```

这验证了 Block Table 寻址、非连续 K/V 读取、QK 计算、Online Softmax 和 V 加权累加。

### 4.2 性能

```text
Context   Reference   Triton          CUDA
128       0.153 ms    0.065 ms 2.36x  0.046 ms 3.32x
512       0.157 ms    0.050 ms 3.16x  0.178 ms 0.89x
1024      0.158 ms    0.058 ms 2.71x  0.354 ms 0.45x
2048      0.178 ms    0.114 ms 1.56x  0.706 ms 0.25x
4096      0.354 ms    0.204 ms 1.73x  1.280 ms 0.28x
```

Triton 在所有长度上都快于 Reference。它直接按 Block Table 读取 K/V，并在单个 Kernel 中完成分块计算和 Online Softmax，减少了 K/V 拼接和中间 Tensor。

CUDA 版本仅在 128 Token 时更快。当前实现按 Token 循环归约并频繁执行 `__syncthreads()`，Context 越长，同步开销越明显。

## 5. torch.profiler Timeline

每个 `Engine.step_N` 被拆分为：

```text
Scheduler
Prefill / Decode
Stack_and_Sample
Update_Request_State
```

RTX 5060 Laptop GPU 的采集结果：

- `Engine.step_0`：1340.8 ms，其中首个 Prefill 为 1107.6 ms。
- `Runtime Triggered Module Loading`：987.9 ms，首步主要是冷启动。
- `Engine.step_2~5`：13.3～15.7 ms。
- Scheduler：12～298 us，不是主要瓶颈。
- CUDA Self Time 合计约 0.42 ms，CPU Self Time 为 1483 ms。当前模型很小，端到端耗时主要来自模块加载、Python 循环和 Kernel Launch，GPU闲置严重。

Trace 文件：

```text
reports/traces/inference_engine.json
```

## 6. 复现命令

```bash
# 2 Block：触发 Preemption
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python evals/demo_inference_streaming.py

# 8 Block：无抢占基线
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python evals/demo_inference_streaming.py --num-blocks 8

# PagedAttention 正确性与性能
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python evals/bench_paged_attention.py

# Engine Timeline
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python evals/demo_inference_streaming.py --profile
```
