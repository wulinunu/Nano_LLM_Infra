# Mini RL Infra 测试报告

## 1. 结论

该 Demo 已跑通以下闭环：

```text
Rollout -> Reward -> GRPO Train -> Weight Sync
```

核心结果：

- 双卡连续运行 20 步，Policy Version 从 1 增长到 20，权重一致性检查全部通过。
- 稳定阶段显存保持在 `18.3 MB`，未发现随 Step 增长的显存占用。
- 双卡吞吐从 `58.95` 提升到 `79.06 samples/s`，提升 `34.1%`。
- GPU Direct Weight Sync 比 CPU 中转快 `42%~50%`。
- Sequence Packing 减少 `40.6%` 的 Padding Token。
- 多进程 Reward 获得 `1.75~1.93x` 加速。

## 2. 测试配置

- 环境：单机双 GPU、Ray、NCCL
- 模型：2 层 Transformer，Hidden Size 64，4 个 Attention Head
- 输入：4 个 Prompt，每个 Prompt 采样 4 条 Response
- 每步样本数：16
- Response Length：8
- KV Block 数量：256
- Benchmark：预热 3 步，统计后续 10 步

单卡和双卡使用相同的模型与样本量。

## 3. 端到端稳定性

双卡连续运行 20 步：

```text
step=0   version=1   memory=18.3MB
step=10  version=11  memory=18.3MB
step=19  version=20  memory=18.3MB
```

- 每一步均完成 Rollout、Reward、Train 和 Weight Sync。
- Training Policy 与 Rollout Policy 每次同步后参数一致。
- Loss、KL、Grad Norm 均为有限值，没有出现 NaN 或 Inf。
- 首步包含 CUDA、NCCL 初始化开销；预热后 Rollout 通常约 `120~146 ms`，Train 约 `40~60 ms`。

## 4. 多卡性能

```text
单卡：avg_step=271.4ms  throughput=58.95 samples/s  peak_blocks=52
双卡：avg_step=202.4ms  throughput=79.06 samples/s  peak_blocks=28
```

双卡结果：

- Step Time 降低 `25.4%`。
- 吞吐提升 `34.1%`，实际加速比为 `1.34x`。
- 单卡 KV Block 峰值从 52 降至 28，说明负载已被两个 Worker 分摊。

模型和 Batch 较小，Ray、NCCL 与进程通信等固定开销占比较高，因此双卡没有达到线性加速。

## 5. 三项优化

### 5.1 GPU Weight Sync

```text
单卡：GPU 1.20ms  CPU 2.08ms  延迟降低 42.3%
双卡：GPU 1.34ms  CPU 2.66ms  延迟降低 49.6%
```

GPU Buffer 直接同步避免了 `GPU -> CPU -> GPU` 中转。当前模型参数量很小，该测试主要验证同步路径。

### 5.2 Sequence Packing

```text
packed_tokens=38
padded_tokens=64
padding_ratio=40.6%
cross_sequence_attention=blocked
```

Packing 少处理 26 个 Padding Token，并通过分段 Causal Mask 阻断不同 Experience 之间的 Attention。

### 5.3 并发 Reward

```text
单卡：406.6ms -> 210.9ms，1.93x
双卡：383.0ms -> 218.6ms，1.75x
```

进程池并行计算同一 Batch 内多条 Experience 的 Reward。训练仍会等待全部 Reward 返回，不属于跨 Batch 异步训练。

## 8. 复现命令

双卡稳定性测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python evals/rl_demo.py \
  --num-workers 2 --steps 20
```

单卡 Benchmark：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python evals/rl_demo.py \
  --num-workers 1 --benchmark
```

双卡 Benchmark：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python evals/rl_demo.py \
  --num-workers 2 --benchmark
```
