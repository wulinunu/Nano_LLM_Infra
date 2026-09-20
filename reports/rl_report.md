# Mini RL Infra 测试报告

## 1. 测试目标

本次测试关注真实运行数据能否证明 Mini RL Infra 的核心闭环有效：

1. `Rollout -> Reward -> GRPO Train -> Weight Sync` 能否连续运行。
2. 双卡 WorkerGroup 能否正确切分负载并提升吞吐。
3. KV Cache 是否正常分配和释放，显存是否稳定。
4. GPU 权重同步、Sequence Packing 和并发 Reward 是否产生实际收益。

本测试用于验证框架机制，不用于证明模型已经收敛，也不代表工业级 RL 集群性能。

## 2. 测试配置

- 环境：单机双 GPU、Ray、NCCL
- Prompt 数量：4
- Group Size：4
- 每步样本数：16
- Response Length：8
- 模型：2 层、Hidden Size 64、4 个 Attention Head
- KV Block 数量：256
- Benchmark：预热 3 步，统计后续 10 步平均值

单卡和双卡使用完全相同的样本量与模型配置。

## 3. 端到端稳定性

双卡连续运行 20 步，关键现象如下：

```text
step=0   version=1   memory=18.3MB
step=10  version=11  memory=18.3MB
step=19  version=20  memory=18.3MB
```

测试过程中：

- `policy_version` 从 1 连续增长到 20，说明每一步都完成了训练和权重同步。
- 每次同步后都会检查 Training Policy 与 Rollout Policy 参数完全一致；20 步全部完成，说明一致性检查均通过。
- 稳定阶段显存始终为 `18.3 MB`，没有随 Step 增长，未发现明显显存泄漏。
- Loss、KL 和 Grad Norm 均为有限值，没有出现 NaN、Inf 或训练中断。

第 0 步 Rollout 耗时 `3191.6 ms`，明显高于后续步骤，这是 CUDA、NCCL 和算子首次初始化造成的冷启动开销。预热后 Rollout 多数位于约 `120~146 ms`，Reward 多数位于 `5~7 ms`，ZeRO-2 Train 多数位于 `40~60 ms`。

Rollout 偶尔出现 `200 ms` 以上波动，因此 Benchmark 不采用单步结果，而是预热后统计 10 步平均值。

这些数据说明框架可以持续完成采样、奖励计算、反向更新和权重回传，并且资源状态保持稳定。

## 4. 单卡与双卡性能

单卡实测：

```text
avg_step=271.4ms
samples/s=58.95
kv_pool=1.00MB
peak_blocks=52
zero2_memory=19.1MB
```

双卡实测：

```text
avg_step=202.4ms
samples/s=79.06
kv_pool=2.00MB
peak_blocks=28
zero2_memory=18.3MB
```

在每步都处理 16 个样本的前提下：

- 双卡平均 Step Time 从 `271.4 ms` 降至 `202.4 ms`，降低约 `25.4%`。
- 吞吐从 `58.95 samples/s` 提升至 `79.06 samples/s`，提升约 `34.1%`。
- 双卡相对单卡的实际加速比约为 `1.34x`。
- 每张卡的 KV Block 峰值从单卡的 52 降至双卡的 28，说明 Prompt 确实被 WorkerGroup 分摊。

双卡没有达到线性加速。当前模型和 Batch 很小，Ray 调度、NCCL Collective、进程通信和阶段切换等固定开销占比较高。尽管如此，固定负载下仍获得了明确的延迟下降和吞吐提升，证明多 GPU WorkerGroup 的切分与执行是有效的。

双卡 `kv_pool=2.00MB` 是两张卡 KV Pool 的总和，单卡为 `1.00MB`。它反映的是每个 Worker 都拥有独立 KV Pool，并不表示双卡单设备显存压力更大。

## 5. Weight Sync

单卡结果：

```text
GPU Direct Sync: 1.20ms
CPU State Dict Sync: 2.08ms
CPU Copy: 0.41MB
```

双卡结果：

```text
GPU Direct Sync: 1.34ms
CPU State Dict Sync: 2.66ms
CPU Copy: 0.81MB
```

单卡 GPU Direct Sync 相比 CPU 中转降低约 `42.3%` 延迟，双卡降低约 `49.6%`。双卡 CPU Copy 接近单卡的两倍，符合两个 Worker 各复制一份模型参数的预期。

结果证明，将 Training Policy 参数直接通过 GPU Buffer 复制到 Rollout Policy，可以避免 `GPU -> CPU -> GPU` 中转。由于当前模型只有约 0.4 MB，这里主要验证同步路径有效；模型增大后，避免 Host Copy 的意义会更加明显。

## 6. Sequence Packing

实测结果：

```text
packed_tokens=38
padded_tokens=64
padding_ratio=40.6%
cross_sequence_attention=blocked
```

如果按最长序列 Padding，需要处理 64 个 Token；Packing 后只处理 38 个 Token，减少了 26 个无效 Token，即降低 `40.6%`。

测试同时检查了第一条序列末尾与第二条序列开头之间的 Attention Mask，结果为 blocked。这说明 Packing 在减少计算量的同时，没有破坏不同 Experience 之间的序列边界。

## 7. 并发 Reward

单卡实验：

```text
串行 Reward: 406.6ms
进程池 Reward: 210.9ms
speedup=1.93x
```

双卡实验：

```text
串行 Reward: 383.0ms
进程池 Reward: 218.6ms
speedup=1.75x
```

两次独立测试都获得了明显加速，说明 CPU 密集型 Reward 可以通过进程池并行执行。加速未达到 4 倍，主要受进程调度、参数传递和结果收集开销影响。

这里的“并发”只发生在当前 Batch 的多条 Experience 之间。Training 仍需等待全部 Reward 返回，因此结果证明的是 Reward 并行计算有效，而不是跨 Batch 的异步训练。

## 8. 训练数值说明

20 步运行中，Reward 在 `0.05~0.21` 之间波动，KL 从接近 0 增长到约 `0.37`。

当前 Reward 依赖随机采样，模型和样本规模都很小，因此 Reward 不会稳定单调上升。KL 增长表示 Training Policy 正逐渐偏离冻结的 Reference Policy，这符合参数持续更新后的预期。

这些数值只能证明 GRPO Loss、Reference KL 和参数更新实际参与了执行，不能作为模型收敛或训练效果良好的证据。

## 9. 结论

真实测试数据验证了该框架的核心有效性：

1. 双卡连续执行 20 步，Policy Version 持续更新，权重一致性检查通过。
2. 显存稳定在 `18.3 MB`，KV Cache 生命周期没有表现出明显泄漏。
3. 双卡相对单卡将吞吐提升约 `34.1%`，并降低了单卡 KV Block 压力。
4. GPU Direct Weight Sync 相比 CPU 中转降低约 `42%~50%` 延迟。
5. Sequence Packing 减少 `40.6%` 的 Padding Token，并正确阻断跨序列 Attention。
6. 多进程 Reward 相比串行实现获得 `1.75~1.93x` 加速。

因此，本次测试能够证明 Mini RL Infra 的控制流、分布式 Worker、KV Cache、GRPO Training、ZeRO-2 和权重同步已经形成可运行闭环。当前限制主要是模型和负载较小，性能结果用于说明机制与趋势，不应外推到生产级大模型训练。

## 10. 原始测试命令与输出

以下省略 Ray 启动日志。

### 10.1 双卡稳定性测试

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python evals/rl_demo.py \
  --num-workers 2 --steps 20
```

```text
workers=2 device=cuda backend=nccl flow=rollout->reward->train->sync
step=0 phase=sync version=1 reward=0.120 loss=0.0000 kl=0.00000 grad_norm=0.8720 memory=18.3MB
  KV allocate: pool=2.00MB allocated=2.4MB
  Continuous rollout: 3191.6ms peak_blocks=28
  KV release: 1.00MB
  Reward: 68.3ms
  ZeRO-2 train: 629.7ms backward_memory=17.9MB
  Weight sync: 0.8ms
step=1 phase=sync version=2 reward=0.166 loss=0.0008 kl=0.02108 grad_norm=0.8706 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 169.9ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.3ms
  ZeRO-2 train: 57.3ms backward_memory=18.3MB
  Weight sync: 1.8ms
step=2 phase=sync version=3 reward=0.110 loss=0.0015 kl=0.03867 grad_norm=0.8859 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 127.6ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.4ms
  ZeRO-2 train: 51.3ms backward_memory=18.3MB
  Weight sync: 2.0ms
step=3 phase=sync version=4 reward=0.139 loss=0.0025 kl=0.06136 grad_norm=0.8473 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 124.4ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.2ms
  ZeRO-2 train: 41.9ms backward_memory=18.3MB
  Weight sync: 1.4ms
step=4 phase=sync version=5 reward=0.084 loss=0.0040 kl=0.09897 grad_norm=0.8300 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 120.0ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.2ms
  ZeRO-2 train: 43.5ms backward_memory=18.3MB
  Weight sync: 1.4ms
step=5 phase=sync version=6 reward=0.080 loss=0.0045 kl=0.11194 grad_norm=0.9643 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 235.8ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.1ms
  ZeRO-2 train: 50.7ms backward_memory=18.3MB
  Weight sync: 1.9ms
step=6 phase=sync version=7 reward=0.095 loss=0.0046 kl=0.11572 grad_norm=0.8764 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 126.8ms peak_blocks=28
  KV release: 1.00MB
  Reward: 5.1ms
  ZeRO-2 train: 39.8ms backward_memory=18.3MB
  Weight sync: 1.0ms
step=7 phase=sync version=8 reward=0.071 loss=0.0051 kl=0.12662 grad_norm=0.8809 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 114.4ms peak_blocks=28
  KV release: 1.00MB
  Reward: 5.3ms
  ZeRO-2 train: 39.4ms backward_memory=18.3MB
  Weight sync: 0.9ms
step=8 phase=sync version=9 reward=0.088 loss=0.0072 kl=0.17929 grad_norm=0.8538 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 140.5ms peak_blocks=28
  KV release: 1.00MB
  Reward: 5.9ms
  ZeRO-2 train: 48.2ms backward_memory=18.3MB
  Weight sync: 1.6ms
step=9 phase=sync version=10 reward=0.051 loss=0.0060 kl=0.15077 grad_norm=0.8297 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 126.6ms peak_blocks=28
  KV release: 1.00MB
  Reward: 5.6ms
  ZeRO-2 train: 61.8ms backward_memory=18.3MB
  Weight sync: 2.1ms
step=10 phase=sync version=11 reward=0.155 loss=0.0063 kl=0.15776 grad_norm=0.8147 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 204.7ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.8ms
  ZeRO-2 train: 60.9ms backward_memory=18.3MB
  Weight sync: 1.9ms
step=11 phase=sync version=12 reward=0.208 loss=0.0076 kl=0.19058 grad_norm=0.8485 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 130.2ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.5ms
  ZeRO-2 train: 43.6ms backward_memory=18.3MB
  Weight sync: 1.3ms
step=12 phase=sync version=13 reward=0.073 loss=0.0078 kl=0.19452 grad_norm=0.8351 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 126.4ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.4ms
  ZeRO-2 train: 43.4ms backward_memory=18.3MB
  Weight sync: 1.2ms
step=13 phase=sync version=14 reward=0.120 loss=0.0079 kl=0.19627 grad_norm=0.8328 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 122.1ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.6ms
  ZeRO-2 train: 42.3ms backward_memory=18.3MB
  Weight sync: 3.5ms
step=14 phase=sync version=15 reward=0.090 loss=0.0084 kl=0.20895 grad_norm=0.8128 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 145.7ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.0ms
  ZeRO-2 train: 57.4ms backward_memory=18.3MB
  Weight sync: 2.4ms
step=15 phase=sync version=16 reward=0.159 loss=0.0111 kl=0.27740 grad_norm=0.8231 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 124.6ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.6ms
  ZeRO-2 train: 43.0ms backward_memory=18.3MB
  Weight sync: 1.2ms
step=16 phase=sync version=17 reward=0.181 loss=0.0112 kl=0.27879 grad_norm=0.7781 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 226.9ms peak_blocks=28
  KV release: 1.00MB
  Reward: 5.1ms
  ZeRO-2 train: 44.0ms backward_memory=18.3MB
  Weight sync: 1.5ms
step=17 phase=sync version=18 reward=0.121 loss=0.0101 kl=0.25153 grad_norm=0.8316 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 120.2ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.1ms
  ZeRO-2 train: 46.4ms backward_memory=18.3MB
  Weight sync: 1.6ms
step=18 phase=sync version=19 reward=0.078 loss=0.0110 kl=0.27452 grad_norm=0.8190 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 127.3ms peak_blocks=28
  KV release: 1.00MB
  Reward: 5.7ms
  ZeRO-2 train: 47.9ms backward_memory=18.3MB
  Weight sync: 1.8ms
step=19 phase=sync version=20 reward=0.065 loss=0.0147 kl=0.36767 grad_norm=0.8712 memory=18.3MB
  KV allocate: pool=2.00MB allocated=19.3MB
  Continuous rollout: 129.1ms peak_blocks=28
  KV release: 1.00MB
  Reward: 6.6ms
  ZeRO-2 train: 44.8ms backward_memory=18.3MB
  Weight sync: 1.6ms
```

### 10.2 单卡 Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python evals/rl_demo.py \
  --num-workers 1 --benchmark
```

```text
workers=1 device=cuda backend=nccl flow=rollout->reward->train->sync
[flow] kv_pool=1.00MB peak_blocks=52 kv_released=1.00MB zero2_memory=19.1MB version=13
[colocation] avg_step=271.4ms samples/s=58.95
[weight_sync] gpu=1.20ms cpu=2.08ms cpu_copy=0.41MB
[packing] packed=38 padded=64 padding_ratio=40.6% cross_sequence_attention=blocked
[reward] sync=406.6ms async=210.9ms speedup=1.93x
```

### 10.3 双卡 Benchmark

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python evals/rl_demo.py \
  --num-workers 2 --benchmark
```

```text
workers=2 device=cuda backend=nccl flow=rollout->reward->train->sync
[flow] kv_pool=2.00MB peak_blocks=28 kv_released=1.00MB zero2_memory=18.3MB version=13
[colocation] avg_step=202.4ms samples/s=79.06
[weight_sync] gpu=1.34ms cpu=2.66ms cpu_copy=0.81MB
[packing] packed=38 padded=64 padding_ratio=40.6% cross_sequence_attention=blocked
[reward] sync=383.0ms async=218.6ms speedup=1.75x
```
