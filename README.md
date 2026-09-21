# Nano-LLM-Infra

一个用于学习和验证 LLM 基础设施核心机制的最小实现。

项目使用 PyTorch、CUDA、Triton、NCCL 和 Ray，覆盖分布式训练、推理引擎、自定义算子、RL 训练流与 AI 编译器。重点不是提供生产框架，而是用可运行、可测试的代码解释这些系统如何工作。

## 项目能力

### 分布式训练

- Data Parallel：Gradient Bucket、NCCL AllReduce、通信计算重叠
- 混合精度：FP16、BF16、Dynamic Loss Scaling
- Tensor Parallel：Column / Row Parallel Linear
- Pipeline Parallel：Naive、GPipe、1F1B
- Expert Parallel：Token Dispatch、All-to-All、Capacity 控制
- Context Parallel：Ring Attention
- ZeRO-1 / 2 / 3：Optimizer、Gradient、Parameter Sharding
- Activation Checkpoint：用重计算降低激活显存

### 推理引擎

- Paged KV Cache 与 Block Table
- Continuous Batching
- Waiting / Running / Preempted 请求状态机
- KV Block 不足时的抢占、恢复与回收
- Tiny Transformer 的 Prefill / Decode 执行路径
- PyTorch、Triton 与 CUDA PagedAttention 实现

### GPU 算子

- CUDA RMSNorm：Shared Memory 与 Warp Shuffle 两种归约实现
- Triton Mini-FlashAttention：二维 Tiling 与 Online Softmax
- Triton / CUDA PagedAttention：通过 Block Table 读取非连续 KV Cache

### RL Infra

实现一个 veRL 风格的轻量 Colocated GRPO 流程：

```text
Rollout -> Reward -> GRPO Train -> Weight Sync
```

- Ray WorkerGroup 负责多 GPU 资源调度和数据切分
- 同一 GPU 分时运行 Rollout Policy 与 Training Policy
- Rollout 复用 NanoEngine、Continuous Batching 和 KV Cache
- Training 复用 ZeRO-2
- Sequence Packing 保留序列边界和 GRPO Group
- GPU Direct Weight Sync 避免 CPU 中转
- 进程池并发计算 Reward

### Mini AI Compiler

实现一条最小但完整的编译链路：

```text
PyTorch FX
  -> Custom Graph IR
  -> Fusion / Memory Planning
  -> Kernel IR
  -> Triton Codegen
  -> Dispatch
```

当前示例捕获 `Add -> RMSNorm -> Linear`，融合 RMSNorm 子图，复用中间 Buffer，并生成可执行的 Triton Kernel。

## 整体结构

```text
Nano_LLM_Infra/
├── csrc/
│   ├── paged_attention/       # CUDA PagedAttention
│   └── rmsnorm/               # CUDA RMSNorm
├── evals/
│   ├── train.py               # 分布式训练统一入口
│   ├── demo_inference_streaming.py
│   ├── rl_demo.py
│   ├── compiler_demo.py
│   └── bench_*.py             # 算子 Benchmark
├── reports/                   # 测试、性能与 Profiling 报告
└── src/nano_llm_infra/
    ├── models/                # 训练、推理和 RL 共用的小模型
    ├── training/              # AMP 与分布式训练 Runtime
    ├── inference/             # KV Cache、Scheduler、Engine
    ├── ops/                   # PyTorch / Triton / CUDA 算子接口
    ├── rl/                    # GRPO Controller、Worker、Packing
    └── compiler/              # IR、Pass、Lowering、Codegen
```

## 核心数据流

训练：

```text
TinyTransformer
  -> DP / TP / PP / EP / CP
  -> AMP / Activation Checkpoint
  -> ZeRO Optimizer
```

推理：

```text
Request
  -> Iteration-level Scheduler
  -> BlockAllocator / KVCachePool
  -> Prefill / Decode
  -> PagedAttention
  -> Sampling
```

RL：

```text
RLController
  -> WorkerGroup
  -> ColocatedWorker
       ├── rollout_policy + NanoEngine
       ├── training_policy + ZeRO-2
       └── reference_policy
```

## 环境安装

要求：

- Python 3.11+
- CUDA GPU 与 CUDA Toolkit
- 支持 NCCL 的多卡环境（运行分布式示例时）

使用 `uv` 安装全部依赖：

```bash
uv sync
```

项目会编译 CUDA RMSNorm 和 PagedAttention 扩展。`setup.py` 默认设置 `TORCH_CUDA_ARCH_LIST=12.0`，其他架构可在安装前覆盖：

```bash
TORCH_CUDA_ARCH_LIST="8.0" uv sync
```

以下命令默认在项目根目录执行。

## 快速运行

### 分布式训练

统一入口支持 `dp / tp / pp / ep / cp / zero`：

```bash
torchrun --nproc_per_node=2 evals/train.py --mode dp
torchrun --nproc_per_node=2 evals/train.py --mode tp
torchrun --nproc_per_node=2 evals/train.py --mode pp --pp-schedule 1f1b
torchrun --nproc_per_node=2 evals/train.py --mode ep
torchrun --nproc_per_node=2 evals/train.py --mode cp
torchrun --nproc_per_node=2 evals/train.py --mode zero --zero-stage 2
```

### 推理引擎

```bash
PYTHONPATH=src python evals/demo_inference_streaming.py
PYTHONPATH=src python evals/demo_inference_streaming.py --num-blocks 2 --profile
```

### 算子 Benchmark

```bash
PYTHONPATH=src python evals/bench_rmsnorm.py
PYTHONPATH=src python evals/bench_flash_attention.py
PYTHONPATH=src python evals/bench_paged_attention.py
```

### RL Demo

```bash
# 单卡运行
PYTHONPATH=src python evals/rl_demo.py

# 双卡运行
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
  python evals/rl_demo.py --num-workers 2

# 双卡 Benchmark
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
  python evals/rl_demo.py --num-workers 2 --benchmark
```

### Mini Compiler

```bash
PYTHONPATH=. python evals/compiler_demo.py
```


## 测试报告

- [分布式训练](reports/training_framework_report.md)
- [推理引擎](reports/inference_engine_report.md)
- [RMSNorm Benchmark](reports/rmsnorm_benchmark.md)
- [FlashAttention Benchmark](reports/flash_attention_benchmark.md)
- [Profiling](reports/profiling_report.md)
- [RL Infra](reports/rl_report.md)
- [Mini Compiler](reports/mini_compiler_report.md)
