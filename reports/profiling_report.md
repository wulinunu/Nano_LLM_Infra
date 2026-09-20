# Profiling 工具使用总览

本项目不在每个模块重复使用 profiling 工具，而是让三种工具各回答一个最适合的问题。

## 1. Nsight Compute：RMSNorm Kernel 微观分析

**回答的问题：** Warp-shuffle 为什么比 Shared Memory 归约更高效？

采集对象：

```text
rmsnorm_shared
rmsnorm_warp_shuffle
```

关键结果：

- Shared Memory 指令总量从 `27,648` 降至 `9,216`，降低 `66.7%`。
- Shared Wavefront 从 `117,056` 降至 `37,641`，降低 `67.8%`。
- Bank Conflict 从 `494` 降至 `273`，降低 `44.7%`。
- NCU Kernel Duration 从 `69.98 us` 降至 `66.05 us`。

采集命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /usr/local/cuda/bin/ncu --set full --nvtx \
  --nvtx-include "rmsnorm_shared/" \
  --nvtx-include "rmsnorm_warp_shuffle/" \
  -o reports/ncu_rmsnorm -f \
  python evals/bench_rmsnorm.py --ncu-profile
```

详细数据见 `reports/rmsnorm_benchmark.md`。

当前容器若返回 `ERR_NVGPUCTRPERM`，需要在宿主机开启 Performance Counter 权限；这属于驱动权限，不是脚本错误。

## 2. torch.profiler：Inference Engine 宏观 Timeline

**回答的问题：** 一次 Engine Step 的 CPU 调度、Prefill、Decode 和 GPU Kernel 是如何排列的？

埋点范围：

```text
Engine.step_N
Scheduler
Prefill
Decode
Stack_and_Sample
Update_Request_State
```

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python evals/demo_inference_streaming.py --profile
```

脚本会打印 CUDA 算子摘要，并输出：

```text
reports/traces/inference_engine.json
```

判读重点：

- 首个 Step 是否主要消耗在 CUDA 冷启动，而不是稳定 Decode。
- CPU Range 之间是否存在明显 Launch Gap。
- CPU Gap 出现时 GPU 是否同时处于空闲状态。
- Prefill 与 Decode 的 Kernel 组成和持续时间是否不同。

本次 RTX 4090 采集结果：

- 首个 Engine Step 为 `228.8 ms`，其中模块加载自耗时 `159.7 ms`，属于冷启动。
- 第二步为 `11.5 ms`，后续稳定 Step 为 `1.82~1.95 ms`。
- 稳定 Scheduler 仅 `12~29 us`；6 步 CUDA Kernel Self Time 合计约 `0.49 ms`，而 CPU Self Time 为 `247.9 ms`。
- 当前 Tiny Model 的主要瓶颈是模块加载、Python 循环和 Kernel Launch，而不是 GPU 算力。

详细说明见 `reports/inference_engine_report.md`。

## 3. Nsight Systems：DP 通信计算重叠

**回答的问题：** Gradient Bucket 的 NCCL AllReduce 是否与后续 Backward Compute 重叠？

NVTX 范围：

```text
dp_forward
dp_backward
bucket_N_allreduce
grad_sync_wait
optimizer_step
```

采集命令：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
nsys profile --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o reports/nsys_dp_overlap \
  torchrun --standalone --nproc_per_node=2 evals/train.py \
  --mode dp --precision fp32 \
  --bucket-size-mb 0.05 --warmup 5 --steps 20 \
  --hidden-size 1024 --num-layers 4 --batch-size 8 --seq-len 256
```

本次 RTX 4090 双卡采集结果：

- 共采集 1350 个 NCCL AllReduce Kernel，其中 862 个与同卡计算 Kernel 重叠，占 `63.9%`。
- NCCL Kernel 总时长 `1102.397 ms`，重叠区间 `541.007 ms`，通信时间重叠率为 `49.1%`。
- `dp_backward` 平均 `28.458 ms`，`grad_sync_wait` 平均 `5.975 ms`，说明部分通信被隐藏，但仍存在尾部等待。
- NCCL 位于独立通信 Stream；AllReduce 占 GPU Kernel 总时长的 `46.8%`，当前 27 个小 Bucket 的通信启动开销仍较明显。

nsys 采集时平均单步为 `43.490 ms`，高于无 profiler 基线；该数值包含采集开销，不用于训练性能对比。

详细说明见 `reports/training_framework_report.md`。

## 4. 如何查看原始 Profiling 文件

原始文件应该在本地保留并实际查看，只是不提交到 Git。最终报告中的每个结论都应该能够在原始 Timeline 或硬件计数器中找到证据。

### 4.1 torch.profiler Trace

原始文件：

```text
reports/traces/inference_engine.json
```

打开方式：

1. 访问 `https://ui.perfetto.dev`。
2. 点击 `Open trace file`，选择 `inference_engine.json`。
3. 搜索 `Engine.step_0`、`Prefill` 或 `Decode`。

重点观察：

- CPU Thread 上 `Engine.step_N` 内部各用户 Range 的先后关系。
- CUDA Stream 上 Kernel 是否连续；空白区域表示 GPU 没有工作。
- CPU 发起 Kernel 与 GPU 开始执行之间是否存在明显间隔。
- `Engine.step_0` 与稳定 Step 的差异，区分冷启动和稳态性能。

面试时可以这样讲：

> 首步主要被 Runtime Module Loading 占据；稳定阶段 Scheduler 只有几十微秒，但 Tiny Model 的 GPU Kernel 很短，CPU Launch 和 Python 循环占比更高，因此 GPU 没有被充分喂满。

### 4.2 Nsight Compute `.ncu-rep`

本地产物：

```text
reports/ncu_rmsnorm.ncu-rep
```

GUI 打开：

```bash
/usr/local/cuda/bin/ncu-ui reports/ncu_rmsnorm.ncu-rep
```

CLI 查看：

```bash
/usr/local/cuda/bin/ncu --import reports/ncu_rmsnorm.ncu-rep \
  --page details
```

重点页面：

- `Summary`：Kernel Duration 和总体瓶颈。
- `Memory Workload Analysis`：Shared/L1/L2/DRAM 访问。
- `Scheduler Statistics`：Eligible Warps、Issue Slot 和 Stall。
- `Source Counters`：将高开销指标定位到 CUDA 源码。

比较 RMSNorm 两个 Kernel 时，应先确认 Shape 和 Launch 配置一致，再比较 Shared Instructions、Shared Wavefront、Bank Conflict 和 Duration，不能只看单个百分比。

面试时可以这样讲：

> Warp-shuffle 版本把 Warp 内归约从 Shared Memory 搬到寄存器 Shuffle，NCU 显示 Shared 指令减少 66.7%、Bank Conflict 减少 44.7%，硬件计数器与普通 Benchmark 的加速方向一致。

### 4.3 Nsight Systems `.nsys-rep`

本地产物：

```text
reports/nsys_dp_overlap.nsys-rep
```

GUI 打开：

```bash
nsys-ui reports/nsys_dp_overlap.nsys-rep
```

CLI 摘要：

```bash
nsys stats --report cuda_gpu_kern_sum,nvtx_sum \
  reports/nsys_dp_overlap.nsys-rep
```

GUI 中按以下顺序查看：

1. 展开两个 Rank 对应的进程。
2. 展开 `CUDA HW` 和各 CUDA Stream。
3. 搜索 NVTX Range `dp_backward` 与 `bucket_N_allreduce`。
4. 找到 NCCL AllReduce Kernel 所在的通信 Stream。
5. 检查 NCCL Kernel 的时间区间是否与默认计算 Stream 上的 Backward Kernel 横向重叠。
6. 查看 `grad_sync_wait` 是否只等待最后几个未完成 Bucket。

面试时可以这样讲：

> Autograd Hook 在 Bucket Ready 时把 AllReduce 发到独立通信 Stream。Nsight Systems 实测 63.9% 的 NCCL Kernel 与计算重叠，按通信时长计算重叠率为 49.1%；剩余尾部通信使 `grad_sync_wait` 平均仍有 5.975 ms。

## 5. 使用边界

- NCU 用于单 Kernel 硬件计数器，不用它分析端到端调度。
- torch.profiler 用于 PyTorch Operator 与用户 Range，不替代多进程 NCCL Timeline。
- Nsight Systems 用于跨 CPU、CUDA Stream 和进程的时间线，不用它解释单个 Kernel 的 Bank Conflict。
- `.ncu-rep`、`.nsys-rep` 和 Trace JSON 应在本地生成、打开和分析；因为体积较大且依赖运行环境，所以通过 `.gitignore` 排除，不提交仓库。
