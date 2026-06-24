# Nano-LLM-Infra: 从算子到编译执行的 AI Infra 实战路线图

> **目标**：不追求大而全，旨在通过 5 个核心模块打通 AI Infra 的关键技术闭环：**逻辑推导、高性能实现、图编译执行、量化评估、深度解释**。

---

## 🏗️ 总体架构原则
* **最小闭环优先 (MVP)**：拒绝过度工程，先跑通主逻辑，再逐步增强性能与抽象层次。
* **数据驱动优化**：所有优化都必须有 Benchmark 数据支持，不仅要快，还要说清为什么快。
* **工程规范化**：采用 `uv` + `PyBind11` + `Triton` + `torch.fx` 的现代工业界标准布局。
* **执行链路完整**：项目目标不是只做单点优化，而是打通 `Graph -> IR/Pass -> Kernel/Runtime -> Benchmark` 的核心链路。

---

## 🛠️ 模块一：CUDA & Triton Kernels（算子性能工程）
**目标**：理解 GPU 访存边界，解决算子融合过程中的显存带宽瓶颈。

### 1. Fused RMSNorm（核心项）
能讲reduction / warp shuffle / memory traffic
* **实现路径**：从 Shared Memory 归约进化到 Warp-shuffle 优化。
* **DoD**：
    * **正确性**：`torch.allclose` 对比原版，FP16 误差可解释。
    * **性能**：提供 `benchmark/bench_rmsnorm.py`，量化 HBM 读写次数减少带来的 Speedup。
* **进阶 todo**：
    - [ ] 实现 Triton 版本并进行对比测试。
    - [ ] 实现 LayerNorm 版本，分析额外引入的同步与统计开销。

### 2. Mini-FlashAttention（可后置）
* **核心逻辑**：不物化 $O(T^2)$ 矩阵，实现 Tiling + Online Softmax。
* **DoD**：
    * **正确性**：输出与 PyTorch reference 对齐。
    * **性能**：量化不同 `T, D, head_dim` 下的吞吐差异。
* **面试考点**：
    * SRAM 限制
    * Tiling 逻辑推导
    * Online Softmax 的数值稳定性
    * 算子融合的收益分析

---

## 🚀 模块二：Nano-vLLM 推理引擎内核 (Inference Engine Core)
**目标**：剥离分布式与量化的复杂性，用 Python/Triton 最小闭环还原大模型高并发推理的“铁三角”架构（PagedAttention + Continuous Batching + 显存池），攻克显存碎片化与吞吐量瓶颈。

### 1. 显存块管理器与缓存池 (Block Memory Manager & KV Pool)
* **核心逻辑**：打破动态显存分配（malloc/free）的魔咒，实现“开局梭哈，池化管理”。在 Python 层面实现虚拟 Token 索引到物理 GPU Block 的映射状态机。
* **实现路径**：
    * 预分配全局级大 Tensor 作为 KV Cache 物理内存池。
    * 实现 `BlockAllocator`：维护空闲队列（Free Block Queue），支持 O(1) 复杂度的 Block ID 借出与回收。
    * 构建请求级别的 `BlockTable`，记录逻辑块到物理块的映射关系。
* **DoD（完成标准）**：
    * **零碎片/零泄漏**：压测证明在长短不一的并发请求下，不产生外部显存碎片；长时运行后空闲队列长度能完全恢复。
* **面试考点**：
    * 为什么不用 PyTorch 自带的 Tensor 拼接？（显存碎片与动态分配的 CUDA 同步开销）
    * 逻辑块（Logical Block）和物理块（Physical Block）的映射关系是如何维护的？

### 2. PagedAttention 自定义算子 (Custom Kernel)
* **核心逻辑**：实现“数据面”的核心算子，让 Attention 机制能够看懂 BlockTable，在物理上不连续的内存块中完成连续的注意力计算。
* **实现路径**：
    * 基础级：使用 Triton 编写一个极简版 PagedAttention Kernel。传入 Q、全局 K/V Pool 指针以及 BlockTable，利用指针偏移（Pointer Arithmetic）实现跨 Block 的点乘与 Softmax。
    * 进阶级：使用cuda代码完成PagedAttention Kernel
* **DoD（完成标准）**：
    * 输出的 Logits 与标准连续内存 Attention 产生的结果在数值上严格对齐（精度误差 < 1e-3）。
* **面试考点**：
    * PagedAttention 在硬件（SM、SRAM）层面为什么能大幅提升访存带宽利用率（Memory Bandwidth Utilization）？
    * Triton/CUDA 算子开发中，如何通过 Block Table 进行非连续内存的寻址？

### 3. Continuous Batching 调度器 (Iteration-level Scheduler)
* **核心逻辑**：打破 Request-level（等最长句子生成完）的傻瓜调度，实现细粒度的 Step-by-Step 状态机。
* **实现路径**：
    * 维护三个核心队列：Waiting（等待）、Running（执行中）、Preempted（被抢占）。
    * 实现 `step()` 逻辑：在每一次迭代中，根据显存池剩余容量，动态决定混入新请求（Prefill）或继续旧请求（Decode）。
    * 核心抢占机制（Preemption）：当显存耗尽时，安全地挂起最晚加入的请求，释放其物理块，确保整个 Batch 不发生 OOM。
* **DoD（完成标准）**：
    * 成功处理并发请求，并在显存承压时正确触发 Preemption 且最终成功恢复生成。
* **面试考点**：
    * Prefill（Compute Bound）和 Decode（Memory Bound）的瓶颈差异是什么？
    * 为什么 Continuous Batching 是提升总体吞吐量（Throughput）和降低首字延迟（TTFT）的决定性技术？

### 4. 引擎驱动执行流 (Engine Loop & Model Runner)
* **核心逻辑**：打造系统的主齿轮，将“调度排班”、“内存分配”与“模型前向传播”串联为一个极低延迟的飞轮。
* **实现路径**：
    * 实现 `Engine.step()`：向调度器索要当前批次（Batch） -> 向内存管理器要物理地址（Block Tables） -> 将数据送入包含 PagedAttention 的极简模型（如迷你 Llama）进行前向传播 -> 采样下一个 Token -> 将新 Token 写入 KV Pool 对应的空闲位。
* **DoD（完成标准）**：
    * 实现一个流式终端输出：能够同时向 Engine 发送 3 个长度不一的 Prompt，并在终端看到它们不阻塞地、同时一字一字蹦出来。

### 5. TODO
* 讲清楚分布式的逻辑，包括如何和ray结合

---

## 🏋️ 模块三：Training Acceleration（训练加速核心）
**目标**：涵盖工业界大模型训练加速工程师的核心知识点，打通分布式训练核心组件，深刻理解显存墙（Memory Wall）优化与极致的通信计算重叠。

### 1. Mini DDP Runtime & Comm-Compute Overlap
* **核心实现**：
    * 自己手写最小 DDP 训练闭环，理解 `autograd hook -> gradient bucketize -> NCCL allreduce -> average gradients`。
    * 在同一套 DDP runtime 中加入 `async all_reduce()` 与 CUDA stream overlap，展示一个 stream 做 backward 计算，另一个 stream 做通信。
* **DoD**：
    * 单机 2 卡训练跑通，可打印 gradient sync timeline。
    * Benchmark 单卡 vs 双卡 scaling efficiency。
    * 成功跑通 overlap 逻辑并提供耗时分析。
* **面试考点**：
    * DDP 为什么比 DataParallel 快？
    * Bucket 是干什么的？Bucket size 如何影响 overlap 效果？
    * Allreduce 为什么是训练通信核心？
    * Overlap 为什么难？如何避免 CUDA Stream 带来的脏读/脏写（Data Hazard）？

### 2. 3D 并行基础 (Data, Tensor & Pipeline Parallelism)
* **核心实现**：
    * 理解大模型必备的 **3D 并行（DP + TP + PP）**。这是 NVIDIA **Megatron-LM** 的核心打法（切分计算图）。
    * 手写一个极简的 Megatron 风格 TP MLP：`ColumnParallelLinear -> GELU -> RowParallelLinear`，讲清楚权重按哪一维切、输入输出在哪一步做 shard / gather / all-reduce。
* **DoD**：
    * 单机 2 卡跑通极简 TP MLP block，并验证与 dense reference 对齐。
* **面试考点**：
    * **[高频] Megatron 和 DeepSpeed 的核心区别是什么？**（Megatron 主打 3D 并行切计算图，需要侵入修改模型代码；DeepSpeed 主打 ZeRO 切存储，对用户更透明）。
    * 3D 并行中，DP、TP、PP 分别解决什么问题？
    * `ColumnParallelLinear` 和 `RowParallelLinear` 各自的 Forward / Backward 产生了什么通信？
    * 为什么大模型必须用 3D 并行？PP（流水线并行）中的 Bubble（气泡）是什么？如何通过 1F1B 调度来缓解？

### 3. AMP Mixed Precision Engine
* **核心实现**：实现 `autocast()` 和 `GradScaler()`。覆盖 FP16/BF16 compute、FP32 master weights、dynamic loss scaling。
* **必须解释**：为什么 AMP 快（Tensor Core 需要 FP16/BF16 tile compute path，不仅是“精度低所以快”）。
* **DoD**：
    * Benchmark：对比 FP32 baseline 与 AMP improved 在吞吐量（throughput）和显存占用（mem）上的差异。
* **面试考点**：
    * Underflow 为什么发生？Scaler 为什么能解决？
    * 为什么大模型训练更偏爱 BF16 而不是 FP16？

### 4. 高级显存优化 (Activation Checkpoint & Memory Pool)
* **核心实现**：基于 `torch.utils.checkpoint`，实现 forward 时不保存 activation，backward 时重新计算 forward。
* **DoD**：
    * 量化显存下降比例与时间增加比例。
* **面试考点**：
    * 什么是 Selective Recompute（选择性重计算）？为什么只重算 Attention 的某些部分收益更高？
    * Transformer 的显存峰值通常出现在哪里？

### 5. Mini ZeRO (Stage 1 到 Stage 3 的演进)
* **核心逻辑**：ZeRO 是 **Data Parallelism (DP) 的极致进化版（数据并行方向的优化）**，它不切分计算图，而是切分了每个 Rank 冗余存储的训练状态（Optimizer States、Gradients、Parameters）。它与 TP/PP 是正交且互补的。
* **核心实现**：不调用 DeepSpeed API，自己模拟实现 ZeRO-1（切分 Optimizer States）和 ZeRO-3 的核心 Hook（Forward 前 Fetch 参数，算完立刻 Release）。
* **DoD**：
    * 成功展示 optimizer state shard partition。
    * 模拟 ZeRO-3 的参数即时获取与释放机制。
* **面试考点**：
    * ZeRO 和 3D 并行（TP/PP）的区别是什么？（ZeRO 仍是 DP，所有卡最终都会算完一遍完整的前向和反向，只是不在显存里一直存着所有参数）。
    * 面试极强加分：准确写出 ZeRO-1/2/3 显存占用的数学公式。
    * ZeRO-3 中的 All-Gather 发生在什么时机？通信量和 DDP 相比有什么变化？

---

## 🧠 模块四：Mini AI Compiler（计算图编译核心

> 本模块专门强化 **AI Compiler 岗位能力**
> 构建一个最小但完整的编译链路
> `Graph -> IR -> Pass -> Lowering -> Codegen/Dispatch -> Run`

### 1. Graph Capture：基于 `torch.fx` 的前端图捕获
* **核心实现**：
    * 使用 `torch.fx.symbolic_trace` 捕获简单模型（如 `add -> rmsnorm -> linear`）
    * 构建统一的 Graph IR 节点表示
* **DoD**：
    * 能正确打印和遍历 Graph
    * 支持导出自定义中间表示（IR）
* **面试考点**：
    * 为什么需要 Graph IR
    * `torch.fx` 捕获的本质是什么
    * 前端图与后端执行之间如何衔接

### 2. Transformation Pass：算子融合与模式匹配
* **核心实现**：
    * 编写一个最小 Pass Manager
    * 实现 `Add + RMSNorm` 或 `MatMul + GELU` 的 pattern matching 与融合
* **DoD**：
    * 提供融合前后 IR 对比
    * 解释融合带来的访存收益与中间张量消除
* **面试考点**：
    * Pattern Matching 的实现方式
    * 为什么融合能减少访存
    * Pass Pipeline 是如何组织的

### 3. Lowering：从 Graph IR 到 Loop/Kernel 级表示
* **核心实现**：
    * 将高层算子逐步 Lower 到更接近执行的形式
    * 对于 `matmul`，至少展示其 Lower 为 loop nest / tile 结构的过程
* **DoD**：
    * 能打印 lowering 前后表示
    * 对至少一个算子给出清晰的 lowering 规则
* **面试考点**：
    * 什么是 lowering
    * Graph IR 和 Loop IR 的区别
    * 为什么 lowering 是编译链路中的关键步骤

### 4. Codegen / Dispatch：生成可运行执行计划
* **核心实现**：
    * 最小方案：根据 lowering 结果派发到已有 CUDA/Triton Kernel
    * 进阶方案：对部分算子生成 Python/CUDA 风格伪代码
* **DoD**：
    * 整条链路可以真正跑通
    * 输入一个简单模型后，能完成 `捕获 -> 融合 -> lowering -> dispatch -> 执行`
* **面试考点**：
    * Codegen 和 Dispatch 的区别
    * 为什么很多 AI 编译器不直接生成机器码，而是生成调用计划
    * 编译器与 Runtime 的边界在哪里

---

## ⚙️ 模块五：Compiler Toolchain & Profiling（工具链与分析）
**目标**：展示自动化调优、性能归因与端到端分析能力，让项目更接近真实 AI 编译器 / 推理系统工程。

### 1. Kernel Autotuner（自动化调优）
* **核心实现**：
    * 针对自定义 CUDA/Triton Kernel 编写自动调优脚本
    * 扫描 Tile Size、Warp 数量、Block 配置等参数
* **DoD**：
    * 提供 `tools/autotune.py`
    * 支持针对不同输入 Shape 自动选择最优配置
    * 输出最优配置表与性能对比结果
* **面试考点**：
    * 为什么同一个 Kernel 在不同 Shape/GPU 上最优配置不同
    * Autotune 的搜索空间如何设计

### 2. End-to-End Profiling（端到端性能分析）
* **核心实现**：
    * 集成 `nvtx` 标记
    * 配合自定义 Profiler 采集 Kernel 执行、显存拷贝与 CPU 调度延迟
* **DoD**：
    * 导出符合 Chrome Trace 标准的分析文件
    * 能量化说明端到端加速中，哪些来自计算优化，哪些来自访存优化，哪些来自调度改进
* **面试考点**：
    * 如何判断瓶颈在 Kernel、Memory 还是 Runtime
    * 为什么端到端加速不等于单算子加速

### 3. Compiler Trace Visualization（可选增强项）
* **核心实现**：
    * 记录 Graph Capture、Pass、Lowering、Dispatch 各阶段的时间与结果
    * 输出一份可视化编译日志
* **DoD**：
    * 能从输入模型一路追踪到最终执行计划
    * 支持调试每个 Pass 对 IR 的修改
* **面试考点**：
    * 编译器调试为什么困难
    * IR 可视化对编译器开发的价值是什么

---

## 📊 总验收指标（DoD）
1. **一键运行**：所有模块支持 `uv run` 或 `python setup.py` 自动化部署与测试。
2. **量化报告**：`reports/` 目录下拥有完整的性能对比图表（Speedup、Memory Saving、Latency Breakdown）。
3. **编译链路可解释**：能清晰展示一个简单模型从 `torch.fx Graph` 到 `Fusion Pass`、再到 `Lowering / Dispatch` 的全过程。
4. **技术博客 / 白板**：能清晰画出：
   * PagedAttention 的逻辑映射图
   * RMSNorm 的归约逻辑
   * 一个简单计算图的融合与 lowering 过程

---
