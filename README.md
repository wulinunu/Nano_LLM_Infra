# Nano-LLM-Infra

> **目标**：不追求大而全，旨在通过 5 个核心模块打通 AI Infra 的关键技术闭环

---

## 🏗️ 总体架构原则
* **最小闭环优先 (MVP)**：拒绝过度工程，先跑通主逻辑，再逐步增强性能与抽象层次。
* **benchmark验证**：所有优化都必须有 Benchmark 数据支持，不仅要快，还要说清为什么快。

---

## 🛠️ 模块一：算子性能工程
**目标**：理解 GPU 访存边界，解决算子融合过程中的显存带宽瓶颈。

### 1. Fused RMSNorm（核心项）
能讲reduction / warp shuffle / memory traffic
* **实现路径**：从 Shared Memory 归约进化到 Warp-shuffle 优化。
* **DoD**：
    * **正确性**：`torch.allclose` 对比原版，FP16 误差可解释。
    * **性能**：提供 `benchmark/bench_rmsnorm.py`，量化 HBM 读写次数减少带来的 Speedup。
    * **微观分析 (Micro Profiling)**：使用 `ncu` (Nsight Compute) 证明 Warp-shuffle 版本比 Shared Memory 版本具有更高的 SM 占用率 (Occupancy) 或更低的 Bank Conflict。

### 2. Mini-FlashAttention（可后置）
* **核心逻辑**：不物化 $O(T^2)$ 矩阵，实现 Tiling + Online Softmax。
* **DoD**：
    * **正确性**：输出与 PyTorch reference 对齐。
    * **性能 & 自动调优 (Autotuning)**：引入 `@triton.autotune` 自动搜索最优 `BLOCK_M/N` 等参数。
    * **微观分析 (Micro Profiling)**：使用 `ncu` 采集真实 Memory Throughput，证明相比 Reference 真正消除了 $O(T^2)$ 读写量。
* **面试考点**：
    * SRAM 限制与 Tiling 逻辑推导。
    * 为什么同一个 Kernel 在不同 Shape 下最优 Grid/Block 组织方式完全不同？(Autotune 的意义)。
    * Memory Bound (访存瓶颈) vs Compute Bound (计算瓶颈) 的本质区别是什么？Roofline 模型怎么看？

---

## 🚀 模块二：推理引擎内核 (Inference Engine Core)
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
    * **微观分析 (Micro Profiling)**：使用 `ncu` 分析在非连续物理内存寻址时的 L1/L2 Cache Hit Rate 变化与访存带宽代价。
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
    * **宏观分析 (Macro Profiling)**：使用 `torch.profiler` + `nvtx` 对 `Engine.step()` 埋点，抓取 Timeline，能够从 Chrome Trace 图中指出 CPU Launch Overhead 以及 GPU 饿死（Starvation）的时间段。
* **面试考点**：
    * CPU 发射延迟 (Launch Overhead) 是什么？如何隐藏它？
    * 怎样从 Trace 图中判断当前系统是 CPU-bound 还是 GPU-bound？

### 5. TODO
* 讲清楚分布式的逻辑，包括如何和ray结合

---

## 🏋️ 模块三：训练加速核心
**目标**：涵盖工业界大模型训练加速工程师的核心知识点，打通分布式训练核心组件，深刻理解显存墙（Memory Wall）优化与极致的通信计算重叠。

### 1. Mini DDP Runtime & Comm-Compute Overlap
* **核心实现**：
    * 自己手写最小 DDP 训练闭环，理解 `autograd hook -> gradient bucketize -> NCCL allreduce -> average gradients`。
    * 在同一套 DDP runtime 中加入 `async all_reduce()` 与 CUDA stream overlap，展示一个 stream 做 backward 计算，另一个 stream 做通信。
* **DoD**：
    * 单机 2 卡训练跑通，可打印 gradient sync timeline。
    * Benchmark 单卡 vs 双卡 scaling efficiency。
    * **宏观分析 (Macro Profiling)**：成功跑通 overlap 逻辑，并使用 `nsys` 抓取 Trace，直观验证 CUDA Stream 中 Compute (计算) 与 NCCL 通信的 Overlap (重叠) 效果。
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
    * **宏观分析 (Macro Profiling)**：使用 `nsys` 抓取执行流，可视化 Pipeline Parallelism 中的气泡（Bubble）时间占比，以及 Tensor Parallelism 中 AllReduce 带来的同步阻塞延迟。
* **面试考点**：
    * **[高频] Megatron 和 DeepSpeed 的核心区别是什么？**（Megatron 主打 3D 并行切计算图，需要侵入修改模型代码；DeepSpeed 主打 ZeRO 切存储，对用户更透明）。
    * 3D 并行中，DP、TP、PP 分别解决什么问题？
    * `ColumnParallelLinear` 和 `RowParallelLinear` 各自的 Forward / Backward 产生了什么通信？
    * 为什么大模型必须用 3D 并行？PP（流水线并行）中的 Bubble（气泡）是什么？如何通过 1F1B 调度来缓解？

### 3. Expert Parallelism & All-to-All（MoE 通信核心）
* **核心实现**：
    * 手写一个最小 MoE 层：`Router -> Token Dispatch -> Local Experts -> Token Combine`。
    * 将 Expert 分布到不同 Rank，使用 `all_to_all_single()` 完成 Token Dispatch 和结果回传。
    * 处理每个 Rank 接收 Token 数不同的问题，维护 Split Size 与 Token 原始位置。
* **DoD**：
    * 单机 2 卡跑通 EP，并验证输出与单卡 Dense MoE Reference 对齐。
    * 打印每个 Expert 的 Token 数量，观察负载是否均衡。
    * 使用 `nsys` 查看两次 All-to-All 的通信耗时，以及通信与 Expert Compute 的执行关系。
* **面试考点**：
    * EP 为什么使用 All-to-All，而 TP / DP 主要使用 AllReduce？
    * MoE 中 Token Dispatch 和 Token Combine 分别在传输什么？
    * Router 负载不均衡为什么会造成 Straggler？Capacity Factor 和 Auxiliary Loss 如何缓解？
    * EP 如何与 DP、TP、PP 组合？

### 4. AMP Mixed Precision Engine
* **核心实现**：实现 `autocast()` 和 `GradScaler()`。覆盖 FP16/BF16 compute、FP32 master weights、dynamic loss scaling。
* **必须解释**：为什么 AMP 快（Tensor Core 需要 FP16/BF16 tile compute path，不仅是“精度低所以快”）。
* **DoD**：
    * Benchmark：对比 FP32 baseline 与 AMP improved 在吞吐量（throughput）和显存占用（mem）上的差异。
* **面试考点**：
    * Underflow 为什么发生？Scaler 为什么能解决？
    * 为什么大模型训练更偏爱 BF16 而不是 FP16？

### 5. 高级显存优化 (Activation Checkpoint & Memory Pool)
* **核心实现**：基于 `torch.utils.checkpoint`，实现 forward 时不保存 activation，backward 时重新计算 forward。
* **DoD**：
    * 量化显存下降比例与时间增加比例。
* **面试考点**：
    * 什么是 Selective Recompute（选择性重计算）？为什么只重算 Attention 的某些部分收益更高？
    * Transformer 的显存峰值通常出现在哪里？

### 6. Mini ZeRO (Stage 1 到 Stage 3 的演进)
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

## 🧠 模块四：Mini AI Compiler（计算图编译核心）
> 构建一个最小但完整的端到端编译链路
> `Graph -> IR -> Pass (Fusion & Memory) -> Lowering -> Codegen (Triton) -> Run`

### 1. Graph Capture：基于 `torch.fx` 的前端图捕获
* **核心实现**：
    * 使用 `torch.fx.symbolic_trace` 捕获典型 Transformer 子块（`Add -> RMSNorm -> Linear`）。
    * 构建统一的 Graph IR 节点表示，提取 `OpType`、`Inputs`、`Outputs` 与 Tensor Meta（Shape / Dtype）。
* **DoD**：
    * 能正确打印、遍历并校验 Graph 的拓扑结构。
    * 支持将 `torch.fx.Graph` 导出为自定义统一中间表示Graph IR。
* **面试考点**：
    * 为什么 AI 编译器需要 Graph IR？相比 Eager 模式优势是什么？
    * `torch.fx` 捕获的本质是什么？`Tracer`、`Node`、`GraphModule` 的关系是什么？
    * `TorchDynamo` 与 `torch.fx.symbolic_trace` 在图捕获逻辑上的根本差异是什么？

### 2. Transformation Pass：算子融合与静态内存规划
* **核心实现**：
    * 编写一个最小 Pass Manager，支持 Pass 注册与按序执行。
    * **Fusion Pass**：实现 `Add + RMSNorm` 或 `MatMul + GELU` 的 Pattern Matching 与节点替换（Rewriting）。
    * **Memory Planning Pass**：对中间张量做生命周期分析（Liveness Analysis），实现可复用 Buffer 的静态内存规划。
* **DoD**：
    * 提供融合前后 IR 对比，验证中间张量消除（Intermediate Tensor Elimination）。（减少一次中间张量的显存写和显存读）
    * 输出内存规划报告，证明静态内存分配降低了峰值显存占用。
* **面试考点**：
    * Pattern Matching 的实现机制：AST 匹配 vs 图拓扑匹配。
    * 为什么算子融合能大幅减少访存开销？它如何优化 Memory Bandwidth Bound 问题？
    * 静态内存分配（Static Buffer Sharing）的算法思路与边界条件是什么？

### 3. Lowering：从 Graph IR 到 Kernel IR
* **核心实现**：
    * 将 `Add`、`Fused RMSNorm` 等高层 Graph 节点 Lower 为接近 Triton Kernel 的 `KernelIR`。
    * `KernelIR` 描述 Kernel 签名、Grid / Block 配置，以及 `Load -> Compute -> Store` 微指令。
* **DoD**：
    * 能打印 Graph IR 与 Kernel IR，直观看到算子如何转换为访存、计算和写回指令。
    * Codegen 实际使用 Kernel IR 中的 Grid / Block 信息生成并发配置。
    * 给出 `Add` 或 `Fused RMSNorm` 的完整 Lowering 映射。
* **面试考点**：
    * 什么是 Lowering？为什么编译器不一步到位直接从 Graph 到 Code？
    * Graph IR 与 Kernel IR 的职责分工和抽象层级差异是什么？
    * Elementwise 算子如何通过 Grid / Block 映射到 GPU 并行执行？

### 4. Codegen & Dispatch：Triton Kernel 动态生成与运行
* **核心实现**：
    * **Codegen**：根据 Lowering 后的 Kernel IR，动态组装可运行的 Triton Kernel 代码。
    * **Dispatch & Run**：编译生成出的 Triton 代码，构建 Dispatcher，建立 PyTorch Tensor 到 Triton Kernel 的输入映射并触发执行。
* **DoD**：
    * 整条 Mini Compiler 链路成功跑通：输入 PyTorch 模型后，依次完成 `捕获 -> 融合/内存优化 -> Lowering -> Triton Codegen -> Dispatch 执行`。
    * 通过 `torch.testing.assert_close` 验证编译后执行结果与 PyTorch Eager 模式数值对齐。
    * **宏观分析 (Macro Profiling)**：对比 Eager 模式与 Compiled 模式的 `nsys` / `torch.profiler` Trace，量化展示算子融合 (Fusion) 后，Kernel 发射间隙（Gap）的显著消除。
* **面试考点**：
    * Codegen 与 Dispatch 的区别是什么？
    * 为什么现代 AI 编译器（如 TorchInductor）倾向于生成 Triton / C++ 代码，而不是直接生成机器码？
    * AI 编译器（Compile-time）与运行时（Runtime）的边界在哪里？如何处理 Dynamic Shape？

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
