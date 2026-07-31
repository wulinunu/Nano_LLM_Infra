from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from nano_llm_infra.inference.TinyTransformerModel import TinyMLP
from nano_llm_infra.training import (
    GradientReducer,
    AmpEngine,
    DataParallelRuntime,
    TinyTrainingTransformer,
    ZeroRuntime,
)
from nano_llm_infra.training.distributed.pipeline_parallel import (
    PipelineRuntime,
    build_pipeline_stage,
)
from nano_llm_infra.training.distributed.tensor_parallel import (
    TPMLP,
    check_tp_mlp,
    shard_dense_mlp_weights_to_tp,
)
from nano_llm_infra.training.parallel_state import initialize_model_parallel
class Benchmark:
    """
    通用测速与显存峰值测量：
    提供一个迭代器，自动跑完 warmup，并在 steps 阶段计时与测量峰值显存。
    用法：
        bench = Benchmark(device, warmup, steps)
        for _ in bench:
            # 跑一步训练
        print(bench.avg_ms, bench.peak_mb)
    """

    def __init__(self, device: torch.device, warmup: int, steps: int):
        self.device = device
        self.warmup = warmup
        self.steps = max(steps, 1)
        self.avg_ms = 0.0
        self.peak_mb = 0.0

    def __iter__(self):
        # 1. Warmup
        for i in range(self.warmup):
            yield i

        # 2. 测量设置
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 3. 跑 Benchmark 阶段
        start_event.record()
        for i in range(self.steps):
            yield self.warmup + i
        end_event.record()
        torch.cuda.synchronize(self.device)

        # 4. 结算
        self.avg_ms = start_event.elapsed_time(end_event) / self.steps
        self.peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training entry for DP / TP / PP / ZeRO")
    parser.add_argument("--mode", choices=["dp", "tp", "pp", "zero"], default="dp")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2, help="DP warmup steps before timing")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--bucket-size-mb", type=float, default=0.05)
    parser.add_argument("--zero-stage", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--num-microbatches", type=int, default=4)
    parser.add_argument(
        "--pp-schedule",
        choices=["naive", "gpipe", "1f1b"],
        default="1f1b",
        help="pipeline schedule: naive / gpipe / 1f1b",
    )
    return parser.parse_args()


def setup() -> tuple[torch.device, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("training entry requires CUDA and torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size()


def run_dp(args: argparse.Namespace, device: torch.device, rank: int, world_size: int) -> None:
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    num_heads = 4 if args.hidden_size % 4 == 0 else 2
    torch.manual_seed(0)
    model = TinyTrainingTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=num_heads,
        use_activation_checkpoint=args.activation_checkpoint,
    ).to(device)
    reducer = GradientReducer(model, args.bucket_size_mb)
    runtime = DataParallelRuntime(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        amp=AmpEngine(device, args.precision),
        gradient_reducer=reducer,
    )

    generator = torch.Generator(device=device)
    loss = None
    bench = Benchmark(device, args.warmup, args.steps)
    
    for step in bench:
        generator.manual_seed(1000 + rank * 100 + step)
        token_ids = torch.randint(
            args.vocab_size,
            (args.batch_size, args.seq_len),
            device=device,
            generator=generator,
        )
        loss = runtime.train_step(token_ids)

    if rank == 0:
        tokens = args.batch_size * args.seq_len * world_size
        tokens_per_s = tokens / (bench.avg_ms / 1000.0) if bench.avg_ms > 0 else 0
        print(
            f"[dp] world={world_size} precision={args.precision} "
            f"ckpt={args.activation_checkpoint} "
            f"hidden={args.hidden_size} layers={args.num_layers} "
            f"batch={args.batch_size} seq={args.seq_len}"
        )
        print(f"[dp] loss={loss.item():.4f} avg_step={bench.avg_ms:.3f} ms "
              f"throughput={tokens_per_s:.1f} tok/s peak_mem={bench.peak_mb:.1f} MB")
        for event in reducer.timeline:
            print(f"  {event}")
    runtime.close()



def run_tp(args: argparse.Namespace, device: torch.device, rank: int, world_size: int) -> None:
    initialize_model_parallel(tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1)
    torch.manual_seed(42)
    dense_mlp = TinyMLP(args.hidden_size, args.hidden_size * args.mlp_ratio).to(device)
    tp_mlp = TPMLP(args.hidden_size, args.mlp_ratio).to(device)
    shard_dense_mlp_weights_to_tp(tp_mlp, dense_mlp.up_proj, dense_mlp.down_proj)

    torch.manual_seed(99)
    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device)
    fwd_diff, grad_diff = check_tp_mlp(dense_mlp, tp_mlp, x)
    if rank == 0:
        print(f"[tp] forward max diff={fwd_diff:.6e}")
        print(f"[tp] rowparallel grad max diff={grad_diff:.6e}")

    bench_dense = Benchmark(device, args.warmup, args.steps)
    for _ in bench_dense:
        dense_mlp.zero_grad(set_to_none=True)
        dense_mlp(x).sum().backward()

    bench_tp = Benchmark(device, args.warmup, args.steps)
    for _ in bench_tp:
        tp_mlp.zero_grad(set_to_none=True)
        tp_mlp(x).sum().backward()

    if rank == 0:
        print(
            f"[tp] world={world_size} hidden={args.hidden_size} "
            f"batch={args.batch_size} seq={args.seq_len}"
        )
        print(f"[tp] dense MLP fwd+bwd: {bench_dense.avg_ms:.3f} ms, peak_mem={bench_dense.peak_mb:.1f} MB")
        print(f"[tp] TP MLP fwd+bwd:    {bench_tp.avg_ms:.3f} ms, peak_mem={bench_tp.peak_mb:.1f} MB")


def run_pp(args: argparse.Namespace, device: torch.device, rank: int, world_size: int) -> None:
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=world_size)
    if args.num_layers % world_size != 0:
        raise ValueError("--num-layers must be divisible by nproc")
    if args.batch_size % args.num_microbatches != 0:
        raise ValueError("--batch-size must be divisible by --num-microbatches")
    if args.pp_schedule == "1f1b" and args.num_microbatches < world_size:
        raise ValueError("1f1b requires --num-microbatches >= pipeline parallel size")

    torch.manual_seed(0)
    full_model = TinyTrainingTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=2,
    ).to(device)
    stage = build_pipeline_stage(full_model)
    del full_model

    optimizer = torch.optim.AdamW(stage.parameters(), lr=1e-3)
    runtime = PipelineRuntime(stage)

    torch.manual_seed(123)
    token_ids = torch.randint(
        args.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
    )
    micro_batches = list(token_ids.chunk(args.num_microbatches, dim=0))
    micro_batch_size = args.batch_size // args.num_microbatches

    loss = None
    bench = Benchmark(device, args.warmup, args.steps)
    for _ in bench:
        optimizer.zero_grad(set_to_none=True)
        loss = runtime.run(
            args.pp_schedule,
            micro_batches,
            hidden_shape=(micro_batch_size, args.seq_len, args.hidden_size),
            device=device,
        )
        optimizer.step()

    if runtime.is_last_stage and loss is not None:
        print(f"[pp][rank {rank}] last microbatch loss={loss.item():.4f}")
    if rank == 0:
        print(
            f"[pp] world={world_size} schedule={args.pp_schedule} "
            f"hidden={args.hidden_size} layers={args.num_layers} "
            f"microbatches={args.num_microbatches}"
        )
        print(f"[pp] avg_step={bench.avg_ms:.3f} ms, peak_mem={bench.peak_mb:.1f} MB")
        print(f"[pp] {args.pp_schedule}: {' '.join(runtime.timeline)}")


def run_zero(args: argparse.Namespace, device: torch.device, rank: int, world_size: int) -> None:
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(0)
    model = TinyTrainingTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=2,
    ).to(device)
    torch.manual_seed(1000 + rank)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    runtime = ZeroRuntime(model, args.zero_stage, bucket_size_mb=args.bucket_size_mb)

    # 留存初始参数做个打样对比
    if args.zero_stage == 3:
        pass # 模型现在是空的，暂时不拿
    else:
        sample = model.blocks[0].qkv.weight
        if rank == 0:
            print(f"[zero{args.zero_stage}] before={sample.data.view(-1)[:4].tolist()}")
    
    # 干跑一次（不计入 benchmark）
    # 这个用于打印，但跑完一定要确保垃圾被回收！
    out = runtime.run_step(x)
    del out  # 删除这部分返回的 tensor 引用，保证干净
    
    # 【重点修复】：干跑完之后，我们清空一次显存峰值统计！
    # 否则刚才干跑时如果是 ZeRO-1，积累的梯度峰值会被带到后面的 Benchmark 里
    torch.cuda.reset_peak_memory_stats(device)
    
    if args.zero_stage == 3:
        if rank == 0:
            print(f"[zero{args.zero_stage}] after rank0=[]")
        if world_size > 1:
            dist.barrier()
            if rank == 1:
                print(f"[zero{args.zero_stage}] after rank1=[]")
    else:
        if rank == 0:
            print(f"[zero{args.zero_stage}] after rank0={sample.data.view(-1)[:4].tolist()}")
        if world_size > 1:
            dist.barrier()
            if rank == 1:
                print(f"[zero{args.zero_stage}] after rank1={sample.data.view(-1)[:4].tolist()}")

    bench = Benchmark(device, args.warmup, args.steps)
    
    # 核心：套上 Profiler 壳子
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,  # 开启它才会记录每一块 Tensor 的分配和释放
        record_shapes=True,   # 记录 Tensor 形状
        on_trace_ready=tensorboard_trace_handler(f'./logs/zero{args.zero_stage}'), # 指定结果保存目录
    ) as prof:
        for _ in bench:
            with record_function("model_step"):
                out = runtime.run_step(x)
                del out # 删除这次跑完的引用，把显存还给下一个 step
            prof.step()  # 告诉 profiler，这一个 step 跑完了

    if rank == 0:
        print(
            f"[zero{args.zero_stage}] world={world_size} "
            f"hidden={args.hidden_size} batch={args.batch_size} seq={args.seq_len}"
        )
        
        bw_mem = 0
        if hasattr(runtime.engine, "mem_probe_list") and len(runtime.engine.mem_probe_list) > 0:
            bw_mem = sum(runtime.engine.mem_probe_list[-args.steps:]) / min(args.steps, len(runtime.engine.mem_probe_list))

        print(f"[zero{args.zero_stage}] avg_step={bench.avg_ms:.3f} ms, "
              f"peak_mem={bench.peak_mb:.1f} MB, bw_end_mem={bw_mem:.1f} MB")
    
    runtime.close()


def main() -> None:
    args = parse_args()
    device, rank, world_size = setup()
    try:
        if args.mode == "dp":
            run_dp(args, device, rank, world_size)
        elif args.mode == "tp":
            run_tp(args, device, rank, world_size)
        elif args.mode == "pp":
            run_pp(args, device, rank, world_size)
        else:
            run_zero(args, device, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
