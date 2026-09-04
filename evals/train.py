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
    build_pipeline_stage_model,
)
from nano_llm_infra.training.distributed.tensor_parallel import (
    TPMLP,
    check_tp_mlp,
    shard_dense_mlp_weights_to_tp,
)
from nano_llm_infra.training.distributed.expert_parallel import (
    DenseMoE,
    ExpertParallelMoE,
    check_expert_parallel,
    copy_dense_moe_weights,
)
from nano_llm_infra.training.distributed.context_patallel import (
    ContextParallelAttention,
    DenseCausalAttention,
    check_context_parallel,
    copy_dense_attention_weights,
)
from nano_llm_infra.training.parallel_state import initialize_model_parallel
class Benchmark:
    """warmup 若干步后 begin()，计时 steps 步，再 end() 写出 avg_ms / peak_mb。"""

    def __init__(self, device: torch.device, warmup: int, steps: int):
        self.device = device
        self.warmup = warmup
        self.steps = max(steps, 1)
        self.avg_ms = 0.0
        self.peak_mb = 0.0
        self._start = None
        self._end = None

    def begin(self) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def end(self) -> None:
        self._end.record()
        torch.cuda.synchronize(self.device)
        self.avg_ms = self._start.elapsed_time(self._end) / self.steps
        self.peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified training entry for DP / TP / PP / EP / CP / ZeRO"
    )
    parser.add_argument(
        "--mode", choices=["dp", "tp", "pp", "ep", "cp", "zero"], default="dp"
    )
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
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
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

    loss = None
    bench = Benchmark(device, args.warmup, args.steps)
    for step in range(bench.warmup + bench.steps):
        if step == bench.warmup:
            bench.begin()
        torch.manual_seed(1000 + rank * 100 + step)
        token_ids = torch.randint(
            args.vocab_size, #上界
            (args.batch_size, args.seq_len), #尺寸
            device=device
        )
        loss = runtime.train_step(token_ids)
    bench.end()

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
    tp_mlp = TPMLP(args.hidden_size, args.hidden_size * args.mlp_ratio).to(device)
    shard_dense_mlp_weights_to_tp(tp_mlp, dense_mlp.up_proj, dense_mlp.down_proj)

    torch.manual_seed(99)
    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device)
    fwd_diff, x_grad_diff, up_grad_diff, down_grad_diff = check_tp_mlp(dense_mlp, tp_mlp, x)
    if rank == 0:
        print(f"[tp] forward max diff={fwd_diff:.6e}")
        print(f"[tp] x.grad max diff={x_grad_diff:.6e}")
        print(f"[tp] up_proj grad max diff={up_grad_diff:.6e}")
        print(f"[tp] down_proj grad max diff={down_grad_diff:.6e}")

    bench_dense = Benchmark(device, args.warmup, args.steps)
    for step in range(bench_dense.warmup + bench_dense.steps):
        if step == bench_dense.warmup:
            bench_dense.begin()
        dense_mlp.zero_grad(set_to_none=True)
        dense_mlp(x).sum().backward()
    bench_dense.end()

    bench_tp = Benchmark(device, args.warmup, args.steps)
    for step in range(bench_tp.warmup + bench_tp.steps):
        if step == bench_tp.warmup:
            bench_tp.begin()
        tp_mlp.zero_grad(set_to_none=True)
        tp_mlp(x).sum().backward()
    bench_tp.end()

    if rank == 0:
        print(
            f"[tp] world={world_size} hidden={args.hidden_size} "
            f"batch={args.batch_size} seq={args.seq_len}"
        )
        print(f"[tp] dense MLP fwd+bwd: {bench_dense.avg_ms:.3f} ms, peak_mem={bench_dense.peak_mb:.1f} MB")
        print(f"[tp] TP MLP fwd+bwd:    {bench_tp.avg_ms:.3f} ms, peak_mem={bench_tp.peak_mb:.1f} MB")


def run_pp(args: argparse.Namespace, device: torch.device, rank: int, world_size: int) -> None:
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=world_size)
    if args.batch_size % args.num_microbatches != 0:
        raise ValueError("--batch-size must be divisible by --num-microbatches")

    torch.manual_seed(0)
    full_model = TinyTrainingTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=2,
    ).to(device)
    stage = build_pipeline_stage_model(full_model)
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
    for step in range(bench.warmup + bench.steps):
        if step == bench.warmup:
            bench.begin()
        optimizer.zero_grad(set_to_none=True)
        loss = runtime.run(
            args.pp_schedule,
            micro_batches,
            hidden_shape=(micro_batch_size, args.seq_len, args.hidden_size),
            device=device,
        )
        optimizer.step()
    bench.end()

    if runtime.is_last_stage and loss is not None: # 证明最后一个stage确实完成了
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

    bench = Benchmark(device, args.warmup, args.steps)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        on_trace_ready=tensorboard_trace_handler(f"./logs/zero{args.zero_stage}"),
    ) as prof:
        for step in range(bench.warmup + bench.steps):
            if step == bench.warmup:
                bench.begin()
            with record_function("model_step"):
                runtime.run_step(x)
        bench.end()

    if rank == 0:
        print(
            f"[zero{args.zero_stage}] world={world_size} "
            f"hidden={args.hidden_size} batch={args.batch_size} seq={args.seq_len}"
        )

        bw_mem = 0
        if runtime.backward_memory_mb:
            measured = runtime.backward_memory_mb[-args.steps:] #只取正式测试的计算 warmup不算
            bw_mem = sum(measured) / len(measured)

        print(f"[zero{args.zero_stage}] avg_step={bench.avg_ms:.3f} ms, "
              f"peak_mem={bench.peak_mb:.1f} MB, bw_end_mem={bw_mem:.1f} MB")
    
    runtime.close()


def run_ep(args: argparse.Namespace, device: torch.device, rank: int, world_size: int) -> None:
    initialize_model_parallel(
        expert_model_parallel_size=world_size,
    )
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by world size")
    if args.batch_size % world_size != 0:
        raise ValueError("--batch-size must be divisible by world size in EP mode")

    torch.manual_seed(42)
    dense_moe = DenseMoE(
        args.hidden_size,
        args.hidden_size * args.mlp_ratio,
        args.num_experts,
    ).to(device)
    ep_moe = ExpertParallelMoE(
        args.hidden_size,
        args.hidden_size * args.mlp_ratio,
        args.num_experts,
        args.capacity_factor,
    ).to(device)
    copy_dense_moe_weights(dense_moe, ep_moe)

    torch.manual_seed(1000)
    full_hidden = torch.randn(
        args.batch_size, args.seq_len, args.hidden_size, device=device
    )
    hidden = full_hidden.chunk(world_size, dim=0)[rank].contiguous()
    output_diff, input_grad_diff, router_grad_diff, expert_grad_diff = (
        check_expert_parallel(dense_moe, ep_moe, hidden)
    )
    dense_moe.zero_grad(set_to_none=True)
    ep_moe.zero_grad(set_to_none=True)

    bench = Benchmark(device, args.warmup, args.steps)
    for step in range(bench.warmup + bench.steps):
        if step == bench.warmup:
            bench.begin()
        ep_moe.zero_grad(set_to_none=True)
        loss = ep_moe(hidden).square().mean()
        loss.backward()
        if ep_moe.router.weight.grad is not None:
            dist.all_reduce(ep_moe.router.weight.grad, group=ep_moe.group)
    bench.end()

    if rank == 0:
        print(
            f"[ep] world={world_size} experts={args.num_experts} "
            f"capacity_factor={args.capacity_factor}"
        )
        print(
            f"[ep] output diff={output_diff:.6e}, input.grad diff={input_grad_diff:.6e}, "
            f"router grad diff={router_grad_diff:.6e}, expert grad diff={expert_grad_diff:.6e}"
        )
        print(
            f"[ep] expert_load={ep_moe.last_expert_load.tolist()} "
            f"dropped_tokens={ep_moe.last_dropped_tokens}"
        )
        print(
            f"[ep] avg_step={bench.avg_ms:.3f} ms, peak_mem={bench.peak_mb:.1f} MB"
        )


def run_cp(args: argparse.Namespace, device: torch.device, rank: int, world_size: int) -> None:
    initialize_model_parallel(
        context_parallel_size=world_size,
    )
    if args.seq_len % world_size != 0:
        raise ValueError("--seq-len must be divisible by world size")

    num_heads = 4 if args.hidden_size % 4 == 0 else 2
    torch.manual_seed(42)
    dense_attention = DenseCausalAttention(args.hidden_size, num_heads).to(device)
    cp_attention = ContextParallelAttention(args.hidden_size, num_heads).to(device)
    copy_dense_attention_weights(dense_attention, cp_attention)

    torch.manual_seed(99)
    full_hidden = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device) # [batch_size, seq_len, hidden_size]
    local_hidden = full_hidden.chunk(world_size, dim=1)[rank].contiguous() # [batch_size, local_seq_len, hidden_size]
    output_diff, input_grad_diff, qkv_grad_diff, out_grad_diff = (
        check_context_parallel(dense_attention, cp_attention, local_hidden)
    )
    dense_attention.zero_grad(set_to_none=True)
    cp_attention.zero_grad(set_to_none=True)

    bench = Benchmark(device, args.warmup, args.steps)
    for step in range(bench.warmup + bench.steps):
        if step == bench.warmup:
            bench.begin()
        cp_attention.zero_grad(set_to_none=True)
        loss = cp_attention(local_hidden).square().mean()
        loss.backward()
        for parameter in cp_attention.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, group=cp_attention.group)
    bench.end()

    if rank == 0:
        print(
            f"[cp] world={world_size} global_seq={args.seq_len} "
            f"local_seq={args.seq_len // world_size}"
        )
        print(
            f"[cp] output diff={output_diff:.6e}, input.grad diff={input_grad_diff:.6e}, "
            f"qkv grad diff={qkv_grad_diff:.6e}, out_proj grad diff={out_grad_diff:.6e}"
        )
        print(
            f"[cp] avg_step={bench.avg_ms:.3f} ms, peak_mem={bench.peak_mb:.1f} MB"
        )


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
        elif args.mode == "ep":
            run_ep(args, device, rank, world_size)
        elif args.mode == "cp":
            run_cp(args, device, rank, world_size)
        else:
            run_zero(args, device, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
