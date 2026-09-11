from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

from nano_llm_infra.rl.controller import (
    AsyncRewardExecutor,
    RLController,
    ResourcePool,
    WorkerGroup,
)
from nano_llm_infra.rl.packing import (
    pack_experiences,
    padding_stats,
    segmented_causal_mask,
)
from nano_llm_infra.rl.types import Experience, Prompt, RLConfig


def build_prompts(num_prompts: int) -> list[Prompt]:
    return [
        Prompt(
            group_id=index,
            prompt_ids=[1, 3 + index, 2] + [4] * (index % 4),
            target_token=10 + index,
        )
        for index in range(num_prompts)
    ]


def cpu_reward_work(value: int) -> int:
    result = 0
    for index in range(300_000):
        result = (result + value * index) % 104_729
    return result


async def run_benchmarks(
    worker_group: WorkerGroup,
    controller: RLController,
    prompts: list[Prompt],
) -> None:
    metrics = await controller.run_step(prompts)
    print(
        f"[flow] kv_pool={metrics.kv_pool_mb:.2f}MB "
        f"peak_blocks={metrics.kv_blocks_peak} "
        f"kv_released={metrics.released_memory_mb:.2f}MB "
        f"zero2_memory={metrics.zero_backward_memory_mb:.1f}MB "
        f"version={metrics.policy_version}"
    )
    colocated_ms = (
        metrics.rollout_ms + metrics.reward_ms + metrics.train_ms + metrics.sync_ms
    )
    static_ms = (
        metrics.rollout_ms * 2
        + metrics.reward_ms
        + metrics.train_ms * 2
        + metrics.sync_ms
    )
    samples = len(prompts) * controller.config.group_size
    print(
        f"[colocation] step={colocated_ms:.1f}ms "
        f"samples/s={samples / colocated_ms * 1000:.2f} "
        f"static_split_estimate={static_ms:.1f}ms"
    )

    gpu_sync = worker_group.sync_weights("gpu")
    cpu_sync = worker_group.sync_weights("cpu")
    print(
        f"[weight_sync] gpu={gpu_sync['latency_ms']:.2f}ms "
        f"cpu={cpu_sync['latency_ms']:.2f}ms "
        f"cpu_copy={cpu_sync['cpu_copy_mb']:.2f}MB"
    )

    experiences = [
        Experience([1, 2], list(range(3, 3 + length)), [0.0] * length, 0, 0)
        for length in [2, 5, 9, 14]
    ]
    packed = pack_experiences(experiences, "cpu")
    mask = segmented_causal_mask(
        packed.cu_seqlens,
        len(packed.token_ids),
        packed.token_ids.device,
    )
    packed_tokens, padded_tokens, padding_ratio = padding_stats(experiences)
    boundary = int(packed.cu_seqlens[1])
    assert not mask[boundary, boundary - 1]
    print(
        f"[packing] packed={packed_tokens} padded={padded_tokens} "
        f"padding_ratio={padding_ratio:.1%} cross_sequence_attention=blocked"
    )

    values = list(range(16))
    start = perf_counter()
    [cpu_reward_work(value) for value in values]
    sync_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(cpu_reward_work, values))
    async_ms = (perf_counter() - start) * 1000
    print(
        f"[reward] sync={sync_ms:.1f}ms async={async_ms:.1f}ms "
        f"speedup={sync_ms / async_ms:.2f}x"
    )


async def run(args: argparse.Namespace) -> None:
    config = RLConfig(
        group_size=args.group_size,
        response_length=args.response_length,
        kv_num_blocks=args.kv_num_blocks,
        rollout_batch_size=args.rollout_batch_size,
        zero_bucket_size_mb=args.zero_bucket_size_mb,
    )
    pool = ResourcePool(args.num_workers, config)
    worker_group = WorkerGroup(pool)
    reward_executor = AsyncRewardExecutor(config.reward_workers)
    controller = RLController(worker_group, reward_executor, config)
    prompts = build_prompts(max(args.num_workers, args.num_prompts))

    print(
        f"workers={args.num_workers} device=cuda backend=nccl "
        "flow=rollout->reward->train->sync"
    )
    try:
        if args.benchmark:
            await run_benchmarks(worker_group, controller, prompts)
            return

        for _ in range(args.steps):
            metrics = await controller.run_step(prompts)
            print(
                f"step={metrics.step} phase={metrics.phase.value} "
                f"version={metrics.policy_version} reward={metrics.mean_reward:.3f} "
                f"loss={metrics.loss:.4f} kl={metrics.kl:.5f} "
                f"grad_norm={metrics.grad_norm:.4f} "
                f"memory={metrics.gpu_memory_mb:.1f}MB\n"
                f"  KV allocate: pool={metrics.kv_pool_mb:.2f}MB "
                f"allocated={metrics.rollout_memory_mb:.1f}MB\n"
                f"  Continuous rollout: {metrics.rollout_ms:.1f}ms "
                f"peak_blocks={metrics.kv_blocks_peak}\n"
                f"  KV release: {metrics.released_memory_mb:.2f}MB\n"
                f"  Reward: {metrics.reward_ms:.1f}ms\n"
                f"  ZeRO-2 train: {metrics.train_ms:.1f}ms "
                f"backward_memory={metrics.zero_backward_memory_mb:.1f}MB\n"
                f"  Weight sync: {metrics.sync_ms:.1f}ms"
            )
    finally:
        reward_executor.close()
        pool.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal veRL-style HybridFlow demo")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--response-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--kv-num-blocks", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--zero-bucket-size-mb", type=float, default=0.05)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
