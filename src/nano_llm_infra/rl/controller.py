import asyncio
import socket
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

from .types import Experience, Phase, Prompt, RLConfig, StepMetrics
from .worker import ColocatedWorker


def token_reward(response_ids: list[int], target_token: int) -> float:
    return 1.0 / (1.0 + abs(response_ids[-1] - target_token))


class AsyncRewardExecutor:
    def __init__(self, max_workers: int = 4) -> None:
        self.pool = ProcessPoolExecutor(max_workers=max_workers) #进程池

    # 异步计算奖励
    async def evaluate(
        self,
        experiences: list[Experience],
        targets: dict[int, int],
    ) -> list[float]:
        loop = asyncio.get_running_loop() # 拿到当前正在运行的事件循环
        futures = [
            loop.run_in_executor(
                self.pool,
                token_reward,
                item.response_ids,
                targets[item.group_id],
            )
            for item in experiences
        ]
        return list(await asyncio.gather(*futures))

    def close(self) -> None:
        self.pool.shutdown()


class ResourcePool:
    """创建一组每卡一个的 ColocatedWorker。"""

    def __init__(self, num_workers: int, config: RLConfig) -> None:
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        ray.init(ignore_reinit_error=True)
        bundles = [{"CPU": 1, "GPU": 1} for _ in range(num_workers)]
        self.placement_group = placement_group(bundles, strategy="STRICT_PACK") #表示这些 bundle 要严格放在同一个节点上 单机多卡
        ray.get(self.placement_group.ready()) # 等待资源准备完成

        # 找一个当前可用的TCP端口
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        actor_class = ray.remote(ColocatedWorker) # 把一个普通的python class变成一个可以由 Ray 启动成远程 Actor 的类
        self.workers = []
        for rank in range(num_workers):
            # 创建调度策略
            strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.placement_group,
                placement_group_bundle_index=rank,
            )
            worker = actor_class.options(
                num_gpus=1,
                scheduling_strategy=strategy,
            ).remote(rank, num_workers, "127.0.0.1", master_port, config) # 启动actor
            self.workers.append(worker)

    def close(self) -> None:
        import ray

        ray.get([worker.close.remote() for worker in self.workers])
        ray.util.remove_placement_group(self.placement_group)
        ray.shutdown()


class WorkerGroup:
    """Controller 的代理层：切分、分发、等待和合并。"""

    def __init__(self, pool: ResourcePool) -> None:
        self.pool = pool
        self.workers = pool.workers
        self.last_rollout_metrics: dict[str, float] = {}

    def _invoke(self, method: str, arguments: list[tuple]) -> list:
        import ray

        futures = [
            getattr(worker, method).remote(*args)
            for worker, args in zip(self.workers, arguments, strict=True)
        ]
        return ray.get(futures)

    def _split(self, values: list) -> list[list]:
        chunks = [[] for _ in self.workers]
        for index, value in enumerate(values):
            chunks[index % len(chunks)].append(value)
        return chunks

    def rollout(self, prompts: list[Prompt]) -> list[Experience]:
        chunks = self._split(prompts)
        results = self._invoke("rollout", [(chunk,) for chunk in chunks])
        self.last_rollout_metrics = {
            "kv_pool_mb": sum(result[1]["kv_pool_mb"] for result in results),
            "kv_blocks_peak": max(result[1]["kv_blocks_peak"] for result in results),
            "rollout_memory_mb": max(result[1]["rollout_memory_mb"] for result in results),
            "released_memory_mb": max(result[1]["released_memory_mb"] for result in results),
        }
        return [item for result in results for item in result[0]]

    def train(self, experiences: list[Experience]) -> dict[str, float]:
        groups: dict[int, list[Experience]] = {}
        for item in experiences:
            groups.setdefault(item.group_id, []).append(item)
        chunks = self._split(list(groups.values()))
        worker_batches = [
            [item for group in chunk for item in group]
            for chunk in chunks
        ]
        results = self._invoke(
            "train",
            [(batch,) for batch in worker_batches],
        )
        total_samples = sum(result["samples"] for result in results)
        return {
            key: sum(result[key] * result["samples"] for result in results)
            / total_samples
            for key in ("loss", "kl", "reward", "grad_norm")
        } | {
            "zero_backward_memory_mb": max(
                result["zero_backward_memory_mb"] for result in results
            )
        }

    def sync_weights(self, mode: str = "gpu") -> dict[str, float]:
        results = self._invoke("sync_weights", [(mode,) for _ in self.workers])
        return {
            "latency_ms": max(result["latency_ms"] for result in results),
            "cpu_copy_mb": sum(result["cpu_copy_mb"] for result in results),
            "version": results[0]["version"],
        }

    def memory_mb(self) -> float:
        return max(self._invoke("memory_mb", [() for _ in self.workers]))

    def weights_match(self) -> bool:
        return all(self._invoke("weights_match", [() for _ in self.workers]))


class RLController:
    """veRL 风格的单控制器：只描述 RL 控制流。"""

    def __init__(
        self,
        worker_group: WorkerGroup,
        reward_executor: AsyncRewardExecutor,
        config: RLConfig,
    ) -> None:
        self.worker_group = worker_group
        self.reward_executor = reward_executor
        self.config = config
        self.phase = Phase.ROLLOUT
        self.policy_version = 0 
        self.step = 0

    async def run_step(self, prompts: list[Prompt]) -> StepMetrics:
        self.phase = Phase.ROLLOUT
        start = perf_counter()
        experiences = self.worker_group.rollout(prompts)
        rollout_ms = (perf_counter() - start) * 1000

        self.phase = Phase.REWARD
        start = perf_counter()
        targets = {prompt.group_id: prompt.target_token for prompt in prompts}
        rewards = await self.reward_executor.evaluate(experiences, targets)
        for experience, reward in zip(experiences, rewards, strict=True):
            experience.reward = reward
        reward_ms = (perf_counter() - start) * 1000

        self.phase = Phase.TRAIN
        start = perf_counter()
        train_metrics = self.worker_group.train(experiences)
        train_ms = (perf_counter() - start) * 1000

        self.phase = Phase.SYNC
        sync_metrics = self.worker_group.sync_weights()
        self.policy_version = int(sync_metrics["version"])
        assert self.worker_group.weights_match()

        metrics = StepMetrics(
            step=self.step,
            phase=self.phase,
            rollout_ms=rollout_ms,
            reward_ms=reward_ms,
            train_ms=train_ms,
            sync_ms=sync_metrics["latency_ms"],
            policy_version=self.policy_version,
            mean_reward=train_metrics["reward"],
            loss=train_metrics["loss"],
            kl=train_metrics["kl"],
            grad_norm=train_metrics["grad_norm"],
            gpu_memory_mb=self.worker_group.memory_mb(),
            kv_pool_mb=self.worker_group.last_rollout_metrics["kv_pool_mb"],
            kv_blocks_peak=int(
                self.worker_group.last_rollout_metrics["kv_blocks_peak"]
            ),
            rollout_memory_mb=self.worker_group.last_rollout_metrics[
                "rollout_memory_mb"
            ],
            released_memory_mb=self.worker_group.last_rollout_metrics[
                "released_memory_mb"
            ],
            zero_backward_memory_mb=train_metrics["zero_backward_memory_mb"],
        )
        self.step += 1
        return metrics
