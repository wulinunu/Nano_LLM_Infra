from __future__ import annotations

import torch

from nano_llm_infra.inference.block_manager import BlockAllocator
from nano_llm_infra.inference.engine import IterationLevelScheduler, NanoEngine, Sampler
from nano_llm_infra.inference.TinyTransformerModel import TinyTransformerModel


def main() -> None:
    torch.manual_seed(0)
    block_size = 4
    allocator = BlockAllocator(num_blocks=8)
    scheduler = IterationLevelScheduler(allocator=allocator, max_batch_size=2)
    model_runner = TinyTransformerModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        mlp_hidden_size=32,
    )
    engine = NanoEngine(
        allocator=allocator,
        scheduler=scheduler,
        model_runner=model_runner,
        sampler=Sampler(temperature=1.0, top_k=None, top_p=1.0, do_sample=False),
        block_size=block_size,
    )

    requests = [
        engine.add_request([1, 2, 3], max_new_tokens=3),
        engine.add_request([10, 11], max_new_tokens=2),
    ]
    inserted_request = False

    step = 0
    while engine.scheduler.waiting or engine.scheduler.running or engine.scheduler.preempted:
        stats = engine.step()
        print(
            f"step={step:02d} "
            f"running={stats.running} waiting={stats.waiting} "
            f"finished={stats.finished} free_blocks={stats.free_blocks} "
            f"generated={stats.generated}"
        )
        if not inserted_request and step == 1:
            request = engine.add_request([20, 21, 22, 23, 24], max_new_tokens=2)
            requests.append(request)
            inserted_request = True
            print(
                f"inserted request={request.request_id} "
                f"prompt={request.prompt_token_ids} max_new_tokens={request.max_new_tokens}"
            )
        step += 1

    print("\nFinal requests:")
    for request in requests:
        print(
            f"request={request.request_id} "
            f"prompt={request.prompt_token_ids} "
            f"generated={request.generated_token_ids} "
            f"total={request.total_token_ids}"
        )


if __name__ == "__main__":
    main()

'''
step=00 running=2 waiting=0 finished=0 free_blocks=6 generated={'0': 1, '1': 7}
step=01 running=1 waiting=0 finished=1 free_blocks=6 generated={'0': 11, '1': 3}
inserted request=2 prompt=[20, 21, 22, 23, 24] max_new_tokens=2
step=02 running=1 waiting=0 finished=2 free_blocks=6 generated={'0': 7, '2': 31}
step=03 running=0 waiting=0 finished=3 free_blocks=8 generated={'2': 19}

Final requests:
request=0 prompt=[1, 2, 3] generated=[1, 11, 7] total=[1, 2, 3, 1, 11, 7]
request=1 prompt=[10, 11] generated=[7, 3] total=[10, 11, 7, 3]
request=2 prompt=[20, 21, 22, 23, 24] generated=[31, 19] total=[20, 21, 22, 23, 24, 31, 19]
'''