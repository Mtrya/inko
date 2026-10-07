"""Real learner DDP must recover the analytic global sample mean."""

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.posttrain.distributed import global_loss, initialize, prime_buckets, wrap


def worker(rank, path, destination, microbatch):
    torch.set_num_threads(1)
    initialize(rank, 2, torch.device("cpu"), "file://" + path, 30)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2)
        learner = wrap(model, torch.device("cpu"))
        # Global x=[1,3,5,7], E[x²]=21. d(0.5*(w*x)²)/dw = 2*21 = 42.
        values = torch.tensor([1.0, 3.0, 5.0, 7.0])[rank::2].reshape(-1, 1)
        before = model.weight.detach().clone()
        random_before = torch.get_rng_state().clone()
        prime_buckets(learner, values[:microbatch])
        assert torch.equal(model.weight, before)
        assert torch.equal(torch.get_rng_state(), random_before)
        assert model.weight.grad is None
        for start in range(0, 2, microbatch):
            context = learner.no_sync() if start + microbatch < 2 else nullcontext()
            with context:
                losses = (
                    0.5 * learner(values[start : start + microbatch]).square().flatten()
                )
                global_loss(losses, 4).backward()
        torch.save(model.weight.grad, destination + str(rank))
    finally:
        dist.destroy_process_group()


def test_global_mean_across_ranks_and_microbatches(tmp_path):
    for microbatch in (1, 2):
        target = str(tmp_path / f"gradient-{microbatch}-")
        mp.spawn(
            worker,
            args=(str(tmp_path / f"rendezvous-{microbatch}"), target, microbatch),
            nprocs=2,
            join=True,
        )
        for rank in range(2):
            torch.testing.assert_close(
                torch.load(target + str(rank), weights_only=True),
                torch.tensor([[42.0]]),
                rtol=0,
                atol=0,
            )
