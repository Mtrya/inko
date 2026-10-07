"""Real learner DDP must recover the analytic global sample mean."""

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.posttrain.distributed import global_loss, initialize, prime_buckets, wrap


def worker(rank, path, destination, microbatch, weighted):
    torch.set_num_threads(1)
    initialize(rank, 2, torch.device("cpu"), "file://" + path, 30)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2)
        learner = wrap(model, torch.device("cpu"))
        # Global x=[1,3,5,7], E[x²]=21. d(0.5*(w*x)²)/dw = 2*21 = 42.
        values = torch.tensor([1.0, 3.0, 5.0, 7.0])[rank::2].reshape(-1, 1)
        weights = (
            torch.tensor([0.4, 0.8, 1.2, 1.6])[rank::2] if weighted else torch.ones(2)
        )
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
                losses = losses * weights[start : start + microbatch]
                global_loss(losses, 4).backward()
        torch.save(model.weight.grad, destination + str(rank))
    finally:
        dist.destroy_process_group()


def test_global_mean_across_ranks_and_microbatches(tmp_path):
    for microbatch, weighted in ((1, False), (2, False), (1, True), (2, True)):
        target = str(tmp_path / f"gradient-{microbatch}-{weighted}-")
        mp.spawn(
            worker,
            args=(
                str(tmp_path / f"rendezvous-{microbatch}-{weighted}"),
                target,
                microbatch,
                weighted,
            ),
            nprocs=2,
            join=True,
        )
        for rank in range(2):
            torch.testing.assert_close(
                torch.load(target + str(rank), weights_only=True),
                torch.tensor([[58.0 if weighted else 42.0]]),
                rtol=1e-7 if weighted else 0,
                atol=1e-6 if weighted else 0,
            )
