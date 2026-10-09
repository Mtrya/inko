"""Independent scalar joint gradients through real two-rank learner DDP."""

from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_posttrain_dmd import config as dmd_config

from src.posttrain import engine
from src.posttrain.distributed import initialize, prime_buckets, wrap
from src.posttrain.objectives.dmd import surrogate
from src.posttrain.rollout import NFTRollout, TrainingRollout
from src.posttrain.training_config import Joint


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))

    def forward(self, z, t, cond, uncond=None, guidance=1):
        return self.weight.expand_as(z)


class GradientCapture:
    def __init__(self, model):
        self.model, self.gradient = model, None

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def step(self, loss):
        self.gradient = self.model.weight.grad.detach().clone()
        return float(self.gradient.abs())


def worker(rank, path, destination, microbatch, coefficients):
    torch.set_num_threads(1)
    initialize(rank, 2, torch.device("cpu"), "file://" + path, 30)
    original_nft, original_dmd, original_fake = (
        engine.nft_inputs,
        engine.dmd_terms,
        engine.fake_updates,
    )
    try:
        dmd, nft, ref = coefficients
        # The math spec is under test, so bypass typed config construction only
        # for this scalar field fixture. Config validation has separate coverage.
        c = dmd_config()
        c = SimpleNamespace(**c.__dict__)
        c.joint = Joint(dmd, nft, ref)
        c.collection = replace(c.collection, draws_per_prompt=2)
        c.nft = replace(c.nft, beta=0.5, advantage_clip=2, time_draws=2)
        c.joint_updates = 1
        model = Constant(1.6)
        learner = wrap(model, torch.device("cpu"))
        old = Constant(0.8).requires_grad_(False)
        reference = Constant(0.1).requires_grad_(False) if ref else None
        roles = SimpleNamespace(
            student=model,
            fake=None,
            old=old if nft else None,
            reference=reference,
            old_field=old,
            reference_field=reference,
            update_ema=lambda: None,
        )
        optimizer = GradientCapture(model)
        prime_buckets(learner, torch.zeros(1, 1), torch.tensor([0.5]), ())
        local = list(range(4))[rank::2]
        nft_rows, dmd_rows = [], []
        for start in range(0, 2, microbatch):
            positions = local[start : start + microbatch]
            keys = [f"0:{p // 2}:{p % 2}" for p in positions]
            x = torch.full((len(keys), 1), 2.0)
            t = torch.full((len(keys),), 0.25)
            nft_rows.append(
                NFTRollout(
                    keys,
                    [str(p // 2) for p in positions],
                    positions,
                    0,
                    0,
                    x,
                    (),
                    None,
                    "joint",
                    0,
                )
            )
            dmd_rows.append(
                TrainingRollout(
                    keys,
                    [],
                    positions,
                    0,
                    torch.zeros_like(x),
                    t,
                    torch.zeros_like(t, dtype=torch.long),
                    torch.full_like(x, 1.2),
                    x,
                    (),
                    None,
                    "joint",
                    0,
                )
            )
        engine.nft_inputs = lambda config, r, step, draw: (
            torch.full((len(r.sample_ids),), 0.25 if draw == 0 else 0.5),
            torch.zeros_like(r.terminal),
        )

        def dmd_terms(config, roles, student, r, scale):
            prediction = 0.75 * student(r.selected_input, r.selected_time, ())
            return (
                surrogate(prediction, torch.full_like(prediction, 4)),
                prediction,
                {"role": "analytic_dmd"},
            )

        engine.dmd_terms = dmd_terms
        engine.fake_updates = lambda *args: {
            "fake_losses": [],
            "fake_grad_norms": [],
            "inputs": [],
        }
        p = dict(
            zip(["0:0:0", "0:0:1", "0:1:0", "0:1:1"], [0.75, 0.25, 0.5, 1], strict=True)
        )
        result = engine.update_joint(
            c,
            roles,
            learner,
            None,
            nft_rows if nft else None,
            dmd_rows if dmd else None,
            p,
            optimizer,
            None,
            0,
            0,
        )
        torch.save(
            {"gradient": optimizer.gradient, "terms": result["term_losses"]},
            destination + str(rank),
        )
    finally:
        engine.nft_inputs, engine.dmd_terms, engine.fake_updates = (
            original_nft,
            original_dmd,
            original_fake,
        )
        dist.destroy_process_group()


def test_joint_gradient_additivity_rank_and_time_microbatch_mean(tmp_path):
    # Four probabilities give NFT gradients [-1.25,+1.25,0,-2.5] after
    # averaging t=.25,.5 -> mean -.625. DMD derivative is4*.75=3.
    # Reference derivative is2*(1.6-.1)=3. Joint=.2*3+.4*(-.625)+.1*3=.65.
    for microbatch in (1, 2):
        for coefficients, expected in (
            ((1, 0, 0), 3.0),
            ((0, 1, 0), -0.625),
            ((0, 1, 0.1), -0.325),
            ((0.2, 0.4, 0.1), 0.65),
        ):
            target = str(tmp_path / f"gradient-{microbatch}-{coefficients}-")
            mp.spawn(
                worker,
                args=(
                    str(tmp_path / f"rendezvous-{microbatch}-{coefficients}"),
                    target,
                    microbatch,
                    coefficients,
                ),
                nprocs=2,
                join=True,
            )
            for rank in range(2):
                result = torch.load(target + str(rank), weights_only=True)
                torch.testing.assert_close(
                    result["gradient"], torch.tensor(expected), rtol=2e-6, atol=2e-6
                )
