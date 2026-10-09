"""Fresh-sample fake regression diagnostics; these never change training state."""

import torch
import torch.distributed as dist

from .distributed import unanimous_finite
from .objectives.dmd import fake_loss, forward_noise
from .rollout import fresh_rollouts, score_inputs


@torch.no_grad()
def fake_regression_probe(config, roles, rollouts, generator_step):
    total = config.collection.prompts_per_round * config.collection.draws_per_prompt
    device = rollouts[0].terminal.device
    sums = torch.zeros(3, device=device, dtype=torch.float32)
    for draw in range(config.fake_probe.draws):
        fresh = fresh_rollouts(
            config,
            roles.student_field,
            rollouts,
            "fake_probe_collection",
            draw,
            generator_step=generator_step,
        )
        for row in fresh:
            target = getattr(row, config.fake.target)
            times, noise = score_inputs(config, row, "fake_probe_score", draw)
            velocity = roles.fake_field(
                forward_noise(target, times, noise), times, row.conditioning
            )
            errors = fake_loss(velocity, target, noise)
            # Flow velocity error -> squared score error: [t/(1-t)]^2.
            weighted = errors * times.square() / (1 - times).square()
            sums += torch.stack(
                [
                    errors.sum(),
                    weighted.sum(),
                    torch.tensor(float(len(times)), device=device, dtype=torch.float32),
                ]
            )
    unanimous_finite(bool(torch.isfinite(sums).all()), device)
    if dist.is_initialized():
        dist.all_reduce(sums)
    count = total * config.fake_probe.draws
    if sums[2].item() != count:
        raise ValueError("incomplete fresh-sample fake regression probe")
    return {
        "samples": count,
        "student_version": generator_step,
        "velocity_mse": float(sums[0] / count),
        "score_weighted_mse": float(sums[1] / count),
        "scope": "Fresh current-student regression risk; not an oracle score error.",
    }
