"""Complete-group reward-endpoint covariance guidance in latent space.

Endpoint latents and their rewards are the only inputs; a prompt group is one
round's draws for one prompt. The aggregate direction is the reward-MAD-scaled
covariance normalized to unit RMS, the magnitude of an active group is one, and
a group with no reward spread is inactive. Every statistic is exact over the
complete group: a caller holding part of a group must pass the collective that
reduces the partial statistics (see `reduce`).
"""

from dataclasses import dataclass

import torch

from .schema import require

EPS = 1e-5


@dataclass
class GroupGuidance:
    directions: torch.Tensor  # [groups, *latent] unit RMS where active, else zero
    magnitudes: torch.Tensor  # [groups] one where active, else zero
    active: torch.Tensor  # [groups] bool


def group_guidance(
    endpoints,
    rewards,
    groups,
    group_count,
    *,
    expected_group_size,
    reduce=None,
    eps=EPS,
):
    """Covariance, direction and magnitude per prompt group.

    `endpoints` are flattened local endpoints [N, D], `rewards` and `groups` the
    matching reward and group index per row. `reduce(value, operation)` must
    support "sum" and "max" across the ranks that hold parts of a group; without
    it the call sees complete groups itself. Every group must contain exactly
    `expected_group_size` draws.
    """
    x = endpoints.detach().float()
    r = rewards.detach().float().reshape(-1)
    index = groups.reshape(-1)
    require(
        x.ndim == 2 and x.shape[0] > 0 and x.shape[0] == r.numel() == index.numel(),
        "guidance needs one reward and group per endpoint",
    )
    require(
        bool(torch.isfinite(x).all()) and bool(torch.isfinite(r).all()),
        "guidance inputs must be finite",
    )
    require(
        index.dtype in (torch.int32, torch.int64)
        and int(index.min()) >= 0
        and int(index.max()) < group_count,
        "invalid guidance group index",
    )
    require(
        type(expected_group_size) is int and expected_group_size >= 2,
        "guidance requires at least two draws per prompt",
    )
    width = x.shape[1]
    shape = (group_count, width)
    counts = torch.zeros(group_count, device=x.device).index_add_(
        0, index, torch.ones_like(r)
    )
    reward_sum = torch.zeros(group_count, device=x.device).index_add_(0, index, r)
    if reduce is not None:
        counts, reward_sum = reduce(torch.cat((counts, reward_sum)), "sum").chunk(2)
    require(
        bool((counts == expected_group_size).all()),
        "incomplete prompt group in guidance",
    )
    means = reward_sum / counts
    maxima = r.new_full((group_count,), -torch.inf).scatter_reduce_(
        0, index, r, reduce="amax"
    )
    neg_minima = r.new_full((group_count,), -torch.inf).scatter_reduce_(
        0, index, -r, reduce="amax"
    )
    if reduce is not None:
        maxima, neg_minima = reduce(torch.stack((maxima, neg_minima)), "max").unbind()
    # A rounded mean can differ from an exactly constant FP32 reward. Detect
    # ties over the complete group before normalization can amplify roundoff.
    constant = maxima == -neg_minima
    centered = torch.where(constant[index], 0.0, r - means[index])
    numerator = torch.zeros(shape, device=x.device).index_add_(
        0, index, centered[:, None] * x
    )
    deviation = torch.zeros(group_count, device=x.device).index_add_(
        0, index, centered.abs()
    )
    if reduce is not None:
        numerator, deviation = reduce(
            torch.cat((numerator.reshape(-1), deviation)), "sum"
        ).split([group_count * width, group_count])
        numerator = numerator.reshape(shape)
    covariance = numerator / (counts - 1)[:, None]
    mad = deviation / counts
    signal = covariance / torch.where(mad > 0, mad, torch.ones_like(mad))[:, None]
    rms = signal.square().mean(1).sqrt()
    active = (mad > 0) & (rms > 0)
    directions = torch.where(
        active[:, None], signal / rms.clamp_min(eps)[:, None], torch.zeros_like(signal)
    )
    return {
        "directions": directions,
        "magnitudes": torch.where(active, torch.ones_like(rms), torch.zeros_like(rms)),
        "active": active,
        "counts": counts,
        "covariance": covariance,
    }


def round_guidance(config, rollouts, rewards, *, reduce=None):
    """Guidance for one ART round from this rank's rollouts and the round rewards.

    A draw's group is its collection slot, `draw_index // draws_per_prompt`:
    prompt groups never key on prompt text, so a prompt repeated within one
    round still yields two independent groups.
    """
    ids = [key for r in rollouts for key in r.sample_ids]
    require(all(key in rewards for key in ids), "missing local reward for the guidance")
    draws = config.collection.draws_per_prompt
    endpoints = torch.cat([r.terminal.flatten(1) for r in rollouts]).float()
    stats = group_guidance(
        endpoints,
        torch.tensor([rewards[key] for key in ids], device=endpoints.device),
        torch.tensor(
            [draw // draws for r in rollouts for draw in r.draw_indices],
            device=endpoints.device,
        ),
        config.collection.prompts_per_round,
        expected_group_size=draws,
        reduce=reduce,
    )
    shape = (-1, *rollouts[0].terminal.shape[1:])
    return GroupGuidance(
        stats["directions"].reshape(shape),
        stats["magnitudes"],
        stats["active"],
    )
