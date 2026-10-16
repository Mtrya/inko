"""ART guidance, frozen-target regression and student updates.

The reference implementation runs in decreasing noise time sigma with velocity
`eps - x0`; Inko's data-increasing time `t = 1 - sigma` and velocity `x0 - eps`
carry the same update after `dt -> -dt`, so the frozen target becomes
`cached + G` and the timestep weight becomes `1 - t`. The trust-region proxy
grades a move against the reference transition variance, whose factor `-dt_ref`
is `dt_ours > 0` here; the ODE coefficient enters only squared, so the proxy
value is unchanged by the mapping.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .distributed import global_loss
from .rollout import ARTRollout, art_window
from .schema import require

NOISE_LEVEL = 0.8
KL_THRESHOLD = 1e-5
EPS = 1e-5


def mean_elements(value):
    return value.float().flatten(1).mean(1)


def finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("nonfinite ART input/loss")


def reference_losses(new, reference):
    if new.shape != reference.shape:
        raise ValueError("reference prediction shapes differ")
    result = mean_elements((new.float() - reference.detach().float()).square())
    finite(result)
    return result


def losses(
    new,
    cached,
    times,
    deltas,
    direction,
    magnitude,
    *,
    eps=EPS,
    noise=NOISE_LEVEL,
    threshold=KL_THRESHOLD,
):
    """Per-item ART loss and keep mask; every input except `new` is frozen.

    `cached` is the rollout policy's prediction at the trained state, `direction`
    its guidance direction (zero for an inactive group) and `magnitude` its
    per-item guidance magnitude. `deltas` are the physical-time steps of the
    collection grid. An item trains only when its group is active and its move is
    not both above the proxy threshold and outward.
    """
    new = new.float()
    cached = cached.detach().float()
    direction = direction.detach().float()
    times = times.detach().float().reshape(-1)
    deltas = deltas.detach().float().reshape(-1)
    magnitude = magnitude.detach().float().reshape(-1)
    if not (
        new.shape == cached.shape == direction.shape
        and new.shape[0] == times.numel() == deltas.numel() == magnitude.numel()
    ):
        raise ValueError("ART item shapes differ")
    finite(new, cached, direction, times, deltas, magnitude)
    if not bool(((times >= 0) & (times < 1)).all()):
        raise ValueError("ART physical times must exclude the data endpoint")
    if not bool((deltas > 0).all()):
        raise ValueError("ART states must move toward data")
    if not bool((magnitude >= 0).all()):
        raise ValueError("invalid ART guidance magnitude")
    shape = (-1,) + (1,) * (new.ndim - 1)
    displacement = magnitude.view(shape) * direction
    target = (cached + displacement).detach()
    residual = target - new
    # Per-item scalars stay flat: a [N] weight broadcast against a [N, C, H, W]
    # mean would otherwise multiply along the latent axis instead of the item axis.
    unmasked = (1 - times) * mean_elements(residual.square())
    movement = (new - cached).detach()
    alignment = mean_elements(movement * residual.detach())
    initial = torch.isclose(times, torch.zeros_like(times), atol=eps, rtol=0)
    variance = (noise**2 * (1 - times) / times.clamp_min(eps) * deltas).clamp_min(0)
    std = torch.where(initial, torch.full_like(times, noise), variance.sqrt())
    coefficient = torch.where(initial, deltas, deltas * (1 + 0.5 * noise**2))
    proxy = (
        coefficient.square()
        * mean_elements(movement.square())
        / (2 * std.square().clamp_min(eps**2))
    )
    raw_rms = direction.square().flatten(1).mean(1).sqrt()
    active = (magnitude > 0) & (raw_rms > 0) & (times < 1)
    masked = active & (proxy >= threshold) & (alignment > 0)
    keep = active & ~masked
    result = torch.where(keep, unmasked, torch.zeros_like(unmasked))
    finite(result, proxy, alignment)
    return result, keep, masked


@dataclass
class GroupGuidance:
    directions: tuple[torch.Tensor, ...]  # Per-group latent shape; unit RMS or zero
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
        x.ndim == 2 and x.shape[0] == r.numel() == index.numel(),
        "guidance needs one reward and group per endpoint",
    )
    require(
        bool(torch.isfinite(x).all()) and bool(torch.isfinite(r).all()),
        "guidance inputs must be finite",
    )
    require(
        index.dtype in (torch.int32, torch.int64)
        and (
            index.numel() == 0
            or (int(index.min()) >= 0 and int(index.max()) < group_count)
        ),
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


def round_guidance(config, rollouts, rewards, group_shapes, *, reduce=None):
    """Guidance for one ART round from this rank's rollouts and the round rewards.

    A draw's group is its collection slot, `draw_index // draws_per_prompt`:
    prompt groups never key on prompt text, so a prompt repeated within one
    round still yields two independent groups.
    """
    draws = config.collection.draws_per_prompt
    require(
        len(group_shapes) == config.collection.prompts_per_round,
        "one latent shape required per prompt group",
    )
    device = rollouts[0].terminal.device
    directions = [None] * len(group_shapes)
    magnitudes = torch.zeros(len(group_shapes), device=device)
    active = torch.zeros(len(group_shapes), dtype=torch.bool, device=device)
    # All ranks know the selected prompt shapes, including buckets absent locally.
    for shape in sorted(set(group_shapes)):
        slots = [i for i, s in enumerate(group_shapes) if s == shape]
        indices = {slot: i for i, slot in enumerate(slots)}
        endpoints, values, groups = [], [], []
        for rollout in rollouts:
            for i, draw in enumerate(rollout.draw_indices):
                slot = draw // draws
                if slot not in indices:
                    continue
                require(
                    tuple(rollout.terminal.shape[1:]) == shape,
                    "group latent shape changed",
                )
                key = rollout.sample_ids[i]
                require(key in rewards, "missing local reward for the guidance")
                endpoints.append(rollout.terminal[i].flatten())
                values.append(rewards[key])
                groups.append(indices[slot])
        width = 1
        for dimension in shape:
            width *= dimension
        x = (
            torch.stack(endpoints).float()
            if endpoints
            else torch.empty((0, width), device=device)
        )
        stats = group_guidance(
            x,
            torch.tensor(values, device=device),
            torch.tensor(groups, device=device, dtype=torch.long),
            len(slots),
            expected_group_size=draws,
            reduce=reduce,
        )
        for i, slot in enumerate(slots):
            directions[slot] = stats["directions"][i].reshape(shape)
        magnitudes[slots] = stats["magnitudes"]
        active[slots] = stats["active"]
    return GroupGuidance(tuple(directions), magnitudes, active)


def accumulation(model, last):
    return model.no_sync() if not last and hasattr(model, "no_sync") else nullcontext()


def validate_art_collection(config, rollouts, round_index, window):
    c = config.collection
    total = c.prompts_per_round * c.draws_per_prompt
    if (
        window not in (0, 1)
        or not rollouts
        or any(
            not isinstance(r, ARTRollout)
            or r.collection_round != round_index
            or r.generator_step != rollouts[0].generator_step
            or len(r.sample_ids) != len(r.draw_indices)
            or any(
                not 0 <= draw < total
                or key
                != f"{round_index}:{draw // c.draws_per_prompt}:{draw % c.draws_per_prompt}"
                for key, draw in zip(r.sample_ids, r.draw_indices, strict=True)
            )
            for r in rollouts
        )
    ):
        raise ValueError("stale or missing ART trajectory collection")


def finish_student(roles, optimizer, loss_total):
    if roles.reference is not None and any(
        p.grad is not None for p in roles.reference.parameters()
    ):
        raise RuntimeError("student update leaked into the frozen reference")
    norm = optimizer.step(loss_total)
    roles.update_ema()
    optimizer.zero_grad()
    return norm


def update_art(
    config, roles, student, rollouts, guidance, optimizer, round_index, window
):
    """One ART optimizer update over one window of the round's trajectory items.

    `round_index` must be the collection round the cached trajectories belong to,
    so a stale cache fails instead of training against another round's anchor. Both
    windows regress the same frozen targets: the rollout's cached velocity at each
    visited state plus that prompt's guidance direction. A window trains each of
    its items once, so the second window's residual `G - d` carries the restoration
    force against whatever motion the first window produced.
    """
    validate_art_collection(config, rollouts, round_index, window)
    steps = config.collection.steps
    chunks = [
        (rollout, index, start)
        for rollout, index in art_window(config, rollouts, window)
        for start in range(0, steps, config.execution.microbatch)
    ]
    if not chunks:
        raise ValueError("an ART update window requires local trajectory items")
    total_items = (
        config.collection.prompts_per_round * config.collection.draws_per_prompt // 2
    ) * steps
    device = rollouts[0].terminal.device
    coefficient = config.reference_coefficient
    optimizer.zero_grad()
    loss_total = torch.zeros((), device=device)
    terms = {key: 0.0 for key in ("art", "reference")}
    items, kept, masked_items = 0, 0, 0
    for position, (rollout, index, start) in enumerate(chunks):
        stop = min(start + config.execution.microbatch, steps)
        size = stop - start
        grid = rollout.grid
        with accumulation(student, position == len(chunks) - 1):
            slot = rollout.draw_indices[index] // config.collection.draws_per_prompt
            z = rollout.states[index, start:stop].to(device).float()
            times = torch.tensor(grid[start:stop], device=device)
            deltas = torch.tensor(
                [grid[k + 1] - grid[k] for k in range(start, stop)], device=device
            )
            direction = (
                guidance.directions[slot].unsqueeze(0).expand(size, *z.shape[1:])
            )
            magnitude = guidance.magnitudes[slot].expand(size)
            conditioning = tuple(
                t[index : index + 1].expand(size, *t.shape[1:])
                for t in rollout.conditioning
            )
            new = student(z, times, conditioning)
            with torch.no_grad():
                anchor = (
                    roles.reference_field(z, times, conditioning)
                    if roles.reference is not None
                    else None
                )
            item_losses, keep, blocked = losses(
                new,
                rollout.predictions[index, start:stop].to(device).float(),
                times,
                deltas,
                direction,
                magnitude,
            )
            terms["art"] += float(global_loss(item_losses, total_items).detach())
            loss = item_losses
            if anchor is not None:
                anchor_term = reference_losses(new, anchor)
                terms["reference"] += float(
                    global_loss(coefficient * anchor_term, total_items).detach()
                )
                loss = loss + coefficient * anchor_term
            items += keep.numel()
            kept += int(keep.sum())
            masked_items += int(blocked.sum())
            loss = global_loss(loss, total_items)
            loss.backward()
            loss_total += loss.detach()
    return {
        "student_loss": float(loss_total),
        "student_grad_norm": finish_student(roles, optimizer, loss_total),
        "term_losses": terms,
        "window": window,
        "art_items": items,
        "art_kept": kept,
        "art_masked": masked_items,
    }
