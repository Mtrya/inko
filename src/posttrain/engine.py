"""Single-stage round updates with explicit role ownership."""

from contextlib import nullcontext

import torch

from .distributed import global_loss
from .objectives.art import losses as art_losses
from .objectives.art import reference_losses
from .rollout import ARTRollout, art_window, tensor_digest


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


def art_record(rollout, index, start, stop, times, *, kept, items, **extra):
    return {
        "id": rollout.sample_ids[index],
        "prompt_id": rollout.prompt_ids[index],
        "generator_step": rollout.generator_step,
        "collection_round": rollout.collection_round,
        "collection_seed": rollout.seeds[index],
        "conditioning": [tensor_digest(t) for t in rollout.conditioning],
        "steps": [start, stop],
        "time": tensor_digest(times),
        "state": tensor_digest(rollout.states[index, start:stop]),
        "cached": tensor_digest(rollout.predictions[index, start:stop]),
        "items": items,
        "kept": kept,
        **extra,
    }


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
    inputs, items, kept, masked_items = [], 0, 0, 0
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
            losses, keep, blocked = art_losses(
                new,
                rollout.predictions[index, start:stop].to(device).float(),
                times,
                deltas,
                direction,
                magnitude,
            )
            terms["art"] += float(global_loss(losses, total_items).detach())
            loss = losses
            if anchor is not None:
                anchor_term = reference_losses(new, anchor)
                terms["reference"] += float(
                    global_loss(coefficient * anchor_term, total_items).detach()
                )
                loss = loss + coefficient * anchor_term
            items += keep.numel()
            kept += int(keep.sum())
            masked_items += int(blocked.sum())
            inputs.append(
                art_record(
                    rollout,
                    index,
                    start,
                    stop,
                    times,
                    kept=int(keep.sum()),
                    items=keep.numel(),
                    window=window,
                    masked=int(blocked.sum()),
                    reference_prediction=(
                        None if anchor is None else tensor_digest(anchor)
                    ),
                )
            )
            loss = global_loss(loss, total_items)
            loss.backward()
            loss_total += loss.detach()
    return {
        "inputs": inputs,
        "student_loss": float(loss_total),
        "student_grad_norm": finish_student(roles, optimizer, loss_total),
        "term_losses": terms,
        "window": window,
        "art_items": items,
        "art_kept": kept,
        "art_masked": masked_items,
    }
