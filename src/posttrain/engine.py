"""Single-stage NFT round updates with explicit role ownership."""

from contextlib import nullcontext

import torch

from .distributed import global_loss
from .objectives.nft import forward_noise, reference_losses
from .objectives.nft import losses as nft_losses
from .rollout import NFTRollout, nft_inputs, tensor_digest


def accumulation(model, last):
    return model.no_sync() if not last and hasattr(model, "no_sync") else nullcontext()


def validate_collection(rollouts, old_version):
    if not rollouts or any(
        not isinstance(r, NFTRollout)
        or r.old_version != old_version
        or r.collection_round != rollouts[0].collection_round
        or r.generator_step != rollouts[0].generator_step
        for r in rollouts
    ):
        raise ValueError("stale or missing old-policy collection")


def admit_values(config, rollouts, probabilities):
    r = rollouts[0].collection_round
    expected = {
        f"{r}:{slot}:{draw}"
        for slot in range(config.collection.prompts_per_round)
        for draw in range(config.collection.draws_per_prompt)
    }
    if probabilities is None or set(probabilities) != expected:
        raise ValueError(
            "complete global reward set required before any optimizer update"
        )
    vector = torch.tensor(list(probabilities.values()), dtype=torch.float32)
    if (
        not bool(torch.isfinite(vector).all())
        or bool((vector < 0).any())
        or bool((vector > 1).any())
    ):
        raise ValueError("invalid global reward probabilities")


def input_record(rollout, times, noise, **extra):
    return {
        "ids": rollout.sample_ids,
        "prompt_ids": rollout.prompt_ids,
        "generator_step": rollout.generator_step,
        "collection_round": rollout.collection_round,
        "conditioning": [tensor_digest(t) for t in rollout.conditioning],
        "collection_seeds": rollout.seeds,
        "time": tensor_digest(times),
        "noise": tensor_digest(noise),
        **extra,
    }


def finish_student(roles, optimizer, loss_total):
    for model in (roles.old, roles.reference):
        if model is not None and any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("student update leaked into a detached role")
    norm = optimizer.step(loss_total)
    roles.update_ema()
    optimizer.zero_grad()
    return norm


def update_round(
    config,
    roles,
    student,
    rollouts,
    probabilities,
    optimizer,
    step,
    old_version,
):
    validate_collection(rollouts, old_version)
    admit_values(config, rollouts, probabilities)
    n = config.nft
    optimizer.zero_grad()
    total = config.collection.prompts_per_round * config.collection.draws_per_prompt
    loss_total = torch.zeros((), device=rollouts[0].terminal.device)
    terms = {key: 0.0 for key in ("nft", "reference")}
    inputs = []
    for index, rollout in enumerate(rollouts):
        with accumulation(student, index == len(rollouts) - 1):
            device = rollout.terminal.device
            probability = torch.tensor(
                [probabilities[key] for key in rollout.sample_ids], device=device
            )
            nft = torch.zeros(len(rollout.sample_ids), device=device)
            reference = torch.zeros_like(nft)
            for draw in range(n.time_draws):
                times, noise = nft_inputs(config, rollout, step, draw)
                z = forward_noise(rollout.terminal, times, noise)
                with torch.no_grad():
                    old = roles.old_field(z, times, rollout.conditioning)
                    anchor = (
                        roles.reference_field(z, times, rollout.conditioning)
                        if roles.reference is not None
                        else None
                    )
                new = student(z, times, rollout.conditioning)
                nft = (
                    nft
                    + nft_losses(
                        new,
                        old,
                        rollout.terminal,
                        noise,
                        times,
                        probability,
                        parameterization=n.parameterization,
                        beta=n.beta,
                        clip=n.advantage_clip,
                        epsilon=n.epsilon,
                    )
                    / n.time_draws
                )
                if anchor is not None:
                    reference = reference + reference_losses(new, anchor) / n.time_draws
                inputs.append(
                    input_record(
                        rollout,
                        times,
                        noise,
                        draw=draw,
                        old_version=old_version,
                        terminal=tensor_digest(rollout.terminal),
                        old_prediction=tensor_digest(old),
                        reference_prediction=(
                            None if anchor is None else tensor_digest(anchor)
                        ),
                    )
                )
            terms["nft"] += float(global_loss(nft, total).detach())
            loss = nft
            if roles.reference is not None:
                terms["reference"] += float(
                    global_loss(n.reference_coefficient * reference, total).detach()
                )
                loss = loss + n.reference_coefficient * reference
            loss = global_loss(loss, total)
            loss.backward()
            loss_total += loss.detach()
    return {
        "inputs": inputs,
        "student_loss": float(loss_total),
        "student_grad_norm": finish_student(roles, optimizer, loss_total),
        "term_losses": terms,
        "old_version": old_version,
    }
