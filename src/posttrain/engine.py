"""Cold and joint updates with explicit role ownership and global reductions."""

from contextlib import nullcontext

import torch

from .distributed import global_loss
from .objectives.dmd import (
    clean_prediction,
    dmd_gradient,
    fake_loss,
    forward_noise,
    surrogate,
)
from .objectives.nft import losses as nft_losses
from .objectives.nft import reference_losses
from .objectives.rtdm import verify_replay, weighted_losses
from .rollout import (
    NFTRollout,
    TrainingRollout,
    nft_inputs,
    score_inputs,
    tensor_digest,
)
from .schedules import cold_point


def accumulation(model, last):
    return model.no_sync() if not last and hasattr(model, "no_sync") else nullcontext()


def validate_collection(rollouts, step):
    if not rollouts or any(
        not isinstance(r, TrainingRollout) or r.student_version != step
        for r in rollouts
    ):
        raise ValueError("stale or empty student collection")


def admit_values(config, rollouts, values, *, probabilities=False):
    r = rollouts[0].collection_round
    expected = {
        f"{r}:{slot}:{draw}"
        for slot in range(config.collection.prompts_per_round)
        for draw in range(config.collection.draws_per_prompt)
    }
    if values is None or set(values) != expected:
        raise ValueError(
            "complete global reward set required before any optimizer update"
        )
    vector = torch.tensor(list(values.values()), dtype=torch.float32)
    if (
        not bool(torch.isfinite(vector).all())
        or bool((vector < 0).any())
        or (probabilities and bool((vector > 1).any()))
    ):
        raise ValueError("invalid global reward weights/probabilities")


def input_record(role, rollout, times, noise, **extra):
    return {
        "role": role,
        "ids": rollout.sample_ids,
        "prompt_ids": rollout.prompt_ids,
        "student_version": rollout.student_version,
        "stage": rollout.stage,
        "collection_round": rollout.collection_round,
        "conditioning": [tensor_digest(t) for t in rollout.conditioning],
        "collection_seeds": rollout.seeds,
        "time": tensor_digest(times),
        "noise": tensor_digest(noise),
        **extra,
    }


def fake_updates(config, roles, fake, rollouts, optimizer, *, count=None, start=0):
    inputs, norms, totals = [], [], []
    total = config.collection.prompts_per_round * config.collection.draws_per_prompt
    for update in range(
        start, start + (config.fake.updates if count is None else count)
    ):
        targets = rollouts
        if config.fake.target_refresh == "update":
            from .rollout import fresh_rollouts

            targets = fresh_rollouts(
                config, roles.student_field, rollouts, "fake_target", optimizer.steps
            )
        optimizer.zero_grad()
        loss_total = torch.zeros((), device=rollouts[0].terminal.device)
        for index, rollout in enumerate(targets):
            target = getattr(rollout, config.fake.target)
            times, noise = score_inputs(config, rollout, "fake", update)
            z = forward_noise(target, times, noise)
            inputs.append(
                input_record(
                    "fake",
                    rollout,
                    times,
                    noise,
                    update=update,
                    target=tensor_digest(target),
                )
            )
            with accumulation(fake, index == len(targets) - 1):
                loss = global_loss(
                    fake_loss(fake(z, times, rollout.conditioning), target, noise),
                    total,
                )
                loss.backward()
                loss_total += loss.detach()
        if any(p.grad is not None for p in roles.student.parameters()):
            raise RuntimeError("fake update leaked into student")
        norms.append(optimizer.step(loss_total))
        totals.append(float(loss_total))
    optimizer.zero_grad()
    return {"fake_losses": totals, "fake_grad_norms": norms, "inputs": inputs}


def dmd_terms(config, roles, student, rollout, scale):
    times, noise = score_inputs(config, rollout, "dmd", 0)
    predicted = clean_prediction(
        rollout.selected_input,
        rollout.selected_time,
        student(
            rollout.selected_input,
            rollout.selected_time,
            rollout.conditioning,
            rollout.unconditional,
            config.collection.guidance,
        ),
    )
    verify_replay(predicted, rollout.selected_prediction)
    with torch.no_grad():
        z = forward_noise(predicted, times, noise)
        fake_clean = clean_prediction(
            z, times, roles.fake_field(z, times, rollout.conditioning)
        )
        with roles.real_view(scale):
            real_clean = clean_prediction(
                z,
                times,
                roles.teacher_field(
                    z,
                    times,
                    rollout.conditioning,
                    rollout.unconditional,
                    config.objective.teacher_guidance,
                ),
            )
        gradient = dmd_gradient(
            predicted,
            real_clean,
            fake_clean,
            times,
            estimator=config.objective.estimator,
            epsilon=config.objective.epsilon,
        )
    record = input_record(
        "dmd",
        rollout,
        times,
        noise,
        selected=tensor_digest(rollout.selected_input),
        selected_time=tensor_digest(rollout.selected_time),
        terminal=tensor_digest(rollout.terminal),
    )
    return surrogate(predicted, gradient), predicted, record


def prediction_norm(losses, predictions, total):
    # Stop at forward outputs: this diagnostic never visits parameter/DDP hooks.
    gradients = torch.autograd.grad(
        global_loss(losses, total), predictions, retain_graph=True
    )
    return float(
        torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt()
    )


def finish_student(config, roles, optimizer, loss_total):
    for model in (roles.fake, roles.old, roles.reference):
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
    fake,
    rollouts,
    student_optimizer,
    fake_optimizer,
    step,
    reward_weights=None,
):
    validate_collection(rollouts, step)
    if (
        step == 0
        and config.fake.initial_steps
        and fake_optimizer.steps != config.fake.initial_steps
    ):
        raise ValueError("first student update requires completed initial fake fitting")
    if config.objective.mode == "rtdm":
        if reward_weights is None:
            raise ValueError(
                "RT-DM requires complete global rewards before any optimizer update"
            )
        admit_values(config, rollouts, reward_weights)
    student_optimizer.zero_grad()
    first_fitted = step == 0 and config.fake.initial_steps > 0
    evidence = fake_updates(
        config, roles, fake, rollouts, fake_optimizer, count=0 if first_fitted else None
    )
    total = config.collection.prompts_per_round * config.collection.draws_per_prompt
    point = cold_point(config, step)
    loss_total = torch.zeros((), device=rollouts[0].terminal.device)
    for index, rollout in enumerate(rollouts):
        with accumulation(student, index == len(rollouts) - 1):
            losses, predicted, record = dmd_terms(
                config, roles, student, rollout, point["real_scale"]
            )
            if config.objective.mode == "rtdm":
                losses = weighted_losses(
                    losses,
                    torch.tensor(
                        [reward_weights[key] for key in rollout.sample_ids],
                        device=predicted.device,
                    ),
                )
            evidence["inputs"].append(record)
            loss = global_loss(losses, total)
            loss.backward()
            loss_total += loss.detach()
    evidence.update(
        student_loss=float(loss_total),
        student_grad_norm=finish_student(config, roles, student_optimizer, loss_total),
        cold_schedule=point,
        selected_replay_exact=True,
    )
    return evidence


def update_joint(
    config,
    roles,
    student,
    fake,
    nft_rollouts,
    dmd_rollouts,
    probabilities,
    student_optimizer,
    fake_optimizer,
    step,
    old_version,
):
    j, n = config.joint, config.nft
    if j.nft_coefficient > 0:
        if (
            not nft_rollouts
            or roles.old is None
            or any(
                not isinstance(r, NFTRollout)
                or r.stage != "joint"
                or r.old_version != old_version
                or not 0 <= step - r.student_version < config.joint_updates
                or r.collection_round != nft_rollouts[0].collection_round
                for r in nft_rollouts
            )
        ):
            raise ValueError("stale or missing old-policy collection")
        admit_values(config, nft_rollouts, probabilities, probabilities=True)
    if j.dmd_coefficient > 0:
        validate_collection(dmd_rollouts, step)
        if any(r.stage != "joint" for r in dmd_rollouts):
            raise ValueError("joint DMD requires fresh current-policy joint collection")
    batches = nft_rollouts if j.nft_coefficient > 0 else dmd_rollouts
    if (
        j.nft_coefficient > 0
        and j.dmd_coefficient > 0
        and (
            len(nft_rollouts) != len(dmd_rollouts)
            or any(
                a.terminal.shape != b.terminal.shape
                for a, b in zip(nft_rollouts, dmd_rollouts, strict=True)
            )
        )
    ):
        raise ValueError("joint objective microbatch partitions differ")
    student_optimizer.zero_grad()
    evidence = (
        fake_updates(config, roles, fake, dmd_rollouts, fake_optimizer)
        if j.dmd_coefficient > 0
        else {"fake_losses": [], "fake_grad_norms": [], "inputs": []}
    )
    total = config.collection.prompts_per_round * config.collection.draws_per_prompt
    device = batches[0].terminal.device
    loss_total = torch.zeros((), device=device)
    terms = {key: 0.0 for key in ("dmd", "nft", "reference")}
    norms = {key: [] for key in terms}
    for index, rollout in enumerate(batches):
        with accumulation(student, index == len(batches) - 1):
            losses = torch.zeros(len(rollout.sample_ids), device=device)
            if j.dmd_coefficient > 0:
                dmd, predicted, record = dmd_terms(
                    config, roles, student, dmd_rollouts[index], 0
                )
                weighted = j.dmd_coefficient * dmd
                norms["dmd"].append(prediction_norm(weighted, [predicted], total))
                terms["dmd"] += float(global_loss(weighted, total).detach())
                losses = losses + weighted
                evidence["inputs"].append(record)
            if j.nft_coefficient > 0:
                probability = torch.tensor(
                    [probabilities[key] for key in rollout.sample_ids], device=device
                )
                nft = torch.zeros_like(losses)
                ref = torch.zeros_like(losses)
                predictions = []
                for draw in range(n.time_draws):
                    times, noise = nft_inputs(config, rollout, step, draw)
                    z = forward_noise(rollout.terminal, times, noise)
                    with torch.no_grad():
                        old = roles.old_field(z, times, rollout.conditioning)
                        reference = (
                            roles.reference_field(z, times, rollout.conditioning)
                            if roles.reference is not None
                            else None
                        )
                    new = student(z, times, rollout.conditioning)
                    predictions.append(new)
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
                    if reference is not None:
                        ref = ref + reference_losses(new, reference) / n.time_draws
                    record = input_record(
                        "nft",
                        rollout,
                        times,
                        noise,
                        draw=draw,
                        old_version=old_version,
                        generator_step=step,
                        terminal=tensor_digest(rollout.terminal),
                        old_prediction=tensor_digest(old),
                        reference_prediction=(
                            None if reference is None else tensor_digest(reference)
                        ),
                    )
                    evidence["inputs"].append(record)
                weighted = j.nft_coefficient * nft
                norms["nft"].append(prediction_norm(weighted, predictions, total))
                terms["nft"] += float(global_loss(weighted, total).detach())
                losses = losses + weighted
                if j.reference_coefficient > 0:
                    weighted = j.reference_coefficient * ref
                    norms["reference"].append(
                        prediction_norm(weighted, predictions, total)
                    )
                    terms["reference"] += float(global_loss(weighted, total).detach())
                    losses = losses + weighted
            loss = global_loss(losses, total)
            loss.backward()
            loss_total += loss.detach()
    evidence.update(
        student_loss=float(loss_total),
        student_grad_norm=finish_student(config, roles, student_optimizer, loss_total),
        term_losses=terms,
        local_prediction_gradient_norms=norms,
        old_version=old_version,
        selected_replay_exact=True if j.dmd_coefficient > 0 else None,
        joint_time_distribution="uniform_physical",
        joint_real_scale=0,
    )
    return evidence
