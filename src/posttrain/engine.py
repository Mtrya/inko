"""One cold round: fixed-student collection, fake updates, student update."""

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
from .objectives.rtdm import verify_replay, weighted_losses
from .rollout import score_inputs, tensor_digest
from .schedules import cold_point


def accumulation(model, last):
    return model.no_sync() if not last and hasattr(model, "no_sync") else nullcontext()


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
    if not rollouts or any(r.student_version != step for r in rollouts):
        raise ValueError("stale or empty student collection")
    total = config.collection.prompts_per_round * config.collection.draws_per_prompt
    device = rollouts[0].terminal.device
    point = cold_point(config, step)
    if config.objective.mode == "rtdm" and reward_weights is None:
        raise ValueError(
            "RT-DM requires complete global rewards before any optimizer update"
        )
    if config.objective.mode == "rtdm":
        expected = {
            f"{step}:{slot}:{draw}"
            for slot in range(config.collection.prompts_per_round)
            for draw in range(config.collection.draws_per_prompt)
        }
        if (
            set(reward_weights) != expected
            or not bool(
                torch.isfinite(torch.tensor(list(reward_weights.values()))).all()
            )
            or min(reward_weights.values()) < 0
        ):
            raise ValueError("RT-DM requires the complete finite global weight set")
    student_optimizer.zero_grad()
    fake_optimizer.zero_grad()
    inputs = []
    fake_norms = []
    fake_losses = []
    for update in range(config.fake.updates):
        fake_optimizer.zero_grad()
        loss_total = torch.zeros((), device=device)
        for index, rollout in enumerate(rollouts):
            target = getattr(rollout, config.fake.target)
            times, noise = score_inputs(config, rollout, step, "fake", update)
            z = forward_noise(target, times, noise)
            inputs.append(
                {
                    "role": "fake",
                    "update": update,
                    "ids": rollout.sample_ids,
                    "prompt_ids": rollout.prompt_ids,
                    "student_version": rollout.student_version,
                    "conditioning": [tensor_digest(t) for t in rollout.conditioning],
                    "collection_seeds": rollout.seeds,
                    "time": tensor_digest(times),
                    "noise": tensor_digest(noise),
                    "target": tensor_digest(target),
                }
            )
            with accumulation(fake, index == len(rollouts) - 1):
                losses = fake_loss(fake(z, times, rollout.conditioning), target, noise)
                loss = global_loss(losses, total)
                loss.backward()
                loss_total += loss.detach()
        if any(p.grad is not None for p in roles.student.parameters()):
            raise RuntimeError("fake update leaked into student")
        fake_norms.append(fake_optimizer.step(loss_total))
        fake_losses.append(float(loss_total))
    fake_optimizer.zero_grad()
    student_optimizer.zero_grad()
    loss_total = torch.zeros((), device=device)
    for index, rollout in enumerate(rollouts):
        times, noise = score_inputs(config, rollout, step, "dmd", 0)
        inputs.append(
            {
                "role": "dmd",
                "ids": rollout.sample_ids,
                "prompt_ids": rollout.prompt_ids,
                "student_version": rollout.student_version,
                "conditioning": [tensor_digest(t) for t in rollout.conditioning],
                "collection_seeds": rollout.seeds,
                "time": tensor_digest(times),
                "noise": tensor_digest(noise),
                "selected": tensor_digest(rollout.selected_input),
                "selected_time": tensor_digest(rollout.selected_time),
                "terminal": tensor_digest(rollout.terminal),
            }
        )
        with accumulation(student, index == len(rollouts) - 1):
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
                with roles.real_view(point["real_scale"]):
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
            losses = surrogate(predicted, gradient)
            if config.objective.mode == "rtdm":
                losses = weighted_losses(
                    losses,
                    torch.tensor(
                        [reward_weights[key] for key in rollout.sample_ids],
                        device=device,
                    ),
                )
            loss = global_loss(losses, total)
            loss.backward()
            loss_total += loss.detach()
    if any(p.grad is not None for p in roles.fake.parameters()):
        raise RuntimeError("student update leaked into fake/teacher")
    student_norm = student_optimizer.step(loss_total)
    roles.update_ema(config.training.ema_decay)
    student_optimizer.zero_grad()
    return {
        "student_loss": float(loss_total),
        "fake_losses": fake_losses,
        "student_grad_norm": student_norm,
        "fake_grad_norms": fake_norms,
        "inputs": inputs,
        "cold_schedule": point,
        "selected_replay_exact": True,
    }
