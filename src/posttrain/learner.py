"""Resident learner process; publish only complete cold/joint collection rounds."""

import json
import os
import random
import time

from .records import atomic_json
from .schema import require


def learner(
    rank,
    config,
    assets,
    contract,
    output,
    resume,
    stop_after,
    rendezvous,
    attempt,
    stop_after_fit=None,
):
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.models.inko import Inko
    from src.utils.encode_text import encode_text

    from . import checkpoint
    from .distributed import initialize, prime_buckets, wrap
    from .engine import fake_updates, update_joint, update_round
    from .initialization import load_boundary
    from .models import peak_memory, set_device, synchronize
    from .probes import fake_regression_probe
    from .roles import Roles
    from .rollout import PromptCursor, collect, collect_nft, tensor_digest
    from .round_scoring import publish_collection, wait_for_rewards
    from .session import TrainingState
    from .stages import Clock, rounds_before
    from .tracking import optimizer_rates

    device = set_device(
        assets["device_type"],
        config.execution.learner_devices[rank],
        config.execution.cpu_threads,
    )
    torch.use_deterministic_algorithms(config.execution.deterministic_algorithms)
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    initialize(
        rank,
        len(config.execution.learner_devices),
        device,
        rendezvous,
        config.execution.collective_timeout,
    )
    try:
        tracking_path = output / "tracking.json"
        tracking = (
            json.loads(tracking_path.read_text()) if tracking_path.exists() else None
        )
        saved = rank_saved = None
        boundary = contract["boundary"]
        initial_generator_steps = 0 if boundary is None else boundary["generator_steps"]
        initial_fake_steps = 0 if boundary is None else boundary["fake_steps"]
        if resume:
            saved, rank_saved = checkpoint.load(resume, contract, rank)
            clock = Clock.from_state(saved)
        else:
            clock = Clock.initial(config, initial_generator_steps, initial_fake_steps)
        stage = clock.stage
        roles = Roles(
            Inko.from_single_file(
                assets["checkpoints"][config.initialization.asset]
            ).to(device),
            (
                Inko.from_single_file(assets["checkpoints"][config.teacher.asset]).to(
                    device
                )
                if config.dmd_enabled(stage)
                else None
            ),
            config,
            stage=stage,
        )
        if not resume and contract["boundary"] is not None:
            g, f = load_boundary(config, assets, roles, contract["boundary"])
            require(
                (g, f) == (clock.global_generator_steps, clock.fake_steps),
                "boundary clocks differ",
            )
        if stage == "joint":
            roles.enter_joint()
        state = TrainingState(
            config,
            roles,
            wrap(roles.student_field, device),
            wrap(roles.fake_field, device) if roles.fake is not None else None,
            clock,
            initial_generator_steps=initial_generator_steps,
            initial_fake_steps=initial_fake_steps,
        )
        if resume:
            state.load_state_dict(saved)
            clock = state.clock
        dtype = getattr(torch, config.execution.precision)
        text = (
            AutoModelForCausalLM.from_pretrained(
                assets["text_encoder"],
                torch_dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            assets["text_encoder"], local_files_only=True
        )
        vae = stats = None
        if config.requires_scores:
            from diffusers import AutoencoderKLQwenImage

            from src.utils.vae_codec import get_vae_stats

            vae = (
                AutoencoderKLQwenImage.from_pretrained(
                    assets["vae"], torch_dtype=dtype, local_files_only=True
                )
                .to(device)
                .eval()
                .requires_grad_(False)
            )
            stats = get_vae_stats(assets["vae"], device)

        @torch.no_grad()
        def conditioning(strings):
            txt, mask, pooled = encode_text(
                strings,
                text,
                tokenizer,
                pooling=True,
                exit_layer=config.collection.text_exit_layer,
            )
            cond = (txt, pooled, mask)
            uncond = None
            if config.collection.guidance != 1 or (
                config.dmd_enabled(clock.stage)
                and config.objective.teacher_guidance != 1
            ):
                ut, um, up = encode_text(
                    [""] * len(strings),
                    text,
                    tokenizer,
                    pooling=True,
                    exit_layer=config.collection.text_exit_layer,
                )
                uncond = (ut, up, um)
            return cond, uncond

        setup_rng = checkpoint.rng_state(device)
        cond, uncond = conditioning(
            [contract["prompts"][0]["text"]] * config.execution.microbatch
        )
        latent = torch.zeros(
            config.execution.microbatch,
            16,
            config.collection.height // 8,
            config.collection.width // 8,
            device=device,
        )
        times = torch.full((config.execution.microbatch,), 0.5, device=device)
        if state.fake is not None:
            prime_buckets(state.fake, latent, times, cond)
        prime_buckets(
            state.student, latent, times, cond, uncond, config.collection.guidance
        )
        checkpoint.restore_rng(setup_rng, device)
        del setup_rng, cond, uncond, latent, times
        cursor = PromptCursor(len(contract["prompts"]), seed)
        if resume:
            cursor.load_state_dict(rank_saved["cursor"])
            checkpoint.restore_rng(rank_saved["rng"], device)
            checkpoint.assert_exact(rank_saved["rng"], checkpoint.rng_state(device))
            atomic_json(
                output / "recovery" / f"{attempt}-rank-{rank:03d}.json",
                {
                    "step": clock.step,
                    "stage": clock.stage,
                    "exact_state": True,
                    "exact_rng": True,
                    "cursor": cursor.state_dict(),
                },
            )
            del saved, rank_saved
        else:
            checkpoint.publish(
                output / "checkpoints",
                0,
                contract,
                state.state_dict(),
                {"cursor": cursor.state_dict()},
                rank,
                device,
                tracking,
            )
        require(clock.step <= stop_after, "resume is past requested committed boundary")
        rounds_before(config, stop_after)
        prompt_map = {p["id"]: p for p in contract["prompts"]}

        def frozen():
            return {
                name: {
                    n: tensor_digest(p)
                    for n, p in model.named_parameters()
                    if name == "reference" or not p.requires_grad
                }
                for name, model in (
                    ("teacher", roles.teacher),
                    ("student_base", roles.student),
                    ("reference", roles.reference),
                )
                if model is not None
            }

        frozen_base = frozen()

        def record_roles():
            if rank == 0:
                atomic_json(
                    output / f"roles-{clock.stage}.json",
                    {
                        "ownership": roles.ownership(),
                        "student_targets": roles.student_targets,
                        "fake_targets": roles.fake_targets,
                        "frozen": frozen_base,
                    },
                )

        record_roles()
        atomic_json(
            output / "ready" / f"{attempt}-rank-{rank:03d}.json",
            {"ready_at": time.time(), "pid": os.getpid(), "stage": clock.stage},
        )

        def collect_round(old_policy):
            chosen = cursor.take(config.collection.prompts_per_round)
            draws = [
                (contract["prompts"][p], f"{clock.committed_rounds}:{slot}:{draw}")
                for slot, p in enumerate(chosen)
                for draw in range(config.collection.draws_per_prompt)
            ]
            local = draws[rank :: len(config.execution.learner_devices)]
            result = []
            for begin in range(0, len(local), config.execution.microbatch):
                batch = local[begin : begin + config.execution.microbatch]
                cond, uncond = conditioning([p["text"] for p, _ in batch])
                prompts = [p for p, _ in batch]
                ids = [key for _, key in batch]
                if old_policy:
                    result.append(
                        collect_nft(
                            roles.old_field,
                            config,
                            prompts,
                            ids,
                            clock.committed_rounds,
                            clock.old_policy_version,
                            clock.global_generator_steps,
                            cond,
                            uncond,
                            device,
                        )
                    )
                else:
                    result.append(
                        collect(
                            roles.student_field,
                            config,
                            prompts,
                            ids,
                            clock.committed_rounds,
                            cond,
                            uncond,
                            device,
                            stage=clock.stage,
                            generator_step=clock.global_generator_steps,
                            stream=(
                                "collection"
                                if clock.stage == "cold"
                                else "dmd_collection"
                            ),
                        )
                    )
            return result

        fit_limit = (
            config.fake.initial_steps if stop_after_fit is None else stop_after_fit
        )
        if stop_after_fit is not None:
            require(
                clock.step == 0 and clock.initial_fit_steps <= fit_limit,
                "resume is past the requested fitting boundary",
            )
        if clock.initial_fit_steps < fit_limit:
            # Replay the first cold collection after fitting. Its prompt cursor,
            # student, keyed noise and cold schedules stay at their entry state.
            entry_cursor = cursor.state_dict()
            fit_batches = collect_round(False)
            cursor.load_state_dict(entry_cursor)
            while clock.initial_fit_steps < fit_limit:
                start = time.monotonic()
                evidence = fake_updates(
                    config,
                    roles,
                    state.fake,
                    fit_batches,
                    state.fake_optimizer,
                    count=1,
                    start=clock.initial_fit_steps,
                )
                clock.fit(config)
                synchronize(device)
                evidence.update(
                    initial_fit_steps=clock.initial_fit_steps,
                    student_version=clock.global_generator_steps,
                    fake_version=clock.fake_steps,
                    seconds=time.monotonic() - start,
                    peak_allocated_bytes=peak_memory(device),
                    attempt=attempt,
                    learning_rates={"fake": optimizer_rates(state.fake_optimizer)},
                )
                atomic_json(
                    output
                    / "fitting"
                    / f"{attempt}-fit-{clock.initial_fit_steps:06d}-rank-{rank:03d}.json",
                    evidence,
                )
                if (
                    clock.initial_fit_steps % config.fake.initial_checkpoint_every == 0
                    or clock.initial_fit_steps == fit_limit
                ):
                    checkpoint.publish(
                        output / "checkpoints",
                        0,
                        contract,
                        state.state_dict(),
                        {"cursor": cursor.state_dict()},
                        rank,
                        device,
                        tracking,
                    )
            del fit_batches

        while stop_after_fit is None and clock.step < stop_after:
            if clock.next_action(config) == "transition":
                require(frozen_base == frozen(), "frozen cold role changed")
                before = clock.state_dict()
                state.enter_joint()
                clock = state.clock
                frozen_base = frozen()
                record_roles()
                atomic_json(
                    output / "transitions" / f"{attempt}-rank-{rank:03d}.json",
                    {
                        "before": before,
                        "after": clock.state_dict(),
                        "optimizer_reset": True,
                        "evaluation_ema_reset": True,
                        "old_initialized_from_live": roles.old is not None,
                        "reference_initialized_from_live": roles.reference is not None,
                    },
                )
            start = time.monotonic()
            r = clock.committed_rounds
            if rank == 0:
                atomic_json(
                    output / "attempts" / f"{attempt}-round-{r:06d}.json",
                    {
                        "attempt": attempt,
                        "round": r,
                        "stage": clock.stage,
                        "started_at": time.time(),
                        "step": clock.step,
                    },
                )
            nft_on = clock.stage == "joint" and config.joint.nft_coefficient > 0
            batches = collect_round(nft_on)
            reward = None
            scored = (
                clock.stage == "cold" and config.objective.mode == "rtdm"
            ) or nft_on
            if scored:
                records = publish_collection(
                    config, batches, prompt_map, vae, stats, output, attempt, r, rank
                )
                reward, state.last_reward = wait_for_rewards(
                    config, output, attempt, r, rank, records
                )
            old_version = clock.old_policy_version
            updates = config.joint_updates if clock.stage == "joint" else 1
            for inner in range(updates):
                dmd = batches
                if nft_on and config.joint.dmd_coefficient > 0:
                    dmd = [
                        collect(
                            roles.student_field,
                            config,
                            [prompt_map[p] for p in b.prompt_ids],
                            [f"{key}:dmd:{inner}" for key in b.sample_ids],
                            r,
                            b.conditioning,
                            b.unconditional,
                            device,
                            stage="joint",
                            generator_step=clock.global_generator_steps,
                            stream="dmd_collection",
                        )
                        for b in batches
                    ]
                if clock.stage == "cold":
                    evidence = update_round(
                        config,
                        roles,
                        state.student,
                        state.fake,
                        batches,
                        state.student_optimizer,
                        state.fake_optimizer,
                        clock.global_generator_steps,
                        reward,
                    )
                else:
                    evidence = update_joint(
                        config,
                        roles,
                        state.student,
                        state.fake,
                        batches if nft_on else None,
                        dmd if config.joint.dmd_coefficient > 0 else None,
                        reward,
                        state.student_optimizer,
                        state.fake_optimizer,
                        clock.global_generator_steps,
                        old_version,
                    )
                clock.update(config)
                if (
                    state.fake is not None
                    and config.fake_probe.every
                    and clock.global_generator_steps % config.fake_probe.every == 0
                ):
                    probe_rng = checkpoint.rng_state(device)
                    evidence["fake_probe"] = fake_regression_probe(
                        config,
                        roles,
                        dmd,
                        clock.global_generator_steps,
                    )
                    checkpoint.restore_rng(probe_rng, device)
                synchronize(device)
                evidence.update(
                    step=clock.step,
                    stage=clock.stage,
                    collection_round=r,
                    inner_update=inner,
                    student_version=clock.global_generator_steps,
                    fake_version=clock.fake_steps,
                    old_version=old_version,
                    cursor=cursor.state_dict(),
                    reward=state.last_reward,
                    seconds=time.monotonic() - start,
                    peak_allocated_bytes=peak_memory(device),
                    attempt=attempt,
                    learning_rates={
                        "student": optimizer_rates(state.student_optimizer),
                        "fake": optimizer_rates(state.fake_optimizer),
                    },
                )
                atomic_json(
                    output
                    / "updates"
                    / f"{attempt}-round-{r:06d}-inner-{inner:03d}-rank-{rank:03d}.json",
                    evidence,
                )
                if rank == 0:
                    print(
                        {
                            k: v
                            for k, v in evidence.items()
                            if k not in {"inputs", "reward"}
                        },
                        flush=True,
                    )
                if nft_on and config.joint.dmd_coefficient > 0:
                    del dmd
            decay = state.commit_round()
            atomic_json(
                output / "rounds" / f"{attempt}-round-{r:06d}-rank-{rank:03d}.json",
                {
                    "clock": clock.state_dict(),
                    "old_refresh_decay": decay,
                    "cursor": cursor.state_dict(),
                    "complete": True,
                },
            )
            del batches
            if (
                clock.stage_generator_steps % config.training.checkpoint_every == 0
                or clock.next_action(config) in {"transition", "done"}
                or clock.step == stop_after
            ):
                checkpoint.publish(
                    output / "checkpoints",
                    clock.step,
                    contract,
                    state.state_dict(),
                    {"cursor": cursor.state_dict()},
                    rank,
                    device,
                    tracking,
                )
        require(frozen_base == frozen(), "frozen training role changed")
        atomic_json(
            output / "completion" / f"{attempt}-rank-{rank:03d}.json",
            {
                **clock.state_dict(),
                "rank": rank,
                "frozen_unchanged": True,
                "student_version": clock.global_generator_steps,
                "fake_version": clock.fake_steps,
                "attempted_rounds": len(list((output / "attempts").glob("*.json"))),
                "peak_allocated_bytes": peak_memory(device),
            },
        )
    finally:
        dist.destroy_process_group()
