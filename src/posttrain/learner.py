"""Resident single-stage learner; publish only complete collection rounds."""

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
):
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.models.inko import Inko
    from src.utils.device import peak_memory, set_device, synchronize
    from src.utils.encode_text import encode_text

    from . import checkpoint
    from .art import round_guidance, update_art
    from .distributed import all_reduce, initialize, prime_buckets, wrap
    from .rewards import publish_collection, wait_for_rewards
    from .roles import Roles
    from .rollout import PromptCursor, collect_art
    from .state import Clock, TrainingState
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
        if resume:
            saved, rank_saved = checkpoint.load(resume, contract, rank)
            clock = Clock.from_state(saved)
        else:
            clock = Clock()
        roles = Roles(
            Inko.from_single_file(
                assets["checkpoints"][config.initialization.asset]
            ).to(device),
            config,
        )
        state = TrainingState(
            config,
            roles,
            wrap(roles.student_field, device),
            clock,
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
            return (txt, pooled, mask)

        setup_rng = checkpoint.rng_state(device)
        cond = conditioning(
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
        prime_buckets(state.student, latent, times, cond)
        checkpoint.restore_rng(setup_rng, device)
        del setup_rng, cond, latent, times
        cursor = PromptCursor(len(contract["prompts"]), seed)
        if resume:
            cursor.load_state_dict(rank_saved["cursor"])
            checkpoint.restore_rng(rank_saved["rng"], device)
            checkpoint.assert_exact(rank_saved["rng"], checkpoint.rng_state(device))
            atomic_json(
                output / "recovery" / f"{attempt}-rank-{rank:03d}.json",
                {
                    "step": clock.step,
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
                keep_last=config.training.checkpoint_keep_last,
            )
        require(
            clock.committed_rounds <= stop_after,
            "resume is past requested committed round",
        )
        prompt_map = {p["id"]: p for p in contract["prompts"]}

        if rank == 0:
            atomic_json(
                output / "roles.json",
                {
                    "ownership": roles.ownership(),
                    "student_targets": roles.student_targets,
                },
            )
        atomic_json(
            output / "ready" / f"{attempt}-rank-{rank:03d}.json",
            {"ready_at": time.time(), "pid": os.getpid()},
        )

        def collect_round():
            chosen = cursor.take(config.collection.prompts_per_round)
            draws = [
                (
                    slot * config.collection.draws_per_prompt + draw,
                    contract["prompts"][p],
                    f"{clock.committed_rounds}:{slot}:{draw}",
                )
                for slot, p in enumerate(chosen)
                for draw in range(config.collection.draws_per_prompt)
            ]
            local = draws[rank :: len(config.execution.learner_devices)]
            result = []
            for begin in range(0, len(local), config.execution.microbatch):
                batch = local[begin : begin + config.execution.microbatch]
                cond = conditioning([p["text"] for _, p, _ in batch])
                result.append(
                    collect_art(
                        roles.student_field,
                        config,
                        [p for _, p, _ in batch],
                        [key for _, _, key in batch],
                        [index for index, _, _ in batch],
                        clock.committed_rounds,
                        clock.step,
                        cond,
                        device,
                    )
                )
            return result

        while clock.committed_rounds < stop_after:
            start = time.monotonic()
            r = clock.committed_rounds
            if rank == 0:
                atomic_json(
                    output / "attempts" / f"{attempt}-round-{r:06d}.json",
                    {
                        "attempt": attempt,
                        "round": r,
                        "started_at": time.time(),
                        "step": clock.step,
                    },
                )
            batches = collect_round()
            records = publish_collection(
                config, batches, prompt_map, vae, stats, output, attempt, r, rank
            )
            synchronize(device)
            collected = time.monotonic()
            values, state.last_reward = wait_for_rewards(
                config, output, attempt, r, records
            )
            rewarded = time.monotonic()
            guidance = round_guidance(config, batches, values, reduce=all_reduce)
            skipped = state.skip_inactive_art_round(guidance)
            synchronize(device)
            guided = time.monotonic()
            for inner in range(0 if skipped else config.updates_per_round):
                update_start = time.monotonic()
                evidence = update_art(
                    config,
                    roles,
                    state.student,
                    batches,
                    guidance,
                    state.student_optimizer,
                    r,
                    inner,
                )
                clock.update(config)
                synchronize(device)
                evidence.update(
                    step=clock.step,
                    collection_round=r,
                    inner_update=inner,
                    cursor=cursor.state_dict(),
                    update_seconds=time.monotonic() - update_start,
                    reference_enabled=bool(config.reference_coefficient),
                    attempt=attempt,
                    learning_rates=optimizer_rates(state.student_optimizer),
                )
                atomic_json(
                    output
                    / "updates"
                    / f"{attempt}-round-{r:06d}-inner-{inner:03d}-rank-{rank:03d}.json",
                    evidence,
                )
                if rank == 0:
                    print(evidence, flush=True)
            if not skipped:
                state.commit_round()
            synchronize(device)
            atomic_json(
                output / "rounds" / f"{attempt}-round-{r:06d}-rank-{rank:03d}.json",
                {
                    "clock": clock.state_dict(),
                    "cursor": cursor.state_dict(),
                    "complete": True,
                    "skipped_zero_guidance": skipped,
                    "active_group_fraction": float(guidance.active.float().mean()),
                    "peak_allocated_bytes": peak_memory(device),
                    "timings": {
                        "collection": collected - start,
                        "scoring_wait": rewarded - collected,
                        "guidance": guided - rewarded,
                        "round": time.monotonic() - start,
                    },
                },
            )
            del batches
            if (
                clock.committed_rounds % config.training.checkpoint_every == 0
                or clock.next_action(config) == "done"
                or clock.committed_rounds == stop_after
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
                    keep_last=config.training.checkpoint_keep_last,
                )
        atomic_json(
            output / "completion" / f"{attempt}-rank-{rank:03d}.json",
            {
                **clock.state_dict(),
                "rank": rank,
                "attempted_rounds": len(list((output / "attempts").glob("*.json"))),
                "peak_allocated_bytes": peak_memory(device),
            },
        )
    finally:
        dist.destroy_process_group()
