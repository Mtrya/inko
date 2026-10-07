"""Single-host plain DMD with resident learner-only DDP and round recovery."""

import argparse
import fcntl
import json
import random
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .records import atomic_json
from .schema import require
from .training_config import load_training_config


def learner(
    rank, config, assets, contract, output, resume, stop_after, rendezvous, attempt
):
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.models.inko import Inko
    from src.utils.encode_text import encode_text

    from . import checkpoint
    from .distributed import initialize, prime_buckets, wrap
    from .engine import update_round
    from .models import peak_memory, set_device, synchronize
    from .optim import RoleOptimizer
    from .roles import Roles
    from .rollout import PromptCursor, collect, tensor_digest

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
        roles = Roles(
            Inko.from_single_file(
                assets["checkpoints"][config.initialization.asset]
            ).to(device),
            Inko.from_single_file(assets["checkpoints"][config.teacher.asset]).to(
                device
            ),
            config,
        )
        student, fake = (
            wrap(roles.student_field, device),
            wrap(roles.fake_field, device),
        )
        so = RoleOptimizer(
            roles.student, config.student_optimizer, config.training.steps
        )
        fo = RoleOptimizer(
            roles.fake,
            config.fake_optimizer,
            config.training.steps * config.fake.updates,
        )
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
        # Prime both wrappers before initialization or restore so every real
        # update uses the steady reduction layout, including the first resumed
        # update. This is setup work: no cursor, role clock or optimizer steps.
        setup_rng = checkpoint.rng_state(device)
        with torch.no_grad():
            captions = [contract["prompts"][0]["text"]] * config.execution.microbatch
            txt, mask, pooled = encode_text(
                captions,
                text,
                tokenizer,
                pooling=True,
                exit_layer=config.collection.text_exit_layer,
            )
            conditioning = (txt, pooled, mask)
            unconditional = None
            if config.collection.guidance != 1:
                ut, um, up = encode_text(
                    [""] * len(captions),
                    text,
                    tokenizer,
                    pooling=True,
                    exit_layer=config.collection.text_exit_layer,
                )
                unconditional = (ut, up, um)
            latent = torch.zeros(
                len(captions),
                16,
                config.collection.height // 8,
                config.collection.width // 8,
                device=device,
            )
            times = torch.full((len(captions),), 0.5, device=device)
        prime_buckets(fake, latent, times, conditioning)
        prime_buckets(
            student,
            latent,
            times,
            conditioning,
            unconditional,
            config.collection.guidance,
        )
        checkpoint.restore_rng(setup_rng, device)
        del (
            setup_rng,
            captions,
            txt,
            mask,
            pooled,
            conditioning,
            unconditional,
            latent,
            times,
        )
        cursor = PromptCursor(len(contract["prompts"]), seed)
        step = 0

        def state():
            return {
                "student": roles.student.state_dict(),
                "fake": roles.fake.state_dict(),
                "ema": roles.ema,
                "student_optimizer": so.state_dict(),
                "fake_optimizer": fo.state_dict(),
                "step": step,
                "committed_rounds": step,
                "stage": "cold",
                "stage_generator_steps": step,
                "global_generator_steps": so.steps,
                "fake_steps": fo.steps,
                "samples_scored": 0,
                "samples_used": step
                * config.collection.prompts_per_round
                * config.collection.draws_per_prompt,
                "architecture": roles.student.get_config(),
                "ownership": roles.ownership(),
            }

        if resume:
            saved, rank_saved = checkpoint.load(resume, contract, rank)
            roles.student.load_state_dict(saved["student"])
            roles.fake.load_state_dict(saved["fake"])
            so.load_state_dict(saved["student_optimizer"])
            fo.load_state_dict(saved["fake_optimizer"])
            if roles.ema is not None:
                require(
                    saved["ema"].keys() == roles.ema.keys(), "EMA state missing/changed"
                )
                for name, value in roles.ema.items():
                    value.copy_(saved["ema"][name].to(device))
            step = saved["step"]
            cursor.load_state_dict(rank_saved["cursor"])
            checkpoint.assert_exact(saved, state())
            checkpoint.restore_rng(rank_saved["rng"], device)
            checkpoint.assert_exact(rank_saved["rng"], checkpoint.rng_state(device))
            atomic_json(
                output / "recovery" / f"{attempt}-rank-{rank:03d}.json",
                {
                    "step": step,
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
                state(),
                {"cursor": cursor.state_dict()},
                rank,
                device,
            )
        require(step <= stop_after, "resume is past requested committed boundary")
        base_digest = {
            n: tensor_digest(p)
            for n, p in roles.teacher.named_parameters()
            if not p.requires_grad
        }
        if rank == 0:
            atomic_json(
                output / "roles.json",
                {
                    "ownership": roles.ownership(),
                    "student_targets": roles.student_targets,
                    "fake_targets": roles.fake_targets,
                    "teacher_frozen": base_digest,
                },
            )
        atomic_json(
            output / "ready" / f"{attempt}-rank-{rank:03d}.json",
            {"ready_at": time.time()},
        )
        while step < stop_after:
            start = time.monotonic()
            if rank == 0:
                atomic_json(
                    output / "attempts" / f"{attempt}-{step:06d}.json",
                    {"attempt": attempt, "round": step, "started_at": time.time()},
                )
            chosen = cursor.take(config.collection.prompts_per_round)
            global_draws = [
                (contract["prompts"][p], f"{step}:{slot}:{draw}")
                for slot, p in enumerate(chosen)
                for draw in range(config.collection.draws_per_prompt)
            ]
            local = global_draws[rank :: len(config.execution.learner_devices)]
            rollouts = []
            for begin in range(0, len(local), config.execution.microbatch):
                batch = local[begin : begin + config.execution.microbatch]
                strings = [p["text"] for p, _ in batch]
                with torch.no_grad():
                    txt, mask, pooled = encode_text(
                        strings,
                        text,
                        tokenizer,
                        pooling=True,
                        exit_layer=config.collection.text_exit_layer,
                    )
                    conditioning = (txt, pooled, mask)
                    unconditional = None
                    if (
                        config.collection.guidance != 1
                        or config.objective.teacher_guidance != 1
                    ):
                        ut, um, up = encode_text(
                            [""] * len(batch),
                            text,
                            tokenizer,
                            pooling=True,
                            exit_layer=config.collection.text_exit_layer,
                        )
                        unconditional = (ut, up, um)
                rollouts.append(
                    collect(
                        roles.student_field,
                        config,
                        [p for p, _ in batch],
                        [key for _, key in batch],
                        step,
                        conditioning,
                        unconditional,
                        device,
                    )
                )
            evidence = update_round(
                config, roles, student, fake, rollouts, so, fo, step
            )
            step += 1
            synchronize(device)
            evidence.update(
                step=step,
                student_version=so.steps,
                fake_version=fo.steps,
                cursor=cursor.state_dict(),
                seconds=time.monotonic() - start,
                peak_allocated_bytes=peak_memory(device),
                attempt=attempt,
            )
            atomic_json(
                output / "updates" / f"{attempt}-round-{step:06d}-rank-{rank:03d}.json",
                evidence,
            )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in evidence.items()
                            if k not in {"inputs", "cursor"}
                        }
                    ),
                    flush=True,
                )
            del rollouts
            if step % config.training.checkpoint_every == 0 or step == stop_after:
                checkpoint.publish(
                    output / "checkpoints",
                    step,
                    contract,
                    state(),
                    {"cursor": cursor.state_dict()},
                    rank,
                    device,
                )
        require(
            base_digest
            == {
                n: tensor_digest(p)
                for n, p in roles.teacher.named_parameters()
                if not p.requires_grad
            },
            "frozen teacher changed",
        )
        atomic_json(
            output / "completion" / f"{attempt}-rank-{rank:03d}.json",
            {
                "step": step,
                "teacher_unchanged": True,
                "student_version": so.steps,
                "fake_version": fo.steps,
                "peak_allocated_bytes": peak_memory(device),
            },
        )
    finally:
        dist.destroy_process_group()


def launch(
    config, assets, contract, output, resume, stop_after, config_path, assets_path
):
    from .scoring.lifecycle import isolated_environment

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "manifest.json"
        if path.exists():
            require(
                json.loads(path.read_text()) == contract,
                "training run contract changed",
            )
            require(resume is not None, "existing run requires explicit --resume")
        else:
            atomic_json(path, contract)
        attempt = uuid.uuid4().hex
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        rendezvous = f"tcp://127.0.0.1:{port}"
        processes = []
        logs = []
        environment = isolated_environment(assets["learner"]["python_paths"])
        if assets["device_type"] == "npu":
            environment["HCCL_INTRA_ROCE_ENABLE"] = str(
                int(config.execution.intra_host_roce)
            )
            environment["HCCL_DETERMINISTIC"] = (
                "true" if config.execution.deterministic_algorithms else "false"
            )
        try:
            for rank in range(len(config.execution.learner_devices)):
                log = (output / f"{attempt}-learner-{rank:03d}.log").open("w")
                logs.append(log)
                argv = [
                    sys.executable,
                    "-m",
                    "src.posttrain.train",
                    "--config",
                    str(config_path),
                    "--assets",
                    str(assets_path),
                    "--output",
                    str(output),
                    "--learner-rank",
                    str(rank),
                    "--rendezvous",
                    rendezvous,
                    "--attempt",
                    attempt,
                    "--stop-after",
                    str(stop_after),
                ]
                if resume:
                    argv += ["--resume", str(resume)]
                processes.append(
                    subprocess.Popen(
                        argv, env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                )
            started = time.time()
            while any(p.poll() is None for p in processes):
                for rank, process in enumerate(processes):
                    if process.poll() is not None:
                        continue
                    ready = output / "ready" / f"{attempt}-rank-{rank:03d}.json"
                    if not ready.exists():
                        require(
                            time.time() - started < config.execution.startup_timeout,
                            "learner startup deadline exceeded",
                        )
                    else:
                        progress = list(
                            (output / "updates").glob(
                                f"{attempt}-round-*-rank-{rank:03d}.json"
                            )
                        )
                        latest = max(
                            [
                                ready.stat().st_mtime,
                                *(p.stat().st_mtime for p in progress),
                            ]
                        )
                        require(
                            time.time() - latest < config.execution.round_timeout,
                            "learner round deadline exceeded",
                        )
                require(
                    not any(p.poll() not in (None, 0) for p in processes),
                    "learner failed; inspect logs, recover committed checkpoint",
                )
                time.sleep(1)
            require(all(p.returncode == 0 for p in processes), "learner failed")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()
    return {
        "completed_step": stop_after,
        "attempt": attempt,
        "checkpoint": str(output / "checkpoints" / f"round-{stop_after:06d}"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--learner-rank", type=int)
    parser.add_argument("--rendezvous")
    parser.add_argument("--attempt")
    args = parser.parse_args()
    config = load_training_config(args.config)
    from src.evaluation.benchmarks.prompts import load_prompts

    from .provenance import REPO, training_contract

    prompts = load_prompts(
        REPO / config.collection.prompts,
        {"1:1": (config.collection.height, config.collection.width)},
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "learner_count": len(config.execution.learner_devices),
                    "prompts": len(prompts),
                }
            )
        )
        return
    if not args.assets or not args.output:
        parser.error("--assets and --output are required")
    assets = json.loads(args.assets.read_text())
    output = args.output.resolve()
    stop = args.stop_after if args.stop_after is not None else config.training.steps
    require(0 < stop <= config.training.steps, "invalid stop boundary")
    if args.learner_rank is not None:
        contract = json.loads((output / "manifest.json").read_text())
        learner(
            args.learner_rank,
            config,
            assets,
            contract,
            output,
            args.resume,
            stop,
            args.rendezvous,
            args.attempt,
        )
    else:
        contract = training_contract(config, assets, prompts)
        print(
            json.dumps(
                launch(
                    config,
                    assets,
                    contract,
                    output,
                    args.resume,
                    stop,
                    args.config.resolve(),
                    args.assets.resolve(),
                )
            )
        )


if __name__ == "__main__":
    main()
