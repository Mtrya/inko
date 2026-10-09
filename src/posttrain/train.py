"""Single-host cold/joint trainer with resident learners and independent scorers."""

import argparse
import fcntl
import json
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .records import atomic_json
from .schema import require
from .training_config import load_training_config


def launch(
    config,
    assets,
    contract,
    output,
    resume,
    stop_after,
    config_path,
    assets_path,
    stop_after_fit=None,
):
    from .round_scoring import score_ready_round
    from .scoring.lifecycle import Services, isolated_environment
    from .stages import round_position, rounds_before

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "manifest.json"
        if path.exists():
            require(
                json.loads(path.read_text()) == contract,
                "training run contract changed",
            )
            # The manifest precedes model loading and the first publication.
            # With no committed state, restart from the declared initialization.
            published = any(
                p.is_dir()
                and not p.name.startswith(".")
                and (p / "manifest.json").exists()
                for p in (output / "checkpoints").glob("*")
            )
            require(
                resume is not None or not published,
                "existing committed run requires explicit --resume",
            )
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
        services = Services(
            config.scorers,
            assets.get("services", {}),
            output,
            config.execution.learner_devices,
            config.execution.startup_timeout,
        )
        next_score_round = 0
        if resume:
            from .checkpoint import inspect

            saved = inspect(resume)
            require(saved["contract"] == contract, "resume contract changed")
            require(saved["step"] <= stop_after, "resume is past requested boundary")
            next_score_round = rounds_before(config, saved["step"])
        last_round = rounds_before(config, stop_after)
        initial_steps = (
            0
            if contract["boundary"] is None
            else contract["boundary"]["generator_steps"]
        )
        try:
            services.__enter__()
            atomic_json(
                output / "service_startup" / f"{attempt}.json", services.startup
            )
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
                ]
                if resume:
                    argv += ["--resume", str(resume)]
                if stop_after_fit is not None:
                    argv += ["--stop-after-fit", str(stop_after_fit)]
                else:
                    argv += ["--stop-after", str(stop_after)]
                processes.append(
                    subprocess.Popen(
                        argv, env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                )
            started = time.time()
            while any(p.poll() is None for p in processes):
                services.check()
                while next_score_round < last_round:
                    stage, _, _ = round_position(config, next_score_round)
                    scored = (stage == "cold" and config.objective.mode == "rtdm") or (
                        stage == "joint" and config.joint.nft_coefficient > 0
                    )
                    if scored and not score_ready_round(
                        config,
                        services,
                        output,
                        attempt,
                        next_score_round,
                        initial_generator_steps=initial_steps,
                    ):
                        break
                    next_score_round += 1
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
                        progress += list(
                            (output / "fitting").glob(
                                f"{attempt}-fit-*-rank-{rank:03d}.json"
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
            services.close()
    return {
        "completed_step": stop_after if stop_after_fit is None else 0,
        "attempt": attempt,
        "checkpoint": str(
            output
            / "checkpoints"
            / (
                f"step-{stop_after:06d}"
                if stop_after_fit is None
                else f"fit-{stop_after_fit:06d}"
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--stop-after-fit",
        type=int,
        help="Commit this many initial fake updates, then stop before any student update",
    )
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
    if args.stop_after_fit is not None:
        require(args.stop_after is None, "choose a fitting or student stop boundary")
        require(
            0 < args.stop_after_fit <= config.fake.initial_steps,
            "invalid initial fitting stop boundary",
        )
    from .stages import rounds_before

    rounds_before(config, stop)
    if args.learner_rank is not None:
        contract = json.loads((output / "manifest.json").read_text())
        from .learner import learner

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
            args.stop_after_fit,
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
                    args.stop_after_fit,
                )
            )
        )


if __name__ == "__main__":
    main()
