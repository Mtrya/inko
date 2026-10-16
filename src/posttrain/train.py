"""Single-host ART trainer with resident learners and independent scorers."""

import argparse
import fcntl
import json
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
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
    track=False,
):
    from .round_scoring import score_ready_round
    from .scoring.lifecycle import Services, isolated_environment
    from .tracking import Tracker, TrainingMetrics

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
        saved = None
        if resume:
            from .checkpoint import inspect

            saved = inspect(resume)
            require(saved["contract"] == contract, "resume contract changed")
            require(
                saved["committed_rounds"] <= stop_after,
                "resume is past requested round",
            )
            next_score_round = saved["committed_rounds"]
        require(0 < stop_after <= config.training.rounds, "invalid stop round")
        try:
            tracked = (
                track
                or (output / "tracking.json").exists()
                or (resume is not None and (resume / "tracking.json").exists())
            )
            tracker = None
            telemetry = None
            if tracked:
                tracker = Tracker(
                    output, config_path.stem, resume=resume, config=asdict(config)
                ).__enter__()
                telemetry = TrainingMetrics(tracker, config, output, attempt, saved)
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
                argv += ["--stop-after", str(stop_after)]
                processes.append(
                    subprocess.Popen(
                        argv, env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                )
            started = time.time()
            while any(p.poll() is None for p in processes):
                if telemetry is not None:
                    telemetry.poll()
                services.check()
                while score_ready_round(
                    config,
                    services,
                    output,
                    attempt,
                    next_score_round,
                    on_scored=None if tracker is None else tracker.scoring,
                ):
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
                        progress.extend(
                            (output / "rounds").glob(
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
            if telemetry is not None:
                telemetry.poll()
                require(
                    telemetry.round == stop_after,
                    "completed training lacks SwanLab collection records",
                )
        finally:
            error = sys.exc_info()
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
            if tracker is not None:
                tracker.__exit__(*error)
    return {
        "completed_rounds": stop_after,
        "attempt": attempt,
        "checkpoint": str(output / "checkpoints" / f"round-{stop_after:06d}"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stop-after", type=int, help="Stop after this many collected rounds"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log this arm online to inko; recovery keeps its identity",
    )
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
    stop = args.stop_after if args.stop_after is not None else config.training.rounds
    require(0 < stop <= config.training.rounds, "invalid stop round")
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
        )
    else:
        contract = training_contract(config, prompts)
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
                    args.track,
                )
            )
        )


if __name__ == "__main__":
    main()
