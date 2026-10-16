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

from .config import load_training_config
from .records import atomic_json
from .schema import require


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
    from .rewards import RoundScores
    from .scoring.client import ScoreClient
    from .scoring.lifecycle import isolated_environment
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
        client = ScoreClient(config.scoring, assets["scoring"])
        active = output / "active_collection.json"
        if active.exists():
            client.release(json.loads(active.read_text())["id"])
            active.unlink()
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
                telemetry = TrainingMetrics(
                    tracker, config, output, attempt, contract["world_size"], saved
                )
            scores = RoundScores(config, client, output, attempt, next_score_round)
            for rank in range(contract["world_size"]):
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
                while scores.poll(
                    on_scored=None if tracker is None else tracker.scoring
                ):
                    pass
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
                        progress.extend(
                            p
                            for p in (
                                output / "progress" / f"{attempt}-rank-{rank:03d}.json",
                                output / "progress" / f"{attempt}-scoring.json",
                            )
                            if p.exists()
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
            if active.exists():
                client.release(json.loads(active.read_text())["id"])
                active.unlink()
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
    from .config import training_contract
    from .prompts import load_rollout_prompts

    assets = json.loads(args.assets.read_text()) if args.assets else None
    prompts = (
        load_rollout_prompts(assets["rollout"], config.collection.shapes)
        if assets
        else None
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "prompts": len(prompts) if prompts is not None else None,
                }
            )
        )
        return
    if not args.assets or not args.output:
        parser.error("--assets and --output are required")
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
        from src.utils.device import device_count

        from .scoring.client import ScoreClient

        info = ScoreClient(config.scoring, assets["scoring"]).info()
        require(
            {c.channel for c in config.rewards.channels} <= set(info["channels"]),
            "scoring service does not supply required rewards",
        )
        contract = training_contract(
            config, prompts, device_count(assets["device_type"]), info["semantics"]
        )
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
