"""Train to committed boundaries and evaluate after learner/scorer processes exit."""

import argparse
import fcntl
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

from src.evaluation.benchmarks.prompts import load_evaluation_prompts, load_prompts
from src.evaluation.benchmarks.run import REPO, run
from src.posttrain.checkpoint import inspect
from src.posttrain.config import Checkpoint
from src.posttrain.evaluation_config import (
    benchmark_config,
    boundaries,
    evaluation_due,
)
from src.posttrain.records import atomic_json
from src.posttrain.schema import require
from src.posttrain.training_config import load_training_config

from .export import export
from .readouts import panel


def committed_checkpoints(root):
    found = []
    for path in Path(root).glob("*"):
        if (
            path.is_dir()
            and not path.name.startswith(".")
            and (path / "manifest.json").exists()
        ):
            found.append((inspect(path)["committed_rounds"], path))
    return sorted(found)


def evaluate_checkpoint(checkpoint, training, assets, output, *, track=False):
    from src.posttrain.tracking import Tracker

    checkpoint = Path(checkpoint)
    context = (
        Tracker(checkpoint.parent.parent, None, resume=checkpoint, phase="evaluation")
        if track or (checkpoint / "tracking.json").exists()
        else nullcontext()
    )
    with context as tracker:
        return _evaluate_checkpoint(checkpoint, training, assets, output, tracker)


def _evaluate_checkpoint(checkpoint, training, assets, output, tracker):
    checkpoint, output = Path(checkpoint), Path(output)
    manifest = inspect(checkpoint)
    require(
        manifest["contract"]["config"] == json.loads(json.dumps(asdict(training))),
        "evaluation checkpoint training config differs",
    )
    interval = training.evaluation
    base = benchmark_config(training)
    exports, arms = [], list(base.arms)
    mapped = dict(assets["checkpoints"])
    for view in interval.weights:
        destination = output / "exports" / view
        if destination.exists():
            record = json.loads((destination / "export.json").read_text())
            require(record["weights"] == view, "saved interval export differs")
        else:
            record = export(checkpoint, view, destination)
        asset = f"student_{view}"
        mapped[asset] = str(destination / "weights.pt")
        arms.append(
            replace(base.arms[0], name=asset, checkpoint=Checkpoint(asset, view))
        )
        exports.append(record)
    config = replace(base, arms=tuple(arms))
    for suite in interval.suites:
        rows = load_evaluation_prompts(config, suite)
        target = output / suite
        report = run(
            config,
            {**assets, "checkpoints": mapped},
            rows,
            target,
            visual_only=suite == "monitor",
            suite=suite,
        )
        require(
            suite == "monitor"
            or all(
                metric["complete"]
                for arm in report["arms"].values()
                for metric in arm.values()
            ),
            "incomplete interval scoring; checkpoint evaluation failed",
        )
        panel(target, output / f"{suite}-blind", interval.bootstrap_seed)
        if tracker is not None:
            tracker.evaluation(
                report,
                target,
                config,
                manifest["step"],
            )
    receipt = {
        "step": manifest["step"],
        "committed_rounds": manifest["committed_rounds"],
        "exports": exports,
        "suites": interval.suites,
    }
    atomic_json(output / "evaluation.json", receipt)
    return receipt


def execute(config_path, assets_path, output, *, track=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    training = load_training_config(config_path)
    assets = json.loads(Path(assets_path).read_text())
    stops = list(boundaries(training))
    with (output / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        training_root = output / "training"
        checkpoints = committed_checkpoints(training_root / "checkpoints")
        if checkpoints:
            from src.posttrain.provenance import training_contract

            prompts = load_prompts(
                REPO / training.collection.prompts,
                {"1:1": (training.collection.height, training.collection.width)},
            )
            expected = training_contract(training, prompts)
            require(
                all(inspect(path)["contract"] == expected for _, path in checkpoints),
                "campaign recovery contract differs",
            )
        # Reconcile evaluations from already committed checkpoints before
        # resuming training. An interrupted evaluation never becomes a skip.
        for round_number, path in checkpoints:
            if evaluation_due(training, round_number):
                destination = output / "evaluation" / path.name
                if (destination / "evaluation.json").exists():
                    continue
                evaluate_checkpoint(
                    path,
                    training,
                    assets,
                    destination,
                    track=track,
                )
        last = checkpoints[-1][1] if checkpoints else None
        completed = checkpoints[-1][0] if checkpoints else 0
        for stop in stops:
            if stop <= completed:
                continue
            argv = [
                sys.executable,
                "-m",
                "src.posttrain.train",
                "--config",
                str(Path(config_path).resolve()),
                "--assets",
                str(Path(assets_path).resolve()),
                "--output",
                str(training_root),
                "--stop-after",
                str(stop),
            ]
            if last is not None:
                argv += ["--resume", str(last)]
            if track:
                argv += ["--track"]
            with (output / f"segment-{stop:06d}.log").open("a") as log:
                subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
            last = training_root / "checkpoints" / f"round-{stop:06d}"
            require(
                inspect(last)["committed_rounds"] == stop,
                "training segment lacks its committed endpoint",
            )
            if evaluation_due(training, stop):
                evaluate_checkpoint(
                    last,
                    training,
                    assets,
                    output / "evaluation" / last.name,
                    track=track,
                )
        atomic_json(
            output / "complete.json",
            {
                "committed_rounds": training.training.rounds,
                "step": inspect(last)["step"],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Keep training and interval readouts in one online SwanLab experiment",
    )
    args = parser.parse_args()
    if args.validate_only:
        c = load_training_config(args.config)
        benchmark_config(c)
        stops = list(boundaries(c))
        print(
            json.dumps(
                {
                    "valid": True,
                    "committed_stops": stops,
                    "evaluations": [s for s in stops if evaluation_due(c, s)],
                }
            )
        )
    else:
        if args.assets is None or args.output is None:
            parser.error("--assets and --output are required")
        execute(args.config, args.assets, args.output, track=args.track)
