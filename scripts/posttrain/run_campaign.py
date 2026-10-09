"""Train to committed boundaries and evaluate after learner/scorer processes exit."""

import argparse
import fcntl
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from src.evaluation.benchmarks.prompts import load_prompts
from src.evaluation.benchmarks.run import REPO, run
from src.posttrain.checkpoint import inspect
from src.posttrain.config import Arm, Checkpoint, load_config
from src.posttrain.evaluation_config import (
    boundaries,
    evaluation_due,
    load_interval_config,
)
from src.posttrain.records import atomic_json, file_hash
from src.posttrain.schema import require
from src.posttrain.training_config import load_training_config

from .export import export
from .readouts import panel

SUITES = {
    "development": "suite",
    "bringup": "bringup_suite",
    "monitor": "monitor_suite",
    "confirmation": "confirmation_suite",
}


def committed_checkpoints(root):
    found = []
    for path in Path(root).glob("*"):
        if (
            path.is_dir()
            and not path.name.startswith(".")
            and (path / "manifest.json").exists()
        ):
            manifest = inspect(path)
            found.append((manifest["step"], manifest["initial_fit_steps"], path))
    return sorted(found)


def evaluate_checkpoint(checkpoint, training, interval, assets, output):
    checkpoint, output = Path(checkpoint), Path(output)
    manifest = inspect(checkpoint)
    require(
        manifest["contract"]["config"] == json.loads(json.dumps(asdict(training))),
        "evaluation checkpoint training config differs",
    )
    sampling_path = REPO / interval.sampling_config
    require(
        file_hash(sampling_path) == interval.sampling_sha256,
        "interval sampling configuration bytes differ",
    )
    base = load_config(sampling_path)
    exports, arms = [], list(base.arms)
    mapped = dict(assets["checkpoints"])
    for view in interval.weights:
        destination = output / "exports" / view
        if destination.exists():
            record = json.loads((destination / "export.json").read_text())
            require(
                record["source_checkpoint"] == file_hash(checkpoint / "manifest.json")
                and record["weights"] == view
                and file_hash(destination / "weights.pt") == record["merged_sha256"]
                and file_hash(destination / "transformer_config.json")
                == record["architecture_sha256"],
                "saved interval export differs",
            )
        else:
            record = export(checkpoint, view, destination)
        asset = f"student_{view}"
        mapped[asset] = str(destination / "weights.pt")
        c = training.collection
        arms.append(
            Arm(
                asset,
                Checkpoint(
                    asset, view, record["merged_sha256"], record["architecture_sha256"]
                ),
                c.rule,
                c.steps,
                c.guidance,
                c.shift,
            )
        )
        exports.append(record)
    config = replace(base, arms=tuple(arms))
    for suite in interval.suites:
        rows = load_prompts(
            REPO / getattr(config.evaluation, SUITES[suite]), config.sampling.shapes
        )
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
        panel(target, output / f"{suite}-blind", interval.blind_seed)
    receipt = {
        "checkpoint_sha256": file_hash(checkpoint / "manifest.json"),
        "step": manifest["step"],
        "exports": exports,
        "suites": interval.suites,
        "scope": "Completed generation/scoring paths; human ballots remain separate evidence.",
    }
    atomic_json(output / "evaluation.json", receipt)
    return receipt


def execute(config_path, interval_path, assets_path, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    training = load_training_config(config_path)
    interval = load_interval_config(interval_path)
    assets = json.loads(Path(assets_path).read_text())
    stops = list(boundaries(training, interval))
    sampling = REPO / interval.sampling_config
    require(
        file_hash(sampling) == interval.sampling_sha256, "sampling config bytes differ"
    )
    identity = {
        "training_sha256": file_hash(config_path),
        "interval": asdict(interval),
        "assets_sha256": file_hash(assets_path),
        "execution": "Train segment exits, checkpoint export/evaluation, exact-contract resume.",
    }
    identity = json.loads(json.dumps(identity))
    with (output / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "campaign.json"
        require(
            not marker.exists() or json.loads(marker.read_text()) == identity,
            "campaign driver identity changed",
        )
        atomic_json(marker, identity)
        training_root = output / "training"
        checkpoints = committed_checkpoints(training_root / "checkpoints")
        if checkpoints:
            from src.posttrain.provenance import training_contract

            prompts = load_prompts(
                REPO / training.collection.prompts,
                {"1:1": (training.collection.height, training.collection.width)},
            )
            expected = training_contract(training, assets, prompts)
            require(
                all(
                    inspect(path)["contract"] == expected for _, _, path in checkpoints
                ),
                "campaign recovery contract differs",
            )
        # Reconcile evaluations from already committed checkpoints before
        # resuming training. An interrupted evaluation never becomes a skip.
        for step, _, path in checkpoints:
            if evaluation_due(training, interval, step):
                evaluate_checkpoint(
                    path, training, interval, assets, output / "evaluation" / path.name
                )
        last = checkpoints[-1][2] if checkpoints else None
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
            with (output / f"segment-{stop:06d}.log").open("a") as log:
                subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
            last = training_root / "checkpoints" / f"step-{stop:06d}"
            require(
                inspect(last)["step"] == stop,
                "training segment lacks its committed endpoint",
            )
            if evaluation_due(training, interval, stop):
                evaluate_checkpoint(
                    last, training, interval, assets, output / "evaluation" / last.name
                )
        atomic_json(
            output / "complete.json",
            {
                "step": training.training.steps,
                "checkpoint_sha256": file_hash(last / "manifest.json"),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interval", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        c, interval = load_training_config(args.config), load_interval_config(
            args.interval
        )
        stops = list(boundaries(c, interval))
        require(
            file_hash(REPO / interval.sampling_config) == interval.sampling_sha256,
            "sampling config bytes differ",
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "committed_stops": stops,
                    "evaluations": [s for s in stops if evaluation_due(c, interval, s)],
                }
            )
        )
    else:
        if args.assets is None or args.output is None:
            parser.error("--assets and --output are required")
        execute(args.config, args.interval, args.assets, args.output)
