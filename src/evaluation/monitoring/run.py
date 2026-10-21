"""Evaluate explicit checkpoint arms through independent scorer services."""

import argparse
import fcntl
import json
import math
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from src.evaluation.generation.workers import ResidentWorkers, run_pipeline
from src.evaluation.monitoring.config import load_config
from src.posttrain.records import atomic_json, identity
from src.posttrain.schema import require
from src.posttrain.scoring.client import ScoreClient, Submissions, group_metadata
from src.posttrain.scoring.contracts import image_payload
from src.utils import prompt_contract

from .diversity import validate_panel
from .prompts import load_evaluation_prompts
from .report import build_report

REPO = Path(__file__).resolve().parents[3]


def verify_assets(config, assets):
    checkpoints = {}
    for arm in config.arms:
        path = Path(assets["checkpoints"][arm.checkpoint.asset])
        require(path.suffix in {".pt", ".safetensors"}, "unsupported checkpoint format")
        checkpoints[arm.name] = str(path.resolve())
    return {
        "checkpoints": checkpoints,
        "text_encoder": str(Path(assets["text_encoder"]).resolve()),
        "vae": str(Path(assets["vae"]).resolve()),
    }


def run(
    config,
    assets,
    prompts,
    output,
    *,
    visual_only=False,
    suite="development",
    reference_cache=None,
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _run(
                config,
                assets,
                prompts,
                output,
                visual_only=visual_only,
                suite=suite,
                reference_cache=reference_cache,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run(config, assets, prompts, output, *, visual_only, suite, reference_cache):
    validate_panel(config, prompts, suite)
    with_diversity = suite in config.evaluation.diversity.suites
    artifacts = verify_assets(config, assets)
    if with_diversity:
        from .diversity import metric_identity

        artifacts["diversity"] = metric_identity(config.evaluation.diversity, assets)
    conditioning = identity(
        {
            "encoder": artifacts["text_encoder"],
            "exit_layer": config.sampling.text_exit_layer,
            "template": prompt_contract.PROMPT_TEMPLATE,
            "system": prompt_contract.SYSTEM_PROMPT,
            "drop_idx": prompt_contract.DROP_IDX,
            "max_sequence_length": prompt_contract.MAX_SEQUENCE_LENGTH,
        }
    )
    policies = {
        a.name: identity(
            {
                "checkpoint": asdict(a.checkpoint),
                "conditioning": conditioning,
                "vae": artifacts["vae"],
                "precision": config.sampling.precision,
            }
        )
        for a in config.arms
    }
    channels = () if visual_only else config.evaluation.channels
    paired = any(channel.startswith("rubric.") for channel in channels)
    require(
        not paired or len(config.arms) == 2,
        "rubric evaluation requires entry and student arms",
    )
    client = ScoreClient(config.scoring, assets["scoring"]) if channels else None
    recipe = asdict(config)
    recipe.pop("execution")
    manifest = {
        "config": recipe,
        "prompts": prompts,
        "artifacts": artifacts,
        "policies": policies,
        "conditioning_identity": conditioning,
        "visual_only": visual_only,
        "suite": suite,
        "scoring": client.info()["semantics"] if client else None,
    }
    # Normalize tuples to their JSON wire representation before restart comparison.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    path = output / "manifest.json"
    require(
        not path.exists() or json.loads(path.read_text()) == manifest,
        "run identity changed; use a new output directory",
    )
    atomic_json(path, manifest)
    records, results, tasks, positions, groups = {}, {}, [], {}, {}
    for arm_index, arm in enumerate(config.arms):
        pending = {}
        for prompt in prompts:
            sample_id = identity(
                {
                    "policy": policies[arm.name],
                    "prompt": prompt,
                    "arm": asdict(arm),
                    "sampling": asdict(config.sampling),
                }
            )
            group = prompt["id"] if paired else sample_id
            positions[sample_id] = (group, arm_index if paired else 0)
            groups[group] = group_metadata(group, prompt, 2 if paired else 1)
            path = output / "records" / f"{sample_id}.json"
            cached = (
                (Path(reference_cache) / f"{sample_id}.json")
                if reference_cache and arm_index == 0
                else None
            )
            source = path if path.exists() else cached
            if source is not None and source.exists():
                record = json.loads(source.read_text())
                require(
                    record["sample_id"] == sample_id
                    and record["prompt"] == prompt
                    and record["arm"] == arm.name,
                    "saved sample identity differs",
                )
                records[sample_id] = record
                if not path.exists():
                    atomic_json(path, record)
                score_path = output / "scores" / f"{sample_id}.json"
                if channels and score_path.exists():
                    row = json.loads(score_path.read_text())
                    require(
                        row["sample_id"] == sample_id
                        and set(row["values"]) == set(channels)
                        and all(math.isfinite(v) for v in row["values"].values()),
                        "saved score identity or channels differ",
                    )
                    results[sample_id] = row
            else:
                pending.setdefault(prompt["aspect"], []).append((prompt, sample_id))
        for pairs in pending.values():
            for start in range(0, len(pairs), config.sampling.batch_size):
                chunk = pairs[start : start + config.sampling.batch_size]
                tasks.append(
                    {
                        "task_id": identity([key for _, key in chunk]),
                        "prompts": [p for p, _ in chunk],
                        "sample_ids": [key for _, key in chunk],
                        "arm": arm,
                        "records": [
                            {
                                "sample_id": key,
                                "prompt": p,
                                "arm": arm.name,
                                "checkpoint": asdict(arm.checkpoint),
                                "image_path": str(output / arm.name / f"{p['id']}.png"),
                                "seed": p["seed"],
                                "policy_version": policies[arm.name],
                                "conditioning_identity": conditioning,
                            }
                            for p, key in chunk
                        ],
                        "image_paths": [
                            str(output / arm.name / f"{p['id']}.png") for p, _ in chunk
                        ],
                        "policy_version": policies[arm.name],
                        "conditioning_identity": conditioning,
                    }
                )

    missing_groups = {
        group for key, (group, _) in positions.items() if key not in results
    }
    # A partially saved pair must be judged together again; do not mix two calls.
    needed = (
        {key for key, (group, _) in positions.items() if group in missing_groups}
        if client
        else set()
    )
    for key in needed:
        results.pop(key, None)
    active = output / "active_collection.json"
    if client and active.exists():
        client.release(json.loads(active.read_text())["id"])
        active.unlink()
    collection = uuid.uuid4().hex
    offset, last_progress, changed_at = 0, None, time.monotonic()
    submissions = None

    def read_scores():
        nonlocal offset, last_progress, changed_at
        packet = client.read(collection, offset)
        if packet["errors"]:
            raise RuntimeError(f"evaluation scoring failed: {packet['errors']}")
        for row in packet["results"]:
            key = row["sample_id"]
            require(
                key in needed
                and key not in results
                and set(row["values"]) == set(channels)
                and all(math.isfinite(v) for v in row["values"].values()),
                "unknown, duplicate or invalid evaluation score",
            )
            results[key] = row
            atomic_json(output / "scores" / f"{key}.json", row)
        offset = packet["next"]
        progress = packet["progress"]
        counters = (
            progress["accepted"],
            progress["completed"],
            *(v["calls"] for _, v in sorted(progress["backends"].items())),
        )
        if counters != last_progress:
            last_progress, changed_at = counters, time.monotonic()
        require(
            time.monotonic() - changed_at < config.execution.task_timeout,
            "evaluation scoring stopped progressing",
        )
        return progress

    def generated(batch):
        for record in batch:
            key = record["sample_id"]
            records[key] = record
            atomic_json(output / "records" / f"{key}.json", record)
            if reference_cache and record["arm"] == config.arms[0].name:
                atomic_json(Path(reference_cache) / f"{key}.json", record)
            if key in needed:
                group, draw = positions[key]
                submissions.submit(
                    key,
                    group,
                    draw,
                    image_payload(Path(record["image_path"]).read_bytes()),
                )
        if needed:
            read_scores()

    started = time.monotonic()
    try:
        if needed:
            atomic_json(active, {"id": collection})
            client.open(
                collection,
                channels,
                config.evaluation.bootstrap_seed,
                [groups[group] for group in sorted(missing_groups)],
            )
            submissions = Submissions(client, collection)
            generated(list(records.values()))
        telemetry, startup = [], []
        if tasks:
            with ResidentWorkers(
                config.sampling,
                config.execution,
                {
                    **assets,
                    "python_paths": assets.get("learner", {}).get("python_paths", []),
                },
            ) as workers:
                startup = workers.startup
                telemetry = run_pipeline(tasks, workers, config.execution, generated)
        if submissions:
            submissions.close()
            submissions = None
            progress = read_scores()
            while not needed <= results.keys():
                time.sleep(0.2)
                progress = read_scores()
            atomic_json(output / "scoring_progress.json", progress)
        stamp = time.time_ns()
        atomic_json(
            output / "execution" / f"{stamp}.json",
            {
                "execution": asdict(config.execution),
                "startup": startup,
                "workers": telemetry,
                "wall_seconds": time.monotonic() - started,
            },
        )
    finally:
        if submissions:
            submissions.close()
        if needed:
            client.release(collection)
            active.unlink(missing_ok=True)
    require(
        len(records) == len(prompts) * len(config.arms),
        "benchmark has missing generated samples",
    )
    report = (
        {"scope": "Visual panel only; no automatic language/length scoring claim."}
        if visual_only
        else build_report(
            list(records.values()), list(results.values()), config, suite=suite
        )
    )
    report["suite"] = suite
    if with_diversity:
        from .diversity import measure_diversity

        report["diversity"] = measure_diversity(
            list(records.values()), config.evaluation.diversity, assets
        )
    report["generated_samples"] = len(records)
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suite", default="development")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log this generation experiment online to inko",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    prompts = load_evaluation_prompts(config, args.suite)
    validate_panel(config, prompts, args.suite)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "prompts": len(prompts),
                    "arms": [a.name for a in config.arms],
                    "samples": len(prompts) * len(config.arms),
                    "execution": asdict(config.execution),
                },
                indent=2,
            )
        )
        return
    if args.assets is None or args.output is None:
        parser.error("--assets and --output are required")
    from src.posttrain.tracking import Tracker

    context = (
        Tracker(
            args.output, args.config.stem, config=asdict(config), phase="evaluation"
        )
        if args.track or (args.output / "tracking.json").exists()
        else nullcontext()
    )
    with context as tracker:
        report = run(
            config,
            json.loads(args.assets.read_text()),
            prompts,
            args.output,
            visual_only=args.suite == "monitor",
            suite=args.suite,
        )
        if tracker is not None:
            from scripts.posttrain.readouts import panel

            blind = args.output / "blind"
            panel(args.output, blind, config.evaluation.bootstrap_seed)
            tracker.evaluation(report, args.output, config, 0)


if __name__ == "__main__":
    main()
