"""Evaluate explicit checkpoint arms through independent scorer services."""

import argparse
import fcntl
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from src.evaluation.benchmarks.config import load_config
from src.evaluation.benchmarks.workers import ResidentWorkers, run_pipeline
from src.posttrain.records import atomic_json, file_hash, identity
from src.posttrain.schema import decode, require
from src.posttrain.scoring.client import request_for_record
from src.posttrain.scoring.contracts import ScoreResult, validate_results
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


def parse_score(row):
    return decode(
        ScoreResult, {k: v for k, v in row.items() if k not in {"scorer", "attempts"}}
    )


def run(config, assets, prompts, output, *, visual_only=False, suite="development"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _run(
                config, assets, prompts, output, visual_only=visual_only, suite=suite
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run(config, assets, prompts, output, *, visual_only, suite):
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
    specs = {} if visual_only else {s.id: s for s in config.scorers}
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
    }
    # Normalize tuples to their JSON wire representation before restart comparison.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    path = output / "manifest.json"
    require(
        not path.exists() or json.loads(path.read_text()) == manifest,
        "run identity changed; use a new output directory",
    )
    atomic_json(path, manifest)
    records, results, tasks = {}, {}, []
    for arm in config.arms:
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
            path = output / "records" / f"{sample_id}.json"
            if path.exists():
                record = json.loads(path.read_text())
                require(
                    record["sample_id"] == sample_id
                    and record["prompt"] == prompt
                    and record["arm"] == arm.name,
                    "saved sample identity differs",
                )
                for kind in ("image", "latent"):
                    require(
                        file_hash(record[kind + "_path"]) == record[kind + "_sha256"],
                        f"saved {kind} bytes changed",
                    )
                records[sample_id] = record
                missing = []
                for role, spec in specs.items():
                    score_path = output / "scores" / role / f"{sample_id}.json"
                    if score_path.exists():
                        row = json.loads(score_path.read_text())
                        require(row["scorer"] == role, "saved service ID differs")
                        score = parse_score(row)
                        validate_results(
                            [request_for_record(record, spec)],
                            [score],
                            spec.outputs,
                            complete=False,
                        )
                        results[(sample_id, role)] = row
                        if score.status != "ok":
                            missing.append(role)
                    else:
                        missing.append(role)
                if missing:
                    tasks.append(
                        {
                            "task_id": f"score-{sample_id}",
                            "records": [record],
                            "missing_scorers": missing,
                        }
                    )
            else:
                pending.setdefault(prompt["aspect"], []).append((prompt, sample_id))
        for pairs in pending.values():
            for start in range(0, len(pairs), config.sampling.batch_size):
                chunk = pairs[start : start + config.sampling.batch_size]
                tasks.append(
                    {
                        "task_id": identity([s for _, s in chunk]),
                        "prompts": [p for p, _ in chunk],
                        "sample_ids": [s for _, s in chunk],
                        "arm": arm,
                        "output": str(output),
                        "policy_version": policies[arm.name],
                        "conditioning_identity": conditioning,
                        "missing_scorers": list(specs),
                    }
                )

    def generated(batch):
        for record in batch:
            records[record["sample_id"]] = record
            atomic_json(output / "records" / f"{record['sample_id']}.json", record)

    def scored(role, batch):
        require(
            bool(batch) and all(r["scorer"] == role for r in batch),
            "empty or wrong service response",
        )
        validate_results(
            [request_for_record(records[r["sample_id"]], specs[role]) for r in batch],
            [parse_score(r) for r in batch],
            specs[role].outputs,
            complete=False,
        )
        for row in batch:
            results[(row["sample_id"], role)] = row
            atomic_json(output / "scores" / role / f"{row['sample_id']}.json", row)

    if tasks:
        started = time.perf_counter()
        with ResidentWorkers(
            config,
            assets,
            output,
            generate=any("records" not in t for t in tasks),
            score=not visual_only,
        ) as workers:
            stamp = time.time_ns()
            atomic_json(
                output / "execution" / f"{stamp}-startup.json",
                {"execution": asdict(config.execution), "workers": workers.startup},
            )
            service_start = time.perf_counter()
            telemetry = run_pipeline(tasks, workers, config, generated, scored)
            atomic_json(
                output / "execution" / f"{stamp}-timings.json",
                {
                    "workers": telemetry,
                    "service_wall_seconds": time.perf_counter() - service_start,
                    "including_load_seconds": time.perf_counter() - started,
                },
            )
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
