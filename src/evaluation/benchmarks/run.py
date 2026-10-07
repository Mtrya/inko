"""Run the initial three-arm benchmark with resident generation/scorer workers."""

import argparse
import fcntl
import importlib.metadata
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from src.posttrain.config import load_config
from src.posttrain.records import (
    ScoreRequest,
    ScoreResult,
    atomic_json,
    file_hash,
    identity,
    join_scores,
)
from src.posttrain.workers import ResidentWorkers, run_pipeline

from .prompts import load_prompts
from .report import build_report

REPO = Path(__file__).resolve().parents[3]


def tree_identity(path):
    path = Path(path)
    if path.is_file():
        return {path.name: file_hash(path)}
    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and not any(
            x.startswith(".") or x == "__pycache__" for x in p.relative_to(path).parts
        )
        and p.suffix != ".pyc"
    )
    if not files:
        raise ValueError(f"empty or missing artifact {path}")
    return {str(p.relative_to(path)): file_hash(p) for p in files}


def verify_assets(config, assets, visual_only=False):
    checkpoint = Path(assets["checkpoint"])
    if file_hash(checkpoint) != config["initialization"]["sha256"]:
        raise ValueError("checkpoint bytes differ from the selected initialization")
    expected_name = (
        "ema_weights.pt"
        if config["initialization"]["weights"] == "ema"
        else "model.safetensors"
    )
    if checkpoint.name != expected_name:
        raise ValueError("checkpoint filename disagrees with live/EMA selection")
    identities = {
        "checkpoint": file_hash(checkpoint),
        "architecture": tree_identity(checkpoint.with_name("transformer_config.json")),
        "text_encoder": tree_identity(assets["text_encoder"]),
        "vae": tree_identity(assets["vae"]),
        "runtime": {
            role: [
                tree_identity(p) for p in assets.get(role, {}).get("python_paths", [])
            ]
            for role in (
                ["learner"] if visual_only else ["learner", "hpsv3", "vqascore"]
            )
        },
        "base_packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    }
    if not visual_only:
        for role in ("hpsv3", "vqascore"):
            source = assets[role]["source"]
            revision = subprocess.check_output(
                ["git", "-C", source, "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != config["scorers"][role]["source_revision"]:
                raise ValueError(f"{role}: source revision mismatch")
            if subprocess.check_output(
                ["git", "-C", source, "status", "--porcelain", "--untracked-files=no"],
                text=True,
            ).strip():
                raise ValueError(f"{role}: source checkout has modifications")
            identities[role] = {
                "source_revision": revision,
                "weights": tree_identity(assets[role]["weights"]),
            }
            if role == "hpsv3":
                identities[role]["backbone"] = tree_identity(assets[role]["backbone"])
    return identities


def run(config, assets, prompts, output, *, visual_only=False):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _run(config, assets, prompts, output, visual_only=visual_only)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run(config, assets, prompts, output, *, visual_only=False):
    output = output.resolve()
    # Hash once in the coordinator, outside measured service time.
    artifacts = verify_assets(config, assets, visual_only)
    policy = identity(
        {k: artifacts[k] for k in ("checkpoint", "architecture", "text_encoder", "vae")}
    )
    scorer_ids = (
        {}
        if visual_only
        else {
            role: identity(
                {
                    "spec": config["scorers"][role],
                    "artifacts": artifacts[role],
                    "runtime": artifacts["runtime"][role],
                    "base_packages": artifacts["base_packages"],
                    "adapter_sha256": file_hash(REPO / "src/posttrain/scoring.py"),
                }
            )
            for role in config["scorers"]
        }
    )
    manifest = {
        "config": config,
        "prompts": prompts,
        "artifacts": artifacts,
        "policy_version": policy,
        "scorer_identities": scorer_ids,
        "visual_only": visual_only,
        "code": {
            str(p.relative_to(REPO)): file_hash(p)
            for p in sorted((REPO / "src").rglob("*.py"))
        },
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("run identity changed; use a new output directory")
    atomic_json(manifest_path, manifest)
    records, results = {}, {}
    tasks = []
    batch_size = config["sampling"]["batch_size"]

    for arm in config["arms"]:
        pending = {}
        for prompt in prompts:
            sample_id = identity(
                {
                    "policy": policy,
                    "prompt": prompt,
                    "arm": arm,
                    "sampling": config["sampling"],
                }
            )
            path = output / "records" / f"{sample_id}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if (
                    record["sample_id"] != sample_id
                    or record["prompt"] != prompt
                    or record["arm"] != arm["name"]
                ):
                    raise ValueError("saved sample identity differs")
                for kind in ("image", "latent"):
                    if file_hash(record[kind + "_path"]) != record[kind + "_sha256"]:
                        raise ValueError(f"saved {kind} bytes changed")
                records[sample_id] = record
                missing = []
                for role in scorer_ids:
                    score_path = output / "scores" / role / f"{sample_id}.json"
                    if score_path.exists():
                        score = ScoreResult(**json.loads(score_path.read_text()))
                        request = ScoreRequest(
                            sample_id,
                            record["image_path"],
                            record["image_sha256"],
                            prompt["text"],
                        )
                        join_scores(
                            [request], [score], scorer_ids, require_complete=False
                        )
                        results[(sample_id, role)] = asdict(score)
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
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start : start + batch_size]
                tasks.append(
                    {
                        "task_id": identity([s for _, s in chunk]),
                        "prompts": [p for p, _ in chunk],
                        "sample_ids": [s for _, s in chunk],
                        "arm": arm,
                        "output": str(output),
                        "policy_version": policy,
                        "missing_scorers": list(scorer_ids),
                    }
                )

    def generated(batch):
        for record in batch:
            records[record["sample_id"]] = record
            atomic_json(output / "records" / f"{record['sample_id']}.json", record)

    def scored(role, batch):
        if not batch:
            raise ValueError("empty scorer response")
        requests = [
            ScoreRequest(
                r["sample_id"],
                records[r["sample_id"]]["image_path"],
                records[r["sample_id"]]["image_sha256"],
                records[r["sample_id"]]["prompt"]["text"],
            )
            for r in batch
        ]
        parsed = [ScoreResult(**r) for r in batch]
        if any(r.scorer != role for r in parsed):
            raise ValueError("worker returned another scorer's result")
        join_scores(requests, parsed, scorer_ids, require_complete=False)
        for result in batch:
            results[(result["sample_id"], role)] = result
            atomic_json(
                output / "scores" / role / f"{result['sample_id']}.json", result
            )

    started = time.perf_counter()
    if tasks:
        with ResidentWorkers(
            config, assets, scorer_ids, roles=["learner"] if visual_only else None
        ) as workers:
            atomic_json(output / "worker_startup.json", workers.startup)
            service_start = time.perf_counter()
            telemetry = run_pipeline(tasks, workers, config, generated, scored)
            atomic_json(
                output / "timings.json",
                {
                    "workers": telemetry,
                    "service_wall_seconds": time.perf_counter() - service_start,
                    "including_load_seconds": time.perf_counter() - started,
                },
            )
    expected = len(prompts) * len(config["arms"])
    if len(records) != expected:
        raise ValueError("benchmark has missing generated samples")
    report = (
        {"scope": "Visual panel only; no automatic language/length scoring claim."}
        if visual_only
        else build_report(list(records.values()), list(results.values()), config)
    )
    report["generated_samples"] = len(records)
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/posttrain.toml")
    parser.add_argument(
        "--assets",
        type=Path,
        help="machine-local JSON artifact paths and role dependency overlays",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--suite", choices=["development", "bringup", "monitor"], default="bringup"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    suite_key = {
        "development": "suite",
        "bringup": "bringup_suite",
        "monitor": "monitor_suite",
    }[args.suite]
    prompts = load_prompts(
        REPO / config["evaluation"][suite_key], config["sampling"]["shapes"]
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "prompts": len(prompts),
                    "arms": [a["name"] for a in config["arms"]],
                    "samples": len(prompts) * len(config["arms"]),
                    "execution": config["execution"],
                },
                indent=2,
            )
        )
        return
    if args.assets is None or args.output is None:
        parser.error("--assets and --output are required for execution")
    run(
        config,
        json.loads(args.assets.read_text()),
        prompts,
        args.output,
        visual_only=args.suite == "monitor",
    )


if __name__ == "__main__":
    main()
