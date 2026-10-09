"""Execute explicit full-model campaign path checks; never select a training recipe."""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from src.posttrain.checkpoint import inspect
from src.posttrain.evaluation_config import boundaries, load_interval_config
from src.posttrain.provenance import REPO
from src.posttrain.records import atomic_json, file_hash
from src.posttrain.schema import decode, digest, name, require
from src.posttrain.training_config import load_training_config

from .run_campaign import committed_checkpoints


@dataclass(frozen=True)
class Case:
    name: str
    purpose: str
    config: str
    config_sha256: str
    interval: str
    interval_sha256: str
    fit_stop: int
    initial_from: str
    initial_step: int


@dataclass(frozen=True)
class Plan:
    cases: tuple[Case, ...]


def validate(path):
    plan = decode(Plan, json.loads(Path(path).read_text()))
    previous = set()
    for case in plan.cases:
        name(case.name)
        require(
            case.name not in previous and bool(case.purpose),
            "duplicate or purposeless qualification case",
        )
        for key in ("config", "interval"):
            digest(getattr(case, key + "_sha256"))
            require(
                file_hash(REPO / getattr(case, key)) == getattr(case, key + "_sha256"),
                "qualification input bytes differ",
            )
        c = load_training_config(REPO / case.config)
        require(
            0 <= case.fit_stop <= c.fake.initial_steps, "invalid fitting recovery point"
        )
        require(
            not case.initial_from or case.initial_from in previous,
            "cold boundary qualification source must run first",
        )
        require(
            (case.initial_step > 0) == bool(case.initial_from),
            "invalid cold source step",
        )
        list(boundaries(c, load_interval_config(REPO / case.interval)))
        previous.add(case.name)
    require(bool(plan.cases), "empty qualification plan")
    return plan


def execute(plan_path, assets_path, output):
    plan = validate(plan_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "plan.json"
    expected = json.loads(json.dumps(asdict(plan)))
    require(
        not marker.exists() or json.loads(marker.read_text()) == expected,
        "qualification plan changed",
    )
    atomic_json(output / "plan.json", asdict(plan))
    for case in plan.cases:
        print(
            json.dumps(
                {"case": case.name, "status": "starting", "purpose": case.purpose}
            ),
            flush=True,
        )
        target = output / case.name
        target.mkdir(parents=True, exist_ok=True)
        c = load_training_config(REPO / case.config)
        assets = json.loads(Path(assets_path).read_text())
        if case.initial_from:
            source = (
                output
                / case.initial_from
                / "training"
                / "checkpoints"
                / f"step-{case.initial_step:06d}"
            )
            manifest = inspect(source)
            require(
                manifest["contract"]["config"]["stages"]["cold_steps"]
                >= case.initial_step,
                "qualification initialization is not cold",
            )
            c = replace(
                c,
                stages=replace(
                    c.stages, initial_sha256=file_hash(source / "manifest.json")
                ),
            )
            assets.setdefault("training_checkpoints", {})[
                c.stages.initial_checkpoint
            ] = str(source)
        frozen = target / "config.json"
        resolved = json.loads(json.dumps(asdict(c)))
        require(
            not frozen.exists() or json.loads(frozen.read_text()) == resolved,
            "resolved qualification config differs",
        )
        atomic_json(frozen, resolved)
        asset_path = target / "assets.json"
        require(
            not asset_path.exists() or json.loads(asset_path.read_text()) == assets,
            "qualification assets differ",
        )
        atomic_json(asset_path, assets)
        training_root = target / "training"
        fit_checkpoint = (
            training_root / "checkpoints" / f"fit-{case.fit_stop:06d}" / "manifest.json"
        )
        if case.fit_stop and not fit_checkpoint.exists():
            checkpoints = committed_checkpoints(training_root / "checkpoints")
            require(
                not checkpoints
                or (checkpoints[-1][0] == 0 and checkpoints[-1][1] < case.fit_stop),
                "qualification passed its fitting stop without a recovery checkpoint",
            )
            argv = [
                sys.executable,
                "-m",
                "src.posttrain.train",
                "--config",
                str(frozen),
                "--assets",
                str(asset_path),
                "--output",
                str(training_root),
                "--stop-after-fit",
                str(case.fit_stop),
            ]
            if checkpoints:
                argv += ["--resume", str(checkpoints[-1][2])]
            with (target / "initial-fit-stop.log").open("a") as log:
                subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        argv = [
            sys.executable,
            "-m",
            "scripts.posttrain.run_campaign",
            "--config",
            str(frozen),
            "--interval",
            str(REPO / case.interval),
            "--assets",
            str(asset_path),
            "--output",
            str(target),
        ]
        with (target / "campaign.log").open("a") as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        final = training_root / "checkpoints" / f"step-{c.training.steps:06d}"
        manifest = inspect(final)
        completions = [
            json.loads(p.read_text())
            for p in (training_root / "completion").glob("*.json")
        ]
        final_ranks = [r for r in completions if r["step"] == c.training.steps]
        require(
            {r["rank"] for r in final_ranks}
            == set(range(len(c.execution.learner_devices)))
            and all(r["frozen_unchanged"] for r in final_ranks),
            "full learner completion evidence missing",
        )
        evaluations = [
            json.loads(p.read_text())
            for p in sorted((target / "evaluation").glob("*/evaluation.json"))
        ]
        require(bool(evaluations), "qualification has no real export/evaluation")
        result = {
            "case": case.name,
            "purpose": case.purpose,
            "config_sha256": file_hash(frozen),
            "checkpoint_sha256": file_hash(final / "manifest.json"),
            "contract_sha256": manifest["contract_sha256"],
            "evaluations": evaluations,
            "scope": "Full-model execution qualification only; stability and recipe quality remain unselected.",
        }
        atomic_json(target / "qualification.json", result)
        print(json.dumps({"case": case.name, "status": "completed"}), flush=True)
    atomic_json(
        output / "qualification.json",
        {
            "runnable": True,
            "cases": [c.name for c in plan.cases],
            "scope": "These exact configs and resource/runtime identities executed; no guarantee for arbitrary scales or horizons.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        plan = validate(args.plan)
        print(json.dumps({"valid": True, "cases": [c.name for c in plan.cases]}))
    else:
        if args.assets is None or args.output is None:
            parser.error("--assets and --output are required")
        execute(args.plan, args.assets, args.output)
