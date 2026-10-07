"""Run explicit sampling/cold qualification cases or submit one guarded job."""

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from src.posttrain.config import load_config
from src.posttrain.records import atomic_json
from src.posttrain.schema import require
from src.posttrain.training_config import load_training_config

from .budget import locked_ledger, reserve


def validate_matrix(matrix):
    require(
        set(matrix) == {"job", "cases"} and bool(matrix["cases"]),
        "qualification requires explicit job and cases",
    )
    for case in matrix["cases"]:
        require(
            case.get("kind") in {"sampling", "cold"},
            "explicit qualification case kind required",
        )
        if case["kind"] == "cold":
            require(
                set(case)
                == {"kind", "config", "assets", "output", "resume", "stop_after"},
                "invalid cold training case fields",
            )
            config = load_training_config(case["config"])
            require(
                type(case["stop_after"]) is int
                and 0 < case["stop_after"] <= config.training.steps,
                "invalid committed stop boundary",
            )
            require(
                case["resume"] is None or isinstance(case["resume"], str),
                "invalid resume path",
            )
            continue
        require(
            set(case) == {"kind", "config", "assets", "output", "suite"},
            "invalid sampling case fields",
        )
        require(
            case["suite"] in {"bringup", "monitor", "development"},
            "selection/training is not a sampling qualification case",
        )
        load_config(case["config"])
    job = matrix["job"]
    require(
        set(job)
        == {
            "name",
            "workspace",
            "project",
            "group",
            "quota",
            "image",
            "command",
            "shm_size",
        },
        "invalid job fields",
    )
    require(job["quota"].split(",")[0] == "16", "one 16-NPU allocation required")
    require(
        job["name"].startswith("inko-posttrain-"),
        "qualification requires a dedicated posttrain job name",
    )


def execute_cases(matrix):
    from src.evaluation.benchmarks.prompts import load_prompts
    from src.evaluation.benchmarks.run import REPO, run

    for case in matrix["cases"]:
        if case["kind"] == "cold":
            argv = [
                sys.executable,
                "-m",
                "src.posttrain.train",
                "--config",
                case["config"],
                "--assets",
                case["assets"],
                "--output",
                case["output"],
                "--stop-after",
                str(case["stop_after"]),
            ]
            if case["resume"] is not None:
                argv += ["--resume", case["resume"]]
            subprocess.run(argv, check=True)
            continue
        config = load_config(case["config"])
        key = {
            "bringup": "bringup_suite",
            "monitor": "monitor_suite",
            "development": "suite",
        }[case["suite"]]
        prompts = load_prompts(
            REPO / getattr(config.evaluation, key), config.sampling.shapes
        )
        report = run(
            config,
            json.loads(Path(case["assets"]).read_text()),
            prompts,
            Path(case["output"]),
            visual_only=case["suite"] == "monitor",
        )
        if case["suite"] != "monitor":
            require(
                all(
                    metric["complete"]
                    for arm in report["arms"].values()
                    for metric in arm.values()
                ),
                "qualification has incomplete scores",
            )
        print(
            json.dumps(
                {
                    "event": "case_complete",
                    "suite": case["suite"],
                    "output": case["output"],
                }
            ),
            flush=True,
        )


def launch(matrix, ledger):
    job = {**matrix["job"], "name": matrix["job"]["name"] + "-" + uuid.uuid4().hex[:12]}
    with locked_ledger(ledger) as state:
        reserve(state, job["name"], job["workspace"], time.time())
    log_path = ledger.with_name(f"{ledger.stem}.{job['name']}.watch.log")
    with log_path.open("a") as log:
        watchdog = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.posttrain.budget",
                "--ledger",
                str(ledger),
                "--job",
                job["name"],
                "--watch",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 10
    while True:
        with locked_ledger(ledger) as state:
            ready = state["jobs"][-1].get("watchdog_ready_at")
        require(
            watchdog.poll() is None and time.monotonic() < deadline,
            "external watchdog did not start; reservation retained",
        )
        if ready:
            break
        time.sleep(0.1)
    argv = ["inspire", "--json", "job", "create"]
    for key in (
        "name",
        "workspace",
        "project",
        "group",
        "quota",
        "image",
        "command",
        "shm_size",
    ):
        argv.extend(["--" + key.replace("_", "-"), str(job[key])])
    argv += ["--nodes", "1", "--no-auto-fault-tolerance", "--no-enable-notification"]
    # No per-job time cap. The external watchdog enforces the cumulative allowance.
    # Persist intent before the create RPC: timeout or launcher death must not
    # strand a remotely accepted allocation outside watchdog observation.
    with locked_ledger(ledger) as state:
        state["jobs"][-1]["submitted_at"] = time.time()
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=120, check=False
    )
    atomic_json(
        ledger.with_name(f"{ledger.stem}.{job['name']}.submission.json"),
        {
            "argv": argv,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    with locked_ledger(ledger) as state:
        if result.returncode:
            state["blocked"] = (
                "submission uncertain; watchdog must reconcile named job before further launches"
            )
    require(
        result.returncode == 0,
        "provider submission failed; inspect ledger and watchdog",
    )
    return {"job": job["name"], "watchdog_pid": watchdog.pid, "ledger": str(ledger)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument(
        "--execute-cases",
        action="store_true",
        help="inside an already guarded allocation",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    validate_matrix(matrix)
    if args.validate_only:
        print(json.dumps(matrix, indent=2))
    elif args.execute_cases:
        execute_cases(matrix)
    else:
        if args.budget_ledger is None:
            parser.error("submission requires --budget-ledger outside the allocation")
        print(json.dumps(launch(matrix, args.budget_ledger.resolve()), indent=2))


if __name__ == "__main__":
    main()
