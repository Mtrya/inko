"""Run explicit sampling/training qualification cases or submit one guarded job."""

import argparse
import json
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from src.posttrain.records import atomic_json, identity
from src.posttrain.schema import digest, require
from src.posttrain.training_config import load_training_config

from .budget import (
    ALLOWANCES,
    CAP_SECONDS,
    SHUTDOWN_MARGIN,
    consumed,
    locked_ledger,
    reserve,
)
from .coverage import Evidence, coverage_report, validate_matrix, validate_reuse


def validate_deployment(deployment, matrix):
    require(
        set(deployment) == {"job", "assets", "output_root", "archives", "artifacts"},
        "invalid qualification deployment fields",
    )
    job = deployment["job"]
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
        job["name"].startswith("inko-posttrain-"), "dedicated posttrain job required"
    )
    require(
        bool(deployment["assets"]) and bool(deployment["output_root"]),
        "explicit local asset/output paths required",
    )
    for case in matrix.cases:
        for key in (case.resume, case.compare):
            if key:
                require(
                    key in deployment["artifacts"],
                    "missing declared checkpoint artifact",
                )
                item = deployment["artifacts"][key]
                require(
                    set(item) == {"path", "sha256"} and item["path"],
                    "invalid artifact declaration",
                )
                digest(item["sha256"])
    evidence = Evidence(deployment["archives"])
    receipts = []
    try:
        for case in matrix.cases:
            if case.execution == "reuse":
                receipts.append(
                    validate_reuse(case, load_training_config(case.config), evidence)
                )
    finally:
        evidence.close()
    return receipts


def budget_admission(path, *, now=None):
    # Validate a copy; dry-run must not reserve a job or write an accounting ledger.
    require(path.exists(), "existing shared implementation ledger required")
    state = json.loads(path.read_text())
    require(
        state["schema"] == 1 and state["cap_seconds"] == CAP_SECONDS,
        "budget ledger contract differs",
    )
    reserve(
        state,
        "qualification-validation",
        "validation",
        time.time() if now is None else now,
    )


def launch(matrix, ledger, *, allowance):
    require(allowance in ALLOWANCES, "unapproved allocation allowance")
    cap_seconds = ALLOWANCES[allowance]
    job = {**matrix["job"], "name": matrix["job"]["name"] + "-" + uuid.uuid4().hex[:12]}
    with locked_ledger(ledger, cap_seconds=cap_seconds) as state:
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
                "--allowance",
                allowance,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 10
    while True:
        with locked_ledger(ledger, cap_seconds=cap_seconds) as state:
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
    with locked_ledger(ledger, cap_seconds=cap_seconds) as state:
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
    with locked_ledger(ledger, cap_seconds=cap_seconds) as state:
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
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument(
        "--allocation-job", help="actual guarded job name for execution"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute-cases",
        action="store_true",
        help="inside an already guarded allocation",
    )
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument(
        "--promote", action="store_true", help="require every execution and review gate"
    )
    args = parser.parse_args()
    if (args.promote or args.reviews is not None) and not args.report:
        parser.error("--promote and --reviews require --report")
    if args.execute_cases and not args.allocation_job:
        parser.error("--execute-cases requires --allocation-job")
    if args.execute_cases and args.budget_ledger is None:
        parser.error("--execute-cases requires --budget-ledger")
    if args.allocation_job and not args.execute_cases:
        parser.error("--allocation-job requires --execute-cases")
    matrix = validate_matrix(json.loads(args.matrix.read_text()))
    deployment = json.loads(args.deployment.read_text())
    reused = validate_deployment(deployment, matrix)
    if args.validate_only:
        if args.budget_ledger:
            budget_admission(args.budget_ledger)
        print(
            json.dumps(
                {
                    "valid": True,
                    "matrix_sha256": identity(asdict(matrix)),
                    "reused_cases": [r["case"] for r in reused],
                    "cases_to_execute": [
                        c.id for c in matrix.cases if c.execution != "reuse"
                    ],
                },
                indent=2,
            )
        )
    elif args.report:
        output = Path(deployment["output_root"])
        require(
            (output / "results.json").exists()
            and (output / "evaluation-summary.json").exists(),
            "missing execution/evaluation evidence",
        )
        results = json.loads((output / "results.json").read_text())
        from .qualification import validate_evaluation

        evidence = Evidence(deployment["archives"])
        try:
            evaluation = validate_evaluation(matrix, results, output, evidence=evidence)
        finally:
            evidence.close()
        reviews = [] if args.reviews is None else json.loads(args.reviews.read_text())
        report = coverage_report(matrix, results, reviews=reviews)
        report["evaluation"] = {
            k: v for k, v in evaluation.items() if k != "evaluation"
        }
        require(
            args.budget_ledger is not None,
            "coverage reporting requires the shared budget ledger",
        )
        state = json.loads(args.budget_ledger.read_text())
        from .receipts import allocation

        owned_job = allocation(
            state,
            json.loads((output / "allocation.json").read_text()),
            json.loads((output / "execution.json").read_text()),
            results,
            identity(asdict(matrix)),
            start=json.loads((output / "allocation-start.json").read_text()),
            cap_seconds=CAP_SECONDS,
            max_age=SHUTDOWN_MARGIN,
        )
        report["budget"] = {
            "used_node_hours": consumed(state, time.time()) / 3600,
            "cap_node_hours": state["cap_seconds"] / 3600,
            "all_jobs_released": all(
                j["released_at"] is not None for j in state["jobs"]
            ),
            "accounting_block": state["blocked"],
            "qualification_job": owned_job,
        }
        report["qualified"] = (
            report["qualified"]
            and not state["blocked"]
            and report["budget"]["all_jobs_released"]
            and report["budget"]["used_node_hours"]
            <= report["budget"]["cap_node_hours"]
        )
        atomic_json(output / "coverage-report.json", report)
        require(
            not args.promote or report["qualified"],
            "unrun or unresolved qualification gate",
        )
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    elif args.execute_cases:
        from .qualification import execute

        execute(
            matrix,
            deployment,
            evaluation_config=matrix.evaluation,
            allocation_job=args.allocation_job,
            budget_ledger=args.budget_ledger,
        )
    else:
        if args.budget_ledger is None:
            parser.error("submission requires --budget-ledger")
        budget_admission(args.budget_ledger)
        print(
            json.dumps(
                launch(
                    deployment, args.budget_ledger.resolve(), allowance="implementation"
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
