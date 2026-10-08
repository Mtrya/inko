"""Execute only declared qualification gaps, retaining real artifact receipts."""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from src.posttrain.records import atomic_json, file_hash, identity
from src.posttrain.schema import require
from src.posttrain.training_config import load_training_config

from .coverage import Evidence, coverage_report, source_files, validate_reuse


def artifact(deployment, key):
    require(key in deployment["artifacts"], "missing declared checkpoint artifact")
    spec = deployment["artifacts"][key]
    require(set(spec) == {"path", "sha256"}, "invalid checkpoint artifact fields")
    path = Path(spec["path"])
    require(
        file_hash(path / "manifest.json") == spec["sha256"],
        "checkpoint artifact hash differs",
    )
    return path


def compare_boundary(control, recovered, config, *, start):
    """Compare original committed containers, not serialized-file byte equality."""
    import torch

    from src.posttrain.checkpoint import assert_exact, inspect

    expected = inspect(control)
    actual = inspect(recovered, expected["contract"])
    require(expected["step"] == actual["step"], "comparison step differs")
    a = torch.load(control / "state.pt", map_location="cpu", weights_only=False)
    b = torch.load(recovered / "state.pt", map_location="cpu", weights_only=False)
    assert_exact(a, b)
    del a, b
    for rank in range(len(config.execution.learner_devices)):
        filename = f"rank-{rank:03d}.pt"
        assert_exact(
            torch.load(control / filename, map_location="cpu", weights_only=False),
            torch.load(recovered / filename, map_location="cpu", weights_only=False),
        )
    from .receipts import compare_replay, replay

    roots = [control.parent.parent, recovered.parent.parent]
    records, original_rows = [], []
    for root in roots:
        names = [str(p.relative_to(root)) for p in root.rglob("*.json")]

        def read(name, root=root):
            return json.loads((root / name).read_text())

        records.append(
            replay(
                read,
                names,
                "",
                start=start,
                finish=actual["step"],
                ranks=len(config.execution.learner_devices),
            )
        )
        attempts = {
            Path(n).stem.rsplit("-rank-", 1)[0]
            for n in names
            if n.startswith("completion/") and read(n)["step"] == actual["step"]
        }
        collected = {}
        for name in names:
            if name.startswith("collections/") and any(
                Path(name).name.startswith(a) for a in attempts
            ):
                row = read(name)
                if start <= row["step"] < actual["step"]:
                    collected["-round-" + name.split("-round-", 1)[1]] = row["records"]
                    for r in row["records"]:
                        for kind in ("image", "latent"):
                            require(
                                file_hash(r[kind + "_path"]) == r[kind + "_sha256"],
                                "replay artifact bytes differ",
                            )
        original_rows.append(collected)
    steps = actual["step"] - start
    rounds = steps // config.nft.updates_per_round
    compare_replay(
        records[0],
        records[1],
        updates=steps * len(config.execution.learner_devices),
        samples=rounds
        * config.collection.prompts_per_round
        * config.collection.draws_per_prompt,
    )
    for key in original_rows[0]:
        for a, b in zip(original_rows[0][key], original_rows[1][key], strict=True):
            assert_exact(
                torch.load(a["latent_path"], map_location="cpu", weights_only=False),
                torch.load(b["latent_path"], map_location="cpu", weights_only=False),
            )
    return {
        "role_optimizer_clock_reward_state_exact": True,
        "all_rank_rng_cursor_state_exact": True,
        "ranks": len(config.execution.learner_devices),
        "control_manifest": file_hash(control / "manifest.json"),
        "recovered_manifest": file_hash(recovered / "manifest.json"),
        "step": actual["step"],
        "serialized_byte_equality_claimed": False,
        "replay_records_exact": True,
        "recollected_samples": sum(len(v["records"]) for v in records[0][0].values()),
        "committed_updates": len(records[0][1]),
    }


def interrupt_boundary(argv, output, config, resume):
    """Kill the restored rank zero before a successful joint optimizer update."""
    with (output / "boundary-failure.log").open("w") as log:
        process = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + config.execution.startup_timeout
            while True:
                require(
                    process.poll() is None,
                    "boundary attempt exited before interruption",
                )
                require(
                    time.monotonic() < deadline,
                    "boundary interruption readiness timeout",
                )
                ready = sorted((output / "ready").glob("*-rank-*.json"))
                restored = sorted((output / "recovery").glob("*-rank-*.json"))
                if len(ready) == len(restored) == len(config.execution.learner_devices):
                    break
                time.sleep(0.025)
            require(
                not list((output / "updates").glob("*.json")),
                "boundary fault missed first joint update",
            )
            zero = next(p for p in ready if p.name.endswith("-rank-000.json"))
            record = json.loads(zero.read_text())
            require(
                record["stage"] == "cold", "restored readiness is not the cold boundary"
            )
            require(
                all(json.loads(p.read_text())["exact_rng"] for p in restored),
                "interruption did not restore every rank",
            )
            os.kill(record["pid"], signal.SIGKILL)
            code = process.wait(timeout=config.execution.round_timeout)
            require(code != 0, "interrupted learner unexpectedly succeeded")
            require(
                not list((output / "checkpoints").glob("step-*")),
                "boundary failure published a partial checkpoint",
            )
            require(
                not list((output / "updates").glob("*.json")),
                "boundary failure included a successful optimizer update",
            )
            receipt = {
                "rank": 0,
                "pid": record["pid"],
                "signal": "SIGKILL",
                "exit_code": code,
                "resume_checkpoint": str(resume),
                "restored_ranks": len(restored),
                "successful_joint_updates": 0,
                "partial_checkpoint_absent": True,
                "limit": "After restoring committed cold boundary and before a successful joint update; transition/collection may already have started",
            }
            atomic_json(output / "boundary-failure.json", receipt)
            return receipt
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def execute_case(case, deployment, assets_path, output):
    from src.posttrain.checkpoint import inspect

    config = load_training_config(case.config)
    output.mkdir(parents=True, exist_ok=False)
    resume = None
    if case.execution == "resume":
        resume = artifact(deployment, case.resume)
        manifest = inspect(resume)
        require(
            manifest["step"] == config.stages.cold_steps,
            "resume is not the declared cold boundary",
        )
        require(
            manifest["contract"]["source"] == source_files(), "boundary source differs"
        )
        require(
            manifest["contract"]["config"] == json.loads(json.dumps(asdict(config))),
            "boundary config differs",
        )
    argv = [
        sys.executable,
        "-m",
        "src.posttrain.train",
        "--config",
        case.config,
        "--assets",
        str(assets_path),
        "--output",
        str(output),
        "--stop-after",
        str(case.steps),
    ]
    if resume is not None:
        argv.extend(["--resume", str(resume)])
    failure = None
    if resume is not None:
        failure = interrupt_boundary(argv, output, config, resume)
    with (output / "driver.log").open("w") as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    contract = json.loads((output / "manifest.json").read_text())
    require(contract["source"] == source_files(), "executed source differs")
    require(
        contract["config"] == json.loads(json.dumps(asdict(config))),
        "executed config differs",
    )
    records = {
        path.name.rsplit("-rank-", 1)[-1].removesuffix(".json"): json.loads(
            path.read_text()
        )
        for path in (output / "completion").glob("*.json")
    }
    require(
        set(records)
        == {f"{r:03d}" for r in range(len(config.execution.learner_devices))}
        and all(
            r["step"] == case.steps and r["frozen_unchanged"] for r in records.values()
        ),
        "not every learner completed the declared qualification",
    )
    checkpoint = output / "checkpoints" / f"step-{case.steps:06d}"
    inspect(checkpoint, contract)
    comparison = None
    if case.compare:
        comparison = compare_boundary(
            artifact(deployment, case.compare),
            checkpoint,
            config,
            start=inspect(resume)["step"],
        )
        atomic_json(output / "recovery-comparison.json", comparison)
        attempts = {
            p.name.rsplit("-rank-", 1)[0]
            for p in (output / "completion").glob("*.json")
        }
        require(
            len(attempts) == 1, "successful completion attempt differs across ranks"
        )
        attempt = next(iter(attempts))
        recoveries = list((output / "recovery").glob(f"{attempt}-rank-*.json"))
        transitions = list((output / "transitions").glob(f"{attempt}-rank-*.json"))
        require(
            len(recoveries) == len(transitions) == len(records),
            "missing boundary restore/transition evidence",
        )
        require(
            all(json.loads(p.read_text())["exact_rng"] for p in recoveries)
            and all(json.loads(p.read_text())["optimizer_reset"] for p in transitions),
            "boundary recovery/transition failed",
        )
    exports = []
    for weights in case.exports:
        path = output / "exports" / weights
        with (output / f"export-{weights}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.posttrain.export",
                    "--checkpoint",
                    str(checkpoint),
                    "--weights",
                    weights,
                    "--output",
                    str(path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        receipt = json.loads((path / "export.json").read_text())
        require(
            receipt["reload_exact"]
            and receipt["source_checkpoint"] == file_hash(checkpoint / "manifest.json")
            and receipt["merged_sha256"] == file_hash(path / "weights.pt")
            and receipt["architecture_sha256"]
            == file_hash(path / "transformer_config.json"),
            "export files/receipt differ",
        )
        exports.append({"path": str(path), **receipt})
    return {
        "case": case.id,
        "execution": "new_process_boundary_recovery"
        if resume is not None
        else "executed",
        "config_sha256": case.config_sha256,
        "source_sha256": identity(contract["source"]),
        "capabilities": list(case.capabilities),
        "step": case.steps,
        "learner_count": len(records),
        "manifest_sha256": file_hash(output / "manifest.json"),
        "checkpoint_manifest_sha256": file_hash(checkpoint / "manifest.json"),
        "maximum_allocated_bytes": max(
            r["peak_allocated_bytes"] for r in records.values()
        ),
        "exports": exports,
        "comparison": comparison,
        "failure": failure,
    }


def evaluate(matrix, results, assets, output, base_config_path):
    """Materialize explicit per-export benchmark inputs; callers remain generic."""
    from src.evaluation.benchmarks.prompts import load_prompts
    from src.evaluation.benchmarks.run import run
    from src.posttrain.config import Arm, Checkpoint, load_config
    from src.posttrain.provenance import REPO

    base = load_config(base_config_path)
    arms = []
    checkpoints = dict(assets["checkpoints"])
    controls = set()
    by_id = {c.id: c for c in matrix.cases}
    for result in results:
        if result["execution"] == "reused_exact_source":
            continue
        config = load_training_config(by_id[result["case"]].config)
        key = (
            config.collection.rule,
            config.collection.steps,
            config.collection.guidance,
            config.collection.shift,
        )
        if key not in controls:
            controls.add(key)
            arms.append(Arm(f"initial_{len(controls)}", config.initialization, *key))
        for export in result["exports"]:
            identifier = f"{result['case']}_{export['weights']}"
            checkpoints[identifier] = str(Path(export["path"]) / "weights.pt")
            arms.append(
                Arm(
                    identifier,
                    Checkpoint(
                        identifier,
                        "live",
                        export["merged_sha256"],
                        export["architecture_sha256"],
                    ),
                    *key,
                )
            )
    require(bool(arms), "no executed exports to evaluate")
    evaluation = replace(base, arms=tuple(arms))
    eval_assets = {**assets, "checkpoints": checkpoints}
    atomic_json(output / "evaluation-config.json", asdict(evaluation))
    prompts = load_prompts(
        REPO / evaluation.evaluation.bringup_suite, evaluation.sampling.shapes
    )
    report = run(evaluation, eval_assets, prompts, output / "evaluation")
    require(
        all(
            metric["complete"]
            for arm in report["arms"].values()
            for metric in arm.values()
        ),
        "incomplete per-arm trained-export evaluation",
    )
    protocol = []
    for case in matrix.cases:
        if "scorer.substitution" not in case.capabilities:
            continue
        result = next(r for r in results if r["case"] == case.id)
        require(
            result["execution"] != "reused_exact_source",
            "complete substitution must actually execute",
        )
        config = load_training_config(case.config)
        arm_ids = {f"{case.id}_{e['weights']}" for e in result["exports"]}
        protocol_config = replace(
            evaluation,
            arms=tuple(a for a in evaluation.arms if a.name in arm_ids),
            scorers=config.scorers,
            evaluation=replace(
                evaluation.evaluation,
                channels=tuple(ch.channel for ch in config.rewards.channels),
                bringup_suite=config.collection.prompts,
            ),
        )
        target = output / "protocol-evaluation" / case.id
        atomic_json(target / "config.json", asdict(protocol_config))
        protocol_prompts = load_prompts(
            REPO / config.collection.prompts, protocol_config.sampling.shapes
        )
        protocol_report = run(
            protocol_config, eval_assets, protocol_prompts, target / "results"
        )
        require(
            all(
                m["complete"]
                for arm in protocol_report["arms"].values()
                for m in arm.values()
            ),
            "incomplete substituted-scorer benchmark",
        )
        rows = list((target / "results" / "scores").rglob("*.json"))
        require(bool(rows), "missing substituted-scorer envelopes")
        require(
            all(
                json.loads(p.read_text()).get("metadata", {}).get("rubric_diagnostic")
                for p in rows
            ),
            "missing structured rubric diagnostic response",
        )
        protocol.append(
            {
                "case": case.id,
                "config_sha256": file_hash(target / "config.json"),
                "report": str(target / "results" / "report.json"),
                "successful_envelopes": len(rows),
                "limit": "Deterministic protocol fixture, not a VLM or a quality metric",
            }
        )
    return {
        "evaluation": report,
        "manifest_sha256": file_hash(output / "evaluation" / "manifest.json"),
        "report_sha256": file_hash(output / "evaluation" / "report.json"),
        "images": len(list((output / "evaluation" / "records").glob("*.json"))),
        "score_envelopes": len(
            list((output / "evaluation" / "scores").rglob("*.json"))
        ),
        "protocol": protocol,
    }


def validate_evaluation(matrix, results, output, *, evidence=None):
    """Require actual named export arms and complete score/image records."""
    from src.evaluation.benchmarks.prompts import load_prompts
    from src.posttrain.config import load_config
    from src.posttrain.provenance import REPO

    summary = json.loads((output / "evaluation-summary.json").read_text())
    require(
        file_hash(output / "evaluation" / "manifest.json") == summary["manifest_sha256"]
        and file_hash(output / "evaluation" / "report.json")
        == summary["report_sha256"],
        "evaluation evidence hash differs",
    )
    manifest = json.loads((output / "evaluation" / "manifest.json").read_text())
    require(manifest["code"] == source_files(), "evaluation execution source differs")
    report = json.loads((output / "evaluation" / "report.json").read_text())
    actual_exports = {
        f"{result['case']}_{export['weights']}"
        for result in results
        if result["execution"] != "reused_exact_source"
        for export in result["exports"]
    }
    require(
        bool(actual_exports) and actual_exports <= report["arms"].keys(),
        "missing trained-export benchmark arm",
    )
    config = load_config(matrix.evaluation)
    prompts = load_prompts(
        REPO / config.evaluation.bringup_suite, config.sampling.shapes
    )
    require(
        all(
            metric["complete"] and metric["expected"] == len(prompts)
            for arm in report["arms"].values()
            for metric in arm.values()
        )
        and all(
            set(arm) == set(config.evaluation.channels)
            for arm in report["arms"].values()
        ),
        "missing complete declared metric coverage",
    )
    expected_records = len(report["arms"]) * len(prompts)
    require(
        summary["images"] == expected_records
        and summary["score_envelopes"] == expected_records * len(config.scorers)
        and len(list((output / "evaluation" / "records").glob("*.json")))
        == expected_records,
        "incomplete image/score evidence",
    )
    replacements = {
        c.id for c in matrix.cases if "scorer.substitution" in c.capabilities
    }
    require(
        {r["case"] for r in summary["protocol"]} == replacements,
        "missing complete scorer substitution evidence",
    )
    for receipt in summary["protocol"]:
        path = output / "protocol-evaluation" / receipt["case"]
        require(
            file_hash(path / "config.json") == receipt["config_sha256"],
            "substitution config differs",
        )
        protocol_report = json.loads((path / "results" / "report.json").read_text())
        require(
            bool(protocol_report["arms"])
            and all(
                m["complete"]
                for a in protocol_report["arms"].values()
                for m in a.values()
            ),
            "substituted benchmark is incomplete",
        )
    from hashlib import sha256

    from .receipts import benchmark, compare_replay, replay

    names = [str(p.relative_to(output)) for p in output.rglob("*.json")]

    def read(name):
        return json.loads((output / name).read_text())

    witness = None
    if evidence is not None and "completed" in evidence.archives:
        frozen = json.loads(evidence.read("completed", "evidence-summary.json"))
        require(
            frozen["source_sha256"] == matrix.source_sha256
            and frozen["matrix_sha256"] == identity(asdict(matrix))
            and frozen["results_sha256"] == identity(results),
            "target witness identities differ",
        )
        proof = json.loads(evidence.read("completed", "image-integrity.json"))
        require(
            all(p["all_recorded_image_latent_bytes_verified"] for p in proof),
            "target artifact verification incomplete",
        )

        def witness(name, record, kind=None):
            original = evidence.read("completed", "output/" + name)
            return file_hash(output / name) == sha256(
                original
            ).hexdigest() and record == json.loads(original)

    checked = benchmark(
        read,
        names,
        "evaluation/",
        prompts=prompts,
        source=source_files(),
        image_witness=witness or (lambda *args: False),
    )
    require(
        checked["score_envelopes"] == summary["score_envelopes"],
        "score summary count differs",
    )
    for result in results:
        if result["execution"] == "reused_exact_source":
            continue
        for export in result["exports"]:
            arm = checked["arms"][f"{result['case']}_{export['weights']}"]
            require(
                arm["checkpoint"]["sha256"] == export["merged_sha256"]
                and arm["checkpoint"]["architecture_sha256"]
                == export["architecture_sha256"],
                "benchmark export identity differs",
            )
    for receipt in summary["protocol"]:
        prefix = "protocol-evaluation/" + receipt["case"] + "/results/"
        pc = read("protocol-evaluation/" + receipt["case"] + "/config.json")
        pp = load_prompts(
            REPO / pc["evaluation"]["bringup_suite"], config.sampling.shapes
        )
        checked_protocol = benchmark(
            read,
            names,
            prefix,
            prompts=pp,
            source=source_files(),
            image_witness=witness or (lambda *args: False),
        )
        require(
            checked_protocol["score_envelopes"] == receipt["successful_envelopes"],
            "protocol envelope count differs",
        )
    reused_views, reused_cases, replay_checks = 0, 0, []
    checked_archives = {}
    for case in matrix.cases:
        if case.execution == "reuse":
            require(evidence is not None, "reused benchmark evidence required")
            key = case.evidence[0].archive
            archive = evidence.archive(key)

            def er(name, key=key):
                return json.loads(evidence.read(key, name))

            if key not in checked_archives:
                ip = er("image-integrity.json")
                require(
                    ip["saved_png_latent_hashes_match"]
                    and ip["all_required_scores_ok"],
                    "reused artifact verification incomplete",
                )

                def prior_witness(name, record, kind, ip=ip):
                    member = (
                        "evaluation/"
                        + record["arm"]
                        + "/"
                        + record["prompt"]["id"]
                        + (".png" if kind == "image" else ".pt")
                    )
                    return (
                        ip["images" if kind == "image" else "latents"][member]["sha256"]
                        == record[kind + "_sha256"]
                    )

                checked_archives[key] = benchmark(
                    er,
                    archive.getnames(),
                    "evaluation/",
                    prompts=prompts,
                    source=source_files(),
                    image_witness=prior_witness,
                )
            for pin in case.evidence:
                if pin.member.endswith("/export.json"):
                    export = evidence.pinned(pin)
                    arm = checked_archives[key]["arms"][Path(pin.member).parent.name]
                    require(
                        arm["checkpoint"]["sha256"] == export["merged_sha256"]
                        and arm["checkpoint"]["architecture_sha256"]
                        == export["architecture_sha256"],
                        "reused benchmark export differs",
                    )
                    reused_views += 1
            reused_cases += 1
        if case.execution == "resume":
            require(evidence is not None, "boundary replay witnesses required")
            c = load_training_config(case.config)
            control = next(
                x
                for x in matrix.cases
                if x.execution == "reuse" and x.config == case.config
            )
            pin = next(
                p
                for p in control.evidence
                if p.member.endswith("/manifest.json")
                and "/checkpoints/" not in p.member
            )
            key = pin.archive
            old_names = evidence.archive(key).getnames()

            def old_read(name, key=key):
                return json.loads(evidence.read(key, name))

            a = replay(
                old_read,
                old_names,
                pin.member.removesuffix("manifest.json"),
                start=c.stages.cold_steps,
                finish=case.steps,
                ranks=len(c.execution.learner_devices),
            )
            b = replay(
                read,
                names,
                "cases/" + case.id + "/",
                start=c.stages.cold_steps,
                finish=case.steps,
                ranks=len(c.execution.learner_devices),
                record_witness=witness,
            )
            expected_samples = (
                (case.steps - c.stages.cold_steps)
                // c.nft.updates_per_round
                * c.collection.prompts_per_round
                * c.collection.draws_per_prompt
            )
            compare_replay(
                a,
                b,
                updates=(case.steps - c.stages.cold_steps)
                * len(c.execution.learner_devices),
                samples=expected_samples,
            )
            samples = sum(len(v["records"]) for v in a[0].values())
            if "completed" in evidence.archives:
                target = json.loads(
                    evidence.read("completed", "boundary-input-comparisons.json")
                )
                measured = next(x for x in target if x["mode"] == c.student.mode)
                require(
                    measured["recollected_samples"] == samples
                    and measured["committed_updates"] == len(a[1])
                    and measured["old_terminal_conditioning_images_noise_exact"]
                    and measured["inputs_rewards_and_losses_exact"],
                    "missing actual target tensor replay witness",
                )
            else:
                result = next(r for r in results if r["case"] == case.id)
                measured = result["comparison"]
                require(
                    measured["replay_records_exact"]
                    and measured["recollected_samples"] == samples
                    and measured["committed_updates"] == len(a[1]),
                    "missing live tensor replay comparison",
                )
            replay_checks.append(
                {"case": case.id, "samples": samples, "updates": len(a[1])}
            )
    return {
        **summary,
        "reused_evaluation": {
            "cases": reused_cases,
            "export_views": reused_views,
            "images": reused_views * len(prompts),
            "score_envelopes": reused_views * len(prompts) * len(config.scorers),
        },
        "recovery_records": replay_checks,
        "artifact_limit": "Accessible bytes rehashed; inaccessible remote bytes require the pinned target verification witness. Remote files are not re-read locally.",
    }


def execute(matrix, deployment, *, evaluation_config, allocation_job, budget_ledger):
    """One guarded allocation, resident services, sequential independent cases."""
    from src.posttrain.scoring.lifecycle import Services

    output = Path(deployment["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    require(
        not (output / "results.json").exists(), "qualification results already exist"
    )
    from .budget import CAP_SECONDS, SHUTDOWN_MARGIN
    from .receipts import active_allocation

    ledger = json.loads(Path(budget_ledger).read_text())
    now = time.time()
    owned = active_allocation(
        ledger,
        allocation_job,
        now,
        cap_seconds=CAP_SECONDS,
        max_age=SHUTDOWN_MARGIN,
    )
    start_path = output / "allocation-start.json"
    if start_path.exists():
        start = json.loads(start_path.read_text())
        require(start["job"] == allocation_job, "executor allocation changed")
        previous = active_allocation(
            start["ledger"],
            allocation_job,
            start["started_at"],
            cap_seconds=CAP_SECONDS,
            max_age=SHUTDOWN_MARGIN,
        )
        require(
            all(
                owned[k] == previous[k]
                for k in ("allocated_at", "reserved_at", "workspace")
            ),
            "executor allocation identity changed",
        )
    else:
        start = {"job": allocation_job, "started_at": now, "ledger": ledger}
        atomic_json(start_path, start)
    assets = json.loads(Path(deployment["assets"]).read_text())
    from src.posttrain.provenance import REPO

    execution = {
        "source": source_files(),
        "matrix_sha256": identity(asdict(matrix)),
        "qualification_tools": {
            str(p.relative_to(REPO)): file_hash(p)
            for p in sorted((REPO / "scripts" / "posttrain").glob("*.py"))
        },
    }
    execution_path = output / "execution.json"
    require(
        not execution_path.exists()
        or json.loads(execution_path.read_text()) == execution,
        "qualification execution identity changed",
    )
    atomic_json(execution_path, execution)
    results = []
    evidence = Evidence(deployment["archives"])
    try:
        for case in matrix.cases:
            if case.execution == "reuse":
                results.append(
                    validate_reuse(case, load_training_config(case.config), evidence)
                )
    finally:
        evidence.close()
    specs = {}
    learner_devices = set()
    timeout = 0
    for case in matrix.cases:
        c = load_training_config(case.config)
        learner_devices.update(c.execution.learner_devices)
        timeout = max(timeout, c.execution.startup_timeout)
        for spec in c.scorers:
            require(
                spec.id not in specs or specs[spec.id] == spec,
                "scorer declarations differ across cases",
            )
            specs[spec.id] = spec
    # Evaluation's channels may be wider than any one training control.
    from src.posttrain.config import load_config

    for spec in load_config(evaluation_config).scorers:
        require(
            spec.id not in specs or specs[spec.id] == spec, "evaluation scorer differs"
        )
        specs[spec.id] = spec
    with Services(
        tuple(specs.values()),
        assets["services"],
        output / "resident",
        tuple(learner_devices),
        timeout,
    ) as services:
        training_assets = {
            **assets,
            "services": {
                key: {"endpoints": client.endpoints, "launches": []}
                for key, client in services.clients.items()
            },
        }
        assets_path = output / "resident-assets.json"
        atomic_json(assets_path, training_assets)
        atomic_json(output / "resident-startup.json", services.startup)
        for case in matrix.cases:
            if case.execution == "reuse":
                continue
            services.check()
            case_output = output / "cases" / case.id
            receipt = case_output / "result.json"
            if receipt.exists():
                result = json.loads(receipt.read_text())
                coverage_report(matrix, [result])
                from src.posttrain.checkpoint import inspect

                contract = json.loads((case_output / "manifest.json").read_text())
                require(
                    contract["source"] == source_files()
                    and file_hash(case_output / "manifest.json")
                    == result["manifest_sha256"],
                    "saved case manifest differs",
                )
                checkpoint = case_output / "checkpoints" / f"step-{case.steps:06d}"
                inspect(checkpoint, contract)
                require(
                    file_hash(checkpoint / "manifest.json")
                    == result["checkpoint_manifest_sha256"],
                    "saved checkpoint identity differs",
                )
                for export in result["exports"]:
                    path = Path(export["path"])
                    require(
                        file_hash(path / "weights.pt") == export["merged_sha256"]
                        and file_hash(path / "transformer_config.json")
                        == export["architecture_sha256"],
                        "saved export differs",
                    )
            else:
                result = execute_case(case, deployment, assets_path, case_output)
                atomic_json(receipt, result)
            results.append(result)
            atomic_json(output / "partial-results.json", results)
        services.check()
        evaluation = evaluate(
            matrix, results, training_assets, output, evaluation_config
        )
    atomic_json(output / "evaluation-summary.json", evaluation)
    atomic_json(output / "results.json", results)
    atomic_json(
        output / "allocation.json",
        {
            "job": allocation_job,
            "execution_sha256": identity(execution),
            "results_sha256": identity(results),
            "start_sha256": identity(start),
            "finished_at": time.time(),
        },
    )
    return results
