import copy
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from PIL import Image
from test_posttrain_dmd import config as dmd_config

from scripts.posttrain import coverage, qualification, qualify
from scripts.posttrain.protocol_fixture import Fixture
from src.posttrain.config import Scorer
from src.posttrain.records import identity
from src.posttrain.scoring.client import ScoreClient, request_for_record
from src.posttrain.scoring.config import Limits
from src.posttrain.scoring.contracts import Channel


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def matrix_input(tmp_path, monkeypatch, sampling_config):
    source = {"src/example.py": "1" * 64}
    monkeypatch.setattr(coverage, "source_files", lambda: source)
    monkeypatch.setattr(
        coverage, "qualification_files", lambda: {"scripts/runner.py": "2" * 64}
    )
    path = tmp_path / "training.toml"
    path.write_text("fixture training input\n")
    config = dmd_config()
    monkeypatch.setattr(coverage, "load_training_config", lambda p: config)
    evaluation = tmp_path / "evaluation.toml"
    evaluation.write_text("fixture evaluation input\n")
    monkeypatch.setattr("src.posttrain.config.load_config", lambda p: sampling_config)
    case = {
        "id": "one",
        "config": str(path),
        "config_sha256": sha(path.read_bytes()),
        "steps": 2,
        "capabilities": ["cold.000.full", "export.live"],
        "execution": "run",
        "resume": "",
        "compare": "",
        "exports": ["live"],
        "evidence": [],
    }
    matrix = {
        "source_sha256": identity(source),
        "evaluation": str(evaluation),
        "evaluation_sha256": sha(evaluation.read_bytes()),
        "requirements": ["cold.000.full", "export.live"],
        "cases": [case],
    }
    return matrix, config, source


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "config",
        "evaluation",
        "unknown",
        "duplicate",
        "unassigned",
        "unreached",
        "steps",
    ],
)
def test_invalid_matrix_blocks_before_execution(matrix_input, fault):
    matrix, _, _ = matrix_input
    matrix = copy.deepcopy(matrix)
    if fault == "source":
        matrix["source_sha256"] = "0" * 64
    elif fault == "config":
        matrix["cases"][0]["config_sha256"] = "0" * 64
    elif fault == "evaluation":
        matrix["evaluation_sha256"] = "0" * 64
    elif fault == "unknown":
        matrix["cases"][0]["capabilities"] += ["fictional.capability"]
    elif fault == "duplicate":
        matrix["cases"] *= 2
    elif fault == "unassigned":
        matrix["requirements"] += ["nft.grid"]
    elif fault == "unreached":
        matrix["cases"][0]["capabilities"] += ["reference.joint"]
    else:
        matrix["cases"][0]["steps"] = 0
    with pytest.raises(ValueError):
        coverage.validate_matrix(matrix)


def test_assignment_is_not_execution_or_review(matrix_input, tmp_path):
    matrix = coverage.validate_matrix(matrix_input[0])
    empty = coverage.coverage_report(matrix, [])
    assert not empty["qualified"]
    assert empty["missing_cases"] == ["one"]
    assert empty["missing_capabilities"] == ["cold.000.full", "export.live"]
    assert empty["missing_reviews"] == ["independent", "native"]
    result = {
        "case": "one",
        "config_sha256": matrix_input[0]["cases"][0]["config_sha256"],
        "source_sha256": matrix_input[0]["source_sha256"],
        "step": 2,
        "capabilities": ["cold.000.full", "export.live"],
    }
    report = coverage.coverage_report(matrix, [result])
    assert report["missing_capabilities"] == []
    assert report["missing_reviews"] == ["independent", "native"]
    assert not report["qualified"]  # A tiny fixture cannot certify the entire family.
    path = tmp_path / "review.txt"
    path.write_text("")
    review = {
        "kind": "native",
        "source_sha256": matrix_input[0]["source_sha256"],
        "matrix_sha256": identity(asdict(matrix)),
        "qualification_sha256": identity({"scripts/runner.py": "2" * 64}),
        "results_sha256": identity([result]),
        "output": str(path),
        "output_sha256": sha(b""),
        "verdict": "pass",
        "exit_code": 0,
    }
    with pytest.raises(ValueError, match="empty review"):
        coverage.coverage_report(matrix, [result], reviews=[review])
    path.write_text("A recorded independent review output\n")
    review["output_sha256"] = sha(path.read_bytes())
    review["results_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identities"):
        coverage.coverage_report(matrix, [result], reviews=[review])
    review["results_sha256"] = identity([result])
    review["qualification_sha256"] = identity({"scripts/runner.py": "3" * 64})
    with pytest.raises(ValueError, match="identities"):
        coverage.coverage_report(matrix, [result], reviews=[review])


def test_archive_identity_and_absent_member_reject(tmp_path):
    path = tmp_path / "evidence.tar.gz"
    raw = b'{"record":"actual fixture bytes"}\n'
    with tarfile.open(path, "w:gz") as tar:
        member = tarfile.TarInfo("receipt.json")
        member.size = len(raw)
        tar.addfile(member, io.BytesIO(raw))
    evidence = coverage.Evidence(
        {"one": {"path": str(path), "sha256": sha(path.read_bytes())}}
    )
    try:
        good = coverage.EvidencePin("one", "receipt.json", sha(raw))
        assert evidence.pinned(good) == {"record": "actual fixture bytes"}
        with pytest.raises(ValueError, match="member hash"):
            evidence.pinned(replace(good, sha256="0" * 64))
        with pytest.raises(ValueError, match="missing evidence member"):
            evidence.pinned(replace(good, member="absent.json"))
    finally:
        evidence.close()
    changed = coverage.Evidence({"one": {"path": str(path), "sha256": "0" * 64}})
    try:
        with pytest.raises(ValueError, match="archive hash"):
            changed.archive("one")
    finally:
        changed.close()


def test_budget_preflight_rejects_exhaustion_without_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(qualify, "CAP_SECONDS", 2000)
    ledger = tmp_path / "ledger.json"
    state = {
        "schema": 1,
        "cap_seconds": 2000,
        "blocked": None,
        "jobs": [{"name": "used", "allocated_at": 0, "released_at": 1950}],
    }
    ledger.write_text(json.dumps(state))
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="budget exhausted"):
        qualify.budget_admission(ledger, now=2000)
    assert ledger.read_bytes() == before
    state["jobs"] = []
    ledger.write_text(json.dumps(state))
    before = ledger.read_bytes()
    qualify.budget_admission(ledger, now=2000)
    assert ledger.read_bytes() == before


@pytest.mark.parametrize(
    "flags",
    [
        ["--validate-only", "--execute-cases"],
        ["--promote"],
        ["--reviews", "unused"],
        ["--execute-cases"],
        ["--execute-cases", "--allocation-job", "inko-posttrain-fixture"],
        ["--allocation-job", "inko-posttrain-fixture"],
    ],
)
def test_ambiguous_cli_cannot_fall_through_to_job_submission(flags):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.posttrain.qualify",
            "--matrix",
            "/absent-input",
            "--deployment",
            "/absent-input",
            *flags,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "FileNotFoundError" not in result.stderr


def test_boundary_fault_kills_restored_rank_without_successful_update(tmp_path):
    config = replace(
        dmd_config(), execution=replace(dmd_config().execution, learner_devices=(0, 1))
    )
    script = """
import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
for folder in ('ready', 'recovery'):
    (root / folder).mkdir()
for rank in range(2):
    name = f'attempt-rank-{rank:03d}.json'
    (root / 'recovery' / name).write_text(json.dumps({'exact_rng': True}))
    (root / 'ready' / name).write_text(json.dumps({'pid': os.getpid(), 'stage': 'cold'}))
time.sleep(30)
"""
    receipt = qualification.interrupt_boundary(
        [sys.executable, "-c", script, str(tmp_path)],
        tmp_path,
        config,
        tmp_path / "committed",
    )
    assert receipt["signal"] == "SIGKILL"
    assert receipt["exit_code"] == -9
    assert receipt["restored_ranks"] == 2
    assert receipt["successful_joint_updates"] == 0
    assert receipt["partial_checkpoint_absent"]


def test_boundary_comparison_checks_containers_and_every_rank(tmp_path, monkeypatch):
    import torch

    runs = [tmp_path / "control", tmp_path / "recovered"]
    control, recovered = [r / "checkpoints" / "step-000008" for r in runs]
    for root in (control, recovered):
        root.mkdir(parents=True)
        (root / "manifest.json").write_text("committed fixture identity")
        torch.save(
            {"student": torch.tensor([1.0, 2.0]), "optimizer": {"step": 3}, "clock": 8},
            root / "state.pt",
        )
        for rank in range(2):
            torch.save(
                {"rng": torch.tensor([rank, 7], dtype=torch.uint8), "cursor": [2, 1]},
                root / f"rank-{rank:03d}.pt",
            )
    for run in runs:
        for folder in ["completion", "collections", "updates"]:
            (run / folder).mkdir()
        for rank in range(2):
            image, latent = run / f"image-{rank}.png", run / f"latent-{rank}.pt"
            image.write_bytes(b"immutable fixture image")
            torch.save({"terminal": torch.tensor([rank, 3.0])}, latent)
            (run / "completion" / f"attempt-rank-{rank:03d}.json").write_text(
                json.dumps({"step": 8})
            )
            row = {
                "sample_id": str(rank),
                "image_path": str(image),
                "image_sha256": sha(image.read_bytes()),
                "latent_path": str(latent),
                "latent_sha256": sha(latent.read_bytes()),
            }
            (
                run / "collections" / f"attempt-round-000007-rank-{rank:03d}.json"
            ).write_text(json.dumps({"step": 7, "rank": rank, "records": [row]}))
            (
                run / "updates" / f"attempt-round-000007-inner-000-rank-{rank:03d}.json"
            ).write_text(
                json.dumps(
                    {"step": 8, "reward": 0.5, "attempt": "telemetry", "seconds": 1.0}
                )
            )
    monkeypatch.setattr(
        "src.posttrain.checkpoint.inspect", lambda *args: {"step": 8, "contract": {}}
    )
    config = replace(
        dmd_config(),
        execution=replace(dmd_config().execution, learner_devices=(0, 1)),
        collection=replace(
            dmd_config().collection, prompts_per_round=2, draws_per_prompt=1
        ),
    )
    result = qualification.compare_boundary(control, recovered, config, start=7)
    assert result["all_rank_rng_cursor_state_exact"] and result["ranks"] == 2
    assert result["replay_records_exact"]
    torch.save(
        {"rng": torch.tensor([1, 8], dtype=torch.uint8), "cursor": [2, 1]},
        recovered / "rank-001.pt",
    )
    with pytest.raises(ValueError, match="tensor differs"):
        qualification.compare_boundary(control, recovered, config, start=7)


@pytest.mark.parametrize("fault", ["empty", "absent_records", "source"])
def test_evaluation_metadata_cannot_promote_missing_execution(
    matrix_input, tmp_path, monkeypatch, fault
):
    matrix = coverage.validate_matrix(matrix_input[0])
    monkeypatch.setattr(qualification, "source_files", lambda: matrix_input[2])
    monkeypatch.setattr(
        "src.evaluation.benchmarks.prompts.load_prompts",
        lambda *args: [{"id": "a"}, {"id": "b"}],
    )
    output = tmp_path / "evaluation"
    output.mkdir()
    manifest = {
        "code": matrix_input[2] if fault != "source" else {"src/example.py": "2" * 64}
    }
    report = {
        "arms": {}
        if fault == "empty"
        else {"one_live": {"unfamiliar.fit": {"complete": True, "expected": 2}}}
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "report.json").write_text(json.dumps(report))
    (tmp_path / "evaluation-summary.json").write_text(
        json.dumps(
            {
                "manifest_sha256": sha((output / "manifest.json").read_bytes()),
                "report_sha256": sha((output / "report.json").read_bytes()),
                "images": 2,
                "score_envelopes": 2,
                "protocol": [],
            }
        )
    )
    result = {"case": "one", "execution": "executed", "exports": [{"weights": "live"}]}
    with pytest.raises(ValueError):
        qualification.validate_evaluation(matrix, [result], tmp_path)


def test_independent_fixture_cli_structured_score_and_diagnostic(tmp_path):
    script = Path("scripts/posttrain/protocol_fixture.py")
    config = Fixture(
        sha(script.read_bytes()),
        "image-rubric-v1",
        ("rubric",),
        (Channel("penalty", "lower", (0.0, 1.0), "Protocol fixture"),),
        "Confirm transport",
        Limits(2, 4, 5.0, 1000000, 1),
    )
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "service"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.posttrain.protocol_fixture",
            "--config",
            str(path),
            "--output",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 10
        while not (output / "ready.json").exists():
            assert process.poll() is None, process.stdout.read().decode()
            assert time.monotonic() < deadline
            time.sleep(0.05)
        ready = json.loads((output / "ready.json").read_text())
        spec = Scorer(
            "unfamiliar_service",
            identity(asdict(config)),
            "image-rubric-v1",
            ("rubric",),
            config.outputs,
            1,
            5.0,
            1,
            2,
            "png_base64",
            "none",
            0,
        )
        client = ScoreClient(spec, [ready["endpoint"]], tmp_path / "client")
        client.handshake()
        records = []
        for value in [0, 255]:
            image = tmp_path / f"{value}.png"
            Image.new("RGB", (8, 8), (value, 10, 20)).save(image)
            records.append(
                {
                    "sample_id": str(value),
                    "prompt": {
                        "text": "given prompt",
                        "structured": {"rubric": "Confirm transport"},
                    },
                    "image_path": str(image),
                    "image_sha256": sha(image.read_bytes()),
                }
            )
        results = client.acquire(
            [request_for_record(record, spec) for record in records]
        )
        assert [r["values"] for r in results] == [{"penalty": 0.0}, {"penalty": 1.0}]
        diagnostic = json.loads(results[0]["metadata"]["rubric_diagnostic"])
        assert diagnostic == {
            "rubric": "Confirm transport",
            "kind": "protocol-fixture",
            "quality_claim": False,
        }
    finally:
        process.terminate()
        process.wait(timeout=10)
