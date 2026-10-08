import copy
import json
from dataclasses import asdict, replace

import pytest

from scripts.posttrain.receipts import allocation, benchmark, compare_replay
from src.posttrain.records import file_hash, identity
from src.posttrain.scoring.client import request_for_record
from src.posttrain.scoring.contracts import ScoreResult, score_record


@pytest.fixture
def benchmark_input(tmp_path, sampling_config):
    arm = replace(sampling_config.arms[0], name="one")
    config = replace(sampling_config, arms=(arm,))
    prompt = {
        "id": "given",
        "text": "given prompt",
        "seed": 5,
        "aspect": "1:1",
        "structured": {"rubric": "input rubric"},
    }
    source = {"src/operator.py": "1" * 64}
    manifest = {
        "code": source,
        "prompts": [prompt],
        "visual_only": False,
        "config": {k: v for k, v in asdict(config).items() if k != "execution"},
        "policies": {"one": "2" * 64},
    }
    sample = identity(
        {
            "policy": "2" * 64,
            "prompt": prompt,
            "arm": asdict(arm),
            "sampling": asdict(config.sampling),
        }
    )
    image, latent = tmp_path / "image.png", tmp_path / "latent.pt"
    image.write_bytes(b"immutable image fixture")
    latent.write_bytes(b"immutable latent fixture")
    record = {
        "sample_id": sample,
        "arm": "one",
        "prompt": prompt,
        "checkpoint": asdict(arm.checkpoint),
        "image_path": str(image),
        "image_sha256": file_hash(image),
        "latent_path": str(latent),
        "latent_sha256": file_hash(latent),
    }
    request = request_for_record(record, config.scorers[0])
    row = score_record(
        "unfamiliar",
        ScoreResult.for_request(
            request,
            values={"fit": 0.25},
            metadata={"rubric_diagnostic": json.dumps({"rubric": "input rubric"})},
        ),
        1,
    )
    data = {
        "manifest.json": manifest,
        "report.json": {
            "arms": {
                "one": {
                    "unfamiliar.fit": {
                        "complete": True,
                        "expected": 1,
                        "scored": 1,
                        "failed_or_missing": [],
                        "mean_on_successes": 0.25,
                    }
                }
            }
        },
        f"records/{sample}.json": record,
        f"scores/unfamiliar/{sample}.json": row,
    }
    return data, source, prompt, image, latent, sample


def test_actual_benchmark_envelopes_and_bytes(benchmark_input):
    data, source, prompt, *_ = benchmark_input
    result = benchmark(
        data.__getitem__,
        data,
        "",
        prompts=[prompt],
        source=source,
        image_witness=lambda *a: False,
    )
    assert result["images"] == 1 and result["score_envelopes"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "score_missing",
        "record_missing",
        "fingerprint",
        "channels",
        "nonfinite",
        "diagnostic",
        "image_bytes",
        "latent_bytes",
        "summary",
    ],
)
def test_missing_or_corrupt_evidence_blocks(benchmark_input, fault):
    data, source, prompt, image, latent, sample = benchmark_input
    data = copy.deepcopy(data)
    score = f"scores/unfamiliar/{sample}.json"
    if fault == "score_missing":
        del data[score]
    elif fault == "record_missing":
        del data[f"records/{sample}.json"]
    elif fault == "fingerprint":
        data[score]["fingerprint"] = "3" * 64
    elif fault == "channels":
        data[score]["values"] = {"wrong": 0.25}
    elif fault == "nonfinite":
        data[score]["values"] = {"fit": float("nan")}
    elif fault == "diagnostic":
        data[score]["metadata"] = {}
    elif fault == "image_bytes":
        image.write_bytes(b"changed")
    elif fault == "latent_bytes":
        latent.write_bytes(b"changed")
    else:
        data["report.json"]["arms"]["one"]["unfamiliar.fit"]["mean_on_successes"] = 0.5
    with pytest.raises(ValueError):
        benchmark(
            data.__getitem__,
            data,
            "",
            prompts=[prompt],
            source=source,
            image_witness=lambda *a: False,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "empty",
        "unrelated",
        "unallocated",
        "unreleased",
        "cap",
        "execution",
        "old_job",
        "wrong_interval",
        "start_hash",
        "reservation",
    ],
)
def test_allocation_binds_actual_released_job(fault):
    execution = {"matrix_sha256": "1" * 64}
    results = [{"case": "given"}]
    state = active_ledger()
    start = {"job": "inko-posttrain-fixture", "started_at": 12, "ledger": state}
    receipt = {
        "job": "inko-posttrain-fixture",
        "execution_sha256": identity(execution),
        "results_sha256": identity(results),
        "start_sha256": identity(start),
        "finished_at": 18,
    }
    state = copy.deepcopy(state)
    state["jobs"][0]["released_at"] = 20
    if fault == "empty":
        state["jobs"] = []
    elif fault == "unrelated":
        state["jobs"][0]["name"] = "unrelated"
    elif fault == "unallocated":
        state["jobs"][0]["allocated_at"] = None
    elif fault == "unreleased":
        state["jobs"][0]["released_at"] = None
    elif fault == "cap":
        state["cap_seconds"] = 2000
    elif fault == "execution":
        receipt["execution_sha256"] = "0" * 64
    elif fault == "old_job":
        state["jobs"].append(
            {
                **state["jobs"][0],
                "name": "inko-posttrain-earlier",
                "allocated_at": 1,
                "released_at": 9,
            }
        )
        receipt["job"] = "inko-posttrain-earlier"
    elif fault == "wrong_interval":
        receipt["finished_at"] = 21
    elif fault == "start_hash":
        receipt["start_sha256"] = "0" * 64
    elif fault == "reservation":
        state["jobs"][0]["reserved_at"] = 5
    if fault == "none":
        assert (
            allocation(
                state,
                receipt,
                execution,
                results,
                "1" * 64,
                start=start,
                cap_seconds=1000,
                max_age=50,
            )
            == "inko-posttrain-fixture"
        )
    else:
        with pytest.raises(ValueError):
            allocation(
                state,
                receipt,
                execution,
                results,
                "1" * 64,
                start=start,
                cap_seconds=1000,
                max_age=50,
            )


def active_ledger():
    return {
        "schema": 1,
        "cap_seconds": 1000,
        "blocked": None,
        "jobs": [
            {
                "name": "inko-posttrain-fixture",
                "workspace": "fixture",
                "reserved_at": 8,
                "allocated_at": 10,
                "released_at": None,
                "stop_requested_at": None,
                "last_observed_at": 11,
                "observations": [{"at": 11, "allocated_at": 10, "released": False}],
            }
        ],
    }


@pytest.mark.parametrize(
    "fault",
    ["none", "released", "stale", "blocked", "stopping", "observation", "other_active"],
)
def test_execution_requires_fresh_active_allocation(fault):
    from scripts.posttrain.receipts import active_allocation

    state = active_ledger()
    now = 12
    if fault == "released":
        state["jobs"][0]["released_at"] = 12
    elif fault == "stale":
        now = 62
    elif fault == "blocked":
        state["blocked"] = "unknown release"
    elif fault == "stopping":
        state["jobs"][0]["stop_requested_at"] = 12
    elif fault == "observation":
        state["jobs"][0]["observations"] = []
    elif fault == "other_active":
        state["jobs"].append({**state["jobs"][0], "name": "inko-posttrain-other"})
    if fault == "none":
        assert (
            active_allocation(
                state, "inko-posttrain-fixture", now, cap_seconds=1000, max_age=50
            )["name"]
            == "inko-posttrain-fixture"
        )
    else:
        with pytest.raises(ValueError):
            active_allocation(
                state, "inko-posttrain-fixture", now, cap_seconds=1000, max_age=50
            )


@pytest.mark.parametrize("fault", ["none", "latent_path", "latent_sha256"])
def test_offline_replay_checks_pinned_references_before_normalizing(fault):
    from scripts.posttrain.receipts import replay

    original = {
        "completion/one-rank-000.json": {"step": 2},
        "collections/one-round-00000000.json": {
            "step": 1,
            "records": [{"latent_path": "/recorded/one.pt", "latent_sha256": "a" * 64}],
        },
        "updates/one-round-00000000-rank-000.json": {"step": 2},
    }
    current = copy.deepcopy(original)
    if fault != "none":
        current["collections/one-round-00000000.json"]["records"][0][fault] = "changed"
    args = {
        "start": 1,
        "finish": 2,
        "ranks": 1,
        "record_witness": lambda name, row: row == original[name],
    }
    if fault == "none":
        assert len(replay(current.__getitem__, current, "", **args)[0]) == 1
    else:
        with pytest.raises(ValueError):
            replay(current.__getitem__, current, "", **args)


@pytest.mark.parametrize("fault", ["reward", "conditioning", "missing"])
def test_replay_records_must_match_with_complete_exposure(fault):
    first = (
        {"round": {"records": [{"conditioning": "original"}]}},
        {"update": {"reward": 0.5}},
    )
    second = copy.deepcopy(first)
    if fault == "reward":
        second[1]["update"]["reward"] = 0.75
    elif fault == "conditioning":
        second[0]["round"]["records"][0]["conditioning"] = "changed"
    else:
        second[1].clear()
    with pytest.raises(ValueError):
        compare_replay(first, second, updates=1, samples=1)
