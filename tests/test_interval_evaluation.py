import json
from dataclasses import replace

import pytest
import torch

from scripts.posttrain.run_campaign import boundaries, evaluation_due
from src.evaluation.generation.config import Arm, Checkpoint
from src.evaluation.monitoring.config import Diversity
from src.evaluation.monitoring.diversity import validate_panel
from src.evaluation.monitoring.prompts import load_evaluation_prompts
from src.posttrain.checkpoint import inspect, publish
from src.posttrain.config import training_contract
from src.posttrain.records import atomic_json


def test_interval_boundaries_count_rounds(training_config):
    config = replace(
        training_config,
        training=replace(training_config.training, rounds=5),
        evaluation=replace(training_config.evaluation, every=2),
    )
    assert list(boundaries(config)) == [2, 4, 5]
    assert [r for r in range(6) if evaluation_due(config, r)] == [2, 4, 5]


def test_evaluation_selects_scenes_and_expands_seeds_without_changing_monitor(
    tmp_path, sampling_config
):
    rows = [
        {
            "id": identifier,
            "text": text,
            "seed": seed,
            "aspect": "1:1",
            "scene_id": identifier,
        }
        for identifier, text, seed in (
            ("cat", "A cat", 11),
            ("dog", "A dog", 12),
            ("bird", "A bird", 13),
        )
    ]
    source = tmp_path / "prompts.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = replace(
        sampling_config,
        evaluation=replace(
            sampling_config.evaluation,
            suite=str(source),
            monitor_suite=str(source),
            prompt_limit=2,
            draws_per_prompt=3,
        ),
    )
    expanded = load_evaluation_prompts(config, "development")
    assert [row["text"] for row in expanded] == ["A cat"] * 3 + ["A dog"] * 3
    assert [row["scene_id"] for row in expanded] == ["cat"] * 3 + ["dog"] * 3
    assert len({row["id"] for row in expanded}) == 6
    assert len({row["seed"] for row in expanded}) == 6
    assert load_evaluation_prompts(config, "monitor") == rows

    larger = replace(config, evaluation=replace(config.evaluation, prompt_limit=3))
    assert load_evaluation_prompts(larger, "development")[:6] == expanded
    single = replace(
        config,
        evaluation=replace(config.evaluation, prompt_limit=0, draws_per_prompt=1),
    )
    assert load_evaluation_prompts(single, "breadth") == rows


def test_diversity_validation_applies_only_to_declared_suites(sampling_config):
    config = replace(
        sampling_config,
        evaluation=replace(
            sampling_config.evaluation,
            diversity=Diversity(("development",), "lpips", 1, 1),
        ),
    )
    prompt = {"id": "p", "scene_id": "s", "text": "caption", "seed": 1, "aspect": "1:1"}
    validate_panel(config, [prompt], "monitor")
    validate_panel(config, [prompt], "confirmation")
    with pytest.raises(ValueError, match="at least two seeds"):
        validate_panel(config, [prompt], "development")
    validate_panel(config, [prompt, dict(prompt, id="q", seed=2)], "development")


def test_interval_sampling_uses_declared_benchmark_rule_and_reuses_export(
    tmp_path, monkeypatch, training_config
):
    from scripts.posttrain import run_campaign

    config = replace(
        training_config,
        evaluation=replace(
            training_config.evaluation,
            suites=("monitor",),
            steps=7,
            guidance=2.5,
            shift="none",
        ),
    )
    point = publish(
        tmp_path / "training/checkpoints",
        2,
        training_contract(config, [], 1, {}),
        {"step": 2, "committed_rounds": 1},
        {},
        0,
        torch.device("cpu"),
        keep_last=0,
    )
    output = tmp_path / "evaluation"
    atomic_json(
        output / "exports/ema/export.json",
        {"weights": "ema", "step": 2, "committed_rounds": 1},
    )
    baseline = Arm("base", Checkpoint("initial", "initial"), 7, 2.5, "none")
    received = []

    def export(*args):
        raise AssertionError("a published export must be reused")

    def evaluate(config, assets, rows, destination, **kwargs):
        received.append(config.arms)
        assert kwargs == {
            "visual_only": True,
            "suite": "monitor",
            "reference_cache": output.parent / "entry" / "monitor",
        }
        return {"suite": "monitor"}

    monkeypatch.setattr(run_campaign, "load_evaluation_prompts", lambda *args: [])
    monkeypatch.setattr(run_campaign, "export", export)
    monkeypatch.setattr(run_campaign, "run", evaluate)
    monkeypatch.setattr(run_campaign, "panel", lambda *args: None)
    receipt = run_campaign._evaluate_checkpoint(
        point, config, {"checkpoints": {}}, output, None
    )
    assert len(received) == 1 and received[0][0] == baseline
    student = received[0][1]
    assert (
        student.name,
        student.checkpoint,
        student.steps,
        student.guidance,
        student.shift,
    ) == ("student_ema", Checkpoint("student_ema", "ema"), 7, 2.5, "none")
    assert receipt["step"] == 2 and receipt["committed_rounds"] == 1


def test_recovery_keeps_complete_evaluations_and_finishes_interrupted_before_training(
    tmp_path, monkeypatch, training_config
):
    from scripts.posttrain import run_campaign

    config = replace(
        training_config,
        training=replace(training_config.training, rounds=3),
        evaluation=replace(
            training_config.evaluation, suites=("development",), weights=("live",)
        ),
    )
    prompts = []
    contract = training_contract(config, prompts, 1, {})
    output = tmp_path / "run"
    checkpoints = output / "training/checkpoints"

    def checkpoint(number):
        return publish(
            checkpoints,
            2 * number,
            contract,
            {"step": 2 * number, "committed_rounds": number},
            {},
            0,
            torch.device("cpu"),
            keep_last=0,
        )

    checkpoint(1)
    checkpoint(2)
    atomic_json(output / "evaluation/round-000001/evaluation.json", {"step": 1})
    events = []

    def evaluate(path, training, assets, destination, **kwargs):
        number = inspect(path)["committed_rounds"]
        events.append(("evaluate", number))
        atomic_json(destination / "evaluation.json", {"step": number})

    def train(argv, **kwargs):
        events.append(
            (
                "train",
                inspect(argv[argv.index("--resume") + 1])["committed_rounds"],
                int(argv[argv.index("--stop-after") + 1]),
            )
        )
        checkpoint(3)

    monkeypatch.setattr(run_campaign, "load_training_config", lambda path: config)
    monkeypatch.setattr(run_campaign, "load_rollout_prompts", lambda *args: prompts)
    monkeypatch.setattr(run_campaign, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(run_campaign.subprocess, "run", train)
    assets = tmp_path / "assets.json"
    atomic_json(
        assets,
        {
            "rollout": "unused.jsonl",
            "device_type": "cpu",
            "scoring": {"endpoint": "http://fixture"},
        },
    )
    monkeypatch.setattr(
        "src.posttrain.scoring.client.ScoreClient.info", lambda self: {"semantics": {}}
    )
    run_campaign.execute(tmp_path / "training.toml", assets, output)
    assert events == [("evaluate", 2), ("train", 2, 3), ("evaluate", 3)]
    assert json.loads((output / "complete.json").read_text()) == {
        "committed_rounds": 3,
        "step": 6,
    }
