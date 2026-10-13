import json
from dataclasses import replace

import pytest
import torch

from scripts.posttrain.run_campaign import committed_checkpoints
from src.evaluation.benchmarks.diversity import validate_panel
from src.posttrain.checkpoint import inspect, load, publish
from src.posttrain.clock import Clock
from src.posttrain.config import Diversity
from src.posttrain.evaluation_config import (
    IntervalEvaluation,
    boundaries,
    evaluation_due,
)
from src.posttrain.provenance import training_contract
from src.posttrain.records import atomic_json
from src.posttrain.rollout import collect_art, collect_nft


@pytest.mark.parametrize(
    "round_index,nft_step,art_step", [(0, 0, 0), (5, 5, 10), (5, 5, 4)]
)
def test_collection_noise_is_shared_across_algorithms_and_optimizer_clocks(
    training_config, art_config, round_index, nft_step, art_step
):
    prompts = [{"id": "p0"}, {"id": "p1"}]
    ids = [f"{round_index}:0:0", f"{round_index}:1:0"]
    nft = collect_nft(
        lambda x, *args: torch.zeros_like(x),
        training_config,
        prompts,
        ids,
        round_index,
        round_index,
        nft_step,
        (),
        None,
        "cpu",
    )
    art = collect_art(
        lambda x, *args: torch.zeros_like(x),
        art_config,
        prompts[::-1],
        ids[::-1],
        [1, 0],
        round_index,
        art_step,
        (),
        "cpu",
    )
    assert nft.seeds == art.seeds[::-1]
    assert nft.seeds[0] != nft.seeds[1]
    torch.testing.assert_close(nft.terminal, art.states.flip(0)[:, 0], rtol=0, atol=0)
    assert nft.generator_step == nft_step and art.generator_step == art_step


def test_skipped_rounds_reach_budget_and_publish_distinct_recovery_points(
    tmp_path, art_config
):
    config = replace(art_config, training=replace(art_config.training, rounds=3))
    clock = Clock()
    device = torch.device("cpu")
    points = []
    for round_number in range(4):
        if round_number:
            clock.skip_round(config)
        clock.validate(config)
        points.append(
            publish(tmp_path, clock.step, {}, clock.state_dict(), {}, 0, device)
        )
    assert [p.name for p in points] == [
        "round-000000",
        "round-000001",
        "round-000002",
        "round-000003",
    ]
    assert clock.next_action(config) == "done"
    assert clock.step == 0 and clock.skipped_rounds == 3
    assert clock.samples_scored == 12 and clock.samples_used == 0
    saved, _ = load(points[-1], {}, 0)
    recovered = Clock.from_state(saved)
    recovered.validate(config)
    assert recovered.next_action(config) == "done"
    with pytest.raises(ValueError, match="unfinished ART"):
        recovered.skip_round(config)
    assert committed_checkpoints(tmp_path) == list(enumerate(points))
    assert all(inspect(p)["step"] == 0 for p in points)


@pytest.mark.parametrize("objective", ["nft", "art"])
def test_collection_budget_does_not_require_equal_optimizer_steps(
    training_config, art_config, objective
):
    config = training_config if objective == "nft" else art_config
    config = replace(config, training=replace(config.training, rounds=3))
    clock = Clock()
    for round_number in range(3):
        if objective == "art" and round_number == 1:
            clock.skip_round(config)
        else:
            for _ in range(config.updates_per_round):
                clock.update(config)
            clock.commit_round(config, 0.0 if objective == "nft" else None)
        clock.validate(config)
    assert clock.committed_rounds == 3 and clock.samples_scored == 12
    assert clock.step == (3 if objective == "nft" else 4)
    assert clock.next_action(config) == "done"
    with pytest.raises(ValueError, match="beyond the training horizon"):
        clock.update(config)


def test_interval_boundaries_count_rounds_for_both_objectives(
    training_config, art_config
):
    interval = IntervalEvaluation(
        "sampling.json", 2, ("development",), ("ema",), True, 7
    )
    for original in (training_config, art_config):
        config = replace(original, training=replace(original.training, rounds=5))
        assert list(boundaries(config, interval)) == [2, 4, 5]
        assert [r for r in range(6) if evaluation_due(config, interval, r)] == [2, 4, 5]


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


def test_recovery_keeps_complete_evaluations_and_finishes_interrupted_before_training(
    tmp_path, monkeypatch, training_config
):
    from scripts.posttrain import run_campaign

    config = replace(
        training_config, training=replace(training_config.training, rounds=3)
    )
    interval = IntervalEvaluation(
        "sampling.json", 1, ("development",), ("live",), True, 7
    )
    prompts = []
    contract = training_contract(config, prompts)
    output = tmp_path / "run"
    checkpoints = output / "training/checkpoints"

    def checkpoint(number):
        return publish(
            checkpoints,
            number,
            contract,
            {"step": number, "committed_rounds": number},
            {},
            0,
            torch.device("cpu"),
        )

    checkpoint(1)
    checkpoint(2)
    atomic_json(output / "evaluation/round-000001/evaluation.json", {"step": 1})
    events = []

    def evaluate(path, training, interval, assets, destination, **kwargs):
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
    monkeypatch.setattr(run_campaign, "load_interval_config", lambda path: interval)
    monkeypatch.setattr(run_campaign, "load_prompts", lambda *args: prompts)
    monkeypatch.setattr(run_campaign, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(run_campaign.subprocess, "run", train)
    assets = tmp_path / "assets.json"
    atomic_json(assets, {})
    run_campaign.execute(
        tmp_path / "training.toml", tmp_path / "interval.toml", assets, output
    )
    assert events == [("evaluate", 2), ("train", 2, 3), ("evaluate", 3)]
    assert json.loads((output / "complete.json").read_text()) == {
        "committed_rounds": 3,
        "step": 3,
    }
