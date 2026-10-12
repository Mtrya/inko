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
