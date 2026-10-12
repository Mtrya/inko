"""Run tracking identity, rank aggregation, and committed update delivery."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from src.posttrain.checkpoint import publish
from src.posttrain.records import atomic_json
from src.posttrain.tracking import (
    Tracker,
    TrainingMetrics,
    optimizer_rates,
    update_metrics,
)


@pytest.fixture
def sdk(monkeypatch):
    class SwanLab:
        def __init__(self):
            self.calls, self.logs, self.finishes = [], [], []
            self.run = None

        def Settings(self, **kwargs):
            return kwargs

        def init(self, **kwargs):
            self.calls.append(kwargs)
            self.run = SimpleNamespace(
                id=kwargs.get("id", "generated-run"), mode="online"
            )
            return self.run

        def get_run(self):
            return self.run

        def log(self, metrics, step):
            self.logs.append((step, metrics))

        def Image(self, path):
            return ("image", path)

        def finish(self, **kwargs):
            self.finishes.append(kwargs)
            self.run = None

    sdk = SwanLab()
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    return sdk


def test_online_start_and_manifest_only_restart_keep_identity(tmp_path, sdk):
    root = tmp_path / "arm"
    with Tracker(root, "nft_reference", config={"setting": "input"}):
        pass
    assert sdk.calls[0]["mode"] == "online"
    assert sdk.calls[0]["resume"] == "never"
    assert sdk.calls[0]["settings"] == {"interactive": False}
    assert sdk.calls[0]["project"] == "inko-posttrain"
    assert sdk.calls[0]["name"] == "nft_reference"
    with Tracker(root, "nft_reference"):
        pass
    assert sdk.calls[1]["id"] == "generated-run"
    assert sdk.calls[1]["resume"] == "must"
    assert sdk.finishes == [{"state": "success"}, {"state": "success"}]


def test_checkpoint_owns_tracking_identity_and_new_arm_does_not_inherit(tmp_path, sdk):
    root = tmp_path / "source"
    with Tracker(root, "run"):
        identity = json.loads((root / "tracking.json").read_text())
        checkpoint = publish(
            root / "checkpoints",
            1,
            {},
            {"step": 1, "committed_rounds": 1},
            {},
            0,
            torch.device("cpu"),
            identity,
        )
    assert json.loads((checkpoint / "tracking.json").read_text()) == {
        "swanlab_project": "inko-posttrain",
        "swanlab_run_id": "generated-run",
    }
    with Tracker(tmp_path / "restored", None, resume=checkpoint):
        pass
    assert sdk.calls[-1]["id"] == "generated-run" and sdk.calls[-1]["resume"] == "must"
    with Tracker(tmp_path / "another_arm", "other"):
        pass
    assert "id" not in sdk.calls[-1] and sdk.calls[-1]["resume"] == "never"


@pytest.mark.parametrize("mismatch", ["project", "id", "mode", "root"])
def test_tracking_mismatch_fails_startup(tmp_path, sdk, mismatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    atomic_json(
        checkpoint / "tracking.json",
        {
            "swanlab_project": (
                "another-project" if mismatch == "project" else "inko-posttrain"
            ),
            "swanlab_run_id": "saved-run",
        },
    )
    if mismatch in {"mode", "id"}:
        sdk.init = lambda **kwargs: SimpleNamespace(
            id="wrong-run" if mismatch == "id" else "saved-run",
            mode="offline" if mismatch == "mode" else "online",
        )
    root = tmp_path / "output"
    if mismatch == "root":
        atomic_json(
            root / "tracking.json",
            {"swanlab_project": "inko-posttrain", "swanlab_run_id": "another-run"},
        )
    with (
        pytest.raises(ValueError, match="project|experiment|online|identity"),
        Tracker(root, "arm", resume=checkpoint),
    ):
        pytest.fail("startup should reject mismatched tracking")


def test_missing_online_credentials_prevents_training_start(
    tmp_path, monkeypatch, sdk, training_config
):
    from src.posttrain import train

    def fail(**kwargs):
        raise RuntimeError("no API key")

    sdk.init = fail
    launched = []
    real_popen = subprocess.Popen

    def popen(argv, **kwargs):
        if "src.posttrain.train" in argv:
            launched.append(argv)
            raise AssertionError("learner must not start without tracking")
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(train.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="no API key"):
        train.launch(
            training_config,
            {
                "learner": {"python_paths": []},
                "device_type": "cpu",
                "services": {"unfamiliar": {"endpoints": [], "launches": []}},
            },
            {"prompts": []},
            tmp_path / "run",
            None,
            3,
            tmp_path / "run.json",
            tmp_path / "assets.json",
            track=True,
        )
    assert launched == []


def test_failure_marks_run_crashed(tmp_path, sdk):
    with pytest.raises(RuntimeError, match="training error"), Tracker(tmp_path, "arm"):
        raise RuntimeError("training error")
    assert sdk.finishes == [{"state": "crashed"}]


def record(loss=2):
    return {
        "step": 1,
        "seconds": 3.0,
        "peak_allocated_bytes": 100,
        "student_loss": loss,
        "student_grad_norm": 0.5,
        "term_losses": {"nft": 1.0, "reference": 0.25},
        "learning_rates": {"student": {"muon/0": 0.01}},
        "reward": {
            "statistics": {"hpsv3.mu": {"mean": 3.0, "std": 2.0}},
            "advantages": [1.0, -1.0],
            "probabilities": [1.0, 0.0],
        },
    }


def test_rank_scaled_losses_average_and_reward_statistics_join():
    metrics = update_metrics([record(2), record(6)])
    assert metrics["train/student_loss"] == 4
    assert metrics["train/student_grad_norm"] == 0.5
    assert metrics["train/nft_loss"] == 1
    assert metrics["train/reference_loss"] == 0.25
    assert metrics["lr/student/muon/0"] == 0.01
    assert metrics["reward/hpsv3.mu/mean"] == 3
    assert metrics["reward/advantages_mean"] == 0
    assert metrics["reward/probabilities_std"] == 0.5


def test_committed_multi_inner_round_logs_every_update_after_all_ranks(
    tmp_path, sdk, training_config
):
    config = replace(
        training_config,
        nft=replace(training_config.nft, updates_per_round=2),
        training=replace(training_config.training, steps=2, checkpoint_every=2),
        collection=replace(training_config.collection, prompts_per_round=2),
    )
    with Tracker(tmp_path, "nft") as tracker:
        delivery = TrainingMetrics(tracker, config, tmp_path, "attempt")
        for inner in range(2):
            for rank in range(1):
                atomic_json(
                    tmp_path
                    / "updates"
                    / f"attempt-round-000000-inner-{inner:03d}-rank-{rank:03d}.json",
                    record(2 + 4 * rank),
                )
        delivery.poll()
        assert sdk.logs == []  # An unfinished round is not committed progress.
        atomic_json(
            tmp_path / "rounds" / "attempt-round-000000-rank-000.json",
            {"skipped_zero_guidance": False},
        )
        delivery.poll()
        delivery.poll()
    assert [step for step, _ in sdk.logs] == [1, 2]
    assert all(metrics["train/student_loss"] == 2 for _, metrics in sdk.logs)


def test_resume_delivers_only_new_attempt_records(tmp_path, sdk, training_config):
    config = replace(
        training_config,
        training=replace(training_config.training, steps=1),
        collection=replace(training_config.collection, prompts_per_round=2),
    )
    with Tracker(tmp_path, "nft") as tracker:
        delivery = TrainingMetrics(
            tracker, config, tmp_path, "new", saved={"step": 0, "committed_rounds": 0}
        )
        for attempt in ("old", "new"):
            atomic_json(
                tmp_path
                / "updates"
                / f"{attempt}-round-000000-inner-000-rank-000.json",
                record(),
            )
            atomic_json(
                tmp_path / "rounds" / f"{attempt}-round-000000-rank-000.json",
                {"skipped_zero_guidance": False},
            )
        delivery.poll()
    assert [step for step, _ in sdk.logs] == [1]


def test_art_tracking_resumes_collection_count_and_passes_skipped_rounds(
    tmp_path, sdk, art_config
):
    config = replace(
        art_config,
        execution=replace(art_config.execution, learner_devices=(0, 1)),
        training=replace(art_config.training, steps=6),
    )
    with Tracker(tmp_path, "art") as tracker:
        delivery = TrainingMetrics(
            tracker, config, tmp_path, "new", saved={"step": 2, "committed_rounds": 3}
        )
        atomic_json(
            tmp_path / "rounds/new-round-000003-rank-000.json",
            {"skipped_zero_guidance": True},
        )
        delivery.poll()
        assert delivery.round == 3 and sdk.logs == []
        atomic_json(
            tmp_path / "rounds/new-round-000003-rank-001.json",
            {"skipped_zero_guidance": True},
        )
        delivery.poll()
        assert delivery.round == 4 and delivery.step == 2 and sdk.logs == []
        for rank in (0, 1):
            for inner in (0, 1):
                atomic_json(
                    tmp_path
                    / "updates"
                    / f"new-round-000004-inner-{inner:03d}-rank-{rank:03d}.json",
                    dict(record(), step=3 + inner),
                )
            atomic_json(
                tmp_path / "rounds" / f"new-round-000004-rank-{rank:03d}.json",
                {"skipped_zero_guidance": False},
            )
        delivery.poll()
        delivery.poll()
    assert [step for step, _ in sdk.logs] == [3, 4]


def test_actual_optimizer_rates_are_read_after_schedule_step(training_config):
    from src.posttrain.optim import RoleOptimizer

    model = torch.nn.Linear(2, 1)
    optimizer = RoleOptimizer(
        model,
        replace(training_config.student_optimizer, lr=0.1, warmup_steps=2),
        3,
    )
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    optimizer.step(loss)
    assert optimizer_rates(optimizer) == {"adamw/0": 0.05}
    assert optimizer_rates(None) == {}


def test_interval_scores_named_views_and_panels_share_training_run(tmp_path, sdk):
    with Tracker(tmp_path, "run"):
        pass
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tracking.json").write_text((tmp_path / "tracking.json").read_text())
    with Tracker(tmp_path, None, resume=checkpoint) as tracker:
        tracker.evaluation(
            {
                "suite": "development",
                "arms": {
                    "student_ema_slow": {
                        "score.value": {
                            "mean_on_successes": 2.5,
                            "population_stddev": 0.3,
                            "scored": 12,
                        }
                    }
                },
                "diversity": {"arms": {"student_ema_slow": {"mean_by_scene": 0.4}}},
            },
            [tmp_path / "panel.png"],
            17,
        )
    assert sdk.calls[-1]["id"] == "generated-run"
    step, metrics = sdk.logs[0]
    assert step == 17
    assert (
        metrics["evaluation/development/student_ema_slow/score.value/mean_on_successes"]
        == 2.5
    )
    assert metrics["evaluation/development/student_ema_slow/lpips"] == 0.4
    assert metrics["evaluation/development/panels"] == [
        ("image", str(tmp_path / "panel.png"))
    ]


def test_coordinator_delivers_final_update_before_finish(
    tmp_path, monkeypatch, sdk, sampling_config, training_config
):
    from test_posttrain_scoring import Backend, Endpoint

    from src.posttrain import train

    config = replace(
        training_config,
        training=replace(training_config.training, steps=1),
        collection=replace(training_config.collection, prompts_per_round=2),
    )
    output = tmp_path / "training"

    class CompletedLearner:
        returncode = 0

        def __init__(self, argv, **kwargs):
            attempt = argv[argv.index("--attempt") + 1]
            assert (output / "tracking.json").exists()
            atomic_json(
                output / "updates" / f"{attempt}-round-000000-inner-000-rank-000.json",
                record(),
            )
            atomic_json(
                output / "rounds" / f"{attempt}-round-000000-rank-000.json",
                {"skipped_zero_guidance": False},
            )

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    real_popen = subprocess.Popen

    def popen(argv, **kwargs):
        if "src.posttrain.train" in argv:
            return CompletedLearner(argv, **kwargs)
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(train.subprocess, "Popen", popen)
    endpoint = Endpoint(tmp_path, sampling_config.scorers[0], Backend())
    try:
        train.launch(
            config,
            {
                "learner": {"python_paths": []},
                "device_type": "cpu",
                "services": {
                    "unfamiliar": {"endpoints": [endpoint.url], "launches": []}
                },
            },
            {"prompts": []},
            output,
            None,
            1,
            tmp_path / "run.json",
            tmp_path / "assets.json",
            track=True,
        )
    finally:
        endpoint.close()
    assert len(sdk.logs) == 1 and sdk.logs[0][0] == 1
    assert sdk.finishes == [{"state": "success"}]


def test_interval_runner_opens_checkpoint_tracker_before_evaluation(
    tmp_path, monkeypatch, sdk
):
    from scripts.posttrain import run_campaign

    root = tmp_path / "training"
    checkpoint = root / "checkpoints" / "step-000001"
    checkpoint.mkdir(parents=True)
    atomic_json(
        checkpoint / "tracking.json",
        {"swanlab_project": "inko-posttrain", "swanlab_run_id": "run"},
    )
    received = []

    def evaluate(*args):
        received.append(args[-1])
        assert sdk.get_run().id == "run"
        return {"step": 1}

    monkeypatch.setattr(run_campaign, "_evaluate_checkpoint", evaluate)
    assert run_campaign.evaluate_checkpoint(
        checkpoint, None, None, {}, tmp_path / "eval"
    ) == {"step": 1}
    assert isinstance(received[0], Tracker)
    assert sdk.calls[0]["resume"] == "must"


def test_generation_only_cli_logs_panels_and_diversity(
    tmp_path, monkeypatch, sdk, sampling_config
):
    from scripts.posttrain import readouts
    from src.evaluation.benchmarks import run as benchmark

    source, assets, output = (
        tmp_path / "guidance.json",
        tmp_path / "assets.json",
        tmp_path / "output",
    )
    atomic_json(source, asdict(sampling_config))
    atomic_json(assets, {})
    monkeypatch.setattr(benchmark, "load_prompts", lambda *args: [])
    monkeypatch.setattr(benchmark, "validate_panel", lambda *args: None)

    def generate(*args, **kwargs):
        assert sdk.get_run().id == "generated-run"
        return {
            "suite": "monitor",
            "diversity": {"arms": {"guidance": {"mean_by_scene": 0.6}}},
        }

    def panels(run, blind, seed):
        blind.mkdir()
        (blind / "p.png").write_bytes(b"panel fixture")

    monkeypatch.setattr(benchmark, "run", generate)
    monkeypatch.setattr(readouts, "panel", panels)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--config",
            str(source),
            "--assets",
            str(assets),
            "--output",
            str(output),
            "--suite",
            "monitor",
            "--track",
        ],
    )
    benchmark.main()
    assert sdk.calls[0]["name"] == "guidance"
    assert sdk.logs[0][1]["evaluation/monitor/guidance/lpips"] == 0.6
    assert sdk.logs[0][1]["evaluation/monitor/panels"] == [
        ("image", str(output / "blind" / "p.png"))
    ]
