"""Campaign tracking identity, rank aggregation, and committed update delivery."""

import json
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from test_posttrain_dmd import config

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
    with Tracker(root, "cold_estimator", config={"setting": "input"}):
        pass
    assert sdk.calls[0]["mode"] == "online"
    assert sdk.calls[0]["resume"] == "never"
    assert sdk.calls[0]["settings"] == {"interactive": False}
    assert sdk.calls[0]["project"] == "inko-posttrain"
    assert sdk.calls[0]["name"] == "cold_estimator"
    with Tracker(root, "cold_estimator"):
        pass
    assert sdk.calls[1]["id"] == "generated-run"
    assert sdk.calls[1]["resume"] == "must"
    assert sdk.finishes == [{"state": "success"}, {"state": "success"}]


def test_checkpoint_owns_tracking_identity_and_new_arm_does_not_inherit(tmp_path, sdk):
    root = tmp_path / "source"
    with Tracker(root, "cold"):
        identity = json.loads((root / "tracking.json").read_text())
        checkpoint = publish(
            root / "checkpoints",
            1,
            {},
            {"step": 1},
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
    with Tracker(tmp_path / "joint_branch", "nft_only"):
        pass
    assert "id" not in sdk.calls[-1] and sdk.calls[-1]["resume"] == "never"


@pytest.mark.parametrize("mismatch", ["project", "id", "mode", "root"])
def test_tracking_mismatch_fails_startup(tmp_path, sdk, mismatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    atomic_json(
        checkpoint / "tracking.json",
        {
            "swanlab_project": "another-project"
            if mismatch == "project"
            else "inko-posttrain",
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


def test_missing_online_credentials_prevents_training_start(tmp_path, monkeypatch, sdk):
    from src.posttrain import train

    def fail(**kwargs):
        raise RuntimeError("no API key")

    sdk.init = fail
    launched = []
    monkeypatch.setattr(train.subprocess, "Popen", lambda *a, **k: launched.append(a))
    with pytest.raises(RuntimeError, match="no API key"):
        train.launch(
            config(),
            {"learner": {"python_paths": []}, "device_type": "cpu"},
            {"boundary": None},
            tmp_path / "run",
            None,
            3,
            tmp_path / "arm.json",
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
        "student_version": 1,
        "fake_version": 5,
        "seconds": 3.0,
        "peak_allocated_bytes": 100,
        "student_loss": loss,
        "student_grad_norm": 0.5,
        "fake_losses": [2.0, 4.0],
        "fake_grad_norms": [0.1, 0.3],
        "learning_rates": {"student": {"adamw/0": 0.001}, "fake": {"muon/0": 0.01}},
        "fake_probe": {"velocity_mse": 1.5, "score_weighted_mse": 2.5, "samples": 4},
        "cold_schedule": {"real_scale": 0.7, "clock": "generator"},
        "reward": {
            "statistics": {"service.metric": {"mean": 3.0, "std": 2.0}},
            "diagnostics": {"effective_samples": 2.0},
        },
    }


def test_rank_scaled_losses_average_and_global_probe_is_not_summed():
    metrics = update_metrics([record(2), record(6)])
    assert metrics["train/student_loss"] == 4
    assert metrics["train/fake_losses"] == 3
    assert metrics["fake_probe/velocity_mse"] == 1.5
    assert metrics["reward/service.metric/std"] == 2
    assert metrics["reward/effective_samples"] == 2
    assert metrics["clock/fake_updates"] == 5
    assert metrics["lr/fake/muon/0"] == 0.01
    assert metrics["cold_schedule/real_scale"] == 0.7
    assert "cold_schedule/clock" not in metrics


def test_committed_multi_inner_round_logs_every_update_after_all_ranks(
    tmp_path, sdk, sampling_config
):
    from test_posttrain_joint import joint_config

    c = joint_config(sampling_config.scorers[0])
    c = replace(c, execution=replace(c.execution, learner_devices=(0, 1)))
    with Tracker(tmp_path, "joint") as tracker:
        delivery = TrainingMetrics(tracker, c, tmp_path, "attempt")
        for inner in range(2):
            for rank in range(2):
                atomic_json(
                    tmp_path
                    / "updates"
                    / f"attempt-round-000000-inner-{inner:03d}-rank-{rank:03d}.json",
                    record(2 + 4 * rank),
                )
        delivery.poll()
        assert sdk.logs == []  # An unfinished NFT round is not committed progress.
        atomic_json(tmp_path / "rounds" / "attempt-round-000000-rank-000.json", {})
        delivery.poll()
        assert sdk.logs == []
        atomic_json(tmp_path / "rounds" / "attempt-round-000000-rank-001.json", {})
        delivery.poll()
        delivery.poll()
    assert [step for step, _ in sdk.logs] == [1, 2]
    assert all(metrics["train/student_loss"] == 4 for _, metrics in sdk.logs)


def test_fitting_and_resume_deliver_only_new_attempt_records(tmp_path, sdk):
    c = config()
    c = replace(c, fake=replace(c.fake, initial_steps=3))
    with Tracker(tmp_path, "cold") as tracker:
        delivery = TrainingMetrics(
            tracker, c, tmp_path, "new", saved={"step": 0, "initial_fit_steps": 1}
        )
        for attempt in ("old", "new"):
            for fit in (1, 2, 3):
                atomic_json(
                    tmp_path / "fitting" / f"{attempt}-fit-{fit:06d}-rank-000.json",
                    record(),
                )
        delivery.poll()
        atomic_json(
            tmp_path / "updates" / "new-round-000000-inner-000-rank-000.json", record()
        )
        atomic_json(tmp_path / "rounds" / "new-round-000000-rank-000.json", {})
        delivery.poll()
    assert [step for step, _ in sdk.logs] == [2, 3, 4]


def test_actual_optimizer_rates_are_read_after_schedule_step():
    from src.posttrain.optim import RoleOptimizer

    c = config()
    model = torch.nn.Linear(2, 1)
    optimizer = RoleOptimizer(
        model, replace(c.student_optimizer, lr=0.1, warmup_steps=2), 3
    )
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    optimizer.step(loss)
    assert optimizer_rates(optimizer) == {"adamw/0": 0.05}
    assert optimizer_rates(None) == {}


def test_interval_scores_named_views_and_panels_share_training_run(tmp_path, sdk):
    with Tracker(tmp_path, "cold"):
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


def test_coordinator_delivers_final_update_before_finish(tmp_path, monkeypatch, sdk):
    from src.posttrain import train

    c = config()
    c = replace(
        c, training=replace(c.training, steps=1), stages=replace(c.stages, cold_steps=1)
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
            atomic_json(output / "rounds" / f"{attempt}-round-000000-rank-000.json", {})

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(train.subprocess, "Popen", CompletedLearner)
    train.launch(
        c,
        {"learner": {"python_paths": []}, "device_type": "cpu"},
        {"boundary": None},
        output,
        None,
        1,
        tmp_path / "cold.json",
        tmp_path / "assets.json",
        track=True,
    )
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
        {"swanlab_project": "inko-posttrain", "swanlab_run_id": "cold-run"},
    )
    received = []

    def evaluate(*args):
        received.append(args[-1])
        assert sdk.get_run().id == "cold-run"
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
