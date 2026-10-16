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
            self.config, self.definitions = {}, {}
            self.step = 0

        def Settings(self, **kwargs):
            return kwargs

        def init(self, **kwargs):
            self.calls.append(kwargs)
            self.config.update(kwargs.get("config") or {})
            self.run = SimpleNamespace(
                id=kwargs.get("id", "generated-run"), mode="online"
            )
            return self.run

        def get_run(self):
            if self.run is None:
                raise RuntimeError("No active Run. Call swanlab.init() first.")
            return self.run

        def define_metric(self, key, **kwargs):
            self.definitions.setdefault(key, {}).update(kwargs)

        def log(self, metrics, step=None):
            step = self.step + 1 if step is None else step
            self.step = max(self.step, step)
            self.logs.append((step, metrics))

        def Image(self, path, **kwargs):
            return ("image", path, kwargs)

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
    assert sdk.calls[0]["project"] == "inko"
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
            keep_last=0,
        )
    assert json.loads((checkpoint / "tracking.json").read_text()) == {
        "swanlab_project": "inko",
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
            "swanlab_project": ("another-project" if mismatch == "project" else "inko"),
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
            {"swanlab_project": "inko", "swanlab_run_id": "another-run"},
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


@pytest.mark.parametrize("stage", ["initialization", "metric_setup"])
def test_startup_failure_preserves_error_and_closes_only_started_run(
    tmp_path, sdk, stage
):
    error = RuntimeError("tracking startup failed")

    def fail(*args, **kwargs):
        raise error

    if stage == "initialization":
        sdk.init = fail
    else:
        sdk.define_metric = fail
    with pytest.raises(RuntimeError) as caught:
        Tracker(tmp_path, "arm").__enter__()
    assert caught.value is error
    assert sdk.finishes == ([] if stage == "initialization" else [{"state": "crashed"}])


def record(loss=2):
    return {
        "step": 1,
        "update_seconds": 0.3,
        "student_loss": loss,
        "student_grad_norm": 0.5,
        "reference_enabled": True,
        "art_items": 4,
        "art_kept": 4,
        "art_masked": 0,
        "term_losses": {"art": loss, "reference": 0.25},
        "learning_rates": {"muon": 0.01},
    }


def publish_round(output, attempt, number, rank=0, *, skipped=False, skips=0):
    from src.posttrain.round_scoring import reward_path

    atomic_json(
        output / "rounds" / f"{attempt}-round-{number:06d}-rank-{rank:03d}.json",
        {
            "skipped_zero_guidance": skipped,
            "active_group_fraction": 0.0 if skipped else 0.75,
            "clock": {"samples_scored": 4 * (number + 1), "skipped_rounds": skips},
            "timings": {
                "collection": 3.0,
                "scoring_wait": 5.0,
                "guidance": 0.2,
                "round": 10.0,
            },
            "peak_allocated_bytes": 1024**3,
        },
    )
    ids = [f"{number}:{slot}:{draw}" for slot in (0, 1) for draw in (0, 1)]
    atomic_json(
        reward_path(output, attempt, number),
        {
            "reward": {
                "sample_ids": ids,
                "combined": [0, 1, 0, 0],
                "raw": {"unfamiliar.fit": [1, 3, 2, 2]},
            }
        },
    )


def test_rank_scaled_objective_losses_average_and_disabled_reference_is_omitted():
    metrics = update_metrics([record(2), record(6)])
    assert metrics == {
        "train/grad_norm": 0.5,
        "train/art_loss": 4,
        "train/reference_loss": 0.25,
        "train/art_inactive_fraction": 0.0,
        "train/art_rejected_fraction": 0.0,
        "train/lr/muon": 0.01,
    }
    disabled = dict(record(), reference_enabled=False)
    assert "train/reference_loss" not in update_metrics([disabled])


def test_committed_multi_inner_round_logs_every_update_after_all_ranks(
    tmp_path, sdk, training_config
):
    config = replace(
        training_config,
        training=replace(training_config.training, rounds=1, checkpoint_every=1),
        collection=replace(training_config.collection, prompts_per_round=2),
    )
    with Tracker(tmp_path, "art") as tracker:
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
        publish_round(tmp_path, "attempt", 0)
        delivery.poll()
        delivery.poll()
    updates = [
        (step, values) for step, values in sdk.logs if "train/art_loss" in values
    ]
    assert [step for step, _ in updates] == [1, 2]
    assert all(metrics["train/art_loss"] == 2 for _, metrics in updates)
    collections = [
        (step, values)
        for step, values in sdk.logs
        if "train/reward/unfamiliar.fit/mean" in values
    ]
    assert len(collections) == 1
    assert collections[0][0] == 1
    assert collections[0][1]["train/reward/unfamiliar.fit/mean"] == 2
    assert collections[0][1]["train/art_active_group_fraction"] == 0.75
    assert sdk.definitions["train/reward/*"]["x_axis"] == "runtime/collection_round"


def test_resume_delivers_only_new_attempt_records(tmp_path, sdk, training_config):
    config = replace(
        training_config,
        training=replace(training_config.training, rounds=1),
        collection=replace(training_config.collection, prompts_per_round=2),
    )
    with Tracker(tmp_path, "art") as tracker:
        delivery = TrainingMetrics(
            tracker, config, tmp_path, "new", saved={"step": 0, "committed_rounds": 0}
        )
        for attempt in ("old", "new"):
            for inner in (0, 1):
                atomic_json(
                    tmp_path
                    / "updates"
                    / f"{attempt}-round-000000-inner-{inner:03d}-rank-000.json",
                    dict(record(), step=inner + 1),
                )
            publish_round(tmp_path, attempt, 0)
        delivery.poll()
    assert [step for step, values in sdk.logs if "train/art_loss" in values] == [1, 2]


def test_art_tracking_resumes_collection_count_and_passes_skipped_rounds(
    tmp_path, sdk, training_config
):
    config = replace(
        training_config,
        execution=replace(training_config.execution, learner_devices=(0, 1)),
        training=replace(training_config.training, rounds=6),
    )
    with Tracker(tmp_path, "art") as tracker:
        delivery = TrainingMetrics(
            tracker, config, tmp_path, "new", saved={"step": 2, "committed_rounds": 3}
        )
        publish_round(tmp_path, "new", 3, skipped=True, skips=3)
        delivery.poll()
        assert delivery.round == 3 and sdk.logs == []
        publish_round(tmp_path, "new", 3, rank=1, skipped=True, skips=3)
        delivery.poll()
        assert delivery.round == 4 and delivery.step == 2
        assert sdk.logs[0][0] == 4
        assert sdk.logs[0][1]["train/art_active_group_fraction"] == 0
        assert sdk.logs[1][1]["runtime/skipped_collections"] == 3
        assert sdk.logs[1][1]["runtime/training_images_scored"] == 16
        for rank in (0, 1):
            for inner in (0, 1):
                atomic_json(
                    tmp_path
                    / "updates"
                    / f"new-round-000004-inner-{inner:03d}-rank-{rank:03d}.json",
                    dict(record(), step=3 + inner),
                )
            publish_round(tmp_path, "new", 4, rank=rank, skips=3)
        delivery.poll()
        delivery.poll()
    assert [step for step, values in sdk.logs if "train/art_loss" in values] == [3, 4]
    assert [
        values["runtime/collection_round"]
        for _, values in sdk.logs
        if "runtime/collection_round" in values
    ] == [4, 5]


def test_actual_optimizer_rates_are_read_after_step(training_config):
    from src.posttrain.optim import RoleOptimizer

    model = torch.nn.ModuleDict(
        {
            "ordinary": torch.nn.Linear(2, 1),
            "c_mlp": torch.nn.Sequential(torch.nn.Linear(2, 1)),
        }
    )
    optimizer = RoleOptimizer(
        model,
        replace(training_config.student_optimizer, lr=0.1),
    )
    loss = sum(layer(torch.ones(1, 2)).square().sum() for layer in model.values())
    loss.backward()
    optimizer.step(loss)
    assert len(optimizer.optimizers[0].param_groups) == 2
    assert optimizer_rates(optimizer) == {"adamw": 0.1}


def test_interval_scores_gains_and_diversity_share_training_run(
    tmp_path, sdk, sampling_config
):
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
                        "unfamiliar.fit": {
                            "mean_on_successes": 2.5,
                            "population_stddev": 0.3,
                            "scored": 12,
                            "expected": 12,
                            "complete": True,
                        }
                    }
                },
                "paired": {
                    "student_ema_slow": {
                        "unfamiliar.fit": {
                            "coverage_complete": True,
                            "mean_delta": 0.5,
                            "interval": [0.2, 0.8],
                        }
                    }
                },
                "diversity": {"arms": {"student_ema_slow": {"mean_by_scene": 0.4}}},
            },
            tmp_path,
            sampling_config,
            17,
        )
    assert sdk.calls[-1]["id"] == "generated-run"
    step, metrics = next(
        (step, metrics)
        for step, metrics in sdk.logs
        if "evaluation/development/student_ema_slow/unfamiliar.fit/mean" in metrics
    )
    assert step == 17
    assert metrics["evaluation/development/student_ema_slow/unfamiliar.fit/mean"] == 2.5
    assert metrics["monitor/lpips/development/student_ema_slow"] == 0.4
    assert (
        metrics["evaluation/development/student_ema_slow/unfamiliar.fit/delta"] == 0.5
    )
    assert (
        metrics["evaluation/development/student_ema_slow/unfamiliar.fit/delta_lower"]
        == 0.2
    )
    assert (
        metrics["evaluation/development/student_ema_slow/unfamiliar.fit/delta_upper"]
        == 0.8
    )
    assert all(not key.endswith("/panels") for key in metrics)


def test_coordinator_delivers_final_update_before_finish(
    tmp_path, monkeypatch, sdk, sampling_config, training_config
):
    from test_scoring import Backend, Endpoint

    from src.posttrain import train

    config = replace(
        training_config,
        training=replace(training_config.training, rounds=1),
        collection=replace(training_config.collection, prompts_per_round=2),
    )
    output = tmp_path / "training"

    class CompletedLearner:
        returncode = 0

        def __init__(self, argv, **kwargs):
            attempt = argv[argv.index("--attempt") + 1]
            assert (output / "tracking.json").exists()
            for inner in (0, 1):
                atomic_json(
                    output
                    / "updates"
                    / f"{attempt}-round-000000-inner-{inner:03d}-rank-000.json",
                    dict(record(), step=inner + 1),
                )
            publish_round(output, attempt, 0)

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
    assert [step for step, values in sdk.logs if "train/art_loss" in values] == [1, 2]
    assert sdk.finishes == [{"state": "success"}]


def test_interval_runner_opens_checkpoint_tracker_before_evaluation(
    tmp_path, monkeypatch, sdk
):
    from scripts.posttrain import run_campaign

    root = tmp_path / "training"
    checkpoint = root / "checkpoints" / "round-000001"
    checkpoint.mkdir(parents=True)
    atomic_json(
        checkpoint / "tracking.json",
        {"swanlab_project": "inko", "swanlab_run_id": "run"},
    )
    received = []

    def evaluate(*args):
        received.append(args[-1])
        assert sdk.get_run().id == "run"
        return {"step": 1}

    monkeypatch.setattr(run_campaign, "_evaluate_checkpoint", evaluate)
    assert run_campaign.evaluate_checkpoint(
        checkpoint, None, {}, tmp_path / "eval"
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
    monkeypatch.setattr(benchmark, "load_evaluation_prompts", lambda *args: [])
    monkeypatch.setattr(benchmark, "validate_panel", lambda *args: None)

    def generate(*args, **kwargs):
        assert sdk.get_run().id == "generated-run"
        monitor_records(output, ("reference", "candidate"))
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
    assert sdk.logs[0][1]["monitor/lpips/monitor/guidance"] == 0.6
    media = sdk.logs[0][1]["monitor/images/scene"]
    assert len(media) == 2
    assert media[0][0] == "image"
    assert "zh / short" in media[0][2]["caption"]
    assert "seed=11" in media[0][2]["caption"]
    assert sdk.config["evaluation"]["sampling"]["text_exit_layer"] == 2


def monitor_records(root, arms):
    from PIL import Image

    for arm_index, arm in enumerate(arms):
        for variant_index, (lang, variant) in enumerate(
            (
                ("zh", "short"),
                ("en", "short"),
                ("zh", "long"),
                ("en", "long"),
            )
        ):
            for seed in (11, 29):
                key = f"{arm}-{lang}-{variant}-{seed}"
                path = root / "images" / f"{key}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                color = (80 + 40 * variant_index, 20 + 120 * arm_index, 30)
                Image.new("RGB", (32, 48), color).save(path)
                atomic_json(
                    root / "records" / f"{key}.json",
                    {
                        "arm": arm,
                        "image_path": str(path),
                        "prompt": {
                            "id": f"{lang}-{variant}-{seed}",
                            "scene_id": "scene",
                            "lang": lang,
                            "variant": variant,
                            "seed": seed,
                            "text": f"完整提示 {lang} {variant}",
                        },
                    },
                )


def test_monitor_grids_preserve_arm_order_variant_order_and_seeds(
    tmp_path, sampling_config
):
    import numpy as np
    from PIL import Image

    from src.evaluation.benchmarks.monitor import scene_grids

    monitor_records(tmp_path, ("reference", "candidate"))
    grids = scene_grids(tmp_path, tuple(reversed(sampling_config.arms)), 17)
    assert set(grids) == {"scene"}
    assert len(grids["scene"]) == 2
    for (path, caption), seed in zip(grids["scene"], (11, 29), strict=True):
        assert "Step 17" in caption and f"seed={seed}" in caption
        assert "完整提示 zh long" in caption
        with Image.open(path) as image:
            pixels = np.asarray(image).astype(float)
        centers = {}
        for arm_index, arm in enumerate(("reference", "candidate")):
            for row in range(4):
                color = np.array((80 + 40 * row, 20 + 120 * arm_index, 30))
                y, x = np.where(np.abs(pixels - color).max(axis=2) < 5)
                assert len(x) > 100
                centers[arm, row] = (x.mean(), y.mean())
        for row in range(4):
            assert centers["candidate", row][0] < centers["reference", row][0]
        assert [centers["reference", row][1] for row in range(4)] == sorted(
            centers["reference", row][1] for row in range(4)
        )


def test_art_fractions_use_global_item_counts():
    first = dict(record(), art_items=4, art_kept=1, art_masked=1)
    second = dict(record(), art_items=12, art_kept=6, art_masked=3)
    metrics = update_metrics([first, second])
    assert metrics["train/art_inactive_fraction"] == 5 / 16
    assert metrics["train/art_rejected_fraction"] == 4 / 16


def test_execution_time_excludes_resume_downtime_and_evaluation_is_separate(
    tmp_path, sdk, monkeypatch
):
    from src.posttrain import tracking

    now = [0.0]
    monkeypatch.setattr(tracking.time, "monotonic", lambda: now[0])
    with Tracker(tmp_path, "run") as tracker:
        now[0] = 2
        tracker.runtime({})
        now[0] = 5
    now[0] = 100
    with Tracker(tmp_path, None, phase="evaluation") as tracker:
        now[0] = 107
        tracker.runtime({})
        assert tracker.state["execution_seconds"] == 12
        assert tracker.state["evaluation_seconds"] == 7
        now[0] = 110
    state = json.loads((tmp_path / "telemetry.json").read_text())
    assert state["execution_seconds"] == 15
    assert state["evaluation_seconds"] == 10


def test_lower_is_better_gain_and_interval_are_oriented_and_baseline_is_logged_once(
    tmp_path,
    sdk,
    sampling_config,
):
    from src.posttrain.scoring.contracts import Channel

    config = replace(
        sampling_config,
        scorers=(
            replace(sampling_config.scorers[0], outputs=(Channel("fit", "lower"),)),
        ),
    )
    report = {
        "suite": "development",
        "diversity": {
            "arms": {
                "reference": {"mean_by_scene": 0.5},
                "candidate": {"mean_by_scene": 0.4},
            }
        },
        "arms": {
            arm: {
                "unfamiliar.fit": {
                    "complete": True,
                    "scored": 2,
                    "expected": 2,
                    "mean_on_successes": value,
                }
            }
            for arm, value in (("reference", 5), ("candidate", 3))
        },
        "paired": {
            "candidate": {
                "unfamiliar.fit": {
                    "coverage_complete": True,
                    "mean_delta": -2,
                    "interval": [-3, -1],
                }
            }
        },
    }
    with Tracker(tmp_path, "run", phase="evaluation") as tracker:
        tracker.evaluation(report, tmp_path, config, 2)
    with Tracker(tmp_path, None, phase="evaluation") as tracker:
        tracker.evaluation(report, tmp_path, config, 4)
    evaluations = [
        metrics
        for _, metrics in sdk.logs
        if "evaluation/development/candidate/unfamiliar.fit/mean" in metrics
    ]
    assert evaluations[0]["evaluation/development/reference/unfamiliar.fit/mean"] == 5
    assert "evaluation/development/reference/unfamiliar.fit/mean" not in evaluations[1]
    assert evaluations[0]["monitor/lpips/development/reference"] == 0.5
    assert "monitor/lpips/development/reference" not in evaluations[1]
    assert evaluations[1]["monitor/lpips/development/candidate"] == 0.4
    assert evaluations[1]["evaluation/development/candidate/unfamiliar.fit/delta"] == 2
    assert (
        evaluations[1]["evaluation/development/candidate/unfamiliar.fit/delta_lower"]
        == 1
    )
    assert (
        evaluations[1]["evaluation/development/candidate/unfamiliar.fit/delta_upper"]
        == 3
    )


def test_failed_scoring_publishes_retries_and_failures_before_round_aborts(
    tmp_path, sdk, training_config
):
    from PIL import Image

    from src.posttrain.records import file_hash
    from src.posttrain.round_scoring import acquire_round

    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)

    class FailedClient:
        def acquire(self, requests):
            return [
                {
                    "scorer": "unfamiliar",
                    "sample_id": r.sample_id,
                    "status": "failed",
                    "attempts": 2,
                    "values": {},
                }
                for r in requests
            ]

    records = [
        {
            "sample_id": f"0:{slot}:{draw}",
            "prompt": {"id": str(slot), "text": "caption"},
            "image_path": str(path),
            "image_sha256": file_hash(path),
        }
        for slot in (0, 1)
        for draw in (0, 1)
    ]
    with (
        pytest.raises(ValueError, match="required score failed"),
        Tracker(tmp_path, "run") as tracker,
    ):
        acquire_round(
            training_config,
            {"unfamiliar": FailedClient()},
            records,
            on_scored=tracker.scoring,
        )
    values = [
        metrics
        for _, metrics in sdk.logs
        if "runtime/scoring/unfamiliar/failed_requests" in metrics
    ]
    assert values[-1]["runtime/scoring/unfamiliar/failed_requests"] == 4
    assert values[-1]["runtime/scoring/unfamiliar/retry_attempts"] == 4
    assert not any("train/art_loss" in metrics for _, metrics in sdk.logs)
    assert sdk.finishes == [{"state": "crashed"}]


def test_scoring_logs_counter_changes_including_next_collection_reset(tmp_path, sdk):
    with Tracker(tmp_path, "run") as tracker:
        for retries in (0, 0, 1, 1, 0):
            tracker.scoring(
                {"unfamiliar": {"retry_attempts": retries, "failed_requests": 0}}
            )
    assert [
        values["runtime/scoring/unfamiliar/retry_attempts"]
        for _, values in sdk.logs
        if "runtime/scoring/unfamiliar/retry_attempts" in values
    ] == [0, 1, 0]


def test_checkpoint_carries_measured_time_to_a_new_output_directory(
    tmp_path, sdk, monkeypatch
):
    from src.posttrain import tracking

    now = [0.0]
    monkeypatch.setattr(tracking.time, "monotonic", lambda: now[0])
    root = tmp_path / "source"
    with Tracker(root, "run") as tracker:
        now[0] = 5
        tracker.runtime({})
        checkpoint = publish(
            root / "checkpoints",
            1,
            {},
            {"step": 1, "committed_rounds": 1},
            {},
            0,
            torch.device("cpu"),
            json.loads((root / "tracking.json").read_text()),
            keep_last=0,
        )
    now[0] = 100
    with Tracker(tmp_path / "restored", None, resume=checkpoint) as tracker:
        assert tracker.state["execution_seconds"] == 5
        now[0] = 103
        tracker.runtime({})
        assert tracker.state["execution_seconds"] == 8
