from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from PIL import Image
from test_posttrain_nft import new_roles, probabilities, round_rollouts
from test_posttrain_scoring import Backend, Endpoint

from src.evaluation.benchmarks import run as benchmark
from src.models.inko import Inko
from src.posttrain.checkpoint import (
    assert_exact,
    inspect,
    load,
    publish,
    restore_rng,
    rng_state,
)
from src.posttrain.clock import Clock
from src.posttrain.engine import update_round
from src.posttrain.records import file_hash
from src.posttrain.rollout import PromptCursor
from src.posttrain.scoring.client import ScoreClient, request_for_record
from src.posttrain.session import TrainingState


def test_committed_round_state_round_trips_through_a_checkpoint(
    tmp_path, training_config, small_model
):
    torch.set_num_threads(1)
    torch.manual_seed(21)
    config = training_config
    base = small_model()
    roles = new_roles(config, base)
    state = TrainingState(config, roles, None, Clock())
    rollouts = round_rollouts(config, roles, base)
    state.last_reward = update_round(
        config,
        roles,
        roles.student_field,
        rollouts,
        probabilities(rollouts),
        state.student_optimizer,
        0,
        state.clock.old_policy_version,
    )
    state.clock.update(config)
    with pytest.raises(ValueError, match="clocks differ"):
        state.state_dict()
    state.commit_round()
    point = publish(
        tmp_path / "checkpoints",
        state.clock.step,
        {"case": "nft"},
        state.state_dict(),
        {"cursor": 7},
        0,
        torch.device("cpu"),
    )
    committed, rank_state = load(point, {"case": "nft"}, 0)
    assert rank_state["cursor"] == 7 and inspect(point)["step"] == 1
    recovered = TrainingState(config, new_roles(config, base), None, Clock())
    recovered.load_state_dict(committed)
    assert_exact(state.state_dict(), recovered.state_dict())
    with pytest.raises(ValueError, match="clocks differ"):
        recovered.load_state_dict(dict(committed, old_policy_version=9))


def test_checkpoint_publication_is_all_or_nothing(tmp_path, monkeypatch):
    import src.posttrain.checkpoint as module

    def fail(*args, **kwargs):
        raise OSError("publication fault")

    monkeypatch.setattr(module, "atomic_json", fail)
    with pytest.raises(OSError):
        publish(tmp_path, 1, {}, {}, {}, 0, torch.device("cpu"))
    assert not (tmp_path / "step-000001").exists()


def test_manifest_in_staging_is_not_a_published_checkpoint(tmp_path):
    device = torch.device("cpu")
    committed = publish(tmp_path, 1, {"case": "publication"}, {}, {}, 0, device)
    with pytest.raises(ValueError, match="overwrite"):
        publish(tmp_path, 1, {"case": "publication"}, {}, {}, 0, device)
    staging = next(tmp_path.glob(".incomplete-*"))
    assert (staging / "manifest.json").exists()
    with pytest.raises(ValueError, match="unpublished"):
        inspect(staging)
    assert inspect(committed)["step"] == 1


def test_prompt_cursor_and_rank_rng_restore_exactly(tmp_path):
    cursor = PromptCursor(5, 123)
    cursor.take(4)
    state = cursor.state_dict()
    expected = cursor.take(7)
    cursor.load_state_dict(state)
    assert cursor.take(7) == expected
    device = torch.device("cpu")
    torch.manual_seed(5)
    saved = rng_state(device)
    first = [torch.rand(3) for _ in range(2)]
    restore_rng(saved, device)
    assert all(
        torch.equal(a, b)
        for a, b in zip(first, [torch.rand(3) for _ in range(2)], strict=True)
    )


@pytest.mark.parametrize(
    "mode,view", [("full", "live"), ("full", "ema"), ("lora", "live"), ("lora", "ema")]
)
def test_export_materializes_the_requested_weight_view(
    tmp_path, training_config, small_model, mode, view
):
    from dataclasses import asdict as to_dict

    from scripts.posttrain.export import export
    from src.posttrain.adapters import merge_student

    config = replace(
        training_config, student=replace(training_config.student, mode=mode)
    )
    roles = new_roles(config, small_model())
    with torch.no_grad():
        for parameter in roles.student.parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)
    architecture = roles.student.get_config()
    path = publish(
        tmp_path / "checkpoints",
        2,
        {"config": to_dict(config)},
        {
            "student": roles.student.state_dict(),
            "ema": roles.ema,
            "architecture": architecture,
            "step": 2,
        },
        {},
        0,
        torch.device("cpu"),
    )
    export(path, view, tmp_path / "export")
    model = Inko.from_single_file(str(tmp_path / "export" / "weights.pt"))
    if view == "ema":
        with torch.no_grad():
            for name, parameter in roles.student.named_parameters():
                if name in roles.ema["ema"]:
                    parameter.copy_(roles.ema["ema"][name])
    assert_exact(merge_student(roles.student).state_dict(), model.state_dict())


def test_config_only_external_service_collection_join_report_and_resume(
    tmp_path, monkeypatch, sampling_config
):
    config = sampling_config
    spec = config.scorers[0]
    endpoint = Endpoint(tmp_path, spec, Backend())
    artifacts = {"checkpoints": {}, "text_encoder": "encoder", "vae": "vae"}
    monkeypatch.setattr(benchmark, "verify_assets", lambda *args: artifacts)
    prompts = [
        {
            "id": "p",
            "scene_id": "p",
            "aspect": "1:1",
            "text": "text",
            "seed": 4,
        }
    ]

    class Workers:
        generated = 0

        def __init__(self, config, assets, output, **kwargs):
            self.messages, self.startup = [], []
            self.client = ScoreClient(
                config.scorers[0],
                assets["services"][spec.id]["endpoints"],
                output / "acquisition" / spec.id,
            )

        def __enter__(self):
            self.client.handshake()
            return self

        def __exit__(self, *args):
            pass

        def submit(self, role, task):
            if role == "learner":
                Workers.generated += 1
                directory = tmp_path / task["arm"].name
                directory.mkdir(exist_ok=True)
                image, latent = directory / "image.png", directory / "latent.pt"
                Image.new("RGB", (8, 8), (20, 100, 200)).save(image)
                latent.write_bytes(b"detached latent fixture")
                result = {
                    "records": [
                        {
                            "sample_id": task["sample_ids"][0],
                            "prompt": prompts[0],
                            "arm": task["arm"].name,
                            "image_path": str(image),
                            "image_sha256": file_hash(image),
                            "latent_path": str(latent),
                            "latent_sha256": file_hash(latent),
                        }
                    ],
                    "timings": {},
                }
            else:
                result = {
                    "scores": self.client.acquire(
                        [
                            request_for_record(r, self.client.spec)
                            for r in task["records"]
                        ]
                    )
                }
            self.messages.append(
                {
                    "event": "result",
                    "task_id": task["task_id"],
                    "role": role,
                    "result": result,
                }
            )

        def receive(self, deadline):
            return self.messages.pop()

    monkeypatch.setattr(benchmark, "ResidentWorkers", Workers)
    assets = {"services": {spec.id: {"endpoints": [endpoint.url]}}}
    try:
        first = benchmark.run(config, assets, prompts, tmp_path / "run")
        second = benchmark.run(config, assets, prompts, tmp_path / "run")
        assert first == second and Workers.generated == 2
        assert first["arms"]["reference"]["unfamiliar.fit"]["mean_on_successes"] == 0.75
        assert first["paired"]["candidate"]["unfamiliar.fit"]["mean_delta"] == 0.0
        # Remove only the caller's final result; durable client results survive service restart.
        next((tmp_path / "run/scores/unfamiliar").glob("*.json")).unlink()
        assert benchmark.run(config, assets, prompts, tmp_path / "run") == first
        assert Workers.generated == 2
        with pytest.raises(ValueError, match="identity changed"):
            benchmark.run(
                config, assets, [dict(prompts[0], text="another")], tmp_path / "run"
            )
        artifacts["generation_runtime"] = {"example": "changed runtime"}
        with pytest.raises(ValueError, match="identity changed"):
            benchmark.run(config, assets, prompts, tmp_path / "run")
        del artifacts["generation_runtime"]
        (tmp_path / "reference/image.png").write_bytes(b"changed bytes")
        with pytest.raises(ValueError, match="image bytes changed"):
            benchmark.run(config, assets, prompts, tmp_path / "run")
    finally:
        endpoint.close()


def test_arm_checkpoint_is_loaded_and_cached(tmp_path, monkeypatch):
    import torch

    from src.models.inko import Inko
    from src.posttrain.config import Checkpoint
    from src.posttrain.models import Generator

    calls = []

    def load(path):
        calls.append(path)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(float(Path(path).read_text()))
        return model

    monkeypatch.setattr(Inko, "from_single_file", load)
    worker = Generator.__new__(Generator)
    worker.device, worker.dtype, worker.checkpoint = (
        torch.device("cpu"),
        torch.float32,
        None,
    )
    worker.assets = {"checkpoints": {}}
    checkpoints = []
    for asset, value, weights in (("first", "2", "ema"), ("trained", "7", "live")):
        directory = tmp_path / asset
        directory.mkdir()
        path = directory / (
            "ema_weights.pt" if weights == "ema" else "model.safetensors"
        )
        path.write_text(value)
        worker.assets["checkpoints"][asset] = str(path)
        checkpoints.append(Checkpoint(asset, weights))
    worker.select_checkpoint(checkpoints[0])
    assert worker.model(torch.ones(1, 1)).item() == 2
    worker.select_checkpoint(checkpoints[1])
    assert worker.model(torch.ones(1, 1)).item() == 7
    worker.select_checkpoint(checkpoints[1])
    assert len(calls) == 2


def test_strict_typed_configs_reject_unknown_and_invalid_fields(sampling_config):
    from src.posttrain.config import Config
    from src.posttrain.schema import decode

    raw = asdict(sampling_config)
    raw["execution"]["hpsv3_devices"] = [1]
    with pytest.raises(ValueError, match="unknown"):
        decode(Config, raw)
    del raw["execution"]["hpsv3_devices"]
    raw["sampling"]["batch_size"] = True
    with pytest.raises(ValueError, match="int"):
        decode(Config, raw)


def test_generation_runtime_uses_content_not_location(tmp_path, monkeypatch):
    from src.posttrain import runtime

    class Distribution:
        version = "1.2"
        files = (Path("module.py"),)

        def locate_file(self, entry):
            return root[0] / entry

    first, second = tmp_path / "first", tmp_path / "second"
    for path in (first, second):
        path.mkdir()
        (path / "module.py").write_text("original implementation")
    root = [first]
    monkeypatch.setattr(runtime, "GENERATION_PACKAGES", ("example",))
    monkeypatch.setattr(
        runtime.importlib.metadata, "distribution", lambda _: Distribution()
    )
    before = runtime.generation_runtime()
    root[0] = second
    assert runtime.generation_runtime() == before
    (second / "module.py").write_text("changed implementation")
    assert runtime.generation_runtime() != before
    (second / "module.py").write_text("original implementation")
    Distribution.version = "1.3"
    assert runtime.generation_runtime() != before


def test_native_generation_runtime_is_required_and_content_addressed(
    tmp_path, monkeypatch, sampling_config
):
    import json
    from types import SimpleNamespace

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"checkpoint fixture")
    checkpoint = sampling_config.arms[0].checkpoint
    config = replace(
        sampling_config, arms=(replace(sampling_config.arms[0], checkpoint=checkpoint),)
    )
    assets = {
        "device_type": "npu",
        "learner": {"python_paths": []},
        "checkpoints": {checkpoint.asset: str(weights)},
        "text_encoder": str(tmp_path),
        "vae": str(tmp_path),
    }
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=json.dumps({"example": "runtime"})),
    )
    monkeypatch.setattr(benchmark, "tree_identity", lambda _: "fixed model contents")
    with pytest.raises(ValueError, match="native runtime inventory"):
        benchmark.verify_assets(config, assets)
    first, second = tmp_path / "version.info", tmp_path / "relocated.info"
    first.write_text("native 1")
    second.write_text("native 1")
    assets["learner"]["runtime_files"] = {"compiler": str(first)}
    before = benchmark.verify_assets(config, assets)
    assets["learner"]["runtime_files"]["compiler"] = str(second)
    assert benchmark.verify_assets(config, assets) == before
    second.write_text("native 2")
    assert benchmark.verify_assets(config, assets) != before
