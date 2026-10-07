from dataclasses import asdict, replace
from pathlib import Path

import pytest
from PIL import Image
from test_posttrain_scoring import Backend, Endpoint

from src.evaluation.benchmarks import run as benchmark
from src.posttrain.records import file_hash
from src.posttrain.scoring.client import ScoreClient, request_for_record


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
            "structured": {"rubric": "identify content"},
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


def test_arm_checkpoint_is_loaded_and_verified(tmp_path, monkeypatch):
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
        architecture = directory / "transformer_config.json"
        architecture.write_text("{}")
        worker.assets["checkpoints"][asset] = str(path)
        checkpoints.append(
            Checkpoint(asset, weights, file_hash(path), file_hash(architecture))
        )
    worker.select_checkpoint(checkpoints[0])
    assert worker.model(torch.ones(1, 1)).item() == 2
    worker.select_checkpoint(checkpoints[1])
    assert worker.model(torch.ones(1, 1)).item() == 7
    worker.select_checkpoint(checkpoints[1])
    assert len(calls) == 2
    with pytest.raises(ValueError, match="bytes changed"):
        worker.select_checkpoint(replace(checkpoints[0], sha256="0" * 64))


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

    weights, architecture = tmp_path / "model.pt", tmp_path / "transformer_config.json"
    weights.write_bytes(b"checkpoint fixture")
    architecture.write_text("{}")
    checkpoint = replace(
        sampling_config.arms[0].checkpoint,
        sha256=file_hash(weights),
        architecture_sha256=file_hash(architecture),
    )
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
