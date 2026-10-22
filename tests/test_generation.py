import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src.evaluation.generation import run as generation
from src.evaluation.generation.config import Config
from src.evaluation.records import load_catalog, read_jsonl, sample_key


def test_jsonl_repairs_only_an_interrupted_last_record(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_bytes(b'{"draw": 0}\n{"draw":')
    assert read_jsonl(path, repair_tail=True) == [{"draw": 0}]
    assert path.read_bytes() == b'{"draw": 0}\n'
    path.write_bytes(b'{"draw": 1}')
    assert read_jsonl(path, repair_tail=True) == [{"draw": 1}]
    assert path.read_bytes().endswith(b"\n")
    path.write_bytes(b'broken\n{"draw": 2}\n')
    with pytest.raises(ValueError):
        read_jsonl(path, repair_tail=True)


def test_catalog_variants_and_native_ids(tmp_path):
    path = tmp_path / "prompts.jsonl"
    row = {
        "benchmark": "suite",
        "split": "test",
        "prompt_id": "scene/中文",
        "variant": "short",
        "text": "一幅画",
        "metadata": {"aspect": "1:1"},
        "seeds": [13, 29],
    }
    rows = [row, {**row, "variant": "long"}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert load_catalog(path) == rows
    path.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="unique"):
        load_catalog(path)


def test_generation_recovers_images_and_appends_another_model(
    tmp_path, monkeypatch, sampling_config
):
    config = Config(
        sampling_config.sampling, sampling_config.execution, sampling_config.arms
    )
    assets = {
        "checkpoints": {"first": "weights.pt"},
        "text_encoder": "text",
        "vae": "vae",
        "device_type": "cpu",
    }
    catalog = [
        {
            "benchmark": "suite",
            "split": "test",
            "prompt_id": "scene/1",
            "variant": "en/short",
            "text": "a painting",
            "metadata": {"aspect": "1:1"},
            "seeds": [13, 29, 47],
        }
    ]

    class Workers:
        generated = []
        interrupt = True

        def __init__(self, sampling, execution, assets):
            self.messages, self.startup = [], []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, task):
            for prompt, row, path in zip(
                task["prompts"], task["records"], task["image_paths"], strict=True
            ):
                assert prompt["seed"] == row["seed"]
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (row["seed"], 0, 0)).save(path)
                self.generated.append((row["model"], row["seed"]))
            self.messages.append(
                {
                    "event": "result",
                    "task_id": task["task_id"],
                    "result": {"records": task["records"], "timings": {}},
                }
            )

        def receive(self, deadline):
            if self.interrupt:
                raise RuntimeError("interrupted after PNG publish")
            return self.messages.pop()

    monkeypatch.setattr(generation, "ResidentWorkers", Workers)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted"):
        generation.run(config, assets, catalog, "reference", output)
    Workers.interrupt = False
    second = generation.run(config, assets, catalog, "reference", output)
    assert second["generated"] == 1 and second["recovered_images"] == 2
    assert Workers.generated == [
        ("reference", 13),
        ("reference", 29),
        ("reference", 47),
    ]
    journal = next((output / "reference").rglob("samples.jsonl"))
    records = read_jsonl(journal)
    assert len({sample_key(r) for r in records}) == 3
    assert {r["draw"]: r["seed"] for r in records} == {0: 13, 1: 29, 2: 47}
    with journal.open("ab") as stream:
        stream.write(b'{"model":')
    changed_batch = replace(config, sampling=replace(config.sampling, batch_size=3))
    assert (
        generation.run(changed_batch, assets, catalog, "reference", output)["generated"]
        == 0
    )
    generation.run(config, assets, catalog, "candidate", output)
    assert read_jsonl(journal) == records
    # A published record cannot hide a missing or corrupt image.
    (output / records[0]["image_path"]).write_bytes(b"broken PNG")
    assert (
        generation.run(config, assets, catalog, "reference", output)["generated"] == 1
    )
    assert len(read_jsonl(journal)) == 3
    with pytest.raises(ValueError, match="settings changed"):
        generation.run(
            config, {**assets, "vae": "different"}, catalog, "reference", output
        )


@pytest.mark.parametrize(
    "pipeline",
    [
        "DreamLitePipeline",
        "MOSSImagePipeline",
        "PixArtSigmaPipeline",
        "StableDiffusionPipeline",
    ],
)
def test_external_call_preserves_text_seed_and_record_order(
    tmp_path, monkeypatch, pipeline
):
    from types import SimpleNamespace

    import torch

    from src.evaluation.generation import external
    from src.evaluation.generation.config import ExternalModel

    model = ExternalModel(
        "external",
        pipeline,
        "weights",
        "float32",
        1,
        {"portrait": (48, 32)},
        {"num_inference_steps": 13},
    )
    calls = []

    class Pipe:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (32, 48))])

    monkeypatch.setattr(external, "load_pipeline", lambda *args: Pipe())
    monkeypatch.setattr(external, "conditioning_lengths", lambda *args: [{"tokens": 7}])
    backend = external.Generator(model, {"path": "unused"}, torch.device("cpu"))
    task = {
        "prompts": [
            {"text": 'Draw "Tea".\n保留原文。', "seed": 37, "aspect": "portrait"}
        ],
        "records": [{"sample_id": "native-id", "draw": 2}],
        "image_paths": [str(tmp_path / "image.png")],
    }
    result = backend.generate(task)
    assert calls[0]["prompt"] == (
        'Draw "Tea".\n保留原文。'
        if pipeline == "DreamLitePipeline"
        else ['Draw "Tea".\n保留原文。']
    )
    rng = calls[0]["generator"]
    assert (rng if pipeline == "DreamLitePipeline" else rng[0]).initial_seed() == 37
    assert (calls[0]["height"], calls[0]["width"], calls[0]["num_inference_steps"]) == (
        48,
        32,
        13,
    )
    assert result["records"] == [
        {"sample_id": "native-id", "draw": 2, "conditioning": {"tokens": 7}}
    ]
    assert Image.open(tmp_path / "image.png").size == (32, 48)


def test_external_run_uses_own_assets_shapes_and_batch(
    tmp_path, monkeypatch, sampling_config
):
    from src.evaluation.generation.config import ExternalModel

    model = ExternalModel(
        "external",
        "PixArtSigmaPipeline",
        "weights",
        "float32",
        2,
        {"portrait": (48, 32)},
        {"num_inference_steps": 13},
    )
    config = Config(sampling_config.sampling, sampling_config.execution, (model,))
    assets = {"models": {"weights": {"path": "pipeline"}}, "device_type": "cpu"}
    catalog = [
        {
            "benchmark": "suite",
            "split": "test",
            "prompt_id": "id/9",
            "variant": "en",
            "text": "Unchanged",
            "metadata": {"aspect": "portrait"},
            "seeds": [37, 53],
        }
    ]

    class Workers:
        def __init__(self, sampling, execution, selected):
            assert selected["external_model"] == model
            self.startup = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, task):
            assert [p["seed"] for p in task["prompts"]] == [37, 53]
            for row, path in zip(task["records"], task["image_paths"], strict=True):
                assert (row["height"], row["width"]) == (48, 32)
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 48)).save(path)
            self.message = {
                "event": "result",
                "task_id": task["task_id"],
                "result": {"records": task["records"], "timings": {}},
            }

        def receive(self, deadline):
            return self.message

    monkeypatch.setattr(generation, "ResidentWorkers", Workers)
    summary = generation.run(config, assets, catalog, "external", tmp_path / "run")
    assert (summary["samples"], summary["generated"], summary["batch_size"]) == (
        2,
        2,
        2,
    )
    records = read_jsonl(next((tmp_path / "run").rglob("samples.jsonl")))
    assert [(r["prompt_id"], r["draw"], r["seed"]) for r in records] == [
        ("id/9", 0, 37),
        ("id/9", 1, 53),
    ]
    resumed = replace(config, models=(replace(model, batch_size=1),))
    assert (
        generation.run(resumed, assets, catalog, "external", tmp_path / "run")[
            "generated"
        ]
        == 0
    )


def test_cli_uses_selected_environment_even_with_same_python_binary(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    from scripts.eval import run as cli
    from src.evaluation.generation import config as config_module
    from src.evaluation.generation.config import ExternalModel

    python = tmp_path / "isolated" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    assets = tmp_path / "assets.toml"
    assets.write_text(
        f'[models.weights]\npath = "pipeline"\npython = "{python}"\n'
        '[models.weights.env]\nHF_HOME = "model-cache"\n'
    )
    model = ExternalModel(
        "external",
        "DreamLitePipeline",
        "weights",
        "float32",
        1,
        {"1:1": (32, 32)},
        {"num_inference_steps": 13},
    )
    monkeypatch.setattr(
        config_module, "load_config", lambda _: SimpleNamespace(models=(model,))
    )
    arguments = [
        "generate",
        "--config",
        "unused.toml",
        "--assets",
        str(assets),
        "--prompts",
        "unused.jsonl",
        "--model",
        "external",
        "--output",
        "unused",
    ]
    monkeypatch.setattr(sys, "argv", ["run.py", *arguments])
    calls = []
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    cli.main()
    assert calls[0][0][0] == [str(python), "-m", "scripts.eval.run", *arguments]
    assert calls[0][1]["env"]["HF_HOME"] == "model-cache"
    assert calls[0][1]["check"] is True


def test_external_details_preserve_infinite_scheduler_bounds():
    from types import SimpleNamespace

    import torch

    from src.evaluation.generation.external import Generator

    backend = Generator.__new__(Generator)
    backend.model = SimpleNamespace(pipeline="example")
    backend.pipe = SimpleNamespace(
        components={"transformer": torch.nn.Linear(2, 3, bias=False)},
        scheduler=SimpleNamespace(config={"lower_bound": float("-inf"), "order": 2}),
    )
    result = backend.details()
    assert result["backbone_parameters"] == 6
    assert result["scheduler"]["config"] == {"lower_bound": "-Infinity", "order": 2}
    json.dumps(result, allow_nan=False)


def test_generation_serializes_each_model_writer_independently(
    tmp_path, monkeypatch, sampling_config
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    config = Config(
        sampling_config.sampling, sampling_config.execution, sampling_config.arms
    )
    entered, release = Event(), Event()

    def generate(config, assets, catalog, arm, output):
        if arm.name == "reference":
            entered.set()
            assert release.wait(timeout=2)
        return arm.name

    monkeypatch.setattr(generation, "_run", generate)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(generation.run, config, {}, [], "reference", tmp_path)
        try:
            assert entered.wait(timeout=2)
            assert generation.run(config, {}, [], "candidate", tmp_path) == "candidate"
            with pytest.raises(BlockingIOError):
                generation.run(config, {}, [], "reference", tmp_path)
        finally:
            release.set()
        assert first.result() == "reference"
