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
