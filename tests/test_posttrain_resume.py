from dataclasses import asdict

import pytest

from src.evaluation.benchmarks import run as benchmark
from src.posttrain.records import ScoreRequest, ScoreResult, file_hash


def test_resume_reuses_verified_samples_and_scores_and_rejects_changed_inputs(
    tmp_path, monkeypatch
):
    config = {
        "arms": [{"name": "control"}],
        "sampling": {"batch_size": 1},
        "scorers": {"s": {"primary_output": "value"}},
        "evaluation": {
            "bootstrap_repeats": 10,
            "bootstrap_seed": 0,
            "confidence": 0.95,
        },
    }
    prompts = [{"id": "p", "scene_id": "p", "aspect": "1:1", "text": "text", "seed": 4}]
    artifacts = {
        "checkpoint": "weights",
        "architecture": {},
        "text_encoder": {},
        "vae": {},
        "s": {},
        "runtime": {"s": []},
        "base_packages": {},
    }
    monkeypatch.setattr(benchmark, "verify_assets", lambda *args: artifacts)

    class Workers:
        calls = 0

        def __init__(self, config, assets, identities, roles=None):
            self.identities = identities
            self.startup = []
            Workers.calls += 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def pipeline(tasks, workers, config, generated, scored):
        for task in tasks:
            image, latent = tmp_path / "image", tmp_path / "latent"
            image.write_bytes(b"image fixture")
            latent.write_bytes(b"latent fixture")
            sample_id = task["sample_ids"][0]
            record = {
                "sample_id": sample_id,
                "prompt": prompts[0],
                "arm": "control",
                "image_path": str(image),
                "image_sha256": file_hash(image),
                "latent_path": str(latent),
                "latent_sha256": file_hash(latent),
            }
            generated([record])
            request = ScoreRequest(sample_id, str(image), file_hash(image), "text")
            result = ScoreResult(
                sample_id,
                request.key(workers.identities["s"]),
                "s",
                workers.identities["s"],
                "ok",
                {"value": 0.3},
                1,
                None,
            )
            scored("s", [asdict(result)])
        return []

    monkeypatch.setattr(benchmark, "ResidentWorkers", Workers)
    monkeypatch.setattr(benchmark, "run_pipeline", pipeline)
    first = benchmark.run(config, {}, prompts, tmp_path / "run")
    second = benchmark.run(config, {}, prompts, tmp_path / "run")
    assert first == second and Workers.calls == 1
    changed = [dict(prompts[0], text="another input")]
    with pytest.raises(ValueError, match="identity changed"):
        benchmark.run(config, {}, changed, tmp_path / "run")
    (tmp_path / "image").write_bytes(b"modified bytes")
    with pytest.raises(ValueError, match="image bytes changed"):
        benchmark.run(config, {}, prompts, tmp_path / "run")
