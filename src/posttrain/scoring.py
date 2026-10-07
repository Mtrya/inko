"""Frozen scorer adapters and failure-aware acquisition; no reward transforms."""

import importlib
import sys
import tempfile
from pathlib import Path

import torch

from .records import ScoreRequest, ScoreResult, file_hash


def float32_head_inputs(module, inputs):
    """HPSv3 specifies FP32 head weights after its BF16 transformer."""
    return (inputs[0].float(), *inputs[1:])


class HPSv3:
    def __init__(self, spec, assets, device):
        import yaml

        sys.path.insert(0, assets["source"])
        module = importlib.import_module("hpsv3.inference")
        config_path = Path(assets["source"]) / "hpsv3/config/HPSv3_7B.yaml"
        config = yaml.safe_load(config_path.read_text())
        config.update(
            model_name_or_path=assets["backbone"],
            disable_flash_attn2=True,
            deepspeed=None,
        )
        # The upstream loader consumes a YAML file and adjusts output_dir from it.
        with tempfile.TemporaryDirectory(prefix="inko-hps-") as directory:
            path = Path(directory) / "inference.yaml"
            path.write_text(yaml.safe_dump(config))
            self.scorer = module.HPSv3RewardInferencer(
                config_path=str(path),
                checkpoint_path=assets["weights"],
                device=str(device),
                differentiable=False,
            )
        self.scorer.model.requires_grad_(False)
        # Upstream uses CUDA autocast(dtype=float32), which disables autocast
        # rather than converting BF16 hidden states. Make the head boundary
        # explicit on every device without changing weights or preprocessing.
        self.scorer.model.rm_head.register_forward_pre_hook(float32_head_inputs)

    def score(self, requests):
        values = self.scorer.reward(
            prompts=[r.prompt for r in requests],
            image_paths=[r.image_path for r in requests],
        )
        if values.shape != (len(requests), 2):
            raise ValueError(f"unexpected HPSv3 output shape {values.shape}")
        return [
            {"mu": float(row[0]), "log_sigma": float(row[1])}
            for row in values.detach().cpu()
        ]


class VQAScore:
    def __init__(self, spec, assets, device):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        sys.path.insert(0, assets["source"])
        module = importlib.import_module(
            "t2v_metrics.models.vqascore_models.qwen3vl_model"
        )

        class ResidentQwen(module.Qwen3VLModel):
            def load_model(self):
                # Override placement/loading only; scoring and preprocessing are upstream.
                self.model = (
                    AutoModelForImageTextToText.from_pretrained(
                        assets["weights"],
                        torch_dtype=torch.bfloat16,
                        attn_implementation="sdpa",
                        device_map={"": str(device)},
                        local_files_only=True,
                    )
                    .eval()
                    .requires_grad_(False)
                )
                self.processor = AutoProcessor.from_pretrained(
                    assets["weights"], local_files_only=True
                )
                self.device = device

        self.scorer = ResidentQwen(
            model_name=spec["model"], device=str(device), checkpoint=assets["weights"]
        )
        self.kwargs = {
            k: spec[k]
            for k in (
                "question_template",
                "answer_template",
                "max_new_tokens",
                "temperature",
            )
        }
        tokens = self.scorer.processor.tokenizer.encode(
            spec["answer_template"], add_special_tokens=False
        )
        if len(tokens) != 1:
            raise ValueError("initial VQAScore contract requires a single Yes token")

    def score(self, requests):
        values = self.scorer.forward(
            images=[r.image_path for r in requests],
            texts=[r.prompt for r in requests],
            **self.kwargs,
        )
        if values.shape != (len(requests),):
            raise ValueError("unexpected VQAScore output shape")
        return [{"probability": float(x)} for x in values]


def acquire(
    backend,
    requests: list[ScoreRequest],
    *,
    scorer: str,
    scorer_identity: str,
    attempts: int,
) -> list[ScoreResult]:
    if attempts < 1:
        raise ValueError("at least one score attempt is required")
    # Retries reuse the same immutable image. Never replace a failure by zero.
    for request in requests:
        if file_hash(request.image_path) != request.image_sha256:
            raise ValueError("score image bytes changed")
    error = None
    for attempt in range(1, attempts + 1):
        try:
            with torch.inference_mode():
                values = backend.score(requests)
            if len(values) != len(requests):
                raise ValueError("scorer result count differs from request count")
            return [
                ScoreResult(
                    r.sample_id,
                    r.key(scorer_identity),
                    scorer,
                    scorer_identity,
                    "ok",
                    value,
                    attempt,
                    None,
                )
                for r, value in zip(requests, values, strict=True)
            ]
        except Exception as exc:  # noqa: BLE001 - explicit failed score records are the API
            error = f"{type(exc).__name__}: {exc}"
    return [
        ScoreResult(
            r.sample_id,
            r.key(scorer_identity),
            scorer,
            scorer_identity,
            "failed",
            {},
            attempts,
            error,
        )
        for r in requests
    ]
