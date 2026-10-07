"""Pinned VQAScore v3.1; resident Qwen3-VL loading only."""

import importlib
import sys

import torch


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
            model_name=spec.model, device=str(device), checkpoint=assets["weights"]
        )
        self.kwargs = {
            k: getattr(spec, k)
            for k in (
                "question_template",
                "answer_template",
                "max_new_tokens",
                "temperature",
            )
        }
        tokens = self.scorer.processor.tokenizer.encode(
            spec.answer_template, add_special_tokens=False
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
