"""Released external pipelines, native conditioning and saved-image records."""

import inspect
import json
import time
from pathlib import Path

import torch

from src.utils.device import peak_memory, synchronize


def load_pipeline(model, assets, device):
    import diffusers

    dtype = getattr(torch, model.precision)
    kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if model.pipeline == "MOSSImagePipeline":
        pipe = diffusers.DiffusionPipeline.from_pretrained(
            assets["path"],
            custom_pipeline=str(Path(assets["path"]) / "pipeline_moss_image.py"),
            trust_remote_code=True,
            **{**kwargs, "torch_dtype": {"default": dtype, "vae": torch.float32}},
        )
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    else:
        if model.pipeline == "SanaPipeline":
            kwargs["variant"] = "fp16"
        pipe = getattr(diffusers, model.pipeline).from_pretrained(
            assets["path"], **kwargs
        )
    pipe.to(device)
    if model.pipeline == "SanaPipeline":
        pipe.text_encoder.to(torch.bfloat16)
        pipe.vae.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def conditioning_lengths(pipe, model, texts):
    """Inspect native tokenization without changing the text sent to the pipeline."""
    options = model.parameters
    limit = options.get("max_sequence_length")
    tokenizers = []
    if model.pipeline == "DreamLitePipeline":
        texts = [pipe.prompt_template_encode_generate.format(text) for text in texts]
        limit += pipe.prompt_template_encode_generate_start_idx
        tokenizers = [("text_encoder", pipe.tokenizer, texts, limit)]
    elif model.pipeline in {"SanaPipeline", "PixArtSigmaPipeline"}:
        texts = pipe._text_preprocessing(texts, clean_caption=options["clean_caption"])
        if model.pipeline == "SanaPipeline":
            instruction = (
                inspect.signature(pipe.__call__)
                .parameters["complex_human_instruction"]
                .default
            )
            prefix = "\n".join(instruction)
            texts = [prefix + text for text in texts]
            limit += len(pipe.tokenizer.encode(prefix)) - 2
        tokenizers = [("text_encoder", pipe.tokenizer, texts, limit)]
    elif model.pipeline == "MOSSImagePipeline":
        texts = [
            pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for text in texts
        ]
        tokenizers = [("text_encoder", pipe.tokenizer, texts, limit)]
    else:
        tokenizers = [
            ("text_encoder", pipe.tokenizer, texts, pipe.tokenizer.model_max_length),
            (
                "text_encoder_2",
                pipe.tokenizer_2,
                texts,
                pipe.tokenizer_2.model_max_length,
            ),
            ("text_encoder_3", pipe.tokenizer_3, texts, limit),
        ]
    result = [{} for _ in texts]
    for name, tokenizer, prepared, cap in tokenizers:
        for row, ids in zip(
            result, tokenizer(prepared, truncation=False)["input_ids"], strict=True
        ):
            row[name] = {
                "tokens": len(ids),
                "limit": cap,
                "truncated": cap is not None and len(ids) > cap,
            }
    return result


class Generator:
    def __init__(self, model, assets, device):
        self.model, self.device = model, device
        self.pipe = load_pipeline(model, assets, device)

    def details(self):
        components = {
            name: component
            for name, component in self.pipe.components.items()
            if isinstance(component, torch.nn.Module)
        }
        counts = {
            name: sum(p.numel() for p in component.parameters())
            for name, component in components.items()
        }
        # Keep infinite scheduler bounds explicit in standard JSON telemetry.
        scheduler_config = json.loads(
            json.dumps(dict(self.pipe.scheduler.config)), parse_constant=str
        )
        return {
            "pipeline": self.model.pipeline,
            "component_parameters": counts,
            "backbone_parameters": counts.get("transformer", counts.get("unet")),
            "pipeline_parameters": sum(counts.values()),
            "component_dtypes": {
                name: sorted({str(p.dtype) for p in component.parameters()})
                for name, component in components.items()
            },
            "scheduler": {
                "class": type(self.pipe.scheduler).__name__,
                "config": scheduler_config,
            },
        }

    @torch.no_grad()
    def generate(self, task):
        synchronize(self.device)
        started = time.perf_counter()
        texts = [p["text"] for p in task["prompts"]]
        lengths = conditioning_lengths(self.pipe, self.model, texts)
        token_end = time.perf_counter()
        height, width = self.model.shapes[task["prompts"][0]["aspect"]]
        rng_device = (
            "cpu" if self.model.pipeline == "MOSSImagePipeline" else self.device
        )
        generators = [
            torch.Generator(device=rng_device).manual_seed(p["seed"])
            for p in task["prompts"]
        ]
        if self.model.pipeline == "DreamLitePipeline":
            if len(texts) != 1:
                raise ValueError(
                    "DreamLite's released pipeline accepts one prompt per call"
                )
            prompts, rng = texts[0], generators[0]
        else:
            prompts, rng = texts, generators
        images = self.pipe(
            prompt=prompts,
            height=height,
            width=width,
            generator=rng,
            **self.model.parameters,
        ).images
        synchronize(self.device)
        generation_end = time.perf_counter()
        records = []
        for image, row, path, tokens in zip(
            images, task["records"], task["image_paths"], lengths, strict=True
        ):
            if image.size != (width, height):
                raise ValueError(
                    f"native pipeline returned {image.size}, requested {(width, height)}"
                )
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".png.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(destination)
            records.append({**row, "conditioning": tokens})
        return {
            "records": records,
            "timings": {
                "conditioning_inspection": token_end - started,
                "generation": generation_end - token_end,
                "artifact_write": time.perf_counter() - generation_end,
            },
            "peak_memory_bytes": peak_memory(self.device),
        }
