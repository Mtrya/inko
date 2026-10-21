"""Text-to-image inference with Inko, Qwen3 conditioning and the Qwen-Image VAE."""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from .flow.solvers import sample_ode
from .models import Inko
from .models.dit_blocks import set_real_rope
from .utils.encode_text import encode_text
from .utils.vae_codec import decode_latents, vae_stats_from_config


@dataclass
class InkoPipelineOutput:
    images: list[Image.Image]


class InkoPipeline:
    """Load a released checkpoint and generate images from English or Chinese text.

    The release's ``pipeline_config.json`` identifies its frozen text encoder,
    VAE and sampling defaults. Solver state and time features stay in FP32;
    ``torch_dtype`` controls the model forwards and VAE decode.
    """

    def __init__(self, transformer, text_encoder, tokenizer, vae, *, config):
        self.transformer = transformer.eval().requires_grad_(False)
        self.text_encoder = text_encoder.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.vae = vae.eval().requires_grad_(False)
        self.config = config
        self.vae_mean, self.vae_std = vae_stats_from_config(vae.config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        torch_dtype: torch.dtype = torch.float32,
        revision: str | None = None,
        local_files_only: bool = False,
        text_encoder_path: str | Path | None = None,
        vae_path: str | Path | None = None,
    ) -> "InkoPipeline":
        """Load a Hub release or a local export directory.

        ``text_encoder_path`` and ``vae_path`` can point to existing local
        copies of the frozen models. ``revision`` selects the Inko Hub revision.
        """
        directory = Path(pretrained_model_name_or_path)
        if not directory.is_dir():
            directory = Path(
                snapshot_download(
                    str(pretrained_model_name_or_path),
                    revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=[
                        "model.safetensors",
                        "transformer_config.json",
                        "pipeline_config.json",
                    ],
                )
            )
        config = json.loads((directory / "pipeline_config.json").read_text())
        text_source = str(text_encoder_path or config["text_encoder"])
        vae_source = str(vae_path or config["vae"])
        transformer = Inko.from_single_file(str(directory / "model.safetensors"))
        text_encoder = AutoModelForCausalLM.from_pretrained(
            text_source,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            local_files_only=local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            text_source, local_files_only=local_files_only
        )
        vae = AutoencoderKLQwenImage.from_pretrained(
            vae_source, torch_dtype=torch_dtype, local_files_only=local_files_only
        )
        return cls(
            transformer.to(dtype=torch_dtype),
            text_encoder,
            tokenizer,
            vae,
            config=config,
        )

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    def to(self, device, dtype: torch.dtype | None = None) -> "InkoPipeline":
        """Move all three models together; latent statistics retain FP32."""
        device = torch.device(device)
        for model in (self.transformer, self.text_encoder, self.vae):
            model.to(device=device, dtype=dtype)
        self.vae_mean = self.vae_mean.to(device)
        self.vae_std = self.vae_std.to(device)
        set_real_rope(self.transformer, device.type == "npu")
        return self

    def _encode(self, prompts):
        return encode_text(
            prompts,
            self.text_encoder,
            self.tokenizer,
            pooling=True,
            exit_layer=self.config["text_exit_layer"],
            exit_mode="stop_at_layer",
            fast_slice=True,
        )

    def _predict(self, x, t, conditioning):
        text, mask, pooled = conditioning
        with torch.autocast(
            self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            return self.transformer(x.to(self.dtype), t, text, pooled, mask).float()

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str],
        *,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | list[str] = "",
        generator: torch.Generator | list[torch.Generator] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> InkoPipelineOutput:
        """Generate one image per prompt, in input order.

        Dimensions must be multiples of 16. A list of generators gives each
        image its own random stream. CPU generators make initial noise
        independent of the inference device. CFG 1 uses one conditional forward
        per step; other values also evaluate ``negative_prompt``.
        """
        prompts = [prompt] if isinstance(prompt, str) else prompt
        if (
            not isinstance(prompts, list)
            or not prompts
            or not all(isinstance(text, str) for text in prompts)
        ):
            raise ValueError("prompt must be a string or a nonempty list of strings")
        height = self.config["height"] if height is None else height
        width = self.config["width"] if width is None else width
        steps = (
            self.config["num_inference_steps"]
            if num_inference_steps is None
            else num_inference_steps
        )
        guidance = (
            self.config["guidance_scale"] if guidance_scale is None else guidance_scale
        )
        if any(type(n) is not int or n < 16 or n % 16 for n in (height, width)):
            raise ValueError("height and width must be positive multiples of 16")
        if type(steps) is not int or steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        negatives = (
            [negative_prompt] * len(prompts)
            if isinstance(negative_prompt, str)
            else negative_prompt
        )
        if (
            not isinstance(negatives, list)
            or len(negatives) != len(prompts)
            or not all(isinstance(text, str) for text in negatives)
        ):
            raise ValueError(
                "negative_prompt must be a string or match the prompt list"
            )
        generators = (
            generator if isinstance(generator, list) else [generator] * len(prompts)
        )
        if len(generators) != len(prompts):
            raise ValueError("provide one generator per prompt")

        conditioning = self._encode(prompts)
        unconditional = self._encode(negatives) if guidance != 1 else None
        shape = (16, height // 8, width // 8)
        noise = torch.stack(
            [
                torch.randn(
                    shape,
                    dtype=torch.float32,
                    device=g.device if g is not None else "cpu",
                    generator=g,
                ).to(self.device)
                for g in generators
            ]
        )

        def velocity(x, t):
            conditional = self._predict(x, t, conditioning)
            if unconditional is None:
                return conditional
            empty = self._predict(x, t, unconditional)
            return empty + guidance * (conditional - empty)

        latents = sample_ode(
            velocity, noise, steps=steps, progress_callback=progress_callback
        )
        images = decode_latents(latents * self.vae_std + self.vae_mean, self.vae)
        return InkoPipelineOutput(images=images)
