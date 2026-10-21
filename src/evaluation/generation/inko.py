"""Resident benchmark generation from exported model weights."""

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from src.flow.solvers import sample_ode
from src.posttrain.sampling import sample, time_grid
from src.utils.device import peak_memory, synchronize
from src.utils.encode_text import encode_text

from .prompts import resolved_prompt_seed


class Generator:
    def __init__(self, sampling, assets, device):
        from diffusers import AutoencoderKLQwenImage
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.utils.vae_codec import get_vae_stats

        self.device, self.sampling = device, sampling
        self.dtype = getattr(torch, sampling.precision)
        self.assets = assets
        self.model = None
        self.checkpoint = None
        self.text = (
            AutoModelForCausalLM.from_pretrained(
                assets["text_encoder"],
                torch_dtype=self.dtype,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            assets["text_encoder"], local_files_only=True
        )
        self.vae = (
            AutoencoderKLQwenImage.from_pretrained(
                assets["vae"], torch_dtype=self.dtype, local_files_only=True
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.mean, self.std = get_vae_stats(assets["vae"], device)

    def select_checkpoint(self, checkpoint):
        from src.models.inko import Inko

        if self.checkpoint == checkpoint:
            return
        path = self.assets["checkpoints"][checkpoint.asset]
        # Replace once at an arm boundary; scorers remain resident independently.
        self.model = None
        self.model = (
            Inko.from_single_file(path)
            .to(self.device, self.dtype)
            .eval()
            .requires_grad_(False)
        )
        self.checkpoint = checkpoint

    def predict_velocity(self, x, t, conditioning):
        txt, pooled, mask = conditioning
        # Time features remain FP32; autocast handles the learned projection
        # after concatenation with lower-precision text conditioning.
        with torch.autocast(
            self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            return self.model(x.to(self.dtype), t, txt, pooled, mask)

    @torch.no_grad()
    def generate(self, task):
        from src.utils.encode_text import encode_text
        from src.utils.vae_codec import decode_latents

        synchronize(self.device)
        start = time.perf_counter()
        prompts, arm = task["prompts"], task["arm"]
        self.select_checkpoint(arm.checkpoint)
        txt, mask, pooled = encode_text(
            [p["text"] for p in prompts],
            self.text,
            self.tokenizer,
            pooling=True,
            exit_layer=self.sampling.text_exit_layer,
        )
        unconditional = None
        if arm.guidance != 1:
            u_txt, u_mask, u_pooled = encode_text(
                [""] * len(prompts),
                self.text,
                self.tokenizer,
                pooling=True,
                exit_layer=self.sampling.text_exit_layer,
            )
            unconditional = (u_txt, u_pooled, u_mask)
        synchronize(self.device)
        text_end = time.perf_counter()
        height, width = self.sampling.shapes[prompts[0]["aspect"]]
        shape = (16, height // 8, width // 8)
        shift = (
            max(1.0, ((shape[1] * shape[2] / 4) / 256) ** 0.5)
            if arm.shift == "shape"
            else 1.0
        )
        grid = time_grid(arm.steps, shift)

        def velocity(x, t):
            conditional = self.predict_velocity(x, t, (txt, pooled, mask))
            if unconditional is None:
                return conditional
            empty = self.predict_velocity(x, t, unconditional)
            return empty.float() + arm.guidance * (conditional.float() - empty.float())

        rollout = sample(
            velocity,
            [p["seed"] for p in prompts],
            shape,
            grid=grid,
            device=self.device,
        )
        synchronize(self.device)
        sampling_end = time.perf_counter()
        images = decode_latents(rollout.terminal * self.std + self.mean, self.vae)
        synchronize(self.device)
        decode_end = time.perf_counter()
        records = []
        for image, record, destination in zip(
            images, task["records"], task["image_paths"], strict=True
        ):
            image_path = Path(destination)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = image_path.with_suffix(".png.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(image_path)
            records.append({**record, "grid": list(grid), "shape": list(shape)})
        return {
            "records": records,
            "timings": {
                "text": text_end - start,
                "sampling": sampling_end - text_end,
                "decode": decode_end - sampling_end,
                "artifact_write": time.perf_counter() - decode_end,
            },
            "peak_memory_bytes": peak_memory(self.device),
        }


@torch.no_grad()
def sample_prompt_images(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    vae_mean: torch.Tensor,
    vae_std: torch.Tensor,
    text_encoder,
    tokenizer,
    prompts: List[Dict[str, Any]],
    shapes: Dict[int, Tuple[int, int]],
    *,
    batch_size: int,
    ode_steps: int,
    pooling: bool,
    device: torch.device,
    exit_layer: Optional[int] = None,
    amp_context: Optional[Callable[[], Any]] = None,
) -> List[Tuple[Dict[str, Any], torch.Tensor]]:
    """Sample one deterministic image per prompt, grouped by resolution bucket.

    This is the per-prompt sampling shared by the training-time prompt grid and
    the offline blind-panel generator, so the same checkpoint and prompt yield
    the same noise in both: the seed is explicit in the prompt record, the
    sampler is ``sample_ode`` (Euler) from t=0 to t=1, text conditioning is
    ``encode_text`` and latents are decoded with the VAE's own statistics.

    Each prompt must carry its ``_bucket`` (see ``load_prompt_plan``) and every
    bucket must appear in ``shapes``.

    Args:
        vae_mean, vae_std: latent statistics from ``get_vae_stats``, already
            cast by the caller to the dtype used for decoding.
        amp_context: zero-argument context-manager factory wrapped around the
            solver call (e.g. ``accelerator.autocast``); ``None`` disables
            autocasting.

    Returns:
        ``(prompt, image)`` pairs in bucket order, image a ``[3, H, W]``
        float32 CPU tensor in [0, 1].
    """
    context_factory = amp_context if amp_context is not None else nullcontext

    # Group prompts by bucket for batched generation
    by_bucket: Dict[int, List[Dict[str, Any]]] = {}
    for prompt in prompts:
        by_bucket.setdefault(prompt["_bucket"], []).append(prompt)

    results: List[Tuple[Dict[str, Any], torch.Tensor]] = []
    for bucket_id, bucket_prompts in sorted(by_bucket.items()):
        h_lat, w_lat = shapes[bucket_id]
        for start in range(0, len(bucket_prompts), batch_size):
            chunk = bucket_prompts[start : start + batch_size]
            txt, txt_mask, txt_pooled = encode_text(
                [p["text"] for p in chunk],
                text_encoder,
                tokenizer,
                pooling,
                exit_layer=exit_layer,
            )
            noise = torch.stack(
                [
                    torch.randn(
                        (16, h_lat, w_lat),
                        generator=torch.Generator(device="cpu").manual_seed(
                            resolved_prompt_seed(p)
                        ),
                    )
                    for p in chunk
                ]
            ).to(device, torch.bfloat16)

            def model_fn(x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask):
                if isinstance(t, float):
                    t = torch.tensor(t, device=x.device).expand(x.shape[0])
                return model(x, t, txt, txt_pooled, txt_mask)

            with context_factory():
                samples = sample_ode(
                    model_fn, noise, steps=ode_steps, t_start=0.0, t_end=1.0
                )

            samples = samples.to(dtype=torch.bfloat16)
            samples = samples * vae_std + vae_mean
            images = vae.decode(samples.unsqueeze(2)).sample.squeeze(2)
            images = torch.clamp((images + 1) / 2, 0, 1).cpu().float()
            for prompt, image in zip(chunk, images):
                results.append((prompt, image))
    return results
