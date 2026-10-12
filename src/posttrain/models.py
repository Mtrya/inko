"""Resident Inko rollout worker; model assets are always supplied explicitly."""

import time
from dataclasses import asdict
from pathlib import Path

import torch

from .records import file_hash
from .sampling import sample, time_grid


def set_device(kind: str, index: int, cpu_threads: int):
    torch.set_num_threads(cpu_threads)
    if kind == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(index)
    elif kind == "cuda":
        torch.cuda.set_device(index)
    elif kind != "cpu":
        raise ValueError(f"unsupported device {kind}")
    return torch.device(kind if kind == "cpu" else f"{kind}:{index}")


def synchronize(device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def peak_memory(device):
    return (
        0
        if device.type == "cpu"
        else getattr(torch, device.type).max_memory_allocated(device)
    )


class Generator:
    def __init__(self, config, assets, device):
        from diffusers import AutoencoderKLQwenImage
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.utils.vae_codec import get_vae_stats

        self.device, self.config = device, config
        self.dtype = getattr(torch, config.sampling.precision)
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
        self.select_checkpoint(config.arms[0].checkpoint)

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
            exit_layer=self.config.sampling.text_exit_layer,
        )
        unconditional = None
        if arm.guidance != 1:
            u_txt, u_mask, u_pooled = encode_text(
                [""] * len(prompts),
                self.text,
                self.tokenizer,
                pooling=True,
                exit_layer=self.config.sampling.text_exit_layer,
            )
            unconditional = (u_txt, u_pooled, u_mask)
        synchronize(self.device)
        text_end = time.perf_counter()
        height, width = self.config.sampling.shapes[prompts[0]["aspect"]]
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
        for i, (prompt, image, sample_id) in enumerate(
            zip(prompts, images, task["sample_ids"], strict=True)
        ):
            directory = Path(task["output"]) / arm.name
            directory.mkdir(parents=True, exist_ok=True)
            image_path = directory / f"{prompt['id']}.png"
            temporary = image_path.with_suffix(".png.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(image_path)
            tensors = {"terminal": rollout.terminal[i].cpu()}
            latent_path = directory / f"{prompt['id']}.pt"
            temporary = latent_path.with_suffix(".pt.tmp")
            torch.save(tensors, temporary)
            temporary.replace(latent_path)
            records.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "arm": arm.name,
                    "checkpoint": asdict(arm.checkpoint),
                    "image_path": str(image_path),
                    "image_sha256": file_hash(image_path),
                    "latent_path": str(latent_path),
                    "latent_sha256": file_hash(latent_path),
                    "grid": list(grid),
                    "shape": list(shape),
                    "seed": prompt["seed"],
                    "conditioning_identity": task["conditioning_identity"],
                    "policy_version": task["policy_version"],
                }
            )
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
