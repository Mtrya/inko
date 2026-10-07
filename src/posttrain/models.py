"""Resident Inko rollout worker; model assets are always supplied explicitly."""

import time
from pathlib import Path

import torch

from .records import file_hash
from .sampling import noise_seed, sample, time_grid


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

        from src.models.inko import Inko
        from src.utils.vae_codec import get_vae_stats

        self.device, self.config = device, config
        self.dtype = getattr(torch, config["sampling"]["precision"])
        self.model = (
            Inko.from_single_file(assets["checkpoint"])
            .to(device, self.dtype)
            .eval()
            .requires_grad_(False)
        )
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
        txt, mask, pooled = encode_text(
            [p["text"] for p in prompts],
            self.text,
            self.tokenizer,
            pooling=True,
            exit_layer=self.config["initialization"]["text_exit_layer"],
        )
        synchronize(self.device)
        text_end = time.perf_counter()
        height, width = self.config["sampling"]["shapes"][prompts[0]["aspect"]]
        shape = (16, height // 8, width // 8)
        shift = max(1.0, ((shape[1] * shape[2] / 4) / 256) ** 0.5)
        grid = time_grid(arm["steps"], shift)
        indices = None
        if arm["rule"] == "renoise" and self.config["sampling"]["retain_selected"]:
            indices = [
                noise_seed(p["seed"], "training_index") % arm["steps"] for p in prompts
            ]

        def velocity(x, t):
            return self.predict_velocity(x, t, (txt, pooled, mask))

        rollout = sample(
            velocity,
            [p["seed"] for p in prompts],
            shape,
            rule=arm["rule"],
            grid=grid,
            device=self.device,
            selected_indices=indices,
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
            directory = Path(task["output"]) / arm["name"]
            directory.mkdir(parents=True, exist_ok=True)
            image_path = directory / f"{prompt['id']}.png"
            temporary = image_path.with_suffix(".png.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(image_path)
            tensors = {"terminal": rollout.terminal[i].cpu()}
            if rollout.selected_input is not None:
                tensors.update(
                    selected_input=rollout.selected_input[i].cpu(),
                    selected_index=rollout.selected_indices[i].cpu(),
                    selected_time=rollout.selected_times[i].cpu(),
                )
            latent_path = directory / f"{prompt['id']}.pt"
            temporary = latent_path.with_suffix(".pt.tmp")
            torch.save(tensors, temporary)
            temporary.replace(latent_path)
            records.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "arm": arm["name"],
                    "image_path": str(image_path),
                    "image_sha256": file_hash(image_path),
                    "latent_path": str(latent_path),
                    "latent_sha256": file_hash(latent_path),
                    "grid": list(grid),
                    "shape": list(shape),
                    "seed": prompt["seed"],
                    "selected_index": None if indices is None else indices[i],
                    "noise_specification": "CPU torch.randn float32; initial seed; SHA256 transition stream",
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
