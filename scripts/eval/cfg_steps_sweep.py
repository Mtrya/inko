"""CFG x ODE-steps sweep over the fixed prompt suite for one checkpoint.

Every cell (steps, guidance) regenerates the whole suite with each prompt's
own seed, so cells stay pairwise comparable. Each cell is written as a
blind-panel arm directory: <out>/steps050-cfg10/<prompt_id>.png. Sampling and
decoding follow the training-time monitor (src.evaluation.prompt_grid); the
guidance combination follows src.posttrain.roles.Velocity (conditional and
unconditional velocities are combined in FP32, and guidance 1 skips the
unconditional forward).

Run under torchrun; chunks (cell, resolution bucket, prompt batch) are
sharded by chunk index modulo world size. There is no collective
communication; rank 0 writes the manifest.

    torchrun --standalone --nproc_per_node=8 -m scripts.eval.cfg_steps_sweep \
        --checkpoint runs/inko-pretrain/checkpoint_step_584000 \
        --out data/cfg-steps-584k --prompts configs/prompts/monitor.jsonl \
        --eval-dataset precomputed_dataset/light-eval@896p \
        --text-encoder models/Qwen3-0.6B --vae models/e2e-qwenimage-vae
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint directory holding the weights and transformer_config.json")
    parser.add_argument("--weights", choices=["ema", "live"], default="ema")
    parser.add_argument("--out", required=True, help="output root for the per-cell arm directories")
    parser.add_argument("--prompts", required=True, help="prompt suite JSONL")
    parser.add_argument("--eval-dataset", required=True,
                        help="precomputed eval dataset; supplies the resolution bucket shapes")
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--steps", default="50,40,20", help="comma-separated ODE step counts")
    parser.add_argument("--cfg", default="1.0,1.5,2.0,3.0", help="comma-separated guidance scales")
    parser.add_argument("--exit-layer", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="npu")
    return parser.parse_args()


def cell_name(steps: int, guidance: float) -> str:
    return f"steps{steps:03d}-cfg{int(round(guidance * 10)):02d}"


def set_device(kind: str, index: int) -> torch.device:
    if kind == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(index)
    elif kind == "cuda":
        torch.cuda.set_device(index)
    elif kind != "cpu":
        raise ValueError(f"unsupported device {kind}")
    return torch.device(kind if kind == "cpu" else f"{kind}:{index}")


def build_chunks(prompts, cells, batch_size):
    chunks = []
    for steps, guidance in cells:
        by_bucket = {}
        for prompt in prompts:
            by_bucket.setdefault(prompt["_bucket"], []).append(prompt)
        for bucket_id in sorted(by_bucket):
            group = by_bucket[bucket_id]
            for start in range(0, len(group), batch_size):
                chunks.append((steps, guidance, bucket_id, group[start : start + batch_size]))
    return chunks


@torch.no_grad()
def run_chunk(model, text_encoder, tokenizer, vae, vae_mean, vae_std, device,
              shapes, chunk, exit_layer, out_root):
    from src.flow.solvers import sample_ode
    from src.utils.encode_text import encode_text
    from src.evaluation.prompt_grid import resolved_prompt_seed

    steps, guidance, bucket_id, batch = chunk
    h_lat, w_lat = shapes[bucket_id]

    txt, txt_mask, txt_pooled = encode_text(
        [p["text"] for p in batch], text_encoder, tokenizer, True, exit_layer=exit_layer
    )
    unconditional = None
    if guidance != 1:
        unconditional = encode_text(
            [""] * len(batch), text_encoder, tokenizer, True, exit_layer=exit_layer
        )

    noise = torch.stack(
        [
            torch.randn(
                (16, h_lat, w_lat),
                generator=torch.Generator(device="cpu").manual_seed(resolved_prompt_seed(p)),
            )
            for p in batch
        ]
    ).to(device, torch.bfloat16)

    def model_fn(x, t):
        if isinstance(t, float):
            t = torch.tensor(t, device=x.device).expand(x.shape[0])
        with torch.autocast(device.type, dtype=torch.bfloat16):
            velocity = model(x, t, txt, txt_pooled, txt_mask).float()
            if unconditional is not None:
                u_txt, u_mask, u_pooled = unconditional
                empty = model(x, t, u_txt, u_pooled, u_mask).float()
                velocity = empty + guidance * (velocity - empty)
        return velocity

    samples = sample_ode(model_fn, noise, steps=steps, t_start=0.0, t_end=1.0)
    samples = samples.to(torch.bfloat16) * vae_std + vae_mean
    images = vae.decode(samples.unsqueeze(2)).sample.squeeze(2)
    images = torch.clamp((images + 1) / 2, 0, 1).cpu().float()

    from PIL import Image

    directory = Path(out_root) / cell_name(steps, guidance)
    directory.mkdir(parents=True, exist_ok=True)
    for prompt, image in zip(batch, images):
        array = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        image_path = directory / f"{prompt['id']}.png"
        temporary = image_path.with_suffix(".png.tmp")
        Image.fromarray(array).save(temporary, format="PNG")
        temporary.replace(image_path)


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = set_device(args.device, local)

    from diffusers import AutoencoderKLQwenImage
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluation.prompt_grid import load_prompt_plan, resolved_prompt_seed
    from src.models.inko import Inko
    from src.utils.vae_codec import get_vae_stats

    checkpoint = Path(args.checkpoint)
    weight_file = checkpoint / (
        "ema_weights.pt" if args.weights == "ema" else "model.safetensors"
    )
    model = (
        Inko.from_single_file(str(weight_file))
        .to(device, torch.float32)
        .eval()
        .requires_grad_(False)
    )
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            args.text_encoder, torch_dtype=torch.bfloat16, local_files_only=True
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder, local_files_only=True)
    vae = (
        AutoencoderKLQwenImage.from_pretrained(
            args.vae, torch_dtype=torch.bfloat16, local_files_only=True
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    vae_mean, vae_std = get_vae_stats(args.vae, device)
    vae_mean, vae_std = vae_mean.to(torch.bfloat16), vae_std.to(torch.bfloat16)

    prompts, shapes = load_prompt_plan(args.prompts, args.eval_dataset)
    steps_list = [int(s) for s in args.steps.split(",")]
    cfg_list = [float(g) for g in args.cfg.split(",")]
    cells = [(s, g) for s in steps_list for g in cfg_list]
    chunks = build_chunks(prompts, cells, args.batch_size)
    assigned = [chunk for i, chunk in enumerate(chunks) if i % world == rank]

    if rank == 0:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        manifest = {
            "checkpoint": str(checkpoint),
            "weights": args.weights,
            "solver": "euler",
            "text_exit_layer": args.exit_layer,
            "cells": [
                {"arm": cell_name(s, g), "steps": s, "guidance": g} for s, g in cells
            ],
            "prompts": [
                {
                    "id": p["id"],
                    "seed": resolved_prompt_seed(p),
                    "scene_id": p.get("scene_id"),
                    "lang": p.get("lang"),
                    "variant": p.get("variant"),
                    "aspect": p.get("aspect", "1:1"),
                    "bucket": p["_bucket"],
                    "latent_shape": list(shapes[p["_bucket"]]),
                }
                for p in prompts
            ],
        }
        (Path(args.out) / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    for i, chunk in enumerate(assigned):
        started = time.monotonic()
        run_chunk(
            model, text_encoder, tokenizer, vae, vae_mean, vae_std, device,
            shapes, chunk, args.exit_layer, args.out,
        )
        steps, guidance, bucket_id, batch = chunk
        print(
            f"[rank {rank}] {i + 1}/{len(assigned)} {cell_name(steps, guidance)} "
            f"bucket={bucket_id} batch={len(batch)} {time.monotonic() - started:.1f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
