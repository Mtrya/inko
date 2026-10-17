"""Fixed prompt panels collected and logged by the training loop."""

import json
import os
from typing import List, Optional

import numpy as np
import swanlab
import torch
import torchvision
from accelerate.utils import gather_object

from src.evaluation.generation.inko import sample_prompt_images
from src.evaluation.generation.prompts import load_prompt_plan, resolved_prompt_seed
from src.utils.vae_codec import get_vae_stats


def grid_due(step: int, interval: int, extra_steps: List[int]) -> bool:
    """Global-step triggers stay fixed across crash resumes; overlaps run once."""
    return step in extra_steps or (interval > 0 and step > 0 and step % interval == 0)


def make_image_grid(
    images: torch.Tensor,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    save_path: Optional[str] = None,
    normalize: bool = True,
    value_range: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Create a grid of images and optionally save it.

    Args:
        images: Tensor of shape [B, C, H, W]
        rows: Number of rows (optional)
        cols: Number of columns (optional)
        save_path: Path to save the grid image
        normalize: Whether to normalize images to [0, 1]
        value_range: Range of values in input images (min, max)

    Returns:
        Grid tensor
    """
    if rows is None and cols is None:
        nrow = int(np.ceil(np.sqrt(images.shape[0])))
    elif cols is not None:
        nrow = cols
    else:
        nrow = int(np.ceil(images.shape[0] / rows))

    grid = torchvision.utils.make_grid(
        images, nrow=nrow, normalize=normalize, value_range=value_range, padding=2
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(grid, save_path)

    return grid


def format_prompt_caption(prompts: List[str], limit: int = 32) -> str:
    """
    Format a list of prompts for display as an image caption.

    Args:
        prompts: List of prompt strings
        limit: Maximum number of prompts to include

    Returns:
        Formatted caption string
    """
    if not prompts:
        return ""
    trimmed = [p.replace("\n", " ").strip() for p in prompts[:limit]]
    lines = [f"{idx + 1}. {text}" for idx, text in enumerate(trimmed)]
    remaining = len(prompts) - len(trimmed)
    if remaining > 0:
        lines.append(f"... (+{remaining} more)")
    return "\n\n".join(lines)


@torch.no_grad()
def run_prompt_grid_eval(
    accelerator,
    model: torch.nn.Module,
    vae_path: str,
    save_path: str,
    current_step: int,
    text_encoder,
    tokenizer,
    pooling: bool,
    prompts_path: str,
    exit_layer: Optional[int] = None,
    eval_dataset_path: str = "./precomputed_dataset/light-eval@256p",
    batch_size: int = 8,
    ode_steps: int = 50,
    weights: str = "unspecified",
) -> None:
    """Generate fixed-seed samples for the prompt suite and log grids."""
    from diffusers import AutoencoderKLQwenImage

    print(f"Running prompt-grid evaluation at step {current_step}...")
    was_training = model.training
    model.eval()

    device = accelerator.device
    num_processes = getattr(accelerator, "num_processes", 1)
    process_index = getattr(accelerator, "process_index", 0)

    prompts, shapes = load_prompt_plan(prompts_path, eval_dataset_path)

    # Shard prompts across ranks
    local_prompts = [
        p for i, p in enumerate(prompts) if i % max(1, num_processes) == process_index
    ]

    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    vae_mean, vae_std = get_vae_stats(vae_path, device=device)
    vae_mean = vae_mean.to(dtype=torch.bfloat16)
    vae_std = vae_std.to(dtype=torch.bfloat16)

    results = sample_prompt_images(
        model,
        vae,
        vae_mean,
        vae_std,
        text_encoder,
        tokenizer,
        local_prompts,
        shapes,
        batch_size=batch_size,
        ode_steps=ode_steps,
        pooling=pooling,
        device=device,
        exit_layer=exit_layer,
        amp_context=accelerator.autocast,
    )

    # Accelerate exposes this as a utility, not an Accelerator method.
    # It concatenates rank-local lists and is a no-op on a single process.
    all_results = gather_object(results)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(save_path, "samples"), exist_ok=True)
        # Gathering groups by rank; restore suite order so changing GPU count
        # does not rearrange the images within each bucket.
        prompt_order = {p["id"]: i for i, p in enumerate(prompts)}
        all_results.sort(key=lambda result: prompt_order[result[0]["id"]])
        # New panels pair language/length variants per scene. Old suites keep
        # their bucket grids. Include the bucket in the key to avoid stacking
        # differently shaped images from malformed scene records.
        grids = {}
        for p, img in all_results:
            key = (p.get("scene_id", ""), p["_bucket"])
            grids.setdefault(key, []).append((p, img))

        for (scene, bid), items in sorted(grids.items()):
            h_lat, w_lat = shapes[bid]
            if scene:
                order = {
                    ("zh", "short"): 0,
                    ("en", "short"): 1,
                    ("zh", "long"): 2,
                    ("en", "long"): 3,
                }
                items.sort(
                    key=lambda item: order[(item[0]["lang"], item[0]["variant"])]
                )
            images = torch.stack([img for _, img in items])
            captions = [
                f"{p['lang']} / {p['variant']} / seed={resolved_prompt_seed(p)}: "
                f"{p['text'][:240]}"
                if scene
                else p["text"]
                for p, _ in items
            ]
            label = f"{scene}_bucket{bid}" if scene else f"bucket{bid}"
            grid_path = os.path.join(
                save_path,
                "samples",
                f"grid_step_{current_step:06d}_{label}_{h_lat * 8}x{w_lat * 8}.png",
            )
            _ = make_image_grid(
                images,
                cols=2 if scene else None,
                save_path=grid_path,
                normalize=True,
                value_range=(0, 1),
            )
            caption = format_prompt_caption(captions, limit=len(captions))
            media = (
                swanlab.Image(grid_path, caption=caption)
                if caption
                else swanlab.Image(grid_path)
            )
            accelerator.log(
                {f"grid/{label}_{h_lat * 8}x{w_lat * 8}": media}, step=current_step
            )
        # Preserve the full text and exact sampling identity, not only the
        # shortened grid caption visible in SwanLab.
        with open(
            os.path.join(save_path, "samples", f"panel_step_{current_step:06d}.json"),
            "w",
        ) as handle:
            json.dump(
                {
                    "step": current_step,
                    "solver": "euler",
                    "ode_steps": ode_steps,
                    "cfg_scale": 1.0,
                    "precision": "bf16",
                    "solver_precision": "fp32",
                    "timestep_precision": "fp32",
                    "exit_layer": exit_layer,
                    "weights": weights,
                    "pooling": pooling,
                    "eval_dataset_path": eval_dataset_path,
                    "prompts": [
                        {
                            **p,
                            "seed": resolved_prompt_seed(p),
                            "latent_shape": shapes[p["_bucket"]],
                        }
                        for p in prompts
                    ],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Logged {len(grids)} prompt grids at step {current_step}")

    del vae
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    if was_training:
        model.train()
