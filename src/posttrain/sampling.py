"""Inko-time sampling with detached selected-step state and sample-local RNG."""

import hashlib
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


def noise_seed(seed: int, stream: str, index: int = 0) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("noise seed must be in [0, 2**63)")
    # Initial noise agrees with the established monitoring seed convention.
    if stream == "initial" and index == 0:
        return seed
    return (
        int.from_bytes(
            hashlib.sha256(f"{seed}:{stream}:{index}".encode()).digest()[:8], "big"
        )
        % 2**63
    )


def sample_noise(
    seeds: list[int],
    shape: tuple[int, int, int],
    *,
    stream: str,
    index: int = 0,
    device="cpu",
) -> torch.Tensor:
    return torch.stack(
        [
            torch.randn(
                shape,
                dtype=torch.float32,
                generator=torch.Generator(device="cpu").manual_seed(
                    noise_seed(s, stream, index)
                ),
            )
            for s in seeds
        ]
    ).to(device)


def time_grid(steps: int, shift: float) -> tuple[float, ...]:
    if type(steps) is not int or steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("steps and shift must be positive")
    base = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
    return tuple((base / (shift - (shift - 1) * base)).tolist())


@dataclass
class Rollout:
    terminal: torch.Tensor
    selected_input: torch.Tensor | None
    selected_indices: torch.Tensor | None
    selected_times: torch.Tensor | None
    grid: tuple[float, ...]


@torch.no_grad()
def sample(
    model: Callable,
    seeds: list[int],
    shape: tuple[int, int, int],
    *,
    rule: str,
    grid: tuple[float, ...],
    device="cpu",
    selected_indices: list[int] | None = None,
) -> Rollout:
    if rule not in {"euler", "renoise"}:
        raise ValueError("unknown sampling rule")
    if (
        len(grid) < 2
        or grid[0] != 0
        or grid[-1] != 1
        or any(not math.isfinite(t) for t in grid)
        or any(a >= b for a, b in itertools.pairwise(grid))
    ):
        raise ValueError("grid must increase from noise=0 to data=1")
    if not seeds:
        raise ValueError("empty sampling batch")
    x = sample_noise(seeds, shape, stream="initial", device=device)
    selected = indices = times = None
    if selected_indices is not None:
        if len(selected_indices) != len(seeds) or any(
            type(k) is not int or not 0 <= k < len(grid) - 1 for k in selected_indices
        ):
            raise ValueError("one valid selected index is required per sample")
        indices = torch.tensor(selected_indices, device=device)
        selected = torch.empty_like(x)
        times = torch.tensor([grid[k] for k in selected_indices], device=device)
    for k, (t, next_t) in enumerate(itertools.pairwise(grid)):
        if selected is not None:
            mask = indices == k
            selected[mask] = x[mask]
        velocity = model(
            x, torch.full((len(seeds),), t, device=device, dtype=torch.float32)
        ).float()
        if velocity.shape != x.shape:
            raise ValueError("model velocity shape differs from latent")
        if rule == "euler":
            x = x + (next_t - t) * velocity
        else:
            clean = x + (1 - t) * velocity
            x = (
                clean
                if k == len(grid) - 2
                else (
                    next_t * clean
                    + (1 - next_t)
                    * sample_noise(
                        seeds, shape, stream="transition", index=k + 1, device=device
                    )
                )
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"nonfinite {rule} latent at step {k}")
    return Rollout(x.detach(), selected, indices, times, grid)
