"""Deterministic Euler sampling with sample-local RNG."""

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
    grid: tuple[float, ...]
    states: torch.Tensor | None = None  # [steps, batch, ...] prediction-site latents
    predictions: torch.Tensor | None = None  # [steps, batch, ...] velocities there


@torch.no_grad()
def sample(
    model: Callable,
    seeds: list[int],
    shape: tuple[int, int, int],
    *,
    grid: tuple[float, ...],
    device="cpu",
    trajectory=False,
) -> Rollout:
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
    states, predictions = [], []
    for k, (t, next_t) in enumerate(itertools.pairwise(grid)):
        velocity = model(
            x, torch.full((len(seeds),), t, device=device, dtype=torch.float32)
        ).float()
        if velocity.shape != x.shape:
            raise ValueError("model velocity shape differs from latent")
        if trajectory:
            # Detached state and prediction at this step, before the update.
            states.append(x)
            predictions.append(velocity)
        x = x + (next_t - t) * velocity
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"nonfinite latent at step {k}")
    return Rollout(
        x.detach(),
        grid,
        None if not trajectory else torch.stack(states),
        None if not trajectory else torch.stack(predictions),
    )
