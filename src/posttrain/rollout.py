"""Round-keyed ART collection, prompt permutation and cached trajectory inputs."""

import hashlib
from dataclasses import dataclass

import torch

from .records import identity
from .sampling import sample, time_grid

NOISE_VERSION = "posttrain-cpu-fp32-v4"


def stream_seed(seed, round_index, sample_id, stream, *, generator_step):
    payload = [
        NOISE_VERSION,
        seed,
        generator_step,
        round_index,
        sample_id,
        stream,
        0,  # Preserve the collection stream shared with the completed control.
    ]
    return int(identity(payload)[:16], 16) % 2**63


class PromptCursor:
    def __init__(self, size, seed):
        self.size, self.seed, self.epoch, self.cursor = size, seed, 0, 0
        self.permutation = self._permutation()

    def _permutation(self):
        rng = torch.Generator().manual_seed(
            stream_seed(
                self.seed,
                self.epoch,
                "prompts",
                "permutation",
                generator_step=0,
            )
        )
        return torch.randperm(self.size, generator=rng).tolist()

    def take(self, count):
        result = []
        while len(result) < count:
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
                self.permutation = self._permutation()
            take = min(count - len(result), self.size - self.cursor)
            result.extend(self.permutation[self.cursor : self.cursor + take])
            self.cursor += take
        return result

    def state_dict(self):
        return {
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "permutation": self.permutation,
        }

    def load_state_dict(self, state):
        if state["size"] != self.size or state["seed"] != self.seed:
            raise ValueError("prompt cursor contract changed")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        if (
            state["permutation"] != self._permutation()
            or not 0 <= self.cursor <= self.size
        ):
            raise ValueError("invalid prompt permutation/cursor")
        self.permutation = state["permutation"]


def tensor_digest(value):
    # A size-one expanded tensor can be "contiguous" with stride zero. Allocate
    # canonical storage before reinterpreting bytes; flatten also admits scalars.
    value = (
        value.detach().cpu().reshape(-1).clone(memory_format=torch.contiguous_format)
    )
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass
class ARTRollout:
    sample_ids: list[str]
    prompt_ids: list[str]
    draw_indices: list[int]
    seeds: list[int]
    generator_step: int
    terminal: torch.Tensor  # [batch, ...] endpoints of the round's draws
    states: torch.Tensor  # [batch, steps, ...] visited trajectory states
    predictions: torch.Tensor  # [batch, steps, ...] rollout velocities there
    grid: tuple[float, ...]
    conditioning: tuple
    collection_round: int


@torch.no_grad()
def collect_art(
    field,
    config,
    prompts,
    sample_ids,
    draw_indices,
    round_index,
    generator_step,
    conditioning,
    device,
):
    """Roll out the live policy and cache the trajectory it visits."""
    c = config.collection
    seeds = [
        stream_seed(
            config.training.seed,
            round_index,
            key,
            "collection",
            generator_step=0,
        )
        for key in sample_ids
    ]
    shape = (16, c.height // 8, c.width // 8)
    shift = (
        max(1, ((shape[1] * shape[2] / 4) / 256) ** 0.5) if c.shift == "shape" else 1
    )
    result = sample(
        lambda x, t: field(x, t, conditioning),
        seeds,
        shape,
        grid=time_grid(c.steps, shift),
        device=device,
        trajectory=True,
    )
    # Trajectory storage is host-resident fp32: the anchor is exactly the
    # prediction the rollout made, matching the reference implementation.
    return ARTRollout(
        sample_ids,
        [p["id"] for p in prompts],
        list(draw_indices),
        seeds,
        generator_step,
        result.terminal,
        result.states.transpose(0, 1).cpu().contiguous(),
        result.predictions.transpose(0, 1).cpu().contiguous(),
        result.grid,
        conditioning,
        round_index,
    )


def art_window(config, rollouts, window):
    """One seeded, equal half of this learner's shuffled trajectory samples."""
    if window not in (0, 1):
        raise ValueError("ART trains two windows per collection round")
    samples = sorted(
        [
            (rollout, index)
            for rollout in rollouts
            for index in range(len(rollout.sample_ids))
        ],
        key=lambda item: item[0].draw_indices[item[1]],
    )
    if not samples or len(samples) % 2:
        raise ValueError("ART windows require an even number of local samples")
    first, index = samples[0]
    generator = torch.Generator().manual_seed(
        stream_seed(
            config.training.seed,
            first.collection_round,
            first.sample_ids[index],
            "art_windows",
            generator_step=first.generator_step,
        )
    )
    order = torch.randperm(len(samples), generator=generator).tolist()
    half = len(samples) // 2
    return [samples[i] for i in order[window * half : (window + 1) * half]]
