"""Round-keyed collection, prompt permutation and selected-state replay."""

import hashlib
from dataclasses import dataclass

import torch

from .records import identity
from .sampling import sample, sample_noise, time_grid
from .schedules import cold_point, time_draws

NOISE_VERSION = "posttrain-cpu-fp32-v3"


def stream_seed(
    seed, round_index, sample_id, stream, update=0, *, stage, generator_step
):
    payload = [
        NOISE_VERSION,
        seed,
        stage,
        generator_step,
        round_index,
        sample_id,
        stream,
        update,
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
                stage="prompt_order",
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


@dataclass
class TrainingRollout:
    sample_ids: list[str]
    prompt_ids: list[str]
    seeds: list[int]
    student_version: int
    selected_input: torch.Tensor
    selected_time: torch.Tensor
    selected_index: torch.Tensor
    selected_prediction: torch.Tensor
    terminal: torch.Tensor
    conditioning: tuple
    unconditional: tuple | None
    stage: str
    collection_round: int


@dataclass
class NFTRollout:
    sample_ids: list[str]
    prompt_ids: list[str]
    seeds: list[int]
    old_version: int
    student_version: int
    terminal: torch.Tensor
    conditioning: tuple
    unconditional: tuple | None
    stage: str
    collection_round: int


def tensor_digest(value):
    # A size-one expanded tensor can be "contiguous" with stride zero. Allocate
    # canonical storage before reinterpreting bytes; flatten also admits scalars.
    value = (
        value.detach().cpu().reshape(-1).clone(memory_format=torch.contiguous_format)
    )
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def collect(
    field,
    config,
    prompts,
    sample_ids,
    round_index,
    conditioning,
    unconditional,
    device,
    *,
    stage,
    generator_step,
    stream="collection",
):
    c = config.collection
    seeds = [
        stream_seed(
            config.training.seed,
            round_index,
            key,
            stream,
            stage=stage,
            generator_step=generator_step,
        )
        for key in sample_ids
    ]
    indices = [
        int(
            torch.multinomial(
                torch.tensor(c.selected_pmf),
                1,
                generator=torch.Generator().manual_seed(
                    stream_seed(
                        config.training.seed,
                        round_index,
                        key,
                        "selected_index",
                        stage=stage,
                        generator_step=generator_step,
                    )
                ),
            )
        )
        for key in sample_ids
    ]
    shape = (16, c.height // 8, c.width // 8)
    shift = (
        max(1, ((shape[1] * shape[2] / 4) / 256) ** 0.5) if c.shift == "shape" else 1
    )
    result = sample(
        lambda x, t: field(x, t, conditioning, unconditional, c.guidance),
        seeds,
        shape,
        rule=c.rule,
        grid=time_grid(c.steps, shift),
        device=device,
        selected_indices=indices,
    )
    return TrainingRollout(
        sample_ids,
        [p["id"] for p in prompts],
        seeds,
        generator_step,
        result.selected_input,
        result.selected_times,
        result.selected_indices,
        result.selected_prediction.detach(),
        result.terminal,
        conditioning,
        unconditional,
        stage,
        round_index,
    )


def score_inputs(config, rollout, stream, update):
    seeds = [
        stream_seed(
            config.training.seed,
            rollout.collection_round,
            key,
            stream,
            update,
            stage=rollout.stage,
            generator_step=rollout.student_version,
        )
        for key in rollout.sample_ids
    ]
    o = config.objective
    times = time_draws(
        seeds,
        (
            cold_point(config, rollout.student_version)
            if rollout.stage == "cold"
            else {"distribution": "uniform_physical"}
        ),
        o.time_min,
        o.time_max,
    ).to(rollout.terminal.device)
    noise = sample_noise(
        seeds,
        tuple(rollout.terminal.shape[1:]),
        stream="score_noise",
        device=rollout.terminal.device,
    )
    return times, noise


@torch.no_grad()
def fresh_rollouts(config, field, rollouts, stream, draw, *, generator_step=None):
    """Independent noise at the same prompts; never consumes the prompt cursor."""
    return [
        collect(
            field,
            config,
            [{"id": key} for key in row.prompt_ids],
            row.sample_ids,
            row.collection_round,
            row.conditioning,
            row.unconditional,
            row.terminal.device,
            stage=row.stage,
            generator_step=(
                row.student_version if generator_step is None else generator_step
            ),
            stream=f"{stream}/{draw}",
        )
        for row in rollouts
    ]


@torch.no_grad()
def collect_nft(
    field,
    config,
    prompts,
    sample_ids,
    round_index,
    old_version,
    generator_step,
    conditioning,
    unconditional,
    device,
):
    c = config.collection
    seeds = [
        stream_seed(
            config.training.seed,
            round_index,
            key,
            f"old_collection/{old_version}",
            stage="joint",
            generator_step=generator_step,
        )
        for key in sample_ids
    ]
    shape = (16, c.height // 8, c.width // 8)
    shift = (
        max(1, ((shape[1] * shape[2] / 4) / 256) ** 0.5) if c.shift == "shape" else 1
    )
    result = sample(
        lambda x, t: field(x, t, conditioning, unconditional, c.guidance),
        seeds,
        shape,
        rule=c.rule,
        grid=time_grid(c.steps, shift),
        device=device,
    )
    return NFTRollout(
        sample_ids,
        [p["id"] for p in prompts],
        seeds,
        old_version,
        generator_step,
        result.terminal,
        conditioning,
        unconditional,
        "joint",
        round_index,
    )


def nft_inputs(config, rollout, generator_step, draw):
    seeds = [
        stream_seed(
            config.training.seed,
            rollout.collection_round,
            key,
            "nft_noise",
            draw,
            stage="joint",
            generator_step=generator_step,
        )
        for key in rollout.sample_ids
    ]
    time_seeds = [
        stream_seed(
            config.training.seed,
            rollout.collection_round,
            key,
            "nft_time",
            draw,
            stage="joint",
            generator_step=generator_step,
        )
        for key in rollout.sample_ids
    ]
    n, c = config.nft, config.collection
    if n.time_distribution == "uniform":
        times = time_draws(
            time_seeds, {"distribution": "uniform_physical"}, n.time_min, n.time_max
        )
    else:
        shape = rollout.terminal.shape[1:]
        shift = (
            max(1, ((shape[1] * shape[2] / 4) / 256) ** 0.5)
            if c.shift == "shape"
            else 1
        )
        grid = time_grid(c.steps, shift)[:-1]
        times = torch.tensor(
            [
                grid[
                    int(
                        torch.randint(
                            len(grid), (), generator=torch.Generator().manual_seed(seed)
                        )
                    )
                ]
                for seed in time_seeds
            ],
            dtype=torch.float32,
        )
    noise = sample_noise(
        seeds,
        tuple(rollout.terminal.shape[1:]),
        stream="nft_noise",
        device=rollout.terminal.device,
    )
    return times.to(rollout.terminal.device), noise
