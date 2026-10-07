"""Round-keyed collection, prompt permutation and selected-state replay."""

import hashlib
from dataclasses import dataclass

import torch

from .records import identity
from .sampling import sample, sample_noise, time_grid
from .schedules import cold_point, time_draws


def stream_seed(seed, round_index, sample_id, stream, update=0):
    payload = [
        "cold-cpu-fp32-v2",
        seed,
        "cold",
        round_index,
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
            stream_seed(self.seed, self.epoch, "prompts", "permutation")
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


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def collect(
    field, config, prompts, sample_ids, round_index, conditioning, unconditional, device
):
    c = config.collection
    seeds = [
        stream_seed(config.training.seed, round_index, key, "collection")
        for key in sample_ids
    ]
    indices = [
        int(
            torch.multinomial(
                torch.tensor(c.selected_pmf),
                1,
                generator=torch.Generator().manual_seed(
                    stream_seed(
                        config.training.seed, round_index, key, "selected_index"
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
        round_index,
        result.selected_input,
        result.selected_times,
        result.selected_indices,
        result.selected_prediction.detach(),
        result.terminal,
        conditioning,
        unconditional,
    )


def score_inputs(config, rollout, round_index, stream, update):
    seeds = [
        stream_seed(config.training.seed, round_index, key, stream, update)
        for key in rollout.sample_ids
    ]
    o = config.objective
    times = time_draws(
        seeds, cold_point(config, round_index), o.time_min, o.time_max
    ).to(rollout.terminal.device)
    noise = sample_noise(
        seeds,
        tuple(rollout.terminal.shape[1:]),
        stream="score_noise",
        device=rollout.terminal.device,
    )
    return times, noise
