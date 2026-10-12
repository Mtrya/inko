"""Explicit cadence, suites and weight views for serialized checkpoint evaluation."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .schema import decode, name, require


@dataclass(frozen=True)
class IntervalEvaluation:
    sampling_config: str
    every: int
    suites: tuple[str, ...]
    weights: tuple[str, ...]
    include_final: bool
    blind_seed: int

    def __post_init__(self):
        require(
            self.every > 0 and bool(self.sampling_config),
            "invalid evaluation cadence/config",
        )
        require(
            bool(self.suites)
            and len(set(self.suites)) == len(self.suites)
            and set(self.suites)
            <= {"development", "bringup", "monitor", "confirmation"},
            "invalid evaluation suites",
        )
        require(
            bool(self.weights) and len(set(self.weights)) == len(self.weights),
            "unique evaluation weight views required",
        )
        for view in self.weights:
            name(view)
        require(0 <= self.blind_seed < 2**63, "invalid blind-panel seed")


def load_interval_config(path):
    path = Path(path)
    value = (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else tomllib.loads(path.read_text())
    )
    return decode(IntervalEvaluation, value)


def boundaries(config, interval):
    from .clock import rounds_before

    require(
        set(interval.weights) <= {"live", *(v.name for v in config.training.ema)},
        "evaluation names an absent EMA view",
    )
    steps = set(range(interval.every, config.training.steps + 1, interval.every))
    steps.add(config.training.steps)
    for step in sorted(steps):
        rounds_before(config, step)
        yield step


def evaluation_due(config, interval, step):
    return step > 0 and (
        step % interval.every == 0
        or (interval.include_final and step == config.training.steps)
    )
