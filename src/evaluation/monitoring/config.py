"""Complete, typed sampling recipes; backend semantics belong to services."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.generation.config import (
    Arm,
    Execution,
    Sampling,
    validate_generation,
)
from src.posttrain.schema import decode, name, require
from src.posttrain.scoring.config import ClientConfig
from src.posttrain.scoring.contracts import CHANNELS


@dataclass(frozen=True)
class Diversity:
    suites: tuple[str, ...]
    asset: str
    pair_batch_size: int
    cpu_threads: int

    def __post_init__(self):
        require(
            min(self.pair_batch_size, self.cpu_threads) > 0,
            "invalid LPIPS batch/thread count",
        )
        require(
            len(set(self.suites)) == len(self.suites),
            "invalid diversity suites",
        )
        for suite in self.suites:
            name(suite)
        if self.suites:
            name(self.asset)
        else:
            require(not self.asset, "disabled diversity must not claim metric assets")


@dataclass(frozen=True)
class Evaluation:
    suite: str
    monitor_suite: str
    channels: tuple[str, ...]
    bootstrap_seed: int
    bootstrap_repeats: int
    confidence: float
    diversity: Diversity
    prompt_limit: int = 0
    draws_per_prompt: int = 1


@dataclass(frozen=True)
class Config:
    sampling: Sampling
    execution: Execution
    evaluation: Evaluation
    arms: tuple[Arm, ...]
    scoring: ClientConfig

    def __post_init__(self):
        v = self.evaluation
        validate_generation(self.sampling, self.execution, self.arms)
        require(
            v.bootstrap_repeats > 0
            and 0 < v.confidence < 1
            and v.prompt_limit >= 0
            and v.draws_per_prompt > 0,
            "invalid bootstrap settings",
        )
        require(
            bool(v.channels)
            and len(set(v.channels)) == len(v.channels)
            and set(v.channels) <= set(CHANNELS),
            "evaluation requires declared unique channels",
        )


def load_config(path: str | Path) -> Config:
    if Path(path).suffix == ".json":
        return decode(Config, json.loads(Path(path).read_text()))
    with Path(path).open("rb") as stream:
        return decode(Config, tomllib.load(stream))
