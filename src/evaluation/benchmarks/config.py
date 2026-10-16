"""Complete, typed sampling recipes; backend semantics belong to services."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.posttrain.schema import decode, name, require
from src.posttrain.scoring.config import Scorer


@dataclass(frozen=True)
class Checkpoint:
    asset: str
    weights: str


@dataclass(frozen=True)
class Arm:
    name: str
    checkpoint: Checkpoint
    steps: int
    guidance: float
    shift: str


@dataclass(frozen=True)
class Sampling:
    precision: str
    batch_size: int
    text_exit_layer: int
    shapes: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class Execution:
    learner_devices: tuple[int, ...]
    max_inflight_batches: int
    startup_timeout: int
    task_timeout: int
    cpu_threads: int


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
    scorers: tuple[Scorer, ...]

    def __post_init__(self):
        s, e, v = self.sampling, self.execution, self.evaluation
        require(
            s.precision in {"bfloat16", "float32"}, "unsupported sampling precision"
        )
        require(
            s.batch_size > 0 and s.text_exit_layer > 0 and bool(s.shapes),
            "invalid sampling dimensions",
        )
        require(
            all(
                all(n > 0 and n % 16 == 0 for n in shape) for shape in s.shapes.values()
            ),
            "image dimensions must be multiples of 16",
        )
        require(
            bool(e.learner_devices)
            and min(e.learner_devices) >= 0
            and len(set(e.learner_devices)) == len(e.learner_devices),
            "invalid learner devices",
        )
        require(
            min(
                e.max_inflight_batches, e.startup_timeout, e.task_timeout, e.cpu_threads
            )
            > 0,
            "execution limits must be positive",
        )
        require(
            v.bootstrap_repeats > 0
            and 0 < v.confidence < 1
            and v.prompt_limit >= 0
            and v.draws_per_prompt > 0,
            "invalid bootstrap settings",
        )
        require(
            bool(self.arms) and len({a.name for a in self.arms}) == len(self.arms),
            "unique nonempty arms required",
        )
        for a in self.arms:
            name(a.name)
            name(a.checkpoint.asset)
            name(a.checkpoint.weights)
            require(a.steps > 0, "invalid sampler")
            require(
                a.guidance >= 0 and a.shift in {"shape", "none"},
                "invalid guidance/shift",
            )
        require(
            bool(self.scorers)
            and len({s.id for s in self.scorers}) == len(self.scorers),
            "unique nonempty scorers required",
        )
        channels = {f"{s.id}.{c.name}" for s in self.scorers for c in s.outputs}
        require(
            bool(v.channels)
            and len(set(v.channels)) == len(v.channels)
            and set(v.channels) <= channels,
            "evaluation requires declared unique channels",
        )


def load_config(path: str | Path) -> Config:
    if Path(path).suffix == ".json":
        return decode(Config, json.loads(Path(path).read_text()))
    with Path(path).open("rb") as stream:
        return decode(Config, tomllib.load(stream))


def benchmark_config(training):
    c, e, v = training.collection, training.execution, training.evaluation
    services = {channel.split(".")[0] for channel in v.channels}
    return Config(
        Sampling(e.precision, e.microbatch, c.text_exit_layer, v.shapes),
        Execution(
            e.learner_devices,
            2 * len(e.learner_devices),
            e.startup_timeout,
            e.round_timeout,
            e.cpu_threads,
        ),
        Evaluation(
            v.suite,
            v.monitor_suite,
            v.channels,
            v.bootstrap_seed,
            v.bootstrap_repeats,
            v.confidence,
            Diversity(("development",), v.diversity_asset, 2, e.cpu_threads),
            v.prompt_limit,
            v.draws_per_prompt,
        ),
        (
            Arm(
                "base",
                Checkpoint(training.initialization.asset, "initial"),
                v.steps,
                v.guidance,
                v.shift,
            ),
        ),
        tuple(s for s in training.scorers if s.id in services),
    )
