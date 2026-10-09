"""Complete, typed sampling recipes; backend semantics belong to services."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .schema import decode, name, require
from .scoring.contracts import Channel


@dataclass(frozen=True)
class Checkpoint:
    asset: str
    weights: str


@dataclass(frozen=True)
class Arm:
    name: str
    checkpoint: Checkpoint
    rule: str
    steps: int
    guidance: float
    shift: str


@dataclass(frozen=True)
class Sampling:
    precision: str
    batch_size: int
    retain_selected: bool
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
    enabled: bool
    asset: str
    pair_batch_size: int
    cpu_threads: int

    def __post_init__(self):
        require(
            min(self.pair_batch_size, self.cpu_threads) > 0,
            "invalid LPIPS batch/thread count",
        )
        if self.enabled:
            name(self.asset)
        else:
            require(not self.asset, "disabled diversity must not claim metric assets")


@dataclass(frozen=True)
class Evaluation:
    suite: str
    bringup_suite: str
    monitor_suite: str
    confirmation_suite: str
    channels: tuple[str, ...]
    bootstrap_seed: int
    bootstrap_repeats: int
    confidence: float
    diversity: Diversity


@dataclass(frozen=True)
class Scorer:
    id: str
    outputs: tuple[Channel, ...]
    attempts: int
    timeout: float
    concurrency: int
    batch_size: int
    transport: str

    def __post_init__(self):
        name(self.id)
        require(self.id != "learner", "learner is reserved for the generation role")
        require(
            bool(self.outputs)
            and len({c.name for c in self.outputs}) == len(self.outputs),
            "unique scorer outputs required",
        )
        require(
            min(self.attempts, self.concurrency, self.batch_size, self.timeout) > 0,
            "acquisition limits must be positive",
        )
        require(
            self.transport in {"shared_file", "png_base64"}, "unknown image transport"
        )


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
            v.bootstrap_repeats > 0 and 0 < v.confidence < 1,
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
            require(a.rule in {"euler", "renoise"} and a.steps > 0, "invalid sampler")
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
