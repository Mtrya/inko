"""Complete single-stage ART training configuration, separate from benchmark arms."""

import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import Checkpoint, Scorer
from .schema import decode, name, require

ART_UPDATES_PER_ROUND = 2


@dataclass(frozen=True)
class Student:
    mode: str
    rank: int
    alpha: float
    targets: str


@dataclass(frozen=True)
class Collection:
    prompts: str
    prompts_per_round: int
    draws_per_prompt: int
    height: int
    width: int
    steps: int
    shift: str
    text_exit_layer: int


@dataclass(frozen=True)
class ART:
    reference_coefficient: float


@dataclass(frozen=True)
class Optimizer:
    kind: str
    lr: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    conditioning_weight_decay: float
    max_grad_norm: float
    muon_lr: float
    muon_weight_decay: float
    muon_momentum: float
    schedule: str
    warmup_steps: int
    minimum_ratio: float

    def __post_init__(self):
        require(self.kind in {"adamw", "muon_adamw"}, "unsupported optimizer")
        require(
            min(self.lr, self.epsilon, self.max_grad_norm) > 0
            and (self.muon_lr > 0 if self.kind == "muon_adamw" else self.muon_lr >= 0),
            "optimizer rates/epsilon/clip must be positive",
        )
        require(
            min(
                self.weight_decay,
                self.conditioning_weight_decay,
                self.muon_weight_decay,
            )
            >= 0,
            "negative decay",
        )
        require(
            all(0 <= b < 1 for b in self.betas) and 0 <= self.muon_momentum < 1,
            "invalid momentum",
        )
        require(
            self.schedule in {"constant", "cosine"}
            and self.warmup_steps >= 0
            and 0 <= self.minimum_ratio <= 1,
            "invalid LR schedule",
        )


@dataclass(frozen=True)
class TrainingExecution:
    learner_devices: tuple[int, ...]
    microbatch: int
    cpu_threads: int
    collective_timeout: int
    startup_timeout: int
    round_timeout: int
    precision: str
    activation_checkpointing: bool
    intra_host_roce: bool
    deterministic_algorithms: bool


@dataclass(frozen=True)
class StudentEMA:
    name: str
    decay: float
    device: str

    def __post_init__(self):
        name(self.name)
        require(self.name != "live", "live is reserved for the current student")
        require(0 <= self.decay < 1, "invalid evaluation EMA decay")
        require(self.device in {"learner", "cpu"}, "invalid EMA storage device")


@dataclass(frozen=True)
class Training:
    seed: int
    rounds: int
    checkpoint_every: int
    ema: tuple[StudentEMA, ...]


@dataclass(frozen=True)
class RewardChannel:
    channel: str
    direction: str
    weight: float
    standardization: str


@dataclass(frozen=True)
class Rewards:
    channels: tuple[RewardChannel, ...]
    epsilon: float


@dataclass(frozen=True)
class TrainingConfig:
    initialization: Checkpoint
    student: Student
    collection: Collection
    art: ART
    student_optimizer: Optimizer
    execution: TrainingExecution
    training: Training
    scorers: tuple[Scorer, ...]
    rewards: Rewards

    @property
    def updates_per_round(self):
        return ART_UPDATES_PER_ROUND

    @property
    def reference_coefficient(self):
        return self.art.reference_coefficient

    def __post_init__(self):
        s, c, e, t = (
            self.student,
            self.collection,
            self.execution,
            self.training,
        )
        require(
            self.initialization.weights in {"live", "ema"},
            "invalid initialization weight view",
        )
        require(
            s.mode in {"full", "lora"}
            and s.rank > 0
            and s.alpha > 0
            and s.targets in {"attention", "attention_mlp"},
            "invalid student mode/targets",
        )
        require(
            c.steps > 0 and c.shift in {"shape", "none"},
            "invalid collection sampler",
        )
        require(
            min(c.prompts_per_round, c.draws_per_prompt, c.text_exit_layer) > 0
            and all(n > 0 and n % 16 == 0 for n in (c.height, c.width)),
            "invalid collection sizes",
        )
        require(c.draws_per_prompt >= 2, "at least two draws per prompt are required")
        require(
            bool(e.learner_devices)
            and min(e.learner_devices) >= 0
            and len(set(e.learner_devices)) == len(e.learner_devices),
            "invalid learner role map",
        )
        require(
            min(
                e.microbatch,
                e.cpu_threads,
                e.collective_timeout,
                e.startup_timeout,
                e.round_timeout,
            )
            > 0
            and e.precision in {"float32", "bfloat16"},
            "invalid training execution",
        )
        require(
            c.prompts_per_round
            * c.draws_per_prompt
            % (len(e.learner_devices) * e.microbatch)
            == 0,
            "round must divide evenly across learners and microbatches",
        )
        require(
            0 <= t.seed < 2**63
            and t.rounds > 0
            and t.checkpoint_every > 0
            and len({ema.name for ema in t.ema}) == len(t.ema),
            "invalid training length/seed/EMA",
        )
        require(
            self.student_optimizer.warmup_steps <= t.rounds * self.updates_per_round,
            "student warmup exceeds the training horizon",
        )
        require(
            c.prompts_per_round * c.draws_per_prompt % (2 * len(e.learner_devices))
            == 0,
            "ART round must split evenly into two update windows",
        )
        require(
            math.isfinite(self.art.reference_coefficient)
            and self.art.reference_coefficient >= 0,
            "invalid ART reference coefficient",
        )
        available = {f"{s.id}.{ch.name}": ch for s in self.scorers for ch in s.outputs}
        channels = self.rewards.channels
        require(
            bool(self.scorers)
            and len({s.id for s in self.scorers}) == len(self.scorers)
            and len({ch.channel for ch in channels}) == len(channels),
            "unique nonempty scorers/reward channels required",
        )
        require(
            bool(channels)
            and math.isclose(sum(ch.weight for ch in channels), 1, abs_tol=1e-7),
            "reward weights must sum to one",
        )
        require(
            {ch.channel.split(".")[0] for ch in channels}
            == {s.id for s in self.scorers},
            "reward/scorer service set differs",
        )
        for channel in channels:
            require(channel.channel in available, "reward channel is undeclared")
            require(
                channel.direction == available[channel.channel].direction
                and channel.weight >= 0
                and channel.standardization in {"none", "round"},
                "invalid reward orientation/weight/standardization",
            )
        require(
            self.rewards.epsilon > 0, "reward normalization epsilon must be positive"
        )


def load_training_config(path):
    if Path(path).suffix == ".json":
        return decode(TrainingConfig, json.loads(Path(path).read_text()))
    with Path(path).open("rb") as stream:
        return decode(TrainingConfig, tomllib.load(stream))
