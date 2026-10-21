"""Complete single-stage ART training configuration, separate from benchmark arms."""

import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from .schema import decode, name, require
from .scoring.config import ClientConfig
from .scoring.contracts import CHANNELS

ART_UPDATES_PER_ROUND = 2
REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class IntervalEvaluation:
    every: int
    suite: str
    monitor_suite: str
    prompt_limit: int
    draws_per_prompt: int
    steps: int
    guidance: float
    shift: str
    shapes: dict[str, tuple[int, int]]
    channels: tuple[str, ...]
    diversity_asset: str
    rubric_prompt_limit: int = 32
    suites: tuple[str, ...] = ("development", "monitor", "rubric")
    weights: tuple[str, ...] = ("ema",)
    bootstrap_seed: int = 7
    bootstrap_repeats: int = 2000
    confidence: float = 0.95

    def __post_init__(self):
        require(
            self.every > 0
            and self.prompt_limit >= 0
            and self.draws_per_prompt > 0
            and self.rubric_prompt_limit > 0,
            "invalid evaluation cadence/prompt selection",
        )
        require(
            bool(self.suites)
            and len(set(self.suites)) == len(self.suites)
            and set(self.suites) <= {"development", "monitor", "rubric"},
            "invalid evaluation suites",
        )
        require(
            bool(self.weights)
            and len(set(self.weights)) == len(self.weights)
            and set(self.weights) <= {"live", "ema"},
            "evaluation weights must be live or ema",
        )
        require(0 <= self.bootstrap_seed < 2**63, "invalid evaluation seed")


@dataclass(frozen=True)
class Initialization:
    asset: str


@dataclass(frozen=True)
class Student:
    mode: str
    rank: int
    alpha: float
    targets: str


@dataclass(frozen=True)
class Collection:
    dataset: str
    prompts_per_round: int
    draws_per_prompt: int
    shapes: dict[str, tuple[int, int]]
    steps: int
    shift: str
    text_exit_layer: int
    split: str = "train"


@dataclass(frozen=True)
class ART:
    reference_coefficient: float


@dataclass(frozen=True)
class Optimizer:
    kind: str
    lr: float
    muon_lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    conditioning_weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    muon_weight_decay: float = 0.0
    muon_momentum: float = 0.95

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


@dataclass(frozen=True)
class TrainingExecution:
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
class Training:
    seed: int
    rounds: int
    checkpoint_every: int
    checkpoint_keep_last: int
    ema_decay: float


@dataclass(frozen=True)
class RewardChannel:
    channel: str
    weight: float
    standardization: str


@dataclass(frozen=True)
class Rewards:
    channels: tuple[RewardChannel, ...]
    epsilon: float = 1e-4


@dataclass(frozen=True)
class TrainingConfig:
    initialization: Initialization
    student: Student
    collection: Collection
    art: ART
    student_optimizer: Optimizer
    execution: TrainingExecution
    training: Training
    scoring: ClientConfig
    rewards: Rewards
    evaluation: IntervalEvaluation

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
        name(self.initialization.asset)
        name(c.split)
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
            and bool(c.dataset)
            and bool(c.shapes)
            and all(
                n > 0 and n % 16 == 0 for shape in c.shapes.values() for n in shape
            ),
            "invalid collection sizes",
        )
        require(c.draws_per_prompt >= 2, "at least two draws per prompt are required")
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
            0 <= t.seed < 2**63
            and t.rounds > 0
            and t.checkpoint_every > 0
            and t.checkpoint_keep_last >= 0
            and 0 <= t.ema_decay < 1,
            "invalid training length/seed/EMA",
        )
        require(
            math.isfinite(self.art.reference_coefficient)
            and self.art.reference_coefficient >= 0,
            "invalid ART reference coefficient",
        )
        channels = self.rewards.channels
        require(
            len({ch.channel for ch in channels}) == len(channels),
            "duplicate reward channel",
        )
        if any(ch.channel.startswith("rubric.") for ch in channels):
            require(c.draws_per_prompt % 2 == 0, "rubric rewards require paired draws")
        require(
            bool(channels)
            and math.isclose(sum(ch.weight for ch in channels), 1, abs_tol=1e-7),
            "reward weights must sum to one",
        )
        for channel in channels:
            require(channel.channel in CHANNELS, "reward channel is undeclared")
            require(
                channel.weight >= 0 and channel.standardization in {"none", "round"},
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


def training_contract(config, prompts, world_size, scoring):
    require(
        world_size > 0
        and config.collection.prompts_per_round
        * config.collection.draws_per_prompt
        % (2 * world_size)
        == 0,
        "ART round must give every learner an even number of draws",
    )
    return json.loads(
        json.dumps(
            {
                "config": asdict(config),
                "prompts": prompts,
                "world_size": world_size,
                "scoring": scoring,
            }
        )
    )
