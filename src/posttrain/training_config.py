"""Complete cold-stage training configuration, separate from benchmark arms."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import Checkpoint, Scorer
from .schema import decode, digest, require


@dataclass(frozen=True)
class Student:
    mode: str
    rank: int
    alpha: float
    targets: str


@dataclass(frozen=True)
class Fake:
    strategy: str
    rank: int
    multiplier: float
    updates: int
    target: str


@dataclass(frozen=True)
class Objective:
    mode: str
    estimator: str
    epsilon: float
    time_min: float
    time_max: float
    teacher_guidance: float


@dataclass(frozen=True)
class Collection:
    prompts: str
    prompts_per_round: int
    draws_per_prompt: int
    height: int
    width: int
    rule: str
    steps: int
    guidance: float
    shift: str
    selected_pmf: tuple[float, ...]
    text_exit_layer: int


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
            min(self.lr, self.muon_lr, self.epsilon, self.max_grad_norm) > 0,
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
class Training:
    seed: int
    steps: int
    checkpoint_every: int
    ema_enabled: bool
    ema_decay: float


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
    weighting: str
    temperature: float


@dataclass(frozen=True)
class DynaDG:
    enabled: bool
    horizon: int
    start_scale: float


@dataclass(frozen=True)
class DynaRS:
    enabled: bool
    horizon: int
    alpha_start: float
    beta_start: float


@dataclass(frozen=True)
class TrainingConfig:
    initialization: Checkpoint
    teacher: Checkpoint
    student: Student
    fake: Fake
    objective: Objective
    collection: Collection
    student_optimizer: Optimizer
    fake_optimizer: Optimizer
    execution: TrainingExecution
    training: Training
    scorers: tuple[Scorer, ...]
    rewards: Rewards
    dyna_dg: DynaDG
    dyna_rs: DynaRS

    def __post_init__(self):
        s, f, o, c, e, t = (
            self.student,
            self.fake,
            self.objective,
            self.collection,
            self.execution,
            self.training,
        )
        for checkpoint in (self.initialization, self.teacher):
            digest(checkpoint.sha256)
            digest(checkpoint.architecture_sha256)
            require(
                checkpoint.weights in {"live", "ema"},
                "invalid initialization weight view",
            )
        require(
            self.initialization.architecture_sha256 == self.teacher.architecture_sha256,
            "teacher/student architectures differ",
        )
        require(
            s.mode in {"full", "lora"}
            and s.rank > 0
            and s.alpha > 0
            and s.targets in {"attention", "attention_mlp"},
            "invalid student mode/targets",
        )
        require(
            f.strategy in {"full", "teacher_qv"}
            and f.rank > 0
            and f.multiplier > 0
            and f.updates > 0
            and f.target in {"selected_prediction", "terminal"},
            "invalid fake role",
        )
        require(
            s.mode == "full" or self.student_optimizer.kind == "adamw",
            "student adapters require AdamW",
        )
        require(
            f.strategy == "full" or self.fake_optimizer.kind == "adamw",
            "fake adapters require AdamW",
        )
        require(
            o.mode in {"dmd", "rtdm"}
            and o.estimator in {"teacher_residual", "clean_difference", "score_kl"},
            "only cold DMD/RT-DM is implemented",
        )
        require(
            o.epsilon > 0
            and 0 < o.time_min < o.time_max < 1
            and o.teacher_guidance >= 0,
            "invalid DMD bounds/guidance",
        )
        require(
            c.rule in {"euler", "renoise"}
            and c.steps > 0
            and c.shift in {"shape", "none"}
            and c.guidance >= 0,
            "invalid collection sampler",
        )
        require(
            min(c.prompts_per_round, c.draws_per_prompt, c.text_exit_layer) > 0
            and all(n > 0 and n % 16 == 0 for n in (c.height, c.width)),
            "invalid collection sizes",
        )
        require(
            len(c.selected_pmf) == c.steps
            and min(c.selected_pmf) >= 0
            and math.isclose(sum(c.selected_pmf), 1, abs_tol=1e-7),
            "selected PMF must sum to one across all generator calls",
        )
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
            and t.steps > 0
            and t.checkpoint_every > 0
            and 0 <= t.ema_decay < 1,
            "invalid training length/seed/EMA",
        )
        require(
            self.student_optimizer.warmup_steps <= t.steps
            and self.fake_optimizer.warmup_steps <= t.steps * f.updates,
            "warmup exceeds role clock horizon",
        )
        dg, rs, reward = self.dyna_dg, self.dyna_rs, self.rewards
        require(dg.horizon > 0 and dg.start_scale >= 0, "invalid DynaDG schedule")
        require(
            rs.horizon > 0 and rs.alpha_start > rs.beta_start > 0,
            "DynaRS requires positive alpha_start > beta_start",
        )
        require(
            (not dg.enabled or dg.horizon <= t.steps)
            and (not rs.enabled or rs.horizon <= t.steps),
            "dynamic horizon exceeds cold generator steps",
        )
        require(
            not dg.enabled or f.strategy == "teacher_qv",
            "DynaDG requires teacher_qv fake",
        )
        require(
            reward.epsilon > 0
            and reward.temperature > 0
            and reward.weighting in {"mean_one_exp", "paper_max_exp"},
            "invalid RT-DM reward transform",
        )
        require(
            len({s.id for s in self.scorers}) == len(self.scorers), "duplicate scorer"
        )
        available = {f"{s.id}.{c.name}": c for s in self.scorers for c in s.outputs}
        channels = reward.channels
        require(
            len({c.channel for c in channels}) == len(channels),
            "duplicate reward channel",
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
                reward.weighting != "paper_max_exp"
                or channel.standardization == "none",
                "paper-max requires raw nonnegative rewards, not standardized channels",
            )
        if o.mode == "rtdm":
            require(
                bool(channels)
                and math.isclose(sum(c.weight for c in channels), 1, abs_tol=1e-7),
                "reward weights must sum to one",
            )
            require(
                {c.channel.split(".")[0] for c in channels}
                == {s.id for s in self.scorers},
                "reward/scorer service set differs",
            )
        else:
            require(not channels and not self.scorers, "DMD has no reward acquisition")


def load_training_config(path):
    with Path(path).open("rb") as stream:
        return decode(TrainingConfig, tomllib.load(stream))
