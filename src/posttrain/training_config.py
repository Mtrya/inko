"""Complete cold/joint training configuration, separate from benchmark arms."""

import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import Checkpoint, Scorer
from .schema import decode, digest, name, require


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
    initial_steps: int
    initial_checkpoint_every: int
    target_refresh: str


@dataclass(frozen=True)
class FakeProbe:
    every: int
    draws: int


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
    steps: int
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
class Stages:
    cold_steps: int
    joint_steps: int
    initial_checkpoint: str
    initial_sha256: str


@dataclass(frozen=True)
class Joint:
    dmd_coefficient: float
    nft_coefficient: float
    reference_coefficient: float


@dataclass(frozen=True)
class NFT:
    parameterization: str
    advantage_scale: str
    advantage_clip: float
    beta: float
    epsilon: float
    updates_per_round: int
    time_draws: int
    time_distribution: str
    time_min: float
    time_max: float


@dataclass(frozen=True)
class OldPolicy:
    strategy: str
    decay: float
    decay_start: float
    decay_end: float
    horizon: int


@dataclass(frozen=True)
class TrainingConfig:
    initialization: Checkpoint
    teacher: Checkpoint
    student: Student
    fake: Fake
    fake_probe: FakeProbe
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
    stages: Stages
    joint: Joint
    nft: NFT
    old_policy: OldPolicy

    @property
    def joint_updates(self):
        return self.nft.updates_per_round if self.joint.nft_coefficient > 0 else 1

    @property
    def requires_scores(self):
        return (self.stages.cold_steps > 0 and self.objective.mode == "rtdm") or (
            self.stages.joint_steps > 0 and self.joint.nft_coefficient > 0
        )

    def dmd_enabled(self, stage):
        return stage == "cold" or self.joint.dmd_coefficient > 0

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
            and f.target in {"selected_prediction", "terminal"}
            and f.initial_steps >= 0
            and f.initial_checkpoint_every > 0
            and f.target_refresh in {"round", "update"},
            "invalid fake role",
        )
        require(
            self.fake_probe.every >= 0 and self.fake_probe.draws > 0,
            "invalid fresh-sample fake probe cadence/draws",
        )
        require(
            o.mode in {"dmd", "rtdm"}
            and o.estimator in {"teacher_residual", "clean_difference", "score_kl"},
            "cold objective must be DMD or RT-DM",
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
            and len({ema.name for ema in t.ema}) == len(t.ema),
            "invalid training length/seed/EMA",
        )
        stages, joint, nft, old = self.stages, self.joint, self.nft, self.old_policy
        require(
            stages.cold_steps > 0 or f.initial_steps == 0,
            "initial fake fitting belongs to a cold-stage run",
        )
        require(
            not self.fake_probe.every
            or stages.cold_steps > 0
            or joint.dmd_coefficient > 0,
            "fake probe requires an active DMD stage",
        )
        require(
            min(stages.cold_steps, stages.joint_steps) >= 0
            and stages.cold_steps + stages.joint_steps == t.steps,
            "stage lengths must sum to training steps",
        )
        if stages.cold_steps:
            require(
                not stages.initial_checkpoint and not stages.initial_sha256,
                "cold stage starts from the declared initial model",
            )
        else:
            require(
                bool(stages.initial_checkpoint), "joint entry needs a cold checkpoint"
            )
            digest(stages.initial_sha256)
        coefficients = (
            joint.dmd_coefficient,
            joint.nft_coefficient,
            joint.reference_coefficient,
        )
        require(
            all(math.isfinite(v) and v >= 0 for v in coefficients),
            "invalid joint coefficients",
        )
        require(
            (
                (joint.dmd_coefficient + joint.nft_coefficient > 0)
                if stages.joint_steps
                else not any(coefficients)
            ),
            "joint stage requires a main objective; absent stage has zero coefficients",
        )
        require(
            joint.reference_coefficient == 0 or joint.nft_coefficient > 0,
            "reference prediction penalty requires NFT forward inputs",
        )
        require(
            nft.parameterization in {"clean_adaptive", "velocity_mse"}
            and nft.advantage_scale in {"group", "round"}
            and min(
                nft.advantage_clip,
                nft.beta,
                nft.epsilon,
                nft.updates_per_round,
                nft.time_draws,
            )
            > 0
            and all(
                math.isfinite(v) for v in (nft.advantage_clip, nft.beta, nft.epsilon)
            )
            and nft.time_distribution in {"uniform", "grid"}
            and 0 < nft.time_min < nft.time_max < 1,
            "invalid NFT loss/time/group configuration",
        )
        require(
            old.strategy in {"copy", "constant", "linear_ramp"}
            and 0 <= old.decay < 1
            and 0 <= old.decay_start <= old.decay_end < 1
            and old.horizon > 0,
            "invalid old-policy refresh schedule",
        )
        if stages.joint_steps and joint.nft_coefficient > 0:
            require(
                c.draws_per_prompt >= 2, "NFT requires at least two draws per prompt"
            )
            require(
                stages.joint_steps % self.joint_updates == 0
                and t.checkpoint_every % self.joint_updates == 0,
                "NFT joint length/checkpoint interval must contain complete rounds",
            )
        for stage, count in (
            ("cold", stages.cold_steps),
            ("joint", stages.joint_steps),
        ):
            if count:
                require(
                    self.student_optimizer.warmup_steps <= count,
                    "student warmup exceeds stage horizon",
                )
                if self.dmd_enabled(stage):
                    require(
                        self.fake_optimizer.warmup_steps <= count * f.updates,
                        "fake warmup exceeds stage horizon",
                    )
        dg, rs, reward = self.dyna_dg, self.dyna_rs, self.rewards
        require(dg.horizon > 0 and dg.start_scale >= 0, "invalid DynaDG schedule")
        require(
            rs.horizon > 0 and rs.alpha_start > rs.beta_start > 0,
            "DynaRS requires positive alpha_start > beta_start",
        )
        require(
            (not dg.enabled or dg.horizon <= stages.cold_steps)
            and (not rs.enabled or rs.horizon <= stages.cold_steps),
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
                not (stages.cold_steps and o.mode == "rtdm")
                or reward.weighting != "paper_max_exp"
                or channel.standardization == "none",
                "paper-max requires raw nonnegative rewards, not standardized channels",
            )
        if self.requires_scores:
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
            require(
                not channels and not self.scorers,
                "unscored run has no reward services/channels",
            )


def load_training_config(path):
    if Path(path).suffix == ".json":
        return decode(TrainingConfig, json.loads(Path(path).read_text()))
    with Path(path).open("rb") as stream:
        return decode(TrainingConfig, tomllib.load(stream))
