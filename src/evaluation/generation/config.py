"""Sampling and worker settings shared by standalone generation and monitors."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.posttrain.schema import decode, name, require


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
class ExternalModel:
    name: str
    pipeline: str
    asset: str
    precision: str
    batch_size: int
    shapes: dict[str, tuple[int, int]]
    parameters: dict[str, object]


@dataclass(frozen=True)
class Sampling:
    precision: str
    batch_size: int
    text_exit_layer: int
    shapes: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class Execution:
    max_inflight_batches: int
    startup_timeout: int
    task_timeout: int
    cpu_threads: int


@dataclass(frozen=True)
class Config:
    sampling: Sampling
    execution: Execution
    models: tuple[Arm | ExternalModel, ...]

    def __post_init__(self):
        validate_generation(self.sampling, self.execution, self.models)


def validate_generation(s, e, arms):
    require(s.precision in {"bfloat16", "float32"}, "unsupported sampling precision")
    require(
        s.batch_size > 0 and s.text_exit_layer > 0 and bool(s.shapes),
        "invalid sampling dimensions",
    )
    require(
        all(
            len(shape) == 2 and all(n > 0 and n % 16 == 0 for n in shape)
            for shape in s.shapes.values()
        ),
        "image dimensions must be multiples of 16",
    )
    require(
        min(e.max_inflight_batches, e.startup_timeout, e.task_timeout, e.cpu_threads)
        > 0,
        "execution limits must be positive",
    )
    require(
        bool(arms) and len({a.name for a in arms}) == len(arms),
        "unique nonempty models required",
    )
    for arm in arms:
        name(arm.name)
        if isinstance(arm, ExternalModel):
            name(arm.asset)
            require(
                arm.pipeline
                in {
                    "DreamLitePipeline",
                    "SanaPipeline",
                    "PixArtSigmaPipeline",
                    "StableDiffusion3Pipeline",
                    "MOSSImagePipeline",
                },
                "unsupported external pipeline",
            )
            require(
                arm.precision in {"float16", "bfloat16", "float32"}
                and arm.batch_size > 0
                and bool(arm.shapes),
                "invalid external precision/batch/shapes",
            )
            require(
                all(
                    len(shape) == 2 and all(n > 0 and n % 16 == 0 for n in shape)
                    for shape in arm.shapes.values()
                ),
                "external image dimensions must be multiples of 16",
            )
            require(
                type(arm.parameters.get("num_inference_steps")) is int
                and arm.parameters["num_inference_steps"] > 0,
                "external pipeline requires positive num_inference_steps",
            )
            continue
        name(arm.checkpoint.asset)
        name(arm.checkpoint.weights)
        require(
            arm.steps > 0 and arm.guidance >= 0 and arm.shift in {"shape", "none"},
            "invalid sampler/guidance/shift",
        )


def load_config(path):
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    raw.pop("benchmarks", None)
    raw.pop("report", None)
    models = tuple(
        decode(ExternalModel if "pipeline" in value else Arm, {"name": key, **value})
        for key, value in raw.pop("models").items()
    )
    if raw.keys() != {"sampling", "execution"}:
        raise ValueError("generation requires sampling, execution and models sections")
    return Config(
        decode(Sampling, raw["sampling"]), decode(Execution, raw["execution"]), models
    )
