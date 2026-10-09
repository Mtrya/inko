"""Scorer-owned source, model and output contracts."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..schema import decode, require
from .contracts import Capabilities, Channel


@dataclass(frozen=True)
class Source:
    repository: str


@dataclass(frozen=True)
class Model:
    repository: str
    files: tuple[str, ...]

    def __post_init__(self):
        require(bool(self.files), "model file patterns required")


@dataclass(frozen=True)
class HPSOptions:
    backbone: Model


@dataclass(frozen=True)
class VQAOptions:
    question_template: str
    answer_template: str
    max_new_tokens: int
    temperature: float

    def __post_init__(self):
        require(
            self.question_template.count("{}") == 1
            and self.answer_template == "Yes"
            and self.max_new_tokens == 1
            and self.temperature > 0,
            "invalid VQA scoring contract",
        )


@dataclass(frozen=True)
class Semantics:
    backend: str
    source: Source
    weights: Model
    outputs: tuple[Channel, ...]

    def __post_init__(self):
        require(self.backend in {"hpsv3", "vqascore"}, "unknown backend")


@dataclass(frozen=True)
class Limits:
    max_batch_size: int
    queue_capacity: int
    request_timeout: float
    max_body_bytes: int
    cpu_threads: int

    def __post_init__(self):
        require(
            min(
                self.max_batch_size,
                self.queue_capacity,
                self.request_timeout,
                self.max_body_bytes,
                self.cpu_threads,
            )
            > 0,
            "service limits must be positive",
        )


@dataclass(frozen=True)
class ServiceConfig:
    semantics: Semantics
    options: HPSOptions | VQAOptions
    limits: Limits

    @property
    def capabilities(self):
        return Capabilities(
            self.semantics.outputs,
            self.limits.max_batch_size,
            ("shared_file", "png_base64"),
        )


def load_service_config(path):
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    require(set(raw) == {"semantics", "options", "limits"}, "invalid service sections")
    semantics = decode(Semantics, raw["semantics"])
    options = decode(
        HPSOptions if semantics.backend == "hpsv3" else VQAOptions, raw["options"]
    )
    result = ServiceConfig(semantics, options, decode(Limits, raw["limits"]))
    expected = {"mu", "log_sigma"} if semantics.backend == "hpsv3" else {"probability"}
    require(
        {c.name for c in semantics.outputs} == expected,
        "backend output contract differs",
    )
    return result
